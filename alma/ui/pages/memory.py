import datetime as dt
from collections import Counter
from typing import Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html, MATCH, ALL

from alma.engine import storage

LABEL_OPTIONS = ["ALL", "DEEP_WORK", "IDEATION", "RECOVERY", "ENGAGEMENT", "INSUFFICIENT_SIGNAL"]
LABEL_NOTES = {
    "DEEP_WORK": "High activation + steady richness",
    "IDEATION": "High variability + rising richness",
    "RECOVERY": "Low activation + settling",
    "ENGAGEMENT": "Balanced activation + richness",
    "INSUFFICIENT_SIGNAL": "Not enough clean signal",
}


def _date_str(offset_days: int = 0) -> str:
    return (dt.date.today() + dt.timedelta(days=offset_days)).isoformat()


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Memory"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Start date"),
                                        dcc.DatePickerSingle(
                                            id="mem-start",
                                            date=_date_str(0),
                                            display_format="YYYY-MM-DD",
                                        ),
                                    ],
                                    md=3,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("End date"),
                                        dcc.DatePickerSingle(
                                            id="mem-end",
                                            date=_date_str(0),
                                            display_format="YYYY-MM-DD",
                                        ),
                                    ],
                                    md=3,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Label filter"),
                                        dcc.Dropdown(
                                            id="mem-label-filter",
                                            options=[{"label": l, "value": l} for l in LABEL_OPTIONS],
                                            value="ALL",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                    sm=12,
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Events", className="fw-bold mb-2"),
                                        html.Div(id="mem-events-list"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Buckets", className="fw-bold mb-2"),
                                        html.Div(id="mem-buckets-list"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Similar States", className="fw-bold mb-2"),
                                        html.Div(id="mem-similar-list"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                            ],
                            className="g-3",
                        ),
                        dcc.Store(id="mem-buckets-store"),
                        dcc.Store(id="mem-events-store"),
                        dcc.Store(id="mem-selected-bucket"),
                    ]
                ),
            ],
            className="page-card",
        )
    ],
    fluid=True,
    className="page-container",
)


def _ts_bounds(start_date: str, end_date: str) -> Tuple[float, float]:
    sd = dt.datetime.fromisoformat(start_date)
    ed = dt.datetime.fromisoformat(end_date)
    ed = ed + dt.timedelta(days=1)
    return sd.timestamp(), ed.timestamp()


def _render_events(events: List[Dict[str, object]]) -> List[html.Div]:
    rows = []
    for idx, e in enumerate(events):
        ts = e.get("ts")
        label = e.get("label") or ""
        note = e.get("note") or ""
        ts_txt = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
        rows.append(
            dbc.Button(
                f"{ts_txt} — {label}" + (f" — {note}" if note else ""),
                color="secondary",
                outline=True,
                size="sm",
                className="mb-1 w-100 text-start",
                id={"type": "mem-ev-btn", "index": idx},
            )
        )
    return rows or [html.Div("No events")]


def _render_buckets(buckets: List[Dict[str, object]]) -> List[html.Div]:
    rows = []
    for idx, b in enumerate(buckets):
        start = b.get("bucket_start_ts")
        end = b.get("bucket_end_ts")
        label = b.get("label") or ""
        ts_txt = dt.datetime.fromtimestamp(start).strftime("%H:%M") if start else ""
        rows.append(
            dbc.Button(
                f"{ts_txt} — {label} (Q={b.get('mean_Q', 0):.3f}, X={b.get('mean_X', 0):.3f})",
                color="secondary",
                outline=True,
                size="sm",
                className="mb-1 w-100 text-start",
                id={"type": "mem-bucket-btn", "index": idx},
            )
        )
    return rows or [html.Div("No buckets")]


def _label_note(label: str) -> str:
    return LABEL_NOTES.get(label, LABEL_NOTES["ENGAGEMENT"])


def _bucket_vector(b: Dict[str, object]) -> Optional[np.ndarray]:
    try:
        return np.array(
            [
                float(b.get("mean_X", 0.0)),
                float(b.get("mean_Q", 0.0)),
                float(b.get("std_Q", 0.0)),
                float(b.get("Q_slope", 0.0)),
            ],
            dtype=float,
        )
    except Exception:
        return None


def _cosine_sim(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    v_norm = np.linalg.norm(v)
    M_norm = np.linalg.norm(M, axis=1)
    denom = (v_norm * M_norm) + 1e-9
    return (M @ v) / denom


def _top_tracks(ts0: float, ts1: float, session_id: Optional[str]) -> List[str]:
    try:
        rows = storage.get_spotify_between(ts0, ts1, session_id=session_id)
    except Exception:
        return []
    if not rows:
        return []
    counts = Counter()
    names = {}
    for r in rows:
        tid = r.get("track_id") or ""
        key = tid or (r.get("track_name") or "")
        if not key:
            continue
        counts[key] += 1
        names[key] = f"{r.get('artists') or ''} — {r.get('track_name') or ''}"
    top = counts.most_common(3)
    return [names[k] for k, _ in top if k in names]


@callback(
    Output("mem-buckets-store", "data"),
    Output("mem-events-store", "data"),
    Output("mem-buckets-list", "children"),
    Output("mem-events-list", "children"),
    Input("mem-start", "date"),
    Input("mem-end", "date"),
    Input("mem-label-filter", "value"),
)
def load_memory(start_date, end_date, label_filter):
    start_date = start_date or _date_str(0)
    end_date = end_date or _date_str(0)
    ts0, ts1 = _ts_bounds(start_date, end_date)
    buckets = storage.get_buckets_between(ts0, ts1, session_id=None)
    events = storage.get_events_between(ts0, ts1, session_id=None)
    if label_filter and label_filter != "ALL":
        buckets = [b for b in buckets if (b.get("label") == label_filter)]
    bucket_list = _render_buckets(buckets)
    event_list = _render_events(events)
    return buckets, events, bucket_list, event_list


@callback(
    Output("mem-selected-bucket", "data"),
    Input({"type": "mem-bucket-btn", "index": ALL}, "n_clicks"),
    State("mem-buckets-store", "data"),
    prevent_initial_call=True,
)
def select_bucket(btns, buckets):
    if not buckets or not btns:
        return None
    for i, n in enumerate(btns):
        if n:
            return buckets[i]
    return None


@callback(
    Output("mem-similar-list", "children"),
    Input("mem-selected-bucket", "data"),
    State("mem-buckets-store", "data"),
    State("mem-label-filter", "value"),
)
def show_similar(selected_bucket, buckets, label_filter):
    if not selected_bucket or not buckets:
        return "Select a bucket to see similar states."

    # Filter candidate set
    cands = buckets
    if label_filter and label_filter != "ALL":
        cands = [b for b in buckets if b.get("label") == label_filter]
    # Build matrix
    vecs = []
    keep = []
    for b in cands:
        v = _bucket_vector(b)
        if v is not None:
            vecs.append(v)
            keep.append(b)
    if not vecs:
        return "No comparable buckets."
    M = np.vstack(vecs)
    # z-score per column
    mean = np.mean(M, axis=0)
    std = np.std(M, axis=0)
    std[std == 0] = 1.0
    Mz = (M - mean) / std
    target = _bucket_vector(selected_bucket)
    if target is None:
        return "Invalid bucket vector."
    vz = (target - mean) / std
    sims = _cosine_sim(vz, Mz)
    # Apply slight penalty to insufficient signal
    for i, b in enumerate(keep):
        if b.get("label") == "INSUFFICIENT_SIGNAL":
            sims[i] *= 0.8
    order = np.argsort(sims)[::-1][:10]
    items = []
    for idx in order:
        b = keep[idx]
        score = sims[idx]
        start = b.get("bucket_start_ts")
        end = b.get("bucket_end_ts")
        label = b.get("label") or ""
        note = _label_note(label)
        tracks = _top_tracks(start or 0, end or 0, session_id=b.get("session_id")) if start and end else []
        tracks_txt = ", ".join(tracks) if tracks else "—"
        ts_txt = f"{dt.datetime.fromtimestamp(start).strftime('%H:%M')}–{dt.datetime.fromtimestamp(end).strftime('%H:%M')}" if start and end else ""
        items.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(f"{ts_txt} | {label}", className="fw-bold"),
                        html.Div(f"Similarity: {score:.3f}", className="text-muted"),
                        html.Div(note),
                        html.Div(f"Tracks during window: {tracks_txt}", className="mt-1 small"),
                    ]
                ),
                className="mb-2",
            )
        )
    return items or "No similar buckets found."
