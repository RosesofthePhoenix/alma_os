import datetime as dt
from collections import Counter
from typing import Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html, MATCH, ALL, ctx

from alma.engine import storage

LABEL_OPTIONS = ["ALL", "DEEP_WORK", "IDEATION", "RECOVERY", "ENGAGEMENT", "INSUFFICIENT_SIGNAL", "TRANSCENDENT"]
LABEL_NOTES = {
    "DEEP_WORK": "High activation + steady richness",
    "IDEATION": "High variability + rising richness",
    "RECOVERY": "Low activation + settling",
    "ENGAGEMENT": "Balanced activation + richness",
    "INSUFFICIENT_SIGNAL": "Not enough clean signal",
    "TRANSCENDENT": "HCE-driven harmony; prioritize synthesis/insight",
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
                                dbc.Col(
                                    [
                                        dbc.Label("Context filter"),
                                        dcc.Dropdown(
                                            id="mem-context-filter",
                                            options=[
                                                {"label": "Any", "value": "ANY"},
                                                {"label": "Alone", "value": "alone"},
                                                {"label": "With others", "value": "with"},
                                                {"label": "Work", "value": "work"},
                                                {"label": "Creative", "value": "creative"},
                                                {"label": "Meditation", "value": "meditation"},
                                                {"label": "Social", "value": "social"},
                                                {"label": "Media", "value": "media"},
                                                {"label": "Exercise", "value": "exercise"},
                                            ],
                                            value="ANY",
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
                                        dbc.Label("Min mean_HCE (scaled)"),
                                        dbc.Input(id="mem-min-hce", type="number", value=0.0, step=0.1, min=0),
                                    ],
                                    md=3,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button("Retrieve similar states", id="mem-sim-btn", color="primary", className="mt-4"),
                                        dbc.Button("Use latest bucket", id="mem-use-latest", color="secondary", outline=True, size="sm", className="ms-2 mt-4"),
                                    ],
                                    md=6,
                                    sm=12,
                                ),
                            ],
                            className="g-2 mb-3",
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
                        html.Hr(),
                        html.Div("Quick Captures", className="fw-bold mb-2"),
                        html.Div(id="mem-quick-list"),
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


def _render_quick_captures(events: List[Dict[str, object]]) -> List[html.Div]:
    rows = []
    for e in events:
        tags = e.get("tags_json") or {}
        if (tags.get("kind") or e.get("kind")) != "quick_capture":
            continue
        ctx_json = e.get("context_json") or {}
        ts = e.get("ts")
        ts_txt = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
        mean_hce = ctx_json.get("mean_HCE", 0.0)
        mean_q = ctx_json.get("mean_Q", 0.0)
        mean_x = ctx_json.get("mean_X", 0.0)
        window_min = ctx_json.get("window_min")
        media = ctx_json.get("media") or ""
        track = ctx_json.get("track") or ""
        note = e.get("note") or ""
        rows.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(f"{ts_txt} — {note}", className="fw-bold"),
                        html.Div(f"Window: {window_min} min" if window_min else "Window: —", className="small"),
                        html.Div(f"HCE={mean_hce:.2f} | Q={mean_q:.3f} | X={mean_x:.3f}", className="small text-muted"),
                        html.Div(f"Media/Person: {media}" if media else "", className="small"),
                        html.Div(f"Track: {track}" if track else "", className="small text-muted"),
                    ]
                ),
                className="mb-2",
            )
        )
    return rows or [html.Div("No quick captures")]


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
                float(b.get("mean_HCE", 0.0)) * 2.0,  # double weight for transcendence
                float(b.get("std_Q", 0.0)),
                float(b.get("valid_fraction", 0.0)),
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


def _matches_context(ev: Dict[str, object], ctx_filter: str) -> bool:
    if ctx_filter == "ANY":
        return True
    tags = ev.get("tags_json") or {}
    ctx_val = (tags.get("social") or "").lower()
    activity = (tags.get("activity") or "").lower()
    if ctx_filter in {"alone", "with"}:
        return ctx_val == ctx_filter
    return activity == ctx_filter


@callback(
    Output("mem-buckets-store", "data"),
    Output("mem-events-store", "data"),
    Output("mem-buckets-list", "children"),
    Output("mem-events-list", "children"),
    Output("mem-quick-list", "children"),
    Input("mem-start", "date"),
    Input("mem-end", "date"),
    Input("mem-label-filter", "value"),
    Input("mem-context-filter", "value"),
    Input("qc-status", "children"),
)
def load_memory(start_date, end_date, label_filter, ctx_filter, _qc_status):
    start_date = start_date or _date_str(0)
    end_date = end_date or _date_str(0)
    ts0, ts1 = _ts_bounds(start_date, end_date)
    buckets = storage.get_buckets_between(ts0, ts1, session_id=None)
    events = storage.get_events_between(ts0, ts1, session_id=None)
    if label_filter and label_filter != "ALL":
        if label_filter == "TRANSCENDENT":
            buckets = [b for b in buckets if (b.get("mean_HCE") or 0) > 3.0]
        else:
            buckets = [b for b in buckets if (b.get("label") == label_filter)]
    if ctx_filter:
        events = [e for e in events if _matches_context(e, ctx_filter)]
    bucket_list = _render_buckets(buckets)
    event_list = _render_events(events)
    quick_list = _render_quick_captures(events)
    return buckets, events, bucket_list, event_list, quick_list


@callback(
    Output("mem-selected-bucket", "data"),
    Input({"type": "mem-bucket-btn", "index": ALL}, "n_clicks"),
    Input("mem-use-latest", "n_clicks"),
    State("mem-buckets-store", "data"),
    prevent_initial_call=True,
)
def select_bucket(btns, use_latest, buckets):
    if not buckets:
        return None
    if ctx.triggered_id == "mem-use-latest":
        return buckets[-1]
    if not btns:
        return None
    for i, n in enumerate(btns):
        if n:
            return buckets[i]
    return None


@callback(
    Output("mem-similar-list", "children"),
    Input("mem-sim-btn", "n_clicks"),
    State("mem-selected-bucket", "data"),
    State("mem-buckets-store", "data"),
    State("mem-label-filter", "value"),
    State("mem-min-hce", "value"),
    prevent_initial_call=True,
)
def show_similar(_n, selected_bucket, buckets, label_filter, min_hce):
    if not buckets:
        return "No buckets to compare."
    if not selected_bucket:
        selected_bucket = buckets[-1]

    # Filter candidate set
    cands = buckets
    if label_filter and label_filter != "ALL":
        cands = [b for b in buckets if b.get("label") == label_filter]
    if min_hce is not None:
        cands = [b for b in cands if (b.get("mean_HCE") or 0) >= float(min_hce)]
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
        hce_val = b.get("mean_HCE", 0.0) or 0.0
        items.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(f"{ts_txt} | {label}", className="fw-bold"),
                        html.Div(f"Similarity: {score:.3f}", className="text-muted"),
                        html.Div(note),
                        html.Div(f"mean_HCE (scaled): {hce_val:.3f}", className="small"),
                        html.Div(f"Tracks during window: {tracks_txt}", className="mt-1 small"),
                    ]
                ),
                className="mb-2",
            )
        )
    return items or "No similar buckets found."
