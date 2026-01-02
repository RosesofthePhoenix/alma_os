import time

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, dcc, html

from alma.ui import pages
from alma.app_state import registry
from alma.engine import storage


def build_sidebar() -> html.Div:
    """Construct the left navigation sidebar."""
    return html.Div(
        [
            html.Div("alma_os", className="sidebar-title"),
            dbc.Nav(pages.get_nav_links(), vertical=True, pills=True, className="sidebar-nav"),
        ],
        className="sidebar",
    )


def build_layout() -> dbc.Container:
    """Create the top-level Dash layout with routing slots."""
    return dbc.Container(
        [
            dcc.Location(id="url"),
            dcc.Store(id="ndjson-emit-store", storage_type="session", data=False),
            dcc.Store(id="live-metrics-store"),
            dcc.Interval(id="live-metrics-interval", interval=1000, n_intervals=0),
            html.Div(
                id="ndjson-indicator",
                className="text-end text-info small my-2",
                children="NDJSON: OFF",
            ),
            html.Div(
                id="live-metrics-bar",
                className="text-end text-info small mb-2",
                children="Live: —",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(dbc.Label("Quick Capture"), md=12, className="fw-bold"),
                                dbc.Col(dbc.Input(id="qc-note", placeholder="Note"), md=5, sm=12, className="mb-2"),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="qc-minutes",
                                        options=[{"label": f"{m} min", "value": m} for m in [1, 5, 10, 15, 30]],
                                        value=5,
                                        clearable=False,
                                    ),
                                    md=2,
                                    sm=6,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    dbc.Input(id="qc-media", placeholder="Media/Person (e.g., Track: X | With: Y)"),
                                    md=5,
                                    sm=12,
                                    className="mb-2",
                                ),
                            ],
                            className="g-2 align-items-center",
                        ),
                        dbc.Button("Capture", id="qc-btn", color="primary", size="sm"),
                        html.Span(id="qc-status", className="ms-2 text-muted small"),
                    ]
                ),
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(build_sidebar(), width=12, md=3, lg=2, className="sidebar-column"),
                    dbc.Col(html.Div(id="page-content", className="page-content"), width=12, md=9, lg=10),
                ],
                className="app-row",
            ),
        ],
        fluid=True,
        className="app-container",
    )


@callback(Output("ndjson-indicator", "children"), Output("ndjson-indicator", "className"), Input("ndjson-emit-store", "data"))
def _update_ndjson_indicator(enabled):
    enabled = bool(enabled)
    text = f"NDJSON: {'ON' if enabled else 'OFF'}"
    cls = "text-end small my-2 " + ("text-success" if enabled else "text-secondary")
    return text, cls


@callback(
    Output("live-metrics-store", "data"),
    Output("live-metrics-bar", "children"),
    Input("live-metrics-interval", "n_intervals"),
)
def _poll_live_metrics(_n):
    status = registry.state_engine.get_status()
    snap = status.get("latest_snapshot") or {}
    X = snap.get("X") or 0.0
    Q = snap.get("Q_vibe_focus") or snap.get("Q_vibe") or 0.0
    HCE = snap.get("HCE") or 0.0
    text = f"Live: X={X:.3f} | Q={Q:.3f} | HCE={HCE:.3f}"
    return {"X": X, "Q": Q, "HCE": HCE, "ts": snap.get("ts_unix")}, text


@callback(
    Output("qc-status", "children"),
    Input("qc-btn", "n_clicks"),
    State("qc-note", "value"),
    State("qc-minutes", "value"),
    State("qc-media", "value"),
    State("live-metrics-store", "data"),
    prevent_initial_call=True,
)
def _quick_capture(n, note, minutes, media, live):
    if not n:
        raise dash.exceptions.PreventUpdate  # type: ignore
    window_min = int(minutes or 5)
    now_ts = time.time()
    ts0 = now_ts - window_min * 60
    buckets = storage.get_buckets_between(ts0, now_ts, session_id=None)
    mean_X = float(np.nanmean([b.get("mean_X", 0.0) for b in buckets])) if buckets else 0.0
    mean_Q = float(np.nanmean([b.get("mean_Q", 0.0) for b in buckets])) if buckets else 0.0
    mean_HCE = float(np.nanmean([b.get("mean_HCE", 0.0) for b in buckets])) if buckets else 0.0
    latest_track = storage.get_latest_spotify(session_id=None)
    track_txt = None
    if latest_track and latest_track.get("is_playing"):
        track_txt = f"{latest_track.get('track_name') or ''} — {latest_track.get('artists') or ''}"
    tags = {"kind": "quick_capture", "window_min": window_min}
    context = {
        "window_min": window_min,
        "mean_X": mean_X,
        "mean_Q": mean_Q,
        "mean_HCE": mean_HCE,
        "live": live or {},
        "media": media or "",
        "track": track_txt,
    }
    storage.insert_event(
        ts=now_ts,
        session_id=registry.state_engine.get_session_id() or "",
        kind="quick_capture",
        label="Quick Capture",
        note=note or "",
        tags_json=tags,
        context_json=context,
    )
    return "Captured."

