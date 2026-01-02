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
            dcc.Store(id="ndjson-emit-store", storage_type="session", data=True),
            dcc.Store(id="live-metrics-store"),
            dcc.Interval(id="live-metrics-interval", interval=1000, n_intervals=0),
            dcc.Interval(id="notif-interval", interval=3000, n_intervals=0),
            dcc.Interval(id="stress-interval", interval=5000, n_intervals=0),
            dcc.Store(id="notif-store"),
            dcc.Store(id="stress-store"),
            html.Div(
                id="notif-banner",
                className="text-center fw-bold",
                style={"display": "none", "marginBottom": "8px"},
            ),
            html.Div(
                id="stress-banner",
                className="text-center fw-bold",
                style={"display": "none", "marginBottom": "8px"},
            ),
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
                                dbc.Col(
                                    [
                                        dbc.Label("Note"),
                                        dbc.Input(id="qc-note", placeholder="Quick note", size="sm"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Activity"),
                                        dcc.Dropdown(
                                            id="qc-activity",
                                            options=[
                                                {"label": "Work", "value": "work"},
                                                {"label": "Creative", "value": "creative"},
                                                {"label": "Meditation", "value": "meditation"},
                                                {"label": "Social", "value": "social"},
                                                {"label": "Media", "value": "media"},
                                                {"label": "Exercise", "value": "exercise"},
                                                {"label": "Other", "value": "other"},
                                            ],
                                            value="work",
                                            clearable=False,
                                        ),
                                    ],
                                    md=2,
                                    sm=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Social"),
                                        dcc.Dropdown(
                                            id="qc-social",
                                            options=[
                                                {"label": "Alone", "value": "alone"},
                                                {"label": "With others", "value": "with_others"},
                                            ],
                                            value="alone",
                                            clearable=False,
                                        ),
                                        dbc.Input(
                                            id="qc-social-count",
                                            type="number",
                                            placeholder="count",
                                            min=0,
                                            step=1,
                                            size="sm",
                                            className="mt-1",
                                        ),
                                    ],
                                    md=2,
                                    sm=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Mood 1–10"),
                                        dcc.Slider(id="qc-mood", min=1, max=10, step=1, value=5, marks=None),
                                    ],
                                    md=2,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("AI/Chatbot"),
                                        dcc.Dropdown(
                                            id="qc-ai",
                                            options=[
                                                {"label": "None", "value": "none"},
                                                {"label": "Grok", "value": "grok"},
                                                {"label": "ChatGPT", "value": "chatgpt"},
                                                {"label": "Claude", "value": "claude"},
                                                {"label": "Custom", "value": "custom"},
                                            ],
                                            value="none",
                                            clearable=False,
                                        ),
                                    ],
                                    md=2,
                                    sm=12,
                                ),
                            ],
                            className="g-2 align-items-center mb-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Media/Person"),
                                        dbc.Input(id="qc-media", placeholder="e.g., Track: X | With: Y", size="sm"),
                                    ],
                                    md=6,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Window (min)"),
                                        dcc.Dropdown(
                                            id="qc-minutes",
                                            options=[{"label": f"{m} min", "value": m} for m in [1, 5, 10, 15, 30]],
                                            value=5,
                                            clearable=False,
                                        ),
                                    ],
                                    md=2,
                                    sm=6,
                                ),
                                dbc.Col(
                                    dbc.Button("Capture", id="qc-btn", color="primary", size="sm", className="mt-4"),
                                    md=2,
                                    sm=6,
                                ),
                                dbc.Col(html.Span(id="qc-status", className="text-muted small mt-4 d-block"), md=2, sm=12),
                            ],
                            className="g-2 align-items-center",
                        ),
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
    Output("notif-banner", "children"),
    Output("notif-banner", "style"),
    Input("notif-interval", "n_intervals"),
)
def _update_notif(_n):
    now = time.time()
    # Pull recent buckets (last 5 min) for stability
    buckets = storage.get_buckets_between(now - 300, now, session_id=None)
    if buckets:
        latest = buckets[-1]
        mean_x = float(latest.get("mean_X") or 0.0)
        std_q = float(latest.get("std_Q") or 0.0)
        mean_q = float(latest.get("mean_Q") or 0.0)
        mean_hce = float(latest.get("mean_HCE") or 0.0)
        q_slope = float(latest.get("Q_slope") or 0.0)
    else:
        status = registry.state_engine.get_status()
        snap = status.get("latest_snapshot") or {}
        raw = snap.get("raw", {}) if isinstance(snap.get("raw", {}), dict) else {}
        mean_x = float(snap.get("X") or 0.0)
        mean_q = float(snap.get("Q_vibe_focus") or snap.get("Q_vibe") or 0.0)
        std_q = float(raw.get("std_Q") or 0.0)
        q_slope = float(raw.get("Q_slope") or 0.0)
        mean_hce = float(raw.get("HCE_raw") or snap.get("HCE") or 0.0)

    msg = None
    color = "#88e"
    # Priority order
    if mean_hce > 15 and mean_x > 1.0:
        msg = "Transcendent harmony — prioritize insight / synthesis"
        color = "#d7b34d"
    elif mean_x > 1.7 and std_q < 0.12 and mean_q > 0.12:
        msg = "DEEP WORK: Stay in flow"
        color = "#55c8ff"
    elif q_slope > 0.0 and std_q >= 0.18:
        msg = "IDEATION rising — capture ideas now"
        color = "#ff9f43"
    elif mean_x < 0.9 and std_q < 0.12:
        msg = "Recovery: movement / breath break"
        color = "#7fd"

    if not msg:
        return "", {"display": "none"}

    style = {
        "display": "block",
        "marginBottom": "8px",
        "padding": "6px 10px",
        "borderRadius": "6px",
        "background": color,
        "color": "#020202",
        "opacity": 0.92,
    }
    return msg, style


@callback(
    Output("stress-banner", "children"),
    Output("stress-banner", "style"),
    Input("stress-interval", "n_intervals"),
)
def _stress_watch(_n):
    now = time.time()
    buckets = storage.get_buckets_between(now - 240, now, session_id=None)
    mean_x = std_q = mean_q = mean_hce = q_slope = 0.0
    valid_drop = False
    if buckets:
        b = buckets[-1]
        mean_x = float(b.get("mean_X") or 0.0)
        std_q = float(b.get("std_Q") or 0.0)
        mean_q = float(b.get("mean_Q") or 0.0)
        mean_hce = float(b.get("mean_HCE") or 0.0)
        q_slope = float(b.get("Q_slope") or 0.0)
        valid_drop = float(b.get("valid_fraction") or 1.0) < 0.75
    else:
        snap = registry.state_engine.get_status().get("latest_snapshot") or {}
        raw = snap.get("raw", {}) if isinstance(snap.get("raw", {}), dict) else {}
        mean_x = float(snap.get("X") or 0.0)
        mean_q = float(snap.get("Q_vibe_focus") or snap.get("Q_vibe") or 0.0)
        std_q = float(raw.get("std_Q") or 0.0)
        mean_hce = float(raw.get("HCE_raw") or snap.get("HCE") or 0.0)
        q_slope = float(raw.get("Q_slope") or 0.0)

    # Stress heuristic
    hce_ratio = mean_hce / (mean_q + 1e-6) if mean_q > 0 else 0.0
    likely_stress = (
        mean_x > 1.7
        and std_q > 0.15
        and mean_hce < 10.0
        and hce_ratio < 80.0
    )
    if q_slope < 0:
        likely_stress = likely_stress or (mean_x > 1.5 and std_q > 0.18)
    if valid_drop:
        likely_stress = likely_stress or (mean_x > 1.4 and std_q > 0.18)

    if not likely_stress:
        return "", {"display": "none"}

    msg = "Stress pattern detected — high activation with volatile richness. Recommend soothing track?"
    sub = "High activation with limited transcendent harmony—similar to past stress moments; relaxation advised."
    text = html.Div([html.Div(msg), html.Div(sub, className="small")])
    style = {
        "display": "block",
        "marginBottom": "8px",
        "padding": "8px 10px",
        "borderRadius": "6px",
        "background": "#ffb3b3",
        "color": "#220000",
        "opacity": 0.94,
    }
    return text, style


@callback(
    Output("qc-status", "children"),
    Input("qc-btn", "n_clicks"),
    State("qc-note", "value"),
    State("qc-minutes", "value"),
    State("qc-media", "value"),
    State("qc-activity", "value"),
    State("qc-social", "value"),
    State("qc-social-count", "value"),
    State("qc-mood", "value"),
    State("qc-ai", "value"),
    State("live-metrics-store", "data"),
    prevent_initial_call=True,
)
def _quick_capture(n, note, minutes, media, activity, social, social_count, mood, ai_use, live):
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
    tags = {
        "kind": "quick_capture",
        "window_min": window_min,
        "activity": activity or "",
        "social": social or "",
        "social_count": social_count,
        "mood": mood,
        "ai": ai_use or "",
    }
    context = {
        "window_min": window_min,
        "mean_X": mean_X,
        "mean_Q": mean_Q,
        "mean_HCE": mean_HCE,
        "live": live or {},
        "media": media or "",
        "activity": activity or "",
        "social": social or "",
        "social_count": social_count,
        "mood": mood,
        "ai": ai_use or "",
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

