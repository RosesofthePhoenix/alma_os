import time
import json
import sqlite3
import datetime as dt
from pathlib import Path
from typing import Dict
import requests

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from alma.ui import pages
from alma.app_state import registry
from alma.engine import storage
from alma import config


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
CANONICAL_PATH = DOCS_DIR / "alma_os_state_layer_canonical.md"
LEGACY_PATH = DOCS_DIR / "Legacy of the Soul Final.txt"
FAE_PATH = DOCS_DIR / "Fractal Architecture of Existence.pdf"


def _safe_read_text(path: Path, max_chars: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()[:max_chars]
    except Exception:
        return ""


CANONICAL_TEXT = _safe_read_text(CANONICAL_PATH, max_chars=8000)
LEGACY_EXCERPT = _safe_read_text(LEGACY_PATH, max_chars=1200)
FAE_EXCERPT = (
    "Fractal Architecture of Existence — recursive self-similarity, embodied rhythm, "
    "alignment of micro/macro states, coherence, intention, iterative refinement toward transcendence."
)
if not LEGACY_EXCERPT:
    LEGACY_EXCERPT = "Legacy of the Soul excerpt unavailable; see docs/Legacy of the Soul Final.txt."
if not CANONICAL_TEXT:
    CANONICAL_TEXT = "Canonical metric guide missing; ensure docs/alma_os_state_layer_canonical.md exists."

ORACLE_SYSTEM_PREFIX = (
    "You are a transcendent oracle embodying FAE/Legacy principles. "
    "Use this canonical metric guide and excerpts as system context. "
    "Always reference the current track from context; avoid repeating past tracks unless truly relevant.\n\n"
    f"{CANONICAL_TEXT}\n\n"
    f"FAE excerpt (Fractal Architecture of Existence): {FAE_EXCERPT}\n\n"
    f"Legacy excerpt: {LEGACY_EXCERPT}\n"
)


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
            dcc.Interval(id="relax-interval", interval=5000, n_intervals=0),
            dcc.Interval(id="transcend-interval", interval=7000, n_intervals=0),
            dcc.Store(id="notif-store"),
            dcc.Store(id="stress-store"),
            dcc.Store(id="relax-store"),
            dcc.Store(id="oracle-open", data=False),
            dcc.Store(id="oracle-history", data=[]),
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
                id="relax-banner",
                className="text-center fw-bold",
                style={"display": "none", "marginBottom": "8px"},
            ),
            html.Div(
                id="transcend-banner",
                className="text-center fw-bold",
                style={"display": "none", "marginBottom": "8px"},
            ),
            dbc.Button(
                "Oracle",
                id="oracle-toggle",
                color="warning",
                size="sm",
                className="position-fixed",
                style={"right": "12px", "top": "12px", "zIndex": 1200},
            ),
            html.Div(
                id="oracle-panel",
                style={
                    "display": "none",
                    "position": "fixed",
                    "top": "50px",
                    "right": "0",
                    "width": "360px",
                    "height": "90vh",
                    "background": "#0b0b12",
                    "color": "#e8e6ff",
                    "borderLeft": "1px solid #444",
                    "boxShadow": "-2px 0 8px rgba(0,0,0,0.5)",
                    "padding": "10px",
                    "zIndex": 1100,
                    "overflowY": "auto",
                },
                children=[
                    dbc.Button("✕", id="oracle-close", size="sm", color="secondary", className="float-end"),
                    html.Div("Oracle (dolphin3)", className="fw-bold mb-2", style={"color": "#ffd700"}),
                    dcc.Dropdown(
                        id="oracle-mode",
                        options=[
                            {"label": "Interpret State", "value": "interpret"},
                            {"label": "Forecast", "value": "forecast"},
                            {"label": "State Story", "value": "story"},
                            {"label": "Well-Being", "value": "wellbeing"},
                            {"label": "Philosophical Mirror", "value": "mirror"},
                            {"label": "Fractal Narrative", "value": "fractal"},
                        ],
                        value="interpret",
                        clearable=False,
                        style={"color": "#000"},
                    ),
                    html.Div(id="oracle-history-view", className="mt-2", style={"maxHeight": "50vh", "overflowY": "auto"}),
                    dcc.Textarea(
                        id="oracle-input",
                        placeholder="Ask the Oracle...",
                        style={"width": "100%", "height": "80px", "marginTop": "8px"},
                    ),
                    dbc.Button("Send", id="oracle-send", color="primary", size="sm", className="mt-2"),
                    html.Div(id="oracle-status", className="small text-muted mt-1"),
                ],
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
    # Load thresholds from profile if present
    try:
        with open(Path(__file__).resolve().parents[3] / "profiles" / "default.json", "r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception:
        profile = {}
    x_thr = float(profile.get("STRESS_X_THR", 1.7))
    stdq_thr = float(profile.get("STRESS_STDQ_THR", 0.12))
    hce_thr = float(profile.get("STRESS_HCE_THR", 10.0))
    ratio_thr = float(profile.get("STRESS_RATIO_THR", 50.0))
    qslope_thr = float(profile.get("STRESS_QSLOPE_THR", -0.001))
    auto_adapt = bool(profile.get("AUTO_ADAPT_STRESS"))

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
        mean_x > x_thr
        and std_q > stdq_thr
        and mean_hce < hce_thr
        and hce_ratio < ratio_thr
    )
    if q_slope < qslope_thr:
        likely_stress = likely_stress or (mean_x > max(1.5, x_thr - 0.2) and std_q > max(stdq_thr, 0.14))
    if valid_drop:
        likely_stress = likely_stress or (mean_x > 1.4 and std_q > 0.18)

    if not likely_stress:
        return "", {"display": "none"}

    # Suggest a recovery track from relax-inducing list
    recovery = ""
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            relax_df = pd.read_sql_query(
                """
                SELECT ts.title, ts.artist, COUNT(*) AS hits, AVG(b.mean_HCE) AS mean_HCE
                FROM track_sessions ts
                JOIN buckets b
                  ON b.bucket_start_ts >= ts.start_ts
                 AND b.bucket_start_ts <= COALESCE(ts.end_ts, ts.start_ts)
                WHERE b.std_Q < 0.10
                  AND b.mean_X BETWEEN 1.6 AND 1.8
                  AND b.mean_HCE < 50.0
                  AND COALESCE(b.valid_fraction, 1.0) >= 0.9
                  AND ABS(COALESCE(b.Q_slope, 0.0)) <= 0.01
                GROUP BY ts.title, ts.artist
                ORDER BY hits DESC, mean_HCE ASC
                LIMIT 1
                """,
                conn,
            )
            if not relax_df.empty:
                r = relax_df.iloc[0]
                recovery = f"{r.get('title','?')} — {r.get('artist','?')}"
    except Exception:
        pass

    msg = "Stress pattern detected — high activation with volatile richness / low harmony."
    sub = "Similar to past stress moments; relaxation advised."
    if recovery:
        sub = f"{sub} Suggested recovery: {recovery}"
    buttons = []
    if recovery:
        buttons.append(
            dbc.Button("Play Now", id="stress-play-now", color="danger", size="sm", className="ms-2")
        )
    if auto_adapt and recovery:
        buttons.append(
            dbc.Button("Auto-Adapt", id="stress-auto-adapt", color="secondary", size="sm", className="ms-2")
        )
    text_children = [html.Div(msg), html.Div(sub, className="small")]
    if buttons:
        text_children.append(html.Div(buttons, className="mt-1"))
    text = html.Div(text_children)
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
    Output("relax-banner", "children"),
    Output("relax-banner", "style"),
    Input("relax-interval", "n_intervals"),
)
def _relax_watch(_n):
    now = time.time()
    buckets = storage.get_buckets_between(now - 240, now, session_id=None)
    mean_x = std_q = mean_q = mean_hce = q_slope = 0.0
    valid_ok = True
    if buckets:
        b = buckets[-1]
        mean_x = float(b.get("mean_X") or 0.0)
        std_q = float(b.get("std_Q") or 0.0)
        mean_q = float(b.get("mean_Q") or 0.0)
        mean_hce = float(b.get("mean_HCE") or 0.0)
        q_slope = float(b.get("Q_slope") or 0.0)
        valid_ok = float(b.get("valid_fraction") or 1.0) >= 0.9
    else:
        snap = registry.state_engine.get_status().get("latest_snapshot") or {}
        raw = snap.get("raw", {}) if isinstance(snap.get("raw", {}), dict) else {}
        mean_x = float(snap.get("X") or 0.0)
        mean_q = float(snap.get("Q_vibe_focus") or snap.get("Q_vibe") or 0.0)
        std_q = float(raw.get("std_Q") or 0.0)
        mean_hce = float(raw.get("HCE_raw") or snap.get("HCE") or 0.0)
        q_slope = float(raw.get("Q_slope") or 0.0)
        valid_ok = bool((snap.get("reliability") or {}).get("valid", True))

    relaxed = (
        std_q < 0.10
        and 1.6 <= mean_x <= 1.8
        and mean_hce < 50.0
        and abs(q_slope) <= 0.01
        and valid_ok
    )
    # Avoid showing relax banner if stress is likely (prevents overlap)
    hce_ratio = mean_hce / (mean_q + 1e-6) if mean_q > 0 else 999.0
    stress_like = (mean_x > 1.7 and std_q > 0.12 and mean_hce < 10.0 and hce_ratio < 50.0)
    if stress_like or not relaxed:
        return "", {"display": "none"}

    recovery = ""
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            relax_df = pd.read_sql_query(
                """
                SELECT ts.title, ts.artist, COUNT(*) AS hits, AVG(b.mean_HCE) AS mean_HCE
                FROM track_sessions ts
                JOIN buckets b
                  ON b.bucket_start_ts >= ts.start_ts
                 AND b.bucket_start_ts <= COALESCE(ts.end_ts, ts.start_ts)
                WHERE b.std_Q < 0.10
                  AND b.mean_X BETWEEN 1.6 AND 1.8
                  AND b.mean_HCE < 50.0
                  AND COALESCE(b.valid_fraction, 1.0) >= 0.9
                  AND ABS(COALESCE(b.Q_slope, 0.0)) <= 0.01
                GROUP BY ts.title, ts.artist
                ORDER BY hits DESC, mean_HCE ASC
                LIMIT 1
                """,
                conn,
            )
            if not relax_df.empty:
                r = relax_df.iloc[0]
                recovery = f"{r.get('title','?')} — {r.get('artist','?')}"
    except Exception:
        pass

    msg = "Entering relaxed harmony — contemplative calm detected."
    sub = "Matches proven relaxation states; extend with a similar track?"
    if recovery:
        sub = f"{sub} Suggested: {recovery}"
    buttons = []
    if recovery:
        buttons.append(
            dbc.Button("Play Now", id="relax-play-now", color="success", size="sm", className="ms-2")
        )
        buttons.append(
            dbc.Button("Auto-Extend", id="relax-auto-extend", color="secondary", size="sm", className="ms-2")
        )
    text_children = [html.Div(msg), html.Div(sub, className="small")]
    if buttons:
        text_children.append(html.Div(buttons, className="mt-1"))
    text = html.Div(text_children)
    style = {
        "display": "block",
        "marginBottom": "8px",
        "padding": "8px 10px",
        "borderRadius": "6px",
        "background": "#b3ffd9",
        "color": "#002218",
        "opacity": 0.94,
    }
    return text, style


@callback(
    Output("transcend-banner", "children"),
    Output("transcend-banner", "style"),
    Input("transcend-interval", "n_intervals"),
)
def _transcend_watch(_n):
    now = time.time()
    window_days = 14
    buckets = storage.get_buckets_between(now - window_days * 86400, now, session_id=None)
    if not buckets:
        return "", {"display": "none"}
    hours = {}
    for b in buckets:
        ts = b.get("bucket_start_ts")
        if not ts:
            continue
        hour = dt.datetime.fromtimestamp(ts).hour
        hours.setdefault(hour, []).append(float(b.get("mean_HCE") or 0.0))
    if not hours:
        return "", {"display": "none"}
    hour_avg = [(h, sum(v) / len(v)) for h, v in hours.items()]
    hour_avg.sort(key=lambda x: x[1], reverse=True)
    top_hour, top_val = hour_avg[0]
    if top_val < 2.0:
        return "", {"display": "none"}
    # Compute next occurrence (today or tomorrow)
    now_dt = dt.datetime.now()
    target_dt = now_dt.replace(hour=top_hour, minute=0, second=0, microsecond=0)
    if target_dt < now_dt:
        target_dt = target_dt + dt.timedelta(days=1)
    eta_hours = (target_dt - now_dt).total_seconds() / 3600.0
    msg = "Transcendent window approaching — prepare synthesis."
    sub = f"Highest historical HCE around {top_hour:02d}:00 (avg {top_val:.2f}). ETA ~{eta_hours:.1f}h."
    text = html.Div([html.Div(msg), html.Div(sub, className="small")])
    style = {
        "display": "block",
        "marginBottom": "8px",
        "padding": "8px 10px",
        "borderRadius": "6px",
        "background": "#ffe9a3",
        "color": "#332400",
        "opacity": 0.94,
    }
    return text, style


@callback(
    Output("oracle-open", "data"),
    Output("oracle-panel", "style"),
    Input("oracle-toggle", "n_clicks"),
    Input("oracle-close", "n_clicks"),
    State("oracle-open", "data"),
    prevent_initial_call=True,
)
def toggle_oracle(open_click, close_click, is_open):
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    if triggered == "oracle-toggle":
        is_open = not bool(is_open)
    elif triggered == "oracle-close":
        is_open = False
    style = {
        "display": "block" if is_open else "none",
        "position": "fixed",
        "top": "50px",
        "right": "0",
        "width": "360px",
        "height": "90vh",
        "background": "#0b0b12",
        "color": "#e8e6ff",
        "borderLeft": "1px solid #444",
        "boxShadow": "-2px 0 8px rgba(0,0,0,0.5)",
        "padding": "10px",
        "zIndex": 1100,
        "overflowY": "auto",
    }
    return is_open, style


def _oracle_context():
    status = registry.state_engine.get_status()
    snap = status.get("latest_snapshot") or {}
    ctx = {
        "X": snap.get("X"),
        "Q": snap.get("Q_vibe_focus") or snap.get("Q_vibe"),
        "HCE": snap.get("HCE"),
        "valid": (snap.get("reliability") or {}).get("valid"),
        "ts": snap.get("ts_unix"),
    }
    now = time.time()
    buckets = storage.get_buckets_between(now - 1800, now, session_id=None)
    if buckets:
        hces = [b.get("mean_HCE", 0.0) or 0.0 for b in buckets if b.get("mean_HCE") is not None]
        ctx["recent_mean_HCE"] = float(np.nanmean(hces)) if hces else None
    # top track
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            top_df = pd.read_sql_query(
                "SELECT title, artist, mean_HCE FROM track_sessions WHERE mean_HCE IS NOT NULL ORDER BY mean_HCE DESC LIMIT 1",
                conn,
            )
            if not top_df.empty:
                row = top_df.iloc[0]
                ctx["top_track"] = f"{row.get('title','?')} — {row.get('artist','?')} (HCE {row.get('mean_HCE',0):.2f})"
    except Exception:
        pass
    try:
        latest_track = storage.get_latest_spotify(session_id=None)
        if latest_track and latest_track.get("track_name"):
            ctx["current_track"] = f"{latest_track.get('track_name','?')} — {latest_track.get('artists','?')}"
    except Exception:
        pass
    return ctx


def _oracle_prompt(mode: str, user_text: str, ctx: Dict[str, object]) -> str:
    system = ORACLE_SYSTEM_PREFIX
    metrics = f"Live: X={ctx.get('X')}, Q={ctx.get('Q')}, HCE={ctx.get('HCE')}, valid={ctx.get('valid')}."
    rec = ctx.get("recent_mean_HCE")
    metrics += f" Recent mean_HCE (30m): {rec:.2f}." if rec is not None else ""
    current_track = ctx.get("current_track") or "n/a"
    top_track = ctx.get("top_track") or "n/a"
    base = (
        f"{system}\nMode: {mode}\nContext: {metrics}\n"
        f"Current track: {current_track}\nTop track (historical): {top_track}\n"
        "Prefer grounding in the current track; avoid repeating historical tracks unless clearly relevant.\n"
        f"User: {user_text}"
    )
    if mode == "forecast":
        base += "\nProvide a short forecast for upcoming transcendence windows and guidance."
    elif mode == "story":
        base += "\nTell a short weekly state story blending peaks, media, intentions."
    elif mode == "wellbeing":
        base += "\nOffer gentle well-being and recovery advice based on signal stability."
    elif mode == "mirror":
        base += "\nSpeak as a philosophical mirror / future self, referencing FAE/Legacy themes."
    elif mode == "fractal":
        base += "\nDescribe the day as a fractal narrative; poetic but concise."
    else:
        base += "\nInterpret the current state and offer a succinct next step."
    return base


def _ollama_call(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": "huihui_ai/dolphin3-abliterated", "prompt": prompt, "stream": False, "options": {"temperature": 0.6}}
    attempts = 3
    delays = [5, 10, 20]

    for attempt in range(attempts):
        try:
            print(f"[oracle] Connecting to Ollama (attempt {attempt + 1}/{attempts})", flush=True)
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response") or "No response."
            return f"Ollama error: {resp.status_code}"
        except requests.exceptions.Timeout:
            if attempt < attempts - 1:
                delay = delays[attempt]
                print(f"[oracle] Ollama timeout; retrying in {delay}s", flush=True)
                time.sleep(delay)
                continue
            return "Ollama unavailable—start server and reload"
        except Exception as exc:
            if attempt < attempts - 1:
                delay = delays[attempt]
                print(f"[oracle] Ollama attempt {attempt + 1} failed: {exc}; retrying in {delay}s", flush=True)
                time.sleep(delay)
                continue
            return "Ollama unavailable—start server and reload"

    return "Ollama unavailable—start server and reload"


@callback(
    Output("oracle-history", "data"),
    Output("oracle-history-view", "children"),
    Output("oracle-status", "children"),
    Input("oracle-send", "n_clicks"),
    State("oracle-mode", "value"),
    State("oracle-input", "value"),
    State("oracle-history", "data"),
    prevent_initial_call=True,
)
def run_oracle(_n, mode, text, history):
    history = history or []
    user_msg = text or ""
    ctx = _oracle_context()
    prompt = _oracle_prompt(mode or "interpret", user_msg, ctx)
    reply = _ollama_call(prompt)
    history.append({"role": "user", "text": user_msg})
    history.append({"role": "oracle", "text": reply})
    view = []
    for m in history[-12:]:
        clr = "#ffd700" if m["role"] == "oracle" else "#9ad1ff"
        who = "Oracle" if m["role"] == "oracle" else "You"
        view.append(html.Div(f"{who}: {m['text']}", style={"color": clr, "marginBottom": "6px"}))
    status = "Responded via Ollama" if "Ollama" not in reply else reply
    return history, view, status


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

