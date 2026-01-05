import time
import json
import sqlite3
import datetime as dt
from pathlib import Path
from typing import Dict
import re
import requests

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.dependencies import ClientsideFunction

from alma.ui import pages
from alma.app_state import registry
from alma.engine import storage
from alma import config


DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
MASTER_PATH = DOCS_DIR / "Canonical Master Document- The Complete Context of Ray Craigs Body of Work and ALMA OS.txt"


def _safe_read_text(path: Path, max_chars: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()[:max_chars]
    except Exception:
        return ""


MASTER_TEXT = _safe_read_text(MASTER_PATH, max_chars=12000)
if not MASTER_TEXT:
    MASTER_TEXT = (
        "Canonical Master Document missing; ensure it exists at "
        "docs/Canonical Master Document- The Complete Context of Ray Craigs Body of Work and ALMA OS.txt"
    )

ORACLE_SYSTEM_PREFIX = (
    "You are a super-intelligent, neutral, analytical AI colleague embedded in ALMA OS. "
    "Your responses must always be formal, precise, evidence-based, concise yet thorough, "
    "and focused on actionable insights derived from live/historical EEG states (X, Q variants, HCE), "
    "reliability/validity, bucket aggregates, events, Spotify outcomes, social context, intentions, recipes, and scheduler data.\n\n"
    "Core context (single source of truth):\n"
    f"{MASTER_TEXT}\n\n"
    "Never use mystical, transcendent, or inspirational language. "
    "Prioritize empirical rigor. "
    "If data is insufficient, state it clearly and suggest improvements. "
    "Reference specific trends (e.g., 'Based on last 30 days buckets...'). "
    "Suggest recipes, scheduler adjustments, or actuations only when empirically supported."
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
            dcc.Interval(id="forecast-interval", interval=8000, n_intervals=0),
            dcc.Store(id="notif-store"),
            dcc.Store(id="stress-store"),
            dcc.Store(id="relax-store"),
            dcc.Store(id="oracle-open", data=False),
            dcc.Store(id="oracle-history", data=[]),
            dcc.Store(id="oracle-speak-input", data=True),
            dcc.Store(id="oracle-read-output", data=True),
            dcc.Store(id="forecast-store"),
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
            html.Div(
                id="forecast-banner",
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
                    "width": "1080px",
                    "maxWidth": "90vw",
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
                    html.Div(
                        id="oracle-history-view",
                        className="mt-2",
                        style={
                            "maxHeight": "55vh",
                            "overflowY": "auto",
                            "whiteSpace": "pre-wrap",
                            "wordBreak": "break-word",
                            "fontSize": "1.1rem",
                            "padding": "10px",
                            "lineHeight": "1.5",
                        },
                    ),
                    dcc.Textarea(
                        id="oracle-input",
                        placeholder="Ask the Oracle...",
                        style={"width": "100%", "height": "80px", "marginTop": "8px"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    "🎤",
                                    id="oracle-mic",
                                    color="info",
                                    size="sm",
                                    className="mt-2",
                                    style={"boxShadow": "0 0 8px #0ff", "minWidth": "48px"},
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.Button(
                                    "Send",
                                    id="oracle-send",
                                    color="primary",
                                    size="sm",
                                    className="mt-2",
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.Checklist(
                                    id="oracle-voice-toggles",
                                    options=[
                                        {"label": "Speak Input", "value": "speak"},
                                        {"label": "Read Responses", "value": "read"},
                                    ],
                                    value=["speak", "read"],
                                    switch=True,
                                    inline=True,
                                    className="mt-2 text-info",
                                ),
                            ),
                        ],
                        className="g-2 align-items-center",
                    ),
                    html.Div(id="oracle-voice-status", className="small text-muted mt-1"),
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
                    dbc.Col(build_sidebar(), xs=12, md=1, className="sidebar-column"),
                    dbc.Col(html.Div(id="page-content", className="page-content"), xs=12, md=11),
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
    Output("forecast-store", "data"),
    Output("forecast-banner", "children"),
    Output("forecast-banner", "style"),
    Input("forecast-interval", "n_intervals"),
)
def _forecast_watch(_n):
    data = _forecast_probabilities()
    if not data:
        return {}, "", {"display": "none"}
    probs = data.get("transcend") or []
    visible_probs = [p for p in probs if p.get("prob") is not None]
    if visible_probs:
        top = max(visible_probs, key=lambda p: p["prob"])
        msg = f"Transcendent window likelihood in {top['h']}h: {top['prob']*100:.1f}% (p90={data.get('p90',0):.2f})"
    else:
        msg = "Forecast pending more history."
    strain = data.get("strain")
    media = data.get("media")
    parts = [msg]
    if strain is not None:
        parts.append(f"Strain risk: {strain*100:.1f}%")
    if media is not None:
        parts.append(f"Media lift prob: {media*100:.1f}%")
    banner_txt = " | ".join(parts)
    style = {
        "display": "block",
        "marginBottom": "8px",
        "padding": "6px 10px",
        "borderRadius": "6px",
        "background": "#cddc39",
        "color": "#202020",
        "opacity": 0.9,
    }
    return data, banner_txt, style


@callback(
    Output("oracle-open", "data"),
    Output("oracle-panel", "style"),
    Output("oracle-speak-input", "data"),
    Output("oracle-read-output", "data"),
    Input("oracle-toggle", "n_clicks"),
    Input("oracle-close", "n_clicks"),
    Input("oracle-voice-toggles", "value"),
    State("oracle-open", "data"),
    State("oracle-speak-input", "data"),
    State("oracle-read-output", "data"),
    prevent_initial_call=True,
)
def toggle_oracle(open_click, close_click, toggles, is_open, speak_on, read_on):
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    if triggered == "oracle-toggle":
        is_open = not bool(is_open)
    elif triggered == "oracle-close":
        is_open = False
    # Update toggles from checklist
    toggles = toggles or []
    speak_on = "speak" in toggles
    read_on = "read" in toggles
    panel_style = {
        "display": "block" if is_open else "none",
        "position": "fixed",
        "top": "50px",
        "right": "0",
        "width": "1080px",
        "maxWidth": "90vw",
        "height": "90vh",
        "background": "#0b0b12",
        "color": "#e8e6ff",
        "borderLeft": "1px solid #444",
        "boxShadow": "-2px 0 8px rgba(0,0,0,0.5)",
        "padding": "10px",
        "zIndex": 1100,
        "overflowY": "auto",
    }
    return is_open, panel_style, speak_on, read_on


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
            ctx["current_track_id"] = latest_track.get("track_id")
            ctx["current_progress_ms"] = latest_track.get("progress_ms")
    except Exception:
        pass
    # Forecast snapshot
    ctx["forecast"] = _forecast_probabilities()
    # Patterns / intentions
    ctx["patterns"] = _pattern_revelations()
    # Section summary
    try:
        tid = ctx.get("current_track_id")
        if tid:
            secs = storage.list_track_sections(tid, limit_sessions=1)
            if secs:
                best = max(secs, key=lambda s: s.get("mean_HCE") or 0.0)
                ctx["section_summary"] = {
                    "label": best.get("section_label"),
                    "mean_HCE": best.get("mean_HCE"),
                    "mean_Q": best.get("mean_Q"),
                    "mean_X": best.get("mean_X"),
                }
                ctx["section_top3"] = [
                    {
                        "label": s.get("section_label"),
                        "mean_HCE": s.get("mean_HCE"),
                        "lift": None,
                    }
                    for s in sorted(secs, key=lambda s: s.get("mean_HCE") or 0.0, reverse=True)[:3]
                ]
    except Exception:
        pass
    return ctx


def _oracle_prompt(mode: str, user_text: str, ctx: Dict[str, object]) -> str:
    metrics = (
        f"Live state: X={ctx.get('X')}, Q={ctx.get('Q')}, HCE={ctx.get('HCE')}, "
        f"validity={'valid' if ctx.get('valid') else 'invalid'}."
    )
    recent_hce = ctx.get("recent_mean_HCE")
    if recent_hce is not None:
        metrics += f" Recent mean HCE (30m rolling): {recent_hce:.3f}."

    current_track = ctx.get("current_track")
    if current_track and current_track != "n/a":
        metrics += f" Current track: {current_track}."

    patterns = ctx.get("patterns") or {}
    pattern_lines = []
    if patterns.get("social_best"):
        s, v = patterns["social_best"]
        pattern_lines.append(f"Social context with highest HCE: {s} (mean {v:.2f}).")
    if patterns.get("activity_best"):
        s, v = patterns["activity_best"]
        pattern_lines.append(f"Activity with highest HCE: {s} (mean {v:.2f}).")
    if patterns.get("mood_peak"):
        m, v = patterns["mood_peak"]
        pattern_lines.append(f"Mood peak: {m} → HCE {v:.2f}.")
    if patterns.get("media_peak"):
        pattern_lines.append(f"Top media lift: {patterns['media_peak']}.")
    if patterns.get("intention_top"):
        inten, delta = patterns["intention_top"]
        pattern_lines.append(f"Intention payoff: \"{inten}\" → HCE Δ {delta:.2f}.")
    patterns_txt = "\n".join(pattern_lines) if pattern_lines else "No strong patterns yet."

    section_lines = []
    section_summary = ctx.get("section_summary") or {}
    if section_summary:
        section_lines.append(
            f"Best section: {section_summary.get('label','?')} "
            f"(HCE {section_summary.get('mean_HCE',0):.2f}, Q {section_summary.get('mean_Q',0):.3f}, X {section_summary.get('mean_X',0):.3f})."
        )
    top3 = ctx.get("section_top3") or []
    if top3:
        for s in top3:
            section_lines.append(f"{s.get('label')}: HCE {s.get('mean_HCE',0):.2f}")
    sections_txt = "\n".join(section_lines) if section_lines else "Sections: n/a"

    forecast = ctx.get("forecast") or {}
    forecast_lines = []
    p90 = forecast.get("p90")
    if p90 is not None:
        forecast_lines.append(f"P90 HCE={p90:.2f}")
    trans = forecast.get("transcend") or []
    vis = [p for p in trans if p.get("prob") is not None]
    if vis:
        best = max(vis, key=lambda p: p["prob"])
        forecast_lines.append(f"Transcendent in {best['h']}h: {best['prob']*100:.1f}%")
    strain = forecast.get("strain")
    if strain is not None:
        forecast_lines.append(f"Strain risk: {strain*100:.1f}%")
    media = forecast.get("media")
    if media is not None:
        forecast_lines.append(f"Media lift prob: {media*100:.1f}%")
    forecast_txt = "\n".join(forecast_lines) if forecast_lines else "Forecast: n/a"

    section = ctx.get("section_summary") or {}
    section_txt = ""
    if section:
        section_txt = (
            f"Section highlight: {section.get('label','?')} "
            f"(HCE {section.get('mean_HCE',0):.2f}, Q {section.get('mean_Q',0):.3f}, X {section.get('mean_X',0):.3f})"
        )

    base_prompt = (
        f"{ORACLE_SYSTEM_PREFIX}\n\n"
        f"Mode: {mode}\n"
        f"Current context: {metrics}\n\n"
        f"Patterns:\n{patterns_txt}\n\n"
        f"Sections:\n{sections_txt}\n\n"
        f"Forecast:\n{forecast_txt}\n\n"
        f"User query: {user_text}\n"
        "Respond with analytical insights and actionable recommendations only."
    )

    mode_guidance = {
        "interpret": "Analyze current state and recent patterns; explain drivers of HCE shifts.",
        "forecast": "Project likely HCE windows based on circadian/hour-of-day historical aggregates.",
        "story": "Generate concise narrative summary of session/day trends (data-driven only).",
        "wellbeing": "Recommend evidence-based adjustments for sustained harmony and recovery.",
        "mirror": "Reflect observed patterns neutrally with empirical correlations.",
        "fractal": "Identify recursive patterns across scales (micro neural → macro daily).",
    }.get(mode.lower(), "")

    return base_prompt + (f"\nMode guidance: {mode_guidance}" if mode_guidance else "")


def _ollama_call(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": "huihui_ai/dolphin3-abliterated", "prompt": prompt, "stream": False, "options": {"temperature": 0.6}}
    attempts = 3
    delays = [5, 10, 20]

    for attempt in range(attempts):
        try:
            print(f"[oracle] Connecting to Ollama (attempt {attempt + 1}/{attempts})", flush=True)
            resp = requests.post(url, json=payload, timeout=90)
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


def _highlight_hce(text: str):
    parts = []
    for seg in re.split(r"(HCE)", text or ""):
        if seg == "HCE":
            parts.append(html.Span("HCE", style={"color": "#d7b34d", "fontWeight": 600}))
        else:
            parts.append(seg)
    return parts


def _forecast_probabilities() -> Dict[str, object]:
    now = time.time()
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT bucket_start_ts, mean_HCE, mean_X, std_Q FROM buckets WHERE bucket_start_ts IS NOT NULL",
                conn,
            )
    except Exception:
        return {}
    if df.empty:
        return {}
    df = df.dropna(subset=["bucket_start_ts", "mean_HCE"])
    if df.empty:
        return {}
    df["hour"] = df["bucket_start_ts"].apply(lambda t: dt.datetime.fromtimestamp(t).hour)
    p90 = float(np.nanpercentile(df["mean_HCE"], 90)) if not df["mean_HCE"].isna().all() else 0.0
    probs = []
    for h in [1, 3, 6]:
        target_hour = int(dt.datetime.fromtimestamp(now + h * 3600).hour)
        subset = df[df["hour"] == target_hour]
        if subset.empty:
            probs.append({"h": h, "prob": None})
        else:
            prob = float((subset["mean_HCE"] > p90).mean())
            probs.append({"h": h, "prob": prob})
    strain_prob = float(
        ((df["mean_X"] > 1.7) & (df["std_Q"] > 0.12)).mean()
    ) if not df.empty else None

    media_prob = None
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            tracks = pd.read_sql_query(
                "SELECT mean_HCE FROM track_sessions WHERE mean_HCE IS NOT NULL", conn
            )
        if not tracks.empty:
            median_hce = tracks["mean_HCE"].median()
            media_prob = float((tracks["mean_HCE"] > median_hce).mean())
    except Exception:
        media_prob = None

    return {"p90": p90, "transcend": probs, "strain": strain_prob, "media": media_prob}


def _pattern_revelations() -> Dict[str, object]:
    out: Dict[str, object] = {}
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            events = pd.read_sql_query(
                "SELECT ts, note, tags_json, context_json FROM events WHERE ts IS NOT NULL", conn
            )
            buckets = pd.read_sql_query(
                "SELECT bucket_start_ts, mean_HCE, mean_X, std_Q, mean_Q FROM buckets WHERE bucket_start_ts IS NOT NULL",
                conn,
            )
            tracks = pd.read_sql_query(
                "SELECT title, artist, mean_HCE, mean_Q, mean_X FROM track_sessions WHERE mean_HCE IS NOT NULL",
                conn,
            )
    except Exception:
        return out

    if not buckets.empty:
        # Social/mood/activity correlations (simple averages)
        def parse_ctx(row):
            raw = row.get("context_json") or row.get("tags_json") or "{}"
            try:
                return json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                return {}

        rows = []
        for _, r in events.iterrows():
            ctx = parse_ctx(r)
            social = ctx.get("social") or ctx.get("home_bookmark_social") or "unknown"
            activity = ctx.get("activity") or ctx.get("home_bookmark_activity") or "unknown"
            mood = ctx.get("mood")
            ts = r.get("ts") or 0
            nearby = buckets[(buckets["bucket_start_ts"] >= ts - 900) & (buckets["bucket_start_ts"] <= ts + 900)]
            if nearby.empty:
                continue
            hce = nearby["mean_HCE"].mean()
            rows.append({"social": social, "activity": activity, "mood": mood, "hce": hce})
        if rows:
            df = pd.DataFrame(rows)
            social_best = df.groupby("social")["hce"].mean().sort_values(ascending=False).head(1)
            if not social_best.empty:
                out["social_best"] = (social_best.index[0], float(social_best.iloc[0]))
            act_best = df.groupby("activity")["hce"].mean().sort_values(ascending=False).head(1)
            if not act_best.empty:
                out["activity_best"] = (act_best.index[0], float(act_best.iloc[0]))
            mood_drop = df.dropna(subset=["mood"])
            if not mood_drop.empty:
                mood_corr = mood_drop.groupby("mood")["hce"].mean().sort_values(ascending=False).head(1)
                if not mood_corr.empty:
                    out["mood_peak"] = (int(mood_corr.index[0]), float(mood_corr.iloc[0]))

    if not tracks.empty:
        top_track = tracks.sort_values("mean_HCE", ascending=False).head(1)
        if not top_track.empty:
            r = top_track.iloc[0]
            out["media_peak"] = f"{r.get('title','?')} — {r.get('artist','?')} (HCE {r.get('mean_HCE',0):.2f})"

    # Intention follow-up: compare HCE after intention vs baseline before
    if not events.empty and not buckets.empty:
        inten_rows = []
        for _, r in events.iterrows():
            ctx_raw = r.get("context_json") or r.get("tags_json") or "{}"
            try:
                ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else (ctx_raw or {})
            except Exception:
                ctx = {}
            intention = ctx.get("intention") or ""
            if not intention:
                continue
            ts = r.get("ts") or 0
            before = buckets[(buckets["bucket_start_ts"] >= ts - 900) & (buckets["bucket_start_ts"] < ts)]
            after = buckets[(buckets["bucket_start_ts"] >= ts) & (buckets["bucket_start_ts"] <= ts + 1200)]
            if after.empty:
                continue
            before_mean = before["mean_HCE"].mean() if not before.empty else np.nan
            after_mean = after["mean_HCE"].mean()
            delta = after_mean - before_mean if not np.isnan(before_mean) else np.nan
            inten_rows.append({"intention": intention, "delta": delta, "after": after_mean})
        inten_df = pd.DataFrame(inten_rows)
        if not inten_df.empty:
            inten_df = inten_df.dropna(subset=["delta"])
            if not inten_df.empty:
                top = inten_df.sort_values("delta", ascending=False).head(1)
                out["intention_top"] = (top.iloc[0]["intention"], float(top.iloc[0]["delta"]))

    return out


# Client-side callbacks for voice features
dash.clientside_callback(
    ClientsideFunction(namespace="oracle", function_name="micHandler"),
    [Output("oracle-input", "value"), Output("oracle-voice-status", "children")],
    Input("oracle-mic", "n_clicks"),
    State("oracle-speak-input", "data"),
    State("oracle-input", "value"),
    prevent_initial_call=True,
)


dash.clientside_callback(
    ClientsideFunction(namespace="oracle", function_name="ttsHandler"),
    Output("oracle-status", "title"),
    Input("oracle-history", "data"),
    State("oracle-read-output", "data"),
    prevent_initial_call=True,
)


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
        view.append(html.Div([f"{who}: "] + _highlight_hce(str(m["text"])), style={"color": clr, "marginBottom": "8px", "lineHeight": "1.5"}))
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

