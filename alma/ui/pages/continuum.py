import time
import json
from typing import List, Dict, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, callback, dcc, html

from alma.engine import storage
from alma.app_state import registry
from alma.utils import ambient_logger


CATEGORY_COLORS = {
    "Deep Work": "#00ff9d",
    "Creative": "#ff00ff",
    "Transcendent Synthesis": "#ffd700",
    "Work": "#76c7ff",
    "Chilling": "#9ba0a6",
    "Reading": "#ff9f43",
    "Socializing": "#ff6b6b",
    "Brainstorming": "#a66cff",
    "Work Phone": "#5bc0de",
    "Exercise": "#00e8ff",
}
TYPE_COLORS = {"time": "#00ff9d", "note": "#ffcc80", "journal": "#80bfff", "live": "#76c7ff"}
TODAY = pd.Timestamp.today().normalize()
DEFAULT_START = TODAY - pd.Timedelta(days=6)


def _category_color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, "#d7b34d")


def _type_color(entry_type: str) -> str:
    return TYPE_COLORS.get(entry_type, "#d7b34d")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _live_hce_badge() -> html.Span:
    status = registry.state_engine.get_status()
    hce = status.get("latest_snapshot", {}).get("HCE")
    txt = f"HCE: {hce:.2f}" if hce is not None else "HCE: n/a"
    return html.Span(txt, className="badge bg-success ms-2")


def _aggregate_metrics(start_ts: float, end_ts: float) -> Dict[str, Optional[float]]:
    rows = storage.list_state_summary(start_ts, end_ts, limit=10000)
    if not rows:
        return {"hce_mean": None, "hce_peak": None, "q_mean": None, "reliability_mean": None, "mean_x": None, "peak_x": None, "peak_q": None}
    df = pd.DataFrame(rows)
    hces = df["hce"].dropna() if "hce" in df else pd.Series([], dtype=float)
    qs = df["q"].dropna() if "q" in df else pd.Series([], dtype=float)
    xs = df["x"].dropna() if "x" in df else pd.Series([], dtype=float)
    return {
        "hce_mean": float(hces.mean()) if not hces.empty else None,
        "hce_peak": float(hces.max()) if not hces.empty else None,
        "q_mean": float(qs.mean()) if not qs.empty else None,
        "mean_x": float(xs.mean()) if not xs.empty else None,
        "peak_x": float(xs.max()) if not xs.empty else None,
        "peak_q": float(qs.max()) if not qs.empty else None,
        "reliability_mean": None,
    }


def _ollama_call(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": "huihui_ai/dolphin3-abliterated", "prompt": prompt, "stream": False, "options": {"temperature": 0.6}}
    attempts = 2
    delays = [5, 10]
    for attempt in range(attempts):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response") or "No response."
            return f"Ollama error: {resp.status_code}"
        except requests.exceptions.Timeout:
            if attempt < attempts - 1:
                time.sleep(delays[attempt])
                continue
            return "Ollama unavailable—start server and reload"
        except Exception as exc:
            if attempt < attempts - 1:
                time.sleep(delays[attempt])
                continue
            return f"Ollama unavailable ({exc})"
    return "Ollama unavailable—start server and reload"


def _combined_entries(start_ts: float, end_ts: float, include_running: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    combined: List[Dict[str, object]] = []
    try:
        time_entries = storage.list_time_entries_between(start_ts, end_ts, limit=2000)
    except Exception:
        time_entries = []
    for t in time_entries:
        combined.append(
            {
                **t,
                "type": "time",
                "category": t.get("category") or "Flow Block",
                "start_time": t.get("start_time"),
                "end_time": t.get("end_time"),
                "duration": t.get("duration") or max(0, (t.get("end_time") or 0) - (t.get("start_time") or 0)),
            }
        )
    try:
        notes = storage.list_continuum_notes(limit=2000)
    except Exception:
        notes = []
    for n in notes:
        st = n.get("start_ts") or n.get("ts") or n.get("end_ts")
        et = n.get("end_ts") or st
        if not st:
            continue
        if st < start_ts or st > end_ts:
            continue
        combined.append(
            {
                "type": "note",
                "category": n.get("category") or "Note",
                "start_time": st,
                "end_time": et,
                "duration": max(0, (et or st) - st) or 60,
                "text": n.get("content_md"),
                "hce_mean": n.get("mean_hce"),
                "hce_peak": n.get("peak_hce"),
            }
        )
    try:
        journals = storage.list_journal_entries(start_ts, end_ts, limit=2000)
    except Exception:
        journals = []
    for j in journals:
        st = j.get("start_ts") or j.get("ts")
        et = j.get("end_ts") or st
        if not st:
            continue
        if st < start_ts or st > end_ts:
            continue
        combined.append(
            {
                "type": "journal",
                "category": "Journal",
                "start_time": st,
                "end_time": et,
                "duration": max(0, (et or st) - st) or 60,
                "text": j.get("text"),
                "hce_mean": j.get("mean_hce"),
                "hce_peak": j.get("peak_hce"),
            }
        )
    if include_running and include_running.get("running"):
        now = time.time()
        st = include_running.get("start_time", now)
        combined.append(
            {
                "type": "live",
                "category": include_running.get("category") or "Live",
                "start_time": st,
                "end_time": now,
                "duration": max(0, now - st),
                "text": include_running.get("desc", ""),
                "hce_mean": None,
                "hce_peak": None,
            }
        )
    combined.sort(key=lambda e: e.get("start_time") or 0)
    return combined


def _continuum_context(window_days: int = 7) -> str:
    now = time.time()
    ts0 = now - window_days * 86400
    entries = _combined_entries(ts0, now)
    if not entries:
        return "No Continuum entries in window."
    df = pd.DataFrame(entries)
    lines = [f"Window: last {window_days} days; total entries: {len(df)}"]
    if "type" in df:
        counts = df["type"].value_counts()
        type_txt = ", ".join([f"{t}={int(v)}" for t, v in counts.items()])
        lines.append(f"Counts: {type_txt}")
    if "duration" in df:
        dur_h = df["duration"].fillna(0).sum() / 3600.0
        lines.append(f"Total logged duration: {dur_h:.2f} hours")
    # Time entries metrics
    hce_df = df[(df["type"] == "time") & df["hce_mean"].notna()] if "type" in df else df[df["hce_mean"].notna()]
    if not hce_df.empty:
        cat_means = hce_df.groupby("category")["hce_mean"].mean().sort_values(ascending=False)
        top_cat = cat_means.head(3)
        if not top_cat.empty:
            lines.append("Top categories by mean HCE: " + "; ".join([f"{c}={v:.2f}" for c, v in top_cat.items()]))
        lines.append(f"Global mean HCE (time entries): {hce_df['hce_mean'].mean():.2f}")
    note_count = int((df["type"] == "note").sum()) if "type" in df else 0
    journal_count = int((df["type"] == "journal").sum()) if "type" in df else 0
    if note_count or journal_count:
        lines.append(f"Notes: {note_count}, Journals: {journal_count}")
    return "\n".join(lines)


def _recent_notes_journals(window_days: int = 7, limit: int = 10) -> str:
    now = time.time()
    ts0 = now - window_days * 86400
    try:
        notes = storage.list_continuum_notes(limit=500)
    except Exception:
        notes = []
    try:
        journals = storage.list_journal_entries(ts0, now, limit=500)
    except Exception:
        journals = []
    note_lines = []
    for n in notes:
        ts = n.get("ts") or 0
        if ts < ts0 or ts > now:
            continue
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
        mood = n.get("mood")
        text = n.get("content_md", "") or ""
        note_lines.append(f"[Note {when}] mood={mood} :: {text}")
    jour_lines = []
    for j in journals:
        ts = j.get("ts") or 0
        if ts < ts0 or ts > now:
            continue
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
        mood = j.get("mood")
        text = j.get("text", "") or ""
        jour_lines.append(f"[Journal {when}] mood={mood} :: {text}")
    note_lines = note_lines[:limit]
    jour_lines = jour_lines[:limit]
    out = []
    if note_lines:
        out.append("Recent Notes:\n" + "\n".join(note_lines))
    if jour_lines:
        out.append("Recent Journals:\n" + "\n".join(jour_lines))
    return "\n\n".join(out) if out else "No recent notes/journals."


def _eeg_summary_window(window_sec: float = 86400.0) -> str:
    now = time.time()
    ts0 = now - window_sec
    try:
        rows = storage.list_state_summary(ts0, now, limit=20000)
    except Exception:
        rows = []
    if not rows:
        return "EEG summary: no state_summary rows in window."
    df = pd.DataFrame(rows)
    parts = []
    for col in ["hce", "q", "x"]:
        if col in df and df[col].notna().any():
            parts.append(f"{col.upper()} mean={df[col].mean():.2f}, peak={df[col].max():.2f}")
    if not parts:
        return "EEG summary: data present but no HCE/Q/X columns."
    return "EEG summary (last 24h): " + "; ".join(parts)


def _entry_eeg_summary(start_ts: float, end_ts: float) -> str:
    try:
        rows = storage.list_state_summary(start_ts, end_ts, limit=20000)
    except Exception:
        rows = []
    if not rows:
        return "EEG: none in window."
    df = pd.DataFrame(rows)
    parts = []
    for col in ["hce", "q", "x"]:
        if col in df and df[col].notna().any():
            parts.append(f"{col.upper()} mean={df[col].mean():.2f}, peak={df[col].max():.2f}")
    return "EEG: " + "; ".join(parts) if parts else "EEG: data present but no HCE/Q/X columns."


def _today_cards(entries: List[Dict[str, object]]) -> html.Div:
    if not entries:
        return html.Div("No entries today.", className="text-muted")
    cards = []
    for e in entries:
        cat = e.get("category") or "Unknown"
        color = _category_color(cat)
        dur = _format_duration(e.get("duration") or 0)
        title = e.get("spotify_title") or ""
        artist = e.get("spotify_artist") or ""
        spotify_txt = f"{title} — {artist}" if title or artist else ""
        body = [
            html.Div(cat, className="fw-bold", style={"color": color}),
            html.Div(f"Duration: {dur}", className="small"),
            html.Div(f"HCE mean/peak: {e.get('hce_mean')}/{e.get('hce_peak')}", className="small"),
            html.Div(f"Q mean: {e.get('q_mean')}", className="small"),
            html.Div(f"Reliability: {e.get('reliability_mean')}", className="small"),
            html.Div(spotify_txt, className="small text-info") if spotify_txt else None,
            html.Div(e.get("description", ""), className="small text-muted"),
        ]
        cards.append(dbc.Card(dbc.CardBody([c for c in body if c is not None]), className="mb-2", style={"borderLeft": f"4px solid {color}"}))
    return html.Div(cards)


layout = dbc.Container(
    [
        html.H1("Continuum Tracker – Engineer Your Flow", style={"textAlign": "center", "color": "#00ff9d"}),
        html.Div(id="ct-hce-badge", className="text-center mb-3"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            dbc.Button(
                                "Start Tracking",
                                id="timer-button",
                                color="secondary",
                                className="mb-2",
                                style={
                                    "width": "220px",
                                    "height": "220px",
                                    "borderRadius": "50%",
                                    "fontSize": "22px",
                                    "border": "3px solid #00ff9d",
                                    "boxShadow": "0 0 16px #00ff9d",
                                },
                            ),
                            className="d-flex justify-content-center",
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Category"),
                                dcc.Dropdown(
                                    id="ct-category",
                                    options=[{"label": k, "value": k} for k in CATEGORY_COLORS.keys()],
                                    value="Deep Work",
                                    style={"width": "100%"},
                                ),
                            ],
                            className="mb-2",
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Tags"),
                                dcc.Dropdown(
                                    id="ct-tags",
                                    options=[
                                        {"label": "#solitude", "value": "solitude"},
                                        {"label": "#high-energy", "value": "high-energy"},
                                        {"label": "#with-others", "value": "with-others"},
                                        {"label": "#low-light", "value": "low-light"},
                                    ],
                                    multi=True,
                                    placeholder="Add tags",
                                    style={"width": "100%"},
                                ),
                            ],
                            className="mb-2",
                        ),
                        dbc.Textarea(id="ct-desc", placeholder="Description", className="mb-2"),
                        dbc.RadioItems(
                            id="ct-mode",
                            options=[{"label": "Live Timer", "value": "live"}, {"label": "Manual Entry", "value": "manual"}],
                            value="live",
                            inline=True,
                            className="mb-2",
                        ),
                        dbc.Button("Quick Capture Flow", id="ct-quick", color="info", className="mb-3"),
                        dcc.Store(id="ct-state", storage_type="session"),
                        dcc.Interval(id="ct-interval", interval=1000, n_intervals=0),
                    ],
                    md=6,
                ),
            ],
            className="mb-4",
        ),
        html.Hr(),
        html.Div(html.H4("Today's Entries")),
        html.Div(id="ct-list"),
        html.Hr(),
        html.Div(html.H4("Continuum Notes")),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button("Add Note", id="ct-note-add", color="success", className="mb-2"),
                        dbc.Textarea(id="ct-note-text", placeholder="Markdown note...", className="mb-2"),
                        dcc.Store(id="ct-note-start-ts"),
                        dbc.Checklist(
                            id="ct-note-links",
                            options=[],
                            value=[],
                            inline=False,
                            className="mb-2",
                            switch=True,
                        ),
                        html.Div(
                            [
                                html.Div("Mood (1-5)", className="small text-muted"),
                                dcc.Slider(id="ct-note-mood", min=1, max=5, step=1, value=3, marks=None),
                            ],
                            className="mb-2",
                        ),
                        dbc.Button("Save Note", id="ct-note-save", color="warning", className="mb-3"),
                        # Infinite canvas placeholder (embedded tldraw)
                        html.Div(
                            html.Iframe(
                                src="https://tldraw.com",
                                style={"width": "100%", "height": "50vh", "border": "1px solid #222", "borderRadius": "8px"},
                            ),
                            className="mb-2",
                        ),
                        html.Div(id="ct-note-grid"),
                    ],
                    md=8,
                    style={"maxHeight": "60vh", "overflowY": "auto"},
                ),
                dbc.Col(
                    [
                        html.Div(html.H5("Hyper-Notes"), className="mb-2"),
                        html.Div(id="ct-note-detail", className="mb-3"),
                        dbc.Button("Reflect/Synthesize", id="ct-note-oracle-btn", color="secondary"),
                        dbc.Checklist(
                            id="ct-ambient-toggle",
                            options=[{"label": "Background Activity Logging", "value": "on"}],
                            value=[],
                            switch=True,
                            className="mt-2 mb-2",
                        ),
                        html.Div(id="ct-ambient-status", className="small text-muted mb-2"),
                        dbc.Modal(
                            [
                                dbc.ModalHeader("Mini-Oracle"),
                                dbc.ModalBody(
                                    [
                                        dcc.Textarea(
                                            id="ct-note-oracle-input",
                                            placeholder="You are mini-Oracle GPT in Continuum mode. Assist in ideation, clarification, thought evolution.",
                                            style={"width": "100%", "height": "120px"},
                                        ),
                                        html.Div(id="ct-note-oracle-output", className="mt-2"),
                                    ]
                                ),
                                dbc.ModalFooter(dbc.Button("Close", id="ct-note-oracle-close", className="ms-auto")),
                            ],
                            id="ct-note-oracle-modal",
                            is_open=False,
                        ),
                    ],
                    md=4,
                    style={"maxHeight": "60vh", "overflowY": "auto"},
                ),
            ],
            className="mb-4",
        ),
        html.Hr(),
        html.Div(html.H4("Daily Timelines & Reports")),
        dbc.Row(
            [
            dbc.Col(dcc.Graph(id="ct-timeline", style={"height": "45vh", "marginBottom": "20px"}), md=12),
            ],
        className="mb-4",
        ),
        dbc.Row(
            [
            dbc.Col(dcc.Graph(id="ct-pie", style={"height": "32vh", "marginBottom": "16px"}), md=4),
            dbc.Col(dcc.Graph(id="ct-bar", style={"height": "32vh", "marginBottom": "16px"}), md=4),
            dbc.Col(html.Div(id="ct-top-flow"), md=4),
            ],
        className="mb-4",
        ),
    html.Div(id="ct-pattern", className="text-info mb-4", style={"marginTop": "8px"}),
        dbc.Button("Analyze Patterns", id="ct-oracle-btn", color="secondary", className="mb-3"),
        dbc.Modal(
            [
                dbc.ModalHeader("Pattern Analysis"),
                dbc.ModalBody(id="ct-oracle-body"),
                dbc.ModalFooter(dbc.Button("Close", id="ct-oracle-close", className="ms-auto")),
            ],
            id="ct-oracle-modal",
            is_open=False,
        ),
        html.Hr(),
        html.Div(html.H4("Real-Time Journal")),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Textarea(id="ct-journal-text", placeholder="Log a moment...", className="mb-2"),
                dcc.Store(id="ct-journal-start-ts"),
                        html.Div(
                            [
                                html.Div("Mood (1-5)", className="small text-muted"),
                                dcc.Slider(id="ct-journal-mood", min=1, max=5, step=1, value=3, marks=None),
                            ],
                            className="mb-2",
                        ),
                        html.Div(
                            [
                                html.Div("Energy (1-5)", className="small text-muted"),
                                dcc.Slider(id="ct-journal-energy", min=1, max=5, step=1, value=3, marks=None),
                            ],
                            className="mb-2",
                        ),
                        dbc.Button("Add Journal Entry", id="ct-journal-submit", color="warning", className="mb-3"),
                    ],
                    md=6,
                ),
            ],
            className="mb-3",
        ),
        html.Div(id="ct-journal-feed"),
        dcc.Interval(id="ct-report-interval", interval=60000, n_intervals=0),
        html.Hr(),
        html.Div(html.H4("Agenda Calendar")),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.RadioItems(
                            id="ct-calendar-view",
                            options=[
                                {"label": "Daily", "value": "daily"},
                                {"label": "Weekly", "value": "weekly"},
                                {"label": "Monthly", "value": "monthly"},
                            ],
                            value="daily",
                            inline=True,
                            className="mb-2",
                        ),
                        dcc.DatePickerRange(
                            id="ct-calendar-range",
                            start_date=str(DEFAULT_START.date()),
                            end_date=str(TODAY.date()),
                        ),
                    ],
                    md=12,
                    className="mb-2",
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="ct-calendar", style={"height": "48vh"}), md=12),
            ],
            className="mb-3",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Entry Details"),
                dbc.ModalBody(id="ct-calendar-modal-body"),
                dbc.ModalFooter(dbc.Button("Close", id="ct-calendar-modal-close", className="ms-auto")),
            ],
            id="ct-calendar-modal",
            is_open=False,
        ),
    ],
    fluid=True,
    className="page-container",
)


@callback(Output("ct-hce-badge", "children"), Input("ct-interval", "n_intervals"))
def _update_hce_badge(_n):
    return _live_hce_badge()


@callback(
    Output("ct-state", "data"),
    Output("timer-button", "children"),
    Output("timer-button", "color"),
    Input("timer-button", "n_clicks"),
    Input("ct-quick", "n_clicks"),
    Input("ct-interval", "n_intervals"),
    State("ct-state", "data"),
    State("ct-category", "value"),
    State("ct-tags", "value"),
    State("ct-desc", "value"),
)
def handle_timer(clicks, quick_clicks, _tick, state, category, tags, desc):
    state = state or {}
    running = state.get("running", False)
    start_time = state.get("start_time")
    if dash.callback_context.triggered:
        trig = dash.callback_context.triggered[0]["prop_id"]
    else:
        trig = ""
    if trig.startswith("ct-quick"):
        now = time.time()
        start = now - 60
        metrics = _aggregate_metrics(start, now)
        try:
            storage.insert_time_entry(
                start_time=start,
                end_time=now,
                duration=60,
                category=category or "Quick Capture",
                tags_json=json.dumps(tags or []),
                description=desc or "",
                hce_mean=metrics.get("hce_mean"),
                hce_peak=metrics.get("hce_peak"),
                mean_x=metrics.get("mean_x"),
                peak_x=metrics.get("peak_x"),
                peak_q=metrics.get("peak_q"),
                q_mean=metrics.get("q_mean"),
                reliability_mean=metrics.get("reliability_mean"),
                spotify_track_uri="",
                spotify_title="",
                spotify_artist="",
            )
        except Exception:
            pass
        return state, "Captured", "info"
    if trig.startswith("timer-button"):
        if not running:
            # start
            state = {
                "running": True,
                "start_time": time.time(),
                "category": category,
                "tags": tags or [],
                "desc": desc or "",
            }
        else:
            # stop
            end_time = time.time()
            start = state.get("start_time")
            duration = max(0.0, end_time - start) if start else 0.0
            metrics = _aggregate_metrics(start, end_time) if start else {}
            # Attempt spotify linkage
            spotify_uri = spotify_title = spotify_artist = ""
            try:
                latest = storage.get_latest_spotify(session_id=None)
                if latest and latest.get("track_id"):
                    spotify_uri = latest.get("track_id") or ""
                    spotify_title = latest.get("track_name") or ""
                    spotify_artist = latest.get("artists") or ""
            except Exception:
                pass
            try:
                storage.insert_time_entry(
                    start_time=start or end_time,
                    end_time=end_time,
                    duration=duration,
                    category=state.get("category") or "",
                    tags_json=json.dumps(state.get("tags") or []),
                    description=state.get("desc") or "",
                    hce_mean=metrics.get("hce_mean"),
                    hce_peak=metrics.get("hce_peak"),
                    mean_x=metrics.get("mean_x"),
                    peak_x=metrics.get("peak_x"),
                    peak_q=metrics.get("peak_q"),
                    q_mean=metrics.get("q_mean"),
                    reliability_mean=metrics.get("reliability_mean"),
                    spotify_track_uri=spotify_uri,
                    spotify_title=spotify_title,
                    spotify_artist=spotify_artist,
                )
            except Exception:
                pass
            state = {"running": False}

    running = state.get("running", False)
    start_time = state.get("start_time")
    label = "Start Tracking"
    if running and start_time:
        elapsed = time.time() - float(start_time)
        label = _format_duration(elapsed)
    color = "success" if running else "secondary"
    return state, label, color


@callback(Output("ct-list", "children"), Input("ct-interval", "n_intervals"))
def update_list(_n):
    try:
        rows = storage.list_today_time_entries()
    except Exception:
        rows = []
    return _today_cards(rows)


def _timeline_fig(entries: List[Dict[str, object]]) -> go.Figure:
    fig = go.Figure()
    if not entries:
        fig.add_annotation(text="No entries today", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", height=400)
        return fig
    for e in entries:
        start = e.get("start_time") or 0
        end = e.get("end_time") or start
        dur = max(0, (end - start))
        cat = e.get("category") or "Unknown"
        color = _type_color(e.get("type") or "time")
        fig.add_trace(
            go.Bar(
                x=[dur],
                y=[cat],
                base=[start],
                orientation="h",
                marker=dict(color=color),
                hovertemplate=f"{cat}<br>Dur={_format_duration(dur)}<extra></extra>",
                name=cat,
                showlegend=False,
            )
        )
        # HCE label
        hce = e.get("hce_mean")
        if hce is not None:
            fig.add_annotation(
                x=start + dur / 2.0,
                y=cat,
                text=f"HCE {hce:.2f}",
                showarrow=False,
                font=dict(color="#ffffff", size=10),
            )
    fig.update_layout(
        template="plotly_dark",
        barmode="stack",
        height=400,
        xaxis_title="Seconds since epoch (today)",
        yaxis_title="Category",
        margin=dict(l=50, r=30, t=30, b=40),
    )
    return fig


def _history_lines(span_s: float = 7200.0) -> Dict[str, List[float]]:
    hist = registry.state_engine.get_history()
    t = hist.get("t", [])
    if not t:
        return {"t": [], "X": [], "Q": [], "HCE": []}
    t_max = t[-1]
    cutoff = t_max - span_s
    idx = [i for i, tv in enumerate(t) if tv >= cutoff]
    def take(key):
        arr = hist.get(key, [])
        return [arr[i] for i in idx if i < len(arr)]
    return {"t": take("t"), "X": take("X"), "Q": take("Q_abs_raw"), "HCE": take("HCE")}


def _reports(entries: List[Dict[str, object]]) -> (go.Figure, go.Figure, html.Div, str):
    if not entries:
        empty_fig = go.Figure().update_layout(template="plotly_dark", height=250)
        empty_fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig, html.Div("No entries", className="text-muted"), ""
    df = pd.DataFrame(entries)
    # Only use time entries for HCE summaries to avoid mixing notes/journal without metrics
    if "type" in df:
        df = df[df["type"] == "time"]
    if df.empty:
        empty_fig = go.Figure().update_layout(template="plotly_dark", height=250)
        empty_fig.add_annotation(text="No time entries", x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig, html.Div("No time entries", className="text-muted"), ""
    if "duration" not in df:
        df["duration"] = 0
    df["duration"] = df["duration"].fillna(0).clip(lower=0)
    df.loc[df["duration"] == 0, "duration"] = 60  # minimum for static entries
    df["category"] = df["category"].fillna("Unknown")
    pie = go.Figure(
        go.Pie(
            labels=df["category"],
            values=df["duration"],
            hole=0.4,
        )
    ).update_layout(template="plotly_dark", height=300, title="Time by Category")
    bar_df = df.groupby("category")["hce_mean"].mean().reset_index()
    bar = go.Figure(
        go.Bar(x=bar_df["category"], y=bar_df["hce_mean"], marker_color=[_category_color(c) for c in bar_df["category"]])
    ).update_layout(template="plotly_dark", height=300, title="Avg HCE by Category")
    top = df.sort_values("hce_peak", ascending=False).head(5)
    rows = []
    for _, r in top.iterrows():
        rows.append(
            html.Tr(
                [
                    html.Td(r.get("category")),
                    html.Td(_format_duration(r.get("duration") or 0)),
                    html.Td(f"{r.get('hce_mean'):.2f}" if pd.notna(r.get("hce_mean")) else "-"),
                    html.Td(f"{r.get('hce_peak'):.2f}" if pd.notna(r.get("hce_peak")) else "-"),
                ]
            )
        )
    table = dbc.Table(
        [html.Thead(html.Tr([html.Th("Category"), html.Th("Duration"), html.Th("HCE mean"), html.Th("HCE peak")])), html.Tbody(rows)],
        size="sm",
        striped=True,
        bordered=False,
        hover=True,
    )
    # Simple pattern text
    if not bar_df.empty:
        best_cat = bar_df.loc[bar_df["hce_mean"].idxmax()]
        pattern_txt = f"{best_cat['category']} shows highest mean HCE ({best_cat['hce_mean']:.2f})."
    else:
        pattern_txt = ""
    return pie, bar, table, pattern_txt


def _journal_cards(rows: List[Dict[str, object]]) -> html.Div:
    if not rows:
        return html.Div("No journal entries yet.", className="text-muted")
    cards = []
    for r in rows:
        ts = r.get("ts") or 0
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
        mood = r.get("mood")
        energy = r.get("energy")
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(when, className="small text-muted"),
                        html.Div(r.get("text", ""), className="mb-1"),
                        html.Div(f"Mood: {mood} | Energy: {energy}", className="small text-info"),
                    ]
                ),
                className="mb-2",
            )
        )
    return html.Div(cards)


def _calendar_fig(entries: List[Dict[str, object]], view: str, start_date: Optional[str], end_date: Optional[str], running: Dict[str, object]) -> go.Figure:
    fig = go.Figure()
    if not start_date or not end_date:
        today = pd.Timestamp.today().normalize()
        start_date = end_date = str(today.date())
    try:
        start_ts = pd.Timestamp(start_date).timestamp()
        end_ts = pd.Timestamp(end_date).timestamp() + 86399
    except Exception:
        start_ts = end_ts = time.time()

    # filter entries
    filtered = [e for e in entries if (e.get("start_time") or 0) >= start_ts and (e.get("start_time") or 0) <= end_ts]
    # also include notes/journals with ts if start_time missing
    for e in entries:
        if e.get("start_time") is None and e.get("ts"):
            if e["ts"] >= start_ts and e["ts"] <= end_ts:
                filtered.append(
                    {
                        **e,
                        "start_time": e["ts"],
                        "end_time": e.get("end_ts") or e.get("ts"),
                        "type": e.get("type") or ("note" if e.get("content_md") else "journal"),
                        "category": e.get("category") or ("Note" if e.get("content_md") else "Journal"),
                    }
                )
    if running and running.get("running"):
        now = time.time()
        filtered.append(
            {
                "start_time": running.get("start_time", now),
                "end_time": now,
                "category": running.get("category", "Live"),
                "hce_mean": None,
            }
        )
    if not filtered:
        fig.add_annotation(text="No entries in range", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", height=400)
        return fig
    filtered = sorted(filtered, key=lambda e: e.get("start_time") or 0)
    for e in filtered:
        start = e.get("start_time") or 0
        end = e.get("end_time") or start
        dur = max(0, end - start)
        cat = e.get("category") or "Unknown"
        color = _type_color(e.get("type") or "time")
        fig.add_trace(
            go.Bar(
                x=[dur],
                y=[time.strftime("%Y-%m-%d", time.localtime(start))],
                base=[start],
                orientation="h",
                marker=dict(color=color),
                name=cat,
                hovertemplate=f"{cat}<br>{time.strftime('%Y-%m-%d %H:%M', time.localtime(start))}<br>Dur={_format_duration(dur)}<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        template="plotly_dark",
        barmode="stack",
        height=420,
        xaxis_title="Time",
        yaxis_title="Date",
        margin=dict(l=50, r=30, t=30, b=40),
    )
    return fig


@callback(
    Output("ct-timeline", "figure"),
    Output("ct-pie", "figure"),
    Output("ct-bar", "figure"),
    Output("ct-top-flow", "children"),
    Output("ct-pattern", "children"),
    Output("ct-calendar", "figure"),
    Input("ct-report-interval", "n_intervals"),
    Input("ct-calendar-view", "value"),
    Input("ct-calendar-range", "start_date"),
    Input("ct-calendar-range", "end_date"),
    State("ct-state", "data"),
)
def update_reports(_n, view, start_date, end_date, state):
    if not start_date or not end_date:
        today = pd.Timestamp.today().normalize()
        start_date = end_date = str(today.date())
    try:
        ts0 = pd.Timestamp(start_date).timestamp()
        ts1 = pd.Timestamp(end_date).timestamp() + 86399
    except Exception:
        ts0 = time.time() - 86400
        ts1 = time.time()
    entries = _combined_entries(ts0, ts1, include_running=state or {})
    # restrict timeline to the chosen window (already filtered)
    pie, bar, table, pattern = _reports(entries)
    calendar_fig = _calendar_fig(entries, view or "daily", start_date, end_date, state or {})
    return _timeline_fig(entries), pie, bar, table, pattern, calendar_fig


@callback(
    Output("ct-calendar-modal", "is_open"),
    Output("ct-calendar-modal-body", "children"),
    Input("ct-calendar", "clickData"),
    Input("ct-calendar-modal-close", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_calendar_modal(clickData, close_clicks):
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    if triggered == "ct-calendar-modal-close":
        return False, dash.no_update
    if clickData and clickData.get("points"):
        pt = clickData["points"][0]
        hover = pt.get("hovertext") or ""
        return True, html.Div([html.Div("Entry"), html.Pre(hover or "No details"), html.Div("Edit placeholder")])
    return dash.no_update, dash.no_update


@callback(
    Output("ct-oracle-modal", "is_open"),
    Output("ct-oracle-body", "children"),
    Input("ct-oracle-btn", "n_clicks"),
    Input("ct-oracle-close", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_oracle_modal(open_click, close_click):
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    if triggered == "ct-oracle-close":
        return False, dash.no_update
    if triggered == "ct-oracle-btn":
        context_txt = _continuum_context(window_days=7)
        notes_block = _recent_notes_journals(window_days=7, limit=12)
        eeg_block = _eeg_summary_window(window_sec=7 * 86400)
        prompt = (
            "You are mini-Oracle GPT in Continuum mode. Analyze patterns from the Continuum Tracker "
            "to surface wow-factor insights: flow blocks (time entries), notes, journals, and EEG HCE/Q/X when present. "
            "Ground your answer strictly in the provided metrics; do not invent values or units. "
            "HCE is an EEG-derived metric (not hours). "
            "Report counts, durations (in hours), top categories by mean HCE, and actionable next steps. "
            "If data is sparse or missing (e.g., no EEG), state that and advise what to log next.\n\n"
            f"Data window (last 7 days):\n{context_txt}\n\n{notes_block}\n\n{eeg_block}"
        )
        reply = _ollama_call(prompt)
        return True, html.Div(reply)
    return dash.no_update, dash.no_update


@callback(
    Output("ct-journal-feed", "children"),
    Input("ct-report-interval", "n_intervals"),
    Input("ct-journal-submit", "n_clicks"),
    State("ct-journal-text", "value"),
    State("ct-journal-mood", "value"),
    State("ct-journal-energy", "value"),
    State("ct-journal-start-ts", "data"),
)
def update_journal(_n, submit, text, mood, energy, start_ts):
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    if triggered == "ct-journal-submit" and (text or "").strip():
        try:
            s_ts = start_ts or time.time()
            e_ts = time.time()
            metrics = _aggregate_metrics(s_ts, e_ts)
            storage.insert_journal_entry(
                ts=e_ts,
                text=text or "",
                mood=mood,
                energy=energy,
                tags_json="[]",
                source="continuum",
                start_ts=s_ts,
                end_ts=e_ts,
                mean_x=metrics.get("mean_x"),
                mean_q=metrics.get("q_mean"),
                mean_hce=metrics.get("hce_mean"),
                peak_x=metrics.get("peak_x"),
                peak_q=metrics.get("peak_q"),
                peak_hce=metrics.get("hce_peak"),
            )
        except Exception:
            pass
        return _journal_cards(storage.list_journal_entries(time.time() - 86400 * 30, time.time(), limit=500))
    try:
        now = time.time()
        rows = storage.list_journal_entries(now - 86400 * 30, now, limit=500)
    except Exception:
        rows = []
    return _journal_cards(rows)


def _note_cards(notes: List[Dict[str, object]]) -> html.Div:
    if not notes:
        return html.Div("No notes yet.", className="text-muted")
    cards = []
    for n in notes:
        ts = n.get("ts") or 0
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
        cat = n.get("category") or "Note"
        color = _category_color(cat)
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(when, className="small text-muted"),
                        html.Div(cat, className="fw-bold", style={"color": color}),
                        dcc.Markdown(n.get("content_md", "") or ""),
                        html.Div(f"Mood: {n.get('mood')}", className="small text-info"),
                        dbc.Button("Select", id={"type": "ct-note-select", "id": n.get("id")}, color="link", size="sm"),
                    ]
                ),
                style={"borderLeft": f"4px solid {color}"},
                className="mb-2",
            )
        )
    return html.Div(cards)


@callback(
    Output("ct-note-grid", "children"),
    Output("ct-note-links", "options"),
    Input("ct-report-interval", "n_intervals"),
)
def refresh_notes(_n):
    try:
        notes = storage.list_continuum_notes(limit=200)
    except Exception:
        notes = []
    opts = [{"label": n.get("content_md", "")[:30] + "...", "value": n.get("id")} for n in notes if n.get("id") is not None]
    return _note_cards(notes), opts


@callback(Output("ct-note-start-ts", "data"), Input("ct-note-text", "value"), prevent_initial_call=True)
def set_note_start(_val):
    return time.time()


@callback(Output("ct-journal-start-ts", "data"), Input("ct-journal-text", "value"), prevent_initial_call=True)
def set_journal_start(_val):
    return time.time()


@callback(
    Output("ct-note-detail", "children"),
    Input({"type": "ct-note-select", "id": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_note(n_clicks):
    if not dash.callback_context.triggered:
        raise dash.exceptions.PreventUpdate
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    try:
        trig_id = json.loads(trig)
        note_id = trig_id.get("id")
    except Exception:
        note_id = None
    if not note_id:
        raise dash.exceptions.PreventUpdate
    try:
        notes = storage.list_continuum_notes(limit=200)
        note = next((n for n in notes if n.get("id") == note_id), None)
    except Exception:
        note = None
    if not note:
        return html.Div("Not found", className="text-muted")
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(time.strftime("%Y-%m-%d %H:%M", time.localtime(note.get("ts") or 0)), className="small text-muted"),
                dcc.Markdown(note.get("content_md", "")),
                html.Div(f"Mood: {note.get('mood')}", className="small text-info"),
                html.Div(f"Links: {note.get('links_json')}", className="small text-muted"),
            ]
        )
    )


@callback(
    Output("ct-note-text", "value"),
    Output("ct-note-links", "value"),
    Input("ct-note-save", "n_clicks"),
    State("ct-note-text", "value"),
    State("ct-note-links", "value"),
    State("ct-note-mood", "value"),
    State("ct-category", "value"),
)
def save_note(n_clicks, text, links, mood, category):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    ts = time.time()
    # Snapshot EEG + spotify
    snap = registry.state_engine.get_status().get("latest_snapshot") or {}
    eeg_json = json.dumps(
        {
            "X": snap.get("X"),
            "Q_abs_raw": snap.get("Q_abs_raw"),
            "HCE": snap.get("HCE"),
            "ts": snap.get("ts_unix"),
        }
    )
    spotify_title = spotify_artist = ""
    try:
        latest = storage.get_latest_spotify(session_id=None)
        if latest and latest.get("track_name"):
            spotify_title = latest.get("track_name") or ""
            spotify_artist = latest.get("artists") or ""
    except Exception:
        pass
    # EEG window stats (use note start ts if available)
    start_store = dash.callback_context.states.get("ct-note-start-ts.data") if dash.callback_context.states else None
    start_ts = None
    try:
        start_ts = float(start_store) if start_store else None
    except Exception:
        start_ts = None
    if not start_ts:
        start_ts = ts
    metrics = _aggregate_metrics(start_ts, ts)
    try:
        storage.insert_continuum_note(
            ts=ts,
            content_md=text or "",
            pos_x=0.0,
            pos_y=0.0,
            color=_category_color(category or ""),
            links_json=json.dumps(links or []),
            eeg_snapshot_json=eeg_json,
            mood=mood,
            category=category or "",
            spotify_title=spotify_title,
            spotify_artist=spotify_artist,
            tags_json="[]",
            start_ts=start_ts,
            end_ts=ts,
            mean_x=metrics.get("mean_x"),
            mean_q=metrics.get("q_mean"),
            mean_hce=metrics.get("hce_mean"),
            peak_x=metrics.get("peak_x"),
            peak_q=metrics.get("peak_q"),
            peak_hce=metrics.get("hce_peak"),
        )
    except Exception:
        pass
    return "", []


@callback(
    Output("ct-note-oracle-modal", "is_open"),
    Output("ct-note-oracle-output", "children"),
    Input("ct-note-oracle-btn", "n_clicks"),
    Input("ct-note-oracle-close", "n_clicks"),
    State("ct-note-oracle-input", "value"),
    prevent_initial_call=True,
)
def toggle_note_oracle(open_click, close_click, user_txt):
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    if triggered == "ct-note-oracle-close":
        return False, dash.no_update
    if triggered == "ct-note-oracle-btn":
        now = time.time()
        window_start = now - 86400
        entries = _combined_entries(window_start, now)
        df = pd.DataFrame(entries)
        summary_parts = []
        if not df.empty:
            summary_parts.append(f"Entries last 24h: {len(df)}; time={int((df['type']=='time').sum())}, notes={int((df['type']=='note').sum())}, journal={int((df['type']=='journal').sum())}.")
            if "hce_mean" in df and df["hce_mean"].notna().any():
                try:
                    top = df.sort_values("hce_mean", ascending=False).head(3)
                    for _, r in top.iterrows():
                        summary_parts.append(f"- {r.get('category','?')} mean HCE {r.get('hce_mean')}")
                except Exception:
                    pass
        notes_block = _recent_notes_journals(window_days=1, limit=12)
        eeg_block = _eeg_summary_window(window_sec=86400)
        ctx_txt = "\n".join(summary_parts) if summary_parts else "No logged entries in the last 24h."
        prompt = (
            "You are mini-Oracle GPT in Continuum mode. Assist in ideation, clarification, and thought evolution. "
            "Ground responses in the recent Continuum data (notes, journal, flow blocks, EEG HCE/Q/X patterns when present). "
            "HCE is an EEG-derived metric (not hours). "
            "Do not invent numbers; if data is sparse, say so and suggest what to log. "
            f"User input: {user_txt or 'No extra input provided.'}\n\nContext (last 24h):\n{ctx_txt}\n\n{notes_block}\n\n{eeg_block}"
        )
        reply = _ollama_call(prompt)
        return True, html.Div(reply)
    raise dash.exceptions.PreventUpdate


@callback(
    Output("ct-ambient-status", "children"),
    Input("ct-ambient-toggle", "value"),
)
def toggle_ambient_logging(val):
    try:
        from alma.utils import ambient_logger
    except Exception:
        return "Ambient logger unavailable (missing permissions or libs)."
    if val and "on" in val:
        try:
            ambient_logger.start()
            return "Ambient logging active (local-only; may need Accessibility permission)."
        except Exception:
            return "Failed to start ambient logging (check permissions)."
    else:
        try:
            ambient_logger.stop()
            return "Ambient logging off."
        except Exception:
            return "Ambient logging off."
