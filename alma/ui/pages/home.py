import json
import time
import datetime as dt
from pathlib import Path

from dash import callback, ctx, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from alma import config
from alma.app_state import registry
from alma.engine import storage


def _read_mode() -> str:
    try:
        with config.MODE_FILE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("mode", "OFF"))
    except Exception:
        return "OFF"

PROFILE_PATH = Path(__file__).resolve().parents[3] / "profiles" / "default.json"


def _load_profile():
    try:
        with PROFILE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _status_item(label: str, value_component, show_dot: bool = False) -> dbc.Col:
    content = (
        [
            html.Span(className="status-dot offline", id="home-lsl-dot"),
            value_component,
        ]
        if show_dot
        else [value_component]
    )
    return dbc.Col(
        [
            html.Div(label, className="status-label"),
            html.Div(content, className="status-value-wrapper" + (" has-dot" if show_dot else "")),
        ],
        xs=12,
        sm=6,
        md=4,
        lg=2,
        className="status-col",
    )


status_strip = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
                _status_item("LSL", html.Span("N/A", id="home-lsl-text"), show_dot=True),
                _status_item("Engine", html.Span("Stopped", id="home-engine-status")),
                _status_item("muselsl", html.Span("Stopped", id="home-muse-status")),
                _status_item("X", html.Span("N/A", id="home-x-text")),
                _status_item("Canonical Q", html.Span("N/A", id="home-q-text")),
                _status_item("HCE", html.Span("N/A", id="home-hce-text")),
                _status_item("Reliability (rolling last 600)", html.Span("N/A", id="home-reliability-text")),
                _status_item("Baseline", html.Span("N/A", id="home-baseline-text")),
                _status_item("Current label", html.Span("(coming soon)", id="home-label-text")),
                _status_item("Now Playing", html.Span("N/A", id="home-now-playing")),
                _status_item("Peak Track HCE", html.Span("—", id="home-peak-hce")),
            ],
            className="g-3 align-items-center status-row",
        )
    ),
    className="status-card",
)

controls_card = dbc.Card(
    [
        dbc.CardHeader("Controls"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("Muse address", className="status-label"),
                                dbc.Input(id="home-muse-address", placeholder="BLE address (e.g., 00:11:22:33:44:55)", size="sm"),
                            ],
                            md=5,
                            sm=12,
                        ),
                        dbc.Col(
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Start Muse Stream", id="home-start-muse-btn", color="primary"),
                                    dbc.Button("Stop Muse Stream", id="home-stop-muse-btn", color="secondary"),
                                ],
                                className="mt-3 mt-md-0",
                            ),
                            md=7,
                            sm=12,
                        ),
                    ],
                    className="g-2 mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Start Engine", id="home-start-engine-btn", color="success"),
                                    dbc.Button("Stop Engine", id="home-stop-engine-btn", color="warning"),
                                ]
                            ),
                            md=6,
                            sm=12,
                            className="mb-2 mb-md-0",
                        ),
                        dbc.Col(
                            dbc.Button("Toggle NDJSON Emit (Off)", id="home-toggle-ndjson-btn", color="info"),
                            md=6,
                            sm=12,
                        ),
                    ],
                    className="g-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.ButtonGroup(
                                [
                                    dbc.Button("Start Spotify Logging", id="home-start-spotify-btn", color="dark"),
                                    dbc.Button("Stop Spotify Logging", id="home-stop-spotify-btn", color="secondary"),
                                ]
                            ),
                            md=12,
                            sm=12,
                        )
                    ],
                    className="g-2 mt-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("Current track section", className="status-label"),
                                dbc.Progress(id="home-track-progress", striped=True, animated=True, value=0, max=100, color="warning", className="mb-1"),
                                html.Div(id="home-track-section", className="small text-muted"),
                            ],
                            md=12,
                            sm=12,
                        )
                    ],
                    className="g-2 mt-2",
                ),
                html.Div(id="home-muse-error", className="text-danger small mt-2"),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Event label"),
                                dbc.Input(id="home-event-label", value="moment", size="sm"),
                            ],
                            md=4,
                            sm=12,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Note (optional)"),
                                dbc.Input(id="home-event-note", placeholder="note", size="sm"),
                            ],
                            md=6,
                            sm=12,
                        ),
                        dbc.Col(
                            dbc.Button("Bookmark Now", id="home-bookmark-btn", color="primary", className="mt-4"),
                            md=2,
                            sm=12,
                        ),
                    ],
                    className="g-2 mb-3",
                ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Recent Events", className="fw-bold mb-2"),
                            html.Div(id="home-recent-events"),
                        ]
                    ),
                    className="page-card mt-2",
                ),
                dcc.Interval(id="home-status-interval", interval=1000, n_intervals=0),
            ]
        ),
    ],
    className="page-card",
)

# Top tracks analytics
top_tracks_card = dbc.Card(
    [
        dbc.CardHeader("HCE Top Tracks"),
        dbc.CardBody(
            [
                html.Div("Top 10 HCE Tracks", className="fw-bold mb-2"),
                html.Div(id="home-top10-tracks"),
                html.Hr(),
                html.Div("Top 100 HCE Tracks", className="fw-bold mb-2"),
                html.Div(id="home-top100-tracks", className="small"),
            ]
        ),
    ],
    className="page-card",
)

# Bookmark modal for richer note entry
bookmark_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Bookmark Now")),
        dbc.ModalBody(
            [
                dbc.Label("Label"),
                dbc.Input(id="home-bookmark-label", value="moment", placeholder="moment"),
                dbc.Label("Note", className="mt-2"),
                dbc.Textarea(id="home-bookmark-note", placeholder="What’s happening right now?"),
                html.Hr(),
                dbc.Checklist(
                    id="home-bookmark-social",
                    options=[{"label": "With others (unchecked = Alone)", "value": "with"}],
                    value=[],
                    switch=True,
                    className="mt-2",
                ),
                dbc.Input(id="home-bookmark-names", placeholder="Names / count (optional)", className="mt-2"),
                dbc.Label("Activity", className="mt-2"),
                dcc.Dropdown(
                    id="home-bookmark-activity",
                    options=[
                        {"label": x, "value": x.lower()}
                        for x in ["Work", "Creative", "Meditation", "Social", "Media", "Exercise"]
                    ],
                    value="work",
                    clearable=False,
                ),
                dbc.Label("Mood (1-10)", className="mt-3"),
                dcc.Slider(id="home-bookmark-mood", min=1, max=10, step=1, value=6, marks=None),
                dbc.Label("Energy (1-10)", className="mt-3"),
                dcc.Slider(id="home-bookmark-energy", min=1, max=10, step=1, value=6, marks=None),
                dbc.Label("Environment", className="mt-3"),
                dbc.RadioItems(
                    id="home-bookmark-lighting",
                    options=[{"label": "Natural", "value": "natural"}, {"label": "Artificial", "value": "artificial"}],
                    value="natural",
                    inline=True,
                ),
                dbc.RadioItems(
                    id="home-bookmark-noise",
                    options=[{"label": "Quiet", "value": "quiet"}, {"label": "Loud", "value": "loud"}],
                    value="quiet",
                    inline=True,
                ),
                dbc.Input(id="home-bookmark-location", placeholder="Location (optional)", className="mt-2"),
                dbc.Label("Intention", className="mt-2"),
                dbc.Textarea(id="home-bookmark-intention", placeholder="Pre-session goal / intention"),
            ]
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Cancel", id="home-bookmark-cancel", color="secondary", className="me-2"),
                dbc.Button("Save Bookmark", id="home-bookmark-save", color="primary"),
            ]
        ),
    ],
    id="home-bookmark-modal",
    is_open=False,
)

optimal_windows_card = dbc.Card(
    [
        dbc.CardHeader("Optimal Windows (HCE)"),
        dbc.CardBody(
            [
                html.Div("Today’s top HCE windows", className="fw-bold mb-2"),
                html.Div(id="home-optimal-windows", className="small"),
            ]
        ),
    ],
    className="page-card",
)

layout = dbc.Container(
    [
        status_strip,
        controls_card,
        optimal_windows_card,
        top_tracks_card,
        bookmark_modal,
    ],
    fluid=True,
    className="page-container",
)


@callback(
    Output("home-lsl-dot", "className"),
    Output("home-lsl-text", "children"),
    Output("home-engine-status", "children"),
    Output("home-muse-status", "children"),
    Output("home-muse-error", "children"),
    Output("home-reliability-text", "children"),
    Output("home-baseline-text", "children"),
    Output("home-x-text", "children"),
    Output("home-q-text", "children"),
    Output("home-hce-text", "children"),
    Output("home-label-text", "children"),
    Output("home-now-playing", "children"),
    Output("home-recent-events", "children"),
    Output("home-start-engine-btn", "disabled"),
    Output("home-stop-engine-btn", "disabled"),
    Output("home-start-muse-btn", "disabled"),
    Output("home-stop-muse-btn", "disabled"),
    Output("home-toggle-ndjson-btn", "children"),
    Output("home-toggle-ndjson-btn", "color"),
    Output("ndjson-emit-store", "data"),
    Output("home-muse-address", "value"),
    Output("home-track-progress", "value"),
    Output("home-track-section", "children"),
    Input("home-status-interval", "n_intervals"),
    Input("home-start-engine-btn", "n_clicks"),
    Input("home-stop-engine-btn", "n_clicks"),
    Input("home-start-muse-btn", "n_clicks"),
    Input("home-stop-muse-btn", "n_clicks"),
    Input("home-toggle-ndjson-btn", "n_clicks"),
    Input("home-start-spotify-btn", "n_clicks"),
    Input("home-stop-spotify-btn", "n_clicks"),
    Input("home-bookmark-btn", "n_clicks"),
    Input("home-bookmark-save", "n_clicks"),
    State("home-muse-address", "value"),
    State("ndjson-emit-store", "data"),
    State("home-event-label", "value"),
    State("home-event-note", "value"),
    State("home-bookmark-label", "value"),
    State("home-bookmark-note", "value"),
    State("home-bookmark-social", "value"),
    State("home-bookmark-names", "value"),
    State("home-bookmark-activity", "value"),
    State("home-bookmark-mood", "value"),
    State("home-bookmark-energy", "value"),
    State("home-bookmark-lighting", "value"),
    State("home-bookmark-noise", "value"),
    State("home-bookmark-location", "value"),
    State("home-bookmark-intention", "value"),
)
def update_home_status(
    _interval,
    start_engine_clicks,
    stop_engine_clicks,
    start_muse_clicks,
    stop_muse_clicks,
    toggle_ndjson_clicks,
    start_spotify_clicks,
    stop_spotify_clicks,
    bookmark_clicks,
    bookmark_save_clicks,
    muse_address,
    ndjson_enabled,
    event_label,
    event_note,
    modal_label,
    modal_note,
    social_vals,
    names,
    activity,
    mood,
    energy,
    lighting,
    noise,
    location,
    intention,
):
    triggered = ctx.triggered_id
    muse_error = ""

    profile = _load_profile()
    profile_address = profile.get("muse_address", "")

    engine = registry.state_engine
    muse = registry.muse_stream_manager
    spotify_logger = registry.spotify_logger
    session_id = engine.get_session_id()
    engine_status = engine.get_status()
    latest_for_events = engine_status.get("latest_snapshot") if isinstance(engine_status.get("latest_snapshot"), dict) else {}

    ndjson_state = bool(ndjson_enabled)

    # Auto-start Spotify logging once session_id is available, then force an immediate poll
    try:
        if session_id and not (spotify_logger.status().get("running")):
            spotify_logger.start(session_id=session_id)
            spotify_logger.force_poll()
        # If running but no latest yet, nudge a poll
        if session_id and spotify_logger.status().get("running") and not spotify_logger.status().get("latest"):
            spotify_logger.force_poll()
    except Exception:
        pass

    if triggered == "home-start-engine-btn":
        try:
            engine.start()
        except Exception as exc:  # defensive
            muse_error = f"Engine start error: {exc}"
    elif triggered == "home-stop-engine-btn":
        try:
            engine.stop()
        except Exception as exc:  # defensive
            muse_error = f"Engine stop error: {exc}"
    elif triggered == "home-start-muse-btn":
        addr = (muse_address or profile_address or "").strip()
        if addr:
            muse.start(addr)
        else:
            muse_error = "Muse address required to start stream."
    elif triggered == "home-stop-muse-btn":
        muse.stop()
    elif triggered == "home-toggle-ndjson-btn":
        ndjson_state = not ndjson_state
        engine.set_emit_ndjson(ndjson_state)
    elif triggered == "home-start-spotify-btn":
        session_id = engine.get_session_id()
        if session_id:
            spotify_logger.start(session_id=session_id)
            spotify_logger.force_poll()
    elif triggered == "home-stop-spotify-btn":
        spotify_logger.stop()
    elif triggered == "home-bookmark-btn":
        # Open modal handled in separate callback; no state change here
        pass
    elif triggered == "home-bookmark-save":
        ts_now = time.time()
        latest = latest_for_events or {}
        bucket = None
        try:
            buckets = storage.get_buckets_between(ts_now - 600, ts_now + 1, session_id=session_id) if session_id else []
            bucket = buckets[-1] if buckets else None
        except Exception:
            bucket = None
        now_play_latest = spotify_logger.status().get("latest") or {}
        try:
            storage.insert_event(
                ts=ts_now,
                session_id=session_id or "",
                kind="manual",
                label=(modal_label or event_label or "moment"),
                note=(modal_note or event_note or ""),
                tags_json={
                    "page": "home",
                    "social": "with" if ("with" in (social_vals or [])) else "alone",
                    "names": names or "",
                    "activity": activity or "",
                    "mood": mood,
                    "energy": energy,
                    "lighting": lighting or "",
                    "noise": noise or "",
                    "location": location or "",
                },
                context_json={
                    "snapshot": {
                        "X": latest.get("X"),
                        "Q_vibe_focus": latest.get("Q_vibe_focus"),
                        "HCE": latest.get("HCE"),
                    },
                    "bucket": {"label": bucket.get("label")} if bucket else {},
                    "mode": _read_mode(),
                    "now_playing": now_play_latest,
                    "intention": intention or "",
                },
                snapshot_json=latest,
            )
        except Exception:
            pass

    # Always keep engine emit flag in sync with stored state
    engine.set_emit_ndjson(ndjson_state)

    engine_status = engine.get_status()
    muse_status = muse.status()

    stream_alive = bool(engine_status.get("stream_alive"))
    lsl_dot_class = "status-dot" if stream_alive else "status-dot offline"

    inlet_info = engine_status.get("inlet_info") or {}
    name = inlet_info.get("name") or "Unknown"
    stream_type = inlet_info.get("type") or "Unknown"
    ch_count = inlet_info.get("channel_count")
    srate = inlet_info.get("nominal_srate")
    ch_part = f"{ch_count} ch" if ch_count is not None else "ch ?"
    srate_part = f"{srate:.0f} Hz" if isinstance(srate, (int, float)) else "Hz ?"
    info_line = f"{name} | {stream_type} | {ch_part} @ {srate_part}" if inlet_info else "No stream detected"

    age = engine_status.get("last_sample_age_s")
    age_line = f"Last sample age: {age:.2f}s" if isinstance(age, (int, float)) else "Last sample age: N/A"
    sr_est = engine_status.get("samples_received_last_interval") or 0
    sr_line = f"Est. sample rate: {sr_est:.0f} Hz"

    lsl_status_text = [
        html.Div("LSL: Connected" if stream_alive else "LSL: Waiting..."),
        html.Div(info_line),
        html.Div(age_line),
        html.Div(sr_line),
    ]

    engine_text = "Running" if engine_status.get("running") else "Stopped"

    muse_running = muse_status.get("running")
    muse_pid = muse_status.get("pid")
    muse_text = "Running" if muse_running else "Stopped"
    if muse_running and muse_pid:
        muse_text += f" (pid {muse_pid})"

    last_error = muse_status.get("last_error")
    error_text = muse_error or (f"muselsl error: {last_error}" if last_error else "")

    reliability_pct = engine_status.get("reliability_pct")
    history_len = engine_status.get("history_len") or 0
    hist_n = min(600, history_len) if history_len else 0
    if reliability_pct is not None:
        reliability_text = f"Reliability (rolling last 600): {reliability_pct:.1f}% (n={hist_n})"
    else:
        reliability_text = "Reliability (rolling last 600): N/A"

    latest_snapshot = engine_status.get("latest_snapshot") if isinstance(engine_status.get("latest_snapshot"), dict) else {}
    meta = latest_snapshot.get("meta") or {}
    baseline_loaded = meta.get("baseline_loaded")
    baseline_path_used = meta.get("baseline_path_used")
    baseline_warning = meta.get("baseline_warning") or engine_status.get("baseline_warning")
    if baseline_loaded and baseline_path_used:
        baseline_text = f"Baseline: LOADED — {baseline_path_used}"
    else:
        warn = baseline_warning or "Using fallback baseline"
        baseline_text = f"Baseline: FALLBACK — {warn}"

    x_val = latest_snapshot.get("X")
    q_vf = latest_snapshot.get("Q_vibe_focus")
    hce_val = latest_snapshot.get("HCE")
    x_text = f"X: {x_val:.3f}" if x_val is not None else "X: N/A"
    q_text = f"Q (canonical): {q_vf:.3f}" if q_vf is not None else "Q (canonical): N/A"
    hce_text = f"HCE: {hce_val:.3f}" if hce_val is not None else "HCE: N/A"
    label_text = "Current label: (coming soon)"
    try:
        sp_status = spotify_logger.status()
    except Exception:
        sp_status = {}
    latest_play = (sp_status.get("latest") or {}) if isinstance(sp_status, dict) else {}

    now_playing = "N/A"
    if latest_play:
        if latest_play.get("is_playing"):
            prog = latest_play.get("progress_ms") or 0
            dur = latest_play.get("duration_ms") or 1

            def _fmt(ms):
                m = int(ms // 60000)
                s = int((ms % 60000) // 1000)
                return f"{m:02d}:{s:02d}"

            now_playing = f"{latest_play.get('artists') or ''} — {latest_play.get('track_name') or ''} ({_fmt(prog)} / {_fmt(dur)})"
        else:
            now_playing = "Paused"
    if isinstance(sp_status, dict) and sp_status.get("last_error"):
        now_playing = f"Not connected: {sp_status.get('last_error')}"

    # Recent events
    recent_events_html = "No events yet."
    try:
        ev = storage.list_recent_events(limit=10, session_id=session_id)
        if ev:
            items = []
            for e in ev:
                ts = e.get("ts")
                label = e.get("label") or ""
                note = e.get("note") or ""
                ts_txt = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""
                items.append(html.Div(f"{ts_txt} — {label}" + (f" — {note}" if note else "")))
            recent_events_html = html.Div(items)
    except Exception:
        pass

    start_engine_disabled = bool(engine_status.get("running"))
    stop_engine_disabled = not engine_status.get("running")
    start_muse_disabled = bool(muse_running)
    stop_muse_disabled = not muse_running

    ndjson_label = f"Toggle NDJSON Emit ({'On' if ndjson_state else 'Off'})"
    ndjson_color = "success" if ndjson_state else "secondary"
    address_value = (muse_address or profile_address or "").strip()

    track_progress = 0
    track_section = "No track"
    try:
        if latest_play and latest_play.get("duration_ms"):
            prog = latest_play.get("progress_ms") or 0
            dur = max(latest_play.get("duration_ms") or 1, 1)
            track_progress = max(0, min(100, (prog / dur) * 100))
            track_id = latest_play.get("track_id")
            if track_id:
                secs = storage.list_track_sections(track_id, limit_sessions=1)
                if secs:
                    current_sec = None
                    for s in secs:
                        rel_start = (s.get("section_start_ts", 0) - s.get("start_ts", 0)) if s.get("start_ts") is not None else 0
                        rel_end = (s.get("section_end_ts", 0) - s.get("start_ts", 0)) if s.get("start_ts") is not None else 0
                        if (prog / 1000.0) >= rel_start and (prog / 1000.0) <= rel_end:
                            current_sec = s
                            break
                    if current_sec is None:
                        current_sec = secs[0]
                    track_section = f"{current_sec.get('section_label','Section')} — HCE {float(current_sec.get('mean_HCE') or 0):.2f}"
    except Exception:
        pass

    return (
        lsl_dot_class,
        lsl_status_text,
        engine_text,
        muse_text,
        error_text,
        reliability_text,
        baseline_text,
        x_text,
        q_text,
        hce_text,
        label_text,
        now_playing,
        recent_events_html,
        start_engine_disabled,
        stop_engine_disabled,
        start_muse_disabled,
        stop_muse_disabled,
        ndjson_label,
        ndjson_color,
        ndjson_state,
        address_value,
        track_progress,
        track_section,
    )


@callback(
    Output("home-bookmark-modal", "is_open"),
    Output("home-bookmark-label", "value"),
    Output("home-bookmark-note", "value"),
    Input("home-bookmark-btn", "n_clicks"),
    Input("home-bookmark-save", "n_clicks"),
    Input("home-bookmark-cancel", "n_clicks"),
    State("home-bookmark-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_bookmark_modal(open_click, save_click, cancel_click, is_open):
    triggered_id = ctx.triggered_id
    if triggered_id == "home-bookmark-save" or triggered_id == "home-bookmark-cancel":
        # Close after save or cancel
        return False, "moment", ""
    if triggered_id == "home-bookmark-btn":
        return True, "moment", ""
    return is_open, dash.no_update, dash.no_update


def _render_top_tracks(rows, limit: int) -> html.Ul:
    items = []
    for idx, r in enumerate(rows[:limit], start=1):
        title = r.get("title") or ""
        artist = r.get("artist") or ""
        avg_hce = r.get("avg_hce", 0.0)
        count = r.get("play_count", 0)
        items.append(
            html.Li(
                f"{idx}. {artist} — {title} | HCE={avg_hce:.2f} | plays={count}",
                className="small",
            )
        )
    return html.Ul(items) if items else html.Div("No tracks yet.", className="text-muted")


@callback(
    Output("home-top10-tracks", "children"),
    Output("home-top100-tracks", "children"),
    Output("home-peak-hce", "children"),
    Output("home-optimal-windows", "children"),
    Input("home-status-interval", "n_intervals"),
)
def update_top_tracks(_n):
    rows_100 = storage.list_top_tracks(limit=100)
    rows_10 = rows_100[:10]
    top10 = _render_top_tracks(rows_10, 10)
    top100 = _render_top_tracks(rows_100, 100)
    peak = f"{rows_100[0].get('avg_hce', 0.0):.2f}" if rows_100 else "—"
    # Optimal windows from today's buckets
    today = dt.date.today()
    ts0 = dt.datetime.combine(today, dt.time.min).timestamp()
    ts1 = dt.datetime.combine(today, dt.time.max).timestamp()
    buckets = storage.get_buckets_between(ts0, ts1, session_id=None)
    buckets = sorted(buckets, key=lambda b: b.get("mean_HCE", 0.0) or 0.0, reverse=True)
    items = []
    for b in buckets[:3]:
        start = b.get("bucket_start_ts")
        hce = b.get("mean_HCE", 0.0) or 0.0
        ttxt = dt.datetime.fromtimestamp(start).strftime("%H:%M") if start else "--"
        items.append(html.Li(f"{ttxt} — HCE {hce:.2f}", className="small"))
    optimal = html.Ul(items) if items else html.Div("No HCE windows yet.", className="text-muted")
    return top10, top100, peak, optimal

