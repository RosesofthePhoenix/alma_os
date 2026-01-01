import json
import time
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
                _status_item("Reliability (rolling last 600)", html.Span("N/A", id="home-reliability-text")),
                _status_item("Baseline", html.Span("N/A", id="home-baseline-text")),
                _status_item("Current label", html.Span("(coming soon)", id="home-label-text")),
                _status_item("Now Playing", html.Span("N/A", id="home-now-playing")),
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
                            dbc.Button("Bookmark Moment", id="home-bookmark-btn", color="primary", className="mt-4"),
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
                dcc.Store(id="home-ndjson-enabled", data=False),
                dcc.Interval(id="home-status-interval", interval=1000, n_intervals=0),
            ]
        ),
    ],
    className="page-card",
)

layout = dbc.Container(
    [
        status_strip,
        controls_card,
        dbc.Card(
            [
                dbc.CardHeader("Home"),
                dbc.CardBody(html.P("Placeholder content for Home page.")),
            ],
            className="page-card",
        ),
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
    Output("home-label-text", "children"),
    Output("home-now-playing", "children"),
    Output("home-recent-events", "children"),
    Output("home-start-engine-btn", "disabled"),
    Output("home-stop-engine-btn", "disabled"),
    Output("home-start-muse-btn", "disabled"),
    Output("home-stop-muse-btn", "disabled"),
    Output("home-toggle-ndjson-btn", "children"),
    Output("home-ndjson-enabled", "data"),
    Output("home-muse-address", "value"),
    Input("home-status-interval", "n_intervals"),
    Input("home-start-engine-btn", "n_clicks"),
    Input("home-stop-engine-btn", "n_clicks"),
    Input("home-start-muse-btn", "n_clicks"),
    Input("home-stop-muse-btn", "n_clicks"),
    Input("home-toggle-ndjson-btn", "n_clicks"),
    Input("home-start-spotify-btn", "n_clicks"),
    Input("home-stop-spotify-btn", "n_clicks"),
    Input("home-bookmark-btn", "n_clicks"),
    State("home-muse-address", "value"),
    State("home-ndjson-enabled", "data"),
    State("home-event-label", "value"),
    State("home-event-note", "value"),
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
    muse_address,
    ndjson_enabled,
    event_label,
    event_note,
):
    triggered = ctx.triggered_id
    muse_error = None

    profile = _load_profile()
    profile_address = profile.get("muse_address", "")

    engine = registry.state_engine
    muse = registry.muse_stream_manager
    spotify_logger = registry.spotify_logger
    session_id = engine.get_session_id()

    ndjson_state = bool(ndjson_enabled)

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
        spotify_logger.start(session_id=session_id)
    elif triggered == "home-stop-spotify-btn":
        spotify_logger.stop()
    elif triggered == "home-bookmark-btn":
        ts_now = time.time()
        latest = engine_status.get("latest_snapshot") or {}
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
                label=(event_label or "moment"),
                note=(event_note or ""),
                tags_json={"page": "home"},
                context_json={
                    "snapshot": {
                        "X": latest.get("X"),
                        "Q_vibe_focus": latest.get("Q_vibe_focus"),
                    },
                    "bucket": {"label": bucket.get("label")} if bucket else {},
                    "mode": _read_mode(),
                    "now_playing": now_play_latest,
                },
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

    meta = (latest.get("meta") or {}) if isinstance(latest, dict) else {}
    baseline_loaded = meta.get("baseline_loaded")
    baseline_path_used = meta.get("baseline_path_used")
    baseline_warning = meta.get("baseline_warning")
    if baseline_loaded and baseline_path_used:
        baseline_text = f"Baseline: LOADED — {baseline_path_used}"
    else:
        warn = baseline_warning or "Using fallback baseline"
        baseline_text = f"Baseline: FALLBACK — {warn}"

    latest = engine_status.get("latest_snapshot") or {}
    x_val = latest.get("X")
    q_vf = latest.get("Q_vibe_focus")
    x_text = f"X: {x_val:.3f}" if x_val is not None else "X: N/A"
    q_text = f"Q (canonical): {q_vf:.3f}" if q_vf is not None else "Q (canonical): N/A"
    label_text = "Current label: (coming soon)"
    sp_status = spotify_logger.status()
    latest_play = sp_status.get("latest") or {}

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
    if sp_status.get("last_error"):
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
    address_value = muse_address or profile_address

    return (
        lsl_dot_class,
        lsl_status_text,
        engine_text,
        muse_text,
        error_text,
        reliability_text,
        x_text,
        q_text,
        label_text,
        now_playing,
        recent_events_html,
        start_engine_disabled,
        stop_engine_disabled,
        start_muse_disabled,
        stop_muse_disabled,
        ndjson_label,
        ndjson_state,
        address_value,
        baseline_text,
    )

