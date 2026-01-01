import json
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, callback_context

from alma import config
from alma.app_state import registry


PROFILE_PATH = Path(__file__).resolve().parents[3] / "profiles" / "default.json"


def _load_profile():
    try:
        with PROFILE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Environment / Turrell"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Display (0 = laptop, 1 = extended)"),
                                        dcc.Dropdown(
                                            id="env-display",
                                            options=[
                                                {"label": "0 (laptop)", "value": 0},
                                                {"label": "1 (extended)", "value": 1},
                                            ],
                                            value=0,
                                            clearable=False,
                                        ),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Checklist(
                                            options=[{"label": "Fullscreen", "value": "fullscreen"}],
                                            value=[],
                                            id="env-fullscreen",
                                            switch=True,
                                        )
                                    ],
                                    md=2,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button("Start Turrell", id="env-start-turrell", color="primary", className="me-2"),
                                        dbc.Button("Stop Turrell", id="env-stop-turrell", color="secondary"),
                                    ],
                                    md=6,
                                    sm=12,
                                    className="d-flex align-items-center mt-3 mt-md-0",
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        html.Div(id="env-status-text", className="mt-2"),
                        html.Div(id="env-ndjson-warning", className="text-warning small mt-2"),
                        dbc.Button(
                            "Enable NDJSON Emit",
                            id="env-enable-ndjson",
                            color="info",
                            size="sm",
                            className="mt-2",
                        ),
                        dcc.Interval(id="env-interval", interval=1000, n_intervals=0),
                    ]
                ),
            ],
            className="page-card",
        )
    ],
    fluid=True,
    className="page-container",
)


@callback(
    Output("env-status-text", "children"),
    Output("env-ndjson-warning", "children"),
    Input("env-interval", "n_intervals"),
    Input("env-start-turrell", "n_clicks"),
    Input("env-stop-turrell", "n_clicks"),
    Input("env-enable-ndjson", "n_clicks"),
    State("env-display", "value"),
    State("env-fullscreen", "value"),
)
def handle_turrell(_n, start_clicks, stop_clicks, enable_ndjson_clicks, display, fullscreen_values):
    runner = registry.turrell_runner
    engine = registry.state_engine
    fullscreen = "fullscreen" in (fullscreen_values or [])

    triggered = dash_triggered()
    # NDJSON status
    engine_status = engine.get_status()
    ndjson_on = bool(engine_status.get("emit_ndjson"))

    warning_text = ""
    if triggered == "env-enable-ndjson":
        try:
            engine.set_emit_ndjson(True)
            ndjson_on = True
            warning_text = "NDJSON emit enabled."
        except Exception as exc:
            warning_text = f"Failed to enable NDJSON emit: {exc}"
    if triggered == "env-start-turrell":
        if not ndjson_on:
            warning_text = "NDJSON emit is OFF. Enable it before launching Turrell."
        else:
            profile = _load_profile()
            ndjson_path = profile.get("ndjson_state_path", str(config.STATE_STREAM_PATH))
            runner.start(display=display or 0, fullscreen=fullscreen, ndjson_path=ndjson_path)
    elif triggered == "env-stop-turrell":
        runner.stop()

    st = runner.status()
    status_parts = [
        f"Turrell: {'Running' if st.get('running') else 'Stopped'}",
        f"PID: {st.get('pid') or '—'}",
    ]
    if st.get("last_error"):
        status_parts.append(f"Error: {st['last_error']}")
    if not warning_text and not ndjson_on:
        warning_text = "NDJSON emit is OFF. Turrell needs live NDJSON; click 'Enable NDJSON Emit' first."
    return html.Div(" | ".join(status_parts)), warning_text


def dash_triggered():
    try:
        ctx = callback_context
        if not ctx.triggered:
            return None
        return ctx.triggered[0]["prop_id"].split(".")[0]
    except Exception:
        return None
