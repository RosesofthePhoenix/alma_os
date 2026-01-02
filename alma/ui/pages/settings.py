import json
from pathlib import Path

from dash import callback, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from alma.engine import lsl_client

PROFILE_PATH = Path(__file__).resolve().parents[3] / "profiles" / "default.json"


def _load_profile():
    try:
        with PROFILE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_profile(data):
    try:
        with PROFILE_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _get_field(profile, key, default=""):
    return profile.get(key, default) if isinstance(profile, dict) else default


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Settings"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Muse Address"),
                                        dbc.Input(id="settings-muse-address", placeholder="BLE address", size="sm"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("NDJSON State Path"),
                                        dbc.Input(id="settings-ndjson-path", placeholder="sessions/current/state_stream.ndjson", size="sm"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Preferred LSL Stream Name"),
                                        dbc.Input(id="settings-preferred-lsl-name", placeholder="Optional stream name to prefer", size="sm"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Spotify Client ID"),
                                        dbc.Input(id="settings-spotify-cid", placeholder="SPOTIPY_CLIENT_ID", size="sm"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Spotify Client Secret"),
                                        dbc.Input(id="settings-spotify-secret", placeholder="SPOTIPY_CLIENT_SECRET", size="sm", type="password"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Spotify Redirect URI"),
                                        dbc.Input(id="settings-spotify-redirect", placeholder="SPOTIPY_REDIRECT_URI", size="sm"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Preferred Spotify Device (optional)"),
                                        dbc.Input(id="settings-spotify-device", placeholder="e.g., iPhone", size="sm"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Stress adaptive playback"),
                                        dbc.Checklist(
                                            id="settings-auto-adapt",
                                            options=[{"label": "Auto-adapt on stress", "value": "auto"}],
                                            value=[],
                                            switch=True,
                                        ),
                                        dbc.Input(id="settings-soothing-source", placeholder="Soothing playlist/artist", size="sm", className="mt-1"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button("Save", id="settings-save-btn", color="primary"),
                                        html.Span(id="settings-save-status", className="ms-3"),
                                    ],
                                    md=6,
                                    sm=12,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button("Refresh Detected Streams", id="settings-refresh-streams", color="secondary", outline=True, size="sm"),
                                        html.Div(id="settings-stream-list", className="mt-2 small"),
                                    ],
                                    md=6,
                                    sm=12,
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        dcc.Store(id="settings-profile-store", data=_load_profile()),
                        dcc.Interval(id="settings-load-interval", interval=1000, max_intervals=1),
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
    Output("settings-profile-store", "data"),
    Output("settings-muse-address", "value"),
    Output("settings-ndjson-path", "value"),
    Output("settings-preferred-lsl-name", "value"),
    Output("settings-spotify-cid", "value"),
    Output("settings-spotify-secret", "value"),
    Output("settings-spotify-redirect", "value"),
    Output("settings-spotify-device", "value"),
    Output("settings-auto-adapt", "value"),
    Output("settings-soothing-source", "value"),
    Input("settings-load-interval", "n_intervals"),
)
def load_profile(_):
    profile = _load_profile()
    return (
        profile,
        _get_field(profile, "muse_address", ""),
        _get_field(profile, "ndjson_state_path", ""),
        _get_field(profile, "preferred_lsl_stream_name", ""),
        _get_field(profile, "SPOTIPY_CLIENT_ID", ""),
        _get_field(profile, "SPOTIPY_CLIENT_SECRET", ""),
        _get_field(profile, "SPOTIPY_REDIRECT_URI", ""),
        _get_field(profile, "SPOTIFY_PREFERRED_DEVICE", ""),
        ["auto"] if profile.get("AUTO_ADAPT_STRESS") else [],
        _get_field(profile, "SOOTHING_SOURCE", ""),
    )


@callback(
    Output("settings-save-status", "children"),
    Input("settings-save-btn", "n_clicks"),
    State("settings-profile-store", "data"),
    State("settings-muse-address", "value"),
    State("settings-ndjson-path", "value"),
    State("settings-preferred-lsl-name", "value"),
    State("settings-spotify-cid", "value"),
    State("settings-spotify-secret", "value"),
    State("settings-spotify-redirect", "value"),
    State("settings-spotify-device", "value"),
    State("settings-auto-adapt", "value"),
    State("settings-soothing-source", "value"),
    prevent_initial_call=True,
)
def save_profile(n_clicks, profile_data, muse_address, ndjson_path, preferred_name, cid, secret, redirect_uri, device, auto_adapt, soothing_source):
    profile = profile_data or {}
    profile["muse_address"] = muse_address or ""
    profile["ndjson_state_path"] = ndjson_path or ""
    profile["preferred_lsl_stream_name"] = preferred_name or ""
    if cid:
        profile["SPOTIPY_CLIENT_ID"] = cid
    if secret:
        profile["SPOTIPY_CLIENT_SECRET"] = secret
    if redirect_uri:
        profile["SPOTIPY_REDIRECT_URI"] = redirect_uri
    if device is not None:
        profile["SPOTIFY_PREFERRED_DEVICE"] = device
    profile["AUTO_ADAPT_STRESS"] = bool(auto_adapt and "auto" in auto_adapt)
    profile["SOOTHING_SOURCE"] = soothing_source or ""
    ok, err = _save_profile(profile)
    if ok:
        return "Saved."
    return f"Error saving: {err}"


@callback(
    Output("settings-stream-list", "children"),
    Input("settings-refresh-streams", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_streams(_):
    streams = lsl_client.list_streams(timeout=1.0)
    if not streams:
        return html.Div("No streams detected.", className="text-muted")
    items = []
    for s in streams:
        items.append(
            html.Div(
                f"{s.get('name') or 'Unknown'} | {s.get('type') or '?'} | {s.get('channel_count') or '?'} ch @ {s.get('nominal_srate') or '?'} Hz | src: {s.get('source_id') or '?'}"
            )
        )
    return html.Div(items)
