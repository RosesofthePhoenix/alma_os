import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

from alma.engine import storage


def _smooth(series: pd.Series, window: int = 5) -> pd.Series:
    if series.empty:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _live_metrics(bin_sec: float = 1.0):
    """Return live raw q_abs_raw / hce_raw / X and metadata (no fallbacks, no smoothing)."""
    try:
        latest = storage.get_latest_spotify(session_id=None)
    except Exception:
        latest = None
    if not latest or not latest.get("is_playing") or not latest.get("track_id"):
        return (
            pd.DataFrame(columns=["t", "Q"]),
            pd.DataFrame(columns=["t", "X"]),
            pd.DataFrame(columns=["t", "HCE"]),
            0.0,
            "No track playing",
            None,
        )
    track_id = latest.get("track_id")
    track_title = latest.get("track_name") or "Now playing"
    artist = latest.get("artists") or ""
    label = f"{track_title} — {artist}" if artist else track_title
    progress_ms = latest.get("progress_ms") or 0
    duration_ms = latest.get("duration_ms") or 1
    progress_sec = max(0.0, min(float(progress_ms) / 1000.0, float(duration_ms) / 1000.0))
    duration_cap = max(float(duration_ms) / 1000.0, progress_sec)

    # Try live_waveform_points first
    live_pts = storage.list_live_waveform_points(track_id, limit_points=8000)
    if live_pts:
        df_live = pd.DataFrame(live_pts).sort_values(["abs_ts", "rel_sec"])
        def _prep(col: str, out_col: str):
            series = df_live[col]
            filled = series.ffill()
            filled_mask = series.isna() & filled.notna()
            return pd.DataFrame(
                {
                    "t": df_live["rel_sec"],
                    out_col: series,
                    f"{out_col}_ffill": filled,
                    f"{out_col}_filled_mask": filled_mask,
                }
            )
        q_df = _prep("q_raw", "Q")
        x_df = _prep("x_raw", "X")
        h_df = _prep("hce_raw", "HCE")
        return q_df, x_df, h_df, progress_sec, label, h_df["HCE_ffill"]

    # Fallback: none
    return (
        pd.DataFrame(columns=["t", "Q", "Q_ffill", "Q_filled_mask"]),
        pd.DataFrame(columns=["t", "X", "X_ffill", "X_filled_mask"]),
        pd.DataFrame(columns=["t", "HCE", "HCE_ffill", "HCE_filled_mask"]),
        progress_sec,
        label,
        None,
    )


def _dot_color(val: float) -> str:
    if val is None or not np.isfinite(val):
        return "#d7b34d"
    if val < 0.3:
        return "#4da6ff"
    if val < 0.8:
        return "#d7b34d"
    return "#ff7ad1"


layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(id="live-media-title", className="h4 text-warning mb-2"),
                        dbc.Checklist(
                            id="live-media-metrics",
                            options=[
                                {"label": "Rolling Q (5s)", "value": "q"},
                                {"label": "Rolling X (5s)", "value": "x"},
                                {"label": "Rolling HCE (5s)", "value": "hce"},
                            ],
                            value=["q"],
                            inline=True,
                            switch=True,
                            className="mb-2",
                        ),
                        dbc.Checklist(
                            id="live-media-trans",
                            options=[{"label": "Highlight Transcendence", "value": "on"}],
                            value=["on"],
                            switch=True,
                            className="mb-3",
                        ),
                    ],
                    md=12,
                ),
            ],
            className="g-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="live-media-graph", style={"height": "70vh"}),
                    md=10,
                ),
                dbc.Col(
                    html.Div(id="live-media-legend", className="text-muted small"),
                    md=2,
                ),
            ],
            className="g-2",
        ),
        dcc.Interval(id="live-media-interval", interval=1500, n_intervals=0),
    ],
    fluid=True,
    className="page-container",
)


@callback(
    Output("live-media-graph", "figure"),
    Output("live-media-title", "children"),
    Output("live-media-legend", "children"),
    Input("live-media-interval", "n_intervals"),
    Input("live-media-metrics", "value"),
    Input("live-media-trans", "value"),
)
def render_live_media(_n, metrics_selected, trans_toggle):
    metrics_selected = metrics_selected or ["q"]
    live_q, live_x, live_h, progress_rel, label, h_series = _live_metrics(bin_sec=1.0)

    fig = go.Figure()
    duration_final = max(float(progress_rel or 0.0), 1.0)
    fig.add_trace(
        go.Scatter(
            x=[0, duration_final],
            y=[0, 0],
            mode="lines",
            line=dict(color="rgba(200,200,200,0.1)", dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    color_map = {"q": "#ff5fb2", "x": "#00e8ff", "hce": "#ffd15c"}
    label_map = {"q": "Q_abs_raw", "x": "X (raw)", "hce": "HCE_raw"}
    legend_items = []
    peak_val = 0.0
    front_marker_trace = None
    primary_for_marker = None

    for metric in ["q", "x", "hce"]:
        if metric not in metrics_selected:
            continue
        df_live = {"q": live_q, "x": live_x, "hce": live_h}[metric]
        col = {"q": "Q", "x": "X", "hce": "HCE"}[metric]
        if df_live.empty:
            continue
        solid = df_live[f"{col}_ffill"]
        filled_mask = df_live.get(f"{col}_filled_mask", pd.Series([], dtype=bool))
        if filled_mask.empty:
            filled_mask = pd.Series([False] * len(df_live))
        # Build solid/dashed segments
        solid_y = np.where(~filled_mask, solid, np.nan)
        dash_y = np.where(filled_mask, solid, np.nan)
        series_clean = pd.Series(solid_y).dropna()
        last_val = float(series_clean.iloc[-1]) if not series_clean.empty else None
        if not series_clean.empty and np.isfinite(series_clean).any():
            peak_val = max(peak_val, float(np.nanmax(series_clean)))
        width = 5 if metric == "q" else 3
        fig.add_trace(
            go.Scatter(
                x=df_live["t"],
                y=solid_y,
                mode="lines",
                line=dict(color=color_map[metric], width=width),
                name=label_map[metric],
                hovertemplate="t=%{x:.1f}s<br>value=%{y:.3f}<extra></extra>",
                showlegend=False,
            )
        )
        if np.isfinite(dash_y).any():
            fig.add_trace(
                go.Scatter(
                    x=df_live["t"],
                    y=dash_y,
                    mode="lines",
                    line=dict(color=color_map[metric], width=width, dash="dash"),
                    name=f"{label_map[metric]} (filled)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        if primary_for_marker is None:
            primary_for_marker = (df_live, col, last_val, color_map[metric], label_map[metric])
        legend_items.append(html.Div([html.Span("● ", style={"color": color_map[metric], "fontWeight": "bold"}), html.Span(label_map[metric], style={"fontWeight": "bold"})]))

    # Transcendence highlight (soft fill) using live HCE
    highlight_on = trans_toggle and ("on" in trans_toggle)
    if highlight_on and h_series is not None and not h_series.empty:
        try:
            trans_thresh = float(np.nanpercentile(h_series, 90))
        except Exception:
            trans_thresh = None
        if trans_thresh is not None and not live_h.empty:
            y_vals = []
            x_vals = []
            for t, h in zip(live_h["t"], live_h["HCE"]):
                if h >= trans_thresh:
                    x_vals.append(t)
                    y_vals.append(h)
                else:
                    x_vals.append(t)
                    y_vals.append(None)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    line=dict(color="rgba(255,154,213,0.35)", width=0.1),
                    fill="tozeroy",
                    fillcolor="rgba(255,154,213,0.18)",
                    name="Transcendence",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            legend_items.append(html.Div([html.Span("● ", style={"color": "#ff9ad5", "fontWeight": "bold"}), html.Span("Transcendence", style={"fontWeight": "bold"})]))

    # Single advancing front dot (primary metric)
    if primary_for_marker is not None:
        df_live, col, last_val, base_color, label_txt = primary_for_marker
        if not df_live.empty:
            fig.add_trace(
                go.Scatter(
                    x=[df_live["t"].iloc[-1]],
                    y=[df_live[col].iloc[-1]],
                    mode="markers",
                    marker=dict(
                        color="#ffd15c",
                        size=13,
                        symbol="circle",
                        line=dict(color=base_color, width=2),
                        opacity=0.95,
                    ),
                    name=f"{label_txt} (now)",
                    hovertemplate="Now: t=%{x:.1f}s<br>value=%{y:.3f}<extra></extra>",
                    showlegend=False,
                )
            )
            peak_val = max(peak_val, float(df_live[col].iloc[-1]))

    fig.update_layout(
        template="plotly_dark",
        title="Pure live resonance",
        xaxis_title="Seconds (relative)",
        yaxis_title="Value",
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    if peak_val and np.isfinite(peak_val):
        fig.update_yaxes(range=[0, peak_val * 1.1])
    if progress_rel is not None:
        fig.add_vline(x=progress_rel, line=dict(color="#ffbf00", width=2, dash="dash"), name="Now")
    return fig, label, legend_items
