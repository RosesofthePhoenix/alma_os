import sqlite3
from typing import List, Dict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from alma import config
from alma.engine import storage


def _dense_tracks(limit: int = 200) -> pd.DataFrame:
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            df = pd.read_sql_query(
                """
                SELECT track_id, title, artist, COUNT(*) AS pts
                FROM live_waveform_points
                WHERE track_id IS NOT NULL
                GROUP BY track_id, title, artist
                HAVING pts > 50
                ORDER BY MAX(abs_ts) DESC
                LIMIT ?
                """,
                conn,
                params=(limit,),
            )
        return df
    except Exception:
        return pd.DataFrame()


def _latest_two_dense() -> List[str]:
    df = _dense_tracks(limit=10)
    if df.empty:
        return []
    return df["track_id"].tolist()[:2]


def _track_options() -> List[Dict[str, str]]:
    df = _dense_tracks(limit=400)
    if df.empty:
        return []
    return [{"label": f"{r['title']} — {r['artist']}", "value": r["track_id"]} for _, r in df.iterrows()]


def _load_raw_waveform(track_id: str, max_points: int = 8000) -> pd.DataFrame:
    try:
        pts = storage.list_live_waveform_points(track_id, limit_points=max_points)
        if not pts:
            return pd.DataFrame()
        df = pd.DataFrame(pts)
        # Normalize column names for plotting and add filled masks via forward fill
        df = df.rename(columns={"q_raw": "q", "hce_raw": "hce", "x_raw": "x"})
        for col in ["q", "hce", "x"]:
            series = df[col]
            filled = series.ffill()
            filled_mask = series.isna() & filled.notna()
            df[col] = series
            df[f"{col}_ffill"] = filled
            df[f"{col}_filled_mask"] = filled_mask
        return df.sort_values(["abs_ts", "rel_sec"])
    except Exception:
        return pd.DataFrame()


def _raw_fig(df: pd.DataFrame, title: str, metrics: List[str]) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text="No dense waveform data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", height=380, title=title)
        return fig
    metrics = metrics or ["hce", "q", "x"]
    # Ensure columns exist
    for col in ["hce", "q", "x"]:
        if col not in df.columns:
            df[col] = np.nan
        if f"{col}_ffill" not in df.columns:
            df[f"{col}_ffill"] = df[col].ffill()
        if f"{col}_filled_mask" not in df.columns:
            df[f"{col}_filled_mask"] = df[col].isna() & df[col].ffill().notna()
    if "hce" in metrics:
        solid = np.where(~df["hce_filled_mask"], df["hce_ffill"], np.nan)
        dash = np.where(df["hce_filled_mask"], df["hce_ffill"], np.nan)
        fig.add_trace(
            go.Scatter(
                x=df["rel_sec"],
                y=solid,
                mode="lines",
                line=dict(color="#d7b34d", width=2),
                name="HCE",
                opacity=0.9,
                showlegend=True,
            )
        )
        if np.isfinite(dash).any():
            fig.add_trace(
                go.Scatter(
                    x=df["rel_sec"],
                    y=dash,
                    mode="lines",
                    line=dict(color="#d7b34d", width=2, dash="dash"),
                    name="HCE (filled)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    if "q" in metrics:
        solid = np.where(~df["q_filled_mask"], df["q_ffill"], np.nan)
        dash = np.where(df["q_filled_mask"], df["q_ffill"], np.nan)
        fig.add_trace(
            go.Scatter(
                x=df["rel_sec"],
                y=solid,
                mode="lines",
                line=dict(color="#5fb3ff", width=1.5),
                name="Q",
                opacity=0.8,
            )
        )
        if np.isfinite(dash).any():
            fig.add_trace(
                go.Scatter(
                    x=df["rel_sec"],
                    y=dash,
                    mode="lines",
                    line=dict(color="#5fb3ff", width=1.5, dash="dash"),
                    name="Q (filled)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    if "x" in metrics:
        solid = np.where(~df["x_filled_mask"], df["x_ffill"], np.nan)
        dash = np.where(df["x_filled_mask"], df["x_ffill"], np.nan)
        fig.add_trace(
            go.Scatter(
                x=df["rel_sec"],
                y=solid,
                mode="lines",
                line=dict(color="#ff7f7f", width=1.5),
                name="X",
                opacity=0.8,
            )
        )
        if np.isfinite(dash).any():
            fig.add_trace(
                go.Scatter(
                    x=df["rel_sec"],
                    y=dash,
                    mode="lines",
                    line=dict(color="#ff7f7f", width=1.5, dash="dash"),
                    name="X (filled)",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Seconds (relative)",
        yaxis_title="Raw value",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _top_tracks(kind: str, limit: int = 20) -> pd.DataFrame:
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            if kind == "relax":
                q = """
                SELECT track_id, title, artist,
                       AVG(mean_Q) AS avg_q,
                       AVG(mean_X) AS avg_x,
                       AVG(mean_HCE) AS avg_hce,
                       MAX(mean_Q) AS peak_q,
                       MAX(mean_X) AS peak_x,
                       MAX(mean_HCE) AS peak_hce
                FROM track_sessions
                WHERE mean_Q IS NOT NULL AND mean_Q < 0.2
                GROUP BY track_id, title, artist
                HAVING COUNT(*) >= 1
                ORDER BY avg_q ASC, avg_x ASC
                LIMIT ?
                """
            elif kind == "flow":
                q = """
                SELECT track_id, title, artist,
                       AVG(mean_Q) AS avg_q,
                       AVG(mean_X) AS avg_x,
                       AVG(mean_HCE) AS avg_hce,
                       MAX(mean_Q) AS peak_q,
                       MAX(mean_X) AS peak_x,
                       MAX(mean_HCE) AS peak_hce
                FROM track_sessions
                WHERE mean_X BETWEEN 1.3 AND 1.8
                GROUP BY track_id, title, artist
                HAVING COUNT(*) >= 1
                ORDER BY avg_x DESC
                LIMIT ?
                """
            else:  # transcendent
                q = """
                SELECT track_id, title, artist,
                       MAX(mean_HCE) AS peak_hce,
                       AVG(mean_HCE) AS avg_hce,
                       AVG(mean_Q) AS avg_q,
                       AVG(mean_X) AS avg_x,
                       MAX(mean_Q) AS peak_q,
                       MAX(mean_X) AS peak_x
                FROM track_sessions
                WHERE mean_HCE IS NOT NULL
                GROUP BY track_id, title, artist
                HAVING COUNT(*) >= 1
                ORDER BY peak_hce DESC
                LIMIT ?
                """
            df = pd.read_sql_query(q, conn, params=(limit,))
            return df
    except Exception:
        return pd.DataFrame()


def _top_metrics(metric: str, limit: int = 100) -> pd.DataFrame:
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            if metric == "x_mean":
                order = "AVG(mean_X) DESC"
                select_extra = "AVG(mean_X) AS avg_x, MAX(mean_X) AS peak_x, AVG(mean_Q) AS avg_q, MAX(mean_Q) AS peak_q, AVG(mean_HCE) AS avg_hce, MAX(mean_HCE) AS peak_hce"
            elif metric == "q_mean":
                order = "AVG(mean_Q) DESC"
                select_extra = "AVG(mean_X) AS avg_x, MAX(mean_X) AS peak_x, AVG(mean_Q) AS avg_q, MAX(mean_Q) AS peak_q, AVG(mean_HCE) AS avg_hce, MAX(mean_HCE) AS peak_hce"
            elif metric == "hce_mean":
                order = "AVG(mean_HCE) DESC"
                select_extra = "AVG(mean_X) AS avg_x, MAX(mean_X) AS peak_x, AVG(mean_Q) AS avg_q, MAX(mean_Q) AS peak_q, AVG(mean_HCE) AS avg_hce, MAX(mean_HCE) AS peak_hce"
            elif metric == "x_peak":
                order = "MAX(mean_X) DESC"
                select_extra = "AVG(mean_X) AS avg_x, MAX(mean_X) AS peak_x, AVG(mean_Q) AS avg_q, MAX(mean_Q) AS peak_q, AVG(mean_HCE) AS avg_hce, MAX(mean_HCE) AS peak_hce"
            elif metric == "q_peak":
                order = "MAX(mean_Q) DESC"
                select_extra = "AVG(mean_X) AS avg_x, MAX(mean_X) AS peak_x, AVG(mean_Q) AS avg_q, MAX(mean_Q) AS peak_q, AVG(mean_HCE) AS avg_hce, MAX(mean_HCE) AS peak_hce"
            else:  # hce_peak
                order = "MAX(mean_HCE) DESC"
                select_extra = "AVG(mean_X) AS avg_x, MAX(mean_X) AS peak_x, AVG(mean_Q) AS avg_q, MAX(mean_Q) AS peak_q, AVG(mean_HCE) AS avg_hce, MAX(mean_HCE) AS peak_hce"
            q = f"""
            SELECT track_id, title, artist, {select_extra}
            FROM track_sessions
            WHERE mean_HCE IS NOT NULL
            GROUP BY track_id, title, artist
            ORDER BY {order}
            LIMIT ?
            """
            df = pd.read_sql_query(q, conn, params=(limit,))
            return df
    except Exception:
        return pd.DataFrame()


layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Track A"),
                                dcc.Dropdown(
                                    id="si-track-a",
                                    options=[],
                                    placeholder="Select dense track",
                                    style={"width": "100%"},
                                ),
                            ],
                            className="mb-2",
                            style={"width": "100%"},
                        ),
                        dbc.Checklist(
                            id="si-metrics-a",
                            options=[
                                {"label": "Show HCE", "value": "hce"},
                                {"label": "Show Q", "value": "q"},
                                {"label": "Show X", "value": "x"},
                            ],
                            value=["hce", "q", "x"],
                            inline=True,
                            switch=True,
                            className="mb-2",
                        ),
                        dcc.Graph(id="si-graph-a", style={"height": "60vh"}),
                    ],
                    md=12,
                ),
            ],
            className="g-3 mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Track B"),
                                dcc.Dropdown(
                                    id="si-track-b",
                                    options=[],
                                    placeholder="Select dense track",
                                    style={"width": "100%"},
                                ),
                            ],
                            className="mb-2",
                            style={"width": "100%"},
                        ),
                        dbc.Checklist(
                            id="si-metrics-b",
                            options=[
                                {"label": "Show HCE", "value": "hce"},
                                {"label": "Show Q", "value": "q"},
                                {"label": "Show X", "value": "x"},
                            ],
                            value=["hce", "q", "x"],
                            inline=True,
                            switch=True,
                            className="mb-2",
                        ),
                        dcc.Graph(id="si-graph-b", style={"height": "60vh"}),
                    ],
                    md=12,
                ),
            ],
            className="g-3 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Checklist(
                            id="si-induce-toggle",
                            options=[
                                {"label": "Relax-inducing", "value": "relax"},
                                {"label": "Flow-inducing", "value": "flow"},
                                {"label": "Transcendent-inducing", "value": "trans"},
                            ],
                            value=["relax", "flow", "trans"],
                            inline=True,
                            switch=True,
                            className="mb-2",
                        ),
                        html.Div(id="si-top20"),
                    ],
                    md=12,
                ),
            ],
            className="g-3 mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.RadioItems(
                            id="si-top100-toggle",
                            options=[
                                {"label": "X mean", "value": "x_mean"},
                                {"label": "Q mean", "value": "q_mean"},
                                {"label": "HCE mean", "value": "hce_mean"},
                                {"label": "X peak", "value": "x_peak"},
                                {"label": "Q peak", "value": "q_peak"},
                                {"label": "HCE peak", "value": "hce_peak"},
                            ],
                            value="hce_mean",
                            inline=True,
                            className="mb-2",
                        ),
                        html.Div(id="si-top100"),
                    ],
                    md=12,
                ),
            ],
            className="g-3",
        ),
        dcc.Interval(id="si-interval", interval=8000, n_intervals=0),
    ],
    fluid=True,
    className="page-container",
)


@callback(
    Output("si-track-a", "options"),
    Output("si-track-b", "options"),
    Output("si-track-a", "value"),
    Output("si-track-b", "value"),
    Input("si-interval", "n_intervals"),
)
def load_dense_tracks(_n):
    opts = _track_options()
    defaults = _latest_two_dense()
    a = defaults[0] if len(defaults) > 0 else (opts[0]["value"] if opts else None)
    b = defaults[1] if len(defaults) > 1 else (opts[1]["value"] if len(opts) > 1 else None)
    return opts, opts, a, b


@callback(
    Output("si-graph-a", "figure"),
    Output("si-graph-b", "figure"),
    Input("si-track-a", "value"),
    Input("si-track-b", "value"),
    Input("si-metrics-a", "value"),
    Input("si-metrics-b", "value"),
)
def update_graphs(a, b, ma, mb):
    df_a = _load_raw_waveform(a) if a else pd.DataFrame()
    df_b = _load_raw_waveform(b) if b else pd.DataFrame()
    return _raw_fig(df_a, "Raw Dense — Track A", ma), _raw_fig(df_b, "Raw Dense — Track B", mb)


def _table_from_df(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div("No data", className="text-muted")
    header = html.Thead(
        html.Tr(
            [
                html.Th("Artist"),
                html.Th("Title"),
                html.Th("Avg X"),
                html.Th("Peak X"),
                html.Th("Avg Q"),
                html.Th("Peak Q"),
                html.Th("Avg HCE"),
                html.Th("Peak HCE"),
            ]
        )
    )
    rows = []
    for _, r in df.iterrows():
        rows.append(
            html.Tr(
                [
                    html.Td(r.get("artist", "")),
                    html.Td(r.get("title", "")),
                    html.Td(f"{r.get('avg_x', 0):.3f}"),
                    html.Td(f"{r.get('peak_x', 0):.3f}"),
                    html.Td(f"{r.get('avg_q', 0):.3f}"),
                    html.Td(f"{r.get('peak_q', 0):.3f}"),
                    html.Td(f"{r.get('avg_hce', 0):.3f}"),
                    html.Td(f"{r.get('peak_hce', 0):.3f}"),
                ]
            )
        )
    body = html.Tbody(rows)
    return dbc.Table([header, body], striped=True, bordered=False, hover=True, size="sm")


@callback(
    Output("si-top20", "children"),
    Input("si-induce-toggle", "value"),
)
def update_top20(kinds):
    kinds = kinds or []
    frames = []
    for k in kinds:
        df = _top_tracks(k, limit=20)
        if not df.empty:
            df = df.assign(kind=k)
            frames.append(df)
    if not frames:
        return html.Div("No inducing tracks yet.", className="text-muted")
    # Sort based on toggle priority: order of kinds list
    combined = pd.concat(frames, ignore_index=True)
    if "trans" in kinds:
        combined = combined.sort_values(["peak_hce", "avg_hce"], ascending=False)
    elif "flow" in kinds:
        combined = combined.sort_values(["avg_x", "peak_x"], ascending=False)
    elif "relax" in kinds:
        combined = combined.sort_values(["avg_q", "peak_q"], ascending=True)
    return _table_from_df(combined.head(20))


@callback(
    Output("si-top100", "children"),
    Input("si-top100-toggle", "value"),
)
def update_top100(metric):
    df = _top_metrics(metric or "hce_mean", limit=100)
    if df.empty:
        return html.Div("No tracks yet.", className="text-muted")
    return _table_from_df(df)
