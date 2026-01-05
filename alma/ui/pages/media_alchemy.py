import datetime as dt
import sqlite3
from typing import List, Dict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from alma import config
from alma.engine import storage


def _fetch_track_sections(track_id: str) -> pd.DataFrame:
    try:
        rows = storage.list_track_sections(track_id, limit_sessions=1)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["section_rel_start"] = df["section_start_ts"] - df["start_ts"]
        df["section_rel_end"] = df["section_end_ts"] - df["start_ts"]
        return df
    except Exception:
        return pd.DataFrame()


def _track_options() -> List[Dict[str, str]]:
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT DISTINCT track_id, title, artist FROM track_sessions WHERE track_id IS NOT NULL ORDER BY start_ts DESC LIMIT 100",
                conn,
            )
        return [{"label": f"{r['title']} — {r['artist']}", "value": r["track_id"]} for _, r in df.iterrows()] if not df.empty else []
    except Exception:
        return []


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Media Alchemy — Intra-Track Resonance"),
                dbc.CardBody(
                    [
                        dcc.Dropdown(id="ma-track-select", placeholder="Select a track", className="mb-2"),
                        dcc.Graph(id="ma-track-plot"),
                        html.Div(id="ma-track-table", className="mt-2"),
                        html.Div(id="ma-status", className="small text-muted mt-1"),
                        dcc.Interval(id="ma-interval", interval=7000, n_intervals=0),
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
    Output("ma-track-select", "options"),
    Output("ma-track-select", "value"),
    Input("ma-interval", "n_intervals"),
)
def _load_options(_n):
    opts = _track_options()
    return opts, (opts[0]["value"] if opts else None)


@callback(
    Output("ma-track-plot", "figure"),
    Output("ma-track-table", "children"),
    Output("ma-status", "children"),
    Input("ma-track-select", "value"),
)
def _render_track(track_id):
    if not track_id:
        return go.Figure(), html.Div("Select a track to view."), ""
    df = _fetch_track_sections(track_id)
    if df.empty:
        return go.Figure(), html.Div("No section data yet (playback or analysis needed)."), "No analysis data."
    avg_hce = df["mean_HCE"].mean() if not df.empty else 0.0
    fig = go.Figure()
    # Placeholder waveform-style backdrop
    try:
        max_rel = df["section_rel_end"].max()
    except Exception:
        max_rel = 0
    t = np.linspace(0, max(max_rel, 1.0), num=300)
    waveform = 0.6 * np.sin(2 * np.pi * t / max(max_rel, 1.0) * 3) + 0.25 * np.sin(2 * np.pi * t / max(max_rel, 1.0) * 7)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=waveform,
            mode="lines",
            line=dict(color="rgba(120,120,120,0.15)"),
            name="waveform",
            hoverinfo="skip",
        )
    )
    for _, r in df.iterrows():
        fig.add_vrect(
            x0=r["section_rel_start"],
            x1=r["section_rel_end"],
            fillcolor="#333",
            opacity=0.08,
            line_width=0,
            annotation_text=r.get("section_label", ""),
            annotation_position="top left",
        )
    fig.add_trace(
        go.Scatter(
            x=df["section_rel_start"],
            y=df["mean_HCE"],
            mode="lines+markers",
            line=dict(color="#d7b34d", width=3),
            marker=dict(size=10, color=df["mean_Q"], colorscale="Plasma", showscale=True, colorbar=dict(title="Q")),
            name="HCE",
            hovertext=[
                f"{lbl}: HCE {h:.2f}, lift {h-avg_hce:+.2f}, X {x:.3f}, Q {q:.3f}"
                for lbl, h, x, q in zip(df["section_label"], df["mean_HCE"], df["mean_X"], df["mean_Q"])
            ],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["section_rel_start"],
            y=df["mean_Q"],
            mode="lines",
            line=dict(color="rgba(80,180,255,0.35)"),
            fill="tozeroy",
            name="Q",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["section_rel_start"],
            y=df["mean_X"],
            mode="lines",
            line=dict(color="rgba(150,150,150,0.35)", dash="dot"),
            name="X",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Sections — HCE/Q/X with lift (waveform-style)",
        xaxis_title="Seconds (relative)",
        yaxis_title="mean_HCE",
        height=460,
    )
    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Section"),
                        html.Th("Start (s)"),
                        html.Th("End (s)"),
                        html.Th("HCE"),
                        html.Th("Lift vs avg"),
                        html.Th("Q"),
                        html.Th("X"),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(r.get("section_label")),
                            html.Td(f"{r.get('section_rel_start',0):.1f}"),
                            html.Td(f"{r.get('section_rel_end',0):.1f}"),
                            html.Td(f"{r.get('mean_HCE',0):.2f}"),
                            html.Td(f"{(r.get('mean_HCE',0)-avg_hce):+.2f}"),
                            html.Td(f"{r.get('mean_Q',0):.3f}"),
                            html.Td(f"{r.get('mean_X',0):.3f}"),
                        ]
                    )
                    for _, r in df.iterrows()
                ]
            ),
        ],
        striped=True,
        bordered=False,
        hover=True,
        size="sm",
        className="mt-2",
    )
    return fig, table, ""

