import datetime as dt
import sqlite3
from typing import Dict, List

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, callback, dcc, html

from alma import config


def _fetch_track_sessions(limit: int = 200) -> pd.DataFrame:
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    session_id,
                    track_id,
                    title,
                    artist,
                    album,
                    start_ts,
                    end_ts,
                    mean_HCE,
                    mean_Q,
                    mean_X,
                    (coalesce(end_ts, start_ts) - start_ts) AS duration
                FROM track_sessions
                WHERE mean_HCE IS NOT NULL
                ORDER BY mean_HCE DESC
                LIMIT ?
                """,
                conn,
                params=(limit,),
            )
        return df
    except Exception:
        return pd.DataFrame()

def _proxy_audio_meta(title: str, artist: str) -> Dict[str, str]:
    """Lightweight heuristic from metadata for genre/tempo placeholders."""
    text = f"{title} {artist}".lower()
    genre = "unknown"
    tempo = "mid"
    if any(k in text for k in ["lofi", "chill", "ambient"]):
        genre = "chill"
        tempo = "slow"
    elif any(k in text for k in ["techno", "edm", "house", "dance"]):
        genre = "electronic"
        tempo = "fast"
    elif any(k in text for k in ["jazz", "bossa", "samba"]):
        genre = "jazz"
        tempo = "mid"
    elif any(k in text for k in ["classical", "piano", "suite", "concerto"]):
        genre = "classical"
        tempo = "slow"
    elif any(k in text for k in ["hip hop", "rap", "trap"]):
        genre = "hiphop"
        tempo = "mid"
    return {"genre": genre, "tempo": tempo}

def _suggest_next_track(df: pd.DataFrame) -> Dict[str, str]:
    """Pick the highest mean_HCE track not currently playing (naive ranking)."""
    if df.empty:
        return {}
    df_valid = df[df["mean_HCE"].notna()].sort_values("mean_HCE", ascending=False)
    if df_valid.empty:
        return {}
    top = df_valid.iloc[0]
    meta = _proxy_audio_meta(top.get("title", ""), top.get("artist", ""))
    return {
        "title": top.get("title", ""),
        "artist": top.get("artist", ""),
        "mean_HCE": f"{top.get('mean_HCE', 0):.2f}",
        "genre": meta["genre"],
        "tempo": meta["tempo"],
    }


def _compute_means(ts0: float, ts1: float, session_id: str) -> Dict[str, float]:
    from alma.engine import storage  # local import to avoid circular

    buckets = storage.get_buckets_between(ts0, ts1, session_id=session_id or None)
    if not buckets:
        return {"mean_HCE": 0.0, "mean_Q": 0.0, "mean_X": 0.0}
    return {
        "mean_HCE": float(np.nanmean([b.get("mean_HCE", 0.0) for b in buckets])),
        "mean_Q": float(np.nanmean([b.get("mean_Q", 0.0) for b in buckets])),
        "mean_X": float(np.nanmean([b.get("mean_X", 0.0) for b in buckets])),
    }


def _agg_tracks(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if df.empty:
        return df
    g = (
        df.groupby(["track_id", "title", "artist", "album"])
        .agg(
            avg_hce=("mean_HCE", "mean"),
            avg_q=("mean_Q", "mean"),
            avg_x=("mean_X", "mean"),
            play_count=("track_id", "count"),
            total_duration=("duration", "sum"),
        )
        .reset_index()
        .sort_values("avg_hce", ascending=False)
    )
    return g.head(top_n)


def _make_tables(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div("No track sessions yet.", className="text-muted")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            html.Tr(
                [
                    html.Td(r.get("artist", "")),
                    html.Td(r.get("title", "")),
                    html.Td(f"{r.get('avg_hce', 0):.2f}"),
                    html.Td(f"{r.get('avg_q', 0):.3f}"),
                    html.Td(f"{r.get('avg_x', 0):.3f}"),
                    html.Td(int(r.get("play_count", 0))),
                    html.Td(f"{r.get('total_duration', 0)/60:.1f}m"),
                ]
            )
        )
    return dbc.Table(
        [html.Thead(html.Tr([html.Th("Artist"), html.Th("Title"), html.Th("mean_HCE"), html.Th("mean_Q"), html.Th("mean_X"), html.Th("Plays"), html.Th("Duration")]))]
        + [html.Tbody(rows)],
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
    )


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Spotify Resonance"),
                dbc.CardBody(
                    [
                        html.Div(id="sr-suggestion", className="mb-3 fw-bold"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("HCE threshold"),
                                        dcc.Slider(id="sr-thr", min=0, max=5, step=0.5, value=0.0, marks=None),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                            ],
                            className="g-2 mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="sr-hist"), md=6, sm=12),
                                dbc.Col(dcc.Graph(id="sr-artist"), md=6, sm=12),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="sr-bar"), md=6, sm=12),
                                dbc.Col(dcc.Graph(id="sr-scatter"), md=6, sm=12),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="sr-timeline"), md=8, sm=12),
                                dbc.Col(dcc.Graph(id="sr-corr"), md=4, sm=12),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Key Findings", className="fw-bold mb-2"),
                                    html.Div(id="sr-findings"),
                                ]
                            ),
                            className="page-card mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Top 20 by mean_HCE", className="fw-bold mb-2"),
                                        html.Div(id="sr-top-table"),
                                    ],
                                    md=12,
                                ),
                            ],
                            className="mb-3",
                        ),
                        dcc.Interval(id="sr-interval", interval=5000, n_intervals=0),
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
    Output("sr-hist", "figure"),
    Output("sr-artist", "figure"),
    Output("sr-bar", "figure"),
    Output("sr-scatter", "figure"),
    Output("sr-timeline", "figure"),
    Output("sr-corr", "figure"),
    Output("sr-findings", "children"),
    Output("sr-top-table", "children"),
    Output("sr-suggestion", "children"),
    Input("sr-interval", "n_intervals"),
    Input("sr-thr", "value"),
)
def update_resonance(_n, thr):
    df = _fetch_track_sessions(limit=400)
    if thr is None:
        thr = 0.0
    if not df.empty and thr > 0:
        df = df[df["mean_HCE"] >= float(thr)]
    # Backfill missing means if any
    if not df.empty:
        need = df[(df["mean_HCE"].isna()) | (df["mean_HCE"] == 0.0)]
        if not need.empty:
            updated_rows = []
            with sqlite3.connect(config.DB_PATH) as conn:
                for idx, row in need.iterrows():
                    start_ts = row.get("start_ts") or 0.0
                    end_ts = row.get("end_ts") or start_ts
                    means = _compute_means(start_ts, end_ts, row.get("session_id") or None)
                    df.at[idx, "mean_HCE"] = means["mean_HCE"]
                    df.at[idx, "mean_Q"] = means["mean_Q"]
                    df.at[idx, "mean_X"] = means["mean_X"]
                    conn.execute(
                        "UPDATE track_sessions SET mean_HCE=?, mean_Q=?, mean_X=? WHERE track_id=? AND start_ts=?",
                        (means["mean_HCE"], means["mean_Q"], means["mean_X"], row.get("track_id"), start_ts),
                    )
                conn.commit()

    agg = _agg_tracks(df, top_n=20)

    # HCE histogram
    if not df.empty:
        hist_fig = px.histogram(df, x="mean_HCE", nbins=30, title="HCE distribution")
        hist_fig.update_layout(template="plotly_dark", height=360, bargap=0.05)
    else:
        hist_fig = px.histogram(title="HCE distribution")
        hist_fig.update_layout(template="plotly_dark", height=360)

    # Artist leaders
    if not df.empty:
        artist = (
            df.groupby("artist")
            .agg(avg_hce=("mean_HCE", "mean"), plays=("track_id", "count"))
            .reset_index()
            .sort_values("avg_hce", ascending=False)
            .head(10)
        )
        artist_fig = px.bar(
            artist,
            x="avg_hce",
            y="artist",
            color="plays",
            orientation="h",
            title="Top artists by mean_HCE",
            labels={"avg_hce": "mean_HCE", "plays": "play count"},
        )
        artist_fig.update_layout(template="plotly_dark", height=360)
    else:
        artist_fig = px.bar(title="Top artists by mean_HCE")
        artist_fig.update_layout(template="plotly_dark", height=360)

    # Bar top 10
    if not agg.empty:
        bar_fig = px.bar(
            agg.head(10),
            x="avg_hce",
            y="title",
            color="artist",
            orientation="h",
            title="Top 10 mean_HCE tracks",
            labels={"avg_hce": "mean_HCE", "title": "Track"},
        )
        bar_fig.update_layout(template="plotly_dark", height=420)
    else:
        bar_fig = px.bar(title="Top 10 mean_HCE tracks")
        bar_fig.update_layout(template="plotly_dark", height=420)

    # Scatter HCE vs Q
    if not agg.empty:
        scatter_fig = px.scatter(
            agg,
            x="avg_q",
            y="avg_hce",
            size="play_count",
            color="artist",
            hover_data=["title"],
            title="mean_HCE vs mean_Q",
            labels={"avg_q": "mean_Q", "avg_hce": "mean_HCE"},
        )
        scatter_fig.update_layout(template="plotly_dark", height=420)
    else:
        scatter_fig = px.scatter(title="mean_HCE vs mean_Q")
        scatter_fig.update_layout(template="plotly_dark", height=420)

    # Timeline of sessions
    if not df.empty:
        tl_fig = px.scatter(
            df.head(500),
            x=df["start_ts"].apply(lambda t: dt.datetime.fromtimestamp(t)),
            y="mean_HCE",
            color="title",
            title="Playback timeline (mean_HCE)",
            labels={"mean_HCE": "mean_HCE"},
        )
        tl_fig.update_layout(template="plotly_dark", height=360)
    else:
        tl_fig = px.scatter(title="Playback timeline (mean_HCE)")
        tl_fig.update_layout(template="plotly_dark", height=360)

    # Correlation heatmap
    corr_mat = np.zeros((3, 3))
    if not df.empty and df[["mean_HCE", "mean_Q", "mean_X"]].dropna().shape[0] > 1:
        corr_mat = np.corrcoef(df[["mean_HCE", "mean_Q", "mean_X"]].dropna().T)
    corr_fig = px.imshow(
        corr_mat,
        x=["HCE", "Q", "X"],
        y=["HCE", "Q", "X"],
        text_auto=".2f",
        title="Metric correlations during playback",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    corr_fig.update_layout(template="plotly_dark", height=320)

    # Findings
    if agg.empty:
        findings = "No track sessions yet."
    else:
        top = agg.iloc[0]
        findings = f"Highest resonance: {top.get('title','?')} by {top.get('artist','?')} — mean_HCE {top.get('avg_hce',0):.2f}."

    # Table
    table = _make_tables(agg)

    suggestion = ""
    s = _suggest_next_track(df)
    if s:
        suggestion = f"Suggested next: {s['title']} — {s['artist']} (mean_HCE {s['mean_HCE']}, {s['genre']}, {s['tempo']})"

    return hist_fig, artist_fig, bar_fig, scatter_fig, tl_fig, corr_fig, findings, table, suggestion

