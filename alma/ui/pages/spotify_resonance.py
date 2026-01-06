import datetime as dt
import sqlite3
from typing import Dict, List, Optional

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, State, callback, dcc, html

from alma import config
from alma.engine import storage


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


def _fetch_track_sections(track_id: str, limit_sessions: int = 1) -> pd.DataFrame:
    try:
        rows = storage.list_track_sections(track_id, limit_sessions=limit_sessions)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["section_rel_start"] = df["section_start_ts"] - df["start_ts"]
        df["section_rel_end"] = df["section_end_ts"] - df["start_ts"]
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


def _fetch_relax_tracks(limit: int = 20) -> pd.DataFrame:
    """Tracks that coincide with relaxed buckets (low volatility, moderate X, low HCE spikes)."""
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            q = """
            SELECT
                ts.track_id,
                ts.title,
                ts.artist,
                ts.album,
                AVG(b.mean_HCE) AS mean_HCE,
                AVG(b.mean_Q) AS mean_Q,
                AVG(b.mean_X) AS mean_X,
                COUNT(*) AS bucket_hits,
                SUM(COALESCE(ts.end_ts, ts.start_ts) - ts.start_ts) AS total_duration
            FROM track_sessions ts
            JOIN buckets b
              ON b.bucket_start_ts >= ts.start_ts
             AND b.bucket_start_ts <= COALESCE(ts.end_ts, ts.start_ts)
            WHERE b.std_Q < 0.10
              AND b.mean_X BETWEEN 1.6 AND 1.8
              AND b.mean_HCE < 50.0
              AND COALESCE(b.valid_fraction, 1.0) >= 0.9
              AND ABS(COALESCE(b.Q_slope, 0.0)) <= 0.01
            GROUP BY ts.track_id, ts.title, ts.artist, ts.album
            ORDER BY bucket_hits DESC, mean_HCE ASC
            LIMIT ?
            """
            df = pd.read_sql_query(q, conn, params=(limit,))
            return df
    except Exception:
        return pd.DataFrame()


def _relax_table(df: pd.DataFrame) -> html.Div:
    if df.empty:
        return html.Div("No relaxed-matching tracks yet.", className="text-muted")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            html.Tr(
                [
                    html.Td(r.get("artist", "")),
                    html.Td(r.get("title", "")),
                    html.Td(f"{r.get('mean_HCE', 0):.2f}"),
                    html.Td(f"{r.get('mean_Q', 0):.3f}"),
                    html.Td(f"{r.get('mean_X', 0):.3f}"),
                    html.Td(int(r.get("bucket_hits", 0))),
                    html.Td(f"{(r.get('total_duration', 0) or 0)/60:.1f}m"),
                ]
            )
        )
    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Artist"),
                        html.Th("Title"),
                        html.Th("mean_HCE"),
                        html.Th("mean_Q"),
                        html.Th("mean_X"),
                        html.Th("Occurrences"),
                        html.Th("Duration"),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        className="mt-2",
    )


def _search_tracks(query: str, limit: int = 50) -> List[Dict[str, str]]:
    if not query:
        return []
    q = f"%{query.lower()}%"
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            df = pd.read_sql_query(
                """
                SELECT DISTINCT track_id, title, artist
                FROM track_sessions
                WHERE track_id IS NOT NULL
                  AND (lower(title) LIKE ? OR lower(artist) LIKE ?)
                ORDER BY MAX(start_ts) DESC
                LIMIT ?
                """,
                conn,
                params=(q, q, limit),
            )
        return [{"label": f"{r['title']} — {r['artist']}", "value": r["track_id"]} for _, r in df.iterrows()] if not df.empty else []
    except Exception:
        return []


@callback(
    Output("sr-search-results", "options"),
    Output("sr-search-results", "value"),
    Input("sr-search-text", "value"),
)
def _search_options_sr(query):
    opts = _search_tracks(query or "")
    return opts, None


    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Artist"),
                        html.Th("Title"),
                        html.Th("mean_HCE"),
                        html.Th("mean_Q"),
                        html.Th("mean_X"),
                        html.Th("Occurrences"),
                        html.Th("Duration"),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
    )

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
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="sr-top-sections"), md=12, sm=12),
            ],
            className="g-3 mb-3",
        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Intra-Track Analysis", className="fw-bold mb-2"),
                                    dcc.Dropdown(id="sr-track-select", placeholder="Select a track", className="mb-2"),
                                    dbc.Input(id="sr-search-text", placeholder="Search all tracks (title/artist)", className="mb-2", debounce=True),
                                    dcc.Dropdown(id="sr-search-results", placeholder="Search results", className="mb-3"),
                                    dcc.Graph(id="sr-track-sections"),
                                    html.Div(id="sr-track-table", className="mt-2"),
                                ]
                            ),
                            className="page-card mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Relax-Inducing Tracks", className="fw-bold mb-2"),
                                        html.Div(id="sr-relax-table"),
                                    ],
                                    md=7,
                                    sm=12,
                                ),
                                dbc.Col(
                                    dcc.Graph(id="sr-relax-bar"),
                                    md=5,
                                    sm=12,
                                ),
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
    Output("sr-top-sections", "figure"),
    Output("sr-findings", "children"),
    Output("sr-top-table", "children"),
    Output("sr-suggestion", "children"),
    Output("sr-relax-table", "children"),
    Output("sr-relax-bar", "figure"),
    Output("sr-track-select", "options"),
    Output("sr-track-select", "value"),
    Input("sr-interval", "n_intervals"),
    Input("sr-thr", "value"),
    Input("sr-search-results", "value"),
    State("sr-track-select", "value"),
)
def update_resonance(_n, thr, search_choice, current_selection):
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

    # Top sections across history
    sec_rows = storage.list_top_sections(limit=10)
    if sec_rows:
        sec_df = pd.DataFrame(sec_rows)
        top_sec_fig = px.bar(
            sec_df,
            x="avg_hce",
            y="section_label",
            color="plays",
            orientation="h",
            title="Top sections by avg HCE",
            labels={"avg_hce": "avg HCE", "section_label": "Section", "plays": "plays"},
        )
        top_sec_fig.update_layout(template="plotly_dark", height=360)
    else:
        top_sec_fig = px.bar(title="Top sections by avg HCE")
        top_sec_fig.update_layout(template="plotly_dark", height=360)

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

    relax_df = _fetch_relax_tracks(limit=20)
    relax_table = _relax_table(relax_df)
    if not relax_df.empty:
        relax_bar = px.bar(
            relax_df.head(10),
            x="bucket_hits",
            y="title",
            color="artist",
            orientation="h",
            title="Top relaxation inducers (consistency)",
            labels={"bucket_hits": "Relax occurrences"},
            hover_data={"mean_HCE": ":.2f", "mean_Q": ":.3f", "mean_X": ":.3f"},
        )
        relax_bar.update_layout(template="plotly_dark", height=400)
    else:
        relax_bar = px.bar(title="Top relaxation inducers (consistency)")
        relax_bar.update_layout(template="plotly_dark", height=400)

    options = [{"label": f"{r['title']} — {r['artist']}", "value": r["track_id"]} for _, r in df.head(150).iterrows()] if not df.empty else []
    if search_choice and any(o["value"] == search_choice for o in options):
        selected = search_choice
    elif current_selection and any(o["value"] == current_selection for o in options):
        selected = current_selection
    else:
        selected = options[0]["value"] if options else None

    return (
        hist_fig,
        artist_fig,
        bar_fig,
        scatter_fig,
        tl_fig,
        corr_fig,
        top_sec_fig,
        findings,
        table,
        suggestion,
        relax_table,
        relax_bar,
        options,
        selected,
    )


@callback(
    Output("sr-track-sections", "figure"),
    Output("sr-track-table", "children"),
    Input("sr-track-select", "value"),
    prevent_initial_call=False,
)
def render_sr_track_sections(track_id):
    if not track_id:
        placeholder = go.Figure()
        placeholder.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 0],
                mode="lines",
                line=dict(color="rgba(200,200,200,0.1)", dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        placeholder.update_layout(
            template="plotly_dark",
            title="Resonance awakening — begin playback to map live HCE",
            xaxis_title="Seconds (relative)",
            yaxis_title="HCE",
            height=420,
        )
        return placeholder, html.Div("Select a track to view sections.")
    df = _fetch_track_sections(track_id)
    source_note = "unknown"
    if "source" in df.columns and not df["source"].empty:
        mode_vals = df["source"].mode()
        if not mode_vals.empty:
            source_note = mode_vals.iloc[0]
    if df.empty or df[["mean_HCE", "mean_Q", "mean_X"]].fillna(0).sum().sum() == 0:
        placeholder = go.Figure()
        placeholder.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 0],
                mode="lines",
                line=dict(color="rgba(200,200,200,0.1)", dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        placeholder.update_layout(
            template="plotly_dark",
            title=f"Resonance awakening — begin playback to map live HCE (source={source_note})",
            xaxis_title="Seconds (relative)",
            yaxis_title="HCE",
            height=420,
        )
        return placeholder, html.Div(f"No data yet (source={source_note}).")
    first_session = df["track_session_id"].iloc[0]
    df = df[df["track_session_id"] == first_session]
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
    # vrects for sections
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
    # HCE line
    fig.add_trace(
        go.Scatter(
            x=df["section_rel_start"],
            y=df["mean_HCE"],
            mode="lines+markers",
            line=dict(color="#d7b34d", width=3),
            marker=dict(size=10, color=df["mean_Q"], colorscale="Plasma", showscale=True, colorbar=dict(title="Q")),
            name="HCE",
            hovertext=[
                f"{lbl}: HCE {h:.2f}, lift {h-avg_hce:+.2f}, X {x:.3f}, Q {q:.3f}, src {s}"
                for lbl, h, x, q, s in zip(
                    df["section_label"],
                    df["mean_HCE"],
                    df["mean_X"],
                    df["mean_Q"],
                    df["source"] if "source" in df.columns else [""] * len(df),
                )
            ],
        )
    )
    # Q fill
    fig.add_trace(
        go.Scatter(
            x=df["section_rel_start"],
            y=df["mean_Q"],
            mode="lines",
            line=dict(color="rgba(80,180,255,0.4)"),
            fill="tozeroy",
            name="Q",
        )
    )
    # X background
    fig.add_trace(
        go.Scatter(
            x=df["section_rel_start"],
            y=df["mean_X"],
            mode="lines",
            line=dict(color="rgba(150,150,150,0.3)", dash="dot"),
            name="X",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Intra-Track Sections — HCE/Q/X",
        xaxis_title="Seconds (relative)",
        yaxis_title="mean_HCE",
        height=420,
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
                        html.Th("Source"),
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
                            html.Td(r.get("source", "")),
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
    return fig, table

