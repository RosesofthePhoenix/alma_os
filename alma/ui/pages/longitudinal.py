import datetime as dt
import io
import json
import sqlite3
from typing import Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, State, callback, ctx, dcc, html

from alma import config


def _fetch_buckets() -> pd.DataFrame:
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            return pd.read_sql_query(
                """
                SELECT bucket_start_ts, mean_X, mean_Q, mean_HCE, std_Q, Q_slope, valid_fraction
                FROM buckets
                WHERE bucket_start_ts IS NOT NULL
                """,
                conn,
            )
    except Exception:
        return pd.DataFrame()


def _fetch_events() -> pd.DataFrame:
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            return pd.read_sql_query(
                """
                SELECT ts, note, tags_json, context_json
                FROM events
                ORDER BY ts DESC
                """,
                conn,
            )
    except Exception:
        return pd.DataFrame()


def _fetch_tracks(limit: int = 200) -> pd.DataFrame:
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            return pd.read_sql_query(
                """
                SELECT track_id, title, artist, mean_HCE, mean_Q, mean_X, start_ts, end_ts
                FROM track_sessions
                WHERE mean_HCE IS NOT NULL
                ORDER BY mean_HCE DESC
                LIMIT ?
                """,
                conn,
                params=(limit,),
            )
    except Exception:
        return pd.DataFrame()


def _media_alchemy(tracks: pd.DataFrame, buckets: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame()
    baseline = buckets["mean_HCE"].median() if not buckets.empty else 0.0
    df = tracks.copy()
    df["hce_lift"] = df["mean_HCE"] - baseline
    df = df.sort_values("hce_lift", ascending=False)
    return df[["title", "artist", "mean_HCE", "hce_lift", "mean_Q", "mean_X"]].head(20)


def _circadian_map(buckets: pd.DataFrame) -> px.imshow:
    if buckets.empty:
        return px.imshow(np.zeros((1, 24)), labels=dict(x="Hour", y="", color="mean_HCE"), title="Circadian transcendence map (no data)")
    buckets = buckets.copy()
    buckets["dt"] = buckets["bucket_start_ts"].apply(lambda t: dt.datetime.fromtimestamp(t))
    buckets["hour"] = buckets["dt"].dt.hour
    grouped = buckets.groupby("hour")["mean_HCE"].mean().reindex(range(24), fill_value=0)
    fig = px.imshow(
        np.array([grouped.values]),
        labels=dict(x="Hour of day", y="", color="mean_HCE"),
        x=list(range(24)),
        y=["HCE"],
        title="Circadian transcendence map",
        aspect="auto",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=40, b=30))
    return fig


def _social_split(events: pd.DataFrame, buckets: pd.DataFrame) -> pd.DataFrame:
    if events.empty or buckets.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for _, r in events.iterrows():
        ctx_raw = r.get("context_json") or "{}"
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except Exception:
            ctx = {}
        social = ctx.get("social") or ctx.get("home_bookmark_social") or "unknown"
        ts = r.get("ts") or 0
        nearby = buckets[(buckets["bucket_start_ts"] >= ts - 600) & (buckets["bucket_start_ts"] <= ts + 600)]
        if nearby.empty:
            continue
        rows.append({"social": social, "mean_HCE": nearby["mean_HCE"].mean()})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby("social")["mean_HCE"].mean().reset_index().sort_values("mean_HCE", ascending=False)


def _intention_loops(events: pd.DataFrame, buckets: pd.DataFrame) -> pd.DataFrame:
    if events.empty or buckets.empty:
        return pd.DataFrame()
    intents: List[Dict[str, object]] = []
    for _, r in events.iterrows():
        ctx_raw = r.get("context_json") or "{}"
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
        except Exception:
            ctx = {}
        intention = ctx.get("intention") or ""
        if not intention:
            continue
        ts = r.get("ts") or 0
        window = buckets[(buckets["bucket_start_ts"] >= ts) & (buckets["bucket_start_ts"] <= ts + 900)]
        if window.empty:
            continue
        intents.append(
            {
                "intention": intention,
                "mean_HCE": window["mean_HCE"].mean(),
                "mean_Q": window["mean_Q"].mean(),
                "mean_X": window["mean_X"].mean(),
            }
        )
    if not intents:
        return pd.DataFrame()
    return pd.DataFrame(intents).sort_values("mean_HCE", ascending=False).head(10)


def _fetch_fractal_data(window: Optional[Tuple[float, float]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    buckets = _fetch_buckets()
    events = _fetch_events()
    if window and not buckets.empty:
        start, end = window
        buckets = buckets[(buckets["bucket_start_ts"] >= start) & (buckets["bucket_start_ts"] <= end)]
        events = events[(events["ts"] >= start) & (events["ts"] <= end)]
    return buckets, events


def _build_fractal_fig(buckets: pd.DataFrame, events: pd.DataFrame) -> go.Figure:
    if buckets.empty:
        return go.Figure()
    df = buckets.copy()
    df["dt"] = df["bucket_start_ts"].apply(lambda t: dt.datetime.fromtimestamp(t))
    hover = (
        "Time=%{x|%Y-%m-%d %H:%M}<br>HCE=%{y:.2f}<br>X=%{customdata[0]:.3f}"
        "<br>std_Q=%{marker.size:.3f}<br>Q_slope=%{customdata[1]:.3f}"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=df["dt"],
            y=df["mean_HCE"],
            mode="markers",
            marker=dict(
                size=np.clip(df["std_Q"] * 120, 6, 28),
                color=df["mean_X"],
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="X"),
                opacity=0.75,
            ),
            customdata=np.stack([df["mean_X"], df["Q_slope"]], axis=1),
            hovertemplate=hover,
            name="Buckets",
        )
    )
    if not events.empty:
        ev = events.copy()
        ev["dt"] = ev["ts"].apply(lambda t: dt.datetime.fromtimestamp(t))
        notes = []
        for _, r in ev.iterrows():
            tags_txt = ""
            try:
                tags = json.loads(r.get("tags_json") or "{}") if isinstance(r.get("tags_json"), str) else r.get("tags_json") or {}
                social = tags.get("social") or tags.get("home_bookmark_social") or ""
                act = tags.get("activity") or ""
                mood = tags.get("mood")
                bits = [r.get("note") or ""]
                if act:
                    bits.append(f"act:{act}")
                if social:
                    bits.append(f"soc:{social}")
                if mood is not None:
                    bits.append(f"mood:{mood}")
                tags_txt = " | ".join([b for b in bits if b])
            except Exception:
                tags_txt = r.get("note") or ""
            notes.append(tags_txt)
        fig.add_trace(
            go.Scatter(
                x=ev["dt"],
                y=[df["mean_HCE"].max() * 1.05 if not df.empty else 0] * len(ev),
                mode="markers",
                marker=dict(symbol="star", size=10, color="#ff9f43", line=dict(color="#ffa", width=1)),
                text=notes,
                hovertemplate="%{text}<br>%{x|%Y-%m-%d %H:%M}",
                name="Events",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=10, r=10, t=40, b=40),
        title="Fractal Life Timeline",
        xaxis=dict(rangeslider=dict(visible=True), title="Time", showgrid=False),
        yaxis=dict(title="HCE"),
        dragmode="zoom",
    )
    return fig


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Longitudinal Insights"),
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "“Fractal echoes of your days reveal the larger harmony.”",
                                    className="text-info mb-1",
                                ),
                                html.Div(
                                    "“Every peak and valley sketches a chapter of the soul.”",
                                    className="text-warning small",
                                ),
                            ],
                            className="mb-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="li-media"), md=6, sm=12),
                                dbc.Col(dcc.Graph(id="li-circadian"), md=6, sm=12),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="li-social"), md=6, sm=12),
                                dbc.Col(dcc.Graph(id="li-intentions"), md=6, sm=12),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Fractal Life Chronicles", className="fw-bold mb-2"),
                                    dcc.Graph(id="li-fractal"),
                                    html.Div(id="li-fractal-narrative", className="mt-2"),
                                    dbc.Button("Export Chapter (PDF)", id="li-export-btn", color="secondary", size="sm", className="mt-2"),
                                    dcc.Download(id="li-export"),
                                ]
                            ),
                            className="page-card mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Oracle", className="fw-bold mb-2"),
                                    html.Div(id="li-oracle"),
                                    dbc.Button("Generate State Story", id="li-story-btn", color="info", size="sm", className="mt-2"),
                                    html.Div(id="li-story-text", className="mt-2"),
                                    dcc.Graph(id="li-story-art", className="mt-2"),
                                ]
                            ),
                            className="page-card",
                        ),
                        dcc.Interval(id="li-interval", interval=6000, n_intervals=0),
                        dcc.Store(id="li-fractal-window"),
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
    Output("li-media", "figure"),
    Output("li-circadian", "figure"),
    Output("li-social", "figure"),
    Output("li-intentions", "figure"),
    Output("li-oracle", "children"),
    Output("li-fractal", "figure"),
    Input("li-interval", "n_intervals"),
    Input("li-fractal-window", "data"),
)
def update_longitudinal(_n, window):
    buckets = _fetch_buckets()
    events = _fetch_events()
    tracks = _fetch_tracks()

    # Media alchemy
    media_df = _media_alchemy(tracks, buckets)
    if not media_df.empty:
        media_fig = px.bar(
            media_df.head(10),
            x="hce_lift",
            y="title",
            color="artist",
            orientation="h",
            labels={"hce_lift": "HCE lift", "title": "Track"},
            title="Top media by HCE lift",
            hover_data={"mean_HCE": ":.2f", "mean_Q": ":.3f", "mean_X": ":.3f"},
        )
        media_fig.update_layout(template="plotly_dark", height=420)
    else:
        media_fig = px.bar(title="Top media by HCE lift")
        media_fig.update_layout(template="plotly_dark", height=420)

    circadian_fig = _circadian_map(buckets)

    # Social split
    social_df = _social_split(events, buckets)
    if not social_df.empty:
        social_fig = px.bar(
            social_df,
            x="mean_HCE",
            y="social",
            orientation="h",
            title="Solitary vs relational harmony",
            labels={"mean_HCE": "mean_HCE", "social": "Context"},
        )
        social_fig.update_layout(template="plotly_dark", height=320)
    else:
        social_fig = px.bar(title="Solitary vs relational harmony")
        social_fig.update_layout(template="plotly_dark", height=320)

    # Intention-outcome
    intent_df = _intention_loops(events, buckets)
    if not intent_df.empty:
        intent_fig = px.bar(
            intent_df,
            x="mean_HCE",
            y="intention",
            orientation="h",
            title="Intention → outcome (HCE)",
            labels={"mean_HCE": "mean_HCE", "intention": "Intention"},
            hover_data={"mean_Q": ":.3f", "mean_X": ":.3f"},
        )
        intent_fig.update_layout(template="plotly_dark", height=360)
    else:
        intent_fig = px.bar(title="Intention → outcome (HCE)")
        intent_fig.update_layout(template="plotly_dark", height=360)

    # Oracle message
    msgs = []
    if not media_df.empty:
        top_media = media_df.iloc[0]
        msgs.append(f"Media alchemy: {top_media['title']} ({top_media['artist']}) shows strongest HCE lift.")
    if not social_df.empty:
        top_social = social_df.iloc[0]
        msgs.append(f"Context: {top_social['social']} associates with mean_HCE {top_social['mean_HCE']:.2f}.")
    if not intent_df.empty:
        top_int = intent_df.iloc[0]
        msgs.append(f"Intention payoff: \"{top_int['intention']}\" → HCE {top_int['mean_HCE']:.2f}.")
    if msgs:
        oracle = html.Ul([html.Li(m) for m in msgs])
    else:
        oracle = html.Div("Not enough history for oracle yet.", className="text-muted")

    # Fractal life figure
    fractal_fig = _build_fractal_fig(*_fetch_fractal_data(window))

    return media_fig, circadian_fig, social_fig, intent_fig, oracle, fractal_fig


@callback(
    Output("li-story-text", "children"),
    Output("li-story-art", "figure"),
    Input("li-story-btn", "n_clicks"),
    prevent_initial_call=True,
)
def generate_story(_n):
    buckets = _fetch_buckets()
    events = _fetch_events()
    tracks = _fetch_tracks(limit=100)

    if buckets.empty:
        return "Not enough data to generate a story yet.", px.scatter(title="State story art")

    # Peaks and calm
    top_peak = buckets.sort_values("mean_HCE", ascending=False).head(1)
    peak_txt = ""
    if not top_peak.empty:
        peak_ts = top_peak.iloc[0]["bucket_start_ts"]
        peak_val = top_peak.iloc[0]["mean_HCE"]
        peak_dt = dt.datetime.fromtimestamp(peak_ts).strftime("%a %H:%M")
        peak_txt = f"Peak HCE {peak_val:.2f} at {peak_dt}."

    # Relax track
    relax_df = _media_alchemy(tracks, buckets)
    relax_txt = ""
    if not relax_df.empty:
        r = relax_df.iloc[-1]
        relax_txt = f"Calm anchor: {r['title']} — {r['artist']} (HCE {r['mean_HCE']:.2f})."

    # Intention outcome
    intents = _intention_loops(events, buckets)
    intent_txt = ""
    if not intents.empty:
        i = intents.iloc[0]
        intent_txt = f"Intention payoff: \"{i['intention']}\" → HCE {i['mean_HCE']:.2f}."

    story_lines = ["Weekly state story:"]
    for t in [peak_txt, relax_txt, intent_txt]:
        if t:
            story_lines.append(t)
    if len(story_lines) == 1:
        story_lines.append("Patterns are forming—collect a bit more data.")

    # Data art: scatter of time vs HCE with size by mean_X and color by std_Q
    buckets_art = buckets.copy()
    buckets_art["dt"] = buckets_art["bucket_start_ts"].apply(lambda t: dt.datetime.fromtimestamp(t))
    art_fig = px.scatter(
        buckets_art.tail(500),
        x="dt",
        y="mean_HCE",
        size="mean_X",
        color="std_Q",
        color_continuous_scale="Viridis",
        title="Life as Data Art — HCE over time",
        labels={"mean_HCE": "HCE", "dt": "Time", "std_Q": "std_Q"},
    )
    art_fig.update_layout(template="plotly_dark", height=420)

    return html.Ul([html.Li(line) for line in story_lines]), art_fig


@callback(
    Output("li-fractal-narrative", "children"),
    Output("li-fractal-window", "data"),
    Input("li-fractal", "clickData"),
    Input("li-fractal", "relayoutData"),
    State("li-fractal-window", "data"),
    prevent_initial_call=True,
)
def update_fractal_narrative(click_data, relayout_data, window):
    # Handle zoom window from double-click or range selection
    if relayout_data and "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        try:
            start = dt.datetime.fromisoformat(relayout_data["xaxis.range[0]"]).timestamp()
            end = dt.datetime.fromisoformat(relayout_data["xaxis.range[1]"]).timestamp()
            window = (start, end)
        except Exception:
            pass
    elif relayout_data and ("xaxis.autorange" in relayout_data or relayout_data.get("autosize")):
        window = None

    if not click_data:
        return html.Div("Click a point to see its story."), window

    pt = click_data["points"][0]
    ts = pt["x"]
    hce = pt.get("y")
    cd = pt.get("customdata") or [None, None]
    x_val, q_slope = cd[0], cd[1]
    text = [
        f"Time: {ts}",
        f"HCE: {hce:.2f}" if hce is not None else "HCE: n/a",
        f"X: {x_val:.3f}" if x_val is not None else "X: n/a",
        f"Q_slope: {q_slope:.3f}" if q_slope is not None else "Q_slope: n/a",
    ]
    return html.Ul([html.Li(t) for t in text]), window


@callback(
    Output("li-export", "data"),
    Input("li-export-btn", "n_clicks"),
    State("li-fractal", "figure"),
    State("li-fractal-narrative", "children"),
    prevent_initial_call=True,
)
def export_chapter(n, fig_dict, narrative):
    if not n:
        raise dash.exceptions.PreventUpdate  # type: ignore
    try:
        fig = go.Figure(fig_dict)
        pdf_bytes = fig.to_image(format="pdf", height=800, width=1200)
    except Exception:
        # Fallback to PNG if pdf unavailable
        pdf_bytes = fig.to_image(format="png", height=800, width=1200)
    buf = io.BytesIO()
    buf.write(pdf_bytes)
    buf.seek(0)
    return dcc.send_bytes(lambda _: buf.read(), "chapter_of_the_soul.pdf")

