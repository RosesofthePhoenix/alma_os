import time
import datetime as dt
import sqlite3
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, ALL
import plotly.express as px
import plotly.graph_objects as go

from alma.engine import storage
from alma.app_state import registry
from alma import config


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_ts(txt: str) -> float | None:
    if not txt:
        return None
    try:
        # Accepts ISO; strips trailing Z if present
        clean = txt.strip().replace("Z", "")
        return dt.datetime.fromisoformat(clean).timestamp()
    except Exception:
        return None


def layout() -> dbc.Container:
    top_bar = dbc.Card(
        dbc.CardBody(
            [
                html.Div("Manifest — Alchemy Forge", className="h4 text-warning mb-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="manifest-target-state",
                                options=[
                                    {"label": "Flow State", "value": "flow"},
                                    {"label": "Relaxed Harmony", "value": "relax"},
                                    {"label": "Transcendent Harmony", "value": "transcend"},
                                    {"label": "Custom State", "value": "custom"},
                                ],
                                value="flow",
                                clearable=False,
                                style={"color": "#000"},
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            html.Div(id="manifest-state-details", className="small text-secondary"),
                            md=8,
                        ),
                    ],
                    className="gy-2 mb-2",
                ),
                html.Div(
                    id="manifest-custom-state",
                    style={"display": "none"},
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(dbc.Input(id="manifest-custom-name", placeholder="Custom state name", type="text"), md=3),
                                dbc.Col(dbc.Input(id="manifest-custom-duration", placeholder="Duration min (minutes)", type="number", min=0, step=1), md=2),
                            ],
                            className="gy-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div("X range", className="small"),
                                        dcc.RangeSlider(id="manifest-custom-x", min=0, max=3, step=0.05, value=[1.2, 1.8], marks=None),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Q range", className="small"),
                                        dcc.RangeSlider(id="manifest-custom-q", min=0, max=0.2, step=0.005, value=[0.02, 0.06], marks=None),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Div("HCE range", className="small"),
                                        dcc.RangeSlider(id="manifest-custom-hce", min=0, max=2, step=0.05, value=[0.4, 1.2], marks=None),
                                    ],
                                    md=4,
                                ),
                            ],
                            className="gy-2 mt-2",
                        ),
                    ],
                ),
                dbc.Row(
                    [
                        dbc.Col(dbc.Input(id="manifest-social", placeholder="Social (who/alone)", type="text"), md=3),
                        dbc.Col(dbc.Input(id="manifest-ambience", placeholder="Ambience (lighting/noise)", type="text"), md=3),
                        dbc.Col(dbc.Input(id="manifest-activity", placeholder="Activity (what are you doing)", type="text"), md=3),
                        dbc.Col(
                            dbc.Input(
                                id="manifest-minutes",
                                placeholder="Capture window (minutes, any length)",
                                type="number",
                                min=0,
                                step=1,
                            ),
                            md=3,
                        ),
                    ],
                    className="gy-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Input(id="manifest-start-ts", placeholder="Start timestamp (ISO)", type="text"),
                                dbc.Button("Set start to now", id="manifest-start-now", color="secondary", size="sm", className="mt-1"),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Input(id="manifest-end-ts", placeholder="Finish timestamp (ISO)", type="text"),
                                dbc.Button("Set finish to now", id="manifest-end-now", color="secondary", size="sm", className="mt-1"),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                dbc.Checklist(
                                    options=[{"label": "Add substance context", "value": "show"}],
                                    value=[],
                                    id="manifest-substance-toggle",
                                    switch=True,
                                ),
                                html.Div(
                                    id="manifest-substance-panel",
                                    style={"display": "none"},
                                    children=[
                                        html.Div("Substance context (0–10)", className="small text-muted mb-1"),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div("Cocaine highness", className="small"),
                                                        dcc.Slider(id="manifest-cocaine", min=0, max=10, step=1, value=0, marks=None),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div("Ketamine highness", className="small"),
                                                        dcc.Slider(id="manifest-ketamine", min=0, max=10, step=1, value=0, marks=None),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div("Tusi highness", className="small"),
                                                        dcc.Slider(id="manifest-tusi", min=0, max=10, step=1, value=0, marks=None),
                                                        dbc.Input(
                                                            id="manifest-tusi-source",
                                                            placeholder="Tusi source (optional)",
                                                            type="text",
                                                            className="mt-1",
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="gy-2",
                                        ),
                                    ],
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="gy-2 mt-2",
                ),
                dbc.Button("Log manifest entry", id="manifest-save", color="warning", className="mt-3"),
                html.Span(id="manifest-status", className="ms-2 text-success"),
            ]
        ),
        className="mb-3",
        style={"backgroundColor": "#111118", "border": "1px solid #333"},
    )

    body_sections = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Patterns & readiness (placeholder)", className="text-muted"),
                            html.Div(id="manifest-patterns", className="small text-secondary"),
                        ]
                    ),
                    style={"backgroundColor": "#0c0c12", "border": "1px solid #222"},
                ),
                md=6,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Calendar / schedule (placeholder)", className="text-muted"),
                            html.Div(id="manifest-calendar", className="small text-secondary"),
                        ]
                    ),
                    style={"backgroundColor": "#0c0c12", "border": "1px solid #222"},
                ),
                md=6,
            ),
        ],
        className="gy-3",
    )

    recipes_row = dbc.Card(
        dbc.CardBody(
            [
                html.Div("Induction Recipes", className="fw-bold text-warning mb-2"),
                html.Div("Auto-suggested from proven patterns", className="small text-muted"),
                html.Div(id="manifest-recipe-suggest", className="small text-secondary mb-2"),
                dbc.Row(
                    [
                        dbc.Col(dbc.Input(id="manifest-recipe-name", placeholder="Recipe name", type="text"), md=4),
                        dbc.Col(dbc.Input(id="manifest-recipe-notes", placeholder="Notes/context", type="text"), md=5),
                        dbc.Col(dbc.Input(id="manifest-recipe-duration", placeholder="Duration min", type="number", min=5, step=5, value=60), md=3),
                    ],
                    className="gy-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(dbc.Button("Save Recipe", id="manifest-recipe-save", color="secondary", size="sm"), width="auto"),
                        dbc.Col(dbc.Button("One-Click Apply", id="manifest-recipe-apply", color="warning", size="sm"), width="auto"),
                        dbc.Col(html.Span(id="manifest-recipe-status", className="ms-2 text-success"), width="auto"),
                    ],
                    className="gy-2 mt-2",
                ),
            ]
        ),
        style={"backgroundColor": "#0c0c12", "border": "1px solid #222"},
        className="mt-3",
    )

    stats_row = dbc.Card(
        dbc.CardBody(
            [
                html.Div("State stats", className="fw-bold text-warning mb-2"),
                dbc.Row(
                    [
                        dbc.Col(html.Div(id="manifest-stat-sessions", className="small"), md=3),
                        dbc.Col(html.Div(id="manifest-stat-hours", className="small"), md=3),
                        dbc.Col(html.Div(id="manifest-stat-percent", className="small"), md=3),
                        dbc.Col(html.Div(id="manifest-stat-portals", className="small"), md=3),
                    ],
                    className="gy-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="manifest-trend", figure=px.line()), md=6),
                        dbc.Col(
                            [
                                dbc.RadioItems(
                                    id="manifest-calendar-mode",
                                    options=[
                                        {"label": "Day", "value": "day"},
                                        {"label": "Week", "value": "week"},
                                        {"label": "Month", "value": "month"},
                                    ],
                                    value="day",
                                    inline=True,
                                    className="text-muted mb-1",
                                ),
                                dcc.Graph(id="manifest-calendar"),
                            ],
                            md=6,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="manifest-portal-trend"), md=6),
                        dbc.Col(
                            [
                                html.Div("Top Portal Days", className="fw-semibold mb-1"),
                                html.Div(id="manifest-portal-days", className="small text-secondary"),
                            ],
                            md=6,
                        ),
                    ],
                    className="mt-2",
                ),
                html.Div("Sessions on selected day", className="fw-semibold mt-2"),
                html.Div(id="manifest-day-list", className="small text-secondary"),
                html.Div("Top tracks & bookmarks in state", className="fw-semibold mt-2"),
                html.Div(id="manifest-top-tracks", className="small text-secondary"),
            ]
        ),
        style={"backgroundColor": "#0c0c12", "border": "1px solid #222"},
        className="mt-3",
    )

    return dbc.Container(
        [top_bar, body_sections, recipes_row, stats_row, _patterns_card()],
        fluid=True,
        className="page-container",
    )


def _patterns_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Induction Patterns from History", className="fw-bold text-warning mb-2"),
                html.Div("Proven Inducers (>=5 occurrences)", className="fw-semibold"),
                html.Div(id="manifest-proven", className="small text-secondary mb-2"),
                html.Div("Emerging Patterns", className="fw-semibold"),
                html.Div(id="manifest-emerging", className="small text-secondary mb-2"),
                dcc.Graph(id="manifest-sunburst", figure=px.sunburst()),
                html.Div("Oracle Mini-Chat (state-guided)", className="fw-semibold mt-3"),
                dcc.Textarea(
                    id="manifest-oracle-input",
                    placeholder="Ask for induction guidance...",
                    style={"width": "100%", "height": "80px"},
                ),
                dbc.Button("Ask Oracle", id="manifest-oracle-send", color="primary", size="sm", className="mt-2"),
                html.Div(id="manifest-oracle-response", className="small text-secondary mt-2"),
            ]
        ),
        style={"backgroundColor": "#0c0c12", "border": "1px solid #222"},
        className="mt-3",
    )


@callback(
    Output("manifest-start-ts", "value"),
    Output("manifest-end-ts", "value"),
    Input("manifest-start-now", "n_clicks"),
    Input("manifest-end-now", "n_clicks"),
    State("manifest-start-ts", "value"),
    State("manifest-end-ts", "value"),
    prevent_initial_call=True,
)
def set_now(start_clicks, end_clicks, start_val, end_val):
    triggered = dash.callback_context.triggered_id
    if triggered == "manifest-start-now":
        return _now_iso(), end_val
    if triggered == "manifest-end-now":
        return start_val, _now_iso()
    raise dash.exceptions.PreventUpdate


@callback(
    Output("manifest-substance-panel", "style"),
    Input("manifest-substance-toggle", "value"),
)
def toggle_substance(toggle_vals):
    show = "show" in (toggle_vals or [])
    return {"display": "block" if show else "none"}


@callback(
    Output("manifest-state-details", "children"),
    Output("manifest-custom-state", "style"),
    Output("manifest-stat-sessions", "children"),
    Output("manifest-stat-hours", "children"),
    Output("manifest-stat-percent", "children"),
    Output("manifest-stat-portals", "children"),
    Output("manifest-trend", "figure"),
    Output("manifest-calendar", "figure"),
    Output("manifest-portal-trend", "figure"),
    Output("manifest-portal-days", "children"),
    Output("manifest-day-list", "children"),
    Output("manifest-top-tracks", "children"),
    Input("manifest-target-state", "value"),
    Input("manifest-custom-x", "value"),
    Input("manifest-custom-q", "value"),
    Input("manifest-custom-hce", "value"),
    Input("manifest-custom-duration", "value"),
    Input("manifest-custom-name", "value"),
    Input("manifest-calendar", "clickData"),
    Input("manifest-calendar-mode", "value"),
)
def describe_state(target, x_range, q_range, hce_range, dur_min, name, click_data, cal_mode):
    def match_bucket(b):
        x = b.get("mean_X") or 0
        qv = b.get("mean_Q") or 0
        h = b.get("mean_HCE") or 0
        std_q = b.get("std_Q") or 0
        if target == "flow":
            return 1.4 <= x <= 1.7 and 0.02 <= qv <= 0.04 and 0.4 <= h <= 1.0
        if target == "relax":
            return std_q < 0.1 and 1.6 <= x <= 1.8
        if target == "transcend":
            return h >= 1.0 and qv >= 0.04
        if target == "custom" and x_range and q_range and hce_range:
            return (
                x_range[0] <= x <= x_range[1]
                and q_range[0] <= qv <= q_range[1]
                and hce_range[0] <= h <= hce_range[1]
            )
        return False

    def match_point(row):
        x = row.get("x") if isinstance(row, dict) else row["x"]
        qv = row.get("q") if isinstance(row, dict) else row["q"]
        h = row.get("hce") if isinstance(row, dict) else row["hce"]
        std_q = row.get("std_q") if isinstance(row, dict) else 0
        if target == "flow":
            return 1.4 <= x <= 1.7 and 0.02 <= qv <= 0.04 and 0.4 <= h <= 1.0
        if target == "relax":
            return std_q < 0.1 and 1.6 <= x <= 1.8
        if target == "transcend":
            return h >= 1.0 and qv >= 0.04
        if target == "custom" and x_range and q_range and hce_range:
            return (
                x_range[0] <= x <= x_range[1]
                and q_range[0] <= qv <= q_range[1]
                and hce_range[0] <= h <= hce_range[1]
            )
        return False

    desc = ""
    show_custom = {"display": "none"}
    if target == "flow":
        desc = "Flow: X 1.4–1.7, stable Q 0.02–0.04, HCE 0.4–1.0 (sustained)."
    elif target == "relax":
        desc = "Relaxed Harmony: use existing relaxed banner criteria (low std_Q, stable X, calm)."
    elif target == "transcend":
        desc = "Transcendent Harmony: use existing insight/synthesis banner criteria (transcendent windows)."
    elif target == "custom":
        desc = f"Custom: {name or 'unnamed'}; X {x_range or '-'}; Q {q_range or '-'}; HCE {hce_range or '-'}; min duration {dur_min or '-'}m."
        show_custom = {"display": "block"}

    now = time.time()
    ts0 = 0  # back-detect across all history
    ts1 = now
    buckets = storage.get_buckets_between(ts0, ts1, session_id=None)
    filt = [b for b in buckets if match_bucket(b)]
    total_sessions = len({b.get("session_id") for b in filt})
    total_secs = sum((b.get("bucket_end_ts", 0) - b.get("bucket_start_ts", 0)) for b in filt)
    hours = total_secs / 3600.0
    all_secs = sum((b.get("bucket_end_ts", 0) - b.get("bucket_start_ts", 0)) for b in buckets) or 1
    pct = (total_secs / all_secs) * 100.0

    trend_counts = {}
    for b in filt:
        week = dt.datetime.utcfromtimestamp(b["bucket_start_ts"]).strftime("%Y-%W")
        trend_counts[week] = trend_counts.get(week, 0) + 1
    trend_fig = px.line(
        x=list(trend_counts.keys()),
        y=list(trend_counts.values()),
        labels={"x": "Week", "y": "Buckets"},
        title="Weekly trend",
    )

    day_minutes: dict[str, float] = {}
    for b in filt:
        day = dt.datetime.utcfromtimestamp(b.get("bucket_start_ts", 0)).strftime("%Y-%m-%d")
        day_minutes[day] = day_minutes.get(day, 0.0) + max(
            0.0, (b.get("bucket_end_ts", 0) - b.get("bucket_start_ts", 0)) / 60.0
        )

    portal_count = 0
    portal_daily: dict[str, int] = {}
    portal_daily_peak: dict[str, float] = {}
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            wf = pd.read_sql_query(
                "SELECT abs_ts, hce, q, x FROM track_waveform_points",
                conn,
            )
        if not wf.empty:
            wf["date"] = wf["abs_ts"].apply(lambda t: dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"))
            wf["match"] = wf.apply(match_point, axis=1)
            wf_match = wf[wf["match"]]
            if not wf_match.empty:
                counts = wf_match.groupby("date")["abs_ts"].count()
                peaks = wf_match.groupby("date")["hce"].max()
                for date_str, cnt in counts.items():
                    day_minutes[date_str] = day_minutes.get(date_str, 0.0) + (cnt / 60.0)
                    portal_daily[date_str] = portal_daily.get(date_str, 0) + int(cnt)
                for date_str, peak_val in peaks.items():
                    portal_daily_peak[date_str] = max(portal_daily_peak.get(date_str, 0.0), float(peak_val))
                portal_count = int((wf_match["hce"] > 1.0).sum())
    except Exception:
        portal_count = 0
        portal_daily = {}
        portal_daily_peak = {}

    def build_calendar(day_minutes_map: dict[str, float], mode: str) -> go.Figure:
        if not day_minutes_map:
            fig = go.Figure()
            fig.update_layout(
                title="No data yet",
                paper_bgcolor="#0c0c12",
                plot_bgcolor="#0c0c12",
                font_color="#e0e0e0",
            )
            return fig
        grouped: dict[str, float] = {}
        if mode == "week":
            for d, v in day_minutes_map.items():
                wk = dt.datetime.fromisoformat(d).strftime("%Y-W%U")
                grouped[wk] = grouped.get(wk, 0.0) + v
            labels = sorted(grouped.keys())
            values = [grouped[l] for l in labels]
            custom = labels
        elif mode == "month":
            for d, v in day_minutes_map.items():
                mo = d[:7]
                grouped[mo] = grouped.get(mo, 0.0) + v
            labels = sorted(grouped.keys())
            values = [grouped[l] for l in labels]
            custom = labels
        else:  # day
            labels = sorted(day_minutes_map.keys())
            values = [day_minutes_map[l] for l in labels]
            custom = labels
        colorscale = [
            [0.0, "#0c0c12"],
            [0.3, "#4c3a1a"],
            [0.6, "#b58a2c"],
            [1.0, "#e4c36a"],
        ]
        fig = go.Figure(
            data=go.Heatmap(
                z=[values],
                x=labels,
                y=[""],
                customdata=custom,
                colorscale=colorscale,
                hovertemplate="%{customdata}<br>%{z:.1f} minutes<extra></extra>",
                colorbar_title="Minutes",
            )
        )
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(
            paper_bgcolor="#0c0c12",
            plot_bgcolor="#0c0c12",
            font_color="#e0e0e0",
            margin=dict(l=30, r=10, t=30, b=40),
            title="State calendar",
        )
        return fig

    cal_fig = build_calendar(day_minutes, cal_mode or "day")

    # Portal trend by month
    def build_portal_trend(portal_daily_map: dict[str, int]) -> go.Figure:
        if not portal_daily_map:
            fig = go.Figure()
            fig.update_layout(
                title="Portal trend (no data)",
                paper_bgcolor="#0c0c12",
                plot_bgcolor="#0c0c12",
                font_color="#e0e0e0",
            )
            return fig
        monthly: dict[str, int] = {}
        for d, c in portal_daily_map.items():
            mo = d[:7]
            monthly[mo] = monthly.get(mo, 0) + c
        months = sorted(monthly.keys())
        vals = [monthly[m] for m in months]
        fig = go.Figure(
            data=go.Scatter(
                x=months,
                y=vals,
                mode="lines+markers",
                line=dict(color="#d7b34d"),
                marker=dict(color="#d7b34d"),
            )
        )
        fig.update_layout(
            title="Portal density by month",
            paper_bgcolor="#0c0c12",
            plot_bgcolor="#0c0c12",
            font_color="#e0e0e0",
            margin=dict(l=30, r=10, t=40, b=40),
            xaxis_title="Month",
            yaxis_title="Portal count",
        )
        return fig

    portal_trend_fig = build_portal_trend(portal_daily)

    # Top portal days list
    top_portal_days = "No portal days yet."
    if portal_daily:
        ranked = sorted(portal_daily.items(), key=lambda kv: kv[1], reverse=True)[:5]
        items = []
        for d, cnt in ranked:
            peak_val = portal_daily_peak.get(d, 0.0)
            items.append(f"{d}: {cnt} portals; peak HCE {peak_val:.2f}")
        top_portal_days = html.Ul([html.Li(t) for t in items], className="mb-0")

    selected_day = None
    if (cal_mode or "day") == "day" and click_data and click_data.get("points"):
        pt = click_data["points"][0]
        sel = pt.get("customdata") or pt.get("x")
        if sel:
            selected_day = str(sel)
    day_list = "Select a day to view sessions (Day view)."
    top_tracks_txt = "No tracks/bookmarks found."
    if selected_day:
        ts_a = dt.datetime.fromisoformat(selected_day).timestamp()
        ts_b = ts_a + 24 * 3600
        day_buckets = [b for b in filt if ts_a <= b.get("bucket_start_ts", 0) < ts_b]
        sess_ids = {b.get("session_id") for b in day_buckets if b.get("session_id")}
        session_lines = []
        for sid in sorted(sess_ids):
            sb = [b for b in day_buckets if b.get("session_id") == sid]
            dur = sum((b.get("bucket_end_ts", 0) - b.get("bucket_start_ts", 0)) for b in sb) / 60.0
            mean_h = sum((b.get("mean_HCE") or 0) for b in sb) / max(len(sb), 1)
            mean_q = sum((b.get("mean_Q") or 0) for b in sb) / max(len(sb), 1)
            mean_x = sum((b.get("mean_X") or 0) for b in sb) / max(len(sb), 1)
            session_lines.append(f"{sid}: {dur:.1f} min — HCE {mean_h:.2f}, Q {mean_q:.3f}, X {mean_x:.2f}")
        if not session_lines:
            session_lines.append("No sessions matched this day.")
        day_list = html.Ul([html.Li(line) for line in session_lines], className="mb-2")

        # Tracks and bookmarks overlapping the day window
        track_items = []
        bookmark_items = []
        try:
            with sqlite3.connect(config.DB_PATH) as conn:
                ts_df = pd.read_sql_query(
                    """
                    SELECT title, artist, mean_HCE, start_ts, end_ts
                    FROM track_sessions
                    WHERE (start_ts BETWEEN ? AND ?) OR (end_ts BETWEEN ? AND ?)
                    ORDER BY mean_HCE DESC
                    LIMIT 10
                    """,
                    conn,
                    params=(ts_a, ts_b, ts_a, ts_b),
                )
            for _, r in ts_df.iterrows():
                track_items.append(
                    f"{r['title']} — {r['artist']} (HCE {r['mean_HCE']:.2f})"
                )
        except Exception:
            track_items = []
        try:
            events = storage.get_events_between(ts_a, ts_b, session_id=None)
            for e in events:
                lbl = e.get("label") or "event"
                tags = e.get("tags_json") or {}
                bookmark_items.append(f"{lbl}: {tags.get('note') or tags.get('activity') or ''}".strip())
        except Exception:
            bookmark_items = []
        tracks_block = html.Ul([html.Li(t) for t in track_items]) if track_items else "No tracks found."
        bookmarks_block = (
            html.Ul([html.Li(b) for b in bookmark_items]) if bookmark_items else "No bookmarks/events."
        )
        top_tracks_txt = html.Div(
            [
                html.Div("Tracks:", className="fw-semibold"),
                tracks_block,
                html.Div("Bookmarks/events:", className="fw-semibold mt-2"),
                bookmarks_block,
            ]
        )

    stat_sessions = f"Sessions: {total_sessions}"
    stat_hours = f"Hours in state: {hours:.2f}h"
    stat_pct = f"% of logged time: {pct:.1f}%"
    stat_portals = f"Portal density (HCE>1): {portal_count}"

    return (
        desc,
        show_custom,
        stat_sessions,
        stat_hours,
        stat_pct,
        stat_portals,
        trend_fig,
        cal_fig,
        portal_trend_fig,
        top_portal_days,
        day_list,
        top_tracks_txt,
    )


@callback(
    Output("manifest-recipe-suggest", "children"),
    Input("manifest-target-state", "value"),
)
def suggest_recipe(target):
    now = time.time()
    ts0 = now - 120 * 24 * 3600
    ts1 = now

    def match_bucket(b):
        x = b.get("mean_X") or 0
        qv = b.get("mean_Q") or 0
        h = b.get("mean_HCE") or 0
        std_q = b.get("std_Q") or 0
        if target == "flow":
            return 1.4 <= x <= 1.7 and 0.02 <= qv <= 0.04 and 0.4 <= h <= 1.0
        if target == "relax":
            return std_q < 0.1 and 1.6 <= x <= 1.8
        if target == "transcend":
            return h >= 1.0 and qv >= 0.04
        return False

    buckets = storage.get_buckets_between(ts0, ts1, session_id=None)
    overall_mean = (
        sum((b.get("mean_HCE") or 0) for b in buckets) / max(len(buckets), 1)
        if buckets
        else 0.4
    ) or 0.4
    matched = [b for b in buckets if match_bucket(b)]
    track_stats: dict[str, dict] = {}
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            ts_df = pd.read_sql_query(
                """
                SELECT track_id, title, artist, mean_HCE, start_ts, end_ts
                FROM track_sessions
                WHERE (start_ts BETWEEN ? AND ?) OR (end_ts BETWEEN ? AND ?)
                """,
                conn,
                params=(ts0, ts1, ts0, ts1),
            )
        for _, row in ts_df.iterrows():
            tid = row["track_id"]
            track_stats.setdefault(
                tid,
                {
                    "track_id": tid,
                    "title": row["title"],
                    "artist": row["artist"],
                    "count": 0,
                    "mean_hce": 0.0,
                    "best_hour": dt.datetime.utcfromtimestamp(row["start_ts"]).hour if row.get("start_ts") else 0,
                },
            )
    except Exception:
        pass

    for b in matched:
        tid = b.get("track_id")
        if not tid:
            continue
        st = track_stats.setdefault(
            tid,
            {
                "track_id": tid,
                "title": b.get("title") or "Unknown track",
                "artist": b.get("artist") or "",
                "count": 0,
                "mean_hce": 0.0,
                "best_hour": dt.datetime.utcfromtimestamp(b.get("bucket_start_ts", 0)).hour if b.get("bucket_start_ts") else 0,
            },
        )
        st["count"] += 1
        st["mean_hce"] += b.get("mean_HCE") or 0

    for st in track_stats.values():
        if st["count"] > 0:
            st["mean_hce"] = st["mean_hce"] / st["count"]
            st["lift"] = st["mean_hce"] / overall_mean if overall_mean else st["mean_hce"]
        else:
            st["lift"] = 0.0

    ranked = sorted(track_stats.values(), key=lambda x: (x.get("lift", 0) * (1 + x.get("count", 0))), reverse=True)
    ranked = [r for r in ranked if r.get("count", 0) > 0][:5]
    if not ranked:
        return "No auto-recipes yet — build history in this state."

    cards = []
    for i, r in enumerate(ranked):
        name = f"{r['title']} — {r['artist']}".strip(" —")
        lift = r.get("lift", 0)
        comps = [
            f"Expected lift: {lift:.1f}x",
            f"Occurrences: {r.get('count', 0)}",
            f"Timing: best around {r.get('best_hour', 0):02d}:00",
        ]
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(name, className="fw-semibold"),
                        html.Div("; ".join(comps), className="small text-muted"),
                        dcc.Store(id={"type": "manifest-recipe-store", "index": i}, data={**r, "name": name}),
                        dbc.Button(
                            "One-Click Apply",
                            id={"type": "manifest-apply-recipe", "index": i},
                            color="warning",
                            size="sm",
                            className="mt-1",
                        ),
                    ]
                ),
                style={"backgroundColor": "#0c0c12", "border": "1px solid #333"},
                className="mb-2",
            )
        )
    return cards


@callback(
    Output("manifest-recipe-status", "children"),
    Input("manifest-recipe-save", "n_clicks"),
    Input("manifest-recipe-apply", "n_clicks"),
    State("manifest-target-state", "value"),
    State("manifest-recipe-name", "value"),
    State("manifest-recipe-notes", "value"),
    State("manifest-recipe-duration", "value"),
    prevent_initial_call=True,
)
def save_or_apply_recipe(n_save, n_apply, target, name, notes, duration):
    triggered = dash.callback_context.triggered_id
    if not triggered:
        raise dash.exceptions.PreventUpdate
    recipe_name = name or f"{target or 'state'} recipe"
    duration_min = int(duration or 60)
    payload_steps = [notes or "Auto-generated from manifest patterns."]
    recipe_id = storage.upsert_recipe(
        recipe_id=None,
        name=recipe_name,
        mode=target or "manifest",
        target_json={"state": target},
        steps_json=payload_steps,
        description=notes or "",
    )
    msg = f"Saved recipe #{recipe_id}"
    if triggered == "manifest-recipe-apply":
        start_ts = time.time() + 3600
        end_ts = start_ts + duration_min * 60
        storage.add_schedule_block(
            title=recipe_name,
            block_type=target or "manifest",
            duration_min=duration_min,
            flexible=False,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        msg += "; applied to schedule in 1h"
    return msg


@callback(
    Output("manifest-recipe-status", "children", allow_duplicate=True),
    Input({"type": "manifest-apply-recipe", "index": ALL}, "n_clicks"),
    State({"type": "manifest-recipe-store", "index": ALL}, "data"),
    State("manifest-target-state", "value"),
    prevent_initial_call=True,
)
def apply_auto_recipe(n_clicks_list, data_list, target):
    if not n_clicks_list or not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    triggered = dash.callback_context.triggered_id
    if not triggered or "index" not in triggered:
        raise dash.exceptions.PreventUpdate
    idx = triggered["index"]
    data = None
    for i, d in enumerate(data_list or []):
        if i == idx:
            data = d
            break
    if not data:
        raise dash.exceptions.PreventUpdate
    recipe_name = data.get("name") or "Auto recipe"
    duration_min = 60
    recipe_id = storage.upsert_recipe(
        recipe_id=None,
        name=recipe_name,
        mode=target or "manifest",
        target_json={"state": target, "track": data.get("track_id")},
        steps_json=[f"Context: {recipe_name}; expected lift {data.get('lift', 0):.1f}x"],
        description="Auto-generated from proven patterns",
    )
    start_ts = time.time() + 900
    end_ts = start_ts + duration_min * 60
    storage.add_schedule_block(
        title=recipe_name,
        block_type=target or "manifest",
        duration_min=duration_min,
        flexible=False,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    return f"Applied auto-recipe #{recipe_id} for {recipe_name} (starts in 15m)"


@callback(
    Output("manifest-proven", "children"),
    Output("manifest-emerging", "children"),
    Output("manifest-sunburst", "figure"),
    Input("manifest-target-state", "value"),
)
def pattern_sunburst(target):
    def match_bucket(b):
        x = b.get("mean_X") or 0
        qv = b.get("mean_Q") or 0
        h = b.get("mean_HCE") or 0
        std_q = b.get("std_Q") or 0
        if target == "flow":
            return 1.4 <= x <= 1.7 and 0.02 <= qv <= 0.04 and 0.4 <= h <= 1.0
        if target == "relax":
            return std_q < 0.1 and 1.6 <= x <= 1.8
        if target == "transcend":
            return h >= 1.0 and qv >= 0.04
        return False

    now = time.time()
    ts0 = now - 90 * 24 * 3600
    ts1 = now
    buckets = storage.get_buckets_between(ts0, ts1, session_id=None)
    filt = [b for b in buckets if match_bucket(b)]

    # Media: top tracks by mean_HCE
    media = []
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            ts_df = pd.read_sql_query(
                """
                SELECT title, artist, mean_HCE FROM track_sessions
                WHERE start_ts BETWEEN ? AND ?
                ORDER BY mean_HCE DESC
                LIMIT 10
                """,
                conn,
                params=(ts0, ts1),
            )
        media = [f"{r['title']} — {r['artist']}" for _, r in ts_df.iterrows()]
    except Exception:
        media = []

    # Events tags
    events = storage.get_events_between(ts0, ts1, session_id=None)
    social = []
    activity = []
    substance = []
    for e in events:
        tags = e.get("tags_json") or {}
        if tags.get("social"):
            social.append(tags.get("social"))
        if tags.get("activity"):
            activity.append(tags.get("activity"))
        if "cocaine_highness" in tags or "ketamine_highness" in tags or "tusi_highness" in tags:
            substance.append("substance_context")

    hours = [dt.datetime.utcfromtimestamp(b.get("bucket_start_ts", 0)).hour for b in filt]

    def top_counts(items, threshold=1):
        counts = {}
        for it in items:
            if not it:
                continue
            counts[it] = counts.get(it, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        proven = [(k, v) for k, v in ranked if v >= 5]
        emerging = [(k, v) for k, v in ranked if threshold <= v < 5]
        return proven, emerging

    proven_media, emerging_media = top_counts(media)
    proven_social, emerging_social = top_counts(social)
    proven_activity, emerging_activity = top_counts(activity)
    proven_hours, emerging_hours = top_counts([f"{h:02d}:00" for h in hours])
    proven_sub, emerging_sub = top_counts(substance)

    proven_lines = []
    emerging_lines = []
    for lbl, data in [
        ("Media", proven_media),
        ("Social", proven_social),
        ("Activity", proven_activity),
        ("Hour", proven_hours),
        ("Substance", proven_sub),
    ]:
        if data:
            proven_lines.append(f"{lbl}: " + "; ".join([f"{k} ({v})" for k, v in data]))
    for lbl, data in [
        ("Media", emerging_media),
        ("Social", emerging_social),
        ("Activity", emerging_activity),
        ("Hour", emerging_hours),
        ("Substance", emerging_sub),
    ]:
        if data:
            emerging_lines.append(f"{lbl}: " + "; ".join([f"{k} ({v})" for k, v in data]))

    # Sunburst
    labels = []
    parents = []
    values = []

    def add_branch(category, items):
        if not items:
            return
        labels.append(category)
        parents.append("state")
        values.append(0)
        for k, v in items:
            labels.append(k)
            parents.append(category)
            values.append(v)

    labels.append("state")
    parents.append("")
    values.append(0)

    add_branch("Media", proven_media or emerging_media)
    add_branch("Social", proven_social or emerging_social)
    add_branch("Activity", proven_activity or emerging_activity)
    add_branch("Hour", proven_hours or emerging_hours)
    add_branch("Substance", proven_sub or emerging_sub)

    sunburst_fig = px.sunburst(
        names=labels,
        parents=parents,
        values=values,
        title="Induction pattern sunburst",
    )
    hover_texts = []
    for lbl, parent, val in zip(labels, parents, values):
        if parent == "state":
            hover_texts.append(f"{lbl} — {val} occurrences")
        else:
            hover_texts.append(f"{parent} • {lbl} — {val} occurrences")
    sunburst_fig.update_traces(
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    )

    proven_txt = "\n".join(proven_lines) if proven_lines else "No proven inducers yet."
    emerging_txt = "\n".join(emerging_lines) if emerging_lines else "No emerging patterns yet."
    return proven_txt, emerging_txt, sunburst_fig


@callback(
    Output("manifest-oracle-response", "children"),
    Output("manifest-oracle-input", "value", allow_duplicate=True),
    Input("manifest-oracle-send", "n_clicks"),
    State("manifest-oracle-input", "value"),
    State("manifest-target-state", "value"),
    State("manifest-stat-sessions", "children"),
    State("manifest-stat-hours", "children"),
    State("manifest-stat-portals", "children"),
    State("manifest-proven", "children"),
    prevent_initial_call=True,
)
def oracle_mini_chat(n, user_text, target, stat_sessions, stat_hours, stat_portals, proven):
    if not n or not user_text:
        raise dash.exceptions.PreventUpdate
    ctx_lines = [
        f"Target state: {target}",
        f"Stats: {stat_sessions or ''}; {stat_hours or ''}; {stat_portals or ''}",
        f"Proven inducers: {proven or ''}",
    ]
    # Placeholder for real Oracle integration
    prefilling = "\n".join(ctx_lines)
    return "Oracle (stub): " + user_text + "\n" + prefilling, prefilling


@callback(
    Output("manifest-status", "children"),
    Input("manifest-save", "n_clicks"),
    State("manifest-social", "value"),
    State("manifest-ambience", "value"),
    State("manifest-activity", "value"),
    State("manifest-minutes", "value"),
    State("manifest-start-ts", "value"),
    State("manifest-end-ts", "value"),
    State("manifest-substance-toggle", "value"),
    State("manifest-cocaine", "value"),
    State("manifest-ketamine", "value"),
    State("manifest-tusi", "value"),
    State("manifest-tusi-source", "value"),
    State("manifest-target-state", "value"),
    State("manifest-custom-name", "value"),
    State("manifest-custom-duration", "value"),
    State("manifest-custom-x", "value"),
    State("manifest-custom-q", "value"),
    State("manifest-custom-hce", "value"),
    prevent_initial_call=True,
)
def save_manifest(
    n_clicks,
    social,
    ambience,
    activity,
    minutes,
    start_ts_txt,
    end_ts_txt,
    sub_toggle,
    cocaine_lvl,
    ketamine_lvl,
    tusi_lvl,
    tusi_source,
    target_state,
    custom_name,
    custom_duration,
    custom_x,
    custom_q,
    custom_hce,
):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    session_id = registry.state_engine.get_session_id() or ""
    now = time.time()
    start_ts = _parse_ts(start_ts_txt) or now
    end_ts = _parse_ts(end_ts_txt) if end_ts_txt else None
    tags = {
        "page": "manifest",
        "social": social or "",
        "ambience": ambience or "",
        "activity": activity or "",
        "minutes": minutes,
        "target_state": target_state,
    }
    show_sub = "show" in (sub_toggle or [])
    if show_sub:
        tags.update(
            {
                "cocaine_highness": cocaine_lvl or 0,
                "ketamine_highness": ketamine_lvl or 0,
                "tusi_highness": tusi_lvl or 0,
                "tusi_source": tusi_source or "",
            }
        )
    if target_state == "custom":
        tags.update(
            {
                "custom_name": custom_name or "",
                "custom_duration_min": custom_duration,
                "custom_x_range": custom_x,
                "custom_q_range": custom_q,
                "custom_hce_range": custom_hce,
            }
        )
    elif target_state == "flow":
        tags.update({"target_profile": "flow", "x_range": [1.4, 1.7], "q_range": [0.02, 0.04], "hce_range": [0.4, 1.0]})
    elif target_state == "relax":
        tags.update({"target_profile": "relax_harmony"})
    elif target_state == "transcend":
        tags.update({"target_profile": "transcendent_harmony"})
    ctx_json = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "saved_ts": now,
    }
    storage.insert_event(
        ts=now,
        session_id=session_id,
        kind="manifest",
        label="Manifest entry",
        note="",
        tags_json=tags,
        context_json=ctx_json,
    )
    return "Saved."

