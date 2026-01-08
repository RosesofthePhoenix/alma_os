import time
import datetime as dt
import sqlite3
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html
import plotly.express as px

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
                        dbc.Col(dcc.Graph(id="manifest-calendar", figure=px.imshow([[0]])), md=6),
                    ]
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
    Output("manifest-day-list", "children"),
    Output("manifest-top-tracks", "children"),
    Input("manifest-target-state", "value"),
    Input("manifest-custom-x", "value"),
    Input("manifest-custom-q", "value"),
    Input("manifest-custom-hce", "value"),
    Input("manifest-custom-duration", "value"),
    Input("manifest-custom-name", "value"),
    Input("manifest-calendar", "clickData"),
)
def describe_state(target, x_range, q_range, hce_range, dur_min, name, click_data):
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
    ts0 = now - 90 * 24 * 3600
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

    day_counts = {}
    for b in filt:
        day = dt.datetime.utcfromtimestamp(b["bucket_start_ts"]).strftime("%Y-%m-%d")
        day_counts[day] = day_counts.get(day, 0) + (b.get("bucket_end_ts", 0) - b.get("bucket_start_ts", 0)) / 60.0
    days_sorted = sorted(day_counts.keys())
    cal_values = [day_counts[d] for d in days_sorted] if days_sorted else [0]
    cal_fig = px.imshow([cal_values], labels={"x": "Date", "color": "Minutes"}, aspect="auto")
    cal_fig.update_xaxes(tickvals=list(range(len(days_sorted))), ticktext=days_sorted)
    cal_fig.update_yaxes(showticklabels=False)

    selected_day = None
    if click_data and click_data.get("points"):
        idx = click_data["points"][0].get("x")
        if idx is not None and idx < len(days_sorted):
            selected_day = days_sorted[idx]
    day_list = "Select a day to view sessions."
    portal_count = 0
    top_tracks_txt = "No tracks found."
    if selected_day:
        ts_a = dt.datetime.fromisoformat(selected_day).timestamp()
        ts_b = ts_a + 24 * 3600
        day_buckets = [b for b in filt if ts_a <= b.get("bucket_start_ts", 0) < ts_b]
        sess_ids = {b.get("session_id") for b in day_buckets}
        day_list = f"{selected_day}: {len(day_buckets)} buckets across {len(sess_ids)} sessions"
        try:
            import pandas as pd
            with sqlite3.connect(config.DB_PATH) as conn:
                wf = pd.read_sql_query(
                    "SELECT hce FROM track_waveform_points WHERE abs_ts BETWEEN ? AND ?",
                    conn,
                    params=(ts_a, ts_b),
                )
            portal_count = int((wf["hce"] > 1.0).sum()) if not wf.empty else 0
        except Exception:
            portal_count = 0
        try:
            import pandas as pd
            with sqlite3.connect(config.DB_PATH) as conn:
                ts_df = pd.read_sql_query(
                    """
                    SELECT title, artist, mean_HCE, start_ts FROM track_sessions
                    WHERE start_ts BETWEEN ? AND ?
                    ORDER BY mean_HCE DESC
                    LIMIT 5
                    """,
                    conn,
                    params=(ts_a, ts_b),
                )
            if not ts_df.empty:
                top_tracks_txt = "; ".join(
                    [f"{r['title']} — {r['artist']} (HCE {r['mean_HCE']:.2f})" for _, r in ts_df.iterrows()]
                )
        except Exception:
            pass

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
        day_list,
        top_tracks_txt,
    )


@callback(
    Output("manifest-recipe-suggest", "children"),
    Input("manifest-target-state", "value"),
)
def suggest_recipe(target):
    now = time.time()
    ts0 = now - 30 * 24 * 3600
    ts1 = now
    track_txt = "No track suggestion yet."
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            ts_df = pd.read_sql_query(
                """
                SELECT title, artist, mean_HCE FROM track_sessions
                WHERE start_ts BETWEEN ? AND ?
                ORDER BY mean_HCE DESC
                LIMIT 1
                """,
                conn,
                params=(ts0, ts1),
            )
        if not ts_df.empty:
            r = ts_df.iloc[0]
            track_txt = f"Try: {r['title']} — {r['artist']} (HCE {r['mean_HCE']:.2f})"
    except Exception:
        pass
    ctx = {
        "flow": "Pair with focused tasks at your peak hour.",
        "relax": "Use calming ambience + minimal social.",
        "transcend": "Stack with reflection/journaling windows.",
    }.get(target, "Tune context to your custom ranges.")
    return f"{track_txt}. {ctx}"


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

    proven_txt = "\n".join(proven_lines) if proven_lines else "No proven inducers yet."
    emerging_txt = "\n".join(emerging_lines) if emerging_lines else "No emerging patterns yet."
    return proven_txt, emerging_txt, sunburst_fig


@callback(
    Output("manifest-oracle-response", "children"),
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
    return "Oracle (stub): " + user_text + "\n" + "\n".join(ctx_lines)


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

