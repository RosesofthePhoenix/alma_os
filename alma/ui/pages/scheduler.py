import datetime as dt
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dcc, html, no_update

from alma.engine import storage

BLOCK_TYPES = ["transcendent_work", "writing", "focused", "research", "creative", "recovery"]
VIEW_OPTIONS = [{"label": "Day", "value": "day"}, {"label": "Week", "value": "week"}]


def _date_str(offset_days: int = 0) -> str:
    return (dt.date.today() + dt.timedelta(days=offset_days)).isoformat()


def _parse_time_str(val: Optional[str]) -> tuple[int, int]:
    try:
        parts = (val or "09:00").split(":")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 9, 0


def _ts_bounds(date_str: str) -> tuple[float, float]:
    d = dt.datetime.fromisoformat(date_str)
    return d.timestamp(), (d + dt.timedelta(days=1)).timestamp()


def _ts_from_date_time(date_val: str, time_val: str) -> float:
    h, m = _parse_time_str(time_val)
    base = dt.datetime.fromisoformat(date_val or _date_str(0))
    return (base + dt.timedelta(hours=h, minutes=m)).timestamp()


def _render_blocks(blocks: List[Dict[str, object]]) -> List[html.Div]:
    if not blocks:
        return [html.Div("No blocks yet", className="text-muted")]
    items = []
    for b in blocks:
        start_ts = b.get("start_ts")
        end_ts = b.get("end_ts")
        st_txt = dt.datetime.fromtimestamp(start_ts).strftime("%H:%M") if start_ts else "--"
        en_txt = dt.datetime.fromtimestamp(end_ts).strftime("%H:%M") if end_ts else "--"
        items.append(
            html.Div(
                f"{st_txt}–{en_txt} | {b.get('title','')} ({b.get('block_type')}) — {b.get('duration_min','?')} min — flex={'yes' if b.get('flexible') else 'no'}"
            )
        )
    return items


def _bucket_hourly_stats(buckets: List[Dict[str, object]]) -> Dict[int, Dict[str, float]]:
    agg = defaultdict(list)
    for b in buckets or []:
        start_ts = b.get("bucket_start_ts")
        hce = b.get("mean_HCE")
        vf = b.get("valid_fraction") or 0.0
        std_q = b.get("std_Q") or 0.0
        mean_x = b.get("mean_X") or 0.0
        if start_ts is None or hce is None:
            continue
        if not np.isfinite(hce) or not np.isfinite(mean_x):
            continue
        # favor reliable slices
        if vf < 0.4:
            continue
        hour = dt.datetime.fromtimestamp(start_ts).hour
        agg[hour].append({"hce": hce, "vf": vf, "std_q": std_q, "mean_x": mean_x})
    stats = {}
    for hour, arr in agg.items():
        hces = np.array([a["hce"] for a in arr], dtype=float)
        vfs = np.array([a["vf"] for a in arr], dtype=float)
        stdqs = np.array([a["std_q"] for a in arr], dtype=float)
        meanxs = np.array([a["mean_x"] for a in arr], dtype=float)
        stats[hour] = {
            "mean_hce": float(np.nanmean(hces)),
            "valid_fraction": float(np.nanmean(vfs)),
            "std_q": float(np.nanmean(stdqs)),
            "mean_x": float(np.nanmean(meanxs)),
            "count": len(arr),
        }
    return stats


def _rank_windows(stats: Dict[int, Dict[str, float]]) -> List[Dict[str, object]]:
    ranked: List[Dict[str, object]] = []
    for hour, s in stats.items():
        mean_hce = s.get("mean_hce", 0.0)
        vf = s.get("valid_fraction", 0.0)
        std_q = s.get("std_q", 0.0)
        if mean_hce < 2.0 or vf < 0.6:
            continue
        score = mean_hce * 2.0 + vf * 1.5 - std_q * 0.5
        ranked.append({**s, "hour": hour, "score": score})
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:4]


def _suggest_windows(date_val: str) -> List[Dict[str, object]]:
    storage.init_db()
    now_ts = time.time()
    buckets = storage.get_buckets_between(0, now_ts, session_id=None)
    stats = _bucket_hourly_stats(buckets)
    ranked = _rank_windows(stats)
    if not ranked:
        return []
    day = dt.datetime.fromisoformat(date_val or _date_str(0))
    suggestions = []
    for r in ranked:
        start_dt = day + dt.timedelta(hours=int(r["hour"]))
        end_dt = start_dt + dt.timedelta(hours=1)
        suggestions.append(
            {
                "start_ts": start_dt.timestamp(),
                "end_ts": end_dt.timestamp(),
                "hour": r["hour"],
                "mean_hce": r["mean_hce"],
                "valid_fraction": r["valid_fraction"],
                "std_q": r["std_q"],
                "mean_x": r["mean_x"],
                "count": r["count"],
            }
        )
    return suggestions


def _format_why(s: Dict[str, object]) -> str:
    return (
        f"Recommended {dt.datetime.fromtimestamp(s['start_ts']).strftime('%H:%M')}–"
        f"{dt.datetime.fromtimestamp(s['end_ts']).strftime('%H:%M')}: "
        f"Historical mean_HCE {s.get('mean_hce', 0):.2f}, "
        f"valid_fraction {s.get('valid_fraction', 0):.2f}, std_Q {s.get('std_q', 0):.2f} — "
        "ideal for harmonious elevation."
    )


def _build_calendar_figure(blocks: List[Dict[str, object]], suggestions: List[Dict[str, object]], view: str, anchor_date: str) -> go.Figure:
    blocks = blocks or []
    suggestions = suggestions or []
    view = view or "day"
    anchor = dt.datetime.fromisoformat(anchor_date or _date_str(0))
    horizon = anchor + (dt.timedelta(days=1) if view == "day" else dt.timedelta(days=7))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[anchor, horizon], y=[-1, -1], mode="markers", marker=dict(opacity=0)))

    shapes = []
    annotations = []
    for idx, b in enumerate(blocks):
        start_ts = b.get("start_ts")
        end_ts = b.get("end_ts") or start_ts
        if start_ts is None:
            continue
        start_dt = dt.datetime.fromtimestamp(start_ts)
        end_dt = dt.datetime.fromtimestamp(end_ts)
        y0 = idx
        y1 = idx + 0.8
        color = "#4e79a7"
        shapes.append(
            dict(
                type="rect",
                x0=start_dt,
                x1=end_dt,
                y0=y0,
                y1=y1,
                xref="x",
                yref="y",
                line={"color": color},
                fillcolor=color,
                opacity=0.35,
            )
        )
        annotations.append(
            dict(
                x=start_dt,
                y=(y0 + y1) / 2,
                text=f"{b.get('title','block')} ({b.get('block_type')})",
                showarrow=False,
                xanchor="left",
                font={"size": 10},
            )
        )

    for idx, s in enumerate(suggestions):
        start_dt = dt.datetime.fromtimestamp(s["start_ts"])
        end_dt = dt.datetime.fromtimestamp(s["end_ts"])
        y0 = len(blocks) + idx + 0.2
        y1 = len(blocks) + idx + 0.8
        shapes.append(
            dict(
                type="rect",
                x0=start_dt,
                x1=end_dt,
                y0=y0,
                y1=y1,
                xref="x",
                yref="y",
                line={"color": "#59a14f", "dash": "dot"},
                fillcolor="#59a14f",
                opacity=0.2,
            )
        )
        annotations.append(
            dict(
                x=start_dt,
                y=(y0 + y1) / 2,
                text=f"Suggest: {start_dt.strftime('%H:%M')} ({s.get('mean_hce',0):.1f} HCE)",
                showarrow=False,
                xanchor="left",
                font={"size": 10, "color": "#2d6a4f"},
            )
        )

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(title="Time", tickformat="%a %H:%M"),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-1, max(len(blocks) + len(suggestions), 4)]),
        margin=dict(t=30, r=30, b=40, l=10),
        dragmode="pan",
        height=max(450, 90 * max(len(blocks), 1)),
    )
    return fig


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Scheduler MVP — Life OS"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Anchor date"),
                                        dcc.DatePickerSingle(id="sched-date", date=_date_str(0), display_format="YYYY-MM-DD"),
                                    ],
                                    md=3,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("View"),
                                        dcc.RadioItems(id="sched-view", options=VIEW_OPTIONS, value="day", inline=True),
                                    ],
                                    md=3,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Start time"),
                                        dbc.Input(id="sched-start-time", type="time", value="09:00", size="sm"),
                                    ],
                                    md=3,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Duration (min)"),
                                        dbc.Input(id="sched-duration", type="number", value=60, min=15, step=15, size="sm"),
                                    ],
                                    md=3,
                                    sm=12,
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Block title"),
                                        dbc.Input(id="sched-title", placeholder="e.g., Transcendent Work", size="sm"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Type"),
                                        dcc.Dropdown(
                                            id="sched-type",
                                            options=[{"label": t.replace('_', ' ').title(), "value": t} for t in BLOCK_TYPES],
                                            value="transcendent_work",
                                            clearable=False,
                                        ),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Flexible?"),
                                        dbc.Checklist(
                                            options=[{"label": "Yes", "value": "flex"}],
                                            value=["flex"],
                                            id="sched-flexible",
                                            switch=True,
                                        ),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Button("Add Block", id="sched-add-btn", color="primary", className="me-2"),
                        dbc.Button("Suggest", id="sched-suggest-btn", color="success"),
                        html.Div("Drag blocks on the timeline to adjust start/end. Suggestions show as green bands.", className="text-muted small mt-2"),
                        html.Hr(),
                        dcc.Graph(id="sched-calendar-graph", config={"displayModeBar": True, "editable": True, "scrollZoom": True}),
                        html.Div("Existing blocks", className="fw-bold mt-3"),
                        html.Div(id="sched-blocks-list", className="mb-3"),
                        html.Div("Suggestions", className="fw-bold mt-2"),
                        html.Div(id="sched-suggestions"),
                        dcc.Store(id="sched-blocks-store"),
                        dcc.Store(id="sched-suggestions-store"),
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
    Output("sched-blocks-store", "data"),
    Output("sched-blocks-list", "children"),
    Input("sched-date", "date"),
    Input("sched-add-btn", "n_clicks"),
    State("sched-title", "value"),
    State("sched-type", "value"),
    State("sched-duration", "value"),
    State("sched-start-time", "value"),
    State("sched-flexible", "value"),
    prevent_initial_call=False,
)
def add_or_load_blocks(date_val, add_clicks, title, block_type, duration, start_time, flexible_vals):
    date_val = date_val or _date_str(0)
    storage.init_db()
    if add_clicks:
        try:
            start_ts = _ts_from_date_time(date_val, start_time or "09:00")
            dur_min = int(duration or 60)
            storage.add_schedule_block(
                title=title or "block",
                block_type=block_type or "focused",
                duration_min=dur_min,
                flexible="flex" in (flexible_vals or []),
                start_ts=start_ts,
                end_ts=start_ts + dur_min * 60,
            )
        except Exception:
            pass
    blocks = storage.list_schedule_blocks_for_date(date_val)
    return blocks, _render_blocks(blocks)


@callback(
    Output("sched-suggestions-store", "data"),
    Output("sched-suggestions", "children"),
    Input("sched-suggest-btn", "n_clicks"),
    State("sched-date", "date"),
    prevent_initial_call=True,
)
def suggest_schedule(_n, date_val):
    date_val = date_val or _date_str(0)
    suggestions = _suggest_windows(date_val)
    if not suggestions:
        return [], "No suggestions yet. Need more historical HCE-rich buckets."
    cards = []
    for s in suggestions:
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(_format_why(s), className="fw-bold"),
                        html.Div(
                            f"Stats — mean_HCE: {s.get('mean_hce', 0):.2f}, "
                            f"valid_fraction: {s.get('valid_fraction', 0):.2f}, "
                            f"std_Q: {s.get('std_q', 0):.2f}, samples: {s.get('count', 0)}",
                            className="small text-muted",
                        ),
                    ]
                ),
                className="mb-2",
            )
        )
    return suggestions, cards


@callback(
    Output("sched-calendar-graph", "figure"),
    Input("sched-blocks-store", "data"),
    Input("sched-suggestions-store", "data"),
    Input("sched-view", "value"),
    Input("sched-date", "date"),
)
def render_calendar(blocks, suggestions, view, date_val):
    date_val = date_val or _date_str(0)
    return _build_calendar_figure(blocks or [], suggestions or [], view, date_val)


def _parse_relayout_time(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) / 1000 if val > 1e12 else float(val)
    try:
        return dt.datetime.fromisoformat(str(val)).timestamp()
    except Exception:
        return None


@callback(
    Output("sched-blocks-store", "data", allow_duplicate=True),
    Output("sched-blocks-list", "children", allow_duplicate=True),
    Input("sched-calendar-graph", "relayoutData"),
    State("sched-blocks-store", "data"),
    State("sched-date", "date"),
    prevent_initial_call=True,
)
def update_block_from_drag(relayout, blocks, date_val):
    if not relayout or not blocks:
        return no_update, no_update
    updates = {}
    for key, val in relayout.items():
        m = re.match(r"shapes\[(\d+)\]\.(x0|x1)", str(key))
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= len(blocks):
            # shapes after blocks are suggestions; ignore
            continue
        updates.setdefault(idx, {})[m.group(2)] = _parse_relayout_time(val)

    if not updates:
        return no_update, no_update

    for idx, vals in updates.items():
        blk = blocks[idx]
        start_ts = vals.get("x0", blk.get("start_ts"))
        end_ts = vals.get("x1", blk.get("end_ts"))
        if start_ts is None or end_ts is None:
            continue
        if end_ts <= start_ts:
            end_ts = start_ts + max(blk.get("duration_min", 60), 15) * 60
        dur_min = int(round((end_ts - start_ts) / 60))
        try:
            storage.update_schedule_block(block_id=blk.get("id"), start_ts=start_ts, end_ts=end_ts, duration_min=dur_min)
        except Exception:
            continue

    refreshed = storage.list_schedule_blocks_for_date(date_val or _date_str(0))
    return refreshed, _render_blocks(refreshed)
