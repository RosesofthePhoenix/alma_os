import datetime as dt
from typing import Dict, List, Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

from alma.engine import storage

BLOCK_TYPES = ["writing", "focused", "research", "explore", "creative"]
LABEL_NOTES = {
    "DEEP_WORK": "steady richness + high activation → good for focused work",
    "IDEATION": "high variability + rising richness → good for creative",
    "RECOVERY": "lower activation → good for recharge/explore",
    "ENGAGEMENT": "balanced activation → good for research",
}


def _date_str(offset_days: int = 0) -> str:
    return (dt.date.today() + dt.timedelta(days=offset_days)).isoformat()


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Scheduler (advice-only)"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Date"),
                                        dcc.DatePickerSingle(
                                            id="sched-date",
                                            date=_date_str(0),
                                            display_format="YYYY-MM-DD",
                                        ),
                                    ],
                                    md=3,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Block title"),
                                        dbc.Input(id="sched-title", placeholder="e.g., Draft chapter", size="sm"),
                                    ],
                                    md=3,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Type"),
                                        dcc.Dropdown(
                                            id="sched-type",
                                            options=[{"label": t, "value": t} for t in BLOCK_TYPES],
                                            value="focused",
                                            clearable=False,
                                        ),
                                    ],
                                    md=2,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Duration (min)"),
                                        dbc.Input(id="sched-duration", type="number", value=60, min=15, step=15, size="sm"),
                                    ],
                                    md=2,
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
                                    md=2,
                                    sm=12,
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Button("Add Block", id="sched-add-btn", color="primary", className="me-2"),
                        dbc.Button("Suggest Schedule", id="sched-suggest-btn", color="success"),
                        html.Hr(),
                        html.Div("Existing blocks", className="fw-bold mt-2"),
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


def _ts_bounds(date_str: str) -> tuple[float, float]:
    d = dt.datetime.fromisoformat(date_str)
    return d.timestamp(), (d + dt.timedelta(days=1)).timestamp()


def _render_blocks(blocks: List[Dict[str, object]]) -> List[html.Div]:
    if not blocks:
        return [html.Div("No blocks")]
    items = []
    for b in blocks:
        start_ts = b.get("start_ts")
        st_txt = dt.datetime.fromtimestamp(start_ts).strftime("%H:%M") if start_ts else "--"
        items.append(
            html.Div(
                f"{st_txt} | {b.get('title','')} ({b.get('block_type')}) — {b.get('duration_min','?')} min — flex={'yes' if b.get('flexible') else 'no'}"
            )
        )
    return items


def _score_bucket_for_type(b: Dict[str, object], block_type: str) -> float:
    label = b.get("label")
    mean_x = b.get("mean_X") or 0.0
    std_q = b.get("std_Q") or 0.0
    q_slope = b.get("Q_slope") or 0.0
    score = 0.0
    if block_type in {"focused", "writing"}:
        if label == "DEEP_WORK":
            score += 3.0
        score += mean_x
        score -= std_q
    elif block_type == "creative":
        if label == "IDEATION":
            score += 3.0
        score += (q_slope * 10.0)
        score += (std_q * 2.0)
    elif block_type == "research":
        if label == "ENGAGEMENT":
            score += 2.0
        score += (mean_x * 0.5)
        score -= abs(q_slope)
    elif block_type == "explore":
        if label in {"RECOVERY", "ENGAGEMENT"}:
            score += 2.0
        score -= mean_x * 0.2
    if label == "INSUFFICIENT_SIGNAL":
        score -= 5.0
    return score


def _explain(label: str) -> str:
    return LABEL_NOTES.get(label, LABEL_NOTES["ENGAGEMENT"])


def _suggest(blocks: List[Dict[str, object]], buckets: List[Dict[str, object]]) -> List[Dict[str, object]]:
    suggestions = []
    flex_blocks = [b for b in blocks if b.get("flexible")]
    if not flex_blocks or not buckets:
        return []
    for blk in flex_blocks:
        best = None
        best_score = -1e9
        for b in buckets:
            sc = _score_bucket_for_type(b, blk.get("block_type"))
            if sc > best_score:
                best_score = sc
                best = b
        if best:
            suggestions.append(
                {
                    "block": blk,
                    "bucket": best,
                    "score": best_score,
                }
            )
    return suggestions


@callback(
    Output("sched-blocks-store", "data"),
    Output("sched-blocks-list", "children"),
    Input("sched-date", "date"),
    Input("sched-add-btn", "n_clicks"),
    State("sched-title", "value"),
    State("sched-type", "value"),
    State("sched-duration", "value"),
    State("sched-flexible", "value"),
    prevent_initial_call=False,
)
def add_or_load_blocks(date_val, add_clicks, title, block_type, duration, flexible_vals):
    date_val = date_val or _date_str(0)
    ts0, _ = _ts_bounds(date_val)
    if add_clicks:
        try:
            storage.init_db()
            storage.add_schedule_block(
                title=title or "block",
                block_type=block_type or "focused",
                duration_min=int(duration or 60),
                flexible="flex" in (flexible_vals or []),
                start_ts=ts0,
                end_ts=ts0 + (int(duration or 60) * 60),
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
    State("sched-blocks-store", "data"),
    prevent_initial_call=True,
)
def suggest_schedule(_n, date_val, blocks):
    date_val = date_val or _date_str(0)
    ts0, ts1 = _ts_bounds(date_val)
    buckets = storage.get_buckets_between(ts0, ts1 + 24 * 3600, session_id=None)  # today + tomorrow
    suggestions = _suggest(blocks or [], buckets or [])
    if not suggestions:
        return [], "No suggestions available."
    cards = []
    for s in suggestions:
        blk = s["block"]
        b = s["bucket"]
        start_ts = b.get("bucket_start_ts")
        end_ts = b.get("bucket_end_ts")
        ts_txt = f"{dt.datetime.fromtimestamp(start_ts).strftime('%H:%M') if start_ts else '?'}"
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(f"{blk.get('title')} ({blk.get('block_type')})", className="fw-bold"),
                        html.Div(f"Suggested start: {ts_txt}"),
                        html.Div(_explain(b.get("label") or "")),
                    ]
                ),
                className="mb-2",
            )
        )
    return suggestions, cards
