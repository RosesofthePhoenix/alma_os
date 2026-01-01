import json
from typing import Dict, List, Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, ctx

from alma import config
from alma.engine import storage
from alma.app_state import registry
import dash
import time

MODES = ["OFF", "MAX_FOCUS", "MAX_Q", "STABILIZE"]


def _load_mode() -> str:
    try:
        with config.MODE_FILE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("mode", "OFF"))
    except Exception:
        return "OFF"


def _write_mode(mode: str) -> None:
    try:
        config.MODE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.MODE_FILE_PATH.open("w", encoding="utf-8") as f:
            json.dump({"mode": mode}, f)
    except Exception:
        pass


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Recipes"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Name"),
                                        dbc.Input(id="recipe-name", placeholder="Recipe name", size="sm"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Mode"),
                                        dcc.Dropdown(
                                            id="recipe-mode",
                                            options=[{"label": m, "value": m} for m in MODES],
                                            value="OFF",
                                            clearable=False,
                                        ),
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
                                        dbc.Label("Target JSON"),
                                        dcc.Textarea(
                                            id="recipe-target",
                                            value='{"mean_X_min":0.65,"mean_Q_min":0.25,"std_Q_max":0.12,"slope_min":-999}',
                                            rows=4,
                                            style={"fontFamily": "monospace"},
                                        ),
                                    ],
                                    md=6,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Steps (markdown)"),
                                        dcc.Textarea(id="recipe-steps", value="- step 1\n- step 2", rows=4),
                                    ],
                                    md=6,
                                    sm=12,
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Button("Save Recipe", id="recipe-save-btn", color="primary", className="me-2"),
                        html.Span(id="recipe-save-status", className="ms-2"),
                        html.Hr(),
                        html.Div(id="recipe-active-banner", className="mb-2"),
                        html.Div("Recipes", className="fw-bold mb-2"),
                        html.Div(id="recipe-list"),
                        dcc.Store(id="recipe-store"),
                        dcc.Store(id="recipe-active"),
                    ]
                ),
            ],
            className="page-card",
        )
    ],
    fluid=True,
    className="page-container",
)


def _render_recipe_cards(recipes: List[Dict[str, object]]) -> List[html.Div]:
    if not recipes:
        return [html.Div("No recipes saved.")]
    cards = []
    for r in recipes:
        rid = r.get("id")
        stats = r.get("stats_json") or {}
        runs = int(stats.get("runs", 0))
        successes = int(stats.get("successes", 0))
        rate = f"{successes}/{runs} successful" if runs else "no runs yet"
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(f"{r.get('name')} ({r.get('mode')})", className="fw-bold"),
                        html.Div(rate, className="text-muted small"),
                        html.Div(f"Target: {r.get('target_json')}", className="small text-muted"),
                        html.Div(dcc.Markdown(r.get("steps_md") or ""), className="mt-1"),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Apply Now", id={"type": "recipe-apply", "id": rid}, color="success", size="sm"),
                                dbc.Button("Edit", id={"type": "recipe-edit", "id": rid}, color="secondary", size="sm"),
                            ],
                            className="mt-2",
                        ),
                    ]
                ),
                className="mb-2",
            )
        )
    return cards


@callback(
    Output("recipe-store", "data"),
    Output("recipe-list", "children"),
    Input("recipe-save-btn", "n_clicks"),
    State("recipe-name", "value"),
    State("recipe-mode", "value"),
    State("recipe-target", "value"),
    State("recipe-steps", "value"),
    prevent_initial_call=True,
)
def save_recipe(n, name, mode, target_str, steps):
    if not n:
        raise dash.exceptions.PreventUpdate  # type: ignore
    try:
        target_json = json.loads(target_str or "{}")
    except Exception:
        target_json = {}
    try:
        storage.init_db()
        storage.upsert_recipe(None, name or "recipe", mode or "OFF", target_json, steps or "")
    except Exception:
        pass
    recipes = storage.list_recipes()
    return recipes, _render_recipe_cards(recipes)


@callback(
    Output("recipe-store", "data", allow_duplicate=True),
    Output("recipe-list", "children", allow_duplicate=True),
    Input("recipe-store", "data"),
    prevent_initial_call=True,
)
def refresh_list(recipes):
    if recipes is None:
        recipes = storage.list_recipes()
    return recipes, _render_recipe_cards(recipes)


@callback(
    Output("recipe-active", "data"),
    Output("recipe-active-banner", "children"),
    Input({"type": "recipe-apply", "id": "*"}, "n_clicks"),
    Input("recipe-active", "data"),
    State("recipe-store", "data"),
    prevent_initial_call=True,
)
def apply_recipe(apply_clicks, active_recipe, recipes):
    triggered = ctx.triggered_id
    if not triggered:
        raise dash.exceptions.PreventUpdate  # type: ignore
    if triggered == "recipe-active":
        raise dash.exceptions.PreventUpdate  # type: ignore
    rid = triggered.get("id")
    rec = None
    for r in recipes or []:
        if r.get("id") == rid:
            rec = r
            break
    if rec is None:
        raise dash.exceptions.PreventUpdate  # type: ignore
    mode = rec.get("mode") or "OFF"
    _write_mode(mode)
    now_ts = time.time()
    try:
        storage.insert_event(
            ts=now_ts,
            session_id=registry.state_engine.get_session_id() or "",
            kind="recipe_start",
            label=f"Recipe Start: {rec.get('name')}",
            note="",
            tags_json={"kind": "recipe_start", "recipe_id": rec.get("id")},
            context_json={"recipe_id": rec.get("id"), "target_json": rec.get("target_json"), "mode": mode},
        )
    except Exception:
        pass
    banner = dbc.Alert(
        [
            html.Span(f"Active recipe: {rec.get('name')} ({mode})"),
            dbc.Button("End Recipe", id="recipe-end-btn", color="warning", size="sm", className="ms-2"),
        ],
        color="info",
        className="mt-2",
    )
    return rec, banner


@callback(
    Output("recipe-active", "data", allow_duplicate=True),
    Output("recipe-active-banner", "children", allow_duplicate=True),
    Input("recipe-end-btn", "n_clicks"),
    State("recipe-active", "data"),
    prevent_initial_call=True,
)
def end_recipe(n, active_recipe):
    if not n or not active_recipe:
        raise dash.exceptions.PreventUpdate  # type: ignore
    now_ts = time.time()
    try:
        storage.insert_event(
            ts=now_ts,
            session_id=registry.state_engine.get_session_id() or "",
            kind="recipe_end",
            label=f"Recipe End: {active_recipe.get('name')}",
            note="",
            tags_json={"kind": "recipe_end", "recipe_id": active_recipe.get("id")},
            context_json={"recipe_id": active_recipe.get("id")},
        )
        rid = active_recipe.get("id")
        if rid:
            storage.score_recipe_run(recipe_id=rid, end_ts=now_ts)
    except Exception:
        pass
    return None, ""
