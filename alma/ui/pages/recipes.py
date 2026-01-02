import json
import time
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, dcc, html, ctx

from alma import config
from alma.app_state import registry
from alma.engine import storage
from alma.integrations import turrell_runner

MODES = ["OFF", "MAX_FOCUS", "MAX_Q", "STABILIZE"]
DEFAULT_TARGET = {"mean_HCE_min": 3.0}
DEFAULT_STEPS = ["Breath priming 2m", "Deep focus 25m", "Short movement break"]


def _load_mode() -> str:
    try:
        with config.MODE_FILE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("mode", "OFF"))
    except Exception:
        return "OFF"


def _write_mode_payload(recipe: Dict[str, object], launch_turrell: bool) -> None:
    payload = {
        "mode": recipe.get("mode", "OFF"),
        "recipe_id": recipe.get("id"),
        "name": recipe.get("name"),
        "target_metrics": recipe.get("target_json") or {},
        "steps": recipe.get("steps_json") or [],
        "launch_turrell": bool(launch_turrell),
    }
    try:
        config.MODE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.MODE_FILE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass
    if launch_turrell:
        try:
            turrell_runner.start({})
        except Exception:
            # keep failures non-fatal
            pass


def _render_recipe_cards(recipes: List[Dict[str, object]]) -> List[html.Div]:
    if not recipes:
        return [html.Div("No recipes saved.")]
    cards = []
    for r in recipes:
        rid = r.get("id")
        stats = r.get("stats_json") or {}
        runs = int(stats.get("runs", 0))
        successes = int(stats.get("successes", 0))
        efficacy = r.get("efficacy_score") or 0.0
        rate = f"{successes}/{runs} successful — efficacy {efficacy:.0%}" if runs else "no runs yet"
        steps = r.get("steps_json") or []
        steps_md = "\n".join([f"- {s}" for s in steps]) if steps else ""
        cards.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(f"{r.get('name')} ({r.get('mode')})", className="fw-bold"),
                        html.Div(r.get("description") or "", className="text-muted"),
                        html.Div(rate, className="text-muted small"),
                        html.Div(f"Target: {r.get('target_json')}", className="small text-muted"),
                        html.Div(dcc.Markdown(steps_md), className="mt-1"),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("Apply Now", id={"type": "recipe-apply", "id": rid}, color="success", size="sm"),
                                dbc.Button("Edit", id={"type": "recipe-edit", "id": rid}, color="secondary", size="sm"),
                                dbc.Button("Delete", id={"type": "recipe-delete", "id": rid}, color="danger", size="sm"),
                            ],
                            className="mt-2",
                        ),
                    ]
                ),
                className="mb-2",
            )
        )
    return cards


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("State Recipes"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button("New Recipe", id="recipe-open-modal", color="primary"),
                                    md=3,
                                    sm=12,
                                ),
                                dbc.Col(
                                    dbc.Checklist(
                                        id="recipe-launch-turrell",
                                        options=[{"label": "Launch Turrell on apply", "value": "launch"}],
                                        value=[],
                                        switch=True,
                                    ),
                                    md=4,
                                    sm=12,
                                ),
                            ],
                            className="g-3 mb-2",
                        ),
                        html.Div(id="recipe-active-banner", className="mb-2"),
                        html.Div("Recipes", className="fw-bold mb-2"),
                        html.Div(_render_recipe_cards(storage.list_recipes()), id="recipe-list"),
                        dcc.Store(id="recipe-store", data=storage.list_recipes()),
                        dcc.Store(id="recipe-active"),
                    ]
                ),
            ],
            className="page-card",
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Recipe")),
                dbc.ModalBody(
                    [
                        dbc.Input(id="recipe-id-hidden", type="hidden"),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Input(id="recipe-name", placeholder="Recipe name"), md=6, sm=12),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="recipe-mode",
                                        options=[{"label": m, "value": m} for m in MODES],
                                        value="OFF",
                                        clearable=False,
                                    ),
                                    md=6,
                                    sm=12,
                                ),
                            ],
                            className="g-2 mb-2",
                        ),
                        dbc.Textarea(
                            id="recipe-description",
                            placeholder="Description / intent",
                            rows=2,
                            className="mb-2",
                        ),
                        dbc.Label("Target metrics (JSON) — e.g., mean_HCE_min"),
                        dcc.Textarea(
                            id="recipe-target",
                            value=json.dumps(DEFAULT_TARGET),
                            rows=3,
                            style={"fontFamily": "monospace"},
                            className="mb-2",
                        ),
                        dbc.Label("Steps (JSON array of strings)"),
                        dcc.Textarea(
                            id="recipe-steps",
                            value=json.dumps(DEFAULT_STEPS, indent=2),
                            rows=4,
                            style={"fontFamily": "monospace"},
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button("Delete", id="recipe-delete-btn", color="danger", className="me-auto"),
                        dbc.Button("Cancel", id="recipe-cancel-btn", className="me-2"),
                        dbc.Button("Save", id="recipe-save-btn", color="primary"),
                    ]
                ),
            ],
            id="recipe-modal",
            is_open=False,
            backdrop="static",
            scrollable=True,
            centered=True,
        ),
    ],
    fluid=True,
    className="page-container",
)


@callback(
    Output("recipe-modal", "is_open"),
    Output("recipe-id-hidden", "value"),
    Output("recipe-name", "value"),
    Output("recipe-mode", "value"),
    Output("recipe-description", "value"),
    Output("recipe-target", "value"),
    Output("recipe-steps", "value"),
    Input("recipe-open-modal", "n_clicks"),
    Input({"type": "recipe-edit", "id": ALL}, "n_clicks"),
    Input("recipe-cancel-btn", "n_clicks"),
    State("recipe-store", "data"),
    prevent_initial_call=True,
)
def open_modal(new_click, edit_click, cancel_click, recipes):
    trig = ctx.triggered_id
    if trig is None:
        raise dash.exceptions.PreventUpdate  # type: ignore
    if trig == "recipe-cancel-btn":
        return False, None, None, "OFF", "", json.dumps(DEFAULT_TARGET), json.dumps(DEFAULT_STEPS, indent=2)
    if trig == "recipe-open-modal":
        return True, None, "", "OFF", "", json.dumps(DEFAULT_TARGET), json.dumps(DEFAULT_STEPS, indent=2)
    rid = trig.get("id")
    rec = next((r for r in recipes or [] if r.get("id") == rid), None)
    if not rec:
        raise dash.exceptions.PreventUpdate  # type: ignore
    return (
        True,
        rid,
        rec.get("name"),
        rec.get("mode"),
        rec.get("description") or "",
        json.dumps(rec.get("target_json") or DEFAULT_TARGET),
        json.dumps(rec.get("steps_json") or DEFAULT_STEPS, indent=2),
    )


@callback(
    Output("recipe-store", "data"),
    Output("recipe-list", "children"),
    Output("recipe-modal", "is_open", allow_duplicate=True),
    Input("recipe-save-btn", "n_clicks"),
    State("recipe-id-hidden", "value"),
    State("recipe-name", "value"),
    State("recipe-mode", "value"),
    State("recipe-description", "value"),
    State("recipe-target", "value"),
    State("recipe-steps", "value"),
    prevent_initial_call=True,
)
def save_recipe(_n, rid, name, mode, description, target_str, steps_str):
    if not _n or ctx.triggered_id != "recipe-save-btn":
        raise dash.exceptions.PreventUpdate  # type: ignore
    try:
        target_json = json.loads(target_str or "{}")
    except Exception:
        target_json = DEFAULT_TARGET
    try:
        steps_json = json.loads(steps_str or "[]")
        if not isinstance(steps_json, list):
            steps_json = DEFAULT_STEPS
    except Exception:
        # Accept newline-delimited steps as fallback
        steps_json = [s.strip() for s in (steps_str or "").splitlines() if s.strip()] or DEFAULT_STEPS
    storage.init_db()
    storage.upsert_recipe(
        recipe_id=int(rid) if rid else None,
        name=name or "recipe",
        mode=mode or "OFF",
        target_json=target_json,
        steps_json=steps_json,
        description=description or "",
    )
    recipes = storage.list_recipes()
    return recipes, _render_recipe_cards(recipes), False


@callback(
    Output("recipe-store", "data", allow_duplicate=True),
    Output("recipe-list", "children", allow_duplicate=True),
    Input({"type": "recipe-delete", "id": ALL}, "n_clicks"),
    State("recipe-store", "data"),
    prevent_initial_call=True,
)
def delete_recipe_from_card(_n, recipes):
    trig = ctx.triggered_id
    if not trig or not trig.get("id"):
        raise dash.exceptions.PreventUpdate  # type: ignore
    rid = trig.get("id")
    try:
        storage.delete_recipe(int(rid))
    except Exception:
        pass
    recipes = storage.list_recipes()
    return recipes, _render_recipe_cards(recipes)


@callback(
    Output("recipe-store", "data", allow_duplicate=True),
    Output("recipe-list", "children", allow_duplicate=True),
    Output("recipe-modal", "is_open", allow_duplicate=True),
    Input("recipe-delete-btn", "n_clicks"),
    State("recipe-id-hidden", "value"),
    prevent_initial_call=True,
)
def delete_from_modal(n, rid):
    if not n or not rid:
        raise dash.exceptions.PreventUpdate  # type: ignore
    try:
        storage.delete_recipe(int(rid))
    except Exception:
        pass
    recipes = storage.list_recipes()
    return recipes, _render_recipe_cards(recipes), False


@callback(
    Output("recipe-active", "data"),
    Output("recipe-active-banner", "children"),
    Input({"type": "recipe-apply", "id": ALL}, "n_clicks"),
    Input("recipe-active", "data"),
    State("recipe-store", "data"),
    State("recipe-launch-turrell", "value"),
    prevent_initial_call=True,
)
def apply_recipe(apply_clicks, active_recipe, recipes, launch_vals):
    triggered = ctx.triggered_id
    if not triggered:
        raise dash.exceptions.PreventUpdate  # type: ignore
    if triggered == "recipe-active":
        raise dash.exceptions.PreventUpdate  # type: ignore
    rid = triggered.get("id")
    rec = next((r for r in recipes or [] if r.get("id") == rid), None)
    if rec is None:
        raise dash.exceptions.PreventUpdate  # type: ignore
    mode = rec.get("mode") or "OFF"
    launch_turrell = "launch" in (launch_vals or [])
    _write_mode_payload(rec, launch_turrell)
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
            html.Span(
                f"Active recipe: {rec.get('name')} ({mode}) — target HCE ≥ {rec.get('target_json',{}).get('mean_HCE_min','?')}"
            ),
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
