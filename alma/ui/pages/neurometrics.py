from typing import List

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from alma.app_state import registry


def _slice_history(history: dict, span_s: float = 420.0):
    t = history.get("t", [])
    if not t:
        return [], {}
    t_max = t[-1]
    cutoff = t_max - span_s
    idx = [i for i, tv in enumerate(t) if tv >= cutoff]
    if not idx:
        return [], {}
    def take(key):
        arr = history.get(key, [])
        return [arr[i] for i in idx if i < len(arr)]
    sliced = {k: take(k) for k in history.keys()}
    sliced["t"] = [t[i] for i in idx]
    return idx, sliced


def _blank_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
    )
    fig.add_annotation(text="No data yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig


def _make_qraw_fig(t: List[float], sliced: dict) -> go.Figure:
    fig = go.Figure()
    traces = [
        ("Q_abs_raw", "Q_abs_raw", "#5bc0de"),
        ("Q_vibe_raw", "Q_vibe_raw", "#a66cff"),
        ("Q_vibe_focus_raw", "Q_vibe_focus_raw", "#ff5fb2"),
    ]
    for key, name, color in traces:
        y = sliced.get(key, [])
        if y:
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=name, line=dict(color=color)))
    if not fig.data:
        return _blank_fig("Q_raw (last 420s)")
    fig.update_layout(
        template="plotly_dark",
        title="Q_raw (last 420s)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


def _make_x_fig(t: List[float], sliced: dict) -> go.Figure:
    y = sliced.get("X", [])
    if not t or not y:
        return _blank_fig("X")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="X", line=dict(color="#00e8ff")))
    fig.update_layout(
        template="plotly_dark",
        title="X (last 420s)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
    )
    return fig


def _make_q_fig(t: List[float], sliced: dict, lens: str) -> go.Figure:
    fig = go.Figure()
    traces = [
        ("Q_abs", "Q_abs", "#5bc0de"),
        ("Q_vibe", "Q_vibe", "#a66cff"),
        ("Q_vibe_focus", "Q_vibe_focus", "#ff5fb2"),
        ("Q_vibe_focus_E", "Q_vibe_focus_E", "#ff7b7b"),
    ]
    for key, name, color in traces:
        y = sliced.get(key, [])
        if y:
            line = dict(color=color, width=3 if key == f"Q_{lens}" else 2)
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=name, line=line))
    if not fig.data:
        return _blank_fig("Q (masked)")
    fig.update_layout(
        template="plotly_dark",
        title="Q (masked)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


def _make_valid_fig(t: List[float], sliced: dict) -> go.Figure:
    valid = sliced.get("valid", [])
    qconf = sliced.get("quality_conf", [])
    if not t or not valid:
        return _blank_fig("Validity / quality")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=[1.0 if bool(v) else 0.0 for v in valid],
            mode="lines",
            name="valid",
            line=dict(color="#00e676"),
            fill="tozeroy",
            fillcolor="rgba(0,230,118,0.2)",
        )
    )
    if qconf:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=qconf,
                mode="lines",
                name="quality_conf",
                line=dict(color="#ffc107"),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        title="Validity / quality_conf",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        yaxis=dict(range=[-0.1, 1.1]),
    )
    return fig


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Neurometrics"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Q lens (Neurometrics only)"),
                                        dcc.Dropdown(
                                            id="neuro-q-lens",
                                            options=[
                                                {"label": "vibe_focus (canonical)", "value": "vibe_focus"},
                                                {"label": "vibe", "value": "vibe"},
                                                {"label": "abs", "value": "abs"},
                                                {"label": "vibe_focus_e (Option E)", "value": "vibe_focus_E"},
                                            ],
                                            value="vibe_focus",
                                            clearable=False,
                                        ),
                                        html.Div(id="neuro-option-e-note", className="text-warning small mt-1"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                            ]
                        )
                    ]
                ),
            ],
            className="page-card",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    dcc.Graph(id="neuro-qraw-graph", config={"displayModeBar": False}),
                    dcc.Graph(id="neuro-x-graph", config={"displayModeBar": False}),
                    dcc.Graph(id="neuro-q-graph", config={"displayModeBar": False}),
                    dcc.Graph(id="neuro-valid-graph", config={"displayModeBar": False}),
                    dcc.Interval(id="neuro-interval", interval=1000, n_intervals=0),
                ]
            ),
            className="page-card",
        ),
    ],
    fluid=True,
    className="page-container",
)


@callback(
    Output("neuro-qraw-graph", "figure"),
    Output("neuro-x-graph", "figure"),
    Output("neuro-q-graph", "figure"),
    Output("neuro-valid-graph", "figure"),
    Output("neuro-option-e-note", "children"),
    Input("neuro-interval", "n_intervals"),
    Input("neuro-q-lens", "value"),
)
def update_neurometrics(_n, lens):
    history = registry.state_engine.get_history()
    _, sliced = _slice_history(history, span_s=420.0)
    t = sliced.get("t", [])

    fig_qraw = _make_qraw_fig(t, sliced)
    fig_x = _make_x_fig(t, sliced)
    fig_q = _make_q_fig(t, sliced, lens=lens or "vibe_focus")
    fig_valid = _make_valid_fig(t, sliced)

    note = ""
    if (lens == "vibe_focus_E") and not sliced.get("Q_vibe_focus_E"):
        note = "Option E requires baseline; data unavailable."
    return fig_qraw, fig_x, fig_q, fig_valid, note
