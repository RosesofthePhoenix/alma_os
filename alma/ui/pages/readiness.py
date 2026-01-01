import datetime as dt
import sqlite3
from typing import Dict, List, Optional, Tuple

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from alma import config


LABEL_COLORS = {
    "DEEP_WORK": "#4caf50",
    "ELEVATION": "#7c4dff",
    "IDEATION": "#ff9800",
    "RECOVERY": "#03a9f4",
    "ENGAGEMENT": "#9c27b0",
    "INSUFFICIENT_SIGNAL": "#9e9e9e",
}

LABEL_NOTES = {
    "DEEP_WORK": "High activation + steady richness",
    "ELEVATION": "Harmonious elevation: strong HCE + balanced richness",
    "IDEATION": "High variability + rising richness",
    "RECOVERY": "Low activation + settling",
    "ENGAGEMENT": "Balanced activation + richness",
    "INSUFFICIENT_SIGNAL": "Not enough clean signal in this window",
}


def _fetch_buckets_for_date(date_str: str) -> List[Dict[str, object]]:
    start = dt.datetime.fromisoformat(date_str)
    end = start + dt.timedelta(days=1)
    rows: List[Dict[str, object]] = []
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            cur = conn.execute(
                """
                SELECT bucket_start_ts, bucket_end_ts, mean_X, mean_Q, mean_HCE, std_Q, Q_slope, valid_fraction, label, session_id
                FROM buckets
                WHERE bucket_start_ts >= ? AND bucket_start_ts < ?
                ORDER BY bucket_start_ts ASC
                """,
                (start.timestamp(), end.timestamp()),
            )
            for r in cur.fetchall():
                rows.append(
                    {
                        "bucket_start_ts": r[0],
                        "bucket_end_ts": r[1],
                        "mean_X": r[2],
                        "mean_Q": r[3],
                        "mean_HCE": r[4],
                        "std_Q": r[5],
                        "Q_slope": r[6],
                        "valid_fraction": r[7],
                        "label": r[8],
                        "session_id": r[9],
                    }
                )
    except Exception:
        return []
    return rows


def _blank_fig(title: str, height: int = 240) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=30),
        height=height,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, visible=False),
    )
    fig.add_annotation(text="No buckets yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    return fig


def _make_timeline(buckets: List[Dict[str, object]], title: str = "Readiness Map") -> go.Figure:
    if not buckets:
        return _blank_fig(title)
    # Build an hourly heatmap-like bar view; color by label, intensity modulated by HCE.
    fig = go.Figure()
    for b in buckets:
        start_ts = b.get("bucket_start_ts")
        end_ts = b.get("bucket_end_ts")
        if start_ts is None or end_ts is None:
            continue
        start_dt = dt.datetime.fromtimestamp(start_ts)
        dur = max(end_ts - start_ts, 1.0)
        label = _classify_bucket(b)
        color = LABEL_COLORS.get(label, "#9e9e9e")
        hce = b.get("mean_HCE", 0.0) or 0.0
        # Modulate opacity to reflect HCE magnitude (capped for clarity).
        opacity = float(max(0.25, min(1.0, 2.5 * hce + 0.25)))
        fig.add_trace(
            go.Bar(
                x=[dur],
                y=["Readiness"],
                base=[start_dt],
                orientation="h",
                marker_color=color,
                opacity=opacity,
                hovertemplate=(
                    f"{label}<br>"
                    "Start: %{base|%H:%M}<br>"
                    "Duration: %{x:.0f}s<br>"
                    "mean_X: %{customdata[0]:.3f}<br>"
                    "mean_Q: %{customdata[1]:.3f}<br>"
                    "mean_HCE: %{customdata[2]:.3f}<br>"
                    "std_Q: %{customdata[3]:.3f}<br>"
                    "Q_slope: %{customdata[4]:.4f}<br>"
                    "Why: %{customdata[5]}<extra></extra>"
                ),
                customdata=[
                    [
                        b.get("mean_X", 0.0),
                        b.get("mean_Q", 0.0),
                        b.get("mean_HCE", 0.0),
                        b.get("std_Q", 0.0),
                        b.get("Q_slope", 0.0),
                        _friendly_label(label),
                        start_ts,
                        end_ts,
                    ]
                ],
                width=0.6,
            )
        )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        barmode="stack",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=40),
        height=260,
        xaxis=dict(type="date", title="Time"),
        yaxis=dict(showticklabels=False),
    )
    return fig


def _make_today_stripe(buckets: List[Dict[str, object]]) -> go.Figure:
    if not buckets:
        return _blank_fig("Today", height=120)
    fig = go.Figure()
    for b in buckets:
        start_ts = b["bucket_start_ts"]
        end_ts = b["bucket_end_ts"]
        start_dt = dt.datetime.fromtimestamp(start_ts)
        dur = max(end_ts - start_ts, 1.0)
        label = _classify_bucket(b)
        color = LABEL_COLORS.get(label, "#9e9e9e")
        fig.add_trace(
            go.Bar(
                x=[dur],
                y=["Today"],
                base=[start_dt],
                orientation="h",
                marker_color=color,
                opacity=float(max(0.25, min(1.0, 2.5 * (b.get("mean_HCE", 0.0) or 0.0) + 0.25))),
                hovertemplate=f"{label}<extra></extra>",
                width=0.6,
            )
        )
    fig.update_layout(
        template="plotly_dark",
        title="Today",
        barmode="stack",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=30),
        height=140,
        xaxis=dict(type="date", title=""),
        yaxis=dict(showticklabels=False),
    )
    return fig


def _friendly_label(label: str) -> str:
    return LABEL_NOTES.get(label, LABEL_NOTES["ENGAGEMENT"])


def _classify_bucket(b: Dict[str, object]) -> str:
    """Client-side label refinement incorporating HCE and stability (masked metrics only)."""
    mean_X = float(b.get("mean_X") or 0.0)
    mean_Q = float(b.get("mean_Q") or 0.0)
    mean_HCE = float(b.get("mean_HCE") or 0.0)
    std_Q = float(b.get("std_Q") or 0.0)
    q_slope = float(b.get("Q_slope") or 0.0)
    valid_fraction = float(b.get("valid_fraction") or 0.0)

    if valid_fraction < 0.5:
        return "INSUFFICIENT_SIGNAL"
    if mean_X >= 0.65 and std_Q <= 0.12 and valid_fraction >= 0.9:
        return "DEEP_WORK"
    if mean_HCE >= 0.005 and mean_Q >= 0.03:
        return "ELEVATION"
    if q_slope > 0.0 and std_Q >= 0.18:
        return "IDEATION"
    if mean_X <= 0.45 and std_Q <= 0.16:
        return "RECOVERY"
    return "ENGAGEMENT"


layout = dbc.Container(
    [
        dbc.Card(
            [
                dbc.CardHeader("Readiness"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Date"),
                                        dcc.DatePickerSingle(
                                            id="readiness-date",
                                            date=dt.date.today(),
                                            display_format="YYYY-MM-DD",
                                        ),
                                        html.Div(id="readiness-count-text", className="text-info small mt-2"),
                                    ],
                                    md=4,
                                    sm=12,
                                ),
                            ],
                            className="g-2 mb-2",
                        ),
                        dcc.Graph(id="readiness-timeline", config={"displayModeBar": False}),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Why panel", className="fw-bold mb-2"),
                                    html.Div(id="readiness-why"),
                                ]
                            ),
                            className="mt-2",
                        ),
                        dcc.Graph(id="readiness-today-stripe", config={"displayModeBar": False}, className="mt-3"),
                        dcc.Store(id="readiness-buckets-store"),
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
    Output("readiness-buckets-store", "data"),
    Output("readiness-timeline", "figure"),
    Output("readiness-today-stripe", "figure"),
    Output("readiness-count-text", "children"),
    Input("readiness-date", "date"),
)
def load_buckets(date_val):
    date_str = date_val if isinstance(date_val, str) else str(dt.date.today())
    buckets = _fetch_buckets_for_date(date_str)
    try:
        print(f"[readiness] date={date_str} buckets={len(buckets)}")
    except Exception:
        pass
    timeline = _make_timeline(buckets, title="Readiness Map")
    stripe = _make_today_stripe(buckets)
    count_text = f"Buckets fetched: {len(buckets)}"
    return buckets, timeline, stripe, count_text


@callback(
    Output("readiness-why", "children"),
    Input("readiness-timeline", "clickData"),
    State("readiness-buckets-store", "data"),
)
def update_why(click_data, buckets):
    if not buckets:
        return "No data yet."
    bucket = None
    if click_data and click_data.get("points"):
        cd = click_data["points"][0].get("customdata", [])
        if len(cd) >= 8:
            start_ts = cd[6]
            for b in buckets:
                if abs(b.get("bucket_start_ts", 0) - start_ts) < 1e-6:
                    bucket = b
                    break
    if bucket is None:
        bucket = buckets[-1]
    label = _classify_bucket(bucket)
    note = _friendly_label(label)
    mean_hce = bucket.get("mean_HCE", 0.0) or 0.0
    why = note
    if label == "ELEVATION":
        why = "Elevated HCE: strong harmonious elevation with balanced richness."
    elif label == "DEEP_WORK":
        why = "High activation with steady richness; stay in the flow."
    elif label == "IDEATION":
        why = "Rising richness with variability; good for brainstorming."
    elif label == "RECOVERY":
        why = "Low activation and stable; good for recharge."
    return html.Div(
        [
            html.Div(f"Label: {label}", className="fw-bold"),
            html.Div(f"mean_X: {bucket.get('mean_X', 0):.3f}"),
            html.Div(f"mean_Q: {bucket.get('mean_Q', 0):.3f}"),
            html.Div(f"mean_HCE: {bucket.get('mean_HCE', 0):.3f}"),
            html.Div(f"std_Q: {bucket.get('std_Q', 0):.3f}"),
            html.Div(f"Q_slope: {bucket.get('Q_slope', 0):.4f}"),
            html.Div(why, className="mt-1"),
        ]
    )
