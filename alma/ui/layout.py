from dash import Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc

from alma.ui import pages


def build_sidebar() -> html.Div:
    """Construct the left navigation sidebar."""
    return html.Div(
        [
            html.Div("alma_os", className="sidebar-title"),
            dbc.Nav(pages.get_nav_links(), vertical=True, pills=True, className="sidebar-nav"),
        ],
        className="sidebar",
    )


def build_layout() -> dbc.Container:
    """Create the top-level Dash layout with routing slots."""
    return dbc.Container(
        [
            dcc.Location(id="url"),
            dcc.Store(id="ndjson-emit-store", storage_type="session", data=False),
            html.Div(
                id="ndjson-indicator",
                className="text-end text-info small my-2",
                children="NDJSON: OFF",
            ),
            dbc.Row(
                [
                    dbc.Col(build_sidebar(), width=12, md=3, lg=2, className="sidebar-column"),
                    dbc.Col(html.Div(id="page-content", className="page-content"), width=12, md=9, lg=10),
                ],
                className="app-row",
            ),
        ],
        fluid=True,
        className="app-container",
    )


@callback(Output("ndjson-indicator", "children"), Output("ndjson-indicator", "className"), Input("ndjson-emit-store", "data"))
def _update_ndjson_indicator(enabled):
    enabled = bool(enabled)
    text = f"NDJSON: {'ON' if enabled else 'OFF'}"
    cls = "text-end small my-2 " + ("text-success" if enabled else "text-secondary")
    return text, cls

