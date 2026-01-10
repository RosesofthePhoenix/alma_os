from dash import Input, Output
import dash_bootstrap_components as dbc

from . import (
    home,
    neurometrics,
    readiness,
    scheduler,
    memory,
    environment,
    recipes,
    settings,
    spotify_resonance,
    longitudinal,
    media_alchemy,
    manifest,
    live_media,
    spotify_insights,
    continuum,
)


PAGES = [
    {"name": "Home", "path": "/", "layout": home.layout},
    {"name": "Neurometrics", "path": "/neurometrics", "layout": neurometrics.layout},
    {"name": "Readiness", "path": "/readiness", "layout": readiness.layout},
    {"name": "Scheduler", "path": "/scheduler", "layout": scheduler.layout},
    {"name": "Memory", "path": "/memory", "layout": memory.layout},
    {"name": "Spotify Resonance", "path": "/spotify", "layout": spotify_resonance.layout},
    {"name": "Media Alchemy", "path": "/media_alchemy", "layout": media_alchemy.layout},
    {"name": "Live Media", "path": "/live_media", "layout": live_media.layout},
    {"name": "Spotify Insights", "path": "/spotify_insights", "layout": spotify_insights.layout},
    {"name": "Continuum Tracker", "path": "/continuum", "layout": continuum.layout},
    {"name": "Longitudinal Insights", "path": "/insights", "layout": longitudinal.layout},
    {"name": "Environment", "path": "/environment", "layout": environment.layout},
    {"name": "Recipes", "path": "/recipes", "layout": recipes.layout},
    {"name": "Settings", "path": "/settings", "layout": settings.layout},
    {"name": "Manifest", "path": "/manifest", "layout": manifest.layout()},
]

PAGE_MAP = {page["path"]: page for page in PAGES}


def get_nav_links():
    """Return navigation links for the sidebar."""
    return [
        dbc.NavLink(page["name"], href=page["path"], active="exact", className="sidebar-link")
        for page in PAGES
    ]


def register_page_callbacks(app):
    """Wire routing callback that swaps page content based on pathname."""

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def render_page_content(pathname: str):
        if not pathname:
            return PAGE_MAP["/"]["layout"]
        page = PAGE_MAP.get(pathname)
        if page:
            return page["layout"]
        return dbc.Container(
            [dbc.Alert(f"404: {pathname} not found", color="danger", className="mt-3")],
            fluid=True,
            className="page-container",
        )


__all__ = ["PAGES", "get_nav_links", "register_page_callbacks"]

