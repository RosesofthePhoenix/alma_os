import os
from dash import Dash
import dash_bootstrap_components as dbc

from alma import config
from alma.engine import storage
from alma.app_state import registry

# Clean up any old Spotipy caches to avoid interactive auth prompts.
for cache_file in [".cache", ".cache-spotify", ".cache-spotify-real-sections", "token.json"]:
    try:
        os.remove(cache_file)
    except Exception:
        pass


def create_app() -> Dash:
    """Instantiate the Dash app with sidebar layout and routing callbacks."""
    storage.init_db()
    config.ensure_required_paths()
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        suppress_callback_exceptions=True,
        assets_folder="alma/ui",
    )
    app.title = "alma_os"

    # Import layout and pages AFTER app is created so @callback decorators bind to this app.
    from alma.ui import pages
    from alma.ui.layout import build_layout

    app.layout = build_layout()
    pages.register_page_callbacks(app)
    # Expose shared service registry on the app instance for future use.
    app.registry = registry
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)

