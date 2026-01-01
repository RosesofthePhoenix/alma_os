from dash import Dash
import dash_bootstrap_components as dbc

from alma import config
from alma.engine import storage
from alma.app_state import registry
from alma.ui.layout import build_layout
from alma.ui import pages


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
    app.layout = build_layout()
    pages.register_page_callbacks(app)
    # Expose shared service registry on the app instance for future use.
    app.registry = registry
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)

