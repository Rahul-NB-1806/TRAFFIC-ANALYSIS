import os
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from flask import Flask
import dash
from dash import dcc, html
from core.config import settings
from core.database import init_db
from dashboard.callbacks import register_callbacks

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app():
    init_db()
    logger.info("Traffic dashboard starting \u2014 database=%s", settings.database_url)

    server = Flask(__name__)
    app = dash.Dash(
        __name__,
        server=server,
        suppress_callback_exceptions=True,
        title="Pulse \u00b7 Traffic Monitor",
    )
    app._favicon = None

    app.layout = html.Div(id="app-container", children=[
        dcc.Location(id="url"),
        html.Div(id="page-content"),
    ])

    register_callbacks(app)

    return app, server


app, server = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    logger.info("Serving on http://127.0.0.1:%d", port)
    app.run(debug=settings.debug, host="0.0.0.0", port=port)
