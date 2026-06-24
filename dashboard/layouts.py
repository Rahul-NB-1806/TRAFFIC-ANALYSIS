from dash import dcc, html
from dashboard.components import page_wrapper, card, filter_bar


def _header(title, subtitle=""):
    return html.Div(className="page-header", children=[
        html.H1(title),
        html.P(subtitle) if subtitle else None,
        html.Div(className="page-header-divider"),
    ])


# ─── HOME ───
home_layout = page_wrapper([
    html.Div(id="home-kpi-strip"),
    html.Div(id="home-peak-hours"),
    html.Div(className="card-dash-grid-3", style={"marginTop": "20px"}, children=[
        card("Traffic Composition Over Time", "home-stacked"),
        card("Vehicle Type Distribution", "home-pie"),
        card("Hourly Traffic Profile", "home-hourly-bar"),
    ]),
    html.Div(className="card-dash", style={"marginTop": "20px", "padding": "0", "overflow": "hidden"}, children=[
        html.Div(id="home-location-table", style={"padding": "20px"}),
    ]),
], active="home")

# ─── FORECAST ───
forecast_layout = page_wrapper([
    _header("Traffic Forecast", "Predict congestion using machine learning models"),
    filter_bar([
        html.Label("Location"),
        dcc.Dropdown(id="forecast-location", options=[], placeholder="All", clearable=True, style={"width": "180px", "minWidth": "140px"}),
        html.Label("Ahead"),
        dcc.Input(id="forecast-hours", type="number", value=24, min=1, className="input-dash", style={"width": "80px"}),
        html.Label("Model"),
        dcc.Dropdown(id="forecast-model", options=[
            {"label": "Auto", "value": "auto"},
            {"label": "Linear", "value": "linear"},
            {"label": "Random Forest", "value": "rf"},
        ], value="auto", clearable=False, style={"width": "160px"}),
        html.Button("Predict", id="forecast-btn", className="btn-dash btn-dash-primary"),
    ]),
    html.Div(id="forecast-status"),
    html.Div(id="forecast-metrics"),
    html.Div(className="card-dash-grid-3", children=[
        card("Traffic Forecast", "forecast-line"),
        card("Vehicle Distribution", "forecast-pie"),
        card("Hourly Profile", "forecast-hourly-bar"),
    ]),
], active="forecast")

# ─── UPLOAD ───
upload_layout = page_wrapper([
    _header("Upload Data", "Import CSV files to feed the dashboard"),
    html.Div(className="card-dash", style={"marginBottom": "20px"}, children=[
        dcc.Upload(
            id="upload-zone",
            children=html.Div([
                html.Div("\u2B06", className="upload-zone-icon"),
                html.Div("Drop CSV here or click to browse", className="upload-zone-text"),
                html.Div("date, time, location, two_wheeler, four_wheeler, heavy_vehicle, emergency_vehicle", className="upload-zone-hint"),
            ]),
            className="upload-zone",
        ),
        html.Div(id="upload-report"),
    ]),
    html.Div(className="card-dash", children=[
        html.H5(className="card-dash-title", children=[html.Span(className="title-indicator"), "Uploaded Files"]),
        html.Div(id="upload-file-list"),
    ]),
], active="upload")

# ─── ANALYTICS ───
analytics_layout = page_wrapper([
    _header("Traffic Analytics", "Deep-dive with radar, gauge, and anomaly detection"),
    filter_bar([
        html.Label("Location"),
        dcc.Input(id="analytics-location", type="text", placeholder="Junction A", className="input-dash", style={"width": "160px"}),
        html.Label("From"),
        dcc.Input(id="analytics-from", type="text", placeholder="YYYY-MM-DD HH:MM", className="input-dash", style={"width": "180px"}),
        html.Label("To"),
        dcc.Input(id="analytics-to", type="text", placeholder="YYYY-MM-DD HH:MM", className="input-dash", style={"width": "180px"}),
        html.Button("Apply", id="analytics-apply", className="btn-dash btn-dash-primary"),
    ]),
    html.Div(className="card-dash-grid-3", children=[
        card("Vehicle Type Radar", "analytics-radar"),
        card("Average Per Hour", "analytics-gauge"),
        card("Hourly Distribution", "analytics-hourly-bar"),
    ]),
    html.Div(className="card-dash", style={"marginTop": "20px"}, children=[
        card("Anomaly Detection", "analytics-anomalies", height="360px"),
    ]),
], active="analytics")
