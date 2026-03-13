# ============================================================
# Smart Traffic Dashboard with Radar Chart + Semi-Circle Gauge
# ============================================================

import os, glob, shutil, base64
from datetime import datetime
import pandas as pd
import numpy as np
import pytz

from flask import Flask
from sklearn.linear_model import LinearRegression

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

# ========================================
# CONFIG
# ========================================
UPLOAD_DIR = "./data/uploads"
ARCHIVE_DIR = "./data/archive"
DEFAULT_TZ = "Asia/Kolkata"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# ========================================
# APP INIT
# ========================================
server = Flask(__name__)
external_stylesheets = [
    "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/darkly/bootstrap.min.css"
]
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)

app.title = "Smart Traffic Dashboard"

# ========================================
# UTILS
# ========================================

def save_uploaded_contents(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    safe = filename
    path = os.path.join(UPLOAD_DIR, safe)

    if os.path.exists(path):
        base, ext = os.path.splitext(safe)
        safe = f"{base}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{ext}"
        path = os.path.join(UPLOAD_DIR, safe)

    with open(path, "wb") as f:
        f.write(decoded)
    return path


def normalize_timestamp_series(ts, timezone=DEFAULT_TZ):
    tz = pytz.timezone(timezone)
    idx = pd.to_datetime(ts, errors="coerce")
    if idx.dt.tz is None:
        return idx.dt.tz_localize(tz)
    return idx


def read_csv_flexible(path, timezone=DEFAULT_TZ):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        lc = {c.lower(): c for c in df.columns}
        if "date" in lc and any("time" in lc for lc in df.columns):
            tcol = next((lc for lc in df.columns if "time" in lc.lower()), None)
            ts_col = "timestamp"
            df[ts_col] = df[lc["date"]].astype(str) + " " + df[tcol].astype(str)
        else:
            ts_col = df.columns[0]

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).copy()
    df["timestamp"] = normalize_timestamp_series(df[ts_col], timezone)
    df = df.sort_values("timestamp")

    return df.reset_index(drop=True)


def combine_all_uploads(timezone=DEFAULT_TZ):
    files = glob.glob(os.path.join(UPLOAD_DIR, "*.csv"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            d = read_csv_flexible(f, timezone)
            dfs.append(d)
        except:
            pass

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True).sort_values("timestamp")


def compute_pie_counts(df):
    if df.empty:
        return None

    return {
        "4wheeler": df["four_wheeler"].sum(),
        "2wheeler": df["two_wheeler"].sum(),
        "heavy": df["heavy_vehicle"].sum(),
        "emergency": df["emergency_vehicle"].sum(),
    }


def compute_mean_vehicle_types(df):
    """Radar Chart uses mean values (option 2)"""
    if df.empty:
        return None

    return {
        "4wheeler": df["four_wheeler"].mean(),
        "2wheeler": df["two_wheeler"].mean(),
        "heavy": df["heavy_vehicle"].mean(),
        "emergency": df["emergency_vehicle"].mean(),
    }


def compute_hourly_avg(df):
    """Used for semi-circle gauge"""
    if df.empty:
        return 0, 1

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["total"] = (
        df["two_wheeler"]
        + df["four_wheeler"]
        + df["heavy_vehicle"]
        + df["emergency_vehicle"]
    )
    avg = df.groupby("hour")["total"].mean()

    return avg.mean(), avg.max()


# ========================================
# NAVBAR
# ========================================
nav = html.Nav(
    className="navbar navbar-dark bg-primary",
    children=[
        html.Div(
            className="container",
            children=[
                html.Span(
                    [
                        html.Img(
                            src="https://img.icons8.com/ios-filled/50/ffffff/dashboard.png",
                            style={"height": "34px", "marginRight": "8px"},
                        ),
                        html.Span(
                            "Smart Traffic Dashboard",
                            className="navbar-brand",
                            style={"fontSize": "20px"},
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Link(
                            "Home",
                            href="/",
                            className="btn btn-link",
                            style={"color": "#fff", "marginRight": "10px"},
                        ),
                        dcc.Link(
                            "Smart Congestion Forecast AI",
                            href="/forecast",
                            className="btn btn-link",
                            style={"color": "#fff", "marginRight": "10px"},
                        ),
                        dcc.Link(
                            "Upload",
                            href="/upload",
                            className="btn btn-link",
                            style={"color": "#fff", "marginRight": "10px"},
                        ),
                        dcc.Link(
                            "Time-Stamped Analysis",
                            href="/analyses",
                            className="btn btn-link",
                            style={"color": "#fff"},
                        ),
                    ],
                    style={"float": "right", "marginTop": "6px"},
                ),
            ],
        )
    ],
)

# ========================================
# HOME PAGE
# ========================================
home_layout = html.Div(
    [
        nav,
        html.Div(
            className="container",
            style={"marginTop": "18px"},
            children=[
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="col-md-6",
                            children=[
                                html.Div(
                                    className="card bg-dark",
                                    style={"padding": "12px", "height": "560px"},
                                    children=[
                                        html.H5(
                                            "Vehicle Count Over Time",
                                            style={"color": "#fff"},
                                        ),
                                        dcc.Graph(
                                            id="home-line",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                            },
                                            style={"height": "500px"},
                                        ),
                                    ],
                                )
                            ],
                        ),
                        html.Div(
                            className="col-md-6",
                            children=[
                                html.Div(
                                    className="card bg-dark",
                                    style={"padding": "12px", "height": "560px"},
                                    children=[
                                        html.H5(
                                            "Vehicle Type Distribution",
                                            style={"color": "#fff"},
                                        ),
                                        dcc.Graph(
                                            id="home-pie",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                            },
                                            style={"height": "500px"},
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                )
            ],
        ),
    ]
)

# ========================================
# FORECAST PAGE (unchanged)
# ========================================
forecast_layout = html.Div(
    [
        nav,
        html.Div(
            className="container",
            style={"marginTop": "18px"},
            children=[
                html.H3(
                    "Smart Congestion Forecast AI", style={"color": "#fff"}
                ),
                html.Div(
                    className="card bg-dark",
                    style={"padding": "12px", "marginBottom": "16px"},
                    children=[
                        html.Div(
                            className="form-inline",
                            children=[
                                html.Label(
                                    "Location:",
                                    style={
                                        "color": "#fff",
                                        "marginRight": "8px",
                                    },
                                ),
                                html.Div(
                                    dcc.Dropdown(
                                        id="forecast-location",
                                        options=[],
                                        placeholder="Select location (optional)",
                                        clearable=True,
                                        style={"width": "260px"},
                                    ),
                                    style={"marginRight": "20px"},
                                ),
                                html.Label(
                                    "Date:",
                                    style={
                                        "color": "#fff",
                                        "marginRight": "8px",
                                    },
                                ),
                                dcc.Input(
                                    id="forecast-date",
                                    type="text",
                                    placeholder="YYYY-MM-DD",
                                    style={"width": "140px", "marginRight": "12px"},
                                ),
                                html.Label(
                                    "Time:",
                                    style={
                                        "color": "#fff",
                                        "marginRight": "8px",
                                    },
                                ),
                                dcc.Input(
                                    id="forecast-time",
                                    type="text",
                                    placeholder="HH:MM",
                                    style={"width": "90px", "marginRight": "12px"},
                                ),
                                html.Label(
                                    "Years Ahead:",
                                    style={
                                        "color": "#fff",
                                        "marginRight": "8px",
                                    },
                                ),
                                dcc.Input(
                                    id="forecast-years",
                                    type="number",
                                    value=1,
                                    min=1,
                                    max=10,
                                    style={
                                        "width": "80px",
                                        "marginRight": "12px",
                                    },
                                ),
                                html.Button(
                                    "Predict",
                                    id="forecast-predict",
                                    className="btn btn-success",
                                ),
                            ],
                        ),
                        html.Div(
                            id="forecast-status",
                            style={"color": "#fff", "marginTop": "10px"},
                        ),
                    ],
                ),
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="col-md-6",
                            children=[
                                html.Div(
                                    className="card bg-dark",
                                    style={"padding": "12px", "height": "560px"},
                                    children=[
                                        html.H5(
                                            "Yearly Average & Prediction",
                                            style={"color": "#fff"},
                                        ),
                                        dcc.Graph(
                                            id="forecast-year-line",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                            },
                                            style={"height": "500px"},
                                        ),
                                    ],
                                )
                            ],
                        ),
                        html.Div(
                            className="col-md-6",
                            children=[
                                html.Div(
                                    className="card bg-dark",
                                    style={"padding": "12px", "height": "560px"},
                                    children=[
                                        html.H5(
                                            "Vehicle Type Distribution",
                                            style={"color": "#fff"},
                                        ),
                                        dcc.Graph(
                                            id="forecast-pie",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                            },
                                            style={"height": "500px"},
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
)


# ================
# UPLOAD PAGE
# ================
upload_layout = html.Div(
    [
        nav,
        html.Div(
            className="container",
            style={"marginTop": "18px"},
            children=[
                html.H3("Upload CSV Files", style={"color": "#fff"}),
                html.Div(
                    className="card bg-dark",
                    style={"padding": "12px"},
                    children=[
                        dcc.Upload(
                            id="upload-component",
                            children=html.Button(
                                "Upload CSV", className="btn btn-secondary"
                            ),
                            multiple=False,
                        ),
                        html.Div(
                            id="upload-result",
                            style={"color": "#fff", "marginTop": "12px"},
                        ),
                    ],
                ),
            ],
        ),
    ]
)

# ============================================
# TIME-STAMPED ANALYSIS PAGE (Radar + Gauge)
# ============================================

analyses_layout = html.Div(
    [
        nav,
        html.Div(
            className="container",
            style={"marginTop": "18px"},
            children=[
                html.H3(
                    "Time-Stamped Traffic Analysis",
                    style={"color": "#fff"},
                ),
                html.Div(
                    className="card bg-dark",
                    style={"padding": "12px", "marginBottom": "20px"},
                    children=[
                        html.Div(
                            className="form-inline",
                            children=[
                                html.Label(
                                    "Location:",
                                    style={"color": "#fff", "marginRight": "8px"},
                                ),
                                dcc.Input(
                                    id="analyses-location",
                                    type="text",
                                    placeholder="Junction A",
                                    style={"marginRight": "12px"},
                                ),
                                html.Label(
                                    "From:",
                                    style={"color": "#fff", "marginRight": "8px"},
                                ),
                                dcc.Input(
                                    id="analyses-from",
                                    type="text",
                                    placeholder="YYYY-MM-DD HH:MM",
                                    style={
                                        "width": "180px",
                                        "marginRight": "12px",
                                    },
                                ),
                                html.Label(
                                    "To:",
                                    style={"color": "#fff", "marginRight": "8px"},
                                ),
                                dcc.Input(
                                    id="analyses-to",
                                    type="text",
                                    placeholder="YYYY-MM-DD HH:MM",
                                    style={
                                        "width": "180px",
                                        "marginRight": "12px",
                                    },
                                ),
                                html.Button(
                                    "Filter",
                                    id="analyses-filter",
                                    className="btn btn-primary",
                                    style={"marginLeft": "10px"},
                                ),
                            ],
                        )
                    ],
                ),
                # ====================================
                # RADAR + SEMI-CIRCLE GAUGE
                # ====================================
                html.Div(
                    className="row",
                    children=[
                        # ---- Radar Chart ----
                        html.Div(
                            className="col-md-6",
                            children=[
                                html.Div(
                                    className="card bg-dark",
                                    style={"padding": "12px", "height": "520px"},
                                    children=[
                                        html.H5(
                                            "Vehicle Type Radar Chart (Avg Values)",
                                            style={"color": "#fff"},
                                        ),
                                        dcc.Graph(
                                            id="analyses-radar",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                            },
                                            style={"height": "450px"},
                                        ),
                                    ],
                                )
                            ],
                        ),
                        # ---- Semi-Circle Gauge ----
                        html.Div(
                            className="col-md-6",
                            children=[
                                html.Div(
                                    className="card bg-dark",
                                    style={"padding": "12px", "height": "520px"},
                                    children=[
                                        html.H5(
                                            "Average Vehicles Per Hour",
                                            style={"color": "#fff"},
                                        ),
                                        dcc.Graph(
                                            id="analyses-gauge",
                                            config={
                                                "displayModeBar": True,
                                                "displaylogo": False,
                                            },
                                            style={"height": "450px"},
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]
)


# ========================================
# MAIN LAYOUT
# ========================================
app.layout = html.Div([
    dcc.Location(id="url"),

    # 🔹 Hidden components so Dash knows these IDs exist
    html.Div([
        dcc.Dropdown(id="forecast-location"),
        dcc.Graph(id="forecast-year-line"),
        dcc.Graph(id="forecast-pie"),
        html.Div(id="forecast-status"),
    ], style={"display": "none"}),

    html.Div(id="page-content")
])



# ========================================
# PAGE ROUTING
# ========================================
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(path):
    if path == "/forecast":
        return forecast_layout
    if path == "/upload":
        return upload_layout
    if path == "/analyses":
        return analyses_layout
    return home_layout


# ========================================
# HOME CALLBACKS
# ========================================
@app.callback(
    Output("home-line", "figure"),
    Output("home-pie", "figure"),
    Input("url", "pathname"),
)
def update_home(path):
    df = combine_all_uploads()

    empty = go.Figure()
    empty.update_layout(
        title="No Data",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="#fff",
    )

    if df.empty:
        return empty, empty

    # --- Line chart (1-min resolution)
    ts = (
        df.set_index("timestamp")
        .resample("1min")
        .size()
        .rename("count")
        .fillna(0)
    )

    line_fig = go.Figure()
    line_fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts.values,
            mode="lines",
            line={"width": 2},
        )
    )
    line_fig.update_layout(
        title="Vehicles Over Time",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="#fff",
    )

    # --- Pie chart
    pie_data = compute_pie_counts(df)

    pie = go.Figure(
        data=[
            go.Pie(
                labels=list(pie_data.keys()),
                values=list(pie_data.values()),
                hole=0.35,
            )
        ]
    )
    pie.update_layout(
        title="Vehicle Type Distribution",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="#fff",
    )

    return line_fig, pie


# ========================================
# UPLOAD CALLBACK
# ========================================
@app.callback(
    Output("upload-result", "children"),
    Input("upload-component", "contents"),
    State("upload-component", "filename"),
)
def upload_csv(contents, filename):
    if contents and filename:
        path = save_uploaded_contents(contents, filename)
        return f"Uploaded: {os.path.basename(path)}"
    return ""


def compute_hourly_series(df):
    
    df = df.copy()

    df["total"] = (
        df["two_wheeler"]
        + df["four_wheeler"]
        + df["heavy_vehicle"]
        + df["emergency_vehicle"]
    )

    hourly = (
        df.set_index("timestamp")
        .resample("1H")["total"]
        .mean()
        .reset_index()
    )

    hourly["hour_index"] = range(len(hourly))
    return hourly


# ================================
# FORECAST CALLBACK  ✅ ADD HERE
# ================================
@app.callback(
    Output("forecast-location", "options"),
    Output("forecast-year-line", "figure"),
    Output("forecast-pie", "figure"),
    Output("forecast-status", "children"),
    Input("forecast-predict", "n_clicks"),
    State("forecast-location", "value"),
    State("forecast-date", "value"),
    State("forecast-time", "value"),
    State("forecast-years", "value"),
)
def run_forecast(n_clicks, location, date, time, years):

    df = combine_all_uploads()

    empty = go.Figure()
    empty.update_layout(
        title="No Data",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="#fff",
    )

    # Populate location dropdown
    options = []
    if not df.empty and "location" in df.columns:
        options = [{"label": l, "value": l} for l in sorted(df["location"].dropna().unique())]

    if not n_clicks:
        return options, empty, empty, "Waiting for input..."

    if df.empty:
        return options, empty, empty, "No CSV data uploaded."

    # -----------------------
    # FILTER DATA
    # -----------------------
    filtered = df.copy()

    if location:
        filtered = filtered[filtered["location"].astype(str).str.contains(location, case=False)]

    if date:
        try:
            filtered = filtered[filtered["timestamp"].dt.date == pd.to_datetime(date).date()]
        except:
            pass



    if filtered.empty:
        return options, empty, empty, "No data after filtering."

    # -----------------------
    # YEARLY AVERAGE
    # -----------------------
    hourly = compute_hourly_series(filtered)

    if len(hourly) < 3:
        return options, empty, empty, "Not enough hourly data."

    X = hourly[["hour_index"]]
    y = hourly["total"]

    model = LinearRegression()
    model.fit(X, y)

    future_hours = int(years) * 24
    last_idx = hourly["hour_index"].iloc[-1]

    future_idx = range(last_idx + 1, last_idx + future_hours + 1)
    predictions = model.predict(
        np.array(list(future_idx)).reshape(-1, 1)
    )


    # -----------------------
    # LINE GRAPH
        # -----------------------
    line = go.Figure()

    line.add_trace(go.Scatter(
        x=hourly["timestamp"],
        y=y,
        mode="lines+markers",
        name="Historical"
    ))

    future_time = pd.date_range(
        start=hourly["timestamp"].iloc[-1],
        periods=len(predictions) + 1,
        freq="H"
    )[1:]

    line.add_trace(go.Scatter(
        x=future_time,
        y=predictions,
        mode="lines",
        name="Predicted",
        line=dict(dash="dash")
    ))

    line.update_layout(
        title="Hourly Traffic Forecast",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="#fff",
    )

    pie = go.Figure(go.Pie(
        labels=["2W", "4W", "Heavy", "Emergency"],
        values=[
            filtered["two_wheeler"].sum(),
            filtered["four_wheeler"].sum(),
            filtered["heavy_vehicle"].sum(),
            filtered["emergency_vehicle"].sum(),
        ],
        hole=0.35,
    ))

    pie.update_layout(
        title="Vehicle Type Distribution",
        paper_bgcolor="#111",
        font_color="#fff",
    )

    return options, line, pie, "Hourly forecast generated successfully."


# ========================================
# ANALYSES CALLBACK  (Radar + Gauge)
# ========================================
@app.callback(
    Output("analyses-radar", "figure"),
    Output("analyses-gauge", "figure"),
    Input("analyses-filter", "n_clicks"),
    State("analyses-location", "value"),
    State("analyses-from", "value"),
    State("analyses-to", "value"),
)
def analyses_run(n, location, from_ts, to_ts):

    df = combine_all_uploads()

    empty = go.Figure()
    empty.update_layout(
        title="No Data",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="#fff",
    )

    if df.empty:
        return empty, empty

    # --------------------------------------
    # APPLY FILTERS
    # --------------------------------------
    filtered = df.copy()

    if location:
        filtered = filtered[
            filtered["location"].astype(str).str.contains(location, case=False)
        ]

    if from_ts:
        try:
            filtered = filtered[
                filtered["timestamp"] >= pd.to_datetime(from_ts)
            ]
        except:
            pass

    if to_ts:
        try:
            filtered = filtered[
                filtered["timestamp"] <= pd.to_datetime(to_ts)
            ]
        except:
            pass

    if filtered.empty:
        return empty, empty

    # ======================================
    # RADAR CHART (mean values)
    # ======================================
    means = compute_mean_vehicle_types(filtered)

    radar = go.Figure()

    radar.add_trace(
        go.Scatterpolar(
            r=list(means.values()),
            theta=list(means.keys()),
            fill="toself",
            name="Average",
        )
    )

    radar.update_layout(
        title="Vehicle Type Radar Chart (Mean Values)",
        polar=dict(
            bgcolor="#111",
            radialaxis=dict(visible=True, color="#fff"),
            angularaxis=dict(color="#fff"),
        ),
        showlegend=False,
        paper_bgcolor="#111",
        font_color="#fff",
    )

    # ======================================
    # SEMI-CIRCLE GAUGE
    # ======================================
    avg, maxv = compute_hourly_avg(filtered)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg,
            gauge={
                "axis": {"range": [0, maxv], "visible": True},
                "bar": {"color": "orange"},
                "bgcolor": "#333",
                "borderwidth": 2,
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    gauge.update_layout(
        title="Avg Vehicles per Hour (Auto-Scaled)",
        paper_bgcolor="#111",
        font_color="#fff",
    )

    return radar, gauge


# ========================================
# RUN SERVER
# ========================================
if __name__ == "__main__":
    print("Running on http://127.0.0.1:8050")
    app.run(debug=True)