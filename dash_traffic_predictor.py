import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from flask import Flask

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px

from backend.config import settings
from backend.logging_config import setup_logging, logger
from backend.database import init_db
from backend.data_pipeline import (
    combine_all_uploads,
    validate_and_ingest,
    compute_pie_counts,
    compute_mean_vehicle_types,
    compute_hourly_avg,
    compute_hourly_series,
)
from ml.features import create_time_features, create_rolling_features, prepare_forecast_data
from ml.models import train_model, predict, evaluate, train_test_split_temporal
from ml.registry import save_model

setup_logging(settings.log_level)
init_db()
logger.info("Traffic dashboard starting up")

server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
)
app.title = "Smart Traffic Dashboard"

# ─── Plotly theme defaults ───
CHART_COLORS = ["#d45d4a", "#4a7c7a", "#c9a84c", "#c94a4a"]

THEME = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=11, color="#5a5a7a"),
        margin=dict(l=40, r=16, t=32, b=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#e2d9cc",
            font=dict(family="DM Sans", size=11, color="#1a1a2e"),
        ),
        xaxis=dict(
            gridcolor="#ede7df",
            linecolor="#ede7df",
            tickcolor="#9a9ab0",
            tickfont=dict(size=10, color="#9a9ab0"),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#ede7df",
            linecolor="#ede7df",
            tickcolor="#9a9ab0",
            tickfont=dict(size=10, color="#9a9ab0"),
            zeroline=False,
        ),
        legend=dict(
            font=dict(size=10, color="#5a5a7a"),
            bgcolor="rgba(0,0,0,0)",
        ),
        dragmode=False,
    )
)


def apply_theme(fig):
    fig.update_layout(**THEME["layout"])
    for t in fig.data:
        if hasattr(t, "line") and t.line is not None:
            t.line.width = 1.5
    return fig


# ─── Shared components ───

def status_bar(active_page="home"):
    links = [
        ("/", "Home", "home"),
        ("/forecast", "Forecast", "forecast"),
        ("/upload", "Upload", "upload"),
        ("/analyses", "Analytics", "analyses"),
    ]
    nav_links = []
    for href, label, key in links:
        cls = "active" if key == active_page else ""
        nav_links.append(html.A(label, href=href, className=cls))
    return html.Header(
        className="status-bar",
        children=[
            html.A(
                className="status-bar-brand",
                href="/",
                children=[
                    html.Span(className="brand-dot"),
                    html.Span([
                        "TRAFFIC CTRL",
                        html.Span(" v1", className="brand-label"),
                    ]),
                ],
            ),
            html.Nav(className="status-nav", children=nav_links),
        ],
    )


def kpi_strip(df):
    total = len(df)
    locations = df["location"].nunique() if not df.empty else 0
    two_w = int(df["two_wheeler"].sum()) if not df.empty else 0
    four_w = int(df["four_wheeler"].sum()) if not df.empty else 0
    heavy = int(df["heavy_vehicle"].sum()) if not df.empty else 0
    emergency = int(df["emergency_vehicle"].sum()) if not df.empty else 0
    total_v = two_w + four_w + heavy + emergency
    if not df.empty:
        tr = f"{df['timestamp'].min().strftime('%d %b %H:%M')} — {df['timestamp'].max().strftime('%d %b %H:%M')}"
    else:
        tr = "—"
    return html.Div(
        className="kpi-strip",
        children=[
            html.Div(className="kpi-item", children=[
                html.Span("Total Vehicles", className="kpi-label"),
                html.Span(f"{total_v:,}", className="kpi-value accent-teal"),
                html.Span(f"Across {locations} locations", className="kpi-sub"),
            ]),
            html.Div(className="kpi-item", children=[
                html.Span("2-Wheelers", className="kpi-label"),
                html.Span(f"{two_w:,}", className="kpi-value accent-blue"),
                html.Span(f"{two_w / max(total_v, 1) * 100:.0f}% of traffic", className="kpi-sub"),
            ]),
            html.Div(className="kpi-item", children=[
                html.Span("4-Wheelers", className="kpi-label"),
                html.Span(f"{four_w:,}", className="kpi-value accent-purple"),
                html.Span(f"{four_w / max(total_v, 1) * 100:.0f}% of traffic", className="kpi-sub"),
            ]),
            html.Div(className="kpi-item", children=[
                html.Span("Heavy / Emergency", className="kpi-label"),
                html.Span(f"{heavy + emergency:,}", className="kpi-value accent-amber"),
                html.Span(f"{(heavy + emergency) / max(total_v, 1) * 100:.0f}% of traffic", className="kpi-sub"),
            ]),
            html.Div(className="kpi-item", children=[
                html.Span("Time Range", className="kpi-label"),
                html.Span(tr if not df.empty else "—", className="kpi-value", style={"fontSize": "14px"}),
                html.Span(f"{total} records", className="kpi-sub"),
            ]),
        ],
    )


def card(title, graph_id, height="460px"):
    return html.Div(
        className="card-dash",
        style={"height": height},
        children=[
            html.H5(className="card-dash-title", children=[
                html.Span(className="title-indicator"),
                title,
            ]),
            dcc.Graph(
                id=graph_id,
                config={"displayModeBar": False, "displaylogo": False},
                style={"height": f"calc({height} - 48px)"},
            ),
        ],
    )


def empty_figure(msg="No Data"):
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=13, color="#9a9ab0", family="DM Sans"),
    )
    return apply_theme(fig)


def page_wrapper(content, active_page="home"):
    return html.Div([
        status_bar(active_page),
        html.Div(className="page-container", children=content),
    ])


# ─── HOME PAGE ───
home_layout = page_wrapper([
    html.Div(id="home-kpi-strip"),
    html.Div(className="card-dash-grid-3", style={"marginTop": "20px"}, children=[
        card("Traffic Composition Over Time", "home-stacked"),
        card("Vehicle Type Distribution", "home-pie"),
        card("Traffic by Location", "home-location-bar"),
    ]),
], active_page="home")

# ─── FORECAST PAGE ───
forecast_layout = page_wrapper([
    html.Div(className="page-header", children=[
        html.H1("Smart Congestion Forecast"),
        html.P("Predict traffic volume using machine learning models"),
        html.Div(className="page-header-divider"),
    ]),
    html.Div(className="filter-bar", children=[
        html.Label("Location"),
        dcc.Dropdown(
            id="forecast-location", options=[], placeholder="All locations",
            clearable=True, style={"width": "220px", "minWidth": "160px"},
        ),
        html.Label("Date"),
        dcc.Input(id="forecast-date", type="text", placeholder="YYYY-MM-DD", className="input-dash", style={"width": "140px"}),
        html.Label("Ahead"),
        dcc.Input(id="forecast-hours", type="number", value=24, min=1, className="input-dash", style={"width": "80px"}),
        html.Label("Model"),
        dcc.Dropdown(id="forecast-model-type", options=[
            {"label": "Linear Reg", "value": "linear"},
            {"label": "Random Forest", "value": "rf"},
            {"label": "Auto", "value": "auto"},
        ], value="auto", clearable=False, style={"width": "160px"}),
        html.Button("Predict", id="forecast-predict", className="btn-dash btn-dash-primary"),
    ]),
    html.Div(id="forecast-status"),
    html.Div(id="forecast-metrics"),
    html.Div(className="card-dash-grid-3", children=[
        card("Traffic Forecast", "forecast-line"),
        card("Vehicle Type Distribution", "forecast-pie"),
        card("Hourly Traffic Profile", "forecast-hourly-bar"),
    ]),
], active_page="forecast")

# ─── UPLOAD PAGE ───
upload_layout = page_wrapper([
    html.Div(className="page-header", children=[
        html.H1("Upload Traffic Data"),
        html.P("Import CSV files to feed the dashboard"),
        html.Div(className="page-header-divider"),
    ]),
    html.Div(className="card-dash", style={"marginBottom": "20px"}, children=[
        dcc.Upload(
            id="upload-component",
            children=html.Div([
                html.Div("⬆", className="upload-zone-icon"),
                html.Div("Drop CSV here or click to browse", className="upload-zone-text"),
                html.Div("Columns: date, time, location, two_wheeler, four_wheeler, heavy_vehicle, emergency_vehicle", className="upload-zone-hint"),
            ]),
            className="upload-zone",
        ),
        html.Div(id="upload-result"),
    ]),
    html.Div(className="card-dash", children=[
        html.H5(className="card-dash-title", children=[
            html.Span(className="title-indicator"),
            "Uploaded Files",
        ]),
        html.Div(id="upload-file-list", style={"color": "#8892a8", "fontFamily": "IBM Plex Mono", "fontSize": "12px"}),
    ]),
], active_page="upload")

# ─── ANALYSES PAGE ───
analyses_layout = page_wrapper([
    html.Div(className="page-header", children=[
        html.H1("Time-Stamped Traffic Analysis"),
        html.P("Filter by location and time range to inspect patterns"),
        html.Div(className="page-header-divider"),
    ]),
    html.Div(className="filter-bar", children=[
        html.Label("Location"),
        dcc.Input(id="analyses-location", type="text", placeholder="Junction A", className="input-dash", style={"width": "160px"}),
        html.Label("From"),
        dcc.Input(id="analyses-from", type="text", placeholder="YYYY-MM-DD HH:MM", className="input-dash", style={"width": "180px"}),
        html.Label("To"),
        dcc.Input(id="analyses-to", type="text", placeholder="YYYY-MM-DD HH:MM", className="input-dash", style={"width": "180px"}),
        html.Button("Filter", id="analyses-filter", className="btn-dash btn-dash-primary"),
    ]),
    html.Div(className="card-dash-grid-3", children=[
        card("Vehicle Type Radar (Avg)", "analyses-radar"),
        card("Average Vehicles Per Hour", "analyses-gauge"),
        card("Traffic by Hour of Day", "analyses-hourly-bar"),
    ]),
], active_page="analyses")

# ─── MAIN LAYOUT ───
app.layout = html.Div(id="app-container", children=[
    dcc.Location(id="url"),
    html.Div([
        dcc.Dropdown(id="forecast-location"),
        dcc.Graph(id="forecast-line"),
        dcc.Graph(id="forecast-pie"),
        dcc.Graph(id="forecast-hourly-bar"),
        dcc.Graph(id="home-stacked"),
        dcc.Graph(id="home-pie"),
        dcc.Graph(id="home-location-bar"),
        dcc.Graph(id="analyses-radar"),
        dcc.Graph(id="analyses-gauge"),
        dcc.Graph(id="analyses-hourly-bar"),
        html.Div(id="home-kpi-strip"),
        html.Div(id="forecast-status"),
        html.Div(id="forecast-metrics"),
        html.Div(id="upload-result"),
        html.Div(id="upload-file-list"),
    ], style={"display": "none"}),
    html.Div(id="page-content"),
])

# ─── PAGE ROUTING ───
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(path):
    if path == "/forecast":
        return forecast_layout
    if path == "/upload":
        return upload_layout
    if path == "/analyses":
        return analyses_layout
    return home_layout


# ─── HOME CALLBACKS ───
@app.callback(
    Output("home-stacked", "figure"),
    Output("home-pie", "figure"),
    Output("home-location-bar", "figure"),
    Output("home-kpi-strip", "children"),
    Input("url", "pathname"),
)
def update_home(path):
    df = combine_all_uploads()
    kpi = kpi_strip(df)
    if df.empty:
        return empty_figure("Upload CSV data to get started"), empty_figure(), empty_figure("Upload CSV data to get started"), kpi

    # ── Stacked area: vehicle types over time ──
    types = ["two_wheeler", "four_wheeler", "heavy_vehicle", "emergency_vehicle"]
    labels_map = {"two_wheeler": "2W", "four_wheeler": "4W", "heavy_vehicle": "Heavy", "emergency_vehicle": "Emergency"}
    colors_map = {"two_wheeler": CHART_COLORS[0], "four_wheeler": CHART_COLORS[1], "heavy_vehicle": CHART_COLORS[2], "emergency_vehicle": CHART_COLORS[3]}

    area_df = df.set_index("timestamp")[types].resample("15min").sum().fillna(0).reset_index()
    stacked = go.Figure()
    for t in types:
        stacked.add_trace(go.Scatter(
            x=area_df["timestamp"], y=area_df[t],
            mode="lines", name=labels_map[t],
            stackgroup="one", groupnorm="percent",
            line=dict(width=0.5, color=colors_map[t]),
            fillcolor=colors_map[t],
            hovertemplate="%{x|%H:%M}<br>%{y:,.0f} " + labels_map[t] + " (%{percent:.1f}%)<extra></extra>",
        ))
    stacked = apply_theme(stacked)
    stacked.update_layout(
        title=None, yaxis=dict(title="Share (%)", ticksuffix="%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=9)),
        margin=dict(t=40, b=32, l=40, r=16),
    )

    # ── Pie chart ──
    pie_data = compute_pie_counts(df)
    pie_labels = list(pie_data.keys())
    pie_values = list(pie_data.values())
    pie_pull = [0.08 if v == min(pie_values) else 0 for v in pie_values]
    pie = go.Figure(data=[go.Pie(
        labels=pie_labels,
        values=pie_values,
        hole=0.45,
        pull=pie_pull,
        marker=dict(colors=CHART_COLORS, line=dict(color="#ffffff", width=2)),
        textfont=dict(family="DM Sans", size=11, color="#1a1a2e"),
        hovertemplate="%{label}<br>%{value:,.0f} (%{percent})<extra></extra>",
        rotation=90,
    )])
    pie = apply_theme(pie)
    pie.update_layout(
        title=None, showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(t=16, b=48, l=16, r=16),
    )

    # ── Location bar chart ──
    loc_counts = df.groupby("location")[types].sum().sum(axis=1).sort_values(ascending=True)
    loc_colors = [CHART_COLORS[0] if v == loc_counts.max() else "#e2d9cc" for v in loc_counts.values]
    loc_bar = go.Figure(go.Bar(
        x=loc_counts.values, y=loc_counts.index,
        orientation="h",
        marker=dict(color=loc_colors, line=dict(color="#ffffff", width=1)),
        hovertemplate="%{y}<br>%{x:,.0f} vehicles<extra></extra>",
    ))
    loc_bar = apply_theme(loc_bar)
    loc_bar.update_layout(
        title=None,
        xaxis=dict(title="Total Vehicles", gridcolor="#ede7df"),
        yaxis=dict(title=None),
        margin=dict(t=16, b=32, l=80, r=16),
        hovermode="y unified",
    )

    return stacked, pie, loc_bar, kpi


# ─── UPLOAD CALLBACK ───
@app.callback(
    Output("upload-result", "children"),
    Output("upload-file-list", "children"),
    Input("upload-component", "contents"),
    State("upload-component", "filename"),
)
def upload_csv(contents, filename):
    if contents and filename:
        result = validate_and_ingest(contents, filename)
        if result["status"] == "ok":
            msg = f"Ingested {result['rows_ingested']} rows from {filename}"
            if result["errors"]:
                msg += html.Br() + html.Span(f"{len(result['errors'])} rows skipped", style={"color": "#ff6b35"})
            out = html.Div(msg, className="msg msg-success")
        else:
            err = result["errors"][0] if result["errors"] else "Unknown error"
            out = html.Div(f"Failed: {err}", className="msg msg-error")
    else:
        out = ""
    files = [f for f in os.listdir(settings.upload_dir) if f.endswith(".csv")]
    fl = html.Ul(className="file-list", children=[html.Li(f) for f in sorted(files)]) if files else html.Span("No files uploaded yet")
    return out, fl


# ─── FORECAST CALLBACK ───
@app.callback(
    Output("forecast-location", "options"),
    Output("forecast-line", "figure"),
    Output("forecast-pie", "figure"),
    Output("forecast-hourly-bar", "figure"),
    Output("forecast-status", "children"),
    Output("forecast-metrics", "children"),
    Input("forecast-predict", "n_clicks"),
    State("forecast-location", "value"),
    State("forecast-date", "value"),
    State("forecast-hours", "value"),
    State("forecast-model-type", "value"),
)
def run_forecast(n_clicks, location, date, ahead_hours, model_type):
    df = combine_all_uploads()
    empty = empty_figure("No data yet")
    options = []
    if not df.empty and "location" in df.columns:
        options = [{"label": l, "value": l} for l in sorted(df["location"].dropna().unique())]
    if not n_clicks or df.empty:
        return options, empty, empty, empty, html.Div("Upload CSV data and click Predict", className="msg msg-info"), ""
    filtered = df.copy()
    if location:
        filtered = filtered[filtered["location"].astype(str).str.contains(location, case=False)]
    if date:
        try:
            filtered = filtered[filtered["timestamp"].dt.date == pd.to_datetime(date).date()]
        except Exception:
            pass
    if filtered.empty:
        return options, empty, empty, empty, html.Div("No data after filtering", className="msg msg-warning"), ""
    hourly = compute_hourly_series(filtered)
    if len(hourly) < 3:
        return options, empty, empty, empty, html.Div("Not enough hourly data (need >= 3 rows)", className="msg msg-warning"), ""

    X_simple = hourly[["hour_index"]].values
    y_simple = hourly["total"].values
    model, simple_metrics = train_model(X_simple, y_simple, model_type=model_type or "linear")

    tag = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        save_model(model, simple_metrics, name=tag)
    except Exception:
        logger.warning("Model save skipped")

    metrics_str = ""
    try:
        train_df, test_df = train_test_split_temporal(hourly, test_hours=max(1, len(hourly) // 5))
        train_f = create_time_features(train_df, "timestamp")
        train_f = create_rolling_features(train_f)
        test_f = create_time_features(test_df, "timestamp")
        test_f = create_rolling_features(test_f)
        X_train, y_train, train_feats = prepare_forecast_data(train_f, target_col="total")
        X_test, y_test, test_feats = prepare_forecast_data(test_f, target_col="total")
        common = [f for f in train_feats if f in test_feats]
        if len(X_train) and len(X_test) and len(common) == len(train_feats):
            ev_model, _ = train_model(X_train, y_train, model_type="auto")
            ev_preds = predict(ev_model, X_test)
            ev_res = evaluate(y_test, ev_preds)
            metrics_str = html.Div(
                f"MAE {ev_res['mae']:.1f}  ·  RMSE {ev_res['rmse']:.1f}  ·  MAPE {ev_res['mape']:.1f}%",
                className="metrics-badge",
            )
    except Exception as exc:
        logger.warning("Eval skipped: %s", exc)

    last_idx = hourly["hour_index"].iloc[-1]
    future_idx = np.array(range(last_idx + 1, last_idx + ahead_hours + 1)).reshape(-1, 1)
    predictions = predict(model, future_idx)

    line = go.Figure()
    line.add_trace(go.Scatter(
        x=hourly["timestamp"], y=hourly["total"],
        mode="lines+markers", name="Historical",
        line=dict(color=CHART_COLORS[0], width=1.5),
        marker=dict(size=4, color=CHART_COLORS[0]),
    ))
    future_time = pd.date_range(
        start=hourly["timestamp"].iloc[-1], periods=len(predictions) + 1, freq="h"
    )[1:]
    line.add_trace(go.Scatter(
        x=future_time, y=predictions,
        mode="lines", name="Predicted",
        line=dict(color=CHART_COLORS[1], width=1.5, dash="dot"),
    ))
    line = apply_theme(line)
    line.update_layout(
        title=None, yaxis=dict(title="Vehicles / Hour"),
        hovermode="x unified",
    )

    f_pie_vals = [
        filtered["two_wheeler"].sum(),
        filtered["four_wheeler"].sum(),
        filtered["heavy_vehicle"].sum(),
        filtered["emergency_vehicle"].sum(),
    ]
    f_pie_pull = [0.08 if v == min(f_pie_vals) else 0 for v in f_pie_vals]
    pie = go.Figure(data=[go.Pie(
        labels=["2W", "4W", "Heavy", "Emergency"],
        values=f_pie_vals,
        hole=0.45,
        pull=f_pie_pull,
        marker=dict(colors=CHART_COLORS, line=dict(color="#ffffff", width=2)),
        textfont=dict(family="DM Sans", size=11, color="#1a1a2e"),
        hovertemplate="%{label}<br>%{value:,.0f} (%{percent})<extra></extra>",
    )])
    pie = apply_theme(pie)
    pie.update_layout(title=None, showlegend=True, legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"), margin=dict(t=16, b=48, l=16, r=16))

    # ── Hourly profile bar chart ──
    types_bar = ["two_wheeler", "four_wheeler", "heavy_vehicle", "emergency_vehicle"]
    labels_bar = ["2W", "4W", "Heavy", "Emergency"]
    filtered["hour"] = filtered["timestamp"].dt.hour
    hourly_profile = filtered.groupby("hour")[types_bar].sum().reset_index()
    hourly_profile["total"] = hourly_profile[types_bar].sum(axis=1)
    hourly_bar = go.Figure()
    for i, t in enumerate(types_bar):
        hourly_bar.add_trace(go.Bar(
            x=hourly_profile["hour"], y=hourly_profile[t],
            name=labels_bar[i], marker_color=CHART_COLORS[i],
            hovertemplate="%{x}:00<br>%{y:,.0f} " + labels_bar[i] + "<extra></extra>",
        ))
    hourly_bar = apply_theme(hourly_bar)
    hourly_bar.update_layout(
        title=None, barmode="group",
        xaxis=dict(title="Hour of Day", dtick=1),
        yaxis=dict(title="Vehicles"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=9)),
        margin=dict(t=40, b=32, l=40, r=16),
    )

    status = html.Div(f"Forecast: {ahead_hours}h ahead  ·  Model: {simple_metrics.get('model_type', model_type)}", className="msg msg-success")
    return options, line, pie, hourly_bar, status, metrics_str


# ─── ANALYSES CALLBACK ───
@app.callback(
    Output("analyses-radar", "figure"),
    Output("analyses-gauge", "figure"),
    Output("analyses-hourly-bar", "figure"),
    Input("analyses-filter", "n_clicks"),
    State("analyses-location", "value"),
    State("analyses-from", "value"),
    State("analyses-to", "value"),
)
def analyses_run(n, location, from_ts, to_ts):
    df = combine_all_uploads()
    empty = empty_figure("No data")
    if df.empty:
        return empty, empty, empty
    filtered = df.copy()
    if location:
        filtered = filtered[filtered["location"].astype(str).str.contains(location, case=False)]
    if from_ts:
        try:
            filtered = filtered[filtered["timestamp"] >= pd.to_datetime(from_ts)]
        except Exception:
            pass
    if to_ts:
        try:
            filtered = filtered[filtered["timestamp"] <= pd.to_datetime(to_ts)]
        except Exception:
            pass
    if filtered.empty:
        return empty_figure("No matching records"), empty_figure("No matching records"), empty_figure("No matching records")

    means = compute_mean_vehicle_types(filtered)
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=list(means.values()), theta=list(means.keys()),
        fill="toself", name="Average",
        line=dict(color=CHART_COLORS[0], width=1.5),
        fillcolor="rgba(212,93,74,0.12)",
    ))
    radar = apply_theme(radar)
    radar.update_layout(
        title=None,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, color="#9a9ab0", gridcolor="#ede7df"),
            angularaxis=dict(color="#5a5a7a", gridcolor="#ede7df"),
        ),
        showlegend=False,
        margin=dict(t=16, b=16, l=48, r=48),
    )

    avg, maxv = compute_hourly_avg(filtered)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg,
        number=dict(font=dict(family="DM Sans", size=36, color="#1a1a2e", weight=300), suffix=" veh/h"),
        delta=dict(reference=maxv * 0.5, font=dict(family="DM Sans", size=12, color="#9a9ab0")),
        gauge=dict(
            axis=dict(range=[0, maxv], tickcolor="#9a9ab0", tickfont=dict(family="DM Sans", size=10, color="#9a9ab0")),
            bar=dict(color=CHART_COLORS[0], thickness=0.4),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, maxv * 0.6], color="rgba(58,125,122,0.06)"),
                dict(range=[maxv * 0.6, maxv * 0.85], color="rgba(201,168,76,0.08)"),
                dict(range=[maxv * 0.85, maxv], color="rgba(212,93,74,0.08)"),
            ],
            threshold=dict(
                line=dict(color=CHART_COLORS[0], width=2),
                thickness=0.6,
                value=maxv * 0.85,
            ),
        ),
    ))
    gauge = apply_theme(gauge)
    gauge.update_layout(
        title=None,
        margin=dict(t=32, b=16, l=48, r=48),
    )

    # ── Hourly distribution bar chart ──
    types_bar = ["two_wheeler", "four_wheeler", "heavy_vehicle", "emergency_vehicle"]
    labels_bar = ["2W", "4W", "Heavy", "Emergency"]
    filtered["hour"] = filtered["timestamp"].dt.hour
    hourly_profile = filtered.groupby("hour")[types_bar].sum().reset_index()
    hourly_bar = go.Figure()
    for i, t in enumerate(types_bar):
        hourly_bar.add_trace(go.Bar(
            x=hourly_profile["hour"], y=hourly_profile[t],
            name=labels_bar[i], marker_color=CHART_COLORS[i],
            hovertemplate="%{x}:00<br>%{y:,.0f} " + labels_bar[i] + "<extra></extra>",
        ))
    hourly_bar = apply_theme(hourly_bar)
    hourly_bar.update_layout(
        title=None, barmode="group",
        xaxis=dict(title="Hour of Day", dtick=1),
        yaxis=dict(title="Vehicles"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=9)),
        margin=dict(t=40, b=32, l=40, r=16),
    )

    return radar, gauge, hourly_bar


# ─── RUN ───
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    logger.info("Starting on http://127.0.0.1:%d", port)
    app.run(debug=settings.debug, host="0.0.0.0", port=port)
