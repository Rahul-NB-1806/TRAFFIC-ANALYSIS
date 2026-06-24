import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Output, Input, State, html

from core.config import settings
from services.analytics import (
    compute_vehicle_totals,
    compute_location_breakdown,
    compute_hourly_profile,
    compute_peak_hours,
    detect_anomalies,
)
from services.forecasting import prepare_trend_data, train_trend_model, generate_forecast
from services.ingestion import load_dataframe, ingest_csv
from dashboard.theme import CHART_COLORS, VEHICLE_LABELS, apply_theme, empty_figure
from dashboard.components import kpi_strip

logger = logging.getLogger(__name__)

_VEHICLE_TYPES = ["two_wheeler", "four_wheeler", "heavy_vehicle", "emergency_vehicle"]
_VEHICLE_LABELS_SHORT = ["2W", "4W", "Heavy", "Emergency"]


def _figure_peak_hours(df):
    peaks = compute_peak_hours(df, threshold_pct=0.8)
    if not peaks:
        return html.Div()
    items = []
    for p in peaks:
        label = f"Peak: {p['hour']:02d}:00–{p['hour']:02d}:59"
        items.append(html.Span(label, className="peak-chip"))
    return html.Div(className="peak-hours-bar", children=items)


def _figure_location_table(df):
    breakdown = compute_location_breakdown(df)
    if breakdown.empty:
        return html.Div("No location data", style={"color": "#9a9ab0"})
    rows = []
    for _, row in breakdown.iterrows():
        rows.append(html.Tr([
            html.Td(row["location"]),
            html.Td(f"{int(row['total']):,}", style={"textAlign": "right", "fontWeight": 600}),
        ]))
    return html.Table(
        className="location-table",
        children=[
            html.Thead(html.Tr([html.Th("Location"), html.Th("Total Vehicles")])),
            html.Tbody(rows),
        ],
    )


def _build_stacked_area(df):
    types = _VEHICLE_TYPES
    labels = _VEHICLE_LABELS_SHORT
    area_df = df.set_index("timestamp")[types].resample("15min").sum().fillna(0).reset_index()
    fig = go.Figure()
    for i, t in enumerate(types):
        fig.add_trace(go.Scatter(
            x=area_df["timestamp"], y=area_df[t],
            mode="lines", name=labels[i],
            stackgroup="one", groupnorm="percent",
            line=dict(width=0.5, color=CHART_COLORS[i]),
            fillcolor=CHART_COLORS[i],
            hovertemplate="%{x|%H:%M}<br>%{y:,.0f} " + labels[i] + " (%{percent:.1f}%)<extra></extra>",
        ))
    fig = apply_theme(fig)
    fig.update_layout(
        yaxis=dict(title="Share (%)", ticksuffix="%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=9)),
        margin=dict(t=40, b=32, l=40, r=16),
    )
    return fig


def _build_pie_chart(df):
    totals = compute_vehicle_totals(df)
    labels = _VEHICLE_LABELS_SHORT
    values = [totals[c] for c in _VEHICLE_TYPES]
    pull = [0.08 if v == min(values) else 0 for v in values]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.45, pull=pull,
        marker=dict(colors=CHART_COLORS, line=dict(color="#ffffff", width=2)),
        textfont=dict(family="Source Sans 3", size=11, color="#1a1a2e"),
        hovertemplate="%{label}<br>%{value:,.0f} (%{percent})<extra></extra>",
        rotation=90,
    )])
    fig = apply_theme(fig)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(t=16, b=48, l=16, r=16),
    )
    return fig


def _build_hourly_bar(df):
    profile = compute_hourly_profile(df)
    if profile.empty:
        return empty_figure("No hourly data")
    fig = go.Figure()
    for i, t in enumerate(_VEHICLE_TYPES):
        fig.add_trace(go.Bar(
            x=profile["hour"], y=profile[t],
            name=_VEHICLE_LABELS_SHORT[i],
            marker_color=CHART_COLORS[i],
            hovertemplate="%{x}:00<br>%{y:,.0f} " + _VEHICLE_LABELS_SHORT[i] + "<extra></extra>",
        ))
    fig = apply_theme(fig)
    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Hour of Day", dtick=1),
        yaxis=dict(title="Vehicles"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=9)),
        margin=dict(t=40, b=32, l=40, r=16),
    )
    return fig


def _build_forecast_line(historical_df, predictions, ahead_hours):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_df["timestamp"], y=historical_df["total"],
        mode="lines+markers", name="Historical",
        line=dict(color=CHART_COLORS[0], width=1.5),
        marker=dict(size=4, color=CHART_COLORS[0]),
    ))
    future_time = pd.date_range(
        start=historical_df["timestamp"].iloc[-1],
        periods=len(predictions) + 1,
        freq="h",
    )[1:]
    fig.add_trace(go.Scatter(
        x=future_time, y=predictions,
        mode="lines", name="Predicted",
        line=dict(color=CHART_COLORS[1], width=1.5, dash="dot"),
    ))
    fig = apply_theme(fig)
    fig.update_layout(
        yaxis=dict(title="Vehicles / Hour"),
        hovermode="x unified",
    )
    return fig


def _build_anomaly_chart(df):
    annotated = detect_anomalies(df)
    if annotated.empty:
        return empty_figure("No anomaly data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annotated["timestamp"], y=annotated["total"],
        mode="lines+markers", name="Traffic",
        line=dict(color=CHART_COLORS[0], width=1.5),
        marker=dict(size=5, color=CHART_COLORS[0]),
        hovertemplate="%{x|%d %b %H:%M}<br>%{y:,.0f} vehicles<extra></extra>",
    ))
    anomalies = annotated[annotated["is_anomaly"]]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"], y=anomalies["total"],
            mode="markers", name="Anomaly",
            marker=dict(color=CHART_COLORS[1], size=8, symbol="x", line=dict(color="#1a1a2e", width=1)),
            hovertemplate="%{x|%d %b %H:%M}<br>%{y:,.0f} vehicles \u26a0<extra></extra>",
        ))
    fig = apply_theme(fig)
    fig.update_layout(
        yaxis=dict(title="Vehicles"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=9)),
        margin=dict(t=40, b=32, l=40, r=16),
    )
    return fig


def register_callbacks(app):

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def display_page(path):
        if path == "/forecast":
            from dashboard.layouts import forecast_layout
            return forecast_layout
        if path == "/upload":
            from dashboard.layouts import upload_layout
            return upload_layout
        if path == "/analytics":
            from dashboard.layouts import analytics_layout
            return analytics_layout
        from dashboard.layouts import home_layout
        return home_layout

    @app.callback(
        Output("home-kpi-strip", "children"),
        Output("home-peak-hours", "children"),
        Output("home-stacked", "figure"),
        Output("home-pie", "figure"),
        Output("home-hourly-bar", "figure"),
        Output("home-location-table", "children"),
        Input("url", "pathname"),
    )
    def update_home(path):
        df = load_dataframe()
        kpi = kpi_strip(df)
        if df.empty:
            return kpi, html.Div(), empty_figure("Upload CSV data to get started"), empty_figure(), empty_figure("Upload CSV data to get started"), html.Div("No data yet", style={"color": "#9a9ab0"})
        peaks = _figure_peak_hours(df)
        stacked = _build_stacked_area(df)
        pie = _build_pie_chart(df)
        hourly = _build_hourly_bar(df)
        table = _figure_location_table(df)
        return kpi, peaks, stacked, pie, hourly, table

    @app.callback(
        Output("forecast-location", "options"),
        Output("forecast-line", "figure"),
        Output("forecast-pie", "figure"),
        Output("forecast-hourly-bar", "figure"),
        Output("forecast-status", "children"),
        Output("forecast-metrics", "children"),
        Input("forecast-btn", "n_clicks"),
        State("forecast-location", "value"),
        State("forecast-hours", "value"),
        State("forecast-model", "value"),
    )
    def run_forecast(n_clicks, location, ahead_hours, model_type):
        df = load_dataframe()
        empty = empty_figure("No data yet")
        options = []
        if not df.empty and "location" in df.columns:
            options = [{"label": l, "value": l} for l in sorted(df["location"].dropna().unique())]
        if not n_clicks or df.empty:
            return options, empty, empty, empty, html.Div("Upload CSV data and click Predict", className="msg msg-info"), ""

        filtered = df.copy()
        if location:
            filtered = filtered[filtered["location"].astype(str).str.contains(location, case=False)]
        if filtered.empty:
            return options, empty, empty, empty, html.Div("No data after filtering", className="msg msg-warning"), ""

        ahead_hours = ahead_hours or 24
        X, y = prepare_trend_data(filtered)
        if len(X) < 2:
            return options, empty, empty, empty, html.Div("Not enough data (need >= 2 rows)", className="msg msg-warning"), ""

        model_type = model_type or "auto"
        try:
            model, metrics = train_trend_model(X, y, model_type=model_type)
        except Exception as e:
            logger.exception("Model training failed")
            return options, empty, empty, empty, html.Div(f"Model failed: {e}", className="msg msg-error"), ""

        predictions = generate_forecast(model, last_idx=len(X) - 1, steps=ahead_hours)
        forecast_fig = _build_forecast_line(filtered, predictions, ahead_hours)
        pie_fig = _build_pie_chart(filtered)
        hourly_fig = _build_hourly_bar(filtered)

        metrics_str = html.Div(
            f"MAE {metrics['mae']}  \u00b7  RMSE {metrics['rmse']}  \u00b7  MAPE {metrics['mape']}%",
            className="metrics-badge",
        )
        status = html.Div(
            f"Forecast: {ahead_hours}h ahead  \u00b7  Model: {model_type}",
            className="msg msg-success",
        )
        return options, forecast_fig, pie_fig, hourly_fig, status, metrics_str

    @app.callback(
        Output("upload-report", "children"),
        Output("upload-file-list", "children"),
        Input("upload-zone", "contents"),
        State("upload-zone", "filename"),
    )
    def handle_upload(contents, filename):
        if contents and filename:
            report = ingest_csv(contents, filename)
            if report.rows_ingested > 0:
                msg = html.Div([
                    html.Strong(f"Ingested {report.rows_ingested} rows"),
                    html.Span(f" from {filename}"),
                    html.Br(),
                    html.Span(f"Total vehicles: {report.total_vehicles:,}  \u00b7  "
                              f"Locations: {', '.join(report.locations)}",
                              style={"fontSize": "12px", "color": "#7a7a9a"}),
                ], className="msg msg-success")
                if report.errors:
                    msg = html.Div([
                        msg,
                        html.Br(),
                        html.Span(f"{len(report.errors)} rows skipped", style={"color": "#e63946", "fontSize": "12px"}),
                    ])
            else:
                err = report.errors[0] if report.errors else "Unknown error"
                msg = html.Div(f"Failed: {err}", className="msg msg-error")
        else:
            msg = ""

        upload_dir = settings.upload_dir
        files = []
        if os.path.isdir(upload_dir):
            files = sorted([f for f in os.listdir(upload_dir) if f.endswith(".csv")])
        fl = html.Ul(className="file-list", children=[html.Li(f) for f in files]) if files else html.Span("No files uploaded yet")
        return msg, fl

    @app.callback(
        Output("analytics-radar", "figure"),
        Output("analytics-gauge", "figure"),
        Output("analytics-hourly-bar", "figure"),
        Output("analytics-anomalies", "figure"),
        Input("analytics-apply", "n_clicks"),
        State("analytics-location", "value"),
        State("analytics-from", "value"),
        State("analytics-to", "value"),
    )
    def run_analytics(n, location, from_ts, to_ts):
        df = load_dataframe(location=location, from_ts=from_ts, to_ts=to_ts)
        if df.empty:
            dd = load_dataframe()
            if dd.empty:
                return empty_figure("No data"), empty_figure("No data"), empty_figure("No data"), empty_figure("No data")
            filtered = dd.copy()
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
            df = filtered

        if df.empty:
            return empty_figure("No matching records"), empty_figure("No matching records"), empty_figure("No matching records"), empty_figure("No matching records")

        means = df[_VEHICLE_TYPES].mean().to_dict()
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=list(means.values()),
            theta=[VEHICLE_LABELS.get(k, k) for k in _VEHICLE_TYPES],
            fill="toself", name="Average",
            line=dict(color=CHART_COLORS[0], width=1.5),
            fillcolor="rgba(0,82,204,0.12)",
        ))
        radar = apply_theme(radar)
        radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, color="#9a9ab0", gridcolor="#e8e5e0"),
                angularaxis=dict(color="#5a5a7a", gridcolor="#e8e5e0"),
            ),
            showlegend=False,
            margin=dict(t=16, b=16, l=48, r=48),
        )

        avg_vehicles = df["total"].mean()
        max_vehicles = df["total"].max()
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_vehicles,
            number=dict(font=dict(family="Source Sans 3", size=36, color="#1a1a2e"), suffix=" veh"),
            delta=dict(reference=max_vehicles * 0.5, font=dict(family="Source Sans 3", size=12, color="#9a9ab0")),
            gauge=dict(
                axis=dict(range=[0, max_vehicles], tickcolor="#9a9ab0", tickfont=dict(family="Source Sans 3", size=10, color="#9a9ab0")),
                bar=dict(color=CHART_COLORS[0], thickness=0.4),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, max_vehicles * 0.6], color="rgba(45,106,79,0.06)"),
                    dict(range=[max_vehicles * 0.6, max_vehicles * 0.85], color="rgba(233,196,106,0.08)"),
                    dict(range=[max_vehicles * 0.85, max_vehicles], color="rgba(230,57,70,0.08)"),
                ],
                threshold=dict(
                    line=dict(color=CHART_COLORS[0], width=2),
                    thickness=0.6,
                    value=max_vehicles * 0.85,
                ),
            ),
        ))
        gauge = apply_theme(gauge)
        gauge.update_layout(margin=dict(t=32, b=16, l=48, r=48))

        hourly_fig = _build_hourly_bar(df)
        anomaly_fig = _build_anomaly_chart(df)

        return radar, gauge, hourly_fig, anomaly_fig
