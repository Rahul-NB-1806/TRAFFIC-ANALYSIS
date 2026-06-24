import pandas as pd
from dash import dcc, html


def status_bar(active="home"):
    pages = [
        ("/", "Home", "home"),
        ("/forecast", "Forecast", "forecast"),
        ("/upload", "Upload", "upload"),
        ("/analytics", "Analytics", "analytics"),
    ]
    links = []
    for href, label, key in pages:
        cls = "active" if key == active else ""
        links.append(html.A(label, href=href, className=cls))
    return html.Header(
        className="status-bar",
        children=[
            html.A(
                className="status-bar-brand",
                href="/",
                children=[
                    html.Span(className="brand-dot"),
                    "Pulse",
                    html.Span(" traffic monitor", className="brand-label"),
                ],
            ),
            html.Nav(className="status-nav", children=links),
        ],
    )


def kpi_strip(df=None):
    if df is None or df.empty:
        return html.Div(className="kpi-strip", children=[
            _kpi("Total Vehicles", "—", "Across — locations"),
            _kpi("2-Wheelers", "—", "—%", "green"),
            _kpi("4-Wheelers", "—", "—%", "blue"),
            _kpi("Heavy + Emergency", "—", "—%", "red"),
            _kpi("Time Range", "—", "— records", "amber"),
        ])
    total_v = int(df["total"].sum())
    locations = df["location"].nunique()
    two_w = int(df["two_wheeler"].sum())
    four_w = int(df["four_wheeler"].sum())
    heavy = int(df["heavy_vehicle"].sum())
    emergency = int(df["emergency_vehicle"].sum())
    he = heavy + emergency
    records = len(df)
    tr = f"{df['timestamp'].min().strftime('%d %b %H:%M')} \u2014 {df['timestamp'].max().strftime('%d %b %H:%M')}"
    return html.Div(className="kpi-strip", children=[
        _kpi("Total Vehicles", f"{total_v:,}", f"Across {locations} locations", "blue"),
        _kpi("2-Wheelers", f"{two_w:,}", f"{two_w/max(total_v,1)*100:.0f}% of traffic", "green"),
        _kpi("4-Wheelers", f"{four_w:,}", f"{four_w/max(total_v,1)*100:.0f}% of traffic", "blue"),
        _kpi("Heavy + Emergency", f"{he:,}", f"{he/max(total_v,1)*100:.0f}% of traffic", "red"),
        _kpi("Time Range", tr, f"{records} records", "amber"),
    ])


def _kpi(label, value, sub, color="blue"):
    cls = f"kpi-value kpi-{color}"
    return html.Div(className="kpi-item", children=[
        html.Span(label, className="kpi-label"),
        html.Span(value, className=cls),
        html.Span(sub, className="kpi-sub"),
    ])


def card(title, graph_id, height="440px"):
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


def page_wrapper(content, active="home"):
    return html.Div([
        status_bar(active),
        html.Div(className="page-container", children=content),
    ])


def filter_bar(children):
    return html.Div(className="filter-bar", children=children)
