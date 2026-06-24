import plotly.graph_objs as go

CHART_COLORS = ["#0052cc", "#e63946", "#2d6a4f", "#e9c46a"]

VEHICLE_LABELS = {"two_wheeler": "2W", "four_wheeler": "4W", "heavy_vehicle": "Heavy", "emergency_vehicle": "Emergency"}

THEME = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3", size=11, color="#5a5a7a"),
        margin=dict(l=40, r=16, t=32, b=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#e0ddd8",
            font=dict(family="Source Sans 3", size=11, color="#1a1a2e"),
        ),
        xaxis=dict(
            gridcolor="#e8e5e0",
            linecolor="#e8e5e0",
            tickcolor="#9a9ab0",
            tickfont=dict(size=10, color="#9a9ab0"),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#e8e5e0",
            linecolor="#e8e5e0",
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
    return fig


def empty_figure(msg="No data yet"):
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#9a9ab0", family="Source Sans 3"),
    )
    return apply_theme(fig)
