import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import norm

# ── Black-Scholes core ──────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return dict(delta=1.0 if S > K else 0.0,
                    gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)

    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == "call"
                                          else norm.cdf(-d2))) / 365
    vega  = S * pdf_d1 * np.sqrt(T) / 100
    rho   = (K * T * np.exp(-r * T) * norm.cdf(d2)  / 100 if option_type == "call"
             else -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


# ── P&L heatmap ─────────────────────────────────────────────────────────────

def pnl_matrix(S, K, T, r, sigma, option_type):
    spot_pcts = np.linspace(-30, 30, 25)
    vol_shifts = np.linspace(-15, 15, 25)
    base = bs_price(S, K, T, r, sigma, option_type)
    matrix = np.zeros((len(vol_shifts), len(spot_pcts)))
    for i, dv in enumerate(vol_shifts):
        for j, ds in enumerate(spot_pcts):
            matrix[i, j] = bs_price(S * (1 + ds / 100), K, T, r,
                                     max(sigma + dv / 100, 0.001), option_type) - base
    return spot_pcts, vol_shifts, matrix


# ── App ─────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
app.title = "Options Greeks Dashboard"

DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"

SLIDER_STYLE = {"marginBottom": "24px"}
LABEL_STYLE  = {"color": MUTED, "fontSize": "11px",
                "letterSpacing": "0.08em", "textTransform": "uppercase",
                "marginBottom": "6px"}

def metric_card(label, value, color=TEXT):
    return html.Div([
        html.Div(label, style={**LABEL_STYLE}),
        html.Div(value, style={"color": color, "fontSize": "22px",
                               "fontWeight": "600", "fontFamily": "monospace"}),
    ], style={"background": PANEL, "border": f"1px solid {BORDER}",
              "borderRadius": "8px", "padding": "16px 20px", "flex": "1"})


app.layout = html.Div(style={"background": DARK, "minHeight": "100vh",
                              "fontFamily": "'Inter', sans-serif", "color": TEXT,
                              "padding": "32px"}, children=[

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div([
        html.H1("Options Greeks Dashboard",
                style={"margin": 0, "fontSize": "24px", "fontWeight": "700",
                       "color": TEXT}),
        html.P("Black-Scholes pricing · real-time Greeks · P&L simulation",
               style={"margin": "4px 0 0", "color": MUTED, "fontSize": "13px"}),
    ], style={"marginBottom": "32px"}),

    html.Div(style={"display": "flex", "gap": "24px", "alignItems": "flex-start"},
             children=[

        # ── Left panel: controls ─────────────────────────────────────────────
        html.Div(style={"width": "260px", "flexShrink": "0",
                        "background": PANEL, "border": f"1px solid {BORDER}",
                        "borderRadius": "10px", "padding": "24px 32px"}, children=[

            html.Div("Parameters", style={**LABEL_STYLE, "marginBottom": "20px"}),

            html.Div("Option Type", style=LABEL_STYLE),
            dcc.RadioItems(
                options=[{"label": " Call", "value": "call"},
                         {"label": " Put",  "value": "put"}],
                value="call", id="option-type",
                inline=True,
                style={"color": TEXT, "marginBottom": "24px", "fontSize": "14px"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "16px"},
            ),

            html.Div("Spot Price (S)", style=LABEL_STYLE),
            dcc.Slider(50, 150, step=1, value=100, id="spot",
                       marks={50: "50", 100: "100", 150: "150"},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div(style=SLIDER_STYLE),

            html.Div("Strike Price (K)", style=LABEL_STYLE),
            dcc.Slider(50, 150, step=1, value=100, id="strike",
                       marks={50: "50", 100: "100", 150: "150"},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div(style=SLIDER_STYLE),

            html.Div("Time to Expiry — years (T)", style=LABEL_STYLE),
            dcc.Slider(0.05, 2, step=0.05, value=1.0, id="time",
                       marks={0.05: "1w", 0.25: "3m", 0.5: "6m",
                              1: "1y", 2: "2y"},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div(style=SLIDER_STYLE),

            html.Div("Implied Volatility (σ)", style=LABEL_STYLE),
            dcc.Slider(0.05, 0.80, step=0.01, value=0.20, id="vol",
                       marks={0.05: "5%", 0.20: "20%",
                              0.50: "50%", 0.80: "80%"},
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div(style=SLIDER_STYLE),

            html.Div("Risk-Free Rate (r)", style=LABEL_STYLE),
            dcc.Slider(0.00, 0.10, step=0.005, value=0.05, id="rate",
                       marks={0: "0%", 0.05: "5%", 0.10: "10%"},
                       tooltip={"placement": "bottom", "always_visible": False}),
        ]),

        # ── Right panel: metrics + charts ───────────────────────────────────
        html.Div(style={"flex": "1", "minWidth": "0"}, children=[

            # Live metrics row
            html.Div(id="metrics-row",
                     style={"display": "flex", "gap": "12px", "marginBottom": "20px"}),

            # Greeks chart
            dcc.Graph(id="greeks-chart", config={"displayModeBar": False},
                      style={"marginBottom": "20px"}),

            # P&L heatmap
            dcc.Graph(id="pnl-chart", config={"displayModeBar": False}),
        ]),
    ]),
])


# ── Callback ────────────────────────────────────────────────────────────────

@app.callback(
    Output("metrics-row",   "children"),
    Output("greeks-chart",  "figure"),
    Output("pnl-chart",     "figure"),
    Input("spot",        "value"),
    Input("strike",      "value"),
    Input("time",        "value"),
    Input("rate",        "value"),   # r first
    Input("vol",         "value"),   # sigma second
    Input("option-type", "value"),
)
def update(S, K, T, r, sigma, option_type):
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)

    price = bs_price(S, K, T, r, sigma, option_type)
    g     = bs_greeks(S, K, T, r, sigma, option_type)

    moneyness = "ATM" if abs(S - K) < 1 else ("ITM" if (
        (option_type == "call" and S > K) or
        (option_type == "put"  and S < K)) else "OTM")

    # ── Metric cards ────────────────────────────────────────────────────────
    metrics = [
        metric_card("Option Price", f"${price:.4f}", ACCENT),
        metric_card("Moneyness",    moneyness,
                    GREEN if moneyness == "ITM" else (MUTED if moneyness == "ATM" else RED)),
        metric_card("Delta",  f"{g['delta']:+.4f}"),
        metric_card("Gamma",  f"{g['gamma']:.4f}"),
        metric_card("Theta",  f"{g['theta']:+.4f}/day", RED),
        metric_card("Vega",   f"{g['vega']:.4f}"),
    ]

    # ── Greeks figure ────────────────────────────────────────────────────────
    spots = np.linspace(max(S * 0.4, 1), S * 1.6, 300)

    greek_configs = [
        ("Delta",  "delta",  ACCENT,   "Probability proxy — how much the option moves per $1 in spot"),
        ("Gamma",  "gamma",  "#d2a8ff","Rate of delta change — highest ATM, drives hedging cost"),
        ("Theta",  "theta",  RED,      "Daily time decay — most negative ATM"),
        ("Vega",   "vega",   GREEN,    "Sensitivity to 1% vol move — peaks ATM"),
        ("Rho",    "rho",    "#ffa657","Sensitivity to 1% rate move"),
    ]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[c[0] for c in greek_configs] + [""],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    fig.update_annotations(font=dict(color=MUTED, size=11))

    positions = [(1,1),(1,2),(1,3),(2,1),(2,2)]
    for idx, (name, key, color, _) in enumerate(greek_configs):
        row, col = positions[idx]
        vals = [bs_greeks(s, K, T, r, sigma, option_type)[key] for s in spots]
        fig.add_trace(go.Scatter(
            x=spots, y=vals, mode="lines", name=name,
            line=dict(color=color, width=2),
            showlegend=False,
        ), row=row, col=col)
        # current spot line
        fig.add_vline(x=S, line_dash="dash", line_color="#8b949e",
                      line_width=1, row=row, col=col)
        # strike line
        fig.add_vline(x=K, line_dash="dot", line_color="#ffa657",
                      line_width=1, row=row, col=col)

    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=DARK,
        font=dict(color=TEXT, size=11),
        margin=dict(l=8, r=8, t=48, b=8),
        height=460,
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, title_font_size=10)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    # hide empty 6th subplot
    fig.update_xaxes(visible=False, row=2, col=3)
    fig.update_yaxes(visible=False, row=2, col=3)

    # legend annotation
    fig.add_annotation(
        text="<b>---</b> Spot  <b>···</b> Strike",
        xref="paper", yref="paper", x=0.98, y=-0.02,
        showarrow=False, font=dict(color=MUTED, size=10), align="right",
    )

    # ── P&L heatmap ─────────────────────────────────────────────────────────
    spot_pcts, vol_shifts, matrix = pnl_matrix(S, K, T, r, sigma, option_type)

    heatmap_fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"{v:+.0f}%" for v in spot_pcts],
        y=[f"{v:+.0f}%" for v in vol_shifts],
        colorscale=[[0, RED], [0.5, DARK], [1, GREEN]],
        zmid=0,
       
        textfont=dict(size=8),
        hovertemplate="Spot %{x} / Vol %{y}<br>P&L: %{z:.4f}<extra></extra>",
        colorbar=dict(
            title="P&L ($)", tickfont=dict(color=TEXT),
            title_font=dict(color=TEXT), bgcolor=PANEL,
        ),
    ))
    heatmap_fig.update_layout(
        title=dict(text="P&L Scenario Matrix — spot move (x) vs vol shift (y)",
                   font=dict(color=TEXT, size=13)),
        paper_bgcolor=PANEL, plot_bgcolor=DARK,
        font=dict(color=TEXT, size=10),
        margin=dict(l=8, r=8, t=48, b=8),
        height=340,
        xaxis=dict(title="Spot move", gridcolor=BORDER),
        yaxis=dict(title="Vol shift", gridcolor=BORDER),
    )

    return metrics, fig, heatmap_fig


if __name__ == "__main__":
    app.run(debug=True)