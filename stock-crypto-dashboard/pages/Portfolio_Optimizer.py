"""
MarketVerse - Portfolio Optimizer
-----------------------------------
Modern Portfolio Theory: efficient frontier, max-Sharpe allocation,
min-volatility allocation, and risk/return analysis.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Optimizer - MarketVerse", layout="wide", page_icon="💼")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 20%, #2d1b4e 0%, #1a0f30 35%, #0d0618 100%);
        color: #e6e1f5;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f1133 0%, #0f0820 100%);
        border-right: 1px solid #6c3fc5;
    }
    h1, h2, h3, h4 {
        color: #d9b8ff !important;
        text-shadow: 0 0 12px rgba(167, 102, 255, 0.45);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(108, 63, 197, 0.25), rgba(40, 16, 70, 0.4));
        border: 1px solid rgba(167, 102, 255, 0.35);
        border-radius: 14px;
        padding: 12px 16px;
        box-shadow: 0 0 18px rgba(124, 58, 237, 0.25);
    }
    div[data-testid="stMetricLabel"] { color: #c9bfe8 !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div[data-testid="stMetricDelta"] { color: #5eead4 !important; }
    p, span, label, .stMarkdown, .stCaption { color: #e6e1f5 !important; }
    .stButton button, .stDownloadButton button {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white;
        border: none;
        border-radius: 8px;
        box-shadow: 0 0 14px rgba(168, 85, 247, 0.5);
    }
    hr { border-color: rgba(167, 102, 255, 0.3) !important; }
    details {
        background: rgba(108, 63, 197, 0.08);
        border: 1px solid rgba(167, 102, 255, 0.2);
        border-radius: 10px;
    }

    /* Selectbox / dropdown styling */
    div[data-testid="stSelectbox"] > div > div {
        background: linear-gradient(145deg, rgba(108, 63, 197, 0.35), rgba(40, 16, 70, 0.5)) !important;
        border: 1px solid rgba(167, 102, 255, 0.4) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    div[data-testid="stSelectbox"] span {
        color: #ffffff !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background: #1f1133 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        background: #1f1133 !important;
        color: #ffffff !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background: rgba(168, 85, 247, 0.3) !important;
    }

    /* Multiselect styling */
    div[data-testid="stMultiSelect"] > div > div {
        background: linear-gradient(145deg, rgba(108, 63, 197, 0.35), rgba(40, 16, 70, 0.5)) !important;
        border: 1px solid rgba(167, 102, 255, 0.4) !important;
        border-radius: 8px !important;
    }
    div[data-testid="stMultiSelect"] span {
        color: #ffffff !important;
    }
    span[data-baseweb="tag"] {
        background: #7c3aed !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_BG = "#160a28"
PLOTLY_GRID = "rgba(167, 102, 255, 0.15)"
PLOTLY_FONT = "#e6e1f5"
ACCENT_VIOLET = "#a855f7"
ACCENT_PINK = "#f472d0"
ACCENT_TEAL = "#5eead4"
ACCENT_GOLD = "#facc15"


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor=PLOTLY_BG,
        font_color="white",
        legend_font_color="white"
    )
    return fig


st.title("Portfolio Optimizer")
st.caption(
    "Modern Portfolio Theory: find allocations that maximize Sharpe Ratio or minimize volatility, "
    "and explore the efficient frontier. Educational purposes only — not financial advice."
)

st.sidebar.header("Portfolio Settings")

PRESET_TICKERS = {
    "Indian Stocks": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    ],
    "US Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
        "JPM", "V", "MA", "WMT", "DIS", "BA", "KO", "PEP",
    ],
    "Crypto": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
        "DOGE-USD", "MATIC-USD",
    ],
}
ALL_TICKERS = sum(PRESET_TICKERS.values(), [])

selected_tickers = st.sidebar.multiselect(
    "Select assets for portfolio (2+ required)",
    options=ALL_TICKERS,
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "BTC-USD"],
)

custom_ticker = st.sidebar.text_input("Add a custom ticker (optional)", value="")
if custom_ticker:
    custom_ticker = custom_ticker.strip().upper()
    if custom_ticker not in selected_tickers:
        selected_tickers = selected_tickers + [custom_ticker]

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start date", pd.to_datetime("2022-01-01"))
end_date = col2.date_input("End date", pd.to_datetime("today"))

risk_free_rate = st.sidebar.slider(
    "Risk-free rate (annual %)", 0.0, 10.0, 6.0, 0.5
) / 100

allow_short = st.sidebar.checkbox(
    "Allow short positions (negative weights)", value=False,
    help="If unchecked, weights are constrained between 0 and 1 (long-only)."
)

num_simulations = st.sidebar.slider(
    "Random portfolios to simulate (for frontier cloud)", 500, 5000, 1500, 500
)

investment_amount = st.sidebar.number_input(
    "Investment amount (for allocation breakdown)", min_value=1000, value=100000, step=1000
)

@st.cache_data(ttl=3600)
def load_close_prices(tickers, start, end) -> pd.DataFrame:
    """Load adjusted close prices for multiple tickers into one DataFrame."""
    closes = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes[t] = df["Close"]
    if not closes:
        return pd.DataFrame()
    price_df = pd.DataFrame(closes)
    price_df = price_df.dropna(how="all")
    return price_df

def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.sum(mean_returns * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return port_return, port_vol


def negative_sharpe(weights, mean_returns, cov_matrix, rf):
    port_return, port_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    if port_vol == 0:
        return 0.0
    return -(port_return - rf) / port_vol


def portfolio_vol(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def portfolio_return_neg(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[0]


def optimize_for(objective, mean_returns, cov_matrix, rf, bounds, n_assets):
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    init_guess = np.array(n_assets * [1.0 / n_assets])

    if objective == "max_sharpe":
        result = minimize(negative_sharpe, init_guess, args=(mean_returns, cov_matrix, rf),
                           method="SLSQP", bounds=bounds, constraints=constraints)
    elif objective == "min_vol":
        result = minimize(portfolio_vol, init_guess, args=(mean_returns, cov_matrix),
                           method="SLSQP", bounds=bounds, constraints=constraints)
    else:
        raise ValueError("Unknown objective")

    return result.x


def efficient_frontier(mean_returns, cov_matrix, bounds, n_assets, n_points=30):
    """Compute the efficient frontier by minimizing volatility for a range of target returns."""
    min_ret = mean_returns.min() * 252
    max_ret = mean_returns.max() * 252
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_vols = []
    frontier_rets = []

    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: portfolio_performance(w, mean_returns, cov_matrix)[0] - target},
        )
        init_guess = np.array(n_assets * [1.0 / n_assets])
        result = minimize(portfolio_vol, init_guess, args=(mean_returns, cov_matrix),
                           method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            frontier_vols.append(result.fun)
            frontier_rets.append(target)

    return np.array(frontier_rets), np.array(frontier_vols)


def random_portfolios(mean_returns, cov_matrix, n_assets, n_sims, allow_short, rf):
    results = np.zeros((3, n_sims))
    weights_record = []
    for i in range(n_sims):
        if allow_short:
            w = np.random.uniform(-1, 1, n_assets)
        else:
            w = np.random.random(n_assets)
        w /= np.sum(w)
        weights_record.append(w)
        port_return, port_vol = portfolio_performance(w, mean_returns, cov_matrix)
        sharpe = (port_return - rf) / port_vol if port_vol != 0 else 0
        results[0, i] = port_vol
        results[1, i] = port_return
        results[2, i] = sharpe
    return results, weights_record

if len(selected_tickers) < 2:
    st.warning("Please select at least 2 assets in the sidebar to build a portfolio.")
else:
    prices = load_close_prices(selected_tickers, start_date, end_date)

    if prices.empty or prices.shape[1] < 2:
        st.error("Could not load enough valid price data for the selected tickers. Check ticker symbols.")
    else:
        prices = prices.dropna()
        valid_tickers = list(prices.columns)

        if len(valid_tickers) < len(selected_tickers):
            missing = set(selected_tickers) - set(valid_tickers)
            st.warning(f"Excluded due to missing/insufficient data: {', '.join(missing)}")

        if len(valid_tickers) < 2 or len(prices) < 30:
            st.error("Not enough overlapping historical data across selected assets. Try a wider date range or different assets.")
        else:
            returns = prices.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            n_assets = len(valid_tickers)

            bounds = tuple((-1, 1) if allow_short else (0, 1) for _ in range(n_assets))

            max_sharpe_weights = optimize_for("max_sharpe", mean_returns, cov_matrix, risk_free_rate, bounds, n_assets)
            min_vol_weights = optimize_for("min_vol", mean_returns, cov_matrix, risk_free_rate, bounds, n_assets)
            equal_weights = np.array(n_assets * [1.0 / n_assets])

            max_sharpe_perf = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
            min_vol_perf = portfolio_performance(min_vol_weights, mean_returns, cov_matrix)
            equal_perf = portfolio_performance(equal_weights, mean_returns, cov_matrix)

            def sharpe_of(perf):
                return (perf[0] - risk_free_rate) / perf[1] if perf[1] != 0 else np.nan

            st.subheader("Optimal Portfolio: Maximum Sharpe Ratio")

            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Annual Return", f"{max_sharpe_perf[0]*100:.2f}%")
            m2.metric("Annual Volatility (Risk)", f"{max_sharpe_perf[1]*100:.2f}%")
            m3.metric("Sharpe Ratio", f"{sharpe_of(max_sharpe_perf):.2f}")

            st.subheader("Recommended Allocation")

            alloc_df = pd.DataFrame({
                "Asset": valid_tickers,
                "Weight (%)": max_sharpe_weights * 100,
                "Allocated Amount": max_sharpe_weights * investment_amount,
            }).sort_values("Weight (%)", ascending=False).reset_index(drop=True)

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.dataframe(
                    alloc_df.style.format({"Weight (%)": "{:.2f}", "Allocated Amount": "{:,.2f}"}),
                    use_container_width=True,
                )
            with col_b:
                pie_fig = go.Figure(data=[go.Pie(
                    labels=alloc_df["Asset"],
                    values=alloc_df["Weight (%)"],
                    hole=0.45,
                    marker=dict(colors=[ACCENT_VIOLET, ACCENT_TEAL, ACCENT_PINK, ACCENT_GOLD,
                                          "#60a5fa", "#fb923c", "#34d399", "#f87171"] * 3),
                )])
                pie_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
                pie_fig = style_fig(pie_fig)
                st.plotly_chart(pie_fig, use_container_width=True)

            st.subheader("Efficient Frontier")

            with st.spinner("Computing efficient frontier..."):
                frontier_rets, frontier_vols = efficient_frontier(mean_returns, cov_matrix, bounds, n_assets)
                random_results, _ = random_portfolios(mean_returns, cov_matrix, n_assets, num_simulations, allow_short, risk_free_rate)

            frontier_fig = go.Figure()

            frontier_fig.add_trace(go.Scatter(
                x=random_results[0] * 100, y=random_results[1] * 100,
                mode="markers", name="Random Portfolios",
                marker=dict(
                    size=4, color=random_results[2], colorscale="Plasma",
                    colorbar=dict(title="Sharpe"), opacity=0.6,
                ),
            ))

            if len(frontier_vols) > 0:
                frontier_fig.add_trace(go.Scatter(
                    x=frontier_vols * 100, y=frontier_rets * 100,
                    mode="lines", name="Efficient Frontier",
                    line=dict(color="#ffffff", width=2, dash="dash"),
                ))

            frontier_fig.add_trace(go.Scatter(
                x=[max_sharpe_perf[1] * 100], y=[max_sharpe_perf[0] * 100],
                mode="markers", name="Max Sharpe Portfolio",
                marker=dict(color=ACCENT_TEAL, size=14, symbol="star"),
            ))

            frontier_fig.add_trace(go.Scatter(
                x=[min_vol_perf[1] * 100], y=[min_vol_perf[0] * 100],
                mode="markers", name="Min Volatility Portfolio",
                marker=dict(color=ACCENT_GOLD, size=14, symbol="diamond"),
            ))

            frontier_fig.add_trace(go.Scatter(
                x=[equal_perf[1] * 100], y=[equal_perf[0] * 100],
                mode="markers", name="Equal Weight Portfolio",
                marker=dict(color=ACCENT_PINK, size=12, symbol="circle"),
            ))

            for t in valid_tickers:
                asset_ret = mean_returns[t] * 252
                asset_vol = np.sqrt(cov_matrix.loc[t, t] * 252)
                frontier_fig.add_trace(go.Scatter(
                    x=[asset_vol * 100], y=[asset_ret * 100],
                    mode="markers+text", name=t, text=[t], textposition="top center",
                    marker=dict(color=ACCENT_VIOLET, size=9, symbol="x"),
                    showlegend=False,
                ))

            frontier_fig.update_layout(
                height=550,
                xaxis_title="Annualized Volatility (%)",
                yaxis_title="Annualized Return (%)",
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            frontier_fig = style_fig(frontier_fig)
            st.plotly_chart(frontier_fig, use_container_width=True)

            st.subheader("Portfolio Strategy Comparison")

            comparison_df = pd.DataFrame({
                "Max Sharpe": [max_sharpe_perf[0]*100, max_sharpe_perf[1]*100, sharpe_of(max_sharpe_perf)],
                "Min Volatility": [min_vol_perf[0]*100, min_vol_perf[1]*100, sharpe_of(min_vol_perf)],
                "Equal Weight": [equal_perf[0]*100, equal_perf[1]*100, sharpe_of(equal_perf)],
            }, index=["Expected Return (%)", "Volatility (%)", "Sharpe Ratio"])

            st.dataframe(comparison_df.style.format("{:.2f}"), use_container_width=True)

            with st.expander("Asset Correlation Matrix (daily returns)"):
                corr = returns.corr()
                corr_fig = go.Figure(data=go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.columns,
                    colorscale="Plasma", zmin=-1, zmax=1,
                    text=np.round(corr.values, 2), texttemplate="%{text}",
                ))
                corr_fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
                corr_fig = style_fig(corr_fig)
                st.plotly_chart(corr_fig, use_container_width=True)

            st.subheader("🧠 Auto-Generated Insights")

            insights = []
            top_asset = alloc_df.iloc[0]
            insights.append(
                f"The maximum Sharpe portfolio allocates the largest share "
                f"({top_asset['Weight (%)']:.1f}%) to {top_asset['Asset']}."
            )

            if sharpe_of(max_sharpe_perf) > sharpe_of(equal_perf):
                insights.append(
                    f"The optimized portfolio improves the Sharpe Ratio from "
                    f"{sharpe_of(equal_perf):.2f} (equal weight) to {sharpe_of(max_sharpe_perf):.2f}."
                )
            else:
                insights.append(
                    f"In this period, equal weighting (Sharpe {sharpe_of(equal_perf):.2f}) "
                    f"performed comparably to or better than the optimized weights "
                    f"(Sharpe {sharpe_of(max_sharpe_perf):.2f})."
                )

            vol_reduction = max_sharpe_perf[1] - min_vol_perf[1]
            ret_diff = max_sharpe_perf[0] - min_vol_perf[0]
            insights.append(
                f"Switching from the Max Sharpe portfolio to the Min Volatility portfolio "
                f"reduces annual volatility by {vol_reduction*100:.2f} percentage points, "
                f"at a cost of {ret_diff*100:.2f} percentage points of expected return."
            )

            corr_no_diag = corr.where(~np.eye(len(corr), dtype=bool))
            stacked = corr_no_diag.stack()
            if not stacked.empty:
                max_pair = stacked.idxmax()
                min_pair = stacked.idxmin()
                insights.append(
                    f"{max_pair[0]} and {max_pair[1]} are the most correlated pair "
                    f"({stacked.max():.2f}), offering limited diversification benefit between them."
                )
                insights.append(
                    f"{min_pair[0]} and {min_pair[1]} are the least correlated pair "
                    f"({stacked.min():.2f}), which may help diversify portfolio risk."
                )

            for insight in insights:
                st.markdown(f"- {insight}")

            # --- Export ---
            with st.expander("Export Allocation Data"):
                csv_data = alloc_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download optimal allocation as CSV",
                    data=csv_data,
                    file_name=f"portfolio_allocation_{start_date}_{end_date}.csv",
                    mime="text/csv",
                )

st.divider()
st.caption(
    "⚠️ Portfolio optimization is based on historical returns and covariances, which may not "
    "predict future performance. This tool is for educational purposes only and does not "
    "constitute financial advice."
)