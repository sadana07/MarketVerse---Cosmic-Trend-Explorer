"""
MarketVerse - Strategy Backtesting
-----------------------------------
Backtest SMA crossover, EMA crossover, and Buy & Hold strategies.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Strategy Backtesting - MarketVerse", layout="wide")

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
ACCENT_BLUE = "#60a5fa"


def style_fig(fig):
    fig.update_layout(
        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor=PLOTLY_BG,
        font=dict(color=PLOTLY_FONT),
        xaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        yaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
    )
    return fig


st.title(" Strategy Backtesting")
st.caption(
    "Compare rule-based trading strategies (SMA crossover, EMA crossover) against Buy & Hold. "
    "Educational simulation only — not financial advice."
)

st.sidebar.header("Backtest Settings")

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

category = st.sidebar.selectbox("Category", list(PRESET_TICKERS.keys()))
quick_pick = st.sidebar.selectbox("Quick pick", PRESET_TICKERS[category])
ticker_input = st.sidebar.text_input("Ticker symbol", value=quick_pick)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start date", pd.to_datetime("2022-01-01"))
end_date = col2.date_input("End date", pd.to_datetime("today"))

st.sidebar.subheader("Strategy")
strategy_type = st.sidebar.radio(
    "Choose strategy",
    ["SMA Crossover", "EMA Crossover"],
)

if strategy_type == "SMA Crossover":
    fast_window = st.sidebar.slider("Fast SMA window", 5, 50, 20)
    slow_window = st.sidebar.slider("Slow SMA window", 20, 200, 50)
else:
    fast_window = st.sidebar.slider("Fast EMA window", 5, 50, 12)
    slow_window = st.sidebar.slider("Slow EMA window", 20, 200, 26)

initial_capital = st.sidebar.number_input(
    "Initial capital", min_value=1000, value=100000, step=1000
)

transaction_cost = st.sidebar.slider(
    "Transaction cost per trade (%)", 0.0, 1.0, 0.1, 0.05,
    help="Applied to trade value whenever the strategy switches position.",
) / 100

risk_free_rate = st.sidebar.slider(
    "Risk-free rate (annual %)", 0.0, 10.0, 6.0, 0.5
) / 100

@st.cache_data(ttl=3600)
def load_data(ticker: str, start, end) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df

def run_backtest(df: pd.DataFrame, fast_window: int, slow_window: int,
                  strategy_type: str, initial_capital: float,
                  transaction_cost: float) -> pd.DataFrame:
    df = df.copy()

    if strategy_type == "SMA Crossover":
        df["Fast"] = df["Close"].rolling(window=fast_window).mean()
        df["Slow"] = df["Close"].rolling(window=slow_window).mean()
    else:
        df["Fast"] = df["Close"].ewm(span=fast_window, adjust=False).mean()
        df["Slow"] = df["Close"].ewm(span=slow_window, adjust=False).mean()

    df = df.dropna().reset_index(drop=True)

    df["Signal"] = np.where(df["Fast"] > df["Slow"], 1, 0)
    df["Position"] = df["Signal"].shift(1).fillna(0)

    # Daily returns
    df["Market Return"] = df["Close"].pct_change().fillna(0)
    df["Strategy Return"] = df["Position"] * df["Market Return"]

    # Transaction costs: applied when position changes
    df["Trade"] = df["Position"].diff().abs().fillna(0)
    df["Strategy Return"] -= df["Trade"] * transaction_cost

    # Equity curves
    df["Buy & Hold Equity"] = initial_capital * (1 + df["Market Return"]).cumprod()
    df["Strategy Equity"] = initial_capital * (1 + df["Strategy Return"]).cumprod()

    return df


def performance_metrics(returns: pd.Series, equity: pd.Series, risk_free_rate: float) -> dict:
    returns = returns.dropna()
    if returns.empty or equity.empty:
        return {}

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    n_days = len(returns)
    years = n_days / 252 if n_days > 0 else np.nan
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100 if years and years > 0 else np.nan

    ann_vol = returns.std() * np.sqrt(252) * 100
    ann_return = returns.mean() * 252
    sharpe = (ann_return - risk_free_rate) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else np.nan

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min() * 100

    win_rate = (returns > 0).sum() / (returns != 0).sum() * 100 if (returns != 0).sum() > 0 else np.nan

    return {
        "Total Return (%)": total_return,
        "CAGR (%)": cagr,
        "Annualized Volatility (%)": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd,
        "Win Rate (%)": win_rate,
    }


def count_trades(df: pd.DataFrame) -> int:
    return int(df["Trade"].sum() / 2) if "Trade" in df else 0

data = load_data(ticker_input, start_date, end_date)

if data.empty:
    st.error(f"No data found for '{ticker_input}'. Check the ticker symbol.")
else:
    min_required = slow_window + 5
    if len(data) < min_required:
        st.warning(
            f"Not enough data points ({len(data)}) for a {slow_window}-period window. "
            "Try a longer date range or smaller window."
        )
    else:
        bt = run_backtest(data, fast_window, slow_window, strategy_type, initial_capital, transaction_cost)

        strategy_metrics = performance_metrics(bt["Strategy Return"], bt["Strategy Equity"], risk_free_rate)
        buyhold_metrics = performance_metrics(bt["Market Return"], bt["Buy & Hold Equity"], risk_free_rate)

        st.subheader(f"Performance Summary: {ticker_input}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Strategy Final Equity",
            f"{bt['Strategy Equity'].iloc[-1]:,.0f}",
            f"{strategy_metrics.get('Total Return (%)', 0):.2f}%",
        )
        m2.metric(
            "Buy & Hold Final Equity",
            f"{bt['Buy & Hold Equity'].iloc[-1]:,.0f}",
            f"{buyhold_metrics.get('Total Return (%)', 0):.2f}%",
        )
        m3.metric("Strategy Sharpe Ratio", f"{strategy_metrics.get('Sharpe Ratio', np.nan):.2f}")
        m4.metric("Number of Trades", f"{count_trades(bt)}")

        st.subheader("Equity Curve: Strategy vs Buy & Hold")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bt["Date"], y=bt["Strategy Equity"], name=f"{strategy_type} Strategy",
            line=dict(color=ACCENT_VIOLET, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=bt["Date"], y=bt["Buy & Hold Equity"], name="Buy & Hold",
            line=dict(color=ACCENT_TEAL, width=2),
        ))
        fig.update_layout(
            height=450,
            yaxis_title="Portfolio Value",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig = style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Price with Strategy Signals")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=bt["Date"], y=bt["Close"], name="Close",
            line=dict(color="#e6e1f5", width=1),
        ))
        fig2.add_trace(go.Scatter(
            x=bt["Date"], y=bt["Fast"], name=f"Fast ({fast_window})",
            line=dict(color=ACCENT_GOLD, width=1.5),
        ))
        fig2.add_trace(go.Scatter(
            x=bt["Date"], y=bt["Slow"], name=f"Slow ({slow_window})",
            line=dict(color=ACCENT_BLUE, width=1.5),
        ))

        entries = bt[bt["Trade"] > 0]
        long_entries = entries[entries["Position"] == 1]
        long_exits = entries[entries["Position"] == 0]

        if not long_entries.empty:
            fig2.add_trace(go.Scatter(
                x=long_entries["Date"], y=long_entries["Close"], mode="markers",
                name="Buy Signal", marker=dict(color=ACCENT_TEAL, size=9, symbol="triangle-up"),
            ))
        if not long_exits.empty:
            fig2.add_trace(go.Scatter(
                x=long_exits["Date"], y=long_exits["Close"], mode="markers",
                name="Sell Signal", marker=dict(color=ACCENT_PINK, size=9, symbol="triangle-down"),
            ))

        fig2.update_layout(
            height=450,
            yaxis_title="Price",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig2 = style_fig(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Drawdown Comparison")

        strat_dd = (bt["Strategy Equity"] - bt["Strategy Equity"].cummax()) / bt["Strategy Equity"].cummax() * 100
        bh_dd = (bt["Buy & Hold Equity"] - bt["Buy & Hold Equity"].cummax()) / bt["Buy & Hold Equity"].cummax() * 100

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=bt["Date"], y=strat_dd, name=f"{strategy_type} Drawdown",
            line=dict(color=ACCENT_VIOLET, width=1.5), fill="tozeroy",
        ))
        fig3.add_trace(go.Scatter(
            x=bt["Date"], y=bh_dd, name="Buy & Hold Drawdown",
            line=dict(color=ACCENT_TEAL, width=1.5), fill="tozeroy",
        ))
        fig3.update_layout(
            height=350,
            yaxis_title="Drawdown (%)",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig3 = style_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Performance Metrics Comparison")

        metrics_df = pd.DataFrame({
            f"{strategy_type} Strategy": strategy_metrics,
            "Buy & Hold": buyhold_metrics,
        })
        st.dataframe(metrics_df.style.format("{:.2f}"), use_container_width=True)

        st.subheader("🧠 Auto-Generated Insights")

        insights = []
        strat_total = strategy_metrics.get("Total Return (%)", 0)
        bh_total = buyhold_metrics.get("Total Return (%)", 0)

        if strat_total > bh_total:
            insights.append(
                f"The {strategy_type} strategy outperformed Buy & Hold by "
                f"{strat_total - bh_total:.2f} percentage points over this period."
            )
        else:
            insights.append(
                f"Buy & Hold outperformed the {strategy_type} strategy by "
                f"{bh_total - strat_total:.2f} percentage points over this period."
            )

        strat_sharpe = strategy_metrics.get("Sharpe Ratio", np.nan)
        bh_sharpe = buyhold_metrics.get("Sharpe Ratio", np.nan)
        if pd.notna(strat_sharpe) and pd.notna(bh_sharpe):
            if strat_sharpe > bh_sharpe:
                insights.append(
                    f"The strategy achieved a better risk-adjusted return "
                    f"(Sharpe {strat_sharpe:.2f} vs {bh_sharpe:.2f})."
                )
            else:
                insights.append(
                    f"Buy & Hold had a better risk-adjusted return "
                    f"(Sharpe {bh_sharpe:.2f} vs {strat_sharpe:.2f})."
                )

        strat_dd_val = strategy_metrics.get("Max Drawdown (%)", 0)
        bh_dd_val = buyhold_metrics.get("Max Drawdown (%)", 0)
        if strat_dd_val > bh_dd_val:
            insights.append(
                f"The strategy had a shallower max drawdown ({strat_dd_val:.2f}%) "
                f"compared to Buy & Hold ({bh_dd_val:.2f}%), suggesting it may reduce downside risk."
            )
        else:
            insights.append(
                f"Buy & Hold had a shallower max drawdown ({bh_dd_val:.2f}%) "
                f"compared to the strategy ({strat_dd_val:.2f}%)."
            )

        trades = count_trades(bt)
        total_cost_pct = bt["Trade"].sum() * transaction_cost * 100
        insights.append(
            f"The strategy executed approximately {trades} round-trip trades, "
            f"incurring an estimated {total_cost_pct:.2f}% cumulative transaction cost drag."
        )

        win_rate = strategy_metrics.get("Win Rate (%)", np.nan)
        if pd.notna(win_rate):
            insights.append(f"The strategy's daily win rate was {win_rate:.1f}% over the test period.")

        for insight in insights:
            st.markdown(f"- {insight}")

        with st.expander("View Backtest Data / Export"):
            display_cols = ["Date", "Close", "Fast", "Slow", "Position",
                             "Strategy Return", "Strategy Equity", "Buy & Hold Equity"]
            st.dataframe(bt[display_cols].tail(100), use_container_width=True)

            csv_data = bt[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download backtest results as CSV",
                data=csv_data,
                file_name=f"{ticker_input}_backtest_{strategy_type.replace(' ', '_')}.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "⚠️ Backtests do not account for slippage, taxes, liquidity constraints, or market impact. "
    "Past performance does not guarantee future results. Educational use only."
)