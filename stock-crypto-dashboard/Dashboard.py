"""
MarketVerse - Cosmic Trend Explorer
--------------------------------
A Streamlit dashboard for analyzing historical price trends of
stocks (Indian & US) and cryptocurrencies using yfinance.

Run with: streamlit run Dashboard.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy.optimize import minimize
from fpdf import FPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="MarketVerse - Cosmic Trend Explorer",
    layout="wide"
)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 50% 30%, #2d1b4e 0%, #1a0f30 45%, #0d0618 100%);
        }
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stHeader"] { background: transparent; }

        .login-bg-svg {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            opacity: 0.12;
            z-index: 0;
            pointer-events: none;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(160deg, #7c3aed 0%, #a855f7 100%);
            border-radius: 18px;
            padding: 10px 28px 28px 28px;
            box-shadow: 0 0 40px rgba(168, 85, 247, 0.45);
            margin-top: 8vh;
        }

        .login-title {
            text-align: center;
            color: #ffffff !important;
            text-shadow: none;
            letter-spacing: 2px;
            margin: 4px 0 20px 0;
        }
        .login-icon {
            width: 56px; height: 56px;
            margin: 16px auto 4px auto;
            display: flex; align-items: center; justify-content: center;
            background: #ffffff;
            border-radius: 50%;
            font-size: 28px;
        }

        /* Input text - black, with visible outline */
        div[data-testid="stTextInput"] input {
            background: rgba(255,255,255,0.9) !important;
            border: 1px solid rgba(255,255,255,0.5) !important;
            color: #000000 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stTextInput"] label {
            color: #f3e8ff !important;
            font-weight: 500;
        }
        div[data-testid="stCheckbox"] label p {
            color: #f3e8ff !important;
        }
        div[data-testid="stForm"] {
            border: none;
            padding: 0;
            background: transparent;
        }

        /* Login button - white text, always visible */
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stFormSubmitButton"] button p,
        div[data-testid="stFormSubmitButton"] button span {
            color: #ffffff !important;
        }
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
            border: none !important;
        }

        /* Forgot password button - subtle link style */
        div[data-testid="stButton"] button[kind="secondary"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #e9d5ff !important;
            font-size: 0.83em !important;
            text-decoration: underline !important;
            padding: 2px 0 !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            color: #ffffff !important;
            background: transparent !important;
        }
        </style>

        <svg class="login-bg-svg" viewBox="0 0 1920 1080" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#a855f7"/>
                    <stop offset="100%" stop-color="#2d1b4e"/>
                </linearGradient>
            </defs>
            <g fill="url(#barGrad)">
                <rect x="50"  y="600" width="40" height="300"/>
                <rect x="110" y="500" width="40" height="400"/>
                <rect x="170" y="650" width="40" height="250"/>
                <rect x="230" y="450" width="40" height="450"/>
                <rect x="290" y="550" width="40" height="350"/>
                <rect x="350" y="400" width="40" height="500"/>
                <rect x="410" y="500" width="40" height="400"/>
                <rect x="1500" y="500" width="40" height="400"/>
                <rect x="1560" y="600" width="40" height="300"/>
                <rect x="1620" y="450" width="40" height="450"/>
                <rect x="1680" y="550" width="40" height="350"/>
                <rect x="1740" y="400" width="40" height="500"/>
                <rect x="1800" y="500" width="40" height="400"/>
                <rect x="1860" y="600" width="40" height="300"/>
            </g>
            <polyline points="50,700 200,550 350,650 500,400 650,500 800,300 950,450 1100,250 1250,400 1400,200 1550,350 1700,150 1850,300"
                      fill="none" stroke="#5eead4" stroke-width="6" opacity="0.5"/>
            <text x="960" y="540" font-family="monospace" font-size="220" fill="#ffffff" text-anchor="middle" opacity="0.04">MarketVerse</text>
        </svg>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.2, 1])

    with center:
        with st.container(border=True):
            st.markdown('<div class="login-icon">🌌</div>', unsafe_allow_html=True)
            st.markdown('<h2 class="login-title">LOG IN</h2>', unsafe_allow_html=True)

            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                remember = st.checkbox("Remember me")
                submitted = st.form_submit_button("Login", use_container_width=True)

                if submitted:
                    if username.strip() != "" and password.strip() != "":
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Please enter both username and password.")

            if "show_forgot" not in st.session_state:
                st.session_state.show_forgot = False

            if st.button("🔑 Forgot Password?", use_container_width=True,
                         type="secondary", key="forgot_btn"):
                st.session_state.show_forgot = not st.session_state.show_forgot

            if st.session_state.show_forgot:
                st.markdown(
                    """
                    <div style="
                        background: rgba(255,255,255,0.10);
                        border: 1px solid rgba(255,255,255,0.22);
                        border-radius: 10px;
                        padding: 12px 16px;
                        margin-top: 6px;
                        color: #f3e8ff;
                        font-size: 0.87em;
                        line-height: 1.65;
                    ">
                        <strong>🔐 Password Reset</strong><br/>
                        To reset your password, please contact your administrator
                        or reach out to support at<br/>
                        <a href="mailto:support@marketverse.app"
                           style="color:#e9d5ff; font-weight:600; text-decoration:underline;">
                            support@marketverse.app
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.stop()

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

    /* Top-right user badge */
    .topbar-user {
        position: fixed;
        top: 8px;
        right: 70px;
        z-index: 999999;
        background: rgba(108, 63, 197, 0.25);
        border: 1px solid rgba(167, 102, 255, 0.35);
        border-radius: 8px;
        padding: 6px 14px;
        color: #ffffff;
        font-size: 0.85em;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="topbar-user">👤 {st.session_state.get("username", "User")}</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    st.divider()

# Plotly template colors for the galaxy theme
PLOTLY_BG = "#160a28"
PLOTLY_GRID = "rgba(167, 102, 255, 0.15)"
PLOTLY_FONT = "#e6e1f5"
ACCENT_VIOLET = "#a855f7"
ACCENT_PINK = "#f472d0"
ACCENT_TEAL = "#5eead4"


def style_fig(fig):
    """Apply galaxy/violet dark theme to a plotly figure."""
    fig.update_layout(
        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor=PLOTLY_BG,
        font=dict(color=PLOTLY_FONT),
        legend=dict(font=dict(color="#ffffff")),
        xaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        yaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
    )
    if hasattr(fig.layout, "yaxis2") and fig.layout.yaxis2 is not None:
        fig.update_layout(
            xaxis2=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
            yaxis2=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        )
    return fig


st.title(" MarketVerse - Cosmic Trend Explorer")
st.caption(
    "Educational data analysis project — not financial advice. "
    "Data sourced via Yahoo Finance (yfinance)."
)

st.sidebar.header("Settings")

PRESET_TICKERS = {
    "Indian Stocks": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
        "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS", "BAJFINANCE.NS",
        "HCLTECH.NS", "ADANIENT.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "ONGC.NS",
        "NTPC.NS", "POWERGRID.NS", "M&M.NS", "TECHM.NS", "JSWSTEEL.NS",
        "COALINDIA.NS", "GRASIM.NS", "DRREDDY.NS", "CIPLA.NS", "BAJAJFINSV.NS",
        "HEROMOTOCO.NS", "EICHERMOT.NS", "DIVISLAB.NS", "BRITANNIA.NS", "ADANIPORTS.NS",
    ],
    "US Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
        "JPM", "V", "MA", "WMT", "DIS", "BA", "KO", "PEP",
        "INTC", "AMD", "ORCL", "CSCO", "ADBE", "PYPL", "UBER", "ABNB",
        "PFE", "XOM", "CVX", "NKE", "MCD", "T", "VZ", "IBM",
    ],
    "Crypto": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
        "DOGE-USD", "MATIC-USD", "DOT-USD", "LTC-USD", "AVAX-USD", "SHIB-USD",
        "TRX-USD", "LINK-USD", "ATOM-USD",
    ],
}

category = st.sidebar.selectbox("Choose category", list(PRESET_TICKERS.keys()))

quick_pick = st.sidebar.selectbox(
    "Quick pick from list",
    PRESET_TICKERS[category],
    help="Pick from common tickers, or type a custom one below",
)

ticker_input = st.sidebar.text_input(
    "Enter ticker symbol (or use quick pick above)",
    value=quick_pick,
    help="Examples: RELIANCE.NS (NSE), AAPL (US stock), BTC-USD (crypto)",
)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start date", pd.to_datetime("2023-01-01"))
end_date = col2.date_input("End date", pd.to_datetime("today"))

st.sidebar.subheader("Indicators")
show_sma = st.sidebar.checkbox("Show SMA (20, 50)", value=True)
show_ema = st.sidebar.checkbox("Show EMA (20)", value=False)
show_volume = st.sidebar.checkbox("Show Volume", value=True)

st.sidebar.subheader("Compare Returns")
compare_tickers = st.sidebar.multiselect(
    "Select tickers to compare (% returns)",
    options=sum(PRESET_TICKERS.values(), []),
    default=["RELIANCE.NS", "BTC-USD"],
)

st.sidebar.subheader("Risk Settings")
risk_free_rate = st.sidebar.slider(
    "Risk-free rate (annual %)", 0.0, 10.0, 6.0, 0.5,
    help="Used in Sharpe Ratio and portfolio optimization calculations.",
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Daily Return (%)"] = df["Close"].pct_change() * 100
    return df


def get_benchmark_ticker(ticker: str):
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "^NSEI"
    elif ticker.endswith("-USD"):
        return "BTC-USD" if ticker != "BTC-USD" else None
    else:
        return "^GSPC"


def compute_risk_metrics(df: pd.DataFrame, benchmark_df: pd.DataFrame = None, risk_free_rate: float = 0.06) -> dict:
    rets = df["Close"].pct_change().dropna()
    if rets.empty:
        return {}

    ann_return = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol else np.nan

    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    beta = np.nan
    if benchmark_df is not None and not benchmark_df.empty:
        bench_rets = benchmark_df["Close"].pct_change().dropna()
        aligned = pd.concat([rets, bench_rets], axis=1, join="inner")
        aligned.columns = ["asset", "bench"]
        if len(aligned) > 1 and aligned["bench"].var() != 0:
            beta = aligned["asset"].cov(aligned["bench"]) / aligned["bench"].var()

    return {
        "Annualized Return (%)": ann_return * 100,
        "Annualized Volatility (%)": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd * 100,
        "Beta": beta,
    }


PREDICTION_FEATURES = [
    "Lag1", "Lag2", "Lag3", "Lag5",
    "RollingMean5", "RollingMean10", "RollingStd5", "Volume_Change",
]


def create_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Lag1"] = df["Close"].shift(1)
    df["Lag2"] = df["Close"].shift(2)
    df["Lag3"] = df["Close"].shift(3)
    df["Lag5"] = df["Close"].shift(5)
    df["RollingMean5"] = df["Close"].rolling(5).mean()
    df["RollingMean10"] = df["Close"].rolling(10).mean()
    df["RollingStd5"] = df["Close"].rolling(5).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_Change"] = df["Volume_Change"].replace([np.inf, -np.inf], np.nan)
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def train_prediction_models(feat_df: pd.DataFrame):
    X = feat_df[PREDICTION_FEATURES]
    y = feat_df["Target"]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    xgb_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)

    return rf, xgb_model, X_train, X_test, y_train, y_test


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.sum(mean_returns * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return port_return, port_vol


def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return, port_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    if port_vol == 0:
        return 0.0
    return -(port_return - risk_free_rate) / port_vol


def optimize_portfolio(returns_df: pd.DataFrame, risk_free_rate: float = 0.06):
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    n_assets = len(mean_returns)

    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    init_guess = np.array(n_assets * [1.0 / n_assets])

    result = minimize(
        negative_sharpe, init_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return result.x, mean_returns, cov_matrix


def generate_insights(ticker, df, risk_metrics, pred_metrics=None, corr_matrix=None, benchmark_name=None):
    insights = []

    latest_close = df["Close"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]

    if pd.notna(sma50):
        if latest_close > sma50:
            insights.append(
                f"{ticker} is currently trading above its 50-day moving average "
                f"({sma50:.2f}), indicating an overall uptrend."
            )
        else:
            insights.append(
                f"{ticker} is currently trading below its 50-day moving average "
                f"({sma50:.2f}), indicating a downtrend or consolidation phase."
            )

    if pd.notna(sma20) and pd.notna(sma50):
        if sma20 > sma50:
            insights.append("The 20-day SMA is above the 50-day SMA — a bullish 'golden cross' style signal.")
        else:
            insights.append("The 20-day SMA is below the 50-day SMA — a bearish 'death cross' style signal.")

    vol = risk_metrics.get("Annualized Volatility (%)") if risk_metrics else None
    if vol is not None:
        if vol > 60:
            insights.append(f"Annualized volatility is high at {vol:.1f}%, suggesting significant price swings and higher risk.")
        elif vol > 25:
            insights.append(f"Annualized volatility is moderate at {vol:.1f}%, typical for actively traded assets.")
        else:
            insights.append(f"Annualized volatility is relatively low at {vol:.1f}%, suggesting more stable price behavior.")

    sharpe = risk_metrics.get("Sharpe Ratio") if risk_metrics else None
    if sharpe is not None and pd.notna(sharpe):
        if sharpe > 1:
            insights.append(f"A Sharpe Ratio of {sharpe:.2f} indicates strong risk-adjusted returns over this period.")
        elif sharpe > 0:
            insights.append(f"A Sharpe Ratio of {sharpe:.2f} indicates positive but modest risk-adjusted returns.")
        else:
            insights.append(f"A negative Sharpe Ratio of {sharpe:.2f} suggests returns haven't compensated for the risk taken.")

    max_dd = risk_metrics.get("Max Drawdown (%)") if risk_metrics else None
    if max_dd is not None:
        insights.append(f"The maximum drawdown over this period was {max_dd:.1f}%, the largest peak-to-trough decline.")

    beta = risk_metrics.get("Beta") if risk_metrics else None
    if beta is not None and pd.notna(beta):
        ref = benchmark_name or "the benchmark"
        if beta > 1.2:
            insights.append(f"A Beta of {beta:.2f} suggests {ticker} is more volatile than {ref}.")
        elif beta < 0.8:
            insights.append(f"A Beta of {beta:.2f} suggests {ticker} is less volatile than {ref}.")
        else:
            insights.append(f"A Beta of {beta:.2f} suggests {ticker} moves roughly in line with {ref}.")

    if pred_metrics:
        best_model = max(pred_metrics.items(), key=lambda kv: kv[1][2])
        insights.append(f"Among the prediction models, {best_model[0]} achieved the best fit with an R² of {best_model[1][2]:.2f} on the test set.")

    if corr_matrix is not None and len(corr_matrix.columns) > 1:
        corr_no_diag = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool))
        stacked = corr_no_diag.stack()
        if not stacked.empty:
            max_pair = stacked.idxmax()
            max_val = stacked.max()
            if pd.notna(max_val):
                insights.append(f"{max_pair[0]} and {max_pair[1]} show the strongest correlation ({max_val:.2f}) among the compared assets.")

    return insights


def generate_pdf_report(ticker, df, risk_metrics, pred_metrics, insights, corr_matrix=None):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "MarketVerse - Cosmic Trend Explorer", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Report for: {ticker}", ln=True)
    pdf.cell(0, 8, f"Period: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}", ln=True)
    pdf.ln(4)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(df["Date"], df["Close"], color="#7c3aed", linewidth=1.5)
    ax.set_title(f"{ticker} Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    pdf.image(buf, x=10, w=190)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Risk Metrics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k, v in (risk_metrics or {}).items():
        val_text = "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
        pdf.cell(0, 8, f"{k}: {val_text}", ln=True)
    pdf.ln(2)

    if pred_metrics:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Price Prediction Model Performance", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for model_name, (rmse, mae, r2) in pred_metrics.items():
            pdf.cell(0, 8, f"{model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}", ln=True)
        pdf.ln(2)

    if corr_matrix is not None and len(corr_matrix.columns) > 1:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Correlation Matrix (daily returns)", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for idx in corr_matrix.index:
            row_text = ", ".join(f"{col}={corr_matrix.loc[idx, col]:.2f}" for col in corr_matrix.columns)
            pdf.multi_cell(0, 6, f"{idx}: {row_text}")
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Auto-Generated Insights", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for insight in (insights or []):
        pdf.multi_cell(0, 7, f"- {insight}")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 6, "Disclaimer: This report is for educational purposes only and does not constitute financial advice.")

    return bytes(pdf.output())


data = load_data(ticker_input, start_date, end_date)

if data.empty:
    st.error(
        f"No data found for '{ticker_input}'. "
        "Check the ticker symbol (e.g. RELIANCE.NS, AAPL, BTC-USD)."
    )
else:
    data = add_indicators(data)

    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    change = latest["Close"] - prev["Close"]
    pct_change = (change / prev["Close"]) * 100 if prev["Close"] else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Close", f"{latest['Close']:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
    m2.metric("52-Week High", f"{data['Close'].max():.2f}")
    m3.metric("52-Week Low", f"{data['Close'].min():.2f}")
    m4.metric("Avg Daily Volatility (%)", f"{data['Daily Return (%)'].std():.2f}")

    st.subheader(f"Price Chart: {ticker_input}")

    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3] if show_volume else [1],
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color=ACCENT_TEAL,
            decreasing_line_color=ACCENT_PINK,
        ),
        row=1,
        col=1,
    )

    if show_sma:
        fig.add_trace(
            go.Scatter(x=data["Date"], y=data["SMA20"], name="SMA 20",
                       line=dict(width=1.5, color=ACCENT_VIOLET)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=data["Date"], y=data["SMA50"], name="SMA 50",
                       line=dict(width=1.5, color="#facc15")),
            row=1, col=1,
        )

    if show_ema:
        fig.add_trace(
            go.Scatter(x=data["Date"], y=data["EMA20"], name="EMA 20",
                       line=dict(width=1.5, dash="dot", color="#60a5fa")),
            row=1, col=1,
        )

    if show_volume:
        fig.add_trace(
            go.Bar(x=data["Date"], y=data["Volume"], name="Volume",
                   marker_color="rgba(168, 85, 247, 0.45)"),
            row=2, col=1,
        )

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig = style_fig(fig)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Daily Returns Distribution"):
        st.bar_chart(data.set_index("Date")["Daily Return (%)"])

    with st.expander("View Raw Data / Export Dataset"):
        st.dataframe(data.tail(50), use_container_width=True)

        csv_data = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download {ticker_input} dataset as CSV",
            data=csv_data,
            file_name=f"{ticker_input}_{start_date}_{end_date}.csv",
            mime="text/csv",
        )


st.divider()
st.subheader(" Compare Normalized Returns")
st.caption("All assets rebased to start at 100 for fair comparison.")

if compare_tickers:
    comp_fig = go.Figure()
    comp_data_frames = {}
    palette = [ACCENT_VIOLET, ACCENT_TEAL, ACCENT_PINK, "#facc15", "#60a5fa", "#fb923c"]
    for i, t in enumerate(compare_tickers):
        df_t = load_data(t, start_date, end_date)
        if df_t.empty:
            st.warning(f"No data for {t}, skipping.")
            continue
        normalized = (df_t["Close"] / df_t["Close"].iloc[0]) * 100
        comp_fig.add_trace(
            go.Scatter(
                x=df_t["Date"], y=normalized, name=t,
                line=dict(color=palette[i % len(palette)], width=2),
            )
        )
        comp_data_frames[t] = df_t.assign(**{f"{t} Normalized": normalized})

    comp_fig.update_layout(
        height=450,
        yaxis_title="Normalized Price (Base = 100)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    comp_fig = style_fig(comp_fig)
    st.plotly_chart(comp_fig, use_container_width=True)

    if comp_data_frames:
        combined = pd.DataFrame()
        for t, df_t in comp_data_frames.items():
            combined[f"{t} Close"] = df_t.set_index("Date")["Close"]
            combined[f"{t} Normalized"] = df_t.set_index("Date")[f"{t} Normalized"]
        combined = combined.reset_index()

        csv_combined = combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download comparison dataset as CSV",
            data=csv_combined,
            file_name=f"comparison_{start_date}_{end_date}.csv",
            mime="text/csv",
        )
else:
    st.info("Select tickers in the sidebar to compare their performance.")


st.divider()
st.caption(
    "⚠️ This dashboard is for educational and analytical purposes only. "
    "It does not constitute financial advice. Data via Yahoo Finance."
)