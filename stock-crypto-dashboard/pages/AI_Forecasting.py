"""
MarketVerse - AI Forecasting Lab
-----------------------------------
Educational demonstration of ML-based price forecasting using
Random Forest and XGBoost, with transparent evaluation metrics
(RMSE, MAE, R²) and feature importance.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(page_title="AI Forecasting Lab - MarketVerse", layout="wide")

if not st.session_state.get("authenticated", False):
    st.error("🔒 Please log in from the Dashboard page first.")
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
    .disclaimer-box {
        background: rgba(244, 114, 208, 0.08);
        border: 1px solid rgba(244, 114, 208, 0.35);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 14px;
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


st.title("AI Forecasting Lab")

st.markdown(
    """
    <div class="disclaimer-box">
    <strong>⚠️ Educational tool only.</strong> This page demonstrates how machine learning
    models (Random Forest, XGBoost) can be applied to historical price data to predict the
    <em>next day's closing price</em>. These models use only lagged price/volume features and
    <strong>cannot reliably forecast real market movements</strong>. Performance metrics
    (RMSE, MAE, R²) are shown transparently so you can judge model quality yourself — high error
    or low R² is common and expected. Do not use these predictions for trading decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Forecasting Settings")

PRESET_TICKERS = {
    "Indian Stocks": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    ],
    "US Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
        "JPM", "V", "MA", "WMT", "DIS", "BA", "KO", "PEP",
    ],
    "Crypto": [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    ],
}

category = st.sidebar.selectbox("Category", list(PRESET_TICKERS.keys()))
quick_pick = st.sidebar.selectbox("Quick pick", PRESET_TICKERS[category])
ticker_input = st.sidebar.text_input("Ticker symbol", value=quick_pick)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start date", pd.to_datetime("2021-01-01"))
end_date = col2.date_input("End date", pd.to_datetime("today"))

test_size = st.sidebar.slider(
    "Test set size (%)", 10, 40, 20, 5,
    help="Most recent portion of data held out for evaluation (chronological split, no shuffling)."
) / 100

st.sidebar.subheader("Random Forest Hyperparameters")
rf_n_estimators = st.sidebar.slider("RF: number of trees", 50, 500, 200, 50)
rf_max_depth = st.sidebar.slider("RF: max depth", 2, 20, 10, 1)

st.sidebar.subheader("XGBoost Hyperparameters")
xgb_n_estimators = st.sidebar.slider("XGB: number of trees", 50, 500, 200, 50)
xgb_max_depth = st.sidebar.slider("XGB: max depth", 2, 12, 5, 1)
xgb_learning_rate = st.sidebar.slider("XGB: learning rate", 0.01, 0.3, 0.05, 0.01)

@st.cache_data(ttl=3600)
def load_data(ticker: str, start, end) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df


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


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


@st.cache_resource(show_spinner=False)
def train_models(feat_df: pd.DataFrame, test_size: float,
                  rf_n_estimators: int, rf_max_depth: int,
                  xgb_n_estimators: int, xgb_max_depth: int, xgb_learning_rate: float):
    X = feat_df[PREDICTION_FEATURES]
    y = feat_df["Target"]

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators, max_depth=rf_max_depth,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    xgb_model = XGBRegressor(
        n_estimators=xgb_n_estimators, max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate, random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    return rf, xgb_model, X_train, X_test, y_train, y_test, split_idx


def naive_baseline_metrics(feat_df: pd.DataFrame, split_idx: int):
    test_slice = feat_df.iloc[split_idx:]
    y_true = test_slice["Target"]
    y_pred_naive = test_slice["Close"]
    return regression_metrics(y_true, y_pred_naive)


data = load_data(ticker_input, start_date, end_date)

if data.empty:
    st.error(f"No data found for '{ticker_input}'. Check the ticker symbol.")
else:
    feat_df = create_prediction_features(data)

    if len(feat_df) < 50:
        st.warning("Not enough data to train models reliably. Try a longer date range.")
    else:
        with st.spinner("Training models..."):
            rf, xgb_model, X_train, X_test, y_train, y_test, split_idx = train_models(
                feat_df, test_size,
                rf_n_estimators, rf_max_depth,
                xgb_n_estimators, xgb_max_depth, xgb_learning_rate,
            )

        rf_pred_test = rf.predict(X_test)
        xgb_pred_test = xgb_model.predict(X_test)

        rf_metrics = regression_metrics(y_test, rf_pred_test)
        xgb_metrics = regression_metrics(y_test, xgb_pred_test)
        naive_metrics = naive_baseline_metrics(feat_df, split_idx)

        pred_metrics = {
            "Random Forest": rf_metrics,
            "XGBoost": xgb_metrics,
            "Naive Baseline (no change)": naive_metrics,
        }

        st.subheader(f"Model Evaluation: {ticker_input} (next-day close)")

        metrics_df = pd.DataFrame(
            pred_metrics, index=["RMSE", "MAE", "R²"]
        ).T

        m1, m2, m3 = st.columns(3)
        m1.metric("Random Forest R²", f"{rf_metrics[2]:.3f}")
        m2.metric("XGBoost R²", f"{xgb_metrics[2]:.3f}")
        m3.metric("Naive Baseline R²", f"{naive_metrics[2]:.3f}")

        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

        st.caption(
            "RMSE and MAE are in price units (lower is better). R² compares the model to a "
            "mean-prediction baseline — values near or below 0 indicate the model adds little "
            "or no predictive value beyond simple persistence."
        )

        st.subheader("Predicted vs Actual Close Price (Test Set)")

        offset = len(data) - len(feat_df)
        test_dates = data["Date"].iloc[split_idx + offset: split_idx + offset + len(y_test)].reset_index(drop=True)

        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(
            x=test_dates, y=y_test.values, name="Actual",
            line=dict(color="#e6e1f5", width=2),
        ))
        pred_fig.add_trace(go.Scatter(
            x=test_dates, y=rf_pred_test, name="Random Forest Prediction",
            line=dict(color=ACCENT_VIOLET, width=1.5, dash="dot"),
        ))
        pred_fig.add_trace(go.Scatter(
            x=test_dates, y=xgb_pred_test, name="XGBoost Prediction",
            line=dict(color=ACCENT_TEAL, width=1.5, dash="dot"),
        ))
        pred_fig.update_layout(
            height=450,
            yaxis_title="Close Price",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        pred_fig = style_fig(pred_fig)
        st.plotly_chart(pred_fig, use_container_width=True)

        st.subheader("Prediction Error (Residuals) on Test Set")

        rf_residuals = y_test.values - rf_pred_test
        xgb_residuals = y_test.values - xgb_pred_test

        resid_fig = go.Figure()
        resid_fig.add_trace(go.Scatter(
            x=test_dates, y=rf_residuals, name="RF Residual",
            line=dict(color=ACCENT_VIOLET, width=1.5),
        ))
        resid_fig.add_trace(go.Scatter(
            x=test_dates, y=xgb_residuals, name="XGBoost Residual",
            line=dict(color=ACCENT_TEAL, width=1.5),
        ))
        resid_fig.add_hline(y=0, line_dash="dash", line_color=PLOTLY_GRID)
        resid_fig.update_layout(
            height=350,
            yaxis_title="Actual - Predicted",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        resid_fig = style_fig(resid_fig)
        st.plotly_chart(resid_fig, use_container_width=True)

        st.subheader("Feature Importance")

        col_a, col_b = st.columns(2)

        with col_a:
            rf_importance = pd.Series(rf.feature_importances_, index=PREDICTION_FEATURES).sort_values()
            rf_fig = go.Figure(go.Bar(
                x=rf_importance.values, y=rf_importance.index, orientation="h",
                marker_color=ACCENT_VIOLET,
            ))
            rf_fig.update_layout(
                title="Random Forest", height=350, margin=dict(l=10, r=10, t=40, b=10),
            )
            rf_fig = style_fig(rf_fig)
            st.plotly_chart(rf_fig, use_container_width=True)

        with col_b:
            xgb_importance = pd.Series(xgb_model.feature_importances_, index=PREDICTION_FEATURES).sort_values()
            xgb_fig = go.Figure(go.Bar(
                x=xgb_importance.values, y=xgb_importance.index, orientation="h",
                marker_color=ACCENT_TEAL,
            ))
            xgb_fig.update_layout(
                title="XGBoost", height=350, margin=dict(l=10, r=10, t=40, b=10),
            )
            xgb_fig = style_fig(xgb_fig)
            st.plotly_chart(xgb_fig, use_container_width=True)

        st.subheader("Next-Day Forecast (Illustrative)")

        last_row = feat_df[PREDICTION_FEATURES].iloc[[-1]]
        rf_next = rf.predict(last_row)[0]
        xgb_next = xgb_model.predict(last_row)[0]
        last_close = data["Close"].iloc[-1]

        f1, f2, f3 = st.columns(3)
        f1.metric("Last Close", f"{last_close:.2f}")
        f2.metric("RF Forecast", f"{rf_next:.2f}", f"{rf_next - last_close:.2f}")
        f3.metric("XGBoost Forecast", f"{xgb_next:.2f}", f"{xgb_next - last_close:.2f}")

        st.caption(
            "These single-point forecasts illustrate model output only. Given the R² values above, "
            "treat them as a demonstration of model mechanics rather than an actionable signal."
        )

        st.subheader("🧠 Auto-Generated Insights")

        insights = []

        best_model_name = max(
            [("Random Forest", rf_metrics), ("XGBoost", xgb_metrics)],
            key=lambda kv: kv[1][2],
        )[0]
        insights.append(
            f"On the test set, {best_model_name} achieved the higher R² "
            f"(RF: {rf_metrics[2]:.3f}, XGBoost: {xgb_metrics[2]:.3f})."
        )

        if rf_metrics[2] < naive_metrics[2] and xgb_metrics[2] < naive_metrics[2]:
            insights.append(
                "Both models underperformed the naive 'no change' baseline on R², which is common "
                "for short-horizon price prediction and highlights how difficult it is to beat "
                "simple persistence using only price/volume history."
            )
        elif rf_metrics[2] > naive_metrics[2] or xgb_metrics[2] > naive_metrics[2]:
            insights.append(
                "At least one model slightly outperformed the naive baseline on this test set, "
                "though this does not guarantee out-of-sample predictive power on future data."
            )

        top_feature_rf = rf_importance.index[-1]
        top_feature_xgb = xgb_importance.index[-1]
        insights.append(
            f"The most influential feature was '{top_feature_rf}' for Random Forest and "
            f"'{top_feature_xgb}' for XGBoost — both models rely heavily on recent price levels "
            f"rather than volume-based signals."
        )

        insights.append(
            "RMSE and MAE are expressed in the same units as the closing price — compare them to "
            "the asset's typical daily price range to judge whether the error magnitude is large or small."
        )

        for insight in insights:
            st.markdown(f"- {insight}")

        with st.expander("Export Model Results"):
            results_df = pd.DataFrame({
                "Date": test_dates,
                "Actual": y_test.values,
                "RF Prediction": rf_pred_test,
                "XGBoost Prediction": xgb_pred_test,
            })
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download model predictions as CSV",
                data=csv_data,
                file_name=f"{ticker_input}_forecast_results.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "⚠️ This is an educational demonstration of machine learning regression techniques applied "
    "to historical price data. It is NOT a reliable forecasting tool and should not be used for "
    "investment decisions. Past patterns do not predict future prices."
)