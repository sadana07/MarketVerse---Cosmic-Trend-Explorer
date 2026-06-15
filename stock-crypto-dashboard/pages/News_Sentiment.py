"""
MarketVerse - News Sentiment Analysis
-----------------------------------------
Fetches recent financial news headlines via yfinance and scores
sentiment using VADER (NLTK).
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import nltk

st.set_page_config(page_title="News Sentiment - MarketVerse", layout="wide")

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

    a { color: #5eead4 !important; }
    /* News card styling */
    .news-card {
        background: rgba(108, 63, 197, 0.12);
        border: 1px solid rgba(167, 102, 255, 0.25);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .news-card a {
        font-weight: 600;
        text-decoration: none;
    }
    .news-card a:hover {
        text-decoration: underline;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.8em;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-positive { background: rgba(94, 234, 212, 0.2); color: #5eead4; border: 1px solid #5eead4; }
    .badge-negative { background: rgba(244, 114, 208, 0.2); color: #f472d0; border: 1px solid #f472d0; }
    .badge-neutral { background: rgba(168, 85, 247, 0.2); color: #d9b8ff; border: 1px solid #a855f7; }
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
        font=dict(color=PLOTLY_FONT),
        xaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        yaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
    )
    return fig

@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer():
    from nltk.sentiment import SentimentIntensityAnalyzer
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


st.title(" News Sentiment Analysis")
st.caption(
    "Recent headlines for the selected ticker, scored using VADER sentiment analysis. "
    "Sentiment reflects headline tone only, not guaranteed market impact. Educational use only."
)

st.sidebar.header("News Settings")

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

max_articles = st.sidebar.slider("Max articles to fetch", 5, 50, 20)

sentiment_filter = st.sidebar.multiselect(
    "Filter by sentiment",
    options=["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"],
)

@st.cache_data(ttl=1800)
def fetch_news(ticker: str, max_items: int):
    try:
        tk = yf.Ticker(ticker)
        news = tk.news or []
    except Exception:
        news = []
    return news[:max_items]


def extract_article_fields(item: dict) -> dict:
    """Handle both legacy yfinance news schema and newer nested 'content' schema."""
    content = item.get("content", item)

    title = content.get("title") or item.get("title") or "Untitled"

    
    provider = content.get("provider")
    if isinstance(provider, dict):
        publisher = provider.get("displayName", "Unknown")
    else:
        publisher = item.get("publisher", "Unknown")

    
    link = None
    canonical = content.get("canonicalUrl")
    if isinstance(canonical, dict):
        link = canonical.get("url")
    if not link:
        click_through = content.get("clickThroughUrl")
        if isinstance(click_through, dict):
            link = click_through.get("url")
    if not link:
        link = item.get("link", "")

    
    pub_date = content.get("pubDate") or item.get("providerPublishTime")
    if isinstance(pub_date, str):
        try:
            pub_dt = pd.to_datetime(pub_date)
        except Exception:
            pub_dt = None
    elif isinstance(pub_date, (int, float)):
        try:
            pub_dt = datetime.fromtimestamp(pub_date)
        except Exception:
            pub_dt = None
    else:
        pub_dt = None

    
    summary = content.get("summary") or content.get("description") or ""

    return {
        "title": title,
        "publisher": publisher,
        "link": link or "",
        "published": pub_dt,
        "summary": summary,
    }


def classify_sentiment(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


raw_news = fetch_news(ticker_input, max_articles)

if not raw_news:
    st.warning(
        f"No news articles found for '{ticker_input}'. "
        "Yahoo Finance news availability varies by ticker and region."
    )
else:
    analyzer = get_sentiment_analyzer()

    rows = []
    for item in raw_news:
        fields = extract_article_fields(item)
        text_for_sentiment = fields["title"]
        if fields["summary"]:
            text_for_sentiment += ". " + fields["summary"]

        scores = analyzer.polarity_scores(text_for_sentiment)
        compound = scores["compound"]
        label = classify_sentiment(compound)

        rows.append({
            "Title": fields["title"],
            "Publisher": fields["publisher"],
            "Link": fields["link"],
            "Published": fields["published"],
            "Compound Score": compound,
            "Positive": scores["pos"],
            "Negative": scores["neg"],
            "Neutral": scores["neu"],
            "Sentiment": label,
        })

    news_df = pd.DataFrame(rows)
    news_df = news_df.sort_values("Published", ascending=False, na_position="last").reset_index(drop=True)

    st.subheader(f"Sentiment Overview: {ticker_input}")

    avg_compound = news_df["Compound Score"].mean()
    pos_count = (news_df["Sentiment"] == "Positive").sum()
    neg_count = (news_df["Sentiment"] == "Negative").sum()
    neu_count = (news_df["Sentiment"] == "Neutral").sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Articles Analyzed", f"{len(news_df)}")
    m2.metric("Avg Sentiment Score", f"{avg_compound:.3f}")
    m3.metric("Positive / Negative", f"{pos_count} / {neg_count}")

    overall_label = classify_sentiment(avg_compound)
    m4.metric("Overall Tone", overall_label)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Sentiment Distribution")
        dist_fig = go.Figure(data=[go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[pos_count, neu_count, neg_count],
            hole=0.45,
            marker=dict(colors=[ACCENT_TEAL, ACCENT_VIOLET, ACCENT_PINK]),
        )])
        dist_fig.update_layout(
    height=350,
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=True,
    legend=dict(
        font=dict(
            color="white",
            size=14
        ),
        bgcolor="rgba(0,0,0,0)"
    )
)
        dist_fig = style_fig(dist_fig)
        st.plotly_chart(dist_fig, use_container_width=True)

    with col_b:
        st.subheader("Sentiment Score by Article")
        bar_colors = news_df["Compound Score"].apply(
            lambda x: ACCENT_TEAL if x >= 0.05 else (ACCENT_PINK if x <= -0.05 else ACCENT_VIOLET)
        )
        bar_fig = go.Figure(data=[go.Bar(
            x=list(range(1, len(news_df) + 1)),
            y=news_df["Compound Score"],
            marker_color=bar_colors,
        )])
        bar_fig.update_layout(
            height=350,
            xaxis_title="Article # (most recent first)",
            yaxis_title="Compound Sentiment Score",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        bar_fig = style_fig(bar_fig)
        st.plotly_chart(bar_fig, use_container_width=True)

    dated = news_df.dropna(subset=["Published"])
    if len(dated) >= 3:
        st.subheader("Sentiment Trend Over Time")
        dated_sorted = dated.sort_values("Published")
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=dated_sorted["Published"], y=dated_sorted["Compound Score"],
            mode="lines+markers", name="Sentiment Score",
            line=dict(color=ACCENT_GOLD, width=2),
            marker=dict(size=7),
        ))
        trend_fig.add_hline(y=0, line_dash="dash", line_color=PLOTLY_GRID)
        trend_fig.update_layout(
            height=350,
            yaxis_title="Compound Sentiment Score",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        trend_fig = style_fig(trend_fig)
        st.plotly_chart(trend_fig, use_container_width=True)

    st.subheader("Recent Headlines")

    filtered_df = news_df[news_df["Sentiment"].isin(sentiment_filter)]

    if filtered_df.empty:
        st.info("No articles match the selected sentiment filters.")
    else:
        badge_class = {
            "Positive": "badge-positive",
            "Negative": "badge-negative",
            "Neutral": "badge-neutral",
        }

        for _, row in filtered_df.iterrows():
            pub_str = row["Published"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["Published"]) else "Date unknown"
            badge = badge_class.get(row["Sentiment"], "badge-neutral")
            link_html = f'<a href="{row["Link"]}" target="_blank">{row["Title"]}</a>' if row["Link"] else row["Title"]

            st.markdown(
                f"""
                <div class="news-card">
                    <span class="sentiment-badge {badge}">{row['Sentiment']} ({row['Compound Score']:.2f})</span>
                    <span style="color:#9d8fc2; font-size:0.85em;">{row['Publisher']} • {pub_str}</span><br>
                    {link_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("🧠 Auto-Generated Insights")

    insights = []

    if overall_label == "Positive":
        insights.append(
            f"Overall news sentiment for {ticker_input} is positive "
            f"(average score {avg_compound:.2f}), suggesting recent coverage leans favorable."
        )
    elif overall_label == "Negative":
        insights.append(
            f"Overall news sentiment for {ticker_input} is negative "
            f"(average score {avg_compound:.2f}), suggesting recent coverage leans unfavorable."
        )
    else:
        insights.append(
            f"Overall news sentiment for {ticker_input} is roughly neutral "
            f"(average score {avg_compound:.2f}), with no strong directional tone in headlines."
        )

    total = len(news_df)
    insights.append(
        f"Of {total} articles analyzed, {pos_count} ({pos_count/total*100:.0f}%) were positive, "
        f"{neu_count} ({neu_count/total*100:.0f}%) neutral, and {neg_count} ({neg_count/total*100:.0f}%) negative."
    )

    if not news_df.empty:
        most_positive = news_df.loc[news_df["Compound Score"].idxmax()]
        most_negative = news_df.loc[news_df["Compound Score"].idxmin()]
        if most_positive["Compound Score"] > 0:
            insights.append(f"Most positive headline: \"{most_positive['Title']}\" (score {most_positive['Compound Score']:.2f}).")
        if most_negative["Compound Score"] < 0:
            insights.append(f"Most negative headline: \"{most_negative['Title']}\" (score {most_negative['Compound Score']:.2f}).")

    for insight in insights:
        st.markdown(f"- {insight}")

    with st.expander("Export Sentiment Data"):
        export_df = news_df.drop(columns=["Link"]).copy()
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download sentiment data as CSV",
            data=csv_data,
            file_name=f"{ticker_input}_news_sentiment.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "⚠️ Sentiment scores are derived from headline/summary text using a general-purpose lexicon "
    "(VADER) and do not represent professional financial analysis. Educational use only."
)