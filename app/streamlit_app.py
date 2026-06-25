"""
app/streamlit_app.py — FinAI Risk Analyst Dashboard

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from dotenv import load_dotenv
from sqlalchemy import select

# ── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.db import init_db, get_session, DailyRiskScore, NLPScore, Article

# ── Config ────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

@st.cache_resource
def setup_db(db_url: str):
    init_db(db_url)

cfg    = load_config()
setup_db(cfg["database"]["url"])

TICKER_MAP = {c["ticker"]: c["name"] for c in cfg["companies"]}

# ── Risk label colours ────────────────────────────────────────
LABEL_COLOR = {
    "low":      "#22c55e",
    "medium":   "#f59e0b",
    "high":     "#ef4444",
    "critical": "#7c3aed",
}

# ── Data loaders ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_risk_scores(days: int = 30) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=days)
    with get_session() as session:
        rows = session.execute(
            select(DailyRiskScore).where(DailyRiskScore.score_date >= cutoff)
        ).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "ticker":        r.ticker,
        "company":       TICKER_MAP.get(r.ticker, r.ticker),
        "score_date":    r.score_date,
        "risk_score":    r.risk_score,
        "risk_label":    r.risk_label,
        "article_count": r.article_count,
        "dominant_event":r.dominant_event,
        "model_version": r.model_version,
    } for r in rows])


@st.cache_data(ttl=300)
def load_articles_for(ticker: str, limit: int = 20) -> pd.DataFrame:
    with get_session() as session:
        scores = session.execute(
            select(NLPScore, Article)
            .join(Article, NLPScore.article_id == Article.id)
            .where(NLPScore.ticker == ticker)
            .order_by(Article.published_at.desc())
            .limit(limit)
        ).all()
    if not scores:
        return pd.DataFrame()
    rows = []
    for s, a in scores:
        rows.append({
            "headline":     a.headline,
            "published_at": a.published_at,
            "source":       a.source,
            "url":          a.url,
            "sentiment":    s.sentiment,
            "neg_score":    s.sentiment_neg,
            "primary_event":s.primary_event,
            "summary":      s.summary,
            "key_risks":    s.get_key_risks(),
        })
    return pd.DataFrame(rows)


# ── Page layout ───────────────────────────────────────────────

st.set_page_config(
    page_title="FinAI Risk Analyst",
    layout="wide",
)

st.title("FinAI Risk Analyst")
st.caption("AI-powered news risk monitoring across 100s of companies.")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    days      = st.slider("Lookback (days)", 7, 90, 30)
    min_label = st.selectbox("Minimum risk level", ["all", "medium", "high", "critical"], index=0)
    st.divider()
    st.info("Data refreshes every 5 min. Pipeline runs daily via GitHub Actions.")

df_all = load_risk_scores(days)

if df_all.empty:
    st.warning("No data found. Run `python -m src.pipeline` to populate the database.")
    st.stop()

# Get latest score per ticker for the leaderboard
df_latest = (
    df_all.sort_values("score_date", ascending=False)
    .groupby("ticker", as_index=False)
    .first()
)

if min_label != "all":
    label_order = ["low", "medium", "high", "critical"]
    min_idx     = label_order.index(min_label)
    df_latest   = df_latest[df_latest["risk_label"].apply(
        lambda l: label_order.index(l) >= min_idx
    )]

df_latest = df_latest.sort_values("risk_score", ascending=False)

# ── KPI Row ───────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Companies tracked",    len(df_all["ticker"].unique()))
col2.metric("High/Critical alerts", int((df_latest["risk_label"].isin(["high","critical"])).sum()))
col3.metric("Avg risk score",       f"{df_latest['risk_score'].mean():.1f} / 10")
col4.metric("Articles processed",   int(df_all["article_count"].sum()))

st.divider()

# ── Risk Leaderboard ──────────────────────────────────────────
st.subheader("Risk Leaderboard — Today")

def colour_label(val):
    c = LABEL_COLOR.get(val, "#6b7280")
    return f"background-color: {c}22; color: {c}; font-weight: 600;"

display_df = df_latest[[
    "ticker", "company", "risk_score", "risk_label",
    "article_count", "dominant_event", "score_date"
]].rename(columns={
    "ticker":        "Ticker",
    "company":       "Company",
    "risk_score":    "Risk Score",
    "risk_label":    "Level",
    "article_count": "Articles",
    "dominant_event":"Top Event",
    "score_date":    "As of",
})
display_df = display_df.sort_values("Risk Score", ascending=False).reset_index(drop=True)
display_df.insert(0, "Rank", range(1, len(display_df) + 1))
st.dataframe(
    display_df.style
        .applymap(colour_label, subset=["Level"])
        .format({"Risk Score": "{:.1f}"}),
    use_container_width=True,
    height=420,
    hide_index=True,
)

# ── Risk Distribution ─────────────────────────────────────────
st.subheader("Risk Score Distribution")

col_a, col_b = st.columns(2)

with col_a:
    fig_hist = px.histogram(
        df_latest, x="risk_score", nbins=20,
        color="risk_label",
        color_discrete_map=LABEL_COLOR,
        labels={"risk_score": "Risk Score", "risk_label": "Level"},
        title="Distribution of latest risk scores",
    )
    fig_hist.update_layout(showlegend=True, height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

with col_b:
    event_counts = df_all["dominant_event"].value_counts().reset_index()
    event_counts.columns = ["event", "count"]
    fig_pie = px.pie(
        event_counts, names="event", values="count",
        title="Dominant event breakdown (all companies, last 30d)",
        hole=0.4,
    )
    fig_pie.update_layout(height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Company Drill-down ────────────────────────────────────────
st.divider()
st.subheader("Company Drill-down")

selected = st.selectbox(
    "Select a company",
    options=df_latest["ticker"].tolist(),
    format_func=lambda t: f"{t}  —  {TICKER_MAP.get(t, '')}",
)

if selected:
    df_company = df_all[df_all["ticker"] == selected].sort_values("score_date")

    # Risk trend over time
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_company["score_date"],
        y=df_company["risk_score"],
        mode="lines+markers",
        name="Risk Score",
        line=dict(width=2, color="#6366f1"),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.1)",
    ))
    fig_trend.add_hline(y=5.0, line_dash="dot", line_color="#ef4444",
                        annotation_text="High threshold")
    fig_trend.update_layout(
        title=f"{selected} — Risk Score Trend",
        xaxis_title="Date", yaxis_title="Risk Score",
        yaxis=dict(range=[0, 10]),
        height=300,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Recent articles
    st.markdown(f"**Recent news for {selected}**")
    df_articles = load_articles_for(selected)

    if df_articles.empty:
        st.info("No articles yet for this company.")
    else:
        for _, row in df_articles.head(8).iterrows():
            sentiment_icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
                row.get("sentiment", ""), "⚪"
            )
            label_color = LABEL_COLOR.get("high" if (row.get("neg_score") or 0) > 0.6 else "low", "#6b7280")

            with st.expander(f"{sentiment_icon}  {row['headline'][:100]}  —  {str(row['published_at'])[:10]}"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"**Summary:** {row.get('summary', 'N/A')}")
                    if row.get("key_risks"):
                        st.markdown("**Key risks identified:**")
                        for r in row["key_risks"]:
                            st.markdown(f"- {r}")
                with c2:
                    st.markdown(f"**Sentiment:** {row.get('sentiment', 'N/A').upper()}")
                    st.markdown(f"**Event type:** {row.get('primary_event', 'N/A')}")
                    st.markdown(f"**Neg score:** {row.get('neg_score', 0):.2f}")
                    if row.get("url"):
                        st.markdown(f"[Read full article →]({row['url']})")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with FinBERT · facebook/bart-large-mnli · Claude · XGBoost  |  "
    "Data: Finnhub + Google News RSS  |  "
    "Risk scores are informational only — not investment advice."
)
