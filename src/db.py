"""
src/db.py — Database layer
Three tables: articles, nlp_scores, daily_risk_scores.

Design principle: append-only. We never update rows, only insert.
This gives us a full audit trail and makes the backtest trivial.
"""

from __future__ import annotations
import json
from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Text, DateTime, Date, Boolean, UniqueConstraint,
    Index
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class Article(Base):
    """One row per unique article. Dedup happens before insert."""
    __tablename__ = "articles"

    id            = Column(Integer, primary_key=True)
    ticker        = Column(String(10), nullable=False, index=True)
    company_name  = Column(String(100))
    headline      = Column(Text, nullable=False)
    body          = Column(Text)
    url           = Column(String(500))
    source        = Column(String(100))          # e.g. "finnhub", "rss"
    published_at  = Column(DateTime, nullable=False, index=True)
    fetched_at    = Column(DateTime, default=datetime.utcnow)
    content_hash  = Column(String(64), unique=True)  # SHA-256 of headline+body

    __table_args__ = (
        Index("ix_article_ticker_date", "ticker", "published_at"),
    )


class NLPScore(Base):
    """
    One row per article per pipeline run.
    Storing model_name allows us to compare old vs new model output later
    without reprocessing everything.
    """
    __tablename__ = "nlp_scores"

    id              = Column(Integer, primary_key=True)
    article_id      = Column(Integer, nullable=False, index=True)
    ticker          = Column(String(10), nullable=False, index=True)
    run_date        = Column(Date, nullable=False)

    # Sentiment
    sentiment       = Column(String(10))           # positive/negative/neutral
    sentiment_pos   = Column(Float)
    sentiment_neg   = Column(Float)
    sentiment_neu   = Column(Float)
    sentiment_model = Column(String(100))

    # Event classification
    primary_event   = Column(String(50))
    event_scores    = Column(Text)                 # JSON: {event_type: score}
    event_model     = Column(String(100))

    # Summary
    summary         = Column(Text)
    key_risks       = Column(Text)                 # JSON list of strings
    summary_model   = Column(String(100))

    processed_at    = Column(DateTime, default=datetime.utcnow)

    # Helpers for JSON fields
    def get_event_scores(self) -> dict:
        return json.loads(self.event_scores) if self.event_scores else {}

    def get_key_risks(self) -> list:
        return json.loads(self.key_risks) if self.key_risks else []


class DailyRiskScore(Base):
    """
    Aggregated risk score per company per day.
    This is what the dashboard shows and what the backtest trains on.
    """
    __tablename__ = "daily_risk_scores"

    id                  = Column(Integer, primary_key=True)
    ticker              = Column(String(10), nullable=False)
    score_date          = Column(Date, nullable=False)
    risk_score          = Column(Float, nullable=False)          # 0-10 scale
    risk_label          = Column(String(20))                     # low/medium/high/critical
    article_count       = Column(Integer, default=0)
    avg_sentiment_neg   = Column(Float)
    dominant_event      = Column(String(50))
    feature_vector      = Column(Text)                           # JSON, for v2 model
    model_version       = Column(String(10))                     # v1 or v2
    created_at          = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "score_date", name="uq_ticker_date"),
        Index("ix_risk_ticker_date", "ticker", "score_date"),
    )

    @property
    def risk_label_from_score(self) -> str:
        if self.risk_score >= 7.5:   return "critical"
        elif self.risk_score >= 5.0: return "high"
        elif self.risk_score >= 2.5: return "medium"
        else:                        return "low"


# ── Session factory ──────────────────────────────────────────

_engine = None
_SessionLocal = None


def init_db(db_url: str, echo: bool = False):
    """Call once at startup with the URL from config."""
    global _engine, _SessionLocal

    kwargs = {}
    if db_url.startswith("sqlite"):
        # Required for SQLite in multi-threaded contexts (e.g. Streamlit)
        kwargs = {"connect_args": {"check_same_thread": False},
                  "poolclass": StaticPool}

    _engine = create_engine(db_url, echo=echo, **kwargs)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def get_session() -> Session:
    if _SessionLocal is None:
        raise RuntimeError("Call init_db() before get_session()")
    return _SessionLocal()
