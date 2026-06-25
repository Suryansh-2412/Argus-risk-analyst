"""
src/pipeline.py — Daily pipeline orchestrator

Run this file to execute one full pipeline cycle:
  1. Fetch news for all companies
  2. Run NLP (sentiment + events + summary) on new articles
  3. Compute daily risk scores
  4. Persist everything to DB

Design principles:
  - Per-company error isolation: one company failing never stops others
  - Idempotent: re-running on the same day updates existing scores (upsert)
  - All model instances created once at startup and reused
"""

from __future__ import annotations
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy import select

# ── Bootstrap ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from src.db import init_db, get_session, Article, NLPScore, DailyRiskScore
from src.ingest.news import NewsIngester
from src.nlp.registry import build_sentiment_model, build_event_classifier, build_summarizer
from src.risk.scorer import RiskEngine


def load_config(path: str = None) -> dict:
    config_path = path or ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _configure_logging(cfg: dict):
    logger.remove()
    level = cfg.get("logging", {}).get("level", "INFO")
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    log_file = ROOT / cfg.get("logging", {}).get("file", "data/pipeline.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), level="DEBUG", rotation="10 MB", retention="30 days")


class Pipeline:
    def __init__(self, config: dict):
        self.config    = config
        self.companies = config["companies"]

        # Build all models once — expensive, reuse for all companies
        logger.info("Loading NLP models...")
        self.sentiment_model   = build_sentiment_model(config)
        self.event_classifier  = build_event_classifier(config)
        self.summarizer        = build_summarizer(config)
        self.risk_engine       = RiskEngine(config)
        self.ingester          = NewsIngester(config)
        logger.info("All models loaded. Pipeline ready.")

    # ── Stage 1: Ingest ───────────────────────────────────────

    def _get_existing_hashes(self, ticker: str, session) -> set:
        rows = session.execute(
            select(Article.content_hash).where(Article.ticker == ticker)
        ).scalars().all()
        return set(rows)

    def _save_articles(self, articles, session) -> List[Article]:
        saved = []
        for raw in articles:
            exists = session.execute(
                select(Article).where(Article.content_hash == raw.content_hash)
            ).scalar_one_or_none()
            if exists:
                continue
            art = Article(
                ticker=raw.ticker,
                company_name=raw.company_name,
                headline=raw.headline,
                body=raw.body,
                url=raw.url,
                source=raw.source,
                published_at=raw.published_at,
                content_hash=raw.content_hash,
            )
            session.add(art)
            saved.append(art)
        session.flush()   # assign IDs before NLP stage
        return saved

    # ── Stage 2: NLP ──────────────────────────────────────────

    def _run_nlp(self, article: Article, run_date: date,
                 session) -> Optional[NLPScore]:
        """Run all three NLP stages on one article. Returns None on total failure."""
        text = f"{article.headline}. {article.body or ''}".strip()
        if not text or text == ".":
            return None

        labels = {k: v["label"] for k, v in self.config["event_taxonomy"].items()}

        try:
            sent    = self.sentiment_model.score(text)
        except Exception as e:
            logger.warning(f"Sentiment failed for article {article.id}: {e}")
            sent = None

        try:
            event   = self.event_classifier.classify(text, labels)
        except Exception as e:
            logger.warning(f"Event classification failed for article {article.id}: {e}")
            event = None

        try:
            summary = self.summarizer.summarize(text, article.company_name or article.ticker)
        except Exception as e:
            logger.warning(f"Summarization failed for article {article.id}: {e}")
            summary = None

        score = NLPScore(
            article_id=article.id,
            ticker=article.ticker,
            run_date=run_date,
            # Sentiment
            sentiment=sent.sentiment.value  if sent    else None,
            sentiment_pos=sent.positive_score         if sent    else None,
            sentiment_neg=sent.negative_score         if sent    else None,
            sentiment_neu=sent.neutral_score          if sent    else None,
            sentiment_model=sent.model_name           if sent    else None,
            # Events
            primary_event=event.primary_event         if event   else None,
            event_scores=json.dumps(event.event_scores) if event else None,
            event_model=event.model_name              if event   else None,
            # Summary
            summary=summary.summary                   if summary else None,
            key_risks=json.dumps(summary.key_risks)   if summary else None,
            summary_model=summary.model_name          if summary else None,
        )
        session.add(score)
        return score

    # ── Stage 3: Risk ─────────────────────────────────────────

    def _compute_risk(self, ticker: str, run_date: date, session):
        # Pull today's NLP scores for this company
        nlp_scores = session.execute(
            select(NLPScore).where(
                NLPScore.ticker  == ticker,
                NLPScore.run_date == run_date,
            )
        ).scalars().all()

        if not nlp_scores:
            logger.debug(f"{ticker}: no NLP scores for {run_date}, skipping risk.")
            return

        daily = self.risk_engine.compute(ticker, nlp_scores, run_date)

        # Upsert: if a score for this ticker+date exists, replace it
        existing = session.execute(
            select(DailyRiskScore).where(
                DailyRiskScore.ticker     == ticker,
                DailyRiskScore.score_date == run_date,
            )
        ).scalar_one_or_none()

        if existing:
            existing.risk_score       = daily.risk_score
            existing.risk_label       = daily.risk_label
            existing.article_count    = daily.article_count
            existing.avg_sentiment_neg= daily.avg_sentiment_neg
            existing.dominant_event   = daily.dominant_event
            existing.feature_vector   = daily.feature_vector
            existing.model_version    = daily.model_version
            existing.created_at       = datetime.utcnow()
        else:
            session.add(daily)

        logger.info(
            f"{ticker}: risk={daily.risk_score:.1f} ({daily.risk_label}) "
            f"| {daily.article_count} articles | event={daily.dominant_event}"
        )

    # ── Main run ──────────────────────────────────────────────

    def run(self, run_date: Optional[date] = None):
        run_date = run_date or date.today()
        logger.info(f"=== Pipeline run for {run_date} ===")

        with get_session() as session:
            for company in self.companies:
                ticker = company["ticker"]
                name   = company["name"]
                logger.info(f"Processing {ticker} ({name})")

                try:
                    # Stage 1: Ingest
                    existing_hashes = self._get_existing_hashes(ticker, session)
                    raw_articles    = self.ingester.fetch(ticker, name, existing_hashes)
                    new_articles    = self._save_articles(raw_articles, session)
                    logger.info(f"{ticker}: {len(new_articles)} new articles saved")

                    # Stage 2: NLP on new articles only
                    for art in new_articles:
                        self._run_nlp(art, run_date, session)

                    session.flush()

                    # Stage 3: Risk score
                    self._compute_risk(ticker, run_date, session)

                except Exception as e:
                    # Isolate: one company failing never kills the loop
                    logger.error(f"{ticker}: pipeline failed — {e}", exc_info=True)
                    session.rollback()
                    continue

            session.commit()

        logger.info(f"=== Pipeline complete for {run_date} ===")


# ── Entrypoint ────────────────────────────────────────────────

def main():
    cfg = load_config()
    _configure_logging(cfg)
    init_db(cfg["database"]["url"])

    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
