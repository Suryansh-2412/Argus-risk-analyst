"""
src/ingest/news.py — News ingestion pipeline

Sources:
  - Finnhub company-news endpoint (per-ticker, financial focus)
  - Google News RSS (broader coverage, free)

Dedup strategy: SHA-256 hash of normalized headline+body for exact duplicates,
plus cosine similarity on sentence embeddings for syndicated variants
(same story, different outlet, slightly different wording).
"""

from __future__ import annotations
import hashlib
import os
import time
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass

import feedparser
import httpx
from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class RawArticle:
    ticker:       str
    company_name: str
    headline:     str
    body:         str
    url:          str
    source:       str
    published_at: datetime
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            raw = (self.headline + " " + self.body[:200]).lower().strip()
            self.content_hash = hashlib.sha256(raw.encode()).hexdigest()


class NewsIngester:
    def __init__(self, cfg: dict):
        self.sources          = cfg["ingest"]["sources"]
        self.lookback_days    = cfg["ingest"]["lookback_days"]
        self.max_per_company  = cfg["ingest"]["max_articles_per_company"]
        self.dedup_threshold  = cfg["ingest"]["dedup_threshold"]
        self.dedup_model_name = cfg["ingest"]["dedup_model"]
        self._dedup_model: Optional[SentenceTransformer] = None

    def _load_dedup_model(self):
        if self._dedup_model is None:
            logger.info(f"Loading dedup model: {self.dedup_model_name}")
            self._dedup_model = SentenceTransformer(self.dedup_model_name)

    # ── Finnhub ───────────────────────────────────────────────

    def _fetch_finnhub(self, ticker: str, company_name: str,
                       from_dt: datetime, to_dt: datetime) -> List[RawArticle]:
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        if not api_key:
            logger.warning("FINNHUB_API_KEY not set — skipping Finnhub source")
            return []

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from":   from_dt.strftime("%Y-%m-%d"),
            "to":     to_dt.strftime("%Y-%m-%d"),
            "token":  api_key,
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                items = resp.json()
        except Exception as e:
            logger.warning(f"Finnhub fetch failed for {ticker}: {e}")
            return []

        articles = []
        for item in items[: self.max_per_company]:
            pub = datetime.utcfromtimestamp(item.get("datetime", time.time()))
            articles.append(RawArticle(
                ticker=ticker,
                company_name=company_name,
                headline=item.get("headline", ""),
                body=item.get("summary", ""),
                url=item.get("url", ""),
                source="finnhub",
                published_at=pub,
            ))
        return articles

    # ── Google News RSS ───────────────────────────────────────

    def _fetch_rss(self, ticker: str, company_name: str,
                   from_dt: datetime) -> List[RawArticle]:
        query   = f"{company_name} stock"
        rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"

        try:
            feed = feedparser.parse(rss_url)
        except Exception as e:
            logger.warning(f"RSS fetch failed for {ticker}: {e}")
            return []

        articles = []
        for entry in feed.entries[: self.max_per_company]:
            pub_str = entry.get("published", "")
            try:
                pub = datetime(*entry.published_parsed[:6])
            except Exception:
                pub = datetime.utcnow()

            if pub < from_dt:
                continue

            articles.append(RawArticle(
                ticker=ticker,
                company_name=company_name,
                headline=entry.get("title", ""),
                body=entry.get("summary", ""),
                url=entry.get("link", ""),
                source="rss",
                published_at=pub,
            ))
        return articles

    # ── Deduplication ─────────────────────────────────────────

    def _dedup(self, articles: List[RawArticle],
               existing_hashes: set) -> List[RawArticle]:
        """
        Two-pass dedup:
          1. Exact hash match (cheap, catches same article from 2 sources)
          2. Embedding cosine similarity (catches syndicated variants)
        """
        # Pass 1: exact hash
        unique_by_hash = []
        seen_hashes    = set(existing_hashes)
        for a in articles:
            if a.content_hash not in seen_hashes:
                seen_hashes.add(a.content_hash)
                unique_by_hash.append(a)

        if len(unique_by_hash) <= 1:
            return unique_by_hash

        # Pass 2: embedding similarity
        self._load_dedup_model()
        texts      = [a.headline + " " + a.body[:200] for a in unique_by_hash]
        embeddings = self._dedup_model.encode(texts, show_progress_bar=False)

        # Normalise for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        keep = []
        kept_embeddings = []
        for i, article in enumerate(unique_by_hash):
            emb = embeddings[i]
            is_dup = False
            for kept_emb in kept_embeddings:
                sim = float(np.dot(emb, kept_emb))
                if sim >= self.dedup_threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(article)
                kept_embeddings.append(emb)

        logger.debug(f"Dedup: {len(articles)} → {len(keep)} articles")
        return keep

    # ── Public API ────────────────────────────────────────────

    def fetch(self, ticker: str, company_name: str,
              existing_hashes: set = None) -> List[RawArticle]:
        """
        Fetch news for a single company from all configured sources.
        existing_hashes: content_hash values already in DB (for dedup).
        """
        from_dt = datetime.utcnow() - timedelta(days=self.lookback_days)
        to_dt   = datetime.utcnow()
        all_articles = []

        if "finnhub" in self.sources:
            articles = self._fetch_finnhub(ticker, company_name, from_dt, to_dt)
            all_articles.extend(articles)
            logger.debug(f"{ticker}: {len(articles)} articles from Finnhub")

        if "rss" in self.sources:
            articles = self._fetch_rss(ticker, company_name, from_dt)
            all_articles.extend(articles)
            logger.debug(f"{ticker}: {len(articles)} articles from RSS")

        deduped = self._dedup(all_articles, existing_hashes or set())
        logger.info(f"{ticker}: {len(deduped)} unique articles after dedup")
        return deduped
