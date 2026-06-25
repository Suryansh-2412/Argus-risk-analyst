"""
src/risk/features.py — Feature engineering for the risk model

Takes raw NLP scores from the DB and produces a flat feature vector
that both v1 (weighted) and v2 (XGBoost) consume.

Keeping feature engineering in one place means:
  - v1 and v2 always see the same inputs
  - adding a new feature touches exactly one file
"""

from __future__ import annotations
import json
import math
from datetime import date, datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from src.db import NLPScore


# Event types we track — pulled from config at call time but defaulted here
DEFAULT_EVENT_TYPES = [
    "regulatory_legal",
    "fraud_accounting",
    "leadership_change",
    "layoffs_restructuring",
    "ma_acquisition",
    "earnings_surprise",
    "product_supply_chain",
    "macro_exposure",
    "general_news",
]


def _recency_weight(published_at: datetime, as_of: date, decay_days: int) -> float:
    """
    Exponential decay: article from today = weight 1.0,
    article from `decay_days` days ago ≈ weight 0.5.
    """
    delta = (as_of - published_at.date()).days
    delta = max(0, delta)
    return math.exp(-math.log(2) * delta / decay_days)


def build_features(
    scores: List[NLPScore],
    as_of: date,
    event_types: Optional[List[str]] = None,
    decay_days: int = 7,
) -> dict:
    """
    Build a flat feature dict from a list of NLPScore rows for one company.

    Returns a dictionary with:
      - sentiment_* : weighted sentiment stats
      - event_* : weighted event type scores
      - volume_* : article count features
      - recency : how recent the news is on average
    """
    event_types = event_types or DEFAULT_EVENT_TYPES

    if not scores:
        # Return zero vector so risk scorer handles missing data cleanly
        feat = {"article_count": 0, "recency_mean": 0.0}
        for et in event_types:
            feat[f"event_{et}"] = 0.0
        for key in ["sentiment_neg_weighted", "sentiment_pos_weighted",
                    "sentiment_neg_mean", "neg_article_fraction",
                    "volume_log", "max_event_score"]:
            feat[key] = 0.0
        return feat

    weights = [
        _recency_weight(s.processed_at or datetime.utcnow(), as_of, decay_days)
        for s in scores
    ]
    total_w = sum(weights) or 1.0

    # ── Sentiment features ────────────────────────────────────
    neg_weighted = sum(
        (s.sentiment_neg or 0.0) * w for s, w in zip(scores, weights)
    ) / total_w

    pos_weighted = sum(
        (s.sentiment_pos or 0.0) * w for s, w in zip(scores, weights)
    ) / total_w

    neg_mean = np.mean([s.sentiment_neg or 0.0 for s in scores])

    neg_count = sum(1 for s in scores if (s.sentiment or "") == "negative")
    neg_fraction = neg_count / len(scores)

    # ── Event features ────────────────────────────────────────
    event_feats = {et: 0.0 for et in event_types}
    for s, w in zip(scores, weights):
        ev_scores = s.get_event_scores()
        for et in event_types:
            event_feats[et] += ev_scores.get(et, 0.0) * w

    # Normalise by total weight
    for et in event_types:
        event_feats[et] = round(event_feats[et] / total_w, 4)

    max_event_score = max(event_feats.values()) if event_feats else 0.0

    # ── Volume features ───────────────────────────────────────
    volume_log = math.log1p(len(scores))   # log(1 + n) to dampen extremes

    # ── Recency feature ───────────────────────────────────────
    recency_mean = np.mean(weights)

    feat = {
        "article_count":         len(scores),
        "volume_log":            round(volume_log, 4),
        "sentiment_neg_weighted": round(neg_weighted, 4),
        "sentiment_pos_weighted": round(pos_weighted, 4),
        "sentiment_neg_mean":    round(float(neg_mean), 4),
        "neg_article_fraction":  round(neg_fraction, 4),
        "recency_mean":          round(float(recency_mean), 4),
        "max_event_score":       round(max_event_score, 4),
    }
    for et in event_types:
        feat[f"event_{et}"] = event_feats[et]

    return feat


def features_to_array(feat: dict, event_types: Optional[List[str]] = None) -> np.ndarray:
    """
    Convert feature dict to a numpy array in a stable column order.
    Used for XGBoost inference.
    """
    event_types = event_types or DEFAULT_EVENT_TYPES
    cols = [
        "article_count", "volume_log",
        "sentiment_neg_weighted", "sentiment_pos_weighted",
        "sentiment_neg_mean", "neg_article_fraction",
        "recency_mean", "max_event_score",
    ] + [f"event_{et}" for et in event_types]
    return np.array([feat.get(c, 0.0) for c in cols], dtype=np.float32)


def column_names(event_types: Optional[List[str]] = None) -> List[str]:
    """Return stable column name list — used when building training DataFrames."""
    event_types = event_types or DEFAULT_EVENT_TYPES
    return [
        "article_count", "volume_log",
        "sentiment_neg_weighted", "sentiment_pos_weighted",
        "sentiment_neg_mean", "neg_article_fraction",
        "recency_mean", "max_event_score",
    ] + [f"event_{et}" for et in event_types]
