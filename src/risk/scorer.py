"""
src/risk/scorer.py — Risk scoring engine

v1: Interpretable weighted formula.
    Good for: launch day, demos, explaining to non-technical interviewers.

v2: XGBoost trained on historical price reactions.
    Good for: after you've collected ~60 days of data + run the training notebook.

Switch versions by changing `risk.version` in config.yaml.
Both versions consume the same feature dict from features.py.
"""

from __future__ import annotations
import json
from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional

import numpy as np
from loguru import logger

from src.db import NLPScore, DailyRiskScore
from src.risk.features import build_features, features_to_array, DEFAULT_EVENT_TYPES


# ── Abstract base ─────────────────────────────────────────────

class BaseRiskScorer(ABC):
    @abstractmethod
    def score(self, features: dict) -> float:
        """Return a risk score on a 0–10 scale."""
        pass


# ── v1: Weighted heuristic ────────────────────────────────────

class WeightedRiskScorer(BaseRiskScorer):
    """
    Score = Σ(event_score × event_weight) × sentiment_modifier × recency_modifier + volume_bonus

    Fully interpretable — you can explain every number.
    Weights come from config so you can tune without touching code.
    """

    def __init__(self, cfg: dict, event_taxonomy: dict):
        self.sentiment_weight = cfg.get("sentiment_weight", 1.0)
        self.volume_weight    = cfg.get("volume_weight", 0.3)
        self.decay_days       = cfg.get("recency_decay_days", 7)
        self.conf_floor       = cfg.get("confidence_floor", 0.25)

        # Build {event_key: base_risk} from taxonomy config
        self.event_weights = {
            k: v["base_risk"] for k, v in event_taxonomy.items()
        }

    def score(self, features: dict) -> float:
        if features.get("article_count", 0) == 0:
            return 0.0

        # Event contribution: weighted sum of event scores × base risk
        event_score = sum(
            features.get(f"event_{et}", 0.0) * weight
            for et, weight in self.event_weights.items()
            if features.get(f"event_{et}", 0.0) >= self.conf_floor
        )

        # Sentiment modifier: negative sentiment amplifies, positive dampens
        neg  = features.get("sentiment_neg_weighted", 0.0)
        pos  = features.get("sentiment_pos_weighted", 0.0)
        sentiment_mod = 1.0 + (neg - pos) * self.sentiment_weight

        # Recency modifier: recent news matters more
        recency = features.get("recency_mean", 0.5)

        # Volume bonus: more articles = more signal
        volume_bonus = features.get("volume_log", 0.0) * self.volume_weight

        raw = (event_score * sentiment_mod * recency) + volume_bonus

        # Clip to 0–10
        return round(min(10.0, max(0.0, raw)), 2)


# ── v2: XGBoost model ─────────────────────────────────────────

class XGBoostRiskScorer(BaseRiskScorer):
    """
    Trained on historical data: features → did the stock drop >2% in 3 days?
    Outputs a calibrated probability, scaled to 0–10.

    Train this via notebooks/train_risk_model.ipynb.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model     = None
        self._load()

    def _load(self):
        try:
            import joblib
            self._model = joblib.load(self.model_path)
            logger.info(f"XGBoost risk model loaded from {self.model_path}")
        except FileNotFoundError:
            logger.warning(
                f"v2 model not found at {self.model_path}. "
                "Run notebooks/train_risk_model.ipynb first. "
                "Falling back to v1 score=0."
            )
            self._model = None

    def score(self, features: dict) -> float:
        if self._model is None:
            return 0.0
        arr = features_to_array(features).reshape(1, -1)
        prob = float(self._model.predict_proba(arr)[0][1])   # P(negative event)
        return round(prob * 10.0, 2)


# ── Public facade ─────────────────────────────────────────────

class RiskEngine:
    """
    Single entry point for the pipeline.
    Reads config to pick v1 vs v2, builds features, returns a DailyRiskScore.
    """

    LABELS = {
        (0.0, 2.5):  "low",
        (2.5, 5.0):  "medium",
        (5.0, 7.5):  "high",
        (7.5, 10.0): "critical",
    }

    def __init__(self, config: dict):
        version      = config["risk"]["version"]
        event_tax    = config["event_taxonomy"]
        decay_days   = config["risk"]["v1"]["recency_decay_days"]

        self.version      = version
        self.event_types  = list(event_tax.keys())
        self.decay_days   = decay_days

        if version == "v1":
            self._scorer = WeightedRiskScorer(config["risk"]["v1"], event_tax)
        elif version == "v2":
            self._scorer = XGBoostRiskScorer(config["risk"]["v2"]["model_path"])
        else:
            raise ValueError(f"Unknown risk version: {version}")

        logger.info(f"RiskEngine using {version} scorer")

    def _label(self, score: float) -> str:
        for (lo, hi), label in self.LABELS.items():
            if lo <= score < hi:
                return label
        return "critical"

    def compute(
        self,
        ticker: str,
        nlp_scores: List[NLPScore],
        score_date: date,
    ) -> DailyRiskScore:
        """
        Build features from NLP scores and return a DailyRiskScore object
        ready to insert into the DB.
        """
        feat = build_features(
            nlp_scores,
            as_of=score_date,
            event_types=self.event_types,
            decay_days=self.decay_days,
        )

        risk_score = self._scorer.score(feat)
        label      = self._label(risk_score)

        # Dominant event for dashboard display
        event_scores = {
            et: feat.get(f"event_{et}", 0.0) for et in self.event_types
        }
        dominant_event = max(event_scores, key=event_scores.get) if event_scores else "general_news"

        return DailyRiskScore(
            ticker=ticker,
            score_date=score_date,
            risk_score=risk_score,
            risk_label=label,
            article_count=feat.get("article_count", 0),
            avg_sentiment_neg=feat.get("sentiment_neg_weighted"),
            dominant_event=dominant_event,
            feature_vector=json.dumps(feat),
            model_version=self.version,
        )
