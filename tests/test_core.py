"""
tests/test_core.py — Core unit tests

Run with: pytest tests/ -v
"""

from __future__ import annotations
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.nlp.base import (
    BaseSentimentModel, BaseEventClassifier, BaseSummarizer,
    Sentiment, SentimentResult, EventResult, SummaryResult,
)
from src.risk.features import build_features, features_to_array, column_names


# ── NLP base contract tests ───────────────────────────────────

def make_sentiment_result(s=Sentiment.NEGATIVE, pos=0.1, neg=0.8, neu=0.1, conf=0.8):
    return SentimentResult(
        sentiment=s, positive_score=pos, negative_score=neg,
        neutral_score=neu, confidence=conf, model_name="test"
    )


class TestSentimentResult:
    def test_signed_score_negative(self):
        r = make_sentiment_result(s=Sentiment.NEGATIVE, conf=0.8)
        assert r.signed_score == pytest.approx(-0.8)

    def test_signed_score_positive(self):
        r = make_sentiment_result(s=Sentiment.POSITIVE, conf=0.9)
        assert r.signed_score == pytest.approx(0.9)

    def test_signed_score_neutral(self):
        r = make_sentiment_result(s=Sentiment.NEUTRAL, conf=0.5)
        assert r.signed_score == pytest.approx(0.0)


class TestEventResult:
    def test_events_above_threshold(self):
        ev = EventResult(
            primary_event="regulatory_legal",
            event_scores={"regulatory_legal": 0.8, "fraud_accounting": 0.15, "general_news": 0.05},
            confidence=0.8,
            model_name="test",
        )
        filtered = ev.events_above(0.25)
        assert "regulatory_legal" in filtered
        assert "fraud_accounting" not in filtered


# ── Feature engineering tests ─────────────────────────────────

class MockNLPScore:
    """Minimal mock of NLPScore for feature tests."""
    def __init__(self, sentiment="negative", neg=0.8, pos=0.1, event_scores=None, dt=None):
        self.sentiment        = sentiment
        self.sentiment_neg    = neg
        self.sentiment_pos    = pos
        self.sentiment_neu    = 1 - neg - pos
        self.processed_at     = dt or datetime.utcnow()
        self._event_scores    = event_scores or {"regulatory_legal": 0.7, "general_news": 0.3}

    def get_event_scores(self):
        return self._event_scores


class TestFeatureEngineering:
    def test_empty_scores_returns_zeros(self):
        feat = build_features([], as_of=date.today())
        assert feat["article_count"] == 0
        assert feat["sentiment_neg_weighted"] == 0.0

    def test_single_negative_article(self):
        scores = [MockNLPScore(sentiment="negative", neg=0.9)]
        feat   = build_features(scores, as_of=date.today())
        assert feat["article_count"] == 1
        assert feat["sentiment_neg_weighted"] > 0.5
        assert feat["event_regulatory_legal"] > 0.5

    def test_feature_array_shape(self):
        scores = [MockNLPScore()]
        feat   = build_features(scores, as_of=date.today())
        arr    = features_to_array(feat)
        assert arr.shape == (len(column_names()),)

    def test_volume_log_increases_with_articles(self):
        scores_1 = [MockNLPScore()]
        scores_5 = [MockNLPScore() for _ in range(5)]
        f1 = build_features(scores_1, as_of=date.today())
        f5 = build_features(scores_5, as_of=date.today())
        assert f5["volume_log"] > f1["volume_log"]


# ── Risk scorer tests ─────────────────────────────────────────

class TestWeightedRiskScorer:
    def _make_scorer(self):
        from src.risk.scorer import WeightedRiskScorer
        cfg = {
            "sentiment_weight": 1.0,
            "volume_weight":    0.3,
            "recency_decay_days": 7,
            "confidence_floor": 0.25,
        }
        taxonomy = {
            "regulatory_legal": {"base_risk": 4.0},
            "general_news":     {"base_risk": 0.5},
        }
        return WeightedRiskScorer(cfg, taxonomy)

    def test_zero_articles_returns_zero(self):
        scorer = self._make_scorer()
        feat = build_features([], as_of=date.today())
        assert scorer.score(feat) == 0.0

    def test_high_risk_event_scores_high(self):
        scorer = self._make_scorer()
        scores = [MockNLPScore(
            sentiment="negative", neg=0.9,
            event_scores={"regulatory_legal": 0.9, "general_news": 0.1}
        )]
        feat = build_features(scores, as_of=date.today())
        risk = scorer.score(feat)
        assert risk > 3.0, f"Expected high risk, got {risk}"

    def test_score_clipped_to_10(self):
        scorer = self._make_scorer()
        # Simulate extreme inputs
        feat = {
            "article_count":          100,
            "volume_log":             4.6,
            "sentiment_neg_weighted": 0.99,
            "sentiment_pos_weighted": 0.01,
            "sentiment_neg_mean":     0.99,
            "neg_article_fraction":   1.0,
            "recency_mean":           1.0,
            "max_event_score":        0.95,
            "event_regulatory_legal": 0.95,
            "event_general_news":     0.05,
        }
        assert scorer.score(feat) <= 10.0


# ── Registry tests ────────────────────────────────────────────

class TestRegistry:
    def test_registered_models_importable(self):
        """Smoke test: importing registry loads all model modules without error."""
        from src.nlp import registry
        registry._import_all_models()
        assert "finbert"       in registry._SENTIMENT_REGISTRY
        assert "bart_zero_shot" in registry._EVENT_REGISTRY
        assert "anthropic"     in registry._SUMMARIZER_REGISTRY
