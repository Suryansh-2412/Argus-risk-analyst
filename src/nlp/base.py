"""
src/nlp/base.py — Abstract interfaces for every NLP stage.

Every model you ever add MUST subclass one of these.
The pipeline only ever calls these interfaces, so:
  - adding a model never breaks existing ones
  - swapping models requires zero changes to pipeline.py

Dataclasses are used for results so they're easy to serialize to the DB.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ── Shared enums ─────────────────────────────────────────────

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"


# ── Result types (plain dataclasses — no framework coupling) ──

@dataclass
class SentimentResult:
    sentiment:       Sentiment
    positive_score:  float
    negative_score:  float
    neutral_score:   float
    confidence:      float
    model_name:      str

    @property
    def signed_score(self) -> float:
        """Single float: positive=+1, negative=-1, scaled by confidence."""
        direction = {
            Sentiment.POSITIVE: 1.0,
            Sentiment.NEGATIVE: -1.0,
            Sentiment.NEUTRAL:   0.0,
        }[self.sentiment]
        return direction * self.confidence


@dataclass
class EventResult:
    primary_event:  str
    event_scores:   dict          # {event_type: float confidence}
    confidence:     float         # confidence of the primary event
    model_name:     str

    def events_above(self, threshold: float) -> dict:
        return {k: v for k, v in self.event_scores.items() if v >= threshold}


@dataclass
class SummaryResult:
    summary:     str
    key_risks:   List[str] = field(default_factory=list)
    model_name:  str = ""


# ── Abstract model bases ──────────────────────────────────────

class BaseSentimentModel(ABC):
    """
    Contract for all sentiment models.

    To add a new sentiment model:
      1. Create src/nlp/sentiment/your_model.py
      2. Subclass BaseSentimentModel
      3. Implement load() and score()
      4. Register in src/nlp/registry.py
      5. Add entry under models.sentiment.available in config.yaml
      6. Set models.sentiment.active to your model name

    Nothing else in the pipeline needs to change.
    """
    name: str = "base_sentiment"

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Called once at startup."""
        pass

    @abstractmethod
    def score(self, text: str) -> SentimentResult:
        """Score the sentiment of a single piece of text."""
        pass

    def score_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Score a batch of texts. Override this for GPU batching efficiency.
        Default: naive loop (fine for CPU inference).
        """
        return [self.score(t) for t in texts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BaseEventClassifier(ABC):
    """
    Contract for all event classification models.
    The labels to classify against come from config (event_taxonomy),
    not hardcoded in the model — so extending the taxonomy never touches
    model code.
    """
    name: str = "base_event"

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def classify(self, text: str, labels: dict) -> EventResult:
        """
        labels: {event_key: natural_language_label_string}
        e.g. {"regulatory_legal": "regulatory investigation or legal action"}
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BaseSummarizer(ABC):
    """
    Contract for all summarization models.
    Summaries must be grounded — the model receives the source text
    and is expected to summarize from it, not from parametric memory.
    """
    name: str = "base_summarizer"

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def summarize(self, text: str, company_name: str) -> SummaryResult:
        """
        text: the raw article body (already cleaned)
        company_name: used to make the prompt more specific
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
