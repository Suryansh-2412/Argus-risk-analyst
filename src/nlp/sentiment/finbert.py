"""
src/nlp/sentiment/finbert.py — FinBERT sentiment model

Key design decisions:
  - Articles often exceed BERT's 512-token limit. We chunk them with overlap
    and aggregate scores (weighted toward the lead paragraph).
  - Batch inference for GPU efficiency.
  - Falls back gracefully to NEUTRAL on any single-article error.
"""

from __future__ import annotations
from typing import List

import torch
from loguru import logger
from transformers import pipeline, AutoTokenizer

from src.nlp.base import BaseSentimentModel, SentimentResult, Sentiment
from src.nlp.registry import register_sentiment


@register_sentiment("finbert")
class FinBERTSentiment(BaseSentimentModel):
    name = "finbert"

    def __init__(self, cfg: dict):
        self.model_name   = cfg.get("model_name", "ProsusAI/finbert")
        self.chunk_size   = cfg.get("chunk_size", 400)       # tokens
        self.chunk_overlap= cfg.get("chunk_overlap", 50)
        self.batch_size   = cfg.get("batch_size", 8)
        self._pipeline    = None
        self._tokenizer   = None

    def load(self) -> None:
        logger.info(f"Loading FinBERT from {self.model_name} ...")
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            top_k=None,                  # return all three class scores
            device=device,
            truncation=True,
            max_length=512,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"FinBERT loaded (device={'cuda' if device == 0 else 'cpu'})")

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping token chunks.
        Returns plain-text chunks (pipeline handles tokenization internally).
        """
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        step   = self.chunk_size - self.chunk_overlap
        chunks = []
        for i in range(0, max(1, len(tokens)), step):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text   = self._tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text)
            if i + self.chunk_size >= len(tokens):
                break
        return chunks or [text[:1000]]   # fallback for very short texts

    def _aggregate_scores(self, chunk_results: List[dict]) -> SentimentResult:
        """
        Weighted average across chunks.
        Weight earlier chunks more (journalism front-loads key facts).
        """
        if not chunk_results:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                positive_score=0.0, negative_score=0.0, neutral_score=1.0,
                confidence=0.0, model_name=self.name
            )

        # Build decaying weights: chunk 0 gets weight 1.0, each next gets 0.85x
        weights   = [0.85 ** i for i in range(len(chunk_results))]
        total_w   = sum(weights)

        pos = neg = neu = 0.0
        for result, w in zip(chunk_results, weights):
            label_map = {item["label"].lower(): item["score"] for item in result}
            pos += label_map.get("positive", 0.0) * w
            neg += label_map.get("negative", 0.0) * w
            neu += label_map.get("neutral",  0.0) * w

        pos, neg, neu = pos / total_w, neg / total_w, neu / total_w

        if pos >= neg and pos >= neu:
            sentiment, conf = Sentiment.POSITIVE, pos
        elif neg >= pos and neg >= neu:
            sentiment, conf = Sentiment.NEGATIVE, neg
        else:
            sentiment, conf = Sentiment.NEUTRAL, neu

        return SentimentResult(
            sentiment=sentiment,
            positive_score=round(pos, 4),
            negative_score=round(neg, 4),
            neutral_score=round(neu, 4),
            confidence=round(conf, 4),
            model_name=self.name,
        )

    def score(self, text: str) -> SentimentResult:
        try:
            chunks  = self._chunk_text(text)
            results = self._pipeline(chunks, batch_size=self.batch_size)
            return self._aggregate_scores(results)
        except Exception as e:
            logger.warning(f"FinBERT scoring failed: {e} — returning NEUTRAL")
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                positive_score=0.0, negative_score=0.0, neutral_score=1.0,
                confidence=0.0, model_name=self.name,
            )

    def score_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Efficient batch scoring — chunks all texts together, one pipeline call per batch.
        """
        return [self.score(t) for t in texts]  # chunking complicates true batching; fast enough
