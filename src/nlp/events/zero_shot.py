"""
src/nlp/events/zero_shot.py — Zero-shot event classifier

Uses BART-large-MNLI (or any MNLI model) with the labels coming from config.
No taxonomy changes require model changes — just update config.yaml.

Multi-label: one article can trigger multiple event types (e.g. an article
about a CEO resigning amid a regulatory probe fires both leadership_change
AND regulatory_legal).
"""

from __future__ import annotations
from typing import List

import torch
from loguru import logger
from transformers import pipeline

from src.nlp.base import BaseEventClassifier, EventResult
from src.nlp.registry import register_event


@register_event("bart_zero_shot")
class ZeroShotEventClassifier(BaseEventClassifier):
    name = "bart_zero_shot"

    def __init__(self, cfg: dict):
        self.model_name  = cfg.get("model_name", "facebook/bart-large-mnli")
        self.threshold   = cfg.get("confidence_threshold", 0.25)
        self.multi_label = cfg.get("multi_label", True)
        self._pipeline   = None

    def load(self) -> None:
        logger.info(f"Loading zero-shot classifier from {self.model_name} ...")
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device,
        )
        logger.info("Zero-shot classifier loaded.")

    def classify(self, text: str, labels: dict) -> EventResult:
        """
        labels: {event_key: natural_language_label_string} from config.yaml
        e.g. {"regulatory_legal": "regulatory investigation or legal action"}
        """
        try:
            # Use the natural-language descriptions as hypothesis labels
            label_strings = list(labels.values())
            label_keys    = list(labels.keys())

            # Truncate text — BART handles up to ~1024 tokens but long texts slow it down
            text_input = text[:2000]

            result = self._pipeline(
                text_input,
                candidate_labels=label_strings,
                multi_label=self.multi_label,
            )

            # Map scores back to config keys
            scores_by_key = {}
            for label_str, score in zip(result["labels"], result["scores"]):
                idx = label_strings.index(label_str)
                key = label_keys[idx]
                scores_by_key[key] = round(float(score), 4)

            # Primary event = highest scoring key
            primary = max(scores_by_key, key=scores_by_key.get)

            return EventResult(
                primary_event=primary,
                event_scores=scores_by_key,
                confidence=scores_by_key[primary],
                model_name=self.name,
            )

        except Exception as e:
            logger.warning(f"Zero-shot classification failed: {e} — returning general_news")
            fallback_scores = {k: 0.0 for k in labels}
            fallback_scores["general_news"] = 1.0
            return EventResult(
                primary_event="general_news",
                event_scores=fallback_scores,
                confidence=1.0,
                model_name=self.name,
            )
