"""
src/nlp/registry.py — Central model registry.

How it works:
  - Each model class registers itself with @register_sentiment / @register_event / @register_summarizer
  - build_from_config() reads config.yaml, looks up the active model name,
    imports and instantiates the right class, and returns it

Adding a new model NEVER requires touching pipeline.py or any other file.
"""

from __future__ import annotations
import importlib
from typing import Type, Dict, Any

from loguru import logger

from src.nlp.base import BaseSentimentModel, BaseEventClassifier, BaseSummarizer


# ── Registries (populated by decorators below) ────────────────

_SENTIMENT_REGISTRY:   Dict[str, Type[BaseSentimentModel]]   = {}
_EVENT_REGISTRY:       Dict[str, Type[BaseEventClassifier]]  = {}
_SUMMARIZER_REGISTRY:  Dict[str, Type[BaseSummarizer]]       = {}


# ── Decorators ────────────────────────────────────────────────

def register_sentiment(name: str):
    """Decorator: @register_sentiment('finbert')"""
    def decorator(cls: Type[BaseSentimentModel]):
        _SENTIMENT_REGISTRY[name] = cls
        logger.debug(f"Registered sentiment model: {name} → {cls.__name__}")
        return cls
    return decorator


def register_event(name: str):
    """Decorator: @register_event('bart_zero_shot')"""
    def decorator(cls: Type[BaseEventClassifier]):
        _EVENT_REGISTRY[name] = cls
        logger.debug(f"Registered event classifier: {name} → {cls.__name__}")
        return cls
    return decorator


def register_summarizer(name: str):
    """Decorator: @register_summarizer('anthropic')"""
    def decorator(cls: Type[BaseSummarizer]):
        _SUMMARIZER_REGISTRY[name] = cls
        logger.debug(f"Registered summarizer: {name} → {cls.__name__}")
        return cls
    return decorator


# ── Builder ───────────────────────────────────────────────────

def _import_all_models():
    """
    Force-import all model modules so their @register_* decorators fire.
    Add new module paths here when you add a new model file.
    """
    modules = [
        "src.nlp.sentiment.finbert",
        "src.nlp.events.zero_shot",
        "src.nlp.summarizer.llm",
    ]
    for mod in modules:
        importlib.import_module(mod)


def build_sentiment_model(config: dict) -> BaseSentimentModel:
    _import_all_models()
    active = config["models"]["sentiment"]["active"]
    model_cfg = config["models"]["sentiment"]["available"][active]

    if active not in _SENTIMENT_REGISTRY:
        raise ValueError(
            f"Sentiment model '{active}' not registered. "
            f"Available: {list(_SENTIMENT_REGISTRY)}"
        )

    cls = _SENTIMENT_REGISTRY[active]
    instance = cls(model_cfg)
    logger.info(f"Built sentiment model: {active} ({cls.__name__})")
    instance.load()
    return instance


def build_event_classifier(config: dict) -> BaseEventClassifier:
    _import_all_models()
    active = config["models"]["event_classifier"]["active"]
    model_cfg = config["models"]["event_classifier"]["available"][active]

    if active not in _EVENT_REGISTRY:
        raise ValueError(
            f"Event classifier '{active}' not registered. "
            f"Available: {list(_EVENT_REGISTRY)}"
        )

    cls = _EVENT_REGISTRY[active]
    instance = cls(model_cfg)
    logger.info(f"Built event classifier: {active} ({cls.__name__})")
    instance.load()
    return instance


def build_summarizer(config: dict) -> BaseSummarizer:
    _import_all_models()
    active = config["models"]["summarizer"]["active"]
    model_cfg = config["models"]["summarizer"]["available"][active]

    if active not in _SUMMARIZER_REGISTRY:
        raise ValueError(
            f"Summarizer '{active}' not registered. "
            f"Available: {list(_SUMMARIZER_REGISTRY)}"
        )

    cls = _SUMMARIZER_REGISTRY[active]
    instance = cls(model_cfg)
    logger.info(f"Built summarizer: {active} ({cls.__name__})")
    instance.load()
    return instance
