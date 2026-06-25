"""
src/nlp/summarizer/llm.py — Summarizers (all free options)

Priority order for a student project:
  1. flan_t5     — runs 100% locally, zero cost, no API key needed (DEFAULT)
  2. groq        — free API tier, Llama 3.3 70B, fast (get key at console.groq.com)
  3. bart_local  — facebook/bart-large-cnn, local, pure extractive summarization
  4. anthropic   — kept for reference, requires paid key
  5. openai      — kept for reference, requires paid key

Switch via config.yaml: models.summarizer.active
"""

from __future__ import annotations
import json
import os
import re
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.nlp.base import BaseSummarizer, SummaryResult
from src.nlp.registry import register_summarizer


# ── Shared helpers ────────────────────────────────────────────

def _parse_json_response(raw: str, model_name: str) -> SummaryResult:
    """Parse JSON response; degrade gracefully on malformed output."""
    try:
        clean = raw.strip()
        # Strip markdown code fences some models add
        clean = re.sub(r"^```(?:json)?", "", clean).rstrip("```").strip()
        data  = json.loads(clean)
        return SummaryResult(
            summary=data.get("summary", "").strip(),
            key_risks=data.get("key_risks", []),
            model_name=model_name,
        )
    except Exception:
        # If JSON fails, use raw text as summary — never crash
        return SummaryResult(
            summary=raw[:400].strip(),
            key_risks=[],
            model_name=model_name,
        )


INSTRUCTION = (
    "You are a financial risk analyst. "
    "Read the article below and respond ONLY with valid JSON — no prose, no code fences. "
    "Format: {{\"summary\": \"2-3 sentence factual summary\", "
    "\"key_risks\": [\"risk1\", \"risk2\"]}}. "
    "Base your answer ONLY on the article text. "
    "If no specific risks exist, set key_risks to [].\n\n"
    "ARTICLE:\n{text}\n\nJSON:"
)


# ── 1. FLAN-T5 (local, completely free) ──────────────────────

@register_summarizer("flan_t5")
class FLANT5Summarizer(BaseSummarizer):
    """
    google/flan-t5-large — instruction-tuned T5, runs on CPU.
    No API key. No cost. ~3GB RAM for large, ~1GB for base.
    Uses direct AutoModel loading to avoid pipeline task-name issues
    across different transformers versions.
    Falls back to flan-t5-base if large OOMs.
    """
    name = "flan_t5"

    def __init__(self, cfg: dict):
        self.model_name     = cfg.get("model_name", "google/flan-t5-large")
        self.max_new_tokens = cfg.get("max_new_tokens", 300)
        self._model         = None
        self._tokenizer     = None
        self._device        = None

    def _load_model(self, model_name: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def load(self) -> None:
        logger.info(f"Loading {self.model_name} locally ...")
        try:
            self._load_model(self.model_name)
        except Exception as e:
            fallback = "google/flan-t5-base"
            logger.warning(f"{self.model_name} failed ({e}), falling back to {fallback}")
            self._load_model(fallback)
        logger.info("FLAN-T5 loaded.")

    def summarize(self, text: str, company_name: str) -> SummaryResult:
        import torch

        prompt = (
            f"You are a financial analyst. Summarize this news article about {company_name} "
            f"in 2-3 sentences and list up to 3 financial risks as JSON with keys 'summary' "
            f"and 'key_risks'. Article: {text[:1500]}"
        )
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                )

            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return _parse_json_response(decoded, self.name)

        except Exception as e:
            logger.warning(f"FLAN-T5 summarization failed: {e}")
            sentences = text.split(". ")[:2]
            return SummaryResult(
                summary=". ".join(sentences) + ".",
                key_risks=[],
                model_name=self.name,
            )


# ── 2. Groq (free API, Llama 3.3 70B) ────────────────────────

@register_summarizer("groq")
class GroqSummarizer(BaseSummarizer):
    """
    Groq free tier — Llama 3.3 70B Instruct.
    Get a free key at: https://console.groq.com
    Free tier: 30 req/min, 14,400 req/day — more than enough.
    Set GROQ_API_KEY in your .env file.
    """
    name = "groq"

    def __init__(self, cfg: dict):
        self.model      = cfg.get("model", "llama-3.3-70b-versatile")
        self.max_tokens = cfg.get("max_tokens", 350)
        self._client    = None

    def load(self) -> None:
        try:
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
                    "and add it to your .env file."
                )
            self._client = Groq(api_key=api_key)
            logger.info(f"Groq summarizer ready (model={self.model})")
        except ImportError:
            raise ImportError("Run: pip install groq")

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
    )
    def summarize(self, text: str, company_name: str) -> SummaryResult:
        prompt = INSTRUCTION.format(text=text[:2500])
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial risk analyst. "
                            "Always respond with valid JSON only. No prose, no markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,   # low temperature for factual, consistent output
            )
            raw = response.choices[0].message.content
            return _parse_json_response(raw, self.name)
        except Exception as e:
            logger.warning(f"Groq summarizer failed: {e}")
            return SummaryResult(
                summary="Summary unavailable.",
                key_risks=[],
                model_name=self.name,
            )


# ── 3. BART-CNN (local, pure summarization, no API) ──────────

@register_summarizer("bart_local")
class BARTSummarizer(BaseSummarizer):
    """
    facebook/bart-large-cnn — extractive summarization, local, free.
    Simpler than FLAN-T5 (no instruction following), but very reliable.
    Doesn't produce JSON naturally so we wrap the output manually.
    Good as a dead-simple fallback if FLAN-T5 is too slow on your machine.
    """
    name = "bart_local"

    def __init__(self, cfg: dict):
        self.model_name   = cfg.get("model_name", "facebook/bart-large-cnn")
        self.max_length   = cfg.get("max_length", 120)
        self.min_length   = cfg.get("min_length", 30)
        self._pipeline    = None

    def load(self) -> None:
        import torch
        from transformers import pipeline as hf_pipeline

        logger.info(f"Loading {self.model_name} locally ...")
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = hf_pipeline(
            "summarization",
            model=self.model_name,
            device=device,
        )
        logger.info("BART-CNN loaded.")

    def summarize(self, text: str, company_name: str) -> SummaryResult:
        try:
            # BART-CNN has a 1024-token limit
            truncated = text[:3000]
            out = self._pipeline(
                truncated,
                max_length=self.max_length,
                min_length=self.min_length,
                do_sample=False,
            )
            summary = out[0]["summary_text"]
            return SummaryResult(
                summary=summary,
                key_risks=[],   # BART-CNN doesn't do structured extraction
                model_name=self.name,
            )
        except Exception as e:
            logger.warning(f"BART-CNN summarization failed: {e}")
            return SummaryResult(
                summary=text[:200],
                key_risks=[],
                model_name=self.name,
            )


# ── 4 & 5. Paid options (kept for reference, optional) ───────

@register_summarizer("anthropic")
class AnthropicSummarizer(BaseSummarizer):
    """Requires paid ANTHROPIC_API_KEY. Use flan_t5 or groq instead."""
    name = "anthropic"

    def __init__(self, cfg: dict):
        self.model      = cfg.get("model", "claude-haiku-4-5-20251001")
        self.max_tokens = cfg.get("max_tokens", 350)
        self._client    = None

    def load(self) -> None:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set. Switch to flan_t5 or groq in config.yaml.")
        self._client = anthropic.Anthropic(api_key=key)
        logger.info(f"Anthropic summarizer ready (model={self.model})")

    def summarize(self, text: str, company_name: str) -> SummaryResult:
        prompt = INSTRUCTION.format(text=text[:2500])
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_json_response(response.content[0].text, self.name)
        except Exception as e:
            logger.warning(f"Anthropic summarizer failed: {e}")
            return SummaryResult(summary="Summary unavailable.", key_risks=[], model_name=self.name)


@register_summarizer("openai")
class OpenAISummarizer(BaseSummarizer):
    """Requires paid OPENAI_API_KEY. Use flan_t5 or groq instead."""
    name = "openai"

    def __init__(self, cfg: dict):
        self.model      = cfg.get("model", "gpt-4o-mini")
        self.max_tokens = cfg.get("max_tokens", 350)
        self._client    = None

    def load(self) -> None:
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set. Switch to flan_t5 or groq in config.yaml.")
        self._client = OpenAI(api_key=key)
        logger.info(f"OpenAI summarizer ready (model={self.model})")

    def summarize(self, text: str, company_name: str) -> SummaryResult:
        prompt = INSTRUCTION.format(text=text[:2500])
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return _parse_json_response(response.choices[0].message.content, self.name)
        except Exception as e:
            logger.warning(f"OpenAI summarizer failed: {e}")
            return SummaryResult(summary="Summary unavailable.", key_risks=[], model_name=self.name)
