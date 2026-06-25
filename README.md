# Argus Risk Analyst

AI-powered news risk monitoring for 100s of companies — in the time it takes a human analyst to cover two.

**Live demo:** `[your-streamlit-url]` | **Stack:** FinBERT · BART-MNLI · Claude · XGBoost · Streamlit

---

## The Problem

A buy-side analyst can meaningfully track ~5–10 companies daily. For a fund with 200+ positions, systematic news risk monitoring is impossible at human scale. Negative sentiment about a regulatory probe, a CEO departure, or a supply chain disruption can move a stock 5–10% in hours. **This system flags those signals automatically across any number of tickers.**

---

## Architecture

```
News (Finnhub + RSS)
        │  dedup via embedding similarity
        ▼
  FinBERT Sentiment          Zero-Shot Event Classifier        LLM Summarizer
  (tone per article,         (regulatory / fraud / M&A /       (grounded, source-
   chunked for long text)     layoffs / earnings / etc.)        text only)
        │                            │                                │
        └──────────────────┬─────────┘                                │
                           ▼                                          │
                   Risk Engine (v1 weighted / v2 XGBoost) ◄──────────┘
                           │
                    SQLite (daily snapshots)
                           │
                    Streamlit Dashboard
                    (leaderboard · trends · drill-down · alerts)
```

### Key design principle: plug-in model architecture

Every NLP stage has an abstract base class. Adding or swapping a model is three steps:

1. Subclass `BaseSentimentModel` / `BaseEventClassifier` / `BaseSummarizer`
2. Add `@register_sentiment("your_name")` decorator
3. Set `models.sentiment.active: your_name` in `config.yaml`

Nothing else changes. One model failing never breaks another.

---

## Risk Model

### v1 — Weighted formula (default)

```
risk_score = Σ(event_confidence × event_weight) × sentiment_modifier × recency_weight + volume_bonus
```

| Event type | Base risk weight |
|---|---|
| Fraud / accounting | 4.5 |
| Regulatory / legal | 4.0 |
| Leadership change | 2.5 |
| Earnings miss | 2.0 |
| Layoffs | 2.0 |
| M&A | 1.5 |
| Supply chain | 1.5 |
| Macro exposure | 1.0 |
| General news | 0.5 |

Scores are decayed exponentially over 7 days so stale news loses weight automatically.

### v2 — XGBoost (train after 60 days)

Uses historical price reactions as labels (did the stock drop >2% in 3 trading days?), with no manual labeling required. Key results after training on N days of data:

| Metric | Our model | Volatility baseline |
|---|---|---|
| ROC-AUC | `TBD` | `TBD` |
| Precision@0.5 | `TBD` | — |
| Recall@0.5 | `TBD` | — |

Top predictive features (by XGBoost importance):
1. `event_regulatory_legal`
2. `sentiment_neg_weighted`
3. `neg_article_fraction`
4. `event_fraud_accounting`
5. `recency_mean`

Switch to v2 by changing `risk.version: v2` in `config.yaml`.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/you/finai-risk-analyst
cd finai-risk-analyst
pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# Fill in FINNHUB_API_KEY

# 3. First run
python -m src.pipeline

# 4. Launch dashboard
streamlit run app/streamlit_app.py

# 5. Tests
pytest tests/ -v
```

---

## Adding a new company

Edit `config.yaml`:

```yaml
companies:
  - { ticker: UBER, name: "Uber Technologies Inc." }   # add this line
```

Restart the pipeline. No code changes.

---

## Adding a new NLP model (e.g. FinGPT sentiment)

```python
# src/nlp/sentiment/fingpt.py
from src.nlp.base import BaseSentimentModel, SentimentResult
from src.nlp.registry import register_sentiment

@register_sentiment("fingpt")
class FinGPTSentiment(BaseSentimentModel):
    name = "fingpt"
    def load(self): ...
    def score(self, text) -> SentimentResult: ...
```

Then in `config.yaml`:
```yaml
models:
  sentiment:
    active: fingpt   # ← one line change
```

---

## Adding a new event type

In `config.yaml`:
```yaml
event_taxonomy:
  cyber_breach:
    label: "data breach or cybersecurity incident or hack"
    base_risk: 3.5
```

The zero-shot model picks it up automatically (no retraining). The risk scorer reads weights from config. One edit, everywhere.

---

## Project structure

```
src/
├── pipeline.py          # daily orchestrator
├── db.py                # SQLAlchemy models (articles, nlp_scores, daily_risk_scores)
├── ingest/
│   └── news.py          # Finnhub + RSS, embedding dedup
├── nlp/
│   ├── base.py          # abstract interfaces (the extensibility contract)
│   ├── registry.py      # model registry + builder
│   ├── sentiment/finbert.py
│   ├── events/zero_shot.py
│   └── summarizer/llm.py
└── risk/
    ├── features.py      # flat feature vector from NLP scores
    ├── scorer.py        # v1 weighted + v2 XGBoost
    └── backtest.py      # training + evaluation
app/
└── streamlit_app.py     # leaderboard + drill-down dashboard
```

---

## Deployment

The pipeline runs daily via **GitHub Actions** (`.github/workflows/daily_run.yml`), caching the SQLite DB between runs. The dashboard is hosted on **Streamlit Community Cloud** (free) — connect your repo and point it at `app/streamlit_app.py`.

API keys are stored as **GitHub Secrets** and never committed.

---

## Limitations and future work

- Zero-shot classification is accurate but slower than a fine-tuned classifier. A labeled dataset of ~300 examples would meaningfully improve event detection precision.
- SQLite is sufficient for hundreds of companies but a Postgres migration path is one config line change (SQLAlchemy handles it).
- Summaries are grounded on source text, but LLMs can still misrepresent nuance. Human review is recommended before acting on any flag.
- v2 model performance is limited by available history — accuracy improves with more data.

---

*Not investment advice. Built for educational and research purposes.*
