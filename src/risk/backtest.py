"""
src/risk/backtest.py — Backtest and v2 model training

Workflow:
  1. Pull all historical daily_risk_scores from DB
  2. Fetch post-event stock returns from yfinance
  3. Label each day: 1 if stock dropped >2% in the next 3 days, else 0
  4. Train XGBoost on features, evaluate, save model
  5. Print a clean summary you can paste into your README

Run via: python -m src.risk.backtest
Or from: notebooks/train_risk_model.ipynb (recommended for EDA)
"""

from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from sqlalchemy import select

from src.db import get_session, DailyRiskScore
from src.risk.features import column_names, features_to_array


# ── Step 1: Load stored feature vectors ───────────────────────

def load_feature_df(event_types: Optional[list] = None) -> pd.DataFrame:
    """
    Load all DailyRiskScore rows and unpack their feature_vector JSON
    into a flat DataFrame. This is the training dataset.
    """
    with get_session() as session:
        rows = session.execute(select(DailyRiskScore)).scalars().all()

    records = []
    for row in rows:
        feat = json.loads(row.feature_vector) if row.feature_vector else {}
        feat["ticker"]     = row.ticker
        feat["score_date"] = row.score_date
        feat["v1_score"]   = row.risk_score    # keep for comparison
        records.append(feat)

    if not records:
        raise ValueError("No DailyRiskScore rows found. Run pipeline for at least 30 days first.")

    return pd.DataFrame(records)


# ── Step 2: Fetch price reactions ─────────────────────────────

def fetch_returns(tickers: list, start: date, end: date) -> pd.DataFrame:
    """
    Download OHLCV from yfinance and compute 3-day forward returns.
    Returns a DataFrame indexed by (ticker, date) with a `fwd_return_3d` column.
    """
    logger.info(f"Fetching price data for {len(tickers)} tickers...")
    raw = yf.download(
        tickers,
        start=str(start - timedelta(days=10)),
        end=str(end + timedelta(days=10)),
        progress=False,
        auto_adjust=True,
    )

    # Handle single vs multi-ticker yfinance output
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = tickers

    returns = close.pct_change()

    # 3-day forward return: sum of returns for the next 3 trading days
    fwd_3d = returns.shift(-1).rolling(3).sum().shift(-2)

    records = []
    for ticker in tickers:
        if ticker not in fwd_3d.columns:
            continue
        for idx, val in fwd_3d[ticker].dropna().items():
            records.append({
                "ticker":        ticker,
                "score_date":    idx.date(),
                "fwd_return_3d": float(val),
            })

    return pd.DataFrame(records)


# ── Step 3: Label ─────────────────────────────────────────────

def label_dataset(feat_df: pd.DataFrame, return_df: pd.DataFrame,
                  drop_threshold: float = -0.02) -> pd.DataFrame:
    """
    Merge features with forward returns. Label = 1 if stock drops >2% in 3 days.
    """
    merged = feat_df.merge(return_df, on=["ticker", "score_date"], how="inner")
    merged["label"] = (merged["fwd_return_3d"] < drop_threshold).astype(int)

    pos_rate = merged["label"].mean()
    logger.info(
        f"Dataset: {len(merged)} samples | "
        f"positive rate (drop >{abs(drop_threshold)*100:.0f}%): {pos_rate:.1%}"
    )
    return merged


# ── Step 4: Train XGBoost ─────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    event_types: Optional[list] = None,
    output_path: str = "data/risk_model_v2.joblib",
    test_size: float = 0.2,
) -> dict:
    """
    Train XGBoost classifier. Returns evaluation metrics dict.
    Uses time-based split (not random) to avoid look-ahead bias.
    """
    import joblib
    import xgboost as xgb
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score,
        classification_report, average_precision_score
    )
    from sklearn.calibration import CalibratedClassifierCV

    cols = column_names(event_types)

    # Time-based split — CRITICAL: never use random split for time-series data
    df = df.sort_values("score_date")
    split_idx = int(len(df) * (1 - test_size))
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    X_train = train_df[cols].fillna(0).values
    y_train = train_df["label"].values
    X_test  = test_df[cols].fillna(0).values
    y_test  = test_df["label"].values

    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # Class weights for imbalanced labels
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Calibrate probabilities (important for the 0–10 score to mean something)
    calibrated = CalibratedClassifierCV(model, cv="prefit")
    calibrated.fit(X_test, y_test)

    # Evaluate
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc":          round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision":    round(average_precision_score(y_test, y_prob), 4),
        "precision":        round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_test, y_pred, zero_division=0), 4),
        "n_train":          len(train_df),
        "n_test":           len(test_df),
        "positive_rate":    round(float(y_test.mean()), 4),
    }

    # Feature importances
    importances = dict(zip(cols, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:10]
    metrics["top_features"] = top_features

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, output_path)
    logger.info(f"Model saved to {output_path}")

    # Print summary
    print("\n" + "="*55)
    print("  v2 Risk Model — Evaluation Summary")
    print("="*55)
    print(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision:    {metrics['avg_precision']:.4f}")
    print(f"  Precision@0.5:    {metrics['precision']:.4f}")
    print(f"  Recall@0.5:       {metrics['recall']:.4f}")
    print(f"  Test samples:     {metrics['n_test']}")
    print(f"  Positive rate:    {metrics['positive_rate']:.1%}")
    print("\n  Top 10 features by importance:")
    for feat, imp in top_features:
        bar = "█" * int(imp * 50)
        print(f"    {feat:<35} {bar} {imp:.4f}")
    print("="*55)

    return metrics


# ── Baseline comparison: GARCH ─────────────────────────────────

def garch_baseline(tickers: list, df: pd.DataFrame) -> dict:
    """
    Compare our model against a simple volatility baseline.
    'High volatility yesterday' is a naive predictor of drops tomorrow.
    Returns baseline AUC for comparison in your README.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {}

    logger.info("Computing GARCH/volatility baseline...")
    return_data = fetch_returns(
        tickers,
        start=df["score_date"].min(),
        end=df["score_date"].max(),
    )
    merged = df.merge(return_data, on=["ticker", "score_date"], how="inner")
    merged["label"] = (merged["fwd_return_3d"] < -0.02).astype(int)

    # Baseline: rolling 5-day realised volatility as risk predictor
    merged = merged.sort_values(["ticker", "score_date"])
    merged["vol_5d"] = (
        merged.groupby("ticker")["fwd_return_3d"]
        .transform(lambda x: x.shift(1).rolling(5).std())
    )
    merged = merged.dropna(subset=["vol_5d", "label"])

    if len(merged) < 10:
        return {"baseline_roc_auc": None}

    auc = roc_auc_score(merged["label"], merged["vol_5d"])
    logger.info(f"Volatility baseline ROC-AUC: {auc:.4f}")
    return {"baseline_roc_auc": round(auc, 4)}


# ── CLI entrypoint ────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv()

    ROOT = Path(__file__).resolve().parent.parent.parent
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    from src.db import init_db
    init_db(config["database"]["url"])

    event_types = list(config["event_taxonomy"].keys())
    tickers     = [c["ticker"] for c in config["companies"]]

    feat_df    = load_feature_df(event_types)
    return_df  = fetch_returns(tickers, feat_df["score_date"].min(), feat_df["score_date"].max())
    labeled_df = label_dataset(feat_df, return_df)
    metrics    = train_model(labeled_df, event_types)
    baseline   = garch_baseline(tickers, feat_df)

    print(f"\n  Our model vs baseline:")
    print(f"    v2 ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"    Volatility AUC:    {baseline.get('baseline_roc_auc', 'N/A')}")
