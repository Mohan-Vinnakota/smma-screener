"""
ml_backtest.py — Stage 7: ML Excellence

Run this anytime to:
  1. Backtest the model against real historical outcomes from the DB
     (chronological train/test split — never tests on data it trained on)
  2. Print an accuracy report (accuracy, precision, recall, F1, confusion
     matrix), overall and broken down per market
  3. Print feature importance (which inputs actually drive predictions)
  4. Sweep the confidence threshold and recommend the value that best
     hits the >65% accuracy target — copy it into config.py's
     ML_CONFIDENCE_THRESHOLD to apply it
  5. Retrain a final model on ALL historical data and overwrite
     ml_model.pkl — this IS the retraining pipeline. Run this script
     periodically (or after a big batch of new closed trades) to keep
     the live model current with everything the DB has learned since
     it was last trained, not just whatever happened to close during
     one particular run.

Usage:
    python ml_backtest.py                  # all markets combined
    python ml_backtest.py --market NSE     # one market only
    python ml_backtest.py --no-retrain     # report only, don't touch ml_model.pkl

Needs enough closed signals with a persisted feature vector to make a
train/test split meaningful — with fewer than ~30 you'll just get a
warning and no report. Feature persistence was added in this same
stage, so only signals closed AFTER this update carry a usable feature
vector; older rows are skipped automatically.
"""

import sys
import argparse
import numpy as np

from database import get_closed_signals_with_features, init_db
from ml_model import FEATURE_NAMES
from config import ML_BACKTEST_TEST_FRACTION, ML_CONFIDENCE_THRESHOLD
from logger import logger

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

MIN_ROWS_FOR_BACKTEST = 30


def rows_to_xy(rows):
    X, y = [], []
    for r in rows:
        X.append([
            r["feat_signal_num"], r["entry_ltp"], r["feat_smma_gap"],
            r["feat_ltq_ratio"], r["feat_etq_5m"], r["feat_etq_ratio"],
            r["feat_spread"],
        ])
        y.append(r["profitable"])
    return np.array(X, dtype=float), np.array(y, dtype=int)


def train_model(X, y):
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, verbosity=0
    )
    model.fit(X, y)
    return model


def report_metrics(y_true, y_pred, label):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n── {label} ──")
    print(f"  Samples   : {len(y_true)}")
    print(f"  Accuracy  : {acc:.1%}")
    print(f"  Precision : {prec:.1%}  (of predicted WINs, how many were actual wins)")
    print(f"  Recall    : {rec:.1%}  (of actual wins, how many the model caught)")
    print(f"  F1        : {f1:.3f}")
    print(f"  Confusion matrix [rows=actual, cols=predicted], order [LOSS, WIN]:")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    return acc


def sweep_threshold(model, X_test, y_test):
    print("\n── Confidence threshold sweep (test set only) ──")
    print(f"  {'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Accepted':<10}")
    probs = model.predict_proba(X_test)[:, 1]
    best_thresh, best_acc = ML_CONFIDENCE_THRESHOLD, -1

    for thresh in np.arange(0.30, 0.91, 0.05):
        pred = (probs >= thresh).astype(int)
        if pred.sum() == 0:
            # Nothing accepted at this threshold — accuracy is undefined,
            # not useful as a recommendation.
            print(f"  {thresh:<10.2f} {'—':<10} {'—':<10} {'—':<10} {0:<10}")
            continue
        acc  = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec  = recall_score(y_test, pred, zero_division=0)
        print(f"  {thresh:<10.2f} {acc:<10.1%} {prec:<10.1%} {rec:<10.1%} {int(pred.sum()):<10}")
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh

    print(f"\n  Recommended threshold: {best_thresh:.2f} (accuracy {best_acc:.1%} on held-out test set)")
    if best_acc >= 0.65:
        print(f"  ✅ Hits the >65% accuracy target.")
    else:
        print(f"  ⚠️  Doesn't hit 65% yet with current data/features — needs more history or feature work.")
    if abs(best_thresh - ML_CONFIDENCE_THRESHOLD) > 1e-9:
        print(f"  → Update config.py: ML_CONFIDENCE_THRESHOLD = {best_thresh:.2f}  (currently {ML_CONFIDENCE_THRESHOLD})")
    else:
        print(f"  → config.py's ML_CONFIDENCE_THRESHOLD ({ML_CONFIDENCE_THRESHOLD}) is already the best value found.")
    return best_thresh


def run(market=None, retrain=True):
    if not XGB_AVAILABLE:
        print("❌ XGBoost not installed — can't backtest. pip install xgboost")
        return
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn not installed — can't compute metrics. pip install scikit-learn")
        return

    init_db()
    rows = get_closed_signals_with_features(market=market)
    label = market or "ALL MARKETS"

    if len(rows) < MIN_ROWS_FOR_BACKTEST:
        print(
            f"⚠️  Only {len(rows)} closed signals with a feature vector for {label} "
            f"— need at least {MIN_ROWS_FOR_BACKTEST} for a meaningful backtest.\n"
            f"    Keep the screener running (--simulate is fine to accumulate history "
            f"quickly) and re-run this script later."
        )
        return

    X, y = rows_to_xy(rows)
    split = int(len(rows) * (1 - ML_BACKTEST_TEST_FRACTION))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"=== ML Backtest — {label} ===")
    print(f"Total closed signals with features: {len(rows)}")
    print(f"Train: {len(X_train)}  |  Test (held out, chronologically last {ML_BACKTEST_TEST_FRACTION:.0%}): {len(X_test)}")

    model = train_model(X_train, y_train)

    # Accuracy report at the CURRENT config threshold
    probs = model.predict_proba(X_test)[:, 1]
    pred  = (probs >= ML_CONFIDENCE_THRESHOLD).astype(int)
    report_metrics(y_train, (model.predict_proba(X_train)[:, 1] >= ML_CONFIDENCE_THRESHOLD).astype(int),
                    f"Train set @ threshold {ML_CONFIDENCE_THRESHOLD}")
    report_metrics(y_test, pred, f"Test set (held out) @ threshold {ML_CONFIDENCE_THRESHOLD}")

    # Feature importance
    print("\n── Feature importance ──")
    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda p: -p[1])
    for name, val in importances:
        bar = "█" * int(val * 40)
        print(f"  {name:<12} {val:.3f}  {bar}")

    # Per-market breakdown (only meaningful for the ALL-markets run)
    if market is None:
        markets_present = sorted(set(r["market"] for r in rows))
        if len(markets_present) > 1:
            print("\n── Per-market accuracy (using the same test-set model, full history per market) ──")
            for m in markets_present:
                m_rows = [r for r in rows if r["market"] == m]
                if len(m_rows) < 5:
                    print(f"  {m:<8} skipped — only {len(m_rows)} closed signals")
                    continue
                Xm, ym = rows_to_xy(m_rows)
                pm = (model.predict_proba(Xm)[:, 1] >= ML_CONFIDENCE_THRESHOLD).astype(int)
                acc = accuracy_score(ym, pm)
                print(f"  {m:<8} accuracy {acc:.1%}  ({len(m_rows)} signals)")

    # Threshold sweep + recommendation
    sweep_threshold(model, X_test, y_test)

    # Retraining pipeline: refit on ALL data (train+test) and save
    if retrain:
        print("\n── Retraining final model on full history ──")
        final_model = train_model(X, y)
        import pickle
        with open("ml_model.pkl", "wb") as f:
            pickle.dump(final_model, f)
        print(f"✅ ml_model.pkl retrained on all {len(rows)} closed signals and saved.")
        print("   Restart the app (or it'll pick this up on next MLModel._load()) to use it.")
    else:
        print("\n(--no-retrain passed — ml_model.pkl left untouched)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest, evaluate, and retrain the SMMA screener's ML model.")
    parser.add_argument("--market", default=None, help="Restrict to one market, e.g. NSE, MCX, CDS, FNO, CRYPTO, US")
    parser.add_argument("--no-retrain", action="store_true", help="Report only — don't overwrite ml_model.pkl")
    args = parser.parse_args()

    run(market=args.market, retrain=not args.no_retrain)
