import numpy as np
import pickle
import os
from config import ML_MIN_SAMPLES
from logger import logger

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed — using rule-based fallback")


class CrossoverRecord:
    """Stores everything about one crossover event."""
    def __init__(self, symbol, signal, entry_ltp,
                 smma_fast, smma_slow,
                 avg_ltq_2m, avg_ltq_5m,
                 etq_5m, etq_20m,
                 bid_price, ask_price):

        self.symbol    = symbol
        self.signal    = signal       # "BUY" or "SELL"
        self.entry_ltp = entry_ltp

        # Features
        self.smma_gap  = (smma_fast - smma_slow) / smma_slow if smma_slow else 0
        self.ltq_ratio = avg_ltq_2m / avg_ltq_5m if avg_ltq_5m else 1.0
        self.etq_ratio = etq_5m / etq_20m * 4 if etq_20m else 1.0
        self.spread    = (ask_price - bid_price) / entry_ltp if entry_ltp else 0
        self.signal_num = 1 if signal == "BUY" else 0

        # Outcome — filled when trade closes
        self.exit_ltp    = None
        self.pnl         = None
        self.profitable  = None   # 1 = WIN, 0 = LOSS

        # Prediction
        self.predicted   = None
        self.confidence  = None
        self.reason      = ""

    def get_features(self):
        return np.array([
            self.signal_num,
            self.entry_ltp,
            self.smma_gap,
            self.ltq_ratio,
            self.etq_5m if hasattr(self, 'etq_5m') else 0,
            self.etq_ratio,
            self.spread,
        ])


class MLModel:
    def __init__(self, model_path="ml_model.pkl", min_samples=ML_MIN_SAMPLES):
        self.model_path  = model_path
        self.min_samples = min_samples
        self.model       = None
        self.X           = []   # feature rows
        self.y           = []   # labels (1=profit, 0=loss)
        self._load()

    def _load(self):
        if os.path.exists(self.model_path) and XGB_AVAILABLE:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("ML model loaded from disk")

    def _save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def record_outcome(self, record):
        """Call when trade closes — adds training sample."""
        if record.profitable is None:
            return
        self.X.append(record.get_features())
        self.y.append(record.profitable)

        if len(self.y) >= self.min_samples and XGB_AVAILABLE:
            self._train()

    def _train(self):
        X = np.array(self.X)
        y = np.array(self.y)
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            verbosity=0
        )
        self.model.fit(X, y)
        self._save()
        logger.info(f"Model trained on {len(y)} samples")

    def predict(self, record):
        """Returns (prediction, confidence, reason)."""
        if self.model and XGB_AVAILABLE:
            feat = record.get_features().reshape(1, -1)
            prob = self.model.predict_proba(feat)[0][1]
            pred = 1 if prob >= 0.5 else 0
            reason = f"XGBoost: {'ACCEPT' if pred else 'AVOID'} ({prob:.0%} confidence)"
            return pred, round(prob, 3), reason
        else:
            return self._rule_based(record)

    def _rule_based(self, record):
        """Fallback when not enough data to train ML yet."""
        score = 0.5

        if record.ltq_ratio > 1.5:
            score += 0.15
        elif record.ltq_ratio < 0.8:
            score -= 0.15

        if abs(record.smma_gap) > 0.002:
            score += 0.10
        else:
            score -= 0.10

        if record.etq_ratio > 1.2:
            score += 0.10
        elif record.etq_ratio < 0.8:
            score -= 0.10

        if record.spread > 0.005:
            score -= 0.05

        score = max(0.05, min(0.95, score))
        pred  = 1 if score >= 0.5 else 0
        verdict = "ACCEPT" if pred else "AVOID"
        reason = f"Rule-based: {verdict} (score={score:.2f})"
        return pred, round(score, 3), reason


if __name__ == "__main__":
    model = MLModel(min_samples=3)

    # Mix of WIN and LOSS records
    test_data = [
        (2000, 1500, 50000, 40000, 105, "WIN"),   # strong LTQ → WIN
        (1000, 1500, 30000, 40000, 98,  "LOSS"),  # weak LTQ → LOSS
        (2500, 1500, 60000, 40000, 107, "WIN"),   # strong LTQ → WIN
        (800,  1500, 25000, 40000, 97,  "LOSS"),  # weak LTQ → LOSS
        (2200, 1500, 55000, 40000, 106, "WIN"),   # strong LTQ → WIN
    ]

    for i, (ltq_2m, ltq_5m, etq5, etq20, exit_ltp, outcome) in enumerate(test_data):
        r = CrossoverRecord(
            symbol="TESTSTOCK",
            signal="BUY",
            entry_ltp=100,
            smma_fast=101,
            smma_slow=100,
            avg_ltq_2m=ltq_2m,
            avg_ltq_5m=ltq_5m,
            etq_5m=etq5,
            etq_20m=etq20,
            bid_price=99.9,
            ask_price=100.1,
        )
        r.exit_ltp   = exit_ltp
        r.pnl        = exit_ltp - 100
        r.profitable = 1 if r.pnl > 0 else 0
        model.record_outcome(r)
        print(f"Record {i+1}: {outcome} | pnl={r.pnl}")

    # Predict on new record
    test = CrossoverRecord(
        symbol="NEWSTOCK",
        signal="BUY",
        entry_ltp=150,
        smma_fast=151,
        smma_slow=149,
        avg_ltq_2m=3000,
        avg_ltq_5m=1500,
        etq_5m=60000,
        etq_20m=40000,
        bid_price=149.9,
        ask_price=150.1,
    )

    pred, conf, reason = model.predict(test)
    print(f"\nPrediction : {'WIN' if pred else 'LOSS'}")
    print(f"Confidence : {conf:.0%}")
    print(f"Reason     : {reason}")