import numpy as np
import pickle
import os
from config import ML_MIN_SAMPLES, ML_CONFIDENCE_THRESHOLD, ML_TRUST_SAMPLES
from logger import logger

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed — using rule-based fallback")

# Order must match CrossoverRecord.get_features() exactly — used by
# ml_backtest.py to label feature-importance output.
FEATURE_NAMES = [
    "signal_num", "entry_ltp", "smma_gap",
    "ltq_ratio", "etq_5m", "etq_ratio", "spread",
]


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
        # etq_5m used to be stored as the RAW 5-minute traded-quantity sum.
        # That's fine within one market, but this model is shared across
        # NSE equity (lakhs of shares), MCX/CDS (lot-based volumes), crypto
        # (fractional BTC-style quantities) and US stocks — so the same
        # raw number means wildly different things depending on market,
        # and a single split point learned by the tree can't generalize
        # across them. Normalizing by avg_ltq_5m turns it into "how many
        # average-trade-sizes worth of volume traded in the last 5 min" —
        # a unitless, market-comparable number.
        self.etq_5m    = etq_5m / avg_ltq_5m if avg_ltq_5m else 0.0
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
            self.etq_5m,
            self.etq_ratio,
            self.spread,
        ])


class MLModel:
    def __init__(self, model_path="ml_model.pkl", min_samples=ML_MIN_SAMPLES,
                 confidence_threshold=ML_CONFIDENCE_THRESHOLD,
                 trust_samples=ML_TRUST_SAMPLES):
        self.model_path    = model_path
        self.min_samples   = min_samples     # when XGBoost starts training
        self.trust_samples = trust_samples   # when verdicts are shown to the user
        self.threshold      = confidence_threshold
        self.model       = None
        self.X           = []   # feature rows
        self.y           = []   # labels (1=profit, 0=loss)
        self._load()

    @property
    def sample_count(self):
        return len(self.y)

    @property
    def is_trusted(self):
        """Whether there's enough closed-trade history for a verdict to
        mean anything. min_samples (default 50) is just when XGBoost has
        enough rows to fit without erroring — it says nothing about
        whether the fit generalizes. trust_samples (default 100) is a
        separate, higher bar for actually showing ACCEPT/AVOID to the
        user instead of a neutral "Learning" state."""
        return self.sample_count >= self.trust_samples

    def _load(self):
        if os.path.exists(self.model_path) and XGB_AVAILABLE:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("ML model loaded from disk")

    def _save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def bootstrap_from_history(self):
        """Load every closed signal with a persisted feature vector
        from the database and seed self.X/self.y with it. Without
        this, online learning only sees trades closed during the
        *current* run — every restart threw away all prior training
        signal even though it was sitting in the DB the whole time.
        Call once at startup, before the app starts taking new ticks."""
        from core.database import get_closed_signals_with_features
        rows = get_closed_signals_with_features()
        if not rows:
            return
        for r in rows:
            feat = [
                r["feat_signal_num"], r["entry_ltp"], r["feat_smma_gap"],
                r["feat_ltq_ratio"], r["feat_etq_5m"], r["feat_etq_ratio"],
                r["feat_spread"],
            ]
            self.X.append(np.array(feat, dtype=float))
            self.y.append(r["profitable"])
        logger.info(f"ML: bootstrapped {len(rows)} historical outcomes from DB")
        if len(self.y) >= self.min_samples and XGB_AVAILABLE:
            self._train()

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

    def feature_importance(self):
        """Returns {feature_name: importance} sorted descending, or
        None if no XGBoost model has been trained yet."""
        if not self.model or not XGB_AVAILABLE:
            return None
        importances = self.model.feature_importances_
        pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda p: -p[1])
        return {name: round(float(val), 4) for name, val in pairs}

    def predict(self, record):
        """Returns (prediction, confidence, reason).
        prediction/confidence are None — not a fake guess — until
        is_trusted is True. A rule-based or freshly-trained XGBoost
        score on a handful of samples is not a real signal; showing
        ACCEPT/AVOID from it just teaches the user to trust noise."""
        if not self.is_trusted:
            reason = f"Learning ({self.sample_count}/{self.trust_samples} closed trades)"
            return None, None, reason

        if self.model and XGB_AVAILABLE:
            feat = record.get_features().reshape(1, -1)
            prob = self.model.predict_proba(feat)[0][1]
            pred = 1 if prob >= self.threshold else 0
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
        pred  = 1 if score >= self.threshold else 0
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