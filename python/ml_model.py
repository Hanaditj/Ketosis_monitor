"""
ml_model.py
-----------
Trains and evaluates a ketosis state classifier
from historical readings. Also does trend analysis
(moving average, anomaly detection).

Run standalone to train:
    python ml_model.py --train

Or import KetosisAnalyser in dashboard/app.py.
"""

import sqlite3
import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Graceful sklearn import
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("scikit-learn not installed — ML features disabled. pip install scikit-learn")

DB_PATH    = Path(__file__).parent.parent / "data" / "readings.db"
MODEL_PATH = Path(__file__).parent.parent / "data" / "model.pkl"
SCALER_PATH = Path(__file__).parent.parent / "data" / "scaler.pkl"

STATE_MAP = {"none": 0, "light": 1, "nutritional": 2, "deep": 3}
STATE_INV = {v: k for k, v in STATE_MAP.items()}

FEATURES = ["ppm", "mmol", "temp_c", "humidity", "rs"]


# ── Data Loading ────────────────────────────────────────

def load_readings(n_hours: int = 24 * 7) -> list[dict]:
    """Load recent readings from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    since = (datetime.utcnow() - timedelta(hours=n_hours)).isoformat()
    rows = conn.execute(
        "SELECT * FROM readings WHERE timestamp > ? ORDER BY timestamp",
        (since,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def simulate_dataset(n: int = 500) -> list[dict]:
    """
    Generate synthetic readings for demo/testing when
    no real sensor data exists yet. Matches realistic
    acetone breath profiles for each ketosis state.
    """
    rng = np.random.default_rng(42)
    records = []

    profiles = {
        "none":        dict(ppm=(0.5, 1.5),  temp=(20, 30), hum=(40, 80)),
        "light":       dict(ppm=(2.0, 5.0),  temp=(20, 30), hum=(40, 80)),
        "nutritional": dict(ppm=(5.0, 25.0), temp=(20, 30), hum=(40, 80)),
        "deep":        dict(ppm=(25.0, 60.0),temp=(20, 30), hum=(40, 80)),
    }

    for state, p in profiles.items():
        for _ in range(n // len(profiles)):
            ppm  = rng.uniform(*p["ppm"])
            temp = rng.uniform(*p["temp"])
            hum  = rng.uniform(*p["hum"])
            records.append({
                "ppm":      round(ppm, 2),
                "mmol":     round(ppm * 0.0572, 4),
                "temp_c":   round(temp, 1),
                "humidity": round(hum, 1),
                "rs":       round(45.0 / (ppm + 0.01), 2),
                "state":    state,
            })

    rng.shuffle(records)
    return records


# ── ML Model ────────────────────────────────────────────

class KetosisAnalyser:
    def __init__(self):
        self.clf     = None
        self.scaler  = None
        self.iso_forest = None  # Anomaly detector

    def _extract_features(self, records: list[dict]) -> tuple:
        X, y = [], []
        for r in records:
            row = [r.get(f, 0.0) or 0.0 for f in FEATURES]
            X.append(row)
            y.append(STATE_MAP.get(r.get("state", "none"), 0))
        return np.array(X, dtype=float), np.array(y, dtype=int)

    def train(self, records: list[dict]) -> dict:
        if not SKLEARN_OK:
            return {"error": "scikit-learn not available"}
        if len(records) < 20:
            print("Not enough real data — using synthetic dataset for demo")
            records = simulate_dataset(600)

        X, y = self._extract_features(records)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        self.clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        )
        self.clf.fit(X_train, y_train)

        # Anomaly detector (trained on all data)
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.iso_forest.fit(X_scaled)

        # Evaluate
        cv_scores = cross_val_score(self.clf, X_scaled, y, cv=5)
        y_pred = self.clf.predict(X_test)

        report = classification_report(
            y_test, y_pred,
            target_names=list(STATE_MAP.keys()),
            output_dict=True
        )
        result = {
            "cv_mean":  round(float(cv_scores.mean()), 3),
            "cv_std":   round(float(cv_scores.std()), 3),
            "n_samples": len(records),
            "report":   report,
        }
        print(f"\nCV Accuracy: {result['cv_mean']:.1%} ± {result['cv_std']:.1%}")
        print(classification_report(y_test, y_pred, target_names=list(STATE_MAP.keys())))
        return result

    def save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH,  "wb") as f: pickle.dump(self, f)
        print(f"Model saved → {MODEL_PATH}")

    @staticmethod
    def load() -> Optional["KetosisAnalyser"]:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        return None

    def predict(self, reading: dict) -> dict:
        """Predict state for a single reading dict."""
        if not SKLEARN_OK or self.clf is None:
            return {"state": reading.get("state", "unknown"), "confidence": None}

        row = np.array([[reading.get(f, 0.0) or 0.0 for f in FEATURES]])
        row_scaled = self.scaler.transform(row)

        state_idx  = int(self.clf.predict(row_scaled)[0])
        proba      = self.clf.predict_proba(row_scaled)[0]
        is_anomaly = self.iso_forest.predict(row_scaled)[0] == -1

        return {
            "state":      STATE_INV[state_idx],
            "confidence": round(float(proba.max()), 3),
            "is_anomaly": bool(is_anomaly),
            "proba":      {STATE_INV[i]: round(float(p), 3) for i, p in enumerate(proba)},
        }


# ── Trend Analysis ──────────────────────────────────────

def compute_trends(records: list[dict], window: int = 10) -> list[dict]:
    """Add rolling average and rate-of-change to each record."""
    if not records:
        return []

    ppms = [r["ppm"] for r in records]
    out  = []

    for i, r in enumerate(records):
        start = max(0, i - window + 1)
        window_ppms = ppms[start: i + 1]
        avg  = float(np.mean(window_ppms))
        roc  = float(ppms[i] - ppms[max(0, i-1)]) if i > 0 else 0.0
        trend = "rising" if roc > 0.5 else "falling" if roc < -0.5 else "stable"

        out.append({
            **r,
            "rolling_avg_ppm": round(avg, 2),
            "rate_of_change":  round(roc, 3),
            "trend":           trend,
        })

    return out


# ── CLI ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KetosenseX ML Model")
    parser.add_argument("--train",    action="store_true", help="Train the model")
    parser.add_argument("--simulate", action="store_true", help="Force synthetic data")
    args = parser.parse_args()

    if args.train:
        analyser = KetosisAnalyser()
        records  = simulate_dataset(800) if args.simulate else load_readings()
        result   = analyser.train(records)
        analyser.save()
        print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2))
    else:
        parser.print_help()
