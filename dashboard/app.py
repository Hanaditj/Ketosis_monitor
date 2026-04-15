"""
app.py  —  KetosenseX Dashboard Backend
----------------------------------------
Flask API + serves the HTML dashboard.

    python app.py
    # → http://localhost:5000
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template, request

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from ml_model import KetosisAnalyser, compute_trends, simulate_dataset

app     = Flask(__name__)
DB_PATH = Path(__file__).parent.parent / "data" / "readings.db"

# Load (or train on synthetic data) the ML model at startup
analyser = KetosisAnalyser.load()
if analyser is None:
    print("No saved model found — training on synthetic data …")
    analyser = KetosisAnalyser()
    analyser.train(simulate_dataset(800))
    analyser.save()


# ── DB Helper ───────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_demo_data():
    """Seed DB with synthetic data if empty (for demo mode)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    if count == 0:
        records = simulate_dataset(200)
        base_ts = datetime.utcnow() - timedelta(hours=48)
        for i, r in enumerate(records):
            ts = (base_ts + timedelta(minutes=i * 15)).isoformat()
            conn.execute(
                """INSERT INTO readings
                   (timestamp, temp_c, humidity, ppm, mmol, rs, voltage, state)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (ts, r["temp_c"], r["humidity"], r["ppm"], r["mmol"],
                 r["rs"], None, r["state"])
            )
        conn.commit()
        print(f"Seeded {len(records)} demo readings")
    conn.close()


# ── API Routes ──────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/latest")
def api_latest():
    """Most recent reading + ML prediction."""
    conn = get_db()
    row  = conn.execute(
        "SELECT * FROM readings ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "no_data"}), 404

    reading = dict(row)
    pred    = analyser.predict(reading)
    return jsonify({**reading, "ml": pred})


@app.route("/api/history")
def api_history():
    """Time-series of readings for charts."""
    hours = int(request.args.get("hours", 24))
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM readings WHERE timestamp > ? ORDER BY timestamp",
        (since,)
    ).fetchall()
    conn.close()

    records = [dict(r) for r in rows]
    records = compute_trends(records)
    return jsonify(records)


@app.route("/api/summary")
def api_summary():
    """Stats summary for the last 24h."""
    since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    conn  = get_db()
    rows  = conn.execute(
        "SELECT * FROM readings WHERE timestamp > ? ORDER BY timestamp",
        (since,)
    ).fetchall()
    conn.close()

    if not rows:
        return jsonify({"error": "no_data"}), 404

    ppms  = [r["ppm"]  for r in rows]
    mmols = [r["mmol"] for r in rows]

    state_counts = {}
    for r in rows:
        state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1

    dominant = max(state_counts, key=state_counts.get)

    return jsonify({
        "n_readings":      len(rows),
        "avg_ppm":         round(sum(ppms)  / len(ppms),  2),
        "max_ppm":         round(max(ppms),  2),
        "min_ppm":         round(min(ppms),  2),
        "avg_mmol":        round(sum(mmols) / len(mmols), 3),
        "max_mmol":        round(max(mmols), 3),
        "state_counts":    state_counts,
        "dominant_state":  dominant,
        "time_window_h":   24,
    })


@app.route("/api/anomalies")
def api_anomalies():
    """Return readings flagged as anomalous by the ML model."""
    since = (datetime.utcnow() - timedelta(hours=48)).isoformat()
    conn  = get_db()
    rows  = conn.execute(
        "SELECT * FROM readings WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 200",
        (since,)
    ).fetchall()
    conn.close()

    anomalies = []
    for row in rows:
        r    = dict(row)
        pred = analyser.predict(r)
        if pred.get("is_anomaly"):
            anomalies.append({**r, "ml": pred})

    return jsonify(anomalies)


# ── Run ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialise DB tables (ml_model will also do this, but safe to repeat)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temp_c    REAL, humidity REAL,
            ppm       REAL NOT NULL,
            mmol      REAL NOT NULL,
            rs        REAL, voltage REAL,
            state     TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    ensure_demo_data()
    app.run(debug=True, host="0.0.0.0", port=5000)
