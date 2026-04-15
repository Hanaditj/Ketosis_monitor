"""
serial_reader.py
----------------
Reads JSON packets from Arduino over serial,
validates them, and appends to SQLite database.

Usage:
    python serial_reader.py --port /dev/ttyUSB0 --baud 9600
"""

import serial
import json
import sqlite3
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path

# ── Config ─────────────────────────────────────────────
DB_PATH   = Path(__file__).parent.parent / "data" / "readings.db"
LOG_PATH  = Path(__file__).parent.parent / "data" / "reader.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

REQUIRED_KEYS = {"ts", "temp_c", "humidity", "ppm", "mmol", "state"}

# ── Database ────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            temp_c    REAL,
            humidity  REAL,
            ppm       REAL    NOT NULL,
            mmol      REAL    NOT NULL,
            rs        REAL,
            voltage   REAL,
            state     TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            started   TEXT NOT NULL,
            port      TEXT NOT NULL,
            note      TEXT
        )
    """)
    conn.commit()
    log.info("Database initialised at %s", DB_PATH)


def insert_reading(conn: sqlite3.Connection, data: dict, ts: str) -> None:
    conn.execute(
        """INSERT INTO readings
           (timestamp, temp_c, humidity, ppm, mmol, rs, voltage, state)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            ts,
            data.get("temp_c"),
            data.get("humidity"),
            data["ppm"],
            data["mmol"],
            data.get("rs"),
            data.get("voltage"),
            data["state"],
        )
    )
    conn.commit()

# ── Validation ──────────────────────────────────────────

def validate(data: dict) -> bool:
    if not REQUIRED_KEYS.issubset(data.keys()):
        log.warning("Missing keys: %s", REQUIRED_KEYS - data.keys())
        return False
    if not (0 <= data["ppm"] <= 500):
        log.warning("PPM out of range: %s", data["ppm"])
        return False
    if not (0 <= data["mmol"] <= 30):
        log.warning("mmol out of range: %s", data["mmol"])
        return False
    return True

# ── Main Loop ───────────────────────────────────────────

def run(port: str, baud: int) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    log.info("Opening serial port %s @ %d baud", port, baud)
    try:
        ser = serial.Serial(port, baud, timeout=5)
    except serial.SerialException as e:
        log.error("Cannot open port: %s", e)
        return

    # Log session start
    conn.execute(
        "INSERT INTO sessions (started, port) VALUES (?, ?)",
        (datetime.utcnow().isoformat(), port)
    )
    conn.commit()

    log.info("Listening — press Ctrl+C to stop")
    error_streak = 0

    while True:
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw or not raw.startswith("{"):
                continue

            data = json.loads(raw)

            # Handle status/error messages from Arduino
            if "status" in data:
                log.info("Arduino: %s", data)
                continue
            if "error" in data:
                log.warning("Arduino error: %s", data["error"])
                continue

            if not validate(data):
                error_streak += 1
                if error_streak > 10:
                    log.error("10 consecutive bad readings — check sensor wiring")
                continue

            error_streak = 0
            ts = datetime.utcnow().isoformat()
            insert_reading(conn, data, ts)

            log.info(
                "%.2f ppm  |  %.3f mmol/L  |  %-12s  |  %.1f°C  %.0f%%RH",
                data["ppm"], data["mmol"], data["state"],
                data["temp_c"], data["humidity"]
            )

        except json.JSONDecodeError:
            log.debug("Non-JSON line skipped")
        except KeyboardInterrupt:
            log.info("Stopped by user")
            break
        except Exception as e:
            log.error("Unexpected error: %s", e)
            time.sleep(1)

    ser.close()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KetosenseX Serial Reader")
    parser.add_argument("--port",  default="/dev/ttyUSB0", help="Arduino serial port")
    parser.add_argument("--baud",  type=int, default=9600,  help="Baud rate")
    args = parser.parse_args()
    run(args.port, args.baud)
