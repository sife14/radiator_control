"""
Datenbank & Logging
===================
SQLite-basierte Speicherung aller Messdaten.
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Measurement:
    """Einzelne Messung."""

    timestamp: datetime
    room_temp: float
    outside_temp: Optional[float]
    window_open: bool
    heating_active: Optional[bool]
    control_offset: float
    target_temp: float
    mode: str


class Database:
    """SQLite Datenbank für Heizungsdaten."""

    def __init__(self, db_path: str = "measurements.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Erstellt Tabellen falls nicht vorhanden."""
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    room_temp REAL NOT NULL,
                    outside_temp REAL,
                    window_open INTEGER NOT NULL DEFAULT 0,
                    heating_active INTEGER,
                    control_offset REAL NOT NULL DEFAULT 0,
                    target_temp REAL NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'control',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_measurements_timestamp
                ON measurements(timestamp)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    parameters TEXT,
                    status TEXT DEFAULT 'running',
                    results TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    param_name TEXT NOT NULL,
                    param_value REAL NOT NULL,
                    confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS controller_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    controller_type TEXT NOT NULL,
                    predicted_temps TEXT,
                    optimal_offsets TEXT,
                    cost_value REAL,
                    solve_time_ms REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    room_temp REAL NOT NULL,
                    outside_temp REAL,
                    target_temp REAL NOT NULL,
                    window_open INTEGER NOT NULL,
                    previous_temp REAL,
                    previous_offset REAL,
                    heating_active INTEGER,
                    offset_action REAL NOT NULL,
                    reward REAL NOT NULL,
                    model_params TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_samples_timestamp
                ON training_samples(timestamp)
            """)
            conn.commit()
            logger.info("Database initialized at %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_measurement(self, measurement: Measurement):
        """Fügt eine Messung ein."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO measurements
                (timestamp, room_temp, outside_temp, window_open,
                 heating_active, control_offset, target_temp, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    measurement.timestamp,
                    measurement.room_temp,
                    measurement.outside_temp,
                    1 if measurement.window_open else 0,
                    1
                    if measurement.heating_active
                    else (0 if measurement.heating_active is False else None),
                    measurement.control_offset,
                    measurement.target_temp,
                    measurement.mode,
                ),
            )
            conn.commit()

    def get_measurement_stats(self) -> Dict[str, Any]:
        """Statistiken über die Datenbank."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM measurements")
            total = cursor.fetchone()[0]
            cursor.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM measurements"
            )
            row = cursor.fetchone()
            cursor.execute(
                "SELECT mode, COUNT(*) FROM measurements GROUP BY mode"
            )
            mode_counts = dict(cursor.fetchall())
        return {
            "total_measurements": total,
            "first_timestamp": row[0],
            "last_timestamp": row[1],
            "measurements_by_mode": mode_counts,
        }

    def start_experiment(self, name: str, exp_type: str, parameters: Dict) -> int:
        """Startet ein neues Experiment."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experiments (name, type, start_time, parameters, status)
                VALUES (?, ?, ?, ?, 'running')
                """,
                (name, exp_type, datetime.now(), json.dumps(parameters)),
            )
            conn.commit()
            return cursor.lastrowid

    def end_experiment(self, exp_id: int, results: Dict):
        """Beendet ein Experiment."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE experiments
                SET end_time = ?, status = 'completed', results = ?
                WHERE id = ?
                """,
                (datetime.now(), json.dumps(results), exp_id),
            )
            conn.commit()

    def log_controller_step(
        self,
        controller_type: str,
        predicted_temps: List[float],
        optimal_offsets: List[float],
        cost_value: float,
        solve_time_ms: float,
    ):
        """Loggt einen Controller-Optimierungsschritt."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO controller_logs
                (timestamp, controller_type, predicted_temps, optimal_offsets,
                 cost_value, solve_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(),
                    controller_type,
                    json.dumps(predicted_temps),
                    json.dumps(optimal_offsets),
                    cost_value,
                    solve_time_ms,
                ),
            )
            conn.commit()

    def log_training_sample(
        self,
        timestamp: datetime,
        state: Dict[str, Any],
        action: Dict[str, float],
        reward: float,
        model_params: Optional[Dict] = None,
    ):
        """Loggt ein Training-Sample für RL/KI."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO training_samples
                (timestamp, room_temp, outside_temp, target_temp, window_open,
                 previous_temp, previous_offset, heating_active,
                 offset_action, reward, model_params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    state.get("room_temp"),
                    state.get("outside_temp"),
                    state.get("target_temp"),
                    1 if state.get("window_open") else 0,
                    state.get("previous_temp"),
                    state.get("previous_offset"),
                    1
                    if state.get("heating_active")
                    else (
                        0 if state.get("heating_active") is False else None
                    ),
                    action.get("offset", 0),
                    reward,
                    json.dumps(model_params) if model_params else None,
                ),
            )
            conn.commit()

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Zusammenfassung der Regelungsperformance."""
        try:
            import numpy as np

            start_time = datetime.now() - timedelta(days=days)
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as samples,
                        AVG(room_temp) as avg_room_temp,
                        AVG(target_temp) as avg_target_temp,
                        AVG(ABS(room_temp - target_temp)) as mae,
                        AVG((room_temp - target_temp) * (room_temp - target_temp)) as mse,
                        AVG(control_offset) as avg_offset,
                        SUM(CASE WHEN ABS(room_temp - target_temp) < 0.5
                            THEN 1 ELSE 0 END) * 100.0 / MAX(COUNT(*), 1) as comfort_percent
                    FROM measurements
                    WHERE timestamp >= ? AND window_open = 0
                    """,
                    (start_time,),
                )
                row = cursor.fetchone()

            if row and row["samples"] and row["samples"] > 0:
                rmse = float(np.sqrt(row["mse"])) if row["mse"] else 0.0
                return {
                    "status": "ok",
                    "total_samples": row["samples"],
                    "current_rmse": rmse,
                    "current_mae": float(row["mae"] or 0),
                    "current_comfort": float(row["comfort_percent"] or 0),
                    "avg_offset": float(row["avg_offset"] or 0),
                    "avg_room_temp": float(row["avg_room_temp"] or 0),
                }
            return {"status": "no_data"}
        except Exception as e:
            logger.error("Error getting performance summary: %s", e)
            return {"status": "error", "message": str(e)}

    def cleanup_old_data(self, retention_days: int = 365):
        """Löscht alte Daten."""
        cutoff = datetime.now() - timedelta(days=retention_days)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM measurements WHERE timestamp < ?", (cutoff,)
            )
            cursor.execute(
                "DELETE FROM model_params WHERE timestamp < ?", (cutoff,)
            )
            cursor.execute(
                "DELETE FROM controller_logs WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            cursor.execute("VACUUM")
        logger.info("Database cleanup completed")
