"""
Datenbank & Logging
===================
SQLite-basierte Speicherung aller Messdaten für Analyse und Modelltraining.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

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
    mode: str  # 'control', 'experiment', 'manual'


class Database:
    """
    SQLite Datenbank für Heizungsdaten.
    
    Tabellen:
    - measurements: Alle Sensordaten + Stellgrößen
    - experiments: Metadaten zu Experimenten
    - model_params: Gespeicherte Modellparameter
    """
    
    def __init__(self, db_path: str = "data/measurements.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Erstellt Tabellen falls nicht vorhanden."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Messungen
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
            
            # Index für schnelle Zeitabfragen
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_measurements_timestamp 
                ON measurements(timestamp)
            """)
            
            # Experimente
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
            
            # Modellparameter (für adaptives Modell)
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
            
            # Controller-Logs
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
            
            # RL/KI Training Samples - komplette Transitions für Replay Buffer
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    
                    -- State (Zustand)
                    room_temp REAL NOT NULL,
                    outside_temp REAL,
                    target_temp REAL NOT NULL,
                    window_open INTEGER NOT NULL,
                    previous_temp REAL,
                    previous_offset REAL,
                    heating_active INTEGER,
                    
                    -- Action (Aktion)
                    offset_action REAL NOT NULL,
                    
                    -- Reward
                    reward REAL NOT NULL,
                    
                    -- Zusätzliche Infos
                    model_params TEXT,  -- JSON mit aktuellen Modellparametern
                    
                    -- Next State kommt vom nächsten Sample
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Index für Training-Queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_samples_timestamp 
                ON training_samples(timestamp)
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def _connect(self) -> sqlite3.Connection:
        """Erstellt Datenbankverbindung."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # -------------------------------------------------------------------------
    # Measurements
    # -------------------------------------------------------------------------
    
    def insert_measurement(self, measurement: Measurement):
        """Fügt eine Messung ein."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO measurements 
                (timestamp, room_temp, outside_temp, window_open, 
                 heating_active, control_offset, target_temp, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                measurement.timestamp,
                measurement.room_temp,
                measurement.outside_temp,
                1 if measurement.window_open else 0,
                1 if measurement.heating_active else (0 if measurement.heating_active is False else None),
                measurement.control_offset,
                measurement.target_temp,
                measurement.mode,
            ))
            conn.commit()
    
    def get_measurements(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        mode: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Holt Messungen als DataFrame.
        
        Args:
            start_time: Frühester Zeitpunkt
            end_time: Spätester Zeitpunkt
            mode: Nur bestimmten Modus
            limit: Maximale Anzahl (neueste zuerst)
        """
        query = "SELECT * FROM measurements WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
        
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def get_recent_measurements(self, hours: int = 24) -> pd.DataFrame:
        """Holt Messungen der letzten N Stunden."""
        start_time = datetime.now() - timedelta(hours=hours)
        return self.get_measurements(start_time=start_time)
    
    def get_measurement_stats(self) -> Dict[str, Any]:
        """Statistiken über die Datenbank."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM measurements")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM measurements")
            row = cursor.fetchone()
            
            cursor.execute("""
                SELECT mode, COUNT(*) FROM measurements GROUP BY mode
            """)
            mode_counts = dict(cursor.fetchall())
        
        return {
            'total_measurements': total,
            'first_timestamp': row[0],
            'last_timestamp': row[1],
            'measurements_by_mode': mode_counts,
        }
    
    # -------------------------------------------------------------------------
    # Experiments
    # -------------------------------------------------------------------------
    
    def start_experiment(
        self, 
        name: str, 
        exp_type: str, 
        parameters: Dict
    ) -> int:
        """
        Startet ein neues Experiment.
        
        Returns:
            Experiment ID
        """
        import json
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (name, type, start_time, parameters, status)
                VALUES (?, ?, ?, ?, 'running')
            """, (name, exp_type, datetime.now(), json.dumps(parameters)))
            conn.commit()
            return cursor.lastrowid
    
    def end_experiment(self, exp_id: int, results: Dict):
        """Beendet ein Experiment und speichert Ergebnisse."""
        import json
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments 
                SET end_time = ?, status = 'completed', results = ?
                WHERE id = ?
            """, (datetime.now(), json.dumps(results), exp_id))
            conn.commit()
    
    def get_experiments(self, exp_type: Optional[str] = None) -> pd.DataFrame:
        """Holt alle Experimente."""
        query = "SELECT * FROM experiments"
        params = []
        
        if exp_type:
            query += " WHERE type = ?"
            params.append(exp_type)
        
        query += " ORDER BY start_time DESC"
        
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params, parse_dates=['start_time', 'end_time'])
    
    # -------------------------------------------------------------------------
    # Model Parameters
    # -------------------------------------------------------------------------
    
    def save_model_params(self, params: Dict[str, Tuple[float, float]]):
        """
        Speichert Modellparameter.
        
        Args:
            params: Dict von param_name -> (value, confidence)
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            now = datetime.now()
            
            for name, (value, confidence) in params.items():
                cursor.execute("""
                    INSERT INTO model_params (timestamp, param_name, param_value, confidence)
                    VALUES (?, ?, ?, ?)
                """, (now, name, value, confidence))
            
            conn.commit()
    
    def get_latest_model_params(self) -> Dict[str, Tuple[float, float]]:
        """Holt die neuesten Modellparameter."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT param_name, param_value, confidence
                FROM model_params mp1
                WHERE timestamp = (
                    SELECT MAX(timestamp) FROM model_params mp2
                    WHERE mp2.param_name = mp1.param_name
                )
            """)
            
            return {
                row['param_name']: (row['param_value'], row['confidence'])
                for row in cursor.fetchall()
            }
    
    def get_model_param_history(
        self, 
        param_name: str, 
        hours: int = 24
    ) -> pd.DataFrame:
        """Holt Verlauf eines Modellparameters."""
        start_time = datetime.now() - timedelta(hours=hours)
        
        with self._connect() as conn:
            return pd.read_sql_query("""
                SELECT timestamp, param_value, confidence
                FROM model_params
                WHERE param_name = ? AND timestamp >= ?
                ORDER BY timestamp
            """, conn, params=(param_name, start_time), parse_dates=['timestamp'])
    
    # -------------------------------------------------------------------------
    # Controller Logs
    # -------------------------------------------------------------------------
    
    def log_controller_step(
        self,
        controller_type: str,
        predicted_temps: List[float],
        optimal_offsets: List[float],
        cost_value: float,
        solve_time_ms: float,
    ):
        """Loggt einen Controller-Optimierungsschritt."""
        import json
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO controller_logs 
                (timestamp, controller_type, predicted_temps, optimal_offsets, 
                 cost_value, solve_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                controller_type,
                json.dumps(predicted_temps),
                json.dumps(optimal_offsets),
                cost_value,
                solve_time_ms,
            ))
            conn.commit()
    
    # -------------------------------------------------------------------------
    # Training Samples (für KI/RL)
    # -------------------------------------------------------------------------
    
    def log_training_sample(
        self,
        timestamp: datetime,
        state: Dict[str, Any],
        action: Dict[str, float],
        reward: float,
        model_params: Optional[Dict] = None,
    ):
        """
        Loggt ein Training-Sample für RL/KI.
        
        Args:
            timestamp: Zeitstempel
            state: Zustandsdict (room_temp, outside_temp, target_temp, etc.)
            action: Aktionsdict (offset)
            reward: Berechneter Reward
            model_params: Aktuelle Modellparameter (optional)
        """
        import json
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_samples 
                (timestamp, room_temp, outside_temp, target_temp, window_open,
                 previous_temp, previous_offset, heating_active,
                 offset_action, reward, model_params)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                state.get('room_temp'),
                state.get('outside_temp'),
                state.get('target_temp'),
                1 if state.get('window_open') else 0,
                state.get('previous_temp'),
                state.get('previous_offset'),
                1 if state.get('heating_active') else (0 if state.get('heating_active') is False else None),
                action.get('offset', 0),
                reward,
                json.dumps(model_params) if model_params else None,
            ))
            conn.commit()
    
    def get_training_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        exclude_window_open: bool = False,
    ) -> pd.DataFrame:
        """
        Holt Training-Daten als DataFrame.
        
        Args:
            start_time: Frühester Zeitpunkt
            end_time: Spätester Zeitpunkt
            exclude_window_open: Fenster-offen Samples ausschließen
            
        Returns:
            DataFrame mit allen Training-Samples
        """
        query = "SELECT * FROM training_samples WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if exclude_window_open:
            query += " AND window_open = 0"
        
        query += " ORDER BY timestamp"
        
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
        
        return df
    
    def export_training_data(
        self,
        output_path: str = "data/training_data.parquet",
        format: str = "parquet",
    ) -> str:
        """
        Exportiert Training-Daten für KI-Training.
        
        Args:
            output_path: Ausgabepfad
            format: "parquet", "csv" oder "pickle"
            
        Returns:
            Pfad zur exportierten Datei
        """
        df = self.get_training_data()
        
        if len(df) == 0:
            logger.warning("No training data to export")
            return ""
        
        # Zusätzliche Features berechnen
        df['temp_error'] = df['room_temp'] - df['target_temp']
        df['temp_change'] = df['room_temp'].diff()
        df['time_of_day'] = pd.to_datetime(df['timestamp']).dt.hour + \
                           pd.to_datetime(df['timestamp']).dt.minute / 60
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Next state für Transitions
        df['next_room_temp'] = df['room_temp'].shift(-1)
        df['next_outside_temp'] = df['outside_temp'].shift(-1)
        df['done'] = df['window_open'].shift(-1).fillna(0).astype(bool)  # Episode endet bei Fenster
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "parquet":
            df.to_parquet(output, index=False)
        elif format == "csv":
            df.to_csv(output, index=False)
        elif format == "pickle":
            df.to_pickle(output)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Exported {len(df)} training samples to {output}")
        return str(output)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Statistiken über Training-Daten."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM training_samples")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM training_samples WHERE window_open = 0")
            valid = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT AVG(reward), MIN(reward), MAX(reward), 
                       AVG(room_temp), AVG(target_temp), AVG(offset_action)
                FROM training_samples WHERE window_open = 0
            """)
            row = cursor.fetchone()
        
        return {
            'total_samples': total,
            'valid_samples': valid,
            'window_open_samples': total - valid,
            'avg_reward': row[0],
            'min_reward': row[1],
            'max_reward': row[2],
            'avg_room_temp': row[3],
            'avg_target_temp': row[4],
            'avg_offset': row[5],
        }
    
    # -------------------------------------------------------------------------
    # Regelungs-Statistiken & Performance
    # -------------------------------------------------------------------------
    
    def get_daily_statistics(self, days: int = 7) -> pd.DataFrame:
        """
        Berechnet tägliche Statistiken zur Regelungsperformance.
        
        Returns:
            DataFrame mit RMSE, MAE, Komfort-Score etc. pro Tag
        """
        start_time = datetime.now() - timedelta(days=days)
        
        with self._connect() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as samples,
                    AVG(room_temp) as avg_room_temp,
                    AVG(target_temp) as avg_target_temp,
                    AVG(ABS(room_temp - target_temp)) as mae,
                    AVG((room_temp - target_temp) * (room_temp - target_temp)) as mse,
                    SUM(CASE WHEN window_open = 0 THEN 1 ELSE 0 END) as valid_samples,
                    SUM(CASE WHEN window_open = 1 THEN 1 ELSE 0 END) as window_open_samples,
                    AVG(control_offset) as avg_offset,
                    MIN(room_temp) as min_room_temp,
                    MAX(room_temp) as max_room_temp,
                    SUM(CASE WHEN ABS(room_temp - target_temp) < 0.5 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as comfort_percent
                FROM measurements 
                WHERE timestamp >= ? AND window_open = 0
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, conn, params=(start_time,), parse_dates=['date'])
        
        # RMSE berechnen
        if len(df) > 0:
            df['rmse'] = np.sqrt(df['mse'])
            df = df.drop(columns=['mse'])
        
        return df
    
    def get_hourly_statistics(self, hours: int = 24) -> pd.DataFrame:
        """
        Berechnet stündliche Statistiken.
        
        Returns:
            DataFrame mit Statistiken pro Stunde
        """
        start_time = datetime.now() - timedelta(hours=hours)
        
        with self._connect() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    STRFTIME('%Y-%m-%d %H:00', timestamp) as hour,
                    COUNT(*) as samples,
                    AVG(room_temp) as avg_room_temp,
                    AVG(target_temp) as avg_target_temp,
                    AVG(ABS(room_temp - target_temp)) as mae,
                    SQRT(AVG((room_temp - target_temp) * (room_temp - target_temp))) as rmse,
                    AVG(control_offset) as avg_offset,
                    AVG(outside_temp) as avg_outside_temp
                FROM measurements 
                WHERE timestamp >= ? AND window_open = 0
                GROUP BY STRFTIME('%Y-%m-%d %H:00', timestamp)
                ORDER BY hour DESC
            """, conn, params=(start_time,), parse_dates=['hour'])
        
        return df
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Gibt eine Zusammenfassung der Regelungsperformance zurück.
        
        Returns:
            Dict mit Gesamt-RMSE, Trend, beste/schlechteste Tage etc.
        """
        daily = self.get_daily_statistics(days=30)
        
        if len(daily) == 0:
            return {'status': 'no_data', 'message': 'Noch keine Daten vorhanden'}
        
        # Letzte 7 Tage vs. vorherige 7 Tage für Trend
        recent = daily.head(7)
        previous = daily.iloc[7:14] if len(daily) > 7 else pd.DataFrame()
        
        summary = {
            'total_days': len(daily),
            'total_samples': int(daily['samples'].sum()),
            
            # Aktuelle Performance (letzte 7 Tage)
            'current_rmse': float(recent['rmse'].mean()) if len(recent) > 0 else None,
            'current_mae': float(recent['mae'].mean()) if len(recent) > 0 else None,
            'current_comfort': float(recent['comfort_percent'].mean()) if len(recent) > 0 else None,
            
            # Vorherige Performance (7-14 Tage)
            'previous_rmse': float(previous['rmse'].mean()) if len(previous) > 0 else None,
            
            # Trend
            'trend': None,
            'trend_percent': None,
            
            # Bester/Schlechtester Tag
            'best_day': None,
            'best_day_rmse': None,
            'worst_day': None,
            'worst_day_rmse': None,
            
            # Gesamtstatistik
            'avg_offset': float(daily['avg_offset'].mean()),
            'avg_room_temp': float(daily['avg_room_temp'].mean()),
        }
        
        # Trend berechnen
        if summary['current_rmse'] and summary['previous_rmse']:
            change = summary['previous_rmse'] - summary['current_rmse']
            change_percent = (change / summary['previous_rmse']) * 100
            summary['trend'] = 'improving' if change > 0.01 else ('worsening' if change < -0.01 else 'stable')
            summary['trend_percent'] = float(change_percent)
        
        # Bester/Schlechtester Tag
        if len(daily) > 0:
            best_idx = daily['rmse'].idxmin()
            worst_idx = daily['rmse'].idxmax()
            summary['best_day'] = str(daily.loc[best_idx, 'date'].date())
            summary['best_day_rmse'] = float(daily.loc[best_idx, 'rmse'])
            summary['worst_day'] = str(daily.loc[worst_idx, 'date'].date())
            summary['worst_day_rmse'] = float(daily.loc[worst_idx, 'rmse'])
        
        return summary
    
    def get_experiment_results(self) -> pd.DataFrame:
        """
        Holt alle Experiment-Ergebnisse mit Details.
        
        Returns:
            DataFrame mit allen Experimenten und ihren Ergebnissen
        """
        with self._connect() as conn:
            df = pd.read_sql_query("""
                SELECT 
                    id,
                    name,
                    type,
                    start_time,
                    end_time,
                    status,
                    parameters,
                    results,
                    ROUND((JULIANDAY(end_time) - JULIANDAY(start_time)) * 24 * 60, 1) as duration_minutes
                FROM experiments
                ORDER BY start_time DESC
            """, conn, parse_dates=['start_time', 'end_time'])
        
        return df
    
    def print_statistics_report(self, days: int = 7):
        """
        Druckt einen formatierten Statistik-Report.
        """
        print("\n" + "═" * 70)
        print("📊 REGELUNGS-STATISTIK".center(70))
        print("═" * 70 + "\n")
        
        summary = self.get_performance_summary()
        
        if summary.get('status') == 'no_data':
            print("  ⚠️  Noch keine Daten vorhanden.")
            print("      Starte zuerst die Regelung mit: python main.py\n")
            return
        
        # Trend-Symbol
        if summary['trend'] == 'improving':
            trend_icon = "📈 VERBESSERT"
            trend_color = "✅"
        elif summary['trend'] == 'worsening':
            trend_icon = "📉 VERSCHLECHTERT"
            trend_color = "⚠️"
        else:
            trend_icon = "➡️  STABIL"
            trend_color = "ℹ️"
        
        print("Performance-Übersicht (letzte 7 Tage):")
        print("─" * 70)
        
        rmse = summary['current_rmse']
        mae = summary['current_mae']
        comfort = summary['current_comfort']
        
        # Bewertung des RMSE
        if rmse is not None:
            if rmse < 0.3:
                rmse_rating = "🌟 Exzellent"
            elif rmse < 0.5:
                rmse_rating = "✅ Gut"
            elif rmse < 0.8:
                rmse_rating = "⚠️ Verbesserungswürdig"
            else:
                rmse_rating = "❌ Schlecht"
        else:
            rmse_rating = "?"
        
        print(f"""
  RMSE (Wurzel mittlerer quadr. Fehler):  {rmse:.3f}°C  {rmse_rating}
  MAE (Mittlerer absoluter Fehler):       {mae:.3f}°C
  Komfort-Quote (±0.5°C vom Sollwert):    {comfort:.1f}%
  
  Durchschnittlicher Offset:              {summary['avg_offset']:.2f}°C
  Durchschnittliche Raumtemperatur:       {summary['avg_room_temp']:.1f}°C
""" if rmse is not None else "  Keine Daten verfügbar\n")
        
        # Trend
        print("Trend:")
        print("─" * 70)
        if summary['trend']:
            trend_pct = summary['trend_percent']
            print(f"  {trend_color} {trend_icon}")
            if trend_pct:
                print(f"     RMSE-Änderung: {trend_pct:+.1f}% gegenüber Vorwoche")
        else:
            print("  ℹ️  Noch nicht genug Daten für Trendanalyse")
        
        # Beste/Schlechteste Tage
        print("\nHighlights:")
        print("─" * 70)
        if summary['best_day']:
            print(f"  🏆 Bester Tag:      {summary['best_day']} (RMSE: {summary['best_day_rmse']:.3f}°C)")
            print(f"  😓 Schlechtester:   {summary['worst_day']} (RMSE: {summary['worst_day_rmse']:.3f}°C)")
        
        # Tagesübersicht
        daily = self.get_daily_statistics(days=days)
        
        if len(daily) > 0:
            print(f"\nTägliche Übersicht (letzte {days} Tage):")
            print("─" * 70)
            print(f"{'Datum':<12} {'RMSE':>8} {'MAE':>8} {'Komfort':>9} {'Samples':>8} {'Offset':>8}")
            print("─" * 70)
            
            for _, row in daily.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
                rmse_bar = "█" * min(int(row['rmse'] * 10), 10) if row['rmse'] else ""
                print(f"{date_str:<12} {row['rmse']:>7.3f}° {row['mae']:>7.3f}° {row['comfort_percent']:>8.1f}% {int(row['samples']):>8} {row['avg_offset']:>+7.2f}°")
        
        print("\n" + "═" * 70)
        print("💡 Tipp: RMSE < 0.3°C = sehr gute Regelung")
        print("═" * 70 + "\n")
    
    def print_experiment_report(self):
        """
        Druckt einen formatierten Experiment-Report.
        """
        import json
        
        print("\n" + "═" * 70)
        print("🔬 EXPERIMENT-ERGEBNISSE".center(70))
        print("═" * 70 + "\n")
        
        df = self.get_experiment_results()
        
        if len(df) == 0:
            print("  ⚠️  Noch keine Experimente durchgeführt.")
            print("      Starte ein Experiment mit: python main.py --experiment step\n")
            return
        
        for _, row in df.iterrows():
            status_icon = "✅" if row['status'] == 'completed' else ("⏳" if row['status'] == 'running' else "❌")
            
            print(f"┌{'─' * 68}┐")
            print(f"│ {status_icon} {row['name']:<63} │")
            print(f"├{'─' * 68}┤")
            
            start = row['start_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['start_time']) else '?'
            end = row['end_time'].strftime('%H:%M') if pd.notna(row['end_time']) else '?'
            duration = f"{row['duration_minutes']:.0f} min" if pd.notna(row['duration_minutes']) else '?'
            
            print(f"│  Typ:     {row['type']:<57} │")
            print(f"│  Zeit:    {start} - {end} ({duration}){' ' * (38 - len(duration))} │")
            print(f"│  Status:  {row['status']:<57} │")
            
            # Ergebnisse parsen und anzeigen
            if row['results'] and row['status'] == 'completed':
                try:
                    results = json.loads(row['results'])
                    print(f"├{'─' * 68}┤")
                    print(f"│  Ergebnisse:{' ' * 55} │")
                    
                    if 'identified_params' in results and results['identified_params']:
                        params = results['identified_params']
                        if 'tau' in params:
                            print(f"│    τ (Zeitkonstante):   {params['tau']:.1f} min{' ' * 37} │")
                        if 'k_heater' in params:
                            print(f"│    k_heater (Gain):     {params['k_heater']:.3f}{' ' * 40} │")
                        if 'k_outside' in params:
                            print(f"│    k_outside:           {params['k_outside']:.4f}{' ' * 39} │")
                    
                    if 'metrics' in results and results['metrics']:
                        metrics = results['metrics']
                        if 'T_initial' in metrics:
                            print(f"│    T_initial:           {metrics['T_initial']:.1f}°C{' ' * 40} │")
                        if 'T_final' in metrics:
                            print(f"│    T_final:             {metrics['T_final']:.1f}°C{' ' * 40} │")
                        if 'T_change' in metrics:
                            print(f"│    ΔT:                  {metrics['T_change']:+.1f}°C{' ' * 40} │")
                    
                    if 'Ku' in results:  # Relay-Feedback
                        print(f"│    Ku (Ultimate Gain):  {results['Ku']:.2f}{' ' * 41} │")
                        print(f"│    Tu (Period):         {results['Tu']:.0f} min{' ' * 38} │")
                        
                except json.JSONDecodeError:
                    pass
            
            print(f"└{'─' * 68}┘\n")
        
        print("═" * 70)
        print("💡 Experiment-Parameter werden automatisch ins Modell übernommen")
        print("═" * 70 + "\n")
    
    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------
    
    def cleanup_old_data(self, retention_days: int = 365):
        """Löscht alte Daten."""
        cutoff = datetime.now() - timedelta(days=retention_days)
        
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM measurements WHERE timestamp < ?", (cutoff,))
            deleted_measurements = cursor.rowcount
            
            cursor.execute("DELETE FROM model_params WHERE timestamp < ?", (cutoff,))
            deleted_params = cursor.rowcount
            
            cursor.execute("DELETE FROM controller_logs WHERE timestamp < ?", (cutoff,))
            deleted_logs = cursor.rowcount
            
            conn.commit()
            
            # Vacuum um Speicher freizugeben
            cursor.execute("VACUUM")
        
        logger.info(
            f"Cleanup: deleted {deleted_measurements} measurements, "
            f"{deleted_params} model params, {deleted_logs} controller logs"
        )
    
    def export_to_csv(self, output_dir: str = "data/export"):
        """Exportiert alle Tabellen als CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tables = ['measurements', 'experiments', 'model_params', 'controller_logs']
        
        with self._connect() as conn:
            for table in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                df.to_csv(output_path / f"{table}.csv", index=False)
                logger.info(f"Exported {table} to CSV ({len(df)} rows)")


# -----------------------------------------------------------------------------
# Hilfsfunktionen für Datenanalyse
# -----------------------------------------------------------------------------

def prepare_training_data(db: Database, hours: int = 168) -> pd.DataFrame:
    """
    Bereitet Daten für Modelltraining vor.
    
    Returns:
        DataFrame mit Features und Targets für Systemidentifikation.
    """
    df = db.get_recent_measurements(hours=hours)
    
    if len(df) < 10:
        raise ValueError("Not enough data for training")
    
    # Zeitliche Differenz berechnen
    df['dt'] = df['timestamp'].diff().dt.total_seconds() / 60  # in Minuten
    
    # Temperaturänderung
    df['dT'] = df['room_temp'].diff()
    
    # Änderungsrate
    df['dT_dt'] = df['dT'] / df['dt']
    
    # Lagged Features
    df['room_temp_lag1'] = df['room_temp'].shift(1)
    df['offset_lag1'] = df['control_offset'].shift(1)
    
    # Entferne NaN
    df = df.dropna()
    
    return df


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.DEBUG)
    
    db = Database("data/test.db")
    
    # Test Measurement
    m = Measurement(
        timestamp=datetime.now(),
        room_temp=20.5,
        outside_temp=5.0,
        window_open=False,
        heating_active=True,
        control_offset=-2.0,
        target_temp=21.0,
        mode='control',
    )
    
    db.insert_measurement(m)
    
    df = db.get_recent_measurements(hours=1)
    print(df)
    
    print(db.get_measurement_stats())
