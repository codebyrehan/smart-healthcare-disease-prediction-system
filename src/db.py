"""Database manager supporting PostgreSQL for production and SQLite for local development."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

# Load DATABASE_URL securely without logging credentials
DATABASE_URL = os.getenv("DATABASE_URL")
IS_PRODUCTION = bool(os.getenv("RENDER") or os.getenv("FLASK_ENV") == "production" or os.getenv("ENV") == "production")


def is_postgres() -> bool:
    return bool(DATABASE_URL and (DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")))


def get_connection():
    """Get a database connection based on environment configuration."""
    if is_postgres():
        import psycopg2
        import psycopg2.extras
        # Render provides postgres:// which psycopg2 accepts or needs postgresql://
        dsn = DATABASE_URL
        if dsn.startswith("postgres://"):
            dsn = dsn.replace("postgres://", "postgresql://", 1)
        conn = psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)
        return conn
    else:
        db_path = Path("data/healthcare.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn


def init_db() -> bool:
    """Initialize database tables idempotently."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        if is_postgres():
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(32) PRIMARY KEY,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    prediction_id VARCHAR(64) UNIQUE NOT NULL,
                    model_name VARCHAR(64) NOT NULL,
                    input_params JSONB NOT NULL,
                    prediction INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    risk_label VARCHAR(32) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);

                CREATE TABLE IF NOT EXISTS experiments (
                    id SERIAL PRIMARY KEY,
                    experiment_id VARCHAR(64) UNIQUE NOT NULL,
                    model_name VARCHAR(64) NOT NULL,
                    metrics JSONB NOT NULL,
                    hyperparameters JSONB NOT NULL,
                    dataset_version VARCHAR(64) NOT NULL,
                    feature_count INTEGER NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at DESC);
            """)
        else:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    input_params TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    risk_label TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);

                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    feature_count INTEGER NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at DESC);
            """)
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass
        return False


# Auto-initialize schema on import
try:
    init_db()
except Exception:
    pass


def log_prediction(
    model_name: str,
    input_params: Dict[str, float],
    prediction: int,
    probability: float,
    risk_label: str,
) -> str:
    """Log a validated prediction assessment to persistent storage."""
    prediction_id = f"pred-{uuid.uuid4().hex[:12]}"
    try:
        conn = get_connection()
    except Exception:
        return prediction_id

    try:
        cur = conn.cursor()
        params_str = json.dumps(input_params)
        if is_postgres():
            cur.execute(
                """
                INSERT INTO predictions (prediction_id, model_name, input_params, prediction, probability, risk_label)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (prediction_id, model_name, params_str, prediction, probability, risk_label),
            )
            conn.commit()
        else:
            try:
                cur.execute(
                    """
                    INSERT INTO predictions (prediction_id, model_name, input_params, prediction, probability, risk_label)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (prediction_id, model_name, params_str, prediction, probability, risk_label),
                )
                conn.commit()
            except sqlite3.OperationalError as op_err:
                if "no such table" in str(op_err).lower():
                    # Auto-initialize and retry once
                    init_db()
                    conn2 = get_connection()
                    cur2 = conn2.cursor()
                    cur2.execute(
                        """
                        INSERT INTO predictions (prediction_id, model_name, input_params, prediction, probability, risk_label)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (prediction_id, model_name, params_str, prediction, probability, risk_label),
                    )
                    conn2.commit()
                    conn2.close()
                else:
                    pass
        return prediction_id
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return prediction_id
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_recent_predictions(limit: int = 30) -> List[Dict[str, Any]]:
    """Retrieve recent clinical assessment runs."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        if is_postgres():
            cur.execute(
                """
                SELECT prediction_id, model_name, input_params, prediction, probability, risk_label, created_at
                FROM predictions
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            results = []
            for r in rows:
                results.append({
                    "prediction_id": r["prediction_id"],
                    "model_name": r["model_name"],
                    "input_params": r["input_params"] if isinstance(r["input_params"], dict) else json.loads(r["input_params"]),
                    "prediction": r["prediction"],
                    "probability": round(float(r["probability"]), 4),
                    "risk_label": r["risk_label"],
                    "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
                })
            conn.close()
            return results
        else:
            cur.execute(
                """
                SELECT prediction_id, model_name, input_params, prediction, probability, risk_label, created_at
                FROM predictions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
            results = []
            for r in rows:
                params = json.loads(r["input_params"]) if isinstance(r["input_params"], str) else r["input_params"]
                results.append({
                    "prediction_id": r["prediction_id"],
                    "model_name": r["model_name"],
                    "input_params": params,
                    "prediction": r["prediction"],
                    "probability": round(float(r["probability"]), 4),
                    "risk_label": r["risk_label"],
                    "created_at": str(r["created_at"]),
                })
            conn.close()
            return results
    except Exception:
        return []


def log_experiment(
    model_name: str,
    metrics: Dict[str, float],
    hyperparameters: Dict[str, Any],
    dataset_version: str = "PIMA-v1.0",
    feature_count: int = 8,
) -> str:
    """Log an ML experiment training record."""
    experiment_id = f"exp-{uuid.uuid4().hex[:8]}"
    try:
        conn = get_connection()
        cur = conn.cursor()
        metrics_json = json.dumps(metrics)
        params_json = json.dumps(hyperparameters)
        if is_postgres():
            cur.execute(
                """
                INSERT INTO experiments (experiment_id, model_name, metrics, hyperparameters, dataset_version, feature_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (experiment_id, model_name, metrics_json, params_json, dataset_version, feature_count),
            )
        else:
            cur.execute(
                """
                INSERT INTO experiments (experiment_id, model_name, metrics, hyperparameters, dataset_version, feature_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (experiment_id, model_name, metrics_json, params_json, dataset_version, feature_count),
            )
        conn.commit()
        conn.close()
        return experiment_id
    except Exception:
        return experiment_id
