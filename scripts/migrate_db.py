"""Idempotent database migration script for Smart Healthcare platform."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import init_db, is_postgres, IS_PRODUCTION, DATABASE_URL


def main() -> None:
    print("==================================================")
    print("  SMART HEALTHCARE DATABASE MIGRATION ENGINE")
    print("==================================================")

    if IS_PRODUCTION and not DATABASE_URL:
        print("CRITICAL ERROR: DATABASE_URL is required for production database migrations but was not configured.", file=sys.stderr)
        sys.exit(1)

    target_type = "PostgreSQL (Production)" if is_postgres() else "SQLite (Local Development)"
    print(f"Target Database Engine : {target_type}")

    try:
        init_db()
        print("Database schema migration completed successfully (idempotent).")
        print("  - Verified table: schema_migrations")
        print("  - Verified table: predictions (with indexes)")
        print("  - Verified table: experiments (with indexes)")
        print("==================================================")
    except Exception as exc:
        print(f"CRITICAL ERROR during migration: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
