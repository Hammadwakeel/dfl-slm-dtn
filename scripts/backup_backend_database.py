"""Create and integrity-check an online SQLite backup of the backend database."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from app.core.config import Settings

    settings = Settings.from_env()
    url = make_url(settings.database_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        raise RuntimeError("Expected a file-backed SQLite database")

    database_path = Path(url.database)
    if not database_path.is_absolute():
        database_path = (ROOT / database_path).resolve()
    if not database_path.is_file():
        raise RuntimeError(f"Configured SQLite database is missing: {database_path}")

    suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = database_path.with_name(f"{database_path.name}.bak-{suffix}")
    if backup_path.exists():
        raise RuntimeError(f"Refusing to overwrite backup: {backup_path}")

    with sqlite3.connect(database_path, timeout=30) as source:
        with sqlite3.connect(backup_path) as target:
            source.backup(target)
            integrity = target.execute("PRAGMA integrity_check").fetchone()

    if integrity != ("ok",):
        raise RuntimeError(f"SQLite backup integrity check failed: {integrity!r}")
    print(f"SQLite backup verified: {backup_path}")


if __name__ == "__main__":
    main()