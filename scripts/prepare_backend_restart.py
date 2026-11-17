"""Verify restart preconditions, back up SQLite, and upgrade to Alembic head."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "dtn" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import Settings  # noqa: E402


class RestartPreparationError(RuntimeError):
    pass


def _autonomy_state(settings: Settings) -> dict[str, object]:
    request = Request(
        f"http://127.0.0.1:{settings.port}/api/autonomy/state",
        headers={
            "Authorization": f"Bearer {settings.api_token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            import json

            state = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RestartPreparationError(
            f"Authenticated autonomy check returned HTTP {error.code}"
        ) from error
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, ValueError) as error:
        raise RestartPreparationError(
            f"Authenticated autonomy check failed: {type(error).__name__}"
        ) from error
    if not isinstance(state, dict):
        raise RestartPreparationError("Autonomy state response was not an object")
    if state.get("stop_requested") is not False:
        raise RestartPreparationError("Global STOP must be inactive before restart")
    if state.get("pending_decisions") != 0:
        raise RestartPreparationError("Pending decisions must be resolved before restart")
    if any(loop.get("status") == "RUNNING" for loop in state.get("loops", [])):
        raise RestartPreparationError("Running autonomy loops must finish before restart")
    if state.get("mode") != "ASK_HUMAN":
        raise RestartPreparationError("Autonomy mode must be ASK_HUMAN before restart")
    print(
        "pre-restart autonomy state: "
        f"mode={state['mode']}; global_STOP={state['stop_requested']}; "
        f"pending_decisions={state['pending_decisions']}; running_loops=0"
    )
    return state


def _database_file(database_url: str) -> Path:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        raise RestartPreparationError("Expected a file-backed SQLite database")
    path = Path(url.database)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.is_file():
        raise RestartPreparationError(f"Configured SQLite database is missing: {path}")
    return path


def _backup_database(database_path: Path) -> Path:
    suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = database_path.with_name(f"{database_path.name}.bak-{suffix}")
    if backup_path.exists():
        raise RestartPreparationError(f"Refusing to overwrite backup: {backup_path}")
    with sqlite3.connect(database_path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
    if not backup_path.is_file() or backup_path.stat().st_size == 0:
        raise RestartPreparationError("SQLite backup was not created correctly")
    print(f"SQLite backup created: {backup_path}")
    return backup_path


def _upgrade_database(database_url: str) -> str:
    engine = create_engine(database_url)
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    connection = engine.connect()
    config.attributes["connection"] = connection
    try:
        command.upgrade(config, "head")
        revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
    finally:
        connection.close()
        engine.dispose()
    if not isinstance(revision, str):
        raise RestartPreparationError("Alembic did not report a revision")
    print(f"Alembic upgrade head complete; current revision={revision}")
    return revision


def main() -> None:
    settings = Settings.from_env()
    _autonomy_state(settings)
    database_path = _database_file(settings.database_url)
    _backup_database(database_path)
    _upgrade_database(settings.database_url)


if __name__ == "__main__":
    main()
