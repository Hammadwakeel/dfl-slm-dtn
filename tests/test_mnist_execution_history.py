from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.db.models import Base  # noqa: E402
from app.db.session import create_db_engine, create_session_factory  # noqa: E402
from app.services.autonomy import AutonomyController  # noqa: E402
from scripts.run_mnist_full_access import (  # noqa: E402
    _history_ids,
    _new_automatic_history,
)


class HistoryBackend:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def request(self, method: str, path: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert method == "GET"
        assert path == "/api/autonomy/history"
        return self.rows


def test_driver_history_identity_matches_controller_shape_and_handles_id_collision(
    tmp_path: Path,
) -> None:
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'history.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    controller = AutonomyController(
        create_session_factory(engine),
        lambda action: {"ok": True},
    )
    controller.dispatch(
        {"action": "set_rounds", "parameters": {"rounds": 5}},
        reason="Create an offline ASK_HUMAN decision-history row",
    )
    history_before = controller.history()
    decision_row = next(row for row in history_before if row["event"] == "created")
    assert isinstance(decision_row["id"], int)

    old_ids = _history_ids(HistoryBackend(history_before))
    controller.set_mode("FULL_ACCESS")
    controller.dispatch(
        {"action": "set_rounds", "parameters": {"rounds": 6}},
        reason="Create an offline FULL_ACCESS execution event",
    )
    history_after = controller.history()
    automatic_row = next(
        row for row in history_after if row["event"] == "executed automatically"
    )

    assert automatic_row["mode"] == "FULL_ACCESS"
    assert automatic_row["action"] == "set_rounds"
    assert automatic_row["result_status"] == "executed"
    assert automatic_row["dataset_name"] is None
    assert automatic_row["details"] == {
        "action": "set_rounds",
        "mode": "FULL_ACCESS",
        "result_status": "executed",
        "dataset_name": None,
    }
    assert isinstance(automatic_row["id"], str)

    filtered = _new_automatic_history(
        HistoryBackend(history_after),
        old_ids,
        action="set_rounds",
        deadline=time.monotonic() + 10,
    )
    assert filtered == [automatic_row]

    collision_snapshot = old_ids | {("created", str(automatic_row["id"]))}
    collision_filtered = _new_automatic_history(
        HistoryBackend(history_after),
        collision_snapshot,
        action="set_rounds",
        deadline=time.monotonic() + 10,
    )
    assert collision_filtered == [automatic_row]
    engine.dispose()
