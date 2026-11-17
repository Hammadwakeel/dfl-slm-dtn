import pytest

from worker import nas_search


class _Captured(Exception):
    pass


def test_isolated_search_passes_fidelity_epochs_without_architectures(monkeypatch, tmp_path) -> None:
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        captured.append(list(command))
        raise _Captured

    monkeypatch.setattr(nas_search.subprocess, "run", fake_run)
    monkeypatch.setattr(nas_search.subprocess, "Popen", lambda command, **_k: fake_run(command))
    with pytest.raises(_Captured):
        nas_search.search_dataset_architectures_isolated(
            str(tmp_path / "shard.npz"),
            seed=7,
            time_limit_seconds=300,
            node_id="worker1",
            logger=lambda *_a, **_k: None,
            architectures=None,
            epochs=3,
        )
    command = captured[0]
    assert command[-2:] == ["3", "null"]
