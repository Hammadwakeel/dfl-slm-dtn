import os

from app.services.dataset_pipeline import prune_processed_runs


def test_only_the_newest_processed_runs_are_kept(tmp_path) -> None:
    processed = tmp_path / "processed"
    for index in range(5):
        run = processed / f"run-{index}"
        run.mkdir(parents=True)
        (run / "train.npz").write_bytes(b"x")
        os.utime(run, ns=(index * 10**9, index * 10**9))

    removed = prune_processed_runs(tmp_path, keep=3)

    assert sorted(removed) == ["run-0", "run-1"]
    assert sorted(path.name for path in processed.iterdir()) == ["run-2", "run-3", "run-4"]


def test_missing_processed_directory_is_a_no_op(tmp_path) -> None:
    assert prune_processed_runs(tmp_path) == []
