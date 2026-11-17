"""Verify installed MNIST initial-model metadata on all configured workers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402
from app.services.worker_shards import get_worker_dataset_initial_model  # noqa: E402


EXPECTED_ARCHITECTURE_SHA256 = "b66616d37317a211cd9dc2cb8d02b6213cfc292d049a2258dc7711d1e7f73bb2"
EXPECTED_WEIGHTS_SHA256 = "d23e3d771bb065c347655ac932ab90cc3a8ce12ea01345c533ffca004be78d75"
EXPECTED_WORKERS = ("worker1", "worker2", "worker3")


def main() -> None:
    settings = Settings.from_env()
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    if tuple(node.node_id for node in workers) != EXPECTED_WORKERS:
        raise RuntimeError("Expected exactly worker1, worker2, and worker3")

    for worker in workers:
        model = get_worker_dataset_initial_model(worker, "mnist_demo")
        architecture_hash = model.get("architecture_sha256")
        weights_hash = model.get("weights_sha256")
        print(
            f"{worker.node_id}: architecture_sha256={architecture_hash}; "
            f"weights_sha256={weights_hash}"
        )
        if (
            architecture_hash != EXPECTED_ARCHITECTURE_SHA256
            or weights_hash != EXPECTED_WEIGHTS_SHA256
        ):
            raise RuntimeError(f"{worker.node_id} MNIST initial-model hash mismatch")


if __name__ == "__main__":
    main()