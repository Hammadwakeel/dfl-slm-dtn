"""Report TASK K fallback configuration and the existing MNIST model path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    model_path = ROOT / "data" / "datasets" / "mnist_demo" / "models" / "initial_model"
    enabled = settings.feedback_rule_fallback_enabled
    print(f"feedback_rule_fallback_enabled={enabled}")
    print(f"mnist_demo_initial_model_exists={model_path.is_dir()}")
    if enabled is not False:
        raise RuntimeError("Feedback rule fallback must be False before TASK K")


if __name__ == "__main__":
    main()
