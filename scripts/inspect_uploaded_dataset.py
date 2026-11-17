"""Read-only verification of the already-uploaded larger CIFAR dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    request = Request(
        "http://127.0.0.1:8000/api/datasets/cifar_10k_demo",
        headers={
            "Authorization": f"Bearer {settings.api_token}",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        metadata = json.loads(response.read().decode("utf-8"))
    print("metadata_key_names=" + json.dumps(sorted(metadata.keys())))
    print("class_counts=" + json.dumps(metadata.get("class_counts"), sort_keys=True))
    print(f"image_count={metadata.get('image_count')}")
    print(f"expanded_image_bytes={metadata.get('expanded_image_bytes')}")


if __name__ == "__main__":
    main()
