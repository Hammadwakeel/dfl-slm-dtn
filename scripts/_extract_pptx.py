from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
ZIP_PATH = Path(r"C:\Users\Lab System\AppData\Local\Temp\dfl-slides.zip")
OUT = Path(__file__).with_name("_pptx_text.txt")
KEYS = (
    "module",
    "DFL-",
    "topology",
    "NAS",
    "OFA",
    "SLM",
    "Digital Twin",
    "node selection",
    "aggregat",
    "hyperparameter",
    "confusion",
    "waiting",
    "architecture",
    "user",
    "UI",
    "dashboard",
)


def slide_text(xml: bytes) -> str:
    root = ET.fromstring(xml)
    return " ".join((node.text or "").strip() for node in root.iter(f"{NS}t") if (node.text or "").strip())


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = sorted(
            [name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")],
            key=lambda name: int(re.search(r"slide(\d+)", name).group(1)),
        )
        chunks: list[str] = []
        for name in names:
            text = slide_text(archive.read(name))
            if not text:
                continue
            lower = text.lower()
            if any(key.lower() in lower for key in KEYS):
                chunks.append(f"### {name}\n{text}\n")
    OUT.write_text("\n".join(chunks), encoding="utf-8")
    print(f"wrote {OUT} slides_matched={len(chunks)}")


if __name__ == "__main__":
    main()
