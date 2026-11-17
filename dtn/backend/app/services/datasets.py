"""Safe storage and metadata extraction for user-uploaded image datasets."""

import hashlib
import io
import json
import os
import re
import stat
import tempfile
import warnings
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image

DATASET_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 50_000
MAX_IMAGE_PIXELS = 100_000_000


class DatasetError(RuntimeError):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def validate_dataset_name(name: str) -> str:
    import re

    if len(name) > 64 or re.fullmatch(DATASET_NAME_PATTERN, name) is None:
        raise DatasetError(
            "dataset_name must match ^[A-Za-z0-9][A-Za-z0-9._-]*$ and be at most 64 characters",
            422,
        )
    return name


class DatasetStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def upload_zip(self, dataset_name: str, archive: bytes) -> dict[str, Any]:
        name = validate_dataset_name(dataset_name)
        if not archive:
            raise DatasetError("The uploaded ZIP is empty")
        if len(archive) > MAX_ARCHIVE_BYTES:
            raise DatasetError(f"ZIP exceeds the {MAX_ARCHIVE_BYTES}-byte upload limit", 413)
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / name
        if target.exists():
            raise DatasetError(f"Dataset already exists: {name}", 409)

        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=self.root) as temporary:
            staging = Path(temporary)
            image_root = staging / "images"
            image_root.mkdir()
            metadata = self._extract_and_inspect(dataset_name=name, archive=archive, image_root=image_root)
            metadata["archive_sha256"] = hashlib.sha256(archive).hexdigest()
            metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            (image_root / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            try:
                os.rename(image_root, target)
            except FileExistsError as error:
                raise DatasetError(f"Dataset already exists: {name}", 409) from error
        return metadata

    def upload_folder(
        self,
        dataset_name: str,
        files: list[tuple[str, bytes]],
    ) -> dict[str, Any]:
        if not files:
            raise DatasetError("Folder upload contains no files")
        if len(files) > MAX_ARCHIVE_ENTRIES:
            raise DatasetError(f"Folder contains more than {MAX_ARCHIVE_ENTRIES} files", 413)
        total_bytes = 0
        for relative_path, contents in files:
            self._safe_parts(relative_path)
            total_bytes += len(contents)
            if total_bytes > MAX_ARCHIVE_BYTES:
                raise DatasetError(
                    f"Folder exceeds the {MAX_ARCHIVE_BYTES}-byte upload limit",
                    413,
                )
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zipped:
            for relative_path, contents in files:
                zipped.writestr(relative_path, contents)
        archive = output.getvalue()
        if len(archive) > MAX_ARCHIVE_BYTES:
            raise DatasetError(
                f"Folder archive exceeds the {MAX_ARCHIVE_BYTES}-byte upload limit",
                413,
            )
        return self.upload_zip(dataset_name, archive)

    def get_metadata(self, dataset_name: str) -> dict[str, Any]:
        name = validate_dataset_name(dataset_name)
        path = self.root / name / "metadata.json"
        if not path.is_file():
            raise DatasetError(f"Dataset not found: {name}", 404)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_metadata(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        results = []
        for directory in sorted(self.root.iterdir()):
            if directory.is_dir() and not directory.name.startswith("."):
                metadata_path = directory / "metadata.json"
                if metadata_path.is_file():
                    results.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        return results

    def get_latest_preprocessing_report(self, dataset_name: str) -> dict[str, Any]:
        name = validate_dataset_name(dataset_name)
        dataset_path = self.root / name
        if not dataset_path.is_dir():
            raise DatasetError(f"Dataset not found: {name}", 404)
        processed_path = dataset_path / "processed"
        reports = list(processed_path.glob("run-*/preprocessing_report.json"))
        if not reports:
            raise DatasetError(f"Dataset has not been preprocessed: {name}", 404)
        latest = max(reports, key=lambda report: report.stat().st_mtime_ns)
        report = json.loads(latest.read_text(encoding="utf-8"))
        worker_reports = report.get("workers", {})
        if isinstance(worker_reports, dict):
            for worker_id, worker_report in worker_reports.items():
                if (
                    not isinstance(worker_id, str)
                    or re.fullmatch(r"worker[1-9][0-9]*", worker_id) is None
                    or not isinstance(worker_report, dict)
                ):
                    continue
                shard_path = latest.parent / "workers" / f"{worker_id}.npz"
                if shard_path.is_file():
                    digest = hashlib.sha256()
                    with shard_path.open("rb") as shard:
                        for chunk in iter(lambda: shard.read(1024 * 1024), b""):
                            digest.update(chunk)
                    worker_report["pc_shard_sha256"] = digest.hexdigest()
        return report

    def _extract_and_inspect(
        self,
        *,
        dataset_name: str,
        archive: bytes,
        image_root: Path,
    ) -> dict[str, Any]:
        try:
            zipped = zipfile.ZipFile(io.BytesIO(archive))
        except zipfile.BadZipFile as error:
            raise DatasetError("Upload is not a valid ZIP archive") from error
        with zipped:
            infos = zipped.infolist()
            if len(infos) > MAX_ARCHIVE_ENTRIES:
                raise DatasetError(f"ZIP contains more than {MAX_ARCHIVE_ENTRIES} entries", 413)
            file_infos = []
            advertised_total = 0
            for info in infos:
                parts = self._safe_parts(info.filename)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise DatasetError("ZIP symbolic links are not allowed")
                if info.flag_bits & 0x1:
                    raise DatasetError("Encrypted ZIP entries are not supported")
                if info.is_dir():
                    continue
                advertised_total += info.file_size
                if advertised_total > MAX_EXPANDED_BYTES:
                    raise DatasetError(
                        f"ZIP expands beyond the {MAX_EXPANDED_BYTES}-byte size limit",
                        413,
                    )
                file_infos.append((info, parts))

            image_infos = [
                (info, parts)
                for info, parts in file_infos
                if Path(parts[-1]).suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not image_infos:
                raise DatasetError("ZIP contains no supported image files")

            wrapper = self._common_wrapper(image_infos)
            class_counts: Counter[str] = Counter()
            image_sizes: dict[str, Counter[str]] = defaultdict(Counter)
            image_channels: dict[str, Counter[str]] = defaultdict(Counter)
            actual_total = 0
            for info, original_parts in image_infos:
                parts = original_parts[1:] if wrapper else original_parts
                if len(parts) != 2:
                    raise DatasetError(
                        "Images must be directly inside class subfolders, optionally under one root folder"
                    )
                class_name, filename = parts
                self._validate_class_name(class_name)
                extension = Path(filename).suffix.lower()
                if extension not in IMAGE_EXTENSIONS:
                    continue
                destination_dir = image_root / class_name
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / filename
                if destination.exists():
                    raise DatasetError(f"Duplicate image path in ZIP: {class_name}/{filename}")
                written = 0
                try:
                    with zipped.open(info, "r") as source, destination.open("xb") as output:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            written += len(chunk)
                            actual_total += len(chunk)
                            if actual_total > MAX_EXPANDED_BYTES:
                                raise DatasetError(
                                    f"ZIP expands beyond the {MAX_EXPANDED_BYTES}-byte size limit",
                                    413,
                                )
                            output.write(chunk)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", Image.DecompressionBombWarning)
                        with Image.open(destination) as image:
                            width, height = image.size
                            channels = len(image.getbands())
                            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                                raise DatasetError(
                                    f"Image exceeds the maximum pixel limit: {class_name}/{filename}"
                                )
                            image.verify()
                except DatasetError:
                    raise
                except (
                    OSError,
                    ValueError,
                    zipfile.BadZipFile,
                    Image.DecompressionBombError,
                    Image.DecompressionBombWarning,
                ) as error:
                    raise DatasetError(f"Invalid or corrupt image: {class_name}/{filename}") from error
                if written != info.file_size:
                    raise DatasetError(f"Image byte count mismatch: {class_name}/{filename}")
                class_counts[class_name] += 1
                image_sizes[class_name][f"{width}x{height}"] += 1
                image_channels[class_name][str(channels)] += 1

            if len(class_counts) < 2:
                raise DatasetError("Image datasets must contain at least two non-empty class folders")
            return {
                "dataset_name": dataset_name,
                "classes": sorted(class_counts),
                "class_counts": dict(sorted(class_counts.items())),
                "image_sizes": {
                    class_name: dict(sorted(sizes.items()))
                    for class_name, sizes in sorted(image_sizes.items())
                },
                "image_channels": {
                    class_name: dict(sorted(counts.items()))
                    for class_name, counts in sorted(image_channels.items())
                },
                "image_count": sum(class_counts.values()),
                "expanded_image_bytes": actual_total,
            }

    @staticmethod
    def _safe_parts(raw_name: str) -> tuple[str, ...]:
        if not raw_name or "\x00" in raw_name or "\\" in raw_name:
            raise DatasetError("ZIP contains an unsafe path")
        windows_path = PureWindowsPath(raw_name)
        posix_path = PurePosixPath(raw_name)
        raw_parts = raw_name.rstrip("/").split("/")
        if (
            raw_name.startswith("/")
            or windows_path.is_absolute()
            or windows_path.drive
            or any(part in {"", ".", ".."} for part in raw_parts)
            or posix_path.is_absolute()
        ):
            raise DatasetError("ZIP path traversal or absolute paths are not allowed")
        return tuple(raw_parts)

    @staticmethod
    def _common_wrapper(image_infos: list[tuple[zipfile.ZipInfo, tuple[str, ...]]]) -> bool:
        file_paths = [parts for _, parts in image_infos]
        if not file_paths or any(len(parts) < 3 for parts in file_paths):
            return False
        first = file_paths[0][0]
        return all(parts[0] == first for parts in file_paths)

    @staticmethod
    def _validate_class_name(name: str) -> None:
        if not name or name in {".", ".."} or len(name) > 128:
            raise DatasetError("Invalid class folder name")
        if any(ord(character) < 32 for character in name):
            raise DatasetError("Class folder names cannot contain control characters")
