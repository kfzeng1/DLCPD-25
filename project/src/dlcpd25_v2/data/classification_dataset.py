"""DLCPD-25 classification dataset backed by the frozen manifest contract."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler

from dlcpd25_v2.common import repo_root

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ClassificationSample:
    path: str
    class_id: int
    host_id: int
    category_id: int


def _load_taxonomy(taxonomy_path: Path) -> dict[str, Any]:
    with taxonomy_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    classes = payload["classes"]
    hosts = payload["hosts"]
    categories = payload["categories"]
    host_index = {str(host["id"]): idx for idx, host in enumerate(hosts)}
    category_index = {str(item["id"]): idx for idx, item in enumerate(categories)}
    class_map: dict[int, tuple[int, int, str, str]] = {}
    for item in classes:
        class_id = int(item["class_id"])
        class_map[class_id] = (
            host_index[str(item["host_id"])],
            category_index[str(item["category"])],
            str(item["official_name"]),
            str(item["local_directory"]),
        )
    return {
        "class_map": class_map,
        "host_names": [str(host["name_zh"]) for host in hosts],
        "category_names": [str(item["name_zh"]) for item in categories],
    }


def load_manifest(manifest_path: Path, split: str) -> list[ClassificationSample]:
    opener = gzip.open if str(manifest_path).endswith(".gz") else open
    samples: list[ClassificationSample] = []
    root = repo_root()
    with opener(manifest_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("decode_status") != "ok" or row.get("split") != split:
                continue
            class_id = int(row["class_id"])
            samples.append(
                ClassificationSample(
                    path=str((root / row["relative_path"]).resolve()),
                    class_id=class_id,
                    host_id=class_id,  # filled below after taxonomy is loaded
                    category_id=class_id,
                )
            )
    return samples


def build_taxonomy(taxonomy_path: Path) -> tuple[dict[int, tuple[int, int, str, str]], list[str], list[str]]:
    payload = _load_taxonomy(taxonomy_path)
    return payload["class_map"], payload["host_names"], payload["category_names"]


class ManifestClassificationDataset(Dataset[tuple[Any, int, int, int]]):
    """Reads images listed in the frozen ``artifacts/data/dlcpd25/manifest.csv.gz``."""

    def __init__(
        self,
        manifest_path: Path | str,
        taxonomy_path: Path | str,
        split: str,
        transform: Any,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.taxonomy_path = Path(taxonomy_path)
        self.split = split
        self.transform = transform
        self.class_map, self.host_names, self.category_names = build_taxonomy(self.taxonomy_path)
        self.samples = load_manifest(self.manifest_path, split)
        if not self.samples:
            raise ValueError(f"manifest contains no usable images for split={split!r}")
        # Fill taxonomy ids into samples.
        filled: list[ClassificationSample] = []
        for sample in self.samples:
            host_id, category_id, _, _ = self.class_map[sample.class_id]
            filled.append(
                ClassificationSample(
                    path=sample.path,
                    class_id=sample.class_id,
                    host_id=host_id,
                    category_id=category_id,
                )
            )
        self.samples = filled
        self._class_counts: dict[int, int] = {}
        for sample in self.samples:
            self._class_counts[sample.class_id] = self._class_counts.get(sample.class_id, 0) + 1

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        try:
            with Image.open(path) as image:
                try:
                    image = ImageOps.exif_transpose(image)
                except Exception:
                    # Malformed EXIF in an otherwise decodable image: ignore
                    # orientation metadata instead of killing the worker.
                    pass
                image = image.convert("RGB")
                return image.copy()
        except Exception as exc:  # noqa: BLE001 - keep training alive on bad files
            print(f"[data-warning] fallback gray image for {path}: {type(exc).__name__}", flush=True)
            return Image.new("RGB", (384, 384), color=(128, 128, 128))

    def __getitem__(self, index: int) -> tuple[Any, int, int, int]:
        sample = self.samples[index]
        image = self._load_rgb(sample.path)
        image = self.transform(image)
        return image, sample.class_id, sample.host_id, sample.category_id

    @property
    def num_classes(self) -> int:
        return len(self.class_map)

    @property
    def num_hosts(self) -> int:
        return len(self.host_names)

    @property
    def num_categories(self) -> int:
        return len(self.category_names)

    @property
    def class_counts(self) -> dict[int, int]:
        return dict(self._class_counts)

    def balanced_sampler(self) -> WeightedRandomSampler:
        """Square-root inverse-frequency sampler for long-tail training."""
        weights: list[float] = []
        for sample in self.samples:
            weights.append(1.0 / float(self._class_counts[sample.class_id] ** 0.5))
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.samples),
            replacement=True,
        )
