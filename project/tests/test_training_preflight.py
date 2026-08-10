import csv
import hashlib
import json
from pathlib import Path

import pytest

from dlcpd25_classifier.training.preflight import (
    PreflightError,
    CSV_FIELDS,
    inspect_splits,
    inspect_taxonomy,
    run_preflight,
)


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "artifacts/data/v1/d5-r1/data-v1-release.json"
DATA_ROOT = ROOT / "data/raw/dlcpd25-203"


def write_taxonomy(path: Path) -> tuple[dict[int, dict[str, object]], str]:
    classes = []
    for class_id in range(203):
        classes.append(
            {
                "class_id": class_id,
                "official_name": f"official-{class_id}",
                "local_directory": f"class-{class_id}",
                "host_id": f"host-{class_id % 22}",
                "category": ("pest", "disease", "healthy", "disorder")[class_id % 4],
            }
        )
    path.write_text(json.dumps({"classes": classes}), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records, _result = inspect_taxonomy(path, digest)
    return records, digest


def write_split(path: Path, split: str, shared_group: str | None = None) -> tuple[int, str]:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for class_id in range(203):
            relative_path = f"class-{class_id}/{split}.jpg"
            source = path.parents[1] / "raw" / relative_path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"{split}-{class_id}".encode())
            writer.writerow(
                {
                    "relative_path": relative_path,
                    "class_id": class_id,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "duplicate_group_id": shared_group if class_id == 0 and shared_group else f"{split}-{class_id}",
                    "split": split,
                }
            )
    return 203, hashlib.sha256(path.read_bytes()).hexdigest()


def make_split_contract(tmp_path: Path, shared_group: str | None = None):
    repo_root = tmp_path
    split_dir = repo_root / "splits"
    split_dir.mkdir()
    data_root = repo_root / "raw"
    data_root.mkdir()
    taxonomy_path = repo_root / "taxonomy.json"
    classes, _digest = write_taxonomy(taxonomy_path)
    contracts = {}
    statistics = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.csv"
        count, digest = write_split(path, split, shared_group)
        contracts[split] = {
            "path": path.relative_to(repo_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
        statistics[split] = {"count": count, "sha256": digest}
    return repo_root, data_root, contracts, statistics, classes


def test_split_inspection_accepts_complete_disjoint_contract(tmp_path: Path) -> None:
    repo_root, data_root, contracts, statistics, classes = make_split_contract(tmp_path)
    result = inspect_splits(
        repo_root, data_root, contracts, statistics, CSV_FIELDS, classes
    )
    assert result["total_rows"] == 609
    assert result["unique_paths"] == 609
    assert result["path_overlap_count"] == 0
    assert result["duplicate_group_leakage_count"] == 0


def test_split_inspection_rejects_duplicate_group_leakage(tmp_path: Path) -> None:
    repo_root, data_root, contracts, statistics, classes = make_split_contract(
        tmp_path, shared_group="leaked-group"
    )
    with pytest.raises(PreflightError, match="duplicate group crosses splits"):
        inspect_splits(repo_root, data_root, contracts, statistics, CSV_FIELDS, classes)


def test_taxonomy_rejects_non_contiguous_class_ids(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "taxonomy.json"
    _classes, digest = write_taxonomy(taxonomy_path)
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    payload["classes"][202]["class_id"] = 999
    taxonomy_path.write_text(json.dumps(payload), encoding="utf-8")
    changed_digest = hashlib.sha256(taxonomy_path.read_bytes()).hexdigest()
    assert changed_digest != digest
    with pytest.raises(PreflightError, match="contiguous"):
        inspect_taxonomy(taxonomy_path, changed_digest)


def test_full_frozen_data_v1_preflight() -> None:
    result = run_preflight(ROOT, RELEASE, DATA_ROOT)
    assert result["status"] == "passed"
    assert result["taxonomy"]["class_count"] == 203
    assert result["split_integrity"]["total_rows"] == 221377
    assert result["split_integrity"]["path_overlap_count"] == 0
    assert result["split_integrity"]["duplicate_group_leakage_count"] == 0
