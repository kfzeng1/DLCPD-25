import csv
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image
from torchvision import transforms

from dlcpd25_classifier.data import DLCPD25Dataset


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "verify_data_d4_r1", ROOT / "scripts" / "verify_data_d4_r1.py"
)
assert SPEC is not None and SPEC.loader is not None
D4_R1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D4_R1)

FIELDS = ("relative_path", "class_id", "sha256", "duplicate_group_id", "split")


def write_fixture(tmp_path: Path, relative_path: str = "class/image.png") -> tuple[Path, Path, Path]:
    data_root = tmp_path / "raw"
    image_path = data_root / "class" / "image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGBA", (8, 6), (1, 2, 3, 4)).save(image_path)
    taxonomy = {
        "classes": [
            {
                "class_id": class_id,
                "local_directory": "class" if class_id == 0 else f"unused-{class_id}",
            }
            for class_id in range(203)
        ]
    }
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")
    split_path = tmp_path / "train.csv"
    with split_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "relative_path": relative_path,
                "class_id": 0,
                "sha256": "a" * 64,
                "duplicate_group_id": "dg-000001",
                "split": "train",
            }
        )
    return data_root, split_path, taxonomy_path


def test_dataset_loads_rgb_tensor_and_fixed_target(tmp_path: Path) -> None:
    data_root, split_path, taxonomy_path = write_fixture(tmp_path)
    dataset = DLCPD25Dataset(
        data_root,
        split_path,
        taxonomy_path,
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    )
    tensor, target = dataset[0]
    assert tuple(tensor.shape) == (3, 224, 224)
    assert target == 0
    assert dataset.get_record(0).relative_path == "class/image.png"


def test_dataset_rejects_path_traversal(tmp_path: Path) -> None:
    data_root, split_path, taxonomy_path = write_fixture(tmp_path, "../outside.png")
    with pytest.raises(ValueError, match="unsafe relative path"):
        DLCPD25Dataset(data_root, split_path, taxonomy_path, verify_files=False)


def test_d4_r1_uses_only_accepted_stage_implementations() -> None:
    source = (ROOT / "scripts" / "verify_data_d4_r1.py").read_text(encoding="utf-8")
    assert D4_R1.D2_SCRIPT.name == "build_duplicates_d2_r2.py"
    assert D4_R1.D3_SCRIPT.name == "build_splits_d3_r2.py"
    assert "repo_relative(D2_SCRIPT)" in source
    assert "repo_relative(D3_SCRIPT)" in source
    assert D4_R1.DEFAULT_OUTPUT.name == "d4-r1"


def test_d2_r1_compatibility_view_uses_rebuilt_r2_manifest(tmp_path: Path) -> None:
    d2_r2 = tmp_path / "d2-r2"
    d2_r2.mkdir()
    manifest = d2_r2 / "manifest-hashed.jsonl"
    manifest.write_bytes(b'{"relative_path":"class/image.jpg"}\n')
    view = tmp_path / "d2-r1-compatible"
    report = D4_R1.build_d2_r1_compatibility_view(d2_r2, view)
    assert report["hardlinked_to_d2_r2_manifest"] is True
    assert (view / "manifest-hashed.jsonl").read_bytes() == manifest.read_bytes()
    D4_R1.verify_checksum_file(view / "checksums.sha256")
    summary = json.loads((view / "d2-r1-summary.json").read_text(encoding="utf-8"))
    assert summary["formal_implementation"] == "D2-R2"


def test_compare_reproduction_rejects_mismatch(tmp_path: Path) -> None:
    formal_d2 = tmp_path / "formal-d2"
    formal_d3 = tmp_path / "formal-d3"
    formal_d2.mkdir()
    formal_d3.mkdir()
    for name in D4_R1.D2_CORE:
        (formal_d2 / name).write_bytes(name.encode("ascii"))
    for name in D4_R1.D3_CORE:
        (formal_d3 / name).write_bytes(name.encode("ascii"))
    good_d2 = D4_R1.hashes(formal_d2, D4_R1.D2_CORE)
    good_d3 = D4_R1.hashes(formal_d3, D4_R1.D3_CORE)
    reproduction = {
        "runs": [
            {"d2_hashes": good_d2, "d3_hashes": good_d3},
            {"d2_hashes": good_d2, "d3_hashes": {**good_d3, "train.csv": "0" * 64}},
        ]
    }
    with pytest.raises(D4_R1.ReproductionError, match="d3 reproduction mismatch"):
        D4_R1.compare_reproduction(formal_d2, formal_d3, reproduction)


def test_full_d4_r1_artifacts_and_dataset_smoke() -> None:
    result = D4_R1.verify_d4_r1(D4_R1.DEFAULT_D2, D4_R1.DEFAULT_D3, D4_R1.DEFAULT_OUTPUT)
    assert result["reproduction_runs"] == 2
    assert result["d2_core_file_count"] == 19
    assert result["d3_core_file_count"] == 5
    assert result["loaded_split_lengths"] == D4_R1.EXPECTED_SPLIT_LENGTHS
