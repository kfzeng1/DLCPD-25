import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "freeze_data_v1_d5_r1", ROOT / "scripts" / "freeze_data_v1_d5_r1.py"
)
assert SPEC is not None and SPEC.loader is not None
D5_R1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D5_R1)


def test_handoff_contains_contract_split_and_a0_boundary() -> None:
    release = {
        "stage": "D5-R1",
        "release_status": "frozen_pending_project_lead_acceptance",
        "input_git_commit": "abc",
        "formal_chain": ["D0", "D1", "D2-R2", "D3-R2", "D4-R1", "D5-R1"],
        "taxonomy_sha256": "d" * 64,
        "statistics": {
            "total_files": 10,
            "usable_files": 9,
            "bad_files": 1,
            "duplicates": {
                "exact_sha256_groups": 1,
                "near_duplicate_groups": 1,
                "duplicate_groups": 1,
                "cross_class_duplicate_groups": 0,
            },
            "splits": {
                split: {"count": count, "ratio": count / 9, "sha256": split * 16}
                for split, count in (("train", 7), ("val", 1), ("test", 1))
            },
        },
        "long_tail_classes": [
            {
                "class_id": 0,
                "official_name": "sample",
                "total": 9,
                "group_count": 5,
                "train": 7,
                "val": 1,
                "test": 1,
            }
        ],
        "known_limitations": ["sample limitation"],
        "commands": [
            "/home/zkf/pytorch-env/bin/python scripts/build_duplicates_d2_r2.py --workers 6",
            "/home/zkf/pytorch-env/bin/python scripts/build_splits_d3_r2.py",
        ],
        "critical_artifacts": {
            "sample": {"path": "sample", "sha256": "e" * 64, "size_bytes": 1}
        },
        "environment": {"python": "3.12"},
    }
    handoff = D5_R1.render_handoff(release)
    assert "D2-R2 -> D3-R2 -> D4-R1 -> D5-R1" in handoff
    assert "固定数据契约" in handoff
    assert "固定 Split" in handoff
    assert "算法工程师只读取固定 split CSV" in handoff
    assert "artifacts/data/v1/d2-r2/" in handoff
    assert "artifacts/data/v1/d3-r2/" in handoff
    assert "artifacts/data/v1/d5-r1/taxonomy-v1.json" in handoff
    assert "未执行 A0" in handoff


def test_formal_critical_paths_use_only_revision_chain() -> None:
    paths = {name: path.as_posix() for name, path in D5_R1.CRITICAL_PATHS.items()}
    assert paths["d2_r2_manifest_hashed"].endswith("artifacts/data/v1/d2-r2/manifest-hashed.jsonl")
    assert paths["d3_r2_train"].endswith("artifacts/data/v1/d3-r2/train.csv")
    assert paths["d4_r1_summary"].endswith("artifacts/data/v1/d4-r1/d4-r1-summary.json")
    assert all("/artifacts/data/v1/d2/" not in path for path in paths.values())
    assert all("/artifacts/data/v1/d3/" not in path for path in paths.values())
    assert all("/artifacts/data/v1/d4/" not in path for path in paths.values())


def test_full_d5_r1_release_contract() -> None:
    result = D5_R1.verify_d5_r1(D5_R1.DEFAULT_OUTPUT)
    assert result["data_version"] == "data-v1"
    assert result["formal_chain"] == ["D0", "D1", "D2-R2", "D3-R2", "D4-R1", "D5-R1"]
    assert result["a0_executed"] is False
