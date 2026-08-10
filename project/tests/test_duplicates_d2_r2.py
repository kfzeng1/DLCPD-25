import importlib.util
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_duplicates_d2_r2", ROOT / "scripts" / "build_duplicates_d2_r2.py"
)
assert SPEC is not None and SPEC.loader is not None
D2_R2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D2_R2)


def record(sha: str, dhash: int | None, phash: int | None, class_id: int) -> dict[str, object]:
    return {
        "sha256": sha * 64,
        "dhash64": None if dhash is None else f"{dhash:016x}",
        "phash64": None if phash is None else f"{phash:016x}",
        "class_id": class_id,
        "decode_status": "bad" if dhash is None else "ok",
    }


def test_phash64_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "gradient.png"
    image = Image.new("L", (32, 32))
    image.putdata([(row * 7 + column * 3) % 256 for row in range(32) for column in range(32)])
    image.save(path)
    assert D2_R2.phash64(path) == D2_R2.phash64(path)
    value, status = D2_R2.phash64(path)
    assert len(f"{value:016x}") == 16
    assert status == "missing"


def test_complete_link_prevents_dhash_chain_expansion() -> None:
    records = [
        record("a", 0x000, 0x1234, 0),
        record("b", 0x01F, 0x1234, 1),
        record("c", 0x3FF, 0x1234, 2),
    ]
    group_ids, stats, _components = D2_R2.assign_duplicate_groups(records)
    assert group_ids[0] == group_ids[1]
    assert group_ids[2] != group_ids[0]
    assert stats["maximum_group_dhash_diameter"] <= D2_R2.DHASH_THRESHOLD


def test_phash_rejects_dhash_candidate() -> None:
    records = [
        record("a", 0, 0, 0),
        record("b", 1, 0xFFFF, 1),
    ]
    group_ids, stats, _components = D2_R2.assign_duplicate_groups(records)
    assert group_ids[0] != group_ids[1]
    assert stats["phash_rejected_fingerprint_pairs"] == 1


def test_exact_bad_files_remain_grouped_without_perceptual_hashes() -> None:
    records = [record("a", None, None, 0), record("a", None, None, 1)]
    group_ids, stats, _components = D2_R2.assign_duplicate_groups(records)
    assert group_ids[0] == group_ids[1]
    assert stats["exact_sha256_groups"] == 1


def test_rejected_group_ids_are_registered_as_regressions() -> None:
    assert D2_R2.REJECTED_GROUP_IDS == ("dg-001396", "dg-004019")


def test_legacy_group_ids_preserve_r0_ordering() -> None:
    records = [
        record("a", 0, 0, 0),
        record("b", 1, 1, 1),
        record("c", 0xFFFF, 0xFFFF, 2),
    ]
    assert D2_R2.assign_legacy_group_ids(records) == ["dg-000001", "dg-000001", "dg-000002"]


def test_standalone_import_without_rejected_d2_script(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    isolated_script = scripts_dir / "build_duplicates_d2_r2.py"
    shutil.copy2(ROOT / "scripts" / "build_duplicates_d2_r2.py", isolated_script)
    assert not (scripts_dir / "build_duplicates_d2.py").exists()
    isolated_spec = importlib.util.spec_from_file_location("isolated_d2_r2", isolated_script)
    assert isolated_spec is not None and isolated_spec.loader is not None
    isolated = importlib.util.module_from_spec(isolated_spec)
    isolated_spec.loader.exec_module(isolated)
    assert isolated.assign_legacy_group_ids(
        [record("a", 0, 0, 0), record("b", 1, 1, 1)]
    ) == ["dg-000001", "dg-000001"]


def test_no_runtime_dependency_on_repository_scripts() -> None:
    source = (ROOT / "scripts" / "build_duplicates_d2_r2.py").read_text(encoding="utf-8")
    assert "importlib" not in source
    assert "OLD_SCRIPT" not in source
    assert D2_R2.CORE_COMPATIBILITY_SHA256["manifest-hashed.jsonl"] == (
        "177e785b0cffd53ad0de7eb5aa3f2a2899127ca77558a774297929c2e2b80828"
    )


def test_frozen_d2_core_matches_accepted_hashes() -> None:
    reference_dir = D2_R2.DEFAULT_OUTPUT
    assert {
        name: D2_R2.sha256_file(reference_dir / name)
        for name in D2_R2.CORE_COMPATIBILITY_SHA256
    } == D2_R2.CORE_COMPATIBILITY_SHA256


def test_full_d2_r2_artifact_regressions_and_diameter() -> None:
    result = D2_R2.verify_d2_r2(D2_R2.DEFAULT_D1, D2_R2.DEFAULT_OUTPUT)
    assert result["regression_groups_split"] == 2
    assert result["maximum_group_dhash_diameter"] <= D2_R2.DHASH_THRESHOLD
    assert result["maximum_group_phash_diameter"] <= D2_R2.PHASH_THRESHOLD
    assert result["core_compatibility_artifacts"] == len(D2_R2.CORE_COMPATIBILITY_SHA256)
    assert result["runtime_source_count"] == 1
