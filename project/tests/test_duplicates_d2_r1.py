import importlib.util
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_duplicates_d2_r1", ROOT / "scripts" / "build_duplicates_d2_r1.py"
)
assert SPEC is not None and SPEC.loader is not None
D2_R1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D2_R1)


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
    assert D2_R1.phash64(path) == D2_R1.phash64(path)
    value, status = D2_R1.phash64(path)
    assert len(f"{value:016x}") == 16
    assert status == "missing"


def test_complete_link_prevents_dhash_chain_expansion() -> None:
    records = [
        record("a", 0x000, 0x1234, 0),
        record("b", 0x01F, 0x1234, 1),
        record("c", 0x3FF, 0x1234, 2),
    ]
    group_ids, stats, _components = D2_R1.assign_duplicate_groups(records)
    assert group_ids[0] == group_ids[1]
    assert group_ids[2] != group_ids[0]
    assert stats["maximum_group_dhash_diameter"] <= D2_R1.DHASH_THRESHOLD


def test_phash_rejects_dhash_candidate() -> None:
    records = [
        record("a", 0, 0, 0),
        record("b", 1, 0xFFFF, 1),
    ]
    group_ids, stats, _components = D2_R1.assign_duplicate_groups(records)
    assert group_ids[0] != group_ids[1]
    assert stats["phash_rejected_fingerprint_pairs"] == 1


def test_exact_bad_files_remain_grouped_without_perceptual_hashes() -> None:
    records = [record("a", None, None, 0), record("a", None, None, 1)]
    group_ids, stats, _components = D2_R1.assign_duplicate_groups(records)
    assert group_ids[0] == group_ids[1]
    assert stats["exact_sha256_groups"] == 1


def test_rejected_group_ids_are_registered_as_regressions() -> None:
    assert D2_R1.REJECTED_GROUP_IDS == ("dg-001396", "dg-004019")


def test_full_d2_r1_artifact_regressions_and_diameter() -> None:
    result = D2_R1.verify_d2_r1(
        D2_R1.DEFAULT_D1,
        D2_R1.DEFAULT_D2_R0,
        D2_R1.DEFAULT_OUTPUT,
    )
    assert result["regression_groups_split"] == 2
    assert result["maximum_group_dhash_diameter"] <= D2_R1.DHASH_THRESHOLD
    assert result["maximum_group_phash_diameter"] <= D2_R1.PHASH_THRESHOLD
