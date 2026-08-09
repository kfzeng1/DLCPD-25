import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "freeze_data_d0", ROOT / "scripts" / "freeze_data_d0.py"
)
assert SPEC is not None and SPEC.loader is not None
FREEZE_DATA_D0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZE_DATA_D0)


def test_d0_config_freezes_current_taxonomy_contract() -> None:
    config_path = ROOT / "metadata" / "d0-freeze-config-v1.json"
    config = FREEZE_DATA_D0.load_config(config_path)
    sources = config["sources"]
    official_names = FREEZE_DATA_D0.load_official_names(
        ROOT / sources["official_class_names"]
    )
    aliases = json.loads((ROOT / sources["class_directory_aliases"]).read_text(encoding="utf-8"))
    taxonomy = json.loads((ROOT / sources["class_taxonomy_json"]).read_text(encoding="utf-8"))

    classes, category_counts = FREEZE_DATA_D0.validate_metadata(
        config,
        official_names,
        aliases,
        taxonomy,
        ROOT / sources["class_taxonomy_csv"],
    )

    assert [item["class_id"] for item in classes] == list(range(203))
    assert len({item["host_id"] for item in classes}) == 22
    assert category_counts == {
        "pest": 126,
        "disease": 57,
        "healthy": 17,
        "disorder": 3,
    }


def test_inventory_fingerprint_is_deterministic_and_size_sensitive(tmp_path: Path) -> None:
    class_dir = tmp_path / "local class"
    class_dir.mkdir()
    (class_dir / "b.jpg").write_bytes(b"bbb")
    (class_dir / "a.jpg").write_bytes(b"a")
    classes = [
        {
            "class_id": 0,
            "official_name": "official class",
            "local_directory": "local class",
            "image_count": 2,
        }
    ]

    first = FREEZE_DATA_D0.scan_data_root(tmp_path, classes)
    second = FREEZE_DATA_D0.scan_data_root(tmp_path, classes)
    assert first["inventory_sha256"] == second["inventory_sha256"]
    assert first["total_files"] == 2
    assert first["total_bytes"] == 4

    (class_dir / "a.jpg").write_bytes(b"aa")
    changed = FREEZE_DATA_D0.scan_data_root(tmp_path, classes)
    assert changed["inventory_sha256"] != first["inventory_sha256"]
