import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_manifest_d1", ROOT / "scripts" / "build_manifest_d1.py"
)
assert SPEC is not None and SPEC.loader is not None
BUILD_MANIFEST_D1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_MANIFEST_D1)


def test_decode_audit_records_dimensions_channels_and_format(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "sample.png"
    Image.new("RGBA", (7, 5), (1, 2, 3, 4)).save(image_path)
    result = BUILD_MANIFEST_D1.decode_image(image_path)
    assert result["decode_status"] == "ok"
    assert result["format"] == "PNG"
    assert result["width"] == 7
    assert result["height"] == 5
    assert result["mode"] == "RGBA"
    assert result["channels"] == 4
    assert result["frame_count"] == 1


def test_decode_audit_marks_corrupt_image_without_raising(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"not an image")
    result = BUILD_MANIFEST_D1.decode_image(image_path)
    assert result["decode_status"] == "bad"
    assert result["format"] is None
    assert result["width"] is None
    assert result["decode_error_type"]
    assert str(tmp_path) not in result["decode_error"]


def test_manifest_record_is_relative_and_excludes_content_hashes(tmp_path: Path) -> None:
    from PIL import Image

    data_root = tmp_path / "raw"
    class_dir = data_root / "class"
    class_dir.mkdir(parents=True)
    image_path = class_dir / "one.jpg"
    Image.new("RGB", (2, 3)).save(image_path)
    class_record = {
        "class_id": 4,
        "official_name": "official",
        "local_directory": "class",
        "host_group": "economic_crop",
        "host_id": "tomato",
        "host_zh": "番茄",
        "category": "pest",
        "category_zh": "农业有害生物",
    }
    record = BUILD_MANIFEST_D1.make_record(
        image_path,
        data_root,
        class_record,
        BUILD_MANIFEST_D1.decode_image(image_path),
    )
    assert record["relative_path"] == "class/one.jpg"
    assert not any("sha" in key or "hash" in key for key in record)
    assert record["class_id"] == 4


def test_manifest_writer_emits_manifest_and_separate_bad_list(tmp_path: Path) -> None:
    from PIL import Image

    data_root = tmp_path / "raw"
    class_dir = data_root / "class"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (2, 3)).save(class_dir / "good.jpg")
    (class_dir / "bad.jpg").write_bytes(b"broken")
    class_record = {
        "class_id": 0,
        "official_name": "official",
        "local_directory": "class",
        "host_group": "economic_crop",
        "host_id": "tomato",
        "host_zh": "番茄",
        "category": "pest",
        "category_zh": "农业有害生物",
        "image_count": 2,
    }
    output = tmp_path / "d1"
    summary, _ = BUILD_MANIFEST_D1.write_manifest_files(output, data_root, [class_record], workers=2)
    assert summary["total_files"] == 2
    assert summary["ok_files"] == 1
    assert summary["bad_files"] == 1
    assert summary["paths_unique"] is True
    assert len((output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len((output / "bad-images.jsonl").read_text(encoding="utf-8").splitlines()) == 1
