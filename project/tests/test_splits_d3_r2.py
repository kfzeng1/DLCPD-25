import importlib.util
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_splits_d3_r2.py"
SPEC = importlib.util.spec_from_file_location("build_splits_d3_r2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
D3_R2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = D3_R2
SPEC.loader.exec_module(D3_R2)


def make_group(group_id: str, classes: dict[int, int]) -> object:
    return D3_R2.GroupInfo(
        group_id=group_id,
        indices=tuple(),
        class_counts=classes,
        size=sum(classes.values()),
        tie_break=D3_R2.stable_tie(group_id),
    )


def test_r2_assignment_is_deterministic_and_group_safe() -> None:
    groups = []
    class_groups = defaultdict(list)
    for class_id in range(3):
        for index in range(6):
            group = make_group(f"dg-r1-{class_id}-{index}", {class_id: index + 1})
            groups.append(group)
            class_groups[class_id].append(group.group_id)
    cross = make_group("dg-r1-cross", {0: 2, 1: 2})
    groups.append(cross)
    class_groups[0].append(cross.group_id)
    class_groups[1].append(cross.group_id)
    first, _strategy = D3_R2.assign_groups(groups, class_groups, class_count=3)
    second, _strategy = D3_R2.assign_groups(groups, class_groups, class_count=3)
    assert first == second
    assert first[cross.group_id] in D3_R2.SPLITS
    for class_id in range(3):
        assert {first[group_id] for group_id in class_groups[class_id]} == set(D3_R2.SPLITS)


def test_standalone_script_imports_and_executes_without_old_scripts(tmp_path: Path) -> None:
    isolated_scripts = tmp_path / "scripts"
    isolated_scripts.mkdir()
    isolated_script = isolated_scripts / SCRIPT.name
    shutil.copy2(SCRIPT, isolated_script)
    assert not (isolated_scripts / "build_splits_d3.py").exists()
    assert not (isolated_scripts / "build_splits_d3_r1.py").exists()
    command = (
        "import importlib.util,sys;"
        f"p={str(isolated_script)!r};"
        "s=importlib.util.spec_from_file_location('isolated_d3_r2',p);"
        "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
        "g=m.GroupInfo('dg-r1-x',(),{0:1},1,m.stable_tie('dg-r1-x'));"
        "a,_=m.assign_groups([g],{0:['dg-r1-x']},1);"
        "assert a=={'dg-r1-x':'train'}"
    )
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_full_d3_r2_artifact_and_source_checksums() -> None:
    result = D3_R2.verify_d3_r2(
        D3_R2.DEFAULT_D2,
        D3_R2.DEFAULT_OUTPUT,
        D3_R2.DEFAULT_TAXONOMY,
    )
    assert result["source_d2_stage"] == "D2-R2"
    assert result["implementation_version"] == "standalone-d3-r2-v2"
    assert result["runtime_source_count"] == 1
    assert result["rows"] == 221377
    assert result["excluded_bad_files"] == 19
    assert result["path_overlap_count"] == 0
    assert result["duplicate_group_leakage_count"] == 0
    assert result["class_coverage"] == {"train": 203, "val": 203, "test": 203}


def test_checksum_has_r2_sources_and_no_rejected_runtime_script() -> None:
    checksum = (D3_R2.DEFAULT_OUTPUT / "checksums.sha256").read_text(encoding="utf-8")
    assert "scripts/build_splits_d3_r2.py" in checksum
    assert "project/tests/test_splits_d3_r2.py" in checksum
    assert "scripts/build_splits_d3.py" not in checksum
    assert "scripts/build_splits_d3_r1.py" not in checksum
