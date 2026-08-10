#!/usr/bin/env python3
"""Rebuild D2-R2/D3-R2 twice and verify frozen split loading for D4-R1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = REPO_ROOT / "project" / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from dlcpd25_classifier.data import DLCPD25Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402


DEFAULT_D2 = REPO_ROOT / "artifacts" / "data" / "v1" / "d2-r2"
DEFAULT_D3 = REPO_ROOT / "artifacts" / "data" / "v1" / "d3-r2"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "data" / "v1" / "d4-r1"
D2_R1_REFERENCE = REPO_ROOT / "artifacts" / "data" / "v1" / "d2-r1"
DATA_ROOT = REPO_ROOT / "data" / "raw" / "dlcpd25-203"
TAXONOMY = REPO_ROOT / "metadata" / "class-taxonomy.json"
TEST_SOURCE = REPO_ROOT / "project" / "tests" / "test_data_d4_r1.py"
DATASET_SOURCE = REPO_ROOT / "project" / "src" / "dlcpd25_classifier" / "data" / "dataset.py"
DATA_INIT_SOURCE = REPO_ROOT / "project" / "src" / "dlcpd25_classifier" / "data" / "__init__.py"
D2_SCRIPT = REPO_ROOT / "scripts" / "build_duplicates_d2_r2.py"
D3_SCRIPT = REPO_ROOT / "scripts" / "build_splits_d3_r2.py"
D2_TEST = REPO_ROOT / "project" / "tests" / "test_duplicates_d2_r2.py"
D3_TEST = REPO_ROOT / "project" / "tests" / "test_splits_d3_r2.py"
INPUT_GIT_COMMIT = "b43e46e67f162a3da5c0fcdffdb0e6f989bd6cac"
D2_CORE = (
    "manifest-hashed.jsonl",
    "duplicate-groups.jsonl",
    "audit-samples.json",
    "rejected-groups-regression.json",
    "audit-cross-class-page-01.jpg",
    "audit-cross-class-page-02.jpg",
    "audit-cross-class-page-03.jpg",
    "audit-cross-class-page-04.jpg",
    "audit-largest-page-01.jpg",
    "audit-largest-page-02.jpg",
    "audit-largest-page-03.jpg",
    "audit-largest-page-04.jpg",
    "audit-random-page-01.jpg",
    "audit-random-page-02.jpg",
    "audit-random-page-03.jpg",
    "audit-random-page-04.jpg",
    "d2-r2-compatibility.json",
    "d2-r2-config.json",
    "d2-r2-summary.json",
)
D3_CORE = ("train.csv", "val.csv", "test.csv", "group-assignments.csv", "excluded-bad-images.csv")
EXPECTED_SPLIT_LENGTHS = {"train": 177021, "val": 22178, "test": 22178}


class ReproductionError(ValueError):
    """Raised when D4 reproduction or loading checks fail."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2-dir", type=Path, default=DEFAULT_D2)
    parser.add_argument("--d3-dir", type=Path, default=DEFAULT_D3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_bytes_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ReproductionError(f"refusing to overwrite a different artifact: {path}")
        return
    path.write_bytes(payload)


def run_stage(command: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise ReproductionError(
            f"stage command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-2000:]}"
        )
    return {
        "command": command,
        "exit_code": result.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "last_stdout_line": result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "",
    }


def hashes(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


def verify_checksum_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        source = REPO_ROOT / name
        if separator != "  " or not source.is_file() or sha256_file(source) != digest:
            raise ReproductionError(f"checksum mismatch: {name}")


def link_idempotent(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise ReproductionError(f"refusing to replace a different compatibility file: {destination}")
        return
    os.link(source, destination)


def build_d2_r1_compatibility_view(d2_r2_dir: Path, view_dir: Path) -> dict[str, Any]:
    manifest = d2_r2_dir / "manifest-hashed.jsonl"
    manifest_digest = sha256_file(manifest)
    link_idempotent(manifest, view_dir / "manifest-hashed.jsonl")
    summary = {
        "schema_version": 1,
        "stage": "D2-R1-compatible-reference",
        "formal_implementation": "D2-R2",
        "source_manifest": repo_relative(manifest),
        "manifest_sha256": manifest_digest,
        "reason": "D3-R2 retains the accepted D2-R1 input contract; D2-R2 core bytes are identical",
    }
    summary_path = view_dir / "d2-r1-summary.json"
    write_bytes_idempotent(summary_path, canonical_json(summary))
    checksum_paths = [view_dir / "manifest-hashed.jsonl", summary_path]
    lines = [f"{sha256_file(path)}  {repo_relative(path)}" for path in checksum_paths]
    write_bytes_idempotent(view_dir / "checksums.sha256", ("\n".join(lines) + "\n").encode("utf-8"))
    verify_checksum_file(view_dir / "checksums.sha256")
    return {
        "manifest_sha256": manifest_digest,
        "hardlinked_to_d2_r2_manifest": (view_dir / "manifest-hashed.jsonl").stat().st_ino
        == manifest.stat().st_ino,
        "compatibility_contract": "D2-R1 byte-compatible input for D3-R2",
    }


def run_reproduction(output_dir: Path, workers: int) -> dict[str, Any]:
    run_reports = []
    for run_number in (1, 2):
        run_root = output_dir / f"repro-run-{run_number}"
        d2_output = run_root / "d2-r2"
        d2_view = run_root / "d2-r1-compatible"
        d3_output = run_root / "d3-r2"
        d2_report = run_stage(
            [
                sys.executable,
                repo_relative(D2_SCRIPT),
                "--workers",
                str(workers),
                "--output-dir",
                str(d2_output),
            ]
        )
        compatibility = build_d2_r1_compatibility_view(d2_output, d2_view)
        d3_report = run_stage(
            [
                sys.executable,
                repo_relative(D3_SCRIPT),
                "--d2-r1-dir",
                str(d2_view),
                "--output-dir",
                str(d3_output),
            ]
        )
        run_reports.append(
            {
                "run": run_number,
                "d2": d2_report,
                "d2_r1_compatibility_view": compatibility,
                "d3": d3_report,
                "d2_hashes": hashes(d2_output, D2_CORE),
                "d3_hashes": hashes(d3_output, D3_CORE),
            }
        )
    return {"runs": run_reports}


def compare_reproduction(d2_dir: Path, d3_dir: Path, reproduction: dict[str, Any]) -> dict[str, Any]:
    references = {"d2": hashes(d2_dir, D2_CORE), "d3": hashes(d3_dir, D3_CORE)}
    comparisons = {}
    for stage, names in (("d2", D2_CORE), ("d3", D3_CORE)):
        comparisons[stage] = {}
        for name in names:
            values = [references[stage][name]] + [run[f"{stage}_hashes"][name] for run in reproduction["runs"]]
            comparisons[stage][name] = {"sha256": values[0], "all_three_match": len(set(values)) == 1}
            if len(set(values)) != 1:
                raise ReproductionError(f"{stage} reproduction mismatch: {name}: {values}")
    return {"reference_hashes": references, "comparisons": comparisons}


def loading_smoke(d3_dir: Path) -> dict[str, Any]:
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    result = {
        "dataset": "dlcpd25_classifier.data.DLCPD25Dataset",
        "preprocessing": "Resize((224,224)) + ToTensor()",
        "taxonomy_sha256": sha256_file(TAXONOMY),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        dataset = DLCPD25Dataset(DATA_ROOT, d3_dir / f"{split}.csv", TAXONOMY, transform=transform)
        indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
        samples = []
        for index in indices:
            tensor, target = dataset[index]
            record = dataset.get_record(index)
            finite = bool(tensor.isfinite().all().item())
            if (
                tuple(tensor.shape) != (3, 224, 224)
                or str(tensor.dtype) != "torch.float32"
                or not finite
                or target != record.class_id
                or record.split != split
                or not 0 <= target < 203
            ):
                raise ReproductionError(f"Dataset preprocessing or target mismatch: {split}:{index}")
            samples.append(
                {
                    "index": index,
                    "relative_path": record.relative_path,
                    "class_id": target,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "finite": finite,
                }
            )
        result["splits"][split] = {"length": len(dataset), "samples": samples}
    return result


def formal_input_preflight(d2_dir: Path, d3_dir: Path) -> dict[str, Any]:
    d2_report = run_stage(
        [sys.executable, repo_relative(D2_SCRIPT), "--output-dir", str(d2_dir), "--verify-only"]
    )
    d3_report = run_stage(
        [
            sys.executable,
            repo_relative(D3_SCRIPT),
            "--d2-r1-dir",
            str(D2_R1_REFERENCE),
            "--output-dir",
            str(d3_dir),
            "--verify-only",
        ]
    )
    d2_digest = sha256_file(d2_dir / "manifest-hashed.jsonl")
    d2_r1_digest = sha256_file(D2_R1_REFERENCE / "manifest-hashed.jsonl")
    if d2_digest != d2_r1_digest:
        raise ReproductionError("formal D2-R2 manifest is not byte-compatible with D2-R1")
    return {
        "d2_r2": d2_report,
        "d3_r2": d3_report,
        "d2_r2_d2_r1_manifest_byte_compatible": True,
        "manifest_sha256": d2_digest,
    }


def source_index() -> list[dict[str, str]]:
    sources = [
        Path(__file__).resolve(),
        TEST_SOURCE,
        DATASET_SOURCE,
        DATA_INIT_SOURCE,
        D2_SCRIPT,
        D2_TEST,
        D3_SCRIPT,
        D3_TEST,
    ]
    for source in sources:
        if not source.is_file():
            raise ReproductionError(f"required D4-R1 source is missing: {source}")
    return [{"path": repo_relative(path), "sha256": sha256_file(path)} for path in sources]


def build_d4_r1(d2_dir: Path, d3_dir: Path, output_dir: Path, workers: int) -> dict[str, Any]:
    if workers < 1:
        raise ReproductionError("workers must be at least 1")
    d2_dir = d2_dir.resolve()
    d3_dir = d3_dir.resolve()
    output_dir = output_dir.resolve()
    sources = source_index()
    preflight = formal_input_preflight(d2_dir, d3_dir)
    reproduction = run_reproduction(output_dir, workers)
    comparison = compare_reproduction(d2_dir, d3_dir, reproduction)
    smoke = loading_smoke(d3_dir)
    config = {
        "schema_version": 2,
        "stage": "D4-R1",
        "data_version": "data-v1-candidate-r1",
        "input_git_commit": INPUT_GIT_COMMIT,
        "reproduction_runs": 2,
        "workers": workers,
        "formal_d2_stage": "D2-R2",
        "formal_d2_dir": repo_relative(d2_dir),
        "formal_d3_stage": "D3-R2",
        "formal_d3_dir": repo_relative(d3_dir),
        "d2_script": repo_relative(D2_SCRIPT),
        "d3_script": repo_relative(D3_SCRIPT),
        "d3_d2_r1_reference_semantics": (
            "D3-R2 keeps the accepted D2-R1 contract; every run consumes a compatibility view "
            "hardlinked to its independently rebuilt, byte-identical D2-R2 manifest"
        ),
        "d2_core_files": list(D2_CORE),
        "d3_core_files": list(D3_CORE),
        "loader": "dlcpd25_classifier.data.DLCPD25Dataset",
        "source_files": sources,
    }
    write_bytes_idempotent(output_dir / "d4-r1-config.json", canonical_json(config))
    write_bytes_idempotent(
        output_dir / "reproduction-summary.json",
        canonical_json({"formal_input_preflight": preflight, **reproduction, **comparison}),
    )
    write_bytes_idempotent(output_dir / "load-smoke.json", canonical_json(smoke))
    summary = {
        "schema_version": 2,
        "stage": "D4-R1",
        "data_version": "data-v1-candidate-r1",
        "reproduction_runs": 2,
        "d2_core_all_match": all(
            item["all_three_match"] for item in comparison["comparisons"]["d2"].values()
        ),
        "d3_core_all_match": all(
            item["all_three_match"] for item in comparison["comparisons"]["d3"].values()
        ),
        "loaded_split_lengths": {
            split: smoke["splits"][split]["length"] for split in ("train", "val", "test")
        },
        "loaded_samples_per_split": 3,
        "runtime": {"python": platform.python_version()},
        "formal_d2_stage": "D2-R2",
        "formal_d3_stage": "D3-R2",
        "d3_d2_r1_reference_is_byte_compatible": True,
        "config_sha256": sha256_file(output_dir / "d4-r1-config.json"),
        "reproduction_summary_sha256": sha256_file(output_dir / "reproduction-summary.json"),
        "load_smoke_sha256": sha256_file(output_dir / "load-smoke.json"),
    }
    write_bytes_idempotent(output_dir / "d4-r1-summary.json", canonical_json(summary))
    checksum_paths = [Path(item["path"]) if Path(item["path"]).is_absolute() else REPO_ROOT / item["path"] for item in sources]
    checksum_paths.extend(
        [
            TAXONOMY,
            d2_dir / "checksums.sha256",
            d3_dir / "checksums.sha256",
            *[d2_dir / name for name in D2_CORE],
            *[d3_dir / name for name in D3_CORE],
        ]
    )
    checksum_paths.extend(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != output_dir / "checksums.sha256"
    )
    lines = [f"{sha256_file(path)}  {repo_relative(path)}" for path in sorted(set(checksum_paths))]
    write_bytes_idempotent(output_dir / "checksums.sha256", ("\n".join(lines) + "\n").encode("utf-8"))
    verify_d4_r1(d2_dir, d3_dir, output_dir)
    return summary


def verify_d4_r1(d2_dir: Path, d3_dir: Path, output_dir: Path) -> dict[str, Any]:
    d2_dir = d2_dir.resolve()
    d3_dir = d3_dir.resolve()
    output_dir = output_dir.resolve()
    verify_checksum_file(output_dir / "checksums.sha256")
    summary = json.loads((output_dir / "d4-r1-summary.json").read_text(encoding="utf-8"))
    reproduction = json.loads((output_dir / "reproduction-summary.json").read_text(encoding="utf-8"))
    smoke = json.loads((output_dir / "load-smoke.json").read_text(encoding="utf-8"))
    config = json.loads((output_dir / "d4-r1-config.json").read_text(encoding="utf-8"))
    comparison = compare_reproduction(d2_dir, d3_dir, reproduction)
    if not summary["d2_core_all_match"] or not summary["d3_core_all_match"]:
        raise ReproductionError("D4-R1 summary reports a reproduction mismatch")
    if any(
        not item["all_three_match"]
        for stage in ("d2", "d3")
        for item in comparison["comparisons"][stage].values()
    ):
        raise ReproductionError("D4-R1 reproduction comparison contains mismatch")
    if reproduction["comparisons"] != comparison["comparisons"]:
        raise ReproductionError("stored D4-R1 reproduction hashes differ from current files")
    if len(reproduction["runs"]) != 2:
        raise ReproductionError("D4-R1 must contain exactly two reproduction runs")
    for run_number in (1, 2):
        run_root = output_dir / f"repro-run-{run_number}"
        run_manifest = run_root / "d2-r2" / "manifest-hashed.jsonl"
        view_manifest = run_root / "d2-r1-compatible" / "manifest-hashed.jsonl"
        if sha256_file(run_manifest) != sha256_file(view_manifest):
            raise ReproductionError(f"D3 compatibility view differs in run {run_number}")
        verify_checksum_file(run_root / "d2-r2" / "checksums.sha256")
        verify_checksum_file(run_root / "d2-r1-compatible" / "checksums.sha256")
        verify_checksum_file(run_root / "d3-r2" / "checksums.sha256")
    if set(smoke["splits"]) != {"train", "val", "test"}:
        raise ReproductionError("D4-R1 load smoke does not cover all splits")
    for split, expected in EXPECTED_SPLIT_LENGTHS.items():
        entry = smoke["splits"][split]
        if entry["length"] != expected or len(entry["samples"]) != 3:
            raise ReproductionError(f"D4-R1 load smoke differs for {split}")
        if any(
            sample["shape"] != [3, 224, 224]
            or sample["dtype"] != "torch.float32"
            or not sample["finite"]
            for sample in entry["samples"]
        ):
            raise ReproductionError(f"D4-R1 tensor smoke failed for {split}")
    current_smoke = loading_smoke(d3_dir)
    if current_smoke != smoke:
        raise ReproductionError("D4-R1 Dataset smoke is not reproducible")
    if config["source_files"] != source_index():
        raise ReproductionError("D4-R1 source index differs")
    if config["d2_script"] != repo_relative(D2_SCRIPT) or config["d3_script"] != repo_relative(D3_SCRIPT):
        raise ReproductionError("D4-R1 stage script selection differs")
    checksum_text = (output_dir / "checksums.sha256").read_text(encoding="utf-8")
    for rejected in ("scripts/build_duplicates_d2.py", "scripts/build_splits_d3.py"):
        if rejected in checksum_text:
            raise ReproductionError(f"D4-R1 checksum contains rejected source: {rejected}")
    return {
        "d2_core_all_match": True,
        "d3_core_all_match": True,
        "loaded_split_lengths": summary["loaded_split_lengths"],
        "loaded_samples_per_split": 3,
        "reproduction_runs": 2,
        "d2_core_file_count": len(D2_CORE),
        "d3_core_file_count": len(D3_CORE),
    }


def main() -> int:
    args = parse_args()
    try:
        result = (
            verify_d4_r1(args.d2_dir, args.d3_dir, args.output_dir)
            if args.verify_only
            else build_d4_r1(args.d2_dir, args.d3_dir, args.output_dir, args.workers)
        )
    except (ReproductionError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
