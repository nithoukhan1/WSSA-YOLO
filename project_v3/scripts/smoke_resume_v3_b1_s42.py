"""
Controlled interruption-and-resume smoke test for Wrist-PriViD V3-B1-S42.

This script proves that the exact-resume workflow behaves correctly
before the expensive official 100-epoch experiment begins.

Workflow:
1. Build a small original-only train/validation subset.
2. Start a 3-epoch YOLO11s run.
3. Intentionally interrupt after the second saved epoch, once the SGD momentum state has had time to initialize under AMP.
4. Verify that last.pt is unstripped and contains optimizer state.
5. Simulate a new Kaggle session by copying the partial run elsewhere.
6. Restore the partial run into a clean writable directory.
7. Resume with resume=True.
8. Verify continuous epoch history, restored optimizer state, preserved
   best.pt, and no test-data access.

This is a smoke test only. It is not an official experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------
# Frozen smoke-test identity
# ---------------------------------------------------------------------

SMOKE_EXPERIMENT_ID = "V3-B1-S42-RESUME-SMOKE"
EXPECTED_BRANCH = "v3-strong-baseline"
EXPECTED_ULTRALYTICS_VERSION = "8.4.50"

SMOKE_RUN_NAME = "wrist_privid_v3_b1_s42_resume_smoke"
SMOKE_TOTAL_EPOCHS = 3
SMOKE_INTERRUPT_AFTER_EPOCH_INDEX = 1

SMOKE_TRAIN_IMAGES = 160
SMOKE_VALID_IMAGES = 64

EXPECTED_PRETRAINED_PARAMETER_COUNT = 9_458_752
EXPECTED_NINE_CLASS_PARAMETER_COUNT = 9_431_275

REQUIRED_DISABLED_FLAGS = {
    "USE_FOCALER": "0",
    "USE_WHFE": "0",
    "USE_CBLOSS": "0",
    "USE_MCAUX": "0",
}


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse smoke-test arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Controlled interruption-and-resume smoke test for "
            "Wrist-PriViD V3-B1-S42."
        )
    )

    parser.add_argument(
        "--expected-commit",
        required=True,
        help=(
            "Full Git commit hash, or unique prefix, required "
            "for the current repository checkout."
        ),
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the official yolo11s.pt checkpoint.",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/kaggle/input/datasets/nithoukhan/"
            "grazpedwri-dx-aug/GRAZPEDWRI-DX"
        ),
        help="Path to the canonical GRAZPEDWRI-DX dataset root.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working"),
        help="Writable root used for all smoke-test artifacts.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader worker count.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Calculate SHA-256 for one file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def run_git(
    repository_root: Path,
    arguments: list[str],
) -> str:
    """Run Git and return stripped stdout."""

    return subprocess.check_output(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
    ).strip()


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def read_csv_rows(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file safely."""

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise AssertionError(
                f"CSV has no header: {path}"
            )

        rows = list(reader)

    return list(reader.fieldnames), rows


def locate_epoch_column(
    fieldnames: list[str],
) -> str:
    """Locate the results.csv epoch column."""

    for name in fieldnames:
        if name.strip() == "epoch":
            return name

    raise AssertionError(
        "Could not locate the epoch column."
    )


def assert_finite_numeric_results(
    rows: list[dict[str, str]],
) -> None:
    """Verify that every numeric results value is finite."""

    for row_index, row in enumerate(rows, start=1):
        for name, raw_value in row.items():
            value = str(raw_value).strip()

            if not value:
                continue

            try:
                numeric = float(value)
            except ValueError:
                continue

            if not math.isfinite(numeric):
                raise AssertionError(
                    f"Non-finite value in row {row_index}, "
                    f"column {name}: {raw_value}"
                )


# ---------------------------------------------------------------------
# Repository, flags, and architecture
# ---------------------------------------------------------------------

def verify_repository(
    repository_root: Path,
    expected_commit: str,
) -> dict[str, str]:
    """Verify branch, commit, and clean working tree."""

    branch = run_git(
        repository_root,
        ["branch", "--show-current"],
    )

    commit = run_git(
        repository_root,
        ["rev-parse", "HEAD"],
    )

    status = run_git(
        repository_root,
        ["status", "--porcelain"],
    )

    print("Branch:", branch)
    print("Commit:", commit)
    print("Working tree clean:", not bool(status))

    if branch != EXPECTED_BRANCH:
        raise AssertionError(
            f"Expected branch {EXPECTED_BRANCH!r}, "
            f"found {branch!r}."
        )

    if not commit.startswith(expected_commit):
        raise AssertionError(
            "Current commit does not match "
            "--expected-commit.\n"
            f"Expected prefix: {expected_commit}\n"
            f"Actual commit:   {commit}"
        )

    if status:
        raise AssertionError(
            "Repository contains uncommitted changes."
        )

    return {
        "branch": branch,
        "commit": commit,
    }


def set_and_verify_flags() -> dict[str, str]:
    """Disable every V2 custom feature."""

    verified: dict[str, str] = {}

    for name, expected in REQUIRED_DISABLED_FLAGS.items():
        os.environ[name] = expected

    for name, expected in REQUIRED_DISABLED_FLAGS.items():
        actual = os.environ.get(name)
        print(f"{name}: {actual}")

        if actual != expected:
            raise AssertionError(
                f"{name} must equal {expected!r}."
            )

        verified[name] = actual

    return verified


def verify_model_architecture(
    model: Any,
    expected_parameter_count: int,
    stage_name: str,
) -> dict[str, Any]:
    """Verify a standard YOLO11s architecture at a named stage."""

    module_names = [
        module.__class__.__name__
        for module in model.model.modules()
    ]

    forbidden = {
        "WHFE",
        "MCAux",
        "MCAuxHead",
    }

    present_forbidden = sorted(
        forbidden & set(module_names)
    )

    if present_forbidden:
        raise AssertionError(
            f"Forbidden custom modules found: "
            f"{present_forbidden}"
        )

    if any(
        "_mcaux" in name.lower()
        for name, _ in model.model.named_modules()
    ):
        raise AssertionError(
            "MCAux attributes were found."
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )

    if parameter_count != expected_parameter_count:
        raise AssertionError(
            f"{stage_name}: expected "
            f"{expected_parameter_count:,} parameters, "
            f"found {parameter_count:,}."
        )

    return {
        "stage": stage_name,
        "parameter_count": parameter_count,
        "module_count": len(module_names),
        "forbidden_modules_present": (
            present_forbidden
        ),
    }


# ---------------------------------------------------------------------
# Smoke dataset construction
# ---------------------------------------------------------------------

def image_to_label_path(
    image_path: Path,
) -> Path:
    """Convert a YOLO image path to its matching label path."""

    path_text = str(image_path).replace(
        "/data/images/",
        "/data/labels/",
    )

    return Path(path_text).with_suffix(".txt")


def build_smoke_dataset(
    repository_root: Path,
    dataset_root: Path,
    runtime_directory: Path,
) -> dict[str, Any]:
    """
    Rebuild the canonical original-only list, then create fixed
    smoke train/validation subsets.
    """

    builder_script = (
        repository_root
        / "project_v3"
        / "scripts"
        / "build_original_only_dataset.py"
    )

    if not builder_script.is_file():
        raise FileNotFoundError(builder_script)

    if runtime_directory.exists():
        shutil.rmtree(runtime_directory)

    subprocess.run(
        [
            sys.executable,
            str(builder_script),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(runtime_directory),
        ],
        cwd=repository_root,
        check=True,
    )

    full_train_list = (
        runtime_directory / "train_original.txt"
    )

    full_train_paths = [
        Path(line.strip())
        for line in full_train_list.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(full_train_paths) != 14_204:
        raise AssertionError(
            "Canonical original-only list is not 14,204 images."
        )

    if not all(
        path.name.startswith("orig_")
        for path in full_train_paths
    ):
        raise AssertionError(
            "A non-orig_ image entered the canonical list."
        )

    if any(
        path.name.startswith("aug_")
        for path in full_train_paths
    ):
        raise AssertionError(
            "An aug_ image entered the canonical list."
        )

    smoke_train_paths = full_train_paths[
        :SMOKE_TRAIN_IMAGES
    ]

    validation_directory = (
        dataset_root
        / "data"
        / "images"
        / "valid"
    )

    smoke_valid_paths = sorted(
        validation_directory.glob("*.png")
    )[:SMOKE_VALID_IMAGES]

    if len(smoke_train_paths) != SMOKE_TRAIN_IMAGES:
        raise AssertionError(
            "Smoke train subset count mismatch."
        )

    if len(smoke_valid_paths) != SMOKE_VALID_IMAGES:
        raise AssertionError(
            "Smoke validation subset count mismatch."
        )

    train_labels = [
        image_to_label_path(path)
        for path in smoke_train_paths
    ]

    valid_labels = [
        image_to_label_path(path)
        for path in smoke_valid_paths
    ]

    missing_train_labels = [
        path
        for path in train_labels
        if not path.is_file()
    ]

    missing_valid_labels = [
        path
        for path in valid_labels
        if not path.is_file()
    ]

    if missing_train_labels:
        raise FileNotFoundError(
            f"Missing train labels: "
            f"{missing_train_labels[:5]}"
        )

    if missing_valid_labels:
        raise FileNotFoundError(
            f"Missing validation labels: "
            f"{missing_valid_labels[:5]}"
        )

    smoke_train_list = (
        runtime_directory
        / "resume_smoke_train_160.txt"
    )

    smoke_valid_list = (
        runtime_directory
        / "resume_smoke_valid_64.txt"
    )

    smoke_data_yaml = (
        runtime_directory
        / "resume_smoke_data.yaml"
    )

    smoke_train_list.write_text(
        "\n".join(
            str(path.resolve())
            for path in smoke_train_paths
        )
        + "\n",
        encoding="utf-8",
    )

    smoke_valid_list.write_text(
        "\n".join(
            str(path.resolve())
            for path in smoke_valid_paths
        )
        + "\n",
        encoding="utf-8",
    )

    names = {
        0: "boneanomaly",
        1: "bonelesion",
        2: "foreignbody",
        3: "fracture",
        4: "metal",
        5: "periostealreaction",
        6: "pronatorsign",
        7: "softtissue",
        8: "text",
    }

    smoke_yaml = {
        "path": str(dataset_root.resolve()),
        "train": str(smoke_train_list.resolve()),
        "val": str(smoke_valid_list.resolve()),
        "nc": 9,
        "names": names,
    }

    smoke_data_yaml.write_text(
        yaml.safe_dump(
            smoke_yaml,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded_yaml = yaml.safe_load(
        smoke_data_yaml.read_text(
            encoding="utf-8"
        )
    )

    if "test" in loaded_yaml:
        raise AssertionError(
            "Smoke data YAML must not contain test."
        )

    return {
        "full_train_list": full_train_list,
        "smoke_train_list": smoke_train_list,
        "smoke_valid_list": smoke_valid_list,
        "smoke_data_yaml": smoke_data_yaml,
        "smoke_train_images": len(
            smoke_train_paths
        ),
        "smoke_valid_images": len(
            smoke_valid_paths
        ),
        "missing_train_labels": len(
            missing_train_labels
        ),
        "missing_valid_labels": len(
            missing_valid_labels
        ),
        "smoke_train_list_sha256": (
            sha256_file(smoke_train_list)
        ),
        "smoke_valid_list_sha256": (
            sha256_file(smoke_valid_list)
        ),
        "smoke_data_yaml_sha256": (
            sha256_file(smoke_data_yaml)
        ),
    }


# ---------------------------------------------------------------------
# Checkpoint verification
# ---------------------------------------------------------------------

def verify_unstripped_partial_checkpoint(
    checkpoint_path: Path,
    torch_module: Any,
) -> dict[str, Any]:
    """Verify that the interrupted last.pt can resume exactly."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch_module.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Partial checkpoint is not a dictionary."
        )

    required_keys = {
        "epoch",
        "optimizer",
        "scaler",
        "train_args",
        "ema",
        "updates",
    }

    missing = required_keys - set(checkpoint)

    if missing:
        raise AssertionError(
            f"Partial checkpoint missing keys: "
            f"{sorted(missing)}"
        )

    epoch_index = int(checkpoint["epoch"])
    optimizer = checkpoint["optimizer"]
    scaler = checkpoint["scaler"]
    train_args = checkpoint["train_args"]

    if epoch_index != SMOKE_INTERRUPT_AFTER_EPOCH_INDEX:
        raise AssertionError(
            "Interrupted checkpoint epoch does not match the "
            "planned second-epoch interruption."
        )

    if optimizer is None:
        raise AssertionError(
            "Partial last.pt optimizer is missing."
        )

    if not isinstance(optimizer, dict):
        raise AssertionError(
            "Optimizer payload has an unexpected type."
        )

    optimizer_state = optimizer.get("state", {})

    if not optimizer_state:
        raise AssertionError(
            "Partial last.pt optimizer state is empty."
        )

    if not isinstance(scaler, dict):
        raise AssertionError(
            "AMP scaler state is missing or invalid."
        )

    if not isinstance(train_args, dict):
        raise AssertionError(
            "train_args is missing or invalid."
        )

    expected_args = {
        "epochs": SMOKE_TOTAL_EPOCHS,
        "imgsz": 1024,
        "batch": 16,
        "seed": 42,
        "name": SMOKE_RUN_NAME,
    }

    for name, expected in expected_args.items():
        actual = train_args.get(name)

        if actual != expected:
            raise AssertionError(
                f"Checkpoint train_args mismatch for {name}: "
                f"expected {expected!r}, found {actual!r}."
            )

    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(
            checkpoint_path
        ),
        "size_bytes": checkpoint_path.stat().st_size,
        "stored_epoch_index_zero_based": epoch_index,
        "completed_epochs": epoch_index + 1,
        "remaining_epochs": (
            SMOKE_TOTAL_EPOCHS - (epoch_index + 1)
        ),
        "optimizer_state_entries": len(
            optimizer_state
        ),
        "scaler_state_entries": len(
            scaler
        ),
        "best_fitness": checkpoint.get(
            "best_fitness"
        ),
        "updates": checkpoint.get("updates"),
        "version": checkpoint.get("version"),
    }


# ---------------------------------------------------------------------
# Cross-session staging
# ---------------------------------------------------------------------

def copy_partial_run_to_simulated_input(
    source_run_directory: Path,
    simulated_input_directory: Path,
) -> dict[str, Any]:
    """Copy the interrupted run to a simulated read-only input."""

    if simulated_input_directory.exists():
        raise FileExistsError(
            simulated_input_directory
        )

    shutil.copytree(
        source_run_directory,
        simulated_input_directory,
    )

    source_files = {
        path.relative_to(
            source_run_directory
        ).as_posix(): path
        for path in source_run_directory.rglob("*")
        if path.is_file()
    }

    copied_files = {
        path.relative_to(
            simulated_input_directory
        ).as_posix(): path
        for path in simulated_input_directory.rglob("*")
        if path.is_file()
    }

    if set(source_files) != set(copied_files):
        raise AssertionError(
            "Simulated input file set differs from source."
        )

    hashes: dict[str, str] = {}

    for relative_path, source in source_files.items():
        copied = copied_files[relative_path]

        source_hash = sha256_file(source)
        copied_hash = sha256_file(copied)

        if source_hash != copied_hash:
            raise AssertionError(
                f"Copied file hash mismatch: "
                f"{relative_path}"
            )

        hashes[relative_path] = copied_hash

    return {
        "directory": str(
            simulated_input_directory
        ),
        "file_count": len(hashes),
        "file_hashes": hashes,
    }


def stage_partial_run_for_resume(
    simulated_input_directory: Path,
    destination_run_directory: Path,
) -> tuple[Path, dict[str, Any]]:
    """
    Restore continuity-critical artifacts into a clean writable
    run directory.
    """

    if destination_run_directory.exists():
        raise FileExistsError(
            destination_run_directory
        )

    source_files = {
        "args_yaml": (
            simulated_input_directory
            / "args.yaml"
        ),
        "results_csv": (
            simulated_input_directory
            / "results.csv"
        ),
        "last_checkpoint": (
            simulated_input_directory
            / "weights"
            / "last.pt"
        ),
        "best_checkpoint": (
            simulated_input_directory
            / "weights"
            / "best.pt"
        ),
    }

    missing = [
        str(path)
        for path in source_files.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Simulated input is missing required files:\n"
            + "\n".join(missing)
        )

    destination_weights = (
        destination_run_directory / "weights"
    )

    destination_weights.mkdir(
        parents=True,
        exist_ok=False,
    )

    destination_files = {
        "args_yaml": (
            destination_run_directory / "args.yaml"
        ),
        "results_csv": (
            destination_run_directory
            / "results.csv"
        ),
        "last_checkpoint": (
            destination_weights / "last.pt"
        ),
        "best_checkpoint": (
            destination_weights / "best.pt"
        ),
    }

    staging_rows: dict[str, Any] = {}

    for name, source in source_files.items():
        destination = destination_files[name]

        shutil.copy2(source, destination)

        source_hash = sha256_file(source)
        destination_hash = sha256_file(
            destination
        )

        if source_hash != destination_hash:
            raise AssertionError(
                f"Staged hash mismatch for {name}."
            )

        staging_rows[name] = {
            "source": str(source),
            "destination": str(destination),
            "sha256": destination_hash,
            "size_bytes": destination.stat().st_size,
        }

    return (
        destination_files["last_checkpoint"],
        {
            "source_directory": str(
                simulated_input_directory
            ),
            "destination_directory": str(
                destination_run_directory
            ),
            "files": staging_rows,
        },
    )


# ---------------------------------------------------------------------
# Intentional interruption callback
# ---------------------------------------------------------------------

class IntentionalSmokeInterruption(RuntimeError):
    """Expected exception used to simulate Kaggle interruption."""


def interrupt_after_second_saved_epoch(
    trainer: Any,
) -> None:
    """
    Interrupt after epoch index 1, once optimizer state exists.

    The first AMP epoch can legitimately contain no serialized
    SGD momentum buffers when GradScaler skips every early step.
    Waiting for the second saved epoch tests a genuinely resumable
    optimizer state without changing the official training setup.
    """

    if (
        int(trainer.epoch)
        == SMOKE_INTERRUPT_AFTER_EPOCH_INDEX
    ):
        optimizer_state_entries = len(trainer.optimizer.state)

        print(
            "In-memory optimizer state entries before interruption:",
            optimizer_state_entries,
        )

        if optimizer_state_entries <= 0:
            raise AssertionError(
                "Optimizer state is still empty after two complete "
                "epochs. Exact optimizer-state resume cannot yet be "
                "validated."
            )

        if not Path(trainer.last).is_file():
            raise AssertionError(
                "last.pt was not written before interruption."
            )

        if not Path(trainer.csv).is_file():
            raise AssertionError(
                "results.csv was not written before interruption."
            )

        raise IntentionalSmokeInterruption(
            "Expected smoke interruption after the second saved epoch."
        )


# ---------------------------------------------------------------------
# Main smoke workflow
# ---------------------------------------------------------------------

def main() -> None:
    """Run the controlled interruption-and-resume smoke test."""

    arguments = parse_arguments()

    repository_root = (
        Path(__file__).resolve().parents[2]
    )

    ultralytics_root = (
        repository_root / "ultralytics-main"
    )

    dataset_root = (
        arguments.dataset_root
        .expanduser()
        .resolve()
    )

    weights_path = (
        arguments.weights
        .expanduser()
        .resolve()
    )

    output_root = (
        arguments.output_root
        .expanduser()
        .resolve()
    )

    runtime_directory = (
        output_root
        / "wrist_privid_v3_resume_smoke_runtime"
    )

    source_run_directory = (
        output_root / SMOKE_RUN_NAME
    )

    simulated_input_root = (
        output_root
        / "wrist_privid_v3_resume_smoke_simulated_input"
    )

    simulated_input_directory = (
        simulated_input_root
        / SMOKE_RUN_NAME
    )

    summary_directory = (
        output_root
        / "wrist_privid_v3_resume_smoke_summary"
    )

    for path in (
        source_run_directory,
        simulated_input_root,
        runtime_directory,
        summary_directory,
    ):
        if path.exists():
            raise FileExistsError(
                f"Smoke-test path already exists: {path}\n"
                "Use a clean Kaggle session or manually remove only "
                "a verified previous smoke-test output."
            )

    print("=" * 96)
    print("WRIST-PRIVID V3 CONTROLLED RESUME SMOKE TEST")
    print("=" * 96)

    print("\n1. REPOSITORY VERIFICATION")
    print("-" * 96)

    repository_info = verify_repository(
        repository_root,
        arguments.expected_commit,
    )

    print("✓ Repository verification passed")

    print("\n2. ENVIRONMENT FLAGS")
    print("-" * 96)

    flags = set_and_verify_flags()

    print("✓ All custom features disabled")

    print("\n3. GPU AND PRETRAINED WEIGHTS")
    print("-" * 96)

    if not weights_path.is_file():
        raise FileNotFoundError(
            weights_path
        )

    if weights_path.name != "yolo11s.pt":
        raise AssertionError(
            "Weights file must be named yolo11s.pt."
        )

    if not ultralytics_root.is_dir():
        raise FileNotFoundError(
            ultralytics_root
        )

    sys.path.insert(
        0,
        str(ultralytics_root),
    )

    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Enable a Kaggle GPU."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Weights:", weights_path)
    print(
        "Weights SHA-256:",
        sha256_file(weights_path),
    )
    print(
        "Ultralytics version:",
        ultralytics.__version__,
    )
    print(
        "Ultralytics source:",
        Path(ultralytics.__file__).resolve(),
    )

    if (
        ultralytics.__version__
        != EXPECTED_ULTRALYTICS_VERSION
    ):
        raise AssertionError(
            f"Expected Ultralytics "
            f"{EXPECTED_ULTRALYTICS_VERSION}."
        )

    if (
        Path(ultralytics.__file__)
        .resolve()
        .parents[1]
        != ultralytics_root.resolve()
    ):
        raise AssertionError(
            "Ultralytics was not imported from the repository."
        )

    print("✓ GPU, weights, and Ultralytics verified")

    print("\n4. SMOKE DATASET")
    print("-" * 96)

    dataset_info = build_smoke_dataset(
        repository_root,
        dataset_root,
        runtime_directory,
    )

    for key, value in dataset_info.items():
        print(f"{key}: {value}")

    print("✓ Smoke dataset verified")
    print("✓ Official test partition absent")

    print("\n5. PHASE 1 — INTENTIONAL INTERRUPTION")
    print("-" * 96)

    model = YOLO(str(weights_path))

    pretrained_architecture = verify_model_architecture(
        model,
        expected_parameter_count=(
            EXPECTED_PRETRAINED_PARAMETER_COUNT
        ),
        stage_name="official pretrained 80-class YOLO11s",
    )

    print(
        "Parameter count:",
        pretrained_architecture["parameter_count"],
    )

    model.add_callback(
        "on_model_save",
        interrupt_after_second_saved_epoch,
    )

    interruption_observed = False

    try:
        model.train(
            data=str(
                dataset_info["smoke_data_yaml"]
            ),
            project=str(output_root),
            name=SMOKE_RUN_NAME,
            exist_ok=False,

            imgsz=1024,
            epochs=SMOKE_TOTAL_EPOCHS,
            patience=SMOKE_TOTAL_EPOCHS,
            batch=16,
            device=arguments.device,
            workers=arguments.workers,

            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            cos_lr=True,

            amp=True,
            deterministic=True,
            seed=42,

            box=7.5,
            cls=2.5,
            dfl=1.5,
            cls_pw=0.0,

            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            close_mosaic=1,

            save=True,
            save_period=1,
            plots=False,
            val=True,
            verbose=True,
        )

    except IntentionalSmokeInterruption as error:
        interruption_observed = True
        print("Expected interruption:", error)

    if not interruption_observed:
        raise AssertionError(
            "The intentional interruption callback did not fire."
        )

    source_args = (
        source_run_directory / "args.yaml"
    )

    source_results = (
        source_run_directory / "results.csv"
    )

    source_last = (
        source_run_directory
        / "weights"
        / "last.pt"
    )

    source_best = (
        source_run_directory
        / "weights"
        / "best.pt"
    )

    for path in (
        source_args,
        source_results,
        source_last,
        source_best,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_fieldnames, source_rows = (
        read_csv_rows(source_results)
    )

    source_epoch_column = (
        locate_epoch_column(
            source_fieldnames
        )
    )

    source_epoch_values = [
        int(float(row[source_epoch_column]))
        for row in source_rows
    ]

    if source_epoch_values != [1, 2]:
        raise AssertionError(
            f"Expected source epoch history [1, 2], "
            f"found {source_epoch_values}."
        )

    assert_finite_numeric_results(
        source_rows
    )

    partial_checkpoint_info = (
        verify_unstripped_partial_checkpoint(
            source_last,
            torch,
        )
    )

    source_results_text = (
        source_results.read_text(
            encoding="utf-8-sig"
        )
    )

    source_results_hash = (
        sha256_file(source_results)
    )

    source_best_hash = (
        sha256_file(source_best)
    )

    source_last_hash = (
        sha256_file(source_last)
    )

    print(
        "Partial last.pt SHA-256:",
        source_last_hash,
    )
    print(
        "Optimizer state entries:",
        partial_checkpoint_info[
            "optimizer_state_entries"
        ],
    )
    print("✓ Two epochs saved before interruption")
    print("✓ Partial last.pt is unstripped")
    print("✓ Optimizer and scaler states are present")

    del model
    torch.cuda.empty_cache()

    print("\n6. SIMULATED NEW KAGGLE SESSION")
    print("-" * 96)

    simulated_input_info = (
        copy_partial_run_to_simulated_input(
            source_run_directory,
            simulated_input_directory,
        )
    )

    shutil.rmtree(source_run_directory)

    if source_run_directory.exists():
        raise AssertionError(
            "Source run directory was not removed."
        )

    staged_last, staging_info = (
        stage_partial_run_for_resume(
            simulated_input_directory,
            source_run_directory,
        )
    )

    if (
        sha256_file(staged_last)
        != source_last_hash
    ):
        raise AssertionError(
            "Staged last.pt differs from source last.pt."
        )

    if (
        sha256_file(
            source_run_directory
            / "results.csv"
        )
        != source_results_hash
    ):
        raise AssertionError(
            "Staged results.csv differs from source."
        )

    if (
        sha256_file(
            source_run_directory
            / "weights"
            / "best.pt"
        )
        != source_best_hash
    ):
        raise AssertionError(
            "Staged best.pt differs from source best.pt."
        )

    print("✓ Previous results.csv staged unchanged")
    print("✓ Previous best.pt staged unchanged")
    print("✓ Previous unstripped last.pt staged unchanged")

    print("\n7. PHASE 2 — EXACT RESUME")
    print("-" * 96)

    resumed_model = YOLO(
        str(staged_last)
    )

    resumed_architecture = (
        verify_model_architecture(
            resumed_model,
            expected_parameter_count=(
                EXPECTED_NINE_CLASS_PARAMETER_COUNT
            ),
            stage_name="resumable 9-class YOLO11s checkpoint",
        )
    )

    if (
        resumed_architecture["forbidden_modules_present"]
        != pretrained_architecture["forbidden_modules_present"]
    ):
        raise AssertionError(
            "Custom-module status changed between initialization "
            "and resume."
        )

    resume_observation: dict[str, Any] = {}

    def observe_resumed_trainer(
        trainer: Any,
    ) -> None:
        """Capture restored state after trainer setup."""

        resume_observation[
            "start_epoch_zero_based"
        ] = int(trainer.start_epoch)

        resume_observation[
            "optimizer_state_entries"
        ] = len(trainer.optimizer.state)

        resume_observation[
            "scheduler_last_epoch"
        ] = int(
            trainer.scheduler.last_epoch
        )

        resume_observation[
            "results_csv_existed_on_train_start"
        ] = Path(trainer.csv).is_file()

    resumed_model.add_callback(
        "on_train_start",
        observe_resumed_trainer,
    )

    resumed_model.train(
        resume=True,
        device=arguments.device,
        workers=arguments.workers,
        plots=False,
    )

    trainer = resumed_model.trainer

    if trainer is None:
        raise RuntimeError(
            "Resumed trainer was not retained."
        )

    actual_run_directory = Path(
        trainer.save_dir
    ).resolve()

    if (
        actual_run_directory
        != source_run_directory.resolve()
    ):
        raise AssertionError(
            "Resumed output directory mismatch.\n"
            f"Expected: {source_run_directory}\n"
            f"Actual:   {actual_run_directory}"
        )

    print("Resume observation:", resume_observation)

    if (
        resume_observation.get(
            "start_epoch_zero_based"
        )
        != 2
    ):
        raise AssertionError(
            "Resume did not start from epoch index 2."
        )

    if (
        int(
            resume_observation.get(
                "optimizer_state_entries",
                0,
            )
        )
        <= 0
    ):
        raise AssertionError(
            "Optimizer state was not restored."
        )

    if not resume_observation.get(
        "results_csv_existed_on_train_start",
        False,
    ):
        raise AssertionError(
            "Previous results.csv was not present at resume start."
        )

    print("✓ Resume started from epoch 3")
    print("✓ Optimizer state restored before resumed training")
    print("✓ Previous results.csv present before resumed training")

    print("\n8. FINAL CONTINUITY VERIFICATION")
    print("-" * 96)

    final_results = (
        source_run_directory
        / "results.csv"
    )

    final_args = (
        source_run_directory
        / "args.yaml"
    )

    final_best = (
        source_run_directory
        / "weights"
        / "best.pt"
    )

    final_last = (
        source_run_directory
        / "weights"
        / "last.pt"
    )

    for path in (
        final_results,
        final_args,
        final_best,
        final_last,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    final_fieldnames, final_rows = (
        read_csv_rows(final_results)
    )

    final_epoch_column = (
        locate_epoch_column(
            final_fieldnames
        )
    )

    final_epoch_values = [
        int(float(row[final_epoch_column]))
        for row in final_rows
    ]

    if final_epoch_values != [1, 2, 3]:
        raise AssertionError(
            "Final epoch history is not continuous.\n"
            f"Expected: [1, 2, 3]\n"
            f"Actual:   {final_epoch_values}"
        )

    if len(final_rows) != SMOKE_TOTAL_EPOCHS:
        raise AssertionError(
            "Final results.csv row count mismatch."
        )

    assert_finite_numeric_results(
        final_rows
    )

    final_results_text = (
        final_results.read_text(
            encoding="utf-8-sig"
        )
    )

    source_lines = [
        line
        for line in source_results_text.splitlines()
        if line.strip()
    ]

    final_lines = [
        line
        for line in final_results_text.splitlines()
        if line.strip()
    ]

    if len(source_lines) != 3:
        raise AssertionError(
            "Interrupted source results should contain "
            "one header and two epoch rows."
        )

    if final_lines[:3] != source_lines:
        raise AssertionError(
            "The original results.csv header or first epoch row "
            "was modified during resume."
        )

    final_checkpoint = torch.load(
        final_last,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(final_checkpoint, dict):
        raise TypeError(
            "Final last.pt is not a checkpoint dictionary."
        )

    # Natural completion strips optimizer and sets epoch to -1.
    if final_checkpoint.get("optimizer") is not None:
        raise AssertionError(
            "Naturally completed final last.pt should be stripped."
        )

    if int(final_checkpoint.get("epoch", -999)) != -1:
        raise AssertionError(
            "Naturally completed final last.pt should have epoch=-1."
        )

    simulated_hashes_after = {
        path.relative_to(
            simulated_input_directory
        ).as_posix(): sha256_file(path)
        for path in simulated_input_directory.rglob("*")
        if path.is_file()
    }

    if (
        simulated_hashes_after
        != simulated_input_info["file_hashes"]
    ):
        raise AssertionError(
            "Simulated input artifacts changed during resume."
        )

    summary_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    summary = {
        "smoke_experiment_id": (
            SMOKE_EXPERIMENT_ID
        ),
        "official_experiment": False,
        "test_partition_used": False,
        "repository": repository_info,
        "flags": flags,
        "dataset": {
            key: str(value)
            if isinstance(value, Path)
            else value
            for key, value in dataset_info.items()
        },
        "architectures": {
            "pretrained_initialization": (
                pretrained_architecture
            ),
            "resumable_checkpoint": (
                resumed_architecture
            ),
        },
        "phase_1": {
            "interrupted_after_epoch_index": (
                SMOKE_INTERRUPT_AFTER_EPOCH_INDEX
            ),
            "results_epoch_values": (
                source_epoch_values
            ),
            "results_sha256": (
                source_results_hash
            ),
            "best_sha256": source_best_hash,
            "last_sha256": source_last_hash,
            "partial_checkpoint": (
                partial_checkpoint_info
            ),
        },
        "simulated_input": (
            simulated_input_info
        ),
        "staging": staging_info,
        "resume_observation": (
            resume_observation
        ),
        "final": {
            "run_directory": str(
                source_run_directory
            ),
            "epoch_values": (
                final_epoch_values
            ),
            "results_sha256": (
                sha256_file(final_results)
            ),
            "best_sha256": (
                sha256_file(final_best)
            ),
            "last_sha256": (
                sha256_file(final_last)
            ),
            "final_checkpoint_stripped": True,
            "source_first_epoch_preserved": True,
            "simulated_input_unchanged": True,
        },
    }

    summary_path = (
        summary_directory
        / "v3_b1_s42_resume_smoke_summary.json"
    )

    write_json(
        summary_path,
        summary,
    )

    print("Summary:", summary_path)
    print(
        "Summary SHA-256:",
        sha256_file(summary_path),
    )

    print("\n" + "=" * 96)
    print("✓ V3-B1-S42 RESUME SMOKE TEST PASSED")
    print("✓ INTENTIONAL INTERRUPTION OCCURRED AFTER CHECKPOINT SAVE")
    print("✓ PARTIAL LAST.PT CONTAINED OPTIMIZER AND SCALER STATE")
    print("✓ PREVIOUS RESULTS.CSV AND BEST.PT WERE PRESERVED")
    print("✓ RESUME STARTED FROM EPOCH 3, NOT EPOCH 1")
    print("✓ OPTIMIZER STATE WAS RESTORED")
    print("✓ EPOCH HISTORY IS CONTINUOUS: 1, 2, 3")
    print("✓ ORIGINAL FIRST-EPOCH RESULT WAS NOT MODIFIED")
    print("✓ SIMULATED INPUT ARTIFACTS REMAINED UNCHANGED")
    print("✓ FINAL CHECKPOINTS AND RESULTS WERE SAVED")
    print("✓ OFFICIAL TEST PARTITION WAS NOT ACCESSED")
    print("=" * 96)


if __name__ == "__main__":
    main()