"""
Session-safe segmented launcher for Wrist-PriViD V3-B1-S42.

This script supports both:

1. Starting the official controlled B1 run from the frozen yolo11s.pt.
2. Resuming the same run from an unstripped weights/last.pt produced by
   an earlier Kaggle session.

For an intermediate segment, the launcher intentionally stops only
after the requested global epoch has been fully trained, validated,
written to results.csv, and saved to an unstripped last.pt. This keeps
the optimizer, scheduler, EMA, AMP scaler, and epoch state available
for exact continuation in a later Kaggle session.

The official test partition is never accessed.
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
# Frozen experiment identity
# ---------------------------------------------------------------------

EXPERIMENT_ID = "V3-B1-S42"
EXPECTED_BRANCH = "v3-strong-baseline"
EXPECTED_ULTRALYTICS_VERSION = "8.4.50"
EXPECTED_RUN_NAME = "wrist_privid_v3_b1_s42"

EXPECTED_TRAIN_IMAGES = 14_204
EXPECTED_VALID_IMAGES = 4_094
EXPECTED_MAX_EPOCHS = 100

EXPECTED_OFFICIAL_WEIGHTS_NAME = "yolo11s.pt"
EXPECTED_OFFICIAL_WEIGHTS_SHA256 = (
    "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
)

EXPECTED_TRAIN_LIST_SHA256 = (
    "2577b30d84e9273386e94ab3737fbf65a4b3f59031d5b604a96eb0e34cc055cc"
)
EXPECTED_DATA_YAML_SHA256 = (
    "8ab985997ee88455ea9861cdf915fcfa6a2c6b0dbb3a731045156250d40fd51d"
)

EXPECTED_PRETRAINED_PARAMETER_COUNT = 9_458_752
EXPECTED_NINE_CLASS_PARAMETER_COUNT = 9_431_275

REQUIRED_DISABLED_FLAGS = {
    "USE_FOCALER": "0",
    "USE_WHFE": "0",
    "USE_CBLOSS": "0",
    "USE_MCAUX": "0",
}


# ---------------------------------------------------------------------
# Frozen controlled B1 arguments
# ---------------------------------------------------------------------

FROZEN_TRAIN_ARGUMENTS: dict[str, Any] = {
    "imgsz": 1024,
    "epochs": 100,
    "patience": 50,
    "batch": 16,
    "device": 0,
    "workers": 4,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "cos_lr": True,
    "amp": True,
    "deterministic": True,
    "seed": 42,
    "box": 7.5,
    "cls": 2.5,
    "dfl": 1.5,
    "cls_pw": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "close_mosaic": 15,
}


# ---------------------------------------------------------------------
# Expected controlled configuration fields
# ---------------------------------------------------------------------

CONFIG_FIELD_MAP = {
    "imgsz": ("training", "image_size"),
    "epochs": ("training", "epochs"),
    "patience": ("training", "patience"),
    "batch": ("training", "batch_size"),
    "workers": ("training", "workers"),
    "optimizer": ("training", "optimizer"),
    "lr0": ("training", "initial_learning_rate"),
    "lrf": ("training", "final_learning_rate_fraction"),
    "momentum": ("training", "momentum"),
    "weight_decay": ("training", "weight_decay"),
    "warmup_epochs": ("training", "warmup_epochs"),
    "cos_lr": ("training", "cosine_learning_rate"),
    "amp": ("training", "amp"),
    "deterministic": ("training", "deterministic"),
    "seed": ("training", "seed"),
    "box": ("loss", "box_gain"),
    "cls": ("loss", "classification_gain"),
    "dfl": ("loss", "dfl_gain"),
    "cls_pw": ("loss", "classification_positive_weight"),
    "hsv_h": ("augmentation", "hsv_h"),
    "hsv_s": ("augmentation", "hsv_s"),
    "hsv_v": ("augmentation", "hsv_v"),
    "degrees": ("augmentation", "degrees"),
    "translate": ("augmentation", "translate"),
    "scale": ("augmentation", "scale"),
    "shear": ("augmentation", "shear"),
    "perspective": ("augmentation", "perspective"),
    "flipud": ("augmentation", "flip_vertical"),
    "fliplr": ("augmentation", "flip_horizontal"),
    "mosaic": ("augmentation", "mosaic"),
    "mixup": ("augmentation", "mixup"),
    "close_mosaic": ("augmentation", "close_mosaic_epochs"),
}


# ---------------------------------------------------------------------
# Planned stop exception
# ---------------------------------------------------------------------

class PlannedSegmentStop(RuntimeError):
    """Expected exception used to end an intermediate Kaggle segment."""


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse the session-safe segment launcher arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Start or exactly resume one session-safe segment of "
            "Wrist-PriViD experiment V3-B1-S42."
        )
    )

    parser.add_argument(
        "--expected-commit",
        required=True,
        help=(
            "Full Git commit hash, or a unique prefix, required "
            "for the current repository checkout."
        ),
    )

    source_group = parser.add_mutually_exclusive_group(
        required=True,
    )

    source_group.add_argument(
        "--weights",
        type=Path,
        help=(
            "Start a fresh official run from the frozen "
            "official yolo11s.pt."
        ),
    )

    source_group.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Resume from a previous segment's unstripped "
            "weights/last.pt."
        ),
    )

    parser.add_argument(
        "--expected-checkpoint-sha256",
        help=(
            "Required with --checkpoint. Exact SHA-256 of the "
            "previous segment's unstripped last.pt."
        ),
    )

    parser.add_argument(
        "--segment-end-epoch",
        type=int,
        required=True,
        help=(
            "Global 1-based epoch after which this Kaggle segment "
            "must stop. Use 100 to allow natural completion."
        ),
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
        help="Writable Kaggle output root.",
    )

    parser.add_argument(
        "--run-name",
        default=EXPECTED_RUN_NAME,
        help="Frozen official run directory name.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device used as an operational override.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers used as an operational override.",
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run all verification checks and stop before staging "
            "or GPU training."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Calculate one file's SHA-256 digest."""

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
    """Write stable, human-readable JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def values_equal(
    actual: Any,
    expected: Any,
) -> bool:
    """Compare stored configuration values safely."""

    if isinstance(expected, bool):
        return bool(actual) is expected

    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False

    if isinstance(expected, float):
        try:
            return math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            return False

    return actual == expected


def read_results_csv(
    results_path: Path,
) -> dict[str, Any]:
    """Read and verify one Ultralytics results.csv."""

    if not results_path.is_file():
        raise FileNotFoundError(results_path)

    with results_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise AssertionError(
                f"CSV has no header: {results_path}"
            )

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if not rows:
        raise AssertionError(
            f"results.csv contains no epoch rows: {results_path}"
        )

    epoch_columns = [
        name
        for name in fieldnames
        if name.strip() == "epoch"
    ]

    if len(epoch_columns) != 1:
        raise AssertionError(
            "Could not uniquely identify the epoch column."
        )

    epoch_column = epoch_columns[0]

    epoch_values = [
        int(float(row[epoch_column]))
        for row in rows
    ]

    expected_epoch_values = list(
        range(1, len(rows) + 1)
    )

    if epoch_values != expected_epoch_values:
        raise AssertionError(
            "Epoch history is not continuous.\n"
            f"Expected: {expected_epoch_values[:5]}...{expected_epoch_values[-5:]}\n"
            f"Actual:   {epoch_values[:5]}...{epoch_values[-5:]}"
        )

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
                    f"Non-finite result in row {row_index}, "
                    f"column {name}: {raw_value}"
                )

    return {
        "path": str(results_path),
        "rows": len(rows),
        "first_epoch": epoch_values[0],
        "last_epoch": epoch_values[-1],
        "sha256": sha256_file(results_path),
    }


# ---------------------------------------------------------------------
# Repository, config, and flags
# ---------------------------------------------------------------------

def verify_repository(
    repository_root: Path,
    expected_commit: str,
) -> dict[str, str]:
    """Verify branch, immutable commit, and clean working tree."""

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
            f"Expected branch {EXPECTED_BRANCH!r}, found {branch!r}."
        )

    if not commit.startswith(expected_commit):
        raise AssertionError(
            "Current commit does not match --expected-commit.\n"
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


def verify_repository_config(
    config_path: Path,
) -> dict[str, Any]:
    """Verify the checked-in controlled B1 configuration."""

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if config["experiment"]["id"] != EXPERIMENT_ID:
        raise AssertionError(
            "Incorrect experiment ID in baseline configuration."
        )

    if config["experiment"]["branch"] != EXPECTED_BRANCH:
        raise AssertionError(
            "Incorrect branch in baseline configuration."
        )

    if config["experiment"]["test_partition_allowed"]:
        raise AssertionError(
            "Baseline configuration incorrectly permits test access."
        )

    if config["data"]["expected_train_images"] != EXPECTED_TRAIN_IMAGES:
        raise AssertionError(
            "Incorrect training-image count in configuration."
        )

    if config["data"]["expected_validation_images"] != EXPECTED_VALID_IMAGES:
        raise AssertionError(
            "Incorrect validation-image count in configuration."
        )

    if config["data"]["train_allowed_prefix"] != "orig_":
        raise AssertionError(
            "Allowed train prefix must be orig_."
        )

    if config["data"]["train_forbidden_prefix"] != "aug_":
        raise AssertionError(
            "Forbidden train prefix must be aug_."
        )

    for argument_name, (
        section,
        field,
    ) in CONFIG_FIELD_MAP.items():
        expected = FROZEN_TRAIN_ARGUMENTS[argument_name]
        actual = config[section][field]

        if not values_equal(actual, expected):
            raise AssertionError(
                f"Configuration mismatch for {argument_name}: "
                f"expected {expected!r}, found {actual!r}."
            )

    return config


def set_and_verify_environment_flags() -> dict[str, str]:
    """Disable and verify every V2 custom feature."""

    for name, value in REQUIRED_DISABLED_FLAGS.items():
        os.environ[name] = value

    verified: dict[str, str] = {}

    for name, expected in REQUIRED_DISABLED_FLAGS.items():
        actual = os.environ.get(name)
        print(f"{name}: {actual}")

        if actual != expected:
            raise AssertionError(
                f"{name} must equal {expected!r}, found {actual!r}."
            )

        verified[name] = actual

    return verified


# ---------------------------------------------------------------------
# Runtime dataset reconstruction
# ---------------------------------------------------------------------

def rebuild_runtime_dataset(
    repository_root: Path,
    dataset_root: Path,
    runtime_directory: Path,
) -> dict[str, Any]:
    """Rebuild and independently verify the original-only dataset."""

    builder = (
        repository_root
        / "project_v3"
        / "scripts"
        / "build_original_only_dataset.py"
    )

    if not builder.is_file():
        raise FileNotFoundError(builder)

    if runtime_directory.exists():
        shutil.rmtree(runtime_directory)

    subprocess.run(
        [
            sys.executable,
            str(builder),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(runtime_directory),
        ],
        cwd=repository_root,
        check=True,
    )

    train_list = runtime_directory / "train_original.txt"
    data_yaml = runtime_directory / "grazped_original_only.yaml"
    summary = runtime_directory / "original_only_dataset_summary.json"

    for path in (train_list, data_yaml, summary):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_paths = [
        Path(line.strip())
        for line in train_list.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(train_paths) != EXPECTED_TRAIN_IMAGES:
        raise AssertionError(
            f"Expected {EXPECTED_TRAIN_IMAGES} train paths, "
            f"found {len(train_paths)}."
        )

    if len({str(path.resolve()) for path in train_paths}) != EXPECTED_TRAIN_IMAGES:
        raise AssertionError(
            "Duplicate training paths were found."
        )

    if not all(
        path.name.startswith("orig_")
        for path in train_paths
    ):
        raise AssertionError(
            "A non-orig_ path entered the training list."
        )

    if any(
        path.name.startswith("aug_")
        for path in train_paths
    ):
        raise AssertionError(
            "An aug_ path entered the training list."
        )

    if any(
        not path.is_file()
        for path in train_paths
    ):
        raise AssertionError(
            "The training list contains missing files."
        )

    runtime_yaml = yaml.safe_load(
        data_yaml.read_text(encoding="utf-8")
    )

    if "test" in runtime_yaml:
        raise AssertionError(
            "Runtime data YAML must not contain a test split."
        )

    validation_directory = Path(runtime_yaml["val"])

    validation_images = sorted(
        validation_directory.glob("*.png")
    )

    if len(validation_images) != EXPECTED_VALID_IMAGES:
        raise AssertionError(
            f"Expected {EXPECTED_VALID_IMAGES} validation images, "
            f"found {len(validation_images)}."
        )

    train_list_hash = sha256_file(train_list)
    data_yaml_hash = sha256_file(data_yaml)

    if train_list_hash != EXPECTED_TRAIN_LIST_SHA256:
        raise AssertionError(
            "Rebuilt train-list SHA-256 differs from the "
            "verified V3-B1 preflight artifact."
        )

    if data_yaml_hash != EXPECTED_DATA_YAML_SHA256:
        raise AssertionError(
            "Rebuilt data-YAML SHA-256 differs from the "
            "verified V3-B1 preflight artifact."
        )

    return {
        "train_list": train_list,
        "data_yaml": data_yaml,
        "summary": summary,
        "training_images": len(train_paths),
        "validation_images": len(validation_images),
        "train_list_sha256": train_list_hash,
        "data_yaml_sha256": data_yaml_hash,
    }


# ---------------------------------------------------------------------
# Model and source-state verification
# ---------------------------------------------------------------------

def verify_standard_architecture(
    model: Any,
    expected_parameter_count: int,
    stage_name: str,
) -> dict[str, Any]:
    """Verify standard YOLO11s and absence of V2 custom modules."""

    module_class_names = [
        module.__class__.__name__
        for module in model.model.modules()
    ]

    forbidden = {
        "WHFE",
        "MCAux",
        "MCAuxHead",
    }

    present_forbidden = sorted(
        forbidden & set(module_class_names)
    )

    if present_forbidden:
        raise AssertionError(
            f"{stage_name}: forbidden modules found: "
            f"{present_forbidden}"
        )

    if any(
        "_mcaux" in name.lower()
        for name, _ in model.model.named_modules()
    ):
        raise AssertionError(
            f"{stage_name}: MCAux attributes were found."
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
        "module_count": len(module_class_names),
        "forbidden_modules_present": present_forbidden,
    }


def verify_official_weights(
    weights_path: Path,
) -> dict[str, Any]:
    """Verify the frozen official YOLO11s initialization."""

    resolved = weights_path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    if resolved.name != EXPECTED_OFFICIAL_WEIGHTS_NAME:
        raise AssertionError(
            f"Official weights must be named "
            f"{EXPECTED_OFFICIAL_WEIGHTS_NAME!r}."
        )

    actual_hash = sha256_file(resolved)

    if actual_hash != EXPECTED_OFFICIAL_WEIGHTS_SHA256:
        raise AssertionError(
            "Official yolo11s.pt SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_OFFICIAL_WEIGHTS_SHA256}\n"
            f"Actual:   {actual_hash}"
        )

    return {
        "path": str(resolved),
        "sha256": actual_hash,
        "size_bytes": resolved.stat().st_size,
    }


def verify_saved_train_arguments(
    args_path: Path,
) -> dict[str, Any]:
    """Verify args.yaml from an earlier official segment."""

    if not args_path.is_file():
        raise FileNotFoundError(args_path)

    saved = yaml.safe_load(
        args_path.read_text(encoding="utf-8")
    )

    if saved.get("task") != "detect":
        raise AssertionError(
            "Source args.yaml task is not detect."
        )

    if saved.get("mode") != "train":
        raise AssertionError(
            "Source args.yaml mode is not train."
        )

    if saved.get("name") != EXPECTED_RUN_NAME:
        raise AssertionError(
            "Source args.yaml belongs to another run name."
        )

    if saved.get("project") != "/kaggle/working":
        raise AssertionError(
            "Source project path is not /kaggle/working."
        )

    if str(saved.get("data", "")) != (
        "/kaggle/working/wrist_privid_v3/"
        "grazped_original_only.yaml"
    ):
        raise AssertionError(
            "Source args.yaml does not reference the frozen "
            "original-only runtime YAML."
        )

    for name, expected in FROZEN_TRAIN_ARGUMENTS.items():
        if name not in saved:
            raise AssertionError(
                f"Source args.yaml is missing {name!r}."
            )

        if not values_equal(saved[name], expected):
            raise AssertionError(
                f"Source args mismatch for {name}: "
                f"expected {expected!r}, found {saved[name]!r}."
            )

    if saved.get("save") is not True:
        raise AssertionError(
            "Source run did not save checkpoints."
        )

    if int(saved.get("save_period", -1)) != 10:
        raise AssertionError(
            "Source save_period is not 10."
        )

    if saved.get("val") is not True:
        raise AssertionError(
            "Source run did not validate each epoch."
        )

    return saved


def verify_unstripped_checkpoint(
    checkpoint_path: Path,
    expected_sha256: str,
    torch_module: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a previous segment's exact-resume state."""

    resolved = checkpoint_path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    if resolved.name != "last.pt":
        raise AssertionError(
            "Exact resume requires weights/last.pt."
        )

    if resolved.parent.name != "weights":
        raise AssertionError(
            "Checkpoint must be inside a weights directory."
        )

    actual_hash = sha256_file(resolved)

    if actual_hash.lower() != expected_sha256.lower():
        raise AssertionError(
            "Checkpoint SHA-256 mismatch.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_hash}"
        )

    checkpoint = torch_module.load(
        resolved,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint payload is not a dictionary."
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
            f"Checkpoint is missing resume keys: "
            f"{sorted(missing)}"
        )

    epoch_index = int(checkpoint["epoch"])
    optimizer = checkpoint["optimizer"]
    scaler = checkpoint["scaler"]
    train_args = checkpoint["train_args"]

    if epoch_index < 1:
        raise AssertionError(
            "Intermediate exact-resume checkpoint must contain "
            "at least two completed epochs."
        )

    if optimizer is None or not isinstance(optimizer, dict):
        raise AssertionError(
            "Checkpoint optimizer state is missing."
        )

    optimizer_entries = len(
        optimizer.get("state", {})
    )

    if optimizer_entries <= 0:
        raise AssertionError(
            "Checkpoint optimizer state is empty."
        )

    if not isinstance(scaler, dict):
        raise AssertionError(
            "Checkpoint AMP scaler state is invalid."
        )

    if not isinstance(train_args, dict):
        raise AssertionError(
            "Checkpoint train_args is invalid."
        )

    if train_args.get("name") != EXPECTED_RUN_NAME:
        raise AssertionError(
            "Checkpoint belongs to another run name."
        )

    for name, expected in FROZEN_TRAIN_ARGUMENTS.items():
        if name not in train_args:
            raise AssertionError(
                f"Checkpoint train_args is missing {name!r}."
            )

        if not values_equal(train_args[name], expected):
            raise AssertionError(
                f"Checkpoint train_args mismatch for {name}: "
                f"expected {expected!r}, "
                f"found {train_args[name]!r}."
            )

    if str(train_args.get("data", "")) != (
        "/kaggle/working/wrist_privid_v3/"
        "grazped_original_only.yaml"
    ):
        raise AssertionError(
            "Checkpoint does not reference the frozen "
            "original-only runtime YAML."
        )

    completed_epochs = epoch_index + 1

    if completed_epochs >= EXPECTED_MAX_EPOCHS:
        raise AssertionError(
            "Checkpoint has already completed the maximum "
            "100 epochs and should not be resumed."
        )

    information = {
        "path": str(resolved),
        "sha256": actual_hash,
        "size_bytes": resolved.stat().st_size,
        "stored_epoch_index_zero_based": epoch_index,
        "completed_epochs": completed_epochs,
        "remaining_to_max_epochs": (
            EXPECTED_MAX_EPOCHS - completed_epochs
        ),
        "optimizer_state_entries": optimizer_entries,
        "scaler_state_entries": len(scaler),
        "best_fitness": checkpoint.get("best_fitness"),
        "updates": checkpoint.get("updates"),
        "version": checkpoint.get("version"),
    }

    return checkpoint, information


# ---------------------------------------------------------------------
# Cross-session staging
# ---------------------------------------------------------------------

def stage_source_run(
    source_run_directory: Path,
    source_checkpoint: Path,
    destination_run_directory: Path,
    expected_checkpoint_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Stage continuity-critical files into writable Kaggle storage."""

    if destination_run_directory.exists():
        raise FileExistsError(
            f"Destination run already exists: "
            f"{destination_run_directory}"
        )

    source_files = {
        "args_yaml": source_run_directory / "args.yaml",
        "results_csv": source_run_directory / "results.csv",
        "last_checkpoint": (
            source_run_directory / "weights" / "last.pt"
        ),
        "best_checkpoint": (
            source_run_directory / "weights" / "best.pt"
        ),
    }

    missing = [
        str(path)
        for path in source_files.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Source segment is missing continuity-critical files:\n"
            + "\n".join(missing)
        )

    if (
        source_checkpoint.resolve()
        != source_files["last_checkpoint"].resolve()
    ):
        raise AssertionError(
            "--checkpoint must point to the source run's "
            "weights/last.pt."
        )

    destination_weights = (
        destination_run_directory / "weights"
    )

    destination_weights.mkdir(
        parents=True,
        exist_ok=False,
    )

    destination_files = {
        "args_yaml": destination_run_directory / "args.yaml",
        "results_csv": destination_run_directory / "results.csv",
        "last_checkpoint": destination_weights / "last.pt",
        "best_checkpoint": destination_weights / "best.pt",
    }

    file_records: dict[str, Any] = {}

    for name, source in source_files.items():
        destination = destination_files[name]

        shutil.copy2(source, destination)

        source_hash = sha256_file(source)
        destination_hash = sha256_file(destination)

        if source_hash != destination_hash:
            raise AssertionError(
                f"Staged file hash mismatch for {name}."
            )

        file_records[name] = {
            "source": str(source),
            "destination": str(destination),
            "sha256": destination_hash,
            "size_bytes": destination.stat().st_size,
        }

    staged_last = destination_files["last_checkpoint"]

    if sha256_file(staged_last).lower() != expected_checkpoint_sha256.lower():
        raise AssertionError(
            "Staged last.pt does not match the expected hash."
        )

    return (
        staged_last,
        {
            "source_run_directory": str(source_run_directory),
            "destination_run_directory": str(
                destination_run_directory
            ),
            "files": file_records,
        },
    )


# ---------------------------------------------------------------------
# Segment stop callback
# ---------------------------------------------------------------------

def make_segment_stop_callback(
    segment_end_epoch: int,
) -> Any:
    """Create a callback that safely stops at one global epoch."""

    def stop_after_saved_epoch(
        trainer: Any,
    ) -> None:
        completed_epoch = int(trainer.epoch) + 1

        if completed_epoch != segment_end_epoch:
            return

        optimizer_entries = len(
            trainer.optimizer.state
        )

        print(
            "In-memory optimizer state entries before "
            "planned segment stop:",
            optimizer_entries,
        )

        if optimizer_entries <= 0:
            raise AssertionError(
                "Optimizer state is empty at the planned "
                "segment boundary."
            )

        last_path = Path(trainer.last)
        csv_path = Path(trainer.csv)

        if not last_path.is_file():
            raise AssertionError(
                "last.pt was not written before the planned stop."
            )

        if not csv_path.is_file():
            raise AssertionError(
                "results.csv was not written before the planned stop."
            )

        results_info = read_results_csv(csv_path)

        if results_info["rows"] != segment_end_epoch:
            raise AssertionError(
                "results.csv does not contain the planned number "
                "of completed epochs before stopping."
            )

        raise PlannedSegmentStop(
            f"Planned session stop after completed epoch "
            f"{segment_end_epoch}."
        )

    return stop_after_saved_epoch


# ---------------------------------------------------------------------
# Post-segment verification
# ---------------------------------------------------------------------

def verify_intermediate_segment(
    run_directory: Path,
    expected_completed_epochs: int,
    torch_module: Any,
) -> dict[str, Any]:
    """Verify an interrupted segment remains exactly resumable."""

    required_files = {
        "args_yaml": run_directory / "args.yaml",
        "results_csv": run_directory / "results.csv",
        "best_checkpoint": (
            run_directory / "weights" / "best.pt"
        ),
        "last_checkpoint": (
            run_directory / "weights" / "last.pt"
        ),
    }

    missing = [
        str(path)
        for path in required_files.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Intermediate segment is missing required files:\n"
            + "\n".join(missing)
        )

    verify_saved_train_arguments(
        required_files["args_yaml"]
    )

    results = read_results_csv(
        required_files["results_csv"]
    )

    if results["rows"] != expected_completed_epochs:
        raise AssertionError(
            "Intermediate results row count does not match "
            "the requested segment boundary."
        )

    checkpoint = torch_module.load(
        required_files["last_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Intermediate last.pt is not a checkpoint dictionary."
        )

    checkpoint_epoch = int(checkpoint.get("epoch", -999))

    if checkpoint_epoch != expected_completed_epochs - 1:
        raise AssertionError(
            "Intermediate checkpoint epoch does not match "
            "the segment boundary."
        )

    optimizer = checkpoint.get("optimizer")

    if optimizer is None or not isinstance(optimizer, dict):
        raise AssertionError(
            "Intermediate optimizer state is missing."
        )

    optimizer_entries = len(
        optimizer.get("state", {})
    )

    if optimizer_entries <= 0:
        raise AssertionError(
            "Intermediate optimizer state is empty."
        )

    scaler = checkpoint.get("scaler")

    if not isinstance(scaler, dict):
        raise AssertionError(
            "Intermediate AMP scaler state is invalid."
        )

    files = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in required_files.items()
    }

    return {
        "status": "intermediate_exact_resume_ready",
        "completed_epochs": expected_completed_epochs,
        "next_epoch": expected_completed_epochs + 1,
        "results": results,
        "checkpoint": {
            "stored_epoch_index_zero_based": checkpoint_epoch,
            "optimizer_state_entries": optimizer_entries,
            "scaler_state_entries": len(scaler),
            "best_fitness": checkpoint.get("best_fitness"),
            "updates": checkpoint.get("updates"),
            "version": checkpoint.get("version"),
        },
        "files": files,
    }


def verify_natural_completion(
    run_directory: Path,
    torch_module: Any,
) -> dict[str, Any]:
    """Verify a naturally completed or early-stopped final run."""

    required_files = {
        "args_yaml": run_directory / "args.yaml",
        "results_csv": run_directory / "results.csv",
        "best_checkpoint": (
            run_directory / "weights" / "best.pt"
        ),
        "last_checkpoint": (
            run_directory / "weights" / "last.pt"
        ),
    }

    missing = [
        str(path)
        for path in required_files.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Completed run is missing required files:\n"
            + "\n".join(missing)
        )

    verify_saved_train_arguments(
        required_files["args_yaml"]
    )

    results = read_results_csv(
        required_files["results_csv"]
    )

    if results["rows"] > EXPECTED_MAX_EPOCHS:
        raise AssertionError(
            "Completed run exceeds the frozen 100-epoch maximum."
        )

    checkpoint = torch_module.load(
        required_files["last_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Final last.pt is not a checkpoint dictionary."
        )

    if checkpoint.get("optimizer") is not None:
        raise AssertionError(
            "Naturally completed last.pt should be stripped."
        )

    if int(checkpoint.get("epoch", -999)) != -1:
        raise AssertionError(
            "Naturally completed last.pt should have epoch=-1."
        )

    files = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in required_files.items()
    }

    return {
        "status": (
            "completed_100_epochs"
            if results["rows"] == EXPECTED_MAX_EPOCHS
            else "completed_by_early_stopping"
        ),
        "completed_epochs": results["rows"],
        "early_stopped": (
            results["rows"] < EXPECTED_MAX_EPOCHS
        ),
        "results": results,
        "final_checkpoint_stripped": True,
        "files": files,
    }


# ---------------------------------------------------------------------
# Main procedure
# ---------------------------------------------------------------------

def main() -> None:
    """Start or resume one controlled official training segment."""

    arguments = parse_arguments()

    if arguments.run_name != EXPECTED_RUN_NAME:
        raise AssertionError(
            f"Run name must remain {EXPECTED_RUN_NAME!r}."
        )

    if not 2 <= arguments.segment_end_epoch <= EXPECTED_MAX_EPOCHS:
        raise AssertionError(
            "--segment-end-epoch must be between 2 and 100."
        )

    is_resume = arguments.checkpoint is not None

    if is_resume and not arguments.expected_checkpoint_sha256:
        raise AssertionError(
            "--expected-checkpoint-sha256 is required "
            "with --checkpoint."
        )

    if (
        not is_resume
        and arguments.expected_checkpoint_sha256
    ):
        raise AssertionError(
            "--expected-checkpoint-sha256 may only be used "
            "with --checkpoint."
        )

    repository_root = Path(__file__).resolve().parents[2]
    ultralytics_root = repository_root / "ultralytics-main"

    config_path = (
        repository_root
        / "project_v3"
        / "configs"
        / "baseline"
        / "baseline_b1_s42.yaml"
    )

    dataset_root = (
        arguments.dataset_root
        .expanduser()
        .resolve()
    )

    output_root = (
        arguments.output_root
        .expanduser()
        .resolve()
    )

    runtime_directory = (
        output_root / "wrist_privid_v3"
    )

    requested_run_directory = (
        output_root / arguments.run_name
    )

    operation = (
        "resume_existing_segment"
        if is_resume
        else "start_fresh_segment"
    )

    print("=" * 96)
    print("WRIST-PRIVID V3 SESSION-SAFE SEGMENT LAUNCHER")
    print(f"EXPERIMENT: {EXPERIMENT_ID}")
    print(f"OPERATION: {operation}")
    print(
        "REQUESTED GLOBAL SEGMENT END EPOCH:",
        arguments.segment_end_epoch,
    )
    print("=" * 96)

    print("\n1. REPOSITORY VERIFICATION")
    print("-" * 96)

    repository_info = verify_repository(
        repository_root,
        arguments.expected_commit,
    )

    print("✓ Repository verification passed")

    print("\n2. CONTROLLED CONFIGURATION")
    print("-" * 96)

    verify_repository_config(config_path)
    config_hash = sha256_file(config_path)

    print("Config:", config_path)
    print("Config SHA-256:", config_hash)
    print("✓ Controlled B1 configuration verified")

    print("\n3. ENVIRONMENT FLAGS")
    print("-" * 96)

    flags = set_and_verify_environment_flags()

    print("✓ All custom features disabled")

    print("\n4. OUTPUT-DIRECTORY SAFETY")
    print("-" * 96)

    if requested_run_directory.exists():
        raise FileExistsError(
            "Writable official run directory already exists:\n"
            f"{requested_run_directory}\n"
            "Use a fresh Kaggle session. Never overwrite an "
            "official segment."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Writable run directory:",
        requested_run_directory,
    )

    print("✓ Writable official run directory is unused")

    print("\n5. RUNTIME DATASET RECONSTRUCTION")
    print("-" * 96)

    dataset_info = rebuild_runtime_dataset(
        repository_root,
        dataset_root,
        runtime_directory,
    )

    for key, value in dataset_info.items():
        print(f"{key}: {value}")

    print("✓ Exact original-only runtime dataset reconstructed")
    print("✓ Official test partition absent")

    print("\n6. ULTRALYTICS AND GPU")
    print("-" * 96)

    if not ultralytics_root.is_dir():
        raise FileNotFoundError(ultralytics_root)

    sys.path.insert(
        0,
        str(ultralytics_root),
    )

    import torch
    import ultralytics
    from ultralytics import YOLO

    print("Ultralytics version:", ultralytics.__version__)
    print(
        "Ultralytics source:",
        Path(ultralytics.__file__).resolve(),
    )

    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise AssertionError(
            f"Expected Ultralytics "
            f"{EXPECTED_ULTRALYTICS_VERSION}, "
            f"found {ultralytics.__version__}."
        )

    if (
        Path(ultralytics.__file__).resolve().parents[1]
        != ultralytics_root.resolve()
    ):
        raise AssertionError(
            "Ultralytics was not imported from the cloned repository."
        )

    if not arguments.preflight_only:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Enable a Kaggle GPU."
            )

        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print(
            "GPU check deferred because --preflight-only is active."
        )

    print("✓ Correct Ultralytics source verified")

    source_state: dict[str, Any]
    source_run_directory: Path | None = None
    source_results_info: dict[str, Any] | None = None
    source_checkpoint_info: dict[str, Any] | None = None
    staging_info: dict[str, Any] | None = None

    print("\n7. SOURCE STATE VERIFICATION")
    print("-" * 96)

    if not is_resume:
        weights_info = verify_official_weights(
            arguments.weights
        )

        print("Weights:", weights_info["path"])
        print("Weights SHA-256:", weights_info["sha256"])

        source_model = YOLO(weights_info["path"])

        architecture_info = verify_standard_architecture(
            source_model,
            expected_parameter_count=(
                EXPECTED_PRETRAINED_PARAMETER_COUNT
            ),
            stage_name=(
                "official pretrained 80-class YOLO11s"
            ),
        )

        source_state = {
            "mode": "fresh",
            "official_weights": weights_info,
            "architecture": architecture_info,
            "completed_epochs_before_segment": 0,
        }

        completed_before = 0

        print("✓ Frozen official YOLO11s initialization verified")

    else:
        checkpoint_path = (
            arguments.checkpoint
            .expanduser()
            .resolve()
        )

        source_run_directory = (
            checkpoint_path.parent.parent
        )

        source_args_path = (
            source_run_directory / "args.yaml"
        )

        source_results_path = (
            source_run_directory / "results.csv"
        )

        source_best_path = (
            source_run_directory
            / "weights"
            / "best.pt"
        )

        for path in (
            source_args_path,
            source_results_path,
            source_best_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        verify_saved_train_arguments(
            source_args_path
        )

        source_results_info = read_results_csv(
            source_results_path
        )

        _, source_checkpoint_info = (
            verify_unstripped_checkpoint(
                checkpoint_path,
                arguments.expected_checkpoint_sha256,
                torch,
            )
        )

        completed_before = int(
            source_checkpoint_info["completed_epochs"]
        )

        if source_results_info["rows"] != completed_before:
            raise AssertionError(
                "Source results.csv rows and checkpoint epoch disagree."
            )

        source_model = YOLO(str(checkpoint_path))

        architecture_info = verify_standard_architecture(
            source_model,
            expected_parameter_count=(
                EXPECTED_NINE_CLASS_PARAMETER_COUNT
            ),
            stage_name=(
                "resumable nine-class YOLO11s checkpoint"
            ),
        )

        source_state = {
            "mode": "resume",
            "source_run_directory": str(
                source_run_directory
            ),
            "source_args": {
                "path": str(source_args_path),
                "sha256": sha256_file(source_args_path),
            },
            "source_results": source_results_info,
            "source_best": {
                "path": str(source_best_path),
                "sha256": sha256_file(source_best_path),
                "size_bytes": source_best_path.stat().st_size,
            },
            "source_checkpoint": source_checkpoint_info,
            "architecture": architecture_info,
            "completed_epochs_before_segment": (
                completed_before
            ),
        }

        print("Source completed epochs:", completed_before)
        print(
            "Source checkpoint SHA-256:",
            source_checkpoint_info["sha256"],
        )
        print(
            "Source optimizer state entries:",
            source_checkpoint_info[
                "optimizer_state_entries"
            ],
        )
        print("✓ Exact-resume source state verified")

    if arguments.segment_end_epoch <= completed_before:
        raise AssertionError(
            "--segment-end-epoch must be greater than the "
            "number of epochs already completed.\n"
            f"Already completed: {completed_before}\n"
            f"Requested end:    {arguments.segment_end_epoch}"
        )

    print("\n8. SEGMENT PLAN")
    print("-" * 96)

    epochs_in_requested_segment = (
        arguments.segment_end_epoch
        - completed_before
    )

    print(
        "Completed before this segment:",
        completed_before,
    )
    print(
        "Requested global end epoch:",
        arguments.segment_end_epoch,
    )
    print(
        "Epochs scheduled in this session:",
        epochs_in_requested_segment,
    )
    print(
        "Natural completion requested:",
        arguments.segment_end_epoch
        == EXPECTED_MAX_EPOCHS,
    )

    plan = {
        "experiment_id": EXPERIMENT_ID,
        "operation": operation,
        "test_partition_used": False,
        "repository": repository_info,
        "config": {
            "path": str(config_path),
            "sha256": config_hash,
        },
        "flags": flags,
        "dataset": {
            key: (
                str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in dataset_info.items()
        },
        "source_state": source_state,
        "segment": {
            "completed_epochs_before_segment": completed_before,
            "requested_global_end_epoch": (
                arguments.segment_end_epoch
            ),
            "epochs_scheduled_in_session": (
                epochs_in_requested_segment
            ),
            "natural_completion_requested": (
                arguments.segment_end_epoch
                == EXPECTED_MAX_EPOCHS
            ),
        },
        "requested_output_directory": str(
            requested_run_directory
        ),
        "operational_overrides": {
            "device": arguments.device,
            "workers": arguments.workers,
        },
    }

    plan_path = (
        runtime_directory
        / "v3_b1_s42_segment_preflight.json"
    )

    write_json(plan_path, plan)

    print("Segment preflight:", plan_path)
    print(
        "Segment preflight SHA-256:",
        sha256_file(plan_path),
    )

    print("\n" + "=" * 96)
    print("ALL SEGMENT PREFLIGHT CHECKS PASSED")
    print("=" * 96)
    print("Training images: 14,204 original images")
    print("Offline augmented images: 0")
    print("Validation images: 4,094")
    print("Image size: 1024")
    print("Frozen maximum epochs: 100")
    print("Seed: 42")
    print("Official test split used: No")
    print("=" * 96)

    if arguments.preflight_only:
        print()
        print("=" * 96)
        print("SEGMENT PREFLIGHT-ONLY MODE COMPLETED")
        print("=" * 96)
        print("✓ Source state verified")
        print("✓ Segment boundary verified")
        print("✓ GPU training was not started")
        print("✓ Official test partition was not accessed")
        print("=" * 96)
        return

    print("\n9. PREPARING WRITABLE OFFICIAL RUN")
    print("-" * 96)

    if not is_resume:
        training_model = source_model

        print(
            "Fresh official run will be created at:",
            requested_run_directory,
        )

    else:
        staged_checkpoint, staging_info = (
            stage_source_run(
                source_run_directory=source_run_directory,
                source_checkpoint=checkpoint_path,
                destination_run_directory=(
                    requested_run_directory
                ),
                expected_checkpoint_sha256=(
                    arguments.expected_checkpoint_sha256
                ),
            )
        )

        print("Staged checkpoint:", staged_checkpoint)
        print(
            "Staged checkpoint SHA-256:",
            sha256_file(staged_checkpoint),
        )
        print("✓ Previous results.csv history preserved")
        print("✓ Previous best.pt preserved")
        print("✓ Writable unstripped last.pt staged")

        training_model = YOLO(
            str(staged_checkpoint)
        )

        staged_architecture = verify_standard_architecture(
            training_model,
            expected_parameter_count=(
                EXPECTED_NINE_CLASS_PARAMETER_COUNT
            ),
            stage_name=(
                "staged resumable nine-class YOLO11s"
            ),
        )

        if (
            staged_architecture["parameter_count"]
            != architecture_info["parameter_count"]
        ):
            raise AssertionError(
                "Staged checkpoint architecture differs "
                "from the verified source."
            )

    print("\n10. REGISTERING SESSION-SAFE STOP")
    print("-" * 96)

    if arguments.segment_end_epoch < EXPECTED_MAX_EPOCHS:
        stop_callback = make_segment_stop_callback(
            arguments.segment_end_epoch
        )

        training_model.add_callback(
            "on_model_save",
            stop_callback,
        )

        print(
            "Planned safe stop after completed epoch:",
            arguments.segment_end_epoch,
        )
    else:
        print(
            "No intentional stop callback registered; "
            "the run may complete naturally or by patience."
        )

    resume_observation: dict[str, Any] = {}

    if is_resume:
        def observe_resume_state(
            trainer: Any,
        ) -> None:
            """Capture restored state before resumed epochs begin."""

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
                "results_csv_present_on_train_start"
            ] = Path(trainer.csv).is_file()

        training_model.add_callback(
            "on_train_start",
            observe_resume_state,
        )

    print("\n11. STARTING OFFICIAL TRAINING SEGMENT")
    print("-" * 96)

    planned_stop_observed = False

    try:
        if not is_resume:
            training_model.train(
                data=str(dataset_info["data_yaml"]),
                model=source_state[
                    "official_weights"
                ]["path"],
                project=str(output_root),
                name=arguments.run_name,
                exist_ok=False,
                save=True,
                save_period=10,
                plots=True,
                val=True,
                verbose=True,
                **FROZEN_TRAIN_ARGUMENTS,
            )
        else:
            training_model.train(
                resume=True,
                device=arguments.device,
                workers=arguments.workers,
                plots=True,
            )

    except PlannedSegmentStop as error:
        planned_stop_observed = True
        print("Expected planned segment stop:", error)

    trainer = training_model.trainer

    if trainer is None:
        raise RuntimeError(
            "Ultralytics trainer was not retained."
        )

    actual_run_directory = Path(
        trainer.save_dir
    ).resolve()

    if actual_run_directory != requested_run_directory.resolve():
        raise AssertionError(
            "Unexpected official run directory.\n"
            f"Expected: {requested_run_directory}\n"
            f"Actual:   {actual_run_directory}"
        )

    if is_resume:
        print("Resume observation:", resume_observation)

        if (
            resume_observation.get(
                "start_epoch_zero_based"
            )
            != completed_before
        ):
            raise AssertionError(
                "Resume did not begin at the correct next epoch."
            )

        if int(
            resume_observation.get(
                "optimizer_state_entries",
                0,
            )
        ) <= 0:
            raise AssertionError(
                "Optimizer state was not restored at resume start."
            )

        if not resume_observation.get(
            "results_csv_present_on_train_start",
            False,
        ):
            raise AssertionError(
                "Previous results.csv was absent at resume start."
            )

        print(
            "✓ Resume began from global epoch:",
            completed_before + 1,
        )
        print("✓ Optimizer state restored")
        print("✓ Previous epoch history present")

    print("\n12. POST-SEGMENT VERIFICATION")
    print("-" * 96)

    if planned_stop_observed:
        if arguments.segment_end_epoch == EXPECTED_MAX_EPOCHS:
            raise AssertionError(
                "A planned stop must not occur at epoch 100."
            )

        segment_result = verify_intermediate_segment(
            actual_run_directory,
            expected_completed_epochs=(
                arguments.segment_end_epoch
            ),
            torch_module=torch,
        )

    else:
        if arguments.segment_end_epoch < EXPECTED_MAX_EPOCHS:
            raise AssertionError(
                "Training returned without reaching the planned "
                "intermediate segment stop."
            )

        segment_result = verify_natural_completion(
            actual_run_directory,
            torch_module=torch,
        )

    segment_record = {
        "experiment_id": EXPERIMENT_ID,
        "operation": operation,
        "test_partition_used": False,
        "repository": repository_info,
        "config_sha256": config_hash,
        "source_state": source_state,
        "staging": staging_info,
        "resume_observation": resume_observation,
        "segment_plan": plan["segment"],
        "segment_result": segment_result,
        "run_directory": str(actual_run_directory),
    }

    segment_record_path = (
        actual_run_directory
        / (
            "v3_b1_s42_segment_"
            f"{segment_result['completed_epochs']:03d}"
            "_manifest.json"
        )
    )

    write_json(
        segment_record_path,
        segment_record,
    )

    manifest_hash = sha256_file(
        segment_record_path
    )

    print("Segment manifest:", segment_record_path)
    print("Segment manifest SHA-256:", manifest_hash)

    print("\n" + "=" * 96)

    if segment_result["status"] == (
        "intermediate_exact_resume_ready"
    ):
        print("V3-B1-S42 INTERMEDIATE SEGMENT COMPLETED SAFELY")
        print("=" * 96)
        print(
            "Completed global epochs:",
            segment_result["completed_epochs"],
        )
        print(
            "Next global epoch:",
            segment_result["next_epoch"],
        )
        print(
            "Unstripped last.pt SHA-256:",
            segment_result[
                "files"
            ]["last_checkpoint"]["sha256"],
        )
        print(
            "Best.pt SHA-256:",
            segment_result[
                "files"
            ]["best_checkpoint"]["sha256"],
        )
        print("✓ Optimizer state preserved")
        print("✓ Scheduler and epoch state preserved")
        print("✓ EMA and AMP scaler state preserved")
        print("✓ Full results.csv history preserved")
        print("✓ Run directory is ready to save as a Kaggle model")
        print("✓ Official test partition was not accessed")
    else:
        print("V3-B1-S42 OFFICIAL TRAINING FINISHED")
        print("=" * 96)
        print(
            "Completion status:",
            segment_result["status"],
        )
        print(
            "Completed epochs:",
            segment_result["completed_epochs"],
        )
        print(
            "Best.pt SHA-256:",
            segment_result[
                "files"
            ]["best_checkpoint"]["sha256"],
        )
        print(
            "Last.pt SHA-256:",
            segment_result[
                "files"
            ]["last_checkpoint"]["sha256"],
        )
        print("✓ Final checkpoints are stripped")
        print("✓ Full results.csv history verified")
        print("✓ Official test partition was not accessed")

    print("=" * 96)


if __name__ == "__main__":
    main()
