"""
Train Wrist-PriViD experiment V3-B1-S42.

Controlled comparison against V3-B0-S42:

    B0: 28,408 offline-augmented training images
    B1: 14,204 original training images

All other verified V2 baseline settings are preserved.

This launcher performs strict pre-training checks and refuses to
start when the repository, dataset, flags, model, or configuration
does not match the frozen experiment protocol.

The official test partition is never accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

EXPECTED_TRAIN_IMAGES = 14204
EXPECTED_VALID_IMAGES = 4094

CLASS_NAMES = [
    "boneanomaly",
    "bonelesion",
    "foreignbody",
    "fracture",
    "metal",
    "periostealreaction",
    "pronatorsign",
    "softtissue",
    "text",
]

REQUIRED_DISABLED_FLAGS = {
    "USE_FOCALER": "0",
    "USE_WHFE": "0",
    "USE_CBLOSS": "0",
    "USE_MCAUX": "0",
}


# ---------------------------------------------------------------------
# Frozen V2 training arguments
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
# Command-line arguments
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse launcher arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Strict launcher for Wrist-PriViD "
            "experiment V3-B1-S42."
        )
    )

    parser.add_argument(
        "--expected-commit",
        required=True,
        help=(
            "Full Git commit hash, or a unique commit prefix, "
            "that must be checked out before training."
        ),
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help=(
            "Path to the frozen official yolo11s.pt "
            "pretrained checkpoint."
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/kaggle/input/datasets/nithoukhan/"
            "grazpedwri-dx-aug/GRAZPEDWRI-DX"
        ),
        help="Path to the GRAZPEDWRI-DX dataset root.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working"),
        help="Root directory for the Ultralytics run.",
    )

    parser.add_argument(
        "--run-name",
        default="wrist_privid_v3_b1_s42",
        help="Frozen experiment output directory name.",
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run all repository, dataset, configuration, "
            "weights, and architecture checks, then stop "
            "before GPU training."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of one file."""

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
    """Run one Git command and return stripped stdout."""

    return subprocess.check_output(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
    ).strip()


def values_equal(
    actual: Any,
    expected: Any,
) -> bool:
    """Compare configuration values safely."""

    if isinstance(expected, float):
        try:
            return abs(
                float(actual) - expected
            ) < 1e-12
        except (TypeError, ValueError):
            return False

    return actual == expected


# ---------------------------------------------------------------------
# Repository verification
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
            f"Expected commit beginning with "
            f"{expected_commit!r}, found {commit!r}."
        )

    if status:
        raise AssertionError(
            "Repository contains uncommitted changes."
        )

    return {
        "branch": branch,
        "commit": commit,
    }


# ---------------------------------------------------------------------
# Experiment configuration verification
# ---------------------------------------------------------------------

def verify_experiment_config(
    config_path: Path,
) -> dict[str, Any]:
    """Verify the repository YAML against the frozen protocol."""

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if config["experiment"]["id"] != EXPERIMENT_ID:
        raise AssertionError(
            "Incorrect experiment ID in configuration."
        )

    if (
        config["experiment"]["branch"]
        != EXPECTED_BRANCH
    ):
        raise AssertionError(
            "Incorrect required branch in configuration."
        )

    if config["experiment"]["test_partition_allowed"]:
        raise AssertionError(
            "Configuration incorrectly allows test access."
        )

    if (
        config["data"]["expected_train_images"]
        != EXPECTED_TRAIN_IMAGES
    ):
        raise AssertionError(
            "Incorrect expected training-image count."
        )

    if (
        config["data"]["expected_validation_images"]
        != EXPECTED_VALID_IMAGES
    ):
        raise AssertionError(
            "Incorrect expected validation-image count."
        )

    if config["data"]["train_allowed_prefix"] != "orig_":
        raise AssertionError(
            "The allowed training prefix must be orig_."
        )

    if config["data"]["train_forbidden_prefix"] != "aug_":
        raise AssertionError(
            "The forbidden training prefix must be aug_."
        )

    config_checks = {
        "image_size": (
            config["training"]["image_size"],
            FROZEN_TRAIN_ARGUMENTS["imgsz"],
        ),
        "epochs": (
            config["training"]["epochs"],
            FROZEN_TRAIN_ARGUMENTS["epochs"],
        ),
        "patience": (
            config["training"]["patience"],
            FROZEN_TRAIN_ARGUMENTS["patience"],
        ),
        "batch_size": (
            config["training"]["batch_size"],
            FROZEN_TRAIN_ARGUMENTS["batch"],
        ),
        "workers": (
            config["training"]["workers"],
            FROZEN_TRAIN_ARGUMENTS["workers"],
        ),
        "initial_learning_rate": (
            config["training"]["initial_learning_rate"],
            FROZEN_TRAIN_ARGUMENTS["lr0"],
        ),
        "final_learning_rate_fraction": (
            config["training"][
                "final_learning_rate_fraction"
            ],
            FROZEN_TRAIN_ARGUMENTS["lrf"],
        ),
        "momentum": (
            config["training"]["momentum"],
            FROZEN_TRAIN_ARGUMENTS["momentum"],
        ),
        "weight_decay": (
            config["training"]["weight_decay"],
            FROZEN_TRAIN_ARGUMENTS["weight_decay"],
        ),
        "warmup_epochs": (
            config["training"]["warmup_epochs"],
            FROZEN_TRAIN_ARGUMENTS["warmup_epochs"],
        ),
        "box_gain": (
            config["loss"]["box_gain"],
            FROZEN_TRAIN_ARGUMENTS["box"],
        ),
        "classification_gain": (
            config["loss"]["classification_gain"],
            FROZEN_TRAIN_ARGUMENTS["cls"],
        ),
        "dfl_gain": (
            config["loss"]["dfl_gain"],
            FROZEN_TRAIN_ARGUMENTS["dfl"],
        ),
        "classification_positive_weight": (
            config["loss"][
                "classification_positive_weight"
            ],
            FROZEN_TRAIN_ARGUMENTS["cls_pw"],
        ),
        "mosaic": (
            config["augmentation"]["mosaic"],
            FROZEN_TRAIN_ARGUMENTS["mosaic"],
        ),
        "close_mosaic_epochs": (
            config["augmentation"][
                "close_mosaic_epochs"
            ],
            FROZEN_TRAIN_ARGUMENTS["close_mosaic"],
        ),
        "seed": (
            config["training"]["seed"],
            FROZEN_TRAIN_ARGUMENTS["seed"],
        ),
    }

    for field_name, (
        actual,
        expected,
    ) in config_checks.items():
        if not values_equal(actual, expected):
            raise AssertionError(
                f"Configuration mismatch for {field_name}: "
                f"expected {expected!r}, found {actual!r}."
            )

    return config


# ---------------------------------------------------------------------
# Environment flag verification
# ---------------------------------------------------------------------

def set_and_verify_environment_flags() -> None:
    """Disable every V2 custom module and loss flag."""

    for flag_name, required_value in (
        REQUIRED_DISABLED_FLAGS.items()
    ):
        os.environ[flag_name] = required_value

    for flag_name, required_value in (
        REQUIRED_DISABLED_FLAGS.items()
    ):
        actual_value = os.environ.get(flag_name)

        print(
            f"{flag_name}: {actual_value}"
        )

        if actual_value != required_value:
            raise AssertionError(
                f"{flag_name} must equal "
                f"{required_value!r}."
            )


# ---------------------------------------------------------------------
# Dataset construction and verification
# ---------------------------------------------------------------------

def build_runtime_dataset(
    repository_root: Path,
    dataset_root: Path,
    runtime_directory: Path,
) -> dict[str, Any]:
    """Build and independently verify the original-only list."""

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

    train_list_path = (
        runtime_directory / "train_original.txt"
    )

    data_yaml_path = (
        runtime_directory
        / "grazped_original_only.yaml"
    )

    dataset_summary_path = (
        runtime_directory
        / "original_only_dataset_summary.json"
    )

    for required_path in [
        train_list_path,
        data_yaml_path,
        dataset_summary_path,
    ]:
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    train_paths = [
        Path(line.strip())
        for line in train_list_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    unique_paths = {
        str(path.resolve())
        for path in train_paths
    }

    original_count = sum(
        path.name.startswith("orig_")
        for path in train_paths
    )

    augmented_count = sum(
        path.name.startswith("aug_")
        for path in train_paths
    )

    missing_count = sum(
        not path.is_file()
        for path in train_paths
    )

    if len(train_paths) != EXPECTED_TRAIN_IMAGES:
        raise AssertionError(
            "Training-list count is not 14,204."
        )

    if len(unique_paths) != EXPECTED_TRAIN_IMAGES:
        raise AssertionError(
            "Training list contains duplicate paths."
        )

    if original_count != EXPECTED_TRAIN_IMAGES:
        raise AssertionError(
            "Not every training path uses orig_."
        )

    if augmented_count != 0:
        raise AssertionError(
            "An aug_ image entered the training list."
        )

    if missing_count != 0:
        raise AssertionError(
            "The training list contains missing files."
        )

    data_yaml_text = data_yaml_path.read_text(
        encoding="utf-8"
    )

    yaml_data = yaml.safe_load(data_yaml_text)

    if "test" in yaml_data:
        raise AssertionError(
            "Runtime data YAML must not contain a test split."
        )

    validation_directory = Path(
        yaml_data["val"]
    )

    validation_images = sorted(
        validation_directory.glob("*.png")
    )

    if len(validation_images) != EXPECTED_VALID_IMAGES:
        raise AssertionError(
            "Validation-image count is not 4,094."
        )

    return {
        "train_list_path": train_list_path,
        "data_yaml_path": data_yaml_path,
        "dataset_summary_path": (
            dataset_summary_path
        ),
        "training_image_count": len(train_paths),
        "unique_training_paths": len(unique_paths),
        "original_training_paths": original_count,
        "augmented_training_paths": augmented_count,
        "missing_training_files": missing_count,
        "validation_image_count": len(
            validation_images
        ),
        "train_list_sha256": sha256_file(
            train_list_path
        ),
        "data_yaml_sha256": sha256_file(
            data_yaml_path
        ),
    }


# ---------------------------------------------------------------------
# Pretrained model verification
# ---------------------------------------------------------------------

def verify_pretrained_weights(
    weights_path: Path,
) -> dict[str, str]:
    """Verify the frozen YOLO11s initialization file."""

    resolved = weights_path.expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    if resolved.name != "yolo11s.pt":
        raise AssertionError(
            "The pretrained file must be named yolo11s.pt."
        )

    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


# ---------------------------------------------------------------------
# Model architecture verification
# ---------------------------------------------------------------------

def verify_model_architecture(
    model: Any,
) -> dict[str, Any]:
    """Ensure no V2 custom architecture is active."""

    module_class_names = [
        module.__class__.__name__
        for module in model.model.modules()
    ]

    forbidden_names = {
        "WHFE",
        "MCAux",
        "MCAuxHead",
    }

    present_forbidden = sorted(
        forbidden_names
        & set(module_class_names)
    )

    if present_forbidden:
        raise AssertionError(
            "Forbidden custom modules are active: "
            f"{present_forbidden}"
        )

    mcaux_attributes = [
        name
        for name, _ in model.model.named_modules()
        if "_mcaux" in name.lower()
    ]

    if mcaux_attributes:
        raise AssertionError(
            "MCAux attributes were found in the model."
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.model.parameters()
        if parameter.requires_grad
    )

    return {
        "parameter_count": parameter_count,
        "trainable_parameter_count": (
            trainable_parameter_count
        ),
        "module_count": len(module_class_names),
        "forbidden_modules_present": (
            present_forbidden
        ),
    }


# ---------------------------------------------------------------------
# Main training procedure
# ---------------------------------------------------------------------

def main() -> None:
    """Verify and launch the controlled B1 experiment."""

    arguments = parse_arguments()

    repository_root = (
        Path(__file__).resolve().parents[2]
    )

    config_path = (
        repository_root
        / "project_v3"
        / "configs"
        / "baseline"
        / "baseline_b1_s42.yaml"
    )

    ultralytics_source_root = (
        repository_root / "ultralytics-main"
    )

    dataset_root = (
        arguments.dataset_root.expanduser().resolve()
    )

    output_root = (
        arguments.output_root.expanduser().resolve()
    )

    runtime_directory = (
        output_root / "wrist_privid_v3"
    )

    requested_run_directory = (
        output_root / arguments.run_name
    )

    print("=" * 92)
    print("WRIST-PRIVID V3 CONTROLLED BASELINE")
    print("EXPERIMENT: V3-B1-S42")
    print("=" * 92)

    print("\n1. REPOSITORY VERIFICATION")
    print("-" * 92)

    repository_info = verify_repository(
        repository_root,
        arguments.expected_commit,
    )

    print("✓ Repository verification passed")

    print("\n2. CONFIGURATION VERIFICATION")
    print("-" * 92)

    experiment_config = (
        verify_experiment_config(
            config_path
        )
    )

    print("Config:", config_path)
    print("Config SHA-256:", sha256_file(config_path))
    print("✓ Controlled B1 configuration verified")

    print("\n3. ENVIRONMENT FLAGS")
    print("-" * 92)

    set_and_verify_environment_flags()

    print("✓ All custom features disabled")

    print("\n4. OUTPUT-DIRECTORY SAFETY")
    print("-" * 92)

    if requested_run_directory.exists():
        raise FileExistsError(
            "The frozen output directory already exists: "
            f"{requested_run_directory}\n"
            "Delete it only when this is definitely a failed "
            "pre-training attempt. Never overwrite a real run."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Requested run directory:",
        requested_run_directory,
    )

    print("✓ Output directory is unused")

    print("\n5. ORIGINAL-ONLY DATASET BUILD")
    print("-" * 92)

    dataset_info = build_runtime_dataset(
        repository_root,
        dataset_root,
        runtime_directory,
    )

    for key, value in dataset_info.items():
        print(f"{key}: {value}")

    print("✓ Original-only dataset verification passed")

    print("\n6. PRETRAINED WEIGHTS")
    print("-" * 92)

    weights_info = verify_pretrained_weights(
        arguments.weights
    )

    print("Weights:", weights_info["path"])
    print("Weights SHA-256:", weights_info["sha256"])
    print("✓ Frozen YOLO11s initialization verified")

    print("\n7. ULTRALYTICS IMPORT")
    print("-" * 92)

    if not ultralytics_source_root.is_dir():
        raise FileNotFoundError(
            ultralytics_source_root
        )

    sys.path.insert(
        0,
        str(ultralytics_source_root),
    )

    import ultralytics
    from ultralytics import YOLO

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
            "Expected Ultralytics "
            f"{EXPECTED_ULTRALYTICS_VERSION}, "
            f"found {ultralytics.__version__}."
        )

    expected_source = (
        ultralytics_source_root.resolve()
    )

    actual_source = (
        Path(ultralytics.__file__)
        .resolve()
        .parents[1]
    )

    if actual_source != expected_source:
        raise AssertionError(
            "Ultralytics was not imported from the "
            "cloned repository."
        )

    print("✓ Correct Ultralytics source verified")

    print("\n8. MODEL INITIALIZATION")
    print("-" * 92)

    model = YOLO(weights_info["path"])

    architecture_info = (
        verify_model_architecture(model)
    )

    for key, value in architecture_info.items():
        print(f"{key}: {value}")

    print("✓ Standard YOLO11s architecture verified")

    print("\n9. PRE-TRAINING RECORD")
    print("-" * 92)

    preflight_record = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "strong_single_view_baseline",
        "test_partition_used": False,
        "repository": repository_info,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "environment_flags": {
            name: os.environ[name]
            for name in REQUIRED_DISABLED_FLAGS
        },
        "dataset": {
            key: (
                str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in dataset_info.items()
        },
        "pretrained_weights": weights_info,
        "architecture": architecture_info,
        "frozen_training_arguments": (
            FROZEN_TRAIN_ARGUMENTS
        ),
        "requested_output_directory": str(
            requested_run_directory
        ),
    }

    preflight_path = (
        runtime_directory
        / "v3_b1_s42_preflight.json"
    )

    preflight_path.write_text(
        json.dumps(
            preflight_record,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Preflight record:", preflight_path)
    print("Preflight SHA-256:", sha256_file(preflight_path))

    print("\n" + "=" * 92)
    print("ALL PRE-TRAINING CHECKS PASSED")
    print("=" * 92)

    print("Experiment:", EXPERIMENT_ID)
    print("Training images: 14,204 original images")
    print("Offline augmented images: 0")
    print("Validation images: 4,094")
    print("Image size: 1024")
    print("Epochs: 100")
    print("Seed: 42")
    print("Test split used: No")
    print("=" * 92)

    if arguments.preflight_only:
        print()
        print("=" * 92)
        print("PREFLIGHT-ONLY MODE COMPLETED")
        print("=" * 92)
        print("✓ All verification checks passed")
        print("✓ GPU training was not started")
        print("✓ Official test partition was not accessed")
        print("=" * 92)
        return

    print("\n10. STARTING GPU TRAINING")
    print("-" * 92)

    model.train(
        data=str(
            dataset_info["data_yaml_path"]
        ),
        model=weights_info["path"],
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

    trainer = model.trainer

    if trainer is None:
        raise RuntimeError(
            "Ultralytics trainer was not retained."
        )

    actual_run_directory = Path(
        trainer.save_dir
    ).resolve()

    if actual_run_directory != (
        requested_run_directory.resolve()
    ):
        raise AssertionError(
            "Unexpected output directory:\n"
            f"Expected: {requested_run_directory}\n"
            f"Actual:   {actual_run_directory}"
        )

    required_outputs = {
        "args_yaml": (
            actual_run_directory / "args.yaml"
        ),
        "results_csv": (
            actual_run_directory / "results.csv"
        ),
        "best_checkpoint": (
            actual_run_directory
            / "weights"
            / "best.pt"
        ),
        "last_checkpoint": (
            actual_run_directory
            / "weights"
            / "last.pt"
        ),
    }

    missing_outputs = [
        str(path)
        for path in required_outputs.values()
        if not path.is_file()
    ]

    if missing_outputs:
        raise FileNotFoundError(
            "Required training outputs are missing:\n"
            + "\n".join(missing_outputs)
        )

    postflight_record = {
        "experiment_id": EXPERIMENT_ID,
        "test_partition_used": False,
        "repository": repository_info,
        "run_directory": str(
            actual_run_directory
        ),
        "output_files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in required_outputs.items()
        },
    }

    postflight_path = (
        actual_run_directory
        / "v3_b1_s42_postflight.json"
    )

    postflight_path.write_text(
        json.dumps(
            postflight_record,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print("V3-B1-S42 TRAINING COMPLETED")
    print("=" * 92)

    print("Run directory:", actual_run_directory)
    print(
        "Best checkpoint SHA-256:",
        postflight_record[
            "output_files"
        ]["best_checkpoint"]["sha256"],
    )
    print(
        "Last checkpoint SHA-256:",
        postflight_record[
            "output_files"
        ]["last_checkpoint"]["sha256"],
    )

    print("✓ Required training artifacts verified")
    print("✓ Test partition was not accessed")
    print("✓ Run is ready for validation verification")
    print("=" * 92)


if __name__ == "__main__":
    main()