"""
Strict cross-session resume launcher for Wrist-PriViD experiment V3-B1-S42.

It resumes only from an unstripped weights/last.pt and verifies:
- branch and immutable Git commit
- controlled B1 configuration
- all custom FWNet flags disabled
- canonical original-only train/validation runtime files
- previous args.yaml and results.csv
- last.pt SHA-256, epoch, optimizer state, and train_args
- standard 9-class YOLO11s architecture
- no official test split access

Ultralytics resume=True restores the stored training state.
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


EXPERIMENT_ID = "V3-B1-S42"
EXPECTED_BRANCH = "v3-strong-baseline"
EXPECTED_ULTRALYTICS_VERSION = "8.4.50"
EXPECTED_RUN_NAME = "wrist_privid_v3_b1_s42"
EXPECTED_TRAIN_IMAGES = 14204
EXPECTED_VALID_IMAGES = 4094
EXPECTED_TOTAL_EPOCHS = 100
EXPECTED_PARAMETER_COUNT = 9431275

EXPECTED_TRAIN_LIST_SHA256 = (
    "2577b30d84e9273386e94ab3737fbf65a4b3f59031d5b604a96eb0e34cc055cc"
)
EXPECTED_DATA_YAML_SHA256 = (
    "8ab985997ee88455ea9861cdf915fcfa6a2c6b0dbb3a731045156250d40fd51d"
)

REQUIRED_DISABLED_FLAGS = {
    "USE_FOCALER": "0",
    "USE_WHFE": "0",
    "USE_CBLOSS": "0",
    "USE_MCAUX": "0",
}

FROZEN_ARGUMENTS: dict[str, Any] = {
    "imgsz": 1024,
    "epochs": 100,
    "patience": 50,
    "batch": 16,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "box": 7.5,
    "cls": 2.5,
    "dfl": 1.5,
    "cls_pw": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "close_mosaic": 15,
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
    "amp": True,
    "cos_lr": True,
    "seed": 42,
    "deterministic": True,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict resume launcher for V3-B1-S42."
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Previous session's unstripped weights/last.pt.",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help="Expected SHA-256 of last.pt.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/kaggle/input/datasets/nithoukhan/"
            "grazpedwri-dx-aug/GRAZPEDWRI-DX"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working"),
    )
    parser.add_argument(
        "--run-name",
        default=EXPECTED_RUN_NAME,
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify exact resume state, then stop before training.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_git(root: Path, args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected or bool(actual) is expected
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verify_repository(root: Path, expected_commit: str) -> dict[str, str]:
    branch = run_git(root, ["branch", "--show-current"])
    commit = run_git(root, ["rev-parse", "HEAD"])
    status = run_git(root, ["status", "--porcelain"])

    print("Branch:", branch)
    print("Commit:", commit)
    print("Working tree clean:", not bool(status))

    if branch != EXPECTED_BRANCH:
        raise AssertionError(
            f"Expected branch {EXPECTED_BRANCH!r}, found {branch!r}."
        )
    if not commit.startswith(expected_commit):
        raise AssertionError(
            f"Expected commit prefix {expected_commit!r}, found {commit!r}."
        )
    if status:
        raise AssertionError("Repository has uncommitted changes.")

    return {"branch": branch, "commit": commit}


def set_and_verify_flags() -> dict[str, str]:
    for name, value in REQUIRED_DISABLED_FLAGS.items():
        os.environ[name] = value

    verified: dict[str, str] = {}
    for name, expected in REQUIRED_DISABLED_FLAGS.items():
        actual = os.environ.get(name)
        print(f"{name}: {actual}")
        if actual != expected:
            raise AssertionError(
                f"{name} must be {expected!r}, found {actual!r}."
            )
        verified[name] = actual
    return verified


def verify_repository_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    if config["experiment"]["id"] != EXPERIMENT_ID:
        raise AssertionError("Wrong experiment ID in config.")
    if config["experiment"]["branch"] != EXPECTED_BRANCH:
        raise AssertionError("Wrong branch in config.")
    if config["experiment"]["test_partition_allowed"]:
        raise AssertionError("Config permits test access.")
    if config["data"]["expected_train_images"] != EXPECTED_TRAIN_IMAGES:
        raise AssertionError("Wrong train count in config.")
    if config["data"]["expected_validation_images"] != EXPECTED_VALID_IMAGES:
        raise AssertionError("Wrong validation count in config.")
    if config["data"]["train_allowed_prefix"] != "orig_":
        raise AssertionError("Allowed train prefix must be orig_.")
    if config["data"]["train_forbidden_prefix"] != "aug_":
        raise AssertionError("Forbidden train prefix must be aug_.")

    actual_values = {
        "imgsz": config["training"]["image_size"],
        "epochs": config["training"]["epochs"],
        "patience": config["training"]["patience"],
        "batch": config["training"]["batch_size"],
        "optimizer": config["training"]["optimizer"],
        "lr0": config["training"]["initial_learning_rate"],
        "lrf": config["training"]["final_learning_rate_fraction"],
        "momentum": config["training"]["momentum"],
        "weight_decay": config["training"]["weight_decay"],
        "warmup_epochs": config["training"]["warmup_epochs"],
        "box": config["loss"]["box_gain"],
        "cls": config["loss"]["classification_gain"],
        "dfl": config["loss"]["dfl_gain"],
        "cls_pw": config["loss"]["classification_positive_weight"],
        "mosaic": config["augmentation"]["mosaic"],
        "mixup": config["augmentation"]["mixup"],
        "close_mosaic": config["augmentation"]["close_mosaic_epochs"],
        "hsv_h": config["augmentation"]["hsv_h"],
        "hsv_s": config["augmentation"]["hsv_s"],
        "hsv_v": config["augmentation"]["hsv_v"],
        "degrees": config["augmentation"]["degrees"],
        "translate": config["augmentation"]["translate"],
        "scale": config["augmentation"]["scale"],
        "shear": config["augmentation"]["shear"],
        "perspective": config["augmentation"]["perspective"],
        "flipud": config["augmentation"]["flip_vertical"],
        "fliplr": config["augmentation"]["flip_horizontal"],
        "amp": config["training"]["amp"],
        "cos_lr": config["training"]["cosine_learning_rate"],
        "seed": config["training"]["seed"],
        "deterministic": config["training"]["deterministic"],
    }

    for name, expected in FROZEN_ARGUMENTS.items():
        actual = actual_values[name]
        if not values_equal(actual, expected):
            raise AssertionError(
                f"Config mismatch for {name}: "
                f"expected {expected!r}, found {actual!r}."
            )

    return config


def rebuild_runtime_dataset(
    repository_root: Path,
    dataset_root: Path,
    runtime_directory: Path,
) -> dict[str, Any]:
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

    for required in (train_list, data_yaml, summary):
        if not required.is_file():
            raise FileNotFoundError(required)

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
        raise AssertionError("Duplicate train paths found.")
    if not all(path.name.startswith("orig_") for path in train_paths):
        raise AssertionError("A non-orig_ path entered the train list.")
    if any(path.name.startswith("aug_") for path in train_paths):
        raise AssertionError("An aug_ path entered the train list.")
    if any(not path.is_file() for path in train_paths):
        raise AssertionError("A listed training image is missing.")

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if "test" in data:
        raise AssertionError("Runtime YAML contains a test split.")

    validation_images = sorted(Path(data["val"]).glob("*.png"))
    if len(validation_images) != EXPECTED_VALID_IMAGES:
        raise AssertionError(
            f"Expected {EXPECTED_VALID_IMAGES} validation images, "
            f"found {len(validation_images)}."
        )

    train_hash = sha256_file(train_list)
    yaml_hash = sha256_file(data_yaml)

    if train_hash != EXPECTED_TRAIN_LIST_SHA256:
        raise AssertionError("Train-list SHA-256 changed.")
    if yaml_hash != EXPECTED_DATA_YAML_SHA256:
        raise AssertionError("Data-YAML SHA-256 changed.")

    return {
        "train_list": train_list,
        "data_yaml": data_yaml,
        "summary": summary,
        "training_images": len(train_paths),
        "validation_images": len(validation_images),
        "train_list_sha256": train_hash,
        "data_yaml_sha256": yaml_hash,
    }


def read_results_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise AssertionError("results.csv has no rows.")

    epoch_key = next(
        (key for key in rows[0] if key.strip() == "epoch"),
        None,
    )
    if epoch_key is None:
        raise AssertionError("Epoch column not found in results.csv.")

    epochs = [int(float(row[epoch_key])) for row in rows]
    if epochs != sorted(epochs):
        raise AssertionError("Epoch values are not sorted.")

    return {
        "rows": len(rows),
        "first_epoch_value": epochs[0],
        "last_epoch_value": epochs[-1],
        "sha256": sha256_file(path),
    }


def verify_source_args(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    args = yaml.safe_load(path.read_text(encoding="utf-8"))

    if args.get("task") != "detect":
        raise AssertionError("Source task is not detect.")
    if args.get("mode") != "train":
        raise AssertionError("Source mode is not train.")
    if args.get("name") != EXPECTED_RUN_NAME:
        raise AssertionError("Source run name is wrong.")

    for name, expected in FROZEN_ARGUMENTS.items():
        if name not in args:
            raise AssertionError(f"Source args missing {name!r}.")
        if not values_equal(args[name], expected):
            raise AssertionError(
                f"Source args mismatch for {name}: "
                f"expected {expected!r}, found {args[name]!r}."
            )

    expected_data = (
        "/kaggle/working/wrist_privid_v3/"
        "grazped_original_only.yaml"
    )
    if str(args.get("data", "")) != expected_data:
        raise AssertionError("Source data path is not the frozen path.")

    return args


def load_and_verify_checkpoint(
    path: Path,
    expected_sha256: str,
    torch_module: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name != "last.pt":
        raise AssertionError("Exact resume requires weights/last.pt.")

    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise AssertionError(
            "Checkpoint SHA-256 mismatch.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_sha256}"
        )

    checkpoint = torch_module.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a dictionary.")

    required = {"epoch", "optimizer", "train_args"}
    missing = required - set(checkpoint)
    if missing:
        raise AssertionError(
            f"Checkpoint missing resume keys: {sorted(missing)}"
        )

    epoch_index = int(checkpoint["epoch"])
    optimizer = checkpoint["optimizer"]
    train_args = checkpoint["train_args"]

    if epoch_index < 0:
        raise AssertionError(
            "Checkpoint appears stripped or finalized."
        )
    if optimizer is None or not isinstance(optimizer, dict):
        raise AssertionError("Optimizer state is missing.")
    if not optimizer.get("state"):
        raise AssertionError("Optimizer state is empty.")
    if not isinstance(train_args, dict):
        raise AssertionError("Checkpoint train_args is invalid.")

    for name, expected in FROZEN_ARGUMENTS.items():
        if name not in train_args:
            raise AssertionError(
                f"Checkpoint train_args missing {name!r}."
            )
        if not values_equal(train_args[name], expected):
            raise AssertionError(
                f"Checkpoint train_args mismatch for {name}: "
                f"expected {expected!r}, "
                f"found {train_args[name]!r}."
            )

    if train_args.get("name") != EXPECTED_RUN_NAME:
        raise AssertionError("Checkpoint run name is wrong.")

    expected_data = (
        "/kaggle/working/wrist_privid_v3/"
        "grazped_original_only.yaml"
    )
    if str(train_args.get("data", "")) != expected_data:
        raise AssertionError("Checkpoint data path is wrong.")

    completed_epochs = epoch_index + 1
    if completed_epochs >= EXPECTED_TOTAL_EPOCHS:
        raise AssertionError(
            "Checkpoint already reached the planned 100 epochs."
        )

    info = {
        "path": str(path),
        "sha256": actual_sha256,
        "size_bytes": path.stat().st_size,
        "stored_epoch_index_zero_based": epoch_index,
        "completed_epochs": completed_epochs,
        "remaining_epochs": EXPECTED_TOTAL_EPOCHS - completed_epochs,
        "optimizer_state_entries": len(
            optimizer.get("state", {})
        ),
        "best_fitness": checkpoint.get("best_fitness"),
        "updates": checkpoint.get("updates"),
        "version": checkpoint.get("version"),
    }
    return checkpoint, info



def stage_source_run_for_resume(
    source_run_directory: Path,
    source_checkpoint: Path,
    destination_run_directory: Path,
    expected_checkpoint_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """
    Copy the previous segment's continuity-critical artifacts into the
    new writable Kaggle run directory before calling resume=True.

    This is essential because the previous run is mounted read-only
    under /kaggle/input, while Ultralytics writes the resumed segment
    under /kaggle/working. Copying results.csv preserves the full epoch
    history, and copying best.pt preserves the best model found before
    the interruption.
    """

    if destination_run_directory.exists():
        raise FileExistsError(
            "Destination run directory already exists:\n"
            f"{destination_run_directory}"
        )

    source_args = source_run_directory / "args.yaml"
    source_results = source_run_directory / "results.csv"
    source_last = source_run_directory / "weights" / "last.pt"
    source_best = source_run_directory / "weights" / "best.pt"

    required_sources = {
        "args_yaml": source_args,
        "results_csv": source_results,
        "last_checkpoint": source_last,
        "best_checkpoint": source_best,
    }

    missing = [
        str(path)
        for path in required_sources.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "The source run is missing continuity-critical files:\n"
            + "\n".join(missing)
        )

    if source_checkpoint.resolve() != source_last.resolve():
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

    for name, source in required_sources.items():
        shutil.copy2(source, destination_files[name])

    staged_information: dict[str, Any] = {
        "source_run_directory": str(source_run_directory),
        "destination_run_directory": str(destination_run_directory),
        "files": {},
    }

    for name, source in required_sources.items():
        destination = destination_files[name]

        source_hash = sha256_file(source)
        destination_hash = sha256_file(destination)

        if source_hash != destination_hash:
            raise AssertionError(
                f"Staged file hash mismatch for {name}."
            )

        staged_information["files"][name] = {
            "source": str(source),
            "destination": str(destination),
            "sha256": destination_hash,
            "size_bytes": destination.stat().st_size,
        }

    staged_last = destination_files["last_checkpoint"]

    if (
        sha256_file(staged_last).lower()
        != expected_checkpoint_sha256.lower()
    ):
        raise AssertionError(
            "The staged last.pt does not match the expected "
            "checkpoint SHA-256."
        )

    return staged_last, staged_information

def verify_model_architecture(model: Any) -> dict[str, Any]:
    names = [
        module.__class__.__name__
        for module in model.model.modules()
    ]
    forbidden = {"WHFE", "MCAux", "MCAuxHead"}
    present = sorted(forbidden & set(names))

    if present:
        raise AssertionError(
            f"Forbidden modules found: {present}"
        )
    if any(
        "_mcaux" in name.lower()
        for name, _ in model.model.named_modules()
    ):
        raise AssertionError("MCAux attributes found.")

    parameters = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )
    if parameters != EXPECTED_PARAMETER_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_PARAMETER_COUNT:,} parameters, "
            f"found {parameters:,}."
        )

    return {
        "parameters": parameters,
        "module_count": len(names),
        "forbidden_modules_present": present,
    }


def verify_completed_outputs(run_directory: Path) -> dict[str, Any]:
    files = {
        "args_yaml": run_directory / "args.yaml",
        "results_csv": run_directory / "results.csv",
        "best_checkpoint": run_directory / "weights" / "best.pt",
        "last_checkpoint": run_directory / "weights" / "last.pt",
    }

    missing = [
        str(file)
        for file in files.values()
        if not file.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing completed outputs:\n" + "\n".join(missing)
        )

    results = read_results_csv(files["results_csv"])
    if results["rows"] != EXPECTED_TOTAL_EPOCHS:
        raise AssertionError(
            "Completed run must have exactly 100 rows."
        )

    return {
        "results": results,
        "files": {
            name: {
                "path": str(file),
                "sha256": sha256_file(file),
                "size_bytes": file.stat().st_size,
            }
            for name, file in files.items()
        },
    }


def main() -> None:
    args = parse_arguments()

    repository_root = Path(__file__).resolve().parents[2]
    ultralytics_root = repository_root / "ultralytics-main"
    config_path = (
        repository_root
        / "project_v3"
        / "configs"
        / "baseline"
        / "baseline_b1_s42.yaml"
    )

    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    runtime_directory = output_root / "wrist_privid_v3"
    requested_run_directory = output_root / args.run_name

    print("=" * 94)
    print("WRIST-PRIVID V3 CONTROLLED BASELINE RESUME")
    print("EXPERIMENT: V3-B1-S42")
    print("=" * 94)

    print("\n1. REPOSITORY VERIFICATION")
    print("-" * 94)
    repository_info = verify_repository(
        repository_root,
        args.expected_commit,
    )
    print("✓ Repository verification passed")

    print("\n2. CONTROLLED CONFIGURATION")
    print("-" * 94)
    verify_repository_config(config_path)
    config_hash = sha256_file(config_path)
    print("Config:", config_path)
    print("Config SHA-256:", config_hash)
    print("✓ Controlled B1 configuration verified")

    print("\n3. ENVIRONMENT FLAGS")
    print("-" * 94)
    flags = set_and_verify_flags()
    print("✓ All custom features disabled")

    print("\n4. OUTPUT-DIRECTORY SAFETY")
    print("-" * 94)
    if args.run_name != EXPECTED_RUN_NAME:
        raise AssertionError(
            f"Run name must remain {EXPECTED_RUN_NAME!r}."
        )
    if requested_run_directory.exists():
        raise FileExistsError(
            "Writable resume directory already exists:\n"
            f"{requested_run_directory}\n"
            "Use a clean Kaggle session."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    print("Writable resume directory:", requested_run_directory)
    print("✓ Writable resume directory is unused")

    print("\n5. RUNTIME DATASET RECONSTRUCTION")
    print("-" * 94)
    dataset_info = rebuild_runtime_dataset(
        repository_root,
        dataset_root,
        runtime_directory,
    )
    for key, value in dataset_info.items():
        print(f"{key}: {value}")
    print("✓ Exact original-only runtime dataset reconstructed")

    print("\n6. ULTRALYTICS IMPORT")
    print("-" * 94)
    if not ultralytics_root.is_dir():
        raise FileNotFoundError(ultralytics_root)

    sys.path.insert(0, str(ultralytics_root))

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
            f"Expected Ultralytics {EXPECTED_ULTRALYTICS_VERSION}, "
            f"found {ultralytics.__version__}."
        )
    if (
        Path(ultralytics.__file__).resolve().parents[1]
        != ultralytics_root.resolve()
    ):
        raise AssertionError(
            "Ultralytics was not imported from the cloned repository."
        )
    print("✓ Correct Ultralytics source verified")

    print("\n7. SOURCE RUN VERIFICATION")
    print("-" * 94)
    source_run_directory = checkpoint_path.parent.parent
    source_args_path = source_run_directory / "args.yaml"
    source_results_path = source_run_directory / "results.csv"

    verify_source_args(source_args_path)
    previous_results = read_results_csv(source_results_path)

    print("Source run directory:", source_run_directory)
    print("Source args SHA-256:", sha256_file(source_args_path))
    print("Previous results rows:", previous_results["rows"])
    print(
        "Previous last epoch value:",
        previous_results["last_epoch_value"],
    )
    print("Previous results SHA-256:", previous_results["sha256"])
    print("✓ Source run artifacts verified")

    print("\n8. CHECKPOINT RESUME-STATE VERIFICATION")
    print("-" * 94)
    _, checkpoint_info = load_and_verify_checkpoint(
        checkpoint_path,
        args.expected_checkpoint_sha256,
        torch,
    )
    for key, value in checkpoint_info.items():
        print(f"{key}: {value}")

    if previous_results["rows"] != checkpoint_info["completed_epochs"]:
        raise AssertionError(
            "results.csv row count and checkpoint epoch disagree.\n"
            f"Rows: {previous_results['rows']}\n"
            f"Completed epochs: {checkpoint_info['completed_epochs']}"
        )
    print("✓ Unstripped checkpoint can exactly resume")

    print("\n9. MODEL ARCHITECTURE")
    print("-" * 94)
    model = YOLO(str(checkpoint_path))
    architecture = verify_model_architecture(model)
    for key, value in architecture.items():
        print(f"{key}: {value}")
    print("✓ Standard YOLO11s resume architecture verified")

    print("\n10. RESUME PREFLIGHT RECORD")
    print("-" * 94)
    preflight = {
        "experiment_id": EXPERIMENT_ID,
        "operation": "exact_cross_session_resume",
        "test_partition_used": False,
        "repository": repository_info,
        "config": {
            "path": str(config_path),
            "sha256": config_hash,
        },
        "flags": flags,
        "dataset": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in dataset_info.items()
        },
        "source_run": {
            "directory": str(source_run_directory),
            "args_path": str(source_args_path),
            "args_sha256": sha256_file(source_args_path),
            "results_path": str(source_results_path),
            **previous_results,
        },
        "checkpoint": checkpoint_info,
        "architecture": architecture,
        "requested_output_directory": str(
            requested_run_directory
        ),
        "resume_overrides": {
            "device": args.device,
            "workers": args.workers,
        },
    }

    preflight_path = (
        runtime_directory
        / "v3_b1_s42_resume_preflight.json"
    )
    write_json(preflight_path, preflight)

    print("Resume preflight:", preflight_path)
    print("Resume preflight SHA-256:", sha256_file(preflight_path))

    print("\n" + "=" * 94)
    print("ALL RESUME CHECKS PASSED")
    print("=" * 94)
    print("Completed epochs:", checkpoint_info["completed_epochs"])
    print("Remaining epochs:", checkpoint_info["remaining_epochs"])
    print("Optimizer state present: Yes")
    print("Test split used: No")
    print("=" * 94)

    if args.preflight_only:
        print()
        print("=" * 94)
        print("RESUME PREFLIGHT-ONLY MODE COMPLETED")
        print("=" * 94)
        print("✓ Exact-resume checkpoint verified")
        print("✓ GPU training was not started")
        print("✓ Official test partition was not accessed")
        print("=" * 94)
        return

    print("\n11. STAGING PREVIOUS RUN HISTORY")
    print("-" * 94)

    staged_checkpoint, staging_info = (
        stage_source_run_for_resume(
            source_run_directory=source_run_directory,
            source_checkpoint=checkpoint_path,
            destination_run_directory=requested_run_directory,
            expected_checkpoint_sha256=(
                args.expected_checkpoint_sha256
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
    print("✓ Writable last.pt staged for exact resume")

    print("\n12. STARTING EXACT RESUME")
    print("-" * 94)

    # Load from the staged writable last.pt, not the read-only
    # /kaggle/input checkpoint. Ultralytics resume=True then restores
    # the optimizer, scheduler, epoch, EMA, and model state.
    resumed_model = YOLO(str(staged_checkpoint))

    resumed_architecture = verify_model_architecture(
        resumed_model
    )

    if resumed_architecture != architecture:
        raise AssertionError(
            "The staged checkpoint architecture differs from "
            "the verified source checkpoint architecture."
        )

    resumed_model.train(
        resume=True,
        device=args.device,
        workers=args.workers,
    )

    trainer = resumed_model.trainer
    if trainer is None:
        raise RuntimeError("Trainer was not retained after resume.")

    actual_run_directory = Path(trainer.save_dir).resolve()
    if actual_run_directory != requested_run_directory.resolve():
        raise AssertionError(
            "Unexpected resumed output directory.\n"
            f"Expected: {requested_run_directory}\n"
            f"Actual:   {actual_run_directory}"
        )

    completed_outputs = verify_completed_outputs(
        actual_run_directory
    )

    completion = {
        "experiment_id": EXPERIMENT_ID,
        "operation": "resume_completed_to_epoch_100",
        "test_partition_used": False,
        "repository": repository_info,
        "source_checkpoint": checkpoint_info,
        "staging": staging_info,
        "run_directory": str(actual_run_directory),
        "completed_outputs": completed_outputs,
    }

    completion_path = (
        actual_run_directory
        / "v3_b1_s42_resume_completion.json"
    )
    write_json(completion_path, completion)

    print("\n" + "=" * 94)
    print("V3-B1-S42 RESUMED TRAINING COMPLETED")
    print("=" * 94)
    print("Run directory:", actual_run_directory)
    print("Final results rows:", completed_outputs["results"]["rows"])
    print(
        "Best checkpoint SHA-256:",
        completed_outputs["files"]["best_checkpoint"]["sha256"],
    )
    print(
        "Last checkpoint SHA-256:",
        completed_outputs["files"]["last_checkpoint"]["sha256"],
    )
    print("✓ Exactly 100 epoch rows verified")
    print("✓ Final artifacts verified")
    print("✓ Official test partition was not accessed")
    print("=" * 94)


if __name__ == "__main__":
    main()