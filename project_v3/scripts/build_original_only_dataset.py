"""
Build the original-only GRAZPEDWRI-DX runtime dataset files for
Wrist-PriViD experiment V3-B1-S42.

The source training directory contains both:

    orig_<filestem>.png
    aug_<filestem>.png

This script creates an explicit training list containing only the
14,204 original images and generates a YOLO data YAML that points to:

    train: train_original.txt
    val:   data/images/valid

The official test partition is never accessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


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

EXPECTED_TRAIN_IMAGES = 14204
EXPECTED_VALID_IMAGES = 4094


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the original-only GRAZPEDWRI-DX "
            "training list for Wrist-PriViD V3."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Path to the GRAZPEDWRI-DX dataset root. "
            "When omitted, GRAZPEDWRI_ROOT or the known "
            "Kaggle path is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/kaggle/working/wrist_privid_v3"
        ),
        help="Directory for generated runtime files.",
    )

    return parser.parse_args()


def resolve_dataset_root(
    supplied_root: Path | None,
) -> Path:
    """Resolve and verify the dataset root."""

    candidates: list[Path] = []

    if supplied_root is not None:
        candidates.append(supplied_root)

    environment_root = os.getenv(
        "GRAZPEDWRI_ROOT"
    )

    if environment_root:
        candidates.append(
            Path(environment_root)
        )

    candidates.extend(
        [
            Path(
                "/kaggle/input/datasets/nithoukhan/"
                "grazpedwri-dx-aug/GRAZPEDWRI-DX"
            ),
            Path(
                "/kaggle/input/grazpedwri-dx-aug/"
                "GRAZPEDWRI-DX"
            ),
        ]
    )

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()

        if (
            (resolved / "train_data.csv").is_file()
            and (resolved / "data").is_dir()
        ):
            return resolved

    searched = "\n".join(
        f"- {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not locate GRAZPEDWRI-DX.\n"
        f"Searched:\n{searched}"
    )


def read_metadata_filestems(
    metadata_path: Path,
) -> list[str]:
    """Read and validate training filestems."""

    with metadata_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"Metadata has no header: {metadata_path}"
            )

        if "filestem" not in reader.fieldnames:
            raise ValueError(
                f"'filestem' column missing from {metadata_path}"
            )

        filestems = [
            str(row["filestem"]).strip()
            for row in reader
        ]

    if len(filestems) != EXPECTED_TRAIN_IMAGES:
        raise AssertionError(
            "Unexpected number of training metadata rows: "
            f"{len(filestems)}"
        )

    if len(set(filestems)) != len(filestems):
        raise AssertionError(
            "Duplicate training filestems detected."
        )

    return sorted(filestems)


def build_original_train_paths(
    dataset_root: Path,
    filestems: list[str],
) -> list[Path]:
    """Build and verify original-image training paths."""

    image_directory = (
        dataset_root
        / "data"
        / "images"
        / "train_aug"
    )

    label_directory = (
        dataset_root
        / "data"
        / "labels"
        / "train_aug"
    )

    image_paths: list[Path] = []

    missing_images: list[Path] = []
    missing_labels: list[Path] = []

    for filestem in filestems:
        image_path = (
            image_directory
            / f"orig_{filestem}.png"
        )

        label_path = (
            label_directory
            / f"orig_{filestem}.txt"
        )

        if not image_path.is_file():
            missing_images.append(image_path)

        if not label_path.is_file():
            missing_labels.append(label_path)

        image_paths.append(image_path)

    if missing_images:
        raise FileNotFoundError(
            "Missing original training images. "
            f"First examples: {missing_images[:5]}"
        )

    if missing_labels:
        raise FileNotFoundError(
            "Missing original training labels. "
            f"First examples: {missing_labels[:5]}"
        )

    if len(image_paths) != EXPECTED_TRAIN_IMAGES:
        raise AssertionError(
            f"Expected {EXPECTED_TRAIN_IMAGES} original images, "
            f"found {len(image_paths)}"
        )

    forbidden_paths = [
        path
        for path in image_paths
        if path.name.startswith("aug_")
    ]

    if forbidden_paths:
        raise AssertionError(
            "Offline augmented images entered the "
            "original-only training list."
        )

    return image_paths


def verify_validation_split(
    dataset_root: Path,
) -> tuple[Path, int]:
    """Verify the validation image and label counts."""

    image_directory = (
        dataset_root
        / "data"
        / "images"
        / "valid"
    )

    label_directory = (
        dataset_root
        / "data"
        / "labels"
        / "valid"
    )

    image_paths = sorted(
        image_directory.glob("*.png")
    )

    label_paths = sorted(
        label_directory.glob("*.txt")
    )

    if len(image_paths) != EXPECTED_VALID_IMAGES:
        raise AssertionError(
            f"Expected {EXPECTED_VALID_IMAGES} validation images, "
            f"found {len(image_paths)}"
        )

    if len(label_paths) != EXPECTED_VALID_IMAGES:
        raise AssertionError(
            f"Expected {EXPECTED_VALID_IMAGES} validation labels, "
            f"found {len(label_paths)}"
        )

    image_stems = {
        path.stem
        for path in image_paths
    }

    label_stems = {
        path.stem
        for path in label_paths
    }

    if image_stems != label_stems:
        raise AssertionError(
            "Validation image and label filestems do not match."
        )

    return image_directory, len(image_paths)


def write_training_list(
    output_path: Path,
    image_paths: list[Path],
) -> None:
    """Write one absolute original-image path per line."""

    output_path.write_text(
        "\n".join(
            str(path.resolve())
            for path in image_paths
        )
        + "\n",
        encoding="utf-8",
    )


def write_data_yaml(
    output_path: Path,
    dataset_root: Path,
    train_list_path: Path,
    validation_directory: Path,
) -> None:
    """Write an Ultralytics-compatible data YAML."""

    yaml_lines = [
        "# Wrist-PriViD V3 original-only dataset",
        "# Experiment: V3-B1-S42",
        "# Official test partition intentionally omitted",
        "",
        f"path: {dataset_root.resolve().as_posix()}",
        f"train: {train_list_path.resolve().as_posix()}",
        f"val: {validation_directory.resolve().as_posix()}",
        "",
        "nc: 9",
        "",
        "names:",
    ]

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        yaml_lines.append(
            f"  {class_id}: {class_name}"
        )

    output_path.write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )


def create_summary(
    dataset_root: Path,
    train_list_path: Path,
    data_yaml_path: Path,
    image_paths: list[Path],
    validation_count: int,
) -> dict[str, Any]:
    """Create the dataset-build audit summary."""

    first_paths = [
        str(path)
        for path in image_paths[:5]
    ]

    last_paths = [
        str(path)
        for path in image_paths[-5:]
    ]

    return {
        "experiment_id": "V3-B1-S42",
        "dataset_root": str(
            dataset_root.resolve()
        ),
        "test_partition_used": False,
        "training_protocol": (
            "original_only_online_augmentation"
        ),
        "training_image_count": len(
            image_paths
        ),
        "validation_image_count": (
            validation_count
        ),
        "allowed_training_prefix": "orig_",
        "forbidden_training_prefix": "aug_",
        "augmented_training_images_included": False,
        "train_list": str(
            train_list_path.resolve()
        ),
        "data_yaml": str(
            data_yaml_path.resolve()
        ),
        "first_training_paths": first_paths,
        "last_training_paths": last_paths,
    }


def main() -> None:
    """Build and verify runtime dataset files."""

    arguments = parse_arguments()

    dataset_root = resolve_dataset_root(
        arguments.dataset_root
    )

    output_directory = (
        arguments.output_dir.expanduser().resolve()
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_metadata_path = (
        dataset_root / "train_data.csv"
    )

    train_list_path = (
        output_directory
        / "train_original.txt"
    )

    data_yaml_path = (
        output_directory
        / "grazped_original_only.yaml"
    )

    summary_path = (
        output_directory
        / "original_only_dataset_summary.json"
    )

    print("=" * 82)
    print("WRIST-PRIVID V3 ORIGINAL-ONLY DATASET BUILD")
    print("=" * 82)
    print("Dataset root:", dataset_root)
    print("Output directory:", output_directory)

    filestems = read_metadata_filestems(
        train_metadata_path
    )

    image_paths = build_original_train_paths(
        dataset_root,
        filestems,
    )

    validation_directory, validation_count = (
        verify_validation_split(
            dataset_root
        )
    )

    write_training_list(
        train_list_path,
        image_paths,
    )

    write_data_yaml(
        data_yaml_path,
        dataset_root,
        train_list_path,
        validation_directory,
    )

    summary = create_summary(
        dataset_root,
        train_list_path,
        data_yaml_path,
        image_paths,
        validation_count,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    written_lines = [
        line.strip()
        for line in train_list_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(written_lines) != EXPECTED_TRAIN_IMAGES:
        raise AssertionError(
            "Written training-list count mismatch."
        )

    if any(
        Path(path).name.startswith("aug_")
        for path in written_lines
    ):
        raise AssertionError(
            "Forbidden aug_ path found in written list."
        )

    if not all(
        Path(path).name.startswith("orig_")
        for path in written_lines
    ):
        raise AssertionError(
            "A non-orig_ path was found in the written list."
        )

    print()
    print("Training images:", len(image_paths))
    print("Validation images:", validation_count)
    print("Augmented training images included: 0")
    print()
    print("Generated:")
    print("-", train_list_path)
    print("-", data_yaml_path)
    print("-", summary_path)
    print()
    print("=" * 82)
    print("✓ ORIGINAL-ONLY DATASET BUILD PASSED")
    print("✓ 14,204 original training images verified")
    print("✓ 14,204 matching training labels verified")
    print("✓ 4,094 validation images verified")
    print("✓ 4,094 matching validation labels verified")
    print("✓ No aug_ training image entered the list")
    print("✓ Official test partition was not accessed")
    print("=" * 82)


if __name__ == "__main__":
    main()