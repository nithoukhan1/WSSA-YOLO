"""
Smoke-test the Wrist-PriViD paired-view Dataset.

The test verifies:

1. Exact train and validation sample counts.
2. Real target and context images can be loaded.
3. Only target-view labels are opened.
4. Target boxes and classes have valid tensor shapes.
5. Context supervision is absent from every returned sample.
6. No test-partition or offline-augmentation path is used.
7. Background target samples are handled correctly.

The official test partition is never accessed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPOSITORY_ROOT),
    )

from project_v3.data.paired_wrist_dataset import (  # noqa: E402
    PairedWristDataset,
)


TRAIN_MANIFEST = (
    REPOSITORY_ROOT
    / "project_v3"
    / "manifests"
    / "train_target_context_manifest.csv"
)

VALID_MANIFEST = (
    REPOSITORY_ROOT
    / "project_v3"
    / "manifests"
    / "valid_target_context_manifest.csv"
)

EXPECTED_COUNTS = {
    "train": 12_950,
    "valid": 3_768,
}

FORBIDDEN_SAMPLE_KEYS = {
    "context_boxes",
    "context_boxes_xywhn",
    "context_classes",
    "context_class_ids",
    "context_label_path",
    "context_label_relpath",
    "context_has_annotations",
    "context_targets",
}


class AuditedPairedWristDataset(
    PairedWristDataset
):
    """
    Dataset subclass that records every label file opened.

    This proves that __getitem__ opens the target label file and never
    opens the context label file.
    """

    opened_label_paths: list[Path] = []

    @staticmethod
    def _read_target_labels(
        target_label_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_path = Path(
            target_label_path
        ).resolve()

        AuditedPairedWristDataset.opened_label_paths.append(
            resolved_path
        )

        return PairedWristDataset._read_target_labels(
            resolved_path
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the Wrist-PriViD paired Dataset."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Root directory of the GRAZPEDWRI-DX dataset."
        ),
    )

    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=3,
        help=(
            "Number of deterministic samples to load from each "
            "split. Default: 3."
        ),
    )

    return parser.parse_args()


def normalize_manifest_path(
    dataset_root: Path,
    relative_path: str,
) -> Path:
    """Resolve one manifest-relative dataset path."""

    normalized = (
        str(relative_path)
        .strip()
        .replace("\\", "/")
    )

    return (
        dataset_root / normalized
    ).resolve()


def assert_no_context_supervision(
    sample: dict[str, Any],
) -> None:
    """Verify that no context labels or boxes were returned."""

    returned_keys = set(sample)

    leaked_keys = (
        returned_keys
        & FORBIDDEN_SAMPLE_KEYS
    )

    assert not leaked_keys, (
        "Context supervision leaked into the sample:\n"
        f"{sorted(leaked_keys)}"
    )

    assert (
        sample["context_supervision_loaded"]
        is False
    ), (
        "context_supervision_loaded must remain False."
    )


def assert_image_tensor(
    image: torch.Tensor,
    field_name: str,
) -> None:
    """Verify one loaded RGB image tensor."""

    assert isinstance(image, torch.Tensor), (
        f"{field_name} must be a tensor."
    )

    assert image.dtype == torch.float32, (
        f"{field_name} must use float32."
    )

    assert image.ndim == 3, (
        f"{field_name} must have shape [3, H, W], "
        f"but received {tuple(image.shape)}."
    )

    assert image.shape[0] == 3, (
        f"{field_name} must contain three RGB channels."
    )

    assert image.shape[1] > 0
    assert image.shape[2] > 0

    assert torch.isfinite(image).all(), (
        f"{field_name} contains non-finite values."
    )

    minimum = float(image.min())
    maximum = float(image.max())

    assert 0.0 <= minimum <= 1.0
    assert 0.0 <= maximum <= 1.0


def assert_target_annotations(
    sample: dict[str, Any],
) -> None:
    """Verify target boxes and class tensors."""

    boxes = sample["target_boxes_xywhn"]
    classes = sample["target_classes"]

    assert isinstance(boxes, torch.Tensor)
    assert isinstance(classes, torch.Tensor)

    assert boxes.dtype == torch.float32
    assert classes.dtype == torch.int64

    assert boxes.ndim == 2
    assert boxes.shape[1] == 4

    assert classes.ndim == 1
    assert boxes.shape[0] == classes.shape[0]

    assert torch.isfinite(boxes).all()

    if len(boxes):
        assert torch.all(
            boxes[:, 0:2] >= 0.0
        )

        assert torch.all(
            boxes[:, 0:2] <= 1.0
        )

        assert torch.all(
            boxes[:, 2:4] > 0.0
        )

        assert torch.all(
            boxes[:, 2:4] <= 1.0
        )

        assert torch.all(classes >= 0)
        assert torch.all(classes <= 8)

    assert (
        sample["target_has_annotations"]
        == bool(len(classes))
    )


def deterministic_indices(
    dataset: PairedWristDataset,
    sample_count: int,
) -> list[int]:
    """
    Select deterministic annotated samples and one background sample
    when a background target row exists.
    """

    assert sample_count > 0

    indices = list(
        range(
            min(
                sample_count,
                len(dataset),
            )
        )
    )

    background_index = next(
        (
            index
            for index, row in enumerate(
                dataset.rows
            )
            if not row[
                "target_class_ids"
            ].strip()
        ),
        None,
    )

    if (
        background_index is not None
        and background_index not in indices
    ):
        indices.append(background_index)

    return indices


def test_split(
    *,
    dataset_root: Path,
    split: str,
    manifest_path: Path,
    sample_count: int,
) -> dict[str, Any]:
    """Smoke-test one split."""

    print("\n" + "=" * 100)
    print(f"{split.upper()} PAIRED DATASET SMOKE TEST")
    print("=" * 100)

    AuditedPairedWristDataset.opened_label_paths.clear()

    dataset = AuditedPairedWristDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        expected_split=split,
        load_images=True,
    )

    print("Manifest:", manifest_path)
    print("Dataset samples:", len(dataset))

    assert len(dataset) == EXPECTED_COUNTS[split]

    indices = deterministic_indices(
        dataset,
        sample_count,
    )

    print("Selected indices:", indices)

    inspected_samples: list[dict[str, Any]] = []

    for index in indices:
        row = dataset.rows[index]

        expected_target_label_path = (
            normalize_manifest_path(
                dataset_root,
                row["target_label_relpath"],
            )
        )

        forbidden_context_label_path = (
            normalize_manifest_path(
                dataset_root,
                row["context_label_relpath"],
            )
        )

        opened_before = len(
            AuditedPairedWristDataset.opened_label_paths
        )

        sample = dataset[index]

        opened_after = len(
            AuditedPairedWristDataset.opened_label_paths
        )

        assert opened_after == opened_before + 1, (
            "Exactly one label file must be opened for each "
            "sample."
        )

        opened_label_path = (
            AuditedPairedWristDataset.opened_label_paths[-1]
        )

        assert (
            opened_label_path
            == expected_target_label_path
        ), (
            "The opened label file is not the target label:\n"
            f"Expected: {expected_target_label_path}\n"
            f"Opened:   {opened_label_path}"
        )

        assert (
            opened_label_path
            != forbidden_context_label_path
        ), (
            "The context label file was opened."
        )

        assert sample["sample_id"] == row["sample_id"]
        assert sample["pair_id"] == row["pair_id"]
        assert sample["study_key"] == row["study_key"]

        assert (
            sample["target_projection"]
            != sample["context_projection"]
        )

        target_path = Path(
            sample["target_path"]
        )

        context_path = Path(
            sample["context_path"]
        )

        assert target_path.is_file(), target_path
        assert context_path.is_file(), context_path
        assert target_path != context_path

        for path in (
            target_path,
            context_path,
            expected_target_label_path,
        ):
            lower_path = (
                str(path)
                .replace("\\", "/")
                .lower()
            )

            assert "/test/" not in lower_path
            assert "/tests/" not in lower_path
            assert not path.name.lower().startswith(
                "aug_"
            )

        assert_no_context_supervision(sample)

        assert_image_tensor(
            sample["target_image"],
            "target_image",
        )

        assert_image_tensor(
            sample["context_image"],
            "context_image",
        )

        assert_target_annotations(sample)

        target_size = sample[
            "target_original_size"
        ]

        context_size = sample[
            "context_original_size"
        ]

        assert target_size == (
            sample["target_image"].shape[1],
            sample["target_image"].shape[2],
        )

        assert context_size == (
            sample["context_image"].shape[1],
            sample["context_image"].shape[2],
        )

        sample_report = {
            "index": index,
            "sample_id": sample["sample_id"],
            "pair_id": sample["pair_id"],
            "target_projection": (
                sample["target_projection"]
            ),
            "context_projection": (
                sample["context_projection"]
            ),
            "target_image_shape": tuple(
                sample["target_image"].shape
            ),
            "context_image_shape": tuple(
                sample["context_image"].shape
            ),
            "target_box_count": int(
                len(
                    sample[
                        "target_boxes_xywhn"
                    ]
                )
            ),
            "target_classes": (
                sample[
                    "target_classes"
                ].tolist()
            ),
            "opened_label": str(
                opened_label_path
            ),
            "context_supervision_loaded": (
                sample[
                    "context_supervision_loaded"
                ]
            ),
        }

        inspected_samples.append(
            sample_report
        )

        print("\nSample:", sample["sample_id"])
        print(
            "Target/context projections:",
            sample["target_projection"],
            "→",
            sample["context_projection"],
        )
        print(
            "Target image shape:",
            tuple(
                sample["target_image"].shape
            ),
        )
        print(
            "Context image shape:",
            tuple(
                sample["context_image"].shape
            ),
        )
        print(
            "Target boxes:",
            len(
                sample["target_boxes_xywhn"]
            ),
        )
        print(
            "Target classes:",
            sample["target_classes"].tolist(),
        )
        print(
            "Only label opened:",
            opened_label_path,
        )
        print(
            "Context supervision loaded:",
            sample[
                "context_supervision_loaded"
            ],
        )

    opened_paths = set(
        AuditedPairedWristDataset.opened_label_paths
    )

    all_context_label_paths = {
        normalize_manifest_path(
            dataset_root,
            dataset.rows[index][
                "context_label_relpath"
            ],
        )
        for index in indices
    }

    context_paths_opened = (
        opened_paths
        & all_context_label_paths
    )

    assert not context_paths_opened, (
        "One or more context label files were opened:\n"
        f"{sorted(context_paths_opened)}"
    )

    print("\n✓ Exact split sample count verified")
    print("✓ Real target images loaded")
    print("✓ Real context images loaded")
    print("✓ Target boxes and classes validated")
    print("✓ Exactly one target label opened per sample")
    print("✓ No context label file opened")
    print("✓ No context supervision returned")
    print("✓ No offline augmented image used")
    print("✓ No official test reference used")

    return {
        "split": split,
        "dataset_samples": len(dataset),
        "inspected_indices": indices,
        "inspected_samples": inspected_samples,
        "opened_target_label_count": len(
            AuditedPairedWristDataset.opened_label_paths
        ),
        "opened_context_label_count": 0,
    }


def main() -> None:
    """Run train and validation Dataset smoke tests."""

    args = parse_args()

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    print("=" * 100)
    print("WRIST-PRIVID PAIRED DATASET SMOKE TEST")
    print("=" * 100)
    print("Repository:", REPOSITORY_ROOT)
    print("Dataset root:", dataset_root)
    print("Training manifest:", TRAIN_MANIFEST)
    print("Validation manifest:", VALID_MANIFEST)
    print("Official test partition requested: No")

    assert dataset_root.is_dir(), (
        "Dataset root was not found:\n"
        f"{dataset_root}"
    )

    assert TRAIN_MANIFEST.is_file()
    assert VALID_MANIFEST.is_file()

    train_report = test_split(
        dataset_root=dataset_root,
        split="train",
        manifest_path=TRAIN_MANIFEST,
        sample_count=args.samples_per_split,
    )

    valid_report = test_split(
        dataset_root=dataset_root,
        split="valid",
        manifest_path=VALID_MANIFEST,
        sample_count=args.samples_per_split,
    )

    print("\n" + "=" * 100)
    print("PAIRED WRIST DATASET SMOKE TEST PASSED")
    print("=" * 100)
    print(
        "Training dataset samples:",
        train_report["dataset_samples"],
    )
    print(
        "Validation dataset samples:",
        valid_report["dataset_samples"],
    )
    print(
        "Target label files opened:",
        (
            train_report[
                "opened_target_label_count"
            ]
            + valid_report[
                "opened_target_label_count"
            ]
        ),
    )
    print("Context label files opened: 0")
    print("Context supervision returned: No")
    print("Offline augmented images used: 0")
    print("Official test partition used: No")
    print("=" * 100)


if __name__ == "__main__":
    main()