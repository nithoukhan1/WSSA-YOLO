"""
Safe paired-view PyTorch Dataset for Wrist-PriViD.

Each dataset item contains:

- one target wrist radiograph;
- one complementary radiograph from the same study;
- detection annotations from the target radiograph only;
- non-label pair metadata.

The context label file is never opened and context annotations are never
returned. This prevents context-view coordinates from entering the
target-view detection loss.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


EXPECTED_COLUMNS = [
    "sample_id",
    "pair_id",
    "split",
    "study_key",
    "patient_id",
    "study_number",
    "laterality",
    "target_projection",
    "context_projection",
    "target_filestem",
    "context_filestem",
    "target_image_relpath",
    "context_image_relpath",
    "target_label_relpath",
    "context_label_relpath",
    "target_class_ids",
    "context_class_ids",
    "study_class_ids",
    "target_has_annotations",
    "context_has_annotations",
    "context_available",
    "timehash_difference",
]

EXPECTED_SAMPLE_COUNTS = {
    "train": 12_950,
    "valid": 3_768,
}

NUM_CLASSES = 9


class PairedWristDataset(Dataset):
    """
    Load same-study directed target-context wrist-image pairs.

    Parameters
    ----------
    dataset_root:
        Root directory of GRAZPEDWRI-DX. Manifest relative paths are
        resolved underneath this directory.

    manifest_path:
        Directed target-context CSV manifest.

    expected_split:
        Must be either ``train`` or ``valid``.

    load_images:
        When True, images are read and converted to RGB float tensors.
        When False, image paths and target labels are returned without
        reading image pixels. This is useful for fast contract tests.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        manifest_path: str | Path,
        expected_split: str,
        *,
        load_images: bool = True,
    ) -> None:
        super().__init__()

        if expected_split not in {"train", "valid"}:
            raise ValueError(
                "expected_split must be 'train' or 'valid', "
                f"but received {expected_split!r}."
            )

        self.dataset_root = Path(
            dataset_root
        ).expanduser().resolve()

        self.manifest_path = Path(
            manifest_path
        ).expanduser().resolve()

        self.expected_split = expected_split
        self.load_images = load_images

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                "Dataset root was not found:\n"
                f"{self.dataset_root}"
            )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                "Manifest was not found:\n"
                f"{self.manifest_path}"
            )

        self.rows = self._read_manifest()

        expected_count = EXPECTED_SAMPLE_COUNTS[
            self.expected_split
        ]

        if len(self.rows) != expected_count:
            raise AssertionError(
                f"{self.expected_split} manifest must contain "
                f"{expected_count:,} rows, but found "
                f"{len(self.rows):,}."
            )

    def _read_manifest(
        self,
    ) -> list[dict[str, str]]:
        """Read and validate the directed-pair manifest."""

        with self.manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise AssertionError(
                    "The manifest has no CSV header."
                )

            fieldnames = list(reader.fieldnames)

            if fieldnames != EXPECTED_COLUMNS:
                raise AssertionError(
                    "Unexpected target-context manifest schema.\n\n"
                    f"Expected:\n{EXPECTED_COLUMNS}\n\n"
                    f"Found:\n{fieldnames}"
                )

            rows = list(reader)

        sample_ids: set[str] = set()

        for row_index, row in enumerate(
            rows,
            start=2,
        ):
            self._validate_manifest_row(
                row=row,
                row_index=row_index,
            )

            sample_id = row["sample_id"]

            if sample_id in sample_ids:
                raise AssertionError(
                    "Duplicate sample_id detected at manifest "
                    f"row {row_index}: {sample_id}"
                )

            sample_ids.add(sample_id)

        return rows

    def _validate_manifest_row(
        self,
        row: dict[str, str],
        row_index: int,
    ) -> None:
        """Validate one manifest row without opening image files."""

        if row["split"] != self.expected_split:
            raise AssertionError(
                f"Manifest row {row_index}: expected split "
                f"{self.expected_split!r}, but found "
                f"{row['split']!r}."
            )

        if row["context_available"] != "1":
            raise AssertionError(
                f"Manifest row {row_index}: context image is "
                "not available."
            )

        if row["target_projection"] not in {"1", "2"}:
            raise AssertionError(
                f"Manifest row {row_index}: invalid target "
                f"projection {row['target_projection']!r}."
            )

        if row["context_projection"] not in {"1", "2"}:
            raise AssertionError(
                f"Manifest row {row_index}: invalid context "
                f"projection {row['context_projection']!r}."
            )

        if (
            row["target_projection"]
            == row["context_projection"]
        ):
            raise AssertionError(
                f"Manifest row {row_index}: target and context "
                "projections must differ."
            )

        if (
            row["target_filestem"]
            == row["context_filestem"]
        ):
            raise AssertionError(
                f"Manifest row {row_index}: target and context "
                "images are identical."
            )

        for path_field in (
            "target_image_relpath",
            "context_image_relpath",
            "target_label_relpath",
            "context_label_relpath",
        ):
            value = (
                row[path_field]
                .strip()
                .replace("\\", "/")
            )

            lower_value = value.lower()
            filename = Path(value).name.lower()

            if (
                "/test/" in lower_value
                or "/tests/" in lower_value
                or "split=test" in lower_value
            ):
                raise AssertionError(
                    f"Manifest row {row_index}: official test "
                    f"reference found in {path_field}: {value}"
                )

            if filename.startswith("aug_"):
                raise AssertionError(
                    f"Manifest row {row_index}: offline "
                    f"augmented file found: {value}"
                )

        time_difference = int(
            row["timehash_difference"]
        )

        if time_difference < 0:
            raise AssertionError(
                f"Manifest row {row_index}: negative "
                "timehash difference."
            )

    def _resolve_under_dataset_root(
        self,
        relative_path: str,
    ) -> Path:
        """
        Resolve a manifest path while preventing traversal outside the
        configured dataset root.
        """

        normalized = (
            str(relative_path)
            .strip()
            .replace("\\", "/")
        )

        candidate = (
            self.dataset_root / normalized
        ).resolve()

        try:
            candidate.relative_to(
                self.dataset_root
            )
        except ValueError as error:
            raise AssertionError(
                "Manifest path escapes the dataset root:\n"
                f"{relative_path}"
            ) from error

        return candidate

    @staticmethod
    def _read_rgb_image(
        image_path: Path,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Read one image as an RGB float tensor in CHW format.

        Returns
        -------
        image_tensor:
            Tensor with shape ``[3, height, width]`` and values in
            the range ``[0, 1]``.

        original_size:
            Tuple ``(height, width)``.
        """

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image was not found:\n{image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            image_array = np.array(
                image,
                dtype=np.uint8,
                copy=True,
            )

        if image_array.ndim != 3:
            raise AssertionError(
                "Expected an RGB image with three dimensions, "
                f"but received shape {image_array.shape}."
            )

        if image_array.shape[2] != 3:
            raise AssertionError(
                "Expected exactly three image channels, but "
                f"received shape {image_array.shape}."
            )

        height, width = image_array.shape[:2]

        image_tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div_(255.0)
        )

        return image_tensor, (height, width)

    @staticmethod
    def _parse_manifest_class_ids(
        value: str,
    ) -> set[int]:
        """Parse a comma-separated manifest target-class field."""

        value = str(value).strip()

        if not value:
            return set()

        class_ids = {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }

        if not all(
            0 <= class_id < NUM_CLASSES
            for class_id in class_ids
        ):
            raise AssertionError(
                "Manifest target class IDs must be between "
                f"0 and {NUM_CLASSES - 1}: "
                f"{sorted(class_ids)}"
            )

        return class_ids

    @staticmethod
    def _read_target_labels(
        target_label_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read target-view YOLO labels.

        This is the only label-reading method in the Dataset.
        The context label path is never passed to this method.

        Returns
        -------
        boxes_xywhn:
            Normalized target boxes with shape ``[N, 4]``.

        classes:
            Integer target classes with shape ``[N]``.
        """

        if not target_label_path.is_file():
            raise FileNotFoundError(
                "Target label file was not found:\n"
                f"{target_label_path}"
            )

        boxes: list[list[float]] = []
        classes: list[int] = []

        text = target_label_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            return (
                torch.empty(
                    (0, 4),
                    dtype=torch.float32,
                ),
                torch.empty(
                    (0,),
                    dtype=torch.int64,
                ),
            )

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            parts = line.strip().split()

            if len(parts) != 5:
                raise AssertionError(
                    f"Invalid YOLO label at "
                    f"{target_label_path}, line {line_number}: "
                    f"expected 5 values, found {len(parts)}."
                )

            raw_class, raw_x, raw_y, raw_w, raw_h = parts

            class_value = float(raw_class)
            class_id = int(class_value)

            if class_value != class_id:
                raise AssertionError(
                    f"Non-integer class ID at "
                    f"{target_label_path}, line {line_number}."
                )

            if not 0 <= class_id < NUM_CLASSES:
                raise AssertionError(
                    f"Class ID {class_id} is outside the valid "
                    f"range 0–{NUM_CLASSES - 1}."
                )

            coordinates = [
                float(raw_x),
                float(raw_y),
                float(raw_w),
                float(raw_h),
            ]

            if not all(
                math.isfinite(value)
                for value in coordinates
            ):
                raise AssertionError(
                    f"Non-finite box coordinate at "
                    f"{target_label_path}, line {line_number}."
                )

            center_x, center_y, width, height = coordinates

            if not (
                0.0 <= center_x <= 1.0
                and 0.0 <= center_y <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            ):
                raise AssertionError(
                    f"Invalid normalized YOLO box at "
                    f"{target_label_path}, line {line_number}: "
                    f"{coordinates}"
                )

            classes.append(class_id)
            boxes.append(coordinates)

        return (
            torch.tensor(
                boxes,
                dtype=torch.float32,
            ),
            torch.tensor(
                classes,
                dtype=torch.int64,
            ),
        )

    def __len__(self) -> int:
        """Return the number of directed target-context samples."""

        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Load one safe directed target-context sample.

        Context label contents are intentionally unavailable in the
        returned dictionary.
        """

        row = self.rows[index]

        target_image_path = (
            self._resolve_under_dataset_root(
                row["target_image_relpath"]
            )
        )

        context_image_path = (
            self._resolve_under_dataset_root(
                row["context_image_relpath"]
            )
        )

        target_label_path = (
            self._resolve_under_dataset_root(
                row["target_label_relpath"]
            )
        )

        # Important safeguard:
        # row["context_label_relpath"] is deliberately not resolved,
        # opened, parsed or returned.

        boxes_xywhn, target_classes = (
            self._read_target_labels(
                target_label_path
            )
        )

        expected_target_classes = (
            self._parse_manifest_class_ids(
                row["target_class_ids"]
            )
        )

        parsed_target_classes = set(
            target_classes.tolist()
        )

        if (
            parsed_target_classes
            != expected_target_classes
        ):
            raise AssertionError(
                f"Target class mismatch for "
                f"{row['sample_id']}.\n"
                f"Manifest: {sorted(expected_target_classes)}\n"
                f"Label: {sorted(parsed_target_classes)}"
            )

        has_annotations = (
            row["target_has_annotations"] == "1"
        )

        if has_annotations != bool(
            len(target_classes)
        ):
            raise AssertionError(
                f"Target annotation flag does not match the "
                f"target label file for {row['sample_id']}."
            )

        sample: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "pair_id": row["pair_id"],
            "study_key": row["study_key"],
            "patient_id": row["patient_id"],
            "laterality": row["laterality"],
            "target_projection": int(
                row["target_projection"]
            ),
            "context_projection": int(
                row["context_projection"]
            ),
            "target_path": str(
                target_image_path
            ),
            "context_path": str(
                context_image_path
            ),
            "target_boxes_xywhn": boxes_xywhn,
            "target_classes": target_classes,
            "target_has_annotations": (
                has_annotations
            ),
            "timehash_difference": int(
                row["timehash_difference"]
            ),
            "context_supervision_loaded": False,
        }

        if self.load_images:
            (
                target_image,
                target_original_size,
            ) = self._read_rgb_image(
                target_image_path
            )

            (
                context_image,
                context_original_size,
            ) = self._read_rgb_image(
                context_image_path
            )

            sample.update(
                {
                    "target_image": target_image,
                    "context_image": context_image,
                    "target_original_size": (
                        target_original_size
                    ),
                    "context_original_size": (
                        context_original_size
                    ),
                }
            )

        return sample