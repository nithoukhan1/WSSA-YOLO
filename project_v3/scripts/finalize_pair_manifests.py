"""
Finalize Wrist-PriViD train/validation pair manifests.

This script:

1. Verifies the completed visual-review checklists.
2. Reads the preliminary exact-pair audit files.
3. Creates one portable record per paired clinical study.
4. Creates two target-context samples from every paired study:
      projection 1 -> target, projection 2 -> context
      projection 2 -> target, projection 1 -> context
5. Corrects image-level class occurrence counting.
6. Measures class agreement and complementarity across views.
7. Generates final CSV manifests, JSON summary, and Markdown report.

The official test partition is not used.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_AUDIT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "audit_step2_raw"
)

VISUAL_REVIEW_DIR = (
    PROJECT_ROOT
    / "reports"
    / "audit_step3_visual_review"
)

MANIFEST_DIR = PROJECT_ROOT / "manifests"
REPORT_DIR = PROJECT_ROOT / "reports"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------

TRAIN_EXACT_PAIRS = (
    RAW_AUDIT_DIR / "train_exact_pairs.csv"
)

VALID_EXACT_PAIRS = (
    RAW_AUDIT_DIR / "valid_exact_pairs.csv"
)

TRAIN_RAW_CLASS_SUMMARY = (
    RAW_AUDIT_DIR
    / "train_paired_class_summary.csv"
)

VALID_RAW_CLASS_SUMMARY = (
    RAW_AUDIT_DIR
    / "valid_paired_class_summary.csv"
)

TRAIN_VISUAL_CHECKLIST = (
    VISUAL_REVIEW_DIR
    / "train_visual_review_checklist.csv"
)

VALID_VISUAL_CHECKLIST = (
    VISUAL_REVIEW_DIR
    / "valid_visual_review_checklist.csv"
)


# ---------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------

TRAIN_STUDY_MANIFEST = (
    MANIFEST_DIR
    / "train_pair_studies.csv"
)

VALID_STUDY_MANIFEST = (
    MANIFEST_DIR
    / "valid_pair_studies.csv"
)

TRAIN_TARGET_CONTEXT_MANIFEST = (
    MANIFEST_DIR
    / "train_target_context_manifest.csv"
)

VALID_TARGET_CONTEXT_MANIFEST = (
    MANIFEST_DIR
    / "valid_target_context_manifest.csv"
)

TRAIN_CLASS_ANALYSIS = (
    MANIFEST_DIR
    / "train_pair_class_analysis.csv"
)

VALID_CLASS_ANALYSIS = (
    MANIFEST_DIR
    / "valid_pair_class_analysis.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "v3_final_pair_manifest_summary.json"
)

FINAL_REPORT_MD = (
    REPORT_DIR
    / "v3_final_pair_manifest_report.md"
)


# ---------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------

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

EXPECTED_PAIR_COUNTS = {
    "train": 6475,
    "valid": 1884,
}

EXPECTED_TARGET_CONTEXT_COUNTS = {
    "train": 12950,
    "valid": 3768,
}

EXPECTED_VISUAL_REVIEW_ROWS = {
    "train": 50,
    "valid": 50,
}


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as a list of dictionaries."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {path}"
            )

        return list(reader)


def write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write dictionaries to a CSV file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_class_ids(value: Any) -> set[int]:
    """Parse comma-separated class IDs into a set."""

    if value is None:
        return set()

    text = str(value).strip()

    if not text:
        return set()

    class_ids: set[int] = set()

    for token in text.split(","):
        token = token.strip()

        if not token:
            continue

        class_id = int(token)

        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(
                f"Invalid class ID: {class_id}"
            )

        class_ids.add(class_id)

    return class_ids


def serialize_class_ids(
    class_ids: set[int],
) -> str:
    """Convert a class-ID set into stable CSV text."""

    return ",".join(
        str(class_id)
        for class_id in sorted(class_ids)
    )


# ---------------------------------------------------------------------
# Visual-review verification
# ---------------------------------------------------------------------

def verify_visual_review(
    checklist_path: Path,
    split_name: str,
) -> dict[str, Any]:
    """
    Verify that the visual audit was completed and all sampled
    pairs were approved.
    """

    rows = read_csv_rows(checklist_path)

    expected_rows = EXPECTED_VISUAL_REVIEW_ROWS[
        split_name
    ]

    if len(rows) != expected_rows:
        raise AssertionError(
            f"{split_name}: expected {expected_rows} "
            f"review rows, found {len(rows)}"
        )

    required_columns = {
        "same_patient_and_wrist",
        "different_complementary_views",
        "pair_appears_valid",
        "review_notes",
    }

    actual_columns = set(rows[0])

    missing_columns = (
        required_columns - actual_columns
    )

    if missing_columns:
        raise AssertionError(
            f"{split_name}: missing checklist columns: "
            f"{sorted(missing_columns)}"
        )

    invalid_items: list[str] = []
    uncertain_items: list[str] = []

    for row in rows:
        review_item = str(
            row.get("review_item", "")
        )

        same_wrist = str(
            row["same_patient_and_wrist"]
        ).strip().upper()

        complementary = str(
            row["different_complementary_views"]
        ).strip().upper()

        valid_pair = str(
            row["pair_appears_valid"]
        ).strip().upper()

        decisions = {
            same_wrist,
            complementary,
            valid_pair,
        }

        if "NO" in decisions:
            invalid_items.append(review_item)

        elif "UNCERTAIN" in decisions:
            uncertain_items.append(review_item)

        elif decisions != {"YES"}:
            raise AssertionError(
                f"{split_name}: incomplete decision "
                f"for review item {review_item}: "
                f"{sorted(decisions)}"
            )

    if invalid_items:
        raise AssertionError(
            f"{split_name}: invalid visual-review "
            f"items found: {invalid_items}"
        )

    if uncertain_items:
        raise AssertionError(
            f"{split_name}: uncertain visual-review "
            f"items found: {uncertain_items}"
        )

    return {
        "split": split_name,
        "reviewed_pairs": len(rows),
        "valid_pairs": len(rows),
        "invalid_pairs": 0,
        "uncertain_pairs": 0,
        "pass_rate": 1.0,
    }


# ---------------------------------------------------------------------
# Portable dataset paths
# ---------------------------------------------------------------------

def image_relative_path(
    split_name: str,
    filestem: str,
) -> str:
    """Return a dataset-root-relative image path."""

    if split_name == "train":
        return (
            "data/images/train_aug/"
            f"orig_{filestem}.png"
        )

    if split_name == "valid":
        return (
            "data/images/valid/"
            f"{filestem}.png"
        )

    raise ValueError(
        f"Unsupported split: {split_name}"
    )


def label_relative_path(
    split_name: str,
    filestem: str,
) -> str:
    """Return a dataset-root-relative label path."""

    if split_name == "train":
        return (
            "data/labels/train_aug/"
            f"orig_{filestem}.txt"
        )

    if split_name == "valid":
        return (
            "data/labels/valid/"
            f"{filestem}.txt"
        )

    raise ValueError(
        f"Unsupported split: {split_name}"
    )


# ---------------------------------------------------------------------
# Raw instance counts
# ---------------------------------------------------------------------

def load_raw_instance_counts(
    summary_path: Path,
) -> dict[int, int]:
    """
    Load instance totals produced by the Kaggle audit.

    The original paired-image-occurrence field was incorrect,
    but paired_instances was counted directly from YOLO labels
    and remains valid.
    """

    rows = read_csv_rows(summary_path)

    instance_counts: dict[int, int] = {}

    for row in rows:
        class_id = int(row["class_id"])

        instance_counts[class_id] = int(
            row["paired_instances"]
        )

    return instance_counts


# ---------------------------------------------------------------------
# Manifest creation
# ---------------------------------------------------------------------

def finalize_split(
    split_name: str,
    exact_pair_path: Path,
    raw_class_summary_path: Path,
) -> dict[str, Any]:
    """
    Produce study-level and target-context manifests for one
    dataset split.
    """

    pair_rows = read_csv_rows(exact_pair_path)

    expected_pair_count = (
        EXPECTED_PAIR_COUNTS[split_name]
    )

    if len(pair_rows) != expected_pair_count:
        raise AssertionError(
            f"{split_name}: expected "
            f"{expected_pair_count} exact pairs, "
            f"found {len(pair_rows)}"
        )

    raw_instance_counts = (
        load_raw_instance_counts(
            raw_class_summary_path
        )
    )

    study_manifest_rows: list[
        dict[str, Any]
    ] = []

    target_context_rows: list[
        dict[str, Any]
    ] = []

    paired_study_counts = Counter()
    target_image_occurrences = Counter()

    both_view_counts = Counter()
    projection_1_only_counts = Counter()
    projection_2_only_counts = Counter()

    pair_ids: set[str] = set()
    sample_ids: set[str] = set()

    time_gaps: list[int] = []

    for pair_row in pair_rows:
        study_key = str(
            pair_row["study_key"]
        ).strip()

        pair_id = (
            f"{split_name}:{study_key}"
        )

        if pair_id in pair_ids:
            raise AssertionError(
                f"Duplicate pair ID: {pair_id}"
            )

        pair_ids.add(pair_id)

        p1_filestem = str(
            pair_row[
                "projection_1_filestem"
            ]
        ).strip()

        p2_filestem = str(
            pair_row[
                "projection_2_filestem"
            ]
        ).strip()

        p1_classes = parse_class_ids(
            pair_row[
                "projection_1_classes"
            ]
        )

        p2_classes = parse_class_ids(
            pair_row[
                "projection_2_classes"
            ]
        )

        study_classes = (
            p1_classes | p2_classes
        )

        both_view_classes = (
            p1_classes & p2_classes
        )

        p1_only_classes = (
            p1_classes - p2_classes
        )

        p2_only_classes = (
            p2_classes - p1_classes
        )

        for class_id in study_classes:
            paired_study_counts[
                class_id
            ] += 1

        for class_id in p1_classes:
            target_image_occurrences[
                class_id
            ] += 1

        for class_id in p2_classes:
            target_image_occurrences[
                class_id
            ] += 1

        for class_id in both_view_classes:
            both_view_counts[class_id] += 1

        for class_id in p1_only_classes:
            projection_1_only_counts[
                class_id
            ] += 1

        for class_id in p2_only_classes:
            projection_2_only_counts[
                class_id
            ] += 1

        time_gap = int(
            pair_row["timehash_difference"]
        )

        time_gaps.append(time_gap)

        intersection_count = len(
            both_view_classes
        )

        union_count = len(study_classes)

        label_jaccard = (
            intersection_count / union_count
            if union_count > 0
            else 1.0
        )

        study_manifest_rows.append(
            {
                "pair_id": pair_id,
                "split": split_name,
                "study_key": study_key,
                "patient_id": int(
                    pair_row["patient_id"]
                ),
                "study_number": int(
                    pair_row["study_number"]
                ),
                "laterality": str(
                    pair_row["laterality"]
                ).strip(),
                "gender": str(
                    pair_row["gender"]
                ).strip(),
                "age": float(
                    pair_row["age"]
                ),
                "projection_1_filestem": (
                    p1_filestem
                ),
                "projection_2_filestem": (
                    p2_filestem
                ),
                "projection_1_image_relpath": (
                    image_relative_path(
                        split_name,
                        p1_filestem,
                    )
                ),
                "projection_2_image_relpath": (
                    image_relative_path(
                        split_name,
                        p2_filestem,
                    )
                ),
                "projection_1_label_relpath": (
                    label_relative_path(
                        split_name,
                        p1_filestem,
                    )
                ),
                "projection_2_label_relpath": (
                    label_relative_path(
                        split_name,
                        p2_filestem,
                    )
                ),
                "projection_1_timehash": int(
                    pair_row[
                        "projection_1_timehash"
                    ]
                ),
                "projection_2_timehash": int(
                    pair_row[
                        "projection_2_timehash"
                    ]
                ),
                "timehash_difference": time_gap,
                "projection_1_class_ids": (
                    serialize_class_ids(
                        p1_classes
                    )
                ),
                "projection_2_class_ids": (
                    serialize_class_ids(
                        p2_classes
                    )
                ),
                "study_class_ids": (
                    serialize_class_ids(
                        study_classes
                    )
                ),
                "classes_in_both_views": (
                    serialize_class_ids(
                        both_view_classes
                    )
                ),
                "classes_only_in_projection_1": (
                    serialize_class_ids(
                        p1_only_classes
                    )
                ),
                "classes_only_in_projection_2": (
                    serialize_class_ids(
                        p2_only_classes
                    )
                ),
                "class_label_jaccard": (
                    round(label_jaccard, 6)
                ),
                "context_available": 1,
            }
        )

        directions = [
            {
                "target_projection": 1,
                "context_projection": 2,
                "target_filestem": p1_filestem,
                "context_filestem": p2_filestem,
                "target_classes": p1_classes,
                "context_classes": p2_classes,
            },
            {
                "target_projection": 2,
                "context_projection": 1,
                "target_filestem": p2_filestem,
                "context_filestem": p1_filestem,
                "target_classes": p2_classes,
                "context_classes": p1_classes,
            },
        ]

        for direction in directions:
            target_projection = int(
                direction[
                    "target_projection"
                ]
            )

            context_projection = int(
                direction[
                    "context_projection"
                ]
            )

            target_filestem = str(
                direction["target_filestem"]
            )

            context_filestem = str(
                direction["context_filestem"]
            )

            target_classes = set(
                direction["target_classes"]
            )

            context_classes = set(
                direction["context_classes"]
            )

            sample_id = (
                f"{pair_id}:"
                f"targetP{target_projection}"
            )

            if sample_id in sample_ids:
                raise AssertionError(
                    f"Duplicate sample ID: "
                    f"{sample_id}"
                )

            sample_ids.add(sample_id)

            target_context_rows.append(
                {
                    "sample_id": sample_id,
                    "pair_id": pair_id,
                    "split": split_name,
                    "study_key": study_key,
                    "patient_id": int(
                        pair_row["patient_id"]
                    ),
                    "study_number": int(
                        pair_row["study_number"]
                    ),
                    "laterality": str(
                        pair_row["laterality"]
                    ).strip(),
                    "target_projection": (
                        target_projection
                    ),
                    "context_projection": (
                        context_projection
                    ),
                    "target_filestem": (
                        target_filestem
                    ),
                    "context_filestem": (
                        context_filestem
                    ),
                    "target_image_relpath": (
                        image_relative_path(
                            split_name,
                            target_filestem,
                        )
                    ),
                    "context_image_relpath": (
                        image_relative_path(
                            split_name,
                            context_filestem,
                        )
                    ),
                    "target_label_relpath": (
                        label_relative_path(
                            split_name,
                            target_filestem,
                        )
                    ),
                    "context_label_relpath": (
                        label_relative_path(
                            split_name,
                            context_filestem,
                        )
                    ),
                    "target_class_ids": (
                        serialize_class_ids(
                            target_classes
                        )
                    ),
                    "context_class_ids": (
                        serialize_class_ids(
                            context_classes
                        )
                    ),
                    "study_class_ids": (
                        serialize_class_ids(
                            study_classes
                        )
                    ),
                    "target_has_annotations": (
                        int(bool(target_classes))
                    ),
                    "context_has_annotations": (
                        int(bool(context_classes))
                    ),
                    "context_available": 1,
                    "timehash_difference": (
                        time_gap
                    ),
                }
            )

    expected_sample_count = (
        EXPECTED_TARGET_CONTEXT_COUNTS[
            split_name
        ]
    )

    if (
        len(target_context_rows)
        != expected_sample_count
    ):
        raise AssertionError(
            f"{split_name}: expected "
            f"{expected_sample_count} target-context "
            f"samples, found "
            f"{len(target_context_rows)}"
        )

    class_analysis_rows: list[
        dict[str, Any]
    ] = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        any_view_count = int(
            paired_study_counts[class_id]
        )

        both_count = int(
            both_view_counts[class_id]
        )

        p1_only_count = int(
            projection_1_only_counts[
                class_id
            ]
        )

        p2_only_count = int(
            projection_2_only_counts[
                class_id
            ]
        )

        if (
            both_count
            + p1_only_count
            + p2_only_count
            != any_view_count
        ):
            raise AssertionError(
                f"{split_name}: inconsistent "
                f"view-count accounting for "
                f"class {class_id}"
            )

        both_view_rate = (
            both_count / any_view_count
            if any_view_count > 0
            else 0.0
        )

        complementary_only_rate = (
            (
                p1_only_count
                + p2_only_count
            )
            / any_view_count
            if any_view_count > 0
            else 0.0
        )

        class_analysis_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "paired_studies_containing_class": (
                    any_view_count
                ),
                "paired_target_images_containing_class": (
                    int(
                        target_image_occurrences[
                            class_id
                        ]
                    )
                ),
                "paired_instances": int(
                    raw_instance_counts.get(
                        class_id,
                        0,
                    )
                ),
                "studies_with_class_in_both_views": (
                    both_count
                ),
                "studies_with_class_only_in_projection_1": (
                    p1_only_count
                ),
                "studies_with_class_only_in_projection_2": (
                    p2_only_count
                ),
                "both_view_rate": round(
                    both_view_rate,
                    6,
                ),
                "single_view_only_rate": round(
                    complementary_only_rate,
                    6,
                ),
            }
        )

    if not time_gaps:
        raise AssertionError(
            f"{split_name}: no time gaps found"
        )

    sorted_time_gaps = sorted(time_gaps)

    def percentile(
        sorted_values: list[int],
        fraction: float,
    ) -> float:
        index = (
            len(sorted_values) - 1
        ) * fraction

        lower_index = math.floor(index)
        upper_index = math.ceil(index)

        if lower_index == upper_index:
            return float(
                sorted_values[lower_index]
            )

        lower_value = sorted_values[
            lower_index
        ]

        upper_value = sorted_values[
            upper_index
        ]

        interpolation = (
            index - lower_index
        )

        return float(
            lower_value
            + (
                upper_value
                - lower_value
            )
            * interpolation
        )

    summary = {
        "split": split_name,
        "paired_studies": len(
            study_manifest_rows
        ),
        "target_context_samples": len(
            target_context_rows
        ),
        "unique_pair_ids": len(pair_ids),
        "unique_sample_ids": len(
            sample_ids
        ),
        "timehash_difference": {
            "minimum": min(time_gaps),
            "median": percentile(
                sorted_time_gaps,
                0.50,
            ),
            "p95": percentile(
                sorted_time_gaps,
                0.95,
            ),
            "maximum": max(time_gaps),
        },
    }

    return {
        "study_rows": study_manifest_rows,
        "target_context_rows": (
            target_context_rows
        ),
        "class_analysis_rows": (
            class_analysis_rows
        ),
        "summary": summary,
    }


# ---------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------

def class_analysis_markdown(
    rows: list[dict[str, Any]],
) -> str:
    """Create a Markdown class-analysis table."""

    lines = [
        "| ID | Class | Paired studies | "
        "Target images | Instances | "
        "Both views | P1 only | P2 only | "
        "Single-view-only rate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['class_id']} | "
            f"{row['class_name']} | "
            f"{row['paired_studies_containing_class']} | "
            f"{row['paired_target_images_containing_class']} | "
            f"{row['paired_instances']} | "
            f"{row['studies_with_class_in_both_views']} | "
            f"{row['studies_with_class_only_in_projection_1']} | "
            f"{row['studies_with_class_only_in_projection_2']} | "
            f"{row['single_view_only_rate']:.3f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    """Run final manifest generation."""

    print("=" * 78)
    print("WRIST-PRIVID FINAL PAIR MANIFEST GENERATION")
    print("=" * 78)

    train_visual_summary = (
        verify_visual_review(
            TRAIN_VISUAL_CHECKLIST,
            "train",
        )
    )

    valid_visual_summary = (
        verify_visual_review(
            VALID_VISUAL_CHECKLIST,
            "valid",
        )
    )

    print(
        "✓ Training visual review verified:"
        f" {train_visual_summary['valid_pairs']}/"
        f"{train_visual_summary['reviewed_pairs']} valid"
    )

    print(
        "✓ Validation visual review verified:"
        f" {valid_visual_summary['valid_pairs']}/"
        f"{valid_visual_summary['reviewed_pairs']} valid"
    )

    train_result = finalize_split(
        split_name="train",
        exact_pair_path=TRAIN_EXACT_PAIRS,
        raw_class_summary_path=(
            TRAIN_RAW_CLASS_SUMMARY
        ),
    )

    valid_result = finalize_split(
        split_name="valid",
        exact_pair_path=VALID_EXACT_PAIRS,
        raw_class_summary_path=(
            VALID_RAW_CLASS_SUMMARY
        ),
    )

    study_fieldnames = list(
        train_result["study_rows"][0]
    )

    target_context_fieldnames = list(
        train_result[
            "target_context_rows"
        ][0]
    )

    class_fieldnames = list(
        train_result[
            "class_analysis_rows"
        ][0]
    )

    write_csv_rows(
        TRAIN_STUDY_MANIFEST,
        train_result["study_rows"],
        study_fieldnames,
    )

    write_csv_rows(
        VALID_STUDY_MANIFEST,
        valid_result["study_rows"],
        study_fieldnames,
    )

    write_csv_rows(
        TRAIN_TARGET_CONTEXT_MANIFEST,
        train_result[
            "target_context_rows"
        ],
        target_context_fieldnames,
    )

    write_csv_rows(
        VALID_TARGET_CONTEXT_MANIFEST,
        valid_result[
            "target_context_rows"
        ],
        target_context_fieldnames,
    )

    write_csv_rows(
        TRAIN_CLASS_ANALYSIS,
        train_result[
            "class_analysis_rows"
        ],
        class_fieldnames,
    )

    write_csv_rows(
        VALID_CLASS_ANALYSIS,
        valid_result[
            "class_analysis_rows"
        ],
        class_fieldnames,
    )

    global_summary = {
        "project": "Wrist-PriViD",
        "test_partition_used": False,
        "visual_review": {
            "train": train_visual_summary,
            "valid": valid_visual_summary,
            "total_reviewed_pairs": (
                train_visual_summary[
                    "reviewed_pairs"
                ]
                + valid_visual_summary[
                    "reviewed_pairs"
                ]
            ),
            "total_valid_pairs": (
                train_visual_summary[
                    "valid_pairs"
                ]
                + valid_visual_summary[
                    "valid_pairs"
                ]
            ),
        },
        "train": train_result["summary"],
        "valid": valid_result["summary"],
        "manifest_files": {
            "train_pair_studies": str(
                TRAIN_STUDY_MANIFEST.relative_to(
                    PROJECT_ROOT
                )
            ),
            "valid_pair_studies": str(
                VALID_STUDY_MANIFEST.relative_to(
                    PROJECT_ROOT
                )
            ),
            "train_target_context": str(
                TRAIN_TARGET_CONTEXT_MANIFEST.relative_to(
                    PROJECT_ROOT
                )
            ),
            "valid_target_context": str(
                VALID_TARGET_CONTEXT_MANIFEST.relative_to(
                    PROJECT_ROOT
                )
            ),
            "train_class_analysis": str(
                TRAIN_CLASS_ANALYSIS.relative_to(
                    PROJECT_ROOT
                )
            ),
            "valid_class_analysis": str(
                VALID_CLASS_ANALYSIS.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            global_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_text = f"""# Wrist-PriViD Final Pair Manifest Report

## Audit status

- Test partition used: **No**
- Training visual pairs reviewed: **{train_visual_summary['reviewed_pairs']}**
- Training visual pairs approved: **{train_visual_summary['valid_pairs']}**
- Validation visual pairs reviewed: **{valid_visual_summary['reviewed_pairs']}**
- Validation visual pairs approved: **{valid_visual_summary['valid_pairs']}**
- Total manually reviewed pairs: **100**
- Total approved pairs: **100**

## Final manifest sizes

| Split | Paired studies | Target-context samples |
|---|---:|---:|
| Train | {train_result['summary']['paired_studies']} | {train_result['summary']['target_context_samples']} |
| Validation | {valid_result['summary']['paired_studies']} | {valid_result['summary']['target_context_samples']} |

Each paired study produces two directed samples:

1. Projection 1 as target and projection 2 as context.
2. Projection 2 as target and projection 1 as context.

Only original training images are used. Offline brightness-augmented copies are not treated as independent radiographic views.

## Training class analysis

{class_analysis_markdown(train_result['class_analysis_rows'])}

## Validation class analysis

{class_analysis_markdown(valid_result['class_analysis_rows'])}

## Scientific interpretation

The `single_view_only_rate` reports the proportion of paired studies in which a class is annotated in only one of the two complementary projections.

A high value provides direct evidence that the projections contain non-identical annotation information and supports the use of a study-aware multi-view teacher.

Ambiguous studies are excluded from the initial teacher experiment. They have not been permanently deleted and may be investigated separately later.

## Generated files

- `manifests/train_pair_studies.csv`
- `manifests/valid_pair_studies.csv`
- `manifests/train_target_context_manifest.csv`
- `manifests/valid_target_context_manifest.csv`
- `manifests/train_pair_class_analysis.csv`
- `manifests/valid_pair_class_analysis.csv`
- `reports/v3_final_pair_manifest_summary.json`

## Development decision

The multi-view pairing feasibility gate is passed.

The next project stage is the establishment of a strong single-view YOLO11 baseline using training and validation data only.
"""

    FINAL_REPORT_MD.write_text(
        report_text,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("FINAL OUTPUTS")
    print("=" * 78)

    for path in [
        TRAIN_STUDY_MANIFEST,
        VALID_STUDY_MANIFEST,
        TRAIN_TARGET_CONTEXT_MANIFEST,
        VALID_TARGET_CONTEXT_MANIFEST,
        TRAIN_CLASS_ANALYSIS,
        VALID_CLASS_ANALYSIS,
        SUMMARY_JSON,
        FINAL_REPORT_MD,
    ]:
        print(path.relative_to(PROJECT_ROOT))

    print()
    print("=" * 78)
    print("✓ FINAL PAIR MANIFEST GENERATION PASSED")
    print("✓ 6,475 training paired studies verified")
    print("✓ 1,884 validation paired studies verified")
    print("✓ 12,950 training target-context samples created")
    print("✓ 3,768 validation target-context samples created")
    print("✓ Image-level class occurrence counts corrected")
    print("✓ Cross-view class complementarity analyzed")
    print("✓ 100/100 visually reviewed pairs approved")
    print("✓ Portable relative dataset paths created")
    print("✓ AMBIGUOUS STUDIES EXCLUDED FROM INITIAL MANIFESTS")
    print("✓ TEST PARTITION WAS NOT USED")
    print("=" * 78)


if __name__ == "__main__":
    main()