"""
Strictly verify the directed target-context data contract used by the
Wrist-PriViD multi-view teacher.

This script verifies:
1. Exact CSV schema.
2. Exact row and pair counts.
3. Two reciprocal directed samples per paired study.
4. Correct P1 <-> P2 target/context relationships.
5. Target/context path reciprocity.
6. Class-list and annotation consistency.
7. No offline augmented filenames.
8. No train-validation patient overlap.
9. No official test-partition reference.

It does not load images and does not access the test partition.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_DIRECTORY = (
    REPOSITORY_ROOT
    / "project_v3"
    / "manifests"
)

REPORT_DIRECTORY = (
    REPOSITORY_ROOT
    / "project_v3"
    / "reports"
)

MANIFESTS = {
    "train": (
        MANIFEST_DIRECTORY
        / "train_target_context_manifest.csv"
    ),
    "valid": (
        MANIFEST_DIRECTORY
        / "valid_target_context_manifest.csv"
    ),
}

EXPECTED = {
    "train": {
        "rows": 12_950,
        "pairs": 6_475,
    },
    "valid": {
        "rows": 3_768,
        "pairs": 1_884,
    },
}

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

RECIPROCAL_FIELDS = [
    (
        "target_filestem",
        "context_filestem",
    ),
    (
        "target_image_relpath",
        "context_image_relpath",
    ),
    (
        "target_label_relpath",
        "context_label_relpath",
    ),
    (
        "target_class_ids",
        "context_class_ids",
    ),
]

SHARED_PAIR_FIELDS = [
    "pair_id",
    "split",
    "study_key",
    "patient_id",
    "study_number",
    "laterality",
    "study_class_ids",
    "timehash_difference",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_manifest(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV manifest and preserve its header order."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest was not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise AssertionError(
                f"CSV has no header:\n{path}"
            )

        rows = list(reader)

    return list(reader.fieldnames), rows


def parse_class_ids(value: str) -> set[int]:
    """Parse a comma-separated class-ID field."""

    value = str(value).strip()

    if not value:
        return set()

    class_ids = {
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    }

    assert all(
        0 <= class_id <= 8
        for class_id in class_ids
    ), (
        f"Class IDs must be between 0 and 8: "
        f"{sorted(class_ids)}"
    )

    return class_ids


def normalized_path(value: str) -> str:
    """Normalize manifest paths for validation."""

    return str(value).strip().replace("\\", "/")


def verify_no_forbidden_reference(
    row: dict[str, str],
) -> None:
    """Reject test references and offline augmented filenames."""

    path_fields = [
        "target_image_relpath",
        "context_image_relpath",
        "target_label_relpath",
        "context_label_relpath",
    ]

    for field in path_fields:
        value = normalized_path(row[field])
        lower_value = value.lower()
        filename = Path(value).name.lower()

        assert "/test/" not in lower_value
        assert "/tests/" not in lower_value
        assert "split=test" not in lower_value

        # Original training images begin with orig_.
        # Offline brightness copies would begin with aug_.
        assert not filename.startswith("aug_"), (
            f"Offline augmented file detected:\n"
            f"{field}: {value}"
        )


def verify_split(
    split: str,
    path: Path,
) -> dict[str, Any]:
    """Verify one directed target-context manifest."""

    expected = EXPECTED[split]
    fieldnames, rows = read_manifest(path)

    print("\n" + "=" * 100)
    print(f"STRICT {split.upper()} MANIFEST VERIFICATION")
    print("=" * 100)

    print("Path:", path)
    print("SHA-256:", sha256_file(path))
    print("Rows:", len(rows))

    assert fieldnames == EXPECTED_COLUMNS, (
        "Manifest schema mismatch.\n\n"
        f"Expected:\n{EXPECTED_COLUMNS}\n\n"
        f"Found:\n{fieldnames}"
    )

    assert len(rows) == expected["rows"], (
        f"{split} row-count mismatch: "
        f"expected {expected['rows']}, "
        f"found {len(rows)}"
    )

    sample_ids = [
        row["sample_id"]
        for row in rows
    ]

    assert len(sample_ids) == len(set(sample_ids)), (
        f"{split} contains duplicate sample_id values."
    )

    pair_groups: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    patient_ids: set[str] = set()
    study_keys: set[str] = set()

    background_target_rows = 0
    background_context_rows = 0

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        assert row["split"] == split, (
            f"Row {row_number}: expected split "
            f"{split!r}, found {row['split']!r}"
        )

        assert row["pair_id"].startswith(
            f"{split}:"
        )

        assert row["sample_id"].startswith(
            f"{split}:"
        )

        assert row["laterality"] in {
            "L",
            "R",
        }

        assert row["target_projection"] in {
            "1",
            "2",
        }

        assert row["context_projection"] in {
            "1",
            "2",
        }

        assert (
            row["target_projection"]
            != row["context_projection"]
        )

        assert row["context_available"] == "1"

        assert (
            row["target_filestem"]
            != row["context_filestem"]
        )

        assert (
            normalized_path(
                row["target_image_relpath"]
            )
            != normalized_path(
                row["context_image_relpath"]
            )
        )

        verify_no_forbidden_reference(row)

        if split == "train":
            assert Path(
                normalized_path(
                    row["target_image_relpath"]
                )
            ).name.startswith("orig_")

            assert Path(
                normalized_path(
                    row["context_image_relpath"]
                )
            ).name.startswith("orig_")

        target_classes = parse_class_ids(
            row["target_class_ids"]
        )

        context_classes = parse_class_ids(
            row["context_class_ids"]
        )

        study_classes = parse_class_ids(
            row["study_class_ids"]
        )

        assert study_classes == (
            target_classes | context_classes
        ), (
            f"Row {row_number}: study classes are not "
            f"the union of target and context classes."
        )

        target_has_annotations = (
            row["target_has_annotations"] == "1"
        )

        context_has_annotations = (
            row["context_has_annotations"] == "1"
        )

        assert target_has_annotations == bool(
            target_classes
        ), (
            f"Row {row_number}: target annotation flag "
            f"does not match target_class_ids."
        )

        assert context_has_annotations == bool(
            context_classes
        ), (
            f"Row {row_number}: context annotation flag "
            f"does not match context_class_ids."
        )

        if not target_classes:
            background_target_rows += 1

        if not context_classes:
            background_context_rows += 1

        time_difference = int(
            row["timehash_difference"]
        )

        assert time_difference >= 0

        patient_ids.add(
            row["patient_id"]
        )

        study_keys.add(
            row["study_key"]
        )

        pair_groups[
            row["pair_id"]
        ].append(row)

    print("Unique directed samples:", len(sample_ids))
    print("Unique pairs:", len(pair_groups))
    print("Unique study keys:", len(study_keys))
    print("Unique patients:", len(patient_ids))

    assert len(pair_groups) == expected["pairs"], (
        f"{split} pair-count mismatch: "
        f"expected {expected['pairs']}, "
        f"found {len(pair_groups)}"
    )

    assert len(study_keys) == expected["pairs"], (
        f"{split} study-key count mismatch: "
        f"expected {expected['pairs']}, "
        f"found {len(study_keys)}"
    )

    for pair_id, pair_rows in pair_groups.items():
        assert len(pair_rows) == 2, (
            f"{pair_id} must contain exactly two "
            f"directed rows; found {len(pair_rows)}."
        )

        rows_by_projection = {
            row["target_projection"]: row
            for row in pair_rows
        }

        assert set(rows_by_projection) == {
            "1",
            "2",
        }, (
            f"{pair_id} does not contain both "
            f"target projection 1 and target projection 2."
        )

        projection_1 = rows_by_projection["1"]
        projection_2 = rows_by_projection["2"]

        assert (
            projection_1["context_projection"]
            == "2"
        )

        assert (
            projection_2["context_projection"]
            == "1"
        )

        assert projection_1[
            "sample_id"
        ].endswith("targetP1")

        assert projection_2[
            "sample_id"
        ].endswith("targetP2")

        for field in SHARED_PAIR_FIELDS:
            assert (
                projection_1[field]
                == projection_2[field]
            ), (
                f"{pair_id}: shared field {field!r} "
                f"does not match between directed rows."
            )

        for target_field, context_field in (
            RECIPROCAL_FIELDS
        ):
            assert (
                projection_1[target_field]
                == projection_2[context_field]
            ), (
                f"{pair_id}: {target_field} and "
                f"{context_field} are not reciprocal."
            )

            assert (
                projection_2[target_field]
                == projection_1[context_field]
            ), (
                f"{pair_id}: reciprocal relationship "
                f"failed for {target_field}."
            )

    print("Background target rows:", background_target_rows)
    print("Background context rows:", background_context_rows)

    print(
        "✓ Exact schema verified"
    )
    print(
        "✓ Exact directed-sample count verified"
    )
    print(
        "✓ Exact paired-study count verified"
    )
    print(
        "✓ Every pair contains P1→P2 and P2→P1"
    )
    print(
        "✓ Target/context paths are reciprocal"
    )
    print(
        "✓ Study classes equal target/context union"
    )
    print(
        "✓ Annotation flags are consistent"
    )
    print(
        "✓ No offline augmented filename found"
    )
    print(
        "✓ No official test reference found"
    )

    return {
        "split": split,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "pairs": len(pair_groups),
        "study_keys": len(study_keys),
        "patients": len(patient_ids),
        "patient_ids": sorted(patient_ids),
        "background_target_rows": (
            background_target_rows
        ),
        "background_context_rows": (
            background_context_rows
        ),
    }


def main() -> None:
    """Run strict teacher-manifest verification."""

    print("=" * 100)
    print("WRIST-PRIVID STRICT TEACHER DATA CONTRACT")
    print("=" * 100)
    print("Repository:", REPOSITORY_ROOT)
    print("Official test partition requested: No")

    reports = {
        split: verify_split(
            split,
            path,
        )
        for split, path in MANIFESTS.items()
    }

    train_patients = set(
        reports["train"].pop("patient_ids")
    )

    valid_patients = set(
        reports["valid"].pop("patient_ids")
    )

    overlap = (
        train_patients
        & valid_patients
    )

    assert not overlap, (
        "Train-validation patient overlap detected:\n"
        f"{sorted(overlap)[:20]}"
    )

    print("\n" + "=" * 100)
    print("CROSS-SPLIT VERIFICATION")
    print("=" * 100)
    print("Train patients:", len(train_patients))
    print("Validation patients:", len(valid_patients))
    print("Patient overlap:", len(overlap))
    print("✓ No train-validation patient overlap")

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIRECTORY
        / "teacher_manifest_contract_verification.json"
    )

    report = {
        "experiment": "Wrist-PriViD",
        "contract": (
            "Directed same-study target-context pairs"
        ),
        "test_partition_used": False,
        "train_validation_patient_overlap": 0,
        "manifests": reports,
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("STRICT TEACHER DATA CONTRACT VERIFIED")
    print("=" * 100)
    print("Training directed samples: 12,950")
    print("Training paired studies: 6,475")
    print("Validation directed samples: 3,768")
    print("Validation paired studies: 1,884")
    print("Each study directions: P1→P2 and P2→P1")
    print("Train-validation patient overlap: 0")
    print("Offline augmented images used: 0")
    print("Official test partition used: No")
    print("Report:", report_path)
    print("=" * 100)


if __name__ == "__main__":
    main()