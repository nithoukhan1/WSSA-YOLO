"""
Inspect and verify the directed target-context manifests used by the
Wrist-PriViD multi-view teacher.

This script does not train a model and does not access the test split.
Its purpose is to freeze the exact CSV schema before implementing the
paired PyTorch Dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
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

EXPECTED_ROWS = {
    "train": 12_950,
    "valid": 3_768,
}

EXPECTED_STUDIES = {
    "train": 6_475,
    "valid": 1_884,
}


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


def read_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Read one CSV with its original field names."""

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


def find_columns(
    fieldnames: list[str],
    required_tokens: tuple[str, ...],
) -> list[str]:
    """Find columns containing all requested name tokens."""

    matches: list[str] = []

    for fieldname in fieldnames:
        normalized = fieldname.strip().lower()

        if all(
            token in normalized
            for token in required_tokens
        ):
            matches.append(fieldname)

    return matches


def contains_forbidden_value(
    value: str,
) -> bool:
    """Detect augmented or test-split references."""

    normalized = str(value).strip().lower()
    normalized = normalized.replace("\\", "/")

    forbidden_fragments = (
        "aug_",
        "/test/",
        "/tests/",
        "split=test",
    )

    return any(
        fragment in normalized
        for fragment in forbidden_fragments
    )


def inspect_manifest(
    split: str,
    path: Path,
) -> dict[str, Any]:
    """Inspect one directed target-context manifest."""

    fieldnames, rows = read_csv(path)

    print("\n" + "=" * 100)
    print(f"{split.upper()} TARGET-CONTEXT MANIFEST")
    print("=" * 100)

    print("Path:", path)
    print("SHA-256:", sha256_file(path))
    print("Rows:", len(rows))
    print("Columns:", len(fieldnames))

    assert len(rows) == EXPECTED_ROWS[split], (
        f"{split} row-count mismatch: "
        f"expected {EXPECTED_ROWS[split]}, "
        f"found {len(rows)}"
    )

    print("\nExact column names:")

    for index, fieldname in enumerate(
        fieldnames,
        start=1,
    ):
        print(f"{index:02d}. {fieldname!r}")

    duplicate_rows = (
        len(rows)
        - len(
            {
                tuple(
                    row.get(field, "")
                    for field in fieldnames
                )
                for row in rows
            }
        )
    )

    print("\nDuplicate complete rows:", duplicate_rows)

    assert duplicate_rows == 0, (
        f"{split} contains duplicate complete rows."
    )

    empty_counts = {
        field: sum(
            not str(row.get(field, "")).strip()
            for row in rows
        )
        for field in fieldnames
    }

    print("\nEmpty-value counts:")

    for field, count in empty_counts.items():
        print(f"{field}: {count}")

    forbidden_hits: list[dict[str, str]] = []

    for row_index, row in enumerate(
        rows,
        start=2,
    ):
        for field, value in row.items():
            if contains_forbidden_value(value):
                forbidden_hits.append(
                    {
                        "row": str(row_index),
                        "field": field,
                        "value": value,
                    }
                )

    assert not forbidden_hits, (
        f"{split} contains forbidden augmented/test references:\n"
        f"{forbidden_hits[:10]}"
    )

    print("✓ No aug_ image reference found")
    print("✓ No test-partition reference found")

    study_candidates = find_columns(
        fieldnames,
        ("study",),
    )

    patient_candidates = find_columns(
        fieldnames,
        ("patient",),
    )

    target_candidates = find_columns(
        fieldnames,
        ("target",),
    )

    context_candidates = find_columns(
        fieldnames,
        ("context",),
    )

    target_image_candidates = [
        field
        for field in target_candidates
        if any(
            token in field.lower()
            for token in (
                "image",
                "path",
                "file",
                "name",
            )
        )
    ]

    context_image_candidates = [
        field
        for field in context_candidates
        if any(
            token in field.lower()
            for token in (
                "image",
                "path",
                "file",
                "name",
            )
        )
    ]

    print("\nCandidate semantic columns:")
    print("Study:", study_candidates)
    print("Patient:", patient_candidates)
    print("Target-related:", target_candidates)
    print("Context-related:", context_candidates)
    print(
        "Target-image candidates:",
        target_image_candidates,
    )
    print(
        "Context-image candidates:",
        context_image_candidates,
    )

    unique_studies = None

    if len(study_candidates) == 1:
        study_column = study_candidates[0]

        study_counts = Counter(
            row[study_column]
            for row in rows
        )

        unique_studies = len(study_counts)

        print(
            "\nUnique studies:",
            unique_studies,
        )

        print(
            "Samples per study distribution:",
            dict(
                sorted(
                    Counter(
                        study_counts.values()
                    ).items()
                )
            ),
        )

        assert (
            unique_studies
            == EXPECTED_STUDIES[split]
        ), (
            f"{split} unique-study mismatch: "
            f"expected {EXPECTED_STUDIES[split]}, "
            f"found {unique_studies}"
        )

        assert set(study_counts.values()) == {2}, (
            f"Every paired study must produce exactly "
            f"two directed samples in {split}."
        )

        print(
            "✓ Every study has exactly two "
            "directed target-context rows"
        )

    print("\nFirst three rows:")

    for row_number, row in enumerate(
        rows[:3],
        start=1,
    ):
        print(
            f"\nRow {row_number}:"
        )
        print(
            json.dumps(
                row,
                indent=2,
                ensure_ascii=False,
            )
        )

    return {
        "split": split,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "fieldnames": fieldnames,
        "duplicate_complete_rows": duplicate_rows,
        "empty_counts": empty_counts,
        "study_candidates": study_candidates,
        "patient_candidates": patient_candidates,
        "target_candidates": target_candidates,
        "context_candidates": context_candidates,
        "target_image_candidates": (
            target_image_candidates
        ),
        "context_image_candidates": (
            context_image_candidates
        ),
        "unique_studies": unique_studies,
        "forbidden_reference_count": len(
            forbidden_hits
        ),
        "first_three_rows": rows[:3],
    }


def main() -> None:
    """Run the teacher-manifest schema inspection."""

    print("=" * 100)
    print("WRIST-PRIVID TEACHER MANIFEST INSPECTION")
    print("=" * 100)

    print("Repository:", REPOSITORY_ROOT)
    print("Manifest directory:", MANIFEST_DIRECTORY)
    print("Official test partition requested: No")

    reports = {
        split: inspect_manifest(
            split,
            path,
        )
        for split, path in MANIFESTS.items()
    }

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORT_DIRECTORY
        / "teacher_manifest_schema_inspection.json"
    )

    report = {
        "experiment": "Wrist-PriViD",
        "purpose": (
            "Freeze the target-context manifest schema "
            "before implementing the paired Dataset."
        ),
        "test_partition_used": False,
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
    print("TEACHER MANIFEST INSPECTION PASSED")
    print("=" * 100)
    print("Training directed samples: 12,950")
    print("Validation directed samples: 3,768")
    print("Training paired studies: 6,475")
    print("Validation paired studies: 1,884")
    print("Offline augmented images used: 0")
    print("Official test partition used: No")
    print("Report:", report_path)
    print("=" * 100)


if __name__ == "__main__":
    main()