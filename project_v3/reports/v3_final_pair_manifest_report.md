# Wrist-PriViD Final Pair Manifest Report

## Audit status

- Test partition used: **No**
- Training visual pairs reviewed: **50**
- Training visual pairs approved: **50**
- Validation visual pairs reviewed: **50**
- Validation visual pairs approved: **50**
- Total manually reviewed pairs: **100**
- Total approved pairs: **100**

## Final manifest sizes

| Split | Paired studies | Target-context samples |
|---|---:|---:|
| Train | 6475 | 12950 |
| Validation | 1884 | 3768 |

Each paired study produces two directed samples:

1. Projection 1 as target and projection 2 as context.
2. Projection 2 as target and projection 1 as context.

Only original training images are used. Offline brightness-augmented copies are not treated as independent radiographic views.

## Training class analysis

| ID | Class | Paired studies | Target images | Instances | Both views | P1 only | P2 only | Single-view-only rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | boneanomaly | 79 | 113 | 173 | 34 | 42 | 3 | 0.570 |
| 1 | bonelesion | 17 | 25 | 25 | 8 | 9 | 0 | 0.529 |
| 2 | foreignbody | 4 | 8 | 8 | 4 | 0 | 0 | 0.000 |
| 3 | fracture | 4552 | 8631 | 11425 | 4079 | 99 | 374 | 0.104 |
| 4 | metal | 229 | 453 | 527 | 224 | 1 | 4 | 0.022 |
| 5 | periostealreaction | 904 | 1445 | 2232 | 541 | 202 | 161 | 0.402 |
| 6 | pronatorsign | 371 | 372 | 373 | 1 | 0 | 370 | 0.997 |
| 7 | softtissue | 260 | 273 | 290 | 13 | 27 | 220 | 0.950 |
| 8 | text | 6474 | 12918 | 15114 | 6444 | 17 | 13 | 0.005 |

## Validation class analysis

| ID | Class | Paired studies | Target images | Instances | Both views | P1 only | P2 only | Single-view-only rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | boneanomaly | 21 | 33 | 48 | 12 | 7 | 2 | 0.429 |
| 1 | bonelesion | 5 | 8 | 8 | 3 | 2 | 0 | 0.400 |
| 2 | foreignbody | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 |
| 3 | fracture | 1329 | 2537 | 3431 | 1208 | 30 | 91 | 0.091 |
| 4 | metal | 65 | 130 | 147 | 65 | 0 | 0 | 0.000 |
| 5 | periostealreaction | 269 | 425 | 652 | 156 | 68 | 45 | 0.420 |
| 6 | pronatorsign | 99 | 99 | 99 | 0 | 0 | 99 | 1.000 |
| 7 | softtissue | 74 | 81 | 86 | 7 | 6 | 61 | 0.905 |
| 8 | text | 1883 | 3760 | 4365 | 1877 | 4 | 2 | 0.003 |

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
