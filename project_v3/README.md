# Wrist-PriViD Project

## Project Title

**Wrist-PriViD: Study-Aware Cross-View Privileged Distillation for Single-View Pediatric Wrist Abnormality Detection**

## Research Objective

The objective of this project is to develop a pediatric wrist abnormality detector that uses complementary radiographic views during training while requiring only one radiograph during inference.

A multi-view teacher will learn from a target radiograph and a complementary radiograph from the same clinical study. The teacher's classification and localization knowledge will then be transferred to a single-view YOLO11 student through knowledge distillation.

## Dataset

The project uses the GRAZPEDWRI-DX dataset with the official patient-disjoint split:

- Original training images: 14,204
- Validation images: 4,094
- Test images: 2,029
- Number of classes: 9

Classes:

1. boneanomaly
2. bonelesion
3. foreignbody
4. fracture
5. metal
6. periostealreaction
7. pronatorsign
8. softtissue
9. text

The offline brightness-augmented training images must not be treated as independent clinical images or complementary radiographic views.

## Core Research Stages

1. Verify dataset metadata and official split integrity.
2. Construct reliable same-study radiographic view pairs.
3. Establish a strong single-view YOLO11 baseline.
4. Train a study-aware multi-view teacher.
5. Distill the teacher into a single-view student.
6. Evaluate dynamic long-tail losses only after distillation succeeds.
7. Perform multi-seed and external validation experiments.

## Development Policy

All V3 model-development decisions must use only the training and validation partitions.

The official test set must remain closed until:

- the final architecture is frozen;
- all loss weights are frozen;
- augmentation settings are frozen;
- the ablation study is complete;
- the final random seeds are fixed;
- the final Git commit is recorded.

## Branch Structure

- `fwnet-v2-validated`: protected V2 reference
- `v3-privid-main`: protected V3 integration branch
- `v3-data-audit`: dataset and pairing audit
- `v3-strong-baseline`: single-view baseline development
- `v3-pair-loader`: multi-view dataset loader
- `v3-multiview-teacher`: teacher-model development
- `v3-student-distillation`: student distillation
- `v3-longtail-loss`: long-tail loss experiments
- `v3-evaluation`: final evaluation tools

## Current Stage

**Stage 1: Dataset metadata and multi-view pairing feasibility audit**

The immediate goal is to determine whether the available dataset contains sufficient metadata to reconstruct reliable PA and lateral image pairs from the same clinical study.

No model training will begin until this feasibility audit is complete.