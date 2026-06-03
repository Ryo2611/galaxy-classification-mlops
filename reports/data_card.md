# Data Card: Galaxy Zoo Morphology Dataset

## Dataset Overview

This project uses Galaxy Zoo morphology probability labels and galaxy RGB images for multi-output image regression. Each image is associated with 37 continuous target probabilities that represent answers in the Galaxy Zoo decision tree.

## Data Sources

| Asset | Path | Purpose |
|---|---|---|
| Labels | `data/external/training_solutions_rev1.csv` | 37-dimensional morphology probabilities |
| Images | `data/processed/rgb_images/` | RGB galaxy images used for training and evaluation |
| Optional FITS | `data/raw/{u,g,r,i,z}/` | Raw SDSS FITS bands for preprocessing experiments |

## Target Definition

The prediction target is a 37-dimensional vector of probabilities. The model uses sigmoid outputs and is trained as a multi-output regression problem.

## Current Data Checks

- Verify that label rows have matching image files.
- Inspect image count and image size distribution.
- Check missing values in the label CSV.
- Inspect per-class label distributions.
- Inspect label correlations because Galaxy Zoo labels follow a decision-tree structure.
- Check simple image quality signals such as brightness and background noise.
- Track processed-data metadata with `data/processed.dvc`.
- Store dataset-wide channel statistics in `data/processed/dataset_stats.json`.

## Train / Validation Split

The Week 1 baseline uses an 80/20 train-validation split with a fixed random seed. This makes the baseline reproducible and keeps model comparison consistent.

## Versioning and Reproducibility

Week 4 adds DVC metadata for the processed dataset and model artifacts. The expected local workflow is:

```bash
dvc add data/processed
dvc repro dataset_stats
```

The DVC stage `dataset_stats` recomputes `data/processed/dataset_stats.json` from `data/processed/rgb_images/`. The `.dvc` metadata should be committed to source control, while large image artifacts should remain local or be pushed to a configured DVC remote.

## Quality Gates

Before training or evaluation, verify:

- `data/external/training_solutions_rev1.csv` exists.
- `data/processed/rgb_images/` contains images whose filenames match Galaxy IDs.
- `data/processed/dataset_stats.json` is present or can be regenerated.
- The label CSV has 37 target columns plus the Galaxy ID column.
- Missing labels and missing image files are reviewed before a full training run.

## Known Limitations

- Galaxy Zoo morphology labels are probabilistic and reflect crowd-sourced uncertainty.
- Some classes are sparse because later decision-tree answers apply only to subsets of galaxies.
- Small, dim, edge-on, or noisy galaxies may be harder to classify.
- The Kaggle label CSV does not include RA/DEC coordinates; SDSS FITS download requires a separate coordinate catalog or matching metadata.

## Intended Use

This dataset is used for portfolio-scale experimentation: EDA, baseline model training, evaluation, error analysis, explainability, and deployment practice.

## Out of Scope

This project is not intended for production scientific catalog generation without additional validation against domain-specific benchmarks.
