# Experiment Report

## Objective

Build an end-to-end, reproducible Galaxy Zoo morphology prediction workflow. The project starts with EDA and a ResNet18 baseline, expands to model comparison and error analysis, then finishes with explainability, API/UI deployment, DVC metadata, W&B experiment tracking, and CI checks.

## Dataset

| Item | Value |
|---|---|
| Label file | `data/external/training_solutions_rev1.csv` |
| Image directory | `data/processed/rgb_images/` |
| Target dimension | 37 |
| Task | Multi-output probability regression |
| Split | 80% train / 20% validation |

## Baseline Model

| Setting | Value |
|---|---|
| Model | ResNet18 |
| Input size | 224 |
| Output layer | Linear projection to 37 probabilities + Sigmoid |
| Loss | MSELoss |
| Optimizer | Adam |
| Config | `configs/resnet18.yaml` |
| Training entry point | `src/training/train.py` |
| Evaluation entry point | `src/training/evaluate.py` |

## Model Comparison Plan

Week 2 adds two comparison models:

| Model | Purpose | Config |
|---|---|---|
| ResNet18 | Baseline | `configs/resnet18.yaml` |
| ResNet50 | Test whether a deeper CNN improves morphology prediction | `configs/resnet50.yaml` |
| EfficientNet-B0 | Test a parameter-efficient CNN alternative | `configs/efficientnet_b0.yaml` |

## Metrics

The Week 1 evaluation code records:

- Validation loss
- RMSE
- MAE
- Class-wise RMSE
- Prediction distribution
- Ground truth vs prediction plot

## Results

Run the following commands to generate the first baseline result:

```bash
python src/training/train.py --config configs/resnet18.yaml --no_wandb
python src/training/evaluate.py --config configs/resnet18.yaml
```

| Model | Input Size | Loss | RMSE | MAE | Notes |
|---|---:|---:|---:|---:|---|
| ResNet18 | 224 | TBD | TBD | TBD | Week 1 baseline |
| ResNet50 | 224 | TBD | TBD | TBD | Week 2 deeper CNN comparison |
| EfficientNet-B0 | 224 | TBD | TBD | TBD | Week 2 efficient CNN comparison |

The result table intentionally keeps metrics as `TBD` until full training/evaluation is run on the local dataset. This avoids mixing implementation readiness with unverified model quality claims.

## Error Analysis

Week 2 error analysis is documented in `reports/error_analysis.md` and can be generated with:

```bash
python src/training/error_analysis.py \
  --labels data/external/training_solutions_rev1.csv \
  --predictions reports/resnet18_predictions.csv \
  --output reports/error_analysis_summary.json
```

## Explainability

Grad-CAM utilities are available in `src/explainability/gradcam.py`. The intended analysis is to compare:

- Correct predictions where attention focuses on galaxy centers, spiral arms, or disk structure.
- Failure cases where attention may drift to background noise, nearby objects, or image edges.

## Reproducibility and CI

Week 4 adds the reproducibility layer:

| Area | Artifact |
|---|---|
| DVC pipeline | `dvc.yaml`, `dvc.lock` |
| Data/model metadata | `data/processed.dvc`, `models/*.pth.dvc` |
| CI workflow | `.github/workflows/ci.yml` |
| Linting | `ruff check app src tests` |
| Tests | `pytest tests/ -q` |
| Docker builds | API and frontend images in GitHub Actions |

Recommended final verification:

```bash
ruff check app src tests
pytest tests/ -q
docker build -f Dockerfile.api -t galaxy-api:test .
docker build -f Dockerfile.frontend -t galaxy-frontend:test .
```

## Week-by-Week Completion

| Week | Focus | Status |
|---|---|---|
| Week 1 | EDA, data card, ResNet18 baseline, metrics | Complete |
| Week 2 | ResNet50/EfficientNet comparison, error analysis, Grad-CAM | Complete |
| Week 3 | FastAPI, Streamlit, Docker, API tests | Complete |
| Week 4 | DVC, W&B cleanup, CI, README, final reports | Complete |

## Early Observations

- The 37 labels are probabilities, so RMSE and MAE are more interpretable than accuracy.
- Some labels are expected to be sparse because Galaxy Zoo questions are conditional.
- Class-wise RMSE is important for identifying difficult morphology answers that are hidden by the average loss.

## Next Steps

- Run full baseline training and fill the result table.
- Run ResNet50 and EfficientNet-B0 comparisons and fill the result table.
- Generate error analysis for high-error validation examples.
- Export representative Grad-CAM images for success and failure cases.
- Configure a real DVC remote if the project is shared across machines.
