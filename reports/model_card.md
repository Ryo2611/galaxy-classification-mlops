# Model Card: Galaxy Morphology Baseline

## Model Details

| Field | Description |
|---|---|
| Model family | CNN image regression |
| Baseline | ResNet18 |
| Comparison models | ResNet50, EfficientNet-B0 |
| Output | 37 Galaxy Zoo morphology probabilities |
| Activation | Sigmoid |
| Training config | YAML-based configs in `configs/` |
| Deployment surface | FastAPI API and Streamlit frontend |

## Intended Use

This model is intended for portfolio-scale galaxy morphology prediction experiments. It demonstrates model training, evaluation, error analysis, explainability, and deployment workflow design.

## Not Intended For

- Production scientific catalog generation
- Replacing expert astronomical classification
- High-stakes automated scientific claims without external validation

## Dataset

The model uses Galaxy Zoo image labels from `data/external/training_solutions_rev1.csv` and RGB galaxy images from `data/processed/rgb_images/`.

## Evaluation

Primary metrics:

- Validation loss
- RMSE
- MAE
- Class-wise RMSE

Qualitative evaluation:

- Prediction distribution
- Ground truth vs prediction plot
- High-error sample review
- Grad-CAM inspection

## Experiment Tracking

Weights & Biases logging is configured through YAML. Week 4 standardizes project/run naming and logs:

- Training loss and validation loss
- RMSE and MAE
- Model name, optimizer, learning rate, and batch size
- Top class-wise RMSE values

| Model | Config | W&B run name | Status |
|---|---|---|---|
| ResNet18 | `configs/resnet18.yaml` | `resnet18-baseline` | Baseline workflow ready |
| ResNet50 | `configs/resnet50.yaml` | `resnet50-comparison` | Comparison workflow ready |
| EfficientNet-B0 | `configs/efficientnet_b0.yaml` | `efficientnet-b0-comparison` | Comparison workflow ready |

## Reproducibility

Model training and evaluation are represented in `dvc.yaml`.

| Stage | Output |
|---|---|
| `train_resnet18` | `models/resnet18_baseline_best.pth` |
| `evaluate_resnet18` | `reports/figures/resnet18_metrics.json`, plots |
| `error_analysis` | `reports/error_analysis_summary.json` |

Large checkpoints are represented by `.dvc` metadata files so the repository can stay source-focused.

## Limitations

- Galaxy Zoo labels are probabilistic and reflect crowd-sourced uncertainty.
- Some labels are sparse because the survey questions are conditional.
- Small, dim, edge-on, or noisy images may have larger errors.
- The current model does not yet use domain-specific astronomical priors.

## Ethical / Scientific Considerations

The project should communicate uncertainty clearly. Model predictions should be treated as experimental outputs, not definitive scientific classifications.

## Future Work

- Complete model comparison across ResNet18, ResNet50, and EfficientNet-B0.
- Add systematic error analysis figures.
- Use Grad-CAM outputs to validate whether the model attends to meaningful galaxy structure.
- Add external validation against astronomy-specific benchmarks.
- Try ViT, ConvNeXt, or MAE pretraining after the CNN baseline workflow is stable.
