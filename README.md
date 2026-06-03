# 🌌 Galaxy Morphology Classification — End-to-End MLOps Portfolio

> **Production-grade deep learning pipeline** for predicting detailed galaxy morphology probabilities from astronomical images, complete with Explainable AI, experiment tracking, and containerized deployment.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/W%26B-FFBE00?logo=weightsandbiases&logoColor=black" alt="W&B">
</p>

![CI](https://github.com/Ryo2611/galaxy-classification-mlops/actions/workflows/ci.yml/badge.svg)

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Problem Formulation](#problem-formulation)
- [Key Features](#key-features)
- [Week 1 Status](#week-1-status)
- [Week 2 Status](#week-2-status)
- [Week 3 Status](#week-3-status)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Experiment Tracking](#experiment-tracking)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)

---

## Overview

This project tackles the [Galaxy Zoo challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) — predicting 37 morphology probabilities for each galaxy image. Unlike typical notebook-based approaches, this repository demonstrates a **complete MLOps workflow**:

1. **Data acquisition** directly from the Sloan Digital Sky Survey (SDSS)
2. **FITS-to-RGB preprocessing** with Lupton astronomical compositing
3. **Configuration-driven training** with YAML + CLI override support
4. **Grad-CAM explainability** to verify the model focuses on galactic structure
5. **Experiment tracking** via Weights & Biases
6. **Containerized deployment** with FastAPI + Streamlit via Docker Compose

### Dataset

| Item | Detail |
|------|--------|
| Source | [Galaxy Zoo - The Galaxy Challenge (Kaggle)](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) |
| Images | 61,578 galaxy RGB images (424×424 px) |
| Labels | 37 continuous probability values per image |
| Task | Multi-output regression (sigmoid activation) |

---

## Motivation

This project is designed as an applied AI engineering portfolio: it demonstrates not only image recognition model training, but also data analysis, reproducible experiments, error analysis, explainability, API serving, UI delivery, Docker execution, automated testing, and CI/CD.

```text
Raw Galaxy Images
        ↓
Preprocessing / EDA
        ↓
PyTorch Model Training
        ↓
Evaluation / Error Analysis
        ↓
Grad-CAM Explainability
        ↓
FastAPI Inference API
        ↓
Streamlit Demo App
        ↓
Docker + CI/CD + DVC
```

---

## Problem Formulation

Galaxy morphology prediction is treated as multi-output probability regression. For each input image, the model predicts 37 Galaxy Zoo probabilities. The model uses sigmoid outputs and is evaluated with validation loss, RMSE, MAE, and class-wise RMSE.

| Element | Choice |
|---|---|
| Input | RGB galaxy image |
| Output | 37 morphology probabilities |
| Model objective | Multi-output regression |
| Loss | MSELoss by default |
| Primary metrics | RMSE, MAE, class-wise RMSE |

---

## Week 1 Status

Week 1 focuses on proving that the project starts from data understanding and a reproducible baseline, not just a model training script.

| Area | Status | Artifact |
|---|---|---|
| README and structure | Complete | `README.md`, `requirements.txt` |
| EDA notebook | Complete | `notebooks/01_eda.ipynb` |
| Data documentation | Complete | `reports/data_card.md` |
| ResNet18 baseline | Complete | `configs/resnet18.yaml`, `src/models/resnet.py`, `src/training/train.py` |
| Evaluation metrics | Complete | `src/training/metrics.py`, `src/training/evaluate.py` |
| Experiment report | Drafted | `reports/experiment_report.md` |

At this stage, the repository supports EDA, preprocessing, ResNet18 baseline training, and evaluation with validation loss, RMSE, MAE, class-wise RMSE, prediction distribution, and ground truth vs prediction plots.

### Baseline Model

The Week 1 baseline uses ResNet18 with a 37-dimensional output layer to predict galaxy morphology probabilities. All experimental settings are managed by YAML configuration files and can be tracked with Weights & Biases.

| Model | Input Size | Loss | RMSE | MAE | Notes |
|---|---:|---:|---:|---:|---|
| ResNet18 | 224 | TBD | TBD | TBD | Week 1 baseline; run `src/training/evaluate.py` to fill results |

---

## Week 2 Status

Week 2 focuses on model comparison, error analysis, and explainability. The goal is to show data science judgment: compare architectures for a reason, inspect failure modes, and verify what the model attends to.

| Area | Status | Artifact |
|---|---|---|
| ResNet50 comparison | Complete | `configs/resnet50.yaml`, `src/models/model_factory.py` |
| EfficientNet-B0 comparison | Complete | `configs/efficientnet_b0.yaml`, `src/models/efficientnet.py` |
| Model comparison notebook | Complete | `notebooks/03_model_comparison.ipynb` |
| Error analysis | Complete | `src/training/error_analysis.py`, `notebooks/04_error_analysis.ipynb`, `reports/error_analysis.md` |
| Explainability module | Complete | `src/explainability/gradcam.py` |
| Model documentation | Complete | `reports/model_card.md` |

### Model Comparison

ResNet18 is used as the baseline, ResNet50 tests whether a deeper CNN improves accuracy, and EfficientNet-B0 tests a parameter-efficient architecture. This comparison focuses on the trade-off between model complexity and prediction quality.

| Model | Purpose | Config |
|---|---|---|
| ResNet18 | Baseline | `configs/resnet18.yaml` |
| ResNet50 | Deeper CNN comparison | `configs/resnet50.yaml` |
| EfficientNet-B0 | Efficient CNN comparison | `configs/efficientnet_b0.yaml` |

### Error Analysis

The error analysis workflow identifies high-error labels, over/under-predicted probabilities, and difficult samples. Expected difficult cases include small or low-brightness galaxies, ambiguous spiral structures, edge-on galaxies, nearby objects, and noisy backgrounds.

### Explainability

Grad-CAM is used to inspect whether the model focuses on meaningful galaxy structures such as the galactic center, spiral arms, or edge-on disk features.

---

## Week 3 Status

Week 3 turns the model workflow into a usable product surface: FastAPI inference, Streamlit demo UI, Docker execution, and API tests.

| Area | Status | Artifact |
|---|---|---|
| FastAPI inference API | Complete | `app/api/main.py`, `app/api/inference.py` |
| API schemas | Complete | `app/api/schemas.py` |
| Model loading | Complete | `app/api/model_loader.py` |
| Health/model metadata | Complete | `app/api/health.py`, `/health`, `/model-info` |
| Prediction endpoints | Complete | `/predict`, `/predict/explain` |
| Streamlit demo | Complete | `app/frontend/app.py` |
| Docker split | Complete | `Dockerfile.api`, `Dockerfile.frontend`, `docker-compose.yml` |
| API tests | Complete | `tests/test_api.py`, `tests/test_inference.py` |

### API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/model-info` | Model name, checkpoint, device, output size |
| POST | `/predict` | Return 37 morphology probabilities and inference time |
| POST | `/predict/explain` | Return predictions plus Grad-CAM image as base64 PNG |

### Streamlit Demo

The demo app supports image upload, sample image selection, top prediction, inference time, a 37-class probability table, a top-10 probability chart, and optional Grad-CAM display.

---

## Week 4 Status

Week 4 finishes the portfolio layer: reproducibility, experiment tracking, CI/CD, README polish, and report completion.

| Area | Status | Artifact |
|---|---|---|
| DVC pipeline | Complete | `dvc.yaml`, `dvc.lock` |
| DVC data/model tracking metadata | Complete | `data/processed.dvc`, `models/*.pth.dvc` |
| W&B logging cleanup | Complete | `src/training/train.py` |
| GitHub Actions CI | Complete | `.github/workflows/ci.yml` |
| README finalization | Complete | `README.md` |
| Reports | Complete | `reports/data_card.md`, `reports/model_card.md`, `reports/experiment_report.md`, `reports/error_analysis.md` |

---

## Key Features

### 🔬 Data Engineering
- Automated FITS download from SDSS via `astroquery`
- Lupton RGB composite synthesis (parameterized Q and stretch)
- Dataset-level statistics computation (per-channel mean/std)

### 🧠 Deep Learning
- **ResNet-50 / ResNet-18 / EfficientNet-B0** backbone (configurable)
- Multi-output sigmoid regression for 37 Galaxy Zoo classes
- Early stopping with configurable patience
- MPS (Apple Silicon), CUDA, and CPU device auto-detection

### 🔍 Explainable AI (XAI)
- **Grad-CAM** heatmap generation on `layer4` of ResNet
- Verifies the model attends to galactic morphological features (spiral arms, bulges) rather than background noise

### 📊 MLOps & Experiment Tracking
- **Weights & Biases** integration for loss curves, gradients, and hyperparameters
- YAML-based centralized configuration (`configs/default_config.yaml`)
- CLI argument override for rapid experimentation

### 🐳 Deployment
- **FastAPI** backend serving predictions via REST API
- **Streamlit** frontend for interactive image upload and classification
- **Docker Compose** orchestration for one-command deployment

### ✅ Testing
- 44 unit tests covering data pipeline, model architecture, training utilities, Week 1/2 modules, and Week 3 API inference
- All tests run with auto-generated dummy data (no external dependencies)

---

## Architecture

```
                    ┌──────────────┐
                    │  SDSS Server │
                    └──────┬───────┘
                           │ astroquery
                    ┌──────▼───────┐
                    │  FITS Files  │  data/raw/{u,g,r,i,z}/
                    └──────┬───────┘
                           │ Lupton RGB
                    ┌──────▼───────┐
                    │  RGB Images  │  data/processed/rgb_images/
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼───┐ ┌──────▼──────┐
       │ Feature Eng. │ │ Train │ │  Grad-CAM   │
       │ (SNR, CI)    │ │ Loop  │ │ Heatmaps    │
       └──────────────┘ └──┬───┘ └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Best Model  │  models/*.pth
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │                         │
       ┌──────▼──────┐          ┌───────▼───────┐
       │   FastAPI    │◄─────── │   Streamlit   │
       │  :8000/docs  │  REST   │   :8501       │
       └─────────────┘          └───────────────┘
```

---

## Project Structure

```
galaxy_portfolio/
├── .github/
│   └── workflows/ci.yml        # GitHub Actions test/build workflow
├── app/
│   ├── api/
│   │   ├── main.py              # FastAPI app and routes
│   │   ├── schemas.py           # Pydantic response models
│   │   ├── inference.py         # Prediction and Grad-CAM helpers
│   │   ├── model_loader.py      # Model/checkpoint loading
│   │   └── health.py            # Health response
│   └── frontend/app.py          # Streamlit interactive UI
├── configs/
│   ├── default_config.yaml      # Centralized hyperparameter config
│   ├── resnet18.yaml            # Week 1 baseline config
│   ├── resnet50.yaml            # Week 2 comparison config
│   └── efficientnet_b0.yaml     # Week 2 comparison config
├── data/
│   ├── external/                # Galaxy Zoo label CSV
│   ├── raw/{u,g,r,i,z}/        # Raw FITS files from SDSS
│   ├── processed.dvc            # DVC metadata for processed data
│   └── processed/
│       └── rgb_images/          # Preprocessed 424×424 JPG images
├── models/
│   ├── baseline_resnet50_best.pth      # Trained model checkpoint
│   ├── baseline_resnet50_best.pth.dvc  # DVC model metadata
│   ├── resnet18_baseline_best.pth      # Week 1 baseline checkpoint
│   └── resnet18_baseline_best.pth.dvc  # DVC model metadata
├── notebooks/
│   ├── 01_eda.ipynb            # Week 1 exploratory data analysis
│   ├── 03_model_comparison.ipynb
│   └── 04_error_analysis.ipynb
├── reports/
│   ├── data_card.md            # Dataset documentation
│   ├── experiment_report.md    # Baseline experiment notes
│   ├── error_analysis.md       # Failure-mode analysis
│   └── model_card.md           # Model documentation
├── src/
│   ├── data/
│   │   ├── make_dataset.py      # SDSS data fetching via astroquery
│   │   └── preprocess.py        # FITS → Lupton RGB conversion
│   ├── features/
│   │   └── build_features.py    # SNR, concentration index, augmentation
│   ├── models/
│   │   ├── build_model.py       # Legacy model + GalaxyDataset class
│   │   ├── resnet.py            # ResNet18/ResNet50 builders
│   │   ├── efficientnet.py      # EfficientNet-B0 builder
│   │   └── model_factory.py     # Shared model factory
│   ├── training/
│   │   ├── train.py             # Week 1 baseline training
│   │   ├── evaluate.py          # Evaluation and plots
│   │   ├── metrics.py           # RMSE/MAE/class-wise RMSE
│   │   └── error_analysis.py    # Failure-mode summary
│   ├── explainability/
│   │   └── gradcam.py           # Reusable Grad-CAM utilities
│   ├── visualization/
│   │   └── visualize_xai.py     # Grad-CAM implementation
│   └── train.py                 # Config-driven training entry point
├── tests/
│   ├── test_data.py             # Data pipeline tests (11 tests)
│   ├── test_model.py            # Model & training tests (26 tests)
│   ├── test_inference.py        # API inference helper tests
│   └── test_api.py              # FastAPI endpoint tests
├── Dockerfile
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Galaxy Zoo label CSV](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data) (place in `data/external/`)
- A galaxy coordinate catalog with RA/DEC columns is required only when downloading raw FITS files from SDSS

### Installation

```bash
git clone https://github.com/Ryo2611/galaxy-classification-mlops.git
cd galaxy-classification-mlops

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### 1. Run Tests (No Data Required)

```bash
pytest tests/ -v
```

All 44 tests use auto-generated dummy data — no downloads needed.

### 2. Run Week 1 EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

The notebook generates portfolio-ready EDA figures under `reports/figures/`, including sample images, label distributions, and label correlations.

### 3. Download and Preprocess FITS Data

```bash
# Download raw SDSS FITS files from a coordinate catalog
python src/data/make_dataset.py \
    --csv_path data/external/galaxy_coordinates.csv \
    --output_dir data/raw \
    --num_samples 500 \
    --bands ugriz

# Convert FITS bands to Lupton RGB JPG images
python src/data/preprocess.py \
    --raw_dir data/raw \
    --processed_dir data/processed/rgb_images \
    --Q 8 \
    --stretch 0.5
```

### 4. Compute Image Metrics

```bash
# Single image analysis (SNR, concentration index, peak brightness)
python src/features/build_features.py \
    --image_path data/processed/rgb_images/100008.jpg

# Dataset-wide channel statistics
python src/features/build_features.py \
    --compute_stats \
    --img_dir data/processed/rgb_images \
    --output data/processed/dataset_stats.json
```

### 5. Train the Week 1 ResNet18 Baseline

```bash
# Week 1 baseline configuration
python src/training/train.py --config configs/resnet18.yaml

# With CLI overrides
python src/training/train.py \
    --config configs/resnet18.yaml \
    --epochs 30 \
    --batch_size 64 \
    --lr 0.0005

# Disable W&B (offline mode)
python src/training/train.py --config configs/resnet18.yaml --no_wandb
```

### 6. Evaluate the Baseline

```bash
python src/training/evaluate.py \
    --config configs/resnet18.yaml \
    --checkpoint models/resnet18_baseline_best.pth
```

Evaluation outputs:

- `reports/figures/resnet18_metrics.json`
- `reports/resnet18_predictions.csv`
- `reports/figures/prediction_distribution.png`
- `reports/figures/truth_vs_prediction.png`

### 7. Generate Grad-CAM Explanations

```bash
python src/visualization/visualize_xai.py \
    --image_path data/processed/rgb_images/100008.jpg \
    --model_path models/baseline_resnet50_best.pth \
    --output_path data/processed/gradcam/100008_gradcam.png
```

### 8. Run Week 2 Model Comparison

```bash
# ResNet50 comparison
python src/training/train.py --config configs/resnet50.yaml --no_wandb
python src/training/evaluate.py \
    --config configs/resnet50.yaml \
    --checkpoint models/resnet50_comparison_best.pth

# EfficientNet-B0 comparison
python src/training/train.py --config configs/efficientnet_b0.yaml --no_wandb
python src/training/evaluate.py \
    --config configs/efficientnet_b0.yaml \
    --checkpoint models/efficientnet_b0_comparison_best.pth

# Compare metrics in notebook
jupyter notebook notebooks/03_model_comparison.ipynb
```

### 9. Run Week 2 Error Analysis

```bash
python src/training/error_analysis.py \
    --labels data/external/training_solutions_rev1.csv \
    --predictions reports/resnet18_predictions.csv \
    --output reports/error_analysis_summary.json

jupyter notebook notebooks/04_error_analysis.ipynb
```

### 10. Generate Reusable Grad-CAM Explanations

```bash
python src/explainability/gradcam.py \
    --image_path data/processed/rgb_images/100008.jpg \
    --checkpoint models/resnet18_baseline_best.pth \
    --config configs/resnet18.yaml \
    --output_path reports/figures/gradcam_resnet18_100008.png
```

### 11. Deploy Web Application

```bash
# Docker Compose (recommended)
docker-compose up --build

# Or run locally without Docker:
# Terminal 1 — API
uvicorn app.api.main:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run app/frontend/app.py
```

| Service | URL |
|---------|-----|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Swagger Docs | http://localhost:8000/docs |
| FastAPI Health | http://localhost:8000/health |
| FastAPI Model Info | http://localhost:8000/model-info |

API smoke checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info

curl -X POST http://localhost:8000/predict \
    -F "file=@data/processed/rgb_images/100008.jpg"
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Lint source and tests
ruff check app src tests

# Data pipeline tests only
pytest tests/test_data.py -v

# Model & training tests only
pytest tests/test_model.py -v
```

### Test Coverage

| Module | Tests | What's Verified |
|--------|-------|-----------------|
| `test_data.py` | 11 | FITS reading, NaN handling, RGB compositing, JPG generation |
| `test_model.py` | 26 | Output shape (batch, 37), sigmoid range [0,1], dataset behavior, config loading, model builders, Week 1 baseline utilities, Week 2 model factory |
| `test_inference.py` | 3 | Image decoding, invalid image handling, prediction response formatting |
| `test_api.py` | 4 | Health, model-info, prediction, and invalid upload endpoints |

---

## Reproducibility

Week 4 adds DVC pipeline metadata so the core workflow can be rerun from a single entry point once local data and checkpoints are available.

```bash
# Reproduce dataset statistics, training, evaluation, and error analysis
dvc repro

# Inspect tracked metrics
dvc metrics show

# Inspect generated plots when available
dvc plots show
```

Tracked pipeline:

| Stage | Purpose | Main output |
|---|---|---|
| `dataset_stats` | Compute image channel statistics | `data/processed/dataset_stats.json` |
| `train_resnet18` | Train the Week 1 baseline | `models/resnet18_baseline_best.pth` |
| `evaluate_resnet18` | Generate metrics and plots | `reports/figures/resnet18_metrics.json` |
| `error_analysis` | Summarize failure modes | `reports/error_analysis_summary.json` |

For a fresh local machine, regenerate data/model DVC cache metadata after placing the actual artifacts:

```bash
dvc add data/processed
dvc add models/resnet18_baseline_best.pth
dvc add models/baseline_resnet50_best.pth
```

The checked-in `.dvc` files document the expected large local artifacts without requiring those artifacts to be committed to source control.

---

## Experiment Tracking

Training supports Weights & Biases through the `wandb` section of each YAML config. Week 4 standardizes the project name and run names across model configs and logs the main hyperparameters with each epoch.

| Config | W&B run name |
|---|---|
| `configs/resnet18.yaml` | `resnet18-baseline` |
| `configs/resnet50.yaml` | `resnet50-comparison` |
| `configs/efficientnet_b0.yaml` | `efficientnet-b0-comparison` |

Logged values include train loss, validation loss, RMSE, MAE, epoch number, learning rate, batch size, optimizer, model name, and the top class-wise RMSE values.

Run with tracking:

```bash
python src/training/train.py --config configs/resnet18.yaml
```

Run locally without W&B:

```bash
python src/training/train.py --config configs/resnet18.yaml --no_wandb
```

---

## Configuration

All hyperparameters are centralized in [`configs/default_config.yaml`](configs/default_config.yaml):

| Section | Key Parameters |
|---------|---------------|
| `data` | CSV path, raw/processed directories, number of samples |
| `preprocessing` | Lupton Q/stretch, image size, normalization values |
| `model` | Architecture (`resnet50`/`resnet18`/`efficientnet_b0`), dropout |
| `training` | Epochs, batch size, learning rate, optimizer, early stopping patience |
| `augmentation` | Horizontal/vertical flip, rotation, color jitter |
| `wandb` | Project name, run name, enable/disable |
| `checkpoint` | Save directory, model filename |
| `xai` | Grad-CAM target layer, output directory |

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11+ |
| **Deep Learning** | PyTorch, torchvision |
| **Data Source** | SDSS via astroquery, astropy (FITS I/O) |
| **Explainability** | pytorch-grad-cam |
| **Experiment Tracking** | Weights & Biases |
| **Backend API** | FastAPI |
| **Frontend UI** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **Testing** | pytest |
| **Image Processing** | Pillow, OpenCV |

---

## Roadmap

- [x] ResNet-50 baseline with multi-output sigmoid regression
- [x] SDSS data acquisition pipeline (astroquery)
- [x] FITS → Lupton RGB preprocessing
- [x] Grad-CAM explainability integration
- [x] Weights & Biases experiment tracking
- [x] YAML-based configuration system
- [x] FastAPI + Streamlit deployment
- [x] Docker Compose orchestration
- [x] Unit test suite (44 tests)
- [x] Week 1 EDA notebook, data card, ResNet18 baseline, and evaluation metrics
- [x] Week 2 model comparison, error analysis, Grad-CAM module, and model card
- [x] Week 3 FastAPI endpoints, Streamlit demo, Docker split, and API tests
- [x] Week 4 DVC pipeline, W&B cleanup, GitHub Actions CI, README polish, and final reports
- [x] Feature engineering module (SNR, concentration index)
- [x] DVC integration for dataset/model versioning metadata
- [x] GitHub Actions CI/CD pipeline
- [ ] Vision Transformer (ViT) / ConvNeXt backbone upgrade
- [ ] Self-supervised pretraining with Masked Autoencoders (MAE)
- [x] Grad-CAM visualization in Streamlit frontend

---

## License

This project is for educational and portfolio purposes.

---

<p align="center">
  Built with 🔭 for the intersection of deep learning and astrophysics.
</p>
