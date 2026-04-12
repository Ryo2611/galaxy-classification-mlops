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

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Testing](#testing)
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
- 30 unit tests covering data pipeline, model architecture, and training utilities
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
├── app/
│   ├── api/main.py              # FastAPI inference endpoint
│   └── frontend/app.py          # Streamlit interactive UI
├── configs/
│   └── default_config.yaml      # Centralized hyperparameter config
├── data/
│   ├── external/                # Galaxy Zoo label CSV
│   ├── raw/{u,g,r,i,z}/        # Raw FITS files from SDSS
│   └── processed/
│       └── rgb_images/          # Preprocessed 424×424 JPG images
├── models/
│   └── baseline_resnet50_best.pth  # Trained model checkpoint (~90MB)
├── src/
│   ├── data/
│   │   ├── make_dataset.py      # SDSS data fetching via astroquery
│   │   └── preprocess.py        # FITS → Lupton RGB conversion
│   ├── features/
│   │   └── build_features.py    # SNR, concentration index, augmentation
│   ├── models/
│   │   └── build_model.py       # ResNet-50 model + GalaxyDataset class
│   ├── visualization/
│   │   └── visualize_xai.py     # Grad-CAM implementation
│   └── train.py                 # Config-driven training entry point
├── tests/
│   ├── test_data.py             # Data pipeline tests (11 tests)
│   └── test_model.py            # Model & training tests (19 tests)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Galaxy Zoo label CSV](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data) (place in `data/external/`)

### Installation

```bash
git clone https://github.com/Ryo2611/galaxy-classification-mlops.git
cd galaxy-classification-mlops

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install pyyaml pytest
```

---

## Usage

### 1. Run Tests (No Data Required)

```bash
pytest tests/ -v
```

All 30 tests use auto-generated dummy data — no downloads needed.

### 2. Compute Image Metrics

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

### 3. Train the Model

```bash
# Default configuration
python src/train.py --config configs/default_config.yaml

# With CLI overrides
python src/train.py \
    --config configs/default_config.yaml \
    --epochs 30 \
    --batch_size 64 \
    --lr 0.0005

# Disable W&B (offline mode)
python src/train.py --no_wandb
```

### 4. Generate Grad-CAM Explanations

```bash
python src/visualization/visualize_xai.py \
    --image_path data/processed/rgb_images/100008.jpg \
    --model_path models/baseline_resnet50_best.pth \
    --output_path data/processed/gradcam/100008_gradcam.png
```

### 5. Deploy Web Application

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

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Data pipeline tests only
pytest tests/test_data.py -v

# Model & training tests only
pytest tests/test_model.py -v
```

### Test Coverage

| Module | Tests | What's Verified |
|--------|-------|-----------------|
| `test_data.py` | 11 | FITS reading, NaN handling, RGB compositing, PNG generation |
| `test_model.py` | 19 | Output shape (batch, 37), sigmoid range [0,1], dataset behavior, config loading, model builders |

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
- [x] Unit test suite (30 tests)
- [x] Feature engineering module (SNR, concentration index)
- [ ] Vision Transformer (ViT) / ConvNeXt backbone upgrade
- [ ] Self-supervised pretraining with Masked Autoencoders (MAE)
- [ ] DVC integration for dataset versioning
- [ ] GitHub Actions CI/CD pipeline
- [ ] Grad-CAM visualization in Streamlit frontend

---

## License

This project is for educational and portfolio purposes.

---

<p align="center">
  Built with 🔭 for the intersection of deep learning and astrophysics.
</p>
