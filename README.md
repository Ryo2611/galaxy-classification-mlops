# 🌌 Galaxy Classification Portfolio
### *An End-to-End MLOps Pipeline for Astronomical Image Analysis*

---

## 🎯 Overview

This project demonstrates a production-grade machine learning pipeline to classify galaxy morphologies (e.g., Spiral, Elliptical) using deep learning. It moves far beyond standard Kaggle notebook analyses by integrating raw data acquisition from the Sloan Digital Sky Survey (SDSS), custom preprocessing, **Explainable AI (XAI)**, **MLOps** tracking via Weights & Biases, and a containerized web application deployment.

**Goal:** Predict the detailed morphology probabilities of galaxies and visually explain *why* the neural network arrived at those conclusions.

## ✨ Key Features & Technical Stack

*   **Data Engineering**: Fetch raw FITS files directly from SDSS using `astroquery` and synthesize high-fidelity Lupton RGB composites.
*   **Deep Learning**: Baseline ResNet-50 built in `PyTorch`, fine-tuned for continuous probability regression across multiple galaxy classes.
*   **Explainable AI (XAI)**: Integrated **Grad-CAM** to map the neural network's activation focus back onto the original galaxy images (e.g., verifying it's looking at spiral arms vs. background noise).
*   **MLOps**: Full experiment tracking of losses, gradients, and model checkpoints using **Weights & Biases (W&B)**.
*   **Deployment**: A **FastAPI** inference backend feeding a **Streamlit** user interface, all orchestrated seamlessly via **Docker Compose**.

## 📂 Project Structure

```text
├── data/
│   ├── raw/               # Downloaded FITS from SDSS
│   └── processed/         # Lupton RGB images and Grad-CAM outputs
├── src/
│   ├── data/
│   │   ├── make_dataset.py # astroquery SDSS fetching
│   │   └── preprocess.py   # FITS to RGB pipeline
│   ├── models/
│   │   └── build_model.py  # PyTorch ResNet-50 with W&B Early Stopping
│   └── visualization/
│       └── visualize_xai.py # Grad-CAM implementation
├── app/
│   ├── api/main.py        # FastAPI ML Model serving
│   └── frontend/app.py    # Streamlit User Interface
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## 🚀 How to Run

### 1. Model Training (Locally)
Install dependencies and build the dataset:
```bash
pip install -r requirements.txt

# 1. Download FITS images
python src/data/make_dataset.py --csv_path <path_to_labels.csv>

# 2. Process FITS -> RGB
python src/data/preprocess.py

# 3. Train Model (Sign up for a free Weights & Biases account first)
wandb login
python src/models/build_model.py --csv_path <path_to_labels.csv>
```

### 2. Run the Web Application (Docker)
Ensure Docker is installed on your system.
```bash
docker-compose up --build
```
*   **Frontend UI:** Navigate to `http://localhost:8501`
*   **FastAPI Docs:** Navigate to `http://localhost:8000/docs`

## 📊 Next Steps / Future Roadmap
- [ ] Upgrade backbone to Vision Transformer (ViT) or ConvNeXt.
- [ ] Implement Masked Autoencoders (MAE) for Self-Supervised Learning on unlabeled SDSS data.
- [ ] Set up DVC for versioning heavy Image Datasets.
