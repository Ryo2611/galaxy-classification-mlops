import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.build_model import GalaxyDataset  # noqa: E402
from src.training.metrics import regression_metrics  # noqa: E402
from src.training.train import build_model_from_config, build_transforms, get_image_dir, split_indices  # noqa: E402


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            predictions.append(outputs.cpu().numpy())
            targets.append(labels.numpy())

    return np.concatenate(targets, axis=0), np.concatenate(predictions, axis=0)


def save_prediction_distribution(y_true, y_pred, output_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(y_true.flatten(), bins=40, alpha=0.55, label="Ground truth")
    plt.hist(y_pred.flatten(), bins=40, alpha=0.55, label="Prediction")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_truth_vs_prediction(y_true, y_pred, output_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true.flatten(), y_pred.flatten(), s=4, alpha=0.25)
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.xlabel("Ground truth")
    plt.ylabel("Prediction")
    plt.title("Ground Truth vs Prediction")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate(config_path, checkpoint_path=None, output_dir=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config["checkpoint"]["save_dir"],
            config["checkpoint"]["best_model_name"],
        )
    if not os.path.exists(checkpoint_path):
        model_name = config["model"].get("name", config["model"].get("architecture", "model"))
        train_command = f"python src/training/train.py --config {config_path} --no_wandb"
        existing_checkpoints = sorted(
            path for path in os.listdir(config["checkpoint"].get("save_dir", "models"))
            if path.endswith(".pth")
        ) if os.path.isdir(config["checkpoint"].get("save_dir", "models")) else []
        existing_text = ", ".join(existing_checkpoints) if existing_checkpoints else "none"
        raise FileNotFoundError(
            f"Checkpoint for {model_name} was not found: {checkpoint_path}\n"
            f"Train it first with:\n  {train_command}\n"
            f"Existing checkpoints in models/: {existing_text}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    transform = build_transforms(config, is_train=False)
    dataset = GalaxyDataset(
        csv_file=config["data"]["csv_path"],
        img_dir=get_image_dir(config),
        transform=transform,
    )

    _, val_indices = split_indices(len(dataset), config)
    val_dataset = Subset(dataset, val_indices)
    dataloader = DataLoader(
        val_dataset,
        batch_size=config["training"].get("batch_size", 32),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
    )

    model = build_model_from_config(config, len(dataset.target_cols)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    y_true, y_pred = predict(model, dataloader, device)
    metrics = regression_metrics(y_true, y_pred, dataset.target_cols)

    output_dir = output_dir or config["evaluation"].get("output_dir", "reports/figures")
    os.makedirs(output_dir, exist_ok=True)

    model_name = config["model"].get("name", config["model"].get("architecture", "model"))
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    pred_path = config["evaluation"].get("prediction_csv", "reports/resnet18_predictions.csv")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pd.DataFrame(y_pred, columns=dataset.target_cols).to_csv(pred_path, index=False)

    save_prediction_distribution(
        y_true,
        y_pred,
        os.path.join(output_dir, "prediction_distribution.png"),
    )
    save_truth_vs_prediction(
        y_true,
        y_pred,
        os.path.join(output_dir, "truth_vs_prediction.png"),
    )

    print(json.dumps({k: v for k, v in metrics.items() if k != "classwise_rmse"}, indent=2))
    print(f"Saved metrics to {metrics_path}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Galaxy Zoo model.")
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        evaluate(args.config, args.checkpoint, args.output_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
