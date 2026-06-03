import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.build_model import GalaxyDataset  # noqa: E402
from src.models.model_factory import build_model_from_config as factory_build_model  # noqa: E402
from src.training.metrics import regression_metrics  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_transforms(config: dict, is_train: bool):
    preproc = config["preprocessing"]
    aug = config.get("augmentation", {})

    transform_list = [transforms.Resize((preproc["image_size"], preproc["image_size"]))]
    if is_train:
        if aug.get("horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())
        if aug.get("vertical_flip", False):
            transform_list.append(transforms.RandomVerticalFlip())
        if aug.get("random_rotation", 0) > 0:
            transform_list.append(transforms.RandomRotation(aug["random_rotation"]))
        if aug.get("color_jitter", False):
            transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preproc["normalize_mean"],
            std=preproc["normalize_std"],
        ),
    ])
    return transforms.Compose(transform_list)


def build_model_from_config(config: dict, num_outputs: int):
    return factory_build_model(config, num_outputs=num_outputs)


def get_image_dir(config: dict) -> str:
    data_cfg = config["data"]
    return data_cfg.get("image_dir", data_cfg.get("processed_dir", "data/processed/rgb_images"))


def build_optimizer(model, config: dict):
    train_cfg = config["training"]
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    name = train_cfg.get("optimizer", "adam").lower()

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_criterion(config: dict):
    name = config["training"].get("loss_function", "mse").lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "bce":
        return nn.BCELoss()
    raise ValueError(f"Unsupported loss function: {name}")


def split_indices(dataset_size: int, config: dict):
    train_ratio = config["data"].get("train_ratio", 0.8)
    num_samples = config["data"].get("num_samples", 0)
    seed = config["training"].get("seed", 42)

    indices = np.arange(dataset_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if num_samples and num_samples > 0:
        indices = indices[: min(num_samples, len(indices))]

    train_size = int(train_ratio * len(indices))
    return indices[:train_size].tolist(), indices[train_size:].tolist()


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_targets = []
    all_outputs = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, targets in tqdm(dataloader, leave=False):
            images = images.to(device)
            targets = targets.to(device)

            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_targets.append(targets.detach().cpu().numpy())
            all_outputs.append(outputs.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_outputs, axis=0)
    return total_loss / len(dataloader.dataset), y_true, y_pred


def train(config: dict):
    train_cfg = config["training"]
    set_seed(train_cfg.get("seed", 42))

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_dataset_full = GalaxyDataset(
        csv_file=config["data"]["csv_path"],
        img_dir=get_image_dir(config),
        transform=build_transforms(config, is_train=True),
    )
    val_dataset_full = GalaxyDataset(
        csv_file=config["data"]["csv_path"],
        img_dir=get_image_dir(config),
        transform=build_transforms(config, is_train=False),
    )

    if len(train_dataset_full) == 0:
        raise RuntimeError("No paired images and labels were found.")

    train_indices, val_indices = split_indices(len(train_dataset_full), config)
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model_from_config(config, len(train_dataset_full.target_cols)).to(device)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)

    use_wandb = False
    if config.get("wandb", {}).get("enabled", False):
        try:
            import wandb

            model_cfg = config.get("model", {})
            wandb.init(
                project=config["wandb"].get("project", "galaxy-classification-mlops"),
                name=config["wandb"].get("run_name", "resnet18-baseline"),
                config=config,
                tags=[
                    model_cfg.get("name", model_cfg.get("architecture", "model")),
                    "galaxy-zoo",
                    "morphology-regression",
                ],
            )
            use_wandb = True
        except ImportError:
            print("wandb is not installed; continuing without experiment tracking.")

    save_dir = config["checkpoint"].get("save_dir", "models")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, config["checkpoint"].get("best_model_name", "resnet18_baseline_best.pth"))

    best_val_loss = float("inf")
    patience = train_cfg.get("patience", 3)
    patience_counter = 0

    for epoch in range(train_cfg.get("epochs", 10)):
        print(f"Epoch {epoch + 1}/{train_cfg.get('epochs', 10)}")
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_true, y_pred = run_epoch(model, val_loader, criterion, device)
        metrics = regression_metrics(y_true, y_pred, train_dataset_full.target_cols)

        log_data = {
            "epoch": epoch + 1,
            "model_name": config["model"].get("name", config["model"].get("architecture", "model")),
            "learning_rate": train_cfg.get("learning_rate", 0.001),
            "batch_size": batch_size,
            "optimizer": train_cfg.get("optimizer", "adam"),
            "train_loss": train_loss,
            "valid_loss": val_loss,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        }
        for class_name, value in sorted(
            metrics["classwise_rmse"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]:
            log_data[f"classwise_rmse_top5/{class_name}"] = value
        print(
            "train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            "rmse={rmse:.6f} mae={mae:.6f}".format(**log_data)
        )

        if use_wandb:
            wandb.log(log_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if use_wandb:
        wandb.finish()

    return best_model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Week 1 ResNet18 baseline.")
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.no_wandb:
        config["wandb"]["enabled"] = False
    train(config)
