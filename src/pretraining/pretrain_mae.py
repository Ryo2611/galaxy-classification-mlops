import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.mae import build_mae_vit  # noqa: E402


class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir: str, transform=None, extensions=(".jpg", ".jpeg", ".png")):
        self.image_paths = sorted(
            path
            for path in Path(image_dir).iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_transform(config: dict):
    preproc = config["preprocessing"]
    return transforms.Compose([
        transforms.Resize((preproc["image_size"], preproc["image_size"])),
        transforms.ToTensor(),
    ])


def select_subset(dataset: Dataset, config: dict):
    num_samples = config["data"].get("num_samples", 0)
    if not num_samples or num_samples <= 0 or num_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(config["training"].get("seed", 42))
    indices = rng.choice(len(dataset), size=num_samples, replace=False).tolist()
    return Subset(dataset, indices)


def resolve_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def save_checkpoint(model, optimizer, epoch: int, loss: float, config: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        output_path,
    )


def save_encoder_checkpoint(model, config: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "patch_embed": model.patch_embed.state_dict(),
            "encoder": model.encoder.state_dict(),
            "pos_embed": model.pos_embed.detach().cpu(),
            "config": config,
        },
        output_path,
    )


def pretrain(config: dict):
    train_cfg = config["training"]
    set_seed(train_cfg.get("seed", 42))
    device = resolve_device()
    print(f"Using device: {device}")

    dataset = ImageOnlyDataset(config["data"]["image_dir"], transform=build_transform(config))
    dataset = select_subset(dataset, config)
    if len(dataset) == 0:
        raise RuntimeError("No pretraining images were found.")

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )
    model = build_mae_vit(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 0.00015),
        weight_decay=train_cfg.get("weight_decay", 0.05),
    )

    use_wandb = False
    if config.get("wandb", {}).get("enabled", False):
        try:
            import wandb

            wandb.init(
                project=config["wandb"].get("project", "galaxy-classification-mlops"),
                name=config["wandb"].get("run_name", "mae-pretrain"),
                config=config,
                tags=["mae", "self-supervised", "galaxy-zoo"],
            )
            use_wandb = True
        except ImportError:
            print("wandb is not installed; continuing without experiment tracking.")

    best_loss = float("inf")
    best_path = config["checkpoint"].get("best_model_path", "models/mae_vit_best.pth")
    encoder_path = config["checkpoint"].get("encoder_path", "models/mae_vit_encoder.pth")
    for epoch in range(train_cfg.get("epochs", 20)):
        model.train()
        running_loss = 0.0
        for images in tqdm(dataloader, leave=False):
            images = images.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        log_data = {"epoch": epoch + 1, "mae_loss": epoch_loss}
        print(f"Epoch {epoch + 1}: mae_loss={epoch_loss:.6f}")
        if use_wandb:
            wandb.log(log_data)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, epoch + 1, best_loss, config, best_path)
            save_encoder_checkpoint(model, config, encoder_path)
            print(f"Saved MAE checkpoint to {best_path}")

    if use_wandb:
        wandb.finish()
    return best_path


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a lightweight MAE on galaxy images.")
    parser.add_argument("--config", default="configs/mae_pretrain.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False
    pretrain(cfg)
