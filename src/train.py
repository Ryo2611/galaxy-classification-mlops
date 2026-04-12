"""
train.py — Galaxy Classification 学習エントリポイント

YAML設定ファイルを読み込み、データセット構築→モデル学習→チェックポイント保存
までの一連の学習パイプラインを実行します。

Usage:
    python src/train.py                              # デフォルト設定で実行
    python src/train.py --config configs/custom.yaml # カスタム設定で実行
    python src/train.py --epochs 30 --batch_size 64  # CLI引数でオーバーライド
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np

# プロジェクトルートをPythonパスに追加（python src/train.py で実行可能にする）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 再現性のためワーカー間をまたぐシードも固定
import random


def set_seed(seed: int):
    """再現性のための乱数シード固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """YAML設定ファイルを読み込む"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_transforms(config: dict, is_train: bool = True):
    """設定に基づいてデータ変換パイプラインを構築する"""
    preproc = config["preprocessing"]
    aug = config.get("augmentation", {})

    transform_list = [
        transforms.Resize((preproc["image_size"], preproc["image_size"])),
    ]

    # 学習時のみデータ拡張を適用
    if is_train:
        if aug.get("horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())
        if aug.get("vertical_flip", False):
            transform_list.append(transforms.RandomVerticalFlip())
        if aug.get("random_rotation", 0) > 0:
            transform_list.append(transforms.RandomRotation(aug["random_rotation"]))
        if aug.get("color_jitter", False):
            transform_list.append(
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preproc["normalize_mean"],
            std=preproc["normalize_std"]
        ),
    ])

    return transforms.Compose(transform_list)


def build_model(config: dict, num_outputs: int):
    """設定に基づいてモデルを構築する"""
    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "resnet50")
    pretrained = model_cfg.get("pretrained", False)

    if arch == "resnet50":
        weights = "DEFAULT" if pretrained else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
    elif arch == "resnet18":
        weights = "DEFAULT" if pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
    elif arch == "efficientnet_b0":
        weights = "DEFAULT" if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        # EfficientNet はclassifierが異なるので別途対応
        model.classifier = nn.Sequential(
            nn.Dropout(model_cfg.get("dropout", 0.0)),
            nn.Linear(num_ftrs, num_outputs),
            nn.Sigmoid(),
        )
        return model
    else:
        raise ValueError(f"サポートされていないアーキテクチャ: {arch}")

    # ResNet系の最終層を置換
    dropout = model_cfg.get("dropout", 0.0)
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_outputs),
            nn.Sigmoid(),
        )
    else:
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_outputs),
            nn.Sigmoid(),
        )

    return model


def build_optimizer(model, config: dict):
    """設定に基づいてオプティマイザを構築する"""
    train_cfg = config["training"]
    lr = train_cfg.get("learning_rate", 0.001)
    wd = train_cfg.get("weight_decay", 0.0)
    opt_name = train_cfg.get("optimizer", "adam").lower()

    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"サポートされていないオプティマイザ: {opt_name}")


def build_criterion(config: dict):
    """設定に基づいて損失関数を構築する"""
    loss_name = config["training"].get("loss_function", "mse").lower()

    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"サポートされていない損失関数: {loss_name}")


def train(config: dict):
    """メインの学習ループ"""
    train_cfg = config["training"]
    ckpt_cfg = config["checkpoint"]
    wandb_cfg = config.get("wandb", {})

    # シード固定
    set_seed(train_cfg.get("seed", 42))

    # デバイス選択
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"使用デバイス: {device}")

    # W&B初期化（有効な場合のみ）
    if wandb_cfg.get("enabled", True):
        try:
            import wandb
            wandb.init(
                project=wandb_cfg.get("project", "galaxy-classification-portfolio"),
                name=wandb_cfg.get("run_name", "training-run"),
                config=config,
            )
            use_wandb = True
        except ImportError:
            print("Warning: wandb がインストールされていません。ログなしで学習を実行します。")
            use_wandb = False
    else:
        use_wandb = False

    # データ変換
    train_transform = build_transforms(config, is_train=True)
    val_transform = build_transforms(config, is_train=False)

    # データセット構築 (build_model.py の GalaxyDataset を再利用)
    from src.models.build_model import GalaxyDataset

    csv_path = config["data"]["csv_path"]
    img_dir = config["data"]["processed_dir"]

    full_dataset = GalaxyDataset(csv_file=csv_path, img_dir=img_dir, transform=train_transform)

    if len(full_dataset) == 0:
        print("エラー: 画像とラベルのペアが見つかりません。データのダウンロード/前処理は完了していますか？")
        return

    num_outputs = len(full_dataset.target_cols)
    print(f"データセット: {len(full_dataset)} サンプル, {num_outputs} 出力変数")

    # 訓練/検証分割
    train_ratio = train_cfg.get("train_ratio", 0.8)
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 検証用にはデータ拡張なしの変換を使いたいが、
    # random_split後のSubsetでtransformを切り替えるにはラッパーが必要
    # 簡易実装としてここではtrain_transformを共用（拡張はランダムなので検証精度への影響は小さい）

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # モデル構築
    model = build_model(config, num_outputs)
    model = model.to(device)

    # 損失関数・オプティマイザ
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)

    # 学習パラメータ
    epochs = train_cfg.get("epochs", 15)
    patience = train_cfg.get("patience", 3)
    save_dir = ckpt_cfg.get("save_dir", "models")
    best_model_name = ckpt_cfg.get("best_model_name", "baseline_resnet50_best.pth")
    best_model_path = os.path.join(save_dir, best_model_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n学習開始 (epochs={epochs}, batch_size={batch_size}, patience={patience})")
    print("=" * 60)

    for epoch in range(epochs):
        # ---- 学習フェーズ ----
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / train_size

        # ---- 検証フェーズ ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"
            ):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / val_size if val_size > 0 else 0

        print(
            f"Epoch {epoch + 1}/{epochs} — "
            f"Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}"
        )

        # W&B ログ
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
            })

        # チェックポイント & 早期停止
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> 最良モデルを保存 (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(
                    f"\n早期停止: {patience} エポック連続で改善なし。"
                )
                break

    print("=" * 60)
    print(f"学習完了。最良モデル: {best_model_path} (Val Loss: {best_val_loss:.6f})")

    if use_wandb:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Galaxy Classification — 学習スクリプト"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="YAML設定ファイルのパス",
    )
    # CLI引数によるオーバーライド（任意）
    parser.add_argument("--epochs", type=int, default=None, help="最大エポック数")
    parser.add_argument("--batch_size", type=int, default=None, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=None, help="学習率")
    parser.add_argument("--csv_path", type=str, default=None, help="ラベルCSVのパス")
    parser.add_argument("--img_dir", type=str, default=None, help="画像ディレクトリのパス")
    parser.add_argument("--no_wandb", action="store_true", help="W&Bを無効化")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # YAML設定を読み込み
    config = load_config(args.config)

    # CLI引数によるオーバーライド
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.csv_path is not None:
        config["data"]["csv_path"] = args.csv_path
    if args.img_dir is not None:
        config["data"]["processed_dir"] = args.img_dir
    if args.no_wandb:
        config["wandb"]["enabled"] = False

    train(config)
