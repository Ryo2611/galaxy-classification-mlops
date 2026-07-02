"""
test_model.py — モデル構造・推論のユニットテスト

テスト対象:
  - ResNet-50 カスタムモデルの出力 shape
  - Sigmoid による出力値範囲 [0, 1] の検証
  - GalaxyDataset のダミーデータ動作
  - train.py のヘルパー関数（build_model, build_transforms 等）

Usage:
    pytest tests/test_model.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# Galaxy Zoo の 37 出力カラム定義
# =============================================================================

TARGET_COLS = [
    'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2',
    'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2',
    'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4',
    'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3',
    'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4',
    'Class8.5', 'Class8.6', 'Class8.7',
    'Class9.1', 'Class9.2', 'Class9.3',
    'Class10.1', 'Class10.2', 'Class10.3',
    'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6',
]

NUM_OUTPUTS = len(TARGET_COLS)  # 37


# =============================================================================
# フィクスチャ
# =============================================================================

@pytest.fixture
def galaxy_model():
    """テスト用のGalaxy分類モデル（ResNet-50ベース）を構築"""
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, NUM_OUTPUTS),
        nn.Sigmoid(),
    )
    model.eval()
    return model


@pytest.fixture
def dummy_input():
    """テスト用のダミー入力テンソル (batch=4, 3ch, 224x224)"""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def dummy_dataset_dir(tmp_path):
    """GalaxyDataset テスト用のダミー画像とCSVを生成"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    num_samples = 5
    galaxy_ids = [str(1000 + i) for i in range(num_samples)]

    # ダミー画像の生成
    for gid in galaxy_ids:
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(img_dir / f"{gid}.jpg"))

    # ダミーCSVの生成
    data = {"GalaxyID": galaxy_ids}
    for col in TARGET_COLS:
        data[col] = np.random.random(num_samples).tolist()

    df = pd.DataFrame(data)
    csv_path = tmp_path / "labels.csv"
    df.to_csv(str(csv_path), index=False)

    return str(csv_path), str(img_dir)


# =============================================================================
# モデルアーキテクチャのテスト
# =============================================================================

class TestModelArchitecture:
    """モデルの構造に関するテストスイート"""

    def test_output_shape(self, galaxy_model, dummy_input):
        """出力が (batch_size, 37) の shape を持つか"""
        with torch.no_grad():
            output = galaxy_model(dummy_input)

        assert output.shape == (4, NUM_OUTPUTS), \
            f"Expected shape (4, {NUM_OUTPUTS}), got {output.shape}"

    def test_output_range_sigmoid(self, galaxy_model, dummy_input):
        """Sigmoid により全出力が [0, 1] の範囲内か"""
        with torch.no_grad():
            output = galaxy_model(dummy_input)

        assert torch.all(output >= 0.0), "出力に負の値が含まれている"
        assert torch.all(output <= 1.0), "出力に1を超える値が含まれている"

    def test_single_sample_inference(self, galaxy_model):
        """バッチサイズ1でも正常に動作するか"""
        single_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = galaxy_model(single_input)

        assert output.shape == (1, NUM_OUTPUTS)

    def test_model_is_deterministic_in_eval(self, galaxy_model, dummy_input):
        """eval モードで同じ入力に対して同じ出力を返すか"""
        galaxy_model.eval()
        with torch.no_grad():
            output1 = galaxy_model(dummy_input)
            output2 = galaxy_model(dummy_input)

        assert torch.allclose(output1, output2), "eval モードで出力が非決定的"

    def test_final_layer_is_sigmoid(self, galaxy_model):
        """最終層が Sigmoid であることを確認"""
        fc = galaxy_model.fc
        # nn.Sequential の最後のモジュールが Sigmoid か
        last_module = list(fc.modules())[-1]
        assert isinstance(last_module, nn.Sigmoid), \
            f"最終層が Sigmoid ではない: {type(last_module)}"


# =============================================================================
# 損失関数のテスト
# =============================================================================

class TestLossFunction:
    """学習に使われる損失関数のテスト"""

    def test_mse_loss_computes(self, galaxy_model, dummy_input):
        """MSE Loss が正しく計算されるか"""
        targets = torch.rand(4, NUM_OUTPUTS)
        criterion = nn.MSELoss()

        with torch.no_grad():
            outputs = galaxy_model(dummy_input)

        loss = criterion(outputs, targets)

        assert loss.item() >= 0.0, "損失が負の値"
        assert not torch.isnan(loss), "損失がNaN"
        assert not torch.isinf(loss), "損失がInf"

    def test_perfect_prediction_gives_zero_loss(self):
        """完全一致の予測で損失が0になるか"""
        criterion = nn.MSELoss()
        predictions = torch.tensor([[0.5, 0.3, 0.8]])
        targets = torch.tensor([[0.5, 0.3, 0.8]])

        loss = criterion(predictions, targets)
        assert torch.isclose(loss, torch.tensor(0.0)), f"完全一致なのに損失が {loss.item()}"


# =============================================================================
# GalaxyDataset のテスト
# =============================================================================

class TestGalaxyDataset:
    """GalaxyDataset クラスのテスト"""

    def test_dataset_length(self, dummy_dataset_dir):
        """データセットの長さが画像数と一致するか"""
        from src.models.build_model import GalaxyDataset

        csv_path, img_dir = dummy_dataset_dir
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = GalaxyDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)
        assert len(dataset) == 5

    def test_getitem_returns_image_and_target(self, dummy_dataset_dir):
        """__getitem__ が (image_tensor, target_tensor) のタプルを返すか"""
        from src.models.build_model import GalaxyDataset

        csv_path, img_dir = dummy_dataset_dir
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = GalaxyDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)
        image, target = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert image.shape == (3, 224, 224), f"画像shapeが不正: {image.shape}"
        assert target.shape == (NUM_OUTPUTS,), f"ターゲットshapeが不正: {target.shape}"

    def test_target_values_are_float(self, dummy_dataset_dir):
        """ターゲット値がfloat型か"""
        from src.models.build_model import GalaxyDataset

        csv_path, img_dir = dummy_dataset_dir
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = GalaxyDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)
        _, target = dataset[0]

        assert target.dtype == torch.float32, f"ターゲットのdtypeが不正: {target.dtype}"


# =============================================================================
# train.py ヘルパー関数のテスト
# =============================================================================

class TestTrainHelpers:
    """train.py のヘルパー関数のテスト"""

    def test_build_transforms_train(self):
        """学習用変換パイプラインが構築できるか"""
        from src.train import build_transforms

        config = {
            "preprocessing": {
                "image_size": 224,
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
            "augmentation": {
                "horizontal_flip": True,
                "vertical_flip": True,
                "random_rotation": 15,
                "color_jitter": True,
            },
        }

        transform = build_transforms(config, is_train=True)
        assert transform is not None

        # ダミー画像に適用できるか
        dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = transform(dummy_img)
        assert result.shape == (3, 224, 224)

    def test_build_transforms_val(self):
        """検証用変換にはデータ拡張が含まれないか"""
        from src.train import build_transforms

        config = {
            "preprocessing": {
                "image_size": 224,
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
            "augmentation": {
                "horizontal_flip": True,
                "vertical_flip": True,
            },
        }

        transform = build_transforms(config, is_train=False)

        # 検証用には Flip が含まれない（Resize + ToTensor + Normalize のみ）
        transform_names = [type(t).__name__ for t in transform.transforms]
        assert "RandomHorizontalFlip" not in transform_names
        assert "RandomVerticalFlip" not in transform_names

    def test_build_model_resnet50(self):
        """ResNet-50 モデルが正しく構築されるか"""
        from src.train import build_model

        config = {
            "model": {
                "architecture": "resnet50",
                "pretrained": False,
                "dropout": 0.0,
            }
        }

        model = build_model(config, num_outputs=37)
        dummy_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 37)

    def test_build_model_resnet18(self):
        """ResNet-18 モデルが正しく構築されるか"""
        from src.train import build_model

        config = {
            "model": {
                "architecture": "resnet18",
                "pretrained": False,
                "dropout": 0.0,
            }
        }

        model = build_model(config, num_outputs=37)
        dummy_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 37)

    def test_build_model_with_dropout(self):
        """Dropout 付きモデルが正しく構築されるか"""
        from src.train import build_model

        config = {
            "model": {
                "architecture": "resnet50",
                "pretrained": False,
                "dropout": 0.5,
            }
        }

        model = build_model(config, num_outputs=37)

        # fc内にDropoutが含まれるか確認
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.fc.modules())
        assert has_dropout, "Dropout がモデルに含まれていない"

    def test_unsupported_architecture_raises(self):
        """サポートされていないアーキテクチャで ValueError が発生するか"""
        from src.train import build_model

        config = {
            "model": {
                "architecture": "vgg99_nonexistent",
                "pretrained": False,
                "dropout": 0.0,
            }
        }

        with pytest.raises(ValueError, match="サポートされていない"):
            build_model(config, num_outputs=37)

    def test_build_criterion_mse(self):
        """MSE 損失関数が構築されるか"""
        from src.train import build_criterion

        config = {"training": {"loss_function": "mse"}}
        criterion = build_criterion(config)
        assert isinstance(criterion, nn.MSELoss)

    def test_build_criterion_smooth_l1(self):
        """SmoothL1 損失関数が構築されるか"""
        from src.train import build_criterion

        config = {"training": {"loss_function": "smooth_l1"}}
        criterion = build_criterion(config)
        assert isinstance(criterion, nn.SmoothL1Loss)

    def test_load_config(self, tmp_path):
        """YAML設定ファイルが正しく読み込めるか"""
        from src.train import load_config

        config_content = "training:\n  epochs: 10\n  batch_size: 16\n"
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        config = load_config(str(config_path))
        assert config["training"]["epochs"] == 10
        assert config["training"]["batch_size"] == 16


class TestWeek1Baseline:
    """Week 1 baseline modules のテスト"""

    def test_build_resnet18_baseline(self):
        """ResNet18 baseline が37次元確率を返すか"""
        from src.models.resnet import build_resnet18

        model = build_resnet18(num_outputs=37, pretrained=False)
        model.eval()

        with torch.no_grad():
            output = model(torch.randn(1, 3, 224, 224))

        assert output.shape == (1, 37)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_regression_metrics(self):
        """RMSE/MAE/class-wise RMSE が計算できるか"""
        from src.training.metrics import regression_metrics

        y_true = np.array([[0.0, 1.0], [1.0, 0.0]])
        y_pred = np.array([[0.0, 0.5], [0.5, 0.0]])
        metrics = regression_metrics(y_true, y_pred, target_names=["a", "b"])

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert set(metrics["classwise_rmse"]) == {"a", "b"}

    def test_split_indices_is_reproducible(self):
        """固定seedで同じtrain/validation splitになるか"""
        from src.training.train import split_indices

        config = {"data": {"train_ratio": 0.8, "num_samples": 0}, "training": {"seed": 42}}
        first = split_indices(10, config)
        second = split_indices(10, config)

        assert first == second
        assert len(first[0]) == 8
        assert len(first[1]) == 2


class TestWeek2Modules:
    """Week 2 model comparison / error analysis modules のテスト"""

    def test_model_factory_builds_efficientnet(self):
        """model_factory から EfficientNet-B0 を構築できるか"""
        from src.models.model_factory import build_model_from_config

        config = {
            "model": {
                "name": "efficientnet_b0",
                "pretrained": False,
                "dropout": 0.2,
            }
        }
        model = build_model_from_config(config, num_outputs=37)

        assert isinstance(model.classifier[-1], nn.Sigmoid)

    def test_model_factory_builds_convnext(self):
        """model_factory から ConvNeXt-Tiny を構築できるか"""
        from src.models.model_factory import build_model_from_config

        config = {
            "model": {
                "name": "convnext_tiny",
                "pretrained": False,
                "dropout": 0.2,
            },
            "preprocessing": {"image_size": 224},
        }
        model = build_model_from_config(config, num_outputs=37)

        assert isinstance(model.classifier[-1][-1], nn.Sigmoid)

    def test_model_factory_builds_vit(self):
        """model_factory から ViT-B/16 を構築できるか"""
        from src.models.model_factory import build_model_from_config

        config = {
            "model": {
                "name": "vit_b_16",
                "pretrained": False,
                "dropout": 0.1,
            },
            "preprocessing": {"image_size": 224},
        }
        model = build_model_from_config(config, num_outputs=37)

        assert isinstance(model.heads.head[-1], nn.Sigmoid)

    def test_mae_forward_loss(self):
        """MAE が画像パッチをマスクして再構成 loss を返すか"""
        from src.models.mae import MaskedAutoencoderViT

        model = MaskedAutoencoderViT(
            image_size=32,
            patch_size=8,
            embed_dim=32,
            encoder_depth=1,
            decoder_depth=1,
            num_heads=4,
            mask_ratio=0.5,
        )
        loss, pred_patches, mask = model(torch.rand(2, 3, 32, 32))

        assert loss.item() >= 0
        assert pred_patches.shape == (2, 16, 3 * 8 * 8)
        assert mask.shape == (2, 16)

    def test_model_factory_rejects_unknown_model(self):
        """未対応モデル名で ValueError が出るか"""
        from src.models.model_factory import build_model_from_name

        with pytest.raises(ValueError, match="Unsupported model"):
            build_model_from_name("unknown_cnn", num_outputs=37)

    def test_gradcam_target_layer_for_resnet(self):
        """ResNet系のGrad-CAM target layer が取得できるか"""
        from src.explainability.gradcam import get_target_layers
        from src.models.resnet import build_resnet18

        model = build_resnet18(num_outputs=37, pretrained=False)
        layers = get_target_layers(model, "resnet18")

        assert layers == [model.layer4[-1]]

    def test_error_analysis_summary(self, tmp_path):
        """予測CSVから誤差分析サマリを生成できるか"""
        from src.training.error_analysis import summarize_errors

        label_path = tmp_path / "labels.csv"
        pred_path = tmp_path / "predictions.csv"
        output_path = tmp_path / "summary.json"

        label_df = pd.DataFrame({
            "GalaxyID": ["1", "2"],
            "Class1.1": [0.0, 1.0],
            "Class1.2": [1.0, 0.0],
        })
        pred_df = pd.DataFrame({
            "Class1.1": [0.1, 0.8],
            "Class1.2": [0.9, 0.2],
        })
        label_df.to_csv(label_path, index=False)
        pred_df.to_csv(pred_path, index=False)

        summary = summarize_errors(str(label_path), str(pred_path), str(output_path))

        assert output_path.exists()
        assert summary["overall_rmse"] > 0
        assert len(summary["hardest_samples"]) == 2
