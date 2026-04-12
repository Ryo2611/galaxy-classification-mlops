"""
build_features.py — 画像特徴量の計算とデータ拡張パイプライン

このモジュールは以下の機能を提供します:
  1. 画像品質メトリクスの算出（SNR, ピーク輝度、集中度）
  2. 天文画像に特化したデータ拡張パイプライン
  3. データセット全体の統計量計算（正規化パラメータ算出用）

Usage:
    # 単一画像の品質メトリクスを計算
    python src/features/build_features.py --image_path data/processed/rgb_images/12345.png

    # データセット全体の統計量を計算
    python src/features/build_features.py --compute_stats --img_dir data/processed/rgb_images
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import json


# =============================================================================
# 画像品質メトリクス
# =============================================================================

def compute_snr(image_array: np.ndarray) -> float:
    """
    画像の信号対雑音比 (Signal-to-Noise Ratio) を推定する。

    中心領域を「信号」、周辺領域を「ノイズ」として計算。
    銀河画像では中心に天体、端にバックグラウンドがある前提。

    Args:
        image_array: (H, W, 3) のRGB画像 (uint8 or float)

    Returns:
        SNR値（dB）
    """
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float64) / 255.0

    # グレースケール化
    gray = np.mean(image_array, axis=2)
    h, w = gray.shape

    # 中心25%を信号領域とする
    ch, cw = h // 4, w // 4
    signal_region = gray[ch:h - ch, cw:w - cw]

    # 周辺10%を背景（ノイズ）領域とする
    border = max(h // 10, 1)
    noise_region = np.concatenate([
        gray[:border, :].flatten(),
        gray[-border:, :].flatten(),
        gray[:, :border].flatten(),
        gray[:, -border:].flatten(),
    ])

    signal_mean = np.mean(signal_region)
    noise_std = np.std(noise_region)

    if noise_std < 1e-10:
        return float('inf')

    snr = 20 * np.log10(signal_mean / noise_std)
    return float(snr)


def compute_peak_brightness(image_array: np.ndarray) -> float:
    """画像のピーク輝度（最大チャネル平均値）を計算する"""
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float64) / 255.0
    return float(np.max(np.mean(image_array, axis=2)))


def compute_concentration_index(image_array: np.ndarray, inner_frac=0.2, outer_frac=0.8) -> float:
    """
    集中度指数（Concentration Index）を計算する。

    天文学で銀河の形態を定量化するために使われる特徴量。
    内側の光度と外側の光度の比率。
      - 高い値 → 中心に光が集中（楕円銀河的）
      - 低い値 → 光が広がっている（渦巻き銀河的）

    Args:
        image_array: (H, W, 3) のRGB画像
        inner_frac: 内側領域の比率
        outer_frac: 外側領域の比率

    Returns:
        集中度指数 (0〜1+)
    """
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float64) / 255.0

    gray = np.mean(image_array, axis=2)
    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # 各ピクセルの中心からの距離
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = np.sqrt(cy ** 2 + cx ** 2)

    # 内側・外側領域のマスク
    inner_mask = r <= (inner_frac * max_r)
    outer_mask = r <= (outer_frac * max_r)

    inner_flux = np.sum(gray[inner_mask])
    outer_flux = np.sum(gray[outer_mask])

    if outer_flux < 1e-10:
        return 0.0

    return float(inner_flux / outer_flux)


def compute_image_metrics(image_path: str) -> dict:
    """
    画像の全品質メトリクスを一度に計算する。

    Args:
        image_path: 画像ファイルのパス

    Returns:
        各メトリクスを含む辞書
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    return {
        "snr_db": round(compute_snr(arr), 2),
        "peak_brightness": round(compute_peak_brightness(arr), 4),
        "concentration_index": round(compute_concentration_index(arr), 4),
        "width": arr.shape[1],
        "height": arr.shape[0],
    }


# =============================================================================
# データセット統計量の計算
# =============================================================================

def compute_dataset_statistics(img_dir: str) -> dict:
    """
    データセット全体のチャネルごとの平均・標準偏差を計算する。

    ImageNet の正規化パラメータの代わりに、自前データセットの
    統計量を使いたい場合に利用。

    Args:
        img_dir: 画像ディレクトリのパス

    Returns:
        {"mean": [R, G, B], "std": [R, G, B], "num_images": int}
    """
    image_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not image_files:
        print(f"画像が見つかりません: {img_dir}")
        return {"mean": [0, 0, 0], "std": [1, 1, 1], "num_images": 0}

    # Welford のオンラインアルゴリズムでメモリ効率よく計算
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for fname in tqdm(image_files, desc="データセット統計量を計算中"):
        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        arr = np.array(img, dtype=np.float64) / 255.0

        pixel_sum += arr.sum(axis=(0, 1))
        pixel_sq_sum += (arr ** 2).sum(axis=(0, 1))
        total_pixels += arr.shape[0] * arr.shape[1]

    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sq_sum / total_pixels - mean ** 2)

    result = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_images": len(image_files),
        "total_pixels": int(total_pixels),
    }

    return result


# =============================================================================
# データ拡張パイプライン定義
# =============================================================================

def get_galaxy_augmentation_transforms(image_size: int = 224):
    """
    天文画像に特化したデータ拡張パイプラインを返す。

    銀河画像の物理的性質を考慮:
      - 回転不変（銀河には「上」がない）→ RandomRotation(360)
      - 反転不変 → 水平・垂直フリップ
      - 軽微な色変動 → ColorJitter（大きすぎると物理的に不自然）

    Returns:
        torchvision.transforms.Compose
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),  # 銀河は回転対称
        transforms.ColorJitter(
            brightness=0.05,    # 微弱な輝度変動
            contrast=0.05,
            saturation=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =============================================================================
# メイン（CLI）
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="画像特徴量の計算とデータセット統計")
    parser.add_argument("--image_path", type=str, default=None, help="分析する画像のパス")
    parser.add_argument("--img_dir", type=str, default=None, help="画像ディレクトリのパス")
    parser.add_argument("--compute_stats", action="store_true", help="データセット統計量を計算")
    parser.add_argument("--output", type=str, default=None, help="結果をJSONファイルに保存")
    args = parser.parse_args()

    if args.image_path:
        print(f"\n画像メトリクス: {args.image_path}")
        print("-" * 40)
        metrics = compute_image_metrics(args.image_path)
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\n結果を保存: {args.output}")

    if args.compute_stats and args.img_dir:
        print(f"\nデータセット統計量: {args.img_dir}")
        print("-" * 40)
        stats = compute_dataset_statistics(args.img_dir)
        print(f"  画像数: {stats['num_images']}")
        print(f"  チャネル平均 (R,G,B): {[round(m, 4) for m in stats['mean']]}")
        print(f"  チャネル標準偏差 (R,G,B): {[round(s, 4) for s in stats['std']]}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\n結果を保存: {args.output}")


if __name__ == "__main__":
    main()
