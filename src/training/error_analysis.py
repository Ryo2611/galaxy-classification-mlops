import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.training.metrics import classwise_rmse, mean_absolute_error, root_mean_squared_error  # noqa: E402


def load_ground_truth(label_csv: str, target_cols=None):
    labels = pd.read_csv(label_csv)
    if target_cols is None:
        target_cols = [col for col in labels.columns if col != "GalaxyID"]
    return labels, target_cols


def summarize_errors(label_csv: str, prediction_csv: str, output_path: str):
    labels, target_cols = load_ground_truth(label_csv)
    predictions = pd.read_csv(prediction_csv)

    if len(predictions) > len(labels):
        raise ValueError("Prediction CSV has more rows than the label CSV.")

    y_true = labels[target_cols].iloc[: len(predictions)].to_numpy(dtype=np.float64)
    y_pred = predictions[target_cols].to_numpy(dtype=np.float64)
    abs_error = np.abs(y_true - y_pred)
    signed_error = y_pred - y_true
    sample_mae = abs_error.mean(axis=1)

    class_rmse = classwise_rmse(y_true, y_pred, target_cols)
    class_bias = {
        col: float(signed_error[:, idx].mean())
        for idx, col in enumerate(target_cols)
    }

    hardest_indices = np.argsort(sample_mae)[::-1][:10]
    hardest_samples = []
    for idx in hardest_indices:
        galaxy_id = labels.iloc[idx].get("GalaxyID", idx)
        hardest_samples.append({
            "rank": len(hardest_samples) + 1,
            "row_index": int(idx),
            "galaxy_id": str(galaxy_id),
            "mae": float(sample_mae[idx]),
        })

    summary = {
        "overall_rmse": root_mean_squared_error(y_true, y_pred),
        "overall_mae": mean_absolute_error(y_true, y_pred),
        "top_classwise_rmse": dict(
            sorted(class_rmse.items(), key=lambda item: item[1], reverse=True)[:10]
        ),
        "most_overpredicted": dict(
            sorted(class_bias.items(), key=lambda item: item[1], reverse=True)[:10]
        ),
        "most_underpredicted": dict(
            sorted(class_bias.items(), key=lambda item: item[1])[:10]
        ),
        "hardest_samples": hardest_samples,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Galaxy Zoo prediction errors.")
    parser.add_argument("--labels", default="data/external/training_solutions_rev1.csv")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="reports/error_analysis_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = summarize_errors(args.labels, args.predictions, args.output)
    print(json.dumps(result, indent=2))
