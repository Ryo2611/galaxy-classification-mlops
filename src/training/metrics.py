import numpy as np


def mean_squared_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def classwise_rmse(y_true, y_pred, target_names=None) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    values = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))

    if target_names is None:
        target_names = [f"class_{idx}" for idx in range(values.shape[0])]
    return {name: float(value) for name, value in zip(target_names, values)}


def regression_metrics(y_true, y_pred, target_names=None) -> dict:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "classwise_rmse": classwise_rmse(y_true, y_pred, target_names),
    }
