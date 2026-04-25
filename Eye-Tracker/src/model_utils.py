from functools import lru_cache

import numpy as np


def _as_2d(features):
    arr = np.asarray(features, dtype=np.float32)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr.reshape(1, -1)
    return arr, squeeze


@lru_cache(maxsize=None)
def _quadratic_indices(dim: int):
    return np.triu_indices(dim)


def transform_features(features, feature_mode: str = "linear"):
    arr, squeeze = _as_2d(features)
    mode = str(feature_mode or "linear").lower()

    if mode == "linear":
        basis = arr
    elif mode == "quadratic":
        ii, jj = _quadratic_indices(arr.shape[1])
        quad = arr[:, ii] * arr[:, jj]
        basis = np.hstack([arr, quad])
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    basis = basis.astype(np.float32)
    return basis[0] if squeeze else basis


def fit_ridge_model(X: np.ndarray, Y: np.ndarray, ridge_lambda: float = 6.0, feature_mode: str = "linear"):
    basis = transform_features(X, feature_mode=feature_mode)
    X_aug = np.hstack([basis, np.ones((basis.shape[0], 1), dtype=np.float32)])
    xtx = X_aug.T @ X_aug
    reg = np.eye(xtx.shape[0], dtype=np.float32) * float(ridge_lambda)
    reg[-1, -1] = 0.0
    xty = X_aug.T @ np.asarray(Y, dtype=np.float32)
    try:
        W = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(xtx + reg) @ xty
    return W.astype(np.float32)


def predict_points(features, W: np.ndarray, feature_mode: str = "linear"):
    basis = transform_features(features, feature_mode=feature_mode)
    squeeze = basis.ndim == 1
    if squeeze:
        basis = basis.reshape(1, -1)
    X_aug = np.hstack([basis, np.ones((basis.shape[0], 1), dtype=np.float32)])
    pred = X_aug @ np.asarray(W, dtype=np.float32)
    return pred[0] if squeeze else pred


def predict_xy(feature_vec: np.ndarray, W: np.ndarray, feature_mode: str = "linear"):
    pred = predict_points(feature_vec, W, feature_mode=feature_mode)
    return float(pred[0]), float(pred[1])


def fit_affine_correction(Y_pred: np.ndarray, Y_true: np.ndarray):
    pred, _ = _as_2d(Y_pred)
    true, _ = _as_2d(Y_true)
    A = np.hstack([pred, np.ones((pred.shape[0], 1), dtype=np.float32)])
    matrix, _, _, _ = np.linalg.lstsq(A, true, rcond=None)
    return {"matrix": matrix.astype(np.float32)}


def apply_calibration_correction(points, corr):
    arr, squeeze = _as_2d(points)

    if corr is None:
        out = arr
    elif isinstance(corr, dict) and len(corr) == 0:
        out = arr
    elif isinstance(corr, dict) and "matrix" in corr:
        matrix = np.asarray(corr["matrix"], dtype=np.float32)
        A = np.hstack([arr, np.ones((arr.shape[0], 1), dtype=np.float32)])
        out = A @ matrix
    elif isinstance(corr, (np.ndarray, list, tuple)):
        matrix = np.asarray(corr, dtype=np.float32)
        if matrix.shape != (3, 2):
            raise ValueError(f"Expected a 3x2 affine correction matrix, got {matrix.shape}.")
        A = np.hstack([arr, np.ones((arr.shape[0], 1), dtype=np.float32)])
        out = A @ matrix
    else:
        x = corr.get("x_a", 1.0) * arr[:, 0] + corr.get("x_b", 0.0)
        y = corr.get("y_a", 1.0) * arr[:, 1] + corr.get("y_b", 0.0)
        out = np.stack([x, y], axis=1).astype(np.float32)

    return out[0] if squeeze else out
