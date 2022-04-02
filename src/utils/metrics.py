import numpy as np
from sklearn import metrics


def AUC(y_true: np.array, y_pred: np.array) -> float:
    return metrics.roc_auc_score(y_true, y_pred)


def MAE(y_true: np.array, y_pred: np.array) -> float:
    return metrics.mean_absolute_error(y_true, y_pred)


def PRAUC(y_true: np.array, y_pred: np.array) -> float:
    return metrics.average_precision_score(y_true, y_pred)


def RMSE(y_true: np.array, y_pred: np.array) -> float:
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5


def QWK(y_true: np.array, y_pred: np.array) -> float:
    return metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")
