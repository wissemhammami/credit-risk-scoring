import logging

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true, y_prob) -> float:
    return float(2 * roc_auc_score(y_true, y_prob) - 1)


def calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return float(np.mean(np.abs(prob_true - prob_pred)))


def evaluate_model(model, X_val, y_val) -> dict:
    logger.info("Evaluating model...")

    y_prob = (
        model.predict_proba(X_val)[:, 1]
        if hasattr(model, "predict_proba")
        else model.predict(X_val)
    )

    metrics = {
        "ks":                ks_statistic(y_val, y_prob),
        "gini":              gini_coefficient(y_val, y_prob),
        "auc":               float(roc_auc_score(y_val, y_prob)),
        "brier":             float(brier_score_loss(y_val, y_prob)),
        "calibration_error": calibration_error(y_val, y_prob),
    }

    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    return metrics