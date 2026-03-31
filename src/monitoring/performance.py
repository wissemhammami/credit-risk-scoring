import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.models.evaluate import calibration_error, gini_coefficient, ks_statistic
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = BASE_DIR / "models_artifacts"


def monitor_model_performance(y_true: pd.Series, y_prob: pd.Series) -> dict:
    logger.info("Computing performance metrics...")

    current = {
        "auc":               float(roc_auc_score(y_true, y_prob)),
        "ks":                ks_statistic(y_true, y_prob),
        "gini":              gini_coefficient(y_true, y_prob),
        "brier":             float(brier_score_loss(y_true, y_prob)),
        "calibration_error": calibration_error(y_true, y_prob),
        "n_samples":         int(len(y_true)),
    }

    registry = ModelRegistry(base_path=MODEL_REGISTRY_PATH)
    _, model_folder = registry.load_latest_model()
    metrics_path = model_folder / "metrics.json"

    if metrics_path.exists():
        with open(metrics_path) as f:
            baseline = json.load(f)
        logger.info(
            "Baseline AUC=%.4f | Current AUC=%.4f",
            baseline.get("auc", 0),
            current["auc"],
        )
        if current["auc"] < baseline.get("auc", 0) - 0.02:
            logger.warning("Model degradation detected — AUC dropped > 0.02")

    return current