import logging
from typing import Tuple

import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.evaluate import evaluate_model
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest":       RandomForestClassifier,
    "xgboost":             XGBClassifier,
}

ID_COL     = "SK_ID_CURR"
TARGET_COL = "TARGET"


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def composite_score(metrics: dict) -> float:
    return (
        0.25 * metrics["ks"]
        + 0.25 * metrics["gini"]
        + 0.20 * metrics["auc"]
        - 0.30 * metrics["calibration_error"]
    )


def train_single_model(X_train, y_train, model_cfg: dict):
    model_type = model_cfg["model_type"]
    params = model_cfg.get("params", {})
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model type: {model_type}")
    model = SUPPORTED_MODELS[model_type](**params)
    logger.info("Training %s...", model_type)
    model.fit(X_train, y_train)
    return model


def train_all_models(
    features_csv: str,
    models_yaml: str,
    training_yaml: str,
) -> Tuple[str, object, dict]:

    df = pd.read_csv(features_csv)
    training_cfg = load_yaml(training_yaml)
    config = load_yaml(models_yaml)
    model_cfgs = config["models"]

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    X_train = df.drop(columns=[TARGET_COL])
    y_train = df[TARGET_COL]

    eval_csv = features_csv.replace("train_", "eval_")
    df_eval = pd.read_csv(eval_csv)
    if ID_COL in df_eval.columns:
        df_eval = df_eval.drop(columns=[ID_COL])
    X_eval = df_eval.drop(columns=[TARGET_COL])
    y_eval = df_eval[TARGET_COL]

    results = {}
    for cfg in model_cfgs:
        name = cfg["model_type"]
        model = train_single_model(X_train, y_train, cfg)
        metrics = evaluate_model(model, X_eval, y_eval)
        results[name] = {"model": model, "metrics": metrics}
        logger.info(
            "%s | KS=%.4f | AUC=%.4f | CalErr=%.4f | Composite=%.4f",
            name,
            metrics["ks"],
            metrics["auc"],
            metrics["calibration_error"],
            composite_score(metrics),
        )

    champion_name = max(
        results, key=lambda k: composite_score(results[k]["metrics"])
    )
    champion_model = results[champion_name]["model"]
    logger.info(
        "Champion: %s | Composite=%.4f",
        champion_name,
        composite_score(results[champion_name]["metrics"]),
    )

    logger.info("Calibrating champion model...")
    calibrated = CalibratedClassifierCV(
        champion_model, method="isotonic", cv=5
    )
    calibrated.fit(X_train, y_train)

    final_metrics = evaluate_model(calibrated, X_eval, y_eval)
    logger.info(
        "Post-calibration | CalErr=%.4f | AUC=%.4f",
        final_metrics["calibration_error"],
        final_metrics["auc"],
    )

    export_dir = training_cfg["paths"]["export_dir"]
    registry = ModelRegistry(base_path=export_dir)
    registry.register_model(
        calibrated,
        final_metrics,
        model_name=f"{champion_name}_champion",
    )

    return champion_name, calibrated, final_metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    train_all_models(
        features_csv="data/processed/train_features_final.csv",
        models_yaml="configs/model.yaml",
        training_yaml="configs/training.yaml",
    )