import json
import logging
from pathlib import Path

import pandas as pd

from src.models.evaluate import evaluate_model
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR       = BASE_DIR / "data" / "processed"
MODEL_REGISTRY_PATH = BASE_DIR / "models_artifacts"

ID_COL     = "SK_ID_CURR"
TARGET_COL = "TARGET"


def run_evaluation_pipeline():
    logger.info("===== EVALUATION PIPELINE START =====")

    eval_csv = PROCESSED_DIR / "eval_features_final.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_csv}")

    df = pd.read_csv(eval_csv)
    ids    = df[ID_COL] if ID_COL in df.columns else None
    X_eval = df.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in df.columns])
    y_eval = df[TARGET_COL]

    registry = ModelRegistry(base_path=MODEL_REGISTRY_PATH)
    model, model_folder = registry.load_latest_model()

    metrics = evaluate_model(model, X_eval, y_eval)

    with open(model_folder / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics saved → %s", model_folder / "metrics.json")

    y_prob = (
        model.predict_proba(X_eval)[:, 1]
        if hasattr(model, "predict_proba")
        else model.predict(X_eval)
    )

    pred_df = pd.DataFrame({"y_true": y_eval.values, "y_proba": y_prob})
    if ids is not None:
        pred_df.insert(0, ID_COL, ids.values)
    pred_df.to_csv(model_folder / "eval_predictions.csv", index=False)
    logger.info("Predictions saved → %s", model_folder / "eval_predictions.csv")

    logger.info("===== EVALUATION PIPELINE COMPLETE =====")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_evaluation_pipeline()