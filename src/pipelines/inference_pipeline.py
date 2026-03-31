import logging
from pathlib import Path

import pandas as pd

from src.models.registry import ModelRegistry
from src.models.scorecard import create_scorecard

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR       = BASE_DIR / "data" / "processed"
MODEL_REGISTRY_PATH = BASE_DIR / "models_artifacts"

ID_COL = "SK_ID_CURR"


def run_inference_pipeline():
    logger.info("===== INFERENCE PIPELINE START =====")

    test_csv = PROCESSED_DIR / "test_features_final.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test file not found: {test_csv}")

    df_test = pd.read_csv(test_csv)
    ids    = df_test[ID_COL] if ID_COL in df_test.columns else pd.Series(range(len(df_test)))
    X_test = df_test.drop(columns=[ID_COL], errors="ignore")

    registry = ModelRegistry(base_path=MODEL_REGISTRY_PATH)
    model, model_folder = registry.load_latest_model()

    y_pred = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.predict(X_test)
    )

    pred_df = pd.DataFrame({ID_COL: ids.values, "TARGET_PROBA": y_pred})
    pred_df.to_csv(model_folder / "predictions.csv", index=False)
    logger.info("Predictions saved → %s", model_folder / "predictions.csv")

    scorecard_df = create_scorecard(
        ids=ids.reset_index(drop=True),
        predictions=pd.Series(y_pred),
    )
    scorecard_df.to_csv(model_folder / "scorecard.csv", index=False)
    logger.info("Scorecard saved → %s", model_folder / "scorecard.csv")

    logger.info("===== INFERENCE PIPELINE COMPLETE =====")
    return scorecard_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_inference_pipeline()