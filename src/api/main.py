import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.registry import ModelRegistry
from src.models.scorecard import create_scorecard

logger = logging.getLogger(__name__)

BASE_DIR            = Path(__file__).resolve().parent.parent.parent
MODEL_REGISTRY_PATH = BASE_DIR / "models_artifacts"
PIPELINE_PATH       = BASE_DIR / "models" / "feature_pipeline.joblib"

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts probability of default and returns credit score.",
    version="1.0.0",
)

try:
    feature_pipeline = joblib.load(PIPELINE_PATH)
    registry = ModelRegistry(base_path=MODEL_REGISTRY_PATH)
    model, _ = registry.load_latest_model()
    logger.info("Pipeline and model loaded successfully.")
except Exception as e:
    logger.error("Startup error: %s", e)
    feature_pipeline = None
    model = None


class ApplicantData(BaseModel):
    SK_ID_CURR:          int
    AMT_INCOME_TOTAL:    float
    AMT_CREDIT:          float
    AMT_ANNUITY:         float
    AMT_GOODS_PRICE:     float
    DAYS_BIRTH:          float
    DAYS_EMPLOYED:       float
    DAYS_REGISTRATION:   float
    DAYS_ID_PUBLISH:     float
    CODE_GENDER:         str
    NAME_CONTRACT_TYPE:  str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS:  str
    NAME_HOUSING_TYPE:   str
    NAME_INCOME_TYPE:    str
    EXT_SOURCE_1:        float = None
    EXT_SOURCE_2:        float = None
    EXT_SOURCE_3:        float = None


@app.get("/")
def root():
    return {"status": "Credit Risk Scoring API is running."}


@app.get("/health")
def health():
    return {
        "model_loaded":    model is not None,
        "pipeline_loaded": feature_pipeline is not None,
    }


@app.post("/predict")
def predict(applicant: ApplicantData):
    if model is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    data = pd.DataFrame([applicant.model_dump()])
    sk_id = data.pop("SK_ID_CURR")

    try:
        X_transformed = feature_pipeline.transform(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature transformation failed: {e}")

    y_prob = (
        model.predict_proba(X_transformed)[:, 1]
        if hasattr(model, "predict_proba")
        else model.predict(X_transformed)
    )

    scorecard = create_scorecard(
        ids=sk_id,
        predictions=pd.Series(y_prob),
    )

    return scorecard.to_dict(orient="records")[0]