import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.ingest import ingest_raw_data
from src.data.validate import validate_raw_data
from src.features.transformers import CATEGORICAL_COLS, CreditFeatureEngineer, build_feature_pipeline

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.joblib"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    return bureau.groupby(ID_COL).agg(
        BUREAU_LOAN_COUNT=(ID_COL, "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        BUREAU_AMT_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_AMT_CREDIT_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_OVERDUE_COUNT=("CREDIT_DAY_OVERDUE", lambda x: (x > 0).sum()),
    ).reset_index()


def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    return prev.groupby(ID_COL).agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        PREV_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
    ).reset_index()


def run_feature_building(eval_size: float = 0.2, random_state: int = 42):
    logger.info("===== FEATURE BUILDING START =====")

    datasets = ingest_raw_data()
    validate_raw_data(datasets)

    train_df = datasets["application_train"].copy()
    test_df = datasets["application_test"].copy()

    bureau_agg = aggregate_bureau(datasets["bureau"])
    prev_agg = aggregate_previous(datasets["previous_application"])

    train_df = train_df.merge(bureau_agg, on=ID_COL, how="left")
    train_df = train_df.merge(prev_agg, on=ID_COL, how="left")
    test_df = test_df.merge(bureau_agg, on=ID_COL, how="left")
    test_df = test_df.merge(prev_agg, on=ID_COL, how="left")

    logger.info("After merge — train: %s | test: %s", train_df.shape, test_df.shape)

    y = train_df[TARGET_COL].copy()
    ids_train = train_df[ID_COL].copy()
    ids_test = test_df[ID_COL].copy()

    cat_cols = [c for c in CATEGORICAL_COLS if c in train_df.columns]
    exclude = [ID_COL, TARGET_COL]

    engineer = CreditFeatureEngineer()
    sample = engineer.fit_transform(train_df.drop(columns=exclude))
    num_cols = [
        c for c in sample.select_dtypes(include="number").columns
        if c not in cat_cols
    ]

    X = train_df.drop(columns=exclude)
    X_test = test_df.drop(columns=[ID_COL], errors="ignore")

    X_train, X_eval, y_train, y_eval, ids_tr, ids_ev = train_test_split(
        X, y, ids_train,
        test_size=eval_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info("Split — train: %d | eval: %d", len(X_train), len(X_eval))

    pipeline = build_feature_pipeline(numeric_cols=num_cols, categorical_cols=cat_cols)
    logger.info("Fitting pipeline on TRAIN only...")

    X_train_t = pipeline.fit_transform(X_train)
    X_eval_t = pipeline.transform(X_eval)
    X_test_t = pipeline.transform(X_test)

    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_t.shape[1])]

    joblib.dump(pipeline, PIPELINE_PATH)
    logger.info("Pipeline saved → %s", PIPELINE_PATH)

    def save(X_arr, y_s, ids_s, path, with_target=True):
        df_out = pd.DataFrame(X_arr, columns=feature_names)
        df_out.insert(0, ID_COL, ids_s.values)
        if with_target:
            df_out[TARGET_COL] = y_s.values
        df_out.to_csv(path, index=False)
        logger.info("Saved → %s | shape=%s", path, df_out.shape)

    save(X_train_t, y_train, ids_tr, PROCESSED_DIR / "train_features_final.csv")
    save(X_eval_t, y_eval, ids_ev, PROCESSED_DIR / "eval_features_final.csv")
    save(X_test_t, None, ids_test, PROCESSED_DIR / "test_features_final.csv", with_target=False)

    logger.info("===== FEATURE BUILDING COMPLETE =====")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_feature_building()