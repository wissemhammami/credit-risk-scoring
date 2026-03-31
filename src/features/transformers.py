import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

CATEGORICAL_COLS = [
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "NAME_INCOME_TYPE",
    "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "NAME_TYPE_SUITE",
]


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):

    DAYS_EMPLOYED_ANOMALY = 365243

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "DAYS_EMPLOYED" in X.columns:
            X["DAYS_EMPLOYED_ANOMALY"] = (
                X["DAYS_EMPLOYED"] == self.DAYS_EMPLOYED_ANOMALY
            ).astype(int)
            X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(
                self.DAYS_EMPLOYED_ANOMALY, np.nan
            )

        if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(X.columns):
            X["ANNUITY_INCOME_PERC"] = X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]

        if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(X.columns):
            X["PAYMENT_RATE"] = X["AMT_ANNUITY"] / X["AMT_CREDIT"]

        if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(X.columns):
            X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]

        if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(X.columns):
            X["GOODS_CREDIT_RATIO"] = X["AMT_GOODS_PRICE"] / X["AMT_CREDIT"]

        if "DAYS_BIRTH" in X.columns:
            X["AGE_YEARS"] = -X["DAYS_BIRTH"] / 365

        if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(X.columns):
            X["EMPLOYED_TO_AGE_RATIO"] = X["DAYS_EMPLOYED"] / X["DAYS_BIRTH"]

        if "DAYS_REGISTRATION" in X.columns:
            X["YEARS_REGISTRATION"] = -X["DAYS_REGISTRATION"] / 365

        if "DAYS_ID_PUBLISH" in X.columns:
            X["YEARS_ID_PUBLISH"] = -X["DAYS_ID_PUBLISH"] / 365

        doc_cols = [c for c in X.columns if c.startswith("FLAG_DOCUMENT_")]
        if doc_cols:
            X["DOCUMENT_COUNT"] = X[doc_cols].sum(axis=1)

        ext_cols = [c for c in X.columns if c.startswith("EXT_SOURCE_")]
        if ext_cols:
            X["EXT_SOURCE_MEAN"] = X[ext_cols].mean(axis=1)
            X["EXT_SOURCE_STD"] = X[ext_cols].std(axis=1)
            X["EXT_SOURCE_MIN"] = X[ext_cols].min(axis=1)
            if "AMT_CREDIT" in X.columns:
                X["EXT_SOURCE_MEAN_x_CREDIT"] = X["EXT_SOURCE_MEAN"] * X["AMT_CREDIT"]

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        return X


def build_feature_pipeline(numeric_cols: list, categorical_cols: list = None) -> Pipeline:
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("domain_engineer", CreditFeatureEngineer()),
        ("preprocessor", preprocessor),
    ])