import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_features_final.csv"
TEST_PATH  = BASE_DIR / "data" / "processed" / "test_features_final.csv"
OUTPUT_PATH = BASE_DIR / "monitoring_reports" / "data_drift_report.csv"


def psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    bins = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        n_bins + 1,
    )
    exp_counts = np.histogram(expected, bins=bins)[0] / len(expected)
    act_counts = np.histogram(actual, bins=bins)[0] / len(actual)
    exp_counts = np.where(exp_counts == 0, 1e-6, exp_counts)
    act_counts = np.where(act_counts == 0, 1e-6, act_counts)
    return float(np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts)))


def detect_data_drift(train_df: pd.DataFrame, new_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    features = [
        c for c in train_df.select_dtypes(include="number").columns
        if c not in ["TARGET", "SK_ID_CURR"] and c in new_df.columns
    ]
    logger.info("Checking drift on %d features...", len(features))

    results = []
    for feat in features:
        ks_stat, p_val = ks_2samp(train_df[feat].dropna(), new_df[feat].dropna())
        psi_val = psi(train_df[feat].dropna(), new_df[feat].dropna())
        results.append({
            "feature":      feat,
            "ks_statistic": ks_stat,
            "p_value":      p_val,
            "psi":          psi_val,
            "ks_drift":     p_val < alpha,
            "psi_drift":    psi_val > 0.2,
        })

    return pd.DataFrame(results)


def main():
    logger.info("Starting drift monitoring...")
    train_df = pd.read_csv(TRAIN_PATH)
    new_df   = pd.read_csv(TEST_PATH)

    report = detect_data_drift(train_df, new_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    logger.info(
        "KS drift: %d features | PSI drift: %d features",
        report["ks_drift"].sum(),
        report["psi_drift"].sum(),
    )
    print(report.sort_values("psi", ascending=False).head(10))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()