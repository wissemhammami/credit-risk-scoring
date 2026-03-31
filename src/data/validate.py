import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "application_train":    ["SK_ID_CURR", "TARGET"],
    "application_test":     ["SK_ID_CURR"],
    "bureau":               ["SK_ID_CURR", "SK_ID_BUREAU"],
    "previous_application": ["SK_ID_CURR", "SK_ID_PREV"],
}


def validate_raw_data(datasets: Dict[str, pd.DataFrame]) -> None:
    logger.info("Starting data validation...")

    for name, required_cols in REQUIRED_COLUMNS.items():
        if name not in datasets:
            raise ValueError(f"Missing dataset: {name}")

        df = datasets[name]

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{name}: missing columns {missing_cols}")

        if name == "application_train":
            dupes = df["SK_ID_CURR"].duplicated().sum()
            if dupes > 0:
                raise ValueError(f"{name}: {dupes} duplicate SK_ID_CURR found")
            invalid = ~df["TARGET"].isin([0, 1])
            if invalid.any():
                raise ValueError(f"{name}: TARGET contains values other than 0 and 1")

        logger.info("%s — rows=%d, cols=%d — OK", name, df.shape[0], df.shape[1])

    logger.info("Data validation passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from src.data.ingest import ingest_raw_data
    datasets = ingest_raw_data()
    validate_raw_data(datasets)