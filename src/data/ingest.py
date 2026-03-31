import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

RAW_FILES = {
    "application_train":     "application_train.csv",
    "application_test":      "application_test.csv",
    "bureau":                "bureau.csv",
    "bureau_balance":        "bureau_balance.csv",
    "credit_card_balance":   "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
    "pos_cash_balance":      "POS_CASH_balance.csv",
    "previous_application":  "previous_application.csv",
}


def ingest_raw_data() -> Dict[str, pd.DataFrame]:
    logger.info("Starting raw data ingestion...")
    datasets = {}
    for name, filename in RAW_FILES.items():
        path = RAW_DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        datasets[name] = pd.read_csv(path, encoding="utf-8")
        logger.info("Loaded %s | shape=%s", filename, datasets[name].shape)
    logger.info("Ingestion complete — %d datasets loaded.", len(datasets))
    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    data = ingest_raw_data()
    for name, df in data.items():
        logger.info("%s: %s", name, df.shape)