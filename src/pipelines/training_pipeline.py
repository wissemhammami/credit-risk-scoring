import logging

from src.features.build import run_feature_building
from src.models.train import train_all_models

logger = logging.getLogger(__name__)

FEATURES_CSV  = "data/processed/train_features_final.csv"
MODEL_YAML    = "configs/model.yaml"
TRAINING_YAML = "configs/training.yaml"


def run_training_pipeline():
    logger.info("===== TRAINING PIPELINE START =====")

    logger.info("Step 1/2: Building features...")
    run_feature_building()

    logger.info("Step 2/2: Training and registering models...")
    champion_name, _, champion_metrics = train_all_models(
        features_csv=FEATURES_CSV,
        models_yaml=MODEL_YAML,
        training_yaml=TRAINING_YAML,
    )

    logger.info("===== TRAINING PIPELINE COMPLETE =====")
    logger.info("Champion: %s", champion_name)
    for k, v in champion_metrics.items():
        logger.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_training_pipeline()