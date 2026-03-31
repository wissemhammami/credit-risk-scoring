import json
import logging
from datetime import datetime
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


class ModelRegistry:

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def register_model(self, model, metrics: dict, model_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.base_path / f"{model_name}_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, folder / "model.pkl")
        logger.info("Model saved → %s", folder / "model.pkl")

        with open(folder / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics saved → %s", folder / "metrics.json")

        return folder

    def load_latest_model(self):
        folders = sorted(
            [f for f in self.base_path.iterdir() if f.is_dir()],
            key=lambda x: x.stat().st_mtime,  # sort by actual modification time
            reverse=True,
        )
        if not folders:
            raise FileNotFoundError(f"No models found in: {self.base_path}")

        model = joblib.load(folders[0] / "model.pkl")
        logger.info("Loaded latest model: %s", folders[0].name)
        return model, folders[0]