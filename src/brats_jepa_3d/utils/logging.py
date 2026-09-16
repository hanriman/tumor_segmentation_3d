import json
import logging
import sys
from pathlib import Path

import pandas as pd


def setup_logger(name: str = "brats_jepa_3d", log_file: str | Path | None = None) -> logging.Logger:
    """Configures console and file loggers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "brats_jepa_3d", log_file: str | Path | None = None) -> logging.Logger:
    """Compatibility alias for setup_logger."""
    return setup_logger(name, log_file)


class MetricTracker:
    """Accumulates epoch metrics and serializes to JSON and CSV."""

    def __init__(self):
        self.history: dict[str, list] = {}

    def update(self, metrics: dict[str, float]):
        for k, v in metrics.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))

    def get_latest(self) -> dict[str, float]:
        return {k: v[-1] for k, v in self.history.items() if v}

    def save_json(self, file_path: str | Path):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def save_csv(self, file_path: str | Path):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.history)
        df.to_csv(path, index=False)
