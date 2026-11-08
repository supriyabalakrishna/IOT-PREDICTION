from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
import uuid

import pandas as pd

from app.config import get_settings
from app.dataset import LABELS
from app.features import FEATURE_NAMES
from app.inference import process_feature_message

LOGGER = logging.getLogger(__name__)


def simulate_predictions(
    bundle: dict[str, object],
    *,
    machine_id: str = "terminal-simulator",
    count: int | None = 100,
    interval_seconds: float = 0.2,
) -> dict[str, object]:
    if count is not None and count < 1:
        raise ValueError("Simulation count must be at least one")
    if interval_seconds < 0:
        raise ValueError("Simulation interval cannot be negative")

    settings = get_settings()
    feature_path = settings.artifact_dir / "test_features.csv.gz"
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Held-out sensor features were not found at {feature_path}. "
            "Train and evaluate a model first with: python -m app.cli train"
        )
    frame = pd.read_csv(feature_path)
    required = {"recording_id", "window_index", "label", *FEATURE_NAMES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Held-out feature file is missing required columns: {missing}")
    frame = frame.sort_values(["window_index", "recording_id"], kind="stable")
    if frame.empty:
        raise ValueError("Held-out feature file contains no sensor readings")

    source_classes = sorted(str(value) for value in frame["label"].unique())
    unsupported = sorted(set(source_classes) - set(LABELS))
    if unsupported:
        raise ValueError(f"Held-out feature file contains unsupported source classes: {unsupported}")
    predictions = {label: 0 for label in LABELS}
    processed = 0
    sequence = 0
    while count is None or processed < count:
        for _, row in frame.iterrows():
            if count is not None and processed >= count:
                break
            payload = {
                "message_id": f"terminal-simulator:{uuid.uuid4()}",
                "machine_id": machine_id,
                "source": "replay",
                "schema_version": settings.feature_schema_version,
                "recording_id": str(row["recording_id"]),
                "window_index": int(row["window_index"]),
                "sequence_number": sequence,
                "event_timestamp": datetime.now(timezone.utc).isoformat(),
                "features": {name: float(row[name]) for name in FEATURE_NAMES},
            }
            prediction, _ = process_feature_message(bundle, payload)
            predicted_class = prediction.predicted_condition
            predictions[predicted_class] += 1
            processed += 1
            sequence += 1
            LOGGER.info(
                "Reading %d | source=%s | predicted=%s | decision=%s",
                processed,
                row["label"],
                predicted_class,
                prediction.decision_status,
            )
            if interval_seconds:
                time.sleep(interval_seconds)
    return {
        "processed": processed,
        "machine_id": machine_id,
        "model_version": bundle["model_version"],
        "source_classes": source_classes,
        "prediction_distribution": predictions,
    }
