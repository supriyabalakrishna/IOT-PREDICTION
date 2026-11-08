from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pydantic import ValidationError

from app.config import get_settings
from app.dataset import LABELS
from app.features import FEATURE_NAMES
from app.reliability import (
    apply_temperature,
    explain_prediction,
    model_probability_classes,
    out_of_distribution,
    population_drift,
    sensor_quality,
)
from app.schemas import FeatureMessage
from app.storage import AppEvent, Prediction, SessionLocal

LOGGER = logging.getLogger(__name__)


class MessageRejected(ValueError):
    pass


def store_event(event_type: str, message: str, severity: str = "info") -> None:
    db = SessionLocal()
    try:
        db.add(AppEvent(event_type=event_type, message=message, severity=severity))
        db.commit()
    except Exception:
        db.rollback()
        LOGGER.exception("Could not persist system event %s", event_type)
        raise
    finally:
        db.close()


def explain_stored_prediction(bundle: dict[str, object], prediction: Prediction) -> dict[str, object] | None:
    if (
        prediction.model_version != bundle.get("model_version")
        or bundle.get("reliability_reference") is None
        or bundle.get("calibration_temperature") is None
    ):
        return None
    features = prediction.features
    if set(features) != set(FEATURE_NAMES):
        raise ValueError(f"Stored prediction {prediction.id} has an incompatible feature schema")
    row = pd.DataFrame(
        [[features[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )
    model = bundle["model"]
    class_names = list(bundle["class_names"])
    probability_classes = model_probability_classes(model, class_names)
    raw_probabilities = np.asarray(model.predict_proba(row), dtype=float)
    if raw_probabilities.shape != (1, len(probability_classes)):
        raise RuntimeError("Loaded model returned an invalid probability vector for stored prediction")
    temperature = float(bundle["calibration_temperature"])
    calibrated = apply_temperature(raw_probabilities, temperature)
    explanation = explain_prediction(
        model,
        row,
        calibrated,
        probability_classes,
        bundle["reliability_reference"],
        temperature,
    )
    prediction.explanation_json = json.dumps(explanation, separators=(",", ":"))
    if not prediction.calibrated_probabilities:
        prediction.calibrated_probabilities_json = json.dumps(
            {label: float(value) for label, value in zip(probability_classes, calibrated[0])},
            separators=(",", ":"),
        )
    reference = bundle["reliability_reference"]
    prediction.reliability_json = json.dumps(
        {
            "sensor_quality": sensor_quality(features, FEATURE_NAMES),
            "out_of_distribution": out_of_distribution(features, reference),
            "drift": population_drift([features], reference),
            "calibration": {
                "status": "available",
                "method": bundle.get("calibration_method"),
                "temperature": temperature,
            },
            "uncertainty": {
                "maximum_calibrated_probability": float(calibrated.max()),
                "acceptance_threshold": float(bundle["acceptance_policy"]["threshold"]),
                "threshold_found": bool(bundle["acceptance_policy"]["threshold_found"]),
                "decision_basis": "validation-selected calibrated probability threshold",
            },
            "decision_status": prediction.decision_status,
            "metadata_source": bundle.get("reliability_metadata_source"),
        },
        separators=(",", ":"),
    )
    return explanation


def process_feature_message(bundle: dict[str, object], payload: dict) -> tuple[Prediction | None, bool]:
    settings = get_settings()
    try:
        message = FeatureMessage.model_validate(payload)
    except ValidationError as exc:
        raise MessageRejected(str(exc)) from exc
    if message.schema_version != settings.feature_schema_version:
        raise MessageRejected(
            f"Unsupported feature schema {message.schema_version!r}; "
            f"expected {settings.feature_schema_version!r}"
        )
    if set(message.features) != set(FEATURE_NAMES):
        missing = sorted(set(FEATURE_NAMES) - set(message.features))
        extra = sorted(set(message.features) - set(FEATURE_NAMES))
        raise MessageRejected(f"Feature schema mismatch (missing={missing}, extra={extra})")
    values = np.asarray([[message.features[name] for name in FEATURE_NAMES]], dtype=np.float64)
    if not np.isfinite(values).all():
        raise MessageRejected("Features must contain only finite numeric values")
    ordered = pd.DataFrame(values, columns=FEATURE_NAMES)

    db = SessionLocal()
    try:
        existing = db.query(Prediction).filter_by(message_id=message.message_id).one_or_none()
        if existing:
            same_payload = (
                existing.machine_id == message.machine_id
                and existing.source == message.source
                and existing.recording_id == message.recording_id
                and existing.window_index == message.window_index
                and existing.schema_version == message.schema_version
                and existing.features == message.features
            )
            if not same_payload:
                raise MessageRejected(f"Message ID {message.message_id!r} was reused with different content")
            return existing, True
    finally:
        db.close()

    started = time.perf_counter()
    model = bundle["model"]
    predicted = str(model.predict(ordered)[0])
    if predicted not in LABELS:
        raise RuntimeError(f"Loaded model returned unsupported class {predicted!r}")
    probabilities = {}
    calibrated_probabilities = {}
    reliability_reference = bundle.get("reliability_reference")
    calibration_temperature = bundle.get("calibration_temperature")
    probability_classes = model_probability_classes(model, list(bundle["class_names"]))
    raw_matrix = np.asarray(model.predict_proba(ordered), dtype=float)
    if raw_matrix.shape != (1, len(probability_classes)) or not np.isfinite(raw_matrix).all():
        raise RuntimeError("Loaded model returned an invalid probability vector")
    if (raw_matrix < 0).any() or not np.isclose(raw_matrix.sum(axis=1), 1.0, atol=1e-5).all():
        raise RuntimeError("Loaded model probabilities must be non-negative and sum to one")
    probabilities = {
        label: float(value) for label, value in zip(probability_classes, raw_matrix[0])
    }

    quality = sensor_quality(message.features, FEATURE_NAMES)
    if reliability_reference:
        distribution = out_of_distribution(message.features, reliability_reference)
        history_db = SessionLocal()
        try:
            history = (
                history_db.query(Prediction)
                .filter_by(machine_id=message.machine_id, source=message.source)
                .order_by(Prediction.received_at.desc())
                .limit(int(reliability_reference["drift_window_size"]) - 1)
                .all()
            )
            historic_features = [row.features for row in reversed(history)]
        finally:
            history_db.close()
        drift = population_drift(
            [*historic_features, message.features],
            reliability_reference,
        )
    else:
        distribution = {"status": "unavailable", "outside_reference_features": []}
        drift = {"status": "unavailable", "window_size": 0, "psi_max": None}

    decision_status = "REVIEW_REQUIRED"
    reliability: dict[str, object] = {
        "sensor_quality": quality,
        "out_of_distribution": distribution,
        "drift": drift,
        "calibration": {
            "status": "available" if calibration_temperature is not None else "unavailable",
            "method": bundle.get("calibration_method"),
            "temperature": calibration_temperature,
        },
        "uncertainty": {},
        "decision_status": decision_status,
    }
    explanation = {}
    if calibration_temperature is not None:
        calibrated_matrix = apply_temperature(raw_matrix, float(calibration_temperature))
        calibrated_probabilities = {
            label: float(value) for label, value in zip(probability_classes, calibrated_matrix[0])
        }
        acceptance = bundle.get("acceptance_policy", {})
        threshold = float(acceptance.get("threshold", 1.01))
        confidence = float(calibrated_matrix.max())
        reliability["uncertainty"] = {
            "maximum_calibrated_probability": confidence,
            "acceptance_threshold": threshold,
            "threshold_found": bool(acceptance.get("threshold_found", False)),
            "decision_basis": "validation-selected calibrated probability threshold",
        }
        if quality["status"] == "degraded":
            decision_status = "DEGRADED_SENSOR_DATA"
        elif distribution.get("status") == "warning" or drift.get("status") == "warning":
            decision_status = "OUT_OF_DISTRIBUTION"
        elif confidence >= threshold and acceptance.get("threshold_found", False):
            decision_status = "ACCEPTED"
        else:
            decision_status = "REVIEW_REQUIRED"
        if reliability_reference:
            explanation = explain_prediction(
                model,
                ordered,
                calibrated_matrix,
                probability_classes,
                reliability_reference,
                float(calibration_temperature),
            )
    reliability["decision_status"] = decision_status
    latency_ms = (time.perf_counter() - started) * 1000

    db = SessionLocal()
    try:
        row = Prediction(
            message_id=message.message_id,
            machine_id=message.machine_id,
            source=message.source,
            recording_id=message.recording_id,
            window_index=message.window_index,
            predicted_condition=predicted,
            probabilities_json=json.dumps(probabilities, separators=(",", ":")),
            calibrated_probabilities_json=json.dumps(calibrated_probabilities, separators=(",", ":")),
            reliability_json=json.dumps(reliability, separators=(",", ":")),
            explanation_json=json.dumps(explanation, separators=(",", ":")),
            decision_status=decision_status,
            features_json=json.dumps(message.features, separators=(",", ":")),
            model_version=str(bundle["model_version"]),
            schema_version=message.schema_version,
            event_timestamp=message.event_timestamp,
            latency_ms=latency_ms,
            received_at=datetime.now(timezone.utc),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row, False
    except Exception:
        db.rollback()
        LOGGER.exception("Prediction persistence failed for message %s", message.message_id)
        raise
    finally:
        db.close()
