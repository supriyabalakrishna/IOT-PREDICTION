import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.inference as inference
from app.dataset import LABELS
from app.features import FEATURE_NAMES
from app.reliability import build_reference
from app.storage import Base, Prediction


class FixedClassifier:
    classes_ = np.array(LABELS)

    def predict(self, values):
        return np.full(len(values), "Normal")

    def predict_proba(self, values):
        return np.tile(np.array([[0.8, 0.05, 0.05, 0.05, 0.05]]), (len(values), 1))


@pytest.fixture
def isolated_prediction_database(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(inference, "SessionLocal", factory)
    yield factory
    engine.dispose()


def test_inference_persists_once_and_detects_conflicting_duplicate_ids(isolated_prediction_database):
    values = {name: float(index) for index, name in enumerate(FEATURE_NAMES)}
    payload = {
        "message_id": "replay-session:0",
        "machine_id": "machine-001",
        "source": "replay",
        "schema_version": "1.0",
        "recording_id": "0Nm_Normal",
        "window_index": 0,
        "features": values,
    }
    bundle = {
        "model": FixedClassifier(),
        "model_version": "model-test",
        "class_names": list(LABELS),
    }

    first, duplicate = inference.process_feature_message(bundle, payload)
    again, is_duplicate = inference.process_feature_message(bundle, payload)
    assert first.predicted_condition == "Normal"
    assert first.probabilities["Normal"] == pytest.approx(0.8)
    assert duplicate is False
    assert is_duplicate is True
    assert again.id == first.id

    conflicting = {**payload, "features": {**values, FEATURE_NAMES[0]: -1.0}}
    with pytest.raises(inference.MessageRejected, match="reused with different content"):
        inference.process_feature_message(bundle, conflicting)
    db = isolated_prediction_database()
    try:
        assert db.query(func.count(Prediction.id)).scalar() == 1
    finally:
        db.close()


def test_inference_rejects_nonfinite_and_wrong_schema_features(isolated_prediction_database):
    bundle = {"model": FixedClassifier(), "model_version": "model-test", "class_names": list(LABELS)}
    values = {name: 1.0 for name in FEATURE_NAMES}
    payload = {
        "message_id": "bad-row",
        "machine_id": "machine-001",
        "source": "live",
        "schema_version": "1.0",
        "features": {**values, FEATURE_NAMES[0]: float("nan")},
    }
    with pytest.raises(inference.MessageRejected):
        inference.process_feature_message(bundle, payload)
    payload["features"] = {name: 1.0 for name in FEATURE_NAMES[:-1]}
    with pytest.raises(inference.MessageRejected, match="Feature schema mismatch"):
        inference.process_feature_message(bundle, payload)


def test_inference_persists_calibration_explanation_and_reliability_decision(isolated_prediction_database):
    rows = []
    for group in range(4):
        for label_index, label in enumerate(LABELS):
            rows.append({
                "recording_id": f"{group}-{label}",
                "label": label,
                **{
                    name: float(group + label_index + feature_index / 100)
                    for feature_index, name in enumerate(FEATURE_NAMES)
                },
            })
    frame = pd.DataFrame(rows)
    training = frame.iloc[:10]
    reference = build_reference(training, frame.iloc[10:15], FEATURE_NAMES)
    features = {name: float(training[name].median()) for name in FEATURE_NAMES}
    payload = {
        "message_id": "reliability-test",
        "machine_id": "machine-001",
        "source": "replay",
        "schema_version": "1.0",
        "features": features,
    }
    bundle = {
        "model": FixedClassifier(),
        "model_version": "model-reliability-test",
        "class_names": list(LABELS),
        "calibration_temperature": 1.0,
        "calibration_method": "test temperature scaling",
        "acceptance_policy": {"threshold": 0.0, "threshold_found": True},
        "reliability_reference": reference,
    }

    prediction, duplicate = inference.process_feature_message(bundle, payload)

    assert duplicate is False
    assert prediction.decision_status == "ACCEPTED"
    assert prediction.calibrated_probabilities["Normal"] == pytest.approx(0.8)
    assert prediction.reliability["calibration"]["status"] == "available"
    assert prediction.explanation["top_features"]
