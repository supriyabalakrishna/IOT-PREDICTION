from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.api as api_module
import app.inference as inference
from app.api import app
from app.config import get_settings
from app.dataset import LABELS
from app.features import FEATURE_NAMES
from app.storage import Base, Prediction
from app.training import load_active_bundle

settings = get_settings()
HAS_REAL_BUNDLE = (
    (settings.artifact_dir / "active.json").is_file()
    and (settings.artifact_dir / "test_features.csv.gz").is_file()
)


@pytest.mark.skipif(not HAS_REAL_BUNDLE, reason="Train and deploy a model to run this data-backed test.")
def test_genuine_held_out_row_matches_offline_model_and_persisted_api_inference(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(api_module, "SessionLocal", factory)
    monkeypatch.setattr(inference, "SessionLocal", factory)
    bundle = load_active_bundle()
    row = pd.read_csv(settings.artifact_dir / "test_features.csv.gz", nrows=1).iloc[0]
    features = {name: float(row[name]) for name in FEATURE_NAMES}
    expected = str(bundle["model"].predict(pd.DataFrame([features], columns=FEATURE_NAMES))[0])
    message_id = f"integration-test:{uuid4()}"
    payload = {
        "message_id": message_id,
        "machine_id": "recorded-test-machine",
        "source": "replay",
        "schema_version": settings.feature_schema_version,
        "recording_id": str(row["recording_id"]),
        "window_index": int(row["window_index"]),
        "features": features,
    }
    monkeypatch.setattr(settings, "admin_api_key", "test-only-key")
    try:
        with TestClient(app) as client:
            assert client.get("/health/ready").status_code == 200
            response = client.post(
                "/api/predict",
                json=payload,
                headers={"X-Admin-API-Key": "test-only-key"},
            )
            assert response.status_code == 200, response.text
            persisted = response.json()["prediction"]
            assert persisted["predicted_condition"] == expected
            assert persisted["predicted_condition"] in LABELS
            assert persisted["recording_id"] == str(row["recording_id"])
            assert persisted["source"] == "replay"
            assert persisted["explanation"]["top_features"]

            db = factory()
            try:
                stored_prediction = db.query(Prediction).filter_by(message_id=message_id).one()
                stored_prediction.explanation_json = None
                stored_prediction.calibrated_probabilities_json = None
                prediction_id = stored_prediction.id
                db.commit()
            finally:
                db.close()
            explanation_response = client.get(f"/api/predictions/{prediction_id}/explanation")
            assert explanation_response.status_code == 200
            explanation_result = explanation_response.json()
            assert explanation_result["explanation_status"] == "computed_for_compatible_legacy_prediction"
            assert explanation_result["explanation"]["top_features"]
            assert explanation_result["reliability"]["calibration"]["status"] == "available"
            assert explanation_result["reliability"]["drift"]["status"] == "warming_up"

            duplicate = client.post(
                "/api/predict",
                json=payload,
                headers={"X-Admin-API-Key": "test-only-key"},
            )
            assert duplicate.status_code == 200
            assert duplicate.json()["duplicate"] is True
            history = client.get("/api/predictions?source=replay&limit=500").json()["items"]
            stored = [item for item in history if item["message_id"] == message_id]
            assert len(stored) == 1
            assert stored[0]["predicted_condition"] == expected
    finally:
        engine.dispose()
