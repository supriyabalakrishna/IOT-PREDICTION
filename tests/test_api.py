import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.api as api_module
from app.api import app
from app.config import get_settings
from app.storage import Base, Prediction


def test_api_timestamps_are_serialized_as_utc():
    assert api_module._utc_iso(datetime(2026, 10, 25, 11, 30)) == "2026-10-25T11:30:00+00:00"
    offset_time = datetime(2026, 10, 25, 17, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert api_module._utc_iso(offset_time) == "2026-10-25T11:30:00+00:00"


@pytest.mark.skipif(
    not get_settings().dataset_path.is_dir(),
    reason="The source dataset is not included in CI.",
)
def test_liveness_and_dataset_manifest_endpoints():
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        response = client.get("/api/dataset")
        assert response.status_code == 200
        manifest = response.json()
        assert manifest["recording_count"] == 45
        assert manifest["recordings_per_class"]["Unbalance"] == 15


def test_protected_administrative_api_is_not_open_without_configured_key(monkeypatch):
    monkeypatch.setattr(api_module, "get_settings", lambda: SimpleNamespace(admin_api_key=None))
    with TestClient(app) as client:
        response = client.post("/api/replay/start", json={})
        assert response.status_code == 503
        assert "ADMIN_API_KEY" in response.json()["detail"]


def test_model_comparison_includes_reliability_and_research_results(monkeypatch, tmp_path):
    report = {
        "status": "completed",
        "selected_model": "Random Forest",
        "selection_metric": "validation macro-F1",
        "comparison": [],
        "test_metrics": {},
        "automl": {"status": "not_run"},
        "recording_splits": {},
        "class_collapse_detected": False,
        "reliability": {"acceptance_policy": {"threshold": 0.8}},
        "cross_load_evaluation": {"status": "completed", "folds": []},
        "sensor_degradation_evaluation": {"status": "completed", "scenarios": []},
    }
    (tmp_path / "latest.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(api_module, "settings", SimpleNamespace(report_dir=tmp_path))

    response = api_module.model_comparison()

    assert response["reliability"] == report["reliability"]
    assert response["cross_load_evaluation"] == report["cross_load_evaluation"]
    assert response["sensor_degradation_evaluation"] == report["sensor_degradation_evaluation"]


def test_human_review_is_authenticated_and_does_not_replace_model_prediction(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(api_module, "SessionLocal", factory)
    monkeypatch.setattr(api_module, "get_settings", lambda: SimpleNamespace(admin_api_key="test-key"))
    db = factory()
    try:
        db.add(Prediction(
            message_id="review-test",
            machine_id="test-machine",
            source="replay",
            predicted_condition="Unbalance",
            probabilities_json='{"Unbalance": 0.8}',
            features_json="{}",
            model_version="model-test",
            schema_version="1.0",
            latency_ms=1.0,
            decision_status="REVIEW_REQUIRED",
        ))
        db.commit()
        prediction_id = db.query(Prediction).one().id
    finally:
        db.close()

    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/admin/predictions/{prediction_id}/review",
                json={"review_label": "Normal", "reviewed_by": "operator-1", "review_note": "Checked"},
                headers={"X-Admin-API-Key": "test-key"},
            )
            assert response.status_code == 200
            result = response.json()["prediction"]
            assert result["predicted_condition"] == "Unbalance"
            assert result["human_review"]["review_label"] == "Normal"
            assert result["human_review"]["reviewed_by"] == "operator-1"
            duplicate = client.post(
                f"/api/admin/predictions/{prediction_id}/review",
                json={"review_label": "Normal", "reviewed_by": "operator-1"},
                headers={"X-Admin-API-Key": "test-key"},
            )
            assert duplicate.status_code == 409
    finally:
        engine.dispose()
