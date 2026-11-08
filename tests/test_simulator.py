from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import app.simulator as simulator
from app.dataset import LABELS
from app.features import FEATURE_NAMES


def _feature_frame():
    return pd.DataFrame([
        {
            "recording_id": f"recording-{index}",
            "window_index": index,
            "label": label,
            **{name: float(index) for name in FEATURE_NAMES},
        }
        for index, label in enumerate(LABELS)
    ])


def test_terminal_simulator_runs_bounded_real_rows_without_passing_labels(monkeypatch, tmp_path):
    frame = _feature_frame()
    (tmp_path / "test_features.csv.gz").touch()
    monkeypatch.setattr(simulator, "get_settings", lambda: SimpleNamespace(
        artifact_dir=Path(tmp_path),
        feature_schema_version="1.0",
    ))
    monkeypatch.setattr(simulator.pd, "read_csv", lambda path: frame.copy())
    payloads = []

    def process(bundle, payload):
        payloads.append(payload)
        return SimpleNamespace(
            predicted_condition=LABELS[payload["window_index"]],
            decision_status="REVIEW_REQUIRED",
        ), False

    monkeypatch.setattr(simulator, "process_feature_message", process)

    result = simulator.simulate_predictions(
        {"model_version": "model-test"},
        count=7,
        interval_seconds=0,
    )

    assert result["processed"] == 7
    assert result["source_classes"] == sorted(LABELS)
    assert result["prediction_distribution"] == {
        "Normal": 2,
        "BPFI": 2,
        "BPFO": 1,
        "Misalignment": 1,
        "Unbalance": 1,
    }
    assert len({payload["message_id"] for payload in payloads}) == 7
    assert all("label" not in payload and "source_class" not in payload for payload in payloads)
    assert all(set(payload["features"]) == set(FEATURE_NAMES) for payload in payloads)


def test_terminal_simulator_fails_clearly_if_training_features_are_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(simulator, "get_settings", lambda: SimpleNamespace(
        artifact_dir=Path(tmp_path),
        feature_schema_version="1.0",
    ))

    with pytest.raises(FileNotFoundError, match="python -m app.cli train"):
        simulator.simulate_predictions({"model_version": "model-test"}, count=1)
