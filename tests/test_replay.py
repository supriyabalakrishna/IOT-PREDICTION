from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import app.replay as replay
from app.dataset import LABELS
from app.features import FEATURE_NAMES


def test_local_replay_runs_model_inference_without_mqtt(monkeypatch, tmp_path):
    rows = []
    for index, label in enumerate(LABELS):
        rows.append({
            "recording_id": f"recording-{index}",
            "window_index": 0,
            "label": label,
            **{name: float(index) for name in FEATURE_NAMES},
        })
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(replay.pd, "read_csv", lambda path: frame.copy())
    monkeypatch.setattr(
        replay,
        "get_settings",
        lambda: SimpleNamespace(artifact_dir=Path(tmp_path), feature_schema_version="1.0"),
    )
    monkeypatch.setattr(
        replay.mqtt,
        "Client",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Local replay must not use MQTT")),
    )
    monkeypatch.setattr(replay.ReplayController, "_record_event", lambda *args, **kwargs: None)

    processed_payloads = []

    def process_message(bundle, payload):
        processed_payloads.append(payload)
        class_index = int(payload["recording_id"].split("-")[1])
        return SimpleNamespace(predicted_condition=LABELS[class_index]), False

    monkeypatch.setattr(replay, "process_feature_message", process_message)
    controller = replay.ReplayController()
    controller._status = {
        "state": "starting",
        "published": 0,
        "processed": 0,
        "failed": 0,
        "cycle_processed": 0,
        "cycles_completed": 0,
        "prediction_distribution": {label: 0 for label in LABELS},
    }
    controller._run(
        "replay-test",
        "machine-test",
        None,
        100,
        0,
        "local",
        False,
        {"model_version": "model-test"},
    )

    status = controller.status()
    assert status["state"] == "completed"
    assert status["processed"] == len(LABELS)
    assert status["published"] == 0
    assert status["cycles_completed"] == 1
    assert set(status["classes"]) == set(LABELS)
    assert status["prediction_distribution"] == {label: 1 for label in LABELS}
    assert len(processed_payloads) == len(LABELS)
    assert all("label" not in payload for payload in processed_payloads)


def test_start_returns_without_reacquiring_its_status_lock(monkeypatch, tmp_path):
    (tmp_path / "test_features.csv.gz").touch()
    monkeypatch.setattr(
        replay,
        "get_settings",
        lambda: SimpleNamespace(artifact_dir=Path(tmp_path)),
    )

    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(replay.threading, "Thread", FakeThread)
    controller = replay.ReplayController()

    status = controller.start("machine-test", None, 1, None, "local", True, {"model_version": "model-test"})

    assert status["state"] == "starting"
    assert status["mode"] == "local"
    assert status["continuous"] is True
    assert status["replay_session_id"]


def test_continuous_replay_cycles_until_stopped(monkeypatch, tmp_path):
    frame = pd.DataFrame([
        {
            "recording_id": f"recording-{index}",
            "window_index": 0,
            "label": label,
            **{name: float(index) for name in FEATURE_NAMES},
        }
        for index, label in enumerate(LABELS)
    ])
    monkeypatch.setattr(replay.pd, "read_csv", lambda path: frame.copy())
    monkeypatch.setattr(
        replay,
        "get_settings",
        lambda: SimpleNamespace(artifact_dir=Path(tmp_path), feature_schema_version="1.0"),
    )
    monkeypatch.setattr(replay.ReplayController, "_record_event", lambda *args, **kwargs: None)
    controller = replay.ReplayController()

    def process_message(bundle, payload):
        class_index = int(payload["recording_id"].split("-")[1])
        if payload["sequence_number"] == len(LABELS) * 2 - 1:
            controller._stop.set()
        return SimpleNamespace(predicted_condition=LABELS[class_index]), False

    monkeypatch.setattr(replay, "process_feature_message", process_message)
    controller._status = {
        "state": "starting",
        "prediction_distribution": {label: 0 for label in LABELS},
        "processed": 0,
        "cycle_processed": 0,
        "cycles_completed": 0,
    }

    controller._run("replay-test", "machine-test", None, 100, 0, "local", True, {})

    status = controller.status()
    assert status["state"] == "stopped"
    assert status["processed"] == len(LABELS) * 2
    assert status["cycles_completed"] == 2
    assert status["prediction_distribution"] == {label: 2 for label in LABELS}
