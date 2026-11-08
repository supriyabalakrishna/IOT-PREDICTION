from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import paho.mqtt.client as mqtt

from app.config import get_settings
from app.dataset import LABELS
from app.features import FEATURE_NAMES
from app.inference import process_feature_message
from app.storage import AppEvent, SessionLocal, utc_now

LOGGER = logging.getLogger(__name__)


class ReplayController:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self._status: dict[str, object] = {"state": "idle"}

    def status(self) -> dict[str, object]:
        with self._lock:
            return dict(self._status)

    def start(
        self,
        machine_id: str,
        recording_ids: list[str] | None,
        speed: float,
        delay_seconds: float | None,
        mode: str,
        continuous: bool,
        bundle: dict[str, object],
    ) -> dict[str, object]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("A replay is already active")
            settings = get_settings()
            features_path = settings.artifact_dir / "test_features.csv.gz"
            if not features_path.exists():
                raise FileNotFoundError("No held-out feature rows found; train and evaluate a model first")
            self._stop.clear()
            self._resume.set()
            session_id = str(uuid.uuid4())
            self._status = {
                "state": "starting",
                "replay_session_id": session_id,
                "machine_id": machine_id,
                "published": 0,
                "processed": 0,
                "failed": 0,
                "mode": mode,
                "continuous": continuous,
                "cycles_completed": 0,
                "cycle_processed": 0,
                "total": 0,
                "classes": [],
                "missing_classes": list(LABELS),
                "prediction_distribution": {label: 0 for label in LABELS},
                "recording_ids": [],
                "last_error": None,
            }
            self._thread = threading.Thread(
                target=self._run,
                args=(session_id, machine_id, recording_ids, speed, delay_seconds, mode, continuous, bundle),
                name=f"dataset-replay-{session_id[:8]}",
                daemon=True,
            )
            self._thread.start()
            return dict(self._status)

    def _run(self, session_id, machine_id, recording_ids, speed, delay_seconds, mode, continuous, bundle):
        settings = get_settings()
        client = None
        try:
            frame = pd.read_csv(settings.artifact_dir / "test_features.csv.gz")
            if recording_ids:
                frame = frame[frame.recording_id.isin(recording_ids)]
            frame = frame.sort_values(["window_index", "recording_id"], kind="stable")
            if frame.empty:
                raise ValueError("No held-out windows match the requested recording IDs")
            classes = sorted(frame.label.unique().tolist())
            records = sorted(frame.recording_id.unique().tolist())
            missing = sorted(set(LABELS) - set(classes))
            with self._lock:
                self._status.update({
                    "state": "running",
                    "total": len(frame),
                    "classes": classes,
                    "missing_classes": missing,
                    "recording_ids": records,
                })
            if mode == "mqtt":
                client = mqtt.Client(
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f"dataset-replay-{session_id[:8]}",
                    protocol=mqtt.MQTTv311,
                )
                if settings.mqtt_username:
                    client.username_pw_set(settings.mqtt_username, settings.mqtt_password)
                if settings.mqtt_tls:
                    client.tls_set()
                client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=60)
                client.loop_start()
                info = client.publish(
                    settings.mqtt_events_topic,
                    json.dumps({
                        "event_type": "replay_started",
                        "replay_session_id": session_id,
                        "machine_id": machine_id,
                        "recording_count": len(records),
                        "class_coverage": classes,
                        "missing_classes": missing,
                        "source": "replay",
                    }),
                    qos=1,
                )
                info.wait_for_publish(timeout=10)
            sequence = 0
            while not self._stop.is_set():
                for _, row in frame.iterrows():
                    while not self._resume.wait(timeout=0.25):
                        if self._stop.is_set():
                            break
                    if self._stop.is_set():
                        break
                    payload = {
                        "message_id": f"{session_id}:{sequence}",
                        "machine_id": machine_id,
                        "source": "replay",
                        "schema_version": settings.feature_schema_version,
                        "recording_id": str(row["recording_id"]),
                        "window_index": int(row["window_index"]),
                        "replay_session_id": session_id,
                        "sequence_number": sequence,
                        "event_timestamp": datetime.now(timezone.utc).isoformat(),
                        "features": {name: float(row[name]) for name in FEATURE_NAMES},
                    }
                    if mode == "local":
                        prediction, _ = process_feature_message(bundle, payload)
                        with self._lock:
                            distribution = self._status["prediction_distribution"]
                            distribution[prediction.predicted_condition] += 1
                            self._status["processed"] += 1
                            self._status["cycle_processed"] += 1
                    else:
                        info = client.publish(
                            settings.mqtt_features_topic,
                            json.dumps(payload, separators=(",", ":")),
                            qos=1,
                        )
                        info.wait_for_publish(timeout=10)
                        if not info.is_published():
                            raise RuntimeError(f"MQTT publish timed out for sequence {sequence}")
                        with self._lock:
                            self._status["published"] += 1
                            self._status["cycle_processed"] += 1
                    sequence += 1
                    sleep_for = delay_seconds if delay_seconds is not None else 1.0 / speed
                    if sleep_for:
                        self._stop.wait(sleep_for)
                else:
                    with self._lock:
                        self._status["cycles_completed"] += 1
                        self._status["cycle_processed"] = 0
                    if not continuous:
                        break
            state = "stopped" if self._stop.is_set() else "completed"
            with self._lock:
                self._status["state"] = state
            self._record_event("replay_completed" if state == "completed" else "replay_stopped", session_id)
        except Exception as exc:
            LOGGER.exception("Dataset replay failed")
            with self._lock:
                self._status.update({"state": "failed", "last_error": f"{type(exc).__name__}: {exc}"})
                self._status["failed"] += 1
            self._record_event("replay_failed", str(exc), "error")
        finally:
            if client is not None:
                try:
                    client.disconnect()
                    client.loop_stop()
                except Exception:
                    LOGGER.exception("Could not close replay MQTT client cleanly")

    def pause(self):
        self._resume.clear()
        with self._lock:
            self._status["state"] = "paused"
            return dict(self._status)

    def resume(self):
        self._resume.set()
        with self._lock:
            self._status["state"] = "running"
            return dict(self._status)

    def stop(self):
        self._stop.set()
        self._resume.set()
        with self._lock:
            self._status["state"] = "stopping"
            return dict(self._status)

    @staticmethod
    def _record_event(event_type: str, detail: str, severity: str = "info"):
        db = SessionLocal()
        try:
            db.add(AppEvent(event_type=event_type, message=detail, severity=severity, created_at=utc_now()))
            db.commit()
        finally:
            db.close()


replay_controller = ReplayController()
