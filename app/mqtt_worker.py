from __future__ import annotations

import json
import logging
import threading
import time

import paho.mqtt.client as mqtt

from app.config import get_settings
from app.features import FEATURE_NAMES
from app.inference import MessageRejected, process_feature_message, store_event
from app.training import load_active_bundle
from app.storage import initialize_database

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("mqtt-worker")


def main() -> None:
    settings = get_settings()
    initialize_database()
    bundle = load_active_bundle()
    while bundle is None:
        LOGGER.warning("No model is deployed; waiting for a validated model activation")
        time.sleep(5)
        bundle = load_active_bundle()

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=settings.mqtt_client_id,
        protocol=mqtt.MQTTv311,
        manual_ack=True,
    )
    if settings.mqtt_username:
        client.username_pw_set(settings.mqtt_username, settings.mqtt_password)
    if settings.mqtt_tls:
        client.tls_set()

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            LOGGER.error("MQTT connection rejected: %s", reason_code)
            return
        client.subscribe(settings.mqtt_features_topic, qos=1)
        LOGGER.info("Connected to broker; subscribed to %s", settings.mqtt_features_topic)
        store_event("broker_connected", "Prediction worker connected to MQTT broker")

    def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
        LOGGER.warning("MQTT disconnected: %s", reason_code)
        try:
            store_event("broker_disconnected", f"Prediction worker disconnected: {reason_code}", "warning")
        except Exception:
            LOGGER.exception("Could not record MQTT disconnection")

    def on_message(client, userdata, message):
        nonlocal bundle
        try:
            payload = json.loads(message.payload)
            if not isinstance(payload, dict) or not isinstance(payload.get("features"), dict):
                raise MessageRejected("MQTT payload must be a JSON object with a feature map")
            if set(payload["features"]) != set(FEATURE_NAMES):
                raise MessageRejected("Incoming feature names do not match the loaded model schema")
            active_pointer = settings.artifact_dir / "active.json"
            active_version = json.loads(active_pointer.read_text(encoding="utf-8"))["version"]
            if active_version != bundle["model_version"]:
                latest_bundle = load_active_bundle()
                if latest_bundle is None:
                    raise RuntimeError("The active model pointer was removed during inference")
                bundle = latest_bundle
                store_event(
                    "model_reloaded",
                    f"Prediction worker loaded model {bundle['model_version']}",
                    "warning",
                )
            prediction, duplicate = process_feature_message(bundle, payload)
            output = {
                "message_id": prediction.message_id,
                "machine_id": prediction.machine_id,
                "source": prediction.source,
                "recording_id": prediction.recording_id,
                "window_index": prediction.window_index,
                "predicted_condition": prediction.predicted_condition,
                "probabilities": prediction.probabilities,
                "calibrated_probabilities": prediction.calibrated_probabilities,
                "decision_status": prediction.decision_status,
                "reliability": prediction.reliability,
                "explanation": prediction.explanation,
                "model_version": prediction.model_version,
                "latency_ms": prediction.latency_ms,
                "received_at": prediction.received_at.isoformat(),
                "duplicate_delivery": duplicate,
            }
            info = client.publish(
                settings.mqtt_predictions_topic,
                json.dumps(output, separators=(",", ":")),
                qos=1,
            )
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(f"Could not publish prediction: MQTT result {info.rc}")
            def acknowledge_after_output():
                info.wait_for_publish(timeout=30)
                if info.is_published():
                    client.ack(message.mid, message.qos)
                else:
                    LOGGER.error(
                        "Prediction output was not acknowledged; retaining input delivery for retry: %s",
                        message.mid,
                    )

            threading.Thread(
                target=acknowledge_after_output,
                name=f"mqtt-ack-{message.mid}",
                daemon=True,
            ).start()
            return
        except (json.JSONDecodeError, MessageRejected) as exc:
            LOGGER.error("Rejected MQTT message: %s", exc)
            store_event("invalid_message", str(exc), "warning")
        except Exception as exc:
            LOGGER.exception("Prediction processing failed; broker delivery will be retried")
            try:
                store_event("prediction_failed", str(exc), "error")
            except Exception:
                LOGGER.exception("Could not persist the prediction error")
            return
        client.ack(message.mid, message.qos)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
