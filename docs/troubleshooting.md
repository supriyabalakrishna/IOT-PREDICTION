# Troubleshooting

## Dataset or feature extraction

- Run `python -m app.cli validate` and inspect `reports/dataset_manifest.json` for recording pairings, class coverage, channel counts, and empty TDMS channel IDs.
- If a recording has no complete one-second window, an unsupported MAT structure, an empty shared temperature/current channel, a label/run-name mismatch, or an unexpected sampling rate, training stops with the recording identifier and reason. Do not copy/invent sensor rows to bypass validation.
- The `ai2` and `ai3` TDMS current channels are empty in the nine BPFO files. The common current channel used by this model is `ai0`; check the report if using a different dataset revision.
- Delete only the specific `artifacts/features.csv.gz` and `artifacts/features_manifest.json` files if intentionally changing the feature extractor and needing a forced re-extraction. Do not delete original source recordings.

## No model or readiness is unavailable

- Run `python -m app.cli train`, inspect `reports/latest.json`, and explicitly deploy the validated model.
- `GET /health/live` indicates that the API process is running. `GET /health/ready` requires a usable database and an activated model.
- If a deployed artifact cannot load, check `artifacts/active.json`, the matching report in `reports/`, the artifact checksum, and the configured `ARTIFACT_DIR`. Never bypass checksum/schema validation by loading a different file manually.
- If the test report has low recall or a missing predicted class, treat that as model evidence. Do not force prediction labels or substitute a default class.

## MQTT worker and replay

- Confirm the broker is reachable at `MQTT_HOST:MQTT_PORT`, the configured topic names match, and local Mosquitto has started. Compose's development broker allows anonymous connections only on localhost.
- The prediction worker waits for a model activation. When a new model is deployed, it checks the atomic active pointer on message arrival and reloads only when the version changes.
- Replay controls and prediction HTTP inference require `X-Admin-API-Key`. An unset key disables administrative routes with HTTP 503; a wrong key returns 401.
- QoS 1 can redeliver. Check `message_id` and `duplicate_delivery`; duplicate predictions are safe to deduplicate by message ID. Repeated persistence failures are logged and remain unacknowledged for broker retry.

## PostgreSQL and Compose

- Check `docker compose ps` and `docker compose logs database mqtt api prediction-worker frontend`.
- The API migration runs as `alembic upgrade head`; verify `DATABASE_URL`, `POSTGRES_PASSWORD`, and database health first.
- If the dataset is outside the repository, set `DATASET_HOST_PATH` in `.env` to its absolute host path. Do not put raw dataset files in the application image.
- Docker Desktop must be running. A successful image build or container start alone does not mean the API/database/broker are ready; check health status and `/health/ready`.
- Preserve Compose volumes for backups. `docker compose down -v` deletes persistent database and broker data.

