# Industrial Predictive Maintenance and Fault Intelligence Platform

**Project title:** IoT Sensor Failure Prediction Using AutoML

**Problem statement:** "To create a predictive maintenance system using AutoML techniques that analyzes IoT sensor data to forecast equipment failures, minimize downtime, and improve operational reliability in industrial environments."

This repository provides a local, evidence-driven **machine-condition classifier**. It extracts features from recorded vibration, temperature, and current channels, evaluates multiple classifiers using recording-level splits, persists predictions, and optionally replays held-out feature rows over MQTT. A condition classification is not a failure-time forecast, remaining-useful-life estimate, or confirmed maintenance diagnosis.

## What is present in this workspace

The application expects the provided dataset directory at the repository root. The inspected copy has 45 vibration MAT recordings and 45 paired current/temperature TDMS recordings. The vibration files have four acceleration channels in `g` at 25.6 kHz. TDMS metadata identifies two temperature channels and three nominal current channels (`A`) sampled at approximately 25.6 kHz. The temperature unit property contains a replacement-character encoding rather than a clean unit string and needs confirmation against acquisition-system documentation. The `ai2` and `ai3` current channels are empty in all nine BPFO recordings. To avoid silently losing an entire required class, the shared classifier schema uses the non-empty common current `ai0` channel plus both temperature channels; the two sparse channels are not used. Matching vibration and TDMS recordings are joined by normalized recording ID; the five acoustic MAT recordings cover only Normal, BPFI, and BPFO and are not used in the five-class model.

Canonical labels are `Normal`, `BPFI`, `BPFO`, `Misalignment`, and `Unbalance`. Source labels are parsed from recording identifiers, including the `Unbalalnce` source spelling variant. Where available, the vibration MAT run-name metadata is checked against the parsed label. The workspace did not include a dataset README or paper citation, so the mapping is traceable to source names and embedded run metadata but has not been independently verified against publisher documentation.

## Components

- `app/dataset.py`, `app/features.py`: dataset discovery, format checks, sensor-feature schema, deterministic one-second windows, and feature extraction.
- `app/training.py`: recording-group train/validation/test split, comparison candidates, bounded FLAML search, validation-only model selection, held-out evaluation, and checksummed bundle generation.
- `app/reliability.py`, `app/research.py`: validation-calibrated probabilities, selective acceptance, sensor-quality and feature-drift checks, prediction explanations, leave-one-load-out evaluation, and controlled sensor-degradation experiments.
- `app/api.py`: FastAPI health, prediction history, condition distribution, events, dataset/model/research reports, reliability review queue, replay controls, training jobs, and protected model deployment/rollback.
- `app/mqtt_worker.py`, `app/replay.py`: QoS 1 feature ingestion with explicit validation, duplicate IDs, retryable persistence, inference, prediction publication, and replay using genuine held-out rows.
- `app/storage.py`, `migrations/`: SQLAlchemy persistence and Alembic schema baseline.
- `frontend/`: React, TypeScript, Vite dashboard showing backend/database results, model evaluation, replay status, events, and dataset provenance.
- `docker-compose.yml`: PostgreSQL, Mosquitto, API, prediction worker, and web frontend.

## Local setup (Windows PowerShell)

Python 3.12 is the supported container/CI runtime. The inspected developer environment uses Python 3.14 and already has the ML packages installed; the pinned dependency set should be installed in a dedicated virtual environment before use.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Update `.env` with a unique `ADMIN_API_KEY`. For local training, `DATABASE_URL` defaults to SQLite. Validate the actual recordings and channel pairing:

```powershell
python -m app.cli validate
```

Extract features, compare the reference and candidate models, run FLAML for up to 20 seconds, and evaluate the selected candidate on held-out recordings:

```powershell
python -m app.cli train
# To skip FLAML (for example, when investigating an AutoML compatibility error):
python -m app.cli train --no-automl
```

The first run reads the large original recording files and may take several minutes. Outputs are written to ignored `reports/` and `artifacts/` directories, and the held-out feature rows used by replay are kept separately in `artifacts/test_features.csv.gz`. Inspect `reports/latest.json` for the measured leaderboard, split counts, prediction distribution, confusion matrix, per-class recall, dependency versions, AutoML status, dataset fingerprint, and artifact checksum.

Candidate training does **not** deploy a model automatically. Deploy the validated winner only after reviewing its report:

```powershell
python -m app.cli deploy --version latest
```

Run database migrations and start the API:

```powershell
alembic upgrade head
python -m app.cli api
```

The additive `0002_reliability_metadata` migration adds calibration, decision, explanation, and human-review fields to prediction records. It preserves existing prediction history. Run it before starting the API against an existing database; do not delete the database to resolve a migration error.

In a second terminal, run the React dashboard:

```powershell
Set-Location frontend
npm ci
npm run dev
```

Open `http://localhost:5173`; API documentation is at `http://localhost:8000/docs`. Protected HTTP operations require `X-Admin-API-Key`; the replay page accepts the key in memory and does not save it in browser storage.

For local development, copy `.env.example` to `.env` and replace `ADMIN_API_KEY` with a long random value. The dashboard's development proxy forwards API requests to `127.0.0.1:8000`; keep the backend running while using the dashboard.

### MQTT replay and prediction worker

The local dashboard/API run without an MQTT broker, but replay publishing and prediction ingestion require a broker. Docker Compose starts a development-only Mosquitto broker and restricts its published port to localhost:

```powershell
docker compose up -d mqtt
```

Start the independently managed worker after deploying a model:

```powershell
python -m app.mqtt_worker
```

The replay page defaults to continuous local simulation, which repeatedly cycles through all selected held-out sensor features, runs the loaded model, and persists real predictions until stopped. The default unfiltered test set includes source recordings from all five classes. Windows are interleaved across recordings so readings from each available class appear throughout each cycle rather than in one long class-ordered block. It displays source recording classes separately from the model's actual predicted-class counts; these differ when the model misclassifies a source condition. Continuous runs keep adding prediction records to the configured database; stop the replay when it is no longer needed. MQTT publishing remains available as an explicit mode: the API page starts, pauses, resumes, and stops a QoS 1 publisher over `machine/features`. Prediction results are persisted before acknowledgement and published to `machine/predictions`; broker and validation events go to the database and `machine/events`. Malformed messages are recorded as invalid-message events and acknowledged after the error is persisted. Database failures are not acknowledged so QoS 1 delivery can retry. Message IDs are unique and duplicate deliveries do not create extra prediction rows.

The dashboard formats all timestamps in `Asia/Kolkata` (IST), independent of the browser computer's local timezone. API timestamps are normalized to UTC before serialization, including timestamps read from SQLite.

Replay always reads the selected model's held-out test feature table; it never sends the ground-truth label to the inference topic. The test split excludes all windows from the same recording used in training or validation. Replay timestamps identify playback time, while the original recording/window identifiers preserve source traceability.

#### Terminal simulator (without dashboard, API, or MQTT)

If dashboard replay is unavailable, the CLI can send held-out sensor readings directly through the active model and persist each prediction to the configured database. Keep the API stopped to avoid conflicting database activity, open a VS Code terminal at the repository root, and run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.cli simulate --count 100 --interval 0.2
```

The command defaults to 100 real held-out readings at 0.2-second intervals. Each result is logged with its source recording class, model-predicted class, and reliability decision; the source class is never passed to inference. For continuous operation, stop with Ctrl+C:

```powershell
python -m app.cli simulate --continuous --interval 0.2
```

Use `--machine-id motor-test-01` to set the persisted machine ID. Predictions are stored as replay records and appear in dashboard history on refresh. This is a local dataset replay, not live physical sensor data or random synthetic values. It requires a deployed active model and `artifacts/test_features.csv.gz`; if no active model exists, deploy an evaluated candidate explicitly before running it.

The reliability/research API is also available directly:

- `GET /api/reliability/status` and `GET /api/reliability/reviews` report decision counts, drift status, and pending review items.
- `GET /api/predictions/{prediction_id}/explanation` returns stored probabilities, decision metadata, and the model explanation.
- `GET /api/research/cross-load` and `GET /api/research/degradation` return the latest offline evaluation reports.
- `POST /api/admin/predictions/{prediction_id}/review` records a confirmed condition separately from the model output and requires the `X-Admin-API-Key` header. The JSON body requires `review_label` and `reviewed_by`; `review_note` is optional.

## Docker Compose

Docker Desktop must be running. Copy `.env.example` to `.env`, set a long random `ADMIN_API_KEY` and `POSTGRES_PASSWORD`, and set `DATASET_HOST_PATH` if the dataset is not at the default repository-relative path. Use a URL-safe password with the sample Compose URL construction or supply a correctly URL-encoded `DATABASE_URL`.

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f api prediction-worker
```

After the services are healthy, run training in the API container and deploy through the protected API or CLI:

```powershell
docker compose exec api python -m app.cli validate
docker compose exec api python -m app.cli train
```

Then, using the model version reported in `reports/latest.json`, deploy via the authenticated endpoint (or CLI and restart API/worker so they load the active pointer):

```powershell
$key = "the value configured as ADMIN_API_KEY"
$version = (Get-Content .\reports\latest.json | ConvertFrom-Json).model_version
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/admin/models/$version/deploy" -Headers @{"X-Admin-API-Key"=$key}
docker compose restart prediction-worker
```

Dashboard: `http://localhost:8080`; API docs: `http://localhost:8000/docs`.

Stop services without deleting persistent data:

```powershell
docker compose down
```

Do **not** use `docker compose down -v` unless intentionally deleting the PostgreSQL and Mosquitto volumes.

### Database backup and restore

For Compose PostgreSQL:

```powershell
docker compose exec -T database pg_dump -U motor_monitoring motor_monitoring > .\motor_monitoring-backup.sql
Get-Content .\motor_monitoring-backup.sql | docker compose exec -T database psql -U motor_monitoring -d motor_monitoring
```

Protect database dumps, model bundles, and `.env` as operational data. The SQLite development database is `data/app.db`; stop the API before copying it for a consistent standalone backup.

## Model methodology and safeguards

- Each one-second window uses four vibration channels, two temperature channels, and the non-empty current channel shared across all recordings. Features are deterministic time-domain statistics plus dominant frequency and spectral centroid for vibration/current. Training aborts if any discovered recording or required class yields no windows.
- Recording identifiers—not individual windows—are split into disjoint, class-stratified training, validation, and test groups. Because Normal has only three source recordings, the fixed split assigns one to each partition (resulting in approximately 56/22/22% by recording); splitting only windows or random stratified holdouts could leave a class absent from a partition. Scalers/imputers are fitted inside each model pipeline on training rows only. The AutoML search uses one aggregate row per training recording to avoid window leakage in its internal holdout.
- Validation macro-F1 selects the model; alphabetical algorithm name breaks exact ties. The test partition is used only for the final selected-model evaluation.
- The leaderboard includes a Dummy prior baseline, Logistic Regression, RBF SVM, Random Forest, Extra Trees, histogram gradient boosting, and LightGBM/XGBoost when they run successfully. FLAML is bounded to 20 seconds. Candidate incompatibilities are reported rather than hidden. It records compressed serialized model size as well as fit time and validation metrics.
- The bundle records ordered feature names, canonical class names, schema version, training dataset fingerprint, and version. Loading verifies the artifact path/checksum/schema. Training creates a `validated` model; deployment is an explicit action and previous deployed versions remain available for rollback.
- The validation recording groups fit a single-parameter temperature scaler and a confidence threshold targeting the configured selective accuracy (`SELECTIVE_TARGET_ACCURACY`, default 0.90). If validation cannot meet the target with the configured minimum number of accepted rows (`SELECTIVE_MINIMUM_ACCEPTED`, default 10), the threshold abstains and results require review. The test report separately measures calibration and selective coverage/risk.
- Sensor quality flags channels with exactly zero standard deviation and peak-to-peak range. Training feature quantiles define out-of-distribution bounds. A rolling population stability index (PSI) checks feature drift after 20 rows; its threshold comes from validation recordings. These are screening signals, not proof of sensor failure or model performance decay.
- Explanations use SHAP TreeExplainer when compatible, otherwise single-feature median-ablation sensitivities. The fallback is non-additive and neither causal nor a guarantee of physical importance. Prediction decisions, uncertainty/reliability metadata, and human review labels are persisted separately.
- Training reports include leave-one-load-out folds over available dataset torque loads and reproducible Gaussian-noise/channel-dropout experiments over copies of the untouched test split. Inspect these results and limitations before deploying a newly trained candidate. The existing active model is not silently replaced during training.
- Dashboard `Reliability` shows calibration, load generalization, degradation experiments, feature drift, and pending review decisions. The prediction history expands stored probabilities and explanations. Human review requires the shared administrator key and records a separate review label; it does not overwrite the model result.
- The API reports stale input, feature-distribution drift, and reliability decisions from persisted inference inputs. It does not infer physical fault severity, maintenance need, or live accuracy degradation without ground-truth labels.

### Verified workspace training run

The saved `reports/latest.json` is the source of truth. In the run bundled in this workspace, FLAML completed its 20-second search (`lgbm` best estimator) and the validation macro-F1 winner was Random Forest:

| Candidate | Validation macro-F1 | Compressed model size (KiB) |
|---|---:|---:|
| Random Forest | 0.7033 | 110.0 |
| Logistic Regression | 0.7027 | 5.0 |
| XGBoost | 0.6976 | 60.1 |
| HistGradientBoosting | 0.6815 | 144.3 |
| SVM (RBF) | 0.6414 | 150.4 |
| LightGBM | 0.6328 | 692.0 |
| Extra Trees | 0.5763 | 212.8 |
| FLAML (LightGBM) | 0.4697 | 92.2 |
| Dummy prior baseline | 0.0960 | 1.5 |

The independent, recording-level test partition contained all five classes (10 recordings, 960 one-second windows). The selected model scored macro-F1 **0.7714**, accuracy **0.8750**, and balanced accuracy **0.8000**. Per-class recall was Normal **0.0000**, BPFI **1.0000**, BPFO **1.0000**, Misalignment **1.0000**, and Unbalance **1.0000**. Across the 10 held-out recordings, accuracy averaged **0.9000** with standard deviation **0.3000**, ranging from **0.0000** to **1.0000**. The model produced no Normal predictions, classifying the held-out Normal recording as Unbalance. Although the model did not collapse to one output class, this Normal-class failure is a material limitation; the local artifact was activated only to exercise model loading/API/dashboard integration and is **not approved as a production maintenance model**. Only one Normal recording is available in each partition, so this estimate is especially uncertain.

The latest reliability research candidate (`model-20261028T052534430071Z`) uses the same dataset fingerprint and recording split, and remains **validated but not deployed**. Its held-out calibration metrics were ECE **0.4090**, Brier **0.5594**, and negative log-likelihood **1.1239**. Validation did not find an acceptance threshold meeting the configured 90% selective-accuracy target, so the policy correctly abstains: test coverage was **0%** and all 960 test windows were referred for review. Leave-one-load-out macro-F1/accuracy were **0.6918/0.7210** at 0 Nm, **0.7818/0.9167** at 2 Nm, and **0.5000/0.5000** at 4 Nm. In deterministic degradation tests, adding Gaussian noise at 0.5 training standard deviations reduced macro-F1 by **0.1038**; dropping vibration channel 1 reduced it by **0.3212**. These limited-data results do not resolve the Normal-class failure and are an explicit reason not to deploy the candidate. Full per-scenario metrics and protocol metadata are in `reports/latest.json`.

## Security and operations

- Administrative prediction, training, deployment, rollback, replay, and acknowledgement routes require the backend `ADMIN_API_KEY`; unset means protected actions return 503. This is a single shared admin credential, **not** a user-account system or granular role-based authorization.
- The example Mosquitto configuration permits anonymous connections for localhost development only. Do not expose it to a network. For deployment, disable anonymous access, configure broker users and ACLs, mount private credential/TLS files securely, and set MQTT authentication/TLS environment variables.
- Restrict API exposure, set a production CORS allowlist, protect PostgreSQL/network access, rotate credentials, and terminate HTTPS at a trusted reverse proxy. Secrets are environment-supplied and are not embedded in images.
- Joblib artifacts are executable Python serialization. Only load artifacts created and checksummed by this trusted local training workflow; never accept user-uploaded bundles.
- The dashboard uses no external font/CDN dependency. No physical sensors were available; no hardware acceptance or industrial safety certification is claimed.

## Tests and verification

```powershell
python -m pytest -q
Set-Location frontend
npm run build
```

The automated tests cover deterministic features, malformed data, source-label normalization, class coverage, recording-group split isolation, reliability calibration and drift, cross-load/degradation protocols, API response schemas, dataset/health responses, and disabled-by-default admin actions. The complete replay-to-PostgreSQL-to-dashboard path additionally requires a running broker, database, a trained/deployed artifact, and a browser/container environment. Docker health and service-to-service integration must be checked on a machine with a running Docker daemon.

## Current limits

- Publisher documentation is absent in the provided workspace. The canonical names and sensor metadata are verified against local recording identifiers and MAT/TDMS metadata, but not independently against the dataset publication.
- Only five acoustic recordings exist and no acoustic recordings exist for Misalignment/Unbalance; acoustic is excluded.
- Data labels describe recorded operating conditions. This classifier is not validated for future failure timing, remaining life, other machines, or field deployment.
- No physical machine, live sensor hardware, notification provider, user-account/role service, or ground-truth-driven online performance monitoring workflow is available in this workspace.
- PostgreSQL/Mosquitto/frontend Compose definitions are provided, but containers cannot be claimed healthy until Docker is running and the full stack is started and verified.
#   I O T - P R E D I C T I O N  
 