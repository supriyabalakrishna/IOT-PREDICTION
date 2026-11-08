from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import hmac
import json
import logging
import threading

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, text

from app.config import get_settings
from app.dataset import LABELS, dataset_manifest
from app.features import FEATURE_NAMES
from app.inference import MessageRejected, explain_stored_prediction, process_feature_message, store_event
from app.replay import replay_controller
from app.reliability import population_drift
from app.schemas import FeatureMessage, HumanReviewRequest, ReplayStartRequest
from app.storage import AppEvent, ModelVersion, Prediction, SessionLocal, TrainingJob, initialize_database, utc_now
from app.training import load_active_bundle, load_bundle, train_and_evaluate

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
settings = get_settings()
training_lock = threading.Lock()
active_bundle: dict[str, object] | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global active_bundle
    initialize_database()
    active_bundle = load_active_bundle()
    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "Classification of recorded machine operating conditions. This service does not "
        "estimate remaining useful life or forecast a failure date."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[value.strip() for value in settings.cors_origins.split(",") if value.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-API-Key"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    detail = json.dumps(exc.errors(), default=str)
    try:
        store_event("invalid_message", f"{request.method} {request.url.path}: {detail}", "warning")
    except Exception:
        LOGGER.exception("Could not persist rejected request validation event")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def require_admin(x_admin_api_key: str | None = Header(default=None)):
    expected = get_settings().admin_api_key
    if not expected:
        raise HTTPException(status_code=503, detail="Administrative actions are disabled: ADMIN_API_KEY is unset")
    if not x_admin_api_key or not hmac.compare_digest(x_admin_api_key, expected):
        raise HTTPException(status_code=401, detail="Valid administrator credentials are required")


def _utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _prediction_json(row: Prediction) -> dict[str, object]:
    return {
        "id": row.id,
        "message_id": row.message_id,
        "machine_id": row.machine_id,
        "source": row.source,
        "recording_id": row.recording_id,
        "window_index": row.window_index,
        "predicted_condition": row.predicted_condition,
        "probabilities": row.probabilities,
        "calibrated_probabilities": row.calibrated_probabilities,
        "reliability": row.reliability,
        "decision_status": row.decision_status,
        "explanation": row.explanation,
        "human_review": {
            "review_label": row.review_label,
            "reviewed_by": row.reviewed_by,
            "review_note": row.review_note,
            "reviewed_at": _utc_iso(row.reviewed_at),
        },
        "model_version": row.model_version,
        "schema_version": row.schema_version,
        "event_timestamp": _utc_iso(row.event_timestamp),
        "received_at": _utc_iso(row.received_at),
        "latency_ms": row.latency_ms,
        "features": row.features,
    }


@app.get("/health/live")
def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
def readiness():
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        database_ready = True
    except Exception as exc:
        LOGGER.exception("Database readiness check failed")
        database_ready = False
        database_error = str(exc)
    finally:
        db.close()
    ready = database_ready and active_bundle is not None
    result = {
        "status": "ready" if ready else "not_ready",
        "database": "ready" if database_ready else "unavailable",
        "model": str(active_bundle["model_version"]) if active_bundle else "not_deployed",
    }
    if not database_ready:
        result["database_error"] = database_error
    if not ready:
        raise HTTPException(status_code=503, detail=result)
    return result


@app.get("/api/overview")
def overview():
    db = SessionLocal()
    try:
        latest = db.query(Prediction).order_by(Prediction.received_at.desc()).first()
        total = db.query(func.count(Prediction.id)).scalar() or 0
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_count = db.query(func.count(Prediction.id)).filter(
            Prediction.received_at >= recent_cutoff
        ).scalar() or 0
        machine_count = db.query(func.count(func.distinct(Prediction.machine_id))).scalar() or 0
        latest_broker_event = db.query(AppEvent).filter(
            AppEvent.event_type.in_(("broker_connected", "broker_disconnected"))
        ).order_by(AppEvent.created_at.desc()).first()
        broker_state = (
            "connected" if latest_broker_event and latest_broker_event.event_type == "broker_connected"
            else "disconnected" if latest_broker_event
            else "unknown"
        )
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.stale_after_seconds)
        stale = latest is None or latest.received_at.replace(tzinfo=timezone.utc) < cutoff
        recent = db.query(Prediction).order_by(Prediction.received_at.desc()).limit(8).all()
        events = db.query(AppEvent).order_by(AppEvent.created_at.desc()).limit(5).all()
        return {
            "registered_machines": machine_count,
            "prediction_count": total,
            "predictions_last_24_hours": recent_count,
            "latest_prediction": _prediction_json(latest) if latest else None,
            "last_data_stale": stale,
            "stale_after_seconds": settings.stale_after_seconds,
            "broker_status": broker_state,
            "model_version": str(active_bundle["model_version"]) if active_bundle else None,
            "recent_predictions": [_prediction_json(row) for row in recent],
            "recent_events": [
                {
                    "id": event.id,
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "message": event.message,
                    "created_at": _utc_iso(event.created_at),
                    "acknowledged": event.acknowledged_at is not None,
                }
                for event in events
            ],
        }
    finally:
        db.close()


@app.get("/api/dataset")
def dataset_info():
    try:
        return dataset_manifest(settings.dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/models")
def models():
    db = SessionLocal()
    try:
        rows = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
        return [
            {
                "version": row.version,
                "algorithm": row.algorithm,
                "status": row.status,
                "checksum": row.checksum,
                "metrics": json.loads(row.metrics_json),
                "created_at": _utc_iso(row.created_at),
            }
            for row in rows
        ]
    finally:
        db.close()


@app.get("/api/models/comparison")
def model_comparison():
    latest = settings.report_dir / "latest.json"
    if not latest.exists():
        return {"status": "not_trained", "comparison": [], "test_metrics": None}
    report = json.loads(latest.read_text(encoding="utf-8"))
    return {
        "status": report["status"],
        "selected_model": report["selected_model"],
        "selection_metric": report["selection_metric"],
        "comparison": report["comparison"],
        "test_metrics": report["test_metrics"],
        "feature_names": report.get("feature_names", []),
        "automl": report["automl"],
        "recording_splits": report["recording_splits"],
        "class_collapse_detected": report["class_collapse_detected"],
        "test_classes_missing_from_predictions": report.get("test_classes_missing_from_predictions", []),
        "reliability": report.get("reliability"),
        "cross_load_evaluation": report.get("cross_load_evaluation"),
        "sensor_degradation_evaluation": report.get("sensor_degradation_evaluation"),
    }


@app.get("/api/research/cross-load")
def cross_load_results():
    latest = settings.report_dir / "latest.json"
    if not latest.exists():
        return {"status": "not_trained", "folds": []}
    report = json.loads(latest.read_text(encoding="utf-8"))
    return report.get("cross_load_evaluation", {"status": "unavailable", "folds": []})


@app.get("/api/research/degradation")
def sensor_degradation_results():
    latest = settings.report_dir / "latest.json"
    if not latest.exists():
        return {"status": "not_trained", "scenarios": []}
    report = json.loads(latest.read_text(encoding="utf-8"))
    return report.get("sensor_degradation_evaluation", {"status": "unavailable", "scenarios": []})


@app.get("/api/reliability/status")
def reliability_status():
    db = SessionLocal()
    try:
        total = db.query(func.count(Prediction.id)).scalar() or 0
        review_count = db.query(func.count(Prediction.id)).filter(
            Prediction.decision_status.in_(("REVIEW_REQUIRED", "DEGRADED_SENSOR_DATA", "OUT_OF_DISTRIBUTION"))
        ).scalar() or 0
        reviewed_count = db.query(func.count(Prediction.id)).filter(Prediction.reviewed_at.is_not(None)).scalar() or 0
        decision_counts = dict(
            db.query(Prediction.decision_status, func.count(Prediction.id))
            .group_by(Prediction.decision_status).all()
        )
        reference = active_bundle.get("reliability_reference") if active_bundle else None
        if reference:
            recent = db.query(Prediction).order_by(Prediction.received_at.desc()).limit(
                int(reference["drift_window_size"])
            ).all()
            drift = population_drift(
                [row.features for row in reversed(recent)],
                reference,
            )
        else:
            drift = {"status": "unavailable", "window_size": 0, "psi_max": None}
        return {
            "status": "available" if active_bundle else "model_unavailable",
            "model_version": str(active_bundle["model_version"]) if active_bundle else None,
            "prediction_count": total,
            "decision_counts": {key: int(value) for key, value in decision_counts.items()},
            "review_required_count": int(review_count),
            "human_reviewed_count": int(reviewed_count),
            "drift": drift,
        }
    finally:
        db.close()


@app.get("/api/reliability/reviews")
def reliability_reviews(limit: int = Query(default=50, ge=1, le=500)):
    db = SessionLocal()
    try:
        rows = (
            db.query(Prediction)
            .filter(Prediction.decision_status != "ACCEPTED", Prediction.reviewed_at.is_(None))
            .order_by(Prediction.received_at.desc())
            .limit(limit)
            .all()
        )
        return {"items": [_prediction_json(row) for row in rows]}
    finally:
        db.close()


@app.get("/api/predictions")
def predictions(
    machine_id: str | None = None,
    condition: str | None = Query(default=None, pattern="^(Normal|BPFI|BPFO|Misalignment|Unbalance)$"),
    source: str | None = Query(default=None, pattern="^(live|replay)$"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    db = SessionLocal()
    try:
        query = db.query(Prediction)
        if machine_id:
            query = query.filter(Prediction.machine_id == machine_id)
        if condition:
            query = query.filter(Prediction.predicted_condition == condition)
        if source:
            query = query.filter(Prediction.source == source)
        total = query.count()
        rows = query.order_by(Prediction.received_at.desc()).offset(offset).limit(limit).all()
        return {"total": total, "limit": limit, "offset": offset, "items": [_prediction_json(row) for row in rows]}
    finally:
        db.close()


@app.get("/api/predictions/{prediction_id}/explanation")
def prediction_explanation(prediction_id: int):
    db = SessionLocal()
    try:
        row = db.get(Prediction, prediction_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        explanation = row.explanation
        explanation_status = "stored" if explanation.get("top_features") else "unavailable"
        if not explanation.get("top_features") and active_bundle is not None:
            explanation = explain_stored_prediction(active_bundle, row) or {}
            if explanation.get("top_features"):
                explanation_status = "computed_for_compatible_legacy_prediction"
                db.commit()
        if not explanation.get("top_features"):
            explanation_status = "unavailable_for_model_version"
        return {
            "prediction_id": row.id,
            "predicted_condition": row.predicted_condition,
            "probabilities": row.probabilities,
            "calibrated_probabilities": row.calibrated_probabilities,
            "decision_status": row.decision_status,
            "reliability": row.reliability,
            "explanation": explanation,
            "explanation_status": explanation_status,
            "model_version": row.model_version,
            "source": row.source,
            "recording_id": row.recording_id,
            "window_index": row.window_index,
            "received_at": _utc_iso(row.received_at),
        }
    finally:
        db.close()


@app.post("/api/admin/predictions/{prediction_id}/review", dependencies=[Depends(require_admin)])
def review_prediction(prediction_id: int, request: HumanReviewRequest):
    db = SessionLocal()
    try:
        row = db.get(Prediction, prediction_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        if row.reviewed_at is not None:
            raise HTTPException(status_code=409, detail="This prediction has already been reviewed")
        row.review_label = request.review_label
        row.reviewed_by = request.reviewed_by
        row.review_note = request.review_note
        row.reviewed_at = utc_now()
        db.add(AppEvent(
            event_type="human_review_recorded",
            message=f"Prediction {row.id} reviewed by {request.reviewed_by}: {request.review_label}",
            severity="info",
            created_at=utc_now(),
        ))
        db.commit()
        db.refresh(row)
        return {"status": "review_recorded", "prediction": _prediction_json(row)}
    finally:
        db.close()


@app.get("/api/machines")
def machines():
    db = SessionLocal()
    try:
        latest_ids = (
            db.query(func.max(Prediction.id).label("prediction_id"))
            .group_by(Prediction.machine_id)
            .subquery()
        )
        latest_by_machine = db.query(Prediction).join(
            latest_ids, Prediction.id == latest_ids.c.prediction_id
        ).all()
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.stale_after_seconds)
        return {
            "items": [
                {
                    "machine_id": row.machine_id,
                    "last_seen": _utc_iso(row.received_at),
                    "status": "stale" if row.received_at.replace(tzinfo=timezone.utc) < cutoff else "active",
                    "source": row.source,
                    "condition": row.predicted_condition,
                    "model_version": row.model_version,
                }
                for row in latest_by_machine
            ]
        }
    finally:
        db.close()


@app.get("/api/condition-distribution")
def condition_distribution(source: str | None = Query(default=None, pattern="^(live|replay)$")):
    db = SessionLocal()
    try:
        query = db.query(Prediction.predicted_condition, func.count(Prediction.id))
        if source:
            query = query.filter(Prediction.source == source)
        counts = dict(query.group_by(Prediction.predicted_condition).all())
        return {"conditions": {label: int(counts.get(label, 0)) for label in LABELS}}
    finally:
        db.close()


@app.get("/api/events")
def events(
    event_type: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    db = SessionLocal()
    try:
        query = db.query(AppEvent)
        if event_type:
            query = query.filter(AppEvent.event_type == event_type)
        total = query.count()
        rows = query.order_by(AppEvent.created_at.desc()).offset(offset).limit(limit).all()
        return {
            "total": total,
            "items": [
                {
                    "id": row.id,
                    "event_type": row.event_type,
                    "severity": row.severity,
                    "message": row.message,
                    "created_at": _utc_iso(row.created_at),
                    "acknowledged": row.acknowledged_at is not None,
                }
                for row in rows
            ],
        }
    finally:
        db.close()


@app.post("/api/predict", dependencies=[Depends(require_admin)])
def predict(message: FeatureMessage):
    if active_bundle is None:
        raise HTTPException(status_code=503, detail="No model is deployed")
    try:
        row, duplicate = process_feature_message(active_bundle, message.model_dump(mode="json"))
    except MessageRejected as exc:
        store_event("invalid_message", f"Rejected feature message: {exc}", "warning")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"duplicate": duplicate, "prediction": _prediction_json(row)}


@app.get("/api/replay")
def replay_status():
    return replay_controller.status()


@app.post("/api/replay/start", dependencies=[Depends(require_admin)])
def replay_start(request: ReplayStartRequest):
    if active_bundle is None:
        raise HTTPException(status_code=503, detail="Deploy a validated model before starting replay")
    try:
        return replay_controller.start(
            request.machine_id,
            request.recording_ids,
            request.speed,
            request.delay_seconds,
            request.mode,
            request.continuous,
            active_bundle,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/replay/pause", dependencies=[Depends(require_admin)])
def replay_pause():
    return replay_controller.pause()


@app.post("/api/replay/resume", dependencies=[Depends(require_admin)])
def replay_resume():
    return replay_controller.resume()


@app.post("/api/replay/stop", dependencies=[Depends(require_admin)])
def replay_stop():
    return replay_controller.stop()


@app.get("/api/training-jobs/{job_id}")
def training_job(job_id: int):
    db = SessionLocal()
    try:
        row = db.get(TrainingJob, job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Training job not found")
        return {
            "id": row.id,
            "status": row.status,
            "started_at": _utc_iso(row.started_at),
            "finished_at": _utc_iso(row.finished_at),
            "result": json.loads(row.result_json) if row.result_json else None,
            "error": row.error,
        }
    finally:
        db.close()


@app.get("/api/training-jobs")
def training_jobs(limit: int = Query(default=50, ge=1, le=500), offset: int = Query(default=0, ge=0)):
    db = SessionLocal()
    try:
        rows = db.query(TrainingJob).order_by(TrainingJob.started_at.desc()).offset(offset).limit(limit).all()
        return {
            "items": [
                {
                    "id": row.id,
                    "status": row.status,
                    "started_at": _utc_iso(row.started_at),
                    "finished_at": _utc_iso(row.finished_at),
                    "error": row.error,
                }
                for row in rows
            ]
        }
    finally:
        db.close()


def _run_training_job(job_id: int, run_automl: bool):
    db = SessionLocal()
    try:
        result = train_and_evaluate(run_automl=run_automl)
        row = db.get(TrainingJob, job_id)
        row.status = "completed"
        row.finished_at = utc_now()
        row.result_json = json.dumps({
            "model_version": result["model_version"],
            "selected_model": result["selected_model"],
            "test_metrics": result["test_metrics"],
        })
        db.commit()
    except Exception as exc:
        db.rollback()
        LOGGER.exception("Training job %s failed", job_id)
        row = db.get(TrainingJob, job_id)
        if row:
            row.status = "failed"
            row.finished_at = utc_now()
            row.error = f"{type(exc).__name__}: {exc}"
            db.commit()
        store_event("training_failed", f"Training job {job_id} failed: {exc}", "error")
    finally:
        db.close()
        training_lock.release()


@app.post("/api/admin/training-jobs", dependencies=[Depends(require_admin)], status_code=202)
def start_training(run_automl: bool = True):
    if not training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A training job is already running")
    db = SessionLocal()
    try:
        row = TrainingJob(status="running", started_at=utc_now())
        db.add(row)
        db.commit()
        db.refresh(row)
        thread = threading.Thread(
            target=_run_training_job,
            args=(row.id, run_automl),
            name=f"training-job-{row.id}",
            daemon=True,
        )
        thread.start()
        return {"id": row.id, "status": "running"}
    except Exception:
        training_lock.release()
        db.rollback()
        raise
    finally:
        db.close()


@app.post("/api/admin/models/{version}/deploy", dependencies=[Depends(require_admin)])
def deploy_model(version: str):
    global active_bundle
    db = SessionLocal()
    try:
        row = db.query(ModelVersion).filter_by(version=version).one_or_none()
        if row is None or row.status not in {"validated", "retired", "deployed"}:
            raise HTTPException(status_code=404, detail="Validated model version not found")
        bundle = load_bundle(version)
        previous = db.query(ModelVersion).filter_by(status="deployed").all()
        previous_version = previous[0].version if previous else None
        for old in previous:
            old.status = "retired"
        row.status = "deployed"
        db.add(AppEvent(
            event_type="model_deployed",
            message=f"Deployed {version} ({row.algorithm})",
            severity="warning",
        ))
        pointer = settings.artifact_dir / "active.json"
        pointer_tmp = pointer.with_suffix(".tmp")
        pointer_tmp.write_text(json.dumps({"version": version}), encoding="utf-8")
        pointer_tmp.replace(pointer)
        try:
            db.commit()
        except Exception:
            db.rollback()
            if previous_version:
                restore = pointer.with_suffix(".restore")
                restore.write_text(json.dumps({"version": previous_version}), encoding="utf-8")
                restore.replace(pointer)
            else:
                pointer.unlink(missing_ok=True)
            raise
        active_bundle = bundle
        return {"status": "deployed", "model_version": version, "previous_versions": [old.version for old in previous]}
    finally:
        db.close()


@app.post("/api/admin/models/rollback", dependencies=[Depends(require_admin)])
def rollback_model():
    db = SessionLocal()
    try:
        previous = db.query(ModelVersion).filter_by(status="retired").order_by(ModelVersion.created_at.desc()).first()
        if previous is None:
            raise HTTPException(status_code=409, detail="No previous validated model is available for rollback")
        version = previous.version
    finally:
        db.close()
    return deploy_model(version)


@app.post("/api/admin/events/{event_id}/acknowledge", dependencies=[Depends(require_admin)])
def acknowledge_event(event_id: int):
    db = SessionLocal()
    try:
        event = db.get(AppEvent, event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        event.acknowledged_at = utc_now()
        db.commit()
        return {"id": event_id, "acknowledged": True}
    finally:
        db.close()
