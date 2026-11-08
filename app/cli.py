from __future__ import annotations

import argparse
import json
import logging

from app.config import get_settings
from app.dataset import LABELS, dataset_manifest, discover_recordings
from app.simulator import simulate_predictions
from app.storage import AppEvent, ModelVersion, SessionLocal, initialize_database, utc_now
from app.training import load_active_bundle, load_bundle, train_and_evaluate


def validate_dataset() -> dict[str, object]:
    settings = get_settings()
    report = dataset_manifest(settings.dataset_path)
    recordings = discover_recordings(settings.dataset_path)
    if not recordings:
        raise RuntimeError("No vibration recordings were discovered")
    missing = [label for label in LABELS if report["recordings_per_class"].get(label, 0) == 0]
    unmatched = [r.recording_id for r in recordings if r.tdms_path is None]
    if missing or unmatched:
        raise RuntimeError(f"Dataset validation failed; missing classes={missing}, unmatched recordings={unmatched}")
    report["validation_status"] = "passed"
    report["recording_id_labels"] = [
        {"recording_id": r.recording_id, "label": r.label, "load_nm": r.load_nm}
        for r in recordings
    ]
    (settings.report_dir / "dataset_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def deploy(version: str) -> dict[str, object]:
    settings = get_settings()
    initialize_database()
    if version == "latest":
        report_path = settings.report_dir / "latest.json"
        if not report_path.exists():
            raise FileNotFoundError("No trained model report exists")
        version = json.loads(report_path.read_text(encoding="utf-8"))["model_version"]
    bundle = load_bundle(version)
    db = SessionLocal()
    pointer = settings.artifact_dir / "active.json"
    previous_version = None
    pointer_written = False
    try:
        model = db.query(ModelVersion).filter_by(version=version).one_or_none()
        if model is None or model.status not in {"validated", "retired", "deployed"}:
            raise ValueError(f"No validated registry entry for model {version}")
        previous = db.query(ModelVersion).filter_by(status="deployed").all()
        previous_version = previous[0].version if previous else None
        for old in previous:
            old.status = "retired"
        model.status = "deployed"
        db.add(AppEvent(
            event_type="model_deployed",
            message=f"Deployed {version} ({model.algorithm}) through CLI",
            severity="warning",
            created_at=utc_now(),
        ))
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": version}), encoding="utf-8")
        temporary.replace(pointer)
        pointer_written = True
        db.commit()
    except Exception:
        db.rollback()
        if pointer_written:
            if previous_version:
                restore = pointer.with_suffix(".restore")
                restore.write_text(json.dumps({"version": previous_version}), encoding="utf-8")
                restore.replace(pointer)
            else:
                pointer.unlink(missing_ok=True)
        raise
    finally:
        db.close()
    return {"status": "deployed", "model_version": bundle["model_version"]}


def main():
    parser = argparse.ArgumentParser(description="Dataset, model training, and deployment commands")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate the configured local dataset manifest")
    train_parser = subparsers.add_parser("train", help="Extract features, compare models, and evaluate on held-out recordings")
    train_parser.add_argument("--no-automl", action="store_true", help="Skip the bounded FLAML search")
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a validated local model by version")
    deploy_parser.add_argument("--version", default="latest")
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run held-out sensor readings through the active model without the dashboard, API, or MQTT",
    )
    simulate_parser.add_argument("--count", type=int, default=100, help="Number of readings (default: 100)")
    simulate_parser.add_argument("--continuous", action="store_true", help="Continue cycling until Ctrl+C")
    simulate_parser.add_argument("--interval", type=float, default=0.2, help="Seconds between readings (default: 0.2)")
    simulate_parser.add_argument("--machine-id", default="terminal-simulator")
    subparsers.add_parser("api", help="Start the FastAPI server")
    args = parser.parse_args()

    if args.command == "validate":
        result = validate_dataset()
    elif args.command == "train":
        result = train_and_evaluate(run_automl=not args.no_automl)
    elif args.command == "deploy":
        result = deploy(args.version)
    elif args.command == "simulate":
        if args.count < 1:
            parser.error("--count must be at least one")
        if args.interval < 0:
            parser.error("--interval cannot be negative")
        bundle = load_active_bundle()
        if bundle is None:
            raise RuntimeError(
                "No active model is deployed. Validate a trained candidate first, then deploy it "
                "with: python -m app.cli deploy --version latest"
            )
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        try:
            result = simulate_predictions(
                bundle,
                machine_id=args.machine_id,
                count=None if args.continuous else args.count,
                interval_seconds=args.interval,
            )
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")
            return
    else:
        import uvicorn

        uvicorn.run("app.api:app", host="0.0.0.0", port=8000)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
