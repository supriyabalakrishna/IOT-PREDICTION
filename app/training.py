from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import platform
import random
import time
from importlib import metadata as package_metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from app.config import get_settings
from app.dataset import LABELS, build_feature_table, dataset_manifest
from app.features import FEATURE_NAMES
from app.reliability import (
    apply_temperature,
    build_reference,
    calibration_metrics,
    fit_temperature,
    model_probability_classes,
    select_acceptance_threshold,
    selective_metrics,
)
from app.research import run_cross_load_evaluation, run_degradation_evaluation

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42


def _installed_version(package: str) -> str:
    try:
        return package_metadata.version(package)
    except package_metadata.PackageNotFoundError:
        return "not installed"


def _split_recordings(frame: pd.DataFrame):
    recordings = (
        frame[["recording_id", "label"]]
        .drop_duplicates()
        .sort_values("recording_id")
        .reset_index(drop=True)
    )
    rng = random.Random(RANDOM_STATE)
    splits = {"train": set(), "validation": set(), "test": set()}
    counts = recordings.groupby("label").recording_id.nunique().to_dict()
    missing = sorted(set(LABELS) - set(counts))
    insufficient = {label: counts.get(label, 0) for label in LABELS if counts.get(label, 0) < 3}
    if missing or insufficient:
        raise ValueError(
            f"At least three independent recordings per class are needed for three-way "
            f"group splitting; missing={missing}, insufficient={insufficient}"
        )
    for label in LABELS:
        ids = recordings.loc[recordings.label == label, "recording_id"].tolist()
        rng.shuffle(ids)
        test_count = max(1, round(len(ids) * 0.2))
        validation_count = max(1, round(len(ids) * 0.2))
        splits["test"].update(ids[:test_count])
        splits["validation"].update(ids[test_count : test_count + validation_count])
        splits["train"].update(ids[test_count + validation_count :])
    return splits


class EncodedLabelClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y):
        self.label_encoder_ = LabelEncoder().fit(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x, self.label_encoder_.transform(y))
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, x):
        encoded = self.estimator_.predict(x).astype(int)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(self, x):
        return self.estimator_.predict_proba(x)


def _pipeline(estimator, scale: bool = False) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def _candidates():
    candidates = {
        "Dummy (prior baseline)": _pipeline(DummyClassifier(strategy="prior")),
        "Logistic Regression": _pipeline(
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
            scale=True,
        ),
        "SVM (RBF)": _pipeline(
            SVC(C=2.0, kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
            scale=True,
        ),
        "Random Forest": _pipeline(
            RandomForestClassifier(
                n_estimators=160, class_weight="balanced_subsample",
                min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE,
            )
        ),
        "Extra Trees": _pipeline(
            ExtraTreesClassifier(
                n_estimators=160, class_weight="balanced",
                min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE,
            )
        ),
        "HistGradientBoosting": _pipeline(
            HistGradientBoostingClassifier(
                max_iter=140, learning_rate=0.08, l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        candidates["LightGBM"] = _pipeline(
            LGBMClassifier(
                n_estimators=160,
                learning_rate=0.06,
                class_weight="balanced",
                verbosity=-1,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        )
    except ImportError:
        LOGGER.warning("Skipping LightGBM candidate: lightgbm is not installed")
    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = _pipeline(
            EncodedLabelClassifier(
                XGBClassifier(
                    n_estimators=120,
                    max_depth=5,
                    learning_rate=0.08,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    verbosity=0,
                )
            )
        )
    except ImportError:
        LOGGER.warning("Skipping XGBoost candidate: xgboost is not installed")
    return candidates


def _metrics(model, x: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    predictions = model.predict(x)
    report = classification_report(
        y, predictions, labels=list(LABELS), output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y, predictions, labels=list(LABELS))
    start = time.perf_counter()
    model.predict(x.iloc[: min(128, len(x))])
    inference_ms = (time.perf_counter() - start) * 1000 / min(128, len(x))
    return {
        "macro_f1": float(f1_score(y, predictions, labels=list(LABELS), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": matrix.tolist(),
        "labels": list(LABELS),
        "prediction_distribution": {
            label: int(np.sum(predictions == label)) for label in LABELS
        },
        "inference_latency_ms_per_row": inference_ms,
    }


def _serialized_model_size(model) -> int:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return buffer.tell()


def train_and_evaluate(dataset_path: Path | None = None, run_automl: bool = True) -> dict[str, object]:
    settings = get_settings()
    root = dataset_path or settings.dataset_path
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading deterministic sensor features from %s", root)
    manifest = dataset_manifest(root)
    cache_path = settings.artifact_dir / "features.csv.gz"
    cache_metadata_path = settings.artifact_dir / "features_manifest.json"
    if cache_path.exists() and cache_metadata_path.exists():
        cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        if (
            cache_metadata.get("fingerprint") == manifest["fingerprint"]
            and cache_metadata.get("feature_names") == FEATURE_NAMES
        ):
            frame = pd.read_csv(cache_path)
            LOGGER.info("Using cached feature table for fingerprint %s", manifest["fingerprint"])
        else:
            frame, manifest = build_feature_table(root)
            frame.to_csv(cache_path, index=False)
            cache_metadata_path.write_text(
                json.dumps({"fingerprint": manifest["fingerprint"], "feature_names": FEATURE_NAMES}),
                encoding="utf-8",
            )
    else:
        frame, manifest = build_feature_table(root)
        frame.to_csv(cache_path, index=False)
        cache_metadata_path.write_text(
            json.dumps({"fingerprint": manifest["fingerprint"], "feature_names": FEATURE_NAMES}),
            encoding="utf-8",
        )
    splits = _split_recordings(frame)
    split_sets = {
        name: frame[frame.recording_id.isin(recording_ids)].copy()
        for name, recording_ids in splits.items()
    }
    for name, split in split_sets.items():
        missing = sorted(set(LABELS) - set(split["label"].unique()))
        if missing:
            raise RuntimeError(f"The {name} split is missing required classes: {missing}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = splits[left] & splits[right]
        if overlap:
            raise RuntimeError(f"Recording leakage between {left} and {right}: {sorted(overlap)}")

    x_train = split_sets["train"][FEATURE_NAMES]
    y_train = split_sets["train"]["label"]
    x_validation = split_sets["validation"][FEATURE_NAMES]
    y_validation = split_sets["validation"]["label"]
    x_test = split_sets["test"][FEATURE_NAMES]
    y_test = split_sets["test"]["label"]
    comparisons: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}
    skipped_candidates: dict[str, str] = {}
    for name, model in _candidates().items():
        try:
            started = time.perf_counter()
            model.fit(x_train, y_train)
            duration = time.perf_counter() - started
            metrics = _metrics(model, x_validation, y_validation)
            comparisons.append({
                "algorithm": name,
                "training_duration_seconds": duration,
                "model_size_bytes": _serialized_model_size(model),
                **metrics,
            })
            fitted[name] = model
            LOGGER.info("%s validation macro-F1=%.4f", name, metrics["macro_f1"])
        except Exception as exc:
            skipped_candidates[name] = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Candidate %s could not be evaluated", name)
    if not comparisons:
        raise RuntimeError(f"No candidate model could be evaluated: {skipped_candidates}")

    automl_status: dict[str, object] = {"status": "not_run"}
    if run_automl:
        try:
            from flaml import AutoML

            automl = AutoML()
            started = time.perf_counter()
            # AutoML sees one row per original recording, preventing windows from the
            # same acquisition from crossing its internal training/validation split.
            group_train = split_sets["train"].groupby("recording_id", sort=True)
            x_group = group_train[FEATURE_NAMES].median()
            y_group = group_train["label"].first()
            automl.fit(
                x_group,
                y_group,
                task="classification",
                metric="macro_f1",
                time_budget=20,
                eval_method="holdout",
                split_type="stratified",
                seed=RANDOM_STATE,
                verbose=0,
            )
            duration = time.perf_counter() - started
            # AutoML.predict applies FLAML's internal target-label encoding; the
            # wrapped estimator exposes integer encodings instead of source labels.
            automl_model = automl
            metrics = _metrics(automl_model, x_validation, y_validation)
            name = f"FLAML ({automl.best_estimator})"
            comparisons.append({
                "algorithm": name,
                "training_duration_seconds": duration,
                "model_size_bytes": _serialized_model_size(automl_model),
                **metrics,
            })
            fitted[name] = automl_model
            automl_status = {"status": "completed", "best_estimator": automl.best_estimator}
        except Exception as exc:
            automl_status = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
            LOGGER.exception("FLAML AutoML did not complete")

    # Validation macro-F1 is the selection metric; stable algorithm name is the tie-break.
    selected = sorted(
        comparisons,
        key=lambda result: (-float(result["macro_f1"]), str(result["algorithm"])),
    )[0]
    selected_name = str(selected["algorithm"])
    model = fitted[selected_name]
    reference = build_reference(
        split_sets["train"],
        split_sets["validation"],
        FEATURE_NAMES,
    )
    probability_classes = model_probability_classes(model, list(LABELS))
    validation_raw_probabilities = model.predict_proba(x_validation)
    calibration_temperature = fit_temperature(
        validation_raw_probabilities,
        y_validation.to_numpy(),
        probability_classes,
    )
    validation_calibrated_probabilities = apply_temperature(
        validation_raw_probabilities,
        calibration_temperature,
    )
    settings = get_settings()
    acceptance_policy = select_acceptance_threshold(
        y_validation.to_numpy(),
        validation_calibrated_probabilities,
        probability_classes,
        target_accuracy=settings.selective_target_accuracy,
        minimum_accepted=settings.selective_minimum_accepted,
    )
    final_metrics = _metrics(model, x_test, y_test)
    test_raw_probabilities = model.predict_proba(x_test)
    test_calibrated_probabilities = apply_temperature(test_raw_probabilities, calibration_temperature)
    final_metrics["raw_probability_calibration"] = calibration_metrics(
        y_test.to_numpy(),
        test_raw_probabilities,
        probability_classes,
    )
    final_metrics["calibrated_probability_metrics"] = calibration_metrics(
        y_test.to_numpy(),
        test_calibrated_probabilities,
        probability_classes,
    )
    final_metrics["selective_prediction"] = selective_metrics(
        y_test.to_numpy(),
        test_calibrated_probabilities,
        probability_classes,
        acceptance_policy["threshold"],
    )
    recording_accuracies = {}
    for recording_id, group in split_sets["test"].groupby("recording_id", sort=True):
        prediction = model.predict(group[FEATURE_NAMES])
        recording_accuracies[str(recording_id)] = {
            "condition": str(group["label"].iloc[0]),
            "accuracy": float(accuracy_score(group["label"], prediction)),
            "window_count": len(group),
        }
    final_metrics["per_recording_accuracy"] = recording_accuracies
    final_metrics["recording_accuracy_mean"] = float(
        np.mean([value["accuracy"] for value in recording_accuracies.values()])
    )
    final_metrics["recording_accuracy_std"] = float(
        np.std([value["accuracy"] for value in recording_accuracies.values()])
    )
    if any(final_metrics["per_class"][label]["support"] == 0 for label in LABELS):
        raise RuntimeError("The independent test partition does not contain examples for every canonical class")
    version = datetime.now(timezone.utc).strftime("model-%Y%m%dT%H%M%S%fZ")
    bundle = {
        "model": model,
        "model_version": version,
        "algorithm": selected_name,
        "feature_names": FEATURE_NAMES,
        "class_names": list(LABELS),
        "schema_version": settings.feature_schema_version,
        "dataset_fingerprint": manifest["fingerprint"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "validation_metrics": selected,
        "calibration_method": "single-parameter temperature scaling fitted on validation recordings",
        "calibration_temperature": calibration_temperature,
        "probability_classes": probability_classes,
        "acceptance_policy": acceptance_policy,
        "reliability_reference": reference,
    }
    cross_load = run_cross_load_evaluation(
        frame,
        model,
        selected_name,
        FEATURE_NAMES,
        final_metrics,
    )
    degradation = run_degradation_evaluation(
        split_sets["train"],
        split_sets["validation"],
        split_sets["test"],
        model,
        FEATURE_NAMES,
        calibration_temperature,
        acceptance_policy,
    )
    artifact_path = settings.artifact_dir / f"{version}.joblib"
    joblib.dump(bundle, artifact_path, compress=3)
    checksum = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    result = {
        "status": "completed",
        "model_version": version,
        "selected_model": selected_name,
        "selection_metric": "validation macro-F1",
        "selection_tie_break": "algorithm name ascending",
        "dataset_fingerprint": manifest["fingerprint"],
        "dataset": manifest,
        "recording_splits": {
            name: {
                "recording_count": len(splits[name]),
                "recording_ids": sorted(splits[name]),
                "recordings_per_class": {
                    label: int(split_sets[name].loc[
                        split_sets[name].label == label, "recording_id"
                    ].nunique())
                    for label in LABELS
                },
                "windows_per_class": {
                    label: int((split_sets[name].label == label).sum())
                    for label in LABELS
                },
            }
            for name in ("train", "validation", "test")
        },
        "comparison": sorted(comparisons, key=lambda item: (-float(item["macro_f1"]), item["algorithm"])),
        "skipped_candidates": skipped_candidates,
        "test_metrics": final_metrics,
        "automl": automl_status,
        "feature_names": FEATURE_NAMES,
        "class_names": list(LABELS),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "flaml": _installed_version("flaml") if run_automl else "not run",
            "lightgbm": _installed_version("lightgbm"),
            "xgboost": _installed_version("xgboost"),
        },
        "artifact": str(artifact_path),
        "artifact_sha256": checksum,
        "artifact_size_bytes": artifact_path.stat().st_size,
        "test_classes_missing_from_predictions": [
            label for label, count in final_metrics["prediction_distribution"].items() if count == 0
        ],
        "test_class_support_complete": all(
            final_metrics["per_class"][label]["support"] > 0 for label in LABELS
        ),
        "class_collapse_detected": sum(
            count > 0 for count in final_metrics["prediction_distribution"].values()
        ) <= 1,
        "reliability": {
            "calibration_method": bundle["calibration_method"],
            "calibration_temperature": calibration_temperature,
            "probability_classes": probability_classes,
            "validation_calibration_metrics": calibration_metrics(
                y_validation.to_numpy(),
                validation_calibrated_probabilities,
                probability_classes,
            ),
            "acceptance_policy": acceptance_policy,
            "ood_feature_count_threshold": reference["ood_feature_count_threshold"],
            "drift_psi_threshold": reference["drift_psi_threshold"],
            "drift_threshold_basis": "95th percentile of per-recording PSI on validation recordings",
            "selective_target_accuracy_config": settings.selective_target_accuracy,
        },
        "cross_load_evaluation": cross_load,
        "sensor_degradation_evaluation": degradation,
    }
    split_sets["test"].to_csv(settings.artifact_dir / "test_features.csv.gz", index=False)
    (settings.report_dir / f"{version}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (settings.report_dir / "latest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (settings.report_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _register_model(version, selected_name, artifact_path, checksum, selected, result)
    return result


def _register_model(version, algorithm, artifact_path, checksum, validation_metrics, result):
    from app.storage import ModelVersion, SessionLocal, initialize_database

    initialize_database()
    db = SessionLocal()
    try:
        db.add(ModelVersion(
            version=version,
            algorithm=algorithm,
            status="validated",
            artifact_path=str(artifact_path),
            checksum=checksum,
            metrics_json=json.dumps({
                "validation": validation_metrics,
                "test": result["test_metrics"],
            }),
        ))
        db.commit()
    finally:
        db.close()


def load_bundle(version: str) -> dict[str, object]:
    settings = get_settings()
    try:
        metadata_path = settings.report_dir / f"{version}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        path = Path(metadata["artifact"]).resolve()
        if not path.is_relative_to(settings.artifact_dir.resolve()):
            raise ValueError("Model artifact is outside the configured artifact directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
            raise ValueError("Model artifact checksum does not match the evaluation report")
        bundle = joblib.load(path)
        if bundle["feature_names"] != FEATURE_NAMES or bundle["class_names"] != list(LABELS):
            raise ValueError("Model feature schema or class mapping is incompatible")
        return bundle
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load validated local model bundle {version}: {exc}") from exc


def load_active_bundle() -> dict[str, object] | None:
    settings = get_settings()
    active = settings.artifact_dir / "active.json"
    if not active.exists():
        return None
    try:
        version = json.loads(active.read_text(encoding="utf-8"))["version"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Active model pointer is invalid: {exc}") from exc
    bundle = load_bundle(version)
    if bundle.get("reliability_reference") is None:
        _attach_compatible_reliability_metadata(bundle)
    return bundle


def _attach_compatible_reliability_metadata(bundle: dict[str, object]) -> None:
    settings = get_settings()
    latest_path = settings.report_dir / "latest.json"
    test_features_path = settings.artifact_dir / "test_features.csv.gz"
    if not latest_path.is_file() or not test_features_path.is_file():
        return
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    candidate_version = latest.get("model_version")
    if (
        not candidate_version
        or candidate_version == bundle.get("model_version")
        or latest.get("dataset_fingerprint") != bundle.get("dataset_fingerprint")
        or latest.get("selected_model") != bundle.get("algorithm")
    ):
        return

    candidate = load_bundle(str(candidate_version))
    for metadata_name in (
        "reliability_reference",
        "calibration_temperature",
        "calibration_method",
        "probability_classes",
        "acceptance_policy",
    ):
        if metadata_name not in candidate:
            return
    if candidate.get("dataset_fingerprint") != bundle.get("dataset_fingerprint"):
        return

    features = pd.read_csv(test_features_path, usecols=FEATURE_NAMES)
    if features.empty:
        return
    sample = features.iloc[::max(1, len(features) // 40)].head(40)
    active_model = bundle["model"]
    candidate_model = candidate["model"]
    if (
        not np.array_equal(active_model.predict(sample), candidate_model.predict(sample))
        or not np.allclose(
            active_model.predict_proba(sample),
            candidate_model.predict_proba(sample),
            rtol=0,
            atol=1e-12,
        )
    ):
        LOGGER.warning(
            "Reliability metadata from %s does not match active model %s; leaving legacy model uncalibrated",
            candidate_version,
            bundle.get("model_version"),
        )
        return
    for metadata_name in (
        "reliability_reference",
        "calibration_temperature",
        "calibration_method",
        "probability_classes",
        "acceptance_policy",
    ):
        bundle[metadata_name] = candidate[metadata_name]
    bundle["reliability_metadata_source"] = str(candidate_version)
    LOGGER.info(
        "Attached reliability metadata from %s to behaviorally identical active model %s",
        candidate_version,
        bundle.get("model_version"),
    )
