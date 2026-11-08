from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from app.dataset import LABELS
from app.reliability import (
    apply_temperature,
    fit_temperature,
    model_probability_classes,
    select_acceptance_threshold,
    selective_metrics,
)

RANDOM_STATE = 42


def _classification_metrics(y_true: pd.Series, predicted: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        y_true, predicted, labels=list(LABELS), output_dict=True, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, predicted, labels=list(LABELS), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=list(LABELS)).tolist(),
        "labels": list(LABELS),
    }


def _group_calibration_split(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_ids = set()
    rng = np.random.default_rng(RANDOM_STATE)
    for label, group in training[["recording_id", "label"]].drop_duplicates().groupby("label", sort=True):
        ids = sorted(group.recording_id.tolist())
        if len(ids) < 2:
            raise ValueError(f"Cross-load calibration needs two training recordings for {label}")
        rng.shuffle(ids)
        calibration_ids.add(ids[0])
    calibration = training[training.recording_id.isin(calibration_ids)].copy()
    fit = training[~training.recording_id.isin(calibration_ids)].copy()
    if set(fit.label.unique()) != set(LABELS) or set(calibration.label.unique()) != set(LABELS):
        raise ValueError("Cross-load fit/calibration recording groups must each cover all classes")
    return fit, calibration


def run_cross_load_evaluation(
    frame: pd.DataFrame,
    selected_model: Any,
    selected_algorithm: str,
    feature_names: list[str] | tuple[str, ...],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    loads = sorted(int(value) for value in frame.load_nm.dropna().unique())
    if len(loads) < 2:
        return {
            "status": "unavailable",
            "reason": "At least two documented load levels are required.",
            "load_levels_nm": loads,
            "folds": [],
        }
    folds = []
    for held_load in loads:
        training = frame[frame.load_nm != held_load].copy()
        test = frame[frame.load_nm == held_load].copy()
        fit, calibration = _group_calibration_split(training)
        model = clone(selected_model)
        model.fit(fit[list(feature_names)], fit["label"])
        probability_classes = model_probability_classes(model, list(LABELS))
        calibration_probabilities = model.predict_proba(calibration[list(feature_names)])
        temperature = fit_temperature(calibration_probabilities, calibration["label"].to_numpy(), probability_classes)
        calibrated = apply_temperature(
            model.predict_proba(test[list(feature_names)]), temperature
        )
        threshold = select_acceptance_threshold(
            calibration["label"].to_numpy(),
            apply_temperature(calibration_probabilities, temperature),
            probability_classes,
            target_accuracy=0.90,
            minimum_accepted=10,
        )
        predictions = model.predict(test[list(feature_names)])
        metrics = _classification_metrics(test["label"], predictions)
        metrics["selective"] = selective_metrics(
            test["label"].to_numpy(), calibrated, probability_classes, threshold["threshold"]
        )
        per_recording = {}
        for recording_id, group in test.groupby("recording_id", sort=True):
            per_recording[str(recording_id)] = {
                "condition": str(group.label.iloc[0]),
                "accuracy": float(accuracy_score(
                    group.label,
                    model.predict(group[list(feature_names)]),
                )),
                "window_count": int(len(group)),
            }
        folds.append({
            "held_out_load_nm": held_load,
            "train_loads_nm": [load for load in loads if load != held_load],
            "train_recording_ids": sorted(fit.recording_id.unique().tolist()),
            "calibration_recording_ids": sorted(calibration.recording_id.unique().tolist()),
            "test_recording_ids": sorted(test.recording_id.unique().tolist()),
            "train_recording_count": int(fit.recording_id.nunique()),
            "calibration_recording_count": int(calibration.recording_id.nunique()),
            "test_recording_count": int(test.recording_id.nunique()),
            "model": selected_algorithm,
            "calibration_temperature": temperature,
            "acceptance_policy": threshold,
            "metrics": metrics,
            "per_recording_accuracy": per_recording,
            "baseline_macro_f1": float(baseline_metrics["macro_f1"]),
            "macro_f1_change_vs_ordinary_test": float(metrics["macro_f1"] - baseline_metrics["macro_f1"]),
        })
    return {
        "status": "completed",
        "protocol": "leave-one-load-out; recording groups remain intact; fixed model configuration selected on the ordinary validation split; per-fold calibration groups sampled only from the non-held-out loads",
        "load_levels_nm": loads,
        "folds": folds,
    }


def run_degradation_evaluation(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    model: Any,
    feature_names: list[str] | tuple[str, ...],
    temperature: float,
    acceptance_policy: dict[str, Any],
) -> dict[str, Any]:
    names = list(feature_names)
    x_train = training[names]
    x_test = test[names]
    y_test = test["label"]
    probability_classes = model_probability_classes(model, list(LABELS))
    feature_std = x_train.std(axis=0).replace(0, 1.0)
    seed = RANDOM_STATE
    rng = np.random.default_rng(seed)
    scenarios = []
    baseline_metrics = None

    def evaluate(name: str, values: pd.DataFrame, perturbation: dict[str, Any], missing: bool = False):
        nonlocal baseline_metrics
        predictions = model.predict(values)
        metrics = _classification_metrics(y_test, predictions)
        raw_probabilities = model.predict_proba(values)
        if missing:
            # Missing groups are evaluated with the existing pipeline's fitted median imputer.
            calibrated = apply_temperature(raw_probabilities, temperature)
            selective = selective_metrics(
                y_test.to_numpy(), calibrated, probability_classes, acceptance_policy["threshold"]
            )
            selective["review_count"] = len(values)
            selective["referred_fraction"] = 1.0
            selective["accepted_count"] = 0
        else:
            calibrated = apply_temperature(raw_probabilities, temperature)
            selective = selective_metrics(
                y_test.to_numpy(), calibrated, probability_classes, acceptance_policy["threshold"]
            )
        scenarios.append({
            "scenario": name,
            "perturbation": perturbation,
            "seed": seed if name.startswith("additive_gaussian_noise") else None,
            "affected_channels": perturbation.get("affected_channels", []),
            "prediction_failure_rate": 0.0,
            "invalid_input_rejection_rate": 0.0,
            "metrics": metrics,
            "selective": selective,
            "macro_f1_change_vs_clean": (
                float(metrics["macro_f1"] - baseline_metrics["macro_f1"])
                if baseline_metrics is not None else 0.0
            ),
        })
        return metrics

    baseline_metrics = evaluate(
        "clean",
        x_test.copy(),
        {"type": "none", "basis": "genuine held-out features"},
    )
    for level in (0.1, 0.5):
        perturbation = rng.normal(size=x_test.shape) * feature_std.to_numpy() * level
        values = x_test + perturbation
        evaluate(
            f"additive_gaussian_noise_{level:g}_training_std",
            values,
            {
                "type": "additive_gaussian_noise",
                "severity_training_standard_deviations": level,
                "affected_channels": sorted({_sensor_group(name) for name in names}),
                "basis": "training-split per-feature standard deviation",
            },
        )
    channel_names = sorted({_sensor_group(name) for name in names})
    for channel in channel_names:
        affected = [name for name in names if _sensor_group(name) == channel]
        values = x_test.copy()
        values.loc[:, affected] = x_train[affected].median(axis=0).to_numpy()
        evaluate(
            f"sensor_dropout_{channel}",
            values,
            {
                "type": "complete_channel_dropout",
                "affected_channels": [channel],
                "affected_features": affected,
                "missing_strategy": "explicit per-feature training-split median; online missing values remain rejected pending this explicit experiment",
            },
            missing=True,
        )
    return {
        "status": "completed",
        "protocol": "Perturb copies of the untouched, recording-level test split with deterministic seed 42. Noise scales use training-only feature standard deviations. Channel-dropout results use the already-fitted training median imputer offline and are referred for review; live missing-feature messages are not imputed.",
        "seed": seed,
        "test_recording_ids": sorted(test.recording_id.unique().tolist()),
        "test_recording_count": int(test.recording_id.nunique()),
        "scenarios": scenarios,
    }


def _sensor_group(feature_name: str) -> str:
    parts = feature_name.split("_")
    return "_".join(parts[:2]) if parts[0] in {"vibration", "temperature", "current"} else parts[0]
