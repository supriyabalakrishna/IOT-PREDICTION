from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax


def model_probability_classes(model: Any, class_names: list[str]) -> list[str]:
    classes = getattr(model, "classes_", class_names)
    normalized = []
    for label in classes:
        if isinstance(label, (int, np.integer)) and 0 <= int(label) < len(class_names):
            normalized.append(class_names[int(label)])
        else:
            normalized.append(str(label))
    return normalized


def _channel_name(feature_name: str) -> str:
    parts = feature_name.split("_")
    return "_".join(parts[:2]) if parts[0] in {"vibration", "temperature", "current"} else parts[0]


def build_reference(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    reference: dict[str, Any] = {"features": {}, "ood_feature_count_threshold": 0}
    outlier_counts = []
    recording_psi = []
    for name in feature_names:
        values = training[name].to_numpy(dtype=float)
        quantile_levels = np.linspace(0, 1, 11)
        edges = np.unique(np.quantile(values, quantile_levels))
        if edges.size > 1:
            edges[0] = -np.inf
            edges[-1] = np.inf
        else:
            edges = np.array([-np.inf, np.inf])
        reference_counts = np.histogram(values, bins=edges)[0].astype(float)
        reference_counts /= max(1, reference_counts.sum())
        reference["features"][name] = {
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "lower": float(np.quantile(values, 0.005)),
            "upper": float(np.quantile(values, 0.995)),
            "psi_edges": [float(x) if math.isfinite(float(x)) else None for x in edges],
            "psi_reference": reference_counts.tolist(),
        }

    lower = np.array([reference["features"][name]["lower"] for name in feature_names])
    upper = np.array([reference["features"][name]["upper"] for name in feature_names])
    validation_values = validation[list(feature_names)].to_numpy(dtype=float)
    outlier_counts = ((validation_values < lower) | (validation_values > upper)).sum(axis=1)
    reference["ood_feature_count_threshold"] = int(np.ceil(np.quantile(outlier_counts, 0.95)))

    for _, recording in validation.groupby("recording_id", sort=True):
        psi = [
            _feature_psi(recording[name].to_numpy(dtype=float), reference["features"][name])
            for name in feature_names
        ]
        recording_psi.append(max(psi, default=0.0))
    reference["drift_psi_threshold"] = (
        float(np.quantile(recording_psi, 0.95)) if recording_psi else 0.2
    )
    reference["drift_calibration_recording_count"] = len(recording_psi)
    reference["drift_window_size"] = 200
    reference["feature_names"] = list(feature_names)
    return reference


def _feature_psi(values: np.ndarray, reference: dict[str, Any]) -> float:
    raw_edges = reference["psi_edges"]
    edges = np.array([
        -np.inf if value is None and index == 0 else
        np.inf if value is None else float(value)
        for index, value in enumerate(raw_edges)
    ])
    baseline = np.asarray(reference["psi_reference"], dtype=float)
    if values.size == 0:
        return 0.0
    observed = np.histogram(values, bins=edges)[0].astype(float)
    observed /= observed.sum()
    epsilon = 1e-6
    baseline = np.clip(baseline, epsilon, None)
    observed = np.clip(observed, epsilon, None)
    return float(np.sum((observed - baseline) * np.log(observed / baseline)))


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray, class_names: list[str]) -> float:
    encoded = np.array([class_names.index(str(label)) for label in labels], dtype=int)
    logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0))

    def nll(log_temperature: float) -> float:
        calibrated = softmax(logits / np.exp(log_temperature), axis=1)
        return float(-np.mean(np.log(np.clip(calibrated[np.arange(encoded.size), encoded], 1e-12, 1.0))))

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError(f"Probability temperature calibration failed: {result.message}")
    return float(np.exp(result.x))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be positive and finite")
    logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0))
    calibrated = softmax(logits / temperature, axis=1)
    if not np.isfinite(calibrated).all():
        raise ValueError("Calibrated probabilities contain non-finite values")
    return calibrated


def calibration_metrics(y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]) -> dict[str, float]:
    encoded = np.array([class_names.index(str(label)) for label in y_true], dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    one_hot = np.eye(len(class_names))[encoded]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == encoded
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "expected_calibration_error_10_bins": float(ece),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "negative_log_likelihood": float(
            -np.mean(np.log(np.clip(probabilities[np.arange(encoded.size), encoded], 1e-12, 1.0)))
        ),
    }


def select_acceptance_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    target_accuracy: float,
    minimum_accepted: int = 10,
) -> dict[str, Any]:
    if not 0 < target_accuracy <= 1:
        raise ValueError("Target selective accuracy must be between zero and one")
    encoded = np.array([class_names.index(str(label)) for label in y_true], dtype=int)
    predicted = np.asarray(probabilities).argmax(axis=1)
    confidence = np.asarray(probabilities).max(axis=1)
    threshold = 1.01
    for candidate in np.sort(np.unique(confidence)):
        accepted = confidence >= candidate
        if accepted.sum() >= minimum_accepted and float(np.mean(predicted[accepted] == encoded[accepted])) >= target_accuracy:
            threshold = float(candidate)
            break
    validation_accepted = confidence >= threshold
    return {
        "threshold": threshold,
        "target_validation_accuracy": float(target_accuracy),
        "minimum_accepted_validation_rows": int(minimum_accepted),
        "validation_coverage": float(validation_accepted.mean()),
        "validation_selective_accuracy": (
            float(np.mean(predicted[validation_accepted] == encoded[validation_accepted]))
            if validation_accepted.any() else None
        ),
        "threshold_found": threshold <= 1.0,
    }


def selective_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    threshold: float,
) -> dict[str, Any]:
    encoded = np.array([class_names.index(str(label)) for label in y_true], dtype=int)
    predicted = np.asarray(probabilities).argmax(axis=1)
    accepted = np.asarray(probabilities).max(axis=1) >= threshold
    count = int(accepted.sum())
    return {
        "coverage": float(accepted.mean()),
        "selective_accuracy": float(np.mean(predicted[accepted] == encoded[accepted])) if count else None,
        "accepted_error_rate": float(np.mean(predicted[accepted] != encoded[accepted])) if count else None,
        "referred_fraction": float(1.0 - accepted.mean()),
        "accepted_count": count,
        "review_count": int(len(accepted) - count),
    }


def sensor_quality(
    features: dict[str, float],
    feature_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    constant_channels = []
    for name in feature_names:
        if name.endswith("_std"):
            channel = _channel_name(name)
            peak_name = f"{channel}_peak_to_peak"
            if name in features and peak_name in features and features[name] == 0 and features[peak_name] == 0:
                constant_channels.append(channel)
    constant_channels = sorted(set(constant_channels))
    return {
        "status": "degraded" if constant_channels else "valid",
        "constant_channels": constant_channels,
        "flags": [f"constant_channel:{channel}" for channel in constant_channels],
        "threshold_basis": "exact zero standard deviation and peak-to-peak range",
    }


def out_of_distribution(
    features: dict[str, float],
    reference: dict[str, Any],
) -> dict[str, Any]:
    outside = [
        name for name, value in features.items()
        if value < reference["features"][name]["lower"] or value > reference["features"][name]["upper"]
    ]
    threshold = int(reference["ood_feature_count_threshold"])
    return {
        "status": "warning" if len(outside) > threshold else "normal",
        "outside_reference_features": outside,
        "outside_feature_count": len(outside),
        "calibrated_feature_count_threshold": threshold,
        "threshold_basis": "95th percentile of per-row training-quantile outlier count on validation recordings",
    }


def population_drift(
    feature_rows: list[dict[str, float]],
    reference: dict[str, Any],
) -> dict[str, Any]:
    if len(feature_rows) < 20:
        return {
            "status": "warming_up",
            "window_size": len(feature_rows),
            "psi_max": None,
            "threshold": reference["drift_psi_threshold"],
        }
    values = pd.DataFrame.from_records(feature_rows, columns=reference["feature_names"])
    psi_by_feature = {
        name: _feature_psi(values[name].to_numpy(dtype=float), reference["features"][name])
        for name in reference["feature_names"]
    }
    max_psi = max(psi_by_feature.values(), default=0.0)
    threshold = float(reference["drift_psi_threshold"])
    return {
        "status": "warning" if max_psi > threshold else "normal",
        "window_size": len(feature_rows),
        "psi_max": max_psi,
        "threshold": threshold,
        "feature_psi": psi_by_feature,
        "threshold_basis": "95th percentile of per-recording PSI measured on validation recordings",
    }


def explain_prediction(
    model: Any,
    row: pd.DataFrame,
    probabilities: np.ndarray,
    class_names: list[str],
    reference: dict[str, Any],
    temperature: float,
    top_n: int = 8,
) -> dict[str, Any]:
    names = list(reference["feature_names"])
    label_index = int(np.asarray(probabilities).argmax(axis=1)[0])
    probability_classes = model_probability_classes(model, class_names)
    explained_class = probability_classes[label_index]
    base_probability = float(probabilities[0, label_index])

    try:
        import shap

        if hasattr(model, "named_steps") and hasattr(model.named_steps.get("classifier"), "estimators_"):
            transformer = model[:-1]
            transformed = transformer.transform(row)
            explainer = shap.TreeExplainer(model.named_steps["classifier"])
            values = explainer.shap_values(transformed)
            class_index = probability_classes.index(explained_class)
            if isinstance(values, list):
                contributions = np.asarray(values[class_index])[0]
            else:
                values = np.asarray(values)
                contributions = values[0, :, class_index] if values.ndim == 3 else values[0]
            impacts = [
                {
                    "feature": name,
                    "value": float(row.iloc[0][name]),
                    "reference_median": float(reference["features"][name]["median"]),
                    "attribution": float(contributions[index]),
                }
                for index, name in enumerate(names)
            ]
            impacts.sort(key=lambda item: abs(item["attribution"]), reverse=True)
            return {
                "method": "SHAP TreeExplainer",
                "explained_class": explained_class,
                "base_calibrated_probability": base_probability,
                "top_features": impacts[:top_n],
                "interpretation": "Tree-model attribution; feature contribution is not causal evidence.",
            }
    except ImportError:
        pass
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        import logging

        logging.getLogger(__name__).warning("SHAP could not explain this model; using median ablation: %s", exc)

    median_row = {name: reference["features"][name]["median"] for name in names}
    variants = []
    for name in names:
        variant = row.iloc[0].to_dict()
        variant[name] = median_row[name]
        variants.append(variant)
    variant_probabilities = apply_temperature(model.predict_proba(pd.DataFrame(variants, columns=names)), temperature)
    impacts = []
    for index, name in enumerate(names):
        value = float(row.iloc[0][name])
        impacts.append({
            "feature": name,
            "value": value,
            "reference_median": median_row[name],
            "probability_delta": base_probability - float(variant_probabilities[index, label_index]),
        })
    impacts.sort(key=lambda item: abs(item["probability_delta"]), reverse=True)
    return {
        "method": "single-feature median ablation",
        "explained_class": explained_class,
        "base_calibrated_probability": base_probability,
        "top_features": impacts[:top_n],
        "interpretation": "Non-additive model sensitivity relative to the training median; not a causal explanation.",
    }
