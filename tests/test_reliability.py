import numpy as np
import pandas as pd
import pytest

from app.dataset import LABELS
from app.features import FEATURE_NAMES
from app.reliability import (
    apply_temperature,
    build_reference,
    calibration_metrics,
    explain_prediction,
    fit_temperature,
    out_of_distribution,
    population_drift,
    select_acceptance_threshold,
    selective_metrics,
    sensor_quality,
)


class ProbabilityClassifier:
    classes_ = np.array(LABELS)

    def predict_proba(self, values):
        return np.tile(np.array([[0.55, 0.2, 0.1, 0.1, 0.05]]), (len(values), 1))


def _reference_frames():
    rows = []
    for group in range(4):
        for label_index, label in enumerate(LABELS):
            rows.append({
                "recording_id": f"recording-{group}-{label}",
                "label": label,
                **{name: float(group + label_index + feature_index / 100) for feature_index, name in enumerate(FEATURE_NAMES)},
            })
    return pd.DataFrame(rows)


def test_temperature_calibration_and_selective_policy_return_valid_outputs():
    names = list(LABELS)
    labels = np.array([names[index % len(names)] for index in range(100)])
    probabilities = np.full((100, len(names)), 0.05)
    probabilities[np.arange(100), np.tile([0, 0, 1, 2, 3], 20)] = 0.8
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    temperature = fit_temperature(probabilities, labels, names)
    calibrated = apply_temperature(probabilities, temperature)
    policy = select_acceptance_threshold(labels, calibrated, names, 0.5, minimum_accepted=5)

    assert np.isfinite(calibrated).all()
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert temperature > 0
    assert policy["threshold"] > 1.0
    assert calibration_metrics(labels, calibrated, names)["brier_score"] >= 0
    assert selective_metrics(labels, calibrated, names, policy["threshold"])["coverage"] == 0


def test_quality_drift_and_out_of_distribution_are_reference_based():
    frame = _reference_frames()
    training = frame.iloc[:10]
    validation = frame.iloc[10:15]
    reference = build_reference(training, validation, FEATURE_NAMES)
    valid_features = {name: float(training[name].median()) for name in FEATURE_NAMES}
    assert sensor_quality(valid_features, FEATURE_NAMES)["status"] == "valid"
    assert out_of_distribution(valid_features, reference)["outside_feature_count"] == 0
    shifted = {**valid_features, FEATURE_NAMES[0]: 1e9}
    assert out_of_distribution(shifted, reference)["outside_feature_count"] == 1
    assert population_drift([valid_features] * 19, reference)["status"] == "warming_up"
    assert population_drift([valid_features] * 25, reference)["window_size"] == 25
    constant = {**valid_features, "vibration_1_std": 0.0, "vibration_1_peak_to_peak": 0.0}
    assert sensor_quality(constant, FEATURE_NAMES)["constant_channels"] == ["vibration_1"]


def test_explanation_uses_real_model_sensitivities_and_ranks_outputs():
    frame = _reference_frames()
    training = frame.iloc[:10]
    validation = frame.iloc[10:15]
    reference = build_reference(training, validation, FEATURE_NAMES)
    model = ProbabilityClassifier()
    row = pd.DataFrame([{name: float(training.iloc[0][name]) for name in FEATURE_NAMES}])
    probabilities = apply_temperature(model.predict_proba(row), 1.0)

    explanation = explain_prediction(model, row, probabilities, list(LABELS), reference, 1.0)

    assert explanation["method"] == "single-feature median ablation"
    assert explanation["top_features"]
    assert explanation["interpretation"].endswith("not a causal explanation.")
