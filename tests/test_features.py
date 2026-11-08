import numpy as np
import pytest

from app.features import CHANNELS, FEATURE_NAMES, extract_window_features


def signals():
    sample_count = 25600
    time = np.arange(sample_count) / sample_count
    return {
        channel: np.sin(2 * np.pi * 1000 * time) if channel.startswith("vibration_")
        else np.full(sample_count, 25.0 if channel.startswith("temperature_") else 0.5)
        for channel in CHANNELS
    }


def test_feature_extraction_is_deterministic_and_ordered():
    values = signals()
    first = extract_window_features(values)
    second = extract_window_features(values)
    assert list(first) == FEATURE_NAMES
    assert first == second
    assert first["vibration_1_dominant_frequency_hz"] == pytest.approx(1000, abs=1)
    assert first["temperature_1_mean"] == pytest.approx(25)


def test_feature_extraction_rejects_missing_and_nonfinite_sensor_data():
    values = signals()
    del values["current_1"]
    with pytest.raises(ValueError, match="missing"):
        extract_window_features(values)
    values = signals()
    values["current_1"][10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract_window_features(values)

