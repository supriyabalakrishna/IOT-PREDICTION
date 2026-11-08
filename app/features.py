from __future__ import annotations

import numpy as np

WINDOW_SECONDS = 1
SAMPLING_RATE_HZ = 25_600

CHANNELS = (
    *(f"vibration_{i}" for i in range(1, 5)),
    "temperature_1",
    "temperature_2",
    "current_1",
)


def feature_names() -> list[str]:
    names: list[str] = []
    for channel in CHANNELS:
        stats = (
            "mean",
            "rms",
            "std",
            "peak_to_peak",
            "kurtosis",
            "crest_factor",
        )
        names.extend(f"{channel}_{name}" for name in stats)
        if channel.startswith(("vibration_", "current_")):
            names.extend(
                f"{channel}_{name}"
                for name in ("dominant_frequency_hz", "spectral_centroid_hz")
            )
    return names


FEATURE_NAMES = feature_names()


def _channel_features(values: np.ndarray, channel: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError(f"Channel {channel} contains empty or non-finite samples")

    mean = float(np.mean(x))
    centered = x - mean
    std = float(np.sqrt(np.mean(centered * centered)))
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(x) - np.min(x))
    fourth_moment = float(np.mean(centered**4))
    kurtosis = fourth_moment / (std**4) if std > np.finfo(float).eps else 0.0
    result = {
        f"{channel}_mean": mean,
        f"{channel}_rms": rms,
        f"{channel}_std": std,
        f"{channel}_peak_to_peak": peak,
        f"{channel}_kurtosis": kurtosis,
        f"{channel}_crest_factor": float(np.max(np.abs(x)) / rms)
        if rms > np.finfo(float).eps
        else 0.0,
    }

    if channel.startswith(("vibration_", "current_")):
        spectrum = np.abs(np.fft.rfft(centered))
        frequencies = np.fft.rfftfreq(x.size, d=1.0 / SAMPLING_RATE_HZ)
        if spectrum.size > 1:
            spectrum[0] = 0.0
            total = float(np.sum(spectrum))
            dominant = float(frequencies[int(np.argmax(spectrum))])
            centroid = float(np.dot(frequencies, spectrum) / total) if total else 0.0
        else:
            dominant = centroid = 0.0
        result[f"{channel}_dominant_frequency_hz"] = dominant
        result[f"{channel}_spectral_centroid_hz"] = centroid
    return result


def extract_window_features(channels: dict[str, np.ndarray]) -> dict[str, float]:
    missing = set(CHANNELS) - set(channels)
    if missing:
        raise ValueError(f"Required sensor channels are missing: {sorted(missing)}")
    result: dict[str, float] = {}
    for channel in CHANNELS:
        result.update(_channel_features(channels[channel], channel))
    if tuple(result) != tuple(FEATURE_NAMES):
        raise RuntimeError("Feature order differs from the saved feature schema")
    return result
