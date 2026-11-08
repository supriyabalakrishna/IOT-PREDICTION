from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.io import loadmat

from app.features import CHANNELS, FEATURE_NAMES, SAMPLING_RATE_HZ, WINDOW_SECONDS, extract_window_features

LABELS = ("Normal", "BPFI", "BPFO", "Misalignment", "Unbalance")
LABEL_PATTERN = re.compile(r"(?i)(BPFI|BPFO|Misalign(?:ment)?|Normal|Unbal(?:ance|alnce))")
LOAD_PATTERN = re.compile(r"(?i)(\d+)Nm")


def canonical_stem(stem: str) -> str:
    return re.sub("unbalalnce", "Unbalance", stem, flags=re.IGNORECASE)


def label_for_stem(stem: str) -> str:
    match = LABEL_PATTERN.search(stem)
    if not match:
        raise ValueError(f"Cannot establish a condition label from recording name: {stem}")
    value = match.group(1).lower()
    return {
        "normal": "Normal",
        "bpfi": "BPFI",
        "bpfo": "BPFO",
        "misalign": "Misalignment",
        "misalignment": "Misalignment",
        "unbalance": "Unbalance",
        "unbalalnce": "Unbalance",
    }[value]


@dataclass(frozen=True)
class Recording:
    recording_id: str
    label: str
    load_nm: int | None
    vibration_path: Path
    tdms_path: Path | None


def discover_recordings(dataset_root: Path) -> list[Recording]:
    vibration_dir = dataset_root / "vibration"
    tdms_dir = dataset_root / "current,temp"
    if not vibration_dir.is_dir():
        raise FileNotFoundError(f"Vibration directory not found: {vibration_dir}")
    tdms_by_stem = {
        canonical_stem(path.stem).casefold(): path
        for path in tdms_dir.glob("*.tdms")
    } if tdms_dir.is_dir() else {}
    records: list[Recording] = []
    for path in sorted(vibration_dir.glob("*.mat")):
        records.append(
            Recording(
                recording_id=canonical_stem(path.stem),
                label=label_for_stem(path.stem),
                load_nm=int(match.group(1)) if (match := LOAD_PATTERN.search(path.stem)) else None,
                vibration_path=path,
                tdms_path=tdms_by_stem.get(canonical_stem(path.stem).casefold()),
            )
        )
    return records


def _mat_channels(path: Path) -> tuple[np.ndarray, float, str | None]:
    raw = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["Signal"])
    signal = raw.get("Signal")
    if signal is None:
        raise ValueError(f"No Signal variable in {path.name}")
    values = np.asarray(signal.y_values.values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected four vibration channels in {path.name}, got {values.shape}")
    rate = 1.0 / float(signal.x_values.increment)
    run_name = None
    try:
        annotation = signal.function_record.TL_export_properties_annotation
        raw_run_name = annotation.run_name
        run_name = "".join(np.asarray(raw_run_name).astype(str).reshape(-1))
    except (AttributeError, TypeError, ValueError):
        pass
    return values, rate, run_name


def _tdms_channels(path: Path, expected_rate: float) -> dict[str, np.ndarray]:
    tdms = TdmsFile.read(path)
    groups = [group for group in tdms.groups() if group.name != "Test Information"]
    all_channels = [channel for group in groups for channel in group.channels()]
    temperatures = [
        channel for channel in all_channels
        if str(channel.properties.get("DAC~Channel~Type", "")).casefold() == "temperature"
    ]
    currents = [
        channel for channel in all_channels
        if str(channel.properties.get("DAC~Channel~Type", "")).casefold() == "current"
    ]
    temperatures.sort(key=lambda c: c.name)
    currents.sort(key=lambda c: c.name)
    common_current = [
        channel for channel in currents
        if channel.name.casefold().endswith("/ai0") and len(channel) > 0
    ]
    if len(temperatures) != 2 or any(len(channel) == 0 for channel in temperatures) or not common_current:
        raise ValueError(
            f"Expected two non-empty temperature channels and a non-empty ai0 current channel "
            f"in {path.name}; found {len(temperatures)} temperature and "
            f"{len(common_current)} common current channels"
        )
    for channel in [*temperatures, common_current[0]]:
        increment = float(channel.properties.get("wf_increment", 0.0))
        if not np.isfinite(increment) or increment <= 0:
            raise ValueError(f"Invalid sample interval for TDMS channel {channel.name} in {path.name}")
        channel_rate = 1.0 / increment
        if not np.isclose(channel_rate, expected_rate, rtol=0, atol=10):
            raise ValueError(
                f"Sampling-rate mismatch in {path.name}/{channel.name}: "
                f"{channel_rate:g} Hz; vibration is {expected_rate:g} Hz"
            )
    return {
        **{f"temperature_{i}": np.asarray(channel[:], dtype=np.float64) for i, channel in enumerate(temperatures, 1)},
        "current_1": np.asarray(common_current[0][:], dtype=np.float64),
    }


def extract_recording(recording: Recording) -> Iterator[dict[str, object]]:
    vibration, sample_rate, embedded_run_name = _mat_channels(recording.vibration_path)
    if not np.isclose(sample_rate, SAMPLING_RATE_HZ, rtol=0, atol=1):
        raise ValueError(
            f"Unexpected vibration sampling rate in {recording.recording_id}: {sample_rate:g} Hz"
        )
    if embedded_run_name and recording.label.lower() not in embedded_run_name.lower():
        raise ValueError(
            f"Label provenance mismatch for {recording.recording_id}: "
            f"filename says {recording.label}, embedded run name is {embedded_run_name!r}"
        )
    channels: dict[str, np.ndarray] = {
        f"vibration_{i}": vibration[:, i - 1] for i in range(1, 5)
    }
    if recording.tdms_path:
        channels.update(_tdms_channels(recording.tdms_path, sample_rate))
    else:
        raise ValueError(f"Matching current/temperature recording is missing for {recording.recording_id}")

    n_samples = min(map(len, channels.values()))
    window_samples = int(sample_rate * WINDOW_SECONDS)
    if n_samples < window_samples:
        raise ValueError(
            f"Recording {recording.recording_id} has only {n_samples} aligned samples; "
            f"at least {window_samples} are required"
        )
    for window_index, start in enumerate(range(0, n_samples - window_samples + 1, window_samples)):
        window = {
            name: values[start : start + window_samples]
            for name, values in channels.items()
        }
        features = extract_window_features(window)
        yield {
            **features,
            "recording_id": recording.recording_id,
            "label": recording.label,
            "load_nm": recording.load_nm,
            "window_index": window_index,
            "timestamp_offset_seconds": float(start / sample_rate),
        }


def dataset_manifest(dataset_root: Path) -> dict[str, object]:
    recordings = discover_recordings(dataset_root)
    class_counts = {label: 0 for label in LABELS}
    for recording in recordings:
        class_counts[recording.label] += 1
    channel_coverage: dict[str, dict[str, object]] = {}
    for recording in recordings:
        if not recording.tdms_path:
            continue
        metadata = TdmsFile.read_metadata(recording.tdms_path)
        for group in metadata.groups():
            for channel in group.channels():
                channel_type = str(channel.properties.get("DAC~Channel~Type", "")).strip()
                if not channel_type:
                    continue
                name = channel.name.split("/")[-1]
                key = f"{channel_type}:{name}"
                entry = channel_coverage.setdefault(
                    key, {"nonempty_recording_count": 0, "empty_recording_ids": []}
                )
                if len(channel) > 0:
                    entry["nonempty_recording_count"] += 1
                else:
                    entry["empty_recording_ids"].append(recording.recording_id)
    files = [
        path
        for folder in ("vibration", "current,temp", "acoustic")
        for path in (dataset_root / folder).glob("*")
        if path.is_file()
    ]
    fingerprint_payload = [
        (path.relative_to(dataset_root).as_posix(), path.stat().st_size)
        for path in sorted(files)
    ]
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "dataset_root": str(dataset_root),
        "recording_count": len(recordings),
        "recordings_per_class": class_counts,
        "matched_vibration_tdms_recordings": sum(r.tdms_path is not None for r in recordings),
        "acoustic_recording_count": len(list((dataset_root / "acoustic").glob("*.mat"))),
        "acoustic_class_coverage": sorted({
            label_for_stem(path.stem) for path in (dataset_root / "acoustic").glob("*.mat")
        }) if (dataset_root / "acoustic").is_dir() else [],
        "tdms_channel_coverage": channel_coverage,
        "feature_schema_version": "1.0",
        "features": FEATURE_NAMES,
        "source_files": [
            {
                "path": path.relative_to(dataset_root).as_posix(),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(files)
        ],
        "fingerprint": fingerprint,
        "label_provenance": (
            "Canonical classes are derived from recording filenames and cross-checked "
            "against vibration MAT run-name metadata where present. No standalone dataset "
            "documentation was present in the workspace."
        ),
        "channel_provenance": (
            "Four vibration channels (g) from MAT files; two temperature channels and "
            "one common non-empty current channel (ai0, A) from TDMS. The TDMS ai2 and ai3 "
            "current channels are empty in the nine BPFO recordings. Sampling is approximately "
            "25.6 kHz. The temperature unit property is encoded as a replacement-character "
            "string in these files and requires acquisition-system confirmation."
        ),
    }


def build_feature_table(dataset_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    recordings = discover_recordings(dataset_root)
    if not recordings:
        raise FileNotFoundError(f"No vibration MAT recordings found under {dataset_root}")
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for recording in recordings:
        try:
            rows.extend(extract_recording(recording))
        except (OSError, ValueError, KeyError, IndexError) as exc:
            failures.append({"recording_id": recording.recording_id, "error": str(exc)})
    if failures:
        examples = "; ".join(f"{f['recording_id']}: {f['error']}" for f in failures[:5])
        raise RuntimeError(
            f"Feature extraction failed for {len(failures)} of {len(recordings)} recordings. "
            f"Examples: {examples}"
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("Feature extraction produced no complete windows")
    extracted_recordings = set(frame["recording_id"].unique())
    missing_recordings = sorted({record.recording_id for record in recordings} - extracted_recordings)
    if missing_recordings:
        raise RuntimeError(
            f"Recordings produced no feature windows and were not silently dropped: {missing_recordings}"
        )
    absent_labels = sorted(set(LABELS) - set(frame["label"].unique()))
    if absent_labels:
        raise RuntimeError(f"Feature extraction produced no examples for required classes: {absent_labels}")
    return frame, dataset_manifest(dataset_root)
