import pandas as pd
import pytest

from app.config import get_settings
from app.dataset import LABELS, canonical_stem, discover_recordings, extract_recording, label_for_stem
from app.training import _split_recordings


def test_filename_label_mapping_handles_source_spelling_variant():
    assert label_for_stem("2Nm_Unbalalnce_0583mg") == "Unbalance"
    assert label_for_stem("4Nm_Misalign_01") == "Misalignment"
    assert canonical_stem("2Nm_Unbalalnce_0583mg") == "2Nm_Unbalance_0583mg"


@pytest.mark.skipif(
    not get_settings().dataset_path.is_dir(),
    reason="The source dataset is not included in CI.",
)
def test_workspace_dataset_has_expected_class_and_sensor_file_coverage():
    from app.config import get_settings

    recordings = discover_recordings(get_settings().dataset_path)
    assert len(recordings) == 45
    assert {record.label for record in recordings} == set(LABELS)
    assert all(record.tdms_path is not None for record in recordings)
    assert sum(record.label == "Normal" for record in recordings) == 3


@pytest.mark.skipif(
    not get_settings().dataset_path.is_dir(),
    reason="The source dataset is not included in CI.",
)
def test_bpfo_recording_produces_real_feature_windows_despite_empty_optional_current_channels():
    recording = next(
        record for record in discover_recordings(get_settings().dataset_path)
        if record.label == "BPFO"
    )
    first = next(extract_recording(recording))
    assert first["label"] == "BPFO"
    assert first["window_index"] == 0
    assert "current_1_rms" in first


def test_recording_split_is_disjoint_and_stratified():
    rows = [
        {"recording_id": f"{label}-{group}", "label": label}
        for label in LABELS
        for group in range(5)
    ]
    frame = pd.DataFrame(rows)
    split = _split_recordings(frame)
    assert split["train"].isdisjoint(split["validation"])
    assert split["train"].isdisjoint(split["test"])
    assert split["validation"].isdisjoint(split["test"])
    for name, ids in split.items():
        labels = frame[frame.recording_id.isin(ids)].label.unique()
        assert set(labels) == set(LABELS), name


def test_recording_split_rejects_classes_without_three_independent_recordings():
    frame = pd.DataFrame([
        {"recording_id": f"{label}-{group}", "label": label}
        for label in LABELS
        for group in range(3 if label != "Normal" else 2)
    ])
    with pytest.raises(ValueError, match="three independent recordings"):
        _split_recordings(frame)
