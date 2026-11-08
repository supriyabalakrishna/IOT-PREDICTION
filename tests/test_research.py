import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from app.dataset import LABELS
from app.features import FEATURE_NAMES
from app.research import run_cross_load_evaluation, run_degradation_evaluation


def _frame():
    rows = []
    for load in (0, 2, 4):
        for class_index, label in enumerate(LABELS):
            for recording in range(2):
                recording_id = f"{load}Nm-{label}-{recording}"
                for window in range(2):
                    rows.append({
                        "recording_id": recording_id,
                        "label": label,
                        "load_nm": load,
                        "window_index": window,
                        **{name: float(load + class_index + recording + window / 10) for name in FEATURE_NAMES},
                    })
    return pd.DataFrame(rows)


def test_cross_load_folds_hold_out_all_recordings_at_a_load():
    frame = _frame()
    model = Pipeline([("classifier", DummyClassifier(strategy="prior"))])
    training = frame[frame.load_nm != 4]
    model.fit(training[list(FEATURE_NAMES)], training.label)

    result = run_cross_load_evaluation(
        frame,
        model,
        "dummy",
        FEATURE_NAMES,
        {"macro_f1": 0.2},
    )

    assert result["status"] == "completed"
    assert {fold["held_out_load_nm"] for fold in result["folds"]} == {0, 2, 4}
    for fold in result["folds"]:
        all_train_groups = set(fold["train_recording_ids"]) | set(fold["calibration_recording_ids"])
        assert not all_train_groups.intersection(fold["test_recording_ids"])
        assert len(fold["test_recording_ids"]) == 10
        assert set(fold["metrics"]["labels"]) == set(LABELS)


def test_degradation_experiments_use_deterministic_copies_and_report_deltas():
    from app.training import _pipeline

    frame = _frame()
    train = frame[frame.load_nm != 4]
    test = frame[frame.load_nm == 4]
    model = _pipeline(DummyClassifier(strategy="prior"))
    model.fit(train[list(FEATURE_NAMES)], train.label)
    policy = {"threshold": 0.0}
    result = run_degradation_evaluation(
        train,
        train.iloc[:],
        test,
        model,
        FEATURE_NAMES,
        1.0,
        policy,
    )

    assert result["status"] == "completed"
    assert result["seed"] == 42
    assert result["scenarios"][0]["scenario"] == "clean"
    assert result["scenarios"][0]["selective"]["selective_accuracy"] == result["scenarios"][0]["metrics"]["accuracy"]
    assert all(row["macro_f1_change_vs_clean"] is not None for row in result["scenarios"])
    assert any(row["perturbation"]["type"] == "complete_channel_dropout" for row in result["scenarios"])
    assert all(row["selective"]["referred_fraction"] == 1.0 for row in result["scenarios"] if "sensor_dropout" in row["scenario"])
