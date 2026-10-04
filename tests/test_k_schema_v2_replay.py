import pandas as pd

from scripts.audit_k_schema_v2_replay import eligible_history, metrics


def test_replay_excludes_future_inputs_and_late_publications():
    observations = pd.DataFrame([
        {"station_id": "03000", "observed_at": "2026-09-20T12:00Z", "demand": 10},
        {"station_id": "03000", "observed_at": "2026-09-20T12:15Z", "demand": 999},
        {"station_id": "03000", "observed_at": "2026-09-20T13:15Z", "demand": 888},
    ])
    releases = pd.DataFrame([
        {"station_id": "03000", "observed_at": "2026-09-20T12:15Z", "released_at": "2026-10-03T23:19Z", "unit": "passengers", "quality": "observed", "value": "61.00"},
        {"station_id": "03000", "observed_at": "2026-09-20T12:15Z", "released_at": "2026-10-04T00:19Z", "unit": "passengers", "quality": "observed", "value": "999.00"},
    ])
    result = eligible_history(observations, releases, pd.Timestamp("2026-09-20T13:00Z"), pd.Timestamp("2026-10-03T23:59Z"))
    assert result.demand.tolist() == [10, 61]
    assert result.observed_at.max() == pd.Timestamp("2026-09-20T12:15Z")


def test_missing_truth_is_not_zero_and_accuracy_uses_station_pooling():
    frame = pd.DataFrame({"station_id": ["A", "A", "B", "B"], "actual": [100, 100, 10, None], "k": [90, 110, 5, 999]})
    summary = metrics(frame, "k")
    assert summary["rows"] == 3
    assert summary["mean_station_accuracy"] == 70


def test_missing_measurement_removes_old_value_without_fabricating_actual():
    observations = pd.DataFrame([{"station_id": "03000", "observed_at": "2026-09-20T12:15Z", "demand": 123}])
    releases = pd.DataFrame([{"station_id": "03000", "observed_at": "2026-09-20T12:15Z", "released_at": "2026-10-03T23:19Z", "unit": "passengers", "quality": "missing", "value": None}])
    result = eligible_history(observations, releases, pd.Timestamp("2026-09-20T13:00Z"), pd.Timestamp("2026-10-03T23:59Z"))
    assert result.empty
