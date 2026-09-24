from __future__ import annotations

import pandas as pd

from scripts.candidate_c_experiments import C_FEATURE_SCHEMA, build_candidate_c_samples, candidate_c_matrix, leakage_checks


def sample_observations(days: int = 22) -> pd.DataFrame:
    rows = []
    observed_at = pd.date_range("2026-01-01T00:00:00Z", periods=days * 96, freq="15min")
    for station_id, offset in [("03000", 100), ("02300", 200)]:
        for idx, timestamp in enumerate(observed_at):
            rows.append(
                {
                    "station_id": station_id,
                    "observed_at": timestamp,
                    "demand": offset + (idx % 96) + (idx // 96),
                }
            )
    return pd.DataFrame(rows)


def test_candidate_c_features_end_at_data_cutoff_for_30_min_horizon() -> None:
    observations = sample_observations(days=22)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    local = observations["observed_at"].dt.tz_convert("America/Bogota")
    observations["hour"] = local.dt.hour
    observations["minute"] = local.dt.minute
    observations["day_of_week"] = local.dt.dayofweek
    observations["is_weekend"] = observations["day_of_week"].isin([5, 6]).astype(int)
    observations["quarter_hour_block"] = observations["hour"] * 4 + observations["minute"] // 15
    cutoffs = [pd.Timestamp("2026-01-20T00:00:00Z")]

    samples = build_candidate_c_samples(observations, cutoffs)
    row = samples[(samples["station_id"] == "03000") & (samples["horizon_minutes"] == 30)].iloc[0]
    lookup = observations.set_index(["station_id", "observed_at"])["demand"].to_dict()

    assert row["target_at"] == pd.Timestamp("2026-01-20T00:30:00Z")
    assert row["data_cutoff"] == pd.Timestamp("2026-01-20T00:00:00Z")
    assert row["lag_15"] != row["lag_15"]
    assert row["lag_30"] == lookup[("03000", pd.Timestamp("2026-01-20T00:00:00Z"))]
    past_hour = [
        lookup[("03000", pd.Timestamp("2026-01-19T23:15:00Z"))],
        lookup[("03000", pd.Timestamp("2026-01-19T23:30:00Z"))],
        lookup[("03000", pd.Timestamp("2026-01-19T23:45:00Z"))],
        lookup[("03000", pd.Timestamp("2026-01-20T00:00:00Z"))],
    ]
    assert row["rolling_mean_1h"] == sum(past_hour) / len(past_hour)
    assert leakage_checks(samples)["target_cutoff_consistent"] is True


def test_candidate_c_matrix_uses_expected_feature_schema() -> None:
    observations = sample_observations(days=22)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    local = observations["observed_at"].dt.tz_convert("America/Bogota")
    observations["hour"] = local.dt.hour
    observations["minute"] = local.dt.minute
    observations["day_of_week"] = local.dt.dayofweek
    observations["is_weekend"] = observations["day_of_week"].isin([5, 6]).astype(int)
    observations["quarter_hour_block"] = observations["hour"] * 4 + observations["minute"] // 15
    samples = build_candidate_c_samples(observations, [pd.Timestamp("2026-01-20T00:00:00Z")])

    matrix = candidate_c_matrix(samples, {"02300": 0, "03000": 1})

    assert list(matrix.columns) == C_FEATURE_SCHEMA
    assert len(matrix) == 8
