from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.candidate_d1_residual_calibration import (
    D1_FEATURE_SCHEMA,
    add_out_of_fold_candidate_b_predictions,
    apply_clip_bounds,
    d1_matrix,
    learn_clip_bounds,
    temporal_fold_audit,
    with_residual_target,
)
from scripts.model_improvement_candidates import HORIZONS, load_observations_from_frame, make_samples, station_code_map


def sample_observations(days: int = 22) -> pd.DataFrame:
    rows = []
    observed_at = pd.date_range("2026-01-01T00:00:00Z", periods=days * 96, freq="15min")
    for station_id, offset in [("03000", 100), ("05000", 500)]:
        for idx, timestamp in enumerate(observed_at):
            rows.append(
                {
                    "station_id": station_id,
                    "observed_at": timestamp,
                    "demand": offset + (idx % 96) + (idx // 96),
                }
            )
    return load_observations_from_frame(pd.DataFrame(rows))


def test_residual_target_is_actual_minus_candidate_b_prediction() -> None:
    samples = pd.DataFrame({"actual": [100.0, 80.0], "candidate_b_prediction": [90.0, 95.0]})

    result = with_residual_target(samples)

    assert result["residual_target"].tolist() == [10.0, -15.0]


def test_d1_matrix_uses_expected_feature_schema() -> None:
    observations = sample_observations()
    samples = make_samples(observations, [pd.Timestamp("2026-01-20T00:00:00Z")])
    samples["candidate_b_prediction"] = samples["actual"].astype(float) - 2.0

    matrix = d1_matrix(samples, {"03000": 0, "05000": 1})

    assert list(matrix.columns) == D1_FEATURE_SCHEMA
    assert len(matrix) == len(HORIZONS) * 2
    assert matrix["candidate_b_prediction"].notna().all()


def test_clipping_bounds_are_learned_from_training_residuals_by_horizon() -> None:
    train = pd.DataFrame(
        {
            "horizon_minutes": [30] * 5 + [60] * 5,
            "residual_target": [-10.0, -5.0, 0.0, 5.0, 10.0, -100.0, -50.0, 0.0, 50.0, 100.0],
        }
    )
    bounds = learn_clip_bounds(train, lower_quantile=0.0, upper_quantile=1.0, min_group_rows=3)
    corrections = np.array([-99.0, 99.0, -99.0, 99.0])
    samples = pd.DataFrame({"horizon_minutes": [30, 30, 60, 60]})

    clipped = apply_clip_bounds(corrections, samples, bounds)

    assert clipped.tolist() == [-10.0, 10.0, -99.0, 99.0]


def test_samples_keep_targets_after_data_cutoff_without_future_lags() -> None:
    observations = sample_observations()
    cutoff = pd.Timestamp("2026-01-20T00:00:00Z")
    samples = make_samples(observations, [cutoff])
    row = samples[(samples["station_id"] == "03000") & (samples["horizon_minutes"] == 60)].iloc[0]

    assert row["target_at"] == cutoff + pd.Timedelta(minutes=60)
    assert row["lag_15"] != row["lag_15"]
    assert row["lag_30"] != row["lag_30"]
    assert row["lag_45"] != row["lag_45"]
    assert pd.notna(row["lag_60"])


def test_candidate_d1_temporal_audit_rejects_training_validation_overlap() -> None:
    observations = sample_observations(days=32)
    cutoffs = [
        ts
        for ts in sorted(observations["observed_at"].drop_duplicates())
        if ts.minute == 0 and pd.Timestamp("2026-01-28T00:00:00Z") <= ts < pd.Timestamp("2026-01-30T00:00:00Z")
    ]

    audit = temporal_fold_audit(observations, cutoffs)

    assert not audit.empty
    assert (audit["b_training_end"] < audit["validation_start"]).all()
    assert (audit["d1_training_end"] < audit["validation_start"]).all()


def test_residual_training_candidate_b_predictions_are_out_of_fold() -> None:
    observations = sample_observations(days=32)
    train_cutoffs = [
        ts
        for ts in sorted(observations["observed_at"].drop_duplicates())
        if ts.minute == 0 and pd.Timestamp("2026-01-20T00:00:00Z") <= ts < pd.Timestamp("2026-01-28T00:00:00Z")
    ]
    samples = make_samples(observations, train_cutoffs)

    with_b = add_out_of_fold_candidate_b_predictions(samples, station_code_map(observations))

    assert not with_b.empty
    assert with_b["candidate_b_prediction"].notna().all()
    assert (with_b["candidate_b_training_end"] < with_b["target_at"]).all()
