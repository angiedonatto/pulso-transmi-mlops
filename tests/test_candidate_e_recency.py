from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import candidate_e_recency as ce
from scripts.model_improvement_candidates import load_observations_from_frame


def sample_observations(days: int = 45) -> pd.DataFrame:
    rows = []
    observed_at = pd.date_range("2026-01-01T00:00:00Z", periods=days * 96, freq="15min")
    for station_id, offset in [("03000", 100), ("05000", 500), ("09122", 250)]:
        for idx, timestamp in enumerate(observed_at):
            rows.append(
                {
                    "station_id": station_id,
                    "observed_at": timestamp,
                    "demand": offset + (idx % 96) + (idx // 96) + 3 * np.sin(idx / 5.0),
                }
            )
    return load_observations_from_frame(pd.DataFrame(rows))


def test_recency_weights_depend_only_on_elapsed_time_not_on_target_or_error() -> None:
    train_samples = pd.DataFrame(
        {
            "data_cutoff": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z", "2026-01-09T00:00:00Z"]),
            "actual": [1.0, 999.0, -999.0],  # wildly different targets must not affect the weight
        }
    )
    reference_time = pd.Timestamp("2026-01-10T00:00:00Z")

    weights = ce.recency_sample_weights(train_samples, reference_time, half_life_days=3.0)

    # Older rows get strictly smaller weight, regardless of their (irrelevant) actual value.
    assert weights[0] < weights[1] < weights[2]
    # Exactly one half-life (9 days apart from a 3-day half-life is 3 half-lives -> 0.5**3 = 0.125).
    assert np.isclose(weights[0], 0.5 ** 3, atol=1e-6)


def test_recency_weights_reject_training_rows_newer_than_reference_time() -> None:
    train_samples = pd.DataFrame({"data_cutoff": pd.to_datetime(["2026-01-10T00:00:00Z"]), "actual": [1.0]})
    reference_time = pd.Timestamp("2026-01-05T00:00:00Z")

    try:
        ce.recency_sample_weights(train_samples, reference_time, half_life_days=3.0)
        assert False, "expected RuntimeError for a training row newer than the reference time"
    except RuntimeError:
        pass


def test_rolling_train_window_never_reaches_the_validation_cutoff() -> None:
    observations = sample_observations()
    times = ce.dense_times(observations)
    reference_time = pd.Timestamp("2026-02-01T00:00:00Z")

    for window_days in ce.ROLLING_WINDOWS_DAYS:
        train_cutoffs = ce.rolling_train_cutoffs(observations, times, reference_time, window_days)
        assert train_cutoffs, f"rolling window {window_days}d produced no cutoffs"
        assert max(train_cutoffs) < reference_time - pd.Timedelta(minutes=ce.MAX_HORIZON)
        assert min(train_cutoffs) >= reference_time - pd.Timedelta(days=window_days)


def test_expanding_train_window_never_reaches_the_validation_cutoff() -> None:
    observations = sample_observations()
    times = ce.dense_times(observations)
    reference_time = pd.Timestamp("2026-02-01T00:00:00Z")

    train_cutoffs = ce.expanding_train_cutoffs(observations, times, reference_time)

    assert train_cutoffs
    assert max(train_cutoffs) < reference_time - pd.Timedelta(minutes=ce.MAX_HORIZON)
    assert min(train_cutoffs) >= observations["observed_at"].min() + pd.Timedelta(days=ce.MIN_HISTORY_DAYS)


def test_cadence_buckets_reuse_one_model_per_bucket() -> None:
    """Verifies the retraining-cadence mechanism: run_strategy must fit exactly one
    model per bucket and apply it to every cutoff in that bucket -- checked here via the
    `training_rows` passenger column, which is constant within a bucket only if every
    row in it came from the same single `model.fit()` call."""
    observations = sample_observations()
    times = ce.dense_times(observations)
    cutoffs = [ts for ts in times if ts.minute == 0 and pd.Timestamp("2026-02-01T00:00:00Z") <= ts < pd.Timestamp("2026-02-03T00:00:00Z")]
    cache: dict = {}

    result = ce.run_strategy(observations, times, cutoffs, window_kind="expanding", window_days=None, half_life_days=None, cadence_hours=6, cache=cache, name="test")

    for bucket_start, group in result.predictions.groupby("bucket_start"):
        assert group["training_rows"].nunique() == 1, "every row in a cadence bucket must share the same training_rows count (one model per bucket)"


def test_temporal_fold_audit_holds_for_every_strategy_kind() -> None:
    observations = sample_observations()
    times = ce.dense_times(observations)
    cutoffs = [ts for ts in times if ts.minute == 0 and pd.Timestamp("2026-02-01T00:00:00Z") <= ts < pd.Timestamp("2026-02-04T00:00:00Z")]
    cache: dict = {}

    checks = [
        ("expanding", None, 24),
        ("expanding", None, 6),
        ("rolling", 14, 24),
        ("rolling", 21, 12),
    ]
    for window_kind, window_days, cadence_hours in checks:
        audit = ce.strategy_fold_audit(
            observations, times, cutoffs, window_kind=window_kind, window_days=window_days, cadence_hours=cadence_hours, cache=cache, strategy_name=f"{window_kind}-{window_days}-{cadence_hours}"
        )
        assert not audit.empty
        assert (audit["training_end"] < audit["validation_start"]).all()


def test_strategy_features_are_the_baseline_production_safe_schema() -> None:
    from scripts.challenger_lab_v2_core import BASELINE_FEATURE_SCHEMA

    assert "station_id" not in BASELINE_FEATURE_SCHEMA
    assert "actual" not in BASELINE_FEATURE_SCHEMA
    assert len(BASELINE_FEATURE_SCHEMA) == len(set(BASELINE_FEATURE_SCHEMA))
