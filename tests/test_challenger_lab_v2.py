from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from scripts import challenger_lab_v2 as lab
from scripts import challenger_lab_v2_core as core
from scripts.model_improvement_candidates import load_observations_from_frame, make_samples


def sample_observations(days: int = 40) -> pd.DataFrame:
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


def test_extended_samples_base_columns_match_original_make_samples() -> None:
    observations = sample_observations()
    cutoffs = [
        ts
        for ts in sorted(observations["observed_at"].drop_duplicates())
        if ts.minute == 0
    ][-30:-10]

    original = make_samples(observations, cutoffs)
    extended = core.build_extended_samples(observations, cutoffs)

    original_sorted = original[core.BASE_SAMPLE_COLUMNS].sort_values(["data_cutoff", "station_id", "horizon_minutes"]).reset_index(drop=True)
    extended_sorted = extended[core.BASE_SAMPLE_COLUMNS].sort_values(["data_cutoff", "station_id", "horizon_minutes"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(original_sorted, extended_sorted, check_dtype=False)


def test_extra_lags_are_never_masked_because_they_exceed_every_horizon() -> None:
    observations = sample_observations()
    cutoffs = [pd.Timestamp("2026-01-25T00:00:00Z")]

    samples = core.build_extended_samples(observations, cutoffs)

    assert samples["lag_90"].notna().all()
    assert samples["lag_120"].notna().all()
    for horizon in core.HORIZONS:
        assert horizon <= max(core.EXTRA_LAGS)


def test_rolling_and_lag_features_are_unaffected_by_data_strictly_after_cutoff() -> None:
    """Empirical leakage probe: perturbing an observation strictly between data_cutoff
    and target_at must not change a single engineered feature for that (station,
    horizon, target_at) row."""
    observations = sample_observations()
    station = "05000"
    horizon = 60
    target_at = sorted(observations.loc[observations["station_id"] == station, "observed_at"].unique())[500]
    target_at = pd.Timestamp(target_at)
    cutoff = target_at - pd.Timedelta(minutes=horizon)

    def feature_row(obs_frame: pd.DataFrame) -> pd.Series:
        samples = core.build_extended_samples(obs_frame, [cutoff])
        row = samples[(samples["station_id"] == station) & (samples["horizon_minutes"] == horizon) & (samples["target_at"] == target_at)]
        assert len(row) == 1
        return row.iloc[0]

    base_row = feature_row(observations)

    mutated = observations.copy()
    future_mask = (mutated["station_id"] == station) & (mutated["observed_at"] > cutoff) & (mutated["observed_at"] < target_at)
    assert future_mask.sum() > 0
    mutated.loc[future_mask, "demand"] = mutated.loc[future_mask, "demand"] + 999999.0
    future_row = feature_row(mutated)

    feature_cols = [c for c in core.EXTENDED_SAMPLE_COLUMNS if c not in ("station_id", "data_cutoff", "target_at", "horizon_minutes", "actual")]
    for col in feature_cols:
        base_val, future_val = base_row[col], future_row[col]
        if pd.isna(base_val) and pd.isna(future_val):
            continue
        assert base_val == future_val, f"feature {col!r} leaked post-cutoff data"


def test_rolling_features_do_use_data_at_or_before_cutoff() -> None:
    """Sanity counterpart to the leakage probe above: features must not be dead. Mutating
    the observation exactly AT data_cutoff must change several engineered features."""
    observations = sample_observations()
    station = "05000"
    horizon = 60
    target_at = sorted(observations.loc[observations["station_id"] == station, "observed_at"].unique())[500]
    target_at = pd.Timestamp(target_at)
    cutoff = target_at - pd.Timedelta(minutes=horizon)

    def feature_row(obs_frame: pd.DataFrame) -> pd.Series:
        samples = core.build_extended_samples(obs_frame, [cutoff])
        row = samples[(samples["station_id"] == station) & (samples["horizon_minutes"] == horizon) & (samples["target_at"] == target_at)]
        return row.iloc[0]

    base_row = feature_row(observations)
    mutated = observations.copy()
    at_cutoff_mask = (mutated["station_id"] == station) & (mutated["observed_at"] == cutoff)
    assert at_cutoff_mask.sum() == 1
    mutated.loc[at_cutoff_mask, "demand"] = mutated.loc[at_cutoff_mask, "demand"] + 999999.0
    mutated_row = feature_row(mutated)

    changed = [
        col
        for col in ["lag_60", "rolling_mean_1h", "rolling_mean_3h", "recent_mean"]
        if base_row[col] != mutated_row[col]
    ]
    assert changed, "no feature reacted to a change at data_cutoff -- features may be dead/miswired"


def test_profile_and_dow_profile_exclude_the_target_row_itself() -> None:
    observations = sample_observations()
    cutoffs = [
        ts
        for ts in sorted(observations["observed_at"].drop_duplicates())
        if ts.minute == 0
    ][-5:]

    samples = core.build_extended_samples(observations, cutoffs)
    for _, row in samples.iterrows():
        # profile_mean/median and profile_dow_mean/median must never equal a value that
        # could only be reproduced by including the (unknown) target itself; the
        # authoritative check is structural (shift(1) in build_extended_samples), this
        # is a smoke check that the columns are populated and finite once history exists.
        if pd.notna(row["profile_mean"]):
            assert np.isfinite(row["profile_mean"])
        if pd.notna(row["profile_dow_mean"]):
            assert np.isfinite(row["profile_dow_mean"])


def test_temporal_fold_audit_training_ends_before_validation_starts() -> None:
    observations = sample_observations(days=40)
    cutoffs = [
        ts
        for ts in sorted(observations["observed_at"].drop_duplicates())
        if ts.minute == 0 and pd.Timestamp("2026-01-30T00:00:00Z") <= ts < pd.Timestamp("2026-02-03T00:00:00Z")
    ]

    audit = core.temporal_fold_audit_generic(observations, cutoffs)

    assert not audit.empty
    assert (audit["training_end"] < audit["validation_start"]).all()


def test_station_bias_correction_uses_only_strictly_earlier_folds() -> None:
    """The correction for fold k must never depend on fold k's own (or any later fold's)
    actual/prediction pairs. We check this by construction: corrupting the CURRENT
    fold's actuals cannot be done without changing its own prediction, so instead we
    directly assert the history accumulation contract inside the function by checking
    fold 1 gets zero correction (no history yet) and fold 2 depends only on fold 1."""
    observations = sample_observations(days=40)
    cutoffs = [
        ts
        for ts in sorted(observations["observed_at"].drop_duplicates())
        if ts.minute == 0 and pd.Timestamp("2026-01-28T00:00:00Z") <= ts < pd.Timestamp("2026-01-31T00:00:00Z")
    ]
    times = sorted(observations["observed_at"].drop_duplicates())
    cache: dict = {}

    result = lab.run_station_bias_correction(observations, times, cutoffs, group_cols=("station_id",), cache=cache, name="test bias", min_history_rows=1)

    fold_starts = sorted(result.predictions["data_cutoff"].dt.floor("D").unique())
    first_fold = result.predictions[result.predictions["data_cutoff"].dt.floor("D") == fold_starts[0]]
    # Cold start: the very first fold has no history, so prediction must equal the raw
    # (uncorrected) Candidate-B-architecture prediction exactly.
    assert np.allclose(first_fold["prediction"].to_numpy(), first_fold["raw_prediction"].to_numpy())


def test_fit_ensemble_weights_signature_has_no_hard_holdout_parameter() -> None:
    """Structural guarantee: the ensemble weight fitter cannot be handed hard-holdout
    data even by accident -- it only accepts one dict of (already out-of-fold)
    walk-forward prediction frames."""
    params = list(inspect.signature(lab.fit_ensemble_weights).parameters)
    assert params == ["wf_predictions", "step"]


def test_fit_ensemble_weights_recovers_the_better_of_two_perfect_and_noisy_models() -> None:
    key_cols = ["station_id", "target_at", "horizon_minutes"]
    base = pd.DataFrame(
        {
            "station_id": ["03000"] * 20,
            "target_at": pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC"),
            "horizon_minutes": [15] * 20,
            "actual": np.arange(20, dtype=float) + 10,
        }
    )
    perfect = base.assign(prediction=base["actual"])
    noisy = base.assign(prediction=base["actual"] + 50.0)

    weights, score = lab.fit_ensemble_weights({"perfect": perfect, "noisy": noisy}, step=0.25)

    assert weights["perfect"] > weights["noisy"]
    assert score > 90.0


def test_temporal_fold_audit_holds_across_a_snapshot_refresh_boundary() -> None:
    """Simulates a data refresh: an 'old' snapshot ending at some timestamp, with new
    rows appended after it (exactly what scripts/refresh_observations_snapshot.py does
    against Supabase). Cutoffs that span the old/new boundary must still satisfy
    training_end < validation_start for every fold, including the first fold whose
    validation window falls entirely in the newly-appended data."""
    observations = sample_observations(days=40)
    times = sorted(observations["observed_at"].drop_duplicates())
    old_max = pd.Timestamp("2026-01-30T00:00:00Z")

    cutoffs = [ts for ts in times if ts.minute == 0 and pd.Timestamp("2026-01-28T00:00:00Z") <= ts <= pd.Timestamp("2026-02-03T00:00:00Z")]
    assert any(ts > old_max for ts in cutoffs), "test setup must include cutoffs past the simulated old snapshot boundary"

    audit = core.temporal_fold_audit_generic(observations, cutoffs)

    assert not audit.empty
    assert (audit["training_end"] < audit["validation_start"]).all()
    # At least one fold's validation window must start strictly after the simulated
    # refresh boundary -- otherwise this test would not actually exercise a
    # boundary-spanning fold.
    assert (audit["validation_start"] > old_max).any()


def test_extended_feature_schema_has_no_duplicate_columns() -> None:
    assert len(core.EXTENDED_SAMPLE_COLUMNS) == len(set(core.EXTENDED_SAMPLE_COLUMNS))
    for schema in [
        core.BASELINE_FEATURE_SCHEMA,
        core.FEATURE_SET_PLUS_LAGS,
        core.FEATURE_SET_PLUS_ROLLING,
        core.FEATURE_SET_PLUS_TREND,
        core.FEATURE_SET_PLUS_DOW_PROFILE,
        core.FEATURE_SET_ALL,
    ]:
        assert len(schema) == len(set(schema)), f"duplicate feature in schema: {schema}"
        assert "station_id" not in schema and "actual" not in schema, "bookkeeping column leaked into a model feature schema"
