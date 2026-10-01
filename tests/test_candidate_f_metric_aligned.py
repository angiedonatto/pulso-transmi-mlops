from __future__ import annotations

import pandas as pd

from scripts.candidate_f_metric_aligned import (
    VariantSpec,
    build_samples,
    category_maps,
    fold_plan,
    leakage_audit,
    matrix,
    outer_cutoffs,
    profile_denominator,
    station_metric_weights,
    train_cutoffs_for_plan,
)
from scripts.model_improvement_candidates import load_observations_from_frame


def sample_observations(days: int = 24) -> pd.DataFrame:
    rows = []
    observed_at = pd.date_range("2026-01-01T00:00:00Z", periods=days * 96, freq="15min")
    for station_id, offset in [("03000", 100), ("05000", 400), ("07111", 700)]:
        for idx, timestamp in enumerate(observed_at):
            rows.append(
                {
                    "station_id": station_id,
                    "observed_at": timestamp,
                    "demand": offset + (idx % 96) + (idx // 96),
                }
            )
    return load_observations_from_frame(pd.DataFrame(rows))


def test_profile_statistics_never_include_target_row() -> None:
    observations = sample_observations()
    cutoff = pd.Timestamp("2026-01-20T00:00:00Z")
    samples = build_samples(observations, [cutoff])
    row = samples[(samples["station_id"] == "03000") & (samples["horizon_minutes"] == 60)].iloc[0]
    history = observations[
        (observations["station_id"] == "03000")
        & (observations["quarter_hour_block"] == row["target_quarter_hour_block"])
        & (observations["observed_at"] < row["target_at"])
    ]

    assert row["profile_mean"] == history["demand"].mean()
    assert row["actual"] not in history["demand"].tail(1).tolist()


def test_station_weights_use_training_only() -> None:
    train = pd.DataFrame({"station_id": ["03000", "03000", "05000"], "actual": [100.0, 200.0, 900.0]})
    validation = pd.DataFrame({"station_id": ["03000"], "actual": [999999.0]})

    before = station_metric_weights(train)
    after = station_metric_weights(train)

    assert validation["actual"].iloc[0] == 999999.0
    assert before.tolist() == after.tolist()


def test_normalized_target_denominator_uses_past_only() -> None:
    observations = sample_observations()
    cutoff = pd.Timestamp("2026-01-20T00:00:00Z")
    baseline = build_samples(observations, [cutoff])
    mutated = observations.copy()
    target_at = cutoff + pd.Timedelta(minutes=60)
    mutated.loc[(mutated["station_id"] == "03000") & (mutated["observed_at"] == target_at), "demand"] = 999999.0
    changed = build_samples(mutated, [cutoff])

    left = baseline[(baseline["station_id"] == "03000") & (baseline["horizon_minutes"] == 60)]
    right = changed[(changed["station_id"] == "03000") & (changed["horizon_minutes"] == 60)]

    assert profile_denominator(left, VariantSpec("x")).iloc[0] == profile_denominator(right, VariantSpec("x")).iloc[0]
    assert left["actual"].iloc[0] != right["actual"].iloc[0]


def test_categorical_mapping_uses_training_known_categories() -> None:
    train = build_samples(sample_observations(), [pd.Timestamp("2026-01-20T00:00:00Z")])
    train = train[train["station_id"] != "07111"].copy()
    fold = build_samples(sample_observations(), [pd.Timestamp("2026-01-20T01:00:00Z")])
    fold = fold[fold["station_id"] == "07111"].copy()
    maps = category_maps(train)

    x = matrix(fold, VariantSpec("cat", feature_representation="categorical_cyclic"), maps)

    assert "station_id_07111" not in x.columns
    assert any(column.startswith("station_id_03000") for column in x.columns)


def test_nested_folds_do_not_overlap_and_oof_predictions_are_oof() -> None:
    observations = sample_observations(days=32)
    cutoffs = outer_cutoffs(observations)
    start = cutoffs[0].floor("D")
    plan = fold_plan(observations, cutoffs, start)
    train_samples = build_samples(observations, train_cutoffs_for_plan(observations, plan))

    assert plan.training_end < plan.evaluation_start
    assert plan.inner_end < plan.evaluation_start
    assert train_samples["target_at"].max() < plan.evaluation_start


def test_future_demand_mutation_does_not_alter_current_features() -> None:
    observations = sample_observations()
    cutoff = pd.Timestamp("2026-01-20T00:00:00Z")
    target_at = cutoff + pd.Timedelta(minutes=60)
    baseline = build_samples(observations, [cutoff])
    mutated = observations.copy()
    mutated.loc[(mutated["station_id"] == "05000") & (mutated["observed_at"] == target_at - pd.Timedelta(minutes=15)), "demand"] = 123456.0
    changed = build_samples(mutated, [cutoff])
    left = baseline[(baseline["station_id"] == "05000") & (baseline["horizon_minutes"] == 60)].iloc[0]
    right = changed[(changed["station_id"] == "05000") & (changed["horizon_minutes"] == 60)].iloc[0]

    assert pd.isna(left["lag_15"])
    assert pd.isna(right["lag_15"])
    assert left["lag_60"] == right["lag_60"]


def test_leakage_audit_passes_on_synthetic_data() -> None:
    observations = sample_observations(days=32)
    audit = leakage_audit(observations, outer_cutoffs(observations))

    assert audit["passed"] is True
    assert audit["tuning_sees_no_outer_evaluation_data"] is True
