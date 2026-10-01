from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import drift_regime as dr
from drift_monitor import _accuracy_by


START = pd.Timestamp("2026-08-03T05:00:00Z")  # Monday 00:00 Bogota


def diurnal(slot: np.ndarray) -> np.ndarray:
    hour = slot / 4.0
    return 100 + 400 * np.exp(-((hour - 7.0) ** 2) / 2.0) + 300 * np.exp(-((hour - 18.0) ** 2) / 2.0)


def synthetic_observations(days: int = 42, stations: tuple[str, ...] = ("03000", "05100"), noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(START, periods=days * 96, freq="15min")
    local = times.tz_convert(dr.BOGOTA_TZ)
    slot = np.asarray(local.hour * 4 + local.minute // 15)
    rows = []
    for station in stations:
        demand = diurnal(slot) * (1 + noise * rng.standard_normal(len(times)))
        rows.append(pd.DataFrame({"station_id": station, "observed_at": times, "demand": np.round(np.maximum(demand, 0))}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ change points


def test_binary_segmentation_finds_a_known_shift_and_nothing_in_noise() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.05, 200)
    step = noise + np.r_[np.zeros(120), np.full(80, 0.6)]
    sigma = dr.robust_sigma(step)
    penalty = 4 * sigma**2 * math.log(step.size)
    points = dr.binary_segmentation(step, min_size=6, penalty=penalty)
    assert points and abs(points[0] - 120) <= 2
    assert dr.binary_segmentation(noise, min_size=6, penalty=4 * dr.robust_sigma(noise) ** 2 * math.log(noise.size)) == []


def test_prune_removes_changes_below_the_minimum_effect() -> None:
    values = np.r_[np.zeros(30), np.full(30, 0.02), np.full(30, 1.0)]
    assert dr.prune_change_points(values, [30, 60], min_effect=0.1) == [60]


def test_segment_series_and_onset_report_the_first_material_departure() -> None:
    index = pd.date_range("2026-09-01", periods=120, freq="h", tz="UTC")
    series = pd.Series(np.r_[np.zeros(80), np.full(40, math.log(2.0))], index=index)
    segments = dr.segment_series(series, dr.ChangePointConfig(min_segment_hours=6))
    onset = dr.onset_from_segments(segments, min_effect=math.log(1.1))
    assert onset["onset"] == index[80]
    assert segments.iloc[-1]["ratio"] == pytest.approx(2.0)


def test_seasonal_baseline_is_causal() -> None:
    wide = dr.demand_matrix(dr.prepare_observations(synthetic_observations(days=42, noise=0.1)))
    base = dr.seasonal_baseline(wide, weeks=4)
    cut = wide.index[35 * 96]
    mutated = wide.copy()
    mutated.loc[mutated.index > cut] *= 9.0
    base_mutated = dr.seasonal_baseline(mutated, weeks=4)
    past = base.index <= cut
    pd.testing.assert_frame_equal(base[past], base_mutated[past])
    t = wide.index[30 * 96 + 10]
    expected = np.median([wide.loc[t - pd.Timedelta(days=7 * k), "03000"] for k in range(1, 5)])
    assert base.loc[t, "03000"] == pytest.approx(expected)


# ------------------------------------------------------------------ windows


def test_reference_period_ends_at_local_midnight_before_onset_and_has_28_days() -> None:
    onset = pd.Timestamp("2026-09-11T03:00:00Z")  # 2026-09-10 22:00 Bogota
    ref = dr.reference_period(onset, pd.Timestamp("2026-07-26T05:00:00Z"), days=28)
    prepared = dr.prepare_observations(synthetic_observations(days=42))
    inside = prepared[ref.mask(prepared["observed_at"])]
    assert inside["observed_at"].max().tz_convert(dr.BOGOTA_TZ) == pd.Timestamp("2026-09-09 23:45", tz=dr.BOGOTA_TZ)
    assert inside["local_date"].nunique() == 28


def test_trailing_window_is_left_open_right_closed() -> None:
    prepared = dr.prepare_observations(synthetic_observations(days=10))
    anchor = prepared["observed_at"].max()
    window = dr.trailing_windows(anchor, [6])[0]
    rows = prepared[window.mask(prepared["observed_at"])]
    assert len(rows) == 24 * prepared["station_id"].nunique()
    assert rows["observed_at"].max() == anchor
    assert dr.window_coverage(prepared, window)["sufficient"]


def test_null_windows_stay_inside_reference_and_match_daytype() -> None:
    reference = dr.Window("reference", pd.Timestamp("2026-08-03T04:45:00Z"), pd.Timestamp("2026-08-31T04:45:00Z"), "reference")
    window = dr.Window("last_6h", pd.Timestamp("2026-09-10T02:00:00Z"), pd.Timestamp("2026-09-10T08:00:00Z"), "trailing")
    nulls = dr.same_clock_null_windows(window, reference)
    assert nulls
    for null in nulls:
        assert null.start >= reference.start and null.end <= reference.end
        assert dr.daytype_signature(null) == dr.daytype_signature(window)
        assert null.hours == window.hours


# ------------------------------------------------------------------ data drift metrics


def _profile_and_windows(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dr.Window]:
    prepared = dr.prepare_observations(obs)
    reference = dr.Window("reference", START - dr.SLOT, START + pd.Timedelta(days=28) - dr.SLOT, "reference")
    return prepared, dr.reference_profile(prepared, reference), reference


def test_pure_level_shift_is_level_not_profile() -> None:
    obs = synthetic_observations(days=35)
    late = obs["observed_at"] >= START + pd.Timedelta(days=33)
    obs.loc[late, "demand"] *= 2.0
    prepared, profile, _ = _profile_and_windows(obs)
    window = dr.trailing_windows(prepared["observed_at"].max(), [48])[0]
    metrics = dr.station_window_metrics(prepared, profile, window).set_index("station_id")
    assert metrics.loc["03000", "level_ratio"] == pytest.approx(2.0, rel=0.02)
    assert metrics.loc["03000", "hourly_ratio_dispersion"] < 0.05
    assert metrics.loc["03000", "timing_shift_minutes"] == 0


def test_time_shifted_profile_is_detected_as_later_timing() -> None:
    obs = synthetic_observations(days=35)
    late = obs["observed_at"] >= START + pd.Timedelta(days=33)
    shifted = obs.copy()
    shifted["observed_at"] = shifted["observed_at"] + pd.Timedelta(minutes=60)  # same pattern one hour later
    obs.loc[late, "demand"] = obs.loc[late, ["station_id", "observed_at"]].merge(shifted, on=["station_id", "observed_at"], how="left")["demand"].to_numpy()
    obs = obs.dropna(subset=["demand"])
    prepared, profile, _ = _profile_and_windows(obs)
    window = dr.trailing_windows(prepared["observed_at"].max(), [24])[0]
    metrics = dr.station_window_metrics(prepared, profile, window).set_index("station_id")
    assert metrics.loc["03000", "timing_shift_minutes"] == 60
    assert metrics.loc["03000", "timing_corr_gain"] > 0
    peaks = dr.daily_peaks(prepared, profile, [prepared["local_date"].max()])
    am = peaks[(peaks["station_id"] == "03000") & (peaks["band"] == "am")].iloc[0]
    assert am["peak_shift_minutes"] == 60


def test_lag_metrics_measure_naive_lag_accuracy() -> None:
    obs = synthetic_observations(days=21)
    wide = dr.demand_matrix(dr.prepare_observations(obs))
    window = dr.trailing_windows(wide.index.max(), [24])[0]
    lags = dr.lag_metrics(wide, window).set_index(["station_id", "lag"])
    assert lags.loc[("03000", "lag_96"), "naive_accuracy"] == pytest.approx(100.0)
    assert lags.loc[("03000", "lag_weekly"), "naive_accuracy"] == pytest.approx(100.0)
    assert lags.loc[("03000", "lag_15"), "naive_accuracy"] < 100.0


def test_lag_reliability_labels() -> None:
    window = pd.DataFrame({"station_id": ["a", "b", "c"], "lag": "lag_96", "naive_accuracy": [90.0, 80.0, 60.0], "lag_corr": [0.9, 0.8, 0.5]})
    null = pd.DataFrame({"station_id": ["a", "a", "b", "b", "c", "c"], "lag": "lag_96", "naive_accuracy": [88, 92, 88, 92, 88, 92], "lag_corr": 0.9})
    labels = dr.lag_reliability(window, null).set_index("station_id")["reliability"].to_dict()
    assert labels == {"a": "reliable", "b": "degraded", "c": "broken"}


# ------------------------------------------------------------------ predictions


def _prediction_tables(*, mismatch_horizon: bool = False, drop_one: bool = False) -> dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp("2026-08-20T10:00:00Z")
    targets = pd.DataFrame(
        [{"cycle_id": "c1", "station_id": s, "target_at": cutoff + pd.Timedelta(minutes=h), "horizon_minutes": h, "target_order": i}
         for i, (s, h) in enumerate([(s, h) for s in ("03000", "05100") for h in (15, 30)])]
    )
    predictions = targets.rename(columns={"target_at": "target_observed_at"})[["cycle_id", "station_id", "target_observed_at", "horizon_minutes"]].copy()
    predictions["model_id"] = "g"
    predictions["prediction"] = 120.0
    if drop_one:
        predictions = predictions.iloc[1:]
    if mismatch_horizon:
        targets.loc[0, "horizon_minutes"] = 45
    obs = pd.DataFrame({"station_id": targets["station_id"], "observed_at": targets["target_at"], "demand": 100.0})
    return {
        "predictions": predictions,
        "forecast_cycles": pd.DataFrame({"cycle_id": ["c1"], "data_cutoff": [cutoff], "expected_predictions": [4]}),
        "forecast_cycle_targets": targets,
        "observations": obs,
        "submission_receipts": pd.DataFrame({"cycle_id": ["c1"], "model_id": ["g"], "status": ["accepted"], "is_official": [True]}),
        "model_versions": pd.DataFrame({"model_id": ["g"], "model_name": ["candidate_g"]}),
    }


def test_resolve_predictions_uses_actual_minus_prediction_and_respects_cutoff() -> None:
    scored = dr.resolve_predictions(**_prediction_tables())
    assert len(scored) == 4
    assert (scored["residual"] == -20.0).all()  # over-prediction is negative
    assert (scored["target_at"] > scored["data_cutoff"]).all()
    assert scored["is_official"].all()
    summary = dr.error_summary(scored, [])
    assert summary.loc[0, "accuracy"] == pytest.approx(80.0)
    assert summary.loc[0, "relative_bias"] == pytest.approx(-0.2)
    assert summary.loc[0, "pred_actual_ratio"] == pytest.approx(1.2)


def test_resolve_predictions_drops_incomplete_cycles_and_rejects_bad_horizons() -> None:
    assert dr.resolve_predictions(**_prediction_tables(drop_one=True)).empty
    with pytest.raises(ValueError, match="horizon_minutes"):
        dr.resolve_predictions(**_prediction_tables(mismatch_horizon=True))


def test_error_summary_matches_canonical_accuracy_and_rescored_columns() -> None:
    scored = dr.resolve_predictions(**_prediction_tables())
    canonical = _accuracy_by(scored.assign(residual=scored["prediction"] - scored["actual"]), ["station_id"])
    ours = dr.error_summary(scored, ["station_id"])
    np.testing.assert_allclose(ours["accuracy"], canonical["accuracy"])
    # a different prediction column must be re-scored, not reuse the stale abs_error
    perfect = dr.error_summary(scored.assign(prediction=scored["actual"]), [])
    assert perfect.loc[0, "accuracy"] == pytest.approx(100.0)


def test_paired_comparison_requires_shared_cycles() -> None:
    scored = dr.resolve_predictions(**_prediction_tables())
    assert dr.paired_comparison(scored, "g", "other")["status"] == "no_comparable_cycles"
    both = pd.concat([scored, scored.assign(model_id="i", prediction=100.0)], ignore_index=True)
    result = dr.paired_comparison(both, "g", "i")
    assert result["shared_cycles"] == 1
    assert result["mean_cycle_delta_pp"] == pytest.approx(20.0)
    assert result["challenger_wins"] == 1


# ------------------------------------------------------------------ recommendations


BASE_ROW = {
    "sustained_level_shift": "none", "sustained_profile_shift": "none", "sustained_variance_shift": "none",
    "shock_level_shift": "none", "shock_profile_shift": "none",
    "lag15_reliability": "reliable", "lag30_reliability": "reliable", "lag60_reliability": "reliable",
    "lag96_reliability": "reliable", "weekly_lag_reliability": "reliable",
    "new_regime_day_consistency": 0.95, "g_sustained_rows": 160, "g_sustained_accuracy": 90.0,
    "g_sustained_relative_bias": 0.0, "g_shock_rows": 13, "g_shock_accuracy": 88.0, "new_regime_hours": 40.0,
}


def test_shift_absorbed_by_g_keeps_current_behavior() -> None:
    response = dr.recommend_station_response(BASE_ROW | {"sustained_level_shift": "moderate_up"})
    assert response.primary == "keep current behavior"


def test_biased_level_shift_without_profile_change_calls_for_recalibration() -> None:
    row = BASE_ROW | {"sustained_level_shift": "high_down", "g_sustained_accuracy": 70.0, "g_sustained_relative_bias": -0.2}
    assert dr.recommend_station_response(row).primary == "station recalibration"


def test_repeatable_profile_shift_with_broken_lag96() -> None:
    row = BASE_ROW | {
        "sustained_level_shift": "extreme_up", "sustained_profile_shift": "strong:hour_dependent_change",
        "lag96_reliability": "broken", "g_sustained_accuracy": 74.0, "g_sustained_relative_bias": -0.1,
    }
    response = dr.recommend_station_response(row)
    assert response.primary == "recent profile refit"
    assert "reduce lag96 reliance" in response.responses
    assert "local retraining" not in response.responses  # regime only 40h old
    assert dr.recommend_station_response(row | {"new_regime_hours": 96.0}).primary == "local retraining"


def test_unstable_profile_and_shock_call_for_robust_baseline() -> None:
    row = BASE_ROW | {"sustained_profile_shift": "strong:shape_loss", "new_regime_day_consistency": 0.4, "g_sustained_accuracy": 70.0}
    assert dr.recommend_station_response(row).primary == "robust recent baseline"
    shock = dr.recommend_station_response(BASE_ROW | {"shock_level_shift": "extreme_up", "g_shock_accuracy": 20.0})
    assert shock.sustained == "keep current behavior"
    assert shock.shock == "robust recent baseline"
    assert shock.primary == "robust recent baseline"


def test_missing_evaluation_rows_mean_insufficient_evidence() -> None:
    response = dr.recommend_station_response(BASE_ROW | {"g_sustained_rows": 0, "g_shock_rows": 0})
    assert response.primary == "insufficient evidence"


def test_timing_shift_on_short_windows_is_not_used_to_classify() -> None:
    config = dr.ClassificationConfig()
    row = pd.Series({"hourly_ratio_dispersion": 0.0, "timing_shift_minutes": 345.0, "timing_corr_gain": 0.5, "shape_corr": 0.99, "null_shape_corr_min": 0.95})
    assert dr.classify_profile(row.copy().rename(None), config).startswith("moderate:timing_later")
    assert dr.classify_profile(pd.concat([row, pd.Series({"window_slots": 13})]), config) == "none"


def test_daily_volatility_is_level_invariant() -> None:
    obs = synthetic_observations(days=3, noise=0.1, seed=4)
    doubled = obs.assign(demand=obs["demand"] * 2)
    base = dr.daily_volatility(dr.prepare_observations(obs)).set_index(["station_id", "local_date"])["volatility"]
    scaled = dr.daily_volatility(dr.prepare_observations(doubled)).set_index(["station_id", "local_date"])["volatility"]
    np.testing.assert_allclose(base, scaled, rtol=0.02)


def test_level_collapse_with_profile_change_still_gets_recalibration() -> None:
    row = BASE_ROW | {
        "sustained_level_shift": "extreme_down", "sustained_profile_shift": "strong:hour_dependent_change",
        "new_regime_day_consistency": 0.6, "g_sustained_accuracy": 55.0, "g_sustained_relative_bias": -0.2,
    }
    response = dr.recommend_station_response(row)
    assert response.primary == "station recalibration"
    assert "robust recent baseline" in response.responses
