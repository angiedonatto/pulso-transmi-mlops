from __future__ import annotations

import pandas as pd

from scripts.candidate_g_drift_robustness_lab import (
    GSpec,
    calibrate,
    feedback_ratio,
    leakage_check,
    level_ratio,
    normalize_station,
    precompute_signals,
    profile_map,
    trend_ratio,
)
from scripts.model_improvement_candidates import load_observations_from_frame


def observations() -> pd.DataFrame:
    rows = []
    times = pd.date_range("2026-01-01T00:00:00Z", periods=40, freq="15min")
    for station_id, offset in [("05000", 100), ("07111", 300)]:
        for idx, observed_at in enumerate(times):
            rows.append({"station_id": station_id, "observed_at": observed_at, "demand": offset + idx})
    frame = load_observations_from_frame(pd.DataFrame(rows))
    frame["station_id"] = frame["station_id"].map(normalize_station)
    return frame


def prediction_frame() -> pd.DataFrame:
    cutoff = pd.Timestamp("2026-01-01T08:00:00Z")
    rows = []
    for station_id in ["05000", "07111"]:
        for order, horizon in enumerate([15, 30, 45, 60]):
            target_at = cutoff + pd.Timedelta(minutes=horizon)
            rows.append(
                {
                    "cycle_id": "c1",
                    "station_id": station_id,
                    "data_cutoff": cutoff,
                    "target_at": target_at,
                    "target_order": order,
                    "horizon_minutes": horizon,
                    "actual": 120.0,
                    "prediction": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_level_and_trend_ignore_observations_after_cutoff() -> None:
    obs = observations()
    profiles = profile_map(obs, end=pd.Timestamp("2026-01-01T04:00:00Z"))
    row = prediction_frame().iloc[0]
    baseline_level = level_ratio(row, obs[obs["observed_at"] <= row["data_cutoff"]], profiles, 8)
    baseline_trend = trend_ratio(row, obs[obs["observed_at"] <= row["data_cutoff"]], 8)
    mutated = obs.copy()
    mutated.loc[mutated["observed_at"] > row["data_cutoff"], "demand"] = 999999.0

    assert level_ratio(row, mutated[mutated["observed_at"] <= row["data_cutoff"]], profiles, 8) == baseline_level
    assert trend_ratio(row, mutated[mutated["observed_at"] <= row["data_cutoff"]], 8) == baseline_trend


def test_feedback_uses_only_resolved_predictions_before_cutoff() -> None:
    frame = prediction_frame()
    later = frame.copy()
    later["cycle_id"] = "future"
    later["data_cutoff"] = pd.Timestamp("2026-01-01T09:00:00Z")
    later["target_at"] = pd.Timestamp("2026-01-01T09:15:00Z")
    later["actual"] = 999999.0
    full = pd.concat([frame, later], ignore_index=True)
    row = later.iloc[0].copy()
    row["data_cutoff"] = pd.Timestamp("2026-01-01T08:30:00Z")

    assert feedback_ratio(row, full, "median") == feedback_ratio(row, frame, "median")


def test_combined_calibration_clips_extreme_factor() -> None:
    frame = prediction_frame()
    obs = observations()
    profiles = profile_map(obs, end=pd.Timestamp("2026-01-01T04:00:00Z"))
    precomputed = precompute_signals(frame, obs, profiles)
    precomputed["g1_w4"] = 10.0
    adjusted = calibrate(precomputed, GSpec("G1", True, False, False, g1_window=4, alpha=1.0, clip_low=0.5, clip_high=2.0))

    assert adjusted["calibration_factor"].max() == 2.0


def test_leakage_check_requires_historical_selection_before_live() -> None:
    hist = prediction_frame()
    hist["data_cutoff"] = pd.Timestamp("2026-01-01T08:00:00Z")
    live = prediction_frame()
    live["data_cutoff"] = pd.Timestamp("2026-01-01T10:00:00Z")
    live["target_at"] = live["data_cutoff"] + pd.to_timedelta(live["horizon_minutes"], unit="m")

    assert leakage_check(hist, live, pd.Timestamp("2026-01-01T09:00:00Z")) == "passed"
    assert leakage_check(hist, live, pd.Timestamp("2026-01-01T11:00:00Z")) == "failed"
