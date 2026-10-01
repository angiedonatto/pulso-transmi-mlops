from __future__ import annotations

import pandas as pd

from scripts.candidate_g_station_guard_lab import (
    GuardVariant,
    apply_variant,
    attach_blocks,
    build_recency_profiles,
    leakage_audit,
    promotion_verdict,
)


def prediction_frame() -> pd.DataFrame:
    rows = []
    cutoffs = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h")
    for cycle_idx, cutoff in enumerate(cutoffs):
        for station_id, factor in [("05100", 1.20), ("07105", 1.20)]:
            for horizon in (15, 30):
                target_at = cutoff + pd.Timedelta(minutes=horizon)
                rows.append(
                    {
                        "cycle_id": f"c{cycle_idx}",
                        "station_id": station_id,
                        "data_cutoff": cutoff,
                        "target_at": target_at,
                        "horizon_minutes": horizon,
                        "target_hour": target_at.tz_convert("America/Bogota").hour,
                        "actual": 110.0,
                        "prediction": 100.0,
                        "prediction_g": 100.0 * factor,
                        "calibration_factor": factor,
                    }
                )
    return pd.DataFrame(rows)


def test_stable_station_shrink_moves_factor_toward_one() -> None:
    frame = prediction_frame()
    guarded = apply_variant(frame, GuardVariant("station_residual_guard", "unchanged", "shrink"))
    stable = guarded[guarded["station_id"] == "07105"]
    alert = guarded[guarded["station_id"] == "05100"]

    assert stable["station_residual_guard_factor"].iloc[0] == 1.10
    assert alert["station_residual_guard_factor"].iloc[0] == 1.20


def test_recency_profile_uses_only_resolved_targets() -> None:
    frame = prediction_frame()
    future_same_cycle = frame.iloc[[0]].copy()
    future_same_cycle["target_at"] = future_same_cycle["data_cutoff"] + pd.Timedelta(minutes=45)
    future_same_cycle["actual"] = 999999.0
    frame = pd.concat([frame, future_same_cycle], ignore_index=True)

    profiled = build_recency_profiles(frame)
    unresolved = profiled[profiled["actual"] == 999999.0].iloc[0]

    assert unresolved["recency_profile_prediction"] == unresolved["prediction_g"]
    assert profiled["recency_profile_prediction"].notna().all()


def test_promotion_verdict_requires_all_gates() -> None:
    comparison = pd.DataFrame(
        [
            {
                "variant": "candidate_g_current",
                "recent_delta_vs_current_pp": 0.0,
                "delta_vs_current_pp": 0.0,
                "alert_abs_bias_delta": 0.0,
                "stable_min_delta": 0.0,
            },
            {
                "variant": "better_recent_but_more_bias",
                "recent_delta_vs_current_pp": 2.0,
                "delta_vs_current_pp": 1.0,
                "alert_abs_bias_delta": 0.1,
                "stable_min_delta": 0.0,
            },
        ]
    )

    verdict = promotion_verdict(comparison)

    assert verdict["promote"] is False
    assert verdict["recommended_variant"] == "candidate_g_current"


def test_leakage_audit_checks_target_after_cutoff() -> None:
    frame = attach_blocks(prediction_frame())

    assert leakage_audit(frame)["target_after_data_cutoff"] is True
