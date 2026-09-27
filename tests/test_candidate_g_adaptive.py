from __future__ import annotations

import pandas as pd

from candidate_g_adaptive import (
    CLIP_HIGH,
    CLIP_LOW,
    MODEL_VERSION,
    build_candidate_g_artifact,
    build_candidate_g_predictions,
    candidate_g_trace_rows,
    g2_factor,
    load_candidate_g_artifact,
    prepare_feedback_frame,
    save_candidate_g_artifact,
)
from forecast_cycle_sync import CycleTarget, ForecastCycle
from scripts.model_improvement_candidates import load_observations_from_frame


def observations(days: int = 18) -> pd.DataFrame:
    rows = []
    times = pd.date_range("2026-01-01T00:00:00Z", periods=days * 96, freq="15min")
    for station_id, offset in [("05000", 100), ("07111", 300)]:
        for idx, observed_at in enumerate(times):
            rows.append({"station_id": station_id, "observed_at": observed_at, "demand": offset + idx % 96})
    return load_observations_from_frame(pd.DataFrame(rows))


def cycle(cutoff: str = "2026-01-17T12:00:00Z") -> ForecastCycle:
    targets = []
    order = 0
    for station_id in ["05000", "07111"]:
        for horizon in [15, 30, 45, 60]:
            target_at = pd.Timestamp(cutoff).tz_convert("UTC") + pd.Timedelta(minutes=horizon)
            targets.append(
                CycleTarget(
                    station_id=station_id,
                    target_at=target_at.isoformat().replace("+00:00", "Z"),
                    horizon_minutes=horizon,
                    target_order=order,
                    raw_target={},
                )
            )
            order += 1
    return ForecastCycle("c1", cutoff, "", len(targets), targets, {})


def feedback(cutoff: str = "2026-01-17T12:00:00Z") -> pd.DataFrame:
    before = pd.Timestamp(cutoff).tz_convert("UTC") - pd.Timedelta(minutes=15)
    future = pd.Timestamp(cutoff).tz_convert("UTC") + pd.Timedelta(minutes=15)
    return pd.DataFrame(
        [
            {"station_id": "05000", "target_at": before, "base_prediction": 100.0, "actual": 200.0},
            {"station_id": "05000", "target_at": future, "base_prediction": 100.0, "actual": 9999.0},
        ]
    )


def test_no_future_feedback_and_unresolved_targets_excluded() -> None:
    cutoff = pd.Timestamp("2026-01-17T12:00:00Z")
    prepared = prepare_feedback_frame(feedback(), cutoff)

    assert len(prepared) == 1
    assert prepared["actual"].iloc[0] == 200.0


def test_missing_station_feedback_fallback_factor_one() -> None:
    factor, samples = g2_factor("07111", prepare_feedback_frame(feedback(), pd.Timestamp("2026-01-17T12:00:00Z")))

    assert factor == 1.0
    assert samples == 0


def test_clipping_exact_bounds() -> None:
    high = pd.DataFrame([{"station_id": "05000", "target_at": "2026-01-01T00:00:00Z", "base_prediction": 1.0, "actual": 999.0}])
    low = pd.DataFrame([{"station_id": "05000", "target_at": "2026-01-01T00:00:00Z", "base_prediction": 999.0, "actual": 1.0}])

    assert g2_factor("05000", prepare_feedback_frame(high, pd.Timestamp("2026-01-02T00:00:00Z")))[0] == CLIP_HIGH
    assert g2_factor("05000", prepare_feedback_frame(low, pd.Timestamp("2026-01-02T00:00:00Z")))[0] == CLIP_LOW


def test_cold_start_uses_base_prediction_and_trace() -> None:
    artifact = build_candidate_g_artifact()
    obs = observations()
    cutoff = pd.Timestamp(cycle().data_cutoff).tz_convert("UTC")
    preds = build_candidate_g_predictions(cycle(), MODEL_VERSION, artifact, obs[obs["observed_at"] <= cutoff], feedback_frame=None)
    traces = candidate_g_trace_rows(preds)

    assert all(item.calibration_factor == 1.0 for item in preds)
    assert all(item.feedback_samples == 0 for item in preds)
    assert traces[0]["base_prediction"] == traces[0]["final_prediction"]


def test_deterministic_state_reconstruction_and_restart_idempotency() -> None:
    artifact = build_candidate_g_artifact()
    obs = observations()
    cutoff = pd.Timestamp(cycle().data_cutoff).tz_convert("UTC")
    obs = obs[obs["observed_at"] <= cutoff]
    fb = feedback()

    first = build_candidate_g_predictions(cycle(), MODEL_VERSION, artifact, obs, feedback_frame=fb)
    second = build_candidate_g_predictions(cycle(), MODEL_VERSION, artifact, obs, feedback_frame=fb)

    assert candidate_g_trace_rows(first) == candidate_g_trace_rows(second)


def test_artifact_roundtrip(tmp_path) -> None:
    uri = str(tmp_path / "candidate_g_v1")
    artifact = build_candidate_g_artifact(artifact_uri=uri)
    save_candidate_g_artifact(artifact, artifact_uri=uri)
    loaded = load_candidate_g_artifact(uri)

    assert loaded.metadata["version"] == MODEL_VERSION
    assert loaded.metadata["adaptive_layer"]["clip"] == [CLIP_LOW, CLIP_HIGH]
