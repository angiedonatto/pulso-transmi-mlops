from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import candidate_i_snr_gate as ci
from candidate_b_shadow import ShadowCandidateError
from forecast_cycle_sync import CycleTarget, ForecastCycle


CUTOFF = pd.Timestamp("2026-09-20T12:00:00Z")


def _history(log_ratios: list[float], *, cutoff: pd.Timestamp = CUTOFF, base: float = 100.0) -> pd.DataFrame:
    n = len(log_ratios)
    target_at = [cutoff - ci.SLOT * (n - 1 - i) for i in range(n)]
    return pd.DataFrame({"station_id": "s1", "target_at": target_at, "base_prediction": base, "actual": base * np.exp(log_ratios)})


def _observations(days: int = 20, stations: tuple[str, ...] = ("01000", "02000", "03000"), drift: dict[str, float] | None = None, drift_hours: int = 10) -> pd.DataFrame:
    start = CUTOFF.floor("D") - pd.Timedelta(days=days - 1)
    times = pd.date_range(start, CUTOFF, freq="15min", tz="UTC")
    rng = np.random.default_rng(3)
    rows = []
    for k, sid in enumerate(stations):
        local_hour = (times - pd.Timedelta(hours=5)).hour + (times - pd.Timedelta(hours=5)).minute / 60.0
        demand = 200.0 * (k + 1) * (1.2 + np.sin((local_hour - 6.0) / 24.0 * 2 * np.pi)) * rng.lognormal(0.0, 0.05, len(times))
        if drift and sid in drift:
            demand = np.where(times > CUTOFF - pd.Timedelta(hours=drift_hours), demand * drift[sid], demand)
        rows.append(pd.DataFrame({"station_id": sid, "observed_at": times, "demand": demand}))
    return pd.concat(rows, ignore_index=True)


def _cycle(stations: tuple[str, ...] = ("01000", "02000", "03000")) -> ForecastCycle:
    targets = [
        CycleTarget(station_id=sid, target_at=(CUTOFF + pd.Timedelta(minutes=h)).isoformat().replace("+00:00", "Z"), horizon_minutes=h, target_order=i * 4 + j, raw_target={})
        for i, sid in enumerate(stations)
        for j, h in enumerate((15, 30, 45, 60))
    ]
    return ForecastCycle(cycle_id="c1", data_cutoff=CUTOFF.isoformat().replace("+00:00", "Z"), closes_at="", expected_predictions=len(targets), targets=targets, raw_cycle={})


def test_level_signal_matches_the_documented_formula() -> None:
    logs = list(np.random.default_rng(0).normal(0.0, 0.1, 24)) + [np.log(0.6)] * 8
    history = _history(logs)
    signal = ci.level_signal(history, CUTOFF)
    ratios = np.exp(np.array(logs[-8:]))
    level = np.average(ratios, weights=np.geomspace(0.2, 1.0, 8))
    z = abs(np.mean(logs[-8:])) / (np.std(logs[-32:], ddof=1) / math.sqrt(8))
    gate = z**2 / (z**2 + ci.KAPPA)
    assert signal.level_ratio == pytest.approx(level)
    assert signal.z == pytest.approx(z)
    assert signal.factor == pytest.approx(float(np.clip(np.exp(gate * np.log(level)), ci.CLIP_LOW, ci.CLIP_HIGH)))
    assert signal.history_slots == 32


def test_gate_ignores_noise_and_corrects_a_persistent_shift() -> None:
    noise = ci.level_signal(_history(list(np.random.default_rng(1).normal(0.0, 0.15, 32))), CUTOFF)
    shifted = ci.level_signal(_history(list(np.random.default_rng(1).normal(0.0, 0.05, 24)) + [np.log(0.5)] * 8), CUTOFF)
    assert noise.gate < 0.5 and abs(np.log(noise.factor)) < 0.1
    assert shifted.gate > 0.85 and shifted.factor < 0.6


def test_level_signal_never_reads_slots_after_the_cutoff() -> None:
    history = _history(list(np.random.default_rng(2).normal(0.0, 0.1, 32)))
    future = pd.DataFrame({"station_id": "s1", "target_at": [CUTOFF + ci.SLOT * k for k in range(1, 5)], "base_prediction": 100.0, "actual": 1e6})
    assert ci.level_signal(pd.concat([history, future], ignore_index=True), CUTOFF) == ci.level_signal(history, CUTOFF)


def test_missing_history_falls_back_to_factor_one() -> None:
    empty = pd.DataFrame(columns=["station_id", "target_at", "base_prediction", "actual"])
    signal = ci.level_signal(empty, CUTOFF)
    assert signal.factor == 1.0 and signal.fallback_reason == "no_resolved_history"


def test_predictions_end_to_end_detect_a_level_shift_only_where_it_happened() -> None:
    observations = _observations(drift={"02000": 0.5})
    artifact = ci.build_candidate_i_artifact()
    out = ci.build_candidate_i_predictions(_cycle(), "model-i", artifact, observations)
    assert len(out) == 12
    assert all(math.isfinite(item.prediction.prediction) and item.prediction.prediction >= 0 for item in out)
    factors = {item.prediction.station_id: item.signal.factor for item in out}
    assert factors["02000"] < 0.8
    assert abs(math.log(factors["01000"])) < 0.1 and abs(math.log(factors["03000"])) < 0.1
    for item in out:
        assert pd.Timestamp(item.signal.history_max_target_at) <= CUTOFF
        trace = ci.candidate_i_trace(item)
        assert trace["final_prediction"] == pytest.approx(trace["base_prediction"] * trace["calibration_factor"])


def test_history_is_out_of_sample_and_resolved() -> None:
    from candidate_b_shadow import _prepare_observations

    prepared = _prepare_observations(_observations())
    history = ci.build_oos_history(prepared, CUTOFF, ["01000", "02000", "03000"])
    assert (history["target_at"] <= CUTOFF).all()
    assert (history["target_at"] > history["history_cutoff"]).all()
    assert (history["model_bucket"] <= history["history_cutoff"]).all()
    assert history["target_at"].nunique() == ci.HISTORY_HOURS * 4


def test_adaptive_failure_serves_the_candidate_f_base(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("history unavailable")

    monkeypatch.setattr(ci, "build_oos_history", boom)
    out = ci.build_candidate_i_predictions(_cycle(), "model-i", ci.build_candidate_i_artifact(), _observations())
    assert all(item.signal.factor == 1.0 and item.prediction.prediction == pytest.approx(item.base_prediction) for item in out)
    assert all(item.signal.fallback_reason.startswith("history_failed") for item in out)


def test_rejects_observations_after_the_cutoff() -> None:
    observations = _observations()
    late = pd.DataFrame({"station_id": ["01000"], "observed_at": [CUTOFF + pd.Timedelta(minutes=15)], "demand": [1.0]})
    with pytest.raises(ShadowCandidateError):
        ci.build_candidate_i_predictions(_cycle(), "model-i", ci.build_candidate_i_artifact(), pd.concat([observations, late], ignore_index=True))


def test_artifact_rejects_changed_parameters() -> None:
    artifact = ci.build_candidate_i_artifact()
    ci.validate_candidate_i_artifact(artifact.metadata)
    tampered = {**artifact.metadata, "adaptive_layer": {**artifact.metadata["adaptive_layer"], "kappa": 1.0}}
    with pytest.raises(ShadowCandidateError):
        ci.validate_candidate_i_artifact(tampered)
