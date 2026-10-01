"""Frozen Candidate K v2.1 runtime (needs PyTorch; skipped where torch is not installed)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import candidate_k_deep as ck  # noqa: E402
import candidate_k_runtime as ckr  # noqa: E402
import candidate_k_shadow as cks  # noqa: E402

STATIONS = ["02300", "03000", "05000"]
TINY = {"hidden": 16, "blocks": 1, "epochs": 1, "batch_size": 256}
SEEDS = (0, 1)
CUTOFF = pd.Timestamp("2026-09-11T14:00:00Z")


def observations(until: pd.Timestamp = pd.Timestamp("2026-09-12T05:00:00Z")) -> pd.DataFrame:
    # starts on the frozen 30-min training grid (an even number of slots after the anchor)
    times = pd.date_range("2026-09-01T05:00:00Z", until, freq="15min")
    rng = np.random.default_rng(3)
    slot = np.arange(len(times)) % ck.DAY_STEPS
    rows = []
    for k, station in enumerate(STATIONS):
        level = 200 + 150 * np.sin(2 * np.pi * slot / ck.DAY_STEPS + k)
        rows.append(pd.DataFrame({"station_id": station, "observed_at": times, "demand": np.maximum(0, level + rng.normal(0, 15, len(times))).round()}))
    return pd.concat(rows, ignore_index=True)


def cycle_targets(cutoff: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        [{"station_id": s, "target_at": cutoff + pd.Timedelta(minutes=h), "horizon_minutes": h} for s in STATIONS for h in (15, 30, 45, 60)]
    )


def predictor(cache_dir: Path | None = None) -> ckr.FrozenKPredictor:
    return ckr.FrozenKPredictor(cks.load_artifact(), cks.artifact_dir(), cache_dir, config_overrides=TINY, seeds=SEEDS, allow_unfrozen=True)


def test_runtime_reproduces_the_walk_forward_procedure_of_the_experiment() -> None:
    obs = observations()
    frame, info = predictor().predict(obs[pd.to_datetime(obs["observed_at"], utc=True) <= CUTOFF], CUTOFF, cycle_targets(CUTOFF))

    # the experiment's procedure, written out with the research module directly
    config = replace(ck.CandidateKConfig(architecture="staged_re"), **TINY)
    local = ck.LocalForecaster(cks.LOCAL_LOOKBACK, config.horizons, ck.CandidateKConfig())
    local.load_state_dict(torch.load(cks.artifact_dir() / "local_pretrained.pt"))
    local.eval()
    refit = cks.refit_time(CUTOFF)
    torch.set_num_threads(1)
    train_panel = ck.build_panel(obs[pd.to_datetime(obs["observed_at"], utc=True) <= refit], config)
    train_samples = ck.build_samples(train_panel, config)
    refit_pos = train_panel.position(refit)
    mask = (train_samples.origin + config.horizons <= refit_pos) & (train_samples.origin % cks.TRAIN_STRIDE == 0)
    panel = ck.build_panel(obs[pd.to_datetime(obs["observed_at"], utc=True) <= CUTOFF], config)
    samples = ck.build_samples(panel, config)
    pos = panel.position(CUTOFF)
    members = []
    for seed in SEEDS:
        model = ck.train_staged(train_samples.subset(mask), config, len(STATIONS), seed, local, realized_error=True,
                                lookup_samples=train_samples.subset(train_samples.origin <= refit_pos))
        model.build_lookup(samples.subset(samples.origin <= pos), len(STATIONS))
        members.append(ck.predict([model], samples.subset(samples.origin == pos))["prediction"])
    expected = np.median(np.stack(members), axis=0)

    got = frame.sort_values(["station_id", "horizon_minutes"])["k_prediction"].to_numpy().reshape(len(STATIONS), 4)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert info["refit_at"] == cks._ts(refit) and info["trained"] is True


def test_cached_refit_gives_identical_predictions(tmp_path: Path) -> None:
    obs = observations()
    obs = obs[pd.to_datetime(obs["observed_at"], utc=True) <= CUTOFF]
    first, info_first = predictor(tmp_path).predict(obs, CUTOFF, cycle_targets(CUTOFF))
    second, info_second = predictor(tmp_path).predict(obs, CUTOFF, cycle_targets(CUTOFF))
    assert info_first["trained"] is True and info_second["trained"] is False
    pd.testing.assert_frame_equal(first, second)
    assert info_first["train_data_sha256"] == info_second["train_data_sha256"]


def test_runtime_ignores_observations_after_the_cutoff(tmp_path: Path) -> None:
    obs = observations()
    model = predictor(tmp_path)
    with_future, _ = model.predict(obs, CUTOFF, cycle_targets(CUTOFF))
    without, _ = model.predict(obs[pd.to_datetime(obs["observed_at"], utc=True) <= CUTOFF], CUTOFF, cycle_targets(CUTOFF))
    pd.testing.assert_frame_equal(with_future, without)


def test_runtime_refuses_unfrozen_settings_unless_explicitly_allowed() -> None:
    with pytest.raises(cks.CandidateKShadowError):
        ckr.FrozenKPredictor(cks.load_artifact(), cks.artifact_dir(), None, config_overrides=TINY)
    metadata = cks.load_artifact()
    with pytest.raises(cks.CandidateKShadowError):
        ckr.FrozenKPredictor({**metadata, "config": {**metadata["config"], "hidden": 64}}, cks.artifact_dir(), None)
