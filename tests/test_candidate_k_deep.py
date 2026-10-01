from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import candidate_k_deep as ck  # noqa: E402


CONFIG = ck.CandidateKConfig(lookback=16, hidden=16, blocks=1, epochs=1, batch_size=64, profile_days=2, baseline_weeks=1, scale_window=96)


def _observations(days: int = 10, stations: tuple[str, ...] = ("02300", "03000")) -> pd.DataFrame:
    index = pd.date_range("2026-09-01T05:00:00Z", periods=days * ck.DAY_STEPS, freq="15min")
    rng = np.random.default_rng(7)
    rows = []
    for k, station in enumerate(stations):
        profile = 200 + 150 * np.sin(2 * np.pi * (np.arange(len(index)) % ck.DAY_STEPS) / ck.DAY_STEPS + k)
        demand = np.maximum(0, profile + rng.normal(0, 10, len(index))).round()
        rows.append(pd.DataFrame({"station_id": station, "observed_at": index, "demand": demand}))
    return pd.concat(rows, ignore_index=True)


def test_features_at_a_cutoff_do_not_depend_on_later_observations() -> None:
    observations = _observations()
    cutoff = pd.Timestamp("2026-09-09T13:00:00Z")
    full = ck.build_panel(observations, CONFIG)
    truncated = ck.build_panel(observations, CONFIG, until=cutoff)
    a = ck.build_samples(full, CONFIG, np.array([full.position(cutoff)]))
    b = ck.build_samples(truncated, CONFIG, np.array([truncated.position(cutoff)]))
    assert len(a) == len(b) == 2
    np.testing.assert_allclose(a.past, b.past)
    np.testing.assert_allclose(a.seasonal, b.seasonal)
    np.testing.assert_allclose(a.calendar, b.calendar)
    np.testing.assert_allclose(a.scale, b.scale)
    assert np.isnan(b.target).all()  # the truncated panel has not observed the targets yet


def test_targets_are_the_next_four_slots_scaled_by_station_level() -> None:
    panel = ck.build_panel(_observations(), CONFIG)
    cutoff = pd.Timestamp("2026-09-08T20:00:00Z")
    origin = panel.position(cutoff)
    samples = ck.build_samples(panel, CONFIG, np.array([origin]))
    for row in range(len(samples)):
        station = samples.station[row]
        expected = panel.demand[origin + 1 : origin + 5, station] / panel.scale[origin, station]
        np.testing.assert_allclose(samples.target[row], expected, rtol=1e-6)
        assert samples.past[row, -1] == pytest.approx(panel.filled[origin, station] / panel.scale[origin, station], rel=1e-6)


def test_augmentation_only_rewrites_the_tail_and_targets() -> None:
    panel = ck.build_panel(_observations(), CONFIG)
    samples = ck.build_samples(panel, CONFIG)
    past = torch.from_numpy(samples.past[:256])
    seasonal = torch.from_numpy(samples.seasonal[:256])
    target = torch.from_numpy(samples.target[:256]).clone()
    target[0, 2] = float("nan")
    config = ck.CandidateKConfig(**{**CONFIG.to_dict(), "augmentation_prob": 1.0, "min_onset": 4})
    seasonal_before = seasonal.clone()
    new_past, new_target = ck.augment(past, seasonal, target, config, torch.Generator().manual_seed(0))
    assert torch.equal(seasonal, seasonal_before)  # seasonal references are never rewritten
    assert (new_past >= 0).all() and (torch.nan_to_num(new_target, nan=0.0) >= 0).all()
    assert torch.isnan(new_target[0, 2])  # unobserved targets stay unobserved
    changed = new_past != past
    # the synthetic regime is a suffix that always covers at least `min_onset` slots before the cutoff
    assert changed[:, -config.min_onset :].all()
    first_changed = changed.float().argmax(dim=1)
    for row in range(len(changed)):
        assert changed[row, first_changed[row] :].all()
    assert (first_changed > 0).any()  # some histories keep their oldest observations
    assert not torch.equal(new_target.nan_to_num(), target.nan_to_num())


def test_training_is_deterministic_and_predictions_are_non_negative() -> None:
    panel = ck.build_panel(_observations(), CONFIG)
    samples = ck.build_samples(panel, CONFIG)
    first = ck.train_model(samples, CONFIG, len(panel.stations), seed=3)
    second = ck.train_model(samples, CONFIG, len(panel.stations), seed=3)
    evaluation = samples.subset(samples.origin == samples.origin.max())
    a = ck.predict([first], evaluation)
    b = ck.predict([second], evaluation)
    np.testing.assert_allclose(a["prediction"], b["prediction"], rtol=1e-5)
    assert (a["prediction"] >= 0).all()
    assert ((a["gate"] >= 0) & (a["gate"] <= 1)).all()


def test_prediction_frame_follows_cycle_target_convention() -> None:
    panel = ck.build_panel(_observations(), CONFIG)
    cutoff = pd.Timestamp("2026-09-08T20:00:00Z")
    samples = ck.build_samples(panel, CONFIG, np.array([panel.position(cutoff)]))
    model = ck.train_model(ck.build_samples(panel, CONFIG), CONFIG, len(panel.stations), seed=0)
    frame = ck.prediction_frame(panel, samples, ck.predict([model], samples), CONFIG)
    assert len(frame) == 2 * CONFIG.horizons
    assert set(frame["horizon_minutes"]) == {15, 30, 45, 60}
    assert (frame["target_at"] - frame["data_cutoff"] == pd.to_timedelta(frame["horizon_minutes"], unit="min")).all()
    assert (frame["data_cutoff"] == cutoff).all()


def test_synthetic_series_are_non_negative_and_reproducible() -> None:
    a = ck.synthetic_series(64, 52, torch.Generator().manual_seed(1))
    b = ck.synthetic_series(64, 52, torch.Generator().manual_seed(1))
    assert a.shape == (64, 52)
    assert torch.equal(a, b)
    assert (a >= 0).all() and torch.isfinite(a).all()


def test_synthetic_regime_augmentation_rewrites_a_suffix_only() -> None:
    past = torch.full((128, 48), 1.0)
    seasonal = torch.ones(128, 3, 52)
    target = torch.ones(128, 4)
    new_past, new_target = ck.augment_synthetic(past, seasonal, target, 1.0, torch.Generator().manual_seed(2), min_onset=16)
    changed = new_past != past
    assert changed[:, -16:].float().mean() > 0.95  # the last min_onset slots belong to the new regime
    first = changed.float().argmax(dim=1)
    assert (first >= 0).all() and (first <= 48 - 16).all()
    assert not torch.equal(new_target, target)


def test_staged_model_freezes_experts_and_backtest_is_causal() -> None:
    config = ck.CandidateKConfig(architecture="staged", lookback=48, hidden=16, blocks=1, epochs=1, batch_size=64)
    local = ck.LocalForecaster(32, 4, config).eval()
    past = torch.rand(8, 48)
    seasonal = torch.rand(8, 3, 52)
    features = ck.local_backtest(local, past, seasonal)
    assert features.shape == (8, 12)
    # backtests only read the recent window: changing the seasonal *target* slots does not change them
    altered = seasonal.clone()
    altered[:, :, 48:] = 99.0
    torch.testing.assert_close(features, ck.local_backtest(local, past, altered))
    seasonal_expert = ck.CandidateKNet(ck.CandidateKConfig(**{**config.to_dict(), "architecture": "tide"}), n_stations=2)
    model = ck.StagedK(config, 2, seasonal_expert, local)
    assert all(not p.requires_grad for p in model.local_expert.parameters())
    assert all(not p.requires_grad for p in model.seasonal_expert.parameters())
    assert all(p.requires_grad for p in model.gate.parameters())


def test_realized_error_lookup_is_only_read_before_the_cutoff() -> None:
    config = ck.CandidateKConfig(architecture="staged_re", lookback=48, hidden=16, blocks=1)
    local = ck.LocalForecaster(32, 4, config).eval()
    seasonal_expert = ck.CandidateKNet(ck.CandidateKConfig(**{**config.to_dict(), "architecture": "tide"}), n_stations=2).eval()
    model = ck.StagedK(config, 2, seasonal_expert, local, realized_error=True)
    model.seasonal_lookup = torch.rand(200, 2, 4) * 100
    past, origin, station, scale = torch.rand(5, 48), torch.tensor([100, 120, 150, 180, 199]), torch.tensor([0, 1, 0, 1, 0]), torch.full((5,), 50.0)
    before = model.realized_error(past, origin, station, scale)
    assert before.shape == (5, 2 * len(ck.REALIZED_LAGS))
    full = model.seasonal_lookup.clone()
    for row, o in enumerate(origin.tolist()):
        model.seasonal_lookup = full.clone()
        model.seasonal_lookup[o - 3 :] = float("nan")  # forecasts issued after cutoff - 4 slots
        sl = slice(row, row + 1)
        torch.testing.assert_close(before[sl], model.realized_error(past[sl], origin[sl], station[sl], scale[sl]))
