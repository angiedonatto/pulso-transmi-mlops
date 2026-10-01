from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import candidate_i_adaptive as ad
from scripts import candidate_i_core as core


def synthetic_bases(days: int = 6, stations: tuple[str, ...] = ("03000", "05100")) -> pd.DataFrame:
    """Bases-like frame: one row per hourly cycle x station x horizon, every 15-min slot
    predicted exactly once (as in the real hourly replay)."""
    rng = np.random.default_rng(3)
    cutoffs = pd.date_range("2026-01-01T00:00:00Z", periods=days * 24, freq="h")
    rows = []
    for c in cutoffs:
        for sid in stations:
            for h in (15, 30, 45, 60):
                t = c + pd.Timedelta(minutes=h)
                level = 100 + 50 * np.sin(2 * np.pi * (t.hour * 60 + t.minute) / 1440)
                rows.append({"data_cutoff": c, "station_id": sid, "target_at": t, "horizon_minutes": h, "actual": level * rng.uniform(0.8, 1.2), "pred_X": level})
    return pd.DataFrame(rows)


def test_signals_never_read_slots_after_the_cutoff() -> None:
    bases = synthetic_bases()
    signals = ad.compute_signals(bases, "pred_X")
    assert (signals["max_source_at"] <= bases["data_cutoff"]).all()

    cut = pd.Timestamp("2026-01-04T10:00:00Z")
    mutated = bases.copy()
    future = mutated["target_at"] > cut
    mutated.loc[future, "actual"] = mutated.loc[future, "actual"] * 9.0 + 77.0
    signals2 = ad.compute_signals(mutated, "pred_X")

    past_rows = bases["data_cutoff"] <= cut
    cols = [c for c in signals.columns if c != "max_source_at"]
    a = signals.loc[past_rows, cols].to_numpy(dtype=float)
    b = signals2.loc[past_rows, cols].to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True), "a signal for a cycle <= cut changed after mutating actuals of slots > cut"


def test_signals_are_live_not_constant() -> None:
    """Sanity: mutating a resolved slot at or before the cutoff must change some signal."""
    bases = synthetic_bases()
    cut = pd.Timestamp("2026-01-04T10:00:00Z")
    mutated = bases.copy()
    at_cut = mutated["target_at"] == cut
    mutated.loc[at_cut, "actual"] = mutated.loc[at_cut, "actual"] * 3.0
    s1 = ad.compute_signals(bases, "pred_X")
    s2 = ad.compute_signals(mutated, "pred_X")
    rows = bases["data_cutoff"] == cut
    assert not np.allclose(s1.loc[rows, "g2"], s2.loc[rows, "g2"])


def test_meta_learner_trains_only_on_resolved_targets() -> None:
    bases = synthetic_bases(days=8)
    signals = ad.compute_signals(bases, "pred_X")
    fr = pd.concat([bases, signals], axis=1)
    fr["lr32_sd"] = fr["lr32_sd"].fillna(0.2)

    cut = pd.Timestamp("2026-01-06T00:00:00Z")
    f1 = ad.walk_forward_meta(fr, base_col="pred_X", refit_hours=24, min_rows=50)
    mutated = fr.copy()
    future = mutated["target_at"] > cut
    mutated.loc[future, "actual"] = mutated.loc[future, "actual"] * 5.0
    f2 = ad.walk_forward_meta(mutated, base_col="pred_X", refit_hours=24, min_rows=50)
    rows = (fr["data_cutoff"] >= cut) & (fr["data_cutoff"] < cut + pd.Timedelta(hours=24))
    assert np.allclose(f1[rows.to_numpy()], f2[rows.to_numpy()]), "meta-learner prediction for bucket starting at cut depended on actuals after cut"


def test_ratio_features_are_causal() -> None:
    rows = []
    for sid, offset in (("03000", 100.0), ("05100", 300.0)):
        for i, t in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=20 * 96, freq="15min")):
            rows.append({"station_id": sid, "observed_at": t, "demand": offset + (i % 96) + 5 * np.sin(i / 7)})
    obs = pd.DataFrame(rows)
    cut = pd.Timestamp("2026-01-15T12:00:00Z")
    t1 = core.ratio_feature_table(core._prepare_observations(obs))
    mutated = obs.copy()
    mutated.loc[mutated["observed_at"] > cut, "demand"] *= 4.0
    t2 = core.ratio_feature_table(core._prepare_observations(mutated))
    keep1 = t1[t1["observed_at"] <= cut].set_index(["station_id", "observed_at"])
    keep2 = t2[t2["observed_at"] <= cut].set_index(["station_id", "observed_at"])
    assert np.allclose(keep1.to_numpy(dtype=float), keep2.to_numpy(dtype=float), equal_nan=True)


def _tiny_tables(days: int = 17) -> dict[str, pd.DataFrame]:
    rows = []
    for sid, offset in (("03000", 100.0), ("05100", 300.0)):
        for i, t in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=days * 96, freq="15min")):
            rows.append({"station_id": sid, "observed_at": t, "demand": offset + 40 * np.sin(2 * np.pi * (i % 96) / 96) + (i % 7)})
    obs = pd.DataFrame(rows)
    return {"observations": obs, "forecast_cycles": pd.DataFrame({"cycle_id": [], "data_cutoff": pd.to_datetime([], utc=True)})}


def test_variant_replay_serves_each_cycle_from_a_model_trained_before_its_bucket() -> None:
    tables = _tiny_tables()
    cutoffs = list(pd.date_range("2026-01-16T00:00:00Z", periods=8, freq="h"))
    variant = core.BaseVariant(name="T", wape_weights=True, ratio_features=True, hgb_overrides=(("max_iter", 20),))
    p1 = core.replay_base_variant(tables, cutoffs, variant, cadence_hours=6, log=lambda _: None)

    # Mutate every observation after the first cycle's cutoff: predictions for that first
    # cycle (bucket 00:00, cutoff 00:00) may only use data <= 00:00, so must not change.
    first = cutoffs[0]
    mutated = {**tables, "observations": tables["observations"].copy()}
    obs = mutated["observations"]
    obs.loc[obs["observed_at"] > first, "demand"] = obs.loc[obs["observed_at"] > first, "demand"] * 6.0
    p2 = core.replay_base_variant(mutated, cutoffs, variant, cadence_hours=6, log=lambda _: None)
    a = p1[p1["data_cutoff"] == first]["pred_T"].to_numpy()
    b = p2[p2["data_cutoff"] == first]["pred_T"].to_numpy()
    assert np.allclose(a, b)
