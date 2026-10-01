"""Candidate I -- synthetic drift stress test.

Unlike a pure actual-perturbation test, drift is injected into the OBSERVATIONS, so every
model sees it consistently: in its lag features, its training data (after refits), and
its resolved-error history. Bases are then re-replayed on the drifted snapshot.

Window: pre-drift period 2026-09-04 00:00 -> 2026-09-07 00:00 UTC (all stations stable in
the real data), drift starts at DRIFT_T0 for the affected stations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import candidate_i_core as core

STRESS_START = pd.Timestamp("2026-09-04T00:00:00Z")
STRESS_END = pd.Timestamp("2026-09-07T00:00:00Z")
DRIFT_T0 = pd.Timestamp("2026-09-04T12:00:00Z")
AFFECTED = ("02300", "05000", "07107", "10009")


@dataclass(frozen=True)
class Scenario:
    name: str
    apply: Callable[[pd.DataFrame], pd.DataFrame]


def _mask(obs: pd.DataFrame, stations: tuple[str, ...] = AFFECTED) -> pd.Series:
    return obs["station_id"].isin(stations) & (obs["observed_at"] >= DRIFT_T0)


def _scale(factor: float, stations: tuple[str, ...] = AFFECTED) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def fn(obs: pd.DataFrame) -> pd.DataFrame:
        out = obs.copy()
        m = _mask(out, stations)
        out.loc[m, "demand"] = out.loc[m, "demand"] * factor
        return out
    return fn


def _ramp(final: float, hours: float = 48.0) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def fn(obs: pd.DataFrame) -> pd.DataFrame:
        out = obs.copy()
        m = _mask(out)
        elapsed = (out.loc[m, "observed_at"] - DRIFT_T0).dt.total_seconds() / 3600.0
        out.loc[m, "demand"] = out.loc[m, "demand"] * (1.0 + (final - 1.0) * np.clip(elapsed / hours, 0.0, 1.0))
        return out
    return fn


def _peak_shift(minutes: int = 60) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Daily pattern moves `minutes` later: demand(t) := original demand(t - minutes)."""
    def fn(obs: pd.DataFrame) -> pd.DataFrame:
        out = obs.copy()
        lookup = obs.set_index(["station_id", "observed_at"])["demand"]
        m = _mask(out)
        src = pd.MultiIndex.from_arrays([out.loc[m, "station_id"], out.loc[m, "observed_at"] - pd.Timedelta(minutes=minutes)])
        out.loc[m, "demand"] = lookup.reindex(src).to_numpy()
        return out.dropna(subset=["demand"])
    return fn


def _peak_amplify(high: float = 1.7, low: float = 0.8) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def fn(obs: pd.DataFrame) -> pd.DataFrame:
        out = obs.copy()
        local = out["observed_at"].dt.tz_convert("America/Bogota")
        out["_slot"] = local.dt.hour * 4 + local.dt.minute // 15
        ref = out[out["observed_at"] < DRIFT_T0].groupby(["station_id", "_slot"])["demand"].mean()
        q = ref.groupby(level=0).transform(lambda v: v.rank(pct=True))
        pct = q.reindex(pd.MultiIndex.from_arrays([out["station_id"], out["_slot"]])).to_numpy()
        factor = np.where(pct >= 0.75, high, np.where(pct <= 0.25, low, 1.0))
        m = _mask(out).to_numpy()
        out.loc[m, "demand"] = out.loc[m, "demand"].to_numpy() * factor[m]
        return out.drop(columns="_slot")
    return fn


def _mixed(obs: pd.DataFrame) -> pd.DataFrame:
    out = _scale(0.5, ("02300", "05000"))(obs)
    return _scale(1.5, ("07107", "10009"))(out)


SCENARIOS = [
    Scenario("no_drift", lambda obs: obs.copy()),
    Scenario("step_down_0.55", _scale(0.55)),
    Scenario("step_up_1.45", _scale(1.45)),
    Scenario("ramp_up_1.6_48h", _ramp(1.6)),
    Scenario("peak_shift_+60min", _peak_shift(60)),
    Scenario("peak_amplify_1.7", _peak_amplify()),
    Scenario("mixed_0.5_and_1.5", _mixed),
]


def stress_tables(tables: dict[str, pd.DataFrame], scenario: Scenario) -> dict[str, pd.DataFrame]:
    obs = scenario.apply(tables["observations"])
    obs = obs[obs["observed_at"] <= STRESS_END + pd.Timedelta(minutes=core.MAX_HORIZON)]
    empty_cycles = pd.DataFrame({"cycle_id": [], "data_cutoff": pd.to_datetime([], utc=True)})
    return {**tables, "observations": obs.sort_values(["station_id", "observed_at"]).reset_index(drop=True), "forecast_cycles": empty_cycles}


HISTORY_START = pd.Timestamp("2026-09-01T00:00:00Z")  # signals need ~3 days of resolved history


def replay_scenario(tables: dict[str, pd.DataFrame], scenario: Scenario, *, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Re-replays the bases on the drifted snapshot.

    - corrected base: F refit every 6h (G's and the challenger's base; the same for both)
    - signal history: F refit once per day (the production-feasible out-of-sample history)
    G2 is computed on the corrected base's own resolved history, exactly as in `replay_g`."""
    import candidate_i_adaptive as ad

    st = stress_tables(tables, scenario)
    cutoffs = [c for c in core.evaluation_cutoffs(st["observations"], HISTORY_START) if c <= STRESS_END]
    base = core.replay_base_variant(st, cutoffs, core.BaseVariant(name="F6h"), cadence_hours=6, log=log)
    hist = core.replay_base_variant(st, cutoffs, core.BaseVariant(name="F24h"), cadence_hours=24, log=log)
    keys = ["data_cutoff", "station_id", "target_at"]
    frame = base.merge(hist[keys + ["pred_F24h"]], on=keys, how="left")
    frame["cycle_id"] = frame["data_cutoff"].map(lambda c: f"stress-{c.isoformat()}")
    frame = core.replay_g(frame, base_col="pred_F6h")
    signals = ad.compute_signals(hist, "pred_F24h")
    hist = pd.concat([hist[keys], signals], axis=1)
    frame = frame.merge(hist, on=keys, how="left")
    frame["scenario"] = scenario.name
    return frame


def score_scenario(frame: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    """Mean cycle accuracy after the drift onset, for affected / unaffected stations."""
    window = (frame["data_cutoff"] >= DRIFT_T0) & (frame["data_cutoff"] < STRESS_END)
    early = window & (frame["data_cutoff"] < DRIFT_T0 + pd.Timedelta(hours=12))
    rows = []
    for col in pred_cols:
        rec = {"model": col}
        for label, mask in (
            ("post_all", window),
            ("post_affected", window & frame["station_id"].isin(AFFECTED)),
            ("post_unaffected", window & ~frame["station_id"].isin(AFFECTED)),
            ("first12h_affected", early & frame["station_id"].isin(AFFECTED)),
        ):
            rec[label] = core.mean_cycle_accuracy(frame[mask], col)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("model")
