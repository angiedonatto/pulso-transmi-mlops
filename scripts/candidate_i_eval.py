"""Candidate I evaluation: windows, per-cycle scoring, comparisons vs G, bootstrap.

Windows are fixed BEFORE looking at any challenger result:

- stable:     cycles 2026-09-03 00:00 .. 2026-09-10 17:00 UTC  (selection + no-drift check)
- drift_dev:  cycles 2026-09-10 18:00 .. 2026-09-13 11:00 UTC  (selection)
- drift_test: cycles 2026-09-13 12:00 .. last resolvable       (UNTOUCHED until final)
- official:   the official cycles (Supabase forecast_cycles)
- recent12 / recent6: last 12 / 6 official cycles (both inside drift_test)
- glab12:     official 2026-09-14 04:00..15:00 (window used by the Candidate G lab)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from candidate_i_core import cycle_scores, scored
from drift_monitor import _accuracy_by

STABLE = (pd.Timestamp("2026-09-03T00:00:00Z"), pd.Timestamp("2026-09-10T17:00:00Z"))
DRIFT_DEV = (pd.Timestamp("2026-09-10T18:00:00Z"), pd.Timestamp("2026-09-13T11:00:00Z"))
DRIFT_TEST_START = pd.Timestamp("2026-09-13T12:00:00Z")
GLAB12 = (pd.Timestamp("2026-09-14T04:00:00Z"), pd.Timestamp("2026-09-14T15:00:00Z"))


def window_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    c = frame["data_cutoff"]
    official = frame["is_official"].astype(bool)
    off_cutoffs = sorted(frame.loc[official, "data_cutoff"].unique())
    last12, last6 = set(off_cutoffs[-12:]), set(off_cutoffs[-6:])
    return {
        "stable": (c >= STABLE[0]) & (c <= STABLE[1]),
        "drift_dev": (c >= DRIFT_DEV[0]) & (c <= DRIFT_DEV[1]),
        "drift_test": c >= DRIFT_TEST_START,
        "drift_all": c >= DRIFT_DEV[0],
        "official": official,
        "recent12": c.isin(last12),
        "recent6": c.isin(last6),
        "glab12": official & (c >= GLAB12[0]) & (c <= GLAB12[1]),
        "all_scored": c >= STABLE[0],
    }


def window_scores(frame: pd.DataFrame, pred_cols: list[str], windows: list[str] | None = None) -> pd.DataFrame:
    masks = window_masks(frame)
    windows = windows or list(masks)
    rows = []
    for col in pred_cols:
        cyc = cycle_scores(frame, col).set_index("data_cutoff")["cycle_accuracy"]
        for name in windows:
            cut = frame.loc[masks[name], "data_cutoff"].unique()
            vals = cyc.reindex(cut).dropna()
            rows.append({"model": col, "window": name, "mean_cycle_accuracy": float(vals.mean()) if len(vals) else np.nan, "cycles": int(len(vals))})
    return pd.DataFrame(rows).pivot(index="model", columns="window", values="mean_cycle_accuracy")


def cycle_delta_table(frame: pd.DataFrame, col: str, ref: str) -> pd.DataFrame:
    a = cycle_scores(frame, col).set_index("data_cutoff")["cycle_accuracy"]
    b = cycle_scores(frame, ref).set_index("data_cutoff")["cycle_accuracy"]
    out = pd.DataFrame({"challenger": a, "reference": b}).dropna()
    out["delta_pp"] = out["challenger"] - out["reference"]
    return out


def delta_summary(frame: pd.DataFrame, col: str, ref: str, window: str) -> dict[str, Any]:
    masks = window_masks(frame)
    cut = set(frame.loc[masks[window], "data_cutoff"].unique())
    d = cycle_delta_table(frame, col, ref)
    d = d[d.index.isin(cut)]["delta_pp"]
    if d.empty:
        return {"cycles": 0}
    return {
        "cycles": int(len(d)),
        "mean_delta": float(d.mean()),
        "median_delta": float(d.median()),
        "worst_delta": float(d.min()),
        "best_delta": float(d.max()),
        "wins": int((d > 1e-9).sum()),
        "losses": int((d < -1e-9).sum()),
        "ties": int(((d >= -1e-9) & (d <= 1e-9)).sum()),
    }


def block_bootstrap_p_better(frame: pd.DataFrame, col: str, ref: str, window: str, *, block_hours: int = 6, n_boot: int = 5000, seed: int = 17) -> dict[str, Any]:
    """Moving-block bootstrap over consecutive cycle deltas (cycles are autocorrelated).
    Returns P(mean delta > 0) and a 90% interval. With few cycles this is a rough,
    descriptive uncertainty estimate, not a formal test."""
    masks = window_masks(frame)
    cut = set(frame.loc[masks[window], "data_cutoff"].unique())
    d = cycle_delta_table(frame, col, ref)
    d = d[d.index.isin(cut)].sort_index()["delta_pp"].to_numpy()
    n = len(d)
    if n < block_hours * 2:
        return {"cycles": n, "p_better": None, "note": "too few cycles for block bootstrap"}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_hours))
    starts = np.arange(0, n - block_hours + 1)
    means = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([d[s:s + block_hours] for s in picks])[:n]
        means[i] = sample.mean()
    return {
        "cycles": n,
        "block_hours": block_hours,
        "p_better": float((means > 0).mean()),
        "ci90_low": float(np.quantile(means, 0.05)),
        "ci90_high": float(np.quantile(means, 0.95)),
        "observed_mean_delta": float(d.mean()),
    }


def station_table(frame: pd.DataFrame, cols: list[str], window_mask: pd.Series) -> pd.DataFrame:
    """Per-station accuracy averaged over cycles (mean of per-cycle station accuracy)."""
    out = []
    sub = frame[window_mask]
    for col in cols:
        s = scored(sub, col)
        per = _accuracy_by(s, ["data_cutoff", "station_id"])
        agg = per.groupby("station_id", as_index=False)["accuracy"].mean().rename(columns={"accuracy": col})
        out.append(agg.set_index("station_id"))
    return pd.concat(out, axis=1)


def horizon_table(frame: pd.DataFrame, cols: list[str], window_mask: pd.Series) -> pd.DataFrame:
    """Per-horizon mean station accuracy (station WAPE pooled across the window's cycles)."""
    out = []
    sub = frame[window_mask]
    for col in cols:
        s = scored(sub, col)
        per = _accuracy_by(s, ["horizon_minutes", "station_id"])
        agg = per.groupby("horizon_minutes", as_index=False)["accuracy"].mean().rename(columns={"accuracy": col})
        out.append(agg.set_index("horizon_minutes"))
    return pd.concat(out, axis=1)
