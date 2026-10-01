"""Candidate I research: resolved-error signals and adaptive correction policies.

Every signal for a row (cycle cutoff c, station s, target T = c + h) is computed ONLY
from resolved slots t <= c of that station's own base-prediction history (each 15-min
slot is predicted exactly once, by the hourly cycle whose window contains it, so the
history is an out-of-sample series). `max_source_at` is recorded per row so the leakage
audit can assert max_source_at <= data_cutoff for every signal.

Signals (all are ratios actual/base, pooled as ratio-of-sums unless noted):
- g2:        production G2 estimator (geomspace(0.2,1) weighted mean of the last 8
             per-slot ratios) -- reproduced exactly and checked against replay_g.
- ros{k}:    ratio of sums over the last k resolved slots (k = 4, 8, 16, 32).
- ss{d}w{w}: same time-of-day d days back, slots T-d*1day +/- w (d = 1, 2, 7).
- ssavg:     ratio of sums over d = 1, 2, 3 at T-d*1day +/- 1 slot.
- day:       ratio of sums over the last 96 resolved slots (24 h).
- ssday{d}:  24-h ratio of sums centered on T-d*1day (slots within +/-12h but <= c);
             used to split a same-slot ratio into "level that day" x "slot shape".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SLOT = pd.Timedelta(minutes=15)
DAY_SLOTS = 96


@dataclass
class StationSeries:
    t0: pd.Timestamp
    actual: np.ndarray
    base: np.ndarray
    valid: np.ndarray
    cum_a: np.ndarray
    cum_b: np.ndarray
    valid_positions: np.ndarray


def build_station_series(bases: pd.DataFrame, base_col: str) -> dict[str, StationSeries]:
    out: dict[str, StationSeries] = {}
    for sid, group in bases.groupby("station_id"):
        g = group.sort_values("target_at")
        t0 = g["target_at"].min()
        n = int((g["target_at"].max() - t0) / SLOT) + 1
        actual = np.full(n, np.nan)
        base = np.full(n, np.nan)
        pos = ((g["target_at"] - t0) / SLOT).astype(int).to_numpy()
        actual[pos] = g["actual"].to_numpy(dtype=float)
        base[pos] = g[base_col].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(base) & (base > 0)
        a0 = np.where(valid, actual, 0.0)
        b0 = np.where(valid, base, 0.0)
        out[sid] = StationSeries(
            t0=t0,
            actual=actual,
            base=base,
            valid=valid,
            cum_a=np.concatenate([[0.0], np.cumsum(a0)]),
            cum_b=np.concatenate([[0.0], np.cumsum(b0)]),
            valid_positions=np.flatnonzero(valid),
        )
    return out


def _ros(series: StationSeries, lo: int, hi: int) -> tuple[float, float]:
    """Ratio of sums over positions [lo, hi] inclusive, clipped to the array. Returns (ratio, base_sum)."""
    lo = max(lo, 0)
    hi = min(hi, len(series.actual) - 1)
    if hi < lo:
        return math.nan, 0.0
    a = series.cum_a[hi + 1] - series.cum_a[lo]
    b = series.cum_b[hi + 1] - series.cum_b[lo]
    if b <= 0:
        return math.nan, 0.0
    return a / b, b


def compute_signals(bases: pd.DataFrame, base_col: str) -> pd.DataFrame:
    series = build_station_series(bases, base_col)
    rows: list[dict[str, Any]] = []
    frame = bases[["data_cutoff", "station_id", "target_at", "horizon_minutes"]]
    for c, sid, T, h in frame.itertuples(index=False):
        s = series[sid]
        pc = int((c - s.t0) / SLOT)  # position of slot == cutoff (resolved, since observed_at <= c)
        pt = int((T - s.t0) / SLOT)
        rec: dict[str, Any] = {}
        max_pos = -1
        # production G2: last 8 valid per-slot ratios with target_at <= c
        k = np.searchsorted(s.valid_positions, pc, side="right")
        last8 = s.valid_positions[max(0, k - 8):k]
        if len(last8):
            ratios = s.actual[last8] / np.clip(s.base[last8], 1e-3, None)
            weights = np.geomspace(0.2, 1.0, num=len(ratios))
            rec["g2"] = float(np.average(ratios, weights=weights))
            rec["g2_n"] = int(len(last8))
            max_pos = max(max_pos, int(last8[-1]))
        else:
            rec["g2"], rec["g2_n"] = math.nan, 0
        for kk in (4, 8, 16, 32):
            rec[f"ros{kk}"], rec[f"ros{kk}_b"] = _ros(s, pc - kk + 1, pc)
        rec["day"], rec["day_b"] = _ros(s, pc - DAY_SLOTS + 1, pc)
        for d in (1, 2, 3, 7):
            center = pt - d * DAY_SLOTS
            for w in (0, 1, 2, 4):
                hi = min(center + w, pc)
                rec[f"ss{d}w{w}"], rec[f"ss{d}w{w}_b"] = _ros(s, center - w, hi)
            lo_day, hi_day = center - DAY_SLOTS // 2, min(center + DAY_SLOTS // 2 - 1, pc)
            rec[f"ssday{d}"], _ = _ros(s, lo_day, hi_day)
            if d <= 3:
                today8, _ = _ros(s, pc - 7, pc)
                back8, _ = _ros(s, pc - 7 - d * DAY_SLOTS, pc - d * DAY_SLOTS)
                rec[f"dod8_d{d}"] = today8 / back8 if (today8 == today8 and back8 == back8 and back8 > 0) else math.nan
        # same day-type (weekday vs weekend, Bogota local): most recent d in 1..3 matching T
        target_weekend = T.tz_convert("America/Bogota").dayofweek >= 5
        rec["sst_d"] = math.nan
        for d in (1, 2, 3):
            if ((T - pd.Timedelta(days=d)).tz_convert("America/Bogota").dayofweek >= 5) == target_weekend:
                rec["sst_d"] = d
                rec["sstw1"], rec["sstw2"] = rec[f"ss{d}w1"], rec[f"ss{d}w2"]
                rec["sstday"], rec["dod8_t"] = rec[f"ssday{d}"], rec[f"dod8_d{d}"]
                break
        else:
            rec["sstw1"] = rec["sstw2"] = rec["sstday"] = rec["dod8_t"] = math.nan
        # de-shaped level change: recent window today vs the same clock window 1 day earlier
        for kk in (8, 16):
            today, _ = _ros(s, pc - kk + 1, pc)
            yday, _ = _ros(s, pc - kk + 1 - DAY_SLOTS, pc - DAY_SLOTS)
            rec[f"dod{kk}"] = today / yday if (today == today and yday == yday and yday > 0) else math.nan
        # signal-to-noise inputs for a regime gate: per-slot log ratios over recent windows
        for kk in (8, 32):
            lo = max(pc - kk + 1, 0)
            idx = np.arange(lo, pc + 1)
            idx = idx[s.valid[idx]] if len(idx) else idx
            if len(idx) >= 3:
                lr = np.log(np.clip(s.actual[idx], 1e-3, None) / np.clip(s.base[idx], 1e-3, None))
                rec[f"lr{kk}_mean"], rec[f"lr{kk}_sd"], rec[f"lr{kk}_n"] = float(lr.mean()), float(lr.std(ddof=1)), int(len(lr))
            else:
                rec[f"lr{kk}_mean"], rec[f"lr{kk}_sd"], rec[f"lr{kk}_n"] = math.nan, math.nan, int(len(idx))
        num = den = 0.0
        for d in (1, 2, 3):
            center = pt - d * DAY_SLOTS
            lo, hi = max(center - 1, 0), min(center + 1, pc, len(s.actual) - 1)
            if hi >= lo:
                num += s.cum_a[hi + 1] - s.cum_a[lo]
                den += s.cum_b[hi + 1] - s.cum_b[lo]
        rec["ssavg"] = num / den if den > 0 else math.nan
        rec["ssavg_b"] = den
        rec["max_source_at"] = s.t0 + SLOT * pc
        rows.append(rec)
    out = pd.DataFrame(rows, index=bases.index)
    return out


# --------------------------------------------------------------------------------------
# Correction policies: signals -> multiplicative factor on the base prediction.
# --------------------------------------------------------------------------------------
def _log(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.where(np.isfinite(arr) & (arr > 0), np.log(np.clip(arr, 1e-3, None)), 0.0)


def clip_factor(log_factor: np.ndarray, low: float = 0.35, high: float = 2.5) -> np.ndarray:
    """Fail-safe: a non-finite correction (missing signal) falls back to factor 1.0."""
    log_factor = np.asarray(log_factor, dtype=float)
    return np.clip(np.exp(np.where(np.isfinite(log_factor), log_factor, 0.0)), low, high)


def soft_threshold(x: np.ndarray, theta: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0.0)


META_SIGNALS = ("g2", "ros4", "ros8", "ros16", "ss1w1", "ss1w2", "ss2w1", "ssavg", "day", "ssday1", "dod8")


def meta_features(fr: pd.DataFrame, *, threshold: float = 0.1) -> pd.DataFrame:
    X = pd.DataFrame(index=fr.index)
    for name in META_SIGNALS:
        X[f"log_{name}"] = _log(fr[name])
    X["shape1"] = X["log_ss1w1"] - X["log_ssday1"]
    for name in ("log_ros4", "log_ros16", "log_ss1w1", "shape1"):
        X[f"{name}_st"] = soft_threshold(X[name].to_numpy(), threshold)
    h = fr["horizon_minutes"].to_numpy(dtype=float) / 60.0
    X["h"] = h
    X["log_ros4_x_h"] = X["log_ros4"] * h
    X["shape1_x_h"] = X["shape1"] * h
    X["noise"] = np.nan_to_num(fr["lr32_sd"].to_numpy(dtype=float), nan=0.3)
    return X


def walk_forward_meta(
    fr: pd.DataFrame,
    *,
    base_col: str = "pred_F",
    kind: str = "huber",
    refit_hours: int = 6,
    window_days: float | None = None,
    start: pd.Timestamp | None = None,
    min_rows: int = 2000,
    log_clip: float = 1.2,
) -> np.ndarray:
    """Returns a multiplicative factor per row. Rows are only ever predicted by a model fit
    on rows whose target_at <= the refit time (<= the row's own data_cutoff)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor

    X_all = meta_features(fr)
    y_log = np.log(np.clip(fr["actual"].to_numpy(dtype=float), 1e-3, None) / np.clip(fr[base_col].to_numpy(dtype=float), 1e-3, None))
    y_log = np.clip(y_log, -2.0, 2.0)
    base_values = fr[base_col].to_numpy(dtype=float)
    stations = fr["station_id"].to_numpy()
    factors = np.ones(len(fr))
    cutoffs = fr["data_cutoff"]
    target_at = fr["target_at"]
    buckets = sorted({c.floor(f"{refit_hours}h") for c in cutoffs.unique()})
    for bucket in buckets:
        if start is not None and bucket < start.floor(f"{refit_hours}h"):
            continue
        train_mask = (target_at <= bucket).to_numpy() & np.isfinite(y_log)
        if window_days is not None:
            train_mask &= (cutoffs >= bucket - pd.Timedelta(days=window_days)).to_numpy()
        pred_mask = ((cutoffs >= bucket) & (cutoffs < bucket + pd.Timedelta(hours=refit_hours))).to_numpy()
        if train_mask.sum() < min_rows or not pred_mask.any():
            continue
        # WAPE-aligned weights; station volume from training rows only (no future info)
        train_base = pd.Series(base_values[train_mask])
        volume = train_base.groupby(stations[train_mask]).transform("mean").to_numpy()
        Xt, yt, wt = X_all[train_mask], y_log[train_mask], base_values[train_mask] / np.clip(volume, 1e-3, None)
        if kind == "huber":
            model = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=500)
            model.fit(Xt, yt, sample_weight=wt)
            out = model.predict(X_all[pred_mask])
        elif kind == "hgb":
            model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0, random_state=17)
            model.fit(Xt, yt, sample_weight=wt)
            out = model.predict(X_all[pred_mask])
        else:
            raise ValueError(kind)
        factors[pred_mask] = np.exp(np.clip(out, -log_clip, log_clip))
    return factors


def shrink(log_value: np.ndarray, base_mass: np.ndarray, tau: float) -> np.ndarray:
    """Empirical-Bayes style shrinkage toward 0 (factor 1): weight = m / (m + tau), where m
    is the base-prediction mass behind the estimate (more mass -> less noise)."""
    m = np.nan_to_num(np.asarray(base_mass, dtype=float), nan=0.0)
    w = m / (m + tau) if tau > 0 else np.ones_like(m)
    return w * log_value
