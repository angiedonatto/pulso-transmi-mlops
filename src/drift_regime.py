"""Drift / regime-change diagnostics for Pulso TransMi.

Pure analysis helpers (no I/O, no plotting) used by `scripts/drift_regime_analysis.py`.

Time semantics
--------------
All timestamps handled here are *historical dataset time* (`observed_at`, `data_cutoff`,
`target_at`), normalized to tz-aware UTC and converted explicitly to America/Bogota for
local-clock features. Execution timestamps (`created_at`, `accepted_at`, `ingested_at`,
`first_seen_at`) are never used to order or window the analysis.

Sign convention
---------------
This module reports `residual = actual - prediction` (positive = under-prediction).
`src/drift_monitor.py` uses the opposite sign (`prediction - actual`); accuracy and WAPE
are sign-agnostic and are reused from it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from drift_monitor import _accuracy_by, accuracy_summary


BOGOTA_TZ = "America/Bogota"
SLOT = pd.Timedelta(minutes=15)
DAY_STEPS = 96
WEEK_STEPS = 7 * DAY_STEPS
# lag name -> number of 15-minute steps (repo naming: lag_96 = previous day, lag_weekly = 7 days)
LAGS: dict[str, int] = {"lag_15": 1, "lag_30": 2, "lag_60": 4, "lag_96": DAY_STEPS, "lag_weekly": WEEK_STEPS}
# additive stabilizer for log-ratios: per 15-min slot and per hourly (4-slot) sum
SLOT_OFFSET = 5.0
HOURLY_OFFSET = 4 * SLOT_OFFSET
# local-clock bands used for peak diagnostics (slot = hour * 4 + minute // 15)
PEAK_BANDS: dict[str, list[int]] = {
    "am": list(range(5 * 4, 11 * 4)),
    "pm": list(range(16 * 4, 22 * 4)),
    "night": list(range(22 * 4, 96)) + list(range(0, 5 * 4)),
}


# --------------------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------------------


def prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Normalize observations: zero-padded station ids, UTC timestamps, Bogota clock fields."""
    missing = {"station_id", "observed_at", "demand"} - set(observations.columns)
    if missing:
        raise ValueError(f"observations missing columns: {sorted(missing)}")
    frame = observations[["station_id", "observed_at", "demand"]].copy()
    frame["station_id"] = frame["station_id"].astype(str).str.zfill(5)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce").astype(float)
    frame = frame.dropna(subset=["observed_at", "demand"])
    frame = frame.drop_duplicates(["station_id", "observed_at"], keep="last")
    local = frame["observed_at"].dt.tz_convert(BOGOTA_TZ)
    frame["local_time"] = local
    frame["local_date"] = local.dt.strftime("%Y-%m-%d")
    frame["local_hour"] = local.dt.hour
    frame["dow"] = local.dt.dayofweek
    frame["slot"] = local.dt.hour * 4 + local.dt.minute // 15
    return frame.sort_values(["station_id", "observed_at"]).reset_index(drop=True)


def demand_matrix(prepared: pd.DataFrame) -> pd.DataFrame:
    """Wide matrix (regular 15-min UTC grid x station). Missing slots stay NaN."""
    wide = prepared.pivot(index="observed_at", columns="station_id", values="demand").sort_index()
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="15min")
    return wide.reindex(grid)


def seasonal_baseline(wide: pd.DataFrame, *, weeks: int = 4, min_weeks: int | None = None) -> pd.DataFrame:
    """Causal same-weekday/same-slot median of the previous `weeks` weeks.

    The value at t only uses y(t - 7d), ..., y(t - weeks * 7d), so it never looks ahead.
    """
    min_weeks = weeks if min_weeks is None else min_weeks
    stack = np.stack([wide.shift(WEEK_STEPS * w).to_numpy(dtype=float) for w in range(1, weeks + 1)])
    counts = np.sum(~np.isnan(stack), axis=0)
    with np.errstate(all="ignore"):
        filled = np.where(np.isnan(stack), np.inf, stack)
        ordered = np.sort(filled, axis=0)
    median = np.full(counts.shape, np.nan)
    for count in range(1, weeks + 1):
        mask = counts == count
        if not mask.any():
            continue
        lo = ordered[(count - 1) // 2]
        hi = ordered[count // 2]
        median[mask] = ((lo + hi) / 2.0)[mask]
    median[counts < min_weeks] = np.nan
    return pd.DataFrame(median, index=wide.index, columns=wide.columns)


def hourly_log_ratio(wide: pd.DataFrame, baseline: pd.DataFrame, *, offset: float = HOURLY_OFFSET) -> pd.DataFrame:
    """Hourly log((sum actual + c) / (sum baseline + c)) per station; NaN unless 4/4 slots exist."""
    valid = wide.notna() & baseline.notna()
    actual = wide.where(valid).resample("1h").sum(min_count=4)
    base = baseline.where(valid).resample("1h").sum(min_count=4)
    return np.log((actual + offset) / (base + offset))


def global_drift_indices(wide: pd.DataFrame, baseline: pd.DataFrame, *, offset: float = HOURLY_OFFSET) -> pd.DataFrame:
    """Two hourly system-wide indices.

    - `global_level`: log ratio of total demand vs total baseline (cancels opposite moves).
    - `global_intensity`: mean over stations of |station log ratio| (does not cancel).
    """
    station_lr = hourly_log_ratio(wide, baseline, offset=offset)
    valid = wide.notna() & baseline.notna()
    total_actual = wide.where(valid).sum(axis=1, min_count=1).resample("1h").sum(min_count=4)
    total_base = baseline.where(valid).sum(axis=1, min_count=1).resample("1h").sum(min_count=4)
    stations = max(1, wide.shape[1])
    level = np.log((total_actual + offset * stations) / (total_base + offset * stations))
    intensity = station_lr.abs().mean(axis=1, skipna=False)
    return pd.DataFrame({"global_level": level, "global_intensity": intensity})


# --------------------------------------------------------------------------------------
# Change-point detection (binary segmentation, L2 cost on the mean)
# --------------------------------------------------------------------------------------


def robust_sigma(values: np.ndarray) -> float:
    """Noise scale from first differences (robust to level shifts): 1.4826 * MAD(diff) / sqrt(2)."""
    diffs = np.diff(np.asarray(values, dtype=float))
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 2:
        return float("nan")
    mad = np.median(np.abs(diffs - np.median(diffs)))
    return float(1.4826 * mad / math.sqrt(2.0))


def binary_segmentation(
    values: Iterable[float],
    *,
    min_size: int,
    penalty: float,
    max_change_points: int = 8,
) -> list[int]:
    """Greedy best-first binary segmentation for changes in mean.

    Returns sorted indices where a new segment starts. A split is accepted only when the
    reduction in within-segment sum of squares exceeds `penalty`.
    """
    x = np.asarray(list(values), dtype=float)
    if np.isnan(x).any():
        raise ValueError("binary_segmentation requires a series without NaN")
    n = x.size
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def cost(a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray:
        length = np.asarray(b) - np.asarray(a)
        total = cs[b] - cs[a]
        return (cs2[b] - cs2[a]) - total * total / length

    def best_split(a: int, b: int) -> tuple[float, int | None]:
        if b - a < 2 * min_size:
            return -math.inf, None
        taus = np.arange(a + min_size, b - min_size + 1)
        gains = cost(a, b) - cost(np.full_like(taus, a), taus) - cost(taus, np.full_like(taus, b))
        idx = int(np.argmax(gains))
        return float(gains[idx]), int(taus[idx])

    boundaries = [0, n]
    change_points: list[int] = []
    while len(change_points) < max_change_points:
        candidates = [best_split(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        gain, tau = max(candidates, key=lambda item: item[0])
        if tau is None or gain <= penalty:
            break
        change_points.append(tau)
        boundaries = sorted(boundaries + [tau])
    return sorted(change_points)


def prune_change_points(values: Iterable[float], change_points: list[int], *, min_effect: float) -> list[int]:
    """Drop change points whose adjacent segment means differ by less than `min_effect`."""
    x = np.asarray(list(values), dtype=float)
    points = sorted(change_points)
    while points:
        bounds = [0] + points + [x.size]
        means = [float(x[bounds[i] : bounds[i + 1]].mean()) for i in range(len(bounds) - 1)]
        jumps = [abs(means[i + 1] - means[i]) for i in range(len(points))]
        weakest = int(np.argmin(jumps))
        if jumps[weakest] >= min_effect:
            break
        points.pop(weakest)
    return points


@dataclass(frozen=True)
class ChangePointConfig:
    min_segment_hours: int = 6
    penalty_beta: float = 4.0
    min_effect: float = math.log(1.15)
    max_change_points: int = 8


def segment_series(series: pd.Series, config: ChangePointConfig | None = None) -> pd.DataFrame:
    """Segment an hourly series; returns one row per segment with its mean (log units)."""
    config = config or ChangePointConfig()
    clean = series.dropna()
    if clean.size < 2 * config.min_segment_hours:
        return pd.DataFrame(columns=["segment", "start", "end", "hours", "mean", "ratio"])
    x = clean.to_numpy(dtype=float)
    sigma = robust_sigma(x)
    sigma = sigma if math.isfinite(sigma) and sigma > 0 else float(np.std(x) or 1.0)
    penalty = config.penalty_beta * sigma * sigma * math.log(x.size)
    points = binary_segmentation(x, min_size=config.min_segment_hours, penalty=penalty, max_change_points=config.max_change_points)
    points = prune_change_points(x, points, min_effect=config.min_effect)
    bounds = [0] + points + [x.size]
    rows = []
    for i in range(len(bounds) - 1):
        part = clean.iloc[bounds[i] : bounds[i + 1]]
        mean = float(part.mean())
        rows.append(
            {
                "segment": i,
                "start": part.index[0],
                "end": part.index[-1] + pd.Timedelta(hours=1),
                "hours": int(part.size),
                "mean": mean,
                "ratio": float(math.exp(mean)),
            }
        )
    return pd.DataFrame(rows)


def detect_regimes(
    station_log_ratio: pd.DataFrame,
    indices: pd.DataFrame,
    config: ChangePointConfig | None = None,
) -> pd.DataFrame:
    """Segment every station series plus the two global indices."""
    config = config or ChangePointConfig()
    frames = []
    for name in ["global_intensity", "global_level"]:
        seg = segment_series(indices[name], config)
        frames.append(seg.assign(series=name))
    for station in station_log_ratio.columns:
        seg = segment_series(station_log_ratio[station], config)
        frames.append(seg.assign(series=str(station)))
    result = pd.concat(frames, ignore_index=True)
    return result[["series", "segment", "start", "end", "hours", "mean", "ratio"]]


def onset_from_segments(segments: pd.DataFrame, *, min_effect: float) -> dict[str, Any]:
    """First segment whose mean departs from the first (baseline) segment by >= min_effect."""
    if segments.empty:
        return {"onset": None, "baseline_mean": None, "first_shift_mean": None, "change_points": []}
    ordered = segments.sort_values("start").reset_index(drop=True)
    baseline = float(ordered.loc[0, "mean"])
    onset = None
    shift_mean = None
    for row in ordered.iloc[1:].itertuples(index=False):
        if abs(float(row.mean) - baseline) >= min_effect:
            onset = row.start
            shift_mean = float(row.mean)
            break
    return {
        "onset": onset,
        "baseline_mean": baseline,
        "first_shift_mean": shift_mean,
        "change_points": [row.start for row in ordered.iloc[1:].itertuples(index=False)],
    }


# --------------------------------------------------------------------------------------
# Reference profile and windows
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    name: str
    start: pd.Timestamp  # exclusive
    end: pd.Timestamp  # inclusive
    kind: str  # "reference" | "trailing" | "regime"

    def mask(self, times: pd.Series) -> pd.Series:
        return (times > self.start) & (times <= self.end)

    @property
    def hours(self) -> float:
        return float((self.end - self.start) / pd.Timedelta(hours=1))


def reference_period(onset: pd.Timestamp, data_start: pd.Timestamp, *, days: int = 28) -> Window:
    """`days` complete local days ending at the local midnight on/before the onset."""
    local_onset = pd.Timestamp(onset).tz_convert(BOGOTA_TZ)
    end_local = local_onset.normalize()
    start_local = end_local - pd.Timedelta(days=days)
    earliest = pd.Timestamp(data_start).tz_convert(BOGOTA_TZ).ceil("D")
    start_local = max(start_local, earliest)
    # Window.start is exclusive: step back one slot so the first local-midnight slot is included
    return Window("reference", start_local.tz_convert("UTC") - SLOT, end_local.tz_convert("UTC") - SLOT, "reference")


def trailing_windows(anchor: pd.Timestamp, hours: Iterable[int] = (6, 12, 24, 48)) -> list[Window]:
    anchor = pd.Timestamp(anchor).tz_convert("UTC")
    return [Window(f"last_{h}h", anchor - pd.Timedelta(hours=h), anchor, "trailing") for h in hours]


def reference_profile(prepared: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Median demand per (station, local weekday, local slot) inside the reference window."""
    rows = prepared[window.mask(prepared["observed_at"])]
    return rows.groupby(["station_id", "dow", "slot"], as_index=False)["demand"].median().rename(columns={"demand": "ref"})


def attach_reference(prepared: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    return prepared.merge(profile, on=["station_id", "dow", "slot"], how="left")


def expected_at(profile: pd.DataFrame, station_id: str, times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Reference expectation for arbitrary (possibly shifted) UTC times of one station."""
    local = pd.DatetimeIndex(times).tz_convert(BOGOTA_TZ)
    lookup = profile[profile["station_id"] == station_id].set_index(["dow", "slot"])["ref"]
    keys = pd.MultiIndex.from_arrays([local.dayofweek, local.hour * 4 + local.minute // 15])
    return lookup.reindex(keys).to_numpy(dtype=float)


def window_coverage(prepared: pd.DataFrame, window: Window) -> dict[str, Any]:
    stations = prepared["station_id"].nunique()
    expected = int(round(window.hours * 4)) * stations
    rows = int(window.mask(prepared["observed_at"]).sum())
    coverage = rows / expected if expected else 0.0
    return {
        "window": window.name,
        "kind": window.kind,
        "start_utc": _iso(window.start + SLOT),
        "end_utc": _iso(window.end),
        "start_local": _iso_local(window.start + SLOT),
        "end_local": _iso_local(window.end),
        "hours": window.hours,
        "rows": rows,
        "expected_rows": expected,
        "coverage": coverage,
        "sufficient": bool(coverage >= 0.9 and rows >= 12 * stations),
    }


def daytype_signature(window: Window) -> tuple[bool, ...]:
    """Weekend flags of every local date touched by the window, including the day before
    (that day feeds lag_96 at the start of the window)."""
    first = (window.start + SLOT).tz_convert(BOGOTA_TZ).normalize() - pd.Timedelta(days=1)
    last = window.end.tz_convert(BOGOTA_TZ).normalize()
    days = pd.date_range(first, last, freq="D")
    return tuple(bool(day.dayofweek >= 5) for day in days)


def same_clock_null_windows(window: Window, reference: Window, *, match_daytype: bool = True) -> list[Window]:
    """Windows with the same length and clock span as `window`, shifted back whole days so
    that they fall completely inside the reference period (and, by default, cover the same
    weekday/weekend pattern)."""
    signature = daytype_signature(window)
    windows = []
    k = 1
    while True:
        shift = pd.Timedelta(days=k)
        start, end = window.start - shift, window.end - shift
        if start < reference.start:
            break
        candidate = Window(f"{window.name}_null_{k}", start, end, "null")
        if end <= reference.end and (not match_daytype or daytype_signature(candidate) == signature):
            windows.append(candidate)
        k += 1
    return windows


def reference_matrix(profile: pd.DataFrame, index: pd.DatetimeIndex, stations: Iterable[str]) -> pd.DataFrame:
    """Reference expectation laid out like `demand_matrix` (for anomaly computations)."""
    return pd.DataFrame({station: expected_at(profile, station, index) for station in stations}, index=index)


# --------------------------------------------------------------------------------------
# Station-level data drift metrics
# --------------------------------------------------------------------------------------


def station_window_metrics(
    prepared: pd.DataFrame,
    profile: pd.DataFrame,
    window: Window,
    *,
    max_shift_slots: int = 32,
) -> pd.DataFrame:
    """Level, profile and volatility metrics of each station in `window` vs the reference profile."""
    rows = attach_reference(prepared[window.mask(prepared["observed_at"])], profile)
    records = []
    for station_id, group in rows.groupby("station_id"):
        group = group.sort_values("observed_at")
        y = group["demand"].to_numpy(dtype=float)
        ref = group["ref"].to_numpy(dtype=float)
        ok = np.isfinite(ref)
        level_ratio = _safe_ratio(y[ok].sum(), ref[ok].sum())
        hourly = group[ok].groupby("local_hour").agg(y=("demand", "sum"), ref=("ref", "sum"))
        hour_lr = np.log((hourly["y"] + HOURLY_OFFSET) / (hourly["ref"] + HOURLY_OFFSET))
        dispersion = float(hour_lr.std(ddof=0)) if len(hour_lr) >= 3 else math.nan
        shape_corr = _corr(y[ok], ref[ok])
        timing = timing_shift(group["observed_at"], y, profile, str(station_id), max_shift_slots=max_shift_slots)
        records.append(
            {
                "station_id": str(station_id),
                "window": window.name,
                "rows": int(len(group)),
                "window_slots": int(len(group)),
                "mean_demand": float(np.mean(y)) if len(y) else math.nan,
                "reference_mean": float(np.mean(ref[ok])) if ok.any() else math.nan,
                "level_ratio": level_ratio,
                "log_level_ratio": _safe_log(level_ratio),
                "hourly_ratio_dispersion": dispersion,
                "shape_corr": shape_corr,
                "volatility": _volatility(y),
                **timing,
            }
        )
    return pd.DataFrame(records)


def timing_shift(
    times: pd.Series,
    values: np.ndarray,
    profile: pd.DataFrame,
    station_id: str,
    *,
    max_shift_slots: int = 32,
) -> dict[str, float]:
    """Cross-correlate the window with the reference expectation shifted in time.

    `timing_shift_minutes` > 0 means the observed pattern happens *later* than the reference
    (actual(t) best matches reference(t - shift)).
    """
    times = pd.DatetimeIndex(times)
    best_corr, best_shift = -math.inf, 0
    corr_zero = math.nan
    if len(values) < 8:
        return {"timing_shift_minutes": math.nan, "timing_corr_at_zero": math.nan, "timing_corr_best": math.nan, "timing_corr_gain": math.nan}
    for k in range(-max_shift_slots, max_shift_slots + 1):
        expected = expected_at(profile, station_id, times - k * SLOT)
        corr = _corr(values, expected)
        if k == 0:
            corr_zero = corr
        if math.isfinite(corr) and (corr > best_corr + 1e-12 or (abs(corr - best_corr) <= 1e-12 and abs(k) < abs(best_shift))):
            best_corr, best_shift = corr, k
    if not math.isfinite(best_corr):
        return {"timing_shift_minutes": math.nan, "timing_corr_at_zero": corr_zero, "timing_corr_best": math.nan, "timing_corr_gain": math.nan}
    return {
        "timing_shift_minutes": float(best_shift * 15),
        "timing_corr_at_zero": corr_zero,
        "timing_corr_best": best_corr,
        "timing_corr_gain": best_corr - corr_zero if math.isfinite(corr_zero) else math.nan,
    }


def null_station_metrics(prepared: pd.DataFrame, window: Window, reference: Window) -> pd.DataFrame:
    """Same metrics on same-clock windows of the reference period, each scored against a
    profile that excludes the null window's own local dates (out-of-sample null)."""
    frames = []
    for null in same_clock_null_windows(window, reference):
        null_dates = set(prepared.loc[null.mask(prepared["observed_at"]), "local_date"])
        source = prepared[reference.mask(prepared["observed_at"]) & ~prepared["local_date"].isin(null_dates)]
        profile = source.groupby(["station_id", "dow", "slot"], as_index=False)["demand"].median().rename(columns={"demand": "ref"})
        frames.append(station_window_metrics(prepared, profile, null, max_shift_slots=8).assign(null_window=null.name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_null(null: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if null.empty:
        return pd.DataFrame(columns=["station_id"])
    parts = []
    for metric in metrics:
        stats = null.groupby("station_id")[metric].agg(["median", "min", "max", "count"])
        stats.columns = [f"null_{metric}_{name}" for name in stats.columns]
        parts.append(stats)
    abs_level = null.assign(abs_log_level=null["log_level_ratio"].abs()).groupby("station_id")["abs_log_level"].max().rename("null_abs_log_level_max")
    return pd.concat(parts + [abs_level], axis=1).reset_index()


def daily_peaks(prepared: pd.DataFrame, profile: pd.DataFrame, dates: Iterable[str]) -> pd.DataFrame:
    """Peak magnitude and clock time per local date and band vs the same-weekday reference."""
    dates = list(dates)
    rows = prepared[prepared["local_date"].isin(dates)]
    records = []
    for (station_id, date), group in rows.groupby(["station_id", "local_date"]):
        dow = int(group["dow"].iloc[0])
        ref = profile[(profile["station_id"] == station_id) & (profile["dow"] == dow)].set_index("slot")["ref"]
        for band, slots in PEAK_BANDS.items():
            observed = group[group["slot"].isin(slots)].set_index("slot")["demand"]
            reference = ref.reindex(observed.index).dropna()
            if observed.empty or reference.empty:
                continue
            obs_slot = int(observed.idxmax())
            ref_band = ref.reindex(slots).dropna()
            ref_slot = int(ref_band.idxmax())
            records.append(
                {
                    "station_id": station_id,
                    "local_date": date,
                    "band": band,
                    "slots_observed": int(observed.size),
                    "band_complete": bool(observed.size == len(slots)),
                    "peak": float(observed.max()),
                    "peak_local_time": _slot_label(obs_slot),
                    "reference_peak": float(ref_band.max()),
                    "reference_peak_local_time": _slot_label(ref_slot),
                    "peak_ratio": _safe_ratio(float(observed.max()), float(ref_band.max())),
                    "peak_shift_minutes": float(_circular_slot_delta(obs_slot, ref_slot) * 15) if band != "night" else math.nan,
                }
            )
    return pd.DataFrame(records)


def daily_volatility(prepared: pd.DataFrame) -> pd.DataFrame:
    """Std of 15-min changes of log1p(demand) per station and local date (level-invariant noise)."""
    frame = prepared.sort_values(["station_id", "observed_at"]).copy()
    frame["dlog"] = frame.groupby("station_id")["demand"].transform(lambda values: np.log1p(values).diff())
    return frame.groupby(["station_id", "local_date"]).agg(volatility=("dlog", "std"), slots=("dlog", "size")).reset_index()


def hour_of_day_change(prepared: pd.DataFrame, profile: pd.DataFrame, window: Window) -> pd.DataFrame:
    """log2 ratio of observed vs reference demand per station x local hour inside `window`."""
    rows = attach_reference(prepared[window.mask(prepared["observed_at"])], profile).dropna(subset=["ref"])
    grouped = rows.groupby(["station_id", "local_hour"]).agg(actual=("demand", "sum"), ref=("ref", "sum"), slots=("demand", "size")).reset_index()
    grouped["log2_ratio"] = np.log2((grouped["actual"] + HOURLY_OFFSET) / (grouped["ref"] + HOURLY_OFFSET))
    grouped["window"] = window.name
    return grouped


# --------------------------------------------------------------------------------------
# Lag relationships
# --------------------------------------------------------------------------------------


def lag_metrics(wide: pd.DataFrame, window: Window, *, ref_wide: pd.DataFrame | None = None) -> pd.DataFrame:
    """How well each lag explains demand inside `window` (per station and pooled).

    - `lag_corr`: Pearson correlation of log1p(y_t) vs log1p(y_{t-k}).
    - `naive_accuracy`: 100 * max(0, 1 - WAPE) of the naive forecast y_hat_t = y_{t-k}.
    - `median_log_ratio`: median log((y_t + c) / (y_{t-k} + c)); 0 = lag unbiased.
    - `anomaly_persistence` (needs `ref_wide`): corr of the profile anomaly at t and t-k.
    """
    times = wide.index[(wide.index > window.start) & (wide.index <= window.end)]
    anomaly = np.log((wide + SLOT_OFFSET) / (ref_wide.reindex_like(wide) + SLOT_OFFSET)) if ref_wide is not None else None
    records = []
    for lag_name, steps in LAGS.items():
        lagged = wide.shift(steps)
        pooled_y, pooled_x = [], []
        for station in wide.columns:
            y = wide.loc[times, station].to_numpy(dtype=float)
            x = lagged.loc[times, station].to_numpy(dtype=float)
            ok = np.isfinite(y) & np.isfinite(x)
            pooled_y.append(y[ok])
            pooled_x.append(x[ok])
            persistence = math.nan
            if anomaly is not None:
                a_now = anomaly.loc[times, station].to_numpy(dtype=float)
                a_lag = anomaly[station].shift(steps).loc[times].to_numpy(dtype=float)
                persistence = _corr(a_now, a_lag)
            records.append({"station_id": str(station), "lag": lag_name, "window": window.name, **_lag_stats(y[ok], x[ok]), "anomaly_persistence": persistence})
        y_all = np.concatenate(pooled_y) if pooled_y else np.array([])
        x_all = np.concatenate(pooled_x) if pooled_x else np.array([])
        records.append({"station_id": "ALL", "lag": lag_name, "window": window.name, **_lag_stats(y_all, x_all), "anomaly_persistence": math.nan})
    return pd.DataFrame(records)


def null_lag_metrics(wide: pd.DataFrame, window: Window, reference: Window) -> pd.DataFrame:
    frames = [lag_metrics(wide, null).assign(null_window=null.name) for null in same_clock_null_windows(window, reference)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def lag_reliability(window_metrics: pd.DataFrame, null_metrics: pd.DataFrame, *, broken_drop_pp: float = 15.0, tolerance_pp: float = 3.0) -> pd.DataFrame:
    """Label each (station, lag) as reliable / degraded / broken.

    reliable: naive accuracy inside the null range (same-clock reference windows) or within
    `tolerance_pp` of the null median; broken: more than `broken_drop_pp` below the null
    median and below the null minimum; degraded: anything in between.
    """
    if null_metrics.empty:
        return window_metrics.assign(null_median=math.nan, null_min=math.nan, null_max=math.nan, null_lag_corr=math.nan,
                                     naive_accuracy_delta_pp=math.nan, lag_corr_delta=math.nan, reliability="insufficient")
    null = null_metrics.groupby(["station_id", "lag"])["naive_accuracy"].agg(null_median="median", null_min="min", null_max="max").reset_index()
    null_corr = null_metrics.groupby(["station_id", "lag"])["lag_corr"].median().rename("null_lag_corr").reset_index()
    merged = window_metrics.merge(null, on=["station_id", "lag"], how="left").merge(null_corr, on=["station_id", "lag"], how="left")
    merged["naive_accuracy_delta_pp"] = merged["naive_accuracy"] - merged["null_median"]
    merged["lag_corr_delta"] = merged["lag_corr"] - merged["null_lag_corr"]

    def label(row: pd.Series) -> str:
        if not math.isfinite(row["naive_accuracy"]) or not math.isfinite(row.get("null_median", math.nan)):
            return "insufficient"
        if row["naive_accuracy"] >= row["null_min"] or row["naive_accuracy_delta_pp"] >= -tolerance_pp:
            return "reliable"
        if row["naive_accuracy_delta_pp"] < -broken_drop_pp:
            return "broken"
        return "degraded"

    merged["reliability"] = merged.apply(label, axis=1)
    return merged


# --------------------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------------------


def resolve_predictions(
    *,
    predictions: pd.DataFrame,
    forecast_cycles: pd.DataFrame,
    forecast_cycle_targets: pd.DataFrame,
    observations: pd.DataFrame,
    submission_receipts: pd.DataFrame,
    model_versions: pd.DataFrame,
) -> pd.DataFrame:
    """Join every persisted prediction (official or shadow) with its target and revealed actual.

    Only (cycle, model) pairs whose full target set is predicted *and* resolved are kept.
    `actual` is information revealed after `data_cutoff`: it is used for evaluation only.
    """
    preds = predictions.copy()
    preds["station_id"] = preds["station_id"].astype(str).str.zfill(5)
    preds["target_at"] = pd.to_datetime(preds["target_observed_at"], utc=True)
    targets = forecast_cycle_targets.copy()
    targets["station_id"] = targets["station_id"].astype(str).str.zfill(5)
    targets["target_at"] = pd.to_datetime(targets["target_at"], utc=True)
    cycles = forecast_cycles[["cycle_id", "data_cutoff", "expected_predictions"]].copy()
    cycles["data_cutoff"] = pd.to_datetime(cycles["data_cutoff"], utc=True)
    obs = prepare_observations(observations)[["station_id", "observed_at", "demand"]]

    frame = preds.merge(
        targets[["cycle_id", "station_id", "target_at", "horizon_minutes"]],
        on=["cycle_id", "station_id", "target_at"],
        how="inner",
        suffixes=("_pred", ""),
    ).merge(cycles, on="cycle_id", how="inner")
    lead = (frame["target_at"] - frame["data_cutoff"]) / pd.Timedelta(minutes=1)
    if (lead <= 0).any():
        raise ValueError("prediction target at or before its data_cutoff")
    if not np.allclose(lead.to_numpy(dtype=float), frame["horizon_minutes"].astype(float).to_numpy()):
        raise ValueError("horizon_minutes does not match target_at - data_cutoff")

    frame = frame.merge(obs.rename(columns={"observed_at": "target_at", "demand": "actual"}), on=["station_id", "target_at"], how="inner")
    expected = targets.groupby("cycle_id").size().rename("target_count")
    counts = frame.groupby(["cycle_id", "model_id"]).size().rename("resolved_count").reset_index()
    counts = counts.merge(expected, left_on="cycle_id", right_index=True, how="left")
    complete = counts[counts["resolved_count"] == counts["target_count"]][["cycle_id", "model_id"]]
    frame = frame.merge(complete, on=["cycle_id", "model_id"], how="inner")

    receipts = submission_receipts.copy()
    official = receipts[(receipts["is_official"].astype(str).str.lower() == "true") & receipts["status"].isin(["accepted", "idempotent_replay"])]
    official_keys = set(zip(official["cycle_id"].astype(str), official["model_id"].astype(str)))
    frame["is_official"] = [(str(c), str(m)) in official_keys for c, m in zip(frame["cycle_id"], frame["model_id"])]
    names = model_versions.set_index(model_versions["model_id"].astype(str))["model_name"].to_dict()
    frame["model_name"] = frame["model_id"].astype(str).map(names).fillna("unknown")

    frame["prediction"] = frame["prediction"].astype(float)
    frame["actual"] = frame["actual"].astype(float)
    frame["residual"] = frame["actual"] - frame["prediction"]
    frame["abs_error"] = frame["residual"].abs()
    frame["pred_actual_ratio"] = np.where(frame["actual"] > 0, frame["prediction"] / frame["actual"], np.nan)
    local = frame["target_at"].dt.tz_convert(BOGOTA_TZ)
    frame["target_hour_local"] = local.dt.hour
    frame["data_cutoff_local"] = frame["data_cutoff"].dt.tz_convert(BOGOTA_TZ)
    frame["actual_revealed_after_cutoff"] = frame["target_at"] > frame["data_cutoff"]
    columns = [
        "cycle_id", "model_id", "model_name", "is_official", "station_id", "data_cutoff", "data_cutoff_local", "target_at",
        "horizon_minutes", "target_hour_local", "prediction", "actual", "residual", "abs_error", "pred_actual_ratio",
        "actual_revealed_after_cutoff",
    ]
    return frame[columns].sort_values(["data_cutoff", "model_id", "station_id", "target_at"]).reset_index(drop=True)


def error_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """WAPE/accuracy/MAE via the canonical helper + bias in the actual - prediction convention."""
    if frame.empty:
        return pd.DataFrame(columns=group_cols + ["count", "wape", "accuracy", "global_wape_accuracy", "mean_station_accuracy", "mae", "signed_bias", "relative_bias", "pred_actual_ratio"])
    # drift_monitor sign; abs_error recomputed so the helper also scores alternative prediction columns
    canonical = frame.assign(residual=frame["prediction"] - frame["actual"], abs_error=(frame["prediction"] - frame["actual"]).abs())
    base = _accuracy_by(canonical, group_cols) if group_cols else _accuracy_by(canonical.assign(_all="all"), ["_all"]).drop(columns="_all")
    base["signed_bias"] = -base.pop("bias")
    base["relative_bias"] = -base.pop("signed_relative_bias")
    sums = frame.groupby(group_cols).agg(pred_sum=("prediction", "sum"), actual_sum=("actual", "sum")).reset_index() if group_cols else pd.DataFrame(
        [{"pred_sum": frame["prediction"].sum(), "actual_sum": frame["actual"].sum()}]
    )
    sums["pred_actual_ratio"] = sums["pred_sum"] / sums["actual_sum"].where(sums["actual_sum"] > 0)
    result = pd.concat([base.reset_index(drop=True), sums[["pred_actual_ratio"]].reset_index(drop=True)], axis=1) if not group_cols else base.merge(sums[group_cols + ["pred_actual_ratio"]], on=group_cols, how="left")
    # `accuracy` pools every row of the group (= global WAPE accuracy); `mean_station_accuracy` is the
    # leaderboard formula (station WAPE pooled over the group, then averaged over stations)
    result["global_wape_accuracy"] = result["accuracy"]
    if "station_id" in frame.columns and "station_id" not in group_cols:
        scored = canonical[[*group_cols, "station_id", "actual", "abs_error"]]
        if group_cols:
            leaderboard = scored.groupby(group_cols).apply(lambda g: accuracy_summary(g)["mean_station_accuracy"], include_groups=False).rename("mean_station_accuracy").reset_index()
            result = result.merge(leaderboard, on=group_cols, how="left")
        else:
            result["mean_station_accuracy"] = accuracy_summary(scored)["mean_station_accuracy"]
    else:
        result["mean_station_accuracy"] = result["accuracy"]
    return result[group_cols + ["count", "wape", "accuracy", "global_wape_accuracy", "mean_station_accuracy", "mae", "signed_bias", "relative_bias", "pred_actual_ratio"]]


def cycle_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Per (cycle, model): `global_wape_accuracy` (= `accuracy`, all 48 rows pooled) and
    `mean_station_accuracy` (= `macro_accuracy`, the leaderboard formula applied to one cycle)."""
    pooled = error_summary(scored, ["cycle_id", "model_id"])
    macro = pooled[["cycle_id", "model_id", "mean_station_accuracy"]].rename(columns={"mean_station_accuracy": "macro_accuracy"})
    meta = scored.groupby(["cycle_id", "model_id"]).agg(model_name=("model_name", "first"), is_official=("is_official", "first"), data_cutoff=("data_cutoff", "first")).reset_index()
    result = meta.merge(pooled, on=["cycle_id", "model_id"]).merge(macro, on=["cycle_id", "model_id"])
    result["data_cutoff_local"] = result["data_cutoff"].dt.tz_convert(BOGOTA_TZ)
    return result.sort_values(["data_cutoff", "model_name"]).reset_index(drop=True)


def cycle_window(cycles: pd.DataFrame, window: Window) -> pd.Series:
    return window.mask(cycles["data_cutoff"])


def extreme_errors(scored: pd.DataFrame, *, top: int = 25, min_actual: float = 50.0) -> pd.DataFrame:
    """Largest absolute errors plus gross ratio misses (pred/actual >= 2 or <= 0.5)."""
    frame = scored.copy()
    frame["gross_overprediction"] = (frame["actual"] >= min_actual) & (frame["pred_actual_ratio"] >= 2.0)
    frame["gross_underprediction"] = (frame["actual"] >= min_actual) & (frame["pred_actual_ratio"] <= 0.5)
    threshold = frame["abs_error"].quantile(0.99) if len(frame) else math.inf
    frame["top1pct_abs_error"] = frame["abs_error"] >= threshold
    flagged = frame[frame["gross_overprediction"] | frame["gross_underprediction"] | frame["top1pct_abs_error"]]
    return flagged.sort_values("abs_error", ascending=False).head(top).reset_index(drop=True)


def paired_comparison(scored: pd.DataFrame, champion_model_id: str, challenger_model_id: str) -> dict[str, Any]:
    """Champion vs challenger only on cycles where both have a complete, resolved prediction set."""
    champion = scored[scored["model_id"].astype(str) == str(champion_model_id)]
    challenger = scored[scored["model_id"].astype(str) == str(challenger_model_id)]
    shared = sorted(set(champion["cycle_id"]) & set(challenger["cycle_id"]))
    if not shared:
        return {"status": "no_comparable_cycles", "shared_cycles": 0}
    champ = champion[champion["cycle_id"].isin(shared)]
    chall = challenger[challenger["cycle_id"].isin(shared)]
    keys = ["cycle_id", "station_id", "target_at"]
    paired = champ[keys + ["horizon_minutes", "actual", "prediction", "data_cutoff"]].merge(
        chall[keys + ["prediction"]], on=keys, suffixes=("_champion", "_challenger")
    )
    per_cycle = []
    for cycle_id, group in paired.groupby("cycle_id"):
        a = _pooled_accuracy(group["actual"], group["prediction_champion"])
        b = _pooled_accuracy(group["actual"], group["prediction_challenger"])
        per_cycle.append({"cycle_id": cycle_id, "data_cutoff": group["data_cutoff"].iloc[0], "champion_accuracy": a, "challenger_accuracy": b, "delta_pp": b - a})
    per_cycle_frame = pd.DataFrame(per_cycle).sort_values("data_cutoff")
    by_station = _paired_by(paired, "station_id")
    by_horizon = _paired_by(paired, "horizon_minutes")
    return {
        "status": "ok",
        "shared_cycles": len(shared),
        "rows": int(len(paired)),
        "champion_pooled_accuracy": _pooled_accuracy(paired["actual"], paired["prediction_champion"]),
        "challenger_pooled_accuracy": _pooled_accuracy(paired["actual"], paired["prediction_challenger"]),
        "mean_cycle_delta_pp": float(per_cycle_frame["delta_pp"].mean()),
        "median_cycle_delta_pp": float(per_cycle_frame["delta_pp"].median()),
        "challenger_wins": int((per_cycle_frame["delta_pp"] > 0).sum()),
        "challenger_losses": int((per_cycle_frame["delta_pp"] < 0).sum()),
        "per_cycle": per_cycle_frame,
        "by_station": by_station,
        "by_horizon": by_horizon,
    }


# --------------------------------------------------------------------------------------
# Station classification and recommendations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationConfig:
    """Diagnostic thresholds (not model parameters). A shift must exceed both the station's
    own null range (same-clock reference windows) and a minimum practical effect."""

    level_min_effect: float = math.log(1.10)
    level_moderate: float = math.log(1.25)
    level_high: float = math.log(1.5)
    level_extreme: float = math.log(2.0)
    profile_dispersion_min_effect: float = 0.20
    timing_min_minutes: float = 30.0
    timing_min_gain: float = 0.05
    shape_corr_drop: float = 0.10
    min_timing_slots: int = 24
    volatility_ratio: float = 1.5
    min_eval_rows: int = 40
    min_shock_rows: int = 12
    accuracy_good: float = 85.0
    bias_absorbed: float = 0.05
    shock_accuracy_floor: float = 70.0
    min_day_consistency: float = 0.8
    min_regime_hours_for_retraining: float = 72.0


def classify_level(log_ratio: float, null_abs_max: float, config: ClassificationConfig) -> str:
    if not math.isfinite(log_ratio):
        return "insufficient"
    effect = abs(log_ratio)
    if effect <= max(config.level_min_effect, null_abs_max if math.isfinite(null_abs_max) else 0.0):
        return "none"
    direction = "up" if log_ratio > 0 else "down"
    if effect >= config.level_extreme:
        return f"extreme_{direction}"
    if effect >= config.level_high:
        return f"high_{direction}"
    if effect >= config.level_moderate:
        return f"moderate_{direction}"
    return f"mild_{direction}"


def classify_profile(row: pd.Series, config: ClassificationConfig) -> str:
    """Profile/temporal shift = hour-dependent change (dispersion), timing shift, or shape loss."""
    dispersion = row.get("hourly_ratio_dispersion", math.nan)
    null_disp = row.get("null_hourly_ratio_dispersion_max", math.nan)
    shape = row.get("shape_corr", math.nan)
    null_shape = row.get("null_shape_corr_min", math.nan)
    timing = row.get("timing_shift_minutes", math.nan)
    gain = row.get("timing_corr_gain", math.nan)
    if row.get("window_slots", config.min_timing_slots) < config.min_timing_slots:
        timing = math.nan  # a +/-8h lag search on a short window is not identifiable
    signals = []
    if math.isfinite(dispersion) and dispersion > max(config.profile_dispersion_min_effect, null_disp if math.isfinite(null_disp) else 0.0):
        signals.append("hour_dependent_change")
    if math.isfinite(shape) and math.isfinite(null_shape) and shape < null_shape - config.shape_corr_drop:
        signals.append("shape_loss")
    if math.isfinite(timing) and abs(timing) >= config.timing_min_minutes and math.isfinite(gain) and gain >= config.timing_min_gain:
        signals.append(f"timing_{'later' if timing > 0 else 'earlier'}_{int(abs(timing))}min")
    if not signals:
        return "none"
    strength = "strong" if len(signals) >= 2 or (math.isfinite(dispersion) and dispersion > 2 * max(config.profile_dispersion_min_effect, null_disp if math.isfinite(null_disp) else 0.0)) else "moderate"
    return f"{strength}:" + "+".join(signals)


def classify_variance(row: pd.Series, config: ClassificationConfig) -> str:
    vol = row.get("volatility", math.nan)
    null_max = row.get("null_volatility_max", math.nan)
    null_med = row.get("null_volatility_median", math.nan)
    if not (math.isfinite(vol) and math.isfinite(null_med) and null_med > 0):
        return "insufficient"
    ratio = vol / null_med
    if ratio >= config.volatility_ratio and (not math.isfinite(null_max) or vol > null_max):
        return f"up_x{ratio:.2f}"
    if ratio <= 1 / config.volatility_ratio:
        return f"down_x{ratio:.2f}"
    return "none"


def severity_label(level_label: str, profile_label: str) -> str:
    order = ["none", "low", "moderate", "high", "extreme"]
    level_map = {"none": "none", "mild": "low", "moderate": "moderate", "high": "high", "extreme": "extreme"}
    level = level_map.get(level_label.split("_")[0], "none")
    profile = "none" if profile_label == "none" else ("high" if profile_label.startswith("strong") else "moderate")
    return max([level, profile], key=order.index)


@dataclass(frozen=True)
class StationResponse:
    primary: str
    sustained: str
    shock: str
    responses: list[str]
    evidence: list[str]


RESPONSE_PRIORITY = [
    "local retraining",
    "recent profile refit",
    "station recalibration",
    "reduce lag96 reliance",
    "robust recent baseline",
    "keep current behavior",
    "insufficient evidence",
]


def recommend_station_response(row: dict[str, Any], config: ClassificationConfig | None = None) -> StationResponse:
    """Map diagnostics to responses, separately for the sustained regime and the latest shock.

    Sustained: data shifts in the 48h window, judged against Candidate G's accuracy/bias in the
    sustained regime (a shift G2 already absorbs maps to "keep current behavior").
    Shock: the newest regime; too young to refit, so it can only call for a robust guard.
    """
    config = config or ClassificationConfig()
    evidence: list[str] = []
    sustained_level = str(row.get("sustained_level_shift", "none"))
    sustained_profile = str(row.get("sustained_profile_shift", "none"))
    variance = str(row.get("sustained_variance_shift", "none"))
    shock_level = str(row.get("shock_level_shift", "none"))
    shock_profile = str(row.get("shock_profile_shift", "none"))
    lag96 = str(row.get("lag96_reliability", "insufficient"))
    weekly = str(row.get("weekly_lag_reliability", "insufficient"))
    short = [str(row.get(f"{name}_reliability", "insufficient")) for name in ("lag15", "lag30", "lag60")]
    consistency = _num(row.get("new_regime_day_consistency"))
    sustained_rows = int(row.get("g_sustained_rows", 0) or 0)
    sustained_acc = _num(row.get("g_sustained_accuracy"))
    sustained_bias = _num(row.get("g_sustained_relative_bias"))
    shock_rows = int(row.get("g_shock_rows", 0) or 0)
    shock_acc = _num(row.get("g_shock_accuracy"))
    regime_hours = _num(row.get("new_regime_hours"))

    has_level = sustained_level not in ("none", "insufficient")
    has_profile = sustained_profile not in ("none", "insufficient")
    has_variance = variance.startswith("up")
    structural = sum([has_level, has_profile, has_variance, lag96 == "broken"])
    g_handles_it = (
        sustained_rows >= config.min_eval_rows
        and math.isfinite(sustained_acc)
        and sustained_acc >= config.accuracy_good
        and (not math.isfinite(sustained_bias) or abs(sustained_bias) < config.bias_absorbed)
    )

    sustained: list[str] = []
    if sustained_rows < config.min_eval_rows:
        sustained.append("insufficient evidence")
        evidence.append(f"only {sustained_rows} resolved Candidate G rows in the sustained regime")
    elif g_handles_it:
        sustained.append("keep current behavior")
        if has_level or has_profile:
            evidence.append(
                f"sustained shift (level {sustained_level}, profile {sustained_profile}) already absorbed: "
                f"G accuracy {sustained_acc:.1f}%, bias {sustained_bias:+.1%}"
            )
        else:
            evidence.append(f"no material sustained shift; G accuracy {sustained_acc:.1f}%")
    else:
        level_biased = math.isfinite(sustained_bias) and abs(sustained_bias) >= config.bias_absorbed
        if has_level and (not has_profile or level_biased):
            sustained.append("station recalibration")
            evidence.append(f"level shift {sustained_level}" + (" without hour-dependent change" if not has_profile else " leaves a material level bias"))
        if has_profile:
            if math.isfinite(consistency) and consistency >= config.min_day_consistency:
                sustained.append("recent profile refit")
                evidence.append(f"profile shift {sustained_profile}; new profile repeats across the last 2 days (corr={consistency:.2f})")
            else:
                sustained.append("robust recent baseline")
                evidence.append(f"profile shift {sustained_profile} not yet repeatable across days (corr={_fmt(consistency)})")
        if lag96 in ("broken", "degraded") or weekly in ("broken", "degraded"):
            if all(state == "reliable" for state in short):
                sustained.append("reduce lag96 reliance")
            evidence.append(f"lag96={lag96}, weekly={weekly}, lag15/30/60={'/'.join(short)}")
        if has_variance:
            sustained.append("robust recent baseline")
            evidence.append(f"volatility {variance}")
        if structural >= 3:
            if math.isfinite(regime_hours) and regime_hours >= config.min_regime_hours_for_retraining:
                sustained.append("local retraining")
                evidence.append(f"{structural} structural changes in a regime of {regime_hours:.0f}h")
            else:
                evidence.append(f"{structural} structural changes; local retraining deferred (new regime only {_fmt(regime_hours)}h old)")
        if not sustained:
            sustained.append("station recalibration")
        if math.isfinite(sustained_bias) and abs(sustained_bias) >= config.bias_absorbed:
            evidence.append(f"G sustained relative bias {sustained_bias:+.1%} ({'under' if sustained_bias > 0 else 'over'}-prediction), accuracy {_fmt(sustained_acc)}%")

    g2_sustained = _num(row.get("g2_effect_pp_sustained"))
    g2_shock = _num(row.get("g2_effect_pp_shock"))
    if math.isfinite(g2_sustained) and abs(g2_sustained) >= 5:
        evidence.append(f"G2 calibration {'adds' if g2_sustained > 0 else 'costs'} {g2_sustained:+.1f} pp vs the base in the sustained regime")
    if math.isfinite(g2_shock) and abs(g2_shock) >= 5:
        evidence.append(f"G2 calibration {'adds' if g2_shock > 0 else 'costs'} {g2_shock:+.1f} pp in the shock")
    severe_shock = shock_level.split("_")[0] in ("high", "extreme") or shock_profile.startswith("strong")
    if shock_rows < config.min_shock_rows:
        shock = "insufficient evidence"
    elif severe_shock and math.isfinite(shock_acc) and shock_acc < config.shock_accuracy_floor:
        shock = "robust recent baseline"
        evidence.append(f"latest shock (level {shock_level}, profile {shock_profile}) took G to {shock_acc:.1f}%; too recent to refit")
    elif severe_shock:
        shock = "keep current behavior"
        evidence.append(f"latest shock (level {shock_level}) but G held {_fmt(shock_acc)}%")
    else:
        shock = "keep current behavior"

    responses = list(dict.fromkeys(sustained + [shock]))
    primary = min(responses, key=RESPONSE_PRIORITY.index)
    if primary == "keep current behavior" and len(responses) > 1:
        primary = min([r for r in responses if r != "keep current behavior"], key=RESPONSE_PRIORITY.index)
    sustained_primary = min(sustained, key=RESPONSE_PRIORITY.index)
    return StationResponse(primary=primary, sustained=sustained_primary, shock=shock, responses=responses, evidence=evidence)


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _lag_stats(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    if y.size < 4:
        return {"rows": int(y.size), "lag_corr": math.nan, "naive_accuracy": math.nan, "median_log_ratio": math.nan}
    actual_sum = float(np.sum(y))
    wape = float(np.sum(np.abs(y - x)) / actual_sum) if actual_sum > 0 else math.nan
    return {
        "rows": int(y.size),
        "lag_corr": _corr(np.log1p(y), np.log1p(x)),
        "naive_accuracy": 100.0 * max(0.0, 1.0 - wape) if math.isfinite(wape) else math.nan,
        "median_log_ratio": float(np.median(np.log((y + SLOT_OFFSET) / (x + SLOT_OFFSET)))),
    }


def _paired_by(paired: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for key, group in paired.groupby(column):
        a = _pooled_accuracy(group["actual"], group["prediction_champion"])
        b = _pooled_accuracy(group["actual"], group["prediction_challenger"])
        rows.append({column: key, "rows": int(len(group)), "champion_accuracy": a, "challenger_accuracy": b, "delta_pp": b - a})
    return pd.DataFrame(rows)


def _pooled_accuracy(actual: pd.Series, prediction: pd.Series) -> float:
    total = float(np.sum(actual))
    if total <= 0:
        return math.nan
    wape = float(np.sum(np.abs(np.asarray(prediction, dtype=float) - np.asarray(actual, dtype=float))) / total)
    return 100.0 * max(0.0, 1.0 - wape)


def _volatility(values: np.ndarray) -> float:
    diffs = np.diff(np.log1p(np.asarray(values, dtype=float)))
    diffs = diffs[np.isfinite(diffs)]
    return float(np.std(diffs, ddof=1)) if diffs.size >= 3 else math.nan


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return math.nan
    a, b = a[ok], b[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator and math.isfinite(denominator) and denominator > 0 else math.nan


def _safe_log(value: float) -> float:
    return float(math.log(value)) if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else math.nan


def _circular_slot_delta(slot: int, reference: int) -> int:
    delta = (slot - reference) % 96
    return delta - 96 if delta > 48 else delta


def _slot_label(slot: int) -> str:
    return f"{slot // 4:02d}:{(slot % 4) * 15:02d}"


def _fmt(value: Any) -> str:
    return f"{float(value):.2f}" if isinstance(value, (int, float)) and math.isfinite(float(value)) else "n/a"


def _iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _iso_local(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).tz_convert(BOGOTA_TZ).isoformat()
