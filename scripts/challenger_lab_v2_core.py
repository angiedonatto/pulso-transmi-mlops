"""Shared infrastructure for Challenger Lab v2.

Reuses proven infrastructure instead of re-deriving it:
- Temporal windows (`experiment_cutoffs`) and full metrics (`official_metrics_full`,
  `SPECIAL_STATIONS`, `SPECIAL_HORIZONS`) come from Candidate D1, whose windows were
  already audited and are being kept fixed here (no re-definition after seeing results).
- Base sample construction constants (`HORIZONS`, `SHORT_LAGS`, `MIN_HISTORY_DAYS`,
  `TRAIN_LOOKBACK_DAYS`) and Candidate B's own model/matrix come from
  `model_improvement_candidates`, so the "Global baseline" experiment calls the exact
  same function the champion pipeline and Candidate D1 use instead of re-implementing it.

All NEW feature engineering (`build_extended_samples`) is additive and self-contained: it
never mutates shared modules, so nothing here can break Candidate B, Candidate D1, the
drift monitor, or their tests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from model_improvement_candidates import (  # noqa: E402
    BOGOTA_TZ,
    HORIZONS,
    MIN_HISTORY_DAYS,
    SHORT_LAGS,
    TRAIN_LOOKBACK_DAYS,
    ModelResult,
    ProfileSeries,
    build_lookup,
    build_profile_lookup,
    candidate_b_predictions,
    fold_starts,
    load_observations,
    load_observations_from_frame,
    station_code_map,
)
from candidate_d1_residual_calibration import (  # noqa: E402
    EvaluationWindow,
    SPECIAL_HORIZONS,
    SPECIAL_STATIONS,
    ascii_slug,
    experiment_cutoffs,
    make_candidate_b_model,
    official_metrics_full,
)

RANDOM_STATE = 17
DAILY_LAG_MINUTES = 24 * 60
WEEKLY_LAG_MINUTES = 7 * 24 * 60
EXTRA_LAGS = (90, 120)
ROLLING_MEAN_WINDOWS = ((4, "1h"), (8, "2h"), (12, "3h"))
ROLLING_STD_WINDOWS = ((4, "1h"), (12, "3h"))
ROLLING_STAT_WINDOW = (12, "3h")
DATA_DIR = ROOT / "data" / "raw"

# Base columns that must exactly match `model_improvement_candidates.make_samples()`.
BASE_SAMPLE_COLUMNS = [
    "station_id",
    "data_cutoff",
    "target_at",
    "horizon_minutes",
    "actual",
    "target_hour",
    "target_minute",
    "target_day_of_week",
    "target_is_weekend",
    "target_quarter_hour_block",
    "lag_15",
    "lag_30",
    "lag_45",
    "lag_60",
    "lag_96",
    "lag_weekly",
    "recent_mean",
    "recent_median",
    "profile_mean",
    "profile_median",
]

NEW_FEATURE_COLUMNS = [
    "lag_90",
    "lag_120",
    "rolling_mean_1h",
    "rolling_mean_2h",
    "rolling_mean_3h",
    "rolling_std_1h",
    "rolling_std_3h",
    "rolling_median_3h",
    "rolling_min_3h",
    "rolling_max_3h",
    "demand_delta_15",
    "demand_delta_30",
    "recent_slope",
    "recent_profile_ratio",
    "profile_dow_mean",
    "profile_dow_median",
]

EXTENDED_SAMPLE_COLUMNS = BASE_SAMPLE_COLUMNS + NEW_FEATURE_COLUMNS

# Candidate B v2's exact feature schema (for documentation; the real baseline experiment
# calls `candidate_b_predictions` directly rather than re-selecting these columns, to
# guarantee bit-for-bit reproduction of the champion pipeline's numbers).
BASELINE_FEATURE_SCHEMA = [
    "station_code",
    "horizon_minutes",
    "target_hour",
    "target_minute",
    "target_day_of_week",
    "target_is_weekend",
    "target_quarter_hour_block",
    "lag_15",
    "lag_30",
    "lag_45",
    "lag_60",
    "lag_96",
    "lag_weekly",
    "recent_mean",
    "recent_median",
    "profile_mean",
    "profile_median",
]

FEATURE_SET_PLUS_LAGS = BASELINE_FEATURE_SCHEMA + ["lag_90", "lag_120"]
FEATURE_SET_PLUS_ROLLING = BASELINE_FEATURE_SCHEMA + [
    "rolling_mean_1h",
    "rolling_mean_2h",
    "rolling_mean_3h",
    "rolling_std_1h",
    "rolling_std_3h",
    "rolling_median_3h",
    "rolling_min_3h",
    "rolling_max_3h",
]
FEATURE_SET_PLUS_TREND = BASELINE_FEATURE_SCHEMA + [
    "demand_delta_15",
    "demand_delta_30",
    "recent_slope",
    "recent_profile_ratio",
]
FEATURE_SET_PLUS_DOW_PROFILE = BASELINE_FEATURE_SCHEMA + ["profile_dow_mean", "profile_dow_median"]
FEATURE_SET_ALL = BASELINE_FEATURE_SCHEMA + NEW_FEATURE_COLUMNS

_TIME_FEATURE_RENAMES = {
    "hour": "target_hour",
    "minute": "target_minute",
    "day_of_week": "target_day_of_week",
    "is_weekend": "target_is_weekend",
    "quarter_hour_block": "target_quarter_hour_block",
}


def build_extended_samples(observations: pd.DataFrame, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    """Production-eligible feature matrix: base Candidate B columns plus new lag,
    rolling, trend and day-of-week profile features.

    Leakage safety, by construction:
    - Every new lag (`lag_90`, `lag_120`) is a `shift()` of the per-station series, so it
      only ever reads observations strictly before `target_at`. Both are >= the maximum
      horizon (60 min), so unlike the short lags they never need masking: at `data_cutoff`
      they are always already in the past.
    - Rolling features are computed *inside* the per-horizon loop as
      `shift(horizon_steps).rolling(window)`, i.e. anchored at `data_cutoff` (target_at -
      horizon), not at `target_at`. This is the critical difference from a naive
      `rolling().shift(horizon_steps)` done once outside the loop: the shift amount must
      track the horizon, because the window has to end at `data_cutoff` for every horizon,
      not at a fixed offset from `target_at`.
    - `profile_mean/median` (station x quarter-hour) and `profile_dow_mean/median`
      (station x day-of-week x quarter-hour) use `shift(1).expanding()` within their group,
      i.e. only strictly-prior same-slot rows. Because each group key recurs at most once
      per day (plain slot) or once per week (dow slot), the nearest same-group row before
      `target_at` is always >= 24h / 7 days before it -- far outside the max 60-minute
      horizon -- so anchoring at `target_at` instead of `data_cutoff` cannot leak here.
    - `recent_mean/median`, `demand_delta_*`, `recent_slope` and `recent_profile_ratio` are
      all built only from the (possibly horizon-masked) short lags and `profile_mean`, so
      they inherit the same safety.
    """
    if not cutoffs:
        return pd.DataFrame(columns=EXTENDED_SAMPLE_COLUMNS)

    base = observations.sort_values(["station_id", "observed_at"]).copy()
    base = base.rename(columns={"observed_at": "target_at", "demand": "actual"})
    by_station = base.groupby("station_id", sort=False)["actual"]
    for lag in SHORT_LAGS:
        base[f"lag_{lag}"] = by_station.shift(lag // 15)
    for lag in EXTRA_LAGS:
        base[f"lag_{lag}"] = by_station.shift(lag // 15)
    base["lag_96"] = by_station.shift(DAILY_LAG_MINUTES // 15)
    base["lag_weekly"] = by_station.shift(WEEKLY_LAG_MINUTES // 15)

    by_slot = base.groupby(["station_id", "quarter_hour_block"], sort=False)["actual"]
    base["profile_mean"] = by_slot.transform(lambda values: values.shift(1).expanding().mean())
    base["profile_median"] = by_slot.transform(lambda values: values.shift(1).expanding().median())

    by_dow_slot = base.groupby(["station_id", "day_of_week", "quarter_hour_block"], sort=False)["actual"]
    base["profile_dow_mean"] = by_dow_slot.transform(lambda values: values.shift(1).expanding().mean())
    base["profile_dow_median"] = by_dow_slot.transform(lambda values: values.shift(1).expanding().median())

    base = base.rename(columns=_TIME_FEATURE_RENAMES)

    cutoff_index = pd.DatetimeIndex(cutoffs)
    parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        part = base.copy()
        shift_steps = horizon // 15
        part["horizon_minutes"] = horizon
        part["data_cutoff"] = part["target_at"] - pd.Timedelta(minutes=horizon)

        by_part_station = part.groupby("station_id", sort=False)["actual"]
        for window, suffix in ROLLING_MEAN_WINDOWS:
            part[f"rolling_mean_{suffix}"] = by_part_station.transform(
                lambda values, steps=shift_steps, size=window: values.shift(steps).rolling(size, min_periods=size).mean()
            )
        for window, suffix in ROLLING_STD_WINDOWS:
            part[f"rolling_std_{suffix}"] = by_part_station.transform(
                lambda values, steps=shift_steps, size=window: values.shift(steps).rolling(size, min_periods=size).std()
            )
        stat_window, stat_suffix = ROLLING_STAT_WINDOW
        part[f"rolling_median_{stat_suffix}"] = by_part_station.transform(
            lambda values, steps=shift_steps, size=stat_window: values.shift(steps).rolling(size, min_periods=size).median()
        )
        part[f"rolling_min_{stat_suffix}"] = by_part_station.transform(
            lambda values, steps=shift_steps, size=stat_window: values.shift(steps).rolling(size, min_periods=size).min()
        )
        part[f"rolling_max_{stat_suffix}"] = by_part_station.transform(
            lambda values, steps=shift_steps, size=stat_window: values.shift(steps).rolling(size, min_periods=size).max()
        )

        part = part[part["data_cutoff"].isin(cutoff_index)].copy()

        for lag in SHORT_LAGS:
            if lag < horizon:
                part[f"lag_{lag}"] = np.nan

        lag_columns = [f"lag_{lag}" for lag in SHORT_LAGS]
        part["recent_mean"] = part[lag_columns].mean(axis=1, skipna=True)
        part["recent_median"] = part[lag_columns].median(axis=1, skipna=True)

        newest_available = part["lag_15"].fillna(part["lag_30"]).fillna(part["lag_45"]).fillna(part["lag_60"])
        oldest_available = part["lag_60"].fillna(part["lag_45"]).fillna(part["lag_30"]).fillna(part["lag_15"])
        part["recent_slope"] = newest_available - oldest_available
        part["demand_delta_15"] = part["lag_15"] - part["lag_30"]
        part["demand_delta_30"] = part["lag_30"] - part["lag_60"]
        part["recent_profile_ratio"] = np.where(part["profile_mean"] > 0, part["recent_mean"] / part["profile_mean"], np.nan)

        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    return result[EXTENDED_SAMPLE_COLUMNS].sort_values(["data_cutoff", "station_id", "horizon_minutes"]).reset_index(drop=True)


def cached_extended_samples(
    cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame],
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
) -> pd.DataFrame:
    if not cutoffs:
        return pd.DataFrame(columns=EXTENDED_SAMPLE_COLUMNS)
    key = (cutoffs[0], cutoffs[-1], len(cutoffs))
    if key not in cache:
        cache[key] = build_extended_samples(observations, cutoffs)
    return cache[key].copy()


def build_matrix(samples: pd.DataFrame, codes: dict[str, int], feature_schema: list[str]) -> pd.DataFrame:
    frame = samples.copy()
    frame["station_code"] = frame["station_id"].map(codes).astype(float)
    if frame["station_code"].isna().any():
        missing = sorted(frame.loc[frame["station_code"].isna(), "station_id"].unique())
        raise RuntimeError(f"missing station codes: {missing}")
    return frame[feature_schema].astype(float)


def training_window_for_fold(observations: pd.DataFrame, fold_start: pd.Timestamp) -> pd.Timestamp:
    return max(
        observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS),
        fold_start - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS),
    )


def train_cutoffs_for_fold(observations: pd.DataFrame, times: list[pd.Timestamp], fold_start: pd.Timestamp) -> list[pd.Timestamp]:
    train_start = training_window_for_fold(observations, fold_start)
    return [ts for ts in times if ts.minute == 0 and train_start <= ts < fold_start - pd.Timedelta(minutes=max(HORIZONS))]


HGB_DEFAULT_PARAMS: dict[str, Any] = {
    "loss": "absolute_error",
    "learning_rate": 0.06,
    "max_iter": 160,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.05,
    "random_state": RANDOM_STATE,
}
EXTRA_TREES_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 80,
    "min_samples_leaf": 3,
    "max_features": 0.85,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
RANDOM_FOREST_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 60,
    "min_samples_leaf": 4,
    "max_features": 0.85,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
GRADIENT_BOOSTING_DEFAULT_PARAMS: dict[str, Any] = {
    "loss": "absolute_error",
    "learning_rate": 0.06,
    "n_estimators": 150,
    "max_depth": 3,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}
RIDGE_DEFAULT_PARAMS: dict[str, Any] = {"alpha": 5.0, "random_state": RANDOM_STATE}
ELASTIC_NET_DEFAULT_PARAMS: dict[str, Any] = {
    "alpha": 0.1,
    "l1_ratio": 0.5,
    "random_state": RANDOM_STATE,
    "max_iter": 5000,
}

MODEL_FAMILIES = (
    "hist_gradient_boosting",
    "extra_trees",
    "random_forest",
    "gradient_boosting",
    "ridge",
    "elastic_net",
)


def make_model(family: str, **overrides: Any) -> Any:
    """Model factory. Every family is wrapped in the same median-imputer pipeline
    (`keep_empty_features=True`) so families are compared on equal footing -- the
    difference measured is the estimator, not who happens to tolerate NaN natively.
    This matches the convention already used by Candidate C and Candidate D1's HGB/tree
    residual models."""
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if family == "hist_gradient_boosting":
        return make_pipeline(imputer, HistGradientBoostingRegressor(**{**HGB_DEFAULT_PARAMS, **overrides}))
    if family == "extra_trees":
        return make_pipeline(imputer, ExtraTreesRegressor(**{**EXTRA_TREES_DEFAULT_PARAMS, **overrides}))
    if family == "random_forest":
        return make_pipeline(imputer, RandomForestRegressor(**{**RANDOM_FOREST_DEFAULT_PARAMS, **overrides}))
    if family == "gradient_boosting":
        return make_pipeline(imputer, GradientBoostingRegressor(**{**GRADIENT_BOOSTING_DEFAULT_PARAMS, **overrides}))
    if family == "ridge":
        return make_pipeline(imputer, StandardScaler(), Ridge(**{**RIDGE_DEFAULT_PARAMS, **overrides}))
    if family == "elastic_net":
        return make_pipeline(imputer, StandardScaler(), ElasticNet(**{**ELASTIC_NET_DEFAULT_PARAMS, **overrides}))
    raise ValueError(f"unsupported model family: {family!r}")


def temporal_fold_audit_generic(observations: pd.DataFrame, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    """Leakage audit: for every daily fold, confirm training data ends before the
    validation window starts. Mirrors Candidate D1's `temporal_fold_audit` but against
    the extended sample builder."""
    times = sorted(observations["observed_at"].drop_duplicates())
    cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame] = {}
    rows = []
    for fold_start in fold_starts(cutoffs):
        fold_cutoffs = [ts for ts in cutoffs if fold_start <= ts < fold_start + pd.Timedelta(days=1)]
        train_cutoffs = train_cutoffs_for_fold(observations, times, fold_start)
        train_samples = cached_extended_samples(cache, observations, train_cutoffs)
        fold_samples = cached_extended_samples(cache, observations, fold_cutoffs)
        if train_samples.empty or fold_samples.empty:
            continue
        rows.append(
            {
                "fold_start": fold_start,
                "training_end": train_samples["target_at"].max(),
                "validation_start": fold_samples["data_cutoff"].min(),
                "validation_end": fold_samples["data_cutoff"].max(),
                "training_rows": int(len(train_samples)),
                "validation_rows": int(len(fold_samples)),
            }
        )
    return pd.DataFrame(rows)


def load_context() -> pd.DataFrame:
    """Local context_observations snapshot (research-only track). Loaded straight from
    the repo's local CSV cache -- same source as `load_observations()` -- so this never
    touches Supabase or production."""
    frame = pd.read_csv(DATA_DIR / "context.csv")
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    return frame.sort_values("observed_at").reset_index(drop=True)
