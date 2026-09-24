from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from candidate_b_shadow import SupabaseShadowStore
from model_improvement_candidates import (
    BOGOTA_TZ,
    HORIZONS,
    MIN_HISTORY_DAYS,
    SHORT_LAGS,
    ModelResult,
    build_lookup,
    build_profile_lookup,
    cached_samples,
    candidate_b_predictions,
    load_observations,
    load_observations_from_frame,
    official_metrics,
)


RANDOM_STATE = 17
WEAK_STATIONS = ("07105", "07107", "07111", "06111")
VALIDATION_DAYS = 7
HARD_HOLDOUT_HOURS = 24
TRAIN_LOOKBACK_DAYS = 28
REPORT_DIR = ROOT / "reports" / "candidate_c"
TABLE_DIR = REPORT_DIR / "tables"

C_FEATURE_SCHEMA = [
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
    "lag15_minus_lag30",
    "lag30_minus_lag60",
    "recent_trend",
    "recent_profile_ratio",
    "rolling_mean_1h",
    "rolling_median_1h",
    "rolling_std_1h",
    "rolling_mean_2h",
    "rolling_median_2h",
    "rolling_std_2h",
    "rolling_mean_4h",
    "rolling_median_4h",
    "rolling_std_4h",
    "same_slot_previous_day",
    "same_slot_previous_week",
    "station_horizon_code",
    "hour_station_code",
]

HGB_PARAM_GRID = [
    {
        "loss": "absolute_error",
        "learning_rate": 0.06,
        "max_iter": 120,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.05,
        "random_state": RANDOM_STATE,
    },
    {
        "loss": "absolute_error",
        "learning_rate": 0.05,
        "max_iter": 140,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 0.10,
        "random_state": RANDOM_STATE,
    },
    {
        "loss": "absolute_error",
        "learning_rate": 0.08,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 35,
        "l2_regularization": 0.10,
        "random_state": RANDOM_STATE,
    },
]

TREE_PARAM_GRID = {
    "extra_trees": {
        "n_estimators": 80,
        "min_samples_leaf": 3,
        "max_features": 0.85,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "random_forest": {
        "n_estimators": 60,
        "min_samples_leaf": 4,
        "max_features": 0.85,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}


@dataclass(frozen=True)
class EvaluationWindow:
    walk_forward_cutoffs: list[pd.Timestamp]
    hard_holdout_cutoffs: list[pd.Timestamp]


@dataclass(frozen=True)
class ExperimentResult:
    result: ModelResult
    estimator: str
    params: dict[str, Any]
    feature_schema: list[str]
    hard_holdout: dict[str, Any] | None = None


def load_observations_from_supabase() -> pd.DataFrame:
    with SupabaseShadowStore() as store:
        frame = store.observations_until("2100-01-01T00:00:00Z")
    return load_observations_from_frame(frame)


def experiment_cutoffs(observations: pd.DataFrame) -> EvaluationWindow:
    times = sorted(observations["observed_at"].drop_duplicates())
    latest_cutoff = times[-1] - pd.Timedelta(minutes=max(HORIZONS))
    hard_start = latest_cutoff - pd.Timedelta(hours=HARD_HOLDOUT_HOURS)
    walk_start = hard_start - pd.Timedelta(days=VALIDATION_DAYS)
    min_cutoff = observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
    walk_forward = [ts for ts in times if ts.minute == 0 and max(walk_start, min_cutoff) <= ts < hard_start]
    hard_holdout = [ts for ts in times if ts.minute == 0 and hard_start <= ts <= latest_cutoff]
    if not walk_forward or not hard_holdout:
        raise RuntimeError("not enough temporal history for walk-forward and hard holdout")
    return EvaluationWindow(walk_forward_cutoffs=walk_forward, hard_holdout_cutoffs=hard_holdout)


def fold_starts(cutoffs: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
    return sorted({ts.floor("D") for ts in cutoffs})


def build_candidate_c_samples(observations: pd.DataFrame, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    if not cutoffs:
        return pd.DataFrame()
    base = observations.sort_values(["station_id", "observed_at"]).copy()
    base = base.rename(columns={"observed_at": "target_at", "demand": "actual"})
    by_station = base.groupby("station_id", sort=False)
    for lag in SHORT_LAGS:
        base[f"lag_{lag}"] = by_station["actual"].shift(lag // 15)
    base["lag_96"] = by_station["actual"].shift(96)
    base["lag_weekly"] = by_station["actual"].shift(7 * 96)
    by_slot = base.groupby(["station_id", "quarter_hour_block"], sort=False)["actual"]
    base["profile_mean"] = by_slot.transform(lambda values: values.shift(1).expanding().mean())
    base["profile_median"] = by_slot.transform(lambda values: values.shift(1).expanding().median())
    base = base.rename(
        columns={
            "hour": "target_hour",
            "minute": "target_minute",
            "day_of_week": "target_day_of_week",
            "is_weekend": "target_is_weekend",
            "quarter_hour_block": "target_quarter_hour_block",
        }
    )

    cutoff_index = pd.DatetimeIndex(cutoffs)
    samples: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        part = base.copy()
        horizon_steps = horizon // 15
        part["horizon_minutes"] = horizon
        part["data_cutoff"] = part["target_at"] - pd.Timedelta(minutes=horizon)
        by_part_station = part.groupby("station_id", sort=False)["actual"]
        for window, suffix in [(4, "1h"), (8, "2h"), (16, "4h")]:
            part[f"rolling_mean_{suffix}"] = by_part_station.transform(
                lambda values, steps=horizon_steps, size=window: values.rolling(size, min_periods=1).mean().shift(steps)
            )
            part[f"rolling_median_{suffix}"] = by_part_station.transform(
                lambda values, steps=horizon_steps, size=window: values.rolling(size, min_periods=1).median().shift(steps)
            )
            part[f"rolling_std_{suffix}"] = by_part_station.transform(
                lambda values, steps=horizon_steps, size=window: values.rolling(size, min_periods=2).std().shift(steps)
            )
        part = part[part["data_cutoff"].isin(cutoff_index)].copy()
        for lag in SHORT_LAGS:
            if lag < horizon:
                part[f"lag_{lag}"] = np.nan
        lag_columns = [f"lag_{lag}" for lag in SHORT_LAGS]
        part["recent_mean"] = part[lag_columns].mean(axis=1, skipna=True)
        part["recent_median"] = part[lag_columns].median(axis=1, skipna=True)
        part["lag15_minus_lag30"] = part["lag_15"] - part["lag_30"]
        part["lag30_minus_lag60"] = part["lag_30"] - part["lag_60"]
        newest_available = part["lag_15"].fillna(part["lag_30"]).fillna(part["lag_45"]).fillna(part["lag_60"])
        oldest_available = part["lag_60"].fillna(part["lag_45"]).fillna(part["lag_30"]).fillna(part["lag_15"])
        part["recent_trend"] = newest_available - oldest_available
        part["recent_profile_ratio"] = np.where(part["profile_mean"] > 0, part["recent_mean"] / part["profile_mean"], np.nan)
        part["same_slot_previous_day"] = part["lag_96"]
        part["same_slot_previous_week"] = part["lag_weekly"]
        samples.append(part)
    result = pd.concat(samples, ignore_index=True)
    columns = [
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
        "lag15_minus_lag30",
        "lag30_minus_lag60",
        "recent_trend",
        "recent_profile_ratio",
        "rolling_mean_1h",
        "rolling_median_1h",
        "rolling_std_1h",
        "rolling_mean_2h",
        "rolling_median_2h",
        "rolling_std_2h",
        "rolling_mean_4h",
        "rolling_median_4h",
        "rolling_std_4h",
        "same_slot_previous_day",
        "same_slot_previous_week",
    ]
    return result[columns].sort_values(["data_cutoff", "station_id", "horizon_minutes"]).reset_index(drop=True)


def leakage_checks(samples: pd.DataFrame) -> dict[str, Any]:
    violations = samples[samples["target_at"] - pd.to_timedelta(samples["horizon_minutes"], unit="m") != samples["data_cutoff"]]
    return {
        "rows": int(len(samples)),
        "target_cutoff_consistent": bool(violations.empty),
        "max_data_cutoff": samples["data_cutoff"].max().isoformat().replace("+00:00", "Z"),
        "max_target_at": samples["target_at"].max().isoformat().replace("+00:00", "Z"),
        "feature_schema": C_FEATURE_SCHEMA,
    }


def candidate_c_matrix(samples: pd.DataFrame, codes: dict[str, int]) -> pd.DataFrame:
    frame = samples.copy()
    frame["station_code"] = frame["station_id"].map(codes).astype(float)
    frame["station_horizon_code"] = frame["station_code"] * 100.0 + frame["horizon_minutes"].astype(float)
    frame["hour_station_code"] = frame["station_code"] * 24.0 + frame["target_hour"].astype(float)
    if frame["station_code"].isna().any():
        missing = sorted(frame.loc[frame["station_code"].isna(), "station_id"].unique())
        raise RuntimeError(f"missing station codes: {missing}")
    return frame[C_FEATURE_SCHEMA].astype(float)


def candidate_c_global(
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    *,
    params: dict[str, Any],
    sample_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame],
    name: str = "candidate C global",
) -> ModelResult:
    start = time.perf_counter()
    codes = {station_id: idx for idx, station_id in enumerate(sorted(observations["station_id"].unique()))}
    predictions: list[pd.DataFrame] = []
    for fold_start in fold_starts(cutoffs):
        fold_cutoffs = [ts for ts in cutoffs if fold_start <= ts < fold_start + pd.Timedelta(days=1)]
        train_samples = training_samples_for_fold(observations, fold_start, sample_cache)
        fold_samples = cached_c_samples(sample_cache, observations, fold_cutoffs)
        model = make_hgb_model(params)
        model.fit(candidate_c_matrix(train_samples, codes), train_samples["actual"].to_numpy())
        fold_samples["prediction"] = np.maximum(0.0, model.predict(candidate_c_matrix(fold_samples, codes)))
        predictions.append(fold_samples)
    result = pd.concat(predictions, ignore_index=True)
    return ModelResult(name=name, predictions=result, summary=official_metrics(result), train_seconds=time.perf_counter() - start, complexity="Media: HGB global con features C seguras.")


def candidate_c_by_horizon(
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    *,
    params: dict[str, Any],
    sample_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame],
) -> ModelResult:
    start = time.perf_counter()
    codes = {station_id: idx for idx, station_id in enumerate(sorted(observations["station_id"].unique()))}
    predictions: list[pd.DataFrame] = []
    for fold_start in fold_starts(cutoffs):
        fold_cutoffs = [ts for ts in cutoffs if fold_start <= ts < fold_start + pd.Timedelta(days=1)]
        train_samples = training_samples_for_fold(observations, fold_start, sample_cache)
        fold_samples = cached_c_samples(sample_cache, observations, fold_cutoffs)
        fold_samples["prediction"] = np.nan
        for horizon in HORIZONS:
            train_part = train_samples[train_samples["horizon_minutes"] == horizon]
            fold_mask = fold_samples["horizon_minutes"] == horizon
            model = make_hgb_model(params)
            model.fit(candidate_c_matrix(train_part, codes), train_part["actual"].to_numpy())
            fold_samples.loc[fold_mask, "prediction"] = np.maximum(0.0, model.predict(candidate_c_matrix(fold_samples.loc[fold_mask], codes)))
        predictions.append(fold_samples)
    result = pd.concat(predictions, ignore_index=True)
    return ModelResult(name="candidate C por horizonte", predictions=result, summary=official_metrics(result), train_seconds=time.perf_counter() - start, complexity="Media-alta: un HGB por horizonte, con foco operacional en +30/+45.")


def candidate_c_station_hybrid(
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    *,
    global_params: dict[str, Any],
    station_estimator: str,
    sample_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame],
) -> ModelResult:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    start = time.perf_counter()
    codes = {station_id: idx for idx, station_id in enumerate(sorted(observations["station_id"].unique()))}
    predictions: list[pd.DataFrame] = []
    for fold_start in fold_starts(cutoffs):
        fold_cutoffs = [ts for ts in cutoffs if fold_start <= ts < fold_start + pd.Timedelta(days=1)]
        train_samples = training_samples_for_fold(observations, fold_start, sample_cache)
        fold_samples = cached_c_samples(sample_cache, observations, fold_cutoffs)
        global_model = make_hgb_model(global_params)
        global_model.fit(candidate_c_matrix(train_samples, codes), train_samples["actual"].to_numpy())
        fold_samples["prediction"] = np.maximum(0.0, global_model.predict(candidate_c_matrix(fold_samples, codes)))
        for station_id in WEAK_STATIONS:
            station_train = train_samples[train_samples["station_id"] == station_id]
            station_mask = fold_samples["station_id"] == station_id
            if len(station_train) < 96:
                continue
            if station_estimator == "hist_gradient_boosting":
                station_model = make_hgb_model(global_params)
            elif station_estimator == "extra_trees":
                station_model = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(**TREE_PARAM_GRID["extra_trees"]))
            elif station_estimator == "random_forest":
                station_model = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(**TREE_PARAM_GRID["random_forest"]))
            else:
                raise ValueError(station_estimator)
            station_model.fit(candidate_c_matrix(station_train, codes), station_train["actual"].to_numpy())
            fold_samples.loc[station_mask, "prediction"] = np.maximum(0.0, station_model.predict(candidate_c_matrix(fold_samples.loc[station_mask], codes)))
        predictions.append(fold_samples)
    result = pd.concat(predictions, ignore_index=True)
    return ModelResult(
        name="candidate C hibrido por estacion",
        predictions=result,
        summary=official_metrics(result),
        train_seconds=time.perf_counter() - start,
        complexity=f"Alta: HGB global + modelos {station_estimator} para {', '.join(WEAK_STATIONS)}.",
    )


def make_hgb_model(params: dict[str, Any]) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), HistGradientBoostingRegressor(**params))


def cached_c_samples(
    cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame],
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
) -> pd.DataFrame:
    if not cutoffs:
        return pd.DataFrame()
    key = (cutoffs[0], cutoffs[-1], len(cutoffs))
    if key not in cache:
        cache[key] = build_candidate_c_samples(observations, cutoffs)
    return cache[key].copy()


def training_samples_for_fold(
    observations: pd.DataFrame,
    fold_start: pd.Timestamp,
    sample_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame],
) -> pd.DataFrame:
    train_start = max(observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS), fold_start - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS))
    train_cutoffs = [
        ts
        for ts in sorted(observations["observed_at"].drop_duplicates())
        if ts.minute == 0 and train_start <= ts < fold_start - pd.Timedelta(minutes=max(HORIZONS))
    ]
    return cached_c_samples(sample_cache, observations, train_cutoffs).dropna(subset=["profile_mean"])


def select_best_hgb_config(
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    sample_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    best_params = HGB_PARAM_GRID[0]
    best_score = -math.inf
    for idx, params in enumerate(HGB_PARAM_GRID, start=1):
        print(f"Running Candidate C global HGB trial {idx}/{len(HGB_PARAM_GRID)}...", flush=True)
        result = candidate_c_global(observations, cutoffs, params=params, sample_cache=sample_cache, name=f"candidate C global trial {idx}")
        score = float(result.summary["mean_station_accuracy"])
        trials.append({"family": "candidate C global", "trial": idx, "accuracy": score, "params": params})
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, trials


def evaluate_hard_holdout(
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    result_builder: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    result = result_builder(observations, cutoffs, **kwargs)
    return {
        "mean_station_accuracy": float(result.summary["mean_station_accuracy"]),
        "station_metrics": result.summary["station_metrics"],
        "horizon_metrics": result.summary["horizon_metrics"],
        "worst_station": result.summary["worst_station"],
        "worst_horizon": result.summary["worst_horizon"],
        "train_seconds": result.train_seconds,
    }


def comparison_frame(results: list[ExperimentResult], baseline: ModelResult) -> pd.DataFrame:
    baseline_station = baseline.summary["station_metrics"][["station_id", "station_accuracy"]].rename(columns={"station_accuracy": "candidate_b_accuracy"})
    rows: list[dict[str, Any]] = []
    base_accuracy = float(baseline.summary["mean_station_accuracy"])
    for experiment in results:
        result = experiment.result
        station = result.summary["station_metrics"][["station_id", "station_accuracy"]].rename(columns={"station_accuracy": "model_accuracy"})
        joined = baseline_station.merge(station, on="station_id", how="inner")
        rows.append(
            {
                "model": result.name,
                "walk_forward_accuracy": float(result.summary["mean_station_accuracy"]),
                "improvement_pp_vs_candidate_b": float(result.summary["mean_station_accuracy"]) - base_accuracy,
                "stations_improve": int((joined["model_accuracy"] > joined["candidate_b_accuracy"] + 1e-9).sum()),
                "stations_worsen": int((joined["model_accuracy"] < joined["candidate_b_accuracy"] - 1e-9).sum()),
                "worst_station": result.summary["worst_station"]["station_id"],
                "worst_station_accuracy": result.summary["worst_station"]["station_accuracy"],
                "worst_horizon": int(result.summary["worst_horizon"]["horizon_minutes"]),
                "worst_horizon_accuracy": result.summary["worst_horizon"]["horizon_accuracy"],
                "estimator": experiment.estimator,
                "train_seconds": result.train_seconds,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    observations: pd.DataFrame,
    window: EvaluationWindow,
    baseline: ModelResult,
    hard_baseline: dict[str, Any],
    experiments: list[ExperimentResult],
    trial_rows: list[dict[str, Any]],
    leakage: dict[str, Any],
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = [ExperimentResult(baseline, "hist_gradient_boosting", {}, C_FEATURE_SCHEMA), *experiments]
    comparison = comparison_frame(experiments, baseline)
    baseline_row = pd.DataFrame(
        [
            {
                "model": "Candidate B actual",
                "walk_forward_accuracy": float(baseline.summary["mean_station_accuracy"]),
                "improvement_pp_vs_candidate_b": 0.0,
                "stations_improve": 0,
                "stations_worsen": 0,
                "worst_station": baseline.summary["worst_station"]["station_id"],
                "worst_station_accuracy": baseline.summary["worst_station"]["station_accuracy"],
                "worst_horizon": int(baseline.summary["worst_horizon"]["horizon_minutes"]),
                "worst_horizon_accuracy": baseline.summary["worst_horizon"]["horizon_accuracy"],
                "estimator": "hist_gradient_boosting",
                "train_seconds": baseline.train_seconds,
            }
        ]
    )
    comparison = pd.concat([baseline_row, comparison], ignore_index=True)
    comparison.to_csv(TABLE_DIR / "comparison_walk_forward.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(TABLE_DIR / "model_trials.csv", index=False)
    for experiment in all_results:
        slug = ascii_slug(experiment.result.name)
        experiment.result.summary["station_metrics"].to_csv(TABLE_DIR / f"{slug}_station_metrics.csv", index=False)
        experiment.result.summary["horizon_metrics"].to_csv(TABLE_DIR / f"{slug}_horizon_metrics.csv", index=False)
        experiment.result.summary["station_horizon_metrics"].to_csv(TABLE_DIR / f"{slug}_station_horizon_metrics.csv", index=False)
    hard_rows = [
        {
            "model": "Candidate B actual",
            "hard_holdout_accuracy": hard_baseline["mean_station_accuracy"],
            "worst_station": hard_baseline["worst_station"]["station_id"],
            "worst_station_accuracy": hard_baseline["worst_station"]["station_accuracy"],
            "worst_horizon": int(hard_baseline["worst_horizon"]["horizon_minutes"]),
            "worst_horizon_accuracy": hard_baseline["worst_horizon"]["horizon_accuracy"],
        }
    ]
    for experiment in experiments:
        if experiment.hard_holdout is None:
            continue
        hard_rows.append(
            {
                "model": experiment.result.name,
                "hard_holdout_accuracy": experiment.hard_holdout["mean_station_accuracy"],
                "worst_station": experiment.hard_holdout["worst_station"]["station_id"],
                "worst_station_accuracy": experiment.hard_holdout["worst_station"]["station_accuracy"],
                "worst_horizon": int(experiment.hard_holdout["worst_horizon"]["horizon_minutes"]),
                "worst_horizon_accuracy": experiment.hard_holdout["worst_horizon"]["horizon_accuracy"],
            }
        )
    hard_comparison = pd.DataFrame(hard_rows)
    hard_comparison.to_csv(TABLE_DIR / "comparison_hard_holdout.csv", index=False)
    payload = {
        "candidate_version": "candidate_c_v1_experiment",
        "data_min": observations["observed_at"].min().isoformat().replace("+00:00", "Z"),
        "data_max": observations["observed_at"].max().isoformat().replace("+00:00", "Z"),
        "stations": int(observations["station_id"].nunique()),
        "walk_forward_start": min(window.walk_forward_cutoffs).isoformat().replace("+00:00", "Z"),
        "walk_forward_end": max(window.walk_forward_cutoffs).isoformat().replace("+00:00", "Z"),
        "hard_holdout_start": min(window.hard_holdout_cutoffs).isoformat().replace("+00:00", "Z"),
        "hard_holdout_end": max(window.hard_holdout_cutoffs).isoformat().replace("+00:00", "Z"),
        "leakage_checks": leakage,
        "comparison_walk_forward": comparison.to_dict(orient="records"),
        "comparison_hard_holdout": hard_comparison.to_dict(orient="records"),
        "trials": trial_rows,
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_markdown(payload, comparison, hard_comparison, baseline, experiments) + "\n")


def render_markdown(
    payload: dict[str, Any],
    comparison: pd.DataFrame,
    hard_comparison: pd.DataFrame,
    baseline: ModelResult,
    experiments: list[ExperimentResult],
) -> str:
    best = comparison.sort_values("walk_forward_accuracy", ascending=False).iloc[0]
    hard_best = hard_comparison.sort_values("hard_holdout_accuracy", ascending=False).iloc[0] if not hard_comparison.empty else None
    return "\n".join(
        [
            "# Candidate C v1 experiment",
            "",
            "## Scope",
            "",
            "- Experimental only; production champion, submissions, scheduler, and workflows were not changed.",
            "- Candidate C starts from Candidate B and adds only features available at or before each `data_cutoff`.",
            "- Hard holdout is the latest temporal block and is not used for model selection.",
            "",
            "## Windows",
            "",
            f"- Observations: {payload['data_min']} to {payload['data_max']}.",
            f"- Walk-forward: {payload['walk_forward_start']} to {payload['walk_forward_end']}.",
            f"- Hard holdout: {payload['hard_holdout_start']} to {payload['hard_holdout_end']}.",
            "",
            "## Walk-forward comparison",
            "",
            _markdown_table(comparison),
            "",
            "## Hard holdout comparison",
            "",
            _markdown_table(hard_comparison),
            "",
            "## Candidate B diagnosis in this window",
            "",
            _diagnosis_lines(baseline),
            "",
            "## Best experimental result",
            "",
            f"- Walk-forward best: {best['model']} at {best['walk_forward_accuracy']:.3f}%.",
            f"- Hard-holdout best: {hard_best['model']} at {hard_best['hard_holdout_accuracy']:.3f}%." if hard_best is not None else "- Hard-holdout best: unavailable.",
            "",
            "## Leakage controls",
            "",
            f"- Target/cutoff consistency: {payload['leakage_checks']['target_cutoff_consistent']}.",
            "- Lags and rolling windows are shifted by the forecast horizon, so they end at `data_cutoff`.",
            "- Profile features use prior same-slot observations only (`shift(1)`), never the target row.",
            "- Station mapping is deterministic from station ids in training/static observations.",
            "",
            "## Recommendation",
            "",
            _recommendation(comparison, hard_comparison),
        ]
    )


def _diagnosis_lines(result: ModelResult) -> str:
    worst_station = result.summary["worst_station"]
    worst_horizon = result.summary["worst_horizon"]
    return "\n".join(
        [
            f"- Mean station accuracy: {result.summary['mean_station_accuracy']:.3f}%.",
            f"- Worst station: {worst_station['station_id']} ({worst_station['station_accuracy']:.3f}%).",
            f"- Worst horizon: +{int(worst_horizon['horizon_minutes'])} ({worst_horizon['horizon_accuracy']:.3f}%).",
        ]
    )


def _recommendation(comparison: pd.DataFrame, hard_comparison: pd.DataFrame) -> str:
    baseline = comparison[comparison["model"] == "Candidate B actual"].iloc[0]
    best = comparison.sort_values("walk_forward_accuracy", ascending=False).iloc[0]
    if best["model"] == "Candidate B actual" or float(best["walk_forward_accuracy"]) < 87.0:
        return "Candidate C no esta listo: no supera claramente el umbral 87-88% en validacion temporal."
    if hard_comparison.empty:
        return "Candidate C requiere hard holdout antes de avanzar."
    hard_best = hard_comparison.sort_values("hard_holdout_accuracy", ascending=False).iloc[0]
    hard_baseline = hard_comparison[hard_comparison["model"] == "Candidate B actual"]
    if not hard_baseline.empty and float(hard_best["hard_holdout_accuracy"]) <= float(hard_baseline.iloc[0]["hard_holdout_accuracy"]):
        return "Candidate C no esta listo: no mejora el hard holdout frente a Candidate B."
    if int(best["stations_worsen"]) >= 4:
        return "Candidate C requiere mas experimentos: mejora agregada, pero degrada demasiadas estaciones."
    return "Candidate C es prometedor para shadow mode, pero no debe promoverse sin observacion en ciclos reales."


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def ascii_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return normalized.lower().replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Candidate C experiments without touching production.")
    parser.add_argument("--source", choices=["local", "supabase"], default=os.getenv("CANDIDATE_C_SOURCE", "local"))
    args = parser.parse_args()

    observations = load_observations_from_supabase() if args.source == "supabase" else load_observations()
    window = experiment_cutoffs(observations)
    lookup = build_lookup(observations)
    profiles = build_profile_lookup(observations)
    b_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame] = {}
    c_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame] = {}
    print(f"Walk-forward cutoffs: {len(window.walk_forward_cutoffs)}; hard holdout cutoffs: {len(window.hard_holdout_cutoffs)}", flush=True)
    cached_samples(b_cache, observations, window.walk_forward_cutoffs, lookup=lookup, profiles=profiles)
    print("Running Candidate B baseline...", flush=True)
    baseline = candidate_b_predictions(observations, window.walk_forward_cutoffs, lookup=lookup, profiles=profiles, sample_cache=b_cache)
    baseline = ModelResult("Candidate B actual", baseline.predictions, baseline.summary, baseline.train_seconds, baseline.complexity)

    best_params, trials = select_best_hgb_config(observations, window.walk_forward_cutoffs, c_cache)
    print("Running selected Candidate C global...", flush=True)
    global_result = candidate_c_global(observations, window.walk_forward_cutoffs, params=best_params, sample_cache=c_cache)
    print("Running Candidate C by horizon...", flush=True)
    horizon_result = candidate_c_by_horizon(observations, window.walk_forward_cutoffs, params=best_params, sample_cache=c_cache)

    hybrid_trials: list[ExperimentResult] = []
    for estimator in ("hist_gradient_boosting", "extra_trees", "random_forest"):
        print(f"Running Candidate C station hybrid with {estimator}...", flush=True)
        result = candidate_c_station_hybrid(observations, window.walk_forward_cutoffs, global_params=best_params, station_estimator=estimator, sample_cache=c_cache)
        trials.append({"family": "candidate C hibrido por estacion", "trial": estimator, "accuracy": result.summary["mean_station_accuracy"], "params": {"station_estimator": estimator, "global_params": best_params}})
        hybrid_trials.append(ExperimentResult(result, estimator, {"station_estimator": estimator, "global_params": best_params}, C_FEATURE_SCHEMA))
    best_hybrid = max(hybrid_trials, key=lambda item: float(item.result.summary["mean_station_accuracy"]))

    experiments = [
        ExperimentResult(global_result, "hist_gradient_boosting", best_params, C_FEATURE_SCHEMA),
        ExperimentResult(horizon_result, "hist_gradient_boosting_by_horizon", best_params, C_FEATURE_SCHEMA),
        best_hybrid,
    ]

    print("Running hard holdout...", flush=True)
    hard_b = candidate_b_predictions(observations, window.hard_holdout_cutoffs, lookup=lookup, profiles=profiles, sample_cache=b_cache)
    hard_b = ModelResult("Candidate B actual", hard_b.predictions, hard_b.summary, hard_b.train_seconds, hard_b.complexity)
    hard_global = evaluate_hard_holdout(observations, window.hard_holdout_cutoffs, candidate_c_global, params=best_params, sample_cache=c_cache)
    hard_horizon = evaluate_hard_holdout(observations, window.hard_holdout_cutoffs, candidate_c_by_horizon, params=best_params, sample_cache=c_cache)
    hard_hybrid = evaluate_hard_holdout(
        observations,
        window.hard_holdout_cutoffs,
        candidate_c_station_hybrid,
        global_params=best_params,
        station_estimator=best_hybrid.estimator,
        sample_cache=c_cache,
    )
    hard_baseline = {
        "mean_station_accuracy": float(hard_b.summary["mean_station_accuracy"]),
        "station_metrics": hard_b.summary["station_metrics"],
        "horizon_metrics": hard_b.summary["horizon_metrics"],
        "worst_station": hard_b.summary["worst_station"],
        "worst_horizon": hard_b.summary["worst_horizon"],
        "train_seconds": hard_b.train_seconds,
    }
    experiments_with_hard = [
        ExperimentResult(hard_b, "hist_gradient_boosting", {}, C_FEATURE_SCHEMA, hard_baseline),
        ExperimentResult(global_result, "hist_gradient_boosting", best_params, C_FEATURE_SCHEMA, hard_global),
        ExperimentResult(horizon_result, "hist_gradient_boosting_by_horizon", best_params, C_FEATURE_SCHEMA, hard_horizon),
        ExperimentResult(best_hybrid.result, best_hybrid.estimator, best_hybrid.params, C_FEATURE_SCHEMA, hard_hybrid),
    ]
    leakage = leakage_checks(build_candidate_c_samples(observations, window.walk_forward_cutoffs + window.hard_holdout_cutoffs))
    write_outputs(observations, window, baseline, hard_baseline, experiments_with_hard[1:], trials, leakage)
    comparison = pd.read_csv(TABLE_DIR / "comparison_walk_forward.csv")
    hard = pd.read_csv(TABLE_DIR / "comparison_hard_holdout.csv")
    print(comparison.to_string(index=False))
    print(hard.to_string(index=False))
    print(f"Report written to {REPORT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
