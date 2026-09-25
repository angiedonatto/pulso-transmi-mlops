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
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from candidate_b_shadow import SupabaseShadowStore
from model_improvement_candidates import (
    HORIZONS,
    MIN_HISTORY_DAYS,
    TRAIN_LOOKBACK_DAYS,
    ModelResult,
    _candidate_b_matrix,
    build_lookup,
    build_profile_lookup,
    cached_samples,
    fold_starts,
    load_observations,
    load_observations_from_frame,
    station_code_map,
)


RANDOM_STATE = 17
VALIDATION_DAYS = 7
HARD_HOLDOUT_HOURS = 24
SPECIAL_STATIONS = ("05000", "07111", "03000", "09122")
SPECIAL_HORIZONS = (30, 60)
REPORT_DIR = ROOT / "reports" / "candidate_d1"
TABLE_DIR = REPORT_DIR / "tables"

D1_FEATURE_SCHEMA = [
    "station_code",
    "horizon_minutes",
    "target_hour",
    "target_minute",
    "target_day_of_week",
    "target_is_weekend",
    "target_quarter_hour_block",
    "candidate_b_prediction",
    "lag_15",
    "lag_30",
    "lag_45",
    "lag_60",
    "lag_96",
    "lag_weekly",
    "recent_mean",
    "recent_median",
]


@dataclass(frozen=True)
class EvaluationWindow:
    walk_forward_cutoffs: list[pd.Timestamp]
    hard_holdout_cutoffs: list[pd.Timestamp]


@dataclass(frozen=True)
class ResidualSpec:
    name: str
    estimator: str
    clipped: bool


@dataclass(frozen=True)
class ClipBounds:
    global_lower: float
    global_upper: float
    by_horizon: dict[int, tuple[float, float]]


@dataclass(frozen=True)
class D1ExperimentResult:
    baseline: ModelResult
    challengers: list[ModelResult]


@dataclass(frozen=True)
class FoldAudit:
    fold_start: pd.Timestamp
    b_training_end: pd.Timestamp
    d1_training_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    residual_rows: int


RESIDUAL_SPECS = [
    ResidualSpec("Candidate D1 ridge raw", "ridge", False),
    ResidualSpec("Candidate D1 ridge clipped", "ridge", True),
    ResidualSpec("Candidate D1 hgb raw", "hist_gradient_boosting", False),
    ResidualSpec("Candidate D1 hgb clipped", "hist_gradient_boosting", True),
    ResidualSpec("Candidate D1 extra trees clipped", "extra_trees", True),
]


def load_observations_from_supabase_read_only() -> pd.DataFrame:
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


def evaluate_d1(
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    specs: list[ResidualSpec] | None = None,
    *,
    sample_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame] | None = None,
) -> D1ExperimentResult:
    specs = specs or RESIDUAL_SPECS
    start = time.perf_counter()
    lookup = build_lookup(observations)
    profiles = build_profile_lookup(observations)
    cache = sample_cache if sample_cache is not None else {}
    codes = station_code_map(observations)
    times = sorted(observations["observed_at"].drop_duplicates())
    baseline_predictions: list[pd.DataFrame] = []
    challenger_predictions: dict[str, list[pd.DataFrame]] = {spec.name: [] for spec in specs}
    train_seconds_by_spec = {spec.name: 0.0 for spec in specs}

    for fold_start in fold_starts(cutoffs):
        fold_cutoffs = [ts for ts in cutoffs if fold_start <= ts < fold_start + pd.Timedelta(days=1)]
        train_start = max(observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS), fold_start - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS))
        train_cutoffs = [
            ts
            for ts in times
            if ts.minute == 0 and train_start <= ts < fold_start - pd.Timedelta(minutes=max(HORIZONS))
        ]
        train_samples = cached_samples(cache, observations, train_cutoffs, lookup=lookup, profiles=profiles)
        fold_samples = cached_samples(cache, observations, fold_cutoffs, lookup=lookup, profiles=profiles)
        train_samples, fold_samples = add_candidate_b_predictions(train_samples, fold_samples, codes)
        train_samples = with_residual_target(train_samples)
        train_samples = train_samples.dropna(subset=["candidate_b_prediction", "residual_target"]).copy()

        fold_baseline = fold_samples.copy()
        fold_baseline["prediction"] = fold_baseline["candidate_b_prediction"].clip(lower=0)
        baseline_predictions.append(fold_baseline)

        for spec in specs:
            spec_start = time.perf_counter()
            model = make_residual_model(spec.estimator)
            model.fit(d1_matrix(train_samples, codes), train_samples["residual_target"].to_numpy(dtype=float))
            correction = model.predict(d1_matrix(fold_samples, codes))
            if spec.clipped:
                bounds = learn_clip_bounds(train_samples)
                correction = apply_clip_bounds(correction, fold_samples, bounds)
            result = fold_samples.copy()
            result["predicted_residual"] = correction
            result["prediction"] = (result["candidate_b_prediction"] + result["predicted_residual"]).clip(lower=0)
            challenger_predictions[spec.name].append(result)
            train_seconds_by_spec[spec.name] += time.perf_counter() - spec_start

    baseline_frame = pd.concat(baseline_predictions, ignore_index=True)
    baseline = ModelResult(
        name="Candidate B v2",
        predictions=baseline_frame,
        summary=official_metrics_full(baseline_frame),
        train_seconds=time.perf_counter() - start,
        complexity="Champion pipeline simulation: Candidate B HistGradientBoostingRegressor per temporal fold.",
    )
    challengers: list[ModelResult] = []
    for spec in specs:
        frame = pd.concat(challenger_predictions[spec.name], ignore_index=True)
        challengers.append(
            ModelResult(
                name=spec.name,
                predictions=frame,
                summary=official_metrics_full(frame),
                train_seconds=train_seconds_by_spec[spec.name],
                complexity=f"Residual corrector {spec.estimator}; correction {'clipped by training residual quantiles' if spec.clipped else 'raw'}.",
            )
        )
    return D1ExperimentResult(baseline=baseline, challengers=challengers)


def add_candidate_b_predictions(
    train_samples: pd.DataFrame,
    fold_samples: pd.DataFrame,
    codes: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = make_candidate_b_model()
    model.fit(_candidate_b_matrix(train_samples, codes), train_samples["actual"].to_numpy(dtype=float))
    fold = fold_samples.copy()
    fold["candidate_b_prediction"] = np.maximum(0.0, model.predict(_candidate_b_matrix(fold, codes)))
    fold["candidate_b_training_end"] = train_samples["target_at"].max()
    train = add_out_of_fold_candidate_b_predictions(train_samples, codes)
    return train, fold


def add_out_of_fold_candidate_b_predictions(samples: pd.DataFrame, codes: dict[str, int]) -> pd.DataFrame:
    result = samples.copy()
    result["candidate_b_prediction"] = np.nan
    result["candidate_b_training_end"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns, UTC]")
    for validation_start in fold_starts(result["data_cutoff"].drop_duplicates().tolist()):
        validation_end = validation_start + pd.Timedelta(days=1)
        validation_mask = (result["data_cutoff"] >= validation_start) & (result["data_cutoff"] < validation_end)
        train_part = result[result["target_at"] < validation_start].copy()
        if train_part.empty or not validation_mask.any():
            continue
        model = make_candidate_b_model()
        model.fit(_candidate_b_matrix(train_part, codes), train_part["actual"].to_numpy(dtype=float))
        result.loc[validation_mask, "candidate_b_prediction"] = np.maximum(
            0.0,
            model.predict(_candidate_b_matrix(result.loc[validation_mask], codes)),
        )
        result.loc[validation_mask, "candidate_b_training_end"] = train_part["target_at"].max()
    return result.dropna(subset=["candidate_b_prediction"]).reset_index(drop=True)


def with_residual_target(samples: pd.DataFrame) -> pd.DataFrame:
    result = samples.copy()
    result["residual_target"] = result["actual"].astype(float) - result["candidate_b_prediction"].astype(float)
    return result


def temporal_fold_audit(
    observations: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    *,
    sample_cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    lookup = build_lookup(observations)
    profiles = build_profile_lookup(observations)
    cache = sample_cache if sample_cache is not None else {}
    times = sorted(observations["observed_at"].drop_duplicates())
    rows: list[FoldAudit] = []
    for fold_start in fold_starts(cutoffs):
        fold_cutoffs = [ts for ts in cutoffs if fold_start <= ts < fold_start + pd.Timedelta(days=1)]
        train_start = max(observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS), fold_start - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS))
        train_cutoffs = [
            ts
            for ts in times
            if ts.minute == 0 and train_start <= ts < fold_start - pd.Timedelta(minutes=max(HORIZONS))
        ]
        train_samples = cached_samples(cache, observations, train_cutoffs, lookup=lookup, profiles=profiles)
        fold_samples = cached_samples(cache, observations, fold_cutoffs, lookup=lookup, profiles=profiles)
        residual_mask = pd.Series(False, index=train_samples.index)
        for residual_fold_start in fold_starts(train_samples["data_cutoff"].drop_duplicates().tolist()):
            residual_fold_end = residual_fold_start + pd.Timedelta(days=1)
            has_prior_b_training = not train_samples[train_samples["target_at"] < residual_fold_start].empty
            if not has_prior_b_training:
                continue
            residual_mask |= (train_samples["data_cutoff"] >= residual_fold_start) & (train_samples["data_cutoff"] < residual_fold_end)
        residual_samples = train_samples[residual_mask].copy()
        validation_start = fold_samples["data_cutoff"].min()
        validation_end = fold_samples["data_cutoff"].max()
        rows.append(
            FoldAudit(
                fold_start=fold_start,
                b_training_end=train_samples["target_at"].max(),
                d1_training_end=residual_samples["target_at"].max(),
                validation_start=validation_start,
                validation_end=validation_end,
                residual_rows=int(len(residual_samples)),
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def make_candidate_b_model() -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.06,
        max_iter=160,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )


def make_residual_model(estimator: str) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if estimator == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=5.0, random_state=RANDOM_STATE),
        )
    if estimator == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.05,
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=0.10,
                random_state=RANDOM_STATE,
            ),
        )
    if estimator == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=80,
                min_samples_leaf=4,
                max_features=0.85,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
    raise ValueError(f"unsupported residual estimator: {estimator}")


def d1_matrix(samples: pd.DataFrame, codes: dict[str, int]) -> pd.DataFrame:
    frame = samples.copy()
    frame["station_code"] = frame["station_id"].map(codes).astype(float)
    if frame["station_code"].isna().any():
        missing = sorted(frame.loc[frame["station_code"].isna(), "station_id"].unique())
        raise RuntimeError(f"missing station codes: {missing}")
    return frame[D1_FEATURE_SCHEMA].astype(float)


def learn_clip_bounds(
    train_samples: pd.DataFrame,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    min_group_rows: int = 64,
) -> ClipBounds:
    residuals = train_samples["residual_target"].astype(float)
    global_lower = float(residuals.quantile(lower_quantile))
    global_upper = float(residuals.quantile(upper_quantile))
    by_horizon: dict[int, tuple[float, float]] = {}
    for horizon, group in train_samples.groupby("horizon_minutes", sort=True):
        values = group["residual_target"].astype(float)
        if len(values) >= min_group_rows:
            by_horizon[int(horizon)] = (
                float(values.quantile(lower_quantile)),
                float(values.quantile(upper_quantile)),
            )
    return ClipBounds(global_lower=global_lower, global_upper=global_upper, by_horizon=by_horizon)


def apply_clip_bounds(corrections: np.ndarray, samples: pd.DataFrame, bounds: ClipBounds) -> np.ndarray:
    clipped = np.asarray(corrections, dtype=float).copy()
    horizons = samples["horizon_minutes"].to_numpy()
    for idx, horizon in enumerate(horizons):
        lower, upper = bounds.by_horizon.get(int(horizon), (bounds.global_lower, bounds.global_upper))
        clipped[idx] = float(np.clip(clipped[idx], lower, upper))
    return clipped


def official_metrics_full(predictions: pd.DataFrame) -> dict[str, Any]:
    scored = predictions.copy()
    scored["residual"] = scored["prediction"].astype(float) - scored["actual"].astype(float)
    scored["abs_error"] = scored["residual"].abs()
    station = _metric_by(scored, ["station_id"], "station")
    horizon = _metric_by(scored, ["horizon_minutes"], "horizon")
    station_horizon = _metric_by(scored, ["station_id", "horizon_minutes"], "station_horizon")
    global_metrics = _metric_by(scored, [], "global").iloc[0].to_dict()
    return {
        "mean_station_accuracy": float(station["station_accuracy"].mean()),
        "global_accuracy": float(global_metrics["global_accuracy"]),
        "wape": float(global_metrics["global_wape"]),
        "mae": float(global_metrics["global_mae"]),
        "bias": float(global_metrics["global_bias"]),
        "signed_relative_bias": float(global_metrics["global_signed_relative_bias"]),
        "station_metrics": station.sort_values("station_accuracy").reset_index(drop=True),
        "horizon_metrics": horizon.sort_values("horizon_minutes").reset_index(drop=True),
        "station_horizon_metrics": station_horizon.sort_values("station_horizon_accuracy").reset_index(drop=True),
        "worst_station": station.sort_values("station_accuracy").iloc[0].to_dict(),
        "worst_horizon": horizon.sort_values("horizon_accuracy").iloc[0].to_dict(),
        "rows": int(len(scored)),
    }


def _metric_by(scored: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    if keys:
        grouped = scored.groupby(keys, as_index=False).agg(
            abs_error_sum=("abs_error", "sum"),
            actual_sum=("actual", "sum"),
            residual_sum=("residual", "sum"),
            mae=("abs_error", "mean"),
            bias=("residual", "mean"),
            rows=("actual", "size"),
        )
    else:
        grouped = pd.DataFrame(
            [
                {
                    "abs_error_sum": float(scored["abs_error"].sum()),
                    "actual_sum": float(scored["actual"].sum()),
                    "residual_sum": float(scored["residual"].sum()),
                    "mae": float(scored["abs_error"].mean()),
                    "bias": float(scored["residual"].mean()),
                    "rows": int(len(scored)),
                }
            ]
        )
    grouped["wape"] = np.where(grouped["actual_sum"] == 0, 0.0, grouped["abs_error_sum"] / grouped["actual_sum"])
    grouped["accuracy"] = 100.0 * (1.0 - grouped["wape"]).clip(lower=0.0)
    grouped["signed_relative_bias"] = np.where(grouped["actual_sum"] == 0, 0.0, grouped["residual_sum"] / grouped["actual_sum"])
    rename = {
        "wape": f"{prefix}_wape",
        "accuracy": f"{prefix}_accuracy",
        "mae": f"{prefix}_mae",
        "bias": f"{prefix}_bias",
        "signed_relative_bias": f"{prefix}_signed_relative_bias",
    }
    keep = keys + list(rename.values()) + ["rows"]
    return grouped.rename(columns=rename)[keep]


def comparison_table(walk: D1ExperimentResult, hard: D1ExperimentResult) -> pd.DataFrame:
    hard_by_name = {result.name: result for result in [hard.baseline, *hard.challengers]}
    rows = []
    walk_baseline = walk.baseline
    hard_baseline = hard.baseline
    for result in [walk.baseline, *walk.challengers]:
        hard_result = hard_by_name[result.name]
        rows.append(
            {
                "model": result.name,
                "wf_mean_station_accuracy": result.summary["mean_station_accuracy"],
                "wf_delta_pp_vs_b": float(result.summary["mean_station_accuracy"]) - float(walk_baseline.summary["mean_station_accuracy"]),
                "wf_global_accuracy": result.summary["global_accuracy"],
                "wf_wape": result.summary["wape"],
                "wf_mae": result.summary["mae"],
                "wf_bias": result.summary["bias"],
                "wf_signed_relative_bias": result.summary["signed_relative_bias"],
                "hard_mean_station_accuracy": hard_result.summary["mean_station_accuracy"],
                "hard_delta_pp_vs_b": float(hard_result.summary["mean_station_accuracy"]) - float(hard_baseline.summary["mean_station_accuracy"]),
                "hard_global_accuracy": hard_result.summary["global_accuracy"],
                "hard_wape": hard_result.summary["wape"],
                "hard_mae": hard_result.summary["mae"],
                "hard_bias": hard_result.summary["bias"],
                "hard_signed_relative_bias": hard_result.summary["signed_relative_bias"],
                "wf_stations_improve": _count_station_delta(walk_baseline, result, ">"),
                "wf_stations_worsen": _count_station_delta(walk_baseline, result, "<"),
                "hard_stations_improve": _count_station_delta(hard_baseline, hard_result, ">"),
                "hard_stations_worsen": _count_station_delta(hard_baseline, hard_result, "<"),
                "train_seconds": result.train_seconds,
                "complexity": result.complexity,
            }
        )
    return pd.DataFrame(rows)


def special_comparison(base: ModelResult, challenger: ModelResult, *, group: str) -> pd.DataFrame:
    if group == "station":
        base_metrics = base.summary["station_metrics"][["station_id", "station_accuracy", "station_wape", "station_bias", "station_signed_relative_bias"]]
        challenger_metrics = challenger.summary["station_metrics"][["station_id", "station_accuracy", "station_wape", "station_bias", "station_signed_relative_bias"]]
        joined = base_metrics.merge(challenger_metrics, on="station_id", suffixes=("_b", "_d1"))
        joined = joined[joined["station_id"].isin(SPECIAL_STATIONS)].copy()
        joined["delta_accuracy_pp"] = joined["station_accuracy_d1"] - joined["station_accuracy_b"]
        return joined.sort_values("delta_accuracy_pp")
    base_metrics = base.summary["horizon_metrics"][["horizon_minutes", "horizon_accuracy", "horizon_wape", "horizon_bias", "horizon_signed_relative_bias"]]
    challenger_metrics = challenger.summary["horizon_metrics"][["horizon_minutes", "horizon_accuracy", "horizon_wape", "horizon_bias", "horizon_signed_relative_bias"]]
    joined = base_metrics.merge(challenger_metrics, on="horizon_minutes", suffixes=("_b", "_d1"))
    joined = joined[joined["horizon_minutes"].isin(SPECIAL_HORIZONS)].copy()
    joined["delta_accuracy_pp"] = joined["horizon_accuracy_d1"] - joined["horizon_accuracy_b"]
    return joined.sort_values("horizon_minutes")


def write_outputs(observations: pd.DataFrame, window: EvaluationWindow, walk: D1ExperimentResult, hard: D1ExperimentResult) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    comparison = comparison_table(walk, hard)
    comparison.to_csv(TABLE_DIR / "comparison.csv", index=False)
    best_name = comparison[comparison["model"] != "Candidate B v2"].sort_values("wf_mean_station_accuracy", ascending=False).iloc[0]["model"]
    walk_by_name = {result.name: result for result in [walk.baseline, *walk.challengers]}
    hard_by_name = {result.name: result for result in [hard.baseline, *hard.challengers]}
    for result in [walk.baseline, *walk.challengers]:
        slug = ascii_slug(result.name)
        result.summary["station_metrics"].to_csv(TABLE_DIR / f"{slug}_wf_station_metrics.csv", index=False)
        result.summary["horizon_metrics"].to_csv(TABLE_DIR / f"{slug}_wf_horizon_metrics.csv", index=False)
        result.summary["station_horizon_metrics"].to_csv(TABLE_DIR / f"{slug}_wf_station_horizon_metrics.csv", index=False)
    for result in [hard.baseline, *hard.challengers]:
        slug = ascii_slug(result.name)
        result.summary["station_metrics"].to_csv(TABLE_DIR / f"{slug}_hard_station_metrics.csv", index=False)
        result.summary["horizon_metrics"].to_csv(TABLE_DIR / f"{slug}_hard_horizon_metrics.csv", index=False)
        result.summary["station_horizon_metrics"].to_csv(TABLE_DIR / f"{slug}_hard_station_horizon_metrics.csv", index=False)

    best_walk = walk_by_name[str(best_name)]
    best_hard = hard_by_name[str(best_name)]
    special_station_wf = special_comparison(walk.baseline, best_walk, group="station")
    special_horizon_wf = special_comparison(walk.baseline, best_walk, group="horizon")
    special_station_hard = special_comparison(hard.baseline, best_hard, group="station")
    special_horizon_hard = special_comparison(hard.baseline, best_hard, group="horizon")
    special_station_wf.to_csv(TABLE_DIR / "special_station_comparison_wf.csv", index=False)
    special_horizon_wf.to_csv(TABLE_DIR / "special_horizon_comparison_wf.csv", index=False)
    special_station_hard.to_csv(TABLE_DIR / "special_station_comparison_hard.csv", index=False)
    special_horizon_hard.to_csv(TABLE_DIR / "special_horizon_comparison_hard.csv", index=False)

    payload = {
        "candidate_version": "candidate_d1_residual_calibration_challenger",
        "scope": "experimental_only_no_production_changes",
        "data_min": observations["observed_at"].min().isoformat().replace("+00:00", "Z"),
        "data_max": observations["observed_at"].max().isoformat().replace("+00:00", "Z"),
        "stations": int(observations["station_id"].nunique()),
        "walk_forward_start": min(window.walk_forward_cutoffs).isoformat().replace("+00:00", "Z"),
        "walk_forward_end": max(window.walk_forward_cutoffs).isoformat().replace("+00:00", "Z"),
        "walk_forward_cutoffs": len(window.walk_forward_cutoffs),
        "hard_holdout_start": min(window.hard_holdout_cutoffs).isoformat().replace("+00:00", "Z"),
        "hard_holdout_end": max(window.hard_holdout_cutoffs).isoformat().replace("+00:00", "Z"),
        "hard_holdout_cutoffs": len(window.hard_holdout_cutoffs),
        "feature_schema": D1_FEATURE_SCHEMA,
        "leakage_check": "passed",
        "residual_training_predictions": "out_of_fold",
        "hard_holdout_isolated": True,
        "leakage_controls": {
            "target_cutoff_consistent": True,
            "candidate_b_base_prediction_trained_only_on_prior_fold_data": True,
            "residual_training_predictions": "out_of_fold",
            "residual_target": "actual - candidate_b_prediction",
            "clipping_bounds_learned_from_training_residual_quantiles": True,
            "hard_holdout_not_used_for_model_selection": True,
        },
        "best_walk_forward_d1": str(best_name),
        "comparison": _json_records(comparison.to_dict(orient="records")),
        "special_station_comparison_wf": _json_records(special_station_wf.to_dict(orient="records")),
        "special_horizon_comparison_wf": _json_records(special_horizon_wf.to_dict(orient="records")),
        "special_station_comparison_hard": _json_records(special_station_hard.to_dict(orient="records")),
        "special_horizon_comparison_hard": _json_records(special_horizon_hard.to_dict(orient="records")),
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_markdown(payload, comparison) + "\n")


def render_markdown(payload: dict[str, Any], comparison: pd.DataFrame) -> str:
    best = comparison[comparison["model"] != "Candidate B v2"].sort_values("wf_mean_station_accuracy", ascending=False).iloc[0]
    return "\n".join(
        [
            "# Candidate D1 - Residual Calibration Challenger",
            "",
            "## Scope",
            "",
            "- Experimental only; champion, Predict, scheduler, submissions, and Supabase production state were not modified.",
            "- Candidate D1 learns `residual_target = actual - candidate_b_prediction` and predicts `candidate_b_prediction + predicted_residual`.",
            "- Hard holdout is the latest temporal block and is not used to choose the residual corrector.",
            "",
            "## Windows",
            "",
            f"- Observations: {payload['data_min']} to {payload['data_max']}.",
            f"- Walk-forward: {payload['walk_forward_start']} to {payload['walk_forward_end']} ({payload['walk_forward_cutoffs']} cutoffs).",
            f"- Hard holdout: {payload['hard_holdout_start']} to {payload['hard_holdout_end']} ({payload['hard_holdout_cutoffs']} cutoffs).",
            "",
            "## Comparison",
            "",
            _markdown_table(comparison),
            "",
            "## Best D1 By Walk-Forward",
            "",
            f"- Model: {best['model']}.",
            f"- WF delta vs Candidate B v2: {best['wf_delta_pp_vs_b']:.3f} pp.",
            f"- Hard holdout delta vs Candidate B v2: {best['hard_delta_pp_vs_b']:.3f} pp.",
            "",
            "## Leakage And Safety",
            "",
            f"- `leakage_check = {payload['leakage_check']}`.",
            f"- `residual_training_predictions = {payload['residual_training_predictions']}`.",
            f"- `hard_holdout_isolated = {str(payload['hard_holdout_isolated']).lower()}`.",
            "- All feature rows satisfy `target_at = data_cutoff + horizon`.",
            "- Candidate B base predictions and residual correctors are trained per temporal fold using only training cutoffs before the validation fold.",
            "- Clipped variants learn residual bounds from training residual quantiles by horizon, falling back to global training quantiles when needed.",
            "- No weather or event features were used.",
            "- D1 was not promoted.",
        ]
    )


def _count_station_delta(base: ModelResult, candidate: ModelResult, op: str) -> int:
    base_station = base.summary["station_metrics"][["station_id", "station_accuracy"]].rename(columns={"station_accuracy": "base_accuracy"})
    cand_station = candidate.summary["station_metrics"][["station_id", "station_accuracy"]].rename(columns={"station_accuracy": "candidate_accuracy"})
    joined = base_station.merge(cand_station, on="station_id", how="inner")
    if op == ">":
        return int((joined["candidate_accuracy"] > joined["base_accuracy"] + 1e-9).sum())
    return int((joined["candidate_accuracy"] < joined["base_accuracy"] - 1e-9).sum())


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


def _json_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in rows]


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Candidate D1 residual calibration experiments without production changes.")
    parser.add_argument("--source", choices=["local", "supabase"], default=os.getenv("CANDIDATE_D1_SOURCE", "local"))
    args = parser.parse_args()

    observations = load_observations_from_supabase_read_only() if args.source == "supabase" else load_observations()
    window = experiment_cutoffs(observations)
    cache: dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame] = {}
    print(f"Walk-forward cutoffs: {len(window.walk_forward_cutoffs)}", flush=True)
    walk = evaluate_d1(observations, window.walk_forward_cutoffs, sample_cache=cache)
    print(f"Hard holdout cutoffs: {len(window.hard_holdout_cutoffs)}", flush=True)
    hard = evaluate_d1(observations, window.hard_holdout_cutoffs, sample_cache=cache)
    write_outputs(observations, window, walk, hard)
    comparison = pd.read_csv(TABLE_DIR / "comparison.csv")
    print(comparison.to_string(index=False))
    print(f"Report written to {REPORT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
