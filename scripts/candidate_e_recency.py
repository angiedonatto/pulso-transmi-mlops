"""Candidate E -- Adaptive / Recency-Aware Training.

Studies whether the recent accuracy drop (historical hard holdout ~88.46% vs
recent-window Candidate B ~84.99%) is explained by stale training / concept drift,
rather than by architecture or features. Everything below holds the model family
(HistGradientBoostingRegressor, Candidate B's own hyperparameters, no imputer -- the
exact `make_candidate_b_model()` used in production) and the feature set
(`BASELINE_FEATURE_SCHEMA`, Candidate B's own 17 production-safe features) fixed. The
only thing that ever varies is the TRAINING STRATEGY: how much history to train on, how
to weight it by recency, and how often to retrain.

Fidelity note: production's actual training function
(`src/candidate_b_shadow.py::_training_samples`) trains on an EXPANDING window (all
history since `observations_min + MIN_HISTORY_DAYS`) built from every 15-minute
timestamp, not the rolling 21-day / hourly-cutoff convention used by
`model_improvement_candidates.candidate_b_predictions()` (which Candidate D1 and
Challenger Lab v2's baseline both reused). Candidate E's control (Part A) reproduces
the REAL production training behavior -- dense (15-min) training cutoffs, true
expanding window -- so this is a genuinely different, and more faithful, control than
"Candidate B v2 (global baseline)" elsewhere in this repo. This is deliberate and is
exactly what was asked for ("Reproduce el comportamiento actual").

Evaluation windows: everything is evaluated on the already-seen 2026-09-09 -> 2026-09-13
window (`data/raw/observations_refresh.csv`, from the Challenger Lab v2 refresh) as a
development/temporal-validation surface -- NOT a virgin holdout, and never used to
declare a final promotion decision. The historical 2026-09-01 -> 2026-09-08 walk-forward
window is reused as a secondary robustness check for the two most relevant strategies
only (the control and whichever strategy wins on the recent window), to bound compute.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from challenger_lab_v2_core import (  # noqa: E402
    BASELINE_FEATURE_SCHEMA,
    HORIZONS,
    ModelResult,
    cached_extended_samples,
    build_matrix,
    experiment_cutoffs,
    load_observations,
    load_observations_from_frame,
    make_candidate_b_model,
    official_metrics_full,
    station_code_map,
)
from challenger_lab_v2 import full_metrics_result, horizon_metrics_table, per_fold_metrics, station_metrics_table  # noqa: E402
from candidate_d1_residual_calibration import SPECIAL_HORIZONS, SPECIAL_STATIONS, special_comparison  # noqa: E402

MIN_HISTORY_DAYS = 14
MAX_HORIZON = max(HORIZONS)
ROLLING_WINDOWS_DAYS = (7, 14, 21, 28)
HALF_LIVES_DAYS = (1.0, 3.0, 7.0, 14.0)
CADENCES_HOURS = (1, 6, 12, 24)
MIN_TRAIN_ROWS = 500

DRIFT_MONITOR_REFERENCE_SIGNED_BIAS = -0.0676
REFRESH_REFERENCE = {"recent_window_b_accuracy": 84.9893, "historical_b_accuracy_wf": 87.2798, "historical_b_accuracy_hard": 88.4628}

DATA_PATH = ROOT / "data" / "raw" / "observations_refresh.csv"
REPORT_DIR = ROOT / "reports" / "candidate_e_recency"
TABLE_DIR = REPORT_DIR / "tables"

Cache = dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_refreshed_observations() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found -- run scripts/refresh_observations_snapshot.py first.")
    frame = pd.read_csv(DATA_PATH, dtype={"station_id": "string"})
    return load_observations_from_frame(frame)


def dense_times(observations: pd.DataFrame) -> list[pd.Timestamp]:
    """Every 15-minute timestamp, NOT restricted to minute==0 -- matches production's
    actual training cutoff density (`_training_samples` in candidate_b_shadow.py)."""
    return sorted(observations["observed_at"].drop_duplicates())


def recent_window_eval_cutoffs(observations: pd.DataFrame, old_max: pd.Timestamp) -> list[pd.Timestamp]:
    times = dense_times(observations)
    latest_cutoff = times[-1] - pd.Timedelta(minutes=MAX_HORIZON)
    return [ts for ts in times if ts.minute == 0 and old_max < ts <= latest_cutoff]


# --------------------------------------------------------------------------------------
# Training-window strategies. `reference_time` is the point in time training happens --
# for cadence bucketing this is the bucket start, not each individual target_at, so the
# weight/window calculation never depends on anything happening inside the bucket.
# --------------------------------------------------------------------------------------
def expanding_train_cutoffs(observations: pd.DataFrame, times: list[pd.Timestamp], reference_time: pd.Timestamp) -> list[pd.Timestamp]:
    train_start = observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
    cutoff_limit = reference_time - pd.Timedelta(minutes=MAX_HORIZON)
    return [ts for ts in times if train_start <= ts < cutoff_limit]


def rolling_train_cutoffs(observations: pd.DataFrame, times: list[pd.Timestamp], reference_time: pd.Timestamp, window_days: float) -> list[pd.Timestamp]:
    train_start = max(
        observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS),
        reference_time - pd.Timedelta(days=window_days),
    )
    cutoff_limit = reference_time - pd.Timedelta(minutes=MAX_HORIZON)
    return [ts for ts in times if train_start <= ts < cutoff_limit]


def recency_sample_weights(train_samples: pd.DataFrame, reference_time: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """Weight depends ONLY on (reference_time - data_cutoff); never on target/error.
    `reference_time` is fixed per training call (the bucket/fold start), so this cannot
    see anything about the fold it is about to predict."""
    age_days = (reference_time - train_samples["data_cutoff"]).dt.total_seconds().to_numpy() / 86400.0
    if (age_days < 0).any():
        raise RuntimeError("recency weighting encountered a training row newer than the training reference time")
    return np.power(0.5, age_days / half_life_days)


def cadence_buckets(cutoffs: list[pd.Timestamp], cadence_hours: int) -> list[tuple[pd.Timestamp, list[pd.Timestamp]]]:
    """Groups evaluation cutoffs into buckets of `cadence_hours`; one retrain per
    bucket, reused for every cutoff inside it -- simulating a fixed retraining cadence
    instead of retraining fresh before every single prediction."""
    bucket_starts = sorted({ts.floor(f"{cadence_hours}h") for ts in cutoffs})
    result: list[tuple[pd.Timestamp, list[pd.Timestamp]]] = []
    for bucket_start in bucket_starts:
        bucket_end = bucket_start + pd.Timedelta(hours=cadence_hours)
        bucket_cutoffs = [ts for ts in cutoffs if bucket_start <= ts < bucket_end]
        if bucket_cutoffs:
            result.append((bucket_start, bucket_cutoffs))
    return result


def training_cutoffs_for(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    reference_time: pd.Timestamp,
    *,
    window_kind: str,
    window_days: float | None,
) -> list[pd.Timestamp]:
    if window_kind == "expanding":
        return expanding_train_cutoffs(observations, times, reference_time)
    if window_kind == "rolling":
        assert window_days is not None
        return rolling_train_cutoffs(observations, times, reference_time, window_days)
    raise ValueError(f"unknown window_kind: {window_kind!r}")


# --------------------------------------------------------------------------------------
# Strategy runner: the ONE function every experiment (A/B/C/D) goes through, so the
# model family, features, and no-leakage guarantees are identical across all of them --
# only window_kind/window_days/half_life_days/cadence_hours differ.
# --------------------------------------------------------------------------------------
def run_strategy(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    eval_cutoffs: list[pd.Timestamp],
    *,
    window_kind: str,
    window_days: float | None,
    half_life_days: float | None,
    cadence_hours: int,
    cache: Cache,
    name: str,
) -> ModelResult:
    start = time.perf_counter()
    codes = station_code_map(observations)
    predictions = []
    skipped_buckets = 0
    for bucket_start, bucket_cutoffs in cadence_buckets(eval_cutoffs, cadence_hours):
        train_cutoffs = training_cutoffs_for(observations, times, bucket_start, window_kind=window_kind, window_days=window_days)
        if not train_cutoffs:
            skipped_buckets += 1
            continue
        train_samples = cached_extended_samples(cache, observations, train_cutoffs).dropna(subset=["profile_mean"])
        fold_samples = cached_extended_samples(cache, observations, bucket_cutoffs)
        if len(train_samples) < MIN_TRAIN_ROWS or fold_samples.empty:
            skipped_buckets += 1
            continue
        X_train = build_matrix(train_samples, codes, BASELINE_FEATURE_SCHEMA)
        y_train = train_samples["actual"].to_numpy(dtype=float)
        model = make_candidate_b_model()
        if half_life_days is not None:
            weights = recency_sample_weights(train_samples, bucket_start, half_life_days)
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            model.fit(X_train, y_train)
        fold = fold_samples.copy()
        fold["prediction"] = np.maximum(0.0, model.predict(build_matrix(fold, codes, BASELINE_FEATURE_SCHEMA)))
        fold["bucket_start"] = bucket_start
        fold["training_rows"] = len(train_samples)
        predictions.append(fold)
    if not predictions:
        raise RuntimeError(f"strategy {name!r} produced no predictions (skipped_buckets={skipped_buckets})")
    frame = pd.concat(predictions, ignore_index=True)
    complexity = (
        f"window={window_kind}"
        + (f"({window_days}d)" if window_days else "")
        + f", half_life={half_life_days}, cadence={cadence_hours}h, skipped_buckets={skipped_buckets}"
    )
    return full_metrics_result(name, frame, train_seconds=time.perf_counter() - start, complexity=complexity)


# --------------------------------------------------------------------------------------
# Leakage audit -- mirrors run_strategy's own cutoff selection but only records dates,
# no model fitting, so it is cheap to run for every strategy/cadence combination tested.
# --------------------------------------------------------------------------------------
def strategy_fold_audit(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    eval_cutoffs: list[pd.Timestamp],
    *,
    window_kind: str,
    window_days: float | None,
    cadence_hours: int,
    cache: Cache,
    strategy_name: str,
) -> pd.DataFrame:
    rows = []
    for bucket_start, bucket_cutoffs in cadence_buckets(eval_cutoffs, cadence_hours):
        train_cutoffs = training_cutoffs_for(observations, times, bucket_start, window_kind=window_kind, window_days=window_days)
        if not train_cutoffs:
            continue
        train_samples = cached_extended_samples(cache, observations, train_cutoffs).dropna(subset=["profile_mean"])
        fold_samples = cached_extended_samples(cache, observations, bucket_cutoffs)
        if train_samples.empty or fold_samples.empty:
            continue
        rows.append(
            {
                "strategy": strategy_name,
                "bucket_start": bucket_start,
                "training_end": train_samples["target_at"].max(),
                "training_rows": int(len(train_samples)),
                "validation_start": fold_samples["data_cutoff"].min(),
                "validation_end": fold_samples["data_cutoff"].max(),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------------------
def comparison_row(result: ModelResult, *, strategy: str, train_window: str, recency_weighting: str, retrain_cadence: str, baseline_accuracy: float, fold_table: pd.DataFrame) -> dict[str, Any]:
    eps = 1e-9
    wins = int((fold_table["delta_pp"] > eps).sum()) if not fold_table.empty else 0
    losses = int((fold_table["delta_pp"] < -eps).sum()) if not fold_table.empty else 0
    return {
        "strategy": strategy,
        "train_window": train_window,
        "recency_weighting": recency_weighting,
        "retrain_cadence": retrain_cadence,
        "mean_station_accuracy": result.summary["mean_station_accuracy"],
        "delta_vs_expanding_baseline_pp": result.summary["mean_station_accuracy"] - baseline_accuracy,
        "wape": result.summary["wape"],
        "mae": result.summary["mae"],
        "signed_relative_bias": result.summary["signed_relative_bias"],
        "fold_wins": wins,
        "fold_losses": losses,
        "train_seconds": result.train_seconds,
        "complexity": result.complexity,
    }


def per_bucket_delta_table(baseline: ModelResult, challenger: ModelResult) -> pd.DataFrame:
    b_fold = per_fold_metrics("baseline", baseline.predictions, split="recent_window")
    c_fold = per_fold_metrics("challenger", challenger.predictions, split="recent_window")
    merged = b_fold[["fold_start", "mean_station_accuracy"]].merge(
        c_fold[["fold_start", "mean_station_accuracy"]], on="fold_start", suffixes=("_baseline", "_challenger")
    )
    merged["delta_pp"] = merged["mean_station_accuracy_challenger"] - merged["mean_station_accuracy_baseline"]
    return merged.sort_values("fold_start").reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
STRATEGY_SPECS: list[tuple[str, str, str, str, dict[str, Any]]] = [
    # (name, train_window_label, recency_weighting_label, retrain_cadence_label, kwargs)
    ("D: expanding, 1h cadence (oracle-like fresh retrain)", "expanding", "none", "1h", dict(window_kind="expanding", window_days=None, half_life_days=None, cadence_hours=1)),
    ("D: expanding, 6h cadence", "expanding", "none", "6h", dict(window_kind="expanding", window_days=None, half_life_days=None, cadence_hours=6)),
    ("D: expanding, 12h cadence", "expanding", "none", "12h", dict(window_kind="expanding", window_days=None, half_life_days=None, cadence_hours=12)),
    ("A: expanding window (production-like control)", "expanding", "none", "24h", dict(window_kind="expanding", window_days=None, half_life_days=None, cadence_hours=24)),
    ("C: expanding + recency half-life=1d", "expanding", "half_life=1d", "24h", dict(window_kind="expanding", window_days=None, half_life_days=1.0, cadence_hours=24)),
    ("C: expanding + recency half-life=3d", "expanding", "half_life=3d", "24h", dict(window_kind="expanding", window_days=None, half_life_days=3.0, cadence_hours=24)),
    ("C: expanding + recency half-life=7d", "expanding", "half_life=7d", "24h", dict(window_kind="expanding", window_days=None, half_life_days=7.0, cadence_hours=24)),
    ("C: expanding + recency half-life=14d", "expanding", "half_life=14d", "24h", dict(window_kind="expanding", window_days=None, half_life_days=14.0, cadence_hours=24)),
    ("B: rolling 7d window", "rolling_7d", "none", "24h", dict(window_kind="rolling", window_days=7, half_life_days=None, cadence_hours=24)),
    ("B: rolling 14d window", "rolling_14d", "none", "24h", dict(window_kind="rolling", window_days=14, half_life_days=None, cadence_hours=24)),
    ("B: rolling 21d window", "rolling_21d", "none", "24h", dict(window_kind="rolling", window_days=21, half_life_days=None, cadence_hours=24)),
    ("B: rolling 28d window", "rolling_28d", "none", "24h", dict(window_kind="rolling", window_days=28, half_life_days=None, cadence_hours=24)),
]
CONTROL_NAME = "A: expanding window (production-like control)"


def render_leakage_audit_md(audit_frames: dict[str, pd.DataFrame], all_passed: bool) -> str:
    lines = [
        "# Candidate E -- Leakage Audit",
        "",
        "## Automated checks (see tests/test_candidate_e_recency.py for the pytest versions)",
        "",
        f"1. **training_end < validation_start for every strategy x cadence bucket**: {'PASSED' if all_passed else 'FAILED'}. Checked for all {len(audit_frames)} strategies tested (every window kind and every cadence), not just the control.",
        "2. **Recency sample weights use only timestamps, never target/error**: `recency_sample_weights(train_samples, reference_time, half_life_days)` reads only `train_samples['data_cutoff']` and a fixed `reference_time` (the bucket start); it never reads `actual`, `prediction`, or any residual. `run_strategy` also asserts (raises if violated) that no training row's age relative to the reference time is negative.",
        "3. **Rolling train window never includes validation data**: `rolling_train_cutoffs` bounds training cutoffs to `< reference_time - max_horizon`, identical in form to the expanding-window bound, so a bucket's own validation cutoffs (which start at `reference_time`) are never inside the training range regardless of window length.",
        "4. **Retraining cadence correctly reuses one model across a bucket**: `cadence_buckets` groups evaluation cutoffs into fixed-width buckets and `run_strategy` fits exactly one model per bucket, applying it to every cutoff in that bucket -- verified by `training_rows`/`bucket_start` being constant within a bucket in the raw predictions frame.",
        "5. **Features remain cutoff-safe**: training and evaluation samples are built by the same `challenger_lab_v2_core.build_extended_samples` machinery already leakage-tested in Challenger Lab v2 (lags masked by horizon, rolling stats anchored at `data_cutoff`, profiles using `shift(1)`); Candidate E only selects a features SUBSET (`BASELINE_FEATURE_SCHEMA`) and changes which rows/weights feed the same, already-audited feature builder.",
        "",
        "## Fixes to prior leakage found while building Candidate E",
        "",
        "- None. This module reuses `challenger_lab_v2_core.build_extended_samples` and `build_matrix` unchanged; the only new logic is training-set selection (expanding/rolling cutoffs), sample weighting, and cadence bucketing, all timestamp-only and covered by the checks above.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    recent_results: dict[str, ModelResult],
    historical_results: dict[str, ModelResult],
    audit_frames: dict[str, pd.DataFrame],
    control_name: str,
    best_name: str,
    test_result: str,
    final_holdout_available: bool,
) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    baseline_accuracy = recent_results[control_name].summary["mean_station_accuracy"]
    comparison_rows = []
    fold_frames = []
    station_frames = []
    horizon_frames = []
    for name, train_window, recency, cadence, _ in STRATEGY_SPECS:
        result = recent_results[name]
        fold_table = per_bucket_delta_table(recent_results[control_name], result) if name != control_name else per_bucket_delta_table(result, result)
        comparison_rows.append(comparison_row(result, strategy=name, train_window=train_window, recency_weighting=recency, retrain_cadence=cadence, baseline_accuracy=baseline_accuracy, fold_table=fold_table))
        fold_frames.append(per_fold_metrics(name, result.predictions, split="recent_window"))
        station_frames.append(station_metrics_table(name, result, split="recent_window"))
        horizon_frames.append(horizon_metrics_table(name, result, split="recent_window"))

    comparison = pd.DataFrame(comparison_rows).sort_values("mean_station_accuracy", ascending=False)
    comparison.to_csv(REPORT_DIR / "strategy_comparison.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(REPORT_DIR / "fold_metrics.csv", index=False)
    pd.concat(station_frames, ignore_index=True).to_csv(REPORT_DIR / "station_metrics.csv", index=False)
    pd.concat(horizon_frames, ignore_index=True).to_csv(REPORT_DIR / "horizon_metrics.csv", index=False)

    special_comparison(recent_results[control_name], recent_results[best_name], group="station").to_csv(TABLE_DIR / "special_station_comparison.csv", index=False)
    special_comparison(recent_results[control_name], recent_results[best_name], group="horizon").to_csv(TABLE_DIR / "special_horizon_comparison.csv", index=False)

    all_audit = pd.concat(audit_frames.values(), ignore_index=True)
    all_audit.to_csv(TABLE_DIR / "temporal_fold_audit.csv", index=False)
    leakage_passed = bool((all_audit["training_end"] < all_audit["validation_start"]).all())

    historical_rows = []
    for name, result in historical_results.items():
        historical_rows.append({"strategy": name, "historical_wf_accuracy": result.summary["mean_station_accuracy"]})
    historical_table = pd.DataFrame(historical_rows)
    historical_table.to_csv(TABLE_DIR / "historical_secondary_check.csv", index=False)

    best_row = comparison.iloc[0]
    control_row = comparison[comparison["strategy"] == control_name].iloc[0]

    payload: dict[str, Any] = {
        "candidate": "candidate_e_recency",
        "scope": "training_strategy_only_no_promotion_no_shadow",
        "baseline_recent_accuracy_given": REFRESH_REFERENCE["recent_window_b_accuracy"],
        "control_name": control_name,
        "control_recent_accuracy": float(control_row["mean_station_accuracy"]),
        "best_strategy": str(best_row["strategy"]),
        "best_accuracy": float(best_row["mean_station_accuracy"]),
        "best_delta_pp": float(best_row["delta_vs_expanding_baseline_pp"]),
        "best_bias": float(best_row["signed_relative_bias"]),
        "control_bias": float(control_row["signed_relative_bias"]),
        "best_train_window": str(best_row["train_window"]),
        "best_retrain_cadence": str(best_row["retrain_cadence"]),
        "best_recency_weighting": str(best_row["recency_weighting"]),
        "best_fold_wins": int(best_row["fold_wins"]),
        "best_fold_losses": int(best_row["fold_losses"]),
        "leakage_passed": leakage_passed,
        "test_result": test_result,
        "final_untouched_holdout_available": final_holdout_available,
        "any_model_promoted": False,
        "shadow_deployment": False,
        "strategy_comparison": json.loads(comparison.to_json(orient="records")),
        "historical_secondary_check": json.loads(historical_table.to_json(orient="records")),
    }

    material_improvement = payload["best_delta_pp"] > 0.5 and payload["best_fold_wins"] > payload["best_fold_losses"]
    payload["material_improvement"] = material_improvement

    (REPORT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_markdown(payload, comparison, historical_table))
    (REPORT_DIR / "leakage_audit.md").write_text(render_leakage_audit_md(audit_frames, leakage_passed))
    return payload


def render_markdown(payload: dict[str, Any], comparison: pd.DataFrame, historical_table: pd.DataFrame) -> str:
    lines = [
        "# Candidate E -- Adaptive / Recency-Aware Training",
        "",
        "## Scope",
        "",
        "- Investigates whether the recent accuracy drop is explained by stale training / concept drift, not by architecture or features.",
        "- Model family (HistGradientBoostingRegressor, Candidate B's own hyperparameters, no imputer) and feature set (`BASELINE_FEATURE_SCHEMA`, Candidate B's own 17 production-safe features) are held FIXED across every strategy. Only the training window, recency weighting, and retraining cadence vary.",
        "- Evaluated on the 2026-09-09 -> 2026-09-13 window as development/temporal validation, NOT a virgin holdout (it was already used in the Challenger Lab v2 refresh). No new experiments beyond training-strategy variants; no feature or algorithm search.",
        "- No production, champion, Predict, scheduler, or Supabase changes. No promotion. No shadow deployment. No commit.",
        "",
        "## Fidelity note",
        "",
        f"- The externally-given recent-window Candidate B reference is {REFRESH_REFERENCE['recent_window_b_accuracy']:.4f}% (rolling 21-day window, hourly-only training cutoffs -- the convention used throughout Challenger Lab v2).",
        f"- This lab's own control (`{payload['control_name']}`) reproduces production's ACTUAL training function (`candidate_b_shadow.py::_training_samples`: true expanding window, every-15-minute training cutoffs) and scores {payload['control_recent_accuracy']:.4f}% on the same window.",
        "- These two numbers differ because they are two different (both legitimate) readings of 'Candidate B': one matches the walk-forward convention used for model *selection* elsewhere in this repo, the other matches what the champion's *own* training code actually does. The gap between them is itself informative about how sensitive recent performance is to training convention.",
        "",
        "## Strategy comparison (recent/development window, sorted by accuracy)",
        "",
        _markdown_table(comparison[["strategy", "train_window", "recency_weighting", "retrain_cadence", "mean_station_accuracy", "delta_vs_expanding_baseline_pp", "wape", "mae", "signed_relative_bias", "fold_wins", "fold_losses"]]),
        "",
        "## Secondary check: does the picture change on the historical (2026-09-01 -> 09-08) window?",
        "",
        _markdown_table(historical_table),
        "",
        "This window was already used for model selection in the original Challenger Lab v2, so it is not a fresh check either -- it only tells us whether the recent-window ranking generalizes to a second already-seen window, run for the control and the best recent-window strategy only (to bound compute).",
        "",
        "## Result",
        "",
        f"- Best strategy on the recent/development window: **{payload['best_strategy']}** at {payload['best_accuracy']:.4f}% (delta vs the expanding-window control: {payload['best_delta_pp']:+.4f} pp; fold record {payload['best_fold_wins']}W-{payload['best_fold_losses']}L).",
        f"- Control (`{payload['control_name']}`): {payload['control_recent_accuracy']:.4f}%, signed_relative_bias {payload['control_bias']:.4f}.",
        f"- Best strategy's signed_relative_bias: {payload['best_bias']:.4f}.",
        f"- `material_improvement = {str(payload['material_improvement']).lower()}` (threshold: >0.5pp AND more fold wins than losses -- a deliberately conservative bar, not forced to declare success).",
        "",
        ("No training strategy tested here materially improves on the expanding-window control. The recent-window accuracy drop does not appear to be a simple retraining-recency artifact fixable by changing window length, recency weighting, or cadence alone." if not payload["material_improvement"] else f"**{payload['best_strategy']}** shows a material, fold-consistent improvement over the expanding-window control on the recent/development window."),
        "",
        "## Leakage check",
        "",
        f"- `leakage_passed = {str(payload['leakage_passed']).lower()}` across every strategy and cadence tested (not just the control). Full detail: `leakage_audit.md`, `tables/temporal_fold_audit.csv`.",
        "",
        "## Test result",
        "",
        f"- `python3 -m pytest`: {payload['test_result']}",
        "",
        "## Final untouched holdout",
        "",
        f"- `final_untouched_holdout_available = {str(payload['final_untouched_holdout_available']).lower()}`. Both the historical (09-01..09-08) and recent (09-09..09-13) windows have now been looked at and used for selection-style comparisons (original Challenger Lab v2, its refresh, and this Candidate E search). There is no unseen data left in the local/Supabase snapshot to confirm a choice against. Any preference expressed here is therefore **provisional**, pending a genuinely fresh window once more data accrues.",
        "",
        "## Decision",
        "",
        "- `any_model_promoted = false`. `shadow_deployment = false`. Nothing here changes production.",
        "",
        "## Files",
        "",
        "- `reports/candidate_e_recency/{summary.md, summary.json, strategy_comparison.csv, fold_metrics.csv, station_metrics.csv, horizon_metrics.csv, leakage_audit.md}`",
        "- `reports/candidate_e_recency/tables/{special_station_comparison.csv, special_horizon_comparison.csv, temporal_fold_audit.csv, historical_secondary_check.csv}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    t_all = time.perf_counter()
    print("Loading refreshed observations (data/raw/observations_refresh.csv)...", flush=True)
    observations = load_refreshed_observations()
    old_observations = load_observations()
    old_max = old_observations["observed_at"].max()
    times = dense_times(observations)
    print(f"  dense timestamps: {len(times)}", flush=True)

    recent_cutoffs = recent_window_eval_cutoffs(observations, old_max)
    print(f"  recent/development window: {min(recent_cutoffs)} .. {max(recent_cutoffs)} ({len(recent_cutoffs)} cutoffs)", flush=True)

    cache: Cache = {}
    recent_results: dict[str, ModelResult] = {}
    audit_frames: dict[str, pd.DataFrame] = {}

    print("\nRunning strategies on the recent/development window (order chosen to maximize feature-cache reuse: 1h cadence first)...", flush=True)
    for name, train_window, recency, cadence, kwargs in STRATEGY_SPECS:
        t0 = time.perf_counter()
        result = run_strategy(observations, times, recent_cutoffs, cache=cache, name=name, **kwargs)
        recent_results[name] = result
        audit_frames[name] = strategy_fold_audit(observations, times, recent_cutoffs, cache=cache, strategy_name=name, window_kind=kwargs["window_kind"], window_days=kwargs["window_days"], cadence_hours=kwargs["cadence_hours"])
        print(f"    {name}: {result.summary['mean_station_accuracy']:.4f}% ({time.perf_counter()-t0:.1f}s)", flush=True)

    control_name = CONTROL_NAME
    best_name = max(recent_results, key=lambda k: recent_results[k].summary["mean_station_accuracy"])
    print(f"\nControl: {control_name} = {recent_results[control_name].summary['mean_station_accuracy']:.4f}%", flush=True)
    print(f"Best on recent window: {best_name} = {recent_results[best_name].summary['mean_station_accuracy']:.4f}%", flush=True)

    print("\nSecondary check on the historical (09-01..09-08) window: control + best strategy only...", flush=True)
    old_window = experiment_cutoffs(old_observations)
    historical_results: dict[str, ModelResult] = {}
    for name in {control_name, best_name}:
        spec = next(s for s in STRATEGY_SPECS if s[0] == name)
        kwargs = spec[4]
        result = run_strategy(observations, times, old_window.walk_forward_cutoffs, cache=cache, name=name, **kwargs)
        historical_results[name] = result
        print(f"    {name}: {result.summary['mean_station_accuracy']:.4f}%", flush=True)

    print("\nRunning tests...", flush=True)
    import subprocess

    test_proc = subprocess.run(["python3", "-m", "pytest", "-q"], cwd=str(ROOT), capture_output=True, text=True)
    test_result_line = test_proc.stdout.strip().splitlines()[-1] if test_proc.stdout.strip() else "unknown"
    test_result = f"{'PASSED' if test_proc.returncode == 0 else 'FAILED'} -- {test_result_line}"
    print(f"  {test_result}", flush=True)

    payload = write_outputs(
        recent_results=recent_results,
        historical_results=historical_results,
        audit_frames=audit_frames,
        control_name=control_name,
        best_name=best_name,
        test_result=test_result,
        final_holdout_available=False,
    )

    print(f"\nTotal wall time: {time.perf_counter() - t_all:.1f}s", flush=True)
    print(f"Report written to {REPORT_DIR / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
