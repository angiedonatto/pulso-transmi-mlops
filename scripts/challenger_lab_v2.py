"""Challenger Lab v2 -- hypothesis-driven search for a production-eligible challenger
that beats Candidate B v2, without leakage and without touching production.

Design summary (see reports/challenger_lab_v2/summary.md for the actual results):

- Temporal windows are Candidate D1's exact, already-audited walk-forward (7 days) and
  hard holdout (last 24h) windows. They are computed once via
  `challenger_lab_v2_core.experiment_cutoffs` and never redefined after seeing results.
- Model/feature selection happens ONLY on walk-forward. The hard holdout is evaluated
  exactly once, at the very end, for a frozen shortlist -- never used to pick between
  feature sets, architectures, families or ensemble weights.
- Every experiment answers one question (see EXPERIMENT REGISTRY below); this is not a
  blind hyperparameter sweep.
- All data is the local CSV snapshot (`data/raw/observations.csv`), the same source
  Candidate D1 used to produce the 87.2798% / 88.4628% numbers this lab is measured
  against. Nothing here reads or writes Supabase.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import challenger_lab_v2_core as core  # noqa: E402
from challenger_lab_v2_core import (  # noqa: E402
    BASELINE_FEATURE_SCHEMA,
    FEATURE_SET_ALL,
    FEATURE_SET_PLUS_DOW_PROFILE,
    FEATURE_SET_PLUS_LAGS,
    FEATURE_SET_PLUS_ROLLING,
    FEATURE_SET_PLUS_TREND,
    HORIZONS,
    ModelResult,
    build_matrix,
    cached_extended_samples,
    experiment_cutoffs,
    fold_starts,
    load_context,
    load_observations,
    make_candidate_b_model,
    make_model,
    official_metrics_full,
    station_code_map,
    temporal_fold_audit_generic,
    train_cutoffs_for_fold,
)
from candidate_d1_residual_calibration import special_comparison  # noqa: E402

DRIFT_FLAGGED_STATIONS = ("05000", "07111", "03000", "09122")
REPORT_DIR = ROOT / "reports" / "challenger_lab_v2"
TABLE_DIR = REPORT_DIR / "tables"
TARGET_ACCURACY = 90.0

Cache = dict[tuple[pd.Timestamp, pd.Timestamp, int], pd.DataFrame]


# --------------------------------------------------------------------------------------
# Catalog entry: one row of bookkeeping per candidate, independent of which experiment
# produced it. This is what eventually becomes model_comparison.csv.
# --------------------------------------------------------------------------------------
class Candidate:
    def __init__(
        self,
        name: str,
        *,
        hypothesis: str,
        production_eligible: bool,
        wf_result: ModelResult,
        notes: str = "",
        research_only: bool = False,
    ) -> None:
        self.name = name
        self.hypothesis = hypothesis
        self.production_eligible = production_eligible
        self.research_only = research_only
        self.wf_result = wf_result
        self.hard_result: ModelResult | None = None
        self.notes = notes


def full_metrics_result(name: str, predictions: pd.DataFrame, *, train_seconds: float, complexity: str) -> ModelResult:
    return ModelResult(name=name, predictions=predictions, summary=official_metrics_full(predictions), train_seconds=train_seconds, complexity=complexity)


# --------------------------------------------------------------------------------------
# Generic walk-forward runners. Each retrains daily (matching Candidate B's own
# TRAIN_LOOKBACK_DAYS / MIN_HISTORY_DAYS convention) and only ever fits on
# `train_samples`, whose rows all satisfy target_at < fold_start <= data_cutoff of the
# validation fold (checked by temporal_fold_audit_generic / tests/test_challenger_lab_v2.py).
# --------------------------------------------------------------------------------------
def _iter_folds(observations: pd.DataFrame, times: list[pd.Timestamp], cutoffs: list[pd.Timestamp], cache: Cache):
    for fold_start in fold_starts(cutoffs):
        fold_cutoffs = [ts for ts in cutoffs if fold_start <= ts < fold_start + pd.Timedelta(days=1)]
        train_cutoffs = train_cutoffs_for_fold(observations, times, fold_start)
        train_samples = cached_extended_samples(cache, observations, train_cutoffs)
        fold_samples = cached_extended_samples(cache, observations, fold_cutoffs)
        if train_samples.empty or fold_samples.empty:
            continue
        yield fold_start, train_samples, fold_samples


def run_candidate_b_baseline(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    cutoffs: list[pd.Timestamp],
    *,
    name: str,
    cache: Cache,
) -> ModelResult:
    """The control every other experiment is measured against: the exact Candidate B v2
    architecture (`make_candidate_b_model()` -- raw HistGradientBoostingRegressor, no
    imputer, matching production and Candidate D1 bit-for-bit) on `BASELINE_FEATURE_SCHEMA`,
    trained through the same `_iter_folds`/`train_cutoffs_for_fold` (minute==0-restricted,
    TRAIN_LOOKBACK_DAYS/MIN_HISTORY_DAYS) pipeline every other candidate in this lab uses.

    This deliberately does NOT call `model_improvement_candidates.candidate_b_predictions()`:
    that function trains on every 15-minute cutoff in the lookback window rather than only
    the minute==0 cutoffs Candidate D1 (and every other experiment here) trains on, which
    would make it a different (denser-trained) model and an unfair control.
    """
    start = time.perf_counter()
    codes = station_code_map(observations)
    predictions = []
    for _, train_samples, fold_samples in _iter_folds(observations, times, cutoffs, cache):
        model = make_candidate_b_model()
        model.fit(build_matrix(train_samples, codes, BASELINE_FEATURE_SCHEMA), train_samples["actual"].to_numpy(dtype=float))
        fold = fold_samples.copy()
        fold["prediction"] = np.maximum(0.0, model.predict(build_matrix(fold, codes, BASELINE_FEATURE_SCHEMA)))
        predictions.append(fold)
    frame = pd.concat(predictions, ignore_index=True)
    return full_metrics_result(
        name,
        frame,
        train_seconds=time.perf_counter() - start,
        complexity="Champion pipeline: HistGradientBoostingRegressor global (sin imputer, igual a produccion), reentrenado por fold diario.",
    )


def run_global_model(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    cutoffs: list[pd.Timestamp],
    *,
    feature_schema: list[str],
    family: str,
    model_kwargs: dict[str, Any] | None,
    name: str,
    cache: Cache,
    complexity: str,
) -> ModelResult:
    start = time.perf_counter()
    codes = station_code_map(observations)
    predictions = []
    for _, train_samples, fold_samples in _iter_folds(observations, times, cutoffs, cache):
        model = make_model(family, **(model_kwargs or {}))
        model.fit(build_matrix(train_samples, codes, feature_schema), train_samples["actual"].to_numpy(dtype=float))
        fold = fold_samples.copy()
        fold["prediction"] = np.maximum(0.0, model.predict(build_matrix(fold, codes, feature_schema)))
        predictions.append(fold)
    frame = pd.concat(predictions, ignore_index=True)
    return full_metrics_result(name, frame, train_seconds=time.perf_counter() - start, complexity=complexity)


def run_horizon_specific(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    cutoffs: list[pd.Timestamp],
    *,
    feature_schema: list[str],
    family: str,
    model_kwargs: dict[str, Any] | None,
    name: str,
    cache: Cache,
) -> ModelResult:
    start = time.perf_counter()
    codes = station_code_map(observations)
    predictions = []
    for _, train_samples, fold_samples in _iter_folds(observations, times, cutoffs, cache):
        fold = fold_samples.copy()
        fold["prediction"] = np.nan
        for horizon in HORIZONS:
            train_part = train_samples[train_samples["horizon_minutes"] == horizon]
            fold_mask = fold["horizon_minutes"] == horizon
            if train_part.empty or not fold_mask.any():
                continue
            model = make_model(family, **(model_kwargs or {}))
            model.fit(build_matrix(train_part, codes, feature_schema), train_part["actual"].to_numpy(dtype=float))
            fold.loc[fold_mask, "prediction"] = np.maximum(0.0, model.predict(build_matrix(fold.loc[fold_mask], codes, feature_schema)))
        predictions.append(fold)
    frame = pd.concat(predictions, ignore_index=True)
    return full_metrics_result(
        name, frame, train_seconds=time.perf_counter() - start, complexity="Media-alta: un modelo independiente por horizonte (+15/+30/+45/+60)."
    )


def run_station_specific(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    cutoffs: list[pd.Timestamp],
    *,
    feature_schema: list[str],
    family: str,
    name: str,
    cache: Cache,
    fallback_predictions: pd.DataFrame,
    min_train_rows: int = 200,
) -> ModelResult:
    start = time.perf_counter()
    codes = station_code_map(observations)
    stations = sorted(observations["station_id"].unique())
    predictions = []
    for _, train_samples, fold_samples in _iter_folds(observations, times, cutoffs, cache):
        fold = fold_samples.copy()
        fold["prediction"] = np.nan
        for station_id in stations:
            station_train = train_samples[train_samples["station_id"] == station_id]
            station_mask = fold["station_id"] == station_id
            if len(station_train) < min_train_rows or not station_mask.any():
                continue
            model = make_model(family)
            model.fit(build_matrix(station_train, codes, feature_schema), station_train["actual"].to_numpy(dtype=float))
            fold.loc[station_mask, "prediction"] = np.maximum(0.0, model.predict(build_matrix(fold.loc[station_mask], codes, feature_schema)))
        predictions.append(fold)
    frame = pd.concat(predictions, ignore_index=True)
    frame = frame.merge(
        fallback_predictions[["station_id", "target_at", "horizon_minutes", "prediction"]].rename(columns={"prediction": "fallback_prediction"}),
        on=["station_id", "target_at", "horizon_minutes"],
        how="left",
    )
    fallback_used = int(frame["prediction"].isna().sum())
    frame["prediction"] = frame["prediction"].fillna(frame["fallback_prediction"])
    frame = frame.drop(columns=["fallback_prediction"])
    return full_metrics_result(
        name,
        frame,
        train_seconds=time.perf_counter() - start,
        complexity=f"Alta: {len(stations)} modelos {family} independientes, uno por estacion (fallback usado en {fallback_used} filas).",
    )


def run_station_override_hybrid(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    cutoffs: list[pd.Timestamp],
    *,
    feature_schema: list[str],
    global_family: str,
    override_family: str,
    override_stations: Iterable[str],
    name: str,
    cache: Cache,
    min_train_rows: int = 64,
) -> ModelResult:
    start = time.perf_counter()
    codes = station_code_map(observations)
    override_stations = tuple(override_stations)
    predictions = []
    for _, train_samples, fold_samples in _iter_folds(observations, times, cutoffs, cache):
        global_model = make_model(global_family)
        global_model.fit(build_matrix(train_samples, codes, feature_schema), train_samples["actual"].to_numpy(dtype=float))
        fold = fold_samples.copy()
        fold["prediction"] = np.maximum(0.0, global_model.predict(build_matrix(fold, codes, feature_schema)))
        for station_id in override_stations:
            station_train = train_samples[train_samples["station_id"] == station_id]
            station_mask = fold["station_id"] == station_id
            if len(station_train) < min_train_rows or not station_mask.any():
                continue
            local_model = make_model(override_family)
            local_model.fit(build_matrix(station_train, codes, feature_schema), station_train["actual"].to_numpy(dtype=float))
            fold.loc[station_mask, "prediction"] = np.maximum(0.0, local_model.predict(build_matrix(fold.loc[station_mask], codes, feature_schema)))
        predictions.append(fold)
    frame = pd.concat(predictions, ignore_index=True)
    return full_metrics_result(
        name,
        frame,
        train_seconds=time.perf_counter() - start,
        complexity=f"Alta: {global_family} global + {override_family} local para {', '.join(override_stations)}.",
    )


def run_station_bias_correction(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    cutoffs: list[pd.Timestamp],
    *,
    group_cols: tuple[str, ...],
    cache: Cache,
    name: str,
    min_history_rows: int = 50,
) -> ModelResult:
    """Candidate B v2 (exact champion architecture) + an additive correction learned
    from EARLIER folds' own out-of-fold residuals, grouped by `group_cols`.

    Fold k's correction uses only rows from folds 1..k-1 -- each of those rows' own
    `raw_prediction` came from a model trained strictly before ITS fold -- so the
    correction for fold k never sees any information from fold k or later. Fold 1 gets
    no correction (cold start, correction=0), which is intentional and documented rather
    than hidden.
    """
    start = time.perf_counter()
    codes = station_code_map(observations)
    predictions = []
    history: pd.DataFrame | None = None
    for _, train_samples, fold_samples in _iter_folds(observations, times, cutoffs, cache):
        model = make_candidate_b_model()
        model.fit(build_matrix(train_samples, codes, BASELINE_FEATURE_SCHEMA), train_samples["actual"].to_numpy(dtype=float))
        fold = fold_samples.copy()
        fold["raw_prediction"] = np.maximum(0.0, model.predict(build_matrix(fold, codes, BASELINE_FEATURE_SCHEMA)))
        if history is not None:
            counts = history.groupby(list(group_cols)).size()
            bias = history.groupby(list(group_cols)).apply(lambda g: (g["actual"] - g["raw_prediction"]).mean(), include_groups=False)
            bias = bias.where(counts >= min_history_rows, 0.0)
            bias_frame = bias.rename("bias_correction").reset_index()
            fold = fold.merge(bias_frame, on=list(group_cols), how="left")
            fold["bias_correction"] = fold["bias_correction"].fillna(0.0)
        else:
            fold["bias_correction"] = 0.0
        fold["prediction"] = np.maximum(0.0, fold["raw_prediction"] + fold["bias_correction"])
        predictions.append(fold)
        history_chunk = fold[["station_id", "horizon_minutes", "actual", "raw_prediction"]]
        history = history_chunk if history is None else pd.concat([history, history_chunk], ignore_index=True)
    frame = pd.concat(predictions, ignore_index=True)
    return full_metrics_result(
        name,
        frame,
        train_seconds=time.perf_counter() - start,
        complexity=f"Media: Candidate B v2 + correccion aditiva de sesgo por {'+'.join(group_cols)} (aprendida de folds anteriores).",
    )


def run_profile_baseline(observations: pd.DataFrame, cutoffs: list[pd.Timestamp], cache: Cache) -> ModelResult:
    start = time.perf_counter()
    samples = cached_extended_samples(cache, observations, cutoffs).copy()
    samples["prediction"] = samples["lag_96"].fillna(samples["profile_mean"]).clip(lower=0)
    return full_metrics_result(
        "Profile baseline (lag96->profile)",
        samples,
        train_seconds=time.perf_counter() - start,
        complexity="Muy baja: persistencia diaria (lag_96) con fallback a perfil historico estacion x quarter-hour.",
    )


def run_global_model_with_context(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    cutoffs: list[pd.Timestamp],
    *,
    feature_schema: list[str],
    context: pd.DataFrame,
    context_columns: list[str],
    name: str,
    cache: Cache,
) -> ModelResult:
    """Research-only: joins context_observations as-of `data_cutoff` (context and
    observations share the exact same 15-minute grid locally, so the join key is simply
    `observed_at == data_cutoff`; no row with `observed_at > data_cutoff` is ever used).
    """
    start = time.perf_counter()
    codes = station_code_map(observations)
    context_indexed = context.set_index("observed_at")[context_columns]
    full_schema = feature_schema + context_columns
    predictions = []
    for _, train_samples, fold_samples in _iter_folds(observations, times, cutoffs, cache):
        train_ctx = train_samples.join(context_indexed, on="data_cutoff")
        fold_ctx = fold_samples.join(context_indexed, on="data_cutoff")
        model = make_model("hist_gradient_boosting")
        model.fit(build_matrix(train_ctx, codes, full_schema), train_ctx["actual"].to_numpy(dtype=float))
        fold = fold_ctx.copy()
        fold["prediction"] = np.maximum(0.0, model.predict(build_matrix(fold, codes, full_schema)))
        predictions.append(fold)
    frame = pd.concat(predictions, ignore_index=True)
    return full_metrics_result(
        name,
        frame,
        train_seconds=time.perf_counter() - start,
        complexity=f"Research-only: HGB + contexto ({', '.join(context_columns)}).",
    )


# --------------------------------------------------------------------------------------
# Ensembling: weights are fit by grid search on walk-forward predictions ONLY. Every
# input prediction frame here is itself already out-of-fold (produced by daily
# retraining that never saw its own fold), so fitting a blend on their concatenation is
# the same legitimacy as stacking on out-of-fold predictions in ordinary CV. The hard
# holdout is never passed into this function.
# --------------------------------------------------------------------------------------
def _simplex_grid(n: int, step: float = 0.1) -> list[tuple[float, ...]]:
    ticks = int(round(1.0 / step))
    results: list[tuple[float, ...]] = []

    def rec(remaining_ticks: int, remaining_slots: int, prefix: list[float]) -> None:
        if remaining_slots == 1:
            results.append(tuple(prefix + [round(remaining_ticks * step, 4)]))
            return
        for t in range(remaining_ticks + 1):
            rec(remaining_ticks - t, remaining_slots - 1, prefix + [round(t * step, 4)])

    rec(ticks, n, [])
    return results


def fit_ensemble_weights(wf_predictions: dict[str, pd.DataFrame], *, step: float = 0.1) -> tuple[dict[str, float], float]:
    names = list(wf_predictions)
    key_cols = ["station_id", "target_at", "horizon_minutes"]
    base = wf_predictions[names[0]][key_cols + ["actual"]].sort_values(key_cols).reset_index(drop=True)
    matrix_cols = []
    for name in names:
        aligned = wf_predictions[name][key_cols + ["prediction"]].sort_values(key_cols).reset_index(drop=True)
        if not aligned[key_cols].equals(base[key_cols]):
            raise RuntimeError(f"prediction rows for {name!r} are not aligned with {names[0]!r}")
        matrix_cols.append(aligned["prediction"].to_numpy(dtype=float))
    pred_matrix = np.column_stack(matrix_cols)

    best_weights = tuple([1.0 / len(names)] * len(names))
    best_score = -np.inf
    for combo in _simplex_grid(len(names), step=step):
        blended = pred_matrix @ np.array(combo)
        scored = base.assign(prediction=np.maximum(0.0, blended))
        score = float(official_metrics_full(scored)["mean_station_accuracy"])
        if score > best_score:
            best_score = score
            best_weights = combo
    return dict(zip(names, best_weights)), best_score


def _slice_to_cutoffs(frame: pd.DataFrame, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    """Restrict a (possibly WF+hard concatenated, e.g. from a bias-correction model
    that needs warm history) prediction frame down to exactly these data_cutoffs."""
    return frame[frame["data_cutoff"].isin(pd.DatetimeIndex(cutoffs))].copy()


def apply_ensemble(predictions: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    names = list(weights)
    key_cols = ["station_id", "target_at", "horizon_minutes"]
    # Carry data_cutoff through too (not part of the join key, just passenger data) so
    # downstream per-fold reporting (per_fold_metrics) works for the ensemble like it
    # does for every other candidate.
    base = predictions[names[0]][key_cols + ["actual", "data_cutoff"]].sort_values(key_cols).reset_index(drop=True)
    blended = np.zeros(len(base), dtype=float)
    for name in names:
        aligned = predictions[name][key_cols + ["prediction"]].sort_values(key_cols).reset_index(drop=True)
        if not aligned[key_cols].equals(base[key_cols]):
            raise RuntimeError(f"prediction rows for {name!r} are not aligned")
        blended += weights[name] * aligned["prediction"].to_numpy(dtype=float)
    result = base.copy()
    result["prediction"] = np.maximum(0.0, blended)
    return result


# --------------------------------------------------------------------------------------
# Error-analysis exports
# --------------------------------------------------------------------------------------
def per_fold_metrics(model_name: str, predictions: pd.DataFrame, *, split: str) -> pd.DataFrame:
    scored = predictions.copy()
    scored["fold_start"] = scored["data_cutoff"].dt.floor("D")
    rows = []
    for fold_start, group in scored.groupby("fold_start"):
        metrics = official_metrics_full(group)
        rows.append(
            {
                "model": model_name,
                "split": split,
                "fold_start": fold_start,
                "mean_station_accuracy": metrics["mean_station_accuracy"],
                "wape": metrics["wape"],
                "mae": metrics["mae"],
                "signed_relative_bias": metrics["signed_relative_bias"],
                "rows": metrics["rows"],
            }
        )
    return pd.DataFrame(rows).sort_values("fold_start")


def station_metrics_table(model_name: str, result: ModelResult, *, split: str) -> pd.DataFrame:
    frame = result.summary["station_metrics"].copy()
    frame.insert(0, "split", split)
    frame.insert(0, "model", model_name)
    return frame


def horizon_metrics_table(model_name: str, result: ModelResult, *, split: str) -> pd.DataFrame:
    frame = result.summary["horizon_metrics"].copy()
    frame.insert(0, "split", split)
    frame.insert(0, "model", model_name)
    return frame


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Challenger Lab v2 -- offline, read-only experiments. Never touches production.")
    parser.add_argument("--skip-context", action="store_true", help="skip the secondary context-research ablation")
    parser.add_argument("--ensemble-step", type=float, default=0.1)
    args = parser.parse_args()

    t_all = time.perf_counter()
    print("Loading local observations snapshot (data/raw/observations.csv)...", flush=True)
    observations = load_observations()
    times = sorted(observations["observed_at"].drop_duplicates())
    window = experiment_cutoffs(observations)
    wf_cutoffs = window.walk_forward_cutoffs
    hard_cutoffs = window.hard_holdout_cutoffs
    print(f"Walk-forward: {len(wf_cutoffs)} cutoffs [{wf_cutoffs[0]} .. {wf_cutoffs[-1]}]", flush=True)
    print(f"Hard holdout: {len(hard_cutoffs)} cutoffs [{hard_cutoffs[0]} .. {hard_cutoffs[-1]}]", flush=True)

    ext_cache: Cache = {}

    catalog: list[Candidate] = []

    # ---- Experiment A: Global baseline (exact Candidate B v2 reproduction; control) ----
    print("\n[A] Global baseline: reproducing Candidate B v2 exactly (control)...", flush=True)
    baseline_wf = run_candidate_b_baseline(observations, times, wf_cutoffs, name="Candidate B v2 (global baseline)", cache=ext_cache)
    baseline_candidate = Candidate(
        "Candidate B v2 (global baseline)",
        hypothesis="A: control -- exact champion architecture reproduced on D1's windows.",
        production_eligible=True,
        wf_result=baseline_wf,
        notes="Control. Active production model; not modified.",
    )
    catalog.append(baseline_candidate)
    print(f"    WF mean station accuracy: {baseline_wf.summary['mean_station_accuracy']:.4f}% (reference: 87.2798%)", flush=True)

    profile_baseline_wf = run_profile_baseline(observations, wf_cutoffs, ext_cache)
    catalog.append(
        Candidate(
            "Profile baseline (lag96->profile)",
            hypothesis="A: naive complementary baseline, used only as ensemble input.",
            production_eligible=True,
            wf_result=profile_baseline_wf,
            notes="Not a challenger by itself; complementary model for ensembling (E).",
        )
    )
    print(f"    Profile baseline WF accuracy: {profile_baseline_wf.summary['mean_station_accuracy']:.4f}%", flush=True)

    # ---- Hypothesis 4: do better temporal/live-safe features help the SAME global HGB? ----
    print("\n[Features] Ablation: Candidate B architecture + one new feature group at a time (WF only)...", flush=True)
    feature_variants = [
        ("B + lag_90/lag_120", FEATURE_SET_PLUS_LAGS),
        ("B + rolling stats (1h/2h/3h)", FEATURE_SET_PLUS_ROLLING),
        ("B + trend features", FEATURE_SET_PLUS_TREND),
        ("B + day-of-week profile", FEATURE_SET_PLUS_DOW_PROFILE),
        ("B + all new features", FEATURE_SET_ALL),
    ]
    feature_results: dict[str, ModelResult] = {}
    for label, schema in feature_variants:
        result = run_global_model(
            observations, times, wf_cutoffs,
            feature_schema=schema, family="hist_gradient_boosting", model_kwargs=None,
            name=label, cache=ext_cache,
            complexity="Media: HGB global, mismos hiperparametros que Candidate B, features ampliadas.",
        )
        feature_results[label] = result
        delta = result.summary["mean_station_accuracy"] - baseline_wf.summary["mean_station_accuracy"]
        print(f"    {label}: WF={result.summary['mean_station_accuracy']:.4f}% (delta={delta:+.4f} pp)", flush=True)
        catalog.append(
            Candidate(
                label,
                hypothesis="4: better temporal/live-safe features on the same global HGB architecture.",
                production_eligible=True,
                wf_result=result,
                notes="Feature-ablation screening row (WF only); not carried to hard holdout unless it wins the ablation.",
            )
        )

    best_feature_label = max(feature_results, key=lambda k: feature_results[k].summary["mean_station_accuracy"])
    best_feature_schema = dict(feature_variants)[best_feature_label]
    best_feature_beats_baseline = feature_results[best_feature_label].summary["mean_station_accuracy"] > baseline_wf.summary["mean_station_accuracy"]
    if best_feature_beats_baseline:
        print(f"    Best feature set: {best_feature_label!r} (beats baseline WF) -> used downstream for B/C/D.", flush=True)
    else:
        best_feature_schema = BASELINE_FEATURE_SCHEMA
        best_feature_label = "B (baseline features; no extended set beat it)"
        print("    No extended feature set beat the baseline on WF -- downstream experiments fall back to baseline features.", flush=True)

    # ---- Hypothesis 1: horizon-specific models vs one multi-horizon global model ----
    print("\n[B] Horizon-specific models (WF only)...", flush=True)
    horizon_result = run_horizon_specific(
        observations, times, wf_cutoffs,
        feature_schema=best_feature_schema, family="hist_gradient_boosting", model_kwargs=None,
        name="Horizon-specific HGB (+15/+30/+45/+60 independientes)", cache=ext_cache,
    )
    delta = horizon_result.summary["mean_station_accuracy"] - baseline_wf.summary["mean_station_accuracy"]
    print(f"    WF={horizon_result.summary['mean_station_accuracy']:.4f}% (delta={delta:+.4f} pp)", flush=True)
    catalog.append(
        Candidate(
            horizon_result.name,
            hypothesis="1: does splitting by horizon beat one multi-horizon global model?",
            production_eligible=True,
            wf_result=horizon_result,
            notes="Confirms/refutes Candidate C's earlier negative finding on horizon-specific models.",
        )
    )

    # ---- Hypothesis 2/3: station specialization / hybrid global+local ----
    print("\n[C] Station specialization and hybrid global+local (WF only)...", flush=True)
    c1_full_local = run_station_specific(
        observations, times, wf_cutoffs,
        feature_schema=best_feature_schema, family="hist_gradient_boosting",
        name="C1: full per-station HGB (12 modelos independientes)", cache=ext_cache,
        fallback_predictions=baseline_wf.predictions, min_train_rows=200,
    )
    print(f"    C1 full per-station: WF={c1_full_local.summary['mean_station_accuracy']:.4f}% (delta={c1_full_local.summary['mean_station_accuracy']-baseline_wf.summary['mean_station_accuracy']:+.4f} pp)", flush=True)

    c2_drift_hybrid = run_station_override_hybrid(
        observations, times, wf_cutoffs,
        feature_schema=best_feature_schema, global_family="hist_gradient_boosting", override_family="extra_trees",
        override_stations=DRIFT_FLAGGED_STATIONS,
        name="C2: global HGB + ExtraTrees local para estaciones con drift (05000/07111/03000/09122)",
        cache=ext_cache,
    )
    print(f"    C2 drift-station hybrid: WF={c2_drift_hybrid.summary['mean_station_accuracy']:.4f}% (delta={c2_drift_hybrid.summary['mean_station_accuracy']-baseline_wf.summary['mean_station_accuracy']:+.4f} pp)", flush=True)

    c3_station_bias = run_station_bias_correction(
        observations, times, wf_cutoffs,
        group_cols=("station_id",), cache=ext_cache,
        name="C3: Candidate B v2 + correccion de sesgo por estacion (OOF de folds previos)",
    )
    print(f"    C3 station bias correction: WF={c3_station_bias.summary['mean_station_accuracy']:.4f}% (delta={c3_station_bias.summary['mean_station_accuracy']-baseline_wf.summary['mean_station_accuracy']:+.4f} pp)", flush=True)

    c4_station_horizon_bias = run_station_bias_correction(
        observations, times, wf_cutoffs,
        group_cols=("station_id", "horizon_minutes"), cache=ext_cache,
        name="C4: Candidate B v2 + correccion de sesgo por estacion x horizonte (OOF de folds previos)",
    )
    print(f"    C4 station x horizon bias correction: WF={c4_station_horizon_bias.summary['mean_station_accuracy']:.4f}% (delta={c4_station_horizon_bias.summary['mean_station_accuracy']-baseline_wf.summary['mean_station_accuracy']:+.4f} pp)", flush=True)

    for cand_name, result, note in [
        (c1_full_local.name, c1_full_local, "Hypothesis 2: full station specialization."),
        (c2_drift_hybrid.name, c2_drift_hybrid, "Hypothesis 3: hybrid global+local, targeted at drift-flagged stations."),
        (c3_station_bias.name, c3_station_bias, "Hypothesis 2/3: cheapest possible local correction (station intercept)."),
        (c4_station_horizon_bias.name, c4_station_horizon_bias, "Hypothesis 2/3: finer-grained station x horizon intercept."),
    ]:
        catalog.append(Candidate(cand_name, hypothesis=note, production_eligible=True, wf_result=result))

    # ---- Hypothesis 6: does a different model family add signal on the same features? ----
    print("\n[D] Model family bake-off on the winning feature set (WF only)...", flush=True)
    family_results: dict[str, ModelResult] = {"hist_gradient_boosting": feature_results.get(best_feature_label, baseline_wf)}
    for family in ("extra_trees", "random_forest", "gradient_boosting", "ridge", "elastic_net"):
        result = run_global_model(
            observations, times, wf_cutoffs,
            feature_schema=best_feature_schema, family=family, model_kwargs=None,
            name=f"D: global {family}", cache=ext_cache,
            complexity=f"Media: {family} global sobre el mismo feature set ganador.",
        )
        family_results[family] = result
        print(f"    {family}: WF={result.summary['mean_station_accuracy']:.4f}% (delta={result.summary['mean_station_accuracy']-baseline_wf.summary['mean_station_accuracy']:+.4f} pp)", flush=True)
        catalog.append(Candidate(result.name, hypothesis="6: does a different model family add signal?", production_eligible=True, wf_result=result))

    # ---- Hypothesis 5: does blending complementary models beat every individual model? ----
    print("\n[E] Ensemble: fitting blend weights on walk-forward OOF predictions only...", flush=True)
    non_ensemble_candidates = [c for c in catalog if c.name != profile_baseline_wf.name]
    best_challenger = max(non_ensemble_candidates, key=lambda c: c.wf_result.summary["mean_station_accuracy"])
    print(f"    Best individual WF challenger so far: {best_challenger.name} ({best_challenger.wf_result.summary['mean_station_accuracy']:.4f}%)", flush=True)

    ensemble_inputs_wf = {baseline_candidate.name: baseline_wf.predictions, "Profile baseline": profile_baseline_wf.predictions}
    if best_challenger.name != baseline_candidate.name:
        ensemble_inputs_wf[best_challenger.name] = best_challenger.wf_result.predictions
    weights, blend_wf_score = fit_ensemble_weights(ensemble_inputs_wf, step=args.ensemble_step)
    print(f"    Fitted weights: {weights} -> WF blended accuracy={blend_wf_score:.4f}%", flush=True)
    ensemble_wf_predictions = apply_ensemble(ensemble_inputs_wf, weights)
    ensemble_wf = full_metrics_result(
        "E: Ensemble (weights fit on WF OOF only)",
        ensemble_wf_predictions,
        train_seconds=0.0,
        complexity=f"Blend de 3 modelos con pesos fijos {weights} (buscados solo en walk-forward).",
    )
    catalog.append(
        Candidate(
            ensemble_wf.name,
            hypothesis="5: does blending complementary models beat any single model?",
            production_eligible=True,
            wf_result=ensemble_wf,
            notes=f"Blend of: Candidate B v2, {best_challenger.name}, Profile baseline. Weights={weights}.",
        )
    )

    # ---- WF leaderboard / shortlist selection (hard holdout not touched yet) ----
    print("\n--- Walk-forward leaderboard (selection happens here; hard holdout not used) ---", flush=True)
    leaderboard = sorted(catalog, key=lambda c: c.wf_result.summary["mean_station_accuracy"], reverse=True)
    for cand in leaderboard[:8]:
        print(f"    {cand.wf_result.summary['mean_station_accuracy']:7.4f}%  {cand.name}", flush=True)

    shortlist_names = {baseline_candidate.name, best_challenger.name, ensemble_wf.name, c3_station_bias.name}
    shortlist = [c for c in catalog if c.name in shortlist_names]
    print(f"\n[Hard holdout] Frozen shortlist (evaluated ONCE, no further selection): {sorted(shortlist_names)}", flush=True)

    # ---- Hard holdout: exactly once, for the frozen shortlist ----
    hard_by_name: dict[str, ModelResult] = {}
    hard_by_name[baseline_candidate.name] = run_candidate_b_baseline(observations, times, hard_cutoffs, name=baseline_candidate.name, cache=ext_cache)

    if best_challenger.name != baseline_candidate.name:
        hard_by_name[best_challenger.name] = _rerun_candidate_on_hard_holdout(
            best_challenger, observations, times, wf_cutoffs, hard_cutoffs, ext_cache, best_feature_schema,
        )

    hard_by_name[c3_station_bias.name] = run_station_bias_correction(
        observations, times, wf_cutoffs + hard_cutoffs,
        group_cols=("station_id",), cache=ext_cache,
        name=c3_station_bias.name,
    )

    # Every entry in hard_by_name may span just hard_cutoffs (plain reruns) or
    # wf_cutoffs+hard_cutoffs (bias-correction models, which need warm history) --
    # normalize all of them down to hard-only rows before doing anything else with them.
    for key, result in list(hard_by_name.items()):
        hard_by_name[key] = full_metrics_result(result.name, _slice_to_cutoffs(result.predictions, hard_cutoffs), train_seconds=result.train_seconds, complexity=result.complexity)

    profile_baseline_hard = run_profile_baseline(observations, hard_cutoffs, ext_cache)
    ensemble_inputs_hard = {baseline_candidate.name: hard_by_name[baseline_candidate.name].predictions, "Profile baseline": profile_baseline_hard.predictions}
    if best_challenger.name != baseline_candidate.name:
        ensemble_inputs_hard[best_challenger.name] = hard_by_name[best_challenger.name].predictions
    assert set(ensemble_inputs_hard) == set(weights), (set(ensemble_inputs_hard), set(weights))
    ensemble_hard_predictions = apply_ensemble(ensemble_inputs_hard, weights)
    hard_by_name[ensemble_wf.name] = full_metrics_result(ensemble_wf.name, ensemble_hard_predictions, train_seconds=0.0, complexity=ensemble_wf.complexity)

    for cand in shortlist:
        if cand.name in hard_by_name:
            cand.hard_result = hard_by_name[cand.name]
            hard_acc = cand.hard_result.summary["mean_station_accuracy"]
            delta_hard = hard_acc - hard_by_name[baseline_candidate.name].summary["mean_station_accuracy"]
            print(f"    {cand.name}: hard={hard_acc:.4f}% (delta vs B={delta_hard:+.4f} pp)", flush=True)

    # ---- Secondary track: context research-only ablation (WF only, clearly labeled) ----
    context_candidates: list[Candidate] = []
    if not args.skip_context:
        print("\n[D2] Secondary track: context research-only ablation (WF only, production_eligible=False)...", flush=True)
        context = load_context()
        safe_historical = ["rain_mm", "temperature_c"]
        research_extra = ["rain_forecast", "temperature_forecast", "event_intensity"]

        b_plus_hist_ctx = run_global_model_with_context(
            observations, times, wf_cutoffs,
            feature_schema=BASELINE_FEATURE_SCHEMA, context=context, context_columns=safe_historical,
            name="D2: B + safe historical context (rain_mm, temperature_c @ cutoff)", cache=ext_cache,
        )
        print(f"    B + safe historical context: WF={b_plus_hist_ctx.summary['mean_station_accuracy']:.4f}% (delta={b_plus_hist_ctx.summary['mean_station_accuracy']-baseline_wf.summary['mean_station_accuracy']:+.4f} pp)", flush=True)
        context_candidates.append(
            Candidate(
                b_plus_hist_ctx.name, hypothesis="D2 ablation: does actual historical weather at cutoff help?",
                production_eligible=False, research_only=True, wf_result=b_plus_hist_ctx,
                notes="Uses only rain_mm/temperature_c with observed_at<=data_cutoff. Not production_eligible: context sync is not live (see reports/candidate_d2).",
            )
        )

        b_plus_temporal = feature_results.get(best_feature_label) or baseline_wf
        context_candidates.append(
            Candidate(
                f"D2: B + temporal/live-safe features ({best_feature_label})", hypothesis="D2 ablation reference: temporal features alone.",
                production_eligible=best_feature_beats_baseline, research_only=False, wf_result=b_plus_temporal,
                notes="Same row as the winning feature-ablation candidate; repeated here for the ablation chain.",
            )
        )

        b_plus_temporal_plus_research_ctx = run_global_model_with_context(
            observations, times, wf_cutoffs,
            feature_schema=best_feature_schema, context=context, context_columns=safe_historical + research_extra,
            name="D2: B + temporal + research context (+forecast/event, NOT proven live)", cache=ext_cache,
        )
        print(f"    B + temporal + research context: WF={b_plus_temporal_plus_research_ctx.summary['mean_station_accuracy']:.4f}% (delta={b_plus_temporal_plus_research_ctx.summary['mean_station_accuracy']-baseline_wf.summary['mean_station_accuracy']:+.4f} pp)", flush=True)
        context_candidates.append(
            Candidate(
                b_plus_temporal_plus_research_ctx.name, hypothesis="D2 ablation: full research stack (upper bound of context's potential).",
                production_eligible=False, research_only=True, wf_result=b_plus_temporal_plus_research_ctx,
                notes="Uses rain_forecast/temperature_forecast/event_intensity, none of which have proven live availability (no issued_at/available_at). research_only=true.",
            )
        )
        catalog.extend(context_candidates)

    write_outputs(
        observations=observations,
        window=window,
        catalog=catalog,
        baseline_candidate=baseline_candidate,
        best_challenger_name=best_challenger.name,
        best_feature_label=best_feature_label,
        weights=weights,
    )

    print(f"\nTotal wall time: {time.perf_counter() - t_all:.1f}s", flush=True)
    print(f"Report written to {REPORT_DIR / 'summary.md'}", flush=True)


def _rerun_candidate_on_hard_holdout(
    candidate: Candidate,
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    wf_cutoffs: list[pd.Timestamp],
    hard_cutoffs: list[pd.Timestamp],
    cache: Cache,
    best_feature_schema: list[str],
) -> ModelResult:
    """Re-run whichever runner produced `candidate` on the hard-holdout window. Kept as
    a single dispatch point so the hard holdout is unmistakably invoked from one place,
    exactly once per candidate, for the frozen shortlist only.

    C3/C4 (bias-correction models) need warm history to be meaningful, so they are run
    on `wf_cutoffs + hard_cutoffs` (chronological, so every fold's correction still only
    ever uses strictly-earlier folds) and the caller slices the result down to the
    hard-only rows -- exactly the same slicing already applied to every shortlisted
    candidate right after this function returns.
    """
    name = candidate.name
    if name.startswith("Horizon-specific"):
        return run_horizon_specific(
            observations, times, hard_cutoffs,
            feature_schema=best_feature_schema, family="hist_gradient_boosting", model_kwargs=None,
            name=name, cache=cache,
        )
    if name.startswith("C1:"):
        fallback = run_candidate_b_baseline(observations, times, hard_cutoffs, name="fallback", cache=cache)
        return run_station_specific(
            observations, times, hard_cutoffs,
            feature_schema=best_feature_schema, family="hist_gradient_boosting",
            name=name, cache=cache, fallback_predictions=fallback.predictions, min_train_rows=200,
        )
    if name.startswith("C2:"):
        return run_station_override_hybrid(
            observations, times, hard_cutoffs,
            feature_schema=best_feature_schema, global_family="hist_gradient_boosting", override_family="extra_trees",
            override_stations=DRIFT_FLAGGED_STATIONS, name=name, cache=cache,
        )
    if name.startswith("C3:"):
        return run_station_bias_correction(observations, times, wf_cutoffs + hard_cutoffs, group_cols=("station_id",), cache=cache, name=name)
    if name.startswith("C4:"):
        return run_station_bias_correction(observations, times, wf_cutoffs + hard_cutoffs, group_cols=("station_id", "horizon_minutes"), cache=cache, name=name)
    if name.startswith("D: global "):
        family = name.removeprefix("D: global ")
        return run_global_model(
            observations, times, hard_cutoffs,
            feature_schema=best_feature_schema, family=family, model_kwargs=None,
            name=name, cache=cache, complexity="",
        )
    if name.startswith("B + ") or name == "B (baseline features; no extended set beat it)":
        schema = best_feature_schema
        return run_global_model(
            observations, times, hard_cutoffs,
            feature_schema=schema, family="hist_gradient_boosting", model_kwargs=None,
            name=name, cache=cache, complexity="",
        )
    raise RuntimeError(f"no hard-holdout dispatch registered for candidate {name!r}")


def write_outputs(
    *,
    observations: pd.DataFrame,
    window,
    catalog: list[Candidate],
    baseline_candidate: Candidate,
    best_challenger_name: str,
    best_feature_label: str,
    weights: dict[str, float],
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_wf_acc = baseline_candidate.wf_result.summary["mean_station_accuracy"]
    baseline_hard_acc = baseline_candidate.hard_result.summary["mean_station_accuracy"] if baseline_candidate.hard_result else None

    comparison_rows = []
    fold_frames = []
    station_frames = []
    horizon_frames = []
    for cand in catalog:
        wf = cand.wf_result
        hard = cand.hard_result
        comparison_rows.append(
            {
                "model": cand.name,
                "hypothesis": cand.hypothesis,
                "production_eligible": cand.production_eligible,
                "research_only": cand.research_only,
                "wf_mean_station_accuracy": wf.summary["mean_station_accuracy"],
                "delta_vs_b_wf_pp": wf.summary["mean_station_accuracy"] - baseline_wf_acc,
                "hard_holdout_accuracy": hard.summary["mean_station_accuracy"] if hard else None,
                "delta_vs_b_holdout_pp": (hard.summary["mean_station_accuracy"] - baseline_hard_acc) if hard and baseline_hard_acc is not None else None,
                "wf_wape": wf.summary["wape"],
                "wf_mae": wf.summary["mae"],
                "wf_signed_relative_bias": wf.summary["signed_relative_bias"],
                "hard_wape": hard.summary["wape"] if hard else None,
                "hard_mae": hard.summary["mae"] if hard else None,
                "hard_signed_relative_bias": hard.summary["signed_relative_bias"] if hard else None,
                "train_seconds": wf.train_seconds,
                "notes": cand.notes,
            }
        )
        fold_frames.append(per_fold_metrics(cand.name, wf.predictions, split="walk_forward"))
        station_frames.append(station_metrics_table(cand.name, wf, split="walk_forward"))
        horizon_frames.append(horizon_metrics_table(cand.name, wf, split="walk_forward"))
        if hard is not None:
            fold_frames.append(per_fold_metrics(cand.name, hard.predictions, split="hard_holdout"))
            station_frames.append(station_metrics_table(cand.name, hard, split="hard_holdout"))
            horizon_frames.append(horizon_metrics_table(cand.name, hard, split="hard_holdout"))

    comparison = pd.DataFrame(comparison_rows).sort_values("wf_mean_station_accuracy", ascending=False)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(REPORT_DIR / "fold_metrics.csv", index=False)
    pd.concat(station_frames, ignore_index=True).to_csv(REPORT_DIR / "station_metrics.csv", index=False)
    pd.concat(horizon_frames, ignore_index=True).to_csv(REPORT_DIR / "horizon_metrics.csv", index=False)

    # Focused special-station / special-horizon tables for the best shortlisted challenger.
    best_shortlisted = max((c for c in catalog if c.hard_result is not None and c.name != baseline_candidate.name), key=lambda c: c.hard_result.summary["mean_station_accuracy"], default=None)
    if best_shortlisted is not None and baseline_candidate.hard_result is not None:
        special_comparison(baseline_candidate.hard_result, best_shortlisted.hard_result, group="station").to_csv(TABLE_DIR / "special_station_comparison_hard.csv", index=False)
        special_comparison(baseline_candidate.hard_result, best_shortlisted.hard_result, group="horizon").to_csv(TABLE_DIR / "special_horizon_comparison_hard.csv", index=False)
    special_comparison(baseline_candidate.wf_result, next(c for c in catalog if c.name == best_challenger_name).wf_result, group="station").to_csv(TABLE_DIR / "special_station_comparison_wf.csv", index=False)
    special_comparison(baseline_candidate.wf_result, next(c for c in catalog if c.name == best_challenger_name).wf_result, group="horizon").to_csv(TABLE_DIR / "special_horizon_comparison_wf.csv", index=False)

    leakage_audit = temporal_fold_audit_generic(observations, window.walk_forward_cutoffs + window.hard_holdout_cutoffs)
    leakage_audit.to_csv(TABLE_DIR / "temporal_fold_audit.csv", index=False)
    leakage_ok = bool((leakage_audit["training_end"] < leakage_audit["validation_start"]).all())

    best_row = comparison.iloc[0]
    best_overall_hard = comparison.dropna(subset=["hard_holdout_accuracy"]).sort_values("hard_holdout_accuracy", ascending=False)
    best_hard_row = best_overall_hard.iloc[0] if not best_overall_hard.empty else None

    payload = {
        "lab": "challenger_lab_v2",
        "scope": "experimental_only_no_production_changes",
        "target_accuracy": TARGET_ACCURACY,
        "data_min": observations["observed_at"].min().isoformat().replace("+00:00", "Z"),
        "data_max": observations["observed_at"].max().isoformat().replace("+00:00", "Z"),
        "stations": int(observations["station_id"].nunique()),
        "walk_forward_start": min(window.walk_forward_cutoffs).isoformat().replace("+00:00", "Z"),
        "walk_forward_end": max(window.walk_forward_cutoffs).isoformat().replace("+00:00", "Z"),
        "walk_forward_cutoffs": len(window.walk_forward_cutoffs),
        "hard_holdout_start": min(window.hard_holdout_cutoffs).isoformat().replace("+00:00", "Z"),
        "hard_holdout_end": max(window.hard_holdout_cutoffs).isoformat().replace("+00:00", "Z"),
        "hard_holdout_cutoffs": len(window.hard_holdout_cutoffs),
        "windows_reused_from": "candidate_d1_residual_calibration.experiment_cutoffs (unchanged)",
        "baseline_wf_accuracy": baseline_wf_acc,
        "baseline_hard_accuracy": baseline_hard_acc,
        "best_wf_model": str(best_row["model"]),
        "best_wf_accuracy": float(best_row["wf_mean_station_accuracy"]),
        "best_hard_model": str(best_hard_row["model"]) if best_hard_row is not None else None,
        "best_hard_accuracy": float(best_hard_row["hard_holdout_accuracy"]) if best_hard_row is not None else None,
        "gap_to_target_pp": TARGET_ACCURACY - float(best_hard_row["hard_holdout_accuracy"]) if best_hard_row is not None else None,
        "best_feature_set": best_feature_label,
        "ensemble_weights": weights,
        "leakage_audit_passed": leakage_ok,
        "any_model_promoted": False,
        "model_comparison": json.loads(comparison.to_json(orient="records")),
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_markdown(payload, comparison, leakage_audit))
    render_leakage_audit_md(leakage_audit, leakage_ok)


def render_markdown(payload: dict[str, Any], comparison: pd.DataFrame, leakage_audit: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# Challenger Lab v2",
        "",
        "## Scope",
        "",
        "- Experimental only. Champion (Candidate B v2), Predict, the scheduler, submissions, and Supabase production state were not modified, and nothing here was promoted.",
        "- All data is the local `data/raw/observations.csv` / `context.csv` snapshot -- the same source Candidate D1 used. No Supabase calls were made.",
        "- Temporal windows are Candidate D1's exact, previously-audited walk-forward and hard-holdout windows (`candidate_d1_residual_calibration.experiment_cutoffs`), reused unchanged.",
        "- Model, feature and ensemble-weight selection happened only on walk-forward. The hard holdout was evaluated exactly once, for a shortlist frozen before it was looked at.",
        "",
        "## Windows",
        "",
        f"- Observations: {payload['data_min']} to {payload['data_max']} ({payload['stations']} stations).",
        f"- Walk-forward: {payload['walk_forward_start']} to {payload['walk_forward_end']} ({payload['walk_forward_cutoffs']} hourly cutoffs / 7 days).",
        f"- Hard holdout: {payload['hard_holdout_start']} to {payload['hard_holdout_end']} ({payload['hard_holdout_cutoffs']} hourly cutoffs / 1 day).",
        "",
        "## Full model comparison (walk-forward selection; hard holdout only for the frozen shortlist)",
        "",
        _markdown_table(comparison),
        "",
        "## Best result",
        "",
        f"- Best walk-forward model: **{payload['best_wf_model']}** at {fmt(payload['best_wf_accuracy'])}%.",
        f"- Best hard-holdout model (frozen shortlist only): **{payload['best_hard_model']}** at {fmt(payload['best_hard_accuracy'])}%.",
        f"- Candidate B v2 control: WF {fmt(payload['baseline_wf_accuracy'])}%, hard holdout {fmt(payload['baseline_hard_accuracy'])}%.",
        f"- Gap to the {payload['target_accuracy']}% experimental target (hard holdout): {fmt(payload['gap_to_target_pp'])} pp.",
        f"- Winning feature set from the ablation: {payload['best_feature_set']}.",
        f"- Ensemble weights (fit on walk-forward OOF only): {payload['ensemble_weights']}.",
        "",
        "## Leakage audit",
        "",
        f"- `leakage_audit_passed = {str(payload['leakage_audit_passed']).lower()}` (training_end < validation_start for every fold; see `leakage_audit.md` and `tables/temporal_fold_audit.csv`).",
        "- Full detail: `reports/challenger_lab_v2/leakage_audit.md`.",
        "",
        "## Decision",
        "",
        f"- `any_model_promoted = false`. Nothing here replaces or overrides Candidate B v2 in production.",
        _recommendation(comparison, payload),
        "",
        "## Artifacts",
        "",
        "- `reports/challenger_lab_v2/summary.md`, `summary.json`",
        "- `reports/challenger_lab_v2/model_comparison.csv`, `fold_metrics.csv`, `station_metrics.csv`, `horizon_metrics.csv`",
        "- `reports/challenger_lab_v2/leakage_audit.md`",
        "- `reports/challenger_lab_v2/tables/*.csv` (special-station / special-horizon focus tables, temporal fold audit)",
    ]
    return "\n".join(lines) + "\n"


def _recommendation(comparison: pd.DataFrame, payload: dict[str, Any]) -> str:
    if payload["best_hard_accuracy"] is None:
        return "RECOMMENDATION: NOT PROMISING -- no shortlisted candidate had a hard-holdout evaluation."
    if payload["best_hard_accuracy"] <= payload["baseline_hard_accuracy"] + 1e-9:
        return (
            "RECOMMENDATION: NOT PROMISING. No shortlisted challenger improved the hard holdout over Candidate B v2. "
            "The champion should remain unchanged; see 'next best experiment' guidance in the final summary message."
        )
    return (
        f"RECOMMENDATION: PROMISING for further shadow evaluation ({payload['best_hard_model']}), but NOT promoted here. "
        "It should run in shadow across multiple real forecast cycles before any promotion decision."
    )


def render_leakage_audit_md(leakage_audit: pd.DataFrame, leakage_ok: bool) -> None:
    lines = [
        "# Challenger Lab v2 -- Leakage Audit",
        "",
        "## Automated checks (see tests/test_challenger_lab_v2.py for the pytest versions)",
        "",
        f"1. **training_end < validation_start for every fold**: {'PASSED' if leakage_ok else 'FAILED'}. See `tables/temporal_fold_audit.csv`.",
        "2. **No future target participates in any lag**: `lag_15..lag_120` are per-station `shift()`s of strictly earlier rows; `lag_15/30/45/60` are additionally masked to NaN whenever `lag < horizon_minutes`, and `lag_90/120` never need masking because both are >= the max horizon (60 min).",
        "3. **Rolling windows contain only the past**: every rolling feature is computed as `shift(horizon_steps).rolling(window)` *inside* the per-horizon loop, so the window ends at `data_cutoff = target_at - horizon`, never at `target_at`. Verified empirically: mutating an observation strictly between `data_cutoff` and `target_at` changes zero features for that row; mutating the observation AT `data_cutoff` changes multiple features (so the features are live, not accidentally constant).",
        "4. **Historical profiles use only training history**: `profile_mean/median` (station x quarter-hour) and `profile_dow_mean/median` (station x day-of-week x quarter-hour) use `groupby(...).shift(1).expanding()`, which excludes the current row and only aggregates strictly earlier rows in the same group.",
        "5. **No aggregate includes its own target row**: same `shift(1)` mechanism as (4); verified empirically by mutating the target row itself and confirming zero features change.",
        "6. **Feature/architecture selection never used the hard holdout**: the feature ablation (hypothesis 4), horizon-specific vs global (B), station specialization/hybrid (C), and the model-family bake-off (D) were all scored on walk-forward only. The hard holdout is invoked exactly once, from a single dispatch point (`_rerun_candidate_on_hard_holdout`), for a shortlist frozen before any hard-holdout number existed.",
        "7. **Ensemble weights never used the hard holdout**: `fit_ensemble_weights` takes only walk-forward prediction frames as input; it has no hard-holdout parameter at all. The frozen weights are applied as-is (`apply_ensemble`) to the hard-holdout predictions afterwards.",
        "8. **Station/fold statistics are computed fold-by-fold, not globally**: the per-station bias correction (C3/C4) only ever uses `history` accumulated from strictly earlier folds (see `run_station_bias_correction`); it is never computed once over the full dataset.",
        "9. **Hard holdout stays isolated during selection**: confirmed by (6); the only place the hard holdout window is read is inside the final, single hard-holdout evaluation block of `challenger_lab_v2.main()`.",
        "",
        "## Fixes to prior leakage found during this lab",
        "",
        "- None. Candidate D1's residual-target leakage (in-sample `prediction_B`) was already found and fixed before this lab started; this lab reused D1's out-of-fold pattern (see `run_station_bias_correction`, which accumulates only strictly-earlier-fold residuals) rather than re-introducing it.",
    ]
    (REPORT_DIR / "leakage_audit.md").write_text("\n".join(lines) + "\n")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("n/a" if pd.isna(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
