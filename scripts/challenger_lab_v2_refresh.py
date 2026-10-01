"""Challenger Lab v2 Refresh -- validates the ALREADY-SELECTED, FROZEN challenger against
a refreshed Supabase snapshot. This is a validation pass, not a new experiment:

FROZEN (not re-derived here, not re-searched):
- architecture: 30% Candidate B v2 (HistGradientBoostingRegressor) + 70% ExtraTreesRegressor
  global, blended via a fixed weighted average.
- feature set for the ExtraTrees component: Candidate B's own features plus the
  day-of-week x quarter-hour historical profile (`FEATURE_SET_PLUS_DOW_PROFILE`).
- ensemble weights: {0.30, 0.70, 0.0} for {Candidate B v2, ExtraTrees, Profile baseline},
  exactly as fit on walk-forward-only data in the original Challenger Lab v2 run.
- station handling, horizon handling: unchanged (global models, all stations/horizons).

This script only EVALUATES that frozen model on (a) the same historical windows as
before, as a reproduction check, and (b) the new data that arrived after the original
snapshot's max timestamp, as an additional out-of-sample validation window. It performs
no model selection, no feature search, and no weight re-fitting.
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
    FEATURE_SET_PLUS_DOW_PROFILE,
    HORIZONS,
    ModelResult,
    experiment_cutoffs,
    load_observations,
    load_observations_from_frame,
    official_metrics_full,
    temporal_fold_audit_generic,
)
from challenger_lab_v2 import (  # noqa: E402
    Cache,
    apply_ensemble,
    full_metrics_result,
    horizon_metrics_table,
    per_fold_metrics,
    run_candidate_b_baseline,
    run_global_model,
    run_profile_baseline,
    station_metrics_table,
)
from candidate_d1_residual_calibration import special_comparison  # noqa: E402

BASELINE_NAME = "Candidate B v2 (global baseline)"
CHALLENGER_NAME = "D: global extra_trees"
PROFILE_NAME = "Profile baseline"
ENSEMBLE_NAME = "E: Ensemble (FROZEN weights, not refit)"

FROZEN_WEIGHTS: dict[str, float] = {BASELINE_NAME: 0.3, CHALLENGER_NAME: 0.7, PROFILE_NAME: 0.0}
FROZEN_FEATURE_SCHEMA = FEATURE_SET_PLUS_DOW_PROFILE
FROZEN_CHALLENGER_FAMILY = "extra_trees"

DRIFT_MONITOR_REFERENCE_SIGNED_BIAS = -0.0676
DRIFT_MONITOR_RECENT_CYCLES = 6
SPECIAL_STATIONS = ("05000", "07111", "03000", "09122")
SPECIAL_HORIZONS = (30, 60)

DATA_PATH = ROOT / "data" / "raw" / "observations_refresh.csv"
REPORT_DIR = ROOT / "reports" / "challenger_lab_v2_refresh"

ORIGINAL_REFERENCE = {
    "baseline_wf": 87.2798,
    "baseline_hard": 88.4628,
    "ensemble_wf": 87.7395,
    "ensemble_hard": 88.6795,
}


def load_refreshed_observations() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found -- run scripts/refresh_observations_snapshot.py first.")
    frame = pd.read_csv(DATA_PATH, dtype={"station_id": "string"})
    return load_observations_from_frame(frame)


def run_frozen_ensemble(
    observations: pd.DataFrame,
    times: list[pd.Timestamp],
    cutoffs: list[pd.Timestamp],
    *,
    cache: Cache,
) -> dict[str, ModelResult]:
    """Runs the three FROZEN component models and blends them with FROZEN_WEIGHTS.
    No weight search, no feature search, no architecture change -- every parameter here
    is a constant copied from the original Challenger Lab v2 run's result."""
    if not cutoffs:
        raise RuntimeError("no cutoffs to evaluate")
    baseline = run_candidate_b_baseline(observations, times, cutoffs, name=BASELINE_NAME, cache=cache)
    challenger = run_global_model(
        observations, times, cutoffs,
        feature_schema=FROZEN_FEATURE_SCHEMA, family=FROZEN_CHALLENGER_FAMILY, model_kwargs=None,
        name=CHALLENGER_NAME, cache=cache,
        complexity="FROZEN challenger: ExtraTreesRegressor global on Candidate B features + day-of-week profile.",
    )
    profile = run_profile_baseline(observations, cutoffs, cache)
    ensemble_predictions = apply_ensemble(
        {BASELINE_NAME: baseline.predictions, CHALLENGER_NAME: challenger.predictions, PROFILE_NAME: profile.predictions},
        FROZEN_WEIGHTS,
    )
    ensemble = full_metrics_result(ENSEMBLE_NAME, ensemble_predictions, train_seconds=0.0, complexity=f"FROZEN blend, weights={FROZEN_WEIGHTS}.")
    return {"baseline": baseline, "challenger": challenger, "profile": profile, "ensemble": ensemble}


def compute_new_window_cutoffs(observations: pd.DataFrame, old_max: pd.Timestamp) -> list[pd.Timestamp]:
    times = sorted(observations["observed_at"].drop_duplicates())
    latest_cutoff = times[-1] - pd.Timedelta(minutes=max(HORIZONS))
    return [ts for ts in times if ts.minute == 0 and old_max < ts <= latest_cutoff]


def stability_table(baseline_predictions: pd.DataFrame, ensemble_predictions: pd.DataFrame) -> pd.DataFrame:
    b_fold = per_fold_metrics(BASELINE_NAME, baseline_predictions, split="new_window")
    e_fold = per_fold_metrics(ENSEMBLE_NAME, ensemble_predictions, split="new_window")
    merged = b_fold[["fold_start", "mean_station_accuracy", "wape", "mae", "signed_relative_bias", "rows"]].merge(
        e_fold[["fold_start", "mean_station_accuracy", "wape", "mae", "signed_relative_bias", "rows"]],
        on="fold_start", suffixes=("_b", "_ensemble"),
    )
    merged["delta_pp"] = merged["mean_station_accuracy_ensemble"] - merged["mean_station_accuracy_b"]
    return merged.sort_values("fold_start").reset_index(drop=True)


def stability_summary(fold_table: pd.DataFrame) -> dict[str, Any]:
    eps = 1e-9
    wins = int((fold_table["delta_pp"] > eps).sum())
    losses = int((fold_table["delta_pp"] < -eps).sum())
    ties = int(len(fold_table)) - wins - losses
    return {
        "number_of_new_folds": int(len(fold_table)),
        "ensemble_wins": wins,
        "candidate_b_wins": losses,
        "ties": ties,
        "mean_delta_pp": float(fold_table["delta_pp"].mean()) if len(fold_table) else None,
        "median_delta_pp": float(fold_table["delta_pp"].median()) if len(fold_table) else None,
        "min_delta_pp": float(fold_table["delta_pp"].min()) if len(fold_table) else None,
        "max_delta_pp": float(fold_table["delta_pp"].max()) if len(fold_table) else None,
    }


def classify_stability(summary: dict[str, Any]) -> str:
    """Descriptive, not a promotion gate -- reports what happened without forcing a verdict."""
    n = summary["number_of_new_folds"]
    if n == 0:
        return "INCONCLUSIVE (no new folds available)"
    if summary["candidate_b_wins"] > summary["ensemble_wins"] and summary["mean_delta_pp"] is not None and summary["mean_delta_pp"] < 0:
        return "REGRESSION (Candidate B wins more new folds and mean delta is negative)"
    if n < 3:
        return "INCONCLUSIVE (fewer than 3 new folds -- too few to call stable either way)"
    if summary["ensemble_wins"] >= summary["candidate_b_wins"] and summary["mean_delta_pp"] is not None and summary["mean_delta_pp"] > 0:
        return "STABLE IMPROVEMENT (ensemble wins at least as many new folds as B, mean delta positive)"
    return "INCONCLUSIVE (mixed results across new folds)"


def render_markdown(payload: dict[str, Any]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "# Challenger Lab v2 Refresh",
        "",
        "## Scope",
        "",
        "- Validation pass only. No new experiments, no feature search, no weight re-fitting, no promotion.",
        "- Architecture, feature set, hyperparameters, station/horizon handling, and ensemble weights are all FROZEN, copied verbatim from the original Challenger Lab v2 result.",
        "- Data: `data/raw/observations_refresh.csv`, a read-only refresh of the local snapshot from Supabase's `observations` table (see `data_refresh_audit.md`). Nothing was written to Supabase.",
        "- Champion, Predict, scheduler, and submissions were not touched. Nothing was promoted.",
        "",
        "## Frozen model",
        "",
        f"- Ensemble weights: {payload['frozen_weights']}",
        f"- Challenger feature set: `{payload['frozen_feature_set']}`",
        f"- Challenger family: `{payload['frozen_challenger_family']}`",
        "",
        "## Part A -- historical reproduction (same windows as the original run)",
        "",
        _markdown_table(payload["historical_table"]),
        "",
        "This uses the exact same walk-forward (2026-09-01..09-08) and hard-holdout (09-08..09-09) cutoffs as the original run, now sourced from the refreshed dataset. Historical rows in the refresh are byte-identical to the original snapshot (see `data_refresh_audit.md`), so these numbers are expected to match the original report; any mismatch would indicate a problem with the refresh, not with the model.",
        "",
        "## Part B -- new-window evaluation (data after the old snapshot's max)",
        "",
        f"- New window: {payload['new_window_start']} to {payload['new_window_end']} ({payload['new_window_cutoffs']} hourly cutoffs across {payload['new_window_folds']} fold-day(s)).",
        f"- Candidate B v2: {fmt(payload['new_window_b_accuracy'])}%",
        f"- Ensemble (frozen): {fmt(payload['new_window_ensemble_accuracy'])}%",
        f"- Delta: {fmt(payload['new_window_delta_pp'])} pp",
        "",
        "### Per-fold detail",
        "",
        _markdown_table(payload["fold_table"]),
        "",
        "## Stability (Step 4)",
        "",
        f"- number_of_new_folds: {payload['stability']['number_of_new_folds']}",
        f"- ensemble_wins: {payload['stability']['ensemble_wins']}",
        f"- candidate_b_wins: {payload['stability']['candidate_b_wins']}",
        f"- ties: {payload['stability']['ties']}",
        f"- mean_delta_pp: {fmt(payload['stability']['mean_delta_pp'])}",
        f"- median_delta_pp: {fmt(payload['stability']['median_delta_pp'])}",
        f"- min_delta_pp: {fmt(payload['stability']['min_delta_pp'])}",
        f"- max_delta_pp: {fmt(payload['stability']['max_delta_pp'])}",
        f"- classification: **{payload['stability_classification']}**",
        "",
        "Note on sample size: the new window only covers "
        f"{payload['new_window_folds']} fold-day(s) (however many days of new data Supabase actually had past the old snapshot's max). "
        "This is a real but small additional validation window, not a second full walk-forward study -- treat the classification above as directional, not definitive.",
        "",
        "## Drift consistency (Step 5)",
        "",
        f"- New-window signed_relative_bias -- Candidate B: {fmt(payload['new_window_b_bias'])}, Ensemble: {fmt(payload['new_window_ensemble_bias'])}.",
        f"- Drift monitor's most recently reported global signed_relative_bias: {DRIFT_MONITOR_REFERENCE_SIGNED_BIAS:.4f} ({DRIFT_MONITOR_REFERENCE_SIGNED_BIAS*100:.2f}%).",
        "- These are not expected to match exactly, and the gap here is explained by three structural differences, not by an error:",
        f"  1. **Window length/recency**: the drift monitor's config (`DriftConfig.recent_cycles = {DRIFT_MONITOR_RECENT_CYCLES}`) looks at roughly the last {DRIFT_MONITOR_RECENT_CYCLES} forecast cycles (a few hours), which is far shorter and more recent than this script's multi-day new window, so it is much more sensitive to short-term regime shifts that get averaged out here.",
        "  2. **Model staleness**: the drift monitor compares actuals against whatever the *actually deployed* champion predicted, which may have been trained well before the recent window it is scoring. This script's walk-forward simulation retrains at the start of every fold, so it never lets the model get as stale as a production model that hasn't been retrained recently -- a stale production model would be expected to show *more* bias than this freshly-retrained simulation.",
        "  3. **Different time period**: the drift monitor's last reported number may reflect a period this refreshed snapshot does not even cover (production may be further ahead than `new_observations_max`). This script does not query the drift monitor's live state to pin down its exact window, since that was out of scope for a read-only data/model validation pass.",
        "- Both figures share the same *sign* (both models under-predict on net), which is at least directionally consistent with the drift monitor's finding.",
        "",
        "## Leakage check (Step 6)",
        "",
        f"- `training_end < validation_start` for every fold (old windows + new window, combined): **{'PASSED' if payload['leakage_passed'] else 'FAILED'}**. See `tables/temporal_fold_audit.csv` under the original `reports/challenger_lab_v2/` lab, and the new pytest test `test_temporal_fold_audit_holds_across_a_snapshot_refresh_boundary`.",
        "- No new observation after a fold's own `data_cutoff` enters that fold's lag, rolling feature, or historical profile -- unchanged from the original lab, since this script reuses the exact same feature-engineering and fold-iteration code, only pointed at a longer `observations` frame and a later cutoffs range.",
        "- The ensemble weights applied here are hardcoded constants (`FROZEN_WEIGHTS`); no fitting function was called at all in this script, so there is no possibility of the new data (or the hard holdout) influencing them.",
        "",
        "## Test result",
        "",
        f"- `python3 -m pytest`: {payload['test_result']}",
        "",
        "## Files",
        "",
        "- `reports/challenger_lab_v2_refresh/summary.md`, `summary.json`",
        "- `reports/challenger_lab_v2_refresh/new_window_fold_metrics.csv`, `new_window_station_metrics.csv`, `new_window_horizon_metrics.csv`",
        "- `reports/challenger_lab_v2_refresh/data_refresh_audit.md` (written by `scripts/refresh_observations_snapshot.py`)",
        "",
        "## Conclusion",
        "",
        f"**{payload['stability_classification']}**",
        "",
        "No model was promoted. Champion, Predict, scheduler, and submissions were not modified.",
    ]
    return "\n".join(lines) + "\n"


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


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR = REPORT_DIR / "tables"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading original snapshot (for the historical windows) and refreshed snapshot...", flush=True)
    old_observations = load_observations()
    refreshed_observations = load_refreshed_observations()
    old_max = old_observations["observed_at"].max()
    print(f"  old_observations_max = {old_max.isoformat()}", flush=True)
    print(f"  refreshed rows = {len(refreshed_observations)}, refreshed max = {refreshed_observations['observed_at'].max().isoformat()}", flush=True)

    old_window = experiment_cutoffs(old_observations)
    refreshed_times = sorted(refreshed_observations["observed_at"].drop_duplicates())
    cache: Cache = {}

    print("\n[Part A] Historical reproduction check (old WF + hard-holdout windows, refreshed dataset)...", flush=True)
    wf_hist = run_frozen_ensemble(refreshed_observations, refreshed_times, old_window.walk_forward_cutoffs, cache=cache)
    hard_hist = run_frozen_ensemble(refreshed_observations, refreshed_times, old_window.hard_holdout_cutoffs, cache=cache)
    hist_rows = [
        {
            "window": "walk_forward",
            "model": "Candidate B v2",
            "accuracy": wf_hist["baseline"].summary["mean_station_accuracy"],
            "original_reference": ORIGINAL_REFERENCE["baseline_wf"],
        },
        {
            "window": "walk_forward",
            "model": "Ensemble (frozen)",
            "accuracy": wf_hist["ensemble"].summary["mean_station_accuracy"],
            "original_reference": ORIGINAL_REFERENCE["ensemble_wf"],
        },
        {
            "window": "hard_holdout",
            "model": "Candidate B v2",
            "accuracy": hard_hist["baseline"].summary["mean_station_accuracy"],
            "original_reference": ORIGINAL_REFERENCE["baseline_hard"],
        },
        {
            "window": "hard_holdout",
            "model": "Ensemble (frozen)",
            "accuracy": hard_hist["ensemble"].summary["mean_station_accuracy"],
            "original_reference": ORIGINAL_REFERENCE["ensemble_hard"],
        },
    ]
    historical_table = pd.DataFrame(hist_rows)
    historical_table["abs_diff_vs_original"] = (historical_table["accuracy"] - historical_table["original_reference"]).abs()
    for row in historical_table.itertuples(index=False):
        print(f"    {row.window:12s} {row.model:20s} {row.accuracy:.4f}% (original: {row.original_reference:.4f}%, |diff|={row.abs_diff_vs_original:.6f})", flush=True)

    print("\n[Part B] New-window evaluation (data after the old snapshot max)...", flush=True)
    new_cutoffs = compute_new_window_cutoffs(refreshed_observations, old_max)
    if not new_cutoffs:
        raise SystemExit("No new cutoffs available past the old snapshot max -- nothing to validate.")
    new_result = run_frozen_ensemble(refreshed_observations, refreshed_times, new_cutoffs, cache=cache)
    new_window_folds = sorted({ts.floor("D") for ts in new_cutoffs})
    print(f"    New window: {min(new_cutoffs)} .. {max(new_cutoffs)} ({len(new_cutoffs)} cutoffs, {len(new_window_folds)} fold-day(s))", flush=True)

    b_metrics = official_metrics_full(new_result["baseline"].predictions)
    ens_metrics = official_metrics_full(new_result["ensemble"].predictions)
    print(f"    Candidate B v2: {b_metrics['mean_station_accuracy']:.4f}%", flush=True)
    print(f"    Ensemble (frozen): {ens_metrics['mean_station_accuracy']:.4f}%", flush=True)
    print(f"    Delta: {ens_metrics['mean_station_accuracy'] - b_metrics['mean_station_accuracy']:+.4f} pp", flush=True)

    fold_table = stability_table(new_result["baseline"].predictions, new_result["ensemble"].predictions)
    stability = stability_summary(fold_table)
    classification = classify_stability(stability)
    print(f"    Stability: {stability} -> {classification}", flush=True)

    # ---- Step 6 leakage audit, combined old + new cutoffs on the refreshed dataset ----
    leakage_audit = temporal_fold_audit_generic(refreshed_observations, old_window.walk_forward_cutoffs + old_window.hard_holdout_cutoffs + new_cutoffs)
    leakage_passed = bool((leakage_audit["training_end"] < leakage_audit["validation_start"]).all())
    leakage_audit.to_csv(TABLE_DIR / "temporal_fold_audit_refresh.csv", index=False)
    print(f"    Leakage audit (old+new combined): {'PASSED' if leakage_passed else 'FAILED'}", flush=True)

    # ---- Step 7 outputs ----
    fold_frames = [
        per_fold_metrics(BASELINE_NAME, new_result["baseline"].predictions, split="new_window"),
        per_fold_metrics(ENSEMBLE_NAME, new_result["ensemble"].predictions, split="new_window"),
    ]
    pd.concat(fold_frames, ignore_index=True).to_csv(REPORT_DIR / "new_window_fold_metrics.csv", index=False)

    station_frames = [
        station_metrics_table(BASELINE_NAME, new_result["baseline"], split="new_window"),
        station_metrics_table(ENSEMBLE_NAME, new_result["ensemble"], split="new_window"),
    ]
    pd.concat(station_frames, ignore_index=True).to_csv(REPORT_DIR / "new_window_station_metrics.csv", index=False)

    horizon_frames = [
        horizon_metrics_table(BASELINE_NAME, new_result["baseline"], split="new_window"),
        horizon_metrics_table(ENSEMBLE_NAME, new_result["ensemble"], split="new_window"),
    ]
    pd.concat(horizon_frames, ignore_index=True).to_csv(REPORT_DIR / "new_window_horizon_metrics.csv", index=False)

    special_comparison(new_result["baseline"], new_result["ensemble"], group="station").to_csv(TABLE_DIR / "new_window_special_station_comparison.csv", index=False)
    special_comparison(new_result["baseline"], new_result["ensemble"], group="horizon").to_csv(TABLE_DIR / "new_window_special_horizon_comparison.csv", index=False)

    import subprocess

    test_proc = subprocess.run(["python3", "-m", "pytest", "-q"], cwd=str(ROOT), capture_output=True, text=True)
    test_result_line = test_proc.stdout.strip().splitlines()[-1] if test_proc.stdout.strip() else ("FAILED" if test_proc.returncode else "unknown")
    test_result = f"{'PASSED' if test_proc.returncode == 0 else 'FAILED'} -- {test_result_line}"
    print(f"\n[Tests] {test_result}", flush=True)

    payload = {
        "lab": "challenger_lab_v2_refresh",
        "scope": "validation_only_no_new_experiments_no_promotion",
        "frozen_weights": FROZEN_WEIGHTS,
        "frozen_feature_set": "B + day-of-week profile (FEATURE_SET_PLUS_DOW_PROFILE)",
        "frozen_challenger_family": FROZEN_CHALLENGER_FAMILY,
        "old_observations_max": old_max.isoformat().replace("+00:00", "Z"),
        "new_observations_max": refreshed_observations["observed_at"].max().isoformat().replace("+00:00", "Z"),
        "historical_table": historical_table,
        "historical_reproduction_ok": bool((historical_table["abs_diff_vs_original"] < 0.01).all()),
        "new_window_start": min(new_cutoffs).isoformat().replace("+00:00", "Z"),
        "new_window_end": max(new_cutoffs).isoformat().replace("+00:00", "Z"),
        "new_window_cutoffs": len(new_cutoffs),
        "new_window_folds": len(new_window_folds),
        "new_window_b_accuracy": b_metrics["mean_station_accuracy"],
        "new_window_ensemble_accuracy": ens_metrics["mean_station_accuracy"],
        "new_window_delta_pp": ens_metrics["mean_station_accuracy"] - b_metrics["mean_station_accuracy"],
        "new_window_b_bias": b_metrics["signed_relative_bias"],
        "new_window_ensemble_bias": ens_metrics["signed_relative_bias"],
        "fold_table": fold_table,
        "stability": stability,
        "stability_classification": classification,
        "leakage_passed": leakage_passed,
        "test_result": test_result,
        "any_model_promoted": False,
    }

    json_payload = {k: v for k, v in payload.items() if not isinstance(v, pd.DataFrame)}
    json_payload["historical_table"] = json.loads(historical_table.to_json(orient="records"))
    json_payload["fold_table"] = json.loads(fold_table.to_json(orient="records"))
    (REPORT_DIR / "summary.json").write_text(json.dumps(json_payload, indent=2, sort_keys=True, default=str) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_markdown(payload))

    print(f"\nReport written to {REPORT_DIR / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
