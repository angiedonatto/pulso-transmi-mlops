from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import candidate_i_eval as iev  # noqa: E402
from candidate_b_shadow import HORIZONS  # noqa: E402
from drift_monitor import _accuracy_by  # noqa: E402


REPORT_DIR = ROOT / "reports" / "accuracy_90_lab"
TABLE_DIR = REPORT_DIR / "tables"
FRAME_PATH = ROOT / "data" / "interim" / "candidate_i" / "final_frame.pkl"

BOGOTA_TZ = "America/Bogota"
PEAK_HOURS = {5, 6, 7, 8, 17, 18, 19, 20}
SELECTION_WINDOWS = ("stable", "drift_dev")
EVALUATION_WINDOWS = ("stable", "drift_dev", "drift_test", "drift_all", "official", "recent12", "recent6", "all_scored")

MODEL_COLUMNS = {
    "B_walk_forward": "pred_B",
    "F_profile_ratio": "pred_F",
    "Candidate_G": "pred_G",
    "Candidate_I": "pred_I",
    "I_minus_oos_history": "pred_abl_hourly_history",
    "I_minus_station_specialization": "pred_abl_no_station",
    "I_without_snr_gate": "pred_abl::L:g2",
    "I_plus_shape": "pred_abl::LSg:g2|snr4+shape3|snr16",
    "I_plus_DoD_trend": "pred_abl::blend:0.75*g2+0.25*DoDt",
}


def load_frame(path: Path = FRAME_PATH) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/candidate_i_lab.py final first.")
    frame = pd.read_pickle(path)
    frame["data_cutoff"] = pd.to_datetime(frame["data_cutoff"], utc=True)
    frame["target_at"] = pd.to_datetime(frame["target_at"], utc=True)
    frame["station_id"] = frame["station_id"].astype(str).str.zfill(5)
    local = frame["target_at"].dt.tz_convert(BOGOTA_TZ)
    frame["target_hour_local"] = local.dt.hour.astype(int)
    frame["is_peak_hour"] = frame["target_hour_local"].isin(PEAK_HOURS)
    frame["demand_level"] = pd.qcut(frame["actual"], q=4, labels=["q1_low", "q2", "q3", "q4_peak"], duplicates="drop")
    return frame.sort_values(["data_cutoff", "station_id", "horizon_minutes"]).reset_index(drop=True)


def dependency_audit() -> pd.DataFrame:
    rows = []
    for package, implemented_as in [
        ("lightgbm", "skipped_optional_dependency_missing"),
        ("catboost", "skipped_optional_dependency_missing"),
        ("xgboost", "skipped_optional_dependency_missing"),
        ("sklearn", "available_used_via_existing_hgb_extra_trees_and_snr_labs"),
        ("pandas", "available"),
        ("numpy", "available"),
    ]:
        rows.append(
            {
                "package": package,
                "available": importlib.util.find_spec(package) is not None,
                "lab_action": "available" if importlib.util.find_spec(package) is not None else implemented_as,
            }
        )
    return pd.DataFrame(rows)


def metric_audit(frame: pd.DataFrame) -> dict[str, Any]:
    g_live_path = ROOT / "reports" / "candidate_g_station_guard_lab" / "summary.json"
    g_live = json.loads(g_live_path.read_text()) if g_live_path.exists() else {}
    official = frame[frame["is_official"].astype(bool)].copy()
    return {
        "official_formula": "WAPE per station = sum(abs(actual-pred))/sum(actual); accuracy = 100 * max(0, 1-WAPE); official accuracy = mean station accuracy.",
        "candidate_g_shadow_83_79_comparable_formula": True,
        "candidate_g_shadow_83_79_comparable_to_leaderboard": False,
        "why_not_leaderboard": [
            "83.79% was an offline shadow/live-window score over 12 resolved cycles, not a submitted hidden leaderboard receipt.",
            "The formula is comparable, but the window, feedback warm-start, and resolved-only availability differ from leaderboard state.",
            "This lab separates offline walk-forward, official resolved cycles, and recent shadow windows.",
        ],
        "candidate_g_station_guard_shadow_accuracy": g_live.get("current_mean_station_accuracy"),
        "candidate_g_official_resolved_accuracy": official_accuracy(official, "pred_G"),
        "official_resolved_cycles": int(official["data_cutoff"].nunique()),
    }


def official_accuracy(frame: pd.DataFrame, pred_col: str) -> float:
    return float(station_metrics(frame, pred_col)["accuracy"].mean())


def wape(frame: pd.DataFrame, pred_col: str) -> float:
    den = float(frame["actual"].sum())
    return float((frame[pred_col].astype(float) - frame["actual"].astype(float)).abs().sum() / den) if den > 0 else math.nan


def station_metrics(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    scored = scored_frame(frame, pred_col)
    return _accuracy_by(scored, ["station_id"])


def scored_frame(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    scored = frame.assign(prediction=frame[pred_col].astype(float))
    scored["residual"] = scored["prediction"].astype(float) - scored["actual"].astype(float)
    scored["abs_error"] = scored["residual"].abs()
    return scored


def window_leaderboard(frame: pd.DataFrame, model_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    masks = iev.window_masks(frame)
    for model, col in model_cols.items():
        if col not in frame.columns:
            continue
        for window in EVALUATION_WINDOWS:
            sub = frame[masks[window]].copy()
            if sub.empty:
                continue
            station = station_metrics(sub, col)
            cycle = cycle_metrics(sub, col)
            rows.append(
                {
                    "model": model,
                    "column": col,
                    "window": window,
                    "official_accuracy": float(station["accuracy"].mean()),
                    "wape": wape(sub, col),
                    "mean_cycle_accuracy": float(cycle["cycle_accuracy"].mean()),
                    "median_cycle_accuracy": float(cycle["cycle_accuracy"].median()),
                    "worst_cycle_accuracy": float(cycle["cycle_accuracy"].min()),
                    "cycles": int(sub["data_cutoff"].nunique()),
                    "rows": int(len(sub)),
                    "bias": float((sub[col].astype(float) - sub["actual"].astype(float)).mean()),
                    "signed_relative_bias": float((sub[col].astype(float) - sub["actual"].astype(float)).sum() / sub["actual"].sum()),
                }
            )
    return pd.DataFrame(rows)


def cycle_metrics(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    scored = scored_frame(frame, pred_col)
    per = _accuracy_by(scored, ["data_cutoff", "station_id"])
    return (
        per.groupby("data_cutoff", as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "cycle_accuracy"})
        .sort_values("data_cutoff")
        .reset_index(drop=True)
    )


def slice_metrics(frame: pd.DataFrame, pred_cols: dict[str, str], window: str, by: str) -> pd.DataFrame:
    masks = iev.window_masks(frame)
    sub = frame[masks[window]].copy()
    rows = []
    for model, col in pred_cols.items():
        if col not in sub.columns:
            continue
        scored = scored_frame(sub, col)
        per = _accuracy_by(scored, [by])
        per["model"] = model
        per["window"] = window
        rows.append(per)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_ensembles(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    candidates = {
        "Candidate_I": "pred_I",
        "I_plus_shape": "pred_abl::LSg:g2|snr4+shape3|snr16",
        "I_without_snr_gate": "pred_abl::L:g2",
        "Candidate_G": "pred_G",
    }
    candidates = {k: v for k, v in candidates.items() if v in frame.columns}
    rows = []
    masks = iev.window_masks(frame)
    selection_mask = masks["stable"] | masks["drift_dev"]
    for left_name, left_col in candidates.items():
        for right_name, right_col in candidates.items():
            if left_name >= right_name:
                continue
            for weight in (0.25, 0.50, 0.75):
                name = f"blend_{weight:g}_{left_name}__{1-weight:g}_{right_name}"
                col = f"lab::{name}"
                frame[col] = weight * frame[left_col].astype(float) + (1.0 - weight) * frame[right_col].astype(float)
                stable = official_accuracy(frame[masks["stable"]], col)
                drift_dev = official_accuracy(frame[masks["drift_dev"]], col)
                objective = 0.5 * stable + 0.5 * drift_dev
                rows.append(
                    {
                        "model": name,
                        "column": col,
                        "left": left_name,
                        "right": right_name,
                        "weight_left": weight,
                        "selection_objective": objective,
                        "selection_accuracy": official_accuracy(frame[selection_mask], col),
                        "stable": stable,
                        "drift_dev": drift_dev,
                    }
                )
    table = pd.DataFrame(rows).sort_values("selection_objective", ascending=False).reset_index(drop=True)
    selected_cols = {row.model: row.column for row in table.head(5).itertuples(index=False)}
    return table, selected_cols


def target_audit(frame: pd.DataFrame) -> dict[str, Any]:
    per_cycle = frame.groupby("data_cutoff").agg(rows=("actual", "size"), stations=("station_id", "nunique"))
    horizon_set = sorted(int(v) for v in frame["horizon_minutes"].unique())
    station_count = int(frame["station_id"].nunique())
    target_relation = ((frame["target_at"] - frame["data_cutoff"]).dt.total_seconds() / 60 == frame["horizon_minutes"]).all()
    leakage_ok = bool((frame["max_source_at"] <= frame["data_cutoff"]).all()) if "max_source_at" in frame.columns else None
    return {
        "rows": int(len(frame)),
        "cycles": int(frame["data_cutoff"].nunique()),
        "stations": station_count,
        "horizons": horizon_set,
        "rows_per_cycle_min": int(per_cycle["rows"].min()),
        "rows_per_cycle_max": int(per_cycle["rows"].max()),
        "stations_per_cycle_min": int(per_cycle["stations"].min()),
        "stations_per_cycle_max": int(per_cycle["stations"].max()),
        "expected_rows_per_cycle": station_count * len(horizon_set),
        "target_relation_matches_horizon": bool(target_relation),
        "max_source_at_lte_data_cutoff": leakage_ok,
        "data_cutoff_min": frame["data_cutoff"].min().isoformat().replace("+00:00", "Z"),
        "data_cutoff_max": frame["data_cutoff"].max().isoformat().replace("+00:00", "Z"),
    }


def ablation_table(leaderboard: pd.DataFrame) -> pd.DataFrame:
    pivot = leaderboard.pivot_table(index="model", columns="window", values="official_accuracy")
    rows = []
    if "Candidate_I" not in pivot.index:
        return pd.DataFrame()
    for model in pivot.index:
        if model == "Candidate_I":
            continue
        if model.startswith("I_") or model in {"F_profile_ratio", "Candidate_G"}:
            row = {"ablation_or_candidate": model}
            for window in ("stable", "drift_dev", "drift_test", "recent12", "recent6", "all_scored"):
                row[f"{window}_delta_vs_I_pp"] = float(pivot.loc[model, window] - pivot.loc["Candidate_I", window])
            rows.append(row)
    return pd.DataFrame(rows).sort_values("all_scored_delta_vs_I_pp", ascending=False)


def write_outputs(payload: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    (REPORT_DIR / "summary.json").write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_markdown(payload, tables) + "\n")


def render_markdown(payload: dict[str, Any], tables: dict[str, pd.DataFrame]) -> str:
    leaderboard = tables["leaderboard"]
    wide = leaderboard.pivot_table(index="model", columns="window", values="official_accuracy")
    cols = [c for c in ("stable", "drift_dev", "drift_test", "official", "recent12", "recent6", "all_scored") if c in wide.columns]
    wide = wide[cols].sort_values("all_scored", ascending=False)
    best_all = wide.iloc[0]
    best_model = str(wide.index[0])
    return "\n".join(
        [
            "# Accuracy 90 Lab",
            "",
            "## Scope",
            "",
            "- Experimental only. Champion, scheduler, official submissions and production parameters were not modified.",
            "- Offline walk-forward and resolved official-cycle results are reported separately from leaderboard claims.",
            "- LightGBM/CatBoost/XGBoost were audited but not run because they are not installed in this environment.",
            "",
            "## Metric Audit",
            "",
            f"- Candidate G 83.79 shadow formula-comparable: `{payload['metric_audit']['candidate_g_shadow_83_79_comparable_formula']}`.",
            f"- Candidate G 83.79 leaderboard-comparable: `{payload['metric_audit']['candidate_g_shadow_83_79_comparable_to_leaderboard']}`.",
            f"- Candidate G resolved official accuracy in this lab: {payload['metric_audit']['candidate_g_official_resolved_accuracy']:.4f}%.",
            "",
            "## Target / Leakage Audit",
            "",
            f"- Cycles: {payload['target_audit']['cycles']}; rows/cycle: {payload['target_audit']['rows_per_cycle_min']}..{payload['target_audit']['rows_per_cycle_max']}; expected rows/cycle: {payload['target_audit']['expected_rows_per_cycle']}.",
            f"- Horizons: {payload['target_audit']['horizons']}; stations: {payload['target_audit']['stations']}.",
            f"- `target_at - data_cutoff == horizon`: {payload['target_audit']['target_relation_matches_horizon']}.",
            f"- `max_source_at <= data_cutoff`: {payload['target_audit']['max_source_at_lte_data_cutoff']}.",
            "",
            "## Leaderboard",
            "",
            _markdown_table(wide.reset_index()),
            "",
            "## 90% Target",
            "",
            f"- Best all_scored model: `{best_model}` at {best_all['all_scored']:.4f}%.",
            f"- Gap to 90%: {90.0 - best_all['all_scored']:.4f} pp.",
            "- Conclusion: 90% was not reached offline; the best generalizing candidates are still around the mid-86% range.",
            "",
            "## Ensemble Selection",
            "",
            _markdown_table(tables["ensemble_selection"].head(10)),
            "",
            "## Ablations",
            "",
            _markdown_table(tables["ablations"]),
            "",
            "## Files",
            "",
            "- `reports/accuracy_90_lab/summary.json`",
            "- `reports/accuracy_90_lab/tables/*.csv`",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(str(c) for c in columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_lab() -> dict[str, Any]:
    started = time.perf_counter()
    frame = load_frame()
    dep = dependency_audit()
    target = target_audit(frame)
    metric = metric_audit(frame)
    ensemble_selection, ensemble_cols = select_ensembles(frame)
    model_cols = {**MODEL_COLUMNS, **ensemble_cols}
    model_cols = {name: col for name, col in model_cols.items() if col in frame.columns}
    leaderboard = window_leaderboard(frame, model_cols)
    masks = iev.window_masks(frame)
    official_cols = {
        "Candidate_G": "pred_G",
        "Candidate_I": "pred_I",
        "I_plus_shape": "pred_abl::LSg:g2|snr4+shape3|snr16",
    }
    official_cols = {k: v for k, v in official_cols.items() if v in frame.columns}
    tables = {
        "dependency_audit": dep,
        "leaderboard": leaderboard,
        "ensemble_selection": ensemble_selection,
        "ablations": ablation_table(leaderboard),
        "station_all_scored": slice_metrics(frame, official_cols, "all_scored", "station_id"),
        "hour_all_scored": slice_metrics(frame, official_cols, "all_scored", "target_hour_local"),
        "horizon_all_scored": slice_metrics(frame, official_cols, "all_scored", "horizon_minutes"),
        "peak_all_scored": slice_metrics(frame, official_cols, "all_scored", "is_peak_hour"),
        "demand_level_all_scored": slice_metrics(frame, official_cols, "all_scored", "demand_level"),
    }
    payload = {
        "lab": "accuracy_90_lab",
        "scope": "experimental_only_no_production_changes",
        "runtime_seconds": time.perf_counter() - started,
        "metric_audit": metric,
        "target_audit": target,
        "model_columns": model_cols,
        "dependency_audit": json.loads(dep.to_json(orient="records")),
        "official_windows": {
            name: int(masks[name].sum()) for name in EVALUATION_WINDOWS if name in masks
        },
    }
    write_outputs(payload, tables)
    return payload


def main() -> None:
    payload = run_lab()
    print(json.dumps(_json_value(payload), indent=2, sort_keys=True))
    print(f"Report written to {REPORT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
