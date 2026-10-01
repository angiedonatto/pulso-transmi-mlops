"""Evaluate Candidate K v2.1 (shadow) against J v1.1 (official) and baseline_latest (read-only).

Only cycles after the K design freeze where J and K both have 48 predictions and every target is
observed are scored, with the leaderboard metric (`drift_monitor.accuracy_summary`,
mean_station_accuracy). baseline_latest = last observation at or before the data_cutoff, for every
horizon. Writes reports/candidate_k_shadow/: rows (cycle x station x horizon), cycles, stations,
horizons, summary.json and latest.md. Never writes to Supabase and never promotes anything.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import candidate_k_shadow as cks  # noqa: E402

REPORT_DIR = ROOT / "reports" / "candidate_k_shadow"
KEYS = ["cycle_id", "station_id", "target_at"]


def load_rows(store: cks.SupabaseCandidateKShadowStore) -> tuple[pd.DataFrame, dict]:
    k_model = store.candidate_k_model_version()
    j_model = store.model_by_name(cks.J_MODEL_NAME)
    if k_model is None or j_model is None:
        raise SystemExit(f"missing model rows: K={k_model is not None}, J={j_model is not None}")
    cycles = store.cycles_since(cks.OOS_FROM_CUTOFF)
    ids = cycles["cycle_id"].tolist()
    j = _predictions(store.predictions_for(str(j_model["model_id"]), ids), "j_prediction")
    k = _predictions(store.predictions_for(str(k_model["model_id"]), ids), "k_prediction")
    traces = store.traces_for(str(k_model["model_id"]), ids)
    rows = j.drop(columns="created_at").merge(k[[*KEYS, "k_prediction", "created_at"]], on=KEYS, how="outer")
    rows = rows.merge(cycles, on="cycle_id", how="left")
    rows["horizon_minutes"] = ((rows["target_at"] - rows["data_cutoff"]) / pd.Timedelta(minutes=1)).round().astype("Int64")
    k_created = k.groupby("cycle_id")["created_at"].max()
    rows["k_on_time"] = rows["cycle_id"].map(k_created) <= rows["closes_at"]
    if not traces.empty:
        traces = traces.assign(station_id=traces["station_id"].astype(str).str.zfill(5), target_at=pd.to_datetime(traces["target_at"], utc=True))
        rows = rows.merge(traces[[*KEYS, "calibration_factor"]].rename(columns={"calibration_factor": "k_seasonal_weight"}), on=KEYS, how="left")
    if rows.empty:
        return rows, {"cycles_since_freeze": len(cycles), "j_cycles": 0, "k_cycles": 0}
    observations = store.observations_between(rows["data_cutoff"].min() - pd.Timedelta(hours=2), rows["target_at"].max())
    actual = observations.rename(columns={"observed_at": "target_at", "demand": "actual"})
    rows = rows.merge(actual, on=["station_id", "target_at"], how="left")
    baseline = cks.baseline_latest(observations, rows[["cycle_id", "station_id", "target_at", "data_cutoff"]].drop_duplicates())
    rows = rows.merge(baseline[KEYS + ["baseline_latest_prediction"]], on=KEYS, how="left")
    coverage = {
        "cycles_since_freeze": int(len(cycles)),
        "j_cycles": int(j["cycle_id"].nunique()),
        "k_cycles": int(k["cycle_id"].nunique()),
        "k_missing_where_j_exists": sorted(set(j["cycle_id"]) - set(k["cycle_id"])),
    }
    return rows.sort_values(["data_cutoff", "station_id", "target_at"]).reset_index(drop=True), coverage


def _predictions(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    out = frame.rename(columns={"target_observed_at": "target_at", "prediction": column}).copy()
    out["station_id"] = out["station_id"].astype(str).str.zfill(5)
    out["target_at"] = pd.to_datetime(out["target_at"], utc=True)
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    return out[[*KEYS, column, "created_at"]]


def render(summary: dict, cycles: pd.DataFrame, stations: pd.DataFrame, horizons: pd.DataFrame) -> str:
    pooled = summary["pooled"]
    lines = [
        "# Candidate K v2.1 shadow vs J v1.1 (official) vs baseline_latest",
        "",
        f"Generated {summary['generated_at']}. Read-only. Only cycles with data_cutoff >= {cks.OOS_FROM_CUTOFF} (K design freeze) count.",
        "",
        f"- **Status:** `{summary['status']}`. Promotion: **{summary['promotion']}**. Nothing is promoted automatically.",
        f"- **Scored cycles:** {summary['cycles']} (J and K both complete, every target observed). A reading needs >= {summary['min_cycles_for_reading']}.",
        f"- **Coverage:** {summary['coverage']['k_cycles']} K cycles, {summary['coverage']['j_cycles']} J cycles, out of {summary['coverage']['cycles_since_freeze']} cycles since the freeze.",
        f"  - K missing where J exists: {len(summary['coverage'].get('k_missing_where_j_exists', []))}.",
        f"  - K cycles written after the cycle closed: {summary['late_k_cycles']}.",
    ]
    if pooled:
        d = summary["per_cycle_k_minus_j"]
        lines += [
            f"- **K - J, pooled leaderboard metric:** {summary['delta_pp']['K_minus_J']:+.2f} pp.",
            f"- **K - J, per cycle:** mean {d['mean']:+.2f} pp, bootstrap 95% CI [{d['bootstrap_95'][0]:+.2f}, {d['bootstrap_95'][1]:+.2f}]; wins/losses/ties {d['wins']}/{d['losses']}/{d['ties']}.",
            f"- **K - baseline_latest:** {summary['delta_pp']['K_minus_baseline_latest']:+.2f} pp.",
            "",
            "## Pooled (leaderboard formula over the scored cycles)",
            "",
            "| model | mean_station_accuracy | global_wape_accuracy | accuracy_at_20 | MAE | signed bias |",
            "|---|---|---|---|---|---|",
            *[f"| {m} | {v['mean_station_accuracy']:.2f} | {v['global_wape_accuracy']:.2f} | {v['accuracy_at_20']:.1f} | {v['mae']:.1f} | {v['signed_bias']:+.1f} |" for m, v in pooled.items()],
            "",
            "## Per horizon (mean_station_accuracy)",
            "",
            _table(horizons),
            "",
            "## Per station (mean_station_accuracy)",
            "",
            _table(stations),
            "",
            "## Per cycle",
            "",
            "`k_mean_seasonal_weight` is K's gate: 1 means seasonal regime, 0 means local dynamics.",
            "",
            _table(cycles.assign(data_cutoff=cycles["data_cutoff"].map(cks._ts))),
        ]
    lines += ["", "Rows per cycle x station x horizon: `tables/rows.csv`."]
    return "\n".join(lines) + "\n"


def _table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if math.isnan(value) else f"{value:.2f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    return "\n".join([header, rule, *("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--env-file", type=Path, help="Load only SUPABASE_URL / SUPABASE_SERVICE_KEY from this file.")
    parser.add_argument("--min-cycles", type=int, default=cks.MIN_CYCLES_FOR_READING)
    args = parser.parse_args()
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
                os.environ.setdefault(key, value)
    with cks.SupabaseCandidateKShadowStore() as store:
        rows, coverage = load_rows(store)
    result = cks.evaluate_comparison(rows, min_cycles=args.min_cycles) if not rows.empty else {
        "rows": rows, "cycles": pd.DataFrame(), "stations": pd.DataFrame(), "horizons": pd.DataFrame(),
        "summary": {"cycles": 0, "min_cycles_for_reading": args.min_cycles, "pooled": {}, "late_k_cycles": 0, "status": "insufficient_evidence", "promotion": "manual_decision_required"}}
    summary = {**result["summary"], "coverage": coverage, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "oos_from_cutoff": cks.OOS_FROM_CUTOFF}
    out = args.output_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for name in ("rows", "cycles", "stations", "horizons"):
        result[name].to_csv(out / "tables" / f"{name}.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out / "latest.md").write_text(render(summary, result["cycles"], result["stations"], result["horizons"]))
    print(json.dumps({k: summary[k] for k in ("status", "cycles", "delta_pp", "coverage", "promotion") if k in summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
