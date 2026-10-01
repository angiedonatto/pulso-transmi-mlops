"""Recent model-zoo benchmark over newly arrived cycles.

Read-only by design:
- no submissions API;
- no PULSO_API_KEY;
- no Supabase writes;
- no model registration or promotion.

The default real-time window starts at 2026-09-30 00:00 America/Bogota. The
cycle timestamps being scored are historical replay timestamps (`data_cutoff`,
`target_at`); the window selection uses Supabase receipt/prediction creation
time, i.e. when those historical cycles arrived operationally.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import candidate_i_snr_gate as ci  # noqa: E402
import candidate_j_regime as cj  # noqa: E402
import candidate_j_shadow as cjs  # noqa: E402
from candidate_b_shadow import build_shadow_predictions, load_shadow_artifact  # noqa: E402
from forecast_cycle_sync import CycleTarget, ForecastCycle  # noqa: E402


REPORT_DIR = ROOT / "reports" / "model_zoo_recent"
BOGOTA_TZ = "America/Bogota"
REAL_WINDOW_START = "2026-09-30T00:00:00-05:00"
EXPECTED_TARGETS = 48
HORIZONS = (15, 30, 45, 60)
MODEL_ORDER = [
    "baseline_lag96",
    "baseline_latest",
    "candidate_b_v1",
    "candidate_b_v2",
    "candidate_f",
    "candidate_g",
    "candidate_i_v1",
    "candidate_j_v1_1",
]


@dataclass(frozen=True)
class ReplayContext:
    cycles: pd.DataFrame
    targets: pd.DataFrame
    observations: pd.DataFrame
    g_traces: pd.DataFrame
    model_versions: pd.DataFrame
    g_model_id: str | None
    j_model_id: str | None
    real_window_start: pd.Timestamp
    real_window_end: pd.Timestamp | None


class ReadOnlySupabase:
    def __init__(self, timeout: float = 60.0) -> None:
        load_env()
        url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        if os.getenv("PULSO_API_KEY"):
            raise RuntimeError("PULSO_API_KEY must not be present for this read-only benchmark")
        self._client = httpx.Client(
            base_url=f"{url}/rest/v1",
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ReadOnlySupabase":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def table(self, name: str, params: dict[str, str], *, page_size: int = 1000) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        offset = 0
        batch_size = min(page_size, 1000)
        while True:
            response = self._client.get(f"/{name}", params={**params, "limit": str(batch_size), "offset": str(offset)})
            response.raise_for_status()
            page = response.json()
            rows.extend(page)
            if len(page) < batch_size:
                break
            offset += batch_size
        return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-window-start", default=REAL_WINDOW_START, help="Operational arrival-time start, e.g. 2026-09-30T00:00:00-05:00.")
    parser.add_argument("--real-window-end", default=None, help="Optional operational arrival-time end.")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--max-cycles", type=int, default=None, help="Optional cap for fast smoke runs.")
    parser.add_argument("--include-candidate-i", action="store_true", help="Include slow Candidate I replay; it rebuilds OOS HGB history per cycle.")
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    context = load_context(args.real_window_start, args.real_window_end, max_cycles=args.max_cycles)
    inventory = model_inventory(context.model_versions)
    predictions, failures = replay_models(context, include_candidate_i=args.include_candidate_i)
    scored = attach_actuals(predictions, context)
    available = summarize_available(scored)
    common, common_cycles = summarize_common(scored)
    per_cycle = cycle_table(scored, context)
    per_station = grouped_table(scored, ["model", "station_id"])
    per_horizon = grouped_table(scored, ["model", "horizon_minutes"])
    regime = regime_table(context)
    if not per_cycle.empty and "cycle_id" in per_cycle and not regime.empty:
        per_cycle = per_cycle.merge(regime[["cycle_id", "regime"]], on="cycle_id", how="left")
    if not common_cycles.empty and "cycle_id" in common_cycles and not regime.empty:
        common_cycles = common_cycles.merge(regime[["cycle_id", "regime"]], on="cycle_id", how="left")
    wins = wins_table(per_cycle)
    temporal = temporal_wide(per_cycle)
    make_figures(per_cycle, args.output_dir / "figures")

    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "real_window_start": context.real_window_start.isoformat(),
        "real_window_end": context.real_window_end.isoformat() if context.real_window_end is not None else None,
        "historical_data_cutoff_min": _maybe_iso(context.cycles["data_cutoff"].min()),
        "historical_data_cutoff_max": _maybe_iso(context.cycles["data_cutoff"].max()),
        "historical_target_at_max": _maybe_iso(context.targets["target_at"].max()),
        "cycles_requested": int(context.cycles["cycle_id"].nunique()),
        "models_scored": sorted(scored["model"].unique().tolist()) if not scored.empty else [],
        "common_cycle_count": int(common_cycles["cycle_id"].nunique()) if not common_cycles.empty else 0,
        "inventory": inventory.to_dict("records"),
        "available_data": available.to_dict("records"),
        "common_cycles_summary": common.to_dict("records"),
        "wins_vs_g_and_j": wins.to_dict("records"),
        "replay_failures": failures,
        "notes": [
            "Selection window is real operational arrival time; data_cutoff/target_at are historical replay timestamps.",
            "Candidate G and Candidate F use persisted G traces: base_prediction is the production F base, final_prediction is production G.",
            "No model is tuned or retrained using target actuals in this window.",
        ],
    }
    write_outputs(args.output_dir, payload, inventory, available, common, per_cycle, per_station, per_horizon, common_cycles, temporal, scored)
    print(json.dumps({"cycles": payload["cycles_requested"], "models": payload["models_scored"], "common_cycles": payload["common_cycle_count"]}, sort_keys=True))


def load_context(start: str, end: str | None, *, max_cycles: int | None) -> ReplayContext:
    start_ts = pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end).tz_convert("UTC") if end else None
    with ReadOnlySupabase() as db:
        model_versions = db.table("model_versions", {"select": "model_id,model_name,model_type,is_active,artifact_uri,trained_until,feature_config,params"}, page_size=1000)
        receipts_params = {
            "select": "cycle_id,model_id,status,is_official,accepted_at,created_at",
            "created_at": f"gte.{_ts(start_ts)}",
            "order": "created_at.asc",
        }
        receipts = db.table("submission_receipts", receipts_params, page_size=2000)
        if end_ts is not None and not receipts.empty:
            receipts["created_at"] = pd.to_datetime(receipts["created_at"], utc=True)
            receipts = receipts[receipts["created_at"] <= end_ts]
        if receipts.empty:
            return ReplayContext(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), model_versions, None, None, start_ts, end_ts)
        cycle_ids = sorted(receipts.loc[receipts["status"].isin(["accepted", "idempotent_replay"]), "cycle_id"].astype(str).unique())
        if max_cycles:
            cycle_ids = cycle_ids[-max_cycles:]
        ids = _in_filter(cycle_ids)
        cycles = db.table("forecast_cycles", {"select": "cycle_id,data_cutoff,expected_predictions", "cycle_id": ids, "order": "data_cutoff.asc"}, page_size=2000)
        targets = db.table("forecast_cycle_targets", {"select": "cycle_id,station_id,target_at,horizon_minutes,target_order", "cycle_id": ids, "order": "cycle_id.asc,target_order.asc"}, page_size=5000)
        g_model_id = _model_id(model_versions, model_name="candidate_g_f4_profile_ratio_g2_ewma")
        j_model_id = _model_id(model_versions, model_type=cjs.MODEL_TYPE)
        trace_params = {
            "select": "cycle_id,model_id,station_id,target_at,horizon_minutes,base_prediction,calibration_factor,final_prediction,data_cutoff",
            "cycle_id": ids,
            "order": "data_cutoff.asc",
        }
        if g_model_id:
            trace_params["model_id"] = f"eq.{g_model_id}"
        g_traces = db.table("prediction_traces", trace_params, page_size=5000)
        obs_start = pd.to_datetime(targets["target_at"], utc=True).min() - pd.Timedelta(days=35) if not targets.empty else start_ts - pd.Timedelta(days=35)
        obs_end = pd.to_datetime(targets["target_at"], utc=True).max() if not targets.empty else start_ts
        observations = db.table(
            "observations",
            {"select": "station_id,observed_at,demand", "observed_at": f"gte.{_ts(obs_start)}", "order": "observed_at.asc,station_id.asc"},
            page_size=10000,
        )
    cycles["data_cutoff"] = pd.to_datetime(cycles["data_cutoff"], utc=True)
    targets["target_at"] = pd.to_datetime(targets["target_at"], utc=True)
    targets["station_id"] = targets["station_id"].astype(str).str.zfill(5)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    observations["station_id"] = observations["station_id"].astype(str).str.zfill(5)
    observations["demand"] = observations["demand"].astype(float)
    observations = observations[observations["observed_at"] <= obs_end].copy()
    if not g_traces.empty:
        g_traces["target_at"] = pd.to_datetime(g_traces["target_at"], utc=True)
        g_traces["data_cutoff"] = pd.to_datetime(g_traces["data_cutoff"], utc=True)
        g_traces["station_id"] = g_traces["station_id"].astype(str).str.zfill(5)
    cycles, targets, g_traces = filter_complete_revealed(cycles, targets, observations, g_traces)
    return ReplayContext(cycles, targets, observations, g_traces, model_versions, g_model_id, j_model_id, start_ts, end_ts)


def filter_complete_revealed(cycles: pd.DataFrame, targets: pd.DataFrame, observations: pd.DataFrame, g_traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cycles.empty or targets.empty:
        return cycles.iloc[0:0], targets.iloc[0:0], g_traces.iloc[0:0]
    actual_index = set(zip(observations["station_id"], observations["observed_at"]))
    target_counts = targets.groupby("cycle_id").size()
    revealed_counts = targets.apply(lambda r: (str(r["station_id"]).zfill(5), pd.Timestamp(r["target_at"])) in actual_index, axis=1).groupby(targets["cycle_id"]).sum()
    complete_ids = set(target_counts[target_counts == EXPECTED_TARGETS].index) & set(revealed_counts[revealed_counts == EXPECTED_TARGETS].index)
    if not g_traces.empty:
        trace_counts = g_traces.groupby("cycle_id").size()
        complete_ids &= set(trace_counts[trace_counts == EXPECTED_TARGETS].index)
    cycles = cycles[cycles["cycle_id"].isin(complete_ids)].sort_values("data_cutoff").reset_index(drop=True)
    targets = targets[targets["cycle_id"].isin(complete_ids)].copy()
    g_traces = g_traces[g_traces["cycle_id"].isin(complete_ids)].copy() if not g_traces.empty else g_traces
    return cycles, targets, g_traces


def replay_models(ctx: ReplayContext, *, include_candidate_i: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    if ctx.cycles.empty:
        return pd.DataFrame(), []
    obs_lookup = observations_lookup(ctx.observations)
    b_artifacts = {
        "candidate_b_v1": safe_load_b("repo://artifacts/shadow/candidate_b_v1", failures),
        "candidate_b_v2": safe_load_b("repo://artifacts/shadow/candidate_b_v2", failures),
    }
    if include_candidate_i:
        i_artifact = safe_load_i(failures)
    else:
        i_artifact = None
        failures.append(
            {
                "model": "candidate_i_v1",
                "cycle_id": None,
                "reason": "skipped by default: causal replay rebuilds OOS Candidate F HGB history per cycle; use --include-candidate-i for a slow full replay.",
            }
        )
    j_artifact = safe_load_j(failures)
    j_thresholds = cjs.thresholds_from_artifact(j_artifact) if j_artifact else None
    g_history = ctx.g_traces.copy()
    for cycle_row in ctx.cycles.sort_values("data_cutoff").itertuples(index=False):
        cycle_targets = ctx.targets[ctx.targets["cycle_id"] == cycle_row.cycle_id].sort_values("target_order")
        cycle = build_cycle(str(cycle_row.cycle_id), pd.Timestamp(cycle_row.data_cutoff), cycle_targets)
        obs_until = ctx.observations[ctx.observations["observed_at"] <= pd.Timestamp(cycle_row.data_cutoff)].copy()
        rows.append(baseline_predictions(cycle, obs_lookup, "baseline_lag96", pd.Timedelta(days=1)))
        rows.append(latest_predictions(cycle, obs_until))
        trace = ctx.g_traces[ctx.g_traces["cycle_id"] == cycle.cycle_id] if not ctx.g_traces.empty else pd.DataFrame()
        if len(trace) == EXPECTED_TARGETS:
            rows.append(trace_predictions(trace, "candidate_f", "base_prediction"))
            rows.append(trace_predictions(trace, "candidate_g", "final_prediction"))
        else:
            failures.append({"model": "candidate_g", "cycle_id": cycle.cycle_id, "reason": "missing complete G trace"})
        for name, artifact in b_artifacts.items():
            if artifact is None:
                continue
            try:
                preds = build_shadow_predictions(cycle, name, artifact, obs_until)
                rows.append(prediction_objects_to_frame(name, preds))
            except Exception as exc:  # noqa: BLE001
                failures.append({"model": name, "cycle_id": cycle.cycle_id, "reason": f"{type(exc).__name__}: {exc}"})
        if i_artifact is not None:
            try:
                preds = ci.build_candidate_i_predictions(cycle, "candidate_i_v1", i_artifact, obs_until)
                rows.append(pd.DataFrame([{"model": "candidate_i_v1", "cycle_id": p.prediction.cycle_id, "station_id": p.prediction.station_id, "target_at": pd.Timestamp(p.prediction.target_at), "horizon_minutes": p.prediction.horizon_minutes, "prediction": p.prediction.prediction} for p in preds]))
            except Exception as exc:  # noqa: BLE001
                failures.append({"model": "candidate_i_v1", "cycle_id": cycle.cycle_id, "reason": f"{type(exc).__name__}: {exc}"})
        if j_thresholds is not None and len(trace) == EXPECTED_TARGETS:
            try:
                j = cjs.build_cycle_predictions(
                    cycle_targets=cycle_targets,
                    g_trace=trace,
                    g_history=g_history[(g_history["data_cutoff"] < pd.Timestamp(cycle_row.data_cutoff)) & (g_history["data_cutoff"] >= pd.Timestamp(cycle_row.data_cutoff) - cjs.HISTORY_WINDOW)],
                    observations=obs_until,
                    data_cutoff=pd.Timestamp(cycle_row.data_cutoff),
                    thresholds=j_thresholds,
                )
                rows.append(pd.DataFrame({"model": "candidate_j_v1_1", "cycle_id": cycle.cycle_id, "station_id": j["station_id"], "target_at": j["target_at"], "horizon_minutes": j["horizon_minutes"], "prediction": j["j_prediction"]}))
            except Exception as exc:  # noqa: BLE001
                failures.append({"model": "candidate_j_v1_1", "cycle_id": cycle.cycle_id, "reason": f"{type(exc).__name__}: {exc}"})
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not frame.empty:
        frame["target_at"] = pd.to_datetime(frame["target_at"], utc=True)
        frame["station_id"] = frame["station_id"].astype(str).str.zfill(5)
    return frame, failures


def baseline_predictions(cycle: ForecastCycle, obs_lookup: dict[tuple[str, pd.Timestamp], float], model: str, lag: pd.Timedelta) -> pd.DataFrame:
    rows = []
    for target in cycle.targets:
        target_at = pd.Timestamp(target.target_at).tz_convert("UTC")
        rows.append({"model": model, "cycle_id": cycle.cycle_id, "station_id": str(target.station_id).zfill(5), "target_at": target_at, "horizon_minutes": int(target.horizon_minutes), "prediction": obs_lookup.get((str(target.station_id).zfill(5), target_at - lag), np.nan)})
    return pd.DataFrame(rows).dropna(subset=["prediction"])


def latest_predictions(cycle: ForecastCycle, obs_until: pd.DataFrame) -> pd.DataFrame:
    latest = obs_until.sort_values("observed_at").groupby("station_id")["demand"].last().to_dict()
    rows = []
    for target in cycle.targets:
        sid = str(target.station_id).zfill(5)
        rows.append({"model": "baseline_latest", "cycle_id": cycle.cycle_id, "station_id": sid, "target_at": pd.Timestamp(target.target_at), "horizon_minutes": int(target.horizon_minutes), "prediction": latest.get(sid, np.nan)})
    return pd.DataFrame(rows).dropna(subset=["prediction"])


def trace_predictions(trace: pd.DataFrame, model: str, column: str) -> pd.DataFrame:
    return pd.DataFrame({"model": model, "cycle_id": trace["cycle_id"], "station_id": trace["station_id"], "target_at": trace["target_at"], "horizon_minutes": trace["horizon_minutes"], "prediction": trace[column].astype(float)})


def prediction_objects_to_frame(model: str, preds: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([{"model": model, "cycle_id": p.cycle_id, "station_id": p.station_id, "target_at": pd.Timestamp(p.target_at), "horizon_minutes": p.horizon_minutes, "prediction": p.prediction} for p in preds])


def attach_actuals(predictions: pd.DataFrame, ctx: ReplayContext) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    actual = ctx.observations.rename(columns={"observed_at": "target_at", "demand": "actual"})[["station_id", "target_at", "actual"]]
    scored = predictions.merge(actual, on=["station_id", "target_at"], how="inner")
    scored = scored.merge(ctx.cycles[["cycle_id", "data_cutoff"]], on="cycle_id", how="left")
    scored["abs_error"] = (scored["prediction"].astype(float) - scored["actual"].astype(float)).abs()
    scored["residual"] = scored["prediction"].astype(float) - scored["actual"].astype(float)
    return scored


def metric_dict(frame: pd.DataFrame) -> dict[str, Any]:
    summary = accuracy_summary_local(frame)
    return {
        "rows": int(len(frame)),
        "cycles": int(frame["cycle_id"].nunique()) if "cycle_id" in frame else 0,
        "mean_station_accuracy": summary["mean_station_accuracy"],
        "raw_wape": summary["global_wape"],
        "global_wape_accuracy": summary["global_wape_accuracy"],
        "mae": float(frame["abs_error"].mean()) if len(frame) else math.nan,
        "signed_bias": float(frame["residual"].mean()) if len(frame) else math.nan,
    }


def accuracy_summary_local(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "mean_station_accuracy": math.nan, "global_wape": math.nan, "global_wape_accuracy": math.nan}
    actual = frame["actual"].astype(float)
    abs_error = frame["abs_error"].astype(float)
    actual_sum = float(actual.sum())
    global_wape = float(abs_error.sum() / actual_sum) if actual_sum > 0 else math.nan
    station = frame.assign(_actual=actual, _abs_error=abs_error).groupby("station_id").agg(abs_error=("_abs_error", "sum"), actual=("_actual", "sum"))
    station = station[station["actual"] > 0]
    station_accuracy = 100.0 * np.maximum(0.0, 1.0 - station["abs_error"] / station["actual"])
    return {
        "rows": int(len(frame)),
        "mean_station_accuracy": float(station_accuracy.mean()) if len(station_accuracy) else math.nan,
        "global_wape": global_wape,
        "global_wape_accuracy": float(100.0 * max(0.0, 1.0 - global_wape)) if math.isfinite(global_wape) else math.nan,
    }


def summarize_available(scored: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"model": model, **metric_dict(group)} for model, group in scored.groupby("model")]).sort_values("mean_station_accuracy", ascending=False) if not scored.empty else pd.DataFrame()


def summarize_common(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()
    complete = scored.groupby(["model", "cycle_id"]).size().rename("rows").reset_index()
    complete = complete[complete["rows"] == EXPECTED_TARGETS]
    models = sorted(complete["model"].unique())
    counts = complete.groupby("cycle_id")["model"].nunique()
    common_ids = set(counts[counts == len(models)].index)
    common = scored[scored["cycle_id"].isin(common_ids) & scored["model"].isin(models)]
    summary = pd.DataFrame([{"model": model, **metric_dict(group)} for model, group in common.groupby("model")]).sort_values("mean_station_accuracy", ascending=False) if not common.empty else pd.DataFrame()
    cycles = cycle_table(common, None) if not common.empty else pd.DataFrame()
    return summary, cycles


def cycle_table(scored: pd.DataFrame, ctx: ReplayContext | None) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rows = []
    for (model, cycle_id), group in scored.groupby(["model", "cycle_id"]):
        rec = {"model": model, "cycle_id": cycle_id, **metric_dict(group)}
        rec["data_cutoff"] = group["data_cutoff"].iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["data_cutoff", "model"])


def grouped_table(scored: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rows = []
    for values, group in scored.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        rows.append({**dict(zip(keys, values, strict=True)), **metric_dict(group)})
    return pd.DataFrame(rows).sort_values(keys)


def regime_table(ctx: ReplayContext) -> pd.DataFrame:
    if ctx.cycles.empty or ctx.observations.empty:
        return pd.DataFrame(columns=["cycle_id", "regime"])
    try:
        metadata = cjs.load_artifact()
        thresholds = cjs.thresholds_from_artifact(metadata)
        wide = cj.observation_matrix(ctx.observations)
        anomaly = cj.hourly_anomaly(wide, cj.seasonal_baseline(wide))
        shares: dict[pd.Timestamp, float] = {}
        rows = []
        for row in ctx.cycles.sort_values("data_cutoff").itertuples(index=False):
            cutoff = pd.Timestamp(row.data_cutoff)
            visible = wide.loc[wide.index <= cutoff]
            previous = [shares.get(cutoff - pd.Timedelta(hours=k), 0.0) for k in range(1, cj.CANDIDATE_J_V1_1.shock_memory_hours)]
            regimes = cj.classify_regimes(cj.signals_at(cutoff, visible, anomaly.loc[anomaly.index <= cutoff]), thresholds, previous_shares=previous)
            shares[cutoff] = float(regimes["shock_share"].iloc[0])
            counts = regimes["regime"].value_counts().to_dict()
            if counts.get(cj.SEVERE_SHOCK, 0):
                label = cj.SEVERE_SHOCK
            elif counts.get(cj.SUSTAINED_DRIFT, 0):
                label = cj.SUSTAINED_DRIFT
            else:
                label = cj.NORMAL
            rows.append({"cycle_id": row.cycle_id, "data_cutoff": cutoff, "regime": label, "shock_share": shares[cutoff], **{f"regime_{k}": int(v) for k, v in counts.items()}})
        return pd.DataFrame(rows)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame({"cycle_id": ctx.cycles["cycle_id"], "data_cutoff": ctx.cycles["data_cutoff"], "regime": f"UNKNOWN: {type(exc).__name__}"})


def wins_table(per_cycle: pd.DataFrame) -> pd.DataFrame:
    if per_cycle.empty:
        return pd.DataFrame()
    pivot = per_cycle.pivot(index="cycle_id", columns="model", values="mean_station_accuracy")
    rows = []
    for ref in ["candidate_g", "candidate_j_v1_1"]:
        if ref not in pivot:
            continue
        for model in pivot.columns:
            if model == ref:
                continue
            delta = pivot[model] - pivot[ref]
            delta = delta.dropna()
            rows.append({"model": model, "reference": ref, "cycles": int(len(delta)), "wins": int((delta > 1e-9).sum()), "losses": int((delta < -1e-9).sum()), "ties": int((delta.abs() <= 1e-9).sum()), "mean_delta_pp": float(delta.mean()) if len(delta) else math.nan})
    return pd.DataFrame(rows)


def temporal_wide(per_cycle: pd.DataFrame) -> pd.DataFrame:
    if per_cycle.empty:
        return pd.DataFrame()
    wide = per_cycle.pivot_table(index=["cycle_id", "data_cutoff", "regime"], columns="model", values="mean_station_accuracy").reset_index()
    if "candidate_g" in wide:
        for col in [c for c in wide.columns if c not in {"cycle_id", "data_cutoff", "regime"}]:
            wide[f"{col}_delta_vs_candidate_g"] = wide[col] - wide["candidate_g"]
    if "candidate_j_v1_1" in wide:
        for col in [c for c in wide.columns if c not in {"cycle_id", "data_cutoff", "regime"} and not c.endswith("_delta_vs_candidate_g")]:
            wide[f"{col}_delta_vs_candidate_j_v1_1"] = wide[col] - wide["candidate_j_v1_1"]
    return wide.sort_values("data_cutoff")


def model_inventory(model_versions: pd.DataFrame) -> pd.DataFrame:
    registered = set(model_versions["model_name"].astype(str)) if not model_versions.empty else set()
    rows = [
        inv("baseline_lag96", "daily persistence", "scripts/model_zoo_recent_replay.py", "", True, "", "baseline", "Uses target_at - 24h, causal for all horizons."),
        inv("baseline_latest", "latest observation", "scripts/model_zoo_recent_replay.py", "", True, "", "baseline", "Repeats latest station observation at data_cutoff."),
        inv("candidate_b_v1", "candidate_b_v1", "src/candidate_b_shadow.py", "artifacts/shadow/candidate_b_v1/model.pkl", True, "", "shadow" if "candidate_b_hist_gradient_boosting_shadow" in registered else "artifact-only", "Frozen HGB artifact."),
        inv("candidate_b_v2", "candidate_b_v2", "src/candidate_b_shadow.py", "artifacts/shadow/candidate_b_v2/model.pkl", True, "", "artifact-only", "Frozen HGB artifact with later training cutoff."),
        inv("candidate_c", "candidate_c experiments", "scripts/candidate_c_experiments.py", "reports/candidate_c/", False, "Experiment reports exist, but no frozen production artifact for this live window.", "experimental/lab", ""),
        inv("candidate_d1", "residual calibration", "scripts/candidate_d1_residual_calibration.py", "reports/candidate_d1/", False, "Report artifacts only; dispatch/runtime wrapper is not available in this worktree.", "experimental/lab", ""),
        inv("candidate_d2", "context audit", "", "", False, "No replayable model implementation in this worktree.", "experimental/lab", ""),
        inv("candidate_e", "recency lab", "", "", False, "No script/artifact present on origin/student-work worktree.", "experimental/lab", ""),
        inv("candidate_f", "profile-ratio HGB base", "src/candidate_g_adaptive.py::build_candidate_f_base_frame; production G traces", "implicit inside Candidate G", True, "", "production component", "Scored from G trace base_prediction."),
        inv("candidate_g", "candidate_g_v1", "src/candidate_g_adaptive.py", "artifacts/shadow/candidate_g_v1/metadata.json", True, "", "registered/production-or-shadow", "Scored from persisted prediction_traces final_prediction."),
        inv("candidate_h", "guard lab", "", "", False, "No script/artifact present on origin/student-work worktree.", "experimental/lab", ""),
        inv("candidate_i_v1", "candidate_i_v1", "src/candidate_i_snr_gate.py", "artifacts/shadow/candidate_i_v1/metadata.json", True, "", "shadow", "Causal OOS F history rebuilt per cycle; skipped by default in retrospective replay because it refits HGB history for every cycle."),
        inv("candidate_j_v1_1", "candidate_j_v1_1", "src/candidate_j_shadow.py", "artifacts/shadow/candidate_j_v1_1/metadata.json", True, "", "shadow/official-capable", "No tuning; wraps Candidate G trace causally."),
    ]
    return pd.DataFrame(rows)


def inv(model: str, version: str, implementation: str, artifact: str, replayable: bool, reason: str, production: str, notes: str) -> dict[str, Any]:
    return {"model": model, "version": version, "implementation": implementation, "artifact": artifact, "replayable": replayable, "reason_if_not_replayable": reason, "production_or_experimental": production, "notes": notes}


def write_outputs(out: Path, payload: dict[str, Any], inventory: pd.DataFrame, available: pd.DataFrame, common: pd.DataFrame, per_cycle: pd.DataFrame, per_station: pd.DataFrame, per_horizon: pd.DataFrame, common_cycles: pd.DataFrame, temporal: pd.DataFrame, scored: pd.DataFrame) -> None:
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    inventory.to_csv(out / "model_inventory.csv", index=False)
    per_cycle.to_csv(out / "per_cycle.csv", index=False)
    per_station.to_csv(out / "per_station.csv", index=False)
    per_horizon.to_csv(out / "per_horizon.csv", index=False)
    common_cycles.to_csv(out / "common_cycles.csv", index=False)
    temporal.to_csv(out / "temporal_wide.csv", index=False)
    scored.to_csv(out / "scored_predictions.csv", index=False)
    (out / "latest.md").write_text(render_markdown(payload, inventory, available, common, per_cycle, per_station, per_horizon, common_cycles, temporal) + "\n")


def render_markdown(payload: dict[str, Any], inventory: pd.DataFrame, available: pd.DataFrame, common: pd.DataFrame, per_cycle: pd.DataFrame, per_station: pd.DataFrame, per_horizon: pd.DataFrame, common_cycles: pd.DataFrame, temporal: pd.DataFrame) -> str:
    lines = [
        "# Model Zoo Recent Replay Benchmark",
        "",
        f"- Generated at: `{payload['generated_at_utc']}`",
        f"- Real arrival-time window: `{payload['real_window_start']}` -> `{payload['real_window_end'] or 'latest available'}`",
        f"- Historical data_cutoff window: `{payload['historical_data_cutoff_min']}` -> `{payload['historical_data_cutoff_max']}`",
        f"- Historical target_at max: `{payload['historical_target_at_max']}`",
        f"- Complete resolved cycles: `{payload['cycles_requested']}`",
        f"- Common-cycle count: `{payload['common_cycle_count']}`",
        "",
        "## Reproducibility Inventory",
        "",
        md_table(inventory),
        "",
        "## Available Data View",
        "",
        md_table(available),
        "",
        "## Common Cycles View",
        "",
        md_table(common),
        "",
        "## Wins / Losses / Ties",
        "",
        md_table(pd.DataFrame(payload["wins_vs_g_and_j"])),
        "",
        "## Severe Shock / Regime Notes",
        "",
    ]
    if not per_cycle.empty and "regime" in per_cycle:
        lines.append(md_table(per_cycle.groupby(["model", "regime"]).apply(lambda g: pd.Series(metric_dict_from_cycle_rows(g)), include_groups=False).reset_index()))
    else:
        lines.append("No regime table available.")
    lines += [
        "",
        "## Best / Worst Patterns",
        "",
        "- Worst cycles and station/horizon slices are in `per_cycle.csv`, `per_station.csv`, and `per_horizon.csv`.",
        "- `temporal_wide.csv` contains cycle-by-cycle model accuracy plus deltas vs Candidate G and Candidate J v1.1.",
        "- Figures are in `figures/`.",
        "",
        "## Replay Failures / Non-scored Cases",
        "",
        md_table(pd.DataFrame(payload["replay_failures"])) if payload["replay_failures"] else "None.",
        "",
        "## Candidate K Design Hints",
        "",
        "- Reuse robust behavior by regime rather than a single global policy.",
        "- Compare station/horizon slices where older baselines beat adaptive models; those are useful fallback signals.",
        "- Treat severe-shock hours as test evidence only; do not tune Candidate K from these outputs without a new protocol.",
    ]
    return "\n".join(lines)


def metric_dict_from_cycle_rows(frame: pd.DataFrame) -> dict[str, Any]:
    return {"cycles": int(frame["cycle_id"].nunique()), "mean_cycle_accuracy": float(frame["mean_station_accuracy"].mean()), "stability_std": float(frame["mean_station_accuracy"].std(ddof=0)), "worst_cycle_accuracy": float(frame["mean_station_accuracy"].min()), "best_cycle_accuracy": float(frame["mean_station_accuracy"].max())}


def md_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame is None or frame.empty:
        return "_(no data)_"
    show = frame.head(max_rows).copy()
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(show[c]) else "---" for c in cols) + "|"]
    for row in show.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"| ... {len(frame) - max_rows} more rows | | | | | | | | |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, pd.Timestamp):
        return _ts(value)
    return str(value).replace("|", "\\|")


def make_figures(per_cycle: pd.DataFrame, out: Path) -> None:
    if per_cycle.empty:
        return
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        for model, group in per_cycle.sort_values("data_cutoff").groupby("model"):
            ax.plot(group["data_cutoff"], group["mean_station_accuracy"], marker="o", label=model)
        ax.set_title("Cycle accuracy by model")
        ax.set_ylabel("leaderboard mean_station_accuracy")
        ax.legend(loc="best", fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / "cycle_accuracy.png", dpi=150)
        plt.close(fig)
    except Exception:
        return


def safe_load_b(uri: str, failures: list[dict[str, Any]]) -> Any | None:
    try:
        return load_shadow_artifact(uri)
    except Exception as exc:  # noqa: BLE001
        failures.append({"model": uri, "cycle_id": None, "reason": f"artifact load failed: {type(exc).__name__}: {exc}"})
        return None


def safe_load_i(failures: list[dict[str, Any]]) -> Any | None:
    try:
        return ci.load_candidate_i_artifact(ci.DEFAULT_ARTIFACT_URI)
    except Exception as exc:  # noqa: BLE001
        failures.append({"model": "candidate_i_v1", "cycle_id": None, "reason": f"artifact load failed: {type(exc).__name__}: {exc}"})
        return None


def safe_load_j(failures: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return cjs.load_artifact()
    except Exception as exc:  # noqa: BLE001
        failures.append({"model": "candidate_j_v1_1", "cycle_id": None, "reason": f"artifact load failed: {type(exc).__name__}: {exc}"})
        return None


def build_cycle(cycle_id: str, cutoff: pd.Timestamp, targets: pd.DataFrame) -> ForecastCycle:
    return ForecastCycle(
        cycle_id=cycle_id,
        data_cutoff=_ts(cutoff),
        closes_at="",
        expected_predictions=len(targets),
        targets=[
            CycleTarget(station_id=str(row.station_id), target_at=_ts(row.target_at), horizon_minutes=int(row.horizon_minutes), target_order=int(row.target_order), raw_target={})
            for row in targets.itertuples(index=False)
        ],
        raw_cycle={},
    )


def observations_lookup(observations: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    return {(str(row.station_id).zfill(5), pd.Timestamp(row.observed_at)): float(row.demand) for row in observations.itertuples(index=False)}


def _model_id(models: pd.DataFrame, *, model_name: str | None = None, model_type: str | None = None) -> str | None:
    if models.empty:
        return None
    frame = models.copy()
    if model_name is not None:
        frame = frame[frame["model_name"] == model_name]
    if model_type is not None:
        frame = frame[frame["model_type"] == model_type]
    if frame.empty:
        return None
    if "trained_until" in frame:
        frame = frame.sort_values("trained_until")
    return str(frame.iloc[-1]["model_id"])


def _in_filter(values: list[str]) -> str:
    escaped = ",".join(values)
    return f"in.({escaped})"


def _ts(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _maybe_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _ts(value)


def load_env() -> None:
    candidates = [ROOT / ".env.local", ROOT.parent / "pulso-transmi-sdk" / ".env.local"]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in {"SUPABASE_URL", "SUPABASE_SERVICE_KEY", "SUPABASE_KEY"}:
                os.environ.setdefault(key, value.strip())


if __name__ == "__main__":
    main()
