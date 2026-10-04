"""Read-only causal replay of frozen K after the observation schema-v2 collector repair.

No submissions, registry writes or historical prediction rewrites. Missing truth remains missing.
V2 inputs additionally respect their real released_at relative to each cycle's first_seen_at.
"""
from __future__ import annotations

import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def eligible_history(observations, releases, cutoff, prediction_time):
    """Virtual-time causal input, with a real publication-time gate for changed-schema rows."""
    cutoff = pd.Timestamp(cutoff)
    prediction_time = pd.Timestamp(prediction_time)
    frame = observations.copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame = frame[frame.observed_at <= cutoff]
    raw = releases.copy()
    raw["observed_at"] = pd.to_datetime(raw["observed_at"], utc=True)
    raw["released_at"] = pd.to_datetime(raw["released_at"], utc=True)
    keys = pd.MultiIndex.from_frame(raw[["station_id", "observed_at"]])
    frame = frame[~pd.MultiIndex.from_frame(frame[["station_id", "observed_at"]]).isin(keys)]
    raw = raw[(raw.observed_at <= cutoff) & (raw.released_at <= prediction_time)]
    raw = raw.sort_values("released_at").drop_duplicates(["station_id", "observed_at"], keep="last")
    restored = []
    for row in raw.itertuples(index=False):
        if row.unit != "passengers" or row.quality != "observed" or row.value is None:
            continue
        value = Decimal(str(row.value))
        if not value.is_finite() or value < 0 or value != value.to_integral_value():
            continue
        restored.append({"station_id": row.station_id, "observed_at": row.observed_at, "demand": int(value)})
    return pd.concat([frame[["station_id", "observed_at", "demand"]], pd.DataFrame(restored)], ignore_index=True).sort_values(["observed_at", "station_id"])


def metrics(frame, column):
    from drift_monitor import accuracy_summary

    scored = frame[frame.actual.notna() & frame[column].notna()].copy()
    scored["abs_error"] = (scored[column] - scored.actual).abs()
    return accuracy_summary(scored)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def deny_writes(request):
    if request.method != "GET":
        raise RuntimeError("audit refuses database mutations")


def main():
    if os.getenv("PULSO_API_KEY"):
        raise RuntimeError("This audit must not have a competition API key")
    import candidate_k_shadow as cks
    from candidate_k_runtime import FrozenKPredictor
    from run_monitoring_snapshot import SupabaseMonitoringStore, TABLE_SELECTS

    output = ROOT / "reports/k_schema_v2_replay/output"
    output.mkdir(parents=True, exist_ok=True)
    releases = pd.DataFrame(json.loads((ROOT / "reports/k_schema_v2_replay/input_release_snapshot.json").read_text()))
    with SupabaseMonitoringStore(timeout=60) as store:
        store._client.event_hooks["request"].append(deny_writes)
        observations = store.table("observations", select=TABLE_SELECTS["observations"], order="observed_at.asc,station_id.asc")
        cycles = store.table("forecast_cycles", select="cycle_id,data_cutoff,closes_at,expected_predictions,first_seen_at", order="data_cutoff.desc", limit=4000)
        targets = store.table("forecast_cycle_targets", select=TABLE_SELECTS["forecast_cycle_targets"], order="target_at.desc,cycle_id.desc,target_order.asc", limit=15000)
        predictions = store.table("predictions", select=TABLE_SELECTS["predictions"], order="created_at.desc,prediction_id.asc", limit=15000)
        models = store.table("model_versions", select=TABLE_SELECTS["model_versions"])
        receipts = store.table("submission_receipts", select=TABLE_SELECTS["submission_receipts"], order="created_at.desc", limit=4000)
        traces = store.table("prediction_traces", select=TABLE_SELECTS["prediction_traces"], order="created_at.desc,cycle_id.asc,model_id.asc,station_id.asc,target_at.asc", limit=3000)

    observations["observed_at"] = pd.to_datetime(observations.observed_at, utc=True)
    targets["target_at"] = pd.to_datetime(targets.target_at, utc=True)
    cycles["data_cutoff"] = pd.to_datetime(cycles.data_cutoff, utc=True)
    predictions["target_observed_at"] = pd.to_datetime(predictions.target_observed_at, utc=True)
    latest = observations.observed_at.max()
    k = models[models.model_name == cks.MODEL_NAME].iloc[0]
    metadata = cks.load_artifact(str(k.artifact_uri))
    predictor = FrozenKPredictor(metadata, cks.artifact_dir(str(k.artifact_uri)), os.getenv("CANDIDATE_K_CACHE_DIR"), log=print)
    end_by_cycle = targets.groupby("cycle_id").target_at.max()
    selected = cycles[(cycles.data_cutoff >= pd.Timestamp("2026-09-20T12:00:00Z")) & cycles.cycle_id.isin(end_by_cycle[end_by_cycle <= latest].index)].sort_values("data_cutoff").tail(12)
    columns = ["k_replay", "k_seasonal_expert", "official_as_sent", "baseline_latest", "baseline_lag96"]
    actuals = observations.rename(columns={"observed_at": "target_at", "demand": "actual"})
    frames, per_cycle = [], []
    for cycle in selected.itertuples(index=False):
        wanted = targets[targets.cycle_id == cycle.cycle_id].sort_values("target_order")
        if len(wanted) != int(cycle.expected_predictions):
            raise RuntimeError("incomplete target set")
        history = eligible_history(observations, releases, cycle.data_cutoff, cycle.first_seen_at)
        assert history.observed_at.max() <= cycle.data_cutoff
        frame, info = predictor.predict(history, cycle.data_cutoff, wanted[["station_id", "target_at", "horizon_minutes"]])
        frame = cks.validate_cycle_output(frame, wanted[["station_id", "target_at", "horizon_minutes"]])
        frame = frame.rename(columns={"k_prediction": "k_replay", "k_seasonal_prediction": "k_seasonal_expert"})
        frame = frame.merge(actuals[["station_id", "target_at", "actual"]], on=["station_id", "target_at"], how="left", validate="one_to_one")
        old = predictions[(predictions.cycle_id == cycle.cycle_id) & (predictions.model_id == str(k.model_id))]
        if len(old) != int(cycle.expected_predictions):
            raise RuntimeError("cannot compare against incomplete stored K predictions")
        frame = frame.merge(old[["station_id", "target_observed_at", "prediction"]].rename(columns={"target_observed_at": "target_at", "prediction": "official_as_sent"}), on=["station_id", "target_at"], validate="one_to_one")
        recent = history.sort_values("observed_at").groupby("station_id").tail(1).set_index("station_id").demand
        lag = history.set_index(["station_id", "observed_at"]).demand
        frame["baseline_latest"] = frame.station_id.map(recent)
        frame["baseline_lag96"] = [lag.get((str(r.station_id), r.target_at - pd.Timedelta(days=1)), float("nan")) for r in frame.itertuples(index=False)]
        frame["cycle_id"] = cycle.cycle_id
        frame["data_cutoff"] = cycle.data_cutoff
        frame["prediction_time"] = cycle.first_seen_at
        frames.append(frame)
        if info.get("train_data_sha256") != info.get("current_data_sha256"):
            raise RuntimeError("cached K training data differs from the causal replay training history")
        record = {"cycle_id": cycle.cycle_id, "data_cutoff": str(cycle.data_cutoff), "expected_rows": len(frame), "revealed_rows": int(frame.actual.notna().sum()), "refit_at": info.get("refit_at"), "cache": info.get("cache"), "train_data_sha256": info.get("train_data_sha256"), "current_data_sha256": info.get("current_data_sha256"), "mean_seasonal_weight": float(frame.k_seasonal_weight.mean()), "metrics": {c: metrics(frame, c) for c in columns}}
        per_cycle.append(record)
        print("AUDIT_CYCLE=" + json.dumps(clean(record), separators=(",", ":")), flush=True)

    if not frames:
        raise RuntimeError("No revealed cycles available for replay")
    all_rows = pd.concat(frames, ignore_index=True)
    windows = {}
    for size in (6, 12):
        ids = selected.cycle_id.tolist()[-size:]
        window = all_rows[all_rows.cycle_id.isin(ids)]
        complete_ids = [c for c, g in window.groupby("cycle_id") if g.actual.notna().all()]
        windows[str(size)] = {"cycles": len(ids), "first_cutoff": str(window.data_cutoff.min()), "last_cutoff": str(window.data_cutoff.max()), "expected_rows": len(window), "revealed_rows": int(window.actual.notna().sum()), "complete_cycles": len(complete_ids), "revealed_coverage": float(window.actual.notna().mean()), "observed_targets_metrics": {c: metrics(window, c) for c in columns}, "complete_cycles_metrics": {c: metrics(window[window.cycle_id.isin(complete_ids)], c) for c in columns}}

    recent = all_rows[all_rows.cycle_id.isin(selected.cycle_id.tolist()[-6:])]
    station_metrics = [{"station_id": str(s), "expected_rows": len(g), "revealed_rows": int(g.actual.notna().sum()), **{c: metrics(g, c)["mean_station_accuracy"] for c in columns}} for s, g in recent.groupby("station_id")]
    live = receipts[(receipts.model_id == str(k.model_id)) & (receipts.is_official == True) & (pd.to_datetime(receipts.accepted_at, utc=True) >= pd.Timestamp("2026-10-04T11:45Z"))]
    live_rows = predictions[(predictions.model_id == str(k.model_id)) & predictions.cycle_id.isin(live.cycle_id)].rename(columns={"target_observed_at": "target_at"}).merge(actuals[["station_id", "target_at", "actual"]], on=["station_id", "target_at"], how="left")
    live_metrics = metrics(live_rows, "prediction") if not live_rows.empty else {"rows": 0, "mean_station_accuracy": None}
    live_traces = traces[(traces.model_id == str(k.model_id)) & traces.cycle_id.isin(live.cycle_id)]
    summary = clean({"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "model_name": str(k.model_name), "model_id": str(k.model_id), "model_type": str(k.model_type), "artifact_uri": str(k.artifact_uri), "latest_revealed_observation": str(latest), "evaluation": "retrospective replay of unchanged frozen K with repaired collector inputs; not new official submissions", "causality": "observed_at<=data_cutoff; v2 released_at<=cycle.first_seen_at; daily refit only observed targets<=refit_time", "database_writes": False, "parameter_changes": False, "windows": windows, "per_cycle": per_cycle, "station_metrics_recent6": station_metrics, "post_fix_official": {"cycles": int(live.cycle_id.nunique()), "expected_rows": len(live_rows), "revealed_rows": int(live_rows.actual.notna().sum()) if not live_rows.empty else 0, "native_k_trace_rows": int(((live_traces.feedback_samples > 0) & live_traces.calibration_factor.notna()).sum()) if not live_traces.empty else 0, "metrics": live_metrics}})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    all_rows.to_csv(output / "replay_predictions.csv", index=False)
    pd.DataFrame(station_metrics).to_csv(output / "station_metrics_recent6.csv", index=False)
    lines = ["# K schema-v2 causal replay", "", "This is a retrospective backtest, not the live leaderboard. Missing truth is excluded and coverage is reported.", "", "| Window | Revealed/expected | K replay | Official as sent | Latest-value baseline |", "| --- | --- | --- | --- | --- |"]
    for size, w in summary["windows"].items():
        m = w["observed_targets_metrics"]
        lines.append(f"| Last {size} cycles | {w['revealed_rows']}/{w['expected_rows']} | {m['k_replay']['mean_station_accuracy']:.4f}% | {m['official_as_sent']['mean_station_accuracy']:.4f}% | {m['baseline_latest']['mean_station_accuracy']:.4f}% |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print("AUDIT_SUMMARY=" + json.dumps(summary, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
