"""Out-of-sample evaluator for Candidate G vs Candidate J shadow.

Only uses cycles after Candidate J's design freeze that are:
- shared by G and J;
- complete for both models (48 predictions);
- fully revealed (48 observations available).

This script is read-only. It does not tune Candidate J, register models, promote models,
write predictions, or submit anything.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import candidate_j_regime as cj  # noqa: E402
import candidate_j_shadow as cjs  # noqa: E402
from drift_monitor import accuracy_summary  # noqa: E402


REPORT_DIR = ROOT / "reports" / "candidate_j_shadow_oos"


class ReadOnlyStore:
    def __init__(self, timeout: float = 45.0) -> None:
        _load_env_file()
        url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self._client = httpx.Client(
            base_url=f"{url}/rest/v1",
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ReadOnlyStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def table(self, name: str, params: dict[str, str], *, page_size: int = 1000) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            response = self._client.get(f"/{name}", params={**params, "limit": str(page_size), "offset": str(offset)})
            response.raise_for_status()
            page = response.json()
            rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--min-new-severe-shock-hours", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload, tables = evaluate(min_new_severe_shock_hours=args.min_new_severe_shock_hours)
    (args.output_dir / "latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    (args.output_dir / "latest.md").write_text(render_markdown(payload) + "\n")
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    print(json.dumps({"verdict": payload["verdict"], "cycles": payload["cycles"], "new_severe_shock_hours": payload["new_severe_shock_hours"]}, sort_keys=True))


def evaluate(*, min_new_severe_shock_hours: int) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    with ReadOnlyStore() as store:
        model_versions = store.table("model_versions", {"select": "model_id,model_name,model_type,is_active,feature_config,artifact_uri"})
        g_model = _one(model_versions[(model_versions["is_active"] == True) & (model_versions["model_name"] == cjs.CHAMPION_MODEL_NAME)], "active Candidate G")
        j_rows = model_versions[model_versions["model_type"] == cjs.MODEL_TYPE].copy()
        j_rows = j_rows[j_rows["feature_config"].map(lambda v: isinstance(v, dict) and v.get("version") == cjs.MODEL_VERSION)]
        j_model = _one(j_rows, "Candidate J shadow")
        if bool(j_model["is_active"]):
            raise RuntimeError("Candidate J is active; expected shadow is_active=false")

        freeze = pd.Timestamp(cjs.OOS_FROM_CUTOFF)
        cycles = store.table(
            "forecast_cycles",
            {"select": "cycle_id,data_cutoff,expected_predictions", "data_cutoff": f"gte.{_ts(freeze)}", "order": "data_cutoff.asc"},
        )
        if cycles.empty:
            return _empty_payload(g_model, j_model, "no post-freeze cycles"), {}
        cycle_ids = cycles["cycle_id"].astype(str).tolist()
        ids_filter = f"in.({','.join(cycle_ids)})"
        predictions = store.table(
            "predictions",
            {"select": "cycle_id,model_id,station_id,target_observed_at,horizon_minutes,prediction", "cycle_id": ids_filter},
            page_size=2000,
        )
        targets = store.table(
            "forecast_cycle_targets",
            {"select": "cycle_id,station_id,target_at,horizon_minutes,target_order", "cycle_id": ids_filter},
            page_size=2000,
        )
        traces = store.table(
            "prediction_traces",
            {"select": "cycle_id,model_id,station_id,target_at,horizon_minutes,base_prediction,calibration_factor,final_prediction,data_cutoff", "cycle_id": ids_filter, "model_id": f"eq.{g_model['model_id']}"},
            page_size=2000,
        )
        latest_target = pd.to_datetime(targets["target_at"], utc=True).max() if not targets.empty else freeze
        obs_start = freeze - cjs.OBSERVATION_LOOKBACK - pd.Timedelta(days=7 * 4)
        observations = store.table(
            "observations",
            {"select": "station_id,observed_at,demand", "observed_at": f"gte.{_ts(obs_start)}", "order": "observed_at.asc,station_id.asc"},
            page_size=5000,
        )

    scored = build_scored(cycles, predictions, targets, observations, str(g_model["model_id"]), str(j_model["model_id"]))
    shared = complete_shared_cycles(scored, cycles, g_model_id=str(g_model["model_id"]), j_model_id=str(j_model["model_id"]))
    if shared.empty:
        return _empty_payload(g_model, j_model, "no complete shared revealed cycles"), {"scored_rows": scored}

    metadata = cjs.load_artifact(str(j_model["artifact_uri"] or cjs.DEFAULT_ARTIFACT_URI))
    thresholds = cjs.thresholds_from_artifact(metadata)
    regime_table = detect_regimes(shared[["cycle_id", "data_cutoff"]].drop_duplicates(), observations, thresholds)
    joined = shared.merge(regime_table[["cycle_id", "regime_detected"]], on="cycle_id", how="left")
    overall = overall_metrics(joined)
    station = grouped_metrics(joined, ["station_id"])
    horizon = grouped_metrics(joined, ["horizon_minutes"])
    cycles_df = cycle_metrics(joined, regime_table)
    wins = cycles_df["result"].value_counts().reindex(["win", "loss", "tie"], fill_value=0).to_dict()
    severe_cycles = cycles_df[cycles_df["regime_detected"].str.contains(cj.SEVERE_SHOCK, na=False)]
    new_severe_shock_hours = int(severe_cycles["cycle_id"].nunique())
    recovery_observed = bool((cycles_df["regime_detected"] != cj.SEVERE_SHOCK).tail(3).all()) if not cycles_df.empty else False
    verdict = "continue_shadow"
    if new_severe_shock_hours < min_new_severe_shock_hours and not recovery_observed:
        verdict = "insufficient_out_of_sample_evidence"

    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "design_freeze_cutoff": cjs.OOS_FROM_CUTOFF,
        "g_model_id": str(g_model["model_id"]),
        "j_model_id": str(j_model["model_id"]),
        "j_is_active": bool(j_model["is_active"]),
        "cycles": int(joined["cycle_id"].nunique()),
        "rows": int(len(joined)),
        "first_data_cutoff": _ts(joined["data_cutoff"].min()),
        "last_data_cutoff": _ts(joined["data_cutoff"].max()),
        "metrics": overall,
        "wins_losses_ties": {k: int(v) for k, v in wins.items()},
        "new_severe_shock_hours": new_severe_shock_hours,
        "min_new_severe_shock_hours": min_new_severe_shock_hours,
        "recovery_observed": recovery_observed,
        "verdict": verdict,
        "decision_rule": "Do not adapt/tune J with this OOS result. Keep continue_shadow unless >=12 new severe-shock hours or shock recovery is observed.",
    }
    return payload, {"cycle_metrics": cycles_df, "station_metrics": station, "horizon_metrics": horizon, "scored_rows": joined}


def build_scored(cycles: pd.DataFrame, predictions: pd.DataFrame, targets: pd.DataFrame, observations: pd.DataFrame, g_model_id: str, j_model_id: str) -> pd.DataFrame:
    for frame, cols in ((cycles, ["data_cutoff"]), (predictions, ["target_observed_at"]), (targets, ["target_at"]), (observations, ["observed_at"])):
        for col in cols:
            if col in frame:
                frame[col] = pd.to_datetime(frame[col], utc=True)
    observations["station_id"] = observations["station_id"].astype(str).str.zfill(5)
    targets["station_id"] = targets["station_id"].astype(str).str.zfill(5)
    predictions["station_id"] = predictions["station_id"].astype(str).str.zfill(5)
    frame = predictions[predictions["model_id"].astype(str).isin([g_model_id, j_model_id])].merge(
        targets[["cycle_id", "station_id", "target_at", "horizon_minutes"]],
        left_on=["cycle_id", "station_id", "target_observed_at"],
        right_on=["cycle_id", "station_id", "target_at"],
        how="inner",
    )
    frame = frame.merge(observations[["station_id", "observed_at", "demand"]], left_on=["station_id", "target_at"], right_on=["station_id", "observed_at"], how="inner")
    frame = frame.merge(cycles[["cycle_id", "data_cutoff", "expected_predictions"]], on="cycle_id", how="inner")
    frame = frame.rename(columns={"demand": "actual"})
    return frame


def complete_shared_cycles(scored: pd.DataFrame, cycles: pd.DataFrame, *, g_model_id: str, j_model_id: str) -> pd.DataFrame:
    if scored.empty:
        return scored
    expected = cycles.set_index("cycle_id")["expected_predictions"].astype(int)
    counts = scored.groupby(["cycle_id", "model_id"]).size().rename("rows").reset_index()
    counts["expected"] = counts["cycle_id"].map(expected)
    complete = counts[counts["rows"] == counts["expected"]]
    complete_models = complete.groupby("cycle_id")["model_id"].nunique()
    shared_ids = set(complete_models[complete_models >= 2].index)
    frame = scored[scored["cycle_id"].isin(shared_ids)].copy()
    pivot = frame.pivot_table(index=["cycle_id", "data_cutoff", "station_id", "target_at", "horizon_minutes", "actual"], columns="model_id", values="prediction", aggfunc="last").reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    if g_model_id not in pivot.columns or j_model_id not in pivot.columns:
        return pd.DataFrame()
    return pivot.rename(columns={g_model_id: "g_prediction", j_model_id: "j_prediction"})


def detect_regimes(cycles: pd.DataFrame, observations: pd.DataFrame, thresholds: cj.RegimeThresholds) -> pd.DataFrame:
    obs = observations.copy()
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    wide = cj.observation_matrix(obs)
    anomaly = cj.hourly_anomaly(wide, cj.seasonal_baseline(wide))
    rows = []
    shares: dict[pd.Timestamp, float] = {}
    for row in cycles.sort_values("data_cutoff").itertuples(index=False):
        cutoff = pd.Timestamp(row.data_cutoff)
        previous = [shares.get(cutoff - pd.Timedelta(hours=k), 0.0) for k in range(1, cj.CANDIDATE_J_V1_1.shock_memory_hours)]
        regimes = cj.classify_regimes(cj.signals_at(cutoff, wide.loc[wide.index <= cutoff], anomaly.loc[anomaly.index <= cutoff]), thresholds, previous_shares=previous)
        share = float(regimes["shock_share"].iloc[0]) if not regimes.empty else 0.0
        shares[cutoff] = share
        counts = regimes["regime"].value_counts().to_dict()
        label = cj.SEVERE_SHOCK if counts.get(cj.SEVERE_SHOCK, 0) else ";".join(sorted(counts))
        rows.append({"cycle_id": row.cycle_id, "data_cutoff": cutoff, "regime_detected": label, "shock_share": share, **{f"regime_{k}": int(v) for k, v in counts.items()}})
    return pd.DataFrame(rows)


def overall_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    g = _metrics(frame, "g_prediction")
    j = _metrics(frame, "j_prediction")
    return {"G": g, "J": j, "delta_mean_station_accuracy_pp": j["mean_station_accuracy"] - g["mean_station_accuracy"]}


def grouped_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        g, j = _metrics(group, "g_prediction"), _metrics(group, "j_prediction")
        rows.append({**dict(zip(keys, values, strict=True)), "g_accuracy": g["mean_station_accuracy"], "j_accuracy": j["mean_station_accuracy"], "delta_pp": j["mean_station_accuracy"] - g["mean_station_accuracy"], "g_wape": g["global_wape"], "j_wape": j["global_wape"], "rows": len(group)})
    return pd.DataFrame(rows).sort_values(keys)


def cycle_metrics(frame: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cycle_id, cutoff), group in frame.groupby(["cycle_id", "data_cutoff"]):
        g, j = _metrics(group, "g_prediction"), _metrics(group, "j_prediction")
        delta = j["mean_station_accuracy"] - g["mean_station_accuracy"]
        rows.append({"cycle_id": cycle_id, "data_cutoff": cutoff, "g_mean_station_accuracy": g["mean_station_accuracy"], "j_mean_station_accuracy": j["mean_station_accuracy"], "delta_pp": delta, "result": "win" if delta > 1e-9 else "loss" if delta < -1e-9 else "tie", "g_wape": g["global_wape"], "j_wape": j["global_wape"], "g_mae": g["mae"], "j_mae": j["mae"], "g_signed_bias": g["signed_bias"], "j_signed_bias": j["signed_bias"]})
    return pd.DataFrame(rows).merge(regimes, on=["cycle_id", "data_cutoff"], how="left").sort_values("data_cutoff")


def _metrics(frame: pd.DataFrame, column: str) -> dict[str, float]:
    scored = frame.assign(prediction=frame[column].astype(float), abs_error=(frame[column].astype(float) - frame["actual"].astype(float)).abs())
    summary = accuracy_summary(scored)
    residual = scored[column].astype(float) - scored["actual"].astype(float)
    return {
        "mean_station_accuracy": float(summary["mean_station_accuracy"]),
        "raw_wape": float(summary["global_wape"]),
        "global_wape": float(summary["global_wape"]),
        "global_wape_accuracy": float(summary["global_wape_accuracy"]),
        "mae": float(scored["abs_error"].mean()),
        "signed_bias": float(residual.mean()),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Candidate J Shadow OOS Evaluation",
        "",
        f"- Generated at: `{payload['generated_at_utc']}`",
        f"- Design freeze cutoff: `{payload['design_freeze_cutoff']}`",
        f"- Complete shared revealed cycles: `{payload['cycles']}` ({payload['rows']} rows)",
        f"- Window: `{payload.get('first_data_cutoff')}` -> `{payload.get('last_data_cutoff')}`",
        f"- Verdict: **{payload['verdict']}**",
        f"- New severe-shock hours: `{payload['new_severe_shock_hours']}` / `{payload['min_new_severe_shock_hours']}`",
        f"- Recovery observed: `{payload['recovery_observed']}`",
        "",
        "## Metrics",
        "",
        f"- G mean_station_accuracy: `{payload['metrics']['G']['mean_station_accuracy']:.3f}`",
        f"- J mean_station_accuracy: `{payload['metrics']['J']['mean_station_accuracy']:.3f}`",
        f"- Delta J-G: `{payload['metrics']['delta_mean_station_accuracy_pp']:.3f}` pp",
        f"- G raw WAPE: `{payload['metrics']['G']['raw_wape']:.6f}`",
        f"- J raw WAPE: `{payload['metrics']['J']['raw_wape']:.6f}`",
        f"- G global_wape_accuracy: `{payload['metrics']['G']['global_wape_accuracy']:.3f}`",
        f"- J global_wape_accuracy: `{payload['metrics']['J']['global_wape_accuracy']:.3f}`",
        f"- G MAE: `{payload['metrics']['G']['mae']:.3f}`",
        f"- J MAE: `{payload['metrics']['J']['mae']:.3f}`",
        f"- G signed bias: `{payload['metrics']['G']['signed_bias']:.3f}`",
        f"- J signed bias: `{payload['metrics']['J']['signed_bias']:.3f}`",
        f"- Wins/losses/ties by cycle: `{payload['wins_losses_ties']}`",
        "",
        "## Rule",
        "",
        payload["decision_rule"],
    ]
    return "\n".join(lines)


def _empty_payload(g_model: pd.Series | dict[str, Any], j_model: pd.Series | dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "design_freeze_cutoff": cjs.OOS_FROM_CUTOFF,
        "g_model_id": str(g_model.get("model_id")),
        "j_model_id": str(j_model.get("model_id")),
        "j_is_active": bool(j_model.get("is_active", False)),
        "cycles": 0,
        "rows": 0,
        "reason": reason,
        "new_severe_shock_hours": 0,
        "min_new_severe_shock_hours": 12,
        "recovery_observed": False,
        "verdict": "insufficient_out_of_sample_evidence",
        "metrics": {},
        "wins_losses_ties": {"win": 0, "loss": 0, "tie": 0},
    }


def _one(frame: pd.DataFrame, label: str) -> pd.Series:
    if len(frame) != 1:
        raise RuntimeError(f"expected exactly one {label}, found {len(frame)}")
    return frame.iloc[0]


def _ts(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _load_env_file(path: Path = ROOT / ".env.local") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value.strip())


if __name__ == "__main__":
    main()
