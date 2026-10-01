"""Candidate K walk-forward experiment vs Candidates G and J (read-only, research only).

Protocol (fixed before any K result was looked at):
  * K is retrained from scratch every day at 00:00 Bogota (05:00Z) on samples whose targets are
    observed by the refit time, and predicts the next 24 hourly cycles -- K never sees a target it
    is scored on, and every input only uses observations <= the cycle data_cutoff;
  * each variant is a seed ensemble (median); the same seeds are used for every variant;
  * G and J are scored on exactly the same (cycle, station, target) rows: the J replay
    (`reports/candidate_j/tables/replay_predictions.csv`) for cycles before the J design freeze and
    the official production predictions afterwards;
  * metric = leaderboard `mean_station_accuracy`; per phase, per cycle, per station, per horizon.

Caveat recorded in the report: the K architecture/augmentation were chosen after inspecting the
shock up to the latest observation, so the post-freeze shock cycles are walk-forward (no training
leakage) but not design-blind for K. A clean test needs cycles after K's own design freeze.

Never writes to Supabase, never submits, never touches Predict or `model_versions`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import candidate_k_deep as ck  # noqa: E402
from drift_monitor import accuracy_summary  # noqa: E402


SNAPSHOT = ROOT / "data" / "interim" / "candidate_k" / "snapshot"
RUNS_DIR = ROOT / "data" / "interim" / "candidate_k" / "runs"
J_REPLAY = ROOT / "reports" / "candidate_j" / "tables" / "replay_predictions.csv"
REPORT_DIR = ROOT / "reports" / "candidate_k"
J_DESIGN_FREEZE = pd.Timestamp("2026-09-18T10:00:00Z")
# K v2.1 was frozen after inspecting observations up to this instant: only cycles whose data_cutoff
# is >= this timestamp are a design-blind test for K.
K_DESIGN_FREEZE = pd.Timestamp("2026-09-18T19:00:00Z")
FIRST_CUTOFF = pd.Timestamp("2026-09-04T23:00:00Z")
REFIT_HOUR_UTC = 5
TRAIN_STRIDE = 2  # train on one origin every 30 min (consecutive 15-min origins are near duplicates)
PHASES = ("stable", "phase1", "phase2", "phase3_shock", "oos_shock")
NORMAL_PHASES = ("stable", "phase1", "phase2")
SHOCK_PHASES = ("phase3_shock", "oos_shock")

VARIANTS: dict[str, ck.CandidateKConfig] = {
    "k_gated_aug": ck.CandidateKConfig(architecture="gated"),
    "k_tide_aug": ck.CandidateKConfig(architecture="tide"),
    "k_gated_noaug": ck.CandidateKConfig(architecture="gated", augmentation_prob=0.0),
    # v2 (designed after the v1 results): seasonal expert on real data + local expert pretrained
    # only on synthetic series (frozen) + gate trained on real data and synthetic regimes
    "k_staged": ck.CandidateKConfig(architecture="staged"),
    # v2.1 (after the v2 gate diagnosis): the gate also sees the seasonal expert's realized error
    "k_staged_re": ck.CandidateKConfig(architecture="staged_re"),
}
LOCAL_PRETRAINED = ROOT / "data" / "interim" / "candidate_k" / "local_pretrained.pt"
LOCAL_LOOKBACK = 32
LOCAL_PRETRAIN = {"lookback": LOCAL_LOOKBACK, "epochs": 24, "per_epoch": 200_000, "seed": 0}

# Primary variant; the others are reported as ablations / earlier iterations.
PRIMARY = "k_staged_re"  # v1 primary was k_gated_aug, v2 k_staged; all variants stay in the report

# Pre-registered acceptance vs the current champion (J), same spirit as J's DecisionCriteria.
CRITERIA = {
    "max_normal_phase_loss_pp": 0.5,
    "max_stable_station_loss_pp": 1.0,
    "min_shock_gain_pp": 5.0,
}


# --------------------------------------------------------------------------------------
# Walk-forward training (worker side)
# --------------------------------------------------------------------------------------

_STATE: dict[str, Any] = {}


def _init_worker(samples: ck.Samples, n_stations: int, local_path: str | None) -> None:
    import torch

    torch.set_num_threads(1)
    _STATE["samples"] = samples
    _STATE["n_stations"] = n_stations
    if local_path:
        local = ck.LocalForecaster(LOCAL_LOOKBACK, ck.CandidateKConfig().horizons, ck.CandidateKConfig())
        local.load_state_dict(torch.load(local_path))
        local.eval()
        _STATE["local"] = local


def _fit_and_predict(job: tuple[str, dict[str, Any], int, list[int], int]) -> dict[str, Any]:
    name, config_dict, refit_pos, eval_origins, seed = job
    config = ck.CandidateKConfig(**{**config_dict, "period_slots": tuple(config_dict["period_slots"]),
                                    "level_factor": tuple(config_dict["level_factor"]), "seeds": tuple(config_dict["seeds"])})
    samples: ck.Samples = _STATE["samples"]
    train_mask = (samples.origin + config.horizons <= refit_pos) & (samples.origin % TRAIN_STRIDE == 0)
    started = time.perf_counter()
    if config.architecture in ("staged", "staged_re"):
        realized = config.architecture == "staged_re"
        # lookup covers every origin up to the last evaluated cutoff; each entry is causal
        lookup = samples.subset(samples.origin <= max(eval_origins)) if realized else None
        model = ck.train_staged(samples.subset(train_mask), config, _STATE["n_stations"], seed, _STATE["local"], realized_error=realized, lookup_samples=lookup)
    else:
        model = ck.train_model(samples.subset(train_mask), config, _STATE["n_stations"], seed)
    evaluation = samples.subset(np.isin(samples.origin, eval_origins))
    output = ck.predict([model], evaluation)
    return {
        "variant": name,
        "seed": seed,
        "refit_pos": refit_pos,
        "origin": evaluation.origin,
        "station": evaluation.station,
        "prediction": output["prediction"],
        "gate": output.get("gate"),
        "train_rows": int(train_mask.sum()),
        "seconds": time.perf_counter() - started,
    }


def run_variant(name: str, config: ck.CandidateKConfig, panel: ck.Panel, samples: ck.Samples, cutoffs: pd.DatetimeIndex, *, seeds: int, workers: int) -> pd.DataFrame:
    positions = np.array([panel.position(c) for c in cutoffs])
    refits = sorted({panel.position(_refit_time(c)) for c in cutoffs})
    jobs = []
    for refit_pos in refits:
        next_refit = refit_pos + 24 * 4
        eval_origins = positions[(positions >= refit_pos) & (positions < next_refit)].tolist()
        for seed in range(seeds):
            jobs.append((name, config.to_dict(), refit_pos, eval_origins, seed))
    rows = []
    started = time.perf_counter()
    local_path = str(ensure_local_pretrained()) if config.architecture.startswith("staged") else None
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(samples, len(panel.stations), local_path)) as pool:
        for done, result in enumerate(pool.map(_fit_and_predict, jobs), start=1):
            for h in range(config.horizons):
                rows.append(pd.DataFrame({
                    "variant": result["variant"],
                    "seed": result["seed"],
                    "refit_at": panel.index[result["refit_pos"]],
                    "train_rows": result["train_rows"],
                    "station_id": [panel.stations[s] for s in result["station"]],
                    "data_cutoff": panel.index[result["origin"]],
                    "target_at": panel.index[result["origin"] + h + 1],
                    "horizon_minutes": 15 * (h + 1),
                    "prediction": result["prediction"][:, h],
                    "seasonal_weight": result["gate"][:, h] if result["gate"] is not None else np.nan,
                }))
            print(f"[{name}] {done}/{len(jobs)} refit={panel.index[result['refit_pos']]:%m-%d} seed={result['seed']} "
                  f"train_rows={result['train_rows']} {result['seconds']:.0f}s elapsed={time.perf_counter() - started:.0f}s", flush=True)
    return pd.concat(rows, ignore_index=True)


def ensure_local_pretrained() -> Path:
    """Pretrain the local expert once on synthetic series only (no real data, so no leakage)."""
    import torch

    meta_path = LOCAL_PRETRAINED.with_suffix(".json")
    if LOCAL_PRETRAINED.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get("pretrain") == LOCAL_PRETRAIN:
        return LOCAL_PRETRAINED
    torch.set_num_threads(2)
    started = time.perf_counter()
    model = ck.pretrain_local_forecaster(ck.CandidateKConfig(), **LOCAL_PRETRAIN, log=print)
    LOCAL_PRETRAINED.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LOCAL_PRETRAINED)
    meta_path.write_text(json.dumps({
        "pretrain": LOCAL_PRETRAIN,
        "data": "synthetic_series only (candidate_k_deep.synthetic_series); no observation is used",
        "sha256": hashlib.sha256(LOCAL_PRETRAINED.read_bytes()).hexdigest(),
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2) + "\n")
    return LOCAL_PRETRAINED


def add_staged_references(frame: pd.DataFrame, runs: pd.DataFrame, panel: ck.Panel, staged: list[str]) -> pd.DataFrame:
    """References derived from the staged runs, scored on the same rows.

    * k_local_zero_shot: the synthetic-pretrained local expert alone (never saw real data);
    * <variant>_j: K's per-seed gate applied to J instead of K's own seasonal expert,
      g * J + (1 - g) * local, median over seeds (post-hoc hybrid: J is computed every cycle in
      production, so this is deployable as a wrapper around the champion).
    """
    import torch

    config = ck.CandidateKConfig()
    local = ck.LocalForecaster(LOCAL_LOOKBACK, config.horizons, config)
    local.load_state_dict(torch.load(LOCAL_PRETRAINED))
    local.eval()
    cutoffs = sorted(frame["data_cutoff"].unique())
    samples = ck.build_samples(panel, config, np.array([panel.position(c) for c in cutoffs]))
    with torch.no_grad():
        values = local(torch.from_numpy(samples.past)).numpy() * samples.scale[:, None]
    local_rows = pd.concat([
        pd.DataFrame({
            "station_id": [panel.stations[k] for k in samples.station],
            "data_cutoff": panel.index[samples.origin],
            "horizon_minutes": 15 * (h + 1),
            "k_local_zero_shot_prediction": values[:, h],
        })
        for h in range(config.horizons)
    ])
    keys = ["station_id", "data_cutoff", "horizon_minutes"]
    frame = frame.merge(local_rows, on=keys, how="left")
    for name in staged:
        per_seed = runs.loc[runs["variant"] == name, [*keys, "seed", "seasonal_weight"]].merge(frame[[*keys, "j_prediction", "k_local_zero_shot_prediction"]], on=keys)
        per_seed["hybrid"] = per_seed["seasonal_weight"] * per_seed["j_prediction"] + (1 - per_seed["seasonal_weight"]) * per_seed["k_local_zero_shot_prediction"]
        hybrid = per_seed.groupby(keys)["hybrid"].median().rename(f"{name}_j_prediction").reset_index()
        frame = frame.merge(hybrid, on=keys, how="left")
    return frame


def _refit_time(cutoff: pd.Timestamp) -> pd.Timestamp:
    day = cutoff.floor("D") + pd.Timedelta(hours=REFIT_HOUR_UTC)
    return day if day <= cutoff else day - pd.Timedelta(days=1)


# --------------------------------------------------------------------------------------
# Comparison frame and metrics
# --------------------------------------------------------------------------------------


def comparison_frame(panel: ck.Panel, last_cutoff: pd.Timestamp) -> pd.DataFrame:
    """G and J on the evaluated targets: J replay before the J freeze, official predictions after."""
    replay = pd.read_csv(J_REPLAY, dtype={"station_id": str}, parse_dates=["target_at", "data_cutoff"])
    replay = replay[replay["data_cutoff"] < J_DESIGN_FREEZE]
    replay = replay[["station_id", "data_cutoff", "target_at", "horizon_minutes", "phase", "regime", "path", "g_prediction", "j_prediction"]]

    official = pd.read_csv(SNAPSHOT / "official_predictions.csv", dtype={"station_id": str})
    cycles = pd.read_csv(SNAPSHOT / "forecast_cycles.csv")
    official = official.merge(cycles[["cycle_id", "data_cutoff"]], on="cycle_id")
    official["data_cutoff"] = pd.to_datetime(official["data_cutoff"], utc=True)
    official["target_at"] = pd.to_datetime(official["target_observed_at"], utc=True)
    official = official[(official["data_cutoff"] >= J_DESIGN_FREEZE) & (official["data_cutoff"] <= last_cutoff)]
    wide = official.pivot_table(index=["station_id", "data_cutoff", "target_at", "horizon_minutes"], columns="model", values="prediction").reset_index()
    wide = wide.rename(columns={"G": "g_prediction", "J": "j_prediction"}).assign(phase="oos_shock", regime="", path="")
    frame = pd.concat([replay, wide[replay.columns]], ignore_index=True)
    frame["actual"] = [panel.demand[panel.position(t), panel.stations.index(s)] for s, t in zip(frame["station_id"], frame["target_at"])]
    return frame[frame["actual"].notna()].reset_index(drop=True)


def ensemble(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "station_id", "data_cutoff", "target_at", "horizon_minutes"]
    out = runs.groupby(keys).agg(
        prediction=("prediction", "median"),
        member_spread=("prediction", lambda v: float(v.max() - v.min())),
        seasonal_weight=("seasonal_weight", "mean"),
        members=("seed", "nunique"),
    ).reset_index()
    return out


def metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    scored = frame.assign(prediction=frame[column], abs_error=(frame[column] - frame["actual"]).abs())
    summary = accuracy_summary(scored)
    residual = scored["actual"] - scored["prediction"]
    return {
        "rows": summary["rows"],
        "mean_station_accuracy": summary["mean_station_accuracy"],
        "global_wape_accuracy": summary["global_wape_accuracy"],
        "global_wape": summary["global_wape"],
        "accuracy_at_20": summary["accuracy_at_20"],
        "mae": float(scored["abs_error"].mean()),
        "signed_bias": float(residual.mean()),
        "relative_bias": float(residual.sum() / scored["actual"].sum()),
    }


def evaluate(frame: pd.DataFrame, models: dict[str, str], champion: str = "J") -> dict[str, pd.DataFrame]:
    overall, cycles, stations, horizons = [], [], [], []
    for phase in [p for p in PHASES if p in set(frame["phase"])]:
        part = frame[frame["phase"] == phase]
        for model, column in models.items():
            overall.append({"phase": phase, "model": model, "cycles": int(part["data_cutoff"].nunique()), **metrics(part, column)})
        for cutoff, cycle in part.groupby("data_cutoff"):
            row = {"phase": phase, "data_cutoff": cutoff}
            for model, column in models.items():
                row[model] = metrics(cycle, column)["mean_station_accuracy"]
            cycles.append(row)
        for station_id, group in part.groupby("station_id"):
            row = {"phase": phase, "station_id": station_id}
            for model, column in models.items():
                row[model] = metrics(group, column)["mean_station_accuracy"]
            stations.append(row)
        for horizon, group in part.groupby("horizon_minutes"):
            row = {"phase": phase, "horizon_minutes": horizon}
            for model, column in models.items():
                row[model] = metrics(group, column)["mean_station_accuracy"]
            horizons.append(row)
    cycles = pd.DataFrame(cycles)
    wins = []
    for model in models:
        if model in ("G", champion):
            continue
        for reference in ("G", champion):
            delta = cycles[model] - cycles[reference]
            for phase, d in delta.groupby(cycles["phase"]):
                wins.append({"model": model, "vs": reference, "phase": phase, "win": int((d > 1e-9).sum()),
                             "loss": int((d < -1e-9).sum()), "tie": int((d.abs() <= 1e-9).sum())})
    return {
        "overall": pd.DataFrame(overall),
        "cycles": cycles,
        "stations": pd.DataFrame(stations),
        "horizons": pd.DataFrame(horizons),
        "wins": pd.DataFrame(wins),
    }


def pooled(frame: pd.DataFrame, models: dict[str, str]) -> pd.DataFrame:
    """Leaderboard metric over pooled windows (missing cycles do not exist here: all models cover all)."""
    cutoff = pd.to_datetime(frame["data_cutoff"], utc=True)
    windows = {
        "all_cycles": frame,
        "normal_phases": frame[frame["phase"].isin(NORMAL_PHASES)],
        "last_7d": frame[cutoff >= cutoff.max() - pd.Timedelta(days=7)],
        "last_24h": frame[cutoff >= cutoff.max() - pd.Timedelta(hours=24)],
    }
    rows = []
    for window, part in windows.items():
        for model, column in models.items():
            rows.append({"window": window, "model": model, "cycles": int(part["data_cutoff"].nunique()), **metrics(part, column)})
    return pd.DataFrame(rows)


def decide(results: dict[str, pd.DataFrame], model: str, champion: str = "J") -> dict[str, Any]:
    overall = results["overall"].set_index(["phase", "model"])["mean_station_accuracy"]
    phases = sorted(set(results["overall"]["phase"]), key=PHASES.index)
    deltas = {phase: float(overall[(phase, model)] - overall[(phase, champion)]) for phase in phases}
    stations = results["stations"]
    normal_stations = stations[stations["phase"].isin(NORMAL_PHASES)]
    worst_station = float((normal_stations[model] - normal_stations[champion]).min())
    checks = {
        "normal_phases_within_tolerance": all(deltas[p] >= -CRITERIA["max_normal_phase_loss_pp"] for p in NORMAL_PHASES if p in deltas),
        "stations_within_tolerance": worst_station >= -CRITERIA["max_stable_station_loss_pp"],
        "shock_gain": all(deltas[p] >= CRITERIA["min_shock_gain_pp"] for p in SHOCK_PHASES if p in deltas),
    }
    verdict = "shadow_candidate" if all(checks.values()) else "reject"
    return {
        "model": model,
        "vs": champion,
        "delta_pp_by_phase": deltas,
        "worst_station_delta_pp_normal": worst_station,
        "checks": checks,
        "verdict": verdict,
        # the post-freeze shock is not design-blind for K, so the best possible verdict is shadow
        "verdict_cap": "shadow_candidate (design informed by the shock; needs post-freeze cycles before any promotion)",
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    started = time.perf_counter()

    observations = pd.read_csv(SNAPSHOT / "observations.csv", dtype={"station_id": str})
    base_config = ck.CandidateKConfig()
    panel = ck.build_panel(observations, base_config)
    samples = ck.build_samples(panel, base_config)
    last_observed = pd.to_datetime(observations["observed_at"], utc=True).max()
    last_cutoff = (last_observed - pd.Timedelta(minutes=15 * base_config.horizons)).floor("h")
    cutoffs = pd.date_range(FIRST_CUTOFF, last_cutoff, freq="h")
    runs_dir = args.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    names = [name for name in args.variants.split(",") if name]
    if not args.report_only:
        for name in names:
            config = VARIANTS[name] if args.epochs is None else replace(VARIANTS[name], epochs=args.epochs)
            path = runs_dir / f"{name}.csv"
            if path.exists():
                print(f"[{name}] cached at {path}; delete it to retrain")
                continue
            runs = run_variant(name, config, panel, samples, cutoffs, seeds=args.seeds, workers=args.workers)
            runs.to_csv(path, index=False)

    available = [name for name in names if (runs_dir / f"{name}.csv").exists()]
    runs = pd.concat([pd.read_csv(runs_dir / f"{name}.csv", dtype={"station_id": str}, parse_dates=["data_cutoff", "target_at", "refit_at"]) for name in available])
    members = ensemble(runs)
    frame = comparison_frame(panel, last_cutoff)
    models = {"G": "g_prediction", "J": "j_prediction"}
    for name in available:
        part = members[members["variant"] == name].drop(columns="variant").rename(columns={
            "prediction": f"{name}_prediction", "member_spread": f"{name}_spread", "seasonal_weight": f"{name}_seasonal_weight", "members": f"{name}_members"})
        frame = frame.merge(part, on=["station_id", "data_cutoff", "target_at", "horizon_minutes"], how="left")
        models[name] = f"{name}_prediction"
    derived: list[str] = []
    staged = [name for name in available if VARIANTS[name].architecture.startswith("staged")]
    if staged and LOCAL_PRETRAINED.exists():
        frame = add_staged_references(frame, runs, panel, staged)
        derived = ["k_local_zero_shot", *[f"{name}_j" for name in staged]]
        for name in derived:
            models[name] = f"{name}_prediction"
    missing = {name: int(frame[col].isna().sum()) for name, col in models.items()}
    if any(missing.values()):
        raise RuntimeError(f"models are not scored on the same targets: {missing}")

    results = evaluate(frame, models)
    results["pooled"] = pooled(frame, models)
    decisions = {name: decide(results, name) for name in [*available, *derived]}
    out = args.output_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for key, table in results.items():
        table.to_csv(out / "tables" / f"{key}.csv", index=False)
    frame.to_csv(out / "tables" / "predictions.csv", index=False)
    summary = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "snapshot_manifest": json.loads((SNAPSHOT / "manifest.json").read_text()),
        "protocol": {
            "refit": f"daily at {REFIT_HOUR_UTC:02d}:00Z from scratch, targets observed by the refit time",
            "train_stride_slots": TRAIN_STRIDE,
            "seeds": int(runs["seed"].nunique()),
            "first_cutoff": FIRST_CUTOFF.isoformat(),
            "last_cutoff": last_cutoff.isoformat(),
            "j_design_freeze": J_DESIGN_FREEZE.isoformat(),
            "k_design_freeze": K_DESIGN_FREEZE.isoformat(),
            "k_iterations": [
                "v1: k_gated_aug (primary), k_tide_aug, k_gated_noaug (not run to completion)",
                "v2: k_staged, designed after v1 results",
                "v2.1: k_staged_re, gate sees the seasonal expert's realized error; designed after the v2 gate diagnosis",
            ],
            "comparison_source": {"before_j_freeze": "candidate J replay", "after_j_freeze": "official production predictions"},
        },
        "variants": {name: VARIANTS[name].to_dict() for name in available},
        "primary_variant": PRIMARY if PRIMARY in available else available[0],
        "criteria": CRITERIA,
        "overall": results["overall"].to_dict(orient="records"),
        "pooled": results["pooled"].to_dict(orient="records"),
        "local_pretrained": json.loads(LOCAL_PRETRAINED.with_suffix(".json").read_text()) if LOCAL_PRETRAINED.with_suffix(".json").exists() else None,
        "code_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in ("src/candidate_k_deep.py", "scripts/candidate_k_experiment.py")},
        "decisions": decisions,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    table = results["overall"].pivot(index="model", columns="phase", values="mean_station_accuracy")
    print(table[[p for p in PHASES if p in table.columns]].round(2).to_string())
    print(json.dumps({name: {"verdict": d["verdict"], "delta": {k: round(v, 2) for k, v in d["delta_pp_by_phase"].items()}} for name, d in decisions.items()}, indent=1))


if __name__ == "__main__":
    main()
