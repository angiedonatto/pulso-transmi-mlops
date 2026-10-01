"""Retrospective shadow replay: Candidate J vs Candidate G on the same targets (read-only).

Hourly cycles from a stable week before the drift to the latest resolved cutoff:
  1. Candidate F base refit per cutoff with observations <= data_cutoff (cached, parallel);
  2. Candidate G replayed with the production G2 code (`candidate_g_adaptive.g2_factor`),
     feedback = resolved targets <= data_cutoff;
  3. Candidate J composed per cycle with `candidate_j_regime` (thresholds calibrated on a stable
     window that ends before the first evaluated cycle);
  4. both scored on the same targets, per phase of `reports/drift_regime_analysis/summary.json`.

Writes `reports/candidate_j/`. Never writes to Supabase, never submits, never touches Predict or
`model_versions`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import candidate_j_regime as cj  # noqa: E402
import drift_regime as dr  # noqa: E402
from drift_monitor import accuracy_summary  # noqa: E402


REPORT_DIR = ROOT / "reports" / "candidate_j"
INTERIM_DIR = ROOT / "data" / "interim" / "candidate_j"
DEFAULT_SNAPSHOT = ROOT / "data" / "interim" / "drift_regime_analysis" / "snapshot_live"
DRIFT_SUMMARY = ROOT / "reports" / "drift_regime_analysis" / "summary.json"
DRIFT_MATRIX = ROOT / "reports" / "drift_regime_analysis" / "station_diagnostics.csv"
G_OFFICIAL_BASE_CACHE = ROOT / "data" / "interim" / "drift_regime_analysis" / "g_base_reconstruction.csv"
HORIZONS = (15, 30, 45, 60)
STRING_COLUMNS = {"station_id": str, "cycle_id": str, "model_id": str}
MODELS = {"G": "g_prediction", "J": "j_prediction", "base_F": "base_prediction"}
PRIMARY = "candidate_j_v1_1"


# --------------------------------------------------------------------------------------
# Phases and cutoffs
# --------------------------------------------------------------------------------------


def phase_windows(summary: dict[str, Any], anchor: pd.Timestamp, *, stable_days: int) -> list[dict[str, Any]]:
    """Phases by cycle cutoff, from the drift analysis global-intensity segments."""
    starts = [pd.Timestamp(row["start_local"]).tz_convert("UTC") for row in summary["change_points"]["global_intensity_segments"]]
    if len(starts) < 4:
        raise RuntimeError("expected a stable segment plus three drift regimes in the drift analysis summary")
    r1, r2, r3 = starts[1], starts[2], starts[3]
    return [
        {"phase": "stable", "start": r1 - pd.Timedelta(days=stable_days), "end": r1},
        {"phase": "phase1", "start": r1, "end": r2},
        {"phase": "phase2", "start": r2, "end": r3},
        {"phase": "phase3_shock", "start": r3, "end": anchor + pd.Timedelta(hours=1)},
    ]


def phase_of(cutoff: pd.Timestamp, phases: list[dict[str, Any]]) -> str | None:
    for phase in phases:
        if phase["start"] <= cutoff < phase["end"]:
            return phase["phase"]
    return None


# --------------------------------------------------------------------------------------
# Candidate F base per cutoff (cached, parallel)
# --------------------------------------------------------------------------------------

_WORKER_OBS: pd.DataFrame | None = None


def _init_worker(observations: pd.DataFrame) -> None:
    global _WORKER_OBS
    _WORKER_OBS = observations


def _base_for_cutoff(args: tuple[str, list[str]]) -> pd.DataFrame:
    from candidate_g_adaptive import build_candidate_f_base_frame
    from forecast_cycle_sync import CycleTarget, ForecastCycle

    cutoff_iso, stations = args
    cutoff = pd.Timestamp(cutoff_iso)
    targets = [
        CycleTarget(station_id=s, target_at=_iso(cutoff + pd.Timedelta(minutes=h)), horizon_minutes=h, target_order=i * len(HORIZONS) + j, raw_target={})
        for i, s in enumerate(stations)
        for j, h in enumerate(HORIZONS)
    ]
    cycle = ForecastCycle(cycle_id=f"replay_{cutoff:%Y%m%dT%H%M%SZ}", data_cutoff=cutoff_iso, closes_at="", expected_predictions=len(targets), targets=targets, raw_cycle={})
    eligible = _WORKER_OBS[_WORKER_OBS["observed_at"] <= cutoff]  # information available at prediction time
    base = build_candidate_f_base_frame(cycle, eligible)
    return pd.DataFrame(
        {
            "data_cutoff": cutoff_iso,
            "station_id": base["station_id"].astype(str),
            "target_at": base["target_at"].map(_iso),
            "horizon_minutes": base["horizon_minutes"].astype(int),
            "base_prediction": base["base_prediction"].astype(float),
            "max_observed_at_used": _iso(eligible["observed_at"].max()),
        }
    )


def compute_bases(cutoffs: list[pd.Timestamp], observations: pd.DataFrame, stations: list[str], *, workers: int, cache_path: Path) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    if cache_path.exists():
        frames.append(pd.read_csv(cache_path, dtype=STRING_COLUMNS))
    if G_OFFICIAL_BASE_CACHE.exists():  # bases already refit for the official G cycles (same cutoff -> same base)
        official = pd.read_csv(G_OFFICIAL_BASE_CACHE, dtype=STRING_COLUMNS).drop(columns=["cycle_id"])
        frames.append(official)
    cache = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["data_cutoff"])
    if not cache.empty:
        cache["data_cutoff"] = pd.to_datetime(cache["data_cutoff"], utc=True).map(_iso)
        cache = cache.drop_duplicates(["data_cutoff", "station_id", "target_at"])
    have = set(cache["data_cutoff"]) if not cache.empty else set()
    todo = [_iso(c) for c in cutoffs if _iso(c) not in have]
    if todo:
        print(f"[candidate_j] refitting Candidate F base for {len(todo)} cutoffs with {workers} workers", file=sys.stderr, flush=True)
        started = time.perf_counter()
        new_frames = []
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(observations,)) as pool:
            for i, frame in enumerate(pool.map(_base_for_cutoff, [(c, stations) for c in todo]), start=1):
                new_frames.append(frame)
                if i % 10 == 0 or i == len(todo):
                    pd.concat([cache] + new_frames, ignore_index=True).to_csv(cache_path, index=False)
                    print(f"[candidate_j] {i}/{len(todo)} ({time.perf_counter() - started:.0f}s)", file=sys.stderr, flush=True)
        cache = pd.concat([cache] + new_frames, ignore_index=True)
    cache["data_cutoff"] = pd.to_datetime(cache["data_cutoff"], utc=True)
    cache["target_at"] = pd.to_datetime(cache["target_at"], utc=True)
    cache["max_observed_at_used"] = pd.to_datetime(cache["max_observed_at_used"], utc=True)
    wanted = set(cutoffs)
    cache = cache[cache["data_cutoff"].isin(wanted)].copy()
    if (cache["max_observed_at_used"] > cache["data_cutoff"]).any():
        raise RuntimeError("a base was refit with observations after its data_cutoff")
    return cache.sort_values(["data_cutoff", "station_id", "target_at"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Replays
# --------------------------------------------------------------------------------------


def replay_g(bases: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Candidate G with the production G2 code; feedback = resolved targets <= each cutoff."""
    from candidate_g_adaptive import g2_factor, prepare_feedback_frame

    frame = bases.copy()
    frame["actual"] = [_lookup(wide, t, s) for s, t in zip(frame["station_id"], frame["target_at"])]
    out = []
    for cutoff, cycle in frame.groupby("data_cutoff", sort=True):
        feedback = frame[frame["target_at"] <= cutoff][["station_id", "target_at", "base_prediction", "actual"]].dropna()
        feedback = prepare_feedback_frame(feedback.tail(48 * 4), cutoff)  # only the last rows per station matter
        factors = {s: g2_factor(s, feedback) for s in cycle["station_id"].unique()}
        cycle = cycle.copy()
        cycle["g2_factor"] = cycle["station_id"].map(lambda s: factors[s][0])
        cycle["feedback_samples"] = cycle["station_id"].map(lambda s: factors[s][1])
        cycle["g_prediction"] = np.maximum(0.0, cycle["base_prediction"] * cycle["g2_factor"])
        out.append(cycle)
    return pd.concat(out, ignore_index=True)


def replay_j(frame: pd.DataFrame, wide: pd.DataFrame, anomaly: pd.DataFrame, thresholds: cj.RegimeThresholds, config: cj.CandidateJConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, regime_rows = [], []
    shares: dict[pd.Timestamp, float] = {}
    for cutoff, cycle in frame.groupby("data_cutoff", sort=True):
        visible = wide.loc[wide.index <= cutoff]
        signals = cj.signals_at(cutoff, visible, anomaly.loc[anomaly.index <= cutoff])
        previous = [shares.get(cutoff - pd.Timedelta(hours=k), 0.0) for k in range(1, config.shock_memory_hours)]
        regimes = cj.classify_regimes(signals, thresholds, config, previous_shares=previous)
        shares[cutoff] = float(regimes["shock_share"].iloc[0])
        history = frame[frame["target_at"] <= cutoff][["station_id", "target_at", "actual", "base_prediction", "g_prediction", "rp_prediction"]]
        usefulness = cj.g2_usefulness(history, cutoff, config)
        composed = cj.compose_predictions(
            cycle[["station_id", "target_at", "horizon_minutes", "base_prediction", "g2_factor"]], regimes, usefulness, thresholds, wide=visible, cutoff=cutoff, config=config
        )
        merged = cycle.merge(composed[["station_id", "target_at", "regime", "path", "j_prediction", "guard_fired", "base_reaction", "g2_useful", "rp_useful"]], on=["station_id", "target_at"], how="left")
        out.append(merged)
        regime_rows.append(regimes.assign(data_cutoff=cutoff).merge(usefulness, on="station_id", how="left"))
    return pd.concat(out, ignore_index=True), pd.concat(regime_rows, ignore_index=True)


def add_recent_profile_counterfactual(frame: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """What the recent-profile path would have predicted for every target (data <= each cutoff); J v1.1
    needs it to check, on resolved targets only, whether that path has been beating G."""
    parts = []
    for cutoff, cycle in frame.groupby("data_cutoff", sort=True):
        parts.append(cycle.assign(rp_prediction=cj.recent_profile_predictions(cycle, wide, cutoff)))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def path_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    """J vs G restricted to the rows each J path produced (where the gains and losses come from)."""
    rows = []
    for (phase, path), group in frame.groupby(["phase", "path"]):
        g, j = metrics(group, "g_prediction"), metrics(group, "j_prediction")
        saved = float(((group["g_prediction"] - group["actual"]).abs() - (group["j_prediction"] - group["actual"]).abs()).sum())
        rows.append({"phase": phase, "path": path, "rows": int(len(group)), "g_global_wape_accuracy": g["global_wape_accuracy"],
                     "j_global_wape_accuracy": j["global_wape_accuracy"], "abs_error_saved": saved})
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    scored = frame.assign(prediction=frame[column], abs_error=(frame[column] - frame["actual"]).abs())
    summary = accuracy_summary(scored)
    residual = scored["actual"] - scored["prediction"]
    return {
        "rows": summary["rows"],
        "global_wape": summary["global_wape"],
        "global_wape_accuracy": summary["global_wape_accuracy"],
        "mean_station_accuracy": summary["mean_station_accuracy"],
        "accuracy_at_20": summary["accuracy_at_20"],
        "mae": float(scored["abs_error"].mean()),
        "signed_bias": float(residual.mean()),
        "relative_bias": float(residual.sum() / scored["actual"].sum()),
    }


def evaluate(frame: pd.DataFrame, stable_stations: list[str]) -> dict[str, pd.DataFrame]:
    phases = [p for p in ["stable", "phase1", "phase2", "phase3_shock"] if p in set(frame["phase"])]
    overall, cycles, stations, horizons, worst = [], [], [], [], []
    for phase in phases:
        part = frame[frame["phase"] == phase]
        for model, column in MODELS.items():
            overall.append({"phase": phase, "model": model, "cycles": int(part["data_cutoff"].nunique()), **metrics(part, column)})
        for cutoff, cycle in part.groupby("data_cutoff"):
            g, j = metrics(cycle, "g_prediction"), metrics(cycle, "j_prediction")
            cycles.append({
                "phase": phase, "data_cutoff": cutoff,
                "g_mean_station_accuracy": g["mean_station_accuracy"], "j_mean_station_accuracy": j["mean_station_accuracy"],
                "g_global_wape_accuracy": g["global_wape_accuracy"], "j_global_wape_accuracy": j["global_wape_accuracy"],
                "delta_pp": j["mean_station_accuracy"] - g["mean_station_accuracy"],
                "paths": ";".join(sorted(cycle["path"].unique())), "regimes": ";".join(sorted(cycle["regime"].unique())),
            })
        for station_id, group in part.groupby("station_id"):
            g, j = metrics(group, "g_prediction"), metrics(group, "j_prediction")
            stations.append({
                "phase": phase, "station_id": station_id, "stable_station": station_id in stable_stations,
                "g_accuracy": g["mean_station_accuracy"], "j_accuracy": j["mean_station_accuracy"],
                "delta_pp": j["mean_station_accuracy"] - g["mean_station_accuracy"],
                "g_relative_bias": g["relative_bias"], "j_relative_bias": j["relative_bias"],
                "share_rows_changed": float(((group["path"] != cj.PATH_G) | group["guard_fired"].astype(bool)).mean()),
            })
        for horizon, group in part.groupby("horizon_minutes"):
            g, j = metrics(group, "g_prediction"), metrics(group, "j_prediction")
            horizons.append({"phase": phase, "horizon_minutes": horizon, "g_accuracy": g["mean_station_accuracy"], "j_accuracy": j["mean_station_accuracy"],
                             "delta_pp": j["mean_station_accuracy"] - g["mean_station_accuracy"], "g_relative_bias": g["relative_bias"], "j_relative_bias": j["relative_bias"]})
        st = pd.DataFrame([s for s in stations if s["phase"] == phase])
        worst.append({"phase": phase, "g_worst_station": st.loc[st["g_accuracy"].idxmin(), "station_id"], "g_worst_accuracy": st["g_accuracy"].min(),
                      "j_worst_station": st.loc[st["j_accuracy"].idxmin(), "station_id"], "j_worst_accuracy": st["j_accuracy"].min()})
    cycles = pd.DataFrame(cycles)
    wins = cycles.assign(result=np.select([cycles["delta_pp"] > 1e-9, cycles["delta_pp"] < -1e-9], ["win", "loss"], "tie")).groupby(["phase", "result"]).size().unstack(fill_value=0)
    paths = frame.groupby(["phase", "regime", "path"]).size().rename("rows").reset_index()
    paths["share"] = paths["rows"] / paths.groupby("phase")["rows"].transform("sum")
    return {
        "overall": pd.DataFrame(overall),
        "cycles": cycles,
        "stations": pd.DataFrame(stations),
        "horizons": pd.DataFrame(horizons),
        "worst": pd.DataFrame(worst),
        "wins": wins.reindex(columns=["win", "loss", "tie"], fill_value=0).reset_index(),
        "paths": paths,
    }


def official_parity(frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    scored = dr.resolve_predictions(**{k: tables[k] for k in ["predictions", "forecast_cycles", "forecast_cycle_targets", "observations", "submission_receipts", "model_versions"]})
    g = scored[(scored["model_name"] == "candidate_g_f4_profile_ratio_g2_ewma") & scored["is_official"]]
    joined = frame.merge(g[["data_cutoff", "station_id", "target_at", "prediction"]], on=["data_cutoff", "station_id", "target_at"], how="inner")
    if joined.empty:
        return {"rows": 0}
    diff = (joined["g_prediction"] - joined["prediction"]).abs()
    official_cycles = joined.groupby("data_cutoff")
    per_cycle = official_cycles.apply(lambda c: pd.Series({
        "official_g": accuracy_summary(c.assign(abs_error=(c["prediction"] - c["actual"]).abs()))["mean_station_accuracy"],
        "replay_g": accuracy_summary(c.assign(abs_error=(c["g_prediction"] - c["actual"]).abs()))["mean_station_accuracy"],
        "j": accuracy_summary(c.assign(abs_error=(c["j_prediction"] - c["actual"]).abs()))["mean_station_accuracy"],
    }), include_groups=False)
    return {
        "rows": int(len(joined)),
        "cycles": int(joined["data_cutoff"].nunique()),
        "share_rows_identical_1e-6": float((diff <= 1e-6 * np.maximum(1.0, joined["prediction"].abs())).mean()),
        "median_abs_diff": float(diff.median()),
        "max_abs_diff": float(diff.max()),
        "per_cycle": per_cycle.reset_index(),
        "official_rows": joined,
    }


def recovery(frame: pd.DataFrame, regimes: pd.DataFrame, phases: list[dict[str, Any]]) -> dict[str, Any]:
    shock = next(p for p in phases if p["phase"] == "phase3_shock")
    flagged = regimes[regimes["regime"] == cj.SEVERE_SHOCK].groupby("data_cutoff").size().index
    in_shock = [c for c in flagged if c >= shock["start"]]
    false_alarms = [c for c in flagged if c < shock["start"]]
    last_cutoff = frame["data_cutoff"].max()
    shock_cutoffs = sorted(frame.loc[frame["phase"] == "phase3_shock", "data_cutoff"].unique())
    return {
        "shock_start": _iso(shock["start"]),
        "shock_hours_observed": float((last_cutoff + pd.Timedelta(hours=1) - shock["start"]) / pd.Timedelta(hours=1)),
        "first_shock_detection": _iso(in_shock[0]) if in_shock else None,
        "detection_latency_hours": float((in_shock[0] - shock["start"]) / pd.Timedelta(hours=1)) if in_shock else None,
        "shock_cycles": len(shock_cutoffs),
        "shock_cycles_flagged": len(in_shock),
        "false_shock_alarms_before_phase3": [_iso(c) for c in false_alarms],
        "shock_still_active_at_last_cutoff": bool(last_cutoff in set(flagged)),
        "recovery_observed": False,  # the drift analysis shows regime_3 still active at the last observation
    }


def pooled_threshold_sensitivity(wide: pd.DataFrame, anomaly: pd.DataFrame, thresholds: cj.RegimeThresholds, regimes: pd.DataFrame,
                                  phases: list[dict[str, Any]], config: cj.CandidateJConfig) -> dict[str, Any]:
    """Shock shares under the first (pooled, all-station max) threshold design vs the per-station one."""
    pooled = cj.RegimeThresholds.uniform(thresholds.per_station, **{k: max(v[k] for v in thresholds.per_station.values()) for k in cj.THRESHOLD_KEYS})
    rows = []
    for cutoff in sorted(regimes["data_cutoff"].unique()):
        phase = phase_of(cutoff, phases)
        if phase is None:
            continue
        signals = cj.signals_at(cutoff, wide.loc[wide.index <= cutoff], anomaly.loc[anomaly.index <= cutoff])
        rows.append({"data_cutoff": cutoff, "phase": phase, "pooled_share": cj.shock_share(signals, pooled), "per_station_share": cj.shock_share(signals, thresholds)})
    frame = pd.DataFrame(rows)
    shock = frame[frame["phase"] == "phase3_shock"]
    return {
        "shock_cutoffs_local": [str(pd.Timestamp(c).tz_convert(dr.BOGOTA_TZ))[:16] for c in shock["data_cutoff"]],
        "pooled_share_in_shock": [round(v, 3) for v in shock["pooled_share"]],
        "per_station_share_in_shock": [round(v, 3) for v in shock["per_station_share"]],
        "max_pooled_share_before_shock": float(frame.loc[frame["phase"] != "phase3_shock", "pooled_share"].max()),
        "max_per_station_share_before_shock": float(frame.loc[frame["phase"] != "phase3_shock", "per_station_share"].max()),
        "global_shock_share": config.global_shock_share,
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--drift-summary", type=Path, default=DRIFT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--stable-days", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()

    tables = _load_snapshot(args.from_snapshot)
    summary = json.loads(args.drift_summary.read_text())
    prepared = dr.prepare_observations(tables["observations"])
    wide = dr.demand_matrix(prepared)
    baseline = dr.seasonal_baseline(wide, weeks=4)
    anomaly = cj.hourly_anomaly(wide, baseline)
    stations = sorted(wide.columns)
    anchor = prepared["observed_at"].max()
    last_cutoff = (anchor - pd.Timedelta(minutes=max(HORIZONS))).floor("h")
    phases = phase_windows(summary, last_cutoff, stable_days=args.stable_days)
    warmup = pd.Timedelta(hours=cj.CandidateJConfig().g2_usefulness_window_hours)
    cutoffs = list(pd.date_range(phases[0]["start"] - warmup, last_cutoff, freq="h"))

    calibration_start = anomaly.dropna(how="all").index.min() + pd.Timedelta(hours=48)
    calibration_end = phases[0]["start"] - pd.Timedelta(hours=1)
    thresholds = cj.calibrate_thresholds(wide, anomaly, calibration_start, calibration_end)

    bases = compute_bases(cutoffs, prepared[["station_id", "observed_at", "demand"]], stations, workers=args.workers, cache_path=INTERIM_DIR / "base_f_by_cutoff.csv")
    replayed_g = add_recent_profile_counterfactual(replay_g(bases, wide), wide)

    drift_matrix = pd.read_csv(DRIFT_MATRIX, dtype={"station_id": str})
    stable_stations = sorted(drift_matrix.loc[drift_matrix["sustained_response"] == "keep current behavior", "station_id"])
    # v1: pre-registered design; v1.1: one structural fix after diagnosing v1 (post-hoc on these data)
    variants = {"candidate_j_v1": (cj.CANDIDATE_J_V1, False), PRIMARY: (cj.CANDIDATE_J_V1_1, True)}
    out = args.output_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)
    report_variants: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for name, (variant_config, post_hoc) in variants.items():
        frame, regimes = replay_j(replayed_g, wide, anomaly, thresholds, variant_config)
        frame["phase"] = frame["data_cutoff"].map(lambda c: phase_of(c, phases))
        frame = frame[frame["phase"].notna() & frame["actual"].notna()].copy()
        regimes["phase"] = regimes["data_cutoff"].map(lambda c: phase_of(c, phases))
        results = evaluate(frame, stable_stations)
        results["path_attribution"] = path_attribution(frame)
        rec = recovery(frame, regimes, phases)
        overall = results["overall"].set_index(["phase", "model"])["mean_station_accuracy"]
        st = results["stations"]
        stable_station_deltas = st[st["stable_station"] & st["phase"].isin(["stable", "phase1", "phase2"])]["delta_pp"]
        decision_inputs = {
            "delta_stable": float(overall[("stable", "J")] - overall[("stable", "G")]),
            "delta_phase1": float(overall[("phase1", "J")] - overall[("phase1", "G")]),
            "delta_phase2": float(overall[("phase2", "J")] - overall[("phase2", "G")]),
            "delta_shock": float(overall[("phase3_shock", "J")] - overall[("phase3_shock", "G")]),
            "worst_stable_station_delta": float(stable_station_deltas.min()) if len(stable_station_deltas) else 0.0,
            "shock_hours": rec["shock_hours_observed"],
            "recovery_observed": rec["recovery_observed"],
        }
        report_variants[name] = {
            "config": {k: v for k, v in vars(variant_config).items() if k != "evidence"},
            "post_hoc": post_hoc,
            "results": results,
            "recovery": rec,
            "decision_inputs": decision_inputs,
            "decision": cj.decide(decision_inputs, post_hoc=post_hoc),
        }
        frames[name] = frame
        (out / "tables" / name).mkdir(parents=True, exist_ok=True)
        for table_name, table in results.items():
            _csv(table, out / "tables" / name / f"{table_name}.csv")
        if name == PRIMARY:
            primary_regimes = regimes
    frame = frames[PRIMARY].merge(
        frames["candidate_j_v1"][["data_cutoff", "station_id", "target_at", "path", "j_prediction"]].rename(columns={"path": "path_v1", "j_prediction": "j_v1_prediction"}),
        on=["data_cutoff", "station_id", "target_at"], how="left",
    )
    regimes = primary_regimes
    parity = official_parity(frame, tables)
    sensitivity = pooled_threshold_sensitivity(wide, anomaly, thresholds, regimes, phases, cj.CANDIDATE_J_V1_1)
    rec = report_variants[PRIMARY]["recovery"]
    decision = report_variants[PRIMARY]["decision"]
    decision_inputs = report_variants[PRIMARY]["decision_inputs"]
    _csv(frame, out / "tables" / "replay_predictions.csv")
    _csv(regimes, out / "tables" / "regimes_by_cycle.csv")
    if parity.get("rows"):
        _csv(parity["per_cycle"], out / "tables" / "official_cycles_comparison.csv")
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "snapshot": str(args.from_snapshot),
        "data_end_utc": _iso(anchor),
        "phases": [{"phase": p["phase"], "start_utc": _iso(p["start"]), "end_utc": _iso(p["end"]), "start_local": str(p["start"].tz_convert(dr.BOGOTA_TZ))} for p in phases],
        "cutoffs_replayed": len(cutoffs),
        "accuracy_definition": "mean_station_accuracy = leaderboard formula (verified vs /v1/leaderboard); global_wape_accuracy = all rows pooled",
        "thresholds": thresholds.to_dict(),
        "primary_variant": PRIMARY,
        "config": report_variants[PRIMARY]["config"],
        "evidence": cj.EVIDENCE,
        "stable_stations_from_drift_analysis": stable_stations,
        "variants": {
            name: {
                "config": v["config"], "post_hoc": v["post_hoc"], "decision_inputs": v["decision_inputs"], "decision": v["decision"],
                "overall": v["results"]["overall"].to_dict("records"), "wins": v["results"]["wins"].to_dict("records"),
                "worst": v["results"]["worst"].to_dict("records"), "path_attribution": v["results"]["path_attribution"].to_dict("records"),
            }
            for name, v in report_variants.items()
        },
        "recovery": rec,
        "sensitivity_pooled_thresholds": sensitivity,
        "official_parity": {k: v for k, v in parity.items() if not isinstance(v, pd.DataFrame)},
        "decision_inputs": decision_inputs,
        "decision": decision,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(_json(payload), indent=2) + "\n")
    results = report_variants[PRIMARY]["results"]
    figures = {} if args.no_figures else render_figures(out / "figures", frame, results, regimes, phases)
    from candidate_j_report import render_markdown

    (out / "latest.md").write_text(render_markdown(payload, report_variants, parity, figures) + "\n")
    print(json.dumps({name: {"decision": v["decision"]["decision"], **{k: round(x, 2) if isinstance(x, float) else x for k, x in v["decision_inputs"].items()}} for name, v in report_variants.items()}))


def render_figures(out: Path, frame: pd.DataFrame, results: dict[str, pd.DataFrame], regimes: pd.DataFrame, phases: list[dict[str, Any]]) -> dict[str, str]:
    from drift_regime_figures import INK_2, MARKERS, SERIES, SURFACE, DIVERGING, plt  # shared style
    import matplotlib.dates as mdates
    from matplotlib.colors import TwoSlopeNorm

    out.mkdir(parents=True, exist_ok=True)
    written = {}
    cycles = results["cycles"].copy()
    cycles["local"] = pd.to_datetime(cycles["data_cutoff"], utc=True).dt.tz_convert(dr.BOGOTA_TZ).dt.tz_localize(None)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.plot(cycles["local"], cycles["g_mean_station_accuracy"], color=SERIES[0], marker=MARKERS[0], markersize=3, linewidth=1.0, label="Candidate G (replay)", markeredgecolor=SURFACE)
    ax1.plot(cycles["local"], cycles["j_mean_station_accuracy"], color=SERIES[1], marker=MARKERS[1], markersize=3, linewidth=1.0, label="Candidate J v1.1 (replay)", markeredgecolor=SURFACE)
    v1 = frame.assign(j_prediction=frame["j_v1_prediction"]).groupby("data_cutoff").apply(lambda c: metrics(c, "j_prediction")["mean_station_accuracy"], include_groups=False)
    ax1.plot(pd.to_datetime(v1.index, utc=True).tz_convert(dr.BOGOTA_TZ).tz_localize(None), v1.values, color=SERIES[2], marker=MARKERS[2], markersize=3, linewidth=0.8, label="Candidate J v1 (replay)", markeredgecolor=SURFACE)
    for phase in phases[1:]:
        for ax in (ax1, ax2):
            ax.axvline(phase["start"].tz_convert(dr.BOGOTA_TZ).tz_localize(None), color=INK_2, linewidth=0.7)
    ax1.set_ylabel("Accuracy per cycle (%)\nleaderboard formula")
    ax1.set_ylim(bottom=-2)
    ax1.legend(loc="lower left")
    ax1.set_title("Candidates J v1 / v1.1 vs Candidate G on the same targets (vertical lines = phase 1, phase 2, shock)", loc="left")
    share = regimes.groupby("data_cutoff")["regime"].apply(lambda r: (r != cj.NORMAL).mean())
    shock = regimes.groupby("data_cutoff")["regime"].apply(lambda r: (r == cj.SEVERE_SHOCK).any())
    local = pd.to_datetime(share.index, utc=True).tz_convert(dr.BOGOTA_TZ).tz_localize(None)
    ax2.bar(local, share.values * 100, width=0.035, color=SERIES[0], label="% stations not NORMAL")
    ax2.bar(local[shock.values], 100, width=0.035, color=SERIES[1], label="SEVERE_SHOCK cycle")
    ax2.set_ylabel("% stations")
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    fig.tight_layout()
    path = out / "01_cycle_accuracy_g_vs_j.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written[path.name] = str(path)

    st = results["stations"].pivot(index="station_id", columns="phase", values="delta_pp")
    st = st[[p for p in ["stable", "phase1", "phase2", "phase3_shock"] if p in st.columns]]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(st.to_numpy(), cmap=DIVERGING.reversed(), norm=TwoSlopeNorm(vcenter=0.0, vmin=-30, vmax=30), aspect="auto")
    ax.set_xticks(range(len(st.columns)), st.columns)
    ax.set_yticks(range(len(st.index)), st.index)
    ax.grid(False)
    for (i, j), value in np.ndenumerate(st.to_numpy()):
        if math.isfinite(value) and abs(value) >= 1:
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7, color="#ffffff" if abs(value) >= 15 else INK_2)
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("J − G accuracy (pp); blue = J better", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_title("Per-station J − G by phase (labels only where |Δ| ≥ 1 pp)", loc="left")
    fig.tight_layout()
    path = out / "02_station_delta_by_phase.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written[path.name] = str(path)

    hz = results["horizons"]
    fig, axes = plt.subplots(1, len(hz["phase"].unique()), figsize=(14, 3.8), sharey=True)
    for ax, (phase, part) in zip(np.atleast_1d(axes), hz.groupby("phase", sort=False)):
        ax.plot(part["horizon_minutes"], part["g_accuracy"], color=SERIES[0], marker=MARKERS[0], label="G", markeredgecolor=SURFACE)
        ax.plot(part["horizon_minutes"], part["j_accuracy"], color=SERIES[1], marker=MARKERS[1], label="J", markeredgecolor=SURFACE)
        ax.set_xticks(list(HORIZONS))
        ax.set_title(phase, loc="left")
        ax.set_xlabel("horizon (min)")
    np.atleast_1d(axes)[0].set_ylabel("Accuracy (%)")
    np.atleast_1d(axes)[0].legend(loc="lower left")
    fig.tight_layout()
    path = out / "03_horizon_by_phase.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written[path.name] = str(path)
    return written


def _load_snapshot(path: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ["observations", "predictions", "forecast_cycles", "forecast_cycle_targets", "submission_receipts", "model_versions", "prediction_traces"]:
        file = path / f"{name}.csv"
        if file.exists():
            frame = pd.read_csv(file, dtype=STRING_COLUMNS)
            if "station_id" in frame:
                frame["station_id"] = frame["station_id"].astype(str).str.zfill(5)
            tables[name] = frame
    return tables


def _lookup(wide: pd.DataFrame, when: pd.Timestamp, station: str) -> float:
    try:
        value = wide.at[when, station]
    except KeyError:
        return math.nan
    return float(value) if not pd.isna(value) else math.nan


def _csv(frame: pd.DataFrame, path: Path) -> None:
    frame = frame.copy()
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.DatetimeTZDtype):
            frame[column] = frame[column].map(lambda ts: ts.isoformat() if not pd.isna(ts) else None)
    frame.to_csv(path, index=False)


def _iso(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    if "--workers" not in sys.argv or sys.argv[sys.argv.index("--workers") + 1] != "1":
        os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker process
    main()
