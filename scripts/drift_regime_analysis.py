"""Drift / regime-change EDA for Pulso TransMi (read-only).

Reproducible source of `reports/drift_regime_analysis/`. The script

1. reads the operational tables from Supabase with GET requests only (or from a local
   snapshot written by a previous run, via `--from-snapshot`);
2. detects change points on causal (4-week same-weekday median) log-ratios;
3. compares a stable reference period against trailing windows (6/12/24/48h) and the
   detected regimes: level, intra-day profile, peaks, volatility and lag relationships;
4. scores Candidate G official predictions (residual = actual - prediction), optionally
   decomposing them into the Candidate F base and the G2 calibration factor by refitting
   the base per cycle with observations <= data_cutoff only;
5. compares challengers only on cycles where both models have complete resolved sets;
6. writes `latest.md`, `summary.json`, `station_diagnostics.csv`, tables and figures.

It never writes to Supabase, never changes `model_versions.is_active` and never calls the
competition API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import drift_regime as dr  # noqa: E402
from drift_monitor import official_scored_predictions  # noqa: E402


REPORT_DIR = ROOT / "reports" / "drift_regime_analysis"
INTERIM_DIR = ROOT / "data" / "interim" / "drift_regime_analysis"
CHAMPION_MODEL_NAME = "candidate_g_f4_profile_ratio_g2_ewma"
CANDIDATE_I_MODEL_NAME = "candidate_i_f4_oos_g2_snr_gate"
CANDIDATE_B_MODEL_NAME = "candidate_b_hist_gradient_boosting_shadow"
TRAILING_HOURS = (6, 12, 24, 48)
SUSTAINED_WINDOW = "last_48h"
SHOCK_WINDOW = "last_6h"
REFERENCE_DAYS = 28
GLOBAL_CP = dr.ChangePointConfig(min_segment_hours=3, penalty_beta=4.0, min_effect=math.log(1.10))
STATION_CP = dr.ChangePointConfig(min_segment_hours=6, penalty_beta=4.0, min_effect=math.log(1.10))
G_CLIP = (0.35, 2.5)

TABLE_SELECTS = {
    "observations": ("station_id,observed_at,demand", "observed_at.asc,station_id.asc"),
    "predictions": ("cycle_id,model_id,station_id,target_observed_at,horizon_minutes,prediction", "target_observed_at.asc"),
    "forecast_cycles": ("cycle_id,data_cutoff,closes_at,expected_predictions,first_seen_at", "data_cutoff.asc"),
    "forecast_cycle_targets": ("cycle_id,station_id,target_at,horizon_minutes,target_order", "cycle_id.asc,target_order.asc"),
    "submission_receipts": ("cycle_id,model_id,status,is_official,accepted_at,created_at", "created_at.asc"),
    "model_versions": ("model_id,model_name,model_type,is_active,trained_until,artifact_uri", "trained_until.asc"),
    "prediction_traces": ("model_id,cycle_id,station_id,target_at,horizon_minutes,base_prediction,calibration_factor,final_prediction,feedback_samples,data_cutoff", "target_at.asc"),
}
STRING_COLUMNS = {"station_id": str, "cycle_id": str, "model_id": str}


# --------------------------------------------------------------------------------------
# Loading (read-only)
# --------------------------------------------------------------------------------------


def load_from_supabase() -> dict[str, pd.DataFrame]:
    from run_monitoring_snapshot import SupabaseMonitoringStore

    tables: dict[str, pd.DataFrame] = {}
    with SupabaseMonitoringStore(timeout=60.0) as store:  # GET requests only
        for name, (select, order) in TABLE_SELECTS.items():
            tables[name] = store.table(name, select=select, order=order, limit=500_000)
    return tables


def load_snapshot(path: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path / f"{name}.csv", dtype=STRING_COLUMNS) for name in TABLE_SELECTS if (path / f"{name}.csv").exists()}


def write_snapshot(tables: dict[str, pd.DataFrame], path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"written_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "tables": {}}
    for name, frame in tables.items():
        file = path / f"{name}.csv"
        frame.to_csv(file, index=False)
        manifest["tables"][name] = {"rows": int(len(frame)), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def normalize_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {}
    for name, frame in tables.items():
        frame = frame.copy()
        for column in ("station_id",):
            if column in frame:
                frame[column] = frame[column].astype(str).str.zfill(5)
        for column in ("cycle_id", "model_id"):
            if column in frame:
                frame[column] = frame[column].astype(str)
        result[name] = frame
    return result


# --------------------------------------------------------------------------------------
# Observation-side analysis
# --------------------------------------------------------------------------------------


def observation_analysis(observations: pd.DataFrame) -> dict[str, Any]:
    prepared = dr.prepare_observations(observations)
    wide = dr.demand_matrix(prepared)
    baseline = dr.seasonal_baseline(wide, weeks=4)
    station_lr = dr.hourly_log_ratio(wide, baseline).dropna(how="all")
    indices = dr.global_drift_indices(wide, baseline).dropna()
    station_lr = station_lr.loc[indices.index.min() :]

    global_segments = pd.concat(
        [dr.segment_series(indices[name], GLOBAL_CP).assign(series=name) for name in ("global_intensity", "global_level")],
        ignore_index=True,
    )
    station_segments = pd.concat(
        [dr.segment_series(station_lr[station].dropna(), STATION_CP).assign(series=station) for station in station_lr.columns],
        ignore_index=True,
    )
    segments = pd.concat([global_segments, station_segments], ignore_index=True)

    onsets = []
    for series, group in segments.groupby("series", sort=False):
        config = GLOBAL_CP if series.startswith("global") else STATION_CP
        info = dr.onset_from_segments(group, min_effect=config.min_effect)
        last = group.sort_values("start").iloc[-1]
        onsets.append(
            {
                "series": series,
                "onset_utc": info["onset"],
                "onset_local": info["onset"].tz_convert(dr.BOGOTA_TZ) if info["onset"] is not None else None,
                "baseline_mean_log_ratio": info["baseline_mean"],
                "first_shift_ratio": math.exp(info["first_shift_mean"]) if info["first_shift_mean"] is not None else None,
                "segments_after_onset": int((group["start"] >= info["onset"]).sum()) if info["onset"] is not None else 0,
                "latest_segment_start_local": last["start"].tz_convert(dr.BOGOTA_TZ),
                "latest_segment_ratio": float(last["ratio"]),
                "latest_segment_hours": int(last["hours"]),
            }
        )
    onsets = pd.DataFrame(onsets)

    sensitivity = []
    for min_hours in (2, 3, 4, 6):
        for beta in (3.0, 4.0, 6.0):
            config = dr.ChangePointConfig(min_segment_hours=min_hours, penalty_beta=beta, min_effect=GLOBAL_CP.min_effect)
            seg = dr.segment_series(indices["global_intensity"], config)
            sensitivity.append(
                {
                    "min_segment_hours": min_hours,
                    "penalty_beta": beta,
                    "change_points_local": "; ".join(str(ts.tz_convert(dr.BOGOTA_TZ))[:16] for ts in seg["start"].iloc[1:]),
                    "segment_ratios": "; ".join(f"{ratio:.2f}" for ratio in seg["ratio"]),
                }
            )

    material = onsets.dropna(subset=["onset_utc"])
    earliest = material["onset_utc"].min()
    reference = dr.reference_period(earliest, prepared["observed_at"].min(), days=REFERENCE_DAYS)
    profile = dr.reference_profile(prepared, reference)
    anchor = prepared["observed_at"].max()
    trailing = dr.trailing_windows(anchor, TRAILING_HOURS)

    global_rows = global_segments[global_segments["series"] == "global_intensity"].sort_values("start").reset_index(drop=True)
    regimes = []
    for i, row in global_rows.iterrows():
        start = max(row["start"], reference.end) if i == 0 else row["start"]
        end = row["end"] if i < len(global_rows) - 1 else anchor
        name = "regime_0_stable" if i == 0 else f"regime_{i}"
        regimes.append(dr.Window(name, start - dr.SLOT, end - dr.SLOT if i < len(global_rows) - 1 else end, "regime"))
    regimes = [window for window in regimes if window.end > window.start]

    windows = [reference] + trailing + regimes[1:]
    coverage = pd.DataFrame([dr.window_coverage(prepared, window) for window in windows])

    ref_wide = dr.reference_matrix(profile, wide.index, wide.columns)
    reference_lags = dr.lag_metrics(wide, reference)[["station_id", "lag", "naive_accuracy", "lag_corr"]].rename(
        columns={"naive_accuracy": "reference_naive_accuracy", "lag_corr": "reference_lag_corr"}
    )
    station_metrics, lag_tables = [], []
    for window in windows:
        metrics = dr.station_window_metrics(prepared, profile, window)
        lags = dr.lag_metrics(wide, window, ref_wide=ref_wide)
        if window.kind != "reference":
            null = dr.null_station_metrics(prepared, window, reference)
            null_summary = dr.summarize_null(null, ["log_level_ratio", "hourly_ratio_dispersion", "shape_corr", "volatility"])
            metrics = metrics.merge(null_summary, on="station_id", how="left")
            null_lags = dr.null_lag_metrics(wide, window, reference)
            lags = dr.lag_reliability(lags, null_lags).assign(null_windows=null["null_window"].nunique() if not null.empty else 0)
        lags = lags.merge(reference_lags, on=["station_id", "lag"], how="left")
        station_metrics.append(metrics.assign(window_kind=window.kind))
        lag_tables.append(lags.assign(window_kind=window.kind))
    station_metrics = pd.concat(station_metrics, ignore_index=True)
    lag_table = pd.concat(lag_tables, ignore_index=True)

    hour_change = pd.concat([dr.hour_of_day_change(prepared, profile, window) for window in windows if window.kind != "reference"], ignore_index=True)
    recent_dates = sorted(prepared["local_date"].unique())[-6:]
    peaks = dr.daily_peaks(prepared, profile, recent_dates)
    consistency = day_consistency(prepared, recent_dates)
    volatility = dr.daily_volatility(prepared)

    return {
        "prepared": prepared,
        "wide": wide,
        "baseline": baseline,
        "station_lr": station_lr,
        "indices": indices,
        "segments": segments,
        "onsets": onsets,
        "sensitivity": pd.DataFrame(sensitivity),
        "reference": reference,
        "profile": profile,
        "anchor": anchor,
        "trailing": trailing,
        "regimes": regimes,
        "windows": windows,
        "coverage": coverage,
        "station_metrics": station_metrics,
        "lag_table": lag_table,
        "hour_change": hour_change,
        "peaks": peaks,
        "consistency": consistency,
        "daily_volatility": volatility,
        "reference_stability": reference_stability(prepared, profile, reference),
    }


def day_consistency(prepared: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """Do the last two complete local days share the same intra-day profile? (new-regime stability)"""
    complete = [d for d in dates if prepared[prepared["local_date"] == d].groupby("station_id").size().min() == 96]
    if len(complete) < 2:
        return pd.DataFrame(columns=["station_id", "day_a", "day_b", "profile_corr", "level_ratio_b_over_a"])
    day_a, day_b = complete[-2], complete[-1]
    rows = []
    for station_id, group in prepared[prepared["local_date"].isin([day_a, day_b])].groupby("station_id"):
        a = group[group["local_date"] == day_a].set_index("slot")["demand"]
        b = group[group["local_date"] == day_b].set_index("slot")["demand"].reindex(a.index)
        rows.append(
            {
                "station_id": station_id,
                "day_a": day_a,
                "day_b": day_b,
                "profile_corr": float(np.corrcoef(a, b)[0, 1]),
                "level_ratio_b_over_a": float(b.sum() / a.sum()) if a.sum() > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def reference_stability(prepared: pd.DataFrame, profile: pd.DataFrame, reference: dr.Window) -> pd.DataFrame:
    rows = dr.attach_reference(prepared[reference.mask(prepared["observed_at"])], profile)
    daily = rows.groupby(["station_id", "local_date"]).agg(actual=("demand", "sum"), ref=("ref", "sum")).reset_index()
    daily["daily_ratio"] = daily["actual"] / daily["ref"]
    return daily.groupby("station_id")["daily_ratio"].agg(
        daily_ratio_median="median", daily_ratio_min="min", daily_ratio_max="max", daily_ratio_std="std"
    ).reset_index()


# --------------------------------------------------------------------------------------
# Prediction-side analysis
# --------------------------------------------------------------------------------------


def prediction_analysis(tables: dict[str, pd.DataFrame], obs: dict[str, Any], g_mode: str) -> dict[str, Any]:
    scored = dr.resolve_predictions(
        predictions=tables["predictions"],
        forecast_cycles=tables["forecast_cycles"],
        forecast_cycle_targets=tables["forecast_cycle_targets"],
        observations=tables["observations"],
        submission_receipts=tables["submission_receipts"],
        model_versions=tables["model_versions"],
    )
    models = tables["model_versions"]
    champion = models[models["model_name"] == CHAMPION_MODEL_NAME]
    if champion.empty:
        raise RuntimeError(f"{CHAMPION_MODEL_NAME} not found in model_versions")
    g_model_id = str(champion.iloc[0]["model_id"])
    canonical = official_scored_predictions(
        predictions=tables["predictions"],
        submission_receipts=tables["submission_receipts"],
        forecast_cycles=tables["forecast_cycles"],
        forecast_cycle_targets=tables["forecast_cycle_targets"],
        observations=tables["observations"],
    )
    official = scored[scored["is_official"]]
    parity = {
        "official_rows_this_script": int(len(official)),
        "official_rows_drift_monitor": int(len(canonical)),
        "official_cycles_this_script": int(official["cycle_id"].nunique()),
        "official_cycles_drift_monitor": int(canonical["cycle_id"].nunique()),
    }

    g = scored[(scored["model_id"] == g_model_id) & scored["is_official"]].copy()
    cycles = dr.cycle_metrics(scored)
    all_cycles = tables["forecast_cycles"].copy()
    all_cycles["data_cutoff"] = pd.to_datetime(all_cycles["data_cutoff"], utc=True)
    unresolved_g = sorted(set(tables["predictions"].loc[tables["predictions"]["model_id"] == g_model_id, "cycle_id"]) - set(g["cycle_id"]))

    windows = obs["trailing"] + obs["regimes"][1:]
    window_rows = []
    for window in windows:
        rows = g[window.mask(g["target_at"])]
        window_rows.append(rows.assign(window=window.name))
    g_windows = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    overall = dr.error_summary(g_windows, ["window"])
    by_station = dr.error_summary(g_windows, ["window", "station_id"])
    by_horizon = dr.error_summary(g_windows, ["window", "horizon_minutes"])
    by_hour = dr.error_summary(g_windows, ["window", "target_hour_local"])
    persistence = bias_persistence(g_windows)
    by_station = by_station.merge(persistence, on=["window", "station_id"], how="left")
    all_g = dr.error_summary(g.assign(window="all_g_cycles"), ["window"])
    overall = pd.concat([all_g, overall], ignore_index=True)

    extremes = dr.extreme_errors(g)
    decomposition = g_decomposition(tables, obs, g, g_model_id, mode=g_mode)
    challengers = challenger_comparisons(scored, g_model_id, models, obs)
    return {
        "scored": scored,
        "g": g,
        "g_model_id": g_model_id,
        "parity": parity,
        "cycles": cycles,
        "unresolved_g_cycles": unresolved_g,
        "g_windows": g_windows,
        "overall": overall,
        "by_station": by_station,
        "by_horizon": by_horizon,
        "by_hour": by_hour,
        "extremes": extremes,
        "decomposition": decomposition,
        "challengers": challengers,
    }


def bias_persistence(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of cycles in which the station residual sum is positive (under-prediction)."""
    if frame.empty:
        return pd.DataFrame(columns=["window", "station_id", "cycles", "share_cycles_underpredicted"])
    per_cycle = frame.groupby(["window", "station_id", "cycle_id"])["residual"].sum().reset_index()
    return per_cycle.groupby(["window", "station_id"]).agg(
        cycles=("cycle_id", "nunique"), share_cycles_underpredicted=("residual", lambda values: float((values > 0).mean()))
    ).reset_index()


def g_decomposition(tables: dict[str, pd.DataFrame], obs: dict[str, Any], g: pd.DataFrame, g_model_id: str, *, mode: str) -> dict[str, Any]:
    """Split Candidate G = max(0, base_F * factor_G2) using a leakage-safe base refit per cycle."""
    if mode == "skip" or g.empty:
        return {"status": "skipped"}
    cache_path = INTERIM_DIR / "g_base_reconstruction.csv"
    cache = pd.read_csv(cache_path, dtype=STRING_COLUMNS) if cache_path.exists() else pd.DataFrame(columns=["cycle_id"])
    done = set(cache["cycle_id"].astype(str))
    todo = sorted(set(g["cycle_id"]) - done)
    if todo and mode == "compute":
        cache = reconstruct_bases(tables, obs["prepared"], todo, cache, cache_path)
    elif todo:
        print(f"[g-decomposition] {len(todo)} cycles not cached; run with --g-decomposition compute", file=sys.stderr)
    if cache.empty:
        return {"status": "no_cache"}

    base = cache.copy()
    base["station_id"] = base["station_id"].astype(str).str.zfill(5)
    base["target_at"] = pd.to_datetime(base["target_at"], utc=True)
    base["max_observed_at_used"] = pd.to_datetime(base["max_observed_at_used"], utc=True)
    base["data_cutoff"] = pd.to_datetime(base["data_cutoff"], utc=True)
    if (base["max_observed_at_used"] > base["data_cutoff"]).any():
        raise RuntimeError("base reconstruction used observations after data_cutoff")
    merged = g.merge(base[["cycle_id", "station_id", "target_at", "base_prediction"]], on=["cycle_id", "station_id", "target_at"], how="inner")
    merged["implied_factor"] = np.where(merged["base_prediction"] > 0, merged["prediction"] / merged["base_prediction"], np.nan)
    merged["factor_clipped_high"] = merged["implied_factor"] >= G_CLIP[1] - 1e-6
    merged["factor_clipped_low"] = merged["implied_factor"] <= G_CLIP[0] + 1e-6
    merged["needed_factor"] = np.where(merged["base_prediction"] > 0, merged["actual"] / merged["base_prediction"], np.nan)

    # G2 applies one factor per (cycle, station): if the implied factor varies across horizons,
    # the reconstructed base does not match what production computed for that cycle.
    factor_spread = merged.groupby(["cycle_id", "station_id"])["implied_factor"].agg(lambda values: float(values.max() - values.min()))
    mismatched = sorted(factor_spread[factor_spread > 1e-6].index.get_level_values("cycle_id").unique())
    merged = merged[~merged["cycle_id"].isin(mismatched)].copy()
    in_clip = merged["implied_factor"].between(G_CLIP[0] - 1e-6, G_CLIP[1] + 1e-6)
    traces = tables.get("prediction_traces", pd.DataFrame())
    trace_parity = None
    if not traces.empty:
        trace = traces[traces["model_id"] == g_model_id].copy()
        trace["station_id"] = trace["station_id"].astype(str).str.zfill(5)
        trace["target_at"] = pd.to_datetime(trace["target_at"], utc=True)
        joined = trace.merge(base, on=["cycle_id", "station_id", "target_at"], how="inner", suffixes=("_trace", ""))
        if not joined.empty:
            trace_parity = {
                "rows": int(len(joined)),
                "cycles": int(joined["cycle_id"].nunique()),
                "max_abs_base_diff": float((joined["base_prediction_trace"].astype(float) - joined["base_prediction"]).abs().max()),
            }

    windows = obs["trailing"] + obs["regimes"][1:]
    base_frame = merged.assign(prediction=merged["base_prediction"])
    rows = []
    for window in [dr.Window("all_g_cycles", pd.Timestamp.min.tz_localize("UTC"), pd.Timestamp.max.tz_localize("UTC"), "all")] + windows:
        mask = window.mask(merged["target_at"])
        if not mask.any():
            continue
        final = dr.error_summary(merged[mask], [])
        base_metrics = dr.error_summary(base_frame[mask], [])
        log_factor = np.log(merged.loc[mask, "implied_factor"].clip(lower=1e-6))
        log_needed = np.log(merged.loc[mask, "needed_factor"].clip(lower=1e-3))
        rows.append(
            {
                "window": window.name,
                "rows": int(mask.sum()),
                "base_accuracy": float(base_metrics["accuracy"].iloc[0]),
                "g_accuracy": float(final["accuracy"].iloc[0]),
                "g2_effect_pp": float(final["accuracy"].iloc[0] - base_metrics["accuracy"].iloc[0]),
                "base_relative_bias": float(base_metrics["relative_bias"].iloc[0]),
                "g_relative_bias": float(final["relative_bias"].iloc[0]),
                "share_factor_clipped_high": float(merged.loc[mask, "factor_clipped_high"].mean()),
                "share_factor_clipped_low": float(merged.loc[mask, "factor_clipped_low"].mean()),
                "median_factor": float(merged.loc[mask, "implied_factor"].median()),
                "corr_log_factor_vs_needed": dr._corr(log_factor.to_numpy(), log_needed.to_numpy()),
                "share_factor_wrong_direction": float((np.sign(log_factor) * np.sign(log_needed) < 0).mean()),
            }
        )
    per_cycle = []
    for cycle_id, group in merged.groupby("cycle_id"):
        per_cycle.append(
            {
                "cycle_id": cycle_id,
                "data_cutoff": group["data_cutoff"].iloc[0],
                "base_accuracy": dr._pooled_accuracy(group["actual"], group["base_prediction"]),
                "g_accuracy": dr._pooled_accuracy(group["actual"], group["prediction"]),
                "share_factor_clipped": float((group["factor_clipped_high"] | group["factor_clipped_low"]).mean()),
                "mean_abs_log_factor": float(np.log(group["implied_factor"].clip(lower=1e-6)).abs().mean()),
            }
        )
    per_cycle = pd.DataFrame(per_cycle).sort_values("data_cutoff")
    per_cycle["g2_effect_pp"] = per_cycle["g_accuracy"] - per_cycle["base_accuracy"]
    station_rows = []
    for window in windows:
        mask = window.mask(merged["target_at"])
        for station_id, group in merged[mask].groupby("station_id"):
            station_rows.append(
                {
                    "window": window.name,
                    "station_id": station_id,
                    "base_accuracy": dr._pooled_accuracy(group["actual"], group["base_prediction"]),
                    "g_accuracy": dr._pooled_accuracy(group["actual"], group["prediction"]),
                    "median_factor": float(group["implied_factor"].median()),
                    "share_factor_clipped": float((group["factor_clipped_high"] | group["factor_clipped_low"]).mean()),
                }
            )
    by_station = pd.DataFrame(station_rows)
    if not by_station.empty:
        by_station["g2_effect_pp"] = by_station["g_accuracy"] - by_station["base_accuracy"]
    return {
        "status": "ok",
        "cycles": int(merged["cycle_id"].nunique()),
        "missing_cycles": sorted(set(g["cycle_id"]) - set(merged["cycle_id"])),
        "parity": {
            "share_factor_inside_clip": float(in_clip.mean()),
            "max_factor_spread_within_cycle_station": float(factor_spread.drop(index=mismatched, level="cycle_id").max()) if len(factor_spread) else math.nan,
            "parity_mismatch_cycles_excluded": mismatched,
            "trace_parity": trace_parity,
        },
        "rows": merged,
        "windows": pd.DataFrame(rows),
        "per_cycle": per_cycle,
        "by_station": by_station,
    }


def reconstruct_bases(tables: dict[str, pd.DataFrame], prepared: pd.DataFrame, cycle_ids: list[str], cache: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    from candidate_g_adaptive import build_candidate_f_base_frame
    from forecast_cycle_sync import CycleTarget, ForecastCycle

    cycles = tables["forecast_cycles"].set_index("cycle_id")
    targets = tables["forecast_cycle_targets"].copy()
    observations = prepared[["station_id", "observed_at", "demand"]]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [cache] if not cache.empty else []
    for i, cycle_id in enumerate(cycle_ids, start=1):
        started = time.perf_counter()
        cutoff = pd.Timestamp(cycles.loc[cycle_id, "data_cutoff"]).tz_convert("UTC")
        cycle_targets = targets[targets["cycle_id"] == cycle_id].sort_values("target_order")
        cycle = ForecastCycle(
            cycle_id=cycle_id,
            data_cutoff=_iso(cutoff),
            closes_at="",
            expected_predictions=int(cycles.loc[cycle_id, "expected_predictions"]),
            targets=[
                CycleTarget(
                    station_id=str(row.station_id).zfill(5),
                    target_at=_iso(pd.Timestamp(row.target_at)),
                    horizon_minutes=int(row.horizon_minutes),
                    target_order=int(row.target_order),
                    raw_target={},
                )
                for row in cycle_targets.itertuples(index=False)
            ],
            raw_cycle={},
        )
        eligible = observations[observations["observed_at"] <= cutoff]  # information available at prediction time
        base = build_candidate_f_base_frame(cycle, eligible)
        frame = pd.DataFrame(
            {
                "cycle_id": cycle_id,
                "station_id": base["station_id"].astype(str),
                "target_at": base["target_at"].map(_iso),
                "horizon_minutes": base["horizon_minutes"].astype(int),
                "base_prediction": base["base_prediction"].astype(float),
                "data_cutoff": _iso(cutoff),
                "max_observed_at_used": _iso(eligible["observed_at"].max()),
            }
        )
        frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(cache_path, index=False)
        print(f"[g-decomposition] {i}/{len(cycle_ids)} {cycle_id} {time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)
    return pd.concat(frames, ignore_index=True)


def challenger_comparisons(scored: pd.DataFrame, g_model_id: str, models: pd.DataFrame, obs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = models.set_index("model_id")["model_name"].to_dict()
    for model_id, name in names.items():
        if str(model_id) == g_model_id:
            continue
        comparison = dr.paired_comparison(scored, g_model_id, str(model_id))
        if comparison["status"] == "ok":
            per_cycle = comparison["per_cycle"]
            per_cycle["regime"] = per_cycle["data_cutoff"].map(lambda ts: regime_of(ts, obs["regimes"]))
            comparison["by_regime"] = per_cycle.groupby("regime").agg(
                cycles=("cycle_id", "size"),
                champion_mean=("champion_accuracy", "mean"),
                challenger_mean=("challenger_accuracy", "mean"),
                mean_delta_pp=("delta_pp", "mean"),
                wins=("delta_pp", lambda values: int((values > 0).sum())),
            ).reset_index()
        own = scored[scored["model_id"].astype(str) == str(model_id)]
        comparison["model_name"] = name
        comparison["model_id"] = str(model_id)
        comparison["own_resolved_cycles"] = int(own["cycle_id"].nunique())
        comparison["own_cutoff_range_local"] = (
            [str(own["data_cutoff_local"].min()), str(own["data_cutoff_local"].max())] if not own.empty else None
        )
        result[f"{name}|{str(model_id)[:8]}"] = comparison
    return result


def regime_of(timestamp: pd.Timestamp, regimes: list[dr.Window]) -> str:
    for window in regimes:
        if window.start < timestamp <= window.end:
            return window.name
    return "before_regimes" if regimes and timestamp <= regimes[0].start else "unassigned"


# --------------------------------------------------------------------------------------
# Station matrix and global synthesis
# --------------------------------------------------------------------------------------


def station_matrix(obs: dict[str, Any], pred: dict[str, Any]) -> pd.DataFrame:
    """One row per station. Classification uses the detected regimes: the sustained regime
    (before the latest change point) and the shock (latest regime); 48h/6h values are kept
    as extra columns because those are the requested trailing windows."""
    config = dr.ClassificationConfig()
    metrics = obs["station_metrics"]
    sustained_w = sustained_regime_window(obs)
    shock_w = shock_window(obs)
    sustained = metrics[metrics["window"] == sustained_w.name].set_index("station_id")
    shock = metrics[metrics["window"] == shock_w.name].set_index("station_id")
    trailing = {name: metrics[metrics["window"] == name].set_index("station_id") for name in ("last_48h", "last_6h")}
    lags = obs["lag_table"]
    onsets = obs["onsets"].set_index("series")
    consistency = obs["consistency"].set_index("station_id") if not obs["consistency"].empty else pd.DataFrame()
    peaks = obs["peaks"]
    by_station = pred["by_station"]
    decomposition = pred["decomposition"]
    regime_hours = float((obs["anchor"] - sustained_w.start) / pd.Timedelta(hours=1)) if sustained_w.kind == "regime" else math.nan
    order = ["none", "low", "moderate", "high", "extreme"]
    rows = []
    for station_id in sorted(sustained.index):
        s = sustained.loc[station_id]
        k = shock.loc[station_id]
        row: dict[str, Any] = {"station_id": station_id, "sustained_window": sustained_w.name, "shock_window": shock_w.name}
        onset = onsets.loc[station_id, "onset_local"] if station_id in onsets.index else None
        row["drift_onset_local"] = str(onset)[:16] if onset is not None and not pd.isna(onset) else "no material change"
        row["sustained_level_shift"] = dr.classify_level(s["log_level_ratio"], s.get("null_abs_log_level_max", math.nan), config)
        row["sustained_profile_shift"] = dr.classify_profile(s, config)
        row["sustained_variance_shift"] = dr.classify_variance(s, config)
        row["shock_level_shift"] = dr.classify_level(k["log_level_ratio"], k.get("null_abs_log_level_max", math.nan), config)
        row["shock_profile_shift"] = dr.classify_profile(k, config)
        row["shock_variance_shift"] = dr.classify_variance(k, config)
        row["sustained_severity"] = dr.severity_label(row["sustained_level_shift"], row["sustained_profile_shift"])
        row["shock_severity"] = dr.severity_label(row["shock_level_shift"], row["shock_profile_shift"])
        row["drift_severity"] = max([row["sustained_severity"], row["shock_severity"]], key=order.index)
        row["level_shift"] = row["sustained_level_shift"]
        row["profile_shift"] = row["sustained_profile_shift"]
        row["variance_shift"] = row["sustained_variance_shift"]
        row["level_ratio_sustained"] = s["level_ratio"]
        row["level_ratio_shock"] = k["level_ratio"]
        for name, frame in trailing.items():
            row[f"level_ratio_{name}"] = frame.loc[station_id, "level_ratio"] if station_id in frame.index else math.nan
        row["hourly_ratio_dispersion_sustained"] = s["hourly_ratio_dispersion"]
        row["timing_shift_minutes_sustained"] = s["timing_shift_minutes"]
        row["timing_shift_minutes_shock"] = k["timing_shift_minutes"]
        row["volatility_ratio_sustained"] = _ratio(s["volatility"], s.get("null_volatility_median", math.nan))
        row["volatility_ratio_shock"] = _ratio(k["volatility"], k.get("null_volatility_median", math.nan))
        for lag_name, column in (("lag_15", "lag15"), ("lag_30", "lag30"), ("lag_60", "lag60"), ("lag_96", "lag96"), ("lag_weekly", "weekly_lag")):
            sus = lags[(lags["window"] == sustained_w.name) & (lags["station_id"] == station_id) & (lags["lag"] == lag_name)]
            sho = lags[(lags["window"] == shock_w.name) & (lags["station_id"] == station_id) & (lags["lag"] == lag_name)]
            row[f"{column}_reliability"] = _first(sus, "reliability", "insufficient")
            row[f"{column}_naive_acc_delta_pp"] = _first(sus, "naive_accuracy_delta_pp", math.nan)
            row[f"{column}_reliability_shock"] = _first(sho, "reliability", "insufficient")
            row[f"{column}_naive_acc_delta_pp_shock"] = _first(sho, "naive_accuracy_delta_pp", math.nan)
        for band in ("am", "pm"):
            band_rows = peaks[(peaks["station_id"] == station_id) & (peaks["band"] == band) & peaks["band_complete"]].sort_values("local_date").tail(2)
            row[f"{band}_peak_ratio_last2d"] = float(band_rows["peak_ratio"].mean()) if not band_rows.empty else math.nan
            row[f"{band}_peak_shift_min_last2d"] = float(band_rows["peak_shift_minutes"].mean()) if not band_rows.empty else math.nan
        night = peaks[(peaks["station_id"] == station_id) & (peaks["band"] == "night")].sort_values("local_date").tail(1)
        row["night_peak_ratio_latest"] = _first(night, "peak_ratio", math.nan)
        row["new_regime_day_consistency"] = float(consistency.loc[station_id, "profile_corr"]) if station_id in consistency.index else math.nan
        row["new_regime_hours"] = regime_hours
        for window, prefix in ((sustained_w.name, "g_sustained"), (shock_w.name, "g_shock"), ("last_48h", "g_last_48h"), ("last_6h", "g_last_6h")):
            g_row = by_station[(by_station["window"] == window) & (by_station["station_id"] == station_id)]
            row[f"{prefix}_rows"] = int(_first(g_row, "count", 0))
            for metric in ("accuracy", "wape", "signed_bias", "relative_bias", "pred_actual_ratio"):
                row[f"{prefix}_{metric}"] = _first(g_row, metric, math.nan)
        if decomposition.get("status") == "ok" and not decomposition["by_station"].empty:
            dec = decomposition["by_station"]
            for window, suffix in ((sustained_w.name, "sustained"), (shock_w.name, "shock")):
                d = dec[(dec["window"] == window) & (dec["station_id"] == station_id)]
                row[f"base_f_accuracy_{suffix}"] = _first(d, "base_accuracy", math.nan)
                row[f"g2_effect_pp_{suffix}"] = _first(d, "g2_effect_pp", math.nan)
                row[f"g2_median_factor_{suffix}"] = _first(d, "median_factor", math.nan)
        response = dr.recommend_station_response(row, config)
        row["recommended_response"] = response.primary
        row["sustained_response"] = response.sustained
        row["shock_response"] = response.shock
        row["all_responses"] = "; ".join(response.responses)
        row["evidence"] = " | ".join(response.evidence)
        rows.append(row)
    matrix = pd.DataFrame(rows).rename(
        columns={"g_sustained_signed_bias": "signed_bias", "g_sustained_relative_bias": "relative_bias", "g_sustained_wape": "wape", "g_sustained_accuracy": "accuracy"}
    )
    lead = [
        "station_id", "drift_onset_local", "drift_severity", "sustained_severity", "shock_severity", "level_shift", "profile_shift",
        "variance_shift", "lag15_reliability", "lag30_reliability", "lag60_reliability", "lag96_reliability", "weekly_lag_reliability",
        "signed_bias", "relative_bias", "wape", "accuracy", "g_shock_accuracy", "g_last_48h_accuracy", "recommended_response",
    ]
    return matrix[lead + [c for c in matrix.columns if c not in lead]]


def sustained_regime_window(obs: dict[str, Any]) -> dr.Window:
    """The regime before the latest one (the latest is the shock); falls back to last_48h."""
    regimes = [window for window in obs["regimes"] if window.name != "regime_0_stable"]
    if len(regimes) >= 2:
        return regimes[-2]
    return next(window for window in obs["trailing"] if window.name == SUSTAINED_WINDOW)


def shock_window(obs: dict[str, Any]) -> dr.Window:
    """The latest detected regime; falls back to last_6h."""
    regimes = [window for window in obs["regimes"] if window.name != "regime_0_stable"]
    if len(regimes) >= 2:
        return regimes[-1]
    return next(window for window in obs["trailing"] if window.name == SHOCK_WINDOW)


def _first(frame: pd.DataFrame, column: str, default: Any) -> Any:
    if frame.empty or column not in frame:
        return default
    value = frame[column].iloc[0]
    return float(value) if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool) else value


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if isinstance(denominator, (int, float)) and math.isfinite(denominator) and denominator > 0 else math.nan


def global_synthesis(obs: dict[str, Any], pred: dict[str, Any], matrix: pd.DataFrame) -> dict[str, Any]:
    n = len(matrix)
    sustained_level = matrix[~matrix["level_shift"].isin(["none", "insufficient"])]
    level_up = sustained_level[sustained_level["level_shift"].str.endswith("up")]["station_id"].tolist()
    level_down = sustained_level[sustained_level["level_shift"].str.endswith("down")]["station_id"].tolist()
    profile = matrix[matrix["profile_shift"] != "none"]["station_id"].tolist()
    variance = matrix[matrix["variance_shift"].str.startswith("up")]["station_id"].tolist()
    variance_down = matrix[matrix["variance_shift"].str.startswith("down")]["station_id"].tolist()
    shock = matrix[matrix["shock_severity"].isin(["high", "extreme"])]["station_id"].tolist()
    lag_counts = {
        column: matrix[column].value_counts().to_dict()
        for column in ["lag15_reliability", "lag30_reliability", "lag60_reliability", "lag96_reliability", "weekly_lag_reliability",
                       "lag15_reliability_shock", "lag30_reliability_shock", "lag60_reliability_shock", "lag96_reliability_shock", "weekly_lag_reliability_shock"]
    }
    stable = matrix[(matrix["sustained_severity"] == "none")]["station_id"].tolist()
    types = []
    if sustained_level.shape[0]:
        types.append("level shift")
    if profile:
        types.append("temporal/profile shift")
    shock_variance = matrix[matrix["shock_variance_shift"].str.startswith("up")]["station_id"].tolist()
    if variance or variance_down or shock_variance:
        types.append("variance shift")
    if lag_counts["lag96_reliability"].get("broken", 0) + lag_counts["weekly_lag_reliability"].get("broken", 0) > 0:
        types.append("lag relationship failure")
    if 0 < len(sustained_level) < n * 0.75:
        types.append("localized station drift")
    if len(shock) >= n * 0.75:
        types.append("global regime change")
    regimes = obs["regimes"]
    latest = regimes[-1] if regimes else None
    latest_hours = latest.hours if latest else 0.0
    sustained_regime = sustained_regime_window(obs)
    sustained_hours = float((obs["anchor"] - sustained_regime.start) / pd.Timedelta(hours=1))
    needs_change = matrix[matrix["sustained_response"] != "keep current behavior"]["station_id"].tolist()
    shock_guard = matrix[matrix["shock_response"] == "robust recent baseline"]["station_id"].tolist()
    system: list[dict[str, str]] = []
    if len(shock_guard) >= 0.75 * n:
        system.append({
            "response": "global robust guard",
            "reason": f"{len(shock_guard)}/{n} stations collapsed in the latest regime ({latest_hours:.1f}h old): too recent to refit, needs a guard against compounding",
        })
    if len(needs_change) >= 0.75 * n and sustained_hours >= dr.ClassificationConfig().min_regime_hours_for_retraining:
        system.append({"response": "global retraining", "reason": f"{len(needs_change)}/{n} stations need structural change in a {sustained_hours:.0f}h regime"})
    elif needs_change:
        system.append({
            "response": "station-level adaptation",
            "reason": f"sustained regime ({sustained_hours:.0f}h) needs changes in {len(needs_change)}/{n} stations only: {', '.join(needs_change)}",
        })
    if len(needs_change) < 0.75 * n or sustained_hours < dr.ClassificationConfig().min_regime_hours_for_retraining:
        system.append({
            "response": "global retraining deferred",
            "reason": f"sustained regime only {sustained_hours:.0f}h old and localized ({len(needs_change)}/{n}); latest regime {latest_hours:.1f}h old",
        })
    return {
        "drift_types": types,
        "classification": "combination" if len(types) > 1 else (types[0] if types else "none"),
        "stations_total": n,
        "sustained_level_up": level_up,
        "sustained_level_down": level_down,
        "sustained_profile_shift": profile,
        "sustained_variance_up": variance,
        "sustained_variance_down": variance_down,
        "shock_affected": shock,
        "sustained_stable": stable,
        "lag_reliability_counts": lag_counts,
        "latest_regime": latest.name if latest else None,
        "latest_regime_hours": latest_hours,
        "latest_regime_too_recent_to_refit": bool(latest_hours < 12),
        "shock_variance_up": shock_variance,
        "sustained_regime": sustained_regime.name,
        "shock_regime": shock_window(obs).name,
        "sustained_regime_hours": sustained_hours,
        "stations_needing_sustained_change": needs_change,
        "stations_needing_shock_guard": shock_guard,
        "system_responses": system,
    }


# --------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------


def write_tables(out: Path, obs: dict[str, Any], pred: dict[str, Any], matrix: pd.DataFrame) -> dict[str, str]:
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def save(name: str, frame: pd.DataFrame) -> None:
        path = tables / f"{name}.csv"
        _local_strings(frame).to_csv(path, index=False)
        written[name] = str(path.relative_to(out))

    save("change_points", obs["segments"].assign(start_local=obs["segments"]["start"].dt.tz_convert(dr.BOGOTA_TZ), end_local=obs["segments"]["end"].dt.tz_convert(dr.BOGOTA_TZ)))
    save("change_point_sensitivity", obs["sensitivity"])
    save("onsets", obs["onsets"])
    save("window_coverage", obs["coverage"])
    save("reference_stability", obs["reference_stability"])
    save("station_window_metrics", obs["station_metrics"])
    save("lag_metrics", obs["lag_table"])
    save("hour_of_day_change", obs["hour_change"])
    save("daily_peaks", obs["peaks"])
    save("new_regime_day_consistency", obs["consistency"])
    save("daily_volatility", obs["daily_volatility"])
    hourly = obs["indices"].join(obs["station_lr"]).reset_index().rename(columns={"index": "hour_utc"})
    save("hourly_log_ratio_indices", hourly)
    save("cycle_metrics_all_models", pred["cycles"])
    save("g_window_overall", pred["overall"])
    save("g_window_by_station", pred["by_station"])
    save("g_window_by_horizon", pred["by_horizon"])
    save("g_window_by_hour", pred["by_hour"])
    save("g_extreme_errors", pred["extremes"])
    save("g_scored_predictions", pred["g"])
    dec = pred["decomposition"]
    if dec.get("status") == "ok":
        save("g_decomposition_windows", dec["windows"])
        save("g_decomposition_cycles", dec["per_cycle"])
        save("g_decomposition_by_station", dec["by_station"])
    for key, comparison in pred["challengers"].items():
        if comparison["status"] != "ok":
            continue
        slug = key.split("|")[0].replace("candidate_", "").split("_")[0] + "_" + key.split("|")[1]
        save(f"challenger_{slug}_per_cycle", comparison["per_cycle"])
        save(f"challenger_{slug}_by_station", comparison["by_station"])
        save(f"challenger_{slug}_by_horizon", comparison["by_horizon"])
    matrix.to_csv(out / "station_diagnostics.csv", index=False)
    written["station_diagnostics"] = "station_diagnostics.csv"
    return written


def build_summary(obs: dict[str, Any], pred: dict[str, Any], matrix: pd.DataFrame, synthesis: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    onsets = obs["onsets"]
    global_onset = onsets[onsets["series"] == "global_intensity"].iloc[0]
    intensity_segments = obs["segments"][obs["segments"]["series"] == "global_intensity"].sort_values("start")
    dec = pred["decomposition"]
    challengers = {
        key: {k: v for k, v in comparison.items() if not isinstance(v, pd.DataFrame)} for key, comparison in pred["challengers"].items()
    }
    return _json(
        {
            "generated_at_utc": pd.Timestamp.now(tz="UTC"),
            "data_source": meta,
            "time_semantics": {
                "analysis_clock": "historical dataset time (observed_at / data_cutoff / target_at), UTC; local = America/Bogota",
                "execution_clock_not_used": ["created_at", "accepted_at", "ingested_at", "first_seen_at"],
                "residual_convention": "residual = actual - prediction (positive = under-prediction)",
                "accuracy": "mean_station_accuracy = leaderboard formula (station WAPE pooled over the window, mean over stations; verified vs /v1/leaderboard); accuracy/global_wape_accuracy = all rows pooled (= leaderboard raw_wape)",
            },
            "observation_range_utc": [obs["prepared"]["observed_at"].min(), obs["anchor"]],
            "reference_window": dr.window_coverage(obs["prepared"], obs["reference"]),
            "windows": obs["coverage"].to_dict("records"),
            "change_points": {
                "global_onset_local": global_onset["onset_local"],
                "global_intensity_segments": [
                    {"start_local": row.start.tz_convert(dr.BOGOTA_TZ), "hours": row.hours, "mean_abs_deviation_ratio": row.ratio}
                    for row in intensity_segments.itertuples()
                ],
                "station_onsets": onsets[~onsets["series"].str.startswith("global")][["series", "onset_local", "first_shift_ratio", "latest_segment_ratio"]].to_dict("records"),
                "detection": {"global": asdict(GLOBAL_CP), "station": asdict(STATION_CP), "baseline": "causal median of same slot 1..4 weeks earlier"},
            },
            "synthesis": synthesis,
            "candidate_g": {
                "model_id": pred["g_model_id"],
                "resolved_official_cycles": int(pred["g"]["cycle_id"].nunique()),
                "unresolved_official_cycles": pred["unresolved_g_cycles"],
                "scoring_parity_with_drift_monitor": pred["parity"],
                "window_overall": pred["overall"].to_dict("records"),
                "by_horizon": pred["by_horizon"].to_dict("records"),
                "decomposition": {k: v for k, v in dec.items() if not isinstance(v, pd.DataFrame)} | (
                    {"windows": dec["windows"].to_dict("records")} if dec.get("status") == "ok" else {}
                ),
            },
            "challengers": challengers,
            "station_matrix": matrix[[c for c in matrix.columns if c in (
                "station_id", "drift_onset_local", "drift_severity", "sustained_severity", "shock_severity", "level_shift", "profile_shift",
                "variance_shift", "lag15_reliability", "lag30_reliability", "lag60_reliability", "lag96_reliability", "weekly_lag_reliability",
                "signed_bias", "relative_bias", "wape", "accuracy", "g_shock_accuracy", "g_last_48h_accuracy", "g_last_6h_accuracy", "recommended_response",
                "sustained_response", "shock_response", "all_responses",
            )]].to_dict("records"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-snapshot", type=Path, help="Read tables from a snapshot directory instead of Supabase.")
    parser.add_argument("--snapshot-dir", type=Path, default=INTERIM_DIR / "snapshot", help="Where to store the Supabase snapshot.")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--g-decomposition", choices=["compute", "cache", "skip"], default="compute", help="Refit the Candidate F base per G cycle (cached).")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.from_snapshot:
        tables = load_snapshot(args.from_snapshot)
        meta = {"source": "snapshot", "path": str(args.from_snapshot)}
        manifest_path = args.from_snapshot / "manifest.json"
        if manifest_path.exists():
            meta["manifest"] = json.loads(manifest_path.read_text())
    else:
        tables = load_from_supabase()
        meta = {"source": "supabase (read-only GET)", "snapshot": str(args.snapshot_dir), "manifest": write_snapshot(tables, args.snapshot_dir)}
    tables = normalize_tables(tables)

    obs = observation_analysis(tables["observations"])
    pred = prediction_analysis(tables, obs, args.g_decomposition)
    matrix = station_matrix(obs, pred)
    synthesis = global_synthesis(obs, pred, matrix)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = write_tables(args.output_dir, obs, pred, matrix)
    figures: dict[str, str] = {}
    if not args.no_figures:
        from drift_regime_figures import render_all

        figures = render_all(args.output_dir / "figures", obs, pred, matrix)
    summary = build_summary(obs, pred, matrix, synthesis, meta)
    summary["outputs"] = {"tables": written, "figures": figures}
    summary["runtime_seconds"] = round(time.perf_counter() - started, 1)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=False, default=str) + "\n")

    from drift_regime_report import render_markdown

    (args.output_dir / "latest.md").write_text(render_markdown(summary, obs, pred, matrix) + "\n")
    print(json.dumps({"classification": synthesis["classification"], "types": synthesis["drift_types"], "runtime_s": summary["runtime_seconds"]}))


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _iso(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _local_strings(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.DatetimeTZDtype):
            frame[column] = frame[column].map(lambda ts: ts.isoformat() if not pd.isna(ts) else None)
    return frame


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NaT:
        return None
    return value


if __name__ == "__main__":
    main()
