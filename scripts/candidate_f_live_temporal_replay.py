from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from candidate_b_shadow import (  # noqa: E402
    HORIZONS,
    MIN_HISTORY_DAYS,
    SHORT_LAGS,
    WEEKLY_LAG_MINUTES,
    _prepare_observations,
    _supervised_samples,
    _target_time_features,
)
from candidate_f_metric_aligned import (  # noqa: E402
    BASE_HGB_PARAMS,
    VariantSpec,
    matrix,
    reconstruct_prediction,
    transformed_target,
)
from drift_monitor import official_scored_predictions  # noqa: E402
from forecast_cycle_sync import CycleTarget, ForecastCycle  # noqa: E402
from run_drift_monitor import load_monitor_tables  # noqa: E402


REPORT_DIR = ROOT / "reports" / "candidate_f_live_temporal_replay"
F_REPORT = ROOT / "reports" / "candidate_f_metric_aligned" / "summary.json"
FROZEN_SPEC = VariantSpec(
    "F4_profile_ratio_live_replay",
    feature_representation="ordinal",
    improved_profile=False,
    weighting="NO_WEIGHT",
    target_representation="profile_ratio",
    model_family="hgb",
    params=BASE_HGB_PARAMS,
)
SPECIAL_STATIONS = ("05100", "07111", "05000", "06000", "09122")
EVALUATION_CYCLE_LIMIT = 12


@dataclass(frozen=True)
class ReplayCycle:
    cycle: ForecastCycle
    b_predictions: pd.DataFrame


def station_code_maps(train_samples: pd.DataFrame) -> dict[str, list[Any]]:
    return {
        "station_id": sorted(train_samples["station_id"].astype(str).unique()),
        "horizon_minutes": list(HORIZONS),
        "target_day_of_week": list(range(7)),
        "target_quarter_hour_block": list(range(96)),
    }


def live_target_samples(cycle: ForecastCycle, observations_until_cutoff: pd.DataFrame, actual_lookup: dict[tuple[str, pd.Timestamp], float]) -> pd.DataFrame:
    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    if (observations_until_cutoff["observed_at"] > cutoff).any():
        raise ValueError("observations_until_cutoff contains rows after data_cutoff")
    lookup = {
        (str(row.station_id), pd.Timestamp(row.observed_at)): float(row.demand)
        for row in observations_until_cutoff[["station_id", "observed_at", "demand"]].itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for target in cycle.targets:
        target_at = pd.Timestamp(target.target_at).tz_convert("UTC")
        station_id = str(target.station_id)
        row: dict[str, Any] = {
            "cycle_id": cycle.cycle_id,
            "station_id": station_id,
            "data_cutoff": cutoff,
            "target_at": target_at,
            "target_order": int(target.target_order),
            "horizon_minutes": int(target.horizon_minutes),
            **_target_time_features(target_at),
        }
        for lag in SHORT_LAGS:
            source_at = target_at - pd.Timedelta(minutes=lag)
            row[f"lag_{lag}"] = lookup.get((station_id, source_at), np.nan) if source_at <= cutoff else np.nan
        row["lag_96"] = lookup.get((station_id, target_at - pd.Timedelta(minutes=24 * 60)), np.nan)
        row["lag_weekly"] = lookup.get((station_id, target_at - pd.Timedelta(minutes=WEEKLY_LAG_MINUTES)), np.nan)
        lag_values = [row[f"lag_{lag}"] for lag in SHORT_LAGS if not pd.isna(row[f"lag_{lag}"])]
        row["recent_mean"] = float(np.mean(lag_values)) if lag_values else np.nan
        row["recent_median"] = float(np.median(lag_values)) if lag_values else np.nan
        profile = observations_until_cutoff[
            (observations_until_cutoff["station_id"] == station_id)
            & (observations_until_cutoff["quarter_hour_block"] == row["target_quarter_hour_block"])
            & (observations_until_cutoff["observed_at"] <= cutoff)
        ]["demand"]
        row["profile_mean"] = float(profile.mean()) if not profile.empty else np.nan
        row["profile_median"] = float(profile.median()) if not profile.empty else np.nan
        row["actual"] = actual_lookup.get((station_id, target_at), np.nan)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("target_order").reset_index(drop=True)
    lag_columns = [f"lag_{lag}" for lag in SHORT_LAGS]
    frame["recent_mean"] = frame[lag_columns].mean(axis=1, skipna=True)
    frame["recent_median"] = frame[lag_columns].median(axis=1, skipna=True)
    return frame


def train_f4_profile_ratio_and_predict(
    observations: pd.DataFrame,
    supervised_samples: pd.DataFrame,
    cycle: ForecastCycle,
    actual_lookup: dict[tuple[str, pd.Timestamp], float],
) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor

    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    eligible = observations[observations["observed_at"] <= cutoff].copy()
    train_start = eligible["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
    train_samples = supervised_samples[(supervised_samples["target_at"] <= cutoff) & (supervised_samples["data_cutoff"] >= train_start)].copy()
    train_samples = train_samples.dropna(subset=["profile_mean"])
    if train_samples.empty:
        raise ValueError(f"no F training samples for cycle {cycle.cycle_id}")
    fold_samples = live_target_samples(cycle, eligible, actual_lookup).dropna(subset=["actual"]).copy()
    if len(fold_samples) != cycle.expected_predictions:
        raise ValueError(f"cycle {cycle.cycle_id} expected {cycle.expected_predictions} resolved targets, got {len(fold_samples)}")
    maps = station_code_maps(train_samples)
    model = HistGradientBoostingRegressor(**BASE_HGB_PARAMS)
    model.fit(matrix(train_samples, FROZEN_SPEC, maps), transformed_target(train_samples, FROZEN_SPEC))
    raw = model.predict(matrix(fold_samples, FROZEN_SPEC, maps))
    result = fold_samples.copy()
    result["prediction"] = reconstruct_prediction(raw, result, FROZEN_SPEC)
    result["model_name"] = "candidate_f_f4_profile_ratio_live_replay"
    return result


def official_accuracy(frame: pd.DataFrame) -> float:
    station = metric_by(frame, ["station_id"])
    return float(station["accuracy"].mean())


def metric_by(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    scored = frame.copy()
    scored["abs_error"] = (scored["prediction"].astype(float) - scored["actual"].astype(float)).abs()
    grouped = scored.groupby(group_cols, as_index=False).agg(
        abs_error_sum=("abs_error", "sum"),
        actual_sum=("actual", "sum"),
        mae=("abs_error", "mean"),
        rows=("actual", "size"),
    )
    grouped["wape"] = np.where(grouped["actual_sum"] > 0, grouped["abs_error_sum"] / grouped["actual_sum"], np.nan)
    grouped["accuracy"] = 100.0 * np.maximum(0.0, 1.0 - grouped["wape"])
    return grouped[group_cols + ["accuracy", "wape", "mae", "rows"]]


def build_replay_cycles(tables: dict[str, pd.DataFrame], selection_end: pd.Timestamp) -> list[ReplayCycle]:
    scored = official_scored_predictions(
        predictions=tables["predictions"],
        submission_receipts=tables["submission_receipts"],
        forecast_cycles=tables["forecast_cycles"],
        forecast_cycle_targets=tables["forecast_cycle_targets"],
        observations=tables["observations"],
    )
    scored = scored[pd.to_datetime(scored["data_cutoff"], utc=True) > selection_end].copy()
    if scored.empty:
        return []
    scored["accepted_at"] = pd.to_datetime(scored["accepted_at"], utc=True)
    scored["data_cutoff"] = pd.to_datetime(scored["data_cutoff"], utc=True)
    cycle_order = (
        scored.groupby("cycle_id", as_index=False)
        .agg(accepted_at=("accepted_at", "min"), data_cutoff=("data_cutoff", "min"))
        .sort_values(["accepted_at", "cycle_id"])
    )
    targets = tables["forecast_cycle_targets"].copy()
    targets["target_at"] = pd.to_datetime(targets["target_at"], utc=True)
    cycles = tables["forecast_cycles"].copy()
    cycles["data_cutoff"] = pd.to_datetime(cycles["data_cutoff"], utc=True)
    result: list[ReplayCycle] = []
    for cycle_id in cycle_order["cycle_id"]:
        b_rows = scored[scored["cycle_id"] == cycle_id].copy()
        cycle_row = cycles[cycles["cycle_id"] == cycle_id].iloc[0]
        target_rows = targets[targets["cycle_id"] == cycle_id].sort_values(["target_order", "station_id", "target_at"])
        cycle_targets = [
            CycleTarget(
                station_id=str(row.station_id),
                target_at=pd.Timestamp(row.target_at).isoformat().replace("+00:00", "Z"),
                horizon_minutes=int(row.horizon_minutes),
                target_order=int(row.target_order),
                raw_target={},
            )
            for row in target_rows.itertuples(index=False)
        ]
        cycle = ForecastCycle(
            cycle_id=str(cycle_id),
            data_cutoff=pd.Timestamp(cycle_row.data_cutoff).isoformat().replace("+00:00", "Z"),
            closes_at=str(cycle_row.get("closes_at", "")),
            expected_predictions=int(cycle_row.expected_predictions),
            targets=cycle_targets,
            raw_cycle={},
        )
        if cycle.expected_predictions == 48 and len(cycle.targets) == 48 and len(b_rows) == 48:
            result.append(ReplayCycle(cycle=cycle, b_predictions=b_rows))
    return result


def run_replay() -> dict[str, Any]:
    start = time.perf_counter()
    tables = load_monitor_tables()
    observations = _prepare_observations(tables["observations"])
    actual_lookup = {
        (str(row.station_id), pd.Timestamp(row.observed_at)): float(row.demand)
        for row in observations[["station_id", "observed_at", "demand"]].itertuples(index=False)
    }
    supervised_samples = _supervised_samples(observations).dropna(subset=["profile_mean"])
    f_summary = json.loads(F_REPORT.read_text())
    selection_end = pd.Timestamp(f_summary["data_max"]).tz_convert("UTC")
    replay_cycles = build_replay_cycles(tables, selection_end)
    eligible_cycle_count = len(replay_cycles)
    if len(replay_cycles) > EVALUATION_CYCLE_LIMIT:
        replay_cycles = replay_cycles[-EVALUATION_CYCLE_LIMIT:]
    f_frames: list[pd.DataFrame] = []
    b_frames: list[pd.DataFrame] = []
    cycle_rows: list[dict[str, Any]] = []
    print(
        f"Candidate F live replay: evaluating {len(replay_cycles)} of {eligible_cycle_count} eligible cycles after {selection_end.isoformat()}",
        flush=True,
    )
    for index, replay in enumerate(replay_cycles, start=1):
        cycle_cutoff = pd.Timestamp(replay.cycle.data_cutoff).tz_convert("UTC")
        train_rows = int((supervised_samples["target_at"] <= cycle_cutoff).sum())
        print(f"[{index}/{len(replay_cycles)}] training F4_profile_ratio for cycle={replay.cycle.cycle_id} cutoff={replay.cycle.data_cutoff} rows={train_rows}", flush=True)
        f_pred = train_f4_profile_ratio_and_predict(observations, supervised_samples, replay.cycle, actual_lookup)
        f_pred["cycle_id"] = replay.cycle.cycle_id
        b_pred = replay.b_predictions.rename(columns={"target_observed_at": "target_at"}).copy()
        b_pred["target_at"] = pd.to_datetime(b_pred["target_at"], utc=True)
        b_pred = b_pred[["cycle_id", "station_id", "target_at", "horizon_minutes", "prediction", "actual", "data_cutoff"]].copy()
        f_frames.append(f_pred)
        b_frames.append(b_pred)
        b_acc = official_accuracy(b_pred)
        f_acc = official_accuracy(f_pred)
        cycle_rows.append(
            {
                "cycle_id": replay.cycle.cycle_id,
                "data_cutoff": replay.cycle.data_cutoff,
                "b_accuracy": b_acc,
                "f_accuracy": f_acc,
                "delta_pp": f_acc - b_acc,
            }
        )
    if not cycle_rows:
        payload = {
            "selection_end": selection_end.isoformat().replace("+00:00", "Z"),
            "cycles": [],
            "cycle_count": 0,
            "eligible_cycle_count": eligible_cycle_count,
            "evaluated_cycle_limit": EVALUATION_CYCLE_LIMIT,
            "leakage_check": "passed",
            "tests": "not_run",
            "reason": "no fully resolved official 48-target cycles after Candidate F selection window",
            "runtime_seconds": time.perf_counter() - start,
        }
        write_outputs(payload, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return payload
    cycles_df = pd.DataFrame(cycle_rows)
    f_all = pd.concat(f_frames, ignore_index=True)
    b_all = pd.concat(b_frames, ignore_index=True)
    payload = summarize_replay(
        cycles_df,
        b_all,
        f_all,
        selection_end=selection_end,
        runtime_seconds=time.perf_counter() - start,
        eligible_cycle_count=eligible_cycle_count,
    )
    write_outputs(payload, cycles_df, b_all, f_all, station_comparison(b_all, f_all))
    return payload


def summarize_replay(
    cycles: pd.DataFrame,
    b_all: pd.DataFrame,
    f_all: pd.DataFrame,
    *,
    selection_end: pd.Timestamp,
    runtime_seconds: float,
    eligible_cycle_count: int,
) -> dict[str, Any]:
    recent6 = cycles.tail(6)
    recent12 = cycles.tail(12)
    horizon = compare_group(b_all, f_all, ["horizon_minutes"])
    stations = station_comparison(b_all, f_all)
    return {
        "selection_end": selection_end.isoformat().replace("+00:00", "Z"),
        "cycle_count": int(len(cycles)),
        "eligible_cycle_count": eligible_cycle_count,
        "evaluated_cycle_limit": EVALUATION_CYCLE_LIMIT,
        "cycles": json.loads(cycles.to_json(orient="records")),
        "b_recent_6": float(recent6["b_accuracy"].mean()),
        "f_recent_6": float(recent6["f_accuracy"].mean()),
        "delta_recent_6": float(recent6["f_accuracy"].mean() - recent6["b_accuracy"].mean()),
        "b_recent_12": float(recent12["b_accuracy"].mean()),
        "f_recent_12": float(recent12["f_accuracy"].mean()),
        "delta_recent_12": float(recent12["f_accuracy"].mean() - recent12["b_accuracy"].mean()),
        "cycles_won": int((cycles["delta_pp"] > 0).sum()),
        "cycles_lost": int((cycles["delta_pp"] < 0).sum()),
        "median_delta": float(cycles["delta_pp"].median()),
        "worst_delta": float(cycles["delta_pp"].min()),
        "stations": json.loads(stations.to_json(orient="records")),
        "horizon": json.loads(horizon.to_json(orient="records")),
        "leakage_check": leakage_check(cycles, f_all),
        "tests": "not_run",
        "runtime_seconds": runtime_seconds,
    }


def compare_group(b_all: pd.DataFrame, f_all: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    b = metric_by(b_all, group_cols).rename(columns={"accuracy": "b_accuracy", "wape": "b_wape", "mae": "b_mae"})
    f = metric_by(f_all, group_cols).rename(columns={"accuracy": "f_accuracy", "wape": "f_wape", "mae": "f_mae"})
    result = b.merge(f, on=group_cols, how="inner", suffixes=("_b", "_f"))
    result["delta_pp"] = result["f_accuracy"] - result["b_accuracy"]
    return result


def station_comparison(b_all: pd.DataFrame, f_all: pd.DataFrame) -> pd.DataFrame:
    return compare_group(b_all, f_all, ["station_id"]).sort_values("station_id")


def leakage_check(cycles: pd.DataFrame, f_all: pd.DataFrame) -> str:
    frame = f_all.copy()
    frame["data_cutoff"] = pd.to_datetime(frame["data_cutoff"], utc=True)
    frame["target_at"] = pd.to_datetime(frame["target_at"], utc=True)
    cutoff_consistent = ((frame["target_at"] - frame["data_cutoff"]).dt.total_seconds() / 60 == frame["horizon_minutes"]).all()
    cycle_targets = frame.groupby("cycle_id").size().eq(48).all()
    after_selection = pd.to_datetime(cycles["data_cutoff"], utc=True).is_monotonic_increasing
    return "passed" if bool(cutoff_consistent and cycle_targets and after_selection) else "failed"


def write_outputs(payload: dict[str, Any], cycles: pd.DataFrame, b_all: pd.DataFrame, f_all: pd.DataFrame, stations: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_summary(payload) + "\n")
    if not cycles.empty:
        cycles.to_csv(REPORT_DIR / "cycle_metrics.csv", index=False)
        b_all.to_csv(REPORT_DIR / "b_official_predictions.csv", index=False)
        f_all.to_csv(REPORT_DIR / "f_live_replay_predictions.csv", index=False)
        stations.to_csv(REPORT_DIR / "station_comparison.csv", index=False)
        pd.DataFrame(payload["horizon"]).to_csv(REPORT_DIR / "horizon_comparison.csv", index=False)


def render_summary(payload: dict[str, Any]) -> str:
    if payload.get("cycle_count", 0) == 0:
        return f"# Candidate F Live Temporal Replay\n\nNo eligible cycles after {payload['selection_end']}.\n"
    lines = [
        "# Candidate F Live Temporal Replay",
        "",
        f"- Selection end: {payload['selection_end']}",
        f"- Cycles: {payload['cycle_count']}",
        f"- B recent 6: {payload['b_recent_6']:.4f}",
        f"- F recent 6: {payload['f_recent_6']:.4f}",
        f"- Delta recent 6: {payload['delta_recent_6']:+.4f} pp",
        f"- B recent 12: {payload['b_recent_12']:.4f}",
        f"- F recent 12: {payload['f_recent_12']:.4f}",
        f"- Delta recent 12: {payload['delta_recent_12']:+.4f} pp",
        f"- Cycles won/lost: {payload['cycles_won']}/{payload['cycles_lost']}",
        f"- Leakage check: {payload['leakage_check']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(run_replay(), indent=2, sort_keys=True, default=str))
