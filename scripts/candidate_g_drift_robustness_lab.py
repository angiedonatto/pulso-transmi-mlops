from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from candidate_b_shadow import _prepare_observations, _supervised_samples  # noqa: E402
from candidate_f_live_temporal_replay import (  # noqa: E402
    F_REPORT,
    metric_by,
    official_accuracy,
    train_f4_profile_ratio_and_predict,
)
from forecast_cycle_sync import CycleTarget, ForecastCycle  # noqa: E402
from model_improvement_candidates import HORIZONS, load_observations  # noqa: E402
from run_drift_monitor import load_monitor_tables  # noqa: E402


REPORT_DIR = ROOT / "reports" / "candidate_g_drift_robustness_lab"
F_LIVE_DIR = ROOT / "reports" / "candidate_f_live_temporal_replay"
RANDOM_STATE = 17
SYNTHETIC_CUTOFF_COUNT = 8
LIVE_CYCLE_COUNT = 12
TARGET_STATIONS = ("05100", "07111", "06000", "05000", "09122")
STABLE_STATIONS = ("02300", "07105", "07107")


@dataclass(frozen=True)
class GSpec:
    name: str
    use_g1: bool
    use_g2: bool
    use_g3: bool
    g1_window: int = 8
    g2_method: str = "median"
    g3_window: int = 8
    alpha: float = 0.65
    clip_low: float = 0.35
    clip_high: float = 2.5


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    station_targets: dict[str, float]


def normalize_station(value: Any) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def synthetic_specs() -> list[GSpec]:
    specs: list[GSpec] = [GSpec("F", False, False, False)]
    mechanism_sets = [
        ("G1", True, False, False),
        ("G2", False, True, False),
        ("G3", False, False, True),
        ("G1+G2", True, True, False),
        ("G1+G3", True, False, True),
        ("G2+G3", False, True, True),
        ("G1+G2+G3", True, True, True),
    ]
    for name, g1, g2, g3 in mechanism_sets:
        for alpha in (0.40, 0.65, 1.00):
            for clip_low, clip_high in ((0.35, 2.5), (0.5, 2.0)):
                if g1:
                    windows = (4, 8, 12, 16, 24)
                else:
                    windows = (8,)
                if g2:
                    methods = ("median", "ewma")
                else:
                    methods = ("median",)
                if g3:
                    trend_windows = (4, 8, 16, 24)
                else:
                    trend_windows = (8,)
                for window in windows:
                    for method in methods:
                        for trend_window in trend_windows:
                            specs.append(
                                GSpec(
                                    name=name,
                                    use_g1=g1,
                                    use_g2=g2,
                                    use_g3=g3,
                                    g1_window=window,
                                    g2_method=method,
                                    g3_window=trend_window,
                                    alpha=alpha,
                                    clip_low=clip_low,
                                    clip_high=clip_high,
                                )
                            )
    return specs


def build_cycle(cutoff: pd.Timestamp, stations: Iterable[str]) -> ForecastCycle:
    targets: list[CycleTarget] = []
    order = 0
    for station_id in sorted(stations):
        for horizon in HORIZONS:
            target_at = cutoff + pd.Timedelta(minutes=horizon)
            targets.append(
                CycleTarget(
                    station_id=station_id,
                    target_at=target_at.isoformat().replace("+00:00", "Z"),
                    horizon_minutes=int(horizon),
                    target_order=order,
                    raw_target={},
                )
            )
            order += 1
    return ForecastCycle(
        cycle_id=f"synthetic_{cutoff.strftime('%Y%m%dT%H%M%SZ')}",
        data_cutoff=cutoff.isoformat().replace("+00:00", "Z"),
        closes_at="",
        expected_predictions=len(targets),
        targets=targets,
        raw_cycle={},
    )


def historical_f_predictions(observations: pd.DataFrame) -> pd.DataFrame:
    cutoffs = sorted(observations["observed_at"].drop_duplicates())
    latest_cutoff = observations["observed_at"].max() - pd.Timedelta(minutes=max(HORIZONS))
    hourly = [ts for ts in cutoffs if ts.minute == 0 and ts <= latest_cutoff]
    selected = hourly[-SYNTHETIC_CUTOFF_COUNT:]
    actual_lookup = {
        (normalize_station(row.station_id), pd.Timestamp(row.observed_at)): float(row.demand)
        for row in observations[["station_id", "observed_at", "demand"]].itertuples(index=False)
    }
    supervised = _supervised_samples(observations).dropna(subset=["profile_mean"])
    frames = []
    stations = [normalize_station(station_id) for station_id in observations["station_id"].unique()]
    for idx, cutoff in enumerate(selected, start=1):
        print(f"[synthetic F {idx}/{len(selected)}] cutoff={cutoff.isoformat()}", flush=True)
        cycle = build_cycle(pd.Timestamp(cutoff).tz_convert("UTC"), stations)
        pred = train_f4_profile_ratio_and_predict(observations, supervised, cycle, actual_lookup)
        pred["cycle_id"] = cycle.cycle_id
        frames.append(pred)
    return pd.concat(frames, ignore_index=True)


def profile_map(observations: pd.DataFrame, *, end: pd.Timestamp) -> dict[tuple[str, int], float]:
    frame = observations[observations["observed_at"] <= end].copy()
    return {
        (normalize_station(station), int(block)): float(group["demand"].median())
        for (station, block), group in frame.groupby(["station_id", "quarter_hour_block"])
        if not group.empty
    }


def make_scenarios(stations: list[str]) -> list[Scenario]:
    rng = np.random.default_rng(RANDOM_STATE)
    level = {station: float(rng.uniform(0.4, 2.0)) for station in stations}
    up = {station: float(rng.uniform(1.2, 2.0)) for station in stations}
    down = {station: float(rng.uniform(0.4, 0.85)) for station in stations}
    mixed: dict[str, float] = {}
    for station in stations:
        mixed[station] = 1.0 if rng.random() < 0.35 else float(rng.uniform(0.4, 2.0))
    no_drift = {station: 1.0 for station in stations}
    return [
        Scenario("level_shift", "level", level),
        Scenario("upward_trend", "trend", up),
        Scenario("downward_trend", "trend", down),
        Scenario("mixed_station_drift", "level", mixed),
        Scenario("no_drift", "level", no_drift),
    ]


def scenario_factor(station_id: str, observed_at: pd.Timestamp, scenario: Scenario, *, drift_start: pd.Timestamp, drift_end: pd.Timestamp) -> float:
    station = normalize_station(station_id)
    target = float(scenario.station_targets.get(station, 1.0))
    timestamp = pd.Timestamp(observed_at).tz_convert("UTC")
    if timestamp < drift_start:
        return 1.0
    if scenario.kind == "trend":
        denominator = max((drift_end - drift_start).total_seconds(), 1.0)
        progress = min(1.0, max(0.0, (timestamp - drift_start).total_seconds() / denominator))
        return 1.0 + (target - 1.0) * progress
    return target


def adjusted_observations(observations: pd.DataFrame, scenario: Scenario, *, drift_start: pd.Timestamp, drift_end: pd.Timestamp) -> pd.DataFrame:
    frame = observations.copy()
    factors = [
        scenario_factor(row.station_id, row.observed_at, scenario, drift_start=drift_start, drift_end=drift_end)
        for row in frame[["station_id", "observed_at"]].itertuples(index=False)
    ]
    frame["demand"] = frame["demand"].astype(float).to_numpy() * np.asarray(factors, dtype=float)
    return frame


def apply_synthetic_actuals(predictions: pd.DataFrame, scenario: Scenario, *, drift_start: pd.Timestamp, drift_end: pd.Timestamp) -> pd.DataFrame:
    frame = predictions.copy()
    factors = [
        scenario_factor(row.station_id, row.target_at, scenario, drift_start=drift_start, drift_end=drift_end)
        for row in frame[["station_id", "target_at"]].itertuples(index=False)
    ]
    frame["actual"] = frame["actual"].astype(float).to_numpy() * np.asarray(factors, dtype=float)
    return frame


def level_ratio(row: pd.Series, observations: pd.DataFrame, profiles: dict[tuple[str, int], float], window: int) -> float:
    cutoff = pd.Timestamp(row["data_cutoff"]).tz_convert("UTC")
    station = normalize_station(row["station_id"])
    recent = observations[(observations["station_id"] == station) & (observations["observed_at"] <= cutoff)].tail(window)
    ratios = []
    for item in recent.itertuples(index=False):
        expected = profiles.get((station, int(item.quarter_hour_block)))
        if expected and expected > 0:
            ratios.append(float(item.demand) / expected)
    return robust_ratio(ratios)


def feedback_ratio(row: pd.Series, predictions: pd.DataFrame, method: str, window: int = 8) -> float:
    cutoff = pd.Timestamp(row["data_cutoff"]).tz_convert("UTC")
    station = normalize_station(row["station_id"])
    resolved = predictions[(predictions["station_id"] == station) & (predictions["target_at"] <= cutoff)].tail(window)
    ratios = (resolved["actual"].astype(float) / resolved["prediction"].astype(float).clip(lower=1e-3)).replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return 1.0
    if method == "ewma":
        values = ratios.to_numpy(dtype=float)
        weights = np.geomspace(0.2, 1.0, num=len(values))
        return float(np.average(values, weights=weights))
    return float(ratios.median())


def trend_ratio(row: pd.Series, observations: pd.DataFrame, window: int) -> float:
    cutoff = pd.Timestamp(row["data_cutoff"]).tz_convert("UTC")
    station = normalize_station(row["station_id"])
    recent = observations[(observations["station_id"] == station) & (observations["observed_at"] <= cutoff)].tail(window)
    if len(recent) < 3:
        return 1.0
    values = recent["demand"].astype(float).clip(lower=1e-3).to_numpy()
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, np.log(values), 1)[0])
    steps = max(1.0, float(row["horizon_minutes"]) / 15.0)
    return float(np.exp(np.clip(slope * steps, -1.0, 1.0)))


def robust_ratio(values: list[float]) -> float:
    clean = np.asarray([value for value in values if math.isfinite(value) and value > 0], dtype=float)
    if len(clean) == 0:
        return 1.0
    return float(np.median(clean))


def precompute_signals(predictions: pd.DataFrame, observations: pd.DataFrame, profiles: dict[tuple[str, int], float]) -> pd.DataFrame:
    frame = normalize_prediction_frame(predictions)
    obs = observations.copy()
    obs["station_id"] = obs["station_id"].map(normalize_station)
    obs = obs.sort_values(["station_id", "observed_at"]).reset_index(drop=True)
    frame = frame.sort_values(["data_cutoff", "cycle_id", "station_id", "horizon_minutes"]).reset_index(drop=True)
    history = {
        station: group.sort_values("observed_at").reset_index(drop=True)
        for station, group in obs.groupby("station_id", sort=False)
    }
    for window in (4, 8, 12, 16, 24):
        frame[f"g1_w{window}"] = [level_ratio_fast(row, history, profiles, window) for _, row in frame.iterrows()]
    for method in ("median", "ewma"):
        frame[f"g2_{method}"] = [feedback_ratio(row, frame, method) for _, row in frame.iterrows()]
    for window in (4, 8, 16, 24):
        frame[f"g3_w{window}"] = [trend_ratio_fast(row, history, window) for _, row in frame.iterrows()]
    return frame


def recent_history(history: dict[str, pd.DataFrame], station: str, cutoff: pd.Timestamp, window: int) -> pd.DataFrame:
    group = history.get(normalize_station(station))
    if group is None or group.empty:
        return pd.DataFrame()
    times = group["observed_at"].to_numpy(dtype="datetime64[ns]")
    cutoff_np = pd.Timestamp(cutoff).tz_convert("UTC").to_datetime64()
    idx = int(np.searchsorted(times, cutoff_np, side="right"))
    start = max(0, idx - window)
    return group.iloc[start:idx]


def level_ratio_fast(row: pd.Series, history: dict[str, pd.DataFrame], profiles: dict[tuple[str, int], float], window: int) -> float:
    station = normalize_station(row["station_id"])
    recent = recent_history(history, station, pd.Timestamp(row["data_cutoff"]).tz_convert("UTC"), window)
    ratios = []
    for item in recent.itertuples(index=False):
        expected = profiles.get((station, int(item.quarter_hour_block)))
        if expected and expected > 0:
            ratios.append(float(item.demand) / expected)
    return robust_ratio(ratios)


def trend_ratio_fast(row: pd.Series, history: dict[str, pd.DataFrame], window: int) -> float:
    station = normalize_station(row["station_id"])
    recent = recent_history(history, station, pd.Timestamp(row["data_cutoff"]).tz_convert("UTC"), window)
    if len(recent) < 3:
        return 1.0
    values = recent["demand"].astype(float).clip(lower=1e-3).to_numpy()
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, np.log(values), 1)[0])
    steps = max(1.0, float(row["horizon_minutes"]) / 15.0)
    return float(np.exp(np.clip(slope * steps, -1.0, 1.0)))


def normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["station_id"] = result["station_id"].map(normalize_station)
    for col in ("data_cutoff", "target_at"):
        result[col] = pd.to_datetime(result[col], utc=True)
    result["actual"] = result["actual"].astype(float)
    result["prediction"] = result["prediction"].astype(float)
    return result


def calibrate(frame: pd.DataFrame, spec: GSpec) -> pd.DataFrame:
    result = frame.copy()
    if not (spec.use_g1 or spec.use_g2 or spec.use_g3):
        result["prediction_g"] = result["prediction"].astype(float)
        result["calibration_factor"] = 1.0
        return result
    logs: list[pd.Series] = []
    if spec.use_g1:
        logs.append(np.log(result[f"g1_w{spec.g1_window}"].clip(lower=1e-3)))
    if spec.use_g2:
        logs.append(np.log(result[f"g2_{spec.g2_method}"].clip(lower=1e-3)))
    if spec.use_g3:
        logs.append(np.log(result[f"g3_w{spec.g3_window}"].clip(lower=1e-3)))
    raw_log = pd.concat(logs, axis=1).mean(axis=1)
    factor = np.exp(spec.alpha * raw_log).clip(lower=spec.clip_low, upper=spec.clip_high)
    result["calibration_factor"] = factor
    result["prediction_g"] = result["prediction"].astype(float) * factor
    return result


def score_predictions(frame: pd.DataFrame, prediction_col: str) -> float:
    scored = frame.copy()
    scored["prediction"] = scored[prediction_col].astype(float)
    return official_accuracy(scored)


def evaluate_specs(precomputed_by_scenario: dict[str, pd.DataFrame], specs: list[GSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for scenario, frame in precomputed_by_scenario.items():
            pred = calibrate(frame, spec)
            rows.append(
                {
                    "scenario": scenario,
                    "variant": spec.name,
                    "spec": json.dumps(asdict(spec), sort_keys=True),
                    "accuracy": score_predictions(pred, "prediction_g"),
                    "f_accuracy": score_predictions(frame, "prediction"),
                }
            )
    return pd.DataFrame(rows)


def choose_best_spec(scores: pd.DataFrame) -> GSpec:
    grouped = scores.groupby(["variant", "spec"], as_index=False).agg(accuracy=("accuracy", "mean"), f_accuracy=("f_accuracy", "mean"))
    grouped["delta_pp"] = grouped["accuracy"] - grouped["f_accuracy"]
    best = grouped.sort_values(["accuracy", "delta_pp"], ascending=False).iloc[0]
    return GSpec(**json.loads(str(best["spec"])))


def compare_station(b: pd.DataFrame, f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    b_metric = metric_by(b, ["station_id"]).rename(columns={"accuracy": "b_accuracy"})
    f_metric = metric_by(f.assign(prediction=f["prediction"].astype(float)), ["station_id"]).rename(columns={"accuracy": "f_accuracy"})
    g_metric = metric_by(g.assign(prediction=g["prediction_g"].astype(float)), ["station_id"]).rename(columns={"accuracy": "g_accuracy"})
    return b_metric[["station_id", "b_accuracy"]].merge(f_metric[["station_id", "f_accuracy"]], on="station_id").merge(g_metric[["station_id", "g_accuracy"]], on="station_id")


def compare_horizon(f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    f_metric = metric_by(f, ["horizon_minutes"]).rename(columns={"accuracy": "f_accuracy"})
    g_metric = metric_by(g.assign(prediction=g["prediction_g"].astype(float)), ["horizon_minutes"]).rename(columns={"accuracy": "g_accuracy"})
    result = f_metric[["horizon_minutes", "f_accuracy"]].merge(g_metric[["horizon_minutes", "g_accuracy"]], on="horizon_minutes")
    result["delta_pp"] = result["g_accuracy"] - result["f_accuracy"]
    return result


def cycle_metrics_all(b: pd.DataFrame, f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle_id, f_cycle in f.groupby("cycle_id"):
        b_cycle = b[b["cycle_id"] == cycle_id]
        g_cycle = g[g["cycle_id"] == cycle_id]
        b_acc = score_predictions(b_cycle, "prediction")
        f_acc = score_predictions(f_cycle, "prediction")
        g_acc = score_predictions(g_cycle, "prediction_g")
        rows.append({"cycle_id": cycle_id, "b_accuracy": b_acc, "f_accuracy": f_acc, "g_accuracy": g_acc, "delta_pp": g_acc - f_acc})
    return pd.DataFrame(rows)


def run_lab() -> dict[str, Any]:
    start = time.perf_counter()
    local_observations = load_observations()
    local_observations["station_id"] = local_observations["station_id"].map(normalize_station)
    hist_f = historical_f_predictions(local_observations)
    hist_f = normalize_prediction_frame(hist_f)
    drift_end = hist_f["target_at"].max()
    drift_start = hist_f["data_cutoff"].min()
    profile_end = drift_start - pd.Timedelta(minutes=15)
    scenarios = make_scenarios(sorted(hist_f["station_id"].unique()))
    scenario_frames: dict[str, pd.DataFrame] = {}
    for scenario in scenarios:
        synthetic_obs = adjusted_observations(local_observations, scenario, drift_start=drift_start, drift_end=drift_end)
        synthetic_preds = apply_synthetic_actuals(hist_f, scenario, drift_start=drift_start, drift_end=drift_end)
        profiles = profile_map(local_observations, end=profile_end)
        scenario_frames[scenario.name] = precompute_signals(synthetic_preds, synthetic_obs, profiles)
    specs = synthetic_specs()
    synthetic_scores = evaluate_specs(scenario_frames, specs)
    best_spec = choose_best_spec(synthetic_scores)

    live_f = normalize_prediction_frame(pd.read_csv(F_LIVE_DIR / "f_live_replay_predictions.csv", dtype={"station_id": "string"}))
    live_b = normalize_prediction_frame(pd.read_csv(F_LIVE_DIR / "b_official_predictions.csv", dtype={"station_id": "string"}))
    tables = load_monitor_tables()
    live_observations = _prepare_observations(tables["observations"])
    live_observations["station_id"] = live_observations["station_id"].map(normalize_station)
    selection_end = pd.Timestamp(json.loads(F_REPORT.read_text())["data_max"]).tz_convert("UTC")
    live_profiles = profile_map(live_observations, end=selection_end)
    live_precomputed = precompute_signals(live_f.tail(LIVE_CYCLE_COUNT * 48), live_observations, live_profiles)
    live_g = calibrate(live_precomputed, best_spec)
    live_b = live_b[live_b["cycle_id"].isin(live_g["cycle_id"].unique())].copy()
    live_f = live_f[live_f["cycle_id"].isin(live_g["cycle_id"].unique())].copy()

    cycle_df = cycle_metrics_all(live_b, live_f, live_g)
    station_df = compare_station(live_b, live_f, live_g)
    horizon_df = compare_horizon(live_f, live_g)
    scenario_summary = (
        synthetic_scores.groupby(["scenario", "variant"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), f_accuracy=("f_accuracy", "mean"))
        .assign(delta_pp=lambda df: df["accuracy"] - df["f_accuracy"])
    )
    best_synth = synthetic_scores[synthetic_scores["spec"] == json.dumps(asdict(best_spec), sort_keys=True)]
    best_synth_summary = best_synth.groupby("scenario", as_index=False).agg(accuracy=("accuracy", "mean"), f_accuracy=("f_accuracy", "mean"))
    best_synth_summary["delta_pp"] = best_synth_summary["accuracy"] - best_synth_summary["f_accuracy"]

    recent6_cycles = sorted(live_g["cycle_id"].unique())[-6:]
    recent12_cycles = sorted(live_g["cycle_id"].unique())[-12:]
    payload = {
        "best_spec": asdict(best_spec),
        "synthetic_drift_f": float(best_synth["f_accuracy"].mean()),
        "synthetic_drift_g": float(best_synth["accuracy"].mean()),
        "synthetic_delta": float(best_synth["accuracy"].mean() - best_synth["f_accuracy"].mean()),
        "level_shift_delta": scenario_delta(best_synth_summary, "level_shift"),
        "trend_drift_delta": float(best_synth_summary[best_synth_summary["scenario"].isin(["upward_trend", "downward_trend"])]["delta_pp"].mean()),
        "no_drift_delta": scenario_delta(best_synth_summary, "no_drift"),
        "b_recent_12": float(cycle_df[cycle_df["cycle_id"].isin(recent12_cycles)]["b_accuracy"].mean()),
        "f_recent_12": float(cycle_df[cycle_df["cycle_id"].isin(recent12_cycles)]["f_accuracy"].mean()),
        "g_recent_12": float(cycle_df[cycle_df["cycle_id"].isin(recent12_cycles)]["g_accuracy"].mean()),
        "b_recent_6": float(cycle_df[cycle_df["cycle_id"].isin(recent6_cycles)]["b_accuracy"].mean()),
        "f_recent_6": float(cycle_df[cycle_df["cycle_id"].isin(recent6_cycles)]["f_accuracy"].mean()),
        "g_recent_6": float(cycle_df[cycle_df["cycle_id"].isin(recent6_cycles)]["g_accuracy"].mean()),
        "cycles_won_vs_f": int((cycle_df["delta_pp"] > 0).sum()),
        "cycles_lost_vs_f": int((cycle_df["delta_pp"] < 0).sum()),
        "median_delta": float(cycle_df["delta_pp"].median()),
        "worst_cycle_delta": float(cycle_df["delta_pp"].min()),
        "station_metrics": json.loads(station_df.to_json(orient="records")),
        "horizon_metrics": json.loads(horizon_df.to_json(orient="records")),
        "stable_stations_max_regression": stable_max_regression(station_df),
        "leakage_check": leakage_check(hist_f, live_g, selection_end),
        "tests": "not_run",
        "runtime_seconds": time.perf_counter() - start,
    }
    write_outputs(payload, synthetic_scores, scenario_summary, best_synth_summary, cycle_df, station_df, horizon_df, live_g)
    return payload


def scenario_delta(frame: pd.DataFrame, scenario: str) -> float:
    row = frame[frame["scenario"] == scenario]
    return float(row["delta_pp"].iloc[0]) if not row.empty else math.nan


def stable_max_regression(station_df: pd.DataFrame) -> float:
    stable = station_df[station_df["station_id"].isin(STABLE_STATIONS)].copy()
    if stable.empty:
        return 0.0
    stable["delta"] = stable["g_accuracy"] - stable["f_accuracy"]
    return float(min(0.0, stable["delta"].min()))


def leakage_check(hist_f: pd.DataFrame, live_g: pd.DataFrame, selection_end: pd.Timestamp) -> str:
    hist_ok = bool((hist_f["data_cutoff"] <= selection_end).all())
    live_ok = bool((live_g["data_cutoff"] > selection_end).all())
    target_relation = bool(((live_g["target_at"] - live_g["data_cutoff"]).dt.total_seconds() / 60 == live_g["horizon_minutes"]).all())
    return "passed" if hist_ok and live_ok and target_relation else "failed"


def write_outputs(
    payload: dict[str, Any],
    synthetic_scores: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    best_synth_summary: pd.DataFrame,
    cycle_df: pd.DataFrame,
    station_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    live_g: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_summary(payload) + "\n")
    synthetic_scores.to_csv(REPORT_DIR / "synthetic_scores.csv", index=False)
    scenario_summary.to_csv(REPORT_DIR / "synthetic_scenario_summary.csv", index=False)
    best_synth_summary.to_csv(REPORT_DIR / "best_synthetic_scenario_summary.csv", index=False)
    cycle_df.to_csv(REPORT_DIR / "live_cycle_metrics.csv", index=False)
    station_df.to_csv(REPORT_DIR / "live_station_metrics.csv", index=False)
    horizon_df.to_csv(REPORT_DIR / "live_horizon_metrics.csv", index=False)
    live_g.to_csv(REPORT_DIR / "live_g_predictions.csv", index=False)


def render_summary(payload: dict[str, Any]) -> str:
    spec = payload["best_spec"]
    return "\n".join(
        [
            "# Candidate G - Drift Robustness Lab",
            "",
            f"- Best variant: {spec['name']}",
            f"- Fast/level window: {spec['g1_window']}",
            f"- Trend window: {spec['g3_window']}",
            f"- Alpha: {spec['alpha']}",
            f"- Clip: [{spec['clip_low']}, {spec['clip_high']}]",
            f"- Synthetic F/G: {payload['synthetic_drift_f']:.4f} / {payload['synthetic_drift_g']:.4f}",
            f"- Live12 B/F/G: {payload['b_recent_12']:.4f} / {payload['f_recent_12']:.4f} / {payload['g_recent_12']:.4f}",
            f"- Live6 B/F/G: {payload['b_recent_6']:.4f} / {payload['f_recent_6']:.4f} / {payload['g_recent_6']:.4f}",
            f"- Leakage: {payload['leakage_check']}",
        ]
    )


if __name__ == "__main__":
    print(json.dumps(run_lab(), indent=2, sort_keys=True, default=str))
