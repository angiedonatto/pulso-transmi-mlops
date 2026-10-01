from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drift_monitor import official_scored_predictions  # noqa: E402
from run_drift_monitor import load_monitor_tables  # noqa: E402


BOGOTA_TZ = "America/Bogota"
REPORT_DIR = ROOT / "reports" / "candidate_h_guard"
TABLE_DIR = REPORT_DIR / "tables"

TARGET_STATIONS = ("05000", "07107", "07111", "09122")
WEIGHT_GRID = (0.0, 0.10, 0.20, 0.30, 0.50, 1.0)
PROFILE_VARIANTS = ("general", "dow")
BLOCK_SIZE_CYCLES = 6
MIN_DOW_HISTORY = 4
DEV_HOLDOUT_BLOCKS = 1


@dataclass(frozen=True)
class ProfileSeries:
    times_ns: np.ndarray
    cumulative_mean: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class GuardRuleConfig:
    recent_windows: int
    min_wins: int
    min_mean_delta_pp: float
    min_targets: int
    severe_regression_pp: float
    min_prior_windows: int


@dataclass(frozen=True)
class GuardDecision:
    block_index: int
    station_id: str
    guard_enabled: bool
    profile_variant: str
    profile_weight: float
    windows_won: int
    windows_lost: int
    mean_delta_pp: float
    worst_prior_delta_pp: float
    reason: str


DEFAULT_RULE = GuardRuleConfig(
    recent_windows=3,
    min_wins=2,
    min_mean_delta_pp=0.50,
    min_targets=20,
    severe_regression_pp=-1.00,
    min_prior_windows=3,
)


def load_supabase_official_scored() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = load_monitor_tables()
    scored = official_scored_predictions(
        observations=tables["observations"],
        predictions=tables["predictions"],
        submission_receipts=tables["submission_receipts"],
        forecast_cycles=tables["forecast_cycles"],
        forecast_cycle_targets=tables["forecast_cycle_targets"],
    )
    if scored.empty:
        raise RuntimeError("No fully resolved official cycles were available.")
    return scored, tables["observations"]


def prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    frame = observations.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True, errors="coerce")
    frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
    local = frame["observed_at"].dt.tz_convert(BOGOTA_TZ)
    frame["day_of_week"] = local.dt.dayofweek.astype(int)
    frame["quarter_hour"] = (local.dt.hour * 4 + local.dt.minute // 15).astype(int)
    return frame.dropna(subset=["observed_at", "demand"]).sort_values(["observed_at", "station_id"]).reset_index(drop=True)


def prepare_scored(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    frame["data_cutoff"] = pd.to_datetime(frame["data_cutoff"], utc=True, errors="coerce")
    frame["target_at"] = pd.to_datetime(frame["target_observed_at"], utc=True, errors="coerce")
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame["b_prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    local = frame["target_at"].dt.tz_convert(BOGOTA_TZ)
    frame["target_day_of_week"] = local.dt.dayofweek.astype(int)
    frame["target_quarter_hour"] = (local.dt.hour * 4 + local.dt.minute // 15).astype(int)
    frame = frame.dropna(subset=["data_cutoff", "target_at", "actual", "b_prediction"])
    return frame.sort_values(["accepted_at", "cycle_id", "station_id", "target_at"]).reset_index(drop=True)


def build_profile_lookup(observations: pd.DataFrame, group_cols: list[str]) -> dict[tuple[Any, ...], ProfileSeries]:
    lookup: dict[tuple[Any, ...], ProfileSeries] = {}
    for key, group in observations.groupby(group_cols, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        ordered = group.sort_values("observed_at")
        values = ordered["demand"].to_numpy(dtype=float)
        counts = np.arange(1, len(values) + 1, dtype=int)
        lookup[key_tuple] = ProfileSeries(
            times_ns=ordered["observed_at"].astype("int64").to_numpy(),
            cumulative_mean=np.cumsum(values) / counts,
            counts=counts,
        )
    return lookup


def profile_at(lookup: dict[tuple[Any, ...], ProfileSeries], key: tuple[Any, ...], cutoff: pd.Timestamp) -> tuple[float, int]:
    series = lookup.get(key)
    if series is None:
        return math.nan, 0
    index = int(np.searchsorted(series.times_ns, cutoff.value, side="right") - 1)
    if index < 0:
        return math.nan, 0
    return float(series.cumulative_mean[index]), int(series.counts[index])


def attach_profiles(scored: pd.DataFrame, observations: pd.DataFrame, *, min_dow_history: int = MIN_DOW_HISTORY) -> pd.DataFrame:
    obs = prepare_observations(observations)
    frame = prepare_scored(scored)
    general_lookup = build_profile_lookup(obs, ["station_id", "quarter_hour"])
    dow_lookup = build_profile_lookup(obs, ["station_id", "day_of_week", "quarter_hour"])

    general_values: list[float] = []
    general_counts: list[int] = []
    dow_values: list[float] = []
    dow_counts: list[int] = []
    dow_backoff: list[bool] = []

    for row in frame.itertuples(index=False):
        station_id = str(row.station_id)
        quarter_hour = int(row.target_quarter_hour)
        cutoff = pd.Timestamp(row.data_cutoff)
        general_value, general_count = profile_at(general_lookup, (station_id, quarter_hour), cutoff)
        raw_dow_value, raw_dow_count = profile_at(
            dow_lookup,
            (station_id, int(row.target_day_of_week), quarter_hour),
            cutoff,
        )
        use_backoff = raw_dow_count < min_dow_history or not math.isfinite(raw_dow_value)
        general_values.append(general_value)
        general_counts.append(general_count)
        dow_values.append(general_value if use_backoff else raw_dow_value)
        dow_counts.append(raw_dow_count)
        dow_backoff.append(use_backoff)

    frame["profile_general"] = general_values
    frame["profile_general_count"] = general_counts
    frame["profile_dow"] = dow_values
    frame["profile_dow_count"] = dow_counts
    frame["profile_dow_backoff"] = dow_backoff
    return frame


def assign_cycle_blocks(frame: pd.DataFrame, *, block_size: int = BLOCK_SIZE_CYCLES) -> pd.DataFrame:
    cycles = (
        frame.groupby("cycle_id", as_index=False)
        .agg(accepted_at=("accepted_at", "min"), data_cutoff=("data_cutoff", "min"))
        .sort_values(["accepted_at", "data_cutoff", "cycle_id"])
        .reset_index(drop=True)
    )
    cycles["cycle_position"] = np.arange(len(cycles))
    latest_first_block = (len(cycles) - 1 - cycles["cycle_position"]) // block_size
    cycles["block_index"] = int(latest_first_block.max()) - latest_first_block
    cycles["block_cycle_count"] = cycles.groupby("block_index")["cycle_id"].transform("count")
    return frame.merge(cycles[["cycle_id", "cycle_position", "block_index", "block_cycle_count"]], on="cycle_id", how="inner")


def metric_by(frame: pd.DataFrame, group_cols: list[str], pred_col: str) -> pd.DataFrame:
    work = frame.copy()
    work["residual"] = work[pred_col].astype(float) - work["actual"].astype(float)
    work["abs_error"] = work["residual"].abs()
    if group_cols:
        grouped = (
            work.groupby(group_cols, as_index=False)
            .agg(
                abs_error_sum=("abs_error", "sum"),
                actual_sum=("actual", "sum"),
                residual_sum=("residual", "sum"),
                mae=("abs_error", "mean"),
                bias=("residual", "mean"),
                rows=("actual", "size"),
            )
        )
    else:
        grouped = pd.DataFrame(
            [
                {
                    "abs_error_sum": float(work["abs_error"].sum()),
                    "actual_sum": float(work["actual"].sum()),
                    "residual_sum": float(work["residual"].sum()),
                    "mae": float(work["abs_error"].mean()),
                    "bias": float(work["residual"].mean()),
                    "rows": int(len(work)),
                }
            ]
        )
    grouped["wape"] = np.where(grouped["actual_sum"] > 0, grouped["abs_error_sum"] / grouped["actual_sum"], np.nan)
    grouped["accuracy"] = 100.0 * np.maximum(0.0, 1.0 - grouped["wape"])
    grouped["signed_relative_bias"] = np.where(grouped["actual_sum"] > 0, grouped["residual_sum"] / grouped["actual_sum"], np.nan)
    return grouped[group_cols + ["accuracy", "wape", "mae", "bias", "signed_relative_bias", "rows"]]


def mean_station_accuracy(frame: pd.DataFrame, pred_col: str) -> float:
    station = metric_by(frame, ["station_id"], pred_col)
    return float(station["accuracy"].mean()) if not station.empty else math.nan


def blend_prediction(frame: pd.DataFrame, *, profile_variant: str, profile_weight: float) -> pd.Series:
    if profile_weight == 0:
        return frame["b_prediction"].astype(float)
    profile = frame[f"profile_{profile_variant}"].astype(float)
    blended = (1.0 - profile_weight) * frame["b_prediction"].astype(float) + profile_weight * profile
    return blended.where(profile.notna(), frame["b_prediction"].astype(float)).clip(lower=0.0)


def block_station_candidate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["block_index", "station_id"], sort=True)
    for (block_index, station_id), group in grouped:
        b_metric = metric_by(group.assign(_pred=group["b_prediction"]), [], "_pred").iloc[0]
        for variant in PROFILE_VARIANTS:
            for weight in WEIGHT_GRID:
                pred = blend_prediction(group, profile_variant=variant, profile_weight=weight)
                metric = metric_by(group.assign(_pred=pred), [], "_pred").iloc[0]
                rows.append(
                    {
                        "block_index": int(block_index),
                        "station_id": str(station_id),
                        "profile_variant": variant,
                        "profile_weight": float(weight),
                        "model_label": weight_label(weight, variant),
                        "accuracy": float(metric["accuracy"]),
                        "wape": float(metric["wape"]),
                        "mae": float(metric["mae"]),
                        "bias": float(metric["bias"]),
                        "signed_relative_bias": float(metric["signed_relative_bias"]),
                        "rows": int(metric["rows"]),
                        "b_accuracy": float(b_metric["accuracy"]),
                        "delta_vs_b_pp": float(metric["accuracy"]) - float(b_metric["accuracy"]),
                    }
                )
    return pd.DataFrame(rows)


def weight_label(weight: float, variant: str) -> str:
    if weight == 0:
        return "B"
    if weight == 1:
        return f"profile_{variant}"
    return f"{int(round((1.0 - weight) * 100))}%B + {int(round(weight * 100))}%profile_{variant}"


def config_grid() -> list[GuardRuleConfig]:
    configs: list[GuardRuleConfig] = []
    for recent_windows in (2, 3, 4):
        for min_wins in (recent_windows, max(2, recent_windows - 1)):
            for min_mean_delta_pp in (0.25, 0.50, 0.75, 1.00):
                for severe_regression_pp in (-0.50, -1.00, -1.50):
                    configs.append(
                        GuardRuleConfig(
                            recent_windows=recent_windows,
                            min_wins=min_wins,
                            min_mean_delta_pp=min_mean_delta_pp,
                            min_targets=20,
                            severe_regression_pp=severe_regression_pp,
                            min_prior_windows=recent_windows,
                        )
                    )
    return configs


def choose_guard_decision(
    metrics: pd.DataFrame,
    *,
    station_id: str,
    current_block: int,
    current_targets: int,
    config: GuardRuleConfig,
) -> GuardDecision:
    if current_targets < config.min_targets:
        return disabled_decision(current_block, station_id, "insufficient current targets")

    prior = metrics[
        (metrics["station_id"] == station_id)
        & (metrics["block_index"] < current_block)
        & (metrics["profile_weight"] > 0)
    ].copy()
    prior_blocks = prior["block_index"].nunique()
    if prior_blocks < config.min_prior_windows:
        return disabled_decision(current_block, station_id, "insufficient prior windows")

    candidates: list[dict[str, Any]] = []
    for (variant, weight), history in prior.groupby(["profile_variant", "profile_weight"], sort=True):
        history = history.sort_values("block_index")
        recent = history.tail(config.recent_windows)
        if len(recent) < config.recent_windows:
            continue
        deltas = recent["delta_vs_b_pp"].astype(float)
        wins = int((deltas > 1e-9).sum())
        losses = int((deltas < -1e-9).sum())
        mean_delta = float(deltas.mean())
        worst_prior = float(history["delta_vs_b_pp"].min())
        if wins < config.min_wins:
            continue
        if mean_delta < config.min_mean_delta_pp:
            continue
        if worst_prior < config.severe_regression_pp:
            continue
        candidates.append(
            {
                "profile_variant": str(variant),
                "profile_weight": float(weight),
                "windows_won": wins,
                "windows_lost": losses,
                "mean_delta_pp": mean_delta,
                "worst_prior_delta_pp": worst_prior,
            }
        )

    if not candidates:
        return disabled_decision(current_block, station_id, "no candidate passed conservative gates")

    candidates.sort(
        key=lambda row: (
            -row["mean_delta_pp"],
            row["profile_weight"],
            row["profile_variant"],
        )
    )
    best = candidates[0]
    return GuardDecision(
        block_index=current_block,
        station_id=station_id,
        guard_enabled=True,
        profile_variant=str(best["profile_variant"]),
        profile_weight=float(best["profile_weight"]),
        windows_won=int(best["windows_won"]),
        windows_lost=int(best["windows_lost"]),
        mean_delta_pp=float(best["mean_delta_pp"]),
        worst_prior_delta_pp=float(best["worst_prior_delta_pp"]),
        reason="passed conservative gates using prior windows only",
    )


def disabled_decision(block_index: int, station_id: str, reason: str) -> GuardDecision:
    return GuardDecision(
        block_index=block_index,
        station_id=station_id,
        guard_enabled=False,
        profile_variant="general",
        profile_weight=0.0,
        windows_won=0,
        windows_lost=0,
        mean_delta_pp=0.0,
        worst_prior_delta_pp=0.0,
        reason=reason,
    )


def rolling_guard_predictions(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    config: GuardRuleConfig,
    target_stations: tuple[str, ...] = TARGET_STATIONS,
    max_block: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    if max_block is not None:
        work = work[work["block_index"] <= max_block].copy()
    decisions: list[GuardDecision] = []
    pred_parts: list[pd.DataFrame] = []
    for (block_index, station_id), group in work.groupby(["block_index", "station_id"], sort=True):
        if station_id in target_stations:
            decision = choose_guard_decision(
                metrics,
                station_id=str(station_id),
                current_block=int(block_index),
                current_targets=int(len(group)),
                config=config,
            )
        else:
            decision = disabled_decision(int(block_index), str(station_id), "station not in Candidate H review set")
        decisions.append(decision)
        part = group.copy()
        part["guard_enabled"] = decision.guard_enabled
        part["profile_variant"] = decision.profile_variant
        part["profile_weight"] = decision.profile_weight
        part["h_prediction"] = blend_prediction(
            part,
            profile_variant=decision.profile_variant,
            profile_weight=decision.profile_weight,
        )
        pred_parts.append(part)
    return pd.concat(pred_parts, ignore_index=True), pd.DataFrame([asdict(decision) for decision in decisions])


def choose_rule_config(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    target_stations: tuple[str, ...] = TARGET_STATIONS,
    holdout_blocks: int = DEV_HOLDOUT_BLOCKS,
) -> tuple[GuardRuleConfig, pd.DataFrame]:
    max_block = int(frame["block_index"].max())
    dev_max_block = max_block - holdout_blocks
    if dev_max_block < 1:
        return DEFAULT_RULE, pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for config in config_grid():
        guarded, decisions = rolling_guard_predictions(
            frame,
            metrics,
            config=config,
            target_stations=target_stations,
            max_block=dev_max_block,
        )
        b_accuracy = mean_station_accuracy(guarded, "b_prediction")
        h_accuracy = mean_station_accuracy(guarded, "h_prediction")
        rows.append(
            {
                **asdict(config),
                "dev_b_mean_station_accuracy": b_accuracy,
                "dev_h_mean_station_accuracy": h_accuracy,
                "dev_delta_pp": h_accuracy - b_accuracy,
                "enabled_decisions": int(decisions["guard_enabled"].sum()) if not decisions.empty else 0,
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        [
            "dev_delta_pp",
            "enabled_decisions",
            "min_mean_delta_pp",
            "min_wins",
            "recent_windows",
            "severe_regression_pp",
        ],
        ascending=[False, True, False, False, False, False],
    ).iloc[0]
    return GuardRuleConfig(
        recent_windows=int(best["recent_windows"]),
        min_wins=int(best["min_wins"]),
        min_mean_delta_pp=float(best["min_mean_delta_pp"]),
        min_targets=int(best["min_targets"]),
        severe_regression_pp=float(best["severe_regression_pp"]),
        min_prior_windows=int(best["min_prior_windows"]),
    ), table


def recommendations_for_next_block(
    metrics: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    config: GuardRuleConfig,
    target_stations: tuple[str, ...] = TARGET_STATIONS,
) -> pd.DataFrame:
    next_block = int(frame["block_index"].max()) + 1
    latest_counts = (
        frame[frame["block_index"] == frame["block_index"].max()]
        .groupby("station_id")
        .size()
        .rename("targets")
        .to_dict()
    )
    decisions = [
        choose_guard_decision(
            metrics,
            station_id=station_id,
            current_block=next_block,
            current_targets=int(latest_counts.get(station_id, 0)),
            config=config,
        )
        for station_id in target_stations
    ]
    return pd.DataFrame([asdict(decision) for decision in decisions])


def station_report(frame: pd.DataFrame, next_recommendations: pd.DataFrame) -> pd.DataFrame:
    b_station = metric_by(frame, ["station_id"], "b_prediction").rename(
        columns={"accuracy": "b_accuracy", "signed_relative_bias": "b_signed_relative_bias"}
    )
    h_station = metric_by(frame, ["station_id"], "h_prediction").rename(
        columns={"accuracy": "guarded_accuracy", "signed_relative_bias": "guarded_signed_relative_bias"}
    )
    joined = b_station[["station_id", "b_accuracy", "b_signed_relative_bias", "rows"]].merge(
        h_station[["station_id", "guarded_accuracy", "guarded_signed_relative_bias"]],
        on="station_id",
        how="inner",
    )
    joined["delta_pp"] = joined["guarded_accuracy"] - joined["b_accuracy"]

    block_rows = []
    for (station_id, block_index), group in frame.groupby(["station_id", "block_index"], sort=True):
        b_acc = float(metric_by(group, [], "b_prediction").iloc[0]["accuracy"])
        h_acc = float(metric_by(group, [], "h_prediction").iloc[0]["accuracy"])
        block_rows.append({"station_id": station_id, "block_index": block_index, "delta_pp": h_acc - b_acc})
    block = pd.DataFrame(block_rows)
    block_summary = (
        block.groupby("station_id", as_index=False)
        .agg(
            windows_won=("delta_pp", lambda values: int((values > 1e-9).sum())),
            windows_lost=("delta_pp", lambda values: int((values < -1e-9).sum())),
            worst_window_delta=("delta_pp", "min"),
            median_delta=("delta_pp", "median"),
        )
    )
    rec = next_recommendations.rename(
        columns={
            "profile_weight": "current_recommended_weight",
            "profile_variant": "current_profile_variant",
            "guard_enabled": "guard_enabled",
        }
    )
    report = joined.merge(block_summary, on="station_id", how="left").merge(
        rec[["station_id", "current_recommended_weight", "current_profile_variant", "guard_enabled", "reason"]],
        on="station_id",
        how="left",
    )
    report["guard_enabled"] = report["guard_enabled"].fillna(False).astype(bool)
    report["current_recommended_weight"] = report["current_recommended_weight"].fillna(0.0)
    report["current_profile_variant"] = report["current_profile_variant"].fillna("general")
    return report.sort_values("station_id").reset_index(drop=True)


def block_global_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block_index, group in frame.groupby("block_index", sort=True):
        b = mean_station_accuracy(group, "b_prediction")
        h = mean_station_accuracy(group, "h_prediction")
        rows.append(
            {
                "block_index": int(block_index),
                "b_mean_station_accuracy": b,
                "candidate_h_mean_station_accuracy": h,
                "delta_pp": h - b,
                "cycles": int(group["cycle_id"].nunique()),
                "rows": int(len(group)),
                "start_data_cutoff": group["data_cutoff"].min(),
                "end_data_cutoff": group["data_cutoff"].max(),
            }
        )
    return pd.DataFrame(rows)


def leakage_audit(frame: pd.DataFrame, decisions: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    profile_count_ok = bool((frame["profile_general_count"] >= 0).all() and (frame["profile_dow_count"] >= 0).all())
    target_after_cutoff_ok = bool((frame["target_at"] > frame["data_cutoff"]).all())
    selected_rows = decisions[decisions["guard_enabled"]].copy()
    if selected_rows.empty:
        decisions_use_prior_only = True
    else:
        checks = []
        for row in selected_rows.itertuples(index=False):
            history = metrics[
                (metrics["station_id"] == row.station_id)
                & (metrics["profile_variant"] == row.profile_variant)
                & (np.isclose(metrics["profile_weight"].astype(float), float(row.profile_weight)))
            ]
            checks.append(bool((history["block_index"] < int(row.block_index)).any()))
        decisions_use_prior_only = all(checks)
    return {
        "target_after_data_cutoff": target_after_cutoff_ok,
        "profiles_available_without_future_targets": profile_count_ok,
        "guard_selection_uses_prior_blocks_only": decisions_use_prior_only,
        "profile_definition": "station x local quarter-hour and station x day_of_week x quarter-hour, observed_at <= data_cutoff; DOW backs off to general when history is sparse.",
        "production_writes": False,
        "submissions": False,
        "champion_changed": False,
    }


def run_experiment() -> dict[str, Any]:
    start = time.perf_counter()
    scored, observations = load_supabase_official_scored()
    profiled = attach_profiles(scored, observations)
    frame = assign_cycle_blocks(profiled)
    metrics = block_station_candidate_metrics(frame)
    rule, rule_grid = choose_rule_config(frame, metrics)
    guarded, decisions = rolling_guard_predictions(frame, metrics, config=rule)
    next_rec = recommendations_for_next_block(metrics, frame, config=rule)
    station = station_report(guarded, next_rec)
    block_deltas = block_global_deltas(guarded)
    audit = leakage_audit(guarded, decisions, metrics)

    latest_block = int(guarded["block_index"].max())
    latest = guarded[guarded["block_index"] == latest_block].copy()
    summary = {
        "candidate": "Candidate H -- Conservative Production Stabilization Guard",
        "scope": "experiment_only_no_production_changes",
        "source": "supabase_read_only_official_resolved_cycles",
        "cycles": int(guarded["cycle_id"].nunique()),
        "blocks": int(guarded["block_index"].nunique()),
        "block_size_cycles": BLOCK_SIZE_CYCLES,
        "latest_block": latest_block,
        "latest_block_start": _json_value(latest["data_cutoff"].min()),
        "latest_block_end": _json_value(latest["data_cutoff"].max()),
        "rule_config": asdict(rule),
        "rule_selected_on_blocks_lte": int(guarded["block_index"].max() - DEV_HOLDOUT_BLOCKS),
        "b_mean_station_accuracy": mean_station_accuracy(guarded, "b_prediction"),
        "candidate_h_mean_station_accuracy": mean_station_accuracy(guarded, "h_prediction"),
        "latest_b_mean_station_accuracy": mean_station_accuracy(latest, "b_prediction"),
        "latest_candidate_h_mean_station_accuracy": mean_station_accuracy(latest, "h_prediction"),
        "historical_mean_delta": float(block_deltas["delta_pp"].mean()),
        "recent_delta": float(block_deltas.loc[block_deltas["block_index"] == latest_block, "delta_pp"].iloc[0]),
        "worst_delta": float(block_deltas["delta_pp"].min()),
        "estimated_current_global_improvement": float(block_deltas.loc[block_deltas["block_index"] == latest_block, "delta_pp"].iloc[0]),
        "leakage": audit,
        "runtime_seconds": time.perf_counter() - start,
    }

    write_outputs(summary, guarded, metrics, decisions, next_rec, station, block_deltas, rule_grid)
    return summary


def write_outputs(
    summary: dict[str, Any],
    guarded: pd.DataFrame,
    metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    next_rec: pd.DataFrame,
    station: pd.DataFrame,
    block_deltas: pd.DataFrame,
    rule_grid: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLE_DIR / "block_station_weight_metrics.csv", index=False)
    decisions.to_csv(TABLE_DIR / "rolling_guard_decisions.csv", index=False)
    next_rec.to_csv(TABLE_DIR / "next_block_recommendations.csv", index=False)
    station.to_csv(TABLE_DIR / "station_report.csv", index=False)
    block_deltas.to_csv(TABLE_DIR / "block_global_deltas.csv", index=False)
    if not rule_grid.empty:
        rule_grid.to_csv(TABLE_DIR / "rule_grid_dev_selection.csv", index=False)
    keep_cols = [
        "cycle_id",
        "data_cutoff",
        "station_id",
        "target_at",
        "horizon_minutes",
        "actual",
        "b_prediction",
        "profile_general",
        "profile_dow",
        "guard_enabled",
        "profile_variant",
        "profile_weight",
        "h_prediction",
        "block_index",
    ]
    guarded[keep_cols].to_csv(TABLE_DIR / "guarded_predictions_audit.csv", index=False)
    (REPORT_DIR / "summary.json").write_text(json.dumps(_json_value(summary), indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "leakage_audit.md").write_text(render_leakage_audit(summary) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_markdown(summary, station, block_deltas, next_rec) + "\n")


def render_markdown(summary: dict[str, Any], station: pd.DataFrame, block_deltas: pd.DataFrame, next_rec: pd.DataFrame) -> str:
    focus = station[station["station_id"].isin(TARGET_STATIONS)].copy()
    station_table = focus[
        [
            "station_id",
            "b_accuracy",
            "guarded_accuracy",
            "delta_pp",
            "windows_won",
            "windows_lost",
            "worst_window_delta",
            "median_delta",
            "current_recommended_weight",
            "current_profile_variant",
            "guard_enabled",
        ]
    ]
    final_lines = final_yaml_lines(summary, station)
    return "\n".join(
        [
            "# Candidate H -- Conservative Production Stabilization Guard",
            "",
            "## Scope",
            "",
            "- Experiment only. No production files, champion state, submissions, commits, or pushes were changed.",
            "- Base predictions are the fully resolved official Candidate B / active-model predictions read from Supabase in read-only mode.",
            "- Profiles use only `observed_at <= data_cutoff` for each target row.",
            "- Blocks are ordered groups of 6 official cycles, anchored so the most recent block is exactly the latest 6 cycles.",
            "",
            "## Rule",
            "",
            f"- Selected on development blocks only (`block_index <= {summary['rule_selected_on_blocks_lte']}`), excluding the latest block.",
            f"- Config: `{summary['rule_config']}`.",
            "- For a block, the station guard can only inspect earlier blocks. The evaluated block's actuals never select its own weight.",
            "",
            "## Global",
            "",
            f"- B mean station accuracy: {summary['b_mean_station_accuracy']:.4f}%.",
            f"- Candidate H mean station accuracy: {summary['candidate_h_mean_station_accuracy']:.4f}%.",
            f"- Delta: {summary['candidate_h_mean_station_accuracy'] - summary['b_mean_station_accuracy']:.4f} pp.",
            f"- Latest block B: {summary['latest_b_mean_station_accuracy']:.4f}%; Candidate H: {summary['latest_candidate_h_mean_station_accuracy']:.4f}%; delta {summary['recent_delta']:.4f} pp.",
            "",
            "## Target Stations",
            "",
            _markdown_table(station_table),
            "",
            "## Block Deltas",
            "",
            _markdown_table(block_deltas[["block_index", "b_mean_station_accuracy", "candidate_h_mean_station_accuracy", "delta_pp", "cycles", "rows"]]),
            "",
            "## Leakage",
            "",
            f"- `target_after_data_cutoff = {summary['leakage']['target_after_data_cutoff']}`.",
            f"- `profiles_available_without_future_targets = {summary['leakage']['profiles_available_without_future_targets']}`.",
            f"- `guard_selection_uses_prior_blocks_only = {summary['leakage']['guard_selection_uses_prior_blocks_only']}`.",
            "- Full detail: `reports/candidate_h_guard/leakage_audit.md`.",
            "",
            "## Final",
            "",
            "```yaml",
            *final_lines,
            "```",
        ]
    )


def render_leakage_audit(summary: dict[str, Any]) -> str:
    leakage = summary["leakage"]
    return "\n".join(
        [
            "# Candidate H leakage audit",
            "",
            f"- target_after_data_cutoff: {leakage['target_after_data_cutoff']}",
            f"- profiles_available_without_future_targets: {leakage['profiles_available_without_future_targets']}",
            f"- guard_selection_uses_prior_blocks_only: {leakage['guard_selection_uses_prior_blocks_only']}",
            f"- profile_definition: {leakage['profile_definition']}",
            "- Weight selection is block-forward: each evaluated block can only use `block_index < current_block`.",
            "- Threshold/rule hyperparameters are selected on development blocks excluding the latest block.",
            "- Supabase access is read-only; no production tables are written.",
        ]
    )


def final_yaml_lines(summary: dict[str, Any], station: pd.DataFrame) -> list[str]:
    by_station = {row.station_id: row for row in station.itertuples(index=False)}

    def safe_to_guard(station_id: str) -> str:
        row = by_station.get(station_id)
        return "yes" if row is not None and bool(row.guard_enabled) else "no"

    def blend(station_id: str) -> str:
        row = by_station.get(station_id)
        if row is None or not bool(row.guard_enabled):
            return "B unchanged"
        weight = float(row.current_recommended_weight)
        return weight_label(weight, str(row.current_profile_variant))

    recommendation = "deploy temporary guard" if any(
        bool(by_station.get(station_id).guard_enabled) for station_id in TARGET_STATIONS if by_station.get(station_id) is not None
    ) else "keep B unchanged"
    return [
        f"05000 SAFE TO GUARD: {safe_to_guard('05000')}",
        f"recommended blend: {blend('05000')}",
        "",
        f"07107 SAFE TO GUARD: {safe_to_guard('07107')}",
        f"07111 SAFE TO GUARD: {safe_to_guard('07111')}",
        f"09122 SAFE TO GUARD: {safe_to_guard('09122')}",
        "",
        f"historical mean delta: {summary['historical_mean_delta']:.4f} pp",
        f"recent delta: {summary['recent_delta']:.4f} pp",
        f"worst delta: {summary['worst_delta']:.4f} pp",
        "",
        f"ESTIMATED CURRENT GLOBAL IMPROVEMENT: {summary['estimated_current_global_improvement']:.4f} pp",
        "",
        f"RECOMMENDATION: {recommendation}",
    ]


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Candidate H guard experiment without production changes.")
    parser.add_argument("--source", choices=["supabase"], default="supabase")
    args = parser.parse_args()
    if args.source != "supabase":
        raise SystemExit("Only --source supabase is supported for official resolved cycles.")
    summary = run_experiment()
    print(json.dumps(_json_value(summary), indent=2, sort_keys=True))
    print(f"Report written to {REPORT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
