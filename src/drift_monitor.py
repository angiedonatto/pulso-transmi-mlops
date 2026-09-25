from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


DriftStatus = Literal["HEALTHY", "WATCH", "DRIFT", "INSUFFICIENT_DATA"]


@dataclass(frozen=True)
class DriftConfig:
    recent_cycles: int = 6
    reference_cycles: int = 12
    min_cycles: int = 6
    min_rows_per_window: int = 50
    min_station_rows: int = 10
    min_psi_rows: int = 80
    min_ks_rows: int = 40
    watch_accuracy_drop_pp: float = 2.0
    drift_accuracy_drop_pp: float = 5.0
    drift_recent_accuracy_floor: float = 78.0
    bias_watch_abs: float = 8.0
    bias_drift_abs: float = 14.0
    relative_bias_watch_abs: float = 0.05
    relative_bias_drift_abs: float = 0.10
    min_residual_persistent_cycles: int = 3
    min_station_horizon_rows: int = 10
    station_accuracy_drop_pp: float = 8.0
    horizon_accuracy_drop_pp: float = 6.0
    median_rel_watch: float = 0.20
    median_rel_drift: float = 0.35
    dispersion_rel_watch: float = 0.35
    dispersion_rel_drift: float = 0.60
    psi_watch: float = 0.10
    psi_drift: float = 0.25
    ks_watch: float = 0.18
    ks_drift: float = 0.30
    min_significant_signals_for_drift: int = 2
    min_new_observations_since_training: int = 500


@dataclass(frozen=True)
class DriftReport:
    status: DriftStatus
    retrain_recommended: bool
    reasons: list[str]
    performance: dict[str, Any]
    data_drift: dict[str, Any]
    residual_drift: dict[str, Any]
    station_alerts: list[dict[str, Any]]
    horizon_alerts: list[dict[str, Any]]
    reference_window: dict[str, Any]
    recent_window: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_drift_report(
    *,
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    submission_receipts: pd.DataFrame,
    forecast_cycles: pd.DataFrame,
    forecast_cycle_targets: pd.DataFrame,
    model_versions: pd.DataFrame | None = None,
    config: DriftConfig | None = None,
) -> DriftReport:
    config = config or DriftConfig()
    scored = official_scored_predictions(
        predictions=predictions,
        submission_receipts=submission_receipts,
        forecast_cycles=forecast_cycles,
        forecast_cycle_targets=forecast_cycle_targets,
        observations=observations,
    )
    performance = performance_drift(scored, config)
    data = data_drift(observations, model_versions=model_versions, config=config)
    residual = residual_drift(scored, config)
    return decide_status(performance, data, residual, config)


def official_scored_predictions(
    *,
    predictions: pd.DataFrame,
    submission_receipts: pd.DataFrame,
    forecast_cycles: pd.DataFrame,
    forecast_cycle_targets: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    if any(frame.empty for frame in (predictions, submission_receipts, forecast_cycles, forecast_cycle_targets, observations)):
        return _empty_scored()

    preds = _normalize_times(predictions.copy(), ["target_observed_at"])
    receipts = _normalize_times(submission_receipts.copy(), ["accepted_at", "created_at"])
    cycles = _normalize_times(forecast_cycles.copy(), ["data_cutoff", "closes_at"])
    targets = _normalize_times(forecast_cycle_targets.copy(), ["target_at"])
    obs = _normalize_times(observations.copy(), ["observed_at"])

    official = receipts[
        (receipts.get("is_official", False) == True)
        & receipts["status"].isin(["accepted", "idempotent_replay"])
    ].copy()
    if official.empty:
        return _empty_scored()

    official = official.sort_values(["cycle_id", "accepted_at", "created_at"]).drop_duplicates("cycle_id", keep="last")
    official = official.merge(cycles, on="cycle_id", how="inner", suffixes=("", "_cycle"))
    expected = targets.groupby("cycle_id").size().rename("target_count").reset_index()
    official = official.merge(expected, on="cycle_id", how="left")
    official = official[official["target_count"] == official["expected_predictions"]]
    if official.empty:
        return _empty_scored()

    keys = ["cycle_id", "model_id"]
    pred_official = preds.merge(
        official[["cycle_id", "model_id", "accepted_at", "data_cutoff", "expected_predictions"]],
        on=keys,
        how="inner",
    )
    pred_official = pred_official.merge(
        targets[["cycle_id", "station_id", "target_at", "horizon_minutes"]],
        left_on=["cycle_id", "station_id", "target_observed_at"],
        right_on=["cycle_id", "station_id", "target_at"],
        how="inner",
        suffixes=("", "_target"),
    )
    pred_official = pred_official.merge(
        obs[["station_id", "observed_at", "demand"]],
        left_on=["station_id", "target_observed_at"],
        right_on=["station_id", "observed_at"],
        how="inner",
    )
    if pred_official.empty:
        return _empty_scored()

    complete_counts = pred_official.groupby("cycle_id").size().rename("resolved_count").reset_index()
    pred_official = pred_official.merge(complete_counts, on="cycle_id", how="left")
    pred_official = pred_official[pred_official["resolved_count"] == pred_official["expected_predictions"]].copy()
    if pred_official.empty:
        return _empty_scored()

    pred_official["actual"] = pred_official["demand"].astype(float)
    pred_official["prediction"] = pred_official["prediction"].astype(float)
    pred_official["residual"] = pred_official["prediction"] - pred_official["actual"]
    pred_official["abs_error"] = pred_official["residual"].abs()
    target_time = pred_official["target_observed_at"].dt.tz_convert("America/Bogota")
    pred_official["target_hour"] = target_time.dt.hour
    return pred_official[
        [
            "cycle_id",
            "model_id",
            "station_id",
            "target_observed_at",
            "horizon_minutes",
            "target_hour",
            "prediction",
            "actual",
            "residual",
            "abs_error",
            "accepted_at",
            "data_cutoff",
        ]
    ].sort_values(["accepted_at", "cycle_id", "station_id", "target_observed_at"]).reset_index(drop=True)


def performance_drift(scored: pd.DataFrame, config: DriftConfig | None = None) -> dict[str, Any]:
    config = config or DriftConfig()
    if scored.empty:
        return {"status": "INSUFFICIENT_DATA", "cycle_count": 0, "reason": "no fully resolved official cycles"}

    cycle_station = _accuracy_by(scored, ["cycle_id", "station_id"])
    cycle_accuracy = (
        cycle_station.groupby("cycle_id", as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "cycle_accuracy"})
        .merge(scored.groupby("cycle_id", as_index=False)["accepted_at"].min(), on="cycle_id", how="left")
        .sort_values(["accepted_at", "cycle_id"])
        .reset_index(drop=True)
    )
    cycle_accuracy["rolling6_accuracy"] = cycle_accuracy["cycle_accuracy"].rolling(6, min_periods=1).mean()
    cycle_accuracy["rolling12_accuracy"] = cycle_accuracy["cycle_accuracy"].rolling(12, min_periods=1).mean()
    cycle_count = int(cycle_accuracy["cycle_id"].nunique())
    if cycle_count < config.min_cycles:
        return {
            "status": "INSUFFICIENT_DATA",
            "cycle_count": cycle_count,
            "cycle_accuracy": _records(cycle_accuracy),
            "reason": f"need at least {config.min_cycles} cycles",
        }

    cycle_order = cycle_accuracy["cycle_id"].tolist()
    recent_ids = cycle_order[-config.recent_cycles :]
    reference_ids = cycle_order[: max(0, len(cycle_order) - config.recent_cycles)]
    if len(reference_ids) > config.reference_cycles:
        reference_ids = reference_ids[-config.reference_cycles :]
    recent = scored[scored["cycle_id"].isin(recent_ids)].copy()
    reference = scored[scored["cycle_id"].isin(reference_ids)].copy()
    recent_acc = float(cycle_accuracy[cycle_accuracy["cycle_id"].isin(recent_ids)]["cycle_accuracy"].mean())
    reference_acc = float(cycle_accuracy[cycle_accuracy["cycle_id"].isin(reference_ids)]["cycle_accuracy"].mean()) if reference_ids else math.nan
    drop = float(reference_acc - recent_acc) if math.isfinite(reference_acc) else 0.0

    station_recent = _accuracy_by(recent, ["station_id"]).rename(columns={"accuracy": "recent_accuracy", "wape": "recent_wape"})
    station_ref = _accuracy_by(reference, ["station_id"]).rename(columns={"accuracy": "reference_accuracy", "wape": "reference_wape"}) if not reference.empty else pd.DataFrame(columns=["station_id", "reference_accuracy", "reference_wape"])
    station = station_recent.merge(station_ref, on="station_id", how="left")
    station["delta_pp"] = station["recent_accuracy"] - station["reference_accuracy"]

    horizon_recent = _accuracy_by(recent, ["horizon_minutes"]).rename(columns={"accuracy": "recent_accuracy", "wape": "recent_wape"})
    horizon_ref = _accuracy_by(reference, ["horizon_minutes"]).rename(columns={"accuracy": "reference_accuracy", "wape": "reference_wape"}) if not reference.empty else pd.DataFrame(columns=["horizon_minutes", "reference_accuracy", "reference_wape"])
    horizon = horizon_recent.merge(horizon_ref, on="horizon_minutes", how="left")
    horizon["delta_pp"] = horizon["recent_accuracy"] - horizon["reference_accuracy"]

    overall = _basic_error_metrics(scored)
    recent_overall = _basic_error_metrics(recent)
    reference_overall = _basic_error_metrics(reference) if not reference.empty else {}
    status = "HEALTHY"
    if drop >= config.drift_accuracy_drop_pp and recent_acc <= config.drift_recent_accuracy_floor:
        status = "DRIFT"
    elif drop >= config.watch_accuracy_drop_pp:
        status = "WATCH"
    return {
        "status": status,
        "cycle_count": cycle_count,
        "cycle_accuracy": _records(cycle_accuracy),
        "recent_cycle_ids": recent_ids,
        "reference_cycle_ids": reference_ids,
        "recent6_accuracy": _last_value(cycle_accuracy["rolling6_accuracy"]),
        "recent12_accuracy": _last_value(cycle_accuracy["rolling12_accuracy"]),
        "recent_accuracy": recent_acc,
        "reference_accuracy": reference_acc,
        "accuracy_drop_pp": drop,
        "overall": overall,
        "recent": recent_overall,
        "reference": reference_overall,
        "station_accuracy": _records(station.sort_values("recent_accuracy")),
        "horizon_accuracy": _records(horizon.sort_values("horizon_minutes")),
    }


def data_drift(
    observations: pd.DataFrame,
    *,
    model_versions: pd.DataFrame | None = None,
    config: DriftConfig | None = None,
) -> dict[str, Any]:
    config = config or DriftConfig()
    if observations.empty:
        return {"status": "INSUFFICIENT_DATA", "reason": "no observations"}

    obs = _prepare_observations(observations)
    if len(obs) < config.min_rows_per_window * 2:
        return {"status": "INSUFFICIENT_DATA", "rows": int(len(obs)), "reason": "not enough observations"}
    times = sorted(obs["observed_at"].drop_duplicates())
    split_idx = max(config.min_rows_per_window, len(times) // 2)
    if len(times) - split_idx < 1:
        return {"status": "INSUFFICIENT_DATA", "rows": int(len(obs)), "reason": "not enough time coverage"}
    reference_end = times[split_idx - 1]
    reference = obs[obs["observed_at"] <= reference_end].copy()
    recent = obs[obs["observed_at"] > reference_end].copy()
    if len(reference) < config.min_rows_per_window or len(recent) < config.min_rows_per_window:
        return {"status": "INSUFFICIENT_DATA", "reference_rows": len(reference), "recent_rows": len(recent)}

    station_rows: list[dict[str, Any]] = []
    alert_rows: list[dict[str, Any]] = []
    for station_id in sorted(set(reference["station_id"]) | set(recent["station_id"])):
        ref_values = reference.loc[reference["station_id"] == station_id, "demand"].astype(float)
        rec_values = recent.loc[recent["station_id"] == station_id, "demand"].astype(float)
        metrics = _distribution_compare(ref_values, rec_values, config)
        metrics["station_id"] = station_id
        station_rows.append(metrics)
        if metrics["sample_status"] != "ok":
            continue
        if _is_distribution_drift(metrics, config):
            alert_rows.append(metrics | {"severity": "drift"})
        elif _is_distribution_watch(metrics, config):
            alert_rows.append(metrics | {"severity": "watch"})

    ref_stations = set(reference["station_id"])
    recent_stations = set(recent["station_id"])
    new_stations = sorted(recent_stations - ref_stations)
    missing_stations = sorted(ref_stations - recent_stations)
    trained_until = _active_trained_until(model_versions)
    new_since_training = int((obs["observed_at"] > trained_until).sum()) if trained_until is not None else None

    drift_alerts = [row for row in alert_rows if row["severity"] == "drift"]
    watch_alerts = [row for row in alert_rows if row["severity"] == "watch"]
    status = "HEALTHY"
    if new_stations or len(drift_alerts) >= 1:
        status = "DRIFT"
    elif missing_stations or watch_alerts:
        status = "WATCH"
    return {
        "status": status,
        "reference": _window_summary(reference),
        "recent": _window_summary(recent),
        "station_metrics": _json_records(station_rows),
        "alerts": _json_records(alert_rows),
        "new_stations": new_stations,
        "missing_stations": missing_stations,
        "new_observations_since_training": new_since_training,
    }


def residual_drift(scored: pd.DataFrame, config: DriftConfig | None = None) -> dict[str, Any]:
    config = config or DriftConfig()
    if scored.empty or scored["cycle_id"].nunique() < config.min_cycles:
        return {"status": "INSUFFICIENT_DATA", "cycle_count": int(scored["cycle_id"].nunique()) if not scored.empty else 0}
    cycle_order = (
        scored.groupby("cycle_id", as_index=False)["accepted_at"]
        .min()
        .sort_values(["accepted_at", "cycle_id"])["cycle_id"]
        .tolist()
    )
    recent_ids = cycle_order[-config.recent_cycles :]
    recent = scored[scored["cycle_id"].isin(recent_ids)].copy()

    by_station = _residual_by(recent, ["station_id"])
    by_horizon = _residual_by(recent, ["horizon_minutes"])
    by_station_horizon = _residual_by(recent, ["station_id", "horizon_minutes"])
    by_station_horizon = by_station_horizon[by_station_horizon["count"] >= config.min_station_horizon_rows].copy()
    by_hour = _residual_by(recent, ["target_hour"])
    by_model = _residual_by(recent, ["model_id"])
    global_residual = _residual_by(recent, [])

    persistence_frames = {
        "global": _attach_relative_bias_persistence(global_residual, recent, [], config),
        "station": _attach_relative_bias_persistence(by_station, recent, ["station_id"], config),
        "horizon": _attach_relative_bias_persistence(by_horizon, recent, ["horizon_minutes"], config),
        "station_horizon": _attach_relative_bias_persistence(by_station_horizon, recent, ["station_id", "horizon_minutes"], config),
        "hour": _attach_relative_bias_persistence(by_hour, recent, ["target_hour"], config),
        "model": _attach_relative_bias_persistence(by_model, recent, ["model_id"], config),
    }
    global_residual = persistence_frames["global"]
    by_station = persistence_frames["station"]
    by_horizon = persistence_frames["horizon"]
    by_station_horizon = persistence_frames["station_horizon"]
    by_hour = persistence_frames["hour"]
    by_model = persistence_frames["model"]

    alerts: list[dict[str, Any]] = []
    for frame, group_name, min_rows in [
        (global_residual, "global", config.min_rows_per_window),
        (by_station, "station", config.min_station_rows),
        (by_horizon, "horizon", config.min_station_rows),
        (by_station_horizon, "station_horizon", config.min_station_horizon_rows),
        (by_hour, "hour", config.min_station_rows),
        (by_model, "model", config.min_station_rows),
    ]:
        for row in frame.to_dict("records"):
            if int(row["count"]) < min_rows:
                continue
            relative_bias = row.get("signed_relative_bias")
            relative_bias_abs = abs(float(relative_bias)) if isinstance(relative_bias, (int, float)) and math.isfinite(float(relative_bias)) else math.nan
            drift_windows = int(row.get("relative_bias_drift_windows", 0))
            watch_windows = int(row.get("relative_bias_watch_windows", 0))
            if (
                math.isfinite(relative_bias_abs)
                and relative_bias_abs >= config.relative_bias_drift_abs
                and drift_windows >= config.min_residual_persistent_cycles
            ):
                alerts.append(row | {"group": group_name, "severity": "drift"})
            elif (
                math.isfinite(relative_bias_abs)
                and relative_bias_abs >= config.relative_bias_watch_abs
                and watch_windows >= config.min_residual_persistent_cycles
            ):
                alerts.append(row | {"group": group_name, "severity": "watch"})
    status = "DRIFT" if any(alert["severity"] == "drift" for alert in alerts) else "WATCH" if alerts else "HEALTHY"
    global_metrics = global_residual.iloc[0].to_dict() if not global_residual.empty else {}
    return {
        "status": status,
        "overall_recent_bias": float(recent["residual"].mean()),
        "overall_recent_signed_relative_bias": _signed_relative_bias(recent),
        "overall_recent_mae": float(recent["abs_error"].mean()),
        "global": _json_value(global_metrics),
        "by_station": _records(by_station),
        "by_horizon": _records(by_horizon),
        "by_station_horizon": _records(by_station_horizon),
        "by_hour": _records(by_hour),
        "by_model": _records(by_model),
        "alerts": _json_records(alerts),
    }


def decide_status(performance: dict[str, Any], data: dict[str, Any], residual: dict[str, Any], config: DriftConfig) -> DriftReport:
    if performance.get("status") == "INSUFFICIENT_DATA":
        return DriftReport(
            status="INSUFFICIENT_DATA",
            retrain_recommended=False,
            reasons=[str(performance.get("reason", "insufficient resolved official cycles"))],
            performance=performance,
            data_drift=data,
            residual_drift=residual,
            station_alerts=[],
            horizon_alerts=[],
            reference_window={},
            recent_window={},
        )

    reasons: list[str] = []
    signals = 0
    watch = False
    if performance.get("status") == "DRIFT":
        signals += 1
        reasons.append(f"performance drop {performance.get('accuracy_drop_pp', 0):.2f}pp")
    elif performance.get("status") == "WATCH":
        watch = True
        reasons.append(f"mild performance drop {performance.get('accuracy_drop_pp', 0):.2f}pp")

    if data.get("status") == "DRIFT":
        signals += 1
        reasons.append("data distribution drift")
    elif data.get("status") == "WATCH":
        watch = True
        reasons.append("data distribution watch")

    if residual.get("status") == "DRIFT":
        signals += 1
        reasons.append("sustained residual bias")
    elif residual.get("status") == "WATCH":
        watch = True
        reasons.append("residual bias watch")

    new_since_training = data.get("new_observations_since_training")
    enough_new_data = isinstance(new_since_training, int) and new_since_training >= config.min_new_observations_since_training
    if enough_new_data:
        reasons.append(f"{new_since_training} observations since active model training")

    status: DriftStatus = "HEALTHY"
    if signals >= config.min_significant_signals_for_drift or (signals >= 1 and performance.get("status") == "DRIFT"):
        status = "DRIFT"
    elif signals or watch:
        status = "WATCH"
    retrain = status == "DRIFT" and (enough_new_data or signals >= config.min_significant_signals_for_drift)

    station_alerts = _station_alerts(performance, residual, config)
    horizon_alerts = _horizon_alerts(performance, residual, config)
    return DriftReport(
        status=status,
        retrain_recommended=bool(retrain),
        reasons=reasons,
        performance=performance,
        data_drift=data,
        residual_drift=residual,
        station_alerts=station_alerts,
        horizon_alerts=horizon_alerts,
        reference_window={"cycle_ids": performance.get("reference_cycle_ids", []), **(data.get("reference", {}) if isinstance(data.get("reference"), dict) else {})},
        recent_window={"cycle_ids": performance.get("recent_cycle_ids", []), **(data.get("recent", {}) if isinstance(data.get("recent"), dict) else {})},
    )


def _accuracy_by(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_cols + ["wape", "accuracy", "mae", "bias", "count"])
    grouped = frame.groupby(group_cols, dropna=False)
    result = grouped.agg(
        abs_error_sum=("abs_error", "sum"),
        actual_sum=("actual", "sum"),
        residual_sum=("residual", "sum"),
        mae=("abs_error", "mean"),
        bias=("residual", "mean"),
        count=("actual", "size"),
    ).reset_index()
    result["wape"] = np.where(result["actual_sum"] > 0, result["abs_error_sum"] / result["actual_sum"], np.nan)
    result["accuracy"] = 100.0 * np.maximum(0.0, 1.0 - result["wape"])
    result["signed_relative_bias"] = np.where(result["actual_sum"] > 0, result["residual_sum"] / result["actual_sum"], np.nan)
    return result[group_cols + ["wape", "accuracy", "mae", "bias", "signed_relative_bias", "count"]]


def _basic_error_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"count": 0}
    actual_sum = float(frame["actual"].sum())
    wape = float(frame["abs_error"].sum() / actual_sum) if actual_sum > 0 else math.nan
    return {
        "count": int(len(frame)),
        "wape": wape,
        "accuracy": float(100.0 * max(0.0, 1.0 - wape)) if math.isfinite(wape) else math.nan,
        "mae": float(frame["abs_error"].mean()),
        "bias": float(frame["residual"].mean()),
        "signed_relative_bias": _signed_relative_bias(frame),
    }


def _residual_by(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "bias",
                "signed_relative_bias",
                "mae",
                "count",
                "residual_sum",
                "actual_sum",
            ]
        )
    if not group_cols:
        residual_sum = float(frame["residual"].sum())
        actual_sum = float(frame["actual"].sum())
        return pd.DataFrame(
            [
                {
                    "bias": float(frame["residual"].mean()),
                    "signed_relative_bias": _safe_divide(residual_sum, actual_sum),
                    "mae": float(frame["abs_error"].mean()),
                    "count": int(len(frame)),
                    "residual_sum": residual_sum,
                    "actual_sum": actual_sum,
                }
            ]
        )
    result = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            bias=("residual", "mean"),
            residual_sum=("residual", "sum"),
            actual_sum=("actual", "sum"),
            mae=("abs_error", "mean"),
            count=("residual", "size"),
        )
        .reset_index()
    )
    result["signed_relative_bias"] = np.where(result["actual_sum"] > 0, result["residual_sum"] / result["actual_sum"], np.nan)
    return result[group_cols + ["bias", "signed_relative_bias", "mae", "count", "residual_sum", "actual_sum"]].sort_values("bias")


def _attach_relative_bias_persistence(
    summary: pd.DataFrame,
    frame: pd.DataFrame,
    group_cols: list[str],
    config: DriftConfig,
) -> pd.DataFrame:
    if summary.empty:
        return summary
    persistence = _relative_bias_persistence(frame, group_cols, config)
    enriched = summary.copy()
    enriched["relative_bias_watch_windows"] = 0
    enriched["relative_bias_drift_windows"] = 0
    enriched["relative_bias_persistent_direction"] = None
    for idx, row in enriched.iterrows():
        key = _group_key(row, group_cols)
        metrics = persistence.get(key, {})
        enriched.at[idx, "relative_bias_watch_windows"] = int(metrics.get("watch_windows", 0))
        enriched.at[idx, "relative_bias_drift_windows"] = int(metrics.get("drift_windows", 0))
        enriched.at[idx, "relative_bias_persistent_direction"] = metrics.get("direction")
    return enriched


def _relative_bias_persistence(frame: pd.DataFrame, group_cols: list[str], config: DriftConfig) -> dict[tuple[Any, ...], dict[str, Any]]:
    if frame.empty:
        return {}
    aggregate_cols = group_cols + ["cycle_id"]
    grouped = (
        frame.groupby(aggregate_cols, dropna=False)
        .agg(residual_sum=("residual", "sum"), actual_sum=("actual", "sum"))
        .reset_index()
    )
    grouped["signed_relative_bias"] = np.where(grouped["actual_sum"] > 0, grouped["residual_sum"] / grouped["actual_sum"], np.nan)
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not group_cols:
        group_values = [((), grouped)]
    else:
        group_values = [(key if isinstance(key, tuple) else (key,), group) for key, group in grouped.groupby(group_cols, dropna=False)]
    for key, group in group_values:
        total_bias = _safe_divide(float(group["residual_sum"].sum()), float(group["actual_sum"].sum()))
        direction = _bias_direction(total_bias)
        if direction is None:
            result[key] = {"watch_windows": 0, "drift_windows": 0, "direction": None}
            continue
        signed = group["signed_relative_bias"].astype(float)
        same_direction = signed.apply(lambda value: _bias_direction(value) == direction)
        watch_windows = int(((signed.abs() >= config.relative_bias_watch_abs) & same_direction).sum())
        drift_windows = int(((signed.abs() >= config.relative_bias_drift_abs) & same_direction).sum())
        result[key] = {
            "watch_windows": watch_windows,
            "drift_windows": drift_windows,
            "direction": direction,
        }
    return result


def _prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_times(observations.copy(), ["observed_at"])
    frame["station_id"] = frame["station_id"].astype(str)
    frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
    local = frame["observed_at"].dt.tz_convert("America/Bogota")
    frame["quarter_hour_block"] = local.dt.hour * 4 + local.dt.minute // 15
    return frame.sort_values(["observed_at", "station_id"]).reset_index(drop=True)


def _distribution_compare(reference: pd.Series, recent: pd.Series, config: DriftConfig) -> dict[str, Any]:
    ref = reference.dropna().astype(float).to_numpy()
    rec = recent.dropna().astype(float).to_numpy()
    metrics = {
        "reference_count": int(len(reference)),
        "recent_count": int(len(recent)),
        "reference_missing_rate": float(reference.isna().mean()) if len(reference) else 1.0,
        "recent_missing_rate": float(recent.isna().mean()) if len(recent) else 1.0,
    }
    if len(ref) < config.min_station_rows or len(rec) < config.min_station_rows:
        return metrics | {"sample_status": "insufficient"}
    ref_stats = _series_stats(ref)
    rec_stats = _series_stats(rec)
    median_rel = _relative_change(rec_stats["median"], ref_stats["median"])
    dispersion_rel = _relative_change(rec_stats["iqr"], ref_stats["iqr"])
    psi = _psi(ref, rec) if min(len(ref), len(rec)) >= config.min_psi_rows else math.nan
    ks = _ks_statistic(ref, rec) if min(len(ref), len(rec)) >= config.min_ks_rows else math.nan
    return metrics | {
        "sample_status": "ok",
        "reference": ref_stats,
        "recent": rec_stats,
        "median_relative_change": median_rel,
        "dispersion_relative_change": dispersion_rel,
        "psi": psi,
        "ks_statistic": ks,
    }


def _series_stats(values: np.ndarray) -> dict[str, float]:
    q10, q25, q50, q75, q90 = np.percentile(values, [10, 25, 50, 75, 90])
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    return {
        "count": float(len(values)),
        "mean": float(np.mean(values)),
        "median": float(q50),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "p10": float(q10),
        "p25": float(q25),
        "p75": float(q75),
        "p90": float(q90),
        "iqr": float(iqr),
        "outlier_rate": float(np.mean((values < lower) | (values > upper))) if iqr > 0 else 0.0,
    }


def _psi(reference: np.ndarray, recent: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    rec_counts, _ = np.histogram(recent, bins=edges)
    epsilon = 1e-6
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), epsilon)
    rec_pct = np.maximum(rec_counts / max(rec_counts.sum(), 1), epsilon)
    return float(np.sum((rec_pct - ref_pct) * np.log(rec_pct / ref_pct)))


def _ks_statistic(reference: np.ndarray, recent: np.ndarray) -> float:
    values = np.sort(np.unique(np.concatenate([reference, recent])))
    ref_sorted = np.sort(reference)
    rec_sorted = np.sort(recent)
    ref_cdf = np.searchsorted(ref_sorted, values, side="right") / len(ref_sorted)
    rec_cdf = np.searchsorted(rec_sorted, values, side="right") / len(rec_sorted)
    return float(np.max(np.abs(ref_cdf - rec_cdf)))


def _is_distribution_drift(metrics: dict[str, Any], config: DriftConfig) -> bool:
    return (
        abs(float(metrics["median_relative_change"])) >= config.median_rel_drift
        or abs(float(metrics["dispersion_relative_change"])) >= config.dispersion_rel_drift
        or _gte(metrics.get("psi"), config.psi_drift)
        or _gte(metrics.get("ks_statistic"), config.ks_drift)
    )


def _is_distribution_watch(metrics: dict[str, Any], config: DriftConfig) -> bool:
    return (
        abs(float(metrics["median_relative_change"])) >= config.median_rel_watch
        or abs(float(metrics["dispersion_relative_change"])) >= config.dispersion_rel_watch
        or _gte(metrics.get("psi"), config.psi_watch)
        or _gte(metrics.get("ks_statistic"), config.ks_watch)
    )


def _station_alerts(performance: dict[str, Any], residual: dict[str, Any], config: DriftConfig) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for row in performance.get("station_accuracy", []):
        delta = row.get("delta_pp")
        if isinstance(delta, (int, float)) and math.isfinite(float(delta)) and float(delta) <= -config.station_accuracy_drop_pp:
            alerts.append(row | {"type": "station_performance_drop"})
    for row in residual.get("alerts", []):
        if row.get("group") == "station":
            alerts.append(row | {"type": "station_bias"})
    return _json_records(alerts)


def _horizon_alerts(performance: dict[str, Any], residual: dict[str, Any], config: DriftConfig) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for row in performance.get("horizon_accuracy", []):
        delta = row.get("delta_pp")
        if isinstance(delta, (int, float)) and math.isfinite(float(delta)) and float(delta) <= -config.horizon_accuracy_drop_pp:
            alerts.append(row | {"type": "horizon_performance_drop"})
    for row in residual.get("alerts", []):
        if row.get("group") == "horizon":
            alerts.append(row | {"type": "horizon_bias"})
    return _json_records(alerts)


def _window_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    return {
        "rows": int(len(frame)),
        "start": _iso(frame["observed_at"].min()),
        "end": _iso(frame["observed_at"].max()),
        "stations": int(frame["station_id"].nunique()),
    }


def _active_trained_until(model_versions: pd.DataFrame | None) -> pd.Timestamp | None:
    if model_versions is None or model_versions.empty or "trained_until" not in model_versions.columns:
        return None
    frame = _normalize_times(model_versions.copy(), ["trained_until"])
    active = frame[frame.get("is_active", False) == True]
    if active.empty:
        return None
    return active["trained_until"].max()


def _normalize_times(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def _empty_scored() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cycle_id",
            "model_id",
            "station_id",
            "target_observed_at",
            "horizon_minutes",
            "target_hour",
            "prediction",
            "actual",
            "residual",
            "abs_error",
            "accepted_at",
            "data_cutoff",
        ]
    )


def _relative_change(recent: float, reference: float) -> float:
    denominator = abs(reference)
    if denominator < 1e-9:
        return 0.0 if abs(recent) < 1e-9 else math.inf
    return float((recent - reference) / denominator)


def _signed_relative_bias(frame: pd.DataFrame) -> float:
    if frame.empty:
        return math.nan
    return _safe_divide(float(frame["residual"].sum()), float(frame["actual"].sum()))


def _safe_divide(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return float(numerator / denominator)


def _bias_direction(value: float) -> str | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or abs(float(value)) < 1e-12:
        return None
    return "positive" if float(value) > 0 else "negative"


def _group_key(row: pd.Series, group_cols: list[str]) -> tuple[Any, ...]:
    return tuple(row[column] for column in group_cols)


def _gte(value: Any, threshold: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= threshold


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _json_records(frame.to_dict("records"))


def _json_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in rows]


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _iso(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _last_value(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = float(series.iloc[-1])
    return value if math.isfinite(value) else None


def _iso(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")
