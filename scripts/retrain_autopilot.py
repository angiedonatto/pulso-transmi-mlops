"""Accuracy-drop autopilot: diagnose drift and run safe remediation.

This script is intentionally conservative. It can write monitoring snapshots and
shadow predictions, but it never changes `model_versions.is_active` and never
submits to the competition API.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from candidate_i_shadow import SupabaseCandidateIShadowStore, run_candidate_i_shadow  # noqa: E402
from drift_monitor import LEADERBOARD_ACCURACY_METRIC, _accuracy_by, accuracy_summary, official_scored_predictions  # noqa: E402
from run_monitoring_snapshot import SupabaseMonitoringStore  # noqa: E402


REPORT_DIR = ROOT / "reports" / "retrain_autopilot"
DEFAULT_THRESHOLD = 82.0


@dataclass(frozen=True)
class CandidateScore:
    """Challenger vs champion on the SAME resolved cycles. `*_accuracy` = leaderboard formula
    (mean_station_accuracy); `*_global_wape_accuracy` = pooled WAPE (leaderboard raw_wape)."""

    model_id: str
    model_name: str
    model_type: str
    cycles: int
    recent_accuracy: float | None
    recent6_accuracy: float | None
    recent12_accuracy: float | None
    delta_vs_champion_recent: float | None
    worst_station: str | None
    worst_station_accuracy: float | None
    recent_global_wape_accuracy: float | None = None
    champion_recent_accuracy: float | None = None
    champion_recent_global_wape_accuracy: float | None = None
    delta_vs_champion_global_wape_pp: float | None = None
    accuracy_metric: str = LEADERBOARD_ACCURACY_METRIC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Trigger remediation below this recent accuracy.")
    parser.add_argument("--recent-cycles", type=int, default=6, help="Recent complete cycles used for trigger and candidate comparison.")
    parser.add_argument("--shadow-time-budget", type=float, default=300.0, help="Seconds allowed for Candidate I shadow inference.")
    parser.add_argument("--no-persist-snapshot", action="store_true", help="Do not write a monitoring snapshot row.")
    parser.add_argument("--no-shadow", action="store_true", help="Diagnose only; do not run Candidate I shadow inference.")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with SupabaseMonitoringStore(timeout=45.0) as store:
        tables = store.load_tables()
        snapshot = __import__("monitoring_snapshot").build_monitoring_snapshot(**tables)
        row = snapshot.row()
        if not args.no_persist_snapshot:
            store.persist_snapshot(row)

    drift = row["report"]["drift"]
    performance = drift.get("performance", {})
    trigger_accuracy = _trigger_accuracy(performance)
    accuracy_view = accuracy_metrics_view(performance)
    should_remediate = bool(
        trigger_accuracy is not None
        and trigger_accuracy < args.threshold
        or row.get("retrain_recommended")
    )
    diagnosis = diagnose_drift(drift, threshold=args.threshold, trigger_accuracy=trigger_accuracy)

    shadow_result: dict[str, Any] | None = None
    if should_remediate and not args.no_shadow:
        shadow_result = run_shadow(time_budget_s=args.shadow_time_budget)

    candidate_scores = score_shadow_candidates(tables, recent_cycles=args.recent_cycles)
    recommendation = recommend_action(
        trigger_accuracy=trigger_accuracy,
        threshold=args.threshold,
        drift=drift,
        diagnosis=diagnosis,
        candidate_scores=candidate_scores,
    )
    payload = {
        "generated_at": row["generated_at"],
        "threshold": args.threshold,
        "trigger_metric": LEADERBOARD_ACCURACY_METRIC,
        "trigger_accuracy": trigger_accuracy,
        "accuracy_metrics": accuracy_view,
        "should_remediate": should_remediate,
        "snapshot_status": {
            "overall_status": row["overall_status"],
            "drift_status": row["drift_status"],
            "retrain_recommended": row["retrain_recommended"],
            "model_id": row["model_id"],
        },
        "diagnosis": diagnosis,
        "shadow_result": shadow_result,
        "candidate_scores": [score.__dict__ for score in candidate_scores],
        "recommendation": recommendation,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (args.output_dir / "latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    (args.output_dir / "latest.md").write_text(render_markdown(payload) + "\n")
    print(json.dumps(_console_summary(payload), sort_keys=True, default=str))


def accuracy_metrics_view(performance: dict[str, Any]) -> dict[str, Any]:
    """Both accuracy families side by side so the trigger is never confused with the pooled WAPE."""
    leaderboard = performance.get("leaderboard_rolling_24h") or {}
    return {
        "trigger_metric": LEADERBOARD_ACCURACY_METRIC,
        "recent6_mean_station_accuracy": performance.get("recent6_accuracy"),
        "recent6_global_wape_accuracy": performance.get("recent6_global_wape_accuracy"),
        "recent6_mean_cycle_accuracy_legacy": performance.get("recent6_mean_cycle_accuracy"),
        "recent12_mean_station_accuracy": performance.get("recent12_accuracy"),
        "recent12_global_wape_accuracy": performance.get("recent12_global_wape_accuracy"),
        "leaderboard_rolling_24h_accuracy": leaderboard.get("mean_station_accuracy"),
        "leaderboard_rolling_24h_raw_wape": leaderboard.get("global_wape"),
        "leaderboard_rolling_24h_coverage": leaderboard.get("coverage"),
        "leaderboard_rolling_24h_missing_cutoffs": leaderboard.get("missing_cutoffs", []),
    }


def _trigger_accuracy(performance: dict[str, Any]) -> float | None:
    """Leaderboard formula (mean_station_accuracy over the last 6 cycles); never the pooled WAPE."""
    for key in ("recent6_accuracy", "recent_accuracy", "recent12_accuracy"):
        value = performance.get(key)
        if isinstance(value, (int, float)) and pd.notna(value):
            return float(value)
    return None


def diagnose_drift(drift: dict[str, Any], *, threshold: float, trigger_accuracy: float | None) -> dict[str, Any]:
    performance = drift.get("performance", {})
    data = drift.get("data_drift", {})
    residual = drift.get("residual_drift", {})
    station_alerts = drift.get("station_alerts", [])
    horizon_alerts = drift.get("horizon_alerts", [])

    drift_types: list[str] = []
    if trigger_accuracy is not None and trigger_accuracy < threshold:
        drift_types.append("accuracy_floor_breach")
    if performance.get("accuracy_drop_pp", 0) >= 2:
        drift_types.append("performance_drop")
    if data.get("status") in {"WATCH", "DRIFT"}:
        drift_types.append("data_distribution")
    if residual.get("status") in {"WATCH", "DRIFT"}:
        drift_types.append("residual_bias")
    if station_alerts:
        drift_types.append("station_localized")
    if horizon_alerts:
        drift_types.append("horizon_localized")

    actions: list[str] = []
    if "residual_bias" in drift_types:
        actions.append("run adaptive residual correction shadow and compare against champion")
    if "data_distribution" in drift_types:
        actions.append("prefer recent-data refit / adaptive calibration over static champion")
    if "station_localized" in drift_types:
        actions.append("inspect station-level guard or station-specific fallback before promotion")
    if "horizon_localized" in drift_types:
        actions.append("inspect horizon-level error before promotion")
    if not actions:
        actions.append("continue monitoring; no remediation trigger")

    return {
        "types": drift_types or ["none"],
        "reasons": drift.get("reasons", []),
        "recent_accuracy": trigger_accuracy,
        "reference_accuracy": performance.get("reference_accuracy"),
        "accuracy_drop_pp": performance.get("accuracy_drop_pp"),
        "data_status": data.get("status"),
        "residual_status": residual.get("status"),
        "station_alerts": station_alerts[:8],
        "horizon_alerts": horizon_alerts[:8],
        "recommended_remediation": actions,
    }


def run_shadow(*, time_budget_s: float) -> dict[str, Any]:
    try:
        with SupabaseCandidateIShadowStore() as store:
            result = run_candidate_i_shadow(store, time_budget_s=time_budget_s)
        return result.to_dict()
    except Exception as exc:  # noqa: BLE001 - autopilot report should survive remediation errors
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def score_shadow_candidates(tables: dict[str, pd.DataFrame], *, recent_cycles: int) -> list[CandidateScore]:
    active_scored = official_scored_predictions(
        observations=tables["observations"],
        predictions=tables["predictions"],
        submission_receipts=tables["submission_receipts"],
        forecast_cycles=tables["forecast_cycles"],
        forecast_cycle_targets=tables["forecast_cycle_targets"],
    )
    if active_scored.empty:
        return []

    cycle_order = (
        active_scored.groupby("cycle_id", as_index=False)["accepted_at"]
        .min()
        .sort_values(["accepted_at", "cycle_id"])["cycle_id"]
        .tolist()
    )
    recent_ids = cycle_order[-recent_cycles:]
    candidate_rows = _resolved_model_predictions(tables, cycle_ids=recent_ids)
    if candidate_rows.empty:
        return []

    models = tables["model_versions"].copy()
    active_model_id = str(active_scored["model_id"].iloc[-1])
    scores: list[CandidateScore] = []
    for model_id, frame in candidate_rows.groupby("model_id"):
        model_id = str(model_id)
        if model_id == active_model_id:
            continue
        # same targets only: the champion is scored on the cycles this challenger also resolved
        champion = accuracy_summary(active_scored[active_scored["cycle_id"].isin(set(frame["cycle_id"]))])
        challenger = accuracy_summary(frame)
        model = models[models["model_id"].astype(str) == model_id]
        if model.empty:
            model_name = model_type = "unknown"
        else:
            model_name = str(model.iloc[0].get("model_name"))
            model_type = str(model.iloc[0].get("model_type"))
        recent_accuracy = _finite(challenger["mean_station_accuracy"])
        champion_recent = _finite(champion["mean_station_accuracy"])
        station = _accuracy_by(frame, ["station_id"]).sort_values("accuracy")
        worst = station.iloc[0] if not station.empty else None
        scores.append(
            CandidateScore(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                cycles=int(frame["cycle_id"].nunique()),
                recent_accuracy=recent_accuracy,
                recent6_accuracy=recent_accuracy,
                recent12_accuracy=None,
                delta_vs_champion_recent=recent_accuracy - champion_recent if recent_accuracy is not None and champion_recent is not None else None,
                worst_station=str(worst["station_id"]) if worst is not None else None,
                worst_station_accuracy=float(worst["accuracy"]) if worst is not None else None,
                recent_global_wape_accuracy=_finite(challenger["global_wape_accuracy"]),
                champion_recent_accuracy=champion_recent,
                champion_recent_global_wape_accuracy=_finite(champion["global_wape_accuracy"]),
                delta_vs_champion_global_wape_pp=_delta(challenger["global_wape_accuracy"], champion["global_wape_accuracy"]),
            )
        )
    return sorted(scores, key=lambda row: row.delta_vs_champion_recent if row.delta_vs_champion_recent is not None else -999, reverse=True)


def _resolved_model_predictions(tables: dict[str, pd.DataFrame], *, cycle_ids: list[str]) -> pd.DataFrame:
    predictions = tables["predictions"].copy()
    targets = tables["forecast_cycle_targets"].copy()
    observations = tables["observations"].copy()
    if predictions.empty or targets.empty or observations.empty:
        return pd.DataFrame()
    predictions["target_observed_at"] = pd.to_datetime(predictions["target_observed_at"], utc=True)
    targets["target_at"] = pd.to_datetime(targets["target_at"], utc=True)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    frame = predictions[predictions["cycle_id"].astype(str).isin(cycle_ids)].merge(
        targets[["cycle_id", "station_id", "target_at", "horizon_minutes"]],
        left_on=["cycle_id", "station_id", "target_observed_at"],
        right_on=["cycle_id", "station_id", "target_at"],
        how="inner",
        suffixes=("", "_target"),
    )
    frame = frame.merge(
        observations[["station_id", "observed_at", "demand"]],
        left_on=["station_id", "target_observed_at"],
        right_on=["station_id", "observed_at"],
        how="inner",
    )
    if frame.empty:
        return frame
    expected = targets[targets["cycle_id"].astype(str).isin(cycle_ids)].groupby("cycle_id").size().rename("expected").reset_index()
    counts = frame.groupby(["cycle_id", "model_id"]).size().rename("resolved").reset_index()
    complete = counts.merge(expected, on="cycle_id", how="left")
    complete = complete[complete["resolved"] == complete["expected"]][["cycle_id", "model_id"]]
    frame = frame.merge(complete, on=["cycle_id", "model_id"], how="inner")
    frame["actual"] = frame["demand"].astype(float)
    frame["prediction"] = frame["prediction"].astype(float)
    frame["residual"] = frame["prediction"] - frame["actual"]
    frame["abs_error"] = frame["residual"].abs()
    return frame


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and pd.notna(value) else None


def _delta(a: Any, b: Any) -> float | None:
    a, b = _finite(a), _finite(b)
    return a - b if a is not None and b is not None else None


def recommend_action(
    *,
    trigger_accuracy: float | None,
    threshold: float,
    drift: dict[str, Any],
    diagnosis: dict[str, Any],
    candidate_scores: list[CandidateScore],
) -> dict[str, Any]:
    best = candidate_scores[0] if candidate_scores else None
    if trigger_accuracy is None:
        return {"decision": "no_action", "reason": "no resolved accuracy available"}
    if trigger_accuracy >= threshold and not drift.get("retrain_recommended"):
        return {"decision": "no_action", "reason": f"accuracy {trigger_accuracy:.3f} >= threshold {threshold:.3f}"}
    if best is None:
        return {
            "decision": "keep_champion_and_collect_shadow",
            "reason": "accuracy trigger fired but no complete shadow candidate comparison is available yet",
        }
    if best.delta_vs_champion_recent is not None and best.delta_vs_champion_recent > 0.25 and (best.recent_accuracy or 0) >= threshold:
        return {
            "decision": "manual_promotion_review",
            "reason": "best shadow candidate beats champion on recent complete cycles",
            "candidate": best.__dict__,
            "required_manual_step": "approve the protected GitHub Actions promotion job after reviewing latest.md",
        }
    return {
        "decision": "keep_champion",
        "reason": "no shadow candidate clears the conservative promotion gate",
        "best_candidate": best.__dict__,
        "diagnosis_types": diagnosis["types"],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Retrain Autopilot Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Accuracy threshold: `{payload['threshold']}`",
        f"- Trigger metric: `{payload['trigger_metric']}` (leaderboard formula: mean over stations of 100*max(0, 1-WAPE_station))",
        f"- Trigger accuracy: `{payload['trigger_accuracy']}`",
        f"- Recent6 global WAPE accuracy (pooled, = leaderboard raw_wape): `{payload['accuracy_metrics']['recent6_global_wape_accuracy']}`",
        f"- Recent6 mean-of-cycles accuracy (legacy, not leaderboard): `{payload['accuracy_metrics']['recent6_mean_cycle_accuracy_legacy']}`",
        f"- Leaderboard rolling 24h estimate: accuracy `{payload['accuracy_metrics']['leaderboard_rolling_24h_accuracy']}`, coverage `{payload['accuracy_metrics']['leaderboard_rolling_24h_coverage']}`",
        f"- Should remediate: `{payload['should_remediate']}`",
        f"- Snapshot status: `{payload['snapshot_status']['overall_status']}` / drift `{payload['snapshot_status']['drift_status']}`",
        "",
        "## Drift Diagnosis",
        "",
        f"- Types: `{', '.join(payload['diagnosis']['types'])}`",
        f"- Reasons: `{', '.join(payload['diagnosis']['reasons']) or 'none'}`",
        f"- Accuracy drop pp: `{payload['diagnosis']['accuracy_drop_pp']}`",
        f"- Data status: `{payload['diagnosis']['data_status']}`",
        f"- Residual status: `{payload['diagnosis']['residual_status']}`",
        "",
        "## Remediation",
        "",
        f"- Shadow result: `{(payload.get('shadow_result') or {}).get('status', 'not_run')}`",
        f"- Recommendation: `{payload['recommendation']['decision']}`",
        f"- Reason: {payload['recommendation']['reason']}",
        "",
        "## Candidate Scores",
        "",
        "| model | shared cycles | mean station acc | champion (same cycles) | delta pp | global WAPE acc | delta global pp | worst station |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["candidate_scores"]:
        lines.append(
            f"| {row['model_name']} | {row['cycles']} | {_fmt(row['recent_accuracy'])} | {_fmt(row['champion_recent_accuracy'])} | {_fmt(row['delta_vs_champion_recent'])} | "
            f"{_fmt(row['recent_global_wape_accuracy'])} | {_fmt(row['delta_vs_champion_global_wape_pp'])} | {row['worst_station']} ({_fmt(row['worst_station_accuracy'])}) |"
        )
    if not payload["candidate_scores"]:
        lines.append("| none | 0 |  |  |  |  |  |  |")
    lines += [
        "",
        "## Safety",
        "",
        "- This job never submits predictions to the competition API.",
        "- This job never changes `model_versions.is_active`.",
        "- Promotion requires a manual protected GitHub Actions step.",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}"


def _console_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "trigger_metric": payload["trigger_metric"],
        "trigger_accuracy": payload["trigger_accuracy"],
        "trigger_global_wape_accuracy": payload["accuracy_metrics"]["recent6_global_wape_accuracy"],
        "should_remediate": payload["should_remediate"],
        "diagnosis_types": payload["diagnosis"]["types"],
        "shadow_status": (payload.get("shadow_result") or {}).get("status", "not_run"),
        "decision": payload["recommendation"]["decision"],
    }


if __name__ == "__main__":
    main()
