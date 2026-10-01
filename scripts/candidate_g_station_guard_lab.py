from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))

from candidate_g_drift_robustness_lab import normalize_prediction_frame, normalize_station  # noqa: E402


REPORT_DIR = ROOT / "reports" / "candidate_g_station_guard_lab"
TABLE_DIR = REPORT_DIR / "tables"
G_LAB_DIR = ROOT / "reports" / "candidate_g_drift_robustness_lab"

BLOCK_SIZE_CYCLES = 6
NIGHT_HOURS = {18, 19, 20, 21, 22, 23}
ALERT_STATIONS = ("02300", "03000", "05000", "05100", "07111", "09122")
STABLE_REGRESSION_STATIONS = ("02300", "03000", "06111", "07105")
MAX_STABLE_REGRESSION_PP = -0.50
MIN_PROMOTION_DELTA_PP = 1.00


@dataclass(frozen=True)
class GuardVariant:
    name: str
    alert_mode: str
    stable_mode: str
    night_alert_mode: str = "unchanged"
    recency_profile_weight: float = 0.0


VARIANTS = (
    GuardVariant("candidate_g_current", "unchanged", "unchanged"),
    GuardVariant("station_residual_guard", "unchanged", "shrink"),
    GuardVariant("station_residual_guard_freeze", "unchanged", "freeze"),
    GuardVariant("night_station_guard", "unchanged", "shrink", night_alert_mode="boost"),
    GuardVariant("recency_profile_guard", "unchanged", "shrink", recency_profile_weight=0.20),
)


def load_live_predictions(path: Path = G_LAB_DIR / "live_g_predictions.csv") -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run scripts/candidate_g_drift_robustness_lab.py first.")
    frame = normalize_prediction_frame(pd.read_csv(path, dtype={"station_id": "string"}))
    frame["station_id"] = frame["station_id"].map(normalize_station)
    frame["target_hour"] = pd.to_datetime(frame["target_at"], utc=True).dt.tz_convert("America/Bogota").dt.hour.astype(int)
    return frame.sort_values(["data_cutoff", "cycle_id", "station_id", "horizon_minutes"]).reset_index(drop=True)


def attach_blocks(frame: pd.DataFrame, *, block_size: int = BLOCK_SIZE_CYCLES) -> pd.DataFrame:
    cycles = (
        frame.groupby("cycle_id", as_index=False)["data_cutoff"]
        .min()
        .sort_values(["data_cutoff", "cycle_id"])
        .reset_index(drop=True)
    )
    cycles["cycle_position"] = np.arange(len(cycles))
    cycles["block_index"] = (cycles["cycle_position"] // block_size).astype(int)
    return frame.merge(cycles[["cycle_id", "cycle_position", "block_index"]], on="cycle_id", how="inner")


def station_bias_signals(frame: pd.DataFrame) -> dict[str, float]:
    metric = metric_by(frame.assign(prediction=frame["prediction_g"]), ["station_id"])
    return {
        str(row.station_id): float(row.signed_relative_bias)
        for row in metric.itertuples(index=False)
        if math.isfinite(float(row.signed_relative_bias))
    }


def build_recency_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["target_quarter_hour_block"] = (
        pd.to_datetime(work["target_at"], utc=True).dt.tz_convert("America/Bogota").dt.hour * 4
        + pd.to_datetime(work["target_at"], utc=True).dt.tz_convert("America/Bogota").dt.minute // 15
    ).astype(int)
    work["recency_profile_prediction"] = np.nan
    for idx, row in work.iterrows():
        history = work[
            (work["station_id"] == row["station_id"])
            & (work["target_quarter_hour_block"] == row["target_quarter_hour_block"])
            & (work["target_at"] <= row["data_cutoff"])
        ].tail(8)
        if not history.empty:
            work.at[idx, "recency_profile_prediction"] = float(history["actual"].median())
    work["recency_profile_prediction"] = work["recency_profile_prediction"].fillna(work["prediction_g"])
    return work


def metric_by(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    work = frame.copy()
    work["residual"] = work["prediction"].astype(float) - work["actual"].astype(float)
    work["abs_error"] = work["residual"].abs()
    grouped = work.groupby(group_cols, as_index=False).agg(
        abs_error_sum=("abs_error", "sum"),
        actual_sum=("actual", "sum"),
        residual_sum=("residual", "sum"),
        mae=("abs_error", "mean"),
        bias=("residual", "mean"),
        rows=("actual", "size"),
    )
    grouped["wape"] = np.where(grouped["actual_sum"] > 0, grouped["abs_error_sum"] / grouped["actual_sum"], np.nan)
    grouped["accuracy"] = 100.0 * np.maximum(0.0, 1.0 - grouped["wape"])
    grouped["signed_relative_bias"] = np.where(grouped["actual_sum"] > 0, grouped["residual_sum"] / grouped["actual_sum"], np.nan)
    return grouped[group_cols + ["accuracy", "wape", "mae", "bias", "signed_relative_bias", "rows"]]


def official_accuracy(frame: pd.DataFrame) -> float:
    station = metric_by(frame, ["station_id"])
    return float(station["accuracy"].mean())


def apply_variant(frame: pd.DataFrame, variant: GuardVariant, *, bias_by_station: dict[str, float] | None = None) -> pd.DataFrame:
    bias_by_station = bias_by_station or station_bias_signals(frame)
    work = build_recency_profiles(frame)
    factor = work["calibration_factor"].astype(float).copy()

    stable_mask = work["station_id"].isin(STABLE_REGRESSION_STATIONS)
    if variant.stable_mode == "shrink":
        factor.loc[stable_mask] = 1.0 + 0.50 * (factor.loc[stable_mask] - 1.0)
    elif variant.stable_mode == "freeze":
        factor.loc[stable_mask] = 1.0

    alert_mask = work["station_id"].isin(ALERT_STATIONS)
    if variant.alert_mode == "freeze":
        factor.loc[alert_mask] = 1.0
    elif variant.alert_mode == "shrink":
        factor.loc[alert_mask] = 1.0 + 0.50 * (factor.loc[alert_mask] - 1.0)

    if variant.night_alert_mode == "boost":
        night_alert = alert_mask & work["target_hour"].isin(NIGHT_HOURS)
        for station_id, signed_bias in bias_by_station.items():
            station_mask = night_alert & (work["station_id"] == station_id)
            if signed_bias < -0.05:
                factor.loc[station_mask] = np.maximum(factor.loc[station_mask], 1.05)
            elif signed_bias > 0.05:
                factor.loc[station_mask] = np.minimum(factor.loc[station_mask], 0.95)

    prediction = (work["prediction"].astype(float) * factor).clip(lower=0.0)
    if variant.recency_profile_weight > 0:
        profile = work["recency_profile_prediction"].astype(float)
        profile_mask = alert_mask & profile.notna()
        prediction.loc[profile_mask] = (
            (1.0 - variant.recency_profile_weight) * prediction.loc[profile_mask]
            + variant.recency_profile_weight * profile.loc[profile_mask]
        ).clip(lower=0.0)

    work[f"{variant.name}_prediction"] = prediction
    work[f"{variant.name}_factor"] = factor
    return work


def score_variant(frame: pd.DataFrame, variant: GuardVariant) -> dict[str, Any]:
    pred_col = f"{variant.name}_prediction"
    station = metric_by(frame.assign(prediction=frame[pred_col]), ["station_id"])
    horizon = metric_by(frame.assign(prediction=frame[pred_col]), ["horizon_minutes"])
    recent_blocks = sorted(frame["block_index"].unique())[-2:]
    recent = frame[frame["block_index"].isin(recent_blocks)].copy()
    stable = station[station["station_id"].astype(str).isin(STABLE_REGRESSION_STATIONS)].copy()
    alerts = station[station["station_id"].astype(str).isin(ALERT_STATIONS)].copy()
    return {
        "variant": variant.name,
        "mean_station_accuracy": float(station["accuracy"].mean()),
        "official_accuracy": official_accuracy(frame.assign(prediction=frame[pred_col])),
        "recent_mean_station_accuracy": float(metric_by(recent.assign(prediction=recent[pred_col]), ["station_id"])["accuracy"].mean()),
        "alert_mean_abs_bias": float(alerts["signed_relative_bias"].abs().mean()) if not alerts.empty else math.nan,
        "stable_min_accuracy": float(stable["accuracy"].min()) if not stable.empty else math.nan,
        "station_metrics": station,
        "horizon_metrics": horizon,
    }


def compare_variants(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    bias_by_station = station_bias_signals(frame)
    scored_frame = frame.copy()
    rows: list[dict[str, Any]] = []
    station_tables: dict[str, pd.DataFrame] = {}
    horizon_tables: dict[str, pd.DataFrame] = {}
    baseline: dict[str, Any] | None = None
    for variant in VARIANTS:
        scored_frame = apply_variant(scored_frame, variant, bias_by_station=bias_by_station)
        score = score_variant(scored_frame, variant)
        if variant.name == "candidate_g_current":
            baseline = score
        assert baseline is not None
        rows.append(
            {
                "variant": variant.name,
                "mean_station_accuracy": score["mean_station_accuracy"],
                "delta_vs_current_pp": score["mean_station_accuracy"] - baseline["mean_station_accuracy"],
                "recent_mean_station_accuracy": score["recent_mean_station_accuracy"],
                "recent_delta_vs_current_pp": score["recent_mean_station_accuracy"] - baseline["recent_mean_station_accuracy"],
                "alert_mean_abs_bias": score["alert_mean_abs_bias"],
                "alert_abs_bias_delta": score["alert_mean_abs_bias"] - baseline["alert_mean_abs_bias"],
                "stable_min_accuracy": score["stable_min_accuracy"],
                "stable_min_delta": score["stable_min_accuracy"] - baseline["stable_min_accuracy"],
            }
        )
        station_tables[variant.name] = score["station_metrics"]
        horizon_tables[variant.name] = score["horizon_metrics"]
    return pd.DataFrame(rows), station_tables, horizon_tables


def promotion_verdict(comparison: pd.DataFrame) -> dict[str, Any]:
    candidates = comparison[comparison["variant"] != "candidate_g_current"].copy()
    candidates["passes"] = (
        (candidates["recent_delta_vs_current_pp"] >= MIN_PROMOTION_DELTA_PP)
        & (candidates["stable_min_delta"] >= MAX_STABLE_REGRESSION_PP)
        & (candidates["alert_abs_bias_delta"] < 0)
    )
    if candidates["passes"].any():
        best = candidates[candidates["passes"]].sort_values(
            ["recent_delta_vs_current_pp", "delta_vs_current_pp", "alert_abs_bias_delta"],
            ascending=[False, False, True],
        ).iloc[0]
        return {
            "promote": True,
            "recommended_variant": str(best["variant"]),
            "reason": "passes recent accuracy, stable-station regression, and alert-bias gates",
        }
    best = candidates.sort_values(["recent_delta_vs_current_pp", "delta_vs_current_pp"], ascending=False).iloc[0]
    return {
        "promote": False,
        "recommended_variant": "candidate_g_current",
        "best_shadow_variant": str(best["variant"]),
        "reason": "no shadow variant cleared all promotion gates",
    }


def leakage_audit(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "target_after_data_cutoff": bool((frame["target_at"] > frame["data_cutoff"]).all()),
        "feedback_feature_origin": "loaded from Candidate G robustness lab; g2 signals use target_at <= data_cutoff",
        "recency_profile_uses_resolved_targets_only": True,
        "production_writes": False,
        "submissions": False,
        "champion_changed": False,
    }


def run_lab() -> dict[str, Any]:
    start = time.perf_counter()
    frame = attach_blocks(load_live_predictions())
    comparison, station_tables, horizon_tables = compare_variants(frame)
    verdict = promotion_verdict(comparison)
    audit = leakage_audit(frame)
    current = comparison[comparison["variant"] == "candidate_g_current"].iloc[0]
    payload = {
        "lab": "candidate_g_station_guard_lab",
        "scope": "shadow_only_no_production_changes",
        "cycles": int(frame["cycle_id"].nunique()),
        "rows": int(len(frame)),
        "blocks": int(frame["block_index"].nunique()),
        "alert_stations": list(ALERT_STATIONS),
        "stable_regression_stations": list(STABLE_REGRESSION_STATIONS),
        "promotion_gates": {
            "min_recent_delta_pp": MIN_PROMOTION_DELTA_PP,
            "max_stable_regression_pp": MAX_STABLE_REGRESSION_PP,
            "requires_alert_abs_bias_reduction": True,
        },
        "current_mean_station_accuracy": float(current["mean_station_accuracy"]),
        "current_recent_mean_station_accuracy": float(current["recent_mean_station_accuracy"]),
        "verdict": verdict,
        "leakage": audit,
        "runtime_seconds": time.perf_counter() - start,
    }
    write_outputs(payload, comparison, station_tables, horizon_tables)
    return payload


def write_outputs(
    payload: dict[str, Any],
    comparison: pd.DataFrame,
    station_tables: dict[str, pd.DataFrame],
    horizon_tables: dict[str, pd.DataFrame],
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(TABLE_DIR / "variant_comparison.csv", index=False)
    station_frames = []
    horizon_frames = []
    for variant, table in station_tables.items():
        station_frames.append(table.assign(variant=variant))
    for variant, table in horizon_tables.items():
        horizon_frames.append(table.assign(variant=variant))
    pd.concat(station_frames, ignore_index=True).to_csv(TABLE_DIR / "station_metrics.csv", index=False)
    pd.concat(horizon_frames, ignore_index=True).to_csv(TABLE_DIR / "horizon_metrics.csv", index=False)
    (REPORT_DIR / "summary.json").write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "summary.md").write_text(render_markdown(payload, comparison) + "\n")


def render_markdown(payload: dict[str, Any], comparison: pd.DataFrame) -> str:
    verdict = payload["verdict"]
    return "\n".join(
        [
            "# Candidate G Station Guard Lab",
            "",
            "## Scope",
            "",
            "- Shadow experiment only. No champion, submission, scheduler, or Candidate G production parameter was changed.",
            "- Uses resolved live predictions from the Candidate G robustness lab.",
            "- Variants test station residual guard, stable-station shrink/freeze, night-hour guard, and recency profile blending.",
            "",
            "## Promotion Gates",
            "",
            f"- Recent accuracy delta must be >= {MIN_PROMOTION_DELTA_PP:.2f} pp.",
            f"- Stable-station minimum accuracy cannot regress below {MAX_STABLE_REGRESSION_PP:.2f} pp.",
            "- Alert-station absolute signed bias must improve.",
            "",
            "## Variant Comparison",
            "",
            _markdown_table(comparison),
            "",
            "## Verdict",
            "",
            f"- Promote: `{verdict['promote']}`.",
            f"- Recommended variant: `{verdict['recommended_variant']}`.",
            f"- Reason: {verdict['reason']}.",
            "",
            "## Leakage",
            "",
            f"- `target_after_data_cutoff`: {payload['leakage']['target_after_data_cutoff']}.",
            f"- `recency_profile_uses_resolved_targets_only`: {payload['leakage']['recency_profile_uses_resolved_targets_only']}.",
            f"- `production_writes`: {payload['leakage']['production_writes']}.",
        ]
    )


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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Candidate G station guard shadow lab.")
    parser.parse_args()
    payload = run_lab()
    print(json.dumps(_json_value(payload), indent=2, sort_keys=True))
    print(f"Report written to {REPORT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
