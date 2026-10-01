from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORT_DIR = ROOT / "reports" / "candidate_d2"
TABLE_DIR = REPORT_DIR / "tables"
HORIZONS = (15, 30, 45, 60)
CONTEXT_COLUMNS = ("rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity")


@dataclass(frozen=True)
class SupabaseReadOnlyClient:
    url: str
    key: str

    @classmethod
    def from_env(cls) -> "SupabaseReadOnlyClient":
        _load_env()
        url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
        key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY") or ""
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        return cls(url=url, key=key)

    def table(self, name: str, *, select: str, order: str | None = None, page_size: int = 1000) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        offset = 0
        with httpx.Client(
            base_url=f"{self.url}/rest/v1",
            headers={"apikey": self.key, "Authorization": f"Bearer {self.key}"},
            timeout=30.0,
        ) as client:
            while True:
                params = {"select": select, "limit": str(page_size), "offset": str(offset)}
                if order:
                    params["order"] = order
                response = client.get(f"/{name}", params=params)
                response.raise_for_status()
                page = response.json()
                rows.extend(page)
                if len(page) < page_size:
                    break
                offset += page_size
        return pd.DataFrame(rows)


def run_audit() -> dict[str, Any]:
    client = SupabaseReadOnlyClient.from_env()
    context = client.table(
        "context_observations",
        select="observed_at,rain_mm,rain_forecast,temperature_c,temperature_forecast,event_intensity",
        order="observed_at.asc",
    )
    observations = client.table("observations", select="station_id,observed_at,demand", order="observed_at.asc,station_id.asc")
    cycles = client.table("forecast_cycles", select="cycle_id,data_cutoff", order="data_cutoff.asc")
    targets = client.table("forecast_cycle_targets", select="cycle_id,station_id,target_at,horizon_minutes", order="cycle_id.asc,target_at.asc")
    public_api = audit_public_context_endpoint()
    payload = build_report_payload(context, observations, cycles, targets, public_api)
    write_outputs(payload)
    return payload


def build_report_payload(
    context: pd.DataFrame,
    observations: pd.DataFrame,
    cycles: pd.DataFrame,
    targets: pd.DataFrame,
    public_api: dict[str, Any],
) -> dict[str, Any]:
    context = _normalize_times(context, ["observed_at"])
    observations = _normalize_times(observations, ["observed_at"])
    cycles = _normalize_times(cycles, ["data_cutoff"])
    targets = _normalize_times(targets, ["target_at"])
    context_times = set(context["observed_at"].dropna())
    context_min = context["observed_at"].min() if not context.empty else pd.NaT
    context_max = context["observed_at"].max() if not context.empty else pd.NaT
    obs_min = observations["observed_at"].min() if not observations.empty else pd.NaT
    obs_max = observations["observed_at"].max() if not observations.empty else pd.NaT
    latest_public_context = pd.Timestamp(public_api["latest_context_at"]).tz_convert("UTC") if public_api.get("latest_context_at") else pd.NaT
    public_api["context_endpoint_reaches_observation_max"] = bool(pd.notna(latest_public_context) and pd.notna(obs_max) and latest_public_context >= obs_max)

    target_coverage = forecast_target_coverage(cycles, targets, context_times)
    supervised_coverage = supervised_target_coverage(observations, context_times)
    feature_audit = feature_availability_table(context_min, context_max, obs_max, public_api)
    safe_offline = [row["feature"] for row in feature_audit if row["safe_for_training"]]
    safe_live = [row["feature"] for row in feature_audit if row["safe_for_live_inference"]]
    missing_infra = []
    if not public_api.get("context_endpoint_reaches_observation_max"):
        missing_infra.append("No verified live context ingestion/sync into Supabase up to observations max.")
    if not any(row["safe_for_live_inference"] for row in feature_audit):
        missing_infra.append("No feature has proven live availability at current prediction time.")
    if "rain_forecast" not in safe_live or "temperature_forecast" not in safe_live:
        missing_infra.append("Forecast columns lack issued_at/available_at metadata proving target-time availability before data_cutoff.")

    return {
        "source": "supabase_read_only_plus_public_api_read_only",
        "context_coverage": {
            "context_rows": int(len(context)),
            "context_min": _iso(context_min),
            "context_max": _iso(context_max),
            "observation_rows": int(len(observations)),
            "observations_min": _iso(obs_min),
            "observations_max": _iso(obs_max),
            "context_lag_minutes_vs_observations_max": _minutes_between(context_max, obs_max),
            "forecast_cycle_targets": target_coverage,
            "supervised_backtest_targets": supervised_coverage,
        },
        "public_context_endpoint": public_api,
        "feature_availability": feature_audit,
        "summary": {
            "D2 usable offline": bool(safe_offline),
            "D2 usable live today": bool(safe_live),
            "safe offline features": safe_offline,
            "safe live features": safe_live,
            "missing live infrastructure": missing_infra,
            "context coverage": {
                "context_min": _iso(context_min),
                "context_max": _iso(context_max),
                "observations_max": _iso(obs_max),
                "forecast_targets_with_context_at_data_cutoff": target_coverage["targets_with_context_at_data_cutoff"],
                "forecast_targets_total": target_coverage["targets_total"],
                "supervised_targets_with_context_at_data_cutoff": supervised_coverage["targets_with_context_at_data_cutoff"],
                "supervised_targets_total": supervised_coverage["targets_total"],
            },
        },
    }


def forecast_target_coverage(cycles: pd.DataFrame, targets: pd.DataFrame, context_times: set[pd.Timestamp]) -> dict[str, Any]:
    if cycles.empty or targets.empty:
        return {"targets_total": 0, "targets_with_context_at_data_cutoff": 0, "targets_with_context_at_target": 0}
    merged = targets.merge(cycles, on="cycle_id", how="left")
    merged["has_context_at_data_cutoff"] = merged["data_cutoff"].isin(context_times)
    merged["has_context_at_target"] = merged["target_at"].isin(context_times)
    return {
        "targets_total": int(len(merged)),
        "targets_with_context_at_data_cutoff": int(merged["has_context_at_data_cutoff"].sum()),
        "targets_with_context_at_target": int(merged["has_context_at_target"].sum()),
        "coverage_at_data_cutoff": _ratio(merged["has_context_at_data_cutoff"].sum(), len(merged)),
        "coverage_at_target": _ratio(merged["has_context_at_target"].sum(), len(merged)),
        "target_min": _iso(merged["target_at"].min()),
        "target_max": _iso(merged["target_at"].max()),
        "data_cutoff_min": _iso(merged["data_cutoff"].min()),
        "data_cutoff_max": _iso(merged["data_cutoff"].max()),
    }


def supervised_target_coverage(observations: pd.DataFrame, context_times: set[pd.Timestamp]) -> dict[str, Any]:
    if observations.empty:
        return {"targets_total": 0, "targets_with_context_at_data_cutoff": 0, "targets_with_context_at_target": 0}
    rows = []
    observed_times = set(observations["observed_at"].dropna())
    for horizon in HORIZONS:
        target_at = observations["observed_at"]
        data_cutoff = target_at - pd.Timedelta(minutes=horizon)
        eligible = data_cutoff.isin(observed_times)
        rows.append(
            pd.DataFrame(
                {
                    "horizon_minutes": horizon,
                    "target_at": target_at[eligible],
                    "data_cutoff": data_cutoff[eligible],
                }
            )
        )
    samples = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    samples["has_context_at_data_cutoff"] = samples["data_cutoff"].isin(context_times)
    samples["has_context_at_target"] = samples["target_at"].isin(context_times)
    by_horizon = (
        samples.groupby("horizon_minutes", as_index=False)
        .agg(
            targets_total=("target_at", "size"),
            targets_with_context_at_data_cutoff=("has_context_at_data_cutoff", "sum"),
            targets_with_context_at_target=("has_context_at_target", "sum"),
        )
        .to_dict(orient="records")
    )
    return {
        "targets_total": int(len(samples)),
        "targets_with_context_at_data_cutoff": int(samples["has_context_at_data_cutoff"].sum()),
        "targets_with_context_at_target": int(samples["has_context_at_target"].sum()),
        "coverage_at_data_cutoff": _ratio(samples["has_context_at_data_cutoff"].sum(), len(samples)),
        "coverage_at_target": _ratio(samples["has_context_at_target"].sum(), len(samples)),
        "by_horizon": [
            {
                **row,
                "coverage_at_data_cutoff": _ratio(row["targets_with_context_at_data_cutoff"], row["targets_total"]),
                "coverage_at_target": _ratio(row["targets_with_context_at_target"], row["targets_total"]),
            }
            for row in by_horizon
        ],
    }


def feature_availability_table(
    context_min: pd.Timestamp,
    context_max: pd.Timestamp,
    obs_max: pd.Timestamp,
    public_api: dict[str, Any],
) -> list[dict[str, Any]]:
    live_context_current = bool(public_api.get("context_endpoint_reaches_observation_max")) and context_max >= obs_max
    rows = []
    for feature in CONTEXT_COLUMNS:
        is_actual = feature in {"rain_mm", "temperature_c"}
        is_forecast = feature in {"rain_forecast", "temperature_forecast"}
        is_event = feature == "event_intensity"
        if is_actual:
            justification = (
                "Actual context at observed_at is safe only when observed_at <= data_cutoff. "
                "Using target_at actual weather would leak future target-time conditions."
            )
            training = True
            live = live_context_current
            risk = "low at data_cutoff; high if joined at target_at"
        elif is_forecast:
            justification = (
                "Forecast value has no issued_at/available_at/valid_for metadata. It is safe as a data_cutoff-time covariate "
                "only when the row observed_at <= data_cutoff exists; it is not proven safe for target_at forecasts."
            )
            training = True
            live = False
            risk = "medium: forecast naming is insufficient proof for target-time use"
        elif is_event:
            justification = (
                "Event intensity has no source, issued_at, or event schedule metadata. It is allowed offline only at observed_at <= data_cutoff; "
                "target-time event intensity may be postcomputed."
            )
            training = True
            live = False
            risk = "medium/high without event schedule availability metadata"
        rows.append(
            {
                "feature": feature,
                "timestamp_corresponds_to": "context_observations.observed_at",
                "available_at_prediction_time": "only if context_observations.observed_at <= data_cutoff and row exists",
                "leakage_risk": risk,
                "allowed_for_D2": training,
                "safe_for_training": training,
                "safe_for_live_inference": live,
                "justification": justification,
            }
        )
    return rows


def audit_public_context_endpoint() -> dict[str, Any]:
    _load_env()
    headers = {"User-Agent": "pulso-transmi-python/0.1.0"}
    if os.getenv("PULSO_API_KEY"):
        headers["Authorization"] = f"Bearer {os.getenv('PULSO_API_KEY')}"
    result: dict[str, Any] = {"endpoint_exists": False, "latest_context_at": None, "error": None}
    try:
        with httpx.Client(base_url=os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io"), headers=headers, timeout=30.0) as client:
            meta = client.get("/v1/meta")
            meta.raise_for_status()
            payload = meta.json()
            result["endpoint_exists"] = True
            result["meta"] = payload.get("dataset", payload)
            start = "2026-09-12T00:00:00-05:00"
            context = client.get("/v1/context", params={"start": start, "limit": "5000"})
            context.raise_for_status()
            rows = context.json().get("data", [])
            if rows:
                times = pd.to_datetime([row["observed_at"] for row in rows], utc=True)
                result["latest_context_at"] = _iso(times.max())
                result["rows_after_2026_09_12_local"] = len(rows)
            else:
                result["rows_after_2026_09_12_local"] = 0
    except Exception as exc:
        result["error"] = str(exc)
    result["context_endpoint_reaches_observation_max"] = False
    return result


def write_outputs(payload: dict[str, Any]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(payload["feature_availability"]).to_csv(TABLE_DIR / "context_feature_availability.csv", index=False)
    (REPORT_DIR / "context_availability_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (REPORT_DIR / "context_availability_audit.md").write_text(render_markdown(payload) + "\n")


def render_markdown(payload: dict[str, Any]) -> str:
    feature_table = pd.DataFrame(payload["feature_availability"])
    summary = payload["summary"]
    return "\n".join(
        [
            "# Candidate D2 Context Availability Audit",
            "",
            "## Feature Availability",
            "",
            _markdown_table(feature_table),
            "",
            "## Summary",
            "",
            "```yaml",
            f"D2 usable offline: {'yes' if summary['D2 usable offline'] else 'no'}",
            f"D2 usable live today: {'yes' if summary['D2 usable live today'] else 'no'}",
            f"safe offline features: {summary['safe offline features']}",
            f"safe live features: {summary['safe live features']}",
            f"missing live infrastructure: {summary['missing live infrastructure']}",
            f"context coverage: {summary['context coverage']}",
            "```",
        ]
    )


def _normalize_times(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def _load_env(path: Path = ROOT / ".env.local") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


def _ratio(numerator: Any, denominator: Any) -> float:
    denominator = int(denominator)
    return float(numerator) / denominator if denominator else 0.0


def _minutes_between(start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    if pd.isna(start) or pd.isna(end):
        return None
    return float((end - start).total_seconds() / 60.0)


def _iso(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        rows.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
