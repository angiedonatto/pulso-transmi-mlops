from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx
import pandas as pd


MODEL_NAME = "daily_persistence_slot_mean_baseline"
MODEL_TYPE = "baseline_lag96_station_slot_mean"


class ChampionBaselineError(RuntimeError):
    """Raised when champion baseline training or loading is unsafe."""


@dataclass(frozen=True)
class StationMetric:
    station_id: str
    wape: float
    accuracy: float


@dataclass(frozen=True)
class BaselineEvaluation:
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    station_metrics: list[StationMetric]
    mean_accuracy: float
    mean_wape: float


class BaselineRegistry(Protocol):
    def register_champion(
        self,
        *,
        artifact: dict[str, Any],
        artifact_uri: str,
        evaluation: BaselineEvaluation,
        git_commit: str | None,
    ) -> dict[str, Any]:
        ...

    def load_active_champion(self) -> dict[str, Any]:
        ...


def train_baseline(
    observations: pd.DataFrame,
    *,
    data_cutoff: str,
    validation_periods: int = 96,
) -> tuple[dict[str, Any], BaselineEvaluation]:
    frame = _prepare_observations(observations)
    cutoff = pd.Timestamp(data_cutoff).tz_convert("UTC")
    eligible = frame[frame["observed_at"] <= cutoff].copy()
    if eligible.empty:
        raise ChampionBaselineError("no observations at or before data_cutoff")

    validation_times = sorted(eligible["observed_at"].drop_duplicates())[-validation_periods:]
    if not validation_times:
        raise ChampionBaselineError("validation window is empty")
    validation_start = validation_times[0]
    validation_end = validation_times[-1]
    train = eligible[eligible["observed_at"] < validation_start].copy()
    validation = eligible[(eligible["observed_at"] >= validation_start) & (eligible["observed_at"] <= validation_end)].copy()
    if train.empty or validation.empty:
        raise ChampionBaselineError("training and validation windows must be non-empty")

    slot_means = (
        train.groupby(["station_id", "quarter_hour_block"], sort=True)["demand"]
        .mean()
        .reset_index()
        .rename(columns={"demand": "historical_station_slot_mean"})
    )
    lag_source = train[["station_id", "observed_at", "demand"]].rename(
        columns={"observed_at": "lag_observed_at", "demand": "lag_96_demand"}
    )
    scored = validation.copy()
    scored["lag_observed_at"] = scored["observed_at"] - pd.to_timedelta(1, unit="D")
    scored = scored.merge(lag_source, on=["station_id", "lag_observed_at"], how="left")
    scored = scored.merge(slot_means, on=["station_id", "quarter_hour_block"], how="left")
    scored["prediction"] = scored["lag_96_demand"].fillna(scored["historical_station_slot_mean"])
    if scored["prediction"].isna().any():
        missing = scored[scored["prediction"].isna()]["station_id"].drop_duplicates().tolist()
        raise ChampionBaselineError(f"missing fallback mean for stations: {missing}")

    station_metrics = _station_metrics(scored)
    evaluation = BaselineEvaluation(
        training_start=_ts(train["observed_at"].min()),
        training_end=_ts(train["observed_at"].max()),
        validation_start=_ts(validation_start),
        validation_end=_ts(validation_end),
        station_metrics=station_metrics,
        mean_accuracy=sum(metric.accuracy for metric in station_metrics) / len(station_metrics),
        mean_wape=sum(metric.wape for metric in station_metrics) / len(station_metrics),
    )
    artifact = {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "trained_until": _ts(cutoff),
        "created_at": _ts(pd.Timestamp.now(tz="UTC")),
        "feature_config": {
            "primary": "lag_96_demand",
            "fallback": "historical_station_slot_mean",
            "group_keys": ["station_id", "quarter_hour_block"],
            "validation_periods": validation_periods,
        },
        "params": {"lag_periods": 96, "lag_duration": "P1D"},
        "validation": _evaluation_payload(evaluation),
        "station_slot_means": {
            f"{row.station_id}|{int(row.quarter_hour_block)}": float(row.historical_station_slot_mean)
            for row in slot_means.itertuples(index=False)
        },
    }
    return artifact, evaluation


def save_artifact(artifact: dict[str, Any], *, directory: str | Path = "artifacts/models") -> str:
    artifact_dir = Path(directory)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trained_until = artifact["trained_until"].replace(":", "").replace("-", "")
    path = artifact_dir / f"{MODEL_NAME}_{trained_until}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return str(path)


class SupabaseBaselineStore:
    def __init__(self, supabase_url: str | None = None, service_key: str | None = None, timeout: float = 30.0) -> None:
        resolved_url = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        resolved_key = service_key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not resolved_url or not resolved_key:
            raise ChampionBaselineError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self._client = httpx.Client(
            base_url=f"{resolved_url}/rest/v1",
            headers={
                "apikey": resolved_key,
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SupabaseBaselineStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def observations(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        offset = 0
        page_size = 1000
        while True:
            response = self._client.get(
                "/observations",
                params={
                    "select": "station_id,observed_at,demand,time_slots(quarter_hour_block)",
                    "order": "observed_at.asc,station_id.asc",
                    "limit": str(page_size),
                    "offset": str(offset),
                },
            )
            response.raise_for_status()
            page = response.json()
            rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return pd.DataFrame(
            [
                {
                    "station_id": row["station_id"],
                    "observed_at": row["observed_at"],
                    "demand": row["demand"],
                    "quarter_hour_block": row["time_slots"]["quarter_hour_block"],
                }
                for row in rows
            ]
        )

    def register_champion(
        self,
        *,
        artifact: dict[str, Any],
        artifact_uri: str,
        evaluation: BaselineEvaluation,
        git_commit: str | None,
    ) -> dict[str, Any]:
        run_id = str(uuid4())
        model_id = str(uuid4())
        self._post(
            "/pipeline_runs",
            {
                "run_id": run_id,
                "status": "running",
                "git_commit": git_commit,
                "data_cutoff": artifact["trained_until"],
                "trigger_type": "manual",
            },
        )
        self._patch("/model_versions", {"is_active": False}, params={"is_active": "eq.true"})
        self._post(
            "/model_versions",
            {
                "model_id": model_id,
                "run_id": run_id,
                "model_name": artifact["model_name"],
                "model_type": artifact["model_type"],
                "artifact_uri": artifact_uri,
                "trained_until": artifact["trained_until"],
                "feature_config": artifact["feature_config"],
                "params": {**artifact["params"], "validation": artifact["validation"]},
                "is_active": True,
            },
        )
        self._post(
            "/metrics_by_station",
            [
                {
                    "run_id": run_id,
                    "model_id": model_id,
                    "station_id": metric.station_id,
                    "split_name": "validation",
                    "evaluation_start": evaluation.validation_start,
                    "evaluation_end": evaluation.validation_end,
                    "wape": metric.wape,
                    "accuracy": metric.accuracy,
                }
                for metric in evaluation.station_metrics
            ],
        )
        self._patch("/pipeline_runs", {"status": "success", "finished_at": _now_iso()}, params={"run_id": f"eq.{run_id}"})
        champion = self.load_active_champion()
        if champion["model_id"] != model_id:
            raise ChampionBaselineError("active champion verification failed")
        return {"run_id": run_id, "model_id": model_id, **champion}

    def load_active_champion(self) -> dict[str, Any]:
        response = self._client.get(
            "/model_versions",
            params={"is_active": "eq.true", "select": "*"},
        )
        response.raise_for_status()
        rows = response.json()
        if len(rows) != 1:
            raise ChampionBaselineError(f"expected exactly one active champion, found {len(rows)}")
        return rows[0]

    def _post(self, path: str, payload: Any) -> None:
        response = self._client.post(path, json=payload)
        response.raise_for_status()

    def _patch(self, path: str, payload: dict[str, Any], *, params: dict[str, str]) -> None:
        response = self._client.patch(path, params=params, json=payload)
        response.raise_for_status()


def train_register_champion(*, data_cutoff: str | None = None) -> dict[str, Any]:
    with SupabaseBaselineStore() as store:
        observations = store.observations()
        cutoff = data_cutoff or _ts(pd.to_datetime(observations["observed_at"], utc=True).max())
        artifact, evaluation = train_baseline(observations, data_cutoff=cutoff)
        artifact["git_commit"] = _git_commit()
        artifact_uri = save_artifact(artifact)
        registered = store.register_champion(
            artifact=artifact,
            artifact_uri=artifact_uri,
            evaluation=evaluation,
            git_commit=artifact["git_commit"],
        )
    return {"artifact": artifact, "artifact_uri": artifact_uri, "evaluation": evaluation, "registered": registered}


def load_active_champion() -> dict[str, Any]:
    with SupabaseBaselineStore() as store:
        champion = store.load_active_champion()
    artifact_uri = champion.get("artifact_uri")
    if not artifact_uri:
        raise ChampionBaselineError("active champion has no artifact_uri")
    artifact = json.loads(Path(artifact_uri).read_text())
    return {"model_version": champion, "artifact": artifact}


def _prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"station_id", "observed_at", "demand"}
    missing = required - set(observations.columns)
    if missing:
        raise ChampionBaselineError(f"missing observation columns: {sorted(missing)}")
    frame = observations.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["demand"] = frame["demand"].astype(float)
    if "quarter_hour_block" not in frame.columns:
        local = frame["observed_at"].dt.tz_convert("America/Bogota")
        frame["quarter_hour_block"] = local.dt.hour * 4 + local.dt.minute // 15
    frame["quarter_hour_block"] = frame["quarter_hour_block"].astype(int)
    return frame.sort_values(["observed_at", "station_id"]).reset_index(drop=True)


def _station_metrics(scored: pd.DataFrame) -> list[StationMetric]:
    metrics: list[StationMetric] = []
    for station_id, group in scored.groupby("station_id", sort=True):
        denominator = group["demand"].abs().sum()
        wape = 0.0 if denominator == 0 else float((group["demand"] - group["prediction"]).abs().sum() / denominator)
        accuracy = 100.0 * max(0.0, 1.0 - wape)
        metrics.append(StationMetric(station_id=str(station_id), wape=wape, accuracy=accuracy))
    return metrics


def _evaluation_payload(evaluation: BaselineEvaluation) -> dict[str, Any]:
    return {
        "training_start": evaluation.training_start,
        "training_end": evaluation.training_end,
        "validation_start": evaluation.validation_start,
        "validation_end": evaluation.validation_end,
        "mean_wape": evaluation.mean_wape,
        "mean_accuracy": evaluation.mean_accuracy,
        "station_metrics": [
            {"station_id": metric.station_id, "wape": metric.wape, "accuracy": metric.accuracy}
            for metric in evaluation.station_metrics
        ],
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ts(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def main() -> None:
    result = train_register_champion()
    evaluation: BaselineEvaluation = result["evaluation"]
    registered = result["registered"]
    print(
        json.dumps(
            {
                "training_start": evaluation.training_start,
                "training_end": evaluation.training_end,
                "validation_start": evaluation.validation_start,
                "validation_end": evaluation.validation_end,
                "mean_wape": evaluation.mean_wape,
                "mean_accuracy": evaluation.mean_accuracy,
                "station_metrics": [
                    {"station_id": metric.station_id, "wape": metric.wape, "accuracy": metric.accuracy}
                    for metric in evaluation.station_metrics
                ],
                "model_id": registered["model_id"],
                "trained_until": result["artifact"]["trained_until"],
                "artifact_uri": result["artifact_uri"],
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
