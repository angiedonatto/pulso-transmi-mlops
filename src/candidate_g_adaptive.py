from __future__ import annotations

import json
import math
import os
import pickle
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from candidate_b_shadow import (
    BOGOTA_TZ,
    FEATURE_SCHEMA,
    HORIZONS,
    HYPERPARAMETERS,
    MIN_HISTORY_DAYS,
    SHORT_LAGS,
    WEEKLY_LAG_MINUTES,
    ShadowArtifact,
    ShadowCandidateError,
    _prepare_observations,
    _supervised_samples,
    _target_time_features,
    _ts,
)
from cycle_predictions import CyclePrediction
from forecast_cycle_sync import ForecastCycle


MODEL_VERSION = "candidate_g_v1"
MODEL_NAME = "candidate_g_f4_profile_ratio_g2_ewma"
MODEL_TYPE = "shadow_f4_profile_ratio_g2_ewma"
DEFAULT_ARTIFACT_URI = f"repo://artifacts/shadow/{MODEL_VERSION}"
TRAINED_UNTIL = "2026-09-09T04:45:00Z"
ALPHA = 1.0
CLIP_LOW = 0.35
CLIP_HIGH = 2.5
FEEDBACK_METHOD = "ewma"
FEEDBACK_WINDOW = 8
TARGET_REPRESENTATION = "profile_ratio"


@dataclass(frozen=True)
class CandidateGArtifact:
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateGPrediction:
    prediction: CyclePrediction
    base_prediction: float
    calibration_factor: float
    feedback_samples: int
    data_cutoff: str


def build_candidate_g_artifact(*, version: str = MODEL_VERSION, artifact_uri: str = DEFAULT_ARTIFACT_URI) -> CandidateGArtifact:
    metadata = {
        "version": version,
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "artifact_uri": artifact_uri,
        "created_at": _now_iso(),
        "trained_until": TRAINED_UNTIL,
        "base_model": {
            "name": "candidate_f_f4_profile_ratio",
            "model_family": "HistGradientBoostingRegressor",
            "target_representation": TARGET_REPRESENTATION,
            "feature_schema": FEATURE_SCHEMA,
            "hyperparameters": HYPERPARAMETERS,
        },
        "adaptive_layer": {
            "name": "g2_resolved_error_feedback",
            "feedback_estimator": FEEDBACK_METHOD,
            "feedback_window": FEEDBACK_WINDOW,
            "alpha": ALPHA,
            "clip": [CLIP_LOW, CLIP_HIGH],
        },
        "data_contract": {
            "feedback_rows": "station_id,target_at,base_prediction,actual",
            "feedback_rule": "target_at <= data_cutoff only",
            "cold_start_factor": 1.0,
        },
        "git_commit": _git_commit(),
    }
    return CandidateGArtifact(metadata=metadata)


def save_candidate_g_artifact(artifact: CandidateGArtifact, *, artifact_uri: str = DEFAULT_ARTIFACT_URI) -> str:
    artifact_dir = _resolve_artifact_uri(artifact_uri)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps(artifact.metadata, indent=2, sort_keys=True) + "\n")
    return artifact_uri


def load_candidate_g_artifact(artifact_uri: str | Path) -> CandidateGArtifact:
    artifact_dir = _resolve_artifact_uri(str(artifact_uri))
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    validate_candidate_g_artifact(metadata)
    return CandidateGArtifact(metadata=metadata)


def validate_candidate_g_artifact(metadata: dict[str, Any]) -> None:
    if metadata.get("model_type") != MODEL_TYPE:
        raise ShadowCandidateError(f"unsupported candidate G model_type: {metadata.get('model_type')!r}")
    if metadata.get("version") != MODEL_VERSION:
        raise ShadowCandidateError(f"unsupported candidate G version: {metadata.get('version')!r}")
    layer = metadata.get("adaptive_layer")
    if not isinstance(layer, dict):
        raise ShadowCandidateError("candidate G artifact missing adaptive_layer")
    if layer.get("feedback_estimator") != FEEDBACK_METHOD or float(layer.get("alpha")) != ALPHA:
        raise ShadowCandidateError("candidate G adaptive parameters changed")
    if [float(value) for value in layer.get("clip", [])] != [CLIP_LOW, CLIP_HIGH]:
        raise ShadowCandidateError("candidate G clip parameters changed")


def build_candidate_g_predictions(
    cycle: ForecastCycle,
    model_id: str,
    artifact: CandidateGArtifact,
    observations: pd.DataFrame,
    *,
    feedback_frame: pd.DataFrame | None = None,
) -> list[CandidateGPrediction]:
    validate_candidate_g_artifact(artifact.metadata)
    frame = _prepare_observations(observations)
    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    if (frame["observed_at"] > cutoff).any():
        raise ShadowCandidateError("observations contain rows after data_cutoff")
    base_rows = build_candidate_f_base_frame(cycle, frame)
    feedback = prepare_feedback_frame(feedback_frame, cutoff) if feedback_frame is not None else empty_feedback_frame()
    outputs: list[CandidateGPrediction] = []
    for row in base_rows.itertuples(index=False):
        factor, count = g2_factor(str(row.station_id), feedback)
        final = max(0.0, float(row.base_prediction) * factor)
        if not math.isfinite(final):
            raise ShadowCandidateError("candidate G prediction is not finite")
        prediction = CyclePrediction(
            cycle_id=cycle.cycle_id,
            model_id=model_id,
            station_id=str(row.station_id),
            target_at=_ts(row.target_at),
            horizon_minutes=int(row.horizon_minutes),
            prediction=final,
        )
        outputs.append(
            CandidateGPrediction(
                prediction=prediction,
                base_prediction=float(row.base_prediction),
                calibration_factor=float(factor),
                feedback_samples=int(count),
                data_cutoff=_ts(cutoff),
            )
        )
    return outputs


def build_candidate_f_base_frame(cycle: ForecastCycle, observations: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    eligible = observations[observations["observed_at"] <= cutoff].copy()
    if eligible.empty:
        raise ShadowCandidateError("no observations at or before data_cutoff")
    supervised = _training_samples(eligible, cutoff)
    if supervised.empty:
        raise ShadowCandidateError("no supervised samples available for Candidate F base")
    station_mapping = {station_id: idx for idx, station_id in enumerate(sorted(supervised["station_id"].unique()))}
    inference = _candidate_f_inference_samples(cycle, eligible, station_mapping)
    raw_model = _fit_f4_profile_ratio(supervised, station_mapping)
    raw = raw_model.predict(_feature_matrix(inference, station_mapping))
    result = inference.copy()
    result["base_prediction"] = _reconstruct_profile_ratio(raw, result)
    return result


def _fit_f4_profile_ratio(samples: pd.DataFrame, station_mapping: dict[str, int]) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(**HYPERPARAMETERS)
    target = _profile_ratio_target(samples)
    model.fit(_feature_matrix(samples, station_mapping), target)
    return model


def _training_samples(observations: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    train_start = observations["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
    samples = _supervised_samples(observations)
    samples = samples[(samples["target_at"] <= cutoff) & (samples["data_cutoff"] >= train_start)].copy()
    return samples.dropna(subset=["profile_mean"])


def _candidate_f_inference_samples(cycle: ForecastCycle, observations: pd.DataFrame, station_mapping: dict[str, int]) -> pd.DataFrame:
    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    lookup = {(str(row.station_id), row.observed_at): float(row.demand) for row in observations[["station_id", "observed_at", "demand"]].itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for target in cycle.targets:
        station_id = str(target.station_id)
        if station_id not in station_mapping:
            raise ShadowCandidateError(f"station_id missing from station mapping: {station_id}")
        target_at = pd.Timestamp(target.target_at).tz_convert("UTC")
        row: dict[str, Any] = {
            "station_id": station_id,
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
        profile = observations[
            (observations["station_id"] == station_id)
            & (observations["quarter_hour_block"] == row["target_quarter_hour_block"])
            & (observations["observed_at"] <= cutoff)
        ]["demand"]
        row["profile_mean"] = float(profile.mean()) if not profile.empty else np.nan
        row["profile_median"] = float(profile.median()) if not profile.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("target_order").reset_index(drop=True)


def _feature_matrix(samples: pd.DataFrame, station_mapping: dict[str, int]) -> pd.DataFrame:
    frame = samples.copy()
    frame["station_code"] = frame["station_id"].map(station_mapping).astype(float)
    if frame["station_code"].isna().any():
        missing = sorted(frame.loc[frame["station_code"].isna(), "station_id"].unique())
        raise ShadowCandidateError(f"station_id missing from station mapping: {missing}")
    return frame[FEATURE_SCHEMA].astype(float)


def _profile_denominator(samples: pd.DataFrame) -> pd.Series:
    return samples["profile_mean"].fillna(samples["profile_median"]).fillna(samples["lag_96"]).fillna(samples["recent_mean"]).clip(lower=1e-3)


def _profile_ratio_target(samples: pd.DataFrame) -> np.ndarray:
    return (samples["actual"].astype(float) / _profile_denominator(samples)).clip(lower=0.0).to_numpy(dtype=float)


def _reconstruct_profile_ratio(raw_prediction: np.ndarray, samples: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, _profile_denominator(samples).to_numpy(dtype=float) * np.clip(np.asarray(raw_prediction, dtype=float), 0.0, None))


def prepare_feedback_frame(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    required = {"station_id", "target_at", "base_prediction", "actual"}
    missing = required - set(frame.columns)
    if missing:
        raise ShadowCandidateError(f"feedback frame missing columns: {sorted(missing)}")
    result = frame.copy()
    result["station_id"] = result["station_id"].astype(str)
    result["target_at"] = pd.to_datetime(result["target_at"], utc=True)
    result["base_prediction"] = result["base_prediction"].astype(float)
    result["actual"] = result["actual"].astype(float)
    result = result[result["target_at"] <= cutoff].copy()
    result = result[np.isfinite(result["base_prediction"]) & np.isfinite(result["actual"]) & (result["base_prediction"] > 0)]
    return result.sort_values(["station_id", "target_at"]).reset_index(drop=True)


def empty_feedback_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["station_id", "target_at", "base_prediction", "actual"])


def g2_factor(station_id: str, feedback: pd.DataFrame) -> tuple[float, int]:
    station_feedback = feedback[feedback["station_id"] == station_id].tail(FEEDBACK_WINDOW)
    if station_feedback.empty:
        return 1.0, 0
    ratios = (station_feedback["actual"] / station_feedback["base_prediction"].clip(lower=1e-3)).replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return 1.0, 0
    values = ratios.to_numpy(dtype=float)
    weights = np.geomspace(0.2, 1.0, num=len(values))
    raw = float(np.average(values, weights=weights))
    factor = float(np.exp(ALPHA * np.log(max(raw, 1e-3))))
    return float(np.clip(factor, CLIP_LOW, CLIP_HIGH)), int(len(values))


def candidate_g_trace_rows(predictions: list[CandidateGPrediction]) -> list[dict[str, Any]]:
    return [
        {
            "cycle_id": item.prediction.cycle_id,
            "model_id": item.prediction.model_id,
            "station_id": item.prediction.station_id,
            "target_observed_at": item.prediction.target_at,
            "horizon_minutes": item.prediction.horizon_minutes,
            "base_prediction": item.base_prediction,
            "calibration_factor": item.calibration_factor,
            "final_prediction": item.prediction.prediction,
            "feedback_samples": item.feedback_samples,
            "data_cutoff": item.data_cutoff,
        }
        for item in predictions
    ]


def save_candidate_g_model_version_json(path: Path, *, artifact_uri: str = DEFAULT_ARTIFACT_URI) -> dict[str, Any]:
    artifact = build_candidate_g_artifact(artifact_uri=artifact_uri)
    save_candidate_g_artifact(artifact, artifact_uri=artifact_uri)
    row = {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "artifact_uri": artifact_uri,
        "trained_until": TRAINED_UNTIL,
        "feature_config": artifact.metadata,
        "params": artifact.metadata["adaptive_layer"],
        "is_active": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
    return row


def _resolve_artifact_uri(artifact_uri: str) -> Path:
    if artifact_uri.startswith("repo://"):
        return Path(__file__).resolve().parents[1] / artifact_uri.removeprefix("repo://")
    return Path(artifact_uri)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
