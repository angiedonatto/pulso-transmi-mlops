from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from candidate_b_shadow import (
    FEATURE_SCHEMA,
    HORIZONS,
    HYPERPARAMETERS,
    MIN_HISTORY_DAYS,
    ShadowCandidateError,
    _prepare_observations,
    _supervised_samples,
    _ts,
)
from candidate_g_adaptive import (
    _candidate_f_inference_samples,
    _feature_matrix,
    _fit_f4_profile_ratio,
    _reconstruct_profile_ratio,
    build_candidate_f_base_frame,
)
from cycle_predictions import CyclePrediction
from forecast_cycle_sync import CycleTarget, ForecastCycle


# Candidate I = Candidate F base x significance-gated resolved-error level correction.
#
#   final = F_base * clip(exp(gate * log(level_ratio)), CLIP_LOW, CLIP_HIGH)
#
# - F_base: production Candidate F (`build_candidate_f_base_frame`), unchanged.
# - History: out-of-sample Candidate F predictions for the resolved slots of the last
#   HISTORY_HOURS, with F refit once per UTC day (bucket = floor(history cutoff, 1 day),
#   trained on target_at <= bucket). Rebuilt from observations only, so it does not depend
#   on which cycles were served, has no cold start after a restart, and never reads a slot
#   after data_cutoff.
# - level_ratio: the G2 estimator (geomspace(0.2, 1) weighted mean of the last
#   LEVEL_WINDOW per-slot ratios actual / base) on that history.
# - gate = z^2 / (z^2 + KAPPA), z = |mean(log r, last LEVEL_WINDOW slots)| /
#   (sd(log r, last NOISE_WINDOW slots) / sqrt(LEVEL_WINDOW)): a persistent residual
#   shift is corrected, residual noise is not (gate -> 0, factor -> 1).
# Any failure of the adaptive layer falls back to factor 1.0 (= Candidate F).
MODEL_VERSION = "candidate_i_v1"
MODEL_NAME = "candidate_i_f4_oos_g2_snr_gate"
MODEL_TYPE = "shadow_f4_oos_g2_snr_gate"
DEFAULT_ARTIFACT_URI = f"repo://artifacts/shadow/{MODEL_VERSION}"
TRAINED_UNTIL = "2026-09-13T11:00:00Z"  # parameters selected on cycles <= this cutoff
TARGET_REPRESENTATION = "profile_ratio"
LEVEL_WINDOW = 8
NOISE_WINDOW = 32
KAPPA = 4.0
CLIP_LOW = 0.35
CLIP_HIGH = 2.5
HISTORY_HOURS = 24
HISTORY_REFIT = "1D"
MIN_LOG_SAMPLES = 3
SLOT = pd.Timedelta(minutes=15)
HISTORY_COLUMNS = ["station_id", "target_at", "base_prediction", "actual", "history_cutoff", "model_bucket"]


@dataclass(frozen=True)
class CandidateIArtifact:
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LevelSignal:
    level_ratio: float
    log_mean: float
    log_sd: float
    z: float
    gate: float
    factor: float
    history_slots: int
    history_max_target_at: str | None
    fallback_reason: str | None = None


@dataclass(frozen=True)
class CandidateIPrediction:
    prediction: CyclePrediction
    base_prediction: float
    signal: LevelSignal
    data_cutoff: str


def build_candidate_i_artifact(*, version: str = MODEL_VERSION, artifact_uri: str = DEFAULT_ARTIFACT_URI) -> CandidateIArtifact:
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
            "refit": "every cycle, target_at <= data_cutoff",
        },
        "history": {
            "name": "out_of_sample_daily_refit_f",
            "hours": HISTORY_HOURS,
            "refit": HISTORY_REFIT,
            "rule": "history slot predicted by F fit on target_at <= floor(history_cutoff, 1 day)",
        },
        "adaptive_layer": {
            "name": "snr_gated_level",
            "level_estimator": "g2_geomspace_0.2_1",
            "level_window": LEVEL_WINDOW,
            "noise_window": NOISE_WINDOW,
            "kappa": KAPPA,
            "clip": [CLIP_LOW, CLIP_HIGH],
        },
        "data_contract": {
            "inputs": "observations <= data_cutoff only",
            "fallback_factor": 1.0,
        },
        "git_commit": _git_commit(),
    }
    return CandidateIArtifact(metadata=metadata)


def save_candidate_i_artifact(artifact: CandidateIArtifact, *, artifact_uri: str = DEFAULT_ARTIFACT_URI) -> str:
    artifact_dir = _resolve_artifact_uri(artifact_uri)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metadata.json").write_text(json.dumps(artifact.metadata, indent=2, sort_keys=True) + "\n")
    return artifact_uri


def load_candidate_i_artifact(artifact_uri: str | Path) -> CandidateIArtifact:
    artifact_dir = _resolve_artifact_uri(str(artifact_uri))
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    validate_candidate_i_artifact(metadata)
    return CandidateIArtifact(metadata=metadata)


def validate_candidate_i_artifact(metadata: dict[str, Any]) -> None:
    if metadata.get("model_type") != MODEL_TYPE:
        raise ShadowCandidateError(f"unsupported candidate I model_type: {metadata.get('model_type')!r}")
    if metadata.get("version") != MODEL_VERSION:
        raise ShadowCandidateError(f"unsupported candidate I version: {metadata.get('version')!r}")
    layer = metadata.get("adaptive_layer")
    if not isinstance(layer, dict):
        raise ShadowCandidateError("candidate I artifact missing adaptive_layer")
    frozen = (int(layer.get("level_window", -1)), int(layer.get("noise_window", -1)), float(layer.get("kappa", -1)))
    if frozen != (LEVEL_WINDOW, NOISE_WINDOW, KAPPA):
        raise ShadowCandidateError("candidate I adaptive parameters changed")
    if [float(value) for value in layer.get("clip", [])] != [CLIP_LOW, CLIP_HIGH]:
        raise ShadowCandidateError("candidate I clip parameters changed")


def build_candidate_i_predictions(
    cycle: ForecastCycle,
    model_id: str,
    artifact: CandidateIArtifact,
    observations: pd.DataFrame,
) -> list[CandidateIPrediction]:
    validate_candidate_i_artifact(artifact.metadata)
    frame = _prepare_observations(observations)
    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    if (frame["observed_at"] > cutoff).any():
        raise ShadowCandidateError("observations contain rows after data_cutoff")
    base_rows = build_candidate_f_base_frame(cycle, frame)
    stations = sorted({str(target.station_id) for target in cycle.targets})
    signals = level_signals_for_cycle(frame, cutoff, stations)
    outputs: list[CandidateIPrediction] = []
    for row in base_rows.itertuples(index=False):
        signal = signals[str(row.station_id)]
        final = max(0.0, float(row.base_prediction) * signal.factor)
        if not math.isfinite(final):
            raise ShadowCandidateError("candidate I prediction is not finite")
        prediction = CyclePrediction(
            cycle_id=cycle.cycle_id,
            model_id=model_id,
            station_id=str(row.station_id),
            target_at=_ts(row.target_at),
            horizon_minutes=int(row.horizon_minutes),
            prediction=final,
        )
        outputs.append(CandidateIPrediction(prediction=prediction, base_prediction=float(row.base_prediction), signal=signal, data_cutoff=_ts(cutoff)))
    return outputs


def level_signals_for_cycle(prepared: pd.DataFrame, cutoff: pd.Timestamp, stations: list[str]) -> dict[str, LevelSignal]:
    """Adaptive layer. Fail-safe: any error -> factor 1.0 (Candidate F) for every station."""
    try:
        history = build_oos_history(prepared, cutoff, stations)
    except Exception as exc:  # noqa: BLE001 - the base prediction must still be served
        reason = f"history_failed: {type(exc).__name__}: {exc}"
        return {sid: neutral_signal(reason) for sid in stations}
    return {sid: level_signal(history[history["station_id"] == sid], cutoff) for sid in stations}


def build_oos_history(prepared: pd.DataFrame, cutoff: pd.Timestamp, stations: list[str], *, hours: int = HISTORY_HOURS) -> pd.DataFrame:
    """Out-of-sample F predictions for the slots of the last `hours` hourly cycles before
    `cutoff` (all targets <= cutoff). The model for history cycle c is fit on
    target_at <= floor(c, 1 day), so every slot is predicted without seeing itself."""
    prepared = prepared[prepared["observed_at"] <= cutoff]
    last = cutoff.floor("h") - pd.Timedelta(hours=1)
    history_cutoffs = pd.date_range(last - pd.Timedelta(hours=hours - 1), last, freq="h")
    supervised = _supervised_samples(prepared)
    train_start = prepared["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
    mapping = {sid: idx for idx, sid in enumerate(sorted(prepared["station_id"].unique()))}
    actuals = prepared.drop_duplicates(["station_id", "observed_at"], keep="last").set_index(["station_id", "observed_at"])["demand"]
    models: dict[pd.Timestamp, Any] = {}
    frames = []
    for history_cutoff in history_cutoffs:
        bucket = history_cutoff.floor("24h")
        if bucket not in models:
            train = supervised[(supervised["target_at"] <= bucket) & (supervised["data_cutoff"] >= train_start)].dropna(subset=["profile_mean"])
            models[bucket] = _fit_f4_profile_ratio(train, mapping) if len(train) else None
        model = models[bucket]
        if model is None:
            continue
        inference = _candidate_f_inference_samples(_history_cycle(history_cutoff, stations), prepared[prepared["observed_at"] <= history_cutoff], mapping)
        base = _reconstruct_profile_ratio(model.predict(_feature_matrix(inference, mapping)), inference)
        target_at = pd.to_datetime(inference["target_at"], utc=True)
        frames.append(
            pd.DataFrame(
                {
                    "station_id": inference["station_id"].astype(str).to_numpy(),
                    "target_at": target_at.to_numpy(),
                    "base_prediction": base,
                    "actual": actuals.reindex(pd.MultiIndex.from_arrays([inference["station_id"].astype(str), target_at])).to_numpy(),
                    "history_cutoff": history_cutoff,
                    "model_bucket": bucket,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    history = pd.concat(frames, ignore_index=True)
    history["target_at"] = pd.to_datetime(history["target_at"], utc=True)
    return history[history["target_at"] <= cutoff].sort_values(["station_id", "target_at"]).reset_index(drop=True)


def level_signal(history: pd.DataFrame, cutoff: pd.Timestamp) -> LevelSignal:
    """Gated level factor for one station from its resolved out-of-sample history."""
    h = history[history["target_at"] <= cutoff]
    actual = h["actual"].to_numpy(dtype=float)
    base = h["base_prediction"].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(base) & (base > 0)
    h, actual, base = h[valid], actual[valid], base[valid]
    if len(h) == 0:
        return neutral_signal("no_resolved_history")
    target_at = h["target_at"]
    last = slice(max(0, len(h) - LEVEL_WINDOW), len(h))
    ratios = actual[last] / np.clip(base[last], 1e-3, None)
    level_ratio = float(np.average(ratios, weights=np.geomspace(0.2, 1.0, num=len(ratios))))
    log_ratio = np.log(np.clip(actual, 1e-3, None) / np.clip(base, 1e-3, None))
    recent = (target_at > cutoff - LEVEL_WINDOW * SLOT).to_numpy()
    noise_window = (target_at > cutoff - NOISE_WINDOW * SLOT).to_numpy()
    log_mean = float(log_ratio[recent].mean()) if recent.sum() >= MIN_LOG_SAMPLES else math.nan
    log_sd = float(log_ratio[noise_window].std(ddof=1)) if noise_window.sum() >= MIN_LOG_SAMPLES else math.nan
    noise = log_sd / math.sqrt(LEVEL_WINDOW)
    z = abs(log_mean) / noise if math.isfinite(noise) and noise > 0 and math.isfinite(log_mean) else 0.0
    gate = z * z / (z * z + KAPPA)
    log_level = math.log(max(level_ratio, 1e-3)) if math.isfinite(level_ratio) and level_ratio > 0 else 0.0
    correction = gate * log_level
    factor = float(np.clip(math.exp(correction), CLIP_LOW, CLIP_HIGH)) if math.isfinite(correction) else 1.0
    return LevelSignal(
        level_ratio=level_ratio,
        log_mean=log_mean,
        log_sd=log_sd,
        z=float(z),
        gate=float(gate),
        factor=factor,
        history_slots=int(noise_window.sum()),
        history_max_target_at=_ts(target_at.max()),
    )


def neutral_signal(reason: str) -> LevelSignal:
    return LevelSignal(math.nan, math.nan, math.nan, 0.0, 0.0, 1.0, 0, None, reason)


def candidate_i_trace(item: CandidateIPrediction) -> dict[str, Any]:
    s = item.signal
    return {
        "candidate": MODEL_VERSION,
        "base_prediction": item.base_prediction,
        "calibration_factor": s.factor,
        "final_prediction": item.prediction.prediction,
        "level_ratio": _finite_or_none(s.level_ratio),
        "snr_z": s.z,
        "gate": s.gate,
        "history_slots": s.history_slots,
        "history_max_target_at": s.history_max_target_at,
        "fallback_reason": s.fallback_reason,
        "data_cutoff": item.data_cutoff,
    }


def save_candidate_i_model_version_json(path: Path, *, artifact_uri: str = DEFAULT_ARTIFACT_URI) -> dict[str, Any]:
    artifact = build_candidate_i_artifact(artifact_uri=artifact_uri)
    save_candidate_i_artifact(artifact, artifact_uri=artifact_uri)
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


def _history_cycle(cutoff: pd.Timestamp, stations: list[str]) -> ForecastCycle:
    targets = [
        CycleTarget(station_id=sid, target_at=_ts(cutoff + pd.Timedelta(minutes=h)), horizon_minutes=int(h), target_order=i * len(HORIZONS) + j, raw_target={})
        for i, sid in enumerate(sorted(stations))
        for j, h in enumerate(HORIZONS)
    ]
    return ForecastCycle(cycle_id=f"history_{cutoff.strftime('%Y%m%dT%H%M%SZ')}", data_cutoff=_ts(cutoff), closes_at="", expected_predictions=len(targets), targets=targets, raw_cycle={})


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _resolve_artifact_uri(artifact_uri: str) -> Path:
    if artifact_uri.startswith("repo://"):
        return Path(__file__).resolve().parents[1] / artifact_uri.removeprefix("repo://")
    return Path(artifact_uri)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
