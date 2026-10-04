"""Candidate L v1: a frozen, per-cycle harmonic regression for the final regime.

Fits real observations only, never imputed labels. Frequency and regression weights
are re-estimated at each data_cutoff; all training provenance is returned. The spec
was selected on development data, so prospective evidence is required for promotion.
No database, credentials, submission API, or mutable global state.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MODEL_NAME = "candidate_l_v1_recent_harmonic"
MODEL_TYPE = "adaptive_harmonic_candidate_l"
VERSION = "candidate_l_v1"
REGIME_START = "2026-09-20T12:00:00Z"


@dataclass(frozen=True)
class Config:
    window: int = 40
    min_rows: int = 32
    period_min: float = 8.0
    period_max: float = 64.0
    period_points: int = 225
    harmonics: int = 2
    ridge: float = 0.05
    decay_slots: float = 64.0
    cap_multiple: float = 2.0


def _utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return stamp.tz_convert("UTC")


def _design(t, periods, config):
    columns = [np.ones((len(periods), len(t)))]
    for harmonic in range(1, config.harmonics + 1):
        angle = 2 * np.pi * harmonic * t[None, :] / periods[:, None]
        columns.extend([np.sin(angle), np.cos(angle)])
    return np.stack(columns, axis=-1)


def fit_station(history, target_times, *, config=Config()):
    if len(history) < config.min_rows:
        raise ValueError("insufficient real post-change observations")
    history = history.sort_values("observed_at").tail(config.window)
    origin = history.observed_at.iloc[-1]
    t = ((history.observed_at - origin).dt.total_seconds() / 900).to_numpy()
    future = ((pd.DatetimeIndex(target_times) - origin).total_seconds() / 900).to_numpy()
    y = history.demand.to_numpy(dtype=float)
    scale = max(float(y.mean()), 1.0)
    periods = np.linspace(config.period_min, config.period_max, config.period_points)
    design = _design(t, periods, config)
    weights = np.exp(t / config.decay_slots)
    regularizer = np.eye(design.shape[-1]) * config.ridge
    regularizer[0, 0] = 1e-6
    lhs = np.einsum("pti,ptj,t->pij", design, design, weights) + regularizer
    rhs = np.einsum("pti,t,t->pi", design, y / scale, weights)
    coefficients = np.linalg.solve(lhs, rhs[..., None])[..., 0]
    fitted = np.einsum("pti,pi->pt", design, coefficients)
    best = int(np.argmin(np.mean((fitted - y / scale) ** 2, axis=1)))
    forecast = _design(future, periods[best:best + 1], config)[0] @ coefficients[best] * scale
    forecast = np.clip(forecast, 0, config.cap_multiple * float(y.max()))
    manifest = {
        "rows": len(history), "training_start": history.observed_at.min().isoformat(),
        "training_end": history.observed_at.max().isoformat(), "scale": scale,
        "period_slots": float(periods[best]), "coefficients": coefficients[best].tolist(),
        "training_sha256": hashlib.sha256(history[["observed_at", "demand"]].to_csv(index=False).encode()).hexdigest(),
    }
    return forecast, manifest


def predict(observations, data_cutoff, targets, *, available_at=None, config=Config()):
    cutoff = _utc(data_cutoff)
    frame = observations[["station_id", "observed_at", "demand"] + (["ingested_at"] if "ingested_at" in observations else [])].copy()
    frame["observed_at"] = pd.to_datetime(frame.observed_at, utc=True)
    frame = frame[(frame.observed_at > pd.Timestamp(REGIME_START)) & (frame.observed_at <= cutoff)]
    if available_at is not None:
        if "ingested_at" not in frame:
            raise ValueError("real-time replay requires observation availability timestamps")
        frame["ingested_at"] = pd.to_datetime(frame.ingested_at, utc=True)
        frame = frame[frame.ingested_at <= _utc(available_at)]
    frame["station_id"] = frame.station_id.astype(str)
    frame["demand"] = pd.to_numeric(frame.demand, errors="raise")
    frame = frame[frame.demand.notna()]
    if not np.isfinite(frame.demand).all() or (frame.demand < 0).any():
        raise ValueError("invalid observation value")
    if frame.duplicated(["station_id", "observed_at"]).any():
        raise ValueError("duplicate observations")
    wanted = targets[["station_id", "target_at", "horizon_minutes"]].copy()
    wanted["station_id"] = wanted.station_id.astype(str)
    wanted["target_at"] = pd.to_datetime(wanted.target_at, utc=True)
    if wanted.empty or wanted.duplicated(["station_id", "target_at"]).any():
        raise ValueError("empty or duplicate target set")
    expected = cutoff + pd.to_timedelta(wanted.horizon_minutes, unit="m")
    if not wanted.horizon_minutes.isin([15, 30, 45, 60]).all() or not (wanted.target_at == expected).all():
        raise ValueError("targets must match horizons and cutoff exactly")
    if not wanted.station_id.str.fullmatch(r"\d{5}").all():
        raise ValueError("station IDs must have five digits")
    wanted["prediction"] = np.nan
    manifest = {"version": VERSION, "config": asdict(config), "data_cutoff": cutoff.isoformat(), "stations": {}}
    for station, group in wanted.groupby("station_id", sort=True):
        history = frame[frame.station_id == station].sort_values("observed_at")
        if history.empty or history.observed_at.max() < cutoff - pd.Timedelta(minutes=30):
            raise ValueError(f"stale station history: {station}")
        values, provenance = fit_station(history, group.target_at, config=config)
        wanted.loc[group.index, "prediction"] = values
        manifest["stations"][station] = provenance
    manifest["weights_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if not np.isfinite(wanted.prediction).all():
        raise ValueError("nonfinite predictions")
    return wanted.reset_index(drop=True), manifest
