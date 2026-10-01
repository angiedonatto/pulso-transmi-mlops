"""Candidate I research core: data snapshot, exact hourly cycle replay, scoring.

Everything here is research infrastructure. Production code is only *imported* (never
modified) so that replayed B/F/G predictions come from the exact production code paths:

- `candidate_b_shadow._prepare_observations / _supervised_samples / FEATURE_SCHEMA /
  HYPERPARAMETERS` for features and the HGB configuration.
- `candidate_g_adaptive._candidate_f_inference_samples / _profile_ratio_target /
  _reconstruct_profile_ratio / prepare_feedback_frame / g2_factor` for F and G.
- `drift_monitor._accuracy_by` for the canonical per-station accuracy.

Speed trick (verified by `verify_fast_base_parity`): production refits F at every cycle
by rebuilding `_supervised_samples(observations <= cutoff)`. Every feature of a training
row depends only on data at or before that row's own `data_cutoff` (per-station shifts
and `shift(1).expanding()` profiles), so building the supervised frame ONCE on the full
snapshot and filtering `target_at <= cutoff` yields byte-identical training rows in the
same order. The parity check asserts identical predictions against the production
function on real cutoffs before any result is trusted.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from candidate_b_shadow import (  # noqa: E402
    FEATURE_SCHEMA,
    HORIZONS,
    HYPERPARAMETERS,
    MIN_HISTORY_DAYS,
    _prepare_observations,
    _supervised_samples,
)
from candidate_g_adaptive import (  # noqa: E402
    _candidate_f_inference_samples,
    _profile_ratio_target,
    _reconstruct_profile_ratio,
    build_candidate_f_base_frame,
    g2_factor,
    prepare_feedback_frame,
)
from drift_monitor import _accuracy_by  # noqa: E402
from forecast_cycle_sync import CycleTarget, ForecastCycle  # noqa: E402

SNAPSHOT_DIR = ROOT / "data" / "raw" / "candidate_i"
INTERIM_DIR = ROOT / "data" / "interim" / "candidate_i"
SNAPSHOT_TABLES = ("observations", "forecast_cycles", "forecast_cycle_targets", "predictions", "model_versions", "submission_receipts")

EVAL_START = pd.Timestamp("2026-09-01T00:00:00Z")
DRIFT_START = pd.Timestamp("2026-09-10T18:00:00Z")  # first official cycle
MAX_HORIZON = max(HORIZONS)


# --------------------------------------------------------------------------------------
# Snapshot (read-only)
# --------------------------------------------------------------------------------------
def refresh_snapshot() -> dict[str, Any]:
    """Read-only GETs against Supabase via the drift monitor's read-only client."""
    from run_drift_monitor import TABLE_SELECTS, ReadOnlySupabaseClient

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    orders = {
        "observations": "observed_at.asc,station_id.asc",
        "predictions": "created_at.asc",
        "submission_receipts": "created_at.asc",
        "forecast_cycles": "data_cutoff.asc",
        "forecast_cycle_targets": "cycle_id.asc,target_order.asc",
        "model_versions": "trained_until.asc",
    }
    summary: dict[str, Any] = {"fetched_at": pd.Timestamp.now(tz="UTC").isoformat()}
    with ReadOnlySupabaseClient() as client:
        for name in SNAPSHOT_TABLES:
            frame = client.table(name, select=TABLE_SELECTS[name], order=orders[name])
            frame.to_csv(SNAPSHOT_DIR / f"{name}.csv", index=False)
            summary[name] = int(len(frame))
    (SNAPSHOT_DIR / "snapshot_meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def normalize_station(value: Any) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def load_snapshot() -> dict[str, pd.DataFrame]:
    if not (SNAPSHOT_DIR / "observations.csv").exists():
        raise SystemExit(f"No snapshot at {SNAPSHOT_DIR}; run `python3 scripts/candidate_i_lab.py snapshot` first.")
    tables = {name: pd.read_csv(SNAPSHOT_DIR / f"{name}.csv", dtype={"station_id": str}) for name in SNAPSHOT_TABLES}
    obs = tables["observations"]
    obs["station_id"] = obs["station_id"].map(normalize_station)
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    obs["demand"] = obs["demand"].astype(float)
    tables["observations"] = obs.sort_values(["station_id", "observed_at"]).reset_index(drop=True)
    fc = tables["forecast_cycles"]
    fc["data_cutoff"] = pd.to_datetime(fc["data_cutoff"], utc=True)
    tgt = tables["forecast_cycle_targets"]
    tgt["station_id"] = tgt["station_id"].map(normalize_station)
    tgt["target_at"] = pd.to_datetime(tgt["target_at"], utc=True)
    pred = tables["predictions"]
    pred["station_id"] = pred["station_id"].map(normalize_station)
    pred["target_observed_at"] = pd.to_datetime(pred["target_observed_at"], utc=True)
    return tables


# --------------------------------------------------------------------------------------
# Cycles
# --------------------------------------------------------------------------------------
def last_resolvable_cutoff(observations: pd.DataFrame) -> pd.Timestamp:
    latest = observations["observed_at"].max() - pd.Timedelta(minutes=MAX_HORIZON)
    return latest.floor("h")


def evaluation_cutoffs(observations: pd.DataFrame, start: pd.Timestamp = EVAL_START) -> list[pd.Timestamp]:
    end = last_resolvable_cutoff(observations)
    return list(pd.date_range(start, end, freq="h", tz="UTC"))


def official_cycle_ids(tables: dict[str, pd.DataFrame]) -> dict[pd.Timestamp, str]:
    fc = tables["forecast_cycles"]
    return {pd.Timestamp(row.data_cutoff): str(row.cycle_id) for row in fc.itertuples(index=False)}


def build_cycle(cutoff: pd.Timestamp, stations: list[str], cycle_id: str | None = None) -> ForecastCycle:
    targets = []
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
    cid = cycle_id or f"sim_{cutoff.strftime('%Y%m%dT%H%M%SZ')}"
    return ForecastCycle(
        cycle_id=cid,
        data_cutoff=cutoff.isoformat().replace("+00:00", "Z"),
        closes_at="",
        expected_predictions=len(targets),
        targets=targets,
        raw_cycle={},
    )


# --------------------------------------------------------------------------------------
# Exact walk-forward base replay (B_wf raw target, F profile_ratio target), hourly refit
# --------------------------------------------------------------------------------------
@dataclass
class SupervisedCache:
    prepared: pd.DataFrame
    supervised: pd.DataFrame
    train_start: pd.Timestamp
    station_mapping: dict[str, int]


def build_supervised_cache(observations: pd.DataFrame) -> SupervisedCache:
    prepared = _prepare_observations(observations)
    supervised = _supervised_samples(prepared)
    train_start = prepared["observed_at"].min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
    mapping = {sid: idx for idx, sid in enumerate(sorted(prepared["station_id"].unique()))}
    return SupervisedCache(prepared=prepared, supervised=supervised, train_start=train_start, station_mapping=mapping)


def training_rows(cache: SupervisedCache, cutoff: pd.Timestamp) -> pd.DataFrame:
    s = cache.supervised
    rows = s[(s["target_at"] <= cutoff) & (s["data_cutoff"] >= cache.train_start)]
    return rows.dropna(subset=["profile_mean"])


def feature_matrix(samples: pd.DataFrame, mapping: dict[str, int], schema: list[str] = FEATURE_SCHEMA) -> pd.DataFrame:
    frame = samples.copy()
    frame["station_code"] = frame["station_id"].map(mapping).astype(float)
    return frame[schema].astype(float)


def inference_rows(cache: SupervisedCache, cutoff: pd.Timestamp, cycle: ForecastCycle) -> pd.DataFrame:
    eligible = cache.prepared[cache.prepared["observed_at"] <= cutoff]
    return _candidate_f_inference_samples(cycle, eligible, cache.station_mapping)


def fit_hgb(X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None, **overrides: Any) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = {**HYPERPARAMETERS, **overrides}
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def replay_bases(
    tables: dict[str, pd.DataFrame],
    cutoffs: list[pd.Timestamp],
    *,
    out_path: Path,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Hourly-exact walk-forward replay of B_wf and F. Resumable: already-computed cutoffs
    in `out_path` are skipped."""
    observations = tables["observations"]
    stations = sorted(observations["station_id"].unique())
    official = official_cycle_ids(tables)
    cache = build_supervised_cache(observations)
    done: dict[pd.Timestamp, pd.DataFrame] = {}
    if out_path.exists():
        previous = pd.read_pickle(out_path)
        for cutoff, group in previous.groupby("data_cutoff"):
            done[pd.Timestamp(cutoff)] = group
    actual_lookup = observations.set_index(["station_id", "observed_at"])["demand"]
    todo = [c for c in cutoffs if c not in done]
    log(f"[bases] {len(done)} cached, {len(todo)} to compute")
    started = time.perf_counter()
    for idx, cutoff in enumerate(todo, start=1):
        t0 = time.perf_counter()
        cycle = build_cycle(cutoff, stations, official.get(cutoff))
        train = training_rows(cache, cutoff)
        X = feature_matrix(train, cache.station_mapping)
        inf = inference_rows(cache, cutoff, cycle)
        X_inf = feature_matrix(inf, cache.station_mapping)
        model_b = fit_hgb(X, train["actual"].to_numpy(dtype=float))
        model_f = fit_hgb(X, _profile_ratio_target(train))
        frame = inf.copy()
        frame["pred_B"] = np.maximum(0.0, model_b.predict(X_inf))
        frame["pred_F"] = _reconstruct_profile_ratio(model_f.predict(X_inf), inf)
        frame["data_cutoff"] = cutoff
        frame["cycle_id"] = cycle.cycle_id
        frame["is_official"] = cutoff in official
        keys = list(zip(frame["station_id"], pd.to_datetime(frame["target_at"], utc=True)))
        frame["actual"] = [float(actual_lookup.get(k, np.nan)) for k in keys]
        frame["train_rows"] = len(train)
        done[cutoff] = frame
        if idx % 6 == 0 or idx == len(todo):
            pd.concat(done.values(), ignore_index=True).to_pickle(out_path)
            elapsed = time.perf_counter() - started
            log(f"[bases] {idx}/{len(todo)} cutoff={cutoff.isoformat()} last={time.perf_counter()-t0:.1f}s eta={elapsed/idx*(len(todo)-idx)/60:.1f}min")
    result = pd.concat(done.values(), ignore_index=True)
    result["target_at"] = pd.to_datetime(result["target_at"], utc=True)
    result.to_pickle(out_path)
    return result


def verify_fast_base_parity(tables: dict[str, pd.DataFrame], cutoffs: list[pd.Timestamp]) -> list[dict[str, Any]]:
    """Asserts that the cached fast path equals production `build_candidate_f_base_frame`."""
    observations = tables["observations"]
    stations = sorted(observations["station_id"].unique())
    cache = build_supervised_cache(observations)
    results = []
    for cutoff in cutoffs:
        cycle = build_cycle(cutoff, stations)
        production = build_candidate_f_base_frame(cycle, observations[observations["observed_at"] <= cutoff].copy())
        train = training_rows(cache, cutoff)
        inf = inference_rows(cache, cutoff, cycle)
        model_f = fit_hgb(feature_matrix(train, cache.station_mapping), _profile_ratio_target(train))
        fast = _reconstruct_profile_ratio(model_f.predict(feature_matrix(inf, cache.station_mapping)), inf)
        max_diff = float(np.max(np.abs(production["base_prediction"].to_numpy() - fast)))
        results.append({"cutoff": cutoff.isoformat(), "max_abs_diff": max_diff, "identical": max_diff < 1e-9})
    return results


# --------------------------------------------------------------------------------------
# Base-model variants (research): one change at a time vs F, refit every `cadence_hours`
# --------------------------------------------------------------------------------------
RATIO_FEATURES = ["lvl4", "lvl16", "d1r", "w1r"]


def ratio_feature_table(prepared: pd.DataFrame) -> pd.DataFrame:
    """Per (station, observed_at) scale-free level features, using only rows <= observed_at.

    profile_at(t) = mean of same-slot observations strictly before t (identical to the
    `profile_mean` production computes for a row whose target_at is t). All features are
    ratios of actual demand to that profile, so they are invariant to station scale."""
    base = prepared.sort_values(["station_id", "observed_at"]).copy()
    by_slot = base.groupby(["station_id", "quarter_hour_block"], sort=False)["demand"]
    base["profile_at"] = by_slot.transform(lambda v: v.shift(1).expanding().mean())
    g = base.groupby("station_id", sort=False)
    a = base["demand"]
    p = base["profile_at"]
    for k, name in ((4, "lvl4"), (16, "lvl16")):
        num = g["demand"].transform(lambda v: v.rolling(k, min_periods=k).sum())
        den = p.groupby(base["station_id"]).transform(lambda v: v.rolling(k, min_periods=k).sum())
        base[name] = num / den.clip(lower=1e-3)
    base["ratio_now"] = a / p.clip(lower=1e-3)
    return base[["station_id", "observed_at", "lvl4", "lvl16", "ratio_now"]]


def attach_ratio_features(samples: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """lvl* at the row's data_cutoff; d1r/w1r = ratio_now at target_at - 1 day / - 7 days."""
    t = table.set_index(["station_id", "observed_at"])
    out = samples.copy()
    key_c = pd.MultiIndex.from_arrays([out["station_id"], pd.to_datetime(out["data_cutoff"], utc=True)])
    for col in ("lvl4", "lvl16"):
        out[col] = t[col].reindex(key_c).to_numpy()
    tgt = pd.to_datetime(out["target_at"], utc=True)
    for col, lag in (("d1r", pd.Timedelta(days=1)), ("w1r", pd.Timedelta(days=7))):
        key = pd.MultiIndex.from_arrays([out["station_id"], tgt - lag])
        out[col] = t["ratio_now"].reindex(key).to_numpy()
    return out


@dataclass(frozen=True)
class BaseVariant:
    name: str
    target: str = "profile_ratio"  # "profile_ratio" | "raw"
    wape_weights: bool = False
    ratio_features: bool = False
    hgb_overrides: tuple[tuple[str, Any], ...] = ()

    def schema(self) -> list[str]:
        return FEATURE_SCHEMA + (RATIO_FEATURES if self.ratio_features else [])


def _variant_weights(train: pd.DataFrame) -> np.ndarray:
    denom = train["profile_mean"].fillna(train["profile_median"]).fillna(train["lag_96"]).fillna(train["recent_mean"]).clip(lower=1e-3)
    volume = train.groupby("station_id")["actual"].transform("mean").clip(lower=1e-3)
    return (denom / volume).to_numpy(dtype=float)


def replay_base_variant(
    tables: dict[str, pd.DataFrame],
    cutoffs: list[pd.Timestamp],
    variant: BaseVariant,
    *,
    cadence_hours: int = 6,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    observations = tables["observations"]
    stations = sorted(observations["station_id"].unique())
    official = official_cycle_ids(tables)
    cache = build_supervised_cache(observations)
    supervised = cache.supervised
    rtable = ratio_feature_table(cache.prepared) if variant.ratio_features else None
    if rtable is not None:
        supervised = attach_ratio_features(supervised, rtable)
    schema = variant.schema()
    actual_lookup = observations.set_index(["station_id", "observed_at"])["demand"]
    frames = []
    buckets = sorted({c.floor(f"{cadence_hours}h") for c in cutoffs})
    started = time.perf_counter()
    for bi, bucket in enumerate(buckets, start=1):
        train = supervised[(supervised["target_at"] <= bucket) & (supervised["data_cutoff"] >= cache.train_start)].dropna(subset=["profile_mean"])
        X = feature_matrix(train, cache.station_mapping, schema)
        if variant.target == "raw":
            y = train["actual"].to_numpy(dtype=float)
        else:
            y = _profile_ratio_target(train)
        weights = _variant_weights(train) if variant.wape_weights else None
        model = fit_hgb(X, y, sample_weight=weights, **dict(variant.hgb_overrides))
        for c in [c for c in cutoffs if bucket <= c < bucket + pd.Timedelta(hours=cadence_hours)]:
            cycle = build_cycle(c, stations, official.get(c))
            inf = inference_rows(cache, c, cycle)
            inf["data_cutoff"] = c
            if rtable is not None:
                inf = attach_ratio_features(inf, rtable[rtable["observed_at"] <= c])
            raw = model.predict(feature_matrix(inf, cache.station_mapping, schema))
            pred = np.maximum(0.0, raw) if variant.target == "raw" else _reconstruct_profile_ratio(raw, inf)
            tgt = pd.to_datetime(inf["target_at"], utc=True)
            actual = actual_lookup.reindex(pd.MultiIndex.from_arrays([inf["station_id"], tgt])).to_numpy()
            frames.append(pd.DataFrame({"data_cutoff": c, "station_id": inf["station_id"], "target_at": tgt, "horizon_minutes": inf["horizon_minutes"], "actual": actual, f"pred_{variant.name}": pred}))
        if bi % 8 == 0 or bi == len(buckets):
            log(f"[variant {variant.name}] bucket {bi}/{len(buckets)} elapsed={time.perf_counter()-started:.0f}s")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# G replay (production g2_factor, G served every hourly cycle -> steady-state feedback)
# --------------------------------------------------------------------------------------
def replay_g(bases: pd.DataFrame, *, base_col: str = "pred_F") -> pd.DataFrame:
    frame = bases.sort_values(["data_cutoff", "station_id", "horizon_minutes"]).reset_index(drop=True)
    feedback_all = frame[["station_id", "target_at", base_col, "actual"]].rename(columns={base_col: "base_prediction"})
    records = []
    for cutoff, group in frame.groupby("data_cutoff", sort=True):
        feedback = prepare_feedback_frame(feedback_all[feedback_all["target_at"] <= cutoff], pd.Timestamp(cutoff))
        for sid in group["station_id"].unique():
            factor, count = g2_factor(sid, feedback)
            records.append({"data_cutoff": cutoff, "station_id": sid, "g_factor": factor, "g_feedback_samples": count})
    frame = frame.merge(pd.DataFrame(records), on=["data_cutoff", "station_id"], how="left")
    frame["pred_G"] = np.maximum(0.0, frame[base_col] * frame["g_factor"])
    return frame


# --------------------------------------------------------------------------------------
# Scoring (canonical: drift_monitor._accuracy_by)
# --------------------------------------------------------------------------------------
def scored(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    out = frame[["cycle_id", "data_cutoff", "station_id", "horizon_minutes", "target_at", "actual"]].copy()
    out["prediction"] = frame[pred_col].astype(float)
    out = out.dropna(subset=["actual", "prediction"])
    out["abs_error"] = (out["prediction"] - out["actual"]).abs()
    out["residual"] = out["prediction"] - out["actual"]
    return out


def cycle_scores(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    s = scored(frame, pred_col)
    per = _accuracy_by(s, ["cycle_id", "data_cutoff", "station_id"])
    return per.groupby(["cycle_id", "data_cutoff"], as_index=False)["accuracy"].mean().rename(columns={"accuracy": "cycle_accuracy"}).sort_values("data_cutoff")


def station_scores(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Per-station accuracy over the whole window (station WAPE pooled across cycles)."""
    return _accuracy_by(scored(frame, pred_col), ["station_id"])


def mean_cycle_accuracy(frame: pd.DataFrame, pred_col: str) -> float:
    return float(cycle_scores(frame, pred_col)["cycle_accuracy"].mean())
