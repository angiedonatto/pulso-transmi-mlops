"""Candidate J v1.1 shadow inference (regime-aware wrapper of Candidate G).

J post-processes what the official Candidate G cycle already persisted in `prediction_traces`
(base_prediction, calibration_factor, final_prediction per target) with observations
<= data_cutoff, and writes 48 predictions under J's own model_id.

Safety contract (enforced here and tested):
- never imports or calls the submission code, never reads PULSO_API_KEY, never talks to the
  competition API; the only network target is Supabase;
- writes only `predictions` rows of J's model_id (delete + insert of that cycle, the same
  idempotent pattern production uses) and a `pipeline_runs`/`model_versions` row at registration;
- refuses to run if J is active, if the frozen artifact differs from the code, or if G's trace
  does not reproduce G's official predictions;
- best effort: failures come back as statuses, never as exceptions to the caller.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd

import candidate_j_regime as cj


MODEL_VERSION = "candidate_j_v1_1"
MODEL_NAME = "candidate_j_v1_1_regime_aware"
MODEL_TYPE = "shadow_regime_aware_candidate_j"
MODEL_LABEL = "Candidate J v1.1 / regime-aware (shadow)"
CHAMPION_MODEL_NAME = "candidate_g_f4_profile_ratio_g2_ewma"
DEFAULT_ARTIFACT_URI = f"repo://artifacts/shadow/{MODEL_VERSION}"
# Design freeze: the replay that produced J v1.1 used data up to 2026-09-18T10:00Z. Cycles whose
# data_cutoff is >= this have targets J's design never saw: they are the out-of-sample evidence.
DESIGN_DATA_END = "2026-09-18T10:00:00Z"
OOS_FROM_CUTOFF = DESIGN_DATA_END
OBSERVATION_LOOKBACK = pd.Timedelta(days=31)  # 28 days of seasonal baseline + 48 h of signals + margin
HISTORY_WINDOW = pd.Timedelta(hours=26)  # G traces needed for the 24 h usefulness gate
EXPECTED_PREDICTIONS = 48
PARITY_TOLERANCE = 1e-6
FORBIDDEN_ENV = ("PULSO_API_KEY",)
FAILURE_STATUSES = {"failed", "refused", "not_registered"}


class CandidateJShadowError(RuntimeError):
    """Raised when J cannot be produced safely for a cycle."""


@dataclass(frozen=True)
class CycleResult:
    cycle_id: str
    data_cutoff: str
    status: str
    predictions_written: int = 0
    regimes: dict[str, int] = field(default_factory=dict)
    paths: dict[str, int] = field(default_factory=dict)
    guard_fired: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class ShadowRunResult:
    status: str
    model_id: str | None = None
    cycles: list[CycleResult] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Frozen artifact
# --------------------------------------------------------------------------------------


def frozen_config_dict(config: cj.CandidateJConfig = cj.CANDIDATE_J_V1_1) -> dict[str, Any]:
    return {k: v for k, v in asdict(config).items() if k != "evidence"}


def build_artifact_metadata(replay_summary: dict[str, Any]) -> dict[str, Any]:
    """Freeze J v1.1 from the replay summary (thresholds calibrated before the evaluated period)."""
    thresholds = replay_summary["thresholds"]
    return {
        "version": MODEL_VERSION,
        "label": MODEL_LABEL,
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "mode": "shadow",
        "champion_dependency": CHAMPION_MODEL_NAME,
        "config": frozen_config_dict(),
        "thresholds": thresholds,
        "design_freeze": {
            "design_data_end": DESIGN_DATA_END,
            "oos_from_cutoff": OOS_FROM_CUTOFF,
            "replay_decision": replay_summary["decision"]["decision"],
            "replay_report": "reports/candidate_j/latest.md",
            "rule": "J v1.1 must not be modified while its shadow evidence is collected",
        },
        "decision_gate": {"min_new_severe_shock_cycles": 12, "or": "end of the shock and recovery observed"},
        "evidence": dict(cj.EVIDENCE),
    }


def load_artifact(artifact_uri: str = DEFAULT_ARTIFACT_URI) -> dict[str, Any]:
    path = _resolve_artifact_uri(artifact_uri) / "metadata.json"
    metadata = json.loads(path.read_text())
    validate_artifact(metadata)
    return metadata


def validate_artifact(metadata: dict[str, Any]) -> None:
    if metadata.get("version") != MODEL_VERSION or metadata.get("model_type") != MODEL_TYPE:
        raise CandidateJShadowError(f"unexpected artifact {metadata.get('version')!r}/{metadata.get('model_type')!r}")
    if metadata.get("config") != frozen_config_dict():
        raise CandidateJShadowError("Candidate J code differs from the frozen v1.1 configuration; refusing to run")
    if metadata.get("design_freeze", {}).get("oos_from_cutoff") != OOS_FROM_CUTOFF:
        raise CandidateJShadowError("design freeze changed; refusing to run")


def thresholds_from_artifact(metadata: dict[str, Any]) -> cj.RegimeThresholds:
    return cj.RegimeThresholds(**metadata["thresholds"])


def model_version_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    """model_versions row for the shadow registration. is_active is always False."""
    return {
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "artifact_uri": DEFAULT_ARTIFACT_URI,
        "trained_until": DESIGN_DATA_END,
        "feature_config": {
            "mode": "shadow",
            "version": MODEL_VERSION,
            "label": MODEL_LABEL,
            "champion_dependency": CHAMPION_MODEL_NAME,
            "design_freeze": metadata["design_freeze"],
            "thresholds": metadata["thresholds"],
            "artifact_storage": "repo",
        },
        "params": metadata["config"],
        "is_active": False,
    }


# --------------------------------------------------------------------------------------
# Pure computation for one cycle
# --------------------------------------------------------------------------------------


def build_cycle_predictions(
    *,
    cycle_targets: pd.DataFrame,
    g_trace: pd.DataFrame,
    g_history: pd.DataFrame,
    observations: pd.DataFrame,
    data_cutoff: pd.Timestamp,
    thresholds: cj.RegimeThresholds,
    config: cj.CandidateJConfig = cj.CANDIDATE_J_V1_1,
) -> pd.DataFrame:
    """J for one cycle. `g_trace`: G's trace rows of this cycle; `g_history`: G's trace rows of
    earlier cycles (any target); `observations`: may contain rows after the cutoff, they are dropped."""
    cutoff = pd.Timestamp(data_cutoff).tz_convert("UTC")
    trace = _normalize_trace(g_trace)
    _check_targets(cycle_targets, trace)
    _check_g_parity(trace)

    obs = observations.copy()
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    obs = obs[obs["observed_at"] <= cutoff]  # information available at prediction time
    wide = cj.observation_matrix(obs)
    if wide.index.max() < cutoff:
        raise CandidateJShadowError(f"observations end at {wide.index.max()}, before data_cutoff {cutoff}")
    anomaly = cj.hourly_anomaly(wide, cj.seasonal_baseline(wide))

    previous = []
    for k in range(1, config.shock_memory_hours):
        earlier = cutoff - pd.Timedelta(hours=k)
        previous.append(cj.shock_share(cj.signals_at(earlier, wide.loc[wide.index <= earlier], anomaly.loc[anomaly.index <= earlier]), thresholds))
    regimes = cj.classify_regimes(cj.signals_at(cutoff, wide, anomaly), thresholds, config, previous_shares=previous)

    history = _usefulness_history(_normalize_trace(g_history) if not g_history.empty else g_history, wide, cutoff)
    usefulness = cj.g2_usefulness(history, cutoff, config)
    targets = trace.rename(columns={"calibration_factor": "g2_factor"})[["station_id", "target_at", "horizon_minutes", "base_prediction", "g2_factor"]]
    composed = cj.compose_predictions(targets, regimes, usefulness, thresholds, wide=wide, cutoff=cutoff, config=config)
    composed = composed.merge(trace[["station_id", "target_at", "final_prediction"]], on=["station_id", "target_at"], how="left")
    if len(composed) != EXPECTED_PREDICTIONS or composed["j_prediction"].isna().any() or not np.isfinite(composed["j_prediction"]).all():
        raise CandidateJShadowError("Candidate J did not produce 48 finite predictions")
    if (composed["j_prediction"] < 0).any():
        raise CandidateJShadowError("negative Candidate J prediction")
    return composed


def _usefulness_history(history: pd.DataFrame, wide: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Resolved targets of earlier G cycles with what G, its base and the recent-profile path predicted."""
    columns = ["station_id", "target_at", "actual", "base_prediction", "g_prediction", "rp_prediction"]
    if history.empty:
        return pd.DataFrame(columns=columns)
    past = history[(history["data_cutoff"] < cutoff) & (history["target_at"] <= cutoff)]
    parts = []
    for past_cutoff, group in past.groupby("data_cutoff"):
        group = group.copy()
        group["rp_prediction"] = cj.recent_profile_predictions(group, wide, past_cutoff)
        parts.append(group)
    if not parts:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(parts, ignore_index=True)
    frame["actual"] = [_value(wide, t, s) for s, t in zip(frame["station_id"], frame["target_at"])]
    frame = frame.rename(columns={"final_prediction": "g_prediction"}).dropna(subset=["actual"])
    return frame[columns]


def _normalize_trace(frame: pd.DataFrame) -> pd.DataFrame:
    trace = frame.copy()
    trace["station_id"] = trace["station_id"].astype(str).str.zfill(5)
    trace["target_at"] = pd.to_datetime(trace["target_at"], utc=True)
    trace["data_cutoff"] = pd.to_datetime(trace["data_cutoff"], utc=True)
    for column in ("base_prediction", "calibration_factor", "final_prediction"):
        trace[column] = trace[column].astype(float)
    trace["horizon_minutes"] = trace["horizon_minutes"].astype(int)
    return trace


def _check_targets(cycle_targets: pd.DataFrame, trace: pd.DataFrame) -> None:
    wanted = {(str(s).zfill(5), pd.Timestamp(t).tz_convert("UTC")) for s, t in zip(cycle_targets["station_id"], pd.to_datetime(cycle_targets["target_at"], utc=True))}
    traced = set(zip(trace["station_id"], trace["target_at"]))
    if len(wanted) != EXPECTED_PREDICTIONS or traced != wanted:
        raise CandidateJShadowError(f"G trace does not cover the cycle targets exactly ({len(traced)} vs {len(wanted)})")


def _check_g_parity(trace: pd.DataFrame) -> None:
    rebuilt = np.maximum(0.0, trace["base_prediction"] * trace["calibration_factor"])
    if float((rebuilt - trace["final_prediction"]).abs().max()) > PARITY_TOLERANCE * max(1.0, float(trace["final_prediction"].abs().max())):
        raise CandidateJShadowError("G trace does not reproduce G's final predictions; J would not wrap the official G")


# --------------------------------------------------------------------------------------
# Supabase store (read: model_versions, prediction_traces, predictions, forecast_cycle_targets,
# observations; write: predictions of J only, and the registration rows)
# --------------------------------------------------------------------------------------


class SupabaseCandidateJShadowStore:
    def __init__(self, supabase_url: str | None = None, service_key: str | None = None, timeout: float = 30.0, transport: httpx.BaseTransport | None = None) -> None:
        resolved_url = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        resolved_key = service_key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not resolved_url or not resolved_key:
            raise CandidateJShadowError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self._client = httpx.Client(
            base_url=f"{resolved_url}/rest/v1",
            headers={"apikey": resolved_key, "Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SupabaseCandidateJShadowStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def candidate_j_model_version(self) -> dict[str, Any] | None:
        for row in self._get("/model_versions", {"model_type": f"eq.{MODEL_TYPE}", "select": "model_id,run_id,is_active,feature_config,model_name"}):
            feature_config = row.get("feature_config") if isinstance(row.get("feature_config"), dict) else {}
            if feature_config.get("version") == MODEL_VERSION:
                return row
        return None

    def champion_model_id(self) -> str:
        rows = self._get("/model_versions", {"is_active": "eq.true", "select": "model_id,model_name"})
        if len(rows) != 1 or rows[0].get("model_name") != CHAMPION_MODEL_NAME:
            raise CandidateJShadowError(f"expected exactly one active champion named {CHAMPION_MODEL_NAME}, found {rows}")
        return str(rows[0]["model_id"])

    def champion_traces(self, model_id: str, since: pd.Timestamp) -> pd.DataFrame:
        rows = self._paged(
            "/prediction_traces",
            {"model_id": f"eq.{model_id}", "data_cutoff": f"gte.{_ts(since)}",
             "select": "cycle_id,station_id,target_at,horizon_minutes,base_prediction,calibration_factor,final_prediction,data_cutoff", "order": "data_cutoff.asc"},
        )
        return pd.DataFrame(rows, columns=["cycle_id", "station_id", "target_at", "horizon_minutes", "base_prediction", "calibration_factor", "final_prediction", "data_cutoff"])

    def official_predictions(self, model_id: str, cycle_id: str) -> pd.DataFrame:
        return pd.DataFrame(self._get("/predictions", {"model_id": f"eq.{model_id}", "cycle_id": f"eq.{cycle_id}", "select": "station_id,target_observed_at,prediction"}))

    def prediction_count(self, model_id: str, cycle_id: str) -> int:
        return len(self._get("/predictions", {"model_id": f"eq.{model_id}", "cycle_id": f"eq.{cycle_id}", "select": "station_id"}))

    def cycle_targets(self, cycle_id: str) -> pd.DataFrame:
        return pd.DataFrame(self._get("/forecast_cycle_targets", {"cycle_id": f"eq.{cycle_id}", "select": "station_id,target_at,horizon_minutes,target_order"}))

    def observations_between(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rows = self._paged(
            "/observations",
            {"and": f"(observed_at.gt.{_ts(start)},observed_at.lte.{_ts(end)})", "select": "station_id,observed_at,demand", "order": "observed_at.asc,station_id.asc"},
        )
        return pd.DataFrame(rows, columns=["station_id", "observed_at", "demand"])

    def persist_predictions(self, model: dict[str, Any], cycle_id: str, frame: pd.DataFrame) -> int:
        """Insert J's rows for this cycle.

        The runner checks for a complete existing cycle before calling this method. If a partial
        cycle exists, replacing only J's own rows is still safe and idempotent: it cannot touch G,
        receipts, submissions, or any other model's predictions.
        """
        if model.get("is_active") is not False:
            raise CandidateJShadowError("Candidate J must stay is_active=false")
        delete = self._client.delete("/predictions", params={"cycle_id": f"eq.{cycle_id}", "model_id": f"eq.{model['model_id']}"})
        delete.raise_for_status()
        response = self._client.post(
            "/predictions",
            json=[
                {
                    "run_id": model["run_id"],
                    "model_id": model["model_id"],
                    "cycle_id": cycle_id,
                    "station_id": row.station_id,
                    "target_observed_at": _ts(row.target_at),
                    "horizon_minutes": int(row.horizon_minutes),
                    "prediction": float(row.j_prediction),
                }
                for row in frame.itertuples(index=False)
            ],
        )
        response.raise_for_status()
        return len(frame)

    def register_model_version(self, payload: dict[str, Any], *, git_commit: str | None) -> dict[str, Any]:
        if payload.get("is_active") is not False:
            raise CandidateJShadowError("Candidate J registration must use is_active=false")
        existing = self.candidate_j_model_version()
        if existing is not None:
            return {"status": "already_registered", "model_id": existing["model_id"], "run_id": existing["run_id"], "is_active": existing["is_active"]}
        run_id, model_id = str(uuid4()), str(uuid4())
        self._post("/pipeline_runs", {"run_id": run_id, "status": "running", "git_commit": git_commit, "data_cutoff": payload["trained_until"], "trigger_type": "shadow_registration"})
        self._post("/model_versions", {"model_id": model_id, "run_id": run_id, **payload})
        self._patch("/pipeline_runs", {"status": "success", "finished_at": pd.Timestamp.now(tz="UTC").isoformat()}, {"run_id": f"eq.{run_id}"})
        return {"status": "registered", "model_id": model_id, "run_id": run_id, "is_active": False}

    def _get(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def _paged(self, path: str, params: dict[str, str], page_size: int = 1000) -> list[dict[str, Any]]:
        rows, offset = [], 0
        while True:
            page = self._get(path, {**params, "limit": str(page_size), "offset": str(offset)})
            rows.extend(page)
            if len(page) < page_size:
                return rows
            offset += page_size

    def _post(self, path: str, payload: Any) -> None:
        self._client.post(path, json=payload).raise_for_status()

    def _patch(self, path: str, payload: dict[str, Any], params: dict[str, str]) -> None:
        self._client.patch(path, params=params, json=payload).raise_for_status()


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


def run_candidate_j_shadow(store: SupabaseCandidateJShadowStore, *, artifact_uri: str = DEFAULT_ARTIFACT_URI, max_cycles: int = 6, cycle_id: str | None = None) -> ShadowRunResult:
    """Produce J for G's traced cycles since the design freeze that lack J predictions (newest
    `max_cycles`, oldest first). Never raises."""
    try:
        _assert_no_submission_access()
        metadata = load_artifact(artifact_uri)
        thresholds = thresholds_from_artifact(metadata)
        model = store.candidate_j_model_version()
        if model is None:
            return ShadowRunResult(status="not_registered", reason="register Candidate J first (scripts/register_candidate_j_shadow.py --apply)")
        if model.get("is_active") is not False:
            return ShadowRunResult(status="refused", model_id=str(model["model_id"]), reason="Candidate J is active; shadow mode requires is_active=false")
        champion_id = store.champion_model_id()
        oos_from = pd.Timestamp(OOS_FROM_CUTOFF)
        traces = store.champion_traces(champion_id, oos_from - HISTORY_WINDOW)
        if traces.empty:
            return ShadowRunResult(status="no_champion_trace", model_id=str(model["model_id"]))
        traces = _normalize_trace(traces)
        complete = traces.groupby("cycle_id").size()
        cycles = (
            traces[traces["cycle_id"].isin(complete[complete == EXPECTED_PREDICTIONS].index)]
            .groupby("cycle_id")["data_cutoff"].first().sort_values()
        )
        cycles = cycles[cycles >= oos_from]
        if cycle_id is not None:
            cycles = cycles[cycles.index == cycle_id]
        already_complete = 0
        pending = []
        for c in cycles.index:
            count = store.prediction_count(str(model["model_id"]), c)
            if count == EXPECTED_PREDICTIONS:
                already_complete += 1
                continue
            pending.append(c)
        pending = pending[-max_cycles:]
        results = []
        for pending_id in pending:
            results.append(_run_cycle(store, model, champion_id, pending_id, cycles[pending_id], traces, thresholds))
        if results and all(r.status == "shadow_predicted" for r in results):
            status = "shadow_predicted"
        elif not results and already_complete:
            status = "already_shadow_predicted"
        elif not results:
            status = "up_to_date"
        else:
            status = "partial"
        return ShadowRunResult(status=status, model_id=str(model["model_id"]), cycles=results)
    except Exception as exc:  # noqa: BLE001 - best effort by contract
        return ShadowRunResult(status="failed", reason=f"{type(exc).__name__}: {exc}")


def _run_cycle(store: SupabaseCandidateJShadowStore, model: dict[str, Any], champion_id: str, cycle_id: str, cutoff: pd.Timestamp, traces: pd.DataFrame, thresholds: cj.RegimeThresholds) -> CycleResult:
    try:
        trace = traces[traces["cycle_id"] == cycle_id]
        official = store.official_predictions(champion_id, cycle_id)
        _check_official_matches_trace(official, trace)
        observations = store.observations_between(cutoff - OBSERVATION_LOOKBACK, cutoff)
        frame = build_cycle_predictions(
            cycle_targets=store.cycle_targets(cycle_id),
            g_trace=trace,
            g_history=traces[(traces["data_cutoff"] < cutoff) & (traces["data_cutoff"] >= cutoff - HISTORY_WINDOW)],
            observations=observations,
            data_cutoff=cutoff,
            thresholds=thresholds,
        )
        written = store.persist_predictions(model, cycle_id, frame)
        return CycleResult(
            cycle_id=cycle_id, data_cutoff=_ts(cutoff), status="shadow_predicted", predictions_written=written,
            regimes=frame.groupby("station_id")["regime"].first().value_counts().to_dict(),
            paths=frame["path"].value_counts().to_dict(), guard_fired=int(frame["guard_fired"].sum()),
        )
    except Exception as exc:  # noqa: BLE001 - one cycle failing must not stop the others
        return CycleResult(cycle_id=cycle_id, data_cutoff=_ts(cutoff), status="failed", reason=f"{type(exc).__name__}: {exc}")


def _check_official_matches_trace(official: pd.DataFrame, trace: pd.DataFrame) -> None:
    if len(official) != EXPECTED_PREDICTIONS:
        raise CandidateJShadowError(f"G has {len(official)} official predictions for this cycle")
    official = official.assign(station_id=official["station_id"].astype(str).str.zfill(5), target_at=pd.to_datetime(official["target_observed_at"], utc=True))
    joined = official.merge(trace, on=["station_id", "target_at"], how="inner")
    if len(joined) != EXPECTED_PREDICTIONS or float((joined["prediction"].astype(float) - joined["final_prediction"]).abs().max()) > PARITY_TOLERANCE * max(1.0, float(joined["prediction"].abs().max())):
        raise CandidateJShadowError("G trace does not match G's official predictions")


def _assert_no_submission_access() -> None:
    for name in FORBIDDEN_ENV:
        if os.environ.get(name):
            raise CandidateJShadowError(f"{name} is present in the environment; the J shadow job must run without it")


def _value(wide: pd.DataFrame, when: pd.Timestamp, station: str) -> float:
    try:
        value = wide.at[pd.Timestamp(when), station]
    except KeyError:
        return math.nan
    return float(value) if not pd.isna(value) else math.nan


def _ts(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _resolve_artifact_uri(artifact_uri: str) -> Path:
    if artifact_uri.startswith("repo://"):
        return Path(__file__).resolve().parents[1] / artifact_uri.removeprefix("repo://")
    return Path(artifact_uri)
