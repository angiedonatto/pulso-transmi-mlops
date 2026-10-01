"""Candidate K v2.1 shadow job: frozen deep-learning challenger, evaluated on cycles it never saw.

K v2.1 (`candidate_k_deep`, design and results in reports/candidate_k/latest.md) is frozen: the
research module and the synthetic-pretrained local expert are pinned by sha256 in
artifacts/shadow/candidate_k_v2_1/metadata.json, together with the walk-forward procedure (daily
refit at 05:00Z on targets observed by then, 3 seeds, median). Only cycles whose data_cutoff is
>= the K design freeze are processed: they are the out-of-sample evidence.

Safety contract (enforced here and tested):
- never imports or calls the submission code, refuses to run if PULSO_API_KEY is present, never
  talks to the competition API; the only network target is Supabase;
- writes only `predictions` and `prediction_traces` rows of K's model_id (delete + insert of a
  cycle, idempotent) and the `pipeline_runs`/`model_versions` rows at registration (is_active=false);
- causal: the predictor receives observations <= data_cutoff only, and a cycle is not predicted
  until observations reach its data_cutoff;
- refuses to run if K is active or if the frozen files/procedure differ from the artifact;
- best effort: failures come back as statuses, never as exceptions to the caller.

This module does not import PyTorch; the frozen predictor lives in `candidate_k_runtime`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd

from drift_monitor import accuracy_summary


ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "candidate_k_v2_1"
MODEL_NAME = "candidate_k_v2_1_regime_robust_dl"
MODEL_TYPE = "shadow_deep_candidate_k"
MODEL_LABEL = "Candidate K v2.1 / regime-robust deep learning (shadow)"
J_MODEL_NAME = "candidate_j_v1_1_regime_aware"
DEFAULT_ARTIFACT_URI = f"repo://artifacts/shadow/{MODEL_VERSION}"
# K v2.1 was designed with observations up to this instant (virtual time): cycles whose data_cutoff
# is >= it have targets K's design never saw.
DESIGN_DATA_END = "2026-09-18T19:00:00Z"
OOS_FROM_CUTOFF = DESIGN_DATA_END
HISTORY_START = "2026-07-26T05:00:00Z"  # first observation of the stream; K trains on all history
TRAIN_STRIDE_ANCHOR = HISTORY_START
TRAIN_STRIDE = 2  # one training origin every 30 min, on the grid anchored at HISTORY_START
REFIT_HOUR_UTC = 5  # 00:00 America/Bogota
SEEDS = (0, 1, 2)
GATE_EPOCHS = 6
SYNTHETIC_PROB = 0.35
LOCAL_LOOKBACK = 32
EXPECTED_PREDICTIONS = 48
FORBIDDEN_ENV = ("PULSO_API_KEY",)
FAILURE_STATUSES = {"failed", "refused", "not_registered"}
FROZEN_MODULE = "src/candidate_k_deep.py"
LOCAL_WEIGHTS = "local_pretrained.pt"
COMPARED_MODELS = {"J": "j_prediction", "K": "k_prediction", "baseline_latest": "baseline_latest_prediction"}
MIN_CYCLES_FOR_READING = 24


class CandidateKShadowError(RuntimeError):
    """Raised when K cannot be produced safely for a cycle."""


class Predictor(Protocol):
    def predict(self, observations: pd.DataFrame, data_cutoff: pd.Timestamp, cycle_targets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return one row per target (station_id, target_at, horizon_minutes, k_prediction,
        k_seasonal_prediction, k_seasonal_weight) and run info (refit_at, trained, train_rows)."""


@dataclass(frozen=True)
class CycleResult:
    cycle_id: str
    data_cutoff: str
    status: str
    predictions_written: int = 0
    refit_at: str | None = None
    trained: bool = False
    mean_seasonal_weight: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ShadowRunResult:
    status: str
    model_id: str | None = None
    cycles: list[CycleResult] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    waiting_for_data: list[str] = field(default_factory=list)
    refits_trained: list[str] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Frozen artifact and procedure
# --------------------------------------------------------------------------------------


def frozen_procedure() -> dict[str, Any]:
    return {
        "architecture": "staged_re",
        "seeds": list(SEEDS),
        "ensemble": "median over seeds",
        "refit": "from scratch every day at 05:00Z (00:00 America/Bogota) on samples whose 4 targets are observed by then",
        "refit_hour_utc": REFIT_HOUR_UTC,
        "history_start": HISTORY_START,
        "train_stride_slots": TRAIN_STRIDE,
        "train_stride_anchor": TRAIN_STRIDE_ANCHOR,
        "gate_epochs": GATE_EPOCHS,
        "synthetic_prob": SYNTHETIC_PROB,
        "local_lookback": LOCAL_LOOKBACK,
        "torch_threads": 1,
    }


def artifact_dir(artifact_uri: str = DEFAULT_ARTIFACT_URI) -> Path:
    if artifact_uri.startswith("repo://"):
        return ROOT / artifact_uri.removeprefix("repo://")
    return Path(artifact_uri)


def build_artifact_metadata(*, config: dict[str, Any], artifact_dir: Path, parity: dict[str, Any] | None = None, local_pretrain: dict[str, Any] | None = None) -> dict[str, Any]:
    """Freeze K v2.1: research config, procedure, and sha256 of the module and local-expert weights."""
    return {
        "version": MODEL_VERSION,
        "label": MODEL_LABEL,
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "mode": "shadow",
        "config": config,
        "procedure": frozen_procedure(),
        "files": {FROZEN_MODULE: _sha256(ROOT / FROZEN_MODULE), LOCAL_WEIGHTS: _sha256(Path(artifact_dir) / LOCAL_WEIGHTS)},
        "local_pretrain": local_pretrain,
        "design_freeze": {
            "design_data_end": DESIGN_DATA_END,
            "oos_from_cutoff": OOS_FROM_CUTOFF,
            "experiment_report": "reports/candidate_k/latest.md",
            "rule": "K v2.1 must not be modified or tuned while its shadow evidence is collected",
        },
        "comparison": {"against": [J_MODEL_NAME, "baseline_latest"], "metric": "leaderboard mean_station_accuracy (drift_monitor.accuracy_summary)"},
        "promotion": "manual decision only; no automatic promotion",
        "parity": parity,
    }


def load_artifact(artifact_uri: str = DEFAULT_ARTIFACT_URI) -> dict[str, Any]:
    directory = artifact_dir(artifact_uri)
    metadata = json.loads((directory / "metadata.json").read_text())
    validate_artifact(metadata, directory)
    return metadata


def validate_artifact(metadata: dict[str, Any], directory: Path) -> None:
    if metadata.get("version") != MODEL_VERSION or metadata.get("model_type") != MODEL_TYPE:
        raise CandidateKShadowError(f"unexpected artifact {metadata.get('version')!r}/{metadata.get('model_type')!r}")
    if metadata.get("procedure") != frozen_procedure():
        raise CandidateKShadowError("the shadow procedure differs from the frozen K v2.1 procedure; refusing to run")
    if metadata.get("design_freeze", {}).get("oos_from_cutoff") != OOS_FROM_CUTOFF:
        raise CandidateKShadowError("design freeze changed; refusing to run")
    files = metadata.get("files", {})
    if files.get(FROZEN_MODULE) != _sha256(ROOT / FROZEN_MODULE):
        raise CandidateKShadowError(f"{FROZEN_MODULE} differs from the frozen K v2.1 module; refusing to run")
    if files.get(LOCAL_WEIGHTS) != _sha256(Path(directory) / LOCAL_WEIGHTS):
        raise CandidateKShadowError("local expert weights differ from the frozen artifact; refusing to run")


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
            "design_freeze": metadata["design_freeze"],
            "procedure": metadata["procedure"],
            "files": metadata["files"],
            "artifact_storage": "repo",
            "runtime": "PyTorch CPU in .github/workflows/shadow_candidate_k.yml (requirements/candidate-k-shadow.txt)",
        },
        "params": metadata["config"],
        "is_active": False,
    }


def refit_time(data_cutoff: Any) -> pd.Timestamp:
    """Latest daily refit (05:00Z) at or before the cutoff."""
    cutoff = _utc(data_cutoff)
    refit = cutoff.floor("D") + pd.Timedelta(hours=REFIT_HOUR_UTC)
    return refit if refit <= cutoff else refit - pd.Timedelta(days=1)


def training_origin_mask(times: pd.DatetimeIndex) -> np.ndarray:
    """Training origins sit on the 30-min grid anchored at the first observation of the stream."""
    slots = (pd.DatetimeIndex(times) - pd.Timestamp(TRAIN_STRIDE_ANCHOR)) // pd.Timedelta(minutes=15)
    return np.asarray(slots % TRAIN_STRIDE == 0)


# --------------------------------------------------------------------------------------
# Supabase store (read: model_versions, forecast_cycles, forecast_cycle_targets, observations,
# predictions, prediction_traces; write: predictions/prediction_traces of K only + registration)
# --------------------------------------------------------------------------------------


class SupabaseCandidateKShadowStore:
    def __init__(self, supabase_url: str | None = None, service_key: str | None = None, timeout: float = 60.0, transport: httpx.BaseTransport | None = None) -> None:
        resolved_url = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        resolved_key = service_key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not resolved_url or not resolved_key:
            raise CandidateKShadowError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self._client = httpx.Client(
            base_url=f"{resolved_url}/rest/v1",
            headers={"apikey": resolved_key, "Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SupabaseCandidateKShadowStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def candidate_k_model_version(self) -> dict[str, Any] | None:
        for row in self._get("/model_versions", {"model_type": f"eq.{MODEL_TYPE}", "select": "model_id,run_id,is_active,feature_config,model_name"}):
            feature_config = row.get("feature_config") if isinstance(row.get("feature_config"), dict) else {}
            if feature_config.get("version") == MODEL_VERSION:
                return row
        return None

    def model_by_name(self, model_name: str) -> dict[str, Any] | None:
        rows = self._get("/model_versions", {"model_name": f"eq.{model_name}", "select": "model_id,run_id,is_active,model_name"})
        return rows[0] if rows else None

    def cycles_since(self, data_cutoff: Any) -> pd.DataFrame:
        rows = self._paged("/forecast_cycles", {"data_cutoff": f"gte.{_ts(data_cutoff)}", "select": "cycle_id,data_cutoff,closes_at", "order": "data_cutoff.asc"})
        frame = pd.DataFrame(rows, columns=["cycle_id", "data_cutoff", "closes_at"])
        frame["data_cutoff"] = pd.to_datetime(frame["data_cutoff"], utc=True)
        frame["closes_at"] = pd.to_datetime(frame["closes_at"], utc=True)
        return frame.sort_values("data_cutoff").reset_index(drop=True)

    def prediction_counts(self, model_id: str, cycle_ids: list[str]) -> Counter:
        counts: Counter = Counter()
        for chunk in _chunks(cycle_ids, 50):
            rows = self._paged("/predictions", {"model_id": f"eq.{model_id}", "cycle_id": f"in.({','.join(chunk)})", "select": "cycle_id"})
            counts.update(row["cycle_id"] for row in rows)
        return counts

    def latest_observed_at(self) -> pd.Timestamp | None:
        rows = self._get("/observations", {"select": "observed_at", "order": "observed_at.desc", "limit": "1"})
        return _utc(rows[0]["observed_at"]) if rows else None

    def cycle_targets(self, cycle_id: str) -> pd.DataFrame:
        rows = self._get("/forecast_cycle_targets", {"cycle_id": f"eq.{cycle_id}", "select": "station_id,target_at,horizon_minutes,target_order"})
        return _normalize_targets(pd.DataFrame(rows, columns=["station_id", "target_at", "horizon_minutes", "target_order"]))

    def observations_until(self, end: Any) -> pd.DataFrame:
        rows = self._paged("/observations", {"observed_at": f"lte.{_ts(end)}", "select": "station_id,observed_at,demand", "order": "observed_at.asc,station_id.asc"})
        return _normalize_observations(pd.DataFrame(rows, columns=["station_id", "observed_at", "demand"]))

    def observations_between(self, start: Any, end: Any) -> pd.DataFrame:
        rows = self._paged(
            "/observations",
            {"and": f"(observed_at.gte.{_ts(start)},observed_at.lte.{_ts(end)})", "select": "station_id,observed_at,demand", "order": "observed_at.asc,station_id.asc"},
        )
        return _normalize_observations(pd.DataFrame(rows, columns=["station_id", "observed_at", "demand"]))

    def predictions_for(self, model_id: str, cycle_ids: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for chunk in _chunks(cycle_ids, 50):
            rows += self._paged("/predictions", {"model_id": f"eq.{model_id}", "cycle_id": f"in.({','.join(chunk)})",
                                                 "select": "cycle_id,station_id,target_observed_at,horizon_minutes,prediction,created_at"})
        return pd.DataFrame(rows, columns=["cycle_id", "station_id", "target_observed_at", "horizon_minutes", "prediction", "created_at"])

    def traces_for(self, model_id: str, cycle_ids: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for chunk in _chunks(cycle_ids, 50):
            rows += self._paged("/prediction_traces", {"model_id": f"eq.{model_id}", "cycle_id": f"in.({','.join(chunk)})",
                                                       "select": "cycle_id,station_id,target_at,base_prediction,calibration_factor,final_prediction"})
        return pd.DataFrame(rows, columns=["cycle_id", "station_id", "target_at", "base_prediction", "calibration_factor", "final_prediction"])

    def persist_predictions(self, model: dict[str, Any], cycle_id: str, frame: pd.DataFrame, *, data_cutoff: Any, train_rows: int) -> int:
        """Replace K's 48 rows (and its trace rows) for this cycle; nothing else is touched."""
        if model.get("is_active") is not False:
            raise CandidateKShadowError("Candidate K must stay is_active=false")
        if len(frame) != EXPECTED_PREDICTIONS:
            raise CandidateKShadowError(f"refusing to persist {len(frame)} rows; a cycle has exactly {EXPECTED_PREDICTIONS}")
        model_id, run_id = str(model["model_id"]), str(model["run_id"])
        self._delete("/predictions", {"cycle_id": f"eq.{cycle_id}", "model_id": f"eq.{model_id}"})
        self._post("/predictions", [
            {"run_id": run_id, "model_id": model_id, "cycle_id": cycle_id, "station_id": row.station_id, "target_observed_at": _ts(row.target_at),
             "horizon_minutes": int(row.horizon_minutes), "prediction": float(row.k_prediction)}
            for row in frame.itertuples(index=False)
        ])
        self._delete("/prediction_traces", {"cycle_id": f"eq.{cycle_id}", "model_id": f"eq.{model_id}"})
        self._post("/prediction_traces", trace_payloads(model_id, cycle_id, frame, data_cutoff=data_cutoff, train_rows=train_rows))
        return len(frame)

    def register_model_version(self, payload: dict[str, Any], *, git_commit: str | None) -> dict[str, Any]:
        if payload.get("is_active") is not False:
            raise CandidateKShadowError("Candidate K registration must use is_active=false")
        existing = self.candidate_k_model_version()
        if existing is not None:
            return {"status": "already_registered", "model_id": existing["model_id"], "run_id": existing["run_id"], "is_active": existing["is_active"]}
        run_id, model_id = str(uuid4()), str(uuid4())
        self._post("/pipeline_runs", {"run_id": run_id, "status": "running", "git_commit": git_commit, "data_cutoff": payload["trained_until"], "trigger_type": "shadow_registration"})
        self._post("/model_versions", {"model_id": model_id, "run_id": run_id, **payload})
        self._client.patch("/pipeline_runs", params={"run_id": f"eq.{run_id}"}, json={"status": "success", "finished_at": pd.Timestamp.now(tz="UTC").isoformat()}).raise_for_status()
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

    def _delete(self, path: str, params: dict[str, str]) -> None:
        self._client.delete(path, params=params).raise_for_status()


def trace_payloads(model_id: str, cycle_id: str, frame: pd.DataFrame, *, data_cutoff: Any, train_rows: int) -> list[dict[str, Any]]:
    """prediction_traces rows with the existing columns (same convention as Candidate I):
    base_prediction = K's seasonal expert, calibration_factor = gate weight on the seasonal expert
    (1 = profile regime, 0 = local dynamics), feedback_samples = training samples of the refit."""
    return [
        {"cycle_id": cycle_id, "model_id": model_id, "station_id": row.station_id, "target_at": _ts(row.target_at),
         "horizon_minutes": int(row.horizon_minutes), "base_prediction": float(row.k_seasonal_prediction),
         "calibration_factor": float(row.k_seasonal_weight), "final_prediction": float(row.k_prediction),
         "feedback_samples": int(train_rows), "data_cutoff": _ts(data_cutoff)}
        for row in frame.itertuples(index=False)
    ]


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


def run_candidate_k_shadow(
    store: SupabaseCandidateKShadowStore,
    *,
    predictor: Predictor | None = None,
    artifact_uri: str = DEFAULT_ARTIFACT_URI,
    cache_dir: str | Path | None = None,
    max_cycles: int = 12,
    cycle_id: str | None = None,
    check_only: bool = False,
) -> ShadowRunResult:
    """Predict, oldest first, every post-freeze cycle without 48 K predictions whose data_cutoff is
    already observed (at most `max_cycles`). `check_only` lists them without predicting. Never raises."""
    try:
        _assert_no_submission_access()
        metadata = load_artifact(artifact_uri)
        model = store.candidate_k_model_version()
        if model is None:
            return ShadowRunResult(status="not_registered", reason="register Candidate K first (scripts/register_candidate_k_shadow.py --apply)")
        model_id = str(model["model_id"])
        if model.get("is_active") is not False:
            return ShadowRunResult(status="refused", model_id=model_id, reason="Candidate K is active; shadow mode requires is_active=false")
        cycles = store.cycles_since(OOS_FROM_CUTOFF)
        if cycle_id is not None:
            cycles = cycles[cycles["cycle_id"] == cycle_id]
        counts = store.prediction_counts(model_id, cycles["cycle_id"].tolist()) if len(cycles) else Counter()
        complete = cycles["cycle_id"].map(lambda c: counts.get(c, 0) == EXPECTED_PREDICTIONS)
        todo = cycles[~complete]
        latest = store.latest_observed_at() if len(todo) else None
        ready = todo[todo["data_cutoff"] <= latest] if latest is not None else todo.iloc[0:0]
        waiting = todo[~todo["cycle_id"].isin(ready["cycle_id"])]
        ready = ready.head(max_cycles)
        pending, waiting_ids = ready["cycle_id"].tolist(), waiting["cycle_id"].tolist()
        if check_only:
            status = "pending" if pending else ("waiting_for_data" if waiting_ids else "up_to_date")
            return ShadowRunResult(status=status, model_id=model_id, pending=pending, waiting_for_data=waiting_ids)
        if ready.empty:
            status = "waiting_for_data" if waiting_ids else ("already_shadow_predicted" if complete.any() else "up_to_date")
            return ShadowRunResult(status=status, model_id=model_id, waiting_for_data=waiting_ids)

        if predictor is None:
            from candidate_k_runtime import FrozenKPredictor  # PyTorch is only needed from here on

            predictor = FrozenKPredictor(metadata, artifact_dir(artifact_uri), cache_dir)
        observations = store.observations_until(ready["data_cutoff"].max())
        observations = observations[observations["observed_at"] >= pd.Timestamp(HISTORY_START)]
        results = [_run_cycle(store, predictor, model, row.cycle_id, row.data_cutoff, observations) for row in ready.itertuples(index=False)]
        ok = [r for r in results if r.status == "shadow_predicted"]
        status = "shadow_predicted" if len(ok) == len(results) else ("failed" if not ok else "partial")
        trained = sorted({r.refit_at for r in results if r.trained and r.refit_at})
        return ShadowRunResult(status=status, model_id=model_id, cycles=results, pending=pending, waiting_for_data=waiting_ids, refits_trained=trained)
    except Exception as exc:  # noqa: BLE001 - best effort by contract
        return ShadowRunResult(status="failed", reason=f"{type(exc).__name__}: {exc}")


def _run_cycle(store: SupabaseCandidateKShadowStore, predictor: Predictor, model: dict[str, Any], cycle_id: str, data_cutoff: pd.Timestamp, observations: pd.DataFrame) -> CycleResult:
    cutoff = _utc(data_cutoff)
    try:
        targets = store.cycle_targets(cycle_id)
        if len(targets) != EXPECTED_PREDICTIONS:
            raise CandidateKShadowError(f"cycle has {len(targets)} targets, expected {EXPECTED_PREDICTIONS}")
        visible = observations[observations["observed_at"] <= cutoff]  # information available at the cutoff
        frame, info = predictor.predict(visible, cutoff, targets)
        frame = validate_cycle_output(frame, targets)
        written = store.persist_predictions(model, cycle_id, frame, data_cutoff=cutoff, train_rows=int(info.get("train_rows", 0)))
        return CycleResult(cycle_id=cycle_id, data_cutoff=_ts(cutoff), status="shadow_predicted", predictions_written=written,
                           refit_at=info.get("refit_at"), trained=bool(info.get("trained")), mean_seasonal_weight=float(frame["k_seasonal_weight"].mean()))
    except Exception as exc:  # noqa: BLE001 - one cycle failing must not stop the others
        return CycleResult(cycle_id=cycle_id, data_cutoff=_ts(cutoff), status="failed", reason=f"{type(exc).__name__}: {exc}")


def validate_cycle_output(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Exactly the cycle's 48 (station, target) pairs, finite and non-negative."""
    out = frame.copy()
    out["station_id"] = out["station_id"].astype(str).str.zfill(5)
    out["target_at"] = pd.to_datetime(out["target_at"], utc=True)
    wanted = set(zip(targets["station_id"], targets["target_at"]))
    got = list(zip(out["station_id"], out["target_at"]))
    if len(got) != EXPECTED_PREDICTIONS or len(set(got)) != len(got) or set(got) != wanted:
        raise CandidateKShadowError(f"K output does not cover the cycle targets exactly ({len(set(got) & wanted)}/{len(wanted)})")
    values = out[["k_prediction", "k_seasonal_prediction", "k_seasonal_weight"]].astype(float)
    if not np.isfinite(values.to_numpy()).all() or (out["k_prediction"] < 0).any():
        raise CandidateKShadowError("K produced non-finite or negative values")
    horizons = dict(zip(zip(targets["station_id"], targets["target_at"]), targets["horizon_minutes"].astype(int)))
    out["horizon_minutes"] = [horizons[key] for key in got]
    return out.sort_values(["station_id", "target_at"]).reset_index(drop=True)


def _assert_no_submission_access() -> None:
    for name in FORBIDDEN_ENV:
        if os.environ.get(name):
            raise CandidateKShadowError(f"{name} is present in the environment; the K shadow job must run without it")


# --------------------------------------------------------------------------------------
# Evaluation: J (official) vs K (shadow) vs baseline_latest, leaderboard metric
# --------------------------------------------------------------------------------------


def baseline_latest(observations: pd.DataFrame, cycle_targets: pd.DataFrame) -> pd.DataFrame:
    """Persistence baseline: the last observation at or before the cycle's data_cutoff, for every horizon."""
    obs = _normalize_observations(observations).sort_values("observed_at")
    obs["observed_at"] = obs["observed_at"].astype("datetime64[ns, UTC]")
    targets = cycle_targets.copy()
    targets["station_id"] = targets["station_id"].astype(str).str.zfill(5)
    targets["data_cutoff"] = pd.to_datetime(targets["data_cutoff"], utc=True).astype("datetime64[ns, UTC]")
    keys = targets[["cycle_id", "station_id", "data_cutoff"]].drop_duplicates().sort_values("data_cutoff")
    last = pd.merge_asof(keys, obs.rename(columns={"demand": "baseline_latest_prediction"}), left_on="data_cutoff", right_on="observed_at",
                         by="station_id", direction="backward")
    return targets.merge(last[["cycle_id", "station_id", "baseline_latest_prediction"]], on=["cycle_id", "station_id"], how="left")


def comparison_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    scored = frame.assign(abs_error=(frame[column] - frame["actual"]).abs())
    summary = accuracy_summary(scored)
    residual = scored["actual"] - scored[column]
    return {
        "rows": summary["rows"],
        "mean_station_accuracy": summary["mean_station_accuracy"],
        "global_wape_accuracy": summary["global_wape_accuracy"],
        "global_wape": summary["global_wape"],
        "accuracy_at_20": summary["accuracy_at_20"],
        "mae": float(scored["abs_error"].mean()),
        "signed_bias": float(residual.mean()),
    }


def evaluate_comparison(rows: pd.DataFrame, *, min_cycles: int = MIN_CYCLES_FOR_READING, bootstrap: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Score J, K and baseline_latest on the cycles where all three have 48 predictions and every
    target is resolved. Never recommends a promotion by itself."""
    columns = ["actual", *COMPARED_MODELS.values()]
    complete = rows.groupby("cycle_id").apply(lambda g: len(g) == EXPECTED_PREDICTIONS and g[columns].notna().all().all(), include_groups=False)
    shared = rows[rows["cycle_id"].isin(complete[complete].index)].copy()
    for model, column in COMPARED_MODELS.items():
        shared[f"{model}_abs_error"] = (shared[column] - shared["actual"]).abs()
    pooled = {model: comparison_metrics(shared, column) for model, column in COMPARED_MODELS.items()} if len(shared) else {}

    cycles = []
    for cid, group in shared.groupby("cycle_id"):
        row = {"cycle_id": cid, "data_cutoff": group["data_cutoff"].iloc[0]}
        for model, column in COMPARED_MODELS.items():
            row[model] = comparison_metrics(group, column)["mean_station_accuracy"]
        if "k_seasonal_weight" in group:
            row["k_mean_seasonal_weight"] = float(group["k_seasonal_weight"].mean())
        if "k_on_time" in group:
            row["k_on_time"] = bool(group["k_on_time"].all())
        cycles.append(row)
    cycles = pd.DataFrame(cycles, columns=["cycle_id", "data_cutoff", *COMPARED_MODELS, "k_mean_seasonal_weight", "k_on_time"])
    cycles["k_minus_j_pp"] = cycles["K"] - cycles["J"]
    cycles["k_minus_baseline_pp"] = cycles["K"] - cycles["baseline_latest"]
    cycles = cycles.sort_values("data_cutoff").reset_index(drop=True)

    stations = _breakdown(shared, "station_id")
    horizons = _breakdown(shared, "horizon_minutes")
    delta = cycles["k_minus_j_pp"].to_numpy()
    interval = _bootstrap_mean(delta, bootstrap, seed) if len(delta) else (math.nan, math.nan)
    n = int(len(cycles))
    if n < min_cycles:
        status = "insufficient_evidence"
    elif interval[0] > 0:
        status = "k_ahead_of_j"
    elif interval[1] < 0:
        status = "k_behind_j"
    else:
        status = "inconclusive"
    summary = {
        "cycles": n,
        "min_cycles_for_reading": min_cycles,
        "pooled": pooled,
        "delta_pp": {"K_minus_J": _delta(pooled, "K", "J"), "K_minus_baseline_latest": _delta(pooled, "K", "baseline_latest")},
        "per_cycle_k_minus_j": {"mean": float(delta.mean()) if n else math.nan, "bootstrap_95": list(interval),
                                "wins": int((delta > 1e-9).sum()), "losses": int((delta < -1e-9).sum()), "ties": int((np.abs(delta) <= 1e-9).sum())},
        "late_k_cycles": int((~cycles["k_on_time"].astype(bool)).sum()) if cycles["k_on_time"].notna().any() else 0,
        "status": status,
        "promotion": "manual_decision_required",
    }
    return {"rows": shared, "cycles": cycles, "stations": stations, "horizons": horizons, "summary": summary}


def _breakdown(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(key):
        row = {key: value}
        for model, column in COMPARED_MODELS.items():
            row[model] = comparison_metrics(group, column)["mean_station_accuracy"]
        rows.append(row)
    out = pd.DataFrame(rows, columns=[key, *COMPARED_MODELS])
    out["k_minus_j_pp"] = out["K"] - out["J"]
    return out


def _bootstrap_mean(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _delta(pooled: dict[str, Any], a: str, b: str) -> float:
    if a not in pooled or b not in pooled:
        return math.nan
    return float(pooled[a]["mean_station_accuracy"] - pooled[b]["mean_station_accuracy"])


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _normalize_observations(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["station_id"] = out["station_id"].astype(str).str.zfill(5)
    out["observed_at"] = pd.to_datetime(out["observed_at"], utc=True)
    out["demand"] = out["demand"].astype(float)
    return out


def _normalize_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["station_id"] = out["station_id"].astype(str).str.zfill(5)
    out["target_at"] = pd.to_datetime(out["target_at"], utc=True)
    out["horizon_minutes"] = out["horizon_minutes"].astype(int)
    return out


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _ts(value: Any) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")
