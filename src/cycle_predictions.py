from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx
import pandas as pd

from champion_baseline import ChampionBaselineError, MODEL_TYPE, SupabaseBaselineStore
from forecast_cycle_sync import ForecastCycle, PulsoForecastCycleApi
from stream_sync import _time_slot_payload


class CyclePredictionError(RuntimeError):
    """Raised when cycle predictions cannot be generated safely."""


CANDIDATE_K_TRACE = "candidate_k_v2_1"  # candidate_k_shadow.MODEL_VERSION, without importing it here


@dataclass(frozen=True)
class CyclePrediction:
    cycle_id: str
    model_id: str
    station_id: str
    target_at: str
    horizon_minutes: int
    prediction: float
    trace: dict[str, Any] | None = None


@dataclass(frozen=True)
class CyclePredictionResult:
    status: str
    cycle_id: str | None = None
    data_cutoff: str | None = None
    model_id: str | None = None
    expected_predictions: int = 0
    predictions_generated: int = 0
    min_prediction: float | None = None
    max_prediction: float | None = None
    target_validation: str = "not_run"


class PredictionStore(Protocol):
    def load_active_champion(self) -> dict[str, Any]:
        ...

    def observations_until(self, data_cutoff: str) -> pd.DataFrame:
        ...

    def persist_cycle_targets(self, cycle: ForecastCycle) -> None:
        ...

    def persist_predictions(self, cycle: ForecastCycle, model: dict[str, Any], predictions: list[CyclePrediction]) -> int:
        ...


def generate_cycle_predictions(api: PulsoForecastCycleApi, store: PredictionStore) -> CyclePredictionResult:
    api.validate_identity()
    cycle = api.current_cycle()
    if cycle is None:
        return CyclePredictionResult(status="no_open_cycle")
    return generate_predictions_for_cycle(cycle, store)


def generate_predictions_for_cycle(cycle: ForecastCycle, store: PredictionStore, *, model: dict[str, Any] | None = None) -> CyclePredictionResult:
    model = model or store.load_active_champion()
    candidate_l = model.get("model_type") == "adaptive_harmonic_candidate_l"
    artifact = None if candidate_l else _load_artifact(model)
    observations = store.observations_until(cycle.data_cutoff)
    dependencies: list[tuple[dict[str, Any], list[CyclePrediction]]] = []
    candidate_k = _is_candidate_k_model(model)
    if candidate_l:
        from candidate_l_runtime import build_official_predictions
        predictions, dependencies = build_official_predictions(cycle, model, observations, store)
    elif candidate_k:
        predictions, dependencies = _build_candidate_k_official_predictions(cycle, model, artifact, observations, store)
    elif _is_candidate_j_model(model):
        predictions, dependency_model, dependency_predictions = _build_candidate_j_official_predictions(
            cycle, model, artifact, observations, store
        )
        dependencies = [(dependency_model, dependency_predictions)]
    else:
        feedback_frame = _feedback_history_for_model(store, cycle, model, artifact)
        predictions = build_predictions(cycle, model["model_id"], artifact, observations, feedback_frame=feedback_frame)
    validate_predictions(cycle, predictions)
    store.persist_cycle_targets(cycle)
    for dependency_model, dependency_predictions in dependencies:
        if not (candidate_k or candidate_l):
            store.persist_predictions(cycle, dependency_model, dependency_predictions)
            continue
        try:  # K's dependencies keep J rollback-ready; a failed write must not cost K's submission
            store.persist_predictions(cycle, dependency_model, dependency_predictions)
        except Exception as exc:  # noqa: BLE001
            _warn("Candidate K dependency not persisted", f"{dependency_model.get('model_id')}: {type(exc).__name__}: {exc}")
    written = store.persist_predictions(cycle, model, predictions)
    values = [prediction.prediction for prediction in predictions]
    return CyclePredictionResult(
        status="predicted",
        cycle_id=cycle.cycle_id,
        data_cutoff=cycle.data_cutoff,
        model_id=model["model_id"],
        expected_predictions=cycle.expected_predictions,
        predictions_generated=written,
        min_prediction=min(values) if values else None,
        max_prediction=max(values) if values else None,
        target_validation="exact",
    )


def build_predictions(
    cycle: ForecastCycle,
    model_id: str,
    artifact: Any,
    observations: pd.DataFrame,
    *,
    feedback_frame: pd.DataFrame | None = None,
) -> list[CyclePrediction]:
    if not isinstance(artifact, dict):
        return _build_non_baseline_predictions(cycle, model_id, artifact, observations, feedback_frame=feedback_frame)
    if artifact.get("model_type") != MODEL_TYPE:
        raise CyclePredictionError(f"unsupported champion model_type: {artifact.get('model_type')!r}")
    frame = _prepare_observations(observations)
    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    if (frame["observed_at"] > cutoff).any():
        raise CyclePredictionError("observations contain rows after data_cutoff")
    lag_lookup = {
        (str(row.station_id), _ts(row.observed_at)): float(row.demand)
        for row in frame.itertuples(index=False)
    }
    slot_means = artifact.get("station_slot_means")
    if not isinstance(slot_means, dict):
        raise CyclePredictionError("champion artifact missing station_slot_means")

    predictions: list[CyclePrediction] = []
    for target in cycle.targets:
        target_at = pd.Timestamp(target.target_at).tz_convert("UTC")
        lag_at = _ts(target_at - pd.to_timedelta(1, unit="D"))
        lag_value = lag_lookup.get((target.station_id, lag_at))
        quarter_hour_block = _quarter_hour_block(target_at)
        fallback = slot_means.get(f"{target.station_id}|{quarter_hour_block}")
        value = lag_value if lag_value is not None else fallback
        if value is None:
            raise CyclePredictionError(f"missing lag and fallback for {target.station_id} {target.target_at}")
        predictions.append(
            CyclePrediction(
                cycle_id=cycle.cycle_id,
                model_id=model_id,
                station_id=target.station_id,
                target_at=target.target_at,
                horizon_minutes=target.horizon_minutes,
                prediction=float(value),
                trace=None,
            )
        )
    return predictions


def validate_predictions(cycle: ForecastCycle, predictions: list[CyclePrediction]) -> None:
    expected_pairs = [(target.station_id, target.target_at) for target in cycle.targets]
    actual_pairs = [(prediction.station_id, prediction.target_at) for prediction in predictions]
    if len(predictions) != cycle.expected_predictions:
        raise CyclePredictionError("prediction count does not match expected_predictions")
    if len(set(actual_pairs)) != len(actual_pairs):
        raise CyclePredictionError("duplicate prediction targets")
    if actual_pairs != expected_pairs:
        raise CyclePredictionError("prediction targets do not match cycle targets exactly")
    for prediction in predictions:
        if not math.isfinite(prediction.prediction):
            raise CyclePredictionError("prediction value is not finite")
        if prediction.prediction < 0:
            raise CyclePredictionError("prediction value is negative")


class SupabaseCyclePredictionStore(SupabaseBaselineStore):
    def persist_l_refit(self, model: dict[str, Any], manifest: dict[str, Any]) -> None:
        config = dict(model.get("feature_config") or {})
        config["latest_refit"] = manifest
        response = self._client.patch(
            "/model_versions", params={"model_id": f"eq.{model['model_id']}"},
            json={"trained_until": model["trained_until"], "feature_config": config},
        )
        response.raise_for_status()

    def observations_until(self, data_cutoff: str) -> pd.DataFrame:
        frame = self.observations()
        parsed_cutoff = pd.Timestamp(data_cutoff).tz_convert("UTC")
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        return frame[frame["observed_at"] <= parsed_cutoff].copy()

    def persist_cycle_targets(self, cycle: ForecastCycle) -> None:
        now = _now_iso()
        self._post_upsert(
            "/forecast_cycles",
            {
                "cycle_id": cycle.cycle_id,
                "data_cutoff": cycle.data_cutoff,
                "closes_at": cycle.closes_at,
                "expected_predictions": cycle.expected_predictions,
                "raw_cycle": cycle.raw_cycle,
                "last_seen_at": now,
            },
            on_conflict="cycle_id",
        )
        if cycle.targets:
            self._post_upsert(
                "/time_slots",
                _time_slot_payloads_for_targets(cycle),
                on_conflict="observed_at",
            )
            self._post_upsert(
                "/forecast_cycle_targets",
                [
                    {
                        "cycle_id": cycle.cycle_id,
                        "station_id": target.station_id,
                        "target_at": target.target_at,
                        "horizon_minutes": target.horizon_minutes,
                        "target_order": target.target_order,
                        "raw_target": target.raw_target,
                    }
                    for target in cycle.targets
                ],
                on_conflict="cycle_id,station_id,target_at",
            )

    def persist_predictions(self, cycle: ForecastCycle, model: dict[str, Any], predictions: list[CyclePrediction]) -> int:
        model_id = model["model_id"]
        run_id = model["run_id"]
        delete_response = self._client.delete(
            "/predictions",
            params={"cycle_id": f"eq.{cycle.cycle_id}", "model_id": f"eq.{model_id}"},
        )
        delete_response.raise_for_status()
        if not predictions:
            return 0
        response = self._client.post(
            "/predictions",
            json=[
                {
                    "run_id": run_id,
                    "model_id": model_id,
                    "cycle_id": prediction.cycle_id,
                    "station_id": prediction.station_id,
                    "target_observed_at": prediction.target_at,
                    "horizon_minutes": prediction.horizon_minutes,
                    "prediction": prediction.prediction,
                }
                for prediction in predictions
            ],
        )
        response.raise_for_status()
        self._persist_prediction_traces(cycle.cycle_id, model_id, predictions)
        return len(predictions)

    def _persist_prediction_traces(self, cycle_id: str, model_id: str, predictions: list[CyclePrediction]) -> None:
        rows = prediction_trace_payloads(predictions)
        delete_response = self._client.delete(
            "/prediction_traces",
            params={"cycle_id": f"eq.{cycle_id}", "model_id": f"eq.{model_id}"},
        )
        if delete_response.status_code == 404:
            return
        delete_response.raise_for_status()
        if not rows:
            return
        response = self._client.post(
            "/prediction_traces",
            params={"on_conflict": "cycle_id,model_id,station_id,target_at"},
            headers={"Prefer": "resolution=merge-duplicates"},
            json=rows,
        )
        response.raise_for_status()

    def feedback_history_until(self, data_cutoff: str, model: dict[str, Any]) -> pd.DataFrame:
        try:
            from candidate_g_adaptive import build_candidate_f_base_frame
            from forecast_cycle_sync import CycleTarget, ForecastCycle
        except ImportError as exc:
            raise CyclePredictionError("candidate G feedback dependencies are unavailable") from exc

        cutoff = pd.Timestamp(data_cutoff).tz_convert("UTC")
        response = self._client.get(
            "/predictions",
            params={
                "model_id": f"eq.{model['model_id']}",
                "target_observed_at": f"lte.{_ts(cutoff)}",
                "select": "cycle_id,station_id,target_observed_at",
                "order": "target_observed_at.desc",
                "limit": "2000",
            },
        )
        response.raise_for_status()
        prediction_rows = response.json()
        if not prediction_rows:
            return pd.DataFrame(columns=["station_id", "target_at", "base_prediction", "actual"])

        observations = self.observations()
        observations["station_id"] = observations["station_id"].astype(str)
        observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
        observations = observations[observations["observed_at"] <= cutoff].copy()
        actual_lookup = {
            (str(row.station_id), _ts(row.observed_at)): float(row.demand)
            for row in observations[["station_id", "observed_at", "demand"]].itertuples(index=False)
        }
        wanted = {
            (str(row["cycle_id"]), str(row["station_id"]), _normalize_timestamp(row["target_observed_at"]))
            for row in prediction_rows
            if row.get("cycle_id")
        }
        by_cycle: dict[str, set[tuple[str, str]]] = {}
        for cycle_id, station_id, target_at in wanted:
            if (station_id, target_at) in actual_lookup:
                by_cycle.setdefault(cycle_id, set()).add((station_id, target_at))
        feedback_rows: list[dict[str, Any]] = []
        for cycle_id, pairs in by_cycle.items():
            cycle_response = self._client.get(
                "/forecast_cycles",
                params={"cycle_id": f"eq.{cycle_id}", "select": "cycle_id,data_cutoff,closes_at,expected_predictions", "limit": "1"},
            )
            cycle_response.raise_for_status()
            cycle_rows = cycle_response.json()
            if not cycle_rows:
                continue
            target_response = self._client.get(
                "/forecast_cycle_targets",
                params={
                    "cycle_id": f"eq.{cycle_id}",
                    "select": "station_id,target_at,horizon_minutes,target_order",
                    "order": "target_order.asc",
                },
            )
            target_response.raise_for_status()
            target_rows = target_response.json()
            cycle_row = cycle_rows[0]
            historical_cycle = ForecastCycle(
                cycle_id=str(cycle_id),
                data_cutoff=_normalize_timestamp(cycle_row["data_cutoff"]),
                closes_at=_normalize_timestamp(cycle_row["closes_at"]) if cycle_row.get("closes_at") else "",
                expected_predictions=int(cycle_row["expected_predictions"]),
                targets=[
                    CycleTarget(
                        station_id=str(row["station_id"]),
                        target_at=_normalize_timestamp(row["target_at"]),
                        horizon_minutes=int(row["horizon_minutes"]),
                        target_order=int(row["target_order"]),
                        raw_target={},
                    )
                    for row in target_rows
                ],
                raw_cycle={},
            )
            hist_cutoff = pd.Timestamp(historical_cycle.data_cutoff).tz_convert("UTC")
            base = build_candidate_f_base_frame(historical_cycle, observations[observations["observed_at"] <= hist_cutoff].copy())
            for row in base.itertuples(index=False):
                target_at = _ts(row.target_at)
                station_id = str(row.station_id)
                if (station_id, target_at) not in pairs:
                    continue
                feedback_rows.append(
                    {
                        "station_id": station_id,
                        "target_at": target_at,
                        "base_prediction": float(row.base_prediction),
                        "actual": actual_lookup[(station_id, target_at)],
                    }
                )
        return pd.DataFrame(feedback_rows).sort_values(["station_id", "target_at"]).reset_index(drop=True)

    def load_model_by_name(self, model_name: str) -> dict[str, Any]:
        response = self._client.get(
            "/model_versions",
            params={
                "model_name": f"eq.{model_name}",
                "select": "*,pipeline_runs(git_commit,started_at)",
                "order": "trained_until.desc",
                "limit": "1",
            },
        )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            raise CyclePredictionError(f"model dependency not found: {model_name}")
        return rows[0]

    def prediction_traces_for_model(
        self,
        model_id: str,
        start_cutoff: str,
        end_cutoff: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        offset = 0
        page_size = 1000
        while True:
            response = self._client.get(
                "/prediction_traces",
                params={
                    "model_id": f"eq.{model_id}",
                    "data_cutoff": f"gte.{start_cutoff}",
                    "and": f"(data_cutoff.lt.{end_cutoff})",
                    "select": "cycle_id,station_id,target_at,horizon_minutes,base_prediction,calibration_factor,final_prediction,data_cutoff",
                    "order": "data_cutoff.asc,target_at.asc,station_id.asc",
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
            rows,
            columns=[
                "cycle_id",
                "station_id",
                "target_at",
                "horizon_minutes",
                "base_prediction",
                "calibration_factor",
                "final_prediction",
                "data_cutoff",
            ],
        )

    def _post_upsert(self, path: str, payload: Any, *, on_conflict: str) -> None:
        response = self._client.post(
            path,
            params={"on_conflict": on_conflict},
            headers={"Prefer": "resolution=merge-duplicates"},
            json=payload,
        )
        response.raise_for_status()


def _load_artifact(model: dict[str, Any]) -> Any:
    active = model.get("is_active")
    if active is not True:
        raise CyclePredictionError("loaded model is not active")
    artifact_uri = model.get("artifact_uri")
    if not isinstance(artifact_uri, str) or not artifact_uri:
        raise CyclePredictionError("active champion missing artifact_uri")
    if artifact_uri.startswith("repo://") or Path(artifact_uri).is_dir():
        try:
            from candidate_b_shadow import MODEL_TYPE as CANDIDATE_B_MODEL_TYPE
            from candidate_b_shadow import load_shadow_artifact
            from candidate_g_adaptive import MODEL_TYPE as CANDIDATE_G_MODEL_TYPE
            from candidate_g_adaptive import load_candidate_g_artifact
            from candidate_j_shadow import MODEL_TYPE as CANDIDATE_J_MODEL_TYPE
            from candidate_j_shadow import load_artifact as load_candidate_j_artifact
            from candidate_k_shadow import MODEL_TYPE as CANDIDATE_K_MODEL_TYPE
            from candidate_k_shadow import load_artifact as load_candidate_k_artifact

            if model.get("model_type") == CANDIDATE_K_MODEL_TYPE:
                return load_candidate_k_artifact(artifact_uri)
            if model.get("model_type") == CANDIDATE_G_MODEL_TYPE:
                return load_candidate_g_artifact(artifact_uri)
            if model.get("model_type") == CANDIDATE_B_MODEL_TYPE:
                return load_shadow_artifact(artifact_uri)
            if model.get("model_type") == CANDIDATE_J_MODEL_TYPE:
                return load_candidate_j_artifact(artifact_uri)
            return load_shadow_artifact(artifact_uri)
        except Exception as exc:
            raise CyclePredictionError(f"cannot load champion artifact: {artifact_uri}") from exc
    try:
        return json.loads(Path(artifact_uri).read_text())
    except OSError as exc:
        raise CyclePredictionError(f"cannot read champion artifact: {artifact_uri}") from exc
    except json.JSONDecodeError as exc:
        raise CyclePredictionError(f"invalid champion artifact JSON: {artifact_uri}") from exc


def _is_candidate_k_model(model: dict[str, Any]) -> bool:
    try:
        from candidate_k_shadow import MODEL_TYPE as CANDIDATE_K_MODEL_TYPE
    except ImportError:
        return False
    return model.get("model_type") == CANDIDATE_K_MODEL_TYPE


def _build_candidate_k_official_predictions(
    cycle: ForecastCycle,
    model: dict[str, Any],
    artifact: dict[str, Any],
    observations: pd.DataFrame,
    store: PredictionStore,
) -> tuple[list[CyclePrediction], list[tuple[dict[str, Any], list[CyclePrediction]]]]:
    """Official Candidate K v2.1 (the frozen model that ran in shadow).

    J v1.1 (with its G dependency) is computed first and returned as dependencies, so J and G keep
    their predictions and traces current under their own inactive model_ids: J stays ready for an
    immediate rollback and is K's fallback. K then runs with the frozen runtime. If K fails, J's
    values are submitted under K's model_id (the champion pinned for the whole run), traced as a
    fallback. Only if K and J both fail is there nothing to submit.
    """
    import candidate_j_shadow as cjs

    dependencies: list[tuple[dict[str, Any], list[CyclePrediction]]] = []
    j_predictions: list[CyclePrediction] | None = None
    j_error: str | None = None
    try:
        j_model = store.load_model_by_name(cjs.MODEL_NAME)  # type: ignore[attr-defined]
        j_artifact = cjs.load_artifact(str(j_model["artifact_uri"]))
        j_predictions, g_model, g_predictions = _build_candidate_j_official_predictions(cycle, j_model, j_artifact, observations, store)
        validate_predictions(cycle, j_predictions)
        dependencies = [(g_model, g_predictions), (j_model, j_predictions)]
    except Exception as exc:  # noqa: BLE001 - J is the fallback, never a reason to skip K
        j_predictions, j_error = None, f"{type(exc).__name__}: {exc}"
        _warn("Candidate J (K fallback) unavailable this cycle", j_error)

    try:
        predictions = _candidate_k_cycle_predictions(cycle, model, artifact, observations)
        validate_predictions(cycle, predictions)
        return predictions, dependencies
    except Exception as exc:  # noqa: BLE001 - K must never cost an official submission while J works
        k_error = f"{type(exc).__name__}: {exc}"
        if j_predictions is None:
            raise CyclePredictionError(f"Candidate K failed ({k_error}) and its J fallback is unavailable ({j_error})") from exc
        _warn("Candidate K failed; submitting J values under K's model_id", k_error)
        cutoff = _ts(cycle.data_cutoff)
        fallback = [
            CyclePrediction(
                cycle_id=prediction.cycle_id,
                model_id=str(model["model_id"]),
                station_id=prediction.station_id,
                target_at=prediction.target_at,
                horizon_minutes=prediction.horizon_minutes,
                prediction=prediction.prediction,
                trace={"candidate": CANDIDATE_K_TRACE, "path": "fallback_j", "fallback_reason": k_error,
                       "seasonal_prediction": prediction.prediction, "seasonal_weight": None, "train_rows": 0, "data_cutoff": cutoff},
            )
            for prediction in j_predictions
        ]
        return fallback, dependencies


def _candidate_k_predictor(model: dict[str, Any], artifact: dict[str, Any]) -> Any:
    """The frozen K v2.1 runtime (PyTorch); daily refits are shared through CANDIDATE_K_CACHE_DIR."""
    import candidate_k_shadow as cks
    from candidate_k_runtime import FrozenKPredictor

    return FrozenKPredictor(artifact, cks.artifact_dir(str(model["artifact_uri"])), os.environ.get("CANDIDATE_K_CACHE_DIR") or None)


def _candidate_k_cycle_predictions(cycle: ForecastCycle, model: dict[str, Any], artifact: dict[str, Any], observations: pd.DataFrame) -> list[CyclePrediction]:
    import candidate_k_shadow as cks

    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")
    targets = cks._normalize_targets(pd.DataFrame(
        [{"station_id": t.station_id, "target_at": t.target_at, "horizon_minutes": t.horizon_minutes} for t in cycle.targets]
    ))
    history = cks._normalize_observations(observations[["station_id", "observed_at", "demand"]])
    history = history[(history["observed_at"] >= pd.Timestamp(cks.HISTORY_START)) & (history["observed_at"] <= cutoff)]
    frame, info = _candidate_k_predictor(model, artifact).predict(history, cutoff, targets)
    frame = cks.validate_cycle_output(frame, targets)
    if info.get("trained"):
        _mark_candidate_k_cache_updated()
    by_target = {(str(row.station_id), _ts(row.target_at)): row for row in frame.itertuples(index=False)}
    predictions = []
    for target in cycle.targets:
        row = by_target[(str(target.station_id), _ts(target.target_at))]
        predictions.append(
            CyclePrediction(
                cycle_id=cycle.cycle_id,
                model_id=str(model["model_id"]),
                station_id=str(target.station_id),
                target_at=target.target_at,
                horizon_minutes=int(target.horizon_minutes),
                prediction=float(row.k_prediction),
                trace={"candidate": CANDIDATE_K_TRACE, "path": "k", "seasonal_prediction": float(row.k_seasonal_prediction),
                       "seasonal_weight": float(row.k_seasonal_weight), "train_rows": int(info.get("train_rows", 0)),
                       "refit_at": info.get("refit_at"), "data_cutoff": _ts(cutoff)},
            )
        )
    return predictions


def _mark_candidate_k_cache_updated() -> None:
    cache_dir = os.environ.get("CANDIDATE_K_CACHE_DIR")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / "UPDATED").write_text(_now_iso() + "\n")


def _warn(title: str, message: str) -> None:
    print(f"::warning title={title}::{message}", flush=True)


def _is_candidate_j_model(model: dict[str, Any]) -> bool:
    try:
        from candidate_j_shadow import MODEL_TYPE as CANDIDATE_J_MODEL_TYPE
    except ImportError:
        return False
    return model.get("model_type") == CANDIDATE_J_MODEL_TYPE


def _build_candidate_j_official_predictions(
    cycle: ForecastCycle,
    model: dict[str, Any],
    artifact: dict[str, Any],
    observations: pd.DataFrame,
    store: PredictionStore,
) -> tuple[list[CyclePrediction], dict[str, Any], list[CyclePrediction]]:
    """Build official Candidate J predictions while keeping Candidate G as its causal dependency.

    J v1.1 was designed as a regime-aware wrapper around G. Once J is active, G is no longer the
    champion, so we reconstruct G in-memory for the same cycle, persist those dependency predictions
    and traces under G's own inactive model_id, and then compose J from that trace. This keeps G2
    feedback causal and current without sending G officially.
    """
    try:
        import candidate_j_shadow as cjs
        from candidate_g_adaptive import load_candidate_g_artifact
    except ImportError as exc:
        raise CyclePredictionError("candidate J production dependencies are unavailable") from exc

    if not isinstance(artifact, dict):
        raise CyclePredictionError("candidate J artifact must be metadata JSON")
    cjs.validate_artifact(artifact)

    model_reader = getattr(store, "load_model_by_name", None)
    trace_reader = getattr(store, "prediction_traces_for_model", None)
    feedback_reader = getattr(store, "feedback_history_until", None)
    if model_reader is None or trace_reader is None or feedback_reader is None:
        raise CyclePredictionError("prediction store does not support Candidate J production dependencies")

    g_model = model_reader(cjs.CHAMPION_MODEL_NAME)
    if str(g_model.get("model_id")) == str(model.get("model_id")):
        raise CyclePredictionError("Candidate J dependency cannot reference itself")
    if not g_model.get("artifact_uri"):
        raise CyclePredictionError("Candidate G dependency is missing artifact_uri")

    try:
        g_artifact = load_candidate_g_artifact(str(g_model["artifact_uri"]))
    except Exception as exc:
        raise CyclePredictionError("cannot load Candidate G dependency artifact") from exc

    feedback_frame = feedback_reader(cycle.data_cutoff, g_model)
    g_predictions = build_predictions(
        cycle,
        str(g_model["model_id"]),
        g_artifact,
        observations,
        feedback_frame=feedback_frame,
    )
    validate_predictions(cycle, g_predictions)

    current_trace_rows: list[dict[str, Any]] = []
    for prediction in g_predictions:
        trace = prediction.trace or {}
        if trace.get("candidate") != "candidate_g_v1":
            raise CyclePredictionError("Candidate G dependency did not emit production trace")
        current_trace_rows.append(
            {
                "cycle_id": prediction.cycle_id,
                "station_id": prediction.station_id,
                "target_at": prediction.target_at,
                "horizon_minutes": prediction.horizon_minutes,
                "base_prediction": trace.get("base_prediction"),
                "calibration_factor": trace.get("calibration_factor"),
                "final_prediction": trace.get("final_prediction", prediction.prediction),
                "data_cutoff": trace.get("data_cutoff", cycle.data_cutoff),
            }
        )
    g_trace = pd.DataFrame(current_trace_rows)

    cutoff = pd.Timestamp(cycle.data_cutoff).tz_convert("UTC")

    def fallback_to_g(reason: str) -> tuple[list[CyclePrediction], dict[str, Any], list[CyclePrediction]]:
        fallback = [
            CyclePrediction(
                cycle_id=prediction.cycle_id,
                model_id=str(model["model_id"]),
                station_id=prediction.station_id,
                target_at=prediction.target_at,
                horizon_minutes=prediction.horizon_minutes,
                prediction=prediction.prediction,
                trace={
                    "candidate": cjs.MODEL_VERSION,
                    "regime": "FALLBACK_G",
                    "path": "g",
                    "guard_fired": False,
                    "g_prediction": prediction.prediction,
                    "fallback_reason": reason,
                    "data_cutoff": _ts(cutoff),
                },
            )
            for prediction in g_predictions
        ]
        return fallback, g_model, g_predictions

    try:
        history_start = cutoff - cjs.HISTORY_WINDOW
        g_history = trace_reader(
            str(g_model["model_id"]),
            _ts(history_start),
            _ts(cutoff),
        )
        cycle_targets = pd.DataFrame(
            [
                {
                    "station_id": target.station_id,
                    "target_at": target.target_at,
                    "horizon_minutes": target.horizon_minutes,
                    "target_order": target.target_order,
                }
                for target in cycle.targets
            ]
        )
        composed = cjs.build_cycle_predictions(
            cycle_targets=cycle_targets,
            g_trace=g_trace,
            g_history=g_history,
            observations=observations,
            data_cutoff=cutoff,
            thresholds=cjs.thresholds_from_artifact(artifact),
        )
        by_target = {
            (str(row.station_id), _ts(row.target_at)): row
            for row in composed.itertuples(index=False)
        }
        predictions: list[CyclePrediction] = []
        for target in cycle.targets:
            key = (str(target.station_id), _ts(target.target_at))
            row = by_target.get(key)
            if row is None:
                raise CyclePredictionError(f"Candidate J missing target {key}")
            predictions.append(
                CyclePrediction(
                    cycle_id=cycle.cycle_id,
                    model_id=str(model["model_id"]),
                    station_id=str(target.station_id),
                    target_at=_ts(target.target_at),
                    horizon_minutes=int(target.horizon_minutes),
                    prediction=float(row.j_prediction),
                    trace={
                        "candidate": cjs.MODEL_VERSION,
                        "regime": str(row.regime),
                        "path": str(row.path),
                        "guard_fired": bool(row.guard_fired),
                        "g_prediction": float(row.final_prediction),
                        "data_cutoff": _ts(cutoff),
                    },
                )
            )
        return predictions, g_model, g_predictions
    except Exception as exc:  # J-specific adaptation must never cost an official submission
        return fallback_to_g(f"{type(exc).__name__}: {exc}")


def _build_non_baseline_predictions(
    cycle: ForecastCycle,
    model_id: str,
    artifact: Any,
    observations: pd.DataFrame,
    *,
    feedback_frame: pd.DataFrame | None = None,
) -> list[CyclePrediction]:
    try:
        from candidate_b_shadow import MODEL_TYPE as CANDIDATE_B_MODEL_TYPE
        from candidate_b_shadow import ShadowArtifact, ShadowCandidateError, build_shadow_predictions
        from candidate_g_adaptive import MODEL_TYPE as CANDIDATE_G_MODEL_TYPE
        from candidate_g_adaptive import CandidateGArtifact, build_candidate_g_predictions
    except ImportError as exc:
        raise CyclePredictionError("candidate model dependencies are unavailable") from exc
    if isinstance(artifact, CandidateGArtifact):
        if artifact.metadata.get("model_type") != CANDIDATE_G_MODEL_TYPE:
            raise CyclePredictionError(f"unsupported candidate G model_type: {artifact.metadata.get('model_type')!r}")
        try:
            traced = build_candidate_g_predictions(cycle, model_id, artifact, observations, feedback_frame=feedback_frame)
            return [
                CyclePrediction(
                    cycle_id=item.prediction.cycle_id,
                    model_id=item.prediction.model_id,
                    station_id=item.prediction.station_id,
                    target_at=item.prediction.target_at,
                    horizon_minutes=item.prediction.horizon_minutes,
                    prediction=item.prediction.prediction,
                    trace={
                        "candidate": "candidate_g_v1",
                        "base_prediction": item.base_prediction,
                        "calibration_factor": item.calibration_factor,
                        "final_prediction": item.prediction.prediction,
                        "feedback_samples": item.feedback_samples,
                        "data_cutoff": item.data_cutoff,
                    },
                )
                for item in traced
            ]
        except ShadowCandidateError as exc:
            raise CyclePredictionError(str(exc)) from exc
    if not isinstance(artifact, ShadowArtifact):
        raise CyclePredictionError(f"unsupported champion artifact type: {type(artifact).__name__}")
    if artifact.metadata.get("model_type") != CANDIDATE_B_MODEL_TYPE:
        raise CyclePredictionError(f"unsupported champion model_type: {artifact.metadata.get('model_type')!r}")
    try:
        return build_shadow_predictions(cycle, model_id, artifact, observations)
    except ShadowCandidateError as exc:
        raise CyclePredictionError(str(exc)) from exc


def _feedback_history_for_model(
    store: PredictionStore,
    cycle: ForecastCycle,
    model: dict[str, Any],
    artifact: Any,
) -> pd.DataFrame | None:
    try:
        from candidate_g_adaptive import CandidateGArtifact
    except ImportError:
        return None
    if not isinstance(artifact, CandidateGArtifact):
        return None
    reader = getattr(store, "feedback_history_until", None)
    if reader is None:
        return None
    return reader(cycle.data_cutoff, model)


def prediction_trace_payloads(predictions: list[CyclePrediction]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        trace = prediction.trace or {}
        if trace.get("candidate") == "candidate_l_v1":
            rows.append({
                "cycle_id": prediction.cycle_id, "model_id": prediction.model_id,
                "station_id": prediction.station_id, "target_at": prediction.target_at,
                "horizon_minutes": prediction.horizon_minutes,
                "base_prediction": prediction.prediction, "calibration_factor": 1.0,
                "final_prediction": prediction.prediction,
                "feedback_samples": trace["train_rows"], "data_cutoff": trace["data_cutoff"],
            })
            continue
        if trace.get("candidate") == CANDIDATE_K_TRACE:
            # same convention as the K shadow: base = seasonal expert (or J on fallback),
            # calibration_factor = gate weight on the seasonal expert, feedback_samples = refit samples
            rows.append(
                {
                    "cycle_id": prediction.cycle_id,
                    "model_id": prediction.model_id,
                    "station_id": prediction.station_id,
                    "target_at": prediction.target_at,
                    "horizon_minutes": prediction.horizon_minutes,
                    "base_prediction": trace.get("seasonal_prediction"),
                    "calibration_factor": trace.get("seasonal_weight"),
                    "final_prediction": prediction.prediction,
                    "feedback_samples": int(trace.get("train_rows") or 0),
                    "data_cutoff": trace.get("data_cutoff"),
                }
            )
            continue
        if trace.get("candidate") != "candidate_g_v1":
            continue
        rows.append(
            {
                "cycle_id": prediction.cycle_id,
                "model_id": prediction.model_id,
                "station_id": prediction.station_id,
                "target_at": prediction.target_at,
                "horizon_minutes": prediction.horizon_minutes,
                "base_prediction": trace.get("base_prediction"),
                "calibration_factor": trace.get("calibration_factor"),
                "final_prediction": trace.get("final_prediction", prediction.prediction),
                "feedback_samples": trace.get("feedback_samples", 0),
                "data_cutoff": trace.get("data_cutoff"),
            }
        )
    return rows


def _time_slot_payloads_for_targets(cycle: ForecastCycle) -> list[dict[str, Any]]:
    by_observed_at = {
        _time_slot_payload(target.target_at)["observed_at"]: _time_slot_payload(target.target_at)
        for target in cycle.targets
    }
    return [by_observed_at[observed_at] for observed_at in sorted(by_observed_at)]


def _prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"station_id", "observed_at", "demand"}
    missing = required - set(observations.columns)
    if missing:
        raise CyclePredictionError(f"missing observation columns: {sorted(missing)}")
    frame = observations.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["demand"] = frame["demand"].astype(float)
    return frame


def _quarter_hour_block(timestamp: pd.Timestamp) -> int:
    local = timestamp.tz_convert("America/Bogota")
    return int(local.hour * 4 + local.minute // 15)


def _ts(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _normalize_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise CyclePredictionError("invalid timestamp")
    return _ts(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    with PulsoForecastCycleApi() as api, SupabaseCyclePredictionStore() as store:
        result = generate_cycle_predictions(api, store)
    print(
        json.dumps(
            {
                "cycle_status": result.status,
                "cycle_id": result.cycle_id,
                "data_cutoff": result.data_cutoff,
                "model_id": result.model_id,
                "expected_predictions": result.expected_predictions,
                "predictions_generated": result.predictions_generated,
                "min_prediction": result.min_prediction,
                "max_prediction": result.max_prediction,
                "target_validation": result.target_validation,
                "submission_post": "not_executed",
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
