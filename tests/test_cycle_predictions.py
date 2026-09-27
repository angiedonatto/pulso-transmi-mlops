from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from candidate_b_shadow import save_shadow_artifact, train_candidate_b_artifact
from candidate_g_adaptive import (
    MODEL_NAME as CANDIDATE_G_MODEL_NAME,
    MODEL_TYPE as CANDIDATE_G_MODEL_TYPE,
    CandidateGArtifact,
    build_candidate_g_artifact,
    save_candidate_g_artifact,
)
from cycle_predictions import (
    CyclePredictionError,
    _load_artifact,
    _time_slot_payloads_for_targets,
    build_predictions,
    generate_cycle_predictions,
    validate_predictions,
)
from forecast_cycle_sync import CycleTarget, ForecastCycle
from official_submission import StoredPrediction, build_submission_payload


def cycle(expected_predictions: int = 2) -> ForecastCycle:
    return ForecastCycle(
        cycle_id="cycle-1",
        data_cutoff="2026-01-02T00:00:00Z",
        closes_at="2026-01-02T01:00:00Z",
        expected_predictions=expected_predictions,
        targets=[
            CycleTarget("03000", "2026-01-02T00:15:00Z", 15, 0, {"station_id": "03000", "target_at": "2026-01-02T00:15:00Z"}),
            CycleTarget("02300", "2026-01-02T00:30:00Z", 30, 1, {"station_id": "02300", "target_at": "2026-01-02T00:30:00Z"}),
        ],
        raw_cycle={"cycle_id": "cycle-1"},
    )


def observations(include_after_cutoff: bool = False) -> pd.DataFrame:
    rows = [
        {"station_id": "03000", "observed_at": "2026-01-01T00:15:00Z", "demand": 10},
        {"station_id": "02300", "observed_at": "2026-01-01T00:30:00Z", "demand": 20},
        {"station_id": "03000", "observed_at": "2026-01-02T00:00:00Z", "demand": 99},
    ]
    if include_after_cutoff:
        rows.append({"station_id": "03000", "observed_at": "2026-01-02T00:15:00Z", "demand": 9999})
    return pd.DataFrame(rows)


def candidate_observations(days: int = 20) -> pd.DataFrame:
    rows = []
    observed_at = pd.date_range("2026-01-01T00:00:00Z", periods=days * 96, freq="15min")
    for station_id, offset in [("03000", 100), ("02300", 200)]:
        for idx, timestamp in enumerate(observed_at):
            rows.append(
                {
                    "station_id": station_id,
                    "observed_at": timestamp.isoformat().replace("+00:00", "Z"),
                    "demand": offset + (idx % 96) + (idx // 96),
                }
            )
    return pd.DataFrame(rows)


def candidate_cycle() -> ForecastCycle:
    return ForecastCycle(
        cycle_id="cycle-candidate",
        data_cutoff="2026-01-18T00:00:00Z",
        closes_at="2026-01-18T01:00:00Z",
        expected_predictions=2,
        targets=[
            CycleTarget("03000", "2026-01-18T00:15:00Z", 15, 0, {}),
            CycleTarget("02300", "2026-01-18T01:00:00Z", 60, 1, {}),
        ],
        raw_cycle={},
    )


def artifact(path: Path) -> str:
    payload = {
        "model_type": "baseline_lag96_station_slot_mean",
        "station_slot_means": {"03000|77": 11.0, "02300|78": 22.0},
    }
    path.write_text(json.dumps(payload))
    return str(path)


class FakeApi:
    def __init__(self, current: ForecastCycle | None) -> None:
        self.current = current
        self.identity_validated = False

    def validate_identity(self):
        self.identity_validated = True
        return {"team": "ok"}

    def current_cycle(self):
        return self.current


class MemoryStore:
    def __init__(self, artifact_uri: str, active_count: int = 1) -> None:
        self.artifact_uri = artifact_uri
        self.active_count = active_count
        self.rows: dict[tuple[str, str, str], object] = {}
        self.persisted_cycle: ForecastCycle | None = None

    def load_active_champion(self):
        if self.active_count != 1:
            raise CyclePredictionError(f"expected exactly one active champion, found {self.active_count}")
        return {"model_id": "model-1", "run_id": "run-1", "artifact_uri": self.artifact_uri, "is_active": True}

    def observations_until(self, data_cutoff: str):
        return observations()

    def persist_cycle_targets(self, cycle: ForecastCycle) -> None:
        self.persisted_cycle = cycle

    def persist_predictions(self, cycle: ForecastCycle, model: dict, predictions: list) -> int:
        self.rows = {
            (prediction.cycle_id, prediction.station_id, prediction.target_at): prediction
            for prediction in predictions
        }
        return len(predictions)


class CandidateStore(MemoryStore):
    def load_active_champion(self):
        return {"model_id": "candidate-model", "run_id": "run-1", "artifact_uri": self.artifact_uri, "is_active": True}

    def observations_until(self, data_cutoff: str):
        frame = candidate_observations()
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        return frame[frame["observed_at"] <= pd.Timestamp(data_cutoff)].copy()


class CandidateGStore(CandidateStore):
    def __init__(self, artifact_uri: str, feedback: list[dict] | None = None) -> None:
        super().__init__(artifact_uri)
        self.feedback = pd.DataFrame(feedback or [], columns=["station_id", "target_at", "base_prediction", "actual"])
        self.feedback_cutoffs: list[str] = []

    def load_active_champion(self):
        test_trained_until = "2026-01-17T00:00:00Z"
        return {
            "model_id": "candidate-g-model",
            "run_id": "run-g",
            "model_name": CANDIDATE_G_MODEL_NAME,
            "model_type": CANDIDATE_G_MODEL_TYPE,
            "artifact_uri": self.artifact_uri,
            "trained_until": test_trained_until,
            "created_at": test_trained_until,
            "is_active": True,
            "pipeline_runs": {"started_at": test_trained_until, "git_commit": "test"},
        }

    def feedback_history_until(self, data_cutoff: str, model: dict) -> pd.DataFrame:
        self.feedback_cutoffs.append(data_cutoff)
        if self.feedback.empty:
            return self.feedback.copy()
        frame = self.feedback.copy()
        frame["target_at"] = pd.to_datetime(frame["target_at"], utc=True)
        return frame[frame["target_at"] <= pd.Timestamp(data_cutoff)].copy()

    def predictions_for_cycle(self, cycle_id: str, model_id: str) -> list[StoredPrediction]:
        return [
            StoredPrediction(prediction.station_id, prediction.target_at, prediction.prediction)
            for prediction in self.rows.values()
        ]


def candidate_g_artifact_uri(tmp_path: Path) -> str:
    artifact_uri = str(tmp_path / "candidate_g_v1")
    save_candidate_g_artifact(build_candidate_g_artifact(artifact_uri=artifact_uri), artifact_uri=artifact_uri)
    return artifact_uri


def test_no_open_cycle_finishes_without_predictions(tmp_path: Path) -> None:
    store = MemoryStore(artifact(tmp_path / "artifact.json"))

    result = generate_cycle_predictions(FakeApi(None), store)

    assert result.status == "no_open_cycle"
    assert store.rows == {}


def test_requires_unique_active_champion(tmp_path: Path) -> None:
    store = MemoryStore(artifact(tmp_path / "artifact.json"), active_count=2)

    with pytest.raises(CyclePredictionError, match="active champion"):
        generate_cycle_predictions(FakeApi(cycle()), store)


def test_observations_after_data_cutoff_are_rejected(tmp_path: Path) -> None:
    payload = json.loads(Path(artifact(tmp_path / "artifact.json")).read_text())

    with pytest.raises(CyclePredictionError, match="after data_cutoff"):
        build_predictions(cycle(), "model-1", payload, observations(include_after_cutoff=True))


def test_target_set_must_match_exactly(tmp_path: Path) -> None:
    payload = json.loads(Path(artifact(tmp_path / "artifact.json")).read_text())
    predictions = build_predictions(cycle(), "model-1", payload, observations())
    wrong_cycle = cycle()
    wrong_cycle.targets.reverse()

    with pytest.raises(CyclePredictionError, match="targets"):
        validate_predictions(wrong_cycle, predictions)


def test_prediction_count_must_match_expected(tmp_path: Path) -> None:
    payload = json.loads(Path(artifact(tmp_path / "artifact.json")).read_text())
    predictions = build_predictions(cycle(), "model-1", payload, observations())

    with pytest.raises(CyclePredictionError, match="expected_predictions"):
        validate_predictions(cycle(expected_predictions=3), predictions)


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    store = MemoryStore(artifact(tmp_path / "artifact.json"))
    api = FakeApi(cycle())

    first = generate_cycle_predictions(api, store)
    first_rows = dict(store.rows)
    second = generate_cycle_predictions(api, store)

    assert first.predictions_generated == 2
    assert second.predictions_generated == 2
    assert store.rows == first_rows
    assert len(store.rows) == 2


def test_future_time_slot_payloads_are_complete_and_deduplicated() -> None:
    duplicated_time_cycle = ForecastCycle(
        cycle_id="cycle-dup",
        data_cutoff="2026-01-02T00:00:00Z",
        closes_at="2026-01-02T01:00:00Z",
        expected_predictions=2,
        targets=[
            CycleTarget("03000", "2026-01-02T00:15:00Z", 15, 0, {}),
            CycleTarget("02300", "2026-01-02T00:15:00Z", 15, 1, {}),
        ],
        raw_cycle={},
    )
    payloads = _time_slot_payloads_for_targets(duplicated_time_cycle)

    assert len(payloads) == 1
    assert payloads[0] == {
        "observed_at": "2026-01-02T00:15:00Z",
        "local_date": "2026-01-01",
        "hour": 19,
        "minute": 15,
        "day_of_week": 3,
        "day_name": "jueves",
        "is_weekend": False,
        "quarter_hour_block": 77,
    }


def test_active_candidate_b_artifact_generates_official_predictions(tmp_path: Path) -> None:
    artifact_uri = str(tmp_path / "candidate_b_v1")
    candidate_artifact = train_candidate_b_artifact(
        candidate_observations(),
        training_data_end="2026-01-18T00:00:00Z",
        artifact_uri=artifact_uri,
    )
    save_shadow_artifact(candidate_artifact, artifact_uri=artifact_uri)
    store = CandidateStore(artifact_uri)

    result = generate_cycle_predictions(FakeApi(candidate_cycle()), store)

    assert result.status == "predicted"
    assert result.predictions_generated == candidate_cycle().expected_predictions
    assert [(row[1], row[2]) for row in store.rows] == [
        ("03000", "2026-01-18T00:15:00Z"),
        ("02300", "2026-01-18T01:00:00Z"),
    ]


def test_candidate_g_artifact_loading_is_explicit_and_self_contained(tmp_path: Path) -> None:
    artifact_uri = candidate_g_artifact_uri(tmp_path)
    artifact = _load_artifact(
        {
            "model_id": "candidate-g-model",
            "artifact_uri": artifact_uri,
            "model_type": CANDIDATE_G_MODEL_TYPE,
            "is_active": True,
        }
    )

    assert isinstance(artifact, CandidateGArtifact)
    assert artifact.metadata["version"] == "candidate_g_v1"
    assert artifact.metadata["base_model"]["name"] == "candidate_f_f4_profile_ratio"
    assert artifact.metadata["adaptive_layer"]["alpha"] == 1.0
    assert artifact.metadata["adaptive_layer"]["clip"] == [0.35, 2.5]


def test_active_candidate_g_dispatches_through_official_prediction_path(tmp_path: Path) -> None:
    store = CandidateGStore(candidate_g_artifact_uri(tmp_path))

    result = generate_cycle_predictions(FakeApi(candidate_cycle()), store)
    payload = build_submission_payload(candidate_cycle(), store.load_active_champion(), store.predictions_for_cycle(result.cycle_id or "", result.model_id or ""))

    assert result.status == "predicted"
    assert result.predictions_generated == candidate_cycle().expected_predictions
    assert store.feedback_cutoffs == [candidate_cycle().data_cutoff]
    assert len(payload["predictions"]) == candidate_cycle().expected_predictions
    assert all(prediction.trace and prediction.trace["candidate"] == "candidate_g_v1" for prediction in store.rows.values())


def test_candidate_g_feedback_uses_only_rows_at_or_before_cutoff(tmp_path: Path) -> None:
    store = CandidateGStore(
        candidate_g_artifact_uri(tmp_path),
        feedback=[
            {"station_id": "03000", "target_at": "2026-01-17T23:45:00Z", "base_prediction": 100.0, "actual": 200.0},
            {"station_id": "03000", "target_at": "2026-01-18T00:15:00Z", "base_prediction": 100.0, "actual": 10.0},
        ],
    )

    generate_cycle_predictions(FakeApi(candidate_cycle()), store)
    station_prediction = next(prediction for prediction in store.rows.values() if prediction.station_id == "03000")

    assert station_prediction.trace is not None
    assert station_prediction.trace["feedback_samples"] == 1
    assert station_prediction.trace["calibration_factor"] == pytest.approx(2.0)


def test_candidate_g_cold_runner_reconstruction_is_deterministic(tmp_path: Path) -> None:
    artifact_uri = candidate_g_artifact_uri(tmp_path)
    feedback = [
        {"station_id": "03000", "target_at": "2026-01-17T23:45:00Z", "base_prediction": 100.0, "actual": 150.0},
        {"station_id": "02300", "target_at": "2026-01-17T23:45:00Z", "base_prediction": 200.0, "actual": 180.0},
    ]
    first = CandidateGStore(artifact_uri, feedback=feedback)
    second = CandidateGStore(artifact_uri, feedback=feedback)

    generate_cycle_predictions(FakeApi(candidate_cycle()), first)
    generate_cycle_predictions(FakeApi(candidate_cycle()), second)

    assert [row.prediction for row in first.rows.values()] == pytest.approx([row.prediction for row in second.rows.values()])
    assert [row.trace for row in first.rows.values()] == [row.trace for row in second.rows.values()]


def test_candidate_g_missing_feedback_falls_back_to_base_factor_one(tmp_path: Path) -> None:
    store = CandidateGStore(candidate_g_artifact_uri(tmp_path), feedback=[])

    generate_cycle_predictions(FakeApi(candidate_cycle()), store)

    assert all(prediction.trace is not None for prediction in store.rows.values())
    assert all(prediction.trace["calibration_factor"] == 1.0 for prediction in store.rows.values())
    assert all(prediction.trace["feedback_samples"] == 0 for prediction in store.rows.values())
