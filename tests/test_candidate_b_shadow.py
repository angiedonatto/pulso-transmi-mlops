from __future__ import annotations

from pathlib import Path

import pandas as pd

from candidate_b_shadow import (
    DEFAULT_ARTIFACT_URI,
    FEATURE_SCHEMA,
    MODEL_TYPE,
    MODEL_VERSION,
    RANDOM_STATE,
    ShadowArtifact,
    ShadowCandidateError,
    build_shadow_predictions,
    load_shadow_artifact,
    run_shadow_cycle,
    save_shadow_artifact,
    train_candidate_b_artifact,
)
from forecast_cycle_sync import CycleTarget, ForecastCycle


def sample_observations(days: int = 20) -> pd.DataFrame:
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


def cycle() -> ForecastCycle:
    return ForecastCycle(
        cycle_id="cyc_shadow_1",
        data_cutoff="2026-01-18T00:00:00Z",
        closes_at="2026-01-18T00:10:00Z",
        expected_predictions=4,
        targets=[
            CycleTarget("03000", "2026-01-18T00:15:00Z", 15, 0, {}),
            CycleTarget("02300", "2026-01-18T00:15:00Z", 15, 1, {}),
            CycleTarget("03000", "2026-01-18T01:00:00Z", 60, 2, {}),
            CycleTarget("02300", "2026-01-18T01:00:00Z", 60, 3, {}),
        ],
        raw_cycle={"cycle_id": "cyc_shadow_1"},
    )


def observations_until_cutoff() -> pd.DataFrame:
    frame = sample_observations()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    return frame[frame["observed_at"] <= pd.Timestamp(cycle().data_cutoff)].copy()


class FakeApi:
    def validate_identity(self):
        return {"ok": True}

    def current_cycle(self):
        return cycle()


class FakeShadowStore:
    def __init__(self, artifact_uri: str, *, complete: bool = False) -> None:
        self.artifact_uri = artifact_uri
        self.complete = complete
        self.predictions = []
        self.persisted_cycle: ForecastCycle | None = None

    def observations_until(self, data_cutoff: str) -> pd.DataFrame:
        return observations_until_cutoff()

    def persist_cycle_targets(self, cycle: ForecastCycle) -> None:
        self.persisted_cycle = cycle

    def load_shadow_model_version(self, version: str = MODEL_VERSION) -> dict:
        assert version == MODEL_VERSION
        return {
            "run_id": "22222222-2222-4222-8222-222222222222",
            "model_id": "11111111-1111-4111-8111-111111111111",
            "is_active": False,
            "artifact_uri": self.artifact_uri,
        }

    def shadow_predictions_complete(self, cycle: ForecastCycle, model: dict) -> bool:
        return self.complete

    def persist_shadow_predictions(self, cycle: ForecastCycle, model: dict, predictions: list) -> int:
        assert model["is_active"] is False
        self.predictions = predictions
        return len(predictions)


def test_train_candidate_b_artifact_has_fixed_version_metadata(tmp_path: Path) -> None:
    uri = str(tmp_path / MODEL_VERSION)
    artifact = train_candidate_b_artifact(
        sample_observations(),
        training_data_end="2026-01-18T00:00:00Z",
        artifact_uri=uri,
    )
    saved_uri = save_shadow_artifact(artifact, artifact_uri=uri)
    loaded = load_shadow_artifact(saved_uri)

    assert loaded.metadata["version"] == MODEL_VERSION
    assert loaded.metadata["model_type"] == MODEL_TYPE
    assert loaded.metadata["feature_schema"] == FEATURE_SCHEMA
    assert loaded.metadata["random_state"] == RANDOM_STATE
    assert loaded.metadata["training_data_end"] == "2026-01-18T00:00:00Z"
    assert loaded.metadata["artifact_uri"] == uri
    assert loaded.metadata["station_mapping"] == {"02300": 0, "03000": 1}
    assert (Path(uri) / "metadata.json").exists()
    assert (Path(uri) / "model.pkl").exists()


def test_default_artifact_uri_is_repo_versioned() -> None:
    assert DEFAULT_ARTIFACT_URI == "repo://artifacts/shadow/candidate_b_v1"


def test_build_shadow_predictions_validates_exact_target_set(tmp_path: Path) -> None:
    artifact = train_candidate_b_artifact(sample_observations(), training_data_end="2026-01-18T00:00:00Z")

    predictions = build_shadow_predictions(cycle(), "model-shadow", artifact, observations_until_cutoff())

    assert [(prediction.station_id, prediction.target_at) for prediction in predictions] == [
        (target.station_id, target.target_at) for target in cycle().targets
    ]
    assert all(prediction.prediction >= 0 for prediction in predictions)


def test_build_shadow_predictions_rejects_artifact_after_cycle_cutoff() -> None:
    artifact = train_candidate_b_artifact(sample_observations(), training_data_end="2026-01-18T00:00:00Z")
    bad_artifact = ShadowArtifact(metadata={**artifact.metadata, "training_data_end": "2026-01-18T00:15:00Z"}, model=artifact.model)

    try:
        build_shadow_predictions(cycle(), "model-shadow", bad_artifact, observations_until_cutoff())
    except ShadowCandidateError as exc:
        assert "training_data_end" in str(exc)
    else:
        raise AssertionError("expected ShadowCandidateError")


def test_run_shadow_cycle_loads_fixed_artifact_without_training(tmp_path: Path) -> None:
    uri = str(tmp_path / MODEL_VERSION)
    artifact = train_candidate_b_artifact(sample_observations(), training_data_end="2026-01-18T00:00:00Z", artifact_uri=uri)
    save_shadow_artifact(artifact, artifact_uri=uri)
    store = FakeShadowStore(uri)

    result = run_shadow_cycle(FakeApi(), store)

    assert result.status == "shadow_predicted"
    assert result.predictions_generated == cycle().expected_predictions
    assert result.target_validation == "exact"
    assert store.persisted_cycle == cycle()
    assert len(store.predictions) == cycle().expected_predictions


def test_run_shadow_cycle_is_idempotent_when_predictions_are_complete(tmp_path: Path) -> None:
    uri = str(tmp_path / MODEL_VERSION)
    artifact = train_candidate_b_artifact(sample_observations(), training_data_end="2026-01-18T00:00:00Z", artifact_uri=uri)
    save_shadow_artifact(artifact, artifact_uri=uri)
    store = FakeShadowStore(uri, complete=True)

    result = run_shadow_cycle(FakeApi(), store)

    assert result.status == "already_shadow_predicted"
    assert result.predictions_generated == cycle().expected_predictions
    assert store.predictions == []
    assert store.persisted_cycle is None
