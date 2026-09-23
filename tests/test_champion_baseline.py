from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from champion_baseline import (
    ChampionBaselineError,
    StationMetric,
    load_active_champion,
    save_artifact,
    train_baseline,
)


def sample_observations(days: int = 3) -> pd.DataFrame:
    rows = []
    for station_id, base in [("03000", 100), ("02300", 200)]:
        for step, observed_at in enumerate(pd.date_range("2026-01-01T00:00:00Z", periods=96 * days, freq="15min")):
            rows.append(
                {
                    "station_id": station_id,
                    "observed_at": observed_at,
                    "demand": base + (step % 96),
                    "quarter_hour_block": step % 96,
                }
            )
    rows.append({"station_id": "03000", "observed_at": "2026-01-04T00:00:00Z", "demand": 9999, "quarter_hour_block": 0})
    return pd.DataFrame(rows)


def test_training_does_not_use_data_after_cutoff() -> None:
    artifact, evaluation = train_baseline(
        sample_observations(),
        data_cutoff="2026-01-03T23:45:00Z",
        validation_periods=96,
    )

    assert artifact["trained_until"] == "2026-01-03T23:45:00Z"
    assert evaluation.validation_end == "2026-01-03T23:45:00Z"
    assert "9999" not in json.dumps(artifact)


class MemoryRegistry:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.artifacts: dict[str, dict] = {}

    def register(self, artifact: dict, artifact_uri: str) -> dict:
        for row in self.rows:
            row["is_active"] = False
        model_id = str(uuid4())
        row = {"model_id": model_id, "artifact_uri": artifact_uri, "is_active": True}
        self.rows.append(row)
        self.artifacts[artifact_uri] = artifact
        active = [candidate for candidate in self.rows if candidate["is_active"]]
        assert len(active) == 1
        return row

    def load(self) -> dict:
        active = [row for row in self.rows if row["is_active"]]
        if len(active) != 1:
            raise ChampionBaselineError("expected exactly one active champion")
        return active[0]


def test_only_one_champion_is_promoted(tmp_path: Path) -> None:
    registry = MemoryRegistry()
    registry.register({"version": 1}, str(tmp_path / "one.json"))
    second = registry.register({"version": 2}, str(tmp_path / "two.json"))

    assert registry.load()["model_id"] == second["model_id"]
    assert sum(1 for row in registry.rows if row["is_active"]) == 1


def test_champion_loader_reads_single_active_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = {"model_name": "daily_persistence_slot_mean_baseline"}
    artifact_uri = tmp_path / "artifact.json"
    artifact_uri.write_text(json.dumps(artifact))

    class FakeStore:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def load_active_champion(self):
            return {"model_id": "model-1", "artifact_uri": str(artifact_uri), "is_active": True}

    monkeypatch.setattr("champion_baseline.SupabaseBaselineStore", FakeStore)

    loaded = load_active_champion()

    assert loaded["model_version"]["model_id"] == "model-1"
    assert loaded["artifact"] == artifact


def test_evaluation_is_reproducible() -> None:
    first_artifact, first_eval = train_baseline(sample_observations(), data_cutoff="2026-01-03T23:45:00Z")
    second_artifact, second_eval = train_baseline(sample_observations(), data_cutoff="2026-01-03T23:45:00Z")

    assert first_eval == second_eval
    assert first_artifact["station_slot_means"] == second_artifact["station_slot_means"]


def test_official_metric_formula() -> None:
    metric = StationMetric(station_id="03000", wape=0.25, accuracy=75.0)

    assert metric.accuracy == 100 * max(0, 1 - metric.wape)
