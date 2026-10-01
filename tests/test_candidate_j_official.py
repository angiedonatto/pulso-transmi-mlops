from __future__ import annotations

import pandas as pd

import candidate_j_shadow as cjs
import cycle_predictions as cp
from forecast_cycle_sync import CycleTarget, ForecastCycle


def _cycle() -> ForecastCycle:
    return ForecastCycle(
        cycle_id="cycle-j-official",
        data_cutoff="2026-09-18T13:00:00Z",
        closes_at="2026-09-18T13:25:00Z",
        expected_predictions=2,
        targets=[
            CycleTarget("02300", "2026-09-18T13:15:00Z", 15, 0, {}),
            CycleTarget("03000", "2026-09-18T13:30:00Z", 30, 1, {}),
        ],
        raw_cycle={},
    )


def _predictions(cycle: ForecastCycle, model_id: str, values: tuple[float, float]) -> list[cp.CyclePrediction]:
    return [
        cp.CyclePrediction(
            cycle_id=cycle.cycle_id,
            model_id=model_id,
            station_id=target.station_id,
            target_at=target.target_at,
            horizon_minutes=target.horizon_minutes,
            prediction=value,
        )
        for target, value in zip(cycle.targets, values)
    ]


class Store:
    def __init__(self) -> None:
        self.persisted_models: list[str] = []
        self.persisted_cycle = False

    def load_active_champion(self) -> dict:
        return {
            "model_id": "j-model",
            "run_id": "j-run",
            "model_name": cjs.MODEL_NAME,
            "model_type": cjs.MODEL_TYPE,
            "artifact_uri": cjs.DEFAULT_ARTIFACT_URI,
            "trained_until": cjs.DESIGN_DATA_END,
            "is_active": True,
        }

    def observations_until(self, data_cutoff: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"station_id": "02300", "observed_at": data_cutoff, "demand": 10.0},
                {"station_id": "03000", "observed_at": data_cutoff, "demand": 20.0},
            ]
        )

    def persist_cycle_targets(self, cycle: ForecastCycle) -> None:
        self.persisted_cycle = True

    def persist_predictions(self, cycle: ForecastCycle, model: dict, predictions: list[cp.CyclePrediction]) -> int:
        assert self.persisted_cycle
        self.persisted_models.append(str(model["model_id"]))
        return len(predictions)


def test_generate_predictions_for_cycle_runs_g_dependency_before_official_j(monkeypatch) -> None:
    cycle = _cycle()
    store = Store()
    g_model = {"model_id": "g-model", "run_id": "g-run", "artifact_uri": "repo://g"}
    g_predictions = _predictions(cycle, "g-model", (11.0, 21.0))
    j_predictions = _predictions(cycle, "j-model", (12.0, 22.0))

    monkeypatch.setattr(cp, "_load_artifact", lambda model: {"version": cjs.MODEL_VERSION})

    def fake_builder(cycle_arg, model, artifact, observations, store_arg):
        assert cycle_arg is cycle
        assert model["model_id"] == "j-model"
        assert store_arg is store
        return j_predictions, g_model, g_predictions

    monkeypatch.setattr(cp, "_build_candidate_j_official_predictions", fake_builder)

    result = cp.generate_predictions_for_cycle(cycle, store)

    assert result.status == "predicted"
    assert result.model_id == "j-model"
    assert result.predictions_generated == 2
    assert store.persisted_models == ["g-model", "j-model"]


def test_load_artifact_supports_active_candidate_j() -> None:
    artifact = cp._load_artifact(
        {
            "model_id": "j-model",
            "model_type": cjs.MODEL_TYPE,
            "artifact_uri": cjs.DEFAULT_ARTIFACT_URI,
            "is_active": True,
        }
    )

    assert artifact["version"] == cjs.MODEL_VERSION
    assert artifact["model_type"] == cjs.MODEL_TYPE
