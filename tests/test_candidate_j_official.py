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


def test_candidate_j_official_falls_back_to_g_when_adaptation_fails(monkeypatch) -> None:
    import candidate_g_adaptive as cg

    cycle = _cycle()
    model = {
        "model_id": "j-model",
        "run_id": "j-run",
        "model_name": cjs.MODEL_NAME,
        "model_type": cjs.MODEL_TYPE,
        "artifact_uri": cjs.DEFAULT_ARTIFACT_URI,
        "trained_until": cjs.DESIGN_DATA_END,
        "is_active": True,
    }
    artifact = cjs.load_artifact()
    g_model = {
        "model_id": "g-model",
        "run_id": "g-run",
        "model_name": cjs.CHAMPION_MODEL_NAME,
        "artifact_uri": "repo://artifacts/shadow/candidate_g_v1",
        "is_active": False,
    }
    g_predictions = [
        cp.CyclePrediction(
            cycle_id=cycle.cycle_id,
            model_id="g-model",
            station_id=target.station_id,
            target_at=target.target_at,
            horizon_minutes=target.horizon_minutes,
            prediction=value,
            trace={
                "candidate": "candidate_g_v1",
                "base_prediction": value,
                "calibration_factor": 1.0,
                "final_prediction": value,
                "data_cutoff": cycle.data_cutoff,
            },
        )
        for target, value in zip(cycle.targets, (11.0, 21.0))
    ]

    class OfficialStore:
        def load_model_by_name(self, name: str) -> dict:
            assert name == cjs.CHAMPION_MODEL_NAME
            return g_model

        def prediction_traces_for_model(self, model_id: str, start: str, end: str) -> pd.DataFrame:
            return pd.DataFrame()

        def feedback_history_until(self, data_cutoff: str, model_row: dict) -> pd.DataFrame:
            return pd.DataFrame(columns=["station_id", "target_at", "base_prediction", "actual"])

    monkeypatch.setattr(cg, "load_candidate_g_artifact", lambda *_a, **_k: object())
    monkeypatch.setattr(cp, "build_predictions", lambda *_a, **_k: g_predictions)
    monkeypatch.setattr(cjs, "build_cycle_predictions", lambda **_k: (_ for _ in ()).throw(RuntimeError("adaptation unavailable")))

    j_predictions, dependency_model, dependency_predictions = cp._build_candidate_j_official_predictions(
        cycle,
        model,
        artifact,
        pd.DataFrame([{"station_id": "02300", "observed_at": cycle.data_cutoff, "demand": 1.0}]),
        OfficialStore(),
    )

    assert dependency_model["model_id"] == "g-model"
    assert dependency_predictions == g_predictions
    assert [p.model_id for p in j_predictions] == ["j-model", "j-model"]
    assert [p.prediction for p in j_predictions] == [11.0, 21.0]
    assert all(p.trace and p.trace["regime"] == "FALLBACK_G" for p in j_predictions)
