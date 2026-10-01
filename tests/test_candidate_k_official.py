"""Official (champion) path of Candidate K v2.1: generation, submission, fallback and rollback.

PyTorch is replaced by a fake predictor except in the last test (skipped without torch).
"""
from __future__ import annotations

import re
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

import candidate_j_shadow as cjs
import candidate_k_shadow as cks
import cycle_predictions as cp
from forecast_cycle_sync import CycleTarget, ForecastCycle
from official_submission import StoredPrediction, run_official_cycle

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["02300", "03000", "05000", "05100", "06000", "06111", "07105", "07107", "07111", "09000", "09122", "10009"]
K_ID, J_ID, G_ID = "8a0e5ad3-f8c8-4cc3-9c1e-e183a80326d4", "3145188c-64ac-465c-93ca-ec06ee0ed998", "g-model"
CUTOFF = "2026-09-19T08:00:00Z"


def _cycle(stations: list[str] = STATIONS, cutoff: str = CUTOFF) -> ForecastCycle:
    base = pd.Timestamp(cutoff)
    targets, order = [], 0
    for station in stations:
        for horizon in (15, 30, 45, 60):
            targets.append(CycleTarget(station, cp._ts(base + pd.Timedelta(minutes=horizon)), horizon, order, {}))
            order += 1
    return ForecastCycle(cycle_id="cyc-k-official", data_cutoff=cutoff, closes_at=cp._ts(base + pd.Timedelta(minutes=25)),
                         expected_predictions=len(targets), targets=targets, raw_cycle={})


def _k_model(active: bool = True) -> dict:
    return {"model_id": K_ID, "run_id": "k-run", "model_name": cks.MODEL_NAME, "model_type": cks.MODEL_TYPE,
            "artifact_uri": cks.DEFAULT_ARTIFACT_URI, "trained_until": cks.DESIGN_DATA_END, "is_active": active}


def _j_model(active: bool = False) -> dict:
    return {"model_id": J_ID, "run_id": "j-run", "model_name": cjs.MODEL_NAME, "model_type": cjs.MODEL_TYPE,
            "artifact_uri": cjs.DEFAULT_ARTIFACT_URI, "trained_until": cjs.DESIGN_DATA_END, "is_active": active}


def _values(cycle: ForecastCycle, model_id: str, offset: float) -> list[cp.CyclePrediction]:
    return [cp.CyclePrediction(cycle.cycle_id, model_id, t.station_id, t.target_at, t.horizon_minutes, 100.0 + offset + i) for i, t in enumerate(cycle.targets)]


class FakeK:
    """Stand-in for candidate_k_runtime.FrozenKPredictor."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[pd.Timestamp] = []

    def predict(self, observations: pd.DataFrame, data_cutoff, cycle_targets: pd.DataFrame):
        self.calls.append(pd.Timestamp(data_cutoff))
        assert pd.to_datetime(observations["observed_at"], utc=True).max() <= pd.Timestamp(data_cutoff)
        if self.fail:
            raise RuntimeError("torch exploded")
        frame = cycle_targets[["station_id", "target_at", "horizon_minutes"]].copy()
        frame["k_prediction"] = 500.0 + np.arange(len(frame))
        frame["k_seasonal_prediction"] = 400.0
        frame["k_seasonal_weight"] = 0.1
        return frame, {"refit_at": "2026-09-19T05:00:00Z", "trained": False, "train_rows": 30500}


class Store:
    """PredictionStore + the dependency readers used by the J path; records every write."""

    def __init__(self, *, active: dict | None = None, fail_dependency_writes: bool = False) -> None:
        self.active = active or _k_model()
        self.persisted: list[tuple[str, list[cp.CyclePrediction]]] = []
        self.writes: list[str] = []
        self.fail_dependency_writes = fail_dependency_writes

    def load_active_champion(self) -> dict:
        return self.active

    def load_model_by_name(self, name: str) -> dict:
        if name == cjs.MODEL_NAME:
            return _j_model(active=self.active["model_id"] == J_ID)
        raise AssertionError(f"unexpected model lookup {name}")

    def observations_until(self, data_cutoff: str) -> pd.DataFrame:
        times = pd.date_range("2026-09-10T00:00:00Z", data_cutoff, freq="15min")
        return pd.concat([pd.DataFrame({"station_id": s, "observed_at": times, "demand": 100.0, "quarter_hour_block": 0}) for s in STATIONS], ignore_index=True)

    def persist_cycle_targets(self, cycle: ForecastCycle) -> None:
        self.writes.append("forecast_cycles")

    def persist_predictions(self, cycle: ForecastCycle, model: dict, predictions: list[cp.CyclePrediction]) -> int:
        if self.fail_dependency_writes and model["model_id"] != self.active["model_id"]:
            raise httpx.ConnectTimeout("supabase blip")
        self.writes.append(f"predictions:{model['model_id']}")
        self.persisted.append((str(model["model_id"]), predictions))
        return len(predictions)

    # submission side (same object plays both stores in the end-to-end test)
    def official_receipt_exists(self, cycle_id: str) -> bool:
        return False

    def predictions_for_cycle(self, cycle_id: str, model_id: str) -> list[StoredPrediction]:
        rows = [p for m, preds in self.persisted if m == model_id for p in preds]
        return [StoredPrediction(p.station_id, p.target_at, p.prediction) for p in rows]

    def persist_receipt(self, receipt: dict) -> None:
        self.receipts = [*getattr(self, "receipts", []), receipt]


@pytest.fixture()
def fake_j(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def builder(cycle, model, artifact, observations, store):
        calls.append(str(model["model_id"]))
        assert model["model_name"] == cjs.MODEL_NAME and artifact["version"] == cjs.MODEL_VERSION
        return _values(cycle, J_ID, 0.0), {"model_id": G_ID, "run_id": "g-run"}, _values(cycle, G_ID, -50.0)

    monkeypatch.setattr(cp, "_build_candidate_j_official_predictions", builder)
    return calls


def _use_k(monkeypatch: pytest.MonkeyPatch, fake: FakeK) -> FakeK:
    monkeypatch.setattr(cp, "_candidate_k_predictor", lambda model, artifact: fake)
    return fake


def test_k_champion_generates_48_predictions_under_its_own_model_id(monkeypatch: pytest.MonkeyPatch, fake_j: list[str]) -> None:
    fake = _use_k(monkeypatch, FakeK())
    store = Store()
    cycle = _cycle()

    result = cp.generate_predictions_for_cycle(cycle, store, model=_k_model())

    assert result.status == "predicted" and result.model_id == K_ID
    assert result.predictions_generated == 48 == cycle.expected_predictions
    assert [m for m, _ in store.persisted] == [G_ID, J_ID, K_ID]  # dependencies first, K last
    k_rows = dict(store.persisted)[K_ID]
    assert len(k_rows) == 48 and {p.model_id for p in k_rows} == {K_ID}
    assert [p.prediction for p in k_rows] == [500.0 + i for i in range(48)]
    assert {p.trace["candidate"] for p in k_rows} == {cks.MODEL_VERSION} and {p.trace["path"] for p in k_rows} == {"k"}
    assert fake.calls == [pd.Timestamp(CUTOFF)]


def test_same_k_model_id_from_generate_to_submit_even_if_the_champion_flips(monkeypatch: pytest.MonkeyPatch, fake_j: list[str]) -> None:
    _use_k(monkeypatch, FakeK())
    store = Store()
    cycle = _cycle()
    flips = iter([_k_model(), _j_model(active=True)])  # a concurrent promotion lands mid-run
    store.load_active_champion = lambda: next(flips)  # type: ignore[method-assign]

    class Api:
        payloads: list[dict] = []

        def validate_identity(self):
            return {"ok": True}

        def current_cycle(self):
            return cycle

        def post_submission(self, payload, *, idempotency_key):
            self.payloads.append(payload)
            return httpx.Response(201, json={"submission_id": "sub_k", "accepted_at": "2026-09-19T08:05:00Z", "is_official": True},
                                  request=httpx.Request("POST", "https://example.test/v1/submissions"))

    api = Api()
    result = run_official_cycle(api, store, store)

    assert result.status == "accepted"
    assert len(api.payloads[0]["predictions"]) == 48
    assert api.payloads[0]["model"]["version"] == K_ID
    assert store.receipts[0]["model_id"] == K_ID and store.receipts[0]["status"] == "accepted"
    assert [p["value"] for p in api.payloads[0]["predictions"]] == [500.0 + i for i in range(48)]


def test_k_failure_falls_back_to_j_values_under_k_model_id(monkeypatch: pytest.MonkeyPatch, fake_j: list[str]) -> None:
    _use_k(monkeypatch, FakeK(fail=True))
    store = Store()
    cycle = _cycle()

    result = cp.generate_predictions_for_cycle(cycle, store, model=_k_model())

    assert result.predictions_generated == 48 and result.model_id == K_ID
    k_rows = dict(store.persisted)[K_ID]
    j_rows = dict(store.persisted)[J_ID]
    assert [p.prediction for p in k_rows] == [p.prediction for p in j_rows]
    assert {p.model_id for p in k_rows} == {K_ID}
    assert {p.trace["path"] for p in k_rows} == {"fallback_j"}
    assert "torch exploded" in k_rows[0].trace["fallback_reason"]


def test_no_submission_material_when_k_and_its_j_fallback_both_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_k(monkeypatch, FakeK(fail=True))
    monkeypatch.setattr(cp, "_build_candidate_j_official_predictions", lambda *args: (_ for _ in ()).throw(RuntimeError("G artifact missing")))
    store = Store()

    with pytest.raises(cp.CyclePredictionError, match="J fallback"):
        cp.generate_predictions_for_cycle(_cycle(), store, model=_k_model())
    assert store.persisted == []


def test_j_failure_does_not_block_k(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_k(monkeypatch, FakeK())
    monkeypatch.setattr(cp, "_build_candidate_j_official_predictions", lambda *args: (_ for _ in ()).throw(RuntimeError("G artifact missing")))
    store = Store()

    result = cp.generate_predictions_for_cycle(_cycle(), store, model=_k_model())

    assert result.predictions_generated == 48
    assert [m for m, _ in store.persisted] == [K_ID]


def test_dependency_write_failure_does_not_block_k(monkeypatch: pytest.MonkeyPatch, fake_j: list[str]) -> None:
    _use_k(monkeypatch, FakeK())
    store = Store(fail_dependency_writes=True)

    result = cp.generate_predictions_for_cycle(_cycle(), store, model=_k_model())

    assert result.predictions_generated == 48
    assert [m for m, _ in store.persisted] == [K_ID]


def test_j_stays_ready_for_rollback(monkeypatch: pytest.MonkeyPatch, fake_j: list[str]) -> None:
    fake = _use_k(monkeypatch, FakeK())
    cycle = _cycle()
    k_store = Store()
    cp.generate_predictions_for_cycle(cycle, k_store, model=_k_model())
    # with K as champion J keeps producing its 48 predictions every cycle (and G its traces)
    assert len(dict(k_store.persisted)[J_ID]) == 48 and len(dict(k_store.persisted)[G_ID]) == 48
    # the K path only writes the cycle and predictions: model_versions is never touched
    assert set(k_store.writes) == {"forecast_cycles", f"predictions:{G_ID}", f"predictions:{J_ID}", f"predictions:{K_ID}"}

    # rollback = flip is_active back to J: Predict then runs the existing J path, without K
    monkeypatch.setattr(cp, "_load_artifact", lambda model: cjs.load_artifact(model["artifact_uri"]))
    j_store = Store(active=_j_model(active=True))
    result = cp.generate_predictions_for_cycle(cycle, j_store, model=_j_model(active=True))
    assert result.model_id == J_ID and result.predictions_generated == 48
    assert [m for m, _ in j_store.persisted] == [G_ID, J_ID]
    assert len(fake.calls) == 1  # K was not called after the rollback


def test_load_artifact_validates_the_frozen_k_artifact() -> None:
    artifact = cp._load_artifact(_k_model())
    assert artifact["version"] == cks.MODEL_VERSION and artifact["procedure"] == cks.frozen_procedure()
    with pytest.raises(cp.CyclePredictionError):
        cp._load_artifact(_k_model(active=False))


def test_k_trace_rows_use_the_existing_trace_columns() -> None:
    cycle = _cycle()
    target = cycle.targets[0]
    k = cp.CyclePrediction(cycle.cycle_id, K_ID, target.station_id, target.target_at, 15, 510.0,
                           trace={"candidate": cks.MODEL_VERSION, "path": "k", "seasonal_prediction": 400.0, "seasonal_weight": 0.2, "train_rows": 30500, "data_cutoff": CUTOFF})
    fallback = cp.CyclePrediction(cycle.cycle_id, K_ID, target.station_id, target.target_at, 15, 101.0,
                                  trace={"candidate": cks.MODEL_VERSION, "path": "fallback_j", "seasonal_prediction": 101.0, "seasonal_weight": None, "train_rows": 0, "data_cutoff": CUTOFF})
    rows = cp.prediction_trace_payloads([k, fallback])
    assert rows[0]["base_prediction"] == 400.0 and rows[0]["calibration_factor"] == 0.2 and rows[0]["feedback_samples"] == 30500
    assert rows[1]["calibration_factor"] is None and rows[1]["final_prediction"] == 101.0
    assert cp.CANDIDATE_K_TRACE == cks.MODEL_VERSION


def test_predict_workflow_prepares_the_k_runtime_only_for_a_k_champion() -> None:
    predict = (ROOT / ".github" / "workflows" / "predict.yml").read_text()
    shadow = (ROOT / ".github" / "workflows" / "shadow_candidate_k.yml").read_text()
    assert "run_official_cycle(api, prediction_store, submission_store)" in predict
    assert re.search(r"if: \$\{\{ steps\.champion\.outputs\.k_active == 'true' \}\}\s*\n\s*run: python -m pip install -r requirements/candidate-k-shadow.txt", predict)
    assert "CANDIDATE_K_CACHE_DIR" in predict and "actions/cache/restore@v4" in predict and "actions/cache/save@v4" in predict
    key = "candidate-k-v2_1-${{ hashFiles('artifacts/shadow/candidate_k_v2_1/metadata.json', 'src/candidate_k_deep.py', 'src/candidate_k_runtime.py') }}"
    assert key in predict and key in shadow  # Predict and the shadow job share the same trained refits
    assert int(re.search(r"timeout-minutes: (\d+)", predict).group(1)) >= 15


def test_official_k_path_with_the_real_frozen_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_j: list[str]) -> None:
    pytest.importorskip("torch")
    import candidate_k_runtime as ckr

    stations = ["02300", "03000", "05000"]
    times = pd.date_range("2026-09-01T05:00:00Z", "2026-09-11T14:00:00Z", freq="15min")
    slot = np.arange(len(times)) % 96
    observations = pd.concat([pd.DataFrame({"station_id": s, "observed_at": times, "demand": (200 + 150 * np.sin(2 * np.pi * slot / 96 + k)).round()})
                              for k, s in enumerate(stations)], ignore_index=True)
    predictor = ckr.FrozenKPredictor(cks.load_artifact(), cks.artifact_dir(), tmp_path, config_overrides={"hidden": 16, "blocks": 1, "epochs": 1, "batch_size": 256},
                                     seeds=(0,), allow_unfrozen=True)
    monkeypatch.setattr(cp, "_candidate_k_predictor", lambda model, artifact: predictor)
    monkeypatch.setenv("CANDIDATE_K_CACHE_DIR", str(tmp_path))
    store = Store()
    store.observations_until = lambda cutoff: observations[observations["observed_at"] <= pd.Timestamp(cutoff)]  # type: ignore[method-assign]
    cycle = _cycle(stations, "2026-09-11T14:00:00Z")

    result = cp.generate_predictions_for_cycle(cycle, store, model=_k_model())

    k_rows = dict(store.persisted)[K_ID]
    assert result.predictions_generated == 12 and {p.trace["path"] for p in k_rows} == {"k"}
    expected, _ = predictor.predict(observations, pd.Timestamp("2026-09-11T14:00:00Z"), cks._normalize_targets(pd.DataFrame(
        [{"station_id": t.station_id, "target_at": t.target_at, "horizon_minutes": t.horizon_minutes} for t in cycle.targets])))
    lookup = {(r.station_id, cp._ts(r.target_at)): r.k_prediction for r in expected.itertuples(index=False)}
    assert [p.prediction for p in k_rows] == pytest.approx([lookup[(p.station_id, p.target_at)] for p in k_rows])
    assert (tmp_path / "UPDATED").exists()  # a freshly trained refit is flagged for the Actions cache
