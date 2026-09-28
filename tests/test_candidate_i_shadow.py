from __future__ import annotations

import ast
import copy
import inspect
import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

import candidate_i_shadow as shadow
import candidate_i_snr_gate as ci
from candidate_g_adaptive import MODEL_TYPE as G_MODEL_TYPE
from candidate_g_adaptive import build_candidate_g_artifact, save_candidate_g_artifact
from cycle_predictions import CyclePrediction
from forecast_cycle_sync import CycleTarget, ForecastCycle
from official_submission import StoredPrediction, run_official_cycle

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ("02300", "03000", "05000", "05100", "06000", "06111", "07105", "07107", "07111", "09000", "09122", "10009")
CUTOFF = pd.Timestamp("2026-09-20T12:00:00Z")
CYCLE_ID = "cyc_shadow_test"
G_ID = "11111111-1111-4111-8111-111111111111"
I_ID = "33333333-3333-4333-8333-333333333333"


def _z(ts: pd.Timestamp) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def observations(days: int = 18, drift: dict[str, float] | None = None) -> pd.DataFrame:
    start = CUTOFF.floor("D") - pd.Timedelta(days=days - 1)
    times = pd.date_range(start, CUTOFF, freq="15min", tz="UTC")
    local = times - pd.Timedelta(hours=5)
    hour = local.hour + local.minute / 60.0
    rng = np.random.default_rng(7)
    frames = []
    for k, sid in enumerate(STATIONS):
        demand = 50.0 * (k + 2) * (1.2 + np.sin((hour - 6.0) / 24.0 * 2 * np.pi)) * rng.lognormal(0.0, 0.05, len(times))
        if drift and sid in drift:
            demand = np.where(times > CUTOFF - pd.Timedelta(hours=10), demand * drift[sid], demand)
        frames.append(pd.DataFrame({"station_id": sid, "observed_at": times, "demand": demand}))
    return pd.concat(frames, ignore_index=True)


def official_cycle() -> ForecastCycle:
    targets = [
        CycleTarget(station_id=sid, target_at=_z(CUTOFF + pd.Timedelta(minutes=h)), horizon_minutes=h, target_order=i * 4 + j, raw_target={})
        for i, sid in enumerate(STATIONS)
        for j, h in enumerate((15, 30, 45, 60))
    ]
    return ForecastCycle(cycle_id=CYCLE_ID, data_cutoff=_z(CUTOFF), closes_at=_z(CUTOFF + pd.Timedelta(minutes=10)), expected_predictions=48, targets=targets, raw_cycle={})


class Db:
    """In-memory stand-in for the Supabase tables both paths touch."""

    def __init__(self, g_artifact_uri: str, i_artifact_uri: str, obs: pd.DataFrame) -> None:
        self.obs = obs
        self.cycles: dict[str, ForecastCycle] = {}
        self.predictions: dict[tuple[str, str], list[CyclePrediction]] = {}
        self.traces: dict[tuple[str, str], list[dict]] = {}
        self.receipts: list[dict] = []
        self.g_model = {
            "model_id": G_ID,
            "run_id": "22222222-2222-4222-8222-222222222222",
            "model_type": G_MODEL_TYPE,
            "artifact_uri": g_artifact_uri,
            "trained_until": "2026-09-09T04:45:00Z",
            "is_active": True,
        }
        self.i_model = {
            "model_id": I_ID,
            "run_id": "44444444-4444-4444-8444-444444444444",
            "model_type": ci.MODEL_TYPE,
            "artifact_uri": i_artifact_uri,
            "trained_until": ci.TRAINED_UNTIL,
            "feature_config": {"version": ci.MODEL_VERSION},
            "is_active": False,
        }


class FakeApi:
    def __init__(self, cycle: ForecastCycle) -> None:
        self.cycle = cycle
        self.payloads: list[dict] = []

    def validate_identity(self):
        return {"ok": True}

    def current_cycle(self):
        return self.cycle

    def post_submission(self, payload, *, idempotency_key):
        self.payloads.append(copy.deepcopy(payload))
        return httpx.Response(201, json={"submission_id": f"sub_{len(self.payloads)}", "is_official": True}, request=httpx.Request("POST", "https://pulso.test/v1/submissions"))


class GPredictionStore:
    def __init__(self, db: Db) -> None:
        self.db = db

    def load_active_champion(self):
        return dict(self.db.g_model)

    def observations_until(self, data_cutoff):
        return self.db.obs[self.db.obs["observed_at"] <= pd.Timestamp(data_cutoff)].copy()

    def persist_cycle_targets(self, cycle):
        self.db.cycles[cycle.cycle_id] = cycle

    def persist_predictions(self, cycle, model, predictions):
        self.db.predictions[(cycle.cycle_id, model["model_id"])] = list(predictions)
        return len(predictions)


class GSubmissionStore:
    def __init__(self, db: Db) -> None:
        self.db = db

    def load_active_champion(self):
        return dict(self.db.g_model)

    def predictions_for_cycle(self, cycle_id, model_id):
        rows = sorted(self.db.predictions[(cycle_id, model_id)], key=lambda p: (p.station_id, p.target_at))
        return [StoredPrediction(p.station_id, p.target_at, p.prediction) for p in rows]

    def official_receipt_exists(self, cycle_id):
        return any(r["cycle_id"] == cycle_id and r.get("is_official") for r in self.db.receipts)

    def persist_receipt(self, receipt):
        self.db.receipts.append(receipt)


class IShadowStore:
    """Implements exactly the CandidateIShadowStore protocol; records every call."""

    def __init__(self, db: Db) -> None:
        self.db = db
        self.calls: list[str] = []
        self.max_observation_seen: pd.Timestamp | None = None

    def candidate_i_model_version(self, version=ci.MODEL_VERSION):
        self.calls.append("candidate_i_model_version")
        return dict(self.db.i_model)

    def champion_cycle(self, cycle_id=None):
        self.calls.append("champion_cycle")
        champion = [key for key in self.db.predictions if key[1] == G_ID]
        if not champion:
            return None
        cid = cycle_id or champion[-1][0]
        rows = self.db.predictions[(cid, G_ID)]
        return self.db.cycles[cid], [(p.station_id, p.target_at, p.horizon_minutes) for p in rows]

    def shadow_predictions_complete(self, cycle, model):
        self.calls.append("shadow_predictions_complete")
        rows = self.db.predictions.get((cycle.cycle_id, model["model_id"]), [])
        return len(rows) == cycle.expected_predictions and {(p.station_id, p.target_at) for p in rows} == {(t.station_id, t.target_at) for t in cycle.targets}

    def observations_until(self, data_cutoff):
        self.calls.append("observations_until")
        frame = self.db.obs[self.db.obs["observed_at"] <= pd.Timestamp(data_cutoff)].copy()
        self.max_observation_seen = frame["observed_at"].max()
        return frame

    def persist_shadow_predictions(self, cycle, model, predictions):
        self.calls.append("persist_shadow_predictions")
        self.db.predictions[(cycle.cycle_id, model["model_id"])] = list(predictions)
        return len(predictions)

    def persist_candidate_i_traces(self, cycle_id, model_id, rows):
        self.calls.append("persist_candidate_i_traces")
        self.db.traces[(cycle_id, model_id)] = list(rows)


def new_db(tmp: Path, obs: pd.DataFrame | None = None) -> Db:
    g_uri, i_uri = tmp / "candidate_g_v1", tmp / "candidate_i_v1"
    if not g_uri.exists():
        save_candidate_g_artifact(build_candidate_g_artifact(), artifact_uri=str(g_uri))
        ci.save_candidate_i_artifact(ci.build_candidate_i_artifact(), artifact_uri=str(i_uri))
    return Db(str(g_uri), str(i_uri), observations() if obs is None else obs)


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    """Production order: official Candidate G cycle first, then the Candidate I shadow."""
    tmp = tmp_path_factory.mktemp("shadow_flow")
    db = new_db(tmp, observations(drift={"05100": 0.5}))
    api = FakeApi(official_cycle())
    t0 = time.perf_counter()
    official = run_official_cycle(api, GPredictionStore(db), GSubmissionStore(db))
    runtime_g = time.perf_counter() - t0
    g_before = copy.deepcopy(db.predictions[(CYCLE_ID, G_ID)])
    store = IShadowStore(db)
    t0 = time.perf_counter()
    result = shadow.run_candidate_i_shadow(store, time_budget_s=300)
    runtime_i = time.perf_counter() - t0
    return {"tmp": tmp, "db": db, "api": api, "official": official, "result": result, "store": store, "g_before": g_before, "runtime": (runtime_g, runtime_i)}


def seeded_db(flow) -> Db:
    """A fresh db holding only the official G state of `flow` (no shadow rows)."""
    db = new_db(flow["tmp"], flow["db"].obs)
    db.cycles = dict(flow["db"].cycles)
    db.predictions = {(CYCLE_ID, G_ID): list(flow["g_before"])}
    db.receipts = list(flow["db"].receipts)
    return db


def test_g_still_makes_the_only_official_submission(flow) -> None:
    db, api = flow["db"], flow["api"]
    assert flow["official"].status == "accepted"
    assert len(api.payloads) == 1
    payload = api.payloads[0]
    assert payload["model"]["version"] == G_ID
    g_rows = sorted(flow["g_before"], key=lambda p: (p.station_id, p.target_at))
    assert [(p["station_id"], p["target_at"], p["value"]) for p in payload["predictions"]] == [(p.station_id, p.target_at, p.prediction) for p in g_rows]
    assert [r["model_id"] for r in db.receipts] == [G_ID]
    assert db.predictions[(CYCLE_ID, G_ID)] == flow["g_before"]  # the shadow never touches G rows


def test_i_never_submits(flow) -> None:
    assert len(flow["api"].payloads) == 1
    assert all(r["model_id"] != I_ID for r in flow["db"].receipts)
    tree = ast.parse(inspect.getsource(shadow))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    paths = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("/")}
    assert "official_submission" not in imported
    assert not attributes & {"post_submission", "persist_receipt", "submit_for_cycle", "run_official_cycle"}
    assert paths <= {"/model_versions", "/predictions", "/forecast_cycles", "/forecast_cycle_targets", "/prediction_traces", "/pipeline_runs"}
    workflow = (ROOT / ".github" / "workflows" / "shadow_candidate_i.yml").read_text()
    assert "PULSO_API_KEY:" not in workflow and "secrets.PULSO_API_KEY" not in workflow


def test_workflow_schedules_keep_predict_untouched_and_shadow_hourly() -> None:
    workflows = ROOT / ".github" / "workflows"
    predict = (workflows / "predict.yml").read_text()
    assert predict.count("cron:") == 1 and 'cron: "2,12,22,32,42,52 * * * *"' in predict
    assert "group: predict" in predict
    shadow_yml = (workflows / "shadow_candidate_i.yml").read_text()
    assert shadow_yml.count("cron:") == 1 and 'cron: "27 * * * *"' in shadow_yml
    assert "workflow_dispatch:" in shadow_yml and "workflow_run" not in shadow_yml
    assert "group: shadow-candidate-i" in shadow_yml and "timeout-minutes:" in shadow_yml
    assert "PULSO_API_KEY:" not in shadow_yml and "secrets.PULSO_API_KEY" not in shadow_yml


def test_i_generates_48_of_48_on_the_same_targets_and_cutoff(flow) -> None:
    result, db = flow["result"], flow["db"]
    assert result.status == "shadow_predicted", result.reason
    assert (result.predictions_generated, result.expected_predictions, result.target_validation) == (48, 48, "exact")
    assert (result.cycle_id, result.data_cutoff) == (CYCLE_ID, _z(CUTOFF))
    i_rows, g_rows = db.predictions[(CYCLE_ID, I_ID)], db.predictions[(CYCLE_ID, G_ID)]
    assert [(p.cycle_id, p.station_id, p.target_at, p.horizon_minutes) for p in i_rows] == [(p.cycle_id, p.station_id, p.target_at, p.horizon_minutes) for p in g_rows]
    assert {p.model_id for p in i_rows} == {I_ID}
    assert all(np.isfinite(p.prediction) and p.prediction >= 0 for p in i_rows)
    traces = db.traces[(CYCLE_ID, I_ID)]
    assert len(traces) == 48 and {t["data_cutoff"] for t in traces} == {_z(CUTOFF)}
    assert all(t["final_prediction"] == pytest.approx(t["base_prediction"] * t["calibration_factor"]) for t in traces)
    # the synthetic level shift on 05100 is detected, stable stations stay close to F
    assert result.station_factors["05100"]["factor"] < 0.8
    assert result.fallback_stations == 0


def test_no_future_data_reaches_candidate_i(flow) -> None:
    assert flow["store"].max_observation_seen <= CUTOFF
    history_max = [pd.Timestamp(row["history_max_target_at"]) for row in flow["result"].station_factors.values()]
    assert len(history_max) == 12 and max(history_max) <= CUTOFF
    assert all(row["history_slots"] <= ci.NOISE_WINDOW for row in flow["result"].station_factors.values())
    # a store that leaks rows after the cutoff is rejected by the model, and nothing is written
    db = seeded_db(flow)
    leaky = IShadowStore(db)
    leaky.observations_until = lambda data_cutoff: db.obs.assign(observed_at=db.obs["observed_at"] + pd.Timedelta(hours=1))
    result = shadow.run_candidate_i_shadow(leaky, log=lambda _m: None)
    assert result.status == "failed" and "after data_cutoff" in result.reason
    assert (CYCLE_ID, I_ID) not in db.predictions


@pytest.mark.parametrize("failure", ["artifact", "predict", "timeout", "persist", "store", "active"])
def test_candidate_i_failures_are_isolated_from_g(flow, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    db = seeded_db(flow)
    store = IShadowStore(db)
    if failure == "artifact":
        db.i_model["artifact_uri"] = str(flow["tmp"] / "missing_artifact")
    elif failure == "predict":
        monkeypatch.setattr(ci, "build_candidate_i_predictions", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    elif failure == "timeout":
        monkeypatch.setattr(ci, "build_candidate_i_predictions", lambda *a, **k: time.sleep(5))
    elif failure == "persist":
        monkeypatch.setattr(ci, "build_candidate_i_predictions", _fake_items)
        store.persist_shadow_predictions = lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("supabase down"))
    elif failure == "store":
        store.champion_cycle = lambda cycle_id=None: (_ for _ in ()).throw(httpx.ReadTimeout("slow"))
    elif failure == "active":
        db.i_model["is_active"] = True
    logs: list[str] = []
    started = time.perf_counter()
    result = shadow.run_candidate_i_shadow(store, time_budget_s=0.5 if failure == "timeout" else 300, log=logs.append)
    assert result.status == ("timeout" if failure == "timeout" else "failed")
    assert logs and "Candidate G path is unaffected" in logs[-1]
    assert time.perf_counter() - started < 5
    assert (CYCLE_ID, I_ID) not in db.predictions
    assert db.predictions[(CYCLE_ID, G_ID)] == flow["g_before"]
    assert [r["model_id"] for r in db.receipts] == [G_ID]


def test_history_failure_serves_candidate_f_and_is_reported(flow, monkeypatch: pytest.MonkeyPatch) -> None:
    db = seeded_db(flow)
    monkeypatch.setattr(ci, "build_oos_history", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("history unavailable")))
    logs: list[str] = []
    result = shadow.run_candidate_i_shadow(IShadowStore(db), log=logs.append)
    assert result.status == "shadow_predicted" and result.fallback_stations == 12
    assert any("fell back to factor 1.0" in m for m in logs)
    assert all(t["calibration_factor"] == 1.0 for t in db.traces[(CYCLE_ID, I_ID)])


def test_g_runs_normally_after_a_failed_shadow(flow, monkeypatch: pytest.MonkeyPatch) -> None:
    db = new_db(flow["tmp"], flow["db"].obs)
    monkeypatch.setattr(ci, "load_candidate_i_artifact", lambda *_a: (_ for _ in ()).throw(OSError("artifact missing")))
    assert shadow.run_candidate_i_shadow(IShadowStore(db), log=lambda _m: None).status == "failed"
    api = FakeApi(official_cycle())
    official = run_official_cycle(api, GPredictionStore(db), GSubmissionStore(db))
    assert official.status == "accepted" and len(api.payloads) == 1
    assert len(db.predictions[(CYCLE_ID, G_ID)]) == 48 and (CYCLE_ID, I_ID) not in db.predictions


def test_candidate_i_stays_inactive(flow) -> None:
    payload = shadow.candidate_i_model_version_payload(ci.build_candidate_i_artifact())
    assert payload["is_active"] is False
    assert (payload["model_name"], payload["model_type"], payload["trained_until"]) == (ci.MODEL_NAME, ci.MODEL_TYPE, ci.TRAINED_UNTIL)
    assert payload["artifact_uri"] == ci.DEFAULT_ARTIFACT_URI
    assert {"version", "base_model", "history", "data_contract"} <= set(payload["feature_config"])
    assert {"adaptive_layer", "base_hyperparameters"} <= set(payload["params"])
    store = shadow.SupabaseCandidateIShadowStore(supabase_url="https://supabase.test", service_key="k")
    with pytest.raises(shadow.ShadowGuardError):
        store.register_candidate_i_model_version({**payload, "is_active": True}, git_commit="abc")
    allowed = {"candidate_i_model_version", "champion_cycle", "shadow_predictions_complete", "observations_until", "persist_shadow_predictions", "persist_candidate_i_traces"}
    assert set(flow["store"].calls) <= allowed
    assert flow["db"].i_model["is_active"] is False


def test_restart_is_idempotent_and_cold_start_safe(flow, monkeypatch: pytest.MonkeyPatch) -> None:
    first = list(flow["db"].predictions[(CYCLE_ID, I_ID)])
    # a re-run for the same cycle (every Predict run) does not recompute
    db = seeded_db(flow)
    db.predictions[(CYCLE_ID, I_ID)] = list(first)
    monkeypatch.setattr(ci, "build_candidate_i_predictions", lambda *a, **k: pytest.fail("must not recompute a complete cycle"))
    again = shadow.run_candidate_i_shadow(IShadowStore(db))
    assert again.status == "already_shadow_predicted" and db.predictions[(CYCLE_ID, I_ID)] == first
    monkeypatch.undo()
    # a crash that left a partial cycle is recomputed from observations only (no state) and
    # reproduces the first run exactly
    db.predictions[(CYCLE_ID, I_ID)] = first[:10]
    restarted = shadow.run_candidate_i_shadow(IShadowStore(db))
    assert restarted.status == "shadow_predicted"
    assert [(p.station_id, p.target_at, p.prediction) for p in db.predictions[(CYCLE_ID, I_ID)]] == [(p.station_id, p.target_at, p.prediction) for p in first]


def test_nothing_to_shadow_is_a_clean_skip(flow) -> None:
    db = new_db(flow["tmp"], flow["db"].obs)
    assert shadow.run_candidate_i_shadow(IShadowStore(db)).status == "skipped"  # G has not predicted yet
    db = seeded_db(flow)
    unregistered = IShadowStore(db)
    unregistered.candidate_i_model_version = lambda version=ci.MODEL_VERSION: None
    assert shadow.run_candidate_i_shadow(unregistered).status == "skipped"
    db.predictions[(CYCLE_ID, G_ID)] = flow["g_before"][:47]
    assert shadow.run_candidate_i_shadow(IShadowStore(db)).status == "skipped"


def test_supabase_store_reads_g_cycle_and_writes_only_candidate_i_rows(flow) -> None:
    db, obs = flow["db"], flow["db"].obs
    g_rows = flow["g_before"]
    cycle = db.cycles[CYCLE_ID]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path, params = request.url.path.removeprefix("/rest/v1"), dict(request.url.params)
        if request.method == "GET" and path == "/model_versions":
            if params.get("is_active") == "eq.true":
                return httpx.Response(200, json=[{"model_id": G_ID}])
            return httpx.Response(200, json=[{**db.i_model, "artifact_uri": db.i_model["artifact_uri"]}])
        if request.method == "GET" and path == "/predictions":
            if params.get("model_id") == f"eq.{G_ID}" and "limit" in params:
                return httpx.Response(200, json=[{"cycle_id": CYCLE_ID, "created_at": "2026-09-20T12:02:00+00:00"}])
            if params.get("model_id") == f"eq.{G_ID}":
                return httpx.Response(200, json=[{"station_id": p.station_id, "target_observed_at": p.target_at.replace("Z", "+00:00"), "horizon_minutes": p.horizon_minutes} for p in g_rows])
            return httpx.Response(200, json=[])
        if request.method == "GET" and path == "/forecast_cycles":
            return httpx.Response(200, json=[{"cycle_id": CYCLE_ID, "data_cutoff": "2026-09-20T12:00:00+00:00", "closes_at": "2026-09-20T12:10:00+00:00", "expected_predictions": 48}])
        if request.method == "GET" and path == "/forecast_cycle_targets":
            return httpx.Response(200, json=[{"station_id": t.station_id, "target_at": t.target_at.replace("Z", "+00:00"), "horizon_minutes": t.horizon_minutes, "target_order": t.target_order} for t in cycle.targets])
        if request.method == "GET" and path == "/observations":
            assert params["observed_at"] == f"lte.{_z(CUTOFF)}"
            eligible = obs[obs["observed_at"] <= CUTOFF].sort_values(["observed_at", "station_id"])
            page = eligible.iloc[int(params["offset"]): int(params["offset"]) + int(params["limit"])]
            return httpx.Response(200, json=[{"station_id": r.station_id, "observed_at": _z(r.observed_at), "demand": r.demand} for r in page.itertuples()])
        if request.method in ("DELETE", "POST") and path in ("/predictions", "/prediction_traces"):
            return httpx.Response(201 if request.method == "POST" else 204)
        return httpx.Response(500, json={"unexpected": f"{request.method} {path}"})

    store = shadow.SupabaseCandidateIShadowStore(supabase_url="https://supabase.test", service_key="k")
    store._client = httpx.Client(base_url="https://supabase.test/rest/v1", transport=httpx.MockTransport(handler))
    result = shadow.run_candidate_i_shadow(store)
    assert result.status == "shadow_predicted", result.reason
    writes = [r for r in requests if r.method != "GET"]
    assert {r.url.path.removeprefix("/rest/v1") for r in writes} == {"/predictions", "/prediction_traces"}
    assert all(dict(r.url.params).get("model_id") == f"eq.{I_ID}" for r in writes if r.method == "DELETE")
    posted = json.loads(next(r for r in writes if r.method == "POST" and r.url.path.endswith("/predictions")).content)
    assert len(posted) == 48 and {row["model_id"] for row in posted} == {I_ID}
    assert [(row["station_id"], row["target_observed_at"]) for row in posted] == [(t.station_id, t.target_at) for t in cycle.targets]
    assert not any("submission" in str(r.url) for r in requests)


def _fake_items(cycle, model_id, artifact, observations):
    from candidate_i_snr_gate import CandidateIPrediction, neutral_signal

    return [
        CandidateIPrediction(
            prediction=CyclePrediction(cycle_id=cycle.cycle_id, model_id=model_id, station_id=t.station_id, target_at=t.target_at, horizon_minutes=t.horizon_minutes, prediction=1.0),
            base_prediction=1.0,
            signal=neutral_signal("test"),
            data_cutoff=cycle.data_cutoff,
        )
        for t in cycle.targets
    ]
