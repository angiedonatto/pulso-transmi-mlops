"""Safety, causality and evaluation contract of the Candidate K v2.1 shadow job.

Runs without PyTorch: the frozen predictor is replaced by a fake that records what it receives,
and every HTTP request is audited against a fake Supabase.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import numpy as np
import pandas as pd
import pytest

import candidate_k_shadow as cks
from drift_monitor import accuracy_summary

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["02300", "03000", "05000", "05100", "06000", "06111", "07105", "07107", "07111", "09000", "09122", "10009"]
K_ID, K_RUN, J_ID = "k-model", "k-run", "j-model"
BEFORE_FREEZE = pd.Timestamp("2026-09-18T18:00:00Z")
CUTOFFS = [pd.Timestamp("2026-09-18T19:00:00Z"), pd.Timestamp("2026-09-18T20:00:00Z")]


def cycle_id(cutoff: pd.Timestamp) -> str:
    return f"cyc_{cutoff:%Y%m%dT%H%M}"


def observations(until: pd.Timestamp = pd.Timestamp("2026-09-18T23:00:00Z")) -> pd.DataFrame:
    times = pd.date_range("2026-09-10T00:00:00Z", until, freq="15min")
    return pd.concat(
        [pd.DataFrame({"station_id": s, "observed_at": times, "demand": 100.0 + i + np.arange(len(times)) % 7}) for i, s in enumerate(STATIONS)],
        ignore_index=True,
    )


def targets(cutoff: pd.Timestamp) -> pd.DataFrame:
    rows, order = [], 0
    for s in STATIONS:
        for h in (15, 30, 45, 60):
            rows.append({"cycle_id": cycle_id(cutoff), "station_id": s, "target_at": (cutoff + pd.Timedelta(minutes=h)).isoformat(), "horizon_minutes": h, "target_order": order})
            order += 1
    return pd.DataFrame(rows)


class FakePredictor:
    """Stands in for the frozen torch predictor; records the data it is given."""

    def __init__(self, *, drop_one: bool = False) -> None:
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self.drop_one = drop_one

    def predict(self, observations: pd.DataFrame, data_cutoff: pd.Timestamp, cycle_targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        self.calls.append((pd.Timestamp(data_cutoff), pd.to_datetime(observations["observed_at"], utc=True).max()))
        frame = pd.DataFrame({
            "station_id": cycle_targets["station_id"].astype(str),
            "target_at": pd.to_datetime(cycle_targets["target_at"], utc=True),
            "horizon_minutes": cycle_targets["horizon_minutes"].astype(int),
            "k_prediction": 123.0,
            "k_seasonal_prediction": 120.0,
            "k_seasonal_weight": 0.25,
        })
        if self.drop_one:
            frame = frame.iloc[1:]
        return frame, {"refit_at": cks._ts(cks.refit_time(data_cutoff)), "trained": False, "train_rows": 1000}


class FakeSupabase:
    """Minimal PostgREST: eq / gte / lte / in filters, order, limit/offset paging; audits requests."""

    def __init__(self, *, k_active: bool = False, k_registered: bool = True, obs_until: pd.Timestamp = pd.Timestamp("2026-09-18T23:00:00Z")) -> None:
        cutoffs = [BEFORE_FREEZE, *CUTOFFS]
        obs = observations(obs_until)
        self.tables = {
            "model_versions": pd.DataFrame(
                [{"model_id": J_ID, "model_name": cks.J_MODEL_NAME, "model_type": "j", "is_active": True, "run_id": "j-run", "feature_config": {}}]
                + ([{"model_id": K_ID, "model_name": cks.MODEL_NAME, "model_type": cks.MODEL_TYPE, "is_active": k_active, "run_id": K_RUN, "feature_config": {"version": cks.MODEL_VERSION}}] if k_registered else [])
            ),
            "forecast_cycles": pd.DataFrame([{"cycle_id": cycle_id(c), "data_cutoff": c.isoformat(), "closes_at": (c + pd.Timedelta(minutes=70)).isoformat()} for c in cutoffs]),
            "forecast_cycle_targets": pd.concat([targets(c) for c in cutoffs], ignore_index=True),
            "observations": obs.assign(observed_at=obs["observed_at"].map(lambda t: t.isoformat())),
            "predictions": pd.DataFrame(columns=["run_id", "model_id", "cycle_id", "station_id", "target_observed_at", "horizon_minutes", "prediction", "created_at"]),
            "prediction_traces": pd.DataFrame(columns=["model_id", "cycle_id", "station_id", "target_at", "horizon_minutes", "base_prediction", "calibration_factor", "final_prediction", "feedback_samples", "data_cutoff"]),
        }
        self.requests: list[tuple[str, str, dict, object]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.split("/rest/v1")[-1].strip("/")
        params = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, f"{request.url.host}/{path}", params, body))
        table = self.tables[path]
        if request.method == "GET":
            frame = table[self._mask(table, params)]
            if "order" in params:
                column, _, direction = params["order"].split(",")[0].partition(".")
                frame = frame.sort_values(column, ascending=direction != "desc", key=lambda c: pd.to_datetime(c, utc=True) if "_at" in column or column == "data_cutoff" else c)
            offset, limit = int(params.get("offset", 0)), int(params.get("limit", 10**9))
            return httpx.Response(200, json=json.loads(frame.iloc[offset : offset + limit].to_json(orient="records", date_format="iso")))
        if request.method == "DELETE":
            self.tables[path] = table[~self._mask(table, params)].reset_index(drop=True)
            return httpx.Response(204)
        if request.method == "POST":
            rows = pd.DataFrame(body).assign(created_at=pd.Timestamp("2026-09-18T20:30:00Z").isoformat()) if path == "predictions" else pd.DataFrame(body)
            self.tables[path] = pd.concat([table, rows], ignore_index=True)
            return httpx.Response(201, json=[])
        return httpx.Response(405)

    def _mask(self, table: pd.DataFrame, params: dict) -> pd.Series:
        mask = pd.Series(True, index=table.index)
        for key, value in params.items():
            if key in ("select", "order", "limit", "offset"):
                continue
            op, _, operand = value.partition(".")
            column = table[key]
            if op == "eq":
                mask &= column.astype(str).str.lower() == operand.lower()
            elif op in ("gte", "lte", "gt"):
                times = pd.to_datetime(column, utc=True)
                bound = pd.Timestamp(operand)
                mask &= times >= bound if op == "gte" else (times <= bound if op == "lte" else times > bound)
            elif op == "in":
                mask &= column.astype(str).isin(operand.strip("()").split(","))
        return mask

    def store(self) -> cks.SupabaseCandidateKShadowStore:
        return cks.SupabaseCandidateKShadowStore("https://supabase.test", "service-key", transport=httpx.MockTransport(self.handler))

    def writes(self) -> list[tuple[str, str, dict, object]]:
        return [r for r in self.requests if r[0] != "GET"]


@pytest.fixture()
def artifact(tmp_path: Path) -> str:
    weights = tmp_path / "local_pretrained.pt"
    weights.write_bytes(b"frozen local expert weights")
    metadata = cks.build_artifact_metadata(config={"architecture": "staged_re"}, artifact_dir=tmp_path, parity={"status": "test"})
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return str(tmp_path)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PULSO_API_KEY", raising=False)


# --------------------------------------------------------------------------------------
# Isolation from submissions and from torch
# --------------------------------------------------------------------------------------


def test_shadow_module_never_imports_submission_code_nor_torch_at_import_time() -> None:
    tree = ast.parse((ROOT / "src" / "candidate_k_shadow.py").read_text())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module)
    assert not top_level & {"official_submission", "torch", "candidate_k_deep", "candidate_k_runtime"}
    every = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    every |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert "official_submission" not in every


def test_refuses_to_run_with_the_submission_key_and_makes_no_request(monkeypatch: pytest.MonkeyPatch, artifact: str) -> None:
    monkeypatch.setenv("PULSO_API_KEY", "must-not-be-here")
    fake = FakeSupabase()
    predictor = FakePredictor()
    with fake.store() as store:
        result = cks.run_candidate_k_shadow(store, predictor=predictor, artifact_uri=artifact)
    assert result.status == "failed" and "PULSO_API_KEY" in str(result.reason)
    assert fake.requests == [] and predictor.calls == []


def test_registration_payload_is_shadow_only(artifact: str) -> None:
    payload = cks.model_version_payload(cks.load_artifact(artifact))
    assert payload["is_active"] is False
    assert payload["model_name"] == cks.MODEL_NAME and payload["model_type"] == cks.MODEL_TYPE
    assert payload["feature_config"]["design_freeze"]["oos_from_cutoff"] == cks.OOS_FROM_CUTOFF
    fake = FakeSupabase(k_registered=False)
    with fake.store() as store, pytest.raises(cks.CandidateKShadowError):
        store.register_model_version({**payload, "is_active": True}, git_commit="abc")
    assert fake.writes() == []


# --------------------------------------------------------------------------------------
# Runner: writes, idempotency, causality
# --------------------------------------------------------------------------------------


def test_run_writes_48_k_predictions_and_traces_for_post_freeze_cycles_only(artifact: str) -> None:
    fake = FakeSupabase()
    predictor = FakePredictor()
    with fake.store() as store:
        result = cks.run_candidate_k_shadow(store, predictor=predictor, artifact_uri=artifact)

    assert result.status == "shadow_predicted", result
    assert [c.cycle_id for c in result.cycles] == [cycle_id(c) for c in CUTOFFS]  # oldest first, pre-freeze skipped
    k_rows = fake.tables["predictions"]
    assert k_rows.groupby("cycle_id").size().to_dict() == {cycle_id(c): 48 for c in CUTOFFS}
    assert set(k_rows["model_id"]) == {K_ID} and set(k_rows["run_id"]) == {K_RUN}
    traces = fake.tables["prediction_traces"]
    assert traces.groupby("cycle_id").size().to_dict() == {cycle_id(c): 48 for c in CUTOFFS}
    assert set(traces["model_id"]) == {K_ID}
    assert set(traces["calibration_factor"]) == {0.25} and set(traces["base_prediction"]) == {120.0}
    for method, url, params, _ in fake.writes():
        assert method in ("DELETE", "POST")
        assert url in ("supabase.test/predictions", "supabase.test/prediction_traces")
        if method == "DELETE":
            assert params["model_id"] == f"eq.{K_ID}"
    assert {url.split("/")[0] for _, url, _, _ in fake.requests} == {"supabase.test"}
    assert not any("submission" in url for _, url, _, _ in fake.requests)


def test_second_run_is_idempotent(artifact: str) -> None:
    fake = FakeSupabase()
    with fake.store() as store:
        cks.run_candidate_k_shadow(store, predictor=FakePredictor(), artifact_uri=artifact)
        writes_before = len(fake.writes())
        predictor = FakePredictor()
        again = cks.run_candidate_k_shadow(store, predictor=predictor, artifact_uri=artifact)
    assert again.status == "already_shadow_predicted"
    assert len(fake.writes()) == writes_before and predictor.calls == []


@pytest.mark.parametrize("registered,active,status", [(False, False, "not_registered"), (True, True, "refused")])
def test_refuses_without_a_shadow_registration(artifact: str, registered: bool, active: bool, status: str) -> None:
    fake = FakeSupabase(k_registered=registered, k_active=active)
    predictor = FakePredictor()
    with fake.store() as store:
        result = cks.run_candidate_k_shadow(store, predictor=predictor, artifact_uri=artifact)
    assert result.status == status
    assert fake.writes() == [] and predictor.calls == []


def test_predictor_never_sees_observations_after_the_data_cutoff(artifact: str) -> None:
    fake = FakeSupabase(obs_until=pd.Timestamp("2026-09-19T06:00:00Z"))  # store holds many later rows
    predictor = FakePredictor()
    with fake.store() as store:
        cks.run_candidate_k_shadow(store, predictor=predictor, artifact_uri=artifact)
    assert len(predictor.calls) == 2
    for cutoff, latest_seen in predictor.calls:
        assert latest_seen == cutoff


def test_waits_until_observations_reach_the_cutoff(artifact: str) -> None:
    fake = FakeSupabase(obs_until=pd.Timestamp("2026-09-18T19:45:00Z"))  # 20:00Z cycle not yet observable
    predictor = FakePredictor()
    with fake.store() as store:
        result = cks.run_candidate_k_shadow(store, predictor=predictor, artifact_uri=artifact)
    assert [c.cycle_id for c in result.cycles] == [cycle_id(CUTOFFS[0])]
    assert cycle_id(CUTOFFS[1]) in result.waiting_for_data
    assert set(fake.tables["predictions"]["cycle_id"]) == {cycle_id(CUTOFFS[0])}


def test_incomplete_cycle_output_is_never_written(artifact: str) -> None:
    fake = FakeSupabase()
    with fake.store() as store:
        result = cks.run_candidate_k_shadow(store, predictor=FakePredictor(drop_one=True), artifact_uri=artifact)
    assert result.status == "partial" or result.status == "failed"
    assert all(c.status == "failed" for c in result.cycles)
    assert fake.writes() == []


def test_check_only_lists_pending_cycles_without_predicting_or_writing(artifact: str) -> None:
    fake = FakeSupabase()
    predictor = FakePredictor()
    with fake.store() as store:
        result = cks.run_candidate_k_shadow(store, predictor=predictor, artifact_uri=artifact, check_only=True)
    assert result.status == "pending"
    assert result.pending == [cycle_id(c) for c in CUTOFFS]
    assert predictor.calls == [] and fake.writes() == []


# --------------------------------------------------------------------------------------
# Frozen procedure
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cutoff,refit",
    [
        ("2026-09-18T19:00:00Z", "2026-09-18T05:00:00Z"),
        ("2026-09-19T04:00:00Z", "2026-09-18T05:00:00Z"),
        ("2026-09-19T05:00:00Z", "2026-09-19T05:00:00Z"),
        ("2026-09-19T05:45:00Z", "2026-09-19T05:00:00Z"),
    ],
)
def test_refit_time_is_the_latest_0500z_at_or_before_the_cutoff(cutoff: str, refit: str) -> None:
    assert cks.refit_time(pd.Timestamp(cutoff)) == pd.Timestamp(refit)


def test_training_origins_follow_the_frozen_30_minute_grid() -> None:
    index = pd.date_range("2026-07-27T00:00:00Z", periods=8, freq="15min")
    mask = cks.training_origin_mask(index)
    anchor_offsets = ((index - pd.Timestamp(cks.TRAIN_STRIDE_ANCHOR)) // pd.Timedelta(minutes=15)).to_numpy()
    assert mask.tolist() == (anchor_offsets % cks.TRAIN_STRIDE == 0).tolist()
    assert mask.sum() == 4


def test_tampered_artifact_is_refused(tmp_path: Path, artifact: str) -> None:
    metadata = json.loads((Path(artifact) / "metadata.json").read_text())
    cases = {
        "procedure": {**metadata, "procedure": {**metadata["procedure"], "seeds": [0, 1]}},
        "module_hash": {**metadata, "files": {**metadata["files"], "src/candidate_k_deep.py": "0" * 64}},
        "freeze": {**metadata, "design_freeze": {**metadata["design_freeze"], "oos_from_cutoff": "2026-09-18T10:00:00Z"}},
    }
    for name, tampered in cases.items():
        with pytest.raises(cks.CandidateKShadowError):
            cks.validate_artifact(tampered, Path(artifact))
    (Path(artifact) / "local_pretrained.pt").write_bytes(b"different weights")
    with pytest.raises(cks.CandidateKShadowError):
        cks.load_artifact(artifact)


def test_artifact_pins_the_frozen_k_module() -> None:
    metadata = json.loads((ROOT / "artifacts" / "shadow" / cks.MODEL_VERSION / "metadata.json").read_text())
    digest = hashlib.sha256((ROOT / "src" / "candidate_k_deep.py").read_bytes()).hexdigest()
    assert metadata["files"]["src/candidate_k_deep.py"] == digest
    assert metadata["design_freeze"]["oos_from_cutoff"] == cks.OOS_FROM_CUTOFF
    assert metadata["procedure"] == cks.frozen_procedure()
    cks.load_artifact()  # the committed artifact validates against the committed code


# --------------------------------------------------------------------------------------
# Workflow isolation
# --------------------------------------------------------------------------------------


def test_workflow_is_separate_read_only_and_without_submission_secret() -> None:
    text = (ROOT / ".github" / "workflows" / "shadow_candidate_k.yml").read_text()
    code = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))
    assert "PULSO_API_KEY" not in code
    assert re.search(r"permissions:\s*\n\s*contents:\s*read", code)
    assert "group: shadow-candidate-k" in code
    assert "scripts/run_candidate_k_shadow.py --check-only" in code
    assert "requirements/candidate-k-shadow.txt" in code
    requirements = (ROOT / "requirements" / "candidate-k-shadow.txt").read_text()
    assert "download.pytorch.org/whl/cpu" in requirements and "torch==2.2.2+cpu" in requirements
    assert "candidate_k" not in (ROOT / ".github" / "workflows" / "predict.yml").read_text()


# --------------------------------------------------------------------------------------
# Evaluation: J vs K vs baseline_latest with the leaderboard metric
# --------------------------------------------------------------------------------------


def test_baseline_latest_uses_the_last_observation_at_or_before_the_cutoff() -> None:
    cutoff = pd.Timestamp("2026-09-18T19:00:00Z")
    obs = pd.DataFrame({
        "station_id": ["02300", "02300", "02300", "03000"],
        "observed_at": pd.to_datetime(["2026-09-18T18:45:00Z", "2026-09-18T19:00:00Z", "2026-09-18T19:15:00Z", "2026-09-18T18:45:00Z"], utc=True),
        "demand": [10.0, 20.0, 999.0, 7.0],
    })
    cycle_targets = targets(cutoff)
    cycle_targets = cycle_targets[cycle_targets["station_id"].isin(["02300", "03000"])].assign(data_cutoff=cutoff)
    baseline = cks.baseline_latest(obs, cycle_targets)
    values = baseline.groupby("station_id")["baseline_latest_prediction"].unique().to_dict()
    assert values["02300"].tolist() == [20.0]  # the 19:15 value is after the cutoff
    assert values["03000"].tolist() == [7.0]  # missing at the cutoff -> last value before it


def test_comparison_uses_the_leaderboard_metric_on_shared_resolved_cycles() -> None:
    rows = []
    rng = np.random.default_rng(0)
    for cutoff, resolved in ((CUTOFFS[0], True), (CUTOFFS[1], False)):
        for t in targets(cutoff).itertuples(index=False):
            actual = float(rng.integers(50, 150))
            rows.append({"cycle_id": t.cycle_id, "data_cutoff": cutoff, "station_id": t.station_id, "target_at": pd.Timestamp(t.target_at),
                         "horizon_minutes": t.horizon_minutes, "actual": actual if resolved or t.station_id != "02300" else np.nan,
                         "j_prediction": actual * 1.3, "k_prediction": actual * 1.1, "baseline_latest_prediction": actual * 0.5, "k_on_time": True})
    result = cks.evaluate_comparison(pd.DataFrame(rows), min_cycles=24)
    shared = result["rows"]
    assert set(shared["cycle_id"]) == {cycle_id(CUTOFFS[0])}  # the cycle with a missing actual is not scored yet
    assert len(shared) == 48
    for model, column in cks.COMPARED_MODELS.items():
        expected = accuracy_summary(shared.assign(abs_error=(shared[column] - shared["actual"]).abs()))["mean_station_accuracy"]
        assert result["summary"]["pooled"][model]["mean_station_accuracy"] == pytest.approx(expected)
    assert result["summary"]["pooled"]["K"]["mean_station_accuracy"] == pytest.approx(90.0)
    assert set(result["cycles"].columns) >= {"cycle_id", "J", "K", "baseline_latest", "k_minus_j_pp"}
    assert {"station_id", "horizon_minutes"} <= set(result["stations"].columns) | set(result["horizons"].columns)
    assert result["summary"]["status"] == "insufficient_evidence"
    assert result["summary"]["promotion"] == "manual_decision_required"
