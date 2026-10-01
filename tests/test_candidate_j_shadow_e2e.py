"""End-to-end safety of the Candidate J shadow job: the whole run against a fake Supabase,
auditing every HTTP request (complements tests/test_candidate_j_shadow.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import numpy as np
import pandas as pd
import pytest

import candidate_j_regime as cj
import candidate_j_shadow as cjs

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["02300", "03000", "05000", "05100", "06000", "06111", "07105", "07107", "07111", "09000", "09122", "10009"]
G_ID, J_ID, J_RUN = "g-model", "j-model", "j-run"
CUTOFFS = [pd.Timestamp("2026-09-20T08:00:00Z"), pd.Timestamp("2026-09-20T09:00:00Z"), pd.Timestamp("2026-09-20T10:00:00Z")]
OLD_CUTOFF = pd.Timestamp("2026-09-18T08:00:00Z")  # before the design freeze


def profile(times: pd.DatetimeIndex) -> np.ndarray:
    local = times.tz_convert("America/Bogota")
    hour = np.asarray(local.hour + local.minute / 60.0)
    return 100 + 400 * np.exp(-((hour - 7.0) ** 2) / 2.0) + 300 * np.exp(-((hour - 18.0) ** 2) / 2.0)


def observations() -> pd.DataFrame:
    times = pd.date_range("2026-08-15T00:00:00Z", "2026-09-20T12:00:00Z", freq="15min")
    base = profile(times)
    return pd.concat(
        [pd.DataFrame({"station_id": s, "observed_at": times, "demand": np.round(base * (1 + 0.05 * i))}) for i, s in enumerate(STATIONS)],
        ignore_index=True,
    )


def trace_rows(cutoff: pd.Timestamp, obs: pd.DataFrame, factor: float = 1.1) -> list[dict]:
    rows = []
    for i, s in enumerate(STATIONS):
        for h in (15, 30, 45, 60):
            target = cutoff + pd.Timedelta(minutes=h)
            base = float(profile(pd.DatetimeIndex([target]))[0] * (1 + 0.05 * i))
            rows.append({"model_id": G_ID, "cycle_id": f"c_{cutoff:%Y%m%dT%H}", "station_id": s, "target_at": target.isoformat(), "horizon_minutes": h,
                         "base_prediction": base, "calibration_factor": factor, "final_prediction": base * factor, "data_cutoff": cutoff.isoformat()})
    return rows


class FakeSupabase:
    """Minimal PostgREST: eq / gte / in filters, and=(gt,lte) on observed_at, limit/offset paging."""

    def __init__(self, *, j_active: bool = False, j_registered: bool = True, cutoffs=CUTOFFS) -> None:
        obs = observations()
        self.tables = {
            "model_versions": pd.DataFrame(
                [{"model_id": G_ID, "model_name": cjs.CHAMPION_MODEL_NAME, "model_type": "g", "is_active": True, "run_id": "g-run", "feature_config": {}}]
                + ([{"model_id": J_ID, "model_name": cjs.MODEL_NAME, "model_type": cjs.MODEL_TYPE, "is_active": j_active, "run_id": J_RUN, "feature_config": {"version": cjs.MODEL_VERSION}}] if j_registered else [])
            ),
            "observations": obs.assign(observed_at=obs["observed_at"].map(lambda t: t.isoformat())),
            "prediction_traces": pd.DataFrame([r for c in cutoffs for r in trace_rows(c, obs)]),
        }
        traces = self.tables["prediction_traces"]
        self.tables["predictions"] = traces.rename(columns={"target_at": "target_observed_at", "final_prediction": "prediction"})[["model_id", "cycle_id", "station_id", "target_observed_at", "horizon_minutes", "prediction"]].copy()
        self.tables["forecast_cycle_targets"] = traces[["cycle_id", "station_id", "target_at", "horizon_minutes"]].assign(target_order=range(len(traces)))
        self.requests: list[tuple[str, str, dict, object]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.split("/rest/v1")[-1].strip("/")
        params = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, f"{request.url.host}/{path}", params, body))
        table = self.tables[path]
        if request.method == "GET":
            frame = self._filter(table, params)
            offset, limit = int(params.get("offset", 0)), int(params.get("limit", 10**9))
            return httpx.Response(200, json=json.loads(frame.iloc[offset : offset + limit].to_json(orient="records")))
        if request.method == "DELETE":
            mask = self._mask(table, params)
            self.tables[path] = table[~mask].reset_index(drop=True)
            return httpx.Response(204)
        if request.method == "POST":
            self.tables[path] = pd.concat([table, pd.DataFrame(body)], ignore_index=True)
            return httpx.Response(201, json=[])
        return httpx.Response(405)

    def _mask(self, table: pd.DataFrame, params: dict) -> pd.Series:
        mask = pd.Series(True, index=table.index)
        for key, value in params.items():
            if key in ("select", "order", "limit", "offset"):
                continue
            if key == "and":
                low, high = re.findall(r"observed_at\.gt\.([^,]+),observed_at\.lte\.([^)]+)", value)[0]
                times = pd.to_datetime(table["observed_at"], utc=True)
                mask &= (times > pd.Timestamp(low)) & (times <= pd.Timestamp(high))
                continue
            op, _, operand = value.partition(".")
            column = table[key]
            if op == "eq":
                mask &= column.astype(str).str.lower() == operand.lower()
            elif op == "gte":
                mask &= pd.to_datetime(column, utc=True) >= pd.Timestamp(operand)
            elif op == "in":
                mask &= column.astype(str).isin(operand.strip("()").split(","))
        return mask

    def _filter(self, table: pd.DataFrame, params: dict) -> pd.DataFrame:
        return table[self._mask(table, params)]

    def store(self) -> cjs.SupabaseCandidateJShadowStore:
        return cjs.SupabaseCandidateJShadowStore("https://supabase.test", "service-key", transport=httpx.MockTransport(self.handler))

    def writes(self) -> list[tuple[str, str, dict, object]]:
        return [r for r in self.requests if r[0] != "GET"]


@pytest.fixture()
def artifact(tmp_path: Path) -> str:
    thresholds = cj.RegimeThresholds.uniform(STATIONS, level24_abs_max=0.15, dispersion24_max=0.2, jump_abs_max=0.8, hour_anomaly_abs_max=0.6)
    metadata = cjs.build_artifact_metadata({"thresholds": thresholds.to_dict(), "decision": {"decision": "continue_shadow"}})
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return str(tmp_path)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PULSO_API_KEY", raising=False)


def test_shadow_run_writes_48_j_predictions_per_cycle_and_nothing_else(artifact: str) -> None:
    fake = FakeSupabase()
    with fake.store() as store:
        result = cjs.run_candidate_j_shadow(store, artifact_uri=artifact)

    assert result.status == "shadow_predicted", result
    assert [c.status for c in result.cycles] == ["shadow_predicted"] * 3
    j_rows = fake.tables["predictions"][fake.tables["predictions"]["model_id"] == J_ID]
    assert j_rows.groupby("cycle_id").size().to_dict() == {f"c_{c:%Y%m%dT%H}": 48 for c in CUTOFFS}
    assert (j_rows["run_id"] == J_RUN).all()
    for method, target, params, body in fake.writes():
        assert target == "supabase.test/predictions"  # the only table J writes
        if method == "DELETE":
            assert params["model_id"] == f"eq.{J_ID}"
        else:
            assert {row["model_id"] for row in body} == {J_ID}
    assert {r[1].split("/")[0] for r in fake.requests} == {"supabase.test"}  # never the competition API
    assert not any("submission" in r[1] for r in fake.requests)


def test_second_run_is_idempotent(artifact: str) -> None:
    fake = FakeSupabase()
    with fake.store() as store:
        cjs.run_candidate_j_shadow(store, artifact_uri=artifact)
        before = len(fake.writes())
        again = cjs.run_candidate_j_shadow(store, artifact_uri=artifact)

    assert again.status == "already_shadow_predicted"
    assert len(fake.writes()) == before


def test_refuses_to_run_with_the_submission_key(artifact: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PULSO_API_KEY", "secret")
    fake = FakeSupabase()
    with fake.store() as store:
        result = cjs.run_candidate_j_shadow(store, artifact_uri=artifact)

    assert result.status == "failed" and "PULSO_API_KEY" in result.reason
    assert fake.requests == []


def test_refuses_when_j_is_active(artifact: str) -> None:
    fake = FakeSupabase(j_active=True)
    with fake.store() as store:
        result = cjs.run_candidate_j_shadow(store, artifact_uri=artifact)

    assert result.status == "refused"
    assert fake.writes() == []


def test_not_registered(artifact: str) -> None:
    fake = FakeSupabase(j_registered=False)
    with fake.store() as store:
        assert cjs.run_candidate_j_shadow(store, artifact_uri=artifact).status == "not_registered"
    assert fake.writes() == []


def test_only_cycles_after_the_design_freeze_are_shadowed(artifact: str) -> None:
    fake = FakeSupabase(cutoffs=[OLD_CUTOFF] + CUTOFFS)
    with fake.store() as store:
        result = cjs.run_candidate_j_shadow(store, artifact_uri=artifact)

    assert OLD_CUTOFF.isoformat().replace("+00:00", "Z") not in [c.data_cutoff for c in result.cycles]
    assert len(result.cycles) == 3


def test_cycle_is_skipped_when_g_trace_does_not_match_official_g(artifact: str) -> None:
    fake = FakeSupabase()
    preds = fake.tables["predictions"]
    preds.loc[(preds["cycle_id"] == "c_20260920T09").idxmax(), "prediction"] += 50.0
    with fake.store() as store:
        result = cjs.run_candidate_j_shadow(store, artifact_uri=artifact)

    statuses = {c.cycle_id: c.status for c in result.cycles}
    assert statuses["c_20260920T09"] == "failed" and statuses["c_20260920T10"] == "shadow_predicted"
    assert "c_20260920T09" not in set(fake.tables["predictions"].loc[fake.tables["predictions"]["model_id"] == J_ID, "cycle_id"])


def test_tampered_artifact_is_refused(tmp_path: Path, artifact: str) -> None:
    metadata = json.loads((Path(artifact) / "metadata.json").read_text())
    metadata["config"]["global_shock_share"] = 0.5
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "metadata.json").write_text(json.dumps(metadata))
    fake = FakeSupabase()
    with fake.store() as store:
        result = cjs.run_candidate_j_shadow(store, artifact_uri=str(tmp_path / "x"))

    assert result.status == "failed" and "frozen" in result.reason
    assert fake.writes() == []


def test_cycle_predictions_ignore_observations_after_the_cutoff(artifact: str) -> None:
    obs = observations()
    thresholds = cjs.thresholds_from_artifact(cjs.load_artifact(artifact))
    cutoff = CUTOFFS[-1]
    traces = pd.DataFrame([r for c in CUTOFFS for r in trace_rows(c, obs)])
    trace = traces[traces["data_cutoff"] == cutoff.isoformat()]
    targets = trace[["station_id", "target_at", "horizon_minutes"]]
    history = traces[traces["data_cutoff"] < cutoff.isoformat()]
    kwargs = dict(cycle_targets=targets, g_trace=trace, g_history=history, data_cutoff=cutoff, thresholds=thresholds)
    clean = cjs.build_cycle_predictions(observations=obs[obs["observed_at"] <= cutoff], **kwargs)
    future = obs.copy()
    future.loc[future["observed_at"] > cutoff, "demand"] *= 100
    leaked = cjs.build_cycle_predictions(observations=future, **kwargs)

    assert len(clean) == 48
    np.testing.assert_allclose(clean["j_prediction"], leaked["j_prediction"])



def test_workflow_cannot_submit_or_block_predict() -> None:
    text = (ROOT / ".github" / "workflows" / "shadow_candidate_j.yml").read_text()
    code_lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    assert not any("PULSO_API_KEY" in line for line in code_lines)
    assert 'workflows: ["Predict"]' in text and "types: [completed]" in text
    assert "github.event.workflow_run.conclusion == 'success'" in text
    assert "group: shadow-candidate-j" in text and "group: predict" not in text
    assert "python scripts/run_candidate_j_shadow.py" in text
