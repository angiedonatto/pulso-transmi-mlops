from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

import candidate_j_shadow as cjs


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_j_shadow_module_does_not_import_submission_code() -> None:
    tree = ast.parse((ROOT / "src" / "candidate_j_shadow.py").read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "official_submission" not in imported


def test_forbidden_pulso_api_key_refuses_to_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PULSO_API_KEY", "not-for-shadow")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metadata.json").write_text(json.dumps(_metadata()) + "\n")

    result = cjs.run_candidate_j_shadow(_NoopStore(), artifact_uri=str(artifact))

    assert result.status == "failed"
    assert "PULSO_API_KEY" in str(result.reason)


def test_model_version_payload_and_registration_force_shadow_mode() -> None:
    payload = cjs.model_version_payload(_metadata())
    assert payload["is_active"] is False
    with pytest.raises(cjs.CandidateJShadowError):
        _NoopStore().register_model_version({**payload, "is_active": True}, git_commit="abc")


def test_persist_predictions_refuses_active_model() -> None:
    with cjs.SupabaseCandidateJShadowStore("https://example.test", "key", transport=httpx.MockTransport(lambda request: httpx.Response(500))) as store:
        with pytest.raises(cjs.CandidateJShadowError):
            store.persist_predictions({"is_active": True, "model_id": "j", "run_id": "r"}, "cycle", pd.DataFrame())


def test_persist_predictions_writes_only_predictions_endpoint() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        return httpx.Response(201 if request.method == "POST" else 204, json=[])

    frame = pd.DataFrame(
        {
            "station_id": ["03000"],
            "target_at": [pd.Timestamp("2026-09-18T10:15:00Z")],
            "horizon_minutes": [15],
            "j_prediction": [123.0],
        }
    )
    with cjs.SupabaseCandidateJShadowStore("https://example.test", "key", transport=httpx.MockTransport(handler)) as store:
        written = store.persist_predictions({"is_active": False, "model_id": "j", "run_id": "r"}, "cycle", frame)

    assert written == 1
    assert requests == [("DELETE", "/rest/v1/predictions"), ("POST", "/rest/v1/predictions")]
    assert all("submission" not in path for _, path in requests)


def test_check_targets_requires_exactly_48_targets() -> None:
    targets = pd.DataFrame({"station_id": ["03000"], "target_at": [pd.Timestamp("2026-09-18T10:15:00Z")]})
    trace = pd.DataFrame({"station_id": ["03000"], "target_at": [pd.Timestamp("2026-09-18T10:15:00Z")]})
    with pytest.raises(cjs.CandidateJShadowError, match="exactly"):
        cjs._check_targets(targets, trace)


def test_check_g_parity_refuses_non_reproducible_trace() -> None:
    trace = pd.DataFrame({"base_prediction": [10.0], "calibration_factor": [2.0], "final_prediction": [99.0]})
    with pytest.raises(cjs.CandidateJShadowError, match="official G"):
        cjs._check_g_parity(trace)


def test_runner_skips_cycles_that_already_have_48_predictions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PULSO_API_KEY", raising=False)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "metadata.json").write_text(json.dumps(_metadata()) + "\n")
    store = _AlreadyCompleteStore()

    result = cjs.run_candidate_j_shadow(store, artifact_uri=str(artifact))

    assert result.status == "already_shadow_predicted"
    assert store.persist_called is False


class _NoopStore:
    def register_model_version(self, payload: dict[str, Any], *, git_commit: str | None) -> dict[str, Any]:
        if payload.get("is_active") is not False:
            raise cjs.CandidateJShadowError("Candidate J registration must use is_active=false")
        return {"status": "registered"}


class _AlreadyCompleteStore:
    persist_called = False

    def candidate_j_model_version(self) -> dict[str, Any]:
        return {"model_id": "j-model", "run_id": "j-run", "is_active": False}

    def champion_model_id(self) -> str:
        return "g-model"

    def champion_traces(self, model_id: str, since: pd.Timestamp) -> pd.DataFrame:
        cutoff = pd.Timestamp(cjs.OOS_FROM_CUTOFF)
        return pd.DataFrame(
            {
                "cycle_id": ["cycle-1"] * cjs.EXPECTED_PREDICTIONS,
                "station_id": [f"{i % 12:05d}" for i in range(cjs.EXPECTED_PREDICTIONS)],
                "target_at": [cutoff + pd.Timedelta(minutes=15 * (1 + i % 4)) for i in range(cjs.EXPECTED_PREDICTIONS)],
                "horizon_minutes": [15 * (1 + i % 4) for i in range(cjs.EXPECTED_PREDICTIONS)],
                "base_prediction": [100.0] * cjs.EXPECTED_PREDICTIONS,
                "calibration_factor": [1.0] * cjs.EXPECTED_PREDICTIONS,
                "final_prediction": [100.0] * cjs.EXPECTED_PREDICTIONS,
                "data_cutoff": [cutoff] * cjs.EXPECTED_PREDICTIONS,
            }
        )

    def prediction_count(self, model_id: str, cycle_id: str) -> int:
        return cjs.EXPECTED_PREDICTIONS

    def persist_predictions(self, *args: Any, **kwargs: Any) -> int:
        self.persist_called = True
        return 0


def _metadata() -> dict[str, Any]:
    return {
        "version": cjs.MODEL_VERSION,
        "model_type": cjs.MODEL_TYPE,
        "config": cjs.frozen_config_dict(),
        "thresholds": {
            "per_station": {"03000": {key: 1.0 for key in cjs.cj.THRESHOLD_KEYS}},
            "calibration_start": "2026-09-01T00:00:00Z",
            "calibration_end": "2026-09-02T00:00:00Z",
            "samples": 1,
        },
        "design_freeze": {"oos_from_cutoff": cjs.OOS_FROM_CUTOFF},
    }
