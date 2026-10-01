"""Champion switch between Candidate J v1.1 and Candidate K v2.1 (model_versions.is_active only).

`promote` makes K the only active model and leaves J inactive but intact (row, artifact and its
per-cycle predictions keep flowing through Predict), so `rollback` restores J immediately. Both
check their preconditions first, flip the two rows, verify that exactly one model is active, and
undo the first flip if the second one fails. Nothing else is written; nothing is deleted.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import httpx

import candidate_j_shadow as cjs
import candidate_k_shadow as cks

K_MODEL_ID = "8a0e5ad3-f8c8-4cc3-9c1e-e183a80326d4"


class PromotionError(RuntimeError):
    """Raised when a champion switch is not safe or did not verify."""


@dataclass(frozen=True)
class SwitchResult:
    action: str
    status: str
    active_before: list[str]
    active_after: list[str]
    k_model_id: str
    j_model_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelVersionsClient:
    def __init__(self, supabase_url: str | None = None, service_key: str | None = None, transport: httpx.BaseTransport | None = None) -> None:
        url = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        key = service_key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise PromotionError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        self._client = httpx.Client(base_url=f"{url}/rest/v1", headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                    timeout=30.0, transport=transport)

    def __enter__(self) -> "ModelVersionsClient":
        return self

    def __exit__(self, *args: object) -> None:
        self._client.close()

    def rows(self) -> list[dict[str, Any]]:
        response = self._client.get("/model_versions", params={"select": "model_id,model_name,model_type,is_active,artifact_uri"})
        response.raise_for_status()
        return response.json()

    def set_active(self, model_id: str, active: bool, *, expect_current: bool) -> None:
        response = self._client.patch(
            "/model_versions",
            params={"model_id": f"eq.{model_id}", "is_active": f"eq.{str(expect_current).lower()}"},
            headers={"Prefer": "return=representation"},
            json={"is_active": active},
        )
        response.raise_for_status()
        if len(response.json()) != 1:
            raise PromotionError(f"expected to flip exactly one row for {model_id} ({expect_current} -> {active})")


def _state(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    k = [r for r in rows if r["model_name"] == cks.MODEL_NAME and r["model_type"] == cks.MODEL_TYPE]
    j = [r for r in rows if r["model_name"] == cjs.MODEL_NAME and r["model_type"] == cjs.MODEL_TYPE]
    if len(k) != 1 or len(j) != 1:
        raise PromotionError(f"expected one K and one J row, found {len(k)} and {len(j)}")
    if k[0]["model_id"] != K_MODEL_ID:
        raise PromotionError(f"K model_id is {k[0]['model_id']}, expected the shadow model {K_MODEL_ID}")
    active = [r["model_id"] for r in rows if r.get("is_active") is True]
    return k[0], j[0], active


def promote(client: ModelVersionsClient, *, apply: bool = False) -> SwitchResult:
    """J (only active) -> K (only active). Dry-run unless `apply`."""
    k, j, active = _state(client.rows())
    if active != [j["model_id"]]:
        raise PromotionError(f"promotion expects J as the only active model, found {active}")
    cks.load_artifact(k["artifact_uri"])  # frozen K artifact must validate against the committed code
    cjs.load_artifact(j["artifact_uri"])  # J must stay loadable for rollback
    if not apply:
        return SwitchResult("promote", "dry_run", active, active, k["model_id"], j["model_id"])
    client.set_active(k["model_id"], True, expect_current=False)
    try:
        client.set_active(j["model_id"], False, expect_current=True)
    except Exception:
        client.set_active(k["model_id"], False, expect_current=True)  # undo: J stays the only champion
        raise
    return _verified(client, "promote", active, k["model_id"], j["model_id"], expected=k["model_id"])


def rollback(client: ModelVersionsClient, *, apply: bool = False) -> SwitchResult:
    """K (only active) -> J (only active). Dry-run unless `apply`."""
    k, j, active = _state(client.rows())
    if active != [k["model_id"]]:
        raise PromotionError(f"rollback expects K as the only active model, found {active}")
    cjs.load_artifact(j["artifact_uri"])
    if not apply:
        return SwitchResult("rollback", "dry_run", active, active, k["model_id"], j["model_id"])
    client.set_active(j["model_id"], True, expect_current=False)
    try:
        client.set_active(k["model_id"], False, expect_current=True)
    except Exception:
        client.set_active(j["model_id"], False, expect_current=True)
        raise
    return _verified(client, "rollback", active, k["model_id"], j["model_id"], expected=j["model_id"])


def _verified(client: ModelVersionsClient, action: str, before: list[str], k_id: str, j_id: str, *, expected: str) -> SwitchResult:
    _, _, after = _state(client.rows())
    if after != [expected]:
        raise PromotionError(f"{action} did not verify: active models are {after}")
    return SwitchResult(action, "applied", before, after, k_id, j_id)
