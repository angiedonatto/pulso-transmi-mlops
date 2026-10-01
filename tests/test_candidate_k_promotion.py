from __future__ import annotations

import json
from urllib.parse import parse_qs

import httpx
import pytest

import candidate_j_shadow as cjs
import candidate_k_promotion as promotion
import candidate_k_shadow as cks

K_ID, J_ID, G_ID = promotion.K_MODEL_ID, "3145188c-64ac-465c-93ca-ec06ee0ed998", "g-model"


class FakeModelVersions:
    def __init__(self, *, active: set[str] | None = None, fail_patch_for: str | None = None, k_id: str = K_ID) -> None:
        active = {J_ID} if active is None else active
        self.rows = [
            {"model_id": G_ID, "model_name": "candidate_g_f4_profile_ratio_g2_ewma", "model_type": "shadow_f4_profile_ratio_g2_ewma", "artifact_uri": "repo://g"},
            {"model_id": J_ID, "model_name": cjs.MODEL_NAME, "model_type": cjs.MODEL_TYPE, "artifact_uri": cjs.DEFAULT_ARTIFACT_URI},
            {"model_id": k_id, "model_name": cks.MODEL_NAME, "model_type": cks.MODEL_TYPE, "artifact_uri": cks.DEFAULT_ARTIFACT_URI},
        ]
        for row in self.rows:
            row["is_active"] = row["model_id"] in active
        self.patches: list[tuple[str, bool]] = []
        self.fail_patch_for = fail_patch_for

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/model_versions")
        params = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        if request.method == "GET":
            return httpx.Response(200, json=[dict(r) for r in self.rows])
        if request.method == "PATCH":
            model_id = params["model_id"].removeprefix("eq.")
            expected = params["is_active"].removeprefix("eq.") == "true"
            body = json.loads(request.content)
            if model_id == self.fail_patch_for and body["is_active"] is False:
                return httpx.Response(503, json={"message": "unavailable"})
            matched = [r for r in self.rows if r["model_id"] == model_id and r["is_active"] is expected]
            for row in matched:
                row["is_active"] = body["is_active"]
                self.patches.append((model_id, body["is_active"]))
            return httpx.Response(200, json=matched)
        return httpx.Response(405)

    def client(self) -> promotion.ModelVersionsClient:
        return promotion.ModelVersionsClient("https://supabase.test", "key", transport=httpx.MockTransport(self.handler))

    def active(self) -> list[str]:
        return [r["model_id"] for r in self.rows if r["is_active"]]


def test_promotion_dry_run_writes_nothing() -> None:
    fake = FakeModelVersions()
    result = promotion.promote(fake.client())
    assert result.status == "dry_run" and fake.patches == [] and fake.active() == [J_ID]


def test_promotion_makes_k_the_only_champion_and_keeps_j_intact() -> None:
    fake = FakeModelVersions()
    result = promotion.promote(fake.client(), apply=True)
    assert result.status == "applied" and result.active_after == [K_ID]
    assert fake.active() == [K_ID]
    assert fake.patches == [(K_ID, True), (J_ID, False)]
    assert any(r["model_id"] == J_ID for r in fake.rows)  # J is never deleted


@pytest.mark.parametrize("active", [{K_ID}, {J_ID, K_ID}, set(), {G_ID}])
def test_promotion_refuses_unless_j_is_the_only_champion(active: set[str]) -> None:
    fake = FakeModelVersions(active=active)
    with pytest.raises(promotion.PromotionError):
        promotion.promote(fake.client(), apply=True)
    assert fake.patches == []


def test_promotion_refuses_a_different_k_model_id() -> None:
    fake = FakeModelVersions(k_id="another-k")
    with pytest.raises(promotion.PromotionError, match="expected the shadow model"):
        promotion.promote(fake.client(), apply=True)
    assert fake.patches == []


def test_failed_second_flip_is_undone_and_j_stays_champion() -> None:
    fake = FakeModelVersions(fail_patch_for=J_ID)
    with pytest.raises(httpx.HTTPStatusError):
        promotion.promote(fake.client(), apply=True)
    assert fake.active() == [J_ID]
    assert fake.patches == [(K_ID, True), (K_ID, False)]


def test_rollback_restores_j_immediately() -> None:
    fake = FakeModelVersions(active={K_ID})
    result = promotion.rollback(fake.client(), apply=True)
    assert result.status == "applied" and fake.active() == [J_ID]
    assert fake.patches == [(J_ID, True), (K_ID, False)]
