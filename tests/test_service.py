# ruff: noqa: E501, E702
from __future__ import annotations

import pytest

from hok_agent.arena import attack_action, wait_action
from hok_agent.service import PixelArenaService, ServiceError


def test_spawned_service_lifecycle_and_capabilities() -> None:
    with PixelArenaService() as first, PixelArenaService() as second:
        health = first.health()
        assert health["identity"] == "pixelarena-structured-1v1-v1"
        assert health["claim_scope"] == "pixelarena_engineering"
        assert health["capabilities"] == {"network": False, "device": False, "external_client": False}
        assert health["process_id"] != second.health()["process_id"]
        reset = first.reset(5); assert reset["outcome"] == "ongoing"
        stepped = first.step(wait_action(), wait_action()); assert stepped["terminal"] is False


def test_service_rejects_illegal_action_and_remains_usable() -> None:
    with PixelArenaService() as service:
        service.reset(5)
        with pytest.raises(ServiceError, match="illegal blue"):
            service.step(attack_action("enemy_crystal"), wait_action())
        response = service.step(wait_action(), wait_action()); assert response["terminal"] is False


def test_close_stops_child_and_fails_closed() -> None:
    service = PixelArenaService(); service.health(); service.close(); assert not service._process.is_alive()
    with pytest.raises(ServiceError, match="closed"):
        service.health()
