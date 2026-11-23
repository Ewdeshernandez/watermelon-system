"""API POST /v1/collector/ack → node_ack en rm_setup_versions."""

import os

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("WATERMELON_API_KEYS", "test-key")
    import api.app as appmod
    app = appmod.create_app() if hasattr(appmod, "create_app") else appmod.app
    return TestClient(app)


def test_ack_marks_version(client, monkeypatch):
    import core.remote_monitoring.config as cfg
    import core.remote_monitoring.setup_versions as sv
    import core.live_readings as lr
    calls = []
    monkeypatch.setattr(lr, "_get_supabase_client", lambda: object())
    monkeypatch.setattr(cfg, "find_setup_cloud_by_instance",
                        lambda iid, client=None: {"id": "SGT300B"} if iid == "turbina_sgt300_b" else None)
    monkeypatch.setattr(sv, "node_ack", lambda sid, v, st, d="", client=None: calls.append((sid, v, st, d)) or {"ok": True})
    r = client.post("/v1/collector/ack", headers={"Authorization": "Bearer test-key"},
                    json={"instance_id": "turbina_sgt300_b", "version": 5, "status": "applied", "detail": "ok"})
    assert r.status_code == 200, r.text
    assert calls == [("SGT300B", 5, "applied", "ok")]
    r2 = client.post("/v1/collector/ack", headers={"Authorization": "Bearer test-key"},
                     json={"instance_id": "nope", "version": 1})
    assert r2.status_code == 404
