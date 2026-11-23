from datetime import datetime, timedelta, timezone

import core.demo_access as D


def _set(monkeypatch, alias=None, access=None):
    vals = {D.ALIAS_KEY: alias or {}, D.ACCESS_KEY: access or {}}
    monkeypatch.setattr(D, "_setting", lambda k, default: vals.get(k, default))


def test_alias_maps_only_demo_instances(monkeypatch):
    _set(monkeypatch, alias={"demo_x": "real_y"})
    assert D.source_for("demo_x") == "real_y"
    assert D.source_for("real_y") == "real_y"
    assert D.is_demo("demo_x") and not D.is_demo("real_y")
    assert D.source_for("") == ""


def test_demo_status(monkeypatch):
    now = datetime(2026, 11, 23, tzinfo=timezone.utc)
    _set(monkeypatch, access={
        "a@x.com": {"until": (now + timedelta(days=3)).isoformat()},
        "b@x.com": {"until": (now - timedelta(days=1)).isoformat()},
        "c@x.com": {"until": "no-es-fecha"}})
    assert D.demo_status("A@x.com", now) == "activo"
    assert D.demo_status("b@x.com", now) == "vencido"
    assert D.demo_status("c@x.com", now) == "vencido"
    assert D.demo_status("otro@x.com", now) is None
