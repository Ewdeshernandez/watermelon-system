"""Collector 1.2.0: mapa Modbus remoto con respaldo local."""

import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wc():
    # pymodbus no está en el entorno de tests: stub mínimo.
    if "pymodbus.client" not in sys.modules:
        pkg = types.ModuleType("pymodbus")
        cli = types.ModuleType("pymodbus.client")
        cli.ModbusTcpClient = object
        sys.modules.setdefault("pymodbus", pkg)
        sys.modules["pymodbus.client"] = cli
    spec = importlib.util.spec_from_file_location("wm_collector_t", ROOT / "collector" / "wm_collector.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wm_collector_t"] = mod  # dataclasses lo busca por nombre
    spec.loader.exec_module(mod)
    return mod


class _Resp:
    def __init__(self, code, body):
        self.status_code = code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _api(wc):
    return wc.ApiConfig(base_url="https://api.example", api_key="k", timeout_sec=1)


def _raw():
    return json.loads((ROOT / "data" / "modbus_maps" / "sgt300b.json").read_text(encoding="utf-8"))


def test_fetch_writes_cache_and_prev(wc, tmp_path, monkeypatch):
    raw = _raw()
    cache = tmp_path / "map.json"
    cache.write_text(json.dumps({"instance_id": raw["instance_id"], "registers": {}}), encoding="utf-8")
    new = dict(raw, _config={"updated_at": "2026-11-22T10:00:00"})
    monkeypatch.setattr(wc.requests, "get", lambda *a, **k: _Resp(200, new))
    got = wc.fetch_remote_map(_api(wc), raw["instance_id"], str(cache), logging.getLogger("t"))
    assert got is not None
    assert json.loads(cache.read_text(encoding="utf-8"))["registers"] == raw["registers"]
    assert (tmp_path / "map.json.prev").exists()


def test_fetch_failure_keeps_local(wc, tmp_path, monkeypatch):
    raw = _raw()
    cache = tmp_path / "map.json"
    cache.write_text(json.dumps(raw), encoding="utf-8")
    before = cache.read_text(encoding="utf-8")
    monkeypatch.setattr(wc.requests, "get", lambda *a, **k: _Resp(404, {"detail": "x"}))
    assert wc.fetch_remote_map(_api(wc), raw["instance_id"], str(cache), logging.getLogger("t")) is None
    assert cache.read_text(encoding="utf-8") == before


def test_fetch_rejects_empty_or_wrong_instance(wc, tmp_path, monkeypatch):
    raw = _raw()
    cache = tmp_path / "map.json"
    monkeypatch.setattr(wc.requests, "get", lambda *a, **k: _Resp(200, dict(raw, registers={})))
    assert wc.fetch_remote_map(_api(wc), raw["instance_id"], str(cache), logging.getLogger("t")) is None
    monkeypatch.setattr(wc.requests, "get", lambda *a, **k: _Resp(200, dict(raw, instance_id="other")))
    assert wc.fetch_remote_map(_api(wc), raw["instance_id"], str(cache), logging.getLogger("t")) is None
    assert not cache.exists()


def test_signature_ignores_config_meta(wc):
    raw = _raw()
    assert wc._map_signature(raw) == wc._map_signature(dict(raw, _config={"updated_at": "x"}))


def test_post_ack_payload(wc, monkeypatch):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, json=json)
        return _Resp(200, {"ok": True})
    monkeypatch.setattr(wc.requests, "post", fake_post)
    assert wc.post_ack(_api(wc), "turbina_sgt300_b", 5, "applied", "74 registers", logging.getLogger("t"))
    assert sent["url"].endswith("/v1/collector/ack")
    assert sent["json"] == {"instance_id": "turbina_sgt300_b", "version": 5, "status": "applied",
                            "detail": "74 registers"}
    assert wc.post_ack(_api(wc), "x", None, "applied", "", logging.getLogger("t")) is False


def test_invalid_remote_map_is_rejected(wc, tmp_path, monkeypatch):
    raw = _raw()
    acks = []
    monkeypatch.setattr(wc.requests, "get",
                        lambda *a, **k: _Resp(200, dict(raw, registers={}, _config={"version": 7})))
    monkeypatch.setattr(wc, "post_ack", lambda api, iid, v, st, d, log: acks.append((v, st)) or True)
    assert wc.fetch_remote_map(_api(wc), raw["instance_id"], str(tmp_path / "m.json"),
                               logging.getLogger("t")) is None
    assert acks == [(7, "rejected")]


def test_same_content_new_version_updates_cache_without_prev(wc, tmp_path, monkeypatch):
    raw = _raw()
    cache = tmp_path / "map.json"
    cache.write_text(json.dumps(dict(raw, _config={"version": 4})), encoding="utf-8")
    monkeypatch.setattr(wc.requests, "get", lambda *a, **k: _Resp(200, dict(raw, _config={"version": 5})))
    got = wc.fetch_remote_map(_api(wc), raw["instance_id"], str(cache), logging.getLogger("t"))
    assert got["_config"]["version"] == 5
    assert json.loads(cache.read_text(encoding="utf-8"))["_config"]["version"] == 5
    assert not (tmp_path / "map.json.prev").exists()
