"""Puente de config.html (campo + web): init, build, validación, versiones."""

import json
from pathlib import Path

from core.remote_monitoring import config_bridge as cb
from core.remote_monitoring.config import setup_from_dict
from core.remote_monitoring.modbus_binding import modbus_map_from_setup, setup_from_modbus_map

ROOT = Path(__file__).resolve().parents[1]


def _sgt():
    raw = json.loads((ROOT / "data" / "modbus_maps" / "sgt300b.json").read_text(encoding="utf-8"))
    return raw, setup_from_modbus_map(raw, name="SGT300B")


def test_init_blank_is_locked_and_built_is_open():
    m = cb.init_message(cb.blank_setup("X"), host="field", library=[], versions=[])
    assert m["unlocked"] == 0 and m["built"] is False and m["step"] == 0
    assert set(m["defaults"]["types"]) == {"proximity", "velometer", "accelerometer", "keyphasor"}
    _, s = _sgt()
    m2 = cb.init_message(s, host="web", library=[], versions=[])
    assert m2["unlocked"] == 6 and m2["built"] is True
    assert m2["setup"]["measure_band"]["prox"] == [1.0, 1000.0]


def test_page_roundtrip_keeps_modbus_map():
    raw, s = _sgt()
    d = json.loads(json.dumps(cb.normalize(s)))   # lo que viaja a la página y vuelve
    assert modbus_map_from_setup(setup_from_dict(d)) == raw


def test_build_and_findings():
    _, s = _sgt()
    spec = {"machines": [{"type": "turb", "name": "T", "bearings": 2, "bearing_type": "journal"}],
            "sensors": [{"n": "1XD", "k": "prox", "ang": "45°R"}, {"n": "9XD", "k": "prox", "ang": "45°R"}]}
    out = cb.handle_build(cb.normalize(s), spec)
    assert out["wm"] == "setup" and "9X_D" in out["report"]["suggested"]   # máquina Modbus: solo sugiere
    fs = cb.findings(out["setup"])
    assert all({"level", "code", "message"} <= set(f) for f in fs)


def test_live_variable_warning_on_ni_switch():
    _, s = _sgt()
    d = cb.normalize(s)
    d["source"] = "ni"
    d["channels"][2]["variable"] = ""
    codes = [f["code"] for f in cb.findings(d)]
    assert "live_variable" in codes


def test_versions_view():
    v = cb.versions_view([{"version": 3, "status": "draft", "origin": "field", "author": "tech",
                           "created_at": "2026-11-22T10:00:00", "base_version": 2}])
    assert v[0] == {"v": 3, "st": "draft", "org": "field", "by": "tech", "when": "2026-11-22 10:00",
                    "note": "", "node": "", "base": 2}
