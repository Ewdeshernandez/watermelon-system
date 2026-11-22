"""Versionado de la config de máquina: lógica pura."""

import json
from pathlib import Path

from core.remote_monitoring.config import setup_to_dict
from core.remote_monitoring.modbus_binding import setup_from_modbus_map
from core.remote_monitoring.setup_versions import (
    diff_is_empty,
    diff_specs,
    next_version,
    publish_conflict,
    published_row,
)

ROOT = Path(__file__).resolve().parents[1]


def _spec():
    raw = json.loads((ROOT / "data" / "modbus_maps" / "sgt300b.json").read_text(encoding="utf-8"))
    return setup_to_dict(setup_from_modbus_map(raw, name="SGT300B"))


def test_next_version_and_published():
    rows = [{"version": 1, "status": "superseded"}, {"version": 2, "status": "published"},
            {"version": 3, "status": "draft"}]
    assert next_version(rows) == 4
    assert next_version([]) == 1
    assert published_row(rows)["version"] == 2


def test_publish_requires_base_equal_current():
    rows = [{"version": 2, "status": "published"}]
    ok = {"version": 3, "status": "draft", "base_version": 2}
    stale = {"version": 4, "status": "draft", "base_version": 1}
    assert publish_conflict(rows, ok) is None
    assert "publicada es v2" in publish_conflict(rows, stale)
    assert publish_conflict(rows, {"version": 2, "status": "published"})
    # Primera publicación: sin publicada previa → libre.
    assert publish_conflict([], {"version": 1, "status": "draft", "base_version": None}) is None


def test_diff_detects_sensor_changes():
    a = _spec()
    b = json.loads(json.dumps(a))
    assert diff_is_empty(diff_specs(a, b))
    ch = next(c for c in b["channels"] if c["point_label"] == "1X_D")
    ch["angle_deg"] = 60.0
    b["channels"].append(dict(ch, point_label="LB0011"))
    b["channels"] = [c for c in b["channels"] if c["point_label"] != "2Y_D"] if any(
        c["point_label"] == "2Y_D" for c in b["channels"]) else b["channels"]
    b["source"] = "ni"
    d = diff_specs(a, b)
    assert d["added"] == ["LB0011"]
    assert any(s.startswith("1X_D:") and "angle_deg" in s for s in d["changed"])
    assert "source 'modbus'→'ni'" in d["source"]
