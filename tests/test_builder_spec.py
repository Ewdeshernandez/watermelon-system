"""Constructor de tren → config de máquina (spec_to_channels + merge)."""

import json
from pathlib import Path

from core.remote_monitoring.builder_spec import (
    canonical_label,
    display_label,
    merge_builder_spec,
    spec_to_channels,
)
from core.remote_monitoring.modbus_binding import modbus_map_from_setup, setup_from_modbus_map

ROOT = Path(__file__).resolve().parents[1]

SPEC = {
    "machines": [{"type": "turb", "name": "Turbine", "bearings": 2, "bearing_type": "journal"},
                 {"type": "gen", "name": "Generator", "bearings": 2, "bearing_type": "journal"}],
    "keyphasors": 0, "aux": [],
    "sensors": [
        {"comp": "Turbine", "n": "KPH", "k": "kph", "ang": "1/rev"},
        {"comp": "Turbine", "n": "1XD", "k": "prox", "ang": "45°R"},
        {"comp": "Turbine", "n": "1YD", "k": "prox", "ang": "45°L"},
        {"comp": "Turbine", "n": "1HD", "k": "accel", "ang": "H carcasa"},
        {"comp": "Generator", "n": "3XD", "k": "prox", "ang": "45°R"},
        {"comp": "Generator", "n": "3YD", "k": "prox", "ang": "45°L"},
        {"comp": "Auxiliar", "n": "LB0011", "k": "accel", "ang": "aux"},
    ],
}


def test_labels():
    assert canonical_label("1XD") == "1X_D"
    assert canonical_label("2ZDa") == "2Z_Da"
    assert canonical_label("KPH") == "KPH"
    assert canonical_label("1A") == "1A"
    assert display_label("1Y_D") == "1YD"


def test_spec_to_channels_order_and_angles():
    chs = spec_to_channels(SPEC)
    assert [c.point_label for c in chs] == ["KPH", "1Y_D", "1X_D", "1H_D", "3Y_D", "3X_D", "LB0011"]
    assert [c.bnc_port for c in chs] == list(range(1, 8))
    x = next(c for c in chs if c.point_label == "1X_D")
    assert (x.angle_deg, x.side, x.sensor_type, x.plane, x.direction) == (45.0, "R", "proximity", 1, "X")
    assert next(c for c in chs if c.point_label == "1H_D").sensor_type == "accelerometer"


def test_merge_keeps_live_contract():
    raw = json.loads((ROOT / "data" / "modbus_maps" / "sgt300b.json").read_text(encoding="utf-8"))
    setup = setup_from_modbus_map(raw, name="SGT300B")
    new, rep = merge_builder_spec(setup, SPEC, add_new=True)
    assert "LB0011" in rep["added"] and "1X_D" in rep["matched"]
    # Lo que alimenta Live queda intacto: el mapa Modbus generado no cambia.
    assert modbus_map_from_setup(new) == raw
    # Nada se borra en silencio.
    assert len(new.channels) == len(setup.channels) + len(rep["added"])
    assert new.layout == SPEC


def test_modbus_machine_only_suggests_new_probes():
    raw = json.loads((ROOT / "data" / "modbus_maps" / "sgt300b.json").read_text(encoding="utf-8"))
    setup = setup_from_modbus_map(raw, name="SGT300B")
    spec = dict(SPEC, sensors=SPEC["sensors"] + [{"comp": "Caja", "n": "3XD", "k": "prox", "ang": "45°R"}])
    new, rep = merge_builder_spec(setup, spec)
    assert rep["added"] == []
    assert "3X_D" in rep["suggested"] and "LB0011" in rep["suggested"]
    assert len(new.channels) == len(setup.channels)
    _, rep2 = merge_builder_spec(setup, spec, add_new=True)
    assert "3X_D" in rep2["added"]
