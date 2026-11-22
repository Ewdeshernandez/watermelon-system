"""Origen Modbus de la máquina: mapa del collector ↔ AcqSetup sin pérdida."""

import json
from pathlib import Path

import pytest

from core.remote_monitoring.config import setup_from_dict, setup_to_dict
from core.remote_monitoring.modbus_binding import (
    diff_modbus_maps,
    modbus_map_from_setup,
    setup_from_modbus_map,
)

MAPS_DIR = Path(__file__).resolve().parents[1] / "data" / "modbus_maps"
MAPS = sorted(p.name for p in MAPS_DIR.glob("*.json"))


@pytest.mark.parametrize("fname", MAPS)
def test_roundtrip_lossless(fname):
    raw = json.loads((MAPS_DIR / fname).read_text(encoding="utf-8"))
    setup = setup_from_modbus_map(raw)
    assert modbus_map_from_setup(setup) == raw


@pytest.mark.parametrize("fname", MAPS)
def test_roundtrip_through_cloud_dict(fname):
    """Pasa por el jsonb de rm_setups (setup_to_dict/setup_from_dict)."""
    raw = json.loads((MAPS_DIR / fname).read_text(encoding="utf-8"))
    setup = setup_from_dict(json.loads(json.dumps(setup_to_dict(setup_from_modbus_map(raw)))))
    assert setup.source == "modbus"
    assert setup.instance_id == raw["instance_id"]
    assert modbus_map_from_setup(setup) == raw


def test_channels_grouped_by_sensor():
    raw = json.loads((MAPS_DIR / "sgt300b.json").read_text(encoding="utf-8"))
    setup = setup_from_modbus_map(raw)
    labels = [c.point_label for c in setup.channels]
    assert len(labels) == len(set(labels))
    ch = next(c for c in setup.channels if c.point_label == "1Y_D")
    assert ch.sensor_type == "proximity"
    assert ch.direction == "Y"
    assert ch.variable == "1YD Turbina DE"
    assert ch.unit_native == "µm pp"
    assert ch.modbus["Direct"]["register"] == 6005
    assert {"Direct", "Gap", "1X_Ampl", "1X_Phase"} <= set(ch.modbus)
    speeds = [c for c in setup.channels if c.sensor_type == "keyphasor"]
    assert [c.variable for c in speeds] == ["Velocidad Turbina", "Velocidad Generador"]


def test_editing_channel_changes_generated_map():
    raw = json.loads((MAPS_DIR / "sgt300b.json").read_text(encoding="utf-8"))
    setup = setup_from_modbus_map(raw)
    ch = next(c for c in setup.channels if c.point_label == "1Y_D")
    ch.location = "Turbine DE"
    new = modbus_map_from_setup(setup)
    assert new["registers"]["6005"]["location"] == "Turbine DE"
    d = diff_modbus_maps(raw, new)
    assert d["added"] == [] and d["removed"] == []
    assert any(c.startswith("6005:") for c in d["changed"])


def test_velocity_type_maps_both_ways():
    raw = json.loads((MAPS_DIR / "sgt300b.json").read_text(encoding="utf-8"))
    setup = setup_from_modbus_map(raw)
    vel = [c for c in setup.channels if c.sensor_type == "velometer"]
    assert vel, "SGT300B tiene sensores de velocidad"
    out = modbus_map_from_setup(setup)
    assert any(r.get("sensor_type") == "velocity" for r in out["registers"].values())
