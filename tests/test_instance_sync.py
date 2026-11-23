"""Puente Machines ↔ instances: enriquecer y sincronizar sin borrar."""

import json
from pathlib import Path

from core.remote_monitoring.instance_sync import enrich_setup_from_sensors, sync_sensors_from_setup
from core.remote_monitoring.modbus_binding import modbus_map_from_setup, setup_from_modbus_map
from core.remote_monitoring.config import ChannelRow
from core.sensor_map import new_sensor, sensor_label

ROOT = Path(__file__).resolve().parents[1]


def _lib():
    return [
        new_sensor(plane=1, plane_label="1YD DE turbina", side="L", angle_deg=45, direction="Y",
                   sensor_type="proximity", unit_native="µm pp", alarm=66, danger=89),
        new_sensor(plane=1, plane_label="1XD DE turbina", side="R", angle_deg=45, direction="X",
                   sensor_type="proximity", unit_native="µm pp", alarm=66, danger=89),
        new_sensor(plane=9, plane_label="9YD old", side="L", angle_deg=45, direction="Y",
                   sensor_type="proximity", alarm=1, danger=2),
    ]


def _setup():
    raw = json.loads((ROOT / "data" / "modbus_maps" / "sgt300b.json").read_text(encoding="utf-8"))
    return raw, setup_from_modbus_map(raw, name="SGT300B")


def test_enrich_fills_only_empty_and_keeps_modbus():
    raw, setup = _setup()
    new, rep = enrich_setup_from_sensors(setup, _lib())
    ch = next(c for c in new.channels if c.point_label == "1Y_D")
    assert (ch.alarm, ch.danger, ch.angle_deg, ch.side) == (66, 89, 45, "L")
    assert ch.location == "Turbina DE"          # no se pisa lo que ya tenía
    assert "1Y_D" in rep["enriched"] and "9Y_D" in rep["only_in_library"]
    assert modbus_map_from_setup(new) == raw     # el collector no cambia


def test_sync_never_deletes_and_adds_new():
    _, setup = _setup()
    setup, _ = enrich_setup_from_sensors(setup, _lib())
    setup.channels.append(ChannelRow(bnc_port=99, point_label="7X_D", plane=7, sensor_type="proximity",
                                     direction="X", angle_deg=45, side="R", alarm=50, danger=80))
    next(c for c in setup.channels if c.point_label == "1X_D").alarm = 70
    out, rep = sync_sensors_from_setup(_lib(), setup)
    labels = [sensor_label(s) for s in out]
    assert "9Y_D" in labels                      # nunca se borra
    assert "7X_D" in rep["added"] and "7X_D" in labels
    assert next(s for s in out if sensor_label(s) == "1X_D")["alarm"] == 70
    assert "1X_D" in rep["updated"]
    assert not rep["label_mismatch"] or all("→" in m for m in rep["label_mismatch"])


def test_header_changes_only_non_empty():
    from types import SimpleNamespace
    from core.remote_monitoring.instance_sync import header_changes
    _, setup = _setup()
    setup.machine.client, setup.machine.location, setup.machine.tag = "Cliente Ejemplo SA", "Planta Norte", ""
    inst = SimpleNamespace(client="CLIENTE EJEMPLO", site="PLANTA NORTE", tag="TG-B",
                           nominal_rpm=float(setup.machine.rpm_nominal))
    hc = header_changes(inst, setup)
    assert hc == {"client": ("CLIENTE EJEMPLO", "Cliente Ejemplo SA"), "site": ("PLANTA NORTE", "Planta Norte")}


def test_header_changes_norm_and_rpm():
    from types import SimpleNamespace
    from core.remote_monitoring.instance_sync import header_changes
    _, setup = _setup()
    setup.machine.iso_norm, setup.machine.iso_class, setup.machine.rpm_nominal = "ISO_20816_3", "2", 14000
    inst = SimpleNamespace(client="", site="", tag="", iso_norm_code="", iso_norm_class="", nominal_rpm=3600.0)
    hc = header_changes(inst, setup)
    assert hc["iso_norm_code"] == ("", "ISO_20816_3") and hc["iso_norm_class"] == ("", "2")
    assert hc["nominal_rpm"] == ("3600", "14000")
