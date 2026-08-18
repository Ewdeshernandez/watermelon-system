"""
tests.test_balance_live_source
=============================

Helpers puros del adaptador Live Monitoring → balanceo (sin red):
agrupar planos desde el mapa de sensores, elegir la sonda por dirección y
parsear el 1X (mag+fase) desde las filas de live_readings.

Corre con pytest o directo:  python tests/test_balance_live_source.py
"""
from __future__ import annotations

from core.balance.live_source import (
    group_planes_from_sensors, pick_sensor_for_plane, parse_1x_rows,
    _section_from_label,
)


def test_section_from_label_descriptive():
    # Convención wizard + etiquetas descriptivas reales (SGT300B).
    assert _section_from_label("DE driven") == "Driven"
    assert _section_from_label("5YD DE generador") == "Driven"
    assert _section_from_label("NDE generador") == "Driven"
    assert _section_from_label("Compresor centrífugo") == "Driven"
    assert _section_from_label("1YD DE turbina") == "Driver"
    assert _section_from_label("Driver bearing 3") == "Driver"
    assert _section_from_label("4YD gearbox") == "Gearbox"


def _prox(plane, plane_label, direction):
    return {
        "plane": plane, "plane_label": plane_label, "direction": direction,
        "sensor_type": "proximity", "side": "L" if direction == "X" else "",
        "angle_deg": 90.0 if direction == "X" else 0.0,
    }


def _sensors_train():
    # turbina (driver) cojinetes 3 y 4 + compresor (driven) cojinete 6,
    # más una axial y un keyphasor que deben quedar EXCLUIDOS.
    return [
        _prox(3, "Driver bearing 3", "X"), _prox(3, "Driver bearing 3", "Y"),
        _prox(4, "Driver bearing 4", "X"), _prox(4, "Driver bearing 4", "Y"),
        _prox(6, "DE driven", "X"), _prox(6, "DE driven", "Y"),
        {"plane": 1, "plane_label": "GP Axial (thrust)", "direction": "axial",
         "sensor_type": "proximity"},
        {"plane": 0, "plane_label": "Coupling (keyphasor)", "direction": "",
         "sensor_type": "keyphasor"},
    ]


def test_group_planes_excludes_axial_and_keyphasor():
    planes = group_planes_from_sensors(_sensors_train())
    nums = [p["plane"] for p in planes]
    assert nums == [3, 4, 6]                       # ordenados, sin axial/keyphasor
    assert all(len(p["sensors"]) == 2 for p in planes)


def test_group_planes_sections():
    by = {p["plane"]: p for p in group_planes_from_sensors(_sensors_train())}
    assert by[3]["section"] == "Driver"
    assert by[4]["section"] == "Driver"
    assert by[6]["section"] == "Driven"           # "driven" en el plane_label


def test_labels_match_convention():
    by = {p["plane"]: p for p in group_planes_from_sensors(_sensors_train())}
    labels3 = {s["label"] for s in by[3]["sensors"]}
    assert labels3 == {"3X_D", "3Y_D"}             # formato sensor_label


def test_pick_sensor_for_plane_direction_and_fallback():
    by = {p["plane"]: p for p in group_planes_from_sensors(_sensors_train())}
    assert pick_sensor_for_plane(by[3], "X") == "3X_D"
    assert pick_sensor_for_plane(by[4], "X") == "4X_D"   # misma dirección en el otro plano
    assert pick_sensor_for_plane(by[3], "Y") == "3Y_D"
    # fallback: plano solo con Y, pido X -> devuelve la que haya
    only_y = {"sensors": [{"label": "9Y_D", "direction": "Y"}]}
    assert pick_sensor_for_plane(only_y, "X") == "9Y_D"


def test_parse_1x_rows():
    rows = [
        {"sensor_label": "3X_D", "metric": "1X_Ampl", "value": 0.35,
         "unit": "mil pp", "captured_at": "2026-08-15T10:00:00"},
        {"sensor_label": "3X_D", "metric": "1X_Phase", "value": 296.0},
        {"sensor_label": "3X_D", "metric": "Direct", "value": 0.50},  # se ignora
        {"sensor_label": "4X_D", "metric": "1X_Ampl", "value": 0.61},
        {"sensor_label": "4X_D", "metric": "1X_Phase", "value": 118.0},
        {"sensor_label": None, "metric": "1X_Ampl", "value": 9.9},     # sin label
    ]
    out = parse_1x_rows(rows)
    assert out["3X_D"]["mag"] == 0.35
    assert out["3X_D"]["phase"] == 296.0
    assert out["3X_D"]["unit"] == "mil pp"
    assert out["4X_D"]["mag"] == 0.61 and out["4X_D"]["phase"] == 118.0
    assert None not in out


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
