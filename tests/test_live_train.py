"""Hero de Live desde el tren de Machines: últimas lecturas → payload en vivo."""

import json
from pathlib import Path

from core.live_train import live_payload
from core.remote_monitoring.modbus_binding import setup_from_modbus_map

ROOT = Path(__file__).resolve().parents[1]


def test_payload_maps_readings_and_severity():
    raw = json.loads((ROOT / "data" / "modbus_maps" / "sgt300b.json").read_text(encoding="utf-8"))
    s = setup_from_modbus_map(raw, name="SGT300B")
    ch = next(c for c in s.channels if c.point_label == "1Y_D")
    ch.alarm, ch.danger, ch.angle_deg, ch.side = 66, 89, 135, "L"
    latest = [
        {"sensor_label": "1Y_D", "metric": "Direct", "value": 70.0, "variable": "1YD Turbina DE"},
        {"sensor_label": "1Y_D", "metric": "1X_Ampl", "value": 49.5},
        {"sensor_label": None, "metric": "Direct", "value": 14072, "variable": "Velocidad Turbina"},
    ]
    p = live_payload(s, latest)
    y = next(x for x in p["probes"] if x["n"] == "1YD")
    assert (y["v"], y["sev"], y["ang"], y["brg"], y["k"], y["x1"]) == (70.0, "alert", "135°L", 1, "prox", 49.5)
    assert p["speeds"][0] == {"label": "Velocidad Turbina", "v": 14072, "u": "rpm"}
    assert next(x for x in p["probes"] if x["n"] == "2YD")["sev"] == "off"   # sin lectura
