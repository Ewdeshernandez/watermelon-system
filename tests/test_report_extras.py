"""Extras del reporte: orden del tren (Y antes que X, auxiliares al final) y Δ 24 h."""

from datetime import datetime, timedelta, timezone

from core.report_extras import delta_24h, order_channels


def test_order_y_before_x_and_aux_last():
    chans = [{"sensor_label": s} for s in ["1X_D", "1Y_D", "4X_A", "2X_D", "4Y_V", "2Y_D", "4Y_A"]]
    aux = {"4YA": "Bomba de lubricación", "4YV": "Bomba de lubricación", "4XA": "Arrancador"}
    out = order_channels(chans, aux, ["Bomba de lubricación", "Arrancador"])
    assert [c["display"] for c in out] == ["1YD", "1XD", "2YD", "2XD", "4YA", "4YV", "4XA"]
    assert out[-1]["aux_group"] == "Arrancador"


def test_delta_24h_compares_same_hour_yesterday():
    now = datetime(2026, 11, 23, 8, tzinfo=timezone.utc)
    pts = [{"bucket": (now - timedelta(hours=h)).isoformat(), "avg_val": 50.0, "max_val": 51}
           for h in range(25, 0, -1)]
    pts.append({"bucket": now.isoformat(), "avg_val": 55.0, "max_val": 56})
    assert round(delta_24h(pts), 1) == 10.0
    assert delta_24h(pts[-3:]) is None          # sin dato de ayer
