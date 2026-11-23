"""Tren de Machines en vivo → SVG del reporte PDF (mismo dibujo que Live)."""

from core.live_train_svg import build_live_train_svg
from core.train_svg import svg_to_train_drawing

LAYOUT = {"machines": [{"type": "turb", "bearings": 2, "bearing_type": "journal"},
                       {"type": "gen", "bearings": 2, "bearing_type": "journal"}],
          "aux": [{"name": "Bomba de lubricación", "k": "accel",
                   "probes": [{"n": "4YA", "k": "accel"}, {"n": "4YV", "k": "vel"}]}]}
LIVE = {"probes": [
    {"n": "1XD", "brg": 1, "k": "prox", "v": 56.0, "u": "µm pp", "sev": "ok", "alarm": 66, "ang": "135°R"},
    {"n": "1YD", "brg": 1, "k": "prox", "v": 57.0, "u": "µm pp", "sev": "ok", "alarm": 66, "ang": "135°L"},
    {"n": "4YV", "brg": 4, "k": "vel", "v": 5.1, "u": "mm/s rms", "sev": "alert", "alarm": 5.0},
    {"n": "4YA", "brg": 4, "k": "accel", "v": 4.8, "u": "g rms", "sev": "ok", "alarm": 9.0}]}


def test_svg_has_train_probes_and_aux():
    svg = build_live_train_svg(LAYOUT, LIVE)
    assert svg.startswith("<svg") and "Bomba de lubricación" in svg
    assert svg.index(">1YD<") < svg.index(">1XD<")          # Y antes que X
    assert "#f5a524" in svg                                  # 4YV en alerta


def test_svg_converts_to_pdf_drawing():
    d = svg_to_train_drawing(build_live_train_svg(LAYOUT, LIVE), 500)
    assert d is not None and round(d.width) == 500
