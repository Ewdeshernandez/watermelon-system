"""Motor común de Rotodinámica (core.rd_figures) — onda, espectro, órbita."""
import numpy as np

from core.analysis.model import Bearing, Channel
from core import rd_figures as F


def _ch(label, axis, phase=0.0, amp=25.0, rpm=3600.0, fs=2560.0, n=2048, unit="µm"):
    t = np.arange(n) / fs
    v = amp * np.sin(2 * np.pi * rpm / 60.0 * t + phase)
    return Channel(label=label, bearing=label[0], axis=axis, t=t, v=v, rpm=rpm,
                   spr=int(fs * 60 / rpm), unit=unit)


def test_waveform_rows_and_scale():
    y, x = _ch("1YD", "Y"), _ch("1XD", "X", phase=np.pi / 2)
    fig, rows, unit = F.waveform_figure([y, x])
    assert unit == "µm"
    assert [r[0] for r in rows] == ["1YD", "1XD"]
    assert abs(float(rows[0][1].split()[0]) - 50.0) < 1.0      # pp = 2·amp
    assert fig.layout.yaxis.range[0] == -fig.layout.yaxis.range[1]


def test_spectrum_1x_peak_pp():
    fig, rows = F.spectrum_figure([_ch("1YD", "Y")])
    assert fig is not None
    one_x = float(rows[0][3])
    assert abs(one_x - 50.0) < 3.0                               # desplazamiento en pp
    assert fig.layout.xaxis.range[1] == 60000                    # auto por tipo


def test_spectrum_empty_returns_none():
    ch = Channel(label="1YD", v=np.array([1.0, 2.0]))
    assert F.spectrum_figure([ch]) == (None, [])


def test_probe_angles_by_bearing():
    assert F.probe_angles("1")[:2] == (135.0, 135.0)
    assert F.probe_angles("5")[2] == "generador"
    assert F.probe_angles("4")[2] == "gearbox"


def test_orbit_row_and_figure():
    bb = Bearing(name="1XD/1YD", x=_ch("1XD", "X", phase=np.pi / 2), y=_ch("1YD", "Y"))
    ax, ay, _ = F.bearing_angles(bb)
    d = F.orbit_data(bb, "Direct", ax, ay)
    row = F.orbit_row(bb, d)
    assert row[0] == "1XD/1YD" and len(row) == 8
    fig = F.orbit_figure(bb, d, d["Rp"], ax, ay, "Direct", compact=True, title="t")
    assert fig.layout.title.text == "t"
