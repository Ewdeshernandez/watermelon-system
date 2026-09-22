"""
tests/test_dynamic_raw.py — pipeline de onda cruda dinámica (System1 → web).

Valida el contrato de intercambio y la reconstrucción SIN nube ni DB real:
serializa una captura sintética (desbalance 1X + keyphasor), la re-parsea y
verifica forma de onda, espectro (pico en 1X), keyphasor y órbita.
"""
import numpy as np
import pytest

from core.dynamic_raw import (
    build_capture_csv, parse_capture_csv, spectrum, top_peaks,
    keyphasor_edges, rpm_from_keyphasor, compute_orbit_from_capture,
    parse_system1_waveform_csv, synth_keyphasor,
    CH_X, CH_Y, CH_KPH,
)

# CSV real que exporta System1 (Export to CSV de la onda), verificado en Parex.
_S1_WF = """Machine Name,SGT300B
Point Name,1XD TURBINA DE
Wf Amp,51.063
Number of Revs,16
X-Axis Unit,ms
Y-Axis Unit,µm
Sample Speed, 14049 rpm
Sample Status,Valid
Timestamp,9/21/2026 8:29:22 PM
Variable,Disp Wf(128X/16revs).KPH TURBINA
X-Axis Value,Y-Axis Value
0,4.581
0.033,5.743
0.067,6.809
0.1,7.681
0.133,8.553
0.167,9.958
0.2,11.315
"""


def test_parse_system1_waveform_csv():
    r = parse_system1_waveform_csv(_S1_WF)
    assert r["point"] == "1XD TURBINA DE"
    assert r["revs"] == 16
    assert abs(r["rpm"] - 14049) < 1
    assert r["y_unit"] == "µm"
    assert r["t_s"].size == 7 and r["values"].size == 7
    # X-Axis en ms → segundos
    assert abs(r["t_s"][1] - 0.033 / 1000.0) < 1e-9
    assert abs(r["values"][0] - 4.581) < 1e-6
    # fs desde dt (0.033 ms) ~ 30 kHz
    assert r["fs_hz"] and 25000 < r["fs_hz"] < 35000


def test_synth_keyphasor():
    kph = synth_keyphasor(2048, 128)
    edges = keyphasor_edges(kph)
    assert edges.size == 15  # 16 vueltas → 15 flancos de subida detectables


def test_agent_vendored_s1_csv_matches_core():
    """El parser vendored del agente (tools/system1_agent/s1_csv.py) debe dar
    lo mismo que core y mapear sensor→cojinete/eje."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "tools" / "system1_agent" / "s1_csv.py"
    spec = importlib.util.spec_from_file_location("s1_csv", p)
    s1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s1)
    r_core = parse_system1_waveform_csv(_S1_WF)
    r_vend = s1.parse_system1_waveform_csv(_S1_WF)
    assert r_vend["rpm"] == r_core["rpm"] and r_vend["revs"] == r_core["revs"]
    np.testing.assert_allclose(r_vend["values"], r_core["values"])
    assert s1.sensor_bearing_axis("1xd") == ("1", "x")
    assert s1.sensor_bearing_axis("6yd") == ("6", "y")
    assert s1.sensor_bearing_axis("state") == (None, None)


def _synthetic_capture(rpm=3000.0, spr=256, revs=32, amp=50.0):
    """Órbita circular (desbalance puro) + keyphasor 1 pulso/vuelta."""
    f1 = rpm / 60.0            # 50 Hz
    fs = spr * f1              # 12800 Hz  → spr entero
    n = spr * revs
    t = np.arange(n) / fs
    x = amp * np.sin(2 * np.pi * f1 * t)
    y = amp * np.cos(2 * np.pi * f1 * t)   # 90° → círculo
    kph = np.zeros(n)
    for k in range(revs):
        kph[k * spr: k * spr + 4] = 1.0
    meta = {
        "asset": "SGT300B", "point": "CDE",
        "captured_at": "2026-09-21T12:00:00Z",
        "rpm": rpm, "fs_hz": fs, "samples_per_rev": spr,
        "units": f"{CH_X}:um,{CH_Y}:um,{CH_KPH}:V",
    }
    channels = {CH_X: x, CH_Y: y, CH_KPH: kph}
    return meta, t, channels, fs, f1


def test_csv_roundtrip():
    meta, t, ch, fs, f1 = _synthetic_capture()
    csv = build_capture_csv(meta, t, ch)
    cap = parse_capture_csv(csv)
    assert cap.asset == "SGT300B"
    assert cap.point == "CDE"
    assert cap.samples_per_rev == 256
    assert abs((cap.rpm or 0) - 3000.0) < 1e-6
    assert cap.has(CH_X) and cap.has(CH_Y) and cap.has(CH_KPH)
    np.testing.assert_allclose(cap.get(CH_X)[:50], ch[CH_X][:50], atol=1e-3)
    assert cap.unit_of(CH_X) == "um"


def test_spectrum_peak_at_1x():
    meta, t, ch, fs, f1 = _synthetic_capture(amp=50.0)
    freqs, amp = spectrum(ch[CH_X], fs)
    i = int(np.argmax(amp))
    assert abs(freqs[i] - f1) < 1.0           # pico en 1X (50 Hz)
    assert abs(amp[i] - 50.0) < 3.0           # amplitud pico ≈ A
    peaks = top_peaks(freqs, amp, rpm=3000.0)
    assert peaks and abs(peaks[0]["order"] - 1.0) < 0.05


def test_keyphasor_and_rpm():
    meta, t, ch, fs, f1 = _synthetic_capture(revs=32)
    edges = keyphasor_edges(ch[CH_KPH])
    assert 30 <= edges.size <= 32            # ~1 marca por vuelta
    rpm = rpm_from_keyphasor(t, ch[CH_KPH])
    assert rpm is not None and abs(rpm - 3000.0) < 20.0


def test_orbit_reconstructs():
    meta, t, ch, fs, f1 = _synthetic_capture()
    cap = parse_capture_csv(build_capture_csv(meta, t, ch))
    res = compute_orbit_from_capture(cap)
    assert res is not None                     # órbita reconstruida sin error


def test_fs_inferred_when_missing():
    meta, t, ch, fs, f1 = _synthetic_capture()
    meta.pop("fs_hz")
    cap = parse_capture_csv(build_capture_csv(meta, t, ch))
    assert cap.fs_hz is not None and abs(cap.fs_hz - fs) / fs < 0.01
