"""
tests.test_loaders
==================

Validación de los parsers universales:
  - core.loaders.csi2140  : Emerson CSI 2140 CSV
  - core.loaders.adre408  : Bently Nevada ADRE 408 CSV
  - core.loaders.uff      : Universal File Format dataset 58

Estrategia: cada test construye un archivo sintético en memoria con
parámetros conocidos (RPM, fs, dur, A1X), pasa el texto al parser, y
verifica que LoadedSignal recuperado es coherente.
"""

from __future__ import annotations

import io
import math
import textwrap

import numpy as np
import pytest

from core.loaders.adre408 import parse_adre408
from core.loaders.base import LoadedSignal, loaded_to_signal
from core.loaders.csi2140 import parse_csi2140
from core.loaders.uff import parse_uff, parse_uff_all


# =============================================================
# CSI 2140
# =============================================================

CSI2140_TIME_TEMPLATE = textwrap.dedent("""\
    Route Name,Bombas Centrífugas
    Equipment,Pump 21A
    Point,Bearing 1 Vertical
    Direction,Vertical
    Date,2026-04-15
    Time,10:23:00
    Sample Rate,5120 Hz
    RPM,1780
    Number of Lines,800
    Sensitivity,100 mV/g
    Fmax,400 Hz
    Window,Hanning
    Averaging,4 Linear
    Units,g pk

    [DATA]
    Time(s),Acceleration(g)
""")


def _build_csi2140_time_csv(fs=5120.0, rpm=1780.0, duration=0.1, amp_g=0.5):
    """Construye un CSV CSI 2140 sintético en modo waveform."""
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    f1 = rpm / 60.0
    y = amp_g * np.sin(2 * np.pi * f1 * t)
    rows = "\n".join(f"{ti:.6f},{yi:.6f}" for ti, yi in zip(t, y))
    return CSI2140_TIME_TEMPLATE + rows


def test_csi2140_parses_time_waveform_basic():
    csv = _build_csi2140_time_csv()
    sig = parse_csi2140(csv, file_name="test.csv")
    assert isinstance(sig, LoadedSignal)
    sig.validate()
    assert sig.domain == "time"
    assert sig.vendor == "csi2140"
    assert sig.x.size == sig.time.size
    assert sig.fs == pytest.approx(5120.0, rel=1e-3)
    assert sig.rpm == pytest.approx(1780.0, rel=1e-6)


def test_csi2140_recovers_metadata():
    csv = _build_csi2140_time_csv()
    sig = parse_csi2140(csv, file_name="test.csv")
    assert sig.metadata.get("Equipment") == "Pump 21A"
    assert sig.metadata.get("Point") == "Bearing 1 Vertical"
    assert sig.metadata.get("Window") == "Hanning"


def test_csi2140_units_extracted_from_header():
    csv = _build_csi2140_time_csv()
    sig = parse_csi2140(csv, file_name="test.csv")
    assert sig.units == "g"


def test_csi2140_amplitude_recovered():
    csv = _build_csi2140_time_csv(amp_g=0.5)
    sig = parse_csi2140(csv, file_name="test.csv")
    measured_peak = float(np.max(np.abs(sig.x)))
    assert measured_peak == pytest.approx(0.5, rel=0.05)


def test_csi2140_spectrum_mode():
    spectrum_csv = textwrap.dedent("""\
        Equipment,Compressor 1
        RPM,3600
        Sample Rate,2560
        Units,mm/s pk

        Frequency(Hz),Amplitude(mm/s)
        0.0,0.001
        60.0,2.500
        120.0,0.800
        180.0,0.300
    """)
    sig = parse_csi2140(spectrum_csv, file_name="spectrum.csv")
    assert sig.domain == "spectrum"
    assert sig.x.size == 4
    assert sig.metadata.get("Equipment") == "Compressor 1"
    assert "axis_freq_hz" in sig.metadata


def test_csi2140_invalid_input_raises():
    with pytest.raises(ValueError):
        parse_csi2140("garbage with no header", file_name="x.csv")


def test_csi2140_empty_input_raises():
    with pytest.raises(ValueError):
        parse_csi2140("", file_name="x.csv")


def test_csi2140_handles_semicolon_separator():
    """Algunos exports europeos usan ';' como separador."""
    csv = textwrap.dedent("""\
        Equipment;Pump 21A
        Sample Rate;1000

        Time(s);Acceleration(g)
        0.0;0.1
        0.001;0.2
        0.002;-0.1
    """)
    sig = parse_csi2140(csv, file_name="eu.csv")
    assert sig.x.size == 3


# =============================================================
# ADRE 408
# =============================================================

ADRE408_TIME_TEMPLATE = textwrap.dedent("""\
    "Header"
    "Machine","Compressor 21B"
    "Point","Bearing A Vertical"
    "Probe","8mm proximity"
    "Date","2026-04-15 10:23:00"
    "Sample Rate","2560"
    "RPM","3600"
    "Units","mils pp"

    "Time","Amplitude"
""")


def _build_adre408_time_csv(fs=2560.0, rpm=3600.0, duration=0.5, amp=10.0):
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    f1 = rpm / 60.0
    y = amp * np.sin(2 * np.pi * f1 * t)
    rows = "\n".join(f"{ti:.6f},{yi:.6f}" for ti, yi in zip(t, y))
    return ADRE408_TIME_TEMPLATE + rows


def test_adre408_parses_time_waveform():
    csv = _build_adre408_time_csv()
    sig = parse_adre408(csv, file_name="adre.csv")
    sig.validate()
    assert sig.vendor == "adre408"
    assert sig.domain == "time"
    assert sig.fs == pytest.approx(2560.0, rel=1e-3)
    assert sig.rpm == pytest.approx(3600.0, rel=1e-6)


def test_adre408_recovers_metadata():
    csv = _build_adre408_time_csv()
    sig = parse_adre408(csv, file_name="adre.csv")
    assert sig.metadata.get("Machine") == "Compressor 21B"
    assert sig.metadata.get("Point") == "Bearing A Vertical"
    assert sig.units == "mils pp"


def test_adre408_amplitude_recovered():
    csv = _build_adre408_time_csv(amp=12.5)
    sig = parse_adre408(csv, file_name="adre.csv")
    measured_peak = float(np.max(np.abs(sig.x)))
    assert measured_peak == pytest.approx(12.5, rel=0.05)


def test_adre408_invalid_raises():
    with pytest.raises(ValueError):
        parse_adre408("not adre format", file_name="x.csv")


# =============================================================
# UFF (Universal File Format)
# =============================================================

def _build_uff_time_block(fs=1024.0, duration=0.5, amp=1.0, rpm_f=60.0):
    """Construye dataset 58 ASCII con time response (function_type=2, even spacing)."""
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    y = amp * np.sin(2 * np.pi * rpm_f * t)

    abs_inc = 1.0 / fs
    abs_min = 0.0

    # ID lines (5 records of 80 chars)
    id1 = "Synthetic UFF dataset 58 — Watermelon test"
    id2 = "Pump bearing vertical"
    id3 = "2026-04-15 10:23:00"
    id4 = ""
    id5 = ""

    # Record 6: function_type=2 (time response), id, version, load, name, node
    record6 = "         2         0         0         0         0         0"

    # Record 7: ord_data_type, n_pts, abs_spacing, abs_min, abs_inc, z
    record7 = f"         2{n:>10d}         1{abs_min:13.5e}{abs_inc:13.5e}{0.0:13.5e}"

    # Records 8-11: axis label specs (simplificados)
    record8 = "         0         1         0         0 0.00000e+00 0.00000e+00"
    record9 = "Time              s              "
    record10 = "         0         1         0         0 0.00000e+00 0.00000e+00"
    record11 = "Acceleration      g              "

    # Data: even spacing → sólo ordenadas, 4 valores por línea
    chunks = []
    for i in range(0, n, 4):
        chunk = y[i:i + 4]
        chunks.append("".join(f"{v:13.5e}" for v in chunk))
    data_block = "\n".join(chunks)

    block_lines = [
        "    -1",
        "    58",
        id1, id2, id3, id4, id5,
        record6, record7, record8, record9, record10, record11,
        data_block,
        "    -1",
    ]
    return "\n".join(block_lines)


def test_uff_parses_dataset58_time_response():
    text = _build_uff_time_block(fs=1024.0, duration=0.5, amp=1.0, rpm_f=60.0)
    sig = parse_uff(text, file_name="synthetic.uff")
    sig.validate()
    assert sig.vendor == "uff"
    assert sig.domain == "time"
    assert sig.x.size == 512
    # fs derivada de abs_inc
    assert sig.fs == pytest.approx(1024.0, rel=1e-3)


def test_uff_recovers_amplitude():
    text = _build_uff_time_block(fs=1024.0, duration=0.5, amp=2.5, rpm_f=60.0)
    sig = parse_uff(text, file_name="synthetic.uff")
    measured_peak = float(np.max(np.abs(sig.x)))
    assert measured_peak == pytest.approx(2.5, rel=0.05)


def test_uff_recovers_id_lines():
    text = _build_uff_time_block()
    sig = parse_uff(text, file_name="synthetic.uff")
    assert "Synthetic UFF" in sig.metadata.get("id_line_1", "")


def test_uff_no_dataset58_raises():
    """Archivo sin dataset 58 debe fallar limpiamente."""
    text = "    -1\n    15\nGeometría\n    -1"
    with pytest.raises(ValueError):
        parse_uff(text, file_name="x.uff")


def test_uff_parse_all_returns_list():
    """Con un dataset 58 → lista de 1 elemento; con varios → más."""
    text = _build_uff_time_block(amp=1.0)
    out = parse_uff_all(text, file_name="x.uff")
    assert isinstance(out, list)
    assert len(out) >= 1


# =============================================================
# Bridge a Signal legacy
# =============================================================

def test_loaded_to_signal_bridge_csi2140():
    """LoadedSignal → Signal Watermelon debería tener .x, .time, .metadata."""
    csv = _build_csi2140_time_csv()
    loaded = parse_csi2140(csv, file_name="b.csv")
    sig = loaded_to_signal(loaded)
    assert hasattr(sig, "x")
    assert hasattr(sig, "time")
    assert hasattr(sig, "metadata")
    assert sig.x.size == loaded.x.size
    # La metadata debe llevar fs/rpm como esperan los downstream consumers
    assert "rpm" in sig.metadata or "RPM" in sig.metadata
    assert "fs" in sig.metadata or "Sample Rate" in sig.metadata


def test_loaded_to_signal_bridge_uff():
    text = _build_uff_time_block()
    loaded = parse_uff(text, file_name="b.uff")
    sig = loaded_to_signal(loaded)
    assert sig.x.size == loaded.x.size


# =============================================================
# Tipos de input: path | str | bytes | file-like
# =============================================================

def test_csi2140_accepts_bytes_input():
    csv = _build_csi2140_time_csv()
    sig = parse_csi2140(csv.encode("utf-8"), file_name="b.csv")
    assert sig.x.size > 0


def test_csi2140_accepts_filelike_input():
    csv = _build_csi2140_time_csv()
    fl = io.StringIO(csv)
    sig = parse_csi2140(fl, file_name="b.csv")
    assert sig.x.size > 0


def test_csi2140_accepts_bytesio_with_bom():
    csv = "﻿" + _build_csi2140_time_csv()
    fl = io.BytesIO(csv.encode("utf-8-sig"))
    sig = parse_csi2140(fl, file_name="bom.csv")
    assert sig.x.size > 0
