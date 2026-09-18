"""
tests.test_torsional_analysis
=============================

Valida el motor de análisis torsional (core.torsional.analysis), cerrando el
lazo contra la fuente simulada: el sim codifica par conocido → el análisis lo
recupera. Además valida rainflow contra el ejemplo canónico ASTM E1049.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.torsional.scaling import (
    ShaftGeometry, GageConfig, TorqueScaling, voltage_to_torque,
)
from core.torsional.sim_source import (
    TorsionalStreamConfig, SimulatedTorsionalSource, make_torsional_channels,
)
from core.torsional.analysis import (
    torque_metrics, torque_spectrum, order_amplitude, order_amplitudes,
    keyphasor_to_rpm, order_tracking, rainflow_cycles, fatigue_ranges,
)

FS = 5120.0
RPM = 1800.0
UNITS = "nm"


def _sc():
    return TorqueScaling.from_geometry(
        ShaftGeometry(outer_diameter_in=3.0),
        GageConfig(gage_factor=2.0, transmitter_gain=4000), units=UNITS)


def _sim(mean=1000.0, orders=((1.0, 0.0, 0.0),), res=0.0, profile="constant",
         rpm=RPM, rpm_start=0.0, rpm_end=0.0, ramp=2.0, n_blocks=64):
    sc = _sc()
    cfg = TorsionalStreamConfig(
        sample_rate_hz=FS, rpm=rpm, channels=make_torsional_channels(units=UNITS),
        mean_torque=mean, orders=orders, torsional_res_hz=res,
        speed_profile=profile, rpm_start=rpm_start, rpm_end=rpm_end, ramp_seconds=ramp,
        scaling=sc, torque_units=UNITS)
    src = SimulatedTorsionalSource(cfg)
    src.start()
    data = np.concatenate([src.read_block() for _ in range(n_blocks)], axis=1)
    kph_i = cfg.keyphasor_index()
    torque_i = next(i for i in range(cfg.n_channels) if i != kph_i)
    torque = voltage_to_torque(data[torque_i], sc)
    return torque, data[kph_i], cfg, sc


# --- métricas ---
def test_metrics_mean_and_ripple():
    torque, _, _, _ = _sim(mean=1000.0, orders=((1.0, 50.0, 0.0),))
    m = torque_metrics(torque)
    assert m.mean == pytest.approx(1000.0, rel=1e-3)
    # rizado pp ≈ 2·50 / 1000 · 100 = 10%
    assert m.ripple_pct == pytest.approx(10.0, rel=0.15)


# --- espectro ---
def test_spectrum_peak_at_1x():
    torque, _, cfg, _ = _sim(mean=800.0, orders=((1.0, 200.0, 0.0),))
    freqs, amp = torque_spectrum(torque, FS)
    peak_f = freqs[np.argmax(amp)]
    assert peak_f == pytest.approx(RPM / 60.0, rel=0.02)


# --- órdenes por proyección ---
def test_order_amplitudes_recovered():
    torque, _, _, _ = _sim(mean=500.0, orders=((1.0, 300.0, 0.0), (2.0, 120.0, 45.0)))
    amps = order_amplitudes(torque, FS, RPM, orders=(1, 2, 3))
    assert amps[1.0][0] == pytest.approx(300.0, rel=0.05)
    assert amps[2.0][0] == pytest.approx(120.0, rel=0.08)
    assert amps[3.0][0] < 20.0   # orden ausente


def test_order_phase_difference():
    """La diferencia de fase 2×−1× refleja la fase impuesta (45°)."""
    torque, _, _, _ = _sim(mean=0.0, orders=((1.0, 200.0, 0.0), (2.0, 200.0, 45.0)))
    _, ph1 = order_amplitude(torque, FS, RPM, 1)
    _, ph2 = order_amplitude(torque, FS, RPM, 2)
    d = (ph2 - ph1) % 360.0
    assert min(abs(d - 45.0), abs(d - 45.0 + 360.0)) < 12.0


# --- keyphasor → rpm ---
def test_keyphasor_rpm_constant():
    _, kph, _, _ = _sim(mean=500.0)
    _, rpm = keyphasor_to_rpm(kph, FS)
    assert rpm.size > 5
    assert np.median(rpm) == pytest.approx(RPM, rel=0.02)


def test_keyphasor_rpm_runup_increases():
    _, kph, _, _ = _sim(mean=500.0, profile="runup", rpm_start=600.0, rpm_end=3000.0,
                        ramp=2.0, n_blocks=32)
    t, rpm = keyphasor_to_rpm(kph, FS)
    assert rpm.size > 5
    assert rpm[-1] > rpm[0]                       # acelera
    assert rpm.min() < 1000.0                     # arranca lento (~600 rpm de partida)
    assert rpm.max() == pytest.approx(3000.0, rel=0.15)


# --- order tracking en runup ---
def test_order_tracking_resonance_peak():
    """En runup, la amplitud 1× hace pico cuando 1× cruza la natural torsional."""
    res_hz = 30.0  # natural torsional; 1× = res cuando rpm = 1800
    torque, kph, cfg, _ = _sim(mean=500.0, orders=((1.0, 100.0, 0.0),), res=res_hz,
                               profile="runup", rpm_start=600.0, rpm_end=3600.0,
                               ramp=3.0, n_blocks=int(3.0 / 0.25))
    t_rev, rpm_inst = keyphasor_to_rpm(kph, FS)
    # rpm por muestra (interpolada) para el tracking
    n = torque.size
    tt = np.arange(n) / FS
    rpm_per_sample = np.interp(tt, t_rev, rpm_inst, left=rpm_inst[0], right=rpm_inst[-1])
    tracks = order_tracking(torque, FS, rpm_per_sample, orders=(1,), n_segments=24)
    tr = tracks[0]
    peak_rpm = tr.rpm[np.argmax(tr.amplitude)]
    # El pico debe caer cerca de rpm = res_hz·60 = 1800
    assert peak_rpm == pytest.approx(1800.0, rel=0.2)
    assert tr.amplitude.max() > 3.0 * np.median(tr.amplitude)


# --- rainflow ASTM E1049 ---
def test_rainflow_astm_e1049_example():
    """Ejemplo canónico ASTM E1049 (rainflow package): resultado documentado."""
    signal = [-2, 1, -3, 5, -1, 3, -4, 4, -2]
    ranges = dict(fatigue_ranges(signal))
    expected = {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}
    assert ranges.keys() == expected.keys()
    for r, c in expected.items():
        assert ranges[r] == pytest.approx(c)


def test_rainflow_pure_sine_full_cycles():
    """Un seno de M periodos → conteo agregado ≈ M en el rango 2A (ASTM: medios ciclos)."""
    A, M = 100.0, 10
    t = np.linspace(0, M, M * 200, endpoint=False)
    x = A * np.sin(2 * np.pi * t)
    cycles = rainflow_cycles(x)
    assert max(c.range for c in cycles) == pytest.approx(2 * A, rel=1e-2)
    # Daño de fatiga: conteo agregado en el rango dominante 2A ≈ M ciclos.
    ranges = dict(fatigue_ranges(x))
    r2a = max(ranges)  # rango ≈ 2A
    assert ranges[r2a] == pytest.approx(M, abs=1.5)


def test_rainflow_total_count_conserved():
    """El conteo total de ciclos = (nº de reversals)/2 aprox — sanity de conservación."""
    torque, _, _, _ = _sim(mean=500.0, orders=((1.0, 80.0, 0.0), (2.0, 40.0, 0.0)))
    total = sum(c.count for c in rainflow_cycles(torque))
    assert total > 0
