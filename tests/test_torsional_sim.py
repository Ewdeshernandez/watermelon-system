"""
tests.test_torsional_sim
========================

Valida la fuente simulada de par (core.torsional.sim_source):

  · Round-trip: el torque codificado a volts se recupera exacto al decodificar
    con voltage_to_torque (par medio + órdenes en el espectro).
  · Keyphasor: un pulso por revolución (conteo == nº de vueltas).
  · Continuidad de fase entre bloques (sin saltos en las fronteras).
  · Runup: la frecuencia instantánea 1× sigue rpm/60 al acelerar.
  · Resonancia torsional: SDOF amplifica una orden al cruzar la natural.
  · Integración con StreamSource (AcqAgent la consume vía on_block).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.torsional.scaling import (
    ShaftGeometry,
    GageConfig,
    TorqueScaling,
    voltage_to_torque,
)
from core.torsional.sim_source import (
    TorsionalStreamConfig,
    SimulatedTorsionalSource,
    make_torsional_channels,
)


FS = 5120.0
RPM = 1800.0          # f1 = 30 Hz
UNITS = "nm"


def _scaling():
    return TorqueScaling.from_geometry(
        ShaftGeometry(outer_diameter_in=3.0),
        GageConfig(gage_factor=2.0, transmitter_gain=4000),
        units=UNITS,
    )


def _collect(source, n_blocks):
    """Concatena n_blocks bloques → (n_channels, N)."""
    blocks = [source.read_block() for _ in range(n_blocks)]
    return np.concatenate(blocks, axis=1)


def _torque_channel(cfg, data):
    """Extrae el canal de torque (el que NO es keyphasor) en volts."""
    kph = cfg.keyphasor_index()
    ci = next(i for i in range(cfg.n_channels) if i != kph)
    return data[ci]


# -----------------------------------------------------------------
# Round-trip torque medio + órdenes
# -----------------------------------------------------------------
def test_mean_torque_roundtrip():
    """El par medio codificado se recupera al decodificar los volts."""
    sc = _scaling()
    cfg = TorsionalStreamConfig(
        sample_rate_hz=FS, rpm=RPM, channels=make_torsional_channels(units=UNITS),
        mean_torque=1200.0, orders=((1.0, 0.0, 0.0),), scaling=sc, torque_units=UNITS,
    )
    src = SimulatedTorsionalSource(cfg)
    src.start()
    data = _collect(src, 8)
    volts = _torque_channel(cfg, data)
    torque = voltage_to_torque(volts, sc)
    assert np.mean(torque) == pytest.approx(1200.0, rel=1e-6)


def test_orders_recovered_in_spectrum():
    """Un 1× y un 2× de amplitudes conocidas aparecen en el espectro de par."""
    sc = _scaling()
    a1, a2 = 300.0, 120.0
    cfg = TorsionalStreamConfig(
        sample_rate_hz=FS, rpm=RPM, channels=make_torsional_channels(units=UNITS),
        mean_torque=1000.0, orders=((1.0, a1, 0.0), (2.0, a2, 0.0)),
        scaling=sc, torque_units=UNITS,
    )
    src = SimulatedTorsionalSource(cfg)
    src.start()
    volts = _torque_channel(cfg, _collect(src, 64))
    torque = voltage_to_torque(volts, sc, remove_dc=True)

    N = len(torque)
    win = np.hanning(N)
    amp = np.abs(np.fft.rfft(torque * win)) / np.sum(win) * 2.0
    freqs = np.fft.rfftfreq(N, 1.0 / FS)
    f1 = RPM / 60.0

    def amp_at(f):
        k = np.argmin(np.abs(freqs - f))
        return amp[max(0, k - 1):k + 2].max()

    assert amp_at(f1) == pytest.approx(a1, rel=0.1)
    assert amp_at(2 * f1) == pytest.approx(a2, rel=0.15)
    # El piso entre órdenes es mucho menor que las órdenes.
    assert amp_at(1.5 * f1) < 0.1 * a1


# -----------------------------------------------------------------
# Keyphasor once-per-rev
# -----------------------------------------------------------------
def test_keyphasor_pulse_per_rev():
    """Nº de pulsos ≈ nº de revoluciones en la ventana capturada."""
    sc = _scaling()
    cfg = TorsionalStreamConfig(
        sample_rate_hz=FS, rpm=RPM, channels=make_torsional_channels(units=UNITS),
        mean_torque=500.0, scaling=sc, torque_units=UNITS,
    )
    src = SimulatedTorsionalSource(cfg)
    src.start()
    n_blocks = 16
    data = _collect(src, n_blocks)
    kph = data[cfg.keyphasor_index()]

    # Flancos de bajada del pulso (0 → -5)
    edges = np.sum((kph[1:] < -1.0) & (kph[:-1] >= -1.0))
    duration_s = data.shape[1] / FS
    expected_revs = (RPM / 60.0) * duration_s
    assert edges == pytest.approx(expected_revs, abs=1)


# -----------------------------------------------------------------
# Continuidad de fase entre bloques
# -----------------------------------------------------------------
def test_phase_continuity_between_blocks():
    """Un 1× puro debe seguir siendo un tono limpio al unir bloques (sin salto)."""
    sc = _scaling()
    cfg = TorsionalStreamConfig(
        sample_rate_hz=FS, rpm=RPM, channels=make_torsional_channels(units=UNITS),
        mean_torque=0.0, orders=((1.0, 200.0, 0.0),), scaling=sc, torque_units=UNITS,
    )
    src = SimulatedTorsionalSource(cfg)
    src.start()
    volts = _torque_channel(cfg, _collect(src, 32))
    torque = voltage_to_torque(volts, sc, remove_dc=True)

    N = len(torque)
    win = np.hanning(N)
    spec = np.abs(np.fft.rfft(torque * win))
    freqs = np.fft.rfftfreq(N, 1.0 / FS)
    peak_f = freqs[np.argmax(spec)]
    # Si hubiese saltos de fase, la energía se dispersaría; el pico debe caer en 1×.
    assert peak_f == pytest.approx(RPM / 60.0, rel=0.02)
    # Concentración espectral: el pico domina la energía total.
    assert spec.max() ** 2 > 0.5 * np.sum(spec ** 2)


# -----------------------------------------------------------------
# Runup: frecuencia instantánea sigue rpm
# -----------------------------------------------------------------
def test_runup_instantaneous_frequency_tracks_rpm():
    """En runup, el nº de pulsos keyphasor cuadra con la rpm promedio de la rampa."""
    sc = _scaling()
    cfg = TorsionalStreamConfig(
        sample_rate_hz=FS, channels=make_torsional_channels(units=UNITS),
        speed_profile="runup", rpm_start=600.0, rpm_end=3000.0, ramp_seconds=2.0,
        mean_torque=800.0, orders=((1.0, 200.0, 0.0),), scaling=sc, torque_units=UNITS,
    )
    src = SimulatedTorsionalSource(cfg)
    src.start()
    # 2 s de rampa a 0.25 s/bloque = 8 bloques
    n_blocks = int(round(2.0 / cfg.block_seconds))
    data = _collect(src, n_blocks)
    kph = data[cfg.keyphasor_index()]
    edges = np.sum((kph[1:] < -1.0) & (kph[:-1] >= -1.0))
    # Revoluciones esperadas = ∫ f1 dt de una rampa lineal 600→3000 rpm en 2 s
    mean_rpm = 0.5 * (600.0 + 3000.0)
    expected_revs = (mean_rpm / 60.0) * 2.0
    assert edges == pytest.approx(expected_revs, rel=0.05)


# -----------------------------------------------------------------
# Resonancia torsional (SDOF amplifica en la natural)
# -----------------------------------------------------------------
def test_torsional_resonance_amplifies_on_natural():
    """Con f_res = 1× (en resonancia), la amplitud 1× supera al caso fuera de resonancia."""
    sc = _scaling()
    f1 = RPM / 60.0  # 30 Hz

    def rms_1x(res_hz):
        cfg = TorsionalStreamConfig(
            sample_rate_hz=FS, rpm=RPM, channels=make_torsional_channels(units=UNITS),
            mean_torque=0.0, orders=((1.0, 100.0, 0.0),),
            torsional_res_hz=res_hz, torsional_zeta=0.03,
            scaling=sc, torque_units=UNITS,
        )
        src = SimulatedTorsionalSource(cfg)
        src.start()
        volts = _torque_channel(cfg, _collect(src, 32))
        torque = voltage_to_torque(volts, sc, remove_dc=True)
        return np.sqrt(np.mean(torque ** 2))

    on_resonance = rms_1x(f1)         # natural coincide con 1×
    off_resonance = rms_1x(10.0 * f1)  # natural muy por encima → ganancia ~1
    assert on_resonance > 3.0 * off_resonance


# -----------------------------------------------------------------
# Integración con StreamSource / AcqAgent (on_block)
# -----------------------------------------------------------------
def test_streamsource_interface_and_on_block():
    """La fuente cumple la interfaz StreamSource y alimenta el AcqAgent."""
    from core.remote_monitoring.agent import AcqAgent

    sc = _scaling()
    cfg = TorsionalStreamConfig(
        sample_rate_hz=FS, rpm=RPM, channels=make_torsional_channels(units=UNITS),
        mean_torque=750.0, orders=((1.0, 150.0, 0.0),), scaling=sc, torque_units=UNITS,
    )
    src = SimulatedTorsionalSource(cfg)

    captured = []
    agent = AcqAgent(source=src, on_block=captured.append)
    # Ejercemos el ciclo manualmente sin arrancar el hilo interno.
    src.start()
    for _ in range(4):
        blk = src.read_block()
        agent._ingest(blk, now=0.0)
    src.stop()

    assert len(captured) == 4
    assert captured[0].shape == (cfg.n_channels, cfg.block_samples)
    assert agent.sample_rate_hz == FS
