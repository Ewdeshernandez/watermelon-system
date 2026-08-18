"""
Tests del módulo Remote Monitoring (streaming en vivo).

Corren SIN hardware — usan SimulatedStreamSource. Validan:
  · RingBuffer: orden cronológico, wrap-around, bloque > capacidad.
  · SimulatedStreamSource: shape, continuidad entre bloques, keyphasor.
  · materialize: produce Signal con fs correcto y pico 1X en rpm/60.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.modal.acq_backend import ChannelConfig
from core.remote_monitoring import (
    RingBuffer,
    SimulatedStreamSource,
    StreamConfig,
    window_to_signals,
)


def _channels():
    return [
        ChannelConfig(name="1Y", coupling="AC", sensitivity_mv_per_eu=200.0, bnc_port=1, units="mil"),
        ChannelConfig(name="1X", coupling="AC", sensitivity_mv_per_eu=200.0, bnc_port=2, units="mil"),
        ChannelConfig(name="KPH", coupling="DC", sensitivity_mv_per_eu=1.0, bnc_port=3, units="V"),
    ]


# --------------------------------------------------------------------- RingBuffer
def test_ring_buffer_chronological_order():
    rb = RingBuffer(n_channels=1, capacity_samples=5)
    rb.write(np.array([[1, 2, 3]], dtype=float))
    rb.write(np.array([[4, 5]], dtype=float))
    snap = rb.snapshot()
    assert snap.shape == (1, 5)
    np.testing.assert_array_equal(snap[0], [1, 2, 3, 4, 5])


def test_ring_buffer_wraparound_keeps_latest():
    rb = RingBuffer(n_channels=1, capacity_samples=5)
    rb.write(np.arange(1, 8, dtype=float)[None, :])  # 1..7 → últimos 5 = 3..7
    snap = rb.snapshot()
    np.testing.assert_array_equal(snap[0], [3, 4, 5, 6, 7])


def test_ring_buffer_block_larger_than_capacity():
    rb = RingBuffer(n_channels=2, capacity_samples=3)
    block = np.vstack([np.arange(10), np.arange(10, 20)]).astype(float)
    rb.write(block)
    snap = rb.snapshot()
    assert snap.shape == (2, 3)
    np.testing.assert_array_equal(snap[0], [7, 8, 9])
    np.testing.assert_array_equal(snap[1], [17, 18, 19])


def test_ring_buffer_partial_snapshot():
    rb = RingBuffer(n_channels=1, capacity_samples=10)
    rb.write(np.arange(1, 9, dtype=float)[None, :])
    np.testing.assert_array_equal(rb.snapshot(3)[0], [6, 7, 8])


# --------------------------------------------------------------- SimulatedStreamSource
def test_sim_source_block_shape_and_continuity():
    cfg = StreamConfig(sample_rate_hz=5120, channels=_channels(),
                       block_seconds=0.1, buffer_seconds=2.0, rpm=3600)
    src = SimulatedStreamSource(cfg)
    src.start()
    b1 = src.read_block()
    b2 = src.read_block()
    assert b1.shape == (3, cfg.block_samples)
    assert b2.shape == (3, cfg.block_samples)
    # continuidad: reconstruyendo con el mismo cursor, no debe haber salto
    # abrupto de fase en la frontera (diferencia acotada en canal de vib).
    boundary_jump = abs(b2[0, 0] - b1[0, -1])
    assert boundary_jump < 0.5
    src.stop()


def test_sim_source_keyphasor_pulses_once_per_rev():
    cfg = StreamConfig(sample_rate_hz=5120, channels=_channels(),
                       block_seconds=1.0, buffer_seconds=2.0, rpm=3600)  # 60 Hz → 60 pulsos/s
    src = SimulatedStreamSource(cfg)
    src.start()
    block = src.read_block()  # 1 segundo
    kph = block[2]
    # detecta flancos descendentes (entra a pulso negativo)
    is_pulse = kph < -1.0
    edges = np.sum((~is_pulse[:-1]) & (is_pulse[1:]))
    assert 55 <= edges <= 65  # ~60 rev/s
    src.stop()


def test_sim_source_requires_start():
    cfg = StreamConfig(sample_rate_hz=1000, channels=_channels()[:1])
    src = SimulatedStreamSource(cfg)
    with pytest.raises(RuntimeError):
        src.read_block()


# --------------------------------------------------------------------- materialize
def test_materialize_produces_signals_with_1x_peak():
    fs = 5120
    rpm = 3600
    f1 = rpm / 60.0  # 60 Hz
    cfg = StreamConfig(sample_rate_hz=fs, channels=_channels(),
                       block_seconds=0.5, buffer_seconds=4.0, rpm=rpm,
                       defect="unbalance", noise_rms=0.005)
    src = SimulatedStreamSource(cfg)
    rb = RingBuffer(cfg.n_channels, cfg.buffer_samples)
    src.start()
    for _ in range(6):  # 3 s de data
        rb.write(src.read_block())
    src.stop()

    snap = rb.snapshot()
    signals = window_to_signals(snap, cfg.channels, fs, rpm=rpm)
    # keyphasor excluido por defecto → 2 señales de vibración
    assert len(signals) == 2

    sig = signals[0]
    assert sig.metadata["fs"] == fs
    assert sig.metadata["rpm"] == rpm
    assert sig.metadata["role"] == "vibration"

    # FFT: el pico dominante debe caer en f1 (60 Hz) por el desbalance
    x = sig.x - np.mean(sig.x)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    peak_f = freqs[np.argmax(spec)]
    assert abs(peak_f - f1) < 2.0  # dentro de 2 Hz del 1X


def test_materialize_can_include_keyphasor():
    fs = 2000
    cfg = StreamConfig(sample_rate_hz=fs, channels=_channels(),
                       block_seconds=0.5, buffer_seconds=2.0)
    src = SimulatedStreamSource(cfg)
    rb = RingBuffer(cfg.n_channels, cfg.buffer_samples)
    src.start(); rb.write(src.read_block()); src.stop()
    sigs = window_to_signals(rb.snapshot(), cfg.channels, fs, include_keyphasor=True)
    roles = sorted(s.metadata["role"] for s in sigs)
    assert roles == ["keyphasor", "vibration", "vibration"]


# ============================================================ keyphasor
def test_keyphasor_detects_rpm():
    from core.remote_monitoring import detect_keyphasor
    fs, rpm = 5120, 3600
    f1 = rpm / 60.0
    t = np.arange(3 * fs) / fs
    period = 1.0 / f1
    phase = np.mod(t, period) / period
    kph = np.where(phase < 0.02, -5.0, 0.0)
    r = detect_keyphasor(kph, fs)
    assert r.rpm is not None and abs(r.rpm - rpm) < 25
    assert r.n_pulses > 150


def test_one_x_vector_recovers_amplitude():
    from core.remote_monitoring import one_x_vector
    fs, f1 = 5120, 60.0
    t = np.arange(2 * fs) / fs
    vib = 3.0 * np.sin(2 * np.pi * f1 * t + 0.7)
    amp, phase = one_x_vector(vib, fs, f1, ref_sample=0)
    assert abs(amp - 3.0) < 0.2
    assert 0.0 <= phase < 360.0


def test_keyphasor_no_pulses_returns_none():
    from core.remote_monitoring import detect_keyphasor
    flat = np.zeros(1000)
    r = detect_keyphasor(flat, 1000)
    assert r.rpm is None and r.n_pulses == 0


# ================================================================ agent
def test_agent_pump_fills_buffer_and_estimates_rpm():
    from core.remote_monitoring import AcqAgent
    cfg = StreamConfig(sample_rate_hz=5120, channels=_channels(),
                       block_seconds=0.25, buffer_seconds=3.0, rpm=3000)
    agent = AcqAgent(SimulatedStreamSource(cfg), instance_id="test")
    agent.run_for(2.0)
    snap = agent.snapshot()
    assert snap.shape[0] == 3 and snap.shape[1] > 0
    rpm = agent.estimate_rpm()
    assert rpm is not None and abs(rpm - 3000) < 40
    sigs = agent.live_signals()
    assert len(sigs) == 2  # keyphasor excluido


def test_agent_persists_to_store():
    tmp_path = Path(tempfile.mkdtemp())
    from core.remote_monitoring import AcqAgent, LocalStore
    store = LocalStore(root=tmp_path / "rm")
    cfg = StreamConfig(sample_rate_hz=2000, channels=_channels(),
                       block_seconds=0.25, buffer_seconds=2.0, rpm=1800)
    agent = AcqAgent(SimulatedStreamSource(cfg), instance_id="C200C",
                     store=store, persist_every_s=0.0)  # persistir cada bloque
    agent.run_for(1.0)
    snaps = store.list_snapshots("C200C")
    assert len(snaps) >= 1
    assert store.count(only_pending=True) >= 1


# ================================================================ store
def test_store_roundtrip_and_sync():
    tmp_path = Path(tempfile.mkdtemp())
    from core.remote_monitoring import LocalStore
    store = LocalStore(root=tmp_path / "rm")
    data = np.random.default_rng(0).standard_normal((4, 500))
    ch_meta = [{"name": f"c{i}", "bnc_port": i + 1, "coupling": "AC",
                "sensitivity_mv_per_eu": 200.0, "units": "mil"} for i in range(4)]
    meta = store.save_snapshot("TES3", data, ch_meta, fs=5120, rpm=3600)
    assert not meta.synced
    loaded = store.load_snapshot(meta.snapshot_id)
    assert loaded is not None
    np.testing.assert_allclose(loaded["data"], data, rtol=1e-5)
    assert loaded["rpm"] == 3600 and loaded["fs"] == 5120
    assert len(store.pending_sync()) == 1
    store.mark_synced(meta.snapshot_id)
    assert len(store.pending_sync()) == 0
    assert store.count(only_pending=True) == 0
