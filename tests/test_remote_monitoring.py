"""
import os
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
    AcqAgent,
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


# ============================================================ config (Fase 1)
def test_auto_layout_pairs_and_keyphasor():
    from core.remote_monitoring.config import MachineConfig, auto_layout
    m = MachineConfig(name="X", n_bearings=3)
    rows = auto_layout(m)
    labels = [r.point_label for r in rows]
    assert labels == ["1Y", "1X", "2Y", "2X", "3Y", "3X", "KPH"]
    # BNC secuencial y único
    assert [r.bnc_port for r in rows] == list(range(1, 8))
    # keyphasor detectado
    assert rows[-1].is_keyphasor()


def test_validate_flags_alert_ge_danger():
    from core.remote_monitoring.config import (
        MachineConfig, AcqSetup, ChannelRow, validate_setup, is_setup_valid)
    ch = ChannelRow(bnc_port=1, point_label="1Y", plane=1, sensor_type="proximity",
                    sensitivity_mv_per_eu=200, unit_native="mil pp", coupling="AC",
                    angle_deg=0, alarm=5.0, danger=4.0)
    setup = AcqSetup(machine=MachineConfig(), channels=[ch])
    codes = {f.code for f in validate_setup(setup)}
    assert "alert_ge_danger" in codes
    assert not is_setup_valid(setup)


def test_validate_flags_non_orthogonal_xy():
    from core.remote_monitoring.config import (
        MachineConfig, AcqSetup, ChannelRow, validate_setup)
    a = ChannelRow(bnc_port=1, point_label="1Y", plane=1, sensor_type="proximity",
                   sensitivity_mv_per_eu=200, angle_deg=0)
    b = ChannelRow(bnc_port=2, point_label="1X", plane=1, sensor_type="proximity",
                   sensitivity_mv_per_eu=200, angle_deg=45)  # 45° != 90°
    setup = AcqSetup(machine=MachineConfig(), channels=[a, b])
    codes = {f.code for f in validate_setup(setup)}
    assert "xy_not_orthogonal" in codes


def test_validate_warns_no_keyphasor():
    from core.remote_monitoring.config import (
        MachineConfig, AcqSetup, ChannelRow, validate_setup)
    a = ChannelRow(bnc_port=1, point_label="1Y", plane=1, sensor_type="proximity",
                   sensitivity_mv_per_eu=200, angle_deg=0)
    b = ChannelRow(bnc_port=2, point_label="1X", plane=1, sensor_type="proximity",
                   sensitivity_mv_per_eu=200, angle_deg=90)
    setup = AcqSetup(machine=MachineConfig(), channels=[a, b])
    codes = {f.code for f in validate_setup(setup)}
    assert "no_keyphasor" in codes


def test_setup_bridges_to_channel_configs_and_sensor_map():
    from core.remote_monitoring.config import (
        MachineConfig, auto_layout, AcqSetup,
        setup_to_channel_configs, setup_to_sensor_map)
    setup = AcqSetup(machine=MachineConfig(n_bearings=2), channels=auto_layout(MachineConfig(n_bearings=2)))
    ccs = setup_to_channel_configs(setup)
    assert [c.bnc_port for c in ccs] == [1, 2, 3, 4, 5]
    assert ccs[0].coupling == "AC" and ccs[-1].coupling == "DC"
    sm = setup_to_sensor_map(setup)
    assert len(sm) == 5
    assert sm[0]["sensitivity_mv_per_eu"] == 200.0
    assert sm[0]["angle_deg"] == 45.0  # Bently: Y a 45° (lado L)


def test_absolute_angle_bently_convention():
    from core.remote_monitoring.config import absolute_angle, angular_separation
    assert absolute_angle(45, "R") == 45.0
    assert absolute_angle(45, "L") == 315.0
    assert absolute_angle(90, "") == 90.0
    # 45R y 45L quedan a 90° (el bug reportado)
    assert angular_separation(absolute_angle(45, "R"), absolute_angle(45, "L")) == 90.0


def test_auto_layout_pair_is_orthogonal_with_sides():
    from core.remote_monitoring.config import MachineConfig, auto_layout, AcqSetup, validate_setup
    m = MachineConfig(n_bearings=1)
    setup = AcqSetup(machine=m, channels=auto_layout(m))
    codes = {f.code for f in validate_setup(setup)}
    assert "xy_not_orthogonal" not in codes  # 45L + 45R = 90°


def test_unit_mismatch_flagged():
    from core.remote_monitoring.config import (
        MachineConfig, AcqSetup, ChannelRow, validate_setup)
    ch = ChannelRow(bnc_port=1, point_label="1Y", plane=1, sensor_type="accelerometer",
                    sensitivity_mv_per_eu=100, unit_native="mil pp", angle_deg=45, side="L")
    setup = AcqSetup(machine=MachineConfig(), channels=[ch])
    codes = {f.code for f in validate_setup(setup)}
    assert "unit_mismatch" in codes


def test_setup_persistence_roundtrip():
    import tempfile, os
    os.environ["WM_PERSIST_DIR"] = tempfile.mkdtemp()
    from core.remote_monitoring import config as cfg
    m = cfg.MachineConfig(name="Persist Test", rpm_nominal=1800, n_bearings=1)
    setup = cfg.AcqSetup(machine=m, channels=cfg.auto_layout(m))
    cfg.save_setup(setup)
    assert "Persist_Test" in cfg.list_setups()
    loaded = cfg.load_setup("Persist Test")
    assert loaded is not None
    assert loaded.machine.rpm_nominal == 1800
    assert [c.point_label for c in loaded.channels] == ["1Y", "1X", "KPH"]
    del os.environ["WM_PERSIST_DIR"]


# ============================================================ states (Fase 2)
def test_classify_state_transitions():
    from core.remote_monitoring.states import (
        classify_state, OFF, SLOW_ROLL, STARTUP, COASTDOWN, STEADY)
    assert classify_state(0, None) == OFF
    assert classify_state(50, 50) == SLOW_ROLL
    assert classify_state(3000, 2900) == STARTUP
    assert classify_state(3000, 3100) == COASTDOWN
    assert classify_state(3000, 3000) == STEADY


# ========================================================= transient (Fase 2)
def _runup_channels():
    return [
        ChannelConfig(name="1Y", coupling="AC", sensitivity_mv_per_eu=200.0, bnc_port=1, units="mil"),
        ChannelConfig(name="KPH", coupling="DC", sensitivity_mv_per_eu=1.0, bnc_port=2, units="V"),
    ]


def test_transient_capture_builds_bode_with_resonance_peak():
    from core.remote_monitoring.transient import TransientCapture, TransientConfig
    from core.remote_monitoring.stream_source import is_keyphasor_channel
    fs = 5120
    chans = _runup_channels()
    cfg = StreamConfig(sample_rate_hz=fs, channels=chans, block_seconds=0.25,
                       buffer_seconds=2.0, speed_profile="runup",
                       rpm_start=600, rpm_end=6000, ramp_seconds=15,
                       sim_critical_rpm=3000, sim_zeta=0.04, noise_rms=0.004)
    agent = AcqAgent(SimulatedStreamSource(cfg), instance_id="runup")
    tc = TransientCapture(TransientConfig(delta_rpm=50, min_rpm=200, capture_samples=2048))
    vib = [(i, c) for i, c in enumerate(chans) if not is_keyphasor_channel(c)]

    captured = 0
    for _ in range(60):  # ~15 s → recorre la rampa
        agent.pump(1)
        snap = agent.snapshot()
        rpm = agent.estimate_rpm(snap)
        if tc.feed(snap, rpm, fs, vib):
            captured += 1

    assert tc.n_samples >= 10, f"pocos puntos capturados: {tc.n_samples}"
    rpms, amp, phase = tc.bode("1Y")
    # rpm ordenado ascendente
    assert np.all(np.diff(rpms) >= 0)
    # el pico de amplitud 1X cae cerca de la crítica (3000 rpm)
    peak_rpm = rpms[int(np.argmax(amp))]
    assert 2500 < peak_rpm < 3500, f"pico de bode en {peak_rpm:.0f}, se esperaba ~3000"
    # la fase gira (0→180) al pasar la crítica: subió de forma monótona neta
    assert phase.max() - phase.min() > 60


def test_transient_cascade_shape():
    from core.remote_monitoring.transient import TransientCapture, TransientConfig
    from core.remote_monitoring.stream_source import is_keyphasor_channel
    fs = 5120
    chans = _runup_channels()
    cfg = StreamConfig(sample_rate_hz=fs, channels=chans, block_seconds=0.25,
                       buffer_seconds=2.0, speed_profile="runup",
                       rpm_start=600, rpm_end=4000, ramp_seconds=10,
                       sim_critical_rpm=2500, sim_zeta=0.05)
    agent = AcqAgent(SimulatedStreamSource(cfg), instance_id="runup")
    tc = TransientCapture(TransientConfig(delta_rpm=50, min_rpm=200, capture_samples=2048, fmax_hz=300))
    vib = [(i, c) for i, c in enumerate(chans) if not is_keyphasor_channel(c)]
    for _ in range(50):
        agent.pump(1)
        tc.feed(agent.snapshot(), agent.estimate_rpm(), fs, vib)
    rpms, freqs, mat = tc.cascade("1Y")
    assert mat.shape == (len(rpms), len(freqs))
    assert len(rpms) >= 5
    assert freqs.max() <= 300 + 5   # respetó fmax


# ================================================= campos ADRE/acq (Fase nueva)
def test_channelrow_new_fields_and_backcompat():
    from core.remote_monitoring.config import MachineConfig, auto_layout
    rows = auto_layout(MachineConfig(n_bearings=1))
    prox = rows[0]
    assert prox.full_scale == 10.0 and prox.gap_bias_v == -9.5 and prox.active is True
    kph = rows[-1]
    assert kph.events_per_rev == 1 and kph.trigger_v == -7.0


def test_acquisition_params_delta_f_and_validation():
    from core.remote_monitoring.config import (
        MachineConfig, AcqSetup, ChannelRow, AcquisitionParams, validate_setup)
    acq = AcquisitionParams(fmax_hz=2000, lines=1600)
    assert abs(acq.delta_f() - 1.25) < 1e-6
    # Fmin >= Fmax → error
    bad = AcqSetup(machine=MachineConfig(),
                   channels=[ChannelRow(1, "1Y", 1, "proximity", 200, "mil pp", "AC", 45, "L")],
                   acquisition=AcquisitionParams(fmax_hz=100, fmin_hz=200))
    assert any(f.code == "acq_freq_range" and f.level == "error" for f in validate_setup(bad))


def test_gap_out_of_range_flagged():
    from core.remote_monitoring.config import (
        MachineConfig, AcqSetup, ChannelRow, validate_setup)
    ch = ChannelRow(1, "1Y", 1, "proximity", 200, "mil pp", "AC", 45, "L", gap_bias_v=+5.0)
    codes = {f.code for f in validate_setup(AcqSetup(machine=MachineConfig(), channels=[ch]))}
    assert "gap_out_of_range" in codes


def test_full_acqsetup_persist_with_acquisition(tmp_path=None):
    import os
    import tempfile
    os.environ["WM_PERSIST_DIR"] = tempfile.mkdtemp()
    from core.remote_monitoring import config as cfg
    m = cfg.MachineConfig(name="Persist Acq", n_bearings=1)
    setup = cfg.AcqSetup(machine=m, channels=cfg.auto_layout(m),
                         acquisition=cfg.AcquisitionParams(fmax_hz=5000, lines=3200, window="flattop"))
    cfg.save_setup(setup)
    loaded = cfg.load_setup("Persist Acq")
    assert loaded.acquisition.fmax_hz == 5000
    assert loaded.acquisition.lines == 3200
    assert loaded.acquisition.window == "flattop"
    assert loaded.channels[0].gap_bias_v == -9.5
    del os.environ["WM_PERSIST_DIR"]


def test_orders_waveform_notch_persist():
    import os, tempfile
    os.environ["WM_PERSIST_DIR"] = tempfile.mkdtemp()
    from core.remote_monitoring import config as cfg
    m = cfg.MachineConfig(name="Orders Test", n_bearings=1)
    rows = cfg.auto_layout(m)
    rows[-1].notch_type = "muesca"  # keyphasor
    setup = cfg.AcqSetup(machine=m, channels=rows,
                         acquisition=cfg.AcquisitionParams(waveform_mode="asynchronous",
                                                           orders=[0.5, 1.0, 2.0, 3.0]))
    cfg.save_setup(setup)
    loaded = cfg.load_setup("Orders Test")
    assert loaded.acquisition.waveform_mode == "asynchronous"
    assert loaded.acquisition.orders == [0.5, 1.0, 2.0, 3.0]
    assert loaded.channels[-1].notch_type == "muesca"
    del os.environ["WM_PERSIST_DIR"]
