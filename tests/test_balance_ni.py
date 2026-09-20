"""Tests del motor de adquisición NI para balanceo (extracción 1× + velocidad)."""
import numpy as np

from core.torsional.ni_source import KeyphasorSensor
from core.balance.ni_balance import (
    extract_1x, accel_to_velocity, velocity_rms_mm_s, one_x_accel_to_velocity,
    VibChannel, NIBalanceConfig, NIBalanceSource, nidaqmx_available,
)

FS = 5120.0
RPM = 1800.0
F1 = RPM / 60.0            # 30 Hz


def _keyphasor_positive(n, p0=10):
    """Tren de pulsos POSITIVOS (foto-tacómetro) una vez por vuelta desde p0."""
    k = np.zeros(n)
    P = FS / F1
    idx = (p0 + np.arange(0, (n - p0) / P) * P).astype(int)
    for i in idx:
        if i + 3 < n:
            k[i:i + 3] = 5.0
    return k, p0


def _vib(n, amp, phase_deg, p0):
    """Vibración A·cos(2π f1 (n−p0)/fs + φ), fase referida al primer pulso."""
    t = (np.arange(n) - p0) / FS
    return amp * np.cos(2 * np.pi * F1 * t + np.radians(phase_deg))


def test_extract_1x_recovers_amp_phase_rpm():
    secs = 20 / F1                      # 20 vueltas exactas (poca fuga)
    n = int(FS * secs)
    kph, p0 = _keyphasor_positive(n)
    sensor = KeyphasorSensor.phototach_reflective(strips=1)
    v = _vib(n, amp=8.0, phase_deg=0.0, p0=p0)          # 0-pk = 8 → pp = 16
    amp, lag, rpm = extract_1x(v, kph, FS, sensor, to_pp=True)
    assert abs(rpm - RPM) / RPM < 0.02
    assert abs(amp - 16.0) / 16.0 < 0.05               # pp
    assert min(lag, 360 - lag) < 5.0                    # fase ≈ 0


def test_extract_1x_phase_is_consistent():
    """Dos señales que difieren 90° dan lag que difiere 90° (para el coef. de influencia)."""
    secs = 20 / F1; n = int(FS * secs)
    kph, p0 = _keyphasor_positive(n)
    sensor = KeyphasorSensor.phototach_reflective(strips=1)
    _, lag_a, _ = extract_1x(_vib(n, 5.0, 0.0, p0), kph, FS, sensor)
    _, lag_b, _ = extract_1x(_vib(n, 5.0, 90.0, p0), kph, FS, sensor)
    d = (lag_b - lag_a) % 360.0
    assert abs(min(d, 360 - d) - 90.0) < 4.0


def test_accel_to_velocity_rms():
    """Aceleración senoidal → velocidad RMS = (Aa/ω)/√2 · 1000 mm/s."""
    secs = 20 / F1; n = int(FS * secs)
    Aa = 2.0                                            # m/s²
    t = np.arange(n) / FS
    a = Aa * np.cos(2 * np.pi * F1 * t)
    vrms = velocity_rms_mm_s(a, FS)
    expected = (Aa / (2 * np.pi * F1)) / np.sqrt(2) * 1000.0
    assert abs(vrms - expected) / expected < 0.05


def test_one_x_accel_to_velocity_math():
    amp_v, ph = one_x_accel_to_velocity(2.0, 63.0, RPM)
    expected = 2.0 / (2 * np.pi * F1) * 1000.0
    assert abs(amp_v - expected) / expected < 1e-6
    assert abs(ph - (63.0 - 90.0) % 360.0) < 1e-9


def test_config_keyphasor_is_channel_zero():
    cfg = NIBalanceConfig(vib_channels=[VibChannel("A"), VibChannel("B")])
    assert cfg.kph_ai == 0
    src = NIBalanceSource(cfg)
    assert src.keyphasor_index() == 0 and src.n_channels == 3    # kph + 2 vib


def test_ni_source_no_driver_raises():
    import pytest
    if nidaqmx_available():
        pytest.skip("nidaqmx presente")
    src = NIBalanceSource(NIBalanceConfig(vib_channels=[VibChannel("A")]))
    with pytest.raises(RuntimeError):
        src.start()


def test_vibchannel_unit_and_coupling():
    prox = VibChannel("1Y", kind="proximity_9229")
    accel = VibChannel("A", kind="accel_9234")
    assert prox.coupling == "DC" and prox.unit == "µm pk-pk"
    assert accel.coupling == "IEPE" and accel.unit == "mm/s RMS"
