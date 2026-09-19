"""Tests de la estructura de adquisición NI real (P1) — sin hardware."""
import numpy as np
import pytest

from core.torsional.analysis import keyphasor_to_rpm
from core.torsional.ni_source import (
    KeyphasorSensor, NITorsionalConfig, NITorsionalSource,
    nidaqmx_available, rpm_from_keyphasor, KEYPHASOR_SENSORS,
)

FS = 2560.0


def _pulse_train(rpm, fs, secs, ppr=1, positive=True, baseline=0.0, amp=5.0):
    """Tren de pulsos periódico a `rpm` (ppr por vuelta). Positivo = foto-tacómetro."""
    n = int(fs * secs)
    x = np.full(n, baseline, dtype=float)
    period = fs / (rpm / 60.0 * ppr)          # muestras entre pulsos
    idx = np.arange(0, n, period).astype(int)
    w = max(2, int(period * 0.1))
    for i in idx:
        x[i:i + w] = baseline + (amp if positive else -amp)
    return x


def test_phototach_rising_edge_rpm():
    """Foto-tacómetro: pulso positivo, flanco de subida → rpm correcta."""
    s = KeyphasorSensor.phototach_reflective(strips=1, level_v=2.5)
    kph = _pulse_train(1800.0, FS, 2.0, ppr=1, positive=True)
    _, rpm = rpm_from_keyphasor(kph, FS, s)
    assert rpm.size > 3
    assert abs(np.median(rpm) - 1800.0) / 1800.0 < 0.03


def test_proximitor_falling_edge_rpm():
    """Bently 3300 XL: señal DC negativa, keyway más negativo → flanco de bajada."""
    s = KeyphasorSensor.bently_3300xl_8mm(keyways=1, gap_bias_v=-10.0)
    # bias -10 V, el keyway baja a ~-15 V; threshold = -13 V
    kph = _pulse_train(3600.0, FS, 2.0, ppr=1, positive=False, baseline=-10.0, amp=5.0)
    _, rpm = rpm_from_keyphasor(kph, FS, s)
    assert rpm.size > 3
    assert abs(np.median(rpm) - 3600.0) / 3600.0 < 0.03


def test_pulses_per_rev_scaling():
    """2 cintas reflectivas → 2 pulsos/vuelta; la rpm debe seguir siendo la real."""
    s = KeyphasorSensor.phototach_reflective(strips=2, level_v=2.5)
    kph = _pulse_train(1200.0, FS, 3.0, ppr=2, positive=True)
    _, rpm = rpm_from_keyphasor(kph, FS, s)
    assert abs(np.median(rpm) - 1200.0) / 1200.0 < 0.03


def test_presets_have_correct_edges():
    assert KeyphasorSensor.bently_3300xl_8mm().edge == "falling"
    assert KeyphasorSensor.phototach_reflective().edge == "rising"
    assert set(KEYPHASOR_SENSORS) == {"proximitor_3300xl", "phototach_reflective", "sim"}


def test_edge_param_backward_compatible():
    """El default de keyphasor_to_rpm sigue siendo flanco de bajada."""
    kph = _pulse_train(1800.0, FS, 1.5, positive=False, baseline=0.0, amp=5.0)
    _, rpm_def = keyphasor_to_rpm(kph, FS, threshold=-1.0)
    _, rpm_fall = keyphasor_to_rpm(kph, FS, threshold=-1.0, edge="falling")
    assert np.allclose(rpm_def, rpm_fall)


def test_ni_config_shape():
    cfg = NITorsionalConfig(device="cDAQ1Mod1")
    assert cfg.n_channels == 2 and cfg.keyphasor_index() == 1
    assert cfg.block_samples == int(round(cfg.sample_rate_hz * cfg.block_seconds))


def test_ni_source_no_driver_raises_clear_error():
    """Sin NI-DAQmx (Mac/CI) start() debe fallar con mensaje claro, NUNCA simular."""
    if nidaqmx_available():
        pytest.skip("nidaqmx presente en este equipo")
    src = NITorsionalSource(NITorsionalConfig(device="cDAQ1Mod1"))
    with pytest.raises(RuntimeError):
        src.start()
