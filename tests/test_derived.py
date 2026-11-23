"""Mediciones derivadas: 1 acelerómetro → aceleración + velocidad (ISO 20816)."""

import numpy as np

from core.remote_monitoring.config import AcqSetup, ChannelRow, MachineConfig, validate_setup
from core.remote_monitoring.derived import G_MM_S2, velocity_rms_from_accel


def test_velocity_from_1g_sine_at_100hz():
    fs, f0, n = 25600.0, 100.0, 25600
    t = np.arange(n) / fs
    a = 1.0 * np.sin(2 * np.pi * f0 * t)                 # 1 g pico
    expected = G_MM_S2 / (2 * np.pi * f0) / np.sqrt(2)    # 11.04 mm/s rms
    assert abs(velocity_rms_from_accel(a, fs) - expected) / expected < 0.01


def test_band_rejects_out_of_band():
    fs, n = 25600.0, 25600
    t = np.arange(n) / fs
    a = np.sin(2 * np.pi * 3000.0 * t)                    # fuera de 10–1000 Hz
    assert velocity_rms_from_accel(a, fs) < 0.05


def _setup(*chs):
    return AcqSetup(machine=MachineConfig(), channels=list(chs))


def test_derived_channel_has_no_bnc_conflict():
    a = ChannelRow(bnc_port=5, point_label="4Y_A", sensor_type="accelerometer", unit_native="g rms")
    v = ChannelRow(bnc_port=5, point_label="4Y_V", sensor_type="velometer", unit_native="mm/s rms",
                   derived_from="4Y_A", measure="velocity")
    codes = {f.code for f in validate_setup(_setup(a, v))}
    assert "dup_bnc" not in codes and "derived_source" not in codes


def test_derived_must_point_to_accelerometer():
    p = ChannelRow(bnc_port=1, point_label="1Y_D", sensor_type="proximity")
    v = ChannelRow(bnc_port=0, point_label="1Y_V", sensor_type="velometer", derived_from="1Y_D", measure="velocity")
    w = ChannelRow(bnc_port=0, point_label="9Y_V", sensor_type="velometer", derived_from="NOPE", measure="velocity")
    codes = [f.code for f in validate_setup(_setup(p, v, w))]
    assert codes.count("derived_source") == 2
