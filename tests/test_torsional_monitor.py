"""Tests del monitoreo torsional de larga duración (streaming rainflow + tendencia)."""
import numpy as np

from core.torsional.analysis import rainflow_cycles, fatigue_ranges, shaft_torsional_fatigue
from core.torsional.monitor import StreamingRainflow, TorsionalMonitor, _block_reversals


def _reversals(series):
    """Puntos de retorno de una serie completa (para alimentar el streaming en tests)."""
    r, carry, cdir = _block_reversals(np.asarray(series, float), None)
    return r


def test_streaming_matches_batch_rainflow():
    """El histograma streaming = el rainflow por lotes sobre la misma serie."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 4000))          # camino aleatorio con muchos retornos
    sf = StreamingRainflow(); sf.feed(_reversals(x))
    stream = dict((round(r, 3), round(c, 3)) for r, c in sf.ranges(drain=True, tail=float(x[-1])))
    batch = dict((round(r, 3), round(c, 3)) for r, c in fatigue_ranges(x))
    # mismos rangos y conteos (tolerancia por redondeo)
    assert set(stream) == set(batch)
    for k in batch:
        assert abs(stream[k] - batch[k]) < 1e-6


def test_streaming_survives_block_split():
    """Alimentar por BLOQUES da el mismo histograma que de una."""
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1, 6000))
    whole = StreamingRainflow(); whole.feed(_reversals(x))
    mon = TorsionalMonitor(trend_dt=1e9)           # sin cortes de tendencia
    for blk in np.array_split(x, 13):              # bloques irregulares
        mon.add_block(blk, rpm=1800, fs=2560)
    a = whole.ranges(drain=True, tail=float(x[-1]))
    b = mon.rf.ranges(drain=True, tail=mon._carry)
    # Fronteras de bloque pueden mover unos pocos medios-ciclos; el total y la
    # energía de daño (Σ rango²·conteo) deben coincidir casi exacto.
    assert abs(sum(c for _, c in a) - sum(c for _, c in b)) <= 13     # ≤ 1 por bloque
    e_a = sum(r * r * c for r, c in a); e_b = sum(r * r * c for r, c in b)
    assert abs(e_a - e_b) / e_a < 0.02


def test_monitor_trend_and_fatigue():
    """La tendencia se llena y el histograma alimenta el diagnóstico de fatiga."""
    fs = 2000.0
    t = np.arange(0, 1.0, 1 / fs)
    mon = TorsionalMonitor(units="ftlb", trend_dt=1.0, event_pp=8000.0)
    # 10 s de par: medio 4000 + dinámico 2000 @30 Hz
    for _ in range(10):
        blk = 4000 + 2000 * np.sin(2 * np.pi * 30 * t)
        mon.add_block(blk, rpm=1800, fs=fs)
    assert len(mon.trend) >= 9                      # ~1 fila por segundo
    assert mon.tmax > 5900 and mon.tmin < 2100
    cyc = mon.fatigue_cycles()
    assert cyc and sum(c.count for c in cyc) > 100
    life = shaft_torsional_fatigue(cyc, 3.0, 0.0, 90000, "ftlb", window_seconds=mon.duration_s)
    assert life.status in ("green", "yellow", "red")


def test_monitor_events_triggered():
    """Un pico grande dispara un evento."""
    fs = 2000.0
    t = np.arange(0, 1.0, 1 / fs)
    mon = TorsionalMonitor(units="ftlb", trend_dt=1.0, event_pp=5000.0)
    mon.add_block(4000 + 500 * np.sin(2 * np.pi * 30 * t), rpm=1800, fs=fs)     # tranquilo
    mon.add_block(4000 + 9000 * np.sin(2 * np.pi * 30 * t), rpm=1800, fs=fs)    # sobrecarga
    assert len(mon.events) == 1
    assert mon.events[0].kind == "pp_over"


def test_summary_is_compact():
    """El resumen no lleva la onda cruda — solo histograma + tendencia + eventos."""
    fs = 2000.0; t = np.arange(0, 1.0, 1 / fs)
    mon = TorsionalMonitor(trend_dt=1.0)
    for _ in range(3):
        mon.add_block(1000 + 300 * np.sin(2 * np.pi * 25 * t), rpm=1500, fs=fs)
    s = mon.summary()
    assert s["kind"] == "monitor"
    assert "ranges" in s and "trend" in s and "events" in s
    assert "torque" not in s and "raw" not in s          # nada de onda cruda
