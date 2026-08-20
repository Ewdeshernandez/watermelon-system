"""
core/remote_monitoring/transient.py — Captura transitoria (bode/cascade)
=======================================================================

Motor que, durante arranque/parada, captura un "punto de velocidad" cada
Δrpm y calcula por canal: espectro + vector 1X (amp/fase). De esa colección
ordenada por velocidad salen:

  · Bode    — 1X amplitud & fase vs rpm.
  · Cascade / Waterfall — espectro vs rpm (matriz).

Es el concepto de "transient data collection" de System1 (taller T00336
Tarea 4), que es lo que hace posible bode/cascade REALES (no se pueden
armar de estado estacionario).

Reusa keyphasor.one_x_vector para el vector 1X. La velocidad viene del
keyphasor (agent.estimate_rpm).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.remote_monitoring.keyphasor import one_x_vector, detect_keyphasor


def _sync_1x(eu, pulses, order=1):
    """Vector de orden `order` por ORDER TRACKING: DFT en el dominio del ÁNGULO
    (los pulsos de keyphasor se mapean a múltiplos de 2π, interpolación lineal
    entre ellos). Devuelve (amp 0-pk, fase deg REFERIDA al keyphasor) — limpio y
    estable aun con la rpm cambiando en la ventana. None si no hay pulsos útiles."""
    if pulses is None or len(pulses) < 2:
        return None
    i0, i1 = int(pulses[0]), int(pulses[-1])
    if i1 - i0 < 8:
        return None
    idx = np.arange(i0, i1 + 1)
    ang = np.interp(idx, np.asarray(pulses, float), 2.0 * np.pi * np.arange(len(pulses)))
    seg = np.asarray(eu[i0:i1 + 1], float)
    seg = seg - np.mean(seg)
    w = np.hanning(len(seg))
    gain = np.sum(w) / len(seg)
    if gain <= 0:
        return None
    c = np.sum(seg * w * np.exp(-1j * order * ang)) / len(seg) / gain
    return 2.0 * float(np.abs(c)), float(np.degrees(np.angle(c)) % 360.0)


def _order_vector(eu, fs, f_target, freqs, mag, ref_sample=0, tol_frac=0.04):
    """Vector 1X (amp 0-pk, fase deg) leyendo el PICO real del espectro cerca de
    f_target y calculando el vector a esa frecuencia. Robusto cuando el rpm
    estimado no coincide exacto con la frecuencia real (transitorio) → evita el
    colapso de la proyección síncrona directa a f_target."""
    if f_target <= 0 or len(freqs) < 2:
        return one_x_vector(eu, fs, f_target, ref_sample=ref_sample)
    df = float(freqs[1] - freqs[0])
    tol = max(3.0 * df, tol_frac * f_target)
    band = np.abs(freqs - f_target) <= tol
    n = min(len(freqs), len(mag))
    band = band[:n]
    if band.any():
        idx = np.where(band)[0]
        j = int(idx[int(np.argmax(mag[:n][idx]))])
        f_peak = float(freqs[j])
        amp_peak = float(mag[j])          # magnitud del pico (robusta al barrido)
    else:
        f_peak, amp_peak = float(f_target), None
    # Amplitud = pico del espectro; fase = vector síncrono a esa frecuencia,
    # REFERIDA al keyphasor (ref_sample) → fase física estable, no ruido.
    a_sync, phase = one_x_vector(eu, fs, f_peak, ref_sample=ref_sample)
    return (amp_peak if amp_peak is not None else a_sync), phase


@dataclass
class TransientConfig:
    delta_rpm: float = 8.0         # capturar un punto cada Δrpm de cambio (denso)
    min_rpm: float = 100.0         # no capturar por debajo (ruido de arranque)
    capture_samples: int = 4096    # ventana FFT por punto (líneas consistentes)
    fmax_hz: float = 500.0         # tope de frecuencia guardado (acota RAM)
    max_samples: int = 200         # máximo de puntos de velocidad en memoria


@dataclass
class SpeedSample:
    rpm: float
    spectra: Dict[str, np.ndarray] = field(default_factory=dict)      # canal → magnitud
    vect1x: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # canal → (amp, fase)


class TransientCapture:
    """Acumula puntos de velocidad durante el transitorio."""

    def __init__(self, config: Optional[TransientConfig] = None) -> None:
        self.config = config or TransientConfig()
        self.samples: List[SpeedSample] = []
        self.freqs: Optional[np.ndarray] = None
        self._last_rpm: Optional[float] = None

    def reset(self) -> None:
        self.samples.clear()
        self.freqs = None
        self._last_rpm = None

    def _spectrum(self, x: np.ndarray, fs: float):
        x = x - np.mean(x)
        w = np.hanning(len(x))
        mag = np.abs(np.fft.rfft(x * w)) / (np.sum(w) / 2)
        freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
        if self.config.fmax_hz > 0:
            keep = freqs <= self.config.fmax_hz
            return freqs[keep], mag[keep]
        return freqs, mag

    def feed(self, snap: np.ndarray, rpm: Optional[float], fs: float,
             vib_channels: List[Tuple[int, object]], kph_idx: Optional[int] = None) -> bool:
        """Evalúa si capturar un punto de velocidad. Devuelve True si capturó.

        vib_channels: lista de (índice_en_snap, ChannelConfig) SOLO de vibración.
        kph_idx: fila del keyphasor en snap → referencia de fase (t=0). Sin él la
        fase 1X sale como ruido (referida al inicio arbitrario de la ventana).
        El snapshot viene en Volts; se escala a EU con la sensitivity del canal.
        """
        if rpm is None or rpm < self.config.min_rpm:
            return False
        if snap.shape[1] == 0:
            return False
        # ¿cambió lo suficiente la velocidad?
        if self._last_rpm is not None and abs(rpm - self._last_rpm) < self.config.delta_rpm:
            return False

        win = min(self.config.capture_samples, snap.shape[1])
        f1 = rpm / 60.0
        # Pulsos del keyphasor en la ventana → order tracking (fase física).
        ref, pulses = 0, None
        if kph_idx is not None and 0 <= kph_idx < snap.shape[0]:
            kr = detect_keyphasor(snap[kph_idx, -win:], fs)
            if kr.ref_sample is not None:
                ref = int(kr.ref_sample)
            pulses = kr.pulse_sample_indices
        sample = SpeedSample(rpm=float(rpm))
        for idx, ch in vib_channels:
            sens = float(getattr(ch, "sensitivity_mv_per_eu", 0.0) or 0.0)
            raw = snap[idx, -win:]
            eu = raw * 1000.0 / sens if sens > 0 else raw
            freqs, mag = self._spectrum(eu, fs)
            # 1X por ORDER TRACKING (amp+fase limpias, referidas al keyphasor);
            # si no hay pulsos, cae al pico del espectro (evita el colapso síncrono).
            sync = _sync_1x(eu, pulses, order=1)
            if sync is not None:
                amp, phase = sync
            else:
                amp, phase = _order_vector(eu, fs, f1, freqs, mag, ref_sample=ref)
            sample.vect1x[ch.name] = (amp, phase)
            if self.freqs is None:
                self.freqs = freqs
            # alinear longitud si fs/ventana variaran (defensivo)
            if len(mag) != len(self.freqs):
                m = min(len(mag), len(self.freqs))
                mag = mag[:m]
            sample.spectra[ch.name] = mag

        self.samples.append(sample)
        self._last_rpm = float(rpm)
        if len(self.samples) > self.config.max_samples:
            self.samples = self.samples[-self.config.max_samples:]
        return True

    # --- lecturas ---
    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def _sorted(self) -> List[SpeedSample]:
        return sorted(self.samples, key=lambda s: s.rpm)

    def bode(self, channel: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rpm, amp1x, fase1x) ordenado por rpm para un canal."""
        pts = [(s.rpm, *s.vect1x.get(channel, (np.nan, np.nan))) for s in self._sorted()
               if channel in s.vect1x]
        if not pts:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        arr = np.array(pts, dtype=float)
        return arr[:, 0], arr[:, 1], arr[:, 2]

    def cascade(self, channel: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rpm[n], freqs[m], magnitudes[n,m]) ordenado por rpm."""
        pts = [s for s in self._sorted() if channel in s.spectra]
        if not pts or self.freqs is None:
            return np.zeros(0), np.zeros(0), np.zeros((0, 0))
        m = len(self.freqs)
        rpms = np.array([s.rpm for s in pts], dtype=float)
        mat = np.zeros((len(pts), m), dtype=float)
        for i, s in enumerate(pts):
            v = s.spectra[channel]
            mat[i, :min(m, len(v))] = v[:m]
        return rpms, self.freqs, mat
