"""
core/modal/live_impact.py — Motor de ENSAYO DE IMPACTO EN VIVO (golpe a golpe)
=============================================================================

Núcleo framework-agnóstico (numpy/scipy) para el ensayo modal con martillo, tal
como lo hace un instrumento de clase mundial y lo pide la norma ISO 7626-5:

  · Cada GOLPE es un registro (fuerza + respuesta).
  · Se aplica **ventana de fuerza** (force window) a la fuerza y **ventana
    exponencial** a la respuesta (ISO 7626-5 §7.4) antes de la FFT.
  · Se ACUMULAN los auto/cross-espectros golpe a golpe → H1 = Gxy/Gxx y
    coherencia γ² = |Gxy|² / (Gxx·Gyy). La coherencia real aparece con ≥2 golpes.
  · Cada golpe se valida en vivo: **doble golpe** (double-hit) y **sobrecarga**
    (overload). El operador ACEPTA o RECHAZA; solo los aceptados promedian.

El generador `synth_impact()` produce golpes sintéticos físicamente correctos
(pulso de fuerza + respuesta multi-DOF con ruido) para DEMO sin hardware — igual
que el simulador de rotodinámica. Con hardware real, el mismo acumulador recibe
el registro disparado por la NI 9234 (IEPE) sin cambiar nada más.

Reutiliza `FRFResult` y `detect_modal_peaks` de core.modal.frf_compute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.modal.frf_compute import FRFResult, ModalPeak, detect_modal_peaks


# =====================================================================
# Modo sintético — física del golpe (para demo sin hardware)
# =====================================================================
@dataclass
class SynthMode:
    """Un modo del espécimen simulado."""
    fn_hz: float
    zeta: float          # razón de amortiguamiento (0..1)
    amplitude: float     # residuo modal (mobilidad relativa)


DEFAULT_SPECIMEN: List[SynthMode] = [
    SynthMode(50.0, 0.010, 1.0),
    SynthMode(120.0, 0.008, 0.7),
    SynthMode(215.0, 0.012, 0.45),
    SynthMode(330.0, 0.015, 0.30),
]


def synth_impact(
    fs: float,
    n: int,
    modes: Sequence[SynthMode] = tuple(DEFAULT_SPECIMEN),
    pretrigger_frac: float = 0.05,
    pulse_ms: float = 0.8,
    force_peak: float = 0.4,
    noise: float = 0.004,
    response_fs: float = 0.6,
    rng: Optional[np.random.Generator] = None,
    double_hit: bool = False,
    overload: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera UN golpe sintético: (force[n], response[n]).

    - force: pulso medio-seno corto tras un pretrigger; opción de doble golpe /
      sobrecarga para probar la validación.
    - response: suma de respuestas impulsivas modales (seno amortiguado) + ruido.
    """
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(n) / fs
    force = np.zeros(n, dtype=float)

    # --- pulso de fuerza (medio seno) ---
    i0 = int(pretrigger_frac * n)
    w = max(2, int(pulse_ms * 1e-3 * fs))               # ancho del pulso en muestras
    amp = force_peak * (0.9 + 0.2 * rng.random())       # variabilidad golpe a golpe
    idx = np.arange(i0, min(i0 + w, n))
    force[idx] = amp * np.sin(np.pi * (idx - i0) / w)
    if double_hit:                                      # segundo golpe accidental
        j0 = i0 + int(2.5 * w)
        jdx = np.arange(j0, min(j0 + w, n))
        force[jdx] += 0.6 * amp * np.sin(np.pi * (jdx - j0) / w)
    if overload:
        force *= 3.0                                    # satura el rango ±1

    # --- respuesta = suma de IRFs modales, excitada por el impulso ---
    resp = np.zeros(n, dtype=float)
    t0 = i0 / fs
    tt = t - t0
    active = tt >= 0
    for m in modes:
        wn = 2.0 * np.pi * m.fn_hz
        wd = wn * np.sqrt(max(1e-9, 1.0 - m.zeta ** 2))
        h = np.zeros(n, dtype=float)
        h[active] = (m.amplitude * amp * np.exp(-m.zeta * wn * tt[active])
                     * np.sin(wd * tt[active]))
        resp += h
    # normaliza la respuesta a un fondo de escala cómodo (golpe bueno < FS)
    pk = float(np.abs(resp).max())
    if pk > 0:
        resp *= (response_fs / pk)
    resp += noise * rng.standard_normal(n)
    if overload:
        resp *= 3.0
    return force, resp


# =====================================================================
# Ventanas de impacto (ISO 7626-5 §7.4)
# =====================================================================
def force_window(n: int, fs: float, pulse_end_idx: int,
                 taper_ms: float = 1.0) -> np.ndarray:
    """Ventana de fuerza: 1 durante el pulso, coseno de bajada, 0 después.
    Elimina el ruido del canal de fuerza tras el impacto."""
    w = np.zeros(n, dtype=float)
    end = int(np.clip(pulse_end_idx, 1, n))
    w[:end] = 1.0
    tp = max(1, int(taper_ms * 1e-3 * fs))
    tail = np.arange(min(tp, n - end))
    if tail.size:
        w[end:end + tail.size] = 0.5 * (1.0 + np.cos(np.pi * tail / tp))
    return w


def exponential_window(n: int, fs: float, decay_frac: float = 0.01) -> np.ndarray:
    """Ventana exponencial para la respuesta: e^{-t/τ}. Fuerza el decaimiento a
    ~decay_frac al final del registro (evita fuga por truncamiento en modos poco
    amortiguados). Añade amortiguamiento artificial conocido (se puede corregir)."""
    t = np.arange(n) / fs
    T = n / fs
    tau = -T / np.log(max(1e-6, decay_frac))
    return np.exp(-t / tau)


def detect_pulse_end(force: np.ndarray, thresh_frac: float = 0.02) -> int:
    """Índice donde termina el pulso de fuerza (cae por debajo de thresh·pico)."""
    a = np.abs(force)
    pk = float(a.max()) if a.size else 0.0
    if pk <= 0:
        return len(force)
    above = np.where(a >= thresh_frac * pk)[0]
    if above.size == 0:
        return len(force)
    return int(above[-1]) + 1


# =====================================================================
# Validación de calidad del golpe (en vivo)
# =====================================================================
@dataclass
class HitQuality:
    peak_force: float
    overload: bool
    double_hit: bool
    n_force_peaks: int


def assess_hit(force: np.ndarray, response: np.ndarray, fs: float,
               overload_level: float = 0.98,
               double_hit_ratio: float = 0.5) -> HitQuality:
    """Evalúa un golpe: sobrecarga (clipping) y doble golpe (2º pico de fuerza)."""
    from scipy.signal import find_peaks
    af = np.abs(force)
    pk = float(af.max()) if af.size else 0.0
    over = bool(pk >= overload_level) or bool(np.abs(response).max() >= overload_level)
    # doble golpe: ≥2 picos prominentes en la fuerza
    peaks, props = find_peaks(af, height=double_hit_ratio * pk, distance=max(1, int(2e-4 * fs)))
    n_peaks = int(peaks.size)
    return HitQuality(peak_force=pk, overload=over,
                      double_hit=(n_peaks >= 2), n_force_peaks=n_peaks)


# =====================================================================
# Acumulador de FRF golpe a golpe (H1 + coherencia)
# =====================================================================
class FRFAccumulator:
    """Acumula auto/cross-espectros de los golpes ACEPTADOS y entrega H1 + γ².

    H1  = Gxy / Gxx
    γ²  = |Gxy|² / (Gxx · Gyy)     (real, aparece con ≥2 promedios)
    """

    def __init__(self, fs: float, n: int,
                 use_force_window: bool = True,
                 use_exp_window: bool = True,
                 exp_decay_frac: float = 0.01):
        self.fs = float(fs)
        self.n = int(n)
        self.use_force_window = use_force_window
        self.use_exp_window = use_exp_window
        self.exp_decay_frac = exp_decay_frac
        self.freqs = np.fft.rfftfreq(self.n, d=1.0 / self.fs)
        self._m = self.freqs.size
        self.reset()

    def reset(self) -> None:
        self.Gxx = np.zeros(self._m, dtype=float)
        self.Gyy = np.zeros(self._m, dtype=float)
        self.Gxy = np.zeros(self._m, dtype=complex)
        self.count = 0

    def _spectra(self, force: np.ndarray, response: np.ndarray):
        x = np.asarray(force, float).copy()
        y = np.asarray(response, float).copy()
        if x.size != self.n:                         # re-encajar por seguridad
            x = np.resize(x, self.n); y = np.resize(y, self.n)
        if self.use_force_window:
            x = x * force_window(self.n, self.fs, detect_pulse_end(x))
        if self.use_exp_window:
            y = y * exponential_window(self.n, self.fs, self.exp_decay_frac)
        X = np.fft.rfft(x)
        Y = np.fft.rfft(y)
        return X, Y

    def add(self, force: np.ndarray, response: np.ndarray) -> None:
        """Suma un golpe ACEPTADO al promedio."""
        X, Y = self._spectra(force, response)
        self.Gxx += (X.conj() * X).real
        self.Gyy += (Y.conj() * Y).real
        self.Gxy += X.conj() * Y
        self.count += 1

    def preview(self, force: np.ndarray, response: np.ndarray) -> FRFResult:
        """FRF del golpe ACTUAL combinado con lo ya acumulado, SIN comprometerlo
        (para el preview en vivo antes de aceptar/rechazar)."""
        X, Y = self._spectra(force, response)
        gxx = self.Gxx + (X.conj() * X).real
        gyy = self.Gyy + (Y.conj() * Y).real
        gxy = self.Gxy + X.conj() * Y
        return self._build(gxx, gyy, gxy, self.count + 1)

    def result(self) -> Optional[FRFResult]:
        """FRF promediada de los golpes aceptados (None si no hay ninguno)."""
        if self.count == 0:
            return None
        return self._build(self.Gxx, self.Gyy, self.Gxy, self.count)

    def _build(self, gxx, gyy, gxy, count) -> FRFResult:
        gxx_safe = np.where(gxx > 1e-30, gxx, 1e-30)
        gyy_safe = np.where(gyy > 1e-30, gyy, 1e-30)
        H1 = gxy / gxx_safe
        coh = (np.abs(gxy) ** 2) / (gxx_safe * gyy_safe)
        coh = np.clip(coh, 0.0, 1.0)
        return FRFResult(frequencies_hz=self.freqs, frf_complex=H1, coherence=coh,
                         estimator="H1", n_averages=int(count),
                         window="force+exp" if self.use_exp_window else "force")


def modes_from_frf(frf: FRFResult, fmin: float = 5.0,
                   fmax: Optional[float] = None) -> List[ModalPeak]:
    """Identifica modos (fn, damping, coherencia) desde la FRF promediada."""
    return detect_modal_peaks(frf.frequencies_hz, frf.magnitude,
                              coherence=frf.coherence, f_min_hz=fmin, f_max_hz=fmax)
