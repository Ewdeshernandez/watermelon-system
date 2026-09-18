"""
core/torsional/analysis.py — Motor de análisis torsional (compartido campo+web)
===============================================================================

Análisis del par medido con el TorqueTrak 10K. Numpy puro, sin dependencias de
hardware ni de Streamlit: lo usan tanto la app de campo (`native/`) como el
módulo web (`pages/`).

Entradas típicas: par en unidades de ingeniería (N·m/ft-lb, ya convertido con
`core.torsional.scaling.voltage_to_torque`), fs, y rpm (constante) o el canal
keyphasor (tacómetro) para velocidad instantánea.

Funciones
---------
· torque_metrics        — media, pico-pico, RMS, cresta, rizado %.
· torque_spectrum       — FFT amplitud-correcta del par.
· order_amplitude(s)    — amplitud y fase de una/varias órdenes (proyección DFT).
· keyphasor_to_rpm      — rpm instantánea desde los pulsos del tacómetro.
· order_tracking        — amplitud por orden vs rpm (runup/coastdown → Campbell).
· rainflow_cycles       — conteo de ciclos ASTM E1049 (fatiga torsional).
· fatigue_ranges        — histograma de rangos de par para vida a fatiga.

Para Campbell/waterfall y TSA de figura se reusa `core.modal.campbell` y
`core.tsa` en la capa de UI; aquí queda la extracción numérica.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------
# Métricas escalares del par
# -----------------------------------------------------------------
@dataclass
class TorqueMetrics:
    """Métricas de una traza de par."""
    mean: float          # par medio (estático)
    peak_to_peak: float  # pico-pico total
    rms: float           # RMS de la traza completa
    crest_factor: float  # pico / RMS (de la componente dinámica)
    ripple_pct: float    # rizado = (pp_dinámico / |media|)·100


def torque_metrics(torque: np.ndarray) -> TorqueMetrics:
    """Métricas básicas de la señal de par."""
    x = np.asarray(torque, dtype=float)
    if x.size == 0:
        raise ValueError("torque vacío")
    mean = float(np.mean(x))
    pp = float(np.ptp(x))
    rms = float(np.sqrt(np.mean(x ** 2)))
    dyn = x - mean
    peak_dyn = float(np.max(np.abs(dyn)))
    rms_dyn = float(np.sqrt(np.mean(dyn ** 2)))
    crest = peak_dyn / rms_dyn if rms_dyn > 0 else 0.0
    ripple = (pp / abs(mean) * 100.0) if abs(mean) > 1e-12 else float("inf")
    return TorqueMetrics(mean=mean, peak_to_peak=pp, rms=rms,
                         crest_factor=crest, ripple_pct=ripple)


# -----------------------------------------------------------------
# Espectro de par
# -----------------------------------------------------------------
def torque_spectrum(torque: np.ndarray, fs: float,
                    remove_dc: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Espectro de amplitud del par (ventana Hann, amplitud-correcta).

    Returns:
        (freqs [Hz], amp [EU pico]) — amp de un tono A·sin ≈ A.
    """
    x = np.asarray(torque, dtype=float)
    if remove_dc:
        x = x - np.mean(x)
    n = x.size
    win = np.hanning(n)
    spec = np.fft.rfft(x * win)
    amp = np.abs(spec) / np.sum(win) * 2.0
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    return freqs, amp


# -----------------------------------------------------------------
# Órdenes (proyección DFT a la frecuencia exacta k·f1)
# -----------------------------------------------------------------
def order_amplitude(torque: np.ndarray, fs: float, rpm: float,
                    order: float) -> Tuple[float, float]:
    """
    Amplitud y fase de una orden por proyección DFT a f = order·rpm/60.

    Más exacto que tomar el bin del FFT: proyecta sobre e^{-j2πft} a la
    frecuencia exacta de la orden, sin depender de la resolución de bin.

    Returns:
        (amplitud [EU pico], fase [grados]).
    """
    x = np.asarray(torque, dtype=float)
    x = x - np.mean(x)
    n = x.size
    t = np.arange(n) / fs
    f = order * rpm / 60.0
    win = np.hanning(n)
    proj = np.sum(x * win * np.exp(-1j * 2.0 * np.pi * f * t))
    amp = 2.0 * np.abs(proj) / np.sum(win)
    phase = np.degrees(np.angle(proj))
    return float(amp), float(phase)


def order_amplitudes(torque: np.ndarray, fs: float, rpm: float,
                     orders: Sequence[float] = (1, 2, 3, 4, 5)
                     ) -> Dict[float, Tuple[float, float]]:
    """Amplitud y fase de varias órdenes → {orden: (amp, fase_deg)}."""
    return {float(o): order_amplitude(torque, fs, rpm, o) for o in orders}


# -----------------------------------------------------------------
# Keyphasor (tacómetro) → rpm instantánea
# -----------------------------------------------------------------
def keyphasor_to_rpm(keyphasor: np.ndarray, fs: float,
                     threshold: float = -1.0, pulses_per_rev: int = 1
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Velocidad instantánea desde el canal keyphasor (pulsos negativos once-per-rev).

    Detecta flancos de bajada (cruce descendente de `threshold`) y calcula
    rpm = 60 / (Δt · pulses_per_rev) entre pulsos sucesivos.

    Returns:
        (t_rev [s], rpm) — un valor por intervalo entre pulsos (len = nº pulsos−1).
        Arrays vacíos si hay menos de 2 pulsos.
    """
    k = np.asarray(keyphasor, dtype=float)
    below = k < threshold
    edges = np.where(below[1:] & ~below[:-1])[0] + 1   # índices de flanco de bajada
    if edges.size < 2:
        return np.array([]), np.array([])
    t_edges = edges / fs
    dt = np.diff(t_edges)
    rpm = 60.0 / (dt * max(1, pulses_per_rev))
    t_mid = 0.5 * (t_edges[1:] + t_edges[:-1])
    return t_mid, rpm


# -----------------------------------------------------------------
# Order tracking vs rpm (runup / coastdown)
# -----------------------------------------------------------------
@dataclass
class OrderTrack:
    """Amplitud de una orden a lo largo de la velocidad."""
    order: float
    rpm: np.ndarray        # rpm media de cada segmento
    amplitude: np.ndarray  # amplitud [EU] de la orden en cada segmento


def order_tracking(torque: np.ndarray, fs: float, rpm: np.ndarray,
                   orders: Sequence[float] = (1, 2, 3),
                   n_segments: int = 20) -> List[OrderTrack]:
    """
    Amplitud por orden vs rpm, segmentando la traza (para Campbell/runup).

    Args:
        torque: Traza de par [EU].
        fs: Frecuencia de muestreo [Hz].
        rpm: rpm por muestra (array, misma longitud que torque) o escalar.
        orders: Órdenes a seguir.
        n_segments: Número de segmentos temporales.

    Returns:
        Lista de OrderTrack (uno por orden), ordenados por rpm creciente.
    """
    x = np.asarray(torque, dtype=float)
    n = x.size
    rpm_arr = np.full(n, float(rpm)) if np.isscalar(rpm) else np.asarray(rpm, dtype=float)
    if rpm_arr.size != n:
        raise ValueError("rpm debe ser escalar o del mismo largo que torque")

    seg_len = n // n_segments
    if seg_len < 8:
        raise ValueError("traza demasiado corta para n_segments pedidos")

    tracks = {float(o): ([], []) for o in orders}
    for s in range(n_segments):
        a, b = s * seg_len, (s + 1) * seg_len
        seg = x[a:b]
        seg_rpm = float(np.mean(rpm_arr[a:b]))
        for o in orders:
            amp, _ = order_amplitude(seg, fs, seg_rpm, o)
            tracks[float(o)][0].append(seg_rpm)
            tracks[float(o)][1].append(amp)

    result = []
    for o in orders:
        rr = np.asarray(tracks[float(o)][0])
        aa = np.asarray(tracks[float(o)][1])
        srt = np.argsort(rr)
        result.append(OrderTrack(order=float(o), rpm=rr[srt], amplitude=aa[srt]))
    return result


# -----------------------------------------------------------------
# Rainflow (ASTM E1049) — fatiga torsional
# -----------------------------------------------------------------
def _reversals(series: Sequence[float]) -> List[float]:
    """Puntos de retorno (turning points) de una serie, incluyendo extremos."""
    it = iter(series)
    try:
        x_last = float(next(it))
        x = float(next(it))
    except StopIteration:
        return [float(v) for v in series]
    d_last = x - x_last
    out = [x_last]
    for nxt in it:
        x_next = float(nxt)
        if x_next == x:
            continue
        d_next = x_next - x
        if d_last * d_next < 0:
            out.append(x)
        x_last, x = x, x_next
        d_last = d_next
    out.append(x)
    return out


@dataclass
class RainflowCycle:
    """Un ciclo (o medio ciclo) rainflow."""
    range: float   # rango pico-pico del ciclo [EU]
    mean: float    # par medio del ciclo [EU]
    count: float   # 1.0 ciclo completo, 0.5 medio ciclo


def rainflow_cycles(torque: Sequence[float]) -> List[RainflowCycle]:
    """
    Conteo de ciclos rainflow ASTM E1049 (para análisis de fatiga torsional).

    Extrae los puntos de retorno y aplica la regla de 3 puntos con residuo.

    Returns:
        Lista de RainflowCycle (rango, media, conteo 1.0/0.5).
    """
    points: deque = deque()
    cycles: List[RainflowCycle] = []
    for x in _reversals(torque):
        points.append(x)
        while len(points) >= 3:
            x1, x2, x3 = points[-3], points[-2], points[-1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)
            if X < Y:
                break
            if len(points) == 3:
                # Y contiene el punto de arranque → medio ciclo
                cycles.append(RainflowCycle(range=Y, mean=0.5 * (x1 + x2), count=0.5))
                points.popleft()
            else:
                cycles.append(RainflowCycle(range=Y, mean=0.5 * (x1 + x2), count=1.0))
                last = points.pop()
                points.pop()   # x2
                points.pop()   # x1
                points.append(last)
    # Residuo → medios ciclos
    while len(points) > 1:
        x1 = points[0]
        x2 = points[1]
        cycles.append(RainflowCycle(range=abs(x2 - x1), mean=0.5 * (x1 + x2), count=0.5))
        points.popleft()
    return cycles


def fatigue_ranges(torque: Sequence[float]) -> List[Tuple[float, float]]:
    """
    Histograma de rangos rainflow → [(rango, conteo)] agregado y ordenado por rango.

    Útil para curvas S-N / acumulación de daño (Miner) en el eje.
    """
    agg: Dict[float, float] = {}
    for c in rainflow_cycles(torque):
        key = round(c.range, 9)
        agg[key] = agg.get(key, 0.0) + c.count
    return sorted(agg.items(), key=lambda kv: kv[0])


# -----------------------------------------------------------------
# Vida a fatiga del eje — Goodman (esfuerzo medio) + Miner (daño)
# -----------------------------------------------------------------
# Base normativa/ingenieril:
#   · Rainflow ASTM E1049 (conteo de ciclos, arriba).
#   · Esfuerzo cortante torsional: τ = 16·T·Do / (π·(Do⁴−Di⁴)).
#   · Endurance en cortante por energía de distorsión (von Mises):
#       Se' = 0.5·Sut (acero, Sut<1400 MPa) → Sse = 0.577·Se'·kf   (kf = derating de campo).
#     Resistencia última en cortante: Ssu ≈ 0.67·Sut.
#   · Corrección de esfuerzo medio: Goodman (τar = τa / (1 − τm/Ssu)).
#   · S-N (Basquin) en cortante entre 1e3 y 1e6 ciclos (Shigley/ASME B106.1M).
#   · Daño acumulado: Palmgren–Miner (Σ n_i/Nf_i). Contexto: API 684.

_NM_TO_LBIN = 8.8507457676   # 1 N·m  = 8.8507 lb·in
_FTLB_TO_LBIN = 12.0         # 1 ft-lb = 12 lb·in
_PSI_TO_MPA = 1.0 / 145.0377


@dataclass
class FatigueLife:
    """Diagnóstico de vida a fatiga torsional del eje."""
    status: str              # "green" | "yellow" | "red"
    label_es: str
    label_en: str
    safety_factor: float     # SF contra el límite de fatiga (Goodman, vida infinita)
    tau_alt_max_psi: float    # mayor esfuerzo cortante alternante equivalente (τar) [psi]
    tau_mean_psi: float       # esfuerzo cortante medio [psi]
    sse_psi: float           # límite de fatiga en cortante corregido [psi]
    ssu_psi: float           # resistencia última en cortante [psi]
    damage_window: float     # daño de Miner en la ventana medida (1.0 = falla)
    life_hours: float        # vida estimada [h] (inf si SF≥1); asume operación continua
    n_cycles: float          # ciclos contados en la ventana
    infinite: bool           # True si vida infinita (bajo el límite de fatiga)


def _torque_to_lbin(value: float, units: str) -> float:
    return value * (_NM_TO_LBIN if units == "nm" else _FTLB_TO_LBIN)


def shaft_torsional_fatigue(
    cycles: Sequence[RainflowCycle],
    outer_diameter_in: float,
    inner_diameter_in: float,
    ultimate_strength_psi: float,
    torque_units: str = "ftlb",
    window_seconds: float = 8.0,
    derating_kf: float = 0.70,
    design_safety_factor: float = 2.0,
) -> FatigueLife:
    """
    Vida a fatiga torsional del eje a partir de los ciclos rainflow del par.

    cycles: salida de `rainflow_cycles` (rango/media en unidades de par).
    outer/inner_diameter_in: geometría del eje en la galga [pulgadas].
    ultimate_strength_psi: resistencia última Sut del material del eje [psi].
    torque_units: "nm" o "ftlb" (unidades de los ciclos).
    window_seconds: duración de la captura (para proyectar vida en horas).
    derating_kf: factor de reducción de campo (superficie/tamaño/confiabilidad).
    design_safety_factor: SF objetivo para "vida infinita" (verde). API/AGMA ~1.5–2.

    Semáforo:
      VERDE  → SF ≥ design_safety_factor  (bajo el límite de fatiga, margen amplio).
      AMARILLO → 1.0 ≤ SF < design_safety_factor (vida finita larga; vigilar).
      ROJO   → SF < 1.0 (acumula daño; se estima vida en horas).
    """
    do = float(outer_diameter_in); di = float(inner_diameter_in)
    sut = max(float(ultimate_strength_psi), 1.0)
    # Módulo de sección polar → esfuerzo cortante por par (τ = T / Zp).
    zp = np.pi * (do ** 4 - di ** 4) / (16.0 * do)     # in³  (T en lb·in → τ en psi)
    ssu = 0.67 * sut                                    # resistencia última en cortante
    sse = 0.577 * 0.5 * sut * float(derating_kf)        # límite de fatiga en cortante corregido
    # Punto de 1e3 ciclos en cortante (Basquin): 0.577 · 0.9 · Sut.
    ss1e3 = 0.577 * 0.9 * sut
    # Basquin  Sf = a·N^b  entre (1e3, ss1e3) y (1e6, sse).
    b = -(np.log10(ss1e3 / sse)) / 3.0                  # pendiente (negativa)
    a = ss1e3 / (1e3 ** b)

    n_total = 0.0
    damage = 0.0
    tau_ar_max = 0.0
    tau_m_at_max = 0.0
    for c in cycles:
        n_total += c.count
        t_amp = _torque_to_lbin(c.range, torque_units) * 0.5   # amplitud de par
        t_mean = _torque_to_lbin(c.mean, torque_units)
        tau_a = abs(t_amp) / zp
        tau_m = abs(t_mean) / zp
        if tau_a <= 0.0:
            continue
        # Goodman: alternante equivalente totalmente reversible.
        denom = max(1.0 - tau_m / ssu, 1e-3)
        tau_ar = tau_a / denom
        if tau_ar > tau_ar_max:
            tau_ar_max = tau_ar; tau_m_at_max = tau_m
        # Daño de Miner (solo por encima del límite de fatiga).
        if tau_ar > sse:
            nf = (tau_ar / a) ** (1.0 / b)              # ciclos a la falla
            nf = max(nf, 1.0)
            damage += c.count / nf

    sf = (sse / tau_ar_max) if tau_ar_max > 0 else float("inf")
    infinite = sf >= 1.0 or damage <= 0.0
    if infinite:
        life_h = float("inf")
    else:
        cps = n_total / max(window_seconds, 1e-6)        # ciclos por segundo
        d_rate = damage / max(window_seconds, 1e-6)      # daño por segundo
        life_h = (1.0 / d_rate) / 3600.0 if d_rate > 0 else float("inf")

    if sf >= design_safety_factor:
        status = "green"
        label_es = "Vida infinita — eje seguro"; label_en = "Infinite life — shaft safe"
    elif sf >= 1.0:
        status = "yellow"
        label_es = "Vida finita larga — vigilar"; label_en = "Long finite life — monitor"
    else:
        status = "red"
        label_es = "Riesgo de fatiga — vida limitada"; label_en = "Fatigue risk — limited life"

    return FatigueLife(
        status=status, label_es=label_es, label_en=label_en,
        safety_factor=float(sf), tau_alt_max_psi=float(tau_ar_max),
        tau_mean_psi=float(tau_m_at_max), sse_psi=float(sse), ssu_psi=float(ssu),
        damage_window=float(damage), life_hours=float(life_h),
        n_cycles=float(n_total), infinite=bool(infinite),
    )
