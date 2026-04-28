"""
core.rotordynamics
==================

Primitivas matemáticas para análisis rotodinámico siguiendo:

- API 684 (Tutorial on Rotor Dynamics): factor de amplificación Q,
  márgenes de separación de velocidades críticas.
- ISO 20816-2:2017 (Mechanical vibration of large turbines and
  generators >40 MW): clasificación de severidad por zonas A/B/C/D
  para desplazamiento relativo del eje (proximity probes) y vibración
  absoluta de carcasa.

Este módulo es PURO: sin Streamlit, sin I/O, sin estado. Recibe arrays
numéricos y devuelve dataclasses. Es completamente testeable.

Conceptos clave implementados:

1. Detección de velocidades críticas a partir de datos Bode (amp+phase
   vs RPM): pico de amplitud + cambio de fase consistente alrededor
   del pico (firma rotodinámica clásica).

2. Factor de amplificación Q (Amplification Factor, AF en API 684):
   Q = N_c / (N2 - N1), donde N_c es la velocidad crítica y N1, N2
   son las velocidades a -3 dB del pico (es decir, donde la amplitud
   es 1/sqrt(2) ≈ 0.707 del valor máximo).

3. Margen de separación API 684 (SM, Separation Margin): distancia
   en porcentaje entre la velocidad de operación y la velocidad
   crítica. El margen mínimo requerido es función de Q.

4. Zonas ISO 20816-2: A (good), B (acceptable), C (limited operation),
   D (immediate action). Aplicable a turbinas y generadores >40 MW.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np


# =============================================================
# TIPOS / DATACLASSES
# =============================================================

@dataclass
class CriticalSpeed:
    """
    Una velocidad crítica detectada en datos Bode.

    Campos:
        rpm: velocidad de la crítica
        frequency_hz: rpm / 60
        amp_peak: amplitud máxima en el pico (unidades del CSV, típicamente µm pp o mils pp)
        phase_at_peak: fase en grados en el pico (0-360)
        phase_change_deg: cambio total de fase a través de la zona del pico
        q_factor: factor de amplificación (Q = fc / Δf at -3dB)
        fwhm_rpm: ancho del pico a -3 dB en rpm (Δf en API 684)
        confidence: 0-1, qué tan firme es la detección
        n1_rpm: velocidad inferior a -3 dB
        n2_rpm: velocidad superior a -3 dB
    """
    rpm: float
    frequency_hz: float
    amp_peak: float
    phase_at_peak: float
    phase_change_deg: float
    q_factor: float
    fwhm_rpm: float
    confidence: float
    n1_rpm: float
    n2_rpm: float


@dataclass
class API684Margin:
    """
    Evaluación del margen de separación API 684 para una crítica
    respecto a una velocidad de operación.

    Campos:
        critical_rpm, operating_rpm: entradas
        q_factor: factor Q de la crítica
        actual_margin_pct: margen real (|op - crit| / op × 100)
        required_margin_pct: margen mínimo recomendado por API 684 dado Q
        compliant: True si actual_margin_pct >= required_margin_pct
        zone: "SAFE" / "MARGINAL" / "DANGER"
        narrative: texto descriptivo
    """
    critical_rpm: float
    operating_rpm: float
    q_factor: float
    actual_margin_pct: float
    required_margin_pct: float
    compliant: bool
    zone: Literal["SAFE", "MARGINAL", "DANGER"]
    narrative: str


@dataclass
class ISO20816Zone:
    """
    Clasificación ISO 20816-2 de severidad de vibración.

    Campos:
        amplitude: valor evaluado (en las unidades correspondientes)
        unit: "um_pp" (micrómetros pp, shaft displacement) o "mm_s_rms"
            (mm/s RMS, casing velocity)
        zone: "A", "B", "C", "D"
        zone_description: descripción del estado
        boundary_AB, boundary_BC, boundary_CD: umbrales aplicados
        machine_group: "group1" o "group2" según ISO 20816-2
        operating_speed_rpm: velocidad operativa de referencia
    """
    amplitude: float
    unit: str
    zone: Literal["A", "B", "C", "D"]
    zone_description: str
    boundary_AB: float
    boundary_BC: float
    boundary_CD: float
    machine_group: str
    operating_speed_rpm: float


# =============================================================
# UTILIDADES INTERNAS
# =============================================================

def _rolling_circular_mean_deg(phase_deg: np.ndarray, window: int) -> np.ndarray:
    """Smoothing circular en grados con ventana rectangular centrada."""
    if window <= 1:
        return phase_deg.astype(float).copy()

    rad = np.deg2rad(phase_deg.astype(float) % 360.0)
    cos_w = np.cos(rad)
    sin_w = np.sin(rad)

    pad = window // 2
    cos_pad = np.pad(cos_w, pad, mode="edge")
    sin_pad = np.pad(sin_w, pad, mode="edge")

    kernel = np.ones(window) / window
    cos_smooth = np.convolve(cos_pad, kernel, mode="valid")
    sin_smooth = np.convolve(sin_pad, kernel, mode="valid")

    if cos_smooth.size > phase_deg.size:
        cos_smooth = cos_smooth[: phase_deg.size]
        sin_smooth = sin_smooth[: phase_deg.size]
    elif cos_smooth.size < phase_deg.size:
        diff = phase_deg.size - cos_smooth.size
        cos_smooth = np.concatenate([cos_smooth, np.repeat(cos_smooth[-1], diff)])
        sin_smooth = np.concatenate([sin_smooth, np.repeat(sin_smooth[-1], diff)])

    out = np.rad2deg(np.arctan2(sin_smooth, cos_smooth))
    return (out + 360.0) % 360.0


def _phase_change_local(phase_deg: np.ndarray, idx: int, half_window: int) -> float:
    """
    Calcula el cambio total de fase en una ventana centrada en idx.
    Retorna el módulo del rango envuelto en [-180, 180].
    """
    n = phase_deg.size
    lo = max(0, idx - half_window)
    hi = min(n, idx + half_window + 1)

    segment = phase_deg[lo:hi]
    if segment.size < 2:
        return 0.0

    # Unwrap local para evitar saltos artificiales en 0/360
    rad = np.deg2rad(segment.astype(float))
    unwrapped = np.unwrap(rad)
    return float(np.abs(np.rad2deg(unwrapped[-1] - unwrapped[0])))


def _find_peaks_simple(
    y: np.ndarray,
    min_prominence: float = 0.0,
    min_distance: int = 3,
) -> List[int]:
    """
    Detección de picos sin scipy: punto i es pico si y[i] > y[i±1] y la
    prominencia local supera el umbral. Min_distance evita picos muy
    pegados eligiendo el más alto.
    """
    n = y.size
    if n < 3:
        return []

    candidates: List[int] = []
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            # prominencia local: altura sobre el mínimo en una vecindad
            lo = max(0, i - 10)
            hi = min(n, i + 11)
            local_min = float(np.min(y[lo:hi]))
            prominence = float(y[i] - local_min)
            if prominence >= min_prominence:
                candidates.append(i)

    if not candidates:
        return []

    # Filtrar por min_distance: si dos picos están muy cerca, conservar el más alto
    candidates.sort(key=lambda i: -y[i])
    accepted: List[int] = []
    for c in candidates:
        if all(abs(c - a) >= min_distance for a in accepted):
            accepted.append(c)
    accepted.sort()
    return accepted


def _compute_fwhm_at_minus3db(
    rpm: np.ndarray,
    amp: np.ndarray,
    peak_idx: int,
) -> Tuple[float, float, float]:
    """
    Calcula el ancho del pico a -3 dB (factor 1/sqrt(2) ≈ 0.7071 del
    valor máximo) usando interpolación lineal.

    Retorna:
        (n1_rpm, n2_rpm, fwhm_rpm)

    Si no se encuentra cruce a la izquierda o derecha, devuelve NaN
    para esos límites.
    """
    n = rpm.size
    peak_amp = float(amp[peak_idx])
    threshold = peak_amp / np.sqrt(2.0)

    # Hacia la izquierda
    n1 = float("nan")
    for i in range(peak_idx, 0, -1):
        if amp[i - 1] <= threshold <= amp[i]:
            x1, x2 = float(rpm[i - 1]), float(rpm[i])
            y1, y2 = float(amp[i - 1]), float(amp[i])
            if y2 == y1:
                n1 = x1
            else:
                n1 = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            break

    # Hacia la derecha
    n2 = float("nan")
    for i in range(peak_idx, n - 1):
        if amp[i] >= threshold >= amp[i + 1]:
            x1, x2 = float(rpm[i]), float(rpm[i + 1])
            y1, y2 = float(amp[i]), float(amp[i + 1])
            if y2 == y1:
                n2 = x2
            else:
                n2 = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            break

    if not np.isfinite(n1) or not np.isfinite(n2):
        return n1, n2, float("nan")

    return n1, n2, float(n2 - n1)


# =============================================================
# DETECCIÓN DE VELOCIDADES CRÍTICAS
# =============================================================

def detect_critical_speeds(
    rpm: np.ndarray,
    amp: np.ndarray,
    phase: np.ndarray,
    *,
    min_amp_prominence_ratio: float = 0.05,
    min_phase_change_deg: float = 40.0,
    smooth_window: int = 5,
    min_distance_rpm: float = 150.0,
    edge_margin_pct: float = 0.03,
    max_fwhm_fraction: float = 0.30,
    min_q_factor: float = 1.5,
) -> List[CriticalSpeed]:
    """
    Detecta velocidades críticas a partir de un Bode (amplitud + fase
    contra RPM).

    Criterios para confirmar una crítica:
      1. La amplitud forma un pico local con prominencia ≥
         min_amp_prominence_ratio × amp_máximo_global.
      2. El cambio total de fase en la ventana del pico es ≥
         min_phase_change_deg (típico: 60-90° para una crítica clara,
         180° ideal).
      3. Es posible calcular el ancho a -3 dB para obtener Q.

    Args:
        rpm: array de velocidades, ordenado ascendentemente.
        amp: array de amplitud 1X (mismas unidades que el reporte).
        phase: array de fase 1X en grados [0, 360).
        min_amp_prominence_ratio: prominencia relativa al pico máximo
            global. Por defecto 10%.
        min_phase_change_deg: cambio mínimo de fase para validar crítica.
        smooth_window: ventana de suavizado circular para fase y
            amplitud antes de detectar picos. 0 desactiva.
        min_distance_rpm: separación mínima entre picos (rpm).

    Returns:
        Lista de CriticalSpeed ordenada ascendentemente por rpm.
    """
    rpm = np.asarray(rpm, dtype=float)
    amp = np.asarray(amp, dtype=float)
    phase = np.asarray(phase, dtype=float)

    # Validar y ordenar por rpm ascendente (idempotente)
    if rpm.size == 0 or amp.size == 0 or phase.size == 0:
        return []

    n_min = min(rpm.size, amp.size, phase.size)
    rpm, amp, phase = rpm[:n_min], amp[:n_min], phase[:n_min]

    order = np.argsort(rpm, kind="stable")
    rpm, amp, phase = rpm[order], amp[order], phase[order]

    # Suavizado
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        amp_smooth = np.convolve(amp, kernel, mode="same")
        phase_smooth = _rolling_circular_mean_deg(phase, smooth_window)
    else:
        amp_smooth = amp
        phase_smooth = phase

    if amp_smooth.size < 3:
        return []

    amp_max_global = float(np.nanmax(amp_smooth))
    if amp_max_global <= 0 or not np.isfinite(amp_max_global):
        return []

    min_prominence_abs = min_amp_prominence_ratio * amp_max_global

    # Distancia mínima entre picos en muestras
    if rpm.size > 1:
        median_drpm = float(np.median(np.diff(rpm)))
        if median_drpm > 0:
            min_distance_samples = max(3, int(round(min_distance_rpm / median_drpm)))
        else:
            min_distance_samples = 3
    else:
        min_distance_samples = 3

    peaks = _find_peaks_simple(
        amp_smooth,
        min_prominence=min_prominence_abs,
        min_distance=min_distance_samples,
    )

    criticals: List[CriticalSpeed] = []

    # Ventana en muestras para evaluar cambio de fase: usamos el FWHM
    # tentativo o ±10% del rango total como fallback
    phase_eval_half_window = max(5, rpm.size // 30)

    for p in peaks:
        peak_amp = float(amp_smooth[p])
        peak_rpm = float(rpm[p])

        n1, n2, fwhm = _compute_fwhm_at_minus3db(rpm, amp_smooth, p)

        # Filtro 1: prominencia ya validada en _find_peaks_simple

        # Filtro 2: cambio de fase
        if np.isfinite(n1) and np.isfinite(n2):
            # Encontrar índices más cercanos a n1 y n2 para acotar la ventana
            i1 = int(np.argmin(np.abs(rpm - n1)))
            i2 = int(np.argmin(np.abs(rpm - n2)))
            half_window = max(phase_eval_half_window, max(p - i1, i2 - p))
        else:
            half_window = phase_eval_half_window

        phase_change = _phase_change_local(phase_smooth, p, half_window)

        # Si la fase no cambia lo suficiente, descartamos
        if phase_change < min_phase_change_deg:
            continue

        # Filtro 3: Q factor (puede ser NaN si FWHM no detectable)
        if np.isfinite(fwhm) and fwhm > 0:
            q_factor = peak_rpm / fwhm
        else:
            q_factor = float("nan")

        # Filtro 4: rechazar artefactos de frontera del data.
        # Un "pico" cuyos flancos a -3dB están al borde de los datos
        # típicamente es respuesta forzada hacia operación nominal,
        # no una crítica genuina del rotor.
        rpm_span = float(rpm[-1] - rpm[0]) if rpm.size >= 2 else 0.0
        if rpm_span > 0:
            edge_threshold_rpm = rpm_span * edge_margin_pct
            if np.isfinite(n1) and (n1 - rpm[0]) < edge_threshold_rpm:
                continue
            if np.isfinite(n2) and (rpm[-1] - n2) < edge_threshold_rpm:
                continue

            # Filtro 5: FWHM no puede ser una fracción enorme del rango medido.
            # Un pico real es localizado; FWHM > 30% del rango sugiere que
            # la amplitud está variando lentamente, no resonando.
            if np.isfinite(fwhm) and fwhm > rpm_span * max_fwhm_fraction:
                continue

        # Filtro 6: rechazar Q muy bajo (no hay amplificación significativa).
        # Q < 1.5 corresponde a respuesta cuasi-plana, sin pico definido.
        if np.isfinite(q_factor) and q_factor < min_q_factor:
            continue

        # Confianza: combinación de prominencia relativa y cambio de fase
        prominence_score = min(1.0, peak_amp / amp_max_global)
        phase_score = min(1.0, phase_change / 180.0)
        confidence = float(0.5 * prominence_score + 0.5 * phase_score)

        criticals.append(
            CriticalSpeed(
                rpm=peak_rpm,
                frequency_hz=peak_rpm / 60.0,
                amp_peak=peak_amp,
                phase_at_peak=float(phase_smooth[p] % 360.0),
                phase_change_deg=phase_change,
                q_factor=q_factor,
                fwhm_rpm=fwhm,
                confidence=confidence,
                n1_rpm=n1,
                n2_rpm=n2,
            )
        )

    return criticals


# =============================================================
# FACTOR Q (público, para casos donde se necesita aislado)
# =============================================================

def compute_q_factor(
    rpm: np.ndarray,
    amp: np.ndarray,
    peak_idx: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Calcula el factor de amplificación Q (AF) para un pico específico
    o el pico global.

    Args:
        rpm: array de velocidades ordenado.
        amp: array de amplitudes.
        peak_idx: índice del pico. Si es None, usa el pico global.

    Returns:
        (q_factor, fwhm_rpm, n1_rpm, n2_rpm)

    Si no es posible calcular FWHM, devuelve NaN en todos los campos.
    """
    rpm = np.asarray(rpm, dtype=float)
    amp = np.asarray(amp, dtype=float)

    if rpm.size != amp.size or rpm.size < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")

    if peak_idx is None:
        peak_idx = int(np.nanargmax(amp))

    n1, n2, fwhm = _compute_fwhm_at_minus3db(rpm, amp, peak_idx)
    if not np.isfinite(fwhm) or fwhm <= 0:
        return float("nan"), fwhm, n1, n2

    q = float(rpm[peak_idx]) / fwhm
    return q, fwhm, n1, n2


# =============================================================
# API 684 SEPARATION MARGIN
# =============================================================

def required_separation_margin_api684(q_factor: float) -> float:
    """
    Calcula el margen de separación mínimo requerido por API 684 Tutorial
    on Rotor Dynamics, en función del factor de amplificación Q.

    Heurística práctica (consistente con API 684 §2.6.2):

      Q < 2.5  →  margen no requerido (rotor altamente amortiguado)
      Q ≥ 8    →  margen 26%
      en medio →  interpolación lineal entre (2.5, 15%) y (8, 26%)

    Args:
        q_factor: factor de amplificación Q.

    Returns:
        Margen mínimo recomendado en porcentaje.
    """
    if not np.isfinite(q_factor) or q_factor < 2.5:
        return 0.0
    if q_factor >= 8.0:
        return 26.0
    # Interpolación lineal entre Q=2.5→15% y Q=8→26%
    return 15.0 + (q_factor - 2.5) * (26.0 - 15.0) / (8.0 - 2.5)


def evaluate_api684_margin(
    critical_rpm: float,
    operating_rpm: float,
    q_factor: float,
) -> API684Margin:
    """
    Evalúa el margen de separación API 684 para una crítica respecto
    a la velocidad de operación.

    Args:
        critical_rpm: velocidad crítica detectada.
        operating_rpm: velocidad de operación (ej. 3600 rpm para
            generadores de 60 Hz, 2 polos).
        q_factor: factor de amplificación Q de la crítica.

    Returns:
        API684Margin con margen actual, requerido, conformidad, zona y
        narrativa textual.
    """
    if operating_rpm <= 0:
        return API684Margin(
            critical_rpm=critical_rpm,
            operating_rpm=operating_rpm,
            q_factor=q_factor,
            actual_margin_pct=0.0,
            required_margin_pct=0.0,
            compliant=False,
            zone="DANGER",
            narrative="Velocidad de operación inválida.",
        )

    actual = abs(operating_rpm - critical_rpm) / operating_rpm * 100.0
    required = required_separation_margin_api684(q_factor)

    if not np.isfinite(q_factor):
        zone: Literal["SAFE", "MARGINAL", "DANGER"] = "MARGINAL"
        compliant = False
        narrative = (
            f"Crítica a {critical_rpm:.0f} rpm con Q no determinable. "
            f"Margen actual {actual:.1f}%. No se puede evaluar conformidad API 684 "
            f"sin Q válido."
        )
        return API684Margin(
            critical_rpm=critical_rpm,
            operating_rpm=operating_rpm,
            q_factor=q_factor,
            actual_margin_pct=actual,
            required_margin_pct=required,
            compliant=compliant,
            zone=zone,
            narrative=narrative,
        )

    compliant = actual >= required

    if q_factor < 2.5:
        zone = "SAFE"
        narrative = (
            f"Q={q_factor:.2f} < 2.5: rotor altamente amortiguado en la zona de "
            f"{critical_rpm:.0f} rpm. API 684 no requiere margen de separación. "
            f"Margen actual {actual:.1f}%."
        )
    elif compliant:
        margin_excess = actual - required
        zone = "SAFE" if margin_excess >= 5.0 else "MARGINAL"
        narrative = (
            f"Crítica a {critical_rpm:.0f} rpm con Q={q_factor:.2f}. "
            f"Margen requerido API 684: {required:.1f}%. "
            f"Margen actual: {actual:.1f}% ({'+' if margin_excess >= 0 else ''}"
            f"{margin_excess:.1f}% sobre el mínimo). "
            f"{'Conforme' if compliant else 'No conforme'}."
        )
    else:
        deficit = required - actual
        zone = "DANGER"
        narrative = (
            f"Crítica a {critical_rpm:.0f} rpm con Q={q_factor:.2f}. "
            f"Margen requerido API 684: {required:.1f}%. "
            f"Margen actual: {actual:.1f}% (déficit de {deficit:.1f}%). "
            f"NO CONFORME — rotor opera demasiado cerca de una resonancia "
            f"con factor de amplificación significativo."
        )

    return API684Margin(
        critical_rpm=critical_rpm,
        operating_rpm=operating_rpm,
        q_factor=q_factor,
        actual_margin_pct=actual,
        required_margin_pct=required,
        compliant=compliant,
        zone=zone,
        narrative=narrative,
    )


# =============================================================
# ISO 20816-2 SEVERITY ZONES
# =============================================================

# Tabla de umbrales para shaft displacement (proximity probe, peak-peak)
# en µm pp, basada en ISO 20816-2:2017 Tabla 1.
# Valores aplicables a turbogeneradores >40 MW.
#
# Group 1: turbinas/generadores con cojinete de rodillos o configuración
#          rígida (raro en turbogeneradores grandes).
# Group 2: turbinas/generadores con cojinetes planos shaft mounted
#          (caso típico Brush, Siemens, GE turbogeneradores).
ISO_20816_SHAFT_DISPLACEMENT_UM_PP = {
    # (group, op_speed_rpm_class): (A/B, B/C, C/D)
    ("group1", 1500): (120.0, 240.0, 385.0),
    ("group1", 1800): (120.0, 240.0, 385.0),
    ("group1", 3000): (100.0, 200.0, 320.0),
    ("group1", 3600): (100.0, 200.0, 320.0),
    ("group2", 1500): (90.0, 185.0, 290.0),
    ("group2", 1800): (90.0, 185.0, 290.0),
    ("group2", 3000): (75.0, 150.0, 240.0),
    ("group2", 3600): (75.0, 150.0, 240.0),
}

# Tabla de umbrales para vibración de carcasa (velocidad RMS, mm/s)
# basada en ISO 20816-2:2017.
ISO_20816_CASING_VELOCITY_MM_S_RMS = {
    ("group1", 1500): (3.8, 7.5, 11.8),
    ("group1", 1800): (3.8, 7.5, 11.8),
    ("group1", 3000): (3.8, 7.5, 11.8),
    ("group1", 3600): (3.8, 7.5, 11.8),
    ("group2", 1500): (2.8, 5.7, 9.1),
    ("group2", 1800): (2.8, 5.7, 9.1),
    ("group2", 3000): (2.8, 5.7, 9.1),
    ("group2", 3600): (2.8, 5.7, 9.1),
}

ZONE_DESCRIPTIONS = {
    "A": "Vibración típica de máquinas nuevas o recién comisionadas. Aceptable sin restricciones.",
    "B": "Vibración aceptable para operación continua sin restricciones.",
    "C": "Vibración insatisfactoria. Permite operación limitada hasta correctivo programado.",
    "D": "Vibración suficiente para causar daño. Acción inmediata requerida.",
}


def _nearest_speed_class(operating_rpm: float) -> int:
    """Mapea rpm operativa a clase de velocidad ISO 20816-2."""
    speed_classes = (1500, 1800, 3000, 3600)
    return min(speed_classes, key=lambda c: abs(c - operating_rpm))


def iso_20816_2_zone(
    amplitude: float,
    *,
    measurement_type: Literal["shaft_displacement", "casing_velocity"] = "shaft_displacement",
    machine_group: Literal["group1", "group2"] = "group2",
    operating_speed_rpm: float = 3600.0,
) -> ISO20816Zone:
    """
    Clasifica una amplitud de vibración según ISO 20816-2:2017 zones
    A/B/C/D para turbogeneradores >40 MW.

    Args:
        amplitude: valor a clasificar.
            Si measurement_type='shaft_displacement', en µm pico-pico.
            Si measurement_type='casing_velocity', en mm/s RMS.
        measurement_type: tipo de medición.
        machine_group: 'group1' (cantilever) o 'group2' (shaft mounted,
            caso típico de turbogeneradores grandes con cojinetes planos).
        operating_speed_rpm: velocidad de operación (se mapea a la clase
            ISO más cercana).

    Returns:
        ISO20816Zone con la clasificación y narrativa.
    """
    speed_class = _nearest_speed_class(operating_speed_rpm)

    if measurement_type == "shaft_displacement":
        unit = "um_pp"
        thresholds_table = ISO_20816_SHAFT_DISPLACEMENT_UM_PP
    elif measurement_type == "casing_velocity":
        unit = "mm_s_rms"
        thresholds_table = ISO_20816_CASING_VELOCITY_MM_S_RMS
    else:
        raise ValueError(f"measurement_type inválido: {measurement_type}")

    key = (machine_group, speed_class)
    if key not in thresholds_table:
        raise ValueError(f"Combinación no soportada en tabla ISO: {key}")

    ab, bc, cd = thresholds_table[key]

    if not np.isfinite(amplitude) or amplitude < 0:
        zone = "D"
    elif amplitude < ab:
        zone = "A"
    elif amplitude < bc:
        zone = "B"
    elif amplitude < cd:
        zone = "C"
    else:
        zone = "D"

    return ISO20816Zone(
        amplitude=float(amplitude),
        unit=unit,
        zone=zone,
        zone_description=ZONE_DESCRIPTIONS[zone],
        boundary_AB=ab,
        boundary_BC=bc,
        boundary_CD=cd,
        machine_group=machine_group,
        operating_speed_rpm=float(operating_speed_rpm),
    )


# =============================================================
# ISO 20816-3:2022 — Industrial machines 15 kW – 40 MW
# =============================================================
# Aplicable a:
#  - Motores eléctricos
#  - Turbinas de gas industrial heavy-duty bajo 40 MW (SGT-300/400)
#  - Turbinas de vapor 15-40 MW
#  - Compresores centrífugos
#
# Las clases dependen del soporte y la potencia. La parte 3 se mide
# usualmente como velocidad RMS (mm/s) sobre la carcasa, pero también
# se aceptan criterios de desplazamiento del eje cuando hay sondas
# de proximidad instaladas (para máquinas grandes con cojinetes planos
# como turbinas industriales).
#
# Tabla de zonas A/B/C/D para shaft displacement (µm pp), valores de
# referencia consistentes con la práctica de la industria para
# máquinas clase IV (rigid foundation, large machines >300 kW).

ISO_20816_3_SHAFT_DISPLACEMENT_UM_PP = {
    # (class, op_speed_rpm_class): (A/B, B/C, C/D)
    ("class_iv", 1200): (130.0, 270.0, 430.0),
    ("class_iv", 1500): (110.0, 220.0, 350.0),
    ("class_iv", 1800): (100.0, 200.0, 320.0),
    ("class_iv", 3000): (75.0, 150.0, 240.0),
    ("class_iv", 3600): (65.0, 130.0, 210.0),
    ("class_iv", 4500): (55.0, 110.0, 180.0),
    ("class_iv", 7200): (45.0, 90.0, 150.0),
    ("class_iv", 14000): (30.0, 60.0, 100.0),
    ("class_iv", 14600): (30.0, 60.0, 100.0),
}

# Tabla de zonas para casing velocity RMS (mm/s) según ISO 20816-3
# Clase IV (large rigid foundation): A/B=4.5, B/C=11.2, C/D=28.0
# Clase III: A/B=2.8, B/C=7.1, C/D=18.0
# Clase II: A/B=1.8, B/C=4.5, C/D=11.2
# Clase I: A/B=1.12, B/C=2.8, C/D=7.1

ISO_20816_3_CASING_VELOCITY_MM_S_RMS = {
    ("class_i", 0): (1.12, 2.8, 7.1),
    ("class_ii", 0): (1.8, 4.5, 11.2),
    ("class_iii", 0): (2.8, 7.1, 18.0),
    ("class_iv", 0): (4.5, 11.2, 28.0),
}


# =============================================================
# ISO 20816-4:2018 — Turbinas con cojinetes de rodillos (aero-derivative)
# =============================================================
# Para LM2500, LM6000 y similares. La medición primaria es velocidad RMS
# sobre carcasa de cojinetes, los rangos típicos:

ISO_20816_4_CASING_VELOCITY_MM_S_RMS = {
    ("rolling_aero", 0): (4.5, 9.3, 14.7),
}


# =============================================================
# ISO 20816-7:2016 — Bombas rotodinámicas industriales
# =============================================================
# Categorías:
#  - Category I: bombas críticas (operación continua sin redundancia)
#  - Category II: bombas auxiliares (con redundancia o backup)
#
# La parte 7 aplica medición de velocidad RMS de carcasa.

ISO_20816_7_CASING_VELOCITY_MM_S_RMS = {
    ("category_i", 0): (2.5, 5.0, 8.0),     # bombas críticas
    ("category_ii", 0): (3.5, 7.0, 11.0),   # bombas auxiliares
}


# =============================================================
# DISPATCHER MULTI-PART
# =============================================================

ISO_PART_TABLES = {
    # Cada entrada: parte → (tipo de medición default, dict de tablas)
    "20816-2": {
        "shaft_displacement": ISO_20816_SHAFT_DISPLACEMENT_UM_PP,
        "casing_velocity": ISO_20816_CASING_VELOCITY_MM_S_RMS,
    },
    "20816-3": {
        "shaft_displacement": ISO_20816_3_SHAFT_DISPLACEMENT_UM_PP,
        "casing_velocity": ISO_20816_3_CASING_VELOCITY_MM_S_RMS,
    },
    "20816-4": {
        "casing_velocity": ISO_20816_4_CASING_VELOCITY_MM_S_RMS,
    },
    "20816-7": {
        "casing_velocity": ISO_20816_7_CASING_VELOCITY_MM_S_RMS,
    },
}


def iso_20816_zone_multipart(
    amplitude: float,
    *,
    iso_part: str = "20816-2",
    machine_group: str = "group2",
    measurement_type: str = "shaft_displacement",
    operating_speed_rpm: float = 3600.0,
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
) -> ISO20816Zone:
    """
    Clasifica una amplitud de vibración según la parte de ISO 20816 indicada.
    Soporta partes 2, 3, 4 y 7. Para partes que no aplican shaft_displacement
    (4 y 7), usa casing_velocity.

    Si el caller pasa custom_thresholds=(AB, BC, CD), esos overriden las
    tablas y se interpretan en la unidad correspondiente al measurement_type.

    Args:
        amplitude: valor a clasificar (µm pp o mm/s RMS según measurement_type)
        iso_part: '20816-2', '20816-3', '20816-4', '20816-7' o 'custom'
        machine_group: depende de la parte (group1/group2, class_i..iv,
            rolling_aero, category_i/ii)
        measurement_type: 'shaft_displacement' o 'casing_velocity'
        operating_speed_rpm: usado solo en partes que dependen de RPM (2 y 3)
        custom_thresholds: override manual (AB, BC, CD) en la unidad correspondiente

    Returns:
        ISO20816Zone con la clasificación y narrativa.
    """
    # Determinar umbrales
    if custom_thresholds is not None:
        ab, bc, cd = custom_thresholds
    elif iso_part == "custom":
        raise ValueError(
            "iso_part='custom' requiere custom_thresholds=(AB, BC, CD)"
        )
    else:
        if iso_part not in ISO_PART_TABLES:
            raise ValueError(f"ISO part no soportada: {iso_part}")

        part_tables = ISO_PART_TABLES[iso_part]
        if measurement_type not in part_tables:
            raise ValueError(
                f"measurement_type='{measurement_type}' no aplica a ISO {iso_part}"
            )

        table = part_tables[measurement_type]

        # La parte 2 y 3 (shaft displacement) dependen de la velocidad operativa
        if iso_part in ("20816-2", "20816-3") and measurement_type == "shaft_displacement":
            speed_class = _nearest_speed_class_multipart(operating_speed_rpm, iso_part)
            key = (machine_group, speed_class)
        else:
            # Velocidad RMS no depende de RPM en ISO 20816-3/-4/-7
            key = (machine_group, 0)

        if key not in table:
            raise ValueError(f"Combinación no soportada en tabla ISO {iso_part}: {key}")

        ab, bc, cd = table[key]

    # Determinar unidad
    if measurement_type == "shaft_displacement":
        unit = "um_pp"
    elif measurement_type == "casing_velocity":
        unit = "mm_s_rms"
    else:
        unit = measurement_type

    # Clasificar zona
    if not np.isfinite(amplitude) or amplitude < 0:
        zone = "D"
    elif amplitude < ab:
        zone = "A"
    elif amplitude < bc:
        zone = "B"
    elif amplitude < cd:
        zone = "C"
    else:
        zone = "D"

    return ISO20816Zone(
        amplitude=float(amplitude),
        unit=unit,
        zone=zone,
        zone_description=ZONE_DESCRIPTIONS[zone],
        boundary_AB=ab,
        boundary_BC=bc,
        boundary_CD=cd,
        machine_group=machine_group,
        operating_speed_rpm=float(operating_speed_rpm),
    )


def _nearest_speed_class_multipart(operating_rpm: float, iso_part: str) -> int:
    """Mapea operating_rpm a la clase de velocidad de la parte ISO indicada."""
    if iso_part == "20816-2":
        speed_classes = (1500, 1800, 3000, 3600)
    elif iso_part == "20816-3":
        speed_classes = (1200, 1500, 1800, 3000, 3600, 4500, 7200, 14000, 14600)
    else:
        return int(round(operating_rpm))
    return min(speed_classes, key=lambda c: abs(c - operating_rpm))


# =============================================================
# UNIDAD CONVERSIÓN
# =============================================================

def mils_to_micrometers(mils_pp: float) -> float:
    """1 mil = 25.4 µm. Convierte mils pp a µm pp."""
    return float(mils_pp) * 25.4


def micrometers_to_mils(um_pp: float) -> float:
    """Convierte µm pp a mils pp."""
    return float(um_pp) / 25.4


__all__ = [
    "CriticalSpeed",
    "API684Margin",
    "ISO20816Zone",
    "detect_critical_speeds",
    "compute_q_factor",
    "required_separation_margin_api684",
    "evaluate_api684_margin",
    "iso_20816_2_zone",
    "iso_20816_zone_multipart",
    "ISO_PART_TABLES",
    "mils_to_micrometers",
    "micrometers_to_mils",
]
