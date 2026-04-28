"""
core.waveform_pattern_detectors
================================

Detectores Cat IV específicos sobre señales de forma de onda en tiempo.

A diferencia del análisis espectral (que promedia y revela frecuencias
discretas), el dominio del tiempo permite identificar **fenómenos
transitorios y no estacionarios** que un FFT enmascara:

  - Modulación de amplitud (AM): la envolvente respira con un patrón
    periódico. Típico de defectos de rodamiento incipientes (BPFO,
    BPFI modulan la portadora 1X), de engranajes con desgaste o de
    condiciones de carga variable.

  - Asimetría direccional: la onda tiene picos más altos hacia un lado
    que hacia el otro. Indicador de rub unidireccional, restricción
    direccional del eje (precarga lateral), o saturación del sensor en
    una polaridad.

  - Beating: dos componentes de frecuencia muy cercanas se interfieren
    constructivamente / destructivamente, creando una envolvente con
    "respiración" lenta. Típico de slip de polos en motores de
    inducción, máquinas vecinas a velocidades parecidas.

  - Clipping: la señal está truncada en el tope porque el sensor o el
    sistema de adquisición saturó. Indica que el rango dinámico
    elegido es insuficiente — los picos reales son mayores.

  - Forma diente de sierra: la onda no es sinusoidal sino que tiene
    pendientes asimétricas tipo serrucho. Indicador de rub severo
    bidireccional o restricción dinámica fuerte.

Las funciones devuelven dicts con: ``detected`` (bool),
``confidence`` (0-1), ``narrative`` (texto Cat IV), y métricas
específicas de cada detector. Todas son tolerantes a entradas
vacías o degeneradas.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


# =============================================================
# UTILIDAD: HILBERT ENVELOPE (sin scipy.signal)
# =============================================================

def _analytic_signal_envelope(x: np.ndarray) -> np.ndarray:
    """
    Calcula la envolvente |x_a(t)| de una señal x(t) vía Hilbert
    transform implementada con FFT (sin depender de scipy.signal).

    Para una señal real x[n] de N muestras:
      X[k] = FFT(x)
      H[k] = X[k] × {1 si k=0 o k=N/2; 2 si 0<k<N/2; 0 si k>N/2}
      x_a[n] = IFFT(H[k])     (señal analítica)
      envelope[n] = |x_a[n]|
    """
    n = x.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.abs(x)

    X = np.fft.fft(x)
    H = np.zeros(n, dtype=complex)
    half = n // 2
    if n % 2 == 0:  # par
        H[0] = X[0]
        H[half] = X[half]
        H[1:half] = 2.0 * X[1:half]
    else:           # impar
        H[0] = X[0]
        H[1:(n + 1) // 2] = 2.0 * X[1:(n + 1) // 2]
    x_a = np.fft.ifft(H)
    return np.abs(x_a)


# =============================================================
# DETECTOR 1: MODULACIÓN DE AMPLITUD
# =============================================================

def detect_amplitude_modulation(
    time_s: np.ndarray,
    amplitude: np.ndarray,
    *,
    min_modulation_depth: float = 0.15,
) -> Dict[str, Any]:
    """
    Detecta modulación de amplitud (AM) buscando variación periódica en
    la envolvente Hilbert. Reporta la frecuencia y profundidad de
    modulación.

    Modulation depth = (max_env - min_env) / (max_env + min_env)
        0 → señal de amplitud constante
        ~0.20 → modulación leve (defectos incipientes)
        ~0.50 → modulación severa (defectos avanzados)
        →1 → cuasi a/m on-off, contacto intermitente

    Frecuencia de modulación detectada vía FFT de la envolvente
    centrada en cero.
    """
    x = np.asarray(amplitude, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 64:
        return {"detected": False, "narrative": "", "modulation_depth": 0.0}

    # Restar media para evitar pico DC dominante en la envolvente
    x_centered = x - np.mean(x)
    env = _analytic_signal_envelope(x_centered)

    # Suavizado leve (media móvil corta) para no contar ruido
    win = max(int(env.size * 0.005), 3)
    if win % 2 == 0:
        win += 1
    if win >= env.size:
        return {"detected": False, "narrative": "", "modulation_depth": 0.0}
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode="same")

    e_min = float(np.min(env_smooth))
    e_max = float(np.max(env_smooth))
    e_mean = float(np.mean(env_smooth))
    if e_max + e_min <= 0 or e_mean <= 0:
        return {"detected": False, "narrative": "", "modulation_depth": 0.0}

    mod_depth = (e_max - e_min) / (e_max + e_min)

    if mod_depth < min_modulation_depth:
        return {
            "detected": False,
            "modulation_depth": mod_depth,
            "narrative": (
                f"Profundidad de modulación de la envolvente: {mod_depth*100:.1f}%, "
                f"por debajo del umbral de significancia "
                f"({min_modulation_depth*100:.0f}%). La envolvente de la señal es "
                f"aproximadamente constante."
            ),
        }

    # Estimar frecuencia de modulación: FFT de la envolvente centrada
    if time_s is not None and time_s.size >= 2:
        dt_s = float(np.median(np.diff(time_s)))
        fs = 1.0 / dt_s if dt_s > 0 else 0.0
    else:
        fs = 0.0

    mod_freq_hz = 0.0
    if fs > 0:
        env_centered = env_smooth - np.mean(env_smooth)
        E = np.abs(np.fft.rfft(env_centered))
        freqs = np.fft.rfftfreq(env_centered.size, d=1.0 / fs)
        # Ignorar primeros bins (DC y muy lentos)
        if freqs.size > 5:
            idx_peak = int(np.argmax(E[1:]) + 1)
            mod_freq_hz = float(freqs[idx_peak])

    if mod_depth >= 0.5:
        severity_word = "severa"
    elif mod_depth >= 0.30:
        severity_word = "moderada"
    else:
        severity_word = "leve"

    freq_clause = ""
    if mod_freq_hz > 0:
        freq_clause = (
            f" La frecuencia de modulación dominante se ubica en "
            f"{mod_freq_hz:.1f} Hz ({mod_freq_hz*60:.0f} CPM), valor que "
            f"debe contrastarse con frecuencias características de defectos "
            f"de rodamiento (BPFO/BPFI/BSF/FTF), de engrane (GMF) o con "
            f"frecuencias de paso típicas del proceso."
        )

    narrative = (
        f"Se detecta modulación {severity_word} de la amplitud (envolvente "
        f"varía un {mod_depth*100:.1f}% pico a pico).{freq_clause} La "
        f"modulación AM en señales de vibración suele indicar defectos de "
        f"rodamiento incipientes, desgaste de engranaje, condiciones de "
        f"carga variable o golpes periódicos de origen mecánico."
    )

    return {
        "detected": True,
        "modulation_depth": mod_depth,
        "modulation_freq_hz": mod_freq_hz,
        "severity_word": severity_word,
        "narrative": narrative,
    }


# =============================================================
# DETECTOR 2: ASIMETRÍA DIRECCIONAL (rub / precarga unidireccional)
# =============================================================

def detect_asymmetry(
    amplitude: np.ndarray,
    *,
    min_asymmetry_ratio: float = 1.30,
) -> Dict[str, Any]:
    """
    Compara el pico positivo absoluto contra el pico negativo absoluto.
    Si la relación |p+| / |p-| (o su inversa) supera ``min_asymmetry_ratio``,
    se reporta asimetría direccional con narrativa Cat IV (rub
    unidireccional, precarga lateral, restricción direccional del
    movimiento del eje).
    """
    x = np.asarray(amplitude, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return {"detected": False, "narrative": ""}

    p_pos = float(np.max(x))
    p_neg = float(-np.min(x))
    if p_pos <= 0 or p_neg <= 0:
        return {"detected": False, "narrative": ""}

    if p_pos >= p_neg:
        ratio = p_pos / p_neg
        direction = "positiva"
    else:
        ratio = p_neg / p_pos
        direction = "negativa"

    if ratio < min_asymmetry_ratio:
        return {
            "detected": False,
            "ratio": ratio,
            "narrative": (
                f"La onda es esencialmente simétrica respecto a su línea base "
                f"(relación pico+/pico- = {ratio:.2f}, por debajo del umbral "
                f"de asimetría {min_asymmetry_ratio:.2f})."
            ),
        }

    if ratio >= 2.0:
        severity_word = "severa"
    elif ratio >= 1.6:
        severity_word = "marcada"
    else:
        severity_word = "leve"

    narrative = (
        f"Se observa asimetría direccional {severity_word} hacia la "
        f"polaridad {direction} (relación pico+/pico- = {ratio:.2f}). Esta "
        f"firma es consistente con rub unidireccional eje–estator, "
        f"restricción dinámica del eje hacia un lado del cojinete "
        f"(precarga lateral), o con saturación del sensor en una sola "
        f"polaridad. Verificar centrado del eje en el cojinete (Shaft "
        f"Centerline), condición de sellos cercanos, alineación del tren "
        f"y escala dinámica del sensor."
    )

    return {
        "detected": True,
        "ratio": ratio,
        "direction": direction,
        "severity_word": severity_word,
        "narrative": narrative,
    }


# =============================================================
# DETECTOR 3: CLIPPING (sensor saturado / rango insuficiente)
# =============================================================

def detect_clipping(
    amplitude: np.ndarray,
    *,
    saturation_tolerance: float = 0.005,
    min_clipped_fraction: float = 0.005,
) -> Dict[str, Any]:
    """
    Detecta clipping inspeccionando si una fracción significativa de
    las muestras está pegada al máximo absoluto. Si el sensor saturó,
    los picos reales se truncan y aparecen muchos puntos exactamente
    al límite del rango.
    """
    x = np.asarray(amplitude, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return {"detected": False, "narrative": ""}

    p_max = float(np.max(np.abs(x)))
    if p_max <= 0:
        return {"detected": False, "narrative": ""}

    band = saturation_tolerance * p_max
    n_clipped = int(np.sum(np.abs(np.abs(x) - p_max) <= band))
    fraction = n_clipped / x.size

    if fraction < min_clipped_fraction:
        return {
            "detected": False,
            "fraction": fraction,
            "narrative": "",
        }

    narrative = (
        f"Se detecta posible saturación del sensor: {fraction*100:.2f}% "
        f"de las muestras se encuentran dentro de una banda muy estrecha "
        f"alrededor del valor pico máximo (±{saturation_tolerance*100:.1f}%), "
        f"comportamiento típico de clipping. Esto significa que los picos "
        f"reales de la señal son mayores que el rango medido — la amplitud "
        f"reportada está subestimada. Verificar escala del sensor o "
        f"settings del sistema de adquisición y repetir la captura con "
        f"rango ampliado antes de concluir severidad."
    )

    return {
        "detected": True,
        "fraction": fraction,
        "narrative": narrative,
    }


# =============================================================
# DETECTOR 4: FORMA DIENTE DE SIERRA (rub severo bidireccional)
# =============================================================

def detect_sawtooth_shape(
    amplitude: np.ndarray,
    *,
    asymmetry_ratio_threshold: float = 1.8,
) -> Dict[str, Any]:
    """
    Detecta forma diente-de-sierra comparando la pendiente promedio
    de subida vs la pendiente promedio de bajada de la señal. En una
    sinusoidal pura ambas son iguales; en una onda diente-de-sierra
    una rampa es mucho más larga que la otra.
    """
    x = np.asarray(amplitude, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 100:
        return {"detected": False, "narrative": ""}

    diffs = np.diff(x)
    rising = diffs[diffs > 0]
    falling = -diffs[diffs < 0]
    if rising.size < 10 or falling.size < 10:
        return {"detected": False, "narrative": ""}

    mean_up = float(np.mean(rising))
    mean_down = float(np.mean(falling))
    if mean_up <= 0 or mean_down <= 0:
        return {"detected": False, "narrative": ""}

    ratio = max(mean_up, mean_down) / min(mean_up, mean_down)

    if ratio < asymmetry_ratio_threshold:
        return {"detected": False, "narrative": "", "slope_ratio": ratio}

    fast_side = "ascendente" if mean_up > mean_down else "descendente"
    narrative = (
        f"La forma de onda presenta pendientes asimétricas (rampa "
        f"{fast_side} {ratio:.2f}× más rápida que la opuesta), patrón "
        f"característico de **forma diente de sierra**. Este "
        f"comportamiento se asocia clásicamente con rub severo del eje "
        f"contra estator o con restricción dinámica fuerte del movimiento "
        f"del rotor. Inspección directa del cojinete y de los sellos "
        f"recomendada en el próximo paro programado."
    )

    return {
        "detected": True,
        "slope_ratio": ratio,
        "fast_side": fast_side,
        "narrative": narrative,
    }


# =============================================================
# DETECTOR 5: BEATING (interferencia entre frecuencias cercanas)
# =============================================================

def detect_beating(
    time_s: np.ndarray,
    amplitude: np.ndarray,
    *,
    min_beating_modulation: float = 0.30,
    max_beat_freq_hz: float = 10.0,
) -> Dict[str, Any]:
    """
    Detecta beating buscando una modulación periódica en la envolvente
    a frecuencia muy baja (típicamente < 10 Hz) con profundidad alta.
    Beating es la firma de dos sinusoides de amplitud comparable a
    frecuencias muy cercanas entre sí.
    """
    am = detect_amplitude_modulation(
        time_s=time_s,
        amplitude=amplitude,
        min_modulation_depth=min_beating_modulation,
    )
    if not am.get("detected"):
        return {"detected": False, "narrative": ""}

    mod_freq = float(am.get("modulation_freq_hz", 0.0))
    if mod_freq <= 0 or mod_freq > max_beat_freq_hz:
        return {"detected": False, "narrative": ""}

    narrative = (
        f"Se detecta beating: modulación lenta de la amplitud a "
        f"{mod_freq:.2f} Hz ({mod_freq*60:.1f} CPM) con profundidad "
        f"{am['modulation_depth']*100:.1f}%. Este patrón es la huella de "
        f"dos componentes de frecuencia cercana cuya diferencia genera la "
        f"frecuencia de batido observada. Causas típicas: slip eléctrico "
        f"en motores de inducción (deslizamiento), interferencia con "
        f"máquinas vecinas a velocidad parecida, o resonancia compuesta. "
        f"Verificar velocidades de operación de máquinas adyacentes y "
        f"comparar con el spectrum para identificar las dos componentes "
        f"de frecuencia generadoras del batido."
    )

    return {
        "detected": True,
        "beat_freq_hz": mod_freq,
        "modulation_depth": am["modulation_depth"],
        "narrative": narrative,
    }


# =============================================================
# CLASIFICACIÓN CAT IV DE CREST FACTOR
# =============================================================

CREST_FACTOR_BUCKETS: Tuple[Tuple[float, float, str, str, str], ...] = (
    (0.0,  3.0,  "SINUSOIDAL",  "CONDICIÓN ACEPTABLE",
        "Forma de onda esencialmente sinusoidal — comportamiento "
        "rotodinámico estable, sin transitorios discretos."),
    (3.0,  4.0,  "NORMAL",      "VIGILANCIA",
        "Crest factor en rango normal con armónicos moderados. "
        "Mantener el monitoreo rutinario."),
    (4.0,  6.0,  "ALERT",       "ATENCIÓN",
        "Presencia de transitorios de alta energía. Investigar fuentes "
        "de impacto o no linealidad antes de progresar."),
    (6.0,  10.0, "SEVERE",      "ACCIÓN REQUERIDA",
        "Comportamiento altamente impulsivo. Defectos discretos "
        "evidentes (rodamiento deteriorado, cavitación, golpeteo "
        "mecánico). Programar intervención."),
    (10.0, 1e6,  "CRITICAL",    "CRÍTICA",
        "Señal con shocks dominantes — daño avanzado en elementos "
        "rodantes o contacto severo. Restringir operación sostenida "
        "hasta inspección directa."),
)


def classify_crest_factor(cf: Optional[float]) -> Dict[str, Any]:
    """
    Clasifica el crest factor en un bucket Cat IV con severidad,
    mensaje y rank ordinal.
    """
    if cf is None or not np.isfinite(cf):
        return {
            "bucket": "UNKNOWN", "severity_label": "VIGILANCIA",
            "rank": 1, "message": "Crest factor no disponible.",
        }
    cf = float(cf)
    rank = 0
    for (lo, hi, bucket, label, msg) in CREST_FACTOR_BUCKETS:
        if lo <= cf < hi:
            return {
                "bucket": bucket, "severity_label": label,
                "rank": rank, "message": msg, "cf_value": cf,
            }
        rank += 1
    return {
        "bucket": "CRITICAL", "severity_label": "CRÍTICA",
        "rank": 4, "message": "Crest factor extremadamente alto.",
        "cf_value": cf,
    }


__all__ = [
    "detect_amplitude_modulation",
    "detect_asymmetry",
    "detect_clipping",
    "detect_sawtooth_shape",
    "detect_beating",
    "classify_crest_factor",
    "CREST_FACTOR_BUCKETS",
]
