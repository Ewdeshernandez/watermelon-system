from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.rotordynamics import (
    detect_critical_speeds,
    evaluate_api684_margin,
    iso_20816_2_zone,
    iso_20816_zone_multipart,
    mils_to_micrometers,
)


SAFE_COLOR = "#16a34a"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR = "#dc2626"
NEUTRAL_COLOR = "#2563eb"


def format_number(value: Any, digits: int = 3, fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        val = float(value)
        if not math.isfinite(val):
            return fallback
        return f"{val:.{digits}f}"
    except Exception:
        return fallback


def get_semaforo_status(
    value: float,
    safe_limit: float = 60.0,
    warning_limit: float = 85.0,
) -> Tuple[str, str]:
    value = float(value)
    if value < safe_limit:
        return "SAFE", SAFE_COLOR
    if value < warning_limit:
        return "WARNING", WARNING_COLOR
    return "DANGER", DANGER_COLOR


def remaining_margin_pct(util_pct: float) -> float:
    return max(0.0, 100.0 - float(util_pct))


def boundary_utilization_pct(
    px: float,
    py: float,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
) -> float:
    nx = (float(px) - float(center_x)) / max(float(clearance_x), 1e-9)
    ny = (float(py) - float(center_y)) / max(float(clearance_y), 1e-9)
    return math.sqrt(nx * nx + ny * ny) * 100.0


def build_clearance_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
    safe_limit: float = 60.0,
    warning_limit: float = 85.0,
) -> Dict[str, Any]:
    if len(x) == 0 or len(y) == 0:
        status, color = get_semaforo_status(0.0, safe_limit, warning_limit)
        return {
            "util_max": 0.0,
            "util_min": 0.0,
            "util_mean": 0.0,
            "margin_min": 100.0,
            "status": status,
            "color": color,
        }

    utils = np.array(
        [
            boundary_utilization_pct(px, py, center_x, center_y, clearance_x, clearance_y)
            for px, py in zip(x, y)
        ],
        dtype=float,
    )

    util_max = float(np.max(utils))
    util_min = float(np.min(utils))
    util_mean = float(np.mean(utils))
    margin_min = remaining_margin_pct(util_max)
    status, color = get_semaforo_status(util_max, safe_limit, warning_limit)

    return {
        "util_max": util_max,
        "util_min": util_min,
        "util_mean": util_mean,
        "margin_min": margin_min,
        "status": status,
        "color": color,
    }


def detect_early_rub(
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
    warning_util_pct: float = 80.0,
    danger_util_pct: float = 95.0,
) -> Dict[str, Any]:
    if len(x) < 3:
        return {
            "triggered": False,
            "severity": "SAFE",
            "color": SAFE_COLOR,
            "message": "Insufficient points",
            "max_util_pct": 0.0,
            "contact_points": 0,
            "warning_points": 0,
            "trend_score": 0.0,
            "first_warning_speed": None,
            "first_danger_speed": None,
        }

    utils = np.array(
        [
            boundary_utilization_pct(px, py, center_x, center_y, clearance_x, clearance_y)
            for px, py in zip(x, y)
        ],
        dtype=float,
    )

    warning_mask = utils >= float(warning_util_pct)
    danger_mask = utils >= float(danger_util_pct)

    warning_points = int(np.sum(warning_mask))
    contact_points = int(np.sum(danger_mask))

    if len(utils) >= 5:
        idx = np.arange(len(utils), dtype=float)
        slope = float(np.polyfit(idx, utils, 1)[0])
    else:
        slope = 0.0

    last_n = min(8, len(utils))
    tail_mean = float(np.mean(utils[-last_n:])) if last_n else 0.0
    max_util = float(np.max(utils)) if len(utils) else 0.0

    if contact_points > 0 or max_util >= float(danger_util_pct):
        severity = "DANGER"
        color = DANGER_COLOR
        triggered = True
        message = "Early rub risk high"
    elif warning_points >= 2 or tail_mean >= float(warning_util_pct) or slope > 1.5:
        severity = "WARNING"
        color = WARNING_COLOR
        triggered = True
        message = "Possible early rub tendency"
    else:
        severity = "SAFE"
        color = SAFE_COLOR
        triggered = False
        message = "No early rub tendency detected"

    first_warning_speed = None
    first_danger_speed = None

    if np.any(warning_mask):
        first_warning_speed = float(speed[np.argmax(warning_mask)])
    if np.any(danger_mask):
        first_danger_speed = float(speed[np.argmax(danger_mask)])

    return {
        "triggered": triggered,
        "severity": severity,
        "color": color,
        "message": message,
        "max_util_pct": max_util,
        "contact_points": contact_points,
        "warning_points": warning_points,
        "trend_score": slope,
        "first_warning_speed": first_warning_speed,
        "first_danger_speed": first_danger_speed,
    }


# ============================================================
# TEXTUAL DIAGNOSTICS
# ============================================================
def build_polar_text_diagnostics(
    *,
    status: str,
    critical_speeds: List[Dict[str, float]],
    max_amp: float,
) -> Dict[str, str]:
    candidate_count = len(critical_speeds)

    if candidate_count == 0:
        headline = "No clear critical speed candidate detected"
        detail = (
            "The polar response does not show a strong peak-phase combination in the current speed range. "
            "Behavior looks relatively stable for this run."
        )
        action = "Continue monitoring during future run-up or coast-down events."
        return {"headline": headline, "detail": detail, "action": action}

    cs1 = critical_speeds[0]
    cs1_speed = int(round(float(cs1["speed"])))
    cs1_amp = float(cs1["amp"])
    cs1_phase = abs(float(cs1["phase_delta"]))

    if status == "SAFE":
        headline = f"Critical speed candidate detected near {cs1_speed} rpm, but response is still controlled"
        detail = (
            f"A candidate appears near {cs1_speed} rpm with amplitude around {cs1_amp:.3f} and "
            f"phase change of {cs1_phase:.1f}°. The response is present but not yet severe."
        )
        action = "Track the same zone in the next startup and compare growth trend."
        return {"headline": headline, "detail": detail, "action": action}

    if status == "WARNING":
        headline = f"Possible proximity to critical speed near {cs1_speed} rpm"
        detail = (
            f"The polar path shows a relevant response near {cs1_speed} rpm with amplitude around {cs1_amp:.3f} "
            f"and phase shift of {cs1_phase:.1f}°. This suggests dynamic amplification, but not a fully developed severe resonance."
        )
        action = "Monitor behavior during ramp-up, compare with Bode phase/amplitude, and verify if the peak repeats consistently."
        return {"headline": headline, "detail": detail, "action": action}

    headline = f"Strong critical speed behavior near {cs1_speed} rpm"
    detail = (
        f"The polar path shows a dominant response near {cs1_speed} rpm with amplitude around {cs1_amp:.3f} "
        f"and phase shift of {cs1_phase:.1f}°. This is consistent with a highly significant dynamic event."
    )
    action = "Review operating avoidance, confirm with Bode/orbit behavior, and evaluate mechanical margin before repeated runs."
    return {"headline": headline, "detail": detail, "action": action}


def build_shaft_text_diagnostics(
    *,
    status: str,
    util_max: float,
    margin_min: float,
    first_warning_speed: float | None = None,
    first_danger_speed: float | None = None,
) -> Dict[str, str]:
    if status == "SAFE":
        headline = "Shaft position is operating with healthy clearance margin"
        detail = (
            f"Maximum clearance utilization is {util_max:.1f}% and minimum remaining margin is {margin_min:.1f}%. "
            "The shaft centerline remains well inside the configured boundary."
        )
        action = "Keep trending clearance utilization and compare future runs for drift."
        return {"headline": headline, "detail": detail, "action": action}

    if status == "WARNING":
        speed_text = ""
        if first_warning_speed is not None:
            speed_text = f" First warning tendency appears near {first_warning_speed:.0f} rpm."
        headline = "Shaft position is approaching the configured clearance boundary"
        detail = (
            f"Maximum clearance utilization is {util_max:.1f}% and minimum remaining margin is {margin_min:.1f}%."
            f"{speed_text} This suggests reduced dynamic margin or possible bearing condition change."
        )
        action = "Review boundary definition, compare with past centerline plots, and verify if the trend is repeatable."
        return {"headline": headline, "detail": detail, "action": action}

    speed_text = ""
    if first_danger_speed is not None:
        speed_text = f" Boundary overutilization tendency begins near {first_danger_speed:.0f} rpm."
    headline = "Shaft position is extremely close to the configured clearance limit"
    detail = (
        f"Maximum clearance utilization is {util_max:.1f}% and minimum remaining margin is {margin_min:.1f}%."
        f"{speed_text} The shaft centerline is operating with very low geometric margin."
    )
    action = "Treat as high priority: verify bearing clearance assumptions, inspect machine condition, and avoid repeated operation until reviewed."
    return {"headline": headline, "detail": detail, "action": action}


# =============================================================
# POLAR DIAGNOSTICS PRO (rotordynamics-based)
# =============================================================

def _build_api684_paragraph(cs, m, operating_rpm: float) -> str:
    """
    Construye un párrafo en prosa describiendo el veredicto API 684 para una
    crítica específica. Conecta el factor Q con el margen de separación y el
    cumplimiento de la norma.
    """
    if not np.isfinite(cs.q_factor):
        return (
            f"Dado que el factor de amplificación Q no fue determinable, la conformidad "
            f"con el margen de separación API 684 no puede ser evaluada de forma "
            f"automática y requiere análisis manual con datos adicionales del run-up."
        )

    if cs.q_factor < 2.5:
        return (
            f"Dado que el factor Q de {cs.q_factor:.2f} es inferior al umbral de 2.5 "
            f"establecido por API 684, el rotor está suficientemente amortiguado en esta "
            f"zona y la norma no exige margen mínimo de separación. La velocidad operativa "
            f"de {operating_rpm:.0f} rpm queda {m.actual_margin_pct:.1f}% por encima del "
            f"modo, lo cual es ampliamente superior al margen requerido."
        )

    if m.compliant:
        excess = m.actual_margin_pct - m.required_margin_pct
        return (
            f"Con factor Q de {cs.q_factor:.2f}, API 684 exige un margen mínimo de "
            f"separación de {m.required_margin_pct:.1f}% respecto a la velocidad operativa. "
            f"El margen actual es {m.actual_margin_pct:.1f}% ({excess:+.1f}% sobre el "
            f"mínimo requerido), por lo que la condición se considera conforme con la norma."
        )

    deficit = m.required_margin_pct - m.actual_margin_pct
    return (
        f"Con factor Q de {cs.q_factor:.2f}, API 684 exige un margen mínimo de separación "
        f"de {m.required_margin_pct:.1f}% respecto a la velocidad operativa. El margen "
        f"actual es solamente {m.actual_margin_pct:.1f}%, lo cual representa un déficit de "
        f"{deficit:.1f}% sobre el mínimo requerido y constituye una NO CONFORMIDAD con la "
        f"norma. El rotor opera demasiado cerca de una resonancia con factor de "
        f"amplificación significativo."
    )


def _amp_unit_to_um_pp(amp_value: float, amp_unit: str) -> Tuple[float, str]:
    """
    Convierte amplitud a µm pico-pico para evaluación ISO 20816-2.
    El note retornado expresa la amplitud en SU UNIDAD DE ORIGEN (no fuerza µm pp).

    Returns:
        (amp_in_um_pp, source_unit_note)
    """
    unit_lower = (amp_unit or "").strip().lower()
    if "mil" in unit_lower:
        return mils_to_micrometers(amp_value), f"{amp_value:.3f} mil pp"
    if "µm" in unit_lower or "um" in unit_lower:
        return float(amp_value), f"{amp_value:.1f} µm pp"
    return float(amp_value), f"{amp_value:.3f} {amp_unit}"


def _um_to_source_value(um_pp: float, amp_unit: str) -> Tuple[float, str]:
    """Convierte un valor en µm pp a la unidad de origen del CSV."""
    unit_lower = (amp_unit or "").strip().lower()
    if "mil" in unit_lower:
        return um_pp / 25.4, "mil pp"
    return um_pp, "µm pp"


def _format_iso_thresholds(iso_eval, amp_unit: str) -> str:
    """
    Formatea los umbrales ISO 20816-2 en la unidad de origen del CSV. Si la
    unidad es mil pp, agrega la equivalencia en µm pp en paréntesis para
    mantener trazabilidad con la norma original.
    """
    unit_lower = (amp_unit or "").strip().lower()
    if "mil" in unit_lower:
        ab_mil = iso_eval.boundary_AB / 25.4
        bc_mil = iso_eval.boundary_BC / 25.4
        cd_mil = iso_eval.boundary_CD / 25.4
        return (
            f"A/B={ab_mil:.2f} mil pp ({iso_eval.boundary_AB:.0f} µm pp), "
            f"B/C={bc_mil:.2f} mil pp ({iso_eval.boundary_BC:.0f} µm pp) y "
            f"C/D={cd_mil:.2f} mil pp ({iso_eval.boundary_CD:.0f} µm pp)"
        )
    return (
        f"A/B={iso_eval.boundary_AB:.0f} µm pp, "
        f"B/C={iso_eval.boundary_BC:.0f} µm pp y "
        f"C/D={iso_eval.boundary_CD:.0f} µm pp"
    )


def _format_amp_in_source(peak_amp_csv: float, amp_unit: str) -> str:
    """Formatea una amplitud en la unidad de origen del CSV."""
    unit_lower = (amp_unit or "").strip().lower()
    if "mil" in unit_lower:
        return f"{peak_amp_csv:.3f} mil pp"
    if "µm" in unit_lower or "um" in unit_lower:
        return f"{peak_amp_csv:.1f} µm pp"
    return f"{peak_amp_csv:.3f} {amp_unit}"


def build_polar_diagnostics_rotordyn(
    *,
    rpm: np.ndarray,
    amp: np.ndarray,
    phase: np.ndarray,
    operating_rpm: float = 3600.0,
    machine_group: str = "group2",
    amp_unit: str = "mil pp",
    measurement_type: str = "shaft_displacement",
    analysis_label: str = "polar",
    analysis_descriptor: str = "trayectoria polar",
    analysis_response_term: str = "respuesta polar",
    complementary_descriptor: str = "Bode amplitud-fase",
    iso_part: str = "20816-2",
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
    profile_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builder de narrativa diagnóstica para análisis rpm-amp-phase
    (Polar o Bode) basado en core.rotordynamics. Detecta velocidades
    críticas, evalúa margen API 684 y zona ISO 20816-2.

    Devuelve un diccionario compatible con el shape esperado por el
    pipeline de exportación de reportes:
        {
            "headline": str,
            "detail": str,
            "action": str,
            "structured": Dict (datos crudos para uso programático)
        }

    Args:
        rpm: array de velocidades
        amp: array de amplitudes 1X
        phase: array de fase 1X en grados
        operating_rpm: velocidad operativa nominal de la máquina
        machine_group: 'group1' o 'group2' (ISO 20816-2)
        amp_unit: unidad de amp del CSV ('mil pp', 'µm pp', etc.)
        measurement_type: 'shaft_displacement' (proximity) o 'casing_velocity'
        analysis_label: rótulo del tipo de análisis ('polar' o 'Bode')
        analysis_descriptor: nombre descriptivo ('trayectoria polar' o
            'curva Bode amplitud-fase')
        analysis_response_term: término para 'respuesta XXX' en prosa
    """
    rpm = np.asarray(rpm, dtype=float)
    amp = np.asarray(amp, dtype=float)
    phase = np.asarray(phase, dtype=float)

    criticals = detect_critical_speeds(rpm=rpm, amp=amp, phase=phase)

    # ISO 20816-2 zone para el peak global de la corrida
    peak_amp_csv = float(np.nanmax(amp)) if amp.size else 0.0
    peak_amp_um, conversion_note = _amp_unit_to_um_pp(peak_amp_csv, amp_unit)
    iso_eval = iso_20816_zone_multipart(
        amplitude=peak_amp_um,
        iso_part=iso_part,
        machine_group=machine_group,
        measurement_type=measurement_type,
        operating_speed_rpm=operating_rpm,
        custom_thresholds=custom_thresholds,
    )

    # Evaluar API 684 para cada crítica
    api_evals = []
    for cs in criticals:
        margin = evaluate_api684_margin(
            critical_rpm=cs.rpm,
            operating_rpm=operating_rpm,
            q_factor=cs.q_factor,
        )
        api_evals.append((cs, margin))

    # Determinar el caso peor
    confirmed = [(cs, m) for cs, m in api_evals if cs.phase_change_deg >= 90.0]
    candidates_only = [(cs, m) for cs, m in api_evals if cs.phase_change_deg < 90.0]
    any_non_compliant = any(not m.compliant for _, m in api_evals)
    any_danger = any(m.zone == "DANGER" for _, m in api_evals) or iso_eval.zone == "D"

    # =============================================================
    # HEADLINE
    # =============================================================
    if not criticals:
        headline = "Respuesta polar sin velocidades críticas detectadas en el rango medido"
    elif any_danger:
        worst = max(api_evals, key=lambda x: x[1].required_margin_pct - x[1].actual_margin_pct)
        cs, m = worst
        headline = (
            f"Crítica con margen API 684 insuficiente en {cs.rpm:.0f} rpm "
            f"(Q={cs.q_factor:.2f}, margen actual {m.actual_margin_pct:.1f}% vs requerido {m.required_margin_pct:.1f}%)"
        )
    elif confirmed:
        cs, m = confirmed[0]
        verdict = "conforme API 684" if m.compliant else "NO conforme API 684"
        headline = (
            f"Velocidad crítica confirmada en {cs.rpm:.0f} rpm "
            f"(Q={cs.q_factor:.2f}, Δfase={cs.phase_change_deg:.0f}°, {verdict})"
        )
    else:
        cs, m = candidates_only[0]
        headline = (
            f"Candidato dinámico en {cs.rpm:.0f} rpm con Δfase {cs.phase_change_deg:.0f}° "
            f"(no cumple criterio de crítica confirmada Δfase ≥ 90°)"
        )

    # =============================================================
    # DETAIL — prosa fluida estilo reporte de ingeniería
    # =============================================================
    paragraphs: List[str] = []

    # Párrafo 1: contexto del análisis (ISO part dinámica según profile)
    iso_label_map = {
        "20816-2": "ISO 20816-2:2017 (turbinas y generadores >40 MW con cojinetes planos)",
        "20816-3": "ISO 20816-3:2022 (máquinas industriales 15 kW–40 MW)",
        "20816-4": "ISO 20816-4:2018 (turbinas con cojinetes de rodillos)",
        "20816-7": "ISO 20816-7:2016 (bombas rotodinámicas industriales)",
        "custom": "umbrales personalizados (definidos por el usuario o por el fabricante)",
    }
    iso_label = iso_label_map.get(iso_part, f"ISO {iso_part}")

    profile_clause = (
        f" El profile activo es '{profile_label}'." if profile_label else ""
    )

    intro = (
        f"El análisis rotodinámico de la {analysis_descriptor} fue evaluado contra los "
        f"criterios de API 684 (Tutorial on Rotor Dynamics) y {iso_label}, tomando como "
        f"referencia una velocidad operativa nominal de {operating_rpm:.0f} rpm y "
        f"considerando la máquina como parte del grupo/clase '{machine_group}'.{profile_clause}"
    )
    paragraphs.append(intro)

    # Párrafo(s) 2: hallazgos rotodinámicos
    if not criticals:
        paragraphs.append(
            "El detector no identifica velocidades críticas dentro del rango medido, "
            "considerando los criterios de prominencia de amplitud, cambio de fase mínimo "
            "de 40 grados y rechazo de artefactos de frontera. Esta ausencia de modos "
            "claros en el run-up es consistente con un rotor bien amortiguado, una unidad "
            "operando en régimen supercrítico cuya primera crítica queda por debajo del "
            "rango de medición, o una corrida que no cruza modos significativos del "
            "sistema rotor-soporte."
        )
    elif len(criticals) == 1:
        cs, m = api_evals[0]
        kind = "una velocidad crítica" if cs.phase_change_deg >= 90.0 else "un candidato dinámico"
        q_clause = (
            f"con factor de amplificación Q igual a {cs.q_factor:.2f}"
            if np.isfinite(cs.q_factor)
            else "cuyo factor de amplificación Q no fue determinable"
        )
        fwhm_clause = (
            f"y un FWHM de {cs.fwhm_rpm:.0f} rpm"
            if np.isfinite(cs.fwhm_rpm)
            else "sin FWHM determinable"
        )
        paragraphs.append(
            f"El análisis identifica {kind} en {cs.rpm:.0f} rpm, {q_clause}. "
            f"La crítica presenta una amplitud máxima de {cs.amp_peak:.3f} {amp_unit} con "
            f"un cambio de fase de {cs.phase_change_deg:.0f} grados a través del pico, "
            f"confianza de detección {cs.confidence:.0%} {fwhm_clause}."
        )
        paragraphs.append(_build_api684_paragraph(cs, m, operating_rpm))
    else:
        paragraphs.append(
            f"El análisis identifica {len(criticals)} eventos rotodinámicos en el rango "
            f"medido, descritos a continuación en orden de aparición."
        )
        for idx, (cs, m) in enumerate(api_evals, start=1):
            kind = "Velocidad crítica" if cs.phase_change_deg >= 90.0 else "Candidato dinámico"
            q_clause = (
                f"factor de amplificación Q igual a {cs.q_factor:.2f}"
                if np.isfinite(cs.q_factor)
                else "factor de amplificación Q indeterminado"
            )
            fwhm_clause = (
                f"con un FWHM de {cs.fwhm_rpm:.0f} rpm"
                if np.isfinite(cs.fwhm_rpm)
                else "con FWHM indeterminado"
            )
            paragraphs.append(
                f"{kind} #{idx} aparece en {cs.rpm:.0f} rpm con {q_clause}, amplitud máxima "
                f"de {cs.amp_peak:.3f} {amp_unit}, cambio de fase de "
                f"{cs.phase_change_deg:.0f} grados {fwhm_clause} y confianza de detección "
                f"{cs.confidence:.0%}. {_build_api684_paragraph(cs, m, operating_rpm)}"
            )

    # Párrafo 3: severidad ISO (parte dinámica) en unidad de origen del CSV
    thresholds_str = _format_iso_thresholds(iso_eval, amp_unit)
    iso_short = (
        "umbrales personalizados"
        if iso_part == "custom"
        else f"ISO {iso_part}"
    )
    paragraphs.append(
        f"En cuanto a la severidad de vibración del peak global de la corrida, la amplitud "
        f"máxima medida es de {conversion_note}, valor que cae dentro de la zona "
        f"{iso_eval.zone} según {iso_short} correspondiente a "
        f"{iso_eval.zone_description.lower().rstrip('.')}. Para esta combinación de máquina "
        f"y velocidad operativa, los umbrales aplicados son {thresholds_str}."
    )

    detail = "\n\n".join(paragraphs)

    # =============================================================
    # ACTION — intro en prosa + lista numerada
    # =============================================================
    if any_danger:
        intro = (
            "Dada la severidad de la condición detectada, se requiere acción inmediata "
            "antes de continuar la operación de la unidad. Las medidas a ejecutar, en "
            "orden de prioridad, son las siguientes:"
        )
        items = [
            "Suspender la operación sostenida cerca de la velocidad crítica identificada hasta completar una evaluación rotodinámica formal.",
            f"Correlacionar la {analysis_response_term} con datos de {complementary_descriptor}, órbita filtrada 1X, shaft centerline y forma de onda en el dominio del tiempo.",
            "Verificar las condiciones reales de balance residual, alineación del tren, lubricación y temperatura de cojinetes contra los valores de comisionamiento.",
            "Evaluar la posibilidad de re-balanceo de campo si la respuesta dominante es atribuible a desbalance progresivo.",
            "Documentar el hallazgo como prioridad alta y notificar formalmente al equipo de ingeniería rotodinámica para revisión técnica.",
        ]
    elif iso_eval.zone == "C":
        intro = (
            "La amplitud de vibración medida ubica al equipo en zona C de ISO 20816-2:2017, "
            "donde la operación está permitida pero con margen dinámico reducido. Se "
            "recomiendan las siguientes acciones de seguimiento y control:"
        )
        items = [
            "Programar acción correctiva en el próximo paro planificado de la unidad.",
            "Mantener seguimiento estrecho de la tendencia de amplitud 1X y del factor de amplificación Q.",
            "Correlacionar la condición con Bode, órbita filtrada 1X y shaft centerline para confirmar la causa raíz.",
            "Verificar las condiciones reales de balance y alineación contra el registro de comisionamiento de la máquina.",
        ]
    elif confirmed and any_non_compliant:
        intro = (
            "La crítica confirmada presenta un margen de separación marginal respecto a la "
            "velocidad operativa según API 684. Las recomendaciones para confirmar la "
            "condición y prevenir su degradación son las siguientes:"
        )
        items = [
            "Evaluar si la condición de balance, alineación o rigidez de soporte ha cambiado respecto al estado de comisionamiento.",
            "Comparar la corrida actual contra el historial de corridas previas y la línea base de fábrica.",
            "Repetir el análisis de factor Q en los próximos arranques para confirmar si la tendencia de amplificación es estable o creciente.",
            "Documentar la corrida actual como referencia técnica para futuras comparaciones.",
        ]
    elif criticals:
        intro = (
            "La condición rotodinámica se encuentra dentro del margen aceptable establecido "
            "por API 684 e ISO 20816-2. A partir de los hallazgos descritos, se establecen "
            "las siguientes recomendaciones de seguimiento, ordenadas por prioridad técnica:"
        )
        items = [
            "Mantener la corrida actual como línea base de aceptación para futuras comparaciones.",
            f"Comparar la {analysis_descriptor} contra corridas históricas en los próximos arranques y paradas, vigilando migraciones del modo o del factor Q.",
            "Verificar que el factor de amplificación Q se mantenga estable en el tiempo, ya que un aumento gradual indicaría degradación del amortiguamiento del sistema rotor-soporte.",
            f"Correlacionar periódicamente con datos de {complementary_descriptor} y shaft centerline para detectar cambios de condición tempranos.",
        ]
    else:
        intro = (
            "El análisis no identifica condiciones rotodinámicas adversas dentro del rango "
            "medido. Las recomendaciones se limitan a un esquema de monitoreo continuo "
            "para preservar la trazabilidad histórica:"
        )
        items = [
            "Mantener la corrida actual como línea base de aceptación.",
            f"Verificar la repetibilidad de la {analysis_response_term} en próximos arranques y paradas.",
            "Documentar la condición observada como referencia rotodinámica para comparaciones futuras.",
        ]

    numbered_list = "\n\n".join(
        f"{idx}. {item}" for idx, item in enumerate(items, start=1)
    )
    action = f"{intro}\n\n{numbered_list}"

    # =============================================================
    # STRUCTURED (datos crudos)
    # =============================================================
    structured = {
        "operating_rpm": float(operating_rpm),
        "machine_group": machine_group,
        "measurement_type": measurement_type,
        "amp_unit_csv": amp_unit,
        "peak_amp_csv": peak_amp_csv,
        "peak_amp_um_pp": peak_amp_um,
        "iso_zone": iso_eval.zone,
        "iso_boundary_AB": iso_eval.boundary_AB,
        "iso_boundary_BC": iso_eval.boundary_BC,
        "iso_boundary_CD": iso_eval.boundary_CD,
        "criticals": [
            {
                "rpm": cs.rpm,
                "amp_peak": cs.amp_peak,
                "phase_change_deg": cs.phase_change_deg,
                "q_factor": cs.q_factor,
                "fwhm_rpm": cs.fwhm_rpm,
                "confidence": cs.confidence,
                "n1_rpm": cs.n1_rpm,
                "n2_rpm": cs.n2_rpm,
                "api684_actual_margin_pct": m.actual_margin_pct,
                "api684_required_margin_pct": m.required_margin_pct,
                "api684_compliant": m.compliant,
                "api684_zone": m.zone,
                "kind": "confirmed" if cs.phase_change_deg >= 90.0 else "candidate",
            }
            for cs, m in api_evals
        ],
    }

    return {
        "headline": headline,
        "detail": detail,
        "action": action,
        "structured": structured,
    }


# =============================================================
# POLAR COMPARE DIAGNOSTICS PRO (multi-fecha, rotordynamics-based)
# =============================================================

def _format_q_evolution(q_values: List[float]) -> str:
    """Formatea evolución de Q como '2.31 → 1.85 → 2.15'."""
    parts = [f"{q:.2f}" if np.isfinite(q) else "—" for q in q_values]
    return " → ".join(parts)


def _classify_q_trend(q_values: List[float]) -> str:
    """Clasifica tendencia del factor Q a lo largo del tiempo."""
    finite_q = [q for q in q_values if np.isfinite(q)]
    if len(finite_q) < 2:
        return "indeterminada (datos insuficientes)"

    diffs = [finite_q[i + 1] - finite_q[i] for i in range(len(finite_q) - 1)]
    all_positive = all(d > 0.05 for d in diffs)
    all_negative = all(d < -0.05 for d in diffs)
    range_q = max(finite_q) - min(finite_q)
    mean_q = sum(finite_q) / len(finite_q)
    rel_variation = range_q / mean_q if mean_q > 0 else 0

    if all_positive and rel_variation > 0.15:
        return "creciente (posible degradación de amortiguamiento)"
    if all_negative and rel_variation > 0.15:
        return "decreciente (mejor amortiguamiento aparente)"
    if rel_variation < 0.10:
        return "estable"
    return "no monotónica con dispersión moderada"


def _classify_speed_migration(rpms: List[float]) -> Tuple[str, float]:
    """Clasifica migración del modo crítico."""
    if len(rpms) < 2:
        return "indeterminada", 0.0
    rng = max(rpms) - min(rpms)
    mean_rpm = sum(rpms) / len(rpms)
    pct_var = rng / mean_rpm * 100 if mean_rpm > 0 else 0
    if pct_var < 3.0:
        return "consistente, sin migración significativa", pct_var
    if pct_var < 7.0:
        return "moderadamente dispersa, dentro de variabilidad típica de detección", pct_var
    return "con migración importante que requiere investigación", pct_var


def build_polar_compare_diagnostics_rotordyn(
    *,
    records: List[Dict[str, Any]],
    operating_rpm: float = 3600.0,
    machine_group: str = "group2",
) -> Dict[str, Any]:
    """
    Builder de narrativa diagnóstica para la comparativa multi-fecha de
    corridas polares. Cada record debe traer:

        - label: nombre del archivo
        - ts_start: timestamp inicio
        - amp_unit: unidad de amplitud
        - primary_critical: CriticalSpeed | None
        - primary_api684: API684Margin | None
        - iso_eval: ISO20816Zone
        - peak_amp_csv: float
        - peak_amp_um_pp: float

    Devuelve diccionario {headline, detail, action, structured} con
    narrativa fluida estilo reporte de ingeniería Cat IV.
    """
    import pandas as _pd  # local para evitar import top en este módulo si no está

    if not records:
        return {
            "headline": "Sin corridas válidas para comparación",
            "detail": "No se proporcionaron corridas polares con datos suficientes para análisis comparativo.",
            "action": "Cargar al menos dos corridas polares con metadatos de fecha y resultado de detección rotodinámica válido.",
            "structured": {},
        }

    ordered = sorted(
        records,
        key=lambda r: _pd.Timestamp(r.get("ts_start") or "1970-01-01"),
    )

    n = len(ordered)
    first_record = ordered[0]
    last_record = ordered[-1]

    def _fmt_date(r):
        ts = r.get("ts_start")
        if ts is not None:
            try:
                return _pd.Timestamp(ts).strftime("%d %b %Y")
            except Exception:
                pass
        return r.get("label", "—")

    first_date = _fmt_date(first_record)
    last_date = _fmt_date(last_record)

    primary_critical_rpms: List[float] = []
    primary_q_factors: List[float] = []
    peak_amps_csv: List[float] = []
    peak_amps_um: List[float] = []
    iso_zones: List[str] = []
    api_compliances: List[bool] = []
    has_critical_count = 0

    # Unidad de origen del primer record (asumimos misma máquina = misma unidad)
    common_amp_unit = ordered[0].get("amp_unit", "mil pp")

    for r in ordered:
        cs = r.get("primary_critical")
        if cs is not None:
            has_critical_count += 1
            primary_critical_rpms.append(float(cs.rpm))
            primary_q_factors.append(
                float(cs.q_factor) if np.isfinite(cs.q_factor) else float("nan")
            )
        peak_amps_csv.append(float(r.get("peak_amp_csv", 0.0)))
        peak_amps_um.append(float(r.get("peak_amp_um_pp", 0.0)))
        zone_obj = r.get("iso_eval")
        iso_zones.append(zone_obj.zone if zone_obj is not None else "—")
        api_eval = r.get("primary_api684")
        if api_eval is not None:
            api_compliances.append(bool(api_eval.compliant))

    unique_zones = set(iso_zones) - {"—"}
    if len(unique_zones) == 1:
        zone_clause = f"zona {next(iter(unique_zones))} ISO 20816-2 estable"
    elif len(unique_zones) > 1:
        zone_clause = f"zonas ISO {' / '.join(sorted(unique_zones))} (cambio de severidad)"
    else:
        zone_clause = "severidad ISO no determinada"

    headline = (
        f"Comparativa multi-fecha de {n} corridas entre {first_date} y {last_date} — "
        f"{has_critical_count} críticas detectadas, {zone_clause}"
    )

    paragraphs: List[str] = []

    paragraphs.append(
        f"Se analizó la evolución de la respuesta polar del rotor a lo largo de {n} corridas "
        f"comprendidas entre {first_date} y {last_date}, evaluando la migración del modo "
        f"crítico, el factor de amplificación Q, la severidad de vibración según "
        f"ISO 20816-2:2017 y la conformidad con los criterios de margen de separación de "
        f"API 684. La velocidad operativa de referencia para todas las corridas es "
        f"{operating_rpm:.0f} rpm y la clasificación corresponde al "
        f"{'Grupo 2 (configuración shaft mounted, típica de turbogeneradores con cojinetes planos)' if machine_group == 'group2' else 'Grupo 1 (configuración cantilever)'}."
    )

    if primary_critical_rpms:
        migration_class, pct_var = _classify_speed_migration(primary_critical_rpms)
        rpms_str = ", ".join(f"{rpm:.0f} rpm" for rpm in primary_critical_rpms)
        rpm_min = min(primary_critical_rpms)
        rpm_max = max(primary_critical_rpms)
        delta_rpm = rpm_max - rpm_min
        paragraphs.append(
            f"La velocidad del primer modo crítico se ubicó en {rpms_str} para las corridas "
            f"analizadas en orden cronológico. La variación máxima observada es de "
            f"{delta_rpm:.0f} rpm ({pct_var:.1f}% respecto al valor medio), lo cual califica "
            f"como una migración {migration_class}."
        )
    else:
        paragraphs.append(
            "Ninguna de las corridas analizadas muestra una velocidad crítica detectable bajo "
            "los criterios automáticos. Esto puede indicar que el rotor opera en régimen "
            "supercrítico, que está bien amortiguado, o que el rango de medición no cubre el "
            "primer modo. La comparación se concentra entonces en la evolución de severidad."
        )

    if primary_q_factors and any(np.isfinite(q) for q in primary_q_factors):
        q_evolution_str = _format_q_evolution(primary_q_factors)
        q_trend = _classify_q_trend(primary_q_factors)
        finite_q = [q for q in primary_q_factors if np.isfinite(q)]
        max_q = max(finite_q) if finite_q else float("nan")
        min_q = min(finite_q) if finite_q else float("nan")

        if max_q < 2.5:
            q_threshold_clause = (
                "En todas las fechas el factor Q permanece por debajo del umbral de 2.5 "
                "establecido por API 684, por lo que la norma no exige margen mínimo de "
                "separación respecto a la velocidad operativa en ninguna de las corridas "
                "analizadas."
            )
        elif min_q < 2.5 <= max_q:
            q_threshold_clause = (
                "El factor Q cruza el umbral de 2.5 establecido por API 684 entre corridas, "
                "lo cual implica que el requerimiento de margen mínimo de separación cambia "
                "entre fechas y debe verificarse caso por caso."
            )
        else:
            q_threshold_clause = (
                "En todas las fechas el factor Q supera el umbral de 2.5 establecido por "
                "API 684, por lo que se aplica un requerimiento de margen mínimo de separación "
                "que debe evaluarse en cada corrida."
            )

        paragraphs.append(
            f"El factor de amplificación Q evolucionó de la siguiente forma: {q_evolution_str}. "
            f"La tendencia es {q_trend}. {q_threshold_clause}"
        )

    if peak_amps_csv:
        # Mostrar amplitudes en la unidad de origen del CSV
        amps_str = " → ".join(_format_amp_in_source(a, common_amp_unit) for a in peak_amps_csv)
        zones_str = " / ".join(iso_zones)
        zone_eval = ordered[0].get("iso_eval")
        boundaries_text = (
            f"Para esta combinación de máquina y velocidad, los umbrales aplicados son "
            f"{_format_iso_thresholds(zone_eval, common_amp_unit)}."
        ) if zone_eval is not None else ""

        if len(unique_zones) == 1:
            zone_summary = (
                f"En todas las corridas la amplitud máxima se mantuvo en zona "
                f"{next(iter(unique_zones))} de ISO 20816-2, sin cambio de clasificación "
                f"de severidad."
            )
        elif len(unique_zones) > 1:
            zone_summary = (
                f"La clasificación de severidad ISO 20816-2 cambió entre corridas: zonas "
                f"{zones_str} en orden cronológico. Este es un indicador relevante de cambio "
                f"de condición."
            )
        else:
            zone_summary = ""

        paragraphs.append(
            f"La amplitud máxima del peak global de cada corrida evolucionó {amps_str} en "
            f"orden cronológico. {zone_summary} {boundaries_text}".strip()
        )

    n_compliant = sum(api_compliances)
    if api_compliances:
        if n_compliant == len(api_compliances):
            verdict = (
                f"En síntesis, las {n} corridas son conformes con API 684 y la condición "
                f"rotodinámica del rotor se considera estable y dentro de los márgenes "
                f"aceptables para operación continua."
            )
        elif n_compliant > 0:
            verdict = (
                f"En síntesis, {n_compliant} de las {n} corridas son conformes con API 684. "
                f"La presencia de corridas no conformes en el conjunto requiere atención e "
                f"investigación de las causas del cambio de condición."
            )
        else:
            verdict = (
                f"En síntesis, ninguna de las {n} corridas analizadas cumple el margen de "
                f"separación requerido por API 684. Se trata de un hallazgo de alta prioridad "
                f"que requiere acción técnica formal."
            )
        paragraphs.append(verdict)

    detail = "\n\n".join(paragraphs)

    intro = (
        "A partir del análisis comparativo descrito, se establecen las siguientes "
        "recomendaciones de seguimiento, ordenadas por prioridad técnica:"
    )

    if has_critical_count == n and len(unique_zones) <= 1 and n_compliant == len(api_compliances) and api_compliances:
        items = [
            f"Adoptar la corrida más reciente ({last_date}) como línea base de aceptación actualizada para futuras comparaciones.",
            "Mantener la frecuencia actual de medición y comparar los próximos arranques contra la línea base actualizada.",
            "Continuar el seguimiento del factor Q en cada nueva corrida, registrando cualquier tendencia monotónica hacia valores mayores como señal temprana de degradación de amortiguamiento.",
            "Correlacionar la evolución polar con datos de Bode amplitud-fase, shaft centerline y temperatura de cojinetes para detectar cambios de condición tempranos antes de que afecten la zona ISO.",
        ]
    elif len(unique_zones) > 1:
        items = [
            "Investigar las causas del cambio de zona ISO 20816-2 entre corridas, revisando registros de operación, mantenimiento y condiciones de proceso.",
            "Verificar consistencia de las condiciones de medición entre fechas (carga, temperatura, condición de balance, alineación, lubricación de cojinetes).",
            "Comparar contra órbitas filtradas 1X y shaft centerline para confirmar si el cambio observado es genuino o atribuible a cambio de condiciones operativas.",
            "Si el cambio es genuino y la tendencia es desfavorable, programar evaluación rotodinámica formal y considerar acción correctiva.",
            "Documentar el cambio detectado como hallazgo trazable con sus condiciones contextuales.",
        ]
    elif api_compliances and n_compliant < len(api_compliances):
        items = [
            "Priorizar las corridas no conformes para análisis detallado caso por caso.",
            "Verificar si el aumento del factor Q corresponde a degradación real del amortiguamiento o a artefactos de detección.",
            "Correlacionar con condición de cojinetes, temperaturas de aceite y registro de eventos operativos.",
            "Considerar análisis modal experimental o re-cálculo rotodinámico con coeficientes dinámicos actualizados de los cojinetes.",
        ]
    else:
        items = [
            "Mantener la corrida más reciente como referencia rotodinámica del activo.",
            "Verificar repetibilidad de las próximas corridas durante arranques y paradas.",
            "Documentar la condición observada para trazabilidad histórica.",
        ]

    numbered_list = "\n\n".join(
        f"{idx}. {item}" for idx, item in enumerate(items, start=1)
    )
    action = f"{intro}\n\n{numbered_list}"

    structured = {
        "n_corridas": n,
        "first_date": first_date,
        "last_date": last_date,
        "operating_rpm": operating_rpm,
        "machine_group": machine_group,
        "amp_unit": common_amp_unit,
        "primary_critical_rpms": primary_critical_rpms,
        "primary_q_factors": primary_q_factors,
        "peak_amps_csv": peak_amps_csv,
        "peak_amps_um": peak_amps_um,
        "iso_zones": iso_zones,
        "api684_compliances": api_compliances,
        "n_compliant": n_compliant,
        "n_with_critical": has_critical_count,
        "ordered_records": ordered,  # para que el caller pueda construir el resumen prosa
    }

    return {
        "headline": headline,
        "detail": detail,
        "action": action,
        "structured": structured,
    }


# =============================================================
# BODE DIAGNOSTICS — wrapper que reusa la maquinaria del Polar
# con redacción Cat IV específica del análisis Bode amplitud/fase
# =============================================================

def build_bode_diagnostics_rotordyn(
    *,
    rpm: np.ndarray,
    amp: np.ndarray,
    phase: np.ndarray,
    operating_rpm: float = 3600.0,
    machine_group: str = "group2",
    amp_unit: str = "µm pp",
    measurement_type: str = "shaft_displacement",
    iso_part: str = "20816-2",
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
    profile_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builder de narrativa diagnóstica para Bode Plot. Delega en
    build_polar_diagnostics_rotordyn (misma matemática rotodinámica)
    pero con la redacción adaptada al contexto del Bode amplitud-fase
    contra RPM.
    """
    return build_polar_diagnostics_rotordyn(
        rpm=rpm,
        amp=amp,
        phase=phase,
        operating_rpm=operating_rpm,
        machine_group=machine_group,
        amp_unit=amp_unit,
        measurement_type=measurement_type,
        analysis_label="Bode",
        analysis_descriptor="curva Bode amplitud-fase",
        analysis_response_term="respuesta del Bode",
        complementary_descriptor="trayectoria polar",
        iso_part=iso_part,
        custom_thresholds=custom_thresholds,
        profile_label=profile_label,
    )


def build_bode_compare_diagnostics_rotordyn(
    *,
    records: List[Dict[str, Any]],
    operating_rpm: float = 3600.0,
    machine_group: str = "group2",
) -> Dict[str, Any]:
    """
    Comparativa multi-fecha de corridas Bode. Reusa la maquinaria del
    Polar compare con string-substitution mínima para narrativa Bode.
    """
    result = build_polar_compare_diagnostics_rotordyn(
        records=records,
        operating_rpm=operating_rpm,
        machine_group=machine_group,
    )

    swaps = [
        ("respuesta polar del rotor", "respuesta Bode del rotor"),
        ("corridas polares", "corridas Bode"),
        ("trayectoria polar", "curva Bode"),
        ("respuesta polar", "respuesta del Bode"),
        ("la corrida polar", "la corrida Bode"),
    ]

    for key in ("headline", "detail", "action"):
        text = result.get(key, "")
        for old, new in swaps:
            text = text.replace(old, new)
        result[key] = text

    return result


# =============================================================
# SHAFT CENTERLINE DIAGNOSTICS PRO (Cat IV / API 670)
# =============================================================

def build_scl_diagnostics_rotordyn(
    *,
    eccentricity_state,  # EccentricityState
    operating_rpm: float,
    profile_label: str = "",
    bearing_inner_diameter_mm: Optional[float] = None,
    diametral_clearance_mm: Optional[float] = None,
    clearance_source: str = "",
    babbitt_material: Optional[str] = None,
    last_rebabbiting_date: Optional[str] = None,
    document_reference: Optional[str] = None,
    lift_off_rpm: Optional[float] = None,
    migration: Optional[Any] = None,  # CenterlineMigration
    amp_unit: str = "mil pp",
    clearance_reference_frame: str = "",
    bearing_center_x: Optional[float] = None,
    bearing_center_y: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Builder de narrativa diagnóstica Cat IV para Shaft Centerline.
    Integra dato físico del cojinete (Vault), eccentricity ratio, attitude
    angle, lift-off y migración multi-fecha en prosa fluida con
    referencias a API 670 e ISO 20816-2.

    Args:
        eccentricity_state: EccentricityState con e/c, attitude angle,
            classification ya calculados.
        operating_rpm: velocidad operativa nominal.
        profile_label: nombre del asset profile activo.
        bearing_inner_diameter_mm, diametral_clearance_mm: datos del Vault.
        clearance_source: origen del clearance (OEM, calculado, estimado).
        babbitt_material: material babbitt si está documentado.
        last_rebabbiting_date: fecha última intervención (ISO).
        document_reference: título del documento del Vault que respalda
            los datos (ej. "Reporte Wersin oct 2018").
        lift_off_rpm: velocidad de lift-off detectada en la corrida.
        migration: CenterlineMigration si hay comparación entre fechas.

    Returns:
        Dict {headline, detail, action} estilo reporte Cat IV.
    """
    es = eccentricity_state
    paragraphs: List[str] = []

    # Headline
    e_c = es.eccentricity_ratio
    if es.classification == "HEALTHY":
        headline = (
            f"Operación hidrodinámica estable a {operating_rpm:.0f} rpm con "
            f"eccentricity ratio e/c = {e_c:.2f} (zona sana 0.40–0.70)"
        )
    elif es.classification == "WHIRL_RISK":
        headline = (
            f"Eccentricity ratio bajo (e/c = {e_c:.2f}) — riesgo de oil whirl"
        )
    elif es.classification == "WIPE_RISK":
        headline = (
            f"Eccentricity ratio crítico (e/c = {e_c:.2f}) — riesgo de wipe / contacto babbitt"
        )
    elif es.classification == "MARGINAL_HIGH":
        headline = (
            f"Eccentricity ratio elevado (e/c = {e_c:.2f}) — margen reducido al límite del clearance"
        )
    else:
        headline = (
            f"Eccentricity ratio bajo-marginal (e/c = {e_c:.2f}) — vigilar subsíncronos"
        )

    # Párrafo 1: contexto del análisis y dato físico del cojinete
    profile_clause = f" El profile activo es '{profile_label}'." if profile_label else ""

    if bearing_inner_diameter_mm and diametral_clearance_mm:
        physical_clause = (
            f" Las dimensiones físicas del cojinete fueron tomadas del Document "
            f"Vault del activo ({clearance_source or 'fuente no especificada'}): "
            f"diámetro interno {bearing_inner_diameter_mm:.2f} mm y clearance "
            f"diametral {diametral_clearance_mm*1000:.0f} µm "
            f"({diametral_clearance_mm/0.0254:.2f} mil pp)."
        )
    elif bearing_inner_diameter_mm:
        physical_clause = (
            f" El diámetro interno del cojinete ({bearing_inner_diameter_mm:.2f} mm) "
            f"fue extraído del Document Vault. El clearance diametral se estimó "
            f"heurísticamente porque el dato OEM directo no está disponible "
            f"({clearance_source or 'estimación típica'})."
        )
    else:
        physical_clause = (
            " Las dimensiones físicas del cojinete no están registradas en el "
            "Document Vault del activo; la evaluación usa valores por defecto. "
            "Recomendamos cargar el manual OEM o el reporte de inspección "
            "para mejorar la precisión del cálculo."
        )

    babbitt_clause = ""
    if babbitt_material and last_rebabbiting_date:
        babbitt_clause = (
            f" El cojinete está revestido con babbitt {babbitt_material} (última "
            f"intervención registrada: {last_rebabbiting_date})."
        )
    elif babbitt_material:
        babbitt_clause = f" El cojinete está revestido con babbitt {babbitt_material}."

    doc_clause = ""
    if document_reference:
        doc_clause = f" Documento de referencia: {document_reference}."

    # Cláusula del marco de referencia (API 670 / práctica estándar de
    # cojinetes hidrodinámicos). Sin nombrar marcas comerciales — la convención
    # es estándar de la industria, no propietaria.
    frame_clause = ""
    if (
        clearance_reference_frame.lower().startswith("bottom load")
        or clearance_reference_frame.lower().startswith("bearing center reference")
    ):
        if bearing_center_y is not None and abs(float(bearing_center_y)) > 1e-6:
            frame_clause = (
                f" El bearing center geométrico se posiciona según práctica "
                f"estándar para cojinetes hidrodinámicos (referencia API 670) en "
                f"(0, {float(bearing_center_y):+.3f}) {amp_unit}, tomando el origen (0,0) "
                f"del registro como la posición de reposo del muñón apoyado sobre "
                f"la babbitt al fondo del cojinete por gravedad. Esto representa el "
                f"clearance circle alrededor del centro geométrico real del cojinete "
                f"y no del punto de medición en reposo, lo que es condición necesaria "
                f"para que el eccentricity ratio y el attitude angle resulten "
                f"físicamente correctos."
            )
        else:
            frame_clause = (
                f" El marco de referencia del clearance sigue la práctica estándar "
                f"para cojinetes hidrodinámicos (referencia API 670), con el bearing "
                f"center geométrico tomado como origen para el cálculo de "
                f"eccentricity ratio y attitude angle."
            )
    elif clearance_reference_frame:
        frame_clause = f" Marco de referencia del clearance: {clearance_reference_frame}."

    paragraphs.append(
        f"El análisis de Shaft Centerline evalúa la posición DC del muñón "
        f"dentro del cojinete plano hidrodinámico siguiendo prácticas Cat IV "
        f"y los criterios de API 670 para instrumentación con sondas de "
        f"proximidad.{profile_clause}{physical_clause}{babbitt_clause}{doc_clause}{frame_clause}"
    )

    # Párrafo 2: eccentricity ratio + interpretación
    paragraphs.append(
        f"A velocidad operativa de {operating_rpm:.0f} rpm el muñón se ubica "
        f"en posición ({es.x_pos:+.3f}, {es.y_pos:+.3f}) {amp_unit}, lo que "
        f"corresponde a un eccentricity ratio de {e_c:.3f} respecto al "
        f"clearance radial promedio ({(es.cx_radial + es.cy_radial)/2.0:.3f} {amp_unit}). "
        f"{es.classification_text}"
    )

    # Párrafo 3: attitude angle
    if es.attitude_angle_deg > 0:
        if 25.0 <= es.attitude_angle_deg <= 55.0:
            attitude_clause = (
                f"El attitude angle de {es.attitude_angle_deg:.1f}° se ubica en el "
                f"rango típico (25°–55°) para cojinetes planos hidrodinámicos en "
                f"carga vertical, lo que indica condición de operación normal."
            )
        elif es.attitude_angle_deg < 25.0:
            attitude_clause = (
                f"El attitude angle de {es.attitude_angle_deg:.1f}° es inferior al "
                f"rango típico (25°–55°), lo que sugiere alta carga estática o "
                f"clearance reducido. Verificar carga del rotor y condición del "
                f"babbitt."
            )
        else:
            attitude_clause = (
                f"El attitude angle de {es.attitude_angle_deg:.1f}° excede el "
                f"rango típico (25°–55°), lo que sugiere baja carga relativa al "
                f"clearance, o cambio en la dirección de carga (desalineación, "
                f"desbalance térmico). Verificar alineación del tren."
            )
        paragraphs.append(attitude_clause)

    # Párrafo 4: lift-off
    if lift_off_rpm is not None and lift_off_rpm > 0:
        margin_pct = (operating_rpm - lift_off_rpm) / operating_rpm * 100.0
        paragraphs.append(
            f"La velocidad de lift-off (transición del régimen de contacto babbitt "
            f"a régimen hidrodinámico completo) se estima en {lift_off_rpm:.0f} rpm, "
            f"lo que deja un margen del {margin_pct:.1f}% respecto a la velocidad "
            f"operativa. Un margen sano para cojinetes de turbogeneradores grandes "
            f"está típicamente entre 80% y 95%."
        )

    # Párrafo 5: migración entre fechas si aplica
    if migration is not None:
        paragraphs.append(migration.narrative)

    detail = "\n\n".join(paragraphs)

    # Action items
    intro = (
        "A partir del análisis del centerline, se establecen las siguientes "
        "recomendaciones de seguimiento, ordenadas por prioridad técnica:"
    )

    if es.classification == "HEALTHY":
        items = [
            "Mantener la corrida actual como línea base de aceptación del centerline para futuras comparaciones.",
            "Vigilar la estabilidad del eccentricity ratio en próximos arranques; un cambio sostenido > 0.10 amerita investigación.",
            "Correlacionar periódicamente con datos de Polar/Bode 1X y temperatura de cojinetes para detectar cambios tempranos de condición.",
            "Si la última fecha de rebabbitado supera 5 años o se acerca a la frecuencia recomendada por OEM, programar inspección preventiva.",
        ]
    elif es.classification == "WHIRL_RISK":
        items = [
            "Verificar el espectro de la sonda en la zona subsíncrona (0.40X–0.50X) para confirmar o descartar oil whirl real.",
            "Revisar carga estática efectiva del rotor — un e/c bajo puede indicar pérdida de carga o exceso de presión de aceite.",
            "Confirmar viscosidad del aceite operativo contra el grado especificado por OEM.",
            "Si el oil whirl se confirma, evaluar reducción de clearance o cambio de geometría de cojinete (lobed, tilting pad).",
        ]
    elif es.classification == "WIPE_RISK":
        items = [
            "PRIORIDAD ALTA: limitar operación sostenida hasta confirmar estado del babbitt por inspección visual o ensayo no destructivo.",
            "Verificar temperatura de salida del cojinete contra el límite de babbitt (típico 130 °C para grado 2 ASTM B-23).",
            "Confirmar carga real del rotor y descartar transient de arranque agresivo.",
            "Si la condición persiste, programar paro para inspección directa del babbitt.",
            "Documentar como hallazgo crítico y notificar al equipo de ingeniería rotodinámica.",
        ]
    elif es.classification == "MARGINAL_HIGH":
        items = [
            "Vigilar de cerca la temperatura del cojinete y la presión de salida del aceite.",
            "Programar tendencia detallada del eccentricity ratio en próximas corridas.",
            "Verificar si el clearance ha sido reducido por desgaste comparando contra el dato OEM o el último reporte de rebabbitado.",
            "Revisar carga real del rotor; un e/c alto puede indicar incremento de carga estática.",
        ]
    else:
        items = [
            "Mantener seguimiento estrecho del eccentricity ratio en próximos arranques.",
            "Vigilar el espectro subsíncrono para descartar oil whirl incipiente.",
            "Correlacionar con condición de Polar/Bode y datos de temperatura.",
        ]

    if migration is not None and migration.classification in ("MODERATE_DRIFT", "MAJOR_DRIFT"):
        items.append(
            "Investigar causas de la migración del centerline detectada entre fechas: "
            "comparar contra eventos de mantenimiento, cambios de carga o "
            "modificaciones de proceso registrados en el activo."
        )

    numbered_list = "\n\n".join(
        f"{idx}. {item}" for idx, item in enumerate(items, start=1)
    )
    action = f"{intro}\n\n{numbered_list}"

    return {
        "headline": headline,
        "detail": detail,
        "action": action,
        "structured": {
            "operating_rpm": operating_rpm,
            "eccentricity_ratio": e_c,
            "attitude_angle_deg": es.attitude_angle_deg,
            "classification": es.classification,
            "lift_off_rpm": lift_off_rpm,
            "bearing_inner_diameter_mm": bearing_inner_diameter_mm,
            "diametral_clearance_mm": diametral_clearance_mm,
        },
    }
