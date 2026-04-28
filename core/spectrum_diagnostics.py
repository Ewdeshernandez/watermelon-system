from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SAFE_COLOR = "#16a34a"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR = "#dc2626"
CAUTION_COLOR = "#84cc16"
CRITICAL_COLOR = "#dc2626"


def build_spectrum_report_notes(text_diag: Dict[str, Any]) -> str:
    title = str(text_diag.get("title", "") or "").strip()
    headline = str(text_diag.get("headline", "") or "").strip()
    narrative = str(text_diag.get("narrative", "") or "").strip()

    blocks = []
    if title:
        blocks.append(title)
    elif headline:
        blocks.append(headline)

    if narrative:
        blocks.append(narrative)

    return "\n\n".join(blocks).strip()


def _amp_of_order(harmonics: List[dict], order: int) -> float:
    for h in harmonics:
        if int(h.get("order", -1)) == int(order):
            try:
                return float(h.get("amp_peak", 0.0))
            except Exception:
                return 0.0
    return 0.0


def _max_order_from(harmonics: List[dict], min_order: int) -> float:
    vals = []
    for h in harmonics:
        try:
            order = int(h.get("order", -1))
            amp = float(h.get("amp_peak", 0.0))
            if order >= min_order:
                vals.append(amp)
        except Exception:
            pass
    return max(vals) if vals else 0.0


def _harmonic_count_above(harmonics: List[dict], threshold: float) -> int:
    c = 0
    for h in harmonics:
        try:
            if float(h.get("amp_peak", 0.0)) >= threshold:
                c += 1
        except Exception:
            pass
    return c


def _append_bearing_text(narrative: str, bearing_text: Optional[str]) -> str:
    extra = str(bearing_text or "").strip()
    if not extra:
        return narrative
    return f"{narrative}\n\n{extra}"


def evaluate_spectrum_diagnostic(
    *,
    one_x_amp: Optional[float],
    harmonics: List[dict],
    overall_spec_rms: Optional[float],
    dominant_peak_freq_cpm: Optional[float],
    dominant_peak_amp: Optional[float],
    rpm: Optional[float],
    bearing_text: Optional[str] = None,
) -> Dict[str, str]:
    one_x = float(one_x_amp) if one_x_amp is not None else 0.0
    overall = float(overall_spec_rms) if overall_spec_rms is not None else 0.0
    dom_freq = float(dominant_peak_freq_cpm) if dominant_peak_freq_cpm is not None else 0.0
    dom_amp = float(dominant_peak_amp) if dominant_peak_amp is not None else 0.0

    two_x = _amp_of_order(harmonics, 2)
    three_x = _amp_of_order(harmonics, 3)
    high_harm = _max_order_from(harmonics, 4)

    ref = max(one_x, dom_amp, 1e-9)
    ratio_2x = two_x / ref
    ratio_3x = three_x / ref
    ratio_high = high_harm / ref

    significant_threshold = ref * 0.25
    strong_harmonic_count = _harmonic_count_above(harmonics, significant_threshold)

    near_1x = False
    if rpm is not None and rpm > 0 and dom_freq > 0:
        near_1x = abs(dom_freq - float(rpm)) <= max(0.08 * float(rpm), 60.0)

    status = "SAFE"
    color = SAFE_COLOR

    if strong_harmonic_count >= 4 and ratio_high >= 0.20:
        status = "DANGER"
        color = DANGER_COLOR
        headline = "Posible holgura mecánica"
        narrative = (
            "El espectro presenta múltiples componentes armónicas con contenido significativo en armónicos altos, "
            "comportamiento consistente con holgura mecánica o no linealidad estructural. "
            "Se recomienda inspeccionar rigidez de soportes, condición de anclajes, pedestal o base, "
            "y contrastar con la forma de onda para buscar impactos o modulación."
        )
        return {
            "status": status,
            "color": color,
            "headline": headline,
            "narrative": _append_bearing_text(narrative, bearing_text),
        }

    if ratio_2x >= 0.35 or (ratio_2x >= 0.25 and ratio_3x >= 0.18):
        status = "WARNING"
        color = WARNING_COLOR
        headline = "Posible desalineación"
        narrative = (
            "La componente 2X es significativa respecto a la componente sincrónica, con posible aporte de armónicos superiores, "
            "lo que puede indicar desalineación o comportamiento asociado al tren de potencia. "
            "Se recomienda revisar alineación, condición del acople y correlacionar con mediciones radiales, axiales y análisis de fase."
        )
        return {
            "status": status,
            "color": color,
            "headline": headline,
            "narrative": _append_bearing_text(narrative, bearing_text),
        }

    if one_x > 0 and near_1x and ratio_2x < 0.25 and ratio_high < 0.15:
        status = "WARNING"
        color = WARNING_COLOR
        headline = "Posible desbalance"
        narrative = (
            "El espectro está dominado por una componente 1X fuerte cercana a la velocidad de giro, con bajo contenido relativo de armónicos, "
            "comportamiento consistente con una condición tipo desbalance. "
            "Se recomienda verificar condición de balanceo, revisar consistencia de fase entre arranques "
            "y correlacionar con los módulos Polar y Bode antes de definir una intervención."
        )
        return {
            "status": status,
            "color": color,
            "headline": headline,
            "narrative": _append_bearing_text(narrative, bearing_text),
        }

    if overall > 0 and dom_amp > 0 and (overall / max(dom_amp, 1e-9)) > 0.65 and strong_harmonic_count <= 2:
        status = "WARNING"
        color = WARNING_COLOR
        headline = "Energía de banda ancha detectada"
        narrative = (
            "La energía global del espectro está elevada respecto al pico discreto dominante, "
            "lo que sugiere contenido de banda ancha y no una firma puramente sincrónica. "
            "Se recomienda revisar condición de proceso, excitación inducida por flujo, "
            "posible cavitación, turbulencia o fuentes no estacionarias."
        )
        return {
            "status": status,
            "color": color,
            "headline": headline,
            "narrative": _append_bearing_text(narrative, bearing_text),
        }

    headline = "Sin patrón anormal dominante"
    narrative = (
        "El espectro actual no muestra una firma armónica fuerte claramente asociada con desbalance, "
        "desalineación o holgura mecánica. Se recomienda continuar el monitoreo y correlacionar con forma de onda, "
        "Bode, Polar y condición operativa antes de concluir un mecanismo de falla."
    )
    return {
        "status": status,
        "color": color,
        "headline": headline,
        "narrative": _append_bearing_text(narrative, bearing_text),
    }


# =============================================================
# CICLO 11 — DETECTORES CAT IV ADICIONALES
# =============================================================
# Los siguientes detectores complementan evaluate_spectrum_diagnostic
# (que ya cubre desbalance, desalineación, holgura y banda ancha) con
# las firmas Cat IV que faltaban: subsincrónicos (oil whirl / whip /
# rub), bandas laterales (engranaje, modulación) y resonancia.


def detect_subsynchronous(
    freq_cpm: np.ndarray,
    amp_peak: np.ndarray,
    rpm: float,
    *,
    relative_amplitude_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Busca picos significativos en bandas subsincrónicas relativas a la
    velocidad de giro:

      - 0.40X – 0.50X → **oil whirl** (precesión sub-síncrona típica de
        cojinetes hidrodinámicos planos con alto clearance / poca carga).
      - 0.50X – 0.95X → posible whip o rub (más severo).
      - <0.40X → muy bajo, posible whirl agresivo o resonancia
        modal sub-rotacional.

    Devuelve {classification, peak_freq_cpm, peak_amp, ratio, narrative}.
    Si no hay pico significativo, classification = 'NONE'.
    """
    if rpm is None or rpm <= 0:
        return {"classification": "NONE", "peak_freq_cpm": 0.0,
                "peak_amp": 0.0, "ratio": 0.0,
                "narrative": "Sin RPM válido — no se evaluó banda subsincrónica."}

    if freq_cpm.size == 0 or amp_peak.size == 0:
        return {"classification": "NONE", "peak_freq_cpm": 0.0,
                "peak_amp": 0.0, "ratio": 0.0,
                "narrative": "Sin datos espectrales — no se evaluó banda subsincrónica."}

    # Ventana sub-sincrónica: desde 0.20X hasta 0.95X
    fmin = 0.20 * rpm
    fmax = 0.95 * rpm
    mask = (freq_cpm >= fmin) & (freq_cpm <= fmax) & np.isfinite(amp_peak)
    if not np.any(mask):
        return {"classification": "NONE", "peak_freq_cpm": 0.0,
                "peak_amp": 0.0, "ratio": 0.0,
                "narrative": "Banda sub-sincrónica vacía — sin contenido espectral medible."}

    sub_freqs = freq_cpm[mask]
    sub_amps = amp_peak[mask]

    # Pico dominante en la sub-banda
    idx_peak = int(np.argmax(sub_amps))
    peak_f = float(sub_freqs[idx_peak])
    peak_a = float(sub_amps[idx_peak])

    # Referencia: amplitud máxima global del espectro (incluyendo 1X)
    ref_amp = float(np.nanmax(amp_peak))
    ratio = peak_a / max(ref_amp, 1e-9)

    if ratio < relative_amplitude_threshold:
        return {
            "classification": "NONE",
            "peak_freq_cpm": peak_f,
            "peak_amp": peak_a,
            "ratio": ratio,
            "narrative": (
                f"El pico sub-sincrónico más alto se observa en {peak_f:.0f} CPM "
                f"({peak_f/rpm:.3f}X) con amplitud relativa {ratio*100:.1f}% del "
                f"pico máximo del espectro, por debajo del umbral de "
                f"significancia ({relative_amplitude_threshold*100:.0f}%). "
                f"No se considera firma sub-sincrónica relevante."
            ),
        }

    rel_freq = peak_f / rpm

    if 0.40 <= rel_freq <= 0.50:
        classification = "OIL_WHIRL"
        narrative = (
            f"Se detecta componente sub-sincrónica significativa en {peak_f:.0f} CPM "
            f"({rel_freq:.3f}X), amplitud relativa {ratio*100:.1f}% del pico máximo. "
            f"La frecuencia coincide con la banda típica de **oil whirl** "
            f"(0.40X–0.50X) en cojinetes hidrodinámicos planos. Mecanismo: el "
            f"film de aceite circula en el cojinete a aproximadamente la mitad "
            f"de la velocidad del rotor, induciendo precesión sub-síncrona. "
            f"Causas comunes: clearance excesivo, baja carga estática, "
            f"viscosidad incorrecta del aceite, o eccentricity ratio bajo "
            f"(e/c < 0.40)."
        )
    elif 0.50 < rel_freq <= 0.95:
        classification = "OIL_WHIP_OR_RUB"
        narrative = (
            f"Se detecta componente sub-sincrónica en {peak_f:.0f} CPM "
            f"({rel_freq:.3f}X), amplitud relativa {ratio*100:.1f}%. La "
            f"frecuencia está por encima del rango típico de oil whirl pero "
            f"por debajo de 1X, configuración que puede asociarse a "
            f"**oil whip** (whirl bloqueado por una resonancia modal) o "
            f"**rub** intermitente con cojinete o sello. Severidad alta — "
            f"requiere verificación en órbita filtrada, fase y forma de onda."
        )
    else:  # 0.20–0.40X
        classification = "DEEP_SUBSYNC"
        narrative = (
            f"Se detecta componente en {peak_f:.0f} CPM ({rel_freq:.3f}X) con "
            f"amplitud relativa {ratio*100:.1f}%. Pico muy por debajo de oil "
            f"whirl convencional, posible asociación con resonancia modal "
            f"sub-rotacional, fenómeno de proceso (cavitación, turbulencia) "
            f"o falla estructural de soporte. Requiere análisis modal y "
            f"correlación con datos de proceso."
        )

    return {
        "classification": classification,
        "peak_freq_cpm": peak_f,
        "peak_amp": peak_a,
        "ratio": ratio,
        "narrative": narrative,
    }


def detect_resonance_at_1x(
    freq_cpm: np.ndarray,
    amp_peak: np.ndarray,
    rpm: float,
    *,
    width_threshold_pct: float = 5.0,
    min_height_ratio: float = 0.50,
) -> Dict[str, Any]:
    """
    Detecta si el pico de 1X tiene una banda ancha (forma de campana
    significativa) — indicador de resonancia o paso por velocidad
    crítica. Mide el ancho del pico al 50% de su altura (FWHM).
    """
    if rpm is None or rpm <= 0 or freq_cpm.size == 0:
        return {"resonance_detected": False, "narrative": ""}

    # Buscar el pico cerca de 1X (±10%)
    fmin = 0.90 * rpm
    fmax = 1.10 * rpm
    mask = (freq_cpm >= fmin) & (freq_cpm <= fmax) & np.isfinite(amp_peak)
    if not np.any(mask):
        return {"resonance_detected": False, "narrative": ""}

    sub_freqs = freq_cpm[mask]
    sub_amps = amp_peak[mask]
    idx_peak = int(np.argmax(sub_amps))
    peak_f = float(sub_freqs[idx_peak])
    peak_a = float(sub_amps[idx_peak])

    # FWHM: encontrar puntos donde la amplitud cae a peak_a/2
    half = peak_a * min_height_ratio
    above = sub_amps >= half
    if not np.any(above):
        return {"resonance_detected": False, "narrative": ""}

    # Índices de puntos por encima del 50%
    idxs_above = np.where(above)[0]
    f_lo = sub_freqs[idxs_above[0]]
    f_hi = sub_freqs[idxs_above[-1]]
    fwhm_pct = (f_hi - f_lo) / max(peak_f, 1e-9) * 100.0

    if fwhm_pct >= width_threshold_pct:
        return {
            "resonance_detected": True,
            "fwhm_pct": fwhm_pct,
            "peak_freq_cpm": peak_f,
            "narrative": (
                f"El pico cercano a 1X ({peak_f:.0f} CPM) presenta un ancho a "
                f"-3 dB de aproximadamente {fwhm_pct:.1f}% de la frecuencia "
                f"central, valor característico de **operación cercana a una "
                f"velocidad crítica** o resonancia estructural. Sugerencia: "
                f"verificar diagrama Bode y Q-factor (API 684) — un Q > 5 con "
                f"separation margin < 15% requiere ajuste de balanceo o cambio "
                f"de velocidad operativa."
            ),
        }

    return {"resonance_detected": False, "narrative": ""}


def _normalize_pattern_findings(
    diag_legacy: Dict[str, Any],
    subsync: Dict[str, Any],
    resonance: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Toma todos los findings (del evaluate_spectrum_diagnostic legacy +
    nuevos detectores) y los normaliza a una lista priorizada por
    severidad para que la narrativa Cat IV los presente ordenados.
    """
    findings: List[Dict[str, Any]] = []

    # Legacy: desbalance, desalineación, holgura, banda ancha
    legacy_status = (diag_legacy.get("status") or "SAFE").upper()
    legacy_headline = (diag_legacy.get("headline") or "").strip()
    if legacy_headline and legacy_headline.lower() != "sin patrón anormal dominante":
        rank_map = {"DANGER": 4, "WARNING": 2, "SAFE": 0}
        findings.append({
            "rank": rank_map.get(legacy_status, 0),
            "type": "general_pattern",
            "headline": legacy_headline,
            "narrative": diag_legacy.get("narrative", ""),
            "severity": legacy_status,
        })

    # Sub-síncrono
    sub_class = subsync.get("classification", "NONE")
    if sub_class != "NONE":
        rank = 5 if sub_class == "OIL_WHIP_OR_RUB" else 3 if sub_class == "OIL_WHIRL" else 3
        findings.append({
            "rank": rank,
            "type": "subsynchronous",
            "headline": {
                "OIL_WHIRL": "Oil whirl detectado",
                "OIL_WHIP_OR_RUB": "Oil whip / rub sub-sincrónico detectado",
                "DEEP_SUBSYNC": "Componente sub-rotacional anómala",
            }.get(sub_class, "Componente sub-sincrónica significativa"),
            "narrative": subsync.get("narrative", ""),
            "severity": "DANGER" if sub_class == "OIL_WHIP_OR_RUB" else "WARNING",
        })

    # Resonancia
    if resonance.get("resonance_detected"):
        findings.append({
            "rank": 2,
            "type": "resonance",
            "headline": "Operación cercana a velocidad crítica",
            "narrative": resonance.get("narrative", ""),
            "severity": "WARNING",
        })

    findings.sort(key=lambda f: -f["rank"])
    return findings


def _build_action_items(
    findings: List[Dict[str, Any]],
    bearing_ai: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Genera lista numerada de recomendaciones priorizadas con normas
    citadas, derivadas de los findings + diagnóstico de rodamientos.
    """
    actions: List[str] = []
    seen = set()

    def _add(action: str) -> None:
        key = action[:80].lower()
        if key not in seen:
            seen.add(key)
            actions.append(action)

    # Recomendaciones por tipo de finding
    for f in findings:
        ftype = f.get("type")
        headline = f.get("headline", "").lower()

        if ftype == "general_pattern" and "desbalance" in headline:
            _add(
                "Realizar balanceo de campo del rotor según ISO 21940-12 "
                "(Mechanical vibration — Rotor balancing — Part 12), nivel "
                "de calidad G 2.5 para turbomaquinaria de proceso. Verificar "
                "consistencia de fase entre arranques antes de la intervención."
            )
        elif ftype == "general_pattern" and "desalineación" in headline:
            _add(
                "Verificar alineación del tren en condición caliente conforme "
                "a ANSI-ASA 2.75 (Mechanical vibration — Shaft alignment for "
                "rotating machinery), considerando el crecimiento térmico "
                "especificado por el fabricante. Inspeccionar también "
                "condición del acople."
            )
        elif ftype == "general_pattern" and "holgura" in headline:
            _add(
                "Inspeccionar rigidez de soportes, condición de anclajes, "
                "torque de pernos del pedestal o base. Contrastar con forma "
                "de onda para identificar impactos o modulación que confirmen "
                "soltura mecánica estructural."
            )
        elif ftype == "general_pattern" and "banda ancha" in headline:
            _add(
                "Revisar condición del proceso (carga, presión diferencial, "
                "caudal) para descartar excitación inducida por flujo, "
                "cavitación o turbulencia. Correlacionar con datos del DCS "
                "del intervalo de medición."
            )
        elif ftype == "subsynchronous" and "oil whirl" in headline.lower():
            _add(
                "PRIORIDAD ALTA: verificar clearance real del cojinete contra "
                "el dato OEM (rango típico Cd ≈ 0.0015 × Φ). Confirmar "
                "viscosidad del aceite operativo contra la grado especificado, "
                "presión de suministro y temperatura. Evaluar carga estática "
                "del rotor — un eccentricity ratio bajo (e/c < 0.40) favorece "
                "el oil whirl."
            )
            _add(
                "Si el oil whirl persiste tras corregir clearance/viscosidad, "
                "evaluar cambio de geometría del cojinete (lobed, presure-dam, "
                "tilting pad) según API 684."
            )
        elif ftype == "subsynchronous" and ("whip" in headline.lower() or "rub" in headline.lower()):
            _add(
                "PRIORIDAD CRÍTICA: programar verificación inmediata en órbita "
                "filtrada y forma de onda. Confirmar si la firma sub-sincrónica "
                "es estacionaria (oil whip) o intermitente (rub). En cualquier "
                "caso, restringir operación sostenida hasta diagnóstico "
                "completo. Revisar proximidad de sellos y juego diametral."
            )
        elif ftype == "resonance":
            _add(
                "Validar diagrama Bode para confirmar paso por velocidad "
                "crítica cercana a la velocidad operativa. Calcular Q-factor "
                "y separation margin según API 684 — si Q > 5 y margen < 15%, "
                "evaluar ajuste de balanceo, cambio de velocidad operativa o "
                "rediseño dinámico."
            )

    # Recomendaciones por bearing AI (si hay diagnóstico claro)
    if bearing_ai:
        fault_type = (bearing_ai.get("fault_type") or "").lower()
        severity = (bearing_ai.get("severity") or "").lower()
        if fault_type and "no clear" not in fault_type:
            severity_pre = "PRIORIDAD ALTA: " if severity in ("severa", "moderada") else ""
            _add(
                f"{severity_pre}Investigar la indicación de "
                f"**{bearing_ai.get('fault_type')}** detectada en el "
                f"análisis de frecuencias características. Programar "
                f"reemplazo del rodamiento conforme a ISO 281 (basic dynamic "
                f"load rating) y análisis de causa raíz: lubricación, carga, "
                f"montaje, alineación y sello."
            )

    if not actions:
        actions.append(
            "Mantener seguimiento periódico del espectro y correlacionar con "
            "Polar/Bode 1X, forma de onda y datos de proceso. Conservar la "
            "corrida actual como línea base de aceptación para futuras "
            "comparaciones."
        )

    return actions


def build_spectrum_diagnostics_rotordyn(
    *,
    freq_cpm: np.ndarray,
    amp_peak: np.ndarray,
    one_x_amp: Optional[float],
    harmonics: List[dict],
    overall_spec_rms: Optional[float],
    dominant_peak_freq_cpm: Optional[float],
    dominant_peak_amp: Optional[float],
    rpm: Optional[float],
    bearing_assessment: Optional[Dict[str, Any]] = None,
    bearing_ai: Optional[Dict[str, Any]] = None,
    profile_label: str = "",
    measurement_unit: str = "mm/s pk",
) -> Dict[str, Any]:
    """
    Builder Cat IV completo para Spectrum. Combina:
      - Detector de patrones generales (legacy evaluate_spectrum_diagnostic)
      - Detector sub-sincrónico (oil whirl / whip / rub)
      - Detector de resonancia en 1X
      - Diagnóstico de rodamiento del usuario (bearing_assessment + ai)

    Devuelve dict {headline, detail, action, severity_global, structured}
    estilo reporte Cat IV, alineado con build_polar_diagnostics_rotordyn /
    build_bode_diagnostics_rotordyn / build_scl_diagnostics_rotordyn.
    """
    if freq_cpm is None:
        freq_cpm = np.array([], dtype=float)
    if amp_peak is None:
        amp_peak = np.array([], dtype=float)
    if not isinstance(freq_cpm, np.ndarray):
        freq_cpm = np.asarray(freq_cpm, dtype=float)
    if not isinstance(amp_peak, np.ndarray):
        amp_peak = np.asarray(amp_peak, dtype=float)

    diag_legacy = evaluate_spectrum_diagnostic(
        one_x_amp=one_x_amp,
        harmonics=harmonics or [],
        overall_spec_rms=overall_spec_rms,
        dominant_peak_freq_cpm=dominant_peak_freq_cpm,
        dominant_peak_amp=dominant_peak_amp,
        rpm=rpm,
        bearing_text=None,
    )

    rpm_val = float(rpm) if rpm is not None else 0.0
    subsync = detect_subsynchronous(freq_cpm=freq_cpm, amp_peak=amp_peak, rpm=rpm_val)
    resonance = detect_resonance_at_1x(freq_cpm=freq_cpm, amp_peak=amp_peak, rpm=rpm_val)

    findings = _normalize_pattern_findings(diag_legacy, subsync, resonance)

    # Severidad global
    severity_levels = {"SAFE": 0, "WARNING": 2, "DANGER": 4}
    rank_global = 0
    for f in findings:
        rank_global = max(rank_global, severity_levels.get(f.get("severity", "SAFE"), 0))

    if bearing_ai:
        sev = (bearing_ai.get("severity") or "").lower()
        if sev == "severa":
            rank_global = max(rank_global, 4)
        elif sev == "moderada":
            rank_global = max(rank_global, 3)
        elif sev == "incipiente":
            rank_global = max(rank_global, 2)

    severity_label = {0: "CONDICIÓN ACEPTABLE", 2: "ATENCIÓN", 3: "ACCIÓN REQUERIDA", 4: "CRÍTICA"}.get(
        rank_global, "VIGILANCIA"
    )

    # Headline: el finding con mayor rank gana, o "Sin patrón anormal dominante"
    if findings:
        headline = findings[0]["headline"]
    elif bearing_ai and "no clear" not in (bearing_ai.get("fault_type") or "").lower():
        headline = f"{bearing_ai.get('fault_type', 'Rodamiento')} ({bearing_ai.get('severity', 'Incipiente')})"
    else:
        headline = "Espectro sin patrón anormal dominante"

    # Detail: composición de párrafos
    paragraphs: List[str] = []

    profile_clause = f" para el activo '{profile_label}'" if profile_label else ""
    rpm_clause = f" a {rpm_val:.0f} rpm" if rpm_val > 0 else ""

    paragraphs.append(
        f"El análisis espectral{profile_clause}{rpm_clause} aplica detectores "
        f"automáticos de firmas mecánicas (1X / 2X / armónicos altos / "
        f"sub-sincrónicos / resonancia) conforme a los criterios de API 684 "
        f"e ISO 13373-1, y, cuando el activo tiene rodamientos "
        f"caracterizados, también busca coincidencias con las frecuencias "
        f"características BPFO / BPFI / BSF / FTF dentro de una tolerancia "
        f"configurable. La unidad de medición empleada es {measurement_unit}."
    )

    if findings:
        for f in findings:
            paragraphs.append(f["narrative"])

    if bearing_assessment and bearing_assessment.get("narrative"):
        paragraphs.append(bearing_assessment["narrative"])
    elif bearing_ai and bearing_ai.get("message"):
        paragraphs.append(bearing_ai["message"])

    if not findings and not bearing_assessment:
        paragraphs.append(
            "El espectro no muestra firmas armónicas dominantes ni componentes "
            "sub-sincrónicas significativas. La severidad global se mantiene "
            "en CONDICIÓN ACEPTABLE. Se recomienda continuar el monitoreo "
            "rutinario y conservar esta corrida como línea base."
        )

    detail = "\n\n".join(paragraphs)

    # Action: lista numerada de recomendaciones
    action_items = _build_action_items(findings, bearing_ai)
    intro = (
        "A partir de los hallazgos del análisis espectral, se establecen las "
        "siguientes recomendaciones técnicas priorizadas:"
    )
    action_numbered = "\n\n".join(f"{i}. {a}" for i, a in enumerate(action_items, start=1))
    action = f"{intro}\n\n{action_numbered}"

    return {
        "headline": headline,
        "detail": detail,
        "action": action,
        "severity_global": severity_label,
        "severity_rank": rank_global,
        "findings": findings,
        "subsync": subsync,
        "resonance": resonance,
        "structured": {
            "legacy_status": diag_legacy.get("status"),
            "n_findings": len(findings),
            "rpm": rpm_val,
        },
    }


__all__ = [
    "evaluate_spectrum_diagnostic",
    "build_spectrum_report_notes",
    "detect_subsynchronous",
    "detect_resonance_at_1x",
    "build_spectrum_diagnostics_rotordyn",
    "SAFE_COLOR",
    "WARNING_COLOR",
    "DANGER_COLOR",
    "CRITICAL_COLOR",
    "CAUTION_COLOR",
]
