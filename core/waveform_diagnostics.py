from __future__ import annotations

from typing import Dict

SAFE_COLOR = "#16a34a"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR = "#dc2626"


def build_waveform_report_notes(text_diag: Dict[str, str]) -> str:
    headline = str(text_diag.get("headline", "") or "").strip()
    narrative = str(text_diag.get("narrative", "") or "").strip()

    blocks = []
    if headline:
        blocks.append(headline)
    if narrative:
        blocks.append(narrative)
    return "\n\n".join(blocks).strip()



def _waveform_machine_context_adjustment(machine: str, narrative: str) -> str:
    machine_upper = (machine or "").upper()

    is_recip = "RECIP" in machine_upper or "RECIPROC" in machine_upper
    is_compressor = "COMPRESOR" in machine_upper or "COMPRESSOR" in machine_upper
    is_plain_bearing_context = is_recip or is_compressor

    updated = narrative

    if is_plain_bearing_context:
        updated = updated.replace("rodamientos", "cojinetes planos")
        updated = updated.replace("rodamiento", "cojinete plano")
        updated = updated.replace(
            "presencia moderada de transitorios de alta energía",
            "presencia moderada de transitorios de alta energía y excitaciones no lineales"
        )
        updated = updated.replace(
            "Se recomienda correlacionar este comportamiento con Spectrum, condiciones operativas y estado de lubricación antes de definir intervención.",
            "Se recomienda correlacionar este comportamiento con Spectrum, condiciones operativas, holgura dinámica y estado de lubricación del cojinete plano antes de definir intervención."
        )

    return updated

def generate_waveform_diagnostic(record, summary: Dict[str, float]) -> Dict[str, str]:
    machine = str(getattr(record, "machine", "") or "máquina no identificada")
    point = str(getattr(record, "point", "") or "punto no identificado")

    cf = summary.get("Crest Factor")
    rms = summary.get("RMS")
    pp = summary.get("Peak-Peak")

    if cf is None:
        return {
            "status": "SAFE",
            "color": SAFE_COLOR,
            "headline": "Sin diagnóstico válido de forma de onda",
            "narrative": (
                f"No fue posible generar un diagnóstico confiable para la forma de onda correspondiente al punto {point} "
                f"en la máquina {machine} por falta de datos válidos."
            ),
        }

    try:
        cf = float(cf)
    except Exception:
        cf = float("nan")

    try:
        rms_txt = f"{float(rms):.3f}"
    except Exception:
        rms_txt = "—"

    try:
        pp_txt = f"{float(pp):.3f}"
    except Exception:
        pp_txt = "—"

    if cf != cf:
        return {
            "status": "SAFE",
            "color": SAFE_COLOR,
            "headline": "Sin diagnóstico válido de forma de onda",
            "narrative": (
                f"No fue posible generar un diagnóstico confiable para la forma de onda correspondiente al punto {point} "
                f"en la máquina {machine}."
            ),
        }

    if cf < 3.0:
        return {
            "status": "SAFE",
            "color": SAFE_COLOR,
            "headline": "Forma de onda con comportamiento estable",
            "narrative": (
                f"Se analizó la forma de onda de vibración correspondiente al punto {point} en la máquina {machine}. "
                f"La señal presenta un comportamiento estable, con un factor de cresta de {cf:.2f}, un valor RMS de {rms_txt} "
                f"y una amplitud pico a pico de {pp_txt}, sin evidencia significativa de transitorios de alta energía o comportamiento no lineal. "
                f"Se recomienda continuar el monitoreo periódico y correlacionar este resultado con el análisis espectral para seguimiento de condición."
            ),
        }

    if cf <= 5.0:
        return {
            "status": "WARNING",
            "color": WARNING_COLOR,
            "headline": "Forma de onda con indicios moderados de transitorios de alta energía",
            "narrative": (
                f"Se analizó la forma de onda de vibración correspondiente al punto {point} en la máquina {machine}. "
                f"La señal presenta un comportamiento ligeramente no lineal, con un factor de cresta de {cf:.2f}, un valor RMS de {rms_txt} "
                f"y una amplitud pico a pico de {pp_txt}, lo cual sugiere presencia moderada de transitorios de alta energía. "
                f"Se recomienda correlacionar este comportamiento con Spectrum, condiciones operativas y estado de lubricación antes de definir intervención."
            ),
        }

    return {
        "status": "DANGER",
        "color": DANGER_COLOR,
        "headline": "Forma de onda con comportamiento altamente impulsivo",
        "narrative": (
            f"Se analizó la forma de onda de vibración correspondiente al punto {point} en la máquina {machine}. "
            f"La señal presenta un comportamiento altamente impulsivo, con un factor de cresta de {cf:.2f}, un valor RMS de {rms_txt} "
            f"y una amplitud pico a pico de {pp_txt}, condición compatible con transitorios mecánicos, contacto intermitente o daño incipiente en elementos rodantes. "
            f"Se recomienda priorizar la inspección mecánica, validar la condición del rodamiento y correlacionar con Envelope Spectrum antes de operación repetida."
        ),
    }


# =============================================================
# CICLO 12 — DIAGNÓSTICO CAT IV COMPLETO DE FORMA DE ONDA
# =============================================================
# Wrapper Cat IV completo, alineado con build_polar_diagnostics_rotordyn
# / build_bode_diagnostics_rotordyn / build_scl_diagnostics_rotordyn /
# build_spectrum_diagnostics_rotordyn. Combina:
#
#   - classify_crest_factor (5 buckets: SINUSOIDAL → CRÍTICA)
#   - detect_amplitude_modulation (envolvente Hilbert) → defectos
#     incipientes de rodamiento, engranajes
#   - detect_asymmetry → rub unidireccional, precarga lateral
#   - detect_clipping → sensor saturado, rango insuficiente
#   - detect_sawtooth_shape → rub severo bidireccional con pendientes
#     asimétricas
#   - detect_beating → slip de polos en motores de inducción,
#     interferencia con máquinas vecinas
#   - statistics_summary del usuario (RMS, peak, p2p, kurtosis,
#     skewness) — sin tocar la lógica existente
#
# Genera narrativa Cat IV con vocabulario técnico riguroso (presesión
# reversa, deflexión térmica, inestabilidad inducida en fluido,
# compensación slow roll) y recomendaciones priorizadas citando
# normas: ISO 13373-1 (condition monitoring general), ISO 7919-X
# (shaft vibration evaluation), API 670 (instrumentación con sondas
# de proximidad), ISO 20816-X según familia de máquina.

def build_waveform_diagnostics_rotordyn(
    *,
    time_s,
    amplitude,
    metrics: Dict[str, float],
    impacts: Optional[Dict[str, Any]] = None,
    machine_label: str = "",
    point_label: str = "",
    amplitude_unit: str = "",
) -> Dict[str, Any]:
    """
    Builder Cat IV completo para forma de onda. Devuelve dict con
    headline + detail + action + severity_global + structured.
    """
    import numpy as np
    from core.waveform_pattern_detectors import (
        detect_amplitude_modulation,
        detect_asymmetry,
        detect_clipping,
        detect_sawtooth_shape,
        detect_beating,
        classify_crest_factor,
    )

    t_arr = np.asarray(time_s, dtype=float).reshape(-1) if time_s is not None else np.array([])
    a_arr = np.asarray(amplitude, dtype=float).reshape(-1) if amplitude is not None else np.array([])

    rms = float(metrics.get("rms", 0.0) or 0.0)
    peak = float(metrics.get("peak", 0.0) or 0.0)
    p2p = float(metrics.get("peak_to_peak", 0.0) or 0.0)
    cf = float(metrics.get("crest_factor", 0.0) or 0.0)
    kurt = float(metrics.get("kurtosis", 0.0) or 0.0)
    skew = float(metrics.get("skewness", 0.0) or 0.0)

    cf_class = classify_crest_factor(cf)
    am = detect_amplitude_modulation(t_arr, a_arr)
    asym = detect_asymmetry(a_arr)
    clip = detect_clipping(a_arr)
    saw = detect_sawtooth_shape(a_arr)
    beat = detect_beating(t_arr, a_arr)

    findings: List[Dict[str, Any]] = []

    # Bucket de crest factor — siempre incluido si no es SINUSOIDAL
    if cf_class.get("bucket") not in ("SINUSOIDAL", "UNKNOWN"):
        findings.append({
            "rank": cf_class.get("rank", 0),
            "type": "crest_factor",
            "headline": f"Crest Factor en zona {cf_class.get('bucket')}",
            "narrative": (
                f"El factor de cresta de {cf:.2f} ubica la forma de onda en "
                f"el bucket Cat IV {cf_class.get('bucket')}. {cf_class.get('message')}"
            ),
            "severity": cf_class.get("severity_label"),
        })

    if clip.get("detected"):
        findings.append({
            "rank": 4, "type": "clipping",
            "headline": "Posible saturación del sensor (clipping)",
            "narrative": clip.get("narrative", ""),
            "severity": "ACCIÓN REQUERIDA",
        })

    if saw.get("detected"):
        findings.append({
            "rank": 4, "type": "sawtooth",
            "headline": "Forma diente de sierra — rub severo bidireccional",
            "narrative": (
                saw.get("narrative", "") + " La presencia repetida de rozamiento "
                "en la misma zona del rotor produce calentamiento local y "
                "deflexión térmica que retroalimenta el desbalance original, "
                "creando un ciclo dinámico inestable."
            ),
            "severity": "CRÍTICA",
        })

    if beat.get("detected"):
        findings.append({
            "rank": 2, "type": "beating",
            "headline": "Beating detectado (interferencia entre frecuencias cercanas)",
            "narrative": beat.get("narrative", ""),
            "severity": "ATENCIÓN",
        })

    if asym.get("detected"):
        findings.append({
            "rank": 3, "type": "asymmetry",
            "headline": (
                f"Asimetría direccional {asym.get('severity_word', 'leve')} "
                f"hacia polaridad {asym.get('direction', '—')}"
            ),
            "narrative": (
                asym.get("narrative", "") + " En presencia de rub repetido en "
                "la misma zona del rotor, la fricción local genera "
                "calentamiento y posterior deflexión térmica, alterando la "
                "dirección de respuesta sincrónica del rotor (mecanismo "
                "secundario que mantiene la asimetría en el tiempo)."
            ),
            "severity": "ACCIÓN REQUERIDA" if asym.get("severity_word") == "severa" else "ATENCIÓN",
        })

    if am.get("detected") and not beat.get("detected"):
        # AM general (no es beating de baja frecuencia)
        findings.append({
            "rank": 2, "type": "amplitude_modulation",
            "headline": (
                f"Modulación de amplitud {am.get('severity_word', 'leve')} "
                f"({am.get('modulation_depth', 0)*100:.0f}% de profundidad)"
            ),
            "narrative": am.get("narrative", ""),
            "severity": "ATENCIÓN",
        })

    # Kurtosis alto (>4 es indicador de no-gaussianidad por impactos)
    if kurt > 4.5 and not saw.get("detected"):
        findings.append({
            "rank": 2, "type": "high_kurtosis",
            "headline": f"Kurtosis elevado ({kurt:.2f}) — distribución no gaussiana",
            "narrative": (
                f"El cuarto momento estadístico de la señal (kurtosis = "
                f"{kurt:.2f}) supera el rango gaussiano (~3.0). Esto indica "
                f"que la distribución de amplitudes tiene colas pesadas, "
                f"firma estadística clásica de impactos discretos episódicos "
                f"que no aparecen en una señal estacionaria. Patrón consistente "
                f"con defectos incipientes de rodamiento, golpeteo mecánico "
                f"intermitente o cavitación (en bombas)."
            ),
            "severity": "ATENCIÓN",
        })

    # Impacts del detector existente del usuario
    if impacts and impacts.get("count", 0) > 0:
        n_impacts = int(impacts.get("count", 0))
        rank_imp = 3 if n_impacts >= 5 else 2
        findings.append({
            "rank": rank_imp, "type": "impacts",
            "headline": f"{n_impacts} impacto(s) discretos detectado(s) en la captura",
            "narrative": (
                f"Se identifican {n_impacts} pico(s) que superan el umbral "
                f"de {impacts.get('threshold', 0):.3f} unidades (≈3.5×RMS). "
                f"Los impactos discretos en señal de vibración sugieren "
                f"contacto intermitente, daño en elementos rodantes o "
                f"transitorios mecánicos no estacionarios. Validar con "
                f"Envelope Spectrum y revisar correlación temporal con "
                f"eventos del proceso."
            ),
            "severity": "ACCIÓN REQUERIDA" if n_impacts >= 5 else "ATENCIÓN",
        })

    findings.sort(key=lambda f: -f["rank"])

    severity_levels = {
        "CONDICIÓN ACEPTABLE": 0, "VIGILANCIA": 1, "ATENCIÓN": 2,
        "ACCIÓN REQUERIDA": 3, "CRÍTICA": 4,
    }
    rank_global = 0
    for f in findings:
        rank_global = max(rank_global, severity_levels.get(f.get("severity", ""), 0))
    if cf_class.get("bucket") == "SINUSOIDAL" and not findings:
        rank_global = 0
    severity_global = {0: "CONDICIÓN ACEPTABLE", 1: "VIGILANCIA", 2: "ATENCIÓN",
                        3: "ACCIÓN REQUERIDA", 4: "CRÍTICA"}.get(rank_global, "VIGILANCIA")

    if findings:
        headline = findings[0]["headline"]
    elif cf_class.get("bucket") == "SINUSOIDAL":
        headline = "Forma de onda sinusoidal estable — sin patrones anormales"
    else:
        headline = "Forma de onda dentro de rango normal"

    paragraphs: List[str] = []
    machine_clause = f" del activo {machine_label}" if machine_label else ""
    point_clause = f" en el punto {point_label}" if point_label else ""
    unit_clause = f" (unidad: {amplitude_unit})" if amplitude_unit else ""

    paragraphs.append(
        f"El análisis de forma de onda en dominio del tiempo{machine_clause}"
        f"{point_clause}{unit_clause} aplica detectores Cat IV para "
        f"identificar firmas mecánicas que el espectro promedia y oculta: "
        f"crest factor por bucket, modulación de amplitud (envolvente "
        f"Hilbert), asimetría direccional, beating, clipping, forma diente "
        f"de sierra y kurtosis estadística. Métricas observadas: RMS = "
        f"{rms:.4f}, Peak = {peak:.4f}, Peak-to-Peak = {p2p:.4f}, Crest "
        f"Factor = {cf:.2f}, Kurtosis = {kurt:.2f}, Skewness = {skew:.2f}."
    )

    if findings:
        for f in findings:
            paragraphs.append(f["narrative"])
    else:
        paragraphs.append(
            "La forma de onda no presenta firmas anormales. La señal es "
            "esencialmente sinusoidal con crest factor en rango aceptable y "
            "ausencia de modulación, asimetría, beating o saturación. La "
            "severidad global se mantiene en CONDICIÓN ACEPTABLE."
        )

    detail = "\n\n".join(paragraphs)

    actions: List[str] = []
    seen = set()

    def _add(action: str) -> None:
        key = action[:80].lower()
        if key not in seen:
            seen.add(key)
            actions.append(action)

    for f in findings:
        ftype = f.get("type")
        if ftype == "crest_factor":
            bucket = (cf_class.get("bucket") or "")
            if bucket in ("ALERT", "SEVERE", "CRITICAL"):
                _add(
                    "Correlacionar el crest factor elevado con el espectro "
                    "de envolvente (Envelope Spectrum) y con frecuencias "
                    "características de defectos de rodamiento "
                    "(BPFO/BPFI/BSF/FTF) para confirmar mecanismo. "
                    "Aplicar criterios de evaluación de ISO 13373-1 "
                    "(Condition monitoring and diagnostics — Vibration condition "
                    "monitoring) para clasificar severidad de impactos."
                )
        elif ftype == "clipping":
            _add(
                "PRIORIDAD ALTA: revisar la escala dinámica del sensor o el "
                "rango configurado en el sistema de adquisición. La amplitud "
                "reportada está subestimada por saturación. Repetir la "
                "captura con rango ampliado conforme a API 670 (norma de "
                "instrumentación con sondas de proximidad) antes de concluir "
                "severidad."
            )
        elif ftype == "sawtooth":
            _add(
                "PRIORIDAD CRÍTICA: programar inspección directa del "
                "cojinete y de los sellos en el próximo paro. La forma "
                "diente de sierra es firma de rub severo bidireccional. "
                "Correlacionar con cambios repentinos en la posición "
                "promedio de la línea central del eje (Shaft Centerline) "
                "que confirmen el mecanismo de rozamiento."
            )
        elif ftype == "asymmetry":
            _add(
                "Verificar centrado del eje en el cojinete (módulo Shaft "
                "Centerline) y eccentricity ratio actual. Inspeccionar "
                "condición de sellos cercanos al punto de medición. "
                "Revisar alineación caliente del tren conforme a ANSI-ASA "
                "2.75 considerando el crecimiento térmico OEM."
            )
        elif ftype == "amplitude_modulation":
            _add(
                "Validar la frecuencia de modulación detectada contra las "
                "frecuencias características del activo (rodamientos, "
                "engranajes, frecuencias de paso del proceso). Si coincide "
                "con BPFO/BPFI/BSF/FTF, clasificar como defecto incipiente "
                "de rodamiento y programar reemplazo conforme a ISO 281."
            )
        elif ftype == "beating":
            _add(
                "Verificar velocidades de operación de máquinas adyacentes "
                "para descartar interferencia mecánica. En motores de "
                "inducción, evaluar slip eléctrico contra valor nominal de "
                "placa. Comparar el spectrum con resolución alta para "
                "identificar las dos componentes de frecuencia generadoras "
                "del batido."
            )
        elif ftype == "high_kurtosis":
            _add(
                "Aplicar Envelope Spectrum (high-frequency demodulation) "
                "para extraer la firma de impactos del rango portador. "
                "Correlacionar con frecuencias características de "
                "rodamientos y datos de proceso (presión, caudal) para "
                "descartar cavitación."
            )
        elif ftype == "impacts":
            _add(
                "Revisar condición de los soportes y anclajes — los "
                "impactos discretos repetidos suelen asociarse a holgura "
                "mecánica estructural. Correlacionar con análisis de "
                "Spectrum (armónicos altos), órbita filtrada y cambios en "
                "Shaft Centerline."
            )

    if not actions:
        _add(
            "Mantener el monitoreo rutinario y conservar esta corrida como "
            "línea base de aceptación de la forma de onda. Correlacionar "
            "periódicamente con Spectrum, Polar/Bode 1X y datos de proceso "
            "del DCS para detección temprana de cambios."
        )

    intro = (
        "A partir del análisis Cat IV de la forma de onda, se establecen "
        "las siguientes recomendaciones técnicas priorizadas:"
    )
    action = intro + "\n\n" + "\n\n".join(f"{i}. {a}" for i, a in enumerate(actions, start=1))

    return {
        "headline": headline,
        "detail": detail,
        "action": action,
        "severity_global": severity_global,
        "severity_rank": rank_global,
        "findings": findings,
        "crest_factor_class": cf_class,
        "structured": {
            "rms": rms, "peak": peak, "peak_to_peak": p2p,
            "crest_factor": cf, "kurtosis": kurt, "skewness": skew,
            "n_findings": len(findings),
        },
    }


# Re-export para facilitar imports en consumers
from typing import Any, List, Optional
