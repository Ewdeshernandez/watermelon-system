from __future__ import annotations

from typing import Any, Dict, List, Optional

SAFE_COLOR = "#16a34a"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR = "#dc2626"


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
