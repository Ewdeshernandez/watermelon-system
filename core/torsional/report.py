"""
core/torsional/report.py — Reporte SIGA de análisis torsional (PDF)
===================================================================

Cierra el ciclo campo → análisis → **reporte PDF** para el módulo Torsional,
reutilizando el MISMO motor SIGA del módulo Modal (`core.report_pdf_shell` +
`core.reports_ext.common`): portada + tabla de contenido + secciones
numeradas (Introducción/Hallazgos/Recomendaciones/Desarrollo) con figuras y
tablas embebidas. Idéntico look al reporte OMA.

Idioma: el reporte va en **español** (política de idioma: la UI del analista
es inglés, el reporte y la vista del cliente en español).

Marco normativo del análisis torsional:
  · API 684 — diagrama de interferencia/Campbell + márgenes de separación ≥10%.
  · ISO 22266 — vibración torsional de maquinaria rotativa.
  · API 617/618/671/674 — requisitos torsionales por tipo de máquina.
  · ASTM E1049 + Goodman — fatiga del eje por conteo rainflow.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from reportlab.lib.units import cm

DEFAULT_NORMS: List[str] = [
    "API 684 — Rotordynamics Tutorial: diagrama de interferencia (Campbell) y márgenes de separación.",
    "ISO 22266 — Vibración torsional de maquinaria rotativa.",
    "API 617 / 618 / 671 / 674 — requisitos torsionales por tipo de máquina.",
    "ASTM E1049 — conteo de ciclos rainflow (entrada para fatiga / Goodman del eje).",
]


def plotly_to_png(fig, width: int = 1280, height: int = 560, scale: float = 1.6) -> Optional[bytes]:
    """Plotly → PNG (kaleido). Defensivo: None si falla (el PDF igual se arma)."""
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", width=width, height=height, scale=scale)
    except Exception:  # noqa: BLE001
        return None


_NORMS_EN: List[str] = [
    "API 684 — Rotordynamics Tutorial: interference (Campbell) diagram and separation margins.",
    "ISO 22266 — Torsional vibration of rotating machinery.",
    "API 617 / 618 / 671 / 674 — machinery-specific torsional requirements.",
    "ASTM E1049 — rainflow cycle counting (input for shaft fatigue / Goodman).",
]

_TXT = {
    "es": {
        "s_intro": "Introducción y alcance",
        "s_find": "Hallazgos", "s_rec": "Recomendaciones",
        "s_dev": "Desarrollo del servicio", "sub_params": "Parámetros de la corrida",
        "sub_wave": "Onda de par", "sub_spec": "Espectro de órdenes",
        "sub_camp": "Diagrama de Campbell / interferencia", "sub_fat": "Fatiga del eje (rainflow)",
        "sub_norms": "Marco normativo",
        "intro": ("Se realizó un análisis de <b>vibración torsional</b> sobre {name} a partir del par "
                  "transmitido en el eje, medido con el sistema de telemetría <b>Binsfeld TorqueTrak 10K</b> "
                  "(galga extensométrica en el eje → transmisor TX10K-S → receptor RX10K → adquisición "
                  "<b>NI 9229</b> de voltaje DC con muestreo simultáneo). El objetivo fue caracterizar el par "
                  "medio y dinámico, las órdenes de excitación, las frecuencias naturales torsionales y su "
                  "margen de separación respecto a la velocidad de operación, y la severidad de fatiga del eje."),
        "dev": ("El servicio se ejecutó midiendo el par transmitido en el eje mediante telemetría de galga "
                "extensométrica (TorqueTrak 10K), convertido a unidades de ingeniería según la geometría del "
                "eje y el factor de galga (Vishay TN-512, Appendix B del fabricante). El desarrollo se "
                "fundamenta en <b>API 684</b> (diagrama de interferencia/Campbell y márgenes de separación "
                "≥10%), <b>ISO 22266</b> (vibración torsional), la norma específica de la máquina "
                "(<b>API 617/618/671/674</b>) y la evaluación de fatiga por conteo rainflow "
                "(<b>ASTM E1049</b>) contra el diagrama de Goodman del eje."),
        "params": ["Parámetro", "Valor"],
        "p_speed": "Velocidad de operación", "p_mean": "Par medio", "p_pp": "Par pico-pico (dinámico)",
        "p_rip": "Rizado (pp/medio)", "p_rms": "RMS", "p_fs": "Frecuencia de muestreo",
        "p_dom": "Orden dominante", "p_nat": "Naturales torsionales",
        "cap_wave": "Figura 1. Par vs tiempo (ventana de vueltas del eje; líneas verticales = keyphasor).",
        "cap_spec": "Figura 2. Espectro de par con líneas de orden (1×–5×).",
        "cap_camp": ("Figura 3. Órdenes k×RPM vs frecuencias naturales torsionales, banda de operación "
                     "±10% y cruces (API 684)."),
        "cap_fat": ("Figura 4. Histograma rainflow del par (ASTM E1049) — entrada para el diagrama de "
                    "Goodman / vida a fatiga."),
        "h_order": ["Orden", "Frecuencia", "Amplitud", "Fase", "Nivel"],
        "h_cross": ["Natural", "Frecuencia", "Orden", "RPM cruce", "Margen", "Estado"],
    },
    "en": {
        "s_intro": "Introduction and scope",
        "s_find": "Findings", "s_rec": "Recommendations",
        "s_dev": "Service development", "sub_params": "Run parameters",
        "sub_wave": "Torque waveform", "sub_spec": "Order spectrum",
        "sub_camp": "Campbell / interference diagram", "sub_fat": "Shaft fatigue (rainflow)",
        "sub_norms": "Standards",
        "intro": ("A <b>torsional vibration</b> analysis was performed on {name} from the shaft torque, "
                  "measured with the <b>Binsfeld TorqueTrak 10K</b> telemetry system (strain gage on the "
                  "shaft → TX10K-S transmitter → RX10K receiver → <b>NI 9229</b> DC voltage acquisition with "
                  "simultaneous sampling). The goal was to characterize the mean and dynamic torque, the "
                  "excitation orders, the torsional natural frequencies and their separation margin from the "
                  "operating speed, and the shaft fatigue severity."),
        "dev": ("The service measured the shaft torque via strain-gage telemetry (TorqueTrak 10K), converted "
                "to engineering units from the shaft geometry and gage factor (Vishay TN-512, manufacturer "
                "Appendix B). The work is grounded in <b>API 684</b> (interference/Campbell diagram and "
                "separation margins ≥10%), <b>ISO 22266</b> (torsional vibration), the machinery-specific "
                "standard (<b>API 617/618/671/674</b>) and fatigue evaluation by rainflow counting "
                "(<b>ASTM E1049</b>) against the shaft Goodman diagram."),
        "params": ["Parameter", "Value"],
        "p_speed": "Operating speed", "p_mean": "Mean torque", "p_pp": "Peak-peak torque (dynamic)",
        "p_rip": "Ripple (pp/mean)", "p_rms": "RMS", "p_fs": "Sample rate",
        "p_dom": "Dominant order", "p_nat": "Torsional naturals",
        "cap_wave": "Figure 1. Torque vs time (shaft-revolution window; vertical lines = keyphasor).",
        "cap_spec": "Figure 2. Torque spectrum with order lines (1×–5×).",
        "cap_camp": ("Figure 3. Orders k×RPM vs torsional natural frequencies, operating band ±10% and "
                     "crossings (API 684)."),
        "cap_fat": ("Figure 4. Torque rainflow histogram (ASTM E1049) — input for the Goodman diagram / "
                    "fatigue life."),
        "h_order": ["Order", "Frequency", "Amplitude", "Phase", "Level"],
        "h_cross": ["Natural", "Frequency", "Order", "Crossing", "Margin", "Status"],
    },
}


def build_torsional_pdf(
    *,
    meta: Dict[str, Any],
    context: Dict[str, Any],
    findings: Optional[Sequence[str]] = None,
    recommendations: Optional[Sequence[str]] = None,
    waveform_png: Optional[bytes] = None,
    spectrum_png: Optional[bytes] = None,
    campbell_png: Optional[bytes] = None,
    fatigue_png: Optional[bytes] = None,
    order_rows: Optional[Sequence[Sequence[Any]]] = None,
    crossing_rows: Optional[Sequence[Sequence[Any]]] = None,
    naturals: Optional[Sequence[float]] = None,
    norms: Optional[Sequence[str]] = None,
    lang: str = "es",
) -> bytes:
    """Arma el PDF SIGA de análisis torsional y devuelve sus bytes.

    `context` = {name, units_label, rpm, mean, pp, ripple, rms, fs, dominant}.
    `lang` = "es" (default) | "en".
    """
    from core.report_pdf_shell import render_report_pdf, make_styles
    from core.reports_ext.common import (section, subsection, p, numbered_list,
                                          bullets, grid_table, safe_image)

    L = _TXT.get(lang, _TXT["es"])
    styles = make_styles()
    body: List[Any] = []
    _sn = [0]

    def _S(title: str) -> None:
        _sn[0] += 1
        body.append(section(f"{_sn[0]}. {title}", styles))

    def _fig(png: Optional[bytes], caption: str, w_cm: float = 16.5, h_cm: float = 7.4) -> None:
        img = safe_image(png, w_cm, h_cm) if png else None
        if img is not None:
            body.append(img)
            if caption:
                body.append(p(caption, styles, "WMFigureCaption"))

    u = context.get("units_label", "N·m")
    rpm = float(context.get("rpm", 0.0) or 0.0)
    name = context.get("name", "el conjunto evaluado" if lang == "es" else "the evaluated unit")

    _S(L["s_intro"])
    body.append(p(L["intro"].format(name=name), styles))

    if findings:
        _S(L["s_find"])
        body.extend(numbered_list([str(x) for x in findings], styles))

    if recommendations:
        _S(L["s_rec"])
        body.extend(numbered_list([str(x) for x in recommendations], styles))

    _S(L["s_dev"])
    body.append(p(L["dev"], styles))

    body.append(subsection(L["sub_params"], styles))
    body.append(grid_table(
        L["params"],
        [[L["p_speed"], f"{rpm:,.0f} rpm  (1× = {rpm/60:.1f} Hz)"],
         [L["p_mean"], f"{context.get('mean', 0):,.1f} {u}"],
         [L["p_pp"], f"{context.get('pp', 0):,.1f} {u}"],
         [L["p_rip"], str(context.get('ripple', '—'))],
         [L["p_rms"], f"{context.get('rms', 0):,.1f} {u}"],
         [L["p_fs"], f"{context.get('fs', 0):,.0f} Hz"],
         [L["p_dom"], str(context.get('dominant', '—'))],
         [L["p_nat"], ", ".join(f"{x:.1f} Hz" for x in (naturals or [])) or "—"]],
        styles, col_widths=[6.5 * cm, 10.0 * cm]))

    body.append(subsection(L["sub_wave"], styles))
    _fig(waveform_png, L["cap_wave"])

    body.append(subsection(L["sub_spec"], styles))
    _fig(spectrum_png, L["cap_spec"])
    if order_rows:
        body.append(grid_table(L["h_order"], list(order_rows), styles))

    body.append(subsection(L["sub_camp"], styles))
    _fig(campbell_png, L["cap_camp"])
    if crossing_rows:
        body.append(grid_table(L["h_cross"], list(crossing_rows), styles))

    body.append(subsection(L["sub_fat"], styles))
    _fig(fatigue_png, L["cap_fat"])

    body.append(subsection(L["sub_norms"], styles))
    body.extend(bullets([str(x) for x in (norms or (DEFAULT_NORMS if lang == "es" else _NORMS_EN))], styles))

    return render_report_pdf(meta, body)
