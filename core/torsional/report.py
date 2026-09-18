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
) -> bytes:
    """Arma el PDF SIGA de análisis torsional y devuelve sus bytes.

    `context` = {name, units_label, rpm, mean, pp, ripple, rms, fs, duration_s,
                 dominant, largest_range, verdict}.
    """
    from core.report_pdf_shell import render_report_pdf, make_styles
    from core.reports_ext.common import (section, subsection, p, numbered_list,
                                          bullets, grid_table, safe_image)

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
    name = context.get("name", "el conjunto evaluado")

    # 1 · Introducción y alcance
    _S("Introducción y alcance")
    body.append(p(
        f"Se realizó un análisis de <b>vibración torsional</b> sobre {name} a partir del par "
        f"transmitido en el eje, medido con el sistema de telemetría <b>Binsfeld TorqueTrak 10K</b> "
        f"(galga extensométrica en el eje → transmisor TX10K-S → receptor RX10K → adquisición "
        f"<b>NI 9229</b> de voltaje DC con muestreo simultáneo). El objetivo fue caracterizar el par "
        f"medio y dinámico, las órdenes de excitación, las frecuencias naturales torsionales y su "
        f"margen de separación respecto a la velocidad de operación, y la severidad de fatiga del eje.",
        styles))

    # 2 · Hallazgos
    if findings:
        _S("Hallazgos")
        body.extend(numbered_list([str(x) for x in findings], styles))

    # 3 · Recomendaciones
    if recommendations:
        _S("Recomendaciones")
        body.extend(numbered_list([str(x) for x in recommendations], styles))

    # 4 · Desarrollo del servicio
    _S("Desarrollo del servicio")
    body.append(p(
        "El servicio se ejecutó midiendo el par transmitido en el eje mediante telemetría de galga "
        "extensométrica (TorqueTrak 10K), convertido a unidades de ingeniería según la geometría del "
        "eje y el factor de galga (Vishay TN-512, Appendix B del fabricante). El desarrollo se "
        "fundamenta en <b>API 684</b> (diagrama de interferencia/Campbell y márgenes de separación "
        "≥10%), <b>ISO 22266</b> (vibración torsional), la norma específica de la máquina "
        "(<b>API 617/618/671/674</b>) y la evaluación de fatiga por conteo rainflow "
        "(<b>ASTM E1049</b>) contra el diagrama de Goodman del eje.", styles))

    body.append(subsection("Parámetros de la corrida", styles))
    body.append(grid_table(
        ["Parámetro", "Valor"],
        [["Velocidad de operación", f"{rpm:,.0f} rpm  (1× = {rpm/60:.1f} Hz)"],
         ["Par medio", f"{context.get('mean', 0):,.1f} {u}"],
         ["Par pico-pico (dinámico)", f"{context.get('pp', 0):,.1f} {u}"],
         ["Rizado (pp/medio)", str(context.get('ripple', '—'))],
         ["RMS", f"{context.get('rms', 0):,.1f} {u}"],
         ["Frecuencia de muestreo", f"{context.get('fs', 0):,.0f} Hz"],
         ["Orden dominante", str(context.get('dominant', '—'))],
         ["Naturales torsionales", ", ".join(f"{x:.1f} Hz" for x in (naturals or [])) or "—"]],
        styles, col_widths=[6.5 * cm, 10.0 * cm]))

    body.append(subsection("Onda de par", styles))
    _fig(waveform_png, "Figura 1. Par vs tiempo (ventana de vueltas del eje; líneas verticales = keyphasor).")

    body.append(subsection("Espectro de órdenes", styles))
    _fig(spectrum_png, "Figura 2. Espectro de par con líneas de orden (1×–5×).")
    if order_rows:
        body.append(grid_table(["Orden", "Frecuencia", "Amplitud", "Fase", "Nivel"],
                               list(order_rows), styles))

    body.append(subsection("Diagrama de Campbell / interferencia", styles))
    _fig(campbell_png, "Figura 3. Órdenes k×RPM vs frecuencias naturales torsionales, banda de "
                       "operación ±10% y cruces (API 684).")
    if crossing_rows:
        body.append(grid_table(["Natural", "Frecuencia", "Orden", "RPM cruce", "Margen", "Estado"],
                               list(crossing_rows), styles))

    body.append(subsection("Fatiga del eje (rainflow)", styles))
    _fig(fatigue_png, "Figura 4. Histograma rainflow del par (ASTM E1049) — entrada para el "
                      "diagrama de Goodman / vida a fatiga.")

    body.append(subsection("Marco normativo", styles))
    body.extend(bullets([str(x) for x in (norms or DEFAULT_NORMS)], styles))

    return render_report_pdf(meta, body)
