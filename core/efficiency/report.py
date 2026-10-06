"""
core/efficiency/report.py — Reporte PDF de eficiencia (formato SIGA Watermelon)
===============================================================================

Reporte de ENTREGA (web) de eficiencia de máquinas rotatorias, branded
Watermelon/SIGA con el MISMO shell que balanceo/torsional/modal
(`core.report_pdf_shell.render_report_pdf`). NO es el preliminar de campo.

Secciones: 1 Datos · 2 Potencias y eficiencia (con semáforo) · 3 Eficiencia de
proceso (según tipo) · 4 Hallazgos y recomendaciones · metodología/normas.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from core.report_pdf_shell import render_report_pdf, make_styles, REGULAR, BOLD

_INK = "#0f172a"
_HEADER_BG = "#0f4c81"

# Semáforo → color de celda (igual que la banda del módulo).
_SEMA_HEX = {"green": "#16a34a", "amber": "#e8890c", "red": "#dc2626"}


def _p(text: str, styles, style: str = "WMBody"):
    return Paragraph(str(text), styles[style])


def _section(title: str, styles):
    return Paragraph(title, styles["WMTOC1"])


def _fmt(x: Any, nd: int = 2) -> str:
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:  # noqa: BLE001
        return "—"


def _kv_table(rows: List[Tuple[str, str]], styles) -> Table:
    data = [[Paragraph(f"<b>{k}</b>", styles["WMTableCell"]),
             Paragraph(str(v), styles["WMTableCell"])] for k, v in rows]
    t = Table(data, colWidths=[5.2 * cm, 11.0 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _grid_table(headers: List[str], rows: List[List[Any]], styles,
                col_widths: Optional[List[float]] = None,
                row_colors: Optional[Dict[int, str]] = None) -> Table:
    head = [Paragraph(f"<b>{h}</b>", styles["WMTableHeader"]) for h in headers]
    body = [[Paragraph(str(c), styles["WMTableCell"]) for c in r] for r in rows]
    t = Table([head] + body, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_HEADER_BG)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    # Resalta filas (p.ej. semáforo) con color de fondo: row_colors={fila: hex}.
    for ridx, hexv in (row_colors or {}).items():
        style.append(("BACKGROUND", (0, ridx), (-1, ridx), colors.HexColor(hexv)))
        style.append(("TEXTCOLOR", (0, ridx), (-1, ridx), colors.white))
    t.setStyle(TableStyle(style))
    return t


def build_efficiency_pdf(*, meta: Dict[str, Any], result: Dict[str, Any]) -> bytes:
    """Arma el PDF de eficiencia (entrega SIGA).

    meta: asset, client, location, specialist, report_date, machine_type_label,
          norm, rpm, data_source, notes, rotation...
    result: dict del cálculo (keys como las de EfficiencyResult + inputs):
          mtype, torque, rpm, volt, curr, pf, design, p_mec, p_elec, eta_motor,
          eta_op, diag_code, diag_es, diag_color, proc_label, eta_proc, p_proc,
          flow, head, dp (opcionales según tipo).
    """
    styles = make_styles()
    body: List[Any] = []
    r = result
    _diag_es = r.get("diag_es", "—")
    _diag_hex = _SEMA_HEX.get(r.get("diag_color", ""), "#64748b")

    # Orden canónico SIGA (igual que torsional/modal):
    #   1 Introducción y alcance · 2 Hallazgos · 3 Recomendaciones ·
    #   4 Desarrollo del servicio · 5 Marco normativo.
    # Resumen ejecutivo (hallazgos + recomendaciones) ANTES del detalle técnico.

    # ---- 1. Introducción y alcance ------------------------------------
    body.append(_section("1. Introducción y alcance", styles))
    body.append(_kv_table([
        ("Activo", meta.get("asset", "—")),
        ("Cliente", meta.get("client", "—")),
        ("Sitio / ubicación", meta.get("location", "—")),
        ("Tipo de máquina", meta.get("machine_type_label", "—")),
        ("Especialista", meta.get("specialist", "—")),
        ("Fecha", meta.get("report_date") or datetime.now().strftime("%d/%m/%Y")),
        ("Velocidad medida", f"{_fmt(r.get('rpm'), 0)} rpm"),
        ("Fuente de datos", meta.get("data_source", "Medición de campo")),
        ("Norma", meta.get("norm", "IEC 60034-2 · ISO 20816")),
    ], styles))
    body.append(Spacer(1, 0.3 * cm))
    body.append(_p("Determinación de la potencia mecánica real absorbida y la eficiencia "
                   "operativa del equipo a partir de mediciones directas de campo bajo "
                   "condiciones reales de operación, con comparación frente a los parámetros "
                   "de diseño para establecer la capacidad disponible y el margen de reserva.",
                   styles, "WMBody"))
    if meta.get("notes"):
        body.append(Spacer(1, 0.2 * cm))
        body.append(_p(meta["notes"], styles, "WMBody"))
    body.append(Spacer(1, 0.5 * cm))

    # ---- 2. Hallazgos -------------------------------------------------
    body.append(_section("2. Hallazgos", styles))
    findings = meta.get("findings") or []
    for fnd in findings:
        body.append(_p(f"• {fnd}", styles, "WMBody"))
    if (r.get("eta_op") or 0) > 0:
        _reserve = (r.get("design") or 0) - (r.get("p_mec") or 0)
        _resv_pct = 100.0 - (r.get("eta_op") or 0)
        body.append(Spacer(1, 0.2 * cm))
        srows = [["Eficiencia operativa", f"{_fmt(r.get('eta_op'), 1)} %"],
                 ["Diagnóstico (semáforo)", _diag_es],
                 ["Margen de reserva", f"{_fmt(_reserve, 0)} kW ({_fmt(_resv_pct, 1)} %)"]]
        body.append(_grid_table(["Evaluación operativa", "Valor"], srows, styles,
                                col_widths=[7.5 * cm, 8.7 * cm], row_colors={2: _diag_hex}))
        body.append(_p("Bandas: &gt;105% sobrecarga · 95–105% normal · 85–95% degradación · "
                       "&lt;85% falla inminente.", styles, "WMBody"))
    body.append(Spacer(1, 0.5 * cm))

    # ---- 3. Recomendaciones (al final del resumen ejecutivo) ----------
    body.append(_section("3. Recomendaciones", styles))
    recs = meta.get("recommendations") or []
    for rc in recs:
        body.append(_p(f"• {rc}", styles, "WMBody"))
    body.append(Spacer(1, 0.5 * cm))

    # ---- 4. Desarrollo del servicio (detalle técnico) -----------------
    body.append(_section("4. Desarrollo del servicio", styles))
    body.append(_p("Potencia mecánica en el eje P_mec = T·ω/1000 (par por telemetría "
                   "rotativa + velocidad por tacómetro). Potencia eléctrica "
                   "P_elec = √3·V·I·cosφ/1000 (analizador de red). "
                   "η_motor = P_mec/P_elec · η_operativa = P_mec/P_diseño.",
                   styles, "WMBody"))
    prows = [["Par medido", f"{_fmt(r.get('torque'), 0)} N·m"],
             ["Velocidad", f"{_fmt(r.get('rpm'), 0)} rpm"],
             ["Potencia mecánica P_mec", f"{_fmt(r.get('p_mec'), 1)} kW"]]
    if (r.get("p_elec") or 0) > 0:
        prows += [["Tensión · Corriente · cosφ",
                   f"{_fmt(r.get('volt'), 0)} V · {_fmt(r.get('curr'), 0)} A · {_fmt(r.get('pf'), 2)}"],
                  ["Potencia eléctrica P_elec", f"{_fmt(r.get('p_elec'), 1)} kW"],
                  ["Eficiencia del motor η_motor", f"{_fmt(r.get('eta_motor'), 1)} %"]]
    prows += [["Potencia de diseño", f"{_fmt(r.get('design'), 1)} kW"],
              ["Eficiencia operativa η_operativa", f"{_fmt(r.get('eta_op'), 1)} %"]]
    body.append(_grid_table(["Magnitud", "Valor"], prows, styles,
                            col_widths=[7.5 * cm, 8.7 * cm]))

    # Eficiencia de proceso (según tipo), como subsección del desarrollo.
    if r.get("proc_label"):
        _pl = {"pump": ("Eficiencia hidráulica de la bomba",
                        "P_hidráulica = ρ·g·Q·H/1000 · η_bomba = P_hidráulica/P_mec (ISO 9906)."),
               "fan": ("Eficiencia aerodinámica del ventilador",
                       "P_aire = Q·Δp/1000 · η_ventilador = P_aire/P_mec (ISO 5801)."),
               "hydro": ("Eficiencia de la turbina hidráulica",
                         "P_hidráulica = ρ·g·Q·H/1000 · η = P_eléctrica/P_hidráulica (IEC 60041)."),
               "compressor": ("Eficiencia isentrópica del compresor",
                              "W_isen = ṁ·Cp·T1·(π^((k−1)/k)−1) · η = W_isen/P_mec (ASME PTC 10).")
               }.get(r["proc_label"], ("Eficiencia de proceso", ""))
        body.append(Spacer(1, 0.35 * cm))
        if _pl[1]:
            body.append(_p(f"<b>{_pl[0]}.</b> {_pl[1]}", styles, "WMBody"))
        crows = []
        if r["proc_label"] in ("pump", "hydro"):
            crows += [["Caudal Q", f"{_fmt(r.get('flow'), 3)} m³/s"],
                      ["Altura H", f"{_fmt(r.get('head'), 2)} m"]]
        elif r["proc_label"] == "fan":
            crows += [["Caudal Q", f"{_fmt(r.get('flow'), 1)} m³/s"],
                      ["Presión total Δp", f"{_fmt(r.get('dp'), 0)} Pa"]]
        if (r.get("p_proc") or 0) > 0:
            crows += [["Potencia de proceso", f"{_fmt(r.get('p_proc'), 1)} kW"]]
        crows += [[_pl[0], f"{_fmt(r.get('eta_proc'), 1)} %"]]
        body.append(_grid_table(["Magnitud", "Valor"], crows, styles,
                                col_widths=[7.5 * cm, 8.7 * cm]))
    body.append(Spacer(1, 0.35 * cm))
    body.append(_p("Metodología: mediciones directas en campo bajo condiciones reales de "
                   "operación. Par por telemetría rotativa (transmisor + antena), velocidad "
                   "por tacómetro, variables eléctricas por analizador de red y variables de "
                   "proceso (caudal, presión, temperatura) por instrumentación directa.",
                   styles, "WMBody"))
    body.append(Spacer(1, 0.5 * cm))

    # ---- 5. Marco normativo -------------------------------------------
    body.append(_section("5. Marco normativo", styles))
    body.append(_p(f"El análisis se rige por: {meta.get('norm', 'IEC 60034-2')} "
                   "(eficiencia de motores de inducción), ISO 5801 (ventiladores), "
                   "ISO 9906 (bombas), ASME PTC 10 (compresores), IEC 60041 (turbinas "
                   "hidráulicas) e ISO 20816 (evaluación de la condición mecánica).",
                   styles, "WMBody"))

    report_meta = {
        "report_title": meta.get("report_title") or "Reporte de Eficiencia",
        "format_code": "WM-EFF",
        "asset": meta.get("asset", ""),
        "asset_class": "Eficiencia de máquina rotatoria",
        "client": meta.get("client", ""),
        "location": meta.get("location", ""),
        "unit": meta.get("machine_type_label", ""),
        "prepared_by": meta.get("specialist", ""),
        "prepared_role": meta.get("specialist_role", "Analista de confiabilidad"),
        "prepared_city": meta.get("location", ""),
        "report_date": meta.get("report_date") or datetime.now().strftime("%d/%m/%Y"),
        "train_description": meta.get("train_description", ""),
    }
    return render_report_pdf(report_meta, body)


__all__ = ["build_efficiency_pdf"]
