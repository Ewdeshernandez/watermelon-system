"""
core.calibration.report
========================

Reporte PDF de calibración / curvas de linealidad, branded Watermelon/SIGA.

Se construye sobre `core.report_pdf_shell.render_report_pdf` (misma portada,
banda de encabezado, pie con version stamp y TOC que el resto de los reportes
de Watermelon). Estructura:

  1. Resumen de calibración — tabla de TODOS los lazos con veredicto PASA/FALLA.
  2. Un CERTIFICADO independiente por lazo (1 a 50), cada uno en su propia
     página, con sus datos, resultados vs criterio API 670 y su curva.

Headless — no depende de Streamlit. Entrada: metadatos + lista de lazos, cada
uno con su análisis (salida de core.calibration.engine) y su tipo de sensor.

Norma: API 670 5.ª ed. (Tabla 1 / Fig. 4) + manual del fabricante.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle,
)

from core.report_pdf_shell import render_report_pdf, make_styles
from core.calibration.curve import curve_png

_HEADER_BG = "#0f4c81"
_OK = "#16a34a"
_BAD = "#dc2626"


# ---------------------------------------------------------------------
# Helpers de flowables
# ---------------------------------------------------------------------
def _p(text: str, styles, style: str = "WMBody"):
    return Paragraph(str(text), styles[style])


def _section(title: str, styles):
    return Paragraph(title, styles["WMTOC1"])


def _subsection(title: str, styles):
    return Paragraph(title, styles["WMTOC2"])


def _section_block(num_title: str, sec: Dict[str, Any], styles,
                   sub: bool = False) -> List[Any]:
    """Bloque de sección con párrafo + imágenes. Devuelve flowables o [] si no
    hay contenido. El encabezado se mantiene JUNTO con lo primero que sigue
    (KeepTogether) para que NO quede huérfano al pie de página."""
    sec = sec or {}
    text = str(sec.get("text", "") or "").strip()
    photos = [p for p in (sec.get("photos") or []) if p.get("bytes")]
    if not text and not photos:
        return []
    head = _subsection(num_title, styles) if sub else _section(num_title, styles)
    out: List[Any] = []
    try:
        from core.reports_ext.common import photo_grid, photo_credit
        grid = photo_grid(photos, styles, cols=2, credit=photo_credit()) if photos else []
    except Exception:
        grid = []
    if text:
        out.append(KeepTogether([head, _p(text, styles)]))
        out += grid
    else:
        # solo fotos: encabezado + primera tabla de imágenes juntos
        out.append(KeepTogether([head] + grid[:1]))
        out += grid[1:]
    out.append(Spacer(1, 0.3 * cm))
    return out


def _fmt(x: Any, nd: int = 2) -> str:
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _kv_table(rows: List[Tuple[str, str]], styles) -> Table:
    data = [[Paragraph(f"<b>{k}</b>", styles["WMTableCell"]),
             Paragraph(str(v), styles["WMTableCell"])] for k, v in rows]
    t = Table(data, colWidths=[5.2 * cm, 11.0 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _grid_table(headers: List[str], rows: List[List[Any]], styles,
                col_widths: Optional[List[float]] = None,
                verdict_col: Optional[int] = None) -> Table:
    head = [Paragraph(f"<b>{h}</b>", styles["WMTableHeader"]) for h in headers]
    body = [[Paragraph(str(c), styles["WMTableCell"]) for c in r] for r in rows]
    t = Table([head] + body, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_HEADER_BG)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # colorear la celda de veredicto (PASA verde / FALLA rojo)
    if verdict_col is not None:
        for ri, r in enumerate(rows, start=1):
            val = str(r[verdict_col]).upper()
            col = _OK if val.startswith("PASA") else _BAD
            style.append(("TEXTCOLOR", (verdict_col, ri), (verdict_col, ri),
                          colors.HexColor(col)))
            style.append(("FONTNAME", (verdict_col, ri), (verdict_col, ri),
                          "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------
# Bloques de resultados por tipo de sensor
# ---------------------------------------------------------------------
def _prox_result_rows(a: Dict[str, Any]) -> List[List[str]]:
    xu = a.get("x_unit", "mil")
    rows = [
        ["Sensibilidad promedio (ASF)",
         f"{_fmt(a['asf_mv_per_x'], 2)} mV/{xu}  ({_fmt(a['asf_err_pct'], 2)} %)",
         f"nominal {_fmt(a['nominal_mv_per_x'], 2)} mV/{xu}",
         "—"],
        ["Máx. error ISF",
         f"{_fmt(a['max_isf_err_pct'], 2)} %", f"± {_fmt(a['isf_tol_pct'], 0)} %",
         "PASA" if a["pass_isf"] else "FALLA"],
        ["Máx. DSL",
         f"{_fmt(a['max_dsl_x'], 3)} {xu}", f"± {_fmt(a['dsl_tol_x'], 3)} {xu}",
         "PASA" if a["pass_dsl"] else "FALLA"],
        ["Rango lineal",
         f"{_fmt(a['span_x'], 0)} {xu}", f"≥ {_fmt(a['min_range_x'], 0)} {xu}",
         "PASA" if a["pass_range"] else "FALLA"],
        ["Linealidad (best-fit)",
         f"{_fmt(a['max_linearity_pct'], 3)} %", "referencia", "—"],
    ]
    lw = a.get("linear_window")
    if lw:
        # Vdc en convención Proximitor (negativa).
        rows.append([
            "Rango lineal útil (calibrable)",
            f"{_fmt(lw['start_x'], 0)}–{_fmt(lw['end_x'], 0)} {xu}  "
            f"({_fmt(-abs(lw['start_v']), 2)} a {_fmt(-abs(lw['end_v']), 2)} Vdc)",
            f"setpoint ~{_fmt(lw['center_x'], 0)} {xu} "
            f"({_fmt(-abs(lw['center_v']), 2)} Vdc)",
            "cumple 80 mil" if lw.get("meets_min_range") else "< 80 mil",
        ])
    return rows


def _amp_result_rows(a: Dict[str, Any]) -> List[List[str]]:
    su = a.get("sensitivity_unit", "")
    rows = [["Sensibilidad (best-fit)", f"{_fmt(a['sensitivity'], 3)} {su}",
             "—", "—"]]
    if a.get("nominal_sensitivity"):
        rows.append(["Error vs nominal", f"{_fmt(a['sens_err_pct'], 2)} %",
                     f"± {_fmt(a.get('tol_pct', 5), 0)} %", "—"])
    rows.append(["Máx. desviación de amplitud", f"{_fmt(a['max_dev_pct_fs'], 3)} %FS",
                 f"± {_fmt(a['tol_pct'], 1)} %FS",
                 "PASA" if a["pass"] else "FALLA"])
    return rows


def _freq_result_rows(a: Dict[str, Any]) -> List[List[str]]:
    band = a.get("band_hz")
    band_s = f"{band[0]:g}–{band[1]:g} Hz" if band else "—"
    return [
        ["Referencia", f"{_fmt(a['ref_sensitivity'], 3)} {a.get('sens_unit','')} "
                       f"@ {_fmt(a['ref_freq_hz'], 0)} Hz", "—", "—"],
        ["Máx. desviación", f"{_fmt(a['max_dev_db'], 2)} dB",
         f"± {_fmt(a['tol_db'], 1)} dB en {band_s}",
         "PASA" if a["pass"] else "FALLA"],
    ]


_RESULT_BUILDERS = {
    "proximity": _prox_result_rows,
    "amplitude": _amp_result_rows,
    "frequency": _freq_result_rows,
}


# ---------------------------------------------------------------------
# Constructor principal
# ---------------------------------------------------------------------
def build_calibration_pdf(*, meta: Dict[str, Any],
                          loops: List[Dict[str, Any]]) -> bytes:
    """Arma el PDF de calibración.

    meta: asset, client, location, specialist, report_date, notes...
    loops: lista de dicts, cada uno:
        tag, sensor_type ("proximity"|"accelerometer"|"velomitor"),
        kind ("linearity"|"amplitude"|"frequency"), manufacturer, model,
        serial, id_number, analysis (dict del engine), test_freq (opc.)
    """
    styles = make_styles()
    body: List[Any] = []

    # ---- Resumen -----------------------------------------------------
    body.append(_section("1. Resumen de calibración", styles))
    n = len(loops)
    n_pass = sum(1 for lp in loops if lp.get("analysis", {}).get("pass"))
    body.append(_p(
        f"Se calibraron <b>{n}</b> lazo(s). Cumplen criterio: "
        f"<b>{n_pass}/{n}</b>. Marco: API 670 5.ª ed. (Tabla 1 / Fig. 4) + "
        "manual del fabricante.", styles, "WMBody"))

    sum_rows = []
    for i, lp in enumerate(loops, 1):
        a = lp.get("analysis", {})
        stype = lp.get("sensor_type", "")
        key = _key_metric(stype, lp.get("kind", "linearity"), a)
        sum_rows.append([str(i), lp.get("tag", "—"),
                         _type_label(stype), lp.get("manufacturer", "—"),
                         key, a.get("verdict", "—")])
    body.append(Spacer(1, 0.2 * cm))
    body.append(_grid_table(
        ["#", "Tag", "Tipo", "Fabricante", "Métrica clave", "Veredicto"],
        sum_rows, styles,
        col_widths=[1.0 * cm, 2.6 * cm, 3.7 * cm, 3.1 * cm, 4.0 * cm, 2.2 * cm],
        verdict_col=5))
    if meta.get("notes"):
        body.append(Spacer(1, 0.3 * cm))
        body.append(_p(meta["notes"], styles, "WMBody"))

    # ---- 2. Hallazgos / 3. Recomendaciones (numeradas, por oración) --
    hall = [h for h in (meta.get("hallazgos") or []) if str(h).strip()]
    reco = [r for r in (meta.get("recomendaciones") or []) if str(r).strip()]
    if hall:
        body.append(Spacer(1, 0.3 * cm))
        body.append(KeepTogether([_section("2. Hallazgos", styles),
                                  _p(f"1.&nbsp;&nbsp;{hall[0]}", styles)]))
        for i, h in enumerate(hall[1:], 2):
            body.append(_p(f"{i}.&nbsp;&nbsp;{h}", styles, "WMBody"))
    if reco:
        body.append(Spacer(1, 0.2 * cm))
        body.append(KeepTogether([_section("3. Recomendaciones", styles),
                                  _p(f"1.&nbsp;&nbsp;{reco[0]}", styles)]))
        for i, r in enumerate(reco[1:], 2):
            body.append(_p(f"{i}.&nbsp;&nbsp;{r}", styles, "WMBody"))

    # ---- 4-7. Secciones descriptivas (párrafo + imágenes) ------------
    body += _section_block("4. Metodología", meta.get("sec_metodologia"), styles)
    body += _section_block("5. Información de la unidad", meta.get("sec_info_unidad"), styles)
    _dev = _section_block("6. Desarrollo del servicio", meta.get("sec_desarrollo"), styles)
    _lin = _section_block("6.1 Prueba de linealidad", meta.get("sec_linealidad"), styles, sub=True)
    _sat = _section_block("6.2 Pruebas SAT", meta.get("sec_sat"), styles, sub=True)
    # Si hay 6.1/6.2 pero no un párrafo en 6, igual emitimos el encabezado 6.
    if (_lin or _sat) and not _dev:
        _dev = [KeepTogether([_section("6. Desarrollo del servicio", styles),
                              (_lin[0] if _lin else _sat[0])])]
        # el primero ya quedó dentro del KeepTogether; evitamos duplicar
        if _lin:
            _lin = _lin[1:]
        elif _sat:
            _sat = _sat[1:]
    body += _dev + _lin + _sat

    # ---- 7. Anexo fotográfico (solo imágenes) ------------------------
    anexo = [p for p in (meta.get("anexo_photos") or []) if p.get("bytes")]
    if anexo:
        body += _section_block("7. Anexo fotográfico", {"photos": anexo}, styles)

    body.append(PageBreak())

    # ---- 8. Certificados (uno por lazo) -----------------------------
    body.append(_section("8. Certificados", styles))
    body.append(Spacer(1, 0.2 * cm))
    for i, lp in enumerate(loops, 1):
        a = lp.get("analysis", {})
        stype = lp.get("sensor_type", "")
        kind = lp.get("kind", "linearity")
        tag = lp.get("tag", f"Lazo {i}")

        body.append(_subsection(f"Certificado {i} — {tag}", styles))
        body.append(_kv_table([
            ("Tag / punto", tag),
            ("Tipo de sensor", _type_label(stype)),
            ("Fabricante", lp.get("manufacturer", "—")),
            ("Modelo", lp.get("model", "—")),
            ("Número de serie", lp.get("serial", "—")),
            ("ID / lazo", lp.get("id_number", "—")),
            ("Ensayo", _kind_label(kind)),
            ("Fecha", meta.get("report_date") or datetime.now().strftime("%d/%m/%Y")),
            ("Norma", lp.get("norm", "API 670 5.ª ed.")),
        ], styles))
        body.append(Spacer(1, 0.3 * cm))

        builder = _RESULT_BUILDERS.get(
            "proximity" if stype == "proximity" else kind, _amp_result_rows)
        rows = builder(a)
        body.append(_grid_table(
            ["Parámetro", "Medido", "Criterio", "Estado"], rows, styles,
            col_widths=[5.4 * cm, 5.0 * cm, 3.6 * cm, 2.4 * cm], verdict_col=3))
        body.append(Spacer(1, 0.3 * cm))

        png = curve_png(stype, a, kind=kind, title=f"{tag} — {_kind_label(kind)}")
        if png:
            img_h = 12.2 if stype == "proximity" else (7.0 if kind == "frequency" else 9.4)
            body.append(Image(io.BytesIO(png), width=15.0 * cm, height=img_h * cm))

        body.append(Spacer(1, 0.2 * cm))

        # Nota de zona lineal / setpoint (proximidad) — aclara el "calibrar aquí".
        lw = a.get("linear_window") if stype == "proximity" else None
        if lw:
            xu = a.get("x_unit", "mil")
            body.append(_p(
                f"<b>Zona lineal (calibrable):</b> de {_fmt(-abs(lw['start_v']), 2)} a "
                f"{_fmt(-abs(lw['end_v']), 2)} Vdc ({_fmt(lw['start_x'], 0)}–"
                f"{_fmt(lw['end_x'], 0)} {xu}), donde el ISF y el DSL cumplen. "
                f"<b>Monte/ajuste el gap para operar en ~{_fmt(-abs(lw['center_v']), 2)} "
                f"Vdc ({_fmt(lw['center_x'], 0)} {xu})</b>, centro de esa zona, de modo que "
                "el eje tenga recorrido lineal simétrico. Nota: el clásico −10 Vdc es el "
                "centro de una zona lineal completa (~80 mil); si la zona útil de este lazo "
                "es menor, el setpoint recomendado se corre en consecuencia.",
                styles, "WMBody"))

        body.append(_p(
            f"Veredicto: <b>{a.get('verdict', '—')}</b>. "
            "Resultados válidos para las condiciones del ensayo. Blanco de "
            "referencia y trazabilidad según hoja de calibración del patrón.",
            styles, "WMBody"))
        if i < len(loops):
            body.append(PageBreak())

    report_meta = {
        "report_title": meta.get("report_title") or "Certificado de Calibración de Sensores",
        "format_code": "WM-CAL",
        "asset": meta.get("asset", ""),
        "asset_class": "Calibración de sensores de vibración",
        "client": meta.get("client", ""),
        "location": meta.get("location", ""),
        "unit": meta.get("unit", ""),
        "prepared_by": meta.get("specialist", ""),
        "prepared_role": meta.get("specialist_role", "Analista de vibraciones"),
        "prepared_city": meta.get("location", ""),
        # Aprobado/Revisado por (autoridad) + consecutivo, igual que los reportes de campo.
        "reviewed_by": meta.get("reviewer", ""),
        "reviewed_role": meta.get("reviewer_role", "Machinery Diagnostic Champion"),
        "reviewed_city": meta.get("location", ""),
        "prepared_label": "Preparado por:",
        "reviewed_label": "Aprobado por:",
        "consecutive": meta.get("consecutive", ""),
        "report_date": meta.get("report_date") or datetime.now().strftime("%d/%m/%Y"),
        "train_description": meta.get("train_description",
                                      "Curvas de linealidad · API 670"),
    }
    return render_report_pdf(report_meta, body)


# ---------------------------------------------------------------------
# Etiquetas / métricas
# ---------------------------------------------------------------------
def _type_label(stype: str) -> str:
    return {"proximity": "Proximidad", "accelerometer": "Acelerómetro",
            "velomitor": "Velomitor"}.get(str(stype).lower(), str(stype) or "—")


def _kind_label(kind: str) -> str:
    return {"linearity": "Linealidad estática", "amplitude": "Linealidad de amplitud",
            "frequency": "Respuesta en frecuencia"}.get(str(kind).lower(),
                                                         str(kind) or "—")


def _key_metric(stype: str, kind: str, a: Dict[str, Any]) -> str:
    if stype == "proximity":
        xu = a.get("x_unit", "mil")
        return (f"ISF {_fmt(a.get('max_isf_err_pct'), 1)}% · "
                f"DSL {_fmt(a.get('max_dsl_x'), 2)} {xu}")
    if kind == "frequency":
        return f"Δ {_fmt(a.get('max_dev_db'), 2)} dB"
    return f"Lin {_fmt(a.get('max_dev_pct_fs'), 2)} %FS"


__all__ = ["build_calibration_pdf"]
