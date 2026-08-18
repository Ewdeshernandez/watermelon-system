"""
core.reports_ext.builders
==========================

Los 5 builders de reportes de campo. Cada uno arma metadatos + cuerpo
(flowables de common.py) y llama a `render_report_pdf` para conservar el
formato SIGA (portada + TOC + banda/pie). Headless.

  build_daily_pdf        — Reporte Diario (SIGA-FMT-136)
  build_preliminary_pdf  — Reporte Preliminar
  build_borescope_pdf    — Inspección Boroscópica (SIGA-FMT-178)
  build_alignment_pdf    — Reporte de Alineación
  build_mechanical_pdf   — Reporte Mecánico

`meta` común: client, plant, equipo, location, specialist, specialist_role,
reviewer, reviewer_role, city, report_date, visit_date, consecutive,
format_code, notes, train_description.
`content` varía por tipo (ver cada función).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Spacer

from core.report_pdf_shell import render_report_pdf
from core.reports_ext.common import (
    make_styles, p, section, bullets, numbered_plan, kv_table, two_col_kv,
    grid_table, photo_grid, severity_table, severity_legend, today_str,
    photo_credit,
)

_CITY = "Cajicá, Cundinamarca · Colombia"


# ---------------------------------------------------------------------
# Helpers de meta
# ---------------------------------------------------------------------
def _shell_meta(meta: Dict[str, Any], *, title: str, format_code: str,
                format_version: str, asset_class: str) -> Dict[str, Any]:
    return {
        "report_title": title,
        "format_code": meta.get("format_code") or format_code,
        "format_version": meta.get("format_version") or format_version,
        # Línea grande de la portada = EQUIPO (lo que el técnico escribió).
        # Antes se usaba asset_class ("Reporte diario de servicio") y repetía
        # el título. Fallback a asset_class solo si no hay equipo. v3.31.493.
        "asset": meta.get("equipo") or asset_class,
        "asset_class": "",
        "client": meta.get("client", ""),
        "location": meta.get("plant") or meta.get("location", ""),
        "train_description": meta.get("train_description", ""),
        "prepared_by": meta.get("specialist", ""),
        "prepared_role": meta.get("specialist_role", "Machinery Diagnostics Engineer"),
        "prepared_city": meta.get("city", _CITY),
        "reviewed_by": meta.get("reviewer", ""),
        "reviewed_role": meta.get("reviewer_role", "Machinery Diagnostic Champion"),
        "reviewed_city": meta.get("city", _CITY),
        "report_date": meta.get("report_date") or today_str(),
        "consecutive": meta.get("consecutive", ""),
        "prepared_label": "Preparado por:",
        "reviewed_label": "Revisado por:",
    }


def _service_data(meta: Dict[str, Any], styles, servicio: str = "") -> List[Any]:
    return [
        section("1. Datos del servicio", styles),
        two_col_kv([
            ("CLIENTE", meta.get("client", "—"), "SERVICIO", servicio or meta.get("servicio", "—")),
            ("PLANTA", meta.get("plant", "—"), "FECHA", meta.get("report_date") or today_str()),
            ("UBICACIÓN", meta.get("location", "—"), "EQUIPO", meta.get("equipo", "—")),
        ], styles),
        Spacer(1, 0.4 * cm),
    ]


def _photos(content: Dict[str, Any], styles, num: str) -> List[Any]:
    photos = content.get("photos") or []
    if not photos:
        return []
    out = [section(f"{num}. Registro fotográfico", styles)]
    out += photo_grid(photos, styles, cols=content.get("photo_cols", 2),
                      credit=photo_credit())
    out.append(Spacer(1, 0.3 * cm))
    return out


# =====================================================================
# 1. REPORTE DIARIO (SIGA-FMT-136)
# =====================================================================
def build_daily_pdf(*, meta: Dict[str, Any], content: Dict[str, Any]) -> bytes:
    """content: servicio, hallazgos[list], observaciones[list],
    plan[list of {title, items}], photos[list of {bytes, caption}]."""
    styles = make_styles()
    body: List[Any] = []
    # 1. Datos del servicio
    body += _service_data(meta, styles, content.get("servicio", ""))

    # 2. Actividades realizadas
    body.append(section("2. Actividades realizadas", styles))
    acts = content.get("plan") or content.get("actividades") or []
    if acts:
        body += numbered_plan(acts, styles)
    else:
        body += bullets(["—"], styles)
    body.append(Spacer(1, 0.3 * cm))

    # 3. Hallazgos
    body.append(section("3. Hallazgos", styles))
    body += bullets(content.get("hallazgos", []) or ["—"], styles)
    body.append(Spacer(1, 0.3 * cm))

    # 4. Observaciones
    body.append(section("4. Observaciones", styles))
    body += bullets(content.get("observaciones", []) or ["—"], styles)
    body.append(Spacer(1, 0.3 * cm))

    # 5. Registro fotográfico
    body += _photos(content, styles, "5")

    return render_report_pdf(
        _shell_meta(meta, title="Reporte Diario", format_code="SIGA-FMT-136",
                    format_version="003", asset_class="Reporte diario de servicio"),
        body)


# =====================================================================
# 2. REPORTE PRELIMINAR
# =====================================================================
def build_preliminary_pdf(*, meta: Dict[str, Any], content: Dict[str, Any]) -> bytes:
    """content: objeto(text), resumen(text), hallazgos[list], observaciones[list],
    recomendaciones[list], photos[list]."""
    styles = make_styles()
    body: List[Any] = []
    body += _service_data(meta, styles, content.get("servicio", ""))

    if content.get("objeto"):
        body.append(section("2. Objeto y alcance", styles))
        body.append(p(content["objeto"], styles)); body.append(Spacer(1, 0.2 * cm))
    if content.get("resumen"):
        body.append(section("3. Resumen preliminar", styles))
        body.append(p(content["resumen"], styles)); body.append(Spacer(1, 0.2 * cm))

    body.append(section("4. Hallazgos preliminares", styles))
    body += bullets(content.get("hallazgos", []) or ["—"], styles)
    body.append(Spacer(1, 0.3 * cm))

    if content.get("observaciones"):
        body.append(section("5. Observaciones", styles))
        body += bullets(content["observaciones"], styles)
        body.append(Spacer(1, 0.3 * cm))

    body.append(section("6. Recomendaciones", styles))
    body += bullets(content.get("recomendaciones", []) or ["—"], styles)
    body.append(Spacer(1, 0.2 * cm))
    body.append(p("Nota: este es un reporte <b>preliminar</b>; los resultados "
                  "definitivos se emiten en el reporte técnico final.", styles))

    body += _photos(content, styles, "7")

    return render_report_pdf(
        _shell_meta(meta, title="Reporte Preliminar", format_code="SIGA-FMT-PRE",
                    format_version="1", asset_class="Reporte preliminar"),
        body)


# =====================================================================
# 3. INSPECCIÓN BOROSCÓPICA (SIGA-FMT-178)
# =====================================================================
def build_borescope_pdf(*, meta: Dict[str, Any], content: Dict[str, Any]) -> bytes:
    """content: introduccion, antecedentes, hallazgos[list], recomendaciones[list],
    metodologia, equipo(text), severity_rows[list of {access,findings,severity,
    comment,image_bytes}], photos[list]."""
    styles = make_styles()
    body: List[Any] = []

    body.append(section("1. Introducción y alcance", styles))
    body.append(p(content.get("introduccion", "—"), styles)); body.append(Spacer(1, 0.2 * cm))

    body.append(section("2. Antecedentes", styles))
    body.append(p(content.get("antecedentes", "—"), styles)); body.append(Spacer(1, 0.2 * cm))

    body.append(section("3. Hallazgos", styles))
    body += bullets(content.get("hallazgos", []) or ["—"], styles)
    body.append(Spacer(1, 0.3 * cm))

    body.append(section("4. Recomendaciones finales", styles))
    body += bullets(content.get("recomendaciones", []) or ["—"], styles)
    body.append(Spacer(1, 0.3 * cm))

    body.append(section("5. Metodología", styles))
    body.append(p(content.get("metodologia",
                  "De acuerdo con el procedimiento SIGA-PRD-124, se realiza la "
                  "inspección visual remota identificando puntos críticos y "
                  "asignando nivel de severidad."), styles))
    if content.get("equipo"):
        body.append(p(content["equipo"], styles))
    body.append(Spacer(1, 0.3 * cm))

    body.append(section("6. Desarrollo del servicio", styles))
    body.append(p("Condición de severidades:", styles))
    body.append(severity_legend(styles))
    body.append(Spacer(1, 0.3 * cm))
    rows = content.get("severity_rows") or []
    if rows:
        body.append(severity_table(rows, styles))

    body += _photos(content, styles, "7")

    return render_report_pdf(
        _shell_meta(meta, title="Reporte Servicio de Inspección Boroscópica",
                    format_code="SIGA-FMT-178", format_version="2",
                    asset_class="Inspección boroscópica"),
        body)


# =====================================================================
# 4. REPORTE DE ALINEACIÓN
# =====================================================================
def build_alignment_pdf(*, meta: Dict[str, Any], content: Dict[str, Any]) -> bytes:
    """content: metodo, objeto, tolerancia_offset, tolerancia_ang,
    align_rows[list of [param, as_found, as_left, tol, estado]],
    shims(text), hallazgos[list], observaciones[list], recomendaciones[list],
    photos[list]."""
    styles = make_styles()
    body: List[Any] = []
    body += _service_data(meta, styles, content.get("servicio", "Alineación de ejes"))

    body.append(section("2. Método y alcance", styles))
    body.append(p(content.get("metodo",
                  "Alineación por láser de doble haz; convención acople "
                  "fijo (estacionaria) → móvil."), styles))
    if content.get("objeto"):
        body.append(p(content["objeto"], styles))
    body.append(Spacer(1, 0.3 * cm))

    rows = content.get("align_rows") or [
        ["Offset vertical", "", "", "", ""],
        ["Offset horizontal", "", "", "", ""],
        ["Angularidad vertical", "", "", "", ""],
        ["Angularidad horizontal", "", "", "", ""],
    ]
    body.append(section("3. Condición encontrada / dejada (As found / As left)", styles))
    body.append(grid_table(
        ["Parámetro", "As found", "As left", "Tolerancia", "Estado"], rows, styles,
        col_widths=[5.0 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm, 2.5 * cm]))
    body.append(Spacer(1, 0.3 * cm))
    if content.get("shims"):
        body.append(section("4. Correcciones (shims / movimientos)", styles))
        body.append(p(content["shims"], styles)); body.append(Spacer(1, 0.2 * cm))

    if content.get("hallazgos"):
        body.append(section("5. Hallazgos", styles))
        body += bullets(content["hallazgos"], styles); body.append(Spacer(1, 0.2 * cm))
    if content.get("observaciones"):
        body.append(section("6. Observaciones", styles))
        body += bullets(content["observaciones"], styles); body.append(Spacer(1, 0.2 * cm))
    if content.get("recomendaciones"):
        body.append(section("7. Recomendaciones", styles))
        body += bullets(content["recomendaciones"], styles); body.append(Spacer(1, 0.2 * cm))

    body += _photos(content, styles, "8")

    return render_report_pdf(
        _shell_meta(meta, title="Reporte de Alineación", format_code="SIGA-FMT-ALI",
                    format_version="1", asset_class="Alineación de ejes"),
        body)


# =====================================================================
# 5. REPORTE MECÁNICO
# =====================================================================
def build_mechanical_pdf(*, meta: Dict[str, Any], content: Dict[str, Any]) -> bytes:
    """content: objeto, actividades[list of {title, items}],
    metrologia_rows[list of [param, valor, unidad, referencia, estado]],
    hallazgos[list], observaciones[list], recomendaciones[list], photos[list]."""
    styles = make_styles()
    body: List[Any] = []
    body += _service_data(meta, styles, content.get("servicio", "Intervención mecánica"))

    if content.get("objeto"):
        body.append(section("2. Objeto y alcance", styles))
        body.append(p(content["objeto"], styles)); body.append(Spacer(1, 0.3 * cm))

    acts = content.get("actividades") or []
    if acts:
        body.append(section("3. Actividades ejecutadas", styles))
        body += numbered_plan(acts, styles)

    metro = content.get("metrologia_rows") or []
    if metro:
        body.append(section("4. Mediciones / Metrología", styles))
        body.append(grid_table(
            ["Parámetro", "Valor", "Unidad", "Referencia", "Estado"], metro, styles,
            col_widths=[4.8 * cm, 2.8 * cm, 2.4 * cm, 3.4 * cm, 2.8 * cm]))
        body.append(Spacer(1, 0.3 * cm))

    body.append(section("5. Hallazgos", styles))
    body += bullets(content.get("hallazgos", []) or ["—"], styles); body.append(Spacer(1, 0.2 * cm))
    if content.get("observaciones"):
        body.append(section("6. Observaciones", styles))
        body += bullets(content["observaciones"], styles); body.append(Spacer(1, 0.2 * cm))
    body.append(section("7. Recomendaciones", styles))
    body += bullets(content.get("recomendaciones", []) or ["—"], styles); body.append(Spacer(1, 0.2 * cm))

    body += _photos(content, styles, "8")

    return render_report_pdf(
        _shell_meta(meta, title="Reporte Mecánico", format_code="SIGA-FMT-MEC",
                    format_version="1", asset_class="Intervención mecánica"),
        body)


BUILDERS = {
    "diario": build_daily_pdf,
    "preliminar": build_preliminary_pdf,
    "boroscopia": build_borescope_pdf,
    "alineacion": build_alignment_pdf,
    "mecanico": build_mechanical_pdf,
}

__all__ = [
    "build_daily_pdf", "build_preliminary_pdf", "build_borescope_pdf",
    "build_alignment_pdf", "build_mechanical_pdf", "BUILDERS",
]
