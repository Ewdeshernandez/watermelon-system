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
from reportlab.platypus import KeepTogether, PageBreak, Spacer

from core.report_pdf_shell import render_report_pdf
from core.reports_ext.common import (
    make_styles, p, section, bullets, numbered_plan, numbered_list,
    machine_info_table, inspection_status_table, kv_table, two_col_kv,
    grid_table, photo_grid, severity_table, severity_blocks, severity_legend,
    today_str, photo_credit, titled_table, free_blocks_flowables,
    activities_progress_table,
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

    # 3 y 4 con viñetas NUMÉRICAS (pedido Ewdes)
    body.append(section("3. Hallazgos", styles))
    body += numbered_list(content.get("hallazgos", []) or ["—"], styles)
    body.append(Spacer(1, 0.3 * cm))

    body.append(section("4. Recomendaciones finales", styles))
    body += numbered_list(content.get("recomendaciones", []) or ["—"], styles)
    body.append(Spacer(1, 0.3 * cm))

    body.append(section("5. Metodología", styles))
    body.append(p(content.get("metodologia",
                  "De acuerdo con el procedimiento SIGA-PRD-124, se realiza la "
                  "inspección visual remota identificando puntos críticos y "
                  "asignando nivel de severidad."), styles))
    if content.get("equipo"):
        body.append(p(content["equipo"], styles))

    # Info de la máquina (Tabla 1) + imágenes del equipo en Metodología
    turbine = content.get("machine_info") or {}
    borescope = content.get("borescope_info") or {}
    if turbine or borescope:
        body.append(Spacer(1, 0.2 * cm))
        body.append(p("<b>Tabla 1.</b> Información de equipos y herramientas.", styles))
        body += machine_info_table(turbine, borescope, styles)

    meth_imgs = content.get("methodology_images") or []
    if meth_imgs:
        body += photo_grid(meth_imgs, styles, cols=content.get("methodology_photo_cols", 2),
                           credit=photo_credit())

    # Tabla de inspección de la máquina (puntos y estado)
    insp = content.get("inspection_rows") or []
    if insp:
        body.append(Spacer(1, 0.2 * cm))
        body.append(p("<b>Puntos de inspección y estado.</b>", styles))
        body.append(inspection_status_table(insp, styles))
    body.append(Spacer(1, 0.3 * cm))

    # 6. Desarrollo del servicio — encabezado + leyenda JUNTOS en la misma página
    body.append(KeepTogether([
        section("6. Desarrollo del servicio", styles),
        p("Condición de severidades:", styles),
        severity_legend(styles),
    ]))
    body.append(Spacer(1, 0.3 * cm))
    rows = content.get("severity_rows") or []
    if rows:
        # severity_blocks (no severity_table): saca las imágenes de la celda a
        # una rejilla que SÍ se parte entre páginas → evita el error de "fila
        # demasiado alta" cuando un acceso tiene muchas fotos. v3.31.508.
        body += severity_blocks(rows, styles)

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
    """Reporte de Alineación — 7 secciones. content:
      introduccion, antecedentes (text)
      hallazgos, recomendaciones (list → numeradas)
      met_text, met_photos[list], met_equipos[list of {title, rows:[[campo,valor]]}]
      dev_blocks[list de bloques orden libre: text/images/table]
      anexo_photos[list]"""
    from core.reports_ext.common import (
        numbered_list as _numbered, titled_table as _titled,
        free_blocks_flowables as _freeblocks, photo_grid as _pg,
        photo_credit as _pc,
    )
    styles = make_styles()
    body: List[Any] = []

    # Numeración CONTINUA de figuras en todo el reporte (Figura 1, 2, 3…),
    # conservando el nombre. Antes cada bloque numeraba desde 1 → todas "Figura 1".
    import re as _re
    _fig = [0]

    def _renum(ph):
        _fig[0] += 1
        _t = str(ph.get("title") or ph.get("caption") or "")
        _t = _re.sub(r"^\s*Figura\s*\d+\.?\s*", "", _t).strip()
        ph["caption"] = (f"Figura {_fig[0]}. {_t}").rstrip(". ")

    for _ph in (content.get("met_photos") or []):
        if _ph.get("bytes"):
            _renum(_ph)
    for _b in (content.get("dev_blocks") or []):
        if _b.get("type") == "images":
            for _ph in (_b.get("photos") or []):
                if _ph.get("bytes"):
                    _renum(_ph)

    body.append(section("1. Introducción y alcance", styles))
    body.append(p(content.get("introduccion", "—"), styles)); body.append(Spacer(1, 0.2 * cm))

    body.append(section("2. Antecedentes", styles))
    body.append(p(content.get("antecedentes", "—"), styles)); body.append(Spacer(1, 0.2 * cm))

    body.append(section("3. Hallazgos", styles))
    if str(content.get("hall_intro", "")).strip():
        body.append(p(content["hall_intro"], styles)); body.append(Spacer(1, 0.1 * cm))
    body += _numbered(content.get("hallazgos", []) or ["—"], styles)
    body.append(Spacer(1, 0.3 * cm))

    body.append(section("4. Recomendaciones finales", styles))
    if str(content.get("reco_intro", "")).strip():
        body.append(p(content["reco_intro"], styles)); body.append(Spacer(1, 0.1 * cm))
    body += _numbered(content.get("recomendaciones", []) or ["—"], styles)
    body.append(Spacer(1, 0.3 * cm))

    # 5. Metodología: texto + imágenes + tablas de equipo (conductor/conducido/alineador)
    body.append(section("5. Metodología", styles))
    if str(content.get("met_text", "")).strip():
        body.append(p(content["met_text"], styles)); body.append(Spacer(1, 0.2 * cm))
    _mp = [ph for ph in (content.get("met_photos") or []) if ph.get("bytes")]
    if _mp:
        body += _pg(_mp, styles, cols=2, credit=_pc()); body.append(Spacer(1, 0.2 * cm))
    for eq in (content.get("met_equipos") or []):
        rows = [r for r in (eq.get("rows") or []) if any(str(c).strip() for c in r)]
        if rows or eq.get("title"):
            # 4 columnas (2 pares Campo/Valor por fila) para que no queden largas.
            pairs = []
            for _i in range(0, len(rows), 2):
                _a = rows[_i]
                _b = rows[_i + 1] if _i + 1 < len(rows) else ["", ""]
                _ca = f"<b>{_a[0]}</b>" if str(_a[0]).strip() else ""
                _cb = f"<b>{_b[0]}</b>" if str(_b[0]).strip() else ""
                pairs.append([_ca, (_a[1] if len(_a) > 1 else ""),
                              _cb, (_b[1] if len(_b) > 1 else "")])
            body.append(_titled(eq.get("title", ""),
                                ["Campo", "Valor", "Campo", "Valor"], pairs, styles,
                                col_widths=[3.0 * cm, 5.1 * cm, 3.0 * cm, 5.1 * cm]))
            body.append(Spacer(1, 0.2 * cm))
    body.append(Spacer(1, 0.1 * cm))

    # 6. Desarrollo del servicio: ORDEN LIBRE (texto/imágenes/tablas)
    body.append(section("6. Desarrollo del servicio", styles))
    _blocks = _freeblocks(content.get("dev_blocks") or [], styles, credit=_pc())
    if _blocks:
        body += _blocks
    else:
        body.append(p("—", styles))
    body.append(Spacer(1, 0.2 * cm))

    # 7. Anexos — NOMBRES de documentos adjuntos (no imágenes).
    docs = [d for d in (content.get("anexo_docs") or []) if str(d).strip()]
    if docs:
        body.append(section("7. Anexos", styles))
        body.append(p("Documentos adjuntos al reporte:", styles))
        body += _numbered(docs, styles)

    return render_report_pdf(
        _shell_meta(meta, title="Reporte de Alineación", format_code="SIGA-FMT-ALI",
                    format_version="1", asset_class="Alineación de ejes"),
        body)


# =====================================================================
# 5. REPORTE MECÁNICO
# =====================================================================
def build_consolidated_pdf(*, meta: Dict[str, Any], content: Dict[str, Any]) -> bytes:
    """Reporte Consolidado Final — 9 secciones.

    content:
      antecedentes(text), tech_rows[list of [campo, valor]], estado_photos[list],
      objetivo(text), act_rows[list of {tipo, descripcion, avance}],
      recurso_rows[list of [recurso, detalle]], docs_ref[list of str],
      dev_blocks[free-order blocks], anexo_docs[list of str].
    """
    styles = make_styles()
    body: List[Any] = []

    # 1. Datos del servicio
    body += _service_data(meta, styles, content.get("servicio", ""))

    # 2. Antecedentes (solo texto)
    body.append(section("2. Antecedentes", styles))
    if str(content.get("antecedentes", "")).strip():
        body.append(p(content["antecedentes"], styles))
    else:
        body.append(p("—", styles))
    body.append(Spacer(1, 0.3 * cm))

    # 3. Datos técnicos y estado del equipo (tabla técnica + foto del equipo)
    body.append(section("3. Datos técnicos y estado del equipo", styles))
    tech = [r for r in (content.get("tech_rows") or [])
            if any(str(c).strip() for c in r)]
    if tech:
        body.append(titled_table("Datos técnicos", ["Campo", "Valor"], tech, styles,
                                 col_widths=[6.2 * cm, 10.0 * cm]))
        body.append(Spacer(1, 0.25 * cm))
    estado = [ph for ph in (content.get("estado_photos") or []) if ph.get("bytes")]
    if estado:
        body += photo_grid(estado, styles, cols=content.get("estado_cols", 2),
                           credit=photo_credit())
    body.append(Spacer(1, 0.3 * cm))

    # 4. Objetivo del trabajo (texto)
    body.append(section("4. Objetivo del trabajo", styles))
    body.append(p(content.get("objetivo") or "—", styles))
    body.append(Spacer(1, 0.3 * cm))

    # 5. Descripción de las actividades (tabla con % de avance)
    body.append(section("5. Descripción de las actividades", styles))
    act_rows = [r for r in (content.get("act_rows") or [])
                if str(r.get("descripcion", "")).strip()]
    if act_rows:
        body.append(activities_progress_table(act_rows, styles))
    else:
        body.append(p("—", styles))
    body.append(Spacer(1, 0.3 * cm))

    # 6. Recurso utilizado para realizar la actividad
    body.append(section("6. Recurso utilizado para realizar la actividad", styles))
    rec = [r for r in (content.get("recurso_rows") or [])
           if any(str(c).strip() for c in r)]
    if rec:
        body.append(titled_table("", ["Recurso", "Detalle"], rec, styles,
                                 col_widths=[6.2 * cm, 10.0 * cm]))
    else:
        body.append(p("—", styles))
    body.append(Spacer(1, 0.3 * cm))

    # 7. Documentos de referencia
    body.append(section("7. Documentos de referencia", styles))
    docs = [d for d in (content.get("docs_ref") or []) if str(d).strip()]
    if docs:
        body += numbered_list(docs, styles)
    else:
        body.append(p("—", styles))
    body.append(Spacer(1, 0.3 * cm))

    # 8. Desarrollo y descripción detallada de las actividades (orden libre)
    body.append(section("8. Desarrollo y descripción detallada de las actividades",
                        styles))
    dev = free_blocks_flowables(content.get("dev_blocks") or [], styles,
                                credit=photo_credit())
    if dev:
        body += dev
    else:
        body.append(p("—", styles))
    body.append(Spacer(1, 0.3 * cm))

    # 9. Anexos (nombres de documentos)
    body.append(section("9. Anexos", styles))
    anexos = [a for a in (content.get("anexo_docs") or []) if str(a).strip()]
    if anexos:
        body += numbered_list(anexos, styles)
    else:
        body.append(p("—", styles))

    return render_report_pdf(
        _shell_meta(meta, title="Reporte Consolidado Final",
                    format_code="SIGA-FMT-CON", format_version="1",
                    asset_class="Reporte consolidado final"),
        body)


# Alias de compatibilidad (nombre viejo).
build_mechanical_pdf = build_consolidated_pdf


BUILDERS = {
    "diario": build_daily_pdf,
    "preliminar": build_preliminary_pdf,
    "boroscopia": build_borescope_pdf,
    "alineacion": build_alignment_pdf,
    "consolidado": build_consolidated_pdf,
    "mecanico": build_consolidated_pdf,  # compat
}

__all__ = [
    "build_daily_pdf", "build_preliminary_pdf", "build_borescope_pdf",
    "build_alignment_pdf", "build_consolidated_pdf", "build_mechanical_pdf",
    "BUILDERS",
]
