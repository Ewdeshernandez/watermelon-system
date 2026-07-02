"""
core.briefing_report_pdf — PDF del Briefing por activo (F2)
===========================================================

Arma el PDF del Briefing Semanal/Mensual de UN activo con el MISMO formato
profesional que los Reports al cliente: portada SIGA-style (logo + título +
bloque del activo + firmas), banda de encabezado con código de formato,
Tabla de Contenido clickeable y pie con disclaimer. El contenido es el del
briefing (resumen gerencial, mapa de sensores, diagnóstico, recomendaciones,
tabla de canales Overall+1X/2X y figuras: tendencia/espectro/onda/órbita).

La carcasa profesional vive en core.report_pdf_shell (compartida). Este módulo
solo arma el BODY y delega el montaje. Recibe TODO como parámetros (figuras +
textos + KPIs + canales) — la obtención de datos vive en el orquestador
(briefing_builder), así queda testeable y headless.

API:
    generate_briefing_pdf(instance_id, tag, train, period_label, health, kpis,
                          figures, summary, diagnosis, recommendations,
                          channels=None, sensor_map_png=None,
                          meta_extra=None) -> bytes
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional

# Paleta de severidad (coherente con la app)
_GREEN = "#1D9E75"
_AMBER = "#EF9F27"
_RED = "#E24B4A"
_GREEN_BG = "#E1F5EE"
_AMBER_BG = "#FAEEDA"
_RED_BG = "#FCEBEB"
_LINE = "#e6ebf2"

_FIG_CAPTIONS = {
    "trend": "Tendencia overall",
    "spectrum": "Espectro — canales apilados",
    "waveform": "Forma de onda — canales apilados",
    "orbit": "Órbitas por cojinete",
}
_FIG_ORDER = ["trend", "spectrum", "waveform", "orbit"]

_MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _fecha_es(val) -> str:
    """Fecha larga en español: '1 de julio de 2026'. Fallback: hoy."""
    from datetime import datetime, date
    d = None
    try:
        if isinstance(val, (datetime, date)):
            d = val if isinstance(val, date) and not isinstance(val, datetime) else val.date() if isinstance(val, datetime) else val
        elif val:
            d = datetime.fromisoformat(str(val).replace("Z", "+00:00")).date()
    except Exception:
        d = None
    if d is None:
        d = date.today()
    try:
        return f"{d.day} de {_MESES_ES[d.month - 1]} de {d.year}"
    except Exception:
        return ""


def _sev_colors(status: str):
    s = (status or "").lower()
    if "danger" in s or "crít" in s or "crit" in s:
        return _RED, _RED_BG
    if "alarma" in s or "alert" in s:
        return _AMBER, _AMBER_BG
    return _GREEN, _GREEN_BG


def generate_briefing_pdf(
    instance_id: str,
    tag: str,
    train: str,
    period_label: str,
    health: Dict[str, Any],
    kpis: Dict[str, Any],
    figures: Dict[str, Optional[bytes]],
    summary: str = "",
    diagnosis: str = "",
    recommendations: Optional[List[str]] = None,
    channels: Optional[List[Dict[str, Any]]] = None,
    sensor_map_png: Optional[bytes] = None,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Devuelve los bytes del PDF del briefing del activo, en formato pro."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY, TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Image, Paragraph, Spacer, Table, TableStyle,
    )

    from core.report_pdf_shell import (
        BOLD, REGULAR, make_styles, paragraph_safe, render_markdown_flowables,
        render_report_pdf,
    )

    styles = make_styles()

    # ---- meta de portada (firmas/consecutivo opcionales vía meta_extra) ----
    meta_extra = meta_extra or {}
    meta: Dict[str, Any] = {
        "report_title": f"BRIEFING {period_label.upper()}",
        "format_code": "WMS-FMT-002",
        "format_version": "1",
        "unit": tag,
        "train_description": train or "",
        "period": period_label,
    }
    meta.update(meta_extra)  # prepared_by / reviewed_by / client / report_date / etc.

    # =================================================================
    # BODY — secciones (los headings WMTOC1/WMTOC2 entran a la TOC)
    # =================================================================
    st_body = ParagraphStyle("bfBody", parent=styles["WMBody"])
    st_meta = ParagraphStyle("bfMeta", fontName="Courier", fontSize=8.5,
                             textColor=colors.HexColor("#94a3b8"), leading=11)
    st_cap = ParagraphStyle("bfCap", fontName=REGULAR, fontSize=8.5,
                            textColor=colors.HexColor("#94a3b8"), spaceBefore=2,
                            spaceAfter=8, alignment=TA_RIGHT)

    body: List[Any] = []

    # ---- Banner de estado + KPIs ----
    hcolor = colors.HexColor(health.get("color", "#94a3b8"))

    def _kpi(label, value, vcolor=None):
        lab = ParagraphStyle("kl", fontName=BOLD, fontSize=8,
                             textColor=colors.HexColor("#94a3b8"))
        val = ParagraphStyle("kv", fontName="Courier-Bold", fontSize=13,
                             textColor=vcolor or colors.HexColor("#0f172a"), leading=15)
        return [Paragraph(label.upper(), lab), Paragraph(str(value), val)]

    body.append(Paragraph("RESUMEN EJECUTIVO", styles["WMTOC1"]))
    kpi_tbl = Table([[
        _kpi("Salud", health.get("score", "—"), hcolor),
        _kpi("Estado", kpis.get("status", "—")),
        _kpi("Velocidad", kpis.get("speed", "—")),
        _kpi("Alarmas", kpis.get("alarms", 0),
             colors.HexColor(_RED) if kpis.get("alarms", 0) else colors.HexColor(_GREEN)),
    ]], colWidths=[4.2 * cm] * 4)
    kpi_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(_LINE)),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor(_LINE)),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))
    body.append(kpi_tbl)
    body.append(Paragraph(f"Zona ISO 20816: {health.get('zone', '—')}", st_meta))
    body.append(Spacer(1, 6))

    # El resumen puede venir como markdown del AI (###, **, listas): se
    # renderiza nativo para que el cliente NO vea sintaxis cruda.
    if summary:
        body.extend(render_markdown_flowables(summary, styles))

    # ---- Mapa de sensores ----
    if sensor_map_png:
        body.append(Paragraph("MAPA DE SENSORES", styles["WMTOC1"]))
        try:
            img = Image(BytesIO(sensor_map_png), width=17.5 * cm, height=12.5 * cm,
                        kind="proportional")
            img.hAlign = "CENTER"
            body.append(img)
            body.append(Paragraph(
                f"Figura — Mapa de sensores y severidad por plano · {tag}", st_cap))
        except Exception:
            pass

    # ---- Diagnóstico ----
    if diagnosis:
        body.append(Paragraph("DIAGNÓSTICO", styles["WMTOC1"]))
        body.extend(render_markdown_flowables(diagnosis, styles))

    # ---- Recomendaciones ----
    if recommendations:
        body.append(Paragraph("RECOMENDACIONES", styles["WMTOC1"]))
        for i, rec in enumerate(recommendations, start=1):
            body.append(Paragraph(f"{i}. {paragraph_safe(rec)}", st_body))

    # ---- Canales (Overall + 1X/2X) ----
    if channels:
        body.append(Paragraph("CANALES — OVERALL + VECTORES 1X / 2X (API 670)", styles["WMTOC1"]))
        st_cell = ParagraphStyle("c", fontName=REGULAR, fontSize=8.5,
                                 textColor=colors.HexColor("#111827"), leading=11)
        st_cn = ParagraphStyle("cn", fontName="Courier", fontSize=8.5,
                               textColor=colors.HexColor("#111827"), leading=11)
        st_hd = ParagraphStyle("hd", fontName=BOLD, fontSize=8.5,
                               textColor=colors.HexColor("#0f172a"), leading=11)
        head = ["Estado", "Canal", "Ubicación", "Overall", "Unit", "1X", "2X"]
        data = [[Paragraph(h, st_hd) for h in head]]
        row_styles = []
        for i, c in enumerate(channels, start=1):
            fg, bg = _sev_colors(c.get("status", ""))
            data.append([
                Paragraph(c.get("status", "—"), st_cell),
                Paragraph(c.get("sensor_label", "—"), st_cn),
                Paragraph(c.get("plane_label", "—"), st_cell),
                Paragraph(str(c.get("value", "—")), st_cn),
                Paragraph(c.get("unit", ""), st_cell),
                Paragraph(str(c.get("x1_amp", "—")), st_cn),
                Paragraph(str(c.get("x2_amp", "—")), st_cn),
            ])
            row_styles.append(("TEXTCOLOR", (0, i), (0, i), colors.HexColor(fg)))
            row_styles.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(bg)))
        ctbl = Table(data, colWidths=[2.0 * cm, 2.4 * cm, 3.4 * cm, 2.2 * cm,
                                      1.9 * cm, 2.0 * cm, 2.0 * cm])
        ctbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(_LINE)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ] + row_styles))
        body.append(ctbl)

    # ---- Figuras ----
    # Ciclo 23.170 — Tendencias SEPARADAS por sección (CRF-TRF, Generador, ...)
    # con líneas de alarma/danger, y caption numerado en español (figura,
    # descripción, fecha, equipo). Fallback al trend único si no hay 'trends'.
    _trends = (figures or {}).get("trends") or []
    if not _trends and (figures or {}).get("trend"):
        _trends = [{"section": "", "unit": "", "descr": "", "png": figures["trend"]}]
    _others = [k for k in ("spectrum", "waveform", "orbit") if (figures or {}).get(k)]
    if _trends or _others:
        body.append(Paragraph("FIGURAS Y ANÁLISIS", styles["WMTOC1"]))
        _fecha = _fecha_es(meta.get("report_date"))
        _equipo = f"Unidad {tag}"
        _n = 0

        def _add_fig(png, head, big_h):
            nonlocal _n
            body.append(Paragraph(head, styles["WMTOC2"]))
            try:
                img = Image(BytesIO(png), width=17.0 * cm, height=big_h,
                            kind="proportional")
                img.hAlign = "CENTER"
                body.append(img)
                _n += 1
                _cap = f"Figura {_n}. {head}"
                if _fecha:
                    _cap += f", {_fecha}"
                _cap += f" · {_equipo}"
                body.append(Paragraph(_cap, st_cap))
            except Exception:
                pass

        for t in _trends:
            sec = (t.get("section") or "").strip()
            descr = (t.get("descr") or "").strip()
            head = "Tendencia de vibración"
            if sec:
                head += f" — {sec}"
            if descr:
                head += f" ({descr})"
            _add_fig(t.get("png"), head, 6.2 * cm)

        for key in _others:
            cap = _FIG_CAPTIONS.get(key, key.title())
            _add_fig(figures[key],
                     cap,
                     10.6 * cm if key in ("spectrum", "waveform") else 8.0 * cm)

    return render_report_pdf(meta, body)


__all__ = ["generate_briefing_pdf"]
