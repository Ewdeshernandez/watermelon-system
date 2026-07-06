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
    wf_history: Optional[List[Dict[str, Any]]] = None,
    overall_history: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Devuelve los bytes del PDF del briefing del activo, en formato pro."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Image, KeepTogether, PageBreak, Paragraph, Spacer,
        Table, TableStyle,
    )

    from core.report_pdf_shell import (
        BOLD, REGULAR, make_styles, paragraph_safe, render_markdown_flowables,
        render_report_pdf,
    )

    styles = make_styles()

    # ---- meta de portada (firmas/consecutivo opcionales vía meta_extra) ----
    meta_extra = meta_extra or {}
    meta: Dict[str, Any] = {
        "report_title": f"REPORTE {period_label.upper()}",
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

    # SEMÁFORO del estado (el mismo de la app: verde CONDICIÓN ACEPTABLE /
    # ámbar ATENCIÓN / rojo ACCIÓN REQUERIDA / gris Sin datos).
    _status_txt = str(kpis.get("status", "—"))
    _sl = _status_txt.lower()
    if "crít" in _sl or "critic" in _sl or "acción" in _sl or "danger" in _sl:
        _sem_color, _sem_label = _RED, "ACCIÓN REQUERIDA"
    elif "atenc" in _sl or "alarma" in _sl or "alert" in _sl:
        _sem_color, _sem_label = _AMBER, "ATENCIÓN"
    elif "normal" in _sl or "aceptable" in _sl:
        _sem_color, _sem_label = _GREEN, "CONDICIÓN ACEPTABLE"
    else:
        _sem_color, _sem_label = "#94a3b8", _status_txt.upper() or "SIN DATOS"

    body.append(Paragraph("RESUMEN EJECUTIVO", styles["WMTOC1"]))
    kpi_tbl = Table([[
        _kpi("Salud", health.get("score", "—"), hcolor),
        _kpi("Estado", f"● {_sem_label}", colors.HexColor(_sem_color)),
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

    # Caption de figura centrado (las imágenes van centradas con su caption)
    st_cap_fig = ParagraphStyle("bfCapFig", parent=st_cap, alignment=TA_CENTER)

    # ---- Estado actual del tren (esquemático del Resumen Ejecutivo) ----
    # Reemplaza al Mapa de Sensores con polares: vista lateral compacta con
    # cojinetes coloreados por severidad + Overall bajo cada plano.
    if sensor_map_png:
        try:
            img = Image(BytesIO(sensor_map_png), width=17.5 * cm, height=7.5 * cm,
                        kind="proportional")
            img.hAlign = "CENTER"
            body.append(KeepTogether([
                img,
                Paragraph(f"Figura — Estado actual del tren · {tag}", st_cap_fig),
            ]))
        except Exception:
            pass

    # ---- Diagnóstico ----
    if diagnosis:
        body.append(Paragraph("DIAGNÓSTICO", styles["WMTOC1"]))
        body.extend(render_markdown_flowables(diagnosis, styles))

    # ---- Recomendaciones ----
    # Acepta dos formatos:
    #   • str  — borrador automático (legacy)
    #   • dict — recomendación gestionada por el especialista:
    #            {"text": ..., "started_at": "YYYY-MM-DD"}; la fecha de
    #            inicio se muestra al final en gris opaco.
    if recommendations:
        # Recomendaciones JUNTAS en una sola página (título incluido — sin
        # títulos huérfanos). Si exceden una página completa, KeepTogether
        # degrada y permite el corte.
        _rec_block: List[Any] = [Paragraph("RECOMENDACIONES", styles["WMTOC1"])]
        for i, rec in enumerate(recommendations, start=1):
            if isinstance(rec, dict):
                txt = paragraph_safe(rec.get("text", ""))
                fecha = _fecha_es(rec.get("started_at"))
                line = (f"{i}. {txt} "
                        f"<font color='#9ca3af' size='8.5'>({fecha})</font>")
            else:
                line = f"{i}. {paragraph_safe(rec)}"
            _rec_block.append(Paragraph(line, st_body))
        body.append(KeepTogether(_rec_block))

    # ---- Tabular List (espejo de la vista de la app) ----
    # Arranca en PÁGINA NUEVA: las recomendaciones quedan solas en la suya
    # y el título nunca queda huérfano al pie de página.
    if channels:
        body.append(PageBreak())
        body.append(Paragraph("TABULAR LIST — CANALES (API 670 / ISO 20816-3)", styles["WMTOC1"]))
        _asof = (meta or {}).get("tabular_asof", "")
        if _asof:
            body.append(Paragraph(
                f"Datos de Live Monitoring · última lectura: {_asof}",
                ParagraphStyle("bfTabAsof", fontName=REGULAR, fontSize=8,
                               textColor=colors.HexColor("#6b7280"),
                               leading=10, spaceAfter=4)))
        st_cell = ParagraphStyle("c", fontName=REGULAR, fontSize=7,
                                 textColor=colors.HexColor("#111827"), leading=9)
        st_cn = ParagraphStyle("cn", fontName="Courier", fontSize=7,
                               textColor=colors.HexColor("#111827"), leading=9)
        st_hd = ParagraphStyle("hd", fontName=BOLD, fontSize=6.1,
                               textColor=colors.HexColor("#1d4ed8"), leading=8)

        def _num(v, digits=2):
            try:
                f = float(v)
                return f"{f:.{digits}f}" if f > 0 else "—"
            except Exception:
                return "—"

        def _rpm_txt(v):
            try:
                f = float(v)
                return f"{f:.0f}" if f > 0 else "—"
            except Exception:
                return "—"

        head = ["MACHINE", "POINT", "RPM", "FAMILY", "ALARM", "DANGER",
                "CRITERION BASED", "STATUS", "OVERALL", "UNIT",
                "0.5X", "1X", "2X"]
        data = [[Paragraph(h, st_hd) for h in head]]
        row_styles = []
        _STATUS_COL = 7
        for i, c in enumerate(channels, start=1):
            fg, bg = _sev_colors(c.get("status", ""))
            data.append([
                Paragraph(c.get("machine", tag), st_cell),
                # plane_label "—" es truthy → caer explícitamente al sensor_label
                Paragraph((c.get("plane_label")
                           if c.get("plane_label") not in (None, "", "—")
                           else c.get("sensor_label", "—")), st_cell),
                Paragraph(_rpm_txt(c.get("rpm")), st_cn),
                Paragraph({"Acceleration": "Accel."}.get(
                    c.get("family", "—"), c.get("family", "—")), st_cell),
                Paragraph(_num(c.get("alarm")), st_cn),
                Paragraph(_num(c.get("danger")), st_cn),
                Paragraph(c.get("criterion", "ISO 20816-3"), st_cell),
                Paragraph(c.get("status", "—"), st_cell),
                Paragraph(str(c.get("value", "—")), st_cn),
                Paragraph(c.get("unit", ""), st_cell),
                Paragraph(str(c.get("x05_amp", "—")), st_cn),
                Paragraph(str(c.get("x1_amp", "—")), st_cn),
                Paragraph(str(c.get("x2_amp", "—")), st_cn),
            ])
            row_styles.append(("TEXTCOLOR", (_STATUS_COL, i), (_STATUS_COL, i), colors.HexColor(fg)))
            row_styles.append(("BACKGROUND", (_STATUS_COL, i), (_STATUS_COL, i), colors.HexColor(bg)))
        ctbl = Table(data, colWidths=[1.45 * cm, 2.1 * cm, 1.0 * cm, 1.45 * cm,
                                      1.05 * cm, 1.2 * cm, 2.35 * cm, 1.25 * cm,
                                      1.35 * cm, 1.05 * cm, 0.85 * cm, 0.85 * cm,
                                      0.85 * cm], repeatRows=1)
        ctbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(_LINE)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ] + row_styles))
        body.append(ctbl)
        st_note_center = ParagraphStyle("bfNoteC", parent=st_cap,
                                        alignment=TA_CENTER)
        _note = "Overall según norma del punto."
        # La aclaración del gas generator SOLO aplica a trenes con planos CRF
        # (LM6000/TM2500) — en SGT300 y similares confundía al lector.
        if any("CRF" in str(c.get("plane_label", "")).upper()
               or "CRF" in str(c.get("sensor_label", "")).upper()
               for c in channels):
            _note = ("Overall según norma del punto · Los órdenes 0.5X/1X/2X se "
                     "referencian al keyphasor; en puntos del gas generator "
                     "(CRF, ~10200 cpm) no aplican y se reporta solo el Overall.")
        body.append(Paragraph(_note, st_note_center))

    # ---- Histórico Overall (últimos 10 días, pico diario, con semáforo) ----
    if overall_history and overall_history.get("rows"):
        _dates = overall_history["dates"]
        _hd7 = ParagraphStyle("hd7", fontName=BOLD, fontSize=6.1,
                              textColor=colors.HexColor("#1d4ed8"), leading=8)
        _c7 = ParagraphStyle("c7", fontName="Courier", fontSize=6.6,
                             textColor=colors.HexColor("#111827"), leading=8.5)
        _cl7 = ParagraphStyle("cl7", fontName=REGULAR, fontSize=6.6,
                              textColor=colors.HexColor("#111827"), leading=8.5)
        _head = [Paragraph("PUNTO", _hd7)] + [Paragraph(d, _hd7) for d in _dates]
        _data = [_head]
        _cell_styles = []
        for ri, row in enumerate(overall_history["rows"], start=1):
            cells = [Paragraph(f"{row['label']} ({row.get('unit','')})", _cl7)]
            for ci, d in enumerate(_dates, start=1):
                v = (row.get("values") or {}).get(d)
                cells.append(Paragraph(f"{v:.2f}" if v is not None else "—", _c7))
                if v is not None:
                    if row.get("danger", 0) > 0 and v >= row["danger"]:
                        bg = _RED_BG
                    elif row.get("alarm", 0) > 0 and v >= row["alarm"]:
                        bg = _AMBER_BG
                    else:
                        bg = _GREEN_BG
                    _cell_styles.append(
                        ("BACKGROUND", (ci, ri), (ci, ri), colors.HexColor(bg)))
            _data.append(cells)
        _wdate = min(1.42, 14.0 / max(len(_dates), 1))
        _htbl = Table(_data, colWidths=[2.8 * cm] + [_wdate * cm] * len(_dates),
                      repeatRows=1)
        _htbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(_LINE)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ] + _cell_styles))
        body.append(KeepTogether([
            Paragraph("Tendencia del Overall — pico diario · últimos 10 días",
                      ParagraphStyle("WMTOC2", parent=styles["WMTOC2"],
                                     alignment=TA_CENTER)),
            _htbl,
            Paragraph("Semáforo por celda: verde = condición aceptable · "
                      "ámbar = sobre Alarma · rojo = sobre Danger.",
                      ParagraphStyle("bfNoteC2", parent=st_cap,
                                     alignment=TA_CENTER)),
        ]))

    # ---- Métricas de forma de onda (últimos snapshots) ----
    if wf_history:
        _hd8 = ParagraphStyle("hd8", fontName=BOLD, fontSize=6.6,
                              textColor=colors.HexColor("#1d4ed8"), leading=8.5)
        _c8 = ParagraphStyle("c8", fontName="Courier", fontSize=7,
                             textColor=colors.HexColor("#111827"), leading=9)
        _cl8 = ParagraphStyle("cl8", fontName=REGULAR, fontSize=7,
                              textColor=colors.HexColor("#111827"), leading=9)
        _wf_head = ["FECHA", "CANAL", "PK", "PK-PK", "RMS", "UNIDAD",
                    "FACTOR CRESTA"]
        _wdata = [[Paragraph(h, _hd8) for h in _wf_head]]
        for r in wf_history[:80]:
            _wdata.append([
                Paragraph(r.get("fecha", "—"), _c8),
                Paragraph(str(r.get("canal", "—")), _cl8),
                Paragraph(f"{r.get('pk', 0):.3f}", _c8),
                Paragraph(f"{r.get('pkpk', 0):.3f}", _c8),
                Paragraph(f"{r.get('rms', 0):.3f}", _c8),
                Paragraph(str(r.get("unit", "")), _cl8),
                Paragraph(f"{r.get('crest', 0):.2f}", _c8),
            ])
        _wtbl = Table(_wdata, colWidths=[3.0 * cm, 3.2 * cm, 1.9 * cm, 1.9 * cm,
                                         1.9 * cm, 2.0 * cm, 2.7 * cm],
                      repeatRows=1)
        _wtbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(_LINE)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        body.append(Spacer(1, 8))
        body.append(KeepTogether([
            Paragraph("Métricas de forma de onda — últimos registros",
                      ParagraphStyle("WMTOC2", parent=styles["WMTOC2"],
                                     alignment=TA_CENTER)),
            _wtbl,
        ]))

    # ---- Figuras ----
    # Ciclo 23.170 — Tendencias SEPARADAS por sección (CRF-TRF, Generador, ...)
    # con líneas de alarma/danger, y caption numerado en español (figura,
    # descripción, fecha, equipo). Fallback al trend único si no hay 'trends'.
    _trends = (figures or {}).get("trends") or []
    if not _trends and (figures or {}).get("trend"):
        _trends = [{"section": "", "unit": "", "descr": "", "png": figures["trend"]}]
    _others = [k for k in ("spectrum", "waveform", "orbit") if (figures or {}).get(k)]
    if _trends or _others:
        # El título de sección viaja DENTRO del bloque de la primera figura
        # (KeepTogether) para que nunca quede huérfano al pie de una página.
        _section_head: List[Any] = [Paragraph("FIGURAS Y ANÁLISIS",
                                              styles["WMTOC1"])]
        _fecha = _fecha_es(meta.get("report_date"))
        _equipo = f"Unidad {tag}"
        _n = 0

        st_analysis = ParagraphStyle("bfAnalysis", parent=styles["WMBody"],
                                     alignment=TA_JUSTIFY, spaceBefore=2,
                                     spaceAfter=10)
        # Título de figura CENTRADO. Se clona con el MISMO name "WMTOC2" para
        # que afterFlowable() del shell lo siga registrando en la TOC (la
        # detección es por style.name).
        st_fig_head = ParagraphStyle("WMTOC2", parent=styles["WMTOC2"],
                                     alignment=TA_CENTER)

        def _add_fig(png, head, big_h, analysis: str = ""):
            nonlocal _n, _section_head
            try:
                img = Image(BytesIO(png), width=17.0 * cm, height=big_h,
                            kind="proportional")
                img.hAlign = "CENTER"
                _n += 1
                _cap = f"Figura {_n}. {head}"
                if _fecha:
                    _cap += f", {_fecha}"
                _cap += f" · {_equipo}"
                # Título + figura + caption + ANÁLISIS juntos en la misma
                # página (para leer la figura y su interpretación sin saltar
                # de hoja). Si el bloque no cabe, KeepTogether lo pasa entero
                # a la página siguiente; solo si excede una página completa
                # se permite el corte. El encabezado de sección va dentro del
                # PRIMER bloque para no dejarlo huérfano.
                _block = list(_section_head) + [
                    Paragraph(head, st_fig_head),
                    img,
                    Paragraph(_cap, st_cap_fig),
                ]
                _section_head = []
                if analysis:
                    # El análisis viene en PÁRRAFOS (separados por línea en
                    # blanco): turbina / generador / veredicto. Se renderiza
                    # cada uno como Paragraph propio para dar aire al texto.
                    for _para in str(analysis).split("\n\n"):
                        _para = _para.strip()
                        if _para:
                            _block.append(Paragraph(paragraph_safe(_para),
                                                    st_analysis))
                body.append(KeepTogether(_block))
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
            _add_fig(t.get("png"), head, 6.2 * cm,
                     analysis=(t.get("analysis") or ""))

        for key in _others:
            cap = _FIG_CAPTIONS.get(key, key.title())
            _h = 10.6 * cm if key in ("spectrum", "waveform") else 8.0 * cm
            # v3.31.412 — apilados troceados: si hay varias partes (>6
            # canales), cada una es su propia figura "(parte i/n)"; el
            # análisis va con la última parte.
            _parts = figures.get(f"{key}_pngs") or [figures[key]]
            for pi, part in enumerate(_parts, start=1):
                # v3.31.415 — cada parte es un paquete por máquina con nombre
                # ({"png","name"}); compat con bytes planos.
                if isinstance(part, dict):
                    _png_bytes = part.get("png")
                    _sfx = (part.get("name") or "").strip()
                else:
                    _png_bytes, _sfx = part, ""
                if not _png_bytes:
                    continue
                if _sfx:
                    _head = f"{cap} — {_sfx}"
                elif len(_parts) > 1:
                    _head = f"{cap} ({pi}/{len(_parts)})"
                else:
                    _head = cap
                _an = (figures.get(f"{key}_analysis") or "") \
                    if pi == len(_parts) else ""
                _add_fig(_png_bytes, _head, _h, analysis=_an)

    return render_report_pdf(meta, body)


__all__ = ["generate_briefing_pdf"]
