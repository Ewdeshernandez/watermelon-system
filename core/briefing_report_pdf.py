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
    train_drawing: Optional[Any] = None,
) -> bytes:
    """Devuelve los bytes del PDF del briefing del activo, en formato pro."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
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
        # Banda de código de formato (WMS-FMT-…) OCULTA: el usuario la quitó por
        # estética. El encabezado interno queda solo con el título del reporte.
        "hide_format_band": True,
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

    # Numeración automática de secciones (entran a la TABLA DE CONTENIDO con
    # su número, porque el shell registra el texto del heading). Nivel 1 = N.,
    # nivel 2 = N.M (subsecciones del Desarrollo, estilo reporte OMA).
    _h1n = [0]
    _h2n = [0]

    def _h1(title: str) -> Paragraph:
        _h1n[0] += 1
        _h2n[0] = 0
        return Paragraph(f"{_h1n[0]}. {title}", styles["WMTOC1"])

    # Título de subsección al MARGEN IZQUIERDO (norma ICONTEC/NTC 1486 de
    # informes técnicos). Conserva el name "WMTOC2" para que la TOC lo detecte.
    _h2_left = ParagraphStyle("WMTOC2", parent=styles["WMSubTitle"],
                              alignment=TA_LEFT, spaceBefore=10, spaceAfter=6)

    def _h2(title: str) -> Paragraph:
        _h2n[0] += 1
        return Paragraph(f"{_h1n[0]}.{_h2n[0]} {title}", _h2_left)

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

    # ==== 1. Introducción y alcance (estilo reporte OMA) ====
    _n_pts = len([c for c in (channels or []) if c.get("value") is not None])
    _intro = (
        f"Se presenta el seguimiento de condición del tren {tag}"
        + (f" ({paragraph_safe(train)})" if train else "")
        + f" para el periodo {period_label.lower()}, evaluando {_n_pts} punto(s) "
        "de medición de vibración distribuidos a lo largo del tren de máquinas, "
        "conforme a ISO 20816 y API 670, a partir de los datos del monitoreo en "
        "línea. A continuación se presentan los hallazgos, las recomendaciones y "
        "el desarrollo técnico del servicio.")
    body.append(_h1("Introducción y alcance"))
    body.append(Paragraph(_intro, st_body))

    # ==== 2. Hallazgos (imagen del estado del tren + resumen breve) ====
    body.append(_h1("Hallazgos"))

    # Caption de figura centrado (las imágenes van centradas con su caption)
    st_cap_fig = ParagraphStyle("bfCapFig", parent=st_cap, alignment=TA_CENTER)

    # Imagen del tren = LA MISMA de Live Monitoring (train_drawing vectorial:
    # tren + valores + barras de umbral); fallback al PNG del schematic.
    _placed_img = False
    if train_drawing is not None:
        try:
            train_drawing.hAlign = "CENTER"
            body.append(KeepTogether([
                train_drawing,
                Paragraph(f"Figura — Estado del tren en vivo · {tag}", st_cap_fig),
            ]))
            _placed_img = True
        except Exception:
            _placed_img = False
    if (not _placed_img) and sensor_map_png:
        try:
            img = Image(BytesIO(sensor_map_png), width=17.5 * cm, height=7.5 * cm,
                        kind="proportional")
            img.hAlign = "CENTER"
            body.append(KeepTogether([
                img,
                Paragraph(f"Figura — Estado del tren · {tag}", st_cap_fig),
            ]))
        except Exception:
            pass
    body.append(Spacer(1, 8))

    # Texto de hallazgos: breve, gerencial, nivel analista Cat. IV (markdown IA)
    if summary:
        body.extend(render_markdown_flowables(summary, styles))

    # ==== 3. Recomendaciones: van ANTES del desarrollo (el gerente lee acción) ====
    if recommendations:
        _rec_block: List[Any] = [_h1("Recomendaciones")]
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

    # ==== 4. Desarrollo del servicio (detalle técnico, después de la acción) ====
    # Salto CONDICIONAL: solo empieza en página nueva si no queda espacio
    # suficiente (evita la página casi vacía que dejaba el PageBreak fijo).
    from reportlab.platypus import CondPageBreak
    body.append(CondPageBreak(7 * cm))

    # 4.1 Diagnóstico — el título de sección + el de subsección + el primer
    # párrafo arrancan juntos (KeepTogether) para no dejar títulos huérfanos.
    if diagnosis:
        _diag_flow = render_markdown_flowables(diagnosis, styles)
        _lead = [_h1("Desarrollo del servicio"), _h2("Diagnóstico")]
        if _diag_flow:
            _lead.append(_diag_flow[0])
        body.append(KeepTogether(_lead))
        body.extend(_diag_flow[1:])
    else:
        body.append(_h1("Desarrollo del servicio"))

    # 4.2 Tabular List (espejo de la vista de la app) — arranca en PÁGINA NUEVA
    # para que el título nunca quede huérfano al pie de página.
    if channels:
        # No forzar página nueva (dejaba la p. de Diagnóstico casi vacía); solo
        # salta si no cabe el arranque de la tabla.
        body.append(CondPageBreak(12 * cm))
        body.append(_h2("Tabular List — Overall + 1X / 2X (API 670 / ISO 20816-3)"))
        _asof = (meta or {}).get("tabular_asof", "")
        if _asof:
            body.append(Paragraph(
                f"Datos de Live Monitoring · última lectura: {_asof}",
                ParagraphStyle("bfTabAsof", fontName=REGULAR, fontSize=8,
                               textColor=colors.HexColor("#6b7280"),
                               leading=10, spaceAfter=4)))
        st_cell = ParagraphStyle("c", fontName=REGULAR, fontSize=7,
                                 textColor=colors.HexColor("#111827"), leading=9)
        st_cn = ParagraphStyle("cn", fontName=REGULAR, fontSize=7,
                               textColor=colors.HexColor("#111827"), leading=9)
        st_hd = ParagraphStyle("hd", fontName=BOLD, fontSize=6.1,
                               textColor=colors.white, leading=8)
        st_est = ParagraphStyle("est", fontName=BOLD, fontSize=6.8,
                                textColor=colors.HexColor("#0f172a"), leading=9)

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
                Paragraph(f'<font color="{fg}">●</font>&nbsp;{c.get("status", "—")}', st_est),
                Paragraph(str(c.get("value", "—")), st_cn),
                Paragraph(c.get("unit", ""), st_cell),
                Paragraph(str(c.get("x05_amp", "—")), st_cn),
                Paragraph(str(c.get("x1_amp", "—")), st_cn),
                Paragraph(str(c.get("x2_amp", "—")), st_cn),
            ])
        ctbl = Table(data, colWidths=[1.45 * cm, 2.1 * cm, 1.0 * cm, 1.45 * cm,
                                      1.05 * cm, 1.2 * cm, 2.35 * cm, 1.25 * cm,
                                      1.35 * cm, 1.05 * cm, 0.85 * cm, 0.85 * cm,
                                      0.85 * cm], repeatRows=1)
        ctbl.setStyle(TableStyle([
            # Header navy + texto blanco (look v2, consistente con el reporte 1-pág)
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TOPPADDING", (0, 0), (-1, 0), 5), ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            # Cuerpo: filas zebra + hairline
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f7f9fc")]),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(_LINE)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 1), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 1), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
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
                              textColor=colors.white, leading=8)
        _c7 = ParagraphStyle("c7", fontName=REGULAR, fontSize=6.6,
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
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TOPPADDING", (0, 0), (-1, 0), 4), ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(_LINE)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 1), (-1, -1), 3), ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
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
                              textColor=colors.white, leading=8.5)
        _c8 = ParagraphStyle("c8", fontName=REGULAR, fontSize=7,
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
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TOPPADDING", (0, 0), (-1, 0), 5), ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f7f9fc")]),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(_LINE)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 1), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 1), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        body.append(Spacer(1, 8))
        body.append(KeepTogether([
            Paragraph("Métricas de forma de onda — últimos registros",
                      ParagraphStyle("WMTOC2", parent=styles["WMTOC2"],
                                     alignment=TA_CENTER)),
            _wtbl,
        ]))

    # ---- Figuras ORGANIZADAS POR MÁQUINA (Turbina → GearBox → Generador) ----
    # Cada máquina es su propia subsección con TODAS sus gráficas (tendencias,
    # espectro, forma de onda, órbita) + su análisis, como presenta un analista.
    _trends = (figures or {}).get("trends") or []
    if not _trends and (figures or {}).get("trend"):
        _trends = [{"section": "", "unit": "", "descr": "", "png": figures["trend"]}]

    def _parts(key):
        raw = (figures or {}).get(f"{key}_pngs")
        if not raw:
            return [((figures or {}).get(key), "")] if (figures or {}).get(key) else []
        out = []
        for x in raw:
            if isinstance(x, dict):
                out.append((x.get("png"), (x.get("name") or "").strip()))
            elif x:
                out.append((x, ""))
        return out

    _spec = _parts("spectrum"); _wave = _parts("waveform"); _orb = _parts("orbit")
    if _trends or _spec or _wave or _orb:
        _fecha = _fecha_es(meta.get("report_date"))
        _equipo = f"Unidad {tag}"
        _n = 0
        st_analysis = ParagraphStyle("bfAnalysis", parent=styles["WMBody"],
                                     alignment=TA_JUSTIFY, leading=15.5,
                                     spaceBefore=4, spaceAfter=13)
        st_fig_head = ParagraphStyle("WMTOC2", parent=styles["WMTOC2"], alignment=TA_CENTER)

        def _add_fig(png, head, big_h, analysis: str = "", lead=None):
            nonlocal _n
            if not png:
                return
            try:
                img = Image(BytesIO(png), width=17.0 * cm, height=big_h, kind="proportional")
                img.hAlign = "CENTER"
                _n += 1
                _cap = f"Figura {_n}. {head}"
                if _fecha:
                    _cap += f", {_fecha}"
                _cap += f" · {_equipo}"
                _block = list(lead or []) + [
                    Paragraph(head, st_fig_head), img, Paragraph(_cap, st_cap_fig),
                ]
                if analysis:
                    for _para in str(analysis).split("\n\n"):
                        _para = _para.strip()
                        if _para:
                            _block.append(Paragraph(paragraph_safe(_para), st_analysis))
                body.append(KeepTogether(_block))
            except Exception:
                pass

        def _mtok(s):
            s = (s or "").strip().lower()
            return s.split()[0] if s else ""

        _MACHINES = ["Turbina", "Gearbox", "Generador"]
        # ¿en qué máquina (la última que la tenga) va el análisis combinado de
        # cada tipo? — para que aparezca UNA sola vez, junto a su figura.
        def _last_machine_with(parts):
            last = None
            for m in _MACHINES:
                if any(m.lower() in (n or "").lower() for _p, n in parts):
                    last = m
            return last
        _sp_last = _last_machine_with(_spec)
        _wv_last = _last_machine_with(_wave)
        _ob_last = _last_machine_with(_orb)

        _first = True
        for machine in _MACHINES:
            ml = machine.lower()
            m_tr = [t for t in _trends if _mtok(t.get("section")).startswith(ml[:4])]
            m_sp = [(p, n) for (p, n) in _spec if ml in (n or "").lower()]
            m_wv = [(p, n) for (p, n) in _wave if ml in (n or "").lower()]
            m_ob = [(p, n) for (p, n) in _orb if ml in (n or "").lower()]
            if not (m_tr or m_sp or m_wv or m_ob):
                continue
            # encabezado: "Figuras y análisis" solo la 1ª vez, luego el nombre de
            # la máquina; viajan con la 1ª figura (KeepTogether) para no quedar
            # huérfanos al pie de página.
            _lead = ([_h2("Figuras y análisis")] if _first else []) + [_h2(machine)]
            _first = False
            for t in m_tr:
                head = "Tendencia de vibración"
                sec = (t.get("section") or "").strip(); descr = (t.get("descr") or "").strip()
                if sec:
                    head += f" — {sec}"
                if descr:
                    head += f" ({descr})"
                _add_fig(t.get("png"), head, 6.2 * cm,
                         analysis=(t.get("analysis") or ""), lead=_lead)
                _lead = []
            for (p, _nm) in m_sp:
                _an = (figures.get("spectrum_analysis") or "") if machine == _sp_last else ""
                _add_fig(p, f"Espectro — canales apilados — {machine}", 10.6 * cm,
                         analysis=_an, lead=_lead); _lead = []
            for (p, _nm) in m_wv:
                _an = (figures.get("waveform_analysis") or "") if machine == _wv_last else ""
                _add_fig(p, f"Forma de onda — canales apilados — {machine}", 10.6 * cm,
                         analysis=_an, lead=_lead); _lead = []
            for (p, _nm) in m_ob:
                _an = (figures.get("orbit_analysis") or "") if machine == _ob_last else ""
                _add_fig(p, f"Órbitas — {machine}", 9.0 * cm,
                         analysis=_an, lead=_lead); _lead = []

    return render_report_pdf(meta, body)


__all__ = ["generate_briefing_pdf"]
