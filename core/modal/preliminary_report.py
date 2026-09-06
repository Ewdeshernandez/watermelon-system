"""
core/modal/preliminary_report.py — Reporte PRELIMINAR AUTOMÁTICO de campo (modal)
================================================================================

PDF liviano y AUTÓNOMO (solo reportlab, sin kaleido/plotly) que se genera en sitio
el mismo día. AUTOMÁTICO: incluye por defecto todos los gráficos y tablas de la
sesión (config 3D, EMA, OMA/densidad espectral, modos, comparativo, Campbell, SSI,
formas modales) + Go/No-Go de calidad + un ANÁLISIS y RECOMENDACIONES automáticos.
No reemplaza el reporte completo (ese sale de la web); lleva el ID de la corrida.

`sections` = lista de dict flexibles:
    {"title": str, "intro": str?, "figures": [(caption, png_bytes)],
     "table": {"headers":[...], "rows":[[...]]} }
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

NAVY = "#0F1E3D"; GREEN = "#16a34a"; AMBER = "#f59e0b"; RED = "#dc2626"; GRAY = "#e2e8f0"


def _status_color(s: str) -> str:
    s = (s or "").lower()
    if s.startswith(("pass", "ok", "go")) and not s.startswith("no-go"):
        return GREEN
    if s.startswith(("fail", "no-go")):
        return RED
    return AMBER


_T = {
    "en": {"sub": "Automatic field preliminary — subject to specialist validation",
           "ident": "Identification", "quality": "Data quality — Go / No-Go (ISO 7626-5)",
           "check": "Check", "status": "Status", "detail": "Detail", "overall": "Overall",
           "analysis": "Automatic analysis", "findings": "Findings",
           "recs": "Recommendations", "evidence": "Field evidence",
           "asset": "Asset", "client": "Client", "type": "Type", "location": "Location",
           "test": "Test", "rpm": "Running speed", "tech": "Specialist", "rev": "Approved by",
           "date": "Date", "equip": "Equipment", "prep": "Prepared by", "rev2": "Reviewed by",
           "disc": ("<b>PRELIMINARY REPORT</b> — valid only for the conditions present during the service. "
                    "Subject to specialist validation and to the full analysis report generated from Watermelon System (web). "
                    "Cloud run ID: <b>{rid}</b>. Standards: ISO 7626 · ISO 20816 · API 684 · API 670.")},
    "es": {"sub": "Preliminar de campo automático — sujeto a validación de especialista",
           "ident": "Identificación", "quality": "Calidad de datos — Go / No-Go (ISO 7626-5)",
           "check": "Verificación", "status": "Estado", "detail": "Detalle", "overall": "Veredicto",
           "analysis": "Análisis automático", "findings": "Hallazgos",
           "recs": "Recomendaciones", "evidence": "Evidencia de campo",
           "asset": "Activo", "client": "Cliente", "type": "Tipo", "location": "Ubicación",
           "test": "Ensayo", "rpm": "Velocidad", "tech": "Especialista", "rev": "Aprobado por",
           "date": "Fecha", "equip": "Equipo", "prep": "Preparado por", "rev2": "Revisado por",
           "disc": ("<b>REPORTE PRELIMINAR</b> — válido únicamente para las condiciones presentes durante el servicio. "
                    "Sujeto a validación de especialista y al reporte de análisis completo generado desde Watermelon System (web). "
                    "ID de corrida en la nube: <b>{rid}</b>. Normas: ISO 7626 · ISO 20816 · API 684 · API 670.")},
}


def build_preliminary_pdf(
    *,
    meta: Dict[str, Any],
    quality: List[Tuple[str, str, str]],
    sections: Optional[List[Dict[str, Any]]] = None,
    analysis: Optional[List[str]] = None,
    findings: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
    photos: Optional[List[bytes]] = None,
    run_id: str = "",
    logo_png: Optional[bytes] = None,
    lang: str = "es",
) -> bytes:
    T = _T.get(lang, _T["es"])
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    Image, HRFlowable, KeepTogether)
    from PIL import Image as PILImage

    ss = getSampleStyleSheet()
    H = ParagraphStyle("H", parent=ss["Heading2"], textColor=colors.HexColor(NAVY), fontSize=12,
                       spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.2, leading=13)
    small = ParagraphStyle("S", parent=ss["BodyText"], fontSize=8, textColor=colors.HexColor("#64748b"))
    from reportlab.lib.enums import TA_CENTER
    cap = ParagraphStyle("C", parent=ss["BodyText"], fontSize=8, textColor=colors.HexColor("#475569"),
                         spaceBefore=2, spaceAfter=10, alignment=TA_CENTER)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title=meta.get("title", "Preliminary Modal Report"))
    story: List[Any] = []

    def _img(raw, max_w_cm, max_h_cm):
        try:
            im = PILImage.open(BytesIO(raw)); w, h = im.size
            r = min(max_w_cm * cm / w, max_h_cm * cm / h)
            img = Image(BytesIO(raw), width=w * r, height=h * r)
            img.hAlign = "CENTER"          # figuras centradas
            return img
        except Exception:  # noqa: BLE001
            return None

    def _table(headers, rows):
        hh = [Paragraph(f"<b><font color='white'>{h}</font></b>", small) for h in headers]
        data = [hh] + [[Paragraph(str(x), body) for x in row] for row in rows]
        cw = (17.0 / max(1, len(headers))) * cm
        t = Table(data, colWidths=[cw] * len(headers), repeatRows=1)
        t.hAlign = "CENTER"
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                               ("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.HexColor(GREEN)),
                               ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fc")]),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        return t

    # ---- Encabezado ----
    logo_flow = _img(logo_png, 3.2, 1.6) if logo_png else Paragraph("<b>Watermelon</b>", H)
    title_cell = Paragraph(
        f"<font color='{NAVY}' size=15><b>{meta.get('title', 'Preliminary Modal Report')}</b></font><br/>"
        f"<font color='#475569' size=9>{meta.get('subtitle', T['sub'])}</font>",
        body)
    head = Table([[logo_flow, title_cell]], colWidths=[3.6 * cm, 12.4 * cm])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(head)
    story.append(HRFlowable(width="100%", thickness=2.4, color=colors.HexColor(GREEN), spaceBefore=4, spaceAfter=8))

    # ---- 1. Identificación ----
    story.append(Paragraph(f"1. {T['ident']}", H))
    idrows = [[T["asset"], meta.get("asset", ""), T["client"], meta.get("client", "")],
              [T["type"], meta.get("machine_type", ""), T["location"], meta.get("location", "")],
              [T["test"], meta.get("test_type", ""), T["rpm"], f"{meta.get('rpm', '')} RPM"],
              [T["tech"], meta.get("technician", ""), T["rev"], meta.get("reviewer", "")],
              [T["date"], meta.get("date", ""), T["equip"], meta.get("equipment", "Watermelon DAQ")]]
    t = Table([[Paragraph(f"<b>{a}</b>", small), Paragraph(str(b), body),
                Paragraph(f"<b>{c}</b>", small), Paragraph(str(d), body)] for a, b, c, d in idrows],
              colWidths=[2.6 * cm, 5.4 * cm, 2.6 * cm, 5.4 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRAY)),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                           ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f1f5f9")),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(t); story.append(Spacer(1, 8))

    # ---- 2. Go/No-Go ----
    story.append(Paragraph(f"2. {T['quality']}", H))
    qrows = [[c, s, d] for c, s, d in quality]
    qt_head = [T["check"], T["status"], T["detail"]]
    qb = [[Paragraph(f"<b><font color='white'>{h}</font></b>", small) for h in qt_head]]
    for c, s, d in quality:
        qb.append([Paragraph(c, body), Paragraph(f"<b><font color='{_status_color(s)}'>{s}</font></b>", body),
                   Paragraph(d, body)])
    qt = Table(qb, colWidths=[5.2 * cm, 2.6 * cm, 8.2 * cm], repeatRows=1)
    qt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRAY)),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(qt)
    verdict = meta.get("verdict", "")
    if verdict:
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>{T['overall']}: <font color='{_status_color(verdict)}'>{verdict}</font></b>", body))
    story.append(Spacer(1, 8))

    # ---- 3. Automatic analysis ----
    if analysis:
        story.append(Paragraph(f"3. {T['analysis']}", H))
        for a in analysis:
            story.append(Paragraph("• " + a, body))
        story.append(Spacer(1, 6))

    # ---- Secciones automáticas (config/EMA/OMA/modes/comparative/campbell/ssi/shapes) ----
    n = 4
    for sec in (sections or []):
        blk: List[Any] = [Paragraph(f"{n}. {sec.get('title', '')}", H)]
        if sec.get("intro"):
            blk.append(Paragraph(sec["intro"], body))
        tab = sec.get("table")
        if tab and tab.get("rows"):
            blk.append(_table(tab["headers"], tab["rows"])); blk.append(Spacer(1, 4))
        for capt, png in (sec.get("figures") or []):
            im = _img(png, 17, 11)                  # figuras más grandes y centradas
            if im is not None:
                blk.append(Spacer(1, 4)); blk.append(im); blk.append(Paragraph(capt, cap))
        story.append(KeepTogether(blk) if len(blk) <= 3 else blk[0])
        if not (len(blk) <= 3):
            for fl in blk[1:]:
                story.append(fl)
        story.append(Spacer(1, 4)); n += 1

    # ---- Findings + recommendations ----
    if findings:
        story.append(Paragraph(f"{n}. {T['findings']}", H)); n += 1
        for i, f in enumerate(findings, 1):
            story.append(Paragraph(f"{i}. {f}", body))
        story.append(Spacer(1, 4))
    if recommendations:
        story.append(Paragraph(f"{n}. {T['recs']}", H)); n += 1
        for i, r in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {r}", body))
        story.append(Spacer(1, 6))

    # ---- Evidence ----
    if photos:
        story.append(Paragraph(f"{n}. {T['evidence']}", H)); n += 1
        row = []
        for ph in photos[:8]:
            im = _img(ph, 7.6, 5.2)
            if im is not None:
                row.append(im)
            if len(row) == 2:
                story.append(Table([row], colWidths=[8 * cm, 8 * cm])); row = []
        if row:
            story.append(Table([row], colWidths=[8 * cm, 8 * cm]))
        story.append(Spacer(1, 6))

    # ---- Cierre ----
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(GRAY), spaceBefore=6, spaceAfter=4))
    story.append(Paragraph(
        T["disc"].format(rid=run_id or "—"), small))
    doc.build(story)
    return buf.getvalue()
