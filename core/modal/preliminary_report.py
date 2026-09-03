"""
core/modal/preliminary_report.py — Reporte PRELIMINAR de campo (modal)
======================================================================

PDF liviano y AUTÓNOMO (solo reportlab, sin kaleido/plotly) para entregar en sitio
el mismo día: identificación + Go/No-Go de calidad de datos + resultados
preliminares (modos, figuras capturadas de la app) + screening de resonancia
(Campbell/API 684) + correlación EMA↔OMA + hallazgos/recomendaciones del técnico +
fotos + firma. NO reemplaza el reporte completo (ese sale de la web); lleva el ID
de la corrida en la nube para regenerarlo.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

NAVY = "#0F1E3D"
GREEN = "#16a34a"
AMBER = "#f59e0b"
RED = "#dc2626"
GRAY = "#e2e8f0"


def _status_color(s: str) -> str:
    s = (s or "").lower()
    if s.startswith("pass") or s.startswith("ok") or "✓" in s:
        return GREEN
    if s.startswith("fail") or "✗" in s:
        return RED
    return AMBER


def build_preliminary_pdf(
    *,
    meta: Dict[str, Any],
    quality: List[Tuple[str, str, str]],
    modes: Optional[List[List[Any]]] = None,
    crossings: Optional[List[List[Any]]] = None,
    ema_oma: Optional[List[List[Any]]] = None,
    figures: Optional[List[Tuple[str, bytes]]] = None,
    findings: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
    photos: Optional[List[bytes]] = None,
    run_id: str = "",
    logo_png: Optional[bytes] = None,
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    Image, HRFlowable)
    from PIL import Image as PILImage

    styles = getSampleStyleSheet()
    H = ParagraphStyle("H", parent=styles["Heading2"], textColor=colors.HexColor(NAVY),
                       fontSize=12, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("B", parent=styles["BodyText"], fontSize=9.2, leading=13)
    small = ParagraphStyle("S", parent=styles["BodyText"], fontSize=8, textColor=colors.HexColor("#64748b"))
    cap = ParagraphStyle("C", parent=styles["BodyText"], fontSize=8, textColor=colors.HexColor("#475569"),
                         alignment=TA_LEFT, spaceBefore=2, spaceAfter=8)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title=meta.get("title", "Preliminary Modal Report"))
    story: List[Any] = []

    def _img(raw: bytes, max_w_cm: float, max_h_cm: float):
        try:
            im = PILImage.open(BytesIO(raw)); w, h = im.size
            ratio = min(max_w_cm * cm / w, max_h_cm * cm / h)
            return Image(BytesIO(raw), width=w * ratio, height=h * ratio)
        except Exception:  # noqa: BLE001
            return None

    # ---- Encabezado (tarjeta blanca + banda verde + título) ----
    logo_flow = _img(logo_png, 3.2, 1.6) if logo_png else Paragraph("<b>Watermelon</b>", H)
    title_cell = Paragraph(
        f"<font color='{NAVY}' size=15><b>Preliminary Modal Report</b></font><br/>"
        f"<font color='#475569' size=9>{meta.get('subtitle', 'Field preliminary — subject to specialist validation')}</font>",
        body)
    head = Table([[logo_flow, title_cell]], colWidths=[3.6 * cm, 12.4 * cm])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(head)
    story.append(HRFlowable(width="100%", thickness=2.4, color=colors.HexColor(GREEN), spaceBefore=4, spaceAfter=8))

    # ---- 1. Identificación ----
    story.append(Paragraph("1. Identification", H))
    idrows = [
        ["Asset", meta.get("asset", ""), "Client", meta.get("client", "")],
        ["Type", meta.get("machine_type", ""), "Location", meta.get("location", "")],
        ["Test", meta.get("test_type", ""), "Running speed", f"{meta.get('rpm', '')} RPM"],
        ["Technician", meta.get("technician", ""), "Reviewed by", meta.get("reviewer", "")],
        ["Date", meta.get("date", ""), "Equipment", meta.get("equipment", "NI 9234 / cDAQ-9178")],
    ]
    t = Table([[Paragraph(f"<b>{a}</b>", small), Paragraph(str(b), body),
                Paragraph(f"<b>{c}</b>", small), Paragraph(str(d), body)] for a, b, c, d in idrows],
              colWidths=[2.6 * cm, 5.4 * cm, 2.6 * cm, 5.4 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRAY)),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                           ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f1f5f9")),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(t); story.append(Spacer(1, 8))

    # ---- 2. Data quality — Go/No-Go ----
    story.append(Paragraph("2. Data quality — Go / No-Go (ISO 7626-5)", H))
    q_head = [Paragraph(f"<b><font color='white'>{h}</font></b>", small) for h in ("Check", "Status", "Detail")]
    q_body = [q_head]
    for chk, status, detail in quality:
        q_body.append([Paragraph(chk, body),
                       Paragraph(f"<b><font color='{_status_color(status)}'>{status}</font></b>", body),
                       Paragraph(detail, body)])
    qt = Table(q_body, colWidths=[5.2 * cm, 2.6 * cm, 8.2 * cm], repeatRows=1)
    qt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRAY)),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(qt)
    verdict = meta.get("verdict", "")
    if verdict:
        vc = _status_color(verdict)
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Overall: <font color='{vc}'>{verdict}</font></b>", body))
    story.append(Spacer(1, 8))

    # ---- 3. Preliminary results ----
    if modes:
        story.append(Paragraph("3. Preliminary results — identified modes", H))
        mh = [Paragraph(f"<b><font color='white'>{h}</font></b>", small)
              for h in ("Freq (Hz)", "Damping (%)", "Complexity (%)", "Class")]
        mb = [mh] + [[Paragraph(str(x), body) for x in row] for row in modes]
        mt = Table(mb, colWidths=[4 * cm, 4 * cm, 4 * cm, 4 * cm], repeatRows=1)
        mt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRAY)),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(mt); story.append(Spacer(1, 6))

    for caption, png in (figures or []):
        im = _img(png, 16, 8)
        if im is not None:
            story.append(im); story.append(Paragraph(caption, cap))

    # ---- 3b. Resonance screening (Campbell / API 684) ----
    if crossings:
        story.append(Paragraph("Resonance screening — Campbell (API 684)", H))
        ch = [Paragraph(f"<b><font color='white'>{h}</font></b>", small)
              for h in ("Mode", "Order", "RPM", "Margin %", "Status")]
        cb = [ch] + [[Paragraph(str(x), body) for x in row] for row in crossings]
        ct = Table(cb, colWidths=[3.4 * cm, 2.2 * cm, 3 * cm, 3 * cm, 4.4 * cm], repeatRows=1)
        ct.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRAY)),
                                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(ct); story.append(Spacer(1, 8))

    # ---- 4. EMA↔OMA correlation ----
    if ema_oma:
        story.append(Paragraph("4. EMA ↔ OMA correlation", H))
        eh = [Paragraph(f"<b><font color='white'>{h}</font></b>", small) for h in ("EMA (Hz)", "OMA (Hz)", "Δf (Hz)")]
        eb = [eh] + [[Paragraph(str(x), body) for x in row] for row in ema_oma]
        et = Table(eb, colWidths=[5.3 * cm, 5.3 * cm, 5.4 * cm], repeatRows=1)
        et.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor(GRAY)),
                                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(et); story.append(Spacer(1, 8))

    # ---- 5. Findings + recommendations ----
    if findings:
        story.append(Paragraph("5. Preliminary findings", H))
        for i, f in enumerate(findings, 1):
            story.append(Paragraph(f"{i}. {f}", body))
        story.append(Spacer(1, 4))
    if recommendations:
        story.append(Paragraph("6. Preliminary recommendations", H))
        for i, r in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {r}", body))
        story.append(Spacer(1, 6))

    # ---- 7. Evidence (photos) ----
    if photos:
        story.append(Paragraph("7. Field evidence", H))
        row = []
        for ph in photos[:6]:
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
        "<b>PRELIMINARY REPORT</b> — valid only for the conditions present during the service. "
        "Subject to specialist validation and to the full analysis report generated from Watermelon System (web). "
        f"Cloud run ID: <b>{run_id or '—'}</b>. Standards: ISO 7626 · ISO 20816 · API 684 · API 670.", small))
    story.append(Spacer(1, 10))
    sig = Table([[Paragraph("_______________________________<br/>Prepared by: "
                            + meta.get("technician", ""), small),
                  Paragraph("_______________________________<br/>Reviewed by: "
                            + meta.get("reviewer", ""), small)]],
                colWidths=[8 * cm, 8 * cm])
    story.append(sig)

    doc.build(story)
    return buf.getvalue()
