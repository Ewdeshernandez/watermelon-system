"""Cotización en PDF (1 página) para la web de ventas (watermelonsystem.com).

La genera la API al instante cuando un prospecto llena "Cotice en 1 minuto".
Mismos precios y descuentos que la plataforma (core.billing.pricing).
"""
from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from typing import Any, Dict

from core.billing import pricing

NAVY = "#12305e"
GREEN = "#2f8f4e"


def quote_number(today: date, seq: str) -> str:
    return f"COT-{today:%Y%m%d}-{seq[-4:].upper()}"


def build_quote_pdf(lead: Dict[str, Any], number: str, today: date | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    today = today or date.today()
    plan = pricing.PLANS[lead["plan"]]
    ent_key = lead.get("entity") or "siga_co"
    ent = pricing.ENTITIES[ent_key]
    n = int(lead["machines"])
    trm = lead.get("trm")
    m = pricing.quote(plan.key, n, ent_key, "mensual", trm=trm)
    y = pricing.quote(plan.key, n, ent_key, "anual", trm=trm)
    cur = m["currency"]
    f = lambda v: pricing.fmt_money(v, cur)  # noqa: E731

    st_t = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=20, leading=24,
                          textColor=colors.white)
    st_k = ParagraphStyle("k", fontName="Helvetica", fontSize=8.5, leading=11,
                          textColor=colors.HexColor("#9fd4a9"))
    st_b = ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=14,
                          textColor=colors.HexColor("#0b1f3a"))
    st_m = ParagraphStyle("m", fontName="Helvetica", fontSize=8.5, leading=12,
                          textColor=colors.HexColor("#64748b"))
    st_h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=12, leading=16,
                          textColor=colors.HexColor(NAVY), spaceBefore=6, spaceAfter=4)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title=f"Cotización {number}", author="Watermelon System")
    W = A4[0] - 3.6 * cm
    head = Table([[[Paragraph("WATERMELON SYSTEM · COTIZACIÓN", st_k), Spacer(1, 4),
                    Paragraph(f"Plan {plan.name} · {n} máquina{'s' if n != 1 else ''}", st_t)],
                   Paragraph(f"<font color='#c9d6ea'>{number}<br/>{today:%d/%m/%Y}<br/>"
                             f"Válida hasta {(today + timedelta(days=30)):%d/%m/%Y}</font>",
                             ParagraphStyle("r", parent=st_m, alignment=2))]],
                 colWidths=[W * 0.68, W * 0.32])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(NAVY)),
                              ("LINEAFTER", (-1, 0), (-1, -1), 4, colors.HexColor(GREEN)),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("TOPPADDING", (0, 0), (-1, -1), 14),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                              ("LEFTPADDING", (0, 0), (-1, -1), 14)]))
    story = [head, Spacer(1, 12)]
    who = " · ".join(x for x in (lead.get("company"), lead.get("name"), lead.get("email")) if x)
    story.append(Paragraph(f"<b>Para:</b> {who}", st_b))
    if lead.get("machine_types"):
        story.append(Paragraph(f"<b>Máquinas:</b> {lead['machine_types']}", st_b))
    story.append(Spacer(1, 10))

    rows = [["Concepto", "Valor"],
            ["Precio por máquina al mes", f(m["unit_price"])],
            ["Máquinas", str(n)]]
    if m["volume_discount_pct"]:
        rows.append([f"Descuento por volumen ({n} máquinas)", f"−{m['volume_discount_pct']:.0f} %"])
    if m["entity_discount_pct"]:
        rows.append(["Descuento por pago en USD", f"−{m['entity_discount_pct']:.0f} %"])
    rows += [["Total mensual", f(m["amount"])],
             ["Total anual pagando el año completo (12 meses por 11)", f(y["amount"])]]
    t = Table(rows, colWidths=[W * 0.66, W * 0.34])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 10),
        ("FONT", (0, -2), (-1, -2), "Helvetica-Bold", 11),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e6ebf2")),
        ("BACKGROUND", (0, -2), (-1, -2), colors.HexColor("#e6f6ec")),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [t, Spacer(1, 4),
              Paragraph(f"Factura {ent['name']} ({ent['country']}, {ent['currency']})"
                        + (f" · TRM de referencia {trm:,.2f}" if cur == "USD" and trm else "")
                        + (". IVA (19 %) incluido." if cur == "COP" else "."), st_m)]

    story.append(Paragraph(f"Qué incluye el plan {plan.name}", st_h))
    for x in plan.includes:
        story.append(Paragraph(f"<font color='{GREEN}'>●</font>&nbsp; {x}", st_b))
    for x in plan.excludes:
        story.append(Paragraph(f"<font color='#94a3b8'>●&nbsp; {x}</font>", st_b))

    story.append(Paragraph("Condiciones", st_h))
    conds = []
    if lead.get("pilot"):
        conds.append("Piloto de 30 días en 1 máquina sin costo, para que vea el servicio funcionando "
                     "antes de firmar.")
    conds += [
        "Si su máquina no tiene sistema de monitoreo, instalamos nuestros sensores de vibración y el "
        "nodo de adquisición sin costo adicional: son de SIGA, se entregan en alquiler incluido en el "
        "plan, deben devolverse en buen estado y se retiran al terminar el contrato.",
        "Incluye instalación, soporte, almacenamiento de datos, comunicación remota, pruebas "
        "metrológicas anuales y una capacitación anual (no certificada) para hasta 10 personas.",
        "Contrato mínimo de 12 meses. Pago mensual a 30 días tras la radicación de la factura.",
        "Garantía de 1 año (Ley 1480 de 2011) y derecho de retracto de 5 días hábiles.",
        "Cotización válida por 30 días calendario. Aplican los Términos y condiciones publicados en "
        "watermelonsystem.com/terminos.",
    ]
    for x in conds:
        story.append(Paragraph(f"<font color='{GREEN}'>●</font>&nbsp; {x}", st_b))

    story.append(Paragraph("Siguientes pasos", st_h))
    for i, x in enumerate((
            "Un especialista le contacta para revisar sus máquinas y sus sensores de vibración.",
            "Definimos la conexión: nodo de adquisición en sitio, con internet satelital si hace falta, "
            "sin intervenir su sistema de protección (API 670).",
            "Activamos el monitoreo en línea y su primer reporte."), start=1):
        story.append(Paragraph(f"{i}. {x}", st_b))
    story += [Spacer(1, 14), Paragraph(
        "Watermelon System · monitoreo de condición en línea de turbomaquinaria · "
        "watermelonsystem.com · Facturación en Colombia por SIGA GROUP SAS y en EE. UU. por "
        "Watermelon System.", st_m)]
    doc.build(story)
    return buf.getvalue()
