"""
core.briefing_monthly_pdf
=========================

Generador del PDF de 1 página para el Briefing Mensual Ejecutivo
(Ciclo 17.31).

Layout:
  ┌──────────────────────────────────────────────────────────────┐
  │  WATERMELON · BRIEFING EJECUTIVO MENSUAL                    │
  │  ECOPETROL · MAGNEX  ·  Mayo 2026                            │
  ├──────────────────────────────────────────────────────────────┤
  │  [N] reportes técnicos · [N] activos cubiertos               │
  │  [chips de severidad: 1 CRÍTICA · 2 ACCIÓN REQUERIDA · ...]  │
  ├──────────────────────────────────────────────────────────────┤
  │  RESUMEN EJECUTIVO DEL MES                                   │
  │  [Párrafo apertura AI con conclusión global del mes]         │
  ├──────────────────────────────────────────────────────────────┤
  │  TOP 3 PRIORIDADES OPERATIVAS                                │
  │  [Bullets numerados con activo + severidad + acción]         │
  ├──────────────────────────────────────────────────────────────┤
  │  ESTADO DEL PORTAFOLIO                                       │
  │  [Lista compacta de TODOS los activos con severidad]         │
  ├──────────────────────────────────────────────────────────────┤
  │  Recomendación global del mes: [1 frase]                     │
  │  El presente briefing... ISO 18436-2 ... operador del activo │
  │  Generado por Watermelon System · v3.X · SIGASAS             │
  └──────────────────────────────────────────────────────────────┘

API pública:
  - generate_monthly_briefing_pdf(briefing_result) → bytes
"""
from __future__ import annotations

import re as _re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List

from core.ai_briefing import SEVERITY_COLORS


# Mapeo de mes ISO → texto español
_SPANISH_MONTHS = [
    "Enero", "Febrero", "Marzo", "Abril",
    "Mayo", "Junio", "Julio", "Agosto",
    "Septiembre", "Octubre", "Noviembre", "Diciembre",
]


def _format_month_es(month_iso: str) -> str:
    """De '2026-04' devuelve 'Abril 2026'."""
    try:
        y, m = month_iso.split("-")
        return f"{_SPANISH_MONTHS[int(m) - 1]} {int(y)}"
    except Exception:
        return month_iso


def _md_inline_to_rl(text: str) -> str:
    """Convierte markdown inline (**bold**, *italic*) a tags ReportLab."""
    escaped = (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    escaped = _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = _re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def generate_monthly_briefing_pdf(
    briefing_result: Dict[str, Any],
) -> bytes:
    """Genera el PDF de 1 página del briefing ejecutivo mensual.

    Args:
        briefing_result: dict devuelto por generate_monthly_briefing(),
                         con keys ok, markdown, asset_aggregates,
                         n_reports, n_assets, month_iso, client_filter.

    Returns:
        bytes del PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

    # Versión del sistema para el footer
    try:
        from core.version import get_version_short
        _ver = get_version_short()
    except Exception:
        _ver = ""

    client_filter = briefing_result.get("client_filter", "") or ""
    month_iso = briefing_result.get("month_iso", "") or ""
    month_label = _format_month_es(month_iso)
    n_reports = briefing_result.get("n_reports", 0)
    n_assets = briefing_result.get("n_assets", 0)
    markdown = briefing_result.get("markdown", "") or ""
    aggregates: List[Dict[str, Any]] = briefing_result.get(
        "asset_aggregates", []
    ) or []

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.6 * cm, rightMargin=1.6 * cm,
        topMargin=1.3 * cm, bottomMargin=1.0 * cm,
        title=f"Briefing Ejecutivo · {client_filter or 'Cliente'} · {month_label}",
        author="Watermelon System",
    )

    styles = getSampleStyleSheet()

    # =============================================================
    # ESTILOS
    # =============================================================
    s_title_top = ParagraphStyle(
        "WMBriefTitleTop", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=8.5, leading=10.5,
        textColor=colors.HexColor("#94a3b8"), spaceAfter=1,
    )
    s_title = ParagraphStyle(
        "WMBriefTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=17, leading=21,
        textColor=colors.HexColor("#0f172a"), spaceAfter=1,
    )
    s_subtitle = ParagraphStyle(
        "WMBriefSubtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=11, leading=14,
        textColor=colors.HexColor("#475569"), spaceAfter=4,
    )
    s_kpi = ParagraphStyle(
        "WMBriefKPI", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=12,
        textColor=colors.HexColor("#0f172a"),
    )
    s_section = ParagraphStyle(
        "WMBriefSection", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=9.5, leading=12,
        textColor=colors.HexColor("#0ea5e9"),
        spaceBefore=10, spaceAfter=4,
    )
    s_body = ParagraphStyle(
        "WMBriefBody", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9.8, leading=13.5,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_JUSTIFY, spaceAfter=4,
    )
    s_priority = ParagraphStyle(
        "WMBriefPriority", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=13,
        textColor=colors.HexColor("#0f172a"),
        leftIndent=14, firstLineIndent=-14,
        spaceAfter=4, alignment=TA_JUSTIFY,
    )
    s_muted = ParagraphStyle(
        "WMBriefMuted", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8, leading=10.5,
        textColor=colors.HexColor("#64748b"),
        spaceBefore=2, spaceAfter=1,
    )
    s_disclaimer = ParagraphStyle(
        "WMBriefDisclaimer", parent=styles["Normal"],
        fontName="Helvetica-Oblique", fontSize=7.5, leading=10,
        textColor=colors.HexColor("#94a3b8"),
        spaceBefore=4, alignment=TA_JUSTIFY,
    )
    s_table_header = ParagraphStyle(
        "WMBriefTableHeader", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=8.5, leading=11,
        textColor=colors.white,
    )
    s_table_cell = ParagraphStyle(
        "WMBriefTableCell", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8.5, leading=11,
        textColor=colors.HexColor("#0f172a"),
    )
    s_severity_chip = ParagraphStyle(
        "WMBriefSeverityChip", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=7.5, leading=10,
        textColor=colors.white, alignment=TA_CENTER,
    )

    flow: List[Any] = []

    # =============================================================
    # HEADER
    # =============================================================
    flow.append(Paragraph(
        "WATERMELON SYSTEM · BRIEFING EJECUTIVO MENSUAL",
        s_title_top,
    ))
    flow.append(Paragraph(
        f"<b>{(client_filter or 'Cliente').upper()}</b>",
        s_title,
    ))
    flow.append(Paragraph(month_label, s_subtitle))

    # KPIs line
    sev_count: Dict[str, int] = {}
    for ag in aggregates:
        sev = ag.get("latest_severity", "").strip() or "(sin severidad)"
        sev_count[sev] = sev_count.get(sev, 0) + 1

    chips_parts: List[str] = []
    severity_order = [
        "CRÍTICA", "ACCIÓN REQUERIDA", "ATENCIÓN",
        "VIGILANCIA", "CONDICIÓN ACEPTABLE",
    ]
    for sev in severity_order:
        n = sev_count.get(sev, 0)
        if n > 0:
            color = SEVERITY_COLORS.get(sev, "#64748b")
            chips_parts.append(
                f"<font color='{color}'><b>{n}</b> {sev}</font>"
            )

    kpi_line = (
        f"<b>{n_reports}</b> reportes técnicos &nbsp;·&nbsp; "
        f"<b>{n_assets}</b> activos cubiertos"
    )
    if chips_parts:
        kpi_line += "<br/>" + " &nbsp;·&nbsp; ".join(chips_parts)
    flow.append(Paragraph(kpi_line, s_kpi))
    flow.append(Spacer(1, 0.18 * cm))
    flow.append(HRFlowable(
        width="100%", thickness=0.6, color=colors.HexColor("#e6ebf2"),
    ))

    # =============================================================
    # CUERPO — Parsear el markdown del AI en bloques
    # =============================================================
    md_lines = markdown.splitlines()

    # Extraer secciones del markdown del AI
    sections: Dict[str, List[str]] = {
        "opening": [],
        "priorities_text": [],
        "portfolio": [],
        "closing": [],
        "disclaimer": [],
    }
    current_section = "opening"
    for line in md_lines:
        stripped = line.strip()
        if stripped.startswith("### Top 3 prioridades"):
            current_section = "priorities_text"
            continue
        if stripped.startswith("### Estado del portafolio"):
            current_section = "portfolio"
            continue
        if "ISO 18436-2" in stripped and "presente briefing" in stripped.lower():
            current_section = "disclaimer"
            sections[current_section].append(line)
            continue
        if (current_section == "portfolio"
                and stripped
                and not stripped.startswith("###")):
            # Tras el portafolio puede venir una recomendación cierre
            # antes del disclaimer. Detectamos si la línea NO es del
            # disclaimer (no menciona ISO 18436-2) y la dejamos en
            # portfolio. Después separamos en post-procesamiento.
            pass
        if stripped.startswith("### "):
            # cualquier otro header lo ignoramos por ahora
            continue
        sections[current_section].append(line)

    # OPENING (párrafo ejecutivo)
    flow.append(Paragraph("RESUMEN EJECUTIVO DEL MES", s_section))
    opening_text = "\n".join(sections["opening"]).strip()
    if opening_text:
        for para in _split_paragraphs(opening_text):
            flow.append(Paragraph(_md_inline_to_rl(para), s_body))
    else:
        flow.append(Paragraph(
            "(El AI no produjo apertura ejecutiva; revisar el "
            "markdown crudo en la sesión de generación.)", s_muted,
        ))

    # PRIORIDADES (parsear bullets numerados del AI)
    flow.append(Paragraph("TOP 3 PRIORIDADES OPERATIVAS", s_section))
    priorities_text = "\n".join(sections["priorities_text"]).strip()
    priority_blocks = _parse_priorities(priorities_text)
    if priority_blocks:
        for i, block in enumerate(priority_blocks, 1):
            flow.append(Paragraph(
                f"<b>{i}.</b>&nbsp;&nbsp;{_md_inline_to_rl(block)}",
                s_priority,
            ))
    else:
        # Fallback: render del texto crudo
        if priorities_text:
            for para in _split_paragraphs(priorities_text):
                flow.append(Paragraph(_md_inline_to_rl(para), s_body))
        else:
            flow.append(Paragraph(
                "Sin prioridades elevadas reportadas en el periodo.",
                s_muted,
            ))

    # ESTADO DEL PORTAFOLIO — usamos la TABLA de aggregates en lugar de
    # la prosa del AI para ser más informativo y menos verboso.
    flow.append(Paragraph("ESTADO DEL PORTAFOLIO POR ACTIVO", s_section))
    if aggregates:
        # Tabla compacta: Activo | Reportes | Severidad final
        rows: List[List[Any]] = [[
            Paragraph("Activo", s_table_header),
            Paragraph("Reportes en el mes", s_table_header),
            Paragraph("Severidad final", s_table_header),
        ]]
        for ag in aggregates[:12]:  # máximo 12 para 1 página
            asset_label = (
                ag.get("asset_blob", "")
                or ag.get("instance_tag", "")
                or ag.get("instance_id", "")
                or "(activo sin tag)"
            )
            sev = ag.get("latest_severity", "") or "—"
            sev_color = SEVERITY_COLORS.get(sev, "#475569")
            rows.append([
                Paragraph(asset_label, s_table_cell),
                Paragraph(
                    str(ag.get("n_reports_in_month", 0)), s_table_cell,
                ),
                Paragraph(
                    f"<font color='{sev_color}'><b>{sev}</b></font>",
                    s_table_cell,
                ),
            ])
        usable_w = doc.width
        col_widths = [
            usable_w * 0.55,
            usable_w * 0.20,
            usable_w * 0.25,
        ]
        tbl = Table(rows, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
            ("TOPPADDING", (0, 0), (-1, 0), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 3.5),
            ("TOPPADDING", (0, 1), (-1, -1), 3.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f1f5f9"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ]))
        flow.append(tbl)
        if len(aggregates) > 12:
            flow.append(Paragraph(
                f"… y {len(aggregates) - 12} activo(s) adicionales en "
                f"condición aceptable o vigilancia rutinaria.",
                s_muted,
            ))
    else:
        flow.append(Paragraph(
            "No hay activos archivados para el periodo seleccionado.",
            s_muted,
        ))

    # CIERRE (recomendación global + disclaimer)
    closing_text = "\n".join(sections["closing"]).strip()
    disclaimer_text = "\n".join(sections["disclaimer"]).strip()
    if not disclaimer_text:
        # Fallback al disclaimer fijo
        disclaimer_text = (
            "El presente briefing ejecutivo se emite conforme a la "
            "metodología Cat IV ISO 18436-2 con base en los reportes "
            "técnicos del periodo. La planificación operativa final "
            "es responsabilidad del operador del activo conforme a "
            "su sistema de gestión de integridad."
        )

    flow.append(Spacer(1, 0.20 * cm))
    flow.append(HRFlowable(
        width="100%", thickness=0.4, color=colors.HexColor("#e6ebf2"),
    ))
    flow.append(Paragraph(_md_inline_to_rl(disclaimer_text), s_disclaimer))

    # FOOTER con versión
    footer_bits: List[str] = ["Generado por Watermelon System"]
    if _ver:
        footer_bits.append(_ver)
    footer_bits.append("SIGASAS")
    footer_bits.append(datetime.now().strftime("%Y-%m-%d"))
    flow.append(Paragraph(
        " · ".join(footer_bits),
        ParagraphStyle(
            "WMBriefFooter", parent=styles["Normal"],
            fontName="Helvetica", fontSize=7, leading=9,
            textColor=colors.HexColor("#94a3b8"),
            alignment=TA_CENTER, spaceBefore=4,
        ),
    ))

    doc.build(flow)
    return buf.getvalue()


def _split_paragraphs(text: str) -> List[str]:
    """Divide texto en párrafos por dobles saltos de línea."""
    if not text:
        return []
    blocks = _re.split(r"\n\s*\n", text.strip())
    return [b.strip() for b in blocks if b.strip()]


def _parse_priorities(text: str) -> List[str]:
    """Extrae los bloques de prioridad numerados (1., 2., 3.) del
    markdown del AI. Devuelve una lista con el texto de cada
    prioridad sin el número leading."""
    if not text:
        return []
    blocks: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        m = _re.match(r"^(\d+)\.\s+(.*)", stripped)
        if m:
            if current:
                blocks.append(" ".join(current).strip())
                current = []
            content = m.group(2)
            if content:
                current.append(content)
        else:
            if stripped and current:
                current.append(stripped)
    if current:
        blocks.append(" ".join(current).strip())
    return blocks


__all__ = [
    "generate_monthly_briefing_pdf",
]
