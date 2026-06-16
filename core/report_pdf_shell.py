"""
core.report_pdf_shell — Carcasa profesional de PDF (portada + TOC + banda)
==========================================================================

Extrae el "shell" clase-mundial de los Reports (portada SIGA-style, banda de
encabezado con código de formato controlado, pie con disclaimer + version
stamp, Tabla de Contenido clickeable con dot-leaders) a un módulo REUSABLE y
HEADLESS (sin Streamlit), para que tanto Reports como el Briefing por activo
generen el MISMO formato profesional.

Uso:
    from core.report_pdf_shell import render_report_pdf, REGULAR, BOLD, make_styles

    styles = make_styles()
    body = [...]   # flowables; usa estilo "WMTOC1"/"WMTOC2" en los headings
                   # que deban entrar a la Tabla de Contenido.
    pdf_bytes = render_report_pdf(meta, body)

`meta` (todas opcionales salvo report_title):
    report_title, format_code, format_version, format_date,
    asset_class, unit, asset, asset_model, location, client,
    train_description,
    prepared_by, prepared_role, prepared_city,
    reviewed_by, reviewed_role, reviewed_city,
    report_date, period, consecutive

Diseño: el contenido (body) lo arma quien llama. Este módulo solo monta la
portada, el TOC y dibuja la banda/pie en cada página vía multiBuild.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ASSETS_DIR = _PROJECT_ROOT / "assets"
_FONTS_DIR = _ASSETS_DIR / "fonts"
WATERMELON_LOGO = _ASSETS_DIR / "watermelon_logo.png"

_CYAN = "#0ea5e9"
_INK = "#0f172a"
_INK2 = "#111827"
_SLATE = "#475569"
_MUTE = "#94a3b8"


def _register_unicode_fonts():
    """Registra una fuente Unicode TrueType desde assets/fonts/ en orden de
    preferencia (IBM Plex Sans → DejaVu Sans → Helvetica). Devuelve
    (regular_name, bold_name). Mismo criterio que el módulo de Reports para
    que ambos se vean idénticos."""
    candidates = (
        ("IBMPlexSans", "IBMPlexSans-Regular.ttf", "IBMPlexSans-Bold.ttf"),
        ("DejaVuSans", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    )
    for family, regular_file, bold_file in candidates:
        try:
            regular_path = _FONTS_DIR / regular_file
            bold_path = _FONTS_DIR / bold_file
            if not (regular_path.exists() and bold_path.exists()):
                continue
            bold_name = f"{family}-Bold"
            if family not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(family, str(regular_path)))
            if bold_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
            registerFontFamily(family, normal=family, bold=bold_name,
                               italic=family, boldItalic=bold_name)
            return family, bold_name
        except Exception:
            continue
    return "Helvetica", "Helvetica-Bold"


REGULAR, BOLD = _register_unicode_fonts()


def _today_str() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def paragraph_safe(text: str) -> str:
    """Escapa para Paragraph pero rehabilita <b>/<i>/<sub>/<sup>."""
    escaped = (
        (text or "")
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    for opener, closer in (("b", "b"), ("i", "i"), ("sub", "sub"), ("sup", "sup")):
        escaped = escaped.replace(f"&lt;{opener}&gt;", f"<{opener}>")
        escaped = escaped.replace(f"&lt;/{closer}&gt;", f"</{closer}>")
    return escaped


def md_inline_to_rl(text: str) -> str:
    """markdown inline (**bold**, *italic*, `code`) → tags de ReportLab."""
    import re as _re
    escaped = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = _re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    escaped = _re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    return escaped


def render_markdown_flowables(md: str, styles) -> List[Any]:
    """Parsea markdown (### heading, - bullets, 1. numbered, párrafos,
    **bold**) a flowables ReportLab nativos. Así el cliente no ve `###` ni
    `**` crudos. Mismo motor que el bloque clínico de Reports."""
    import re as _re
    out: List[Any] = []
    if not md:
        return out
    lines = md.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("#"):
            out.append(Paragraph(md_inline_to_rl(stripped.lstrip("#").strip()),
                                  styles["WMClinicalHeading"]))
            i += 1
            continue
        if stripped in ("---", "***", "___"):
            out.append(Spacer(1, 0.18 * cm))
            i += 1
            continue
        if _re.match(r"^[-*+]\s+", stripped):
            while i < n and _re.match(r"^[-*+]\s+", lines[i].strip()):
                content = _re.sub(r"^[-*+]\s+", "", lines[i].strip())
                j = i + 1
                while j < n:
                    nxt = lines[j].strip()
                    if (not nxt or _re.match(r"^[-*+]\s+", nxt)
                            or _re.match(r"^\d+\.\s+", nxt) or nxt.startswith("#")
                            or nxt in ("---", "***", "___")):
                        break
                    content += " " + nxt
                    j += 1
                out.append(Paragraph("•&nbsp;&nbsp;" + md_inline_to_rl(content),
                                     styles["WMClinicalBullet"]))
                i = j
            continue
        if _re.match(r"^\d+\.\s+", stripped):
            while i < n and _re.match(r"^\d+\.\s+", lines[i].strip()):
                m = _re.match(r"^(\d+)\.\s+(.*)", lines[i].strip())
                if not m:
                    break
                num, content = m.group(1), m.group(2)
                j = i + 1
                while j < n:
                    nxt = lines[j].strip()
                    if (not nxt or _re.match(r"^[-*+]\s+", nxt)
                            or _re.match(r"^\d+\.\s+", nxt) or nxt.startswith("#")
                            or nxt in ("---", "***", "___")):
                        break
                    content += " " + nxt
                    j += 1
                out.append(Paragraph(f"<b>{num}.</b>&nbsp;&nbsp;{md_inline_to_rl(content)}",
                                     styles["WMClinicalNumbered"]))
                i = j
            continue
        # párrafo regular
        para = stripped
        j = i + 1
        while j < n:
            nxt = lines[j].strip()
            if (not nxt or nxt.startswith("#") or _re.match(r"^[-*+]\s+", nxt)
                    or _re.match(r"^\d+\.\s+", nxt) or nxt in ("---", "***", "___")):
                break
            para += " " + nxt
            j += 1
        out.append(Paragraph(md_inline_to_rl(para), styles["WMClinicalBody"]))
        i = j
    return out


def make_styles():
    """Stylesheet idéntico al de Reports. Los headings que deban entrar al
    TOC usan WMTOC1 (nivel 0) / WMTOC2 (nivel 1)."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="WMTitle", parent=styles["Title"], fontName=BOLD, fontSize=15, leading=18, alignment=TA_LEFT, textColor=colors.HexColor(_INK), spaceAfter=6))
    styles.add(ParagraphStyle(name="WMSubTitle", parent=styles["Normal"], fontName=BOLD, fontSize=12.5, leading=15, alignment=TA_LEFT, textColor=colors.HexColor(_INK2), spaceAfter=5))
    styles.add(ParagraphStyle(name="WMBody", parent=styles["BodyText"], fontName=REGULAR, fontSize=10.5, leading=15.5, alignment=TA_JUSTIFY, textColor=colors.HexColor(_INK2), spaceAfter=10))
    styles.add(ParagraphStyle(name="WMMeta", parent=styles["Normal"], fontName=REGULAR, fontSize=10.4, leading=14.2, alignment=TA_LEFT, textColor=colors.HexColor(_INK2), spaceAfter=5))
    styles.add(ParagraphStyle(name="WMSection", parent=styles["Heading2"], fontName=BOLD, fontSize=14.6, leading=18.5, alignment=TA_LEFT, textColor=colors.HexColor(_INK), spaceBefore=6, spaceAfter=11))
    styles.add(ParagraphStyle(name="WMFigureCaption", parent=styles["Normal"], fontName=BOLD, fontSize=10.5, leading=13.5, alignment=TA_CENTER, textColor=colors.HexColor(_INK2), spaceBefore=6, spaceAfter=8))
    styles.add(ParagraphStyle(name="WMFigureText", parent=styles["BodyText"], fontName=REGULAR, fontSize=10.2, leading=14.8, alignment=TA_JUSTIFY, textColor=colors.HexColor(_INK2), spaceAfter=16))
    styles.add(ParagraphStyle(name="WMTableCell", parent=styles["Normal"], fontName=REGULAR, fontSize=8.4, leading=11, alignment=TA_LEFT, textColor=colors.HexColor(_INK2)))
    styles.add(ParagraphStyle(name="WMTableHeader", parent=styles["Normal"], fontName=BOLD, fontSize=8.5, leading=11, alignment=TA_LEFT, textColor=colors.HexColor("#ffffff")))
    # Estilos del bloque clínico (markdown del AI → ReportLab nativo)
    styles.add(ParagraphStyle(name="WMClinicalHeading", parent=styles["Normal"], fontName=BOLD, fontSize=10.6, leading=14, alignment=TA_LEFT, textColor=colors.HexColor(_INK), spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="WMClinicalBody", parent=styles["BodyText"], fontName=REGULAR, fontSize=10.2, leading=14.8, alignment=TA_JUSTIFY, textColor=colors.HexColor(_INK2), spaceAfter=8))
    styles.add(ParagraphStyle(name="WMClinicalBullet", parent=styles["BodyText"], fontName=REGULAR, fontSize=10.2, leading=14.6, alignment=TA_JUSTIFY, textColor=colors.HexColor(_INK2), leftIndent=14, firstLineIndent=-14, spaceAfter=5))
    styles.add(ParagraphStyle(name="WMClinicalNumbered", parent=styles["BodyText"], fontName=REGULAR, fontSize=10.2, leading=14.6, alignment=TA_JUSTIFY, textColor=colors.HexColor(_INK2), leftIndent=18, firstLineIndent=-18, spaceAfter=5))
    # Entradas que SÍ van al TOC (visualmente = WMSection / WMFigureCaption)
    styles.add(ParagraphStyle(name="WMTOC1", parent=styles["WMSection"]))
    styles.add(ParagraphStyle(name="WMTOC2", parent=styles["WMFigureCaption"]))
    return styles


_TOC_L0 = ParagraphStyle(name="WMTOCLevel0", fontName=BOLD, fontSize=11, leading=16, leftIndent=0, firstLineIndent=0, spaceBefore=8, spaceAfter=2, textColor=colors.HexColor(_INK))
_TOC_L1 = ParagraphStyle(name="WMTOCLevel1", fontName=REGULAR, fontSize=10, leading=14, leftIndent=18, firstLineIndent=0, spaceBefore=2, spaceAfter=1, textColor=colors.HexColor("#334155"))


class WMDocTemplate(SimpleDocTemplate):
    """multiBuild + afterFlowable para registrar entradas TOC con su página.
    WMTOC1 → nivel 0, WMTOC2 → nivel 1. Key estable por id(flowable) para que
    multiBuild converja en 2 pasadas."""

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        try:
            style_name = flowable.style.name
        except Exception:
            return
        if style_name == "WMTOC1":
            level = 0
        elif style_name == "WMTOC2":
            level = 1
        else:
            return
        text = flowable.getPlainText()
        key = f"toc-{level}-{id(flowable):x}"
        self.canv.bookmarkPage(key)
        self.notify("TOCEntry", (level, text, self.page, key))


def _make_drawers(meta: Dict[str, Any]):
    page_width, page_height = A4
    format_code = meta.get("format_code") or "WMS-FMT-001"
    format_version = meta.get("format_version") or "1"
    format_date = meta.get("format_date") or _today_str()
    format_header = f"{format_code} | Versión {format_version} | Fecha {format_date}"
    report_title = meta.get("report_title") or "Reporte técnico"
    internal_left = 2.1 * cm
    internal_right = 2.1 * cm
    internal_width_end = page_width - internal_right
    footer = ("INFORME VÁLIDO ÚNICAMENTE PARA LAS CONDICIONES PRESENTES "
              "DURANTE EL SERVICIO. NO PODRÁ SER COPIADO PARCIAL O TOTALMENTE "
              "SIN PREVIA AUTORIZACIÓN.")

    def _version_full() -> str:
        try:
            from core.version import get_version_info as _gvi
            v = _gvi()
            line = f"Generado con Watermelon System {v['version']}"
            if v.get("commit"):
                line += f" · build {v['commit']}"
            if v.get("date"):
                line += f" · {v['date']}"
            return line
        except Exception:
            return ""

    def _version_short() -> str:
        try:
            from core.version import get_version_short as _gvs
            return f"Watermelon System {_gvs()}"
        except Exception:
            return ""

    def draw_cover_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#ffffff"))
        canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor(_INK))
        canvas.setFont(BOLD, 7.8)
        canvas.drawString(internal_left, page_height - 1.0 * cm, format_header)
        canvas.setFont(BOLD, 9.0)
        canvas.drawRightString(page_width - internal_right, page_height - 1.0 * cm, f"Página {doc.page}")
        canvas.setStrokeColor(colors.HexColor(_CYAN))
        canvas.setLineWidth(0.8)
        canvas.line(internal_left, page_height - 1.35 * cm, internal_width_end, page_height - 1.35 * cm)
        canvas.line(internal_left, 0.95 * cm, internal_width_end, 0.95 * cm)
        canvas.setFillColor(colors.HexColor(_SLATE))
        canvas.setFont(REGULAR, 6.4)
        canvas.drawCentredString((internal_left + internal_width_end) / 2, 0.55 * cm, footer)
        _vf = _version_full()
        if _vf:
            canvas.setFillColor(colors.HexColor(_MUTE))
            canvas.setFont(REGULAR, 5.6)
            canvas.drawRightString(page_width - internal_right, 0.30 * cm, _vf)
        canvas.restoreState()

    def draw_internal_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#ffffff"))
        canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
        canvas.setFont(BOLD, 11)
        canvas.setFillColor(colors.HexColor(_INK2))
        canvas.drawRightString(page_width - 1.2 * cm, page_height - 1.0 * cm, f"Página {doc.page}")
        canvas.setStrokeColor(colors.HexColor(_CYAN))
        canvas.setLineWidth(1.1)
        canvas.line(internal_left, page_height - 1.35 * cm, internal_width_end, page_height - 1.35 * cm)
        canvas.setFillColor(colors.HexColor(_INK))
        canvas.setFont(BOLD, 7.8)
        canvas.drawString(internal_left, page_height - 1.0 * cm, format_header)
        canvas.setFont(REGULAR, 7.8)
        canvas.drawString(internal_left + 7.2 * cm, page_height - 1.0 * cm, f"| {report_title}")
        canvas.setStrokeColor(colors.HexColor(_CYAN))
        canvas.setLineWidth(1.0)
        canvas.line(internal_left, 0.95 * cm, internal_width_end, 0.95 * cm)
        canvas.setFillColor(colors.HexColor(_INK2))
        canvas.setFont(REGULAR, 6.4)
        canvas.drawCentredString((internal_left + internal_width_end) / 2, 0.55 * cm, footer)
        _vs = _version_short()
        if _vs:
            canvas.setFillColor(colors.HexColor(_MUTE))
            canvas.setFont(REGULAR, 5.6)
            canvas.drawRightString(page_width - internal_right, 0.30 * cm, _vs)
        canvas.restoreState()

    return draw_cover_page, draw_internal_page


def build_cover_flowables(meta: Dict[str, Any], styles) -> List[Any]:
    """Portada SIGA-style centrada: logo, eyebrow, título, marca, bloque del
    activo, firmas paralelas, mini-tabla de fecha/consecutivo."""
    story: List[Any] = []

    if WATERMELON_LOGO.exists():
        logo = Image(str(WATERMELON_LOGO), width=5.8 * cm, height=2.7 * cm)
        logo.hAlign = "CENTER"
        story.append(Spacer(1, 0.40 * cm))
        story.append(logo)
        story.append(Spacer(1, 0.85 * cm))

    story.append(Paragraph("Machinery Diagnostics Engineering", ParagraphStyle(
        name="WMCoverEyebrow", parent=styles["Normal"], fontName=BOLD, fontSize=11,
        leading=14, alignment=TA_CENTER, textColor=colors.HexColor(_SLATE), spaceAfter=6)))

    story.append(Paragraph(paragraph_safe(meta.get("report_title") or "REPORTE TÉCNICO"),
        ParagraphStyle(name="WMCoverReportTitle", parent=styles["Normal"], fontName=BOLD,
        fontSize=20, leading=24, alignment=TA_CENTER, textColor=colors.HexColor(_INK), spaceAfter=4)))

    story.append(Paragraph("Watermelon System", ParagraphStyle(
        name="WMCoverBrand", parent=styles["Normal"], fontName=REGULAR, fontSize=12,
        leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#64748b"), spaceAfter=2)))

    # Bloque grande del activo
    asset_class = (meta.get("asset_class") or "").strip()
    asset_name = (meta.get("asset") or "").strip()
    unit_name = (meta.get("unit") or "").strip()
    asset_model = (meta.get("asset_model") or "").strip()
    location_name = (meta.get("location") or "").strip()
    client_name = (meta.get("client") or "").strip()

    line1_parts = []
    if asset_class:
        line1_parts.append(asset_class)
    if unit_name:
        line1_parts.append(unit_name)
    elif asset_name and not asset_class:
        line1_parts.append(asset_name)
    line1 = " ".join(line1_parts).strip().upper()
    cover_block_lines = [ln for ln in [
        line1, asset_model.upper() if asset_model else "",
        location_name.upper() if location_name else "",
        client_name.upper() if client_name else "",
    ] if ln]

    story.append(HRFlowable(width="40%", thickness=0.7, color=colors.HexColor(_MUTE),
                            spaceBefore=4, spaceAfter=14, hAlign="CENTER"))
    for idx, line in enumerate(cover_block_lines):
        story.append(Paragraph(paragraph_safe(line), ParagraphStyle(
            name=f"WMCoverBlock_{idx}", parent=styles["Normal"], fontName=BOLD,
            fontSize=24 if idx == 0 else 16, leading=28 if idx == 0 else 20,
            alignment=TA_CENTER, textColor=colors.HexColor(_INK), spaceAfter=2)))

    train_text = (meta.get("train_description") or "").strip()
    if train_text:
        story.append(Spacer(1, 0.30 * cm))
        story.append(Paragraph(paragraph_safe(train_text), ParagraphStyle(
            name="WMCoverTrain", parent=styles["Normal"], fontName=REGULAR, fontSize=10.5,
            leading=14, alignment=TA_CENTER, textColor=colors.HexColor(_SLATE),
            spaceBefore=4, spaceAfter=3)))

    story.append(HRFlowable(width="40%", thickness=0.7, color=colors.HexColor(_MUTE),
                            spaceBefore=14, spaceAfter=14, hAlign="CENTER"))
    story.append(Spacer(1, 3.50 * cm))

    # Firmas
    prepared_by = (meta.get("prepared_by") or "").strip()
    prepared_role = (meta.get("prepared_role") or "Junior Condition Monitoring Engineer").strip()
    prepared_city = (meta.get("prepared_city") or "Cajicá, Cundinamarca · Colombia").strip()
    reviewed_by = (meta.get("reviewed_by") or "").strip()
    reviewed_role = (meta.get("reviewed_role") or "Machinery Diagnostic Champion").strip()
    reviewed_city = (meta.get("reviewed_city") or "Cajicá, Cundinamarca · Colombia").strip()

    sig_label = ParagraphStyle(name="WMSigLabel", parent=styles["Normal"], fontName=BOLD, fontSize=10.2, leading=13, alignment=TA_CENTER, textColor=colors.HexColor(_INK), spaceAfter=4)
    sig_name = ParagraphStyle(name="WMSigName", parent=styles["Normal"], fontName=BOLD, fontSize=11, leading=14, alignment=TA_CENTER, textColor=colors.HexColor(_INK), spaceAfter=2)
    sig_role = ParagraphStyle(name="WMSigRole", parent=styles["Normal"], fontName=REGULAR, fontSize=9.5, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#374151"), spaceAfter=2)
    sig_city = ParagraphStyle(name="WMSigCity", parent=styles["Normal"], fontName=REGULAR, fontSize=9.0, leading=11.5, alignment=TA_CENTER, textColor=colors.HexColor("#64748b"))

    def _cell(label, name, role, city):
        if not name:
            return [Paragraph("", sig_label)]
        c = [Paragraph(label, sig_label), Paragraph(paragraph_safe(name), sig_name)]
        if role:
            c.append(Paragraph(paragraph_safe(role), sig_role))
        if city:
            c.append(Paragraph(paragraph_safe(city), sig_city))
        return c

    if prepared_by and reviewed_by:
        # Dos firmas → dos columnas paralelas centradas.
        sig_tbl = Table([[
            _cell("Preparado por:", prepared_by, prepared_role, prepared_city),
            _cell("Revisado por:", reviewed_by, reviewed_role, reviewed_city),
        ]], colWidths=[8.3 * cm, 8.3 * cm])
        sig_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        sig_tbl.hAlign = "CENTER"
        story.append(sig_tbl)
        story.append(Spacer(1, 4.50 * cm))
    elif prepared_by or reviewed_by:
        # Una sola firma → columna ÚNICA centrada en la página (el nombre del
        # especialista queda al centro, no corrido a la izquierda).
        if prepared_by:
            label, name, role, city = "Preparado por:", prepared_by, prepared_role, prepared_city
        else:
            label, name, role, city = "Revisado por:", reviewed_by, reviewed_role, reviewed_city
        sig_tbl = Table([[_cell(label, name, role, city)]], colWidths=[11.0 * cm])
        sig_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        sig_tbl.hAlign = "CENTER"
        story.append(sig_tbl)
        story.append(Spacer(1, 4.50 * cm))

    # Mini-tabla fecha/periodo/consecutivo
    label_style = ParagraphStyle(name="WMMetaLabel", parent=styles["WMMeta"], fontName=BOLD, fontSize=10.0, textColor=colors.HexColor(_INK))
    value_style = ParagraphStyle(name="WMMetaValue", parent=styles["WMMeta"], fontName=REGULAR, fontSize=10.0, textColor=colors.HexColor(_INK2))
    meta_rows: List[List[Any]] = [[
        Paragraph("Fecha del reporte", label_style),
        Paragraph(paragraph_safe(meta.get("report_date") or _today_str()), value_style),
    ]]
    period_value = (meta.get("period") or "").strip()
    if period_value and period_value.lower() not in ("no aplica", "n/a", "-"):
        meta_rows.append([Paragraph("Periodo evaluado", label_style), Paragraph(paragraph_safe(period_value), value_style)])
    consecutive_value = (meta.get("consecutive") or "").strip()
    if consecutive_value:
        meta_rows.append([Paragraph("Consecutivo", label_style), Paragraph(paragraph_safe(consecutive_value), value_style)])

    meta_tbl = Table(meta_rows, colWidths=[4.4 * cm, 6.6 * cm])
    meta_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor("#cbd5e1")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
    ]))
    meta_tbl.hAlign = "CENTER"
    story.append(meta_tbl)
    story.append(PageBreak())
    return story


def build_toc_flowables(styles) -> List[Any]:
    story: List[Any] = [Paragraph("TABLA DE CONTENIDO", styles["WMSection"]), Spacer(1, 0.20 * cm)]
    toc = TableOfContents()
    toc.levelStyles = [_TOC_L0, _TOC_L1]
    toc.dotsMinLevel = 0
    story.append(toc)
    story.append(PageBreak())
    return story


def render_report_pdf(meta: Dict[str, Any], body_flowables: List[Any]) -> bytes:
    """Monta portada + TOC + body y devuelve los bytes del PDF (multiBuild)."""
    buffer = BytesIO()
    styles = make_styles()
    doc = WMDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2.1 * cm, rightMargin=2.1 * cm,
        topMargin=1.6 * cm, bottomMargin=1.5 * cm,
        title=meta.get("report_title") or "Watermelon System Report",
        author=meta.get("prepared_by") or "Watermelon System",
    )
    draw_cover_page, draw_internal_page = _make_drawers(meta)

    story: List[Any] = []
    story += build_cover_flowables(meta, styles)
    story += build_toc_flowables(styles)
    story += body_flowables

    doc.multiBuild(story, onFirstPage=draw_cover_page, onLaterPages=draw_internal_page)
    return buffer.getvalue()


__all__ = [
    "render_report_pdf", "make_styles", "build_cover_flowables",
    "build_toc_flowables", "paragraph_safe", "render_markdown_flowables",
    "md_inline_to_rl", "REGULAR", "BOLD", "WMDocTemplate", "WATERMELON_LOGO",
]
