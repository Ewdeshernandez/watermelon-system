"""
core.briefing_report_pdf — PDF multi-sección del Briefing por activo (F2)
========================================================================

Arma el PDF del Briefing Semanal/Mensual de UN activo, en el MISMO estilo y
marca de los reportes (paleta SIGA, Helvetica/Courier, header con logo + ISO).
A diferencia del reporte ejecutivo de 1 página (live_report_pdf), este es
multi-sección y embebe las figuras que produce core.briefing_figures (F1):
Tendencia, Espectro, Forma de onda, Órbita.

Diseño: recibe TODO el contenido como parámetros (figuras + textos IA + KPIs +
canales). La obtención de datos y la redacción IA viven en el orquestador (F3/F4)
— este módulo solo MAQUETA. Así queda testeable y desacoplado.

API:
    generate_briefing_pdf(instance_id, tag, train, period_label, health, kpis,
                          figures, summary, diagnosis, recommendations,
                          channels=None) -> bytes
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

# Paleta coherente con la app / reporte ejecutivo
_NAVY = "#0f172a"
_SLATE = "#475569"
_MUTE = "#94a3b8"
_LINE = "#e6ebf2"
_GREEN = "#1D9E75"
_AMBER = "#EF9F27"
_RED = "#E24B4A"
_GREEN_BG = "#E1F5EE"
_AMBER_BG = "#FAEEDA"
_RED_BG = "#FCEBEB"

_FIG_CAPTIONS = {
    "trend":    "Tendencia overall",
    "spectrum": "Espectro — canales apilados",
    "waveform": "Forma de onda — canales apilados",
    "orbit":    "Órbitas por cojinete",
}
_FIG_ORDER = ["trend", "spectrum", "waveform", "orbit"]


def _sev_colors(status: str):
    s = (status or "").lower()
    if "danger" in s or "crít" in s or "crit" in s:
        return _RED, _RED_BG
    if "alarma" in s or "alert" in s:
        return _AMBER, _AMBER_BG
    return _GREEN, _GREEN_BG


def _now_bogota_txt() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M")


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
) -> bytes:
    """Devuelve los bytes del PDF del briefing del activo."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
        HRFlowable,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.6 * cm, rightMargin=1.6 * cm,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
    )

    st_title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=20,
                              textColor=colors.HexColor(_NAVY), spaceAfter=2, leading=23)
    st_sub = ParagraphStyle("s", fontName="Helvetica", fontSize=10.5,
                            textColor=colors.HexColor(_SLATE), spaceAfter=1, leading=13)
    st_meta = ParagraphStyle("m", fontName="Courier", fontSize=8.5,
                             textColor=colors.HexColor(_MUTE), leading=11)
    st_section = ParagraphStyle("sec", fontName="Helvetica-Bold", fontSize=12,
                                textColor=colors.HexColor(_NAVY), spaceBefore=12,
                                spaceAfter=4, leading=15)
    st_body = ParagraphStyle("b", fontName="Helvetica", fontSize=9.5,
                             textColor=colors.HexColor(_NAVY), leading=14,
                             alignment=TA_JUSTIFY, spaceAfter=3)
    st_cap = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.5,
                            textColor=colors.HexColor(_MUTE), spaceBefore=2,
                            spaceAfter=8, alignment=TA_RIGHT)

    story: List[Any] = []

    # ---- Header con branding ----
    header_left = []
    header_left.append(Paragraph(f"{tag} — Briefing {period_label}", st_title))
    header_left.append(Paragraph(train or "—", st_sub))
    header_left.append(Paragraph(
        f"Generado {_now_bogota_txt()} · ISO 20816 / API 670", st_meta))

    logo_cell: Any = ""
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "watermelon_logo.png"
    if logo_path.exists():
        try:
            logo_cell = Image(str(logo_path), width=3.6 * cm, height=3.6 * cm * 0.494)
            logo_cell.hAlign = "RIGHT"
        except Exception:
            logo_cell = ""
    htbl = Table([[header_left, logo_cell]], colWidths=[12.5 * cm, 5.5 * cm])
    htbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(htbl)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor(_LINE)))
    story.append(Spacer(1, 8))

    # ---- KPIs ----
    def _kpi(label, value, vcolor=None):
        lab = ParagraphStyle("kl", fontName="Helvetica-Bold", fontSize=8,
                             textColor=colors.HexColor(_MUTE))
        val = ParagraphStyle("kv", fontName="Courier-Bold", fontSize=12,
                             textColor=vcolor or colors.HexColor(_NAVY), leading=14)
        return [Paragraph(label.upper(), lab), Paragraph(str(value), val)]

    hcolor = colors.HexColor(health.get("color", _MUTE))
    kpi_tbl = Table([[
        _kpi("Salud", health.get("score", "—"), hcolor),
        _kpi("Estado", kpis.get("status", "—")),
        _kpi("Velocidad", kpis.get("speed", "—")),
        _kpi("Alarmas", kpis.get("alarms", 0),
             colors.HexColor(_RED) if kpis.get("alarms", 0) else colors.HexColor(_GREEN)),
    ]], colWidths=[4.45 * cm] * 4)
    kpi_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(kpi_tbl)
    story.append(Paragraph(f"Zona ISO: {health.get('zone', '—')}", st_meta))

    # ---- Resumen gerencial (IA) ----
    if summary:
        story.append(Paragraph("Resumen gerencial", st_section))
        for para in [p for p in summary.split("\n") if p.strip()]:
            story.append(Paragraph(para.strip(), st_body))

    # ---- Diagnóstico (IA) ----
    if diagnosis:
        story.append(Paragraph("Diagnóstico", st_section))
        for para in [p for p in diagnosis.split("\n") if p.strip()]:
            story.append(Paragraph(para.strip(), st_body))

    # ---- Recomendaciones (IA) ----
    if recommendations:
        story.append(Paragraph("Recomendaciones", st_section))
        for i, rec in enumerate(recommendations, start=1):
            story.append(Paragraph(f"{i}. {rec}", st_body))

    # ---- Tabla de canales (Overall + 1X/2X) ----
    if channels:
        st_cell = ParagraphStyle("c", fontName="Helvetica", fontSize=8.5,
                                 textColor=colors.HexColor(_NAVY), leading=11)
        st_cn = ParagraphStyle("cn", fontName="Courier", fontSize=8.5,
                               textColor=colors.HexColor(_NAVY), leading=11)
        story.append(Paragraph("Canales — Overall + vectores 1X / 2X (API 670)", st_section))
        head = ["Estado", "Canal", "Ubicación", "Overall", "Unit", "1X", "2X"]
        data = [[Paragraph(f"<b>{h}</b>", st_cell) for h in head]]
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
        ctbl = Table(data, colWidths=[2.0 * cm, 2.4 * cm, 3.0 * cm, 2.2 * cm,
                                      1.8 * cm, 2.0 * cm, 2.0 * cm])
        ctbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(_LINE)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ] + row_styles))
        story.append(ctbl)

    # ---- Figuras (Tendencia, Espectro, Onda, Órbita) ----
    for key in _FIG_ORDER:
        png = (figures or {}).get(key)
        if not png:
            continue
        story.append(Paragraph(_FIG_CAPTIONS.get(key, key.title()), st_section))
        try:
            # ancho útil ~17.8cm; alto proporcional al tipo
            w = 17.8 * cm
            h = (5.0 * cm if key == "trend" else
                 10.6 * cm if key in ("spectrum", "waveform") else 8.0 * cm)
            img = Image(BytesIO(png), width=w, height=h)
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Paragraph(
                f"Figura — {_FIG_CAPTIONS.get(key, key)} · {tag}", st_cap))
        except Exception:
            pass

    # ---- Footer ----
    foot = ParagraphStyle("f", fontName="Helvetica", fontSize=8,
                          textColor=colors.HexColor(_MUTE))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor(_LINE)))
    story.append(Paragraph(
        "Generado por Watermelon System · SIGA GROUP S.A.S · "
        "Monitoreo de condición de maquinaria rotativa · ISO 20816-3 / API 670 · "
        "Documento válido únicamente para las condiciones presentes durante el servicio.",
        foot))

    doc.build(story)
    return buf.getvalue()


__all__ = ["generate_briefing_pdf"]
