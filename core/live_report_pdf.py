"""
core.live_report_pdf
====================

Reporte ejecutivo de UNA PÁGINA del estado en vivo de un activo, para
enviar al cliente (Parex, Ecopetrol, etc.). Es la entrega tangible del
servicio — el "executive summary" que un gerente lee en 30 segundos.

Contenido (1 página A4):
  - Header con branding SIGA + nombre del activo + fecha
  - KPIs: salud (0-100), velocidad, estado, # alarmas
  - Tabla de canales: Overall + 1X/2X (API 670) con color de severidad
  - Registro de eventos recientes (cruces de umbral)
  - Footer con norma ISO 20816 / API 670 + marca

Diseño minimalista flat coherente con la app (sin gradientes, mono para
cifras, color solo para severidad).
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional


# Paleta (coherente con la app)
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


def _sev_colors(status: str):
    s = (status or "").lower()
    if "danger" in s or "crít" in s or "crit" in s:
        return _RED, _RED_BG
    if "alarma" in s or "alert" in s:
        return _AMBER, _AMBER_BG
    return _GREEN, _GREEN_BG


def generate_live_report_pdf(
    instance_id: str,
    instance_obj: Any,
    health: Dict[str, Any],
    kpis: Dict[str, Any],
    channels: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    trend_png: Optional[bytes] = None,
    train_png: Optional[bytes] = None,
) -> bytes:
    """Genera el PDF ejecutivo de 1 página. Devuelve bytes.

    Args:
        health: {"score": int|None, "zone": str, "color": hex}
        kpis: {"speed": str, "status": str, "alarms": int, "last": str}
        channels: lista de dicts {sensor_label, plane_label, value, unit,
                  status, x1_amp, x1_ph, x2_amp, x2_ph}
        events: lista de dicts {sensor_label, to, value, unit, age, rising}
        trend_png: PNG opcional del gráfico de tendencia (kaleido)
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, mm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.4 * cm, rightMargin=1.4 * cm,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm,
        title=f"Reporte {instance_id}", author="Watermelon System · SIGASAS",
    )
    story: List[Any] = []

    mono = "Courier"
    st_title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=17,
                              textColor=colors.HexColor(_NAVY), spaceAfter=1, leading=20)
    st_sub = ParagraphStyle("s", fontName="Helvetica", fontSize=9,
                            textColor=colors.HexColor(_SLATE), spaceAfter=1)
    st_meta = ParagraphStyle("m", fontName="Courier", fontSize=8,
                             textColor=colors.HexColor(_MUTE))
    st_section = ParagraphStyle("sec", fontName="Helvetica-Bold", fontSize=9,
                                textColor=colors.HexColor(_SLATE), spaceBefore=10,
                                spaceAfter=5, leading=11)
    st_cell = ParagraphStyle("c", fontName="Helvetica", fontSize=8,
                             textColor=colors.HexColor(_NAVY))
    st_cellnum = ParagraphStyle("cn", fontName="Courier", fontSize=8,
                                textColor=colors.HexColor(_NAVY), alignment=TA_RIGHT)

    # ---------- Header ----------
    logo_cell = ""
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "siga_logo.png"
    header_left = []
    tag = getattr(instance_obj, "tag", "") or instance_id
    driver = " ".join(p for p in [getattr(instance_obj, "driver_manufacturer", ""),
                                  getattr(instance_obj, "driver_model", "")] if p)
    driven = " ".join(p for p in [getattr(instance_obj, "driven_manufacturer", ""),
                                  getattr(instance_obj, "driven_model", "")] if p)
    train = f"{driver} → {driven}" if driver and driven else (driver or driven or "")
    client = getattr(instance_obj, "client", "") or ""
    site = getattr(instance_obj, "site", "") or getattr(instance_obj, "location", "") or ""
    now_txt = datetime.now().strftime("%Y-%m-%d %H:%M")

    header_left.append(Paragraph(f"{tag} — Reporte de condición", st_title))
    header_left.append(Paragraph(train or "—", st_sub))
    sub2 = " · ".join(p for p in [client, site] if p)
    if sub2:
        header_left.append(Paragraph(sub2, st_sub))
    header_left.append(Paragraph(f"Generado {now_txt} · ISO 20816 / API 670", st_meta))

    if logo_path.exists():
        try:
            logo_cell = Image(str(logo_path), width=3.2 * cm, height=3.2 * cm * 0.32)
            logo_cell.hAlign = "RIGHT"
        except Exception:
            logo_cell = ""

    htbl = Table([[header_left, logo_cell]], colWidths=[12.5 * cm, 5.5 * cm])
    htbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(htbl)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor(_LINE)))
    story.append(Spacer(1, 8))

    # ---------- KPIs ----------
    score = health.get("score")
    score_txt = str(score) if score is not None else "—"
    zone = health.get("zone", "—")
    hcolor = colors.HexColor(health.get("color", _MUTE))

    def _kpi(label, value, vcolor=None):
        lab = ParagraphStyle("kl", fontName="Helvetica-Bold", fontSize=7,
                             textColor=colors.HexColor(_MUTE))
        val = ParagraphStyle("kv", fontName="Courier-Bold", fontSize=16,
                             textColor=vcolor or colors.HexColor(_NAVY), leading=18)
        return [Paragraph(label.upper(), lab), Paragraph(str(value), val)]

    kpi_tbl = Table([[
        _kpi("Salud", f"{score_txt}", hcolor),
        _kpi("Estado", kpis.get("status", "—")),
        _kpi("Velocidad", kpis.get("speed", "—")),
        _kpi("Alarmas", kpis.get("alarms", 0),
             colors.HexColor(_RED) if kpis.get("alarms", 0) else colors.HexColor(_GREEN)),
        _kpi("Última lectura", kpis.get("last", "—")),
    ]], colWidths=[3.4 * cm] * 5)
    kpi_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(kpi_tbl)
    story.append(Paragraph(f"Zona ISO: {zone}", st_meta))
    story.append(Spacer(1, 6))

    # ---------- Diagrama del tren (si hay PNG) ----------
    if train_png:
        try:
            story.append(Image(BytesIO(train_png), width=18 * cm, height=4.3 * cm))
            story.append(Spacer(1, 2))
        except Exception:
            pass

    # ---------- Tendencia (si hay PNG) ----------
    if trend_png:
        try:
            story.append(Paragraph("Tendencia overall", st_section))
            img = Image(BytesIO(trend_png), width=18 * cm, height=6 * cm)
            story.append(img)
        except Exception:
            pass

    # ---------- Tabla de canales ----------
    story.append(Paragraph("Canales — Overall + vectores 1X / 2X (API 670)", st_section))
    head = ["Estado", "Canal", "Ubicación", "Overall", "Unit", "1X ampl", "1X °", "2X ampl", "2X °"]
    data = [head]
    row_styles = []
    for i, c in enumerate(channels, start=1):
        fg, bg = _sev_colors(c.get("status", ""))
        data.append([
            (c.get("status") or "—"),
            c.get("sensor_label", "—"),
            c.get("plane_label") or "—",
            c.get("value", "—"),
            c.get("unit", ""),
            c.get("x1_amp", "—"),
            c.get("x1_ph", "—"),
            c.get("x2_amp", "—"),
            c.get("x2_ph", "—"),
        ])
        row_styles.append(("TEXTCOLOR", (0, i), (0, i), colors.HexColor(fg)))
        row_styles.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(bg)))

    ctbl = Table(data, colWidths=[1.9*cm, 1.6*cm, 2.3*cm, 1.9*cm, 1.5*cm,
                                  1.7*cm, 1.3*cm, 1.7*cm, 1.3*cm])
    base_style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(_SLATE)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ("FONTNAME", (1, 1), (-1, -1), "Courier"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, -1), 7.5),
        ("FONTSIZE", (0, 1), (0, -1), 6),
        ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (2, -1), "LEFT"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(_LINE)),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    ctbl.setStyle(TableStyle(base_style + row_styles))
    story.append(ctbl)
    story.append(Paragraph(
        "1X = componente síncrona (desbalance) · 2X = segunda armónica (desalineamiento / soltura)",
        st_meta))

    # ---------- Eventos ----------
    if events:
        story.append(Paragraph("Registro de eventos — cruces de umbral", st_section))
        ev_data = [["", "Canal", "Estado", "Valor", "Hace"]]
        ev_styles = []
        for i, e in enumerate(events[:8], start=1):
            arrow = "▲" if e.get("rising") else "▼"
            acolor = _RED if e.get("rising") else _GREEN
            ev_data.append([arrow, e.get("sensor_label", "—"), e.get("to", "—"),
                            f"{e.get('value','—')} {e.get('unit','')}", e.get("age", "—")])
            ev_styles.append(("TEXTCOLOR", (0, i), (0, i), colors.HexColor(acolor)))
        etbl = Table(ev_data, colWidths=[0.7*cm, 2.0*cm, 2.2*cm, 3.5*cm, 2.5*cm])
        etbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(_MUTE)),
            ("FONTNAME", (1, 1), (-1, -1), "Courier"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(_LINE)),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ] + ev_styles))
        story.append(etbl)

    # ---------- Footer ----------
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor(_LINE)))
    foot = ParagraphStyle("f", fontName="Helvetica", fontSize=7,
                          textColor=colors.HexColor(_MUTE), alignment=TA_CENTER)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Generado por Watermelon System · SIGASAS · Monitoreo de condición de maquinaria rotativa · "
        "ISO 20816-3 / API 670", foot))

    doc.build(story)
    return buf.getvalue()


def render_trend_png(sensor_series: List[Dict[str, Any]],
                     alarm: float = 0.0, danger: float = 0.0,
                     y_title: str = "valor") -> Optional[bytes]:
    """Renderiza el gráfico de tendencia a PNG para embeber en el PDF.

    sensor_series: lista de {label, x (lista datetime), y (lista float), color}
    Devuelve PNG bytes o None si kaleido/plotly no están disponibles
    (degradación graceful — el PDF se genera igual sin la tendencia).
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    try:
        fig = go.Figure()
        if danger > 0:
            fig.add_hline(y=danger, line=dict(color="#dc2626", width=1.2, dash="dash"),
                          annotation_text="Danger ", annotation_position="left",
                          annotation=dict(font=dict(color="#dc2626", size=9)))
        if alarm > 0:
            fig.add_hline(y=alarm, line=dict(color="#d97706", width=1.2, dash="dash"),
                          annotation_text="Alarma ", annotation_position="left",
                          annotation=dict(font=dict(color="#d97706", size=9)))
        for s in sensor_series:
            fig.add_trace(go.Scatter(
                x=s.get("x", []), y=s.get("y", []), mode="lines",
                line=dict(color=s.get("color", "#1e40af"), width=1.6, shape="spline", smoothing=0.6),
                name=s.get("label", ""),
            ))
        fig.update_layout(
            height=320, width=1000, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=55, r=20, t=30, b=30),
            font=dict(size=11, color="#475569"),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", showline=True, linecolor="#e5edf7"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=y_title,
                       showline=True, linecolor="#e5edf7"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        )
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


def render_train_png(driver_label: str, driven_label: str,
                     channels: List[Dict[str, Any]]) -> Optional[bytes]:
    """Esquema horizontal del tren (driver → coupling → driven) con dots de
    severidad por sensor, dibujado en plotly (nativo, funciona en Render sin
    libcairo). channels: [{sensor_label, plane_label, status, side}] donde
    side ∈ {'driver','driven'}. Devuelve PNG o None.
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    try:
        fig = go.Figure()
        # Cuerpos: driver (izq) y driven (der) como rectángulos redondeados
        def _body(x0, x1, color, label):
            fig.add_shape(type="rect", x0=x0, y0=-0.55, x1=x1, y1=0.55,
                          line=dict(color="#cbd5e1", width=1.2), fillcolor=color)
            fig.add_annotation(x=(x0+x1)/2, y=0.78, text=f"<b>{label}</b>", showarrow=False,
                               font=dict(size=13, color="#1e3a8a"))
        _body(0.3, 3.7, "#eef2f7", driver_label or "Driver")
        _body(6.3, 9.7, "#eef2f7", driven_label or "Driven")
        # Coupling
        fig.add_shape(type="rect", x0=4.4, y0=-0.18, x1=5.6, y1=0.18,
                      line=dict(color="#92400e", width=1), fillcolor="#b45309")
        # Eje
        fig.add_shape(type="line", x0=3.7, y0=0, x1=6.3, y1=0,
                      line=dict(color="#334155", width=3))

        # Posiciones de sensores: driver bearings 1,2 / driven 3,4
        # X aproximado de cada bearing
        bx = {"1": 1.3, "2": 3.2, "3": 6.8, "4": 8.7}
        cmap = {"Danger": "#E24B4A", "Alarma": "#EF9F27", "Normal": "#1D9E75"}
        placed = {}
        for c in channels:
            sl = c.get("sensor_label", "")
            num = sl[0] if sl and sl[0].isdigit() else None
            if num not in bx:
                continue
            x = bx[num]
            # alternar arriba/abajo por dirección X/Y para no encimar
            updown = 1 if ("Y" in sl.upper()) else -1
            key = (num, updown)
            y = updown * (0.55 + 0.32 * (placed.get(key, 0) + 1))
            placed[key] = placed.get(key, 0) + 1
            color = cmap.get(c.get("status", "Normal"), "#1D9E75")
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                marker=dict(size=15, color=color, line=dict(color="white", width=2)),
                text=[f"{sl.replace('_','')}"], textposition="middle right",
                textfont=dict(size=9, color="#0f172a", family="monospace"),
                hoverinfo="skip", showlegend=False,
            ))
            # línea fina del dot al eje del bearing
            fig.add_shape(type="line", x0=x, y0=updown*0.55, x1=x, y1=y,
                          line=dict(color="#cbd5e1", width=1))

        fig.update_layout(
            height=240, width=1000, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(visible=False, range=[-0.2, 10.6]),
            yaxis=dict(visible=False, range=[-2.2, 2.2]),
            showlegend=False,
        )
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None
