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
    trend_title: str = "Tendencia overall",
    schematic_png: Optional[bytes] = None,
    train_drawing: Any = None,
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
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.4 * cm, rightMargin=1.4 * cm,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm,
        title=f"Reporte {instance_id}", author="Watermelon System · SIGASAS",
    )
    story: List[Any] = []

    # Tipografía premium (misma familia del briefing: DejaVuSans vía shell) —
    # glifos unicode correctos (●, ▲, ▼, ·) y look consistente. v2 pulido.
    from core.report_pdf_shell import REGULAR as SANS, BOLD as SANSB
    mono = SANS  # números alineados a la derecha (sin typewriter Courier)
    st_title = ParagraphStyle("t", fontName=SANSB, fontSize=18,
                              textColor=colors.HexColor(_NAVY), spaceAfter=2, leading=21)
    st_sub = ParagraphStyle("s", fontName=SANS, fontSize=10,
                            textColor=colors.HexColor(_SLATE), spaceAfter=1, leading=13)
    st_meta = ParagraphStyle("m", fontName=SANS, fontSize=8.2,
                             textColor=colors.HexColor(_MUTE), leading=11)
    st_section = ParagraphStyle("sec", fontName=SANSB, fontSize=10.5,
                                textColor=colors.HexColor(_NAVY), spaceBefore=8,
                                spaceAfter=5, leading=13)
    st_cell = ParagraphStyle("c", fontName=SANS, fontSize=9,
                             textColor=colors.HexColor(_NAVY))
    st_cellnum = ParagraphStyle("cn", fontName=SANS, fontSize=9,
                                textColor=colors.HexColor(_NAVY), alignment=TA_RIGHT)
    # Estilo chico para la columna "Ubicación": envuelve el texto en varias
    # líneas DENTRO de la celda en vez de desbordarse sobre "Overall".
    st_loc = ParagraphStyle("loc", fontName=SANS, fontSize=7.6,
                            leading=8.8, textColor=colors.HexColor(_SLATE))

    # ---------- Header ----------
    logo_cell = ""
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "watermelon_logo.png"
    header_left = []
    tag = getattr(instance_obj, "tag", "") or instance_id
    driver = " ".join(p for p in [getattr(instance_obj, "driver_manufacturer", ""),
                                  getattr(instance_obj, "driver_model", "")] if p)
    driven = " ".join(p for p in [getattr(instance_obj, "driven_manufacturer", ""),
                                  getattr(instance_obj, "driven_model", "")] if p)
    train = f"{driver} → {driven}" if driver and driven else (driver or driven or "")
    client = getattr(instance_obj, "client", "") or ""
    site = getattr(instance_obj, "site", "") or getattr(instance_obj, "location", "") or ""
    # Ciclo 23.157 — Hora LOCAL del cliente (America/Bogota), no UTC del
    # servidor. El reporte llegaba 2:00 AM mostrando "07:00" → desconfianza.
    try:
        from zoneinfo import ZoneInfo
        now_txt = datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        now_txt = datetime.now().strftime("%Y-%m-%d %H:%M")

    header_left.append(Paragraph(f"{tag} — Reporte de condición", st_title))
    header_left.append(Paragraph(train or "—", st_sub))
    sub2 = " · ".join(p for p in [client, site] if p)
    if sub2:
        header_left.append(Paragraph(sub2, st_sub))
    header_left.append(Paragraph(f"Generado {now_txt} · ISO 20816 / API 670", st_meta))

    # Color de acento por estado (para la línea superior + chip de estado).
    _hs = str(kpis.get("status", "") or "")
    if kpis.get("offline"):
        _acc, _acc_bg, _acc_label = "#475569", "#f1f5f9", "FUERA DE LÍNEA"
    else:
        _acc, _acc_bg = _sev_colors(_hs)
        _sll = _hs.lower()
        _acc_label = ("PELIGRO" if ("crít" in _sll or "crit" in _sll)
                      else "ALARMA" if ("atenci" in _sll or "alarm" in _sll)
                      else "NORMAL")

    if logo_path.exists():
        try:
            logo_cell = Image(str(logo_path), width=3.6 * cm, height=3.6 * cm * 0.494)
            logo_cell.hAlign = "RIGHT"
        except Exception:
            logo_cell = ""

    # Chip de estado (pill) a la derecha, bajo el logo.
    st_pill = ParagraphStyle("pill", fontName=SANSB, fontSize=9.5,
                             textColor=colors.HexColor(_acc), alignment=TA_CENTER,
                             leading=12)
    pill = Table([[Paragraph(f"● {_acc_label}", st_pill)]], colWidths=[3.6 * cm])
    pill.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(_acc_bg)),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    pill.hAlign = "RIGHT"
    right_cell = [logo_cell, Spacer(1, 6), pill] if logo_cell else [pill]

    htbl = Table([[header_left, right_cell]], colWidths=[12.5 * cm, 5.5 * cm])
    htbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(htbl)
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1.6, color=colors.HexColor(_acc)))
    story.append(Spacer(1, 9))

    # ---------- KPIs ----------
    score = health.get("score")
    score_txt = str(score) if score is not None else "—"
    zone = health.get("zone", "—")
    hcolor = colors.HexColor(health.get("color", _MUTE))

    def _kpi(label, value, vcolor=None):
        lab = ParagraphStyle("kl", fontName=SANSB, fontSize=7.3,
                             textColor=colors.HexColor(_MUTE), leading=9, spaceAfter=3)
        val = ParagraphStyle("kv", fontName=SANSB, fontSize=14.5,
                             textColor=vcolor or colors.HexColor(_NAVY), leading=17)
        return [Paragraph(label.upper(), lab), Paragraph(str(value), val)]

    # En OFFLINE, la velocidad es el ÚLTIMO dato conocido (no la actual, que
    # se desconoce). Se rotula "Velocidad (último)" para no leerla como en vivo.
    _off = bool(kpis.get("offline"))
    _vel_label = "Velocidad (último)" if _off else "Velocidad"
    kpi_tbl = Table([[
        _kpi("Salud", f"{score_txt}", hcolor),
        _kpi("Estado", _acc_label.capitalize(), colors.HexColor(_acc)),
        _kpi(_vel_label, kpis.get("speed", "—")),
        _kpi("Alarmas", kpis.get("alarms", 0),
             colors.HexColor(_RED) if kpis.get("alarms", 0) else colors.HexColor(_GREEN)),
        _kpi("Última lectura", kpis.get("last", "—")),
    ]], colWidths=[3.44 * cm] * 5)
    kpi_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(_LINE)),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor(_LINE)),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(kpi_tbl)
    _zline = f"Zona ISO: {zone}"
    _spall = kpis.get("speeds_all")
    if _spall and _spall not in ("—", kpis.get("speed")) and "·" in _spall:
        _vl = "Velocidades (último dato)" if _off else "Velocidades"
        _zline += f"  ·  {_vl}: {_spall}"   # tren turbo-gen: turbina + generador
    story.append(Paragraph(_zline, st_meta))
    story.append(Spacer(1, 8))

    # ---------- Banner de estado + diagnóstico en lenguaje simple ----------
    _status = str(kpis.get("status", "—"))
    _sl = _status.lower()
    _fg, _bg = _sev_colors(_status)
    _alarms = kpis.get("alarms", 0) or 0
    _offline = bool(kpis.get("offline"))
    if _offline:
        # FUERA DE LÍNEA — banner gris, honesto. Muestra los ÚLTIMOS datos
        # medidos (no simula condición actual). Para activos con servicio
        # contratado que están parados o sin enlace.
        _fg, _bg = "#475569", "#f1f5f9"
        _since = kpis.get("offline_since") or ""
        _age = kpis.get("offline_age") or ""
        _band = "FUERA DE LÍNEA — SIN COMUNICACIÓN"
        _diag = ("Equipo sin reportar al monitoreo en línea"
                 + (f" desde {_since}" if _since else "")
                 + (f" ({_age})" if _age else "")
                 + ". A continuación se muestran los ÚLTIMOS datos válidos "
                 "medidos; NO representan la condición actual del equipo. "
                 "Acción: verificar operación del equipo, el colector y el "
                 "enlace del sitio.")
    elif "crít" in _sl or "crit" in _sl:
        _band = "CONDICIÓN CRÍTICA — ACCIÓN INMEDIATA"
        _diag = (f"{_alarms} punto(s) en nivel de PELIGRO. Requiere atención inmediata: "
                 "inspeccionar el equipo y evaluar parada según criticidad y contexto operativo.")
    elif "atenci" in _sl or "alarm" in _sl:
        _band = "ATENCIÓN — SEGUIMIENTO"
        _diag = (f"{_alarms} punto(s) en ALARMA. Programar revisión y vigilar la tendencia. "
                 "Aún no exige parada, pero no debe ignorarse.")
    else:
        _band = "OPERACIÓN NORMAL"
        _diag = (f"El activo opera dentro de parámetros normales (ISO 20816, {zone}). "
                 "Sin acciones requeridas; continuar el monitoreo en línea.")
    st_band = ParagraphStyle("bd", fontName=SANSB, fontSize=12.5,
                             textColor=colors.HexColor(_fg), leading=15)
    st_diag = ParagraphStyle("dg", fontName=SANS, fontSize=9.5,
                             textColor=colors.HexColor(_NAVY), leading=13)
    band_tbl = Table([[Paragraph(_band, st_band)], [Paragraph(_diag, st_diag)]],
                     colWidths=[17.2 * cm])
    band_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(_bg)),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(_fg)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (0, 0), 9),
        ("BOTTOMPADDING", (0, 0), (0, 0), 1),
        ("TOPPADDING", (0, 1), (-1, 1), 1),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
    ]))
    story.append(band_tbl)
    story.append(Spacer(1, 10))

    # ---------- Esquemático del tren (HÉROE) — sensores por severidad ----------
    # Preferimos el DIAGRAMA LIVE (vector, siluetas realistas + barras de umbral,
    # vía svglib) — el mismo de la web. Fallback: PNG matplotlib.
    _train_title = ("Estado del tren — ÚLTIMO estado conocido" if _offline
                    else "Estado del tren — sensores por severidad")
    if train_drawing is not None:
        try:
            story.append(Paragraph(_train_title, st_section))
            try:
                train_drawing.hAlign = "CENTER"
            except Exception:  # noqa: BLE001
                pass
            story.append(train_drawing)
            story.append(Spacer(1, 10))
        except Exception:  # noqa: BLE001
            pass
    elif schematic_png:
        try:
            from reportlab.lib.utils import ImageReader
            _ir = ImageReader(BytesIO(schematic_png))
            _iw, _ih = _ir.getSize()
            _w = 17.4 * cm
            _h = _w * _ih / _iw
            if _h > 8.6 * cm:                 # cap alto (landscape); recentra
                _h = 8.6 * cm; _w = _h * _iw / _ih
            story.append(Paragraph("Estado del tren — sensores coloreados por severidad", st_section))
            _simg = Image(BytesIO(schematic_png), width=_w, height=_h)
            _simg.hAlign = "CENTER"
            story.append(_simg)
            story.append(Spacer(1, 10))
        except Exception:
            pass

    # ---------- Tendencia (si hay PNG) ----------
    if trend_png:
        try:
            story.append(Paragraph(trend_title, st_section))
            img = Image(BytesIO(trend_png), width=18 * cm, height=5.3 * cm)
            story.append(img)
        except Exception:
            pass

    # ---------- Tabla de canales ----------
    # El reporte es de UNA hoja (gerencial). Si hay muchos canales, se
    # priorizan por severidad (Danger > Alarma > Normal) y se muestran los
    # 10 más críticos, con nota al pie indicando cuántos quedaron fuera.
    _MAX_CH = 24

    def _sev_rank(c):
        s = (c.get("status") or "").lower()
        if "danger" in s or "crít" in s or "crit" in s:
            return 0
        if "alarma" in s or "alert" in s:
            return 1
        return 2

    total_ch = len(channels)
    if total_ch > _MAX_CH:
        channels = sorted(channels, key=_sev_rank)[:_MAX_CH]
        ch_truncated = total_ch - _MAX_CH
    else:
        ch_truncated = 0

    _ch_title = Paragraph("Canales — Overall + vectores 1X / 2X (API 670)", st_section)
    head = ["Estado", "Canal", "Ubicación", "Overall", "Unit", "1X ampl", "1X °", "2X ampl", "2X °"]
    st_estado = ParagraphStyle("est", fontName=SANSB, fontSize=8.2,
                               textColor=colors.HexColor(_NAVY), leading=10)
    data = [head]
    for c in channels:
        fg, _bg = _sev_colors(c.get("status", ""))
        _stt = c.get("status") or "—"
        data.append([
            Paragraph(f'<font color="{fg}">●</font>&nbsp;{_stt}', st_estado),
            c.get("sensor_label", "—"),
            Paragraph(str(c.get("plane_label") or "—"), st_loc),
            c.get("value", "—"),
            c.get("unit", ""),
            c.get("x1_amp", "—"),
            c.get("x1_ph", "—"),
            c.get("x2_amp", "—"),
            c.get("x2_ph", "—"),
        ])

    ctbl = Table(data, colWidths=[1.9*cm, 1.4*cm, 3.5*cm, 1.8*cm, 1.4*cm,
                                  1.6*cm, 1.2*cm, 1.6*cm, 1.2*cm])
    base_style = [
        # Header: fondo navy suave + texto claro (más definido)
        ("FONTNAME", (0, 0), (-1, 0), SANSB),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_NAVY)),
        ("TOPPADDING", (0, 0), (-1, 0), 5), ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        # Cuerpo
        ("FONTNAME", (1, 1), (-1, -1), SANS),
        ("FONTSIZE", (1, 1), (-1, -1), 9),
        ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (2, -1), "LEFT"),
        ("TOPPADDING", (0, 1), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # Zebra + hairline inferior
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(_LINE)),
        ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white),
    ]
    ctbl.setStyle(TableStyle(base_style))
    story.append(KeepTogether([_ch_title, ctbl]))
    meta_txt = "1X = componente síncrona (desbalance) · 2X = segunda armónica (desalineamiento / soltura)"
    if ch_truncated:
        meta_txt += (f" · Mostrando {_MAX_CH} de {total_ch} canales "
                     f"(priorizados por severidad)")
    story.append(Paragraph(meta_txt, st_meta))

    # ---------- Eventos ----------
    if events:
        story.append(Paragraph("Registro de eventos — cruces de umbral", st_section))
        ev_data = [["", "Canal", "Estado", "Valor", "Hace"]]
        ev_styles = []
        for i, e in enumerate(events[:6], start=1):
            arrow = "▲" if e.get("rising") else "▼"
            acolor = _RED if e.get("rising") else _GREEN
            ev_data.append([arrow, e.get("sensor_label", "—"), e.get("to", "—"),
                            f"{e.get('value','—')} {e.get('unit','')}", e.get("age", "—")])
            ev_styles.append(("TEXTCOLOR", (0, i), (0, i), colors.HexColor(acolor)))
        etbl = Table(ev_data, colWidths=[0.7*cm, 2.0*cm, 2.2*cm, 3.5*cm, 2.5*cm])
        etbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), SANSB),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_NAVY)),
            ("FONTNAME", (1, 1), (-1, -1), SANS),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f7f9fc")]),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(_LINE)),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ] + ev_styles))
        story.append(etbl)

    # ---------- Footer ----------
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor(_LINE)))
    foot = ParagraphStyle("f", fontName=SANS, fontSize=8,
                          textColor=colors.HexColor(_MUTE), alignment=TA_CENTER)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Generado por Watermelon System · SIGASAS · Monitoreo de condición de maquinaria rotativa · "
        "ISO 20816-3 / API 670", foot))
    story.append(Paragraph(
        "Documento confidencial — uso exclusivo del cliente. Reporte automático de condición; "
        "ante decisiones críticas, validar con el especialista.", foot))

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
                          annotation_text="Danger", annotation_position="top left",
                          annotation=dict(font=dict(color="#dc2626", size=9), xshift=-44))
        if alarm > 0:
            fig.add_hline(y=alarm, line=dict(color="#d97706", width=1.2, dash="dash"),
                          annotation_text="Alarma", annotation_position="top left",
                          annotation=dict(font=dict(color="#d97706", size=9), xshift=-44))
        # Ciclo 23.157 — Eje de tiempo en hora LOCAL (America/Bogota).
        # Los captured_at vienen en UTC desde Supabase.
        def _to_local(xs):
            from datetime import timezone as _tz
            try:
                from zoneinfo import ZoneInfo
                _bog = ZoneInfo("America/Bogota")
            except Exception:
                return list(xs or [])
            out = []
            for v in xs or []:
                try:
                    dt = (datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                          if not isinstance(v, datetime) else v)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_tz.utc)
                    out.append(dt.astimezone(_bog).replace(tzinfo=None))
                except Exception:
                    out.append(v)
            return out

        for s in sensor_series:
            fig.add_trace(go.Scatter(
                x=_to_local(s.get("x", [])), y=s.get("y", []), mode="lines",
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


