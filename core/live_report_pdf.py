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
    if "alarma" in s or "alert" in s or "atenci" in s or "attention" in s:
        return _AMBER, _AMBER_BG
    return _GREEN, _GREEN_BG


def _canon_status(status: Any, alarms: Any) -> str:
    """Estado canónico en ESPAÑOL (el reporte es del cliente). Acepta el inglés de
    la UI ('Attention', 'Critical') y nunca dice 'Normal' si hay alarmas."""
    s = str(status or "").strip()
    sl = s.lower()
    if "fuera" in sl or "offline" in sl:
        return "Fuera de línea"
    if "detenid" in sl or "stopped" in sl:
        return "Máquina detenida"
    if "crít" in sl or "crit" in sl or "danger" in sl or "peligro" in sl:
        return "Crítica"
    if "atenci" in sl or "attention" in sl or "alarm" in sl or "alert" in sl:
        return "Atención"
    try:
        if int(alarms or 0) > 0:
            return "Atención"
    except (TypeError, ValueError):
        pass
    return "Operación normal"


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
    watch: Optional[List[Dict[str, Any]]] = None,
    watch_png: Optional[bytes] = None,
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

    # Estado canónico (español) — la descarga desde Live pasaba 'Attention' en
    # inglés y el PDF lo leía como 'Normal' aunque hubiera alarmas.
    if not kpis.get("offline") and not kpis.get("stopped"):
        kpis = dict(kpis, status=_canon_status(kpis.get("status"), kpis.get("alarms")))

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
    if kpis.get("machine_type"):          # Machines = fuente única del tipo
        train = str(kpis["machine_type"])
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
    zone = str(health.get("zone", "—")).replace("Zone ", "Zona ").replace("No data", "Sin datos")
    hcolor = colors.HexColor(health.get("color", _MUTE))

    # Banda de datos = la MISMA de Live Monitoring (navy + franja ámbar):
    # salud, estado, turbina, generador, vibración de eje máx, alarmas, última lectura.
    import re as _re
    _off = bool(kpis.get("offline"))
    _spall = str(kpis.get("speeds_all") or "")
    _sp_turb, _sp_gen = str(kpis.get("speed", "—")).replace(" rpm", ""), None
    for _part in _spall.replace(" rpm", "").split("·"):
        _m = _re.search(r"([A-Za-zÁÉÍÓÚáéíóú ]+)\s+([\d.,]+)", _part.strip())
        if _m and _m.group(1).strip().lower().startswith("gen"):
            _sp_gen = _m.group(2)
        elif _m and _m.group(1).strip().lower().startswith("turb"):
            _sp_turb = _m.group(2)
    _maxv, _maxl, _maxu = None, "", "µm pp"
    for _c in channels or []:
        _u = str(_c.get("unit") or "").lower()
        if "µm" in _u or "um" in _u or "mil" in _u:
            try:
                _v = float(_c.get("value"))
            except (TypeError, ValueError):
                continue
            if _maxv is None or _v > _maxv:
                _maxv, _maxl = _v, str(_c.get("sensor_label") or "").replace("_", "")
                _maxu = (_c.get("unit") or "µm pp")
    _st_lab = ParagraphStyle("bl", fontName=SANSB, fontSize=6.6, leading=8,
                             textColor=colors.HexColor("#9fc0e6"))

    def _bk(label, value, unit="", vcolor="#ffffff", size=12.5):
        _v = ParagraphStyle("bv", fontName=SANSB, fontSize=size, leading=size + 3,
                            textColor=colors.HexColor(vcolor))
        _txt = f"{value}<font size='6.5' color='#9fc0e6'>&nbsp;{unit}</font>" if unit else f"{value}"
        return [Paragraph(_txt, _v), Spacer(1, 2), Paragraph(label.upper(), _st_lab)]

    _hc = health.get("color", _MUTE)
    _alarms_n = kpis.get("alarms", 0) or 0
    _st_txt = str(kpis.get("status", "—"))
    _st_col, _ = _sev_colors(_st_txt)
    if _off or kpis.get("stopped"):
        _st_col = "#cbd5e1"
    cells = [
        _bk("Salud · " + str(zone).split("·")[0].strip(), score_txt, "/100", _hc),
        _bk("Estado", _st_txt, "", _st_col, 10.5),
        _bk("Turbina (último)" if _off else "Turbina", _sp_turb, "rpm"),
    ]
    widths = [2.5, 2.7, 2.7]
    if _sp_gen:
        cells.append(_bk("Generador", _sp_gen, "rpm")); widths.append(2.5)
    cells.append(_bk("Vib. eje máx" + (f" · {_maxl}" if _maxl else ""),
                     f"{_maxv:.1f}" if _maxv is not None else "—",
                     _maxu.replace(" pp", "") + " pp" if _maxv is not None else ""))
    widths.append(2.9)
    cells.append(_bk("Alarmas", _alarms_n, "", _RED if _alarms_n else "#4ade80"))
    widths.append(1.95)
    cells.append(_bk("Última lectura", str(kpis.get("last", "—")).replace("hace ", ""), "", "#ffffff", 11))
    widths.append(18.0 - sum(widths))
    kpi_tbl = Table([cells], colWidths=[w * cm for w in widths])
    kpi_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#12305e")),
        ("LINEAFTER", (-1, 0), (-1, -1), 4, colors.HexColor("#e8890c")),
        ("LINEAFTER", (0, 0), (-2, -1), 0.4, colors.HexColor("#2b4a7a")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 3))
    story.append(Paragraph(f"Zona ISO 20816: {zone}", st_meta))
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
    elif bool(kpis.get("stopped")):
        # MÁQUINA DETENIDA — telemetría en línea pero RPM≈0. Datos ACTUALES,
        # con la máquina parada: no es "operación normal" ni condición evaluable.
        _fg, _bg = "#475569", "#f1f5f9"
        try:
            _rpm_txt = f"{float(kpis.get('stopped_rpm')):.0f} rpm"
        except Exception:
            _rpm_txt = "≈0 rpm"
        _band = "MÁQUINA DETENIDA — FUERA DE OPERACIÓN"
        _diag = (f"El colector está en línea, pero la máquina está detenida "
                 f"(velocidad {_rpm_txt}). Los valores mostrados son los actuales "
                 "con la máquina parada; NO representan su condición en operación "
                 "y la salud no aplica a 0 rpm. Al arrancar, confirmar la vibración "
                 "en régimen antes de emitir diagnóstico.")
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
        _w0 = (watch or [None])[0]
        if _w0 and (_w0.get("pct") or 0) >= 90:
            _diag += (f" <b>Punto a vigilar:</b> {_w0.get('display') or _w0.get('sensor_label')} "
                      f"está al {_w0['pct']:.0f} % de su nivel de alerta.")
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

    # ---------- Puntos a vigilar (aviso temprano) ----------
    if watch:
        try:
            st_wt = ParagraphStyle("wt", fontName=SANS, fontSize=8.6, leading=10.5,
                                   textColor=colors.HexColor(_NAVY))
            wdata = [["Sonda", "Ubicación", "Valor", "Alerta", "% de alerta", "Δ 24 h"]]
            wstyles = []
            for i, w in enumerate(watch, start=1):
                pct = w.get("pct") or 0
                pc = _RED if pct >= 100 else (_AMBER if pct >= 90 else _GREEN)
                bar_w = 2.6 * min(pct, 100) / 100.0
                bar = Table([["", ""]], colWidths=[max(0.05, bar_w) * cm, max(0.0, 2.6 - bar_w) * cm],
                            rowHeights=[0.22 * cm])
                bar.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(pc)),
                    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#e6ebf2")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
                d = w.get("delta")
                dtxt = "—" if d is None else f"{d:+.1f} %"
                dcol = _MUTE if d is None else (_RED if d >= 10 else (_GREEN if d <= -10 else "#475569"))
                loc = w.get("aux_group") or w.get("plane_label") or "—"
                wdata.append([
                    Paragraph(f"<b>{w.get('display') or w.get('sensor_label')}</b>", st_wt),
                    Paragraph(str(loc), st_loc),
                    f"{w.get('value', '—')} {w.get('unit', '')}",
                    f"{w.get('alarm'):g}" if w.get("alarm") else "—",
                    Table([[Paragraph(f'<font color="{pc}"><b>{pct:.0f} %</b></font>', st_wt), bar]],
                          colWidths=[1.3 * cm, 2.7 * cm],
                          style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                            ("LEFTPADDING", (0, 0), (-1, -1), 0)])),
                    Paragraph(f'<font color="{dcol}">{dtxt}</font>', st_wt),
                ])
            wtbl = Table(wdata, colWidths=[1.8 * cm, 4.6 * cm, 2.6 * cm, 1.6 * cm, 4.4 * cm, 1.9 * cm])
            wtbl.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), SANSB), ("FONTSIZE", (0, 0), (-1, -1), 8.6),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_NAVY)),
                ("FONTNAME", (0, 1), (-1, -1), SANS),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(_LINE)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ] + wstyles))
            blk = [Paragraph("Puntos a vigilar — más cerca de su nivel de alerta", st_section), wtbl]
            if watch_png:
                blk += [Spacer(1, 4),
                        Paragraph(f"{watch[0].get('display') or watch[0].get('sensor_label')} — últimas 24 h "
                                  f"(máx por hora) con sus niveles de alerta y peligro", st_meta),
                        Image(BytesIO(watch_png), width=18 * cm, height=4.2 * cm)]
            story.append(KeepTogether(blk))
            story.append(Spacer(1, 8))
        except Exception:  # noqa: BLE001
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
    head = ["Estado", "Canal", "Ubicación", "Overall", "Unidad", "% alerta", "Δ 24 h",
            "1X ampl", "1X °", "2X ampl", "2X °"]
    st_estado = ParagraphStyle("est", fontName=SANSB, fontSize=7.6,
                               textColor=colors.HexColor(_NAVY), leading=9)
    st_grp = ParagraphStyle("grp", fontName=SANSB, fontSize=8,
                            textColor=colors.HexColor("#12305e"), leading=10)
    _ES = {"normal": "Normal", "alarma": "Alarma", "danger": "Peligro",
           "sin norma": "Sin norma", "no data": "Sin dato"}
    data = [head]
    extra_styles = []
    _cur_grp = None
    for c in channels:
        _grp = c.get("aux_group") or ""
        if _grp and _grp != _cur_grp:
            _cur_grp = _grp
            data.append([Paragraph(f"AUXILIAR · {_grp}", st_grp)] + [""] * (len(head) - 1))
            r = len(data) - 1
            extra_styles += [("SPAN", (0, r), (-1, r)),
                             ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef3f9")),
                             ("TOPPADDING", (0, r), (-1, r), 3), ("BOTTOMPADDING", (0, r), (-1, r), 3)]
        fg, _bg = _sev_colors(c.get("status", ""))
        _stt = _ES.get(str(c.get("status") or "").lower(), c.get("status") or "—")
        pct = c.get("pct")
        pcol = _MUTE if pct is None else (_RED if pct >= 100 else (_AMBER if pct >= 90 else "#475569"))
        d = c.get("delta")
        dcol = _MUTE if d is None else (_RED if d >= 10 else (_GREEN if d <= -10 else "#475569"))
        data.append([
            Paragraph(f'<font color="{fg}">●</font>&nbsp;{_stt}', st_estado),
            c.get("display") or str(c.get("sensor_label", "—")).replace("_", ""),
            Paragraph(str(_grp or c.get("plane_label") or "—"), st_loc),
            c.get("value", "—"),
            Paragraph(str(c.get("unit", "")), st_loc),
            Paragraph(f'<font color="{pcol}">{"—" if pct is None else f"{pct:.0f} %"}</font>', st_cellnum),
            Paragraph(f'<font color="{dcol}">{"—" if d is None else f"{d:+.0f} %"}</font>', st_cellnum),
            c.get("x1_amp", "—"),
            c.get("x1_ph", "—"),
            c.get("x2_amp", "—"),
            c.get("x2_ph", "—"),
        ])

    ctbl = Table(data, colWidths=[1.75*cm, 1.15*cm, 2.75*cm, 1.45*cm, 1.55*cm,
                                  1.45*cm, 1.35*cm, 1.45*cm, 1.05*cm, 1.45*cm, 1.05*cm])
    base_style = [
        # Header: fondo navy suave + texto claro (más definido)
        ("FONTNAME", (0, 0), (-1, 0), SANSB),
        ("FONTSIZE", (0, 0), (-1, 0), 7.6),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_NAVY)),
        ("TOPPADDING", (0, 0), (-1, 0), 5), ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        # Cuerpo
        ("FONTNAME", (1, 1), (-1, -1), SANS),
        ("FONTSIZE", (1, 1), (-1, -1), 8.6),
        ("ALIGN", (3, 0), (3, -1), "RIGHT"),
        ("ALIGN", (5, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (2, -1), "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # Zebra + hairline inferior
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(_LINE)),
        ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white),
    ] + extra_styles
    ctbl.setStyle(TableStyle(base_style))
    story.append(KeepTogether([_ch_title, ctbl]))
    meta_txt = ("% alerta = valor / nivel de alerta · Δ 24 h = última hora vs la misma hora de ayer · "
                "1X = componente síncrona (desbalance) · 2X = segunda armónica (desalineamiento / soltura)")
    if ch_truncated:
        meta_txt += (f" · Mostrando {_MAX_CH} de {total_ch} canales "
                     f"(priorizados por severidad)")
    story.append(Paragraph(meta_txt, st_meta))

    # ---------- Alarmas del día (últimas 24 h) ----------
    story.append(Paragraph("Alarmas del día — últimas 24 h", st_section))
    if not events:
        _ok_style = ParagraphStyle("evok", fontName=SANS, fontSize=9.5,
                                   textColor=colors.HexColor(_GREEN))
        story.append(Paragraph("● Sin alarmas en las últimas 24 h.", _ok_style))
    else:
        # Resumen POR SONDA (antes: una fila por cada cruce → una sonda oscilando en
        # su límite llenaba una página entera). Entradas a alarma/peligro, valor
        # máximo y cuándo fue la última.
        _agg: Dict[str, Dict[str, Any]] = {}
        _order: List[str] = []
        for e in events:
            k = str(e.get("sensor_label", "—")).replace("_", "")
            if k not in _agg:
                _agg[k] = {"n": 0, "max": None, "unit": e.get("unit", ""), "last": None,
                           "worst": "Alarma", "now": None}
                _order.append(k)
            a = _agg[k]
            if a["now"] is None:
                a["now"] = e.get("to")
            if e.get("rising"):
                a["n"] += 1
                if a["last"] is None:
                    a["last"] = e.get("age")
                if e.get("to") == "Danger":
                    a["worst"] = "Peligro"
            try:
                v = float(e.get("value"))
                a["max"] = v if a["max"] is None or v > a["max"] else a["max"]
            except (TypeError, ValueError):
                pass
        ev_data = [["", "Sonda", "Entradas a alarma", "Máximo", "Última", "Ahora"]]
        ev_styles = []
        for i, k in enumerate(_order[:6], start=1):
            a = _agg[k]
            col = _RED if a["worst"] == "Peligro" else _AMBER
            _now = {"Danger": "Peligro", "Alarma": "Alarma"}.get(a["now"], "Normal")
            ev_data.append(["▲", k, f"{a['n']} ({a['worst'].lower()})",
                            f"{a['max']:.2f} {a['unit']}" if a["max"] is not None else "—",
                            f"hace {a['last']}" if a["last"] else "—", _now])
            ev_styles.append(("TEXTCOLOR", (0, i), (0, i), colors.HexColor(col)))
            ev_styles.append(("TEXTCOLOR", (5, i), (5, i),
                              colors.HexColor(_GREEN if _now == "Normal" else col)))
        etbl = Table(ev_data, colWidths=[0.7*cm, 1.8*cm, 3.6*cm, 3.2*cm, 2.6*cm, 2.0*cm])
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


