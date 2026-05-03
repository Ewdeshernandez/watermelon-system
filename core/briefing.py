"""
core.briefing
=============

Generador de **briefing diario** del sistema (Ciclo 17.13 — Nivel 3).

Produce un PDF de UNA página con el estado de la flota para que el
operador (o el cliente final) lo reciba todas las mañanas y sepa
en 30 segundos qué activo mirar primero.

Layout 1 página A4:

  ┌──────────────────────────────────────────────────────────┐
  │  WATERMELON · BRIEFING DIARIO          domingo 03 may   │
  │  4 activos · 0 críticos · 1 atención · 3 OK             │
  ├──────────────────────────────────────────────────────────┤
  │  TOP 3 ACTIVOS QUE REQUIEREN ATENCIÓN                   │
  │  ⚖ TES1   65/100  ATENCIÓN   sin baseline reciente      │
  │  ⚖ ...    ...                                           │
  ├──────────────────────────────────────────────────────────┤
  │  CAMBIOS VS AYER                                        │
  │  ↓ TES1 bajó de OK a ATENCIÓN (último análisis)         │
  │  • C-200C sin cambios                                   │
  ├──────────────────────────────────────────────────────────┤
  │  PRÓXIMOS VENCIMIENTOS                                  │
  │  ⏰ 2 baselines vencidos (>90 días)                      │
  │  ⏰ 1 norma pendiente de asignar                         │
  ├──────────────────────────────────────────────────────────┤
  │  Próximos pasos sugeridos: ...                          │
  │  Generado por Watermelon System · v3.1.5 · SIGASAS      │
  └──────────────────────────────────────────────────────────┘

Snapshot diario:
  Cada vez que se genera el briefing, se guarda un JSON en
  data/briefings/snapshots/YYYY-MM-DD.json con el fleet status
  del día. El briefing del día siguiente compara contra el
  snapshot más reciente para mostrar deltas.

Output:
  data/briefings/briefing_YYYY-MM-DD.pdf
  Y devuelve los bytes del PDF para que el caller lo entregue
  al usuario (download_button del Home, attach a email, etc.).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BRIEFINGS_DIR = DATA_DIR / "briefings"
SNAPSHOTS_DIR = BRIEFINGS_DIR / "snapshots"


def _ensure_dirs():
    BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================
# SNAPSHOTS — guardar / cargar estado de la flota por día
# =============================================================

def save_daily_snapshot(snapshot: Dict[str, Any], date: Optional[datetime] = None) -> Path:
    """Persiste un snapshot del fleet status del día indicado.
    Si ya existe uno para esa fecha, lo sobreescribe (idempotente).
    """
    _ensure_dirs()
    d = date or datetime.now()
    fname = f"{d.strftime('%Y-%m-%d')}.json"
    path = SNAPSHOTS_DIR / fname
    payload = {
        "date": d.strftime("%Y-%m-%d"),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        **snapshot,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path


def load_yesterday_snapshot(today: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Carga el snapshot del día anterior si existe. Si no hay, busca
    el más reciente que sea anterior a hoy.
    """
    if not SNAPSHOTS_DIR.exists():
        return None
    today = today or datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    candidates = sorted(
        [p for p in SNAPSHOTS_DIR.glob("*.json") if p.stem < today_str],
        reverse=True,
    )
    if not candidates:
        return None
    try:
        return json.loads(candidates[0].read_text(encoding="utf-8"))
    except Exception:
        return None


# =============================================================
# COMPUTE — armado del payload del briefing
# =============================================================

def compute_fleet_snapshot() -> Dict[str, Any]:
    """Captura el estado actual de la flota como dict serializable."""
    from core.home_metrics import compute_fleet_status
    from core.health_score import compute_health_score
    from core.instance_state import get_instance

    fleet = compute_fleet_status()
    instances_payload: List[Dict[str, Any]] = []
    sum_score = 0
    n_score = 0

    for h in fleet["instances"]:
        # Cargar instancia full para health_score real + severidad ejec
        try:
            inst = get_instance(h.instance_id)
        except Exception:
            inst = None

        hs_data = {
            "tag": h.tag,
            "iso_norm_code": getattr(inst, "iso_norm_code", "") if inst else "",
            "last_balance_date": getattr(inst, "last_balance_date", "") if inst else "",
            "documents": getattr(inst, "documents", []) if inst else [],
            "setpoint_warning_override": getattr(inst, "setpoint_warning_override", 0.0) if inst else 0.0,
            "setpoint_danger_override": getattr(inst, "setpoint_danger_override", 0.0) if inst else 0.0,
            "override_justification": getattr(inst, "override_justification", "") if inst else "",
        }
        hs = compute_health_score(hs_data)
        sum_score += hs.score
        n_score += 1

        instances_payload.append({
            "instance_id": h.instance_id,
            "tag": h.tag,
            "asset_class": h.asset_class,
            "location": h.location,
            "severity": h.severity,
            "severity_label": h.severity_label,
            "health_score": hs.score,
            "health_band": hs.band_label,
            "health_color": hs.color,
            "health_one_liner": hs.one_liner,
            "health_breakdown": hs.breakdown,
            "exec_severity": getattr(inst, "last_executive_severity", "") if inst else "",
            "exec_summary": getattr(inst, "last_executive_summary", "") if inst else "",
            "last_report_date": getattr(inst, "last_report_date", "") if inst else "",
            "last_balance_date": getattr(inst, "last_balance_date", "") if inst else "",
            "iso_norm_code": getattr(inst, "iso_norm_code", "") if inst else "",
            "n_documents": h.n_documents,
        })

    avg_score = round(sum_score / n_score) if n_score > 0 else 0

    return {
        "total": fleet["total"],
        "by_severity": fleet["by_severity"],
        "instances": instances_payload,
        "avg_health_score": avg_score,
    }


def compute_deltas_vs_yesterday(today_snap: Dict[str, Any],
                                 yesterday_snap: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compara snapshot de hoy vs el del día anterior. Detecta:
       - Activos cuya banda subió o bajó
       - Activos nuevos / removidos
       - Cambio en avg health score
    """
    if not yesterday_snap:
        return {"has_yesterday": False, "lines": [
            "No hay snapshot del día anterior para comparar."
        ]}

    y_inst = {i.get("instance_id"): i for i in yesterday_snap.get("instances", [])}
    t_inst = {i.get("instance_id"): i for i in today_snap.get("instances", [])}

    lines: List[str] = []

    BAND_RANK = {"OPTIMO":4, "ÓPTIMO":4, "BUENO":3, "ATENCIÓN":2, "ATENCION":2, "CRÍTICO":1, "CRITICO":1}

    # Activos con cambio de banda
    for iid, t in t_inst.items():
        y = y_inst.get(iid)
        if not y:
            lines.append(f"+ Activo nuevo: {t.get('tag','—')}")
            continue
        ts_band = (t.get("health_band") or "").upper()
        ys_band = (y.get("health_band") or "").upper()
        if ts_band != ys_band:
            ar = BAND_RANK.get(ys_band, 0)
            br = BAND_RANK.get(ts_band, 0)
            arrow = "↓" if br < ar else ("↑" if br > ar else "→")
            lines.append(
                f"{arrow} {t.get('tag','—')} pasó de {ys_band} a {ts_band} "
                f"({y.get('health_score',0)} → {t.get('health_score',0)})"
            )

    # Activos removidos
    for iid, y in y_inst.items():
        if iid not in t_inst:
            lines.append(f"− Activo removido: {y.get('tag','—')}")

    # Cambio promedio
    da = today_snap.get("avg_health_score", 0) - yesterday_snap.get("avg_health_score", 0)
    if abs(da) >= 2:
        arrow = "↑" if da > 0 else "↓"
        lines.append(
            f"{arrow} Score promedio de la flota cambió "
            f"{yesterday_snap.get('avg_health_score',0)} → "
            f"{today_snap.get('avg_health_score',0)}"
        )

    if not lines:
        lines = ["Sin cambios significativos desde el último briefing."]

    return {"has_yesterday": True, "lines": lines}


def compute_upcoming_items(snapshot: Dict[str, Any]) -> List[str]:
    """Lista de cosas por vencer / pendientes derivadas del snapshot."""
    items: List[str] = []
    now = datetime.now()
    n_no_norm = 0
    n_baseline_old = 0
    n_no_baseline = 0

    for inst in snapshot.get("instances", []):
        if not (inst.get("iso_norm_code") or "").strip():
            n_no_norm += 1
        last_bal = (inst.get("last_balance_date") or "").strip()
        if not last_bal:
            n_no_baseline += 1
        else:
            try:
                d = datetime.strptime(last_bal[:10], "%Y-%m-%d")
                if (now - d).days > 90:
                    n_baseline_old += 1
            except Exception:
                pass

    if n_no_norm > 0:
        items.append(f"📐 {n_no_norm} activo(s) sin norma ISO/API asignada")
    if n_no_baseline > 0:
        items.append(f"⚖ {n_no_baseline} activo(s) sin baseline de balanceo registrado")
    if n_baseline_old > 0:
        items.append(f"⏰ {n_baseline_old} activo(s) con baseline >90 días (re-balance recomendado)")
    if not items:
        items.append("✓ No hay vencimientos próximos. Flota al día.")
    return items


def suggest_next_actions(snapshot: Dict[str, Any]) -> List[str]:
    """Próximos pasos sugeridos basados en el peor estado."""
    actions: List[str] = []
    by_sev = snapshot.get("by_severity", {})
    if by_sev.get("danger", 0) > 0:
        actions.append(
            f"🚨 Prioridad: revisar los {by_sev['danger']} activo(s) en CRÍTICO. "
            "Ir a Trends y validar el último análisis."
        )
    if by_sev.get("warning", 0) > 0:
        actions.append(
            f"🟡 Después: completar config de los {by_sev['warning']} en ATENCIÓN "
            "(asignar norma, registrar baseline)."
        )
    if not actions:
        actions.append(
            "✅ Flota estable. Buen día para correr análisis preventivos en "
            "los activos saludables."
        )
    return actions


# =============================================================
# PDF — generación con reportlab
# =============================================================

def _es_date_long(d: datetime) -> str:
    DAYS = ["lunes","martes","miércoles","jueves","viernes","sábado","domingo"]
    MONTHS = ["enero","febrero","marzo","abril","mayo","junio",
              "julio","agosto","septiembre","octubre","noviembre","diciembre"]
    return f"{DAYS[d.weekday()]} {d.day} de {MONTHS[d.month-1]} de {d.year}"


def generate_briefing_pdf(snapshot: Optional[Dict[str, Any]] = None,
                           deltas: Optional[Dict[str, Any]] = None,
                           date: Optional[datetime] = None) -> bytes:
    """Genera el PDF de 1 página con el briefing diario.

    Si snapshot/deltas no se pasan, los computa al vuelo (lo cual hace
    también un save_daily_snapshot del día actual).

    Returns:
        bytes del PDF listo para download_button o adjuntar a email.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable,
    )
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER

    d = date or datetime.now()

    if snapshot is None:
        snapshot = compute_fleet_snapshot()
        save_daily_snapshot(snapshot, d)
    if deltas is None:
        deltas = compute_deltas_vs_yesterday(snapshot, load_yesterday_snapshot(d))

    upcoming = compute_upcoming_items(snapshot)
    actions = suggest_next_actions(snapshot)

    # Versión del sistema para el footer
    try:
        from core.version import get_version_short
        _ver = get_version_short()
    except Exception:
        _ver = ""

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.6 * cm, rightMargin=1.6 * cm,
        topMargin=1.4 * cm, bottomMargin=1.2 * cm,
        title=f"Briefing diario · {d.strftime('%Y-%m-%d')}",
    )

    styles = getSampleStyleSheet()
    s_title = ParagraphStyle(
        "WMHeroTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=18, leading=22,
        textColor=colors.HexColor("#0f172a"), spaceAfter=2,
    )
    s_subtitle = ParagraphStyle(
        "WMHeroSubtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10.5, leading=13,
        textColor=colors.HexColor("#475569"), spaceAfter=2,
    )
    s_section = ParagraphStyle(
        "WMSection", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=10, leading=12,
        textColor=colors.HexColor("#0ea5e9"),
        spaceBefore=10, spaceAfter=4,
        textTransform="uppercase", letterSpacing=2,
    )
    s_body = ParagraphStyle(
        "WMBody", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10, leading=14,
        textColor=colors.HexColor("#0f172a"),
    )
    s_muted = ParagraphStyle(
        "WMMuted", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8.5, leading=11,
        textColor=colors.HexColor("#64748b"),
    )

    flow: List[Any] = []

    # ─── HEADER HERO ───
    flow.append(Paragraph(
        f"<b>WATERMELON</b> &nbsp;·&nbsp; Briefing diario", s_title
    ))
    flow.append(Paragraph(_es_date_long(d).capitalize(), s_subtitle))

    # KPIs línea
    by = snapshot.get("by_severity", {})
    kpi_line = (
        f"<b>{snapshot.get('total',0)}</b> activos &nbsp;·&nbsp; "
        f"<font color='#dc2626'><b>{by.get('danger',0)}</b> en crítico</font> &nbsp;·&nbsp; "
        f"<font color='#d97706'><b>{by.get('warning',0)}</b> en atención</font> &nbsp;·&nbsp; "
        f"<font color='#059669'><b>{by.get('healthy',0)}</b> saludables</font> &nbsp;·&nbsp; "
        f"score promedio <b>{snapshot.get('avg_health_score',0)}/100</b>"
    )
    flow.append(Paragraph(kpi_line, s_body))
    flow.append(Spacer(1, 0.3 * cm))
    flow.append(HRFlowable(width="100%", thickness=0.6,
                            color=colors.HexColor("#e6ebf2")))

    # ─── TOP 3 ATENCIÓN ───
    flow.append(Paragraph("Top activos que requieren atención", s_section))
    sev_order = {"critico":0,"danger":0,"atencion":1,"warning":1,"unknown":2,"healthy":3}
    sorted_inst = sorted(
        snapshot.get("instances", []),
        key=lambda i: (sev_order.get(i.get("severity","unknown"), 9),
                       i.get("health_score", 100))
    )
    top3 = sorted_inst[:3]
    if not top3:
        flow.append(Paragraph("Sin activos para reportar.", s_muted))
    else:
        rows = [["Tag", "Score", "Banda", "Última señal", "Razón"]]
        for inst in top3:
            tag = inst.get("tag", "—")
            score = inst.get("health_score", 0)
            band = inst.get("health_band", "—")
            exec_sev = inst.get("exec_severity", "")
            last_signal = exec_sev if exec_sev else inst.get("severity_label", "—")
            reason = inst.get("health_one_liner", "")[:55]
            rows.append([tag, f"{score}/100", band, last_signal, reason])
        t = Table(rows, colWidths=[2.4*cm, 2*cm, 2.4*cm, 4*cm, 7*cm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#475569")),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f8fafc")),
            ("BOX", (0,0), (-1,-1), 0.4, colors.HexColor("#e6ebf2")),
            ("LINEBELOW", (0,0), (-1,0), 0.4, colors.HexColor("#e6ebf2")),
            ("LINEBELOW", (0,-2), (-1,-2), 0.2, colors.HexColor("#f1f5f9")),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        flow.append(t)

    # ─── CAMBIOS VS AYER ───
    flow.append(Paragraph("Cambios desde el último briefing", s_section))
    for line in deltas.get("lines", [])[:6]:
        flow.append(Paragraph(f"&nbsp;&nbsp;{line}", s_body))

    # ─── PRÓXIMOS VENCIMIENTOS ───
    flow.append(Paragraph("Próximos vencimientos / pendientes", s_section))
    for item in upcoming[:5]:
        flow.append(Paragraph(f"&nbsp;&nbsp;{item}", s_body))

    # ─── PRÓXIMOS PASOS ───
    flow.append(Paragraph("Próximos pasos sugeridos", s_section))
    for action in actions[:4]:
        flow.append(Paragraph(f"&nbsp;&nbsp;{action}", s_body))

    flow.append(Spacer(1, 0.4 * cm))
    flow.append(HRFlowable(width="100%", thickness=0.4,
                            color=colors.HexColor("#e6ebf2")))
    flow.append(Spacer(1, 0.2 * cm))
    footer = (
        f"Generado por Watermelon System {_ver} &nbsp;·&nbsp; "
        f"{d.strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp; "
        f"<i>Análisis predictivo de vibración industrial — SIGASAS</i>"
    )
    flow.append(Paragraph(footer, s_muted))

    doc.build(flow)
    return buf.getvalue()


def generate_and_save_briefing(date: Optional[datetime] = None) -> Tuple[bytes, Path]:
    """Conveniencia: genera el PDF, lo guarda en disco, devuelve (bytes, path)."""
    _ensure_dirs()
    d = date or datetime.now()
    snapshot = compute_fleet_snapshot()
    save_daily_snapshot(snapshot, d)
    deltas = compute_deltas_vs_yesterday(snapshot, load_yesterday_snapshot(d))
    pdf_bytes = generate_briefing_pdf(snapshot, deltas, d)
    fname = f"briefing_{d.strftime('%Y-%m-%d')}.pdf"
    path = BRIEFINGS_DIR / fname
    path.write_bytes(pdf_bytes)
    return pdf_bytes, path


__all__ = [
    "compute_fleet_snapshot",
    "compute_deltas_vs_yesterday",
    "compute_upcoming_items",
    "suggest_next_actions",
    "save_daily_snapshot",
    "load_yesterday_snapshot",
    "generate_briefing_pdf",
    "generate_and_save_briefing",
    "BRIEFINGS_DIR",
    "SNAPSHOTS_DIR",
]
