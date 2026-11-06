"""
core.report_message
====================

Genera el mensaje corto, preciso y conciso que acompaña al reporte
ejecutivo PDF cuando se envía al cliente (email y WhatsApp).

Tono: profesional, claro, sin jerga. El cliente (gerente / mantenimiento)
debe entender el estado del activo en 5 segundos. El detalle vive en el PDF.

El email HTML usa la identidad visual de Watermelon System (cáscara verde +
franja rebanada 🍉), igual que el correo del código de acceso (core.auth_otp).

Funciones puras (sin Streamlit) para reusar desde el botón manual y el cron.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Tuple


def _now_local() -> datetime:
    """Hora local del cliente (America/Bogota) — Ciclo 23.157.
    El servidor corre en UTC; sin esto el correo decía 07:00 a las 2 AM."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Bogota"))
    except Exception:
        return datetime.now()


def _tag(instance_obj: Any, instance_id: str) -> str:
    return (getattr(instance_obj, "tag", "") or instance_id or "Activo").strip()


def _client(instance_obj: Any) -> str:
    return (getattr(instance_obj, "client", "") or "").strip()


def _status_emoji(status: str) -> str:
    s = (status or "").lower()
    if "crít" in s or "crit" in s or "danger" in s:
        return "🔴"
    if "atenci" in s or "alarma" in s or "alert" in s:
        return "🟠"
    return "🟢"


def _status_color(status: str) -> str:
    """Color de acento del estado (para chips/badges del email)."""
    s = (status or "").lower()
    if "crít" in s or "crit" in s or "danger" in s:
        return "#dc2626"
    if "atenci" in s or "alarma" in s or "alert" in s:
        return "#e8890c"
    return "#1f9d55"


def _period_kind(period_label: str) -> Dict[str, str]:
    """Normaliza el período a (título, cadencia, alcance, ventana_dias).

    title    → nombre del reporte para asunto y encabezado.
    cadence  → qué cubre, en una frase corta.
    scope    → qué trae el PDF (varía: diario = 1 pág; semanal/mensual = completo).
    days     → ventana en días (0 = snapshot del día).
    """
    p = (period_label or "").strip().lower()
    if p.startswith("seman"):
        return {"title": "Reporte Semanal", "cadence": "Resumen de los últimos 7 días",
                "scope": "Arranque (Bode, Full Spectrum, Órbitas), Shaft Centerline, "
                         "tendencias y eventos", "days": "7"}
    if p.startswith("mensu"):
        return {"title": "Reporte Mensual", "cadence": "Resumen de los últimos 30 días",
                "scope": "Arranque (Bode, Full Spectrum, Órbitas), Shaft Centerline, "
                         "tendencias y eventos", "days": "30"}
    if p.startswith("rápid") or p.startswith("rapid") or p.startswith("diari"):
        return {"title": "Reporte Diario", "cadence": "Snapshot de condición del día",
                "scope": "tendencia, canales (Overall + 1X/2X) y eventos recientes",
                "days": "0"}
    return {"title": "Reporte de condición", "cadence": "Condición actual del activo",
            "scope": "tendencia, canales y eventos", "days": "0"}


def _period_range(days: str) -> str:
    """Rango de fechas legible para la ventana (ej. '30 oct – 06 nov 2026')."""
    try:
        d = int(days)
    except Exception:
        d = 0
    if d <= 0:
        return _now_local().strftime("%d %b %Y")
    hoy = _now_local()
    ini = hoy - timedelta(days=d)
    _es = {"Jan": "ene", "Feb": "feb", "Mar": "mar", "Apr": "abr", "May": "may",
           "Jun": "jun", "Jul": "jul", "Aug": "ago", "Sep": "sep", "Oct": "oct",
           "Nov": "nov", "Dec": "dic"}

    def _f(dt: datetime, year: bool = False) -> str:
        s = dt.strftime("%d %b") + (dt.strftime(" %Y") if year else "")
        for en, es in _es.items():
            s = s.replace(en, es)
        return s
    return f"{_f(ini)} – {_f(hoy, year=True)}"


def build_email_subject(instance_obj: Any, instance_id: str, status: str,
                        period_label: str = "") -> str:
    tag = _tag(instance_obj, instance_id)
    fecha = _now_local().strftime("%d/%m/%Y")
    titulo = _period_kind(period_label)["title"]
    return f"{titulo} — {tag} — {status} ({fecha})"


def build_short_message(
    instance_obj: Any,
    instance_id: str,
    status: str,
    score: Any = None,
    alarms: int = 0,
    period_label: str = "",
) -> str:
    """Mensaje corto para WhatsApp (caption) y cuerpo de texto del email."""
    tag = _tag(instance_obj, instance_id)
    client = _client(instance_obj)
    fecha = _now_local().strftime("%d/%m/%Y %H:%M")
    emoji = _status_emoji(status)
    pk = _period_kind(period_label)
    rango = _period_range(pk["days"])

    linea_activo = f"*{tag}*" + (f" — {client}" if client else "")
    salud = f" · Salud {score}/100" if score is not None else ""
    alarmas = ""
    if alarms:
        alarmas = f"\n⚠ {alarms} canal(es) sobre umbral — ver detalle en el PDF."

    return (
        f"🍉 *Watermelon System* — {pk['title']}\n"
        f"{linea_activo}\n"
        f"{emoji} Estado: {status}{salud}\n"
        f"🗓 {pk['cadence']} ({rango})\n"
        f"📄 Adjuntamos el PDF: {pk['scope']}."
        f"{alarmas}\n"
        f"{fecha} · SIGA · ISO 20816-3 / API 670"
    )


def build_email_html(
    instance_obj: Any,
    instance_id: str,
    status: str,
    score: Any = None,
    alarms: int = 0,
    period_label: str = "",
) -> str:
    """Cuerpo HTML con identidad Watermelon (cáscara verde + franja rebanada)."""
    tag = _tag(instance_obj, instance_id)
    client = _client(instance_obj)
    fecha = _now_local().strftime("%d/%m/%Y %H:%M")
    emoji = _status_emoji(status)
    sc = _status_color(status)
    pk = _period_kind(period_label)
    rango = _period_range(pk["days"])
    cliente = f" · {client}" if client else ""

    score_chip = ""
    if score is not None:
        score_chip = (
            f"<td style=\"vertical-align:middle;padding-left:10px\">"
            f"<div style=\"display:inline-block;background:#f1f5f9;color:#0f172a;"
            f"font-size:13px;font-weight:700;padding:6px 13px;border-radius:999px\">"
            f"Salud&nbsp;{score}/100</div></td>"
        )
    alarma_block = ""
    if alarms:
        alarma_block = (
            f"<div style=\"margin:0 0 16px;background:#fff7ed;border:1px solid #fed7aa;"
            f"border-left:4px solid {sc};border-radius:10px;padding:11px 14px;"
            f"font-size:13.5px;color:#9a3412\">"
            f"&#9888; <b>{alarms} canal(es) sobre umbral</b> — ver detalle en el "
            f"PDF adjunto.</div>"
        )

    return (
        "<div style=\"font-family:-apple-system,Segoe UI,Arial,Helvetica,sans-serif;"
        "background:#eef2f6;padding:28px 16px\">"
        "<div style=\"max-width:560px;margin:0 auto;background:#ffffff;border-radius:18px;"
        "overflow:hidden;box-shadow:0 8px 30px rgba(15,23,42,.08)\">"
        # ---- Encabezado cáscara verde ----
        "<div style=\"background:#1f7a43;padding:22px 28px\">"
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" width=\"100%\"><tr>"
        "<td style=\"vertical-align:middle;width:48px\"><div style=\"width:48px;height:48px;"
        "border-radius:50%;background:#ffffff;text-align:center;line-height:48px;"
        "font-size:27px\">&#127817;</div></td>"
        "<td style=\"vertical-align:middle;padding-left:14px\">"
        "<div style=\"font-size:13px;letter-spacing:.16em;color:#ffffff;font-weight:700\">"
        "WATERMELON&nbsp;SYSTEM</div>"
        f"<div style=\"font-size:12px;color:#cdeed6\">{pk['title']}</div></td>"
        "</tr></table></div>"
        # ---- Franja rebanada (verde / blanco / rojo) ----
        "<div style=\"height:6px;background:#2fa65a\"></div>"
        "<div style=\"height:4px;background:#ffffff\"></div>"
        "<div style=\"height:5px;background:#e23b55\"></div>"
        # ---- Badge de tipo de reporte ----
        "<div style=\"padding:22px 30px 4px\">"
        f"<div style=\"display:inline-block;background:{sc};color:#fff;font-size:11px;"
        "font-weight:700;letter-spacing:.06em;padding:4px 12px;border-radius:999px;"
        f"text-transform:uppercase\">{pk['title']}</div></div>"
        # ---- Activo + estado ----
        "<div style=\"padding:8px 30px 4px;color:#243041;font-size:15px;line-height:1.6\">"
        f"<p style=\"margin:0 0 2px;font-size:19px;font-weight:700;color:#0f172a\">{tag}"
        f"<span style=\"font-size:14px;font-weight:400;color:#64748b\">{cliente}</span></p>"
        "<table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" "
        "style=\"margin:10px 0 14px\"><tr>"
        "<td style=\"vertical-align:middle\"><div style=\"display:inline-block;"
        f"background:{sc};color:#fff;font-size:13px;font-weight:700;padding:6px 14px;"
        f"border-radius:999px\">{emoji}&nbsp;{status}</div></td>"
        f"{score_chip}</tr></table>"
        f"{alarma_block}"
        # ---- Qué cubre ----
        f"<p style=\"margin:0 0 4px\"><b>{pk['cadence']}</b> "
        f"<span style=\"color:#64748b\">({rango})</span></p>"
        f"<p style=\"margin:0 0 2px;color:#475569;font-size:14px\">Adjuntamos el PDF con "
        f"{pk['scope']}.</p>"
        "</div>"
        # ---- Pie ----
        "<div style=\"padding:20px 30px 26px\"><div style=\"border-top:1px solid #eef2f6;"
        "padding-top:14px;font-size:12px;color:#9aa7b5\">"
        f"{fecha} &middot; Watermelon System &middot; SIGA Group SAS &middot; "
        "<span style=\"color:#1f7a43\">ISO&nbsp;20816-3 / API&nbsp;670 · monitoreo que "
        "protege tus activos</span></div></div>"
        "</div></div>"
    )


__all__ = ["build_email_subject", "build_short_message", "build_email_html"]
