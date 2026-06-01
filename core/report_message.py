"""
core.report_message
====================

Genera el mensaje corto, preciso y conciso que acompaña al reporte
ejecutivo PDF cuando se envía al cliente (email y WhatsApp).

Tono: profesional, claro, sin jerga. El cliente (gerente / mantenimiento)
debe entender el estado del activo en 5 segundos. El detalle vive en el PDF.

Funciones puras (sin Streamlit) para reusar desde el botón manual y el cron.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


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


def build_email_subject(instance_obj: Any, instance_id: str, status: str) -> str:
    tag = _tag(instance_obj, instance_id)
    fecha = datetime.now().strftime("%d/%m/%Y")
    return f"Reporte de condición — {tag} — {status} ({fecha})"


def build_short_message(
    instance_obj: Any,
    instance_id: str,
    status: str,
    score: Any = None,
    alarms: int = 0,
) -> str:
    """Mensaje corto para WhatsApp (caption) y cuerpo de texto del email."""
    tag = _tag(instance_obj, instance_id)
    client = _client(instance_obj)
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    emoji = _status_emoji(status)

    linea_activo = f"*{tag}*" + (f" — {client}" if client else "")
    salud = f" · Salud {score}/100" if score is not None else ""
    alarmas = ""
    if alarms:
        alarmas = f"\n⚠ {alarms} canal(es) sobre umbral — ver detalle en el PDF."

    return (
        f"🍉 *Watermelon System* — Reporte de condición\n"
        f"{linea_activo}\n"
        f"{emoji} Estado: {status}{salud}\n"
        f"📄 Adjuntamos el reporte ejecutivo (tendencia, canales y eventos)."
        f"{alarmas}\n"
        f"🗓 {fecha} · SIGA · ISO 20816-3 / API 670"
    )


def build_email_html(
    instance_obj: Any,
    instance_id: str,
    status: str,
    score: Any = None,
    alarms: int = 0,
) -> str:
    """Cuerpo HTML simple y sobrio para el email (preferido por mail clients)."""
    tag = _tag(instance_obj, instance_id)
    client = _client(instance_obj)
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    emoji = _status_emoji(status)
    salud = f" · Salud <b>{score}/100</b>" if score is not None else ""
    alarma_block = (
        f"<p style='margin:8px 0;color:#b45309;'>⚠ {alarms} canal(es) sobre "
        f"umbral — ver detalle en el PDF adjunto.</p>" if alarms else ""
    )
    cliente_block = f" — {client}" if client else ""
    return f"""\
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#0f172a;font-size:14px;line-height:1.5;">
  <p style="margin:0 0 4px;font-size:16px;"><b>🍉 Watermelon System</b> — Reporte de condición</p>
  <p style="margin:0 0 2px;"><b>{tag}</b>{cliente_block}</p>
  <p style="margin:0 0 2px;">{emoji} Estado: <b>{status}</b>{salud}</p>
  {alarma_block}
  <p style="margin:8px 0 2px;">Adjuntamos el reporte ejecutivo de 1 página con
  la tendencia, los canales (Overall + 1X/2X) y los eventos recientes.</p>
  <p style="margin:12px 0 0;color:#64748b;font-size:12px;">{fecha} · SIGA ·
  Monitoreo de condición de maquinaria rotativa · ISO 20816-3 / API 670</p>
</div>"""


__all__ = ["build_email_subject", "build_short_message", "build_email_html"]
