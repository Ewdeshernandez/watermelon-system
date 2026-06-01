"""
core.report_delivery
====================

Orquesta el envío del reporte ejecutivo PDF al cliente por los canales
configurados (email y/o WhatsApp). Función pura — la usan tanto el botón
manual de Live Monitoring como el cron headless de envíos programados.

Entrada típica:
    deliver_report(
        instance_obj, pdf_bytes,
        meta={"instance_id": "tes1", "status": "Operación normal",
              "score": 74, "alarms": 0},
        channels=("email", "whatsapp"),  # opcional; default = los configurados
    )

Devuelve un dict con el resultado por canal:
    {"email": {"ok": True, ...}, "whatsapp": {"ok": False, "error": ...},
     "any_ok": True}
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from core.report_message import (
    build_email_subject,
    build_short_message,
    build_email_html,
)


def _pdf_filename(tag: str) -> str:
    safe = "".join(c for c in (tag or "activo") if c.isalnum() or c in ("-", "_")) or "activo"
    return f"Reporte_{safe}_{datetime.now().strftime('%Y%m%d')}.pdf"


def available_channels(instance_obj: Any) -> Dict[str, bool]:
    """Qué canales tienen destinatario cargado en el activo."""
    email = (getattr(instance_obj, "client_email", "") or "").strip()
    wa = (getattr(instance_obj, "whatsapp_number", "") or "").strip()
    return {"email": bool(email and "@" in email), "whatsapp": bool(wa)}


def deliver_report(
    instance_obj: Any,
    pdf_bytes: bytes,
    meta: Optional[Dict[str, Any]] = None,
    channels: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Envía el PDF por los canales pedidos. Ver docstring del módulo."""
    meta = meta or {}
    instance_id = meta.get("instance_id", "") or getattr(instance_obj, "instance_id", "")
    status = meta.get("status", "—")
    score = meta.get("score")
    alarms = int(meta.get("alarms", 0) or 0)
    tag = (getattr(instance_obj, "tag", "") or instance_id or "activo").strip()

    result: Dict[str, Any] = {"any_ok": False}

    if not pdf_bytes:
        result["error"] = "No hay PDF para enviar."
        return result

    avail = available_channels(instance_obj)
    if channels is None:
        channels = [c for c, ok in avail.items() if ok]
    channels = list(channels)

    if not channels:
        result["error"] = "El activo no tiene email ni WhatsApp configurado."
        return result

    filename = _pdf_filename(tag)
    short_msg = build_short_message(instance_obj, instance_id, status, score, alarms)

    # ---- Email ----
    if "email" in channels:
        to = (getattr(instance_obj, "client_email", "") or "").strip()
        if not (to and "@" in to):
            result["email"] = {"ok": False, "error": "Email del cliente no configurado."}
        else:
            try:
                from core.email_sender import send_email
                subject = build_email_subject(instance_obj, instance_id, status)
                html = build_email_html(instance_obj, instance_id, status, score, alarms)
                result["email"] = send_email(
                    to=to, subject=subject,
                    body_text=short_msg, body_html=html,
                    attachments=[(filename, pdf_bytes, "application/pdf")],
                )
            except Exception as e:
                result["email"] = {"ok": False, "error": f"email falló: {e}"}

    # ---- WhatsApp ----
    if "whatsapp" in channels:
        to = (getattr(instance_obj, "whatsapp_number", "") or "").strip()
        if not to:
            result["whatsapp"] = {"ok": False, "error": "WhatsApp del cliente no configurado."}
        else:
            try:
                from core.whatsapp_sender import send_report_document
                # body_params para la plantilla Meta (orden {{1}},{{2}},{{3}}):
                # 1=activo, 2=estado, 3=fecha. Ajustar al template aprobado.
                body_params = [
                    tag,
                    f"{status}" + (f" (Salud {score}/100)" if score is not None else ""),
                    datetime.now().strftime("%d/%m/%Y %H:%M"),
                ]
                result["whatsapp"] = send_report_document(
                    to=to, pdf_bytes=pdf_bytes, filename=filename,
                    caption=short_msg, body_params=body_params,
                )
            except Exception as e:
                result["whatsapp"] = {"ok": False, "error": f"whatsapp falló: {e}"}

    result["any_ok"] = any(
        isinstance(result.get(c), dict) and result[c].get("ok") for c in channels
    )
    return result


__all__ = ["deliver_report", "available_channels"]
