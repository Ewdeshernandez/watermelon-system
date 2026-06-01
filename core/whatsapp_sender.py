"""
core.whatsapp_sender
====================

Envío del reporte ejecutivo (PDF) al cliente por WhatsApp usando la
**Meta WhatsApp Cloud API** (la misma cuenta del watermelon-bot).

Flujo:
    1. Subir el PDF como media → media_id.
    2. Enviar el documento al número del cliente.

Dos modos:
    * TEMPLATE (producción): mensaje "iniciado por el negocio" (programado,
      fuera de la ventana de 24 h) — Meta EXIGE una plantilla pre-aprobada
      con header de tipo documento. Configurar `template_name` en secrets.
    * DOCUMENTO directo (testing): solo funciona dentro de la ventana de
      24 h posterior a que el cliente le escriba al número. Útil para
      probar sin esperar la aprobación de la plantilla.

Config (secrets.toml → sección [whatsapp]):

    [whatsapp]
    phone_number_id = "1234567890"      # Phone Number ID del WABA
    access_token    = "EAAG..."          # token del system user (largo)
    api_version     = "v21.0"            # opcional
    template_name   = "reporte_condicion"  # opcional (si no, modo documento)
    template_lang   = "es"               # opcional, default "es"

También se leen de variables de entorno (para el cron headless):
    WM_WA_PHONE_NUMBER_ID, WM_WA_ACCESS_TOKEN, WM_WA_API_VERSION,
    WM_WA_TEMPLATE_NAME, WM_WA_TEMPLATE_LANG.

Todas las funciones son PURAS (no dependen de sesión Streamlit) para poder
correr desde el cron. Si no hay config, devuelven {"ok": False, ...} sin
crashear.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

_DEFAULT_API_VERSION = "v21.0"
_GRAPH = "https://graph.facebook.com"


# =============================================================
# Config
# =============================================================

def _read_secret_section(section: str) -> Dict[str, Any]:
    """Lee una sección de secrets.toml vía st.secrets si hay Streamlit.
    Headless (cron) cae a {} y se completa con variables de entorno."""
    try:
        import streamlit as st  # type: ignore
        node = st.secrets
        if section in node:
            sub = node[section]
            return dict(sub) if hasattr(sub, "keys") else {}
    except Exception:
        pass
    return {}


def _config() -> Dict[str, str]:
    cfg = _read_secret_section("whatsapp")
    return {
        "phone_number_id": str(
            cfg.get("phone_number_id") or os.environ.get("WM_WA_PHONE_NUMBER_ID", "")
        ).strip(),
        "access_token": str(
            cfg.get("access_token") or os.environ.get("WM_WA_ACCESS_TOKEN", "")
        ).strip(),
        "api_version": str(
            cfg.get("api_version") or os.environ.get("WM_WA_API_VERSION", _DEFAULT_API_VERSION)
        ).strip() or _DEFAULT_API_VERSION,
        "template_name": str(
            cfg.get("template_name") or os.environ.get("WM_WA_TEMPLATE_NAME", "")
        ).strip(),
        "template_lang": str(
            cfg.get("template_lang") or os.environ.get("WM_WA_TEMPLATE_LANG", "es")
        ).strip() or "es",
    }


def whatsapp_status() -> Dict[str, Any]:
    """Estado de configuración (para mostrar en UI)."""
    cfg = _config()
    configured = bool(cfg["phone_number_id"] and cfg["access_token"])
    return {
        "configured": configured,
        "has_template": bool(cfg["template_name"]),
        "phone_number_id": (cfg["phone_number_id"][:6] + "…") if cfg["phone_number_id"] else "",
        "template_name": cfg["template_name"],
        "mode": "template" if cfg["template_name"] else ("document" if configured else "none"),
    }


def _norm_number(raw: str) -> str:
    """Normaliza a dígitos E.164 sin '+', espacios ni guiones."""
    if not raw:
        return ""
    return "".join(ch for ch in str(raw) if ch.isdigit())


# =============================================================
# API calls
# =============================================================

def _upload_media(cfg: Dict[str, str], pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Sube el PDF como media. Devuelve {"ok", "media_id"} o {"ok": False, "error"}."""
    try:
        import requests
    except Exception as e:
        return {"ok": False, "error": f"requests no disponible: {e}"}

    url = f"{_GRAPH}/{cfg['api_version']}/{cfg['phone_number_id']}/media"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {cfg['access_token']}"},
            files={"file": (filename, pdf_bytes, "application/pdf")},
            data={"messaging_product": "whatsapp", "type": "application/pdf"},
            timeout=60,
        )
        if resp.status_code == 200:
            mid = resp.json().get("id")
            if mid:
                return {"ok": True, "media_id": mid}
            return {"ok": False, "error": f"sin media id en respuesta: {resp.text[:200]}"}
        return {"ok": False, "error": f"upload status={resp.status_code} {resp.text[:300]}"}
    except Exception as e:
        return {"ok": False, "error": f"upload falló: {e}"}


def _post_message(cfg: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import requests
    except Exception as e:
        return {"ok": False, "error": f"requests no disponible: {e}"}

    url = f"{_GRAPH}/{cfg['api_version']}/{cfg['phone_number_id']}/messages"
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {cfg['access_token']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            j = resp.json()
            mid = (j.get("messages") or [{}])[0].get("id", "")
            return {"ok": True, "message_id": mid}
        return {"ok": False, "error": f"send status={resp.status_code} {resp.text[:400]}"}
    except Exception as e:
        return {"ok": False, "error": f"send falló: {e}"}


# =============================================================
# Envío de alto nivel
# =============================================================

def send_report_document(
    to: str,
    pdf_bytes: bytes,
    filename: str,
    caption: str = "",
    body_params: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Envía el PDF al cliente por WhatsApp.

    Si hay `template_name` configurado → envía como plantilla (producción,
    funciona en cualquier momento, requiere aprobación de Meta). El template
    debe tener header de tipo DOCUMENTO y un body con las variables que se
    pasen en `body_params` (en orden {{1}}, {{2}}, …).

    Si NO hay template → envía un mensaje de documento directo (solo funciona
    dentro de la ventana de 24 h; útil para testing).

    Returns: {"ok": bool, "mode": "template"|"document", "message_id"|"error"}
    """
    cfg = _config()
    if not (cfg["phone_number_id"] and cfg["access_token"]):
        return {"ok": False, "error": "WhatsApp no configurado (falta phone_number_id/access_token)."}

    number = _norm_number(to)
    if not number:
        return {"ok": False, "error": f"Número WhatsApp inválido: {to!r}"}

    # 1) subir media
    up = _upload_media(cfg, pdf_bytes, filename)
    if not up.get("ok"):
        return up
    media_id = up["media_id"]

    # 2) enviar
    if cfg["template_name"]:
        components: List[Dict[str, Any]] = [{
            "type": "header",
            "parameters": [{
                "type": "document",
                "document": {"id": media_id, "filename": filename},
            }],
        }]
        if body_params:
            components.append({
                "type": "body",
                "parameters": [{"type": "text", "text": str(p)} for p in body_params],
            })
        payload = {
            "messaging_product": "whatsapp",
            "to": number,
            "type": "template",
            "template": {
                "name": cfg["template_name"],
                "language": {"code": cfg["template_lang"]},
                "components": components,
            },
        }
        res = _post_message(cfg, payload)
        res["mode"] = "template"
        return res

    # modo documento directo (ventana 24h)
    document: Dict[str, Any] = {"id": media_id, "filename": filename}
    if caption:
        document["caption"] = caption[:1024]
    payload = {
        "messaging_product": "whatsapp",
        "to": number,
        "type": "document",
        "document": document,
    }
    res = _post_message(cfg, payload)
    res["mode"] = "document"
    return res


__all__ = ["send_report_document", "whatsapp_status"]
