"""
core.email_sender
=================

Envío de emails desde Watermelon System (Ciclo 17.16).

Soporta dos backends — el sistema detecta cuál usar según los secrets
configurados:

  1) SMTP con AUTH (Microsoft 365 SMTP, Gmail App Password, etc.)
     Requiere en st.secrets:
       [email.smtp]
       host = "smtp.office365.com"
       port = 587
       starttls = true
       username = "noreply@sigasas.com"
       password = "<app password o pwd del mailbox>"
       from_email = "noreply@sigasas.com"
       from_name = "Watermelon System"

  2) Microsoft Graph API (OAuth 2.0 client credentials)
     Más moderno, no requiere SMTP basic auth en el tenant.
     Requiere en st.secrets:
       [email.graph]
       tenant_id = "xxxxx"
       client_id = "xxxxx"
       client_secret = "xxxxx"
       from_email = "noreply@sigasas.com"
       from_name = "Watermelon System"

Si NINGUNO está configurado, send_email devuelve {"ok": False,
"error": "..."} sin tirar excepción — el caller decide qué hacer
(ej. log + mostrar mensaje al usuario "el sistema no puede enviar
emails todavía, contactá al admin").

API pública:
  - send_email(to, subject, body_text, body_html=None, attachments=None)
  - send_password_reset_email(email, token_url, full_name="")
  - send_briefing_email(email, pdf_bytes, date_str, full_name="")
  - get_email_backend_status() — para diagnóstico en admin panel
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple


# =============================================================
# CONFIG / DETECCIÓN DE BACKEND
# =============================================================

def _safe_subdict(parent: Any, key: str) -> Dict[str, Any]:
    """Devuelve un subdict de `parent[key]` aunque parent sea un AttrDict
    de Streamlit Cloud (que no se convierte limpio a dict puro con
    dict()). Ciclo 17.16.1 fix: en Streamlit Cloud, st.secrets es un
    AttrDict y st.secrets['email'] devuelve OTRO AttrDict, no un dict
    puro — por eso isinstance(x, dict) daba False y todo el chequeo
    de backends fallaba con "no_backend" aunque el TOML estuviera bien.
    """
    if parent is None:
        return {}
    try:
        sub = parent[key]
    except (KeyError, TypeError):
        try:
            sub = parent.get(key)  # AttrDict tiene .get
        except Exception:
            sub = None
    if sub is None:
        return {}
    # Convertir a dict puro recursivamente
    try:
        return {k: sub[k] for k in sub}
    except Exception:
        try:
            return dict(sub)
        except Exception:
            return {}


def _read_secret_section(*path: str) -> Dict[str, Any]:
    """Lee una sección anidada de st.secrets como dict puro.
    Ej: _read_secret_section('email', 'graph') → contenido de [email.graph].
    Devuelve dict vacío si no existe o si st no está disponible.
    """
    try:
        import streamlit as st  # type: ignore
        if not hasattr(st, "secrets"):
            return {}
        node: Any = st.secrets
        for k in path:
            sub = _safe_subdict(node, k)
            if not sub:
                return {}
            node = sub
        return node if isinstance(node, dict) else {}
    except Exception:
        return {}


def _get_secrets() -> Dict[str, Any]:
    """Compat: lee st.secrets como dict si está disponible.
    Solo usado por código viejo — los nuevos accesos usan
    _read_secret_section() que es más robusto en Cloud.
    """
    try:
        import streamlit as st  # type: ignore
        return dict(st.secrets) if hasattr(st, "secrets") else {}
    except Exception:
        return {}


def get_email_backend_status() -> Dict[str, Any]:
    """Devuelve qué backend está configurado y si tiene los campos
    mínimos necesarios. Útil para mostrar diagnóstico en admin panel.

    Ciclo 17.16.1: usa _read_secret_section() para evitar el bug del
    AttrDict de Streamlit Cloud que rompía el isinstance(x, dict) check.
    """
    smtp = _read_secret_section("email", "smtp")
    graph = _read_secret_section("email", "graph")

    smtp_ok = bool(
        smtp.get("host") and smtp.get("port")
        and smtp.get("username") and smtp.get("password")
        and smtp.get("from_email")
    )
    graph_ok = bool(
        graph.get("tenant_id") and graph.get("client_id")
        and graph.get("client_secret") and graph.get("from_email")
    )

    if smtp_ok:
        return {
            "configured": True,
            "backend": "smtp",
            "from_email": smtp.get("from_email", ""),
            "host": smtp.get("host", ""),
            "details": "SMTP backend listo.",
        }
    if graph_ok:
        return {
            "configured": True,
            "backend": "graph",
            "from_email": graph.get("from_email", ""),
            "tenant_id": graph.get("tenant_id", "")[:8] + "…",
            "details": "Microsoft Graph API backend listo.",
        }
    return {
        "configured": False,
        "backend": None,
        "details": (
            "NO hay backend de email configurado. Agregá en secrets:\n"
            "  [email.smtp]  con host/port/username/password/from_email\n"
            "  o [email.graph]  con tenant_id/client_id/client_secret/from_email"
        ),
    }


# =============================================================
# SMTP backend
# =============================================================

def _send_via_smtp(
    *,
    to: str,
    subject: str,
    body_text: str,
    body_html: str = "",
    attachments: Optional[List[Tuple[str, bytes, str]]] = None,
) -> Dict[str, Any]:
    """Envía email vía SMTP (Office 365, Gmail, etc.)."""
    cfg = _read_secret_section("email", "smtp")
    host = cfg.get("host", "")
    port = int(cfg.get("port", 587) or 587)
    starttls = bool(cfg.get("starttls", True))
    username = cfg.get("username", "")
    password = cfg.get("password", "")
    from_email = cfg.get("from_email", username)
    from_name = cfg.get("from_name", "Watermelon System")

    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.application import MIMEApplication
        from email.utils import formataddr
    except Exception as e:
        return {"ok": False, "error": f"smtplib no disponible: {e}"}

    msg = MIMEMultipart("alternative")
    msg["From"] = formataddr((from_name, from_email))
    msg["To"] = to
    msg["Subject"] = subject

    msg.attach(MIMEText(body_text or "", "plain", "utf-8"))
    if body_html:
        msg.attach(MIMEText(body_html, "html", "utf-8"))

    for fname, data, mime in (attachments or []):
        part = MIMEApplication(data, _subtype=mime.split("/")[-1] if "/" in mime else "octet-stream")
        part.add_header("Content-Disposition", "attachment", filename=fname)
        msg.attach(part)

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            if starttls:
                server.starttls()
                server.ehlo()
            server.login(username, password)
            server.send_message(msg)
        return {"ok": True, "backend": "smtp", "to": to}
    except smtplib.SMTPAuthenticationError as e:
        return {"ok": False, "error": (
            f"Falló la autenticación SMTP ({e.smtp_code}). "
            f"Posibles causas: (1) Office 365 tiene SMTP AUTH desactivado "
            f"para este mailbox — el admin del tenant debe habilitarlo, "
            f"(2) MFA está activo y necesitás un App Password en lugar "
            f"de la pwd del usuario, (3) credenciales incorrectas. "
            f"Mensaje: {e.smtp_error.decode('utf-8', errors='replace')[:200]}"
        )}
    except Exception as e:
        return {"ok": False, "error": f"SMTP error: {type(e).__name__}: {e}"}


# =============================================================
# Microsoft Graph backend
# =============================================================

def _send_via_graph(
    *,
    to: str,
    subject: str,
    body_text: str,
    body_html: str = "",
    attachments: Optional[List[Tuple[str, bytes, str]]] = None,
) -> Dict[str, Any]:
    """Envía email vía Microsoft Graph API (OAuth client_credentials).

    Requiere que la App Registration en Azure AD tenga el permiso
    'Mail.Send' (Application permission) con consent del admin del tenant.
    """
    cfg = _read_secret_section("email", "graph")
    tenant_id = cfg.get("tenant_id", "")
    client_id = cfg.get("client_id", "")
    client_secret = cfg.get("client_secret", "")
    from_email = cfg.get("from_email", "")
    from_name = cfg.get("from_name", "Watermelon System")

    # 1. Obtener access token vía client_credentials
    try:
        import httpx
    except ImportError:
        return {"ok": False, "error": (
            "El paquete 'httpx' es requerido para el backend Graph. "
            "Instalá: pip install httpx"
        )}

    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(token_url, data={
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "https://graph.microsoft.com/.default",
                "grant_type": "client_credentials",
            })
            if r.status_code != 200:
                return {"ok": False, "error": (
                    f"Falló obtener token de Microsoft (status {r.status_code}): "
                    f"{r.text[:300]}"
                )}
            access_token = r.json().get("access_token")
            if not access_token:
                return {"ok": False, "error": "Microsoft no devolvió access_token"}

            # 2. Construir el mensaje y enviar
            content_type = "HTML" if body_html else "Text"
            content = body_html if body_html else body_text
            payload: Dict[str, Any] = {
                "message": {
                    "subject": subject,
                    "body": {"contentType": content_type, "content": content},
                    "toRecipients": [{"emailAddress": {"address": to}}],
                    "from": {"emailAddress": {"address": from_email, "name": from_name}},
                },
                "saveToSentItems": False,
            }
            if attachments:
                payload["message"]["attachments"] = [
                    {
                        "@odata.type": "#microsoft.graph.fileAttachment",
                        "name": fname,
                        "contentType": mime,
                        "contentBytes": base64.b64encode(data).decode("ascii"),
                    }
                    for fname, data, mime in attachments
                ]

            send_url = f"https://graph.microsoft.com/v1.0/users/{from_email}/sendMail"
            r2 = client.post(
                send_url,
                json=payload,
                headers={"Authorization": f"Bearer {access_token}",
                         "Content-Type": "application/json"},
            )
            if r2.status_code in (200, 202):
                return {"ok": True, "backend": "graph", "to": to}
            return {"ok": False, "error": (
                f"Graph sendMail falló (status {r2.status_code}): {r2.text[:300]}"
            )}
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"Graph HTTP error: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Graph error: {type(e).__name__}: {e}"}


# =============================================================
# API PÚBLICA
# =============================================================

def send_email(
    to: str,
    subject: str,
    body_text: str,
    body_html: str = "",
    attachments: Optional[List[Tuple[str, bytes, str]]] = None,
) -> Dict[str, Any]:
    """Envía un email usando el backend configurado.

    Args:
        to:           email del destinatario
        subject:      asunto
        body_text:    cuerpo en texto plano (siempre incluido)
        body_html:    cuerpo HTML opcional (preferido por mail clients)
        attachments:  lista de tuplas (filename, bytes, mime_type)

    Returns:
        {"ok": True, "backend": "smtp"|"graph", "to": ...}
        o {"ok": False, "error": "mensaje"}

    Ciclo 17.16.1: SIEMPRE logea a stderr el resultado (ok o falla)
    para que aparezca en los logs de Streamlit Cloud y se pueda
    diagnosticar problemas de envío que antes quedaban silenciosos.
    """
    import sys as _sys
    if not to or "@" not in to:
        msg = f"Email inválido: {to!r}"
        print(f"[WM_EMAIL] FAIL · validation · {msg}", file=_sys.stderr, flush=True)
        return {"ok": False, "error": msg}

    status = get_email_backend_status()
    if not status["configured"]:
        print(f"[WM_EMAIL] FAIL · no_backend · {status['details']}",
              file=_sys.stderr, flush=True)
        return {"ok": False, "error": status["details"]}

    backend = status["backend"]
    print(f"[WM_EMAIL] sending via {backend} · to={to} · subject={subject!r}",
          file=_sys.stderr, flush=True)

    if backend == "smtp":
        result = _send_via_smtp(
            to=to, subject=subject,
            body_text=body_text, body_html=body_html,
            attachments=attachments,
        )
    elif backend == "graph":
        result = _send_via_graph(
            to=to, subject=subject,
            body_text=body_text, body_html=body_html,
            attachments=attachments,
        )
    else:
        result = {"ok": False, "error": "Backend desconocido."}

    if result.get("ok"):
        print(f"[WM_EMAIL] OK · {backend} · to={to}", file=_sys.stderr, flush=True)
    else:
        # CRÍTICO: si falla, imprimimos el error completo a stderr para
        # que aparezca en logs de Cloud y se pueda diagnosticar.
        err = result.get("error", "?")
        print(f"[WM_EMAIL] FAIL · {backend} · to={to} · ERROR: {err}",
              file=_sys.stderr, flush=True)

    return result


def send_password_reset_email(
    email: str,
    token_url: str,
    full_name: str = "",
    ttl_minutes: int = 60,
) -> Dict[str, Any]:
    """Envía un mail con el link para resetear password.

    Args:
        email:       destinatario
        token_url:   URL completa con el token, ej.
                     https://wm-home-final-2026.streamlit.app/reset_password?token=xxx
        full_name:   opcional, para personalizar saludo
        ttl_minutes: cuánto vale el link (para mostrar en el email)
    """
    name = (full_name or email.split("@")[0]).strip()
    subject = "Watermelon System — Restablecer contraseña"

    text = (
        f"Hola {name},\n\n"
        f"Recibimos una solicitud para restablecer tu contraseña en\n"
        f"Watermelon System (SIGASAS).\n\n"
        f"Hacé click en el siguiente link para elegir una nueva clave:\n\n"
        f"  {token_url}\n\n"
        f"El link es válido por {ttl_minutes} minutos. Si no fuiste vos\n"
        f"quien lo pidió, podés ignorar este correo y tu cuenta seguirá\n"
        f"intacta.\n\n"
        f"Saludos,\n"
        f"Watermelon System — SIGASAS\n"
    )

    html = f"""\
<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;
             background:#f8fafc;padding:24px;color:#0f172a;">
  <div style="max-width:540px;margin:0 auto;background:white;
              border-radius:14px;padding:32px;border:1px solid #e6ebf2;">
    <div style="font-size:11px;font-weight:800;letter-spacing:0.18em;
                text-transform:uppercase;color:#0ea5e9;margin-bottom:8px;">
      🍉 Watermelon System
    </div>
    <h1 style="margin:0 0 16px 0;font-size:22px;color:#0f172a;font-weight:800;">
      Restablecer contraseña
    </h1>
    <p style="line-height:1.6;color:#475569;">
      Hola <b>{name}</b>,
    </p>
    <p style="line-height:1.6;color:#475569;">
      Recibimos una solicitud para restablecer la contraseña de tu cuenta en
      Watermelon System (SIGASAS).
    </p>
    <p style="text-align:center;margin:28px 0;">
      <a href="{token_url}"
         style="display:inline-block;background:linear-gradient(135deg,#21478c,#2a6dd1);
                color:white;text-decoration:none;padding:14px 32px;border-radius:10px;
                font-weight:700;font-size:15px;">
        Elegir nueva contraseña
      </a>
    </p>
    <p style="line-height:1.6;color:#64748b;font-size:13px;">
      Este link es válido por <b>{ttl_minutes} minutos</b>. Si no fuiste vos
      quien lo solicitó, podés ignorar este correo — tu cuenta sigue intacta.
    </p>
    <p style="line-height:1.6;color:#94a3b8;font-size:12px;
              border-top:1px solid #e6ebf2;padding-top:16px;margin-top:28px;">
      Si el botón no funciona, copiá y pegá esta URL en tu navegador:<br/>
      <span style="font-family:ui-monospace,monospace;word-break:break-all;color:#475569;">
        {token_url}
      </span>
    </p>
    <p style="color:#94a3b8;font-size:11px;text-align:center;margin-top:20px;">
      Watermelon System — SIGASAS<br/>
      Industrial Vibration Intelligence
    </p>
  </div>
</body></html>"""

    return send_email(email, subject, text, body_html=html)


def send_briefing_email(
    to: str,
    pdf_bytes: bytes,
    date_str: str,
    full_name: str = "",
) -> Dict[str, Any]:
    """Envía el briefing diario PDF como adjunto."""
    name = (full_name or to.split("@")[0]).strip()
    subject = f"Watermelon — Briefing diario · {date_str}"
    text = (
        f"Hola {name},\n\n"
        f"Adjunto el briefing diario de la flota generado automáticamente.\n"
        f"Fecha: {date_str}\n\n"
        f"Saludos,\n"
        f"Watermelon System — SIGASAS\n"
    )
    return send_email(
        to=to, subject=subject, body_text=text,
        attachments=[(f"briefing_{date_str}.pdf", pdf_bytes, "application/pdf")],
    )


__all__ = [
    "send_email",
    "send_password_reset_email",
    "send_briefing_email",
    "get_email_backend_status",
]
