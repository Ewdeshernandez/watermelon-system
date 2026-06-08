"""
core.session_cookie
===================

Persistencia de sesión en una cookie firmada del navegador (Ciclo 23.163).

Problema que resuelve: Streamlit guarda `st.session_state` en memoria del
servidor por websocket. Al refrescar la página —o cuando el websocket se
reconecta tras inactividad/red— se crea una sesión NUEVA vacía y el usuario
aparece deslogueado, aunque su sesión siga vigente. Esto hacía la app
"inestable": refresh = re-login.

Solución: al iniciar sesión se escribe una cookie `wm_session` con un token
firmado (HMAC-SHA256) que incluye los datos del usuario y una expiración.
En cada carga (incluido refresh) el navegador envía la cookie en el request
HTTP; `st.context.cookies` la entrega de inmediato y rehidratamos
`st.session_state`. La cookie tiene `max-age` de 6 h y se renueva en cada
navegación → ventana DESLIZANTE: el usuario permanece logueado mientras use
la app, y solo se cierra tras ~6 h sin navegar.

Lectura: nativa (`st.context.cookies`), sin iframe, sin parpadeo.
Escritura/borrado: JS al `document` padre (iframe srcdoc, mismo origen).
Sin dependencias externas.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components

COOKIE_NAME = "wm_session"
TTL_SECONDS = 6 * 60 * 60  # 6 horas de inactividad (ventana deslizante)


def _secret() -> bytes:
    """Clave HMAC para firmar el token. Prioriza un secreto dedicado;
    si no existe, deriva de credenciales estables ya presentes. Nunca se
    expone al cliente (solo se usa server-side para firmar/verificar)."""
    try:
        auth = st.secrets.get("auth", {})
        s = auth.get("session_secret") if hasattr(auth, "get") else None
        if s:
            return str(s).encode("utf-8")
    except Exception:
        pass
    for path in (("supabase", "service_key"), ("supabase", "url")):
        try:
            node = st.secrets
            for k in path:
                node = node[k]
            if node:
                return hashlib.sha256(str(node).encode("utf-8")).digest()
        except Exception:
            continue
    # Último recurso (evita crash; en prod siempre hay secret de supabase)
    return b"watermelon-session-fallback-key-v1"


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(txt: str) -> bytes:
    pad = "=" * (-len(txt) % 4)
    return base64.urlsafe_b64decode(txt + pad)


def make_token(payload: Dict[str, Any], ttl: int = TTL_SECONDS) -> str:
    body = dict(payload)
    body["exp"] = int(time.time()) + int(ttl)
    raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    p = _b64e(raw)
    sig = hmac.new(_secret(), p.encode("ascii"), hashlib.sha256).digest()
    return f"{p}.{_b64e(sig)}"


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        p, sig = token.split(".", 1)
        expected = hmac.new(_secret(), p.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64d(sig), expected):
            return None
        body = json.loads(_b64d(p).decode("utf-8"))
        if int(body.get("exp", 0)) < int(time.time()):
            return None
        return body
    except Exception:
        return None


def _write_cookie(value: str, max_age: int) -> None:
    """Escribe la cookie en el documento padre (mismo origen, vía iframe
    srcdoc). Secure + SameSite=Lax. height=0 → invisible."""
    components.html(
        f"""<script>
        try {{
            window.parent.document.cookie =
                "{COOKIE_NAME}={value}; path=/; max-age={int(max_age)}; SameSite=Lax; Secure";
        }} catch (e) {{}}
        </script>""",
        height=0,
    )


def persist(user: Dict[str, Any]) -> None:
    """Escribe/renueva la cookie de sesión con los datos del usuario."""
    try:
        token = make_token({
            "email":     user.get("email", ""),
            "username":  user.get("username", ""),
            "full_name": user.get("full_name", ""),
            "role":      user.get("role", ""),
            "user_id":   user.get("user_id", "") or user.get("id", ""),
            "is_admin":  bool(user.get("is_admin", False)),
            "source":    user.get("source", "supabase"),
        })
        _write_cookie(token, TTL_SECONDS)
    except Exception:
        pass


def restore() -> Optional[Dict[str, Any]]:
    """Lee la cookie del request (nativo) y devuelve el payload si el token
    es válido y no expiró. None si no hay cookie o es inválida."""
    try:
        cookies = getattr(st.context, "cookies", None)
        if not cookies:
            return None
        token = cookies.get(COOKIE_NAME)
        if not token:
            return None
        return verify_token(token)
    except Exception:
        return None


def clear() -> None:
    """Borra la cookie (logout)."""
    try:
        _write_cookie("", 0)
    except Exception:
        pass


__all__ = ["persist", "restore", "clear", "make_token", "verify_token",
           "COOKIE_NAME", "TTL_SECONDS"]
