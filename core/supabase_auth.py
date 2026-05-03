"""
core.supabase_auth
==================

Wrapper de Supabase Auth para Watermelon System (Ciclo 17.14).

Reemplaza el sistema viejo de usuarios hardcoded en .streamlit/secrets.toml
por usuarios reales persistidos en la tabla `auth.users` de Supabase.

Roles automáticos por dominio del email:
  - ehernandez@sigasas.com  → admin (único administrador del sistema)
  - *@sigasas.com           → specialist (todos los empleados internos)
  - cualquier otro dominio  → client (acceso restringido para clientes)

El role se persiste en `user_metadata.role` del Auth user, junto con
`user_metadata.full_name`. Esos campos se leen en cada login y se usan
para construir el usuario activo en st.session_state.

Filosofía:
  - Toda operación admin (crear/listar/eliminar/banear) usa la
    service_key de st.secrets (con bypass RLS)
  - El signin usa la misma client (anon API funciona con service_key
    también para sign_in_with_password)
  - Cualquier error de red o config se reporta sin tirar excepción —
    devuelve {"ok": False, "error": "mensaje legible"}
  - Si Supabase no está configurado, todas las funciones devuelven
    inmediatamente sin colgar la app (fallback al sistema viejo en
    core/auth.py)

Compatibilidad supabase-py: testeado con 2.29.0.
"""

from __future__ import annotations

import secrets as _secrets_module
from typing import Any, Dict, List, Optional


# =============================================================
# CONFIG GLOBAL
# =============================================================

ADMIN_EMAIL = "ehernandez@sigasas.com"
SIGASAS_DOMAIN = "sigasas.com"

# Catálogo de roles + label legible
ROLES: Dict[str, str] = {
    "admin":      "Administrador",
    "specialist": "Especialista",
    "client":     "Cliente",
    "viewer":     "Visor (legacy)",
}


# =============================================================
# REGLA DE DOMINIO
# =============================================================

def infer_role_from_email(email: str) -> str:
    """Aplica la regla de dominio para asignar role automáticamente.

    - El email del admin único se reconoce explícitamente
    - Cualquier @sigasas.com cae a 'specialist'
    - Cualquier otro dominio cae a 'client' (acceso restringido)
    """
    e = (email or "").strip().lower()
    if not e:
        return "client"
    if e == ADMIN_EMAIL.lower():
        return "admin"
    if "@" in e and e.split("@", 1)[1] == SIGASAS_DOMAIN:
        return "specialist"
    return "client"


def is_admin_email(email: str) -> bool:
    """True solo para el admin único del sistema."""
    return (email or "").strip().lower() == ADMIN_EMAIL.lower()


# =============================================================
# CLIENTE SUPABASE
# =============================================================

def is_supabase_auth_enabled() -> bool:
    """True si st.secrets['supabase'] tiene url + service_key."""
    try:
        import streamlit as st
        sb = st.secrets.get("supabase", {})
        return bool(sb.get("url") and sb.get("service_key"))
    except Exception:
        return False


def get_admin_client():
    """Devuelve un cliente Supabase con service_key (bypass RLS, admin).

    Lazy + cached por sesión Streamlit.
    """
    try:
        import streamlit as st
    except Exception:
        return None
    if not is_supabase_auth_enabled():
        return None

    cached = st.session_state.get("_supabase_admin_client")
    if cached is not None:
        return cached
    try:
        from supabase import create_client
        sb = st.secrets["supabase"]
        client = create_client(sb["url"], sb["service_key"])
        st.session_state["_supabase_admin_client"] = client
        return client
    except Exception:
        return None


# =============================================================
# CRUD DE USUARIOS (admin only)
# =============================================================

def create_user(
    email: str,
    password: str,
    full_name: str = "",
    role: Optional[str] = None,
) -> Dict[str, Any]:
    """Crea un usuario nuevo via admin API. Auto-confirma email
    porque SMTP todavía no está listo (ciclo 17.16).

    Returns:
        {"ok": True, "user": {...}} si éxito
        {"ok": False, "error": "..."} si falla
    """
    client = get_admin_client()
    if client is None:
        return {"ok": False, "error": "Supabase Auth no configurado en secrets."}

    email_norm = (email or "").strip().lower()
    if not email_norm or "@" not in email_norm:
        return {"ok": False, "error": "Email inválido."}
    if not password or len(password) < 6:
        return {"ok": False, "error": "Password debe tener al menos 6 caracteres."}

    role_final = (role or infer_role_from_email(email_norm)).strip().lower()
    if role_final not in ROLES:
        role_final = infer_role_from_email(email_norm)

    try:
        resp = client.auth.admin.create_user({
            "email": email_norm,
            "password": password,
            "email_confirm": True,  # auto-confirmar (SMTP no está listo)
            "user_metadata": {
                "full_name": (full_name or email_norm).strip(),
                "role": role_final,
            },
        })
        u = getattr(resp, "user", None) or resp
        return {"ok": True, "user": _user_to_dict(u)}
    except Exception as e:
        msg = str(e) or repr(e)
        cls = e.__class__.__name__
        low = msg.lower()
        # Loggear a stderr también para que aparezca en Streamlit Cloud logs
        import sys as _sys
        print(f"[WM_AUTH] create_user FAIL · {cls}: {msg}", file=_sys.stderr, flush=True)
        # Intentar extraer status code y body si es AuthApiError
        status = getattr(e, "status", None) or getattr(e, "code", None) or ""
        body = getattr(e, "message", "") or getattr(e, "msg", "") or ""
        if isinstance(body, dict):
            body = str(body)

        if "already" in low or "duplicate" in low or "exists" in low:
            return {"ok": False, "error": f"Ya existe un usuario con email {email_norm}."}
        if "user not allowed" in low or "not_admin" in low or "forbidden" in low:
            return {"ok": False, "error": (
                f"❌ Permiso denegado por Supabase Auth.\n\n"
                f"**Error:** `{cls}: {msg}`\n"
                f"**Status:** `{status}`\n\n"
                f"Causas posibles:\n"
                f"1. La service_key efectivamente cargada NO es service_role legacy. "
                f"Andá a Streamlit Cloud → app → Manage app → Reboot, y volvé a intentar.\n"
                f"2. Email Provider no está habilitado en el proyecto Supabase.\n"
                f"3. Hay rate limiting (más de 30 creaciones/hora)."
            )}
        return {"ok": False, "error": f"{cls}: {msg} (status={status})"}


def signin_user(email: str, password: str) -> Dict[str, Any]:
    """Autentica usuario con email + password. Devuelve user + session.

    Returns:
        {"ok": True, "user": {...}, "session": {...}} si éxito
        {"ok": False, "error": "..."} con mensaje legible si falla
    """
    client = get_admin_client()
    if client is None:
        return {"ok": False, "error": "Supabase Auth no configurado."}

    email_norm = (email or "").strip().lower()
    if not email_norm or not password:
        return {"ok": False, "error": "Email y password son obligatorios."}

    try:
        resp = client.auth.sign_in_with_password({
            "email": email_norm,
            "password": password,
        })
        u = getattr(resp, "user", None)
        s = getattr(resp, "session", None)
        if not u:
            return {"ok": False, "error": "Email o contraseña incorrectos."}
        return {
            "ok": True,
            "user": _user_to_dict(u),
            "session": _session_to_dict(s),
        }
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "invalid" in low or "credentials" in low:
            return {"ok": False, "error": "Email o contraseña incorrectos."}
        if "banned" in low or "blocked" in low or "ban_duration" in low:
            return {"ok": False, "error": "Tu cuenta está bloqueada. Contactá al administrador."}
        if "not confirmed" in low or "not_confirmed" in low:
            return {"ok": False, "error": "Cuenta no confirmada. Contactá al administrador."}
        return {"ok": False, "error": f"Error de autenticación: {msg}"}


def list_all_users() -> List[Dict[str, Any]]:
    """Lista todos los usuarios del proyecto. Solo admin debería llamar."""
    client = get_admin_client()
    if client is None:
        return []
    try:
        resp = client.auth.admin.list_users()
        # supabase-py puede devolver lista directa o objeto con .users
        if isinstance(resp, list):
            users = resp
        else:
            users = getattr(resp, "users", []) or []
        return [_user_to_dict(u) for u in users if u]
    except Exception:
        return []


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Busca un usuario por email. Lineal sobre list_all_users."""
    e = (email or "").strip().lower()
    if not e:
        return None
    for u in list_all_users():
        if (u.get("email") or "").lower() == e:
            return u
    return None


def update_user_role(user_id: str, new_role: str) -> Dict[str, Any]:
    """Cambia el role de un usuario en user_metadata."""
    client = get_admin_client()
    if client is None:
        return {"ok": False, "error": "Supabase Auth no configurado."}
    if new_role not in ROLES:
        return {"ok": False, "error": f"Role inválido: {new_role}. Válidos: {list(ROLES.keys())}"}
    try:
        # Cargar metadata actual y mergear
        current = client.auth.admin.get_user_by_id(user_id)
        cur_user = getattr(current, "user", None)
        meta = dict(getattr(cur_user, "user_metadata", {}) or {})
        meta["role"] = new_role

        resp = client.auth.admin.update_user_by_id(
            user_id, {"user_metadata": meta}
        )
        u = getattr(resp, "user", None) or resp
        return {"ok": True, "user": _user_to_dict(u)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def update_user_full_name(user_id: str, full_name: str) -> Dict[str, Any]:
    """Cambia el full_name de un usuario."""
    client = get_admin_client()
    if client is None:
        return {"ok": False, "error": "Supabase Auth no configurado."}
    try:
        current = client.auth.admin.get_user_by_id(user_id)
        cur_user = getattr(current, "user", None)
        meta = dict(getattr(cur_user, "user_metadata", {}) or {})
        meta["full_name"] = (full_name or "").strip()
        resp = client.auth.admin.update_user_by_id(
            user_id, {"user_metadata": meta}
        )
        u = getattr(resp, "user", None) or resp
        return {"ok": True, "user": _user_to_dict(u)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def reset_user_password(user_id: str, new_password: str) -> Dict[str, Any]:
    """Resetea password manualmente desde admin (no usa email)."""
    client = get_admin_client()
    if client is None:
        return {"ok": False, "error": "Supabase Auth no configurado."}
    if not new_password or len(new_password) < 6:
        return {"ok": False, "error": "Password debe tener al menos 6 caracteres."}
    try:
        resp = client.auth.admin.update_user_by_id(
            user_id, {"password": new_password}
        )
        return {"ok": True, "user": _user_to_dict(getattr(resp, "user", None) or resp)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def block_user(user_id: str) -> Dict[str, Any]:
    """Banea al usuario por 100 años (efectivamente permanente).
    Supabase soporta ban_duration con formato '<n>h' (horas).
    876000h = 100 años aprox.
    """
    client = get_admin_client()
    if client is None:
        return {"ok": False, "error": "Supabase Auth no configurado."}
    try:
        resp = client.auth.admin.update_user_by_id(
            user_id, {"ban_duration": "876000h"}
        )
        return {"ok": True, "user": _user_to_dict(getattr(resp, "user", None) or resp)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def unblock_user(user_id: str) -> Dict[str, Any]:
    """Desbanea — ban_duration = 'none'."""
    client = get_admin_client()
    if client is None:
        return {"ok": False, "error": "Supabase Auth no configurado."}
    try:
        resp = client.auth.admin.update_user_by_id(
            user_id, {"ban_duration": "none"}
        )
        return {"ok": True, "user": _user_to_dict(getattr(resp, "user", None) or resp)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def delete_user(user_id: str) -> Dict[str, Any]:
    """Elimina permanentemente. ¡Irreversible!"""
    client = get_admin_client()
    if client is None:
        return {"ok": False, "error": "Supabase Auth no configurado."}
    try:
        client.auth.admin.delete_user(user_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# =============================================================
# UTILS
# =============================================================

def generate_temp_password(length: int = 12) -> str:
    """Genera una password temporal segura para nuevos usuarios.
    Mezcla letras + números, sin caracteres ambiguos (0, O, l, 1).
    """
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789"
    return "".join(_secrets_module.choice(alphabet) for _ in range(length))


def _user_to_dict(u: Any) -> Dict[str, Any]:
    """Convierte un user de Supabase Auth a dict serializable."""
    if u is None:
        return {}
    meta = getattr(u, "user_metadata", {}) or {}
    if not isinstance(meta, dict):
        try:
            meta = dict(meta)
        except Exception:
            meta = {}

    email = getattr(u, "email", "") or ""
    role = (meta.get("role") or infer_role_from_email(email)).lower()

    banned_until = getattr(u, "banned_until", None)
    is_blocked = bool(banned_until)

    return {
        "id":              getattr(u, "id", "") or "",
        "email":           email,
        "full_name":       meta.get("full_name", "") or email,
        "role":            role,
        "role_label":      ROLES.get(role, role.capitalize()),
        "created_at":      str(getattr(u, "created_at", "") or ""),
        "last_sign_in_at": str(getattr(u, "last_sign_in_at", "") or ""),
        "banned_until":    str(banned_until or ""),
        "is_blocked":      is_blocked,
        "is_admin":        is_admin_email(email),
    }


def _session_to_dict(s: Any) -> Dict[str, Any]:
    if s is None:
        return {}
    return {
        "access_token":  getattr(s, "access_token", "") or "",
        "refresh_token": getattr(s, "refresh_token", "") or "",
        "expires_at":    int(getattr(s, "expires_at", 0) or 0),
    }


__all__ = [
    "ADMIN_EMAIL",
    "SIGASAS_DOMAIN",
    "ROLES",
    "infer_role_from_email",
    "is_admin_email",
    "is_supabase_auth_enabled",
    "get_admin_client",
    "create_user",
    "signin_user",
    "list_all_users",
    "get_user_by_email",
    "update_user_role",
    "update_user_full_name",
    "reset_user_password",
    "block_user",
    "unblock_user",
    "delete_user",
    "generate_temp_password",
]
