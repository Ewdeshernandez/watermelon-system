"""
planta/auth_planta.py — Auth con Supabase, JWT cacheado local
==============================================================

Maneja login del técnico al Watermelon Cloud (vía Supabase Auth) y guarda
el token JWT en disco para que sync_uploader pueda subir TDMS sin pedir
re-login cada vez.

Flujo:
1. Usuario hace login UNA VEZ (necesita internet ese momento)
2. JWT se guarda en planta/data/.auth.json
3. Sync uploader lee el JWT y lo usa para autenticar al subir
4. Si el JWT vence, intenta refresh con el refresh_token (sin pedirle al user)
5. Si el refresh falla (sin internet o token muerto), pide re-login

Las credenciales del proyecto (SUPABASE_URL + SUPABASE_ANON_KEY) se leen
de:
1. Variables de entorno SUPABASE_URL y SUPABASE_ANON_KEY
2. st.secrets["supabase"] si está disponible
3. planta/.streamlit/secrets.toml (formato Streamlit estándar)

El ANON_KEY es público por diseño (lo expone el frontend de Watermelon Cloud
todo el tiempo). Lo que protege los datos es RLS (Row Level Security) sobre
el bucket — cada user solo puede leer/escribir SUS propios TDMS.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Dict

_AUTH_FILE = Path(__file__).parent / "data" / ".auth.json"


def _get_supabase_credentials() -> tuple[str, str]:
    """Lee SUPABASE_URL y ANON_KEY de env vars o st.secrets."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "supabase" in st.secrets:
                url = url or st.secrets["supabase"].get("url")
                key = key or st.secrets["supabase"].get("anon_key")
        except (ImportError, FileNotFoundError, KeyError):
            pass

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL y SUPABASE_ANON_KEY no configurados. "
            "Crea planta/.streamlit/secrets.toml con:\n"
            "[supabase]\n"
            'url = "https://TU-PROYECTO.supabase.co"\n'
            'anon_key = "eyJ..."\n'
            "(las claves del proyecto Watermelon Cloud — son públicas, "
            "el ANON_KEY está OK distribuirlo)"
        )
    return url, key


def _get_supabase_client():
    """Crea cliente Supabase. Lazy import para que el resto del módulo
    funcione sin supabase-py instalado (e.g. si solo se quiere current_user)."""
    try:
        from supabase import create_client
    except ImportError as exc:
        raise ImportError(
            "supabase-py no instalado. Corre: pip install supabase"
        ) from exc

    url, key = _get_supabase_credentials()
    return create_client(url, key)


def _save_session(resp, email: str) -> Dict:
    """Persiste la sesión Supabase en planta/data/.auth.json (JWT + refresh)."""
    if not resp.session or not resp.session.access_token:
        raise RuntimeError("Supabase no devolvió un session token válido")
    data = {
        "email": resp.user.email if resp.user else email,
        "user_id": resp.user.id if resp.user else None,
        "access_token": resp.session.access_token,
        "refresh_token": resp.session.refresh_token,
        "expires_at": resp.session.expires_at,
        "logged_in_at": int(time.time()),
    }
    _AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _AUTH_FILE.write_text(json.dumps(data, indent=2))
    return data


# ---------------------------------------------------------------------------
# Login por CÓDIGO OTP (v3.31.398) — igual que Watermelon Cloud: ya no hay
# passwords. Supabase manda un código de 6 dígitos al email del técnico;
# verify_otp devuelve la MISMA sesión JWT que antes (RLS intacto).
# ---------------------------------------------------------------------------
def request_login_code(email: str) -> None:
    """Pide a Supabase que envíe el código OTP al email. Requiere internet.

    should_create_user=False: solo usuarios ya registrados en Watermelon
    Cloud pueden loguearse desde Planta."""
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("Email inválido")
    client = _get_supabase_client()
    try:
        client.auth.sign_in_with_otp({
            "email": email,
            "options": {"should_create_user": False},
        })
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo enviar el código: {exc}. Verifica el email (debe "
            f"existir en Watermelon Cloud) y la conexión a internet."
        ) from exc


def verify_login_code(email: str, code: str) -> Dict:
    """Verifica el código OTP y guarda la sesión JWT (planta/data/.auth.json).

    Returns: dict con email + access_token + expires_at."""
    email = (email or "").strip().lower()
    code = (code or "").strip().replace(" ", "")
    if not email or not code:
        raise ValueError("Email y código requeridos")
    client = _get_supabase_client()
    try:
        resp = client.auth.verify_otp({
            "email": email, "token": code, "type": "email",
        })
    except Exception as exc:
        raise RuntimeError(
            f"Código inválido o vencido: {exc}. Pide un código nuevo."
        ) from exc
    return _save_session(resp, email)


def login(email: str, password: str) -> Dict:
    """
    Hace login al Watermelon Cloud y guarda JWT en planta/data/.auth.json.

    Returns:
        dict con email + access_token + expires_at (timestamp unix)
    Raises:
        RuntimeError si las credenciales son incorrectas o sin internet
    """
    if not email or not password:
        raise ValueError("Email y password requeridos")

    client = _get_supabase_client()
    try:
        resp = client.auth.sign_in_with_password({
            "email": email.strip(),
            "password": password,
        })
    except Exception as exc:
        raise RuntimeError(
            f"Login fallo: {exc}. Verifica email/password o conexión a internet."
        ) from exc

    if not resp.session or not resp.session.access_token:
        raise RuntimeError("Supabase no devolvió un session token válido")

    data = {
        "email": resp.user.email if resp.user else email,
        "user_id": resp.user.id if resp.user else None,
        "access_token": resp.session.access_token,
        "refresh_token": resp.session.refresh_token,
        "expires_at": resp.session.expires_at,
        "logged_in_at": int(time.time()),
    }

    _AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _AUTH_FILE.write_text(json.dumps(data, indent=2))
    return data


def logout() -> None:
    """Borra el JWT cacheado. Útil para forzar re-login o cambiar de user."""
    if _AUTH_FILE.exists():
        _AUTH_FILE.unlink()


def current_user() -> Optional[Dict]:
    """
    Devuelve el user actualmente logueado (con token válido) o None.

    Si el access_token venció pero hay refresh_token, intenta refresh
    automático. Si el refresh falla (sin internet o token muerto), devuelve
    None — el caller debe pedirle al user que haga login otra vez.
    """
    if not _AUTH_FILE.exists():
        return None
    try:
        data = json.loads(_AUTH_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Si el token está válido (con margen de 60s), devolver
    expires_at = data.get("expires_at", 0)
    if expires_at and time.time() < expires_at - 60:
        return {
            "email": data["email"],
            "user_id": data.get("user_id"),
            "access_token": data["access_token"],
            "expires_at": expires_at,
        }

    # Token venció — intentar refresh
    refresh_token = data.get("refresh_token")
    if not refresh_token:
        return None
    try:
        client = _get_supabase_client()
        new_resp = client.auth.refresh_session(refresh_token)
        if not new_resp.session:
            return None
        data["access_token"] = new_resp.session.access_token
        data["refresh_token"] = new_resp.session.refresh_token
        data["expires_at"] = new_resp.session.expires_at
        _AUTH_FILE.write_text(json.dumps(data, indent=2))
        return {
            "email": data["email"],
            "user_id": data.get("user_id"),
            "access_token": data["access_token"],
            "expires_at": data["expires_at"],
        }
    except Exception:
        # Sin internet o refresh fallo → user debe re-login cuando vuelva la red
        return None


def is_token_valid() -> bool:
    """Check rápido si hay un user logueado con token válido. Sin intentar refresh."""
    if not _AUTH_FILE.exists():
        return False
    try:
        data = json.loads(_AUTH_FILE.read_text())
        return bool(
            data.get("access_token")
            and time.time() < data.get("expires_at", 0) - 60
        )
    except Exception:  # noqa: BLE001
        return False
