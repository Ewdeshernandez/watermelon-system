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
import sys
import time
from pathlib import Path
from typing import Optional, Dict

def planta_data_dir() -> Path:
    """Directorio de datos PERSISTENTE del Planta (single source of truth).

    En el .exe congelado va JUNTO al ejecutable (no en el temporal _MEIPASS,
    que se borra al cerrar). Igual que license_manager/updater. Antes esto
    usaba Path(__file__).parent y en el .exe caía al _MEIPASS → la sesión de
    login NO persistía y el sync escaneaba una carpeta vacía (nunca aparecía
    el botón 'Sync ahora')."""
    _env = os.environ.get("WATERMELON_DATA_DIR")
    if _env:
        return Path(_env)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "data"
    return Path(__file__).parent / "data"


_AUTH_FILE = planta_data_dir() / ".auth.json"


def _read_bundled_secrets() -> tuple[Optional[str], Optional[str]]:
    """Lee url + anon_key de un secrets.toml EMPAQUETADO en el .exe.

    En el .exe congelado (PyInstaller onefile) NO hay variables de entorno
    ni st.secrets: st.secrets busca .streamlit/secrets.toml relativo al CWD,
    que no existe. Por eso el login OTP fallaba en campo con
    "SUPABASE_URL y SUPABASE_ANON_KEY no configurados".

    El build (CI) escribe planta/.streamlit/secrets.toml con las claves
    PÚBLICAS del proyecto (url + anon_key) y el .spec lo empaqueta. Acá lo
    leemos DIRECTO por ruta, de forma determinista, probando varias
    ubicaciones candidatas (frozen y dev). Nunca contiene el service_key.
    """
    candidates = []
    # 1) Junto a este módulo: <bundle>/planta/.streamlit/secrets.toml
    candidates.append(Path(__file__).resolve().parent / ".streamlit" / "secrets.toml")
    # 2) En el root del bundle PyInstaller (_MEIPASS) cuando está frozen
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "planta" / ".streamlit" / "secrets.toml")
        candidates.append(Path(meipass) / ".streamlit" / "secrets.toml")
    # 3) Relativo al directorio de trabajo actual
    candidates.append(Path.cwd() / ".streamlit" / "secrets.toml")

    for path in candidates:
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            url, key = _parse_supabase_toml(text)
            if url and key:
                return url, key
        except Exception:  # noqa: BLE001 — un secrets.toml roto no debe tumbar el login
            continue
    return None, None


def _parse_supabase_toml(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extrae supabase.url + supabase.anon_key de un secrets.toml.

    Usa tomllib/toml si están disponibles; si no (p.ej. Python <3.11 sin el
    paquete toml), cae a un mini-parser manual — el archivo que genera el CI
    es trivial (una sección [supabase] con dos líneas key = "valor")."""
    # 1) Parser TOML de verdad si está a mano.
    for _mod in ("tomllib", "toml", "tomli"):
        try:
            _p = __import__(_mod)
            data = _p.loads(text)
            sup = data.get("supabase", {}) or {}
            u, k = sup.get("url"), sup.get("anon_key")
            if u and k:
                return u, k
        except Exception:  # noqa: BLE001
            continue
    # 2) Mini-parser manual (sin dependencias).
    url = key = None
    in_supabase = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_supabase = line.lower().replace(" ", "") == "[supabase]"
            continue
        if not in_supabase or "=" not in line:
            continue
        name, _, val = line.partition("=")
        name = name.strip().lower()
        val = val.split("#", 1)[0].strip().strip('"').strip("'")
        if name == "url":
            url = val
        elif name == "anon_key":
            key = val
    return url, key


def _get_supabase_credentials() -> tuple[str, str]:
    """Lee SUPABASE_URL y ANON_KEY de env vars, st.secrets o secrets.toml
    empaquetado (este último es el que hace funcionar el login en el .exe)."""
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

    # Fallback determinista para el .exe congelado (sin env ni st.secrets).
    if not url or not key:
        b_url, b_key = _read_bundled_secrets()
        url = url or b_url
        key = key or b_key

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
# Login por CÓDIGO OTP (v3.31.420+, Plan B) — UNIFICADO con la app principal.
# Ya NO usamos el OTP nativo de Supabase (dependía de la plantilla de email
# "Magic link or OTP", que no incluía {{ .Token }} → el código no llegaba).
# Ahora llamamos a la Edge Function `planta-auth`, que corre server-side con
# el service_role: genera el código, lo envía por NUESTRO correo (Microsoft
# Graph, ehernandez@sigasas.com) y, al verificar, acuña una SESIÓN Supabase
# (JWT + refresh) que devolvemos acá. El client_secret de Graph vive solo en
# el servidor, nunca en el .exe. sync_uploader sigue subiendo con ese JWT.
# ---------------------------------------------------------------------------
def _planta_auth_endpoint() -> str:
    url, _ = _get_supabase_credentials()
    return url.rstrip("/") + "/functions/v1/planta-auth"


def _post_planta_auth(payload: Dict) -> Dict:
    """POST JSON a la Edge Function planta-auth. Devuelve el body parseado
    (incluye 'ok' o 'error'). Levanta RuntimeError solo si no hay red."""
    import urllib.request
    import urllib.error

    url, key = _get_supabase_credentials()
    endpoint = url.rstrip("/") + "/functions/v1/planta-auth"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("apikey", key)
    req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {"error": f"http_{exc.code}"}
    except Exception as exc:  # noqa: BLE001 — red caída / DNS / timeout
        raise RuntimeError(
            f"No se pudo contactar el servidor de acceso: {exc}. "
            "Verifica tu conexión a internet."
        ) from exc


def _save_session_dict(res: Dict) -> Dict:
    """Persiste en planta/data/.auth.json la sesión devuelta por la Edge
    Function (mismo formato de siempre)."""
    if not res.get("access_token"):
        raise RuntimeError("El servidor no devolvió un token de sesión válido")
    data = {
        "email": res.get("email"),
        "user_id": res.get("user_id"),
        "access_token": res["access_token"],
        "refresh_token": res.get("refresh_token", ""),
        "expires_at": res.get("expires_at"),
        "logged_in_at": int(time.time()),
    }
    _AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _AUTH_FILE.write_text(json.dumps(data, indent=2))
    return data


def request_login_code(email: str) -> None:
    """Pide a la Edge Function que envíe el código OTP al email (por Graph).

    Requiere internet. Por seguridad la función responde OK aunque el correo
    no exista (no filtra qué correos están registrados); solo envía el código
    a usuarios ya registrados en Watermelon Cloud."""
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("Email inválido")
    res = _post_planta_auth({"action": "request", "email": email})
    if not res.get("ok"):
        raise RuntimeError(
            "No se pudo enviar el código "
            f"({res.get('error', 'error desconocido')}). Reintenta en un momento."
        )


def verify_login_code(email: str, code: str) -> Dict:
    """Verifica el código OTP contra la Edge Function y guarda la sesión JWT
    (planta/data/.auth.json).

    Returns: dict con email + access_token + expires_at."""
    email = (email or "").strip().lower()
    code = (code or "").strip().replace(" ", "")
    if not email or not code:
        raise ValueError("Email y código requeridos")
    res = _post_planta_auth({"action": "verify", "email": email, "code": code})
    if not res.get("ok"):
        err = res.get("error", "desconocido")
        msg = {
            "invalid_code": "Código incorrecto.",
            "invalid_code_format": "El código debe ser de 6 dígitos.",
            "code_expired": "El código venció. Pide uno nuevo.",
            "too_many_attempts": "Demasiados intentos. Pide un código nuevo.",
            "no_challenge": "No hay un código pendiente para ese correo. Pide uno nuevo.",
        }.get(err, f"No se pudo verificar el código ({err}). Pide uno nuevo.")
        raise RuntimeError(msg)
    return _save_session_dict(res)


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
