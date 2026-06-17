"""
core.auth_otp — Login passwordless por código de un solo uso (OTP) al correo.
============================================================================

Flujo de 2 pasos:
  1) request_code(email)  → valida que el email esté registrado, genera un
     código de 6 dígitos, lo guarda (solo el hash) en session_state y lo
     envía por correo (core.email_sender).
  2) submit_code(email, code) → verifica el código contra el challenge:
     expiración (10 min), nº de intentos (5) y hash. Si pasa, devuelve el
     dict del usuario para que core.auth arme la sesión.

Diseño:
  - El código NUNCA se guarda en claro: solo HMAC-SHA256 (secreto compartido
    con la cookie de sesión).
  - El challenge vive en st.session_state (sobrevive el flujo de 2 pasos en
    la misma sesión). No requiere tabla en DB.
  - Rate limit: cooldown de reenvío (60 s) y máximo de envíos por ventana.
  - La lógica de generación/hash/verificación es PURA (testeable headless);
    las funciones request_code/submit_code son los wrappers con Streamlit.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any, Dict, Optional, Tuple

# ── Parámetros de política ──────────────────────────────────────────────
OTP_TTL_SECONDS = 10 * 60          # el código es válido 10 minutos
OTP_LENGTH = 6
MAX_VERIFY_ATTEMPTS = 5            # intentos de verificación por código
RESEND_COOLDOWN_SECONDS = 60      # mínimo entre reenvíos
MAX_SENDS_PER_WINDOW = 5          # códigos por ventana
SEND_WINDOW_SECONDS = 30 * 60     # ventana de conteo de envíos

_CHALLENGE_KEY = "_wm_otp_challenge"
_SENDS_KEY = "_wm_otp_sends"


# ── Helpers puros (testeables sin Streamlit) ────────────────────────────
def _secret() -> bytes:
    try:
        from core.session_cookie import _secret as _ck_secret
        return _ck_secret()
    except Exception:
        return b"wm-otp-fallback-secret-change-me"


def _now() -> int:
    return int(time.time())


def generate_code() -> str:
    """Código numérico de OTP_LENGTH dígitos, uniforme y sin sesgo."""
    return f"{secrets.randbelow(10 ** OTP_LENGTH):0{OTP_LENGTH}d}"


def hash_code(code: str, email: str) -> str:
    """HMAC-SHA256 del (email, code). El código jamás se guarda en claro."""
    msg = f"{(email or '').strip().lower()}::{(code or '').strip()}".encode("utf-8")
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()


def mask_email(email: str) -> str:
    """j***@dominio.com — para mostrar a dónde fue el código sin exponerlo."""
    e = (email or "").strip()
    if "@" not in e:
        return e
    local, _, domain = e.partition("@")
    if len(local) <= 1:
        masked = local + "***"
    else:
        masked = local[0] + "***"
    return f"{masked}@{domain}"


def make_challenge(email: str, code: str, now: Optional[int] = None) -> Dict[str, Any]:
    """Construye el challenge (lo que se guarda) para un código recién emitido."""
    t = now if now is not None else _now()
    return {
        "email": (email or "").strip().lower(),
        "code_hash": hash_code(code, email),
        "expires_at": t + OTP_TTL_SECONDS,
        "attempts": 0,
        "sent_at": t,
    }


def verify_challenge(
    challenge: Optional[Dict[str, Any]],
    email: str,
    code: str,
    now: Optional[int] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Verifica un código contra el challenge. PURA.

    Returns: (ok, error, challenge_actualizado). El challenge actualizado
    refleja el incremento de intentos; si ok=True el caller debe descartarlo
    (código de un solo uso)."""
    if not challenge:
        return False, "No hay un código activo. Solicitá uno nuevo.", None

    t = now if now is not None else _now()
    if t >= int(challenge.get("expires_at", 0)):
        return False, "El código expiró. Solicitá uno nuevo.", None

    if (email or "").strip().lower() != challenge.get("email"):
        return False, "El correo no coincide con el código solicitado.", challenge

    attempts = int(challenge.get("attempts", 0))
    if attempts >= MAX_VERIFY_ATTEMPTS:
        return False, "Demasiados intentos. Solicitá un código nuevo.", None

    updated = dict(challenge)
    updated["attempts"] = attempts + 1

    expected = challenge.get("code_hash", "")
    got = hash_code(code, email)
    if not hmac.compare_digest(expected, got):
        restantes = MAX_VERIFY_ATTEMPTS - updated["attempts"]
        if restantes <= 0:
            return False, "Código incorrecto. Demasiados intentos, solicitá uno nuevo.", None
        return False, f"Código incorrecto. Te quedan {restantes} intento(s).", updated

    return True, "", None


# ── Lookup de usuario (sin password) ────────────────────────────────────
def lookup_user(email: str) -> Optional[Dict[str, Any]]:
    """Devuelve el dict del usuario si el email está registrado, o None.
    Prueba Supabase Auth y luego el store legacy."""
    e = (email or "").strip().lower()
    if not e or "@" not in e:
        return None
    try:
        from core.supabase_auth import is_supabase_auth_enabled, get_user_by_email
        if is_supabase_auth_enabled():
            u = get_user_by_email(e)
            if u:
                u = dict(u)
                u.setdefault("source", "supabase")
                return u
    except Exception:
        pass
    try:
        from core.auth import _find_user
        u = _find_user(e)
        if u:
            return {
                "id": "",
                "email": u.get("email", e),
                "full_name": u.get("full_name", u.get("username", "")),
                "role": u.get("role", "viewer"),
                "is_admin": (u.get("role") == "admin"),
                "is_blocked": False,
                "source": "legacy",
            }
    except Exception:
        pass
    return None


# ── Wrappers con Streamlit (session_state + envío de correo) ─────────────
def _ss():
    import streamlit as st
    return st.session_state


def _can_send(now: int) -> Tuple[bool, str]:
    """Rate limit de envíos basado en session_state."""
    ss = _ss()
    sends = [t for t in ss.get(_SENDS_KEY, []) if now - t < SEND_WINDOW_SECONDS]
    ss[_SENDS_KEY] = sends
    if sends and (now - sends[-1]) < RESEND_COOLDOWN_SECONDS:
        wait = RESEND_COOLDOWN_SECONDS - (now - sends[-1])
        return False, f"Esperá {wait}s antes de pedir otro código."
    if len(sends) >= MAX_SENDS_PER_WINDOW:
        return False, "Demasiados códigos solicitados. Intentá más tarde."
    return True, ""


def _record_send(now: int) -> None:
    ss = _ss()
    ss.setdefault(_SENDS_KEY, []).append(now)


def request_code(email: str) -> Dict[str, Any]:
    """Paso 1: valida email registrado, genera y envía el código.

    Returns: {ok, error, masked_email}. Por seguridad NO revela si el email
    existe o no de forma explícita en el mensaje al usuario final (mismo
    texto), pero internamente solo envía si está registrado."""
    e = (email or "").strip().lower()
    now = _now()
    if not e or "@" not in e:
        return {"ok": False, "error": "Ingresá un correo válido.", "masked_email": ""}

    ok_rate, rate_msg = _can_send(now)
    if not ok_rate:
        return {"ok": False, "error": rate_msg, "masked_email": ""}

    user = lookup_user(e)
    # Anti-enumeración: respondemos "enviado" igual, pero solo mandamos correo
    # y guardamos challenge si el usuario existe y no está bloqueado.
    generic_ok = {"ok": True, "error": "", "masked_email": mask_email(e)}
    if not user or user.get("is_blocked"):
        return generic_ok

    code = generate_code()
    _ss()[_CHALLENGE_KEY] = make_challenge(e, code, now)

    try:
        from core.email_sender import send_email
        subject = "Tu código de acceso · Watermelon System"
        body_text = (
            f"Tu código de acceso es: {code}\n\n"
            f"Vence en {OTP_TTL_SECONDS // 60} minutos. "
            f"Si no solicitaste este acceso, ignorá este correo."
        )
        body_html = (
            f"<div style='font-family:sans-serif'>"
            f"<p>Tu código de acceso a <b>Watermelon System</b> es:</p>"
            f"<p style='font-size:30px;font-weight:800;letter-spacing:6px;"
            f"color:#0f172a'>{code}</p>"
            f"<p style='color:#64748b'>Vence en {OTP_TTL_SECONDS // 60} minutos. "
            f"Si no solicitaste este acceso, ignorá este correo.</p></div>"
        )
        res = send_email(e, subject, body_text, body_html)
        if not res.get("ok"):
            # No filtrar detalle del backend al usuario; log queda en email_sender
            return {"ok": False, "error": "No se pudo enviar el código. Intentá más tarde.",
                    "masked_email": ""}
    except Exception:
        return {"ok": False, "error": "No se pudo enviar el código. Intentá más tarde.",
                "masked_email": ""}

    _record_send(now)
    return generic_ok


def submit_code(email: str, code: str) -> Dict[str, Any]:
    """Paso 2: verifica el código. Si pasa, devuelve {ok, user}."""
    e = (email or "").strip().lower()
    ss = _ss()
    challenge = ss.get(_CHALLENGE_KEY)
    ok, err, updated = verify_challenge(challenge, e, code)
    if updated is not None:
        ss[_CHALLENGE_KEY] = updated  # persistir intentos
    else:
        ss.pop(_CHALLENGE_KEY, None)  # consumido o inválido

    if not ok:
        return {"ok": False, "error": err, "user": None}

    user = lookup_user(e)
    if not user or user.get("is_blocked"):
        return {"ok": False, "error": "La cuenta no está disponible.", "user": None}
    return {"ok": True, "error": "", "user": user}


__all__ = [
    "request_code", "submit_code", "lookup_user", "mask_email",
    "generate_code", "hash_code", "make_challenge", "verify_challenge",
    "OTP_TTL_SECONDS", "MAX_VERIFY_ATTEMPTS",
]
