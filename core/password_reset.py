"""
core.password_reset
===================

Sistema de recovery de password con tokens TTL (Ciclo 17.16).

Flujo:
  1. Usuario click "Olvidé mi contraseña" → ingresa email
  2. request_reset(email) → si el email existe en Supabase Auth:
       - genera token UUID v4
       - guarda en data/password_reset_tokens/{token}.json con TTL 1h
       - envía email vía core.email_sender con el link de reset
     Si NO existe, igual devolvemos {"ok": True} sin enviar nada
     (para evitar enumeration attacks).
  3. Usuario hace click en el link del email → llega a /reset_password?token=xxx
  4. consume_token(token, new_password) → valida + cambia pwd en Supabase
     + invalida el token (one-shot use).

Storage:
  data/password_reset_tokens/{token}.json contiene:
    {
      "token": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "email": "user@sigasas.com",
      "user_id": "uuid de Supabase Auth",
      "issued_at": "2026-05-03T22:30:15",
      "expires_at": "2026-05-03T23:30:15",
      "consumed": false,
      "consumed_at": ""
    }

Cleanup:
  cleanup_expired_tokens() borra los archivos de tokens cuya expiración
  haya pasado hace más de N días. Idempotente. Se invoca al inicio de
  cada request_reset para mantener limpia la carpeta.

API pública:
  - request_reset(email, base_url) → genera token + envía email
  - validate_token(token) → devuelve dict con email/válido/expired
  - consume_token(token, new_password) → cambia pwd en Supabase
  - cleanup_expired_tokens(max_age_days=7)
"""

from __future__ import annotations

import json
import secrets
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TOKENS_DIR = DATA_DIR / "password_reset_tokens"

DEFAULT_TTL_MINUTES = 60   # 1 hora
CLEANUP_GRACE_DAYS = 7     # mantener tokens consumidos/expirados N días por audit


# =============================================================
# UTILS
# =============================================================

def _ensure_dir() -> None:
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)


def _token_path(token: str) -> Path:
    """Devuelve el path al JSON del token. Sanitiza por las dudas."""
    safe = "".join(c for c in token if c.isalnum() or c in "-_")[:64]
    return TOKENS_DIR / f"{safe}.json"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _is_expired(record: Dict[str, Any]) -> bool:
    try:
        exp = datetime.fromisoformat(record.get("expires_at", ""))
        return datetime.now() >= exp
    except Exception:
        return True


# =============================================================
# CLEANUP
# =============================================================

def cleanup_expired_tokens(max_age_days: int = CLEANUP_GRACE_DAYS) -> int:
    """Borra tokens expirados/consumidos cuya expires_at sea más vieja
    que `max_age_days` días. Devuelve cantidad eliminada.
    """
    if not TOKENS_DIR.exists():
        return 0
    cutoff = datetime.now() - timedelta(days=max_age_days)
    n = 0
    for p in TOKENS_DIR.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            exp = datetime.fromisoformat(rec.get("expires_at", ""))
            if exp < cutoff:
                p.unlink()
                n += 1
        except Exception:
            # Si el archivo está corrupto y es viejo, también lo borramos
            try:
                if p.stat().st_mtime < cutoff.timestamp():
                    p.unlink()
                    n += 1
            except Exception:
                pass
    return n


# =============================================================
# REQUEST RESET — generar token + enviar email
# =============================================================

def request_reset(
    email: str,
    base_url: str = "",
    ttl_minutes: int = DEFAULT_TTL_MINUTES,
) -> Dict[str, Any]:
    """Inicia el flujo de reset para `email`.

    IMPORTANTE: por seguridad (evitar email enumeration), siempre
    devolvemos {"ok": True, "message": "..."} sin importar si el
    email existe o no. Solo si EXISTE, generamos token + enviamos
    email. Si no, no hacemos nada visible.

    Args:
        email:        email del usuario que olvidó password
        base_url:     URL base de la app, ej.
                      "https://wm-home-final-2026.streamlit.app"
                      Se usa para construir el link del email.
        ttl_minutes:  validez del token en minutos (default 60)

    Returns:
        {"ok": True, "message": "..."} (siempre, por seguridad)
        En caso de error de config (ej. email backend no configurado),
        devolvemos {"ok": False, "error": "..."} para que el admin
        pueda diagnosticar — pero ese error NO se debería mostrar al
        usuario final.
    """
    _ensure_dir()
    cleanup_expired_tokens()

    email_norm = (email or "").strip().lower()
    if not email_norm or "@" not in email_norm:
        return {"ok": False, "error": "Email inválido."}

    # Buscar usuario en Supabase Auth
    try:
        from core.supabase_auth import get_user_by_email
        user = get_user_by_email(email_norm)
    except Exception as e:
        return {"ok": False, "error": f"No se pudo consultar Supabase: {e}"}

    if not user:
        # No revelamos que el email no existe — silencio total, ok=True
        return {
            "ok": True,
            "message": (
                "Si el email está registrado, recibirás instrucciones para "
                "restablecer tu contraseña en los próximos minutos."
            ),
            "_debug": "email_not_found",
        }

    if user.get("is_blocked"):
        # Misma respuesta genérica
        return {
            "ok": True,
            "message": (
                "Si el email está registrado, recibirás instrucciones para "
                "restablecer tu contraseña en los próximos minutos."
            ),
            "_debug": "user_blocked",
        }

    # Generar token y persistir
    token = str(uuid.uuid4()) + "-" + secrets.token_urlsafe(8)
    issued = datetime.now()
    expires = issued + timedelta(minutes=ttl_minutes)
    record = {
        "token":      token,
        "email":      email_norm,
        "user_id":    user.get("id", ""),
        "full_name":  user.get("full_name", ""),
        "issued_at":  issued.isoformat(timespec="seconds"),
        "expires_at": expires.isoformat(timespec="seconds"),
        "consumed":   False,
        "consumed_at": "",
    }
    try:
        _token_path(token).write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        return {"ok": False, "error": f"No se pudo persistir token: {e}"}

    # Construir URL del link
    base = (base_url or "").rstrip("/")
    if not base:
        base = "https://wm-home-final-2026.streamlit.app"
    token_url = f"{base}/reset_password?token={token}"

    # Enviar email
    try:
        from core.email_sender import send_password_reset_email
        send_res = send_password_reset_email(
            email=email_norm,
            token_url=token_url,
            full_name=user.get("full_name", ""),
            ttl_minutes=ttl_minutes,
        )
    except Exception as e:
        return {"ok": False, "error": f"Error invocando email_sender: {e}"}

    if not send_res.get("ok"):
        # Igual devolvemos ok=True al usuario por seguridad, pero
        # logueamos el error completo a stderr para que aparezca en
        # los logs de Streamlit Cloud y el admin pueda diagnosticar.
        import sys as _sys
        err_full = send_res.get("error", "?")
        print(
            f"[WM_RESET] email_send_failed · email={email_norm} · "
            f"backend_error: {err_full}",
            file=_sys.stderr, flush=True,
        )
        return {
            "ok": True,  # cara al usuario (anti-enumeration)
            "message": (
                "Si el email está registrado, recibirás instrucciones para "
                "restablecer tu contraseña en los próximos minutos."
            ),
            "_debug": f"email_send_failed: {err_full}",
        }

    return {
        "ok": True,
        "message": (
            "Te enviamos un email con instrucciones para restablecer tu "
            "contraseña. El link es válido por 1 hora."
        ),
    }


# =============================================================
# VALIDATE TOKEN
# =============================================================

def validate_token(token: str) -> Dict[str, Any]:
    """Valida un token. Devuelve {valid, email, expired, consumed}."""
    if not token:
        return {"valid": False, "error": "Token vacío."}
    p = _token_path(token)
    if not p.exists():
        return {"valid": False, "error": "Token inválido o ya usado."}
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"valid": False, "error": "Token corrupto."}

    if rec.get("consumed"):
        return {"valid": False, "error": "Este link ya fue usado."}
    if _is_expired(rec):
        return {"valid": False, "error": "El link expiró. Pedí uno nuevo."}

    return {
        "valid": True,
        "email": rec.get("email", ""),
        "user_id": rec.get("user_id", ""),
        "full_name": rec.get("full_name", ""),
        "expires_at": rec.get("expires_at", ""),
    }


# =============================================================
# CONSUME TOKEN — cambiar password + invalidar
# =============================================================

def consume_token(token: str, new_password: str) -> Dict[str, Any]:
    """Si el token es válido, cambia la password del user y marca el
    token como consumido (one-shot use).
    """
    if not new_password or len(new_password) < 8:
        return {"ok": False, "error": "La nueva password debe tener al menos 8 caracteres."}

    val = validate_token(token)
    if not val.get("valid"):
        return {"ok": False, "error": val.get("error", "Token inválido.")}

    user_id = val.get("user_id", "")
    if not user_id:
        return {"ok": False, "error": "Token sin user_id asociado."}

    # Cambiar password en Supabase Auth
    try:
        from core.supabase_auth import reset_user_password
        res = reset_user_password(user_id, new_password)
    except Exception as e:
        return {"ok": False, "error": f"Error al cambiar password: {e}"}

    if not res.get("ok"):
        return {"ok": False, "error": res.get("error", "Falló el cambio en Supabase.")}

    # Marcar token como consumido (one-shot)
    try:
        p = _token_path(token)
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["consumed"] = True
        rec["consumed_at"] = _now_iso()
        p.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass  # no bloqueante

    return {
        "ok": True,
        "email": val.get("email", ""),
        "message": "Password actualizada correctamente. Ya podés iniciar sesión.",
    }


__all__ = [
    "request_reset",
    "validate_token",
    "consume_token",
    "cleanup_expired_tokens",
    "DEFAULT_TTL_MINUTES",
]
