"""Cuenta demo de la plataforma: máquina "espejo" + accesos que caducan.

Máquina espejo: una instance demo (p. ej. "demo_tg01", tag "TG-DEMO-01",
cliente "Cliente Demo", in_service=False → fuera de reportes y alarmas) que al
LEER datos toma los de una máquina real (wm_billing_settings.demo_alias =
{demo_id: source_id}). No se copia ni se escribe nada: las funciones de lectura
(live_readings, severity_events, Machines) traducen el id. La pantalla solo
conoce los nombres de la instance demo, así que el cliente real no aparece.

Accesos demo: wm_billing_settings.demo_access = {email: {"until": iso, "name"}}.
Vencido el plazo, el usuario ya no entra (auth.require_login lo detiene).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

ALIAS_KEY = "demo_alias"
ACCESS_KEY = "demo_access"
DEMO_DAYS = 14
_TTL = 60.0
_cache: Dict[str, tuple] = {}


def _setting(key: str, default):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    try:
        from core.billing.store import get_setting
        val = get_setting(key, default)
    except Exception:  # noqa: BLE001
        val = default
    _cache[key] = (time.time(), val)
    return val


def clear_cache() -> None:
    _cache.clear()


def aliases() -> Dict[str, str]:
    v = _setting(ALIAS_KEY, {}) or {}
    return v if isinstance(v, dict) else {}


def source_for(instance_id: str) -> str:
    """Id real de donde LEER datos (la misma si no es una máquina demo)."""
    if not instance_id:
        return instance_id
    return aliases().get(instance_id, instance_id)


def is_demo(instance_id: str) -> bool:
    return bool(instance_id) and instance_id in aliases()


# ------------------------------------------------------------------ accesos

def access_list() -> Dict[str, dict]:
    v = _setting(ACCESS_KEY, {}) or {}
    return v if isinstance(v, dict) else {}


def demo_status(email: str, now: Optional[datetime] = None) -> Optional[str]:
    """None si no es usuario demo; 'activo' o 'vencido'."""
    rec = access_list().get((email or "").strip().lower())
    if not rec:
        return None
    now = now or datetime.now(timezone.utc)
    try:
        until = datetime.fromisoformat(str(rec.get("until")).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return "vencido"
    return "activo" if now <= until else "vencido"


def grant(email: str, name: str = "", days: int = DEMO_DAYS) -> dict:
    """Registra/renueva un acceso demo por `days` días."""
    from core.billing.store import set_setting
    email = (email or "").strip().lower()
    acc = dict(access_list())
    until = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(timespec="seconds")
    acc[email] = {"until": until, "name": name, "granted": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    set_setting(ACCESS_KEY, acc)
    clear_cache()
    return acc[email]


def revoke(email: str) -> None:
    from core.billing.store import set_setting
    acc = dict(access_list())
    acc.pop((email or "").strip().lower(), None)
    set_setting(ACCESS_KEY, acc)
    clear_cache()
