"""
core/severity_events.py — Registro PERSISTENTE de cruces de umbral (Event log).
================================================================================

El "Event List" de System1 es un HISTORIAL permanente: cuándo cada canal entró
en Alarma/Danger, cuánto duró, cuándo volvió a Normal, y quién lo reconoció.
Aquí lo replicamos sobre una tabla `severity_events` en Supabase.

- record_events(instance_id, rendered_rows): compara el estado ACTUAL por canal
  contra el ÚLTIMO estado registrado; inserta un evento solo cuando CAMBIA
  (idempotente entre viewers/cron — no duplica).
- list_recent(instance_id, limit): últimos N eventos (más nuevo primero), con
  duración en estado calculada.
- ack_event(event_id, user_email): marca el evento como reconocido.

Severidad canónica: 'Normal' | 'Alarma' | 'Danger' (capa de display en EN).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("wm.severity_events")

_TABLE = "severity_events"
_RANK = {"Normal": 0, "Alarma": 1, "Danger": 2}


def _client():
    try:
        from core.live_readings import _get_supabase_client
        return _get_supabase_client()
    except Exception:  # noqa: BLE001
        return None


def _last_status_by_sensor(client, instance_id: str) -> Dict[str, str]:
    """Último to_status registrado por canal (para detectar cambios)."""
    try:
        resp = (client.table(_TABLE)
                .select("sensor_label,to_status,crossed_at")
                .eq("instance_id", instance_id)
                .order("crossed_at", desc=True)
                .limit(400).execute())
        out: Dict[str, str] = {}
        for r in (getattr(resp, "data", None) or []):
            sl = r.get("sensor_label")
            if sl and sl not in out:
                out[sl] = r.get("to_status") or "Normal"
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("last_status query failed: %s", e)
        return {}


def record_events(instance_id: str, rendered_rows: List[Dict[str, Any]]) -> int:
    """Inserta un evento por cada canal cuyo estado cambió vs el último registro.
    Devuelve cuántos eventos nuevos insertó. No inserta la primera vez que ve un
    canal en Normal (evita ruido de arranque); sí registra el primer cruce real."""
    client = _client()
    if client is None or not rendered_rows:
        return 0
    last = _last_status_by_sensor(client, instance_id)
    now_iso = datetime.now(timezone.utc).isoformat()
    new_rows = []
    for r in rendered_rows:
        cur = r.get("status")
        if cur not in _RANK:
            continue
        sl = r.get("sensor_label")
        if not sl:
            continue
        prev = last.get(sl)
        # Primera vez que vemos el canal: solo registrar si ya arranca en alarma.
        if prev is None:
            if cur == "Normal":
                continue
            prev = "Normal"
        if cur == prev:
            continue
        try:
            val = float(r.get("value")) if r.get("value") is not None else None
        except (TypeError, ValueError):
            val = None
        new_rows.append({
            "instance_id": instance_id,
            "sensor_label": sl,
            "variable": r.get("variable"),
            "from_status": prev,
            "to_status": cur,
            "value": val,
            "unit": r.get("unit"),
            "alarm": r.get("alarm_used"),
            "danger": r.get("danger_used"),
            "crossed_at": now_iso,
        })
    if not new_rows:
        return 0
    try:
        client.table(_TABLE).insert(new_rows).execute()
        return len(new_rows)
    except Exception as e:  # noqa: BLE001
        log.warning("insert events failed (¿falta la tabla severity_events?): %s", e)
        return 0


def _fmt_dur(seconds: float) -> str:
    s = int(max(seconds, 0))
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m"
    h = m // 60
    if h < 24:
        return f"{h}h {m % 60}m"
    d = h // 24
    return f"{d}d {h % 24}h"


def list_recent(instance_id: str, limit: int = 40) -> List[Dict[str, Any]]:
    """Últimos eventos (más nuevo primero) con duración en estado. Para el que
    sigue vigente (último de su canal y no-Normal) la duración es hasta AHORA."""
    client = _client()
    if client is None:
        return []
    try:
        resp = (client.table(_TABLE)
                .select("*")
                .eq("instance_id", instance_id)
                .order("crossed_at", desc=True)
                .limit(limit).execute())
        rows = list(getattr(resp, "data", None) or [])
    except Exception as e:  # noqa: BLE001
        log.warning("list_recent failed: %s", e)
        return []

    now = datetime.now(timezone.utc)

    def _parse(ts):
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return None

    # duración: hasta el SIGUIENTE evento del mismo canal (rows viene desc)
    next_by_sensor: Dict[str, datetime] = {}
    out: List[Dict[str, Any]] = []
    for r in rows:                       # más nuevo → más viejo
        sl = r.get("sensor_label")
        t0 = _parse(r.get("crossed_at"))
        t_end = next_by_sensor.get(sl)   # evento posterior ya visto
        active = t_end is None and (r.get("to_status") != "Normal")
        if t0 is not None:
            end = t_end or now
            r["_duration"] = _fmt_dur((end - t0).total_seconds())
            r["_age"] = _fmt_dur((now - t0).total_seconds())
        else:
            r["_duration"] = "—"
            r["_age"] = "—"
        r["_active"] = active
        out.append(r)
        if sl and t0 is not None:
            next_by_sensor[sl] = t0
    return out


def ack_event(event_id: Any, user_email: str) -> bool:
    client = _client()
    if client is None:
        return False
    try:
        client.table(_TABLE).update({
            "ack_by": user_email,
            "ack_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", event_id).execute()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("ack_event failed: %s", e)
        return False


__all__ = ["record_events", "list_recent", "ack_event"]
