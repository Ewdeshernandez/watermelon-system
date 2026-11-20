"""
core/trend_annotations.py — Notas del analista sobre la tendencia (bitácora).
================================================================================

Anotaciones ancladas a un instante + una máquina: "antes de cambiar bomba",
"cambio de aceite", "parada por proceso", etc. Se dibujan como línea vertical con
etiqueta en las tendencias de Live Monitoring y quedan como bitácora de
intervenciones (auditoría). Tabla Supabase `trend_annotations`.

API:
- add_annotation(instance_id, ts_iso, text, author="") -> id | None
- list_annotations(instance_id, from_iso=None, to_iso=None, limit=200) -> list
- delete_annotation(ann_id) -> bool

Degrada con gracia: si la tabla no existe o no hay cliente, las funciones no
rompen (devuelven None/[]/False) y el caller avisa.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("wm.trend_annotations")

_TABLE = "trend_annotations"


def _client():
    try:
        from core.live_readings import _get_supabase_client
        return _get_supabase_client()
    except Exception:  # noqa: BLE001
        return None


def add_annotation(instance_id: str, ts_iso: str, text: str,
                   author: str = "") -> Optional[Any]:
    """Inserta una nota. Devuelve el id (o True) si ok, None si falla."""
    client = _client()
    if client is None or not instance_id or not (text or "").strip() or not ts_iso:
        return None
    row = {
        "instance_id": instance_id,
        "ts": ts_iso,
        "text": (text or "").strip()[:280],
        "author": (author or "")[:120],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        resp = client.table(_TABLE).insert(row).execute()
        data = getattr(resp, "data", None) or []
        return (data[0].get("id") if data and isinstance(data[0], dict) else True)
    except Exception as e:  # noqa: BLE001
        log.warning("add_annotation failed (¿falta la tabla trend_annotations?): %s", e)
        return None


def list_annotations(instance_id: str, from_iso: Optional[str] = None,
                     to_iso: Optional[str] = None, limit: int = 200
                     ) -> List[Dict[str, Any]]:
    """Notas de la máquina en [from, to], más antigua primero."""
    client = _client()
    if client is None or not instance_id:
        return []
    try:
        q = client.table(_TABLE).select("*").eq("instance_id", instance_id)
        if from_iso:
            q = q.gte("ts", from_iso)
        if to_iso:
            q = q.lte("ts", to_iso)
        resp = q.order("ts", desc=False).limit(limit).execute()
        return list(getattr(resp, "data", None) or [])
    except Exception as e:  # noqa: BLE001
        log.warning("list_annotations failed: %s", e)
        return []


def delete_annotation(ann_id: Any) -> bool:
    client = _client()
    if client is None or ann_id is None:
        return False
    try:
        client.table(_TABLE).delete().eq("id", ann_id).execute()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("delete_annotation failed: %s", e)
        return False
