"""
core/machines_meta.py
=====================

Ficha de la máquina desde **Machines** (fuente única) para encabezados de Live,
reportes y app: tipo, cliente, ubicación, tag, rpm nominal, norma. Caché corto
para no consultar la nube en cada refresco de 10 s.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

_CACHE: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
_TTL = 60.0


def machine_header(instance_id: str, client: Any = None) -> Optional[Dict[str, Any]]:
    now = time.time()
    hit = _CACHE.get(instance_id)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    out: Optional[Dict[str, Any]] = None
    try:
        from core.remote_monitoring.config import find_setup_cloud_by_instance
        if client is None:
            from core.live_readings import _get_supabase_client
            client = _get_supabase_client()
        found = find_setup_cloud_by_instance(instance_id, client=client)
        if found:
            m = found["setup"].machine
            out = {"type": (m.machine_type or "").strip(), "client": m.client, "location": m.location,
                   "tag": m.tag or m.name, "rpm_nominal": m.rpm_nominal,
                   "iso_norm": m.iso_norm, "iso_class": m.iso_class,
                   "version": found.get("version")}
    except Exception:  # noqa: BLE001
        out = None
    _CACHE[instance_id] = (now, out)
    return out
