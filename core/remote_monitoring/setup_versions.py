"""
core/remote_monitoring/setup_versions.py
========================================

Versionado de la config de máquina (Machines / AcqSetup) — Fase 2.

    rm_setups          → puntero a la versión PUBLICADA (collector, nodo, campo)
    rm_setup_versions  → cada versión: draft | published | superseded | rejected

Reglas (acordadas con el usuario, ver mockup Machines):
  * Web y campo pueden construir. Todo cambio = nueva versión (nunca se edita
    una versión publicada).
  * Máquina EN LÍNEA: lo que sube el campo entra como BORRADOR; se acepta en la
    web y recién ahí el nodo lo aplica.
  * Sin sobrescritura silenciosa: publicar exige que el borrador parta de la
    publicada actual (base_version). Si no → conflicto con diff.
  * El nodo en línea confirma "aplicó vN" o rechaza (hardware no cuadra).

La lógica pura (siguiente versión, conflicto, diff) no toca red → testeable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.remote_monitoring.config import (
    AcqSetup,
    _RM_SETUPS_TABLE,
    _rm_client,
    _slug,
    setup_from_dict,
    setup_to_dict,
)

TABLE = "rm_setup_versions"
STATUSES = ("draft", "published", "superseded", "rejected")
ORIGINS = ("web", "field", "import", "api")

# Campos de canal que importan para revisar un cambio (diff legible).
_CH_FIELDS = ("sensor_type", "plane", "angle_deg", "side", "direction", "location",
              "variable", "unit_native", "sensitivity_mv_per_eu", "alarm", "danger",
              "bnc_port", "active", "pair_ref", "keyphasor_ref")
_MACHINE_FIELDS = ("name", "rpm_nominal", "rotation", "bearing_type", "n_bearings",
                   "machine_type", "tag", "iso_norm")


# ---------------------------------------------------------------------
# Lógica pura
# ---------------------------------------------------------------------
def next_version(rows: List[Dict[str, Any]]) -> int:
    return max((int(r.get("version") or 0) for r in rows), default=0) + 1


def published_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    pub = [r for r in rows if r.get("status") == "published"]
    return max(pub, key=lambda r: int(r.get("version") or 0)) if pub else None


def publish_conflict(rows: List[Dict[str, Any]], draft: Dict[str, Any]) -> Optional[str]:
    """None si se puede publicar; si no, el motivo (para la UI)."""
    if draft.get("status") != "draft":
        return f"v{draft.get('version')} no es borrador ({draft.get('status')})"
    pub = published_row(rows)
    if pub is None:
        return None
    base = draft.get("base_version")
    if base is None or int(base) != int(pub["version"]):
        return (f"v{draft.get('version')} parte de v{base}, pero la publicada es "
                f"v{pub['version']} — revisa diferencias antes de publicar")
    return None


def _ch_map(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(c.get("point_label")): c for c in (spec.get("channels") or [])}


def diff_specs(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, List[str]]:
    """Diferencias legibles entre 2 specs (setup_to_dict): sensores agregados /
    quitados / cambiados, máquina y origen."""
    old = old or {}
    o, n = _ch_map(old), _ch_map(new)
    added = [k for k in n if k not in o]
    removed = [k for k in o if k not in n]
    changed: List[str] = []
    for k in n:
        if k not in o:
            continue
        parts = [f"{f} {o[k].get(f)!r}→{n[k].get(f)!r}"
                 for f in _CH_FIELDS if o[k].get(f) != n[k].get(f)]
        if (o[k].get("modbus") or {}) != (n[k].get("modbus") or {}):
            parts.append("origen Modbus")
        if parts:
            changed.append(f"{k}: " + ", ".join(parts))
    om, nm = old.get("machine") or {}, new.get("machine") or {}
    machine = [f"{f} {om.get(f)!r}→{nm.get(f)!r}" for f in _MACHINE_FIELDS if om.get(f) != nm.get(f)]
    source = [f"{f} {old.get(f)!r}→{new.get(f)!r}"
              for f in ("source", "mode", "instance_id") if old.get(f) != new.get(f)]
    if (old.get("modbus_gateway") or {}) != (new.get("modbus_gateway") or {}):
        source.append("gateway Modbus")
    if (old.get("layout") or {}) != (new.get("layout") or {}):
        source.append("train drawing")
    return {"added": added, "removed": removed, "changed": changed,
            "machine": machine, "source": source}


def diff_is_empty(d: Dict[str, List[str]]) -> bool:
    return not any(d.values())


# ---------------------------------------------------------------------
# Nube
# ---------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def setup_id_for(setup: AcqSetup) -> str:
    return _slug(setup.machine.name or "Machine")


def list_versions(setup_id: str, client: Any = None) -> List[Dict[str, Any]]:
    client = client or _rm_client()
    if client is None:
        return []
    try:
        res = (client.table(TABLE).select("*").eq("setup_id", setup_id)
               .order("version", desc=True).execute())
        return list(res.data or [])
    except Exception:  # noqa: BLE001
        return []


def versions_available(client: Any = None) -> bool:
    """False si la tabla aún no existe (antes de correr el SQL)."""
    client = client or _rm_client()
    if client is None:
        return False
    try:
        client.table(TABLE).select("id").limit(1).execute()
        return True
    except Exception:  # noqa: BLE001
        return False


def save_draft(setup: AcqSetup, origin: str = "web", author: str = "",
               base_version: Optional[int] = None, note: str = "",
               client: Any = None, setup_id: str = "") -> Dict[str, Any]:
    """Guarda una NUEVA versión borrador. base_version = versión publicada de la
    que parte (None = la publicada actual). `setup_id` fijo: renombrar la
    máquina NO cambia su identidad (default: slug del nombre, máquina nueva)."""
    client = client or _rm_client()
    if client is None:
        return {"ok": False, "reason": "offline"}
    sid = setup_id or setup_id_for(setup)
    rows = list_versions(sid, client)
    if base_version is None:
        pub = published_row(rows)
        base_version = int(pub["version"]) if pub else None
    row = {"setup_id": sid, "version": next_version(rows), "status": "draft",
           "origin": origin if origin in ORIGINS else "web", "author": author or None,
           "base_version": base_version, "note": note or None,
           "spec": setup_to_dict(setup)}
    try:
        client.table(TABLE).insert(row).execute()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True, "setup_id": sid, "version": row["version"]}


def publish(setup_id: str, version: int, by: str = "", client: Any = None) -> Dict[str, Any]:
    """Publica un borrador: la publicada anterior pasa a superseded y rm_setups
    apunta a esta versión (lo que leen collector / nodo / campo)."""
    client = client or _rm_client()
    if client is None:
        return {"ok": False, "reason": "offline"}
    rows = list_versions(setup_id, client)
    draft = next((r for r in rows if int(r.get("version") or 0) == int(version)), None)
    if draft is None:
        return {"ok": False, "reason": f"v{version} no existe"}
    why = publish_conflict(rows, draft)
    if why:
        return {"ok": False, "reason": why, "conflict": True}
    try:
        prev = published_row(rows)
        if prev is not None:
            client.table(TABLE).update({"status": "superseded"}).eq("id", prev["id"]).execute()
        client.table(TABLE).update({"status": "published", "published_at": _now(),
                                    "published_by": by or None}).eq("id", draft["id"]).execute()
        spec = dict(draft["spec"])
        spec["_version"] = int(version)
        name = (spec.get("machine") or {}).get("name") or setup_id
        client.table(_RM_SETUPS_TABLE).upsert(
            {"id": setup_id, "name": name, "metadata": spec, "updated_at": _now()}).execute()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True, "setup_id": setup_id, "version": int(version)}


def reject(setup_id: str, version: int, client: Any = None) -> Dict[str, Any]:
    client = client or _rm_client()
    if client is None:
        return {"ok": False, "reason": "offline"}
    try:
        (client.table(TABLE).update({"status": "rejected"})
         .eq("setup_id", setup_id).eq("version", int(version)).eq("status", "draft").execute())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True}


def node_ack(setup_id: str, version: int, status: str, detail: str = "",
             client: Any = None) -> Dict[str, Any]:
    """El nodo en línea confirma que aplicó (applied) o rechazó (rejected) vN."""
    client = client or _rm_client()
    if client is None:
        return {"ok": False, "reason": "offline"}
    try:
        (client.table(TABLE).update({"node_status": status, "node_detail": detail or None,
                                     "node_at": _now()})
         .eq("setup_id", setup_id).eq("version", int(version)).execute())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True}


def baseline_from_current(setup_id: str, by: str = "import", client: Any = None) -> Dict[str, Any]:
    """Siembra v1 PUBLICADA desde lo que hoy hay en rm_setups (máquinas que
    existían antes del versionado)."""
    client = client or _rm_client()
    if client is None:
        return {"ok": False, "reason": "offline"}
    if list_versions(setup_id, client):
        return {"ok": True, "skipped": True}
    try:
        res = client.table(_RM_SETUPS_TABLE).select("metadata").eq("id", setup_id).single().execute()
        meta = (res.data or {}).get("metadata")
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    if not meta:
        return {"ok": False, "reason": "sin config"}
    spec = setup_to_dict(setup_from_dict(meta))
    try:
        client.table(TABLE).insert({"setup_id": setup_id, "version": 1, "status": "published",
                                    "origin": "import", "author": by, "spec": spec,
                                    "published_at": _now(), "published_by": by}).execute()
        spec["_version"] = 1
        client.table(_RM_SETUPS_TABLE).update({"metadata": spec}).eq("id", setup_id).execute()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True, "version": 1}
