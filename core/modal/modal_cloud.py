"""
core/modal/modal_cloud.py — Guardado/carga en la NUBE del setup modal
=====================================================================

Sube el OMALayout (geometría + puntos + adquisición) a Supabase (tabla
`modal_setups`) para compartir la configuración entre el campo y la web: se
configura acá y se analiza/reporta en la web, o se configura en la web y el
campo solo la carga. Offline-first: si no hay cliente/internet, no falla.

Requiere la tabla `modal_setups` (id text PK, name text, metadata jsonb,
updated_at text) en Supabase. Reusa el cliente embebido del recorder.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_TABLE = "modal_setups"


def _client():
    try:
        from core.remote_monitoring.recorder import _sb_client
        return _sb_client()
    except Exception:  # noqa: BLE001
        return None


def save_layout_cloud(layout) -> Dict[str, Any]:
    c = _client()
    if c is None:
        return {"ok": False, "reason": "offline"}
    try:
        from datetime import datetime
        from core.modal.oma_layout import _slug
        name = layout.name or "Modal"
        row = {"id": _slug(name), "name": name, "metadata": layout.to_dict(),
               "updated_at": datetime.now().isoformat(timespec="seconds")}
        try:
            c.table(_TABLE).upsert(row).execute()
        except Exception:  # noqa: BLE001
            c.table(_TABLE).insert(row).execute()
        return {"ok": True, "name": name}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def list_layouts_cloud() -> List[Dict[str, Any]]:
    c = _client()
    if c is None:
        return []
    try:
        r = c.table(_TABLE).select("id, name, updated_at").execute()
        return sorted(r.data or [], key=lambda x: x.get("updated_at", ""), reverse=True)
    except Exception:  # noqa: BLE001
        return []


_RUNS_TABLE = "modal_runs"


def save_run(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sube una CORRIDA OMA (modos + config) a la nube (tabla `modal_runs`) para
    que la web genere el reporte. payload libre (jsonb)."""
    c = _client()
    if c is None:
        return {"ok": False, "reason": "offline"}
    try:
        from datetime import datetime
        from core.modal.oma_layout import _slug
        ts = datetime.now().isoformat(timespec="seconds")
        row = {"id": f"{_slug(name)}_{ts.replace(':', '').replace('-', '')}",
               "name": name or "Modal run", "metadata": payload, "updated_at": ts}
        try:
            c.table(_RUNS_TABLE).upsert(row).execute()
        except Exception:  # noqa: BLE001
            c.table(_RUNS_TABLE).insert(row).execute()
        return {"ok": True, "name": name, "id": row["id"]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def load_layout_cloud(name_or_slug: str) -> Optional[Any]:
    c = _client()
    if c is None:
        return None
    try:
        from core.modal.oma_layout import OMALayout, _slug
        r = c.table(_TABLE).select("metadata").eq("id", _slug(name_or_slug)).single().execute()
        if r.data and r.data.get("metadata"):
            return OMALayout.from_dict(r.data["metadata"])
    except Exception:  # noqa: BLE001
        return None
    return None
