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


_RAW_BUCKET = "modal-raw"


def new_run_id(name: str):
    """Genera (run_id, ts_iso) determinístico para una corrida — para nombrar la
    data cruda en Storage y la fila en la tabla con el MISMO id."""
    from datetime import datetime
    from core.modal.oma_layout import _slug
    ts = datetime.now().isoformat(timespec="seconds")
    return f"{_slug(name)}_{ts.replace(':', '').replace('-', '')}", ts


def save_run(name: str, payload: Dict[str, Any], run_id: str = "", ts: str = "") -> Dict[str, Any]:
    """Sube una CORRIDA OMA (modos + config) a la nube (tabla `modal_runs`) para
    que la web genere el reporte. payload libre (jsonb). Si se pasan run_id/ts se
    usan (para que coincidan con la data cruda subida a Storage)."""
    c = _client()
    if c is None:
        return {"ok": False, "reason": "offline"}
    try:
        if not run_id or not ts:
            run_id, ts = new_run_id(name)
        row = {"id": run_id, "name": name or "Modal run", "metadata": payload, "updated_at": ts}
        try:
            c.table(_RUNS_TABLE).upsert(row).execute()
        except Exception:  # noqa: BLE001
            c.table(_RUNS_TABLE).insert(row).execute()
        return {"ok": True, "name": name, "id": row["id"]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def upload_raw(run_id: str, data, fs: float, channels=None) -> Dict[str, Any]:
    """Sube la DATA CRUDA (onda de todos los canales) a Supabase Storage, gzip.
    Devuelve un `raw_ref` para guardar en el payload y que la web la recalcule."""
    c = _client()
    if c is None:
        return {"ok": False, "reason": "offline"}
    try:
        import io, gzip
        import numpy as np
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        buf = io.BytesIO(); np.save(buf, arr)
        raw = gzip.compress(buf.getvalue(), 6)
        key = f"{run_id}.npy.gz"
        try:
            c.storage.create_bucket(_RAW_BUCKET)            # idempotente
        except Exception:  # noqa: BLE001
            pass
        store = c.storage.from_(_RAW_BUCKET)
        try:
            store.upload(key, raw, {"upsert": "true"})
        except Exception:  # noqa: BLE001
            store.update(key, raw)
        return {"ok": True, "bucket": _RAW_BUCKET, "path": key, "fs": float(fs),
                "n_ch": int(arr.shape[1]), "n_samples": int(arr.shape[0]),
                "channels": list(channels or []), "size_bytes": len(raw)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def download_raw(ref: Dict[str, Any]):
    """Descarga la data cruda referida por `raw_ref`. Devuelve (data[N,ch], fs) o None."""
    if not ref or not ref.get("path"):
        return None
    c = _client()
    if c is None:
        return None
    try:
        import io, gzip
        import numpy as np
        store = c.storage.from_(ref.get("bucket", _RAW_BUCKET))
        raw = store.download(ref["path"])
        data = np.load(io.BytesIO(gzip.decompress(raw)))
        return data, float(ref.get("fs", 0.0))
    except Exception:  # noqa: BLE001
        return None


def list_runs() -> List[Dict[str, Any]]:
    """Lista las corridas OMA subidas por el campo (para elegir en la web)."""
    c = _client()
    if c is None:
        return []
    try:
        r = c.table(_RUNS_TABLE).select("id, name, updated_at").execute()
        return sorted(r.data or [], key=lambda x: x.get("updated_at", ""), reverse=True)
    except Exception:  # noqa: BLE001
        return []


def load_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Descarga el payload de una corrida OMA por su id."""
    c = _client()
    if c is None:
        return None
    try:
        r = c.table(_RUNS_TABLE).select("metadata").eq("id", run_id).single().execute()
        if r.data and r.data.get("metadata"):
            return r.data["metadata"]
    except Exception:  # noqa: BLE001
        return None
    return None


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
