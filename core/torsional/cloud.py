"""
core/torsional/cloud.py — Guardado/subida a la NUBE de corridas torsionales
===========================================================================

Espejo de `core.modal.modal_cloud` para el módulo Torsional: sube la CORRIDA
(metadata + data cruda de par) a Supabase para que la web la analice/reporte.
Offline-first: si no hay cliente/internet, no falla (devuelve reason=offline).

Requiere en Supabase la tabla `torsional_runs` (id text PK, name text,
metadata jsonb, updated_at text, account/client/tag/hostname text) y el bucket
de Storage `torsional-raw`. Reusa el cliente embebido del recorder (mismas
credenciales que el modal).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional

_RUNS_TABLE = "torsional_runs"
_RAW_BUCKET = "torsional-raw"


def _client():
    try:
        from core.remote_monitoring.recorder import _sb_client
        return _sb_client()
    except Exception:  # noqa: BLE001
        return None


def _slug(name: str) -> str:
    """Slug ASCII (Supabase rechaza tildes/ñ en keys de Storage)."""
    s = unicodedata.normalize("NFKD", str(name or "run")).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_").lower()
    return s or "run"


def new_run_id(name: str):
    """(run_id, ts_iso) determinístico para nombrar la fila y la data cruda igual."""
    from datetime import datetime
    ts = datetime.now().isoformat(timespec="seconds")
    return f"{_slug(name)}_{ts.replace(':', '').replace('-', '')}", ts


def save_run(name: str, payload: Dict[str, Any], run_id: str = "", ts: str = "",
             account: str = "", client: str = "", tag: str = "", hostname: str = "",
             ip: str = "", geo: str = "") -> Dict[str, Any]:
    """Sube una corrida torsional (metadata) a la tabla `torsional_runs`.
    account = cuenta/licencia · ip/geo = IP pública + ubicación aprox. del PC de
    campo (trazabilidad para el aviso por correo, como el modal)."""
    c = _client()
    if c is None:
        return {"ok": False, "reason": "offline"}
    try:
        if not run_id or not ts:
            run_id, ts = new_run_id(name)
        row = {"id": run_id, "name": name or "Torsional run", "metadata": payload, "updated_at": ts,
               "account": account or "", "client": client or "", "tag": tag or "", "hostname": hostname or "",
               "ip": ip or "", "geo": geo or "", "module": "Torsional"}
        try:
            c.table(_RUNS_TABLE).upsert(row).execute()
        except Exception:  # noqa: BLE001
            c.table(_RUNS_TABLE).insert(row).execute()
        return {"ok": True, "name": name, "id": row["id"]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def upload_raw(run_id: str, data, fs: float, channels=None) -> Dict[str, Any]:
    """Sube la DATA CRUDA (voltios/par) a Supabase Storage (gzip). Devuelve raw_ref."""
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
        key = f"{_slug(run_id)}.npy.gz"
        try:
            c.storage.create_bucket(_RAW_BUCKET)            # idempotente
        except Exception:  # noqa: BLE001
            pass
        store = c.storage.from_(_RAW_BUCKET)
        try:
            store.upload(key, raw)
        except Exception as _e:  # noqa: BLE001
            _m = str(_e).lower()
            if not any(x in _m for x in ("exist", "409", "duplicate", "resource already")):
                raise
        return {"ok": True, "bucket": _RAW_BUCKET, "path": key, "fs": float(fs),
                "n_ch": int(arr.shape[1]), "n_samples": int(arr.shape[0]),
                "channels": list(channels or []), "size_bytes": len(raw)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
