"""
core.history_storage
====================

Backend unificado para persistir snapshots históricos de análisis
vibratorios. Reemplaza el storage directo a disco que tienen los
módulos *_history.py existentes (scl, polar, bode, trend) y habilita
los 4 nuevos (waveform, spectrum, orbit, tabular).

Ciclo 23.77 — diseño:

  • Backend principal: Supabase Storage bucket `instance-history`
    (creado por data/storage_instance_history_setup.sql en v3.31.76).
  • Fallback: disco local en data/instances/{instance_id}/history/
    cuando Supabase no está configurado (desarrollo, smoke tests).
  • Compresión: JSON → gzip antes de upload. 5-10× reducción de tamaño.
  • Retención: LRU automático, max N snapshots por (instance, type).
    Default N=10. Cuando se inserta el #11, el más viejo se borra.
  • Layout: instance-history/{instance_id}/{type}/{snapshot_id}.json.gz

Naming convention: snapshot_id incluye timestamp en formato
ISO compacto (`{type}_{YYYYMMDD_HHMMSS}`). Esto garantiza que
sorting lexicográfico == sorting cronológico, simplificando la
rotación LRU.

API pública:

  save_snapshot(instance_id, snapshot_type, snapshot_id, data) -> str
  list_snapshots(instance_id, snapshot_type) -> List[Dict]
  load_snapshot(instance_id, snapshot_type, snapshot_id) -> Optional[Dict]
  delete_snapshot(instance_id, snapshot_type, snapshot_id) -> bool
  count_snapshots(instance_id, snapshot_type) -> int
  list_all_snapshots(instance_id) -> Dict[str, List[Dict]]
  export_all_as_zip_bytes(instance_id) -> bytes  (para v3.31.83)

Constantes:
  MAX_SNAPSHOTS_PER_TYPE = 10   (LRU rotation threshold)
  BUCKET_NAME = "instance-history"
  COMPRESSION = "gzip"
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# =============================================================
# CONFIGURACIÓN
# =============================================================

BUCKET_NAME = "instance-history"
BACKUP_BUCKET_NAME = "instance-history-backups"
MAX_SNAPSHOTS_PER_TYPE = 10  # Hot tier — LRU rotation kicks in al #11

# snapshot_id valid pattern.
#   Legacy (pre v3.31.236): type_YYYYMMDD_HHMMSS
#   Actual  (v3.31.236+):    type_YYYYMMDD_HHMMSS_xxxxxx  (sufijo UUID short)
# Ambos siguen siendo lexicográficamente ordenables por timestamp.
_SNAPSHOT_ID_PATTERN = re.compile(r"^[a-z_]+_\d{8}_\d{6}(_[a-f0-9]{6,8})?$")

# Tipos válidos de snapshot
KNOWN_TYPES = (
    "scl", "polar", "bode", "trend",                # legacy modules
    "waveform", "spectrum", "orbit", "tabular",     # nuevos en v3.31.80
)


# =============================================================
# BACKEND DETECTION
# =============================================================

def _get_supabase_client():
    """Reusa el client del módulo live_readings para consistencia."""
    try:
        from core.live_readings import _get_supabase_client as _gsc
        return _gsc()
    except Exception:
        return None


def _is_supabase_available() -> bool:
    return _get_supabase_client() is not None


def _local_storage_root() -> Path:
    """Fallback path cuando Supabase no está configurado."""
    try:
        from core.instance_repository import INSTANCES_DIR
        return INSTANCES_DIR
    except Exception:
        # Último fallback: data/instances directo
        return Path(__file__).resolve().parents[1] / "data" / "instances"


# =============================================================
# COMPRESSION
# =============================================================

def _compress_json(data: Dict[str, Any]) -> bytes:
    """Serializa dict a JSON UTF-8 y comprime con gzip."""
    payload = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
        gz.write(payload)
    return buf.getvalue()


def _decompress_json(blob: bytes) -> Dict[str, Any]:
    """Descomprime gzip → JSON → dict."""
    with gzip.GzipFile(fileobj=io.BytesIO(blob), mode="rb") as gz:
        payload = gz.read()
    return json.loads(payload.decode("utf-8"))


# =============================================================
# PATH HELPERS
# =============================================================

def _validate_type(snapshot_type: str) -> None:
    if snapshot_type not in KNOWN_TYPES:
        raise ValueError(
            f"snapshot_type desconocido: {snapshot_type!r}. "
            f"Opciones: {KNOWN_TYPES}"
        )


def _validate_instance_id(instance_id: str) -> None:
    if not instance_id or not isinstance(instance_id, str):
        raise ValueError("instance_id debe ser string no vacío")
    if "/" in instance_id or ".." in instance_id:
        raise ValueError(f"instance_id inválido: {instance_id!r}")


def _storage_path(instance_id: str, snapshot_type: str, snapshot_id: str) -> str:
    """Path canónico dentro del bucket Supabase o relativo al INSTANCES_DIR local."""
    return f"{instance_id}/{snapshot_type}/{snapshot_id}.json.gz"


def _local_path(instance_id: str, snapshot_type: str, snapshot_id: str) -> Path:
    """Path absoluto en disco local (fallback)."""
    root = _local_storage_root()
    return root / instance_id / "history" / snapshot_type / f"{snapshot_id}.json.gz"


# =============================================================
# CORE API — SAVE
# =============================================================

def save_snapshot(
    instance_id: str,
    snapshot_type: str,
    snapshot_id: str,
    data: Dict[str, Any],
) -> str:
    """Guarda un snapshot y aplica rotación LRU. Devuelve storage_path."""
    _validate_instance_id(instance_id)
    _validate_type(snapshot_type)
    if not snapshot_id or not isinstance(snapshot_id, str):
        raise ValueError("snapshot_id debe ser string no vacío")

    blob = _compress_json(data)
    path = _storage_path(instance_id, snapshot_type, snapshot_id)

    if _is_supabase_available():
        _save_supabase(path, blob)
    else:
        _save_local(instance_id, snapshot_type, snapshot_id, blob)

    # LRU rotation: si tras el insert hay > MAX, borrar los más viejos
    _enforce_lru(instance_id, snapshot_type)
    return path


def _save_supabase(path: str, blob: bytes) -> None:
    client = _get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase client no disponible")
    # supabase-py v2 API: client.storage.from_(bucket).upload(path, blob, options)
    try:
        # Upsert = sobreescribir si ya existe (caso edge: re-save mismo id)
        client.storage.from_(BUCKET_NAME).upload(
            path=path,
            file=blob,
            file_options={
                "content-type": "application/gzip",
                "x-upsert": "true",
            },
        )
    except Exception as e:
        log.warning("history_storage: upload supabase failed (%s) — fallback local", e)
        # Si el upload falla, salvamos local para no perder data
        instance_id, snapshot_type, snapshot_id = _split_path(path)
        _save_local(instance_id, snapshot_type, snapshot_id, blob)


def _save_local(instance_id: str, snapshot_type: str, snapshot_id: str, blob: bytes) -> None:
    p = _local_path(instance_id, snapshot_type, snapshot_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)


def _split_path(path: str) -> Tuple[str, str, str]:
    """Inversa de _storage_path: extrae (instance_id, type, snapshot_id) de un path."""
    parts = path.split("/")
    if len(parts) != 3:
        raise ValueError(f"path inválido: {path}")
    instance_id, snapshot_type, filename = parts
    snapshot_id = filename.removesuffix(".json.gz")
    return instance_id, snapshot_type, snapshot_id


# =============================================================
# CORE API — LIST
# =============================================================

def list_snapshots(instance_id: str, snapshot_type: str) -> List[Dict[str, Any]]:
    """Lista snapshots ordenados DESCENDENTE por snapshot_id (más nuevos primero).

    Returns list of dicts:
        {
            "snapshot_id": "scl_20260511_153022",
            "storage_path": "tes1/scl/scl_20260511_153022.json.gz",
            "size_bytes": 12345 (opcional, depende del backend)
        }
    """
    _validate_instance_id(instance_id)
    _validate_type(snapshot_type)

    if _is_supabase_available():
        items = _list_supabase(instance_id, snapshot_type)
    else:
        items = _list_local(instance_id, snapshot_type)

    # Ordenar descendentemente por snapshot_id (= timestamp inverso)
    items.sort(key=lambda x: x["snapshot_id"], reverse=True)
    return items


def _list_supabase(instance_id: str, snapshot_type: str) -> List[Dict[str, Any]]:
    client = _get_supabase_client()
    if client is None:
        return []
    prefix = f"{instance_id}/{snapshot_type}"
    try:
        resp = client.storage.from_(BUCKET_NAME).list(
            path=prefix,
            options={"limit": 100, "offset": 0, "sortBy": {"column": "name", "order": "desc"}},
        )
        items = []
        for entry in resp or []:
            name = entry.get("name", "")
            if not name.endswith(".json.gz"):
                continue
            snapshot_id = name.removesuffix(".json.gz")
            items.append({
                "snapshot_id": snapshot_id,
                "storage_path": f"{prefix}/{name}",
                "size_bytes": (entry.get("metadata") or {}).get("size"),
            })
        return items
    except Exception as e:
        log.warning("history_storage: list supabase failed: %s", e)
        return []


def _list_local(instance_id: str, snapshot_type: str) -> List[Dict[str, Any]]:
    folder = _local_storage_root() / instance_id / "history" / snapshot_type
    if not folder.exists():
        return []
    items = []
    for p in folder.glob("*.json.gz"):
        snapshot_id = p.stem.removesuffix(".json")  # stem da "scl_20260511_153022.json"
        # .stem de "scl_X.json.gz" → "scl_X.json" → removesuffix → "scl_X"
        items.append({
            "snapshot_id": snapshot_id,
            "storage_path": str(p.relative_to(_local_storage_root())),
            "size_bytes": p.stat().st_size,
        })
    return items


# =============================================================
# CORE API — LOAD
# =============================================================

def load_snapshot(
    instance_id: str,
    snapshot_type: str,
    snapshot_id: str,
) -> Optional[Dict[str, Any]]:
    """Descarga y descomprime un snapshot. None si no existe."""
    _validate_instance_id(instance_id)
    _validate_type(snapshot_type)

    if _is_supabase_available():
        blob = _load_supabase(instance_id, snapshot_type, snapshot_id)
    else:
        blob = _load_local(instance_id, snapshot_type, snapshot_id)

    if blob is None:
        return None
    try:
        return _decompress_json(blob)
    except Exception as e:
        log.warning("history_storage: decompress failed for %s/%s/%s: %s",
                    instance_id, snapshot_type, snapshot_id, e)
        return None


def _load_supabase(instance_id: str, snapshot_type: str, snapshot_id: str) -> Optional[bytes]:
    client = _get_supabase_client()
    if client is None:
        return None
    path = _storage_path(instance_id, snapshot_type, snapshot_id)
    try:
        return client.storage.from_(BUCKET_NAME).download(path)
    except Exception as e:
        log.info("history_storage: download supabase miss (%s): %s", path, e)
        return None


def _load_local(instance_id: str, snapshot_type: str, snapshot_id: str) -> Optional[bytes]:
    p = _local_path(instance_id, snapshot_type, snapshot_id)
    if not p.exists():
        return None
    return p.read_bytes()


# =============================================================
# CORE API — DELETE
# =============================================================

def delete_snapshot(instance_id: str, snapshot_type: str, snapshot_id: str) -> bool:
    """Borra un snapshot. True si tuvo éxito, False si no existía."""
    _validate_instance_id(instance_id)
    _validate_type(snapshot_type)

    if _is_supabase_available():
        return _delete_supabase(instance_id, snapshot_type, snapshot_id)
    else:
        return _delete_local(instance_id, snapshot_type, snapshot_id)


def _delete_supabase(instance_id: str, snapshot_type: str, snapshot_id: str) -> bool:
    client = _get_supabase_client()
    if client is None:
        return False
    path = _storage_path(instance_id, snapshot_type, snapshot_id)
    try:
        client.storage.from_(BUCKET_NAME).remove([path])
        return True
    except Exception as e:
        log.warning("history_storage: delete supabase failed: %s", e)
        return False


def _delete_local(instance_id: str, snapshot_type: str, snapshot_id: str) -> bool:
    p = _local_path(instance_id, snapshot_type, snapshot_id)
    if not p.exists():
        return False
    try:
        p.unlink()
        return True
    except Exception:
        return False


# =============================================================
# CORE API — UTILITIES
# =============================================================

def count_snapshots(instance_id: str, snapshot_type: str) -> int:
    return len(list_snapshots(instance_id, snapshot_type))


def list_all_snapshots(instance_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Devuelve dict {snapshot_type: [snapshots...]} para todos los tipos."""
    return {t: list_snapshots(instance_id, t) for t in KNOWN_TYPES}


# =============================================================
# LRU RETENTION
# =============================================================

def _enforce_lru(instance_id: str, snapshot_type: str) -> None:
    """Si hay más de MAX_SNAPSHOTS_PER_TYPE, borra los más viejos.

    Llamado automáticamente desde save_snapshot. Best-effort: si falla
    el delete de alguno, no propaga el error (la app no debe romper
    por una rotación fallida).
    """
    items = list_snapshots(instance_id, snapshot_type)
    if len(items) <= MAX_SNAPSHOTS_PER_TYPE:
        return
    # items vienen ordenados desc — los del final son los más viejos
    to_delete = items[MAX_SNAPSHOTS_PER_TYPE:]
    for item in to_delete:
        try:
            delete_snapshot(instance_id, snapshot_type, item["snapshot_id"])
        except Exception as e:
            log.warning("history_storage: LRU delete failed for %s: %s",
                        item["snapshot_id"], e)


# =============================================================
# EXPORT (preparación para v3.31.83)
# =============================================================

def export_instance_as_zip_bytes(
    instance_id: str,
    include_diagram_svg: Optional[bytes] = None,
    include_diagram_png: Optional[bytes] = None,
    manifest_extra: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Empaqueta TODOS los snapshots de una instance en un ZIP descargable.

    Layout del ZIP:
        manifest.json
        README.txt
        diagram.svg  (si se provee)
        diagram.png  (si se provee)
        snapshots/
            scl/
                scl_*.json    (descomprimidos para que el cliente
                               pueda abrirlos en cualquier editor)
            polar/
            ...

    Returns:
        bytes del archivo ZIP (in-memory). Caller lo escribe a disco o
        lo sube a Storage para enviar al cliente.
    """
    import zipfile

    all_snaps = list_all_snapshots(instance_id)
    total_count = sum(len(v) for v in all_snaps.values())

    manifest = {
        "format": "watermelon-instance-export-v1",
        "instance_id": instance_id,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "total_snapshots": total_count,
        "snapshots_by_type": {t: len(v) for t, v in all_snaps.items()},
        "extra": manifest_extra or {},
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        readme = (
            f"Watermelon System — Instance History Export\n"
            f"=============================================\n\n"
            f"Instance ID:       {instance_id}\n"
            f"Exported at:       {manifest['exported_at']}\n"
            f"Total snapshots:   {total_count}\n\n"
            f"Layout:\n"
            f"  manifest.json     — metadata del export\n"
            f"  diagram.svg/png   — snapshot visual del Live Monitoring\n"
            f"  snapshots/        — un directorio por tipo de análisis\n\n"
            f"Cada snapshot es un JSON con la captura completa del análisis.\n"
            f"Pueden abrirse con cualquier editor de texto, o reimportarse\n"
            f"a Watermelon System.\n"
        )
        zf.writestr("README.txt", readme)

        if include_diagram_svg:
            zf.writestr("diagram.svg", include_diagram_svg)
        if include_diagram_png:
            zf.writestr("diagram.png", include_diagram_png)

        for snapshot_type, snaps in all_snaps.items():
            for snap in snaps:
                sid = snap["snapshot_id"]
                data = load_snapshot(instance_id, snapshot_type, sid)
                if data is None:
                    continue
                # Descomprimir y guardar como .json (no .json.gz) para
                # que el cliente pueda abrirlo sin herramientas extra
                zf.writestr(
                    f"snapshots/{snapshot_type}/{sid}.json",
                    json.dumps(data, ensure_ascii=False, indent=2, default=str),
                )

    return buf.getvalue()


# =============================================================
# UTILITY — id generation
# =============================================================

def new_snapshot_id(snapshot_type: str, now: Optional[datetime] = None) -> str:
    """Helper para generar IDs consistentes: {type}_YYYYMMDD_HHMMSS_xxxxxx.

    Ciclo 17.31 (v3.31.236) — antes el formato era {type}_YYYYMMDD_HHMMSS.
    Eso fallaba cuando dos saves caían en el mismo segundo (race entre dos
    usuarios o doble click rápido en "Guardar snapshot"): el segundo save
    SOBREESCRIBÍA el primero porque generaba exactamente el mismo path
    ({instance}/{type}/{id}.json.gz). El sufijo UUID short (6 hex chars)
    elimina la colisión sin romper sorting cronológico — el prefijo
    timestamp sigue siendo lexicográficamente ordenable y la LRU
    rotation sigue funcionando.
    """
    _validate_type(snapshot_type)
    now = now or datetime.now()
    suffix = uuid.uuid4().hex[:6]
    return f"{snapshot_type}_{now.strftime('%Y%m%d_%H%M%S')}_{suffix}"


if __name__ == "__main__":
    # Smoke test (requiere Supabase configurado o disco local)
    print("=== Watermelon history_storage smoke test ===\n")
    test_instance = "_smoke_test"
    test_type = "trend"

    test_id = new_snapshot_id(test_type)
    test_data = {
        "snapshot_id": test_id,
        "instance_id": test_instance,
        "timestamp": datetime.utcnow().isoformat(),
        "test_payload": [1, 2, 3, 4, 5],
    }
    print(f"Saving: {test_id}")
    path = save_snapshot(test_instance, test_type, test_id, test_data)
    print(f"  → path: {path}")

    print(f"\nListing snapshots:")
    for snap in list_snapshots(test_instance, test_type):
        print(f"  {snap['snapshot_id']}  ({snap.get('size_bytes', '?')} bytes)")

    print(f"\nLoading back...")
    loaded = load_snapshot(test_instance, test_type, test_id)
    assert loaded is not None
    assert loaded["test_payload"] == [1, 2, 3, 4, 5]
    print(f"  ✓ payload matches")

    print(f"\nCleaning up...")
    delete_snapshot(test_instance, test_type, test_id)
    print(f"  ✓ deleted")

    print("\n=== Smoke test OK ===")
