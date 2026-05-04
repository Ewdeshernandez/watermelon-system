"""
core.document_cache
===================

Cache local en disco para los archivos del bucket de Supabase Storage
(Ciclo 17.18).

Por qué existe:
  Antes, cada llamada a get_instance_document_bytes() bajaba el archivo
  COMPLETO desde Supabase Storage, en cada rerun de Streamlit. Como
  cada interacción del usuario (slider, selectbox, click) dispara un
  rerun, los mismos archivos se descargaban decenas de veces por
  sesión por usuario. Resultado: 8.6 GB de cached egress en 4 días en
  un bucket que solo pesa 2 MB en total. El plan Free de Supabase
  (5 GB/mes) se desbordaba con un solo specialist activo.

Diseño:
  - Cache en disco (no solo memoria): sobrevive reruns, redeploys
    de Streamlit Cloud, y restarts del proceso
  - Path: data/cache/instance_documents/{instance_id_safe}__{filename_safe}
  - TTL configurable (default 30 días). Pasado el TTL se considera
    stale y se re-baja para detectar updates remotos
  - Write atómico (tmp + os.replace) para que un crash mid-write no
    deje archivos corruptos
  - Invalidación explícita en upload/remove para mantener consistencia

API pública:
  - cached_download_bytes(instance_id, storage_filename) → Optional[bytes]
  - invalidate_document(instance_id, storage_filename="")  → cantidad invalidada
  - cleanup_stale(max_age_days=30) → cantidad limpiada
  - get_cache_stats() → dict con n_files / total_size / oldest_age
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache" / "instance_documents"

# TTL: pasado este tiempo, se considera stale y se vuelve a bajar.
# 30 días es seguro porque los documentos de instancia (.bn, schematics
# PNG, datasheets) prácticamente nunca cambian una vez subidos.
DEFAULT_TTL_SECONDS = 30 * 24 * 3600


# =============================================================
# Helpers de paths
# =============================================================

def _safe_segment(raw: str, max_len: int = 120) -> str:
    """Sanitiza un string para usarlo como segmento de filename."""
    s = str(raw or "").strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s[:max_len] or "_"


def _cache_path_for(instance_id: str, storage_filename: str) -> Path:
    """Devuelve el path local donde se cachea (instance_id, filename)."""
    safe_inst = _safe_segment(instance_id, 64)
    safe_file = _safe_segment(storage_filename, 120)
    return CACHE_DIR / f"{safe_inst}__{safe_file}"


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write atómico (tmp + os.replace) para que un crash no deje
    archivos corruptos en el cache."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=target.stem + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(str(tmp_path), str(target))
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def _is_fresh(p: Path, ttl_seconds: int) -> bool:
    """True si el archivo existe y su mtime está dentro del TTL."""
    try:
        age = time.time() - p.stat().st_mtime
        return age <= ttl_seconds
    except Exception:
        return False


# =============================================================
# API PÚBLICA
# =============================================================

def cached_download_bytes(
    instance_id: str,
    storage_filename: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    log_to_stderr: bool = False,
) -> Optional[bytes]:
    """Devuelve los bytes del archivo, sirviendo desde cache local si
    está fresh. Si no, baja del repository activo y guarda en cache.

    Args:
        instance_id:       id de la instancia
        storage_filename:  nombre del archivo en el storage del repo
        ttl_seconds:       cuánto considerar válido el cache (default 30d)
        log_to_stderr:     si True, loguea HIT/MISS para debugging

    Returns:
        bytes del archivo, o None si no se pudo obtener.
    """
    if not instance_id or not storage_filename:
        return None

    cache_p = _cache_path_for(instance_id, storage_filename)

    # 1) HIT — cache local fresh
    if cache_p.exists() and _is_fresh(cache_p, ttl_seconds):
        try:
            data = cache_p.read_bytes()
            if log_to_stderr:
                print(
                    f"[WM_DOC_CACHE] HIT  · {instance_id}/{storage_filename} "
                    f"({len(data)} B from cache)",
                    file=sys.stderr, flush=True,
                )
            return data
        except Exception:
            # Cache corrupto, lo borramos y caemos a MISS
            try:
                cache_p.unlink()
            except Exception:
                pass

    # 2) MISS — bajar del repo
    try:
        from core.instance_state import get_active_repository
        repo = get_active_repository()
    except Exception as e:
        if log_to_stderr:
            print(
                f"[WM_DOC_CACHE] FAIL · no se pudo obtener repo: {e}",
                file=sys.stderr, flush=True,
            )
        return None

    try:
        data = repo.download_document_bytes(instance_id, storage_filename)
    except Exception as e:
        if log_to_stderr:
            print(
                f"[WM_DOC_CACHE] FAIL · download_document_bytes: {e}",
                file=sys.stderr, flush=True,
            )
        return None

    if data is None:
        if log_to_stderr:
            print(
                f"[WM_DOC_CACHE] MISS · {instance_id}/{storage_filename} "
                f"(repo devolvió None)",
                file=sys.stderr, flush=True,
            )
        return None

    # 3) Persistir en cache para próximas
    try:
        _atomic_write_bytes(cache_p, data)
        if log_to_stderr:
            print(
                f"[WM_DOC_CACHE] MISS · {instance_id}/{storage_filename} "
                f"({len(data)} B downloaded + cached)",
                file=sys.stderr, flush=True,
            )
    except Exception as e:
        # No bloqueante: la descarga ya fue exitosa
        if log_to_stderr:
            print(
                f"[WM_DOC_CACHE] WARN · descarga OK pero falló write cache: {e}",
                file=sys.stderr, flush=True,
            )

    return data


def invalidate_document(instance_id: str, storage_filename: str = "") -> int:
    """Invalida el cache de un archivo específico, o de TODOS los archivos
    de una instancia si storage_filename está vacío.

    Llamar después de upload_document_bytes() o delete_document_file()
    para garantizar que la próxima cached_download_bytes() vea el
    estado fresco del bucket.

    Returns:
        cantidad de archivos de cache eliminados.
    """
    if not instance_id:
        return 0
    if not CACHE_DIR.exists():
        return 0

    n = 0
    if storage_filename:
        p = _cache_path_for(instance_id, storage_filename)
        if p.exists():
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
    else:
        # Borrar TODOS los archivos de esa instancia
        safe_inst = _safe_segment(instance_id, 64)
        for p in CACHE_DIR.glob(f"{safe_inst}__*"):
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
    return n


def cleanup_stale(max_age_days: int = 30) -> int:
    """Borra archivos de cache cuyo mtime sea más viejo que `max_age_days`.
    Idempotente. Pensado para invocar al inicio del proceso o periódicamente.

    Returns:
        cantidad de archivos eliminados.
    """
    if not CACHE_DIR.exists():
        return 0
    cutoff = time.time() - (max_age_days * 24 * 3600)
    n = 0
    for p in CACHE_DIR.iterdir():
        if not p.is_file():
            continue
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                n += 1
        except Exception:
            pass
    return n


def clear_all_cache() -> int:
    """Borra TODO el cache. Útil para troubleshooting o si cambia el
    storage backend. Returns: cantidad de archivos eliminados.
    """
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for p in CACHE_DIR.iterdir():
        if not p.is_file():
            continue
        try:
            p.unlink()
            n += 1
        except Exception:
            pass
    return n


def get_cache_stats() -> Dict[str, Any]:
    """Devuelve métricas del estado actual del cache. Útil para mostrar
    en sidebar de admin o para diagnóstico.
    """
    out = {
        "cache_dir": str(CACHE_DIR),
        "exists": CACHE_DIR.exists(),
        "n_files": 0,
        "total_size_bytes": 0,
        "total_size_human": "0 B",
        "oldest_age_seconds": 0,
        "newest_age_seconds": 0,
    }
    if not CACHE_DIR.exists():
        return out

    now = time.time()
    sizes = []
    ages = []
    for p in CACHE_DIR.iterdir():
        if not p.is_file():
            continue
        try:
            stat = p.stat()
            sizes.append(stat.st_size)
            ages.append(now - stat.st_mtime)
        except Exception:
            continue

    out["n_files"] = len(sizes)
    out["total_size_bytes"] = sum(sizes)
    out["total_size_human"] = _human_size(sum(sizes))
    if ages:
        out["oldest_age_seconds"] = max(ages)
        out["newest_age_seconds"] = min(ages)
    return out


def _human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024:
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


__all__ = [
    "cached_download_bytes",
    "invalidate_document",
    "cleanup_stale",
    "clear_all_cache",
    "get_cache_stats",
    "DEFAULT_TTL_SECONDS",
    "CACHE_DIR",
]
