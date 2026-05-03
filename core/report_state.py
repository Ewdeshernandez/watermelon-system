from __future__ import annotations

import base64
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_STATE_FILE = DATA_DIR / "report_state.json"
REPORT_DRAFTS_DIR = DATA_DIR / "report_drafts"

# =============================================================
# Ciclo 17.14.1 HOTFIX — Anti-pérdida de trabajo
# =============================================================
# Bug crítico reportado: con 50-100 imágenes, se pierde TODO el
# trabajo del día porque:
#   1. save_report_state escribía al JSON directo (NO atómico).
#      Si Streamlit crasheaba mid-write (timeout/OOM con 67MB
#      de JSON con base64 inline) → archivo corrupto.
#   2. load_report_state silenciosamente devolvía {} sin
#      avisar al usuario que el JSON estaba roto.
#
# Fix:
#   - Write atómico vía tempfile + os.replace
#   - Backup rotativo (.bak.1 → .bak.5) ANTES de cada write
#     exitoso, así siempre tenemos N versiones de respaldo
#   - load_report_state ahora intenta recovery desde backup
#     si el archivo principal está corrupto, y EXPONE el flag
#     'recovered_from' para que la UI muestre banner al usuario
#   - cleanup_old_backups limita a max N backups por archivo
# =============================================================

MAX_BACKUPS = 5  # cantidad de versiones de respaldo a mantener


def _encode_image_bytes(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("utf-8")
    return ""


def _decode_image_bytes(value: Any) -> bytes | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return base64.b64decode(text.encode("utf-8"))
    except Exception:
        return None


def _safe_slug(text: Any) -> str:
    raw = str(text or "").strip().lower()
    raw = re.sub(r"[^a-z0-9_-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "draft"


def sanitize_report_items(items: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    if not isinstance(items, list):
        return out

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        out.append(
            {
                "id": str(item.get("id") or f"report_item_{idx+1}"),
                "type": str(item.get("type") or "figure"),
                "title": str(item.get("title") or f"Figura {idx+1}"),
                "notes": str(item.get("notes") or ""),
                "signal_id": str(item.get("signal_id") or ""),
                "machine": str(item.get("machine") or ""),
                "point": str(item.get("point") or ""),
                "variable": str(item.get("variable") or ""),
                "timestamp": str(item.get("timestamp") or ""),
                "image_bytes_b64": _encode_image_bytes(item.get("image_bytes")),
            }
        )

    return out


def restore_report_items(items: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    if not isinstance(items, list):
        return out

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        out.append(
            {
                "id": str(item.get("id") or f"report_item_{idx+1}"),
                "type": str(item.get("type") or "figure"),
                "title": str(item.get("title") or f"Figura {idx+1}"),
                "notes": str(item.get("notes") or ""),
                "signal_id": str(item.get("signal_id") or ""),
                "machine": str(item.get("machine") or ""),
                "point": str(item.get("point") or ""),
                "variable": str(item.get("variable") or ""),
                "timestamp": str(item.get("timestamp") or ""),
                "figure": None,
                "image_bytes": _decode_image_bytes(item.get("image_bytes_b64")),
            }
        )

    return out


def _serialize_state(*, items: Any, meta: Any) -> Dict[str, Any]:
    return {
        "items": sanitize_report_items(items),
        "meta": meta if isinstance(meta, dict) else {},
    }


def _restore_state(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "items": restore_report_items(raw.get("items", [])),
        "meta": raw.get("meta", {}) if isinstance(raw.get("meta", {}), dict) else {},
    }


# =============================================================
# Ciclo 17.14.1 — Helpers internos de write atómico + backups
# =============================================================

def _backup_path(target: Path, idx: int) -> Path:
    """Devuelve el path del backup número `idx` (1..MAX_BACKUPS)
    para el archivo `target`. Ej: report_state.json → report_state.json.bak.1
    """
    return target.with_suffix(target.suffix + f".bak.{idx}")


def _rotate_backups(target: Path) -> None:
    """Rota los backups del archivo `target`:
       - .bak.5 se elimina (más viejo)
       - .bak.4 → .bak.5
       - .bak.3 → .bak.4
       - ...
       - .bak.1 → .bak.2
       - target → .bak.1 (más reciente)
    Si el target no existe (primer save), simplemente no rota nada.
    """
    if not target.exists():
        return
    # Eliminar el más viejo (.bak.MAX_BACKUPS) si existe
    oldest = _backup_path(target, MAX_BACKUPS)
    if oldest.exists():
        try:
            oldest.unlink()
        except Exception:
            pass
    # Rotar desde el (MAX-1) hacia atrás
    for i in range(MAX_BACKUPS - 1, 0, -1):
        src = _backup_path(target, i)
        dst = _backup_path(target, i + 1)
        if src.exists():
            try:
                src.replace(dst)
            except Exception:
                pass
    # target → .bak.1 (copia, no movimiento, para que el target
    # quede disponible mientras escribimos el nuevo)
    try:
        shutil.copy2(target, _backup_path(target, 1))
    except Exception:
        pass


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> None:
    """Escribe `payload` a `target` ATÓMICAMENTE:
       1. Escribe a un archivo temporal en el mismo directorio
       2. fsync para asegurar bytes en disco antes del rename
       3. os.replace (atómico en POSIX y Windows) → si crashea,
          el archivo final queda intacto en su versión previa
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=target.stem + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        # Rename atómico — si crashea aquí, target queda con
        # contenido viejo, NO con archivo corrupto a medio escribir
        os.replace(str(tmp_path), str(target))
    except Exception:
        # Si algo falló, asegurar limpieza del temp
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


# =============================================================
# API PÚBLICA
# =============================================================

def save_report_state(*, items: Any, meta: Any, filename: Path | None = None) -> None:
    """Persiste el estado del reporte de forma SEGURA.

    Ciclo 17.14.1 HOTFIX:
      - Antes del write, rota backups (mantiene 5 versiones)
      - Write ATÓMICO via tmp file + os.replace
      - Si crashea mid-write, el archivo final queda intacto
        y se puede recuperar desde el backup más reciente
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = filename or REPORT_STATE_FILE
    payload = _serialize_state(items=items, meta=meta)
    # Inyectar metadata interna del save (útil para debugging)
    payload["_save_meta"] = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "n_items": len(payload.get("items", [])),
    }
    _rotate_backups(target)
    _atomic_write_json(target, payload)


def load_report_state(*, filename: Path | None = None) -> Dict[str, Any]:
    """Carga el estado del reporte. Si el archivo principal está
    corrupto, intenta recovery automático desde los backups.

    Ciclo 17.14.1 HOTFIX:
      - Si el JSON principal falla → intenta .bak.1, luego .bak.2, etc.
      - Si recupera desde backup, agrega `_recovered_from` al dict
        para que la UI pueda mostrar un banner al usuario
      - Si TODOS fallan, devuelve {} pero con `_load_error` poblado
        para que la UI sepa que hubo un problema (no silent fail)
    """
    target = filename or REPORT_STATE_FILE
    if not target.exists():
        return {"items": [], "meta": {}}

    # Intento principal
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        result = _restore_state(raw)
        # No agregamos _recovered_from porque cargó normal
        return result
    except Exception as primary_err:
        # Recovery desde backups (1..MAX, del más reciente al más viejo)
        for i in range(1, MAX_BACKUPS + 1):
            bk = _backup_path(target, i)
            if not bk.exists():
                continue
            try:
                raw = json.loads(bk.read_text(encoding="utf-8"))
                result = _restore_state(raw)
                # Marcar de dónde se recuperó para que la UI avise
                result["_recovered_from"] = bk.name
                result["_recovered_at"] = datetime.now().isoformat(timespec="seconds")
                # Reescribir el archivo principal con el contenido del backup
                # para que las próximas cargas no tengan que recurrir al backup.
                # (Si esto falla, no es bloqueante — la lectura ya tuvo éxito)
                try:
                    _atomic_write_json(target, raw)
                except Exception:
                    pass
                return result
            except Exception:
                continue
        # Ningún backup salvable
        return {
            "items": [],
            "meta": {},
            "_load_error": str(primary_err),
            "_load_error_at": datetime.now().isoformat(timespec="seconds"),
        }


def clear_report_state(*, filename: Path | None = None) -> None:
    """Borra el estado actual del reporte.
    Ciclo 17.14.1: NO toca los backups (.bak.*) — siguen disponibles
    si después el usuario quiere recuperar.
    """
    target = filename or REPORT_STATE_FILE
    if target.exists():
        target.unlink()


def list_available_backups(filename: Path | None = None) -> List[Dict[str, Any]]:
    """Devuelve lista de backups disponibles con metadata útil.
    Útil para UI de "restaurar desde backup específico" si se necesita.
    """
    target = filename or REPORT_STATE_FILE
    out: List[Dict[str, Any]] = []
    for i in range(1, MAX_BACKUPS + 1):
        bk = _backup_path(target, i)
        if not bk.exists():
            continue
        try:
            stat = bk.stat()
            # Intentar leer el _save_meta del backup
            saved_at = ""
            n_items = 0
            try:
                raw = json.loads(bk.read_text(encoding="utf-8"))
                sm = raw.get("_save_meta", {}) or {}
                saved_at = sm.get("saved_at", "") or ""
                n_items = int(sm.get("n_items", 0) or 0)
            except Exception:
                pass
            out.append({
                "backup_idx": i,
                "filename": bk.name,
                "path": str(bk),
                "size_bytes": stat.st_size,
                "size_human": _human_size(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "saved_at": saved_at,
                "n_items": n_items,
            })
        except Exception:
            continue
    return out


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# =============================================================
# Ciclo 17.5.6 — Helpers compartidos para envío al reporte
# =============================================================
# Bug histórico: cada módulo (Polar, Bode, SCL, Spectrum,
# Time Waveform, Trends, Tabular) hacía
# st.session_state.report_items.append(...) directo. La página
# Reports cargaba report_state.json desde disco SOLO la primera
# vez que se entraba a Reports en la sesión, vía un flag
# `report_state_loaded`. Si el usuario añadía items desde un
# módulo ANTES de visitar Reports por primera vez, al entrar a
# Reports se cargaba el estado de disco encima del estado en
# memoria — perdiendo los items recién agregados. Por eso la UX
# era "envío al reporte → no aparece → debo ir a Reports y
# volver y reenviar para que cargue".
#
# Solución: dos helpers compartidos que TODOS los módulos
# llaman antes/después de modificar report_items:
#
#   ensure_report_state_loaded()
#       Carga report_state.json a session_state si todavía no
#       se cargó. Idempotente vía flag. Hace merge con lo que
#       ya esté en memoria (priority a memoria sobre disco para
#       casos de race entre upload y load).
#
#   append_report_item_and_persist(item)
#       Asegura load + appende item + persiste a disco. Así la
#       fuente de verdad (disco) siempre queda sincronizada con
#       memoria, y aunque el usuario nunca abra Reports, el
#       reporte queda armado.

def ensure_report_state_loaded() -> None:
    """Asegura que `st.session_state['report_items']` y
    `st.session_state['report_meta']` reflejen el contenido
    persistido en disco. Idempotente — múltiples llamadas en la
    misma sesión sólo cargan una vez.

    Si el caller ya añadió items a memoria antes de la primera
    carga (caso del bug original), se hace merge: los items en
    memoria se preservan y se añaden los faltantes del disco
    detectados por id.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return

    if st.session_state.get("report_state_loaded"):
        return

    persisted = load_report_state()
    disk_items = persisted.get("items", []) if isinstance(persisted, dict) else []
    disk_meta = persisted.get("meta", {}) if isinstance(persisted, dict) else {}

    # Ciclo 17.14.1 — Propagar flags de recovery/error a session_state
    # para que la UI de Reports pueda mostrar banners visibles al usuario.
    if isinstance(persisted, dict):
        rec_from = persisted.get("_recovered_from")
        rec_at = persisted.get("_recovered_at")
        load_err = persisted.get("_load_error")
        load_err_at = persisted.get("_load_error_at")
        if rec_from:
            st.session_state["wm_report_recovered_from"] = rec_from
            st.session_state["wm_report_recovered_at"] = rec_at or ""
            st.session_state["wm_report_recovered_n_items"] = len(disk_items)
        if load_err:
            st.session_state["wm_report_load_error"] = load_err
            st.session_state["wm_report_load_error_at"] = load_err_at or ""

    # Si memoria ya tiene items (race del bug), preservamos esos y
    # adjuntamos los faltantes del disco por id.
    in_memory = list(st.session_state.get("report_items", []) or [])
    if in_memory:
        seen_ids = {str(it.get("id") or "") for it in in_memory if it.get("id")}
        merged = list(in_memory)
        if isinstance(disk_items, list):
            for it in disk_items:
                if not isinstance(it, dict):
                    continue
                _iid = str(it.get("id") or "")
                if _iid and _iid in seen_ids:
                    continue
                merged.append(it)
        st.session_state["report_items"] = merged
    else:
        st.session_state["report_items"] = disk_items if isinstance(disk_items, list) else []

    if not st.session_state.get("report_meta"):
        st.session_state["report_meta"] = (
            disk_meta if isinstance(disk_meta, dict) else {}
        )

    st.session_state["report_state_loaded"] = True


def append_report_item_and_persist(item: Dict[str, Any]) -> bool:
    """Añade un item al reporte y lo persiste a disco al mismo
    tiempo. Devuelve True si la persistencia tuvo éxito (la
    adición a memoria siempre se hace).

    Pensado para uso desde cualquier módulo (Polar, Bode, SCL,
    Spectrum, Time Waveform, Trends, Tabular) — reemplaza el
    patrón frágil ``st.session_state.report_items.append(item)``
    que dependía de que Reports estuviera ya inicializado.
    """
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return False

    ensure_report_state_loaded()
    st.session_state.setdefault("report_items", [])
    if isinstance(item, dict):
        st.session_state["report_items"].append(item)

    try:
        save_report_state(
            items=st.session_state["report_items"],
            meta=st.session_state.get("report_meta", {}) or {},
        )
        return True
    except Exception:
        # La memoria ya está actualizada, lo único que falla es
        # la persistencia a disco — no rompemos al usuario.
        return False


def _draft_path(draft_name: Any) -> Path:
    REPORT_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DRAFTS_DIR / f"{_safe_slug(draft_name)}.json"


def list_report_drafts() -> List[str]:
    if not REPORT_DRAFTS_DIR.exists():
        return []

    drafts: List[str] = []
    for path in sorted(REPORT_DRAFTS_DIR.glob("*.json")):
        drafts.append(path.stem)
    return drafts


def save_named_report_draft(*, draft_name: Any, items: Any, meta: Any) -> str:
    target = _draft_path(draft_name)
    payload = _serialize_state(items=items, meta=meta)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target.stem


def load_named_report_draft(draft_name: Any) -> Dict[str, Any]:
    target = _draft_path(draft_name)
    if not target.exists():
        return {"items": [], "meta": {}}
    return load_report_state(filename=target)


def delete_named_report_draft(draft_name: Any) -> None:
    target = _draft_path(draft_name)
    if target.exists():
        target.unlink()
