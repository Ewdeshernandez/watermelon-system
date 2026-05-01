from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_STATE_FILE = DATA_DIR / "report_state.json"
REPORT_DRAFTS_DIR = DATA_DIR / "report_drafts"


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


def save_report_state(*, items: Any, meta: Any, filename: Path | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = filename or REPORT_STATE_FILE
    payload = _serialize_state(items=items, meta=meta)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_report_state(*, filename: Path | None = None) -> Dict[str, Any]:
    target = filename or REPORT_STATE_FILE
    if not target.exists():
        return {"items": [], "meta": {}}

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {"items": [], "meta": {}}

    return _restore_state(raw)


def clear_report_state(*, filename: Path | None = None) -> None:
    target = filename or REPORT_STATE_FILE
    if target.exists():
        target.unlink()


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
