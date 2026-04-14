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
