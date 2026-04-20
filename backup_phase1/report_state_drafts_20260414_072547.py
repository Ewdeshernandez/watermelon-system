from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_STATE_FILE = DATA_DIR / "report_state.json"


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


def save_report_state(*, items: Any, meta: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "items": sanitize_report_items(items),
        "meta": meta if isinstance(meta, dict) else {},
    }

    REPORT_STATE_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_report_state() -> Dict[str, Any]:
    if not REPORT_STATE_FILE.exists():
        return {"items": [], "meta": {}}

    try:
        raw = json.loads(REPORT_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"items": [], "meta": {}}

    return {
        "items": restore_report_items(raw.get("items", [])),
        "meta": raw.get("meta", {}) if isinstance(raw.get("meta", {}), dict) else {},
    }


def clear_report_state() -> None:
    if REPORT_STATE_FILE.exists():
        REPORT_STATE_FILE.unlink()
