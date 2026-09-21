"""
core.briefing_recommendations — Recomendaciones editables por activo
====================================================================

Las recomendaciones del briefing NO las inventa el sistema: las gestiona
el especialista y PERSISTEN entre reportes. Cada una tiene texto + fecha
de inicio (cuándo se emitió por primera vez). El PDF las lista con la
fecha en gris opaco. Cuando el cliente ejecuta una recomendación, el
especialista la borra (o la edita si cambia el alcance).

Persistencia: captured_parameters de la instancia (repositorio activo →
Supabase), clave "briefing_recommendations". Headless (sin Streamlit),
así el cron del lunes las toma igual que la UI.

Modelo de cada recomendación:
    {"id": str, "text": str, "started_at": "YYYY-MM-DD"}

API:
    list_recommendations(instance_id)   -> List[Dict]
    save_recommendations(instance_id, recs) -> bool   (reemplaza el set)
    add_recommendation(instance_id, text, started_at=None) -> bool
    update_recommendation(instance_id, rec_id, text=None, started_at=None) -> bool
    delete_recommendation(instance_id, rec_id) -> bool
"""
from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

PARAM_KEY = "briefing_recommendations"


def _norm_date(val: Any) -> str:
    """Normaliza a 'YYYY-MM-DD'. Acepta date/datetime/pd.Timestamp/str ISO.
    Fallback: hoy (también para NaT/valores inválidos)."""
    if val is None or val == "":
        return date.today().isoformat()
    try:
        from datetime import datetime as _dt
        if isinstance(val, _dt):
            val = val.date()
        if isinstance(val, date):
            s = val.isoformat()[:10]
        else:
            s = str(val).strip()[:10]
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d)).isoformat()
    except Exception:
        return date.today().isoformat()


def _norm_rec(rec: Dict[str, Any]) -> Optional[Dict[str, str]]:
    text = str(rec.get("text", "") or "").strip()
    if not text:
        return None
    return {
        "id": str(rec.get("id") or uuid.uuid4().hex[:10]),
        "text": text,
        "started_at": _norm_date(rec.get("started_at")),
    }


def list_recommendations(instance_id: str) -> List[Dict[str, str]]:
    """Recomendaciones vigentes del activo, en el ORDEN EN QUE FUERON
    INGRESADAS/REDACTADAS (no se reordenan por fecha). La fecha es solo
    información de referencia y no debe afectar el orden de visualización."""
    try:
        from core.instance_state import get_instance_parameters
        raw = get_instance_parameters(instance_id).get(PARAM_KEY) or []
        recs = [r for r in (_norm_rec(x) for x in raw if isinstance(x, dict)) if r]
        # NO ordenar por fecha — se mantiene el orden de ingreso.
        return recs
    except Exception as e:
        log.warning("list_recommendations(%s) falló: %s", instance_id, e)
        return []


def save_recommendations(instance_id: str, recs: List[Dict[str, Any]]) -> bool:
    """Reemplaza el set completo (lo que usa el editor de la UI).
    Filas sin texto se descartan; lista vacía borra el parámetro."""
    try:
        from core.instance_state import update_instance_parameter
        clean = [r for r in (_norm_rec(x) for x in recs or [] if isinstance(x, dict)) if r]
        # NO ordenar por fecha — se conserva el orden en que el editor las pasó.
        # update_instance_parameter con "" elimina la clave → usar None/[] explícito
        return update_instance_parameter(instance_id, PARAM_KEY, clean or None)
    except Exception as e:
        log.warning("save_recommendations(%s) falló: %s", instance_id, e)
        return False


def add_recommendation(instance_id: str, text: str,
                       started_at: Any = None) -> bool:
    recs = list_recommendations(instance_id)
    recs.append({"id": uuid.uuid4().hex[:10], "text": text,
                 "started_at": _norm_date(started_at)})
    return save_recommendations(instance_id, recs)


def update_recommendation(instance_id: str, rec_id: str,
                          text: Optional[str] = None,
                          started_at: Any = None) -> bool:
    recs = list_recommendations(instance_id)
    hit = False
    for r in recs:
        if r["id"] == rec_id:
            hit = True
            if text is not None and str(text).strip():
                r["text"] = str(text).strip()
            if started_at is not None:
                r["started_at"] = _norm_date(started_at)
    return save_recommendations(instance_id, recs) if hit else False


def delete_recommendation(instance_id: str, rec_id: str) -> bool:
    recs = list_recommendations(instance_id)
    keep = [r for r in recs if r["id"] != rec_id]
    if len(keep) == len(recs):
        return False
    return save_recommendations(instance_id, keep)


# ---------------------------------------------------------------------------
# Propuestas del sistema — DESCARTADAS (para no volver a proponer lo mismo)
# ---------------------------------------------------------------------------
PARAM_KEY_DISMISSED = "briefing_reco_dismissed"


def _norm_text(t: str) -> str:
    """Clave de comparación de una recomendación (dedup): minúsculas, sin
    espacios extra ni puntuación final."""
    import re
    s = re.sub(r"\s+", " ", str(t or "").strip().lower())
    return s.rstrip(" .;:")


def list_dismissed(instance_id: str) -> List[str]:
    """Textos normalizados de propuestas del sistema que el analista descartó
    (no se vuelven a proponer)."""
    try:
        from core.instance_state import get_instance_parameters
        raw = get_instance_parameters(instance_id).get(PARAM_KEY_DISMISSED) or []
        return [str(x) for x in raw if str(x).strip()]
    except Exception as e:
        log.warning("list_dismissed(%s) falló: %s", instance_id, e)
        return []


def dismiss_proposal(instance_id: str, text: str) -> bool:
    """Marca una propuesta del sistema como descartada (por su texto normalizado)."""
    try:
        from core.instance_state import update_instance_parameter
        key = _norm_text(text)
        if not key:
            return False
        cur = list_dismissed(instance_id)
        if key not in cur:
            cur.append(key)
        return update_instance_parameter(instance_id, PARAM_KEY_DISMISSED, cur)
    except Exception as e:
        log.warning("dismiss_proposal(%s) falló: %s", instance_id, e)
        return False


def clear_dismissed(instance_id: str) -> bool:
    """Reactiva todas las propuestas descartadas (vuelven a proponerse)."""
    try:
        from core.instance_state import update_instance_parameter
        return update_instance_parameter(instance_id, PARAM_KEY_DISMISSED, None)
    except Exception as e:
        log.warning("clear_dismissed(%s) falló: %s", instance_id, e)
        return False


def filter_new_proposals(instance_id: str, proposals: List[str]) -> List[str]:
    """De una lista de propuestas del sistema, devuelve SOLO las que no están
    ya en la lista gestionada del analista ni fueron descartadas (dedup por
    texto normalizado)."""
    have = {_norm_text(r["text"]) for r in list_recommendations(instance_id)}
    gone = set(list_dismissed(instance_id))
    out, seen = [], set()
    for p in proposals or []:
        k = _norm_text(p)
        if not k or k in have or k in gone or k in seen:
            continue
        seen.add(k)
        out.append(str(p).strip())
    return out


__all__ = ["list_recommendations", "save_recommendations",
           "add_recommendation", "update_recommendation",
           "delete_recommendation", "PARAM_KEY",
           "PARAM_KEY_DISMISSED", "list_dismissed", "dismiss_proposal",
           "clear_dismissed", "filter_new_proposals"]
