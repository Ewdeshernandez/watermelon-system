"""
core.omnibox_search
===================

Búsqueda fuzzy global del Home (Ciclo 17.12 — Nivel 3).

Indexa y busca contra:
  - Instancias del Vault (tag, asset_class, location, profile_key, id)
  - Drafts de reportes (filename / draft_name)
  - Normas ISO/API (código, name, applies_to)

Devuelve un List[OmniHit] con:
  kind:         'instance' | 'report' | 'norm'
  title:        texto principal a mostrar
  subtitle:     texto secundario
  icon:         emoji
  score:        ranking (mayor = mejor match)
  target_page:  página a navegar (puede ser '')
  payload:      dict con info para activar la acción
                (ej. instance_id, draft_name, norm_code)

Ranking heurístico simple — sin dependencias externas:
  +50 si query es prefijo del campo principal
  +30 si query es substring exacto
  +15 por cada token del query encontrado en algún campo
  +10 bonus si query es match exacto del código
  -ranking ajustado por longitud del campo (más corto, mejor)

Filosofía: rápido, determinístico, sin Streamlit deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class OmniHit:
    kind: str = ""
    title: str = ""
    subtitle: str = ""
    icon: str = "•"
    score: int = 0
    target_page: str = ""
    payload: Dict[str, Any] = None


def _normalize(s: Any) -> str:
    return str(s or "").strip().lower()


def _score_match(query: str, *fields: str) -> int:
    """Calcula un score de match entre query y un conjunto de fields."""
    q = _normalize(query)
    if not q:
        return 0
    score = 0
    tokens = [t for t in q.split() if t]
    for f in fields:
        fl = _normalize(f)
        if not fl:
            continue
        if fl == q:
            score += 80  # match exacto
            continue
        if fl.startswith(q):
            score += 50
            continue
        if q in fl:
            score += 30
            continue
        # Tokens individuales
        hit_tokens = sum(1 for t in tokens if t in fl)
        if hit_tokens > 0:
            score += hit_tokens * 12
    return score


# =============================================================
# COLECTORES DE FUENTES
# =============================================================

def _search_instances(query: str) -> List[OmniHit]:
    out: List[OmniHit] = []
    try:
        from core.instance_state import list_instances
        instances = list_instances() or []
    except Exception:
        return out

    for inst in instances:
        s = _score_match(
            query,
            inst.get("tag", ""),
            inst.get("instance_id", ""),
            inst.get("profile_key", ""),
            inst.get("location", ""),
            inst.get("notes", ""),
        )
        if s <= 0:
            continue
        tag = inst.get("tag", "") or inst.get("instance_id", "")
        loc = inst.get("location", "") or ""
        prof = inst.get("profile_key", "") or ""
        sub_parts = [p for p in [prof, loc] if p]
        out.append(OmniHit(
            kind="instance",
            title=tag,
            subtitle=(" · ".join(sub_parts)) or "Vault · instancia",
            icon="🛡️",
            score=s,
            target_page="pages/00_Machinery_Library.py",
            payload={"instance_id": inst.get("instance_id", "")},
        ))
    return out


def _search_reports(query: str) -> List[OmniHit]:
    out: List[OmniHit] = []
    try:
        from core.report_state import list_report_drafts
        drafts = list_report_drafts() or []
    except Exception:
        return out

    for name in drafts:
        s = _score_match(query, name)
        if s <= 0:
            continue
        out.append(OmniHit(
            kind="report",
            title=name,
            subtitle="Reports · draft",
            icon="📄",
            score=s,
            target_page="pages/16_Reports.py",
            payload={"draft_name": name},
        ))
    return out


def _search_norms(query: str) -> List[OmniHit]:
    out: List[OmniHit] = []
    try:
        from core.iso_thresholds import list_norms
        norms = list_norms() or []
    except Exception:
        return out

    for n in norms:
        s = _score_match(
            query,
            n.get("code", ""),
            n.get("name", ""),
            n.get("applies_to", ""),
        )
        if s <= 0:
            continue
        out.append(OmniHit(
            kind="norm",
            title=n.get("name", n.get("code", "")),
            subtitle=f"{n.get('code', '')} · {n.get('n_classes', 0)} clases",
            icon="📐",
            score=s,
            target_page="pages/00_Machinery_Library.py",
            payload={"norm_code": n.get("code", "")},
        ))
    return out


# =============================================================
# API PÚBLICA
# =============================================================

def omnibox_search(query: str, limit: int = 8) -> List[OmniHit]:
    """Busca query contra instancias + reportes + normas y devuelve
    top N hits ordenados por score desc.

    Si query está vacío, devuelve [].
    """
    q = _normalize(query)
    if len(q) < 1:
        return []

    hits: List[OmniHit] = []
    hits.extend(_search_instances(q))
    hits.extend(_search_reports(q))
    hits.extend(_search_norms(q))

    # Orden: score desc, luego kind (instances primero), luego title
    KIND_ORDER = {"instance": 0, "report": 1, "norm": 2}
    hits.sort(key=lambda h: (-h.score, KIND_ORDER.get(h.kind, 9), h.title.lower()))

    return hits[:limit]


def kind_label(kind: str) -> str:
    return {
        "instance": "Activo",
        "report":   "Reporte",
        "norm":     "Norma",
    }.get(kind, kind)


def kind_color(kind: str) -> str:
    return {
        "instance": "#0ea5e9",
        "report":   "#a855f7",
        "norm":     "#10b981",
    }.get(kind, "#94a3b8")


__all__ = [
    "OmniHit",
    "omnibox_search",
    "kind_label",
    "kind_color",
]
