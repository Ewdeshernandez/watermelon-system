"""
core.health_score
=================

Health Score 0-100 por instancia (Ciclo 17.12 — Nivel 3).

Algoritmo determinístico que combina señales del Vault:

    Base                                      = 100
    − 35  si NO tiene norma ISO/API asignada
    − 20  si NO tiene baseline (last_balance_date vacío)
    − 15  si NO tiene documentos en Vault
    − 10  si baseline > 90 días (drift de balance)
    − 10  si tiene override pero sin justificación escrita
    +  5  si tiene reporte generado en últimos 30 días (bonus)
    + 10  si tiene reporte + norma + baseline reciente (combo bonus)

    clip a [0, 100]

Bandas:
    0-39   → CRÍTICO   (rojo)
    40-69  → ATENCIÓN  (ámbar)
    70-89  → BUENO     (lima)
    90-100 → ÓPTIMO    (verde)

Devuelve además el breakdown explicable para tooltip / drill-down,
por si el operador quiere saber por qué tal activo está en 65 en
lugar de 100.

Filosofía: NO depende de Streamlit ni de Plotly. Solo lectura del
Vault + paths del filesystem. Es testeable de forma determinística.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INSTANCES_DIR = DATA_DIR / "instances"
REPORT_DRAFTS_DIR = DATA_DIR / "report_drafts"


# =============================================================
# RESULTADO
# =============================================================

@dataclass
class HealthScore:
    score: int = 100             # 0..100
    band: str = "optimo"         # critico|atencion|bueno|optimo
    band_label: str = "ÓPTIMO"
    color: str = "#10b981"
    breakdown: List[Dict[str, Any]] = field(default_factory=list)
    one_liner: str = ""          # Texto corto para tooltip


_BANDS = [
    # (low, high_excl, key, label, color)
    (90, 101, "optimo",  "ÓPTIMO",   "#10b981"),
    (70,  90, "bueno",   "BUENO",    "#84cc16"),
    (40,  70, "atencion","ATENCIÓN", "#f59e0b"),
    ( 0,  40, "critico", "CRÍTICO",  "#ef4444"),
]


def _band_for_score(score: int):
    for low, high, key, label, color in _BANDS:
        if low <= score < high:
            return key, label, color
    return "critico", "CRÍTICO", "#ef4444"


# =============================================================
# UTILS DE LECTURA
# =============================================================

def _parse_date_safe(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:len(fmt) + 5], fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except Exception:
        return None


def _has_recent_report_for_tag(tag: str, within_days: int = 30) -> bool:
    """Busca si existe algún draft de reporte cuyo nombre/contenido
    referencie esta instancia, modificado en los últimos N días.
    """
    if not REPORT_DRAFTS_DIR.exists() or not tag:
        return False
    cutoff = datetime.now() - timedelta(days=within_days)
    tag_low = tag.lower()
    for path in REPORT_DRAFTS_DIR.glob("*.json"):
        try:
            mt = datetime.fromtimestamp(path.stat().st_mtime)
            if mt < cutoff:
                continue
            # Heurística: filename contiene el tag, o los items mencionan
            if tag_low in path.stem.lower():
                return True
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")[:50_000]
                if tag_low in txt.lower():
                    return True
            except Exception:
                pass
        except Exception:
            continue
    return False


# =============================================================
# CÁLCULO PRINCIPAL
# =============================================================

def compute_health_score(instance_data: Dict[str, Any]) -> HealthScore:
    """Calcula el HealthScore para los datos completos de una instancia.

    Args:
        instance_data: dict con campos del Instance (al menos:
            tag, iso_norm_code, last_balance_date,
            documents (list), override_justification,
            setpoint_warning_override, setpoint_danger_override).

    Returns:
        HealthScore con score, band, color, breakdown explicable.
    """
    score = 100
    breakdown: List[Dict[str, Any]] = []

    # Penalidades
    norm = (instance_data.get("iso_norm_code") or "").strip()
    if not norm:
        score -= 35
        breakdown.append({
            "label": "Sin norma ISO/API asignada",
            "delta": -35, "kind": "-",
        })

    last_bal = (instance_data.get("last_balance_date") or "").strip()
    if not last_bal:
        score -= 20
        breakdown.append({
            "label": "Sin baseline de balanceo registrado",
            "delta": -20, "kind": "-",
        })

    docs = instance_data.get("documents", []) or []
    n_docs = len(docs) if isinstance(docs, list) else 0
    if n_docs == 0:
        score -= 15
        breakdown.append({
            "label": "Sin documentos en Vault",
            "delta": -15, "kind": "-",
        })

    # Drift de baseline
    bal_dt = _parse_date_safe(last_bal)
    if bal_dt is not None:
        age_days = (datetime.now() - bal_dt).days
        if age_days > 90:
            score -= 10
            breakdown.append({
                "label": f"Baseline viejo ({age_days} días, recomendado <90)",
                "delta": -10, "kind": "-",
            })

    # Override sin justificación
    has_override = bool(
        float(instance_data.get("setpoint_warning_override", 0) or 0) > 0
        or float(instance_data.get("setpoint_danger_override", 0) or 0) > 0
    )
    has_just = bool((instance_data.get("override_justification") or "").strip())
    if has_override and not has_just:
        score -= 10
        breakdown.append({
            "label": "Override del especialista sin justificación escrita",
            "delta": -10, "kind": "-",
        })

    # Bonus por reporte reciente
    tag = (instance_data.get("tag") or "").strip()
    if tag and _has_recent_report_for_tag(tag, within_days=30):
        score += 5
        breakdown.append({
            "label": "Reporte generado en los últimos 30 días",
            "delta": +5, "kind": "+",
        })
        # Combo bonus si además tiene norma + baseline reciente
        if norm and bal_dt and (datetime.now() - bal_dt).days <= 90:
            score += 10
            breakdown.append({
                "label": "Combo: norma + baseline reciente + reporte (excelencia)",
                "delta": +10, "kind": "+",
            })

    # Clip
    score = max(0, min(100, score))
    band, label, color = _band_for_score(score)

    # One-liner según banda
    if band == "optimo":
        one = "Activo en condiciones óptimas — sigue así."
    elif band == "bueno":
        one = "Activo en buenas condiciones — pequeños puntos de mejora."
    elif band == "atencion":
        one = "Atención: faltan piezas clave (norma, baseline o reportes)."
    else:
        one = "Crítico: este activo no está adecuadamente configurado."

    return HealthScore(
        score=int(round(score)),
        band=band,
        band_label=label,
        color=color,
        breakdown=breakdown,
        one_liner=one,
    )


def compute_health_score_for_instance_id(instance_id: str) -> HealthScore:
    """Versión por ID que carga la instancia del Vault."""
    try:
        from core.instance_state import get_instance
        inst = get_instance(instance_id)
        if inst is None:
            return HealthScore(score=0, band="critico", band_label="SIN DATOS",
                               color="#94a3b8", one_liner="Instancia no encontrada.")
        # Convertir Instance a dict mínimo
        data = {
            "tag": getattr(inst, "tag", "") or "",
            "iso_norm_code": getattr(inst, "iso_norm_code", "") or "",
            "last_balance_date": getattr(inst, "last_balance_date", "") or "",
            "documents": getattr(inst, "documents", []) or [],
            "setpoint_warning_override": getattr(inst, "setpoint_warning_override", 0.0) or 0.0,
            "setpoint_danger_override": getattr(inst, "setpoint_danger_override", 0.0) or 0.0,
            "override_justification": getattr(inst, "override_justification", "") or "",
        }
        return compute_health_score(data)
    except Exception:
        return HealthScore(score=0, band="critico", band_label="ERROR",
                           color="#ef4444", one_liner="No se pudo calcular el score.")


# =============================================================
# SVG GAUGE SEMICIRCULAR (tipo Bently)
# =============================================================
#
# Dibuja un gauge de 180° con:
#   - Arco fondo gris claro
#   - Arco de progreso coloreado según banda
#   - Aguja apuntando al valor
#   - Número grande al centro
#
# Devuelve un string HTML/SVG listo para st.markdown(.., unsafe_html).

def render_score_gauge(score: int, color: str, size: int = 110) -> str:
    """SVG semicircular gauge con score 0-100.

    size: ancho en px (alto = ~size/2 + padding)
    """
    score = max(0, min(100, int(score)))
    w = size
    h = int(size * 0.62)
    cx = w / 2
    cy = h * 0.95
    r = (size / 2) * 0.86

    import math
    # Ángulo: 180° (izquierda) → 360° (derecha) en notación SVG
    # Mapeamos score 0..100 a 180..360
    angle_deg = 180 + (score / 100.0) * 180
    angle_rad = math.radians(angle_deg)
    px = cx + r * math.cos(angle_rad)
    py = cy + r * math.sin(angle_rad)

    # Path del arco fondo (semicirculo completo)
    # M (cx-r, cy) A r r 0 0 1 (cx+r, cy)
    bg_path = f"M {cx - r:.2f} {cy:.2f} A {r:.2f} {r:.2f} 0 0 1 {cx + r:.2f} {cy:.2f}"

    # Path del arco de progreso (de izq hasta el ángulo)
    # large_arc_flag = 0 si score < 50 (arco corto), 0 igual porque siempre <180
    fg_path = f"M {cx - r:.2f} {cy:.2f} A {r:.2f} {r:.2f} 0 0 1 {px:.2f} {py:.2f}"

    # IMPORTANTE: devolver el SVG en UNA sola línea sin comentarios
    # HTML. Streamlit pasa st.markdown a marko/markdown-it que rompe
    # bloques HTML al encontrar blank lines o ciertos comentarios,
    # y entonces el resto del card se muestra como texto plano.
    fs_num = int(size * 0.30)
    svg = (
        f'<svg width="{w}" height="{h + 8}" viewBox="0 0 {w} {h + 8}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;">'
        f'<path d="{bg_path}" fill="none" stroke="#e6ebf2" stroke-width="9" stroke-linecap="round"/>'
        f'<path d="{fg_path}" fill="none" stroke="{color}" stroke-width="9" stroke-linecap="round"/>'
        f'<text x="{cx}" y="{cy - 6}" text-anchor="middle" '
        f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
        f'font-size="{fs_num}" font-weight="800" fill="#0f172a">{score}</text>'
        f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
        f'font-family="ui-sans-serif,system-ui" font-size="9" font-weight="600" '
        f'fill="#94a3b8" letter-spacing="0.12em">/ 100</text>'
        f'</svg>'
    )
    return (
        f'<div style="display:flex;flex-direction:column;'
        f'align-items:center;margin:0;padding:0;">{svg}</div>'
    )


def render_score_pill(score: int, band_label: str, color: str) -> str:
    """Pill compacto alternativo al gauge para listas."""
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:999px;'
        f'background:{color}1A;color:{color};font-weight:800;font-size:11px;'
        f'letter-spacing:0.06em;">{band_label} · {score}</span>'
    )


__all__ = [
    "HealthScore",
    "compute_health_score",
    "compute_health_score_for_instance_id",
    "render_score_gauge",
    "render_score_pill",
]
