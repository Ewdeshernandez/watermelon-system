"""
pages/02_Live_Monitoring.py
===========================

Live Monitoring v3 — Pro industrial control room (Ciclo 23.3).

Diferenciador clave vs el legacy de la industria (System1, AMS, @ptitude):

  * Hero **Live Sensor Map**: schematic del activo con sensor dots vivos,
    coloreados por severidad y animados (pulse en danger).
  * **Sparklines** inline por sensor — micro-trend de últimas 30 lecturas.
  * **Phasor mini-charts** del 1X (vector amplitud + fase) por proximity.
  * **Severidad ISO/API** con fallback automático cuando faltan thresholds.
  * **Alarm strip** prominente cuando hay sensores en danger.
  * Refresh manual con botón (auto-refresh removido en v3.31.59 — ruido en
    análisis profundo, el operador refresca cuando lo necesita).

Nada de marcas externas en el copy — el diferenciador es nuestro pipeline.
"""

from __future__ import annotations

import base64
import math
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from core.auth import require_login, render_user_menu, require_role
from core.live_readings import (
    count_for_instance,
    history_for_metric,
    latest_for_instance,
    recent_history_all_direct,
)
from core.severity import (
    CLASS_LABELS,
    detect_asset_class,
    compute_severity as _core_compute_severity,
    family_from as _family_from,
    thresholds_for,
)
from core.ui_theme import apply_watermelon_page_style, page_header

st.set_page_config(page_title="Watermelon System | Live Monitoring", layout="wide")
require_login()
require_role(allowed_roles=("admin", "specialist", "client"))
render_user_menu()
apply_watermelon_page_style()


# ============================================================
# CSS local — animaciones + estilos industriales
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <style>
        @keyframes wm-live-pulse {
            0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.55); transform: scale(1); }
            70%  { box-shadow: 0 0 0 10px rgba(239,68,68,0);  transform: scale(1.05); }
            100% { box-shadow: 0 0 0 0 rgba(239,68,68,0);    transform: scale(1); }
        }
        @keyframes wm-dot-danger {
            0%   { r: 1.5; opacity: 1; }
            50%  { r: 2.6; opacity: 0.55; }
            100% { r: 1.5; opacity: 1; }
        }
        @keyframes wm-dot-alarm {
            0%, 100% { opacity: 1; }
            50%      { opacity: 0.55; }
        }

        .wm-live-dot {
            display: inline-block;
            width: 10px; height: 10px;
            border-radius: 50%;
            background: #ef4444;
            animation: wm-live-pulse 1.6s infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        .wm-live-badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 12px;
            border-radius: 999px;
            background: linear-gradient(135deg, #fef2f2 0%, #fff7ed 100%);
            border: 1px solid #fecaca;
            color: #991b1b;
            font-weight: 800; font-size: 11px; letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .wm-asset-title {
            font-size: 28px; font-weight: 800; color: #0f172a;
            letter-spacing: -0.01em;
        }
        .wm-asset-sub {
            font-size: 13px; color: #64748b;
            font-variant-numeric: tabular-nums;
        }
        .wm-kpi-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbe5f0;
            border-radius: 16px;
            padding: 14px 18px;
            box-shadow: 0 8px 22px rgba(15,23,42,0.04);
            min-height: 100px;
        }
        .wm-kpi-icon { font-size: 18px; margin-bottom: 4px; }
        .wm-kpi-label {
            font-size: 10px; color: #64748b;
            text-transform: uppercase; letter-spacing: 0.08em; font-weight: 800;
        }
        .wm-kpi-value {
            font-size: 28px; font-weight: 800; color: #0f172a;
            line-height: 1.1; font-variant-numeric: tabular-nums;
        }
        .wm-kpi-sub { font-size: 11px; color: #94a3b8; margin-top: 3px; }

        .wm-status-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-weight: 800; font-size: 10px; letter-spacing: 0.08em;
            text-transform: uppercase;
            min-width: 60px; text-align: center;
        }

        /* Tabla industrial densa */
        table.wm-live-table {
            width: 100%;
            border-collapse: separate; border-spacing: 0;
            font-size: 13px;
            background: #ffffff;
            border-radius: 14px; overflow: hidden;
            border: 1px solid #e5edf7;
            font-variant-numeric: tabular-nums;
        }
        table.wm-live-table thead tr {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        }
        table.wm-live-table thead th {
            text-align: left;
            color: #1d4ed8;
            font-weight: 800; font-size: 10px;
            text-transform: uppercase; letter-spacing: 0.08em;
            padding: 11px 14px;
            border-bottom: 2px solid #d9e8fb;
        }
        table.wm-live-table tbody td {
            padding: 11px 14px;
            border-bottom: 1px solid #f1f5f9;
            color: #0f172a;
        }
        table.wm-live-table tbody tr:last-child td { border-bottom: none; }
        table.wm-live-table tbody tr.row-alarm  { background: #fff8e7; }
        table.wm-live-table tbody tr.row-danger { background: #fef2f2; }
        table.wm-live-table tbody tr.row-danger td:first-child { border-left: 3px solid #ef4444; }
        table.wm-live-table tbody tr.row-alarm  td:first-child { border-left: 3px solid #f59e0b; }
        table.wm-live-table .col-num {
            text-align: right; font-weight: 700;
            font-family: "SF Mono", Menlo, Consolas, monospace;
        }
        table.wm-live-table .col-mono {
            font-family: "SF Mono", Menlo, Consolas, monospace;
            font-size: 12px; color: #475569;
        }
        table.wm-live-table .col-spark { width: 100px; padding: 6px 14px; }

        /* Alarm strip */
        .wm-alarm-strip {
            background: linear-gradient(90deg, #7f1d1d 0%, #991b1b 50%, #7f1d1d 100%);
            color: #fee2e2;
            border-radius: 14px;
            padding: 12px 18px;
            margin-bottom: 14px;
            box-shadow: 0 12px 28px rgba(127,29,29,0.25);
            display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
            font-weight: 700;
        }
        .wm-alarm-strip .wm-alarm-icon {
            font-size: 24px;
            animation: wm-live-pulse 1.6s infinite;
            background: white; color: #ef4444;
            border-radius: 50%;
            width: 36px; height: 36px;
            display: flex; align-items: center; justify-content: center;
        }

        /* Machine Map hero — limpio, sin marco oscuro tapando el activo */
        .wm-map-hero {
            background: transparent;
            border: none;
            padding: 0;
        }
        .wm-map-frame {
            background: white;
            border-radius: 14px;
            position: relative;
            overflow: hidden;
            border: 1px solid #e5edf7;
            box-shadow: 0 8px 22px rgba(15,23,42,0.06);
        }
        .wm-map-legend {
            display: flex; gap: 16px;
            padding: 10px 4px 0;
            font-size: 10px; color: #64748b;
            text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700;
        }
        .wm-map-legend .lg-dot {
            display: inline-block; width: 9px; height: 9px;
            border-radius: 50%; margin-right: 5px; vertical-align: middle;
        }

        /* Phasor card */
        .wm-phasor-card {
            background: white;
            border: 1px solid #e5edf7;
            border-radius: 14px;
            padding: 14px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(15,23,42,0.04);
        }
        .wm-phasor-label {
            font-weight: 800; color: #0f172a; font-size: 13px;
            margin-bottom: 6px; letter-spacing: 0.04em;
        }
        .wm-phasor-values {
            font-family: "SF Mono", Menlo, monospace;
            font-size: 11px; color: #475569;
            margin-top: 4px;
        }
        </style>
        """
    ).strip(),
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS — tiempo, severidad, formato
# ============================================================

def _format_age(captured_at_iso: str) -> str:
    try:
        captured = datetime.fromisoformat(captured_at_iso.replace("Z", "+00:00"))
    except Exception:
        return "—"
    delta = (datetime.now(timezone.utc) - captured).total_seconds()
    if delta < 0:
        return "ahora"
    if delta < 60:
        return f"{int(delta)} s"
    if delta < 3600:
        return f"{int(delta / 60)} min"
    if delta < 86400:
        return f"{int(delta / 3600)} h"
    return f"{int(delta / 86400)} d"


def _seconds_since(captured_at_iso: str) -> float:
    try:
        captured = datetime.fromisoformat(captured_at_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - captured).total_seconds()
    except Exception:
        return 999999.0


def _staleness_color(seconds_old: float) -> str:
    if seconds_old < 30:
        return "#22c55e"
    if seconds_old < 300:
        return "#f59e0b"
    return "#ef4444"


# Severity computation se delega 100% a core.severity (asset-class aware,
# OEM thresholds para LM6000 / Frame 9 / SGT, transparencia de fuente).
def compute_severity(
    value: Optional[float],
    sensor_match: Optional[Dict[str, Any]],
    unit: str,
    instance_obj: Any = None,
    sensor_type_hint: str = "",
) -> Dict[str, Any]:
    """Pasa a core.severity.compute_severity con instance context."""
    return _core_compute_severity(
        value=value,
        sensor_match=sensor_match,
        unit=unit,
        instance_obj=instance_obj,
        sensor_type_hint=sensor_type_hint,
    )


# ============================================================
# Timezone — convertir UTC del backend a hora local del cliente
# ============================================================

# Default a Colombia (donde están la mayoría de clientes actuales).
# A futuro mover a la metadata del instance (instance.timezone).
DEFAULT_TZ_NAME = "America/Bogota"


def _local_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(DEFAULT_TZ_NAME)
    except Exception:
        return timezone.utc


def _to_local(ts) -> Any:
    """Convierte un Timestamp / datetime UTC a hora local del cliente."""
    try:
        return pd.to_datetime(ts, utc=True).tz_convert(_local_tz())
    except Exception:
        return ts


def status_pill_html(status: str, fg: str, bg: str) -> str:
    return f'<span class="wm-status-pill" style="background:{bg};color:{fg};">{status}</span>'


def _build_sensor_lookup(instance_obj) -> Dict[str, Dict[str, Any]]:
    if instance_obj is None or not getattr(instance_obj, "sensors", None):
        return {}
    try:
        from core.sensor_map import sensor_label as _sensor_label_fn
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for s in instance_obj.sensors or []:
        try:
            lbl = _sensor_label_fn(s)
            out[lbl] = s
        except Exception:
            continue
    return out


# ============================================================
# Sparkline SVG
# ============================================================

def sparkline_svg(values: List[float], color: str = "#3b82f6", width: int = 90, height: int = 22) -> str:
    """Mini line chart SVG. ~22px alto, sin labels — solo la silueta."""
    pts = [v for v in values if v is not None and isinstance(v, (int, float))]
    if len(pts) < 2:
        return ""
    vmin = min(pts)
    vmax = max(pts)
    rng = max(vmax - vmin, 1e-9)
    pad_y = 2
    points = []
    for i, v in enumerate(pts):
        x = i * width / (len(pts) - 1)
        y = (height - pad_y * 2) - ((v - vmin) / rng) * (height - pad_y * 2) + pad_y
        points.append(f"{x:.1f},{y:.1f}")
    path = "M " + " L ".join(points)
    last_x, last_y = points[-1].split(",")
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="display:block;">'
        f'<path d="{path}" stroke="{color}" stroke-width="1.5" fill="none" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2.2" fill="{color}"/>'
        f'</svg>'
    )


# ============================================================
# Phasor SVG (1X vector visualization)
# ============================================================

def phasor_svg(amp: Optional[float], phase: Optional[float], max_amp: float, color: str = "#3b82f6", size: int = 110) -> str:
    """Mini polar plot showing amplitude/phase as a vector arrow."""
    cx = size / 2
    cy = size / 2
    radius = size / 2 - 8

    # Convención API 670: 0° arriba, sentido CCW estándar industrial
    if amp is None or phase is None or max_amp <= 0:
        end_x = cx
        end_y = cy
        valid = False
    else:
        try:
            ang_rad = math.radians(phase - 90.0)  # 0° en top
            ratio = min(max(float(amp) / max_amp, 0.0), 1.0)
            arrow_r = ratio * radius
            end_x = cx + arrow_r * math.cos(ang_rad)
            end_y = cy + arrow_r * math.sin(ang_rad)
            valid = True
        except Exception:
            end_x = cx
            end_y = cy
            valid = False

    grid = (
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="white" stroke="#cbd5e1" stroke-width="1"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{radius * 0.66:.1f}" fill="none" stroke="#e5e7eb" stroke-width="0.6"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{radius * 0.33:.1f}" fill="none" stroke="#e5e7eb" stroke-width="0.6"/>'
        f'<line x1="{cx - radius}" y1="{cy}" x2="{cx + radius}" y2="{cy}" stroke="#e5e7eb" stroke-width="0.6"/>'
        f'<line x1="{cx}" y1="{cy - radius}" x2="{cx}" y2="{cy + radius}" stroke="#e5e7eb" stroke-width="0.6"/>'
        f'<text x="{cx}" y="{cy - radius - 1}" text-anchor="middle" font-size="9" fill="#64748b" font-weight="700">0°</text>'
        f'<text x="{cx + radius + 6}" y="{cy + 3}" text-anchor="start" font-size="9" fill="#64748b" font-weight="700">90°</text>'
    )
    if valid:
        arrow = (
            f'<line x1="{cx}" y1="{cy}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
            f'stroke="{color}" stroke-width="2.2" stroke-linecap="round"/>'
            f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3" fill="{color}"/>'
        )
    else:
        arrow = (
            f'<text x="{cx}" y="{cy + 3}" text-anchor="middle" font-size="10" '
            f'fill="#94a3b8">— sin datos —</text>'
        )

    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="display:block;margin:0 auto;">'
        f'{grid}{arrow}'
        f'</svg>'
    )


# ============================================================
# Hero — Live Sensor Map vía Asset Library 2D (Ciclo 23.13)
# ============================================================
# Cuando la instancia tiene driver_icon_key + driven_icon_key, rinde
# el tren acoplado completo usando core.asset_library.composer en vez
# del PNG 3D legacy. Es lo que Bently System1 / Emerson AMS hacen:
# iconografía 2D vectorial curada por tipo de máquina + sensor dots
# en los anchors físicos correctos (DE / NDE / TRF / CRF).
# ============================================================

def _infer_side_anchor(
    sensor_label: str,
    sensor_match: Optional[Dict[str, Any]],
    instance_obj: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Mapea (sensor_label, sensor_dict) → (side, anchor) para el composer.

    Override explícito: si el sensor tiene icon_side / icon_anchor, los usa.
    Si no, aplica heurística por convención industrial Bently / API 670 —
    la numeración de bearings va del lado LIBRE del driver hacia el
    generador (corregido en Ciclo 23.19):
      - Label empieza con "1" → bearing #1 = NDE del driver / CRF (aero)
        = lado libre, intake del compresor
      - Label empieza con "2" → bearing #2 = DE del driver / TRF (aero)
        = lado coupling, output de la turbina
      - Label empieza con "3" → bearing #3 = DE del driven (lado coupling)
      - Label empieza con "4" → bearing #4 = NDE del driven (lado libre)
    Fallback a plane_label matching si label no empieza con dígito.

    Devuelve (None, None) si no se pudo mapear → sensor se omite del SVG.
    """
    # 1. Override explícito (wizard editor)
    if sensor_match:
        s_side = sensor_match.get("icon_side")
        s_anchor = sensor_match.get("icon_anchor")
        if s_side and s_anchor:
            return s_side, s_anchor

    drv_key = (getattr(instance_obj, "driver_icon_key", "") or "").lower()
    is_aero = "aero" in drv_key
    label_l = (sensor_label or "").strip().lower()
    plane_l = ((sensor_match or {}).get("plane_label") or "").lower()

    # 2. Convención Bently — primer carácter del label es el bearing #.
    # Numeración: desde el extremo LIBRE del driver hasta el extremo libre
    # del driven (cuenta a lo largo del tren mecánico).
    if label_l and label_l[0].isdigit():
        bearing_num = int(label_l[0])
        if bearing_num == 1:
            return "driver", ("CRF" if is_aero else "NDE")  # lado libre
        if bearing_num == 2:
            return "driver", ("TRF" if is_aero else "DE")   # lado coupling
        if bearing_num == 3:
            return "driven", "DE"                            # lado coupling
        if bearing_num == 4:
            return "driven", "NDE"                           # lado libre
        # 5+ → ignoramos (solo soportamos hasta gen NDE)
        return None, None

    # 3. Fallback por plane_label (sensores generados por wizard)
    side: Optional[str] = None
    anchor: Optional[str] = None
    if any(t in plane_l for t in ("driver", "motor", "turbina", "engine")):
        side = "driver"
    elif any(t in plane_l for t in (
        "driven", "compresor", "compressor", "generador", "generator",
        "bomba", "pump", "frame", "cilindro", "cylinder",
    )):
        side = "driven"

    if "nde" in plane_l:
        anchor = "CRF" if (side == "driver" and is_aero) else "NDE"
    elif "de" in plane_l:
        anchor = "TRF" if (side == "driver" and is_aero) else "DE"

    if side and anchor:
        return side, anchor
    return None, None


def _build_library_sensors(
    latest: List[Dict[str, Any]],
    sensor_lookup: Dict[str, Dict[str, Any]],
    instance_obj: Any,
    spark_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Construye la lista sensors_with_status para compose_train() a partir
    de las lecturas live y los sensores configurados de la instancia.
    Solo incluye sensores que se pudieron mapear a (side, anchor).

    spark_data (Ciclo 23.23): dict opcional sensor_label → [{value: ...}, ...]
    para alimentar la sparkline mini al lado del dot. Si None, los dots
    se renderizan sin sparkline (back-compat).
    """
    out: List[Dict[str, Any]] = []
    seen_anchors: set = set()  # evitar 2 sensores apilados en el mismo dot

    # Stale threshold (Ciclo 23.34): si una lectura tiene más de 60s
    # sin actualizar, se considera "stale" y se renderiza grayscale en el
    # diagrama. El operador ve de un vistazo qué sensores no están vivos.
    STALE_AGE_SECONDS = 60.0

    for r in latest:
        if r.get("metric") != "Direct":
            continue
        if (r.get("variable") or "").lower().startswith("velocidad"):
            continue
        lbl = r.get("sensor_label")
        if not lbl:
            continue
        sensor_match = sensor_lookup.get(lbl)
        side, anchor = _infer_side_anchor(lbl, sensor_match, instance_obj)
        if not side or not anchor:
            continue
        # Si ya hay un sensor en este anchor, lo dejamos pasar igual —
        # el composer renderiza ambos textos, solo el dot queda overlapped.
        # (Caso típico: 1Y_V y 1X_V están en el mismo cojinete pero ortogonales)
        unit = r.get("unit") or ""
        sev = compute_severity(r.get("value"), sensor_match, unit, instance_obj)
        try:
            val_num = float(r.get("value"))
            val_str = f"{val_num:.2f}"
        except Exception:
            val_num = None
            val_str = "—"
        title = (
            f"{lbl} | {val_str} {unit} | {sev['status']} | "
            f"alarm={sev['alarm']:.2f} / danger={sev['danger']:.2f} ({sev['source']})"
        )
        # Display label sin underscore (Ciclo 23.18) — '2Y_A' → '2YA'.
        # NO afecta CSV matches ni el sensor_lookup, que siguen usando lbl original.
        display_label = lbl.replace("_", "")

        # Sparkline values (Ciclo 23.23) — extraer last N readings de spark_data
        spark_values: Optional[List[float]] = None
        if spark_data:
            history = spark_data.get(lbl, [])
            extracted = [h.get("value") for h in history if h.get("value") is not None]
            try:
                spark_values = [float(v) for v in extracted]
                if len(spark_values) < 2:
                    spark_values = None
            except Exception:
                spark_values = None

        # Threshold setpoints (Ciclo 23.23) — solo si compute_severity devolvió
        # alarm/danger reales (>0). compute_severity garantiza floats pero
        # con 0 si no hay setpoint, así que filtramos.
        alarm_val = sev.get("alarm")
        danger_val = sev.get("danger")
        if alarm_val is not None and danger_val is not None:
            try:
                alarm_val = float(alarm_val)
                danger_val = float(danger_val)
                if alarm_val <= 0 or danger_val <= alarm_val:
                    alarm_val, danger_val = None, None
            except Exception:
                alarm_val, danger_val = None, None

        # Click-to-drill link (Ciclo 23.26) — path ABSOLUTO para sortear
        # el base href de Streamlit que rompía URLs relativas. La página
        # `02_Live_Monitoring.py` se sirve en `/Live_Monitoring`.
        from urllib.parse import quote as _urlquote
        link = f"/Live_Monitoring?sensor={_urlquote(lbl)}"

        # Stale check (Ciclo 23.34) — captured_at viene en ISO. Si
        # _seconds_since() falla por any razón, asumimos NO stale (no
        # ocultamos datos por error de parseo).
        try:
            age_sec = _seconds_since(r.get("captured_at"))
            is_stale = age_sec > STALE_AGE_SECONDS
        except Exception:
            is_stale = False

        out.append({
            "label": display_label,
            "side": side,
            "anchor": anchor,
            "status": sev["status"],
            "value": val_str,
            "unit": unit,
            "title": title,
            "spark_values": spark_values,
            "alarm": alarm_val,
            "danger": danger_val,
            "value_num": val_num,
            "link": link,
            "sensor_label": lbl,  # para que zoom panel pueda lookupear
            "is_stale": is_stale,
        })
        seen_anchors.add((side, anchor))
    return out


def render_sensor_map_library(
    instance_obj: Any,
    latest: List[Dict[str, Any]],
    sensor_lookup: Dict[str, Dict[str, Any]],
    spark_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> bool:
    """
    Renderiza el tren acoplado vía core.asset_library.composer.

    Devuelve True si pudo (la instancia tiene driver_icon_key + driven_icon_key),
    False si no hay icon_keys configuradas → caller debe usar fallback PNG.

    spark_data (Ciclo 23.23): opcional, sensor_label → list of reading dicts,
    se forwardea a _build_library_sensors para alimentar las sparklines de
    los dots.
    """
    drv_key = getattr(instance_obj, "driver_icon_key", "") or ""
    drvn_key = getattr(instance_obj, "driven_icon_key", "") or ""
    if not drv_key or not drvn_key:
        return False

    try:
        from core.asset_library.composer import compose_train
    except ImportError:
        return False

    sensors_with_status = _build_library_sensors(
        latest, sensor_lookup, instance_obj, spark_data=spark_data,
    )

    # Titles cortos y limpios (Ciclo 23.26) — el manufacturer/model
    # crudo de la instance trae duplicados ("GE Vernova / Brush GE
    # LM6000") que se ven feos en el SVG. Detectamos asset class y
    # generamos prefijos consistentes ("Turbina", "Generador", etc.).
    def _clean_model(text: str) -> str:
        if not text:
            return ""
        # Stripear prefijos repetidos del manufacturer (ej "GE LM6000" → "LM6000"
        # cuando manufacturer ya dice "GE")
        return text.strip()

    drv_prefix_map = {
        "gas_turbine_aero": "Turbina",
        "gas_turbine_industrial": "Turbina",
        "steam_turbine": "Turbina vapor",
        "electric_motor_sleeve": "Motor",
        "electric_motor_rolling": "Motor",
        "recip_engine_8cyl_inline": "Motor recip.",
        "recip_engine_16cyl_inline": "Motor recip.",
    }
    drvn_prefix_map = {
        "generator_synchronous": "Generador",
        "centrifugal_compressor": "Compresor",
        "recip_compressor": "Compresor recip.",
        "pump_centrifugal": "Bomba",
        "gearbox": "Gearbox",
    }

    drv_prefix = drv_prefix_map.get(drv_key, "")
    drvn_prefix = drvn_prefix_map.get(drvn_key, "")

    drv_model = _clean_model(instance_obj.driver_model or "")
    drvn_mfr = (instance_obj.driven_manufacturer or "").strip()
    drvn_model = _clean_model(instance_obj.driven_model or "")

    drv_label = f"{drv_prefix} {drv_model}".strip() if drv_prefix else (drv_model or "Driver")
    # Para driven preferimos manufacturer (Brush, Westinghouse) sobre model
    drvn_main = drvn_mfr or drvn_model
    drvn_label = f"{drvn_prefix} {drvn_main}".strip() if drvn_prefix else (drvn_main or "Driven")

    try:
        svg = compose_train(
            driver_key=drv_key,
            driven_key=drvn_key,
            driver_label=drv_label,
            driven_label=drvn_label,
            coupling=getattr(instance_obj, "coupling_class", "") or "flexible",
            sensors_with_status=sensors_with_status,
        )
    except Exception as e:
        st.caption(f"(Asset library no pudo rendir: {e}) — cayendo al PNG legacy.")
        return False

    # Ciclo 23.33 — SVG display puro. El click directo en SVG no es
    # viable en Streamlit (browser full-reload pierde session_state →
    # auth falla → redirect a login). Drill-down se hace via selectbox
    # discreto debajo del diagrama (usa st.rerun interno, mantiene auth).
    st.markdown(svg, unsafe_allow_html=True)

    # Ciclo 23.61 — Exportador HD. Botones de descarga debajo del diagrama
    # para que el operador pueda mandar el snapshot por WhatsApp, email, o
    # ponerlo en un reporte/PPT. SVG raw (vectorial, infinita resolución) +
    # PNG 4K (cairosvg si está disponible). El timestamp queda en el nombre
    # del archivo para auditoría.
    _render_export_bar(svg, getattr(instance_obj, "instance_id", "asset"))
    return True


@st.cache_data(show_spinner=False, ttl=300, max_entries=8)
def _svg_to_png_hd(svg_str: str, width: int = 4000) -> Optional[bytes]:
    """Convierte SVG → PNG de alta resolución.

    Intenta resvg-py primero (Rust, zero apt deps, wheels precompilados).
    Fallback a cairosvg si está disponible. Devuelve None si nada funciona.
    Cacheado 5 min para que el botón download no re-renderee cada click
    (el SVG es idéntico hasta que cambian los valores live).
    """
    # Path A: resvg-py (Ciclo 23.63 — preferido por zero system deps)
    try:
        import resvg_py
        # resvg_py.svg_to_bytes acepta svg_string + resolución del output
        png_list = resvg_py.svg_to_bytes(
            svg_string=svg_str,
            resolution=width,
        )
        # La API devuelve list[int] (bytes serialized) o bytes según versión
        if isinstance(png_list, (bytes, bytearray)):
            return bytes(png_list)
        if isinstance(png_list, list):
            return bytes(png_list)
    except Exception:
        pass

    # Path B: cairosvg fallback (si algún día se reinstalan system libs)
    try:
        import cairosvg
        return cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=width,
        )
    except Exception:
        return None


def _render_export_bar(svg: str, instance_id: str) -> None:
    """Barra de exportación debajo del SVG — SVG raw + PNG 4K."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    safe_id = (instance_id or "asset").replace("/", "_").replace(" ", "_")

    exp_cols = st.columns([2, 2, 1, 5])
    with exp_cols[0]:
        st.download_button(
            label="📐 SVG vectorial",
            data=svg.encode("utf-8"),
            file_name=f"{safe_id}_diagram_{ts}.svg",
            mime="image/svg+xml",
            use_container_width=True,
            help=(
                "Descarga el diagrama como SVG. Vectorial (escala infinita), "
                "ideal para reportes técnicos y edición en Illustrator/Inkscape."
            ),
            key=f"export_svg_{safe_id}",
        )
    with exp_cols[1]:
        png_bytes = _svg_to_png_hd(svg, width=4000)
        if png_bytes:
            st.download_button(
                label="🖼 PNG 4K",
                data=png_bytes,
                file_name=f"{safe_id}_diagram_{ts}.png",
                mime="image/png",
                use_container_width=True,
                help=(
                    "Descarga PNG de 4000 px de ancho. Ideal para WhatsApp, "
                    "email, PowerPoint, Slack."
                ),
                key=f"export_png_{safe_id}",
            )
        else:
            st.button(
                "🖼 PNG 4K (no disponible)",
                disabled=True,
                use_container_width=True,
                help=(
                    "cairosvg no está instalado. Reinstala dependencias del "
                    "servidor (requirements.txt + packages.txt)."
                ),
                key=f"export_png_disabled_{safe_id}",
            )
    with exp_cols[2]:
        st.caption(f"📅 {ts}")


# ============================================================
# Hero LEGACY — Live Sensor Map (PNG 3D + dots por severidad)
# Solo se usa cuando la instancia NO tiene driver_icon_key/driven_icon_key.
# ============================================================

def render_sensor_map_hero(
    instance_obj,
    instance_id: str,
    latest: List[Dict[str, Any]],
    sensor_lookup: Dict[str, Dict[str, Any]],
) -> bool:
    """
    Render del schematic con sensor dots vivos + value badge SIEMPRE
    visible (no solo en hover). Devuelve True si renderizó.
    """
    if instance_obj is None or not instance_obj.schematic_png:
        return False
    try:
        from core.instance_state import get_instance_document_bytes
        png_bytes = get_instance_document_bytes(instance_id, instance_obj.schematic_png)
    except Exception:
        return False
    if not png_bytes:
        return False

    img_b64 = base64.b64encode(png_bytes).decode()

    # Lookup readings → severity por sensor
    readings_by_sensor: Dict[str, Dict[str, Any]] = {}
    for r in latest:
        if r.get("metric") != "Direct":
            continue
        lbl = r.get("sensor_label")
        if not lbl:
            continue
        unit = r.get("unit") or ""
        sensor_match = sensor_lookup.get(lbl)
        sev = compute_severity(r.get("value"), sensor_match, unit, instance_obj)
        try:
            val_num = float(r.get("value"))
            val_str = f"{val_num:.2f}"
        except Exception:
            val_str = "—"
        readings_by_sensor[lbl] = {
            "value": val_str,
            "unit": unit,
            "status": sev["status"],
            "fg": sev["fg"], "bg": sev["bg"],
            "alarm": sev["alarm"], "danger": sev["danger"],
            "source": sev["source"],
        }

    # Build SVG dots
    dots_svg_parts: List[str] = []
    has_position = False
    try:
        from core.sensor_map import sensor_label as _lbl_fn
    except Exception:
        return False

    for sensor in (instance_obj.sensors or []):
        x_pct = sensor.get("x_pct")
        y_pct = sensor.get("y_pct")
        if x_pct is None or y_pct is None:
            continue
        has_position = True
        try:
            label = _lbl_fn(sensor)
        except Exception:
            continue
        info = readings_by_sensor.get(label, {})
        status = info.get("status", "No Data")

        # Color por severidad — sin recuadros con texto encima del esquema.
        # Inspiración: System1 / AMS Suite usan dots clean y panel separado
        # para los valores (no etiquetas pegadas al render del activo).
        if status == "Danger":
            fill = "#ef4444"
            anim = '<animate attributeName="r" values="1.8;2.6;1.8" dur="1.2s" repeatCount="indefinite"/>'
        elif status == "Alarma":
            fill = "#f59e0b"
            anim = '<animate attributeName="opacity" values="1;0.55;1" dur="1.6s" repeatCount="indefinite"/>'
        elif status == "Normal":
            fill = "#22c55e"; anim = ""
        elif status == "Sin Norma":
            fill = "#94a3b8"; anim = ""
        else:
            fill = "#64748b"; anim = ""

        cx = float(x_pct)
        cy = float(y_pct)
        val_str = info.get("value", "—")
        unit = info.get("unit", "")
        title = (
            f"{label} | {val_str} {unit} | {status} | "
            f"alarm={info.get('alarm', 0):.2f}/danger={info.get('danger', 0):.2f} "
            f"({info.get('source', 'n/a')})"
        )

        # ===== Diseño System1: label + valor compacto al lado del dot
        # Sin recuadros gigantes (esos eran horribles). Solo texto con
        # halo blanco (paint-order:stroke) que se lee sobre cualquier
        # fondo del render. Inspirado en cómo Bently System1 etiqueta
        # cada sensor con su valor numérico junto al dot.
        if status != "No Data" and val_str != "—":
            inline_text = f"{label} {val_str}"
        else:
            inline_text = label

        dots_svg_parts.append(
            f'<g><title>{title}</title>'
            # Halo translúcido del color del estado
            f'<circle cx="{cx}" cy="{cy}" r="2.6" fill="{fill}" fill-opacity="0.20" '
            f'stroke="{fill}" stroke-width="0.25" stroke-opacity="0.45"/>'
            # Dot interno con borde blanco
            f'<circle cx="{cx}" cy="{cy}" r="1.4" fill="{fill}" stroke="white" '
            f'stroke-width="0.55">{anim}</circle>'
            # Texto inline "label valor" arriba del dot
            f'<text x="{cx}" y="{cy - 3.4}" text-anchor="middle" '
            f'font-size="1.85" font-weight="800" '
            f'font-family="SF Mono, Menlo, Consolas, monospace" '
            f'fill="{fill}" letter-spacing="-0.04" '
            f'style="paint-order:stroke;stroke:white;stroke-width:0.7;'
            f'stroke-linejoin:round;">{inline_text}</text>'
            # Unidad chica debajo del label (si hay valor real)
            + (
                f'<text x="{cx}" y="{cy + 4.2}" text-anchor="middle" '
                f'font-size="1.3" font-weight="500" '
                f'font-family="SF Mono, Menlo, monospace" '
                f'fill="#475569" letter-spacing="-0.03" '
                f'style="paint-order:stroke;stroke:white;stroke-width:0.55;'
                f'stroke-linejoin:round;">{unit}</text>'
                if (status != "No Data" and unit) else ''
            )
            + '</g>'
        )

    if not has_position:
        # No tenemos coordenadas — no vale renderizar el hero, mejor el legacy
        return False

    legend = (
        '<div class="wm-map-legend">'
        '<span><span class="lg-dot" style="background:#22c55e;"></span>Normal</span>'
        '<span><span class="lg-dot" style="background:#f59e0b;"></span>Alarma</span>'
        '<span><span class="lg-dot" style="background:#ef4444;"></span>Danger</span>'
        '<span><span class="lg-dot" style="background:#94a3b8;"></span>Sin Norma</span>'
        '<span style="margin-left:auto;color:#94a3b8;text-transform:none;letter-spacing:0;">'
        'Hover sensor para ver valor</span>'
        '</div>'
    )

    html = textwrap.dedent(
        f"""
        <div class="wm-map-hero">
            <div class="wm-map-frame">
                <div style="position:relative;">
                    <img src="data:image/png;base64,{img_b64}" style="width:100%;display:block;"/>
                    <svg style="position:absolute;top:0;left:0;width:100%;height:100%;"
                         viewBox="0 0 100 100" preserveAspectRatio="none">
                        {''.join(dots_svg_parts)}
                    </svg>
                </div>
            </div>
            {legend}
        </div>
        """
    ).strip()

    st.markdown(html, unsafe_allow_html=True)
    return True


# ============================================================
# Sensor Zoom Panel — click-to-drill desde el diagrama
# ============================================================

def render_sensor_zoom_panel(
    selected_sensor: str,
    latest: List[Dict[str, Any]],
    sensor_lookup: Dict[str, Dict[str, Any]],
    instance_obj: Any,
    spark_data: Dict[str, List[Dict[str, Any]]],
) -> None:
    """
    Card focalizado para un sensor (Ciclo 23.24).

    Se dispara cuando el URL trae `?sensor=<lbl>` (de un click en el
    diagrama). Muestra:
      - Header con label + valor + status + thresholds
      - Mini trend chart con la data ya fetcheada en spark_data
      - Botones de cerrar y links a otras pages (Spectrum, Orbit)
    """
    match = next(
        (r for r in latest
         if r.get("sensor_label") == selected_sensor and r.get("metric") == "Direct"),
        None,
    )
    if not match:
        # Sensor no existe en latest — limpiar query param y avisar
        st.warning(
            f"Sensor `{selected_sensor}` no tiene lectura activa. "
            "Click otro dot del diagrama o ✕ para cerrar."
        )
        if st.button("✕ Cerrar selección", key="zoom_close_orphan"):
            st.query_params.clear()
            st.rerun()
        return

    sensor_match = sensor_lookup.get(selected_sensor)
    unit = match.get("unit") or ""
    sev = compute_severity(match.get("value"), sensor_match, unit, instance_obj)
    try:
        val_str = f"{float(match.get('value')):.3f}"
    except Exception:
        val_str = "—"

    display_label = selected_sensor.replace("_", "")
    status = sev["status"]
    alarm = sev.get("alarm") or 0
    danger = sev.get("danger") or 0

    # Color por severidad — paleta aligned con el composer
    color_map = {
        "Normal":    ("#22c55e", "#dcfce7"),
        "Alarma":    ("#f59e0b", "#fef3c7"),
        "Danger":    ("#ef4444", "#fee2e2"),
        "Sin Norma": ("#94a3b8", "#f1f5f9"),
        "No Data":   ("#64748b", "#e2e8f0"),
    }
    fg, bg = color_map.get(status, ("#64748b", "#e2e8f0"))

    threshold_txt = ""
    if alarm > 0 and danger > alarm:
        threshold_txt = f"alarm {alarm:.2f} · danger {danger:.2f}"

    st.markdown(
        textwrap.dedent(
            f"""
            <style>
            .wm-zoom-card {{
                border: 2px solid {fg};
                border-radius: 14px;
                padding: 14px 18px; margin: 14px 0 10px 0;
                background: {bg};
                box-shadow: 0 8px 24px rgba(15,23,42,0.10);
            }}
            .wm-zoom-header {{
                display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
            }}
            .wm-zoom-label {{
                font-size: 22px; font-weight: 900;
                font-family: SF Mono, Menlo, monospace;
                color: {fg}; letter-spacing: -0.01em;
            }}
            .wm-zoom-value {{
                font-size: 32px; font-weight: 900; color: #0f172a;
                font-variant-numeric: tabular-nums;
            }}
            .wm-zoom-unit {{
                font-size: 13px; font-weight: 700; color: #475569;
                margin-left: 4px;
            }}
            .wm-zoom-status {{
                background: {fg}; color: white;
                padding: 4px 12px; border-radius: 99px;
                font-size: 11px; font-weight: 800;
                letter-spacing: 0.1em; text-transform: uppercase;
            }}
            .wm-zoom-thresholds {{
                font-size: 11px; color: #475569;
                margin-left: auto;
                font-family: SF Mono, monospace;
            }}
            </style>
            <div class="wm-zoom-card">
                <div class="wm-zoom-header">
                    <span class="wm-zoom-label">📍 {display_label}</span>
                    <span class="wm-zoom-value">{val_str}<span class="wm-zoom-unit">{unit}</span></span>
                    <span class="wm-zoom-status">{status}</span>
                    <span class="wm-zoom-thresholds">{threshold_txt}</span>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    # Selector de gráfico (Ciclo 23.59 — Tendencia universal).
    # TODO el análisis es tendencia: Direct, 1X amp, 1X fase, 2X amp, 2X fase,
    # Gap. Cada métrica usa el mismo framework de plot (multi-sensor + bands +
    # paginación). Phasor / Polar plot quedan para más adelante.
    CHART_TYPE_TO_METRIC = {
        "📈 Tendencia · Direct":       "Direct",
        "📈 Tendencia · 1X amplitud":  "1X_Ampl",
        "📈 Tendencia · 1X fase":      "1X_Phase",
        "📈 Tendencia · 2X amplitud":  "2X_Ampl",
        "📈 Tendencia · 2X fase":      "2X_Phase",
        "📈 Tendencia · Gap":          "Gap",
    }
    sel_col, range_col, close_col = st.columns([3, 2, 1])
    with sel_col:
        chart_type = st.selectbox(
            "Tipo de gráfico",
            options=list(CHART_TYPE_TO_METRIC.keys()) + [
                "🔍 Espectro (próximamente)",
                "🌊 Forma de onda (próximamente)",
                "🌀 Orbit (próximamente)",
                "📊 Cascade / Waterfall (próximamente)",
            ],
            index=0,
            key=f"zoom_chart_type_{selected_sensor}",
            label_visibility="collapsed",
        )
    with range_col:
        if chart_type in CHART_TYPE_TO_METRIC:
            range_choice = st.selectbox(
                "Rango",
                ["Última hora", "6 horas", "24 horas", "7 días", "30 días", "Todo"],
                index=1,
                key=f"zoom_range_{selected_sensor}",
                label_visibility="collapsed",
                help="Cuanta historia traer del registro append-only en Supabase",
            )
        else:
            range_choice = "6 horas"
            st.empty()
    with close_col:
        if st.button("✕ Cerrar", key="zoom_close", use_container_width=True):
            st.query_params.clear()
            st.rerun()

    if chart_type in CHART_TYPE_TO_METRIC:
        # Tendencia universal (Ciclo 23.59) — funciona para Direct, 1X/2X
        # amplitud, 1X/2X fase, y Gap. Mismo framework: multi-sensor con
        # checkbox, paginación, severity bands (solo aplican para Direct),
        # X-axis datetime, Y-axis con unidad real de la metric.
        target_metric = CHART_TYPE_TO_METRIC[chart_type]
        inst_id = instance_obj.instance_id if hasattr(instance_obj, "instance_id") else ""

        # Todos los sensores de la instance que tienen al menos una lectura
        # de la metric seleccionada (latest).
        all_sensors_for_metric = sorted({
            r.get("sensor_label")
            for r in latest
            if r.get("metric") == target_metric and r.get("sensor_label")
        })

        if not all_sensors_for_metric:
            st.info(
                f"Ningún sensor de este activo envía la métrica `{target_metric}` "
                f"todavía. Verificá la configuración del collector o probá otro tipo "
                f"de gráfico."
            )
        else:
            # Sensor primario para el chart (color de severidad + KPIs + bands)
            primary = selected_sensor if selected_sensor in all_sensors_for_metric else all_sensors_for_metric[0]
            primary_row = next(
                (r for r in latest
                 if r.get("sensor_label") == primary and r.get("metric") == target_metric),
                None,
            ) or {}
            sensor_unit = primary_row.get("unit", "")
            metric_label = chart_type.replace("📈 Tendencia · ", "")

            # Checkbox grid de sensores — todos los disponibles para la metric.
            # CSS override para que NO use el rojo nativo de Streamlit.
            st.markdown("""
            <style>
            /* Override del check rojo de Streamlit a azul royal */
            div[data-testid="stCheckbox"] label > div:first-child > div:first-child > div[aria-checked="true"] {
                background-color: #2563eb !important;
                border-color: #2563eb !important;
            }
            div[data-testid="stCheckbox"] label > div:first-child > div:first-child > div {
                border-color: #94a3b8;
            }
            .wm-sensor-pickbar {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 10px 14px;
                margin: 4px 0 12px 0;
            }
            </style>
            """, unsafe_allow_html=True)

            # Init session_state para checkboxes (default: solo primary marcado)
            sensors_key = f"zoom_trend_sensors_{selected_sensor}_{target_metric}"
            if sensors_key not in st.session_state:
                st.session_state[sensors_key] = {primary: True}
            sensor_state = st.session_state[sensors_key]

            st.markdown('<div class="wm-sensor-pickbar">', unsafe_allow_html=True)
            with st.container():
                top_row = st.columns([6, 1, 1])
                with top_row[0]:
                    n_active = sum(1 for s in all_sensors_for_metric if sensor_state.get(s, s == primary))
                    st.caption(
                        f"🎚 **Sensores en gráfico** · {n_active}/{len(all_sensors_for_metric)} activos · "
                        f"metric `{target_metric}` · unidad `{sensor_unit or '—'}`"
                    )
                with top_row[1]:
                    if st.button("✓ Todos", key=f"all_{sensors_key}", use_container_width=True):
                        for s in all_sensors_for_metric:
                            st.session_state[sensors_key][s] = True
                        st.rerun()
                with top_row[2]:
                    if st.button("✗ Solo 1", key=f"clear_{sensors_key}", use_container_width=True):
                        st.session_state[sensors_key] = {primary: True}
                        st.rerun()

                # Grid de checkboxes (4 columnas)
                n_cols = 4
                cols = st.columns(n_cols)
                for i, s in enumerate(all_sensors_for_metric):
                    with cols[i % n_cols]:
                        current = sensor_state.get(s, s == primary)
                        new_val = st.checkbox(
                            s, value=current,
                            key=f"chk_{sensors_key}_{s}",
                        )
                        if new_val != current:
                            st.session_state[sensors_key][s] = new_val
            st.markdown('</div>', unsafe_allow_html=True)

            picked = [
                s for s in all_sensors_for_metric
                if st.session_state[sensors_key].get(s, s == primary)
            ]
            if not picked:
                picked = [primary]

            range_to_limit = {
                "Última hora": 400,
                "6 horas":     2200,
                "24 horas":    9000,
                "7 días":      62000,
                "30 días":     270000,
                "Todo":        500000,
            }
            limit = range_to_limit.get(range_choice, 2200)

            def _fetch_paginated(inst, var, met, total_limit, page_size=1000):
                """Paginación manual — PostgREST cap default es 1000 rows."""
                from core.live_readings import history_for_metric as _hfm
                first = _hfm(inst, var, met, limit=min(total_limit, page_size))
                if len(first) < page_size or total_limit <= page_size:
                    return first[:total_limit]
                from core.live_readings import _get_supabase_client, _TABLE
                client = _get_supabase_client()
                if client is None:
                    return first
                acc = list(first)
                while len(acc) < total_limit:
                    oldest_cursor = acc[-1].get("captured_at")
                    if not oldest_cursor:
                        break
                    try:
                        resp = (
                            client.table(_TABLE)
                            .select("captured_at,value,unit,quality")
                            .eq("instance_id", inst)
                            .eq("variable", var)
                            .eq("metric", met)
                            .lt("captured_at", oldest_cursor)
                            .order("captured_at", desc=True)
                            .limit(min(page_size, total_limit - len(acc)))
                            .execute()
                        )
                        chunk = list(getattr(resp, "data", []) or [])
                    except Exception:
                        break
                    if not chunk:
                        break
                    acc.extend(chunk)
                    if len(chunk) < page_size:
                        break
                return acc

            # Recolectar series por sensor — resolver variable real desde latest
            series_by_sensor = {}
            for s in picked:
                s_row = next(
                    (r for r in latest
                     if r.get("sensor_label") == s and r.get("metric") == target_metric),
                    None,
                )
                if not s_row:
                    continue
                s_var = s_row.get("variable", s)
                try:
                    rows = _fetch_paginated(inst_id, s_var, target_metric, limit)
                except Exception:
                    rows = []
                if rows:
                    series_by_sensor[s] = rows

            if not series_by_sensor:
                st.info(
                    f"Sin historial todavía para los sensores seleccionados en `{target_metric}`. "
                    "Esperá unos minutos para que el collector acumule lecturas."
                )
            else:
                import pandas as pd
                frames = []
                for s, rows in series_by_sensor.items():
                    dfx = pd.DataFrame(rows)
                    dfx["sensor_label"] = s
                    dfx["captured_at"] = pd.to_datetime(
                        dfx["captured_at"], utc=True
                    ).dt.tz_convert(_local_tz())
                    frames.append(dfx)
                df_trend = pd.concat(frames, ignore_index=True).sort_values(
                    by="captured_at"
                ).reset_index(drop=True)

                # KPIs sobre el sensor primario (3 columnas, sin Σ Lecturas)
                primary_df = df_trend[df_trend["sensor_label"] == primary]
                if primary_df.empty:
                    primary_df = df_trend
                c1, c2, c3 = st.columns(3)
                with c1: st.metric(f"Mín · {primary}", f"{primary_df['value'].min():.3f} {sensor_unit}")
                with c2: st.metric(f"Máx · {primary}", f"{primary_df['value'].max():.3f} {sensor_unit}")
                with c3: st.metric(f"Promedio · {primary}", f"{primary_df['value'].mean():.3f} {sensor_unit}")

                # Severity bands solo aplican para Direct (los thresholds están en
                # término de amplitud Direct, no de fase ni de gap voltage).
                show_bands = target_metric == "Direct"
                alarm_val = alarm if (alarm > 0 and show_bands) else 0
                danger_val = danger if (danger > 0 and show_bands) else 0

                try:
                    import plotly.graph_objects as go
                    palette = [
                        "#1d4ed8", "#b45309", "#15803d", "#7c3aed",
                        "#dc2626", "#0e7490", "#be185d", "#525252",
                    ]
                    fig = go.Figure()

                    if show_bands and danger_val > 0:
                        y_max = max(df_trend["value"].max(), danger_val) * 1.10
                        fig.add_hrect(
                            y0=danger_val, y1=y_max,
                            fillcolor="#fee2e2", opacity=0.50, line_width=0,
                            annotation_text="Danger",
                            annotation_position="top right",
                            annotation=dict(font=dict(color="#991b1b", size=11)),
                        )
                    if show_bands and alarm_val > 0 and danger_val > alarm_val:
                        fig.add_hrect(
                            y0=alarm_val, y1=danger_val,
                            fillcolor="#fef3c7", opacity=0.50, line_width=0,
                            annotation_text="Alarma",
                            annotation_position="top right",
                            annotation=dict(font=dict(color="#92400e", size=11)),
                        )
                    if show_bands and alarm_val > 0:
                        fig.add_hrect(
                            y0=0, y1=alarm_val,
                            fillcolor="#dcfce7", opacity=0.40, line_width=0,
                            annotation_text="Normal",
                            annotation_position="bottom right",
                            annotation=dict(font=dict(color="#166534", size=11)),
                        )

                    for idx, s in enumerate(picked):
                        sub = df_trend[df_trend["sensor_label"] == s]
                        if sub.empty:
                            continue
                        color = fg if s == primary else palette[idx % len(palette)]
                        line_w = 1.8 if s == primary else 1.2
                        fig.add_trace(go.Scatter(
                            x=sub["captured_at"],
                            y=sub["value"],
                            mode="lines",
                            line=dict(color=color, width=line_w),
                            name=s,
                            hovertemplate=(
                                f"<b>{s}</b><br>"
                                "%{x|%Y-%m-%d %H:%M:%S}<br>"
                                f"%{{y:.3f}} {sensor_unit}<extra></extra>"
                            ),
                        ))

                    fig.update_layout(
                        margin=dict(l=10, r=10, t=30, b=10),
                        height=420,
                        plot_bgcolor="white",
                        xaxis=dict(
                            showgrid=True, gridcolor="#f1f5f9",
                            title="",
                            tickformat="%d %b\n%H:%M",
                        ),
                        yaxis=dict(
                            showgrid=True, gridcolor="#f1f5f9",
                            title=f"{metric_label} ({sensor_unit})" if sensor_unit else metric_label,
                        ),
                        showlegend=len(picked) > 1,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", y=1.02,
                            xanchor="right", x=1,
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor="#e2e8f0", borderwidth=1,
                        ),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.caption(f"(plotly fallback: {e})")
                    pivot = df_trend.pivot_table(
                        index="captured_at", columns="sensor_label",
                        values="value", aggfunc="last",
                    )
                    st.line_chart(pivot, height=340)
    else:
        # Placeholder para Espectro / Waveform / Orbit / Cascade
        st.info(
            f"**{chart_type}** — esta vista todavía no está integrada en el "
            f"zoom panel. Pronto se integra inline para `{display_label}`."
        )


# ============================================================
# Header & Alarm strip
# ============================================================

def render_asset_header(
    instance_obj,
    instance_id: str,
    latest: Optional[List[Dict[str, Any]]] = None,
    severity_summary: Optional[Dict[str, int]] = None,
) -> None:
    """
    Asset banner card industrial — Ciclo 23.16.

    Reemplaza el header simple por una card grande estilo control room:
      - Title del activo (TES1) con tag + asset class
      - LIVE badge + status overall (Normal/Atención/Crítica) en grande
      - Driver / Driven / Cliente / Sitio en chips
      - 4 KPIs en grilla: velocidad, sensores activos, última lectura,
        alarmas (con color por severidad)
    """
    title_text = instance_id
    if instance_obj is not None and instance_obj.tag and instance_obj.tag != instance_id:
        title_text = f"{instance_obj.tag} · {instance_id}"

    # Asset class chip
    asset_class = detect_asset_class(instance_obj)
    class_label = CLASS_LABELS.get(asset_class, "—")
    class_color = {
        "aero_turbine":       ("#dbeafe", "#1e40af"),
        "industrial_turbine": ("#dcfce7", "#166534"),
        "recip_compressor":   ("#fef3c7", "#92400e"),
        "rotating_general":   ("#f1f5f9", "#475569"),
    }.get(asset_class, ("#f1f5f9", "#475569"))

    # Status overall — derivado del severity_summary
    summary = severity_summary or {}
    n_danger = summary.get("Danger", 0)
    n_alarm = summary.get("Alarma", 0)
    if n_danger > 0:
        status_label = "CRÍTICA"
        status_fg = "#7f1d1d"
        status_bg = "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
        status_border = "#ef4444"
        status_icon = "🚨"
    elif n_alarm > 0:
        status_label = "ATENCIÓN"
        status_fg = "#92400e"
        status_bg = "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
        status_border = "#f59e0b"
        status_icon = "⚠️"
    else:
        status_label = "OPERACIÓN NORMAL"
        status_fg = "#14532d"
        status_bg = "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)"
        status_border = "#22c55e"
        status_icon = "✓"

    # KPIs
    direct_rows = [r for r in (latest or []) if r.get("metric") == "Direct"]
    speed_row = next(
        (r for r in (latest or []) if (r.get("variable") or "").lower().startswith("velocidad")),
        None,
    )
    speed_val = speed_row.get("value") if speed_row else None
    speed_txt = f"{float(speed_val):.0f}" if speed_val is not None else "—"
    n_direct = len(direct_rows)

    if latest:
        min_age_seconds = min(_seconds_since(r["captured_at"]) for r in latest)
        oldest_row = min(latest, key=lambda r: _seconds_since(r["captured_at"]))
        age_txt = _format_age(oldest_row["captured_at"])
        age_color = _staleness_color(min_age_seconds)
    else:
        age_txt = "—"
        age_color = "#94a3b8"

    if n_danger + n_alarm == 0:
        alarms_txt = "0"
        alarms_color = "#16a34a"
    else:
        alarms_txt = f"{n_danger + n_alarm}"
        alarms_color = "#ef4444" if n_danger else "#f59e0b"

    # Chips (driver, driven, cliente, sitio) — compact inline pills
    chip_items: List[str] = []
    if instance_obj is not None:
        if instance_obj.driver_model:
            mfr = (instance_obj.driver_manufacturer or "").strip()
            mdl = instance_obj.driver_model.strip()
            chip_items.append(
                f'<span class="wm-bar-chip"><b>DRIVER</b>{(mfr + " " + mdl).strip()}</span>'
            )
        if instance_obj.driven_model:
            mfr = (instance_obj.driven_manufacturer or "").strip()
            mdl = instance_obj.driven_model.strip()
            chip_items.append(
                f'<span class="wm-bar-chip"><b>DRIVEN</b>{(mfr + " " + mdl).strip()}</span>'
            )
        if instance_obj.client:
            chip_items.append(
                f'<span class="wm-bar-chip"><b>CLIENTE</b>{instance_obj.client}</span>'
            )
        if instance_obj.site:
            chip_items.append(
                f'<span class="wm-bar-chip"><b>SITIO</b>{instance_obj.site}</span>'
            )
    chips_html = "".join(chip_items)

    # Asset banner compacto (Ciclo 23.23) — barra oscura de 1 línea con
    # título + KPIs inline + status pill. Reemplaza la card grande de
    # Ciclo 23.16 que ocupaba 1/3 de la pantalla. Diagrama es protagonista.
    # Ciclo 23.49 — STICKY: la barra se queda visible al scrollear.
    st.markdown(
        textwrap.dedent(
            f"""
            <style>
            .wm-asset-bar {{
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border-radius: 12px;
                padding: 12px 18px;
                margin: 4px 0 8px 0;
                display: flex; align-items: center; gap: 14px;
                flex-wrap: wrap;
                box-shadow: 0 6px 18px rgba(15,23,42,0.18);
                position: sticky;
                top: 56px;
                z-index: 50;
            }}
            .wm-bar-live {{
                display: inline-flex; align-items: center; gap: 7px;
                color: #f1f5f9; font-weight: 800; font-size: 12px;
                letter-spacing: 0.18em;
            }}
            .wm-bar-live-dot {{
                width: 8px; height: 8px; border-radius: 50%;
                background: #ef4444;
                animation: wm-pulse 1.4s ease-in-out infinite;
                box-shadow: 0 0 8px rgba(239,68,68,0.7);
            }}
            @keyframes wm-pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.35; }}
            }}
            .wm-bar-divider {{
                width: 1px; height: 32px; background: #334155;
            }}
            .wm-bar-title-block {{
                display: flex; flex-direction: column; gap: 2px;
            }}
            .wm-bar-title {{
                font-size: 22px; font-weight: 900; color: #f8fafc;
                letter-spacing: -0.01em; line-height: 1;
            }}
            .wm-bar-class {{
                font-size: 10px; color: #94a3b8; font-weight: 600;
                letter-spacing: 0.12em; text-transform: uppercase;
            }}
            .wm-bar-kpi {{
                display: flex; flex-direction: column; gap: 2px;
                background: #1e293b; border: 1px solid #334155;
                border-radius: 8px; padding: 6px 12px; min-width: 90px;
            }}
            .wm-bar-kpi-label {{
                font-size: 9px; color: #64748b; font-weight: 700;
                letter-spacing: 0.1em; text-transform: uppercase;
            }}
            .wm-bar-kpi-value {{
                font-size: 16px; font-weight: 800; color: #f1f5f9;
                font-variant-numeric: tabular-nums; line-height: 1.1;
            }}
            .wm-bar-kpi-unit {{
                font-size: 10px; color: #94a3b8;
                font-weight: 600; margin-left: 2px;
            }}
            .wm-bar-status {{
                margin-left: auto;
                display: inline-flex; align-items: center; gap: 7px;
                padding: 8px 16px; border-radius: 18px;
                background: {status_bg}; color: {status_fg};
                font-weight: 800; font-size: 11px;
                letter-spacing: 0.12em; text-transform: uppercase;
                border: 1.5px solid {status_border};
                box-shadow: 0 3px 8px rgba(0,0,0,0.18);
            }}
            .wm-bar-chips {{
                display: flex; gap: 6px; flex-wrap: wrap;
                margin: 0 0 12px 0;
            }}
            .wm-bar-chip {{
                background: white; border: 1px solid #dbe5f0;
                border-radius: 7px; padding: 4px 10px;
                font-size: 11px; color: #0f172a; font-weight: 700;
            }}
            .wm-bar-chip b {{
                color: #64748b; font-size: 9px; font-weight: 800;
                letter-spacing: 0.1em; margin-right: 6px;
                text-transform: uppercase;
            }}
            </style>
            <div class="wm-asset-bar">
                <span class="wm-bar-live">
                    <span class="wm-bar-live-dot"></span>LIVE
                </span>
                <div class="wm-bar-divider"></div>
                <div class="wm-bar-title-block">
                    <div class="wm-bar-title">{title_text}</div>
                    <div class="wm-bar-class">{class_label} · ISO 20816 / API 670</div>
                </div>
                <div class="wm-bar-kpi">
                    <span class="wm-bar-kpi-label">VELOCIDAD</span>
                    <span class="wm-bar-kpi-value">{speed_txt}<span class="wm-bar-kpi-unit">rpm</span></span>
                </div>
                <div class="wm-bar-kpi">
                    <span class="wm-bar-kpi-label">ÚLTIMA LECTURA</span>
                    <span class="wm-bar-kpi-value" style="color:{age_color};">hace {age_txt}</span>
                </div>
                <div class="wm-bar-kpi">
                    <span class="wm-bar-kpi-label">ALARMAS</span>
                    <span class="wm-bar-kpi-value" style="color:{alarms_color};">{alarms_txt}</span>
                </div>
                <span class="wm-bar-status">{status_icon} {status_label}</span>
            </div>
            <div class="wm-bar-chips">{chips_html}</div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_alarm_strip(rendered_rows: List[Dict[str, Any]]) -> None:
    """Banner rojo prominente si hay sensores en danger."""
    danger_rows = [r for r in rendered_rows if r["status"] == "Danger"]
    if not danger_rows:
        return
    items = []
    for r in danger_rows[:4]:
        try:
            v = f"{float(r['value']):.3f}"
        except Exception:
            v = "—"
        items.append(
            f'<span style="background:rgba(255,255,255,0.12);padding:5px 12px;border-radius:8px;'
            f'font-family:SF Mono,monospace;">'
            f'<b>{r["sensor_label"]}</b> · {r["variable"]} = {v} {r["unit"]}'
            f'</span>'
        )
    extra = ""
    if len(danger_rows) > 4:
        extra = f'<span style="opacity:0.8;">+{len(danger_rows)-4} más</span>'

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="wm-alarm-strip">
                <div class="wm-alarm-icon">⚠</div>
                <div style="display:flex;flex-direction:column;gap:6px;flex:1;min-width:240px;">
                    <div style="font-size:11px;letter-spacing:0.1em;text-transform:uppercase;opacity:0.85;">
                        Sensores en danger ({len(danger_rows)})
                    </div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;font-size:13px;">
                        {''.join(items)} {extra}
                    </div>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


# ============================================================
# KPIs
# ============================================================

def render_status_bar(
    latest: List[Dict[str, Any]],
    severity_summary: Dict[str, int],
) -> None:
    """
    Status bar fina (1 línea) con KPIs separados por divisores. Reemplaza
    las 4 cards gigantes que ocupaban demasiado espacio y se veían tipo
    'PowerPoint'. Inspirada en Emerson AMS Machine Works + breadcrumb
    pattern de aplicaciones SCADA modernas.
    """
    direct_rows = [r for r in latest if r.get("metric") == "Direct"]
    speed_row = next(
        (r for r in latest if (r.get("variable") or "").lower().startswith("velocidad")),
        None,
    )
    speed_val = speed_row.get("value") if speed_row else None
    speed_txt = f"{float(speed_val):.0f}" if speed_val is not None else "—"
    n_direct = len(direct_rows)

    if latest:
        min_age = min(_seconds_since(r["captured_at"]) for r in latest)
        age_txt = _format_age(
            min(latest, key=lambda r: _seconds_since(r["captured_at"]))["captured_at"]
        )
        age_color = _staleness_color(min_age)
    else:
        age_txt = "—"
        age_color = "#94a3b8"

    n_alarm = severity_summary.get("Alarma", 0)
    n_danger = severity_summary.get("Danger", 0)
    alarms_total = n_alarm + n_danger
    if alarms_total == 0:
        alarms_color = "#22c55e"
        alarms_text = "Sin alarmas"
    elif n_danger == 0:
        alarms_color = "#f59e0b"
        alarms_text = f"{n_alarm} alarma{'s' if n_alarm != 1 else ''}"
    else:
        alarms_color = "#ef4444"
        alarms_text = f"{n_danger} danger · {n_alarm} alarma{'s' if n_alarm != 1 else ''}"

    st.markdown(
        textwrap.dedent(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 28px;
                padding: 10px 16px;
                margin: 4px 0 16px 0;
                background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                font-size: 13px;
                color: #0f172a;
                flex-wrap: wrap;
            ">
                <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="color:#64748b;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;font-size:10px;">Velocidad</span>
                    <span style="font-weight:800;font-variant-numeric:tabular-nums;">{speed_txt}</span>
                    <span style="color:#94a3b8;font-size:11px;">rpm</span>
                </span>
                <span style="color:#cbd5e1;">·</span>
                <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="color:#64748b;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;font-size:10px;">Sensores</span>
                    <span style="font-weight:800;">{n_direct}</span>
                    <span style="color:#94a3b8;font-size:11px;">activos</span>
                </span>
                <span style="color:#cbd5e1;">·</span>
                <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="color:#64748b;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;font-size:10px;">Última lectura</span>
                    <span style="font-weight:800;color:{age_color};">hace {age_txt}</span>
                </span>
                <span style="color:#cbd5e1;">·</span>
                <span style="display:inline-flex;align-items:center;gap:6px;">
                    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{alarms_color};"></span>
                    <span style="font-weight:700;color:{alarms_color};">{alarms_text}</span>
                </span>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


# Compatibilidad con código que aún llame a render_kpi_strip
render_kpi_strip = render_status_bar


# ============================================================
# Tabla principal — Valores Actuales (con sparklines y location)
# ============================================================

def compute_rendered_rows(
    latest: List[Dict[str, Any]],
    sensor_lookup: Dict[str, Dict[str, Any]],
    instance_obj: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    direct_rows = [
        r for r in latest
        if r.get("metric") == "Direct"
        and not (r.get("variable") or "").lower().startswith("velocidad")
    ]
    summary = {"Normal": 0, "Alarma": 0, "Danger": 0, "Sin Norma": 0, "No Data": 0}
    rendered: List[Dict[str, Any]] = []

    for r in direct_rows:
        sensor_label = r.get("sensor_label") or "—"
        sensor_match = sensor_lookup.get(sensor_label)
        unit = r.get("unit") or ""
        sev = compute_severity(r.get("value"), sensor_match, unit, instance_obj)
        summary[sev["status"]] = summary.get(sev["status"], 0) + 1
        rendered.append({
            "sensor_label": sensor_label,
            "plane_label": (sensor_match or {}).get("plane_label", ""),
            "variable": r.get("variable"),
            "value": r.get("value"),
            "unit": unit,
            "age": _format_age(r.get("captured_at", "")),
            "quality": r.get("quality") or "good",
            "status": sev["status"], "fg": sev["fg"], "bg": sev["bg"],
            "alarm_used": sev["alarm"], "danger_used": sev["danger"],
            "threshold_source": sev["source"],
            "_sort_key": (
                {"Danger": 0, "Alarma": 1, "Sin Norma": 2, "Normal": 3, "No Data": 4}.get(sev["status"], 9),
                sensor_label,
            ),
        })
    rendered.sort(key=lambda r: r["_sort_key"])
    return rendered, summary


def render_channels_table(
    rendered_rows: List[Dict[str, Any]],
    spark_data: Dict[str, List[Dict[str, Any]]],
) -> None:
    if not rendered_rows:
        st.info("Sin lecturas Direct para mostrar.")
        return

    body_rows = []
    for r in rendered_rows:
        row_class = ""
        if r["status"] == "Alarma":
            row_class = "row-alarm"
        elif r["status"] == "Danger":
            row_class = "row-danger"
        try:
            value_str = f"{float(r['value']):,.4f}" if r["value"] is not None else "—"
        except Exception:
            value_str = "—"

        # Sparkline
        history = spark_data.get(r["sensor_label"], [])
        values = [h.get("value") for h in history if h.get("value") is not None]
        spark_color = (
            "#ef4444" if r["status"] == "Danger"
            else ("#f59e0b" if r["status"] == "Alarma" else "#3b82f6")
        )
        spark_html = sparkline_svg(values, color=spark_color) if values else "—"

        loc = r.get("plane_label") or "—"

        body_rows.append(
            f'<tr class="{row_class}">'
            f'<td>{status_pill_html(r["status"], r["fg"], r["bg"])}</td>'
            f'<td><b>{r["sensor_label"]}</b></td>'
            f'<td class="col-mono">{loc}</td>'
            f'<td>{r["variable"]}</td>'
            f'<td class="col-num">{value_str}</td>'
            f'<td class="col-mono">{r["unit"]}</td>'
            f'<td class="col-spark">{spark_html}</td>'
            f'<td class="col-mono">{r["age"]}</td>'
            f'</tr>'
        )

    table_html = textwrap.dedent(
        """
        <table class="wm-live-table">
            <thead>
                <tr>
                    <th>Status</th>
                    <th>Sensor</th>
                    <th>Ubicación</th>
                    <th>Variable</th>
                    <th style="text-align:right;">Valor</th>
                    <th>Unit</th>
                    <th>Trend (30 lecturas)</th>
                    <th>Edad</th>
                </tr>
            </thead>
            <tbody>__ROWS__</tbody>
        </table>
        """
    ).strip().replace("__ROWS__", "\n".join(body_rows))

    st.markdown(table_html, unsafe_allow_html=True)


# ============================================================
# Vectores 1X / 2X — phasor cards
# ============================================================

def render_vectors_phasors(latest: List[Dict[str, Any]]) -> None:
    by_sensor: Dict[str, Dict[str, Any]] = {}
    for r in latest:
        s = r.get("sensor_label")
        m = r.get("metric")
        if not s or m not in ("1X_Ampl", "1X_Phase", "2X_Ampl", "2X_Phase"):
            continue
        slot = by_sensor.setdefault(s, {"sensor": s})
        slot[m] = r.get("value")
        if "Ampl" in (m or ""):
            slot["unit_ampl"] = r.get("unit") or slot.get("unit_ampl", "")

    if not by_sensor:
        st.info(
            "Este activo no envía vectores 1X/2X. "
            "Solo proximity probes con módulos vibration monitor generan datos vectoriales."
        )
        return

    st.caption(
        "🎯 **Vectores síncronos 1X / 2X** · Amplitud y fase por sensor — "
        "el dato fundamental para Polar Plot, Bode y diagnóstico de balanceo / desalineamiento."
    )

    # Determinamos max_amp por unidad para escala consistente del phasor
    all_amps = [
        slot.get("1X_Ampl") or 0
        for slot in by_sensor.values()
        if slot.get("1X_Ampl") is not None
    ]
    max_1x = max(all_amps) * 1.1 if all_amps else 1.0
    if max_1x <= 0:
        max_1x = 1.0

    # Rendering: 4 columns wide
    sensors_sorted = sorted(by_sensor.keys())
    n_cols = 4
    for i in range(0, len(sensors_sorted), n_cols):
        cols = st.columns(n_cols, gap="medium")
        for j, col in enumerate(cols):
            if i + j >= len(sensors_sorted):
                continue
            sensor_lbl = sensors_sorted[i + j]
            slot = by_sensor[sensor_lbl]
            with col:
                amp_1x = slot.get("1X_Ampl")
                ph_1x = slot.get("1X_Phase")
                amp_2x = slot.get("2X_Ampl")
                ph_2x = slot.get("2X_Phase")
                unit = slot.get("unit_ampl") or ""

                # Filtrar fases inválidas (~1e-41 → ampl ~0)
                def fmt_pair(amp, ph):
                    if amp is None:
                        return "—"
                    a_s = f"{amp:.3f}"
                    if amp < 1e-4 or ph is None or abs(ph) < 1e-30:
                        return f"{a_s} ∠ —"
                    return f"{a_s} ∠ {ph:.0f}°"

                phasor_html = phasor_svg(amp_1x, ph_1x, max_amp=max_1x, color="#3b82f6", size=120)

                st.markdown(
                    textwrap.dedent(
                        f"""
                        <div class="wm-phasor-card">
                            <div class="wm-phasor-label">{sensor_lbl}</div>
                            {phasor_html}
                            <div class="wm-phasor-values">
                                <div><b>1X</b>: {fmt_pair(amp_1x, ph_1x)} {unit}</div>
                                <div style="opacity:0.75;"><b>2X</b>: {fmt_pair(amp_2x, ph_2x)} {unit}</div>
                            </div>
                        </div>
                        """
                    ).strip(),
                    unsafe_allow_html=True,
                )


# ============================================================
# Diagnostic
# ============================================================

def render_diagnostic_table(latest: List[Dict[str, Any]]) -> None:
    diag_rows = [r for r in latest if r.get("metric") in ("Gap", "BiasVoltage")]
    if not diag_rows:
        st.info("Sin métricas de diagnostic (Gap / BiasVoltage) en esta instancia.")
        return

    st.caption(
        "🔧 **Health del transducer.** "
        "Gap = posición DC del eje en el cojinete (rango típico −7 a −10 V DC). "
        "BiasVoltage = bias del acelerómetro (rango típico 10–12 V DC). "
        "Valores fuera de rango indican sensor degradado, no problema mecánico."
    )

    body_rows = []
    for r in sorted(diag_rows, key=lambda x: (x.get("sensor_label") or "", x.get("metric") or "")):
        sensor = r.get("sensor_label") or "—"
        var = r.get("variable") or ""
        metric = r.get("metric") or ""
        val = r.get("value")
        unit = r.get("unit") or ""

        status = "OK"
        st_fg, st_bg = "#166534", "#dcfce7"
        try:
            v = float(val) if val is not None else None
            if v is not None:
                if metric == "Gap" and (v < -11 or v > -6):
                    status = "Fuera de rango"
                    st_fg, st_bg = "#92400e", "#fef3c7"
                elif metric == "BiasVoltage" and (v < 8 or v > 14):
                    status = "Fuera de rango"
                    st_fg, st_bg = "#92400e", "#fef3c7"
        except Exception:
            pass

        try:
            val_str = f"{float(val):,.4f}" if val is not None else "—"
        except Exception:
            val_str = "—"

        body_rows.append(
            f'<tr>'
            f'<td>{status_pill_html(status, st_fg, st_bg)}</td>'
            f'<td><b>{sensor}</b></td>'
            f'<td>{var}</td>'
            f'<td><span class="col-mono">{metric}</span></td>'
            f'<td class="col-num">{val_str}</td>'
            f'<td class="col-mono">{unit}</td>'
            f'<td class="col-mono">{_format_age(r.get("captured_at",""))}</td>'
            f'</tr>'
        )

    table_html = textwrap.dedent(
        """
        <table class="wm-live-table">
            <thead><tr>
                <th>Health</th><th>Sensor</th><th>Variable</th><th>Métrica</th>
                <th style="text-align:right;">Valor</th><th>Unit</th><th>Edad</th>
            </tr></thead>
            <tbody>__ROWS__</tbody>
        </table>
        """
    ).strip().replace("__ROWS__", "\n".join(body_rows))
    st.markdown(table_html, unsafe_allow_html=True)


# ============================================================
# Tendencia con bandas de severidad
# ============================================================

def render_history_chart(
    instance_id: str,
    latest: List[Dict[str, Any]],
    sensor_lookup: Dict[str, Dict[str, Any]],
    instance_obj: Any = None,
) -> None:
    """
    Trend chart multi-sensor (Ciclo 23.56) — el usuario puede overlayar
    1..N sensores sobre el mismo gráfico con colores por sensor.
    Refactor del antiguo single-sensor select. Compite directamente con
    el "Trend Plot" de Bently System1 (que también permite multi).

    Severity bands se muestran SOLO si todos los sensores seleccionados
    comparten el mismo alarm/danger (e.g. todos g pk con misma norma).
    En selección mixta de unidades, bands se omiten.
    """
    direct_rows = [
        r for r in latest
        if r.get("metric") == "Direct"
        and not (r.get("variable") or "").lower().startswith("velocidad")
    ]
    if not direct_rows:
        st.info("Sin variables Direct para graficar todavía.")
        return

    options = sorted([
        (r.get("sensor_label") or "—", r.get("variable")) for r in direct_rows
    ])
    labels = [f"{s} — {v}" for (s, v) in options]

    # Multi-sensor selector (Ciclo 23.56)
    col_var, col_range = st.columns([3, 1])
    with col_var:
        chosen_labels = st.multiselect(
            "📈 Sensores a graficar (1 o varios — overlay)",
            labels,
            default=labels[:1],  # por defecto el primero
            key="live_history_vars_multi",
            help="Seleccioná uno o varios sensores. Útil para comparar 1YA vs 2YA, "
                 "o ver patrones cross-sensor de un eventos de carga.",
        )
    with col_range:
        range_choice = st.selectbox(
            "Rango", ["Última hora", "6 horas", "24 horas", "7 días", "Todo"],
            index=1, key="live_history_range",
            help="Cuánto historial traer del registro append-only en Supabase",
        )

    if not chosen_labels:
        st.info("Seleccioná al menos 1 sensor para ver tendencia.")
        return

    # Caps de lecturas por sensor (cada poll del collector son ~10 s, 360/h)
    range_to_limit = {
        "Última hora": 400,
        "6 horas":     2200,
        "24 horas":    9000,
        "7 días":      62000,
        "Todo":        200000,
    }
    limit = range_to_limit.get(range_choice, 2200)

    # Fetch data por sensor — guardamos por label
    sensor_data: List[Dict[str, Any]] = []
    for chosen in chosen_labels:
        idx = labels.index(chosen)
        sensor_lbl, var_name = options[idx]
        rows = history_for_metric(
            instance_id, var_name, "Direct", limit=limit,
        )
        if not rows:
            continue
        df_one = pd.DataFrame(rows)
        df_one["captured_at"] = pd.to_datetime(
            df_one["captured_at"], utc=True
        ).dt.tz_convert(_local_tz())
        df_one = df_one.sort_values(by="captured_at").reset_index(drop=True)
        # Compute severity para este sensor
        sensor_match = sensor_lookup.get(sensor_lbl)
        sample_unit = direct_rows[idx].get("unit", "")
        sev = compute_severity(
            df_one["value"].iloc[-1] if len(df_one) else None,
            sensor_match, sample_unit, instance_obj=instance_obj,
        )
        sensor_data.append({
            "label": sensor_lbl,
            "var": var_name,
            "display": chosen,
            "unit": sample_unit,
            "df": df_one,
            "alarm": sev.get("alarm", 0),
            "danger": sev.get("danger", 0),
        })

    if not sensor_data:
        st.info("Sin histórico aún en ningún sensor seleccionado.")
        return

    # KPIs combinados (sobre TODOS los sensores)
    all_values = pd.concat([sd["df"]["value"] for sd in sensor_data])
    total_readings = sum(len(sd["df"]) for sd in sensor_data)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Mín (global)", f"{all_values.min():.3f}")
    with c2: st.metric("Máx (global)", f"{all_values.max():.3f}")
    with c3: st.metric("Promedio (global)", f"{all_values.mean():.3f}")
    with c4: st.metric("Σ Lecturas", f"{total_readings:,}")

    # Decidir si mostramos bandas: solo si todos comparten unit + alarm + danger
    units = {sd["unit"] for sd in sensor_data}
    alarms = {sd["alarm"] for sd in sensor_data}
    dangers = {sd["danger"] for sd in sensor_data}
    show_bands = (
        len(sensor_data) >= 1
        and len(units) == 1
        and len(alarms) == 1
        and len(dangers) == 1
        and sensor_data[0]["alarm"] > 0
        and sensor_data[0]["danger"] > sensor_data[0]["alarm"]
    )

    # Plotly chart multi-trace
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Bandas (solo en caso compatible)
        if show_bands:
            alarm = sensor_data[0]["alarm"]
            danger = sensor_data[0]["danger"]
            y_max = max(all_values.max(), danger * 1.1)
            fig.add_hrect(y0=danger, y1=y_max * 1.05,
                          fillcolor="#fee2e2", opacity=0.45, line_width=0,
                          annotation_text="Danger", annotation_position="top right",
                          annotation=dict(font=dict(color="#991b1b", size=10)))
            fig.add_hrect(y0=alarm, y1=danger,
                          fillcolor="#fef3c7", opacity=0.45, line_width=0,
                          annotation_text="Alarma", annotation_position="top right",
                          annotation=dict(font=dict(color="#92400e", size=10)))
            fig.add_hrect(y0=0, y1=alarm,
                          fillcolor="#dcfce7", opacity=0.35, line_width=0,
                          annotation_text="Normal", annotation_position="bottom right",
                          annotation=dict(font=dict(color="#166534", size=10)))

        # Paleta corporativa (navy + accent variants — coherente con el design system)
        palette = [
            "#1e40af", "#15803d", "#b45309", "#dc2626",
            "#7c3aed", "#0891b2", "#be185d", "#475569",
        ]
        for i, sd in enumerate(sensor_data):
            color = palette[i % len(palette)]
            fig.add_trace(go.Scatter(
                x=sd["df"]["captured_at"],
                y=sd["df"]["value"],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=4),
                name=sd["display"],
                hovertemplate=(
                    f"<b>{sd['label']} {sd['var']}</b><br>"
                    "%{x|%Y-%m-%d %H:%M:%S}<br>"
                    f"%{{y:.3f}} {sd['unit']}<extra></extra>"
                ),
            ))

        # Y-axis title: si todos comparten unit, mostrar; sino "valor"
        y_title = list(units)[0] if len(units) == 1 else "valor"

        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=420,
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=""),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=y_title),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=11),
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Caption si bands omitidas
        if not show_bands and len(sensor_data) > 1:
            mixed_units = ", ".join(sorted(units))
            st.caption(
                f"📐 Bandas de severidad omitidas — los sensores seleccionados "
                f"tienen unidades mixtas ({mixed_units}) o thresholds distintos. "
                f"Seleccioná sensores del mismo tipo (todos `g pk`, todos `in/s pk`, "
                f"todos `mil pp`) para ver las bandas."
            )
    except Exception as e:
        # Fallback a st.line_chart si plotly falla
        st.caption(f"(plotly no disponible — fallback: {e})")
        combined = pd.DataFrame()
        for sd in sensor_data:
            df_one = sd["df"].set_index("captured_at")[["value"]].copy()
            df_one.columns = [sd["display"]]
            combined = combined.join(df_one, how="outer") if not combined.empty else df_one
        st.line_chart(combined, height=380)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Header limpio sin subtítulo redundante (Ciclo 23.15).
    # subtitle="" porque page_header requiere el kwarg pero no queremos texto.
    page_header(title="Live Monitoring", subtitle="")

    from core.instance_state import list_instances, get_instance
    instances = list_instances()
    if not instances:
        st.info("No hay activos registrados aún. Andá a Machinery Library para crear uno.")
        return

    # Build mapping instance_id → display label "TAG · Cliente"
    inst_meta = {i.get("instance_id"): i for i in instances if i.get("instance_id")}
    options = sorted(inst_meta.keys())
    default_idx = options.index("tes1") if "tes1" in options else 0

    def _fmt_option(iid: str) -> str:
        meta = inst_meta.get(iid, {})
        tag = meta.get("tag", "") or iid.upper()
        client = meta.get("client", "")
        if client:
            return f"{tag}  ·  {client}"
        return tag

    # Top bar — selector de activo + auto-refresh con look industrial
    # estilo SCADA / control room (Ciclo 23.17). Sin etiqueta arriba del
    # selectbox (CSS oculta el label nativo de Streamlit), reemplazada
    # por header propio con icono + texto profesional.
    st.markdown(
        textwrap.dedent("""
        <style>
        /* Header industrial sobre el selectbox */
        .wm-picker-header {
            display: flex; align-items: center; gap: 10px;
            margin: 8px 0 6px 0;
        }
        .wm-picker-header-icon {
            width: 32px; height: 32px; border-radius: 9px;
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            display: flex; align-items: center; justify-content: center;
            color: white; font-size: 16px; font-weight: 900;
            box-shadow: 0 4px 10px rgba(30,64,175,0.25);
        }
        .wm-picker-header-text { display: flex; flex-direction: column; }
        .wm-picker-header-title {
            font-size: 13px; font-weight: 800; color: #0f172a;
            letter-spacing: -0.01em; line-height: 1.1;
        }
        .wm-picker-header-sub {
            font-size: 10px; color: #64748b; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.08em;
        }
        /* Esconder label nativo del selectbox (lo reemplazamos por wm-picker-header) */
        div[data-testid="stSelectbox"] label[data-testid="stWidgetLabel"] {
            display: none !important;
        }
        /* Selectbox con look industrial */
        div[data-testid="stSelectbox"] > div > div {
            background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
            border: 1.5px solid #c7d9eb;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(15,23,42,0.05);
            padding: 4px 8px;
            min-height: 46px;
        }
        div[data-testid="stSelectbox"] > div > div:hover {
            border-color: #3b82f6;
            box-shadow: 0 4px 14px rgba(59,130,246,0.15);
        }
        </style>
        """).strip(),
        unsafe_allow_html=True,
    )

    # Header industrial sin auto-refresh (Ciclo 23.59 — el operador refresca
    # con el botón cuando lo necesita; el meta-refresh era ruido en el flujo
    # de análisis profundo).
    st.markdown(
        textwrap.dedent("""
        <div class="wm-picker-header">
            <div class="wm-picker-header-icon">🛰</div>
            <div class="wm-picker-header-text">
                <div class="wm-picker-header-title">Activo monitoreado</div>
                <div class="wm-picker-header-sub">Real-time · ISO 20816 / API 670</div>
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )
    instance_id = st.selectbox(
        "Activo",   # label oculto por CSS
        options,
        index=default_idx,
        key="live_asset_v3",
        format_func=_fmt_option,
        label_visibility="collapsed",
    )

    if not instance_id:
        return

    instance_obj = get_instance(instance_id)
    sensor_lookup = _build_sensor_lookup(instance_obj)
    latest = latest_for_instance(instance_id)

    # Anti-flicker cache (Ciclo 23.60) — Supabase REST a veces devuelve []
    # por timeouts transitorios o reconexión. Si tenemos un snapshot reciente
    # en session_state lo reusamos para no mostrar el warning catastrófico
    # cada vez que parpadea la red. TTL = 5 min; pasado eso sí asumimos
    # que el activo no tiene datos.
    cache_key = f"wm_latest_cache_{instance_id}"
    now_ts = datetime.now().timestamp()
    if latest:
        st.session_state[cache_key] = {"data": latest, "ts": now_ts}
        using_cache = False
        cache_age = 0.0
    else:
        cached = st.session_state.get(cache_key)
        if cached and (now_ts - cached.get("ts", 0)) < 300:  # 5 min TTL
            latest = cached["data"]
            using_cache = True
            cache_age = now_ts - cached["ts"]
        else:
            using_cache = False
            cache_age = 0.0

    if not latest:
        # Sin datos y sin cache reciente → activo no configurado o caído
        render_asset_header(instance_obj, instance_id)
        st.warning(
            "**Sin datos en tiempo real para este activo.** Verificá:\n"
            "1. Tabla `live_readings` creada en Supabase.\n"
            "2. wm-collector corriendo en el PC de planta.\n"
            "3. El collector usa el mismo `instance_id`."
        )
        return

    if using_cache:
        st.caption(
            f"⏳ Reintentando conexión con Supabase — mostrando última lectura "
            f"válida (hace {int(cache_age)}s). Refrescá si persiste."
        )

    rendered_rows, severity_summary = compute_rendered_rows(latest, sensor_lookup, instance_obj)

    # Spark data se trae ANTES del diagrama (Ciclo 23.23) para alimentar
    # las sparklines mini al lado de cada sensor dot. La misma data se
    # reusa abajo en tab_curr para la tabla de canales — sin doble fetch.
    spark_data = recent_history_all_direct(instance_id, n_per_sensor=30)

    # Asset banner compacto — barra oscura de 1 línea con título + KPIs
    # inline + status pill. Reemplaza la card grande que dominaba la pantalla.
    render_asset_header(instance_obj, instance_id, latest, severity_summary)

    # Alarm strip prominente arriba si hay danger
    render_alarm_strip(rendered_rows)

    # Hero — Machine Map.
    # Prioridad (Ciclo 23.13):
    #   1) Asset library 2D vectorial (System1 / Emerson AMS-style) si la
    #      instancia tiene driver_icon_key + driven_icon_key configurados.
    #      Sensor dots se posicionan en los anchors físicos del icono
    #      (DE / NDE / TRF / CRF) por convención Bently / API 670.
    #   2) PNG 3D legacy con overlay x_pct/y_pct si no hay icon_keys.
    #   3) PNG plano sin overlay si tampoco hay coordenadas.
    has_map = render_sensor_map_library(
        instance_obj, latest, sensor_lookup, spark_data=spark_data,
    )
    if not has_map:
        has_map = render_sensor_map_hero(instance_obj, instance_id, latest, sensor_lookup)

    # Inline severity legend (Ciclo 23.49) — debajo del diagrama, hace
    # el SVG self-documenting. Usuario nuevo entiende los colores sin
    # tutorial. Render solo si tenemos diagrama vectorial.
    # Ciclo 23.56: margen negativo arriba para tightear gap dibujo-legend.
    if has_map:
        st.markdown(
            textwrap.dedent("""
            <style>
            .wm-legend-row {
                display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
                padding: 8px 14px; margin: -8px 0 12px 0;
                font-size: 11px; color: #475569;
                background: rgba(255,255,255,0.6);
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                font-family: -apple-system, "SF Pro Text", system-ui, sans-serif;
            }
            .wm-legend-label {
                font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
                color: #64748b; font-size: 10px;
            }
            .wm-legend-item {
                display: inline-flex; align-items: center; gap: 5px;
                font-weight: 600;
            }
            .wm-legend-dot {
                width: 10px; height: 10px; border-radius: 50%;
                border: 1.5px solid white;
                box-shadow: 0 0 0 1px rgba(15,23,42,0.10);
            }
            .wm-legend-spark {
                width: 28px; height: 10px;
                border-radius: 3px;
                background: linear-gradient(90deg, #86efac 0%, #86efac 70%, #fcd34d 70%, #fcd34d 88%, #fca5a5 88%, #fca5a5 100%);
                border: 1px solid rgba(15,23,42,0.10);
            }
            .wm-legend-stale {
                width: 14px; height: 14px; border-radius: 50%;
                background: #94a3b8;
                filter: grayscale(1);
                opacity: 0.6;
            }
            </style>
            <div class="wm-legend-row">
                <span class="wm-legend-label">Leyenda</span>
                <span class="wm-legend-item"><span class="wm-legend-dot" style="background:#15803d;"></span>Normal</span>
                <span class="wm-legend-item"><span class="wm-legend-dot" style="background:#b45309;"></span>Alarma</span>
                <span class="wm-legend-item"><span class="wm-legend-dot" style="background:#dc2626;"></span>Danger</span>
                <span class="wm-legend-item"><span class="wm-legend-stale"></span>Stale (sin lectura > 60s)</span>
                <span class="wm-legend-item"><span class="wm-legend-spark"></span>Threshold bar (verde→rojo escala alarm/danger)</span>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

    # Sensor selection (Ciclo 23.33) — selectbox discreto debajo del
    # diagrama. Razón técnica para no usar click directo en SVG:
    # Streamlit + browser full-reload pierde session_state → auth falla
    # → redirect a login. El selectbox usa st.rerun() interno que
    # mantiene el websocket de la sesión vivo.
    direct_labels = sorted({
        r.get("sensor_label") for r in latest
        if r.get("metric") == "Direct" and r.get("sensor_label")
    })
    selected_sensor = st.query_params.get("sensor")

    if direct_labels:
        st.markdown(
            textwrap.dedent("""
            <style>
            /* Selectbox más discreto para drill-down */
            div[data-testid="stSelectbox"]:has(input[aria-label="drilldown_select"]) > div > div {
                background: #f8fafc;
                border: 1.5px solid #94a3b8;
                border-radius: 999px;
                padding: 2px 10px;
                font-size: 13px;
            }
            </style>
            """).strip(),
            unsafe_allow_html=True,
        )
        sel_options = ["🔍 Análisis detallado — picker un sensor…"] + direct_labels
        try:
            sel_idx = sel_options.index(selected_sensor) if selected_sensor in direct_labels else 0
        except ValueError:
            sel_idx = 0
        new_sel = st.selectbox(
            "drilldown_select",
            options=sel_options,
            index=sel_idx,
            key="live_sensor_drilldown",
            format_func=lambda x: x.replace("_", "") if x in direct_labels else x,
            label_visibility="collapsed",
        )
        new_value = new_sel if new_sel != "🔍 Análisis detallado — picker un sensor…" else None
        if new_value != selected_sensor:
            if new_value:
                st.query_params["sensor"] = new_value
            else:
                st.query_params.clear()
            st.rerun()
        selected_sensor = new_value

    # Zoom panel — si hay sensor seleccionado, render del card focalizado
    # con trend chart + selector de tipo de gráfico.
    if selected_sensor:
        render_sensor_zoom_panel(
            selected_sensor=selected_sensor,
            latest=latest,
            sensor_lookup=sensor_lookup,
            instance_obj=instance_obj,
            spark_data=spark_data,
        )
    elif direct_labels:
        # Empty state polish (Ciclo 23.49) — sin sensor seleccionado
        # mostramos un hint sutil con icono + arrow apuntando al dropdown.
        # Mejor que vacío silencioso; enseña la affordance al usuario nuevo.
        st.markdown(
            textwrap.dedent("""
            <style>
            .wm-zoom-hint {
                display: flex; align-items: center; gap: 12px;
                margin: 14px 0 18px 0; padding: 14px 18px;
                background: linear-gradient(135deg, #f0f7ff 0%, #e6f0fb 100%);
                border: 1px dashed #c7d9eb;
                border-radius: 12px;
                color: #475569;
                font-size: 13px;
                font-style: italic;
            }
            .wm-zoom-hint-icon {
                font-size: 22px;
                opacity: 0.6;
                animation: wm-zoom-bounce 2s ease-in-out infinite;
            }
            @keyframes wm-zoom-bounce {
                0%, 100% { transform: translateY(0); }
                50%      { transform: translateY(-3px); }
            }
            .wm-zoom-hint b { color: #1e40af; font-style: normal; font-weight: 700; }
            </style>
            <div class="wm-zoom-hint">
                <span class="wm-zoom-hint-icon">⤴</span>
                <span>
                    <b>Análisis detallado:</b> seleccioná un sensor en el dropdown de arriba
                    para ver tendencia, vectores 1X/2X, gap voltage y más.
                </span>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )
    if not has_map and instance_obj is not None and instance_obj.schematic_png:
        # Tiene schematic pero sin posiciones — render sin overlay
        try:
            from core.instance_state import get_instance_document_bytes
            png_bytes = get_instance_document_bytes(instance_id, instance_obj.schematic_png)
            if png_bytes:
                st.image(png_bytes, caption="Esquemático del activo", use_container_width=True)
                st.caption(
                    "💡 Configurá las posiciones (x_pct, y_pct) de cada sensor en el "
                    "Sensor Map del activo para activar el Live Sensor Map con dots de severidad."
                )
        except Exception:
            pass

    # Ciclo 23.58 — Diagnostic eliminado (redundante con gap/bias voltage
    # ya presente en los gráficos del zoom panel). Single source of truth:
    # todo el análisis per-sensor pasa por el panel de zoom.

    st.markdown("---")
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🔄 Refrescar ahora", key="live_refresh_v3", use_container_width=True):
            st.rerun()
    with c2:
        st.caption(
            f"📅 Sync local: {datetime.now().strftime('%H:%M:%S')} · "
            "Engine: Watermelon System · ISO 20816-3 / API 670"
        )


if __name__ == "__main__":
    main()
