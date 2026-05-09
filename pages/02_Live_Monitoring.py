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
  * Auto-refresh 10s opcional.

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
    Si no, aplica heurística por convención industrial Bently / API 670:
      - Label empieza con "1" → bearing #1 = DE del driver (TRF en aero)
      - Label empieza con "2" → bearing #2 = NDE del driver (CRF en aero)
      - Label empieza con "3" → bearing #3 = DE del driven
      - Label empieza con "4" → bearing #4 = NDE del driven
    Fallback a plane_label matching si label no empieza con dígito.

    Devuelve (None, None) si no se pudo mapear → sensor se omite del SVG.
    """
    # 1. Override explícito (futuro Commit 4 — wizard editor)
    if sensor_match:
        s_side = sensor_match.get("icon_side")
        s_anchor = sensor_match.get("icon_anchor")
        if s_side and s_anchor:
            return s_side, s_anchor

    drv_key = (getattr(instance_obj, "driver_icon_key", "") or "").lower()
    is_aero = "aero" in drv_key
    label_l = (sensor_label or "").strip().lower()
    plane_l = ((sensor_match or {}).get("plane_label") or "").lower()

    # 2. Convención Bently — primer carácter del label es el bearing #
    if label_l and label_l[0].isdigit():
        bearing_num = int(label_l[0])
        if bearing_num == 1:
            return "driver", ("TRF" if is_aero else "DE")
        if bearing_num == 2:
            return "driver", ("CRF" if is_aero else "NDE")
        if bearing_num == 3:
            return "driven", "DE"
        if bearing_num == 4:
            return "driven", "NDE"
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
) -> List[Dict[str, Any]]:
    """
    Construye la lista sensors_with_status para compose_train() a partir
    de las lecturas live y los sensores configurados de la instancia.
    Solo incluye sensores que se pudieron mapear a (side, anchor).
    """
    out: List[Dict[str, Any]] = []
    seen_anchors: set = set()  # evitar 2 sensores apilados en el mismo dot

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
            val_str = f"{float(r.get('value')):.2f}"
        except Exception:
            val_str = "—"
        title = (
            f"{lbl} | {val_str} {unit} | {sev['status']} | "
            f"alarm={sev['alarm']:.2f} / danger={sev['danger']:.2f} ({sev['source']})"
        )
        out.append({
            "label": lbl,
            "side": side,
            "anchor": anchor,
            "status": sev["status"],
            "value": val_str,
            "unit": unit,
            "title": title,
        })
        seen_anchors.add((side, anchor))
    return out


def render_sensor_map_library(
    instance_obj: Any,
    latest: List[Dict[str, Any]],
    sensor_lookup: Dict[str, Dict[str, Any]],
) -> bool:
    """
    Renderiza el tren acoplado vía core.asset_library.composer.

    Devuelve True si pudo (la instancia tiene driver_icon_key + driven_icon_key),
    False si no hay icon_keys configuradas → caller debe usar fallback PNG.
    """
    drv_key = getattr(instance_obj, "driver_icon_key", "") or ""
    drvn_key = getattr(instance_obj, "driven_icon_key", "") or ""
    if not drv_key or not drvn_key:
        return False

    try:
        from core.asset_library.composer import compose_train
    except ImportError:
        return False

    sensors_with_status = _build_library_sensors(latest, sensor_lookup, instance_obj)

    drv_label = (
        f"{instance_obj.driver_manufacturer} {instance_obj.driver_model}".strip()
        or instance_obj.driver_model
        or "Driver"
    )
    drvn_label = (
        f"{instance_obj.driven_manufacturer} {instance_obj.driven_model}".strip()
        or instance_obj.driven_model
        or "Driven"
    )

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

    # Render limpio sin frame ni legend redundante (Ciclo 23.14):
    # - Sin recuadro blanco con borde alrededor (estaba feo).
    # - Sin legend con normal/alarma/danger debajo (los colores se entienden
    #   por contexto + tooltip por sensor; redundante con la tabla de abajo).
    # - Sin contador "N sensores mapeados" (información ya visible en otras
    #   secciones de la página).
    st.markdown(svg, unsafe_allow_html=True)
    return True


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
# Header & Alarm strip
# ============================================================

def render_asset_header(instance_obj, instance_id: str) -> None:
    title_text = instance_id
    if instance_obj is not None and instance_obj.tag and instance_obj.tag != instance_id:
        title_text = f"{instance_obj.tag} · {instance_id}"

    sub_parts: List[str] = []
    if instance_obj is not None:
        if instance_obj.driver_model:
            sub_parts.append(instance_obj.driver_model)
        if instance_obj.driven_model:
            sub_parts.append(instance_obj.driven_model)
        if instance_obj.client:
            sub_parts.append(f"📍 {instance_obj.client}")
    sub_html = " · ".join(sub_parts) if sub_parts else ""

    # Asset class chip — transparencia de qué thresholds OEM se aplican
    asset_class = detect_asset_class(instance_obj)
    class_label = CLASS_LABELS.get(asset_class, "—")
    class_color = {
        "aero_turbine":       ("#dbeafe", "#1e40af"),  # azul
        "industrial_turbine": ("#dcfce7", "#166534"),  # verde
        "recip_compressor":   ("#fef3c7", "#92400e"),  # ámbar
        "rotating_general":   ("#f1f5f9", "#475569"),  # neutro
    }.get(asset_class, ("#f1f5f9", "#475569"))

    st.markdown(
        textwrap.dedent(
            f"""
            <div style="display:flex;align-items:center;gap:14px;margin-top:4px;flex-wrap:wrap;">
                <div class="wm-asset-title">{title_text}</div>
                <span class="wm-live-badge"><span class="wm-live-dot"></span>LIVE</span>
                <span style="
                    display:inline-flex;align-items:center;gap:5px;
                    padding:4px 11px;border-radius:999px;
                    background:{class_color[0]};color:{class_color[1]};
                    font-weight:800;font-size:10px;letter-spacing:0.06em;text-transform:uppercase;
                " title="Thresholds aplicados según esta clase de activo">
                    🛡️ {class_label}
                </span>
            </div>
            <div class="wm-asset-sub" style="margin-bottom:10px;">{sub_html}</div>
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

    # Variable + range selector (Ciclo 23.9: el usuario pidió ver más historial)
    col_var, col_range = st.columns([3, 1])
    with col_var:
        chosen = st.selectbox(
            "Variable a graficar", labels, key="live_history_var_v3",
        )
    with col_range:
        range_choice = st.selectbox(
            "Rango", ["Última hora", "6 horas", "24 horas", "7 días", "Todo"],
            index=1, key="live_history_range",
            help="Cuanta historia traer del registro append-only en Supabase",
        )
    idx = labels.index(chosen)
    sensor_lbl, var_name = options[idx]

    # Cada poll del collector son ~10 s. Fórmula: 360 lecturas/h
    # Caps generosos pero acotados para no crashear con asset que tenga 1M+ lecturas.
    range_to_limit = {
        "Última hora": 400,     # 1h × 360 lecturas + buffer
        "6 horas":     2200,    # 6h × 360 + buffer
        "24 horas":    9000,    # 24h × 360
        "7 días":      62000,   # 7d × 360 × 24 ≈ 60.5k + buffer
        "Todo":        200000,  # cap alto para no crashear; en activos con
                                # +1M lecturas hay que paginar (TODO ciclo 24)
    }
    rows = history_for_metric(
        instance_id, var_name, "Direct",
        limit=range_to_limit.get(range_choice, 2200),
    )
    if not rows:
        st.info("Sin histórico aún. Esperá unos minutos para que el collector acumule.")
        return

    df = pd.DataFrame(rows)
    # CRITICAL: convertir UTC del backend a hora local del cliente.
    # Sin esto el chart muestra timestamps "futuristas" (UTC interpretado como local).
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True).dt.tz_convert(_local_tz())
    df = df.sort_values(by="captured_at").reset_index(drop=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Mín", f"{df['value'].min():.3f}")
    with c2: st.metric("Máx", f"{df['value'].max():.3f}")
    with c3: st.metric("Promedio", f"{df['value'].mean():.3f}")
    with c4: st.metric("Σ Lecturas", f"{len(df):,}")

    # Severity bands con asset class awareness (no más fallback ISO genérico)
    sensor_match = sensor_lookup.get(sensor_lbl)
    sample_unit = direct_rows[idx].get("unit", "") if idx < len(direct_rows) else ""
    sev_sample = compute_severity(
        df["value"].iloc[-1] if len(df) else None,
        sensor_match, sample_unit, instance_obj=instance_obj,
    )
    alarm = sev_sample["alarm"]
    danger = sev_sample["danger"]
    threshold_source = sev_sample["source"]

    # Plotly chart con bandas
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        # Bandas
        if danger > 0:
            fig.add_hrect(y0=danger, y1=max(df["value"].max(), danger) * 1.05,
                          fillcolor="#fee2e2", opacity=0.5, line_width=0,
                          annotation_text="Danger", annotation_position="top right",
                          annotation=dict(font=dict(color="#991b1b", size=10)))
        if alarm > 0 and danger > alarm:
            fig.add_hrect(y0=alarm, y1=danger,
                          fillcolor="#fef3c7", opacity=0.5, line_width=0,
                          annotation_text="Alarma", annotation_position="top right",
                          annotation=dict(font=dict(color="#92400e", size=10)))
        if alarm > 0:
            fig.add_hrect(y0=0, y1=alarm,
                          fillcolor="#dcfce7", opacity=0.4, line_width=0,
                          annotation_text="Normal", annotation_position="bottom right",
                          annotation=dict(font=dict(color="#166534", size=10)))
        # Línea principal
        fig.add_trace(go.Scatter(
            x=df["captured_at"], y=df["value"],
            mode="lines+markers", line=dict(color="#1e40af", width=2),
            marker=dict(size=4),
            name=f"{sensor_lbl} {var_name}",
        ))
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=360,
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=sample_unit),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # fallback a st.line_chart si plotly falla
        df_chart = df.set_index("captured_at")[["value"]]
        df_chart.columns = [f"{sensor_lbl} — {var_name}"]
        st.line_chart(df_chart, height=320)


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

    # Top bar — selector de activo + auto-refresh, en card industrial
    st.markdown(
        textwrap.dedent("""
        <style>
        .wm-asset-picker-row { display: flex; align-items: center; gap: 14px;
            padding: 12px 16px; margin: 8px 0 18px 0;
            background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
            border: 1px solid #c7d9eb; border-radius: 14px;
            box-shadow: 0 6px 18px rgba(15,23,42,0.05);
        }
        .wm-asset-picker-row .wm-picker-label {
            font-size: 10px; color: #64748b; font-weight: 800;
            text-transform: uppercase; letter-spacing: 0.1em;
        }
        </style>
        """).strip(),
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([3, 1])
    with top_left:
        instance_id = st.selectbox(
            "🎛️ Activo monitoreado",
            options,
            index=default_idx,
            key="live_asset_v3",
            format_func=_fmt_option,
        )
    with top_right:
        auto_refresh = st.toggle(
            "⟳ Auto-refresh 10s", value=False, key="live_autorefresh_v3",
            help="Recarga automática cada 10 segundos.",
        )

    if not instance_id:
        return

    instance_obj = get_instance(instance_id)
    sensor_lookup = _build_sensor_lookup(instance_obj)
    latest = latest_for_instance(instance_id)

    render_asset_header(instance_obj, instance_id)

    if not latest:
        st.warning(
            "**Sin datos en tiempo real para este activo.** Verificá:\n"
            "1. Tabla `live_readings` creada en Supabase.\n"
            "2. wm-collector corriendo en el PC de planta.\n"
            "3. El collector usa el mismo `instance_id`."
        )
        return

    rendered_rows, severity_summary = compute_rendered_rows(latest, sensor_lookup, instance_obj)

    # Alarm strip prominente arriba si hay danger
    render_alarm_strip(rendered_rows)

    # KPIs
    render_kpi_strip(latest, severity_summary)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # Hero — Machine Map.
    # Prioridad (Ciclo 23.13):
    #   1) Asset library 2D vectorial (System1 / Emerson AMS-style) si la
    #      instancia tiene driver_icon_key + driven_icon_key configurados.
    #      Sensor dots se posicionan en los anchors físicos del icono
    #      (DE / NDE / TRF / CRF) por convención Bently / API 670.
    #   2) PNG 3D legacy con overlay x_pct/y_pct si no hay icon_keys.
    #   3) PNG plano sin overlay si tampoco hay coordenadas.
    has_map = render_sensor_map_library(instance_obj, latest, sensor_lookup)
    if not has_map:
        has_map = render_sensor_map_hero(instance_obj, instance_id, latest, sensor_lookup)
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

    st.markdown("&nbsp;", unsafe_allow_html=True)

    tab_curr, tab_vec, tab_diag, tab_hist = st.tabs([
        "📊 Canales en Vivo",
        "🎯 Vectores 1X/2X",
        "🩺 Diagnostic",
        "📈 Tendencia",
    ])

    with tab_curr:
        # Filtros estilo legacy "monitoreo-estatico" (Plano + Tipo de sensor)
        # Para activos con muchos sensores el usuario puede acotar la vista.
        all_planes = sorted({
            (r.get("plane_label") or "").strip() for r in rendered_rows if r.get("plane_label")
        })
        all_types = sorted({
            r.get("status") for r in rendered_rows if r.get("status")
        })
        f1, f2 = st.columns([2, 1])
        with f1:
            sel_planes = st.multiselect(
                "Filtrar por ubicación / plano",
                options=all_planes, default=[],
                key="live_filter_planes",
                placeholder="Todos los planos" if all_planes else "Sin planos definidos",
            )
        with f2:
            sel_status = st.multiselect(
                "Filtrar por estado",
                options=["Danger", "Alarma", "Normal", "Sin Norma", "No Data"],
                default=[],
                key="live_filter_status",
                placeholder="Todos los estados",
            )

        filtered_rows = rendered_rows
        if sel_planes:
            filtered_rows = [r for r in filtered_rows if (r.get("plane_label") or "") in sel_planes]
        if sel_status:
            filtered_rows = [r for r in filtered_rows if r.get("status") in sel_status]

        if sel_planes or sel_status:
            st.caption(
                f"Mostrando **{len(filtered_rows)}** de **{len(rendered_rows)}** sensores."
            )

        # Trae historial para sparklines
        spark_data = recent_history_all_direct(instance_id, n_per_sensor=30)
        render_channels_table(filtered_rows, spark_data)

    with tab_vec:
        render_vectors_phasors(latest)

    with tab_diag:
        render_diagnostic_table(latest)

    with tab_hist:
        render_history_chart(instance_id, latest, sensor_lookup, instance_obj)
        st.markdown("---")
        total = count_for_instance(instance_id)
        st.caption(f"Total readings históricas almacenadas para `{instance_id}`: **{total:,}**")

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

    if auto_refresh:
        st.markdown('<meta http-equiv="refresh" content="10">', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
