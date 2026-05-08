"""
pages/02_Live_Monitoring.py
===========================

Live Monitoring (Tier 0 A — Ciclo 23.2 visual refresh).

Lee de `live_readings` (Supabase) los datos en tiempo real que envía el
wm-collector instalado en planta y los renderiza con calidad de producto
internacional:

  • 🔴 LIVE pulsante en el header
  • 4 KPI cards (Velocidad, Sensores, Última lectura, Alarmas activas)
  • Tabs: Valores Actuales · Vectores 1X/2X · Diagnostic · Tendencia
  • Status badges (Normal/Alarma/Danger) con colores por severidad,
    computados desde alarm/danger del sensor o fallback ISO 20816
  • Schematic embebido del activo si está disponible
  • Auto-refresh opcional cada 10s
  • Filtros por plano y tipo de sensor

Diferenciador estratégico: vectores 1X/2X gratis (System1 cobra premium).
"""

from __future__ import annotations

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
)
from core.ui_theme import apply_watermelon_page_style, page_header

st.set_page_config(page_title="Watermelon System | Live Monitoring", layout="wide")
require_login()
require_role(allowed_roles=("admin", "specialist", "client"))
render_user_menu()
apply_watermelon_page_style()


# ============================================================
# CSS local — animaciones + estilos de tabla custom
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <style>
        @keyframes wm-live-pulse {
            0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.55); transform: scale(1); }
            70%  { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);   transform: scale(1.05); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);   transform: scale(1); }
        }
        .wm-live-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ef4444;
            animation: wm-live-pulse 1.6s infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        .wm-live-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 999px;
            background: linear-gradient(135deg, #fef2f2 0%, #fff7ed 100%);
            border: 1px solid #fecaca;
            color: #991b1b;
            font-weight: 800;
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            vertical-align: middle;
        }
        .wm-kpi-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbe5f0;
            border-radius: 16px;
            padding: 14px 18px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
        }
        .wm-kpi-icon {
            font-size: 20px;
            margin-bottom: 4px;
        }
        .wm-kpi-label {
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
        }
        .wm-kpi-value {
            font-size: 26px;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.2;
        }
        .wm-kpi-sub {
            font-size: 11px;
            color: #94a3b8;
            margin-top: 2px;
        }
        .wm-status-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 11px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        table.wm-live-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 13px;
            background: #ffffff;
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid #e5edf7;
        }
        table.wm-live-table thead tr {
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
        }
        table.wm-live-table thead th {
            text-align: left;
            color: #1d4ed8;
            font-weight: 800;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 11px 14px;
            border-bottom: 1px solid #d9e8fb;
        }
        table.wm-live-table tbody td {
            padding: 11px 14px;
            border-bottom: 1px solid #f1f5f9;
            color: #0f172a;
        }
        table.wm-live-table tbody tr:last-child td {
            border-bottom: none;
        }
        table.wm-live-table tbody tr.row-alarm  { background: #fff8e7; }
        table.wm-live-table tbody tr.row-danger { background: #fef2f2; }
        table.wm-live-table .col-num { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
        table.wm-live-table .col-mono { font-family: "SF Mono", Menlo, monospace; font-size: 12px; color: #475569; }
        </style>
        """
    ).strip(),
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS — tiempo / severidad
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


# ISO 20816 / API 670 fallback thresholds cuando el sensor del Sensor Map
# no tiene alarm/danger configurados. Conservadores para máquinas Class III
# (turbogeneradores) — ver ISO 20816-3 Class IV grandes. El usuario puede
# override en el Sensor Map de cada activo.
_ISO_FALLBACK = {
    # (family, unit_lower) → (alarm, danger)
    ("Velocity",     "in/s pk"):    (0.39, 0.61),  # ISO 20816-3 III in pk
    ("Velocity",     "mm/s pk"):    (10.0, 15.5),  # equivalente
    ("Velocity",     "mm/s rms"):   (4.5,  7.1),   # ISO 20816-3 III rms
    ("Velocity",     "in/s rms"):   (0.18, 0.28),  # equivalente
    ("Acceleration", "g pk"):       (2.0,  5.0),   # común para turbinas aero
    ("Acceleration", "g rms"):      (1.4,  3.5),
    ("Proximity",    "mil pp"):     (2.5,  4.0),   # API 670 turbogen 3600 rpm
    ("Proximity",    "µm pp"):      (63.0, 100.0),
    ("Proximity",    "um pp"):      (63.0, 100.0),
}


def _family_from(sensor_type: str, unit: str) -> str:
    s = (sensor_type or "").lower()
    u = (unit or "").lower()
    if s == "velocity" or "mm/s" in u or "in/s" in u:
        return "Velocity"
    if s == "accelerometer" or "g " in u or u.startswith("g") and "ap" not in u:
        return "Acceleration"
    if s == "proximity" or "mil" in u or "µm" in u or "um" in u:
        return "Proximity"
    return ""


def compute_severity(
    value: Optional[float],
    sensor_match: Optional[Dict[str, Any]],
    unit: str,
    sensor_type_hint: str = "",
) -> Tuple[str, str, str]:
    """
    Devuelve (label, color_text, color_bg) para la severidad de un valor.
    Prefiere alarm/danger del sensor del map; si no hay, cae a ISO defaults.
    """
    if value is None:
        return ("No Data", "#475569", "#f1f5f9")

    alarm = float((sensor_match or {}).get("alarm", 0) or 0)
    danger = float((sensor_match or {}).get("danger", 0) or 0)

    if alarm <= 0 or danger <= 0:
        family = _family_from(
            sensor_type_hint or (sensor_match or {}).get("sensor_type", ""),
            unit,
        )
        u_norm = (unit or "").lower().strip()
        # Match con keys del fallback (relajado)
        for (fam, u_key), (a, d) in _ISO_FALLBACK.items():
            if fam == family and u_key == u_norm:
                if alarm <= 0:
                    alarm = a
                if danger <= 0:
                    danger = d
                break

    if alarm <= 0 and danger <= 0:
        return ("Sin Norma", "#92400e", "#fef3c7")

    try:
        v = float(value)
    except Exception:
        return ("No Data", "#475569", "#f1f5f9")

    if danger > 0 and v >= danger:
        return ("Danger", "#991b1b", "#fee2e2")
    if alarm > 0 and v >= alarm:
        return ("Alarma", "#92400e", "#fef3c7")
    return ("Normal", "#166534", "#dcfce7")


def status_pill_html(status: str, fg: str, bg: str) -> str:
    return f'<span class="wm-status-pill" style="background:{bg};color:{fg};">{status}</span>'


# ============================================================
# Sensor map matcher — para acceder a thresholds del activo
# ============================================================

def _build_sensor_lookup(instance_obj) -> Dict[str, Dict[str, Any]]:
    """
    Devuelve un dict {sensor_label: sensor_dict} para hacer match rápido
    entre las readings y los sensores configurados del activo.
    """
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
# RENDER — header & KPIs
# ============================================================

def render_header_with_live_badge(instance_obj, instance_id: str) -> None:
    title_text = instance_id
    sub_parts: List[str] = []
    if instance_obj is not None:
        if instance_obj.tag and instance_obj.tag != instance_id:
            title_text = f"{instance_obj.tag} · {instance_id}"
        if instance_obj.driver_model:
            sub_parts.append(instance_obj.driver_model)
        if instance_obj.driven_model:
            sub_parts.append(instance_obj.driven_model)
        if instance_obj.client:
            sub_parts.append(f"📍 {instance_obj.client}")

    sub_html = " · ".join(sub_parts) if sub_parts else "—"

    st.markdown(
        textwrap.dedent(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 6px;
                flex-wrap: wrap;
            ">
                <div style="font-size: 26px; font-weight: 800; color: #0f172a;">
                    {title_text}
                </div>
                <span class="wm-live-badge">
                    <span class="wm-live-dot"></span> LIVE
                </span>
            </div>
            <div style="font-size: 13px; color: #64748b; margin-bottom: 14px;">
                {sub_html}
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_kpi_strip(
    instance_id: str,
    latest: List[Dict[str, Any]],
    severity_summary: Dict[str, int],
) -> None:
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
    alarms_color = "#22c55e" if alarms_total == 0 else ("#f59e0b" if n_danger == 0 else "#ef4444")
    alarms_sub = (
        "Todo en zona normal" if alarms_total == 0
        else f"{n_alarm} alarma{'s' if n_alarm != 1 else ''} · {n_danger} danger"
    )

    cols = st.columns(4, gap="medium")
    with cols[0]:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="wm-kpi-card">
                    <div class="wm-kpi-icon">⚡</div>
                    <div class="wm-kpi-label">Velocidad</div>
                    <div class="wm-kpi-value">{speed_txt} <span style="font-size:13px;color:#64748b;">rpm</span></div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="wm-kpi-card">
                    <div class="wm-kpi-icon">📡</div>
                    <div class="wm-kpi-label">Sensores reportando</div>
                    <div class="wm-kpi-value">{n_direct}</div>
                    <div class="wm-kpi-sub">canales Direct activos</div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="wm-kpi-card">
                    <div class="wm-kpi-icon">⏱️</div>
                    <div class="wm-kpi-label">Última lectura</div>
                    <div class="wm-kpi-value" style="color:{age_color};">hace {age_txt}</div>
                    <div class="wm-kpi-sub">poll cada ~10s</div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="wm-kpi-card">
                    <div class="wm-kpi-icon">🚨</div>
                    <div class="wm-kpi-label">Alarmas activas</div>
                    <div class="wm-kpi-value" style="color:{alarms_color};">{alarms_total}</div>
                    <div class="wm-kpi-sub">{alarms_sub}</div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )


# ============================================================
# RENDER — tabla de valores actuales con severidad
# ============================================================

def render_current_values_v2(
    latest: List[Dict[str, Any]],
    sensor_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Devuelve (rows_render, severity_summary) — el caller usa el summary
    para alimentar el KPI de alarmas activas.
    """
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
        status, fg, bg = compute_severity(r.get("value"), sensor_match, unit)
        summary[status] = summary.get(status, 0) + 1
        rendered.append({
            "sensor_label": sensor_label,
            "variable": r.get("variable"),
            "value": r.get("value"),
            "unit": unit,
            "age": _format_age(r.get("captured_at", "")),
            "quality": r.get("quality") or "good",
            "status": status,
            "fg": fg,
            "bg": bg,
            "_sort_key": (
                {"Danger": 0, "Alarma": 1, "Sin Norma": 2, "Normal": 3, "No Data": 4}.get(status, 9),
                sensor_label,
            ),
        })

    rendered.sort(key=lambda r: r["_sort_key"])
    return rendered, summary


def render_current_table_html(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        st.info("Sin lecturas Direct para mostrar.")
        return

    body_rows = []
    for r in rows:
        row_class = ""
        if r["status"] == "Alarma":
            row_class = "row-alarm"
        elif r["status"] == "Danger":
            row_class = "row-danger"
        try:
            value_str = f"{float(r['value']):,.4f}" if r["value"] is not None else "—"
        except Exception:
            value_str = "—"
        body_rows.append(
            f'<tr class="{row_class}">'
            f'<td>{status_pill_html(r["status"], r["fg"], r["bg"])}</td>'
            f'<td><b>{r["sensor_label"]}</b></td>'
            f'<td>{r["variable"]}</td>'
            f'<td class="col-num">{value_str}</td>'
            f'<td class="col-mono">{r["unit"]}</td>'
            f'<td class="col-mono">{r["age"]}</td>'
            f'<td class="col-mono">{r["quality"]}</td>'
            f'</tr>'
        )

    table_html = textwrap.dedent(
        """
        <table class="wm-live-table">
            <thead>
                <tr>
                    <th>Status</th>
                    <th>Sensor</th>
                    <th>Variable</th>
                    <th style="text-align:right;">Valor</th>
                    <th>Unidad</th>
                    <th>Edad</th>
                    <th>Quality</th>
                </tr>
            </thead>
            <tbody>
                __ROWS__
            </tbody>
        </table>
        """
    ).strip().replace("__ROWS__", "\n".join(body_rows))

    st.markdown(table_html, unsafe_allow_html=True)


# ============================================================
# RENDER — schematic embebido
# ============================================================

def render_schematic(instance_obj, instance_id: str) -> bool:
    """Devuelve True si renderizó algo, False si no había schematic."""
    if instance_obj is None or not instance_obj.schematic_png:
        return False
    try:
        from core.instance_state import get_instance_document_bytes
        png_bytes = get_instance_document_bytes(instance_id, instance_obj.schematic_png)
        if not png_bytes:
            return False
        st.image(png_bytes, caption="Esquemático del activo", use_container_width=True)
        return True
    except Exception:
        return False


# ============================================================
# RENDER — vectores 1X / 2X
# ============================================================

def render_vectors_table(latest: List[Dict[str, Any]]) -> None:
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
            "Este activo no envía vectores 1X/2X. Sólo proximity probes "
            "Bently 3500/42M (vibration monitor) generan estos datos."
        )
        return

    st.caption(
        "📌 Datos vectoriales que System1 cobra como feature premium. "
        "Acá vienen del 3500/92 directo sin sobreprecio. Útiles para "
        "Polar Plot y Bode Plot al arranque/parada."
    )

    body_rows = []
    for s, slot in sorted(by_sensor.items()):
        def fmt_phasor(amp, ph, threshold=1e-30):
            if amp is None and ph is None:
                return "—"
            # Filtrar fases inválidas (Bently reporta ~1e-41 cuando ampl es ~0)
            if amp is not None and abs(amp) < 1e-6 and ph is not None and abs(ph) < 1e-30:
                return f"{amp:.3f} ∠ —"
            a = f"{amp:.3f}" if amp is not None else "—"
            p = f"{ph:.0f}°" if ph is not None and abs(ph) > threshold else "—"
            return f"{a} ∠ {p}"

        unit = slot.get("unit_ampl") or ""
        body_rows.append(
            f'<tr>'
            f'<td><b>{s}</b></td>'
            f'<td class="col-mono">{fmt_phasor(slot.get("1X_Ampl"), slot.get("1X_Phase"))}</td>'
            f'<td class="col-mono">{fmt_phasor(slot.get("2X_Ampl"), slot.get("2X_Phase"))}</td>'
            f'<td class="col-mono">{unit}</td>'
            f'</tr>'
        )

    table_html = textwrap.dedent(
        """
        <table class="wm-live-table">
            <thead>
                <tr>
                    <th>Sensor</th>
                    <th>1X (Ampl ∠ Phase)</th>
                    <th>2X (Ampl ∠ Phase)</th>
                    <th>Unit</th>
                </tr>
            </thead>
            <tbody>
                __ROWS__
            </tbody>
        </table>
        """
    ).strip().replace("__ROWS__", "\n".join(body_rows))

    st.markdown(table_html, unsafe_allow_html=True)


# ============================================================
# RENDER — diagnostic (Gap / BiasVoltage)
# ============================================================

def render_diagnostic_table(latest: List[Dict[str, Any]]) -> None:
    diag_rows = [r for r in latest if r.get("metric") in ("Gap", "BiasVoltage")]
    if not diag_rows:
        st.info("Sin métricas de diagnostic (Gap / BiasVoltage) en esta instancia.")
        return

    st.caption(
        "🔧 Health del transducer. **Gap** = voltaje DC del proximity probe "
        "(rango típico -7 a -10 V DC). **BiasVoltage** = bias del acelerómetro "
        "(rango típico 10-12 V DC). Valores fuera de rango → sensor degradado."
    )

    body_rows = []
    for r in sorted(diag_rows, key=lambda x: (x.get("sensor_label") or "", x.get("metric") or "")):
        sensor = r.get("sensor_label") or "—"
        var = r.get("variable") or ""
        metric = r.get("metric") or ""
        val = r.get("value")
        unit = r.get("unit") or ""

        # Health check rápido para Gap y Bias
        status = "OK"
        status_color_fg = "#166534"
        status_color_bg = "#dcfce7"
        try:
            v = float(val) if val is not None else None
            if v is not None:
                if metric == "Gap":
                    # Gap típico -7 a -10 V DC
                    if v < -11 or v > -6:
                        status = "Out of Range"
                        status_color_fg = "#92400e"
                        status_color_bg = "#fef3c7"
                elif metric == "BiasVoltage":
                    if v < 8 or v > 14:
                        status = "Out of Range"
                        status_color_fg = "#92400e"
                        status_color_bg = "#fef3c7"
        except Exception:
            pass

        try:
            val_str = f"{float(val):,.4f}" if val is not None else "—"
        except Exception:
            val_str = "—"

        body_rows.append(
            f'<tr>'
            f'<td>{status_pill_html(status, status_color_fg, status_color_bg)}</td>'
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
            <thead>
                <tr>
                    <th>Health</th>
                    <th>Sensor</th>
                    <th>Variable</th>
                    <th>Métrica</th>
                    <th style="text-align:right;">Valor</th>
                    <th>Unidad</th>
                    <th>Edad</th>
                </tr>
            </thead>
            <tbody>
                __ROWS__
            </tbody>
        </table>
        """
    ).strip().replace("__ROWS__", "\n".join(body_rows))

    st.markdown(table_html, unsafe_allow_html=True)


# ============================================================
# RENDER — tendencia histórica
# ============================================================

def render_history_chart(instance_id: str, latest: List[Dict[str, Any]]) -> None:
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
    chosen = st.selectbox("Variable a graficar", labels, key="live_history_var_v2")
    idx = labels.index(chosen)
    sensor_lbl, var_name = options[idx]

    rows = history_for_metric(instance_id, var_name, "Direct", limit=500)
    if not rows:
        st.info("Sin histórico aún. Esperá unos minutos para que el collector acumule.")
        return

    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values(by="captured_at").reset_index(drop=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mín", f"{df['value'].min():.3f}")
    with c2:
        st.metric("Máx", f"{df['value'].max():.3f}")
    with c3:
        st.metric("Promedio", f"{df['value'].mean():.3f}")
    with c4:
        st.metric("Σ Lecturas", f"{len(df):,}")

    df_chart = df.set_index("captured_at")[["value"]]
    df_chart.columns = [f"{sensor_lbl} — {var_name}"]
    st.line_chart(df_chart, height=320)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    page_header(
        title="Live Monitoring",
        subtitle=(
            "Datos en tiempo real desde wm-collector (Bently 3500/92 · OPC UA · MQTT). "
            "Lecturas cada ~10s con vectores 1X/2X síncronos."
        ),
    )

    from core.instance_state import list_instances, get_instance
    instances = list_instances()
    if not instances:
        st.info("No hay activos registrados aún. Andá a Machinery Library para crear uno.")
        return

    options = sorted([i.get("instance_id", "") for i in instances if i.get("instance_id")])
    default_idx = options.index("tes1") if "tes1" in options else 0

    top_left, top_right = st.columns([3, 1])
    with top_left:
        instance_id = st.selectbox("Activo", options, index=default_idx, key="live_asset_v2")
    with top_right:
        auto_refresh = st.toggle("⟳ Auto-refresh 10s", value=False, key="live_autorefresh_v2",
                                  help="Recarga automática cada 10 segundos para mantener la vista al día.")

    if not instance_id:
        return

    instance_obj = get_instance(instance_id)
    sensor_lookup = _build_sensor_lookup(instance_obj)
    latest = latest_for_instance(instance_id)

    # Header con LIVE pulsante
    render_header_with_live_badge(instance_obj, instance_id)

    if not latest:
        st.warning(
            "**Sin datos en tiempo real para este activo.** Verificá que:\n\n"
            "1. La tabla `live_readings` está creada en Supabase.\n"
            "2. El wm-collector está corriendo en el PC de planta.\n"
            "3. El collector usa el mismo `instance_id` que estás filtrando aquí.\n\n"
            "Comandos útiles en el server de planta:\n"
            "```\nGet-Service WatermelonCollector\nGet-Content C:\\watermelon\\collector\\logs\\nssm_stdout.log -Tail 30\n```"
        )
        return

    # Pre-computamos severidad para alimentar KPIs y la tabla
    rendered_rows, severity_summary = render_current_values_v2(latest, sensor_lookup)
    render_kpi_strip(instance_id, latest, severity_summary)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # Tabs principales
    tab_curr, tab_vec, tab_diag, tab_hist = st.tabs([
        "📊 Valores Actuales",
        "🎯 Vectores 1X/2X",
        "🩺 Diagnostic",
        "📈 Tendencia",
    ])

    with tab_curr:
        col_left, col_right = st.columns([3, 2], gap="large")
        with col_left:
            render_current_table_html(rendered_rows)
        with col_right:
            had_schematic = render_schematic(instance_obj, instance_id)
            if not had_schematic:
                # Resumen de severidad por familia si no hay schematic
                st.markdown("##### 🧭 Resumen por estado")
                sev_data = pd.DataFrame([
                    {"Estado": "🟢 Normal",     "Cantidad": severity_summary.get("Normal", 0)},
                    {"Estado": "🟡 Alarma",     "Cantidad": severity_summary.get("Alarma", 0)},
                    {"Estado": "🔴 Danger",     "Cantidad": severity_summary.get("Danger", 0)},
                    {"Estado": "⚪ Sin Norma",  "Cantidad": severity_summary.get("Sin Norma", 0)},
                    {"Estado": "⚫ No Data",    "Cantidad": severity_summary.get("No Data", 0)},
                ])
                st.dataframe(sev_data, use_container_width=True, hide_index=True)

                with st.expander("ℹ️ ¿Cómo se computa la severidad?"):
                    st.markdown(
                        "1. Si el sensor del **Sensor Map** del activo tiene "
                        "`alarm` y `danger` configurados → se usan esos.\n"
                        "2. Si no, se cae a thresholds **ISO 20816-3 / API 670** "
                        "según la unidad nativa del sensor.\n"
                        "3. Si no hay match con ISO tampoco → estado **Sin Norma** "
                        "(configurar thresholds en Machinery Library)."
                    )

    with tab_vec:
        render_vectors_table(latest)

    with tab_diag:
        render_diagnostic_table(latest)

    with tab_hist:
        render_history_chart(instance_id, latest)
        st.markdown("---")
        total = count_for_instance(instance_id)
        st.caption(f"Total readings históricas almacenadas para `{instance_id}`: **{total:,}**")

    st.markdown("---")
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🔄 Refrescar ahora", key="live_refresh_v2", use_container_width=True):
            st.rerun()
    with c2:
        st.caption(
            f"📅 Última sync local: {datetime.now().strftime('%H:%M:%S')} · "
            f"Source: `live_readings` Supabase · Collector: wm-collector v1.0"
        )

    # Auto-refresh via meta tag (simple y confiable, no depende de st_extras)
    if auto_refresh:
        st.markdown(
            '<meta http-equiv="refresh" content="10">',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
