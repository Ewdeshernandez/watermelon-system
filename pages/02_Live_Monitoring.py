"""
pages/02_Live_Monitoring.py
===========================

Live Monitoring (Tier 0 A — Ciclo 23.1).

Vista de "valores actuales" + histórico que lee de la tabla
`live_readings` de Supabase. Cada fila es la última lectura
recibida desde un wm-collector instalado en planta.

Diferenciador vs System1: damos vectores 1X/2X (Ampl + Phase)
GRATIS — System1 cobra por feature de runup tracking que entrega
exactamente lo mismo.

Pasos para que la pagina muestre datos:
    1. Aplicar migración supabase/migrations/20260508_001_live_readings.sql
       en Supabase SQL editor.
    2. Instalar wm-collector en el PC de planta (ver collector/README.md).
    3. Esperar 10s — el collector empieza a postear automáticamente.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

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
# HELPERS
# ============================================================

def _format_age(captured_at_iso: str) -> str:
    """Devuelve '4 s' / '12 s' / '3 min' / '1 h' según antigüedad del dato."""
    try:
        captured = datetime.fromisoformat(captured_at_iso.replace("Z", "+00:00"))
    except Exception:
        return "—"
    now = datetime.now(timezone.utc)
    delta = (now - captured).total_seconds()
    if delta < 0:
        return "ahora"
    if delta < 60:
        return f"{int(delta)} s"
    if delta < 3600:
        return f"{int(delta / 60)} min"
    if delta < 86400:
        return f"{int(delta / 3600)} h"
    return f"{int(delta / 86400)} d"


def _staleness_color(seconds_old: float) -> str:
    if seconds_old < 30:
        return "#22c55e"   # verde
    if seconds_old < 300:
        return "#f59e0b"   # ámbar
    return "#ef4444"        # rojo


def _seconds_since(captured_at_iso: str) -> float:
    try:
        captured = datetime.fromisoformat(captured_at_iso.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - captured).total_seconds()
    except Exception:
        return 999999.0


# ============================================================
# RENDER
# ============================================================

def render_health_strip(instance_id: str, latest: List[Dict[str, Any]]) -> None:
    """Strip superior con metadata global del activo."""
    direct_rows = [r for r in latest if r.get("metric") == "Direct"]
    speed_row = next(
        (r for r in latest if (r.get("variable") or "").lower().startswith("velocidad")),
        None,
    )

    if not latest:
        st.warning(
            "Sin datos en tiempo real para este activo. Verificá que:\n\n"
            "1. La tabla `live_readings` está creada en Supabase "
            "(aplicá la migración `supabase/migrations/20260508_001_live_readings.sql`).\n"
            "2. El wm-collector está corriendo en el PC de planta.\n"
            "3. El collector usa el mismo `instance_id` que estás filtrando aquí."
        )
        return

    # Edad mínima de los datos = la lectura más reciente del activo
    min_age = min(_seconds_since(r["captured_at"]) for r in latest)
    color = _staleness_color(min_age)

    speed_val = speed_row.get("value") if speed_row else None
    speed_txt = f"{float(speed_val):.0f} rpm" if speed_val is not None else "—"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbe5f0;
            border-radius: 18px;
            padding: 16px 20px;
            box-shadow: 0 8px 22px rgba(15,23,42,0.04);
            display: flex;
            gap: 28px;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 14px;
        ">
            <div>
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em;">Velocidad</div>
                <div style="font-size: 22px; font-weight: 800; color: #0f172a;">{speed_txt}</div>
            </div>
            <div style="width: 1px; height: 40px; background: #e2e8f0;"></div>
            <div>
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em;">Sensores reportando</div>
                <div style="font-size: 22px; font-weight: 800; color: #0f172a;">{len(direct_rows)}</div>
            </div>
            <div style="width: 1px; height: 40px; background: #e2e8f0;"></div>
            <div>
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em;">Última lectura</div>
                <div style="font-size: 18px; font-weight: 700; color: {color};">
                    hace {_format_age(min(latest, key=lambda r: _seconds_since(r['captured_at']))['captured_at'])}
                </div>
            </div>
            <div style="width: 1px; height: 40px; background: #e2e8f0;"></div>
            <div>
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em;">Activo</div>
                <div style="font-size: 18px; font-weight: 700; color: #0f172a;">{instance_id}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_current_values(latest: List[Dict[str, Any]]) -> None:
    """Tabla de valores actuales (Direct overall por sensor)."""
    direct_rows = [
        r for r in latest
        if r.get("metric") == "Direct"
        and not (r.get("variable") or "").lower().startswith("velocidad")
    ]
    if not direct_rows:
        st.info("Sin lecturas Direct para mostrar.")
        return

    st.markdown("##### 📊 Valores actuales (Direct overall)")

    df = pd.DataFrame([
        {
            "Sensor": r.get("sensor_label") or "—",
            "Variable": r.get("variable"),
            "Valor": r.get("value"),
            "Unidad": r.get("unit") or "",
            "Edad": _format_age(r.get("captured_at", "")),
            "Quality": r.get("quality") or "good",
        }
        for r in direct_rows
    ])
    df = df.sort_values(by="Sensor", kind="stable").reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_vector_table(latest: List[Dict[str, Any]]) -> None:
    """Tabla de vectores 1X y 2X — diferenciador vs System1 que oculta esto."""
    by_sensor: Dict[str, Dict[str, Any]] = {}
    for r in latest:
        s = r.get("sensor_label")
        m = r.get("metric")
        if not s or m not in ("1X_Ampl", "1X_Phase", "2X_Ampl", "2X_Phase"):
            continue
        slot = by_sensor.setdefault(s, {"sensor": s})
        slot[m] = r.get("value")
        slot["unit_ampl"] = r.get("unit") if "Ampl" in m else slot.get("unit_ampl", "")

    if not by_sensor:
        return

    st.markdown("##### 🎯 Vectores síncronos (1X / 2X — Ampl ∠ Phase)")
    st.caption(
        "Datos vectoriales que System1 cobra como feature premium. "
        "Acá vienen del 3500/92 directo sin sobreprecio."
    )

    rows = []
    for s, slot in by_sensor.items():
        def fmt(amp, ph):
            if amp is None and ph is None:
                return "—"
            a = f"{amp:.3f}" if amp is not None else "—"
            p = f"{ph:.0f}°" if ph is not None else "—"
            return f"{a} ∠ {p}"
        rows.append({
            "Sensor": s,
            "1X": fmt(slot.get("1X_Ampl"), slot.get("1X_Phase")),
            "2X": fmt(slot.get("2X_Ampl"), slot.get("2X_Phase")),
            "Unit": slot.get("unit_ampl") or "",
        })

    df = pd.DataFrame(rows).sort_values(by="Sensor", kind="stable").reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_diagnostic_table(latest: List[Dict[str, Any]]) -> None:
    """Diagnostic: gap voltages + bias voltages — health del transducer."""
    diag_rows = [r for r in latest if r.get("metric") in ("Gap", "BiasVoltage")]
    if not diag_rows:
        return

    st.markdown("##### 🩺 Diagnostic (Gap / Bias Voltage — health del transducer)")
    df = pd.DataFrame([
        {
            "Sensor": r.get("sensor_label"),
            "Variable": r.get("variable"),
            "Métrica": r.get("metric"),
            "Valor": r.get("value"),
            "Unidad": r.get("unit"),
            "Edad": _format_age(r.get("captured_at", "")),
        }
        for r in diag_rows
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_history_chart(instance_id: str, latest: List[Dict[str, Any]]) -> None:
    """Selector de variable + chart de tendencia."""
    direct_rows = [r for r in latest if r.get("metric") == "Direct"]
    if not direct_rows:
        return

    st.markdown("##### 📈 Tendencia histórica")
    options = sorted([(r.get("sensor_label") or "—", r.get("variable")) for r in direct_rows])
    labels = [f"{s} — {v}" for (s, v) in options]
    if not labels:
        return
    chosen = st.selectbox("Variable", labels, key="live_history_var")
    idx = labels.index(chosen)
    _, var_name = options[idx]

    rows = history_for_metric(instance_id, var_name, "Direct", limit=500)
    if not rows:
        st.info("Sin histórico aún para esta variable.")
        return

    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"])
    df = df.sort_values(by="captured_at").reset_index(drop=True)
    df_chart = df.set_index("captured_at")[["value"]]
    st.line_chart(df_chart)


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

    from core.instance_state import list_instances
    instances = list_instances()
    if not instances:
        st.info("No hay activos registrados aún. Andá a Machinery Library para crear uno.")
        return

    options = sorted([i.get("instance_id", "") for i in instances if i.get("instance_id")])
    default_idx = options.index("tes1") if "tes1" in options else 0
    instance_id = st.selectbox("Activo", options, index=default_idx)

    if not instance_id:
        return

    latest = latest_for_instance(instance_id)

    render_health_strip(instance_id, latest)

    if not latest:
        return

    col1, col2 = st.columns([3, 2])
    with col1:
        render_current_values(latest)
    with col2:
        render_diagnostic_table(latest)

    render_vector_table(latest)

    st.markdown("---")
    render_history_chart(instance_id, latest)

    # Refresh button
    st.markdown("---")
    if st.button("🔄 Refrescar datos", key="live_refresh"):
        st.rerun()
    total = count_for_instance(instance_id)
    st.caption(f"Total readings históricas almacenadas para {instance_id}: **{total:,}**")


if __name__ == "__main__":
    main()
