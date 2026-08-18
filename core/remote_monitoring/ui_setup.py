"""
core/remote_monitoring/ui_setup.py — Tab "Setup" (config amigable ADRE/System1)
===============================================================================

Render Streamlit de la configuración de máquina + grid de canales. Estilo
ADRE 408 / System1 pero simple: tarjeta de máquina compacta + grid editable
+ validación en vivo API 670. Escribe al modelo único (sensor_map/instance)
vía core.remote_monitoring.config.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.remote_monitoring import config as cfg


_GRID_COLS = ["bnc_port", "point_label", "plane", "sensor_type",
              "sensitivity_mv_per_eu", "unit_native", "coupling",
              "angle_deg", "side", "alarm", "danger"]


def _init_machine_defaults() -> None:
    d = {
        "rm_m_name": "Máquina ad-hoc", "rm_m_rpm": 3600.0,
        "rm_m_rpmin": 0.0, "rm_m_rpmax": 0.0, "rm_m_rot": "CCW",
        "rm_m_speed": "constant", "rm_m_brgtype": "plain", "rm_m_nbrg": 2,
        "rm_m_iso": "",
    }
    for k, v in d.items():
        st.session_state.setdefault(k, v)


def _machine_from_state() -> cfg.MachineConfig:
    return cfg.MachineConfig(
        name=st.session_state["rm_m_name"],
        rpm_nominal=float(st.session_state["rm_m_rpm"]),
        rpm_min=float(st.session_state["rm_m_rpmin"]),
        rpm_max=float(st.session_state["rm_m_rpmax"]),
        rotation=st.session_state["rm_m_rot"],
        speed_control=st.session_state["rm_m_speed"],
        bearing_type=st.session_state["rm_m_brgtype"],
        n_bearings=int(st.session_state["rm_m_nbrg"]),
        iso_norm=st.session_state.get("rm_m_iso", ""),
    )


def _rows_from_grid() -> List[cfg.ChannelRow]:
    raw = st.session_state.get("rm_setup_rows", [])
    rows: List[cfg.ChannelRow] = []
    for r in raw:
        try:
            rows.append(cfg.ChannelRow(
                bnc_port=int(r.get("bnc_port", 0) or 0),
                point_label=str(r.get("point_label", "") or ""),
                plane=int(r.get("plane", 0) or 0),
                sensor_type=str(r.get("sensor_type", "proximity") or "proximity"),
                sensitivity_mv_per_eu=float(r.get("sensitivity_mv_per_eu", 0) or 0),
                unit_native=str(r.get("unit_native", "") or ""),
                coupling=str(r.get("coupling", "AC") or "AC"),
                angle_deg=float(r.get("angle_deg", 0) or 0),
                side=str(r.get("side", "") or ""),
                alarm=float(r.get("alarm", 0) or 0),
                danger=float(r.get("danger", 0) or 0),
            ))
        except Exception:  # noqa: BLE001 — fila incompleta durante edición
            continue
    return rows


def render_setup() -> None:
    _init_machine_defaults()

    st.markdown("#### 1 · Máquina")
    st.caption("Definí el tren (API 684). Elegí una plantilla para autocompletar "
               "rpm, cojinete y norma ISO, o cargá manual.")

    # --- plantilla ---
    try:
        from core.machine_templates import list_templates, get_template, suggest_norm_for_template
        templates = list_templates()
    except Exception:  # noqa: BLE001
        templates = []

    if templates:
        tpl_labels = ["— (manual) —"] + [t.label for t in templates]
        tcol1, tcol2 = st.columns([3, 1])
        with tcol1:
            pick = st.selectbox("Plantilla de máquina", tpl_labels, key="rm_tpl_pick")
        with tcol2:
            st.write("")
            st.write("")
            if st.button("Aplicar plantilla", use_container_width=True) and pick != tpl_labels[0]:
                t = templates[tpl_labels.index(pick) - 1]
                st.session_state["rm_m_name"] = t.label
                if t.operating_rpm_nominal:
                    st.session_state["rm_m_rpm"] = float(t.operating_rpm_nominal)
                if t.operating_rpm_range and len(t.operating_rpm_range) >= 2:
                    st.session_state["rm_m_rpmin"] = float(min(t.operating_rpm_range))
                    st.session_state["rm_m_rpmax"] = float(max(t.operating_rpm_range))
                if t.bearing_type:
                    bt = t.bearing_type.lower()
                    st.session_state["rm_m_brgtype"] = bt if bt in cfg.BEARING_TYPES else "plain"
                try:
                    norm, _ = suggest_norm_for_template(t.template_id)
                    st.session_state["rm_m_iso"] = norm or ""
                except Exception:  # noqa: BLE001
                    pass
                st.rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Nombre / tag", key="rm_m_name")
        st.number_input("RPM nominal", 0.0, 60000.0, key="rm_m_rpm", step=60.0)
    with c2:
        st.number_input("RPM mín (rango)", 0.0, 60000.0, key="rm_m_rpmin", step=60.0)
        st.number_input("RPM máx (rango)", 0.0, 60000.0, key="rm_m_rpmax", step=60.0)
    with c3:
        st.radio("Sentido de giro", cfg.ROTATIONS, key="rm_m_rot", horizontal=True)
        st.radio("Speed control", cfg.SPEED_CONTROLS, key="rm_m_speed", horizontal=True,
                 help="Variable habilita lógica de arranque/parada (Fase 2).")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.selectbox("Tipo de cojinete", cfg.BEARING_TYPES, key="rm_m_brgtype")
    with c5:
        st.number_input("Nº de cojinetes", 1, 16, key="rm_m_nbrg")
    with c6:
        if st.session_state.get("rm_m_iso"):
            st.text_input("Norma ISO", key="rm_m_iso", disabled=True)

    st.divider()

    # --- grid de canales ---
    st.markdown("#### 2 · Canales (BNC → punto de medición)")
    gcol1, gcol2 = st.columns([1, 3])
    with gcol1:
        if st.button("🧩 Auto-generar layout", use_container_width=True,
                     help="Genera pares X/Y por cojinete + keyphasor desde la máquina."):
            machine = _machine_from_state()
            rows = cfg.auto_layout(machine)
            st.session_state["rm_setup_rows"] = [
                {c: getattr(r, c) for c in _GRID_COLS} for r in rows]
            st.rerun()
    with gcol2:
        st.caption("Editá cada fila. Tipos: proximity / velometer / accelerometer / "
                   "keyphasor. Convención Bently: Y=0° (TDC), X=90°. Sensib. en mV/EU.")

    if not st.session_state.get("rm_setup_rows"):
        st.info("Pulsá **Auto-generar layout** para empezar, o agregá filas manualmente.")
        st.session_state["rm_setup_rows"] = []

    df = pd.DataFrame(st.session_state["rm_setup_rows"], columns=_GRID_COLS)
    edited = st.data_editor(
        df,
        key="rm_grid_editor",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "bnc_port": st.column_config.NumberColumn("BNC", min_value=1, max_value=32, step=1),
            "point_label": st.column_config.TextColumn("Punto"),
            "plane": st.column_config.NumberColumn("Cojinete", min_value=0, max_value=16, step=1),
            "sensor_type": st.column_config.SelectboxColumn("Tipo", options=cfg.SENSOR_TYPES),
            "sensitivity_mv_per_eu": st.column_config.NumberColumn("Sensib. mV/EU", step=1.0),
            "unit_native": st.column_config.TextColumn("Unidad"),
            "coupling": st.column_config.SelectboxColumn("Coupling", options=cfg.COUPLINGS),
            "angle_deg": st.column_config.NumberColumn("Ángulo °", min_value=0.0, max_value=360.0, step=5.0),
            "side": st.column_config.SelectboxColumn("Lado", options=["", "L", "R"]),
            "alarm": st.column_config.NumberColumn("Alert", step=0.1),
            "danger": st.column_config.NumberColumn("Danger", step=0.1),
        },
    )
    # persistir edición
    st.session_state["rm_setup_rows"] = edited.to_dict("records")

    # --- validación en vivo API 670 ---
    st.markdown("#### 3 · Validación (API 670 / ISO 20816)")
    machine = _machine_from_state()
    setup = cfg.AcqSetup(machine=machine, channels=_rows_from_grid())
    findings = cfg.validate_setup(setup)
    n_err = sum(1 for f in findings if f.level == "error")
    n_warn = sum(1 for f in findings if f.level == "warn")

    for f in findings:
        if f.level == "error":
            st.error(f"❌ {f.message}")
        elif f.level == "warn":
            st.warning(f"⚠ {f.message}")
        else:
            st.success(f"✅ {f.message}")

    st.divider()

    # --- guardar ---
    scol1, scol2 = st.columns([1, 3])
    with scol1:
        can_save = n_err == 0 and len(setup.channels) > 0
        if st.button("💾 Guardar configuración", type="primary",
                     use_container_width=True, disabled=not can_save):
            _save_and_activate(setup)
    with scol2:
        if n_err:
            st.caption(f"Corregí los {n_err} error(es) antes de guardar.")
        elif n_warn:
            st.caption(f"{n_warn} advertencia(s) — podés guardar igual, pero revisá.")
        else:
            st.caption("Todo OK. Guardá y pasá al tab **Monitor** para adquirir.")


def _save_and_activate(setup: cfg.AcqSetup) -> None:
    try:
        path = cfg.save_setup(setup)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo guardar: {type(e).__name__}: {e}")
        return
    # Activar para el tab Monitor
    st.session_state["rm_channels"] = cfg.setup_to_channel_configs(setup)
    st.session_state["rm_machine_rpm"] = float(setup.machine.rpm_nominal)
    st.session_state["rm_machine_name"] = setup.machine.name
    st.session_state["rm_active_setup"] = setup.machine.name
    st.success(f"💾 Guardado: `{path.name}` · {len(setup.channels)} canales. "
               "Andá al tab **Monitor** y pulsá ▶ Iniciar.")
