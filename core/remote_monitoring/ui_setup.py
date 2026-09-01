"""
core/remote_monitoring/ui_setup.py — Tab "Setup" (config amigable ADRE/System1)
===============================================================================

Render Streamlit de la configuración de máquina + grid de canales. Estilo
ADRE 408 / System1 pero simple: tarjeta de máquina compacta + grid editable
+ diagrama de sección de cojinete (SVG, bolitas de color) + validación en
vivo API 670. Escribe al modelo único (sensor_map/instance) vía
core.remote_monitoring.config.
"""
from __future__ import annotations

import html
import math
from dataclasses import asdict, fields
from typing import List

import streamlit as st

from core.remote_monitoring import config as cfg


# Paleta (misma que Calibración / Balanceo — clase mundial, no PowerPoint)
NAVY = "#0F1E3D"
CYAN = "#1AAEE5"
CYAN_DARK = "#0F7FB0"
AMBER = "#D89B22"
GRAY = "#6B7280"
GRAY_LIGHT = "#F4F7FB"

# Color de bolita por tipo de sensor
_TYPE_COLOR = {
    "proximity": "#8b5cf6",       # violeta
    "velometer": "#06b6d4",       # cian
    "accelerometer": "#ef4444",   # rojo
    "keyphasor": AMBER,           # ámbar
}

_GRID_COLS = ["bnc_port", "point_label", "plane", "sensor_type",
              "sensitivity_mv_per_eu", "unit_native", "coupling",
              "angle_deg", "side", "alarm", "danger"]


# =====================================================================
# Estado de máquina
# =====================================================================
def _init_machine_defaults() -> None:
    d = {
        "rm_m_name": "Ad-hoc machine", "rm_m_rpm": 3600.0,
        "rm_m_rpmin": 0.0, "rm_m_rpmax": 0.0, "rm_m_rot": "CCW",
        "rm_m_speed": "constant", "rm_m_brgtype": "plain", "rm_m_nbrg": 2,
        "rm_m_iso": "",
        "rm_m_type": "", "rm_m_tag": "", "rm_m_client": "", "rm_m_loc": "",
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
        machine_type=st.session_state.get("rm_m_type", ""),
        tag=st.session_state.get("rm_m_tag", ""),
        client=st.session_state.get("rm_m_client", ""),
        location=st.session_state.get("rm_m_loc", ""),
    )


_INT_FIELDS = {"bnc_port", "plane", "events_per_rev"}
_BOOL_FIELDS = {"active"}
_STR_FIELDS = {"point_label", "sensor_type", "unit_native", "coupling", "side",
               "notch_type", "keyphasor_ref", "pair_ref"}


def _rows_from_records(records: list) -> List[cfg.ChannelRow]:
    """dict → ChannelRow incluyendo TODOS los campos (full_scale, gap, active,
    keyphasor_ref, pair_ref, etc.). Coerción de tipos tolerante."""
    valid = {f.name for f in fields(cfg.ChannelRow)}
    rows: List[cfg.ChannelRow] = []
    for r in records:
        try:
            kw = {}
            for k in valid:
                if k not in r:
                    continue
                v = r[k]
                if k in _INT_FIELDS:
                    kw[k] = int(v or 0)
                elif k in _BOOL_FIELDS:
                    kw[k] = bool(v)
                elif k in _STR_FIELDS:
                    kw[k] = str(v if v is not None else "")
                else:
                    kw[k] = float(v or 0)
            rows.append(cfg.ChannelRow(**kw))
        except Exception:  # noqa: BLE001 — fila incompleta durante edición
            continue
    return rows


def _inject_css() -> None:
    st.markdown(f"""
        <style>
        .rm-sec-head {{
            background:{NAVY}; color:#fff; padding:10px 16px; border-radius:10px 10px 0 0;
            font-weight:700; font-size:14px; letter-spacing:.02em;
            border-bottom:3px solid {CYAN}; margin-top:6px;
        }}
        .rm-sec-head small {{ color:{CYAN}; font-weight:600; }}
        /* Marco vistoso de la tabla — tipo software internacional */
        div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{
            border:1px solid #d6deea; border-radius:0 0 12px 12px;
            box-shadow:0 6px 18px rgba(15,30,61,.08); overflow:hidden;
        }}
        .rm-legend span {{
            display:inline-flex; align-items:center; gap:5px; margin-right:14px;
            font-size:12px; color:{GRAY};
        }}
        .rm-legend i {{ width:13px; height:13px; border-radius:50%;
            display:inline-block; border:2px solid #fff; box-shadow:0 0 0 1px #cbd5e1; }}
        </style>
    """, unsafe_allow_html=True)


# =====================================================================
# Render principal
# =====================================================================
def _load_setup_into_state(name: str) -> None:
    """Carga una configuración guardada al estado del Setup (máquina + canales
    + adquisición). Se puede editar y re-guardar (sobrescribe) o cambiar el
    nombre para crear una nueva a partir de ésta."""
    s = cfg.load_setup(name)
    if s is None:                       # no está local → buscar en la nube (rm_setups)
        s = cfg.load_setup_cloud(name)
    if s is None:
        st.warning(f"Could not load '{name}'.")
        return
    m = s.machine
    st.session_state["rm_m_name"] = m.name
    st.session_state["rm_m_rpm"] = float(m.rpm_nominal)
    st.session_state["rm_m_rpmin"] = float(m.rpm_min)
    st.session_state["rm_m_rpmax"] = float(m.rpm_max)
    st.session_state["rm_m_rot"] = m.rotation if m.rotation in cfg.ROTATIONS else "CCW"
    st.session_state["rm_m_speed"] = m.speed_control if m.speed_control in cfg.SPEED_CONTROLS else "constant"
    st.session_state["rm_m_brgtype"] = m.bearing_type if m.bearing_type in cfg.BEARING_TYPES else "plain"
    st.session_state["rm_m_nbrg"] = int(m.n_bearings)
    st.session_state["rm_m_iso"] = m.iso_norm
    st.session_state["rm_m_type"] = getattr(m, "machine_type", "") or ""
    st.session_state["rm_m_tag"] = getattr(m, "tag", "") or ""
    st.session_state["rm_m_client"] = getattr(m, "client", "") or ""
    st.session_state["rm_m_loc"] = getattr(m, "location", "") or ""
    st.session_state["rm_setup_rows"] = [asdict(c) for c in s.channels]
    st.session_state["rm_acq"] = asdict(s.acquisition)
    st.session_state.pop("rm_edit_idx", None)
    st.success(f"📂 Loaded: {m.name} · {len(s.channels)} channels.")
    st.rerun()


def render_setup() -> None:
    """Configuración de máquina — MISMAS pestañas que el módulo de campo:
    Machine · Sensors & layout · Channel editor · Acquisition · Validation · Summary.
    Lo que se guarda aquí sube a la NUBE (rm_setups) y se puede cargar en campo."""
    _init_machine_defaults()
    _inject_css()
    st.session_state.setdefault("rm_setup_rows", [])

    t_machine, t_sensors, t_chan, t_acq, t_valid, t_summary = st.tabs(
        ["Machine", "Sensors & layout", "Channel editor",
         "Acquisition", "Validation", "Summary"])

    # ============================ MACHINE ============================
    with t_machine:
        st.markdown('<div class="rm-sec-head">Machine '
                    '<small>— asset record &amp; train (API 684)</small></div>',
                    unsafe_allow_html=True)
        st.caption("Enter the machine data, or load a **saved configuration** to "
                   "edit it or create a new one based on it.")

        # --- Configuraciones GUARDADAS (local + nube) — cargar / borrar ---
        # Se unen las locales con las de la NUBE (rm_setups) para que aparezcan también
        # las máquinas creadas en el MÓDULO DE CAMPO. Las de la nube van marcadas con ☁.
        _local = cfg.list_setups()
        try:
            _cloud_rows = cfg.list_setups_cloud()
        except Exception:  # noqa: BLE001
            _cloud_rows = []
        _cloud_names = [(r.get("name") or r.get("id") or "") for r in _cloud_rows
                        if (r.get("name") or r.get("id"))]
        _saved = list(dict.fromkeys([*_local, *_cloud_names]))
        _cloud_only = set(_cloud_names) - set(_local)
        if _saved:
            scol1, scol2, scol3 = st.columns([3, 1, 1])
            with scol1:
                _opts = ["—"] + [(f"☁ {n}" if n in _cloud_only else n) for n in _saved]
                pick_raw = st.selectbox("📂 Load saved configuration", _opts,
                                        key="rm_load_pick",
                                        help="Your saved machines (local + cloud ☁). ☁ = created in "
                                             "the field module. Load it, edit it and re-save.")
                pick_s = pick_raw[1:].strip() if pick_raw.startswith("☁") else pick_raw
            with scol2:
                st.write(""); st.write("")
                if st.button("📂 Load", use_container_width=True) and pick_s != "—":
                    _load_setup_into_state(pick_s)
            with scol3:
                st.write(""); st.write("")
                if st.button("🗑 Delete", use_container_width=True, key="rm_saved_delete") and pick_s != "—":
                    cfg.delete_setup(pick_s)
                    st.session_state.pop("rm_load_pick", None)
                    st.rerun()

        st.markdown("**Asset record**")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.text_input("Machine name / tag", key="rm_m_name")
            st.text_input("Type", key="rm_m_type", placeholder="Turbogenerator, Motor+Pump")
        with a2:
            st.text_input("Asset tag / nameplate", key="rm_m_tag")
            st.text_input("Client", key="rm_m_client")
        with a3:
            st.text_input("Plant / location", key="rm_m_loc")

        st.markdown("**Train & operation**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("RPM nominal", 0.0, 60000.0, key="rm_m_rpm", step=60.0)
            st.selectbox("Bearing type", cfg.BEARING_TYPES, key="rm_m_brgtype")
        with c2:
            st.number_input("RPM min (range)", 0.0, 60000.0, key="rm_m_rpmin", step=60.0)
            st.number_input("No. of bearings", 1, 16, key="rm_m_nbrg")
        with c3:
            st.number_input("RPM max (range)", 0.0, 60000.0, key="rm_m_rpmax", step=60.0)
            st.radio("Rotation direction", cfg.ROTATIONS, key="rm_m_rot", horizontal=True)
        st.radio("Speed control", cfg.SPEED_CONTROLS, key="rm_m_speed", horizontal=True,
                 help="Variable enables start/stop logic (transients).")

    machine = _machine_from_state()

    # ======================= SENSORS & LAYOUT =======================
    with t_sensors:
        st.markdown('<div class="rm-sec-head">Sensors &amp; layout '
                    '<small>— BNC → measurement point</small></div>', unsafe_allow_html=True)
        gcol1, gcol2 = st.columns([1, 3])
        with gcol1:
            if st.button("🧩 Auto-generate layout", use_container_width=True,
                         help="Generates X/Y pairs per bearing + keyphasor from the machine."):
                st.session_state["rm_setup_rows"] = [asdict(r) for r in cfg.auto_layout(machine)]
                st.rerun()
        with gcol2:
            st.caption("Angle convention (API 670): from **TDC (top)**, "
                       "**R** = clockwise, **L** = counter-clockwise → 45°L + 45°R = 90°.")
        row_objs = _rows_from_records(st.session_state["rm_setup_rows"])
        if row_objs:
            st.markdown(_channels_compact_table(row_objs), unsafe_allow_html=True)
            _render_bearing_diagram(row_objs, machine)
        else:
            st.info("Press **🧩 Auto-generate layout**, or add channels in **Channel editor**.")

    # ========================= CHANNEL EDITOR =========================
    with t_chan:
        st.markdown('<div class="rm-sec-head">Channel editor '
                    '<small>— per-channel, full parameters</small></div>', unsafe_allow_html=True)
        _render_channel_editor()
    rows = _rows_from_records(st.session_state["rm_setup_rows"])

    # =========================== ACQUISITION ===========================
    with t_acq:
        st.markdown('<div class="rm-sec-head">Acquisition '
                    '<small>— train + per sensor type (ISO 20816)</small></div>',
                    unsafe_allow_html=True)
        train_acq, acq_by_type = _render_acq_params(rows)

    # ---- setup + hallazgos (se usan en Validation y Summary) ----
    setup = cfg.AcqSetup(machine=machine, channels=rows, acquisition=train_acq,
                         acquisition_by_type=acq_by_type)
    findings = cfg.validate_setup(setup)
    n_err = sum(1 for f in findings if f.level == "error")
    n_warn = sum(1 for f in findings if f.level == "warn")

    # =========================== VALIDATION ===========================
    with t_valid:
        st.markdown('<div class="rm-sec-head">Validation '
                    '<small>— API 670 / ISO 20816</small></div>', unsafe_allow_html=True)
        for f in findings:
            if f.level == "error":
                st.error(f"❌ {f.message}")
            elif f.level == "warn":
                st.warning(f"⚠ {f.message}")
            else:
                st.success(f"✅ {f.message}")
        st.divider()
        scol1, scol2 = st.columns([1, 3])
        with scol1:
            can_save = n_err == 0 and len(setup.channels) > 0
            if st.button("💾 Save configuration", type="primary",
                         use_container_width=True, disabled=not can_save):
                _save_and_activate(setup)
        with scol2:
            if n_err:
                st.caption(f"Fix the {n_err} error(s) before saving.")
            elif n_warn:
                st.caption(f"{n_warn} warning(s) — you can still save, but review them.")
            else:
                st.caption("All OK. Save → shared with the field module (☁) and go to **Monitor**.")

    # ============================= SUMMARY =============================
    with t_summary:
        st.markdown('<div class="rm-sec-head">Summary '
                    '<small>— full read-only view</small></div>', unsafe_allow_html=True)
        if rows:
            _bt = st.session_state.get("rm_acq_by_type", {})
            _unit = st.session_state.get("rm_acq", {}).get("freq_unit", "cpm")
            st.markdown(_channels_html_table(rows, _bt, _unit), unsafe_allow_html=True)
            st.caption(f"**{machine.name}** · {len(rows)} channels · rotation {machine.rotation} · "
                       f"{n_err} error(s), {n_warn} warning(s)")
        else:
            st.info("No channels yet — configure them in **Sensors & layout** / **Channel editor**.")


# =====================================================================
# Editor maestro-detalle de canales (custom, con form + botón Actualizar)
# =====================================================================
def _render_channel_editor() -> None:
    rows = st.session_state["rm_setup_rows"]
    ctrl = st.columns([2, 1, 1])
    idx = None
    with ctrl[0]:
        if rows:
            idx = st.selectbox(
                "Edit channel", list(range(len(rows))),
                format_func=lambda i: f"BNC {rows[i].get('bnc_port', '?')} · {rows[i].get('point_label', '?')}",
                key="rm_edit_idx")
        else:
            st.caption("No channels — use Auto-generate or Add.")
    with ctrl[1]:
        st.write(""); st.write("")
        if st.button("➕ Add", use_container_width=True):
            nb = max([int(r.get("bnc_port", 0) or 0) for r in rows], default=0) + 1
            rows.append(asdict(cfg.ChannelRow(bnc_port=min(nb, 32), point_label=f"CH{nb}")))
            st.rerun()
    with ctrl[2]:
        st.write(""); st.write("")
        if idx is not None and st.button("🗑 Delete", use_container_width=True, key="rm_ch_delete"):
            rows.pop(idx)
            st.session_state.pop("rm_edit_idx", None)
            st.rerun()

    if idx is None or idx >= len(rows):
        return
    _render_channel_form(idx)


def _render_channel_form(idx: int) -> None:
    rows = st.session_state["rm_setup_rows"]
    row = rows[idx]
    is_kph = str(row.get("sensor_type", "")) == "keyphasor"

    def _num(col, label, field, mn, mx, step, as_int=False, help=None):
        v = row.get(field, 0) or 0
        col.number_input(label, min_value=mn, max_value=mx,
                         value=(int(v) if as_int else float(v)), step=step,
                         key=f"f_{field}_{idx}", help=help)

    def _sel(col, label, field, options, help=None):
        cur = row.get(field, options[0])
        col.selectbox(label, options, index=options.index(cur) if cur in options else 0,
                      key=f"f_{field}_{idx}", help=help)

    # st.form: editás libre y NADA se aplica hasta pulsar "Actualizar".
    with st.form(f"rm_chan_form_{idx}"):
        st.markdown(f"**Editing:** `{row.get('point_label','?')}` — fill it in and press "
                    "**🔄 Update** to push it to the table.")
        st.caption("Identification")
        c = st.columns(4)
        c[0].text_input("Point", value=row.get("point_label", ""), key=f"f_point_label_{idx}")
        _num(c[1], "BNC", "bnc_port", 1, 32, 1, as_int=True)
        _num(c[2], "Bearing", "plane", 0, 16, 1, as_int=True, help="0 = no bearing")
        c[3].checkbox("Active", value=bool(row.get("active", True)), key=f"f_active_{idx}")

        st.caption("Transducer")
        c = st.columns(4)
        cur_t = row.get("sensor_type", "proximity")
        c[0].selectbox("Type", cfg.SENSOR_TYPES,
                       index=cfg.SENSOR_TYPES.index(cur_t) if cur_t in cfg.SENSOR_TYPES else 0,
                       key=f"f_sensor_type_{idx}")
        _num(c[1], "Sensitivity mV/EU", "sensitivity_mv_per_eu", 0.0, 5000.0, 1.0)
        _sel(c[2], "Unit", "unit_native", cfg.ALL_UNITS, help="Adjusts to the type on Update.")
        _sel(c[3], "Coupling", "coupling", cfg.COUPLINGS)
        c = st.columns(4)
        _num(c[0], "Full-scale (EU)", "full_scale", 0.0, 100000.0, 1.0, help="Measurement range. 0 = auto")
        if not is_kph:
            _num(c[1], "Gap/Bias (V)", "gap_bias_v", -24.0, 24.0, 0.1,
                 help="Probe DC voltage (prox ~ -9 to -11 V)")

        st.caption("Orientation (TDC top, R clockwise, L counter-clockwise)")
        c = st.columns(4)
        _num(c[0], "Angle °", "angle_deg", 0.0, 360.0, 5.0)
        _sel(c[1], "Side", "side", ["", "L", "R"])

        if is_kph:
            st.caption("Keyphasor")
            c = st.columns(4)
            _num(c[0], "Events/rev", "events_per_rev", 1, 360, 1, as_int=True)
            _num(c[1], "Trigger threshold (V)", "trigger_v", -24.0, 24.0, 0.5)
            _sel(c[2], "Notch type", "notch_type", cfg.NOTCH_TYPES)
        else:
            st.caption("Associations (phase reference + orbit pair)")
            c = st.columns(4)
            kph_opts = [""] + [str(r.get("point_label", "")) for r in rows
                               if str(r.get("sensor_type", "")) == "keyphasor"]
            cur_k = row.get("keyphasor_ref", "")
            c[0].selectbox("Associated keyphasor", kph_opts,
                           index=kph_opts.index(cur_k) if cur_k in kph_opts else 0,
                           key=f"f_keyphasor_ref_{idx}",
                           help="1X phase reference. A train can have several keyphasors.")
            pair_opts = [""] + [str(r.get("point_label", "")) for j, r in enumerate(rows)
                                if j != idx and str(r.get("sensor_type", "")) != "keyphasor"
                                and int(r.get("plane", 0) or 0) == int(row.get("plane", 0) or 0)]
            cur_p = row.get("pair_ref", "")
            c[1].selectbox("X/Y pair (orbit)", pair_opts,
                           index=pair_opts.index(cur_p) if cur_p in pair_opts else 0,
                           key=f"f_pair_ref_{idx}",
                           help="Orthogonal sensor for the orbit, e.g. 1XD ↔ 1YD.")
            st.caption("Alarms (API 670)")
            c = st.columns(4)
            _num(c[0], "Alert", "alarm", 0.0, 100000.0, 0.1)
            _num(c[1], "Danger", "danger", 0.0, 100000.0, 0.1)

        submitted = st.form_submit_button("🔄 Update channel", type="primary",
                                          use_container_width=True)
    if submitted:
        _commit_channel_form(idx, is_kph)
        st.rerun()


def _commit_channel_form(idx: int, is_kph: bool) -> None:
    """Sube los valores del form a la fila idx (al pulsar Actualizar)."""
    rows = st.session_state["rm_setup_rows"]
    if not (0 <= idx < len(rows)):
        return
    r = rows[idx]

    def g(field, default=None):
        return st.session_state.get(f"f_{field}_{idx}", default)

    r["point_label"] = str(g("point_label", r.get("point_label", "")) or "")
    r["bnc_port"] = int(g("bnc_port", r.get("bnc_port", 1)) or 1)
    r["plane"] = int(g("plane", r.get("plane", 0)) or 0)
    r["active"] = bool(g("active", True))
    t = str(g("sensor_type", r.get("sensor_type", "proximity")) or "proximity")
    r["sensor_type"] = t
    u = str(g("unit_native", r.get("unit_native", "")) or "")
    valid = cfg.valid_units_for(t)
    r["unit_native"] = u if (not valid or u in valid) else cfg.default_unit(t)  # autocorrige al tipo
    r["sensitivity_mv_per_eu"] = float(g("sensitivity_mv_per_eu", 0) or 0)
    r["coupling"] = str(g("coupling", "AC") or "AC")
    r["full_scale"] = float(g("full_scale", 0) or 0)
    r["angle_deg"] = float(g("angle_deg", 0) or 0)
    r["side"] = str(g("side", "") or "")
    if is_kph:
        r["events_per_rev"] = int(g("events_per_rev", 1) or 1)
        r["trigger_v"] = float(g("trigger_v", 0) or 0)
        r["notch_type"] = str(g("notch_type", "") or "")
    else:
        r["gap_bias_v"] = float(g("gap_bias_v", 0) or 0)
        r["keyphasor_ref"] = str(g("keyphasor_ref", "") or "")
        r["pair_ref"] = str(g("pair_ref", "") or "")
        r["alarm"] = float(g("alarm", 0) or 0)
        r["danger"] = float(g("danger", 0) or 0)


# =====================================================================
# Parámetros de adquisición — general del tren + POR TIPO de sensor
# =====================================================================
_TYPE_ES = {"proximity": "Proximity", "velometer": "Velocity", "accelerometer": "Accelerometer"}


def _render_acq_params(rows: List[cfg.ChannelRow]):
    """Devuelve (train_acq, acquisition_by_type)."""
    st.markdown('<div class="rm-sec-head">3 · Acquisition parameters '
                '<small>— train-wide + per sensor type</small></div>',
                unsafe_allow_html=True)

    # Tipos espectrales presentes en la config (prox/vel/accel)
    present = [t for t in cfg.SPECTRAL_TYPES
               if any(r.sensor_type == t for r in rows)] or ["proximity"]

    st.session_state.setdefault("rm_acq", asdict(cfg.AcquisitionParams()))
    st.session_state.setdefault(
        "rm_acq_by_type",
        {t: asdict(cfg.default_acq_for_type(t)) for t in cfg.SPECTRAL_TYPES})
    a = st.session_state["rm_acq"]
    bt = st.session_state["rm_acq_by_type"]
    for t in present:
        bt.setdefault(t, asdict(cfg.default_acq_for_type(t)))

    # Toggle de unidad FUERA del form → reconvierte el display en vivo.
    unit = a.get("freq_unit", "cpm")
    ufc = st.columns([1, 3])
    unit = ufc[0].radio("Frequency in", cfg.FREQ_UNITS, horizontal=True,
                        index=cfg.FREQ_UNITS.index(unit) if unit in cfg.FREQ_UNITS else 0,
                        format_func=lambda u: u.upper(), key="rm_acq_funit")
    ul = cfg.freq_label(unit)
    fstep = 60 if unit == "cpm" else 50
    fmax_max = 2_400_000 if unit == "cpm" else 40_000

    with st.form("rm_acq_form"):
        st.caption("Train-wide")
        c = st.columns(3)
        wmode = c[0].selectbox("Waveform", cfg.WAVEFORM_MODES,
                               index=cfg.WAVEFORM_MODES.index(a.get("waveform_mode", "synchronous"))
                               if a.get("waveform_mode", "synchronous") in cfg.WAVEFORM_MODES else 0,
                               key="rm_acq_wmode",
                               help="Synchronous (per revolution, bode/cascade) or asynchronous (fixed Hz)")
        spr = c[1].number_input("Samples/rev (0=auto)", 0, 1024, int(a.get("samples_per_rev", 0)),
                                key="rm_acq_spr")
        orders = c[2].multiselect("Orders (×rpm)", cfg.COMMON_ORDERS,
                                  default=list(a.get("orders", [1.0, 2.0])),
                                  format_func=lambda o: f"{o:g}X", key="rm_acq_orders")

        # Un bloque por tipo de sensor presente
        edited = {}
        for t in present:
            e = bt[t]
            st.divider()
            st.caption(f"📡 {_TYPE_ES.get(t, t)} — own band")
            c = st.columns(3)
            fmax_v = c[0].number_input(
                f"Fmax ({ul})", 60, fmax_max,
                int(round(cfg.hz_to_display(float(e.get("fmax_hz", 1000)), unit))),
                step=fstep, key=f"acq_{t}_fmax_{unit}",
                help="Prox ~1000 Hz (60k CPM); accel ~10 kHz (600k CPM)")
            fmin_v = c[1].number_input(
                f"Fmin ({ul})", 0.0, float(fmax_max),
                float(round(cfg.hz_to_display(float(e.get("fmin_hz", 2.0)), unit), 1)),
                step=float(fstep), key=f"acq_{t}_fmin_{unit}")
            lines = c[2].selectbox("Lines", cfg.LINES_OPTIONS,
                                   index=cfg.LINES_OPTIONS.index(int(e.get("lines", 1600)))
                                   if int(e.get("lines", 1600)) in cfg.LINES_OPTIONS else 2,
                                   key=f"acq_{t}_lines")
            c2 = st.columns(3)
            avg = c2[0].number_input("Averages", 1, 64, int(e.get("averages", 4)), key=f"acq_{t}_avg")
            win = c2[1].selectbox("Window", cfg.WINDOWS,
                                  index=cfg.WINDOWS.index(e.get("window", "hanning"))
                                  if e.get("window", "hanning") in cfg.WINDOWS else 0,
                                  key=f"acq_{t}_win")
            edited[t] = (fmax_v, fmin_v, lines, avg, win)

        submitted = st.form_submit_button("🔄 Update parameters", type="primary",
                                          use_container_width=True)

    if submitted:
        for t, (fmax_v, fmin_v, lines, avg, win) in edited.items():
            bt[t] = asdict(cfg.AcquisitionParams(
                fmax_hz=cfg.display_to_hz(float(fmax_v), unit),
                fmin_hz=cfg.display_to_hz(float(fmin_v), unit),
                lines=int(lines), averages=int(avg), window=win,
                waveform_mode=wmode, samples_per_rev=int(spr),
                orders=[float(o) for o in (orders or [1.0])], freq_unit=unit))
        # Train-global: hereda fmax del primer tipo presente como fallback
        base = bt[present[0]]
        st.session_state["rm_acq"] = asdict(cfg.AcquisitionParams(
            fmax_hz=float(base["fmax_hz"]), fmin_hz=float(base["fmin_hz"]),
            lines=int(base["lines"]), averages=int(base["averages"]), window=base["window"],
            waveform_mode=wmode, samples_per_rev=int(spr),
            orders=[float(o) for o in (orders or [1.0])], freq_unit=unit))
        st.session_state["rm_acq_by_type"] = bt
        st.rerun()

    # Estado COMMITTED (desde session)
    valid = {f.name for f in fields(cfg.AcquisitionParams)}
    train = cfg.AcquisitionParams(**{k: v for k, v in a.items() if k in valid})
    by_type = {t: cfg.AcquisitionParams(**{k: v for k, v in bt[t].items() if k in valid})
               for t in bt}
    resume = " · ".join(
        f"{_TYPE_ES.get(t, t)}: {cfg.hz_to_display(by_type[t].fmax_hz, unit):.0f} {ul}/"
        f"{by_type[t].lines}L (Δf {cfg.hz_to_display(by_type[t].delta_f(), unit):.3g})"
        for t in present)
    st.caption(f"{resume} · {train.waveform_mode} · orders "
               f"{', '.join(f'{o:g}X' for o in train.orders)}")
    return train, by_type


# =====================================================================
# Diagrama SVG de sección de cojinete
# =====================================================================
def _rot_arrow(cx: float, cy: float, r: float, rotation: str) -> str:
    """Flecha circular de sentido de giro (SVG)."""
    cw = (rotation or "").upper() == "CW"
    start_deg, end_deg, sweep = (135, 45, 1) if cw else (45, 135, 0)

    def pt(a):
        rad = math.radians(a)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    x0, y0 = pt(start_deg)
    x1, y1 = pt(end_deg)
    arc = (f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 1 {sweep} {x1:.1f},{y1:.1f}" '
           f'fill="none" stroke="{CYAN_DARK}" stroke-width="3.5" stroke-linecap="round"/>')
    tangent = end_deg + (90 if cw else -90)
    a1 = math.radians(tangent + 150)
    a2 = math.radians(tangent - 150)
    s = 12
    p1 = (x1 + s * math.cos(a1), y1 + s * math.sin(a1))
    p2 = (x1 + s * math.cos(a2), y1 + s * math.sin(a2))
    head = (f'<path d="M{x1:.1f},{y1:.1f} L{p1[0]:.1f},{p1[1]:.1f} '
            f'L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{CYAN_DARK}"/>')
    lbl = (f'<text x="{cx:.0f}" y="{cy+5:.0f}" text-anchor="middle" font-size="15" '
           f'font-weight="800" fill="{NAVY}">{"CW ↻" if cw else "CCW ↺"}</text>')
    return arc + head + lbl


def _bearing_diagram_svg(probes: List[cfg.ChannelRow], machine: cfg.MachineConfig) -> str:
    C = 210.0
    R_out = 160.0
    R_ball = 138.0
    rb = 28.0
    parts: List[str] = []
    parts.append(f'''<defs>
      <radialGradient id="rm_house" cx="42%" cy="38%" r="72%">
        <stop offset="0%" stop-color="#26406e"/><stop offset="100%" stop-color="{NAVY}"/>
      </radialGradient>
      <radialGradient id="rm_shaft" cx="40%" cy="35%" r="75%">
        <stop offset="0%" stop-color="#e8eef6"/><stop offset="100%" stop-color="#9aa8bd"/>
      </radialGradient>
      <filter id="rm_sh" x="-40%" y="-40%" width="180%" height="180%">
        <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#0f1e3d" flood-opacity="0.35"/>
      </filter>
    </defs>''')
    # alojamiento + banda + eje
    parts.append(f'<circle cx="{C}" cy="{C}" r="{R_out}" fill="url(#rm_house)" stroke="{NAVY}" stroke-width="2"/>')
    parts.append(f'<circle cx="{C}" cy="{C}" r="116" fill="#ffffff"/>')
    parts.append(f'<circle cx="{C}" cy="{C}" r="60" fill="url(#rm_shaft)" stroke="#7c8ba3" stroke-width="1.5"/>')
    # marca TDC
    parts.append(f'<path d="M{C-9},{C-R_out+3} L{C+9},{C-R_out+3} L{C},{C-R_out+17} Z" fill="{CYAN}"/>')
    parts.append(f'<text x="{C}" y="{C-R_out-9}" text-anchor="middle" font-size="12" '
                 f'font-weight="700" fill="{NAVY}">TDC 0°</text>')
    # flecha de giro
    parts.append(_rot_arrow(C, C, 84, machine.rotation))
    # bolitas de sondas
    for p in probes:
        theta = cfg.absolute_angle(p.angle_deg, p.side)
        rad = math.radians(theta)
        x = C + R_ball * math.sin(rad)
        y = C - R_ball * math.cos(rad)
        color = _TYPE_COLOR.get(p.sensor_type, "#475569")
        label = html.escape((p.point_label or "")[:5])
        parts.append(f'<line x1="{C+80*math.sin(rad):.1f}" y1="{C-80*math.cos(rad):.1f}" '
                     f'x2="{x:.1f}" y2="{y:.1f}" stroke="#cbd5e1" stroke-width="2"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rb}" fill="{color}" '
                     f'stroke="#ffffff" stroke-width="3.5" filter="url(#rm_sh)"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="12.5" '
                     f'font-weight="800" fill="#ffffff" font-family="monospace">{label}</text>')
        lx = C + (R_out + 24) * math.sin(rad)
        ly = C - (R_out + 24) * math.cos(rad)
        side_txt = f"{p.angle_deg:.0f}°{p.side}".strip()
        parts.append(f'<text x="{lx:.1f}" y="{ly+4:.1f}" text-anchor="middle" font-size="11" '
                     f'fill="{GRAY}">{html.escape(side_txt)}</text>')
    return (f'<svg viewBox="0 0 420 452" width="100%" style="max-width:430px;display:block;'
            f'margin:0 auto" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>')


def _render_bearing_diagram(rows: List[cfg.ChannelRow], machine: cfg.MachineConfig) -> None:
    planes = sorted({r.plane for r in rows if r.plane > 0})
    if not planes:
        return
    st.markdown('<div class="rm-sec-head">Bearing section '
                '<small>— physical probe positions</small></div>', unsafe_allow_html=True)
    dcol1, dcol2 = st.columns([3, 2])
    with dcol1:
        sel = st.selectbox("Bearing", planes, key="rm_polar_brg",
                           format_func=lambda p: f"Bearing {p}")
        probes = [r for r in rows if r.plane == sel]   # incluye keyphasor asignado al cojinete
        st.markdown(_bearing_diagram_svg(probes, machine), unsafe_allow_html=True)
    with dcol2:
        # leyenda de colores
        st.markdown(
            '<div class="rm-legend" style="margin:8px 0 12px 0;">'
            f'<span><i style="background:{_TYPE_COLOR["proximity"]}"></i>Proximity</span>'
            f'<span><i style="background:{_TYPE_COLOR["velometer"]}"></i>Velocity</span>'
            f'<span><i style="background:{_TYPE_COLOR["accelerometer"]}"></i>Accelerometer</span>'
            f'<span><i style="background:{_TYPE_COLOR["keyphasor"]}"></i>Keyphasor</span>'
            '</div>', unsafe_allow_html=True)
        radials = [r for r in probes if r.sensor_type in ("proximity", "velometer", "accelerometer")]
        if len(radials) >= 2:
            a0 = cfg.absolute_angle(radials[0].angle_deg, radials[0].side)
            a1 = cfg.absolute_angle(radials[1].angle_deg, radials[1].side)
            sep = cfg.angular_separation(a0, a1)
            if abs(sep - 90.0) <= 5.0:
                st.success(f"✅ Orthogonal pair: {sep:.0f}° between {radials[0].point_label} "
                           f"and {radials[1].point_label}.")
            else:
                st.warning(f"⚠ {radials[0].point_label}–{radials[1].point_label} at {sep:.0f}° "
                           "(not 90°). For a correct orbit they should be at 90°.")
        kph = [r for r in probes if r.sensor_type == "keyphasor"]
        if kph:
            st.caption(f"🔑 Keyphasor **{kph[0].point_label}** on this bearing "
                       f"({cfg.absolute_angle(kph[0].angle_deg, kph[0].side):.0f}° abs).")


def _channels_compact_table(rows: List[cfg.ChannelRow]) -> str:
    """Tabla compacta (read-only) — solo Channel/Type/BNC/Angle/Side, como en el
    módulo de campo. Deja más espacio al plano gráfico."""
    heads = ["Channel", "Type", "BNC", "Angle", "Side"]
    th = "".join(
        f'<th style="padding:9px 14px;text-align:left;font-size:11px;'
        f'letter-spacing:.04em;text-transform:uppercase;font-weight:700;'
        f'color:{CYAN};border:none;white-space:nowrap;">{html.escape(h)}</th>' for h in heads)
    dash = '<span style="color:#94a3b8;">—</span>'
    body = []
    for i, r in enumerate(rows):
        color = _TYPE_COLOR.get(r.sensor_type, "#475569")
        muted = "" if r.active else "opacity:.45;"
        bg = "#ffffff" if i % 2 == 0 else GRAY_LIGHT
        dot = (f'<span style="display:inline-block;width:13px;height:13px;'
               f'border-radius:50%;background:{color};border:2px solid #fff;'
               f'box-shadow:0 0 0 1px #cbd5e1;margin-right:9px;vertical-align:-2px;"></span>')
        tds = [
            f'{dot}<b style="color:{NAVY};">{html.escape(r.point_label)}</b>',
            html.escape(r.sensor_type),
            f'<span style="font-family:monospace;">{r.bnc_port}</span>',
            f'<span style="font-family:monospace;">{r.angle_deg:.0f}°</span>',
            html.escape(r.side) if r.side else dash,
        ]
        cells = "".join(
            f'<td style="padding:11px 14px;font-size:13px;color:#334155;{muted}'
            f'border-top:1px solid #e8edf5;white-space:nowrap;">{c}</td>' for c in tds)
        body.append(f'<tr style="background:{bg};">{cells}</tr>')
    return (
        f'<div style="border:1px solid #d6deea;border-radius:12px;overflow:hidden;'
        f'box-shadow:0 6px 18px rgba(15,30,61,.08);margin:6px 0 4px 0;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="background:{NAVY};">{th}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _channels_html_table(rows: List[cfg.ChannelRow], acq_by_type: dict = None,
                         freq_unit: str = "cpm") -> str:
    """Tabla HTML pulida (read-only) de los canales — clase mundial.

    Incluye la columna 'Banda' = Fmax/líneas que USA cada canal según su tipo
    (proximidad/velocidad/acelerómetro), leído de acquisition_by_type. Así al
    marcar un canal como proximidad se ve de una qué banda de adquisición usa.
    """
    acq_by_type = acq_by_type or {}
    ul = cfg.freq_label(freq_unit)

    def _band(r):
        if r.sensor_type not in cfg.SPECTRAL_TYPES:
            return None
        e = acq_by_type.get(r.sensor_type)
        fmax = float(e.get("fmax_hz")) if e and "fmax_hz" in e else cfg.default_acq_for_type(r.sensor_type).fmax_hz
        lines = int(e.get("lines")) if e and "lines" in e else cfg.default_acq_for_type(r.sensor_type).lines
        return f"{cfg.hz_to_display(fmax, freq_unit):.0f} {ul}·{lines}L"

    heads = ["Point", "BNC", "Brg.", "Type", "Sens.", "Unit", "Band", "FS", "Gap V",
             "Coupl.", "Angle", "Kph", "Pair", "Alert", "Danger", "Act."]
    th = "".join(
        f'<th style="padding:10px 12px;text-align:left;font-size:11px;'
        f'letter-spacing:.04em;text-transform:uppercase;font-weight:700;'
        f'color:{CYAN};border:none;white-space:nowrap;">{html.escape(h)}</th>' for h in heads)

    def _num(v):
        try:
            return f"{float(v):g}"
        except Exception:  # noqa: BLE001
            return html.escape(str(v))

    dash = '<span style="color:#94a3b8;">—</span>'
    body = []
    for i, r in enumerate(rows):
        color = _TYPE_COLOR.get(r.sensor_type, "#475569")
        muted = "" if r.active else "opacity:.45;"
        bg = "#ffffff" if i % 2 == 0 else GRAY_LIGHT
        dot = (f'<span style="display:inline-block;width:13px;height:13px;'
               f'border-radius:50%;background:{color};border:2px solid #fff;'
               f'box-shadow:0 0 0 1px #cbd5e1;margin-right:9px;vertical-align:-2px;"></span>')
        ang = f"{r.angle_deg:.0f}°{(' ' + r.side) if r.side else ''}"
        coup = (f'<span style="background:{NAVY};color:#fff;font-size:10.5px;'
                f'font-weight:700;padding:2px 9px;border-radius:999px;">{html.escape(r.coupling)}</span>')
        act = ('<span style="color:#16a34a;font-weight:800;">✓</span>' if r.active
               else '<span style="color:#94a3b8;font-weight:800;">✗</span>')
        tds = [
            f'{dot}<b style="color:{NAVY};">{html.escape(r.point_label)}</b>',
            f'<span style="font-family:monospace;">{r.bnc_port}</span>',
            html.escape(str(r.plane)) if r.plane else dash,
            html.escape(r.sensor_type),
            f'<span style="font-family:monospace;">{_num(r.sensitivity_mv_per_eu)}</span>',
            html.escape(r.unit_native),
            (f'<span style="font-family:monospace;color:{CYAN_DARK};">{html.escape(_band(r))}</span>'
             if _band(r) else dash),
            f'<span style="font-family:monospace;">{_num(r.full_scale)}</span>' if r.full_scale else dash,
            f'<span style="font-family:monospace;">{_num(r.gap_bias_v)}</span>' if r.gap_bias_v else dash,
            coup,
            f'<span style="font-family:monospace;">{html.escape(ang)}</span>',
            f'<span style="font-family:monospace;color:{AMBER};">🔑{html.escape(r.keyphasor_ref)}</span>' if r.keyphasor_ref else dash,
            f'<span style="font-family:monospace;color:{CYAN_DARK};font-weight:700;">{html.escape(r.pair_ref)}</span>' if r.pair_ref else dash,
            f'<span style="font-family:monospace;color:{AMBER};">{_num(r.alarm)}</span>' if r.alarm else dash,
            f'<span style="font-family:monospace;color:#dc2626;">{_num(r.danger)}</span>' if r.danger else dash,
            act,
        ]
        cells = "".join(
            f'<td style="padding:11px 12px;font-size:13px;color:#334155;{muted}'
            f'border-top:1px solid #e8edf5;white-space:nowrap;">{c}</td>' for c in tds)
        body.append(f'<tr style="background:{bg};">{cells}</tr>')

    return (
        f'<div style="border:1px solid #d6deea;border-radius:12px;overflow-x:auto;'
        f'box-shadow:0 6px 18px rgba(15,30,61,.08);margin:6px 0 4px 0;">'
        f'<table style="width:100%;border-collapse:collapse;min-width:1180px;">'
        f'<thead><tr style="background:{NAVY};">{th}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _watermelon_success(title: str, detail: str) -> None:
    """Celebración con sandías cayendo + banner de éxito (on-brand 🍉)."""
    positions = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]
    spans = "".join(
        f'<span style="left:{p}%;animation-delay:{i * 0.11:.2f}s;'
        f'font-size:{26 + (i % 3) * 9}px;">🍉</span>'
        for i, p in enumerate(positions))
    st.markdown(f"""
        <div class="wm-rain">{spans}</div>
        <div class="wm-save-ok">
          <div class="wm-save-emoji">🍉</div>
          <div>
            <div class="wm-save-title">{html.escape(title)}</div>
            <div class="wm-save-detail">{html.escape(detail)}</div>
          </div>
        </div>
        <style>
        .wm-rain {{ position:fixed; inset:0; pointer-events:none; z-index:9999; overflow:hidden; }}
        .wm-rain span {{ position:absolute; top:-60px;
            animation: wm-fall 2.9s cubic-bezier(.35,0,.65,1) forwards; }}
        @keyframes wm-fall {{ 0%{{transform:translateY(0) rotate(0);opacity:1}}
            100%{{transform:translateY(110vh) rotate(340deg);opacity:.12}} }}
        .wm-save-ok {{ display:flex; align-items:center; gap:16px;
            background:linear-gradient(135deg,{NAVY} 0%,#0F766E 100%); color:#fff;
            padding:18px 22px; border-radius:14px; margin:12px 0 4px 0;
            box-shadow:0 10px 28px rgba(15,118,110,.28); animation: wm-pop .4s ease; }}
        @keyframes wm-pop {{ from{{transform:scale(.96);opacity:0}} to{{transform:scale(1);opacity:1}} }}
        .wm-save-emoji {{ font-size:40px; line-height:1; }}
        .wm-save-title {{ font-weight:800; font-size:18px; }}
        .wm-save-detail {{ color:rgba(226,232,240,.9); font-size:13px; margin-top:3px; }}
        </style>
    """, unsafe_allow_html=True)


def _save_and_activate(setup: cfg.AcqSetup) -> None:
    try:
        path = cfg.save_setup(setup)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not save: {type(e).__name__}: {e}")
        return
    # También subir a la NUBE (rm_setups) → la máquina queda disponible en el módulo
    # de campo (y viceversa). No es fatal si no hay internet: queda guardada local.
    try:
        _rc = cfg.save_setup_cloud(setup)
        if _rc.get("ok"):
            st.caption("☁ Also saved to the cloud (shared with the field module).")
        elif _rc.get("reason") and _rc.get("reason") != "offline":
            st.caption(f"⚠ Could not upload to the cloud: {_rc.get('reason')}")
    except Exception:  # noqa: BLE001
        pass
    st.session_state["rm_channels"] = cfg.setup_to_channel_configs(setup)
    st.session_state["rm_machine_rpm"] = float(setup.machine.rpm_nominal)
    st.session_state["rm_machine_name"] = setup.machine.name
    st.session_state["rm_machine_rotation"] = setup.machine.rotation
    st.session_state["rm_active_setup"] = setup.machine.name
    # Params de adquisición → el Monitor los usa (Fmax en spectrum/cascade)
    st.session_state["rm_acq_saved"] = asdict(setup.acquisition)
    # Por tipo → el Monitor elige Fmax según el tipo del canal graficado
    st.session_state["rm_acq_by_type_saved"] = {t: asdict(p) for t, p in setup.acquisition_by_type.items()}
    st.session_state["rm_type_by_name"] = {c.point_label: c.sensor_type for c in setup.channels}
    st.session_state["rm_alarms_by_name"] = {c.point_label: (float(c.alarm or 0), float(c.danger or 0))
                                             for c in setup.channels}
    st.session_state["rm_gap_by_name"] = {c.point_label: float(c.gap_bias_v or 0.0)
                                          for c in setup.channels}
    # Ángulo ABSOLUTO de montaje de cada sonda (desde TDC, horario) → órbita real
    st.session_state["rm_angle_by_name"] = {
        c.point_label: float(cfg.absolute_angle(c.angle_deg, c.side)) for c in setup.channels}
    # Pares X/Y explícitos → órbita en el Monitor
    st.session_state["rm_pairs_saved"] = [list(p) for p in cfg.orbit_pairs(setup.channels)]
    _watermelon_success(
        "Configuration saved successfully! 🍉",
        f"{setup.machine.name} · {len(setup.channels)} channels. "
        f"Go to the Monitoring tab and press ▶ Start to acquire.")
