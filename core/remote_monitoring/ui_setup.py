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
from typing import List

import pandas as pd
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


def _rows_from_records(records: list) -> List[cfg.ChannelRow]:
    rows: List[cfg.ChannelRow] = []
    for r in records:
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
def render_setup() -> None:
    _init_machine_defaults()
    _inject_css()

    st.markdown('<div class="rm-sec-head">1 · Máquina '
                '<small>— tren (API 684)</small></div>', unsafe_allow_html=True)
    st.caption("Elegí una plantilla para autocompletar rpm, cojinete y norma ISO, o cargá manual.")

    # --- plantilla ---
    try:
        from core.machine_templates import list_templates, suggest_norm_for_template
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
                 help="Variable habilita lógica de arranque/parada (transitorios).")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.selectbox("Tipo de cojinete", cfg.BEARING_TYPES, key="rm_m_brgtype")
    with c5:
        st.number_input("Nº de cojinetes", 1, 16, key="rm_m_nbrg")
    with c6:
        if st.session_state.get("rm_m_iso"):
            st.text_input("Norma ISO", key="rm_m_iso", disabled=True)

    st.divider()

    # =================================================================
    # 2 · Canales
    # =================================================================
    st.markdown('<div class="rm-sec-head">2 · Canales '
                '<small>— BNC → punto de medición</small></div>', unsafe_allow_html=True)

    gcol1, gcol2 = st.columns([1, 3])
    with gcol1:
        if st.button("🧩 Auto-generar layout", use_container_width=True,
                     help="Genera pares X/Y por cojinete + keyphasor desde la máquina."):
            machine = _machine_from_state()
            rows = cfg.auto_layout(machine)
            st.session_state["rm_setup_rows"] = [{c: getattr(r, c) for c in _GRID_COLS} for r in rows]
            st.rerun()
    with gcol2:
        st.caption("Convención Bently (API 670): ángulo desde **TDC (arriba)**, "
                   "**R** = horario, **L** = antihorario → 45°L + 45°R = 90°. "
                   "La unidad se ajusta al tipo de sensor.")

    st.session_state.setdefault("rm_setup_rows", [])
    has_rows = bool(st.session_state["rm_setup_rows"])

    # Edición en un expander (el grid canvas no se puede estilizar) —
    # Patrón correcto de data_editor: la FUENTE (rm_setup_rows) es estable
    # (solo cambia en Auto-generar / Guardar). Las ediciones se leen del
    # valor de retorno → el cambio de ángulo se toma de una.
    with st.expander("✏️  Editar canales (tabla)", expanded=not has_rows):
        df = pd.DataFrame(st.session_state["rm_setup_rows"], columns=_GRID_COLS)
        edited = st.data_editor(
            df, key="rm_grid_editor", num_rows="dynamic", use_container_width=True,
            column_config={
                "bnc_port": st.column_config.NumberColumn("BNC", min_value=1, max_value=32, step=1, width="small"),
                "point_label": st.column_config.TextColumn("Punto", width="small"),
                "plane": st.column_config.NumberColumn("Cojinete", min_value=0, max_value=16, step=1, width="small"),
                "sensor_type": st.column_config.SelectboxColumn("Tipo", options=cfg.SENSOR_TYPES),
                "sensitivity_mv_per_eu": st.column_config.NumberColumn("Sensib. mV/EU", step=1.0),
                "unit_native": st.column_config.SelectboxColumn("Unidad", options=cfg.ALL_UNITS),
                "coupling": st.column_config.SelectboxColumn("Coupling", options=cfg.COUPLINGS, width="small"),
                "angle_deg": st.column_config.NumberColumn("Ángulo °", min_value=0.0, max_value=360.0, step=5.0, width="small"),
                "side": st.column_config.SelectboxColumn("Lado", options=["", "L", "R"], width="small"),
                "alarm": st.column_config.NumberColumn("Alert", step=0.1, width="small"),
                "danger": st.column_config.NumberColumn("Danger", step=0.1, width="small"),
            },
        )
    rows = _rows_from_records(edited.to_dict("records"))

    # Vista bonita (read-only) de los canales — tipo software internacional
    if rows:
        st.markdown(_channels_html_table(rows), unsafe_allow_html=True)
    else:
        st.info("Pulsá **🧩 Auto-generar layout** para empezar, o abrí "
                "**✏️ Editar canales** para agregar filas.")

    # =================================================================
    # Diagrama de sección de cojinete (SVG pro — bolitas de color)
    # =================================================================
    _render_bearing_diagram(rows, _machine_from_state())

    # =================================================================
    # 3 · Validación
    # =================================================================
    st.markdown('<div class="rm-sec-head">3 · Validación '
                '<small>— API 670 / ISO 20816</small></div>', unsafe_allow_html=True)
    machine = _machine_from_state()
    setup = cfg.AcqSetup(machine=machine, channels=rows)
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
    scol1, scol2 = st.columns([1, 3])
    with scol1:
        can_save = n_err == 0 and len(setup.channels) > 0
        if st.button("💾 Guardar configuración", type="primary",
                     use_container_width=True, disabled=not can_save):
            st.session_state["rm_setup_rows"] = edited.to_dict("records")  # commit
            _save_and_activate(setup)
    with scol2:
        if n_err:
            st.caption(f"Corregí los {n_err} error(es) antes de guardar.")
        elif n_warn:
            st.caption(f"{n_warn} advertencia(s) — podés guardar igual, pero revisá.")
        else:
            st.caption("Todo OK. Guardá y pasá al tab **Monitor** para adquirir.")


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
    st.markdown('<div class="rm-sec-head">Sección de cojinete '
                '<small>— posición física de las sondas</small></div>', unsafe_allow_html=True)
    dcol1, dcol2 = st.columns([3, 2])
    with dcol1:
        sel = st.selectbox("Cojinete", planes, key="rm_polar_brg",
                           format_func=lambda p: f"Cojinete {p}")
        probes = [r for r in rows if r.plane == sel]   # incluye keyphasor asignado al cojinete
        st.markdown(_bearing_diagram_svg(probes, machine), unsafe_allow_html=True)
    with dcol2:
        # leyenda de colores
        st.markdown(
            '<div class="rm-legend" style="margin:8px 0 12px 0;">'
            f'<span><i style="background:{_TYPE_COLOR["proximity"]}"></i>Proximidad</span>'
            f'<span><i style="background:{_TYPE_COLOR["velometer"]}"></i>Velocidad</span>'
            f'<span><i style="background:{_TYPE_COLOR["accelerometer"]}"></i>Acelerómetro</span>'
            f'<span><i style="background:{_TYPE_COLOR["keyphasor"]}"></i>Keyphasor</span>'
            '</div>', unsafe_allow_html=True)
        radials = [r for r in probes if r.sensor_type in ("proximity", "velometer", "accelerometer")]
        if len(radials) >= 2:
            a0 = cfg.absolute_angle(radials[0].angle_deg, radials[0].side)
            a1 = cfg.absolute_angle(radials[1].angle_deg, radials[1].side)
            sep = cfg.angular_separation(a0, a1)
            if abs(sep - 90.0) <= 5.0:
                st.success(f"✅ Par ortogonal: {sep:.0f}° entre {radials[0].point_label} "
                           f"y {radials[1].point_label}.")
            else:
                st.warning(f"⚠ {radials[0].point_label}–{radials[1].point_label} a {sep:.0f}° "
                           "(no 90°). Para órbita correcta van a 90°.")
        kph = [r for r in probes if r.sensor_type == "keyphasor"]
        if kph:
            st.caption(f"🔑 Keyphasor **{kph[0].point_label}** en este cojinete "
                       f"({cfg.absolute_angle(kph[0].angle_deg, kph[0].side):.0f}° abs).")


def _channels_html_table(rows: List[cfg.ChannelRow]) -> str:
    """Tabla HTML pulida (read-only) de los canales — clase mundial.

    Cabecera navy, bolita de color por tipo, filas zebra, badges de coupling,
    monospace para números. El data_editor (canvas) no se puede estilizar, así
    que esta es la vista 'linda'; la edición vive en el expander.
    """
    heads = ["Punto", "BNC", "Cojinete", "Tipo", "Sensib.", "Unidad",
             "Coupling", "Ángulo", "Alert", "Danger"]
    th = "".join(
        f'<th style="padding:10px 12px;text-align:left;font-size:11px;'
        f'letter-spacing:.04em;text-transform:uppercase;font-weight:700;'
        f'color:{CYAN};border:none;">{html.escape(h)}</th>' for h in heads)

    def _num(v):
        try:
            f = float(v)
            return f"{f:g}"
        except Exception:  # noqa: BLE001
            return html.escape(str(v))

    body = []
    for i, r in enumerate(rows):
        color = _TYPE_COLOR.get(r.sensor_type, "#475569")
        bg = "#ffffff" if i % 2 == 0 else GRAY_LIGHT
        dot = (f'<span style="display:inline-block;width:13px;height:13px;'
               f'border-radius:50%;background:{color};border:2px solid #fff;'
               f'box-shadow:0 0 0 1px #cbd5e1;margin-right:9px;vertical-align:-2px;"></span>')
        ang = f"{r.angle_deg:.0f}°{(' ' + r.side) if r.side else ''}"
        coup = (f'<span style="background:{NAVY};color:#fff;font-size:10.5px;'
                f'font-weight:700;padding:2px 9px;border-radius:999px;">{html.escape(r.coupling)}</span>')
        tds = [
            f'{dot}<b style="color:{NAVY};">{html.escape(r.point_label)}</b>',
            f'<span style="font-family:monospace;">{r.bnc_port}</span>',
            html.escape(str(r.plane)) if r.plane else '<span style="color:#94a3b8;">—</span>',
            html.escape(r.sensor_type),
            f'<span style="font-family:monospace;">{_num(r.sensitivity_mv_per_eu)}</span>',
            html.escape(r.unit_native),
            coup,
            f'<span style="font-family:monospace;">{html.escape(ang)}</span>',
            f'<span style="font-family:monospace;color:{AMBER};">{_num(r.alarm)}</span>',
            f'<span style="font-family:monospace;color:#dc2626;">{_num(r.danger)}</span>',
        ]
        cells = "".join(
            f'<td style="padding:11px 12px;font-size:13px;color:#334155;'
            f'border-top:1px solid #e8edf5;">{c}</td>' for c in tds)
        body.append(f'<tr style="background:{bg};">{cells}</tr>')

    return (
        f'<div style="border:1px solid #d6deea;border-radius:12px;overflow:hidden;'
        f'box-shadow:0 6px 18px rgba(15,30,61,.08);margin:6px 0 4px 0;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="background:{NAVY};">{th}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _save_and_activate(setup: cfg.AcqSetup) -> None:
    try:
        path = cfg.save_setup(setup)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo guardar: {type(e).__name__}: {e}")
        return
    st.session_state["rm_channels"] = cfg.setup_to_channel_configs(setup)
    st.session_state["rm_machine_rpm"] = float(setup.machine.rpm_nominal)
    st.session_state["rm_machine_name"] = setup.machine.name
    st.session_state["rm_active_setup"] = setup.machine.name
    st.success(f"💾 Guardado: `{path.name}` · {len(setup.channels)} canales. "
               "Andá al tab **Monitor** y pulsá ▶ Iniciar.")
