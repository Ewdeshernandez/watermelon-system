"""
pages/00_Machinery_Library.py
=============================

Machinery Library — cockpit central del sistema (Ciclo 14a).
Renombrado desde pages/17_Asset_Documents.py + promovido a primera
página post-login. Cada instancia (máquina física específica, ej.
"TES1" del cliente Ecopetrol) tiene su propio perfil técnico
extendido + Vault con manuales y parámetros físicos del cojinete.

Diferencia clave con la versión anterior (Ciclo 7):

  Antes: los datos se asociaban al "profile" (familia/tipo de
  máquina). Esto significaba que si tenías dos turbogeneradores
  Brush idénticos, compartían los mismos manuales y parámetros —
  un bug grave.

  Ahora (Ciclo 8): los datos viven por **instance_id**, identificador
  único de la máquina física. Dos instancias del mismo profile son
  independientes: TES1 puede tener un clearance ligeramente distinto
  al de TES2 después de un rebabbiting, y no se pisan.

Adicionalmente: el formulario de parámetros muestra en vivo los
valores derivados (Cd calculado de los diámetros, Cr, L/D, carga
unitaria) usando core/bearing_calculations, sin que el usuario
tenga que hacer las cuentas a mano.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from core.auth import require_login, render_user_menu, require_role
from core.bearing_calculations import compute_all_derived
from core.document_vault import CAPTURED_PARAMETER_FIELDS, DOCUMENT_TYPES
from core.instance_selector import render_instance_selector
from core.instance_state import (
    add_uploaded_file_to_instance,
    compose_train_description,
    delete_instance,
    get_instance,
    get_instance_document_bytes,
    list_instances,
    remove_instance_document,
    update_instance_header,
    update_instance_parameters_bulk,
)
# Ciclo 22.3 — `create_instance` y `PROFILES` ya no se importan acá: el
# único camino de creación es el wizard (pages/_machinery_wizard.py).
from core.machine_profiles import get_profile
from core.ui_theme import apply_watermelon_page_style, page_header


st.set_page_config(page_title="Watermelon System | Machinery Library", layout="wide")
require_login()
# Ciclo 17.16 — Machinery Library es para staff
require_role(allowed_roles=("admin", "specialist"))
apply_watermelon_page_style()


# ============================================================
# HELPERS UI
# ============================================================

def _bytes_to_human(n: int) -> str:
    if not n:
        return "—"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _format_date(iso_str: str) -> str:
    if not iso_str:
        return "—"
    try:
        return datetime.fromisoformat(iso_str).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_str


# ============================================================
# RENDER SECCIONES
# ============================================================

def render_create_instance_section() -> None:
    """
    Ciclo 22.3 — CTA único al asistente.

    El form legacy inline se eliminó. El wizard guiado de 6 pasos
    (`pages/_machinery_wizard.py`) es el único camino para crear
    activos: garantiza que toda nueva máquina arranca con tren
    coherente, sensor map auto-generado, y parámetros capturados
    sembrados desde el profile. Esto elimina la fuente principal
    de instancias mal configuradas (sin sensores, sin profile coherente).
    """
    st.markdown(
        textwrap.dedent(
            """
            <div style="
                background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
                border: 1px solid #bfd8ff;
                border-radius: 18px;
                padding: 18px 22px;
                box-shadow: 0 8px 22px rgba(37, 99, 235, 0.06);
                margin: 8px 0 14px 0;
                display: flex;
                gap: 16px;
                align-items: center;
                flex-wrap: wrap;
            ">
                <div style="font-size: 38px; line-height: 1;">🧙</div>
                <div style="flex: 1; min-width: 240px;">
                    <div style="font-weight: 800; color: #1e3a8a; font-size: 16px; margin-bottom: 2px;">
                        Create a new machine with the wizard
                    </div>
                    <div style="color: #475569; font-size: 13px; line-height: 1.45;">
                        Guided 6-step wizard: profile → train → operation →
                        supports and sensors → sensor map → seeded parameters.
                        Every new machine starts correctly configured.
                    </div>
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )
    if st.button(
        "🧙 Open creation wizard",
        key="wm_open_wizard_cta",
        use_container_width=True,
        type="primary",
    ):
        try:
            st.switch_page("pages/_machinery_wizard.py")
        except Exception as _e:
            st.error(
                f"Could not open the wizard automatically "
                f"({type(_e).__name__}). Go to the sidebar → "
                f"'🧙 Create asset (wizard)'."
            )


def render_instance_header(state: Dict[str, Any]) -> None:
    """
    Header de la instancia + formulario completo de metadata (Ciclo 14a).
    Tabs por categoría: Identificación · Tren · Operación · Soportes ·
    Sondas · Setpoints · Mantenimiento · Esquemático.
    """
    instance_id = state["instance_id"]
    profile_label = state["profile_label"]
    profile = get_profile(state["profile_key"])

    inst = get_instance(instance_id)
    if inst is None:
        st.warning("Instance not found.")
        return

    # Cabecera resumen
    title_text = inst.tag or instance_id
    if inst.driver_model:
        title_text += f" · {inst.driver_model}"
    if inst.driven_model and "generador" in inst.driven_model.lower():
        title_text += f" + {inst.driven_model}"
    st.markdown(f"## {title_text}")

    sub_parts = [profile_label]
    if profile:
        sub_parts.append(f"ISO {profile.iso_part}")
        sub_parts.append(f"{profile.operating_rpm:.0f} rpm nominal")
        sub_parts.append(profile.bearing_type)
    st.caption(" · ".join(sub_parts))

    if inst.client or inst.site:
        loc_parts = [p for p in [inst.client, inst.site or inst.location] if p]
        st.caption(f"📍 {' · '.join(loc_parts)}")
    if inst.notes:
        with st.expander("Instance notes", expanded=False):
            st.write(inst.notes)

    # Preview del esquemático (si está cargado)
    if inst.schematic_png:
        try:
            png_bytes = get_instance_document_bytes(instance_id, inst.schematic_png)
            if png_bytes:
                st.image(png_bytes, caption="Coupled train schematic", width=480)
        except Exception:
            pass

    with st.expander("Edit full metadata for this instance", expanded=False):
        (tab_id, tab_train, tab_op, tab_sup, tab_pr, tab_set, tab_norm,
         tab_mnt, tab_sch, tab_envio) = st.tabs([
            "Identification", "Coupled train", "Operation", "Supports",
            "Probes", "Setpoints", "ISO Standard", "Maintenance", "Schematic",
            "Client delivery",
        ])

        with st.form(f"edit_header_{instance_id}"):
            with tab_id:
                c1, c2 = st.columns(2)
                with c1:
                    new_tag = st.text_input("Tag", value=inst.tag or "", help="Short operating identifier, e.g. TES1")
                    new_client = st.text_input("Client", value=inst.client or "", help="e.g. ECOPETROL - MAGNEX")
                    new_site = st.text_input("Site / Plant", value=inst.site or "", help="e.g. TERMOSURIA - VILLAVICENCIO")
                with c2:
                    new_asset_class = st.text_input("Asset class", value=inst.asset_class or "", help="e.g. TURBOGENERATOR")
                    new_loc = st.text_input("Location (legacy)", value=inst.location or "", help="legacy free-text field")
                new_notes = st.text_area("Notes", value=inst.notes or "", height=70)

            with tab_train:
                st.markdown("**Driver (driving machine)**")
                d1, d2, d3 = st.columns(3)
                with d1:
                    new_drv_mfr = st.text_input("Driver manufacturer", value=inst.driver_manufacturer or "")
                with d2:
                    new_drv_mdl = st.text_input("Driver model", value=inst.driver_model or "")
                with d3:
                    new_drv_ser = st.text_input("Driver S/N (internal)", value=inst.driver_serial or "")
                st.markdown("**Driven (driven machine)**")
                e1, e2, e3 = st.columns(3)
                with e1:
                    new_dvn_mfr = st.text_input("Driven manufacturer", value=inst.driven_manufacturer or "")
                with e2:
                    new_dvn_mdl = st.text_input("Driven model", value=inst.driven_model or "")
                with e3:
                    new_dvn_ser = st.text_input("Driven S/N (internal)", value=inst.driven_serial or "")
                p1, p2 = st.columns(2)
                with p1:
                    new_power = st.number_input("Nominal power (MW)", value=float(inst.nominal_power_mw or 0.0), min_value=0.0, max_value=2000.0, step=1.0)
                with p2:
                    new_coupling = st.selectbox(
                        "Coupling class",
                        ["", "rigid", "flexible", "fluid"],
                        index=["", "rigid", "flexible", "fluid"].index(inst.coupling_class) if inst.coupling_class in ["", "rigid", "flexible", "fluid"] else 0,
                    )

                # Ciclo 23.165 — Sentido de giro por sección (órbitas).
                # Turbina (driver) y generador (driven) pueden girar opuesto
                # si hay caja reductora. Vacío = auto (se infiere de la órbita).
                st.markdown("**Rotation direction** "
                            "<span style='color:#94a3b8;font-size:0.8rem'>"
                            "(viewed toward the coupling · used in the orbits)</span>",
                            unsafe_allow_html=True)
                _ROT = ["", "CW", "CCW"]
                _ROT_LBL = {"": "Auto (infer)", "CW": "CW — clockwise",
                            "CCW": "CCW — counter-clockwise"}
                rg1, rg2 = st.columns(2)
                with rg1:
                    new_rot_driver = st.selectbox(
                        "Turbine / driver",
                        _ROT, format_func=lambda v: _ROT_LBL[v],
                        index=_ROT.index(inst.rotation_driver) if getattr(inst, "rotation_driver", "") in _ROT else 0,
                    )
                with rg2:
                    new_rot_driven = st.selectbox(
                        "Generator / driven",
                        _ROT, format_func=lambda v: _ROT_LBL[v],
                        index=_ROT.index(inst.rotation_driven) if getattr(inst, "rotation_driven", "") in _ROT else 0,
                    )

            with tab_op:
                o1, o2 = st.columns(2)
                with o1:
                    new_nom_rpm = st.number_input("Nominal RPM", value=float(inst.nominal_rpm or 0.0), min_value=0.0, max_value=200000.0, step=10.0)
                    new_min_rpm = st.number_input("Min operating RPM", value=float(inst.min_rpm or 0.0), min_value=0.0, max_value=200000.0, step=10.0)
                with o2:
                    new_max_rpm = st.number_input("Max operating RPM", value=float(inst.max_rpm or 0.0), min_value=0.0, max_value=200000.0, step=10.0)
                    new_trip_rpm = st.number_input("Trip RPM (overspeed)", value=float(inst.trip_rpm or 0.0), min_value=0.0, max_value=200000.0, step=10.0)
                new_iso_group = st.text_input("ISO group", value=inst.iso_group or "", help="rigid / flexible")

            with tab_sup:
                s1, s2 = st.columns(2)
                with s1:
                    new_sup_type = st.selectbox(
                        "Support type",
                        ["", "fluid_film", "rolling_element", "magnetic", "mixed"],
                        index=["", "fluid_film", "rolling_element", "magnetic", "mixed"].index(inst.support_type) if inst.support_type in ["", "fluid_film", "rolling_element", "magnetic", "mixed"] else 0,
                    )
                with s2:
                    new_sup_count = st.number_input("Number of supports", value=int(inst.support_count or 0), min_value=0, max_value=50, step=1)
                new_sup_detail = st.text_area(
                    "Detail (free text)",
                    value=inst.support_detail or "",
                    height=80,
                    help="e.g. '4 tilting-pad journal bearings, 5 pads, ID 254mm, clearance 8mil'",
                )

            with tab_pr:
                p1, p2 = st.columns(2)
                with p1:
                    new_px = st.number_input("Probe X orientation (°)", value=float(inst.probe_x_orientation_deg or 0.0), min_value=-180.0, max_value=180.0, step=1.0, help="typical 45° (XL) or 0° (vertical)")
                with p2:
                    new_py = st.number_input("Probe Y orientation (°)", value=float(inst.probe_y_orientation_deg or 0.0), min_value=-180.0, max_value=180.0, step=1.0, help="typical -45° (YR) or 90° (horizontal)")

            with tab_set:
                st.caption("If defined, the severity engine uses these real thresholds before the generic ISO ones.")
                a1, a2, a3 = st.columns(3)
                with a1:
                    new_alert = st.number_input("Alert level", value=float(inst.alert_level or 0.0), min_value=0.0, step=0.1)
                with a2:
                    new_danger = st.number_input("Danger level", value=float(inst.danger_level or 0.0), min_value=0.0, step=0.1)
                with a3:
                    new_trip = st.number_input("Trip level", value=float(inst.trip_level or 0.0), min_value=0.0, step=0.1)
                new_sp_unit = st.text_input("Unit", value=inst.setpoint_unit or "", help="e.g. mil pp / mm/s rms")

            # =========================================================
            # Ciclo 17.9 — Tab "Norma ISO"
            # =========================================================
            # Permite asignar la norma de evaluación de vibración
            # (ISO 20816-2/3/4/8, API 670, API 618, etc.) y la clase
            # dentro de la norma. Los setpoints sugeridos se muestran
            # en vivo y el usuario puede overrideearlos con
            # justificación que queda en el reporte.
            with tab_norm:
                from core.iso_thresholds import (
                    list_norms, list_norm_groups, list_classes_for_norm,
                    get_thresholds, suggest_norm_for_machine,
                    suggest_class_for_machine,
                )
                st.caption(
                    "The evaluation standard defines the Warning/Danger setpoints "
                    "for structural vibration, shaft vibration, rotor balancing "
                    "or rotordynamic analysis. If you do not assign one, the system "
                    "falls back to heuristic defaults. If you assign one, Trend, Tabular and "
                    "Reports use these values and cite the standard in the PDF."
                )

                # =====================================================
                # Ciclo 17.10 — Lista AGRUPADA por dominio (4 grupos)
                # =====================================================
                # Vibración (carcasa) · Vibración eje · Balanceo · Rotor.
                # Cada opción del selectbox lleva un prefijo de grupo
                # para que el usuario sepa qué tipo de norma elige.
                _GROUP_TAGS = {
                    "Vibración (carcasa)":          "📳 VIB",
                    "Vibración de eje (proximity)": "🛡️ SHAFT",
                    "Balanceo de rotor":            "⚖️ BAL",
                    "Análisis rotodinámico":        "🔬 ROT",
                }
                _groups = list_norm_groups()
                _norm_codes = [""]
                _norm_labels = ["(unassigned — use heuristic defaults)"]
                _norm_meta_by_code: Dict[str, Dict[str, Any]] = {}
                for grp_name, items in _groups.items():
                    tag = _GROUP_TAGS.get(grp_name, grp_name)
                    for it in items:
                        _norm_codes.append(it["code"])
                        _norm_labels.append(f"{tag}  ·  {it['name']}")
                        _norm_meta_by_code[it["code"]] = {
                            "group": grp_name,
                            "metric": it["metric"],
                            "unit": it["unit"],
                        }

                # Auto-sugerir norma si no hay seleccionada
                _current_norm = inst.iso_norm_code
                if not _current_norm:
                    _suggested = suggest_norm_for_machine(
                        inst.asset_class or "",
                        inst.driver_model or "",
                        inst.driven_model or "",
                    )
                    if _suggested and _suggested in _norm_codes:
                        _sname = next(
                            (lbl for c, lbl in zip(_norm_codes, _norm_labels)
                             if c == _suggested),
                            _suggested,
                        )
                        st.info(f"💡 Suggested standard for this asset: **{_sname}**")

                _norm_idx = _norm_codes.index(_current_norm) if _current_norm in _norm_codes else 0
                _selected_norm_idx = st.selectbox(
                    "Evaluation standard (grouped by domain)",
                    options=range(len(_norm_codes)),
                    format_func=lambda i: _norm_labels[i],
                    index=_norm_idx,
                    key=f"iso_norm_select_{instance_id}",
                )
                new_norm_code = _norm_codes[_selected_norm_idx]

                # Pista de unidad/metric apenas selecciona la norma
                if new_norm_code:
                    _nm = _norm_meta_by_code.get(new_norm_code, {})
                    _grp = _nm.get("group", "")
                    _met = _nm.get("metric", "")
                    _un  = _nm.get("unit", "")
                    _MET_HINTS = {
                        "velocity_rms":         "Structural vibration in RMS velocity.",
                        "velocity_pk":          "Structural vibration in peak velocity.",
                        "displacement_pp":      "Shaft vibration peak-to-peak (proximity probe).",
                        "acceleration_rms":     "RMS acceleration (high frequency).",
                        "unbalance_grade":      "Rotor balance grade (e_per · ω).",
                        "amplification_factor": "Rotordynamic amplification factor (Q/AF) or separation margin.",
                    }
                    st.caption(
                        f"📐 **Domain:** {_grp}  ·  **Metric:** `{_met}`  ·  "
                        f"**Unit:** {_un}.  {_MET_HINTS.get(_met, '')}"
                    )

                new_norm_class = ""
                new_warn_override = 0.0
                new_danger_override = 0.0
                new_justification = inst.override_justification or ""

                if new_norm_code:
                    _classes = list_classes_for_norm(new_norm_code)
                    _class_codes = [c["code"] for c in _classes]
                    _class_labels = [c["label"] for c in _classes]
                    # Sugerir clase si no hay
                    _current_class = inst.iso_norm_class
                    if not _current_class:
                        _suggested_class = suggest_class_for_machine(
                            new_norm_code,
                            float(inst.nominal_power_mw or 0) * 1000.0,  # MW → kW
                            inst.support_type or "",
                        )
                        if _suggested_class:
                            _current_class = _suggested_class

                    _class_idx = (
                        _class_codes.index(_current_class)
                        if _current_class in _class_codes else 0
                    )
                    _selected_class_idx = st.selectbox(
                        "Class / Category within the standard",
                        options=range(len(_class_codes)),
                        format_func=lambda i: _class_labels[i],
                        index=_class_idx,
                        key=f"iso_class_select_{instance_id}",
                    )
                    new_norm_class = _class_codes[_selected_class_idx]

                    _info = get_thresholds(new_norm_code, new_norm_class)
                    if _info:
                        _w = _info["warning"]
                        _d = _info["danger"]
                        _u = _info["unit"]
                        st.success(
                            f"**Setpoints suggested by the standard:**  "
                            f"Warning **{_w} {_u}** · Danger **{_d} {_u}**"
                        )
                        st.caption(f"📚 {_info['reference']}")

                        # Specialist override
                        st.markdown("**Specialist override (optional)**")
                        oc1, oc2 = st.columns(2)
                        with oc1:
                            new_warn_override = float(st.number_input(
                                f"Warning override ({_u})",
                                value=float(inst.setpoint_warning_override or 0.0),
                                min_value=0.0,
                                step=0.1,
                                format="%.3f",
                                help="0 = use the standard value. Any other value "
                                     "overrides it (more conservative or more lenient).",
                                key=f"warn_override_{instance_id}",
                            ))
                        with oc2:
                            new_danger_override = float(st.number_input(
                                f"Danger override ({_u})",
                                value=float(inst.setpoint_danger_override or 0.0),
                                min_value=0.0,
                                step=0.1,
                                format="%.3f",
                                key=f"danger_override_{instance_id}",
                            ))
                        if new_warn_override > 0 or new_danger_override > 0:
                            new_justification = st.text_area(
                                "Override justification (kept in the report)",
                                value=new_justification,
                                height=80,
                                placeholder="e.g. Client requires the conservative "
                                            "Class 1 criterion instead of the standard "
                                            "Class 2 — new machine with no baseline.",
                                key=f"override_just_{instance_id}",
                            )
                            # Mostrar resumen del override efectivo
                            _eff_w = new_warn_override or _w
                            _eff_d = new_danger_override or _d
                            _diff_w = ((_eff_w - _w) / _w * 100) if _w > 0 else 0
                            _diff_d = ((_eff_d - _d) / _d * 100) if _d > 0 else 0
                            st.caption(
                                f"⚙️ **Effective setpoints:** "
                                f"Warning {_eff_w:.3f} ({_diff_w:+.0f}% vs standard) · "
                                f"Danger {_eff_d:.3f} ({_diff_d:+.0f}% vs standard)"
                            )

            with tab_mnt:
                m1, m2 = st.columns(2)
                with m1:
                    new_lb = st.text_input("Last balancing (YYYY-MM-DD)", value=inst.last_balance_date or "")
                    new_la = st.text_input("Last alignment", value=inst.last_alignment_date or "")
                with m2:
                    new_lo = st.text_input("Last major overhaul", value=inst.last_overhaul_date or "")
                    new_co = st.text_input("Commissioning date", value=inst.commissioning_date or "")

            with tab_sch:
                st.caption(
                    "Any image in the asset Vault (PNG / JPG / JPEG / GIF / "
                    "WEBP / SVG) appears as an option here, regardless of the 'document_type' "
                    "you uploaded it with. Select which one to use as the main train "
                    "schematic so it appears in the PDF Executive Summary."
                )
                # Filtro permisivo (Ciclo 14a hotfix 6): acepta documentos que
                # sean imágenes por extensión, además del tipo 'schematic'.
                # Así el usuario no tiene que re-subir si eligió otro tipo
                # cuando lo cargó.
                _SCH_TYPES = ("schematic", "esquematico", "diagram")
                _SCH_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tiff")
                schematic_options = [("", "(no schematic)")]
                for d in inst.documents:
                    dtype = (d.get("document_type") or "").lower()
                    fname = (d.get("filename") or "").lower()
                    is_type_match = dtype in _SCH_TYPES
                    is_ext_match = any(fname.endswith(ext) for ext in _SCH_EXTS)
                    if is_type_match or is_ext_match:
                        label = d.get("title") or d.get("filename") or "—"
                        schematic_options.append((d.get("id", ""), label))
                option_ids = [o[0] for o in schematic_options]
                option_labels = [o[1] for o in schematic_options]
                current_idx = option_ids.index(inst.schematic_png) if inst.schematic_png in option_ids else 0
                new_sch_idx = st.selectbox(
                    "Main schematic",
                    options=range(len(option_ids)),
                    format_func=lambda i: option_labels[i],
                    index=current_idx,
                )
                new_sch_id = option_ids[new_sch_idx]
                if len(schematic_options) == 1:
                    st.warning(
                        "This asset's Vault does not have any image yet. "
                        "Upload a PNG/JPG in the 'Upload new document' section "
                        "below and come back here."
                    )

            with tab_envio:
                st.caption(
                    "Configure who receives the **executive condition report** "
                    "(1 page, PDF) and when it is sent automatically. It is delivered "
                    "by email and/or WhatsApp on the chosen day and time. You can also "
                    "send it manually from Live Monitoring."
                )
                _DOW = ["Monday", "Tuesday", "Wednesday", "Thursday",
                        "Friday", "Saturday", "Sunday"]
                ev1, ev2 = st.columns(2)
                with ev1:
                    new_client_email = st.text_input(
                        "Client email(s)", value=inst.client_email or "",
                        help="One or more, separated by COMMA. "
                             "e.g. boss@client.com, maintenance@client.com",
                    )
                with ev2:
                    new_whatsapp_number = st.text_input(
                        "Client WhatsApp(s)", value=inst.whatsapp_number or "",
                        help="One or more separated by COMMA. With country code, without '+'. "
                             "e.g. 573001234567, 573009998877",
                    )
                new_report_enabled = st.checkbox(
                    "Enable scheduled automatic delivery",
                    value=bool(getattr(inst, "report_send_enabled", False)),
                    help="If enabled, the system sends the report only on the chosen days and times.",
                )
                # Defaults: usar listas nuevas si existen, sino el campo single (back-compat)
                _def_days = [int(x) for x in (getattr(inst, "report_send_days", None) or [])] \
                    or [int(getattr(inst, "report_send_day", 0) or 0)]
                _def_hours = [int(x) for x in (getattr(inst, "report_send_hours", None) or [])] \
                    or [int(getattr(inst, "report_send_hour", 6) or 6)]

                # Días como CHECKBOXES (no st.multiselect: dentro de st.form +
                # st.tabs el multiselect tiene un bug de Streamlit que resetea
                # la pestaña al seleccionar). Las casillas funcionan estable.
                st.markdown("**Delivery days** (check one or more)")
                _abbr = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                _dcols = st.columns(7)
                new_report_days = []
                for _i in range(7):
                    with _dcols[_i]:
                        if st.checkbox(_abbr[_i], value=(_i in _def_days),
                                       key=f"rsday_{instance_id}_{_i}"):
                            new_report_days.append(_i)

                new_report_hours_raw = st.text_input(
                    "Delivery hours (0-23, local time — one or more, separated by comma)",
                    value=", ".join(str(h) for h in _def_hours),
                    help="e.g. 6, 18  → sends at 06:00 and 18:00.",
                )
                new_report_hours = []
                for _tok in new_report_hours_raw.replace(";", " ").replace(",", " ").split():
                    try:
                        _hv = int(_tok)
                        if 0 <= _hv <= 23:
                            new_report_hours.append(_hv)
                    except ValueError:
                        pass
                new_report_hours = sorted(set(new_report_hours))
                st.divider()
                new_alarm_enabled = st.checkbox(
                    "Auto-notify on alarm / danger",
                    value=bool(getattr(inst, "alarm_send_enabled", False)),
                    help="If enabled, it sends the report as soon as a channel crosses into "
                         "Alarm or Danger (checked every 15 min). One notification per episode: "
                         "it does not repeat until the asset returns to normal or worsens.",
                )

                if not (inst.client_email or inst.whatsapp_number):
                    st.info("Enter at least one email or WhatsApp to be able to send.")

            saved = st.form_submit_button("💾 Update full metadata", width="stretch")
            if saved:
                update_instance_header(
                    instance_id,
                    tag=new_tag.strip(),
                    client_email=new_client_email.strip(),
                    whatsapp_number=new_whatsapp_number.strip().replace("+", "").replace(" ", ""),
                    report_send_enabled=bool(new_report_enabled),
                    report_send_days=sorted(int(d) for d in new_report_days),
                    report_send_hours=sorted(int(h) for h in new_report_hours),
                    # back-compat: el primer día/hora también en los campos single
                    report_send_day=int(sorted(new_report_days)[0]) if new_report_days else 0,
                    report_send_hour=int(sorted(new_report_hours)[0]) if new_report_hours else 6,
                    alarm_send_enabled=bool(new_alarm_enabled),
                    client=new_client.strip(),
                    site=new_site.strip(),
                    asset_class=new_asset_class.strip(),
                    location=new_loc.strip(),
                    notes=new_notes.strip(),
                    driver_manufacturer=new_drv_mfr.strip(),
                    driver_model=new_drv_mdl.strip(),
                    driver_serial=new_drv_ser.strip(),
                    driven_manufacturer=new_dvn_mfr.strip(),
                    driven_model=new_dvn_mdl.strip(),
                    driven_serial=new_dvn_ser.strip(),
                    nominal_power_mw=float(new_power),
                    coupling_class=new_coupling.strip(),
                    rotation_driver=new_rot_driver,
                    rotation_driven=new_rot_driven,
                    nominal_rpm=float(new_nom_rpm),
                    min_rpm=float(new_min_rpm),
                    max_rpm=float(new_max_rpm),
                    trip_rpm=float(new_trip_rpm),
                    iso_group=new_iso_group.strip(),
                    support_type=new_sup_type.strip(),
                    support_count=int(new_sup_count),
                    support_detail=new_sup_detail.strip(),
                    probe_x_orientation_deg=float(new_px),
                    probe_y_orientation_deg=float(new_py),
                    alert_level=float(new_alert),
                    danger_level=float(new_danger),
                    trip_level=float(new_trip),
                    setpoint_unit=new_sp_unit.strip(),
                    last_balance_date=new_lb.strip(),
                    last_alignment_date=new_la.strip(),
                    last_overhaul_date=new_lo.strip(),
                    commissioning_date=new_co.strip(),
                    schematic_png=new_sch_id.strip(),
                    # Ciclo 17.9 — norma de evaluación
                    iso_norm_code=(new_norm_code or "").strip(),
                    iso_norm_class=(new_norm_class or "").strip(),
                    setpoint_warning_override=float(new_warn_override or 0.0),
                    setpoint_danger_override=float(new_danger_override or 0.0),
                    override_justification=(new_justification or "").strip(),
                )
                st.success("Metadata updated.")
                st.rerun()


# ============================================================
# Ciclo 14c.1 — SENSOR MAP (mapa de sensores per-instancia)
# ============================================================

def render_sensor_map_section(instance_id: str) -> None:
    """
    Sección "📍 Mapa de Sensores" — editable in-place con st.data_editor.

    Cada fila describe un sensor de vibración con su ubicación física
    (plano + lado + ángulo + dirección) + tipo + unidad nativa +
    setpoints individuales + patrón para matchear el Point del CSV.

    Botones:
      - Generar mapa estándar: pre-llena 8 sensores típicos
        (4 cojinetes × 2 sondas X-Y proximity API 670 a 45° R/L).
      - Limpiar mapa: borra todos los sensores.
      - Guardar mapa: persiste los cambios del data_editor.
    """
    from core.sensor_map import generate_standard_sensor_map, sensor_label

    inst = get_instance(instance_id)
    if inst is None:
        return

    st.markdown("### 📍 Sensor Map")
    st.caption(
        "Configure the asset's physical sensors once: API 670 / "
        "ISO 20816-1 location (planes numbered from **driver → driven**), X/Y direction at 45° R/L, "
        "type (proximity / velocity / accelerometer), native unit and individual "
        "DCS setpoints. Tabular List then classifies each loaded CSV with "
        "the correct thresholds of the matching sensor."
    )

    # Form de generación + botón limpiar
    # Ciclo 14c.1.1 — antes el botón "Generar mapa estándar" asumía un layout
    # único (8 proxímetros + 2 acelerómetros). Ahora pregunta el tipo de
    # soporte de driver y driven por separado, soportando trenes mixtos como
    # turbina aero (rolling_element con TRF/CRF) + generador (fluid_film X-Y).

    with st.expander(
        "🪄 Generate standard map (configurable)",
        expanded=(len(inst.sensors) == 0),
    ):
        st.caption(
            "Configure driver and driven separately. "
            "**Driver = driving machine** (turbine, motor). "
            "**Driven = driven machine** (generator, pump, compressor). "
            "For each side you choose how many planes (bearings) it has and what "
            "support type: `fluid_film` generates an X-Y proximity pair at 45° R/L "
            "(API 670); `rolling_element` generates 1 radial accelerometer per plane."
        )

        # 3 modos de instrumentación. Para que sean amigables, traducimos
        # internamente los keys técnicos a etiquetas claras.
        _MODE_LABELS = {
            "proximity_xy": "X-Y proximity probes (API 670, fluid_film)",
            "axial_accel": "Radial accelerometer (1 per bearing)",
            "accel_plus_velocity": "Accelerometer + Velometer (aero turbines, TRF/CRF)",
        }
        _MODE_KEYS = list(_MODE_LABELS.keys())
        _MODE_LABEL_LIST = list(_MODE_LABELS.values())

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("**Driver (driving)**")
            gen_driver_planes = st.number_input(
                "Driver planes",
                min_value=1, max_value=50, value=2, step=1,
                key=f"gen_driver_planes_{instance_id}",
                help="Up to 50 supports per section (driver+driven = 100 max).",
            )
            # Default modo según support_type ya configurado en la instancia
            _sup = (inst.support_type or "").lower()
            _driver_default_mode_idx = (
                _MODE_KEYS.index("accel_plus_velocity") if _sup == "rolling_element"
                else _MODE_KEYS.index("proximity_xy")
            )
            _gen_driver_mode_label = st.selectbox(
                "Driver instrumentation",
                options=_MODE_LABEL_LIST,
                index=_driver_default_mode_idx,
                key=f"gen_driver_mode_{instance_id}",
                help=(
                    "**X-Y proximity probes**: hydrodynamic journal bearings (Brush, large Siemens). "
                    "**Radial accelerometer**: simple rolling-element bearings (small motors, pumps). "
                    "**Accel + Velocity**: modern aero turbines (LM6000, TM2500) with "
                    "full instrumentation on TRF and CRF."
                ),
            )
            gen_driver_mode = _MODE_KEYS[_MODE_LABEL_LIST.index(_gen_driver_mode_label)]
            gen_driver_prefix = ""
            if gen_driver_mode in ("axial_accel", "accel_plus_velocity"):
                gen_driver_prefix = st.text_input(
                    "CSV Point prefix (accelerometers)",
                    value="acell",
                    key=f"gen_driver_prefix_{instance_id}",
                    help="Text that appears in the CSV Point. e.g. 'TRF', 'CRF', 'BRG', 'casing', 'acell'. "
                         "If your unit has CRF and TRF (LM6000), generate once with the 'CRF' prefix and "
                         "then manually edit the second plane so its pattern says 'TRF'.",
                )

        with gcol2:
            st.markdown("**Driven (driven)**")
            gen_driven_planes = st.number_input(
                "Driven planes",
                min_value=1, max_value=50, value=2, step=1,
                key=f"gen_driven_planes_{instance_id}",
                help="Up to 50 supports per section (driver+driven = 100 max).",
            )
            _gen_driven_mode_label = st.selectbox(
                "Driven instrumentation",
                options=_MODE_LABEL_LIST,
                index=_MODE_KEYS.index("proximity_xy"),
                key=f"gen_driven_mode_{instance_id}",
                help="Large generators and centrifugal compressors are typically = "
                     "X-Y proximity probes. Small pumps and motors = radial accelerometer.",
            )
            gen_driven_mode = _MODE_KEYS[_MODE_LABEL_LIST.index(_gen_driven_mode_label)]
            gen_driven_prefix = ""
            if gen_driven_mode in ("axial_accel", "accel_plus_velocity"):
                gen_driven_prefix = st.text_input(
                    "CSV Point prefix (driven accelerometers)",
                    value="acell",
                    key=f"gen_driven_prefix_{instance_id}",
                )

        # Keyphasor (1X phase reference)
        gen_include_keyphasor = st.checkbox(
            "Include keyphasor at coupling (1X reference for Polar/Bode)",
            value=False,
            key=f"gen_keyphasor_{instance_id}",
            help="Phase sensor typically mounted on the coupling side "
                 "between driver and driven. Generates 1 pulse per revolution and is used "
                 "as the angular reference for polar plots and Bode diagrams.",
        )

        # Confirmación si ya hay sensores configurados
        confirm_overwrite = True
        if len(inst.sensors) > 0:
            st.warning(
                f"⚠️ There are already **{len(inst.sensors)} sensors** configured. "
                "Generating a new one replaces ALL existing ones."
            )
            confirm_overwrite = st.checkbox(
                "I confirm overwriting the current map",
                key=f"confirm_overwrite_{instance_id}",
            )

        if st.button(
            "🪄 Generate map with this configuration",
            key=f"gen_sensor_map_{instance_id}",
            type="primary",
            disabled=(len(inst.sensors) > 0 and not confirm_overwrite),
        ):
            new_map = generate_standard_sensor_map(
                driver_planes=int(gen_driver_planes),
                driver_instrumentation=gen_driver_mode,
                driver_accel_prefix=gen_driver_prefix.strip() or "acell",
                driven_planes=int(gen_driven_planes),
                driven_instrumentation=gen_driven_mode,
                driven_accel_prefix=gen_driven_prefix.strip() or "acell",
                include_keyphasor=gen_include_keyphasor,
            )
            update_instance_header(instance_id, sensors=new_map)
            st.success(f"Map generated with {len(new_map)} sensors.")
            st.rerun()

    # Clear button (separate, always available)
    if st.button(
        "🗑️ Clear sensor map",
        key=f"clear_sensor_map_{instance_id}",
        disabled=len(inst.sensors) == 0,
    ):
        update_instance_header(instance_id, sensors=[])
        st.success("Sensor map cleared.")
        st.rerun()

    if not inst.sensors:
        st.info(
            "This asset has no configured sensors. "
            "Expand **🪄 Generate standard map** above to start with a "
            "layout configurable by driver and driven support type, "
            "or configure sensor by sensor manually with the editor below."
        )

    # ============================================================
    # Ciclo 22.2b — Resumen visual + validación de coherencia
    # ============================================================
    if inst.sensors:
        _EXPECTED_UNITS = {
            "proximity":     ["mil pp", "µm pp", "mm pp"],
            "velocity":      ["mm/s RMS", "mm/s peak", "in/s RMS", "in/s peak"],
            "velometer":     ["mm/s RMS", "mm/s peak", "in/s RMS", "in/s peak"],
            "accelerometer": ["g RMS", "g peak", "g pk", "m/s² RMS", "m/s² peak"],
            "keyphasor":     ["", "pulses/rev"],
        }
        _type_emojis = {
            "proximity":     "🎯",
            "velocity":      "📊",
            "velometer":     "📊",
            "accelerometer": "⚡",
            "keyphasor":     "🔑",
        }

        type_counts: Dict[str, int] = {}
        unit_inconsistencies: List[Dict[str, Any]] = []
        for _s in inst.sensors:
            _stype = (_s.get("sensor_type") or "").strip().lower()
            type_counts[_stype] = type_counts.get(_stype, 0) + 1
            _unit = (_s.get("unit_native") or "").strip()
            _expected = _EXPECTED_UNITS.get(_stype, [])
            # Comparar normalizado (lower) para que "g rms" == "g RMS"
            _expected_lower = [e.lower() for e in _expected]
            if _expected_lower and _unit.lower() not in _expected_lower:
                unit_inconsistencies.append({
                    "label": (_s.get("plane_label") or "").strip()
                             or f"plane {_s.get('plane', '?')}",
                    "type": _stype or "—",
                    "unit": _unit or "(empty)",
                    "expected": _expected,
                })

        # Chips de resumen
        chip_html_list = []
        for k, v in sorted(type_counts.items()):
            emoji = _type_emojis.get(k, "📡")
            chip_html_list.append(
                f'<span style="display:inline-block;padding:4px 11px;'
                f'background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;'
                f'border-radius:999px;font-size:12px;font-weight:600;'
                f'margin-right:6px;margin-bottom:4px;">'
                f'{emoji} {v} {k or "?"}</span>'
            )
        chip_html_list.append(
            f'<span style="display:inline-block;padding:4px 11px;'
            f'background:#f1f5f9;color:#334155;border:1px solid #e2e8f0;'
            f'border-radius:999px;font-size:12px;font-weight:600;'
            f'margin-right:6px;margin-bottom:4px;">'
            f'📊 Total: {len(inst.sensors)}</span>'
        )
        st.markdown(
            f'<div style="margin:10px 0;">{"".join(chip_html_list)}</div>',
            unsafe_allow_html=True,
        )

        # Validación de coherencia (el bug de C-200-C)
        if unit_inconsistencies:
            with st.expander(
                f"⚠️ {len(unit_inconsistencies)} sensor(s) with a unit inconsistent "
                f"with its type · click to review",
                expanded=True,
            ):
                st.markdown(
                    "These sensors have a mismatch between their configured **type** "
                    "and **unit**. It can cause Tabular List to show the CSV unit "
                    "instead of the configured one (historic bug C-200-C). "
                    "Fix them in the editor below by choosing a consistent unit."
                )
                for inc in unit_inconsistencies:
                    st.markdown(
                        f"- **`{inc['label']}`** · type **`{inc['type']}`** + "
                        f"unit **`{inc['unit']}`** → should be: "
                        f"`{', '.join(inc['expected'])}`"
                    )

    # Editor in-place del mapa
    df_sensors = pd.DataFrame(inst.sensors)
    if df_sensors.empty:
        # Skeleton de columnas para que el data_editor permita agregar filas
        df_sensors = pd.DataFrame(columns=[
            "plane", "plane_label", "side", "angle_deg", "direction",
            "sensor_type", "unit_native", "alarm", "danger",
            "csv_match_pattern", "notes",
        ])

    edited_df = st.data_editor(
        df_sensors,
        num_rows="dynamic",
        key=f"sensor_map_editor_{instance_id}",
        column_config={
            "plane": st.column_config.NumberColumn(
                "Plane", min_value=1, max_value=100, step=1, default=1,
                help="Sequential number from driver (1) to driven (last). "
                     "API 670. Up to 100 planes on machines with many supports.",
            ),
            "plane_label": st.column_config.TextColumn(
                "Plane label",
                help="e.g. 'DE driver', 'NDE driven'. Optional, for UI display.",
            ),
            "side": st.column_config.SelectboxColumn(
                "Side",
                options=["L", "R", "top", "bottom", "—"],
                default="L",
                help="Hemisphere viewed from the driver end. L=left, R=right.",
            ),
            "angle_deg": st.column_config.NumberColumn(
                "Angle (°)", min_value=-180.0, max_value=180.0, step=1.0, default=45.0,
                help="0° = top. Typical X-Y API 670 probes: ±45°.",
            ),
            "direction": st.column_config.SelectboxColumn(
                "Dir",
                options=["X", "Y", "radial", "axial"],
                default="Y",
            ),
            "sensor_type": st.column_config.SelectboxColumn(
                "Type",
                options=["proximity", "velocity", "accelerometer", "keyphasor"],
                default="proximity",
                help=(
                    "proximity → Displacement (mil pp / µm pp). "
                    "velocity → Velocity (mm/s RMS / in/s peak). "
                    "accelerometer → Acceleration (g RMS / m/s² RMS). "
                    "keyphasor → 1X phase reference (pulses/rev), "
                    "typically at the coupling."
                ),
            ),
            "unit_native": st.column_config.SelectboxColumn(
                "Unit",
                options=[
                    # Desplazamiento
                    "mil pp",
                    "µm pp",
                    "mm pp",
                    # Velocidad
                    "mm/s RMS",
                    "mm/s peak",
                    "in/s RMS",
                    "in/s peak",
                    # Aceleración
                    "g RMS",
                    "g peak",
                    "m/s² RMS",
                    "m/s² peak",
                ],
                default="mil pp",
                help=(
                    "Native sensor unit according to its type. The options are "
                    "grouped: the first 3 are displacement (proximity), "
                    "the next 4 velocity (velocity), the last 4 "
                    "acceleration (accelerometer). Choose the one your DCS / OEM uses."
                ),
            ),
            "alarm": st.column_config.NumberColumn(
                "Alarm", min_value=0.0, step=0.1, format="%.3f", default=4.0,
                help="Alert setpoint in the native unit. When the amplitude exceeds this value, status = ATTENTION.",
            ),
            "danger": st.column_config.NumberColumn(
                "Danger", min_value=0.0, step=0.1, format="%.3f", default=6.0,
                help="Trip setpoint in the native unit. When the amplitude exceeds this value, status = ACTION REQUIRED / CRITICAL.",
            ),
            "csv_match_pattern": st.column_config.TextColumn(
                "CSV Point text",
                help=(
                    "Text that appears in the Point field of the loaded CSV. "
                    "Three valid formats:\n"
                    "  • Simple substring: 'VE5807' matches 'VE5807 (Y)' or 'VE5807-Y'.\n"
                    "  • Comma list: 'VE5807 (Y), VE5807-Y, 5807_Y' matches any of the variants.\n"
                    "  • Glob: '*5807*y*' uses wildcards for advanced cases.\n"
                    "The match is case-insensitive."
                ),
            ),
            "notes": st.column_config.TextColumn("Notes"),
        },
        width="stretch",
    )

    if st.button(
        "💾 Save sensor map",
        key=f"save_sensor_map_{instance_id}",
        type="primary",
        width="stretch",
    ):
        # Convertir DataFrame a lista de dicts limpios. Ciclo 15.2 —
        # preservar x_pct/y_pct (coordenadas click-to-place) que no
        # estan en el data_editor pero viven en el sensor original.
        # Mapeamos por (plane, direction, side, sensor_type) que en
        # conjunto identifican univocamente al sensor.
        existing_coords: Dict[tuple, tuple] = {}
        for _s in (inst.sensors or []):
            try:
                k = (
                    int(_s.get("plane", 0) or 0),
                    str(_s.get("direction", "") or ""),
                    str(_s.get("side", "") or ""),
                    str(_s.get("sensor_type", "") or ""),
                )
                xp = _s.get("x_pct")
                yp = _s.get("y_pct")
                if xp is not None or yp is not None:
                    existing_coords[k] = (xp, yp)
            except Exception:
                continue

        new_sensors = []
        for _, row in edited_df.iterrows():
            try:
                sensor_dict = {
                    "plane": int(row.get("plane", 1) or 1),
                    "plane_label": str(row.get("plane_label", "") or ""),
                    "side": str(row.get("side", "L") or "L"),
                    "angle_deg": float(row.get("angle_deg", 45.0) or 45.0),
                    "direction": str(row.get("direction", "Y") or "Y"),
                    "sensor_type": str(row.get("sensor_type", "proximity") or "proximity"),
                    "unit_native": str(row.get("unit_native", "mil pp") or "mil pp"),
                    "alarm": float(row.get("alarm", 0.0) or 0.0),
                    "danger": float(row.get("danger", 0.0) or 0.0),
                    "csv_match_pattern": str(row.get("csv_match_pattern", "") or ""),
                    "notes": str(row.get("notes", "") or ""),
                }
                # Re-asociar coordenadas previas si las habia
                k = (
                    sensor_dict["plane"],
                    sensor_dict["direction"],
                    sensor_dict["side"],
                    sensor_dict["sensor_type"],
                )
                if k in existing_coords:
                    sensor_dict["x_pct"] = existing_coords[k][0]
                    sensor_dict["y_pct"] = existing_coords[k][1]
                else:
                    sensor_dict["x_pct"] = None
                    sensor_dict["y_pct"] = None
                new_sensors.append(sensor_dict)
            except Exception:
                continue
        update_instance_header(instance_id, sensors=new_sensors)
        st.success(f"Map saved with {len(new_sensors)} sensors.")
        st.rerun()

    # Current map preview with formatted labels + visual diagram
    if inst.sensors:
        with st.expander(f"Current map preview ({len(inst.sensors)} sensors)", expanded=False):
            preview_lines = []
            for s in inst.sensors:
                lbl = sensor_label(s)
                ploc = (
                    f"plane {s.get('plane', '?')} ({s.get('plane_label', '')}) · "
                    f"{s.get('side', '')} {s.get('angle_deg', 0):.0f}° · "
                    f"{s.get('direction', '')}"
                )
                tinfo = (
                    f"{s.get('sensor_type', '')} ({s.get('unit_native', '')}) · "
                    f"A={s.get('alarm', 0):.2f} D={s.get('danger', 0):.2f}"
                )
                pat = s.get('csv_match_pattern', '') or '(no pattern)'
                preview_lines.append(f"- **{lbl}** · {ploc} · {tinfo} · match=`{pat}`")
            st.markdown("\n".join(preview_lines))

        # ============================================================
        # Ciclo 16.1 — Wizard auto-pattern desde CSVs cargados
        # ------------------------------------------------------------
        # Detecta sensores sin match definitivo y propone patterns
        # basados en los Point names de los CSVs en sesion. El usuario
        # acepta/rechaza por sensor y aplica en bulk.
        # ============================================================
        st.markdown("---")
        st.markdown("#### 🪄 Suggest patterns from loaded CSVs")
        st.caption(
            "If you uploaded CSVs in Load Data and some sensors appear without "
            "a match against the Sensor Map, this wizard analyzes the CSV Point "
            "names and proposes a concrete `csv_match_pattern` "
            "for each sensor without a pattern. Accept or reject per row."
        )

        try:
            from core.sensor_map import (
                detect_definitive_matches,
                suggest_pattern_for_sensor,
                sensor_label as _slbl,
            )

            # Recolectar metadata de los signals en sesion
            _wiz_signals_meta = []
            for _signame, _sigobj in (st.session_state.get("signals", {}) or {}).items():
                try:
                    _md = (
                        getattr(_sigobj, "metadata", None)
                        or (_sigobj.get("metadata") if isinstance(_sigobj, dict) else {})
                        or {}
                    )
                    _wiz_signals_meta.append({
                        "File": _signame,
                        "Point": str(_md.get("Point", "") or ""),
                        "Variable": str(_md.get("Variable", "") or ""),
                        "Y-Axis Unit": str(
                            _md.get("Y-Axis Unit", "")
                            or _md.get("Unit", "")
                            or ""
                        ),
                    })
                except Exception:
                    continue

            if not _wiz_signals_meta:
                st.info(
                    "No signals loaded in this session. Go to **Load Data** "
                    "to upload the CSVs and come back here — the wizard analyzes "
                    "the Point names and proposes patterns for your sensors."
                )
            else:
                # Detectar matches definitivos y proponer para los que faltan
                _definitive = detect_definitive_matches(inst.sensors, _wiz_signals_meta)
                _claimed = set(_definitive.values())

                _wiz_rows = []
                for _s in inst.sensors:
                    _lbl = _slbl(_s)
                    if _lbl in _definitive:
                        continue  # ya matched definitivamente
                    # Solo proponer para los SIN pattern; si ya tiene pattern
                    # pero no matchea, el usuario lo escribió mal — lo
                    # mostramos también.
                    _sug = suggest_pattern_for_sensor(
                        _s, _wiz_signals_meta, already_claimed_signals=_claimed
                    )
                    if _sug is None:
                        continue
                    _wiz_rows.append({
                        "_sensor_label": _lbl,
                        "_sensor_obj": _s,
                        "_suggestion": _sug,
                    })

                if not _wiz_rows:
                    if len(_definitive) == sum(
                        1 for _s in inst.sensors
                        if str(_s.get("sensor_type", "")).lower() != "keyphasor"
                    ):
                        st.success(
                            "✓ All vibration sensors have a "
                            "pattern that matches correctly with a loaded "
                            "CSV. There are no pending suggestions."
                        )
                    else:
                        st.info(
                            "No additional suggestions. Sensors without "
                            "a match have no compatible CSVs loaded."
                        )
                else:
                    st.markdown(
                        f"**{len(_wiz_rows)}** "
                        f"{'sensor without a match' if len(_wiz_rows) == 1 else 'sensors without a definitive match'}"
                        f" — proposals:"
                    )

                    # Inicializar checkboxes en session_state
                    _wiz_state_key = f"ml_wiz_apply_{instance_id}"
                    if _wiz_state_key not in st.session_state:
                        st.session_state[_wiz_state_key] = {
                            row["_sensor_label"]: True for row in _wiz_rows
                        }

                    # Render: una fila por sensor con checkbox + info
                    for row in _wiz_rows:
                        _lbl = row["_sensor_label"]
                        _sug = row["_suggestion"]
                        _sensor = row["_sensor_obj"]
                        _cur_pattern = (_sensor.get("csv_match_pattern") or "").strip()
                        _conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                            _sug["confidence"], "🟡"
                        )

                        cols_w = st.columns([0.5, 4, 4])
                        with cols_w[0]:
                            checked = st.checkbox(
                                "Apply",
                                value=st.session_state[_wiz_state_key].get(_lbl, True),
                                key=f"wiz_apply_cb_{instance_id}_{_lbl}",
                                label_visibility="collapsed",
                            )
                            st.session_state[_wiz_state_key][_lbl] = checked
                        with cols_w[1]:
                            st.markdown(
                                f"**{_lbl}** · {_sensor.get('plane_label', '')} "
                                f"· {_sensor.get('sensor_type', '')} "
                                f"`{_sensor.get('direction', '')}`"
                            )
                            st.caption(
                                f"Current pattern: `{_cur_pattern or '(empty)'}`"
                            )
                        with cols_w[2]:
                            st.markdown(
                                f"{_conf_emoji} → proposes `{_sug['proposed_pattern']}` "
                                f"(matches **{_sug['candidate_point']}**)"
                            )
                            st.caption(
                                f"From `{_sug['candidate_signal']}` · "
                                f"{_sug['reason']}"
                            )

                    if st.button(
                        f"✨ Apply {sum(st.session_state[_wiz_state_key].values())} selected pattern(s)",
                        key=f"wiz_apply_btn_{instance_id}",
                        type="primary",
                        width="stretch",
                    ):
                        _new_sensors = list(inst.sensors)
                        # Map por label para update inplace
                        _by_label = {_slbl(_s): i for i, _s in enumerate(_new_sensors)}
                        _applied = 0
                        for row in _wiz_rows:
                            _lbl = row["_sensor_label"]
                            if not st.session_state[_wiz_state_key].get(_lbl):
                                continue
                            if _lbl not in _by_label:
                                continue
                            _idx = _by_label[_lbl]
                            _s_copy = dict(_new_sensors[_idx])
                            _s_copy["csv_match_pattern"] = row["_suggestion"]["proposed_pattern"]
                            _new_sensors[_idx] = _s_copy
                            _applied += 1
                        if _applied > 0:
                            update_instance_header(instance_id, sensors=_new_sensors)
                            st.success(
                                f"✓ {_applied} pattern(s) applied to the Sensor Map. "
                                f"Go back to Tabular List or Reports and the sensors "
                                f"will appear with their Overall values."
                            )
                            # limpiar el state para que no quede pegado
                            st.session_state.pop(_wiz_state_key, None)
                            st.rerun()
                        else:
                            st.info("Select at least one row to apply.")
        except Exception as _wiz_e:
            st.caption(f"_(wizard unavailable: {_wiz_e})_")

        # Ciclo 14c.2 — diagrama visual del mapa de sensores
        st.markdown("#### 🎯 Visual map diagram")
        st.caption(
            "Side view of the train with numbered bearings (API 670 / "
            "ISO 20816-1 driver→driven convention) and a polar view per plane with probes at their "
            "physical angles. R/L viewed from the driver end, 0° at top."
        )

        # Ciclo 23.13 — Si la instancia tiene driver_icon_key + driven_icon_key
        # configurados (wizard >= v3.31.14), preferimos el SVG vectorial de
        # core.asset_library en lugar del diagrama legacy de rectángulos.
        # Mismo render que Live Monitoring + Wizard editor, así el especialista
        # ve el mismo activo en las 3 pantallas.
        _used_library = False
        _drv_icon_key = getattr(inst, "driver_icon_key", "") or ""
        _drvn_icon_key = getattr(inst, "driven_icon_key", "") or ""
        if _drv_icon_key and _drvn_icon_key:
            try:
                from core.asset_library.composer import compose_train
                from core.sensor_map import sensor_label as _slbl_lib
                _drv_lbl_lib = " ".join(p for p in [
                    inst.driver_manufacturer, inst.driver_model,
                ] if p) or "Driver"
                _dvn_lbl_lib = " ".join(p for p in [
                    inst.driven_manufacturer, inst.driven_model,
                ] if p) or "Driven"
                _s_for_svg = []
                from core.sensor_map import gearbox_overlay_anchor as _gbx_ov
                for s in (inst.sensors or []):
                    _side = (s.get("icon_side") or "").strip()
                    _anchor = (s.get("icon_anchor") or "").strip()
                    if not _side or not _anchor:
                        _gov = _gbx_ov(s)
                        if _gov:
                            _side, _anchor = _gov
                        else:
                            continue
                    try:
                        _lbl = _slbl_lib(s)
                    except Exception:
                        _lbl = s.get("plane_label", "?")
                    # Display label sin underscore para SVG (Ciclo 23.18)
                    _display_lbl = _lbl.replace("_", "")
                    _s_for_svg.append({
                        "label": _display_lbl,
                        "side": _side,
                        "anchor": _anchor,
                        "status": "Sin Norma",  # sin readings live en este contexto
                        "value": "",
                        "unit": "",
                        "title": f"{_lbl} · {s.get('sensor_type','')}",
                    })
                from core.instance_state import detect_gearbox_kwargs as _gbx_kw
                _svg_lib = compose_train(
                    driver_key=_drv_icon_key,
                    driven_key=_drvn_icon_key,
                    driver_label=_drv_lbl_lib,
                    driven_label=_dvn_lbl_lib,
                    coupling=getattr(inst, "coupling_class", "") or "flexible",
                    sensors_with_status=_s_for_svg,
                    **_gbx_kw(inst),
                )
                st.markdown(
                    f'<div style="background:#ffffff;border:1px solid #e2e8f0;'
                    f'border-radius:10px;padding:14px;">{_svg_lib}</div>',
                    unsafe_allow_html=True,
                )
                _n_total = len(inst.sensors or [])
                _n_mapped = len(_s_for_svg)
                if _n_mapped < _n_total:
                    st.caption(
                        f"📍 **{_n_mapped} of {_n_total}** sensors assigned to an icon anchor. "
                        f"The remaining {_n_total - _n_mapped} do not appear — assign them side/anchor "
                        f"in the wizard (Step 5 · Visual editor)."
                    )
                else:
                    st.caption(
                        f"📍 {_n_mapped} sensors · all assigned to their physical bearing."
                    )
                _used_library = True
            except Exception as _lib_e:
                import logging as _lg_lib
                _lg_lib.getLogger(__name__).warning(
                    "Asset library no pudo rendir, fallback a diagrama legacy: %s",
                    _lib_e,
                )

        if not _used_library:
            try:
                from core.sensor_diagram import render_sensor_map_diagram, _infer_machine_kind
                _train_lbl = compose_train_description(inst) or ""
                _drv_lbl = " ".join(p for p in [inst.driver_manufacturer, inst.driver_model] if p) or "Driver"
                _dvn_lbl = " ".join(p for p in [inst.driven_manufacturer, inst.driven_model] if p) or "Driven"
                # Ciclo 17.5.11 — inferimos kind del driver y driven a
                # partir de su label combinado con el asset_class. Esto
                # hace que el silhouette dibuje motor en vez de turbina,
                # compresor recip en vez de generador, etc.
                _drv_kind = (
                    _infer_machine_kind(_drv_lbl)
                    or _infer_machine_kind(inst.asset_class)
                    or "turbine"
                )
                _dvn_kind = (
                    _infer_machine_kind(_dvn_lbl)
                    or _infer_machine_kind(inst.asset_class)
                    or "generator"
                )

                # Ciclo 21.4 v3 — Si es compresor reciprocante, usar el
                # schematic boxer nuevo (cilindros opuestos, acople con
                # flanges). El render genérico dibujaba todo en línea.
                _is_recip = (
                    _dvn_kind == "recip_compressor"
                    or any("cilindro" in (s.get("plane_label", "") or "").lower()
                           for s in (inst.sensors or []))
                )
                _diag_png = None
                if _is_recip:
                    try:
                        from core.recip_schematic import generate_recip_png
                        import re as _re_recip
                        _cyl_nums = set()
                        _motor_planes_set = set()
                        for s in (inst.sensors or []):
                            lbl = (s.get("plane_label", "") or "").lower()
                            if "cilindro" in lbl and "rod drop" not in lbl:
                                _m = _re_recip.search(r"cilindro\s*(\d+)", lbl)
                                if _m:
                                    _cyl_nums.add(int(_m.group(1)))
                            if "motor" in lbl:
                                _motor_planes_set.add(s.get("plane", 0))
                        _n_cyl = max(_cyl_nums) if _cyl_nums else 4
                        _n_motor = max(len(_motor_planes_set), 2)
                        _diag_png = generate_recip_png(
                            n_cylinders=_n_cyl,
                            n_motor_planes=_n_motor,
                            motor_label=_drv_lbl,
                            compressor_label=_dvn_lbl,
                        )
                    except Exception as _re_err:
                        import logging as _lg
                        _lg.getLogger(__name__).warning(
                            "Fallo recip_schematic, fallback a render genérico: %s",
                            _re_err,
                        )

                if not _diag_png:
                    _diag_png = render_sensor_map_diagram(
                        inst.sensors,
                        train_label=_train_lbl,
                        driver_label=_drv_lbl,
                        driven_label=_dvn_lbl,
                        driver_kind=_drv_kind,
                        driven_kind=_dvn_kind,
                    )
                if _diag_png:
                    st.image(_diag_png, use_container_width=True)
                else:
                    st.warning(
                        "Could not render the diagram. "
                        "Check that matplotlib is available in the environment."
                    )
            except Exception as e:
                st.warning(f"Error rendering diagram: {e}")

        # ============================================================
        # Ciclo 15.2 — Click-to-place sobre el schematic_png real
        # ------------------------------------------------------------
        # Permite asignar coordenadas (x_pct, y_pct) a cada sensor del
        # mapa haciendo clic en la imagen del activo. Una vez
        # configurado, el Resumen Ejecutivo del PDF y la pagina
        # Machine Map renderizan los markers de severidad + valores
        # Overall sobre la foto/dibujo real en lugar del esquematico
        # generico turbomachinery.
        #
        # Si no hay schematic_png cargado para esta instancia, se omite
        # la seccion. Si no hay streamlit_image_coordinates instalado,
        # se ofrece un fallback de inputs numericos (defensivo).
        # ============================================================
        if inst.schematic_png:
            st.markdown("---")
            st.markdown("#### 📍 Position sensors on the schematic")
            st.caption(
                "Place each bearing on the asset photo/drawing. Once "
                "positioned, the reports show the Overall vibration values "
                "colored by severity on your real schematic, "
                "not on the generic turbomachinery one."
            )

            try:
                _sch_bytes = get_instance_document_bytes(
                    inst.instance_id, inst.schematic_png
                )
            except Exception:
                _sch_bytes = None

            if not _sch_bytes:
                st.info("Could not load the asset schematic.")
            else:
                # Inventario de planos del Sensor Map (un boton por plano —
                # no por sensor — porque los sensores que comparten plano
                # comparten posicion fisica).
                planes_map: Dict[int, Dict[str, Any]] = {}
                for _s in inst.sensors:
                    p = int(_s.get("plane", 0) or 0)
                    if p <= 0 or str(_s.get("sensor_type", "")).lower() == "keyphasor":
                        # Keyphasor lo manejamos aparte (suele ir en coupling)
                        if str(_s.get("sensor_type", "")).lower() == "keyphasor":
                            planes_map.setdefault("KP", {
                                "is_kp": True,
                                "plane_label": "Keyphasor",
                                "x_pct": _s.get("x_pct"),
                                "y_pct": _s.get("y_pct"),
                            })
                        continue
                    if p not in planes_map:
                        planes_map[p] = {
                            "is_kp": False,
                            "plane_label": _s.get("plane_label", "") or f"Plane {p}",
                            "x_pct": _s.get("x_pct"),
                            "y_pct": _s.get("y_pct"),
                        }

                if not planes_map:
                    st.info(
                        "Configure the map's sensors above first "
                        "so you can position them on the schematic."
                    )
                else:
                    # UI de seleccion: que plano vamos a posicionar.
                    # Sort: planos numericos primero, KP al final.
                    plane_keys_sorted = sorted(
                        planes_map.keys(),
                        key=lambda k: (1, 999) if k == "KP" else (0, k),
                    )
                    plane_options = []
                    for k in plane_keys_sorted:
                        info = planes_map[k]
                        coord_status = (
                            f" · ✓ positioned ({info['x_pct']:.1f}%, {info['y_pct']:.1f}%)"
                            if info["x_pct"] is not None and info["y_pct"] is not None
                            else " · ✗ not positioned"
                        )
                        if k == "KP":
                            plane_options.append(("KP", f"⭐ Keyphasor{coord_status}"))
                        else:
                            plane_options.append((k, f"Plane {k} · {info['plane_label']}{coord_status}"))

                    # Mantener seleccion entre reruns mediante session_state.
                    # Usamos la KEY del plano (no la label) porque la label
                    # cambia cuando se guarda una posicion (pasa de "sin
                    # posicionar" a "posicionado") y eso hacia que Streamlit
                    # no pudiera matchear el value previo y caia al indice 0
                    # (= Keyphasor con sort viejo).
                    _ctp_state_key = f"ctp_selected_plane_key_{instance_id}"
                    keys_in_order = [k for k, _ in plane_options]
                    if _ctp_state_key not in st.session_state or \
                       st.session_state[_ctp_state_key] not in keys_in_order:
                        st.session_state[_ctp_state_key] = keys_in_order[0]
                    default_idx = keys_in_order.index(st.session_state[_ctp_state_key])

                    selected_label = st.selectbox(
                        "Plane to position (click on the image below)",
                        [lbl for _, lbl in plane_options],
                        index=default_idx,
                        key=f"ctp_plane_select_widget_{instance_id}",
                    )
                    sel_label_to_key = {lbl: k for k, lbl in plane_options}
                    selected_plane = sel_label_to_key[selected_label]
                    # Persistir la key seleccionada para sobrevivir reruns
                    st.session_state[_ctp_state_key] = selected_plane

                    # Render con streamlit_image_coordinates
                    captured_xy: Optional[tuple] = None
                    img_w_px: Optional[int] = None
                    img_h_px: Optional[int] = None
                    try:
                        from streamlit_image_coordinates import streamlit_image_coordinates
                        from PIL import Image as PILImage
                        from io import BytesIO as _BIO

                        # Renderizar overlay con TODOS los sensores ya
                        # posicionados (modo configuracion — sin severity)
                        from core.sensor_diagram import render_on_schematic
                        _preview_png = render_on_schematic(
                            _sch_bytes, inst.sensors,
                            severity_by_label=None,
                            overall_by_label=None,
                            unit_by_label=None,
                            show_values=False,
                            show_labels=True,
                        ) or _sch_bytes

                        # streamlit_image_coordinates exige un PIL.Image o
                        # path/numpy array — no acepta bytes crudos. Lo
                        # decodificamos antes de pasarlo.
                        _preview_pil = PILImage.open(_BIO(_preview_png))
                        img_w_px, img_h_px = _preview_pil.size

                        coords = streamlit_image_coordinates(
                            _preview_pil,
                            key=f"ctp_canvas_{instance_id}",
                            use_column_width=True,
                        )
                        if coords is not None:
                            # streamlit_image_coordinates devuelve coords en
                            # pixeles relativos al tamaño REAL de la imagen
                            # (no el display) desde v0.1.6+.
                            try:
                                cx = float(coords["x"])
                                cy = float(coords["y"])
                                xp_pct = (cx / img_w_px) * 100.0
                                yp_pct = (cy / img_h_px) * 100.0
                                xp_pct = max(0.0, min(100.0, xp_pct))
                                yp_pct = max(0.0, min(100.0, yp_pct))
                                captured_xy = (xp_pct, yp_pct)
                            except Exception:
                                captured_xy = None
                    except ImportError:
                        st.warning(
                            "The `streamlit-image-coordinates` package is not "
                            "installed. Use the numeric fallback below."
                        )

                    # Fallback / edicion manual + confirmacion
                    cur_xp = planes_map[selected_plane].get("x_pct")
                    cur_yp = planes_map[selected_plane].get("y_pct")

                    cols_xy = st.columns([1, 1, 1])
                    new_xp = cols_xy[0].number_input(
                        "X (%)", min_value=0.0, max_value=100.0,
                        value=float(captured_xy[0]) if captured_xy else (
                            float(cur_xp) if cur_xp is not None else 50.0
                        ),
                        step=0.5, format="%.1f",
                        key=f"ctp_x_{instance_id}_{selected_plane}",
                    )
                    new_yp = cols_xy[1].number_input(
                        "Y (%)", min_value=0.0, max_value=100.0,
                        value=float(captured_xy[1]) if captured_xy else (
                            float(cur_yp) if cur_yp is not None else 50.0
                        ),
                        step=0.5, format="%.1f",
                        key=f"ctp_y_{instance_id}_{selected_plane}",
                    )
                    if cols_xy[2].button(
                        "💾 Save this plane's position",
                        key=f"ctp_save_{instance_id}_{selected_plane}",
                        type="primary",
                        width="stretch",
                    ):
                        # Aplicar coords a TODOS los sensores que comparten el plano
                        updated_sensors = []
                        for _s in inst.sensors:
                            _s2 = dict(_s)
                            if selected_plane == "KP":
                                if str(_s.get("sensor_type", "")).lower() == "keyphasor":
                                    _s2["x_pct"] = float(new_xp)
                                    _s2["y_pct"] = float(new_yp)
                            else:
                                if int(_s.get("plane", 0) or 0) == int(selected_plane):
                                    _s2["x_pct"] = float(new_xp)
                                    _s2["y_pct"] = float(new_yp)
                            updated_sensors.append(_s2)
                        update_instance_header(instance_id, sensors=updated_sensors)
                        st.success(
                            f"Position saved for "
                            f"{'Keyphasor' if selected_plane == 'KP' else f'Plane {selected_plane}'}"
                            f" → ({new_xp:.1f}%, {new_yp:.1f}%)"
                        )
                        st.rerun()

                    # Boton para limpiar todas las coords (rehacer desde cero)
                    if any(
                        v.get("x_pct") is not None for v in planes_map.values()
                    ):
                        if st.button(
                            "🧹 Clear all positions",
                            key=f"ctp_clear_{instance_id}",
                        ):
                            cleared = []
                            for _s in inst.sensors:
                                _s2 = dict(_s)
                                _s2["x_pct"] = None
                                _s2["y_pct"] = None
                                cleared.append(_s2)
                            update_instance_header(instance_id, sensors=cleared)
                            st.info("Coordinates cleared.")
                            st.rerun()


def render_documents_section(instance_id: str) -> None:
    """Lista de documentos cargados de la instancia + acciones."""
    inst = get_instance(instance_id)
    if inst is None:
        return
    docs = list(inst.documents)
    docs.sort(key=lambda d: d.get("uploaded_at", ""), reverse=True)

    st.markdown("### Asset documents")
    if not docs:
        st.info(
            "This instance has no documents yet. Upload OEM manuals, "
            "maintenance reports, certificates or specifications "
            "from the 'Upload new document' section below."
        )
        return

    # ----- Ciclo 22.2d — filtros + grid de cards -----
    type_emoji = {
        "manual_oem":     "📕",
        "datasheet":      "📊",
        "drawing":        "📐",
        "certificate":    "🏆",
        "report":         "📄",
        "photo":          "📷",
        "schematic":      "🗺️",
        "maintenance":    "🛠️",
        "other":          "📎",
    }

    def _icon_for_doc(d: Dict[str, Any]) -> str:
        dtype = d.get("document_type", "other")
        if dtype in type_emoji:
            return type_emoji[dtype]
        ext = (d.get("filename", "").rsplit(".", 1)[-1] or "").lower()
        if ext in ("png", "jpg", "jpeg", "gif", "webp"):
            return "🖼️"
        if ext == "pdf":
            return "📕"
        if ext in ("xlsx", "xls", "csv"):
            return "📊"
        if ext in ("dwg", "dxf"):
            return "📐"
        return "📎"

    # Filtros
    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        type_options = ["Todos"] + sorted({
            DOCUMENT_TYPES.get(d.get("document_type", "other"),
                              d.get("document_type", "—"))
            for d in docs
        })
        sel_type = st.selectbox(
            "Filter by type",
            options=type_options,
            key=f"docs_type_filter_{instance_id}",
        )
    with fcol2:
        search_q = st.text_input(
            "Search by title / description / tag",
            key=f"docs_search_{instance_id}",
            placeholder="e.g. rebabbiting, manual, october",
        ).strip().lower()

    # Aplicar filtros
    filtered = []
    for d in docs:
        if sel_type != "Todos":
            d_type_label = DOCUMENT_TYPES.get(
                d.get("document_type", "other"),
                d.get("document_type", "—"),
            )
            if d_type_label != sel_type:
                continue
        if search_q:
            haystack = " ".join([
                d.get("title", ""),
                d.get("filename", ""),
                d.get("description", ""),
                " ".join(d.get("tags", []) or []),
            ]).lower()
            if search_q not in haystack:
                continue
        filtered.append(d)

    st.caption(
        f"Showing {len(filtered)} of {len(docs)} document(s)."
    )

    if not filtered:
        st.info("No document matches the filter.")
        return

    # Grid de cards (3 columnas)
    cards_per_row = 3
    for i in range(0, len(filtered), cards_per_row):
        cols = st.columns(cards_per_row, gap="medium")
        for j, col in enumerate(cols):
            if i + j >= len(filtered):
                continue
            d = filtered[i + j]
            with col:
                _render_doc_card(instance_id, d, _icon_for_doc(d))


def _render_doc_card(instance_id: str, d: Dict[str, Any], icon: str) -> None:
    """Card individual de documento (Ciclo 22.2d)."""
    import textwrap as _tw
    title = d.get("title") or d.get("filename") or "—"
    filename = d.get("filename", "—")
    size = _bytes_to_human(int(d.get("size_bytes", 0)))
    uploaded = _format_date(d.get("uploaded_at", ""))
    description = d.get("description") or ""
    tags = d.get("tags", []) or []

    tags_html = "".join(
        f'<span style="display:inline-block;padding:1px 7px;background:#eef2ff;'
        f'color:#3730a3;border-radius:999px;font-size:10px;margin:1px 2px;">{t}</span>'
        for t in tags
    )

    card_html = _tw.dedent(f"""\
    <div style="border:1px solid #e5e7eb;border-radius:10px;padding:12px;background:white;margin-bottom:6px;">
      <div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;">
        <div style="font-size:28px;line-height:1;flex-shrink:0;">{icon}</div>
        <div style="min-width:0;flex:1;">
          <div style="font-weight:700;color:#0f172a;font-size:13px;line-height:1.25;
                     overflow:hidden;text-overflow:ellipsis;display:-webkit-box;
                     -webkit-line-clamp:2;-webkit-box-orient:vertical;">{title}</div>
          <div style="color:#64748b;font-size:11px;margin-top:2px;">{filename}</div>
        </div>
      </div>
      <div style="font-size:11px;color:#475569;margin-bottom:6px;min-height:30px;
                 overflow:hidden;text-overflow:ellipsis;display:-webkit-box;
                 -webkit-line-clamp:2;-webkit-box-orient:vertical;">
        {description or '<span style="color:#94a3b8;">_(no description)_</span>'}
      </div>
      <div style="margin-bottom:6px;">{tags_html}</div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#94a3b8;
                 border-top:1px solid #f1f5f9;padding-top:6px;">
        <span>📦 {size}</span>
        <span>📅 {uploaded}</span>
      </div>
    </div>
    """)
    st.markdown(card_html, unsafe_allow_html=True)

    # Acciones
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        file_bytes = get_instance_document_bytes(instance_id, d["id"])
        if file_bytes is not None:
            st.download_button(
                "📥 Download",
                data=file_bytes,
                file_name=d.get("filename", "document"),
                key=f"dl_v2_{instance_id}_{d['id']}",
                use_container_width=True,
            )
        else:
            st.caption("⚠️ unavailable")
    with bcol2:
        if st.button("🗑️ Delete",
                     key=f"del_v2_{instance_id}_{d['id']}",
                     use_container_width=True):
            remove_instance_document(instance_id, d["id"])
            st.success(f"'{d.get('title')}' deleted.")
            st.rerun()


def render_upload_section(instance_id: str) -> None:
    """Formulario de upload de nuevo documento a la instancia activa."""
    st.markdown("### Upload new document")
    st.caption(
        f"Documents uploaded here are associated exclusively with "
        f"the active instance (`{instance_id}`). They are not shared with other "
        f"instances of the same profile."
    )

    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "File (PDF, image, document, datasheet)",
            type=None,
            accept_multiple_files=False,
        )

        col1, col2 = st.columns(2)
        with col1:
            doc_title = st.text_input(
                "Descriptive title",
                placeholder="e.g. Bearing rebabbiting report (Wersin, Oct 2018)",
            )
            doc_type_label_to_key = {label: key for key, label in DOCUMENT_TYPES.items()}
            doc_type_label = st.selectbox("Document type", list(DOCUMENT_TYPES.values()))
            doc_type = doc_type_label_to_key[doc_type_label]
        with col2:
            doc_description = st.text_area(
                "Description / context",
                placeholder="Brief summary of the document content or context.",
                height=100,
            )
            doc_tags_str = st.text_input(
                "Tags (comma-separated)",
                placeholder="bearing, rebabbiting, wersin, 2018",
            )

        submitted = st.form_submit_button("Upload document", width="stretch")

        if submitted:
            if uploaded_file is None:
                st.error("Select a file before uploading.")
                return
            tags = [t.strip() for t in doc_tags_str.split(",") if t.strip()]
            doc_id = add_uploaded_file_to_instance(
                instance_id,
                uploaded_file,
                title=doc_title or uploaded_file.name,
                document_type=doc_type,
                description=doc_description,
                tags=tags,
            )
            if doc_id:
                st.success(
                    f"Document '{uploaded_file.name}' uploaded to instance "
                    f"'{instance_id}'. ID: `{doc_id}`"
                )
                st.rerun()
            else:
                st.error("The document could not be uploaded.")


_CATEGORY_ICONS = {
    "Identificación":           "🆔",
    "Cojinete - geometría":     "🔩",
    "Cojinete - rodamiento":    "⚙️",
    "Cojinete - cargas":        "⚖️",
    "Cojinete - tolerancias":   "📐",
    "Operación":                "⚡",
    "Rotor":                    "🌀",
    "Acople":                   "🔗",
    "Lubricación":              "💧",
    "Otros":                    "📋",
}


def render_captured_parameters_section(instance_id: str) -> None:
    """Form de parámetros estructurados con auto-cálculos en vivo (Ciclo 22.2c)."""
    inst = get_instance(instance_id)
    if inst is None:
        return

    st.markdown("### Asset technical parameters")
    st.caption(
        "Capture the asset's physical parameters extracted from OEM "
        "manuals or field measurements. These values feed the analysis "
        "modules (Shaft Centerline, Polar, Bode) when they require "
        "bearing- or rotor-specific data."
    )

    current_values = dict(inst.captured_parameters)

    # ----- Ciclo 22.2c — Indicador de completitud por categoría -----
    by_cat_count: Dict[str, Dict[str, int]] = {}
    for field_key, field_def in CAPTURED_PARAMETER_FIELDS.items():
        cat = field_def.get("category", "Otros")
        cell = by_cat_count.setdefault(cat, {"filled": 0, "total": 0})
        cell["total"] += 1
        v = current_values.get(field_key)
        is_filled = (
            v is not None
            and (not isinstance(v, str) or v.strip() != "")
        )
        if is_filled:
            cell["filled"] += 1

    total_filled = sum(c["filled"] for c in by_cat_count.values())
    total_fields = sum(c["total"] for c in by_cat_count.values())
    overall_pct = int(100 * total_filled / total_fields) if total_fields else 0

    # Barra de progreso global
    bar_color = "#10b981" if overall_pct >= 80 else "#f59e0b" if overall_pct >= 40 else "#ef4444"
    st.markdown(
        f'<div style="margin:8px 0;">'
        f'<div style="display:flex;justify-content:space-between;font-size:12px;'
        f'color:#475569;font-weight:600;margin-bottom:4px;">'
        f'<span>Asset completeness</span>'
        f'<span style="color:{bar_color};">{total_filled}/{total_fields} fields · {overall_pct}%</span>'
        f'</div>'
        f'<div style="background:#f1f5f9;border-radius:999px;height:8px;overflow:hidden;">'
        f'<div style="background:{bar_color};height:100%;width:{overall_pct}%;'
        f'border-radius:999px;transition:width .3s;"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Chips por categoría
    chip_pieces = []
    for cat in sorted(by_cat_count.keys()):
        cell = by_cat_count[cat]
        cat_pct = int(100 * cell["filled"] / cell["total"]) if cell["total"] else 0
        cat_color = "#10b981" if cat_pct >= 80 else "#f59e0b" if cat_pct >= 40 else "#94a3b8"
        cat_bg = f"{cat_color}1a"  # alpha
        icon = _CATEGORY_ICONS.get(cat, "📋")
        chip_pieces.append(
            f'<span style="display:inline-block;padding:3px 10px;'
            f'background:{cat_bg};color:{cat_color};border:1px solid {cat_color}55;'
            f'border-radius:999px;font-size:11px;font-weight:600;'
            f'margin-right:5px;margin-bottom:5px;">'
            f'{icon} {cat} · {cell["filled"]}/{cell["total"]}</span>'
        )
    st.markdown(
        f'<div style="margin:8px 0 14px;">{"".join(chip_pieces)}</div>',
        unsafe_allow_html=True,
    )

    # Panel de auto-cálculos en vivo (solo lectura)
    derived = compute_all_derived(current_values)
    if derived:
        st.markdown("#### 🧮 Automatically calculated values")
        st.caption(
            "Derived live from the entered parameters. If you type "
            "Cd manually below, that manual value wins over the calculation."
        )
        cols = st.columns(min(len(derived), 4) or 1)
        col_idx = 0
        for key, info in derived.items():
            with cols[col_idx % len(cols)]:
                if key == "diametral_clearance":
                    st.metric(
                        "Diametral Cd",
                        f"{info['value_mm']:.3f} mm",
                        delta=f"{info['value_mil']:.2f} mil pp",
                        help=info["explanation"],
                    )
                elif key == "radial_clearance":
                    st.metric(
                        "Radial Cr",
                        f"{info['value_mm']:.3f} mm",
                        delta=f"{info['value_mil']:.2f} mil pp",
                        help=info["explanation"],
                    )
                elif key == "l_over_d":
                    st.metric(
                        "L/D",
                        f"{info['value']:.2f}",
                        delta=info["interpretation"],
                        delta_color="off",
                        help=info["explanation"],
                    )
                elif key == "unit_load":
                    st.metric(
                        "Unit load",
                        f"{info['value_mpa']:.2f} MPa",
                        delta=info["interpretation"],
                        delta_color="off",
                        help=info["explanation"],
                    )
                elif key == "lift_off_speed":
                    st.metric(
                        "Lift-off est.",
                        f"{info['value_rpm']:.0f} rpm",
                        help=info["explanation"],
                    )
            col_idx += 1
        st.markdown("---")

    # Agrupar campos por categoría
    by_category: Dict[str, List[tuple]] = {}
    for field_key, field_def in CAPTURED_PARAMETER_FIELDS.items():
        cat = field_def.get("category", "Otros")
        by_category.setdefault(cat, []).append((field_key, field_def))

    new_values: Dict[str, Any] = {}

    with st.form("captured_params_form"):
        for category in sorted(by_category.keys()):
            cell = by_cat_count.get(category, {"filled": 0, "total": 0})
            cat_pct = int(100 * cell["filled"] / cell["total"]) if cell["total"] else 0
            cat_icon = _CATEGORY_ICONS.get(category, "📋")
            with st.expander(
                f"{cat_icon} {category} — {cell['filled']}/{cell['total']} ({cat_pct}%)",
                expanded=(category in ("Cojinete - geometría", "Identificación")),
            ):
                fields = by_category[category]
                cols = st.columns(2)
                for idx, (field_key, field_def) in enumerate(fields):
                    with cols[idx % 2]:
                        ftype = field_def.get("type", "str")
                        label = field_def.get("label", field_key)
                        help_text = field_def.get("help", "")
                        current = current_values.get(field_key)

                        if ftype == "float":
                            raw = st.text_input(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{instance_id}_{field_key}",
                                help=help_text,
                            )
                            if raw.strip() == "":
                                new_values[field_key] = None
                            else:
                                try:
                                    new_values[field_key] = float(raw.replace(",", "."))
                                except ValueError:
                                    new_values[field_key] = current
                        elif ftype == "date":
                            raw_date = st.text_input(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{instance_id}_{field_key}",
                                placeholder="YYYY-MM-DD",
                                help=help_text or "YYYY-MM-DD format",
                            )
                            new_values[field_key] = raw_date.strip() if raw_date.strip() else None
                        elif ftype == "text":
                            raw = st.text_area(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{instance_id}_{field_key}",
                                help=help_text,
                            )
                            new_values[field_key] = raw if raw.strip() else None
                        else:
                            raw = st.text_input(
                                label,
                                value=str(current) if current is not None else "",
                                key=f"param_{instance_id}_{field_key}",
                                help=help_text,
                            )
                            new_values[field_key] = raw if raw.strip() else None

        submitted = st.form_submit_button("Save parameters", width="stretch")

    if submitted:
        update_instance_parameters_bulk(instance_id, new_values)
        st.success(
            "Parameters saved to the instance. The auto-calculations update "
            "when you reload the page."
        )
        st.rerun()


def render_danger_zone(instance_id: str) -> None:
    """Acciones destructivas sobre la instancia (eliminar)."""
    with st.expander("⚠️ Danger zone", expanded=False):
        st.warning(
            "Deleting the instance removes all its associated parameters and "
            "documents. Irreversible operation."
        )
        confirm = st.text_input(
            f"To confirm, type the instance ID (`{instance_id}`):",
            key=f"confirm_delete_{instance_id}",
        )
        if st.button(
            "Delete instance permanently",
            disabled=(confirm.strip() != instance_id),
            key=f"delete_btn_{instance_id}",
        ):
            ok = delete_instance(instance_id)
            if ok:
                st.session_state.pop("wm_active_instance_id", None)
                st.success(f"Instance '{instance_id}' deleted.")
                st.rerun()
            else:
                st.error("Could not delete the instance.")


# ============================================================
# Ciclo 14a — GRID DE MÁQUINAS (cockpit)
# ============================================================

def _set_active_instance(target_instance_id: str) -> None:
    """
    Callback del botón "Activar" en cada card del grid.

    Hotfix v2.6: setea la key persistente y TODAS las posibles keys
    de widget de instance_selector que pueda haber en sesión. La
    Machinery Library usa module_name="documents" pero otras páginas
    pueden tener "polar", "bode", "tabular", etc. Sincronizamos todas
    las que ya estén instanciadas para evitar que un selectbox quede
    "pegado" en una página por la que pasamos antes.

    Los callbacks corren en una fase pre-render donde session_state
    se puede modificar libremente — incluso keys de widgets ya
    instanciados en el cycle anterior.
    """
    st.session_state["wm_active_instance_id"] = target_instance_id
    # Sincronizar todos los widgets de instance_selector ya instanciados
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("wm_instance_select_"):
            st.session_state[k] = target_instance_id


def render_machinery_grid() -> None:
    """
    Grilla de cards con todas las máquinas registradas. Cada card resume
    tag · driver · driven · cliente · sitio · cantidad de docs/parámetros.
    Click → activa esa instancia y dispara rerun.
    """
    instances = list_instances()
    if not instances:
        return

    st.markdown("### Registered machines")
    st.caption(
        f"{len(instances)} machine(s) in the system. "
        "Click any card to activate it across all analysis modules."
    )

    cards_per_row = 3
    rows = [instances[i:i + cards_per_row] for i in range(0, len(instances), cards_per_row)]
    for row in rows:
        cols = st.columns(cards_per_row)
        for idx, summary in enumerate(row):
            with cols[idx]:
                inst_id = summary.get("instance_id", "")
                inst = get_instance(inst_id)
                if inst is None:
                    continue

                tag = inst.tag or inst_id
                driver_part = " ".join(p for p in [inst.driver_manufacturer, inst.driver_model] if p) or "(no driver)"
                driven_part = " ".join(p for p in [inst.driven_manufacturer, inst.driven_model] if p)
                client = inst.client or "(no client)"
                site = inst.site or inst.location or ""
                n_docs = len(inst.documents)
                power_str = f"{inst.nominal_power_mw:.0f} MW" if inst.nominal_power_mw > 0 else ""
                rpm_str = f"{inst.nominal_rpm:.0f} rpm" if inst.nominal_rpm > 0 else ""

                # Card con el esquemático si existe
                with st.container(border=True):
                    if inst.schematic_png:
                        try:
                            png = get_instance_document_bytes(inst_id, inst.schematic_png)
                            if png:
                                st.image(png, use_container_width=True)
                        except Exception:
                            pass
                    st.markdown(f"**{tag}**")
                    st.caption(driver_part)
                    if driven_part:
                        st.caption(f"+ {driven_part}")
                    meta_bits = [b for b in [power_str, rpm_str] if b]
                    if meta_bits:
                        st.caption(" · ".join(meta_bits))
                    if client or site:
                        st.caption(" · ".join(p for p in [client, site] if p))
                    st.caption(f"📄 {n_docs} document(s)")
                    # Ciclo 14a — badge claro de estado del esquemático.
                    # Le dice al usuario de un vistazo si esta maquina ya
                    # tiene el esquematico vinculado para que aparezca en
                    # el Resumen Ejecutivo del PDF.
                    if inst.schematic_png:
                        st.caption("🖼️ schematic linked")
                    else:
                        st.caption("⚠️ no main schematic")

                    # Indicador si esta es la activa
                    if st.session_state.get("wm_active_instance_id") == inst_id:
                        st.success("✓ active", icon="🟢")
                    else:
                        # Usamos on_click callback porque la key
                        # 'wm_active_instance_id' ya está instanciada por el
                        # selectbox del sidebar; modificarla directo con
                        # st.session_state[...] = ... lanzaría
                        # 'cannot be modified after widget instantiated'.
                        # Los callbacks corren en una fase especial donde
                        # session_state se puede escribir libremente.
                        st.button(
                            "Activate",
                            key=f"activate_{inst_id}",
                            on_click=_set_active_instance,
                            args=(inst_id,),
                            width="stretch",
                        )

    st.markdown("---")


# ============================================================
# Ciclo 22.2a — GRID MODERNO (cards con severidad + chips + schematic)
# ============================================================

_SEVERITY_CONFIG = {
    "CRÍTICA":            ("#dc2626", "#fef2f2", "🔴"),
    "ACCIÓN REQUERIDA":   ("#dc2626", "#fef2f2", "🟠"),
    "ATENCIÓN":           ("#f59e0b", "#fffbeb", "🟡"),
    "VIGILANCIA":         ("#3b82f6", "#eff6ff", "🔵"),
    "CONDICIÓN ACEPTABLE": ("#10b981", "#f0fdf4", "🟢"),
}


def _render_machinery_card_v2(inst: Any, inst_id: str) -> None:
    """Card moderno individual (Ciclo 22.2a)."""
    import textwrap
    from base64 import b64encode

    tag = inst.tag or inst_id
    driver = " ".join(p for p in [inst.driver_manufacturer, inst.driver_model] if p) or ""
    driven = " ".join(p for p in [inst.driven_manufacturer, inst.driven_model] if p) or ""
    title = driver if not driven else f"{driver} → {driven}"

    severity = (inst.last_executive_severity or "").upper().strip()
    sev_color, sev_bg, sev_icon = _SEVERITY_CONFIG.get(
        severity, ("#64748b", "#f1f5f9", "⚪"),
    )
    sev_label = severity or "SIN ANÁLISIS"

    # Schematic embebido
    schematic_html = ""
    if inst.schematic_png:
        try:
            png = get_instance_document_bytes(inst_id, inst.schematic_png)
            if png:
                b64 = b64encode(png).decode("ascii")
                schematic_html = (
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="width:100%;height:130px;object-fit:contain;'
                    f'background:#f9fafb;border-radius:8px;'
                    f'border:1px solid #e5e7eb;margin-bottom:10px;" />'
                )
        except Exception:
            schematic_html = ""

    if not schematic_html:
        schematic_html = (
            '<div style="width:100%;height:130px;background:linear-gradient(135deg,'
            '#f1f5f9 0%,#e2e8f0 100%);border-radius:8px;display:flex;'
            'align-items:center;justify-content:center;margin-bottom:10px;'
            'font-size:42px;border:1px dashed #cbd5e1;">⚙️</div>'
        )

    # Chips de metadata
    chips = []
    if inst.client:
        chips.append(("👤", inst.client))
    if inst.site or inst.location:
        chips.append(("📍", inst.site or inst.location))
    if inst.nominal_rpm and inst.nominal_rpm > 0:
        chips.append(("⚡", f"{int(inst.nominal_rpm):,} RPM"))
    if inst.nominal_power_mw and inst.nominal_power_mw > 0:
        chips.append(("🔋", f"{inst.nominal_power_mw:.1f} MW"))

    chips_html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:3px;'
        f'padding:2px 8px;background:#f3f4f6;border-radius:999px;'
        f'font-size:11px;color:#475569;margin-right:4px;margin-bottom:4px;'
        f'border:1px solid #e5e7eb;">{icon} {text}</span>'
        for icon, text in chips
    )
    if not chips_html:
        chips_html = '<span style="font-size:11px;color:#94a3b8;">(no metadata)</span>'

    n_sensors = len(inst.sensors or [])
    n_docs = len(inst.documents or [])

    is_active = st.session_state.get("wm_active_instance_id") == inst_id

    # Card HTML (Ciclo 23.54 polish) — hover lift + accent ring para
    # active, footer separado con icons monocromos en vez de emojis
    # rainbow. Look enterprise SaaS, no consumer-app.
    card_html = textwrap.dedent(f"""\
    <style>
    .wmlib-card {{
        border-radius: 14px;
        padding: 16px;
        background: white;
        margin-bottom: 8px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        height: 100%;
        position: relative;
    }}
    .wmlib-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(15,23,42,0.10);
        border-color: #cbd5e1;
    }}
    .wmlib-card.is-active {{
        border: 2px solid #2563eb !important;
        box-shadow: 0 8px 24px rgba(37,99,235,0.18);
    }}
    .wmlib-card.is-active::before {{
        content: ""; position: absolute;
        top: 0; left: 0; bottom: 0; width: 4px;
        background: linear-gradient(180deg, #2563eb 0%, #1e40af 100%);
        border-radius: 14px 0 0 14px;
    }}
    .wmlib-card-head {{
        display: flex; justify-content: space-between;
        align-items: center; margin-bottom: 10px;
    }}
    .wmlib-card-tag {{
        font-weight: 800; color: #0f172a;
        font-size: 17px; letter-spacing: -0.01em;
    }}
    .wmlib-card-title {{
        font-size: 13px; color: #1f2937;
        font-weight: 600; margin-bottom: 10px;
        line-height: 1.35;
    }}
    .wmlib-card-footer {{
        display: flex; justify-content: space-between;
        font-size: 11px; color: #64748b;
        border-top: 1px solid #f1f5f9;
        padding-top: 10px; margin-top: 10px;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
    }}
    .wmlib-card-footer b {{ color: #0f172a; font-weight: 800; }}
    </style>
    <div class="wmlib-card{' is-active' if is_active else ''}"
         style="border:{'2px solid #2563eb' if is_active else '1px solid #e6ebf2'};
                box-shadow:{'0 8px 24px rgba(37,99,235,0.18)' if is_active else '0 1px 3px rgba(15,23,42,0.04)'};">
      <div class="wmlib-card-head">
        <div class="wmlib-card-tag">{tag}</div>
        <span style="background:{sev_bg};color:{sev_color};padding:3px 9px;border-radius:999px;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:0.05em;border:1px solid {sev_color}33;">{sev_icon} {sev_label}</span>
      </div>
      {schematic_html}
      <div class="wmlib-card-title">{title or "(no train defined)"}</div>
      <div style="margin-bottom:6px;">{chips_html}</div>
      <div class="wmlib-card-footer">
        <span>◉ <b>{n_sensors}</b> sensors</span>
        <span>⎙ <b>{n_docs}</b> docs</span>
      </div>
    </div>
    """)
    st.markdown(card_html, unsafe_allow_html=True)

    if is_active:
        st.success("✓ ACTIVE", icon="🟢")
    else:
        st.button(
            "Activate →",
            key=f"activate_v2_{inst_id}",
            on_click=_set_active_instance,
            args=(inst_id,),
            use_container_width=True,
        )


def render_machinery_grid_v2() -> None:
    """
    Grid moderno de máquinas (Ciclo 22.2a + 23.54 international polish).

    Iteraciones:
      22.2a: Cards con severidad coloreada, chips de metadata, schematic.
      23.54: + Header refinado con count badges, search box, filtros por
             cliente y severidad, sort, hover states, density mejorada.
             Target: superar System1/Bently en first impression de la
             librería de máquinas.
    """
    instances = list_instances()
    if not instances:
        return

    # Cargamos los Instance objects una vez para extraer metadata para
    # filtros y para render. Cache en local list para no re-llamar
    # get_instance() en el loop de cards.
    inst_pairs: List[Tuple[str, Any]] = []
    for meta in instances:
        iid = meta.get("instance_id", "")
        if not iid:
            continue
        inst = get_instance(iid)
        if inst is not None:
            inst_pairs.append((iid, inst))

    if not inst_pairs:
        return

    # Ciclo 23.55 (v3.31.260/263) — Auto-filtrar instancias placeholder.
    # CONSERVADOR: solo filtramos por tag LITERAL "(default)" (creada
    # por sistema) o si TODAS las 4 condiciones de vacía se cumplen.
    # NO filtramos por tag vacío solo — algunas instancias reales como
    # 'tes1' (Ecopetrol-Magnex con 9 sensores) tienen tag vacío pero
    # data real y monitoreo en línea.
    def _is_placeholder_instance(inst: Any) -> bool:
        tag = (inst.tag or "").strip()
        # Solo tags LITERALES del sistema
        if tag in ("(default)", "default", "(sin tren)", "(sin nombre)"):
            return True
        # O totalmente vacía (incluye tag vacío + sin nada más)
        no_sensors = not (inst.sensors or [])
        no_docs = not (inst.documents or [])
        no_client = not (inst.client or "").strip()
        no_train = (
            not (inst.driver_manufacturer or "").strip()
            and not (inst.driver_model or "").strip()
            and not (inst.driven_manufacturer or "").strip()
            and not (inst.driven_model or "").strip()
        )
        return no_sensors and no_docs and no_client and no_train

    inst_pairs = [(iid, inst) for iid, inst in inst_pairs
                  if not _is_placeholder_instance(inst)]

    if not inst_pairs:
        st.info(
            "No machines registered yet. Tap "
            "**🧙 Open creation wizard** below to create the first one."
        )
        return

    # Header refinado (Ciclo 23.54) — small caps + count badges en vez
    # del 🏭 emoji "consumer-app". Look enterprise SaaS.
    total_machines = len(inst_pairs)
    total_sensors = sum(len(inst.sensors or []) for _, inst in inst_pairs)
    total_docs = sum(len(inst.documents or []) for _, inst in inst_pairs)

    st.markdown(
        textwrap.dedent(f"""
        <style>
        .wmlib-header-row {{
            display: flex; align-items: baseline;
            justify-content: space-between; flex-wrap: wrap;
            gap: 10px; margin: 8px 0 6px 0;
        }}
        .wmlib-header-title {{
            font-size: 12px; font-weight: 800;
            letter-spacing: 0.18em; text-transform: uppercase;
            color: #475569;
        }}
        .wmlib-header-counts {{
            display: inline-flex; gap: 14px; align-items: center;
            font-size: 11px; color: #64748b;
            font-family: ui-monospace, "SF Mono", Menlo, monospace;
        }}
        .wmlib-header-counts b {{ color: #0f172a; font-weight: 800; }}
        .wmlib-header-counts .sep {{ color: #cbd5e1; }}
        .wmlib-filter-bar {{
            background: white;
            border: 1px solid #e6ebf2;
            border-radius: 12px;
            padding: 10px 14px;
            margin: 8px 0 18px 0;
            box-shadow: 0 2px 8px rgba(15,23,42,0.04);
        }}
        .wmlib-filter-hint {{
            font-size: 11px; color: #94a3b8;
            margin-top: 6px;
            font-family: ui-monospace, "SF Mono", Menlo, monospace;
        }}
        </style>
        <div class="wmlib-header-row">
            <span class="wmlib-header-title">Registered machines</span>
            <span class="wmlib-header-counts">
                <span><b>{total_machines}</b> machines</span>
                <span class="sep">·</span>
                <span><b>{total_sensors}</b> sensors</span>
                <span class="sep">·</span>
                <span><b>{total_docs}</b> docs</span>
            </span>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    # ─── Fleet condition cockpit (Ciclo 23.56) ─────────────────
    # KPI strip con colores ISO 20816 (Vigilancia / Atención /
    # Crítico). Da lectura de flota de un vistazo — el lenguaje que
    # usan System1 / AMS para diferenciarse de una simple tabla.
    def _sevn(i: Any) -> str:
        return (i.last_executive_severity or "").upper().strip()

    n_watch = sum(1 for _, i in inst_pairs if _sevn(i) == "VIGILANCIA")
    n_warn = sum(1 for _, i in inst_pairs if _sevn(i) == "ATENCIÓN")
    n_crit = sum(
        1 for _, i in inst_pairs
        if _sevn(i) in ("CRÍTICA", "ACCIÓN REQUERIDA")
    )

    st.markdown(
        textwrap.dedent(f"""
        <style>
        .wmlib-kpis {{
            display: grid; grid-template-columns: repeat(5, 1fr);
            gap: 10px; margin: 4px 0 16px 0;
        }}
        .wmlib-kpi {{
            border-radius: 12px; padding: 12px 14px; background: white;
            border: 1px solid #e6ebf2;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        }}
        .wmlib-kpi.accent {{ border-left-width: 4px; border-left-style: solid; }}
        .wmlib-kpi .lbl {{
            font-size: 10px; font-weight: 800; letter-spacing: 0.1em;
            text-transform: uppercase; color: #94a3b8;
        }}
        .wmlib-kpi .val {{
            font-size: 26px; font-weight: 800; line-height: 1.1; margin-top: 3px;
            font-family: ui-monospace, "SF Mono", Menlo, monospace;
        }}
        @media (max-width: 640px) {{
            .wmlib-kpis {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        </style>
        <div class="wmlib-kpis">
          <div class="wmlib-kpi">
            <div class="lbl">Assets</div>
            <div class="val" style="color:#0f172a;">{total_machines}</div>
          </div>
          <div class="wmlib-kpi">
            <div class="lbl">Sensors</div>
            <div class="val" style="color:#0f172a;">{total_sensors}</div>
          </div>
          <div class="wmlib-kpi accent" style="border-left-color:#3b82f6;">
            <div class="lbl">Vigilancia</div>
            <div class="val" style="color:#3b82f6;">{n_watch}</div>
          </div>
          <div class="wmlib-kpi accent" style="border-left-color:#f59e0b;">
            <div class="lbl">Atención</div>
            <div class="val" style="color:#f59e0b;">{n_warn}</div>
          </div>
          <div class="wmlib-kpi accent" style="border-left-color:#dc2626;">
            <div class="lbl">Crítico</div>
            <div class="val" style="color:#dc2626;">{n_crit}</div>
          </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )

    # ─── Filtros (Ciclo 23.54) ─────────────────────────────────
    # Search + cliente + severidad. Auto-derivamos las opciones de
    # cliente de los instances cargados — no hay que mantener lista.
    all_clients = sorted({
        (inst.client or "").strip() for _, inst in inst_pairs
        if (inst.client or "").strip()
    })
    all_sites = sorted({
        (inst.site or inst.location or "").strip() for _, inst in inst_pairs
        if (inst.site or inst.location or "").strip()
    })
    all_severities = sorted({
        (inst.last_executive_severity or "SIN ANÁLISIS").upper().strip()
        for _, inst in inst_pairs
    })

    f1, f2, f3, f4 = st.columns([3, 2, 2, 1.2])
    with f1:
        search_q = st.text_input(
            "Search",
            placeholder="🔍 Search machine, model, client, site…",
            key="wmlib_search",
            label_visibility="collapsed",
        ).strip().lower()
    with f2:
        sel_clients = st.multiselect(
            "Client",
            options=all_clients,
            default=[],
            key="wmlib_filter_client",
            placeholder="All clients" if all_clients else "No clients",
            label_visibility="collapsed",
        )
    with f3:
        sel_severities = st.multiselect(
            "Severity",
            options=all_severities,
            default=[],
            key="wmlib_filter_sev",
            placeholder="All severities",
            label_visibility="collapsed",
        )
    with f4:
        sort_by = st.selectbox(
            "Sort",
            options=["A → Z", "Z → A", "Más sensores", "Menos sensores"],
            index=0,
            key="wmlib_sort",
            label_visibility="collapsed",
        )

    # Aplicar filtros
    def _matches(iid: str, inst: Any) -> bool:
        if sel_clients and (inst.client or "").strip() not in sel_clients:
            return False
        if sel_severities:
            sev = (inst.last_executive_severity or "SIN ANÁLISIS").upper().strip()
            if sev not in sel_severities:
                return False
        if search_q:
            haystack = " ".join([
                iid, inst.tag or "",
                inst.driver_manufacturer or "", inst.driver_model or "",
                inst.driven_manufacturer or "", inst.driven_model or "",
                inst.client or "", inst.site or "", inst.location or "",
            ]).lower()
            if search_q not in haystack:
                return False
        return True

    filtered = [(iid, inst) for iid, inst in inst_pairs if _matches(iid, inst)]

    # Sort
    if sort_by == "A → Z":
        filtered.sort(key=lambda p: (p[1].tag or p[0]).lower())
    elif sort_by == "Z → A":
        filtered.sort(key=lambda p: (p[1].tag or p[0]).lower(), reverse=True)
    elif sort_by == "Más sensores":
        filtered.sort(key=lambda p: -len(p[1].sensors or []))
    elif sort_by == "Menos sensores":
        filtered.sort(key=lambda p: len(p[1].sensors or []))

    # Si los filtros activos redujeron el set, mostrar contador
    if len(filtered) != len(inst_pairs):
        st.markdown(
            f'<div class="wmlib-filter-hint">'
            f'Showing <b>{len(filtered)}</b> of <b>{len(inst_pairs)}</b> machines. '
            f'<a href="#" onclick="window.location.reload();return false;">Clear filters</a> ↻'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not filtered:
        st.info(
            "No machines match the applied filters. "
            "Try adjusting the search or clearing the filters."
        )
        st.markdown("---")
        return

    # Ciclo 23.55 (v3.31.260) — Tabla minimalista internacional
    # (reemplaza el grid de cards con imágenes grandes que ocupaban
    # demasiado espacio vertical). Look enterprise SaaS tipo
    # System1/AMS/Linear — densidad alta, escaneable de un vistazo.
    # Las imágenes/schematic siguen accesibles dentro del detalle de
    # cada máquina al activarla.
    _render_machinery_table(filtered)

    st.markdown("---")


def _render_machinery_table(filtered: List[Tuple[str, Any]]) -> None:
    """Tabla compacta de máquinas — densidad enterprise SaaS.

    Columnas: Activo | Cliente · Sitio | Tren | Sensores | Severidad | Acción
    """
    if not filtered:
        return

    # CSS de la tabla (una sola vez)
    st.markdown("""
    <style>
    .wmlib-table {
        background: white;
        border: 1px solid #e6ebf2;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        margin-bottom: 8px;
    }
    .wmlib-row {
        display: grid;
        grid-template-columns: 1.4fr 1.6fr 2.2fr 0.7fr 1.1fr 0.8fr;
        gap: 14px;
        padding: 12px 18px;
        align-items: center;
        border-bottom: 1px solid #f1f5f9;
        transition: background 0.12s ease;
    }
    .wmlib-row:last-child { border-bottom: none; }
    .wmlib-row:hover { background: #f8fafc; }
    .wmlib-row.is-active {
        background: linear-gradient(90deg, rgba(37,99,235,0.06), transparent);
        border-left: 3px solid #2563eb;
        padding-left: 15px;
    }
    .wmlib-row.is-header {
        background: #f8fafc;
        border-bottom: 1px solid #e6ebf2;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #64748b;
        padding: 10px 18px;
    }
    .wmlib-row.is-header:hover { background: #f8fafc; }
    .wmlib-tag {
        font-size: 14px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .wmlib-sub {
        font-size: 11px;
        color: #94a3b8;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        margin-top: 2px;
    }
    .wmlib-client {
        font-size: 13px;
        color: #0f172a;
        font-weight: 600;
    }
    .wmlib-site {
        font-size: 11px;
        color: #64748b;
        margin-top: 2px;
    }
    .wmlib-train {
        font-size: 12px;
        color: #1f2937;
        line-height: 1.35;
    }
    .wmlib-count {
        font-size: 13px;
        font-weight: 800;
        color: #0f172a;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
    }
    .wmlib-count-sub {
        font-size: 10px;
        color: #94a3b8;
        margin-top: 1px;
    }
    .wmlib-sev-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.3;
        white-space: nowrap;
    }
    .wmlib-dim {
        color: #cbd5e1;
        font-style: italic;
        font-size: 12px;
    }
    /* Botón Activar/Abrir compacto que matchea la altura de fila */
    div[data-testid="stHorizontalBlock"] .wmlib-action-cell button {
        min-height: 30px !important;
        padding: 4px 12px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        border-radius: 7px !important;
    }
    </style>
    <div class="wmlib-table">
      <div class="wmlib-row is-header">
        <div>Asset</div>
        <div>Client · Site</div>
        <div>Train</div>
        <div style="text-align:right;">Sensors</div>
        <div>Severity</div>
        <div style="text-align:right;">Action</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Render fila por fila con st.columns (necesario para botones interactivos)
    for inst_id, inst in filtered:
        _render_machinery_row(inst, inst_id)

    # Leyenda de severidad ISO 20816 — explica el código de color de las
    # bandas laterales y los badges, alineado con el cockpit de KPIs.
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:16px;margin:10px 2px 0;'
        'font-size:11px;color:#94a3b8;font-family:ui-monospace,Menlo,monospace;">'
        + "".join(
            f'<span><span style="display:inline-block;width:9px;height:9px;'
            f'border-radius:2px;background:{c};margin-right:5px;'
            f'vertical-align:middle;"></span>{lbl}</span>'
            for c, lbl in [
                ("#10b981", "Condición aceptable"),
                ("#3b82f6", "Vigilancia"),
                ("#f59e0b", "Atención"),
                ("#dc2626", "Crítico / acción requerida"),
                ("#94a3b8", "Sin análisis"),
            ]
        )
        + '<span style="margin-left:auto;font-style:italic;">ISO 20816</span>'
        + '</div>',
        unsafe_allow_html=True,
    )


def _render_machinery_row(inst: Any, inst_id: str) -> None:
    """Una fila de la tabla minimalista."""
    tag = inst.tag or inst_id
    client = (inst.client or "").strip()
    site = (inst.site or inst.location or "").strip()

    driver = " ".join(p for p in [inst.driver_manufacturer, inst.driver_model] if p) or ""
    driven = " ".join(p for p in [inst.driven_manufacturer, inst.driven_model] if p) or ""
    if driver and driven:
        train = f"{driver} → {driven}"
    else:
        train = driver or driven or ""

    severity = (inst.last_executive_severity or "").upper().strip()
    sev_color, sev_bg, _ = _SEVERITY_CONFIG.get(
        severity, ("#94a3b8", "#f1f5f9", ""),
    )
    sev_label = severity or "Sin análisis"

    n_sensors = len(inst.sensors or [])
    n_docs = len(inst.documents or [])

    is_active = st.session_state.get("wm_active_instance_id") == inst_id

    # Columnas: 1.4 | 1.6 | 2.2 | 0.7 | 1.1 | 0.8  (mismas proporciones del header CSS)
    c1, c2, c3, c4, c5, c6 = st.columns([1.4, 1.6, 2.2, 0.7, 1.1, 0.8])

    with c1:
        active_dot = (
            '<span style="display:inline-block;width:6px;height:6px;'
            'border-radius:50%;background:#2563eb;margin-right:6px;'
            'vertical-align:middle;"></span>' if is_active else ''
        )
        # Banda de severidad a la izquierda (ISO 20816) — cue de color
        # que permite escanear el estado de toda la flota verticalmente.
        st.markdown(
            f'<div style="border-left:4px solid {sev_color};padding-left:10px;">'
            f'<div class="wmlib-tag">{active_dot}{tag}</div>'
            f'<div class="wmlib-sub">{inst_id[:12]}{"…" if len(inst_id) > 12 else ""}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c2:
        if client:
            site_str = (
                f'<div class="wmlib-site">{site}</div>' if site else ''
            )
            st.markdown(
                f'<div class="wmlib-client">{client}</div>{site_str}',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="wmlib-dim">—</span>', unsafe_allow_html=True)

    with c3:
        train_parts = [p for p in [driver, driven] if p]
        if train_parts:
            sep = (
                '<span style="color:#cbd5e1;font-size:11px;'
                'margin:0 3px;vertical-align:middle;">→</span>'
            )
            chips = sep.join(
                f'<span style="display:inline-block;padding:2px 8px;'
                f'background:#f3f4f6;border:1px solid #e5e7eb;border-radius:6px;'
                f'font-size:11px;color:#334155;margin:1px 0;'
                f'font-family:ui-monospace,\'SF Mono\',Menlo,monospace;">{p}</span>'
                for p in train_parts
            )
            st.markdown(
                f'<div class="wmlib-train">{chips}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="wmlib-dim">no train defined</span>',
                unsafe_allow_html=True,
            )

    with c4:
        st.markdown(
            f'<div style="text-align:right;">'
            f'<div class="wmlib-count">'
            f'<span style="display:inline-block;width:7px;height:7px;'
            f'border-radius:50%;background:{sev_color};margin-right:5px;'
            f'vertical-align:middle;"></span>{n_sensors}</div>'
            f'<div class="wmlib-count-sub">{n_docs} docs</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c5:
        st.markdown(
            f'<span class="wmlib-sev-badge" '
            f'style="background:{sev_bg};color:{sev_color};'
            f'border:1px solid {sev_color}33;">{sev_label}</span>',
            unsafe_allow_html=True,
        )

    with c6:
        if is_active:
            st.markdown(
                '<div style="text-align:right;font-size:11px;font-weight:800;'
                'color:#2563eb;letter-spacing:0.05em;text-transform:uppercase;">'
                '● Active</div>',
                unsafe_allow_html=True,
            )
        else:
            st.button(
                "Open →",
                key=f"open_row_{inst_id}",
                on_click=_set_active_instance,
                args=(inst_id,),
                use_container_width=True,
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    require_login()
    render_user_menu()

    page_header(
        title="Machinery Library",
        subtitle="Central machinery cockpit — technical profile, schematics, OEM manuals and physical parameters per asset instance.",
    )

    with st.sidebar:
        st.markdown("---")
        # Pasamos module_name="documents" (alias histórico) en vez de
        # "library" porque el sistema de profiles valida que el module_name
        # esté declarado en core/machine_profiles. La Library es universal
        # (toda máquina debe ser configurable acá), así que reusamos el
        # alias 'documents' que ya está aceptado por todos los profiles.
        state = render_instance_selector(module_name="documents")

    instance_id = state.get("instance_id")

    # Ciclo 14a — Grid de cards de TODAS las máquinas registradas (cockpit)
    # Ciclo 22.2a — Nuevo grid moderno por defecto.
    # El antiguo render_machinery_grid() se mantiene como fallback
    # legacy en caso de necesitar rollback.
    render_machinery_grid_v2()

    # Sección siempre visible: crear instancia nueva
    render_create_instance_section()

    if not instance_id:
        st.info(
            "No active machine. Tap **🧙 Open creation wizard** "
            "above to create a new one, or select one from the machines "
            "grid / sidebar."
        )
        return

    st.markdown("---")
    render_instance_header(state)

    st.markdown("---")
    render_sensor_map_section(instance_id)

    st.markdown("---")
    render_captured_parameters_section(instance_id)

    st.markdown("---")
    render_documents_section(instance_id)

    st.markdown("---")
    render_upload_section(instance_id)

    st.markdown("---")
    render_danger_zone(instance_id)


if __name__ == "__main__":
    main()
