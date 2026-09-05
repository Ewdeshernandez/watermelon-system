"""
pages/18_Modal_Analysis.py — Módulo de Análisis Modal Watermelon
=================================================================

Stack modal nativo de Watermelon. Implementa EMA + OMA + comparación con FEA
con todos los algoritmos in-house, sin dependencias de software de terceros.

Tabs
----
1. Setup           — Geometría 3D + sensor → DOF mapping
2. Adquisición    — Captura live + importer de archivos pre-grabados
3. EMA Processing — FRF + LSCF + stability diagram
4. OMA Processing — FDD + SSI
5. Mode Shapes 3D — Animación Plotly Mesh3d + export GIF/MP4
6. FEA Compare    — Importer FEA + MAC matrix

Marco normativo
---------------
· ISO 7626-1 a 7626-6 (EMA)
· ISO 20816 (OMA)
· API 684 (rotor dynamics validation)
· API 618 secc. 7.9 (criterio separación modal)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.auth import (
    require_login, render_user_menu, get_current_user, is_page_allowed_for_role,
)
from core.ui_theme import page_header
from core.modal.ui_components import (
    modal_hero_card,
    modal_footer_norms,
    modal_kpi_row,
    modal_section_header,
    modal_plot_caption,
    modal_status_banner,
    modal_empty_state,
)


# =====================================================================
# Setup de página
# =====================================================================
st.set_page_config(
    page_title="Watermelon System | Modal Analysis",
    page_icon="🌐",
    layout="wide",
)

# =====================================================================
# v3.31.200 — Capturar traceback real en producción
# =====================================================================
# Cuando Streamlit Cloud kill-ea el proceso por OOM, no hay traceback
# en logs (solo "Oh no"). Pero si la excepción es Python pura, este
# hook la imprime al stdout (visible en logs Streamlit Cloud) ANTES de
# que Streamlit la atrape y muestre "Oh no". Así diagnosticamos el
# próximo crash si no es OOM puro.
# =====================================================================
import sys as _sys
import traceback as _tb
import logging as _logging

_logger = _logging.getLogger("watermelon.modal")
_logger.setLevel(_logging.ERROR)

def _modal_excepthook(exc_type, exc_value, exc_tb):
    """Imprime traceback completo a stdout para Streamlit Cloud logs."""
    print("=" * 80, flush=True)
    print(f"[MODAL FATAL] {exc_type.__name__}: {exc_value}", flush=True)
    print("=" * 80, flush=True)
    _tb.print_exception(exc_type, exc_value, exc_tb)
    print("=" * 80, flush=True)
    # Delega al handler default de Streamlit para que muestre "Oh no" con el error
    _sys.__excepthook__(exc_type, exc_value, exc_tb)

# Solo instalar una vez por sesión (Streamlit hace muchos reruns)
if not getattr(_sys, "_watermelon_modal_excepthook_installed", False):
    _sys.excepthook = _modal_excepthook
    _sys._watermelon_modal_excepthook_installed = True

require_login()
render_user_menu()

_user = get_current_user() or {}
_my_role = str(_user.get("role", "")).lower()
if not is_page_allowed_for_role("pages/18_Modal_Analysis.py", _my_role):
    st.error("Your role does not have access to this module.")
    st.stop()

# =====================================================================
# Session state — guardar FRFs cargados entre reruns
# =====================================================================
if "modal_frfs" not in st.session_state:
    st.session_state["modal_frfs"] = []  # list[LegacyFRF | FRFResult]


# =====================================================================
# HERO global — siempre visible arriba
# =====================================================================
_tdms = st.session_state.get("modal_tdms")
_active_method = "—"
_record_info = ""
_asset_name = "(no asset selected)"

if _tdms is not None:
    _mode = (_tdms.mode or "").lower()
    if "oma" in _mode or "continuous" in _mode:
        _active_method = "OMA"
    elif "ema" in _mode or "triggered" in _mode:
        _active_method = "EMA"
    else:
        _active_method = "—"
    _dur = _tdms.channels[0].duration_s if _tdms.channels else 0
    _record_info = (
        f"{_dur:.0f}s @ {_tdms.sample_rate_hz:.0f} Hz · "
        f"{len(_tdms.channels)} ch"
    )

# Hero global lee EITHER:
#   1. Metadata ad-hoc del Tab Setup (cliente externo / one-off)
#   2. Instancia de Machinery Library (activo registrado)
#   3. Vacío si nada configurado
_asset_location = ""
_asset_client = ""
_adhoc_meta = st.session_state.get("modal_adhoc_meta") or {}

if _adhoc_meta.get("tag"):
    # Modo ad-hoc tiene prioridad si está configurado
    _asset_name = _adhoc_meta["tag"].upper()
    _asset_client = _adhoc_meta.get("client", "")
    _asset_location = _adhoc_meta.get("station", "")
else:
    _modal_inst_key = st.session_state.get("modal_inst", "")
    if _modal_inst_key and _modal_inst_key not in ("(seleccionar)", ""):
        try:
            from core.instance_state import get_instance as _get_inst
            _inst_obj = _get_inst(_modal_inst_key)
            if _inst_obj is not None:
                _asset_name = (_inst_obj.tag or _modal_inst_key).upper()
                _asset_location = getattr(_inst_obj, "location", "") or ""
            else:
                _asset_name = _modal_inst_key.upper()
        except Exception:
            _asset_name = _modal_inst_key.upper()

modal_hero_card(
    asset_name=_asset_name,
    client_name=_asset_client,
    station_name=_asset_location,
    method_active=_active_method,
    record_info=_record_info,
)


# =====================================================================
# Tabs
# =====================================================================
# Navegación con estado PERSISTENTE. Antes se usaba st.tabs, que se reinicia a
# la 1ª pestaña en cada st.rerun() → tras cualquier acción (cargar, aplicar,
# ejecutar) el flujo "saltaba" de vuelta a Setup y había que re-navegar. El
# radio guarda la pestaña activa en session_state y sobrevive los reruns, así
# el usuario se queda donde estaba (v3.31.445).
_MODAL_TABS = [
    "🛠 Setup", "📥 Adquisición", "🔨 EMA", "🌊 OMA",
    "🎬 Mode Shapes 3D", "🧮 FEA Compare", "📊 Reports",
]
_active_modal_tab = st.radio(
    "Modal module navigation", _MODAL_TABS,
    horizontal=True, key="modal_active_tab", label_visibility="collapsed",
)
st.divider()


# ---------------------------------------------------------------------
# Tab 1 — Setup
# ---------------------------------------------------------------------
if _active_modal_tab == "🛠 Setup":
    modal_section_header(
        title="Modal test configuration",
        subtitle="Select or register the asset under analysis",
        norm_ref="ISO 7626-6 sec. 6",
        icon="🛠",
    )

    # Restaurar la última selección (modo + activo + metadata ad-hoc) que se
    # persistió a disco, para NO tener que re-seleccionar el activo cada vez que
    # la app recarga / se actualiza / se vuelve al Setup (feedback v3.31.438).
    from core.modal.modal_session import (
        load_last_selection as _load_sel,
        save_last_selection as _save_sel,
    )
    if not st.session_state.get("_modal_sel_restored"):
        st.session_state["_modal_sel_restored"] = True
        _persist = _load_sel()
        _pm = _persist.get("setup_mode", "")
        _valid_modes = ("📦 Activo registrado en Machinery Library",
                        "🎯 Análisis ad-hoc — equipo externo / one-off")
        if _pm in _valid_modes and "modal_setup_mode" not in st.session_state:
            st.session_state["modal_setup_mode"] = _pm
        # El activo registrado se aplica más abajo (cuando conocemos opciones).
        st.session_state["_modal_pending_asset"] = _persist.get("asset_id", "")
        _adh = _persist.get("adhoc", {}) or {}
        for _k in ("tag", "client", "station", "model", "notes"):
            _v = _adh.get(_k)
            if _v and f"modal_adhoc_{_k}" not in st.session_state:
                st.session_state[f"modal_adhoc_{_k}"] = _v
        if _adh.get("tag"):
            st.session_state.setdefault("modal_adhoc_meta", _adh)

    # ─── Modo dual: activo registrado vs análisis ad-hoc ──────────────
    # Ad-hoc cubre clientes que solo contratan análisis modal puntual sin
    # registrar el activo en Machinery Library (consultoría one-off, equipo
    # externo, comisionamiento previo a monitoreo). Bureau Veritas / DNV / SIGA
    # usan este patrón frecuentemente.
    setup_mode = st.radio(
        "Asset source",
        [
            "📦 Activo registrado en Machinery Library",
            "🎯 Análisis ad-hoc — equipo externo / one-off",
        ],
        horizontal=False,
        key="modal_setup_mode",
        help=(
            "Registered asset: uses the sensor configuration already defined "
            "(Sensor Map). Ad-hoc: enter metadata manually, useful for "
            "clients who only contract a one-off modal analysis without "
            "continuous monitoring."
        ),
    )
    st.divider()

    # ─── MODO AD-HOC ──────────────────────────────────────────────────
    if setup_mode.startswith("🎯"):
        st.warning(
            "⏱️ **Ad-hoc mode = TEMPORARY.** This data lives only in this "
            "session and **does NOT register a permanent asset** (it will not "
            "appear in the 'Asset under analysis' list). It is meant for a "
            "one-off analysis of an external machine.\n\n"
            "➡️ **To create a PERMANENT asset** that gets saved and can be "
            "reused in future analyses, use **'✦ Create asset (wizard)'** "
            "in the sidebar (Machinery Library). That one does register it."
        )
        st.markdown("**Asset data (one-off analysis · temporary)**")
        st.caption(
            "Fill in the minimum metadata of the machine under analysis. "
            "This data will appear in the module Hero and in the final report "
            "of THIS session (it is not saved as an asset)."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            adhoc_tag = st.text_input(
                "Machine name / Tag *",
                value=st.session_state.get("modal_adhoc_tag", ""),
                placeholder="e.g. GE LM6000 Motor — Pad 2",
                key="modal_adhoc_tag",
            )
            adhoc_client = st.text_input(
                "Client",
                value=st.session_state.get("modal_adhoc_client", ""),
                placeholder="e.g. MAGNEX, PAREX, Ecopetrol",
                key="modal_adhoc_client",
            )
        with col_b:
            adhoc_station = st.text_input(
                "Station / Location",
                value=st.session_state.get("modal_adhoc_station", ""),
                placeholder="e.g. La Belleza, Isla 6, Termosuria",
                key="modal_adhoc_station",
            )
            adhoc_model = st.text_input(
                "Model / Type",
                value=st.session_state.get("modal_adhoc_model", ""),
                placeholder="e.g. 45 MW aeroderivative turbogenerator",
                key="modal_adhoc_model",
            )

        adhoc_notes = st.text_area(
            "Technical notes (optional)",
            value=st.session_state.get("modal_adhoc_notes", ""),
            placeholder="Operating conditions, reason for the analysis, "
                         "client observations...",
            key="modal_adhoc_notes",
            height=68,
        )

        if adhoc_tag:
            # Guardar como pseudo-instance en session_state
            _adhoc_dict = {
                "tag": adhoc_tag,
                "client": adhoc_client,
                "station": adhoc_station,
                "model": adhoc_model,
                "notes": adhoc_notes,
            }
            st.session_state["modal_adhoc_meta"] = _adhoc_dict
            # Persistir a disco para restaurar tras recarga (v3.31.438).
            _save_sel(setup_mode=setup_mode, asset_id="", adhoc=_adhoc_dict)
            # Limpiar el selector de registrado para evitar conflicto
            if st.session_state.get("modal_inst") not in (None, "(seleccionar)", ""):
                st.session_state["modal_inst"] = "(seleccionar)"

            modal_status_banner(
                title=f"Ad-hoc analysis configured · {adhoc_tag}",
                detail=(
                    f"Asset registered for this modal analysis session. "
                    f"{('Client: ' + adhoc_client + ' · ') if adhoc_client else ''}"
                    f"{('Location: ' + adhoc_station + ' · ') if adhoc_station else ''}"
                    f"{('Model: ' + adhoc_model) if adhoc_model else ''}"
                ),
                severity="ok",
            )

            modal_status_banner(
                title="Ad-hoc analysis — configuration limitations",
                detail=(
                    "Without a predefined Sensor Map, the Modal module uses the "
                    "channel configuration of the capture file "
                    "(.tdms) loaded in Acquisition. The **Bar chart 2D**, "
                    "**Complexity Polar**, **AutoMAC**, "
                    "**Campbell diagram** features and the entire EMA/OMA analysis "
                    "are available. The **3D Mode Shapes with arrows "
                    "over the asset** visualization requires position_3d configured in "
                    "Machinery Library — not available in ad-hoc mode."
                ),
                severity="info",
            )
        else:
            st.info(
                "👆 Enter at least the **Machine name / Tag** to "
                "enable modal analysis in ad-hoc mode."
            )

        # Salir del Tab Setup ya — no continuar con flujo de Machinery Library
        st.stop() if False else None  # noop - solo para claridad

    # ─── MODO ACTIVO REGISTRADO ──────────────────────────────────────
    # Limpiar metadata ad-hoc cuando se vuelve a modo registrado
    if st.session_state.get("modal_adhoc_meta"):
        st.session_state.pop("modal_adhoc_meta", None)

    # Selector real de activos — lee Machinery Library
    from core.instance_state import list_instances, get_instance

    try:
        _all_insts = list_instances() or []
    except Exception:
        _all_insts = []

    if not _all_insts:
        modal_empty_state(
            icon="📭",
            title="No registered assets",
            description=(
                "The Modal Analysis module runs on assets defined in "
                "Machinery Library. Create an asset first using the "
                "'Create asset' wizard in the sidebar, configure the sensors and "
                "come back here. Or switch to "
                "**Ad-hoc analysis** mode above to register a one-off machine without "
                "having to create it in Machinery Library."
            ),
            cta_label="Or use Ad-hoc mode above ↑",
            norm_ref="",
        )
    else:
        # Construir opciones del selector con label legible
        def _inst_label(entry):
            tag = entry.get("tag") or entry.get("instance_id", "")
            location = entry.get("location", "")
            if location:
                return f"{tag.upper()} · {location}"
            return tag.upper()

        _options = ["(seleccionar)"] + [e.get("instance_id", "") for e in _all_insts]
        _labels_by_id = {
            e.get("instance_id", ""): _inst_label(e) for e in _all_insts
        }

        # Aplicar el activo pendiente restaurado desde disco (si existe y es
        # una opción válida), antes de instanciar el selectbox.
        _pending_asset = st.session_state.pop("_modal_pending_asset", "")
        if ("modal_inst" not in st.session_state
                and _pending_asset and _pending_asset in _options):
            st.session_state["modal_inst"] = _pending_asset

        col_sel, col_meta = st.columns([3, 2])
        with col_sel:
            picked_id = st.selectbox(
                "Asset under analysis",
                _options,
                format_func=lambda x: _labels_by_id.get(x, x) if x != "(seleccionar)" else x,
                key="modal_inst",
            )
            # Persistir la selección para restaurarla tras recarga (v3.31.438).
            if picked_id and picked_id != "(seleccionar)":
                _save_sel(setup_mode=setup_mode, asset_id=picked_id, adhoc={})

        with col_meta:
            st.caption(
                f"📦 **{len(_all_insts)} assets** available in Machinery Library"
            )

        # Si selecciona un activo, mostrar preview con sensores y geometría
        if picked_id and picked_id != "(seleccionar)":
            inst = get_instance(picked_id)
            if inst is None:
                modal_status_banner(
                    title="Asset not found",
                    detail=f"Could not load the information for '{picked_id}'.",
                    severity="fail",
                )
            else:
                # Hero secundario del activo seleccionado
                _location = getattr(inst, "location", "") or "(no location)"
                _profile = getattr(inst, "profile_key", "") or "(generic)"
                _serial = getattr(inst, "serial_number", "") or "—"

                st.markdown(
                    f"""
                    <div style="background:#F4F7FB; border-left:4px solid #1AAEE5;
                                 padding:14px 18px; border-radius:6px;
                                 margin-top:12px;">
                        <div style="font-size:11px; font-weight:700; color:#0F7FB0;
                                     letter-spacing:0.12em; text-transform:uppercase;">
                            Selected asset
                        </div>
                        <div style="font-size:18px; font-weight:800; color:#0F1E3D;
                                     margin:4px 0;">
                            {inst.tag or inst.instance_id}
                        </div>
                        <div style="font-size:12px; color:#6B7280;">
                            📍 {_location} &nbsp;·&nbsp;
                            🏷 Model: {_profile} &nbsp;·&nbsp;
                            S/N: {_serial}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Análisis de sensores configurados
                sensors = list(inst.sensors or [])
                if not sensors:
                    modal_status_banner(
                        title="Asset without sensor map in Machinery Library",
                        detail=(
                            "This asset has no sensor map registered in "
                            "Machinery Library (that map is used for Live "
                            "Monitoring). **It is NOT required for modal "
                            "analysis:** define your sensors and their DOF below, "
                            "in «Asset 3D geometry», and save them as "
                            "«📦 COMPLETE analysis configuration» to "
                            "reuse them. The modal test uses those sensors and "
                            "the channels of the captured file."
                        ),
                        severity="info",
                    )
                else:
                    # Contadores por tipo
                    n_accel = sum(1 for s in sensors
                                    if s.get("sensor_type", "") == "accelerometer")
                    n_prox = sum(1 for s in sensors
                                   if s.get("sensor_type", "") == "proximity")
                    n_vel = sum(1 for s in sensors
                                  if s.get("sensor_type", "") == "velocity")
                    n_other = len(sensors) - n_accel - n_prox - n_vel

                    # Contadores de configuración modal 3D
                    from core.sensor_map import has_modal_3d_config
                    n_3d_ready = sum(1 for s in sensors if has_modal_3d_config(s))
                    n_sens_only = sum(
                        1 for s in sensors
                        if s.get("sensitivity_mv_per_eu") is not None
                        and not has_modal_3d_config(s)
                    )

                    modal_section_header(
                        title="Configured sensors",
                        subtitle="Distribution by type and readiness for 3D modal analysis",
                        norm_ref="ISO 7626-6 sec. 6.2",
                        icon="📡",
                    )
                    modal_kpi_row([
                        (str(len(sensors)), "Total sensors",
                         "registered in Sensor Map", "navy"),
                        (str(n_accel), "Accelerometers",
                         "Typical IEPE accelerometer 100 mV/g", "cyan"),
                        (str(n_prox), "Proximity",
                         "Typical proximity probe 200 mV/mil", "amber"),
                        (str(n_3d_ready), "Modal 3D ready",
                         "with position_3d + DOF", "green"),
                    ])

                    # Status del activo para modal
                    if n_3d_ready == len(sensors):
                        modal_status_banner(
                            title="Asset fully configured for 3D modal analysis",
                            detail=(
                                f"All {len(sensors)} sensors have position_3d + "
                                "dof_direction defined. Animated 3D mode shapes available."
                            ),
                            severity="ok",
                        )
                    elif n_3d_ready > 0:
                        modal_status_banner(
                            title=f"Partial configuration — {n_3d_ready}/{len(sensors)} sensors 3D-ready",
                            detail=(
                                f"{len(sensors) - n_3d_ready} sensors lack position_3d "
                                "or dof_direction. Bar chart 2D and MAC available for all, "
                                "3D arrows only for the configured sensors."
                            ),
                            severity="warning",
                        )
                    else:
                        modal_status_banner(
                            title="No sensors with 3D modal configuration",
                            detail=(
                                "To enable 3D mode shape visualization, complete "
                                "the 'Modal configuration' expander of each sensor in the "
                                "Machinery Library wizard. In the meantime, Bar chart 2D "
                                "and AutoMAC remain available."
                            ),
                            severity="info",
                        )

                    # Tabla de sensores con tag de configuración
                    with st.expander(f"▸ Sensor list ({len(sensors)})",
                                       expanded=False):
                        import pandas as pd
                        _rows = []
                        for s in sensors:
                            _rows.append({
                                "Plane": s.get("plane_label", "—"),
                                "Type": s.get("sensor_type", "—"),
                                "Direction": s.get("direction", "—"),
                                "Unit": s.get("unit_native", "—"),
                                "Sens (mV/EU)": (
                                    f"{s.get('sensitivity_mv_per_eu'):.1f}"
                                    if s.get("sensitivity_mv_per_eu") is not None
                                    else "—"
                                ),
                                "Coupling": s.get("coupling") or "—",
                                "3D config": "✓" if has_modal_3d_config(s) else "—",
                            })
                        st.dataframe(pd.DataFrame(_rows),
                                      use_container_width=True, hide_index=True)

        else:
            st.info(
                "👆 Select an asset above to view its sensor configuration "
                "and validate readiness for modal analysis."
            )

    # =================================================================
    # Sub-sección: Editor de Geometría 3D — visualización profesional
    # =================================================================
    st.divider()
    modal_section_header(
        title="Asset 3D geometry",
        subtitle=(
            "Draw the mechanical train and position the sensors with their "
            "DOF direction. The geometry is used as visual support for the mode "
            "shapes in Tab 5. Persisted per asset or session-only in ad-hoc mode."
        ),
        norm_ref="ISO 7626-6 sec. 6 · DOF and spatial orientation documented",
    )

    from core.modal.geometry_3d import (
        ModalGeometry, GeometryBlock, GeometrySensor,
        TEMPLATES, build_geometry_figure,
        save_geometry, load_geometry,
        autosave_geometry, load_autosave,
    )

    # Resolver asset_id para persistencia (None si ad-hoc)
    _geom_asset_id = ""
    _adhoc_for_geom = st.session_state.get("modal_adhoc_meta")
    _inst_key_for_geom = st.session_state.get("modal_inst", "")
    if not _adhoc_for_geom and _inst_key_for_geom and _inst_key_for_geom != "(seleccionar)":
        _geom_asset_id = _inst_key_for_geom

    # CONFIG COMPLETA POR ACTIVO (v3.31.453). Cada activo registrado tiene su
    # propia configuración completa (geometría + sensores + parámetros del
    # ensayo) guardada bajo la clave 'asset__<instance_id>'. Al SELECCIONAR el
    # activo se restaura sola — sin cargar presets a mano. Antes había que
    # guardar/cargar por nombre y el usuario reportaba que "la configuración de
    # la máquina no queda guardada".
    from core.modal.analysis_preset import (
        save_preset as _save_asset_cfg, load_preset as _load_asset_cfg,
        PRESET_PARAM_KEYS as _ASSET_PARAM_KEYS,
    )
    import json as _json_asset

    def _asset_cfg_key(aid: str) -> str:
        return f"asset__{aid}"

    def _apply_asset_params(_params: dict) -> None:
        """Restaura los parámetros del ensayo en session_state."""
        for _k, _v in (_params or {}).items():
            if _k in _ASSET_PARAM_KEYS and _k not in st.session_state:
                st.session_state[_k] = _v

    # Si cambió el activo seleccionado, recargar SU configuración completa.
    if _geom_asset_id and st.session_state.get("_modal_cfg_asset") != _geom_asset_id:
        _cfg = _load_asset_cfg(_asset_cfg_key(_geom_asset_id))
        if _cfg:
            try:
                if _cfg.get("geometry"):
                    st.session_state["modal_geometry"] = ModalGeometry.from_dict(
                        _json_asset.loads(_cfg["geometry"]))
                _apply_asset_params(_cfg.get("params") or {})
                st.session_state["_modal_cfg_restored"] = _geom_asset_id
            except Exception:  # noqa: BLE001
                pass
        st.session_state["_modal_cfg_asset"] = _geom_asset_id

    # Cargar geometría existente o inicializar.
    # Prioridad: config completa del activo → geometría guardada del activo →
    # autosave de la última sesión → vacío.
    if "modal_geometry" not in st.session_state:
        _loaded = load_geometry(_geom_asset_id) if _geom_asset_id else None
        if _loaded:
            st.session_state["modal_geometry"] = _loaded
        else:
            _auto = load_autosave()
            if _auto and (_auto.blocks or _auto.sensors):
                st.session_state["modal_geometry"] = _auto
                st.session_state["_geom_restored_autosave"] = True
            else:
                st.session_state["modal_geometry"] = TEMPLATES["empty"]()

    geom: ModalGeometry = st.session_state["modal_geometry"]

    if st.session_state.pop("_modal_cfg_restored", None):
        st.success(
            f"✅ Complete configuration of **{_geom_asset_id}** restored "
            f"automatically: {len(geom.blocks)} block(s), "
            f"{len(geom.sensors)} sensor(s) and the test parameters. "
            "You do not need to reconfigure anything.", icon="📦")

    # Autosave de la geometría de trabajo (con detección de cambios para no
    # reescribir en cada rerun). Restaura sola si el usuario recarga la página.
    try:
        _geom_json_now = geom.to_json()
        if st.session_state.get("_geom_autosave_last") != _geom_json_now:
            autosave_geometry(geom)
            st.session_state["_geom_autosave_last"] = _geom_json_now
    except Exception:  # noqa: BLE001
        pass

    if st.session_state.pop("_geom_restored_autosave", False):
        st.caption("↩️ The last configuration you were working on was "
                   "restored. Save it with a name below if you want to reuse it.")

    # --- Guía por etapas: feedback inmediato del avance de configuración ----
    # (v3.31.438) Flujo más claro: el usuario ve en qué paso va y cuál sigue.
    _step_asset = bool(_geom_asset_id) or bool(
        st.session_state.get("modal_adhoc_meta"))
    _step_blocks = len(geom.blocks) > 0
    _step_sensors = len(geom.sensors) > 0
    _step_ready = _step_blocks and _step_sensors

    def _chk(ok: bool) -> str:
        return "✅" if ok else "⬜"

    st.markdown(
        f"**Configuration progress:**  "
        f"{_chk(_step_asset)} 1· Asset  →  "
        f"{_chk(_step_blocks)} 2· Geometry ({len(geom.blocks)} blocks)  →  "
        f"{_chk(_step_sensors)} 3· Sensors ({len(geom.sensors)})  →  "
        f"{_chk(_step_ready)} 4· Ready to capture")
    if not _step_blocks:
        st.caption("➡️ **Next step:** add at least one block (the shape "
                   "of the machine) in «➕ Add / edit block». The 3D model "
                   "above updates when you apply each change.")
    elif not _step_sensors:
        st.caption("➡️ **Next step:** add your sensors with their position "
                   "(x, y, z) and DOF direction in «➕ Add / edit sensor».")
    else:
        st.caption("✅ **Geometry ready.** Review the 3D model above "
                   "(sensors in green, DOF arrows in orange). Save it with "
                   "a name below to reuse it in future campaigns.")

    # ----- Toolbar -----
    # Callback de auto-aplicación del template al cambiar el selectbox
    def _apply_template_on_change():
        choice = st.session_state.get("geom_tpl_choice", "(mantener actual)")
        if choice != "(mantener actual)" and choice in TEMPLATES:
            st.session_state["modal_geometry"] = TEMPLATES[choice]()
            st.session_state["_geom_just_applied"] = choice

    col_t1, col_t2, col_t3 = st.columns([3, 1, 1])
    with col_t1:
        tpl_choice = st.selectbox(
            "Load template (applied automatically on selection)",
            options=["(mantener actual)", "empty",
                      "motor_compressor", "turbine_generator", "pump_motor"],
            format_func=lambda k: {
                "(mantener actual)": "— Keep current configuration —",
                "empty": "Empty — build from scratch (no shaft/couplings)",
                "motor_compressor": "Motor + Compressor (6 sensors)",
                "turbine_generator": "Turbine + Generator · LM6000+Brush (6 sensors)",
                "pump_motor": "Pump + Motor (4 sensors)",
            }.get(k, k),
            key="geom_tpl_choice",
            on_change=_apply_template_on_change,
        )
    with col_t2:
        if st.button("💾 Save EVERYTHING for this asset", key="geom_save",
                       use_container_width=True, type="primary",
                       disabled=not _geom_asset_id,
                       help=("Select a registered asset above to be able to "
                             "save its configuration."
                             if not _geom_asset_id else
                             "Saves geometry + sensors + test parameters "
                             "for THIS asset. When you select it again, "
                             "everything is restored automatically.")):
            geom.asset_id = _geom_asset_id
            try:
                save_geometry(geom)  # compat: geometría por activo
                # Config COMPLETA atada al activo (geometría + parámetros).
                _params_now = {k: st.session_state.get(k)
                               for k in _ASSET_PARAM_KEYS if k in st.session_state}
                _save_asset_cfg(_asset_cfg_key(_geom_asset_id),
                                geom.to_json(), _params_now)
                st.success(
                    f"✅ Complete configuration of **{_geom_asset_id}** saved: "
                    f"{len(geom.blocks)} block(s), {len(geom.sensors)} sensor(s) "
                    f"and {len(_params_now)} parameter(s). The next time you "
                    "select this asset, it restores by itself.", icon="📦")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error while saving: {exc}")
    with col_t3:
        if st.button("⬇ Export JSON", key="geom_export",
                       use_container_width=True,
                       help="Download the geometry as JSON to reuse "
                            "or share it. Works in any mode."):
            st.session_state["_geom_export_ready"] = geom.to_json()

    # Feedback de aplicación del template
    if st.session_state.pop("_geom_just_applied", None):
        _label_map = {
            "motor_compressor": "Motor + Compressor",
            "turbine_generator": "Turbine + Generator (LM6000 + Brush)",
            "pump_motor": "Pump + Motor",
        }
        _applied = _label_map.get(
            st.session_state.get("geom_tpl_choice", ""), "Template"
        )
        st.success(
            f"✓ Template **{_applied}** applied · "
            f"{len(geom.blocks)} blocks · {len(geom.sensors)} sensors. "
            "Edit the names and positions below if you need to adjust it to "
            "your real asset."
        )

    if st.session_state.get("_geom_export_ready"):
        st.download_button(
            "Download geometry.json",
            data=st.session_state["_geom_export_ready"],
            file_name=f"{geom.asset_id or 'adhoc'}_geometry.json",
            mime="application/json",
            key="geom_download_btn",
        )

    if not _geom_asset_id:
        st.caption(
            "Ad-hoc mode — the geometry lives only in this session. "
            "To persist it across sessions, save it with a name below, "
            "select a registered asset in the Setup Tab, or use "
            "**⬇ Export JSON** to save it externally."
        )

    # --- Guardar / cargar CONFIGURACIONES PERSONALIZADAS (también en ad-hoc) ---
    # Antes solo se podían guardar geometrías de activos registrados; las que
    # creaba el usuario se perdían. Ahora se pueden guardar con nombre y reusar.
    with st.expander("💾 Save / load ONLY the geometry "
                     "(for the complete config use the 📦 panel below)",
                     expanded=False):
        from core.modal.geometry_3d import (
            list_geometries as _list_geoms, delete_geometry as _del_geom,
        )
        _saved_all = _list_geoms()
        _custom_saved = [s for s in _saved_all if s.startswith("custom_")]
        _cc1, _cc2 = st.columns(2)
        with _cc1:
            st.markdown("**Save the current geometry**")
            _save_name = st.text_input(
                "Configuration name", key="geom_custom_name",
                placeholder="e.g. Parex compressor skid")
            if st.button("Save configuration", key="geom_custom_save",
                         disabled=not _save_name.strip(),
                         use_container_width=True):
                _slug = "".join(c if c.isalnum() or c in "-_" else "_"
                                for c in _save_name.strip().lower())
                geom.asset_id = "custom_" + _slug
                geom.name = _save_name.strip() or geom.name
                try:
                    save_geometry(geom)
                    st.success(
                        f"✓ Saved as **{_save_name}**. You can now "
                        f"reload it at any time here on the right.")
                except Exception as _e:  # noqa: BLE001
                    st.error(f"Could not save: {_e}")
        with _cc2:
            st.markdown("**Load / delete a saved one**")
            if _custom_saved:
                _pick = st.selectbox(
                    "Saved configuration",
                    ["(elegir)"] + _custom_saved,
                    format_func=lambda s: (s[len("custom_"):].replace("_", " ")
                                            if s.startswith("custom_") else s),
                    key="geom_custom_load_pick")
                _lc1, _lc2 = st.columns(2)
                if _lc1.button("Load", key="geom_custom_load_btn",
                               disabled=_pick == "(elegir)",
                               use_container_width=True):
                    _g = load_geometry(_pick)
                    if _g:
                        st.session_state["modal_geometry"] = _g
                        st.rerun()
                    else:
                        st.error("Could not load that configuration.")
                if _lc2.button("Delete", key="geom_custom_del_btn",
                               disabled=_pick == "(elegir)",
                               use_container_width=True):
                    _del_geom(_pick)
                    st.rerun()
            else:
                st.caption("No custom configurations saved yet.")

    # --- Configuración COMPLETA del análisis (geometría + sensores + parámetros)
    # Guarda TODO el ensayo como una plantilla reutilizable de un clic. Resuelve
    # el pedido de no reconfigurar sensores/parámetros en cada ensayo del mismo
    # activo (feedback de campo v3.31.440).
    # expanded=True (v3.31.451): estaba colapsado y los usuarios no lo
    # encontraban ("no hay forma clara de guardar la config completa"), aunque
    # existía. Se abre por defecto para que sea evidente.
    with st.expander("📦 SAVE / LOAD the COMPLETE asset configuration "
                     "(geometry + sensors + test parameters) — 1 click",
                     expanded=True):
        import json as _json_preset
        from core.modal.analysis_preset import (
            save_preset as _save_preset, load_preset as _load_preset,
            list_presets as _list_presets, delete_preset as _del_preset,
            PRESET_PARAM_KEYS as _PRESET_KEYS,
        )
        st.caption(
            "A preset saves **the entire test in one file**: the geometry "
            "(blocks + sensors + DOF) and the test parameters (f_min/f_max, "
            "duration, prominence, RPM, averages and the acquisition unit's "
            "channel map). Load it later with **one click** to repeat the same test "
            "without reconfiguring anything. Ideal for assets you measure periodically. "
            "No Machinery Library required — everything is saved from here.")

        _pc1, _pc2 = st.columns(2)
        with _pc1:
            st.markdown("**Save complete configuration**")
            _preset_name = st.text_input(
                "Preset name", key="modal_preset_name",
                placeholder="e.g. TES3 · OMA bearings · standard setup")
            if st.button("💾 Save complete preset", key="modal_preset_save",
                         disabled=not _preset_name.strip(),
                         use_container_width=True):
                _params = {k: st.session_state.get(k)
                           for k in _PRESET_KEYS if k in st.session_state}
                try:
                    _save_preset(_preset_name, geom.to_json(), _params)
                    st.success(
                        f"✓ Preset **{_preset_name}** saved: "
                        f"{len(geom.blocks)} block(s), {len(geom.sensors)} "
                        f"sensor(s) and {len(_params)} parameter(s). "
                        "Reload it whenever you want here on the right.")
                except Exception as _e:  # noqa: BLE001
                    st.error(f"Could not save: {_e}")
        with _pc2:
            st.markdown("**Load / delete a preset**")
            _presets = _list_presets()
            if _presets:
                _psel = st.selectbox("Saved preset", ["(elegir)"] + _presets,
                                     key="modal_preset_pick")
                _pb1, _pb2 = st.columns(2)
                if _pb1.button("📂 Load preset", key="modal_preset_load",
                               disabled=_psel == "(elegir)",
                               use_container_width=True):
                    _pre = _load_preset(_psel)
                    if _pre:
                        # 1) Geometría (bloques + sensores + DOF)
                        if _pre.get("geometry"):
                            try:
                                st.session_state["modal_geometry"] = \
                                    ModalGeometry.from_dict(
                                        _json_preset.loads(_pre["geometry"]))
                            except Exception:  # noqa: BLE001
                                pass
                        # 2) Parámetros del ensayo → session_state (se aplican
                        #    al re-render de las pestañas Adquisición/OMA/EMA).
                        for _k, _v in (_pre.get("params") or {}).items():
                            if _k in _PRESET_KEYS:
                                st.session_state[_k] = _v
                        st.success(
                            f"✓ Preset **{_psel}** loaded (geometry + "
                            "parameters). Go to Acquisition / OMA / EMA: the "
                            "values have been restored.")
                        st.rerun()
                    else:
                        st.error("Could not load the preset.")
                if _pb2.button("🗑 Delete", key="modal_preset_del",
                               disabled=_psel == "(elegir)",
                               use_container_width=True):
                    _del_preset(_psel)
                    st.rerun()
            else:
                st.caption("No presets saved yet. Configure your test and "
                           "save it above with a name.")

    # ----- Preview Plotly 3D -----
    # v3.31.208 — Envuelto en try/except defensivo. Si geom está vacía o
    # malformada (típico al entrar al módulo sin seleccionar activo), antes
    # crasheaba toda la página y los demás tabs (Adquisición, EMA, OMA)
    # nunca renderizaban. Ahora muestra mensaje amigable y sigue.
    try:
        fig_geom = build_geometry_figure(geom)
        st.plotly_chart(
            fig_geom, use_container_width=True,
            config={
                "displaylogo": False,
                # En una escena 3D el botón "Zoom" no acerca (solo resetea la
                # cámara) y hay dos botones de reset redundantes (Home y Reset
                # camera). Quitamos el Zoom y el reset duplicado; se rota/pan
                # con el mouse y el zoom con la rueda.
                "modeBarButtonsToRemove": ["zoom3d", "resetCameraLastSave3d"],
            })
        st.caption(
            "🖱️ Zoom with the mouse wheel · drag to rotate · "
            "the 🏠 button resets the view.")
    except Exception as _exc_geom:  # noqa: BLE001
        st.warning(
            "⚠ Could not render the 3D geometry preview. "
            "Possible causes:\n"
            "1. You have not selected an asset or loaded a geometry.\n"
            "2. The geometry has no blocks defined yet.\n"
            "3. There is an invalid value in a block.\n\n"
            "**Solutions:** go to the geometry section above and "
            "select a registered asset, or add blocks manually "
            "in the editing section below. The other tabs (Acquisition, "
            "EMA, OMA) are still available for use."
        )
        with st.expander("Technical error detail (for support)"):
            st.code(f"{type(_exc_geom).__name__}: {_exc_geom}", language="text")

    # ----- Editor de bloques + sensores -----
    col_edit_b, col_edit_s = st.columns(2)

    with col_edit_b:
        st.markdown("**Train sections (blocks)**")
        if geom.blocks:
            import pandas as pd
            df_b = pd.DataFrame([
                {"Name": b.name, "Shape": b.shape, "Layer": b.kind,
                 "x_start": b.x_start, "x_end": b.x_end,
                 "R / hw,hh": (
                     f"{b.radius:.0f}" if b.shape == "cylinder"
                     else f"{b.half_width:.0f}, {b.half_height:.0f}"
                 )}
                for b in geom.blocks
            ])
            st.dataframe(df_b, hide_index=True, use_container_width=True)

        with st.expander("➕ Add / edit block", expanded=False):
            _action_b = st.radio("Action", ["Agregar nuevo", "Editar existente",
                                                "Eliminar"],
                                  horizontal=True, key="geom_block_action",
                                  format_func=lambda o: {
                                      "Agregar nuevo": "Add new",
                                      "Editar existente": "Edit existing",
                                      "Eliminar": "Delete",
                                  }.get(o, o))
            if _action_b == "Editar existente" and geom.blocks:
                _idx_b = st.selectbox("Block to edit",
                                        options=list(range(len(geom.blocks))),
                                        format_func=lambda i: geom.blocks[i].name,
                                        key="geom_block_edit_idx")
                _b_default = geom.blocks[_idx_b]
            elif _action_b == "Eliminar" and geom.blocks:
                _idx_b = st.selectbox("Block to delete",
                                        options=list(range(len(geom.blocks))),
                                        format_func=lambda i: geom.blocks[i].name,
                                        key="geom_block_del_idx")
                if st.button("Confirm deletion", key="geom_block_del_btn"):
                    geom.blocks.pop(_idx_b)
                    st.rerun()
                _b_default = None
            else:
                _b_default = GeometryBlock(id=f"b{len(geom.blocks)+1}",
                                            name="New block")

            if _b_default is not None:
                # Sufijo de key ligado al bloque/acción seleccionado. Sin esto,
                # Streamlit conserva el valor viejo del widget al cambiar de
                # bloque (el value= se ignora si la key ya tiene estado) → los
                # campos "no se actualizaban" al editar otro bloque.
                _bsfx = (f"e{_idx_b}"
                         if _action_b == "Editar existente" and geom.blocks
                         else "new")
                c1, c2 = st.columns(2)
                with c1:
                    _nm = st.text_input("Name", value=_b_default.name,
                                          key=f"geom_b_name_{_bsfx}")
                    _shape = st.selectbox(
                        "Shape", ["cylinder", "box"],
                        index=0 if _b_default.shape == "cylinder" else 1,
                        format_func=lambda s: {
                            "cylinder": "Cylinder (shaft / rotor / round casing)",
                            "box": "Rectangle / plate (skid, fan cooler, base, casing)",
                        }.get(s, s),
                        help="The 'Rectangle/plate' is a box: for a skid or "
                             "fan cooler use a large half_width and a small "
                             "half_height (it becomes flat and wide).",
                        key=f"geom_b_shape_{_bsfx}")
                    _x0 = st.number_input("x_start", value=float(_b_default.x_start),
                                            step=10.0, key=f"geom_b_x0_{_bsfx}")
                    _x1 = st.number_input("x_end", value=float(_b_default.x_end),
                                            step=10.0, key=f"geom_b_x1_{_bsfx}")
                with c2:
                    if _shape == "cylinder":
                        _r = st.number_input("Radius", value=float(_b_default.radius),
                                               step=10.0, min_value=1.0,
                                               key=f"geom_b_r_{_bsfx}")
                        _hw, _hh = _r, _r
                    else:
                        _hw = st.number_input("half_width (Y)",
                                                value=float(_b_default.half_width),
                                                step=10.0, min_value=1.0,
                                                key=f"geom_b_hw_{_bsfx}")
                        _hh = st.number_input("half_height (Z)",
                                                value=float(_b_default.half_height),
                                                step=10.0, min_value=1.0,
                                                key=f"geom_b_hh_{_bsfx}")
                        _r = max(_hw, _hh)
                    _color = st.color_picker("Color", value=_b_default.color,
                                                key=f"geom_b_color_{_bsfx}")
                    _op = st.slider("Opacity", 0.1, 1.0, float(_b_default.opacity),
                                      0.05, key=f"geom_b_op_{_bsfx}")
                    _kind_opts = ["casing", "shaft", "coupling"]
                    _kind = st.selectbox(
                        "Deformation layer",
                        _kind_opts,
                        index=_kind_opts.index(
                            _b_default.kind if _b_default.kind in _kind_opts
                            else "casing"
                        ),
                        key=f"geom_b_kind_{_bsfx}",
                        help=("casing: deforms with accels · shaft: deforms "
                              "with proximity probes · coupling: static or interpolated"),
                    )

                if st.button("✓ Apply to block", key="geom_b_apply",
                                use_container_width=True):
                    new_b = GeometryBlock(
                        id=_b_default.id, name=_nm, shape=_shape,
                        x_start=_x0, x_end=_x1,
                        radius=_r, half_width=_hw, half_height=_hh,
                        color=_color, opacity=_op, kind=_kind,
                    )
                    if _action_b == "Editar existente":
                        geom.blocks[_idx_b] = new_b
                    else:
                        geom.blocks.append(new_b)
                    st.rerun()

    with col_edit_s:
        st.markdown("**Sensors with DOF direction**")
        if geom.sensors:
            import pandas as pd
            df_s = pd.DataFrame([
                {"Name": s.name, "Type": s.sensor_type,
                 "Mounting": s.effective_mounting(),
                 "x": s.x, "y": s.y, "z": s.z, "DOF": s.dof}
                for s in geom.sensors
            ])
            st.dataframe(df_s, hide_index=True, use_container_width=True)

        with st.expander("➕ Add / edit sensor", expanded=False):
            _action_s = st.radio("Action", ["Agregar nuevo", "Editar existente",
                                                "Eliminar"],
                                  horizontal=True, key="geom_sensor_action",
                                  format_func=lambda o: {
                                      "Agregar nuevo": "Add new",
                                      "Editar existente": "Edit existing",
                                      "Eliminar": "Delete",
                                  }.get(o, o))
            if _action_s == "Editar existente" and geom.sensors:
                _idx_s = st.selectbox("Sensor to edit",
                                        options=list(range(len(geom.sensors))),
                                        format_func=lambda i: geom.sensors[i].name,
                                        key="geom_sensor_edit_idx")
                _s_default = geom.sensors[_idx_s]
            elif _action_s == "Eliminar" and geom.sensors:
                _idx_s = st.selectbox("Sensor to delete",
                                        options=list(range(len(geom.sensors))),
                                        format_func=lambda i: geom.sensors[i].name,
                                        key="geom_sensor_del_idx")
                if st.button("Confirm deletion", key="geom_sensor_del_btn"):
                    geom.sensors.pop(_idx_s)
                    st.rerun()
                _s_default = None
            else:
                _s_default = GeometrySensor(id=f"s{len(geom.sensors)+1}",
                                              name=f"S{len(geom.sensors)+1}")

            if _s_default is not None:
                # Sufijo de key ligado al sensor/acción (mismo motivo que en
                # bloques: evita que el value= se ignore al cambiar de sensor).
                _ssfx = (f"e{_idx_s}"
                         if _action_s == "Editar existente" and geom.sensors
                         else "new")
                c1, c2 = st.columns(2)
                with c1:
                    _snm = st.text_input("Name", value=_s_default.name,
                                           key=f"geom_s_name_{_ssfx}")
                    _styp = st.selectbox("Type",
                                           ["accelerometer", "proximity", "velocity"],
                                           index=["accelerometer", "proximity", "velocity"].index(
                                               _s_default.sensor_type
                                               if _s_default.sensor_type in
                                                  ["accelerometer", "proximity", "velocity"]
                                               else "accelerometer"),
                                           key=f"geom_s_type_{_ssfx}")
                    _sdof = st.selectbox("DOF",
                                           ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
                                           index=["+X", "-X", "+Y", "-Y", "+Z", "-Z"].index(
                                               _s_default.dof if _s_default.dof
                                               in ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
                                               else "+Y"),
                                           key=f"geom_s_dof_{_ssfx}")
                with c2:
                    _sx = st.number_input("x", value=float(_s_default.x), step=10.0,
                                            key=f"geom_s_x_{_ssfx}")
                    _sy = st.number_input("y", value=float(_s_default.y), step=10.0,
                                            key=f"geom_s_y_{_ssfx}")
                    _sz = st.number_input("z", value=float(_s_default.z), step=10.0,
                                            key=f"geom_s_z_{_ssfx}")
                    _mnt_opts = ["(auto)", "casing", "shaft_proximity"]
                    _cur_mnt = _s_default.mounting if _s_default.mounting in _mnt_opts else "(auto)"
                    _mnt_sel = st.selectbox(
                        "Mounting (what it measures)",
                        _mnt_opts,
                        index=_mnt_opts.index(_cur_mnt),
                        key=f"geom_s_mounting_{_ssfx}",
                        help=("Auto = inferred from type (accel/vel → casing, "
                              "proximity → shaft_proximity). Manual override if "
                              "you have a special case."),
                    )
                    _mnt_final = "" if _mnt_sel == "(auto)" else _mnt_sel

                if st.button("✓ Apply to sensor", key="geom_s_apply",
                                use_container_width=True):
                    new_s = GeometrySensor(
                        id=_s_default.id, name=_snm, sensor_type=_styp,
                        x=_sx, y=_sy, z=_sz, dof=_sdof,
                        mounting=_mnt_final,
                    )
                    if _action_s == "Editar existente":
                        geom.sensors[_idx_s] = new_s
                    else:
                        geom.sensors.append(new_s)
                    st.rerun()


# ---------------------------------------------------------------------
# Tab 2 — Adquisición
# ---------------------------------------------------------------------
if _active_modal_tab == "📥 Adquisición":
    st.subheader("Data acquisition")
    st.caption("Three paths: live capture from the acquisition unit, import a pre-captured file, or import legacy FRFs.")

    acq_mode = st.radio(
        "Data source",
        [
            "📡 Live capture with acquisition unit",
            "📁 Import existing capture file",
            "🔄 Import legacy data (.txt)",
        ],
        horizontal=True,
        key="acq_mode_radio",
    )

    # -------- módulo de adquisición live --------
    if acq_mode.startswith("📡"):
        st.markdown("**Watermelon capture configuration**")

        # --- Selector de modo (gobierna el resto del formulario) ---
        ni_mode_sel = st.selectbox(
            "Test mode",
            ["EMA — Impact Hammer", "OMA — Continuous"],
            key="ni_mode",
            help=("EMA: impacts with an instrumented hammer, requires ≥3 averages. "
                  "OMA: continuous recording under operational conditions, "
                  "requires ≥ 2000 × T_low per Brincker & Ventura 2015."),
        )
        is_oma = ni_mode_sel.startswith("OMA")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Sample rate (Hz)", value=5120, step=1024, key="ni_fs",
                              min_value=1024, max_value=51200,
                              help="The system accepts up to 51.2 kS/s/ch. "
                                   "Typical: 5120 Hz (useful band 0–2 kHz).")

        # --- Bifurcación EMA vs OMA con tiempos normativos ---
        # Usamos keys separados por modo (ni_fn_low_oma, ni_dur_oma, ni_dur_ema,
        # ni_avg_ema) para que el switch entre modos no cause crashes por
        # valores fuera del rango del widget del modo opuesto.
        if is_oma:
            # OMA: regla Brincker & Ventura 2015 — T_min ≥ 2000 × T_low
            with col2:
                fn_low_hz = st.number_input(
                    "f_min of interest (Hz)",
                    value=float(st.session_state.get("ni_fn_low_oma", 5.0)),
                    min_value=0.5, max_value=200.0, step=0.5,
                    key="ni_fn_low_oma",
                    help=("Lowest natural frequency you expect to identify. "
                          "Defines the minimum capture time: "
                          "T_min ≥ 2000 / f_min (Brincker & Ventura 2015)."),
                )
            with col3:
                # T_min normativo según fn_low
                _t_min_strict = 2000.0 / max(fn_low_hz, 0.1)   # 2000 × T_low (recomendado)
                _t_min_floor  = 1000.0 / max(fn_low_hz, 0.1)   # 1000 × T_low (mínimo absoluto)
                _t_default = max(120.0, _t_min_strict)
                ni_dur = st.number_input(
                    "Duration (s)",
                    value=float(st.session_state.get("ni_dur_oma", _t_default)),
                    min_value=30.0, max_value=3600.0, step=30.0,
                    key="ni_dur_oma",
                    help=f"Recommended T_min = 2000/f_min = {_t_min_strict:.0f} s. "
                         f"Absolute T_min = 1000/f_min = {_t_min_floor:.0f} s.",
                )
            # avg no aplica para OMA
            ni_avg = 1

            # --- Diagnóstico normativo OMA ---
            if ni_dur < _t_min_floor:
                modal_status_banner(
                    title=f"Insufficient duration · {ni_dur:.0f} s < absolute T_min {_t_min_floor:.0f} s",
                    detail=(
                        f"For f_min = {fn_low_hz:.1f} Hz, the standard requires at least "
                        f"**1000 × T_low = {_t_min_floor:.0f} s**, recommended "
                        f"**2000 × T_low = {_t_min_strict:.0f} s** "
                        "(Brincker & Ventura 2015 · ISO 18649). With less time, "
                        "the FDD loses spectral resolution and the damping ratios "
                        "have unacceptable variance. **Increase the duration before "
                        "starting the capture.**"
                    ),
                    severity="fail",
                )
                _can_capture = False
            elif ni_dur < _t_min_strict:
                modal_status_banner(
                    title=f"Acceptable but suboptimal duration · {ni_dur:.0f} s",
                    detail=(
                        f"You meet the floor 1000 × T_low ({_t_min_floor:.0f} s) but "
                        f"you are below the recommended 2000 × T_low "
                        f"({_t_min_strict:.0f} s). The modes will be identified but "
                        "the uncertainty in damping may be high. "
                        "Raise the duration if the asset allows it."
                    ),
                    severity="warning",
                )
                _can_capture = True
            else:
                modal_status_banner(
                    title=f"Duration compliant with standard · {ni_dur:.0f} s ≥ {_t_min_strict:.0f} s",
                    detail=(
                        f"You meet 2000 × T_low for f_min = {fn_low_hz:.1f} Hz. "
                        "Brincker & Ventura 2015 · ISO 18649."
                    ),
                    severity="ok",
                )
                _can_capture = True

        else:
            # EMA: ISO 7626-5 secc. 6.3 — ≥3 promedios, duración 1–2 s por impacto
            with col2:
                ni_dur = st.number_input(
                    "Duration per impact (s)",
                    value=float(st.session_state.get("ni_dur_ema", 2.0)),
                    min_value=0.5, max_value=10.0, step=0.5,
                    key="ni_dur_ema",
                    help="Window long enough for the response to decay to < 1% "
                         "of the peak (avoids leakage). Typically 1–2 s for industrial machines.",
                )
            with col3:
                ni_avg = st.number_input(
                    "Number of impacts to average",
                    value=int(st.session_state.get("ni_avg_ema", 5)),
                    min_value=1, max_value=30, step=1,
                    key="ni_avg_ema",
                    help="ISO 7626-5 sec. 6.3: minimum 3, recommended 5–10. "
                         "More averages → better signal-to-noise ratio.",
                )
            # fn_low no aplica para EMA
            fn_low_hz = 0.0

            # --- Diagnóstico normativo EMA ---
            if ni_avg < 3:
                modal_status_banner(
                    title=f"Number of impacts {ni_avg} insufficient — standard requires ≥ 3",
                    detail=(
                        "ISO 7626-5 sec. 6.3 requires **at least 3 averages** for "
                        "a valid FRF estimate. With a single impact there is no "
                        "coherence control and the modes may be noise. "
                        "Increase to 5–10 averages before starting."
                    ),
                    severity="fail",
                )
                _can_capture = False
            elif ni_avg < 5:
                modal_status_banner(
                    title=f"Number of impacts {ni_avg} meets the minimum · recommended 5–10",
                    detail=(
                        "ISO 7626-5 sec. 6.3 allows 3 averages as a floor but "
                        "recommends 5–10 to reduce the variance of the FRF. "
                        "The post-capture coherence checklist will be more demanding."
                    ),
                    severity="warning",
                )
                _can_capture = True
            else:
                modal_status_banner(
                    title=f"EMA configuration compliant with standard · {ni_avg} averages × {ni_dur:.1f} s",
                    detail=(
                        "ISO 7626-5 sec. 6.3 met (≥ 5 averages). Estimated total "
                        f"capture: ≈ {ni_avg * ni_dur:.0f} s + waits between impacts."
                    ),
                    severity="ok",
                )
                _can_capture = True

        # --- Canales activos: grid 32 BNC con auto-discovery ---
        # v3.31.202 — Reemplaza el grid hardcoded de 4 checkboxes por una
        # tabla editable de 32 filas (1 por BNC port). Auto-detecta qué
        # módulos módulo de adquisición están instalados en la maleta y pre-popula el
        # default. Genera el comando --channels para el companion script.
        st.markdown("**Active channels · Watermelon acquisition unit (BNC 1..32)**")

        # Auto-discovery del hardware (silencioso si no hay driver NI)
        _ni_chassis = st.session_state.get("ni_chassis_name", "cDAQ1")
        _installed_slots: set = set()
        _discovery_msg = ""
        try:
            from core.modal.acq_backend import discover_acq_modules
            _modules = discover_acq_modules(_ni_chassis)
            _installed_slots = {m["slot"] for m in _modules}
            if _modules:
                _bnc_max = max(m["bnc_range"][1] for m in _modules)
                _discovery_msg = (
                    f"✓ Detected {len(_modules)} modules in the Watermelon acquisition unit "
                    f"→ BNC 1..{_bnc_max} available"
                )
            else:
                _discovery_msg = (
                    "⚠ Watermelon acquisition unit not detected. "
                    "You can configure channels for remote capture, but the "
                    "execution will run from the plant laptop via the companion."
                )
        except ImportError:
            _discovery_msg = (
                "ℹ Watermelon acquisition drivers not available on this machine (expected in Cloud mode). "
                "Configure the channels here and run the technical command from the "
                "plant laptop with the companion script."
            )
        except Exception as _exc:  # noqa: BLE001
            _discovery_msg = f"⚠ Discovery failed: {_exc}"

        st.caption(_discovery_msg)

        # Plantilla default por modo: EMA reserva BNC 1 al martillo, OMA es
        # todo acelerómetros IEPE 100 mV/g.
        import pandas as _pd
        _default_rows = []
        for _bnc in range(1, 33):
            _slot = (_bnc - 1) // 4 + 1
            _slot_installed = (_bnc <= len(_installed_slots) * 4
                                if _installed_slots
                                else True)  # si no hay discovery, mostrar todos
            if not is_oma and _bnc == 1:
                _default_rows.append({
                    "BNC": _bnc, "Slot": _slot,
                    "Habilitado": True, "Nombre": "Hammer",
                    "Coupling": "IEPE", "Sens (mV/EU)": 2.4, "Unidad": "N",
                    "HW": "✓" if _slot_installed else "—",
                })
            else:
                _default_rows.append({
                    "BNC": _bnc, "Slot": _slot,
                    "Habilitado": _bnc <= 4 if not is_oma else _bnc <= 16,
                    "Nombre": f"Ch{_bnc:02d}",
                    "Coupling": "IEPE", "Sens (mV/EU)": 100.0, "Unidad": "g",
                    "HW": "✓" if _slot_installed else "—",
                })

        # Usar session_state para preservar edits entre reruns. La key
        # depende del modo para que cambiar EMA↔OMA no se "lleve" sensores
        # configurados del otro modo.
        _grid_key = f"ni_channel_grid_{'oma' if is_oma else 'ema'}"
        if _grid_key not in st.session_state:
            st.session_state[_grid_key] = _default_rows

        _ch_df = _pd.DataFrame(st.session_state[_grid_key])

        _edited = st.data_editor(
            _ch_df,
            key=f"ni_grid_editor_{'oma' if is_oma else 'ema'}",
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",  # exactamente 32 BNC, no se pueden agregar
            column_config={
                "BNC": st.column_config.NumberColumn(
                    "BNC", help="Front BNC port of the acquisition unit (1..32)",
                    disabled=True, width="small",
                ),
                "Slot": st.column_config.NumberColumn(
                    "Slot", help="Module (1..8) inside the acquisition unit",
                    disabled=True, width="small",
                ),
                "Habilitado": st.column_config.CheckboxColumn(
                    "✓", help="Check to include this channel in the capture",
                    width="small",
                ),
                "Nombre": st.column_config.TextColumn(
                    "Sensor", help="Sensor label (e.g. 1YA, VE5807, Hammer)",
                    max_chars=20, width="medium",
                ),
                "Coupling": st.column_config.SelectboxColumn(
                    "Coupling", options=["IEPE", "AC", "DC"],
                    help="IEPE for accelerometers, AC for proximity probes, DC rare",
                    width="small",
                ),
                "Sens (mV/EU)": st.column_config.NumberColumn(
                    "Sens", help="Sensitivity: 100 mV/g accelerometer, 200 mV/mil proximity probe, 2.4 mV/N modal hammer",
                    min_value=0.1, max_value=10000.0, step=0.1, format="%.2f",
                    width="small",
                ),
                "Unidad": st.column_config.SelectboxColumn(
                    "EU", options=["g", "mil", "N", "ips", "mm/s"],
                    help="Engineering unit of the sensor",
                    width="small",
                ),
                "HW": st.column_config.TextColumn(
                    "HW", help="✓ = module installed in this slot, — = empty",
                    disabled=True, width="small",
                ),
            },
        )
        # Persistir edits para próximos reruns
        st.session_state[_grid_key] = _edited.to_dict("records")

        _enabled_rows = [r for r in st.session_state[_grid_key] if r.get("Habilitado")]
        _n_enabled = len(_enabled_rows)

        # KPI rápido + alertas
        _kpi_col1, _kpi_col2, _kpi_col3 = st.columns(3)
        _kpi_col1.metric("Enabled channels", _n_enabled, delta=f"of 32 max")
        _slots_used = sorted({r["Slot"] for r in _enabled_rows})
        _kpi_col2.metric("Required modules", len(_slots_used),
                          delta=f"slots {_slots_used}" if _slots_used else "—")
        _ram_est_mb = (_n_enabled * float(st.session_state.get("ni_dur_oma", ni_dur))
                        * float(st.session_state.get("ni_fs", 5120)) * 4) / (1024 * 1024)
        _kpi_col3.metric("Estimated streaming RAM", f"{_ram_est_mb:.0f} MB",
                          help="With TDMS streaming, RAM stays ~5 MB constant "
                               "regardless of duration (this is the total size of the "
                               "final TDMS on disk, not the RAM during capture).")

        # Validación hardware vs slots requeridos
        if _installed_slots and _slots_used:
            _missing = [s for s in _slots_used if s not in _installed_slots]
            if _missing:
                modal_status_banner(
                    title=f"⚠ Missing hardware in {len(_missing)} slot(s)",
                    detail=(
                        f"You enabled channels in slots {_missing} but those "
                        f"modules are not installed in the acquisition unit. Slots with modules: "
                        f"{sorted(_installed_slots)}. **Before capturing:** either disable "
                        f"those channels or install the missing modules."
                    ),
                    severity="fail",
                )
                _can_capture = False

        if _n_enabled == 0:
            modal_status_banner(
                title="No channels enabled",
                detail="Check at least one channel in the table to be able to capture.",
                severity="warning",
            )
            _can_capture = False
        elif not is_oma and not any(
            r.get("Habilitado") and (r.get("Coupling") == "IEPE" and r.get("Sens (mV/EU)", 100) < 10)
            for r in _enabled_rows
        ):
            modal_status_banner(
                title="EMA mode without an identifiable hammer channel",
                detail=(
                    "For impact hammer testing you need at least 1 channel with "
                    "low sensitivity (typically 2.4 mV/N for a PCB modal hammer). "
                    "Configure the hammer on BNC 1."
                ),
                severity="warning",
            )

        # Persistir flag para el bloque del comando técnico
        st.session_state["_modal_can_capture"] = _can_capture

        # Nota técnica para especialistas — accesible vía role admin/specialist.
        # Detalles de implementación NO se muestran al cliente.
        _user_role = (_user.get("role", "") or "").strip().lower()
        if _user_role in ("admin", "specialist"):
            with st.expander("▸ Capture module command (technical)",
                              expanded=False):
                st.caption(
                    "Technical reference for the operator with access to the "
                    "acquisition unit. This section is only visible to internal "
                    "users (admin/specialist)."
                )
                if not st.session_state.get("_modal_can_capture", True):
                    st.error(
                        "⚠ Configuration not compliant with the standard. Adjust the parameters "
                        "above before running the capture."
                    )
                _mode_token = "oma" if is_oma else "ema"
                _dur_key = "ni_dur_oma" if is_oma else "ni_dur_ema"
                _dur_default = 200.0 if is_oma else 2.0
                _cmd_lines = [
                    f"--mode {_mode_token}",
                    f"--chassis {_ni_chassis}",
                    f"--fs {int(st.session_state.get('ni_fs', 5120))}",
                    f"--duration {float(st.session_state.get(_dur_key, _dur_default))}",
                ]
                if is_oma:
                    _cmd_lines.append(
                        f"--fn-low {float(st.session_state.get('ni_fn_low_oma', 5.0))}"
                    )
                else:
                    _cmd_lines.append(
                        f"--averages {int(st.session_state.get('ni_avg_ema', 5))}"
                    )

                # v3.31.202 — Generar --channels desde la tabla editable
                # (antes hardcoded a 4 canales fijos). Detecta el martillo
                # (sens < 10 mV/N en EMA) y lo pone como trigger-bnc.
                _hammer_bnc = None
                for _r in _enabled_rows:
                    _coupling = _r.get("Coupling", "IEPE").upper()
                    _sens = float(_r.get("Sens (mV/EU)", 100.0))
                    _name = str(_r.get("Nombre", "")).strip() or f"Ch{_r['BNC']:02d}"
                    _bnc = int(_r["BNC"])
                    _cmd_lines.append(
                        f"--channels {_name}:{_bnc}:{_coupling}:{_sens:g}"
                    )
                    if _hammer_bnc is None and _coupling == "IEPE" and _sens < 10:
                        _hammer_bnc = _bnc

                if not is_oma and _hammer_bnc is not None:
                    _cmd_lines.insert(-len(_enabled_rows),
                                       f"--trigger-bnc {_hammer_bnc}")
                _cmd_lines.append(f"--output ./capture_{_mode_token}.tdms")

                # Stats arriba del bloque para el operador
                st.caption(
                    f"📊 {len(_enabled_rows)} channels enabled · "
                    f"{len(_slots_used)} modules required · "
                    f"{_mode_token.upper()} mode"
                )

                if _user_role == "admin":
                    st.code(" \\\n    ".join(["python capture.py"] + _cmd_lines),
                             language="bash")
                else:
                    st.code(" \\\n    ".join(_cmd_lines), language="text")

                st.caption(
                    "Copy and run this command on the plant laptop where "
                    "the acquisition unit is connected. It does NOT require internet. The resulting "
                    ".tdms is then uploaded in '📁 Import file'."
                )

                # =====================================================
                # v3.31.203 — Botón "Capturar ahora" (laptop local)
                # =====================================================
                # Alternativa al copy/paste del CLI: si nidaqmx está
                # disponible en este equipo (laptop de planta con driver
                # driver del fabricante instalado), ofrecemos correr capture() directo
                # desde Watermelon con progress bar. En Streamlit Cloud
                # driver del fabricante no existe → el bloque entero se esconde.
                _nidaqmx_available = False
                try:
                    import nidaqmx as _nidaqmx_probe  # noqa: F401
                    _nidaqmx_available = True
                except ImportError:
                    pass

                if _nidaqmx_available:
                    st.divider()
                    st.markdown(
                        "**🎙 Direct capture from Watermelon · this machine**"
                    )
                    st.caption(
                        "Alternative to the technical command: run the capture "
                        "right now from this machine without going through a terminal. "
                        "Requires a Watermelon acquisition unit connected via USB and "
                        "acquisition drivers installed (✓ detected)."
                    )

                    _capture_disabled = (
                        _n_enabled == 0
                        or not st.session_state.get("_modal_can_capture", True)
                    )

                    if st.button(
                        "🎙 Start capture now",
                        type="primary",
                        disabled=_capture_disabled,
                        key="ni_capture_now_btn",
                        help=(
                            "Runs the capture immediately. The UI is "
                            "blocked until it finishes (typically EMA ~30s, "
                            "OMA per --duration ~60-300s+). The TDMS is "
                            "saved in data/modal/captures/ with a timestamp."
                        ),
                    ):
                        from datetime import datetime as _dt
                        from core.modal.acq_backend import (
                            AcquisitionConfig as _AcqCfg,
                            ChannelConfig as _ChCfg,
                            capture as _ni_capture,
                        )

                        # Construir lista de ChannelConfig desde el grid
                        _capture_channels = []
                        for _r in _enabled_rows:
                            _capture_channels.append(_ChCfg(
                                bnc_port=int(_r["BNC"]),
                                name=str(
                                    _r.get("Nombre", f"Ch{_r['BNC']:02d}")
                                ).strip(),
                                coupling=str(
                                    _r.get("Coupling", "IEPE")
                                ).upper(),
                                sensitivity_mv_per_eu=float(
                                    _r.get("Sens (mV/EU)", 100.0)
                                ),
                                units=str(_r.get("Unidad", "g")),
                            ))

                        # Output path con timestamp
                        _captures_dir = Path("data/modal/captures")
                        _captures_dir.mkdir(parents=True, exist_ok=True)
                        _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                        _out_path = (
                            _captures_dir
                            / f"capture_{_mode_token}_{_ts}.tdms"
                        )

                        # Build config — modo real (NO simulated; eso es
                        # otra ruta en el código companion CLI)
                        _capture_cfg = _AcqCfg(
                            mode=(
                                "ema_triggered" if not is_oma
                                else "oma_continuous"
                            ),
                            sample_rate_hz=float(
                                st.session_state.get("ni_fs", 5120)
                            ),
                            duration_s=float(
                                st.session_state.get(_dur_key, _dur_default)
                            ),
                            channels=_capture_channels,
                            chassis_name=_ni_chassis,
                            trigger_channel=(
                                _hammer_bnc
                                if (not is_oma and _hammer_bnc is not None)
                                else None
                            ),
                            n_averages=int(
                                st.session_state.get("ni_avg_ema", 5)
                            ),
                            output_tdms_path=_out_path,
                        )

                        # Progress bar + callback
                        _progress_bar = st.progress(0.0, text="Starting capture...")

                        def _on_capture_progress(_p, _s):
                            try:
                                _progress_bar.progress(
                                    min(max(_p, 0.0), 1.0), text=_s
                                )
                            except Exception:
                                pass  # progress bar puede no existir si rerun

                        try:
                            with st.spinner(
                                f"Capturing {len(_capture_channels)} channels "
                                f"× {_capture_cfg.duration_s:.0f}s..."
                            ):
                                _result_path = _ni_capture(
                                    _capture_cfg,
                                    on_progress=_on_capture_progress,
                                )
                            _progress_bar.empty()
                            st.success(
                                f"✓ Capture complete: "
                                f"`{_result_path.name}` in "
                                f"`{_result_path.parent}/`"
                            )
                            st.session_state["_last_capture_path"] = str(
                                _result_path
                            )
                        except Exception as _exc:  # noqa: BLE001
                            _progress_bar.empty()
                            st.error(f"✗ Capture failed: {_exc}")
                            st.caption(
                                "Check that the acquisition unit is connected and the "
                                "modules installed. For diagnostics run "
                                "`python scripts/capture_companion/capture.py "
                                "--list-modules` in the CLI."
                            )

                    # Auto-load: si hay captura reciente, ofrecer botón
                    # para procesarla en este análisis sin tener que ir
                    # al uploader.
                    _last_capture = st.session_state.get(
                        "_last_capture_path"
                    )
                    if _last_capture and Path(_last_capture).exists():
                        if st.button(
                            "📥 Load last capture into this analysis",
                            key="auto_load_captured_tdms",
                            help=(
                                f"Processes {Path(_last_capture).name} "
                                "as if you had uploaded it via '📁 "
                                "Import file'."
                            ),
                        ):
                            from core.modal.tdms_importer import load_tdms
                            try:
                                _tdms_obj = load_tdms(Path(_last_capture))
                                st.session_state["modal_tdms"] = _tdms_obj
                                st.session_state["modal_tdms_settings"] = {
                                    "f_target": 500.0,
                                    "coh_thr": 0.8,
                                }
                                st.success(
                                    "✓ TDMS loaded. Switch the radio to "
                                    "'📁 Import file' to see it "
                                    "processed against ISO 7626-5."
                                )
                            except Exception as _exc:  # noqa: BLE001
                                st.error(f"Error loading TDMS: {_exc}")

    # -------- TDMS existente --------
    elif acq_mode.startswith("📁"):
        st.markdown("**Load capture file (.tdms)**")

        # v3.31.209 — Fuente del TDMS: upload local vs Cloud (planta sync)
        _tdms_source = st.radio(
            "File source",
            ["💾 Subir desde mi PC", "☁ Desde Watermelon Cloud (sync planta)"],
            horizontal=True,
            key="tdms_source",
            format_func=lambda o: {
                "💾 Subir desde mi PC": "💾 Upload from my PC",
                "☁ Desde Watermelon Cloud (sync planta)":
                    "☁ From Watermelon Cloud (plant sync)",
            }.get(o, o),
            help=(
                "Cloud: shows the TDMS files you synced from Watermelon "
                "Plant Edition (plant laptop). Requires having synced "
                "at least one TDMS previously."
            ),
        )

        tdms_up = None

        if _tdms_source.startswith("💾"):
            st.caption(
                "Load the .tdms generated by the companion script, Watermelon "
                "Plant or LabVIEW. Watermelon automatically runs the "
                "ISO 7626-5 checklist on the test."
            )
            tdms_up = st.file_uploader(
                "Select .tdms", type=["tdms"], key="tdms_up",
            )
        else:
            # ☁ Cloud — lista los TDMS del bucket modal-captures del user
            st.caption(
                "These are the TDMS files you synced to Watermelon Cloud "
                "from Watermelon Plant Edition. The most recent are listed "
                "first."
            )
            try:
                # Usar el cliente Supabase compartido del Watermelon Cloud
                # (mismo helper que reports_archive.py para reusar cache).
                # v3.31.211 fix: nombre correcto es _get_archive_supabase_client
                # (no _get_supabase_client como tenía antes — bug reportado en
                # producción al cargar Tab Adquisición → Cloud sync).
                from core.reports_archive import _get_archive_supabase_client
                _sb = _get_archive_supabase_client()
                if _sb is None:
                    raise RuntimeError(
                        "Supabase client not initialized. Check that "
                        "st.secrets[supabase].service_key is configured in "
                        "Streamlit Cloud."
                    )
                _user_email = _user.get("email", "")
                if not _user_email:
                    st.error("Could not determine your email. Log in again.")
                else:
                    # Lista objetos en modal-captures/{email}/
                    try:
                        _objs = _sb.storage.from_("modal-captures").list(
                            _user_email,
                            options={
                                "limit": 200,
                                "sortBy": {"column": "created_at",
                                            "order": "desc"},
                            },
                        )
                    except Exception as _exc:
                        # Bucket puede no existir todavía o sin policies
                        _objs = []
                        st.warning(
                            f"Could not list the modal-captures bucket: "
                            f"{_exc}. Check that the bucket exists in "
                            f"Supabase and that you have the RLS policies applied."
                        )

                    # Aplanar recursivamente las carpetas año/mes
                    _all_tdms = []
                    def _walk(prefix):
                        try:
                            items = _sb.storage.from_("modal-captures").list(
                                prefix,
                                options={"limit": 200,
                                          "sortBy": {"column": "created_at",
                                                      "order": "desc"}},
                            )
                        except Exception:
                            return
                        for item in items:
                            name = item.get("name", "")
                            if not name:
                                continue
                            # Carpeta (no tiene metadata) → recurse
                            if item.get("id") is None:
                                _walk(f"{prefix}/{name}")
                            elif name.endswith(".tdms"):
                                _all_tdms.append({
                                    "full_path": f"{prefix}/{name}",
                                    "name": name,
                                    "created_at": item.get("created_at"),
                                    "size": item.get("metadata", {}).get(
                                        "size", 0
                                    ),
                                })

                    _walk(_user_email)

                    if not _all_tdms:
                        st.info(
                            f"No TDMS files synced for `{_user_email}`. "
                            f"For them to appear here, open Watermelon Plant "
                            f"Edition on your plant laptop, capture some "
                            f"data and click **'Sync now'** while "
                            f"online."
                        )
                    else:
                        st.success(
                            f"✓ **{len(_all_tdms)} TDMS** available from "
                            f"Watermelon Plant"
                        )
                        _options = {
                            f"{t['name']} ({t['size']/(1024*1024):.1f} MB · "
                            f"{(t['created_at'] or 'N/A')[:19]})": t
                            for t in _all_tdms
                        }
                        _selected_label = st.selectbox(
                            "Select a TDMS to process",
                            options=list(_options.keys()),
                            key="cloud_tdms_select",
                        )
                        _selected_meta = _options.get(_selected_label)

                        if _selected_meta and st.button(
                            "⬇ Load and process this TDMS",
                            type="primary",
                            use_container_width=True,
                            key="cloud_tdms_load_btn",
                        ):
                            with st.spinner(
                                f"Downloading {_selected_meta['name']}..."
                            ):
                                _bytes = _sb.storage.from_(
                                    "modal-captures"
                                ).download(_selected_meta["full_path"])
                                # v3.31.212 fix — Cargar el TDMS YA al descargar
                                # y guardar en session_state directamente, en
                                # vez de esperar al botón "Procesar". El botón
                                # 'Procesar' del modo file uploader hacía rerun
                                # y perdía tdms_up porque era variable local.
                                try:
                                    from core.modal.tdms_importer import load_tdms
                                    _tmp_cloud = Path(
                                        f"/tmp/_cloud_{_selected_meta['name']}"
                                    )
                                    _tmp_cloud.write_bytes(_bytes)
                                    _tdms_obj_cloud = load_tdms(_tmp_cloud)
                                    st.session_state["modal_tdms"] = (
                                        _tdms_obj_cloud
                                    )
                                    st.session_state["modal_tdms_settings"] = {
                                        "f_target": float(
                                            st.session_state.get(
                                                "tdms_ftarget", 500.0
                                            )
                                        ),
                                        "coh_thr": float(
                                            st.session_state.get(
                                                "tdms_coh", 0.8
                                            )
                                        ),
                                    }
                                    st.success(
                                        f"✓ TDMS loaded from the Cloud "
                                        f"({len(_bytes)/(1024*1024):.1f} MB) "
                                        f"— processing..."
                                    )
                                    # Forzar rerun para que el resto del Tab
                                    # vea el TDMS en session_state y renderice
                                    # la validación ISO 7626-5
                                    st.rerun()
                                except Exception as _exc_load:
                                    st.error(
                                        f"Error loading TDMS: {_exc_load}"
                                    )
            except Exception as _exc:  # noqa: BLE001
                st.error(
                    f"Error accessing the Cloud: {_exc}. "
                    f"Check that your session is valid and that the "
                    f"`modal-captures` bucket exists in Supabase."
                )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tdms_f_target = st.number_input(
                "Target frequency (Hz)", value=500.0, step=50.0,
                key="tdms_ftarget",
                help="Upper band of interest of the test. ISO 7626-5 validates that "
                "the hammer excites flat up to this frequency.",
            )
        with col_t2:
            tdms_coh_thr = st.number_input(
                "Minimum acceptable γ²", value=0.8, step=0.05,
                min_value=0.5, max_value=1.0, key="tdms_coh",
                help="ISO 7626-5 sec. 7.4 — minimum coherence in the band of interest. "
                "Typical 0.8, strict 0.9.",
            )

        if tdms_up and st.button("🔬 Process and validate against ISO 7626-5",
                                   type="primary", use_container_width=True,
                                   key="tdms_process_btn"):
            from core.modal.tdms_importer import load_tdms
            from core.modal.frf_compute import compute_frf_h1
            from core.modal.iso7626_validator import build_compliance_report

            tmp = Path(f"/tmp/_modal_tdms_{tdms_up.name}")
            tmp.write_bytes(tdms_up.read())
            try:
                tdms = load_tdms(tmp)
                st.session_state["modal_tdms"] = tdms
                st.session_state["modal_tdms_settings"] = {
                    "f_target": float(tdms_f_target),
                    "coh_thr": float(tdms_coh_thr),
                }
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error loading TDMS: {exc}")
                st.stop()

        # Renderizar vista TDMS si está cargado
        tdms = st.session_state.get("modal_tdms")
        if tdms is not None:
            from core.modal.frf_compute import compute_frf_h1
            from core.modal.iso7626_validator import build_compliance_report

            settings = st.session_state.get("modal_tdms_settings", {})
            f_target = settings.get("f_target", 500.0)
            coh_thr = settings.get("coh_thr", 0.8)

            st.divider()
            # Metadata header
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Sample rate", f"{tdms.sample_rate_hz:.0f} Hz")
            mc2.metric("Mode", tdms.mode or "—")
            mc3.metric("Channels", len(tdms.channels))
            mc4.metric("Averages", tdms.n_averages or "—")

            # Ciclo 23.155 — Bifurcación EMA vs OMA según mode del TDMS.
            # ISO 7626-5 SOLO aplica a EMA (impact hammer). Para OMA aplica
            # ISO 20816 — sin requerir martillo. Detectamos el mode y dirigimos
            # al usuario al tab correcto.
            #
            # Ciclo 23.156 — Heurística de fallback: si el mode no es claro,
            # inferimos por presencia de un canal con kurtosis alta y baja
            # sensitivity (martillo) vs todos similares (OMA).
            tdms_mode = (tdms.mode or "").lower()
            is_oma_tdms = "oma" in tdms_mode or "continuous" in tdms_mode
            is_ema_tdms = "ema" in tdms_mode or "triggered" in tdms_mode

            # Fallback heurístico: si no hay martillo detectable, asumimos OMA
            if not is_oma_tdms and not is_ema_tdms:
                _hammer_test = tdms.detect_hammer_channel()
                if _hammer_test is None:
                    # Sin martillo → es OMA con altísima probabilidad
                    is_oma_tdms = True
                    st.caption(
                        "ℹ TDMS without explicit mode metadata — detected as OMA "
                        "(no identifiable hammer channel)."
                    )
                else:
                    is_ema_tdms = True

            if is_oma_tdms:
                st.markdown(
                    f'<div style="background:#dbeafe;border:1.5px solid #2563eb;'
                    f'border-radius:8px;padding:14px 18px;">'
                    f'<div style="font-weight:800;color:#1e3a8a;font-size:16px;">'
                    f'🌊 OMA mode TDMS detected</div>'
                    f'<div style="color:#1e40af;font-size:13px;margin-top:4px;">'
                    f'Operational Modal Analysis — no hammer, evaluated under '
                    f'<b>ISO 20816</b> (not ISO 7626-5). Process with FDD in the '
                    f'<b>OMA Tab →</b>.'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                # Preview rápido: time-domain de cada canal
                st.markdown("### Operational channels preview")
                from plotly.subplots import make_subplots
                n_show = len(tdms.channels)
                fig_prev = make_subplots(
                    rows=n_show, cols=1,
                    subplot_titles=[f"{ch.name} ({ch.units})"
                                     for ch in tdms.channels],
                    vertical_spacing=0.06,
                    shared_xaxes=True,
                )
                for i, ch in enumerate(tdms.channels, 1):
                    # Decimar para velocidad si el record es muy largo
                    step = max(1, ch.n_samples // 5000)
                    fig_prev.add_trace(go.Scatter(
                        x=ch.time_s[::step], y=ch.data[::step],
                        mode="lines", showlegend=False,
                        line=dict(width=1, color="#1AAEE5"),
                    ), row=i, col=1)
                    fig_prev.update_yaxes(title_text=ch.units, row=i, col=1)
                fig_prev.update_xaxes(title_text="Time (s)",
                                        row=n_show, col=1)
                fig_prev.update_layout(
                    height=max(280, 150 * n_show),
                    template="plotly_white",
                    margin=dict(l=50, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_prev, use_container_width=True)
                st.caption(
                    "ℹ Operational data ready for FDD. Switch to the **OMA Tab** "
                    "to identify natural modes without needing a hammer."
                )
                # Ciclo 23.159 — NO usar st.stop() aquí: detendría TODOS los tabs
                # (OMA, Mode Shapes, FEA). En su lugar, marcamos flag para skipear
                # el resto del flujo EMA y dejar que los demás tabs renderen.
                _skip_ema_validation = True
            else:
                _skip_ema_validation = False

            # Detección automática de martillo (EMA mode) — solo si no es OMA
            if not _skip_ema_validation:
                hammer = tdms.detect_hammer_channel()
                responses = tdms.response_channels()

                if hammer is None:
                    if not is_ema_tdms:
                        st.warning(
                            "⚠ TDMS without mode metadata. If it is OMA, process "
                            "it with FDD in the OMA Tab."
                        )
                    else:
                        st.warning(
                            "⚠ No hammer channel detected automatically. "
                            "ISO 7626-5 requires a clear input. Check that the "
                            "first channel is the hammer with low sensitivity "
                            "(~2.4 mV/N) or a name 'Hammer'/'Martillo'."
                        )
                    _skip_ema_validation = True

            if not _skip_ema_validation:
                st.success(
                    f"🔨 Hammer detected: **{hammer.name}** "
                    f"(kurtosis {hammer.kurtosis:.1f}, "
                    f"peak/RMS {hammer.peak_to_rms:.1f}, "
                    f"sens {hammer.sensitivity_mv_per_eu} mV/{hammer.units})"
                )

                # Selector de canal de respuesta
                resp_names = [r.name for r in responses]
                if not resp_names:
                    st.error("No response channels — the TDMS only has a hammer.")
                else:
                    resp_pick = st.selectbox(
                        "Response channel to analyze",
                        resp_names, key="tdms_resp_pick",
                    )
                    resp = next(r for r in responses if r.name == resp_pick)

                    # Calcular FRF + coherencia
                    nperseg = min(1024, hammer.data.size // 4)
                    frf = compute_frf_h1(
                        input_signal=hammer.data,
                        output_signal=resp.data,
                        sample_rate_hz=tdms.sample_rate_hz,
                        nperseg=nperseg,
                    )

                    # Validar conforme ISO 7626-5
                    report = build_compliance_report(
                        input_signal=hammer.data,
                        output_signal=resp.data,
                        coherence=frf.coherence,
                        coherence_frequencies_hz=frf.frequencies_hz,
                        sample_rate_hz=tdms.sample_rate_hz,
                        f_target_hz=f_target,
                        n_averages=tdms.n_averages or 1,
                        coherence_threshold=coh_thr,
                        test_setup_name=f"{hammer.name} → {resp.name}",
                    )

                    # CHECKLIST ISO 7626-5 (banner superior)
                    st.markdown("### ISO 7626-5 · Test validation")
                    if report.overall_pass:
                        st.markdown(
                            f'<div style="background:#dcfce7;border:1.5px solid #16a34a;'
                            f'border-radius:8px;padding:14px 18px;">'
                            f'<div style="font-weight:800;color:#14532d;font-size:18px;">'
                            f'✓ Test compliant with ISO 7626-5</div>'
                            f'<div style="color:#166534;font-size:13px;margin-top:4px;">'
                            f'{report.n_passed}/{report.n_total} checks passed</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    elif report.has_fails:
                        st.markdown(
                            f'<div style="background:#fee2e2;border:1.5px solid #dc2626;'
                            f'border-radius:8px;padding:14px 18px;">'
                            f'<div style="font-weight:800;color:#7f1d1d;font-size:18px;">'
                            f'✗ Test NOT compliant with ISO 7626-5</div>'
                            f'<div style="color:#991b1b;font-size:13px;margin-top:4px;">'
                            f'{report.n_passed}/{report.n_total} checks passed · review failures</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="background:#fef3c7;border:1.5px solid #d97706;'
                            f'border-radius:8px;padding:14px 18px;">'
                            f'<div style="font-weight:800;color:#78350f;font-size:18px;">'
                            f'⚠ Test with observations</div>'
                            f'<div style="color:#92400e;font-size:13px;margin-top:4px;">'
                            f'{report.n_passed}/{report.n_total} checks · review warnings</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # Tabla de checks
                    check_cols = st.columns(len(report.checks))
                    for col, check in zip(check_cols, report.checks):
                        with col:
                            if check.severity == "ok":
                                bg, fg, icon = "#dcfce7", "#14532d", "✓"
                            elif check.severity == "warning":
                                bg, fg, icon = "#fef3c7", "#78350f", "⚠"
                            else:
                                bg, fg, icon = "#fee2e2", "#7f1d1d", "✗"
                            st.markdown(
                                f'<div style="background:{bg};border-radius:8px;'
                                f'padding:10px 12px;min-height:80px;">'
                                f'<div style="color:{fg};font-weight:700;font-size:13px;">'
                                f'{icon} {check.title}</div>'
                                f'<div style="color:{fg};font-size:11px;margin-top:4px;'
                                f'opacity:0.85;">{check.norm_ref}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    # Expander con detalles
                    with st.expander("📋 View detail of each check", expanded=False):
                        for c in report.checks:
                            icon = "✓" if c.passed else ("⚠" if c.severity == "warning" else "✗")
                            st.markdown(f"**{icon} {c.title}** · `{c.norm_ref}`")
                            st.caption(c.detail)
                            st.divider()

                    # PANEL DE 6 PLOTS — Input / Output / FRF / Coherencia
                    st.markdown(f"### ISO 7626-5 panel · {hammer.name} → {resp.name}")

                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=3, cols=2,
                        subplot_titles=(
                            f"Input — {hammer.name} (time)",
                            f"Input — {hammer.name} (spectrum)",
                            f"Response — {resp.name} (time)",
                            f"Response — {resp.name} (spectrum)",
                            "FRF — Magnitude + Phase",
                            "Coherence γ²(f)",
                        ),
                        vertical_spacing=0.10,
                        horizontal_spacing=0.08,
                    )

                    fig.add_trace(go.Scatter(
                        x=hammer.time_s, y=hammer.data, mode="lines",
                        name="Input time", line=dict(color="#0F1E3D", width=1),
                        showlegend=False,
                    ), row=1, col=1)
                    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
                    fig.update_yaxes(title_text=f"{hammer.units}", row=1, col=1)

                    from scipy.signal import welch as _welch
                    f_in, psd_in = _welch(hammer.data, fs=tdms.sample_rate_hz,
                                            nperseg=nperseg)
                    fig.add_trace(go.Scatter(
                        x=f_in, y=10 * np.log10(np.maximum(psd_in, 1e-30)),
                        mode="lines", name="Input spec",
                        line=dict(color="#0F1E3D", width=1),
                        showlegend=False,
                    ), row=1, col=2)
                    fig.add_vline(x=f_target, line=dict(color="#D89B22", dash="dash"),
                                   row=1, col=2)
                    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
                    fig.update_yaxes(title_text=f"PSD (dB ref {hammer.units}²/Hz)", row=1, col=2)

                    fig.add_trace(go.Scatter(
                        x=resp.time_s, y=resp.data, mode="lines",
                        name="Response time", line=dict(color="#1AAEE5", width=1),
                        showlegend=False,
                    ), row=2, col=1)
                    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                    fig.update_yaxes(title_text=f"{resp.units}", row=2, col=1)

                    f_out, psd_out = _welch(resp.data, fs=tdms.sample_rate_hz,
                                              nperseg=nperseg)
                    fig.add_trace(go.Scatter(
                        x=f_out, y=10 * np.log10(np.maximum(psd_out, 1e-30)),
                        mode="lines", name="Response spec",
                        line=dict(color="#1AAEE5", width=1),
                        showlegend=False,
                    ), row=2, col=2)
                    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
                    fig.update_yaxes(title_text=f"PSD (dB ref {resp.units}²/Hz)", row=2, col=2)

                    mag_db = 20 * np.log10(np.maximum(frf.magnitude, 1e-30))
                    fig.add_trace(go.Scatter(
                        x=frf.frequencies_hz, y=mag_db, mode="lines",
                        name="FRF Mag", line=dict(color="#0F7FB0", width=1.5),
                        showlegend=False,
                    ), row=3, col=1)
                    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
                    fig.update_yaxes(title_text="Magnitude (dB)", row=3, col=1)

                    fig.add_trace(go.Scatter(
                        x=frf.frequencies_hz, y=frf.coherence, mode="lines",
                        name="γ²", line=dict(color="#16a34a", width=1.5),
                        showlegend=False,
                    ), row=3, col=2)
                    fig.add_hline(y=coh_thr, line=dict(color="#D89B22", dash="dash"),
                                   row=3, col=2)
                    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=2)
                    fig.update_yaxes(title_text="γ² (0-1)", row=3, col=2,
                                      range=[0, 1.05])

                    fig.update_layout(
                        height=750,
                        showlegend=False,
                        template="plotly_white",
                        margin=dict(l=60, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Persistir FRF para uso en Tab EMA
                    st.session_state["modal_tdms_frf"] = frf
                    st.session_state["modal_tdms_pair"] = (hammer.name, resp.name)
                    st.caption(
                        f"📊 FRF computed via {frf.estimator} estimator · "
                        f"{frf.n_averages} Welch segments · {frf.window} window · "
                        f"nperseg = {nperseg}. Available in the EMA Tab for modal identification."
                    )

    # -------- FRFs legacy (formato .txt) --------
    else:
        st.markdown("**Import legacy FRFs (.txt from previous modal software)**")

        uploaded = st.file_uploader(
            "Upload .txt files",
            type=["txt"],
            accept_multiple_files=True,
            key="art_up",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            art_fs = st.number_input(
                "Original sample rate (Hz)",
                value=2560, step=100, min_value=10, key="art_fs",
            )
        with col_b:
            art_bw = st.number_input(
                "Bandwidth (Hz)",
                value=1280, step=100, min_value=1, key="art_bw",
            )
        st.caption(
            "The frequency axis is reconstructed as Δf = bandwidth / (N_bins - 1). "
            "Legacy files do NOT store the axis — must be completed manually."
        )

        if uploaded and st.button("🔍 Process legacy files",
                                    type="primary", use_container_width=True,
                                    key="art_process_btn"):
            from core.modal.artemis_importer import load_artemis_file, detect_file_type

            loaded = []
            errors = []
            for up in uploaded:
                # Persistimos a temp path para leerlo con numpy
                tmp = Path(f"/tmp/_modal_{up.name}")
                tmp.write_bytes(up.read())
                try:
                    frf = load_artemis_file(
                        tmp,
                        sample_rate_hz=float(art_fs),
                        bandwidth_hz=float(art_bw),
                        channel_label=up.name.replace(".txt", "").replace(" (1)", "").strip(),
                    )
                    loaded.append(frf)
                except (ValueError, OSError) as exc:
                    errors.append(f"{up.name}: {exc}")

            if loaded:
                st.session_state["modal_frfs"] = loaded
                st.success(
                    f"✓ {len(loaded)} files processed · "
                    f"Δf = {loaded[0].df:.3f} Hz · "
                    f"{loaded[0].n_bins} bins · "
                    f"band 0 → {loaded[0].frequencies_hz[-1]:.0f} Hz"
                )
            if errors:
                for e in errors:
                    st.error(e)

        # Mostrar FRFs cargadas
        frfs = st.session_state.get("modal_frfs", [])
        if frfs:
            st.divider()
            st.markdown(f"### Bode plot — {len(frfs)} channel(s) loaded")

            # Plot magnitud
            fig_mag = go.Figure()
            for frf in frfs:
                mag = frf.magnitude_linear()
                if mag.size == 0:
                    continue
                fig_mag.add_trace(go.Scatter(
                    x=frf.frequencies_hz,
                    y=20.0 * np.log10(np.maximum(mag, 1e-30)),
                    mode="lines",
                    name=frf.channel_label or frf.source_file,
                    line=dict(width=1.2),
                ))
            fig_mag.update_layout(
                title="Magnitude — dB",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude (dB)",
                height=380,
                margin=dict(l=50, r=20, t=40, b=40),
                template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(fig_mag, use_container_width=True)

            # Plot fase si hay FRF complejas
            any_complex = any(frf.is_complex_frf for frf in frfs)
            if any_complex:
                fig_phase = go.Figure()
                for frf in frfs:
                    phase = frf.phase_deg()
                    if phase is None:
                        continue
                    fig_phase.add_trace(go.Scatter(
                        x=frf.frequencies_hz,
                        y=phase,
                        mode="lines",
                        name=frf.channel_label or frf.source_file,
                        line=dict(width=1.2),
                    ))
                fig_phase.update_layout(
                    title="Phase — degrees",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Phase (°)",
                    height=280,
                    margin=dict(l=50, r=20, t=40, b=40),
                    template="plotly_white",
                    hovermode="x unified",
                )
                st.plotly_chart(fig_phase, use_container_width=True)


# ---------------------------------------------------------------------
# Tab 3 — EMA Processing
# ---------------------------------------------------------------------
if _active_modal_tab == "🔨 EMA":
    st.subheader("Experimental Modal Analysis")
    st.caption(
        "Identification of modal parameters (natural frequency, damping, mode shape) "
        "by the Circle-Fit Nyquist (Kennedy-Pancu) + half-power method. "
        "Complies with ISO 7626-6 sec. 6.3."
    )

    # ─── Si hay TDMS procesado, ofrecer identificación moderna ───────
    tdms_frf = st.session_state.get("modal_tdms_frf")
    tdms_pair = st.session_state.get("modal_tdms_pair")
    if tdms_frf is not None and tdms_pair:
        st.markdown(
            f'<div style="background:#dcfce7;border:1.5px solid #16a34a;'
            f'border-radius:8px;padding:14px 18px;margin-bottom:18px;">'
            f'<div style="font-weight:800;color:#14532d;font-size:15px;">'
            f'🎯 FRF available from ISO 7626-5 TDMS capture</div>'
            f'<div style="color:#166534;font-size:13px;margin-top:4px;">'
            f'Pair: <b>{tdms_pair[0]} → {tdms_pair[1]}</b> · '
            f'{len(tdms_frf.frequencies_hz)} bins · '
            f'estimator {tdms_frf.estimator} · '
            f'mean γ² {tdms_frf.coherence.mean():.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Modal identification on measured FRF (Circle-Fit Nyquist)**")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            ti_f_min = st.number_input(
                "f min (Hz)", value=5.0, step=1.0, key="ti_fmin",
            )
        with col_t2:
            _f_max_def = float(tdms_frf.frequencies_hz[-1])
            ti_f_max = st.number_input(
                "f max (Hz)", value=_f_max_def, step=10.0, key="ti_fmax",
            )
        with col_t3:
            ti_prom = st.number_input(
                "Prominence (dB)", value=12.0, step=1.0, key="ti_prom",
                help="Default 12 dB · strict. Lower it for more sensitivity.",
            )
        with col_t4:
            ti_dist = st.number_input(
                "Min distance (Hz)", value=5.0, step=1.0, key="ti_dist",
            )

        if st.button("🎯 Identify modes (Circle-Fit Nyquist + half-power)",
                      type="primary", use_container_width=True, key="ti_run"):
            from core.modal.ema_engine import identify_modes_robust

            modes = identify_modes_robust(
                frf_complex=tdms_frf.frf_complex,
                frequencies_hz=tdms_frf.frequencies_hz,
                f_min_hz=float(ti_f_min),
                f_max_hz=float(ti_f_max),
                prominence_db=float(ti_prom),
                min_distance_hz=float(ti_dist),
            )
            st.session_state["modal_tdms_modes"] = modes

        tdms_modes = st.session_state.get("modal_tdms_modes", [])
        if tdms_modes:
            st.divider()
            st.markdown(f"### Identified modes — {len(tdms_modes)} (TDMS FRF)")

            import pandas as pd

            def _method_for(m):
                return "Circle-Fit Nyquist" if m.confidence >= 0.9 else "Half-power"

            df_tdms = pd.DataFrame([
                {
                    "Mode": m.mode_number,
                    "Frequency (Hz)": round(m.natural_frequency_hz, 2),
                    "Damping (%)": round(m.damping_ratio_pct, 3),
                    "Method": _method_for(m),
                    "Confidence": round(m.confidence, 2),
                }
                for m in tdms_modes
            ])
            st.dataframe(df_tdms, use_container_width=True, hide_index=True)

            # Plot FRF con modos
            mag_db = 20.0 * np.log10(np.maximum(np.abs(tdms_frf.frf_complex), 1e-30))
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=tdms_frf.frequencies_hz, y=mag_db, mode="lines",
                name="FRF", line=dict(width=1.2, color="#1AAEE5"),
            ))
            peak_freqs = [m.natural_frequency_hz for m in tdms_modes]
            peak_mag = []
            for fn in peak_freqs:
                idx = int(np.argmin(np.abs(tdms_frf.frequencies_hz - fn)))
                peak_mag.append(mag_db[idx])
            colors = ["#16a34a" if m.confidence >= 0.9 else "#D89B22"
                       for m in tdms_modes]
            fig_t.add_trace(go.Scatter(
                x=peak_freqs, y=peak_mag, mode="markers+text",
                name="Modes",
                marker=dict(color=colors, size=12, symbol="diamond",
                             line=dict(width=1.5, color="#0F1E3D")),
                text=[str(m.mode_number) for m in tdms_modes],
                textposition="top center",
                textfont=dict(size=10, color="#0F1E3D"),
                customdata=[
                    f"Mode {m.mode_number}<br>{m.natural_frequency_hz:.2f} Hz · "
                    f"ζ={m.damping_ratio_pct:.3f}%<br>"
                    f"{_method_for(m)} · conf={m.confidence:.2f}"
                    for m in tdms_modes
                ],
                hovertemplate="%{customdata}<extra></extra>",
            ))
            fig_t.update_layout(
                title="Measured FRF with identified modes (green = Circle-Fit, amber = Half-power)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude (dB)",
                height=420,
                margin=dict(l=50, r=20, t=60, b=40),
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig_t, use_container_width=True)

            st.caption(
                "🔬 Circle-Fit Nyquist method (Kennedy-Pancu 1947) — classic in EMA "
                "for SDOF modes. Modes in green have passed the circle fit "
                "(typical accuracy < 1% in fn). Modes in amber use half-power as a "
                "fallback (accuracy 2-5%). Both comply with ISO 7626-6 sec. 6.3."
            )

        st.divider()

    # ─── Sección FRFs legacy (cargadas via .txt) ──────────────────────
    frfs = st.session_state.get("modal_frfs", [])
    if not frfs and tdms_frf is None:
        st.info("📭 No FRFs loaded. Load data in the Acquisition tab first "
                "(legacy .txt file or Watermelon acquisition unit capture).")
    elif not frfs:
        pass  # solo TDMS cargado — UI ya mostrada arriba
    else:
        st.markdown(f"**{len(frfs)} FRF(s) loaded — ready for modal identification**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ema_f_min = st.number_input("f min (Hz)", value=5.0, step=1.0,
                                          key="ema_fmin")
        with col2:
            _f_max_default = float(frfs[0].frequencies_hz[-1])
            ema_f_max = st.number_input("f max (Hz)", value=_f_max_default,
                                          step=10.0, key="ema_fmax")
        with col3:
            ema_prom = st.number_input("Prominence (dB)", value=6.0,
                                         step=1.0, key="ema_prom",
                                         help="Minimum peak height vs surroundings")
        with col4:
            ema_dist = st.number_input("Min distance (Hz)", value=2.0,
                                         step=0.5, key="ema_dist",
                                         help="Minimum separation between peaks")

        if st.button("🎯 Identify modes", type="primary",
                       use_container_width=True, key="ema_run_btn"):
            from core.modal.frf_compute import detect_modal_peaks

            # Selección de FRF principal — la primera de 2 columnas (FRF compleja)
            # o la primera del listado si no hay FRF compleja
            primary = next((f for f in frfs if f.is_complex_frf), frfs[0])
            mag = primary.magnitude_linear()
            if mag.size == 0:
                st.error("The selected FRF has no computable magnitude.")
            else:
                peaks = detect_modal_peaks(
                    frequencies_hz=primary.frequencies_hz,
                    magnitude=mag,
                    coherence=None,  # los exports legacy no incluyen coherencia
                    f_min_hz=float(ema_f_min),
                    f_max_hz=float(ema_f_max),
                    prominence_db=float(ema_prom),
                    min_distance_hz=float(ema_dist),
                )
                st.session_state["modal_peaks"] = peaks

        peaks = st.session_state.get("modal_peaks", [])
        if peaks:
            st.divider()
            st.markdown(f"### Identified modes — {len(peaks)}")

            # Tabla modal
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Mode": i + 1,
                    "Frequency (Hz)": round(p.frequency_hz, 2),
                    "Damping (%)": round(p.damping_ratio_pct, 3),
                    "Bandwidth (Hz)": round(p.bandwidth_hz, 3),
                    "Q factor": round(p.quality_factor, 1),
                    "Peak magnitude": f"{p.magnitude_peak:.3e}",
                }
                for i, p in enumerate(peaks)
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Plot con picos — solo número del modo sobre la vline + markers
            # en los peaks. El detalle (freq + damping) vive en la tabla arriba.
            # Ciclo 23.152 — Fix de legibilidad cuando hay muchos modos (50+):
            # las anotaciones text-stacked se solapaban. Ahora solo marker + index.
            primary = next((f for f in frfs if f.is_complex_frf), frfs[0])
            mag_lin = primary.magnitude_linear()
            mag_db = 20.0 * np.log10(np.maximum(mag_lin, 1e-30))

            fig_peaks = go.Figure()
            # Curva FRF
            fig_peaks.add_trace(go.Scatter(
                x=primary.frequencies_hz, y=mag_db, mode="lines",
                name="FRF", line=dict(width=1.2, color="#1AAEE5"),
                hovertemplate="f=%{x:.1f} Hz<br>|H|=%{y:.1f} dB<extra></extra>",
            ))
            # Markers en cada pico — más legible que vlines + annotations
            peak_freqs = [p.frequency_hz for p in peaks]
            peak_mag_db = [20.0 * np.log10(max(p.magnitude_peak, 1e-30))
                            for p in peaks]
            peak_labels = [
                f"Mode {i+1}<br>{p.frequency_hz:.1f} Hz · ζ={p.damping_ratio_pct:.2f}%"
                for i, p in enumerate(peaks)
            ]
            fig_peaks.add_trace(go.Scatter(
                x=peak_freqs, y=peak_mag_db, mode="markers+text",
                name="Modes",
                marker=dict(color="#D89B22", size=10, symbol="diamond",
                             line=dict(width=1.5, color="#7c2d12")),
                text=[str(i + 1) for i in range(len(peaks))],
                textposition="top center",
                textfont=dict(size=10, color="#7c2d12"),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=peak_labels,
            ))
            fig_peaks.update_layout(
                title=f"FRF with {len(peaks)} identified modes — hover over diamonds for detail",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude (dB)",
                height=420,
                margin=dict(l=50, r=20, t=60, b=40),
                template="plotly_white",
                hovermode="closest",
                showlegend=False,
            )
            st.plotly_chart(fig_peaks, use_container_width=True)

            st.caption(
                "🔬 Damping computed by the half-power method (-3 dB · ISO 7626-6 sec. 6.3.2). "
                "Amber diamonds mark the detected modes — hover for frequency + damping. "
                "For mode shapes and LSCF curve fit, pyEMA integration is required (next sprint)."
            )


# ---------------------------------------------------------------------
# Tab 4 — OMA Processing (FDD)
# ---------------------------------------------------------------------
if _active_modal_tab == "🌊 OMA":
    st.subheader("Operational Modal Analysis — FDD")
    st.caption(
        "Frequency Domain Decomposition (Brincker 2001) on the system's operational "
        "data. No hammer needed. Complies with ISO 20816 + API 684."
    )

    tdms_oma = st.session_state.get("modal_tdms")
    if tdms_oma is None:
        modal_empty_state(
            icon="🌊",
            title="No operational data loaded",
            description=(
                "OMA analysis requires a .tdms file with a continuous capture "
                "of the system — at least 60 seconds at constant speed (ISO 20816 + "
                "Brincker 2001). Load the file in the Acquisition Tab using the "
                "'Import existing .tdms' option."
            ),
            cta_label="Switch to the Acquisition Tab",
            norm_ref="ISO 20816 · FDD requirements",
        )
    else:
        # Mostrar metadata
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("TDMS mode", tdms_oma.mode or "—")
        mc2.metric("Channels", len(tdms_oma.channels))
        mc3.metric("Duration",
                    f"{tdms_oma.channels[0].duration_s:.1f}s" if tdms_oma.channels else "—")
        mc4.metric("Fs", f"{tdms_oma.sample_rate_hz:.0f} Hz")

        _record_dur = (tdms_oma.channels[0].duration_s if tdms_oma.channels else 0)

        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
        with col_o1:
            oma_fmin = st.number_input(
                "f min (Hz)", value=5.0, step=1.0, key="oma_fmin",
                help=("Lowest natural frequency to identify. Defines the "
                      "minimum record time: T_min ≥ 2000 / f_min "
                      "(Brincker & Ventura 2015)."),
            )

        # --- Validación normativa del record contra fn_low ---
        _t_min_strict_tdms = 2000.0 / max(float(oma_fmin), 0.1)
        _t_min_floor_tdms  = 1000.0 / max(float(oma_fmin), 0.1)
        if _record_dur > 0:
            if _record_dur < _t_min_floor_tdms:
                modal_status_banner(
                    title=(f"Record {_record_dur:.0f} s < absolute T_min "
                             f"{_t_min_floor_tdms:.0f} s for f_min = {oma_fmin:.1f} Hz"),
                    detail=(
                        f"The standard requires at least **1000 × T_low = "
                        f"{_t_min_floor_tdms:.0f} s** and recommends "
                        f"**2000 × T_low = {_t_min_strict_tdms:.0f} s** "
                        "(Brincker & Ventura 2015 · ISO 18649). The FDD will run "
                        "but the damping ratios may have variance > 30%. "
                        "For report-grade results, recapture with more time or "
                        "raise f_min if the low modes are not of interest."
                    ),
                    severity="fail",
                )
            elif _record_dur < _t_min_strict_tdms:
                modal_status_banner(
                    title=(f"Record {_record_dur:.0f} s accepts the normative floor · "
                             f"recommended {_t_min_strict_tdms:.0f} s"),
                    detail=(
                        f"For f_min = {oma_fmin:.1f} Hz you meet "
                        f"1000 × T_low ({_t_min_floor_tdms:.0f} s) but "
                        f"you are below the recommended 2000 × T_low "
                        f"({_t_min_strict_tdms:.0f} s). Mode identification OK, "
                        "moderate uncertainty in damping."
                    ),
                    severity="warning",
                )
            else:
                modal_status_banner(
                    title=(f"Record compliant with standard · {_record_dur:.0f} s ≥ "
                             f"{_t_min_strict_tdms:.0f} s for f_min = {oma_fmin:.1f} Hz"),
                    detail="Brincker & Ventura 2015 · ISO 18649 — met.",
                    severity="ok",
                )
        with col_o2:
            _f_max_def = float(tdms_oma.sample_rate_hz / 2.0 * 0.9)
            oma_fmax = st.number_input("f max (Hz)", value=min(500.0, _f_max_def),
                                         step=10.0, key="oma_fmax")
        with col_o3:
            oma_prom = st.number_input("Prominence (dB)", value=8.0, step=1.0,
                                         key="oma_prom")
        with col_o4:
            oma_rpm = st.number_input("Running speed (rpm, optional)",
                                        value=0, step=100, key="oma_rpm",
                                        help="If given, marks peaks near "
                                             "1×, 2×, 3× as harmonics")

        # --- Recorte del intervalo a procesar (v3.31.453) -------------------
        # En OMA los primeros segundos suelen traer transitorios (arranque de la
        # adquisición, asentamiento del sensor IEPE, manipulación) que
        # contaminan la identificación de modos. Permite descartar inicio/final.
        st.markdown("**✂️ Interval to process**")
        _dur_total = float(_record_dur or 0.0)
        _tc1, _tc2, _tc3 = st.columns([1, 1, 2])
        with _tc1:
            _trim_start = st.number_input(
                "Discard at start (s)", value=0.0, min_value=0.0,
                max_value=max(_dur_total - 1.0, 0.0), step=0.5, key="oma_trim_start",
                help="Recommended 1–2 s: discards the startup transient of "
                     "the acquisition and the sensor settling.")
        with _tc2:
            _trim_end = st.number_input(
                "Discard at end (s)", value=0.0, min_value=0.0,
                max_value=max(_dur_total - 1.0, 0.0), step=0.5, key="oma_trim_end",
                help="Useful if there was sensor handling or a shutdown at the end.")
        _dur_eff = max(_dur_total - float(_trim_start) - float(_trim_end), 0.0)
        with _tc3:
            if _dur_eff <= 0:
                st.error("The trim leaves the record empty — reduce the values.")
            elif _trim_start or _trim_end:
                st.info(f"**{_dur_eff:.1f} s** of {_dur_total:.1f} s will be processed "
                        f"(from {_trim_start:.1f} s to "
                        f"{_dur_total - _trim_end:.1f} s).", icon="✂️")
            else:
                st.caption(f"The full record is processed ({_dur_total:.1f} s). "
                           "If you see a transient at the start, discard 1–2 s.")

        if st.button("🌊 Run FDD + identify modes", type="primary",
                       use_container_width=True, key="oma_run",
                       disabled=(_dur_eff <= 0)):
            from core.modal.oma_engine import run_oma

            time_data = np.stack([ch.data for ch in tdms_oma.channels], axis=1)
            # Aplicar el recorte por muestras
            _fs_oma = float(tdms_oma.sample_rate_hz)
            _i0 = int(round(float(_trim_start) * _fs_oma))
            _i1 = time_data.shape[0] - int(round(float(_trim_end) * _fs_oma))
            _i0 = max(0, min(_i0, time_data.shape[0] - 1))
            _i1 = max(_i0 + 1, min(_i1, time_data.shape[0]))
            _n_before = time_data.shape[0]
            time_data = time_data[_i0:_i1, :]
            if _i0 or _i1 < _n_before:
                st.caption(
                    f"✂️ Trim applied: {time_data.shape[0]:,} of "
                    f"{_n_before:,} samples ({time_data.shape[0] / _fs_oma:.1f} s).")
            running_hz = (oma_rpm / 60.0) if oma_rpm > 0 else None
            nperseg = min(4096, max(time_data.shape[0] // 8, 16))
            with st.spinner("Computing PSD matrix + SVD per frequency..."):
                fdd_result = run_oma(
                    time_data=time_data,
                    sample_rate_hz=tdms_oma.sample_rate_hz,
                    nperseg=nperseg,
                    channel_names=[ch.name for ch in tdms_oma.channels],
                    f_min_hz=float(oma_fmin), f_max_hz=float(oma_fmax),
                    prominence_db=float(oma_prom),
                    min_distance_hz=2.0,
                    running_speed_hz=running_hz,
                )
            st.session_state["modal_oma_result"] = fdd_result

        fdd = st.session_state.get("modal_oma_result")
        if fdd is not None:
            # KPI row con resumen de clasificación
            _n_natural = sum(1 for m in fdd.modes if m.classification == "natural")
            _n_harm = sum(1 for m in fdd.modes if m.classification == "harmonic")
            _n_sp = sum(1 for m in fdd.modes if m.classification == "spurious")
            _avg_conf = (sum(m.confidence for m in fdd.modes) / max(len(fdd.modes), 1)) * 100
            modal_kpi_row([
                (str(_n_natural), "Natural modes", "✓ identified", "green"),
                (str(_n_harm), "Harmonics", "× running speed", "red"),
                (str(_n_sp), "Spurious",
                 f"discarded MPC > 75%", "gray"),
                (f"{_avg_conf:.0f}%", "Average confidence",
                 "aggregated by MPC + harm", "cyan"),
            ])

            modal_section_header(
                title="Spectral density — Singular Values",
                subtitle="Multi-SVD of the PSD matrix · peaks = natural modes",
                norm_ref="ISO 20816 · Brincker 2001",
                icon="🌊",
            )

            # Multi-SVD plot — equivalente al "Singular Values of Spectral Densities"
            # estándar OMA. SVD Line 1 (principal) + Line 2 + Line 3 si hay ≥ 3 canales.
            fig_sv = go.Figure()
            svd_colors = ["#0F7FB0", "#dc2626", "#16a34a", "#a855f7"]
            for k in range(min(fdd.n_channels, 3)):
                sv_k_db = 10.0 * np.log10(np.maximum(fdd.singular_values[k, :], 1e-30))
                fig_sv.add_trace(go.Scatter(
                    x=fdd.frequencies_hz, y=sv_k_db,
                    mode="lines",
                    name=f"SVD Line {k+1}",
                    line=dict(width=1.2 if k == 0 else 0.9,
                              color=svd_colors[k % len(svd_colors)],
                              dash="solid" if k == 0 else "dot"),
                    opacity=1.0 if k == 0 else 0.6,
                ))

            # Markers de modos identificados con color según clasificación
            sv1_db = fdd.first_sv_db()
            class_colors = {
                "natural": "#16a34a",
                "harmonic": "#dc2626",
                "spurious": "#9ca3af",
            }
            for m in fdd.modes:
                idx = int(np.argmin(np.abs(fdd.frequencies_hz - m.natural_frequency_hz)))
                label = "fn" if m.classification == "natural" else (
                    f"{m.harmonic_order}×" if m.is_harmonic else "?"
                )
                fig_sv.add_trace(go.Scatter(
                    x=[m.natural_frequency_hz], y=[sv1_db[idx]],
                    mode="markers+text",
                    marker=dict(color=class_colors.get(m.classification, "#0F1E3D"),
                                size=11, symbol="diamond",
                                line=dict(width=1.2, color="#0F1E3D")),
                    text=[label], textposition="top center",
                    textfont=dict(size=9, color="#0F1E3D"),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Mode {m.mode_number} — ACCEPTED</b><br>"
                        f"Frequency: {m.natural_frequency_hz:.2f} Hz<br>"
                        f"Damping ζ: {m.damping_ratio_pct:.2f}%<br>"
                        f"Singular value: {sv1_db[idx]:.1f} dB<br>"
                        f"MPC complexity: {m.complexity_pct:.1f}%<br>"
                        f"Confidence: {m.confidence * 100:.0f}%<br>"
                        f"Classification: {m.classification}<extra></extra>"
                    ),
                ))

            # Overlay de picos CANDIDATOS (transparencia del criterio): marca
            # los bumps más marcados aunque NO superen la prominencia mínima,
            # así el usuario ve por qué se aceptó/descartó cada pico.
            try:
                from core.modal.oma_interpret import (
                    svd_detection_report as _svd_report)
                _det = _svd_report(fdd.frequencies_hz, sv1_db,
                                    float(oma_fmin), float(oma_fmax),
                                    float(oma_prom))
                _accepted = {round(m.natural_frequency_hz, 1) for m in fdd.modes}
                for _cf, _cp in _det.get("candidates", []):
                    if round(_cf, 1) in _accepted:
                        continue
                    _ci = int(np.argmin(np.abs(fdd.frequencies_hz - _cf)))
                    fig_sv.add_trace(go.Scatter(
                        x=[_cf], y=[sv1_db[_ci]], mode="markers",
                        marker=dict(color="rgba(148,163,184,0.9)", size=9,
                                    symbol="circle-open",
                                    line=dict(width=1.5, color="#64748b")),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Candidate peak — DISCARDED</b><br>"
                            f"Frequency: {_cf:.2f} Hz<br>"
                            f"Prominence: {_cp:.1f} dB<br>"
                            f"Singular value: {sv1_db[_ci]:.1f} dB<br>"
                            f"Required threshold: {float(oma_prom):.0f} dB<br>"
                            f"Reason: does not stand out from noise<extra></extra>"),
                    ))
            except Exception:  # noqa: BLE001
                _det = {}

            fig_sv.update_layout(
                title=("Singular Values of Spectral Densities — "
                       "green: natural mode (fn) · red: harmonic (Nx) · gray: spurious"),
                xaxis_title="Frequency (Hz)",
                yaxis_title="dB | (EU)² / Hz",
                height=440,
                template="plotly_white",
                margin=dict(l=50, r=20, t=60, b=40),
                hovermode="closest",
                legend=dict(orientation="h", y=1.05, x=0.65),
            )
            st.plotly_chart(fig_sv, use_container_width=True)

            # Criterios de detección usados (transparencia del algoritmo)
            st.caption(
                f"**Detection criteria used:** band "
                f"**{float(oma_fmin):.0f}–{float(oma_fmax):.0f} Hz** · "
                f"minimum peak prominence **≥ {float(oma_prom):.0f} dB** above "
                "the local noise · classification by MPC (natural < 40% · spurious "
                "> 75%) and proximity to harmonics of the running speed. The Y axis "
                "is **10·log₁₀(singular value)**: a **relative** scale in dB (not an "
                "absolute physical value). The **gray hollow circles** are candidate "
                "peaks that did NOT exceed the prominence threshold.")

            # Explicación específica cuando NO se identificó ningún modo natural
            if _n_natural == 0:
                _best_hz = _det.get("best_hz")
                _best_prom = float(_det.get("best_prom", 0.0) or 0.0)
                if _det.get("n_at_threshold", 0) == 0 and _best_hz is not None:
                    st.warning(
                        f"**Why 0 modes:** no peak exceeds the minimum "
                        f"prominence of **{float(oma_prom):.0f} dB** in the band "
                        f"{float(oma_fmin):.0f}–{float(oma_fmax):.0f} Hz. The most "
                        f"marked peak is only **{_best_prom:.1f} dB** "
                        f"(~{_best_hz:.1f} Hz). Options: lower the prominence to "
                        f"~{max(_best_prom * 0.8, 1):.0f} dB if you think it is a real "
                        "mode, adjust the band, or the record has no clear modes "
                        "(very low ambient excitation or poorly coupled sensor).",
                        icon="🔎")
                elif _best_hz is None:
                    st.warning(
                        "**Why 0 modes:** the SVD curve shows no peaks in the "
                        "chosen band (it looks like broadband noise). Check that "
                        "the sensor measures real vibration and that the band covers "
                        "the range of interest.", icon="🔎")

            # --- Tabla de PICOS CANDIDATOS: todos los detectados, con el motivo
            # de aceptación o rechazo. Antes, si ninguno pasaba el umbral la
            # tabla salía "0 candidatos" aunque la gráfica sí marcaba círculos
            # → confusión. Ahora se ve cada pico y por qué se descartó.
            _cd = _det.get("candidates_detail", []) if isinstance(_det, dict) else []
            if _cd:
                import pandas as _pd_c
                _acc_by_hz = {round(m.natural_frequency_hz, 1): m for m in fdd.modes}
                _rows_c = []
                for _c in sorted(_cd, key=lambda d: -d["prominence_db"]):
                    _m = _acc_by_hz.get(round(_c["freq_hz"], 1))
                    if _m is not None:
                        _estado = "✅ Accepted"
                        _clase = _m.classification
                        _motivo = (
                            f"Exceeds prominence (≥{float(oma_prom):.0f} dB) · "
                            f"MPC {_m.complexity_pct:.0f}% → {_clase}")
                        _conf = f"{_m.confidence * 100:.0f}%"
                    else:
                        _estado = "❌ Discarded"
                        _clase = "—"
                        _motivo = (
                            f"Prominence {_c['prominence_db']:.1f} dB < threshold "
                            f"{float(oma_prom):.0f} dB (peak does not stand out from noise)")
                        _conf = "—"
                    _rows_c.append({
                        "Frequency (Hz)": round(_c["freq_hz"], 2),
                        "Prominence (dB)": round(_c["prominence_db"], 2),
                        "Singular value (dB)": round(_c["sv_db"], 1),
                        "Status": _estado,
                        "Classification": _clase,
                        "Confidence": _conf,
                        "Reason": _motivo,
                    })
                st.markdown(
                    f"### Candidate peaks — {len(_rows_c)} detected · "
                    f"{sum(1 for r in _rows_c if r['Status'].startswith('✅'))} accepted")
                st.caption(
                    "All the peaks the algorithm evaluated, with the reason for "
                    "acceptance or rejection. The ❌ ones are the gray hollow circles "
                    "in the chart: they exist, but they do not stand out enough from "
                    "the noise to be considered a mode.")
                st.dataframe(_pd_c.DataFrame(_rows_c),
                             use_container_width=True, hide_index=True)

            st.markdown(f"### OMA modal table — {len(fdd.modes)} candidates")
            import pandas as pd

            def _note(m):
                if m.classification == "harmonic":
                    return f"{m.harmonic_order}×"
                if m.classification == "spurious":
                    return "spurious"
                if m.is_harmonic:
                    return f"{m.harmonic_order}×, fn"
                return "fn"

            df_oma = pd.DataFrame([
                {
                    "Mode": m.mode_number,
                    "Frequency (Hz)": round(m.natural_frequency_hz, 3),
                    "Damping (%)": round(m.damping_ratio_pct, 3),
                    "Complexity (%)": round(m.complexity_pct, 1),
                    "Note": _note(m),
                    "Confidence": round(m.confidence, 2),
                }
                for m in fdd.modes
            ])

            def _style_row(row):
                if row["Note"] == "spurious":
                    return ["background-color: #f3f4f6; color: #6b7280"] * len(row)
                if "×" in str(row["Note"]) and "fn" not in str(row["Note"]):
                    return ["background-color: #fee2e2"] * len(row)
                return ["background-color: #dcfce7"] * len(row)

            try:
                st.dataframe(
                    df_oma.style.apply(_style_row, axis=1),
                    use_container_width=True, hide_index=True,
                )
            except Exception:
                st.dataframe(df_oma, use_container_width=True, hide_index=True)

            # Caption final
            modal_plot_caption(
                text=(
                    f"FDD result: {fdd.n_segments} Welch segments · "
                    f"nperseg = {fdd.nperseg} · "
                    f"Δf = {fdd.frequencies_hz[1]:.2f} Hz. "
                    "Complex mode shapes available in the Mode Shapes Tab."
                ),
                norm_ref="ISO 20816 + ISO 7626-6 sec. 6.4",
                algorithm="FDD · Brincker, Zhang, Andersen 2001 · MPC Pappa & Eishan 1995",
            )

            # --- Diagnóstico automático del ensayo (interpretación + validez) -
            try:
                from core.modal.oma_interpret import interpret_fdd
                _diag = interpret_fdd(fdd)
                st.markdown("#### 🔎 Automatic test diagnosis")
                _sev_ui = {"ok": st.success, "info": st.info,
                           "warn": st.warning, "crit": st.error}
                for _o in _diag["observations"]:
                    _sev_ui.get(_o.severity, st.info)(
                        f"**{_o.title}** — {_o.detail}")
                st.markdown(f"**Technical conclusion:** {_diag['conclusion']}")
            except Exception as _e_diag:  # noqa: BLE001
                st.caption(f"(Could not generate the diagnosis: {_e_diag})")

            # --- Panel explicativo: cómo leer los resultados OMA -------------
            with st.expander("❓ How do you read these results? "
                             "(scale, peaks, damping, complexity, confidence)",
                             expanded=False):
                st.markdown(
                    "**The chart (PSD Singular Values, in dB)**\n\n"
                    "The vertical axis is in **decibels: `dB = 10·log₁₀(singular "
                    "value)`**. It is a **relative and logarithmic** scale, not an "
                    "absolute physical value — that is why it starts near 0 dB (the "
                    "highest-energy peak) and drops to negative values (−40, −60 dB) in "
                    "the low-energy zones. What matters **is not the number "
                    "itself, but the SHAPE**: each **local peak** marks a frequency "
                    "where the structure responds with much more energy → a possible "
                    "**natural mode**. The valleys between peaks are zones without "
                    "resonance."
                )
                st.markdown(
                    "**Why a peak is marked as a natural mode (🟢)**\n\n"
                    "The **FDD** algorithm (Frequency Domain Decomposition, Brincker "
                    "2001) decomposes the spectral density matrix into singular "
                    "values. A peak is classified as a **natural mode** if: (1) "
                    "it is a clear local maximum of the 1st singular value, (2) its "
                    "mode shape is **coherent** (high purity), and (3) it does **not** "
                    "coincide with a multiple of the running speed (that would be a "
                    "harmonic 🔴) nor is it an isolated peak with no physical shape "
                    "(spurious ⚪)."
                )
                st.markdown(
                    "**The table parameters**\n\n"
                    "- **Frequency (Hz):** where the peak is = natural frequency "
                    "of the mode.\n"
                    "- **Damping (%):** damping, estimated by the "
                    "**half-power (−3 dB)** method: the wider the peak, the more "
                    "damped. Typical structural values: 0.5–5%. High "
                    "values (>10%) usually indicate a short record or a poorly defined mode.\n"
                    "- **Complexity / MPC (%):** *Modal Phase Collinearity* (Pappa "
                    "1995). A real mode has its points moving in phase (0% = "
                    "normal, clean mode). High complexity = scattered phases = "
                    "possible noise or mode overlap.\n"
                    "- **Confidence:** combined index (peak sharpness + shape "
                    "coherence + harmonic separation). It is a relative guide, not "
                    "a probability."
                )
                st.info(
                    "Rule of thumb: trust first the **well-separated green "
                    "peaks with damping 0.5–5% and low complexity**. If a mode "
                    "comes out with damping >10% or high complexity, it usually improves "
                    "by **recapturing with more record time** (raise T_low).",
                    icon="💡")


# ---------------------------------------------------------------------
# Tab 5 — Mode Shapes (visualización)
# ---------------------------------------------------------------------
if _active_modal_tab == "🎬 Mode Shapes 3D":
    modal_section_header(
        title="Mode Shape Visualization",
        subtitle="5 complementary representations of the same natural mode",
        norm_ref="ISO 7626-6 sec. 7.2",
        icon="🎬",
    )

    fdd = st.session_state.get("modal_oma_result")
    if fdd is None or not fdd.modes:
        modal_empty_state(
            icon="🎬",
            title="No modes identified yet",
            description=(
                "Mode shape visualization requires having run a "
                "modal analysis in the OMA Tab (or the future EMA Tab). Once "
                "the modes are identified, come back here to visualize the "
                "mode shape of each one from 5 perspectives: bar chart, "
                "complexity polar, AutoMAC matrix, Campbell diagram and "
                "3D arrows over the asset."
            ),
            cta_label="Go to the OMA Tab and run FDD",
            norm_ref="ISO 7626-6 sec. 7.2",
        )
    else:
        # ─── Selector global de modo (siempre arriba) ────────────────
        mode_options = {
            f"Mode {m.mode_number} · {m.natural_frequency_hz:.2f} Hz · "
            f"ζ={m.damping_ratio_pct:.2f}% · {m.classification}":
            m for m in fdd.modes
        }
        pick = st.selectbox(
            "Mode under analysis",
            list(mode_options.keys()),
            key="ms_pick",
            help=("Select the natural mode to visualize. The expanders below "
                  "update automatically with the selected mode."),
        )
        mode_sel = mode_options[pick]

        # KPI row del modo seleccionado
        _conf_color = {
            "natural": "green",
            "harmonic": "red",
            "spurious": "gray",
        }.get(mode_sel.classification, "navy")
        modal_kpi_row([
            (f"{mode_sel.natural_frequency_hz:.2f} Hz", "Natural frequency",
             "of the identified mode", "cyan"),
            (f"{mode_sel.damping_ratio_pct:.2f} %", "Damping ratio",
             "damping factor", "navy"),
            (f"{mode_sel.complexity_pct:.1f} %", "MPC complexity",
             "< 40% real · > 75% spurious", "amber"),
            (mode_sel.classification.upper(), "Classification",
             f"confidence {mode_sel.confidence*100:.0f}%", _conf_color),
        ])

        from core.modal.modal_animator import (
            build_bar_chart_mode_shape,
            build_arrows_3d_wireframe,
            build_complexity_polar_plot,
            build_mac_matrix_plot,
            build_campbell_diagram,
        )
        from core.modal.oma_engine import compute_mac_matrix, detect_redundant_modes

        st.divider()

        # ═══════════════════════════════════════════════════════════════
        # EXPANDER 1 — Nivel 1 Bar chart (abierto por default)
        # ═══════════════════════════════════════════════════════════════
        with st.expander(
            "📊  Bar chart 2D — Mode shape magnitude + phase  ·  ISO 7626-6 sec. 7.2",
            expanded=True,
        ):
            modal_plot_caption(
                text=(
                    "Magnitude (normalized) and phase of each component of the mode "
                    "shape vector. It is the most mathematically direct "
                    "representation and valid under the standard."
                ),
                norm_ref="ISO 7626-6 sec. 7.2",
                algorithm="Complex mode shape vector from FDD",
            )
            fig_bar = build_bar_chart_mode_shape(
                mode_shape=mode_sel.mode_shape,
                channel_names=fdd.channel_names,
                mode_label=(f"Mode {mode_sel.mode_number} · "
                              f"{mode_sel.natural_frequency_hz:.2f} Hz · "
                              f"ζ = {mode_sel.damping_ratio_pct:.3f}%"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # EXPANDER 2 — Complexity Polar Plot
        # ═══════════════════════════════════════════════════════════════
        with st.expander(
            f"🎯  Complexity Polar Plot — MPC = {mode_sel.complexity_pct:.1f}%  ·  "
            "Pappa & Eishan 1995",
            expanded=False,
        ):
            modal_plot_caption(
                text=(
                    "Each arrow is a component of the mode shape in the complex "
                    "plane. **Collinear vectors** (aligned at 0° or 180°) "
                    "= real natural mode. **Scattered vectors** = complex "
                    "or spurious mode. Standard modal complexity visualization."
                ),
                norm_ref="ISO 7626-6 sec. 7.2",
                algorithm="Modal Phase Collinearity (Pappa & Eishan 1995)",
            )
            fig_pol = build_complexity_polar_plot(
                mode_shape=mode_sel.mode_shape,
                channel_names=fdd.channel_names,
                mode_label=(f"Mode {mode_sel.mode_number} · "
                              f"{mode_sel.natural_frequency_hz:.2f} Hz · "
                              f"MPC complexity = {mode_sel.complexity_pct:.1f}% · "
                              f"class: {mode_sel.classification}"),
            )
            st.plotly_chart(fig_pol, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # EXPANDER 3 — AutoMAC Matrix
        # ═══════════════════════════════════════════════════════════════
        mac = compute_mac_matrix(fdd.modes)
        labels = [f"{m.natural_frequency_hz:.1f} Hz" for m in fdd.modes]
        redundants = detect_redundant_modes(fdd.modes, threshold=0.7)

        _redundant_warning = f"  ·  ⚠ {len(redundants)} redundant pairs" if redundants else ""

        with st.expander(
            f"🔗  AutoMAC Matrix — Correlation between modes{_redundant_warning}  ·  "
            "ISO 7626-6 sec. 6.5 + API 684 sec. 1.6",
            expanded=False,
        ):
            modal_plot_caption(
                text=(
                    "Modal Assurance Criterion between each pair of modes. "
                    "**Diagonal = 1** (always). **Off-diagonal > 0.7** "
                    "indicates redundant modes (same mode identified twice "
                    "— one should be removed). Standard AutoMAC matrix."
                ),
                norm_ref="ISO 7626-6 sec. 6.5 · API 684 sec. 1.6",
                algorithm="AutoMAC matrix (Allemang & Brown 1982)",
            )
            view_3d = st.toggle("3D bar view (professional)",
                                  value=False, key="mac_3d_toggle")
            fig_mac = build_mac_matrix_plot(
                mac, labels, title="AutoMAC", use_3d=view_3d,
            )
            st.plotly_chart(fig_mac, use_container_width=True)

            # Diagnóstico de redundancia
            if redundants:
                modal_status_banner(
                    title=f"{len(redundants)} linearly dependent mode pairs",
                    detail=(
                        "MAC off-diagonal > 0.7. Detected pairs: " +
                        ", ".join([
                            f"Mode {i+1} ({fdd.modes[i].natural_frequency_hz:.1f} Hz) ↔ "
                            f"Mode {j+1} ({fdd.modes[j].natural_frequency_hz:.1f} Hz, "
                            f"MAC={mac_val:.2f})"
                            for i, j, mac_val in redundants[:5]
                        ]) + ". Consider removing the one with lower confidence."
                    ),
                    severity="warning",
                )
            else:
                modal_status_banner(
                    title="Clean modal set — all modes are linearly independent",
                    detail="Off-diagonal MAC < 0.7 in all pairs.",
                    severity="ok",
                )

        # ═══════════════════════════════════════════════════════════════
        # EXPANDER 4 — Diagrama de Campbell
        # ═══════════════════════════════════════════════════════════════
        with st.expander(
            "📈  Campbell diagram — Critical speeds  ·  API 684 sec. 1.6",
            expanded=False,
        ):
            modal_plot_caption(
                text=(
                    "Crosses the identified natural modes (horizontal lines) "
                    "against the running-speed harmonics "
                    "(inclined lines 1×, 2×, ...). The **red X's** are "
                    "critical speeds — points where a harmonic excites "
                    "a natural mode and can cause resonance."
                ),
                norm_ref="API 684 sec. 1.6",
                algorithm="Campbell diagram — standard rotor dynamics",
            )

            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                camp_rpm_min = st.number_input("RPM min", value=0, step=100,
                                                  key="camp_rpm_min")
            with col_c2:
                camp_rpm_max = st.number_input("RPM max", value=4000, step=500,
                                                  key="camp_rpm_max")
            with col_c3:
                camp_op_rpm = st.number_input("Operating speed (rpm)",
                                                value=3600, step=100,
                                                key="camp_op_rpm")

            natural_modes_for_camp = [m for m in fdd.modes
                                         if m.classification == "natural"]
            if natural_modes_for_camp:
                fig_camp, crit_speeds = build_campbell_diagram(
                    natural_frequencies_hz=[m.natural_frequency_hz
                                              for m in natural_modes_for_camp],
                    natural_freq_labels=[f"Mode {m.mode_number}"
                                           for m in natural_modes_for_camp],
                    rpm_min=float(camp_rpm_min),
                    rpm_max=float(camp_rpm_max),
                    operating_rpm=float(camp_op_rpm) if camp_op_rpm > 0 else None,
                    n_orders=6,
                    classification=[m.classification for m in natural_modes_for_camp],
                    title="Campbell diagram",
                )
                st.plotly_chart(fig_camp, use_container_width=True)

                if crit_speeds:
                    import pandas as pd
                    df_crit = pd.DataFrame([
                        {
                            "Critical speed (rpm)": round(rpm, 0),
                            "Mode": label,
                            "Frequency (Hz)": round(fn, 2),
                            "Order": f"{order}× rpm",
                            "Status": "⚠ WITHIN operating range" if (
                                camp_op_rpm > 0
                                and abs(rpm - camp_op_rpm) / max(camp_op_rpm, 1) < 0.10
                            ) else "Outside nearby operating range",
                        }
                        for rpm, fn, order, label in crit_speeds
                    ])
                    _n_dentro = sum(1 for r in crit_speeds
                                       if camp_op_rpm > 0
                                       and abs(r[0] - camp_op_rpm) / max(camp_op_rpm, 1) < 0.10)
                    if _n_dentro > 0:
                        modal_status_banner(
                            title=f"{_n_dentro} critical speed(s) WITHIN the operating range",
                            detail=(
                                "The asset operates near a mode×harmonic crossing. "
                                "Risk of resonant amplification — review API 618 "
                                "sec. 7.9.4.2.5.3.2 (separation ≥ 10%)."
                            ),
                            severity="fail",
                        )
                    st.markdown("**Critical speeds detected:**")
                    st.dataframe(df_crit, use_container_width=True, hide_index=True)
                else:
                    modal_status_banner(
                        title="No critical speeds detected",
                        detail=f"No natural mode ↔ harmonic crossing in the band "
                                 f"{camp_rpm_min}-{camp_rpm_max} rpm.",
                        severity="ok",
                    )
            else:
                st.info("No classified natural modes — no data for Campbell.")

        # ═══════════════════════════════════════════════════════════════
        # EXPANDER 5 — Nivel 2: Flechas 3D sobre activo
        # ═══════════════════════════════════════════════════════════════
        # Prioridad de fuentes de geometría:
        #   1) modal_geometry en session_state (editor de Tab Setup) — preferido
        #   2) Sensor Map del activo registrado (fallback legacy)
        #   3) Sin geometría → mensaje informativo
        _geom_session = st.session_state.get("modal_geometry")
        _adhoc_meta_for_3d = st.session_state.get("modal_adhoc_meta")
        _inst_key_for_3d = st.session_state.get("modal_inst", "")
        _inst_for_3d = None
        _geom_source = "none"  # "modal_geometry" | "sensor_map" | "none"

        # Fuente preferida: editor de geometría
        if _geom_session is not None and getattr(_geom_session, "sensors", None):
            _geom_source = "modal_geometry"
            _3d_status_label = (f"edited geometry · "
                                  f"{len(_geom_session.sensors)} sensors")
        else:
            # Fallback: Sensor Map del activo registrado
            if _adhoc_meta_for_3d:
                _3d_status_label = "not available · ad-hoc mode without geometry"
            elif _inst_key_for_3d and _inst_key_for_3d != "(seleccionar)":
                try:
                    from core.instance_state import get_instance as _get_inst_3d
                    _inst_for_3d = _get_inst_3d(_inst_key_for_3d)
                    if _inst_for_3d:
                        _geom_source = "sensor_map"
                        _3d_status_label = "Sensor Map (legacy fallback)"
                    else:
                        _3d_status_label = "not available · asset without sensors"
                except Exception:
                    _inst_for_3d = None
                    _3d_status_label = "not available · error loading asset"
            else:
                _3d_status_label = "not available · no asset in Setup"

        with st.expander(
            f"🌐  3D arrows over the asset layout · {_3d_status_label}",
            expanded=False,
        ):
            modal_plot_caption(
                text=(
                    "3D visualization of the mode shape over the asset's real "
                    "geometry. Each arrow indicates the direction of motion "
                    "of a sensor in the selected mode. Green = in-phase · "
                    "red = anti-phase. Preferred source: the geometry editor "
                    "in the Setup Tab. Fallback: the asset's Sensor Map."
                ),
                norm_ref="ISO 7626-6 sec. 7.2",
                algorithm="Plotly Cone3D + mode shape vector (phase-signed)",
            )

            if _geom_source == "modal_geometry":
                # === Camino preferido: usa el editor de geometría ===
                from core.modal.geometry_3d import build_geometry_with_mode_shape

                # Controles de visualización
                _ms_c1, _ms_c2, _ms_c3, _ms_c4 = st.columns([1, 1, 1, 1.5])
                with _ms_c1:
                    # v3.31.199 HOTFIX: animación OFF default — agrega 20
                    # frames × N traces que pesan ~5x lo del plot estático
                    _animate_ms = st.toggle(
                        "🎞 Animate",
                        value=False, key="modeshape_animate_toggle",
                        help=("Generates N frames for Play. ⚠ Increases memory "
                              "usage — enable it only when you need it."),
                    )
                with _ms_c2:
                    _show_arrows = st.toggle(
                        "DOF arrows",
                        value=False, key="modeshape_arrows_toggle",
                        help="Shows Cone arrows on each sensor. Off = only "
                             "the mesh heatmap (Watermelon visualization).",
                    )
                with _ms_c3:
                    # v3.31.199 HOTFIX: ghost OFF default — agrega 4 traces
                    # extra pesados que duplican consumo de memoria del browser
                    _show_ghost = st.toggle(
                        "Original ghost",
                        value=False, key="modeshape_ghost_toggle",
                        help="Semi-transparent overlay of the undeformed state "
                             "for comparison. ⚠ Increases memory usage.",
                    )
                with _ms_c4:
                    _cmap = st.selectbox(
                        "Colormap",
                        ["RdBu_r", "RdYlBu_r", "Spectral_r", "Jet", "Viridis"],
                        index=0, key="modeshape_cmap",
                        help="Red in-phase, blue anti-phase (RdBu_r). Alternative: Jet.",
                    )

                # ===== Banner KPI grande estilo industrial premium =====
                _fn_hz = float(mode_sel.natural_frequency_hz)
                _fn_cpm = _fn_hz * 60.0
                _zeta = float(mode_sel.damping_ratio_pct)
                _running_rpm_for_order = float(
                    st.session_state.get("camp_op_rpm", 0) or 3600.0
                )
                _order = _fn_cpm / max(_running_rpm_for_order, 1.0)
                _mpc_pct = float(getattr(mode_sel, "complexity_pct", 0.0))
                _q_factor = 1.0 / (2 * max(_zeta / 100.0, 1e-6))
                _cls = getattr(mode_sel, "classification", "natural")
                _cls_color = {"natural": "#16a34a",
                                "harmonic": "#D89B22",
                                "spurious": "#dc2626"}.get(_cls, "#475569")
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(90deg, #0F1E3D 0%, #1e3a5f 100%);
                                color: white; border-radius: 12px;
                                padding: 14px 22px; margin: 14px 0;
                                display: flex; gap: 28px; align-items: center;
                                flex-wrap: wrap;">
                      <div>
                        <div style="font-size:10.5px; opacity:0.7; letter-spacing:1px;
                                    text-transform: uppercase;">Identified mode</div>
                        <div style="font-size: 26px; font-weight: 700;">
                            #{mode_sel.mode_number}</div>
                      </div>
                      <div style="border-left: 1px solid rgba(255,255,255,0.18);
                                  padding-left: 22px;">
                        <div style="font-size:10.5px; opacity:0.7; letter-spacing:1px;
                                    text-transform: uppercase;">Frequency</div>
                        <div style="font-size: 24px; font-weight: 600;">
                            {_fn_hz:.2f} Hz</div>
                        <div style="font-size: 12px; opacity:0.7;">
                            {_fn_cpm:,.0f} CPM · {_order:.3f}× run</div>
                      </div>
                      <div style="border-left: 1px solid rgba(255,255,255,0.18);
                                  padding-left: 22px;">
                        <div style="font-size:10.5px; opacity:0.7; letter-spacing:1px;
                                    text-transform: uppercase;">Damping ζ</div>
                        <div style="font-size: 24px; font-weight: 600;">
                            {_zeta:.3f}%</div>
                        <div style="font-size: 12px; opacity:0.7;">
                            Q = {_q_factor:.1f}</div>
                      </div>
                      <div style="border-left: 1px solid rgba(255,255,255,0.18);
                                  padding-left: 22px;">
                        <div style="font-size:10.5px; opacity:0.7; letter-spacing:1px;
                                    text-transform: uppercase;">MPC</div>
                        <div style="font-size: 24px; font-weight: 600;">
                            {_mpc_pct:.1f}%</div>
                        <div style="font-size: 12px; opacity:0.7;">
                            complexity</div>
                      </div>
                      <div style="border-left: 1px solid rgba(255,255,255,0.18);
                                  padding-left: 22px;">
                        <div style="font-size:10.5px; opacity:0.7; letter-spacing:1px;
                                    text-transform: uppercase;">Classification</div>
                        <div style="font-size: 18px; font-weight: 600; color:{_cls_color};">
                            {_cls.upper()}</div>
                        <div style="font-size: 12px; opacity:0.7;">MPC + harmonic check</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ===== Presets de vista FIJA + Plotly animation =====
                # En vez de drag-rotate libre (que Plotly resetea al Play),
                # damos al usuario PRESETS de cámara. El video corre desde
                # el preset elegido y NO se mueve la cámara automáticamente.
                _cam_preset_key = f"_modal_cam_preset_{mode_sel.mode_number}"
                if _cam_preset_key not in st.session_state:
                    st.session_state[_cam_preset_key] = "lateral"

                _PRESETS = {
                    "lateral": {
                        "label": "🔍 Side",
                        "eye": dict(x=0.0, y=2.4, z=0.4),
                        "up": dict(x=0, y=0, z=1),
                        "help": "Side A view — classic vertical bending",
                    },
                    "lateral_opp": {
                        "label": "↩ Opposite side",
                        "eye": dict(x=0.0, y=-2.4, z=0.4),
                        "up": dict(x=0, y=0, z=1),
                        "help": "Mirror side view (opposite side)",
                    },
                    "frontal": {
                        "label": "👁 Front",
                        "eye": dict(x=2.4, y=0.0, z=0.4),
                        "up": dict(x=0, y=0, z=1),
                        "help": "View from the free end of the shaft",
                    },
                    "posterior": {
                        "label": "👀 Rear",
                        "eye": dict(x=-2.4, y=0.0, z=0.4),
                        "up": dict(x=0, y=0, z=1),
                        "help": "View from the opposite end of the shaft",
                    },
                    "superior": {
                        "label": "⬇ Top",
                        "eye": dict(x=0.0, y=0.0, z=2.5),
                        "up": dict(x=0, y=1, z=0),
                        "help": "Plan view from above",
                    },
                    "isometrica": {
                        "label": "🔮 Isometric",
                        "eye": dict(x=1.6, y=1.6, z=1.2),
                        "up": dict(x=0, y=0, z=1),
                        "help": "Balanced 3D view — classic",
                    },
                    "diagonal": {
                        "label": "🎯 3/4 view",
                        "eye": dict(x=1.8, y=1.2, z=0.6),
                        "up": dict(x=0, y=0, z=1),
                        "help": "Low front-side diagonal",
                    },
                }

                st.markdown("**Camera view for the video**")
                st.caption(
                    "Select the plane from which you want the animation to be "
                    "seen. The video will run from that fixed angle."
                )
                _preset_cols = st.columns(len(_PRESETS))
                for _idx, (_pkey, _pdata) in enumerate(_PRESETS.items()):
                    with _preset_cols[_idx]:
                        _is_active = (st.session_state[_cam_preset_key] == _pkey)
                        if st.button(
                            _pdata["label"],
                            key=f"preset_{_pkey}_{mode_sel.mode_number}",
                            type="primary" if _is_active else "secondary",
                            use_container_width=True,
                            help=_pdata["help"],
                        ):
                            st.session_state[_cam_preset_key] = _pkey
                            st.rerun()

                _selected = _PRESETS[st.session_state[_cam_preset_key]]

                # v3.31.199 HOTFIX: reducidos a 20 frames (era 48) y ghost
                # OFF por default para bajar consumo memoria browser.
                # Si el usuario quiere mas detalle hace click en toggles.
                fig_3d = build_geometry_with_mode_shape(
                    geom=_geom_session,
                    mode_shape=mode_sel.mode_shape,
                    channel_names=fdd.channel_names,
                    mode_label=(f"Mode {mode_sel.mode_number} · "
                                  f"{mode_sel.natural_frequency_hz:.2f} Hz · "
                                  f"ζ = {mode_sel.damping_ratio_pct:.3f}%"),
                    animate=_animate_ms,
                    n_frames=20,
                    frame_duration_ms=200,
                    show_arrows=_show_arrows,
                    show_ghost=_show_ghost,
                    colormap=_cmap,
                    camera_eye=_selected["eye"],
                    camera_up=_selected["up"],
                )
                st.plotly_chart(
                    fig_3d,
                    use_container_width=True,
                    key=f"modeshape_3d_{mode_sel.mode_number}",
                    config={"scrollZoom": True,
                            "displayModeBar": True,
                            "displaylogo": False},
                )

                # ===== Botones Descargar MP4 / GIF con header KPI integrado =====
                # Helper: resolver nombre del activo de forma defensiva
                def _resolve_asset_name():
                    _adhoc_safe = st.session_state.get("modal_adhoc_meta")
                    if _adhoc_safe and isinstance(_adhoc_safe, dict):
                        return _adhoc_safe.get("equipment_name", "Ad-hoc asset")
                    if _inst_for_3d is not None and hasattr(_inst_for_3d, "display_name"):
                        return _inst_for_3d.display_name
                    return (_geom_session.name if _geom_session
                            else "Watermelon Modal")

                _exp_c1, _exp_c2, _exp_c3 = st.columns([1.2, 1.2, 2.6])
                with _exp_c1:
                    _gen_mp4 = st.button(
                        "🎥 Generate MP4 Video",
                        key="modeshape_gen_mp4",
                        use_container_width=True,
                        type="primary",
                        help="MP4 H.264 (best quality, ~2-5 MB, compatible "
                             "with WhatsApp/iPhone/Android). Render ~30 s.",
                    )
                with _exp_c2:
                    _gen_gif = st.button(
                        "🖼 Generate GIF",
                        key="modeshape_gen_gif",
                        use_container_width=True,
                        help="Animated GIF (universal alternative, ~5-10 MB). "
                             "Render ~25 s.",
                    )

                # ---- MP4 ----
                if _gen_mp4:
                    from core.modal.geometry_3d import export_mode_shape_mp4
                    _asset_lbl = _resolve_asset_name()
                    _prog_bar = st.progress(0.0, text="Starting render…")
                    def _on_progress(idx, total, stage):
                        pct = idx / max(total, 1)
                        if stage == "encoding":
                            _prog_bar.progress(
                                1.0,
                                text=f"Encoding H.264 with ffmpeg… "
                                       f"({total} frames ready)")
                        else:
                            _prog_bar.progress(
                                pct,
                                text=f"Frame {idx + 1}/{total} "
                                       f"({pct*100:.0f}%) · plotly→PNG via kaleido")
                    try:
                        _mp4_bytes = export_mode_shape_mp4(
                            geom=_geom_session,
                            mode_shape=mode_sel.mode_shape,
                            channel_names=fdd.channel_names,
                            mode_number=mode_sel.mode_number,
                            freq_hz=_fn_hz,
                            damping_pct=_zeta,
                            running_rpm=_running_rpm_for_order,
                            classification=_cls,
                            mpc_pct=_mpc_pct,
                            n_frames=30, fps=12, n_loops=2,
                            width_px=1280, height_px=720,
                            colormap=_cmap,
                            show_ghost=_show_ghost,
                            asset_name=_asset_lbl,
                            quality=8,
                            progress_cb=_on_progress,
                        )
                        st.session_state["_modeshape_mp4"] = _mp4_bytes
                        st.session_state["_modeshape_mp4_filename"] = (
                            f"modeshape_M{mode_sel.mode_number}_"
                            f"{_fn_hz:.1f}Hz.mp4"
                        )
                        st.session_state.pop("_modeshape_gif", None)
                        _prog_bar.empty()
                        st.success(
                            f"✓ MP4 ready · "
                            f"{len(_mp4_bytes) / 1024:.0f} KB. "
                            "Click below to download."
                        )
                    except Exception as exc:  # noqa: BLE001
                        _prog_bar.empty()
                        import traceback
                        st.error(
                            f"**MP4 export error:** `{type(exc).__name__}: {exc}`"
                        )
                        with st.expander("Technical detail (traceback)"):
                            st.code(traceback.format_exc(), language="text")
                        st.info(
                            "Common causes: (1) imageio-ffmpeg did not download the "
                            "binary, (2) kaleido failed to render Plotly, "
                            "(3) Streamlit Cloud out-of-memory. "
                            "Try 'Generate GIF' as an alternative."
                        )

                # ---- GIF ----
                if _gen_gif:
                    from core.modal.geometry_3d import export_mode_shape_gif
                    _asset_lbl = _resolve_asset_name()
                    _prog_bar_g = st.progress(0.0, text="Starting GIF render…")
                    def _on_progress_g(idx, total, stage):
                        pct = idx / max(total, 1)
                        if stage == "encoding":
                            _prog_bar_g.progress(
                                1.0,
                                text=f"Assembling GIF… ({total} frames ready)")
                        else:
                            _prog_bar_g.progress(
                                pct,
                                text=f"Frame {idx + 1}/{total} ({pct*100:.0f}%)")
                    try:
                        _gif_bytes = export_mode_shape_gif(
                            geom=_geom_session,
                            mode_shape=mode_sel.mode_shape,
                            channel_names=fdd.channel_names,
                            mode_number=mode_sel.mode_number,
                            freq_hz=_fn_hz,
                            damping_pct=_zeta,
                            running_rpm=_running_rpm_for_order,
                            classification=_cls,
                            mpc_pct=_mpc_pct,
                            n_frames=24,
                            frame_duration_ms=280,
                            width_px=1280, height_px=720,
                            colormap=_cmap,
                            show_ghost=_show_ghost,
                            asset_name=_asset_lbl,
                            progress_cb=_on_progress_g,
                        )
                        st.session_state["_modeshape_gif"] = _gif_bytes
                        st.session_state["_modeshape_gif_filename"] = (
                            f"modeshape_M{mode_sel.mode_number}_"
                            f"{_fn_hz:.1f}Hz.gif"
                        )
                        st.session_state.pop("_modeshape_mp4", None)
                        _prog_bar_g.empty()
                        st.success(
                            f"✓ GIF ready · "
                            f"{len(_gif_bytes) / 1024:.0f} KB."
                        )
                    except Exception as exc:  # noqa: BLE001
                        _prog_bar_g.empty()
                        import traceback
                        st.error(f"**GIF export error:** `{type(exc).__name__}: {exc}`")
                        with st.expander("Technical detail"):
                            st.code(traceback.format_exc(), language="text")

                # ---- Download buttons (muestra el que esté listo) ----
                with _exp_c3:
                    if st.session_state.get("_modeshape_mp4"):
                        st.download_button(
                            "⬇ Download generated MP4 Video",
                            data=st.session_state["_modeshape_mp4"],
                            file_name=st.session_state.get(
                                "_modeshape_mp4_filename", "modeshape.mp4"),
                            mime="video/mp4",
                            use_container_width=True,
                            type="primary",
                        )
                    elif st.session_state.get("_modeshape_gif"):
                        st.download_button(
                            "⬇ Download generated GIF",
                            data=st.session_state["_modeshape_gif"],
                            file_name=st.session_state.get(
                                "_modeshape_gif_filename", "modeshape.gif"),
                            mime="image/gif",
                            use_container_width=True,
                        )

                # ===== Botón Enviar a Reporte Watermelon =====
                st.divider()
                modal_section_header(
                    title="Inject into the Watermelon Report",
                    subtitle=(
                        "Generates PNG snapshots of all identified modes + "
                        "AutoMAC + summary table and adds them to your "
                        "current report. They are rendered into the standard SIGA PDF "
                        "alongside the rest of the figures."
                    ),
                    norm_ref="ISO 7626-6 sec. 8 · Modal documentation",
                )

                _rep_c1, _rep_c2 = st.columns([2, 3])
                with _rep_c1:
                    _include_non_natural = st.toggle(
                        "Include harmonic/spurious (advanced)",
                        value=False,
                        key=f"modal_report_inc_non_nat_{mode_sel.mode_number}",
                        help=(
                            "By default only natural modes are injected "
                            "(physical modes of the asset). Enable it to "
                            "also include the running-speed harmonics "
                            "(1×, 2×, ...) and spurious modes. Useful for "
                            "auditing or advanced technical reports."
                        ),
                    )
                with _rep_c2:
                    _natural_count = sum(
                        1 for m in fdd.modes
                        if getattr(m, "classification", "natural") == "natural"
                    )
                    _total_count = len(fdd.modes)
                    _will_inject = (_total_count if _include_non_natural
                                       else _natural_count)
                    st.caption(
                        f"**{_will_inject}** modes × 3 plots will be injected "
                        f"({_will_inject * 3} mode figures) + AutoMAC "
                        f"heatmap + summary table = "
                        f"**{_will_inject * 3 + 2} items** into the report."
                    )

                if st.button(
                    "📄 Send all modes to the Report",
                    key=f"modal_send_report_{mode_sel.mode_number}",
                    type="primary",
                    use_container_width=True,
                    help="The report is saved in your session. View it "
                         "and download the PDF from the Reports page.",
                ):
                    from core.modal.modal_report import (
                        build_modal_report_items,
                        append_modal_items_to_report,
                    )
                    _rep_prog = st.progress(0.0, text="Generating snapshots…")

                    def _rep_cb(idx, total, stage):
                        _pct = idx / max(total, 1)
                        _rep_prog.progress(
                            min(_pct, 1.0),
                            text=f"{stage} ({idx + 1}/{total})",
                        )

                    try:
                        _asset_lbl_rep = _resolve_asset_name()
                        _new_items = build_modal_report_items(
                            fdd_result=fdd,
                            geom=_geom_session,
                            include_non_natural=_include_non_natural,
                            asset_name=_asset_lbl_rep,
                            method="OMA",
                            running_rpm=_running_rpm_for_order,
                            colormap=_cmap,
                            camera_eye=_selected["eye"],
                            camera_up=_selected["up"],
                            progress_cb=_rep_cb,
                        )
                        _n_added = append_modal_items_to_report(_new_items)
                        _rep_prog.empty()
                        st.success(
                            f"✓ **{_n_added} items** added to the report. "
                            "Go to **Reports** (sidebar) → "
                            "you will see all modal figures listed to "
                            "include in the final PDF."
                        )
                    except Exception as _exc:  # noqa: BLE001
                        _rep_prog.empty()
                        import traceback as _tb
                        st.error(
                            f"Error generating snapshots: "
                            f"`{type(_exc).__name__}: {_exc}`"
                        )
                        with st.expander("Technical detail"):
                            st.code(_tb.format_exc(), language="text")

                # Diagnóstico de matching
                _ch_set = {n.strip().upper() for n in fdd.channel_names}
                _geom_set = {s.name.strip().upper() for s in _geom_session.sensors}
                _matched = _ch_set & _geom_set
                _missing = _ch_set - _geom_set
                if _missing:
                    modal_status_banner(
                        title=f"{len(_missing)} channel(s) without a sensor in the geometry",
                        detail=(
                            f"{len(_matched)}/{len(_ch_set)} channels matched "
                            f"with geometry sensors. Without match: "
                            f"{', '.join(sorted(_missing)[:8])}"
                            f"{' …' if len(_missing) > 8 else ''}. "
                            "Add or rename sensors in the Setup Tab → "
                            "3D Geometry to cover all channels."
                        ),
                        severity="warning",
                    )
                else:
                    modal_status_banner(
                        title=f"100% match · all {len(_ch_set)} channels in the geometry",
                        detail="The arrows faithfully represent the complete mode shape.",
                        severity="ok",
                    )

            elif _geom_source == "sensor_map" and _inst_for_3d is not None:
                # === Camino legacy: Sensor Map ===
                sensors_3d = []
                for ch_name in fdd.channel_names:
                    match = None
                    for s in (_inst_for_3d.sensors or []):
                        if str(s.get("plane_label", "")).strip().upper() == ch_name.strip().upper():
                            match = s
                            break
                    if match and match.get("position_3d") and match.get("dof_direction"):
                        sensors_3d.append({
                            "name": ch_name,
                            "position_3d": match["position_3d"],
                            "dof_direction": match["dof_direction"],
                        })

                if len(sensors_3d) == len(fdd.channel_names):
                    positions = [tuple(s["position_3d"]) for s in sensors_3d]
                    directions = [tuple(s["dof_direction"]) for s in sensors_3d]
                    fig_3d = build_arrows_3d_wireframe(
                        mode_shape=mode_sel.mode_shape,
                        channel_positions_3d=positions,
                        channel_directions_3d=directions,
                        channel_names=fdd.channel_names,
                        mode_label=(f"Mode {mode_sel.mode_number} · "
                                      f"{mode_sel.natural_frequency_hz:.2f} Hz — "
                                      f"green: in-phase · red: anti-phase"),
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                    modal_status_banner(
                        title="Using Sensor Map (without edited geometry)",
                        detail=(
                            "For a richer visualization with mechanical-train "
                            "blocks (motor, coupling, casing), build "
                            "the geometry in the Setup Tab → 3D Geometry."
                        ),
                        severity="info",
                    )
                else:
                    modal_status_banner(
                        title=f"Partial 3D configuration — {len(sensors_3d)}/{len(fdd.channel_names)} channels",
                        detail=(
                            "Complete the 'Modal configuration' expander of each "
                            "sensor in Machinery Library or build the "
                            "geometry in the Setup Tab → 3D Geometry."
                        ),
                        severity="warning",
                    )

            elif _adhoc_meta_for_3d:
                modal_status_banner(
                    title="Ad-hoc mode · build the 3D geometry to enable the arrows",
                    detail=(
                        "Without a registered asset, there is no Sensor Map. But "
                        "you can go to the Setup Tab → 'Asset 3D geometry', "
                        "apply a template (motor+compressor, turbine+gen, "
                        "pump+motor) or build it manually, and the 3D "
                        "arrows enable immediately with the match by channel "
                        "name. Levels 1-4 already comply with ISO 7626-6 sec. 7.2."
                    ),
                    severity="info",
                )
            else:
                modal_status_banner(
                    title="Select an asset or build the geometry",
                    detail=(
                        "To enable the 3D arrows: (a) select a registered "
                        "asset in the Setup Tab with 3D sensors configured, "
                        "or (b) build the geometry manually in the Setup Tab → "
                        "Asset 3D geometry."
                    ),
                    severity="info",
                )

        # ═══════════════════════════════════════════════════════════════
        # ROADMAP nota — Mesh3D animado (Sprint próximo)
        # ═══════════════════════════════════════════════════════════════
        st.caption(
            "📅 **Roadmap next sprint:** Level 3 — Animated Mesh3D with "
            "professional Watermelon colormap. The current Levels 1-2 (bar chart + "
            "3D arrows) already comply with ISO 7626-6 sec. 7.2 — the animated mesh is "
            "a visual feature, not a normative requirement."
        )


# ---------------------------------------------------------------------
# Tab 6 — FEA Compare
# ---------------------------------------------------------------------
if _active_modal_tab == "🧮 FEA Compare":
    modal_section_header(
        title="EMA / OMA ↔ FEA Correlation",
        subtitle="Cross-validation of the numerical model against experimental results",
        norm_ref="API 684 sec. 1.6 · MAC ≥ 0.7 + Δf ≤ 10%",
        icon="🧮",
    )

    from core.modal.fea_compare import (
        load_fea_json,
        compute_fea_experimental_cross_mac,
        pair_modes,
        build_cross_mac_heatmap,
        example_fea_payload,
    )

    # ----- Resolver fuente experimental -----
    fdd_for_fea = st.session_state.get("modal_oma_result")
    peaks_for_fea = st.session_state.get("modal_peaks", [])

    exp_source = None
    exp_label = None
    exp_freqs: list = []
    exp_shapes: list = []
    exp_channels: list = []
    exp_mode_labels: list = []

    if fdd_for_fea and getattr(fdd_for_fea, "modes", None):
        exp_source = "oma"
        exp_label = "OMA · FDD result"
        exp_freqs = [m.natural_frequency_hz for m in fdd_for_fea.modes]
        exp_shapes = [m.mode_shape for m in fdd_for_fea.modes]
        exp_channels = list(fdd_for_fea.channel_names)
        exp_mode_labels = [f"M{m.mode_number} ({m.natural_frequency_hz:.1f} Hz)"
                            for m in fdd_for_fea.modes]
    elif peaks_for_fea:
        # EMA peaks no tienen mode_shape multi-canal nativo aún — solo freq.
        # En ese caso solo podemos hacer correlación de frecuencias, no MAC.
        exp_source = "ema_freq_only"
        exp_label = "EMA · frequencies only (no multi-channel shapes)"
        exp_freqs = [p.frequency_hz for p in peaks_for_fea]
        exp_shapes = []
        exp_channels = []
        exp_mode_labels = [f"P{i+1} ({p.frequency_hz:.1f} Hz)"
                            for i, p in enumerate(peaks_for_fea)]

    if exp_source is None:
        modal_empty_state(
            icon="🧮",
            title="No experimental modes to compare",
            description=(
                "You need to have run at least one experimental analysis "
                "before comparing against FEA: run the FDD in the OMA Tab "
                "(preferred — it delivers multi-channel mode shapes) or detect "
                "peaks in the EMA Tab. Then come back to this tab and upload your FEA JSON."
            ),
            cta_label="Switch to the OMA Tab or EMA Tab",
            norm_ref="API 684 sec. 1.6",
        )
    else:
        col_src1, col_src2, col_src3 = st.columns(3)
        col_src1.metric("Experimental source", exp_label)
        col_src2.metric("Experimental modes", len(exp_freqs))
        col_src3.metric("Channels", len(exp_channels) if exp_channels else "—")

        st.divider()
        st.markdown("**1 · Upload the FEA file**")
        st.caption(
            "Watermelon JSON format — export from ANSYS/Nastran/Abaqus with "
            "`freq_hz` + `mode_shape` + `dof_names` that match the channels "
            "of your experimental identification. Supports real or complex shapes."
        )

        col_up, col_tpl = st.columns([2, 1])
        with col_up:
            fea_up = st.file_uploader(
                "FEA JSON", type=["json"], key="fea_json_up",
                help="Upcoming roadmap: native parsers .rst (Ansys), "
                     ".op2 (Nastran), .odb (Abaqus). Today JSON only.",
            )
        with col_tpl:
            if exp_channels:
                tpl_json = json.dumps(example_fea_payload(exp_channels), indent=2)
            else:
                tpl_json = json.dumps(example_fea_payload(
                    [f"DOF{i+1}" for i in range(5)]), indent=2)
            st.download_button(
                "⬇ JSON Template",
                data=tpl_json,
                file_name="fea_template.json",
                mime="application/json",
                use_container_width=True,
                help="Download a template with your experimental channels "
                     "already filled in — just edit the freqs and shapes with your "
                     "real FEA values.",
            )

        fea_result = None
        if fea_up is not None:
            try:
                fea_result = load_fea_json(fea_up.getvalue().decode("utf-8"))
                st.session_state["fea_result"] = fea_result
            except Exception as exc:  # noqa: BLE001
                modal_status_banner(
                    title=f"Error parsing the FEA JSON",
                    detail=str(exc),
                    severity="fail",
                )
                fea_result = None
        elif st.session_state.get("fea_result"):
            fea_result = st.session_state["fea_result"]
            st.caption(f"Using previously loaded FEA: **{fea_result.model_name}**")

        if fea_result is not None:
            st.divider()
            st.markdown("**2 · FEA model summary**")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Model", fea_result.model_name[:30])
            mc2.metric("Software", fea_result.software[:20])
            mc3.metric("FEA modes", fea_result.n_modes)
            _fmin, _fmax = fea_result.freq_range
            mc4.metric("FEA band", f"{_fmin:.1f} – {_fmax:.1f} Hz")

            st.divider()
            st.markdown("**3 · Correlation configuration**")
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                mac_thr = st.number_input(
                    "MAC threshold for validity",
                    value=0.70, min_value=0.5, max_value=0.95, step=0.05,
                    help="API 684 sec. 1.6 / Ewins: MAC ≥ 0.7 indicates a "
                         "correlated shape. Industrial standard.",
                )
            with col_cfg2:
                freq_tol = st.number_input(
                    "Δf tolerance (%)",
                    value=10.0, min_value=2.0, max_value=30.0, step=1.0,
                    help="API 684 sec. 1.6: |Δf|/f_exp ≤ 10% is acceptable for "
                         "rotor dynamics validation. < 5% is excellent.",
                )

            st.divider()
            st.markdown("**4 · Results**")

            # ----- Caso OMA: Cross-MAC completo -----
            if exp_source == "oma":
                # Validar que los DOF names del FEA cubren los canales exp
                _exp_set = {c.strip().upper() for c in exp_channels}
                _fea_set = {n.strip().upper() for n in fea_result.dof_names}
                _missing = _exp_set - _fea_set
                if _missing:
                    modal_status_banner(
                        title=f"FEA does not cover {len(_missing)} experimental channel(s)",
                        detail=(
                            f"Channels without a DOF in the FEA model: "
                            f"{', '.join(sorted(_missing)[:10])}"
                            f"{' …' if len(_missing) > 10 else ''}. "
                            "Review your FEA export — the DOFs must match "
                            "the measured channels. In the meantime, only "
                            "the frequency correlation is shown."
                        ),
                        severity="warning",
                    )
                    mac = None
                else:
                    mac = compute_fea_experimental_cross_mac(
                        fea_modes=fea_result.modes,
                        fea_dof_names=fea_result.dof_names,
                        exp_mode_shapes=exp_shapes,
                        exp_dof_names=exp_channels,
                    )

                if mac is not None:
                    # Heatmap
                    fea_labels = [f"FEA M{m.mode_number} ({m.freq_hz:.1f} Hz)"
                                   for m in fea_result.modes]
                    fig_mac = build_cross_mac_heatmap(
                        mac=mac,
                        fea_labels=fea_labels,
                        exp_labels=exp_mode_labels,
                        title="Cross-MAC FEA ↔ Experimental (OMA)",
                    )
                    st.plotly_chart(fig_mac, use_container_width=True)

                    # Pareo
                    pairs = pair_modes(
                        mac_matrix=mac,
                        fea_freqs=[m.freq_hz for m in fea_result.modes],
                        exp_freqs=exp_freqs,
                        mac_threshold=float(mac_thr),
                        freq_tolerance_pct=float(freq_tol),
                    )
                    st.markdown("**FEA ↔ experimental mode pairing**")
                    import pandas as pd
                    status_label = {
                        "valid": "✓ Valid",
                        "shape_only": "≈ Shape OK · freq out",
                        "freq_only": "≈ Freq OK · weak shape",
                        "weak": "✗ Weak",
                        "no_match": "✗ No match",
                    }
                    df_pairs = pd.DataFrame([
                        {
                            "FEA": f"M{p['fea_mode']} ({p['fea_freq']:.2f} Hz)",
                            "Exp": (f"M{p['exp_mode']} ({p['exp_freq']:.2f} Hz)"
                                     if p["exp_mode"] else "—"),
                            "MAC": f"{p['mac']:.3f}",
                            "Δf (%)": (f"{p['delta_freq_pct']:.1f}"
                                        if p["delta_freq_pct"] is not None else "—"),
                            "Status": status_label.get(p["status"], p["status"]),
                        }
                        for p in pairs
                    ])
                    st.dataframe(df_pairs, hide_index=True,
                                  use_container_width=True)

                    # Banner de diagnóstico global
                    n_valid = sum(1 for p in pairs if p["status"] == "valid")
                    n_total = len(pairs)
                    if n_total == 0:
                        pass
                    elif n_valid == n_total:
                        modal_status_banner(
                            title=f"FEA model validated · {n_valid}/{n_total} modes "
                                    "with MAC ≥ threshold and Δf ≤ tolerance",
                            detail=(
                                "All FEA modes have an experimental counterpart "
                                "with valid correlation. The model is considered fit "
                                "for rotor dynamics prediction under API 684 sec. 1.6."
                            ),
                            severity="ok",
                        )
                    elif n_valid >= n_total * 0.7:
                        modal_status_banner(
                            title=f"FEA model acceptable · {n_valid}/{n_total} valid modes",
                            detail=(
                                "Most modes correlate, but there are individual "
                                "modes with a weak shape or a frequency out of "
                                "tolerance. Review the model's local "
                                "masses/stiffnesses for the pairs marked 'shape_only' or "
                                "'freq_only'."
                            ),
                            severity="warning",
                        )
                    else:
                        modal_status_banner(
                            title=f"FEA model requires iteration · only {n_valid}/{n_total} valid",
                            detail=(
                                "More than 30% of FEA modes do not correlate. "
                                "Possible causes: poorly defined boundary conditions, "
                                "missing lumped masses, insufficient mesh in "
                                "critical zones, or incorrect material properties. "
                                "Re-iterate the model before using it for prediction."
                            ),
                            severity="fail",
                        )

            # ----- Caso EMA: solo frecuencias -----
            if exp_source == "ema_freq_only":
                modal_status_banner(
                    title="Comparison limited to frequencies (EMA without multi-channel mode shapes)",
                    detail=(
                        "The peaks from the current EMA Tab do not include "
                        "multi-channel mode shapes — only frequencies and damping. "
                        "For full Cross-MAC, run the FDD workflow in the OMA Tab or "
                        "use the upcoming EMA-LSCF sprint with mode shapes."
                    ),
                    severity="info",
                )
                # Tabla simple de match por frecuencia
                import pandas as pd
                rows = []
                used = set()
                for fm in fea_result.modes:
                    best_j, best_delta = -1, 9e9
                    for j, ef in enumerate(exp_freqs):
                        if j in used:
                            continue
                        d = abs(fm.freq_hz - ef) / max(ef, 1e-6) * 100.0
                        if d < best_delta:
                            best_delta = d; best_j = j
                    if best_j < 0:
                        rows.append({"FEA": f"M{fm.mode_number} ({fm.freq_hz:.2f} Hz)",
                                       "Exp": "—", "Δf (%)": "—", "Status": "✗ No match"})
                        continue
                    used.add(best_j)
                    ok = best_delta <= float(freq_tol)
                    rows.append({
                        "FEA": f"M{fm.mode_number} ({fm.freq_hz:.2f} Hz)",
                        "Exp": f"P{best_j+1} ({exp_freqs[best_j]:.2f} Hz)",
                        "Δf (%)": f"{best_delta:.1f}",
                        "Status": "✓ Freq OK" if ok else "✗ Freq out",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                              use_container_width=True)


# ---------------------------------------------------------------------
# Tab 7 — Reports (selector granular + auto-análisis + IA)
# ---------------------------------------------------------------------
if _active_modal_tab == "📊 Reports":
    modal_section_header(
        title="Reports — granular selector + analysis",
        subtitle=(
            "Choose which figures to send to the SIGA report. Automatic "
            "standards-based analysis + optional interpretive AI."
        ),
        norm_ref="ISO 7626-6 sec. 8 · Modal documentation",
        icon="📊",
    )

    # ---- Corridas OMA subidas desde el CAMPO (nube) → reporte SIGA con un clic ----
    with st.expander("☁ OMA runs from the field (cloud) → SIGA report", expanded=False):
        try:
            from core.modal import modal_cloud
            _runs = modal_cloud.list_runs()
        except Exception as _e:  # noqa: BLE001
            _runs = []; st.caption(f"Cloud unavailable: {_e}")
        if not _runs:
            st.info("No field runs found. Capture in Watermelon Modal (field) and press "
                    "**☁ Upload run to cloud**. Requires the `modal_runs` table in Supabase.")
        else:
            _opts = {f"{r.get('name','?')} · {r.get('updated_at','')}": r.get("id") for r in _runs}
            _pick = st.selectbox("Field run", list(_opts.keys()), key="modal_cloud_run_pick")
            if st.button("📄 Generate SIGA OMA report", key="modal_cloud_run_report"):
                try:
                    _payload = modal_cloud.load_run(_opts[_pick])
                    if not _payload:
                        st.error("Could not download the run.")
                    else:
                        from core.modal.run_report import build_report_from_run
                        _pdf = build_report_from_run(_payload)
                        st.success(f"Report generated · {len(_pdf)//1024} KB.")
                        st.download_button("⬇ Download report (PDF)", data=_pdf,
                                           file_name=f"OMA_{_payload.get('name','run')}.pdf",
                                           mime="application/pdf", key="modal_cloud_run_dl")
                except Exception as _e:  # noqa: BLE001
                    st.error(f"Report error: {type(_e).__name__}: {_e}")

    # Detectar si hay análisis previo disponible
    _fdd_for_rep = st.session_state.get("modal_oma_result")
    _peaks_for_rep = st.session_state.get("modal_peaks", [])
    _geom_for_rep = st.session_state.get("modal_geometry")
    _fea_for_rep = st.session_state.get("fea_result")

    _has_oma = bool(_fdd_for_rep and getattr(_fdd_for_rep, "modes", None))
    _has_ema = bool(_peaks_for_rep)
    _has_geom = bool(_geom_for_rep)
    _has_fea = bool(_fea_for_rep)

    if not (_has_oma or _has_ema):
        modal_empty_state(
            icon="📊",
            title="No modal analysis loaded",
            description=(
                "To use this section you need to have run at least one "
                "analysis: EMA in the EMA Tab or FDD in the OMA Tab. Then come "
                "back here and you can select which figures to send to the report."
            ),
            cta_label="Switch to the EMA Tab or OMA Tab",
            norm_ref="ISO 7626-6 sec. 8",
        )
    else:
        # =================================================================
        # SECCION A — Selector granular de figuras
        # =================================================================
        st.markdown("### 📋 A · Figure selector to inject")
        st.caption(
            "Check which figures from the modal analysis you want to send to "
            "the report. Each figure is rendered as a PNG and appended to the "
            "standard Reports system (it appears in the final SIGA PDF)."
        )

        # Sub-checkbox: por modo natural (1 row por modo) + globales
        _all_selections: Dict[str, bool] = {}

        if _has_oma:
            _natural_modes_rep = [
                m for m in _fdd_for_rep.modes
                if getattr(m, "classification", "natural") == "natural"
            ]
            _non_natural_count = len(_fdd_for_rep.modes) - len(_natural_modes_rep)

            with st.expander(
                f"🌊 OMA · {len(_natural_modes_rep)} natural modes "
                f"({_non_natural_count} non-natural)",
                expanded=True,
            ):
                # Bulk controls
                _bulk_c1, _bulk_c2, _bulk_c3 = st.columns([1, 1, 2])
                with _bulk_c1:
                    if st.button("✓ All", key="rep_sel_all_oma",
                                   use_container_width=True):
                        for m in _natural_modes_rep:
                            for sfx in ("3d", "bar", "polar"):
                                st.session_state[
                                    f"_rep_sel_m{m.mode_number}_{sfx}"
                                ] = True
                        st.session_state["_rep_sel_automac"] = True
                        st.session_state["_rep_sel_summary"] = True
                        st.rerun()
                with _bulk_c2:
                    if st.button("✗ None", key="rep_sel_none_oma",
                                   use_container_width=True):
                        for m in _natural_modes_rep:
                            for sfx in ("3d", "bar", "polar"):
                                st.session_state[
                                    f"_rep_sel_m{m.mode_number}_{sfx}"
                                ] = False
                        st.session_state["_rep_sel_automac"] = False
                        st.session_state["_rep_sel_summary"] = False
                        st.rerun()
                with _bulk_c3:
                    _include_non_nat_rep = st.toggle(
                        "Include harmonic/spurious",
                        value=False, key="_rep_include_non_nat",
                        help="Off by default — physical modes only.",
                    )

                if _include_non_nat_rep:
                    _natural_modes_rep = list(_fdd_for_rep.modes)

                # Headers de tabla
                st.markdown(
                    "<div style='display:grid; grid-template-columns: 2fr 1fr 1fr 1fr; "
                    "gap:8px; padding:6px 0; border-bottom:1px solid #e5e7eb; "
                    "font-size:11px; color:#64748b; text-transform:uppercase;'>"
                    "<div>Mode</div><div style='text-align:center;'>3D snapshot</div>"
                    "<div style='text-align:center;'>Bar chart</div>"
                    "<div style='text-align:center;'>Polar complexity</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                # Una row por modo
                for m in _natural_modes_rep:
                    _row_c1, _row_c2, _row_c3, _row_c4 = st.columns(
                        [2, 1, 1, 1]
                    )
                    with _row_c1:
                        _cls = getattr(m, "classification", "natural")
                        _cls_emoji = {"natural": "🟢", "harmonic": "🟡",
                                        "spurious": "🔴"}.get(_cls, "⚪")
                        st.markdown(
                            f"**M{m.mode_number}** · "
                            f"{m.natural_frequency_hz:.2f} Hz · "
                            f"ζ={m.damping_ratio_pct:.2f}% {_cls_emoji}"
                        )
                    with _row_c2:
                        _sel_3d = st.checkbox(
                            "", value=True,
                            key=f"_rep_sel_m{m.mode_number}_3d",
                            label_visibility="collapsed",
                        )
                    with _row_c3:
                        _sel_bar = st.checkbox(
                            "", value=True,
                            key=f"_rep_sel_m{m.mode_number}_bar",
                            label_visibility="collapsed",
                        )
                    with _row_c4:
                        _sel_pol = st.checkbox(
                            "", value=True,
                            key=f"_rep_sel_m{m.mode_number}_polar",
                            label_visibility="collapsed",
                        )
                    _all_selections[f"m{m.mode_number}_3d"] = _sel_3d
                    _all_selections[f"m{m.mode_number}_bar"] = _sel_bar
                    _all_selections[f"m{m.mode_number}_polar"] = _sel_pol

                st.divider()
                # Globals
                st.markdown("**Global blocks:**")
                _gc1, _gc2 = st.columns(2)
                with _gc1:
                    _sel_mac = st.checkbox(
                        "🔗 AutoMAC heatmap matrix",
                        value=True,
                        key="_rep_sel_automac",
                        help="MAC across all modes · ISO 7626-6 sec. 6.5",
                    )
                    _all_selections["automac"] = _sel_mac
                with _gc2:
                    _sel_sum = st.checkbox(
                        "📑 Modal summary table",
                        value=True,
                        key="_rep_sel_summary",
                        help="Table with freq/CPM/ζ/Q/MPC/class for all modes",
                    )
                    _all_selections["summary"] = _sel_sum

        # Setup — geometría 3D estática + tablas
        if _has_geom:
            with st.expander(
                f"🛠 Setup · 3D geometry ({len(_geom_for_rep.blocks)} blocks "
                f"+ {len(_geom_for_rep.sensors)} sensors)",
                expanded=False,
            ):
                _sel_setup_geom = st.checkbox(
                    "🌐 3D snapshot of the asset (undeformed)",
                    value=True, key="_rep_sel_setup_geom",
                )
                _sel_setup_blk = st.checkbox(
                    "📋 Mechanical blocks table",
                    value=True, key="_rep_sel_setup_blocks",
                )
                _sel_setup_sen = st.checkbox(
                    "📋 Instrumented sensors table",
                    value=True, key="_rep_sel_setup_sensors",
                )
                _all_selections["setup_geom"] = _sel_setup_geom
                _all_selections["setup_blocks"] = _sel_setup_blk
                _all_selections["setup_sensors"] = _sel_setup_sen

        # Adquisición — waveforms TDMS
        _tdms_for_rep = st.session_state.get("modal_tdms")
        if _tdms_for_rep and getattr(_tdms_for_rep, "channels", None):
            _n_ch = len(_tdms_for_rep.channels)
            with st.expander(
                f"📥 Acquisition · {_n_ch} TDMS waveforms",
                expanded=False,
            ):
                _sel_acq_all = st.checkbox(
                    f"📈 Waveforms time-series ({_n_ch} channels)",
                    value=False, key="_rep_sel_acq_waveforms",
                    help=("Each channel is rendered as a time-vs-amplitude plot "
                          "downsampled to max 5000 points."),
                )
                _all_selections["acq_waveforms"] = _sel_acq_all

        # EMA — FRF + peaks
        if _has_ema or st.session_state.get("modal_frfs"):
            _frfs_count = len(st.session_state.get("modal_frfs", []))
            with st.expander(
                f"🔨 EMA · {_frfs_count} FRFs + "
                f"{len(_peaks_for_rep)} identified peaks",
                expanded=False,
            ):
                _sel_ema_frf = st.checkbox(
                    "📊 FRF Bode with peaks marked",
                    value=bool(_peaks_for_rep),
                    key="_rep_sel_ema_frf",
                    help="Magnitude (dB) vs frequency with mode peaks.",
                )
                _sel_ema_tbl = st.checkbox(
                    "📋 EMA peaks table",
                    value=bool(_peaks_for_rep),
                    key="_rep_sel_ema_table",
                    help="Freq, damping, bandwidth, Q per peak.",
                )
                _all_selections["ema_frf"] = _sel_ema_frf
                _all_selections["ema_table"] = _sel_ema_tbl

        # OMA — SVD plot del FDD
        if _has_oma:
            with st.expander(
                f"🌊 OMA · FDD SVD plot",
                expanded=False,
            ):
                _sel_oma_svd = st.checkbox(
                    "📈 First Singular Value plot (FDD)",
                    value=True, key="_rep_sel_oma_svd",
                    help=("Plot of the 1st SV of the cross-spectrum with the "
                          "identified modes marked (green natural, yellow "
                          "harmonic, red spurious)."),
                )
                _all_selections["oma_svd"] = _sel_oma_svd

        # FEA Compare — Cross-MAC + pareo
        if _has_fea:
            with st.expander(
                f"🧮 FEA Compare · {_fea_for_rep.n_modes} FEA modes",
                expanded=False,
            ):
                _sel_fea_mac = st.checkbox(
                    "🔥 Cross-MAC heatmap (FEA ↔ Experimental)",
                    value=True, key="_rep_sel_fea_mac",
                )
                _sel_fea_pair = st.checkbox(
                    "📋 Mode pairing table",
                    value=True, key="_rep_sel_fea_pairing",
                )
                _all_selections["fea_mac"] = _sel_fea_mac
                _all_selections["fea_pairing"] = _sel_fea_pair

        # ----- Resumen + Acción -----
        st.divider()
        _n_selected = sum(1 for v in _all_selections.values() if v)
        _ac1, _ac2 = st.columns([3, 2])
        with _ac1:
            st.metric("Selected figures", _n_selected)
        with _ac2:
            _send_disabled = (_n_selected == 0)
            if st.button(
                f"📄 Inject {_n_selected} figures into report",
                key="rep_send_selected",
                type="primary",
                use_container_width=True,
                disabled=_send_disabled,
            ):
                from core.modal.modal_report import (
                    build_modal_report_items,
                    append_modal_items_to_report,
                )
                _prog = st.progress(0.0, text="Generating snapshots…")

                def _cb_sel(idx, total, stage):
                    _prog.progress(
                        min(idx / max(total, 1), 1.0),
                        text=f"{stage} ({idx + 1}/{total})",
                    )

                try:
                    _asset_lbl_r = (
                        st.session_state.get("modal_adhoc_meta",
                                              {}).get("equipment_name",
                                                       "Ad-hoc asset")
                        if st.session_state.get("modal_adhoc_meta")
                        else st.session_state.get("modal_inst", "Asset")
                    )
                    _asset_lbl_r = str(_asset_lbl_r)
                    _aggregated: List[Dict[str, Any]] = []

                    # --- Modal (mode shapes + AutoMAC + summary) ---
                    if _has_oma:
                        _all_items = build_modal_report_items(
                            fdd_result=_fdd_for_rep,
                            geom=_geom_for_rep,
                            include_non_natural=_include_non_nat_rep,
                            asset_name=_asset_lbl_r,
                            method="OMA",
                            progress_cb=_cb_sel,
                        )
                        for it in _all_items:
                            _t = it.get("type", "")
                            _kept = False
                            if _t == "modal_3d":
                                for m in _natural_modes_rep:
                                    if (f"Modo {m.mode_number}" in it["title"]
                                        and _all_selections.get(
                                            f"m{m.mode_number}_3d")):
                                        _kept = True; break
                            elif _t == "modal_bar":
                                for m in _natural_modes_rep:
                                    if (f"Modo {m.mode_number}" in it["title"]
                                        and _all_selections.get(
                                            f"m{m.mode_number}_bar")):
                                        _kept = True; break
                            elif _t == "modal_polar":
                                for m in _natural_modes_rep:
                                    if (f"Modo {m.mode_number}" in it["title"]
                                        and _all_selections.get(
                                            f"m{m.mode_number}_polar")):
                                        _kept = True; break
                            elif _t == "modal_automac":
                                _kept = bool(_all_selections.get("automac"))
                            elif _t == "modal_summary_table":
                                _kept = bool(_all_selections.get("summary"))
                            if _kept:
                                _aggregated.append(it)

                    # --- Setup ---
                    if (_all_selections.get("setup_geom")
                        or _all_selections.get("setup_blocks")
                        or _all_selections.get("setup_sensors")):
                        from core.modal.modal_report import build_setup_items
                        _setup_items = build_setup_items(
                            _geom_for_rep, _asset_lbl_r
                        )
                        for it in _setup_items:
                            _t = it.get("type", "")
                            if (_t == "modal_setup_geometry"
                                and _all_selections.get("setup_geom")):
                                _aggregated.append(it)
                            elif (_t == "modal_setup_blocks"
                                  and _all_selections.get("setup_blocks")):
                                _aggregated.append(it)
                            elif (_t == "modal_setup_sensors"
                                  and _all_selections.get("setup_sensors")):
                                _aggregated.append(it)

                    # --- Adquisición ---
                    if _all_selections.get("acq_waveforms"):
                        from core.modal.modal_report import (
                            build_acquisition_items)
                        _tdms_for_rep_x = st.session_state.get("modal_tdms")
                        _acq_items = build_acquisition_items(
                            _tdms_for_rep_x, _asset_lbl_r,
                        )
                        _aggregated.extend(_acq_items)

                    # --- EMA ---
                    if (_all_selections.get("ema_frf")
                        or _all_selections.get("ema_table")):
                        from core.modal.modal_report import build_ema_items
                        _ema_items = build_ema_items(
                            st.session_state.get("modal_frfs", []),
                            _peaks_for_rep,
                            _asset_lbl_r,
                        )
                        for it in _ema_items:
                            _t = it.get("type", "")
                            if (_t == "modal_ema_frf"
                                and _all_selections.get("ema_frf")):
                                _aggregated.append(it)
                            elif (_t == "modal_ema_peaks_table"
                                  and _all_selections.get("ema_table")):
                                _aggregated.append(it)

                    # --- OMA SVD ---
                    if _all_selections.get("oma_svd"):
                        from core.modal.modal_report import build_oma_items
                        _oma_items = build_oma_items(
                            _fdd_for_rep, _asset_lbl_r,
                        )
                        _aggregated.extend(_oma_items)

                    # --- FEA Compare ---
                    if (_all_selections.get("fea_mac")
                        or _all_selections.get("fea_pairing")):
                        from core.modal.modal_report import build_fea_items
                        _fea_items = build_fea_items(
                            _fea_for_rep, _fdd_for_rep, _asset_lbl_r,
                        )
                        for it in _fea_items:
                            _t = it.get("type", "")
                            if (_t == "modal_fea_cross_mac"
                                and _all_selections.get("fea_mac")):
                                _aggregated.append(it)
                            elif (_t == "modal_fea_pairing"
                                  and _all_selections.get("fea_pairing")):
                                _aggregated.append(it)

                    _n_added = append_modal_items_to_report(_aggregated)
                    _prog.empty()
                    st.success(
                        f"✓ {_n_added} figures added to the report. "
                        "Go to **Reports** (sidebar) to review and "
                        "generate the PDF."
                    )
                except Exception as _exc:  # noqa: BLE001
                    _prog.empty()
                    import traceback as _tb
                    st.error(
                        f"Error: `{type(_exc).__name__}: {_exc}`"
                    )
                    with st.expander("Technical detail"):
                        st.code(_tb.format_exc(), language="text")

        # =================================================================
        # SECCION B — Auto-análisis normativo (rule-based)
        # =================================================================
        st.divider()
        st.markdown("### 🧠 B · Standards-based auto-analysis")
        st.caption(
            "Rule-based analysis (ISO 7626 + API 684 + API 618). "
            "No AI — deterministic text generated from the modal data."
        )

        if not _has_oma:
            modal_status_banner(
                title="Auto-analysis requires OMA modes",
                detail="Run the FDD in the OMA Tab to enable this section.",
                severity="info",
            )
        else:
            from core.modal.auto_analysis import (
                analyze_modal_results,
                build_analysis_report_item,
            )

            # Parámetros configurables
            _ab_c1, _ab_c2, _ab_c3 = st.columns(3)
            with _ab_c1:
                _ab_running = st.number_input(
                    "Running speed (rpm)",
                    value=int(st.session_state.get(
                        "_modal_auto_running_rpm", 3600)),
                    min_value=100, max_value=20000, step=100,
                    key="_modal_auto_running_rpm",
                    help="To evaluate crossings with harmonics",
                )
            with _ab_c2:
                _ab_mac_thr = st.number_input(
                    "MAC redundancy threshold",
                    value=0.70, min_value=0.5, max_value=0.95, step=0.05,
                    key="_modal_auto_mac_thr",
                )
            with _ab_c3:
                _ab_mpc_thr = st.number_input(
                    "High MPC threshold (%)",
                    value=30.0, min_value=10.0, max_value=70.0, step=5.0,
                    key="_modal_auto_mpc_thr",
                )

            # Ejecutar análisis
            _findings = analyze_modal_results(
                fdd_result=_fdd_for_rep,
                running_rpm=float(_ab_running),
                mac_threshold=float(_ab_mac_thr),
                mpc_complex_threshold_pct=float(_ab_mpc_thr),
                method="OMA",
            )

            # Resumen estadístico
            _n_fail = sum(1 for f in _findings if f.severity == "fail")
            _n_warn = sum(1 for f in _findings if f.severity == "warning")
            _n_ok = sum(1 for f in _findings if f.severity == "ok")
            _n_info = sum(1 for f in _findings if f.severity == "info")

            _kc1, _kc2, _kc3, _kc4 = st.columns(4)
            _kc1.metric("Compliant ✓", _n_ok)
            _kc2.metric("Warnings ⚠", _n_warn)
            _kc3.metric("Critical ✗", _n_fail)
            _kc4.metric("Informational ℹ", _n_info)

            # Render findings
            for _f in _findings:
                modal_status_banner(
                    title=_f.title,
                    detail=(_f.text + (f" · Standard: {_f.norm_ref}"
                                          if _f.norm_ref else "")),
                    severity=_f.severity,
                )

            # Botón inyectar al reporte
            st.divider()
            if st.button(
                f"📄 Inject auto-analysis ({len(_findings)} findings) "
                "into report",
                key="rep_send_auto_analysis",
                type="primary",
                use_container_width=True,
            ):
                from core.modal.modal_report import append_modal_items_to_report
                try:
                    _asset_lbl_b = (
                        st.session_state.get("modal_adhoc_meta",
                                              {}).get("equipment_name",
                                                       "Ad-hoc asset")
                        if st.session_state.get("modal_adhoc_meta")
                        else st.session_state.get("modal_inst", "Asset")
                    )
                    _auto_item = build_analysis_report_item(
                        findings=_findings,
                        asset_name=str(_asset_lbl_b),
                        method="OMA",
                    )
                    _n = append_modal_items_to_report([_auto_item])
                    st.success(
                        f"✓ Auto-analysis ({len(_findings)} findings) "
                        "added to the report as 1 PNG figure. "
                        "Go to Reports (sidebar) to review it."
                    )
                except Exception as _exc:  # noqa: BLE001
                    import traceback as _tb
                    st.error(
                        f"Error: `{type(_exc).__name__}: {_exc}`"
                    )
                    with st.expander("Technical detail"):
                        st.code(_tb.format_exc(), language="text")

        # =================================================================
        # SECCION C — Análisis IA interpretativo (paid, via Anthropic)
        # =================================================================
        st.divider()
        st.markdown("### 🤖 C · Interpretive AI analysis")
        st.caption(
            "Interpretive narrative generated by Claude with full modal "
            "context. Same AI quota used by Spectrum/SCL/Polar/Waveform. "
            "Local 30-day caching to avoid repeated costs."
        )

        from core.ai_diagnostic import (
            generate_ai_diagnostic,
            is_ai_available,
        )

        if not _has_oma:
            modal_status_banner(
                title="AI requires OMA modes",
                detail="Run the FDD in the OMA Tab before generating the AI analysis.",
                severity="info",
            )
        elif not is_ai_available():
            modal_status_banner(
                title="AI not configured",
                detail=(
                    "The AI module requires `[anthropic] api_key` in "
                    "Streamlit secrets + the `anthropic` package installed. "
                    "Contact the admin to enable it."
                ),
                severity="warning",
            )
        else:
            _ac_c1, _ac_c2 = st.columns([2, 1])
            with _ac_c1:
                _ai_operator_notes = st.text_area(
                    "Operator notes (optional context)",
                    value="", height=100,
                    key="_modal_ai_operator_notes",
                    help=("Context the AI should consider: operating "
                          "conditions, recent events, suspicions, etc. "
                          "Improves analysis quality."),
                )
            with _ac_c2:
                _ai_use_cache = st.toggle(
                    "Use cache if available",
                    value=True, key="_modal_ai_use_cache",
                    help="30-day TTL · avoids repeated costs for the same data.",
                )
                _ai_running = st.number_input(
                    "Running rpm",
                    value=int(st.session_state.get(
                        "_modal_auto_running_rpm", 3600)),
                    min_value=100, max_value=20000, step=100,
                    key="_modal_ai_running_rpm",
                )

            if st.button(
                "🤖 Generate modal AI analysis",
                key="rep_gen_ai",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Calling Claude · may take 15-30s…"):
                    from core.modal.modal_report import (
                        build_modal_ai_payload,
                    )
                    _asset_lbl_c = (
                        st.session_state.get("modal_adhoc_meta",
                                              {}).get("equipment_name",
                                                       "Ad-hoc asset")
                        if st.session_state.get("modal_adhoc_meta")
                        else st.session_state.get("modal_inst", "Asset")
                    )
                    _payload = build_modal_ai_payload(
                        fdd_result=_fdd_for_rep,
                        asset_name=str(_asset_lbl_c),
                        method="OMA",
                        running_rpm=float(_ai_running),
                        operator_notes=_ai_operator_notes or "",
                    )
                    _ai_result = generate_ai_diagnostic(
                        payload=_payload,
                        module_type="modal",
                        use_cache=_ai_use_cache,
                    )
                    st.session_state["_modal_ai_result"] = _ai_result

            # Mostrar resultado IA si existe
            _ai_res = st.session_state.get("_modal_ai_result")
            if _ai_res:
                if not _ai_res.get("ok"):
                    modal_status_banner(
                        title=f"AI error: {_ai_res.get('error', 'unknown')}",
                        detail=_ai_res.get("markdown", ""),
                        severity="fail",
                    )
                else:
                    _ai_kc1, _ai_kc2, _ai_kc3 = st.columns(3)
                    _ai_kc1.metric(
                        "Model", _ai_res.get("model", "n/a"),
                    )
                    _ai_kc2.metric(
                        "Tokens in/out",
                        (f"{_ai_res.get('input_tokens', 0):,} / "
                          f"{_ai_res.get('output_tokens', 0):,}"),
                    )
                    _ai_kc3.metric(
                        "Cache hit",
                        "✓ Yes" if _ai_res.get("cached") else "✗ No (new)",
                    )
                    st.divider()
                    st.markdown(_ai_res.get("markdown", ""))
                    st.divider()
                    if st.button(
                        "📄 Inject AI analysis into report",
                        key="rep_send_ai",
                        type="primary",
                        use_container_width=True,
                    ):
                        from core.modal.modal_report import (
                            build_ai_diagnostic_report_item,
                            append_modal_items_to_report,
                        )
                        try:
                            _ai_item = build_ai_diagnostic_report_item(
                                ai_result=_ai_res,
                                asset_name=str(_asset_lbl_c)
                                            if "_asset_lbl_c" in dir()
                                            else "Asset",
                                method="OMA",
                            )
                            _n_ai = append_modal_items_to_report([_ai_item])
                            st.success(
                                f"✓ AI analysis added to the report as "
                                "1 PNG figure (navy header + text in notes)."
                            )
                        except Exception as _exc:  # noqa: BLE001
                            import traceback as _tb
                            st.error(
                                f"Error: `{type(_exc).__name__}: {_exc}`"
                            )
                            with st.expander("Technical detail"):
                                st.code(_tb.format_exc(), language="text")


# =====================================================================
# Footer normativo permanente
# =====================================================================
modal_footer_norms(
    active_norms=[
        "ISO 7626-1..6",
        "ISO 20816",
        "API 684",
        "API 618 secc. 7.9.4.2.5.3.2",
    ],
    algorithms=[
        "Circle-Fit Nyquist (Kennedy-Pancu 1947)",
        "FDD (Brincker, Zhang, Andersen 2001)",
        "Modal Complexity MPC (Pappa & Eishan 1995)",
        "AutoMAC (ISO 7626-6 secc. 6.5)",
        "Half-power method (ISO 7626-6 secc. 6.3.2)",
        "Diagrama de Campbell (API 684 secc. 1.6)",
    ],
    # version=None → lee VERSION dinámicamente vía core.version
)
