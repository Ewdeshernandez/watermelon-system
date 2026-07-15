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
    st.error("Tu rol no tiene acceso a este módulo.")
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
_asset_name = "(sin activo seleccionado)"

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
    "Navegación del módulo modal", _MODAL_TABS,
    horizontal=True, key="modal_active_tab", label_visibility="collapsed",
)
st.divider()


# ---------------------------------------------------------------------
# Tab 1 — Setup
# ---------------------------------------------------------------------
if _active_modal_tab == "🛠 Setup":
    modal_section_header(
        title="Configuración del ensayo modal",
        subtitle="Selecciona o registra el activo bajo análisis",
        norm_ref="ISO 7626-6 secc. 6",
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
        "Origen del activo",
        [
            "📦 Activo registrado en Machinery Library",
            "🎯 Análisis ad-hoc — equipo externo / one-off",
        ],
        horizontal=False,
        key="modal_setup_mode",
        help=(
            "Activo registrado: usa la configuración de sensores ya definida "
            "(Sensor Map). Ad-hoc: ingresa metadata manualmente, útil para "
            "clientes que solo contratan análisis modal puntual sin monitoreo "
            "continuo."
        ),
    )
    st.divider()

    # ─── MODO AD-HOC ──────────────────────────────────────────────────
    if setup_mode.startswith("🎯"):
        st.warning(
            "⏱️ **Modo ad-hoc = TEMPORAL.** Estos datos viven solo en esta "
            "sesión y **NO registran un activo permanente** (no aparecen en la "
            "lista de 'Activo bajo análisis'). Sirve para un análisis puntual de "
            "un equipo externo.\n\n"
            "➡️ **Para crear un activo PERMANENTE** que quede guardado y puedas "
            "reutilizar en futuros análisis, usa **'✦ Crear activo (wizard)'** "
            "en el menú lateral (Machinery Library). Ahí sí se registra."
        )
        st.markdown("**Datos del activo (análisis puntual · temporal)**")
        st.caption(
            "Completa la metadata mínima del equipo bajo análisis. "
            "Estos datos aparecerán en el Hero del módulo y en el reporte final "
            "de ESTA sesión (no se guardan como activo)."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            adhoc_tag = st.text_input(
                "Nombre / Tag del equipo *",
                value=st.session_state.get("modal_adhoc_tag", ""),
                placeholder="e.g. Motor GE LM6000 — Pad 2",
                key="modal_adhoc_tag",
            )
            adhoc_client = st.text_input(
                "Cliente",
                value=st.session_state.get("modal_adhoc_client", ""),
                placeholder="e.g. MAGNEX, PAREX, Ecopetrol",
                key="modal_adhoc_client",
            )
        with col_b:
            adhoc_station = st.text_input(
                "Estación / Ubicación",
                value=st.session_state.get("modal_adhoc_station", ""),
                placeholder="e.g. La Belleza, Isla 6, Termosuria",
                key="modal_adhoc_station",
            )
            adhoc_model = st.text_input(
                "Modelo / Tipo",
                value=st.session_state.get("modal_adhoc_model", ""),
                placeholder="e.g. Turbogenerador aeroderivado 45 MW",
                key="modal_adhoc_model",
            )

        adhoc_notes = st.text_area(
            "Notas técnicas (opcional)",
            value=st.session_state.get("modal_adhoc_notes", ""),
            placeholder="Condiciones de operación, motivo del análisis, "
                         "observaciones del cliente...",
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
                title=f"Análisis ad-hoc configurado · {adhoc_tag}",
                detail=(
                    f"Activo registrado para esta sesión de análisis modal. "
                    f"{('Cliente: ' + adhoc_client + ' · ') if adhoc_client else ''}"
                    f"{('Ubicación: ' + adhoc_station + ' · ') if adhoc_station else ''}"
                    f"{('Modelo: ' + adhoc_model) if adhoc_model else ''}"
                ),
                severity="ok",
            )

            modal_status_banner(
                title="Análisis ad-hoc — limitaciones de configuración",
                detail=(
                    "Sin Sensor Map predefinido, el módulo Modal usa la "
                    "configuración de canales del archivo de captura "
                    "(.tdms) cargado en Adquisición. Las funcionalidades "
                    "**Bar chart 2D**, **Complexity Polar**, **AutoMAC**, "
                    "**Diagrama de Campbell** y todo el análisis EMA/OMA están "
                    "disponibles. La visualización **Mode Shapes 3D con flechas "
                    "sobre el activo** requiere position_3d configurado en "
                    "Machinery Library — no disponible en modo ad-hoc."
                ),
                severity="info",
            )
        else:
            st.info(
                "👆 Ingresa al menos el **Nombre / Tag del equipo** para "
                "habilitar el análisis modal en modo ad-hoc."
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
            title="No hay activos registrados",
            description=(
                "El módulo Modal Analysis se ejecuta sobre activos definidos en "
                "Machinery Library. Crea un activo primero usando el wizard "
                "'Crear activo' en la barra lateral, configura los sensores y "
                "vuelve aquí. O cambia al modo "
                "**Análisis ad-hoc** arriba para registrar un equipo puntual sin "
                "necesidad de crearlo en Machinery Library."
            ),
            cta_label="O usa modo Ad-hoc arriba ↑",
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
                "Activo bajo análisis",
                _options,
                format_func=lambda x: _labels_by_id.get(x, x) if x != "(seleccionar)" else x,
                key="modal_inst",
            )
            # Persistir la selección para restaurarla tras recarga (v3.31.438).
            if picked_id and picked_id != "(seleccionar)":
                _save_sel(setup_mode=setup_mode, asset_id=picked_id, adhoc={})

        with col_meta:
            st.caption(
                f"📦 **{len(_all_insts)} activos** disponibles en Machinery Library"
            )

        # Si selecciona un activo, mostrar preview con sensores y geometría
        if picked_id and picked_id != "(seleccionar)":
            inst = get_instance(picked_id)
            if inst is None:
                modal_status_banner(
                    title="Activo no encontrado",
                    detail=f"No se pudo cargar la información de '{picked_id}'.",
                    severity="fail",
                )
            else:
                # Hero secundario del activo seleccionado
                _location = getattr(inst, "location", "") or "(sin ubicación)"
                _profile = getattr(inst, "profile_key", "") or "(genérico)"
                _serial = getattr(inst, "serial_number", "") or "—"

                st.markdown(
                    f"""
                    <div style="background:#F4F7FB; border-left:4px solid #1AAEE5;
                                 padding:14px 18px; border-radius:6px;
                                 margin-top:12px;">
                        <div style="font-size:11px; font-weight:700; color:#0F7FB0;
                                     letter-spacing:0.12em; text-transform:uppercase;">
                            Activo seleccionado
                        </div>
                        <div style="font-size:18px; font-weight:800; color:#0F1E3D;
                                     margin:4px 0;">
                            {inst.tag or inst.instance_id}
                        </div>
                        <div style="font-size:12px; color:#6B7280;">
                            📍 {_location} &nbsp;·&nbsp;
                            🏷 Modelo: {_profile} &nbsp;·&nbsp;
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
                        title="Activo sin sensor map en Machinery Library",
                        detail=(
                            "Este activo no tiene un sensor map registrado en "
                            "Machinery Library (ese mapa se usa para Live "
                            "Monitoring). **Para el análisis modal NO es "
                            "obligatorio:** define tus sensores y su DOF abajo, "
                            "en «Geometría 3D del activo», y guárdalos como "
                            "«📦 Configuración COMPLETA del análisis» para "
                            "reutilizarlos. El ensayo modal usa esos sensores y "
                            "los canales del archivo capturado."
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
                        title="Sensores configurados",
                        subtitle="Distribución por tipo y readiness para análisis modal 3D",
                        norm_ref="ISO 7626-6 secc. 6.2",
                        icon="📡",
                    )
                    modal_kpi_row([
                        (str(len(sensors)), "Total sensores",
                         "registrados en Sensor Map", "navy"),
                        (str(n_accel), "Acelerómetros",
                         "Acelerómetro IEPE 100 mV/g típico", "cyan"),
                        (str(n_prox), "Proximidad",
                         "Bently 200 mV/mil típico", "amber"),
                        (str(n_3d_ready), "Modal 3D ready",
                         "con position_3d + DOF", "green"),
                    ])

                    # Status del activo para modal
                    if n_3d_ready == len(sensors):
                        modal_status_banner(
                            title="Activo completamente configurado para análisis modal 3D",
                            detail=(
                                f"Todos los {len(sensors)} sensores tienen position_3d + "
                                "dof_direction definidos. Mode shapes 3D animados disponibles."
                            ),
                            severity="ok",
                        )
                    elif n_3d_ready > 0:
                        modal_status_banner(
                            title=f"Configuración parcial — {n_3d_ready}/{len(sensors)} sensores 3D-ready",
                            detail=(
                                f"{len(sensors) - n_3d_ready} sensores carecen de position_3d "
                                "o dof_direction. Bar chart 2D y MAC disponibles para todos, "
                                "Flechas 3D solo para los sensores configurados."
                            ),
                            severity="warning",
                        )
                    else:
                        modal_status_banner(
                            title="Sin sensores con configuración modal 3D",
                            detail=(
                                "Para activar visualización 3D de mode shapes, completa "
                                "el expander 'Configuración modal' de cada sensor en el "
                                "wizard de Machinery Library. Mientras tanto, Bar chart 2D "
                                "y AutoMAC siguen disponibles."
                            ),
                            severity="info",
                        )

                    # Tabla de sensores con tag de configuración
                    with st.expander(f"▸ Lista de sensores ({len(sensors)})",
                                       expanded=False):
                        import pandas as pd
                        _rows = []
                        for s in sensors:
                            _rows.append({
                                "Plane": s.get("plane_label", "—"),
                                "Tipo": s.get("sensor_type", "—"),
                                "Dirección": s.get("direction", "—"),
                                "Unidad": s.get("unit_native", "—"),
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
                "👆 Selecciona un activo arriba para ver su configuración de sensores "
                "y validar readiness para análisis modal."
            )

    # =================================================================
    # Sub-sección: Editor de Geometría 3D — visualización profesional
    # =================================================================
    st.divider()
    modal_section_header(
        title="Geometría 3D del activo",
        subtitle=(
            "Dibuja el tren mecánico y posiciona los sensores con su dirección "
            "DOF. La geometría se usa como soporte visual para las mode shapes "
            "en Tab 5. Persistencia por activo o session-only en modo ad-hoc."
        ),
        norm_ref="ISO 7626-6 secc. 6 · DOF y orientación espacial documentadas",
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

    # Cargar geometría existente o inicializar.
    # Prioridad: geometría guardada del activo → autosave de la última sesión
    # → vacío. El autosave evita perder la config al recargar la página
    # (session_state se limpia, pero el autosave persiste en disco).
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
        st.caption("↩️ Se restauró la última configuración en la que estabas "
                   "trabajando. Guárdala con nombre abajo si quieres reusarla.")

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
        f"**Progreso de configuración:**  "
        f"{_chk(_step_asset)} 1· Activo  →  "
        f"{_chk(_step_blocks)} 2· Geometría ({len(geom.blocks)} bloques)  →  "
        f"{_chk(_step_sensors)} 3· Sensores ({len(geom.sensors)})  →  "
        f"{_chk(_step_ready)} 4· Listo para capturar")
    if not _step_blocks:
        st.caption("➡️ **Siguiente paso:** agrega al menos un bloque (la forma "
                   "del equipo) en «➕ Agregar / editar bloque». El modelo 3D "
                   "de arriba se actualiza al aplicar cada cambio.")
    elif not _step_sensors:
        st.caption("➡️ **Siguiente paso:** agrega tus sensores con su posición "
                   "(x, y, z) y dirección DOF en «➕ Agregar / editar sensor».")
    else:
        st.caption("✅ **Geometría lista.** Revisa el modelo 3D de arriba "
                   "(sensores en verde, flechas DOF en naranja). Guárdala con "
                   "nombre abajo para reutilizarla en futuras campañas.")

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
            "Cargar template (se aplica automáticamente al seleccionar)",
            options=["(mantener actual)", "empty",
                      "motor_compressor", "turbine_generator", "pump_motor"],
            format_func=lambda k: {
                "(mantener actual)": "— Mantener configuración actual —",
                "empty": "Vacío — construir desde cero (sin eje/acoples)",
                "motor_compressor": "Motor + Compresor (6 sensores)",
                "turbine_generator": "Turbina + Generador · LM6000+Brush (6 sensores)",
                "pump_motor": "Bomba + Motor (4 sensores)",
            }.get(k, k),
            key="geom_tpl_choice",
            on_change=_apply_template_on_change,
        )
    with col_t2:
        if st.button("💾 Guardar", key="geom_save",
                       use_container_width=True,
                       disabled=not _geom_asset_id,
                       help=("Persiste a data/modal/geometries/<asset_id>.json. "
                             "Disponible solo con activo registrado."
                             if not _geom_asset_id else
                             "Persiste a disco para este activo")):
            geom.asset_id = _geom_asset_id
            try:
                _p = save_geometry(geom)
                st.toast(f"Geometría guardada en {_p.name}", icon="✅")
            except Exception as exc:  # noqa: BLE001
                st.toast(f"Error al guardar: {exc}", icon="⚠")
    with col_t3:
        if st.button("⬇ Export JSON", key="geom_export",
                       use_container_width=True,
                       help="Descarga la geometría como JSON para reutilizarla "
                            "o compartirla. Funciona en cualquier modo."):
            st.session_state["_geom_export_ready"] = geom.to_json()

    # Feedback de aplicación del template
    if st.session_state.pop("_geom_just_applied", None):
        _label_map = {
            "motor_compressor": "Motor + Compresor",
            "turbine_generator": "Turbina + Generador (LM6000 + Brush)",
            "pump_motor": "Bomba + Motor",
        }
        _applied = _label_map.get(
            st.session_state.get("geom_tpl_choice", ""), "Template"
        )
        st.success(
            f"✓ Template **{_applied}** aplicado · "
            f"{len(geom.blocks)} bloques · {len(geom.sensors)} sensores. "
            "Edita los nombres y posiciones abajo si necesitas ajustarlo a "
            "tu activo real."
        )

    if st.session_state.get("_geom_export_ready"):
        st.download_button(
            "Descargar geometry.json",
            data=st.session_state["_geom_export_ready"],
            file_name=f"{geom.asset_id or 'adhoc'}_geometry.json",
            mime="application/json",
            key="geom_download_btn",
        )

    if not _geom_asset_id:
        st.caption(
            "Modo ad-hoc — la geometría vive solo en esta sesión. "
            "Para persistir entre sesiones, guárdala con un nombre abajo, "
            "selecciona un activo registrado en Tab Setup, o usa "
            "**⬇ Export JSON** para guardarla externamente."
        )

    # --- Guardar / cargar CONFIGURACIONES PERSONALIZADAS (también en ad-hoc) ---
    # Antes solo se podían guardar geometrías de activos registrados; las que
    # creaba el usuario se perdían. Ahora se pueden guardar con nombre y reusar.
    with st.expander("💾 Guardar / cargar configuración personalizada",
                     expanded=False):
        from core.modal.geometry_3d import (
            list_geometries as _list_geoms, delete_geometry as _del_geom,
        )
        _saved_all = _list_geoms()
        _custom_saved = [s for s in _saved_all if s.startswith("custom_")]
        _cc1, _cc2 = st.columns(2)
        with _cc1:
            st.markdown("**Guardar la geometría actual**")
            _save_name = st.text_input(
                "Nombre de la configuración", key="geom_custom_name",
                placeholder="ej. Skid compresor Parex")
            if st.button("Guardar configuración", key="geom_custom_save",
                         disabled=not _save_name.strip(),
                         use_container_width=True):
                _slug = "".join(c if c.isalnum() or c in "-_" else "_"
                                for c in _save_name.strip().lower())
                geom.asset_id = "custom_" + _slug
                geom.name = _save_name.strip() or geom.name
                try:
                    save_geometry(geom)
                    st.success(
                        f"✓ Guardada como **{_save_name}**. Ya puedes "
                        f"recargarla en cualquier momento aquí a la derecha.")
                except Exception as _e:  # noqa: BLE001
                    st.error(f"No se pudo guardar: {_e}")
        with _cc2:
            st.markdown("**Cargar / borrar una guardada**")
            if _custom_saved:
                _pick = st.selectbox(
                    "Configuración guardada",
                    ["(elegir)"] + _custom_saved,
                    format_func=lambda s: (s[len("custom_"):].replace("_", " ")
                                            if s.startswith("custom_") else s),
                    key="geom_custom_load_pick")
                _lc1, _lc2 = st.columns(2)
                if _lc1.button("Cargar", key="geom_custom_load_btn",
                               disabled=_pick == "(elegir)",
                               use_container_width=True):
                    _g = load_geometry(_pick)
                    if _g:
                        st.session_state["modal_geometry"] = _g
                        st.rerun()
                    else:
                        st.error("No se pudo cargar esa configuración.")
                if _lc2.button("Borrar", key="geom_custom_del_btn",
                               disabled=_pick == "(elegir)",
                               use_container_width=True):
                    _del_geom(_pick)
                    st.rerun()
            else:
                st.caption("Aún no hay configuraciones personalizadas guardadas.")

    # --- Configuración COMPLETA del análisis (geometría + sensores + parámetros)
    # Guarda TODO el ensayo como una plantilla reutilizable de un clic. Resuelve
    # el pedido de no reconfigurar sensores/parámetros en cada ensayo del mismo
    # activo (feedback de campo v3.31.440).
    with st.expander("📦 Configuración COMPLETA del análisis "
                     "(geometría + sensores + parámetros) — plantilla de 1 clic",
                     expanded=False):
        import json as _json_preset
        from core.modal.analysis_preset import (
            save_preset as _save_preset, load_preset as _load_preset,
            list_presets as _list_presets, delete_preset as _del_preset,
            PRESET_PARAM_KEYS as _PRESET_KEYS,
        )
        st.caption(
            "Un preset guarda **todo el ensayo en un archivo**: la geometría "
            "(bloques + sensores + DOF) y los parámetros del ensayo (f_min/f_max, "
            "duración, prominencia, RPM, promedios y el mapa de canales de la "
            "maleta). Cárgalo luego con **un clic** para repetir el mismo ensayo "
            "sin reconfigurar nada. Ideal para activos que mides periódicamente. "
            "No requiere Machinery Library — todo se guarda desde aquí.")

        _pc1, _pc2 = st.columns(2)
        with _pc1:
            st.markdown("**Guardar configuración completa**")
            _preset_name = st.text_input(
                "Nombre del preset", key="modal_preset_name",
                placeholder="ej. TES3 · OMA cojinetes · setup estándar")
            if st.button("💾 Guardar preset completo", key="modal_preset_save",
                         disabled=not _preset_name.strip(),
                         use_container_width=True):
                _params = {k: st.session_state.get(k)
                           for k in _PRESET_KEYS if k in st.session_state}
                try:
                    _save_preset(_preset_name, geom.to_json(), _params)
                    st.success(
                        f"✓ Preset **{_preset_name}** guardado: "
                        f"{len(geom.blocks)} bloque(s), {len(geom.sensors)} "
                        f"sensor(es) y {len(_params)} parámetro(s). "
                        "Recárgalo cuando quieras aquí a la derecha.")
                except Exception as _e:  # noqa: BLE001
                    st.error(f"No se pudo guardar: {_e}")
        with _pc2:
            st.markdown("**Cargar / borrar un preset**")
            _presets = _list_presets()
            if _presets:
                _psel = st.selectbox("Preset guardado", ["(elegir)"] + _presets,
                                     key="modal_preset_pick")
                _pb1, _pb2 = st.columns(2)
                if _pb1.button("📂 Cargar preset", key="modal_preset_load",
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
                            f"✓ Preset **{_psel}** cargado (geometría + "
                            "parámetros). Ve a Adquisición / OMA / EMA: los "
                            "valores quedaron restaurados.")
                        st.rerun()
                    else:
                        st.error("No se pudo cargar el preset.")
                if _pb2.button("🗑 Borrar", key="modal_preset_del",
                               disabled=_psel == "(elegir)",
                               use_container_width=True):
                    _del_preset(_psel)
                    st.rerun()
            else:
                st.caption("Aún no hay presets guardados. Configura tu ensayo y "
                           "guárdalo arriba con un nombre.")

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
            "🖱️ Zoom con la rueda del mouse · arrastra para rotar · "
            "el botón 🏠 restablece la vista.")
    except Exception as _exc_geom:  # noqa: BLE001
        st.warning(
            "⚠ No se pudo renderizar el preview 3D de la geometría. "
            "Posibles causas:\n"
            "1. No has seleccionado un activo o cargado una geometría.\n"
            "2. La geometría no tiene bloques definidos todavía.\n"
            "3. Hay un valor inválido en algún bloque.\n\n"
            "**Soluciones:** ve a la sección de geometría arriba y "
            "selecciona un activo registrado, o agrega bloques manualmente "
            "en la sección de edición debajo. Los demás tabs (Adquisición, "
            "EMA, OMA) sí están disponibles para usar."
        )
        with st.expander("Detalle técnico del error (para soporte)"):
            st.code(f"{type(_exc_geom).__name__}: {_exc_geom}", language="text")

    # ----- Editor de bloques + sensores -----
    col_edit_b, col_edit_s = st.columns(2)

    with col_edit_b:
        st.markdown("**Secciones del tren (bloques)**")
        if geom.blocks:
            import pandas as pd
            df_b = pd.DataFrame([
                {"Nombre": b.name, "Forma": b.shape, "Capa": b.kind,
                 "x_start": b.x_start, "x_end": b.x_end,
                 "R / hw,hh": (
                     f"{b.radius:.0f}" if b.shape == "cylinder"
                     else f"{b.half_width:.0f}, {b.half_height:.0f}"
                 )}
                for b in geom.blocks
            ])
            st.dataframe(df_b, hide_index=True, use_container_width=True)

        with st.expander("➕ Agregar / editar bloque", expanded=False):
            _action_b = st.radio("Acción", ["Agregar nuevo", "Editar existente",
                                                "Eliminar"],
                                  horizontal=True, key="geom_block_action")
            if _action_b == "Editar existente" and geom.blocks:
                _idx_b = st.selectbox("Bloque a editar",
                                        options=list(range(len(geom.blocks))),
                                        format_func=lambda i: geom.blocks[i].name,
                                        key="geom_block_edit_idx")
                _b_default = geom.blocks[_idx_b]
            elif _action_b == "Eliminar" and geom.blocks:
                _idx_b = st.selectbox("Bloque a eliminar",
                                        options=list(range(len(geom.blocks))),
                                        format_func=lambda i: geom.blocks[i].name,
                                        key="geom_block_del_idx")
                if st.button("Confirmar eliminación", key="geom_block_del_btn"):
                    geom.blocks.pop(_idx_b)
                    st.rerun()
                _b_default = None
            else:
                _b_default = GeometryBlock(id=f"b{len(geom.blocks)+1}",
                                            name="Nuevo bloque")

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
                    _nm = st.text_input("Nombre", value=_b_default.name,
                                          key=f"geom_b_name_{_bsfx}")
                    _shape = st.selectbox(
                        "Forma", ["cylinder", "box"],
                        index=0 if _b_default.shape == "cylinder" else 1,
                        format_func=lambda s: {
                            "cylinder": "Cilindro (eje / rotor / carcasa redonda)",
                            "box": "Rectángulo / placa (skid, fan cooler, base, carcasa)",
                        }.get(s, s),
                        help="El 'Rectángulo/placa' es una caja: para un skid o "
                             "fan cooler usa half_width grande y half_height "
                             "pequeño (queda plano y ancho).",
                        key=f"geom_b_shape_{_bsfx}")
                    _x0 = st.number_input("x_start", value=float(_b_default.x_start),
                                            step=10.0, key=f"geom_b_x0_{_bsfx}")
                    _x1 = st.number_input("x_end", value=float(_b_default.x_end),
                                            step=10.0, key=f"geom_b_x1_{_bsfx}")
                with c2:
                    if _shape == "cylinder":
                        _r = st.number_input("Radio", value=float(_b_default.radius),
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
                    _op = st.slider("Opacidad", 0.1, 1.0, float(_b_default.opacity),
                                      0.05, key=f"geom_b_op_{_bsfx}")
                    _kind_opts = ["casing", "shaft", "coupling"]
                    _kind = st.selectbox(
                        "Capa de deformación",
                        _kind_opts,
                        index=_kind_opts.index(
                            _b_default.kind if _b_default.kind in _kind_opts
                            else "casing"
                        ),
                        key=f"geom_b_kind_{_bsfx}",
                        help=("casing: deforma con accels · shaft: deforma "
                              "con proxies · coupling: estático o interpolado"),
                    )

                if st.button("✓ Aplicar al bloque", key="geom_b_apply",
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
        st.markdown("**Sensores con dirección DOF**")
        if geom.sensors:
            import pandas as pd
            df_s = pd.DataFrame([
                {"Nombre": s.name, "Tipo": s.sensor_type,
                 "Mounting": s.effective_mounting(),
                 "x": s.x, "y": s.y, "z": s.z, "DOF": s.dof}
                for s in geom.sensors
            ])
            st.dataframe(df_s, hide_index=True, use_container_width=True)

        with st.expander("➕ Agregar / editar sensor", expanded=False):
            _action_s = st.radio("Acción", ["Agregar nuevo", "Editar existente",
                                                "Eliminar"],
                                  horizontal=True, key="geom_sensor_action")
            if _action_s == "Editar existente" and geom.sensors:
                _idx_s = st.selectbox("Sensor a editar",
                                        options=list(range(len(geom.sensors))),
                                        format_func=lambda i: geom.sensors[i].name,
                                        key="geom_sensor_edit_idx")
                _s_default = geom.sensors[_idx_s]
            elif _action_s == "Eliminar" and geom.sensors:
                _idx_s = st.selectbox("Sensor a eliminar",
                                        options=list(range(len(geom.sensors))),
                                        format_func=lambda i: geom.sensors[i].name,
                                        key="geom_sensor_del_idx")
                if st.button("Confirmar eliminación", key="geom_sensor_del_btn"):
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
                    _snm = st.text_input("Nombre", value=_s_default.name,
                                           key=f"geom_s_name_{_ssfx}")
                    _styp = st.selectbox("Tipo",
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
                        "Mounting (qué mide)",
                        _mnt_opts,
                        index=_mnt_opts.index(_cur_mnt),
                        key=f"geom_s_mounting_{_ssfx}",
                        help=("Auto = inferido del tipo (accel/vel → casing, "
                              "proximity → shaft_proximity). Override manual si "
                              "tienes un caso especial."),
                    )
                    _mnt_final = "" if _mnt_sel == "(auto)" else _mnt_sel

                if st.button("✓ Aplicar al sensor", key="geom_s_apply",
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
    st.subheader("Adquisición de datos")
    st.caption("Tres rutas: captura live desde la maleta, importar archivo pre-capturado, o importar FRFs legacy.")

    acq_mode = st.radio(
        "Origen de datos",
        [
            "📡 Captura en vivo con unidad de adquisición",
            "📁 Importar archivo de captura existente",
            "🔄 Importar datos legacy (.txt)",
        ],
        horizontal=True,
        key="acq_mode_radio",
    )

    # -------- módulo de adquisición live --------
    if acq_mode.startswith("📡"):
        st.markdown("**Configuración de captura Watermelon**")

        # --- Selector de modo (gobierna el resto del formulario) ---
        ni_mode_sel = st.selectbox(
            "Modo de ensayo",
            ["EMA — Impact Hammer", "OMA — Continuous"],
            key="ni_mode",
            help=("EMA: impactos con martillo instrumentado, requiere ≥3 promedios. "
                  "OMA: registro continuo bajo condiciones operacionales, "
                  "requiere ≥ 2000 × T_low según Brincker & Ventura 2015."),
        )
        is_oma = ni_mode_sel.startswith("OMA")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Sample rate (Hz)", value=5120, step=1024, key="ni_fs",
                              min_value=1024, max_value=51200,
                              help="El sistema acepta hasta 51.2 kS/s/ch. "
                                   "Típico: 5120 Hz (banda útil 0–2 kHz).")

        # --- Bifurcación EMA vs OMA con tiempos normativos ---
        # Usamos keys separados por modo (ni_fn_low_oma, ni_dur_oma, ni_dur_ema,
        # ni_avg_ema) para que el switch entre modos no cause crashes por
        # valores fuera del rango del widget del modo opuesto.
        if is_oma:
            # OMA: regla Brincker & Ventura 2015 — T_min ≥ 2000 × T_low
            with col2:
                fn_low_hz = st.number_input(
                    "f_min de interés (Hz)",
                    value=float(st.session_state.get("ni_fn_low_oma", 5.0)),
                    min_value=0.5, max_value=200.0, step=0.5,
                    key="ni_fn_low_oma",
                    help=("Frecuencia natural más baja que esperas identificar. "
                          "Define el tiempo mínimo de captura: "
                          "T_min ≥ 2000 / f_min (Brincker & Ventura 2015)."),
                )
            with col3:
                # T_min normativo según fn_low
                _t_min_strict = 2000.0 / max(fn_low_hz, 0.1)   # 2000 × T_low (recomendado)
                _t_min_floor  = 1000.0 / max(fn_low_hz, 0.1)   # 1000 × T_low (mínimo absoluto)
                _t_default = max(120.0, _t_min_strict)
                ni_dur = st.number_input(
                    "Duración (s)",
                    value=float(st.session_state.get("ni_dur_oma", _t_default)),
                    min_value=30.0, max_value=3600.0, step=30.0,
                    key="ni_dur_oma",
                    help=f"T_min recomendado = 2000/f_min = {_t_min_strict:.0f} s. "
                         f"T_min absoluto = 1000/f_min = {_t_min_floor:.0f} s.",
                )
            # avg no aplica para OMA
            ni_avg = 1

            # --- Diagnóstico normativo OMA ---
            if ni_dur < _t_min_floor:
                modal_status_banner(
                    title=f"Duración insuficiente · {ni_dur:.0f} s < T_min absoluto {_t_min_floor:.0f} s",
                    detail=(
                        f"Para f_min = {fn_low_hz:.1f} Hz, la norma exige al menos "
                        f"**1000 × T_low = {_t_min_floor:.0f} s**, recomendado "
                        f"**2000 × T_low = {_t_min_strict:.0f} s** "
                        "(Brincker & Ventura 2015 · ISO 18649). Con menos tiempo, "
                        "el FDD pierde resolución espectral y los damping ratios "
                        "tienen varianza inaceptable. **Aumenta la duración antes "
                        "de iniciar la captura.**"
                    ),
                    severity="fail",
                )
                _can_capture = False
            elif ni_dur < _t_min_strict:
                modal_status_banner(
                    title=f"Duración aceptable pero subóptima · {ni_dur:.0f} s",
                    detail=(
                        f"Cumples el piso 1000 × T_low ({_t_min_floor:.0f} s) pero "
                        f"estás por debajo del recomendado 2000 × T_low "
                        f"({_t_min_strict:.0f} s). Los modos se identificarán pero "
                        "la incertidumbre en damping puede ser alta. "
                        "Sube la duración si el activo lo permite."
                    ),
                    severity="warning",
                )
                _can_capture = True
            else:
                modal_status_banner(
                    title=f"Duración conforme a norma · {ni_dur:.0f} s ≥ {_t_min_strict:.0f} s",
                    detail=(
                        f"Cumples 2000 × T_low para f_min = {fn_low_hz:.1f} Hz. "
                        "Brincker & Ventura 2015 · ISO 18649."
                    ),
                    severity="ok",
                )
                _can_capture = True

        else:
            # EMA: ISO 7626-5 secc. 6.3 — ≥3 promedios, duración 1–2 s por impacto
            with col2:
                ni_dur = st.number_input(
                    "Duración por impacto (s)",
                    value=float(st.session_state.get("ni_dur_ema", 2.0)),
                    min_value=0.5, max_value=10.0, step=0.5,
                    key="ni_dur_ema",
                    help="Window suficiente para que la respuesta decaiga a < 1% "
                         "del pico (evita leakage). Típico 1–2 s para máquinas industriales.",
                )
            with col3:
                ni_avg = st.number_input(
                    "N° de impactos a promediar",
                    value=int(st.session_state.get("ni_avg_ema", 5)),
                    min_value=1, max_value=30, step=1,
                    key="ni_avg_ema",
                    help="ISO 7626-5 secc. 6.3: mínimo 3, recomendado 5–10. "
                         "Más promedios → mejor relación señal/ruido.",
                )
            # fn_low no aplica para EMA
            fn_low_hz = 0.0

            # --- Diagnóstico normativo EMA ---
            if ni_avg < 3:
                modal_status_banner(
                    title=f"N° de impactos {ni_avg} insuficiente — norma exige ≥ 3",
                    detail=(
                        "ISO 7626-5 secc. 6.3 requiere **mínimo 3 promedios** para "
                        "estimación válida de FRF. Con un solo impacto no hay "
                        "control de coherencia y los modos pueden ser ruido. "
                        "Aumenta a 5–10 promedios antes de iniciar."
                    ),
                    severity="fail",
                )
                _can_capture = False
            elif ni_avg < 5:
                modal_status_banner(
                    title=f"N° de impactos {ni_avg} cumple el mínimo · recomendado 5–10",
                    detail=(
                        "ISO 7626-5 secc. 6.3 permite 3 promedios como piso pero "
                        "recomienda 5–10 para reducir la varianza de la FRF. "
                        "El checklist de coherencia post-captura será más exigente."
                    ),
                    severity="warning",
                )
                _can_capture = True
            else:
                modal_status_banner(
                    title=f"Configuración EMA conforme a norma · {ni_avg} promedios × {ni_dur:.1f} s",
                    detail=(
                        "ISO 7626-5 secc. 6.3 cumplido (≥ 5 promedios). Total estimado "
                        f"de captura: ≈ {ni_avg * ni_dur:.0f} s + esperas entre impactos."
                    ),
                    severity="ok",
                )
                _can_capture = True

        # --- Canales activos: grid 32 BNC con auto-discovery ---
        # v3.31.202 — Reemplaza el grid hardcoded de 4 checkboxes por una
        # tabla editable de 32 filas (1 por BNC port). Auto-detecta qué
        # módulos módulo de adquisición están instalados en la maleta y pre-popula el
        # default. Genera el comando --channels para el companion script.
        st.markdown("**Canales activos · Maleta Watermelon (BNC 1..32)**")

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
                    f"✓ Detectados {len(_modules)} módulos en la maleta '{_ni_chassis}' "
                    f"→ BNC 1..{_bnc_max} disponibles"
                )
            else:
                _discovery_msg = (
                    f"⚠ No se detectó la maleta Watermelon '{_ni_chassis}'. "
                    "Puedes configurar canales para captura remota, pero la "
                    "ejecución se hará desde la laptop de planta vía companion."
                )
        except ImportError:
            _discovery_msg = (
                "ℹ Drivers de adquisición Watermelon no disponibles en este equipo (esperado en modo Cloud). "
                "Configura los canales aquí y ejecuta el comando técnico desde la "
                "laptop de planta con el companion script."
            )
        except Exception as _exc:  # noqa: BLE001
            _discovery_msg = f"⚠ Discovery falló: {_exc}"

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
                    "BNC", help="Puerto BNC frontal de la maleta (1..32)",
                    disabled=True, width="small",
                ),
                "Slot": st.column_config.NumberColumn(
                    "Slot", help="Módulo (1..8) dentro de la maleta",
                    disabled=True, width="small",
                ),
                "Habilitado": st.column_config.CheckboxColumn(
                    "✓", help="Marca para incluir este canal en la captura",
                    width="small",
                ),
                "Nombre": st.column_config.TextColumn(
                    "Sensor", help="Etiqueta del sensor (ej: 1YA, VE5807, Hammer)",
                    max_chars=20, width="medium",
                ),
                "Coupling": st.column_config.SelectboxColumn(
                    "Coupling", options=["IEPE", "AC", "DC"],
                    help="IEPE para acelerómetros, AC para sondas de proximidad, DC raro",
                    width="small",
                ),
                "Sens (mV/EU)": st.column_config.NumberColumn(
                    "Sens", help="Sensibilidad: 100 mV/g acelerómetro, 200 mV/mil sonda de proximidad, 2.4 mV/N martillo modal",
                    min_value=0.1, max_value=10000.0, step=0.1, format="%.2f",
                    width="small",
                ),
                "Unidad": st.column_config.SelectboxColumn(
                    "EU", options=["g", "mil", "N", "ips", "mm/s"],
                    help="Unidad de ingeniería del sensor",
                    width="small",
                ),
                "HW": st.column_config.TextColumn(
                    "HW", help="✓ = módulo instalado en este slot, — = vacío",
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
        _kpi_col1.metric("Canales habilitados", _n_enabled, delta=f"de 32 max")
        _slots_used = sorted({r["Slot"] for r in _enabled_rows})
        _kpi_col2.metric("Módulos requeridos", len(_slots_used),
                          delta=f"slots {_slots_used}" if _slots_used else "—")
        _ram_est_mb = (_n_enabled * float(st.session_state.get("ni_dur_oma", ni_dur))
                        * float(st.session_state.get("ni_fs", 5120)) * 4) / (1024 * 1024)
        _kpi_col3.metric("RAM streaming estimada", f"{_ram_est_mb:.0f} MB",
                          help="Con streaming TDMS la RAM se mantiene ~5 MB constante "
                               "sin importar la duración (esto es el tamaño total del "
                               "TDMS final en disco, no la RAM durante captura).")

        # Validación hardware vs slots requeridos
        if _installed_slots and _slots_used:
            _missing = [s for s in _slots_used if s not in _installed_slots]
            if _missing:
                modal_status_banner(
                    title=f"⚠ Hardware faltante en {len(_missing)} slot(s)",
                    detail=(
                        f"Habilitaste canales en slots {_missing} pero esos módulos "
                        f"módulos no están instalados en la maleta. Slots con módulos: "
                        f"{sorted(_installed_slots)}. **Antes de capturar:** o desactiva "
                        f"esos canales o instala los módulos faltantes."
                    ),
                    severity="fail",
                )
                _can_capture = False

        if _n_enabled == 0:
            modal_status_banner(
                title="Sin canales habilitados",
                detail="Marca al menos un canal en la tabla para poder capturar.",
                severity="warning",
            )
            _can_capture = False
        elif not is_oma and not any(
            r.get("Habilitado") and (r.get("Coupling") == "IEPE" and r.get("Sens (mV/EU)", 100) < 10)
            for r in _enabled_rows
        ):
            modal_status_banner(
                title="Modo EMA sin canal de martillo identificable",
                detail=(
                    "Para impact hammer testing necesitas al menos 1 canal con "
                    "sensitivity baja (típicamente 2.4 mV/N para PCB modal hammer). "
                    "Configura el martillo en BNC 1."
                ),
                severity="warning",
            )

        # Persistir flag para el bloque del comando técnico
        st.session_state["_modal_can_capture"] = _can_capture

        # Nota técnica para especialistas — accesible vía role admin/specialist.
        # Detalles de implementación NO se muestran al cliente.
        _user_role = (_user.get("role", "") or "").strip().lower()
        if _user_role in ("admin", "specialist"):
            with st.expander("▸ Comando del módulo de captura (técnico)",
                              expanded=False):
                st.caption(
                    "Referencia técnica para el operador con acceso a la unidad "
                    "de adquisición. Esta sección solo es visible para usuarios "
                    "internos (admin/specialist)."
                )
                if not st.session_state.get("_modal_can_capture", True):
                    st.error(
                        "⚠ Configuración no conforme a norma. Ajusta los parámetros "
                        "arriba antes de ejecutar la captura."
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
                    f"📊 {len(_enabled_rows)} canales habilitados · "
                    f"{len(_slots_used)} módulos requeridos · "
                    f"modo {_mode_token.upper()}"
                )

                if _user_role == "admin":
                    st.code(" \\\n    ".join(["python capture.py"] + _cmd_lines),
                             language="bash")
                else:
                    st.code(" \\\n    ".join(_cmd_lines), language="text")

                st.caption(
                    "Copia y ejecuta este comando en la laptop de planta donde "
                    "está conectada la maleta. NO requiere internet. El .tdms "
                    "resultante se sube luego en '📁 Importar archivo'."
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
                        "**🎙 Captura directa desde Watermelon · este equipo**"
                    )
                    st.caption(
                        "Alternativa al comando técnico: ejecutar la captura "
                        "ahora mismo desde este equipo sin pasar por terminal. "
                        "Requiere maleta Watermelon conectada por USB y "
                        "drivers de adquisición instalados (✓ detectado)."
                    )

                    _capture_disabled = (
                        _n_enabled == 0
                        or not st.session_state.get("_modal_can_capture", True)
                    )

                    if st.button(
                        "🎙 Iniciar captura ahora",
                        type="primary",
                        disabled=_capture_disabled,
                        key="ni_capture_now_btn",
                        help=(
                            "Ejecuta la captura inmediatamente. La UI se "
                            "bloquea hasta que termine (típico EMA ~30s, "
                            "OMA según --duration ~60-300s+). El TDMS se "
                            "guarda en data/modal/captures/ con timestamp."
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
                        _progress_bar = st.progress(0.0, text="Iniciando captura...")

                        def _on_capture_progress(_p, _s):
                            try:
                                _progress_bar.progress(
                                    min(max(_p, 0.0), 1.0), text=_s
                                )
                            except Exception:
                                pass  # progress bar puede no existir si rerun

                        try:
                            with st.spinner(
                                f"Capturando {len(_capture_channels)} canales "
                                f"× {_capture_cfg.duration_s:.0f}s..."
                            ):
                                _result_path = _ni_capture(
                                    _capture_cfg,
                                    on_progress=_on_capture_progress,
                                )
                            _progress_bar.empty()
                            st.success(
                                f"✓ Captura completa: "
                                f"`{_result_path.name}` en "
                                f"`{_result_path.parent}/`"
                            )
                            st.session_state["_last_capture_path"] = str(
                                _result_path
                            )
                        except Exception as _exc:  # noqa: BLE001
                            _progress_bar.empty()
                            st.error(f"✗ Captura falló: {_exc}")
                            st.caption(
                                "Verifica que la maleta esté conectada y los "
                                "módulos instalados. Para diagnóstico corre "
                                "`python scripts/capture_companion/capture.py "
                                "--list-modules` en CLI."
                            )

                    # Auto-load: si hay captura reciente, ofrecer botón
                    # para procesarla en este análisis sin tener que ir
                    # al uploader.
                    _last_capture = st.session_state.get(
                        "_last_capture_path"
                    )
                    if _last_capture and Path(_last_capture).exists():
                        if st.button(
                            "📥 Cargar última captura en este análisis",
                            key="auto_load_captured_tdms",
                            help=(
                                f"Procesa {Path(_last_capture).name} "
                                "como si lo hubieras subido por '📁 "
                                "Importar archivo'."
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
                                    "✓ TDMS cargado. Cambia el radio a "
                                    "'📁 Importar archivo' para verlo "
                                    "procesado contra ISO 7626-5."
                                )
                            except Exception as _exc:  # noqa: BLE001
                                st.error(f"Error cargando TDMS: {_exc}")

    # -------- TDMS existente --------
    elif acq_mode.startswith("📁"):
        st.markdown("**Cargar archivo de captura (.tdms)**")

        # v3.31.209 — Fuente del TDMS: upload local vs Cloud (planta sync)
        _tdms_source = st.radio(
            "Fuente del archivo",
            ["💾 Subir desde mi PC", "☁ Desde Watermelon Cloud (sync planta)"],
            horizontal=True,
            key="tdms_source",
            help=(
                "Cloud: muestra los TDMS que sincronizaste desde Watermelon "
                "Planta Edition (laptop de planta). Requiere haber sincronizado "
                "al menos un TDMS previamente."
            ),
        )

        tdms_up = None

        if _tdms_source.startswith("💾"):
            st.caption(
                "Carga el .tdms generado por el companion script, Watermelon "
                "Planta o LabVIEW. Watermelon ejecuta automáticamente el "
                "checklist ISO 7626-5 sobre el ensayo."
            )
            tdms_up = st.file_uploader(
                "Selecciona .tdms", type=["tdms"], key="tdms_up",
            )
        else:
            # ☁ Cloud — lista los TDMS del bucket modal-captures del user
            st.caption(
                "Estos son los TDMS que sincronizaste al Watermelon Cloud "
                "desde Watermelon Planta Edition. Se listan los más recientes "
                "primero."
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
                        "Cliente Supabase no inicializado. Verifica que "
                        "st.secrets[supabase].service_key esté configurado en "
                        "Streamlit Cloud."
                    )
                _user_email = _user.get("email", "")
                if not _user_email:
                    st.error("No se pudo determinar tu email. Re-loguéate.")
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
                            f"No se pudo listar el bucket modal-captures: "
                            f"{_exc}. Verifica que el bucket existe en "
                            f"Supabase y que tienes las RLS policies aplicadas."
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
                            f"No hay TDMS sincronizados para `{_user_email}`. "
                            f"Para que aparezcan acá, abre Watermelon Planta "
                            f"Edition en tu laptop de planta, captura algunos "
                            f"datos y haz click en **'Sync ahora'** estando "
                            f"online."
                        )
                    else:
                        st.success(
                            f"✓ **{len(_all_tdms)} TDMS** disponibles desde "
                            f"Watermelon Planta"
                        )
                        _options = {
                            f"{t['name']} ({t['size']/(1024*1024):.1f} MB · "
                            f"{(t['created_at'] or 'N/A')[:19]})": t
                            for t in _all_tdms
                        }
                        _selected_label = st.selectbox(
                            "Selecciona un TDMS para procesar",
                            options=list(_options.keys()),
                            key="cloud_tdms_select",
                        )
                        _selected_meta = _options.get(_selected_label)

                        if _selected_meta and st.button(
                            "⬇ Cargar y procesar este TDMS",
                            type="primary",
                            use_container_width=True,
                            key="cloud_tdms_load_btn",
                        ):
                            with st.spinner(
                                f"Descargando {_selected_meta['name']}..."
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
                                        f"✓ TDMS cargado del Cloud "
                                        f"({len(_bytes)/(1024*1024):.1f} MB) "
                                        f"— procesando..."
                                    )
                                    # Forzar rerun para que el resto del Tab
                                    # vea el TDMS en session_state y renderice
                                    # la validación ISO 7626-5
                                    st.rerun()
                                except Exception as _exc_load:
                                    st.error(
                                        f"Error cargando TDMS: {_exc_load}"
                                    )
            except Exception as _exc:  # noqa: BLE001
                st.error(
                    f"Error accediendo al Cloud: {_exc}. "
                    f"Verifica que tu sesión es válida y que el bucket "
                    f"`modal-captures` existe en Supabase."
                )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tdms_f_target = st.number_input(
                "Frecuencia objetivo (Hz)", value=500.0, step=50.0,
                key="tdms_ftarget",
                help="Banda alta de interés del ensayo. ISO 7626-5 valida que "
                "el martillo excite plano hasta esta frecuencia.",
            )
        with col_t2:
            tdms_coh_thr = st.number_input(
                "γ² mínimo aceptable", value=0.8, step=0.05,
                min_value=0.5, max_value=1.0, key="tdms_coh",
                help="ISO 7626-5 secc. 7.4 — coherencia mínima en banda de interés. "
                "Típico 0.8, estricto 0.9.",
            )

        if tdms_up and st.button("🔬 Procesar y validar contra ISO 7626-5",
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
                st.error(f"Error cargando TDMS: {exc}")
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
            mc2.metric("Modo", tdms.mode or "—")
            mc3.metric("Canales", len(tdms.channels))
            mc4.metric("Promedios", tdms.n_averages or "—")

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
                        "ℹ TDMS sin metadata explícita de mode — detectado como OMA "
                        "(sin canal de martillo identificable)."
                    )
                else:
                    is_ema_tdms = True

            if is_oma_tdms:
                st.markdown(
                    f'<div style="background:#dbeafe;border:1.5px solid #2563eb;'
                    f'border-radius:8px;padding:14px 18px;">'
                    f'<div style="font-weight:800;color:#1e3a8a;font-size:16px;">'
                    f'🌊 TDMS modo OMA detectado</div>'
                    f'<div style="color:#1e40af;font-size:13px;margin-top:4px;">'
                    f'Operational Modal Analysis — sin martillo, evaluado bajo '
                    f'<b>ISO 20816</b> (no ISO 7626-5). Procesa con FDD en el '
                    f'<b>Tab OMA →</b>.'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                # Preview rápido: time-domain de cada canal
                st.markdown("### Preview canales operacionales")
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
                fig_prev.update_xaxes(title_text="Tiempo (s)",
                                        row=n_show, col=1)
                fig_prev.update_layout(
                    height=max(280, 150 * n_show),
                    template="plotly_white",
                    margin=dict(l=50, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_prev, use_container_width=True)
                st.caption(
                    "ℹ Datos operacionales listos para FDD. Cambia al **Tab OMA** "
                    "para identificar modos naturales sin necesidad de martillo."
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
                            "⚠ TDMS sin metadata de mode. Si es OMA, procesa "
                            "con FDD en el Tab OMA."
                        )
                    else:
                        st.warning(
                            "⚠ No se detectó canal de martillo automáticamente. "
                            "ISO 7626-5 requiere un input claro. Verifica que el "
                            "primer canal sea el martillo con sensitivity baja "
                            "(~2.4 mV/N) o nombre 'Hammer'/'Martillo'."
                        )
                    _skip_ema_validation = True

            if not _skip_ema_validation:
                st.success(
                    f"🔨 Martillo detectado: **{hammer.name}** "
                    f"(kurtosis {hammer.kurtosis:.1f}, "
                    f"peak/RMS {hammer.peak_to_rms:.1f}, "
                    f"sens {hammer.sensitivity_mv_per_eu} mV/{hammer.units})"
                )

                # Selector de canal de respuesta
                resp_names = [r.name for r in responses]
                if not resp_names:
                    st.error("No hay canales de respuesta — el TDMS solo tiene martillo.")
                else:
                    resp_pick = st.selectbox(
                        "Canal de respuesta a analizar",
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
                    st.markdown("### ISO 7626-5 · Validación del ensayo")
                    if report.overall_pass:
                        st.markdown(
                            f'<div style="background:#dcfce7;border:1.5px solid #16a34a;'
                            f'border-radius:8px;padding:14px 18px;">'
                            f'<div style="font-weight:800;color:#14532d;font-size:18px;">'
                            f'✓ Ensayo conforme ISO 7626-5</div>'
                            f'<div style="color:#166534;font-size:13px;margin-top:4px;">'
                            f'{report.n_passed}/{report.n_total} checks aprobados</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    elif report.has_fails:
                        st.markdown(
                            f'<div style="background:#fee2e2;border:1.5px solid #dc2626;'
                            f'border-radius:8px;padding:14px 18px;">'
                            f'<div style="font-weight:800;color:#7f1d1d;font-size:18px;">'
                            f'✗ Ensayo NO conforme ISO 7626-5</div>'
                            f'<div style="color:#991b1b;font-size:13px;margin-top:4px;">'
                            f'{report.n_passed}/{report.n_total} checks aprobados · revisar fallos</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div style="background:#fef3c7;border:1.5px solid #d97706;'
                            f'border-radius:8px;padding:14px 18px;">'
                            f'<div style="font-weight:800;color:#78350f;font-size:18px;">'
                            f'⚠ Ensayo con observaciones</div>'
                            f'<div style="color:#92400e;font-size:13px;margin-top:4px;">'
                            f'{report.n_passed}/{report.n_total} checks · revisar warnings</div>'
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
                    with st.expander("📋 Ver detalle de cada check", expanded=False):
                        for c in report.checks:
                            icon = "✓" if c.passed else ("⚠" if c.severity == "warning" else "✗")
                            st.markdown(f"**{icon} {c.title}** · `{c.norm_ref}`")
                            st.caption(c.detail)
                            st.divider()

                    # PANEL DE 6 PLOTS — Input / Output / FRF / Coherencia
                    st.markdown(f"### Panel ISO 7626-5 · {hammer.name} → {resp.name}")

                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=3, cols=2,
                        subplot_titles=(
                            f"Input — {hammer.name} (tiempo)",
                            f"Input — {hammer.name} (espectro)",
                            f"Response — {resp.name} (tiempo)",
                            f"Response — {resp.name} (espectro)",
                            "FRF — Magnitud + Fase",
                            "Coherencia γ²(f)",
                        ),
                        vertical_spacing=0.10,
                        horizontal_spacing=0.08,
                    )

                    fig.add_trace(go.Scatter(
                        x=hammer.time_s, y=hammer.data, mode="lines",
                        name="Input time", line=dict(color="#0F1E3D", width=1),
                        showlegend=False,
                    ), row=1, col=1)
                    fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
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
                    fig.update_xaxes(title_text="Frecuencia (Hz)", row=1, col=2)
                    fig.update_yaxes(title_text=f"PSD (dB ref {hammer.units}²/Hz)", row=1, col=2)

                    fig.add_trace(go.Scatter(
                        x=resp.time_s, y=resp.data, mode="lines",
                        name="Response time", line=dict(color="#1AAEE5", width=1),
                        showlegend=False,
                    ), row=2, col=1)
                    fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
                    fig.update_yaxes(title_text=f"{resp.units}", row=2, col=1)

                    f_out, psd_out = _welch(resp.data, fs=tdms.sample_rate_hz,
                                              nperseg=nperseg)
                    fig.add_trace(go.Scatter(
                        x=f_out, y=10 * np.log10(np.maximum(psd_out, 1e-30)),
                        mode="lines", name="Response spec",
                        line=dict(color="#1AAEE5", width=1),
                        showlegend=False,
                    ), row=2, col=2)
                    fig.update_xaxes(title_text="Frecuencia (Hz)", row=2, col=2)
                    fig.update_yaxes(title_text=f"PSD (dB ref {resp.units}²/Hz)", row=2, col=2)

                    mag_db = 20 * np.log10(np.maximum(frf.magnitude, 1e-30))
                    fig.add_trace(go.Scatter(
                        x=frf.frequencies_hz, y=mag_db, mode="lines",
                        name="FRF Mag", line=dict(color="#0F7FB0", width=1.5),
                        showlegend=False,
                    ), row=3, col=1)
                    fig.update_xaxes(title_text="Frecuencia (Hz)", row=3, col=1)
                    fig.update_yaxes(title_text="Magnitud (dB)", row=3, col=1)

                    fig.add_trace(go.Scatter(
                        x=frf.frequencies_hz, y=frf.coherence, mode="lines",
                        name="γ²", line=dict(color="#16a34a", width=1.5),
                        showlegend=False,
                    ), row=3, col=2)
                    fig.add_hline(y=coh_thr, line=dict(color="#D89B22", dash="dash"),
                                   row=3, col=2)
                    fig.update_xaxes(title_text="Frecuencia (Hz)", row=3, col=2)
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
                        f"📊 FRF computada via estimador {frf.estimator} · "
                        f"{frf.n_averages} segmentos Welch · ventana {frf.window} · "
                        f"nperseg = {nperseg}. Disponible en Tab EMA para identificación modal."
                    )

    # -------- FRFs legacy (formato .txt) --------
    else:
        st.markdown("**Importar FRFs legacy (.txt de software modal anterior)**")

        uploaded = st.file_uploader(
            "Subir archivos .txt",
            type=["txt"],
            accept_multiple_files=True,
            key="art_up",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            art_fs = st.number_input(
                "Sample rate original (Hz)",
                value=2560, step=100, min_value=10, key="art_fs",
            )
        with col_b:
            art_bw = st.number_input(
                "Bandwidth (Hz)",
                value=1280, step=100, min_value=1, key="art_bw",
            )
        st.caption(
            "El eje de frecuencia se reconstruye como Δf = bandwidth / (N_bins - 1). "
            "Los archivos legacy NO guardan el eje — requerido completar manualmente."
        )

        if uploaded and st.button("🔍 Procesar archivos legacy",
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
                    f"✓ {len(loaded)} archivos procesados · "
                    f"Δf = {loaded[0].df:.3f} Hz · "
                    f"{loaded[0].n_bins} bins · "
                    f"banda 0 → {loaded[0].frequencies_hz[-1]:.0f} Hz"
                )
            if errors:
                for e in errors:
                    st.error(e)

        # Mostrar FRFs cargadas
        frfs = st.session_state.get("modal_frfs", [])
        if frfs:
            st.divider()
            st.markdown(f"### Plot Bode — {len(frfs)} canal(es) cargado(s)")

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
                title="Magnitud — dB",
                xaxis_title="Frecuencia (Hz)",
                yaxis_title="Magnitud (dB)",
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
                    title="Fase — grados",
                    xaxis_title="Frecuencia (Hz)",
                    yaxis_title="Fase (°)",
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
    st.subheader("Análisis Modal Experimental")
    st.caption(
        "Identificación de parámetros modales (frecuencia natural, damping, mode shape) "
        "por método Circle-Fit Nyquist (Kennedy-Pancu) + half-power. "
        "Cumple ISO 7626-6 secc. 6.3."
    )

    # ─── Si hay TDMS procesado, ofrecer identificación moderna ───────
    tdms_frf = st.session_state.get("modal_tdms_frf")
    tdms_pair = st.session_state.get("modal_tdms_pair")
    if tdms_frf is not None and tdms_pair:
        st.markdown(
            f'<div style="background:#dcfce7;border:1.5px solid #16a34a;'
            f'border-radius:8px;padding:14px 18px;margin-bottom:18px;">'
            f'<div style="font-weight:800;color:#14532d;font-size:15px;">'
            f'🎯 FRF disponible desde captura TDMS ISO 7626-5</div>'
            f'<div style="color:#166534;font-size:13px;margin-top:4px;">'
            f'Pair: <b>{tdms_pair[0]} → {tdms_pair[1]}</b> · '
            f'{len(tdms_frf.frequencies_hz)} bins · '
            f'estimator {tdms_frf.estimator} · '
            f'γ² promedio {tdms_frf.coherence.mean():.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Identificación modal sobre FRF medida (Circle-Fit Nyquist)**")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            ti_f_min = st.number_input(
                "f mín (Hz)", value=5.0, step=1.0, key="ti_fmin",
            )
        with col_t2:
            _f_max_def = float(tdms_frf.frequencies_hz[-1])
            ti_f_max = st.number_input(
                "f máx (Hz)", value=_f_max_def, step=10.0, key="ti_fmax",
            )
        with col_t3:
            ti_prom = st.number_input(
                "Prominencia (dB)", value=12.0, step=1.0, key="ti_prom",
                help="Default 12 dB · estricto. Bájalo para más sensibilidad.",
            )
        with col_t4:
            ti_dist = st.number_input(
                "Distancia mín (Hz)", value=5.0, step=1.0, key="ti_dist",
            )

        if st.button("🎯 Identificar modos (Circle-Fit Nyquist + half-power)",
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
            st.markdown(f"### Modos identificados — {len(tdms_modes)} (FRF TDMS)")

            import pandas as pd

            def _method_for(m):
                return "Circle-Fit Nyquist" if m.confidence >= 0.9 else "Half-power"

            df_tdms = pd.DataFrame([
                {
                    "Modo": m.mode_number,
                    "Frecuencia (Hz)": round(m.natural_frequency_hz, 2),
                    "Damping (%)": round(m.damping_ratio_pct, 3),
                    "Método": _method_for(m),
                    "Confianza": round(m.confidence, 2),
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
                name="Modos",
                marker=dict(color=colors, size=12, symbol="diamond",
                             line=dict(width=1.5, color="#0F1E3D")),
                text=[str(m.mode_number) for m in tdms_modes],
                textposition="top center",
                textfont=dict(size=10, color="#0F1E3D"),
                customdata=[
                    f"Modo {m.mode_number}<br>{m.natural_frequency_hz:.2f} Hz · "
                    f"ζ={m.damping_ratio_pct:.3f}%<br>"
                    f"{_method_for(m)} · conf={m.confidence:.2f}"
                    for m in tdms_modes
                ],
                hovertemplate="%{customdata}<extra></extra>",
            ))
            fig_t.update_layout(
                title="FRF medida con modos identificados (verde = Circle-Fit, ámbar = Half-power)",
                xaxis_title="Frecuencia (Hz)",
                yaxis_title="Magnitud (dB)",
                height=420,
                margin=dict(l=50, r=20, t=60, b=40),
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig_t, use_container_width=True)

            st.caption(
                "🔬 Método Circle-Fit Nyquist (Kennedy-Pancu 1947) — clásico en EMA "
                "para modos SDOF. Modos en verde han pasado el ajuste circular "
                "(precisión típica < 1% en fn). Modos en ámbar usan half-power como "
                "fallback (precisión 2-5%). Ambos cumplen ISO 7626-6 secc. 6.3."
            )

        st.divider()

    # ─── Sección FRFs legacy (cargadas via .txt) ──────────────────────
    frfs = st.session_state.get("modal_frfs", [])
    if not frfs and tdms_frf is None:
        st.info("📭 No hay FRFs cargadas. Carga datos en el tab Adquisición primero "
                "(archivo legacy .txt o captura de la maleta Watermelon).")
    elif not frfs:
        pass  # solo TDMS cargado — UI ya mostrada arriba
    else:
        st.markdown(f"**{len(frfs)} FRF(s) cargadas — listas para identificación modal**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ema_f_min = st.number_input("f mín (Hz)", value=5.0, step=1.0,
                                          key="ema_fmin")
        with col2:
            _f_max_default = float(frfs[0].frequencies_hz[-1])
            ema_f_max = st.number_input("f máx (Hz)", value=_f_max_default,
                                          step=10.0, key="ema_fmax")
        with col3:
            ema_prom = st.number_input("Prominencia (dB)", value=6.0,
                                         step=1.0, key="ema_prom",
                                         help="Mínima altura del pico vs entorno")
        with col4:
            ema_dist = st.number_input("Distancia mín (Hz)", value=2.0,
                                         step=0.5, key="ema_dist",
                                         help="Separación mínima entre picos")

        if st.button("🎯 Identificar modos", type="primary",
                       use_container_width=True, key="ema_run_btn"):
            from core.modal.frf_compute import detect_modal_peaks

            # Selección de FRF principal — la primera de 2 columnas (FRF compleja)
            # o la primera del listado si no hay FRF compleja
            primary = next((f for f in frfs if f.is_complex_frf), frfs[0])
            mag = primary.magnitude_linear()
            if mag.size == 0:
                st.error("La FRF seleccionada no tiene magnitud computable.")
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
            st.markdown(f"### Modos identificados — {len(peaks)}")

            # Tabla modal
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Modo": i + 1,
                    "Frecuencia (Hz)": round(p.frequency_hz, 2),
                    "Damping (%)": round(p.damping_ratio_pct, 3),
                    "Bandwidth (Hz)": round(p.bandwidth_hz, 3),
                    "Q factor": round(p.quality_factor, 1),
                    "Magnitud peak": f"{p.magnitude_peak:.3e}",
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
                f"Modo {i+1}<br>{p.frequency_hz:.1f} Hz · ζ={p.damping_ratio_pct:.2f}%"
                for i, p in enumerate(peaks)
            ]
            fig_peaks.add_trace(go.Scatter(
                x=peak_freqs, y=peak_mag_db, mode="markers+text",
                name="Modos",
                marker=dict(color="#D89B22", size=10, symbol="diamond",
                             line=dict(width=1.5, color="#7c2d12")),
                text=[str(i + 1) for i in range(len(peaks))],
                textposition="top center",
                textfont=dict(size=10, color="#7c2d12"),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=peak_labels,
            ))
            fig_peaks.update_layout(
                title=f"FRF con {len(peaks)} modos identificados — hover en diamantes para detalle",
                xaxis_title="Frecuencia (Hz)",
                yaxis_title="Magnitud (dB)",
                height=420,
                margin=dict(l=50, r=20, t=60, b=40),
                template="plotly_white",
                hovermode="closest",
                showlegend=False,
            )
            st.plotly_chart(fig_peaks, use_container_width=True)

            st.caption(
                "🔬 Damping calculado por método half-power (-3 dB · ISO 7626-6 secc. 6.3.2). "
                "Diamantes ámbar marcan los modos detectados — hover para frecuencia + damping. "
                "Para mode shapes y curve fit LSCF, se requiere integración pyEMA (próximo sprint)."
            )


# ---------------------------------------------------------------------
# Tab 4 — OMA Processing (FDD)
# ---------------------------------------------------------------------
if _active_modal_tab == "🌊 OMA":
    st.subheader("Análisis Modal Operacional — FDD")
    st.caption(
        "Frequency Domain Decomposition (Brincker 2001) sobre datos operacionales "
        "del sistema. Sin necesidad de martillo. Cumple ISO 20816 + API 684."
    )

    tdms_oma = st.session_state.get("modal_tdms")
    if tdms_oma is None:
        modal_empty_state(
            icon="🌊",
            title="Sin datos operacionales cargados",
            description=(
                "El análisis OMA requiere un archivo .tdms con captura continua "
                "del sistema — mínimo 60 segundos a velocidad constante (ISO 20816 + "
                "Brincker 2001). Carga el archivo en el Tab Adquisición usando la "
                "opción 'Importar .tdms existente'."
            ),
            cta_label="Cambia al Tab Adquisición",
            norm_ref="ISO 20816 · FDD requirements",
        )
    else:
        # Mostrar metadata
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Modo TDMS", tdms_oma.mode or "—")
        mc2.metric("Canales", len(tdms_oma.channels))
        mc3.metric("Duración",
                    f"{tdms_oma.channels[0].duration_s:.1f}s" if tdms_oma.channels else "—")
        mc4.metric("Fs", f"{tdms_oma.sample_rate_hz:.0f} Hz")

        _record_dur = (tdms_oma.channels[0].duration_s if tdms_oma.channels else 0)

        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
        with col_o1:
            oma_fmin = st.number_input(
                "f mín (Hz)", value=5.0, step=1.0, key="oma_fmin",
                help=("Frecuencia natural más baja a identificar. Define el "
                      "tiempo mínimo del record: T_min ≥ 2000 / f_min "
                      "(Brincker & Ventura 2015)."),
            )

        # --- Validación normativa del record contra fn_low ---
        _t_min_strict_tdms = 2000.0 / max(float(oma_fmin), 0.1)
        _t_min_floor_tdms  = 1000.0 / max(float(oma_fmin), 0.1)
        if _record_dur > 0:
            if _record_dur < _t_min_floor_tdms:
                modal_status_banner(
                    title=(f"Record {_record_dur:.0f} s < T_min absoluto "
                             f"{_t_min_floor_tdms:.0f} s para f_min = {oma_fmin:.1f} Hz"),
                    detail=(
                        f"La norma exige al menos **1000 × T_low = "
                        f"{_t_min_floor_tdms:.0f} s** y recomienda "
                        f"**2000 × T_low = {_t_min_strict_tdms:.0f} s** "
                        "(Brincker & Ventura 2015 · ISO 18649). El FDD se ejecutará "
                        "pero los damping ratios pueden tener varianza > 30%. "
                        "Para resultados de reporte, recapturar con más tiempo o "
                        "subir el f_min si los modos bajos no son de interés."
                    ),
                    severity="fail",
                )
            elif _record_dur < _t_min_strict_tdms:
                modal_status_banner(
                    title=(f"Record {_record_dur:.0f} s acepta el piso normativo · "
                             f"recomendado {_t_min_strict_tdms:.0f} s"),
                    detail=(
                        f"Para f_min = {oma_fmin:.1f} Hz cumples "
                        f"1000 × T_low ({_t_min_floor_tdms:.0f} s) pero "
                        f"estás por debajo del recomendado 2000 × T_low "
                        f"({_t_min_strict_tdms:.0f} s). Identificación de modos OK, "
                        "incertidumbre en damping moderada."
                    ),
                    severity="warning",
                )
            else:
                modal_status_banner(
                    title=(f"Record conforme a norma · {_record_dur:.0f} s ≥ "
                             f"{_t_min_strict_tdms:.0f} s para f_min = {oma_fmin:.1f} Hz"),
                    detail="Brincker & Ventura 2015 · ISO 18649 — cumplido.",
                    severity="ok",
                )
        with col_o2:
            _f_max_def = float(tdms_oma.sample_rate_hz / 2.0 * 0.9)
            oma_fmax = st.number_input("f máx (Hz)", value=min(500.0, _f_max_def),
                                         step=10.0, key="oma_fmax")
        with col_o3:
            oma_prom = st.number_input("Prominencia (dB)", value=8.0, step=1.0,
                                         key="oma_prom")
        with col_o4:
            oma_rpm = st.number_input("Running speed (rpm, opcional)",
                                        value=0, step=100, key="oma_rpm",
                                        help="Si se da, marca picos cercanos a "
                                             "1×, 2×, 3× como armónicos")

        if st.button("🌊 Ejecutar FDD + identificar modos", type="primary",
                       use_container_width=True, key="oma_run"):
            from core.modal.oma_engine import run_oma

            time_data = np.stack([ch.data for ch in tdms_oma.channels], axis=1)
            running_hz = (oma_rpm / 60.0) if oma_rpm > 0 else None
            nperseg = min(4096, time_data.shape[0] // 8)
            with st.spinner("Computando matriz PSD + SVD por frecuencia..."):
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
                (str(_n_natural), "Modos naturales", "✓ identificados", "green"),
                (str(_n_harm), "Armónicas", "× running speed", "red"),
                (str(_n_sp), "Espurios",
                 f"descartados MPC > 75%", "gray"),
                (f"{_avg_conf:.0f}%", "Confianza promedio",
                 "agregado por MPC + harm", "cyan"),
            ])

            modal_section_header(
                title="Densidad espectral — Singular Values",
                subtitle="Multi-SVD del PSD matrix · picos = modos naturales",
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
                        f"Modo {m.mode_number}<br>"
                        f"{m.natural_frequency_hz:.2f} Hz · ζ={m.damping_ratio_pct:.2f}%<br>"
                        f"complexity={m.complexity_pct:.1f}%<br>"
                        f"{m.classification}<extra></extra>"
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
                            f"Pico candidato<br>{_cf:.1f} Hz · prominencia "
                            f"{_cp:.1f} dB<br>(umbral: {float(oma_prom):.0f} dB)"
                            "<extra></extra>"),
                    ))
            except Exception:  # noqa: BLE001
                _det = {}

            fig_sv.update_layout(
                title=("Singular Values of Spectral Densities — "
                       "verde: modo natural (fn) · rojo: armónica (Nx) · gris: espurio"),
                xaxis_title="Frecuencia (Hz)",
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
                f"**Criterios de detección usados:** banda "
                f"**{float(oma_fmin):.0f}–{float(oma_fmax):.0f} Hz** · "
                f"prominencia mínima del pico **≥ {float(oma_prom):.0f} dB** sobre "
                "el ruido local · clasificación por MPC (natural < 40% · espurio "
                "> 75%) y cercanía a armónicas de la velocidad de giro. El eje Y "
                "es **10·log₁₀(valor singular)**: escala **relativa** en dB (no un "
                "valor físico absoluto). Los **círculos huecos grises** son picos "
                "candidatos que NO superaron el umbral de prominencia.")

            # Explicación específica cuando NO se identificó ningún modo natural
            if _n_natural == 0:
                _best_hz = _det.get("best_hz")
                _best_prom = float(_det.get("best_prom", 0.0) or 0.0)
                if _det.get("n_at_threshold", 0) == 0 and _best_hz is not None:
                    st.warning(
                        f"**Por qué 0 modos:** ningún pico supera la prominencia "
                        f"mínima de **{float(oma_prom):.0f} dB** en la banda "
                        f"{float(oma_fmin):.0f}–{float(oma_fmax):.0f} Hz. El pico "
                        f"más marcado es de solo **{_best_prom:.1f} dB** "
                        f"(~{_best_hz:.1f} Hz). Opciones: baja la prominencia a "
                        f"~{max(_best_prom * 0.8, 1):.0f} dB si crees que es un modo "
                        "real, ajusta la banda, o el registro no tiene modos claros "
                        "(excitación ambiental muy baja o sensor mal acoplado).",
                        icon="🔎")
                elif _best_hz is None:
                    st.warning(
                        "**Por qué 0 modos:** la curva SVD no presenta picos en la "
                        "banda elegida (parece ruido de banda ancha). Verifica que "
                        "el sensor mida vibración real y que la banda cubra el rango "
                        "de interés.", icon="🔎")

            st.markdown(f"### Tabla modal OMA — {len(fdd.modes)} candidatos")
            import pandas as pd

            def _note(m):
                if m.classification == "harmonic":
                    return f"{m.harmonic_order}×"
                if m.classification == "spurious":
                    return "espurio"
                if m.is_harmonic:
                    return f"{m.harmonic_order}×, fn"
                return "fn"

            df_oma = pd.DataFrame([
                {
                    "Modo": m.mode_number,
                    "Frecuencia (Hz)": round(m.natural_frequency_hz, 3),
                    "Damping (%)": round(m.damping_ratio_pct, 3),
                    "Complexity (%)": round(m.complexity_pct, 1),
                    "Nota": _note(m),
                    "Confianza": round(m.confidence, 2),
                }
                for m in fdd.modes
            ])

            def _style_row(row):
                if row["Nota"] == "espurio":
                    return ["background-color: #f3f4f6; color: #6b7280"] * len(row)
                if "×" in str(row["Nota"]) and "fn" not in str(row["Nota"]):
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
                    f"FDD result: {fdd.n_segments} segmentos Welch · "
                    f"nperseg = {fdd.nperseg} · "
                    f"Δf = {fdd.frequencies_hz[1]:.2f} Hz. "
                    "Mode shapes complejos disponibles en Tab Mode Shapes."
                ),
                norm_ref="ISO 20816 + ISO 7626-6 secc. 6.4",
                algorithm="FDD · Brincker, Zhang, Andersen 2001 · MPC Pappa & Eishan 1995",
            )

            # --- Diagnóstico automático del ensayo (interpretación + validez) -
            try:
                from core.modal.oma_interpret import interpret_fdd
                _diag = interpret_fdd(fdd)
                st.markdown("#### 🔎 Diagnóstico automático del ensayo")
                _sev_ui = {"ok": st.success, "info": st.info,
                           "warn": st.warning, "crit": st.error}
                for _o in _diag["observations"]:
                    _sev_ui.get(_o.severity, st.info)(
                        f"**{_o.title}** — {_o.detail}")
                st.markdown(f"**Conclusión técnica:** {_diag['conclusion']}")
            except Exception as _e_diag:  # noqa: BLE001
                st.caption(f"(No se pudo generar el diagnóstico: {_e_diag})")

            # --- Panel explicativo: cómo leer los resultados OMA -------------
            with st.expander("❓ ¿Cómo se leen estos resultados? "
                             "(escala, picos, damping, complejidad, confianza)",
                             expanded=False):
                st.markdown(
                    "**La gráfica (Singular Values del PSD, en dB)**\n\n"
                    "El eje vertical está en **decibeles: `dB = 10·log₁₀(valor "
                    "singular)`**. Es una escala **relativa y logarítmica**, no un "
                    "valor físico absoluto — por eso arranca cerca de 0 dB (el pico "
                    "de mayor energía) y baja a valores negativos (−40, −60 dB) en "
                    "las zonas de poca energía. Lo que importa **no es el número en "
                    "sí, sino la FORMA**: cada **pico local** marca una frecuencia "
                    "donde la estructura responde con mucha más energía → un posible "
                    "**modo natural**. Los valles entre picos son zonas sin "
                    "resonancia."
                )
                st.markdown(
                    "**Por qué un pico se marca como modo natural (🟢)**\n\n"
                    "El algoritmo **FDD** (Frequency Domain Decomposition, Brincker "
                    "2001) descompone la matriz de densidad espectral en valores "
                    "singulares. Un pico se clasifica como **modo natural** si: (1) "
                    "es un máximo local claro del 1er valor singular, (2) su forma "
                    "modal es **coherente** (alta pureza), y (3) **no** coincide con "
                    "un múltiplo de la velocidad de giro (eso sería armónica 🔴) ni "
                    "es un pico aislado sin forma física (espurio ⚪)."
                )
                st.markdown(
                    "**Los parámetros de la tabla**\n\n"
                    "- **Frecuencia (Hz):** dónde está el pico = frecuencia natural "
                    "del modo.\n"
                    "- **Damping (%):** amortiguamiento, estimado por el método de "
                    "**half-power (−3 dB)**: entre más ancho el pico, más "
                    "amortiguado. Valores típicos estructurales: 0.5–5%. Valores "
                    "altos (>10%) suelen indicar registro corto o modo mal definido.\n"
                    "- **Complejidad / MPC (%):** *Modal Phase Collinearity* (Pappa "
                    "1995). Un modo real tiene los puntos moviéndose en fase (0% = "
                    "modo normal, limpio). Complejidad alta = fases dispersas = "
                    "posible ruido o solape de modos.\n"
                    "- **Confianza:** índice combinado (nitidez del pico + coherencia "
                    "de forma + separación de armónicas). Es una guía relativa, no "
                    "una probabilidad."
                )
                st.info(
                    "Regla práctica: fíate primero de los **picos verdes bien "
                    "separados con damping 0.5–5% y complejidad baja**. Si un modo "
                    "sale con damping >10% o complejidad alta, suele mejorar "
                    "**recapturando con más tiempo de registro** (sube el T_low).",
                    icon="💡")


# ---------------------------------------------------------------------
# Tab 5 — Mode Shapes (visualización)
# ---------------------------------------------------------------------
if _active_modal_tab == "🎬 Mode Shapes 3D":
    modal_section_header(
        title="Visualización de Mode Shapes",
        subtitle="5 representaciones complementarias del mismo modo natural",
        norm_ref="ISO 7626-6 secc. 7.2",
        icon="🎬",
    )

    fdd = st.session_state.get("modal_oma_result")
    if fdd is None or not fdd.modes:
        modal_empty_state(
            icon="🎬",
            title="Sin modos identificados todavía",
            description=(
                "La visualización de mode shapes requiere haber ejecutado un "
                "análisis modal en el Tab OMA (o futuro Tab EMA). Una vez "
                "identificados los modos, regresa aquí para visualizar la "
                "forma modal de cada uno desde 5 perspectivas: bar chart, "
                "complexity polar, AutoMAC matrix, diagrama de Campbell y "
                "flechas 3D sobre el activo."
            ),
            cta_label="Ve al Tab OMA y ejecuta FDD",
            norm_ref="ISO 7626-6 secc. 7.2",
        )
    else:
        # ─── Selector global de modo (siempre arriba) ────────────────
        mode_options = {
            f"Modo {m.mode_number} · {m.natural_frequency_hz:.2f} Hz · "
            f"ζ={m.damping_ratio_pct:.2f}% · {m.classification}":
            m for m in fdd.modes
        }
        pick = st.selectbox(
            "Modo bajo análisis",
            list(mode_options.keys()),
            key="ms_pick",
            help=("Selecciona el modo natural a visualizar. Los expanders abajo "
                  "se actualizan automáticamente con el modo seleccionado."),
        )
        mode_sel = mode_options[pick]

        # KPI row del modo seleccionado
        _conf_color = {
            "natural": "green",
            "harmonic": "red",
            "spurious": "gray",
        }.get(mode_sel.classification, "navy")
        modal_kpi_row([
            (f"{mode_sel.natural_frequency_hz:.2f} Hz", "Frecuencia natural",
             "del modo identificado", "cyan"),
            (f"{mode_sel.damping_ratio_pct:.2f} %", "Damping ratio",
             "factor de amortiguamiento", "navy"),
            (f"{mode_sel.complexity_pct:.1f} %", "MPC complexity",
             "< 40% real · > 75% espurio", "amber"),
            (mode_sel.classification.upper(), "Clasificación",
             f"confianza {mode_sel.confidence*100:.0f}%", _conf_color),
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
            "📊  Bar chart 2D — Magnitud + fase del mode shape  ·  ISO 7626-6 secc. 7.2",
            expanded=True,
        ):
            modal_plot_caption(
                text=(
                    "Magnitud (normalizada) y fase de cada componente del mode "
                    "shape vector. Es la representación matemáticamente más "
                    "directa y válida bajo norma."
                ),
                norm_ref="ISO 7626-6 secc. 7.2",
                algorithm="Mode shape vector complejo del FDD",
            )
            fig_bar = build_bar_chart_mode_shape(
                mode_shape=mode_sel.mode_shape,
                channel_names=fdd.channel_names,
                mode_label=(f"Modo {mode_sel.mode_number} · "
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
                    "Cada flecha es un componente del mode shape en el plano "
                    "complejo. **Vectores colineales** (alineados en 0° o 180°) "
                    "= modo natural real. **Vectores dispersos** = modo complejo "
                    "o espurio. Visualización estándar de complejidad modal."
                ),
                norm_ref="ISO 7626-6 secc. 7.2",
                algorithm="Modal Phase Collinearity (Pappa & Eishan 1995)",
            )
            fig_pol = build_complexity_polar_plot(
                mode_shape=mode_sel.mode_shape,
                channel_names=fdd.channel_names,
                mode_label=(f"Modo {mode_sel.mode_number} · "
                              f"{mode_sel.natural_frequency_hz:.2f} Hz · "
                              f"MPC complexity = {mode_sel.complexity_pct:.1f}% · "
                              f"clase: {mode_sel.classification}"),
            )
            st.plotly_chart(fig_pol, use_container_width=True)

        # ═══════════════════════════════════════════════════════════════
        # EXPANDER 3 — AutoMAC Matrix
        # ═══════════════════════════════════════════════════════════════
        mac = compute_mac_matrix(fdd.modes)
        labels = [f"{m.natural_frequency_hz:.1f} Hz" for m in fdd.modes]
        redundants = detect_redundant_modes(fdd.modes, threshold=0.7)

        _redundant_warning = f"  ·  ⚠ {len(redundants)} pares redundantes" if redundants else ""

        with st.expander(
            f"🔗  AutoMAC Matrix — Correlación entre modos{_redundant_warning}  ·  "
            "ISO 7626-6 secc. 6.5 + API 684 secc. 1.6",
            expanded=False,
        ):
            modal_plot_caption(
                text=(
                    "Modal Assurance Criterion entre cada par de modos. "
                    "**Diagonal = 1** (siempre). **Off-diagonal > 0.7** "
                    "indica modos redundantes (mismo modo identificado 2 veces "
                    "— uno debería eliminarse). Matriz AutoMAC estándar."
                ),
                norm_ref="ISO 7626-6 secc. 6.5 · API 684 secc. 1.6",
                algorithm="AutoMAC matrix (Allemang & Brown 1982)",
            )
            view_3d = st.toggle("Vista 3D barras (profesional)",
                                  value=False, key="mac_3d_toggle")
            fig_mac = build_mac_matrix_plot(
                mac, labels, title="AutoMAC", use_3d=view_3d,
            )
            st.plotly_chart(fig_mac, use_container_width=True)

            # Diagnóstico de redundancia
            if redundants:
                modal_status_banner(
                    title=f"{len(redundants)} pares de modos linealmente dependientes",
                    detail=(
                        "MAC off-diagonal > 0.7. Pares detectados: " +
                        ", ".join([
                            f"Modo {i+1} ({fdd.modes[i].natural_frequency_hz:.1f} Hz) ↔ "
                            f"Modo {j+1} ({fdd.modes[j].natural_frequency_hz:.1f} Hz, "
                            f"MAC={mac_val:.2f})"
                            for i, j, mac_val in redundants[:5]
                        ]) + ". Considera eliminar el de menor confianza."
                    ),
                    severity="warning",
                )
            else:
                modal_status_banner(
                    title="Set modal limpio — todos los modos son linealmente independientes",
                    detail="Off-diagonal MAC < 0.7 en todos los pares.",
                    severity="ok",
                )

        # ═══════════════════════════════════════════════════════════════
        # EXPANDER 4 — Diagrama de Campbell
        # ═══════════════════════════════════════════════════════════════
        with st.expander(
            "📈  Diagrama de Campbell — Velocidades críticas  ·  API 684 secc. 1.6",
            expanded=False,
        ):
            modal_plot_caption(
                text=(
                    "Cruza los modos naturales identificados (líneas horizontales) "
                    "contra las armónicas de velocidad operativa "
                    "(líneas inclinadas 1×, 2×, ...). Las **X rojas** son "
                    "velocidades críticas — puntos donde una armónica excita "
                    "un modo natural y puede causar resonancia."
                ),
                norm_ref="API 684 secc. 1.6",
                algorithm="Diagrama de Campbell — rotor dynamics estándar",
            )

            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                camp_rpm_min = st.number_input("RPM mín", value=0, step=100,
                                                  key="camp_rpm_min")
            with col_c2:
                camp_rpm_max = st.number_input("RPM máx", value=4000, step=500,
                                                  key="camp_rpm_max")
            with col_c3:
                camp_op_rpm = st.number_input("Velocidad operativa (rpm)",
                                                value=3600, step=100,
                                                key="camp_op_rpm")

            natural_modes_for_camp = [m for m in fdd.modes
                                         if m.classification == "natural"]
            if natural_modes_for_camp:
                fig_camp, crit_speeds = build_campbell_diagram(
                    natural_frequencies_hz=[m.natural_frequency_hz
                                              for m in natural_modes_for_camp],
                    natural_freq_labels=[f"Modo {m.mode_number}"
                                           for m in natural_modes_for_camp],
                    rpm_min=float(camp_rpm_min),
                    rpm_max=float(camp_rpm_max),
                    operating_rpm=float(camp_op_rpm) if camp_op_rpm > 0 else None,
                    n_orders=6,
                    classification=[m.classification for m in natural_modes_for_camp],
                    title="Diagrama de Campbell",
                )
                st.plotly_chart(fig_camp, use_container_width=True)

                if crit_speeds:
                    import pandas as pd
                    df_crit = pd.DataFrame([
                        {
                            "Velocidad crítica (rpm)": round(rpm, 0),
                            "Modo": label,
                            "Frecuencia (Hz)": round(fn, 2),
                            "Orden": f"{order}× rpm",
                            "Estado": "⚠ DENTRO de rango operativo" if (
                                camp_op_rpm > 0
                                and abs(rpm - camp_op_rpm) / max(camp_op_rpm, 1) < 0.10
                            ) else "Fuera de rango operativo cercano",
                        }
                        for rpm, fn, order, label in crit_speeds
                    ])
                    _n_dentro = sum(1 for r in crit_speeds
                                       if camp_op_rpm > 0
                                       and abs(r[0] - camp_op_rpm) / max(camp_op_rpm, 1) < 0.10)
                    if _n_dentro > 0:
                        modal_status_banner(
                            title=f"{_n_dentro} velocidad(es) crítica(s) DENTRO del rango operativo",
                            detail=(
                                "El activo opera cerca de un cruce modo×armónica. "
                                "Riesgo de amplificación resonante — revisar API 618 "
                                "secc. 7.9.4.2.5.3.2 (separación ≥ 10%)."
                            ),
                            severity="fail",
                        )
                    st.markdown("**Velocidades críticas detectadas:**")
                    st.dataframe(df_crit, use_container_width=True, hide_index=True)
                else:
                    modal_status_banner(
                        title="Sin velocidades críticas detectadas",
                        detail=f"Ningún cruce modo natural ↔ armónica en la banda "
                                 f"{camp_rpm_min}-{camp_rpm_max} rpm.",
                        severity="ok",
                    )
            else:
                st.info("Sin modos naturales clasificados — no hay datos para Campbell.")

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
            _3d_status_label = (f"geometría editada · "
                                  f"{len(_geom_session.sensors)} sensores")
        else:
            # Fallback: Sensor Map del activo registrado
            if _adhoc_meta_for_3d:
                _3d_status_label = "no disponible · modo ad-hoc sin geometría"
            elif _inst_key_for_3d and _inst_key_for_3d != "(seleccionar)":
                try:
                    from core.instance_state import get_instance as _get_inst_3d
                    _inst_for_3d = _get_inst_3d(_inst_key_for_3d)
                    if _inst_for_3d:
                        _geom_source = "sensor_map"
                        _3d_status_label = "Sensor Map (legacy fallback)"
                    else:
                        _3d_status_label = "no disponible · activo sin sensores"
                except Exception:
                    _inst_for_3d = None
                    _3d_status_label = "no disponible · error cargando activo"
            else:
                _3d_status_label = "no disponible · sin activo en Setup"

        with st.expander(
            f"🌐  Flechas 3D sobre layout del activo · {_3d_status_label}",
            expanded=False,
        ):
            modal_plot_caption(
                text=(
                    "Visualización 3D del mode shape sobre la geometría real "
                    "del activo. Cada flecha indica la dirección de movimiento "
                    "de un sensor en el modo seleccionado. Verde = cofase · "
                    "rojo = anti-fase. Fuente preferida: editor de geometría "
                    "en Tab Setup. Fallback: Sensor Map del activo."
                ),
                norm_ref="ISO 7626-6 secc. 7.2",
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
                        "🎞 Animar",
                        value=False, key="modeshape_animate_toggle",
                        help=("Genera N frames para Play. ⚠ Aumenta uso de "
                              "memoria — actívalo solo cuando lo necesites."),
                    )
                with _ms_c2:
                    _show_arrows = st.toggle(
                        "Flechas DOF",
                        value=False, key="modeshape_arrows_toggle",
                        help="Muestra flechas Cone en cada sensor. Off = solo "
                             "heatmap del mesh (visualización Watermelon).",
                    )
                with _ms_c3:
                    # v3.31.199 HOTFIX: ghost OFF default — agrega 4 traces
                    # extra pesados que duplican consumo de memoria del browser
                    _show_ghost = st.toggle(
                        "Ghost original",
                        value=False, key="modeshape_ghost_toggle",
                        help="Overlay semi-transparente del estado sin deformar "
                             "para comparar. ⚠ Aumenta uso de memoria.",
                    )
                with _ms_c4:
                    _cmap = st.selectbox(
                        "Colormap",
                        ["RdBu_r", "RdYlBu_r", "Spectral_r", "Jet", "Viridis"],
                        index=0, key="modeshape_cmap",
                        help="Rojo cofase, azul anti-fase (RdBu_r). Alternativa: Jet.",
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
                                    text-transform: uppercase;">Modo identificado</div>
                        <div style="font-size: 26px; font-weight: 700;">
                            #{mode_sel.mode_number}</div>
                      </div>
                      <div style="border-left: 1px solid rgba(255,255,255,0.18);
                                  padding-left: 22px;">
                        <div style="font-size:10.5px; opacity:0.7; letter-spacing:1px;
                                    text-transform: uppercase;">Frecuencia</div>
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
                            complejidad</div>
                      </div>
                      <div style="border-left: 1px solid rgba(255,255,255,0.18);
                                  padding-left: 22px;">
                        <div style="font-size:10.5px; opacity:0.7; letter-spacing:1px;
                                    text-transform: uppercase;">Clasificación</div>
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
                        "label": "🔍 Lateral",
                        "eye": dict(x=0.0, y=2.4, z=0.4),
                        "up": dict(x=0, y=0, z=1),
                        "help": "Vista de lado A — flexión vertical clásica",
                    },
                    "lateral_opp": {
                        "label": "↩ Lat. opuesto",
                        "eye": dict(x=0.0, y=-2.4, z=0.4),
                        "up": dict(x=0, y=0, z=1),
                        "help": "Vista lateral espejo (lado opuesto)",
                    },
                    "frontal": {
                        "label": "👁 Frontal",
                        "eye": dict(x=2.4, y=0.0, z=0.4),
                        "up": dict(x=0, y=0, z=1),
                        "help": "Vista por el extremo libre del shaft",
                    },
                    "posterior": {
                        "label": "👀 Posterior",
                        "eye": dict(x=-2.4, y=0.0, z=0.4),
                        "up": dict(x=0, y=0, z=1),
                        "help": "Vista por el extremo opuesto del shaft",
                    },
                    "superior": {
                        "label": "⬇ Superior",
                        "eye": dict(x=0.0, y=0.0, z=2.5),
                        "up": dict(x=0, y=1, z=0),
                        "help": "Vista en planta desde arriba",
                    },
                    "isometrica": {
                        "label": "🔮 Isométrica",
                        "eye": dict(x=1.6, y=1.6, z=1.2),
                        "up": dict(x=0, y=0, z=1),
                        "help": "Vista 3D balanceada — clásica",
                    },
                    "diagonal": {
                        "label": "🎯 Vista 3/4",
                        "eye": dict(x=1.8, y=1.2, z=0.6),
                        "up": dict(x=0, y=0, z=1),
                        "help": "Diagonal frontal-lateral baja",
                    },
                }

                st.markdown("**Vista de cámara para el video**")
                st.caption(
                    "Selecciona el plano desde el que querés que se vea la "
                    "animación. El video correrá desde ese ángulo fijo."
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
                    mode_label=(f"Modo {mode_sel.mode_number} · "
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
                        return _adhoc_safe.get("equipment_name", "Activo ad-hoc")
                    if _inst_for_3d is not None and hasattr(_inst_for_3d, "display_name"):
                        return _inst_for_3d.display_name
                    return (_geom_session.name if _geom_session
                            else "Watermelon Modal")

                _exp_c1, _exp_c2, _exp_c3 = st.columns([1.2, 1.2, 2.6])
                with _exp_c1:
                    _gen_mp4 = st.button(
                        "🎥 Generar Video MP4",
                        key="modeshape_gen_mp4",
                        use_container_width=True,
                        type="primary",
                        help="MP4 H.264 (mejor calidad, ~2-5 MB, compatible "
                             "WhatsApp/iPhone/Android). Render ~30 s.",
                    )
                with _exp_c2:
                    _gen_gif = st.button(
                        "🖼 Generar GIF",
                        key="modeshape_gen_gif",
                        use_container_width=True,
                        help="GIF animado (alternativa universal, ~5-10 MB). "
                             "Render ~25 s.",
                    )

                # ---- MP4 ----
                if _gen_mp4:
                    from core.modal.geometry_3d import export_mode_shape_mp4
                    _asset_lbl = _resolve_asset_name()
                    _prog_bar = st.progress(0.0, text="Iniciando render…")
                    def _on_progress(idx, total, stage):
                        pct = idx / max(total, 1)
                        if stage == "encoding":
                            _prog_bar.progress(
                                1.0,
                                text=f"Encoding H.264 con ffmpeg… "
                                       f"({total} frames listos)")
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
                            f"✓ MP4 listo · "
                            f"{len(_mp4_bytes) / 1024:.0f} KB. "
                            "Click abajo para descargar."
                        )
                    except Exception as exc:  # noqa: BLE001
                        _prog_bar.empty()
                        import traceback
                        st.error(
                            f"**Error en MP4 export:** `{type(exc).__name__}: {exc}`"
                        )
                        with st.expander("Detalle técnico (traceback)"):
                            st.code(traceback.format_exc(), language="text")
                        st.info(
                            "Causas comunes: (1) imageio-ffmpeg no descargó el "
                            "binario, (2) kaleido falló al renderizar Plotly, "
                            "(3) Streamlit Cloud out-of-memory. "
                            "Prueba 'Generar GIF' como alternativa."
                        )

                # ---- GIF ----
                if _gen_gif:
                    from core.modal.geometry_3d import export_mode_shape_gif
                    _asset_lbl = _resolve_asset_name()
                    _prog_bar_g = st.progress(0.0, text="Iniciando render GIF…")
                    def _on_progress_g(idx, total, stage):
                        pct = idx / max(total, 1)
                        if stage == "encoding":
                            _prog_bar_g.progress(
                                1.0,
                                text=f"Ensamblando GIF… ({total} frames listos)")
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
                            f"✓ GIF listo · "
                            f"{len(_gif_bytes) / 1024:.0f} KB."
                        )
                    except Exception as exc:  # noqa: BLE001
                        _prog_bar_g.empty()
                        import traceback
                        st.error(f"**Error en GIF export:** `{type(exc).__name__}: {exc}`")
                        with st.expander("Detalle técnico"):
                            st.code(traceback.format_exc(), language="text")

                # ---- Download buttons (muestra el que esté listo) ----
                with _exp_c3:
                    if st.session_state.get("_modeshape_mp4"):
                        st.download_button(
                            "⬇ Descargar Video MP4 generado",
                            data=st.session_state["_modeshape_mp4"],
                            file_name=st.session_state.get(
                                "_modeshape_mp4_filename", "modeshape.mp4"),
                            mime="video/mp4",
                            use_container_width=True,
                            type="primary",
                        )
                    elif st.session_state.get("_modeshape_gif"):
                        st.download_button(
                            "⬇ Descargar GIF generado",
                            data=st.session_state["_modeshape_gif"],
                            file_name=st.session_state.get(
                                "_modeshape_gif_filename", "modeshape.gif"),
                            mime="image/gif",
                            use_container_width=True,
                        )

                # ===== Botón Enviar a Reporte Watermelon =====
                st.divider()
                modal_section_header(
                    title="Inyectar al Reporte Watermelon",
                    subtitle=(
                        "Genera snapshots PNG de todos los modos identificados + "
                        "AutoMAC + tabla resumen y los agrega a tu reporte "
                        "actual. Se renderizan al PDF estándar SIGA junto al "
                        "resto de figuras."
                    ),
                    norm_ref="ISO 7626-6 secc. 8 · Documentación modal",
                )

                _rep_c1, _rep_c2 = st.columns([2, 3])
                with _rep_c1:
                    _include_non_natural = st.toggle(
                        "Incluir harmonic/spurious (avanzado)",
                        value=False,
                        key=f"modal_report_inc_non_nat_{mode_sel.mode_number}",
                        help=(
                            "Por default solo se inyectan modos naturales "
                            "(physical modes del activo). Activar para "
                            "incluir también las armónicas de velocidad "
                            "(1×, 2×, ...) y modos espurios. Útil para "
                            "auditoría o reportes técnicos avanzados."
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
                        f"Se inyectarán **{_will_inject}** modos × 3 plots "
                        f"({_will_inject * 3} figuras de modos) + AutoMAC "
                        f"heatmap + tabla resumen = "
                        f"**{_will_inject * 3 + 2} items** al reporte."
                    )

                if st.button(
                    "📄 Enviar todos los modos al Reporte",
                    key=f"modal_send_report_{mode_sel.mode_number}",
                    type="primary",
                    use_container_width=True,
                    help="El reporte queda guardado en tu sesión. Visualízalo "
                         "y descarga el PDF desde la página Reports.",
                ):
                    from core.modal.modal_report import (
                        build_modal_report_items,
                        append_modal_items_to_report,
                    )
                    _rep_prog = st.progress(0.0, text="Generando snapshots…")

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
                            f"✓ **{_n_added} items** agregados al reporte. "
                            "Ve a **Reports** (sidebar) → "
                            "verás todas las figuras modales listadas para "
                            "incluir en el PDF final."
                        )
                    except Exception as _exc:  # noqa: BLE001
                        _rep_prog.empty()
                        import traceback as _tb
                        st.error(
                            f"Error generando snapshots: "
                            f"`{type(_exc).__name__}: {_exc}`"
                        )
                        with st.expander("Detalle técnico"):
                            st.code(_tb.format_exc(), language="text")

                # Diagnóstico de matching
                _ch_set = {n.strip().upper() for n in fdd.channel_names}
                _geom_set = {s.name.strip().upper() for s in _geom_session.sensors}
                _matched = _ch_set & _geom_set
                _missing = _ch_set - _geom_set
                if _missing:
                    modal_status_banner(
                        title=f"{len(_missing)} canal(es) sin sensor en geometría",
                        detail=(
                            f"Matchearon {len(_matched)}/{len(_ch_set)} canales "
                            f"con sensores de la geometría. Sin match: "
                            f"{', '.join(sorted(_missing)[:8])}"
                            f"{' …' if len(_missing) > 8 else ''}. "
                            "Agrega o renombra sensores en Tab Setup → "
                            "Geometría 3D para cubrir todos los canales."
                        ),
                        severity="warning",
                    )
                else:
                    modal_status_banner(
                        title=f"Match 100% · todos los {len(_ch_set)} canales en geometría",
                        detail="Las flechas representan fielmente el mode shape completo.",
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
                        mode_label=(f"Modo {mode_sel.mode_number} · "
                                      f"{mode_sel.natural_frequency_hz:.2f} Hz — "
                                      f"verde: cofase · rojo: anti-fase"),
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)
                    modal_status_banner(
                        title="Usando Sensor Map (sin geometría editada)",
                        detail=(
                            "Para una visualización más rica con bloques del "
                            "tren mecánico (motor, coupling, casing), construye "
                            "la geometría en Tab Setup → Geometría 3D."
                        ),
                        severity="info",
                    )
                else:
                    modal_status_banner(
                        title=f"Configuración 3D parcial — {len(sensors_3d)}/{len(fdd.channel_names)} canales",
                        detail=(
                            "Completa el expander 'Configuración modal' de cada "
                            "sensor en Machinery Library o construye la "
                            "geometría en Tab Setup → Geometría 3D."
                        ),
                        severity="warning",
                    )

            elif _adhoc_meta_for_3d:
                modal_status_banner(
                    title="Modo ad-hoc · construye la geometría 3D para activar las flechas",
                    detail=(
                        "Sin activo registrado, no hay Sensor Map. Pero "
                        "puedes ir a Tab Setup → 'Geometría 3D del activo', "
                        "aplicar un template (motor+compresor, turbina+gen, "
                        "bomba+motor) o construirla manualmente, y las flechas "
                        "3D se activan inmediatamente con el match por nombre "
                        "de canal. Los Niveles 1-4 ya cumplen ISO 7626-6 secc. 7.2."
                    ),
                    severity="info",
                )
            else:
                modal_status_banner(
                    title="Selecciona un activo o construye la geometría",
                    detail=(
                        "Para activar las flechas 3D: (a) selecciona un activo "
                        "registrado en Tab Setup con sensores 3D configurados, "
                        "o (b) construye la geometría manualmente en Tab Setup → "
                        "Geometría 3D del activo."
                    ),
                    severity="info",
                )

        # ═══════════════════════════════════════════════════════════════
        # ROADMAP nota — Mesh3D animado (Sprint próximo)
        # ═══════════════════════════════════════════════════════════════
        st.caption(
            "📅 **Roadmap próximo sprint:** Nivel 3 — Mesh3D animado con "
            "colormap profesional Watermelon. Los Niveles 1-2 actuales (bar chart + "
            "flechas 3D) ya cumplen ISO 7626-6 secc. 7.2 — el animated mesh es "
            "feature visual, no requisito normativo."
        )


# ---------------------------------------------------------------------
# Tab 6 — FEA Compare
# ---------------------------------------------------------------------
if _active_modal_tab == "🧮 FEA Compare":
    modal_section_header(
        title="Correlación EMA / OMA ↔ FEA",
        subtitle="Validación cruzada del modelo numérico contra resultados experimentales",
        norm_ref="API 684 secc. 1.6 · MAC ≥ 0.7 + Δf ≤ 10%",
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
        exp_label = "EMA · solo frecuencias (sin shapes multi-canal)"
        exp_freqs = [p.frequency_hz for p in peaks_for_fea]
        exp_shapes = []
        exp_channels = []
        exp_mode_labels = [f"P{i+1} ({p.frequency_hz:.1f} Hz)"
                            for i, p in enumerate(peaks_for_fea)]

    if exp_source is None:
        modal_empty_state(
            icon="🧮",
            title="Sin modos experimentales para comparar",
            description=(
                "Necesitas haber corrido al menos un análisis experimental "
                "antes de comparar contra FEA: corre el FDD en Tab OMA "
                "(preferido — entrega mode shapes multi-canal) o detecta "
                "picos en Tab EMA. Luego vuelve a este tab y sube tu JSON FEA."
            ),
            cta_label="Cambia a Tab OMA o Tab EMA",
            norm_ref="API 684 secc. 1.6",
        )
    else:
        col_src1, col_src2, col_src3 = st.columns(3)
        col_src1.metric("Fuente experimental", exp_label)
        col_src2.metric("Modos experimentales", len(exp_freqs))
        col_src3.metric("Canales", len(exp_channels) if exp_channels else "—")

        st.divider()
        st.markdown("**1 · Sube el archivo FEA**")
        st.caption(
            "Formato JSON Watermelon — exporta desde ANSYS/Nastran/Abaqus con "
            "`freq_hz` + `mode_shape` + `dof_names` que coincidan con los canales "
            "de tu identificación experimental. Soporta shapes reales o complejos."
        )

        col_up, col_tpl = st.columns([2, 1])
        with col_up:
            fea_up = st.file_uploader(
                "JSON FEA", type=["json"], key="fea_json_up",
                help="Roadmap próximo: parsers nativos .rst (Ansys), "
                     ".op2 (Nastran), .odb (Abaqus). Hoy solo JSON.",
            )
        with col_tpl:
            if exp_channels:
                tpl_json = json.dumps(example_fea_payload(exp_channels), indent=2)
            else:
                tpl_json = json.dumps(example_fea_payload(
                    [f"DOF{i+1}" for i in range(5)]), indent=2)
            st.download_button(
                "⬇ Template JSON",
                data=tpl_json,
                file_name="fea_template.json",
                mime="application/json",
                use_container_width=True,
                help="Descarga un template con tus canales experimentales "
                     "ya rellenados — solo edita freqs y shapes con tus "
                     "valores reales del FEA.",
            )

        fea_result = None
        if fea_up is not None:
            try:
                fea_result = load_fea_json(fea_up.getvalue().decode("utf-8"))
                st.session_state["fea_result"] = fea_result
            except Exception as exc:  # noqa: BLE001
                modal_status_banner(
                    title=f"Error al parsear el JSON FEA",
                    detail=str(exc),
                    severity="fail",
                )
                fea_result = None
        elif st.session_state.get("fea_result"):
            fea_result = st.session_state["fea_result"]
            st.caption(f"Usando FEA previamente cargado: **{fea_result.model_name}**")

        if fea_result is not None:
            st.divider()
            st.markdown("**2 · Resumen del modelo FEA**")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Modelo", fea_result.model_name[:30])
            mc2.metric("Software", fea_result.software[:20])
            mc3.metric("Modos FEA", fea_result.n_modes)
            _fmin, _fmax = fea_result.freq_range
            mc4.metric("Banda FEA", f"{_fmin:.1f} – {_fmax:.1f} Hz")

            st.divider()
            st.markdown("**3 · Configuración de correlación**")
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                mac_thr = st.number_input(
                    "Umbral MAC para validez",
                    value=0.70, min_value=0.5, max_value=0.95, step=0.05,
                    help="API 684 secc. 1.6 / Ewins: MAC ≥ 0.7 indica forma "
                         "correlacionada. Estándar industrial.",
                )
            with col_cfg2:
                freq_tol = st.number_input(
                    "Tolerancia Δf (%)",
                    value=10.0, min_value=2.0, max_value=30.0, step=1.0,
                    help="API 684 secc. 1.6: |Δf|/f_exp ≤ 10% es aceptable para "
                         "validación de rotor dynamics. < 5% es excelente.",
                )

            st.divider()
            st.markdown("**4 · Resultados**")

            # ----- Caso OMA: Cross-MAC completo -----
            if exp_source == "oma":
                # Validar que los DOF names del FEA cubren los canales exp
                _exp_set = {c.strip().upper() for c in exp_channels}
                _fea_set = {n.strip().upper() for n in fea_result.dof_names}
                _missing = _exp_set - _fea_set
                if _missing:
                    modal_status_banner(
                        title=f"FEA no cubre {len(_missing)} canal(es) experimental(es)",
                        detail=(
                            f"Canales sin DOF en el modelo FEA: "
                            f"{', '.join(sorted(_missing)[:10])}"
                            f"{' …' if len(_missing) > 10 else ''}. "
                            "Revisa tu export FEA — los DOF deben coincidir "
                            "con los canales medidos. Mientras tanto se "
                            "muestra solo la correlación de frecuencias."
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
                    st.markdown("**Pareo modo FEA ↔ experimental**")
                    import pandas as pd
                    status_label = {
                        "valid": "✓ Válido",
                        "shape_only": "≈ Forma OK · freq fuera",
                        "freq_only": "≈ Freq OK · forma débil",
                        "weak": "✗ Débil",
                        "no_match": "✗ Sin match",
                    }
                    df_pairs = pd.DataFrame([
                        {
                            "FEA": f"M{p['fea_mode']} ({p['fea_freq']:.2f} Hz)",
                            "Exp": (f"M{p['exp_mode']} ({p['exp_freq']:.2f} Hz)"
                                     if p["exp_mode"] else "—"),
                            "MAC": f"{p['mac']:.3f}",
                            "Δf (%)": (f"{p['delta_freq_pct']:.1f}"
                                        if p["delta_freq_pct"] is not None else "—"),
                            "Estado": status_label.get(p["status"], p["status"]),
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
                            title=f"Modelo FEA validado · {n_valid}/{n_total} modos "
                                    "con MAC ≥ umbral y Δf ≤ tolerancia",
                            detail=(
                                "Todos los modos FEA tienen contraparte experimental "
                                "con correlación válida. El modelo se considera apto "
                                "para predicción de rotor dynamics bajo API 684 secc. 1.6."
                            ),
                            severity="ok",
                        )
                    elif n_valid >= n_total * 0.7:
                        modal_status_banner(
                            title=f"Modelo FEA aceptable · {n_valid}/{n_total} modos válidos",
                            detail=(
                                "La mayoría de modos correlacionan, pero hay modos "
                                "individuales con forma débil o frecuencia fuera de "
                                "tolerancia. Revisar masas/rigideces locales del "
                                "modelo para los pares marcados 'shape_only' o "
                                "'freq_only'."
                            ),
                            severity="warning",
                        )
                    else:
                        modal_status_banner(
                            title=f"Modelo FEA requiere iteración · solo {n_valid}/{n_total} válidos",
                            detail=(
                                "Más del 30% de modos FEA no correlacionan. "
                                "Posibles causas: condiciones de borde mal definidas, "
                                "masas concentradas faltantes, malla insuficiente en "
                                "zonas críticas, o material properties incorrectas. "
                                "Re-iterar el modelo antes de usar para predicción."
                            ),
                            severity="fail",
                        )

            # ----- Caso EMA: solo frecuencias -----
            if exp_source == "ema_freq_only":
                modal_status_banner(
                    title="Comparación limitada a frecuencias (EMA sin mode shapes multi-canal)",
                    detail=(
                        "Los peaks del Tab EMA actual no incluyen mode shapes "
                        "multi-canal — solo frecuencias y damping. Para Cross-MAC "
                        "completo, corre el flujo FDD en Tab OMA o usa el sprint "
                        "futuro EMA-LSCF con mode shapes."
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
                                       "Exp": "—", "Δf (%)": "—", "Estado": "✗ Sin match"})
                        continue
                    used.add(best_j)
                    ok = best_delta <= float(freq_tol)
                    rows.append({
                        "FEA": f"M{fm.mode_number} ({fm.freq_hz:.2f} Hz)",
                        "Exp": f"P{best_j+1} ({exp_freqs[best_j]:.2f} Hz)",
                        "Δf (%)": f"{best_delta:.1f}",
                        "Estado": "✓ Freq OK" if ok else "✗ Freq fuera",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                              use_container_width=True)


# ---------------------------------------------------------------------
# Tab 7 — Reports (selector granular + auto-análisis + IA)
# ---------------------------------------------------------------------
if _active_modal_tab == "📊 Reports":
    modal_section_header(
        title="Reports — selector granular + análisis",
        subtitle=(
            "Elige qué figuras enviar al reporte SIGA. Análisis "
            "automático normativo + opcional IA interpretativa."
        ),
        norm_ref="ISO 7626-6 secc. 8 · Documentación modal",
        icon="📊",
    )

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
            title="Sin análisis modal cargado",
            description=(
                "Para usar esta sección necesitas haber corrido al menos un "
                "análisis: EMA en Tab EMA o FDD en Tab OMA. Después vuelve "
                "aquí y podrás seleccionar qué figuras enviar al reporte."
            ),
            cta_label="Cambia a Tab EMA o Tab OMA",
            norm_ref="ISO 7626-6 secc. 8",
        )
    else:
        # =================================================================
        # SECCION A — Selector granular de figuras
        # =================================================================
        st.markdown("### 📋 A · Selector de figuras a inyectar")
        st.caption(
            "Tilda qué figuras del análisis modal querés enviar al reporte. "
            "Cada figura se renderiza como PNG y se appendea al sistema "
            "Reports estándar (sale en el PDF SIGA final)."
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
                f"🌊 OMA · {len(_natural_modes_rep)} modos naturales "
                f"({_non_natural_count} no-naturales)",
                expanded=True,
            ):
                # Bulk controls
                _bulk_c1, _bulk_c2, _bulk_c3 = st.columns([1, 1, 2])
                with _bulk_c1:
                    if st.button("✓ Todos", key="rep_sel_all_oma",
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
                    if st.button("✗ Ninguno", key="rep_sel_none_oma",
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
                        "Incluir harmonic/spurious",
                        value=False, key="_rep_include_non_nat",
                        help="Off por default — solo modos físicos.",
                    )

                if _include_non_nat_rep:
                    _natural_modes_rep = list(_fdd_for_rep.modes)

                # Headers de tabla
                st.markdown(
                    "<div style='display:grid; grid-template-columns: 2fr 1fr 1fr 1fr; "
                    "gap:8px; padding:6px 0; border-bottom:1px solid #e5e7eb; "
                    "font-size:11px; color:#64748b; text-transform:uppercase;'>"
                    "<div>Modo</div><div style='text-align:center;'>3D snapshot</div>"
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
                # Globales
                st.markdown("**Bloques globales:**")
                _gc1, _gc2 = st.columns(2)
                with _gc1:
                    _sel_mac = st.checkbox(
                        "🔗 AutoMAC heatmap matrix",
                        value=True,
                        key="_rep_sel_automac",
                        help="MAC entre todos los modos · ISO 7626-6 secc. 6.5",
                    )
                    _all_selections["automac"] = _sel_mac
                with _gc2:
                    _sel_sum = st.checkbox(
                        "📑 Tabla resumen modal",
                        value=True,
                        key="_rep_sel_summary",
                        help="Tabla con freq/CPM/ζ/Q/MPC/clase de todos los modos",
                    )
                    _all_selections["summary"] = _sel_sum

        # Setup — geometría 3D estática + tablas
        if _has_geom:
            with st.expander(
                f"🛠 Setup · Geometría 3D ({len(_geom_for_rep.blocks)} bloques "
                f"+ {len(_geom_for_rep.sensors)} sensores)",
                expanded=False,
            ):
                _sel_setup_geom = st.checkbox(
                    "🌐 Snapshot 3D del activo (sin deformar)",
                    value=True, key="_rep_sel_setup_geom",
                )
                _sel_setup_blk = st.checkbox(
                    "📋 Tabla de bloques mecánicos",
                    value=True, key="_rep_sel_setup_blocks",
                )
                _sel_setup_sen = st.checkbox(
                    "📋 Tabla de sensores instrumentados",
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
                f"📥 Adquisición · {_n_ch} waveforms TDMS",
                expanded=False,
            ):
                _sel_acq_all = st.checkbox(
                    f"📈 Waveforms time-series ({_n_ch} canales)",
                    value=False, key="_rep_sel_acq_waveforms",
                    help=("Cada canal se renderiza como un plot tiempo-vs-amplitud "
                          "downsamplado a max 5000 puntos."),
                )
                _all_selections["acq_waveforms"] = _sel_acq_all

        # EMA — FRF + peaks
        if _has_ema or st.session_state.get("modal_frfs"):
            _frfs_count = len(st.session_state.get("modal_frfs", []))
            with st.expander(
                f"🔨 EMA · {_frfs_count} FRFs + "
                f"{len(_peaks_for_rep)} peaks identificados",
                expanded=False,
            ):
                _sel_ema_frf = st.checkbox(
                    "📊 FRF Bode con peaks marcados",
                    value=bool(_peaks_for_rep),
                    key="_rep_sel_ema_frf",
                    help="Magnitud (dB) vs frecuencia con peaks de modos.",
                )
                _sel_ema_tbl = st.checkbox(
                    "📋 Tabla de peaks EMA",
                    value=bool(_peaks_for_rep),
                    key="_rep_sel_ema_table",
                    help="Freq, damping, bandwidth, Q por peak.",
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
                    help=("Plot del 1st SV del cross-spectrum con los modos "
                          "identificados marcados (verde natural, amarillo "
                          "harmonic, rojo spurious)."),
                )
                _all_selections["oma_svd"] = _sel_oma_svd

        # FEA Compare — Cross-MAC + pareo
        if _has_fea:
            with st.expander(
                f"🧮 FEA Compare · {_fea_for_rep.n_modes} modos FEA",
                expanded=False,
            ):
                _sel_fea_mac = st.checkbox(
                    "🔥 Cross-MAC heatmap (FEA ↔ Experimental)",
                    value=True, key="_rep_sel_fea_mac",
                )
                _sel_fea_pair = st.checkbox(
                    "📋 Tabla de pareo de modos",
                    value=True, key="_rep_sel_fea_pairing",
                )
                _all_selections["fea_mac"] = _sel_fea_mac
                _all_selections["fea_pairing"] = _sel_fea_pair

        # ----- Resumen + Acción -----
        st.divider()
        _n_selected = sum(1 for v in _all_selections.values() if v)
        _ac1, _ac2 = st.columns([3, 2])
        with _ac1:
            st.metric("Figuras seleccionadas", _n_selected)
        with _ac2:
            _send_disabled = (_n_selected == 0)
            if st.button(
                f"📄 Inyectar {_n_selected} figuras al reporte",
                key="rep_send_selected",
                type="primary",
                use_container_width=True,
                disabled=_send_disabled,
            ):
                from core.modal.modal_report import (
                    build_modal_report_items,
                    append_modal_items_to_report,
                )
                _prog = st.progress(0.0, text="Generando snapshots…")

                def _cb_sel(idx, total, stage):
                    _prog.progress(
                        min(idx / max(total, 1), 1.0),
                        text=f"{stage} ({idx + 1}/{total})",
                    )

                try:
                    _asset_lbl_r = (
                        st.session_state.get("modal_adhoc_meta",
                                              {}).get("equipment_name",
                                                       "Activo ad-hoc")
                        if st.session_state.get("modal_adhoc_meta")
                        else st.session_state.get("modal_inst", "Activo")
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
                        f"✓ {_n_added} figuras agregadas al reporte. "
                        "Ve a **Reports** (sidebar) para revisar y "
                        "generar el PDF."
                    )
                except Exception as _exc:  # noqa: BLE001
                    _prog.empty()
                    import traceback as _tb
                    st.error(
                        f"Error: `{type(_exc).__name__}: {_exc}`"
                    )
                    with st.expander("Detalle técnico"):
                        st.code(_tb.format_exc(), language="text")

        # =================================================================
        # SECCION B — Auto-análisis normativo (rule-based)
        # =================================================================
        st.divider()
        st.markdown("### 🧠 B · Auto-análisis normativo")
        st.caption(
            "Análisis basado en reglas (ISO 7626 + API 684 + API 618). "
            "Sin IA — texto generado determinístico desde los datos modales."
        )

        if not _has_oma:
            modal_status_banner(
                title="Auto-análisis requiere modos OMA",
                detail="Corre el FDD en Tab OMA para activar esta sección.",
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
                    help="Para evaluar cruces con armónicas",
                )
            with _ab_c2:
                _ab_mac_thr = st.number_input(
                    "Umbral MAC redundancia",
                    value=0.70, min_value=0.5, max_value=0.95, step=0.05,
                    key="_modal_auto_mac_thr",
                )
            with _ab_c3:
                _ab_mpc_thr = st.number_input(
                    "Umbral MPC alto (%)",
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
            _kc1.metric("Conformes ✓", _n_ok)
            _kc2.metric("Advertencias ⚠", _n_warn)
            _kc3.metric("Críticos ✗", _n_fail)
            _kc4.metric("Informativos ℹ", _n_info)

            # Render findings
            for _f in _findings:
                modal_status_banner(
                    title=_f.title,
                    detail=(_f.text + (f" · Norma: {_f.norm_ref}"
                                          if _f.norm_ref else "")),
                    severity=_f.severity,
                )

            # Botón inyectar al reporte
            st.divider()
            if st.button(
                f"📄 Inyectar auto-análisis ({len(_findings)} hallazgos) "
                "al reporte",
                key="rep_send_auto_analysis",
                type="primary",
                use_container_width=True,
            ):
                from core.modal.modal_report import append_modal_items_to_report
                try:
                    _asset_lbl_b = (
                        st.session_state.get("modal_adhoc_meta",
                                              {}).get("equipment_name",
                                                       "Activo ad-hoc")
                        if st.session_state.get("modal_adhoc_meta")
                        else st.session_state.get("modal_inst", "Activo")
                    )
                    _auto_item = build_analysis_report_item(
                        findings=_findings,
                        asset_name=str(_asset_lbl_b),
                        method="OMA",
                    )
                    _n = append_modal_items_to_report([_auto_item])
                    st.success(
                        f"✓ Auto-análisis ({len(_findings)} hallazgos) "
                        "agregado al reporte como 1 figura PNG. "
                        "Ve a Reports (sidebar) para revisarlo."
                    )
                except Exception as _exc:  # noqa: BLE001
                    import traceback as _tb
                    st.error(
                        f"Error: `{type(_exc).__name__}: {_exc}`"
                    )
                    with st.expander("Detalle técnico"):
                        st.code(_tb.format_exc(), language="text")

        # =================================================================
        # SECCION C — Análisis IA interpretativo (paid, via Anthropic)
        # =================================================================
        st.divider()
        st.markdown("### 🤖 C · Análisis IA interpretativo")
        st.caption(
            "Narrativa interpretativa generada por Claude con contexto modal "
            "completo. Misma cuota AI que usan Spectrum/SCL/Polar/Waveform. "
            "Caching local 30 días para evitar costos repetidos."
        )

        from core.ai_diagnostic import (
            generate_ai_diagnostic,
            is_ai_available,
        )

        if not _has_oma:
            modal_status_banner(
                title="IA requiere modos OMA",
                detail="Corre el FDD en Tab OMA antes de generar el análisis IA.",
                severity="info",
            )
        elif not is_ai_available():
            modal_status_banner(
                title="IA no configurada",
                detail=(
                    "El módulo AI requiere `[anthropic] api_key` en "
                    "Streamlit secrets + paquete `anthropic` instalado. "
                    "Contacta al admin para habilitar."
                ),
                severity="warning",
            )
        else:
            _ac_c1, _ac_c2 = st.columns([2, 1])
            with _ac_c1:
                _ai_operator_notes = st.text_area(
                    "Notas del operador (contexto opcional)",
                    value="", height=100,
                    key="_modal_ai_operator_notes",
                    help=("Contexto que la IA debe considerar: condiciones "
                          "de operación, eventos recientes, sospechas, etc. "
                          "Mejora la calidad del análisis."),
                )
            with _ac_c2:
                _ai_use_cache = st.toggle(
                    "Usar caché si existe",
                    value=True, key="_modal_ai_use_cache",
                    help="30 días TTL · evita costos repetidos para misma data.",
                )
                _ai_running = st.number_input(
                    "Running rpm",
                    value=int(st.session_state.get(
                        "_modal_auto_running_rpm", 3600)),
                    min_value=100, max_value=20000, step=100,
                    key="_modal_ai_running_rpm",
                )

            if st.button(
                "🤖 Generar análisis IA modal",
                key="rep_gen_ai",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Llamando a Claude · puede tomar 15-30s…"):
                    from core.modal.modal_report import (
                        build_modal_ai_payload,
                    )
                    _asset_lbl_c = (
                        st.session_state.get("modal_adhoc_meta",
                                              {}).get("equipment_name",
                                                       "Activo ad-hoc")
                        if st.session_state.get("modal_adhoc_meta")
                        else st.session_state.get("modal_inst", "Activo")
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
                        title=f"Error IA: {_ai_res.get('error', 'unknown')}",
                        detail=_ai_res.get("markdown", ""),
                        severity="fail",
                    )
                else:
                    _ai_kc1, _ai_kc2, _ai_kc3 = st.columns(3)
                    _ai_kc1.metric(
                        "Modelo", _ai_res.get("model", "n/a"),
                    )
                    _ai_kc2.metric(
                        "Tokens in/out",
                        (f"{_ai_res.get('input_tokens', 0):,} / "
                          f"{_ai_res.get('output_tokens', 0):,}"),
                    )
                    _ai_kc3.metric(
                        "Cache hit",
                        "✓ Sí" if _ai_res.get("cached") else "✗ No (nuevo)",
                    )
                    st.divider()
                    st.markdown(_ai_res.get("markdown", ""))
                    st.divider()
                    if st.button(
                        "📄 Inyectar análisis IA al reporte",
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
                                            else "Activo",
                                method="OMA",
                            )
                            _n_ai = append_modal_items_to_report([_ai_item])
                            st.success(
                                f"✓ Análisis IA agregado al reporte como "
                                "1 figura PNG (header navy + texto en notes)."
                            )
                        except Exception as _exc:  # noqa: BLE001
                            import traceback as _tb
                            st.error(
                                f"Error: `{type(_exc).__name__}: {_exc}`"
                            )
                            with st.expander("Detalle técnico"):
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
