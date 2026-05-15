"""
pages/18_Modal_Analysis.py — Módulo de Análisis Modal Watermelon
=================================================================

Módulo nuevo para reemplazar Artemis Modal con stack nativo open-source.
Implementa EMA + OMA + comparación con FEA.

Tabs
----
1. Setup           — Geometría 3D + sensor → DOF mapping
2. Adquisición    — TDMS importer + NI-9234 live + Artemis legacy
3. EMA Processing — FRF + LSCF + stability diagram
4. OMA Processing — FDD + SSI
5. Mode Shapes 3D — Animación Plotly Mesh3d + export GIF/MP4
6. FEA Compare    — Importer FEA + MAC matrix

Marco normativo
---------------
· ISO 7626-1 a 7626-6 (EMA)
· ISO 20816 (OMA)
· API 684 (rotor dynamics validation)
· API 618 §7.9 (criterio separación modal)

Estado v3.31.151:
- Tab Adquisición: Legacy Artemis funcional (parsea .txt + plot Bode + tabla picos)
- Tab EMA: detección automática de modos con half-power damping
- Tab OMA / Mode Shapes 3D / FEA: scaffold (próximo sprint)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

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
    st.session_state["modal_frfs"] = []  # list[ArtemisFRF | FRFResult]


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
tab_setup, tab_acq, tab_ema, tab_oma, tab_3d, tab_fea = st.tabs([
    "🛠 Setup",
    "📥 Adquisición",
    "🔨 EMA",
    "🌊 OMA",
    "🎬 Mode Shapes 3D",
    "🧮 FEA Compare",
])


# ---------------------------------------------------------------------
# Tab 1 — Setup
# ---------------------------------------------------------------------
with tab_setup:
    modal_section_header(
        title="Configuración del ensayo modal",
        subtitle="Selecciona o registra el activo bajo análisis",
        norm_ref="ISO 7626-6 §6",
        icon="🛠",
    )

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
        st.markdown("**Datos del activo (análisis puntual)**")
        st.caption(
            "Completa la metadata mínima del equipo bajo análisis. "
            "Estos datos aparecerán en el Hero del módulo y en el reporte final."
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
            st.session_state["modal_adhoc_meta"] = {
                "tag": adhoc_tag,
                "client": adhoc_client,
                "station": adhoc_station,
                "model": adhoc_model,
                "notes": adhoc_notes,
            }
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

        col_sel, col_meta = st.columns([3, 2])
        with col_sel:
            picked_id = st.selectbox(
                "Activo bajo análisis",
                _options,
                format_func=lambda x: _labels_by_id.get(x, x) if x != "(seleccionar)" else x,
                key="modal_inst",
            )

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
                        title="Activo sin sensores configurados",
                        detail=(
                            "Este activo no tiene sensores definidos en Machinery "
                            "Library. Configura el sensor map antes de ejecutar "
                            "el análisis modal."
                        ),
                        severity="warning",
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
                        norm_ref="ISO 7626-6 §6.2",
                        icon="📡",
                    )
                    modal_kpi_row([
                        (str(len(sensors)), "Total sensores",
                         "registrados en Sensor Map", "navy"),
                        (str(n_accel), "Acelerómetros",
                         "Wilcoxon 100 mV/g típico", "cyan"),
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


# ---------------------------------------------------------------------
# Tab 2 — Adquisición
# ---------------------------------------------------------------------
with tab_acq:
    st.subheader("Adquisición de datos")
    st.caption("Tres rutas: NI-9234 live, importar TDMS pre-capturado, o legacy Artemis.")

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

    # -------- NI-9234 live --------
    if acq_mode.startswith("📡"):
        st.markdown("**Configuración de captura NI-9234**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Modo", ["EMA — Impact Hammer", "OMA — Continuous"], key="ni_mode")
            st.number_input("Sample rate (Hz)", value=5120, step=1024, key="ni_fs")
        with col2:
            st.number_input("Duración (s)", value=2.0, step=0.5, key="ni_dur")
            st.number_input("N° promedios (EMA)", value=5, step=1, key="ni_avg")
        with col3:
            st.markdown("**Canales activos**")
            for i in range(4):
                st.checkbox(f"Ch{i} habilitado", value=True, key=f"ni_ch_{i}_en")

        modal_status_banner(
            title="Próximo paso · Captura con la unidad NI cDAQ-9234",
            detail=(
                "Conecta la unidad de adquisición al activo bajo análisis siguiendo "
                "el procedimiento técnico SIGA. La configuración de canales y "
                "sample rate definida arriba se aplica al ejecutar la captura. "
                "Una vez generado el archivo de captura, súbelo en la opción "
                "'Importar archivo de captura existente'."
            ),
            severity="info",
        )

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
                _cmd_lines = [
                    f"--mode {st.session_state.get('ni_mode', 'EMA').lower().split()[0]}",
                    f"--fs {int(st.session_state.get('ni_fs', 5120))}",
                    f"--duration {float(st.session_state.get('ni_dur', 2.0))}",
                    "--output ./capture.tdms",
                ]
                if _user_role == "admin":
                    st.code(" \\\n    ".join(["watermelon-modal-capture"] + _cmd_lines),
                             language="bash")
                else:
                    st.code(" \\\n    ".join(_cmd_lines), language="text")

    # -------- TDMS existente --------
    elif acq_mode.startswith("📁"):
        st.markdown("**Subir archivo .tdms del NI-9234**")
        st.caption(
            "Carga el .tdms generado por el companion script o LabVIEW. "
            "Watermelon ejecuta automáticamente el checklist ISO 7626-5 sobre el ensayo."
        )
        tdms_up = st.file_uploader("Selecciona .tdms", type=["tdms"], key="tdms_up")

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
                help="ISO 7626-5 §7.4 — coherencia mínima en banda de interés. "
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

    # -------- Legacy Artemis --------
    else:
        st.markdown("**Importar exports legacy de Artemis Modal**")

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
            "Artemis NO guarda el eje en los .txt — requerido completar manualmente."
        )

        if uploaded and st.button("🔍 Procesar archivos Artemis",
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
with tab_ema:
    st.subheader("Análisis Modal Experimental")
    st.caption(
        "Identificación de parámetros modales (frecuencia natural, damping, mode shape) "
        "por método Circle-Fit Nyquist (Kennedy-Pancu) + half-power. "
        "Cumple ISO 7626-6 §6.3."
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
                "fallback (precisión 2-5%). Ambos cumplen ISO 7626-6 §6.3."
            )

        st.divider()

    # ─── Sección legacy Artemis (FRFs cargadas via .txt) ──────────────
    frfs = st.session_state.get("modal_frfs", [])
    if not frfs and tdms_frf is None:
        st.info("📭 No hay FRFs cargadas. Carga datos en el tab Adquisición primero "
                "(legacy Artemis .txt o TDMS del NI-9234).")
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
                    coherence=None,  # Artemis exports no incluyen coherencia
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
                "🔬 Damping calculado por método half-power (-3 dB · ISO 7626-6 §6.3.2). "
                "Diamantes ámbar marcan los modos detectados — hover para frecuencia + damping. "
                "Para mode shapes y curve fit LSCF, se requiere integración pyEMA (próximo sprint)."
            )


# ---------------------------------------------------------------------
# Tab 4 — OMA Processing (FDD)
# ---------------------------------------------------------------------
with tab_oma:
    st.subheader("Análisis Modal Operacional — FDD")
    st.caption(
        "Frequency Domain Decomposition (Brincker 2001) sobre datos operacionales "
        "del NI-9234. Sin necesidad de martillo. Cumple ISO 20816 + API 684."
    )

    tdms_oma = st.session_state.get("modal_tdms")
    if tdms_oma is None:
        modal_empty_state(
            icon="🌊",
            title="Sin datos operacionales cargados",
            description=(
                "El análisis OMA requiere un archivo .tdms con captura continua "
                "del NI-9234 — mínimo 60 segundos a velocidad constante (ISO 20816 + "
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
        if _record_dur < 30:
            st.warning(
                f"⚠ Duración del record {_record_dur:.0f} s. **OMA requiere ≥ 60s** "
                "para SVD estable y damping confiable (ISO 20816 + Brincker 2001). "
                "Resultados con < 30s deben tratarse como preliminares."
            )
        elif _record_dur < 60:
            st.info(
                f"ℹ Duración del record {_record_dur:.0f} s. Para resultados "
                "robustos OMA recomienda **60-300s**. Estás en zona aceptable "
                "pero damping puede tener varianza alta."
            )

        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
        with col_o1:
            oma_fmin = st.number_input("f mín (Hz)", value=5.0, step=1.0,
                                         key="oma_fmin")
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
            # de Artemis. SVD Line 1 (principal) + Line 2 + Line 3 si hay ≥ 3 canales.
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
                norm_ref="ISO 20816 + ISO 7626-6 §6.4",
                algorithm="FDD · Brincker, Zhang, Andersen 2001 · MPC Pappa & Eishan 1995",
            )


# ---------------------------------------------------------------------
# Tab 5 — Mode Shapes (visualización)
# ---------------------------------------------------------------------
with tab_3d:
    st.subheader("Visualización de Mode Shapes")
    st.caption(
        "Bar chart 2D (magnitud + fase) y flechas 3D sobre el layout del activo. "
        "Cumple ISO 7626-6 §7.2."
    )

    fdd = st.session_state.get("modal_oma_result")
    if fdd is None or not fdd.modes:
        st.info(
            "📭 Mode shapes disponibles solo desde resultados OMA. "
            "Ejecuta FDD en el Tab OMA primero."
        )
    else:
        # Selector de modo
        mode_options = {
            f"Modo {m.mode_number} · {m.natural_frequency_hz:.2f} Hz · "
            f"ζ={m.damping_ratio_pct:.2f}% "
            f"({'⚠ armónico' if m.is_harmonic else 'natural'})":
            m for m in fdd.modes
        }
        pick = st.selectbox("Seleccionar modo", list(mode_options.keys()),
                             key="ms_pick")
        mode_sel = mode_options[pick]

        from core.modal.modal_animator import (
            build_bar_chart_mode_shape,
            build_arrows_3d_wireframe,
            build_complexity_polar_plot,
        )

        # ─── Nivel 1: Bar chart ──────────────────────────────────────
        st.markdown(f"### Nivel 1 — Bar chart")
        fig_bar = build_bar_chart_mode_shape(
            mode_shape=mode_sel.mode_shape,
            channel_names=fdd.channel_names,
            mode_label=(f"Modo {mode_sel.mode_number} · "
                          f"{mode_sel.natural_frequency_hz:.2f} Hz · "
                          f"ζ = {mode_sel.damping_ratio_pct:.3f}%"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ─── Complexity Polar Plot — estilo Artemis Fig 10 ────────────
        st.markdown(f"### Complexity Polar Plot · ISO 7626-6 §7.2")
        st.caption(
            "Cada flecha es un componente del mode shape en el plano complejo. "
            "**Vectores colineales** (alineados en 0° o 180°) = modo natural real. "
            "**Vectores dispersos** = modo complejo o espurio."
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

        # ─── AutoMAC Matrix — estilo Artemis Fig 9 ─────────────────────
        st.markdown("### AutoMAC Matrix · ISO 7626-6 §6.5 + API 684 §1.6")
        st.caption(
            "Correlación entre modos identificados. Diagonal = 1 (siempre). "
            "**Off-diagonal > 0.7** indica modos redundantes (mismo modo "
            "identificado 2 veces — uno debería eliminarse)."
        )

        from core.modal.oma_engine import compute_mac_matrix, detect_redundant_modes
        mac = compute_mac_matrix(fdd.modes)
        labels = [f"{m.natural_frequency_hz:.1f} Hz" for m in fdd.modes]

        from core.modal.modal_animator import build_mac_matrix_plot
        col_v1, col_v2 = st.columns([3, 1])
        with col_v1:
            view_3d = st.toggle("Vista 3D (estilo Artemis)",
                                  value=False, key="mac_3d_toggle")
        fig_mac = build_mac_matrix_plot(
            mac, labels, title="AutoMAC", use_3d=view_3d,
        )
        st.plotly_chart(fig_mac, use_container_width=True)

        # ─── Diagrama de Campbell — API 684 §1.6 ──────────────────────
        st.markdown("### Diagrama de Campbell · API 684 §1.6")
        st.caption(
            "Cruza los modos naturales identificados (líneas horizontales) "
            "contra las armónicas de velocidad operativa (líneas inclinadas 1×, 2×, ...). "
            "Las **intersecciones marcadas con X roja** son velocidades críticas — "
            "puntos donde una armónica excita un modo natural."
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
                                            key="camp_op_rpm",
                                            help="Si está dentro del rango, se "
                                            "marca con vline ámbar")

        natural_modes_for_camp = [m for m in fdd.modes
                                     if m.classification == "natural"]
        if natural_modes_for_camp:
            from core.modal.modal_animator import build_campbell_diagram
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
                title="Diagrama de Campbell — Modos naturales vs Velocidad operativa",
            )
            st.plotly_chart(fig_camp, use_container_width=True)

            if crit_speeds:
                st.markdown("**Velocidades críticas detectadas:**")
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
                st.dataframe(df_crit, use_container_width=True, hide_index=True)
            else:
                st.success(
                    "✓ Ningún cruce modo natural ↔ armónica en la banda "
                    f"{camp_rpm_min}-{camp_rpm_max} rpm."
                )
        else:
            st.info("Sin modos naturales clasificados — no hay datos para Campbell.")

        st.divider()

        # Detección automática de redundantes
        redundants = detect_redundant_modes(fdd.modes, threshold=0.7)
        if redundants:
            st.warning(
                f"⚠ **{len(redundants)} pares de modos linealmente dependientes "
                f"(MAC > 0.7):**\n\n"
                + "\n".join([
                    f"- Modo {i+1} ({fdd.modes[i].natural_frequency_hz:.1f} Hz) ↔ "
                    f"Modo {j+1} ({fdd.modes[j].natural_frequency_hz:.1f} Hz) → "
                    f"MAC = {mac_val:.3f}"
                    for i, j, mac_val in redundants
                ])
                + "\n\nConsidera eliminar el de menor confianza."
            )
        else:
            st.success(
                "✓ Todos los modos identificados son linealmente independientes "
                "(off-diagonal MAC < 0.7). Set modal limpio."
            )

        # ─── Nivel 2: Flechas 3D — requiere position_3d en sensores ──
        st.markdown(f"### Nivel 2 — Flechas 3D sobre layout del activo")

        # Intentar leer posiciones 3D del sensor_map del activo
        # (Ciclo 23.148 — configuradas en wizard)
        try:
            from core.instance_state import get_instance
            # TODO: Activo seleccionado debe venir del Tab Setup; por ahora hardcode TES1
            inst = get_instance("tes1")
        except Exception:
            inst = None

        sensors_3d = []
        if inst is not None:
            for ch_name in fdd.channel_names:
                # Buscar sensor por plane_label
                match = None
                for s in (inst.sensors or []):
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
            # Todos los sensores tienen 3D → renderizar
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
        else:
            st.warning(
                f"⚠ Solo {len(sensors_3d)} de {len(fdd.channel_names)} canales tienen "
                f"configuración 3D (position_3d + dof_direction) en Sensor Map.\n\n"
                "Para activar el render 3D: completa el expander "
                "**⚙ Configuración modal** de cada sensor en el wizard de "
                "**Machinery Library** del activo. Sin esa data, el Nivel 1 (bar chart) "
                "ya es técnicamente válido bajo ISO 7626-6 §7.2."
            )

        st.caption(
            "🎬 Nivel 3 (Mesh3D animado con colormap estilo Artemis) en sprint próximo. "
            "Niveles 1-2 actuales ya cumplen ISO 7626-6 §7.2 — indican magnitud + fase "
            "+ pattern espacial."
        )


# ---------------------------------------------------------------------
# Tab 6 — FEA Compare
# ---------------------------------------------------------------------
with tab_fea:
    modal_section_header(
        title="Correlación EMA / OMA ↔ FEA",
        subtitle="Validación cruzada del modelo numérico contra resultados experimentales",
        norm_ref="API 684 §1.6",
        icon="🧮",
    )

    modal_empty_state(
        icon="🧮",
        title="Importer FEA en desarrollo",
        description=(
            "Cuando esté listo: subes el archivo de modos FEA (Ansys .rst, "
            "Nastran .op2, Abaqus .odb o JSON con freq + mode shape) y "
            "Watermelon calcula la matriz Cross-MAC entre tus modos "
            "experimentales (EMA u OMA) y los modos del modelo. Resultado: "
            "tabla de correlación + recomendación de iteración del modelo."
        ),
        cta_label="Disponible próximo sprint",
        norm_ref="API 684 §1.6 — Rotor dynamics validation",
    )


# =====================================================================
# Footer normativo permanente
# =====================================================================
modal_footer_norms(
    active_norms=[
        "ISO 7626-1..6",
        "ISO 20816",
        "API 684",
        "API 618 §7.9.4.2.5.3.2",
    ],
    algorithms=[
        "Circle-Fit Nyquist (Kennedy-Pancu 1947)",
        "FDD (Brincker, Zhang, Andersen 2001)",
        "Modal Complexity MPC (Pappa & Eishan 1995)",
        "AutoMAC (ISO 7626-6 §6.5)",
        "Half-power method (ISO 7626-6 §6.3.2)",
        "Diagrama de Campbell (API 684 §1.6)",
    ],
    # version=None → lee VERSION dinámicamente vía core.version
)
