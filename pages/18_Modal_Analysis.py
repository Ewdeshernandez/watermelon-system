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

page_header(
    "Modal Analysis",
    "EMA · OMA · FEA — Análisis modal experimental y operacional bajo ISO 7626 / ISO 20816 / API 684",
)


# =====================================================================
# Session state — guardar FRFs cargados entre reruns
# =====================================================================
if "modal_frfs" not in st.session_state:
    st.session_state["modal_frfs"] = []  # list[ArtemisFRF | FRFResult]


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
    st.subheader("Configuración del ensayo modal")
    st.caption(
        "Define la geometría 3D del activo y el mapeo de sensores a DOFs. "
        "Cumple ISO 7626-6 §6 — documentación de DOFs y orientación espacial."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Activo bajo análisis**")
        st.selectbox("Instancia", ["(seleccionar)", "tes1", "tes3"], key="modal_inst")
        st.caption("Reusa la jerarquía cliente → estación → activo de Machinery Library")

    with col2:
        st.markdown("**Geometría 3D**")
        st.button("📂 Cargar geometría JSON", disabled=True, help="Sprint próximo")
        st.button("➕ Crear geometría desde Machine Map", disabled=True, help="Sprint próximo")

    st.divider()
    st.markdown("**Mapeo de sensores → DOFs 3D**")
    st.info(
        "Configura sensitivities + posición 3D + DOF en el wizard de Machinery Library "
        "(expander '⚙ Configuración modal' por sensor). Aquí solo se visualiza el resumen."
    )


# ---------------------------------------------------------------------
# Tab 2 — Adquisición
# ---------------------------------------------------------------------
with tab_acq:
    st.subheader("Adquisición de datos")
    st.caption("Tres rutas: NI-9234 live, importar TDMS pre-capturado, o legacy Artemis.")

    acq_mode = st.radio(
        "Origen de datos",
        ["📡 Captura live NI-9234", "📁 Importar .tdms existente", "🔄 Legacy Artemis (.txt)"],
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

        st.warning(
            "⚠ Captura live requiere companion script local con `nidaqmx` instalado. "
            "Esta UI valida config y dispara el script via `scripts/ni_companion/capture.py`. "
            "Streamlit Cloud no tiene hardware NI conectado."
        )
        st.code(
            "python scripts/ni_companion/capture.py \\\n"
            "    --mode oma --output ./run1.tdms \\\n"
            "    --fs 10240 --duration 120 \\\n"
            "    --channels 1YA:0:IEPE:100 \\\n"
            "    --channels 2YA:1:IEPE:100",
            language="bash",
        )

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

            # Detección automática de martillo
            hammer = tdms.detect_hammer_channel()
            responses = tdms.response_channels()

            if hammer is None:
                st.warning(
                    "⚠ No se detectó canal de martillo automáticamente. "
                    "El validador ISO 7626-5 requiere un input claro."
                )
                st.stop()

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
                st.stop()
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

            # ============================================================
            # CHECKLIST ISO 7626-5 (banner superior)
            # ============================================================
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

            # ============================================================
            # PANEL DE 6 PLOTS — Input / Output / FRF / Coherencia
            # ============================================================
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

            # Input tiempo (row 1, col 1)
            fig.add_trace(go.Scatter(
                x=hammer.time_s, y=hammer.data, mode="lines",
                name="Input time", line=dict(color="#0F1E3D", width=1),
                showlegend=False,
            ), row=1, col=1)
            fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
            fig.update_yaxes(title_text=f"{hammer.units}", row=1, col=1)

            # Input espectro (row 1, col 2)
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

            # Response tiempo (row 2, col 1)
            fig.add_trace(go.Scatter(
                x=resp.time_s, y=resp.data, mode="lines",
                name="Response time", line=dict(color="#1AAEE5", width=1),
                showlegend=False,
            ), row=2, col=1)
            fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
            fig.update_yaxes(title_text=f"{resp.units}", row=2, col=1)

            # Response espectro (row 2, col 2)
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

            # FRF magnitud (row 3, col 1)
            mag_db = 20 * np.log10(np.maximum(frf.magnitude, 1e-30))
            fig.add_trace(go.Scatter(
                x=frf.frequencies_hz, y=mag_db, mode="lines",
                name="FRF Mag", line=dict(color="#0F7FB0", width=1.5),
                showlegend=False,
            ), row=3, col=1)
            fig.update_xaxes(title_text="Frecuencia (Hz)", row=3, col=1)
            fig.update_yaxes(title_text="Magnitud (dB)", row=3, col=1)

            # Coherencia (row 3, col 2)
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
        "Detección automática de modos por half-power method "
        "(ISO 7626-6 §6.3.2). Aplica a FRFs cargadas en el tab Adquisición."
    )

    frfs = st.session_state.get("modal_frfs", [])
    if not frfs:
        st.info("📭 No hay FRFs cargadas. Carga archivos en el tab Adquisición primero.")
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

            # Plot con picos anotados
            primary = next((f for f in frfs if f.is_complex_frf), frfs[0])
            mag_db = 20.0 * np.log10(np.maximum(primary.magnitude_linear(), 1e-30))
            fig_peaks = go.Figure()
            fig_peaks.add_trace(go.Scatter(
                x=primary.frequencies_hz, y=mag_db, mode="lines",
                name="FRF", line=dict(width=1.2, color="#1AAEE5"),
            ))
            for i, p in enumerate(peaks):
                fig_peaks.add_vline(
                    x=p.frequency_hz,
                    line=dict(color="#D89B22", width=1, dash="dash"),
                    annotation_text=f"Modo {i+1}<br>{p.frequency_hz:.1f} Hz<br>ζ={p.damping_ratio_pct:.2f}%",
                    annotation_position="top",
                    annotation_font_size=10,
                )
            fig_peaks.update_layout(
                title="FRF con modos identificados",
                xaxis_title="Frecuencia (Hz)",
                yaxis_title="Magnitud (dB)",
                height=420,
                margin=dict(l=50, r=20, t=60, b=40),
                template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(fig_peaks, use_container_width=True)

            st.caption(
                "🔬 Damping calculado por método half-power (-3 dB). "
                "Para mode shapes y curve fit LSCF, se requiere integración pyEMA "
                "(próximo sprint)."
            )


# ---------------------------------------------------------------------
# Tab 4 — OMA Processing
# ---------------------------------------------------------------------
with tab_oma:
    st.subheader("Análisis Modal Operacional (FDD + SSI)")
    st.caption(
        "Identificación modal a partir de datos operacionales sin excitación controlada. "
        "Cumple ISO 20816 + API 684."
    )

    st.info(
        "Sprint pendiente: integración con `PyOMA2`. Algoritmos disponibles:\n"
        "- FDD (Frequency Domain Decomposition) — primera pasada rápida\n"
        "- SSI-COV / SSI-DATA — más preciso para damping\n\n"
        "**Requisitos de datos:** records de 60-300 segundos continuos a velocidad constante "
        "capturados via NI-9234 en modo OMA (companion script con --mode oma)."
    )


# ---------------------------------------------------------------------
# Tab 5 — Mode Shapes 3D
# ---------------------------------------------------------------------
with tab_3d:
    st.subheader("Visualización 3D de Mode Shapes")
    st.caption("Animación Plotly Mesh3d con colormap — equivalente Artemis.")

    st.info(
        "Sprint pendiente: render del mode shape seleccionado de la tabla modal.\n"
        "Tres niveles de fidelidad disponibles:\n"
        "- Nivel 1: Bar chart 2D (más simple)\n"
        "- Nivel 2: Wireframe con flechas vectoriales\n"
        "- Nivel 3: Mesh3D animado con colormap (V1 target)\n\n"
        "Export como GIF/MP4 para inclusión en reportes."
    )


# ---------------------------------------------------------------------
# Tab 6 — FEA Compare
# ---------------------------------------------------------------------
with tab_fea:
    st.subheader("Correlación EMA/OMA ↔ FEA")
    st.caption("MAC matrix + recomendación de iteración del modelo.")

    st.info(
        "Sprint largo plazo: importer de modelos FEA (Ansys, Nastran, Abaqus output) y "
        "cálculo de MAC (Modal Assurance Criterion) entre modos experimentales y FEA.\n\n"
        "Sin norma única definida — se usa como input para iteración del modelo FEA "
        "buscando coincidencia con resultados experimentales."
    )


# =====================================================================
# Footer normativo
# =====================================================================
st.divider()
st.caption(
    "**Marco normativo · ISO 7626-1..6 · ISO 20816 · API 684 · API 618 §7.9.4.2.5.3.2** — "
    "Módulo Modal Analysis · v3.31.151"
)
