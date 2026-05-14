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

Estado: SCAFFOLD (v3.31.146)
- Tabs renderizan estructura pero acciones requieren implementación
- Importadores y motores en core/modal/ pendientes (NotImplementedError)
"""

from __future__ import annotations

import streamlit as st

# Auth + layout reusables del sistema Watermelon
from core.auth import require_login, render_user_menu, is_page_allowed_for_role
from core.page_header import page_header


# =====================================================================
# Setup de página
# =====================================================================
st.set_page_config(
    page_title="Watermelon System | Modal Analysis",
    page_icon="🌐",
    layout="wide",
)

# Auth (solo admin/specialist por ahora — cliente NO ve modal)
session = require_login()
_my_email = (session.get("email") or "").lower()
_my_role = (session.get("role") or "").lower()

if not is_page_allowed_for_role("pages/18_Modal_Analysis.py", _my_role):
    st.error("Tu rol no tiene acceso a este módulo.")
    st.stop()

render_user_menu(current_page="pages/18_Modal_Analysis.py")

# Header internacional
page_header(
    icon="🌐",
    title="Modal Analysis",
    subtitle="EMA · OMA · FEA — Análisis modal experimental y operacional bajo ISO 7626 / ISO 20816 / API 684",
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
        "Sprint pendiente: editor de tabla para asignar a cada sensor del Sensor Map "
        "su posición 3D `[x, y, z]` y dirección DOF `[dx, dy, dz]`. "
        "Inferencia automática desde icon_anchor 2D ya planificada."
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
    )

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
            "Esta UI valida config y dispara el script via `scripts/ni_companion/capture.py`."
        )
        st.button("▶ Iniciar captura", type="primary", disabled=True, help="Sprint NI-DAQ")

    elif acq_mode.startswith("📁"):
        st.markdown("**Subir archivo .tdms del NI**")
        st.file_uploader("Selecciona .tdms", type=["tdms"], key="tdms_up")

    else:
        st.markdown("**Importar exports legacy de Artemis Modal**")
        st.file_uploader("Subir archivos .txt", type=["txt"], accept_multiple_files=True, key="art_up")
        col_a, col_b = st.columns(2)
        with col_a:
            st.number_input("Sample rate original (Hz)", value=2560, step=100, key="art_fs")
        with col_b:
            st.number_input("Bandwidth (Hz)", value=1280, step=100, key="art_bw")
        st.caption(
            "El eje de frecuencia se reconstruye como Δf = bandwidth / (N_bins - 1). "
            "Artemis NO guarda el eje en los .txt — requerido completar manualmente."
        )


# ---------------------------------------------------------------------
# Tab 3 — EMA Processing
# ---------------------------------------------------------------------
with tab_ema:
    st.subheader("Análisis Modal Experimental (LSCF)")
    st.caption(
        "Curve fitting LSCF sobre FRFs medidas con martillo modal. "
        "Cumple ISO 7626-5 (martillo) e ISO 7626-6 (curve fit)."
    )

    st.info(
        "Sprint pendiente: integración con `pyEMA`. La UI mostrará:\n"
        "1. Selección de FRFs a procesar (desde adquisición previa)\n"
        "2. Slider de model order (típico 30-80)\n"
        "3. Stability diagram interactivo\n"
        "4. Tabla modal: frecuencia · damping · complejidad por modo identificado"
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
        "**Requisitos de datos:** records de 60-300 segundos continuos a velocidad constante. "
        "Los CSVs de proximidad existentes (532 ms) NO son suficientes — necesita modo "
        "'long acquisition' del NI-9234."
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
# Footer normativo (siempre visible)
# =====================================================================
st.divider()
st.caption(
    "**Marco normativo · ISO 7626-1..6 · ISO 20816 · API 684 · API 618 §7.9.4.2.5.3.2** — "
    "Módulo en fase scaffolding · v3.31.146"
)
