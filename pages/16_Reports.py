from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.auth import require_login, render_user_menu

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Watermelon System | Reports",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

require_login()
render_user_menu()

# =========================================================
# STYLES
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    .wm-hero {
        border-radius: 24px;
        padding: 1.3rem 1.5rem;
        background: linear-gradient(135deg, #0f172a 0%, #13213b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 40px rgba(15,23,42,0.18);
        margin-bottom: 1rem;
    }

    .wm-kicker {
        color: #66c2ff;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .wm-title {
        color: white;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        margin-top: 0.25rem;
    }

    .wm-copy {
        color: #b9c6d8;
        font-size: 0.97rem;
        margin-top: 0.35rem;
    }

    .wm-card {
        background: white;
        border: 1px solid #e5edf5;
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(15,23,42,0.05);
        margin-bottom: 1rem;
    }

    .wm-section-title {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.65rem;
    }

    .wm-muted {
        color: #64748b;
        font-size: 0.92rem;
    }

    .wm-report-cover {
        background: #ffffff;
        border: 1px solid #dbe5ef;
        border-radius: 22px;
        padding: 2rem 2.2rem;
        box-shadow: 0 12px 28px rgba(15,23,42,0.05);
    }

    .wm-cover-top {
        color: #1696ff;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .wm-cover-main {
        margin-top: 1.4rem;
        color: #111827;
        font-size: 2rem;
        font-weight: 900;
        line-height: 1.08;
    }

    .wm-cover-sub {
        margin-top: 0.35rem;
        color: #334155;
        font-size: 1.1rem;
        font-weight: 700;
    }

    .wm-cover-meta {
        margin-top: 1.4rem;
        color: #475569;
        font-size: 0.96rem;
        line-height: 1.75;
    }

    .wm-chip {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: #eef6ff;
        border: 1px solid #d3e8ff;
        color: #167dd8;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .wm-outline-item {
        background: #f8fbff;
        border: 1px solid #dbe7f3;
        border-radius: 16px;
        padding: 0.7rem 0.8rem;
        margin-bottom: 0.55rem;
    }

    .wm-outline-title {
        color: #0f172a;
        font-size: 0.92rem;
        font-weight: 800;
    }

    .wm-outline-sub {
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATA MODELS
# =========================================================
@dataclass
class AssetItem:
    asset_id: str
    category: str
    source_name: str
    title: str
    dataframe: Optional[pd.DataFrame] = None


# =========================================================
# SESSION
# =========================================================
if "report_outline" not in st.session_state:
    st.session_state["report_outline"] = []

if "report_recommendations" not in st.session_state:
    st.session_state["report_recommendations"] = (
        "1. \n"
        "2. \n"
        "3. "
    )

if "report_service_development" not in st.session_state:
    st.session_state["report_service_development"] = (
        "Describa aquí el desarrollo del servicio, las condiciones observadas y el análisis técnico."
    )


# =========================================================
# HELPERS
# =========================================================
def safe_dataframe(obj: Any) -> Optional[pd.DataFrame]:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None
    if isinstance(obj, list):
        try:
            return pd.DataFrame(obj)
        except Exception:
            return None
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def detect_waveform_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = {c.lower(): c for c in df.columns}
    time_candidates = ["time", "tiempo", "t", "x"]
    amp_candidates = ["amplitude", "amplitud", "value", "valor", "y", "signal", "senal", "señal"]

    time_col = None
    amp_col = None

    for c in time_candidates:
        if c in cols:
            time_col = cols[c]
            break

    for c in amp_candidates:
        if c in cols and cols[c] != time_col:
            amp_col = cols[c]
            break

    if time_col and amp_col:
        return time_col, amp_col

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]

    return None


def detect_spectrum_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = {c.lower(): c for c in df.columns}
    freq_candidates = ["frequency", "frecuencia", "freq", "cpm", "hz", "x"]
    amp_candidates = ["amplitude", "amplitud", "value", "valor", "y", "overall"]

    freq_col = None
    amp_col = None

    for c in freq_candidates:
        if c in cols:
            freq_col = cols[c]
            break

    for c in amp_candidates:
        if c in cols and cols[c] != freq_col:
            amp_col = cols[c]
            break

    if freq_col and amp_col:
        return freq_col, amp_col

    return None


def detect_orbit_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    cols = {c.lower(): c for c in df.columns}
    x_candidates = ["x", "x_axis", "orbit_x", "horizontal"]
    y_candidates = ["y", "y_axis", "orbit_y", "vertical"]

    x_col = None
    y_col = None

    for c in x_candidates:
        if c in cols:
            x_col = cols[c]
            break

    for c in y_candidates:
        if c in cols and cols[c] != x_col:
            y_col = cols[c]
            break

    if x_col and y_col:
        return x_col, y_col

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]

    return None


def discover_signal_assets() -> List[AssetItem]:
    signals = st.session_state.get("signals", {})
    items: List[AssetItem] = []

    if not isinstance(signals, dict):
        return items

    for key, value in signals.items():
        df = safe_dataframe(value)
        if df is None or df.empty:
            continue

        df = normalize_columns(df)
        source_name = str(key)

        # Tabular list always allowed
        items.append(
            AssetItem(
                asset_id=f"table::{source_name}",
                category="Tabular List",
                source_name=source_name,
                title=f"Tabla - {source_name}",
                dataframe=df,
            )
        )

        if detect_waveform_columns(df):
            items.append(
                AssetItem(
                    asset_id=f"waveform::{source_name}",
                    category="Formas de Onda",
                    source_name=source_name,
                    title=f"Forma de onda - {source_name}",
                    dataframe=df,
                )
            )

        if detect_spectrum_columns(df):
            items.append(
                AssetItem(
                    asset_id=f"spectrum::{source_name}",
                    category="Espectros",
                    source_name=source_name,
                    title=f"Espectro - {source_name}",
                    dataframe=df,
                )
            )

        if detect_orbit_columns(df):
            items.append(
                AssetItem(
                    asset_id=f"orbit::{source_name}",
                    category="Órbitas",
                    source_name=source_name,
                    title=f"Órbita - {source_name}",
                    dataframe=df,
                )
            )

    return items


def get_asset_map(assets: List[AssetItem]) -> Dict[str, AssetItem]:
    return {a.asset_id: a for a in assets}


def add_to_outline(asset_id: str) -> None:
    outline = st.session_state["report_outline"]
    outline.append(asset_id)
    st.session_state["report_outline"] = outline


def move_outline_item(idx: int, direction: int) -> None:
    outline = st.session_state["report_outline"]
    new_idx = idx + direction
    if 0 <= idx < len(outline) and 0 <= new_idx < len(outline):
        outline[idx], outline[new_idx] = outline[new_idx], outline[idx]
        st.session_state["report_outline"] = outline


def remove_outline_item(idx: int) -> None:
    outline = st.session_state["report_outline"]
    if 0 <= idx < len(outline):
        outline.pop(idx)
        st.session_state["report_outline"] = outline


def render_waveform(asset: AssetItem) -> None:
    df = asset.dataframe
    cols = detect_waveform_columns(df)
    if not cols:
        st.warning(f"No se pudieron detectar columnas válidas para forma de onda en {asset.source_name}.")
        return

    x_col, y_col = cols
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=asset.source_name,
        )
    )
    fig.update_layout(
        title=asset.title,
        height=380,
        template="plotly_white",
        margin=dict(l=30, r=20, t=60, b=30),
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_spectrum(asset: AssetItem) -> None:
    df = asset.dataframe
    cols = detect_spectrum_columns(df)
    if not cols:
        st.warning(f"No se pudieron detectar columnas válidas para espectro en {asset.source_name}.")
        return

    x_col, y_col = cols
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=asset.source_name,
        )
    )
    fig.update_layout(
        title=asset.title,
        height=380,
        template="plotly_white",
        margin=dict(l=30, r=20, t=60, b=30),
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_orbit(asset: AssetItem) -> None:
    df = asset.dataframe
    cols = detect_orbit_columns(df)
    if not cols:
        st.warning(f"No se pudieron detectar columnas válidas para órbita en {asset.source_name}.")
        return

    x_col, y_col = cols
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=asset.source_name,
        )
    )
    fig.update_layout(
        title=asset.title,
        height=420,
        template="plotly_white",
        margin=dict(l=30, r=20, t=60, b=30),
        xaxis_title=x_col,
        yaxis_title=y_col,
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tabular(asset: AssetItem) -> None:
    st.markdown(f"**{asset.title}**")
    st.dataframe(asset.dataframe, use_container_width=True, hide_index=True)


# =========================================================
# DISCOVER DATA
# =========================================================
assets = discover_signal_assets()
asset_map = get_asset_map(assets)

assets_by_category: Dict[str, List[AssetItem]] = {
    "Tabular List": [],
    "Formas de Onda": [],
    "Espectros": [],
    "Órbitas": [],
}
for asset in assets:
    assets_by_category.setdefault(asset.category, []).append(asset)

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="wm-hero">
        <div class="wm-kicker">Module 16</div>
        <div class="wm-title">Reports</div>
        <div class="wm-copy">
            Constructor de reportes técnicos para Watermelon System.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 📑 Reports")
    st.markdown("Configura el contenido y el orden del reporte.")

    st.markdown("---")
    st.markdown("### Datos generales")
    report_title = st.text_input("Título del reporte", value="REPORTE DE MONITOREO EN LÍNEA")
    report_system = st.text_input("Sistema", value="Watermelon System")
    unit_name = st.text_input("Unidad / Equipo", value="TURBOGENERADOR TES1")
    equipment_model = st.text_input("Modelo", value="LM6000")
    plant_location = st.text_input("Ubicación", value="VILLAVICENCIO")
    company_name = st.text_input("Cliente / Planta", value="TERMOSURIA")
    prepared_by = st.text_input("Preparado por", value="Ing. Responsable")
    reviewed_by = st.text_input("Revisado por", value="Revisor Técnico")
    evaluated_period = st.text_input("Periodo evaluado", value="9/03/2026 al 16/03/2026")
    report_date = st.text_input("Fecha de reporte", value="17 de marzo de 2026")
    consecutive = st.text_input("Consecutivo", value="SIGA-REP-TEC-WM-001")

    st.markdown("---")
    st.markdown("### Texto técnico")
    st.session_state["report_recommendations"] = st.text_area(
        "Recomendaciones",
        value=st.session_state["report_recommendations"],
        height=200,
    )
    st.session_state["report_service_development"] = st.text_area(
        "Desarrollo del servicio",
        value=st.session_state["report_service_development"],
        height=220,
    )

    st.markdown("---")
    st.markdown("### Contenido disponible")

    if not assets:
        st.info("No se detectaron DataFrames válidos en st.session_state['signals'].")
    else:
        for category in ["Tabular List", "Formas de Onda", "Espectros", "Órbitas"]:
            st.markdown(f"**{category}**")
            category_items = assets_by_category.get(category, [])
            if not category_items:
                st.caption("Sin datos detectados.")
                continue

            labels = [f"{item.source_name}" for item in category_items]
            selected_label = st.selectbox(
                f"Seleccionar {category}",
                labels,
                key=f"select_{category}",
            )

            selected_item = next((i for i in category_items if i.source_name == selected_label), None)
            if selected_item and st.button(f"Agregar {category}", key=f"add_{category}"):
                add_to_outline(selected_item.asset_id)
                st.rerun()

    st.markdown("---")
    st.markdown("### Orden del reporte")

    if not st.session_state["report_outline"]:
        st.caption("Aún no has agregado bloques al reporte.")
    else:
        for idx, asset_id in enumerate(st.session_state["report_outline"]):
            asset = asset_map.get(asset_id)
            if not asset:
                continue

            st.markdown(
                f"""
                <div class="wm-outline-item">
                    <div class="wm-outline-title">{idx + 1}. {asset.title}</div>
                    <div class="wm-outline-sub">{asset.category}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("↑", key=f"up_{idx}", use_container_width=True):
                    move_outline_item(idx, -1)
                    st.rerun()
            with c2:
                if st.button("↓", key=f"down_{idx}", use_container_width=True):
                    move_outline_item(idx, 1)
                    st.rerun()
            with c3:
                if st.button("✕", key=f"remove_{idx}", use_container_width=True):
                    remove_outline_item(idx)
                    st.rerun()

# =========================================================
# MAIN LAYOUT
# =========================================================
left_col, right_col = st.columns([1.1, 1.9], gap="large")

with left_col:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-section-title">Resumen del reporte</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='wm-muted'><b>Título:</b> {report_title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='wm-muted'><b>Unidad:</b> {unit_name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='wm-muted'><b>Modelo:</b> {equipment_model}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='wm-muted'><b>Ubicación:</b> {plant_location}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='wm-muted'><b>Cliente:</b> {company_name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='wm-muted'><b>Consecutivo:</b> {consecutive}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-section-title">Bloques seleccionados</div>', unsafe_allow_html=True)
    if not st.session_state["report_outline"]:
        st.info("Agrega bloques desde la barra lateral izquierda.")
    else:
        for idx, asset_id in enumerate(st.session_state["report_outline"]):
            asset = asset_map.get(asset_id)
            if asset:
                st.markdown(
                    f"<div class='wm-chip'>{idx + 1}. {asset.title}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="wm-report-cover">', unsafe_allow_html=True)
    st.markdown(f"<div class='wm-cover-top'>{report_system}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='wm-cover-main'>{report_title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='wm-cover-sub'>{unit_name}</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="wm-cover-meta">
            <b>Modelo:</b> {equipment_model}<br>
            <b>Ubicación:</b> {plant_location}<br>
            <b>Cliente:</b> {company_name}<br><br>
            <b>Preparado por:</b> {prepared_by}<br>
            <b>Revisado por:</b> {reviewed_by}<br><br>
            <b>Periodo evaluado:</b> {evaluated_period}<br>
            <b>Fecha de reporte:</b> {report_date}<br>
            <b>Consecutivo:</b> {consecutive}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-section-title">1. Recomendaciones</div>', unsafe_allow_html=True)
    st.text(st.session_state["report_recommendations"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-section-title">2. Desarrollo del servicio</div>', unsafe_allow_html=True)
    st.write(st.session_state["report_service_development"])
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state["report_outline"]:
        st.markdown("## Contenido seleccionado")
        for idx, asset_id in enumerate(st.session_state["report_outline"], start=1):
            asset = asset_map.get(asset_id)
            if not asset:
                continue

            st.markdown('<div class="wm-card">', unsafe_allow_html=True)
            st.markdown(
                f"<div class='wm-section-title'>{idx + 2}. {asset.title}</div>",
                unsafe_allow_html=True,
            )

            if asset.category == "Tabular List":
                render_tabular(asset)
            elif asset.category == "Formas de Onda":
                render_waveform(asset)
            elif asset.category == "Espectros":
                render_spectrum(asset)
            elif asset.category == "Órbitas":
                render_orbit(asset)

            st.markdown("</div>", unsafe_allow_html=True)
