from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Safe auth / sidebar integration
# ============================================================

def _safe_auth_bootstrap() -> None:
    """
    Intenta integrarse con core.auth sin romper si cambió el nombre
    de funciones entre módulos.
    """
    try:
        import core.auth as auth  # type: ignore

        # Patrones comunes usados en el proyecto
        for fn_name in [
            "require_auth",
            "require_login",
            "check_auth",
            "protect_page",
            "ensure_authenticated",
        ]:
            fn = getattr(auth, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    break
                except TypeError:
                    pass

        for fn_name in [
            "render_sidebar",
            "build_sidebar",
            "show_sidebar",
            "sidebar",
        ]:
            fn = getattr(auth, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    break
                except TypeError:
                    pass
    except Exception:
        pass


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Watermelon System | Polar Plot",
    page_icon="🌀",
    layout="wide",
)

_safe_auth_bootstrap()


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .wm-header {
        background: linear-gradient(135deg, #111827 0%, #0f172a 60%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 20px 24px 18px 24px;
        margin-bottom: 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    }
    .wm-kicker {
        color: #93c5fd;
        font-size: 0.88rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .wm-title {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0 0 6px 0;
    }
    .wm-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin: 0;
    }
    .wm-card {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 14px 16px 10px 16px;
        margin-bottom: 18px;
        box-shadow: 0 12px 24px rgba(0,0,0,0.14);
    }
    .wm-card h3 {
        color: white;
        margin: 0 0 4px 0;
        font-size: 1.12rem;
    }
    .wm-card p {
        color: #cbd5e1;
        margin: 0;
        font-size: 0.92rem;
    }
    .wm-metric-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-top: 10px;
    }
    .wm-metric {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 10px 12px;
    }
    .wm-metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-bottom: 4px;
    }
    .wm-metric-value {
        color: white;
        font-size: 1rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding-left: 14px;
        padding-right: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="wm-header">
        <div class="wm-kicker">Watermelon System · Rotordynamics</div>
        <div class="wm-title">Polar Plot</div>
        <p class="wm-subtitle">
            Visualización polar premium con múltiples gráficos independientes en una sola página,
            lectura robusta de CSV, exportación HD y agrupación automática por canal / punto / etiqueta.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

@dataclass
class PolarColumns:
    amplitude: str
    phase: str
    speed: Optional[str]
    group: Optional[str]


def _normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace(".", " ")
    )


def _read_csv_robust(file) -> pd.DataFrame:
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="ignore")

    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            text = raw.decode(enc)
            break
        except Exception:
            text = None
    if text is None:
        raise ValueError("No se pudo decodificar el archivo CSV.")

    sample = text[:10000]
    candidates = [",", ";", "\t", "|"]
    counts = {sep: sample.count(sep) for sep in candidates}
    sep = max(counts, key=counts.get)

    df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_best_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    scored: List[Tuple[int, str]] = []
    for col in columns:
        norm = _normalize_name(col)
        score = 0
        for i, p in enumerate(patterns):
            if p in norm:
                score += max(1, len(patterns) - i)
        if score > 0:
            scored.append((score, col))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def _to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = (
        s.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9\.\-+eE]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _infer_columns(df: pd.DataFrame) -> PolarColumns:
    cols = list(df.columns)

    amplitude = _find_best_column(
        cols,
        [
            "amplitude",
            "amp",
            "overall",
            "direct",
            "1x",
            "mils",
            "mm s",
            "um",
            "displacement",
            "vibration",
        ],
    )
    phase = _find_best_column(
        cols,
        [
            "phase",
            "angulo",
            "angle",
            "deg",
            "degrees",
        ],
    )
    speed = _find_best_column(
        cols,
        [
            "speed",
            "rpm",
            "velocity",
            "turning",
            "rotational",
        ],
    )
    group = _find_best_column(
        cols,
        [
            "point name",
            "point",
            "channel",
            "tag",
            "probe",
            "measurement",
            "name",
            "label",
            "axis",
            "trace",
            "location",
        ],
    )

    if amplitude is None or phase is None:
        raise ValueError(
            "No se pudieron inferir columnas de amplitud y fase. "
            "El CSV debe contener al menos amplitude/direct y phase."
        )

    return PolarColumns(amplitude=amplitude, phase=phase, speed=speed, group=group)


def _prepare_dataframe(df: pd.DataFrame, cols: PolarColumns) -> pd.DataFrame:
    work = df.copy()

    work["__amplitude__"] = _to_numeric_series(work[cols.amplitude])
    work["__phase__"] = _to_numeric_series(work[cols.phase])

    if cols.speed and cols.speed in work.columns:
        work["__speed__"] = _to_numeric_series(work[cols.speed])
    else:
        work["__speed__"] = np.arange(1, len(work) + 1, dtype=float)

    if cols.group and cols.group in work.columns:
        work["__group__"] = work[cols.group].astype(str).fillna("Polar")
    else:
        work["__group__"] = "Polar"

    work = work.dropna(subset=["__amplitude__", "__phase__", "__speed__"]).copy()

    # Normalización prudente de fase
    work["__phase__"] = np.mod(work["__phase__"], 360.0)

    return work


def _detect_critical_speeds(speed: np.ndarray, amp: np.ndarray, max_peaks: int = 3) -> List[float]:
    """
    Heurística prudente: detecta máximos locales amplios.
    No pretende sustituir API 684; solo da referencias visuales.
    """
    if len(speed) < 5 or len(amp) < 5:
        return []

    amp = np.asarray(amp, dtype=float)
    speed = np.asarray(speed, dtype=float)

    # suavizado simple
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    smooth = np.convolve(amp, kernel, mode="same")

    peaks: List[Tuple[float, float]] = []
    threshold = np.nanpercentile(smooth, 70)

    for i in range(2, len(smooth) - 2):
        if smooth[i] >= threshold and smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            peaks.append((smooth[i], speed[i]))

    peaks.sort(reverse=True, key=lambda x: x[0])

    selected: List[float] = []
    for _, s in peaks:
        if all(abs(s - prev) > max(150, 0.05 * max(speed)) for prev in selected):
            selected.append(float(s))
        if len(selected) >= max_peaks:
            break
    return selected


def _polar_annotation_text(
    title: str,
    amp: np.ndarray,
    speed: np.ndarray,
    crit: List[float],
    amp_unit: str,
) -> str:
    amp_min = float(np.nanmin(amp)) if len(amp) else 0.0
    amp_max = float(np.nanmax(amp)) if len(amp) else 0.0
    rpm_min = float(np.nanmin(speed)) if len(speed) else 0.0
    rpm_max = float(np.nanmax(speed)) if len(speed) else 0.0
    crit_txt = ", ".join(f"{c:,.0f}" for c in crit) if crit else "N/A"

    return (
        f"<b>{title}</b><br>"
        f"Amp max: {amp_max:,.3f} {amp_unit}<br>"
        f"Amp min: {amp_min:,.3f} {amp_unit}<br>"
        f"RPM range: {rpm_min:,.0f} - {rpm_max:,.0f}<br>"
        f"Crit. heurísticos: {crit_txt}"
    )


def _build_polar_figure(
    title: str,
    data: pd.DataFrame,
    amp_unit: str,
    speed_unit: str,
    clockwise: bool,
    phase_offset: float,
    label_every: int,
    show_markers: bool,
    show_speed_labels: bool,
    max_critical: int,
) -> go.Figure:
    d = data.sort_values("__speed__").copy()

    amp = d["__amplitude__"].to_numpy(dtype=float)
    phase = d["__phase__"].to_numpy(dtype=float)
    speed = d["__speed__"].to_numpy(dtype=float)

    if clockwise:
        theta = np.mod(phase_offset - phase, 360.0)
        rotation = 90
        direction = "clockwise"
    else:
        theta = np.mod(phase + phase_offset, 360.0)
        rotation = 90
        direction = "counterclockwise"

    crit = _detect_critical_speeds(speed, amp, max_peaks=max_critical)

    text = [
        f"{title}<br>"
        f"Amp: {a:,.4f} {amp_unit}<br>"
        f"Phase: {p:,.2f}°<br>"
        f"Speed: {s:,.0f} {speed_unit}"
        for a, p, s in zip(amp, phase, speed)
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=amp,
            theta=theta,
            mode="lines+markers" if show_markers else "lines",
            line=dict(width=3, color="#38bdf8"),
            marker=dict(
                size=8 if show_markers else 0,
                color=speed,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(
                    title=speed_unit,
                    thickness=14,
                    len=0.75,
                    y=0.5,
                ),
                line=dict(width=0.5, color="rgba(255,255,255,0.35)"),
            ) if show_markers else None,
            name=title,
            text=text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Punto de inicio
    if len(amp) > 0:
        fig.add_trace(
            go.Scatterpolar(
                r=[amp[0]],
                theta=[theta[0]],
                mode="markers+text",
                marker=dict(size=14, color="#22c55e", symbol="diamond"),
                text=["START"],
                textposition="top center",
                name="Start",
                hovertemplate=f"START<br>Amp: {amp[0]:,.4f} {amp_unit}<br>Phase: {phase[0]:,.2f}°<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=[amp[-1]],
                theta=[theta[-1]],
                mode="markers+text",
                marker=dict(size=14, color="#ef4444", symbol="diamond"),
                text=["END"],
                textposition="bottom center",
                name="End",
                hovertemplate=f"END<br>Amp: {amp[-1]:,.4f} {amp_unit}<br>Phase: {phase[-1]:,.2f}°<extra></extra>",
            )
        )

    # Labels RPM discretos
    if show_speed_labels and len(amp) > 0 and label_every > 0:
        idx = list(range(0, len(d), label_every))
        if idx[-1] != len(d) - 1:
            idx.append(len(d) - 1)

        fig.add_trace(
            go.Scatterpolar(
                r=amp[idx],
                theta=theta[idx],
                mode="text",
                text=[f"{speed[i]:,.0f}" for i in idx],
                textfont=dict(size=10, color="white"),
                name="RPM Labels",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Anillos de referencia
    rmax = float(np.nanmax(amp)) if len(amp) else 1.0
    if not np.isfinite(rmax) or rmax <= 0:
        rmax = 1.0
    rmax *= 1.15

    annotation_text = _polar_annotation_text(title, amp, speed, crit, amp_unit)

    fig.update_layout(
        template="plotly_dark",
        height=760,
        margin=dict(l=50, r=90, t=70, b=40),
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            y=0.97,
            font=dict(size=22),
        ),
        polar=dict(
            bgcolor="#0b1220",
            radialaxis=dict(
                title=amp_unit,
                angle=90,
                gridcolor="rgba(255,255,255,0.10)",
                linecolor="rgba(255,255,255,0.18)",
                tickcolor="rgba(255,255,255,0.35)",
                showline=True,
                linewidth=1,
                ticks="outside",
                range=[0, rmax],
            ),
            angularaxis=dict(
                direction=direction,
                rotation=rotation,
                gridcolor="rgba(255,255,255,0.10)",
                linecolor="rgba(255,255,255,0.18)",
                tickcolor="rgba(255,255,255,0.35)",
                showline=True,
                linewidth=1,
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        annotations=[
            dict(
                x=1.02,
                y=0.92,
                xref="paper",
                yref="paper",
                align="left",
                text=annotation_text,
                showarrow=False,
                bordercolor="rgba(255,255,255,0.10)",
                borderwidth=1,
                borderpad=10,
                bgcolor="rgba(15,23,42,0.92)",
                font=dict(size=12, color="white"),
            )
        ],
    )

    # Marcadores heurísticos de critical speed
    if crit:
        for c in crit:
            idx = int(np.argmin(np.abs(speed - c)))
            fig.add_trace(
                go.Scatterpolar(
                    r=[amp[idx]],
                    theta=[theta[idx]],
                    mode="markers+text",
                    marker=dict(size=12, color="#f59e0b", symbol="x"),
                    text=[f"CS ~ {speed[idx]:,.0f}"],
                    textposition="middle right",
                    name=f"Critical ~ {speed[idx]:,.0f} {speed_unit}",
                    hovertemplate=(
                        f"Critical heuristic<br>"
                        f"Speed: {speed[idx]:,.0f} {speed_unit}<br>"
                        f"Amp: {amp[idx]:,.4f} {amp_unit}<br>"
                        f"Phase: {phase[idx]:,.2f}°<extra></extra>"
                    ),
                )
            )

    return fig


def _candidate_group_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    candidates = []
    for c in cols:
        norm = _normalize_name(c)
        if any(k in norm for k in [
            "point", "channel", "tag", "probe", "name", "label", "measurement",
            "axis", "trace", "location", "sensor"
        ]):
            candidates.append(c)

    nunique_candidates = []
    for c in candidates:
        try:
            n = df[c].astype(str).nunique(dropna=True)
            if 1 < n <= 100:
                nunique_candidates.append(c)
        except Exception:
            pass

    return nunique_candidates


def _infer_unit_from_name(name: str, default: str) -> str:
    n = _normalize_name(name)
    if "mils" in n:
        return "mils"
    if "um" in n or "µm" in n:
        return "µm"
    if "mm/s" in n or "mm s" in n:
        return "mm/s"
    if "in/s" in n or "in s" in n:
        return "in/s"
    if "rpm" in n:
        return "RPM"
    if "deg" in n or "phase" in n or "angulo" in n or "angle" in n:
        return "deg"
    return default


# ============================================================
# Main UI
# ============================================================

with st.sidebar:
    st.markdown("### Polar Controls")

    clockwise = st.toggle("Clockwise machine rotation", value=True)
    phase_offset = st.number_input("Phase offset (deg)", value=0.0, step=1.0, format="%.1f")
    show_markers = st.toggle("Show markers", value=True)
    show_speed_labels = st.toggle("Show RPM labels", value=True)
    label_every = st.number_input("Label every N points", min_value=1, max_value=200, value=10, step=1)
    max_critical = st.number_input("Max critical heuristics", min_value=0, max_value=5, value=3, step=1)


tab_upload, tab_preview = st.tabs(["CSV Input", "Data Preview"])

with tab_upload:
    uploaded = st.file_uploader(
        "Upload Polar CSV",
        type=["csv", "txt"],
        accept_multiple_files=False,
        help="Carga un CSV con columnas de amplitud, fase y opcionalmente velocidad + etiqueta de canal/punto.",
    )

df_raw: Optional[pd.DataFrame] = None
cols: Optional[PolarColumns] = None
df: Optional[pd.DataFrame] = None

if uploaded is not None:
    try:
        df_raw = _read_csv_robust(uploaded)
        cols = _infer_columns(df_raw)
        df = _prepare_dataframe(df_raw, cols)
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        st.stop()

    with tab_preview:
        st.dataframe(df_raw, use_container_width=True, height=340)

        st.markdown(
            f"""
            <div class="wm-card">
                <h3>Column detection</h3>
                <p>
                    Amplitude: <b>{cols.amplitude}</b> ·
                    Phase: <b>{cols.phase}</b> ·
                    Speed: <b>{cols.speed if cols.speed else 'Auto-generated index'}</b> ·
                    Group: <b>{cols.group if cols.group else 'Single chart mode'}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    amp_unit = _infer_unit_from_name(cols.amplitude, "Amp")
    speed_unit = _infer_unit_from_name(cols.speed, "RPM") if cols.speed else "Index"

    possible_group_cols = _candidate_group_columns(df_raw)
    default_group_col = cols.group if cols.group in possible_group_cols else (possible_group_cols[0] if possible_group_cols else None)

    control_col1, control_col2, control_col3 = st.columns([1.2, 1.2, 1.6])

    with control_col1:
        group_mode = st.radio(
            "Grouping mode",
            options=["Automatic", "Single chart", "Choose column"],
            horizontal=True,
            index=0,
        )

    selected_group_col = None
    if group_mode == "Choose column":
        with control_col2:
            selected_group_col = st.selectbox(
                "Group column",
                options=possible_group_cols if possible_group_cols else [cols.group] if cols.group else [],
                index=0 if possible_group_cols else None,
            )

    work = df.copy()

    if group_mode == "Single chart":
        work["__group__"] = "Polar"
    elif group_mode == "Choose column" and selected_group_col:
        work["__group__"] = df_raw.loc[work.index, selected_group_col].astype(str).fillna("Polar")
    else:
        # Automatic
        pass

    available_groups = list(work["__group__"].astype(str).dropna().unique())
    available_groups = sorted(available_groups)

    selected_groups = st.multiselect(
        "Curves / channels to render",
        options=available_groups,
        default=available_groups,
    )

    if not selected_groups:
        st.warning("Selecciona al menos un gráfico polar.")
        st.stop()

    render_df = work[work["__group__"].isin(selected_groups)].copy()

    st.markdown(
        f"""
        <div class="wm-card">
            <h3>Multi-Polar Mode</h3>
            <p>
                Se renderizarán <b>{len(selected_groups)}</b> gráficos polares independientes,
                cada uno con el mismo estándar visual premium y exportación HD por modebar.
            </p>
            <div class="wm-metric-row">
                <div class="wm-metric">
                    <div class="wm-metric-label">Rows loaded</div>
                    <div class="wm-metric-value">{len(df_raw):,}</div>
                </div>
                <div class="wm-metric">
                    <div class="wm-metric-label">Valid rows</div>
                    <div class="wm-metric-value">{len(df):,}</div>
                </div>
                <div class="wm-metric">
                    <div class="wm-metric-label">Charts selected</div>
                    <div class="wm-metric-value">{len(selected_groups)}</div>
                </div>
                <div class="wm-metric">
                    <div class="wm-metric-label">Amplitude unit</div>
                    <div class="wm-metric-value">{amp_unit}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_config = {
        "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "watermelon_polar_plot",
            "height": 1400,
            "width": 1800,
            "scale": 2,
        },
    }

    for group_name in selected_groups:
        g = render_df[render_df["__group__"].astype(str) == str(group_name)].copy()

        if g.empty:
            continue

        title = str(group_name).strip() if str(group_name).strip() else "Polar"

        fig = _build_polar_figure(
            title=title,
            data=g,
            amp_unit=amp_unit,
            speed_unit=speed_unit,
            clockwise=clockwise,
            phase_offset=float(phase_offset),
            label_every=int(label_every),
            show_markers=bool(show_markers),
            show_speed_labels=bool(show_speed_labels),
            max_critical=int(max_critical),
        )

        st.markdown(
            f"""
            <div class="wm-card">
                <h3>{title}</h3>
                <p>Gráfico polar independiente con amplitud, fase, velocidad, labels RPM y referencias heurísticas.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(fig, use_container_width=True, config=chart_config)

else:
    st.info("Carga un archivo CSV para visualizar uno o varios gráficos polares.")
