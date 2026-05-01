from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

from core.auth import require_login, render_user_menu
from core.csv_common import (
    circular_mean_deg,
    circular_smooth_deg,
    decode_csv_text,
    filter_status_valid,
    find_header_line,
    parse_metadata_block,
    unwrap_deg,
)
from core.diagnostics import (
    build_bode_compare_diagnostics_rotordyn,
    build_bode_diagnostics_rotordyn,
    format_number,
    get_semaforo_status,
)
from core.profile_state import render_profile_selector  # legacy compat
from core.instance_selector import render_instance_selector
from core.report_state import append_report_item_and_persist
from core.rotordynamics import (
    detect_critical_speeds,
    evaluate_api684_margin,
    iso_20816_2_zone,
    iso_20816_zone_multipart,
    mils_to_micrometers,
)
from core.module_patterns import export_report_row, helper_card, panel_card
from core.ui_theme import apply_watermelon_page_style, draw_top_strip, page_header


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Watermelon System | Bode Plot", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"

apply_watermelon_page_style()


# ============================================================
# BODE FILE PERSISTENCE — sobrevive navegación entre módulos
# ============================================================
BODE_UPLOAD_FILES_KEY = "wm_bode_upload_files"


class BodePersistedUploadedFile:
    """Wrapper que mantiene el contenido del CSV en session_state."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def read(self) -> bytes:
        return self._data

    def getvalue(self) -> bytes:
        return self._data

    def seek(self, pos: int) -> None:
        return None


def set_bode_persisted_files(files: List[Any]) -> None:
    packed = []
    for f in files or []:
        try:
            data = f.getvalue()
        except Exception:
            try:
                f.seek(0)
            except Exception:
                pass
            data = f.read()

        packed.append({
            "name": getattr(f, "name", "Bode.csv"),
            "data": data,
        })

    st.session_state[BODE_UPLOAD_FILES_KEY] = packed


def get_bode_persisted_files() -> List[BodePersistedUploadedFile]:
    return [
        BodePersistedUploadedFile(item["name"], item["data"])
        for item in st.session_state.get(BODE_UPLOAD_FILES_KEY, [])
    ]


def clear_bode_persisted_files() -> None:
    st.session_state.pop(BODE_UPLOAD_FILES_KEY, None)


# ============================================================
# STATE
# ============================================================
def ensure_report_state() -> None:
    if "report_items" not in st.session_state:
        st.session_state["report_items"] = []


def get_logo_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


# ============================================================
# LOAD / TRANSFORM
# ============================================================
# circular_mean_deg, circular_smooth_deg y unwrap_deg ahora se importan
# desde core.csv_common (mantienen el mismo comportamiento).


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window is None or window < 2:
        return series.astype(float).copy()
    return series.astype(float).rolling(window=window, center=True, min_periods=1).median()


def read_bode_csv(file_obj) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    text = decode_csv_text(file_obj, errors="replace")
    lines = text.splitlines()
    if not lines:
        raise ValueError("Archivo vacío.")

    # Bently Nevada exporta Bode en dos formatos:
    # Formato A (legacy): X-Axis Value=RPM, Y-Axis Value=amp, Phase, Timestamp
    # Formato B (moderno): X-Axis Value=Timestamp, Y-Axis Value=amp, Phase, Speed=RPM
    # Detectamos por presencia de columnas
    header_idx = find_header_line(
        lines,
        required_signals=("X-Axis Value", "Y-Axis Value", "Phase"),
    )
    if header_idx is None:
        raise ValueError("No se encontró el encabezado real del CSV Bode.")

    meta = parse_metadata_block(lines[:header_idx])
    data_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(data_text), encoding="utf-8-sig")

    has_speed_col = "Speed" in df.columns
    has_timestamp_col = "Timestamp" in df.columns

    if has_speed_col:
        # Formato B (moderno): RPM en columna Speed, X-Axis Value es timestamp
        required = ["X-Axis Value", "Y-Axis Value", "Y-Axis Status", "Phase", "Phase Status", "Speed", "Speed Status"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el CSV (formato moderno): {missing}")

        df["rpm"] = pd.to_numeric(df["Speed"], errors="coerce")
        df["amp"] = pd.to_numeric(df["Y-Axis Value"], errors="coerce")
        df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")
        df["Timestamp"] = pd.to_datetime(df["X-Axis Value"], errors="coerce")
        status_cols = ["Y-Axis Status", "Phase Status", "Speed Status"]
    elif has_timestamp_col:
        # Formato A (legacy): RPM en X-Axis Value, Timestamp en columna Timestamp
        required = ["X-Axis Value", "Y-Axis Value", "Y-Axis Status", "Phase", "Phase Status", "Timestamp"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el CSV (formato legacy): {missing}")

        df["rpm"] = pd.to_numeric(df["X-Axis Value"], errors="coerce")
        df["amp"] = pd.to_numeric(df["Y-Axis Value"], errors="coerce")
        df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        status_cols = ["Y-Axis Status", "Phase Status"]
    else:
        raise ValueError(
            "No se reconoce el formato del Bode. Esperaba columna 'Speed' (formato "
            "moderno) o 'Timestamp' (formato legacy)."
        )

    df = df.dropna(subset=["rpm", "amp", "phase", "Timestamp"]).copy()
    df = filter_status_valid(df, status_cols)

    if df.empty:
        raise ValueError("No quedaron filas válidas después del filtrado.")

    raw_df = df.sort_values(["Timestamp", "rpm"]).reset_index(drop=True)

    grouped_df = (
        raw_df.groupby("rpm", as_index=False)
        .agg(
            amp=("amp", "median"),
            phase=("phase", lambda s: circular_mean_deg(s)),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("rpm", kind="stable")
        .reset_index(drop=True)
    )

    return meta, raw_df, grouped_df


def uploaded_file_label(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Bode.csv")).name


def uploaded_file_stem(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Bode.csv")).stem


def parse_uploaded_bode_files(files: List[Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    parsed_items: List[Dict[str, Any]] = []
    failed_items: List[Tuple[str, str]] = []

    for file_obj in files:
        try:
            meta, raw_df, grouped_df = read_bode_csv(file_obj)
            label = uploaded_file_label(file_obj)
            machine = meta.get("Machine Name", "-")
            point = meta.get("Point Name", label)
            item_id = f"{label}::{machine}::{point}"

            parsed_items.append(
                {
                    "id": item_id,
                    "label": label,
                    "file_name": label,
                    "file_stem": uploaded_file_stem(file_obj),
                    "meta": meta,
                    "raw_df": raw_df,
                    "grouped_df": grouped_df,
                    "machine": machine,
                    "point": point,
                    "variable": meta.get("Variable", "-"),
                }
            )
        except Exception as e:
            failed_items.append((uploaded_file_label(file_obj), str(e)))

    return parsed_items, failed_items


# ============================================================
# ANALYSIS
# ============================================================
def nearest_row_for_rpm(df: pd.DataFrame, rpm_value: float) -> pd.Series:
    idx = int((df["rpm"] - float(rpm_value)).abs().idxmin())
    return df.loc[idx]


def estimate_critical_speeds_api684_style(df: pd.DataFrame, max_count: int = 2) -> List[Dict[str, float]]:
    if df.empty or len(df) < 12:
        return []

    amp = df["amp"].astype(float).to_numpy()
    rpm = df["rpm"].astype(float).to_numpy()
    phase = df["phase_continuous_internal"].astype(float).to_numpy()

    candidates: List[Dict[str, float]] = []

    if find_peaks is not None:
        prominence = max(np.nanmax(amp) * 0.08, 0.12)
        distance = max(8, len(df) // 16)
        peaks, props = find_peaks(amp, prominence=prominence, distance=distance)

        for i, p in enumerate(peaks):
            left = max(0, p - 8)
            right = min(len(df) - 1, p + 8)

            amp_peak = float(amp[p])
            prom = float(props["prominences"][i])
            phase_delta = float(phase[right] - phase[left])

            if amp_peak < np.nanmax(amp) * 0.50:
                continue
            if abs(phase_delta) < 10.0:
                continue
            if amp_peak < np.nanmax(amp) * 0.85 and abs(phase_delta) < 20.0:
                continue

            candidates.append(
                {
                    "rpm": float(rpm[p]),
                    "amp": amp_peak,
                    "phase_delta": phase_delta,
                    "idx": int(p),
                    "prominence": prom,
                }
            )
    else:
        p = int(np.nanargmax(amp))
        left = max(0, p - 8)
        right = min(len(df) - 1, p + 8)
        candidates.append(
            {
                "rpm": float(rpm[p]),
                "amp": float(amp[p]),
                "phase_delta": float(phase[right] - phase[left]),
                "idx": int(p),
                "prominence": float(amp[p]),
            }
        )

    candidates = sorted(candidates, key=lambda x: (x["prominence"], x["amp"]), reverse=True)

    filtered: List[Dict[str, float]] = []
    for cand in candidates:
        if all(abs(cand["rpm"] - kept["rpm"]) > 120 for kept in filtered):
            filtered.append(cand)
        if len(filtered) >= max_count:
            break

    return sorted(filtered, key=lambda x: x["rpm"])


def bode_health_status(
    critical_speeds: List[Dict[str, float]],
    amp_series: pd.Series,
) -> Tuple[str, str, Dict[str, float]]:
    max_amp = float(amp_series.max()) if len(amp_series) else 0.0
    candidate_count = len(critical_speeds)

    if candidate_count == 0:
        score = 15.0
    else:
        dominant_amp = max(float(cs["amp"]) for cs in critical_speeds)
        phase_delta = max(abs(float(cs["phase_delta"])) for cs in critical_speeds)
        score = min(100.0, dominant_amp * 10.0 + abs(phase_delta) * 0.35)

    status, color = get_semaforo_status(score, safe_limit=35.0, warning_limit=70.0)
    return status, color, {
        "score": score,
        "max_amp": max_amp,
        "candidate_count": candidate_count,
    }


def build_bode_text_diagnostics(
    *,
    status: str,
    critical_speeds: List[Dict[str, float]],
    max_amp: float,
) -> Dict[str, str]:
    status_up = str(status or "").upper()
    max_amp = float(max_amp or 0.0)

    if not critical_speeds:
        headline = "Respuesta Bode sin velocidad crítica dominante claramente identificada"
        detail = (
            f"La curva Bode no evidencia un candidato dominante de velocidad crítica dentro del rango evaluado. "
            f"La amplitud máxima observada es {max_amp:.3f}. La ausencia de un pico dominante acompañado por rotación clara de fase "
            f"sugiere una respuesta relativamente controlada para esta corrida.\n\n"
            f"Desde el punto de vista rotodinámico, esta condición debe conservarse como referencia histórica para comparación futura con nuevas corridas, "
            f"ya que el valor analítico del Bode aumenta cuando se contrasta con Polar Plot, órbita 1X y shaft centerline."
        )
        action = (
            "Mantener esta corrida como línea base de comparación.\n"
            "Comparar futuras corridas Bode para identificar migración de fase, incremento de amplitud o aparición de picos nuevos.\n"
            "Correlacionar con Polar Plot, órbitas 1X, espectro y condiciones operativas."
        )
        return {"headline": headline, "detail": detail, "action": action}

    cs1 = critical_speeds[0]
    rpm = float(cs1.get("rpm", 0.0))
    amp = float(cs1.get("amp", 0.0))
    phase_delta = abs(float(cs1.get("phase_delta", 0.0)))

    if phase_delta >= 60:
        modal_sentence = (
            "El giro de fase es suficientemente representativo para considerar una transición modal marcada. "
            "Antes de esta zona el rotor responde de forma predominantemente rígida; al cruzar la forma modal, "
            "la respuesta pasa a estar gobernada por flexibilidad dinámica del sistema rotor-soporte."
        )
    elif phase_delta >= 20:
        modal_sentence = (
            "El giro de fase es moderado y sugiere aproximación a una zona de amplificación dinámica. "
            "Existe modificación de rigidez dinámica aparente, aunque no puede hablarse aún de una velocidad crítica completamente definida."
        )
    else:
        modal_sentence = (
            "El giro de fase es bajo; por tanto, el pico debe tratarse como candidato dinámico no confirmado. "
            "La elevación de amplitud puede estar influenciada por desbalance, excentricidad o condición operativa."
        )

    if status_up == "DANGER":
        headline = f"Respuesta Bode severa compatible con velocidad crítica cerca de {rpm:.0f} rpm"
    elif status_up == "WARNING":
        headline = f"Respuesta Bode con indicios de amplificación dinámica cerca de {rpm:.0f} rpm"
    else:
        headline = f"Respuesta Bode controlada con candidato modal cerca de {rpm:.0f} rpm"

    detail = (
        f"La curva Bode identifica una zona de interés alrededor de {rpm:.0f} rpm, con amplitud aproximada de {amp:.3f} "
        f"y variación de fase de {phase_delta:.1f}°. {modal_sentence}\n\n"
        f"Desde el enfoque de análisis de vibraciones y dinámica de rotores, cuando el máximo de amplitud aparece acompañado por rotación de fase "
        f"en el mismo corredor de velocidad, aumenta la probabilidad de estar frente a una velocidad crítica o forma modal del rotor."
    )

    action = (
        "Correlacionar esta zona con Polar Plot y órbita 1X.\n"
        "Verificar si el cambio de fase ocurre antes, durante o después del máximo de amplitud.\n"
        "Comparar contra corridas históricas para confirmar repetibilidad o migración modal.\n"
        "Validar condiciones de balance, alineación, rigidez de soporte, lubricación y carga."
    )

    return {"headline": headline, "detail": detail, "action": action}

# ============================================================
# FIGURE UI
# ============================================================
def rounded_rect_path(x0: float, y0: float, x1: float, y1: float, r: float) -> str:
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    return (
        f"M {x0+r},{y0} "
        f"L {x1-r},{y0} "
        f"Q {x1},{y0} {x1},{y0+r} "
        f"L {x1},{y1-r} "
        f"Q {x1},{y1} {x1-r},{y1} "
        f"L {x0+r},{y1} "
        f"Q {x0},{y1} {x0},{y1-r} "
        f"L {x0},{y0+r} "
        f"Q {x0},{y0} {x0+r},{y0} Z"
    )


def draw_right_info_box(fig: go.Figure, rows: Sequence[Tuple[str, str]]) -> None:
    panel_x0 = 0.805
    panel_x1 = 0.970
    panel_y0 = 0.60
    panel_y1 = 0.94
    header_h = 0.045
    row_h = 0.055

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y0, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(255,255,255,0.74)",
        layer="above",
    )

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y1 - header_h, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(147,197,253,0.94)",
        layer="above",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=(panel_x0 + panel_x1) / 2.0, y=panel_y1 - header_h / 2.0,
        text="<b>Bode Information</b>",
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=11.1, color="#111827"),
    )

    current_top = panel_y1 - header_h - 0.008
    for title, value in rows:
        title_y = current_top - 0.003
        value_y = current_top - 0.026

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.026, y=title_y,
            xanchor="left", yanchor="top",
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=10.2, color="#111827"),
            align="left",
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.026, y=value_y,
            xanchor="left", yanchor="top",
            text=value,
            showarrow=False,
            font=dict(size=9.9, color="#111827"),
            align="left",
        )
        current_top -= row_h


def build_bode_info_rows(
    row_a: pd.Series,
    row_b: pd.Series,
    phase_mode: str,
    y_unit: str,
    x_unit: str,
    critical_speeds: List[Dict[str, float]],
    semaforo_status: str,
    semaforo_color: str,
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = [
        ("Cursor A", f"{format_number(row_a['amp'],3)} {y_unit} @ {int(round(row_a['rpm']))} {x_unit} | ∠{format_number(row_a['phase_header'],1)}°"),
        ("Cursor B", f"{format_number(row_b['amp'],3)} {y_unit} @ {int(round(row_b['rpm']))} {x_unit} | ∠{format_number(row_b['phase_header'],1)}°"),
        ("Phase Mode", phase_mode),
        ("Status", f"<span style='color:{semaforo_color};'><b>{semaforo_status}</b></span>"),
    ]

    for i, cs in enumerate(critical_speeds, start=1):
        title = f"Critical Speed {i}" if i == 1 else f"Secondary Candidate {i}"
        rows.append((title, f"{int(round(cs['rpm']))} {x_unit} | {format_number(cs['amp'],3)} {y_unit}"))
        rows.append((f"Phase Delta {i}", f"{format_number(cs['phase_delta'],1)}°"))

    return rows


def add_crosshair(fig: go.Figure, rpm_val: float, phase_val: float, amp_val: float, color: str) -> None:
    fig.add_vline(x=rpm_val, line_width=1.3, line_dash="dot", line_color=color, row=1, col=1)
    fig.add_vline(x=rpm_val, line_width=1.3, line_dash="dot", line_color=color, row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=[rpm_val], y=[phase_val], mode="markers",
            marker=dict(size=6, color=color, line=dict(width=1, color="#ffffff")),
            showlegend=False, hoverinfo="skip"
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[rpm_val], y=[amp_val], mode="markers",
            marker=dict(size=6, color=color, line=dict(width=1, color="#ffffff")),
            showlegend=False, hoverinfo="skip"
        ),
        row=2, col=1,
    )


# ============================================================
# FIGURE BUILD
# ============================================================
def build_bode_figure(
    df: pd.DataFrame,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    x_min: float,
    x_max: float,
    logo_uri: Optional[str],
    phase_mode: str,
    critical_speeds: List[Dict[str, float]],
    show_info_box: bool,
    semaforo_status: str,
    semaforo_color: str,
    *,
    operating_rpm: Optional[float] = None,
    iso_thresholds: Optional[Dict[str, float]] = None,
    critical_speeds_pro: Optional[List[Dict[str, Any]]] = None,
    # Ciclo 17.2 — multi-snapshot overlay (lista de dicts con
    # trajectory_speed/amp/phase + label + timestamp). Cada snapshot
    # se dibuja como linea amp vs RPM y phase vs RPM con gradiente
    # cronologico (azul claro = mas viejo, rojo = mas reciente).
    prev_snapshots: Optional[List[Dict[str, Any]]] = None,
) -> go.Figure:
    x_unit = meta.get("X-Axis Unit", "rpm") or "rpm"
    y_unit = meta.get("Y-Axis Unit", "") or ""

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.48, 0.52],
    )

    fig.add_trace(
        go.Scattergl(
            x=df["rpm"],
            y=df["phase_plot"],
            mode="lines",
            line=dict(width=1.10, color="#5b9cf0"),
            name="Phase",
            hovertemplate=f"Speed: %{{x:.0f}} {x_unit}<br>Phase: %{{y:.1f}}°<extra></extra>",
            showlegend=False,
            connectgaps=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df["rpm"],
            y=df["amp"],
            mode="lines",
            line=dict(width=1.35, color="#5b9cf0"),
            name="Amplitude",
            hovertemplate=f"Speed: %{{x:.0f}} {x_unit}<br>Amplitude: %{{y:.3f}} {y_unit}<extra></extra>",
            showlegend=False,
            connectgaps=False,
        ),
        row=2, col=1,
    )

    # ============================================================
    # Ciclo 17.2 — Overlays históricos (amp + phase vs RPM)
    # ------------------------------------------------------------
    # Para cada snapshot anterior con trayectoria, dibujamos una línea
    # amp vs RPM en row=2 y phase vs RPM en row=1 con gradiente
    # cronológico (azul claro = más viejo, rojo = más reciente). Se
    # ven como "ghost curves" debajo del actual y permiten comparar:
    #   - Migración del peak en RPM (mode shift)
    #   - Cambio de amplitud del peak (Q factor degradation)
    #   - Deriva de la fase a través del modo
    # ============================================================
    if prev_snapshots:
        _snaps_sorted = sorted(
            [s for s in prev_snapshots if s.get("trajectory_speed")],
            key=lambda s: s.get("timestamp", "") or "",
        )

        def _bode_gradient(idx: int, total: int) -> str:
            if total <= 1:
                return "rgba(148,163,184,0.55)"
            pos = idx / max(1, total - 1)
            # Mismo esquema que Polar: light blue -> amber -> red
            stops = [
                (0.00, (125, 211, 252)),
                (0.50, (245, 158,  11)),
                (1.00, (220,  38,  38)),
            ]
            for i in range(len(stops) - 1):
                t0, c0 = stops[i]
                t1, c1 = stops[i + 1]
                if t0 <= pos <= t1:
                    frac = (pos - t0) / (t1 - t0)
                    r = int(c0[0] + (c1[0] - c0[0]) * frac)
                    g = int(c0[1] + (c1[1] - c0[1]) * frac)
                    b = int(c0[2] + (c1[2] - c0[2]) * frac)
                    return f"rgba({r},{g},{b},0.55)"
            return "rgba(148,163,184,0.55)"

        for _idx, _snap in enumerate(_snaps_sorted):
            _color = _bode_gradient(_idx, len(_snaps_sorted))
            _lbl = _snap.get("label", "anterior") or "anterior"
            _t_speed = _snap.get("trajectory_speed", []) or []
            _t_amp = _snap.get("trajectory_amp", []) or []
            _t_phase = _snap.get("trajectory_phase", []) or []
            if not (len(_t_speed) > 1 and len(_t_speed) == len(_t_amp) == len(_t_phase)):
                continue

            # Phase vs RPM (row 1)
            fig.add_trace(
                go.Scattergl(
                    x=_t_speed,
                    y=_t_phase,
                    mode="lines",
                    line=dict(width=1.4, color=_color, dash="solid"),
                    name=f"Phase {_lbl[:18]}",
                    hovertemplate=(
                        f"<b>{_lbl}</b><br>"
                        f"Speed: %{{x:.0f}} {x_unit}<br>"
                        f"Phase: %{{y:.1f}}°<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1, col=1,
            )
            # Amp vs RPM (row 2)
            fig.add_trace(
                go.Scattergl(
                    x=_t_speed,
                    y=_t_amp,
                    mode="lines",
                    line=dict(width=1.4, color=_color, dash="solid"),
                    name=f"Amp {_lbl[:18]}",
                    hovertemplate=(
                        f"<b>{_lbl}</b><br>"
                        f"Speed: %{{x:.0f}} {x_unit}<br>"
                        f"Amplitude: %{{y:.3f}} {y_unit}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=2, col=1,
            )
            # Marker en el peak de amplitud (= velocidad critica historica)
            try:
                _i_pk = int(np.argmax(_t_amp))
                fig.add_trace(
                    go.Scattergl(
                        x=[float(_t_speed[_i_pk])],
                        y=[float(_t_amp[_i_pk])],
                        mode="markers",
                        marker=dict(
                            size=10, color=_color,
                            symbol="diamond-open",
                            line=dict(width=1.8, color="#0f172a"),
                        ),
                        name=f"Peak {_lbl[:18]}",
                        hovertemplate=(
                            f"<b>Peak {_lbl}</b><br>"
                            f"Speed: %{{x:.0f}} {x_unit}<br>"
                            f"Amplitude: %{{y:.3f}} {y_unit}<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=2, col=1,
                )
            except Exception:
                pass

    # ============================================================
    # ISO 20816-2 ZONE BANDS — fondo de la amplitud
    # ============================================================
    # Pintamos bandas horizontales A/B/C/D que reflejan los umbrales
    # de severidad de la norma. Esto hace que la zona en la que el
    # rotor opera sea visualmente obvia: si la curva queda en el área
    # verde es zona A (excelente), amarilla zona B (aceptable), naranja
    # zona C (operación restringida), roja zona D (acción inmediata).
    if iso_thresholds is not None:
        ab = float(iso_thresholds.get("AB", 0.0))
        bc = float(iso_thresholds.get("BC", 0.0))
        cd = float(iso_thresholds.get("CD", 0.0))
        if ab > 0 and bc > ab and cd > bc:
            # Determinar el techo del eje amp para la banda D
            try:
                amp_max_data = float(df["amp"].max())
            except Exception:
                amp_max_data = cd * 1.4
            band_top = max(amp_max_data * 1.15, cd * 1.25)

            # Zona A (verde tenue)
            fig.add_hrect(
                y0=0.0, y1=ab,
                fillcolor="rgba(34, 197, 94, 0.10)",
                line_width=0,
                row=2, col=1, layer="below",
            )
            # Zona B (verde-amarillo)
            fig.add_hrect(
                y0=ab, y1=bc,
                fillcolor="rgba(234, 179, 8, 0.10)",
                line_width=0,
                row=2, col=1, layer="below",
            )
            # Zona C (naranja)
            fig.add_hrect(
                y0=bc, y1=cd,
                fillcolor="rgba(249, 115, 22, 0.13)",
                line_width=0,
                row=2, col=1, layer="below",
            )
            # Zona D (rojo)
            fig.add_hrect(
                y0=cd, y1=band_top,
                fillcolor="rgba(220, 38, 38, 0.15)",
                line_width=0,
                row=2, col=1, layer="below",
            )

            # Etiquetas A/B/C/D al borde derecho del eje
            label_x = x_max - (x_max - x_min) * 0.015
            for letter, y_band in (
                ("A", ab * 0.5),
                ("B", (ab + bc) * 0.5),
                ("C", (bc + cd) * 0.5),
                ("D", (cd + band_top) * 0.5),
            ):
                fig.add_annotation(
                    x=label_x, y=y_band,
                    xref="x2", yref="y2",
                    text=f"<b>{letter}</b>",
                    showarrow=False,
                    font=dict(size=11, color="#475569"),
                    bgcolor="rgba(255,255,255,0.65)",
                    bordercolor="rgba(148,163,184,0.4)",
                    borderwidth=1,
                    borderpad=2,
                )

    # ============================================================
    # OPERATING SPEED — línea vertical de referencia
    # ============================================================
    if operating_rpm is not None and x_min <= operating_rpm <= x_max:
        fig.add_vline(
            x=operating_rpm,
            line_width=2.0,
            line_dash="dot",
            line_color="#0f172a",
            row=1, col=1,
        )
        fig.add_vline(
            x=operating_rpm,
            line_width=2.0,
            line_dash="dot",
            line_color="#0f172a",
            row=2, col=1,
        )
        fig.add_annotation(
            x=operating_rpm, y=1.0,
            xref="x2", yref="paper",
            text=f"<b>Op. {operating_rpm:.0f} rpm</b>",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color="#0f172a"),
            bgcolor="rgba(248,250,252,0.95)",
            bordercolor="#0f172a",
            borderwidth=1,
            borderpad=3,
        )

    add_crosshair(fig, float(row_a["rpm"]), float(row_a["phase_plot"]), float(row_a["amp"]), "#efb08c")
    add_crosshair(fig, float(row_b["rpm"]), float(row_b["phase_plot"]), float(row_b["amp"]), "#7ac77b")

    # ============================================================
    # CRÍTICAS PRO (con label de Q + RPM enriquecido)
    # ============================================================
    if critical_speeds_pro:
        cs_pro_colors = ["#dc2626", "#ea580c", "#9333ea"]
        for idx, cs_pro in enumerate(critical_speeds_pro):
            color = cs_pro_colors[idx % len(cs_pro_colors)]
            cs_rpm_pro = float(cs_pro.get("rpm", 0.0))
            q_pro = cs_pro.get("q_factor")
            label_q = f"Q={q_pro:.2f}" if (q_pro is not None and np.isfinite(q_pro)) else "Q=—"

            if not (x_min <= cs_rpm_pro <= x_max):
                continue

            fig.add_vline(
                x=cs_rpm_pro,
                line_width=2.4,
                line_dash="solid",
                line_color=color,
                row=1, col=1,
            )
            fig.add_vline(
                x=cs_rpm_pro,
                line_width=2.4,
                line_dash="solid",
                line_color=color,
                row=2, col=1,
            )

            # Label en la parte superior con RPM + Q
            fig.add_annotation(
                x=cs_rpm_pro, y=1.0,
                xref="x", yref="paper",
                text=f"<b>Crítica #{idx+1}</b><br>{int(round(cs_rpm_pro))} rpm · {label_q}",
                showarrow=False,
                yanchor="top",
                font=dict(size=10, color="#fff"),
                bgcolor=color,
                bordercolor=color,
                borderwidth=1,
                borderpad=4,
            )

    # Si las críticas PRO ya se pintaron arriba, omitimos las legacy para
    # evitar duplicación de líneas/anotaciones que confundan visualmente.
    legacy_criticals = [] if critical_speeds_pro else critical_speeds

    cs_colors = ["#ef4444", "#f59e0b"]
    for idx, cs in enumerate(legacy_criticals):
        color = cs_colors[idx % len(cs_colors)]
        cs_rpm = float(cs["rpm"])
        cs_amp = float(cs["amp"])
        cs_phase_row = nearest_row_for_rpm(df, cs_rpm)
        cs_phase = float(cs_phase_row["phase_plot"])

        fig.add_vline(x=cs_rpm, line_width=1.8, line_dash="dash", line_color=color, row=1, col=1)
        fig.add_vline(x=cs_rpm, line_width=1.8, line_dash="dash", line_color=color, row=2, col=1)

        fig.add_annotation(
            x=cs_rpm, y=cs_phase,
            xref="x", yref="y",
            text=f"Critical Speed {idx+1}<br>{int(round(cs_rpm))} rpm",
            showarrow=True, arrowhead=2, arrowcolor=color,
            ax=34, ay=-28,
            font=dict(size=9.6, color="#7f1d1d" if idx == 0 else "#92400e"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#fecaca" if idx == 0 else "#fde68a",
        )

        fig.add_annotation(
            x=cs_rpm, y=cs_amp,
            xref="x2", yref="y2",
            text=f"{format_number(cs_amp,3)} {y_unit}",
            showarrow=True, arrowhead=2, arrowcolor=color,
            ax=35, ay=-26,
            font=dict(size=9.4, color="#7f1d1d" if idx == 0 else "#92400e"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#fecaca" if idx == 0 else "#fde68a",
        )

    dt_start = pd.to_datetime(df["ts_min"], errors="coerce").min()
    dt_end = pd.to_datetime(df["ts_max"], errors="coerce").max()
    dt_text = "—"
    if pd.notna(dt_start) and pd.notna(dt_end):
        dt_text = f"{dt_start.strftime('%Y-%m-%d %H:%M:%S')} → {dt_end.strftime('%Y-%m-%d %H:%M:%S')}"

    draw_top_strip(
        fig=fig,
        machine=meta.get("Machine Name", ""),
        point_text=meta.get("Point Name", ""),
        variable=meta.get("Variable", "-"),
        dt_text=dt_text,
        rpm_text=f"{int(round(df['rpm'].min()))} - {int(round(df['rpm'].max()))} {x_unit}",
        logo_uri=logo_uri,
    )

    if show_info_box:
        rows = build_bode_info_rows(row_a, row_b, phase_mode, y_unit, x_unit, critical_speeds, semaforo_status, semaforo_color)
        draw_right_info_box(fig, rows)

    x_domain = [0.0, 0.77] if show_info_box else [0.0, 1.0]

    fig.update_layout(
        height=820,
        margin=dict(l=48, r=20, t=145, b=48),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        hovermode="closest",
        dragmode="pan",
        showlegend=False,
    )

    fig.update_xaxes(
        title=f"Speed ({x_unit})",
        range=[x_min, x_max],
        domain=x_domain,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=2, col=1,
    )

    fig.update_xaxes(
        range=[x_min, x_max],
        domain=x_domain,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=1, col=1,
    )

    fig.update_yaxes(
        title="Phase (°)",
        autorange="reversed",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=1, col=1,
    )

    fig.update_yaxes(
        title=f"Amplitude ({y_unit})" if y_unit else "Amplitude",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        tickcolor="#6b7280",
        ticklen=4,
        row=2, col=1,
    )

    return fig


# ============================================================
# EXPORT / REPORT
# ============================================================
def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    export_fig = go.Figure()

    for trace in fig.data:
        if isinstance(trace, go.Scattergl):
            tj = trace.to_plotly_json()
            export_fig.add_trace(
                go.Scatter(
                    x=tj.get("x"),
                    y=tj.get("y"),
                    mode=tj.get("mode"),
                    line=tj.get("line"),
                    marker=tj.get("marker"),
                    hovertemplate=tj.get("hovertemplate"),
                    showlegend=tj.get("showlegend"),
                    name=tj.get("name"),
                    xaxis=tj.get("xaxis"),
                    yaxis=tj.get("yaxis"),
                    connectgaps=tj.get("connectgaps", False),
                )
            )
        else:
            export_fig.add_trace(trace)

    export_fig.update_layout(fig.layout)
    return export_fig


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    scaled = []
    for trace in fig.data:
        tj = trace.to_plotly_json()
        if tj.get("type") == "scatter":
            mode = tj.get("mode", "")
            if "lines" in mode:
                line = dict(tj.get("line", {}) or {})
                line["width"] = max(3.6, float(line.get("width", 1.0)) * 2.25)
                tj["line"] = line
            if "markers" in mode:
                marker = dict(tj.get("marker", {}) or {})
                marker["size"] = max(10, float(marker.get("size", 6)) * 1.6)
                tj["marker"] = marker
        scaled.append(go.Scatter(**tj))

    fig = go.Figure(data=scaled, layout=fig.layout)

    fig.update_layout(
        width=4300,
        height=2200,
        margin=dict(l=110, r=60, t=320, b=110),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=27, color="#111827"),
    )

    fig.update_xaxes(title_font=dict(size=36), tickfont=dict(size=23))
    fig.update_yaxes(title_font=dict(size=36), tickfont=dict(size=23))

    for shape in fig.layout.shapes or []:
        if shape.line is not None:
            width = getattr(shape.line, "width", 1) or 1
            shape.line.width = max(1.8, width * 1.9)

    for ann in fig.layout.annotations or []:
        if ann.font is not None:
            ann.font.size = max(20, int((ann.font.size or 12) * 1.8))

    for img in fig.layout.images or []:
        sx = getattr(img, "sizex", None)
        sy = getattr(img, "sizey", None)
        if sx is not None:
            img.sizex = sx * 1.12
        if sy is not None:
            img.sizey = sy * 1.12

    return fig


def _build_bode_report_notes(text_diag: Dict[str, str]) -> str:
    headline = str(text_diag.get("headline", "") or "").strip()
    detail = str(text_diag.get("detail", "") or "").strip()
    action = str(text_diag.get("action", "") or "").strip()
    # Ciclo 17.2 P3 — narrativa comparativa modal (cuando hay snapshot)
    comparison_narrative = str(text_diag.get("comparison_narrative", "") or "").strip()

    blocks = []
    if headline:
        blocks.append(headline)
    if detail:
        blocks.append(detail)
    # Inyectar el comparativo entre detalle y acciones — flujo natural
    # del reporte: estado actual → contexto histórico → acciones.
    if comparison_narrative:
        blocks.append(comparison_narrative)
    if action:
        blocks.append("Se recomienda:\n" + action)

    return "\n\n".join(blocks).strip()


def build_export_png_bytes(fig: go.Figure, text_diag: Dict[str, str]) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        return export_fig.to_image(format="png", width=4300, height=2200, scale=2), None
    except Exception as e:
        return None, str(e)


def queue_bode_to_report(
    meta: Dict[str, str],
    fig: go.Figure,
    title: str,
    text_diag: Dict[str, str],
    image_bytes: Optional[bytes] = None,
) -> None:
    ensure_report_state()

    if image_bytes is None:
        image_bytes = build_export_png_bytes(fig, text_diag)[0]

    append_report_item_and_persist(
        {
            "id": f"report-bode-{meta.get('Machine Name','')}-{meta.get('Point Name','')}-{title}",
            "type": "bode",
            "title": title,
            "notes": _build_bode_report_notes(text_diag),
            "signal_id": meta.get("Point Name", ""),
            "image_bytes": image_bytes,
            "machine": meta.get("Machine Name", ""),
            "point": meta.get("Point Name", ""),
            "variable": meta.get("Variable", ""),
            "timestamp": "",
        }
    )


# ============================================================
# BODE MULTI-FECHA COMPARE
# ============================================================
def render_bode_compare_section(
    items: List[Dict[str, Any]],
    *,
    smooth_window: int,
    phase_mode: str,
    detect_cs: bool,
    max_critical_speeds: int,
    logo_uri: Optional[str],
) -> None:
    if len(items) < 2:
        return

    st.markdown("---")
    st.markdown("## Comparación multi-fecha · Bode Plot")

    palette = ["#2563eb","#16a34a","#9333ea","#ea580c","#dc2626","#0891b2"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.055, row_heights=[0.48,0.52])

    summary_rows = []
    records = []

    for idx, item in enumerate(items):
        df = item["grouped_df"].copy()
        df["amp"] = smooth_series(df["amp"], smooth_window)

        phase_wrapped_raw = df["phase"].astype(float) % 360.0
        phase_wrapped_smooth = circular_smooth_deg(phase_wrapped_raw, min(smooth_window, 5))

        if phase_mode == "Wrapped Raw 0-360":
            df["phase_plot"] = phase_wrapped_raw
        else:
            df["phase_plot"] = phase_wrapped_smooth

        critical_speeds = estimate_critical_speeds_api684_style(df, max_count=max_critical_speeds) if detect_cs else []

        if critical_speeds:
            cs = critical_speeds[0]
            dom_rpm = float(cs["rpm"])
            dom_amp = float(cs["amp"])
            dom_phase = float(cs["phase_delta"])
        else:
            peak_idx = int(df["amp"].idxmax())
            dom_rpm = float(df.loc[peak_idx,"rpm"])
            dom_amp = float(df.loc[peak_idx,"amp"])
            dom_phase = 0.0

        records.append((dom_rpm, dom_amp, dom_phase, item["file_name"]))

        color = palette[idx % len(palette)]

        fig.add_trace(go.Scatter(x=df["rpm"], y=df["phase_plot"], mode="lines", line=dict(width=2.2,color=color), showlegend=False), row=1,col=1)
        fig.add_trace(go.Scatter(x=df["rpm"], y=df["amp"], mode="lines", line=dict(width=2.5,color=color), name=item["file_name"]), row=2,col=1)

        fig.add_vline(x=dom_rpm, line_width=1.5, line_dash="dash", line_color=color, row=1,col=1)
        fig.add_vline(x=dom_rpm, line_width=1.5, line_dash="dash", line_color=color, row=2,col=1)

        summary_rows.append({
            "Archivo": item["file_name"],
            "RPM candidata": round(dom_rpm,0),
            "Amp dominante": round(dom_amp,3),
            "Delta fase": round(dom_phase,1),
        })

    draw_top_strip(
        fig=fig,
        machine=items[0]["machine"],
        point_text="Bode Plot · Comparación multi-fecha",
        variable=items[0]["meta"].get("Variable","-"),
        dt_text="Comparación histórica",
        rpm_text="Superposición multi-corrida",
        logo_uri=logo_uri,
    )

    fig.update_layout(
        height=860,
        margin=dict(l=60,r=50,t=145,b=105),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        legend=dict(orientation="h",yanchor="top",y=-0.08,xanchor="center",x=0.5),
    )

    fig.update_yaxes(title="Phase (°)", autorange="reversed", row=1,col=1)
    fig.update_yaxes(title="Amplitude", row=2,col=1)
    fig.update_xaxes(title="Speed (rpm)", row=2,col=1)

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="wm_bode_compare_plot")

    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary, width="stretch", hide_index=True)

    baseline = records[0]
    latest = records[-1]

    diag = {
        "headline": "Comparación multi-fecha Bode de amplitud, fase y respuesta modal",
        "detail": (
            f"Entre la corrida base ({baseline[3]}) y la más reciente ({latest[3]}) se observa variación de "
            f"{latest[1]-baseline[1]:+.3f} en amplitud dominante, desplazamiento de {latest[0]-baseline[0]:+.0f} rpm "
            f"y cambio de {latest[2]-baseline[2]:+.1f}° en fase dominante.\n\n"
            f"Este comportamiento permite evaluar migración modal, modificación de rigidez efectiva o cambios en soporte/carga del rotor."
        ),
        "action": (
            "Correlacionar las corridas Bode con Polar Plot y órbitas 1X.\n"
            "Verificar si la velocidad candidata se mantiene o migra entre fechas.\n"
            "Usar la corrida más estable como línea base histórica."
        )
    }

    st.markdown("### Diagnóstico comparativo automático")
    st.markdown(f"**{diag['headline']}**")
    st.write(diag["detail"])
    st.write("Se recomienda:")
    st.write(diag["action"])

    png_bytes = build_export_png_bytes(fig, diag)[0]

    if st.button("Enviar comparativo Bode a reporte", key="wm_bode_compare_report_btn"):
        ensure_report_state()
        append_report_item_and_persist(
            {
                "type": "bode_compare",
                "title": "Bode Plot · Comparación multi-fecha",
                "notes": _build_bode_report_notes(diag),
                "image_bytes": png_bytes,
            }
        )
        st.success("Comparativo Bode enviado al reporte.")


# ============================================================
# PANEL RENDER
# ============================================================
def render_bode_panel(
    item: Dict[str, Any],
    panel_index: int,
    *,
    logo_uri: Optional[str],
    smooth_window: int,
    auto_x: bool,
    x_min_global: float,
    x_max_global: float,
    phase_mode: str,
    detect_cs: bool,
    max_critical_speeds: int,
    show_info_box: bool,
    use_rotordyn_pro: bool = True,
    operating_rpm: float = 3600.0,
    machine_group: str = "group2",
    iso_part: str = "20816-2",
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
    profile_label: Optional[str] = None,
) -> None:
    meta = item["meta"]
    raw_df = item["raw_df"]
    grouped_df = item["grouped_df"]

    plot_df = grouped_df.copy()
    plot_df["amp"] = smooth_series(plot_df["amp"], smooth_window)

    phase_wrapped_raw = plot_df["phase"].astype(float) % 360.0
    phase_wrapped_smooth = circular_smooth_deg(phase_wrapped_raw, min(smooth_window, 5))
    phase_continuous_internal = unwrap_deg(phase_wrapped_smooth)

    plot_df["phase_wrapped_raw"] = phase_wrapped_raw
    plot_df["phase_wrapped_smooth"] = phase_wrapped_smooth
    plot_df["phase_continuous_internal"] = phase_continuous_internal

    if phase_mode == "Wrapped Raw 0-360":
        plot_df["phase_plot"] = plot_df["phase_wrapped_raw"]
        plot_df["phase_header"] = plot_df["phase_wrapped_raw"]
    else:
        plot_df["phase_plot"] = plot_df["phase_wrapped_smooth"]
        plot_df["phase_header"] = plot_df["phase_wrapped_smooth"]

    rpm_min_default = float(plot_df["rpm"].min())
    rpm_max_default = float(plot_df["rpm"].max())

    if auto_x:
        x_min = rpm_min_default
        x_max = rpm_max_default
    else:
        x_min = x_min_global
        x_max = x_max_global

    display_df = plot_df[(plot_df["rpm"] >= x_min) & (plot_df["rpm"] <= x_max)].copy()
    if display_df.empty:
        st.warning(f"Panel {panel_index + 1}: no hay puntos en el rango RPM seleccionado.")
        return

    rpm_min_display = int(display_df["rpm"].min())
    rpm_max_display = int(display_df["rpm"].max())

    c1, c2 = st.columns(2)
    with c1:
        cursor_a_rpm = st.slider(
            f"Cursor A (RPM) · Panel {panel_index + 1}",
            rpm_min_display,
            rpm_max_display,
            rpm_min_display,
            key=f"bode_cursor_a_{panel_index}_{item['id']}",
        )
    with c2:
        cursor_b_rpm = st.slider(
            f"Cursor B (RPM) · Panel {panel_index + 1}",
            rpm_min_display,
            rpm_max_display,
            rpm_max_display,
            key=f"bode_cursor_b_{panel_index}_{item['id']}",
        )

    row_a = nearest_row_for_rpm(display_df, cursor_a_rpm)
    row_b = nearest_row_for_rpm(display_df, cursor_b_rpm)

    critical_speeds: List[Dict[str, float]] = []
    if detect_cs:
        critical_speeds = estimate_critical_speeds_api684_style(display_df, max_count=max_critical_speeds)

    semaforo_status, semaforo_color, bode_diag = bode_health_status(
        critical_speeds=critical_speeds,
        amp_series=display_df["amp"],
    )

    machine = meta.get("Machine Name", "-")
    point = meta.get("Point Name", "-")
    variable = meta.get("Variable", "-")
    probe_angle = meta.get("Probe Angle", "-")
    x_unit = meta.get("X-Axis Unit", "rpm")
    y_unit = meta.get("Y-Axis Unit", "")

    if use_rotordyn_pro:
        # measurement_type depende de la parte ISO
        if iso_part in ("20816-4", "20816-7"):
            mtype = "casing_velocity"
        else:
            mtype = "shaft_displacement"

        text_diag = build_bode_diagnostics_rotordyn(
            rpm=display_df["rpm"].to_numpy(),
            amp=display_df["amp"].to_numpy(),
            phase=display_df["phase_continuous_internal"].to_numpy(),
            operating_rpm=operating_rpm,
            machine_group=machine_group,
            amp_unit=y_unit or "µm pp",
            measurement_type=mtype,
            iso_part=iso_part,
            custom_thresholds=custom_thresholds,
            profile_label=profile_label,
        )
    else:
        text_diag = build_bode_text_diagnostics(
            status=semaforo_status,
            critical_speeds=critical_speeds,
            max_amp=bode_diag["max_amp"],
        )

    # =========================================================
    # Datos para overlay visual Cat IV: críticas PRO + umbrales ISO
    # =========================================================
    pro_overlay_criticals: List[Dict[str, Any]] = []
    iso_thresholds_overlay: Optional[Dict[str, float]] = None
    if use_rotordyn_pro:
        try:
            crits_pro = detect_critical_speeds(
                rpm=display_df["rpm"].to_numpy(),
                amp=display_df["amp"].to_numpy(),
                phase=display_df["phase_continuous_internal"].to_numpy(),
            )
            pro_overlay_criticals = [
                {"rpm": cs.rpm, "q_factor": cs.q_factor}
                for cs in crits_pro
            ]
            mtype_overlay = (
                "casing_velocity" if iso_part in ("20816-4", "20816-7")
                else "shaft_displacement"
            )
            iso_eval_overlay = iso_20816_zone_multipart(
                amplitude=display_df["amp"].max() * (25.4 if "mil" in (y_unit or "").lower() else 1.0),
                iso_part=iso_part,
                machine_group=machine_group,
                measurement_type=mtype_overlay,
                operating_speed_rpm=operating_rpm,
                custom_thresholds=custom_thresholds,
            )
            unit_lower = (y_unit or "").lower()
            if "mil" in unit_lower:
                iso_thresholds_overlay = {
                    "AB": iso_eval_overlay.boundary_AB / 25.4,
                    "BC": iso_eval_overlay.boundary_BC / 25.4,
                    "CD": iso_eval_overlay.boundary_CD / 25.4,
                }
            else:
                iso_thresholds_overlay = {
                    "AB": iso_eval_overlay.boundary_AB,
                    "BC": iso_eval_overlay.boundary_BC,
                    "CD": iso_eval_overlay.boundary_CD,
                }
        except Exception:
            pass

    panel_card(
        title=f"Bode {panel_index + 1} · {machine} · {point}",
        subtitle="Run-up / coast-down amplitude and phase view",
        meta_html=(
            f"Variable: <b>{variable}</b> &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Probe Angle: <b>{probe_angle}</b> &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Speed Range: <b>{int(display_df['rpm'].min())} - {int(display_df['rpm'].max())} {x_unit}</b>"
        ),
        chips=[
            f"File: {item['file_name']}",
            f"Raw rows: {len(raw_df):,}",
            f"Grouped points: {len(display_df):,}",
            f"Phase mode: {phase_mode}",
            f"Smoothing: {smooth_window}",
            f"Critical speeds: {len(critical_speeds)}",
        ],
    )

    # Ciclo 17.2 — buscar snapshots elegidos y armar lista de prev
    _bode_prev_snapshots_list: List[Dict[str, Any]] = []
    try:
        _bode_cmp_snap_ids = (
            st.session_state.get("wm_bode_compare_snapshot_ids") or []
        )
        if _bode_cmp_snap_ids:
            from core.bode_history import load_bode_snapshot
            from core.sensor_map import (
                resolve_sensor_for_point as _sm_resolve,
                sensor_label as _sm_slbl,
            )
            from core.instance_state import get_instance as _sm_get_inst
            _bode_inst_id_local = (
                st.session_state.get("wm_active_instance_id", "")
                or st.session_state.get("wm_bode_compare_inst_id", "")
            )
            if _bode_inst_id_local:
                _inst_obj = _sm_get_inst(_bode_inst_id_local)
                _curr_panel_lbl = None
                if _inst_obj is not None and _inst_obj.sensors:
                    _sensor_match = _sm_resolve(
                        list(_inst_obj.sensors),
                        str(meta.get("Point Name", "") or item.get("point", "")),
                        str(meta.get("Variable", "") or item.get("variable", "")),
                        str(meta.get("Y-Axis Unit", "") or meta.get("Unit", "") or ""),
                    )
                    if _sensor_match is not None:
                        _curr_panel_lbl = _sm_slbl(_sensor_match)
                if _curr_panel_lbl:
                    for _snap_id in _bode_cmp_snap_ids:
                        _prev_snap_full = load_bode_snapshot(
                            _bode_inst_id_local, _snap_id,
                        )
                        if _prev_snap_full is None:
                            continue
                        for _ps in _prev_snap_full.get("sensors", []):
                            if str(_ps.get("sensor_label", "")) == _curr_panel_lbl:
                                _bode_prev_snapshots_list.append({
                                    "amp": float(_ps.get("amp_at_op", 0) or 0),
                                    "phase": float(_ps.get("phase_at_op", 0) or 0),
                                    "label": _prev_snap_full.get("corrida_label", ""),
                                    "op_speed": float(_prev_snap_full.get("operating_speed_rpm", 0) or 0),
                                    "timestamp": _prev_snap_full.get("timestamp", ""),
                                    "trajectory_speed": _ps.get("trajectory_speed", []) or [],
                                    "trajectory_amp": _ps.get("trajectory_amp", []) or [],
                                    "trajectory_phase": _ps.get("trajectory_phase", []) or [],
                                })
                                break
    except Exception:
        _bode_prev_snapshots_list = []

    # Ciclo 17.2 P3 — narrativa modal completa estilo Bently/API 684
    # para el PDF Reports cuando hay snapshot anterior elegido.
    if (
        _bode_prev_snapshots_list and use_rotordyn_pro
        and operating_rpm is not None
    ):
        try:
            from core.bode_history import (
                phase_shift_classifier,
                amplitude_change_classifier,
                shortest_arc_phase_diff,
            )
            _first_prev = _bode_prev_snapshots_list[0]
            _prev_amp = float(_first_prev.get("amp", 0))
            _prev_phase = float(_first_prev.get("phase", 0))
            _prev_label = str(_first_prev.get("label", "corrida anterior"))
            _prev_op_speed = float(_first_prev.get("op_speed", 0) or 0)

            if _prev_amp > 0 and len(display_df) > 0:
                _amp_unit_local = (
                    meta.get("Amp Unit", "")
                    or meta.get("Y-Axis Unit", "")
                    or ""
                )
                # Op point actual
                _diff = (display_df["rpm"] - operating_rpm).abs()
                _curr_row = display_df.loc[int(_diff.idxmin())]
                _curr_amp_op = float(_curr_row.get("amp", 0))
                _curr_phase_op = float(_curr_row.get("phase_plot", _curr_row.get("phase", 0))) % 360.0

                _delta_amp = _curr_amp_op - _prev_amp
                _delta_amp_pct = (
                    (_delta_amp / _prev_amp * 100.0) if _prev_amp > 0 else None
                )
                _delta_phase = shortest_arc_phase_diff(_prev_phase, _curr_phase_op)
                _phase_class = phase_shift_classifier(_delta_phase)

                # Datos del modo en la corrida actual
                _curr_cs_rpm = None
                _curr_q = None
                _curr_cs_phase_delta = None
                if pro_overlay_criticals:
                    try:
                        _cs0 = pro_overlay_criticals[0]
                        _curr_cs_rpm = float(_cs0.get("rpm", 0) or 0)
                        _curr_q = float(_cs0.get("q_factor", 0) or 0)
                        _curr_cs_phase_delta = float(_cs0.get("phase_delta", 0) or 0)
                    except Exception:
                        pass

                # Clasificar el modo por phase delta a la critica
                _mode_type = "modo no caracterizado"
                if _curr_cs_phase_delta and _curr_cs_phase_delta != 0:
                    _abs_pd = abs(_curr_cs_phase_delta)
                    if 150.0 <= _abs_pd <= 210.0:
                        _mode_type = (
                            "primer modo translacional (Δφ ≈ 180°), "
                            "consistente con bending mode tipo 'in-phase' "
                            "del rotor según la nomenclatura de Bently y "
                            "API 684 §6"
                        )
                    elif 70.0 <= _abs_pd < 150.0:
                        _mode_type = (
                            "modo cónico / pivotal o segundo modo "
                            "translacional (Δφ ≈ 90–150°), respuesta "
                            "típica cuando el rotor pivota alrededor de "
                            "un punto nodal cercano al cojinete"
                        )
                    elif 210.0 < _abs_pd <= 360.0:
                        _mode_type = (
                            "segundo modo flexural o respuesta acoplada "
                            "rotor-estructura (Δφ > 210°)"
                        )
                    else:
                        _mode_type = (
                            "respuesta de baja deflexión modal "
                            "(Δφ < 70°) — posible resonancia estructural "
                            "del soporte / fundación más que modo del "
                            "rotor"
                        )

                _narr_parts: List[str] = []

                # 1) Encabezado factual
                _narr_parts.append(
                    f"Análisis comparativo rotodinámico Bode contra "
                    f"«{_prev_label}». A la velocidad operativa "
                    f"({operating_rpm:.0f} rpm), la respuesta sincrónica "
                    f"1X del sensor evolucionó de "
                    f"{_prev_amp:.3f} {_amp_unit_local} @ "
                    f"{_prev_phase:.1f}° a "
                    f"{_curr_amp_op:.3f} {_amp_unit_local} @ "
                    f"{_curr_phase_op:.1f}°, vector change de "
                    + (
                        f"{_delta_amp:+.3f} {_amp_unit_local} "
                        f"({_delta_amp_pct:+.1f}%)"
                        if _delta_amp_pct is not None
                        else f"{_delta_amp:+.3f} {_amp_unit_local}"
                    )
                    + f" en magnitud y un shift de fase 1X de "
                    f"{_delta_phase:+.1f}° en arco menor."
                )

                # 2) Caracterizacion del modo + critical speed evolution
                if _curr_cs_rpm and _curr_cs_rpm > 0:
                    _ratio_op_cs = operating_rpm / _curr_cs_rpm
                    _separation_pct = (_ratio_op_cs - 1.0) * 100.0
                    _q_str = (
                        f", con factor de amplificación Q={_curr_q:.2f}"
                        if _curr_q else ""
                    )
                    _mode_para = (
                        f"El Bode actual identifica una velocidad crítica en "
                        f"{_curr_cs_rpm:.0f} rpm con un cambio de fase de "
                        f"{abs(_curr_cs_phase_delta or 0):.0f}° a través "
                        f"del peak{_q_str}, patrón consistente con "
                        f"{_mode_type}. "
                    )
                    if _separation_pct >= 15.0:
                        _mode_para += (
                            f"La separación operativa-modo es de "
                            f"{_separation_pct:+.1f}%, dentro del margen "
                            f"recomendado por API 684 §6 (≥15%). "
                        )
                    elif _separation_pct >= 0:
                        _mode_para += (
                            f"La separación operativa-modo es de solo "
                            f"{_separation_pct:+.1f}%, estrecha frente al "
                            f"margen recomendado por API 684 §6 (≥15%) — "
                            f"merece evaluación detallada si el Q crece. "
                        )
                    else:
                        _mode_para += (
                            f"La operativa está {abs(_separation_pct):.1f}% "
                            f"por debajo del modo identificado, configuración "
                            f"sub-crítica estable mientras el Q se mantenga "
                            f"acotado. "
                        )

                    # Comparativo de critical speed entre snapshots
                    _prev_cs_strs = []
                    for _ps in _bode_prev_snapshots_list:
                        _ts_amps = _ps.get("trajectory_amp", []) or []
                        _ts_speeds = _ps.get("trajectory_speed", []) or []
                        if len(_ts_amps) > 1 and len(_ts_amps) == len(_ts_speeds):
                            try:
                                _i_pk_p = int(np.argmax(_ts_amps))
                                _prev_cs_rpm_i = float(_ts_speeds[_i_pk_p])
                                _prev_cs_amp_i = float(_ts_amps[_i_pk_p])
                                _prev_cs_strs.append(
                                    f"{_ps.get('label','prev')[:18]} pico en "
                                    f"{_prev_cs_rpm_i:.0f} rpm "
                                    f"({_prev_cs_amp_i:.2f} {_amp_unit_local})"
                                )
                            except Exception:
                                pass
                    if _prev_cs_strs:
                        _mode_para += (
                            "Comparado contra los snapshots seleccionados — "
                            + "; ".join(_prev_cs_strs[:3])
                            + " — se evalúa migración del modo "
                            "(desplazamiento del peak en RPM = mode shift) "
                            "y evolución del Q (cambio de amplitud del "
                            "peak = damping change)."
                        )
                    _narr_parts.append(_mode_para)

                # 3) Diagnostico diferencial del shift
                if _phase_class == "shift_critical":
                    _narr_parts.append(
                        "El shift de fase 1X supera 60° en arco menor, "
                        "magnitud crítica según los criterios de Bently / "
                        "API 684. Este nivel es inconsistente con simple "
                        "deriva térmica u operacional y apunta a un cambio "
                        "mecánico estructural del rotor: pérdida de masa, "
                        "propagación de fisura, asentamiento súbito del "
                        "cojinete o pérdida de contacto en el sello / "
                        "impeller. Se recomienda parada controlada, "
                        "análisis de orbits filtrados a 1X y forma de "
                        "onda en ambos planos del cojinete antes de "
                        "continuar operación."
                    )
                elif _phase_class == "shift_major":
                    _narr_parts.append(
                        "El shift de fase 1X entre 30° y 60° es la firma "
                        "clásica de un cambio de balance del rotor según "
                        "la metodología de vector polar response que "
                        "documentan Bently y API 684. Magnitud y dirección "
                        "del vector change consistentes con redistribución "
                        "de masa rotativa (suciedad acumulada / desprendida "
                        "en álabes, pérdida de balance weights, "
                        "deformación térmica residual). Se recomienda "
                        "balance de campo según ISO 21940-12 nivel G 2.5 "
                        "en próxima ventana."
                    )
                elif _phase_class == "shift_minor":
                    _narr_parts.append(
                        "El shift de fase 1X entre 10° y 30° es menor y "
                        "puede atribuirse a deriva operacional normal "
                        "(temperatura, carga, expansión térmica). No "
                        "evidencia cambio mecánico por sí solo, pero una "
                        "tendencia consolidada en la misma dirección "
                        "vectorial a lo largo de varias corridas indica "
                        "cambio incipiente de balance que conviene "
                        "caracterizar antes que cruce la zona mayor."
                    )
                elif _phase_class == "stable":
                    _narr_parts.append(
                        "El shift de fase 1X (<10°) está dentro de la "
                        "variación normal de la respuesta sincrónica. La "
                        "forma del vector se considera estable entre "
                        "corridas."
                    )

                # 4) Sensitividad
                if _delta_amp_pct is not None:
                    _amp_class = amplitude_change_classifier(_delta_amp_pct)
                    if _amp_class == "amp_critical":
                        _narr_parts.append(
                            "El crecimiento de amplitud 1X supera 50% entre "
                            "corridas. Combinado con el shift de fase, "
                            "refuerza el diagnóstico de degradación activa "
                            "de la respuesta modal — la sensibilidad del "
                            "rotor a la fuerza de excitación residual está "
                            "creciendo, típico de pérdida de "
                            "amortiguamiento (damping degradation) en "
                            "soportes hidrodinámicos según API 684."
                        )
                    elif _amp_class == "amp_high":
                        _narr_parts.append(
                            "El crecimiento de amplitud 1X (≥20%) "
                            "acompañando al shift de fase es consistente "
                            "con cambio activo en la respuesta modal del "
                            "sistema rotor-soporte. Conviene revisar el Q "
                            "factor en próximas corridas para descartar "
                            "pérdida progresiva de damping."
                        )
                    elif _amp_class in ("amp_down_strong", "amp_down"):
                        _narr_parts.append(
                            "La amplitud 1X bajó respecto a la corrida "
                            "anterior. Si esto coincide con shift de fase "
                            "mayor, puede tratarse de balance "
                            "compensatorio (intervención previa, "
                            "redistribución térmica) más que degradación. "
                            "Revisar registro operacional y reportes de "
                            "mantenimiento entre corridas."
                        )

                # 5) Distinción modal rotor vs estructural
                if (
                    _phase_class in ("shift_major", "shift_critical")
                    and _curr_cs_phase_delta is not None
                    and abs(_curr_cs_phase_delta) < 90.0
                ):
                    _narr_parts.append(
                        "Nota diferencial: el cambio de fase a través del "
                        "peak (<90°) es atípico de un modo libre del rotor "
                        "y sugiere que el peak podría corresponder a una "
                        "resonancia estructural del soporte o fundación "
                        "más que a un modo del rotor. Importante validar "
                        "antes de atribuir el cambio observado a balance "
                        "del rotor — falla en fundación / grouting / "
                        "anclajes produce el mismo patrón en el Bode pero "
                        "requiere intervención estructural, no balance."
                    )

                _comp_narr = " ".join(_narr_parts)
                if isinstance(text_diag, dict):
                    text_diag["comparison_narrative"] = _comp_narr
        except Exception:
            pass

    fig = build_bode_figure(
        df=display_df,
        meta=meta,
        row_a=row_a,
        row_b=row_b,
        x_min=float(display_df["rpm"].min()),
        x_max=float(display_df["rpm"].max()),
        logo_uri=logo_uri,
        phase_mode=phase_mode,
        critical_speeds=critical_speeds,
        show_info_box=show_info_box,
        semaforo_status=semaforo_status,
        semaforo_color=semaforo_color,
        operating_rpm=operating_rpm if use_rotordyn_pro else None,
        iso_thresholds=iso_thresholds_overlay,
        critical_speeds_pro=pro_overlay_criticals if pro_overlay_criticals else None,
        prev_snapshots=_bode_prev_snapshots_list if _bode_prev_snapshots_list else None,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False},
        key=f"wm_bode_plot_{panel_index}_{item['id']}",
    )

    helper_card(
        title=f"API RP 684 Helper · Bode {panel_index + 1}",
        subtitle=text_diag["headline"],
        chips=[
            (f"Semáforo: {semaforo_status}", semaforo_color),
            (f"Health score: {bode_diag['score']:.1f}", None),
            (f"Max amplitude: {bode_diag['max_amp']:.3f} {y_unit}", None),
            (f"Critical candidates: {bode_diag['candidate_count']}", None),
            (f"Cursor A: {row_a['amp']:.3f} {y_unit}", None),
            (f"Cursor B: {row_b['amp']:.3f} {y_unit}", None),
        ],
    )

    st.info(
        f"**Diagnostic detail:** {text_diag['detail']}\n\n"
        f"**Recommended action:** {text_diag['action']}"
    )

    title = f"Bode {panel_index + 1} — {machine} — {point}"
    export_state_key = (
        f"bode::{item['id']}::{panel_index}::{phase_mode}::{smooth_window}::"
        f"{detect_cs}::{max_critical_speeds}::{show_info_box}::"
        f"{int(display_df['rpm'].min())}::{int(display_df['rpm'].max())}::"
        f"{cursor_a_rpm}::{cursor_b_rpm}"
    )

    export_report_row(
        export_key=export_state_key,
        fig=fig,
        export_builder=lambda export_fig: build_export_png_bytes(export_fig, text_diag),
        report_callback=lambda: queue_bode_to_report(
            meta,
            fig,
            title,
            text_diag,
            image_bytes=build_export_png_bytes(fig, text_diag)[0],
        ),
        file_name=f"{item['file_stem']}_bode_hd.png",
    )



# ============================================================
# BODE COMPARISON PRO
# ============================================================
def _bode_temporal_palette(n: int) -> List[str]:
    """Paleta temporal para multi-fecha Bode (oldest → newest)."""
    if n <= 1:
        return ["#2563eb"]
    if n == 2:
        return ["#3b82f6", "#ea580c"]
    if n == 3:
        return ["#3b82f6", "#16a34a", "#ea580c"]
    if n == 4:
        return ["#3b82f6", "#16a34a", "#f59e0b", "#dc2626"]
    base = ["#3b82f6", "#0891b2", "#16a34a", "#84cc16", "#f59e0b", "#ea580c", "#dc2626", "#7c3aed"]
    if n <= len(base):
        return base[:n]
    return base + ["#7c3aed"] * (n - len(base))


def _bode_compare_record(
    item: Dict[str, Any],
    smooth_window: int,
    phase_mode: str,
    *,
    operating_rpm: float,
    machine_group: str,
    iso_part: str = "20816-2",
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """Computa rotordynamics + dataframe de plotting para un Bode en compare."""
    df = item["grouped_df"].copy()
    df["amp"] = smooth_series(df["amp"], smooth_window)

    phase_wrapped_raw = df["phase"].astype(float) % 360.0
    phase_wrapped_smooth = circular_smooth_deg(phase_wrapped_raw, min(smooth_window, 5))
    df["phase_continuous_internal"] = unwrap_deg(phase_wrapped_smooth)

    if phase_mode == "Wrapped Raw 0-360":
        df["phase_plot"] = phase_wrapped_raw
    else:
        df["phase_plot"] = phase_wrapped_smooth

    amp_unit = item.get("meta", {}).get("Y-Axis Unit", "µm pp") or "µm pp"
    criticals_rotordyn = []
    primary_critical = None
    primary_api684 = None
    iso_eval = None
    peak_amp_csv = float(df["amp"].max()) if len(df) else 0.0
    peak_amp_um_pp = 0.0

    if len(df) >= 8:
        try:
            criticals_rotordyn = detect_critical_speeds(
                rpm=df["rpm"].to_numpy(),
                amp=df["amp"].to_numpy(),
                phase=df["phase_continuous_internal"].to_numpy(),
            )
        except Exception:
            criticals_rotordyn = []

        if criticals_rotordyn:
            primary_critical = criticals_rotordyn[0]
            primary_api684 = evaluate_api684_margin(
                critical_rpm=primary_critical.rpm,
                operating_rpm=operating_rpm,
                q_factor=primary_critical.q_factor,
            )

        unit_lower = amp_unit.lower()
        if "mil" in unit_lower:
            peak_amp_um_pp = mils_to_micrometers(peak_amp_csv)
        elif "µm" in unit_lower or "um" in unit_lower:
            peak_amp_um_pp = peak_amp_csv
        else:
            peak_amp_um_pp = peak_amp_csv

        try:
            mtype_compare = (
                "casing_velocity" if iso_part in ("20816-4", "20816-7")
                else "shaft_displacement"
            )
            iso_eval = iso_20816_zone_multipart(
                amplitude=peak_amp_um_pp,
                iso_part=iso_part,
                machine_group=machine_group,
                measurement_type=mtype_compare,
                operating_speed_rpm=operating_rpm,
                custom_thresholds=custom_thresholds,
            )
        except Exception:
            iso_eval = None

    # Timestamp del Bode (toma el min de la columna Timestamp si existe en raw_df)
    ts_start = None
    raw_df = item.get("raw_df")
    if raw_df is not None and "Timestamp" in raw_df.columns:
        try:
            ts_start = pd.to_datetime(raw_df["Timestamp"], errors="coerce").min()
        except Exception:
            ts_start = None

    return {
        "label": item.get("file_name", "Bode.csv"),
        "df": df,
        "ts_start": ts_start,
        "amp_unit": amp_unit,
        "primary_critical": primary_critical,
        "primary_api684": primary_api684,
        "iso_eval": iso_eval,
        "peak_amp_csv": peak_amp_csv,
        "peak_amp_um_pp": peak_amp_um_pp,
    }


def render_bode_compare_section(
    items: List[Dict[str, Any]],
    *,
    smooth_window: int,
    phase_mode: str,
    detect_cs: bool,
    max_critical_speeds: int,
    logo_uri: Optional[str],
    use_rotordyn_pro: bool = True,
    operating_rpm: float = 3600.0,
    machine_group: str = "group2",
    iso_part: str = "20816-2",
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
) -> None:
    if len(items) < 2:
        return

    st.markdown("---")
    st.markdown("## Comparación multi-fecha · Bode Plot")

    records = [
        _bode_compare_record(
            item, smooth_window, phase_mode,
            operating_rpm=operating_rpm, machine_group=machine_group,
            iso_part=iso_part, custom_thresholds=custom_thresholds,
        )
        for item in items
    ]

    # Ordenar cronológicamente para que la paleta refleje secuencia temporal
    records_chrono = sorted(
        records,
        key=lambda r: pd.Timestamp(r["ts_start"]) if r["ts_start"] is not None else pd.Timestamp.min,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.48, 0.52],
    )

    palette = _bode_temporal_palette(len(records_chrono))
    legacy_records: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records_chrono):
        df = rec["df"]
        color = palette[idx]
        date_label = (
            pd.Timestamp(rec["ts_start"]).strftime("%d %b %Y")
            if rec["ts_start"] is not None
            else rec["label"]
        )

        if use_rotordyn_pro and rec.get("primary_critical") is not None:
            cs_pro = rec["primary_critical"]
            zone = rec.get("iso_eval").zone if rec.get("iso_eval") else "—"
            q_str = f"{cs_pro.q_factor:.2f}" if np.isfinite(cs_pro.q_factor) else "—"
            trace_name = f"{date_label}  ·  Q={q_str}  ·  zona {zone}"
            dom_rpm = float(cs_pro.rpm)
        else:
            trace_name = f"{date_label}  ·  {rec['label']}"
            peak_idx = int(df["amp"].idxmax()) if len(df) else 0
            dom_rpm = float(df.loc[peak_idx, "rpm"]) if len(df) else 0.0

        fig.add_trace(
            go.Scatter(
                x=df["rpm"], y=df["phase_plot"],
                mode="lines", line=dict(width=2.2, color=color),
                name=f"{date_label} · fase", showlegend=False,
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["rpm"], y=df["amp"],
                mode="lines", line=dict(width=2.8, color=color),
                name=trace_name, showlegend=True,
            ), row=2, col=1,
        )

        # Línea vertical en la crítica detectada
        if dom_rpm > 0:
            fig.add_vline(x=dom_rpm, line_width=1.6, line_dash="dash", line_color=color, row=1, col=1)
            fig.add_vline(x=dom_rpm, line_width=1.6, line_dash="dash", line_color=color, row=2, col=1)

        # Línea vertical en velocidad operativa (común a todas)
        if idx == 0 and operating_rpm > 0:
            x_min_check = float(df["rpm"].min())
            x_max_check = float(df["rpm"].max())
            if x_min_check <= operating_rpm <= x_max_check:
                fig.add_vline(
                    x=operating_rpm,
                    line_width=2.0, line_dash="dot", line_color="#0f172a",
                    annotation_text=f"Op. {operating_rpm:.0f} rpm",
                    annotation_position="top right",
                    row=2, col=1,
                )

        legacy_records.append({
            "Archivo": rec["label"],
            "RPM candidata": dom_rpm,
            "Amp dominante": float(df["amp"].max()),
            "Delta fase": rec.get("primary_critical").phase_change_deg if rec.get("primary_critical") is not None else 0.0,
            "Max amp": float(df["amp"].max()),
        })

    combined = pd.concat([item["grouped_df"] for item in items], ignore_index=True)
    x_min = float(combined["rpm"].min())
    x_max = float(combined["rpm"].max())

    first_meta = items[0]["meta"]
    x_unit = first_meta.get("X-Axis Unit", "rpm") or "rpm"
    y_unit = first_meta.get("Y-Axis Unit", "") or "µm pp"

    draw_top_strip(
        fig=fig,
        machine=items[0].get("machine", ""),
        point_text="Bode Plot · Comparación multi-fecha",
        variable=first_meta.get("Variable", "-"),
        dt_text="Comparación histórica",
        rpm_text=f"{int(x_min)} - {int(x_max)} {x_unit}",
        logo_uri=logo_uri,
    )

    fig.update_layout(
        height=860,
        margin=dict(l=60, r=50, t=145, b=105),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )

    fig.update_xaxes(range=[x_min, x_max], showgrid=True, gridcolor="rgba(148,163,184,0.18)", row=1, col=1)
    fig.update_xaxes(title=f"Speed ({x_unit})", range=[x_min, x_max], showgrid=True, gridcolor="rgba(148,163,184,0.18)", row=2, col=1)
    fig.update_yaxes(title="Phase (°)", autorange="reversed", showgrid=True, gridcolor="rgba(148,163,184,0.18)", row=1, col=1)
    fig.update_yaxes(title=f"Amplitude ({y_unit})", showgrid=True, gridcolor="rgba(148,163,184,0.18)", row=2, col=1)

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="wm_bode_compare_plot")

    # =========================================================
    # Tabla rotodinámica Cat IV (en unidad fuente del CSV)
    # =========================================================
    if use_rotordyn_pro:
        amp_unit_common = records_chrono[0].get("amp_unit", y_unit)
        peak_col_label = f"Peak ({amp_unit_common})"
        unit_lower = amp_unit_common.lower()
        peak_fmt = "{:.3f}" if "mil" in unit_lower else "{:.1f}"

        rows = []
        for r in records_chrono:
            cs_pro = r.get("primary_critical")
            api_pro = r.get("primary_api684")
            iso_eval = r.get("iso_eval")
            rows.append({
                "Fecha": pd.Timestamp(r["ts_start"]).strftime("%Y-%m-%d") if r["ts_start"] is not None else "—",
                "Archivo": r["label"],
                "RPM crítica": f"{cs_pro.rpm:.0f}" if cs_pro is not None else "—",
                "Q factor": f"{cs_pro.q_factor:.2f}" if (cs_pro is not None and np.isfinite(cs_pro.q_factor)) else "—",
                "Δfase (°)": f"{cs_pro.phase_change_deg:.0f}" if cs_pro is not None else "—",
                "FWHM (rpm)": f"{cs_pro.fwhm_rpm:.0f}" if (cs_pro is not None and np.isfinite(cs_pro.fwhm_rpm)) else "—",
                peak_col_label: peak_fmt.format(r.get("peak_amp_csv", 0.0)),
                "Zona ISO": iso_eval.zone if iso_eval is not None else "—",
                "API 684": ("✓" if api_pro is not None and api_pro.compliant else "✗") if api_pro is not None else "—",
            })
        summary = pd.DataFrame(rows)
    else:
        summary = pd.DataFrame(legacy_records)
        summary["RPM candidata"] = summary["RPM candidata"].round(0)
        summary["Amp dominante"] = summary["Amp dominante"].round(3)
        summary["Delta fase"] = summary["Delta fase"].round(1)
        summary["Max amp"] = summary["Max amp"].round(3)

    st.dataframe(summary, width="stretch", hide_index=True)

    # =========================================================
    # Diagnóstico comparativo Cat IV
    # =========================================================
    if use_rotordyn_pro:
        diag = build_bode_compare_diagnostics_rotordyn(
            records=records_chrono,
            operating_rpm=operating_rpm,
            machine_group=machine_group,
        )
    else:
        # Legacy fallback
        base = legacy_records[0]
        last = legacy_records[-1]
        delta_amp = float(last["Amp dominante"] - base["Amp dominante"])
        delta_rpm = float(last["RPM candidata"] - base["RPM candidata"])
        delta_phase = float(last["Delta fase"] - base["Delta fase"])
        diag = {
            "headline": "Comparación multi-fecha Bode de amplitud, fase y respuesta modal",
            "detail": (
                f"Se compararon {len(legacy_records)} corridas Bode. Entre la corrida base "
                f"({base['Archivo']}) y la más reciente ({last['Archivo']}) se observa una "
                f"variación de {delta_amp:+.3f} en amplitud dominante, desplazamiento de "
                f"{delta_rpm:+.0f} rpm y cambio de fase dominante de {delta_phase:+.1f}°."
            ),
            "action": (
                "Correlacionar las corridas Bode con Polar Plot y órbitas 1X.\n"
                "Verificar si la velocidad candidata se mantiene, migra o incrementa "
                "amplitud entre fechas.\n"
                "Usar la corrida más estable como línea base histórica."
            ),
        }

    st.markdown("### Diagnóstico comparativo automático")
    st.markdown(f"**{diag['headline']}**")
    st.write(diag["detail"])
    st.write(diag["action"])

    # =========================================================
    # Notes para reporte: prosa cronológica por corrida
    # =========================================================
    if use_rotordyn_pro:
        amp_unit_common = records_chrono[0].get("amp_unit", y_unit)
        prose_lines = []
        for r in records_chrono:
            cs_pro = r.get("primary_critical")
            api_pro = r.get("primary_api684")
            iso_eval = r.get("iso_eval")
            date_str = (
                pd.Timestamp(r["ts_start"]).strftime("%d %b %Y")
                if r["ts_start"] is not None else r["label"]
            )

            if cs_pro is not None and iso_eval is not None and api_pro is not None:
                amp_str = (
                    f"{r.get('peak_amp_csv', 0.0):.3f} {amp_unit_common}"
                    if "mil" in amp_unit_common.lower()
                    else f"{r.get('peak_amp_csv', 0.0):.1f} {amp_unit_common}"
                )
                q_str = f"{cs_pro.q_factor:.2f}" if np.isfinite(cs_pro.q_factor) else "—"
                compliant_str = "conforme API 684" if api_pro.compliant else "NO conforme API 684"
                prose_lines.append(
                    f"La corrida del {date_str} ({r['label']}) reportó velocidad crítica en "
                    f"{cs_pro.rpm:.0f} rpm con factor Q de {q_str} y amplitud pico de {amp_str}, "
                    f"clasificada en zona {iso_eval.zone} de ISO 20816-2 y {compliant_str}."
                )
            else:
                prose_lines.append(
                    f"La corrida del {date_str} ({r['label']}) no presenta velocidad crítica "
                    f"detectable bajo los criterios automáticos."
                )

        prose_summary = "\n\n".join(prose_lines)

        notes = (
            f"{diag['detail']}\n\n"
            f"Síntesis cronológica de las corridas Bode analizadas:\n\n"
            f"{prose_summary}\n\n"
            f"{diag['action']}"
        )
    else:
        summary_lines = [
            f"- {r['Archivo']}: candidato {r['RPM candidata']:.0f} rpm, amplitud dominante "
            f"{r['Amp dominante']:.3f}, Δfase {r['Delta fase']:.1f}°, máximo {r['Max amp']:.3f}."
            for r in legacy_records
        ]
        notes = (
            _build_bode_report_notes(diag)
            + "\n\nResumen comparativo de corridas:\n"
            + "\n".join(summary_lines)
        )

    # =========================================================
    # HD export: send-to-report + download PNG (botones lado a lado)
    # =========================================================
    png_bytes, png_error = build_export_png_bytes(fig, diag)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Enviar comparativo Bode a reporte", key="wm_bode_compare_report_btn"):
            ensure_report_state()
            append_report_item_and_persist({
                "type": "bode_compare",
                "title": "Bode Plot · Comparación multi-fecha",
                "notes": notes,
                "image_bytes": png_bytes,
            })
            st.success("Comparativo Bode enviado al reporte.")
    with c2:
        if png_bytes is not None:
            st.download_button(
                "Descargar PNG comparativo Bode HD",
                data=png_bytes,
                file_name="bode_compare_hd.png",
                mime="image/png",
                key="wm_bode_compare_download_btn",
                width="stretch",
            )
        elif png_error:
            st.caption(f"Export PNG no disponible: {png_error}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    require_login()
    ensure_report_state()

    if "wm_bode_selected_ids" not in st.session_state:
        st.session_state.wm_bode_selected_ids = []

    page_header(
        title="Bode Plot",
        subtitle="Amplitude and phase versus speed from Bode CSV files.",
    )

    with st.sidebar:
        render_user_menu()
        st.markdown("---")
        st.markdown("### Upload Bode CSV")
        uploaded_files = st.file_uploader(
            "Upload one or more Bode CSV",
            type=["csv"],
            accept_multiple_files=True,
            key="wm_bode_uploader",
        )

        # Persistencia: si el usuario subió archivos, los guardamos. Si no
        # subió pero hay archivos persistidos de una sesión previa, los
        # reusamos para que sobrevivan a la navegación entre módulos.
        if uploaded_files:
            set_bode_persisted_files(uploaded_files)
            active_files = uploaded_files
        else:
            active_files = get_bode_persisted_files()
            if active_files:
                st.caption(f"{len(active_files)} archivo(s) cargado(s) en sesión.")

        if active_files:
            if st.button("Limpiar archivos cargados", key="wm_bode_clear_files"):
                clear_bode_persisted_files()
                st.session_state.wm_bode_selected_ids = []
                st.rerun()

    if not active_files:
        panel_card(
            title="Carga archivos para comenzar",
            subtitle="Sube uno o varios archivos CSV Bode desde el panel izquierdo.",
            meta_html="",
            chips=[],
        )
        return

    parsed_items, failed_items = parse_uploaded_bode_files(active_files)

    if failed_items:
        for file_name, error_text in failed_items:
            st.warning(f"No pude leer {file_name}: {error_text}")

    if not parsed_items:
        st.error("No se pudo cargar ningún archivo Bode válido.")
        return

    id_to_item = {item["id"]: item for item in parsed_items}
    label_to_id = {
        f"{item['machine']} · {item['point']} · {item['file_name']}": item["id"]
        for item in parsed_items
    }
    selection_labels = list(label_to_id.keys())

    valid_ids = set(id_to_item.keys())

    # Detección de archivos NUEVOS desde el último render
    seen_key = "wm_bode_seen_ids"
    prev_seen = set(st.session_state.get(seen_key, []))
    newly_parsed = [item["id"] for item in parsed_items if item["id"] not in prev_seen]
    st.session_state[seen_key] = [item["id"] for item in parsed_items]

    current_ids = [sid for sid in st.session_state.wm_bode_selected_ids if sid in valid_ids]

    if not current_ids:
        # Primera carga: seleccionar TODOS automáticamente para graficar el lote completo.
        current_ids = [item["id"] for item in parsed_items]
    elif newly_parsed:
        # Archivos nuevos: añadir a la selección sin alterar las elecciones previas del usuario.
        current_ids = current_ids + [nid for nid in newly_parsed if nid not in current_ids]

    st.session_state.wm_bode_selected_ids = current_ids

    default_labels = [label for label, sid in label_to_id.items() if sid in current_ids]

    with st.sidebar:
        st.markdown("### Bode Selection")
        selected_labels = st.multiselect(
            "Bodes to display",
            options=selection_labels,
            default=default_labels,
        )
        st.session_state.wm_bode_selected_ids = [label_to_id[label] for label in selected_labels if label in label_to_id]

        selected_ids_for_sidebar = [sid for sid in st.session_state.wm_bode_selected_ids if sid in id_to_item]
        candidate_frames = [id_to_item[sid]["grouped_df"] for sid in selected_ids_for_sidebar]
        candidate_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.concat([parsed_items[0]["grouped_df"]], ignore_index=True)

        st.markdown("### X Axis Control")
        auto_x = st.checkbox("Auto scale X", value=True)
        x_min_default = float(candidate_df["rpm"].min())
        x_max_default = float(candidate_df["rpm"].max())

        if auto_x:
            x_min = x_min_default
            x_max = x_max_default
        else:
            x_min = st.number_input("Min RPM", value=float(x_min_default), step=10.0)
            x_max = st.number_input("Max RPM", value=float(x_max_default), step=10.0)

        st.markdown("### Phase Mode")
        phase_mode = st.selectbox("Phase display", ["Wrapped Raw 0-360", "Wrapped Smoothed"], index=1)

        st.markdown("### Smoothing")
        smooth_window = st.slider("Median smoothing window", 1, 21, 3, step=2)

        st.markdown("### Critical Speed Detection")
        detect_cs = st.checkbox("Estimate critical speeds (API RP 684 heuristic)", value=True)
        max_critical_speeds = st.selectbox("Max critical speeds", [1, 2], index=1)

        # Asset Instance selector (Ciclo 8) — antes solo seleccionaba profile,
        # ahora selecciona la máquina física específica con sus propios datos.
        instance_state = render_instance_selector(module_name="bode")
        use_rotordyn_pro = instance_state["is_applicable"]
        operating_rpm = instance_state["operating_rpm"]
        machine_group = instance_state["machine_group"]
        active_iso_part = instance_state["iso_part"]
        active_custom_thresholds = instance_state["custom_thresholds"]
        active_profile_label = instance_state["profile_label"]

        if not instance_state["is_applicable"]:
            st.warning(instance_state["applicability_message"])

        st.markdown("### Information Box")
        show_info_box = st.checkbox("Show Bode Information", value=True)

    selected_ids = [sid for sid in st.session_state.wm_bode_selected_ids if sid in id_to_item]

    # ============================================================
    # Ciclo 17.2 — Histórico Bode (multi-snapshot trail overlay)
    # ============================================================
    _bode_inst_id = (
        instance_state.get("instance_id")
        or st.session_state.get("wm_active_instance_id", "")
    )
    _bode_inst = None
    _bode_sensors_map: List[Dict[str, Any]] = []
    if _bode_inst_id:
        try:
            from core.instance_state import get_instance as _bode_get_inst
            _bode_inst = _bode_get_inst(_bode_inst_id)
            if _bode_inst is not None:
                _bode_sensors_map = list(_bode_inst.sensors or [])
        except Exception:
            _bode_inst = None

    def _wm_extract_bode_readings(
        items: List[Dict[str, Any]],
        sensors_map: List[Dict[str, Any]],
        op_speed_rpm: float,
    ) -> List[Dict[str, Any]]:
        """Extrae amp/phase a op_speed + trayectoria por sensor matched."""
        from core.sensor_map import (
            resolve_sensor_for_point as _wm_resolve,
            sensor_label as _wm_slbl,
        )
        out = []
        for it in items:
            try:
                meta = it.get("meta") or {}
                point = str(meta.get("Point Name", "") or it.get("point", "") or "")
                variable = str(meta.get("Variable", "") or it.get("variable", "") or "")
                unit = str(meta.get("Y-Axis Unit", "") or meta.get("Unit", "") or "")
                sensor_match = None
                if sensors_map:
                    sensor_match = _wm_resolve(sensors_map, point, variable, unit)
                if sensor_match is None:
                    continue
                df = it.get("grouped_df")
                if df is None or len(df) == 0:
                    continue

                # Punto operativo: nearest row to op_speed
                _diff = (df["speed"] - op_speed_rpm).abs()
                _row = df.loc[int(_diff.idxmin())]
                amp_at_op = float(_row.get("amp", 0.0))
                phase_at_op = float(_row.get("phase", 0.0)) % 360.0

                # Trayectoria downsampleada a 80 puntos
                _df_sorted = df.sort_values("speed").reset_index(drop=True)
                _N = 80
                if len(_df_sorted) > _N:
                    _idx = np.linspace(0, len(_df_sorted) - 1, _N).astype(int)
                    _df_ds = _df_sorted.iloc[_idx]
                else:
                    _df_ds = _df_sorted
                traj_speed = _df_ds["speed"].astype(float).tolist()
                traj_amp = _df_ds["amp"].astype(float).tolist()
                traj_phase = (_df_ds["phase"].astype(float) % 360.0).tolist()

                out.append({
                    "sensor_label": _wm_slbl(sensor_match),
                    "csv_file": it.get("file_name", ""),
                    "amp_at_op": amp_at_op,
                    "phase_at_op": phase_at_op,
                    "amp_unit": unit or "µm pp",
                    "phase_unit": "deg",
                    "csv_timestamp": str(meta.get("Timestamp", "") or ""),
                    "trajectory_speed": traj_speed,
                    "trajectory_amp": traj_amp,
                    "trajectory_phase": traj_phase,
                })
            except Exception:
                continue
        return out

    _bode_curr_readings: List[Dict[str, Any]] = []
    if selected_ids and _bode_sensors_map:
        try:
            _selected_for_snap = [id_to_item[sid] for sid in selected_ids]
            _bode_curr_readings = _wm_extract_bode_readings(
                _selected_for_snap, _bode_sensors_map, float(operating_rpm),
            )
        except Exception:
            _bode_curr_readings = []

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📚 Histórico Bode")

        try:
            from core.bode_history import (
                save_bode_snapshot,
                list_bode_snapshots,
                load_bode_snapshot,
                delete_bode_snapshot,
                _bode_snapshot_is_identical_to,
            )
            _bode_hist_ok = True
        except Exception as _e:
            _bode_hist_ok = False
            st.caption(f"_(Histórico Bode no disponible: {_e})_")

        if _bode_hist_ok and _bode_inst_id:
            _bode_existing_snaps = list_bode_snapshots(_bode_inst_id)
            st.caption(
                f"{len(_bode_existing_snaps)} snapshot(s) Bode guardado(s) "
                f"para esta unidad."
            )

            if not _bode_curr_readings:
                if not _bode_sensors_map:
                    st.caption(
                        "_(No hay Sensor Map configurado para esta instancia. "
                        "Andá a Machinery Library a configurarlo.)_"
                    )
                elif not selected_ids:
                    st.caption(
                        "_(No hay CSVs Bode cargados todavía. Subilos arriba "
                        "en 'Upload Bode CSV' y volvé.)_"
                    )
                else:
                    st.warning(
                        f"⚠️ Hay {len(selected_ids)} CSV(s) Bode cargado(s) "
                        f"pero ninguno matchea con los {len(_bode_sensors_map)} "
                        f"sensores del Sensor Map de esta unidad."
                    )
                    with st.expander("🔍 Diagnóstico — ver CSVs vs patterns"):
                        st.caption(
                            "El matcher usa el `csv_match_pattern` de cada "
                            "sensor contra el Point name del CSV. Si los "
                            "Point names del DCS no siguen la convención "
                            "API 670 (3X/3Y/4X/4Y), editá los patterns en "
                            "Machinery Library → Mapa de Sensores."
                        )
                        # Listar CSVs con sus Point names
                        _diag_csv_rows = []
                        for sid in selected_ids:
                            it = id_to_item.get(sid)
                            if it is None:
                                continue
                            m = it.get("meta") or {}
                            _diag_csv_rows.append({
                                "Archivo": it.get("file_name", ""),
                                "Point": str(m.get("Point Name", "") or ""),
                                "Variable": str(m.get("Variable", "") or ""),
                                "Unit": str(m.get("Y-Axis Unit", "")
                                            or m.get("Unit", "") or ""),
                            })
                        if _diag_csv_rows:
                            st.markdown("**CSVs Bode cargados:**")
                            st.dataframe(
                                pd.DataFrame(_diag_csv_rows),
                                width="stretch", hide_index=True,
                            )
                        # Listar sensores con sus patterns
                        _diag_sensor_rows = []
                        from core.sensor_map import sensor_label as _diag_slbl
                        for s in _bode_sensors_map:
                            _diag_sensor_rows.append({
                                "Sensor": _diag_slbl(s),
                                "Plano": s.get("plane_label", "") or "",
                                "Tipo": s.get("sensor_type", ""),
                                "Pattern": s.get("csv_match_pattern", "")
                                           or "(vacío)",
                            })
                        if _diag_sensor_rows:
                            st.markdown("**Sensores del mapa con sus patterns:**")
                            st.dataframe(
                                pd.DataFrame(_diag_sensor_rows),
                                width="stretch", hide_index=True,
                            )
                        st.info(
                            "💡 **Tip:** podés usar el wizard automático en "
                            "**Machinery Library → Sugerir patterns desde "
                            "CSVs cargados** para que el sistema proponga "
                            "patterns que matcheen con tus CSVs reales."
                        )
            else:
                with st.expander("📸 Guardar snapshot Bode actual", expanded=False):
                    st.caption(
                        f"Captura amp + fase a {operating_rpm:.0f} rpm + "
                        f"trayectoria completa para {len(_bode_curr_readings)} "
                        f"sensor(es) matched."
                    )
                    _bode_snap_label = st.text_input(
                        "Etiqueta de la corrida",
                        value="",
                        placeholder="Ej. Run-up abril 27",
                        key=f"wm_bode_snap_label_{_bode_inst_id}",
                    )
                    _bode_snap_notes = st.text_area(
                        "Observaciones (opcional)",
                        value="",
                        key=f"wm_bode_snap_notes_{_bode_inst_id}",
                        height=70,
                    )
                    if st.button(
                        "💾 Guardar snapshot Bode",
                        type="primary",
                        width="stretch",
                        key=f"wm_bode_snap_save_{_bode_inst_id}",
                    ):
                        try:
                            sid = save_bode_snapshot(
                                _bode_inst_id,
                                operating_speed_rpm=float(operating_rpm),
                                sensors_data=_bode_curr_readings,
                                corrida_label=_bode_snap_label,
                                notes=_bode_snap_notes,
                            )
                            st.success(f"✓ Snapshot Bode guardado: {sid}")
                            st.rerun()
                        except Exception as _e:
                            st.error(f"No se pudo guardar: {_e}")

            # Selector multi-snapshot
            _selected_bode_cmp_ids: List[str] = []
            if _bode_existing_snaps:
                _curr_by_lbl = {
                    r["sensor_label"]: {"amp": r["amp_at_op"], "phase": r["phase_at_op"]}
                    for r in _bode_curr_readings
                }
                _bode_opt_pairs: List[Tuple[str, str]] = []
                _bode_first_non_current = None
                for _i, s in enumerate(_bode_existing_snaps):
                    _is_current = False
                    if _curr_by_lbl:
                        try:
                            _snap_full = load_bode_snapshot(
                                _bode_inst_id, s["snapshot_id"])
                            if _snap_full is not None:
                                _is_current = _bode_snapshot_is_identical_to(
                                    _snap_full, _curr_by_lbl)
                        except Exception:
                            pass
                    _suffix = " · (corrida actual)" if _is_current else ""
                    _opspeed = s.get("operating_speed_rpm")
                    _opspeed_str = f" @ {_opspeed:.0f}rpm" if _opspeed else ""
                    _lbl = (f"{s['corrida_label'][:28]}{_opspeed_str} "
                            f"({s['timestamp'][:10]}){_suffix}")
                    _bode_opt_pairs.append((s["snapshot_id"], _lbl))
                    if not _is_current and _bode_first_non_current is None:
                        _bode_first_non_current = _lbl

                _bode_opt_lbls = [l for _, l in _bode_opt_pairs]
                _bode_lbl_to_key = {l: k for k, l in _bode_opt_pairs}
                _bode_default_pick = []
                if _bode_first_non_current:
                    _bode_default_pick = [_bode_first_non_current]
                _bode_cmp_state_key = f"wm_bode_cmp_picks_{_bode_inst_id}"
                if _bode_cmp_state_key in st.session_state:
                    _saved = st.session_state[_bode_cmp_state_key]
                    _bode_default_pick = [l for l in _saved if l in _bode_opt_lbls]
                _bode_picked = st.multiselect(
                    "Corridas a superponer en el Bode",
                    options=_bode_opt_lbls,
                    default=_bode_default_pick,
                    key=f"wm_bode_cmp_multi_{_bode_inst_id}",
                    help=(
                        "0 = solo actual; 1 = comparativo simple; "
                        "N = superposición histórica con gradiente "
                        "cronológico sobre amp y fase vs RPM."
                    ),
                )
                st.session_state[_bode_cmp_state_key] = _bode_picked
                _selected_bode_cmp_ids = [
                    _bode_lbl_to_key[l] for l in _bode_picked
                    if l in _bode_lbl_to_key
                ]
                if not _selected_bode_cmp_ids:
                    st.caption("ℹ️ _Solo se mostrará la corrida actual._")
                else:
                    st.caption(
                        f"Se superpondrán **{len(_selected_bode_cmp_ids)}** "
                        f"corrida(s) anterior(es) sobre la actual."
                    )

                # Lista con borrar
                with st.expander(f"🗂️ Gestionar snapshots Bode ({len(_bode_existing_snaps)})"):
                    for s in _bode_existing_snaps:
                        cols_h = st.columns([4, 1])
                        cols_h[0].markdown(
                            f"**{s['corrida_label'][:30]}**  \n"
                            f"_{s['timestamp']} · {s['n_sensors']} sensores · "
                            f"{s.get('operating_speed_rpm', 0):.0f} rpm_"
                        )
                        if cols_h[1].button(
                            "🗑️",
                            key=f"wm_bode_del_{s['snapshot_id']}",
                            help="Borrar este snapshot",
                        ):
                            if delete_bode_snapshot(_bode_inst_id, s["snapshot_id"]):
                                st.success("Borrado.")
                                st.rerun()

            # Persistir picks
            st.session_state["wm_bode_compare_snapshot_ids"] = _selected_bode_cmp_ids
            st.session_state["wm_bode_compare_inst_id"] = _bode_inst_id
    if not selected_ids:
        st.info("Selecciona uno o más Bodes en la barra lateral.")
        return

    selected_items = [id_to_item[sid] for sid in selected_ids]

    # ============================================================
    # Ciclo 17.2 — Comparativo Bode inline (multi-snapshot)
    # ============================================================
    _bode_cmp_ids: List[str] = st.session_state.get(
        "wm_bode_compare_snapshot_ids", []) or []
    if _bode_cmp_ids and _bode_curr_readings:
        try:
            from core.bode_history import (
                load_bode_snapshot,
                phase_shift_classifier,
                amplitude_change_classifier,
                shortest_arc_phase_diff,
            )

            _cmp_rows = []
            _snap_meta_by_id = {}
            for _snap_id in _bode_cmp_ids:
                _snap_full = load_bode_snapshot(_bode_inst_id, _snap_id)
                if _snap_full is None:
                    continue
                _snap_meta_by_id[_snap_id] = _snap_full
                _prev_by_lbl = {
                    str(s.get("sensor_label", "")): s
                    for s in _snap_full.get("sensors", [])
                }
                _snap_label_short = _snap_full.get("corrida_label", _snap_id)[:22]
                _snap_ts = (_snap_full.get("timestamp", "") or "")[:10]

                for r in _bode_curr_readings:
                    _lbl = r["sensor_label"]
                    _prev = _prev_by_lbl.get(_lbl)
                    if _prev is None:
                        continue
                    _prev_amp = float(_prev.get("amp_at_op", 0))
                    _prev_phase = float(_prev.get("phase_at_op", 0))
                    _delta_amp = r["amp_at_op"] - _prev_amp
                    _delta_amp_pct = (_delta_amp / _prev_amp * 100.0) if _prev_amp > 0 else None
                    _delta_phase = shortest_arc_phase_diff(_prev_phase, r["phase_at_op"])
                    _phase_class = phase_shift_classifier(_delta_phase)
                    _amp_class = amplitude_change_classifier(_delta_amp_pct)

                    # Critical speed comparison
                    _prev_cs_rpm = _prev.get("critical_speed_rpm")
                    _prev_q = _prev.get("q_factor")
                    _cs_str = "—"
                    if _prev_cs_rpm:
                        _cs_str = f"{_prev_cs_rpm:.0f} rpm"
                        if _prev_q:
                            _cs_str += f" Q={_prev_q:.2f}"

                    _diag_parts = []
                    if _phase_class == "shift_critical":
                        _diag_parts.append("⚠️ Shift fase crítico")
                    elif _phase_class == "shift_major":
                        _diag_parts.append("⚠️ Shift fase mayor")
                    elif _phase_class == "shift_minor":
                        _diag_parts.append("Shift fase menor")
                    elif _phase_class == "stable":
                        _diag_parts.append("Fase estable")
                    if _delta_amp_pct is not None:
                        if _amp_class in ("amp_critical", "amp_high"):
                            _diag_parts.append(f"Amp {_delta_amp_pct:+.0f}%")
                        elif _amp_class == "amp_up":
                            _diag_parts.append(f"Amp {_delta_amp_pct:+.0f}%")
                        elif _amp_class in ("amp_down_strong", "amp_down"):
                            _diag_parts.append(f"Amp {_delta_amp_pct:+.0f}%")

                    _cmp_rows.append({
                        "Sensor": _lbl,
                        "vs Corrida": f"{_snap_label_short} ({_snap_ts})",
                        "Crítica anterior": _cs_str,
                        "Anterior amp": f"{_prev_amp:.3f} {r['amp_unit']}",
                        "Actual amp": f"{r['amp_at_op']:.3f} {r['amp_unit']}",
                        "Δ amp": (
                            f"{_delta_amp:+.3f} ({_delta_amp_pct:+.1f}%)"
                            if _delta_amp_pct is not None else "—"
                        ),
                        "Anterior fase": f"{_prev_phase:.1f}°",
                        "Actual fase": f"{r['phase_at_op']:.1f}°",
                        "Δ fase": f"{_delta_phase:+.1f}°",
                        "Diagnóstico": " · ".join(_diag_parts) if _diag_parts else "—",
                    })

            if _cmp_rows:
                st.markdown("### 📈 Comparativo Bode — vs corridas anteriores")
                _n_snaps = len(_snap_meta_by_id)
                if _n_snaps == 1:
                    _only = list(_snap_meta_by_id.values())[0]
                    st.caption(
                        f"Comparando contra **{_only.get('corrida_label', '')}** "
                        f"del {_only.get('timestamp', '')[:10]}. Los overlays "
                        f"sobre el Bode permiten ver migración del peak en RPM, "
                        f"cambios del Q factor y deriva de fase a través del modo."
                    )
                else:
                    st.caption(
                        f"Mostrando comparativo contra **{_n_snaps} corridas "
                        f"anteriores**. Cada fila es una combinación "
                        f"(sensor × corrida)."
                    )
                _cmp_disp = pd.DataFrame(_cmp_rows)
                st.dataframe(_cmp_disp, width="stretch", hide_index=True)
        except Exception as _bode_cmp_e:
            st.caption(f"_(Comparativo Bode no disponible: {_bode_cmp_e})_")
    logo_uri = get_logo_data_uri(LOGO_PATH)

    for panel_index, item in enumerate(selected_items):
        render_bode_panel(
            item=item,
            panel_index=panel_index,
            logo_uri=logo_uri,
            smooth_window=smooth_window,
            auto_x=auto_x,
            x_min_global=float(x_min),
            x_max_global=float(x_max),
            phase_mode=phase_mode,
            detect_cs=detect_cs,
            max_critical_speeds=max_critical_speeds,
            show_info_box=show_info_box,
            use_rotordyn_pro=use_rotordyn_pro,
            operating_rpm=float(operating_rpm),
            machine_group=machine_group,
            iso_part=active_iso_part,
            custom_thresholds=active_custom_thresholds,
            profile_label=active_profile_label,
        )

        if panel_index < len(selected_items) - 1:
            st.markdown("---")

    if len(selected_items) >= 2:
        render_bode_compare_section(
            selected_items,
            smooth_window=smooth_window,
            phase_mode=phase_mode,
            detect_cs=detect_cs,
            max_critical_speeds=max_critical_speeds,
            logo_uri=logo_uri,
            use_rotordyn_pro=use_rotordyn_pro,
            operating_rpm=float(operating_rpm),
            machine_group=machine_group,
            iso_part=active_iso_part,
            custom_thresholds=active_custom_thresholds,
        )



if __name__ == "__main__":
    main()
