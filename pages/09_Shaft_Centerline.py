from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from core.auth import render_user_menu, require_login
from core.csv_common import (
    decode_csv_text,
    filter_status_valid,
    find_header_line,
    parse_metadata_block,
)
from core.ui_theme import apply_watermelon_page_style, page_header


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Shaft Centerline", layout="wide")
LOGO_PATH = Path("assets/watermelon_logo.png")

require_login()
apply_watermelon_page_style()


# ============================================================
# SESSION KEYS
# ============================================================
SCL_UPLOAD_FILES_KEY = "wm_scl_upload_files"
REPORT_ITEMS_KEY = "report_items"


# ============================================================
# HELPERS
# ============================================================
class PersistedUploadedFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def read(self) -> bytes:
        return self._data

    def getvalue(self) -> bytes:
        return self._data

    def seek(self, pos: int) -> None:
        return None


def ensure_report_state() -> None:
    if REPORT_ITEMS_KEY not in st.session_state:
        st.session_state[REPORT_ITEMS_KEY] = []


def set_scl_persisted_files(file_objs) -> None:
    packed = []
    for file_obj in file_objs or []:
        if file_obj is None:
            continue
        try:
            data = file_obj.getvalue()
        except Exception:
            try:
                file_obj.seek(0)
            except Exception:
                pass
            data = file_obj.read()
        packed.append(
            {
                "name": getattr(file_obj, "name", "Shaft_Centerline.csv"),
                "data": data,
            }
        )
    st.session_state[SCL_UPLOAD_FILES_KEY] = packed


def get_scl_persisted_files() -> List[PersistedUploadedFile]:
    out: List[PersistedUploadedFile] = []
    for item in st.session_state.get(SCL_UPLOAD_FILES_KEY, []):
        out.append(PersistedUploadedFile(name=item["name"], data=item["data"]))
    return out


def clear_scl_persisted_files() -> None:
    st.session_state.pop(SCL_UPLOAD_FILES_KEY, None)


def get_logo_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window is None or window < 2:
        return series.astype(float).copy()
    return series.astype(float).rolling(window=window, center=True, min_periods=1).mean()


def nearest_row_for_speed(df: pd.DataFrame, speed_value: float) -> pd.Series:
    idx = int((df["speed"] - speed_value).abs().idxmin())
    return df.loc[idx]


def parse_probe_angle_text(text: str) -> Tuple[float, str]:
    text = str(text or "").strip()
    angle = 0.0
    side = ""
    if not text:
        return angle, side

    import re
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if m:
        try:
            angle = float(m.group(1))
        except Exception:
            angle = 0.0

    low = text.lower()
    if "left" in low:
        side = "Left"
    elif "right" in low:
        side = "Right"

    return angle, side


def compute_xy_ranges(
    x: np.ndarray,
    y: np.ndarray,
    auto_scale_xy: bool,
    manual_x_min: float,
    manual_x_max: float,
    manual_y_min: float,
    manual_y_max: float,
) -> Tuple[List[float], List[float]]:
    if auto_scale_xy:
        x_span = max(float(np.nanmax(np.abs(x))) if len(x) else 0.0, 0.1) * 1.20
        y_span = max(float(np.nanmax(np.abs(y))) if len(y) else 0.0, 0.1) * 1.20
        return [-x_span, x_span], [-y_span, y_span]

    x_lo = min(float(manual_x_min), float(manual_x_max))
    x_hi = max(float(manual_x_min), float(manual_x_max))
    y_lo = min(float(manual_y_min), float(manual_y_max))
    y_hi = max(float(manual_y_min), float(manual_y_max))

    if math.isclose(x_lo, x_hi):
        x_hi = x_lo + 1.0
    if math.isclose(y_lo, y_hi):
        y_hi = y_lo + 1.0

    return [x_lo, x_hi], [y_lo, y_hi]


def resolve_clearance_boundary(
    x: np.ndarray,
    y: np.ndarray,
    mode: str,
    center_mode: str,
    manual_cx: float,
    manual_cy: float,
    manual_center_x: float,
    manual_center_y: float,
) -> Dict[str, float]:
    if center_mode == "Origin (0,0)":
        cx0 = 0.0
        cy0 = 0.0
    elif center_mode == "Data Mean":
        cx0 = float(np.nanmean(x)) if len(x) else 0.0
        cy0 = float(np.nanmean(y)) if len(y) else 0.0
    else:
        cx0 = float(manual_center_x)
        cy0 = float(manual_center_y)

    x_rel = x - cx0
    y_rel = y - cy0

    if mode == "Auto":
        cx = max(float(np.nanmax(np.abs(x_rel))) if len(x_rel) else 0.0, 0.1) * 1.08
        cy = max(float(np.nanmax(np.abs(y_rel))) if len(y_rel) else 0.0, 0.1) * 1.08
    else:
        cx = max(abs(float(manual_cx)), 0.001)
        cy = max(abs(float(manual_cy)), 0.001)

    return {
        "center_x": cx0,
        "center_y": cy0,
        "clearance_x": cx,
        "clearance_y": cy,
    }


def build_boundary_curve(center_x: float, center_y: float, clearance_x: float, clearance_y: float) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    bx = center_x + clearance_x * np.cos(theta)
    by = center_y + clearance_y * np.sin(theta)
    return bx, by


def boundary_utilization_pct(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
) -> np.ndarray:
    if clearance_x <= 0 or clearance_y <= 0:
        return np.zeros_like(x, dtype=float)
    x_rel = (x - center_x) / clearance_x
    y_rel = (y - center_y) / clearance_y
    util = np.sqrt(x_rel**2 + y_rel**2) * 100.0
    return util


def detect_early_rub(
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
    warning_util_pct: float = 80.0,
    danger_util_pct: float = 95.0,
) -> Dict[str, Any]:
    util = boundary_utilization_pct(x, y, center_x, center_y, clearance_x, clearance_y)
    warning_idx = np.where(util >= warning_util_pct)[0]
    danger_idx = np.where(util >= danger_util_pct)[0]

    first_warning_speed = float(speed[warning_idx[0]]) if len(warning_idx) else None
    first_danger_speed = float(speed[danger_idx[0]]) if len(danger_idx) else None

    max_util = float(np.max(util)) if len(util) else 0.0

    if max_util >= danger_util_pct:
        severity = "DANGER"
        color = "#dc2626"
        message = "Riesgo alto de pérdida de margen geométrico / rub"
    elif max_util >= warning_util_pct:
        severity = "WARNING"
        color = "#f59e0b"
        message = "Aproximación significativa al límite geométrico"
    else:
        severity = "NORMAL"
        color = "#16a34a"
        message = "Operación dentro del margen geométrico"

    trend_score = 0.0
    if len(util) > 1:
        trend_score = float(util[-1] - util[0])

    return {
        "severity": severity,
        "color": color,
        "message": message,
        "max_util_pct": max_util,
        "first_warning_speed": first_warning_speed,
        "first_danger_speed": first_danger_speed,
        "warning_points": int(len(warning_idx)),
        "contact_points": int(len(danger_idx)),
        "trend_score": trend_score,
    }


def get_semaforo_status(max_util_pct: float) -> Tuple[str, str]:
    if max_util_pct >= 95.0:
        return "DANGER", "#dc2626"
    if max_util_pct >= 80.0:
        return "WARNING", "#f59e0b"
    return "NORMAL", "#16a34a"


def build_export_png_bytes(fig: go.Figure) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        png = pio.to_image(fig, format="png", width=1800, height=1100, scale=2)
        return png, None
    except Exception as e:
        return None, str(e)


def push_report_item(title: str, notes: str, image_bytes: Optional[bytes]) -> None:
    ensure_report_state()
    st.session_state[REPORT_ITEMS_KEY].append(
        {
            "type": "figure",
            "title": title,
            "notes": notes,
            "image_bytes": image_bytes,
        }
    )


def build_shaft_text_diagnostics(
    status: str,
    util_max: float,
    margin_min: float,
    first_warning_speed: Optional[float],
    first_danger_speed: Optional[float],
) -> Dict[str, str]:
    status_up = str(status or "").upper()
    util_max = float(util_max or 0.0)
    margin_min = float(margin_min or 0.0)

    warning_txt = f"{float(first_warning_speed):.0f} rpm" if first_warning_speed is not None else "no identificado"
    danger_txt = f"{float(first_danger_speed):.0f} rpm" if first_danger_speed is not None else "no identificado"

    if status_up == "DANGER" or util_max >= 100.0 or margin_min <= 0.0:
        headline = "Posición de eje fuera del margen geométrico admisible del cojinete"
        detail = (
            f"La trayectoria del eje (shaft centerline) evidencia una condición de operación fuera de la envolvente geométrica del cojinete, "
            f"con una utilización máxima del clearance del {util_max:.1f}% y un margen residual de {margin_min:.1f}%. "
            f"Se identifica ingreso a condición de advertencia alrededor de {warning_txt} y condición severa alrededor de {danger_txt}.\n\n"
            f"Desde el punto de vista rotodinámico, este comportamiento es consistente con un desplazamiento excéntrico elevado del rotor dentro del cojinete, "
            f"lo que sugiere sobrecarga radial efectiva o pérdida de capacidad de centrado hidrodinámico. El patrón observado puede asociarse a desalineación, "
            f"incremento de carga transmitida, pérdida de rigidez del film lubricante, clearances reales diferentes a los asumidos o combinación de estos mecanismos.\n\n"
            f"La pérdida de margen geométrico incrementa de forma significativa la probabilidad de interacción rotor-estator (rub), "
            f"especialmente durante transitorios, cambios de carga o pasos por velocidad crítica."
        )
        action = (
            "Se recomienda como acción prioritaria:\n"
            "- Verificar alineación en condición fría y caliente\n"
            "- Evaluar carga radial real del tren y condición de soporte\n"
            "- Revisar presión, temperatura y viscosidad del sistema de lubricación\n"
            "- Validar clearances reales del cojinete frente a los valores de diseño\n"
            "- Evitar operación sostenida en este régimen hasta completar la evaluación técnica"
        )
    elif status_up == "WARNING" or util_max >= 80.0 or margin_min <= 20.0:
        headline = "Posición de eje con reducción significativa del margen geométrico"
        detail = (
            f"La trayectoria del eje muestra aproximación relevante al límite geométrico del cojinete, "
            f"con una utilización máxima del clearance del {util_max:.1f}% y un margen mínimo remanente de {margin_min:.1f}%. "
            f"Se identifica inicio de condición de advertencia alrededor de {warning_txt}.\n\n"
            f"Este comportamiento sugiere incremento de excentricidad operativa y reducción de la capacidad de centrado del sistema rotor-cojinete. "
            f"Desde la perspectiva rotodinámica, la condición requiere seguimiento cercano para evitar evolución hacia pérdida total de margen y eventual interacción rotor-estator."
        )
        action = (
            "Se recomienda:\n"
            "- Correlacionar esta condición con historial de operación y tendencia de vibración\n"
            "- Revisar alineación, carga radial y comportamiento térmico\n"
            "- Confirmar condición de lubricación y estabilidad del film\n"
            "- Mantener seguimiento estrecho antes de extender operación en este régimen"
        )
    else:
        headline = "Posición de eje dentro del margen geométrico esperado"
        detail = (
            f"La trayectoria del eje se mantiene dentro de la envolvente geométrica del cojinete, "
            f"con una utilización máxima del clearance del {util_max:.1f}% y un margen mínimo remanente de {margin_min:.1f}%.\n\n"
            f"Desde el punto de vista rotodinámico, no se observan indicios de pérdida relevante de margen geométrico en la condición analizada. "
            f"La respuesta es compatible con operación estable del sistema rotor-cojinete dentro del rango evaluado."
        )
        action = (
            "Se recomienda:\n"
            "- Mantener seguimiento periódico de la posición de eje\n"
            "- Correlacionar con vibración, fase y variables operativas\n"
            "- Confirmar estabilidad del comportamiento en futuras corridas"
        )

    return {
        "headline": headline,
        "detail": detail,
        "action": action,
    }


def read_scl_csv(file_obj) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    text = decode_csv_text(file_obj, errors="replace")

    lines = text.splitlines()
    if not lines:
        raise ValueError("Archivo vacío.")

    header_idx = find_header_line(
        lines,
        required_signals=("Point Value", "Paired Point Value", "Speed", "Timestamp"),
    )
    if header_idx is None:
        raise ValueError("No se encontró el encabezado real del CSV Shaft Centerline.")

    meta = parse_metadata_block(lines[:header_idx])
    data_text = "\n".join(lines[header_idx:])

    df = pd.read_csv(io.StringIO(data_text), encoding="utf-8-sig")

    required = [
        "Point Value",
        "Value Status",
        "Paired Point Value",
        "Paired Value Status",
        "Speed",
        "Speed Status",
        "Timestamp",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df["Point Value"] = pd.to_numeric(df["Point Value"], errors="coerce")
    df["Paired Point Value"] = pd.to_numeric(df["Paired Point Value"], errors="coerce")
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["Point Value", "Paired Point Value", "Speed", "Timestamp"]).copy()
    df = filter_status_valid(df, ["Value Status", "Paired Value Status", "Speed Status"])

    if df.empty:
        raise ValueError("No quedaron filas válidas después del filtrado.")

    raw_df = df.sort_values(["Speed", "Timestamp"], kind="stable").reset_index(drop=True)

    grouped_df = (
        raw_df.groupby("Speed", as_index=False)
        .agg(
            y_gap=("Point Value", "median"),
            x_gap=("Paired Point Value", "median"),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("Speed", kind="stable")
        .reset_index(drop=True)
        .rename(columns={"Speed": "speed"})
    )

    return meta, raw_df, grouped_df


def uploaded_file_label(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Shaft_Centerline.csv")).name


def uploaded_file_stem(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Shaft_Centerline.csv")).stem


def parse_uploaded_scl_files(files: List[Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    parsed_items: List[Dict[str, Any]] = []
    failed_items: List[Tuple[str, str]] = []

    for file_obj in files:
        try:
            meta, raw_df, grouped_df = read_scl_csv(file_obj)
            label = uploaded_file_label(file_obj)
            machine = meta.get("Machine Name", "-")
            point = meta.get("Point Name", label)
            paired_point = meta.get("Paired Point Name", "-")
            item_id = f"{label}::{machine}::{point}::{paired_point}"

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
                    "paired_point": paired_point,
                    "variable": meta.get("Variable", "-"),
                }
            )
        except Exception as e:
            failed_items.append((uploaded_file_label(file_obj), str(e)))

    return parsed_items, failed_items


def build_scl_figure(
    df: pd.DataFrame,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    logo_uri: Optional[str],
    show_info_box: bool,
    show_rpm_labels: bool,
    marker_stride: int,
    normalize_to_origin: bool,
    x_range: List[float],
    y_range: List[float],
    clearance_center_x: float,
    clearance_center_y: float,
    clearance_x: float,
    clearance_y: float,
    semaforo_status: str,
    semaforo_color: str,
) -> Tuple[go.Figure, Dict[str, float]]:
    gap_unit = meta.get("Gap Unit", "").strip() or "mil"
    speed_unit = meta.get("Speed Unit", "rpm").strip() or "rpm"

    plot_df = df.copy()

    if normalize_to_origin:
        x0 = float(plot_df["x_gap"].iloc[0])
        y0 = float(plot_df["y_gap"].iloc[0])
        plot_df["x_plot"] = plot_df["x_gap"] - x0
        plot_df["y_plot"] = plot_df["y_gap"] - y0
        row_a_x = float(row_a["x_gap"] - x0)
        row_a_y = float(row_a["y_gap"] - y0)
        row_b_x = float(row_b["x_gap"] - x0)
        row_b_y = float(row_b["y_gap"] - y0)
    else:
        plot_df["x_plot"] = plot_df["x_gap"]
        plot_df["y_plot"] = plot_df["y_gap"]
        row_a_x = float(row_a["x_gap"])
        row_a_y = float(row_a["y_gap"])
        row_b_x = float(row_b["x_gap"])
        row_b_y = float(row_b["y_gap"])

    x = plot_df["x_plot"].to_numpy(dtype=float)
    y = plot_df["y_plot"].to_numpy(dtype=float)

    fig = go.Figure()

    bx, by = build_boundary_curve(
        center_x=clearance_center_x,
        center_y=clearance_center_y,
        clearance_x=clearance_x,
        clearance_y=clearance_y,
    )

    fig.add_trace(
        go.Scatter(
            x=bx,
            y=by,
            mode="lines",
            line=dict(color=semaforo_color, width=2.4, dash="dot"),
            hovertemplate=(
                f"Boundary<br>Center X: {clearance_center_x:.3f} {gap_unit}<br>"
                f"Center Y: {clearance_center_y:.3f} {gap_unit}<br>"
                f"Cx: {clearance_x:.3f} {gap_unit}<br>"
                f"Cy: {clearance_y:.3f} {gap_unit}<extra></extra>"
            ),
            showlegend=False,
            name="Boundary",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(width=2.0, color="#5b9cf0"),
            marker=dict(
                size=6,
                color=plot_df["speed"],
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title=speed_unit, thickness=14, len=0.75, y=0.5),
                line=dict(width=0.5, color="rgba(255,255,255,0.35)"),
            ),
            customdata=np.stack([plot_df["speed"]], axis=1),
            hovertemplate=(
                f"X: %{{x:.3f}} {gap_unit}<br>"
                f"Y: %{{y:.3f}} {gap_unit}<br>"
                f"Speed: %{{customdata[0]:.0f}} {speed_unit}<extra></extra>"
            ),
            showlegend=False,
            name="Centerline",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x[0]],
            y=[y[0]],
            mode="markers+text",
            marker=dict(size=11, color="#22c55e", symbol="diamond"),
            text=["START"],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x[-1]],
            y=[y[-1]],
            mode="markers+text",
            marker=dict(size=11, color="#ef4444", symbol="diamond"),
            text=["END"],
            textposition="bottom center",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[row_a_x],
            y=[row_a_y],
            mode="markers",
            marker=dict(size=10, color="#efb08c", line=dict(width=1.2, color="#ffffff")),
            showlegend=False,
            hovertemplate=(
                f"Cursor A<br>X: {row_a_x:.3f} {gap_unit}<br>Y: {row_a_y:.3f} {gap_unit}<br>"
                f"Speed: {int(round(row_a['speed']))} {speed_unit}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[row_b_x],
            y=[row_b_y],
            mode="markers",
            marker=dict(size=10, color="#7ac77b", line=dict(width=1.2, color="#ffffff")),
            showlegend=False,
            hovertemplate=(
                f"Cursor B<br>X: {row_b_x:.3f} {gap_unit}<br>Y: {row_b_y:.3f} {gap_unit}<br>"
                f"Speed: {int(round(row_b['speed']))} {speed_unit}<extra></extra>"
            ),
        )
    )

    if show_rpm_labels:
        stride = max(int(marker_stride), 1)
        label_df = plot_df.iloc[::stride, :]
        fig.add_trace(
            go.Scatter(
                x=label_df["x_plot"],
                y=label_df["y_plot"],
                mode="text",
                text=[f"{int(round(v))}" for v in label_df["speed"]],
                textposition="top center",
                textfont=dict(size=10, color="#334155"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=760,
        xaxis_title=f"Paired probe ({gap_unit})",
        yaxis_title=f"Probe ({gap_unit})",
        xaxis=dict(range=x_range, zeroline=True, showline=True, linecolor="#9ca3af", ticks="outside"),
        yaxis=dict(range=y_range, zeroline=True, showline=True, linecolor="#9ca3af", ticks="outside", scaleanchor="x", scaleratio=1),
        margin=dict(l=60, r=60, t=60, b=60),
        showlegend=False,
    )

    util = boundary_utilization_pct(
        x=plot_df["x_plot"].to_numpy(dtype=float),
        y=plot_df["y_plot"].to_numpy(dtype=float),
        center_x=clearance_center_x,
        center_y=clearance_center_y,
        clearance_x=clearance_x,
        clearance_y=clearance_y,
    )

    diag = {
        "util_max": float(np.max(util)) if len(util) else 0.0,
        "margin_min": max(0.0, 100.0 - (float(np.max(util)) if len(util) else 0.0)),
        "util_a": float(boundary_utilization_pct(np.array([row_a_x]), np.array([row_a_y]), clearance_center_x, clearance_center_y, clearance_x, clearance_y)[0]),
        "util_b": float(boundary_utilization_pct(np.array([row_b_x]), np.array([row_b_y]), clearance_center_x, clearance_center_y, clearance_x, clearance_y)[0]),
    }

    return fig, diag


def _build_scl_report_notes(text_diag: Dict[str, str]) -> str:
    return f"{text_diag['headline']}\n\n{text_diag['detail']}\n\n{text_diag['action']}"



def _add_scl_export_footer(fig: go.Figure, text_diag: Dict[str, str]) -> go.Figure:
    """
    Exportación limpia de la gráfica Shaft Centerline.
    El diagnóstico NO debe ir incrustado en la imagen porque el reporte ya lo coloca debajo.
    """
    export_fig = go.Figure(fig)
    export_fig.update_layout(
        margin=dict(l=70, r=70, t=60, b=80),
        height=900,
    )
    return export_fig


def queue_scl_to_report(meta, fig, title, text_diag, image_bytes=None):
    notes = _build_scl_report_notes(text_diag)
    push_report_item(title=title, notes=notes, image_bytes=image_bytes)


def _scl_prepare_compare_df(
    grouped_df: pd.DataFrame,
    smooth_window: int,
    normalize_to_origin: bool,
    rpm_min_filter: Optional[float],
    rpm_max_filter: Optional[float],
) -> pd.DataFrame:
    df = grouped_df.copy()

    if rpm_min_filter is not None and rpm_max_filter is not None:
        rpm_lo = min(float(rpm_min_filter), float(rpm_max_filter))
        rpm_hi = max(float(rpm_min_filter), float(rpm_max_filter))
        df = df[
            (df["speed"] >= rpm_lo) &
            (df["speed"] <= rpm_hi)
        ].copy()

    df["x_gap"] = smooth_series(df["x_gap"], smooth_window)
    df["y_gap"] = smooth_series(df["y_gap"], smooth_window)

    if normalize_to_origin and len(df) > 0:
        x0 = float(df["x_gap"].iloc[0])
        y0 = float(df["y_gap"].iloc[0])
        df["x_plot"] = df["x_gap"] - x0
        df["y_plot"] = df["y_gap"] - y0
    else:
        df["x_plot"] = df["x_gap"]
        df["y_plot"] = df["y_gap"]
    return df


def _scl_compare_metrics(
    item: Dict[str, Any],
    smooth_window: int,
    normalize_to_origin: bool,
    rpm_min_filter: Optional[float],
    rpm_max_filter: Optional[float],
) -> Dict[str, Any]:
    df = _scl_prepare_compare_df(
        item["grouped_df"],
        smooth_window,
        normalize_to_origin,
        rpm_min_filter,
        rpm_max_filter,
    )

    x = df["x_plot"].to_numpy(dtype=float)
    y = df["y_plot"].to_numpy(dtype=float)
    speed = df["speed"].to_numpy(dtype=float)

    boundary = resolve_clearance_boundary(
        x=x,
        y=y,
        mode="Auto",
        center_mode="Data Mean" if not normalize_to_origin else "Origin (0,0)",
        manual_cx=5.0,
        manual_cy=5.0,
        manual_center_x=0.0,
        manual_center_y=0.0,
    )

    early_rub = detect_early_rub(
        x=x,
        y=y,
        speed=speed,
        center_x=boundary["center_x"],
        center_y=boundary["center_y"],
        clearance_x=boundary["clearance_x"],
        clearance_y=boundary["clearance_y"],
        warning_util_pct=80.0,
        danger_util_pct=95.0,
    )

    max_util = float(early_rub.get("max_util_pct", 0.0) or 0.0)
    min_margin = max(0.0, 100.0 - max_util)
    centroid_x = float(np.mean(x)) if len(x) else 0.0
    centroid_y = float(np.mean(y)) if len(y) else 0.0
    radial_peak = float(np.max(np.sqrt(x**2 + y**2))) if len(x) else 0.0

    ts_start = pd.to_datetime(df["ts_min"], errors="coerce").min() if "ts_min" in df.columns else None
    ts_end = pd.to_datetime(df["ts_max"], errors="coerce").max() if "ts_max" in df.columns else None

    return {
        "label": item["label"],
        "machine": item["machine"],
        "point": item["point"],
        "paired_point": item["paired_point"],
        "df": df,
        "max_util": max_util,
        "min_margin": min_margin,
        "first_warning_speed": early_rub.get("first_warning_speed"),
        "first_danger_speed": early_rub.get("first_danger_speed"),
        "severity": early_rub.get("severity", "NORMAL"),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "radial_peak": radial_peak,
        "ts_start": ts_start,
        "ts_end": ts_end,
    }


def _scl_compare_diagnostic(records: List[Dict[str, Any]]) -> Dict[str, str]:
    ordered = sorted(
        records,
        key=lambda r: pd.Timestamp(r["ts_start"]) if r["ts_start"] is not None else pd.Timestamp.min
    )

    baseline = ordered[0]
    latest = ordered[-1]

    delta_util = float(latest["max_util"] - baseline["max_util"])
    delta_margin = float(latest["min_margin"] - baseline["min_margin"])
    delta_radial = float(latest["radial_peak"] - baseline["radial_peak"])

    baseline_centroid = np.array([baseline["centroid_x"], baseline["centroid_y"]], dtype=float)
    latest_centroid = np.array([latest["centroid_x"], latest["centroid_y"]], dtype=float)
    centroid_shift = float(np.linalg.norm(latest_centroid - baseline_centroid))

    all_over_limit = all(float(r["max_util"]) >= 100.0 for r in ordered)
    all_zero_margin = all(float(r["min_margin"]) <= 0.0 for r in ordered)
    latest_critical = float(latest["max_util"]) >= 100.0 or float(latest["min_margin"]) <= 0.0

    deterioration_score = 0
    if delta_util > 5.0:
        deterioration_score += 1
    if delta_margin < -5.0:
        deterioration_score += 1
    if delta_radial > 0.2:
        deterioration_score += 1
    if centroid_shift > 0.1:
        deterioration_score += 1

    improvement_score = 0
    if delta_util < -5.0:
        improvement_score += 1
    if delta_margin > 5.0:
        improvement_score += 1
    if delta_radial < -0.2:
        improvement_score += 1

    if all_over_limit or all_zero_margin:
        trend_class = "condición crítica sostenida"
        headline = "Comparación multi-fecha con condición crítica sostenida del sistema rotor-cojinete"
        trend_sentence = (
            "Todas las corridas analizadas muestran operación fuera del margen geométrico admisible del cojinete, "
            "por lo que no se trata de un evento aislado sino de una condición persistente del sistema."
        )
    elif latest_critical and deterioration_score >= 2:
        trend_class = "deterioro progresivo hacia condición crítica"
        headline = "Comparación multi-fecha con deterioro progresivo hacia condición crítica"
        trend_sentence = (
            "La corrida más reciente evidencia empeoramiento respecto a la línea base, "
            "con reducción adicional del margen geométrico y mayor compromiso dinámico del eje dentro del cojinete."
        )
    elif deterioration_score >= 2:
        trend_class = "deterioro progresivo"
        headline = "Comparación multi-fecha con deterioro progresivo de la condición rotodinámica"
        trend_sentence = (
            "La comparación secuencial evidencia una tendencia desfavorable, compatible con incremento de excentricidad operativa "
            "y pérdida de capacidad de centrado hidrodinámico."
        )
    elif improvement_score >= 2 and not latest_critical:
        trend_class = "mejora parcial"
        headline = "Comparación multi-fecha con mejora parcial respecto a la condición base"
        trend_sentence = (
            "La corrida más reciente muestra reducción del compromiso geométrico frente a la línea base; "
            "sin embargo, la condición aún debe validarse contra criterios de aceptación del sistema."
        )
    else:
        trend_class = "cambio moderado"
        headline = "Comparación multi-fecha con cambios operativos medibles en la trayectoria del eje"
        trend_sentence = (
            "No se identifica una variación concluyente compatible con deterioro progresivo severo, "
            "pero sí cambios medibles en la posición del eje y en la respuesta geométrica del cojinete."
        )

    detail = (
        f"Se compararon {len(ordered)} corridas de shaft centerline correspondientes a diferentes fechas de adquisición. "
        f"La comparación entre la corrida base ({baseline['label']}) y la más reciente ({latest['label']}) muestra una variación de "
        f"{delta_util:+.1f} puntos porcentuales en la utilización máxima del clearance, "
        f"{delta_margin:+.1f} puntos en el margen geométrico remanente y "
        f"{delta_radial:+.3f} en el desplazamiento radial máximo.\n\n"
        f"El desplazamiento del centro medio de la trayectoria entre ambas corridas es de {centroid_shift:.3f}, "
        f"parámetro útil para evaluar migración del eje dentro del cojinete y cambios en la condición de centrado hidrodinámico. "
        f"En clasificación global, la tendencia observada corresponde a: {trend_class}.\n\n"
        f"{trend_sentence}\n\n"
        f"Desde el punto de vista de dinámica del rotor, una migración sostenida del centerline acompañada por incremento de utilización de clearance "
        f"es consistente con aumento de excentricidad operativa, modificación de la carga radial efectiva, cambios en la rigidez del film lubricante, "
        f"variación de clearances reales o alteraciones en alineación y condición de soporte."
    )

    if all_over_limit or latest_critical:
        action = (
            "Se recomienda:\n"
            "- Tratar la condición comparativa como hallazgo de alta criticidad\n"
            "- Contrastar las corridas contra condición base de aceptación o condición post-mantenimiento\n"
            "- Correlacionar el cambio del centerline con carga, temperatura, lubricación, vibración y fase\n"
            "- Verificar alineación, condición de soporte y clearances reales del cojinete\n"
            "- Restringir operación sostenida en el régimen comprometido hasta completar evaluación técnica"
        )
    else:
        action = (
            "Se recomienda:\n"
            "- Mantener seguimiento multi-fecha para confirmar si la tendencia es progresiva o dependiente del régimen operativo\n"
            "- Correlacionar el cambio de centerline con carga, temperatura, lubricación y vibración\n"
            "- Validar la condición frente a la línea base de aceptación del equipo"
        )

    return {
        "headline": headline,
        "detail": detail,
        "action": action,
    }


def render_scl_panel(
    item: Dict[str, Any],
    panel_index: int,
    logo_uri: Optional[str],
    smooth_window: int,
    show_info_box: bool,
    show_rpm_labels_global: bool,
    marker_stride_global: int,
    normalize_to_origin: bool,
    clearance_mode: str,
    clearance_center_mode: str,
    manual_center_x: float,
    manual_center_y: float,
    manual_clearance_x: float,
    manual_clearance_y: float,
    auto_scale_xy: bool,
    manual_x_min: float,
    manual_x_max: float,
    manual_y_min: float,
    manual_y_max: float,
    early_rub_warning_pct: int,
    early_rub_danger_pct: int,
    rpm_min_filter: Optional[float],
    rpm_max_filter: Optional[float],
) -> None:
    meta = item["meta"]
    raw_df = item["raw_df"]
    grouped_df = item["grouped_df"].copy()

    grouped_df["x_gap"] = smooth_series(grouped_df["x_gap"], smooth_window)
    grouped_df["y_gap"] = smooth_series(grouped_df["y_gap"], smooth_window)

    display_df = grouped_df.copy()

    if rpm_min_filter is not None and rpm_max_filter is not None:
        rpm_lo = min(float(rpm_min_filter), float(rpm_max_filter))
        rpm_hi = max(float(rpm_min_filter), float(rpm_max_filter))
        display_df = display_df[
            (display_df["speed"] >= rpm_lo) &
            (display_df["speed"] <= rpm_hi)
        ].copy()

    if display_df.empty:
        st.warning(f"Panel {panel_index + 1}: no hay datos válidos en el rango RPM seleccionado.")
        return

    speed_min = int(display_df["speed"].min())
    speed_max = int(display_df["speed"].max())

    c1, c2 = st.columns(2)
    with c1:
        cursor_a_speed = st.slider(
            f"Cursor A (RPM) · Panel {panel_index + 1}",
            speed_min,
            speed_max,
            speed_min,
            key=f"scl_cursor_a_{panel_index}_{item['id']}",
        )
    with c2:
        cursor_b_speed = st.slider(
            f"Cursor B (RPM) · Panel {panel_index + 1}",
            speed_min,
            speed_max,
            speed_max,
            key=f"scl_cursor_b_{panel_index}_{item['id']}",
        )

    row_a = nearest_row_for_speed(display_df, cursor_a_speed)
    row_b = nearest_row_for_speed(display_df, cursor_b_speed)

    if normalize_to_origin:
        base_x = float(display_df["x_gap"].iloc[0])
        base_y = float(display_df["y_gap"].iloc[0])
        x_plot = (display_df["x_gap"] - base_x).to_numpy(dtype=float)
        y_plot = (display_df["y_gap"] - base_y).to_numpy(dtype=float)
    else:
        x_plot = display_df["x_gap"].to_numpy(dtype=float)
        y_plot = display_df["y_gap"].to_numpy(dtype=float)

    x_range, y_range = compute_xy_ranges(
        x=x_plot,
        y=y_plot,
        auto_scale_xy=auto_scale_xy,
        manual_x_min=manual_x_min,
        manual_x_max=manual_x_max,
        manual_y_min=manual_y_min,
        manual_y_max=manual_y_max,
    )

    boundary = resolve_clearance_boundary(
        x=x_plot,
        y=y_plot,
        mode=clearance_mode,
        center_mode=clearance_center_mode,
        manual_cx=manual_clearance_x,
        manual_cy=manual_clearance_y,
        manual_center_x=manual_center_x,
        manual_center_y=manual_center_y,
    )

    early_rub = detect_early_rub(
        x=x_plot,
        y=y_plot,
        speed=display_df["speed"].to_numpy(dtype=float),
        center_x=boundary["center_x"],
        center_y=boundary["center_y"],
        clearance_x=boundary["clearance_x"],
        clearance_y=boundary["clearance_y"],
        warning_util_pct=float(early_rub_warning_pct),
        danger_util_pct=float(early_rub_danger_pct),
    )

    semaforo_status, semaforo_color = get_semaforo_status(early_rub["max_util_pct"])

    machine = meta.get("Machine Name", "-")
    point = meta.get("Point Name", "-")
    paired_point = meta.get("Paired Point Name", "-")
    variable = meta.get("Variable", "-")
    speed_unit = meta.get("Speed Unit", "rpm")
    gap_unit = meta.get("Gap Unit", "mil")

    probe_angle, probe_side = parse_probe_angle_text(meta.get("Probe Angle", ""))
    paired_angle, paired_side = parse_probe_angle_text(meta.get("Paired Probe Angle", ""))

    st.markdown(f"### Shaft Centerline {panel_index + 1} · {machine}")
    st.caption(
        f"{point} / {paired_point} | Variable: {variable} | "
        f"Probe Angles: {probe_angle:.0f}° {probe_side} / {paired_angle:.0f}° {paired_side} | "
        f"Visible Speed Range: {int(display_df['speed'].min())} - {int(display_df['speed'].max())} {speed_unit}"
    )

    fig, diag = build_scl_figure(
        df=display_df,
        meta=meta,
        row_a=row_a,
        row_b=row_b,
        logo_uri=logo_uri,
        show_info_box=show_info_box,
        show_rpm_labels=show_rpm_labels_global,
        marker_stride=marker_stride_global,
        normalize_to_origin=normalize_to_origin,
        x_range=x_range,
        y_range=y_range,
        clearance_center_x=boundary["center_x"],
        clearance_center_y=boundary["center_y"],
        clearance_x=boundary["clearance_x"],
        clearance_y=boundary["clearance_y"],
        semaforo_status=semaforo_status,
        semaforo_color=semaforo_color,
    )

    text_diag = build_shaft_text_diagnostics(
        status=semaforo_status,
        util_max=diag["util_max"],
        margin_min=diag["margin_min"],
        first_warning_speed=early_rub["first_warning_speed"],
        first_danger_speed=early_rub["first_danger_speed"],
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False},
        key=f"wm_scl_plot_{panel_index}_{item['id']}",
    )

    with st.expander(f"Diagnóstico automático · Panel {panel_index + 1}", expanded=True):
        st.markdown(f"**{text_diag['headline']}**")
        st.write(text_diag["detail"])
        st.write(text_diag["action"])

    title = f"Shaft Centerline {panel_index + 1} — {machine} — {point} / {paired_point}"
    notes = _build_scl_report_notes(text_diag)
    export_fig = _add_scl_export_footer(fig, text_diag)

    b1, b2 = st.columns(2)
    with b1:
        png_bytes, png_error = build_export_png_bytes(export_fig)
        if st.button("Enviar panel a reporte", key=f"scl_report_btn_{panel_index}_{item['id']}"):
            push_report_item(title=title, notes=notes, image_bytes=png_bytes)
            st.success("Panel individual enviado al reporte.")
    with b2:
        if png_bytes is not None:
            st.download_button(
                "Descargar PNG panel",
                data=png_bytes,
                file_name=f"{item['file_stem']}_shaft_centerline_hd.png",
                mime="image/png",
                key=f"scl_dl_btn_{panel_index}_{item['id']}",
                width="stretch",
            )
        elif png_error:
            st.warning(f"No fue posible generar PNG: {png_error}")


def render_scl_compare_section(
    items: List[Dict[str, Any]],
    *,
    smooth_window: int,
    normalize_to_origin: bool,
    rpm_min_filter: Optional[float] = None,
    rpm_max_filter: Optional[float] = None,
    clearance_mode: str = "Auto",
    clearance_center_mode: str = "Origin (0,0)",
    manual_center_x: float = 0.0,
    manual_center_y: float = 0.0,
    manual_clearance_x: float = 5.0,
    manual_clearance_y: float = 5.0,
    auto_scale_xy: bool = True,
    manual_x_min: float = -10.0,
    manual_x_max: float = 10.0,
    manual_y_min: float = -10.0,
    manual_y_max: float = 10.0,
) -> None:
    if len(items) < 2:
        return

    compare_records = [
        _scl_compare_metrics(
            item,
            smooth_window=smooth_window,
            normalize_to_origin=normalize_to_origin,
            rpm_min_filter=rpm_min_filter,
            rpm_max_filter=rpm_max_filter,
        )
        for item in items
    ]

    compare_records = sorted(
        compare_records,
        key=lambda r: pd.Timestamp(r["ts_start"]) if r["ts_start"] is not None else pd.Timestamp.min
    )

    st.markdown("---")
    st.markdown("## Comparación multi-fecha · Shaft Centerline")

    fig = go.Figure()

    # Envolvente visual común para el comparativo multi-fecha.
    # Usa todos los puntos comparados para construir una referencia geométrica similar a los paneles individuales.
    valid_dfs = [rec["df"] for rec in compare_records if not rec["df"].empty]
    if valid_dfs:
        all_x = np.concatenate([df["x_plot"].to_numpy(dtype=float) for df in valid_dfs])
        all_y = np.concatenate([df["y_plot"].to_numpy(dtype=float) for df in valid_dfs])

        x_range, y_range = compute_xy_ranges(
            x=all_x,
            y=all_y,
            auto_scale_xy=auto_scale_xy,
            manual_x_min=manual_x_min,
            manual_x_max=manual_x_max,
            manual_y_min=manual_y_min,
            manual_y_max=manual_y_max,
        )

        boundary = resolve_clearance_boundary(
            x=all_x,
            y=all_y,
            mode=clearance_mode,
            center_mode=clearance_center_mode,
            manual_cx=manual_clearance_x,
            manual_cy=manual_clearance_y,
            manual_center_x=manual_center_x,
            manual_center_y=manual_center_y,
        )

        bx, by = build_boundary_curve(
            center_x=boundary["center_x"],
            center_y=boundary["center_y"],
            clearance_x=boundary["clearance_x"],
            clearance_y=boundary["clearance_y"],
        )

        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                name="Clearance / Bearing envelope",
                line=dict(color="#dc2626", width=2.4, dash="dot"),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    palette = ["#2563eb", "#16a34a", "#9333ea", "#ea580c", "#dc2626", "#0891b2", "#7c3aed", "#0f766e"]

    for idx, rec in enumerate(compare_records):
        df = rec["df"]
        color = palette[idx % len(palette)]
        date_label = "sin fecha"
        if rec["ts_start"] is not None:
            date_label = pd.Timestamp(rec["ts_start"]).strftime("%Y-%m-%d %H:%M")

        fig.add_trace(
            go.Scatter(
                x=df["x_plot"],
                y=df["y_plot"],
                mode="lines+markers",
                name=f"{date_label} · {rec['label']}",
                line=dict(width=2.2, color=color),
                marker=dict(size=5, color=color),
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Shaft Centerline · Comparación multi-fecha",
        xaxis_title="Paired probe (mil)",
        yaxis_title="Probe (mil)",
        height=720,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    fig.update_xaxes(
        range=x_range,
        zeroline=True,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        gridcolor="rgba(148,163,184,0.20)",
    )
    fig.update_yaxes(
        range=y_range,
        zeroline=True,
        showline=True,
        linecolor="#9ca3af",
        ticks="outside",
        gridcolor="rgba(148,163,184,0.20)",
        scaleanchor="x",
        scaleratio=1,
    )

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="wm_scl_compare_plot")

    summary_rows = []
    for rec in compare_records:
        summary_rows.append(
            {
                "Archivo": rec["label"],
                "Fecha inicio": pd.Timestamp(rec["ts_start"]).strftime("%Y-%m-%d %H:%M") if rec["ts_start"] is not None else "—",
                "Fecha fin": pd.Timestamp(rec["ts_end"]).strftime("%Y-%m-%d %H:%M") if rec["ts_end"] is not None else "—",
                "Max util %": round(rec["max_util"], 2),
                "Min margin %": round(rec["min_margin"], 2),
                "1st warning": "—" if rec["first_warning_speed"] is None else round(float(rec["first_warning_speed"]), 0),
                "1st danger": "—" if rec["first_danger_speed"] is None else round(float(rec["first_danger_speed"]), 0),
                "Radial peak": round(rec["radial_peak"], 4),
                "Centro X": round(rec["centroid_x"], 4),
                "Centro Y": round(rec["centroid_y"], 4),
                "Severity": rec["severity"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, width="stretch", hide_index=True)

    diag = _scl_compare_diagnostic(compare_records)
    st.markdown("### Diagnóstico comparativo automático")
    st.markdown(f"**{diag['headline']}**")
    st.write(diag["detail"])
    st.write(diag["action"])

    notes = (
        f"{diag['headline']}\n\n"
        f"{diag['detail']}\n\n"
        f"{diag['action']}\n\n"
        f"--- RESUMEN ---\n{summary_df.to_string(index=False)}"
    )

    png_bytes, png_error = build_export_png_bytes(fig)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Enviar comparativo a reporte", key="wm_scl_compare_report_btn"):
            push_report_item(
                title="Shaft Centerline · Comparación multi-fecha",
                notes=notes,
                image_bytes=png_bytes,
            )
            st.success("Comparación multi-fecha enviada al reporte.")
    with c2:
        if png_bytes is not None:
            st.download_button(
                "Descargar PNG comparativo",
                data=png_bytes,
                file_name="shaft_centerline_compare.png",
                mime="image/png",
                key="wm_scl_compare_dl_btn",
                width="stretch",
            )
        elif png_error:
            st.warning(f"No fue posible generar PNG del comparativo: {png_error}")


def main():
    ensure_report_state()

    page_header(
        title="Shaft Centerline",
        subtitle="Centerline position from paired X/Y gap probes versus speed.",
    )

    with st.sidebar:
        render_user_menu()
        st.markdown("---")
        st.markdown("### Shaft Centerline input")

        uploaded_files_new = st.file_uploader(
            "Cargar CSV Shaft Centerline",
            type=["csv"],
            accept_multiple_files=True,
            key="wm_scl_file_uploader",
        )

        if uploaded_files_new:
            set_scl_persisted_files(uploaded_files_new)

        active_files = get_scl_persisted_files()

        col1, col2 = st.columns(2)
        with col1:
            if active_files:
                st.caption(f"Archivos Shaft activos: {len(active_files)}")
            else:
                st.caption("No hay archivos Shaft cargados")
        with col2:
            if st.button("Limpiar archivos Shaft", key="wm_scl_clear_file_btn"):
                clear_scl_persisted_files()
                st.rerun()

        st.markdown("### Global Controls")
        smooth_window = st.slider("Gap smoothing", 1, 11, 3, step=2)
        show_info_box = st.checkbox("Show information box", value=True)
        show_rpm_labels = st.checkbox("Show RPM labels", value=True)
        marker_stride = st.slider("RPM label step", 10, 150, 45, step=5)
        normalize_to_origin = st.checkbox("Normalize to first point", value=False)

        st.markdown("### RPM filter")
        rpm_filter_enabled = st.checkbox("Filtrar rango RPM", value=False)
        rpm_min_filter_ui = st.number_input("RPM inicio", value=0.0, step=100.0, format="%.0f")
        rpm_max_filter_ui = st.number_input("RPM fin", value=100000.0, step=100.0, format="%.0f")

        rpm_min_filter = rpm_min_filter_ui if rpm_filter_enabled else None
        rpm_max_filter = rpm_max_filter_ui if rpm_filter_enabled else None

        st.markdown("### Boundary controls")
        clearance_mode = st.selectbox("Boundary mode", ["Auto", "Manual"], index=0)
        clearance_center_mode = st.selectbox("Boundary center", ["Origin (0,0)", "Data Mean", "Manual"], index=0)

        manual_center_x = st.number_input("Boundary center X", value=0.0, step=0.1, format="%.3f")
        manual_center_y = st.number_input("Boundary center Y", value=0.0, step=0.1, format="%.3f")

        manual_clearance_x = st.number_input("Clearance X (Cx)", value=5.0, min_value=0.001, step=0.1, format="%.3f")
        manual_clearance_y = st.number_input("Clearance Y (Cy)", value=5.0, min_value=0.001, step=0.1, format="%.3f")

        st.markdown("### Axis controls")
        auto_scale_xy = st.checkbox("Auto X/Y", value=True)
        manual_x_min = st.number_input("X min", value=-10.0, step=0.5, format="%.3f")
        manual_x_max = st.number_input("X max", value=10.0, step=0.5, format="%.3f")
        manual_y_min = st.number_input("Y min", value=-10.0, step=0.5, format="%.3f")
        manual_y_max = st.number_input("Y max", value=10.0, step=0.5, format="%.3f")

        st.markdown("### Early Rub Detection")
        early_rub_warning_pct = st.slider("Warning utilization %", min_value=50, max_value=98, value=80, step=1)
        early_rub_danger_pct = st.slider("Danger utilization %", min_value=60, max_value=100, value=95, step=1)

    if not active_files:
        st.info("Carga uno o varios archivos CSV de Shaft Centerline desde el panel izquierdo.")
        return

    parsed_items, failed_items = parse_uploaded_scl_files(active_files)

    if failed_items:
        for file_name, error_text in failed_items:
            st.warning(f"{file_name}: {error_text}")

    if not parsed_items:
        st.info("No se pudo procesar ningún archivo válido.")
        return

    logo_uri = get_logo_data_uri(LOGO_PATH)

    for panel_index, item in enumerate(parsed_items):
        render_scl_panel(
            item=item,
            panel_index=panel_index,
            logo_uri=logo_uri,
            smooth_window=smooth_window,
            show_info_box=show_info_box,
            show_rpm_labels_global=show_rpm_labels,
            marker_stride_global=marker_stride,
            normalize_to_origin=normalize_to_origin,
            clearance_mode=clearance_mode,
            clearance_center_mode=clearance_center_mode,
            manual_center_x=manual_center_x,
            manual_center_y=manual_center_y,
            manual_clearance_x=manual_clearance_x,
            manual_clearance_y=manual_clearance_y,
            auto_scale_xy=auto_scale_xy,
            manual_x_min=manual_x_min,
            manual_x_max=manual_x_max,
            manual_y_min=manual_y_min,
            manual_y_max=manual_y_max,
            early_rub_warning_pct=early_rub_warning_pct,
            early_rub_danger_pct=early_rub_danger_pct,
            rpm_min_filter=rpm_min_filter,
            rpm_max_filter=rpm_max_filter,
        )

        if panel_index < len(parsed_items) - 1:
            st.markdown("---")

    if len(parsed_items) >= 2:
        render_scl_compare_section(
            parsed_items,
            smooth_window=smooth_window,
            normalize_to_origin=normalize_to_origin,
            rpm_min_filter=rpm_min_filter,
            rpm_max_filter=rpm_max_filter,
            clearance_mode=clearance_mode,
            clearance_center_mode=clearance_center_mode,
            manual_center_x=manual_center_x,
            manual_center_y=manual_center_y,
            manual_clearance_x=manual_clearance_x,
            manual_clearance_y=manual_clearance_y,
            auto_scale_xy=auto_scale_xy,
            manual_x_min=manual_x_min,
            manual_x_max=manual_x_max,
            manual_y_min=manual_y_min,
            manual_y_max=manual_y_max,
        )


if __name__ == "__main__":
    main()
