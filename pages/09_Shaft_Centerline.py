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
from core.diagnostics import build_scl_diagnostics_rotordyn
from core.document_vault import get_captured_parameters, list_documents
from core.profile_state import render_profile_selector  # legacy compat
from core.instance_selector import render_instance_selector
from core.scl_diagnostics import (
    compare_centerline_migration,
    compute_eccentricity_state,
    derive_radial_clearance_from_vault,
    detect_lift_off_speed,
)
from core.ui_theme import apply_watermelon_page_style, page_header
from core.ai_diagnostic import (  # Ciclo 17.26: interpretación clínica AI
    generate_ai_diagnostic,
    is_ai_available,
)


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
    *,
    clearance_x: Optional[float] = None,
    clearance_y: Optional[float] = None,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """
    Calcula los rangos del plot X/Y. En modo Auto, si hay clearance del
    cojinete disponible, fija la escala a ±1.2× clearance respecto al
    centro (así el círculo de clearance siempre queda visible y el dato
    no se ve aplastado contra los bordes). Si no hay clearance, usa el
    rango de los datos como fallback.
    """
    if auto_scale_xy:
        if clearance_x is not None and clearance_y is not None and clearance_x > 0 and clearance_y > 0:
            # Asegurar que tanto el clearance como los datos quepan
            cx = float(clearance_x)
            cy = float(clearance_y)
            data_max_x = float(np.nanmax(np.abs(x - center_x))) if len(x) else 0.0
            data_max_y = float(np.nanmax(np.abs(y - center_y))) if len(y) else 0.0
            span_x = max(cx, data_max_x) * 1.20
            span_y = max(cy, data_max_y) * 1.20
            return [center_x - span_x, center_x + span_x], [center_y - span_y, center_y + span_y]

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
    """
    Resuelve la geometría del bearing clearance circle.

    center_mode acepta:
      - "Bottom load reference (API 670 / práctica estándar)" — convención
        estándar para máquinas horizontales con carga gravitacional vertical.
        El (0,0) del registro corresponde al muñón en reposo apoyado en la
        babbitt al fondo del cojinete. El bearing center geométrico queda Cr
        (radio del clearance) por encima → (0, +Cr). Esta es la convención
        correcta para cálculo de eccentricity ratio y attitude angle.
      - "Origin (0,0)" — bearing center forzado al origen del data. Solo para
        debug, máquinas verticales o sistemas con calibración no estándar.
      - "Data Mean" — bearing center en el centroide del data. Útil cuando
        el data no fue calibrado al rest position.
      - "Manual" — bearing center especificado por el usuario.
    """
    # Primero determinar Cx, Cy radial (necesario para Bottom load reference)
    if mode == "Manual":
        cx_radial_initial = max(abs(float(manual_cx)), 0.001)
        cy_radial_initial = max(abs(float(manual_cy)), 0.001)
    else:
        # Auto heurístico: estimación basada en datos (fallback)
        cx_radial_initial = max(float(np.nanmax(np.abs(x))) if len(x) else 0.0, 0.1) * 1.08
        cy_radial_initial = max(float(np.nanmax(np.abs(y))) if len(y) else 0.0, 0.1) * 1.08

    if center_mode.startswith("Bottom load reference"):
        # Práctica estándar API 670 para cojinetes hidrodinámicos: bearing
        # center está Cr por encima del rest. Si el registro está normalizado
        # a su origen, rest está en (0,0). Para casos especiales el usuario
        # puede overridear con Manual.
        cx0 = 0.0
        cy0 = float(cy_radial_initial)
    elif center_mode == "Origin (0,0)":
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


def build_eccentricity_ring(
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
    eccentricity_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye un anillo a una fracción dada del clearance (e/c = fraction).
    Útil para superponer los límites de zonas Cat IV: 0.40 / 0.70 / 0.85.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    bx = center_x + clearance_x * eccentricity_fraction * np.cos(theta)
    by = center_y + clearance_y * eccentricity_fraction * np.sin(theta)
    return bx, by


def add_scl_cat_iv_overlay(
    fig: go.Figure,
    *,
    center_x: float,
    center_y: float,
    clearance_x: float,
    clearance_y: float,
    show_rest_marker: bool = True,
    show_load_arrow: bool = True,
) -> None:
    """
    Agrega elementos Cat IV de referencia al plot SCL:
      - Anillos de eccentricity (zonas verde/amarilla/naranja/roja)
      - Marker BEARING CENTER
      - Marker REST (en (0,0) si aplica)
      - Flecha de load direction (gravity, hacia abajo)
    """
    # Anillos de eccentricity Cat IV
    for fraction, color, label in (
        (0.40, "rgba(34, 197, 94, 0.55)", "e/c=0.40"),
        (0.70, "rgba(234, 179, 8, 0.55)", "e/c=0.70"),
        (0.85, "rgba(220, 38, 38, 0.55)", "e/c=0.85"),
    ):
        rx, ry = build_eccentricity_ring(center_x, center_y, clearance_x, clearance_y, fraction)
        fig.add_trace(
            go.Scatter(
                x=rx, y=ry, mode="lines",
                line=dict(width=1.0, color=color, dash="dot"),
                name=label,
                hoverinfo="skip", showlegend=True,
            )
        )

    # BEARING CENTER marker
    fig.add_trace(
        go.Scatter(
            x=[center_x], y=[center_y], mode="markers+text",
            marker=dict(size=11, color="#0f172a", symbol="cross", line=dict(width=2, color="white")),
            text=["BEARING CENTER"], textposition="top right",
            textfont=dict(size=10, color="#0f172a", family="Arial Black"),
            name="Bearing center", hoverinfo="text",
            hovertext=f"Bearing center geométrico ({center_x:.2f}, {center_y:.2f}) mil pp",
            showlegend=False,
        )
    )

    # REST marker (en data origin si bearing center está desplazado)
    if show_rest_marker and abs(center_y) > 0.01:
        fig.add_trace(
            go.Scatter(
                x=[0.0], y=[0.0], mode="markers+text",
                marker=dict(size=10, color="#dc2626", symbol="circle", line=dict(width=2, color="white")),
                text=["REST"], textposition="bottom left",
                textfont=dict(size=10, color="#dc2626", family="Arial Black"),
                name="Rest position", hoverinfo="text",
                hovertext="Posición de reposo (rotor parado, muñón al fondo del cojinete por gravedad)",
                showlegend=False,
            )
        )

    # Flecha de load direction (gravity, hacia abajo desde bearing center)
    if show_load_arrow:
        load_arrow_length = clearance_y * 0.95
        fig.add_annotation(
            x=center_x,
            y=center_y - load_arrow_length,
            ax=center_x,
            ay=center_y,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.8,
            arrowcolor="#475569",
            text="W (load)",
            font=dict(size=10, color="#475569"),
            xshift=8,
        )


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
    # Ciclo 23.155 — anti-OOM: vía core.plot_export.fig_to_png_bytes (decima + scale=1).
    try:
        from core.plot_export import fig_to_png_bytes
        return fig_to_png_bytes(fig, width=1800, height=1100, scale=1)
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
    *,
    # Ciclo 17.3 P4 — overlays históricos (multi-snapshot)
    # Lista de dicts con {label, timestamp, x_gap_at_op,
    # y_gap_at_op, trajectory_speed/x_gap/y_gap, op_speed,
    # eccentricity_ratio, attitude_angle}. Se dibujan en gradient
    # cronológico (azul claro = más viejo, rojo = más reciente)
    # con el operating point destacado y la trayectoria del lift-off
    # como línea suave debajo del actual.
    prev_snapshots: Optional[List[Dict[str, Any]]] = None,
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

    # Cat IV overlay (eccentricity rings + bearing center + rest + load arrow)
    add_scl_cat_iv_overlay(
        fig,
        center_x=clearance_center_x,
        center_y=clearance_center_y,
        clearance_x=clearance_x,
        clearance_y=clearance_y,
    )

    # ============================================================
    # Ciclo 17.3 P4 — Overlays de centerlines históricos
    # ------------------------------------------------------------
    # Para cada snapshot anterior con trayectoria, dibuja la curva
    # de lift-off (X vs Y) en gradiente cronológico (azul claro =
    # más viejo, rojo = más reciente, ámbar intermedio) con
    # opacidad ~0.55. Marker GHOST en el operating point del
    # snapshot. Se dibuja PRIMERO (zorder bajo) para que el actual
    # quede arriba.
    # ============================================================
    if prev_snapshots:
        _snaps_sorted = sorted(
            [s for s in prev_snapshots if s.get("trajectory_x_gap")
             and s.get("trajectory_y_gap")],
            key=lambda s: s.get("timestamp", "") or "",
        )

        def _scl_gradient(idx: int, total: int) -> str:
            if total <= 1:
                return "rgba(148,163,184,0.55)"
            pos = idx / max(1, total - 1)
            stops = [
                (0.00, (125, 211, 252)),   # azul claro
                (0.50, (245, 158,  11)),   # ámbar
                (1.00, (220,  38,  38)),   # rojo
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
            _color = _scl_gradient(_idx, len(_snaps_sorted))
            _lbl = _snap.get("label", "anterior") or "anterior"
            _t_speed = _snap.get("trajectory_speed", []) or []
            _t_x = _snap.get("trajectory_x_gap", []) or []
            _t_y = _snap.get("trajectory_y_gap", []) or []
            if not (len(_t_x) > 1 and len(_t_x) == len(_t_y)):
                continue
            # Si el actual está en modo normalize_to_origin, también
            # normalizar la trayectoria histórica para que sea comparable.
            if normalize_to_origin and len(_t_x) > 0:
                _x0 = float(_t_x[0])
                _y0 = float(_t_y[0])
                _t_x_plot = [float(v) - _x0 for v in _t_x]
                _t_y_plot = [float(v) - _y0 for v in _t_y]
            else:
                _t_x_plot = [float(v) for v in _t_x]
                _t_y_plot = [float(v) for v in _t_y]

            # Trayectoria de lift-off histórica
            fig.add_trace(
                go.Scatter(
                    x=_t_x_plot,
                    y=_t_y_plot,
                    mode="lines",
                    line=dict(width=1.6, color=_color, dash="solid"),
                    opacity=0.55,
                    customdata=np.array(_t_speed).reshape(-1, 1) if _t_speed else None,
                    hovertemplate=(
                        f"<b>{_lbl}</b><br>"
                        f"X: %{{x:.3f}} {gap_unit}<br>"
                        f"Y: %{{y:.3f}} {gap_unit}"
                        + (f"<br>Speed: %{{customdata[0]:.0f}} {speed_unit}"
                           if _t_speed else "")
                        + "<extra></extra>"
                    ),
                    showlegend=False,
                    name=f"SCL {_lbl}",
                )
            )

            # Marker GHOST en el operating point del snapshot
            _x_op = float(_snap.get("x_gap_at_op", 0))
            _y_op = float(_snap.get("y_gap_at_op", 0))
            if normalize_to_origin and len(_t_x) > 0:
                _x_op_plot = _x_op - float(_t_x[0])
                _y_op_plot = _y_op - float(_t_y[0])
            else:
                _x_op_plot = _x_op
                _y_op_plot = _y_op
            _ecc_p = float(_snap.get("eccentricity_ratio", 0))
            _att_p = float(_snap.get("attitude_angle", 0))
            _opspeed_p = _snap.get("op_speed")
            _opspeed_str = (
                f" @ {int(round(_opspeed_p))} rpm"
                if _opspeed_p else ""
            )
            fig.add_trace(
                go.Scatter(
                    x=[_x_op_plot],
                    y=[_y_op_plot],
                    mode="markers+text",
                    marker=dict(
                        size=14, color=_color,
                        symbol="diamond-open",
                        line=dict(width=2.0, color="#0f172a"),
                    ),
                    text=[f"<i>{_lbl[:18]}{_opspeed_str}</i>"],
                    textposition="top center",
                    textfont=dict(size=8.5, color="#0f172a"),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Op {_lbl}</b><br>"
                        f"X: {_x_op_plot:.3f} {gap_unit}<br>"
                        f"Y: {_y_op_plot:.3f} {gap_unit}<br>"
                        f"e/c: {_ecc_p:.3f}<br>"
                        f"attitude: {_att_p:.1f}°<extra></extra>"
                    ),
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
    headline = str(text_diag.get('headline', '') or '').strip()
    detail = str(text_diag.get('detail', '') or '').strip()
    action = str(text_diag.get('action', '') or '').strip()
    # Ciclo 17.3 — narrativa comparativa SCL (cuando hay snapshot)
    comparison_narrative = str(
        text_diag.get('comparison_narrative', '') or ''
    ).strip()

    blocks = []
    if headline:
        blocks.append(headline)
    if detail:
        blocks.append(detail)
    # Inyectar comparativo entre detalle y acciones
    if comparison_narrative:
        blocks.append(comparison_narrative)
    if action:
        blocks.append(action)
    return "\n\n".join(blocks).strip()



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


def _mils_to_gap_unit(value_mils: Optional[float], gap_unit: str) -> Optional[float]:
    """Convierte un clearance radial expresado en MILS a la unidad del gap de
    los datos (la misma en la que vienen las posiciones del muñón del CSV).

    Ciclo 23.153 — FIX crítico de unidades. El clearance del Vault SIEMPRE se
    deriva en mils, pero las máquinas con sondas en MICRAS (ej. SGT300A/B de
    Parex, configuradas en µm) traen las posiciones en µm. Si no se convierte,
    el eccentricity ratio, attitude angle, lift-off y el círculo de clearance
    salen mal (ej. Cd 0.127 mm = 127 µm se graficaba como 5 µm = 5 mils).
        1 mil = 25.4 µm = 0.0254 mm.
    """
    if value_mils is None:
        return None
    u = (gap_unit or "mil").strip().lower()
    if u in ("um", "µm", "μm", "micron", "microns", "micra", "micras",
             "micrometer", "micrometro", "micrómetro", "mic"):
        return float(value_mils) * 25.4
    if u == "mm":
        return float(value_mils) * 0.0254
    return float(value_mils)  # 'mil' o desconocido → asumir mils


def _amp_unit_label(gap_unit: str) -> str:
    """Etiqueta legible de unidad pico-pico para la narrativa diagnóstica."""
    u = (gap_unit or "mil").strip().lower()
    if u in ("um", "µm", "μm", "micron", "microns", "micra", "micras", "mic"):
        return "µm pp"
    if u == "mm":
        return "mm pp"
    return "mil pp"


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
    *,
    vault_clearance_radial_mil: Optional[float] = None,
    vault_params: Optional[Dict[str, Any]] = None,
    vault_doc_ref: Optional[str] = None,
    profile_label: Optional[str] = None,
    operating_rpm: float = 3600.0,
    cr_source: str = "",
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

    # Ciclo 23.153 — FIX de unidades: el clearance del Vault viene en MILS,
    # pero x_plot/y_plot están en la unidad del gap del CSV (µm en SGT300A/B
    # de Parex). Convertir a la MISMA unidad antes de usarlo como clearance,
    # o eccentricity/attitude/lift-off/círculo salen mal.
    _gap_unit_data = meta.get("Gap Unit", "mil")
    vault_cr_in_gap_unit = _mils_to_gap_unit(vault_clearance_radial_mil, _gap_unit_data)

    # Resolver clearance — prioridad:
    #   1. Manual (el usuario siempre puede sobrescribir desde sidebar)
    #   2. Vault (smart default cuando hay datos físicos del cojinete)
    #   3. Heurístico Auto (legacy, basado en datos)
    if clearance_mode == "Manual":
        boundary = resolve_clearance_boundary(
            x=x_plot, y=y_plot,
            mode="Manual",
            center_mode=clearance_center_mode,
            manual_cx=manual_clearance_x,
            manual_cy=manual_clearance_y,
            manual_center_x=manual_center_x,
            manual_center_y=manual_center_y,
        )
        boundary["source"] = "manual (sidebar)"
    elif vault_cr_in_gap_unit is not None:
        boundary = resolve_clearance_boundary(
            x=x_plot, y=y_plot,
            mode="Manual",  # internamente usamos Manual con valores del Vault
            center_mode=clearance_center_mode,
            manual_cx=float(vault_cr_in_gap_unit),
            manual_cy=float(vault_cr_in_gap_unit),
            manual_center_x=manual_center_x,
            manual_center_y=manual_center_y,
        )
        boundary["source"] = f"Vault ({cr_source})"
    else:
        boundary = resolve_clearance_boundary(
            x=x_plot, y=y_plot,
            mode=clearance_mode,  # Auto heurístico
            center_mode=clearance_center_mode,
            manual_cx=manual_clearance_x,
            manual_cy=manual_clearance_y,
            manual_center_x=manual_center_x,
            manual_center_y=manual_center_y,
        )
        boundary["source"] = "auto heurístico (datos)"

    # Auto X/Y ahora consciente del clearance: la escala visible incluye
    # siempre el círculo de clearance del cojinete
    x_range, y_range = compute_xy_ranges(
        x=x_plot,
        y=y_plot,
        auto_scale_xy=auto_scale_xy,
        manual_x_min=manual_x_min,
        manual_x_max=manual_x_max,
        manual_y_min=manual_y_min,
        manual_y_max=manual_y_max,
        clearance_x=boundary.get("clearance_x"),
        clearance_y=boundary.get("clearance_y"),
        center_x=boundary.get("center_x", 0.0),
        center_y=boundary.get("center_y", 0.0),
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

    # Ciclo 17.3 P4 — buscar snapshots SCL elegidos en sidebar y armar
    # lista de prev_snapshots para el panel actual. Cada panel = un
    # bearing (X-Y pair); resolvemos el bearing label a partir del
    # sensor matched al Point/Paired Point del CSV.
    _scl_prev_snapshots_list: List[Dict[str, Any]] = []
    try:
        _scl_cmp_snap_ids = (
            st.session_state.get("wm_scl_compare_snapshot_ids") or []
        )
        if _scl_cmp_snap_ids:
            from core.scl_history import load_scl_snapshot
            from core.sensor_map import (
                resolve_sensor_for_point as _sm_resolve,
            )
            from core.instance_state import get_instance as _sm_get_inst
            _scl_inst_id_local = (
                st.session_state.get("wm_active_instance_id", "")
                or st.session_state.get("wm_scl_compare_inst_id", "")
            )
            if _scl_inst_id_local:
                _inst_obj = _sm_get_inst(_scl_inst_id_local)
                _curr_panel_bearing = None
                if _inst_obj is not None and _inst_obj.sensors:
                    _sensor_match = _sm_resolve(
                        list(_inst_obj.sensors),
                        str(meta.get("Point Name", "") or item.get("point", "")),
                        str(meta.get("Variable", "") or item.get("variable", "")),
                        str(meta.get("Y-Axis Unit", "") or meta.get("Unit", "") or ""),
                    )
                    if _sensor_match is None:
                        _paired = str(meta.get("Paired Point Name", "") or "")
                        if _paired:
                            _sensor_match = _sm_resolve(
                                list(_inst_obj.sensors),
                                _paired,
                                str(meta.get("Variable", "") or ""),
                                str(meta.get("Y-Axis Unit", "") or meta.get("Unit", "") or ""),
                            )
                    if _sensor_match is not None:
                        _curr_panel_bearing = (
                            _sensor_match.get("plane_label", "")
                            or f"Plano {_sensor_match.get('plane', 0)}"
                        )
                if _curr_panel_bearing:
                    for _snap_id in _scl_cmp_snap_ids:
                        _prev_snap_full = load_scl_snapshot(
                            _scl_inst_id_local, _snap_id,
                        )
                        if _prev_snap_full is None:
                            continue
                        for _pb in _prev_snap_full.get("bearings", []):
                            if str(_pb.get("bearing_label", "")) == _curr_panel_bearing:
                                _scl_prev_snapshots_list.append({
                                    "label": _prev_snap_full.get("corrida_label", ""),
                                    "timestamp": _prev_snap_full.get("timestamp", ""),
                                    "op_speed": float(_prev_snap_full.get("operating_speed_rpm", 0) or 0),
                                    "x_gap_at_op": float(_pb.get("x_gap_at_op", 0) or 0),
                                    "y_gap_at_op": float(_pb.get("y_gap_at_op", 0) or 0),
                                    "eccentricity_ratio": float(_pb.get("eccentricity_ratio", 0) or 0),
                                    "attitude_angle": float(_pb.get("attitude_angle", 0) or 0),
                                    "lift_off_speed": float(_pb.get("lift_off_speed", 0) or 0),
                                    "trajectory_speed": _pb.get("trajectory_speed", []) or [],
                                    "trajectory_x_gap": _pb.get("trajectory_x_gap", []) or [],
                                    "trajectory_y_gap": _pb.get("trajectory_y_gap", []) or [],
                                })
                                break
    except Exception:
        _scl_prev_snapshots_list = []

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
        prev_snapshots=_scl_prev_snapshots_list if _scl_prev_snapshots_list else None,
    )

    text_diag = build_shaft_text_diagnostics(
        status=semaforo_status,
        util_max=diag["util_max"],
        margin_min=diag["margin_min"],
        first_warning_speed=early_rub["first_warning_speed"],
        first_danger_speed=early_rub["first_danger_speed"],
    )

    # Ciclo 17.3 P3+P4 — narrativa modal completa SCL para el PDF.
    # 5 bloques estilo Bently/API 670: encabezado factual, evolución
    # de centerline + clearance, clasificación migración + attitude,
    # análisis de lift-off + viscosidad/carga, distinción
    # bearing wear vs operacional vs alineación.
    if _scl_prev_snapshots_list:
        try:
            from core.scl_history import (
                eccentricity_change_classifier,
                attitude_shift_classifier,
            )
            _first_prev = _scl_prev_snapshots_list[0]
            _prev_label = str(_first_prev.get("label", "corrida anterior"))
            _prev_x = float(_first_prev.get("x_gap_at_op", 0))
            _prev_y = float(_first_prev.get("y_gap_at_op", 0))
            _prev_ecc = float(_first_prev.get("eccentricity_ratio", 0))
            _prev_att = float(_first_prev.get("attitude_angle", 0))
            _prev_lo = float(_first_prev.get("lift_off_speed", 0))

            # Datos actuales
            _curr_x = float(row_b.get("x_gap", 0)) if row_b is not None else 0.0
            _curr_y = float(row_b.get("y_gap", 0)) if row_b is not None else 0.0
            try:
                _diff_op = (display_df["speed"] - operating_rpm).abs()
                _row_op = display_df.loc[int(_diff_op.idxmin())]
                _curr_x = float(_row_op.get("x_gap", 0))
                _curr_y = float(_row_op.get("y_gap", 0))
            except Exception:
                pass
            from core.scl_diagnostics import compute_eccentricity_state as _ces
            _cr_local = float(boundary.get("clearance_x", 0) or 0)
            _curr_ecc = 0.0
            _curr_att = 0.0
            try:
                if _cr_local > 0:
                    _es = _ces(
                        x_pos=_curr_x, y_pos=_curr_y,
                        rpm=operating_rpm,
                        cx_radial=_cr_local, cy_radial=_cr_local,
                        bearing_center_x=boundary.get("center_x", 0) or 0,
                        bearing_center_y=boundary.get("center_y", 0) or 0,
                    )
                    _curr_ecc = float(getattr(_es, "eccentricity_ratio", 0))
                    _curr_att = float(getattr(_es, "attitude_angle_deg", 0))
            except Exception:
                pass

            _delta_x = _curr_x - _prev_x
            _delta_y = _curr_y - _prev_y
            _delta_ecc = _curr_ecc - _prev_ecc
            _delta_att = _curr_att - _prev_att
            _ecc_class = eccentricity_change_classifier(_delta_ecc)
            _att_class = attitude_shift_classifier(_delta_att)

            _gap_unit = meta.get("Gap Unit", "mil") or "mil"
            _narr: List[str] = []

            # 1) Encabezado factual
            _narr.append(
                f"Análisis comparativo de centerline contra «{_prev_label}». "
                f"A la velocidad operativa ({operating_rpm:.0f} rpm), la "
                f"posición DC del muñón evolucionó de "
                f"({_prev_x:+.3f}, {_prev_y:+.3f}) {_gap_unit} a "
                f"({_curr_x:+.3f}, {_curr_y:+.3f}) {_gap_unit}, lo que "
                f"representa una migración vectorial de "
                f"({_delta_x:+.3f}, {_delta_y:+.3f}) {_gap_unit}. La "
                f"eccentricity ratio cambió de {_prev_ecc:.3f} a "
                f"{_curr_ecc:.3f} (Δe/c = {_delta_ecc:+.3f}) y el "
                f"attitude angle pasó de {_prev_att:.1f}° a "
                f"{_curr_att:.1f}° (Δ = {_delta_att:+.1f}°)."
            )

            # 2) Evolución del centerline + clearance
            if _cr_local > 0:
                _used_pct_prev = abs(_prev_ecc) * 100.0
                _used_pct_curr = abs(_curr_ecc) * 100.0
                _para = (
                    f"En términos del clearance hidrodinámico ({_cr_local:.3f} "
                    f"{_gap_unit} radial), el muñón consume {_used_pct_curr:.0f}% "
                    f"del clearance disponible (anterior: {_used_pct_prev:.0f}%). "
                )
                if _curr_ecc > 0.85:
                    _para += (
                        "Esta posición está en zona crítica del clearance "
                        "(>85%) según los criterios de API 670 §6.7 — el "
                        "espacio entre muñón y carcasa está reducido a "
                        "valores donde el contacto sólido es probable bajo "
                        "transitorios. Se recomienda inspección inmediata "
                        "del cojinete, sello y juego diametral. "
                    )
                elif _curr_ecc > 0.70:
                    _para += (
                        "La posición está en zona de alarma del clearance "
                        "(70–85%), consumo elevado pero todavía dentro del "
                        "margen operacional según API 670. Se recomienda "
                        "vigilancia estrecha en próximas corridas. "
                    )
                else:
                    _para += (
                        "La posición está dentro del rango operacional "
                        "normal del cojinete hidrodinámico. "
                    )
                _narr.append(_para)

            # 3) Clasificación migración + attitude
            _diag_parts: List[str] = []
            if _ecc_class == "migration_critical":
                _diag_parts.append(
                    "La migración del centerline (Δe/c ≥ 0.25) es crítica "
                    "según los criterios de Bently / API 670, magnitud que "
                    "no se explica por deriva operacional normal. Sugiere "
                    "asentamiento del cojinete, pérdida significativa de "
                    "clearance por wiping del babbitt, deformación del "
                    "soporte por carga o cambio severo de la condición de "
                    "alineación entre cojinetes adyacentes."
                )
            elif _ecc_class == "migration_major":
                _diag_parts.append(
                    "La migración del centerline (Δe/c entre 0.15 y 0.25) "
                    "es mayor y requiere investigación. Causas típicas: "
                    "cambio en distribución de carga estática (alineación, "
                    "expansión térmica del soporte), variación apreciable "
                    "de la viscosidad del aceite (temperatura, "
                    "contaminación), o desgaste asimétrico incipiente del "
                    "babbitt del cojinete."
                )
            elif _ecc_class == "migration_minor":
                _diag_parts.append(
                    "La migración del centerline (Δe/c entre 0.05 y 0.15) "
                    "es menor y puede deberse a variación normal de "
                    "condiciones operativas (temperatura del aceite, carga "
                    "del proceso). Vale comparar con la próxima corrida "
                    "para confirmar si la tendencia se consolida."
                )
            else:
                _diag_parts.append(
                    "La eccentricity ratio se mantiene estable entre "
                    "corridas (Δ < 5% del clearance), sin evidencia de "
                    "migración del muñón."
                )

            if _att_class == "shift_critical":
                _diag_parts.append(
                    "Adicionalmente, el shift de attitude angle (≥30°) "
                    "indica un cambio severo en la dirección de la fuerza "
                    "hidrodinámica reactiva. Este patrón se asocia "
                    "típicamente con misalignment progresivo entre "
                    "cojinetes acoplados o redistribución mayor de carga "
                    "axial-radial — recomendable verificar alineación de "
                    "los cojinetes según API 686."
                )
            elif _att_class == "shift_major":
                _diag_parts.append(
                    "El shift de attitude angle (15–30°) sugiere cambio "
                    "moderado de la dirección de la fuerza hidrodinámica, "
                    "consistente con redistribución de carga entre "
                    "cojinetes adyacentes. Vale revisar las lecturas de "
                    "alineación al frío en próxima parada programada."
                )
            elif _att_class == "shift_minor":
                _diag_parts.append(
                    "El shift de attitude angle (5–15°) es menor y "
                    "normalmente se atribuye a variación operacional. "
                    "Monitorear evolución."
                )

            if _diag_parts:
                _narr.append(" ".join(_diag_parts))

            # 4) Lift-off speed evolution
            if _prev_lo > 0 and "lift_off_speed" in dir(diag):
                pass  # diag is a dict, this branch never runs — handled below
            _curr_lo = 0.0
            try:
                if _cr_local > 0:
                    for _, r in display_df.sort_values("speed").iterrows():
                        _xx = float(r.get("x_gap", 0))
                        _yy = float(r.get("y_gap", 0))
                        _ee = (_xx ** 2 + _yy ** 2) ** 0.5 / _cr_local
                        if _ee < 0.95:
                            _curr_lo = float(r.get("speed", 0))
                            break
            except Exception:
                pass
            if _prev_lo > 0 or _curr_lo > 0:
                _delta_lo = _curr_lo - _prev_lo
                _lo_para = ""
                if abs(_delta_lo) > 100:
                    _lo_para = (
                        f"La velocidad de lift-off cambió de "
                        f"{_prev_lo:.0f} rpm a {_curr_lo:.0f} rpm "
                        f"(Δ = {_delta_lo:+.0f} rpm). "
                    )
                    if _delta_lo > 200:
                        _lo_para += (
                            "Un incremento >200 rpm en lift-off es señal "
                            "de degradación del soporte hidrodinámico — "
                            "viscosidad efectiva reducida, carga "
                            "incrementada o pérdida de clearance. "
                            "Evaluación preventiva del aceite (viscosidad, "
                            "contaminación) recomendada según API 670 §6.7."
                        )
                    elif _delta_lo < -200:
                        _lo_para += (
                            "Una disminución >200 rpm en lift-off "
                            "típicamente indica condición más favorable "
                            "del aceite (más fresco, mejor viscosidad) o "
                            "redistribución de carga que descargó este "
                            "cojinete."
                        )
                    if _lo_para:
                        _narr.append(_lo_para)

            # 5) Distinción wear vs operacional vs alineación
            if _ecc_class in ("migration_major", "migration_critical"):
                _disc = ""
                if _att_class in ("shift_major", "shift_critical"):
                    _disc = (
                        "Diagnóstico diferencial: la combinación de "
                        "migración significativa del centerline + shift "
                        "mayor del attitude angle apunta más a un cambio "
                        "ESTRUCTURAL (alineación, asentamiento, daño "
                        "mecánico del cojinete) que a deriva operacional. "
                        "Recomendable verificación al frío y revisión del "
                        "babbit antes de descartar bearing wear."
                    )
                else:
                    _disc = (
                        "Diagnóstico diferencial: la migración del "
                        "centerline sin shift apreciable de attitude "
                        "angle es más compatible con cambio de condición "
                        "OPERACIONAL (carga, temperatura del aceite) que "
                        "con daño mecánico del cojinete. Aún así, "
                        "monitorear de cerca para descartar consolidación "
                        "de la tendencia."
                    )
                _narr.append(_disc)

            _comp_narr = " ".join(_narr)
            if isinstance(text_diag, dict):
                text_diag["comparison_narrative"] = _comp_narr
        except Exception:
            pass

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

    # =========================================================
    # Diagnóstico Cat IV (rotordynamics + Vault) — solo si hay clearance
    # válido (sea del Vault o manual configurado por el usuario)
    # =========================================================
    cat_iv_text_diag = None
    if boundary["clearance_x"] > 0 and boundary["clearance_y"] > 0 and len(display_df) > 5:
        # Buscar posición a operating_rpm
        op_speed_target = float(operating_rpm)
        rpms_arr = display_df["speed"].to_numpy(dtype=float)
        if rpms_arr.size > 0 and rpms_arr.min() <= op_speed_target <= rpms_arr.max():
            op_idx = int(np.argmin(np.abs(rpms_arr - op_speed_target)))
        else:
            # Si operating_rpm está fuera del rango medido, usar el máximo
            op_idx = int(np.argmax(rpms_arr))

        x_at_op = float(x_plot[op_idx])
        y_at_op = float(y_plot[op_idx])
        actual_op_rpm = float(rpms_arr[op_idx])

        ecc_state = compute_eccentricity_state(
            x_pos=x_at_op,
            y_pos=y_at_op,
            rpm=actual_op_rpm,
            cx_radial=float(boundary["clearance_x"]),
            cy_radial=float(boundary["clearance_y"]),
            bearing_center_x=float(boundary["center_x"]),
            bearing_center_y=float(boundary["center_y"]),
            load_direction_deg=270.0,
        )

        lift_off_rpm = detect_lift_off_speed(
            rpms=rpms_arr,
            x_positions=x_plot,
            y_positions=y_plot,
            cx_radial=float(boundary["clearance_x"]),
            cy_radial=float(boundary["clearance_y"]),
        )

        diametral_clearance_mm_value = None
        if vault_params and vault_params.get("diametral_clearance_mm"):
            diametral_clearance_mm_value = float(vault_params["diametral_clearance_mm"])
        elif vault_clearance_radial_mil is not None:
            # Reconstruir Cd_mm desde el radial en mil
            diametral_clearance_mm_value = float(vault_clearance_radial_mil) * 0.0254 * 2.0

        cat_iv_text_diag = build_scl_diagnostics_rotordyn(
            eccentricity_state=ecc_state,
            operating_rpm=actual_op_rpm,
            profile_label=profile_label or "",
            bearing_inner_diameter_mm=(
                vault_params.get("bearing_inner_diameter_mm") if vault_params else None
            ),
            diametral_clearance_mm=diametral_clearance_mm_value,
            clearance_source=cr_source or "configuración manual de la sidebar",
            babbitt_material=(vault_params.get("babbitt_material") if vault_params else None),
            last_rebabbiting_date=(
                vault_params.get("last_rebabbiting_date") if vault_params else None
            ),
            document_reference=vault_doc_ref,
            lift_off_rpm=lift_off_rpm,
            amp_unit=_amp_unit_label(gap_unit),
            clearance_reference_frame=clearance_center_mode or "",
            bearing_center_x=float(boundary["center_x"]),
            bearing_center_y=float(boundary["center_y"]),
        )

        with st.expander(
            f"Diagnóstico avanzado (rotordynamics + Vault) · Panel {panel_index + 1}",
            expanded=True,
        ):
            st.markdown(f"**{cat_iv_text_diag['headline']}**")
            st.write(cat_iv_text_diag["detail"])
            st.write(cat_iv_text_diag["action"])

    # Título con etiqueta de fecha de la corrida (más útil para el PDF)
    date_tag = ""
    if "ts_min" in display_df.columns and not display_df["ts_min"].isna().all():
        try:
            date_tag = pd.Timestamp(display_df["ts_min"].min()).strftime("%d %b %Y")
        except Exception:
            date_tag = ""
    if not date_tag and item.get("file_stem"):
        date_tag = str(item["file_stem"])
    title_date_clause = f" · {date_tag}" if date_tag else ""
    title = (
        f"Shaft Centerline {panel_index + 1}{title_date_clause} — "
        f"{machine} — {point} / {paired_point}"
    )

    # Cuando hay narrativa Cat IV, el bloque legacy basado en (0,0) confunde
    # (mide utilización contra la posición de reposo, no contra el bearing
    # center real). Lo suprimimos del PDF para no contradecir al Cat IV.
    bently_frame = (clearance_center_mode or "").lower().startswith("bottom load")
    if cat_iv_text_diag is not None and bently_frame:
        notes = f"{cat_iv_text_diag['detail']}\n\n{cat_iv_text_diag['action']}"
    elif cat_iv_text_diag is not None:
        notes = (
            f"{cat_iv_text_diag['detail']}\n\n{cat_iv_text_diag['action']}\n\n"
            f"---\nDiagnóstico de utilización de boundary (referencia rest position):\n\n"
            f"{_build_scl_report_notes(text_diag)}"
        )
    else:
        notes = _build_scl_report_notes(text_diag)
    export_fig = _add_scl_export_footer(fig, text_diag)

    # ------------------------------------------------------------
    # Ciclo 17.26 — Interpretación clínica AI (Shaft Centerline)
    # ------------------------------------------------------------
    ai_state_key_scl = f"wm_ai_diag_scl_{panel_index}_{item['id']}"
    if ai_state_key_scl not in st.session_state:
        st.session_state[ai_state_key_scl] = None

    with st.expander(
        "Interpretación clínica AI · Diagnóstico Cat IV asistido",
        expanded=False,
    ):
        if not is_ai_available():
            st.info(
                "**AI Diagnóstico no disponible.** Falta configurar "
                "`[anthropic] api_key` en los secrets de Streamlit."
            )
        else:
            stored_scl = st.session_state.get(ai_state_key_scl)
            ai_btn_col1, ai_btn_col2, ai_btn_col3 = st.columns([1.4, 1.4, 2.4])
            with ai_btn_col1:
                gen_clicked_scl = st.button(
                    "Generar diagnóstico AI"
                    if stored_scl is None
                    else "Diagnóstico generado",
                    key=f"ai_gen_btn_scl_{panel_index}_{item['id']}",
                    use_container_width=True,
                    type="primary" if stored_scl is None else "secondary",
                    disabled=stored_scl is not None and stored_scl.get("ok", False),
                )
            with ai_btn_col2:
                regen_clicked_scl = st.button(
                    "Regenerar",
                    key=f"ai_regen_btn_scl_{panel_index}_{item['id']}",
                    use_container_width=True,
                    disabled=stored_scl is None,
                )
            with ai_btn_col3:
                st.caption(
                    "Claude Sonnet 4.5 · ~$0.015 por diagnóstico · "
                    "cacheado 30 días si no regenerás."
                )

            should_call_scl = bool(gen_clicked_scl) and (stored_scl is None)
            should_regen_scl = bool(regen_clicked_scl) and (stored_scl is not None)

            if should_call_scl or should_regen_scl:
                # Payload SCL: eccentricity, attitude angle, lift-off
                # speed, clearance, posición del muñón.
                ai_payload_scl: Dict[str, Any] = {
                    "machine": {
                        "tag": str(machine or ""),
                        "punto_medicion": str(point or ""),
                        "punto_acoplado": str(paired_point or ""),
                        "clearance_center_mode": str(clearance_center_mode or ""),
                    },
                    "norm": {
                        "headline_tecnico": str(text_diag.get("headline", "") or "") if text_diag else "",
                        "cat_iv_headline": str(
                            cat_iv_text_diag.get("headline", "") or ""
                        ) if cat_iv_text_diag else "",
                    },
                    "technical": {
                        "diagnostic_detail": str(
                            cat_iv_text_diag.get("detail", "") or ""
                        )[:1500] if cat_iv_text_diag else "",
                        "diagnostic_action": str(
                            cat_iv_text_diag.get("action", "") or ""
                        )[:1500] if cat_iv_text_diag else "",
                        "legacy_notes": str(
                            _build_scl_report_notes(text_diag)
                        )[:1000] if text_diag else "",
                    },
                    "trend": {},
                }

                with st.spinner("Claude analizando el shaft centerline... (5-15 seg)"):
                    try:
                        result_scl = generate_ai_diagnostic(
                            ai_payload_scl,
                            module_type="shaft_centerline",
                            use_cache=not should_regen_scl,
                        )
                    except Exception as exc:
                        result_scl = {
                            "ok": False,
                            "markdown": (
                                f"_Error inesperado al generar diagnóstico AI:_\n\n"
                                f"```\n{type(exc).__name__}: {exc}\n```"
                            ),
                            "error": str(exc)[:500],
                            "model": "",
                            "cached": False,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "fallback_used": False,
                            "fallback_reason": "",
                            "generated_at": "",
                        }
                st.session_state[ai_state_key_scl] = result_scl
                stored_scl = result_scl

            if stored_scl is not None:
                if stored_scl.get("ok"):
                    if stored_scl.get("fallback_used"):
                        st.info(
                            "Diagnóstico generado con modelo de respaldo "
                            "(Haiku 4.5)."
                        )
                    st.markdown(stored_scl.get("markdown", ""))
                    model_used_scl = str(stored_scl.get("model", "") or "")
                    if model_used_scl.startswith("claude-haiku"):
                        in_p_scl, out_p_scl = 1.0, 5.0
                    else:
                        in_p_scl, out_p_scl = 3.0, 15.0
                    cost_usd_scl = (
                        stored_scl.get("input_tokens", 0) * in_p_scl
                        + stored_scl.get("output_tokens", 0) * out_p_scl
                    ) / 1_000_000
                    fallback_tag_scl = (
                        " · modelo de respaldo"
                        if stored_scl.get("fallback_used") else ""
                    )
                    st.caption(
                        f"Modelo: `{model_used_scl}` · "
                        f"Tokens: {stored_scl.get('input_tokens', 0)} → "
                        f"{stored_scl.get('output_tokens', 0)} · "
                        f"Costo: ~${cost_usd_scl:.4f} · "
                        f"{'(cacheado)' if stored_scl.get('cached') else '(generado nuevo)'}"
                        f"{fallback_tag_scl}"
                    )
                else:
                    st.error(
                        stored_scl.get("markdown", "Error al generar diagnóstico AI.")
                    )

    b1, b2 = st.columns(2)
    with b1:
        png_bytes, png_error = build_export_png_bytes(export_fig)
        if st.button("Enviar panel a reporte", key=f"scl_report_btn_{panel_index}_{item['id']}"):
            # Ciclo 17.26 — armar bloque AI si está generado
            ai_stored_for_scl_report = st.session_state.get(ai_state_key_scl)
            final_notes_scl = notes
            if (ai_stored_for_scl_report
                    and ai_stored_for_scl_report.get("ok")
                    and ai_stored_for_scl_report.get("markdown")):
                ai_md_scl = str(
                    ai_stored_for_scl_report.get("markdown", "")
                ).strip()
                if ai_md_scl:
                    quant_lines_scl: List[str] = ["Parámetro|Valor"]
                    if machine:
                        quant_lines_scl.append(f"Máquina|{machine}")
                    if point:
                        quant_lines_scl.append(f"Punto X|{point}")
                    if paired_point:
                        quant_lines_scl.append(f"Punto Y|{paired_point}")
                    if clearance_center_mode:
                        quant_lines_scl.append(
                            f"Modo de referencia|{clearance_center_mode}"
                        )
                    if cat_iv_text_diag:
                        cat_iv_headline_scl = str(
                            cat_iv_text_diag.get("headline", "") or ""
                        ).strip()
                        if cat_iv_headline_scl:
                            quant_lines_scl.append(
                                f"Diagnóstico Cat IV|{cat_iv_headline_scl}"
                            )

                    final_notes_scl = (
                        "<<<WM_AI_BLOCK>>>\n"
                        + "\n".join(quant_lines_scl)
                        + "\n<<<WM_AI_NARRATIVE>>>\n"
                        + ai_md_scl
                    )
            push_report_item(
                title=title, notes=final_notes_scl, image_bytes=png_bytes
            )
            ai_extra_scl = (
                " (con Diagnóstico AI)"
                if final_notes_scl != notes else ""
            )
            st.success(
                f"Panel individual enviado al reporte{ai_extra_scl}."
            )
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
    vault_clearance_radial_mil: Optional[float] = None,
    vault_params: Optional[Dict[str, Any]] = None,
    vault_doc_ref: Optional[str] = None,
    profile_label: Optional[str] = None,
    operating_rpm: float = 3600.0,
    cr_source: str = "",
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
    boundary = None
    if valid_dfs:
        all_x = np.concatenate([df["x_plot"].to_numpy(dtype=float) for df in valid_dfs])
        all_y = np.concatenate([df["y_plot"].to_numpy(dtype=float) for df in valid_dfs])

        # Ciclo 23.153 — FIX de unidades también en el comparativo: convertir
        # el clearance del Vault (mils) a la unidad del gap de los datos.
        _gap_unit_cmp = (items[0].get("meta", {}) or {}).get("Gap Unit", "mil") if items else "mil"
        vault_cr_cmp = _mils_to_gap_unit(vault_clearance_radial_mil, _gap_unit_cmp)

        # Misma lógica de prioridad que el panel individual:
        # Manual > Vault > Auto heurístico
        if clearance_mode == "Manual":
            boundary = resolve_clearance_boundary(
                x=all_x, y=all_y, mode="Manual",
                center_mode=clearance_center_mode,
                manual_cx=manual_clearance_x, manual_cy=manual_clearance_y,
                manual_center_x=manual_center_x, manual_center_y=manual_center_y,
            )
            boundary["source"] = "manual (sidebar)"
        elif vault_cr_cmp is not None:
            boundary = resolve_clearance_boundary(
                x=all_x, y=all_y, mode="Manual",
                center_mode=clearance_center_mode,
                manual_cx=float(vault_cr_cmp),
                manual_cy=float(vault_cr_cmp),
                manual_center_x=manual_center_x, manual_center_y=manual_center_y,
            )
            boundary["source"] = f"Vault ({cr_source})"
        else:
            boundary = resolve_clearance_boundary(
                x=all_x, y=all_y, mode=clearance_mode,
                center_mode=clearance_center_mode,
                manual_cx=manual_clearance_x, manual_cy=manual_clearance_y,
                manual_center_x=manual_center_x, manual_center_y=manual_center_y,
            )
            boundary["source"] = "auto heurístico (datos)"

        # Auto X/Y consciente del clearance también en el comparativo
        x_range, y_range = compute_xy_ranges(
            x=all_x, y=all_y,
            auto_scale_xy=auto_scale_xy,
            manual_x_min=manual_x_min, manual_x_max=manual_x_max,
            manual_y_min=manual_y_min, manual_y_max=manual_y_max,
            clearance_x=boundary.get("clearance_x"),
            clearance_y=boundary.get("clearance_y"),
            center_x=boundary.get("center_x", 0.0),
            center_y=boundary.get("center_y", 0.0),
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

        # Cat IV overlay (eccentricity rings + bearing center + rest + load arrow)
        add_scl_cat_iv_overlay(
            fig,
            center_x=boundary["center_x"],
            center_y=boundary["center_y"],
            clearance_x=boundary["clearance_x"],
            clearance_y=boundary["clearance_y"],
        )

    palette = ["#2563eb", "#16a34a", "#9333ea", "#ea580c", "#dc2626", "#0891b2", "#7c3aed", "#0f766e"]

    # Collect operating-speed point per record for migration overlays
    op_points: List[Dict[str, Any]] = []

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

        # Identificar el punto a velocidad operativa para overlay Cat IV
        if not df.empty:
            rpms_arr = df["speed"].to_numpy(dtype=float)
            x_arr = df["x_plot"].to_numpy(dtype=float)
            y_arr = df["y_plot"].to_numpy(dtype=float)
            target = float(operating_rpm)
            if rpms_arr.size > 0 and rpms_arr.min() <= target <= rpms_arr.max():
                k_idx = int(np.argmin(np.abs(rpms_arr - target)))
            elif rpms_arr.size > 0:
                k_idx = int(np.argmax(rpms_arr))
            else:
                continue
            op_points.append({
                "x": float(x_arr[k_idx]),
                "y": float(y_arr[k_idx]),
                "rpm": float(rpms_arr[k_idx]),
                "color": color,
                "date_label": date_label,
                "ts_start": rec["ts_start"],
            })

    # Marcadores de punto operativo por fecha (estrella con label de fecha)
    for op in op_points:
        fig.add_trace(
            go.Scatter(
                x=[op["x"]], y=[op["y"]],
                mode="markers+text",
                marker=dict(size=14, color=op["color"], symbol="star",
                            line=dict(width=1.5, color="white")),
                text=[op["date_label"].split(" ")[0]],
                textposition="top center",
                textfont=dict(size=10, color=op["color"], family="Arial Black"),
                name=f"Op @ {op['rpm']:.0f} rpm · {op['date_label']}",
                hovertemplate=(
                    f"Punto operativo<br>{op['date_label']}<br>"
                    f"X: {op['x']:.3f} mil pp<br>Y: {op['y']:.3f} mil pp<br>"
                    f"RPM: {op['rpm']:.0f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Vectores de migración entre fechas consecutivas
    if boundary is not None and len(op_points) >= 2:
        cx_b = float(boundary.get("clearance_x", 0.0)) or 0.0
        cy_b = float(boundary.get("clearance_y", 0.0)) or 0.0
        clr_ref = max(cx_b, cy_b, 1e-9)
        for i in range(1, len(op_points)):
            p0 = op_points[i - 1]
            p1 = op_points[i]
            dx = p1["x"] - p0["x"]
            dy = p1["y"] - p0["y"]
            mag = float(np.hypot(dx, dy))
            pct_clr = (mag / clr_ref) * 100.0
            fig.add_annotation(
                x=p1["x"], y=p1["y"],
                ax=p0["x"], ay=p0["y"],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.0,
                arrowcolor="#0f172a",
                text=f"Δ={mag:.2f} mil pp ({pct_clr:.1f}% c)",
                font=dict(size=10, color="#0f172a"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#0f172a", borderwidth=1, borderpad=2,
                xshift=6, yshift=6,
            )

        # Línea de attitude angle (bearing center → punto operativo de la última fecha)
        last_op = op_points[-1]
        fig.add_trace(
            go.Scatter(
                x=[boundary["center_x"], last_op["x"]],
                y=[boundary["center_y"], last_op["y"]],
                mode="lines",
                line=dict(width=1.5, color="#0f172a", dash="dash"),
                name="Attitude angle (última fecha)",
                hoverinfo="skip",
                showlegend=True,
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

    # =========================================================
    # Diagnóstico legacy (boundary utilization)
    # =========================================================
    diag = _scl_compare_diagnostic(compare_records)
    with st.expander("Diagnóstico comparativo automático (boundary)", expanded=False):
        st.markdown(f"**{diag['headline']}**")
        st.write(diag["detail"])
        st.write(diag["action"])

    # =========================================================
    # Diagnóstico Cat IV multi-fecha (rotordynamics + Vault)
    # =========================================================
    cat_iv_compare_md = ""
    if boundary is not None and boundary.get("clearance_x", 0) > 0:
        # Calcular eccentricity_state para cada record a operating_rpm
        ecc_states = []
        for rec in compare_records:
            df = rec["df"]
            if df.empty:
                continue
            rpms_arr = df["speed"].to_numpy(dtype=float)
            x_arr = df["x_plot"].to_numpy(dtype=float)
            y_arr = df["y_plot"].to_numpy(dtype=float)

            if rpms_arr.size == 0:
                continue
            target = float(operating_rpm)
            if rpms_arr.min() <= target <= rpms_arr.max():
                idx = int(np.argmin(np.abs(rpms_arr - target)))
            else:
                idx = int(np.argmax(rpms_arr))

            es = compute_eccentricity_state(
                x_pos=float(x_arr[idx]),
                y_pos=float(y_arr[idx]),
                rpm=float(rpms_arr[idx]),
                cx_radial=float(boundary["clearance_x"]),
                cy_radial=float(boundary["clearance_y"]),
                bearing_center_x=float(boundary["center_x"]),
                bearing_center_y=float(boundary["center_y"]),
                load_direction_deg=270.0,
            )
            ecc_states.append({
                "label": rec.get("label", "—"),
                "ts_start": rec.get("ts_start"),
                "ecc_state": es,
            })

        if len(ecc_states) >= 2:
            # Ordenar por fecha
            ecc_states.sort(
                key=lambda e: pd.Timestamp(e["ts_start"]) if e["ts_start"] is not None else pd.Timestamp.min
            )

            with st.expander("Diagnóstico avanzado multi-fecha (rotordynamics + Vault)", expanded=True):
                # Tabla de e/c por fecha
                rows_cat = []
                for e in ecc_states:
                    es = e["ecc_state"]
                    rows_cat.append({
                        "Fecha": pd.Timestamp(e["ts_start"]).strftime("%Y-%m-%d") if e["ts_start"] is not None else "—",
                        "Archivo": e["label"],
                        "RPM": f"{es.rpm:.0f}",
                        "X (mil pp)": f"{es.x_pos:+.3f}",
                        "Y (mil pp)": f"{es.y_pos:+.3f}",
                        "e/c": f"{es.eccentricity_ratio:.3f}",
                        "α (°)": f"{es.attitude_angle_deg:.1f}",
                        "Clasificación": es.classification,
                    })
                st.dataframe(pd.DataFrame(rows_cat), width="stretch", hide_index=True)

                # Narrativa: introducción + síntesis cronológica + migración
                first = ecc_states[0]
                last = ecc_states[-1]
                first_date = pd.Timestamp(first["ts_start"]).strftime("%d %b %Y") if first["ts_start"] is not None else first["label"]
                last_date = pd.Timestamp(last["ts_start"]).strftime("%d %b %Y") if last["ts_start"] is not None else last["label"]

                # Comparación entre primera y última corrida
                migration = compare_centerline_migration(first["ecc_state"], last["ecc_state"])

                profile_clause = f"El profile activo es '{profile_label}'." if profile_label else ""
                doc_clause = f" Documento de referencia: {vault_doc_ref}." if vault_doc_ref else ""

                clearance_clause = ""
                if boundary.get("source"):
                    clearance_clause = f" Clearance radial usado en el análisis: {boundary['clearance_x']:.3f} mil pp ({boundary['source']})."

                # Construir narrativa fluida
                paragraphs_cat = []

                paragraphs_cat.append(
                    f"Se analizó la evolución del centerline del muñón a velocidad operativa "
                    f"{operating_rpm:.0f} rpm a lo largo de {len(ecc_states)} corridas comprendidas "
                    f"entre {first_date} y {last_date}. {profile_clause}{clearance_clause}{doc_clause}"
                )

                # Síntesis cronológica
                prose_lines = []
                for e in ecc_states:
                    es = e["ecc_state"]
                    date_str = pd.Timestamp(e["ts_start"]).strftime("%d %b %Y") if e["ts_start"] is not None else e["label"]
                    prose_lines.append(
                        f"La corrida del {date_str} ubicó el muñón en posición "
                        f"({es.x_pos:+.3f}, {es.y_pos:+.3f}) mil pp, con eccentricity ratio "
                        f"e/c = {es.eccentricity_ratio:.3f} y attitude angle "
                        f"{es.attitude_angle_deg:.1f}°, clasificación {es.classification}."
                    )
                paragraphs_cat.append(
                    "Síntesis cronológica de las posiciones medidas:\n\n" +
                    "\n\n".join(prose_lines)
                )

                # Migración
                paragraphs_cat.append(migration.narrative)

                detail_cat = "\n\n".join(paragraphs_cat)

                # Acciones según severidad de migración
                if migration.classification == "STABLE":
                    items_cat = [
                        f"Adoptar la corrida del {last_date} como línea base actualizada del centerline.",
                        "Mantener la frecuencia actual de medición y comparar próximos arranques contra la línea base.",
                        "Vigilar e/c y attitude angle en cada nueva corrida para detectar tendencias tempranas.",
                        "Correlacionar con datos de Polar/Bode 1X y temperatura de cojinetes para confirmar estabilidad de condición.",
                    ]
                elif migration.classification == "MINOR_DRIFT":
                    items_cat = [
                        "Continuar el monitoreo con frecuencia mayor para confirmar si la migración es transient o tendencia.",
                        "Verificar consistencia de las condiciones de medición entre fechas (carga, temperatura del aceite, balance).",
                        "Correlacionar con eventos de mantenimiento u operación entre fechas.",
                    ]
                elif migration.classification == "MODERATE_DRIFT":
                    items_cat = [
                        "Investigar causas de la migración: cambio de carga, alineación del tren, condición del babbitt.",
                        "Inspeccionar visualmente el cojinete en próximo paro programado.",
                        "Verificar viscosidad del aceite y temperatura de cojinetes contra valores de comisionamiento.",
                        "Correlacionar con espectro 1X (Polar/Bode) y órbita filtrada.",
                    ]
                else:
                    items_cat = [
                        "PRIORIDAD ALTA: programar inspección directa del babbitt en próxima oportunidad.",
                        "Verificar inmediatamente temperatura, viscosidad y caudal del aceite contra especificación OEM.",
                        "Confirmar carga real del rotor y descartar desalineación del tren.",
                        "Documentar como hallazgo crítico, notificar al equipo de ingeniería rotodinámica.",
                        "Si la condición persiste, restringir operación sostenida hasta confirmación del estado del cojinete.",
                    ]

                intro_cat = (
                    "A partir del análisis de migración del centerline entre las fechas "
                    "evaluadas, se establecen las siguientes recomendaciones:"
                )
                action_cat = intro_cat + "\n\n" + "\n\n".join(
                    f"{i+1}. {item}" for i, item in enumerate(items_cat)
                )

                st.markdown(f"**{migration.narrative.split('.')[0]}.**")
                st.write(detail_cat)
                st.write(action_cat)

                cat_iv_compare_md = (
                    f"{detail_cat}\n\n{action_cat}"
                )

    # Cuando hay narrativa Cat IV, se suprime el bloque legacy basado en (0,0)
    # para evitar contradicciones en el PDF (la legacy mide utilización contra
    # la posición de reposo, no contra el bearing center real). Se conserva en
    # los expanders de la UI para inspección, pero no entra al PDF.
    bently_frame = (clearance_center_mode or "").lower().startswith("bottom load")
    summary_block = f"--- RESUMEN ---\n{summary_df.to_string(index=False)}"
    if cat_iv_compare_md and bently_frame:
        notes = f"{cat_iv_compare_md}\n\n{summary_block}"
    elif cat_iv_compare_md:
        notes = (
            f"{cat_iv_compare_md}\n\n---\n\n"
            f"Diagnóstico de utilización de boundary (referencia rest position):\n\n"
            f"{diag['headline']}\n\n{diag['detail']}\n\n{diag['action']}\n\n"
            f"{summary_block}"
        )
    else:
        notes = (
            f"{diag['headline']}\n\n"
            f"{diag['detail']}\n\n"
            f"{diag['action']}\n\n"
            f"{summary_block}"
        )

    png_bytes, png_error = build_export_png_bytes(fig)

    # Rango temporal del comparativo en el título (más informativo en el PDF)
    valid_starts = [r["ts_start"] for r in compare_records if r["ts_start"] is not None]
    range_clause = ""
    if valid_starts:
        try:
            t_min = pd.Timestamp(min(valid_starts)).strftime("%d %b %Y")
            t_max = pd.Timestamp(max(valid_starts)).strftime("%d %b %Y")
            range_clause = f" · {t_min} → {t_max}"
        except Exception:
            range_clause = ""
    compare_title = f"Shaft Centerline · Comparación multi-fecha{range_clause}"

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Enviar comparativo a reporte", key="wm_scl_compare_report_btn"):
            push_report_item(
                title=compare_title,
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

        # Asset Instance + Vault integration (Cat IV) — Ciclo 8
        # Cada máquina física tiene su propia instancia con sus propios
        # parámetros y documentos. Antes (Ciclo 7) la selección era por
        # profile, lo que mezclaba data entre máquinas físicamente
        # distintas del mismo modelo.
        instance_state = render_instance_selector(module_name="shaft_centerline")
        active_instance_id = instance_state["instance_id"]
        active_profile_key = instance_state["profile_key"]
        active_profile_label = instance_state["profile_label"]
        active_operating_rpm = instance_state["operating_rpm"]

        # Ciclo 23.156 — quitado el cuadro de applicability (ruido sin valor).

        # Lookup del Vault PER-INSTANCIA (no per-profile)
        vault_params = dict(instance_state.get("captured_parameters", {}))
        vault_docs = list(instance_state.get("documents", []))
        vault_doc_ref = vault_docs[0]["title"] if vault_docs else None

        cr_mil_vault, cr_source = derive_radial_clearance_from_vault(
            bearing_inner_diameter_mm=vault_params.get("bearing_inner_diameter_mm"),
            shaft_journal_diameter_mm=vault_params.get("shaft_journal_diameter_mm"),
            diametral_clearance_mm=vault_params.get("diametral_clearance_mm"),
            target_unit="mil",
        )

        if cr_mil_vault is not None:
            st.success(
                f"**Vault:** clearance radial = {cr_mil_vault:.2f} mil pp "
                f"({cr_source})"
            )
        else:
            st.info(
                "Sin datos de cojinete en el Vault. Captura el diámetro interno "
                "y/o clearance del cojinete en Asset Documents para análisis "
                "preciso conforme a API 670. Usando valores manuales de la sidebar."
            )

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
        clearance_center_mode = st.selectbox(
            "Boundary center",
            options=[
                "Bottom load reference (API 670 / práctica estándar)",
                "Origin (0,0)",
                "Data Mean",
                "Manual",
            ],
            index=0,
            help=(
                "Convención de placement del clearance circle:\n\n"
                "**Bottom load reference**: práctica estándar para cojinetes hidrodinámicos en "
                "máquinas horizontales con carga gravitacional (referencia API 670). "
                "El (0,0) del registro = muñón en reposo apoyado al fondo del cojinete. "
                "Bearing center automáticamente en (0, +Cr). "
                "Usar este default a menos que haya razón específica.\n\n"
                "**Origin (0,0)**: bearing center forzado al origen. Solo para debug, máquinas "
                "verticales o sistemas con calibración no estándar.\n\n"
                "**Data Mean**: bearing center en el centroide. Útil cuando el registro no fue "
                "calibrado al rest position.\n\n"
                "**Manual**: especifica el centro tú mismo."
            ),
        )

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

    # ============================================================
    # Ciclo 17.3 — Histórico SCL (multi-snapshot trail)
    # ============================================================
    _scl_inst_id = (
        instance_state.get("instance_id")
        or st.session_state.get("wm_active_instance_id", "")
    )
    _scl_inst = None
    _scl_sensors_map: List[Dict[str, Any]] = []
    if _scl_inst_id:
        try:
            from core.instance_state import get_instance as _scl_get_inst
            _scl_inst = _scl_get_inst(_scl_inst_id)
            if _scl_inst is not None:
                _scl_sensors_map = list(_scl_inst.sensors or [])
        except Exception:
            _scl_inst = None

    def _wm_extract_scl_readings(
        items: List[Dict[str, Any]],
        sensors_map: List[Dict[str, Any]],
        op_speed_rpm: float,
        clearance_radial_mil: float,
    ) -> List[Dict[str, Any]]:
        """Por cada CSV SCL extrae bearing match + posición + eccentricity."""
        from core.sensor_map import (
            resolve_sensor_for_point as _wm_resolve,
            sensor_label as _wm_slbl,
        )
        from core.scl_diagnostics import compute_eccentricity_state
        out = []
        for it in items:
            try:
                meta = it.get("meta") or {}
                # Para SCL, el matching usa Point Name (Y probe usual)
                point = str(meta.get("Point Name", "") or it.get("point", "") or "")
                paired = str(meta.get("Paired Point Name", "") or "")
                variable = str(meta.get("Variable", "") or it.get("variable", "") or "")
                unit = str(meta.get("Y-Axis Unit", "") or meta.get("Unit", "") or "")

                # Buscar el sensor matched (cualquiera del par X/Y)
                sensor_match = None
                if sensors_map:
                    sensor_match = _wm_resolve(sensors_map, point, variable, unit)
                    if sensor_match is None and paired:
                        sensor_match = _wm_resolve(sensors_map, paired, variable, unit)
                if sensor_match is None:
                    continue

                df = it.get("grouped_df")
                if df is None or len(df) == 0:
                    continue

                # Punto operativo
                _diff = (df["speed"] - op_speed_rpm).abs()
                _row = df.loc[int(_diff.idxmin())]
                x_at_op = float(_row.get("x_gap", 0.0))
                y_at_op = float(_row.get("y_gap", 0.0))

                # Bearing label (del plano del sensor matched)
                bearing_label = (
                    sensor_match.get("plane_label", "")
                    or f"Plano {sensor_match.get('plane', 0)}"
                )

                # Eccentricity state
                ecc_ratio = 0.0
                attitude_angle = 0.0
                try:
                    if clearance_radial_mil and clearance_radial_mil > 0:
                        ecc_state = compute_eccentricity_state(
                            x_pos=x_at_op,
                            y_pos=y_at_op,
                            rpm=op_speed_rpm,
                            cx_radial=clearance_radial_mil,
                            cy_radial=clearance_radial_mil,
                            bearing_center_x=0.0,
                            bearing_center_y=0.0,
                            load_direction_deg=270.0,
                        )
                        ecc_ratio = float(getattr(ecc_state, "eccentricity_ratio", 0))
                        attitude_angle = float(getattr(ecc_state, "attitude_angle_deg", 0))
                except Exception:
                    pass

                # Lift-off speed: heuristica simple — primer punto donde
                # eccentricity_ratio cae por debajo de 0.95.
                lift_off_speed = 0.0
                try:
                    if clearance_radial_mil and clearance_radial_mil > 0:
                        for _, r in df.sort_values("speed").iterrows():
                            _x = float(r.get("x_gap", 0))
                            _y = float(r.get("y_gap", 0))
                            _ecc = (_x ** 2 + _y ** 2) ** 0.5 / clearance_radial_mil
                            if _ecc < 0.95:
                                lift_off_speed = float(r.get("speed", 0))
                                break
                except Exception:
                    pass

                # Trayectoria downsampleada
                _df_sorted = df.sort_values("speed").reset_index(drop=True)
                _N = 80
                if len(_df_sorted) > _N:
                    _idx = np.linspace(0, len(_df_sorted) - 1, _N).astype(int)
                    _df_ds = _df_sorted.iloc[_idx]
                else:
                    _df_ds = _df_sorted

                out.append({
                    "bearing_label": bearing_label,
                    "csv_file": it.get("file_name", ""),
                    "x_gap_at_op": x_at_op,
                    "y_gap_at_op": y_at_op,
                    "gap_unit": unit or "mil",
                    "eccentricity_ratio": ecc_ratio,
                    "attitude_angle": attitude_angle,
                    "clearance_radial": clearance_radial_mil or 0.0,
                    "lift_off_speed": lift_off_speed,
                    "csv_timestamp": str(meta.get("Timestamp", "") or ""),
                    "trajectory_speed": _df_ds["speed"].astype(float).tolist(),
                    "trajectory_x_gap": _df_ds["x_gap"].astype(float).tolist(),
                    "trajectory_y_gap": _df_ds["y_gap"].astype(float).tolist(),
                })
            except Exception:
                continue
        return out

    _scl_curr_readings: List[Dict[str, Any]] = []
    if _scl_sensors_map and parsed_items:
        try:
            _scl_curr_readings = _wm_extract_scl_readings(
                parsed_items, _scl_sensors_map,
                float(active_operating_rpm),
                float(cr_mil_vault) if cr_mil_vault else 0.0,
            )
        except Exception:
            _scl_curr_readings = []

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📚 Histórico SCL")
        try:
            from core.scl_history import (
                save_scl_snapshot,
                list_scl_snapshots,
                load_scl_snapshot,
                delete_scl_snapshot,
                _scl_snapshot_is_identical_to,
            )
            _scl_hist_ok = True
        except Exception as _e:
            _scl_hist_ok = False
            st.caption(f"_(Histórico SCL no disponible: {_e})_")

        if _scl_hist_ok and _scl_inst_id:
            _scl_existing_snaps = list_scl_snapshots(_scl_inst_id)
            st.caption(
                f"{len(_scl_existing_snaps)} snapshot(s) SCL guardado(s)."
            )

            if not _scl_curr_readings:
                if not _scl_sensors_map:
                    st.caption("_(No hay Sensor Map configurado.)_")
                elif not parsed_items:
                    st.caption("_(No hay CSVs SCL cargados todavía.)_")
                else:
                    st.warning(
                        f"{len(parsed_items)} CSV(s) SCL cargado(s) "
                        f"pero ninguno matchea sensores del Sensor Map."
                    )
                    with st.expander("Diagnóstico — CSVs vs patterns"):
                        _dr = []
                        for it in parsed_items:
                            m = it.get("meta") or {}
                            _dr.append({
                                "Archivo": it.get("file_name", ""),
                                "Point": str(m.get("Point Name", "") or ""),
                                "Paired": str(m.get("Paired Point Name", "") or ""),
                            })
                        if _dr:
                            st.markdown("**CSVs SCL cargados:**")
                            st.dataframe(
                                pd.DataFrame(_dr),
                                width="stretch", hide_index=True,
                            )
                        from core.sensor_map import sensor_label as _diag_slbl
                        _sr = []
                        for s in _scl_sensors_map:
                            if str(s.get("sensor_type", "")).lower() != "proximity":
                                continue
                            _sr.append({
                                "Sensor": _diag_slbl(s),
                                "Plano": s.get("plane_label", "") or "",
                                "Pattern": s.get("csv_match_pattern", "") or "(vacío)",
                            })
                        if _sr:
                            st.markdown("**Sensores proximity del mapa:**")
                            st.dataframe(
                                pd.DataFrame(_sr),
                                width="stretch", hide_index=True,
                            )
            else:
                with st.expander("📸 Guardar snapshot SCL actual", expanded=False):
                    st.caption(
                        f"Captura X/Y position + eccentricity + attitude a "
                        f"{active_operating_rpm:.0f} rpm para "
                        f"{len(_scl_curr_readings)} bearing(s)."
                    )
                    _scl_snap_label = st.text_input(
                        "Etiqueta de la corrida",
                        value="",
                        placeholder="Ej. Coastdown abril 27",
                        key=f"wm_scl_snap_label_{_scl_inst_id}",
                    )
                    _scl_snap_notes = st.text_area(
                        "Observaciones (opcional)",
                        value="",
                        key=f"wm_scl_snap_notes_{_scl_inst_id}",
                        height=70,
                    )
                    if st.button(
                        "Guardar snapshot SCL",
                        type="primary", width="stretch",
                        key=f"wm_scl_snap_save_{_scl_inst_id}",
                    ):
                        try:
                            sid = save_scl_snapshot(
                                _scl_inst_id,
                                operating_speed_rpm=float(active_operating_rpm),
                                bearings_data=_scl_curr_readings,
                                corrida_label=_scl_snap_label,
                                notes=_scl_snap_notes,
                            )
                            st.success(f"✓ Snapshot SCL guardado: {sid}")
                            st.rerun()
                        except Exception as _e:
                            st.error(f"No se pudo guardar: {_e}")

            # Multiselect comparativo
            _selected_scl_cmp_ids: List[str] = []
            if _scl_existing_snaps:
                _curr_by_lbl = {
                    r["bearing_label"]: {
                        "x_gap": r["x_gap_at_op"],
                        "y_gap": r["y_gap_at_op"],
                        "eccentricity_ratio": r["eccentricity_ratio"],
                    }
                    for r in _scl_curr_readings
                }
                _scl_opt_pairs: List[Tuple[str, str]] = []
                _scl_first_non_current = None
                for s in _scl_existing_snaps:
                    _is_current = False
                    if _curr_by_lbl:
                        try:
                            _full = load_scl_snapshot(_scl_inst_id, s["snapshot_id"])
                            if _full is not None:
                                _is_current = _scl_snapshot_is_identical_to(
                                    _full, _curr_by_lbl)
                        except Exception:
                            pass
                    _suffix = " · (corrida actual)" if _is_current else ""
                    _opspeed = s.get("operating_speed_rpm")
                    _opspeed_str = f" @ {_opspeed:.0f}rpm" if _opspeed else ""
                    _lbl = (f"{s['corrida_label'][:28]}{_opspeed_str} "
                            f"({s['timestamp'][:10]}){_suffix}")
                    _scl_opt_pairs.append((s["snapshot_id"], _lbl))
                    if not _is_current and _scl_first_non_current is None:
                        _scl_first_non_current = _lbl

                _scl_opt_lbls = [l for _, l in _scl_opt_pairs]
                _scl_lbl_to_key = {l: k for k, l in _scl_opt_pairs}
                _scl_default_pick = []
                if _scl_first_non_current:
                    _scl_default_pick = [_scl_first_non_current]
                _scl_cmp_state_key = f"wm_scl_cmp_picks_{_scl_inst_id}"
                if _scl_cmp_state_key in st.session_state:
                    _saved = st.session_state[_scl_cmp_state_key]
                    _scl_default_pick = [l for l in _saved if l in _scl_opt_lbls]
                _scl_picked = st.multiselect(
                    "Corridas a superponer en el SCL",
                    options=_scl_opt_lbls,
                    default=_scl_default_pick,
                    key=f"wm_scl_cmp_multi_{_scl_inst_id}",
                    help=(
                        "0 = solo actual; 1 = comparativo simple; "
                        "N = superposición histórica con gradiente "
                        "cronológico de las trayectorias del muñón."
                    ),
                )
                st.session_state[_scl_cmp_state_key] = _scl_picked
                _selected_scl_cmp_ids = [
                    _scl_lbl_to_key[l] for l in _scl_picked
                    if l in _scl_lbl_to_key
                ]

                # Lista borrar
                with st.expander(
                    f"️ Gestionar snapshots SCL ({len(_scl_existing_snaps)})"
                ):
                    for s in _scl_existing_snaps:
                        cols_h = st.columns([4, 1])
                        cols_h[0].markdown(
                            f"**{s['corrida_label'][:30]}**  \n"
                            f"_{s['timestamp']} · {s['n_bearings']} bearings · "
                            f"{s.get('operating_speed_rpm', 0):.0f} rpm_"
                        )
                        if cols_h[1].button(
                            "️",
                            key=f"wm_scl_del_{s['snapshot_id']}",
                            help="Borrar este snapshot",
                        ):
                            if delete_scl_snapshot(_scl_inst_id, s["snapshot_id"]):
                                st.success("Borrado.")
                                st.rerun()

            st.session_state["wm_scl_compare_snapshot_ids"] = _selected_scl_cmp_ids
            st.session_state["wm_scl_compare_inst_id"] = _scl_inst_id

    # Comparativo SCL inline
    _scl_cmp_ids: List[str] = st.session_state.get(
        "wm_scl_compare_snapshot_ids", []) or []
    if _scl_cmp_ids and _scl_curr_readings:
        try:
            from core.scl_history import (
                load_scl_snapshot,
                eccentricity_change_classifier,
                attitude_shift_classifier,
            )
            _cmp_rows = []
            for _snap_id in _scl_cmp_ids:
                _snap_full = load_scl_snapshot(_scl_inst_id, _snap_id)
                if _snap_full is None:
                    continue
                _prev_by_lbl = {
                    str(b.get("bearing_label", "")): b
                    for b in _snap_full.get("bearings", [])
                }
                _slbl = _snap_full.get("corrida_label", _snap_id)[:22]
                _sts = (_snap_full.get("timestamp", "") or "")[:10]

                for r in _scl_curr_readings:
                    _lbl = r["bearing_label"]
                    _prev = _prev_by_lbl.get(_lbl)
                    if _prev is None:
                        continue
                    _prev_x = float(_prev.get("x_gap_at_op", 0))
                    _prev_y = float(_prev.get("y_gap_at_op", 0))
                    _prev_ecc = float(_prev.get("eccentricity_ratio", 0))
                    _prev_att = float(_prev.get("attitude_angle", 0))
                    _prev_lo = float(_prev.get("lift_off_speed", 0))

                    _delta_ecc = r["eccentricity_ratio"] - _prev_ecc
                    _delta_att = r["attitude_angle"] - _prev_att
                    _delta_x = r["x_gap_at_op"] - _prev_x
                    _delta_y = r["y_gap_at_op"] - _prev_y
                    _delta_lo = r["lift_off_speed"] - _prev_lo

                    _ecc_class = eccentricity_change_classifier(_delta_ecc)
                    _att_class = attitude_shift_classifier(_delta_att)

                    _diag = []
                    if _ecc_class == "migration_critical":
                        _diag.append("Migración crítica (>25% clearance)")
                    elif _ecc_class == "migration_major":
                        _diag.append("Migración mayor (>15% clearance)")
                    elif _ecc_class == "migration_minor":
                        _diag.append("Migración menor")
                    elif _ecc_class == "stable":
                        _diag.append("Eccentricity estable")
                    if _att_class == "shift_critical":
                        _diag.append("Shift attitude crítico (≥30°)")
                    elif _att_class == "shift_major":
                        _diag.append("Shift attitude mayor")

                    _cmp_rows.append({
                        "Bearing": _lbl,
                        "vs Corrida": f"{_slbl} ({_sts})",
                        "Anterior X/Y": f"{_prev_x:.2f} / {_prev_y:.2f} mil",
                        "Actual X/Y": f"{r['x_gap_at_op']:.2f} / {r['y_gap_at_op']:.2f} mil",
                        "Δ X/Y": f"{_delta_x:+.2f} / {_delta_y:+.2f}",
                        "e/c anterior": f"{_prev_ecc:.3f}",
                        "e/c actual": f"{r['eccentricity_ratio']:.3f}",
                        "Δ e/c": f"{_delta_ecc:+.3f}",
                        "Anterior attitude": f"{_prev_att:.1f}°",
                        "Actual attitude": f"{r['attitude_angle']:.1f}°",
                        "Δ attitude": f"{_delta_att:+.1f}°",
                        "Diagnóstico": " · ".join(_diag) if _diag else "—",
                    })

            if _cmp_rows:
                st.markdown("### Comparativo SCL — vs corridas anteriores")
                st.caption(
                    "Migración del centerline del muñón entre corridas. "
                    "Cambio de eccentricity ratio o shift de attitude angle "
                    "indican cambio en distribución de carga, viscosidad del "
                    "aceite o pérdida de clearance del cojinete (API 670 §6.7)."
                )
                st.dataframe(
                    pd.DataFrame(_cmp_rows),
                    width="stretch", hide_index=True,
                )
        except Exception as _scl_cmp_e:
            st.caption(f"_(Comparativo SCL no disponible: {_scl_cmp_e})_")

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
            vault_clearance_radial_mil=cr_mil_vault,
            vault_params=vault_params,
            vault_doc_ref=vault_doc_ref,
            profile_label=active_profile_label,
            operating_rpm=float(active_operating_rpm),
            cr_source=cr_source,
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
            vault_clearance_radial_mil=cr_mil_vault,
            vault_params=vault_params,
            vault_doc_ref=vault_doc_ref,
            profile_label=active_profile_label,
            operating_rpm=float(active_operating_rpm),
            cr_source=cr_source,
        )


if __name__ == "__main__":
    main()
