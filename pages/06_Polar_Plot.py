from __future__ import annotations

import base64
import html
import io
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

from core.auth import require_login, render_user_menu
from core.csv_common import (
    circular_mean_deg,
    decode_csv_text,
    filter_status_valid,
    find_header_line,
    parse_metadata_block,
)
from core.diagnostics import (
    build_polar_compare_diagnostics_rotordyn,
    build_polar_diagnostics_rotordyn,
    build_polar_text_diagnostics,
    format_number,
    get_semaforo_status,
)
from core.module_patterns import export_report_row, helper_card, panel_card
from core.profile_state import render_profile_selector  # legacy compat
from core.instance_selector import render_instance_selector
from core.report_state import append_report_item_and_persist
from core.ai_diagnostic import (  # Ciclo 17.26: interpretación clínica AI
    generate_ai_diagnostic,
    is_ai_available,
)
from core.rotordynamics import (
    detect_critical_speeds,
    evaluate_api684_margin,
    iso_20816_2_zone,
    iso_20816_zone_multipart,
    mils_to_micrometers,
)
from core.ui_theme import (
    apply_watermelon_page_style,
    draw_info_box,
    draw_top_strip,
    page_header,
)


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Watermelon System | Polar Plot", layout="wide")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"

apply_watermelon_page_style()


# ============================================================
# POLAR FILE PERSISTENCE
# ============================================================
POLAR_UPLOAD_FILES_KEY = "wm_polar_upload_files"

class PolarPersistedUploadedFile:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def read(self):
        return self._data

    def getvalue(self):
        return self._data

    def seek(self, pos):
        return None


def set_polar_persisted_files(files):
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
            "name": getattr(f, "name", "Polar.csv"),
            "data": data,
        })

    st.session_state[POLAR_UPLOAD_FILES_KEY] = packed


def get_polar_persisted_files():
    return [
        PolarPersistedUploadedFile(item["name"], item["data"])
        for item in st.session_state.get(POLAR_UPLOAD_FILES_KEY, [])
    ]


def clear_polar_persisted_files():
    st.session_state.pop(POLAR_UPLOAD_FILES_KEY, None)


# ============================================================
# HELPERS
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


# circular_mean_deg ahora se importa desde core.csv_common


def circular_smooth_deg(phase_deg: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return phase_deg.astype(float).copy()
    rad = np.deg2rad(phase_deg.astype(float).to_numpy() % 360.0)
    c = pd.Series(np.cos(rad)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    s = pd.Series(np.sin(rad)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    out = np.rad2deg(np.arctan2(s, c))
    out = (out + 360.0) % 360.0
    return pd.Series(out, index=phase_deg.index)


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window is None or window < 2:
        return series.astype(float).copy()

    smoothed = series.astype(float).rolling(window=window, center=True, min_periods=1).mean()
    std = smoothed.std()
    mean = smoothed.mean()

    if pd.notna(std) and pd.notna(mean) and std > 0:
        smoothed = smoothed.clip(lower=mean - 3 * std, upper=mean + 3 * std)

    return smoothed


def nearest_row_for_speed(df: pd.DataFrame, speed_value: float) -> pd.Series:
    idx = int((df["speed"] - speed_value).abs().idxmin())
    return df.loc[idx]


def polar_health_status(
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


# ============================================================
# CSV LOADER
# ============================================================
def read_polar_csv(file_obj) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame]:
    text = decode_csv_text(file_obj, errors="replace")

    lines = text.splitlines()
    if not lines:
        raise ValueError("Archivo vacío.")

    header_idx = find_header_line(
        lines,
        required_signals=("Amp", "Phase", "Speed", "Timestamp"),
    )
    if header_idx is None:
        raise ValueError("No se encontró el encabezado real del CSV Polar.")

    meta = parse_metadata_block(lines[:header_idx])
    data_text = "\n".join(lines[header_idx:])

    df = pd.read_csv(io.StringIO(data_text), encoding="utf-8-sig")

    required = ["Amp", "Amp Status", "Phase", "Phase Status", "Speed", "Speed Status", "Timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    df["amp"] = pd.to_numeric(df["Amp"], errors="coerce")
    df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")
    df["speed"] = pd.to_numeric(df["Speed"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["amp", "phase", "speed", "Timestamp"]).copy()
    df = filter_status_valid(df, ["Amp Status", "Phase Status", "Speed Status"])

    if df.empty:
        raise ValueError("No quedaron filas válidas después del filtrado.")

    raw_df = df.sort_values(["Timestamp", "speed"]).reset_index(drop=True)

    grouped_df = (
        raw_df.groupby("speed", as_index=False)
        .agg(
            amp=("amp", "median"),
            phase=("phase", lambda s: circular_mean_deg(s)),
            samples=("Timestamp", "size"),
            ts_min=("Timestamp", "min"),
            ts_max=("Timestamp", "max"),
        )
        .sort_values("speed", kind="stable")
        .reset_index(drop=True)
    )

    return meta, raw_df, grouped_df


# ============================================================
# POLAR ORIENTATION ENGINE
# ============================================================
def compute_probe_base_angle(axis_label: str, side_label: str, install_angle_deg: float) -> float:
    axis_label = str(axis_label).strip().upper()
    side_label = str(side_label).strip().capitalize()

    base = 0.0
    if axis_label == "X":
        base = 0.0 if side_label == "Right" else 180.0
    elif axis_label == "Y":
        base = 90.0 if side_label == "Right" else 270.0

    return (base + float(install_angle_deg)) % 360.0


def get_polar_axis_rotation_and_direction(
    axis_label: str,
    side_label: str,
    install_angle_deg: float,
    rotation_direction: str,
) -> Tuple[float, str, float]:
    probe_ref = compute_probe_base_angle(axis_label, side_label, install_angle_deg)
    axis_rotation = (90.0 - probe_ref) % 360.0
    angular_direction = "clockwise" if str(rotation_direction).upper() == "CCW" else "counterclockwise"
    return axis_rotation, angular_direction, probe_ref


def compute_polar_display_theta(
    phase_deg: pd.Series,
    axis_label: str,
    side_label: str,
    install_angle_deg: float,
    rotation_direction: str,
) -> pd.Series:
    return phase_deg.astype(float) % 360.0


# ============================================================
# API 684 HEURISTIC FOR POLAR
# ============================================================
def estimate_critical_speeds_api684_style(df: pd.DataFrame, max_count: int = 2) -> List[Dict[str, float]]:
    if df.empty or len(df) < 12:
        return []

    amp = df["amp"].astype(float).to_numpy()
    speed = df["speed"].astype(float).to_numpy()
    phase = df["phase_for_detection"].astype(float).to_numpy()

    candidates: List[Dict[str, float]] = []

    if find_peaks is not None:
        prominence = max(np.nanmax(amp) * 0.08, 0.12)
        distance = max(8, len(df) // 16)
        peaks, props = find_peaks(amp, prominence=prominence, distance=distance)

        for i, p in enumerate(peaks):
            left = max(0, p - 10)
            right = min(len(df) - 1, p + 10)

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
                    "speed": float(speed[p]),
                    "amp": amp_peak,
                    "phase_delta": phase_delta,
                    "idx": int(p),
                    "prominence": prom,
                }
            )
    else:
        p = int(np.nanargmax(amp))
        left = max(0, p - 10)
        right = min(len(df) - 1, p + 10)
        candidates.append(
            {
                "speed": float(speed[p]),
                "amp": float(amp[p]),
                "phase_delta": float(phase[right] - phase[left]),
                "idx": int(p),
                "prominence": float(amp[p]),
            }
        )

    candidates = sorted(candidates, key=lambda x: (x["prominence"], x["amp"]), reverse=True)

    filtered = []
    for cand in candidates:
        if all(abs(cand["speed"] - kept["speed"]) > 120 for kept in filtered):
            filtered.append(cand)
        if len(filtered) >= max_count:
            break

    filtered = sorted(filtered, key=lambda x: x["speed"])
    return filtered


# ============================================================
# FIGURE
# ============================================================
def build_probe_reference_overlay(fig: go.Figure, max_r: float) -> None:
    ref_r0 = max_r * 0.10
    ref_r1 = max_r * 0.98

    fig.add_trace(
        go.Scatterpolar(
            r=[ref_r0, ref_r1],
            theta=[0, 0],
            mode="lines",
            line=dict(color="#111827", width=2.2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    body_r0 = max_r * 1.02
    body_r1 = max_r * 1.12
    fig.add_trace(
        go.Scatterpolar(
            r=[body_r0, body_r1],
            theta=[0, 0],
            mode="lines",
            line=dict(color="#111827", width=5.0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    tip_r = max_r * 1.145
    fig.add_trace(
        go.Scatterpolar(
            r=[tip_r],
            theta=[0],
            mode="markers",
            marker=dict(size=11, color="#111827", symbol="diamond"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    cone_r = [max_r * 1.00, max_r * 1.06, max_r * 1.00]
    cone_t = [-4, 0, 4]
    fig.add_trace(
        go.Scatterpolar(
            r=cone_r,
            theta=cone_t,
            mode="lines",
            line=dict(color="#111827", width=2.0),
            fill="toself",
            fillcolor="rgba(17,24,39,0.12)",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[max_r * 1.18],
            theta=[0],
            mode="text",
            text=["Probe"],
            textposition="top center",
            textfont=dict(size=10, color="#111827"),
            showlegend=False,
            hoverinfo="skip",
        )
    )


def build_polar_figure(
    df: pd.DataFrame,
    meta: Dict[str, str],
    row_a: pd.Series,
    row_b: pd.Series,
    logo_uri: Optional[str],
    show_info_box: bool,
    show_rpm_labels: bool,
    marker_stride: int,
    axis_label: str,
    side_label: str,
    install_angle_deg: float,
    rotation_direction: str,
    critical_speeds: List[Dict[str, float]],
    semaforo_status: str,
    semaforo_color: str,
    *,
    operating_rpm: Optional[float] = None,
    iso_thresholds: Optional[Dict[str, float]] = None,
    critical_speeds_pro: Optional[List[Dict[str, Any]]] = None,
    # Ciclo 17.1 P3 — overlay del snapshot anterior
    prev_snapshot_amp: Optional[float] = None,
    prev_snapshot_phase: Optional[float] = None,
    prev_snapshot_label: Optional[str] = None,
    prev_snapshot_op_speed: Optional[float] = None,
    # Ciclo 17.1.1 — multi-snapshot overlays (lista de dicts con
    # {amp, phase, label, op_speed, timestamp, trajectory_speed,
    # trajectory_amp, trajectory_phase}). Si está, prevalece sobre
    # single-snapshot legacy. Cada uno con color gradient cronologico
    # (mas viejo = azul claro, mas reciente = rojo). Si trajectory_*
    # está presente, se dibuja el LOOP COMPLETO superpuesto (lo que
    # permite ver paso por la velocidad critica entre corridas).
    prev_snapshots: Optional[List[Dict[str, Any]]] = None,
) -> go.Figure:
    amp_unit = meta.get("Amp Unit", "") or ""
    speed_unit = meta.get("Speed Unit", "rpm") or "rpm"

    axis_rotation, angular_direction, _ = get_polar_axis_rotation_and_direction(
        axis_label=axis_label,
        side_label=side_label,
        install_angle_deg=install_angle_deg,
        rotation_direction=rotation_direction,
    )
    max_r = max(0.1, float(df["amp"].max()) * 1.18)

    # Permitir que los anillos ISO empujen el max_r si caen fuera de la curva
    if iso_thresholds is not None:
        cd_iso = float(iso_thresholds.get("CD", 0.0))
        if cd_iso > max_r:
            max_r = cd_iso * 1.10

    fig = go.Figure()

    # ============================================================
    # ISO 20816-2 ZONE RINGS — anillos concéntricos sobre el polar
    # ============================================================
    if iso_thresholds is not None:
        ab = float(iso_thresholds.get("AB", 0.0))
        bc = float(iso_thresholds.get("BC", 0.0))
        cd = float(iso_thresholds.get("CD", 0.0))
        if ab > 0 and bc > ab and cd > bc:
            theta_circle = np.linspace(0, 360, 181)

            # Anillos como superficies anulares (relleno entre dos radios)
            for r_outer, r_inner, fill_color, label_letter in (
                (ab, 0.0, "rgba(34, 197, 94, 0.10)", "A"),
                (bc, ab, "rgba(234, 179, 8, 0.10)", "B"),
                (cd, bc, "rgba(249, 115, 22, 0.13)", "C"),
                (max_r, cd, "rgba(220, 38, 38, 0.15)", "D"),
            ):
                # Borde de cada zona (anillo punteado tenue)
                fig.add_trace(
                    go.Scatterpolar(
                        r=[r_outer] * len(theta_circle),
                        theta=theta_circle,
                        mode="lines",
                        line=dict(width=1.2, color="rgba(100,116,139,0.45)", dash="dot"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                # Etiqueta de la zona en ángulo 60° (esquina superior derecha)
                if r_outer > 0:
                    label_r = (r_outer + r_inner) / 2.0 if r_inner > 0 else r_outer * 0.5
                    fig.add_trace(
                        go.Scatterpolar(
                            r=[label_r],
                            theta=[60.0],
                            mode="text",
                            text=[f"<b>{label_letter}</b>"],
                            textfont=dict(size=11, color="#475569"),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

    fig.add_trace(
        go.Scatterpolar(
            r=df["amp"],
            theta=df["theta_display"],
            mode="lines",
            line=dict(width=1.9, color="#5b9cf0"),
            hovertemplate=(
                f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                f"Phase Display: %{{theta:.1f}}°<br>"
                f"Speed: %{{customdata[0]:.0f}} {speed_unit}<extra></extra>"
            ),
            customdata=np.stack([df["speed"]], axis=1),
            showlegend=False,
            name="Polar Path",
        )
    )

    for row, color, name in [
        (row_a, "#efb08c", "Cursor A"),
        (row_b, "#7ac77b", "Cursor B"),
    ]:
        fig.add_trace(
            go.Scatterpolar(
                r=[row["amp"]],
                theta=[row["theta_display"]],
                mode="markers",
                marker=dict(size=10, color=color, line=dict(width=1.2, color="#ffffff")),
                name=name,
                showlegend=False,
                hovertemplate=(
                    f"{name}<br>"
                    f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                    f"Phase Display: %{{theta:.1f}}°<br>"
                    f"Speed: {int(round(row['speed']))} {speed_unit}<extra></extra>"
                ),
            )
        )

    if show_rpm_labels and len(df) > 0:
        idxs = list(range(0, len(df), max(1, marker_stride)))
        if idxs[-1] != len(df) - 1:
            idxs.append(len(df) - 1)

        fig.add_trace(
            go.Scatterpolar(
                r=df.iloc[idxs]["amp"],
                theta=df.iloc[idxs]["theta_display"],
                mode="text",
                text=[str(int(round(v))) for v in df.iloc[idxs]["speed"]],
                textfont=dict(size=9, color="#6b7280"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # ============================================================
    # CRÍTICAS PRO con label enriquecido (RPM + Q)
    # ============================================================
    if critical_speeds_pro:
        cs_pro_colors = ["#dc2626", "#ea580c", "#9333ea"]
        for idx, cs_pro in enumerate(critical_speeds_pro):
            color = cs_pro_colors[idx % len(cs_pro_colors)]
            cs_rpm_pro = float(cs_pro.get("rpm", 0.0))
            q_pro = cs_pro.get("q_factor")
            label_q = f"Q={q_pro:.2f}" if (q_pro is not None and np.isfinite(q_pro)) else "Q=—"
            cs_row_pro = nearest_row_for_speed(df, cs_rpm_pro)

            fig.add_trace(
                go.Scatterpolar(
                    r=[cs_row_pro["amp"]],
                    theta=[cs_row_pro["theta_display"]],
                    mode="markers+text",
                    marker=dict(
                        size=14, color=color, symbol="diamond",
                        line=dict(width=2, color="white"),
                    ),
                    text=[f"<b>{int(round(cs_rpm_pro))} rpm · {label_q}</b>"],
                    textposition="top center",
                    textfont=dict(size=11, color=color, family="Arial Black"),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Crítica #{idx+1}</b><br>"
                        f"RPM: {int(round(cs_rpm_pro))}<br>"
                        f"{label_q}<br>"
                        f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                        f"Phase Display: %{{theta:.1f}}°<extra></extra>"
                    ),
                )
            )
    else:
        # Modo legacy (sin rotordyn pro): conservamos los CS antiguos
        cs_colors = ["#ef4444", "#f59e0b"]
        for idx, cs in enumerate(critical_speeds):
            color = cs_colors[idx % len(cs_colors)]
            cs_row = nearest_row_for_speed(df, cs["speed"])

            fig.add_trace(
                go.Scatterpolar(
                    r=[cs_row["amp"]],
                    theta=[cs_row["theta_display"]],
                    mode="markers+text",
                    marker=dict(size=9, color=color, symbol="diamond"),
                    text=[f"CS{idx+1} {int(round(cs['speed']))}"],
                    textposition="top center",
                    textfont=dict(size=10, color=color),
                    showlegend=False,
                    hovertemplate=(
                        f"Critical Speed {idx+1}<br>"
                        f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                        f"Phase Display: %{{theta:.1f}}°<br>"
                        f"Speed: {int(round(cs['speed']))} {speed_unit}<extra></extra>"
                    ),
                )
            )

    # ============================================================
    # OPERATING SPEED — marker estrella con label
    # ============================================================
    if operating_rpm is not None:
        sp_min = float(df["speed"].min()) if len(df) else 0.0
        sp_max = float(df["speed"].max()) if len(df) else 0.0
        if sp_min <= operating_rpm <= sp_max:
            op_row = nearest_row_for_speed(df, operating_rpm)

            # ============================================================
            # Ciclo 17.1.1 — Overlay multi-snapshot (lista) o single
            # legacy. Cada snapshot anterior se dibuja con marker
            # ghost + linea conectora dotted al actual. Color en
            # gradiente cronologico: mas viejo = azul claro, mas
            # reciente = rojo intenso. La paleta es Viridis-ish para
            # buena percepción de "edad".
            # ============================================================
            _curr_amp = float(op_row["amp"])
            _curr_phase = float(op_row["theta_display"])

            # Construir la lista efectiva de snapshots a dibujar
            _snaps_to_draw: List[Dict[str, Any]] = []
            if prev_snapshots:
                # Ordenar por timestamp asc (mas viejo primero) para
                # mapear el gradiente correctamente
                _snaps_to_draw = sorted(
                    [s for s in prev_snapshots if s.get("amp", 0) > 0],
                    key=lambda s: s.get("timestamp", "") or "",
                )
            elif (
                prev_snapshot_amp is not None
                and prev_snapshot_phase is not None
                and prev_snapshot_amp > 0
            ):
                _snaps_to_draw = [{
                    "amp": float(prev_snapshot_amp),
                    "phase": float(prev_snapshot_phase),
                    "label": prev_snapshot_label or "anterior",
                    "op_speed": prev_snapshot_op_speed,
                    "timestamp": "",
                }]

            # Paleta cronologica: mas viejo = azul claro (#7dd3fc),
            # medio = ambar (#f59e0b), mas reciente = rojo (#dc2626).
            # Si solo hay 1, usamos gris/rojo neutro.
            def _gradient_color(idx: int, total: int) -> Tuple[str, str]:
                """Devuelve (marker_color, line_color) hex segun posicion."""
                if total <= 1:
                    return ("rgba(148,163,184,0.55)",
                            "rgba(220,38,38,0.55)")
                # Interpolacion lineal en HSV-ish via 3 paradas
                # 0.0 -> light blue
                # 0.5 -> amber
                # 1.0 -> red (mas cercano al actual)
                pos = idx / max(1, total - 1)
                stops = [
                    (0.00, (125, 211, 252)),   # light blue
                    (0.50, (245, 158,  11)),   # amber
                    (1.00, (220,  38,  38)),   # red
                ]
                # Encontrar segmento
                for i in range(len(stops) - 1):
                    t0, c0 = stops[i]
                    t1, c1 = stops[i + 1]
                    if t0 <= pos <= t1:
                        frac = (pos - t0) / (t1 - t0)
                        r = int(c0[0] + (c1[0] - c0[0]) * frac)
                        g = int(c0[1] + (c1[1] - c0[1]) * frac)
                        b = int(c0[2] + (c1[2] - c0[2]) * frac)
                        return (
                            f"rgba({r},{g},{b},0.65)",
                            f"rgba({r},{g},{b},0.75)",
                        )
                return ("rgba(148,163,184,0.55)",
                        "rgba(220,38,38,0.55)")

            for _idx, _snap in enumerate(_snaps_to_draw):
                _prev_amp = float(_snap["amp"])
                _prev_phase_disp = float(_snap["phase"]) % 360.0
                _prev_lbl_text = _snap.get("label", "anterior") or "anterior"
                _prev_op = _snap.get("op_speed")
                _prev_op_text = (
                    f" @ {int(round(_prev_op))} rpm" if _prev_op else ""
                )

                _marker_color, _line_color = _gradient_color(
                    _idx, len(_snaps_to_draw)
                )

                # ============================================================
                # Ciclo 17.1.2 — TRAYECTORIA COMPLETA superpuesta
                # Si el snapshot trae trajectory_speed/amp/phase, dibujamos
                # el loop polar entero del run-up/coast-down. Asi se ve el
                # paso por la velocidad critica (peak de la curva), la
                # forma del loop (sub-síncronos, fase de Bode) y como
                # cambio entre corridas. Lo dibujamos PRIMERO (zorder bajo)
                # para que el actual quede arriba.
                # ============================================================
                _traj_speed = _snap.get("trajectory_speed") or []
                _traj_amp = _snap.get("trajectory_amp") or []
                _traj_phase = _snap.get("trajectory_phase") or []
                _has_full_traj = (
                    len(_traj_speed) > 1
                    and len(_traj_speed) == len(_traj_amp) == len(_traj_phase)
                )
                if _has_full_traj:
                    _traj_theta = [float(p) % 360.0 for p in _traj_phase]
                    fig.add_trace(
                        go.Scatterpolar(
                            r=_traj_amp,
                            theta=_traj_theta,
                            mode="lines",
                            line=dict(
                                width=1.8,
                                color=_line_color,
                                dash="solid",
                            ),
                            opacity=0.55,
                            customdata=np.array(_traj_speed).reshape(-1, 1),
                            hovertemplate=(
                                f"<b>{_prev_lbl_text}</b><br>"
                                f"Speed: %{{customdata[0]:.0f}} rpm<br>"
                                f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                                f"Phase: %{{theta:.1f}}°<extra></extra>"
                            ),
                            showlegend=False,
                            name=f"Polar {_prev_lbl_text}",
                        )
                    )
                    # Marcador del PICO (max amplitud) de la trayectoria
                    # historica = velocidad critica de esa corrida.
                    try:
                        _i_pk = int(np.argmax(_traj_amp))
                        _peak_amp = float(_traj_amp[_i_pk])
                        _peak_phase = float(_traj_theta[_i_pk])
                        _peak_speed = float(_traj_speed[_i_pk])
                        fig.add_trace(
                            go.Scatterpolar(
                                r=[_peak_amp],
                                theta=[_peak_phase],
                                mode="markers",
                                marker=dict(
                                    size=12,
                                    color=_marker_color,
                                    symbol="diamond-open",
                                    line=dict(width=2.0, color="#0f172a"),
                                ),
                                opacity=0.85,
                                showlegend=False,
                                hovertemplate=(
                                    f"<b>Pico {_prev_lbl_text}</b><br>"
                                    f"Speed: {_peak_speed:.0f} rpm<br>"
                                    f"Amp: {_peak_amp:.3f} {amp_unit}<br>"
                                    f"Phase: {_peak_phase:.1f}°<extra></extra>"
                                ),
                            )
                        )
                    except Exception:
                        pass
                else:
                    # Sin trayectoria — fallback: linea conectora simple
                    # del snapshot anterior al punto actual (legacy).
                    _n_seg = 12
                    _frac = np.linspace(0.0, 1.0, _n_seg)
                    _seg_r = _prev_amp + (_curr_amp - _prev_amp) * _frac
                    _delta_seg = ((_curr_phase - _prev_phase_disp + 540.0) % 360.0) - 180.0
                    _seg_theta = (_prev_phase_disp + _delta_seg * _frac) % 360.0
                    fig.add_trace(
                        go.Scatterpolar(
                            r=_seg_r,
                            theta=_seg_theta,
                            mode="lines",
                            line=dict(width=2.0, color=_line_color, dash="dot"),
                            showlegend=False,
                            hoverinfo="skip",
                            name=f"Trail {_prev_lbl_text} → actual",
                        )
                    )

                # Marker GHOST del punto operativo del snapshot anterior
                _delta_ph = ((_curr_phase - _prev_phase_disp + 540.0) % 360.0) - 180.0
                fig.add_trace(
                    go.Scatterpolar(
                        r=[_prev_amp],
                        theta=[_prev_phase_disp],
                        mode="markers+text",
                        marker=dict(
                            size=18, color=_marker_color,
                            symbol="star-open",
                            line=dict(width=2.5, color="#0f172a"),
                        ),
                        text=[f"<i>{_prev_lbl_text[:18]}{_prev_op_text}</i>"],
                        textposition="top center",
                        textfont=dict(size=9, color="#0f172a", family="Arial"),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>Op {_prev_lbl_text}</b><br>"
                            f"Amplitude: {_prev_amp:.3f} {amp_unit}<br>"
                            f"Phase: {_prev_phase_disp:.1f}°<br>"
                            f"Δ amp: {(_curr_amp - _prev_amp):+.3f} "
                            f"({((_curr_amp - _prev_amp) / _prev_amp * 100.0):+.1f}%)<br>"
                            f"Δ phase: {_delta_ph:+.1f}°<extra></extra>"
                        ),
                    )
                )

            fig.add_trace(
                go.Scatterpolar(
                    r=[op_row["amp"]],
                    theta=[op_row["theta_display"]],
                    mode="markers+text",
                    marker=dict(
                        size=18, color="#0f172a", symbol="star",
                        line=dict(width=2, color="white"),
                    ),
                    text=[f"<b>Op. {int(round(operating_rpm))} rpm</b>"],
                    textposition="bottom center",
                    textfont=dict(size=10, color="#0f172a", family="Arial Black"),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Operación nominal</b><br>"
                        f"RPM: {int(round(operating_rpm))}<br>"
                        f"Amplitude: %{{r:.3f}} {amp_unit}<br>"
                        f"Phase Display: %{{theta:.1f}}°<extra></extra>"
                    ),
                )
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
        variable=f"{meta.get('Variable', '-')} | {axis_label} | {install_angle_deg:.0f}° {side_label} | Rotation {rotation_direction}",
        dt_text=dt_text,
        rpm_text=f"{int(round(df['speed'].min()))} - {int(round(df['speed'].max()))} {speed_unit}",
        logo_uri=logo_uri,
    )

    if show_info_box:
        rows = [
            ("Cursor A", f"{format_number(row_a['amp'],3)} {amp_unit} @ {int(round(row_a['speed']))} {speed_unit} | ∠{format_number(row_a['theta_display'],1)}°"),
            ("Cursor B", f"{format_number(row_b['amp'],3)} {amp_unit} @ {int(round(row_b['speed']))} {speed_unit} | ∠{format_number(row_b['theta_display'],1)}°"),
            ("Probe Orientation", f"{axis_label} | {install_angle_deg:.0f}° {side_label}"),
            ("Rotation", rotation_direction),
            ("RPM Labels", "Enabled" if show_rpm_labels else "Disabled"),
            ("Label Step", f"Every {marker_stride} points"),
            ("Status", f"<span style='color:{semaforo_color};'><b>{semaforo_status}</b></span>"),
        ]

        for i, cs in enumerate(critical_speeds, start=1):
            title = f"Critical Speed {i}" if i == 1 else f"Secondary Candidate {i}"
            rows.append((title, f"{int(round(cs['speed']))} {speed_unit} | {format_number(cs['amp'],3)} {amp_unit}"))
            rows.append((f"Phase Delta {i}", f"{format_number(cs['phase_delta'],1)}°"))

        draw_info_box(fig=fig, title="Polar Information", rows=rows)

    fig.update_layout(
        polar=dict(
            domain=dict(x=[0.0, 0.78] if show_info_box else [0.0, 1.0], y=[0.05, 0.96]),
            bgcolor="#f8fafc",
            angularaxis=dict(
                rotation=axis_rotation,
                direction=angular_direction,
                tickfont=dict(size=12, color="#111827"),
                gridcolor="rgba(148, 163, 184, 0.18)",
                linecolor="#9ca3af",
                showline=True,
                ticks="outside",
            ),
            radialaxis=dict(
                range=[0, max_r],
                tickfont=dict(size=11, color="#111827"),
                gridcolor="rgba(148, 163, 184, 0.18)",
                linecolor="#9ca3af",
                showline=True,
                ticks="outside",
                angle=225,
            ),
        ),
        height=860,
        margin=dict(l=48, r=20, t=145, b=48),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f3f4f6",
        font=dict(color="#111827"),
        showlegend=False,
    )

    build_probe_reference_overlay(fig, max_r)

    return fig


# ============================================================
# EXPORT / REPORT
# ============================================================
def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    return go.Figure(fig.to_dict())


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    for trace in fig.data:
        tj = trace.to_plotly_json()
        mode = tj.get("mode", "") or ""

        if "lines" in mode:
            line = dict(tj.get("line", {}) or {})
            line["width"] = max(3.2, float(line.get("width", 1.0)) * 2.0)
            trace.line = line

        if "markers" in mode:
            marker = dict(tj.get("marker", {}) or {})
            marker["size"] = max(10, float(marker.get("size", 6)) * 1.5)
            trace.marker = marker

        if "text" in mode:
            textfont = dict(tj.get("textfont", {}) or {})
            textfont["size"] = max(16, int(float(textfont.get("size", 10)) * 1.8))
            trace.textfont = textfont

    fig.update_layout(
        width=4300,
        height=2900,
        margin=dict(l=110, r=80, t=320, b=760),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=26, color="#111827"),
    )

    polar_cfg = dict(fig.layout.polar.to_plotly_json()) if getattr(fig.layout, "polar", None) is not None else {}
    domain_cfg = dict(polar_cfg.get("domain", {}) or {})
    current_x = domain_cfg.get("x", [0.0, 0.78])

    domain_cfg["x"] = [current_x[0], min(0.80, current_x[1])]
    domain_cfg["y"] = [0.06, 0.95]
    polar_cfg["domain"] = domain_cfg

    angular_cfg = dict(polar_cfg.get("angularaxis", {}) or {})
    angular_cfg["tickfont"] = dict(size=22, color="#111827")
    polar_cfg["angularaxis"] = angular_cfg

    radial_cfg = dict(polar_cfg.get("radialaxis", {}) or {})
    radial_cfg["tickfont"] = dict(size=20, color="#111827")
    polar_cfg["radialaxis"] = radial_cfg

    fig.update_layout(polar=polar_cfg)

    for ann in fig.layout.annotations or []:
        if ann.font is not None:
            ann.font.size = max(20, int((ann.font.size or 12) * 1.75))

    for img in fig.layout.images or []:
        sx = getattr(img, "sizex", None)
        sy = getattr(img, "sizey", None)
        if sx is not None:
            img.sizex = sx * 1.10
        if sy is not None:
            img.sizey = sy * 1.10

    return fig


def _build_polar_report_notes(text_diag: Dict[str, str]) -> str:
    headline = str(text_diag.get("headline", "") or "").strip()
    detail = str(text_diag.get("detail", "") or "").strip()
    action = str(text_diag.get("action", "") or "").strip()
    # Ciclo 17.1 P5 — narrativa del comparativo Polar (vs corrida anterior)
    comparison_narrative = str(text_diag.get("comparison_narrative", "") or "").strip()

    def clean_text(value: str) -> str:
        value = str(value or "")
        value = value.replace("\\n", "\n")
        value = value.replace("\r", "")
        value = re.sub(r"\n{3,}", "\n\n", value)
        value = value.replace("Se recomienda se recomienda:", "Se recomienda:")
        value = value.replace("Se recomienda: Se recomienda:", "Se recomienda:")
        return value.strip()

    headline = clean_text(headline)
    detail = clean_text(detail)
    action = clean_text(action)
    comparison_narrative = clean_text(comparison_narrative)

    blocks: List[str] = []
    if headline:
        blocks.append(headline)
    if detail:
        blocks.append(detail)
    # Inyectar el comparativo después del diagnostico principal y antes
    # de las acciones priorizadas — lo natural en un reporte de
    # vibraciones es ver primero la severidad actual, luego el contexto
    # historico, y por ultimo las recomendaciones.
    if comparison_narrative:
        blocks.append(comparison_narrative)

    if action:
        action_clean = action
        action_clean = re.sub(r"^Se recomienda:\s*", "", action_clean, flags=re.IGNORECASE)
        action_clean = action_clean.strip()
        if action_clean:
            blocks.append("Se recomienda:\n" + action_clean)

    return "\n\n".join([b for b in blocks if b]).strip()


def _add_export_diagnostic_footer(fig: go.Figure, text_diag: Dict[str, str]) -> go.Figure:
    headline = str(text_diag.get("headline", "") or "").strip()
    detail = str(text_diag.get("detail", "") or "").strip()
    action = str(text_diag.get("action", "") or "").strip()

    if not any([headline, detail, action]):
        return go.Figure(fig)

    export_fig = go.Figure(fig)

    current_annotations = list(export_fig.layout.annotations) if export_fig.layout.annotations else []
    current_shapes = list(export_fig.layout.shapes) if export_fig.layout.shapes else []

    footer_y0 = -0.285
    footer_y1 = -0.035

    current_shapes.extend(
        [
            dict(
                type="line",
                xref="paper",
                yref="paper",
                x0=0.03,
                x1=0.97,
                y0=-0.008,
                y1=-0.008,
                line=dict(color="rgba(148,163,184,0.55)", width=2),
            ),
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.03,
                x1=0.97,
                y0=footer_y0,
                y1=footer_y1,
                line=dict(color="rgba(148,163,184,0.55)", width=2),
                fillcolor="rgba(255,255,255,0.98)",
                layer="below",
            ),
        ]
    )

    summary_html = (
        f"<b>{html.escape(headline)}</b><br><br>"
        f"<b>Detail:</b> {html.escape(detail)}<br><br>"
        f"<b>Action:</b> {html.escape(action)}"
    )

    current_annotations.extend(
        [
            dict(
                xref="paper",
                yref="paper",
                x=0.05,
                y=-0.055,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                text="<b>DIAGNOSTIC SUMMARY</b>",
                font=dict(size=24, color="#0f172a"),
            ),
            dict(
                xref="paper",
                yref="paper",
                x=0.05,
                y=-0.112,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                text=summary_html,
                font=dict(size=20, color="#111827"),
            ),
        ]
    )

    export_fig.update_layout(
        annotations=current_annotations,
        shapes=current_shapes,
    )
    return export_fig


def build_export_png_bytes(fig: go.Figure, text_diag: Dict[str, str]) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        # Imagen limpia: el diagnóstico va debajo en el reporte, no incrustado en el PNG.
        # Ciclo 23.155 — anti-OOM: vía core.plot_export.fig_to_png_bytes (decima + scale=1).
        from core.plot_export import fig_to_png_bytes
        return fig_to_png_bytes(export_fig, width=2400, height=1620, scale=1)
    except Exception as e:
        return None, str(e)


def queue_polar_to_report(
    meta: Dict[str, str],
    fig: go.Figure,
    title: str,
    text_diag: Dict[str, str],
    image_bytes: Optional[bytes] = None,
    notes_override: Optional[str] = None,
) -> None:
    """Encola la figura Polar al reporte. Ciclo 17.26: si
    `notes_override` viene con contenido (típicamente bloque AI con
    marcadores <<<WM_AI_BLOCK>>>), reemplaza la narrativa
    determinística de _build_polar_report_notes."""
    ensure_report_state()
    final_notes = (
        notes_override
        if notes_override is not None and notes_override.strip()
        else _build_polar_report_notes(text_diag)
    )
    append_report_item_and_persist(
        {
            "id": f"report-polar-{meta.get('Machine Name','')}-{meta.get('Point Name','')}-{title}",
            "type": "polar",
            "title": title,
            "notes": final_notes,
            "signal_id": meta.get("Point Name", ""),
            "image_bytes": image_bytes,
            "machine": meta.get("Machine Name", ""),
            "point": meta.get("Point Name", ""),
            "variable": meta.get("Variable", ""),
            "timestamp": "",
        }
    )



# ============================================================
# POLAR PRO OVERRIDES - DIAGNOSTIC + CLEAN HD EXPORT
# ============================================================
def build_polar_text_diagnostics(
    status: str,
    critical_speeds: List[Dict[str, float]],
    max_amp: float,
) -> Dict[str, str]:
    status_up = str(status or "").upper()
    max_amp = float(max_amp or 0.0)

    if critical_speeds:
        dominant = critical_speeds[0]
        cs_speed = float(dominant.get("speed", 0.0) or 0.0)
        cs_amp = float(dominant.get("amp", 0.0) or 0.0)
        phase_delta = float(dominant.get("phase_delta", 0.0) or 0.0)

        if status_up == "DANGER":
            headline = f"Respuesta polar severa compatible con amplificación dinámica cerca de {cs_speed:.0f} rpm"
            detail = (
                f"La trayectoria polar evidencia una respuesta dinámica significativa alrededor de {cs_speed:.0f} rpm, "
                f"con amplitud aproximada de {cs_amp:.3f} y variación de fase de {phase_delta:.1f}°. "
                f"La combinación de incremento de amplitud y cambio de fase sugiere proximidad a una velocidad crítica, "
                f"pérdida de margen dinámico o cambio relevante de rigidez/amortiguamiento del sistema rotor-soporte.\n\n"
                f"Desde el punto de vista de dinámica de rotores, esta condición debe correlacionarse con Bode, órbitas, "
                f"forma de onda, shaft centerline y condiciones reales de carga."
            )
            action = (
                "Se recomienda como acción prioritaria:\n"
                "- Correlacionar el pico polar con Bode de amplitud y fase\n"
                "- Confirmar repetibilidad durante arranque/parada\n"
                "- Verificar alineación, rigidez de soporte, balance y condición de cojinetes\n"
                "- Revisar el cambio de fase alrededor del régimen identificado\n"
                "- Evitar operación sostenida cerca del régimen crítico hasta completar evaluación"
            )
        elif status_up == "WARNING":
            headline = f"Respuesta polar con indicios de amplificación dinámica cerca de {cs_speed:.0f} rpm"
            detail = (
                f"La trayectoria polar muestra una zona de respuesta relevante alrededor de {cs_speed:.0f} rpm, "
                f"con amplitud aproximada de {cs_amp:.3f} y cambio de fase de {phase_delta:.1f}°. "
                f"El comportamiento es consistente con amplificación dinámica moderada, sin evidencia suficiente para clasificarla como severa.\n\n"
                f"Desde el enfoque de análisis de vibraciones, esta condición debe mantenerse bajo seguimiento, especialmente si el pico se repite "
                f"en corridas posteriores o si se acompaña de incremento en 1X, cambio de fase o alteración de órbita."
            )
            action = (
                "Se recomienda:\n"
                "- Comparar contra corridas históricas y condición base\n"
                "- Validar la respuesta con Bode y espectro 1X\n"
                "- Confirmar si existe tendencia creciente de amplitud\n"
                "- Mantener seguimiento durante próximos arranques/paradas"
            )
        else:
            headline = f"Respuesta polar controlada con candidato dinámico cerca de {cs_speed:.0f} rpm"
            detail = (
                f"Se identifica un candidato dinámico alrededor de {cs_speed:.0f} rpm, con amplitud aproximada de {cs_amp:.3f} "
                f"y cambio de fase de {phase_delta:.1f}°. La trayectoria polar no evidencia una respuesta severa en esta condición.\n\n"
                f"El comportamiento es compatible con operación estable, aunque el punto identificado debe conservarse como referencia para comparación futura."
            )
            action = (
                "Se recomienda:\n"
                "- Mantener la corrida como línea base\n"
                "- Comparar con futuras trayectorias polares\n"
                "- Correlacionar con Bode, órbita y tendencia de amplitud 1X"
            )
    else:
        headline = "Respuesta polar sin velocidad crítica dominante claramente identificada"
        detail = (
            f"La trayectoria polar presenta amplitud máxima de {max_amp:.3f} y no muestra un candidato claro de velocidad crítica bajo la heurística aplicada. "
            f"La condición debe correlacionarse con Bode, espectro, órbita y variables operativas antes de concluir el mecanismo dominante."
        )
        action = (
            "Se recomienda:\n"
            "- Mantener seguimiento histórico\n"
            "- Comparar contra futuras corridas\n"
            "- Validar con Bode, espectro y órbita si cambia la condición"
        )

    return {"headline": headline, "detail": detail, "action": action}


def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    fig = go.Figure(export_fig)

    fig.update_layout(
        width=4300,
        height=2450,
        margin=dict(l=60, r=50, t=220, b=120),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=25, color="#111827"),
    )

    polar_cfg = dict(fig.layout.polar.to_plotly_json()) if getattr(fig.layout, "polar", None) is not None else {}
    domain_cfg = dict(polar_cfg.get("domain", {}) or {})
    domain_cfg["x"] = [0.01, 0.86]
    domain_cfg["y"] = [0.02, 0.98]
    polar_cfg["domain"] = domain_cfg

    angular_cfg = dict(polar_cfg.get("angularaxis", {}) or {})
    angular_cfg["tickfont"] = dict(size=22, color="#111827")
    angular_cfg["gridcolor"] = "rgba(148, 163, 184, 0.18)"
    polar_cfg["angularaxis"] = angular_cfg

    radial_cfg = dict(polar_cfg.get("radialaxis", {}) or {})
    radial_cfg["tickfont"] = dict(size=20, color="#111827")
    radial_cfg["gridcolor"] = "rgba(148, 163, 184, 0.18)"
    polar_cfg["radialaxis"] = radial_cfg

    fig.update_layout(polar=polar_cfg)

    for trace in fig.data:
        tj = trace.to_plotly_json()
        mode = tj.get("mode", "") or ""
        if "lines" in mode and hasattr(trace, "line"):
            line = dict(tj.get("line", {}) or {})
            line["width"] = max(3.0, float(line.get("width", 1.0)) * 1.8)
            trace.line = line
        if "markers" in mode and hasattr(trace, "marker"):
            marker = dict(tj.get("marker", {}) or {})
            marker["size"] = max(9, float(marker.get("size", 6)) * 1.35)
            trace.marker = marker
        if "text" in mode:
            textfont = dict(tj.get("textfont", {}) or {})
            textfont["size"] = max(15, int(float(textfont.get("size", 10)) * 1.6))
            trace.textfont = textfont

    return fig


def build_export_png_bytes(fig: go.Figure, text_diag: Dict[str, str]) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        export_fig = _build_export_safe_figure(fig)
        export_fig = _scale_export_figure(export_fig)
        # Ciclo 23.155 — anti-OOM: vía core.plot_export.fig_to_png_bytes (decima + scale=1).
        from core.plot_export import fig_to_png_bytes
        return fig_to_png_bytes(export_fig, width=2400, height=1360, scale=1)
    except Exception as e:
        return None, str(e)



# ============================================================
# POLAR PRO MODAL DIAGNOSTICS + COMPARISON
# ============================================================
def build_polar_text_diagnostics(
    status: str,
    critical_speeds: List[Dict[str, float]],
    max_amp: float,
) -> Dict[str, str]:
    status_up = str(status or "").upper()
    max_amp = float(max_amp or 0.0)

    if critical_speeds:
        dominant = critical_speeds[0]
        cs_speed = float(dominant.get("speed", 0.0) or 0.0)
        cs_amp = float(dominant.get("amp", 0.0) or 0.0)
        phase_delta = float(dominant.get("phase_delta", 0.0) or 0.0)

        if abs(phase_delta) >= 45.0:
            modal_txt = (
                "El cambio de fase es suficientemente representativo para sospechar transición modal marcada. "
                "Antes de la velocidad crítica el rotor tiende a comportarse con respuesta más rígida; después del paso por la zona modal, "
                "la respuesta se vuelve más flexible y la fase evidencia el cambio de relación entre fuerza excitadora y desplazamiento."
            )
        elif abs(phase_delta) >= 15.0:
            modal_txt = (
                "El cambio de fase es moderado y sugiere aproximación a una zona de amplificación dinámica. "
                "La respuesta aún no confirma por sí sola un paso crítico plenamente desarrollado, pero sí muestra una modificación de rigidez dinámica aparente."
            )
        else:
            modal_txt = (
                "El cambio de fase es bajo, por lo que el punto identificado debe tratarse como candidato dinámico y no como velocidad crítica confirmada. "
                "La confirmación requiere correlación con Bode, fase, órbita y repetibilidad entre corridas."
            )

        if status_up == "DANGER":
            headline = f"Respuesta polar severa asociada a posible velocidad crítica cerca de {cs_speed:.0f} rpm"
        elif status_up == "WARNING":
            headline = f"Respuesta polar con indicios de amplificación dinámica cerca de {cs_speed:.0f} rpm"
        else:
            headline = f"Respuesta polar controlada con candidato modal cerca de {cs_speed:.0f} rpm"

        detail = (
            f"La trayectoria polar identifica una zona de interés alrededor de {cs_speed:.0f} rpm, con amplitud aproximada de {cs_amp:.3f} "
            f"y variación de fase de {phase_delta:.1f}°. Esta condición es compatible con una posible aproximación a velocidad crítica o forma modal del sistema rotor-soporte.\\n\\n"
            f"{modal_txt}\\n\\n"
            f"Desde el punto de vista de análisis rotodinámico, la interpretación debe enfocarse en la relación entre amplitud, fase y velocidad. "
            f"Un incremento de amplitud acompañado por cambio de fase consistente puede indicar paso por una forma modal; si la amplitud aumenta sin cambio de fase suficiente, "
            f"la condición puede estar más asociada a desbalance, excentricidad, respuesta forzada o cambios operativos."
        )

        action = (
            "Correlacionar la trayectoria polar con Bode de amplitud y fase.\\n"
            "Verificar repetibilidad de la zona modal entre arranques/paradas.\\n"
            "Comparar contra órbitas filtradas 1X y shaft centerline.\\n"
            "Confirmar si el cambio de fase ocurre antes, durante o después del máximo de amplitud.\\n"
            "Validar condiciones de balance, alineación, rigidez de soporte, lubricación y carga."
        )
    else:
        headline = "Respuesta polar sin velocidad crítica dominante claramente identificada"
        detail = (
            f"La trayectoria polar presenta amplitud máxima de {max_amp:.3f}, sin un candidato modal dominante bajo la heurística aplicada. "
            f"No se observa una combinación suficientemente clara de incremento de amplitud y cambio de fase para confirmar velocidad crítica.\\n\\n"
            f"Esta condición puede representar una respuesta estable, una excitación forzada o una corrida donde el régimen crítico no fue cruzado de forma suficientemente clara."
        )
        action = (
            "Mantener la corrida como referencia histórica.\\n"
            "Comparar contra futuras trayectorias polares.\\n"
            "Correlacionar con Bode, espectro 1X, órbita y variables operativas."
        )

    return {"headline": headline, "detail": detail, "action": action}


def _prepare_polar_compare_df(item: Dict[str, Any], smooth_window: int, amp_smooth_window: int) -> pd.DataFrame:
    orient = get_panel_orientation(item["id"])
    df = item["grouped_df"].copy()

    df["amp"] = smooth_series(df["amp"], amp_smooth_window)
    df["phase_smoothed"] = circular_smooth_deg(df["phase"], smooth_window) % 360.0
    df["theta_display"] = compute_polar_display_theta(
        phase_deg=df["phase_smoothed"],
        axis_label=orient["axis_label"],
        side_label=orient["side_label"],
        install_angle_deg=float(orient["install_angle_deg"]),
        rotation_direction=orient["rotation_direction"],
    )
    df["phase_for_detection"] = np.rad2deg(np.unwrap(np.deg2rad(df["phase_smoothed"].to_numpy())))
    return df


def _polar_compare_metrics(
    item: Dict[str, Any],
    smooth_window: int,
    amp_smooth_window: int,
    max_critical_speeds: int,
    *,
    operating_rpm: float = 3600.0,
    machine_group: str = "group2",
    iso_part: str = "20816-2",
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    df = _prepare_polar_compare_df(item, smooth_window, amp_smooth_window)
    cs_legacy = estimate_critical_speeds_api684_style(df, max_count=max_critical_speeds)
    max_amp = float(df["amp"].max()) if len(df) else 0.0

    if cs_legacy:
        dom = cs_legacy[0]
        dom_speed = float(dom.get("speed", 0.0))
        dom_amp = float(dom.get("amp", 0.0))
        dom_phase = float(dom.get("phase_delta", 0.0))
    else:
        idx_legacy = int(df["amp"].idxmax()) if len(df) else 0
        dom_speed = float(df.loc[idx_legacy, "speed"]) if len(df) else 0.0
        dom_amp = max_amp
        dom_phase = 0.0

    ts_start = pd.to_datetime(df["ts_min"], errors="coerce").min() if "ts_min" in df.columns else None
    ts_end = pd.to_datetime(df["ts_max"], errors="coerce").max() if "ts_max" in df.columns else None

    # =========================================================
    # ROTORDYNAMICS — análisis Cat IV por corrida
    # =========================================================
    amp_unit = item.get("meta", {}).get("Amp Unit", "mil pp") or "mil pp"
    criticals_rotordyn = []
    primary_critical = None
    primary_api684 = None
    api684_evals = []
    iso_eval = None
    peak_amp_um_pp = 0.0

    if len(df) >= 8:
        try:
            criticals_rotordyn = detect_critical_speeds(
                rpm=df["speed"].to_numpy(),
                amp=df["amp"].to_numpy(),
                phase=df["phase"].to_numpy(),
            )
        except Exception:
            criticals_rotordyn = []

        if criticals_rotordyn:
            primary_critical = criticals_rotordyn[0]
            for cs_obj in criticals_rotordyn:
                api_eval = evaluate_api684_margin(
                    critical_rpm=cs_obj.rpm,
                    operating_rpm=operating_rpm,
                    q_factor=cs_obj.q_factor,
                )
                api684_evals.append(api_eval)
            primary_api684 = api684_evals[0]

        peak_amp_csv = max_amp
        unit_lower = amp_unit.strip().lower()
        if "mil" in unit_lower:
            peak_amp_um_pp = mils_to_micrometers(peak_amp_csv)
        elif "µm" in unit_lower or "um" in unit_lower:
            peak_amp_um_pp = peak_amp_csv
        else:
            peak_amp_um_pp = peak_amp_csv

        try:
            mtype_polar_compare = (
                "casing_velocity" if iso_part in ("20816-4", "20816-7")
                else "shaft_displacement"
            )
            iso_eval = iso_20816_zone_multipart(
                amplitude=peak_amp_um_pp,
                iso_part=iso_part,
                machine_group=machine_group,
                measurement_type=mtype_polar_compare,
                operating_speed_rpm=operating_rpm,
                custom_thresholds=custom_thresholds,
            )
        except Exception:
            iso_eval = None

    return {
        "label": item["label"],
        "machine": item["machine"],
        "point": item["point"],
        "df": df,
        "critical_speeds": cs_legacy,
        "max_amp": max_amp,
        "dominant_speed": dom_speed,
        "dominant_amp": dom_amp,
        "dominant_phase_delta": dom_phase,
        "ts_start": ts_start,
        "ts_end": ts_end,
        # rotordynamics
        "amp_unit": amp_unit,
        "criticals_rotordyn": criticals_rotordyn,
        "primary_critical": primary_critical,
        "api684_evals": api684_evals,
        "primary_api684": primary_api684,
        "iso_eval": iso_eval,
        "peak_amp_csv": max_amp,
        "peak_amp_um_pp": peak_amp_um_pp,
    }


def _polar_compare_diagnostic(records: List[Dict[str, Any]]) -> Dict[str, str]:
    ordered = sorted(
        records,
        key=lambda r: pd.Timestamp(r["ts_start"]) if r["ts_start"] is not None else pd.Timestamp.min
    )
    baseline = ordered[0]
    latest = ordered[-1]

    delta_amp = float(latest["dominant_amp"] - baseline["dominant_amp"])
    delta_speed = float(latest["dominant_speed"] - baseline["dominant_speed"])
    delta_phase = float(latest["dominant_phase_delta"] - baseline["dominant_phase_delta"])

    amp_trend = "incremento" if delta_amp > 0.15 else "reducción" if delta_amp < -0.15 else "estabilidad"
    speed_shift = "desplazamiento hacia mayor velocidad" if delta_speed > 100 else "desplazamiento hacia menor velocidad" if delta_speed < -100 else "sin desplazamiento relevante de velocidad"

    headline = "Comparación multi-fecha de trayectoria polar y respuesta modal"

    detail = (
        f"Se compararon {len(ordered)} corridas polares. Entre la corrida base ({baseline['label']}) y la más reciente ({latest['label']}) "
        f"se observa {amp_trend} de la amplitud dominante ({delta_amp:+.3f}), {speed_shift} ({delta_speed:+.0f} rpm) "
        f"y variación de fase dominante de {delta_phase:+.1f}°.\\n\\n"
        f"Desde el punto de vista rotodinámico, la comparación polar permite evaluar si la respuesta del rotor mantiene el mismo patrón modal o si existe migración de la zona crítica. "
        f"Cuando el máximo de amplitud y el cambio de fase se desplazan entre corridas, puede existir modificación de rigidez efectiva, amortiguamiento, condición de soporte, balance o carga.\\n\\n"
        f"Antes de una velocidad crítica el rotor tiende a comportarse como un sistema más rígido; al cruzar una forma modal, la fase y la trayectoria cambian y el rotor manifiesta comportamiento flexible. "
        f"Por eso, la lectura conjunta de amplitud, fase y velocidad es más concluyente que la amplitud por sí sola."
    )

    action = (
        "Correlacionar las corridas polares con Bode de amplitud/fase.\\n"
        "Verificar si la velocidad candidata se repite o migra entre fechas.\\n"
        "Comparar contra órbitas 1X y shaft centerline.\\n"
        "Validar si hubo cambios de balance, alineación, lubricación, temperatura o carga.\\n"
        "Usar la corrida más estable como línea base de aceptación."
    )

    return {"headline": headline, "detail": detail, "action": action}


def _temporal_palette(n: int) -> List[str]:
    """
    Paleta de colores para visualización temporal (oldest → newest).
    Gradiente azul claro → naranja oscuro pasando por gris/verde.
    Colores seleccionados para máxima distinción y legibilidad.
    """
    if n <= 1:
        return ["#2563eb"]
    if n == 2:
        return ["#3b82f6", "#ea580c"]
    if n == 3:
        return ["#3b82f6", "#16a34a", "#ea580c"]
    if n == 4:
        return ["#3b82f6", "#16a34a", "#f59e0b", "#dc2626"]
    if n == 5:
        return ["#3b82f6", "#0891b2", "#16a34a", "#f59e0b", "#dc2626"]
    # Más de 5: usar gradiente generado
    base = ["#3b82f6", "#0891b2", "#16a34a", "#84cc16", "#f59e0b", "#ea580c", "#dc2626"]
    if n <= len(base):
        return base[:n]
    return base + ["#7c3aed"] * (n - len(base))


def render_polar_compare_section(
    items: List[Dict[str, Any]],
    *,
    smooth_window: int,
    amp_smooth_window: int,
    max_critical_speeds: int,
    logo_uri: Optional[str],
    use_rotordyn_pro: bool = True,
    operating_rpm: float = 3600.0,
    machine_group: str = "group2",
    speed_min_filter: Optional[float] = None,
    speed_max_filter: Optional[float] = None,
    iso_part: str = "20816-2",
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
) -> None:
    if len(items) < 2:
        return

    records = [
        _polar_compare_metrics(
            item,
            smooth_window,
            amp_smooth_window,
            max_critical_speeds,
            operating_rpm=operating_rpm,
            machine_group=machine_group,
            iso_part=iso_part,
            custom_thresholds=custom_thresholds,
        )
        for item in items
    ]

    # Aplicar Speed Range a cada record (filtra el df que se va a graficar
    # y a usar para rotordynamics)
    if speed_min_filter is not None and speed_max_filter is not None:
        lo = float(min(speed_min_filter, speed_max_filter))
        hi = float(max(speed_min_filter, speed_max_filter))
        for r in records:
            df_full = r["df"]
            df_filtered = df_full[(df_full["speed"] >= lo) & (df_full["speed"] <= hi)].copy()
            if not df_filtered.empty:
                r["df"] = df_filtered
                # Recalcular peak amp en el rango filtrado para que la zona ISO
                # del comparativo refleje SOLO los datos visibles
                new_peak_csv = float(df_filtered["amp"].max())
                r["peak_amp_csv"] = new_peak_csv
                amp_unit_lower = (r.get("amp_unit") or "").lower()
                if "mil" in amp_unit_lower:
                    r["peak_amp_um_pp"] = mils_to_micrometers(new_peak_csv)
                else:
                    r["peak_amp_um_pp"] = new_peak_csv
                # Re-detectar críticas dentro del rango
                try:
                    crits = detect_critical_speeds(
                        rpm=df_filtered["speed"].to_numpy(),
                        amp=df_filtered["amp"].to_numpy(),
                        phase=df_filtered["phase"].to_numpy(),
                    )
                    r["criticals_rotordyn"] = crits
                    r["primary_critical"] = crits[0] if crits else None
                    if r["primary_critical"] is not None:
                        r["primary_api684"] = evaluate_api684_margin(
                            critical_rpm=r["primary_critical"].rpm,
                            operating_rpm=operating_rpm,
                            q_factor=r["primary_critical"].q_factor,
                        )
                    else:
                        r["primary_api684"] = None
                    r["iso_eval"] = iso_20816_2_zone(
                        amplitude=r["peak_amp_um_pp"],
                        measurement_type="shaft_displacement",
                        machine_group=machine_group,
                        operating_speed_rpm=operating_rpm,
                    )
                except Exception:
                    pass

    # Ordenar cronológicamente para que la paleta refleje secuencia temporal
    records_chrono = sorted(
        records,
        key=lambda r: pd.Timestamp(r["ts_start"]) if r["ts_start"] is not None else pd.Timestamp.min,
    )

    st.markdown("---")
    st.markdown("## Comparación multi-fecha · Polar Plot")

    fig = go.Figure()
    palette = _temporal_palette(len(records_chrono))

    for idx, rec in enumerate(records_chrono):
        df = rec["df"]
        color = palette[idx]
        date_label = (
            pd.Timestamp(rec["ts_start"]).strftime("%d %b %Y")
            if rec["ts_start"] is not None
            else rec["label"]
        )

        # Etiqueta enriquecida con métricas Cat IV cuando hay rotordyn
        if use_rotordyn_pro and rec.get("primary_critical") is not None:
            cs_pro = rec["primary_critical"]
            zone = rec.get("iso_eval").zone if rec.get("iso_eval") else "—"
            q_str = f"{cs_pro.q_factor:.2f}" if np.isfinite(cs_pro.q_factor) else "—"
            trace_name = f"{date_label}  ·  Q={q_str}  ·  zona {zone}"
        else:
            trace_name = f"{date_label}  ·  {rec['label']}"

        fig.add_trace(
            go.Scatterpolar(
                r=df["amp"],
                theta=df["theta_display"],
                mode="lines",
                name=trace_name,
                line=dict(width=2.6, color=color),
                hovertemplate="Amp: %{r:.3f}<br>Phase: %{theta:.1f}°<extra></extra>",
            )
        )

        # Marker de la crítica detectada (rotordyn si está, legacy si no)
        cs_marker_speed = None
        cs_marker_label = ""
        if use_rotordyn_pro and rec.get("primary_critical") is not None:
            cs_marker_speed = float(rec["primary_critical"].rpm)
            cs_marker_label = f"{int(round(cs_marker_speed))} rpm"
        elif rec["critical_speeds"]:
            cs_legacy = rec["critical_speeds"][0]
            cs_marker_speed = float(cs_legacy.get("speed", 0.0))
            cs_marker_label = f"{int(round(cs_marker_speed))} rpm"

        if cs_marker_speed and cs_marker_speed > 0:
            row_marker = nearest_row_for_speed(df, cs_marker_speed)
            fig.add_trace(
                go.Scatterpolar(
                    r=[row_marker["amp"]],
                    theta=[row_marker["theta_display"]],
                    mode="markers+text",
                    marker=dict(size=12, color=color, symbol="diamond", line=dict(width=2, color="white")),
                    text=[cs_marker_label],
                    textposition="top center",
                    textfont=dict(size=11, color=color, family="Arial Black"),
                    showlegend=False,
                    hovertemplate=f"<b>Crítica detectada</b><br>{cs_marker_label}<br>Amp: %{{r:.3f}}<br>Phase: %{{theta:.1f}}°<extra></extra>",
                )
            )

        # Marker de velocidad operativa si está dentro del rango medido
        if (
            float(df["speed"].min()) <= operating_rpm <= float(df["speed"].max())
        ):
            row_op = nearest_row_for_speed(df, operating_rpm)
            fig.add_trace(
                go.Scatterpolar(
                    r=[row_op["amp"]],
                    theta=[row_op["theta_display"]],
                    mode="markers",
                    marker=dict(size=14, color=color, symbol="star", line=dict(width=2, color="white")),
                    showlegend=False,
                    hovertemplate=f"<b>Operación nominal</b><br>{operating_rpm:.0f} rpm<br>Amp: %{{r:.3f}}<br>Phase: %{{theta:.1f}}°<extra></extra>",
                )
            )

    max_r = max([float(rec["df"]["amp"].max()) for rec in records if len(rec["df"])], default=1.0) * 1.18

    base_orient = get_panel_orientation(items[0]["id"])
    axis_label = base_orient["axis_label"]
    side_label = base_orient["side_label"]
    install_angle_deg = float(base_orient["install_angle_deg"])
    rotation_direction = base_orient["rotation_direction"]

    axis_rotation, angular_direction, _ = get_polar_axis_rotation_and_direction(
        axis_label=axis_label,
        side_label=side_label,
        install_angle_deg=install_angle_deg,
        rotation_direction=rotation_direction,
    )

    all_speeds = []
    for rec in records:
        if len(rec["df"]):
            all_speeds.extend(rec["df"]["speed"].astype(float).tolist())

    rpm_text = "—"
    if all_speeds:
        rpm_text = f"{int(min(all_speeds))} - {int(max(all_speeds))} rpm"

    dt_values = [r["ts_start"] for r in records if r["ts_start"] is not None]
    dt_text = "Comparación multi-fecha"
    if dt_values:
        dt_text = " / ".join([pd.Timestamp(v).strftime("%Y-%m-%d") for v in dt_values[:4]])

    draw_top_strip(
        fig=fig,
        machine=items[0].get("machine", ""),
        point_text="Polar Plot · Comparación multi-fecha",
        variable=f"{axis_label} | {install_angle_deg:.0f}° {side_label} | Rotation {rotation_direction}",
        dt_text=dt_text,
        rpm_text=rpm_text,
        logo_uri=logo_uri,
    )

    build_probe_reference_overlay(fig, max_r)

    fig.update_layout(
        title=None,
        polar=dict(
            bgcolor="#f8fafc",
            domain=dict(x=[0.0, 0.86], y=[0.01, 0.86]),
            radialaxis=dict(
                range=[0, max_r],
                tickfont=dict(size=11, color="#111827"),
                gridcolor="rgba(148,163,184,0.18)",
                linecolor="#9ca3af",
                showline=True,
                ticks="outside",
                angle=225,
            ),
            angularaxis=dict(
                rotation=axis_rotation,
                direction=angular_direction,
                tickfont=dict(size=12, color="#111827"),
                gridcolor="rgba(148,163,184,0.18)",
                linecolor="#9ca3af",
                showline=True,
                ticks="outside",
            ),
        ),
        height=820,
        template="plotly_white",
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.50,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key="wm_polar_compare_plot")

    # =========================================================
    # Tabla comparativa rotodinámica Cat IV (en unidad de origen del CSV)
    # =========================================================
    if use_rotordyn_pro:
        amp_unit_common = records_chrono[0].get("amp_unit", "mil pp") if records_chrono else "mil pp"
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
        summary = pd.DataFrame([
            {
                "Archivo": r["label"],
                "Fecha inicio": pd.Timestamp(r["ts_start"]).strftime("%Y-%m-%d %H:%M") if r["ts_start"] is not None else "—",
                "Fecha fin": pd.Timestamp(r["ts_end"]).strftime("%Y-%m-%d %H:%M") if r["ts_end"] is not None else "—",
                "Amp dominante": round(r["dominant_amp"], 3),
                "RPM candidata": round(r["dominant_speed"], 0),
                "Delta fase": round(r["dominant_phase_delta"], 1),
                "Max amp": round(r["max_amp"], 3),
            }
            for r in records_chrono
        ])

    st.dataframe(summary, width="stretch", hide_index=True)

    # =========================================================
    # Diagnóstico comparativo: nuevo (Cat IV) o legacy
    # =========================================================
    if use_rotordyn_pro:
        diag = build_polar_compare_diagnostics_rotordyn(
            records=records_chrono,
            operating_rpm=operating_rpm,
            machine_group=machine_group,
        )
    else:
        diag = _polar_compare_diagnostic(records_chrono)

    st.markdown("### Diagnóstico comparativo automático")
    st.markdown(f"**{diag['headline']}**")
    st.write(diag["detail"])
    st.write(diag["action"])

    # Notas para reporte: prosa fluida + síntesis por corrida en oraciones
    if use_rotordyn_pro:
        # Resumen prosa: una oración natural por cada corrida cronológicamente
        amp_unit_common = records_chrono[0].get("amp_unit", "mil pp") if records_chrono else "mil pp"
        prose_lines = []
        for r in records_chrono:
            cs_pro = r.get("primary_critical")
            api_pro = r.get("primary_api684")
            iso_eval = r.get("iso_eval")
            date_str = (
                pd.Timestamp(r["ts_start"]).strftime("%d %b %Y")
                if r["ts_start"] is not None
                else r["label"]
            )

            if cs_pro is not None and iso_eval is not None and api_pro is not None:
                amp_str = (
                    f"{r.get('peak_amp_csv', 0.0):.3f} {amp_unit_common}"
                    if "mil" in amp_unit_common.lower()
                    else f"{r.get('peak_amp_csv', 0.0):.1f} {amp_unit_common}"
                )
                q_str = (
                    f"{cs_pro.q_factor:.2f}"
                    if np.isfinite(cs_pro.q_factor)
                    else "—"
                )
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
            f"Síntesis cronológica de las corridas analizadas:\n\n"
            f"{prose_summary}\n\n"
            f"{diag['action']}"
        )
    else:
        summary_lines = []
        for _, row in summary.iterrows():
            summary_lines.append(
                f"- {row['Archivo']}: candidato {row['RPM candidata']:.0f} rpm, "
                f"amplitud dominante {row['Amp dominante']:.3f}, "
                f"Δfase {row['Delta fase']:.1f}°, máximo {row['Max amp']:.3f}."
            )

        notes = (
            _build_polar_report_notes(diag)
            + "\n\nResumen comparativo de corridas:\n"
            + "\n".join(summary_lines)
        )

    png_bytes, png_error = build_export_png_bytes(fig, diag)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Enviar comparativo Polar a reporte", key="wm_polar_compare_report_btn"):
            ensure_report_state()
            append_report_item_and_persist(
                {
                    "type": "polar_compare",
                    "title": "Polar Plot · Comparación multi-fecha",
                    "notes": notes,
                    "image_bytes": png_bytes,
                }
            )
            st.success("Comparativo Polar enviado al reporte.")
    with c2:
        if png_bytes is not None:
            st.download_button(
                "Descargar PNG comparativo Polar",
                data=png_bytes,
                file_name="polar_compare_hd.png",
                mime="image/png",
                key="wm_polar_compare_download_btn",
                width="stretch",
            )
        elif png_error:
            st.warning(f"No fue posible generar PNG comparativo: {png_error}")

# ============================================================
# MULTI-FILE LOADER
# ============================================================
def uploaded_file_label(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Polar.csv")).name


def uploaded_file_stem(file_obj) -> str:
    return Path(getattr(file_obj, "name", "Polar.csv")).stem


def parse_uploaded_polar_files(files: List[Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    parsed_items: List[Dict[str, Any]] = []
    failed_items: List[Tuple[str, str]] = []

    for file_obj in files:
        try:
            meta, raw_df, grouped_df = read_polar_csv(file_obj)
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
# PER-PANEL ORIENTATION STATE
# ============================================================
def get_panel_orientation(item_id: str) -> Dict[str, Any]:
    key = f"wm_polar_orientation::{item_id}"
    if key not in st.session_state:
        st.session_state[key] = {
            "axis_label": "Y",
            "side_label": "Left",
            "install_angle_deg": 45,
            "rotation_direction": "CCW",
        }
    return st.session_state[key]


# ============================================================
# PANEL RENDER
# ============================================================
def render_polar_panel(
    item: Dict[str, Any],
    panel_index: int,
    *,
    logo_uri: Optional[str],
    smooth_window: int,
    amp_smooth_window: int,
    show_info_box: bool,
    show_rpm_labels: bool,
    marker_stride: int,
    detect_cs: bool,
    max_critical_speeds: int,
    use_rotordyn_pro: bool = True,
    operating_rpm: float = 3600.0,
    machine_group: str = "group2",
    speed_min_filter: Optional[float] = None,
    speed_max_filter: Optional[float] = None,
    iso_part: str = "20816-2",
    custom_thresholds: Optional[Tuple[float, float, float]] = None,
    profile_label: Optional[str] = None,
) -> None:
    meta = item["meta"]
    raw_df = item["raw_df"]
    grouped_df = item["grouped_df"]
    orient = get_panel_orientation(item["id"])

    axis_label = orient["axis_label"]
    side_label = orient["side_label"]
    install_angle_deg = float(orient["install_angle_deg"])
    rotation_direction = orient["rotation_direction"]

    plot_df = grouped_df.copy()

    # Aplicar filtro de Speed Range si fue configurado por sidebar
    if speed_min_filter is not None and speed_max_filter is not None:
        lo = float(min(speed_min_filter, speed_max_filter))
        hi = float(max(speed_min_filter, speed_max_filter))
        plot_df = plot_df[(plot_df["speed"] >= lo) & (plot_df["speed"] <= hi)].copy()
        if plot_df.empty:
            st.warning(
                f"Panel {panel_index + 1}: no hay puntos en el rango RPM "
                f"[{lo:.0f} – {hi:.0f}]. Ajusta el rango en la sidebar."
            )
            return

    plot_df["amp"] = smooth_series(plot_df["amp"], amp_smooth_window)
    plot_df["phase_smoothed"] = circular_smooth_deg(plot_df["phase"], smooth_window) % 360.0
    plot_df["theta_display"] = compute_polar_display_theta(
        phase_deg=plot_df["phase_smoothed"],
        axis_label=axis_label,
        side_label=side_label,
        install_angle_deg=install_angle_deg,
        rotation_direction=rotation_direction,
    )

    phase_internal = np.rad2deg(np.unwrap(np.deg2rad(plot_df["phase_smoothed"].to_numpy())))
    plot_df["phase_for_detection"] = phase_internal

    speed_min = int(plot_df["speed"].min())
    speed_max = int(plot_df["speed"].max())

    cursor_col1, cursor_col2 = st.columns(2)
    with cursor_col1:
        cursor_a_speed = st.slider(
            f"Cursor A (RPM) · Panel {panel_index + 1}",
            speed_min,
            speed_max,
            speed_min,
            key=f"polar_cursor_a_{panel_index}_{item['id']}",
        )
    with cursor_col2:
        cursor_b_speed = st.slider(
            f"Cursor B (RPM) · Panel {panel_index + 1}",
            speed_min,
            speed_max,
            speed_max,
            key=f"polar_cursor_b_{panel_index}_{item['id']}",
        )

    row_a = nearest_row_for_speed(plot_df, cursor_a_speed)
    row_b = nearest_row_for_speed(plot_df, cursor_b_speed)

    critical_speeds: List[Dict[str, float]] = []
    if detect_cs:
        critical_speeds = estimate_critical_speeds_api684_style(plot_df, max_count=max_critical_speeds)

    semaforo_status, semaforo_color, polar_diag = polar_health_status(
        critical_speeds=critical_speeds,
        amp_series=plot_df["amp"],
    )

    machine = meta.get("Machine Name", "-")
    point = meta.get("Point Name", "-")
    variable = meta.get("Variable", "-")
    speed_unit = meta.get("Speed Unit", "rpm")
    amp_unit = meta.get("Amp Unit", "")

    if use_rotordyn_pro:
        mtype_polar = (
            "casing_velocity" if iso_part in ("20816-4", "20816-7")
            else "shaft_displacement"
        )
        text_diag = build_polar_diagnostics_rotordyn(
            rpm=plot_df["speed"].to_numpy(),
            amp=plot_df["amp"].to_numpy(),
            phase=plot_df["phase"].to_numpy(),
            operating_rpm=operating_rpm,
            machine_group=machine_group,
            amp_unit=amp_unit or "mil pp",
            measurement_type=mtype_polar,
            iso_part=iso_part,
            custom_thresholds=custom_thresholds,
            profile_label=profile_label,
        )
    else:
        # Modo legacy (preservado para compatibilidad)
        text_diag = build_polar_text_diagnostics(
            status=semaforo_status,
            critical_speeds=critical_speeds,
            max_amp=polar_diag["max_amp"],
        )

    panel_card(
        title=f"Polar {panel_index + 1} · {machine} · {point}",
        subtitle="Dynamic polar view",
        meta_html=(
            f"Variable: <b>{variable}</b> &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Orientation: <b>{axis_label} | {install_angle_deg:.0f}° {side_label}</b> &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Rotation: <b>{rotation_direction}</b> &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Speed Range: <b>{int(plot_df['speed'].min())} - {int(plot_df['speed'].max())} {speed_unit}</b>"
        ),
        chips=[
            f"File: {item['file_name']}",
            f"Raw rows: {len(raw_df):,}",
            f"Grouped points: {len(plot_df):,}",
            f"Phase smoothing: {smooth_window}",
            f"Amplitude smoothing: {amp_smooth_window}",
            f"Critical speeds: {len(critical_speeds)}",
        ],
    )

    # Datos para overlay visual Cat IV: críticas PRO + umbrales ISO
    pro_overlay_criticals: List[Dict[str, Any]] = []
    iso_thresholds_overlay: Optional[Dict[str, float]] = None
    if use_rotordyn_pro:
        try:
            crits_pro_polar = detect_critical_speeds(
                rpm=plot_df["speed"].to_numpy(),
                amp=plot_df["amp"].to_numpy(),
                phase=plot_df["phase"].to_numpy(),
            )
            pro_overlay_criticals = [
                {"rpm": cs.rpm, "q_factor": cs.q_factor}
                for cs in crits_pro_polar
            ]
            unit_lower = (amp_unit or "").lower()
            if "mil" in unit_lower:
                amp_for_iso = mils_to_micrometers(float(plot_df["amp"].max()))
            else:
                amp_for_iso = float(plot_df["amp"].max())
            mtype_overlay_polar = (
                "casing_velocity" if iso_part in ("20816-4", "20816-7")
                else "shaft_displacement"
            )
            iso_eval_overlay = iso_20816_zone_multipart(
                amplitude=amp_for_iso,
                iso_part=iso_part,
                machine_group=machine_group,
                measurement_type=mtype_overlay_polar,
                operating_speed_rpm=operating_rpm,
                custom_thresholds=custom_thresholds,
            )
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

    # Ciclo 17.1.1 — buscar TODOS los snapshots elegidos en sidebar
    # y extraer la lectura del sensor actual para overlay multi-snap.
    _prev_amp = None
    _prev_phase = None
    _prev_label = None
    _prev_op_speed = None
    _prev_snapshots_list: List[Dict[str, Any]] = []
    try:
        _polar_cmp_snap_ids = (
            st.session_state.get("wm_polar_compare_snapshot_ids") or []
        )
        # Backward-compat: si no hay multi pero hay single, usar single
        if not _polar_cmp_snap_ids:
            _single = st.session_state.get("wm_polar_compare_snapshot_id")
            if _single and _single != "__none__":
                _polar_cmp_snap_ids = [_single]
        if _polar_cmp_snap_ids:
            from core.polar_history import load_polar_snapshot
            from core.sensor_map import resolve_sensor_for_point as _sm_resolve, sensor_label as _sm_slbl
            _polar_inst_id_local = st.session_state.get("wm_active_instance_id", "")
            if not _polar_inst_id_local:
                _polar_inst_id_local = st.session_state.get("wm_polar_compare_inst_id", "")
            if _polar_inst_id_local:
                # Identificar a qué sensor corresponde ESTE panel UNA vez
                from core.instance_state import get_instance as _sm_get_inst
                _inst_obj = _sm_get_inst(_polar_inst_id_local)
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
                    for _snap_id in _polar_cmp_snap_ids:
                        _prev_snap_full = load_polar_snapshot(
                            _polar_inst_id_local, _snap_id,
                        )
                        if _prev_snap_full is None:
                            continue
                        for _ps in _prev_snap_full.get("sensors", []):
                            if str(_ps.get("sensor_label", "")) == _curr_panel_lbl:
                                _prev_snapshots_list.append({
                                    "amp": float(_ps.get("amp_at_op", 0) or 0),
                                    "phase": float(_ps.get("phase_at_op", 0) or 0),
                                    "label": _prev_snap_full.get("corrida_label", ""),
                                    "op_speed": float(_prev_snap_full.get("operating_speed_rpm", 0) or 0),
                                    "timestamp": _prev_snap_full.get("timestamp", ""),
                                    # Ciclo 17.1.2 — trayectoria completa para overlay
                                    "trajectory_speed": _ps.get("trajectory_speed", []) or [],
                                    "trajectory_amp": _ps.get("trajectory_amp", []) or [],
                                    "trajectory_phase": _ps.get("trajectory_phase", []) or [],
                                })
                                break
                # Para backward-compat con narrativa PDF: usar el primero
                # como _prev_amp/_prev_phase/_prev_label (que va al
                # comparison_narrative).
                if _prev_snapshots_list:
                    _first = _prev_snapshots_list[0]
                    _prev_amp = _first["amp"]
                    _prev_phase = _first["phase"]
                    _prev_label = _first["label"]
                    _prev_op_speed = _first["op_speed"]
    except Exception:
        _prev_amp = None
        _prev_phase = None
        _prev_snapshots_list = []

    # Ciclo 17.1.3 — Narrativa modal completa del comparativo Polar
    # para el PDF report. Reemplaza la narrativa simple de balance por
    # un análisis rotodinámico estilo Bently Nevada Technical Training:
    # caracterización del modo (translacional / cónico / flexural por
    # Δfase a la crítica), análisis de sensitividad vectorial,
    # distinción modal del rotor vs estructural, persistencia /
    # migración del modo, y diagnóstico diferencial (balance vs fault
    # vs operacional).
    if (
        _prev_amp is not None and _prev_phase is not None
        and _prev_amp > 0 and use_rotordyn_pro and operating_rpm is not None
    ):
        try:
            from core.polar_history import (
                phase_shift_classifier,
                amplitude_change_classifier,
                shortest_arc_phase_diff,
            )
            sp_min_chk = float(plot_df["speed"].min()) if len(plot_df) else 0.0
            sp_max_chk = float(plot_df["speed"].max()) if len(plot_df) else 0.0
            if sp_min_chk <= operating_rpm <= sp_max_chk:
                _curr_op_row = nearest_row_for_speed(plot_df, operating_rpm)
                _curr_amp_op = float(_curr_op_row["amp"])
                _curr_phase_op = float(_curr_op_row["theta_display"])
                _delta_amp = _curr_amp_op - float(_prev_amp)
                _delta_amp_pct = (
                    (_delta_amp / float(_prev_amp) * 100.0)
                    if float(_prev_amp) > 0 else None
                )
                _delta_phase = shortest_arc_phase_diff(
                    float(_prev_phase), _curr_phase_op,
                )
                _phase_class = phase_shift_classifier(_delta_phase)
                _amp_unit_local = meta.get("Amp Unit", "") or meta.get("Y-Axis Unit", "") or ""
                _prev_lbl_text = _prev_label or "corrida anterior"

                # --- Datos del modo en la corrida ACTUAL ---
                # Phase delta a través del peak (via critical_speeds_pro
                # si está disponible).
                _curr_cs_rpm = None
                _curr_cs_amp = None
                _curr_cs_phase_delta = None
                _curr_q = None
                if pro_overlay_criticals:
                    try:
                        _cs_pro_first = pro_overlay_criticals[0]
                        _curr_cs_rpm = float(_cs_pro_first.get("rpm", 0) or 0)
                        _curr_cs_amp = float(_cs_pro_first.get("amp", 0) or 0)
                        _curr_cs_phase_delta = float(
                            _cs_pro_first.get("phase_delta", 0) or 0
                        )
                        _curr_q = float(_cs_pro_first.get("q_factor", 0) or 0)
                    except Exception:
                        pass

                # Caracterizar el modo por el Δfase a través del peak:
                # ~180° = primer modo translacional/cylindrical (clásico
                #         para balance shift)
                # ~90°  = modo conical/pivotal o segundo modo
                # >270° = posible flexural / segundo modo bending
                # <90°  = anti-resonancia o resonancia estructural
                _mode_type = "modo no caracterizado"
                if _curr_cs_phase_delta is not None and _curr_cs_phase_delta != 0:
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
                            "típica cuando el rotor pivota alrededor "
                            "de un punto nodal cercano al cojinete"
                        )
                    elif 210.0 < _abs_pd <= 360.0:
                        _mode_type = (
                            "segundo modo flexural o respuesta acoplada "
                            "rotor-estructura (Δφ > 210°)"
                        )
                    else:
                        _mode_type = (
                            "respuesta de baja deflexión modal "
                            "(Δφ < 70°) — posible resonancia "
                            "estructural del soporte / fundación más "
                            "que modo del rotor"
                        )

                _narr_parts: List[str] = []

                # 1) Encabezado factual
                _narr_parts.append(
                    f"Análisis comparativo rotodinámico contra "
                    f"«{_prev_lbl_text}»"
                    + (
                        f" del {_prev_label[:10]}"
                        if _prev_label and len(_prev_label) >= 10 else ""
                    )
                    + ". A la velocidad operativa "
                    f"({operating_rpm:.0f} rpm), la respuesta sincrónica "
                    f"1X del sensor evolucionó de "
                    f"{float(_prev_amp):.3f} {_amp_unit_local} @ "
                    f"{float(_prev_phase):.1f}° a "
                    f"{_curr_amp_op:.3f} {_amp_unit_local} @ "
                    f"{_curr_phase_op:.1f}°, lo que representa un "
                    + (
                        f"vector change de {_delta_amp:+.3f} {_amp_unit_local} "
                        f"({_delta_amp_pct:+.1f}%)"
                        if _delta_amp_pct is not None
                        else f"vector change de {_delta_amp:+.3f} {_amp_unit_local}"
                    )
                    + f" en magnitud y un shift de fase 1X de "
                    f"{_delta_phase:+.1f}° en arco menor."
                )

                # 2) Caracterización del modo y forma estructural
                if _curr_cs_rpm and _curr_cs_rpm > 0:
                    _ratio_op_cs = operating_rpm / _curr_cs_rpm
                    _separation_pct = (_ratio_op_cs - 1.0) * 100.0
                    _q_str = (
                        f", con factor de amplificación Q={_curr_q:.2f}"
                        if _curr_q else ""
                    )
                    _mode_para = (
                        f"La trayectoria polar revela una velocidad "
                        f"crítica en {_curr_cs_rpm:.0f} rpm con un "
                        f"cambio de fase de "
                        f"{abs(_curr_cs_phase_delta or 0):.0f}° a través "
                        f"del peak{_q_str}. Este patrón se interpreta "
                        f"como {_mode_type}. "
                    )
                    if _separation_pct >= 15.0:
                        _mode_para += (
                            f"La velocidad operativa queda "
                            f"{_separation_pct:+.1f}% por encima del "
                            f"modo, separación amplia que cumple el "
                            f"requisito de margen de API 684 §6. "
                        )
                    elif _separation_pct >= 0:
                        _mode_para += (
                            f"La velocidad operativa queda solo "
                            f"{_separation_pct:+.1f}% por encima del "
                            f"modo. Es estrecho contra el margen "
                            f"recomendado de API 684 §6 (≥15%) y "
                            f"merece evaluación detallada de "
                            f"separation margin si el Q se incrementa. "
                        )
                    else:
                        _mode_para += (
                            f"La velocidad operativa está "
                            f"{abs(_separation_pct):.1f}% por debajo "
                            f"del modo identificado, configuración de "
                            f"sub-crítica que se considera estable "
                            f"siempre que el Q se mantenga acotado. "
                        )
                    _narr_parts.append(_mode_para)

                # 3) Diagnóstico diferencial del shift
                if _phase_class == "shift_critical":
                    _narr_parts.append(
                        "El shift de fase 1X supera 60° en arco menor, "
                        "magnitud considerada crítica según los criterios "
                        "de Bently / API 684. Magnitudes de esta escala "
                        "son inconsistentes con simple deriva térmica o "
                        "operacional y apuntan a un cambio mecánico "
                        "estructural del rotor: pérdida de masa por "
                        "desprendimiento de pieza, propagación de "
                        "fisura / crack, asentamiento súbito del cojinete "
                        "o pérdida del contacto con el sello / impeller. "
                        "Se recomienda parada controlada para inspección "
                        "y análisis complementario de orbits filtrados a "
                        "1X y forma de onda en ambos planos del cojinete "
                        "antes de continuar operación."
                    )
                elif _phase_class == "shift_major":
                    _narr_parts.append(
                        "El shift de fase 1X entre 30° y 60° es la firma "
                        "clásica de un cambio de balance del rotor según "
                        "la metodología de vector polar response que "
                        "documentan Bently y API 684. La magnitud y "
                        "dirección del vector change son consistentes "
                        "con una redistribución de masa rotativa "
                        "(suciedad acumulada o desprendida en álabes, "
                        "pérdida progresiva de balance weights, "
                        "deformación térmica residual). Se recomienda "
                        "programar balance de campo según ISO 21940-12 "
                        "nivel G 2.5 en próxima ventana, verificando "
                        "previamente la consistencia de fase entre "
                        "arranques sucesivos para descartar componente "
                        "transitoria."
                    )
                elif _phase_class == "shift_minor":
                    _narr_parts.append(
                        "El shift de fase 1X entre 10° y 30° es menor y "
                        "puede atribuirse a deriva operacional normal "
                        "(temperatura, carga, expansión térmica del "
                        "rotor o de los soportes). No constituye por sí "
                        "solo evidencia de cambio mecánico, pero merece "
                        "vigilancia: una tendencia consolidada a lo "
                        "largo de varias corridas en la misma dirección "
                        "vectorial sí indicaría cambio incipiente de "
                        "balance que conviene caracterizar antes de que "
                        "cruce la zona mayor."
                    )
                elif _phase_class == "stable":
                    _narr_parts.append(
                        "El shift de fase 1X (<10°) está dentro de la "
                        "variación normal de la respuesta sincrónica y "
                        "no constituye evidencia de cambio mecánico ni "
                        "de balance. La forma del vector se considera "
                        "estable entre corridas."
                    )

                # 4) Análisis del cambio de amplitud (sensitividad)
                if _delta_amp_pct is not None:
                    _amp_class = amplitude_change_classifier(_delta_amp_pct)
                    if _amp_class == "amp_critical":
                        _narr_parts.append(
                            "El crecimiento de amplitud 1X supera el "
                            "50% entre corridas. Combinado con el shift "
                            "de fase descrito, refuerza el diagnóstico "
                            "de degradación activa de la respuesta "
                            "modal — la sensibilidad del rotor a la "
                            "fuerza de excitación residual está "
                            "creciendo, lo que es típico de pérdida de "
                            "amortiguamiento (damping degradation) en "
                            "los soportes hidrodinámicos según el "
                            "marco de análisis de API 684."
                        )
                    elif _amp_class == "amp_high":
                        _narr_parts.append(
                            "El crecimiento de amplitud 1X (≥20%) "
                            "acompañando al shift de fase es consistente "
                            "con un cambio activo en la respuesta modal "
                            "del sistema rotor-soporte. Vale revisar "
                            "el factor de amplificación Q en próximas "
                            "corridas para descartar pérdida progresiva "
                            "de damping."
                        )
                    elif _amp_class in ("amp_down_strong", "amp_down"):
                        _narr_parts.append(
                            "La amplitud 1X bajó de manera significativa "
                            "respecto a la corrida anterior. Si esto "
                            "coincide con un shift de fase mayor, puede "
                            "tratarse de un cambio de balance "
                            "compensatorio (ej. una intervención previa, "
                            "redistribución térmica) más que de "
                            "degradación. Conviene revisar el "
                            "registro operacional y los reportes de "
                            "mantenimiento entre corridas para "
                            "confirmar."
                        )

                # 5) Distinción modal del rotor vs estructural
                if (
                    _phase_class in ("shift_major", "shift_critical")
                    and _curr_cs_phase_delta is not None
                    and abs(_curr_cs_phase_delta) < 90.0
                ):
                    _narr_parts.append(
                        "Nota diferencial: el cambio de fase a través "
                        "del peak observado (<90°) es atípico de un "
                        "modo libre del rotor y sugiere que el peak "
                        "podría corresponder a una resonancia "
                        "estructural del soporte o fundación más que a "
                        "un modo del rotor. Es importante validar "
                        "antes de atribuir el cambio observado a "
                        "balance del rotor — un cambio mecánico en la "
                        "fundación (suelta de anclajes, deterioro de "
                        "grouting) produce el mismo patrón en el "
                        "polar pero requiere intervención estructural, "
                        "no balance."
                    )

                _comp_narr = " ".join(_narr_parts)
                if isinstance(text_diag, dict):
                    text_diag["comparison_narrative"] = _comp_narr
        except Exception:
            pass

    fig = build_polar_figure(
        df=plot_df,
        meta=meta,
        row_a=row_a,
        row_b=row_b,
        logo_uri=logo_uri,
        show_info_box=show_info_box,
        show_rpm_labels=show_rpm_labels,
        marker_stride=marker_stride,
        axis_label=axis_label,
        side_label=side_label,
        install_angle_deg=install_angle_deg,
        rotation_direction=rotation_direction,
        critical_speeds=critical_speeds,
        semaforo_status=semaforo_status,
        semaforo_color=semaforo_color,
        operating_rpm=operating_rpm if use_rotordyn_pro else None,
        iso_thresholds=iso_thresholds_overlay,
        critical_speeds_pro=pro_overlay_criticals if pro_overlay_criticals else None,
        prev_snapshot_amp=_prev_amp,
        prev_snapshot_phase=_prev_phase,
        prev_snapshot_label=_prev_label,
        prev_snapshot_op_speed=_prev_op_speed,
        prev_snapshots=_prev_snapshots_list if _prev_snapshots_list else None,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False},
        key=f"wm_polar_plot_{panel_index}_{item['id']}",
    )

    helper_card(
        title=f"API 684 Helper · Polar {panel_index + 1}",
        subtitle=text_diag["headline"],
        chips=[
            (f"Semáforo: {semaforo_status}", semaforo_color),
            (f"Health score: {polar_diag['score']:.1f}", None),
            (f"Max amplitude: {polar_diag['max_amp']:.3f} {amp_unit}", None),
            (f"Critical candidates: {polar_diag['candidate_count']}", None),
            (f"Cursor A: {row_a['amp']:.3f} {amp_unit}", None),
            (f"Cursor B: {row_b['amp']:.3f} {amp_unit}", None),
        ],
    )

    st.info(
        f"**Diagnostic detail:** {text_diag['detail']}\n\n"
        f"**Recommended action:** {text_diag['action']}"
    )

    title = f"Polar {panel_index + 1} — {machine} — {point}"

    export_state_key = (
        f"polar::{item['id']}::{panel_index}::{variable}::{smooth_window}::{amp_smooth_window}::"
        f"{show_info_box}::{show_rpm_labels}::{marker_stride}::{axis_label}::{side_label}::"
        f"{install_angle_deg}::{rotation_direction}::{detect_cs}::{max_critical_speeds}::"
        f"{cursor_a_speed}::{cursor_b_speed}"
    )

    # ------------------------------------------------------------
    # Ciclo 17.26 — Interpretación clínica AI (Polar)
    # ------------------------------------------------------------
    ai_state_key_pol = f"wm_ai_diag_polar_{export_state_key}"
    if ai_state_key_pol not in st.session_state:
        st.session_state[ai_state_key_pol] = None

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
            stored_pol = st.session_state.get(ai_state_key_pol)
            ai_btn_col1, ai_btn_col2, ai_btn_col3 = st.columns([1.4, 1.4, 2.4])
            with ai_btn_col1:
                gen_clicked_pol = st.button(
                    "Generar diagnóstico AI"
                    if stored_pol is None
                    else "Diagnóstico generado",
                    key=f"ai_gen_btn_pol_{export_state_key}",
                    use_container_width=True,
                    type="primary" if stored_pol is None else "secondary",
                    disabled=stored_pol is not None and stored_pol.get("ok", False),
                )
            with ai_btn_col2:
                regen_clicked_pol = st.button(
                    "Regenerar",
                    key=f"ai_regen_btn_pol_{export_state_key}",
                    use_container_width=True,
                    disabled=stored_pol is None,
                )
            with ai_btn_col3:
                st.caption(
                    "Claude Sonnet 4.5 · ~$0.015 por diagnóstico · "
                    "cacheado 30 días si no regenerás."
                )

            should_call_pol = bool(gen_clicked_pol) and (stored_pol is None)
            should_regen_pol = bool(regen_clicked_pol) and (stored_pol is not None)

            if should_call_pol or should_regen_pol:
                # Payload Polar: amplitud máxima, velocidades críticas
                # detectadas, semáforo, score de salud, contexto del modo.
                _crit_speeds_payload: List[Dict[str, Any]] = []
                try:
                    for _cs in (pro_overlay_criticals or [])[:5]:
                        _crit_speeds_payload.append({
                            "rpm": float(_cs.get("rpm", 0) or 0),
                            "amp": float(_cs.get("amp", 0) or 0),
                            "q_factor": float(_cs.get("q_factor", 0) or 0),
                            "phase_delta": float(_cs.get("phase_delta", 0) or 0),
                        })
                except Exception:
                    pass

                ai_payload_pol: Dict[str, Any] = {
                    "machine": {
                        "tag": str(meta.get("Machine Name", "") or ""),
                        "punto_medicion": str(meta.get("Point Name", "") or ""),
                        "variable": str(meta.get("Variable", "") or ""),
                        "rpm_operativa": float(operating_rpm or 0.0),
                        "rotation_direction": str(rotation_direction),
                    },
                    "norm": {
                        "headline_tecnico": str(text_diag.get("headline", "") or ""),
                        "semaforo": str(semaforo_status or ""),
                        "amp_unit": str(amp_unit or ""),
                    },
                    "technical": {
                        "max_amplitude": round(
                            float(polar_diag.get("max_amp", 0.0) or 0.0), 4
                        ),
                        "health_score": round(
                            float(polar_diag.get("score", 0.0) or 0.0), 2
                        ),
                        "candidate_count": int(
                            polar_diag.get("candidate_count", 0) or 0
                        ),
                        "critical_speeds": _crit_speeds_payload,
                        "diagnostic_detail": str(text_diag.get("detail", "") or "")[:1500],
                        "diagnostic_action": str(text_diag.get("action", "") or "")[:1500],
                    },
                    "trend": {},
                }

                with st.spinner("Claude analizando la respuesta polar... (5-15 seg)"):
                    try:
                        result_pol = generate_ai_diagnostic(
                            ai_payload_pol,
                            module_type="polar",
                            use_cache=not should_regen_pol,
                        )
                    except Exception as exc:
                        result_pol = {
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
                st.session_state[ai_state_key_pol] = result_pol
                stored_pol = result_pol

            if stored_pol is not None:
                if stored_pol.get("ok"):
                    if stored_pol.get("fallback_used"):
                        st.info(
                            "Diagnóstico generado con modelo de respaldo "
                            "(Haiku 4.5)."
                        )
                    st.markdown(stored_pol.get("markdown", ""))
                    model_used_pol = str(stored_pol.get("model", "") or "")
                    if model_used_pol.startswith("claude-haiku"):
                        in_p_pol, out_p_pol = 1.0, 5.0
                    else:
                        in_p_pol, out_p_pol = 3.0, 15.0
                    cost_usd_pol = (
                        stored_pol.get("input_tokens", 0) * in_p_pol
                        + stored_pol.get("output_tokens", 0) * out_p_pol
                    ) / 1_000_000
                    fallback_tag_pol = (
                        " · modelo de respaldo"
                        if stored_pol.get("fallback_used") else ""
                    )
                    st.caption(
                        f"Modelo: `{model_used_pol}` · "
                        f"Tokens: {stored_pol.get('input_tokens', 0)} → "
                        f"{stored_pol.get('output_tokens', 0)} · "
                        f"Costo: ~${cost_usd_pol:.4f} · "
                        f"{'(cacheado)' if stored_pol.get('cached') else '(generado nuevo)'}"
                        f"{fallback_tag_pol}"
                    )
                else:
                    st.error(
                        stored_pol.get("markdown", "Error al generar diagnóstico AI.")
                    )

    # Helper para construir el bloque AI cuando se envía a reporte
    def _build_polar_ai_block_for_report() -> Optional[str]:
        ai_stored_local = st.session_state.get(ai_state_key_pol)
        if not (ai_stored_local
                and ai_stored_local.get("ok")
                and ai_stored_local.get("markdown")):
            return None
        ai_md_local = str(ai_stored_local.get("markdown", "")).strip()
        if not ai_md_local:
            return None
        quant_lines_pol: List[str] = ["Parámetro|Valor"]
        if operating_rpm:
            quant_lines_pol.append(
                f"Velocidad operativa|{float(operating_rpm):.0f} RPM"
            )
        _max_amp = float(polar_diag.get("max_amp", 0.0) or 0.0)
        if _max_amp > 0:
            quant_lines_pol.append(
                f"Amplitud máxima|{_max_amp:.3f} {amp_unit}".strip()
            )
        _score = float(polar_diag.get("score", 0.0) or 0.0)
        if _score > 0:
            quant_lines_pol.append(f"Health score|{_score:.1f}")
        _candidates = int(polar_diag.get("candidate_count", 0) or 0)
        quant_lines_pol.append(f"Velocidades críticas detectadas|{_candidates}")
        if semaforo_status:
            quant_lines_pol.append(f"Semáforo|{semaforo_status}")
        _pt = str(meta.get("Point Name", "") or "").strip()
        if _pt:
            quant_lines_pol.append(f"Punto de medición|{_pt}")
        return (
            "<<<WM_AI_BLOCK>>>\n"
            + "\n".join(quant_lines_pol)
            + "\n<<<WM_AI_NARRATIVE>>>\n"
            + ai_md_local
        )

    export_report_row(
        export_key=export_state_key,
        fig=fig,
        export_builder=lambda export_fig: build_export_png_bytes(export_fig, text_diag),
        report_callback=lambda: queue_polar_to_report(
            meta,
            fig,
            title,
            text_diag,
            image_bytes=build_export_png_bytes(fig, text_diag)[0],
            notes_override=_build_polar_ai_block_for_report(),
        ),
        file_name=f"{item['file_stem']}_polar_hd.png",
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    require_login()
    ensure_report_state()

    if "wm_polar_selected_ids" not in st.session_state:
        st.session_state.wm_polar_selected_ids = []

    page_header(
        title="Polar Plot",
        subtitle="Dynamic polar trajectory from amplitude, phase and speed.",
    )

    with st.sidebar:
        render_user_menu()
        st.markdown("---")
        st.markdown("### Upload Polar CSV")

        uploaded_files_new = st.file_uploader(
            "Upload one or more Polar CSV",
            type=["csv"],
            accept_multiple_files=True,
            key="wm_polar_file_uploader",
        )

        if uploaded_files_new:
            set_polar_persisted_files(uploaded_files_new)

        active_polar_files = get_polar_persisted_files()

        col1, col2 = st.columns(2)
        with col1:
            if active_polar_files:
                st.caption(f"Archivos Polar activos: {len(active_polar_files)}")
            else:
                st.caption("No hay archivos Polar cargados")

        with col2:
            if st.button("Limpiar archivos Polar", key="wm_polar_clear_files_btn"):
                clear_polar_persisted_files()
                st.rerun()

        uploaded_files = active_polar_files

    if not uploaded_files:
        panel_card(
            title="Carga archivos para comenzar",
            subtitle="Sube uno o varios archivos CSV Polar desde el panel izquierdo.",
            meta_html="",
            chips=[],
        )
        return

    parsed_items, failed_items = parse_uploaded_polar_files(uploaded_files)

    if failed_items:
        for file_name, error_text in failed_items:
            st.warning(f"No pude leer {file_name}: {error_text}")

    if not parsed_items:
        st.error("No se pudo cargar ningún archivo Polar válido.")
        return

    id_to_item = {item["id"]: item for item in parsed_items}
    label_to_id = {
        f"{item['machine']} · {item['point']} · {item['file_name']}": item["id"]
        for item in parsed_items
    }
    selection_labels = list(label_to_id.keys())

    valid_ids = set(id_to_item.keys())
    current_ids = [sid for sid in st.session_state.wm_polar_selected_ids if sid in valid_ids]
    if not current_ids:
        current_ids = [item["id"] for item in parsed_items]
        st.session_state.wm_polar_selected_ids = current_ids

    default_labels = [label for label, sid in label_to_id.items() if sid in current_ids]

    with st.sidebar:
        st.markdown("### Polar Selection")
        selected_labels = st.multiselect(
            "Polars to display",
            options=selection_labels,
            default=default_labels,
        )
        st.session_state.wm_polar_selected_ids = [label_to_id[label] for label in selected_labels if label in label_to_id]

        selected_ids_for_sidebar = [sid for sid in st.session_state.wm_polar_selected_ids if sid in id_to_item]

        if selected_ids_for_sidebar:
            st.markdown("### Probe Orientation by Polar")
            for panel_index, sid in enumerate(selected_ids_for_sidebar, start=1):
                item = id_to_item[sid]
                orient_key = f"wm_polar_orientation::{sid}"
                current = get_panel_orientation(sid)

                with st.expander(f"Polar {panel_index} · {item['point']}", expanded=(panel_index == 1)):
                    axis_value = st.selectbox(
                        "Probe Axis",
                        ["X", "Y"],
                        index=0 if current["axis_label"] == "X" else 1,
                        key=f"{orient_key}::axis",
                    )
                    side_value = st.selectbox(
                        "Probe Side",
                        ["Right", "Left"],
                        index=0 if current["side_label"] == "Right" else 1,
                        key=f"{orient_key}::side",
                    )
                    angle_value = st.slider(
                        "Probe Installation Angle",
                        0,
                        90,
                        int(current["install_angle_deg"]),
                        step=5,
                        key=f"{orient_key}::angle",
                    )
                    rotation_value = st.selectbox(
                        "Rotation Direction",
                        ["CCW", "CW"],
                        index=0 if current["rotation_direction"] == "CCW" else 1,
                        key=f"{orient_key}::rotation",
                    )

                    st.session_state[orient_key] = {
                        "axis_label": axis_value,
                        "side_label": side_value,
                        "install_angle_deg": angle_value,
                        "rotation_direction": rotation_value,
                    }

        # Speed Range Control — restringe el rango de RPM a graficar y analizar.
        # En modo Auto graficamos todos los datos; en modo manual el usuario fija
        # los límites con number_input. El filtro se aplica antes del rotordynamics
        # para que la crítica detectada esté siempre dentro del rango visible.
        sel_items = [id_to_item[sid] for sid in selected_ids_for_sidebar] if selected_ids_for_sidebar else parsed_items[:1]
        candidate_speed_frames = [it["grouped_df"] for it in sel_items if "grouped_df" in it]
        if candidate_speed_frames:
            sp_combined = pd.concat(candidate_speed_frames, ignore_index=True)
            speed_min_default = float(sp_combined["speed"].min())
            speed_max_default = float(sp_combined["speed"].max())
        else:
            speed_min_default, speed_max_default = 0.0, 3600.0

        st.markdown("### Speed Range Control")
        auto_speed = st.checkbox("Auto scale speed range", value=True, key="wm_polar_auto_speed")
        if auto_speed:
            speed_min_filter = speed_min_default
            speed_max_filter = speed_max_default
        else:
            speed_min_filter = st.number_input(
                "Min RPM",
                value=float(speed_min_default),
                step=10.0,
                key="wm_polar_speed_min",
            )
            speed_max_filter = st.number_input(
                "Max RPM",
                value=float(speed_max_default),
                step=10.0,
                key="wm_polar_speed_max",
            )

        st.markdown("### Polar Controls")
        smooth_window = st.slider("Circular phase smoothing", 1, 11, 3, step=2)
        amp_smooth_window = st.slider("Amplitude smoothing", 1, 11, 3, step=2)
        show_info_box = st.checkbox("Show Polar Information", value=True)
        show_rpm_labels = st.checkbox("Show RPM labels", value=True)
        marker_stride = st.slider("RPM label step", 10, 150, 45, step=5)

        st.markdown("### Critical Speed Detection")
        detect_cs = st.checkbox("Estimate critical speeds (API-684 heuristic)", value=True)
        max_critical_speeds = st.selectbox("Max critical speeds", [1, 2], index=1)

        # Asset Instance selector (Ciclo 8) — antes solo seleccionaba profile,
        # ahora selecciona la máquina física específica con sus propios datos.
        instance_state = render_instance_selector(module_name="polar")
        use_rotordyn_pro = instance_state["is_applicable"]
        operating_rpm = instance_state["operating_rpm"]
        machine_group = instance_state["machine_group"]
        active_iso_part = instance_state["iso_part"]
        active_custom_thresholds = instance_state["custom_thresholds"]
        active_profile_label = instance_state["profile_label"]

        # Ciclo 23.156 — quitado el cuadro de applicability (ruido). El flag
        # is_applicable se sigue usando arriba (use_rotordyn_pro).

    selected_ids = [sid for sid in st.session_state.wm_polar_selected_ids if sid in id_to_item]

    # ============================================================
    # Ciclo 17.1 — Histórico Polar (1X amp + phase trends)
    # ------------------------------------------------------------
    # Snapshotea por sensor matched el 1X amp + fase a la velocidad
    # operativa actual. Permite comparar contra corridas previas y
    # diagnosticar cambios de balance (shift de fase >30° = sintoma
    # API 684).
    # ============================================================
    _polar_inst_id = (
        instance_state.get("instance_id")
        or st.session_state.get("wm_active_instance_id", "")
    )
    _polar_inst = None
    _polar_sensors_map: List[Dict[str, Any]] = []
    if _polar_inst_id:
        try:
            from core.instance_state import get_instance as _polar_get_inst
            _polar_inst = _polar_get_inst(_polar_inst_id)
            if _polar_inst is not None:
                _polar_sensors_map = list(_polar_inst.sensors or [])
        except Exception:
            _polar_inst = None

    # Helper: para cada parsed_item (= cada CSV polar), encontrar el
    # sensor del Sensor Map matched y extraer amp/phase a op_speed +
    # trayectoria completa downsampleada para superposicion historica.
    def _wm_extract_polar_readings(
        items: List[Dict[str, Any]],
        sensors_map: List[Dict[str, Any]],
        op_speed_rpm: float,
    ) -> List[Dict[str, Any]]:
        from core.sensor_map import resolve_sensor_for_point as _wm_resolve, sensor_label as _wm_slbl
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
                # Si op_speed está fuera del rango cargado, usar el más cercano
                row = nearest_row_for_speed(df, op_speed_rpm)
                amp_at_op = float(row.get("amp", 0.0))
                phase_at_op = float(row.get("phase", 0.0)) % 360.0

                # Ciclo 17.1.2 — Trayectoria completa downsampleada.
                # Polar/Bode tipicos tienen 1000+ puntos. Para mantener
                # el JSON liviano (~10KB por sensor) bajamos a ~80 puntos
                # uniformemente espaciados por RPM. Esto preserva la
                # forma del loop incluyendo el paso por la critica.
                _df_sorted = df.sort_values("speed").reset_index(drop=True)
                _N_TARGET = 80
                if len(_df_sorted) > _N_TARGET:
                    _idx = np.linspace(0, len(_df_sorted) - 1, _N_TARGET).astype(int)
                    _df_ds = _df_sorted.iloc[_idx]
                else:
                    _df_ds = _df_sorted
                traj_speed = _df_ds["speed"].astype(float).tolist()
                traj_amp = _df_ds["amp"].astype(float).tolist()
                traj_phase = (_df_ds["phase"].astype(float) % 360.0).tolist()

                entry = {
                    "sensor_label": _wm_slbl(sensor_match),
                    "csv_file": it.get("file_name", ""),
                    "amp_at_op": amp_at_op,
                    "phase_at_op": phase_at_op,
                    "amp_unit": unit or "mil pp",
                    "phase_unit": "deg",
                    "csv_timestamp": str(meta.get("Timestamp", "") or ""),
                    "trajectory_speed": traj_speed,
                    "trajectory_amp": traj_amp,
                    "trajectory_phase": traj_phase,
                }
                out.append(entry)
            except Exception:
                continue
        return out

    _polar_curr_readings: List[Dict[str, Any]] = []
    if selected_ids and _polar_sensors_map:
        try:
            _selected_items_for_snap = [id_to_item[sid] for sid in selected_ids]
            _polar_curr_readings = _wm_extract_polar_readings(
                _selected_items_for_snap, _polar_sensors_map, float(operating_rpm),
            )
        except Exception:
            _polar_curr_readings = []

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📚 Histórico Polar")

        try:
            from core.polar_history import (
                save_polar_snapshot,
                list_polar_snapshots,
                load_polar_snapshot,
                delete_polar_snapshot,
                get_previous_polar_snapshot,
                phase_shift_classifier,
                amplitude_change_classifier,
                shortest_arc_phase_diff,
            )
            _polar_hist_ok = True
        except Exception as _hist_e:
            _polar_hist_ok = False
            st.caption(f"_(Histórico Polar no disponible: {_hist_e})_")

        if _polar_hist_ok and _polar_inst_id:
            # Ciclo 17.34 (v3.31.240) — sensor isolation. Si UN solo
            # sensor está cargado actualmente, filtramos solo sus
            # snapshots históricos (evita mezclar 1XA con 1YA).
            # Multi-sensor → sin filtro, como antes.
            _polar_filter_sensor = ""
            if len(_polar_curr_readings) == 1:
                _polar_filter_sensor = str(
                    _polar_curr_readings[0].get("sensor_label") or ""
                )
            _polar_existing_snaps = list_polar_snapshots(
                _polar_inst_id,
                sensor_id=_polar_filter_sensor,
            )
            _polar_filter_hint = (
                f" (filtrado por sensor **{_polar_filter_sensor}**)"
                if _polar_filter_sensor else ""
            )
            st.caption(
                f"{len(_polar_existing_snaps)} snapshot(s) Polar guardado(s) "
                f"para esta unidad{_polar_filter_hint}."
            )

            if not _polar_curr_readings:
                if not _polar_sensors_map:
                    st.caption(
                        "_(No hay Sensor Map configurado para esta instancia. "
                        "Andá a Machinery Library a configurarlo.)_"
                    )
                else:
                    st.caption(
                        "_(Ningún CSV Polar cargado matchea sensores del "
                        "Sensor Map de esta instancia.)_"
                    )
            else:
                with st.expander("📸 Guardar snapshot Polar actual", expanded=False):
                    st.caption(
                        f"Captura el 1X amp + fase a {operating_rpm:.0f} rpm "
                        f"para {len(_polar_curr_readings)} sensor(es) matched."
                    )
                    _polar_snap_label = st.text_input(
                        "Etiqueta de la corrida",
                        value="",
                        placeholder="Ej. Coastdown abril 27",
                        key=f"wm_polar_snap_label_{_polar_inst_id}",
                    )
                    _polar_snap_notes = st.text_area(
                        "Observaciones (opcional)",
                        value="",
                        placeholder="Velocidad operativa, condición, evento.",
                        key=f"wm_polar_snap_notes_{_polar_inst_id}",
                        height=70,
                    )
                    if st.button(
                        "Guardar snapshot Polar",
                        type="primary",
                        width="stretch",
                        key=f"wm_polar_snap_save_{_polar_inst_id}",
                    ):
                        try:
                            sid = save_polar_snapshot(
                                _polar_inst_id,
                                operating_speed_rpm=float(operating_rpm),
                                sensors_data=_polar_curr_readings,
                                corrida_label=_polar_snap_label,
                                notes=_polar_snap_notes,
                            )
                            st.success(
                                f"✓ Snapshot Polar guardado: {sid} "
                                f"({len(_polar_curr_readings)} sensores)"
                            )
                            st.rerun()
                        except Exception as _e:
                            st.error(f"No se pudo guardar: {_e}")

            # Selector de comparación — Ciclo 17.1.1 multi-select
            # Permite 0 snapshots (solo corrida actual), 1 (vs una corrida
            # especifica) o N (todas con gradiente cronologico).
            _selected_polar_cmp_ids: List[str] = []
            if _polar_existing_snaps:
                _curr_by_lbl = {
                    r["sensor_label"]: {"amp": r["amp_at_op"], "phase": r["phase_at_op"]}
                    for r in _polar_curr_readings
                }
                # Marcar snapshots identicos a la corrida actual y armar
                # opciones del multiselect (label + key)
                _polar_opt_pairs: List[Tuple[str, str]] = []
                _polar_default_pick: List[str] = []
                _polar_first_non_current = None
                for _i, s in enumerate(_polar_existing_snaps):
                    _is_current = False
                    if _curr_by_lbl:
                        try:
                            _snap_full = load_polar_snapshot(_polar_inst_id, s["snapshot_id"])
                            if _snap_full is not None:
                                from core.polar_history import _polar_snapshot_is_identical_to
                                _is_current = _polar_snapshot_is_identical_to(
                                    _snap_full, _curr_by_lbl,
                                )
                        except Exception:
                            pass
                    _suffix = " · (corrida actual)" if _is_current else ""
                    _opspeed = s.get("operating_speed_rpm")
                    _opspeed_str = f" @ {_opspeed:.0f}rpm" if _opspeed else ""
                    _lbl = (f"{s['corrida_label'][:28]}{_opspeed_str} "
                            f"({s['timestamp'][:10]}){_suffix}")
                    _polar_opt_pairs.append((s["snapshot_id"], _lbl))
                    if not _is_current and _polar_first_non_current is None:
                        _polar_first_non_current = _lbl

                _polar_opt_lbls = [l for _, l in _polar_opt_pairs]
                _polar_lbl_to_key = {l: k for k, l in _polar_opt_pairs}
                # Default: el primer snapshot no-actual (uno preseleccionado).
                # El usuario puede agregar mas o quitarlo.
                if _polar_first_non_current:
                    _polar_default_pick = [_polar_first_non_current]
                _polar_cmp_state_key = f"wm_polar_cmp_picks_{_polar_inst_id}"
                if _polar_cmp_state_key in st.session_state:
                    # Filtrar picks que ya no existen
                    _saved = st.session_state[_polar_cmp_state_key]
                    _polar_default_pick = [
                        l for l in _saved if l in _polar_opt_lbls
                    ]
                _picked = st.multiselect(
                    "Corridas a superponer en el polar",
                    options=_polar_opt_lbls,
                    default=_polar_default_pick,
                    key=f"wm_polar_cmp_multi_{_polar_inst_id}",
                    help=(
                        "Elegí 0 corridas para ver solo la actual, 1 para "
                        "comparativo simple, o varias para superposición "
                        "histórica con gradiente cronológico (más viejas "
                        "más claritas, más recientes más oscuras)."
                    ),
                )
                st.session_state[_polar_cmp_state_key] = _picked
                _selected_polar_cmp_ids = [
                    _polar_lbl_to_key[l] for l in _picked if l in _polar_lbl_to_key
                ]
                if not _selected_polar_cmp_ids:
                    st.caption("_Solo se mostrará la corrida actual._")
                else:
                    st.caption(
                        f"Se superpondrán **{len(_selected_polar_cmp_ids)}** "
                        f"corrida(s) anterior(es) sobre la actual."
                    )

            # Lista de snapshots con borrar + indicador de trayectoria
            if _polar_existing_snaps:
                # Pre-cargar para detectar cuales tienen trayectoria completa
                _legacy_count = 0
                _snap_has_trail: Dict[str, bool] = {}
                for s in _polar_existing_snaps:
                    try:
                        _full = load_polar_snapshot(_polar_inst_id, s["snapshot_id"])
                        if _full is not None:
                            _has_traj = any(
                                len(sens.get("trajectory_speed", []) or []) > 1
                                for sens in _full.get("sensors", [])
                            )
                            _snap_has_trail[s["snapshot_id"]] = _has_traj
                            if not _has_traj:
                                _legacy_count += 1
                    except Exception:
                        _snap_has_trail[s["snapshot_id"]] = False

                if _legacy_count > 0:
                    st.warning(
                        f"{_legacy_count} snapshot(s) viejos sin "
                        f"trayectoria completa — solo muestran el operating "
                        f"point en el polar. Para ver el loop completo, "
                        f"resnapshoteá cargando esa corrida y volviendo a "
                        f"guardar."
                    )

                with st.expander(f"️ Gestionar snapshots Polar ({len(_polar_existing_snaps)})"):
                    if _legacy_count > 0:
                        if st.button(
                            f"🧹 Borrar los {_legacy_count} snapshot(s) sin trayectoria",
                            key=f"wm_polar_del_legacy_{_polar_inst_id}",
                            help=(
                                "Borra todos los snapshots que se guardaron "
                                "antes del Ciclo 17.1.2 (sin trail completo). "
                                "Los snapshots actuales con trayectoria NO se "
                                "tocan."
                            ),
                        ):
                            _deleted = 0
                            for s in _polar_existing_snaps:
                                if not _snap_has_trail.get(s["snapshot_id"], False):
                                    if delete_polar_snapshot(_polar_inst_id, s["snapshot_id"]):
                                        _deleted += 1
                            st.success(
                                f"✓ {_deleted} snapshot(s) viejos borrados. "
                                f"Cargá las corridas y guardá snapshots "
                                f"nuevos para reconstruir el histórico con "
                                f"trayectoria."
                            )
                            st.rerun()
                        st.markdown("---")

                    for s in _polar_existing_snaps:
                        cols_h = st.columns([4, 1])
                        _has_traj = _snap_has_trail.get(s["snapshot_id"], False)
                        _traj_chip = (
                            "con trayectoria" if _has_traj
                            else "solo punto Op (legacy)"
                        )
                        cols_h[0].markdown(
                            f"**{s['corrida_label'][:30]}** · {_traj_chip}  \n"
                            f"_{s['timestamp']} · {s['n_sensors']} sensores · "
                            f"{s.get('operating_speed_rpm', 0):.0f} rpm_"
                        )
                        if cols_h[1].button(
                            "️",
                            key=f"wm_polar_del_{s['snapshot_id']}",
                            help="Borrar este snapshot",
                        ):
                            if delete_polar_snapshot(_polar_inst_id, s["snapshot_id"]):
                                st.success("Borrado.")
                                st.rerun()

            # Persistir los snapshot ids elegidos (lista) para que el
            # render del polar y del PDF los pueda leer.
            st.session_state["wm_polar_compare_snapshot_ids"] = _selected_polar_cmp_ids
            # Backward-compat: si hay 1 elegido, lo expongo como single
            st.session_state["wm_polar_compare_snapshot_id"] = (
                _selected_polar_cmp_ids[0] if _selected_polar_cmp_ids else "__none__"
            )
            st.session_state["wm_polar_compare_skip_identical"] = bool(_polar_curr_readings)
            st.session_state["wm_polar_compare_inst_id"] = _polar_inst_id
        elif _polar_hist_ok and not _polar_inst_id:
            st.caption(
                "_(Activá una Asset Instance arriba para guardar histórico.)_"
            )

    if not selected_ids:
        st.info("Selecciona uno o más polares en la barra lateral.")
        return

    selected_items = [id_to_item[sid] for sid in selected_ids]
    logo_uri = get_logo_data_uri(LOGO_PATH)

    # ============================================================
    # Ciclo 17.1.1 — Comparativo Polar inline (multi-snapshot)
    # ------------------------------------------------------------
    # Cuando el usuario elige una o más corridas anteriores,
    # mostramos arriba de los paneles individuales una tabla
    # comparativa con UNA fila por (sensor × snapshot) + diagnostico
    # de shift de fase + crecimiento de amplitud.
    # ============================================================
    _polar_cmp_ids: List[str] = st.session_state.get(
        "wm_polar_compare_snapshot_ids", []
    ) or []
    if _polar_cmp_ids and _polar_curr_readings:
        try:
            from core.polar_history import (
                load_polar_snapshot,
                phase_shift_classifier,
                amplitude_change_classifier,
                shortest_arc_phase_diff,
            )

            _cmp_rows = []
            _snap_meta_by_id = {}
            for _snap_id in _polar_cmp_ids:
                _snap_full = load_polar_snapshot(_polar_inst_id, _snap_id)
                if _snap_full is None:
                    continue
                _snap_meta_by_id[_snap_id] = _snap_full
                _prev_by_lbl = {
                    str(s.get("sensor_label", "")): s
                    for s in _snap_full.get("sensors", [])
                }
                _snap_label_short = _snap_full.get("corrida_label", _snap_id)[:22]
                _snap_ts = (_snap_full.get("timestamp", "") or "")[:10]

                for r in _polar_curr_readings:
                    _lbl = r["sensor_label"]
                    _prev = _prev_by_lbl.get(_lbl)
                    if _prev is None:
                        _cmp_rows.append({
                            "Sensor": _lbl,
                            "vs Corrida": f"{_snap_label_short} ({_snap_ts})",
                            "Anterior amp": "—",
                            "Actual amp": f"{r['amp_at_op']:.3f} {r['amp_unit']}",
                            "Δ amp": "—",
                            "Anterior fase": "—",
                            "Actual fase": f"{r['phase_at_op']:.1f}°",
                            "Δ fase": "—",
                            "Diagnóstico": "Sin lectura previa para este sensor",
                        })
                        continue
                    _prev_amp = float(_prev.get("amp_at_op", 0))
                    _prev_phase = float(_prev.get("phase_at_op", 0))
                    _delta_amp = r["amp_at_op"] - _prev_amp
                    _delta_amp_pct = (_delta_amp / _prev_amp * 100.0) if _prev_amp > 0 else None
                    _delta_phase = shortest_arc_phase_diff(_prev_phase, r["phase_at_op"])
                    _phase_class = phase_shift_classifier(_delta_phase)
                    _amp_class = amplitude_change_classifier(_delta_amp_pct)

                    _diag_parts = []
                    if _phase_class == "shift_critical":
                        _diag_parts.append("Shift fase crítico (>60°)")
                    elif _phase_class == "shift_major":
                        _diag_parts.append("Shift fase mayor (≥30°)")
                    elif _phase_class == "shift_minor":
                        _diag_parts.append("Shift fase menor (10–30°)")
                    elif _phase_class == "stable":
                        _diag_parts.append("Fase estable (<10°)")
                    if _delta_amp_pct is not None:
                        if _amp_class in ("amp_critical", "amp_high"):
                            _diag_parts.append(f"Amp {_delta_amp_pct:+.0f}% (alza)")
                        elif _amp_class == "amp_up":
                            _diag_parts.append(f"Amp {_delta_amp_pct:+.0f}%")
                        elif _amp_class in ("amp_down_strong", "amp_down"):
                            _diag_parts.append(f"Amp {_delta_amp_pct:+.0f}%")

                    _cmp_rows.append({
                        "Sensor": _lbl,
                        "vs Corrida": f"{_snap_label_short} ({_snap_ts})",
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
                st.markdown("### Comparativo Polar — vs corridas anteriores")
                _n_snaps = len(_snap_meta_by_id)
                if _n_snaps == 1:
                    _only = list(_snap_meta_by_id.values())[0]
                    st.caption(
                        f"Comparando contra **{_only.get('corrida_label', '')}** "
                        f"del {_only.get('timestamp', '')[:10]}. Shift de fase "
                        f"1X >30° es síntoma diagnóstico de cambio de balance "
                        f"del rotor (API 684 / ISO 21940-12)."
                    )
                else:
                    st.caption(
                        f"Mostrando comparativo contra **{_n_snaps} corridas "
                        f"anteriores**. Cada fila es una combinación "
                        f"(sensor × corrida). Shift de fase 1X >30° = "
                        f"síntoma de cambio de balance (API 684)."
                    )
                _cmp_disp = pd.DataFrame(_cmp_rows)
                st.dataframe(_cmp_disp, width="stretch", hide_index=True)
        except Exception as _polar_cmp_e:
            st.caption(f"_(Comparativo Polar no disponible: {_polar_cmp_e})_")

    for panel_index, item in enumerate(selected_items):
        render_polar_panel(
            item=item,
            panel_index=panel_index,
            logo_uri=logo_uri,
            smooth_window=smooth_window,
            amp_smooth_window=amp_smooth_window,
            show_info_box=show_info_box,
            show_rpm_labels=show_rpm_labels,
            marker_stride=marker_stride,
            detect_cs=detect_cs,
            max_critical_speeds=max_critical_speeds,
            use_rotordyn_pro=use_rotordyn_pro,
            operating_rpm=float(operating_rpm),
            machine_group=machine_group,
            speed_min_filter=None if auto_speed else float(speed_min_filter),
            speed_max_filter=None if auto_speed else float(speed_max_filter),
            iso_part=active_iso_part,
            custom_thresholds=active_custom_thresholds,
            profile_label=active_profile_label,
        )

        if panel_index < len(selected_items) - 1:
            st.markdown("---")

    if len(selected_items) >= 2:
        render_polar_compare_section(
            selected_items,
            smooth_window=smooth_window,
            amp_smooth_window=amp_smooth_window,
            max_critical_speeds=max_critical_speeds,
            logo_uri=logo_uri,
            use_rotordyn_pro=use_rotordyn_pro,
            operating_rpm=float(operating_rpm),
            machine_group=machine_group,
            speed_min_filter=None if auto_speed else float(speed_min_filter),
            speed_max_filter=None if auto_speed else float(speed_max_filter),
            iso_part=active_iso_part,
            custom_thresholds=active_custom_thresholds,
        )



if __name__ == "__main__":
    main()
