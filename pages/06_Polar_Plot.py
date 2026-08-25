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
        raise ValueError("Empty file.")

    header_idx = find_header_line(
        lines,
        required_signals=("Amp", "Phase", "Speed", "Timestamp"),
    )
    if header_idx is None:
        raise ValueError("Could not find the actual header row in the Polar CSV.")

    meta = parse_metadata_block(lines[:header_idx])
    data_text = "\n".join(lines[header_idx:])

    df = pd.read_csv(io.StringIO(data_text), encoding="utf-8-sig")

    required = ["Amp", "Amp Status", "Phase", "Phase Status", "Speed", "Speed Status", "Timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in the CSV: {missing}")

    df["amp"] = pd.to_numeric(df["Amp"], errors="coerce")
    df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")
    df["speed"] = pd.to_numeric(df["Speed"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["amp", "phase", "speed", "Timestamp"]).copy()
    df = filter_status_valid(df, ["Amp Status", "Phase Status", "Speed Status"])

    if df.empty:
        raise ValueError("No valid rows remained after filtering.")

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
                        f"<b>Critical #{idx+1}</b><br>"
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
                    "label": prev_snapshot_label or "previous",
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
                _prev_lbl_text = _snap.get("label", "previous") or "previous"
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
                                    f"<b>Peak {_prev_lbl_text}</b><br>"
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
                            name=f"Trail {_prev_lbl_text} → current",
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
                        f"<b>Rated operation</b><br>"
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
        action_clean = re.sub(r"^(Se recomienda|Recommended):\s*", "", action_clean, flags=re.IGNORECASE)
        action_clean = action_clean.strip()
        if action_clean:
            blocks.append("Recommended:\n" + action_clean)

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
            headline = f"Severe polar response consistent with dynamic amplification near {cs_speed:.0f} rpm"
            detail = (
                f"The polar trajectory shows a significant dynamic response around {cs_speed:.0f} rpm, "
                f"with an approximate amplitude of {cs_amp:.3f} and a phase change of {phase_delta:.1f}°. "
                f"The combination of rising amplitude and phase change suggests proximity to a critical speed, "
                f"loss of dynamic margin, or a relevant change in stiffness/damping of the rotor-support system.\n\n"
                f"From a rotordynamics standpoint, this condition should be correlated with the Bode plot, orbits, "
                f"waveform, shaft centerline and actual load conditions."
            )
            action = (
                "Priority actions:\n"
                "- Correlate the polar peak with the amplitude and phase Bode plot\n"
                "- Confirm repeatability during startup/coastdown\n"
                "- Verify alignment, support stiffness, balance and bearing condition\n"
                "- Review the phase change around the identified regime\n"
                "- Avoid sustained operation near the critical regime until the evaluation is complete"
            )
        elif status_up == "WARNING":
            headline = f"Polar response with signs of dynamic amplification near {cs_speed:.0f} rpm"
            detail = (
                f"The polar trajectory shows a relevant response zone around {cs_speed:.0f} rpm, "
                f"with an approximate amplitude of {cs_amp:.3f} and a phase change of {phase_delta:.1f}°. "
                f"The behavior is consistent with moderate dynamic amplification, without sufficient evidence to classify it as severe.\n\n"
                f"From a vibration-analysis standpoint, this condition should be kept under monitoring, especially if the peak repeats "
                f"in later runs or is accompanied by a rise in 1X, a phase change, or a change in the orbit."
            )
            action = (
                "- Compare against historical runs and the baseline condition\n"
                "- Validate the response with the Bode plot and 1X spectrum\n"
                "- Confirm whether a rising amplitude trend exists\n"
                "- Keep monitoring during upcoming startups/coastdowns"
            )
        else:
            headline = f"Controlled polar response with a dynamic candidate near {cs_speed:.0f} rpm"
            detail = (
                f"A dynamic candidate is identified around {cs_speed:.0f} rpm, with an approximate amplitude of {cs_amp:.3f} "
                f"and a phase change of {phase_delta:.1f}°. The polar trajectory shows no severe response under this condition.\n\n"
                f"The behavior is compatible with stable operation, although the identified point should be kept as a reference for future comparison."
            )
            action = (
                "- Keep the run as a baseline\n"
                "- Compare with future polar trajectories\n"
                "- Correlate with the Bode plot, orbit and 1X amplitude trend"
            )
    else:
        headline = "Polar response with no clearly identified dominant critical speed"
        detail = (
            f"The polar trajectory shows a peak amplitude of {max_amp:.3f} and no clear critical-speed candidate under the applied heuristic. "
            f"The condition should be correlated with the Bode plot, spectrum, orbit and operating variables before concluding the dominant mechanism."
        )
        action = (
            "- Keep historical monitoring\n"
            "- Compare against future runs\n"
            "- Validate with the Bode plot, spectrum and orbit if the condition changes"
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
                "The phase change is representative enough to suspect a marked modal transition. "
                "Below the critical speed the rotor tends to behave with a stiffer response; after passing through the modal zone, "
                "the response becomes more flexible and the phase reflects the change in the relationship between exciting force and displacement."
            )
        elif abs(phase_delta) >= 15.0:
            modal_txt = (
                "The phase change is moderate and suggests approach to a dynamic amplification zone. "
                "The response does not by itself confirm a fully developed critical crossing, but it does show a change in apparent dynamic stiffness."
            )
        else:
            modal_txt = (
                "The phase change is low, so the identified point should be treated as a dynamic candidate and not as a confirmed critical speed. "
                "Confirmation requires correlation with the Bode plot, phase, orbit and repeatability across runs."
            )

        if status_up == "DANGER":
            headline = f"Severe polar response associated with a possible critical speed near {cs_speed:.0f} rpm"
        elif status_up == "WARNING":
            headline = f"Polar response with signs of dynamic amplification near {cs_speed:.0f} rpm"
        else:
            headline = f"Controlled polar response with a modal candidate near {cs_speed:.0f} rpm"

        detail = (
            f"The polar trajectory identifies a zone of interest around {cs_speed:.0f} rpm, with an approximate amplitude of {cs_amp:.3f} "
            f"and a phase change of {phase_delta:.1f}°. This condition is compatible with a possible approach to a critical speed or modal shape of the rotor-support system.\\n\\n"
            f"{modal_txt}\\n\\n"
            f"From a rotordynamics-analysis standpoint, interpretation should focus on the relationship between amplitude, phase and speed. "
            f"A rise in amplitude accompanied by a consistent phase change may indicate passage through a modal shape; if amplitude rises without a sufficient phase change, "
            f"the condition may be more associated with unbalance, eccentricity, forced response or operating changes."
        )

        action = (
            "Correlate the polar trajectory with the amplitude and phase Bode plot.\\n"
            "Verify repeatability of the modal zone across startups/coastdowns.\\n"
            "Compare against 1X-filtered orbits and shaft centerline.\\n"
            "Confirm whether the phase change occurs before, during or after the amplitude peak.\\n"
            "Validate balance, alignment, support stiffness, lubrication and load conditions."
        )
    else:
        headline = "Polar response with no clearly identified dominant critical speed"
        detail = (
            f"The polar trajectory shows a peak amplitude of {max_amp:.3f}, with no dominant modal candidate under the applied heuristic. "
            f"There is no sufficiently clear combination of rising amplitude and phase change to confirm a critical speed.\\n\\n"
            f"This condition may represent a stable response, a forced excitation, or a run where the critical regime was not crossed clearly enough."
        )
        action = (
            "Keep the run as a historical reference.\\n"
            "Compare against future polar trajectories.\\n"
            "Correlate with the Bode plot, 1X spectrum, orbit and operating variables."
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

    amp_trend = "an increase" if delta_amp > 0.15 else "a reduction" if delta_amp < -0.15 else "stability"
    speed_shift = "a shift toward higher speed" if delta_speed > 100 else "a shift toward lower speed" if delta_speed < -100 else "no relevant speed shift"

    headline = "Multi-date comparison of polar trajectory and modal response"

    detail = (
        f"{len(ordered)} polar runs were compared. Between the baseline run ({baseline['label']}) and the most recent one ({latest['label']}), "
        f"the dominant amplitude shows {amp_trend} ({delta_amp:+.3f}), {speed_shift} ({delta_speed:+.0f} rpm) "
        f"and a dominant phase change of {delta_phase:+.1f}°.\\n\\n"
        f"From a rotordynamics standpoint, the polar comparison makes it possible to assess whether the rotor response keeps the same modal pattern or whether the critical zone has migrated. "
        f"When the amplitude peak and the phase change shift between runs, there may be a change in effective stiffness, damping, support condition, balance or load.\\n\\n"
        f"Below a critical speed the rotor tends to behave as a stiffer system; on crossing a modal shape, the phase and the trajectory change and the rotor shows flexible behavior. "
        f"For that reason, reading amplitude, phase and speed together is more conclusive than amplitude alone."
    )

    action = (
        "Correlate the polar runs with the amplitude/phase Bode plot.\\n"
        "Verify whether the candidate speed repeats or migrates across dates.\\n"
        "Compare against 1X orbits and shaft centerline.\\n"
        "Validate whether there were changes in balance, alignment, lubrication, temperature or load.\\n"
        "Use the most stable run as the acceptance baseline."
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
    st.markdown("## Multi-date comparison · Polar Plot")

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
            trace_name = f"{date_label}  ·  Q={q_str}  ·  zone {zone}"
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
                    hovertemplate=f"<b>Detected critical</b><br>{cs_marker_label}<br>Amp: %{{r:.3f}}<br>Phase: %{{theta:.1f}}°<extra></extra>",
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
                    hovertemplate=f"<b>Rated operation</b><br>{operating_rpm:.0f} rpm<br>Amp: %{{r:.3f}}<br>Phase: %{{theta:.1f}}°<extra></extra>",
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
    dt_text = "Multi-date comparison"
    if dt_values:
        dt_text = " / ".join([pd.Timestamp(v).strftime("%Y-%m-%d") for v in dt_values[:4]])

    draw_top_strip(
        fig=fig,
        machine=items[0].get("machine", ""),
        point_text="Polar Plot · Multi-date comparison",
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
                "Date": pd.Timestamp(r["ts_start"]).strftime("%Y-%m-%d") if r["ts_start"] is not None else "—",
                "File": r["label"],
                "Critical RPM": f"{cs_pro.rpm:.0f}" if cs_pro is not None else "—",
                "Q factor": f"{cs_pro.q_factor:.2f}" if (cs_pro is not None and np.isfinite(cs_pro.q_factor)) else "—",
                "Δphase (°)": f"{cs_pro.phase_change_deg:.0f}" if cs_pro is not None else "—",
                "FWHM (rpm)": f"{cs_pro.fwhm_rpm:.0f}" if (cs_pro is not None and np.isfinite(cs_pro.fwhm_rpm)) else "—",
                peak_col_label: peak_fmt.format(r.get("peak_amp_csv", 0.0)),
                "ISO zone": iso_eval.zone if iso_eval is not None else "—",
                "API 684": ("✓" if api_pro is not None and api_pro.compliant else "✗") if api_pro is not None else "—",
            })
        summary = pd.DataFrame(rows)
    else:
        summary = pd.DataFrame([
            {
                "File": r["label"],
                "Start date": pd.Timestamp(r["ts_start"]).strftime("%Y-%m-%d %H:%M") if r["ts_start"] is not None else "—",
                "End date": pd.Timestamp(r["ts_end"]).strftime("%Y-%m-%d %H:%M") if r["ts_end"] is not None else "—",
                "Dominant amp": round(r["dominant_amp"], 3),
                "Candidate RPM": round(r["dominant_speed"], 0),
                "Phase delta": round(r["dominant_phase_delta"], 1),
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

    st.markdown("### Automatic comparative diagnostic")
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
                compliant_str = "API 684 compliant" if api_pro.compliant else "NOT API 684 compliant"
                prose_lines.append(
                    f"The {date_str} run ({r['label']}) reported a critical speed at "
                    f"{cs_pro.rpm:.0f} rpm with a Q factor of {q_str} and a peak amplitude of {amp_str}, "
                    f"classified in zone {iso_eval.zone} of ISO 20816-2 and {compliant_str}."
                )
            else:
                prose_lines.append(
                    f"The {date_str} run ({r['label']}) shows no critical speed "
                    f"detectable under the automatic criteria."
                )

        prose_summary = "\n\n".join(prose_lines)

        notes = (
            f"{diag['detail']}\n\n"
            f"Chronological synthesis of the analyzed runs:\n\n"
            f"{prose_summary}\n\n"
            f"{diag['action']}"
        )
    else:
        summary_lines = []
        for _, row in summary.iterrows():
            summary_lines.append(
                f"- {row['File']}: candidate {row['Candidate RPM']:.0f} rpm, "
                f"dominant amplitude {row['Dominant amp']:.3f}, "
                f"Δphase {row['Phase delta']:.1f}°, max {row['Max amp']:.3f}."
            )

        notes = (
            _build_polar_report_notes(diag)
            + "\n\nRun comparison summary:\n"
            + "\n".join(summary_lines)
        )

    png_bytes, png_error = build_export_png_bytes(fig, diag)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Send Polar comparison to report", key="wm_polar_compare_report_btn"):
            ensure_report_state()
            append_report_item_and_persist(
                {
                    "type": "polar_compare",
                    "title": "Polar Plot · Multi-date comparison",
                    "notes": notes,
                    "image_bytes": png_bytes,
                }
            )
            st.success("Polar comparison sent to the report.")
    with c2:
        if png_bytes is not None:
            st.download_button(
                "Download Polar comparison PNG",
                data=png_bytes,
                file_name="polar_compare_hd.png",
                mime="image/png",
                key="wm_polar_compare_download_btn",
                width="stretch",
            )
        elif png_error:
            st.warning(f"Could not generate the comparison PNG: {png_error}")

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
                f"Panel {panel_index + 1}: no points in the RPM range "
                f"[{lo:.0f} – {hi:.0f}]. Adjust the range in the sidebar."
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
                _prev_lbl_text = _prev_label or "previous run"

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
                _mode_type = "uncharacterized mode"
                if _curr_cs_phase_delta is not None and _curr_cs_phase_delta != 0:
                    _abs_pd = abs(_curr_cs_phase_delta)
                    if 150.0 <= _abs_pd <= 210.0:
                        _mode_type = (
                            "first translational mode (Δφ ≈ 180°), "
                            "consistent with an 'in-phase' rotor "
                            "bending mode per Bently nomenclature and "
                            "API 684 §6"
                        )
                    elif 70.0 <= _abs_pd < 150.0:
                        _mode_type = (
                            "conical / pivotal mode or second "
                            "translational mode (Δφ ≈ 90–150°), a typical "
                            "response when the rotor pivots about "
                            "a nodal point near the bearing"
                        )
                    elif 210.0 < _abs_pd <= 360.0:
                        _mode_type = (
                            "second flexural mode or coupled "
                            "rotor-structure response (Δφ > 210°)"
                        )
                    else:
                        _mode_type = (
                            "low modal deflection response "
                            "(Δφ < 70°) — possibly a structural "
                            "resonance of the support / foundation rather "
                            "than a rotor mode"
                        )

                _narr_parts: List[str] = []

                # 1) Encabezado factual
                _narr_parts.append(
                    f"Rotordynamic comparative analysis against "
                    f"«{_prev_lbl_text}»"
                    + (
                        f" from {_prev_label[:10]}"
                        if _prev_label and len(_prev_label) >= 10 else ""
                    )
                    + ". At the operating speed "
                    f"({operating_rpm:.0f} rpm), the sensor's 1X synchronous "
                    f"response evolved from "
                    f"{float(_prev_amp):.3f} {_amp_unit_local} @ "
                    f"{float(_prev_phase):.1f}° to "
                    f"{_curr_amp_op:.3f} {_amp_unit_local} @ "
                    f"{_curr_phase_op:.1f}°, representing a "
                    + (
                        f"vector change of {_delta_amp:+.3f} {_amp_unit_local} "
                        f"({_delta_amp_pct:+.1f}%)"
                        if _delta_amp_pct is not None
                        else f"vector change of {_delta_amp:+.3f} {_amp_unit_local}"
                    )
                    + f" in magnitude and a 1X phase shift of "
                    f"{_delta_phase:+.1f}° along the minor arc."
                )

                # 2) Caracterización del modo y forma estructural
                if _curr_cs_rpm and _curr_cs_rpm > 0:
                    _ratio_op_cs = operating_rpm / _curr_cs_rpm
                    _separation_pct = (_ratio_op_cs - 1.0) * 100.0
                    _q_str = (
                        f", with amplification factor Q={_curr_q:.2f}"
                        if _curr_q else ""
                    )
                    _mode_para = (
                        f"The polar trajectory reveals a critical "
                        f"speed at {_curr_cs_rpm:.0f} rpm with a "
                        f"phase change of "
                        f"{abs(_curr_cs_phase_delta or 0):.0f}° across "
                        f"the peak{_q_str}. This pattern is interpreted "
                        f"as {_mode_type}. "
                    )
                    if _separation_pct >= 15.0:
                        _mode_para += (
                            f"The operating speed sits "
                            f"{_separation_pct:+.1f}% above the "
                            f"mode, a wide separation that meets the "
                            f"API 684 §6 margin requirement. "
                        )
                    elif _separation_pct >= 0:
                        _mode_para += (
                            f"The operating speed sits only "
                            f"{_separation_pct:+.1f}% above the "
                            f"mode. This is narrow against the "
                            f"recommended API 684 §6 margin (≥15%) and "
                            f"warrants a detailed separation-margin "
                            f"evaluation if Q increases. "
                        )
                    else:
                        _mode_para += (
                            f"The operating speed is "
                            f"{abs(_separation_pct):.1f}% below "
                            f"the identified mode, a sub-critical "
                            f"configuration considered stable "
                            f"as long as Q stays bounded. "
                        )
                    _narr_parts.append(_mode_para)

                # 3) Diagnóstico diferencial del shift
                if _phase_class == "shift_critical":
                    _narr_parts.append(
                        "The 1X phase shift exceeds 60° along the minor "
                        "arc, a magnitude considered critical under the "
                        "Bently / API 684 criteria. Shifts of this scale "
                        "are inconsistent with simple thermal or "
                        "operational drift and point to a structural "
                        "mechanical change in the rotor: mass loss from "
                        "a detached part, crack propagation, sudden "
                        "bearing settlement, or loss of contact at the "
                        "seal / impeller. A controlled shutdown is "
                        "recommended for inspection and complementary "
                        "analysis of 1X-filtered orbits and waveform in "
                        "both bearing planes before continuing operation."
                    )
                elif _phase_class == "shift_major":
                    _narr_parts.append(
                        "The 1X phase shift between 30° and 60° is the "
                        "classic signature of a rotor balance change "
                        "under the polar vector-response methodology "
                        "documented by Bently and API 684. The magnitude "
                        "and direction of the vector change are "
                        "consistent with a redistribution of rotating "
                        "mass (dirt accumulated or shed on blades, "
                        "progressive loss of balance weights, "
                        "residual thermal distortion). Scheduling a "
                        "field balance per ISO 21940-12 grade G 2.5 in "
                        "the next window is recommended, first verifying "
                        "phase consistency across successive startups to "
                        "rule out a transient component."
                    )
                elif _phase_class == "shift_minor":
                    _narr_parts.append(
                        "The 1X phase shift between 10° and 30° is minor "
                        "and can be attributed to normal operational "
                        "drift (temperature, load, thermal expansion of "
                        "the rotor or the supports). By itself it is not "
                        "evidence of a mechanical change, but it warrants "
                        "monitoring: a consolidated trend across several "
                        "runs in the same vector direction would indicate "
                        "an incipient balance change worth characterizing "
                        "before it crosses into the major zone."
                    )
                elif _phase_class == "stable":
                    _narr_parts.append(
                        "The 1X phase shift (<10°) is within the normal "
                        "variation of the synchronous response and is "
                        "not evidence of a mechanical or balance change. "
                        "The vector shape is considered stable across "
                        "runs."
                    )

                # 4) Análisis del cambio de amplitud (sensitividad)
                if _delta_amp_pct is not None:
                    _amp_class = amplitude_change_classifier(_delta_amp_pct)
                    if _amp_class == "amp_critical":
                        _narr_parts.append(
                            "The 1X amplitude growth exceeds "
                            "50% between runs. Combined with the "
                            "described phase shift, it reinforces the "
                            "diagnosis of active degradation of the "
                            "modal response — the rotor's sensitivity to "
                            "residual excitation force is increasing, "
                            "which is typical of damping degradation in "
                            "the hydrodynamic bearings under the "
                            "API 684 analysis framework."
                        )
                    elif _amp_class == "amp_high":
                        _narr_parts.append(
                            "The 1X amplitude growth (≥20%) "
                            "accompanying the phase shift is consistent "
                            "with an active change in the modal response "
                            "of the rotor-support system. It is worth "
                            "reviewing the amplification factor Q in "
                            "upcoming runs to rule out progressive "
                            "damping loss."
                        )
                    elif _amp_class in ("amp_down_strong", "amp_down"):
                        _narr_parts.append(
                            "The 1X amplitude dropped significantly "
                            "relative to the previous run. If this "
                            "coincides with a major phase shift, it may "
                            "reflect a compensatory balance change "
                            "(e.g. a prior intervention, thermal "
                            "redistribution) rather than degradation. It "
                            "is advisable to review the operational log "
                            "and the maintenance reports between runs to "
                            "confirm."
                        )

                # 5) Distinción modal del rotor vs estructural
                if (
                    _phase_class in ("shift_major", "shift_critical")
                    and _curr_cs_phase_delta is not None
                    and abs(_curr_cs_phase_delta) < 90.0
                ):
                    _narr_parts.append(
                        "Differential note: the phase change across "
                        "the observed peak (<90°) is atypical of a "
                        "free rotor mode and suggests the peak "
                        "could correspond to a structural "
                        "resonance of the support or foundation rather "
                        "than a rotor mode. It is important to validate "
                        "before attributing the observed change to "
                        "rotor balance — a mechanical change in the "
                        "foundation (loose anchor bolts, grouting "
                        "deterioration) produces the same pattern in the "
                        "polar plot but requires structural "
                        "intervention, not balancing."
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
            (f"Status: {semaforo_status}", semaforo_color),
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
        "AI clinical interpretation · Assisted Cat IV diagnostic",
        expanded=False,
    ):
        if not is_ai_available():
            st.info(
                "**AI diagnostic not available.** `[anthropic] api_key` must be "
                "configured in the Streamlit secrets."
            )
        else:
            stored_pol = st.session_state.get(ai_state_key_pol)
            ai_btn_col1, ai_btn_col2, ai_btn_col3 = st.columns([1.4, 1.4, 2.4])
            with ai_btn_col1:
                gen_clicked_pol = st.button(
                    "Generate AI diagnostic"
                    if stored_pol is None
                    else "Diagnostic generated",
                    key=f"ai_gen_btn_pol_{export_state_key}",
                    use_container_width=True,
                    type="primary" if stored_pol is None else "secondary",
                    disabled=stored_pol is not None and stored_pol.get("ok", False),
                )
            with ai_btn_col2:
                regen_clicked_pol = st.button(
                    "Regenerate",
                    key=f"ai_regen_btn_pol_{export_state_key}",
                    use_container_width=True,
                    disabled=stored_pol is None,
                )
            with ai_btn_col3:
                st.caption(
                    "Claude Sonnet 4.5 · ~$0.015 per diagnostic · "
                    "cached 30 days unless you regenerate."
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

                with st.spinner("Claude analyzing the polar response... (5-15 s)"):
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
                                f"_Unexpected error generating the AI diagnostic:_\n\n"
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
                            "Diagnostic generated with the fallback model "
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
                        " · fallback model"
                        if stored_pol.get("fallback_used") else ""
                    )
                    st.caption(
                        f"Model: `{model_used_pol}` · "
                        f"Tokens: {stored_pol.get('input_tokens', 0)} → "
                        f"{stored_pol.get('output_tokens', 0)} · "
                        f"Cost: ~${cost_usd_pol:.4f} · "
                        f"{'(cached)' if stored_pol.get('cached') else '(newly generated)'}"
                        f"{fallback_tag_pol}"
                    )
                else:
                    st.error(
                        stored_pol.get("markdown", "Error generating the AI diagnostic.")
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
        quant_lines_pol: List[str] = ["Parameter|Value"]
        if operating_rpm:
            quant_lines_pol.append(
                f"Operating speed|{float(operating_rpm):.0f} RPM"
            )
        _max_amp = float(polar_diag.get("max_amp", 0.0) or 0.0)
        if _max_amp > 0:
            quant_lines_pol.append(
                f"Peak amplitude|{_max_amp:.3f} {amp_unit}".strip()
            )
        _score = float(polar_diag.get("score", 0.0) or 0.0)
        if _score > 0:
            quant_lines_pol.append(f"Health score|{_score:.1f}")
        _candidates = int(polar_diag.get("candidate_count", 0) or 0)
        quant_lines_pol.append(f"Detected critical speeds|{_candidates}")
        if semaforo_status:
            quant_lines_pol.append(f"Status|{semaforo_status}")
        _pt = str(meta.get("Point Name", "") or "").strip()
        if _pt:
            quant_lines_pol.append(f"Measurement point|{_pt}")
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
                st.caption(f"Active Polar files: {len(active_polar_files)}")
            else:
                st.caption("No Polar files loaded")

        with col2:
            if st.button("Clear Polar files", key="wm_polar_clear_files_btn"):
                clear_polar_persisted_files()
                st.rerun()

        uploaded_files = active_polar_files

    if not uploaded_files:
        panel_card(
            title="Load files to begin",
            subtitle="Upload one or more Polar CSV files from the left panel.",
            meta_html="",
            chips=[],
        )
        return

    parsed_items, failed_items = parse_uploaded_polar_files(uploaded_files)

    if failed_items:
        for file_name, error_text in failed_items:
            st.warning(f"Could not read {file_name}: {error_text}")

    if not parsed_items:
        st.error("No valid Polar file could be loaded.")
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
        st.markdown("### 📚 Polar History")

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
            st.caption(f"_(Polar history not available: {_hist_e})_")

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
                f" (filtered by sensor **{_polar_filter_sensor}**)"
                if _polar_filter_sensor else ""
            )
            st.caption(
                f"{len(_polar_existing_snaps)} Polar snapshot(s) saved "
                f"for this unit{_polar_filter_hint}."
            )

            if not _polar_curr_readings:
                if not _polar_sensors_map:
                    st.caption(
                        "_(No Sensor Map configured for this instance. "
                        "Go to Machinery Library to set it up.)_"
                    )
                else:
                    st.caption(
                        "_(No loaded Polar CSV matches sensors in this "
                        "instance's Sensor Map.)_"
                    )
            else:
                with st.expander("📸 Save current Polar snapshot", expanded=False):
                    st.caption(
                        f"Captures 1X amp + phase at {operating_rpm:.0f} rpm "
                        f"for {len(_polar_curr_readings)} matched sensor(s)."
                    )
                    _polar_snap_label = st.text_input(
                        "Run label",
                        value="",
                        placeholder="e.g. Coastdown Apr 27",
                        key=f"wm_polar_snap_label_{_polar_inst_id}",
                    )
                    _polar_snap_notes = st.text_area(
                        "Notes (optional)",
                        value="",
                        placeholder="Operating speed, condition, event.",
                        key=f"wm_polar_snap_notes_{_polar_inst_id}",
                        height=70,
                    )
                    if st.button(
                        "Save Polar snapshot",
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
                                f"✓ Polar snapshot saved: {sid} "
                                f"({len(_polar_curr_readings)} sensors)"
                            )
                            st.rerun()
                        except Exception as _e:
                            st.error(f"Could not save: {_e}")

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
                    _suffix = " · (current run)" if _is_current else ""
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
                    "Runs to overlay on the polar",
                    options=_polar_opt_lbls,
                    default=_polar_default_pick,
                    key=f"wm_polar_cmp_multi_{_polar_inst_id}",
                    help=(
                        "Pick 0 runs to see only the current one, 1 for a "
                        "simple comparison, or several for a historical "
                        "overlay with a chronological gradient (older ones "
                        "lighter, more recent ones darker)."
                    ),
                )
                st.session_state[_polar_cmp_state_key] = _picked
                _selected_polar_cmp_ids = [
                    _polar_lbl_to_key[l] for l in _picked if l in _polar_lbl_to_key
                ]
                if not _selected_polar_cmp_ids:
                    st.caption("_Only the current run will be shown._")
                else:
                    st.caption(
                        f"**{len(_selected_polar_cmp_ids)}** previous "
                        f"run(s) will be overlaid on the current one."
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
                        f"{_legacy_count} old snapshot(s) without a "
                        f"full trajectory — they only show the operating "
                        f"point on the polar. To see the full loop, "
                        f"re-snapshot by loading that run and saving "
                        f"again."
                    )

                with st.expander(f"️ Manage Polar snapshots ({len(_polar_existing_snaps)})"):
                    if _legacy_count > 0:
                        if st.button(
                            f"🧹 Delete the {_legacy_count} snapshot(s) without trajectory",
                            key=f"wm_polar_del_legacy_{_polar_inst_id}",
                            help=(
                                "Deletes all snapshots saved before "
                                "Cycle 17.1.2 (without a full trail). "
                                "Current snapshots with a trajectory are "
                                "NOT touched."
                            ),
                        ):
                            _deleted = 0
                            for s in _polar_existing_snaps:
                                if not _snap_has_trail.get(s["snapshot_id"], False):
                                    if delete_polar_snapshot(_polar_inst_id, s["snapshot_id"]):
                                        _deleted += 1
                            st.success(
                                f"✓ {_deleted} old snapshot(s) deleted. "
                                f"Load the runs and save new snapshots "
                                f"to rebuild the history with a "
                                f"trajectory."
                            )
                            st.rerun()
                        st.markdown("---")

                    for s in _polar_existing_snaps:
                        cols_h = st.columns([4, 1])
                        _has_traj = _snap_has_trail.get(s["snapshot_id"], False)
                        _traj_chip = (
                            "with trajectory" if _has_traj
                            else "operating point only (legacy)"
                        )
                        cols_h[0].markdown(
                            f"**{s['corrida_label'][:30]}** · {_traj_chip}  \n"
                            f"_{s['timestamp']} · {s['n_sensors']} sensors · "
                            f"{s.get('operating_speed_rpm', 0):.0f} rpm_"
                        )
                        if cols_h[1].button(
                            "️",
                            key=f"wm_polar_del_{s['snapshot_id']}",
                            help="Delete this snapshot",
                        ):
                            if delete_polar_snapshot(_polar_inst_id, s["snapshot_id"]):
                                st.success("Deleted.")
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
                "_(Activate an Asset Instance above to save history.)_"
            )

    if not selected_ids:
        st.info("Select one or more polars in the sidebar.")
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
                            "vs Run": f"{_snap_label_short} ({_snap_ts})",
                            "Previous amp": "—",
                            "Current amp": f"{r['amp_at_op']:.3f} {r['amp_unit']}",
                            "Δ amp": "—",
                            "Previous phase": "—",
                            "Current phase": f"{r['phase_at_op']:.1f}°",
                            "Δ phase": "—",
                            "Diagnostic": "No previous reading for this sensor",
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
                        _diag_parts.append("Critical phase shift (>60°)")
                    elif _phase_class == "shift_major":
                        _diag_parts.append("Major phase shift (≥30°)")
                    elif _phase_class == "shift_minor":
                        _diag_parts.append("Minor phase shift (10–30°)")
                    elif _phase_class == "stable":
                        _diag_parts.append("Stable phase (<10°)")
                    if _delta_amp_pct is not None:
                        if _amp_class in ("amp_critical", "amp_high"):
                            _diag_parts.append(f"Amp {_delta_amp_pct:+.0f}% (rising)")
                        elif _amp_class == "amp_up":
                            _diag_parts.append(f"Amp {_delta_amp_pct:+.0f}%")
                        elif _amp_class in ("amp_down_strong", "amp_down"):
                            _diag_parts.append(f"Amp {_delta_amp_pct:+.0f}%")

                    _cmp_rows.append({
                        "Sensor": _lbl,
                        "vs Run": f"{_snap_label_short} ({_snap_ts})",
                        "Previous amp": f"{_prev_amp:.3f} {r['amp_unit']}",
                        "Current amp": f"{r['amp_at_op']:.3f} {r['amp_unit']}",
                        "Δ amp": (
                            f"{_delta_amp:+.3f} ({_delta_amp_pct:+.1f}%)"
                            if _delta_amp_pct is not None else "—"
                        ),
                        "Previous phase": f"{_prev_phase:.1f}°",
                        "Current phase": f"{r['phase_at_op']:.1f}°",
                        "Δ phase": f"{_delta_phase:+.1f}°",
                        "Diagnostic": " · ".join(_diag_parts) if _diag_parts else "—",
                    })

            if _cmp_rows:
                st.markdown("### Polar comparison — vs previous runs")
                _n_snaps = len(_snap_meta_by_id)
                if _n_snaps == 1:
                    _only = list(_snap_meta_by_id.values())[0]
                    st.caption(
                        f"Comparing against **{_only.get('corrida_label', '')}** "
                        f"from {_only.get('timestamp', '')[:10]}. A 1X phase "
                        f"shift >30° is a diagnostic symptom of a rotor "
                        f"balance change (API 684 / ISO 21940-12)."
                    )
                else:
                    st.caption(
                        f"Showing a comparison against **{_n_snaps} previous "
                        f"runs**. Each row is a (sensor × run) combination. "
                        f"A 1X phase shift >30° = symptom of a balance "
                        f"change (API 684)."
                    )
                _cmp_disp = pd.DataFrame(_cmp_rows)
                st.dataframe(_cmp_disp, width="stretch", hide_index=True)
        except Exception as _polar_cmp_e:
            st.caption(f"_(Polar comparison not available: {_polar_cmp_e})_")

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
