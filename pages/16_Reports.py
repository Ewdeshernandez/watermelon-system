from __future__ import annotations

import inspect
import math
import uuid
from dataclasses import dataclass, asdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.auth import require_login, render_user_menu
from core.page_module_loader import (
    get_orbit_builder,
    get_spectrum_builder,
    get_trend_builder,
    get_waveform_builder,
)


# ============================================================
# Page config
# ============================================================

st.set_page_config(page_title="Watermelon System | Reports", layout="wide")
require_login()
render_user_menu()


# ============================================================
# Premium styling
# ============================================================

st.markdown(
    """
    <style>
        .wm-page-title {
            font-size: 2rem;
            font-weight: 700;
            color: #f5f7fb;
            margin-bottom: 0.25rem;
            letter-spacing: 0.2px;
        }
        .wm-page-subtitle {
            color: #9aa6b2;
            font-size: 0.98rem;
            margin-bottom: 1.25rem;
        }
        .wm-card {
            background: linear-gradient(180deg, rgba(18,24,34,0.96) 0%, rgba(12,17,25,0.96) 100%);
            border: 1px solid rgba(90,110,140,0.22);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 10px 28px rgba(0,0,0,0.18);
            margin-bottom: 1rem;
        }
        .wm-block-title {
            color: #f5f7fb;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .wm-block-subtitle {
            color: #95a2b1;
            font-size: 0.9rem;
            margin-bottom: 0.85rem;
        }
        .wm-kpi {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            min-height: 82px;
        }
        .wm-kpi-label {
            color: #8fa0b5;
            font-size: 0.83rem;
            margin-bottom: 0.2rem;
        }
        .wm-kpi-value {
            color: #ffffff;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .wm-section-title {
            color: #ffffff;
            font-weight: 700;
            font-size: 1.1rem;
            margin-top: 0.2rem;
            margin-bottom: 0.75rem;
        }
        .wm-muted {
            color: #93a1b3;
            font-size: 0.9rem;
        }
        .wm-divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
            margin: 0.85rem 0 1rem 0;
        }
        .wm-badge {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(41, 182, 246, 0.12);
            color: #7fd7ff;
            border: 1px solid rgba(41, 182, 246, 0.24);
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.35rem;
        }
        .wm-note {
            color: #b8c3cf;
            font-size: 0.9rem;
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data models
# ============================================================

@dataclass
class ReportBlock:
    block_id: str
    block_type: str
    signal_key: str
    title: str
    notes: str = ""
    order: int = 1


REPORT_BLOCK_TYPES = [
    "Tabular List",
    "Time Waveform",
    "Spectrum",
    "Orbit",
    "Trends",
]


# ============================================================
# State init
# ============================================================

if "report_blocks" not in st.session_state:
    st.session_state["report_blocks"] = []

if "report_meta" not in st.session_state:
    st.session_state["report_meta"] = {
        "report_title": "Technical Vibration Report",
        "client": "",
        "asset": "",
        "service_development": "",
        "recommendations": "",
    }


# ============================================================
# Utility functions
# ============================================================

def _safe_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    try:
        return np.asarray(value)
    except Exception:
        return np.array([])


def _first_present_attr(obj: Any, names: List[str], default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _signal_dict() -> Dict[str, Any]:
    raw = st.session_state.get("signals", {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {f"signal_{i+1}": item for i, item in enumerate(raw)}
    return {}


def _signal_display_name(key: str, sig: Any) -> str:
    file_name = _first_present_attr(sig, ["file_name", "filename", "name"], "")
    metadata = _first_present_attr(sig, ["metadata"], {}) or {}
    point_name = ""
    if isinstance(metadata, dict):
        point_name = metadata.get("point_name") or metadata.get("channel_name") or metadata.get("tag") or ""

    parts = [str(x).strip() for x in [file_name, point_name, key] if str(x).strip()]
    return " | ".join(parts[:3])


def _extract_signal_xy(sig: Any) -> Tuple[np.ndarray, np.ndarray]:
    t = _safe_to_numpy(_first_present_attr(sig, ["time", "t", "timestamp", "timestamps"]))
    x = _safe_to_numpy(_first_present_attr(sig, ["x", "y", "values", "amplitude", "data"]))
    if t.size == 0 and x.size > 0:
        t = np.arange(len(x), dtype=float)
    if x.size == 0 and t.size > 0:
        x = np.zeros_like(t, dtype=float)

    n = min(len(t), len(x))
    if n == 0:
        return np.array([]), np.array([])
    return t[:n], x[:n]


def _estimate_fs(time_array: np.ndarray) -> Optional[float]:
    if time_array.size < 2:
        return None
    dt = np.diff(time_array.astype(float))
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def _build_generic_record(sig: Any) -> SimpleNamespace:
    t, x = _extract_signal_xy(sig)
    metadata = _first_present_attr(sig, ["metadata"], {}) or {}
    file_name = _first_present_attr(sig, ["file_name", "filename", "name"], "")
    fs = _first_present_attr(sig, ["fs", "sample_rate", "sampling_rate"], None)
    if fs is None:
        fs = _estimate_fs(t)

    record = SimpleNamespace(
        time=t,
        x=x,
        metadata=metadata,
        file_name=file_name,
        fs=fs,
        sample_rate=fs,
        signal_name=metadata.get("point_name") if isinstance(metadata, dict) else "",
        units=metadata.get("units") if isinstance(metadata, dict) else "",
    )
    return record


def _call_builder_flex(builder, sig: Any, block_type: str):
    """
    Blindado:
    - intenta con el objeto señal original
    - intenta con un record normalizado
    - rellena kwargs comunes solo si el builder los pide
    """
    if builder is None:
        raise RuntimeError(f"No builder available for {block_type}")

    original = sig
    normalized = _build_generic_record(sig)
    candidates = [original, normalized]

    for candidate in candidates:
        try:
            sig_params = inspect.signature(builder).parameters
            kwargs = {}

            for name in sig_params:
                lname = name.lower()

                if lname in {"signal", "record", "signal_record", "data", "sig"}:
                    kwargs[name] = candidate
                elif lname in {"signals"}:
                    kwargs[name] = [candidate]
                elif lname in {"title"}:
                    kwargs[name] = _first_present_attr(candidate, ["file_name"], "") or block_type
                elif lname in {"channel_name", "signal_name", "name"}:
                    md = getattr(candidate, "metadata", {}) or {}
                    kwargs[name] = md.get("point_name") if isinstance(md, dict) else ""
                elif lname in {"fs", "sample_rate", "sampling_rate"}:
                    kwargs[name] = getattr(candidate, "fs", None) or getattr(candidate, "sample_rate", None)
                elif lname in {"time"}:
                    kwargs[name] = getattr(candidate, "time", np.array([]))
                elif lname in {"x", "values", "amplitude"}:
                    kwargs[name] = getattr(candidate, "x", np.array([]))
                elif lname in {"metadata"}:
                    kwargs[name] = getattr(candidate, "metadata", {}) or {}
                elif lname in {"show_controls", "show_toolbar", "editable"}:
                    kwargs[name] = False

            result = builder(**kwargs)
            if isinstance(result, go.Figure):
                return result

            if hasattr(result, "to_plotly_json"):
                return result
        except TypeError:
            continue
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(
        f"Could not build {block_type} with runtime-safe adapter. "
        f"Please align builder signature for report reuse."
    )


def _fallback_waveform(sig: Any) -> go.Figure:
    t, x = _extract_signal_xy(sig)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=t, y=x, mode="lines", name="Waveform"))
    fig.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(l=30, r=30, t=50, b=35),
        title="Waveform Preview",
        xaxis_title="Time",
        yaxis_title="Amplitude",
    )
    return fig


def _fallback_spectrum(sig: Any) -> go.Figure:
    t, x = _extract_signal_xy(sig)
    fs = _estimate_fs(t)
    if x.size < 4 or not fs:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=440, title="Spectrum Preview")
        return fig

    x = x.astype(float)
    x = x - np.mean(x)
    n = len(x)
    window = np.hanning(n)
    xf = np.fft.rfftfreq(n, d=1.0 / fs)
    yf = np.abs(np.fft.rfft(x * window))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=xf, y=yf, mode="lines", name="Spectrum"))
    fig.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(l=30, r=30, t=50, b=35),
        title="Spectrum Preview",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
    )
    return fig


def _fallback_orbit(sig: Any) -> go.Figure:
    t, x = _extract_signal_xy(sig)
    if x.size == 0:
        x = np.array([0.0])
    # fallback simple pseudo-orbit to avoid crash only when no real builder works
    shifted = np.roll(x, -1)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=shifted, mode="lines", name="Orbit"))
    fig.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(l=30, r=30, t=50, b=35),
        title="Orbit Preview",
        xaxis_title="X",
        yaxis_title="Y",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
    )
    return fig


def _fallback_trend(sig: Any) -> go.Figure:
    t, x = _extract_signal_xy(sig)
    if x.size == 0:
        x = np.array([])
    rolling = pd.Series(x).rolling(20, min_periods=1).mean().to_numpy() if x.size else np.array([])
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=t, y=rolling, mode="lines", name="Trend"))
    fig.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(l=30, r=30, t=50, b=35),
        title="Trend Preview",
        xaxis_title="Time",
        yaxis_title="Value",
    )
    return fig


def _build_block_figure(block_type: str, sig: Any) -> go.Figure:
    if block_type == "Time Waveform":
        builder = get_waveform_builder()
        if builder is not None:
            try:
                return _call_builder_flex(builder, sig, block_type)
            except Exception:
                pass
        return _fallback_waveform(sig)

    if block_type == "Spectrum":
        builder = get_spectrum_builder()
        if builder is not None:
            try:
                return _call_builder_flex(builder, sig, block_type)
            except Exception:
                pass
        return _fallback_spectrum(sig)

    if block_type == "Orbit":
        builder = get_orbit_builder()
        if builder is not None:
            try:
                return _call_builder_flex(builder, sig, block_type)
            except Exception:
                pass
        return _fallback_orbit(sig)

    if block_type == "Trends":
        builder = get_trend_builder()
        if builder is not None:
            try:
                return _call_builder_flex(builder, sig, block_type)
            except Exception:
                pass
        return _fallback_trend(sig)

    raise ValueError(f"Unsupported block type: {block_type}")


def _build_tabular_dataframe(sig: Any) -> pd.DataFrame:
    t, x = _extract_signal_xy(sig)
    metadata = _first_present_attr(sig, ["metadata"], {}) or {}
    n = min(len(t), len(x))
    df = pd.DataFrame(
        {
            "index": np.arange(n),
            "time": t[:n],
            "value": x[:n],
        }
    )

    for k, v in list(metadata.items())[:8] if isinstance(metadata, dict) else []:
        df.attrs[k] = v

    return df


def _signal_metrics(sig: Any) -> Dict[str, str]:
    t, x = _extract_signal_xy(sig)
    fs = _estimate_fs(t)
    if x.size == 0:
        return {
            "Samples": "0",
            "Fs": "-",
            "RMS": "-",
            "Pk-Pk": "-",
        }

    xrms = float(np.sqrt(np.mean(np.square(x.astype(float))))) if x.size else 0.0
    xpkpk = float(np.max(x) - np.min(x)) if x.size else 0.0
    return {
        "Samples": f"{len(x):,}",
        "Fs": f"{fs:,.2f} Hz" if fs else "-",
        "RMS": f"{xrms:,.4f}",
        "Pk-Pk": f"{xpkpk:,.4f}",
    }


def _sorted_blocks() -> List[ReportBlock]:
    blocks = [ReportBlock(**b) if isinstance(b, dict) else b for b in st.session_state["report_blocks"]]
    return sorted(blocks, key=lambda b: (b.order, b.title.lower(), b.block_id))


def _persist_blocks(blocks: List[ReportBlock]):
    st.session_state["report_blocks"] = [asdict(b) for b in blocks]


def _add_block(block_type: str, signal_key: str):
    signals = _signal_dict()
    sig = signals.get(signal_key)
    title_base = block_type
    if sig is not None:
        title_base = f"{block_type} | {_signal_display_name(signal_key, sig)}"

    blocks = _sorted_blocks()
    new_block = ReportBlock(
        block_id=str(uuid.uuid4())[:8],
        block_type=block_type,
        signal_key=signal_key,
        title=title_base,
        notes="",
        order=len(blocks) + 1,
    )
    blocks.append(new_block)
    _persist_blocks(blocks)


def _remove_block(block_id: str):
    blocks = [b for b in _sorted_blocks() if b.block_id != block_id]
    for i, b in enumerate(blocks, start=1):
        b.order = i
    _persist_blocks(blocks)


def _move_block(block_id: str, direction: int):
    blocks = _sorted_blocks()
    idx = next((i for i, b in enumerate(blocks) if b.block_id == block_id), None)
    if idx is None:
        return
    new_idx = idx + direction
    if new_idx < 0 or new_idx >= len(blocks):
        return
    blocks[idx], blocks[new_idx] = blocks[new_idx], blocks[idx]
    for i, b in enumerate(blocks, start=1):
        b.order = i
    _persist_blocks(blocks)


# ============================================================
# Header
# ============================================================

st.markdown('<div class="wm-page-title">Reports</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="wm-page-subtitle">Construcción de entregables técnicos premium usando los motores visuales reales del sistema.</div>',
    unsafe_allow_html=True,
)

signals = _signal_dict()

if not signals:
    st.warning("No hay señales cargadas en st.session_state['signals']. Primero carga datos en Load Data.")
    st.stop()


signal_options = {key: _signal_display_name(key, sig) for key, sig in signals.items()}


# ============================================================
# Top summary
# ============================================================

sample_sig = next(iter(signals.values()))
sample_metrics = _signal_metrics(sample_sig)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-label">Signals Loaded</div>
            <div class="wm-kpi-value">{len(signals):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-label">Report Blocks</div>
            <div class="wm-kpi-value">{len(_sorted_blocks()):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-label">Sample RMS</div>
            <div class="wm-kpi-value">{sample_metrics["RMS"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="wm-kpi">
            <div class="wm-kpi-label">Sample Rate</div>
            <div class="wm-kpi-value">{sample_metrics["Fs"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)


# ============================================================
# Metadata section
# ============================================================

st.markdown('<div class="wm-section-title">Report Metadata</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.session_state["report_meta"]["report_title"] = st.text_input(
        "Report Title",
        value=st.session_state["report_meta"]["report_title"],
    )
with m2:
    st.session_state["report_meta"]["client"] = st.text_input(
        "Client",
        value=st.session_state["report_meta"]["client"],
    )
with m3:
    st.session_state["report_meta"]["asset"] = st.text_input(
        "Asset / Machine",
        value=st.session_state["report_meta"]["asset"],
    )

t1, t2 = st.columns(2)
with t1:
    st.session_state["report_meta"]["service_development"] = st.text_area(
        "Desarrollo del servicio",
        value=st.session_state["report_meta"]["service_development"],
        height=180,
        placeholder="Describe la intervención, alcance, condiciones observadas, metodología y hallazgos clave.",
    )
with t2:
    st.session_state["report_meta"]["recommendations"] = st.text_area(
        "Recomendaciones",
        value=st.session_state["report_meta"]["recommendations"],
        height=180,
        placeholder="Redacta acciones recomendadas, criticidad, seguimiento y próximos pasos.",
    )

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)


# ============================================================
# Block builder
# ============================================================

st.markdown('<div class="wm-section-title">Add Report Blocks</div>', unsafe_allow_html=True)

a1, a2, a3 = st.columns([1.1, 1.8, 0.8])

with a1:
    new_block_type = st.selectbox("Block Type", REPORT_BLOCK_TYPES, key="reports_new_block_type")
with a2:
    selected_signal_key = st.selectbox(
        "Source Signal",
        options=list(signal_options.keys()),
        format_func=lambda k: signal_options[k],
        key="reports_new_signal_key",
    )
with a3:
    st.write("")
    st.write("")
    if st.button("Add Block", use_container_width=True):
        _add_block(new_block_type, selected_signal_key)
        st.rerun()


# ============================================================
# Block list / order
# ============================================================

blocks = _sorted_blocks()

st.markdown('<div class="wm-section-title">Report Structure</div>', unsafe_allow_html=True)

if not blocks:
    st.info("Agrega bloques para empezar a construir el reporte.")
else:
    for i, block in enumerate(blocks, start=1):
        sig = signals.get(block.signal_key)

        st.markdown('<div class="wm-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="wm-block-title">{i:02d}. {block.title}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="wm-block-subtitle"><span class="wm-badge">{block.block_type}</span>Source: {signal_options.get(block.signal_key, block.signal_key)}</div>',
            unsafe_allow_html=True,
        )

        b1, b2, b3, b4 = st.columns([2.0, 1.2, 0.8, 0.8])

        with b1:
            new_title = st.text_input(
                "Block Title",
                value=block.title,
                key=f"title_{block.block_id}",
            )
            block.title = new_title

        with b2:
            new_signal_key = st.selectbox(
                "Source Signal",
                options=list(signal_options.keys()),
                index=list(signal_options.keys()).index(block.signal_key) if block.signal_key in signal_options else 0,
                format_func=lambda k: signal_options[k],
                key=f"sig_{block.block_id}",
            )
            block.signal_key = new_signal_key
            sig = signals.get(block.signal_key)

        with b3:
            st.write("")
            st.write("")
            up_disabled = i == 1
            if st.button("↑ Up", key=f"up_{block.block_id}", use_container_width=True, disabled=up_disabled):
                _move_block(block.block_id, -1)
                st.rerun()

        with b4:
            st.write("")
            st.write("")
            down_disabled = i == len(blocks)
            if st.button("↓ Down", key=f"down_{block.block_id}", use_container_width=True, disabled=down_disabled):
                _move_block(block.block_id, +1)
                st.rerun()

        notes = st.text_area(
            "Block Notes",
            value=block.notes,
            key=f"notes_{block.block_id}",
            height=90,
            placeholder="Comentario técnico específico para este bloque.",
        )
        block.notes = notes

        r1, r2 = st.columns([5, 1])
        with r2:
            if st.button("Remove", key=f"remove_{block.block_id}", use_container_width=True):
                _remove_block(block.block_id)
                st.rerun()

        if sig is None:
            st.error("La señal asociada a este bloque ya no existe en sesión.")
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        if block.block_type == "Tabular List":
            df = _build_tabular_dataframe(sig)
            metrics = _signal_metrics(sig)

            k1, k2, k3, k4 = st.columns(4)
            for col, (label, value) in zip([k1, k2, k3, k4], metrics.items()):
                with col:
                    st.markdown(
                        f"""
                        <div class="wm-kpi">
                            <div class="wm-kpi-label">{label}</div>
                            <div class="wm-kpi-value">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.dataframe(df, use_container_width=True, height=320)

            metadata = _first_present_attr(sig, ["metadata"], {}) or {}
            if isinstance(metadata, dict) and metadata:
                st.markdown('<div class="wm-muted">Metadata</div>', unsafe_allow_html=True)
                st.json(metadata, expanded=False)

        else:
            fig = _build_block_figure(block.block_type, sig)
            st.plotly_chart(fig, use_container_width=True, theme=None)

        st.markdown("</div>", unsafe_allow_html=True)

    _persist_blocks(blocks)


# ============================================================
# Final preview
# ============================================================

st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="wm-section-title">Report Assembly Preview</div>', unsafe_allow_html=True)

meta = st.session_state["report_meta"]

preview_left, preview_right = st.columns([1.2, 1.8])

with preview_left:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="wm-block-title">{meta["report_title"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="wm-note">
            <strong>Client:</strong> {meta["client"] or "-"}<br>
            <strong>Asset:</strong> {meta["asset"] or "-"}<br>
            <strong>Blocks:</strong> {len(_sorted_blocks())}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Desarrollo del servicio</div>', unsafe_allow_html=True)
    st.write(meta["service_development"] or "—")
    st.markdown('<div class="wm-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-subtitle">Recomendaciones</div>', unsafe_allow_html=True)
    st.write(meta["recommendations"] or "—")
    st.markdown("</div>", unsafe_allow_html=True)

with preview_right:
    st.markdown('<div class="wm-card">', unsafe_allow_html=True)
    st.markdown('<div class="wm-block-title">Ordered Block Summary</div>', unsafe_allow_html=True)
    if not _sorted_blocks():
        st.markdown('<div class="wm-note">Todavía no hay bloques en el reporte.</div>', unsafe_allow_html=True)
    else:
        for idx, block in enumerate(_sorted_blocks(), start=1):
            st.markdown(
                f"""
                <div class="wm-note">
                    <span class="wm-badge">{idx:02d}</span>
                    <strong>{block.block_type}</strong> — {block.title}
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Footer notes
# ============================================================

st.caption(
    "Arquitectura actual: Reports reutiliza motores visuales existentes mediante runtime bridge seguro desde pages/, "
    "sin imports inválidos por nombre de archivo numérico."
)