"""
core.snapshot_hydrator
======================

Hidrata snapshots históricos al `st.session_state` para que los módulos
de análisis (Time Waveforms, Spectrum, Orbit, Tabular) los muestren con
su UX completa — sin re-implementar el render (Ciclo 23.87).

Approach: cada módulo de análisis lee de `st.session_state["signals"]`
(o similar) con un formato específico. Acá construimos signal_obj-like
objects desde el payload de history_storage y los inyectamos.

Flow del cliente:
  1. Live Monitoring → card "Forma de onda" → click "Abrir en Time Waveforms"
  2. Redirige a `/Time_Waveforms?snapshot=waveform_20260511_222442&instance=tes1`
  3. Time Waveforms detecta URL params al iniciar
  4. Llama `hydrate_waveform_snapshot(instance_id, snapshot_id)`
  5. Snapshot queda en session_state["signals"]
  6. El módulo renderiza normal — con header completo, cursores,
     métricas, watermark — IDÉNTICO al flow Load Data → análisis.

API:
    hydrate_waveform_snapshot(instance_id, snapshot_id) → bool
    consume_pending_snapshot_url() → tuple[str, str] | None
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st


# =============================================================
# WAVEFORM HYDRATION
# =============================================================

def hydrate_waveform_snapshot(instance_id: str, snapshot_id: str) -> bool:
    """Carga snapshot waveform y popula st.session_state["signals"].

    El formato esperado por pages/02_Time_Waveforms.py:
        st.session_state["signals"] = {
            key: signal_obj con .time, .x, .metadata, .file_name
        }

    Construimos signal_obj con SimpleNamespace para compatibilidad
    con `signal_object_to_record()` del módulo original.

    Returns:
        True si se hidrató con éxito, False si falló.
    """
    try:
        from core.waveform_history import load_waveform_snapshot
    except Exception:
        return False

    payload = load_waveform_snapshot(instance_id, snapshot_id)
    if payload is None:
        return False

    sensors = payload.get("sensors", [])
    if not sensors:
        return False

    operating_rpm = payload.get("operating_speed_rpm")
    corrida_label = payload.get("corrida_label", "")
    snapshot_timestamp = payload.get("timestamp", "")

    new_signals: Dict[str, Any] = {}
    for s in sensors:
        sensor_label = s.get("sensor_label", "") or "sensor"
        csv_file = s.get("csv_file", "") or f"{sensor_label}.csv"
        time_arr = np.array(s.get("time") or [], dtype=float)
        value_arr = np.array(s.get("values") or [], dtype=float)
        if time_arr.size < 2 or value_arr.size < 2:
            continue

        unit = s.get("unit", "") or ""
        fs = s.get("sampling_rate_hz")
        csv_ts = s.get("csv_timestamp", "") or snapshot_timestamp

        # Metadata en el formato que find_meta espera
        metadata = {
            "Machine":       payload.get("instance_id", instance_id),
            "Point":         sensor_label,
            "Variable":      "Waveform",
            "Timestamp":     csv_ts,
            "Y Axis Unit":   unit,
            "Amplitude Unit": unit,
            "RPM":           operating_rpm if operating_rpm else None,
            "Sampling Rate": fs,
            "_source":       "snapshot",  # marker para debug
            "_snapshot_id":  snapshot_id,
            "_corrida_label": corrida_label,
        }

        # signal_obj-like con SimpleNamespace
        signal_obj = SimpleNamespace(
            time=time_arr,
            x=value_arr,
            file_name=csv_file,
            metadata=metadata,
        )

        # Key único — el módulo usa esto como signal_id (signals.{key})
        key = f"snapshot_{sensor_label}".replace(" ", "_").replace("/", "_")
        new_signals[key] = signal_obj

    if not new_signals:
        return False

    # Override session_state["signals"] — el caller asumió esto al venir
    # con URL param. Para evitar overrides accidentales, marcamos la
    # source en session_state también.
    st.session_state["signals"] = new_signals
    st.session_state["_loaded_from_snapshot"] = {
        "snapshot_id": snapshot_id,
        "instance_id": instance_id,
        "corrida_label": corrida_label,
        "timestamp": snapshot_timestamp,
        "n_signals": len(new_signals),
    }
    return True


# =============================================================
# URL PARAM HELPER
# =============================================================

def consume_pending_snapshot_url() -> Optional[Tuple[str, str]]:
    """Detecta `?snapshot=X&instance=Y` en URL.

    Returns:
        (instance_id, snapshot_id) si están presentes, None si no.
    """
    try:
        params = st.query_params
    except Exception:
        return None

    snap = params.get("snapshot")
    inst = params.get("instance")
    if not snap or not inst:
        return None
    return (str(inst), str(snap))


def _get_watermelon_logo_b64() -> str:
    """Carga el logo Watermelon como base64 para embedding inline."""
    try:
        from pathlib import Path
        import base64
        logo_path = Path(__file__).resolve().parents[1] / "assets" / "watermelon_logo.png"
        if logo_path.exists():
            return base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except Exception:
        pass
    return ""


def _resolve_client_label(instance_id: str) -> str:
    """Intenta obtener el cliente asociado a la instancia."""
    try:
        from core.instance_state import get_instance
        inst = get_instance(instance_id)
        if inst is not None:
            client = getattr(inst, "client", "") or ""
            site = getattr(inst, "site", "") or ""
            if client and site:
                return f"{client} · {site}"
            return client or site
    except Exception:
        pass
    return ""


def render_snapshot_loaded_banner() -> None:
    """Header pro cuando se vino desde un snapshot histórico (Ciclo 23.95).

    Layout:
      ┌────────────────────────────────────────────────────────────┐
      │ [← Volver]                          TES1 · Ecopetrol       │
      │                                                             │
      │ [Logo]  Análisis de Formas de Onda                         │
      │         11 May 2026 · 22:24 · 8 sensores                   │
      │                                                             │
      │ ───────────────────────────────────────────────────────── │
      └────────────────────────────────────────────────────────────┘
    """
    info = st.session_state.get("_loaded_from_snapshot")
    if not info:
        return
    inst = info.get("instance_id", "") or ""
    ts_raw = info.get("timestamp", "") or ""
    n_signals = info.get("n_signals", 0)

    # Fecha amigable
    fecha = ts_raw
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        fecha = f"{dt.day} {meses[dt.month - 1]} {dt.year} · {dt.hour:02d}:{dt.minute:02d}"
    except Exception:
        pass

    client_label = _resolve_client_label(inst)
    logo_b64 = _get_watermelon_logo_b64()

    # ── Row 1: Botón Volver (izq) + Tag activo / cliente (der) ──
    row1_left, row1_right = st.columns([2, 4])
    with row1_left:
        st.markdown('<div class="wm-return-btn-wrap">', unsafe_allow_html=True)
        if st.button(
            "← Volver a Live Monitoring",
            key="_wm_return_live_monitoring",
            use_container_width=False,
        ):
            st.session_state.pop("_loaded_from_snapshot", None)
            st.session_state.pop("signals", None)
            try:
                st.switch_page("pages/02_Live_Monitoring.py")
            except Exception:
                st.error("No se pudo volver. Refrescá la página.")
        st.markdown('</div>', unsafe_allow_html=True)
    with row1_right:
        client_html = (
            f"<span style='color:#94a3b8;'>·</span>"
            f"<span style='color:#475569;font-size:12px;font-weight:600;"
            f"letter-spacing:0.04em;margin-left:8px;'>{client_label}</span>"
            if client_label else ""
        )
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:flex-end;"
            f"gap:10px;padding-top:10px;'>"
            f"<span style='font-size:13px;font-weight:800;color:#0f172a;"
            f"letter-spacing:0.05em;text-transform:uppercase;"
            f"background:linear-gradient(135deg,#1e40af 0%,#3b82f6 100%);"
            f"color:white;padding:5px 12px;border-radius:8px;"
            f"box-shadow:0 2px 6px rgba(30,64,175,0.20);'>{inst.upper()}</span>"
            f"{client_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Row 2: Logo + Título + Meta ──
    logo_html = (
        f"<img src='data:image/png;base64,{logo_b64}' "
        f"style='width:56px;height:56px;object-fit:contain;"
        f"margin-right:18px;flex-shrink:0;'/>"
        if logo_b64 else
        '<div style="width:56px;height:56px;margin-right:18px;flex-shrink:0;'
        'border-radius:14px;background:linear-gradient(135deg,#1e40af,#3b82f6);"></div>'
    )

    st.markdown(
        f"<div style='display:flex;align-items:center;margin:10px 0 4px 0;'>"
        f"{logo_html}"
        f"<div>"
        f"<div style='font-size:26px;font-weight:800;color:#0f172a;"
        f"letter-spacing:-0.015em;line-height:1.1;'>"
        f"Análisis de Formas de Onda</div>"
        f"<div style='font-size:13px;color:#64748b;font-weight:600;"
        f"margin-top:4px;letter-spacing:0.01em;'>"
        f"{fecha}"
        f"<span style='color:#cbd5e1;margin:0 6px;'>·</span>"
        f"{n_signals} sensores"
        f"<span style='color:#cbd5e1;margin:0 6px;'>·</span>"
        f"<span style='color:#16a34a;'>✓ Operación normal</span>"
        f"</div>"
        f"</div>"
        f"</div>"
        f"<hr style='border:0;height:1px;"
        f"background:linear-gradient(90deg,transparent,#cbd5e1 20%,#cbd5e1 80%,transparent);"
        f"margin:14px 0 8px 0;'/>",
        unsafe_allow_html=True,
    )


def render_synchronized_cursors_controls() -> None:
    """Sliders A/B sincronizados que afectan TODAS las waveforms (Ciclo 23.95).

    Sets st.session_state.wm_cursor_a y wm_cursor_b, que son leídos por
    build_waveform_figure() para cada sensor. Resultado: mover el slider
    A en el banner mueve el cursor A en TODAS las gráficas al tiempo.

    Solo se renderiza si:
      - estamos en modo cliente (_loaded_from_snapshot está set)
      - hay signals cargadas en session_state
    """
    if not st.session_state.get("_loaded_from_snapshot"):
        return
    signals = st.session_state.get("signals", {})
    if not signals:
        return

    # Calcular t_min y t_max del primer signal (asumimos rango uniforme)
    try:
        import numpy as np
        first = next(iter(signals.values()))
        time_arr = getattr(first, "time", None)
        if time_arr is None or len(time_arr) < 2:
            return
        t_min = float(np.min(time_arr))
        t_max = float(np.max(time_arr))
        if t_max <= t_min:
            return
    except Exception:
        return

    # Init defaults si no existen
    if "wm_cursor_a" not in st.session_state:
        st.session_state.wm_cursor_a = t_min
    if "wm_cursor_b" not in st.session_state:
        st.session_state.wm_cursor_b = t_min + (t_max - t_min) * 0.1
    # Clamp a rango actual (por si las signals cambian)
    st.session_state.wm_cursor_a = min(max(float(st.session_state.wm_cursor_a), t_min), t_max)
    st.session_state.wm_cursor_b = min(max(float(st.session_state.wm_cursor_b), t_min), t_max)

    # En ms para mejor UX (más intuitivo para el cliente que segundos)
    t_min_ms = t_min * 1000.0
    t_max_ms = t_max * 1000.0
    delta_total_ms = t_max_ms - t_min_ms
    step_ms = max(delta_total_ms / 1000.0, 0.001)

    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;"
        "margin:4px 0 12px 0;font-size:11px;font-weight:700;"
        "color:#64748b;letter-spacing:0.08em;text-transform:uppercase;'>"
        "⊕ Cursores sincronizados</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b, col_delta = st.columns([3, 3, 1.5])

    with col_a:
        cursor_a_ms = st.slider(
            "Cursor A (ms)",
            min_value=float(t_min_ms),
            max_value=float(t_max_ms),
            value=float(st.session_state.wm_cursor_a * 1000.0),
            step=float(step_ms),
            format="%.3f ms",
            key="_wm_sync_cursor_a_slider",
            label_visibility="visible",
        )
        st.session_state.wm_cursor_a = cursor_a_ms / 1000.0

    with col_b:
        cursor_b_ms = st.slider(
            "Cursor B (ms)",
            min_value=float(t_min_ms),
            max_value=float(t_max_ms),
            value=float(st.session_state.wm_cursor_b * 1000.0),
            step=float(step_ms),
            format="%.3f ms",
            key="_wm_sync_cursor_b_slider",
            label_visibility="visible",
        )
        st.session_state.wm_cursor_b = cursor_b_ms / 1000.0

    with col_delta:
        delta = abs(cursor_b_ms - cursor_a_ms)
        delta_freq = 1000.0 / delta if delta > 0.001 else 0
        st.markdown(
            f"<div style='padding-top:30px;'>"
            f"<div style='font-size:10px;color:#94a3b8;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.08em;'>Δ B-A</div>"
            f"<div style='font-size:14px;color:#0f172a;font-weight:800;"
            f"font-family:ui-monospace,monospace;'>"
            f"{delta:.2f} ms</div>"
            f"<div style='font-size:10px;color:#64748b;"
            f"font-family:ui-monospace,monospace;'>"
            f"{delta_freq:.1f} Hz</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


__all__ = [
    "hydrate_waveform_snapshot",
    "consume_pending_snapshot_url",
    "render_snapshot_loaded_banner",
    "render_synchronized_cursors_controls",
]
