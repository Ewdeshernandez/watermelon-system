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
# SPECTRUM HYDRATION (Ciclo 23.107)
# =============================================================

def hydrate_spectrum_snapshot(instance_id: str, snapshot_id: str) -> bool:
    """Hidrata `signals` para Spectrum desde el waveform pareado.

    Spectrum recibe time-domain en `st.session_state["signals"]` y computa
    FFT internamente. El snapshot de spectrum solo guarda freqs/amps ya
    procesados — no sirve directo. Truco: usamos `corrida_label` como join.
    Cada Load Data guarda waveform + spectrum + orbit + tabular con la
    misma etiqueta, así que basta con encontrar el waveform_snapshot que
    comparte `corrida_label` con este spectrum_snapshot e hidratar desde ese.
    """
    try:
        from core.spectrum_history import load_spectrum_snapshot, list_spectrum_snapshots
        from core.waveform_history import list_waveform_snapshots
    except Exception:
        return False

    # 1) Cargar metadata del spectrum snapshot para sacar corrida_label
    spec_payload = load_spectrum_snapshot(instance_id, snapshot_id)
    if not spec_payload:
        return False
    target_label = (spec_payload.get("corrida_label") or "").strip()

    # 2) Buscar waveform snapshot con el mismo corrida_label
    try:
        wf_snaps = list_waveform_snapshots(instance_id) or []
    except Exception:
        wf_snaps = []

    matching_wf_id = None
    for ws in wf_snaps:
        if (ws.get("corrida_label") or "").strip() == target_label and target_label:
            matching_wf_id = ws.get("snapshot_id")
            break

    if not matching_wf_id:
        # Sin pareja waveform → no podemos render Spectrum completo.
        # Marcamos session_state con la info del spectrum directamente,
        # y el módulo Spectrum mostrará el preview simple (peaks + plot).
        st.session_state["_loaded_from_snapshot"] = {
            "snapshot_id": snapshot_id,
            "instance_id": instance_id,
            "corrida_label": target_label,
            "timestamp": spec_payload.get("timestamp", ""),
            "n_signals": len(spec_payload.get("sensors", [])),
            "module_title": "Análisis Espectral",
            "spectrum_payload": spec_payload,  # fallback rendering
            "no_waveform_pair": True,
        }
        return True

    # 3) Hidratar via el waveform helper (re-usa toda la lógica probada)
    ok = hydrate_waveform_snapshot(instance_id, matching_wf_id)
    if not ok:
        return False

    # 4) Override el module_title para que el banner diga "Espectral"
    info = st.session_state.get("_loaded_from_snapshot", {})
    info["module_title"] = "Análisis Espectral"
    info["spectrum_snapshot_id"] = snapshot_id
    st.session_state["_loaded_from_snapshot"] = info
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
    # Ciclo 23.107 — título dinámico según el módulo activo. Si el hydrator
    # no lo seteó (default), usamos "Análisis de Formas de Onda" por bw-compat.
    module_title = (info.get("module_title") or "Análisis de Formas de Onda").strip()

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

    # ════════════════════════════════════════════════════════════════════
    # Ciclo 23.106 — HERO CARD clase mundial estilo System1/AMS.
    # Una sola card blanca elevada con:
    #   * Top bar: botón Volver (gradient azul royal, via type="primary")
    #               + breadcrumb del cliente + tag del activo
    #   * Hero: logo grande + título + subtitle con fecha
    #   * Meta strip: 3 metric cards con líneas divisorias (sensores / medición / estado)
    # ════════════════════════════════════════════════════════════════════

    # 1) Inyectar TODA la CSS de la hero card (una sola vez)
    st.markdown(
        """
        <style>
        /* ── HERO CARD container ── */
        .wm-hero {
            background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 0;
            margin: 4px 0 18px 0;
            box-shadow:
                0 1px 2px rgba(15,23,42,0.04),
                0 4px 14px rgba(15,23,42,0.06),
                0 0 0 1px rgba(255,255,255,0.6) inset;
            overflow: hidden;
        }
        .wm-hero-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 22px 8px 22px;
            gap: 18px;
            min-height: 56px;
        }
        .wm-hero-breadcrumb {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .wm-hero-tag {
            display: inline-flex;
            align-items: center;
            font-size: 12px;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            padding: 5px 12px;
            border-radius: 7px;
            box-shadow:
                0 1px 2px rgba(30,64,175,0.25),
                0 3px 8px rgba(30,64,175,0.18);
        }
        .wm-hero-dot {
            color: #cbd5e1;
            font-weight: 700;
        }
        .wm-hero-client {
            font-size: 12px;
            font-weight: 600;
            color: #475569;
            letter-spacing: 0.02em;
        }
        /* ── HERO main (logo + título) ── */
        .wm-hero-main {
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 4px 22px 18px 22px;
        }
        .wm-hero-logo {
            width: 52px;
            height: 52px;
            flex-shrink: 0;
            border-radius: 12px;
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow:
                0 1px 2px rgba(30,64,175,0.2),
                0 4px 10px rgba(30,64,175,0.15);
        }
        .wm-hero-logo img {
            width: 38px;
            height: 38px;
            object-fit: contain;
            filter: drop-shadow(0 1px 2px rgba(0,0,0,0.10));
        }
        .wm-hero-titles {
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        .wm-hero-title {
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }
        .wm-hero-subtitle {
            font-size: 13px;
            color: #64748b;
            font-weight: 600;
            margin-top: 2px;
            letter-spacing: 0.005em;
        }
        /* ── META strip (3 metric cards) ── */
        .wm-hero-meta {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0;
            border-top: 1px solid #e2e8f0;
            background: #f8fafc;
        }
        .wm-hero-meta-item {
            display: flex;
            flex-direction: column;
            padding: 12px 22px;
            border-right: 1px solid #e2e8f0;
        }
        .wm-hero-meta-item:last-child { border-right: 0; }
        .wm-hero-meta-label {
            font-size: 10px;
            font-weight: 800;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.10em;
        }
        .wm-hero-meta-value {
            font-size: 15px;
            font-weight: 800;
            color: #0f172a;
            margin-top: 3px;
            letter-spacing: -0.01em;
        }
        .wm-hero-meta-value.is-green { color: #15803d; }
        .wm-hero-meta-value .dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #16a34a;
            margin-right: 6px;
            box-shadow: 0 0 0 3px rgba(22,163,74,0.18);
            vertical-align: middle;
            position: relative;
            top: -1px;
        }

        /* ── BOTÓN Volver (Streamlit st.button con type="primary") ──
           Selectores múltiples para máxima compatibilidad: */
        #wm-return-btn-host button,
        #wm-return-btn-host + div button,
        #wm-return-btn-host ~ div [data-testid="stButton"] button,
        div[data-testid="stButton"]:has(button p:where(.wm-return-btn-text)) button,
        button[data-testid="baseButton-primary"]:has(p.wm-return-btn-text),
        button[kind="primary"]:has(p.wm-return-btn-text) {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 9px !important;
            font-weight: 700 !important;
            font-size: 12.5px !important;
            padding: 8px 18px !important;
            min-height: 36px !important;
            height: 36px !important;
            line-height: 1 !important;
            white-space: nowrap !important;
            box-shadow:
                0 1px 2px rgba(30,64,175,0.30),
                0 4px 12px rgba(30,64,175,0.20),
                inset 0 1px 0 rgba(255,255,255,0.22) !important;
            transition: all 0.2s cubic-bezier(.4,0,.2,1) !important;
        }
        #wm-return-btn-host ~ div [data-testid="stButton"] button p,
        #wm-return-btn-host ~ div [data-testid="stButton"] button span {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        #wm-return-btn-host ~ div [data-testid="stButton"] button:hover {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%) !important;
            box-shadow:
                0 6px 16px rgba(30,64,175,0.34),
                0 0 0 4px rgba(59,130,246,0.20),
                inset 0 1px 0 rgba(255,255,255,0.28) !important;
            transform: translateY(-1px) !important;
        }
        #wm-return-btn-host ~ div [data-testid="stButton"] button:active {
            transform: translateY(0) !important;
        }
        /* El span marker no debe consumir espacio */
        #wm-return-btn-host { display: block; height: 0; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 2) Renderizar la hero card visualmente: necesitamos meter el botón
    #    DENTRO de la card. Truco: dividimos el render en columnas.
    #    Top bar = col_btn (botón Streamlit) + col_meta (HTML breadcrumb+tag)
    #    luego HTML para hero-main + meta strip.

    col_btn, col_meta = st.columns([2, 5])
    with col_btn:
        # Span host para que el sibling selector de CSS enganche al botón
        st.markdown('<span id="wm-return-btn-host"></span>', unsafe_allow_html=True)
        if st.button(
            "← Volver a Live Monitoring",
            key="_wm_return_live_monitoring",
            type="primary",
        ):
            st.session_state.pop("_loaded_from_snapshot", None)
            st.session_state.pop("signals", None)
            try:
                st.switch_page("pages/02_Live_Monitoring.py")
            except Exception:
                st.error("No se pudo volver. Refrescá la página.")

    with col_meta:
        client_html = (
            f"<span class='wm-hero-dot'>·</span>"
            f"<span class='wm-hero-client'>{client_label}</span>"
            if client_label else ""
        )
        st.markdown(
            f"<div class='wm-hero-breadcrumb' style='justify-content:flex-end; padding-top:6px;'>"
            f"<span class='wm-hero-tag'>{inst.upper()}</span>"
            f"{client_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # 3) Hero main (logo + título) + meta strip — todo HTML en un solo bloque
    logo_inner = (
        f"<img src='data:image/png;base64,{logo_b64}' alt='Watermelon System'/>"
        if logo_b64 else
        '<div style="width:38px;height:38px;background:white;border-radius:9px;opacity:0.9;"></div>'
    )

    st.markdown(
        f"""
        <div class='wm-hero'>
            <div class='wm-hero-main' style='padding-top:14px;'>
                <div class='wm-hero-logo'>{logo_inner}</div>
                <div class='wm-hero-titles'>
                    <div class='wm-hero-title'>{module_title}</div>
                    <div class='wm-hero-subtitle'>{fecha}</div>
                </div>
            </div>
            <div class='wm-hero-meta'>
                <div class='wm-hero-meta-item'>
                    <div class='wm-hero-meta-label'>Sensores</div>
                    <div class='wm-hero-meta-value'>{n_signals} activos</div>
                </div>
                <div class='wm-hero-meta-item'>
                    <div class='wm-hero-meta-label'>Medición</div>
                    <div class='wm-hero-meta-value'>Snapshot histórico</div>
                </div>
                <div class='wm-hero-meta-item'>
                    <div class='wm-hero-meta-label'>Estado</div>
                    <div class='wm-hero-meta-value is-green'><span class='dot'></span>Operación normal</div>
                </div>
            </div>
        </div>
        """,
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
