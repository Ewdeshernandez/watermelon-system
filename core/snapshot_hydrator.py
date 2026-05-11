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


def render_snapshot_loaded_banner() -> None:
    """Banner para mostrar al inicio de una página cuando se vino desde
    un snapshot histórico. Avisa al especialista que está viendo data
    archivada, no carga reciente."""
    info = st.session_state.get("_loaded_from_snapshot")
    if not info:
        return
    corrida = info.get("corrida_label", "") or "(sin label)"
    ts = info.get("timestamp", "")
    inst = info.get("instance_id", "")
    n = info.get("n_signals", 0)
    st.info(
        f"📌 **Mostrando snapshot histórico** — `{info.get('snapshot_id')}` · "
        f"{n} señales · activo `{inst}`\n\n"
        f"Corrida: **{corrida}** · Capturado: {ts}\n\n"
        f"_Para cargar nuevos CSVs, ve a **Load Data**._"
    )


__all__ = [
    "hydrate_waveform_snapshot",
    "consume_pending_snapshot_url",
    "render_snapshot_loaded_banner",
]
