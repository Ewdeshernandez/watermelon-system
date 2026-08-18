"""
core/remote_monitoring — Módulo "Remote Monitoring" (Live rotordynamics)
========================================================================

Adquisición CONTINUA (streaming) de vibración para monitoreo en vivo,
independiente del módulo `core/modal` (captura por sesión → TDMS modal).

  acq_backend (modal)  → captura N s, escribe UN TDMS, termina.
  remote_monitoring    → lee bloques sin fin, buffer rodante en RAM,
                         materializa ventanas como Signal → reusa TODOS
                         los motores de gráficos existentes.

Despliegue: ACQ Agent headless en PC Windows de sitio (ÚNICO que toca el
NI) → store local offline → sync Supabase cuando hay red → Streamlit
cliente (Mac/navegador) que lee el store, nunca el hardware.

No importa streamlit ni nidaqmx a nivel de módulo:
  · streamlit → capa UI (core/remote_monitoring/ui.py, cargado por la page).
  · nidaqmx   → lazy dentro de NIStreamSource (Windows/Linux con driver).

Piezas:
  stream_source     — StreamConfig + StreamSource + SimulatedStreamSource
  ni_stream_source  — NIStreamSource (nidaqmx continuo, hardware de campo)
  ring_buffer       — RingBuffer circular multicanal
  keyphasor         — rpm + vector 1X (referencia de fase)
  materialize       — ventana → List[Signal]
  agent             — AcqAgent (fuente → buffer → store)
  store             — LocalStore offline (SQLite + npz)
  ui                — render Streamlit del modo Remote Monitoring
"""

from __future__ import annotations

__version__ = "0.1.0-scaffold"

__all__ = [
    "StreamConfig",
    "StreamSource",
    "SimulatedStreamSource",
    "NIStreamSource",
    "RingBuffer",
    "window_to_signals",
    "window_to_loaded_signals",
    "detect_keyphasor",
    "one_x_vector",
    "AcqAgent",
    "LocalStore",
]

from core.remote_monitoring.stream_source import (
    StreamConfig,
    StreamSource,
    SimulatedStreamSource,
)
from core.remote_monitoring.ni_stream_source import NIStreamSource
from core.remote_monitoring.ring_buffer import RingBuffer
from core.remote_monitoring.materialize import (
    window_to_signals,
    window_to_loaded_signals,
)
from core.remote_monitoring.keyphasor import detect_keyphasor, one_x_vector
from core.remote_monitoring.agent import AcqAgent
from core.remote_monitoring.store import LocalStore
