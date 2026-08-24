"""
core/remote_monitoring/agent.py — ACQ Agent (orquestador de streaming)
======================================================================

El ACQ Agent es el ÚNICO componente que toca la fuente de datos. Corre el
loop: source.read_block() → RingBuffer.write() → (opcional) persistir al
store local cada N segundos.

En sitio corre headless como servicio (Windows). En dev/UI corre igual con
SimulatedStreamSource. Dos modos de operación:

  · run_for(seconds) / pump(n_blocks)  → SÍNCRONO. Para tests, headless y
    para el patrón de refresco de Streamlit (bombear unos bloques por rerun).
  · start()/stop()                     → hilo en background. Para el
    servicio de sitio y para "live" continuo real.

Thread-safe: snapshot() copia bajo lock; la UI puede leer mientras el hilo
escribe.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import numpy as np

from core.remote_monitoring.ring_buffer import RingBuffer
from core.remote_monitoring.stream_source import StreamSource
from core.remote_monitoring.materialize import window_to_signals


class AcqAgent:
    def __init__(
        self,
        source: StreamSource,
        instance_id: str = "adhoc",
        store: Optional["object"] = None,          # LocalStore | None
        persist_every_s: Optional[float] = None,   # None = no persistir auto
        on_block: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self.source = source
        self.instance_id = instance_id
        self.store = store
        self.persist_every_s = persist_every_s
        self.on_block = on_block

        cfg = source.config
        self.buffer = RingBuffer(cfg.n_channels, cfg.buffer_samples)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._last_persist = 0.0
        self._blocks_read = 0
        self._started = False

    # ------------------------------------------------------------ helpers
    @property
    def sample_rate_hz(self) -> float:
        return self.source.sample_rate_hz

    @property
    def channels(self):
        return self.source.channels

    @property
    def blocks_read(self) -> int:
        return self._blocks_read

    def _ingest(self, block: np.ndarray, now: float) -> None:
        with self._lock:
            self.buffer.write(block)
        self._blocks_read += 1
        if self.on_block is not None:
            self.on_block(block)
        if self.store is not None and self.persist_every_s is not None:
            if now - self._last_persist >= self.persist_every_s:
                self._persist(now)

    def _persist(self, now: float) -> None:
        snap = self.snapshot()
        if snap.shape[1] == 0:
            return
        ch_meta = [
            {
                "name": ch.name, "bnc_port": ch.bnc_port, "coupling": ch.coupling,
                "sensitivity_mv_per_eu": float(ch.sensitivity_mv_per_eu or 0.0),
                "units": ch.units,
            }
            for ch in self.channels
        ]
        rpm = self.estimate_rpm(snap)
        self.store.save_snapshot(
            self.instance_id, snap, ch_meta, self.sample_rate_hz, rpm=rpm,
        )
        self._last_persist = now

    # --------------------------------------------------------- lifecycle
    def _ensure_started(self) -> None:
        if not self._started:
            self.source.start()
            self._started = True

    def pump(self, n_blocks: int = 1) -> None:
        """Lee n bloques SÍNCRONO (para tests / refresco de Streamlit)."""
        self._ensure_started()
        for _ in range(n_blocks):
            block = self.source.read_block()
            self._ingest(block, time.monotonic())

    def run_for(self, seconds: float) -> None:
        """Bombea bloques SÍNCRONO durante ~seconds (sin hilo)."""
        self._ensure_started()
        cfg = self.source.config
        n = max(1, int(round(seconds / cfg.block_seconds)))
        self.pump(n)

    def _loop(self) -> None:
        self._ensure_started()
        # Con hardware real, read_block() BLOQUEA hasta que el DAQ llena el bloque
        # (ya corre a tiempo real). Con el SIMULADOR, read_block() devuelve al
        # instante → hay que PACEAR a tiempo real, si no: (1) el arranque de 90 s
        # se completa en un segundo (transitorios con 4 puntos) y (2) el hilo satura
        # un núcleo de CPU generando datos sin parar (traba la PC).
        sim = type(self.source).__name__ == "SimulatedStreamSource"
        dt = float(getattr(self.source.config, "block_seconds", 0.1))
        next_t = time.monotonic()
        while not self._stop_evt.is_set():
            block = self.source.read_block()
            self._ingest(block, time.monotonic())
            if sim:
                next_t += dt
                slp = next_t - time.monotonic()
                if slp > 0:
                    self._stop_evt.wait(slp)      # duerme pero responde al stop
                else:
                    next_t = time.monotonic()     # se atrasó → resync

    def start(self) -> None:
        """Arranca el loop en un hilo background (live continuo)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="acq-agent", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self.source.stop()
        except Exception:  # noqa: BLE001
            pass
        self._started = False

    # ------------------------------------------------------------ reading
    def snapshot(self, seconds: Optional[float] = None) -> np.ndarray:
        n = None
        if seconds is not None:
            n = int(round(seconds * self.sample_rate_hz))
        with self._lock:
            return self.buffer.snapshot(n)

    def estimate_rpm(self, snap: Optional[np.ndarray] = None) -> Optional[float]:
        """rpm desde el canal keyphasor, si existe."""
        if snap is None:
            snap = self.snapshot()
        if snap.shape[1] == 0:
            return None
        kph_idx = self.source.config.keyphasor_index()
        if kph_idx is None:
            return None
        from core.remote_monitoring.keyphasor import detect_keyphasor
        # Ventana corta (≤2 s) para SEGUIR transitorios sin retraso: el buffer
        # largo (8 s) promedia la rampa y deja el rpm ~cientos de rpm atrás,
        # descuadrando el Bode. 2 s alcanza para varios pulsos aun a baja rpm.
        kph = snap[kph_idx]
        win = min(len(kph), int(round(2.0 * self.sample_rate_hz)))
        return detect_keyphasor(kph[-win:], self.sample_rate_hz).rpm

    def live_signals(self, seconds: Optional[float] = None,
                     include_keyphasor: bool = False) -> List:
        """Ventana actual → List[Signal] lista para los gráficos."""
        snap = self.snapshot(seconds)
        rpm = self.estimate_rpm(snap)
        return window_to_signals(
            snap, self.channels, self.sample_rate_hz, rpm=rpm,
            keyphasor_name=self.source.config.keyphasor_name,
            include_keyphasor=include_keyphasor,
        )
