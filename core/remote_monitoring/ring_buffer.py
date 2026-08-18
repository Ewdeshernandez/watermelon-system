"""
core/remote_monitoring/ring_buffer.py — Buffer circular multicanal
==================================================================

Mantiene los últimos N samples de cada canal en RAM constante. El ACQ
Agent escribe bloques nuevos con write(); la UI/análisis lee ventanas con
snapshot() sin consumir (no destruye el contenido → varias vistas pueden
leer la misma ventana).

RAM = n_channels × capacity_samples × 8 bytes (float64).
Ej: 32 ch × 51200 Hz × 10 s ≈ 131 MB. Por eso buffer_seconds se acota;
el histórico largo va a disco (store local), no a RAM.
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Buffer circular (n_channels, capacity_samples), float64.

    Semántica: snapshot(n) devuelve los últimos n samples en orden
    cronológico (el más viejo primero, el más nuevo último).
    """

    def __init__(self, n_channels: int, capacity_samples: int) -> None:
        if n_channels <= 0:
            raise ValueError("n_channels debe ser > 0")
        if capacity_samples <= 0:
            raise ValueError("capacity_samples debe ser > 0")
        self._n_ch = int(n_channels)
        self._cap = int(capacity_samples)
        self._buf = np.zeros((self._n_ch, self._cap), dtype=float)
        self._wp = 0        # write pointer (próxima posición a escribir)
        self._count = 0     # total de samples escritos (satura conceptualmente)

    # --- escritura ---
    def write(self, block: np.ndarray) -> None:
        """Escribe un bloque (n_channels, n_samples)."""
        block = np.asarray(block, dtype=float)
        if block.ndim != 2 or block.shape[0] != self._n_ch:
            raise ValueError(
                f"block debe ser ({self._n_ch}, n); recibí {block.shape}"
            )
        n = block.shape[1]
        if n == 0:
            return

        # Bloque más grande que la capacidad → solo la cola cabe.
        if n >= self._cap:
            self._buf[:] = block[:, -self._cap:]
            self._wp = 0
            self._count += n
            return

        end = self._wp + n
        if end <= self._cap:
            self._buf[:, self._wp:end] = block
        else:
            first = self._cap - self._wp
            self._buf[:, self._wp:] = block[:, :first]
            self._buf[:, : n - first] = block[:, first:]
        self._wp = end % self._cap
        self._count += n

    # --- lectura ---
    @property
    def filled(self) -> int:
        """Cuántos samples válidos hay (≤ capacity)."""
        return min(self._count, self._cap)

    @property
    def n_channels(self) -> int:
        return self._n_ch

    @property
    def capacity(self) -> int:
        return self._cap

    def snapshot(self, n_samples: int | None = None) -> np.ndarray:
        """Últimos n_samples en orden cronológico, shape (n_channels, n).

        Si n_samples es None o mayor que lo disponible, devuelve todo lo
        disponible. Copia (no vista) → seguro para mutar downstream.
        """
        avail = self.filled
        if n_samples is None or n_samples > avail:
            n_samples = avail
        if n_samples <= 0:
            return np.zeros((self._n_ch, 0), dtype=float)

        start = (self._wp - n_samples) % self._cap
        if start + n_samples <= self._cap:
            return self._buf[:, start : start + n_samples].copy()
        first = self._cap - start
        return np.concatenate(
            [self._buf[:, start:], self._buf[:, : n_samples - first]], axis=1
        )

    def clear(self) -> None:
        self._buf.fill(0.0)
        self._wp = 0
        self._count = 0
