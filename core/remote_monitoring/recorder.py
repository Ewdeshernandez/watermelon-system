"""Grabador de transitorio — guarda la forma de onda CRUDA en continuo a disco
durante un arranque/parada, para que NO se pierda ningún dato pase lo que pase
(estilo Bently *transient data recorder*). Se engancha al hook `on_block` del
AcqAgent, así persiste CADA bloque (no por refresco). Post-procesable a
Bode/Cascada/Órbita a cualquier resolución de rpm desde el registro completo.

Layout en disco (WM_PERSIST_DIR/remote_monitoring/transients/<instance>/<rec_id>/):
  · manifest.json   — fs, canales, máquina, timestamps, totales
  · data.f32        — bloques crudos float32 concatenados, cada uno C-order (canales, n)
  · index.jsonl     — un renglón por bloque: {seq, off, n, ts}   (para reconstruir)
Se hace flush tras cada bloque → crash-safe (si la app muere, queda todo hasta
el último bloque).
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def _persist_root() -> str:
    root = os.environ.get("WM_PERSIST_DIR") or os.path.join(os.path.expanduser("~"), ".watermelon")
    return os.path.join(root, "remote_monitoring", "transients")


@dataclass
class RecorderStatus:
    rec_id: str
    n_channels: int
    fs: float
    blocks: int = 0
    samples: int = 0
    bytes: int = 0
    started_ts: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.samples / self.fs if self.fs else 0.0

    @property
    def size_mb(self) -> float:
        return self.bytes / 1e6


class TransientRecorder:
    """Graba bloques crudos a disco. Thread-safe (on_block puede venir de un hilo)."""

    def __init__(self, instance_id: str, fs: float, ch_meta: List[dict],
                 machine: str = "", rec_id: Optional[str] = None) -> None:
        self.fs = float(fs)
        self.ch_meta = list(ch_meta)
        self.n_channels = len(ch_meta)
        self.rec_id = rec_id or time.strftime("rec_%Y%m%d_%H%M%S")
        self.dir = os.path.join(_persist_root(), instance_id, self.rec_id)
        os.makedirs(self.dir, exist_ok=True)
        self._lock = threading.Lock()
        self._data = open(os.path.join(self.dir, "data.f32"), "ab")
        self._index = open(os.path.join(self.dir, "index.jsonl"), "a")
        self._offset = 0
        self._open = True
        self.status = RecorderStatus(self.rec_id, self.n_channels, self.fs, started_ts=time.time())
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump({"rec_id": self.rec_id, "fs": self.fs, "machine": machine,
                       "n_channels": self.n_channels, "channels": self.ch_meta,
                       "started": self.status.started_ts}, f, indent=2)

    def append(self, block: np.ndarray) -> None:
        """Persiste un bloque crudo (canales, muestras). Se llama por on_block."""
        if not self._open:
            return
        b = np.ascontiguousarray(block, dtype=np.float32)
        if b.ndim != 2 or b.shape[0] != self.n_channels:
            return
        n = int(b.shape[1])
        with self._lock:
            self._data.write(b.tobytes())
            self._data.flush()
            self._index.write(json.dumps({"seq": self.status.blocks, "off": self._offset,
                                          "n": n, "ts": time.time()}) + "\n")
            self._index.flush()
            self._offset += n
            self.status.blocks += 1
            self.status.samples += n
            self.status.bytes += b.nbytes

    def stop(self) -> RecorderStatus:
        with self._lock:
            if self._open:
                self._open = False
                for fh in (self._data, self._index):
                    try:
                        fh.flush(); fh.close()
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    p = os.path.join(self.dir, "manifest.json")
                    m = json.load(open(p))
                    m.update(stopped=time.time(), samples=self.status.samples,
                             blocks=self.status.blocks, duration_s=self.status.duration_s)
                    json.dump(m, open(p, "w"), indent=2)
                except Exception:  # noqa: BLE001
                    pass
        return self.status

    @property
    def open(self) -> bool:
        return self._open


def list_recordings(instance_id: str) -> List[dict]:
    """Manifiestos de las grabaciones de una instancia, más recientes primero."""
    base = os.path.join(_persist_root(), instance_id)
    out = []
    if not os.path.isdir(base):
        return out
    for rid in os.listdir(base):
        mp = os.path.join(base, rid, "manifest.json")
        if os.path.isfile(mp):
            try:
                m = json.load(open(mp))
                m["_dir"] = os.path.join(base, rid)
                out.append(m)
            except Exception:  # noqa: BLE001
                pass
    out.sort(key=lambda m: m.get("started", 0), reverse=True)
    return out


def load_recording(rec_dir: str) -> Tuple[dict, np.ndarray]:
    """Reconstruye (manifest, waveform completa (canales, muestras)) del registro."""
    manifest = json.load(open(os.path.join(rec_dir, "manifest.json")))
    nch = int(manifest["n_channels"])
    raw = np.fromfile(os.path.join(rec_dir, "data.f32"), dtype=np.float32)
    idx_path = os.path.join(rec_dir, "index.jsonl")
    cols, pos = [], 0
    if os.path.isfile(idx_path):
        for line in open(idx_path):
            try:
                e = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            n = int(e["n"])
            chunk = raw[pos:pos + nch * n]
            if chunk.size == nch * n:
                cols.append(chunk.reshape(nch, n))
            pos += nch * n
    full = np.hstack(cols) if cols else np.zeros((nch, 0), dtype=np.float32)
    return manifest, full
