"""
core/torsional/monitor.py — Monitoreo torsional de larga duración (24 h)
=========================================================================

Acumula la fatiga y la tendencia SIN guardar la onda cruda continua (que a
2560 Hz × 2 canales × 24 h son ~1.8 GB). Tres flujos livianos:

1. **Rainflow en streaming** (`StreamingRainflow`) — procesa los puntos de
   retorno al vuelo con la regla ASTM E1049 y guarda solo el HISTOGRAMA de
   ciclos (rango→conteo). Tamaño ~KB, crece de forma despreciable.
2. **Tendencia** — cada `trend_dt` s guarda escalares (media, pp, rizado, rpm,
   par máx). 24 h @1 Hz ≈ 86 400 filas de pocos floats (MB).
3. **Eventos** — marca de tiempo + pico cuando el pp supera un umbral
   (sobrecarga / transitorio). Sin la onda de las 24 h.

El histograma alimenta directo `shaft_torsional_fatigue` → daño de Miner REAL
acumulado en la campaña, no extrapolado de una captura de 8 s.

Numpy puro (sin Qt/hardware): lo usa el campo y lo puede reprocesar la web.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.torsional.analysis import RainflowCycle


def _rf_step(pts: deque, hist: Dict[float, float], x: float) -> None:
    """Un paso de la regla de 3 puntos ASTM E1049 sobre (pts, hist)."""
    pts.append(x)
    while len(pts) >= 3:
        x1, x2, x3 = pts[-3], pts[-2], pts[-1]
        X = abs(x3 - x2)
        Y = abs(x2 - x1)
        if X < Y:
            break
        k = round(Y, 9)
        if len(pts) == 3:
            hist[k] = hist.get(k, 0.0) + 0.5      # contiene el arranque → medio ciclo
            pts.popleft()
        else:
            hist[k] = hist.get(k, 0.0) + 1.0
            last = pts.pop(); pts.pop(); pts.pop(); pts.append(last)


class StreamingRainflow:
    """Rainflow ASTM E1049 incremental: se alimenta con puntos de retorno
    (reversals) y mantiene un histograma de rangos + un residuo acotado."""

    def __init__(self) -> None:
        self._pts: deque = deque()
        self.hist: Dict[float, float] = {}

    def feed(self, reversals) -> None:
        for x in np.asarray(reversals, dtype=float).tolist():
            _rf_step(self._pts, self.hist, float(x))

    def ranges(self, drain: bool = False, tail: Optional[float] = None) -> List[Tuple[float, float]]:
        """[(rango, conteo)] ordenado. tail = última muestra de la serie (cierra el
        último tramo, como el extremo final del rainflow por lotes). drain=True suma
        el residuo restante como medios ciclos. NO muta el estado (trabaja en clon)."""
        pts = deque(self._pts); hist = dict(self.hist)
        if tail is not None:
            _rf_step(pts, hist, float(tail))
        if drain:
            p = list(pts)
            for i in range(len(p) - 1):
                r = round(abs(p[i + 1] - p[i]), 9)
                hist[r] = hist.get(r, 0.0) + 0.5
        return sorted(hist.items(), key=lambda kv: kv[0])


def _block_reversals(x: np.ndarray, carry: Optional[float],
                     cdir: float = 0.0) -> Tuple[np.ndarray, float, float]:
    """Puntos de retorno de un bloque (vectorizado), cosidos EXACTAMENTE con el
    bloque previo vía `carry` (última muestra) y `cdir` (última dirección de
    pendiente). Devuelve (reversals, nueva_carry, nueva_dir). Cargar la dirección
    es lo que hace la detección invariante al tamaño de bloque."""
    xin = np.asarray(x, dtype=float)
    if xin.size == 0:
        return np.empty(0), (carry if carry is not None else 0.0), cdir
    first = carry is None
    seq = xin if first else np.concatenate([[carry], xin])
    if seq.size < 2:
        return (seq[:1].copy() if first else np.empty(0)), float(seq[-1]), cdir
    s = np.sign(np.diff(seq))                     # dirección de cada segmento
    if (s == 0).any():                            # aplana mesetas: forward-fill con la dir previa
        idx = np.where(s != 0, np.arange(s.size), -1)
        np.maximum.accumulate(idx, out=idx)
        s = np.where(idx >= 0, s[np.where(idx >= 0, idx, 0)], (cdir if not first else 0.0))
    # extremo en seq[k] cuando la dir ENTRANTE ≠ la SALIENTE. La entrante a seq[0] es
    # `cdir` (dir del bloque previo) → así se captura un extremo que cae justo en la frontera.
    inc = np.concatenate([[cdir], s[:-1]])        # dir entrante a los puntos 0..n-2
    out = s                                        # dir saliente de los puntos 0..n-2
    mask = (inc != out) & (inc != 0)
    revs = seq[np.where(mask)[0]]
    if first:                                     # primer punto = extremo inicial (como el batch)
        revs = np.concatenate([seq[:1], revs])
    new_dir = float(s[-1]) if s.size else cdir
    return revs, float(seq[-1]), new_dir


@dataclass
class MonitorEvent:
    t: float          # segundos desde el inicio
    peak: float       # par pico del evento [EU]
    rpm: float
    kind: str = "pp_over"


@dataclass
class TorsionalMonitor:
    """Acumulador de una campaña de monitoreo torsional de larga duración."""
    units: str = "nm"
    trend_dt: float = 1.0             # cada cuánto se guarda una fila de tendencia [s]
    event_pp: Optional[float] = None  # umbral de pico-pico para marcar evento [EU]
    flat_std_eu: float = 0.5          # std por debajo de esto = señal MUERTA (batería/cable)

    def __post_init__(self) -> None:
        self.rf = StreamingRainflow()
        self.t = 0.0
        self.n_samples = 0
        self.sum_t = 0.0
        self.tmax: Optional[float] = None
        self.tmin: Optional[float] = None
        self.trend: List[Tuple[float, float, float, float, float, float]] = []
        self.events: List[MonitorEvent] = []
        self.bad_seconds = 0.0        # tiempo con señal muerta (para trazabilidad/salud)
        self.signal_ok = True         # estado de la última ventana
        self.last_std = 0.0
        self._carry: Optional[float] = None
        self._cdir: float = 0.0
        self._win: List[np.ndarray] = []
        self._win_rpm = 0.0
        self._acc = 0.0

    def add_block(self, torque, rpm: float, fs: float) -> None:
        x = np.asarray(torque, dtype=float)
        if x.size == 0 or fs <= 0:
            return
        dur = x.size / fs
        self.t += dur
        self.n_samples += x.size
        self.sum_t += float(x.sum())
        bmn, bmx = float(x.min()), float(x.max())
        self.tmin = bmn if self.tmin is None else min(self.tmin, bmn)
        self.tmax = bmx if self.tmax is None else max(self.tmax, bmx)
        # Señal plana → NO tiene reversals → no aporta fatiga; igual la marcamos.
        revs, self._carry, self._cdir = _block_reversals(x, self._carry, self._cdir)
        if revs.size:
            self.rf.feed(revs)
        self._win.append(x); self._win_rpm = float(rpm); self._acc += dur
        if self._acc >= self.trend_dt:
            allx = np.concatenate(self._win)
            mean = float(allx.mean()); pp = float(allx.max() - allx.min())
            self.last_std = float(allx.std())
            self.signal_ok = self.last_std >= self.flat_std_eu       # watchdog de señal
            if not self.signal_ok:
                self.bad_seconds += self._acc
            ripple = (pp / abs(mean) * 100.0) if abs(mean) > 1e-9 else float("inf")
            self.trend.append((round(self.t, 3), mean, pp, ripple, self._win_rpm, float(allx.max())))
            if self.event_pp is not None and pp > self.event_pp:
                self.events.append(MonitorEvent(round(self.t, 3), float(allx.max()), self._win_rpm))
            self._win = []; self._acc = 0.0

    # --- Checkpoint (sobrevive a corte de energía / cierre del PC) ---
    def to_dict(self) -> Dict:
        return {
            "units": self.units, "trend_dt": self.trend_dt, "event_pp": self.event_pp,
            "flat_std_eu": self.flat_std_eu, "t": self.t, "n_samples": self.n_samples,
            "sum_t": self.sum_t, "tmax": self.tmax, "tmin": self.tmin,
            "hist": {str(k): v for k, v in self.rf.hist.items()},
            "residual": list(self.rf._pts), "carry": self._carry, "cdir": self._cdir,
            "trend": self.trend, "events": [e.__dict__ for e in self.events],
            "bad_seconds": self.bad_seconds,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TorsionalMonitor":
        m = cls(units=d.get("units", "nm"), trend_dt=d.get("trend_dt", 1.0),
                event_pp=d.get("event_pp"), flat_std_eu=d.get("flat_std_eu", 0.5))
        m.t = d.get("t", 0.0); m.n_samples = d.get("n_samples", 0); m.sum_t = d.get("sum_t", 0.0)
        m.tmax = d.get("tmax"); m.tmin = d.get("tmin"); m.bad_seconds = d.get("bad_seconds", 0.0)
        m.rf.hist = {float(k): float(v) for k, v in (d.get("hist") or {}).items()}
        m.rf._pts = deque(float(v) for v in (d.get("residual") or []))
        m._carry = d.get("carry"); m._cdir = d.get("cdir", 0.0)
        m.trend = [tuple(r) for r in (d.get("trend") or [])]
        m.events = [MonitorEvent(**e) for e in (d.get("events") or [])]
        return m

    @property
    def duration_s(self) -> float:
        return self.t

    def mean_torque(self) -> float:
        return self.sum_t / max(1, self.n_samples)

    def fatigue_cycles(self) -> List[RainflowCycle]:
        """Ciclos para `shaft_torsional_fatigue`. Media del ciclo ≈ par medio global
        (el histograma no guarda la media por ciclo; el estático domina el Goodman)."""
        m = self.mean_torque()
        return [RainflowCycle(range=r, mean=m, count=c)
                for r, c in self.rf.ranges(drain=True, tail=self._carry) if r > 0]

    def summary(self) -> Dict:
        """Resumen compacto y subible a la nube (histograma + tendencia + eventos)."""
        return {
            "kind": "monitor", "units": self.units,
            "duration_s": round(self.t, 2), "n_samples": self.n_samples,
            "mean": self.mean_torque(), "tmax": self.tmax, "tmin": self.tmin,
            "ranges": self.rf.ranges(drain=True, tail=self._carry),
            "trend": self.trend,
            "events": [e.__dict__ for e in self.events],
            "bad_seconds": round(self.bad_seconds, 1), "signal_ok": self.signal_ok,
        }
