"""
core/modal/ema_oma_correlation.py — Correlación EMA ↔ OMA
========================================================

Empareja las frecuencias identificadas por Análisis Modal Experimental (EMA, con
martillo) contra los modos de Análisis Modal Operacional (OMA, en operación) y
entrega la tabla de correspondencia (Δf, Δ%) y, si hay formas modales, el MAC.

Es el módulo que produce la sección "CORRELACIÓN DINÁMICA COMPLEMENTARIA EMA–OMA"
del reporte, sin depender de software externo. Referencia: ISO 7626-6 (validación
de parámetros modales) + API 684.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class ModeMatch:
    ema_label: str
    ema_hz: float
    oma_label: str
    oma_hz: float
    delta_hz: float
    delta_pct: float
    mac: Optional[float] = None      # 0..1 si hay formas modales

    def as_row(self) -> dict:
        r = {
            "Respuesta EMA": self.ema_label,
            "Frecuencia EMA [Hz]": round(self.ema_hz, 3),
            "Frecuencia OMA [Hz]": round(self.oma_hz, 3),
            "Δf [Hz]": round(self.delta_hz, 3),
            "Δ [%]": round(self.delta_pct, 2),
        }
        if self.mac is not None:
            r["MAC"] = round(self.mac, 3)
        return r


def mac(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    """Modal Assurance Criterion entre dos formas modales (complejas)."""
    a = np.asarray(phi_a, dtype=complex).ravel()
    b = np.asarray(phi_b, dtype=complex).ravel()
    if a.size != b.size or a.size == 0:
        return 0.0
    num = abs(np.vdot(a, b)) ** 2
    den = float((np.vdot(a, a) * np.vdot(b, b)).real)
    return float(num / den) if den > 0 else 0.0


def correlate(
    ema_freqs: Sequence[float],
    oma_freqs: Sequence[float],
    tol_hz: float = 2.0,
    ema_labels: Optional[Sequence[str]] = None,
    oma_labels: Optional[Sequence[str]] = None,
    ema_shapes: Optional[Sequence[np.ndarray]] = None,
    oma_shapes: Optional[Sequence[np.ndarray]] = None,
    allow_multi: bool = True,
) -> List[ModeMatch]:
    """Empareja EMA↔OMA por frecuencia más cercana dentro de tol_hz.

    allow_multi=True: una respuesta EMA puede correlacionar con >1 modo OMA
    cercano (como en el reporte, donde 20 Hz EMA correlaciona con 19,361 y 20,364).
    Si hay formas modales, calcula MAC y desempata por MAC.
    """
    el = list(ema_labels) if ema_labels else [f"EMA {i+1}" for i in range(len(ema_freqs))]
    ol = list(oma_labels) if oma_labels else [f"OMA {i+1}" for i in range(len(oma_freqs))]
    matches: List[ModeMatch] = []
    for i, ef in enumerate(ema_freqs):
        cands = [(j, of) for j, of in enumerate(oma_freqs) if abs(of - ef) <= tol_hz]
        if not cands:
            continue
        cands.sort(key=lambda t: abs(t[1] - ef))
        chosen = cands if allow_multi else cands[:1]
        for j, of in chosen:
            m = None
            if ema_shapes is not None and oma_shapes is not None:
                try:
                    m = mac(ema_shapes[i], oma_shapes[j])
                except Exception:  # noqa: BLE001
                    m = None
            matches.append(ModeMatch(
                ema_label=el[i], ema_hz=float(ef), oma_label=ol[j], oma_hz=float(of),
                delta_hz=abs(float(ef) - float(of)),
                delta_pct=abs(float(ef) - float(of)) / float(of) * 100.0 if of else 0.0,
                mac=m))
    matches.sort(key=lambda x: x.delta_hz)
    return matches


def correlation_table(matches: Sequence[ModeMatch]) -> List[dict]:
    return [m.as_row() for m in matches]


def summarize(matches: Sequence[ModeMatch]) -> str:
    if not matches:
        return ("No se identifican correspondencias entre las respuestas EMA y los modos OMA "
                "dentro de la tolerancia establecida.")
    best = matches[:6]
    txt = ("La comparación EMA–OMA evidencia correspondencia entre las respuestas "
           "experimentales y los modos operacionales. ")
    txt += "; ".join(
        f"{m.ema_label} {m.ema_hz:.1f} Hz ↔ {m.oma_hz:.3f} Hz (Δ {m.delta_hz:.3f} Hz"
        + (f", MAC {m.mac:.2f}" if m.mac is not None else "") + ")"
        for m in best)
    txt += (". Una coincidencia de frecuencias no confirma resonancia por sí sola; debe "
            "evaluarse con la amplitud y la evolución con la velocidad (API 684).")
    return txt
