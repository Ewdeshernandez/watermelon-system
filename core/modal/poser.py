"""
core/modal/poser.py — PoSER: fusión de mode shapes de varios setups (roving)
============================================================================

Cuando hay MÁS puntos de medición que canales, se mide en varios "setups": unos
sensores de REFERENCIA se dejan fijos en todos los setups y el resto ("roving")
se va moviendo. Cada setup estima la forma modal en sus DOFs; PoSER (Post Separate
Estimation Re-scaling, Brincker & Ventura) las une en UNA forma global re-escalando
cada setup para que los DOF de referencia coincidan.

Idea:
  1. Empareja los modos entre setups por frecuencia (el mismo modo físico).
  2. Para cada modo global, toma los DOF de referencia (comunes a todos los setups).
  3. Escala complejo por mínimos cuadrados: α_k = <φ_ref^(0), φ_ref^(k)> / <φ_ref^(k),φ_ref^(k)>
     así φ_ref^(k)·α_k ≈ φ_ref^(0) (setup 0 = referencia global).
  4. Ensambla: cada DOF toma su valor del setup que lo midió, escalado por α_k;
     los DOF de referencia se promedian entre setups (ya re-escalados).

No depende de nada externo: solo numpy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class Setup:
    """Un setup de medición: modos estimados + qué DOF global mide cada canal.

    freqs      : (m,) frecuencias naturales identificadas [Hz]
    shapes     : (m, c) formas modales complejas (c = canales de ESTE setup)
    dof_ids    : (c,) id GLOBAL del DOF que mide cada canal (str o int)
    ref_ids    : ids de los DOF de REFERENCIA (subconjunto de dof_ids, común a todos)
    """

    def __init__(self, freqs: Sequence[float], shapes, dof_ids: Sequence,
                 ref_ids: Optional[Sequence] = None):
        self.freqs = np.asarray(freqs, float)
        self.shapes = np.asarray(shapes, complex)
        if self.shapes.ndim == 1:
            self.shapes = self.shapes[None, :]
        self.dof_ids = list(dof_ids)
        self.ref_ids = list(ref_ids) if ref_ids is not None else []
        self._idx = {d: i for i, d in enumerate(self.dof_ids)}

    def col(self, dof) -> Optional[int]:
        return self._idx.get(dof)


def _pair_mode(setup: Setup, f0: float, tol_hz: float) -> Optional[int]:
    """Índice del modo de `setup` más cercano a f0 dentro de tol_hz (o None)."""
    if setup.freqs.size == 0:
        return None
    j = int(np.argmin(np.abs(setup.freqs - f0)))
    return j if abs(setup.freqs[j] - f0) <= tol_hz else None


def poser_merge(setups: Sequence[Setup], tol_hz: float = 1.0
                ) -> Tuple[List[float], np.ndarray, List]:
    """Une las formas modales de varios setups en formas GLOBALES.

    Devuelve (freqs_global, shapes_global[m, D], dof_order[D]) donde D es el número
    total de DOF distintos entre todos los setups. El setup 0 es la referencia de fase.
    Requiere ≥1 DOF de referencia común a todos los setups.
    """
    setups = list(setups)
    if not setups:
        return [], np.zeros((0, 0), complex), []
    if len(setups) == 1:
        s = setups[0]
        return list(s.freqs), s.shapes.copy(), list(s.dof_ids)

    # Orden global de DOF: primero los del setup 0, luego los nuevos de cada setup
    dof_order: List = []
    for s in setups:
        for d in s.dof_ids:
            if d not in dof_order:
                dof_order.append(d)
    D = len(dof_order)
    dpos = {d: i for i, d in enumerate(dof_order)}

    # Referencias comunes a TODOS los setups
    common_ref = set(setups[0].ref_ids or setups[0].dof_ids)
    for s in setups[1:]:
        common_ref &= set(s.ref_ids or s.dof_ids)
    common_ref = [d for d in dof_order if d in common_ref]
    if not common_ref:
        raise ValueError("PoSER necesita al menos un DOF de referencia común a todos los setups.")

    freqs_global: List[float] = []
    rows: List[np.ndarray] = []
    # Recorre los modos del setup 0 como maestro
    for mi, f0 in enumerate(setups[0].freqs):
        acc = np.zeros(D, complex)
        cnt = np.zeros(D, float)
        ok = True
        phi0_ref = None
        per_setup = []
        for k, s in enumerate(setups):
            j = mi if k == 0 else _pair_mode(s, f0, tol_hz)
            if j is None:
                ok = False
                break
            phi = s.shapes[j]
            # vector de referencia de este setup en el orden de common_ref
            ref_vec = np.array([phi[s.col(d)] for d in common_ref if s.col(d) is not None], complex)
            per_setup.append((s, phi, ref_vec))
            if k == 0:
                phi0_ref = ref_vec
        if not ok or phi0_ref is None:
            continue
        for k, (s, phi, ref_vec) in enumerate(per_setup):
            if k == 0:
                alpha = 1.0 + 0j
            else:
                den = np.vdot(ref_vec, ref_vec)
                alpha = (np.vdot(ref_vec, phi0_ref) / den) if abs(den) > 1e-30 else 1.0 + 0j
            for ci, d in enumerate(s.dof_ids):
                acc[dpos[d]] += alpha * phi[ci]
                cnt[dpos[d]] += 1.0
        shape = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1.0), 0.0)
        # normaliza a máxima componente unitaria (real-max)
        m = np.max(np.abs(shape)) or 1.0
        rows.append(shape / m)
        freqs_global.append(float(f0))

    S = np.array(rows, complex) if rows else np.zeros((0, D), complex)
    return freqs_global, S, dof_order


def mac(a, b) -> float:
    """Modal Assurance Criterion entre dos vectores complejos."""
    a = np.asarray(a, complex).ravel(); b = np.asarray(b, complex).ravel()
    num = abs(np.vdot(a, b)) ** 2
    den = (np.vdot(a, a).real * np.vdot(b, b).real) or 1e-30
    return float(num / den)
