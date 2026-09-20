"""
core/balancing/influence.py — Motor de balanceo por coeficientes de influencia
==============================================================================

Balanceo de campo (1 y 2 planos) por el método de **coeficientes de influencia**,
el estándar de la industria (Bently/Emerson/IRD). Todo en números complejos:
la vibración 1× y los pesos se representan como fasores (amplitud ∠ fase).

Física
------
El coeficiente de influencia α relaciona el cambio de vibración con el peso de
prueba:  α = (V_trial − V_ref) / W_trial   [vibración / gramo].
El peso de corrección que anula la vibración de referencia V0 es:
    W_c = − V0 / α        (un plano)
En dos planos se resuelve el sistema 2×2 complejo:
    [αa1 αa2] [W1]   [Va0]
    [αb1 αb2] [W2] = −[Vb0]

Calidad de balanceo: **ISO 21940-11** (ex ISO 1940). Desbalance permisible:
    U_per[g·mm] = 9549 · G[mm/s] · M[kg] / n[rpm]

Numpy puro (sin Qt/Streamlit/hardware): lo comparten campo y web.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# =====================================================================
# Fasores (amplitud ∠ fase) ↔ complejo
# =====================================================================
def phasor(amplitude: float, phase_deg: float) -> complex:
    """Fasor A∠θ → complejo."""
    return float(amplitude) * np.exp(1j * np.radians(float(phase_deg)))


def to_polar(z: complex) -> Tuple[float, float]:
    """Complejo → (amplitud, fase en grados 0..360)."""
    amp = float(abs(z))
    ang = float(np.degrees(np.angle(z))) % 360.0
    return amp, ang


def _fmt(z: complex, unit: str = "") -> str:
    a, p = to_polar(z)
    return f"{a:,.2f}{unit} ∠ {p:.0f}°"


# =====================================================================
# Resultados
# =====================================================================
@dataclass
class Weight:
    """Peso de corrección: masa (g) a un ángulo (°). `add`=agregar, `remove`=quitar."""
    grams: float
    angle_deg: float
    action: str = "add"          # "add" | "remove"

    @property
    def vector(self) -> complex:
        return phasor(self.grams, self.angle_deg)

    def __str__(self) -> str:
        return f"{self.grams:,.1f} g ∠ {self.angle_deg:.0f}°  ({self.action})"


@dataclass
class BalanceResult:
    """Resultado de un cálculo de balanceo."""
    planes: int                              # 1 o 2
    corrections: List[Weight]                # peso(s) de corrección
    influence: List[complex]                 # coeficiente(s) de influencia
    v_initial: List[complex]                 # vibración de referencia por plano de medida
    v_residual: List[complex]                # vibración residual PREDICHA tras corregir
    units: str = "um_pp"                     # unidad de vibración (informativa)

    @property
    def effectiveness_pct(self) -> float:
        """Reducción de vibración esperada (%) — el peor plano."""
        effs = []
        for v0, vr in zip(self.v_initial, self.v_residual):
            a0 = abs(v0)
            if a0 > 1e-12:
                effs.append((a0 - abs(vr)) / a0 * 100.0)
        return float(min(effs)) if effs else 0.0


# =====================================================================
# Un plano
# =====================================================================
def single_plane(v_ref: complex, v_trial: complex, trial_weight: Weight,
                 units: str = "um_pp") -> BalanceResult:
    """
    Balanceo de UN plano.

    v_ref:        vibración 1× inicial (fasor complejo).
    v_trial:      vibración 1× con el peso de prueba puesto.
    trial_weight: peso de prueba (masa ∠ ángulo).
    """
    wt = trial_weight.vector
    if abs(wt) < 1e-12:
        raise ValueError("El peso de prueba no puede ser cero.")
    if abs(v_trial - v_ref) < 1e-15:
        raise ValueError("La vibración no cambió con el peso de prueba — revisa el montaje/keyphasor.")
    alpha = (v_trial - v_ref) / wt          # coeficiente de influencia
    wc = -v_ref / alpha                      # peso de corrección
    g, ang = to_polar(wc)
    v_res = v_ref + alpha * wc               # residual predicho (≈ 0)
    return BalanceResult(planes=1, corrections=[Weight(g, ang, "add")],
                         influence=[alpha], v_initial=[v_ref], v_residual=[v_res], units=units)


# =====================================================================
# Dos planos
# =====================================================================
def two_plane(v_ref: Sequence[complex],
              trial1: Tuple[Weight, Sequence[complex]],
              trial2: Tuple[Weight, Sequence[complex]],
              units: str = "um_pp") -> BalanceResult:
    """
    Balanceo de DOS planos.

    v_ref:  [Va0, Vb0]  vibración inicial en los planos de medida A y B.
    trial1: (peso1, [Va1, Vb1])  corrida con peso de prueba SOLO en el plano 1.
    trial2: (peso2, [Va2, Vb2])  corrida con peso de prueba SOLO en el plano 2.
    Devuelve W1, W2 (pesos de corrección en los planos 1 y 2).
    """
    va0, vb0 = complex(v_ref[0]), complex(v_ref[1])
    w1, (va1, vb1) = trial1[0].vector, (complex(trial1[1][0]), complex(trial1[1][1]))
    w2, (va2, vb2) = trial2[0].vector, (complex(trial2[1][0]), complex(trial2[1][1]))
    if abs(w1) < 1e-12 or abs(w2) < 1e-12:
        raise ValueError("Los pesos de prueba no pueden ser cero.")
    # Matriz de coeficientes de influencia (vib / gramo).
    a11 = (va1 - va0) / w1; a12 = (va2 - va0) / w2
    a21 = (vb1 - vb0) / w1; a22 = (vb2 - vb0) / w2
    A = np.array([[a11, a12], [a21, a22]], dtype=complex)
    b = -np.array([va0, vb0], dtype=complex)
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-18:
        raise ValueError("Sistema singular — los pesos de prueba no son independientes; cámbialos.")
    w = np.linalg.solve(A, b)                # [W1, W2] complejos
    corr = [Weight(*to_polar(w[0]), action="add"), Weight(*to_polar(w[1]), action="add")]
    v_res = (A @ w + np.array([va0, vb0])).tolist()   # residual predicho ≈ [0, 0]
    return BalanceResult(planes=2, corrections=corr, influence=[a11, a12, a21, a22],
                         v_initial=[va0, vb0], v_residual=[complex(x) for x in v_res], units=units)


# =====================================================================
# ISO 21940-11 (ex ISO 1940) — calidad de balanceo
# =====================================================================
@dataclass
class IsoBalanceCheck:
    grade: float                 # G elegido (mm/s)
    rotor_mass_kg: float
    rpm: float
    u_per_g_mm: float            # desbalance permisible total [g·mm]
    e_per_um: float              # excentricidad permisible [µm]
    residual_g_mm: Optional[float] = None   # desbalance residual estimado [g·mm]
    passed: Optional[bool] = None

    def __str__(self) -> str:
        s = f"G{self.grade:g} · Uper={self.u_per_g_mm:,.0f} g·mm (eper={self.e_per_um:,.1f} µm)"
        if self.residual_g_mm is not None:
            s += f" · residual={self.residual_g_mm:,.0f} g·mm → {'PASA' if self.passed else 'NO PASA'}"
        return s


def iso21940_permissible(grade: float, rotor_mass_kg: float, rpm: float,
                         residual_g_mm: Optional[float] = None) -> IsoBalanceCheck:
    """
    Desbalance permisible por ISO 21940-11.
        U_per[g·mm] = 9549 · G[mm/s] · M[kg] / n[rpm]
        e_per[µm]   = U_per / M[kg] / 1000   (excentricidad permisible)
    Grados típicos: G2.5 (máquinas rotativas generales), G6.3 (ventiladores/bombas),
    G1.0 (máquina-herramienta), G0.4 (husillos de precisión).
    """
    g = float(grade); m = max(float(rotor_mass_kg), 1e-6); n = max(float(rpm), 1e-6)
    u_per = 9549.0 * g * m / n                        # g·mm
    e_per = u_per / m / 1.0                            # µm  (U[g·mm]/M[kg] = µm)
    chk = IsoBalanceCheck(grade=g, rotor_mass_kg=m, rpm=n, u_per_g_mm=u_per, e_per_um=e_per)
    if residual_g_mm is not None:
        chk.residual_g_mm = float(residual_g_mm)
        chk.passed = float(residual_g_mm) <= u_per
    return chk


# =====================================================================
# Utilidades de campo
# =====================================================================
def combine_weights(existing: Weight, correction: Weight) -> Weight:
    """Suma vectorial de un peso ya puesto + la corrección → peso neto resultante."""
    z = existing.vector + correction.vector
    g, ang = to_polar(z)
    return Weight(g, ang, "add")


def split_to_holes(correction: Weight, n_holes: int) -> Tuple[Weight, Weight]:
    """Reparte un peso de corrección entre los DOS agujeros fijos más cercanos
    (cuando no se puede poner masa en el ángulo exacto). Descomposición vectorial
    en los ángulos de agujero adyacentes."""
    if n_holes < 2:
        raise ValueError("Se requieren al menos 2 agujeros.")
    step = 360.0 / n_holes
    target = correction.angle_deg % 360.0
    i0 = int(np.floor(target / step)) % n_holes
    a0 = i0 * step
    a1 = ((i0 + 1) % n_holes) * step
    # Resolver m0·û(a0) + m1·û(a1) = W  (dos incógnitas reales, 2 ecuaciones re/im).
    u0 = phasor(1.0, a0); u1 = phasor(1.0, a1)
    M = np.array([[u0.real, u1.real], [u0.imag, u1.imag]])
    b = np.array([correction.vector.real, correction.vector.imag])
    m0, m1 = np.linalg.solve(M, b)
    return (Weight(float(m0), a0, correction.action), Weight(float(m1), a1, correction.action))
