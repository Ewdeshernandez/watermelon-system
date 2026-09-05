"""
core/modal/ssi.py — Stochastic Subspace Identification (SSI-COV) + incertidumbre
================================================================================

Identificación modal en dominio del TIEMPO (OMA), el método premium que hace fuerte
a ARTeMIS Pro. SSI-COV: covarianzas de salida → Toeplitz por bloques → SVD →
matriz de observabilidad → A, C → autovalores → frecuencias / amortiguamiento /
formas modales. Barrido de ÓRDENES → diagrama de ESTABILIZACIÓN. La dispersión de
los polos estables da la INCERTIDUMBRE (std de fn y ζ) — equivalente a las barras
de error del "Crystal Clear SSI".

Núcleo numpy puro (testeado). Referencia: Van Overschee & De Moor; Brincker & Ventura.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SSIMode:
    frequency_hz: float
    damping_ratio_pct: float
    std_frequency_hz: float          # incertidumbre (dispersión en el diagrama)
    std_damping_pct: float
    mode_shape: np.ndarray           # complejo (n_ch,)
    n_stable: int                    # en cuántos órdenes apareció estable
    complexity_pct: float = 0.0


@dataclass
class SSIResult:
    modes: List[SSIMode]
    orders: List[int]
    diagram: List[Tuple[int, np.ndarray, np.ndarray]]  # (order, freqs_hz, stable_mask)
    fmin_hz: float
    fmax_hz: float


def _output_covariances(y: np.ndarray, max_lag: int) -> np.ndarray:
    """R[k] = (1/(N-k)) Σ y_{t+k} y_t^T, k=0..max_lag. y: (N, ch)."""
    y = np.asarray(y, float)
    y = y - y.mean(axis=0, keepdims=True)
    N, ch = y.shape
    R = np.zeros((max_lag + 1, ch, ch))
    for k in range(max_lag + 1):
        R[k] = (y[k:].T @ y[:N - k]) / max(1, (N - k))
    return R


def _block_toeplitz(R: np.ndarray, i: int) -> np.ndarray:
    """Toeplitz por bloques T_{1|i} (ch·i × ch·i), bloque(a,b)=R[i+a-b]."""
    ch = R.shape[1]
    T = np.zeros((ch * i, ch * i))
    for a in range(i):
        for b in range(i):
            T[a * ch:(a + 1) * ch, b * ch:(b + 1) * ch] = R[i + a - b]
    return T


def _poles_at_order(U: np.ndarray, s: np.ndarray, ch: int, n: int, fs: float):
    """Polos (fn, zeta, shape) para el orden n (par)."""
    O = U[:, :n] * np.sqrt(s[:n])[None, :]          # observabilidad (ch·i × n)
    O_up = O[:-ch, :]; O_dn = O[ch:, :]
    A = np.linalg.pinv(O_up) @ O_dn
    C = O[:ch, :]
    mu, V = np.linalg.eig(A)
    out = []
    dt = 1.0 / fs
    for k in range(len(mu)):
        m = mu[k]
        if np.abs(m) < 1e-12 or m.imag <= 0:         # un polo por par conjugado
            continue
        lam = np.log(m) / dt
        wn = np.abs(lam)
        fn = wn / (2 * np.pi)
        zeta = -lam.real / wn if wn > 0 else 1.0
        shape = C @ V[:, k]
        out.append((float(fn), float(zeta), shape))
    return out


def _mpc(shape: np.ndarray) -> float:
    phi = np.asarray(shape, complex).ravel()
    if phi.size == 0:
        return 0.0
    re, im = phi.real, phi.imag
    sxx = float(re @ re); syy = float(im @ im); sxy = float(re @ im)
    if sxx + syy < 1e-30:
        return 0.0
    # MPC de Pappa (invariante a una fase global e^{iθ} del modo): depende solo de
    # los autovalores de la matriz de dispersión 2×2 [[sxx,sxy],[sxy,syy]].
    tr = sxx + syy
    discr = max(tr ** 2 / 4.0 - (sxx * syy - sxy ** 2), 0.0)
    lam1 = tr / 2.0 + np.sqrt(discr); lam2 = tr / 2.0 - np.sqrt(discr)
    mpc = ((lam1 - lam2) / (lam1 + lam2)) ** 2 if (lam1 + lam2) > 1e-30 else 0.0
    return float(np.clip((1.0 - mpc) * 100.0, 0.0, 100.0))


def run_ssi_cov(
    data: np.ndarray,
    fs: float,
    orders: Optional[Sequence[int]] = None,
    i_block: int = 25,
    fmin_hz: float = 2.0,
    fmax_hz: Optional[float] = None,
    f_tol: float = 0.01,             # 1% para estabilidad en frecuencia
    z_tol: float = 0.05,             # 5% (abs) en amortiguamiento
    max_damp: float = 0.20,
) -> SSIResult:
    """SSI-COV con diagrama de estabilización + incertidumbre por dispersión."""
    y = np.asarray(data, float)
    if y.ndim == 1:
        y = y[:, None]
    N, ch = y.shape
    fmax_hz = fmax_hz or fs / 2.56
    if orders is None:
        orders = list(range(2, 41, 2))
    orders = sorted(int(o) for o in orders if o >= 2)
    i_block = max(orders[-1] // ch + 2, i_block)      # asegura Toeplitz suficientemente grande
    R = _output_covariances(y, 2 * i_block)
    T = _block_toeplitz(R, i_block)
    U, s, _ = np.linalg.svd(T)

    per_order = []                                     # [(order, [(fn,zeta,shape)...])]
    for n in orders:
        n = min(n, U.shape[1] - ch)
        if n < 2:
            continue
        poles = [p for p in _poles_at_order(U, s, ch, n, fs)
                 if fmin_hz <= p[0] <= fmax_hz and 0 < p[1] < max_damp]
        per_order.append((n, poles))

    # estabilidad: un polo es estable si hay uno cercano en el orden anterior
    diagram = []
    prev = []
    stable_pool = []                                   # (fn, zeta, shape) estables
    for (n, poles) in per_order:
        freqs = np.array([p[0] for p in poles]); mask = np.zeros(len(poles), bool)
        for idx, (fn, z, sh) in enumerate(poles):
            for (pfn, pz, _psh) in prev:
                if pfn > 0 and abs(fn - pfn) / pfn < f_tol and abs(z - pz) < z_tol:
                    mask[idx] = True
                    stable_pool.append((fn, z, sh)); break
        diagram.append((n, freqs, mask))
        prev = poles

    # clustering de polos estables por frecuencia → modos + incertidumbre
    stable_pool.sort(key=lambda t: t[0])
    clusters: List[List[Tuple[float, float, np.ndarray]]] = []
    for (fn, z, sh) in stable_pool:
        if clusters and abs(fn - np.mean([c[0] for c in clusters[-1]])) / fn < 2 * f_tol:
            clusters[-1].append((fn, z, sh))
        else:
            clusters.append([(fn, z, sh)])
    modes: List[SSIMode] = []
    for cl in clusters:
        if len(cl) < 2:                                # exige aparecer estable ≥2 veces
            continue
        fns = np.array([c[0] for c in cl]); zs = np.array([c[1] for c in cl]) * 100.0
        sh = cl[-1][2]
        modes.append(SSIMode(
            frequency_hz=float(np.mean(fns)), damping_ratio_pct=float(np.mean(zs)),
            std_frequency_hz=float(np.std(fns)), std_damping_pct=float(np.std(zs)),
            mode_shape=sh, n_stable=len(cl), complexity_pct=_mpc(sh)))
    modes.sort(key=lambda m: m.frequency_hz)
    return SSIResult(modes=modes, orders=orders, diagram=diagram,
                     fmin_hz=fmin_hz, fmax_hz=fmax_hz)
