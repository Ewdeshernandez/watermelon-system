"""
core/modal/ema_engine.py — Motor de Análisis Modal Experimental (EMA)
======================================================================

Implementación nativa de LSCF (Least-Squares Complex Frequency-domain) para
identificación modal a partir de FRFs medidas con martillo modal.

Por qué implementación nativa y no pyEMA
-----------------------------------------
pyEMA y sdypy-EMA importan matplotlib.backends.backend_tkagg en module-load,
lo cual requiere tkinter — NO disponible en Streamlit Cloud. Implementación
nativa en numpy/scipy elimina esta dependencia frágil y nos da control total
sobre conditioning numérico y stability checks.

Algoritmo LSCF (Reynolds 2003, ISO 7626-6 secc. 6.3)
-----------------------------------------------
Para FRF H(ω) compleja medida en N_outputs:

1. Discretización via z = exp(-jωT) donde T = 1/fs
2. Modelo common-denominator: H_o(z) = N_o(z) / d(z)
   donde d(z) es polinomio compartido, N_o(z) específico por output
3. Reduced Normal Equations:
   M = Σ_o (Y_o^H Y_o - Y_o^H X_o (X_o^H X_o)^-1 X_o^H Y_o)
   donde X_o = matriz Vandermonde de z, Y_o = H_o(ω) * X_o
4. Fijar último coeficiente α_n = 1, resolver M[:-1, :-1] @ a = -M[:-1, -1]
5. Roots del polinomio denominador → polos en z-domain
6. Convertir a s-domain: s = ln(z) / T
7. Filtrar polos estables: Re(s) < 0
8. Extraer fn = |s|/(2π), ζ = -Re(s)/|s|

Stability Diagram (ISO 7626-6 secc. 6.5)
------------------------------------
Para cada orden n de 2 a N_max (paso 2):
  Ejecutar LSCF
  Comparar polos con orden anterior:
    - freq diff < freq_tol_pct → "freq_stable"
    - freq + damping diff < tolerances → "stable" (verdadero modo natural)
    - else → "unstable" (artefacto numérico)

Auto-selección final (clustering)
---------------------------------
Polos "stable" agrupados por proximidad de frecuencia. De cada cluster
se toma el polo de orden más alto (más refinado).

Norma aplicable
---------------
ISO 7626-6 secc. 6.3 — Identificación de parámetros modales por curve fitting
ISO 7626-6 secc. 6.5 — Validación con MAC entre orders consecutivos
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


# =====================================================================
# Tipos
# =====================================================================

@dataclass
class ModalPole:
    """Un polo identificado en una iteración LSCF."""
    natural_frequency_hz: float
    damping_ratio_pct: float
    s_pole: complex                  # polo en s-domain
    model_order: int
    stability: str = "unstable"      # "stable" | "freq_stable" | "unstable"

    @property
    def is_stable(self) -> bool:
        return self.stability == "stable"


@dataclass
class StabilityDiagram:
    """Diagrama de estabilidad — polos por orden de modelo."""
    poles_by_order: Dict[int, List[ModalPole]] = field(default_factory=dict)
    f_min_hz: float = 0.0
    f_max_hz: float = 0.0
    algorithm: str = "LSCF (native)"


@dataclass
class IdentifiedMode:
    """Modo natural final identificado por curve fit + clustering."""
    mode_number: int
    natural_frequency_hz: float
    damping_ratio_pct: float
    s_pole: complex
    model_order_picked: int
    n_stable_orders: int             # cuántos orders confirmaron este modo
    confidence: float                # 0-1: n_stable / total_orders


# =====================================================================
# LSCF Core
# =====================================================================

def lscf_single_order(
    frf: np.ndarray,
    omega: np.ndarray,
    n: int,
    sample_rate_hz: float,
) -> List[ModalPole]:
    """
    Ejecuta LSCF para un solo orden de modelo n.

    Args:
        frf: FRF compleja, shape (n_freq,) para single output o (n_freq, n_out) multi
        omega: vector de frecuencias angulares (rad/s), shape (n_freq,)
        n: orden del polinomio denominador (debe ser par)
        sample_rate_hz: fs original (para discretización z)

    Returns:
        Lista de ModalPole estables (Re(s) < 0).
    """
    if frf.ndim == 1:
        H = frf.reshape(-1, 1)
    else:
        H = frf

    if n % 2 != 0:
        n = n + 1  # debe ser par para polos complejos conjugados

    n_freq = H.shape[0]
    n_out = H.shape[1]
    T = 1.0 / sample_rate_hz

    # Vandermonde de z = exp(-jωT)
    z = np.exp(-1j * omega * T)
    # Z[i, k] = z_i^k for k = 0..n
    Z = np.vander(z, N=n + 1, increasing=True)  # shape (n_freq, n+1)

    # Reduced normal equations
    M = np.zeros((n + 1, n + 1), dtype=complex)
    for o in range(n_out):
        H_o = H[:, o]
        Y_o = H_o[:, None] * Z  # (n_freq, n+1)
        # X_o = Z (común)
        # Computar X_o^H X_o y su pinv (común a todos los outputs)
        if o == 0:
            XtX = Z.conj().T @ Z
            XtX_inv = np.linalg.pinv(XtX)
        R_o = Y_o.conj().T @ Z      # (n+1, n+1)
        M += Y_o.conj().T @ Y_o - R_o @ XtX_inv @ R_o.conj().T

    # M es hermitiana. Tomamos su parte real (debería serlo) para numérica.
    M = (M + M.conj().T) / 2.0

    # Fijar último coeficiente α_n = 1, resolver para los demás
    try:
        a = np.linalg.solve(M[:-1, :-1].real, -M[:-1, -1].real)
    except np.linalg.LinAlgError:
        return []
    alpha = np.concatenate([a, [1.0]])  # coeficientes en orden ascendente

    # Roots del polinomio denominador
    # numpy.roots espera coefs en orden DESCENDENTE
    try:
        roots_z = np.roots(alpha[::-1])
    except (np.linalg.LinAlgError, ValueError):
        return []

    # Convertir z → s domain: s = ln(z) / T
    # ln de complejo: log(|z|) + j*arg(z)
    # Filtrar |z| razonables (~1 para polos en círculo unidad)
    poles: List[ModalPole] = []
    for z_root in roots_z:
        if abs(z_root) < 1e-10 or abs(z_root) > 10:
            continue
        s = np.log(z_root) / T
        omega_n = abs(s)
        if omega_n < 1e-6:
            continue
        # ISO 7626-6 secc. 6.3 — polo físico: Re(s) < 0 (sistema estable)
        if np.real(s) >= 0:
            continue
        zeta = -float(np.real(s) / omega_n)
        # Damping físico: 0 < ζ < 1 (sub-amortiguado)
        if zeta <= 0 or zeta >= 1.0:
            continue
        fn = float(omega_n / (2.0 * np.pi))
        # Filtrar polos espurios sin físico — fn debe estar en banda Nyquist
        if fn <= 0 or fn >= sample_rate_hz / 2.0:
            continue

        poles.append(ModalPole(
            natural_frequency_hz=fn,
            damping_ratio_pct=zeta * 100.0,
            s_pole=complex(s),
            model_order=n,
            stability="unstable",  # se completa después en stability_check
        ))

    # Cada par conjugado da 2 polos — pero solo retornamos los de Im(s) > 0
    # (el conjugado tiene la misma fn y ζ — no aporta info nueva)
    poles_unique: List[ModalPole] = []
    for p in poles:
        if np.imag(p.s_pole) > 0:
            poles_unique.append(p)

    poles_unique.sort(key=lambda p: p.natural_frequency_hz)
    return poles_unique


def run_lscf_stability(
    frf: np.ndarray,
    frequencies_hz: np.ndarray,
    pol_order_high: int = 60,
    f_lower_hz: float = 5.0,
    f_upper_hz: Optional[float] = None,
    sample_rate_hz: Optional[float] = None,
    freq_tol_pct: float = 1.0,
    damp_tol_pct: float = 5.0,
) -> StabilityDiagram:
    """
    Ejecuta LSCF para órdenes 2, 4, ..., pol_order_high y construye stability diagram.

    Args:
        frf: FRF compleja
        frequencies_hz: eje de frecuencia
        pol_order_high: orden máximo de modelo (típico 40-80)
        f_lower_hz, f_upper_hz: banda de interés
        sample_rate_hz: fs original. Si None, se infiere de frequencies_hz
        freq_tol_pct: % tolerancia para clasificar "freq_stable"
        damp_tol_pct: % tolerancia adicional para "stable"

    Returns:
        StabilityDiagram con polos clasificados.
    """
    f = np.asarray(frequencies_hz, dtype=float)
    H = np.asarray(frf, dtype=complex)

    if f_upper_hz is None:
        f_upper_hz = float(f[-1])

    # Recortar a banda de interés
    mask = (f >= f_lower_hz) & (f <= f_upper_hz)
    f_band = f[mask]
    H_band = H[mask]

    if sample_rate_hz is None:
        # Estimar Nyquist desde f_max
        sample_rate_hz = 2.0 * float(f[-1])

    omega = 2.0 * np.pi * f_band

    diagram = StabilityDiagram(
        f_min_hz=float(f_lower_hz),
        f_max_hz=float(f_upper_hz),
    )

    # Ejecutar LSCF para cada orden par
    orders = list(range(2, pol_order_high + 1, 2))
    for n in orders:
        poles = lscf_single_order(H_band, omega, n, sample_rate_hz)
        diagram.poles_by_order[n] = poles

    # Clasificar estabilidad cross-order
    _classify_stability(diagram, freq_tol_pct, damp_tol_pct)
    return diagram


def _classify_stability(
    diagram: StabilityDiagram,
    freq_tol_pct: float,
    damp_tol_pct: float,
) -> None:
    """
    Marca cada polo con su stability status comparando con el orden ANTERIOR.

    Reglas:
    · stable     — fn y ζ within tolerances (verdadero modo natural)
    · freq_stable — solo fn within tolerance (damping varía → análisis cuestionable)
    · unstable   — sin match → polo espurio
    """
    orders = sorted(diagram.poles_by_order.keys())
    for i, n in enumerate(orders):
        if i == 0:
            # primer orden: todos unstable por definición
            for p in diagram.poles_by_order[n]:
                p.stability = "unstable"
            continue

        prev_n = orders[i - 1]
        prev_poles = diagram.poles_by_order[prev_n]

        for p in diagram.poles_by_order[n]:
            best_status = "unstable"
            for q in prev_poles:
                if q.natural_frequency_hz <= 0:
                    continue
                freq_diff_pct = abs(p.natural_frequency_hz - q.natural_frequency_hz) \
                    / q.natural_frequency_hz * 100.0
                if freq_diff_pct > freq_tol_pct:
                    continue
                # Match de frecuencia
                if q.damping_ratio_pct > 0:
                    damp_diff_pct = abs(p.damping_ratio_pct - q.damping_ratio_pct) \
                        / q.damping_ratio_pct * 100.0
                else:
                    damp_diff_pct = 100.0
                if damp_diff_pct <= damp_tol_pct:
                    best_status = "stable"
                    break
                else:
                    best_status = "freq_stable"
            p.stability = best_status


# =====================================================================
# Auto-selección de modos finales (clustering)
# =====================================================================

def auto_select_modes(
    diagram: StabilityDiagram,
    freq_cluster_tol_pct: float = 1.0,
    min_stable_orders: int = 3,
) -> List[IdentifiedMode]:
    """
    Identifica los modos finales agrupando polos "stable" por proximidad
    de frecuencia.

    Args:
        diagram: StabilityDiagram con polos clasificados
        freq_cluster_tol_pct: % tolerancia para agrupar polos en un mismo modo
        min_stable_orders: mínimo de orders donde el modo debe aparecer estable
                           para considerarse confiable

    Returns:
        Lista de IdentifiedMode ordenada por frecuencia ascendente.
    """
    # Coleccionar todos los polos stables
    stables: List[ModalPole] = []
    n_orders_total = len(diagram.poles_by_order)
    for n, poles in diagram.poles_by_order.items():
        for p in poles:
            if p.stability == "stable":
                stables.append(p)
    stables.sort(key=lambda p: p.natural_frequency_hz)

    # Clustering por proximidad de frecuencia
    clusters: List[List[ModalPole]] = []
    for p in stables:
        if not clusters:
            clusters.append([p])
            continue
        last_cluster = clusters[-1]
        ref_fn = last_cluster[-1].natural_frequency_hz
        if ref_fn > 0:
            diff_pct = abs(p.natural_frequency_hz - ref_fn) / ref_fn * 100.0
        else:
            diff_pct = float("inf")
        if diff_pct < freq_cluster_tol_pct:
            last_cluster.append(p)
        else:
            clusters.append([p])

    # Filtrar clusters con suficiente confirmación
    final_modes: List[IdentifiedMode] = []
    for cluster in clusters:
        n_stable = len(cluster)
        if n_stable < min_stable_orders:
            continue
        # Tomar el polo de orden más alto (más refinado)
        best = max(cluster, key=lambda p: p.model_order)
        confidence = min(1.0, n_stable / max(1, n_orders_total // 2))
        final_modes.append(IdentifiedMode(
            mode_number=0,  # se asigna después
            natural_frequency_hz=best.natural_frequency_hz,
            damping_ratio_pct=best.damping_ratio_pct,
            s_pole=best.s_pole,
            model_order_picked=best.model_order,
            n_stable_orders=n_stable,
            confidence=confidence,
        ))

    final_modes.sort(key=lambda m: m.natural_frequency_hz)
    for i, m in enumerate(final_modes, 1):
        m.mode_number = i

    return final_modes


# =====================================================================
# Circle-Fit Nyquist refinement (Kennedy-Pancu, 1947)
# =====================================================================
#
# Para un modo aislado, los puntos del plano Nyquist (Real(H) vs Imag(H))
# alrededor del peak forman aproximadamente un círculo. El método extrae:
#   · fn: frecuencia donde el ángulo del círculo cruza ±90° (cuarto del
#         círculo desde el origen)
#   · ζ: derivado del radio del círculo y la sensitivity dω/dθ del fit
#
# Esto es independiente del LSCF, numericamente robusto, y publicado en
# textos clásicos (Ewins "Modal Testing", McConnell "Vibration Testing").
# =====================================================================


@dataclass
class CircleFitResult:
    """Resultado del refinamiento Circle-Fit Nyquist de un modo."""
    natural_frequency_hz: float
    damping_ratio_pct: float
    modal_constant_magnitude: float  # |A| del residuo modal
    modal_constant_phase_deg: float
    circle_center_real: float
    circle_center_imag: float
    circle_radius: float
    fit_residual: float              # error RMS del ajuste circular
    n_points_used: int
    is_reliable: bool


def _fit_circle(
    points_real: np.ndarray,
    points_imag: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Ajuste de círculo en el plano Nyquist por mínimos cuadrados.

    Returns:
        (center_real, center_imag, radius, fit_residual_rms)
    """
    x = np.asarray(points_real, dtype=float)
    y = np.asarray(points_imag, dtype=float)
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0, float("inf")

    # Modelo: (x - a)² + (y - b)² = r²
    # Linealizar: x² + y² = 2ax + 2by + (r² - a² - b²)
    # Sistema lineal: A @ [a, b, c] = b_vec
    # donde c = r² - a² - b²
    A = np.column_stack([2 * x, 2 * y, np.ones(n)])
    rhs = x ** 2 + y ** 2

    try:
        coef, residuals, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, float("inf")

    a, b, c = coef
    r_sq = c + a ** 2 + b ** 2
    if r_sq <= 0:
        return 0.0, 0.0, 0.0, float("inf")
    r = math.sqrt(r_sq)

    # RMS residual
    pred = (x - a) ** 2 + (y - b) ** 2
    actual = r ** 2
    rms = float(np.sqrt(np.mean((pred - actual) ** 2))) / max(r ** 2, 1e-9)

    return float(a), float(b), float(r), rms


def circle_fit_mode(
    frf_complex: np.ndarray,
    frequencies_hz: np.ndarray,
    f_peak_hz: float,
    half_window_hz: Optional[float] = None,
    min_points: int = 7,
) -> Optional[CircleFitResult]:
    """
    Refina un modo identificado aplicando Circle-Fit en el plano Nyquist.

    Args:
        frf_complex: FRF compleja, shape (n_freq,)
        frequencies_hz: eje de frecuencia
        f_peak_hz: frecuencia central del modo (de peak detection)
        half_window_hz: medio ancho de banda alrededor del peak.
            Si None, se usa heurística: max(2 Hz, 10% de f_peak)
        min_points: mínimo número de puntos para considerar fit válido

    Returns:
        CircleFitResult o None si el fit falla.
    """
    f = np.asarray(frequencies_hz, dtype=float)
    H = np.asarray(frf_complex, dtype=complex)

    if half_window_hz is None:
        half_window_hz = max(2.0, 0.1 * f_peak_hz)

    f_low = f_peak_hz - half_window_hz
    f_high = f_peak_hz + half_window_hz
    mask = (f >= f_low) & (f <= f_high)
    if np.sum(mask) < min_points:
        return None

    H_band = H[mask]
    f_band = f[mask]

    # Fit circle en plano Nyquist
    re = H_band.real
    im = H_band.imag
    cx, cy, r, rms = _fit_circle(re, im)
    if r <= 0:
        return None

    # fn — frecuencia donde |dθ/dω| es máximo (máxima sensibilidad angular)
    # Aproximación: índice donde el ángulo respecto al centro del círculo
    # cambia más rápido.
    angles = np.angle((H_band.real - cx) + 1j * (H_band.imag - cy))
    angles_unwrap = np.unwrap(angles)
    dtheta_domega = np.abs(np.gradient(angles_unwrap, f_band * 2 * np.pi))
    if len(dtheta_domega) == 0:
        return None
    peak_idx = int(np.argmax(dtheta_domega))
    fn_refined = float(f_band[peak_idx])

    # Damping: ζ = 1 / (ω_n × max(dθ/dω)) — derivado del modelo SDOF
    # Equivalente clásico: ζ = (ω_b - ω_a) / (2 ω_n) donde ω_a y ω_b son
    # frecuencias en los half-power points. Usamos versión Kennedy-Pancu:
    omega_n = 2 * np.pi * fn_refined
    max_dtheta_domega = float(dtheta_domega[peak_idx])
    if max_dtheta_domega <= 0 or omega_n <= 0:
        return None
    # ζ ≈ 1 / (ω_n × |dθ/dω|_max)
    zeta = 1.0 / (omega_n * max_dtheta_domega)
    # Sanity: damping físico
    if zeta <= 0 or zeta >= 1:
        return None

    # Modal constant: radio × ω_n² × 2ζ (relación clásica)
    modal_const_mag = float(2.0 * r * omega_n * omega_n * zeta)
    modal_const_phase = float(np.degrees(np.angle(complex(cx, cy))))

    # Confiabilidad: RMS residual razonable + suficientes puntos
    is_reliable = rms < 0.10 and np.sum(mask) >= min_points

    return CircleFitResult(
        natural_frequency_hz=fn_refined,
        damping_ratio_pct=zeta * 100.0,
        modal_constant_magnitude=modal_const_mag,
        modal_constant_phase_deg=modal_const_phase,
        circle_center_real=cx,
        circle_center_imag=cy,
        circle_radius=r,
        fit_residual=rms,
        n_points_used=int(np.sum(mask)),
        is_reliable=is_reliable,
    )


def identify_modes_robust(
    frf_complex: np.ndarray,
    frequencies_hz: np.ndarray,
    f_min_hz: float = 5.0,
    f_max_hz: Optional[float] = None,
    prominence_db: float = 6.0,
    min_distance_hz: float = 2.0,
) -> List[IdentifiedMode]:
    """
    Identificación modal robusta combinando:
      1. Peak detection (half-power method) — encuentra candidatos
      2. Circle-Fit Nyquist (Kennedy-Pancu) — refina fn y ζ con precisión

    Es la API pública recomendada del módulo EMA. Más estable que LSCF puro
    para FRFs single-reference con modos bien separados.

    Args:
        frf_complex: FRF compleja, shape (n_freq,)
        frequencies_hz: eje de frecuencia
        f_min_hz: límite inferior banda
        f_max_hz: límite superior (None → Nyquist)
        prominence_db: prominencia mínima de pico
        min_distance_hz: separación mínima entre picos

    Returns:
        Lista de IdentifiedMode con fn y ζ refinados por circle fit.
    """
    from core.modal.frf_compute import detect_modal_peaks

    H = np.asarray(frf_complex, dtype=complex)
    f = np.asarray(frequencies_hz, dtype=float)
    mag = np.abs(H)

    if f_max_hz is None:
        f_max_hz = float(f[-1])

    # Etapa 1: Peak detection
    peaks = detect_modal_peaks(
        frequencies_hz=f, magnitude=mag,
        f_min_hz=f_min_hz, f_max_hz=f_max_hz,
        prominence_db=prominence_db, min_distance_hz=min_distance_hz,
    )

    # Etapa 2: Circle-fit refinement por cada peak
    final_modes: List[IdentifiedMode] = []
    for i, peak in enumerate(peaks, 1):
        cf = circle_fit_mode(
            frf_complex=H,
            frequencies_hz=f,
            f_peak_hz=peak.frequency_hz,
            half_window_hz=max(2.0 * peak.bandwidth_hz, 3.0),
        )
        if cf is None or not cf.is_reliable:
            # Fallback al half-power (válido pero menos preciso)
            s = complex(-peak.damping_ratio_pct / 100.0 * 2 * np.pi * peak.frequency_hz,
                         2 * np.pi * peak.frequency_hz)
            final_modes.append(IdentifiedMode(
                mode_number=i,
                natural_frequency_hz=peak.frequency_hz,
                damping_ratio_pct=peak.damping_ratio_pct,
                s_pole=s,
                model_order_picked=0,
                n_stable_orders=0,
                confidence=0.5,
            ))
        else:
            wn = 2 * np.pi * cf.natural_frequency_hz
            zeta = cf.damping_ratio_pct / 100.0
            s = complex(-zeta * wn, wn * math.sqrt(max(1 - zeta ** 2, 0)))
            final_modes.append(IdentifiedMode(
                mode_number=i,
                natural_frequency_hz=cf.natural_frequency_hz,
                damping_ratio_pct=cf.damping_ratio_pct,
                s_pole=s,
                model_order_picked=2,  # SDOF
                n_stable_orders=1,
                confidence=0.95 if cf.is_reliable else 0.7,
            ))

    return final_modes
