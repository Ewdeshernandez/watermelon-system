"""
core/modal/oma_engine.py — Motor de Análisis Modal Operacional (OMA)
=====================================================================

Implementación nativa de FDD (Frequency Domain Decomposition) — el método
clásico de OMA para identificar modos naturales sin necesidad de martillo
modal, usando solo respuestas durante operación normal.

Por qué FDD y no SSI en V1
--------------------------
FDD (Brincker, Zhang, Andersen 2001) es el método OMA más usado en
industria por:
  · Simplicidad implementacional (solo SVD por frecuencia)
  · Robustez numérica
  · Resultados visuales claros (singular value curves)
  · Fácil interpretación

SSI (Stochastic Subspace Identification) da damping más preciso pero es
significativamente más complejo y queda para V2.

Algoritmo FDD (ISO 20816 + Brincker 2001)
-----------------------------------------
1. Capturar tiempo continuo de N sensores sincronizados (60-300 seg)
2. Para cada par (i, j) de canales:
     Computar Sxy_ij(f) = cross-spectral density (Welch)
3. Construir PSD matrix S_y(f) de shape (N, N, n_freq), hermitiana
4. Por cada frecuencia f_k:
     SVD: S_y(f_k) = U_k Σ_k V_k^H
     Singular values σ_1(f_k) ≥ σ_2(f_k) ≥ ...
5. La curva σ_1(f) muestra picos en los modos naturales del sistema
6. Para cada pico fn:
     · fn = frecuencia del peak en σ_1(f)
     · ζ ≈ bandwidth_3dB / (2·fn) (half-power sobre el primer SV)
     · Mode shape φ_fn = primer singular vector U_k[:, 0]

Caveat sobre modos armónicos forzados
-------------------------------------
OMA captura TANTO modos naturales COMO excitaciones forzadas (1×, 2× rpm).
La detección automática asume que si un peak está a múltiplo entero exacto
de running speed (tolerance ±0.5%), es harmonic — flag is_harmonic=True.
Esto se valida con el sensor de phase reference si está disponible.

Norma aplicable
---------------
ISO 20816 — Evaluación de vibraciones en máquinas en operación
ISO 7626-6 secc. 6.4 — Identificación output-only / OMA
Brincker, Zhang, Andersen 2001 — paper original FDD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import numpy as np


@dataclass
class OMAMode:
    """Modo identificado por OMA."""
    mode_number: int
    natural_frequency_hz: float
    damping_ratio_pct: float
    mode_shape: np.ndarray  # vector complejo (N_channels,)
    singular_value_peak: float
    bandwidth_3db_hz: float
    is_harmonic: bool = False
    harmonic_order: Optional[int] = None  # 1, 2, 3... × running speed
    confidence: float = 1.0
    complexity_pct: float = 0.0  # 0 = real (natural), 100 = totalmente complejo (espurio)
    classification: str = "natural"  # "natural" | "harmonic" | "spurious"


def modal_complexity_mpc(mode_shape: np.ndarray) -> float:
    """
    Calcula el Modal Phase Collinearity (MPC) y devuelve la complejidad en %.

    MPC (Pappa & Eishan 1995) mide qué tan colineales son las fases del mode
    shape complejo. Un modo natural real (damping proporcional, sistema
    estable) tiene fases colineales → MPC ≈ 1 → complejidad ≈ 0%.

    Un "modo" que es realmente una armónica forzada o ruido tiene fases
    aleatorias → MPC ≈ 0 → complejidad ≈ 100%.

    Fórmula:
      S = [[Σ Re², Σ Re·Im], [Σ Re·Im, Σ Im²]]
      eigvals λ₁ ≥ λ₂ ≥ 0
      MPC = ((λ₁ - λ₂) / (λ₁ + λ₂))²
      complexity_pct = (1 - MPC) × 100

    Returns:
        Complejidad en porcentaje (0 = puro real, 100 = fully complex)
    """
    phi = np.asarray(mode_shape, dtype=complex).flatten()
    if phi.size == 0:
        return 0.0
    re = np.real(phi)
    im = np.imag(phi)

    sxx = float((re * re).sum())
    syy = float((im * im).sum())
    sxy = float((re * im).sum())

    tr = sxx + syy
    if tr < 1e-12:
        return 0.0

    det = sxx * syy - sxy ** 2
    discr = max(tr ** 2 / 4.0 - det, 0.0)
    lambda1 = tr / 2.0 + math.sqrt(discr)
    lambda2 = tr / 2.0 - math.sqrt(discr)

    if (lambda1 + lambda2) < 1e-12:
        return 0.0
    mpc = ((lambda1 - lambda2) / (lambda1 + lambda2)) ** 2
    complexity_pct = max(0.0, min(100.0, (1.0 - mpc) * 100.0))
    return float(complexity_pct)


def classify_mode(
    natural_frequency_hz: float,
    complexity_pct: float,
    running_speed_hz: Optional[float] = None,
    harmonic_tol_pct: float = 0.5,
    complexity_natural_threshold: float = 40.0,
    complexity_spurious_threshold: float = 75.0,
) -> Tuple[str, bool, Optional[int]]:
    """
    Clasifica un modo en natural / harmonic / spurious usando 2 criterios:
      1. Complejidad modal (MPC) — > 75% = espurio
      2. Coincidencia con armónicas de running speed (si se da)

    Args:
        natural_frequency_hz: fn identificada
        complexity_pct: complejidad MPC (0-100)
        running_speed_hz: velocidad operativa para detectar armónicas
        harmonic_tol_pct: tolerancia para clasificar como armónica
        complexity_natural_threshold: < este valor → claramente natural
        complexity_spurious_threshold: > este valor → claramente espurio/harmonic

    Returns:
        (classification, is_harmonic, harmonic_order)
        classification: "natural" | "harmonic" | "spurious"
    """
    is_harmonic = False
    harmonic_order: Optional[int] = None
    if running_speed_hz and running_speed_hz > 0:
        for n in range(1, 16):
            expected = n * running_speed_hz
            if expected <= 0:
                continue
            diff_pct = abs(natural_frequency_hz - expected) / expected * 100.0
            if diff_pct < harmonic_tol_pct:
                is_harmonic = True
                harmonic_order = n
                break

    # Clasificación:
    # - Coincide con un orden de giro (k×RPM) → HARMONIC (criterio primario). Las
    #   componentes forzadas están fase-bloqueadas y suelen tener BAJA complejidad,
    #   así que NO se puede exigir complejidad alta para llamarlas armónico.
    # - Complejidad muy alta sin coincidencia → spurious.
    # - Resto → natural (modo estructural).
    if is_harmonic:
        classification = "harmonic"
    elif complexity_pct >= complexity_spurious_threshold:
        classification = "spurious"
    else:
        classification = "natural"

    return classification, is_harmonic, harmonic_order


@dataclass
class FDDResult:
    """Resultado del análisis FDD."""
    frequencies_hz: np.ndarray
    singular_values: np.ndarray  # shape (N_channels, n_freq) — todos los SVs
    mode_shapes_at_freq: np.ndarray  # (N_channels, N_channels, n_freq) — todos los U
    channel_names: List[str] = field(default_factory=list)
    sample_rate_hz: float = 0.0
    duration_s: float = 0.0
    n_segments: int = 1
    nperseg: int = 0
    modes: List[OMAMode] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return self.singular_values.shape[0]

    def first_singular_value(self) -> np.ndarray:
        """Primer singular value en función de frecuencia — donde aparecen los modos."""
        return self.singular_values[0, :]

    def first_sv_db(self) -> np.ndarray:
        """Primer singular value en dB para visualización."""
        return 10.0 * np.log10(np.maximum(self.first_singular_value(), 1e-30))


# =====================================================================
# FDD Core
# =====================================================================

def _build_psd_matrix(
    time_data: np.ndarray,
    sample_rate_hz: float,
    nperseg: int,
    noverlap: Optional[int],
    window: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Construye la matriz PSD S_y[i,j](f) entre todos los pares de canales.

    Returns:
        (frequencies_hz, S_matrix, n_segments)
        S_matrix shape: (n_ch, n_ch, n_freq)
    """
    try:
        from scipy.signal import csd
    except ImportError as exc:
        raise ImportError("scipy es requerido para FDD") from exc

    n_samples, n_ch = time_data.shape
    if noverlap is None:
        noverlap = nperseg // 2

    # Primera CSD para obtener tamaño de frecuencia
    f, S00 = csd(time_data[:, 0], time_data[:, 0],
                  fs=sample_rate_hz, nperseg=nperseg,
                  noverlap=noverlap, window=window)
    n_freq = len(f)
    n_segments = max(1, (n_samples - noverlap) // (nperseg - noverlap))

    # Matriz hermitiana
    S = np.zeros((n_ch, n_ch, n_freq), dtype=complex)
    S[0, 0, :] = S00
    for i in range(n_ch):
        for j in range(i, n_ch):
            if i == 0 and j == 0:
                continue
            _, Sij = csd(time_data[:, i], time_data[:, j],
                         fs=sample_rate_hz, nperseg=nperseg,
                         noverlap=noverlap, window=window)
            S[i, j, :] = Sij
            if i != j:
                S[j, i, :] = np.conj(Sij)

    return f, S, n_segments


def run_fdd(
    time_data: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 4096,
    noverlap: Optional[int] = None,
    window: str = "hann",
    channel_names: Optional[List[str]] = None,
) -> FDDResult:
    """
    Frequency Domain Decomposition.

    Args:
        time_data: Matriz (N_samples, N_channels) con señales temporales
        sample_rate_hz: Frecuencia de muestreo
        nperseg: Tamaño del segmento para Welch
        noverlap: Solape (default nperseg//2)
        window: Función de ventana
        channel_names: Etiquetas opcionales de cada canal

    Returns:
        FDDResult con singular values y mode shapes por frecuencia
    """
    data = np.asarray(time_data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_ch = data.shape
    if n_ch < 1:
        raise ValueError("Al menos 1 canal requerido para FDD")

    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(n_ch)]

    f, S, n_segments = _build_psd_matrix(
        data, sample_rate_hz, nperseg, noverlap, window
    )
    n_freq = S.shape[2]

    # SVD por frecuencia
    singular_values = np.zeros((n_ch, n_freq))
    mode_shapes = np.zeros((n_ch, n_ch, n_freq), dtype=complex)
    for k in range(n_freq):
        try:
            U, sv, _ = np.linalg.svd(S[:, :, k], full_matrices=False)
            singular_values[:, k] = sv
            mode_shapes[:, :, k] = U
        except np.linalg.LinAlgError:
            continue

    return FDDResult(
        frequencies_hz=f,
        singular_values=singular_values,
        mode_shapes_at_freq=mode_shapes,
        channel_names=list(channel_names),
        sample_rate_hz=float(sample_rate_hz),
        duration_s=float(n_samples / sample_rate_hz),
        n_segments=n_segments,
        nperseg=nperseg,
    )


# =====================================================================
# Detección de picos en first singular value
# =====================================================================

def detect_oma_modes(
    fdd_result: FDDResult,
    f_min_hz: float = 5.0,
    f_max_hz: Optional[float] = None,
    prominence_db: float = 6.0,
    min_distance_hz: float = 2.0,
    running_speed_hz: Optional[float] = None,
    harmonic_tol_pct: float = 0.5,
) -> List[OMAMode]:
    """
    Detecta modos OMA picos en el first singular value.

    Args:
        fdd_result: Resultado del run_fdd
        f_min_hz, f_max_hz: Banda de búsqueda
        prominence_db: Prominencia mínima del pico
        min_distance_hz: Separación mínima entre picos
        running_speed_hz: Si se proporciona, marca picos cercanos como armónicos
        harmonic_tol_pct: Tolerancia para clasificar como armónico

    Returns:
        Lista de OMAMode ordenada por frecuencia.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError("scipy requerido") from exc

    freq = fdd_result.frequencies_hz
    sv1 = fdd_result.first_singular_value()
    sv1_db = fdd_result.first_sv_db()

    if f_max_hz is None:
        f_max_hz = float(freq[-1])

    band_mask = (freq >= f_min_hz) & (freq <= f_max_hz)
    freq_band = freq[band_mask]
    sv1_db_band = sv1_db[band_mask]

    df = float(freq[1] - freq[0]) if len(freq) > 1 else 1.0
    distance_samples = max(1, int(round(min_distance_hz / df)))

    peak_indices_band, _ = find_peaks(
        sv1_db_band, prominence=prominence_db, distance=distance_samples,
    )

    modes: List[OMAMode] = []
    for idx_band in peak_indices_band:
        fn = float(freq_band[idx_band])
        # Index en array completo
        idx_full = int(np.argmin(np.abs(freq - fn)))
        sv_peak = float(sv1[idx_full])
        sv_peak_db = float(sv1_db[idx_full])

        # Half-power bandwidth en SV1 (no en magnitud H, pero conceptualmente igual)
        target_db = sv_peak_db - 3.0
        # Hacia la izquierda
        f1 = fn
        for i in range(idx_full, -1, -1):
            if sv1_db[i] <= target_db:
                f1 = float(freq[i])
                break
        # Hacia la derecha
        f2 = fn
        for i in range(idx_full, len(sv1_db)):
            if sv1_db[i] <= target_db:
                f2 = float(freq[i])
                break
        bw = max(f2 - f1, 1e-9)
        damping_pct = bw / (2.0 * fn) * 100.0

        # Mode shape: primer singular vector en esta frecuencia
        mode_shape = fdd_result.mode_shapes_at_freq[:, 0, idx_full]

        # Modal Complexity (MPC) — criterio Artemis para natural vs harmonic
        complexity_pct = modal_complexity_mpc(mode_shape)

        # Clasificación combinada: complexity + harmonic match
        classification, is_harmonic, harmonic_order = classify_mode(
            natural_frequency_hz=fn,
            complexity_pct=complexity_pct,
            running_speed_hz=running_speed_hz,
            harmonic_tol_pct=harmonic_tol_pct,
        )

        # Confianza basada en clasificación
        if classification == "natural":
            conf = 0.95 if complexity_pct < 20 else 0.80
        elif classification == "harmonic":
            conf = 0.40  # confiable como harmonic, no como modo natural
        else:  # spurious
            conf = 0.15

        modes.append(OMAMode(
            mode_number=0,  # se asigna después
            natural_frequency_hz=fn,
            damping_ratio_pct=damping_pct,
            mode_shape=mode_shape,
            singular_value_peak=sv_peak,
            bandwidth_3db_hz=bw,
            is_harmonic=is_harmonic,
            harmonic_order=harmonic_order,
            confidence=conf,
            complexity_pct=complexity_pct,
            classification=classification,
        ))

    modes.sort(key=lambda m: m.natural_frequency_hz)
    for i, m in enumerate(modes, 1):
        m.mode_number = i

    return modes


def run_oma(
    time_data: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 4096,
    channel_names: Optional[List[str]] = None,
    f_min_hz: float = 5.0,
    f_max_hz: Optional[float] = None,
    prominence_db: float = 6.0,
    min_distance_hz: float = 2.0,
    running_speed_hz: Optional[float] = None,
) -> FDDResult:
    """
    Pipeline OMA completo: FDD + detección automática de modos.

    Returns:
        FDDResult con .modes poblado.
    """
    result = run_fdd(time_data, sample_rate_hz, nperseg=nperseg,
                      channel_names=channel_names)
    result.modes = detect_oma_modes(
        result,
        f_min_hz=f_min_hz, f_max_hz=f_max_hz,
        prominence_db=prominence_db,
        min_distance_hz=min_distance_hz,
        running_speed_hz=running_speed_hz,
    )
    return result


def compute_mac_matrix(modes: List[OMAMode]) -> np.ndarray:
    """
    Compute Modal Assurance Criterion (MAC) matrix entre todos los modos.

    MAC(i, j) = |φ_i^H · φ_j|² / ((φ_i^H · φ_i) (φ_j^H · φ_j))

    Interpretación:
      · MAC = 1 → mode shapes idénticos (mismo modo identificado dos veces)
      · MAC = 0 → completamente ortogonales (linealmente independientes)
      · MAC > 0.7 off-diagonal → modos REDUNDANTES (eliminar uno)

    Norma aplicable:
      · ISO 7626-6 secc. 6.5 — Validación entre orders consecutivos
      · API 684 secc. 1.6 — Correlación EMA vs FEA

    Args:
        modes: lista de OMAMode con mode_shape complejo

    Returns:
        Matriz cuadrada (N, N) con valores MAC entre cada par de modos.
        Diagonal siempre = 1.0
    """
    n = len(modes)
    mac = np.zeros((n, n))
    for i in range(n):
        phi_i = np.asarray(modes[i].mode_shape, dtype=complex).flatten()
        for j in range(n):
            phi_j = np.asarray(modes[j].mode_shape, dtype=complex).flatten()
            num = abs(np.vdot(phi_i, phi_j)) ** 2  # |φ_i^H · φ_j|²
            denom = float(np.vdot(phi_i, phi_i).real * np.vdot(phi_j, phi_j).real)
            mac[i, j] = num / max(denom, 1e-30)
    return mac


def compute_cross_mac(
    modes_a: List[OMAMode],
    modes_b: List[OMAMode],
) -> np.ndarray:
    """
    Cross-MAC entre dos sets de modos (e.g. EMA vs OMA, o experimental vs FEA).

    MAC alto en la diagonal → mismos modos físicos identificados por
    diferentes métodos. Validación cruzada esencial bajo API 684.

    Args:
        modes_a, modes_b: dos listas de OMAMode

    Returns:
        Matriz (len(a), len(b)) con MAC cruzado.
    """
    na = len(modes_a)
    nb = len(modes_b)
    mac = np.zeros((na, nb))
    for i in range(na):
        phi_i = np.asarray(modes_a[i].mode_shape, dtype=complex).flatten()
        for j in range(nb):
            phi_j = np.asarray(modes_b[j].mode_shape, dtype=complex).flatten()
            if phi_i.size != phi_j.size:
                continue
            num = abs(np.vdot(phi_i, phi_j)) ** 2
            denom = float(np.vdot(phi_i, phi_i).real * np.vdot(phi_j, phi_j).real)
            mac[i, j] = num / max(denom, 1e-30)
    return mac


def detect_redundant_modes(
    modes: List[OMAMode],
    threshold: float = 0.7,
) -> List[Tuple[int, int, float]]:
    """
    Identifica pares de modos con MAC off-diagonal > threshold (redundantes).

    Returns:
        Lista de tuplas (idx_i, idx_j, mac_value) — pares de índices que
        son linealmente dependientes (probable duplicación).
    """
    mac = compute_mac_matrix(modes)
    n = mac.shape[0]
    duplicates = []
    for i in range(n):
        for j in range(i + 1, n):
            if mac[i, j] > threshold:
                duplicates.append((i, j, float(mac[i, j])))
    return duplicates


def detect_harmonic_modes(
    modes: List[OMAMode],
    operating_rpm: float,
    tolerance_pct: float = 1.0,
) -> List[OMAMode]:
    """
    Post-hoc: marca como is_harmonic=True los modos cuya frecuencia coincide
    con armónicas de la velocidad de operación.
    """
    running_hz = operating_rpm / 60.0
    for m in modes:
        for n in range(1, 11):
            expected = n * running_hz
            if expected <= 0:
                continue
            diff_pct = abs(m.natural_frequency_hz - expected) / expected * 100.0
            if diff_pct < tolerance_pct:
                m.is_harmonic = True
                m.harmonic_order = n
                m.confidence = min(m.confidence, 0.3)
                break
    return modes
