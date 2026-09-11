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

def _efdd_mode_shape(fdd_result: FDDResult, idx_full: int, sv1: np.ndarray,
                     mac_min: float = 0.80, max_lines: int = 40) -> np.ndarray:
    """EFDD — refina la forma modal promediando la **campana SDOF** alrededor del pico:
    toma las líneas de frecuencia cuyo primer vector singular tiene MAC ≥ mac_min con el
    del pico, las alinea en fase y las promedia ponderadas por SV1. Resultado: una forma
    más colineal (menos dispersa) → complejidad más baja y confiable. Cae al vector del
    pico (FDD clásico) si algo falla."""
    U = fdd_result.mode_shapes_at_freq                      # (N, N, n_freq)
    n_freq = U.shape[2]
    phi0 = np.asarray(U[:, 0, idx_full], dtype=complex)
    if np.vdot(phi0, phi0).real <= 1e-30:
        return phi0

    def _mac(a, b):
        den = np.vdot(a, a).real * np.vdot(b, b).real
        return float(abs(np.vdot(a, b)) ** 2 / den) if den > 1e-30 else 0.0

    acc = phi0 * float(sv1[idx_full])
    for step in (-1, 1):
        k = idx_full + step; used = 0
        while 0 <= k < n_freq and used < max_lines:
            phik = np.asarray(U[:, 0, k], dtype=complex)
            if _mac(phi0, phik) < mac_min:
                break
            proj = np.vdot(phi0, phik)                      # alinear fase con el pico
            if abs(proj) > 1e-30:
                phik = phik * (np.conjugate(proj) / abs(proj))
            acc = acc + phik * float(sv1[k])
            k += step; used += 1
    nrm = np.sqrt(np.vdot(acc, acc).real)
    return (acc / nrm) if nrm > 1e-30 else phi0


def _efdd_sdof_damping(fdd_result: FDDResult, idx_full: int, sv1: np.ndarray,
                       freq: np.ndarray, mac_min: float = 0.80, max_lines: int = 120):
    """EFDD clásico (Brincker 2001) — damping por DECREMENTO LOGARÍTMICO de la
    autocorrelación de la campana SDOF, y frecuencia por cruces por cero (no half-power).

      1) Campana SDOF = SV1 en las líneas cuyo 1er vector singular tiene MAC≥mac_min con el
         del pico (0 fuera) → aísla el modo de 1 GDL.
      2) IRFFT de la campana (PSD de un solo lado, Hermitiana) → función de autocorrelación
         SDOF r(τ): una sinusoide amortiguada de decaimiento exponencial.
      3) Extremos de r(τ) → regresión de ln|r_k| vs t_k; pendiente = -ζ·ω_n (log-dec).
      4) Cruces por cero de r(τ) → frecuencia amortiguada f_d → f_n = f_d/√(1-ζ²).

    Devuelve (fn_hz, zeta_pct) o None si la campana es pobre o el ajuste no es físico
    (entonces el llamador cae al damping half-power). dt = 1/fs (fs real de la corrida).
    """
    U = fdd_result.mode_shapes_at_freq
    n = U.shape[2]
    if not (0 <= idx_full < n):
        return None
    phi0 = np.asarray(U[:, 0, idx_full], dtype=complex)
    if np.vdot(phi0, phi0).real <= 1e-30:
        return None

    def _mac(a, b):
        den = np.vdot(a, a).real * np.vdot(b, b).real
        return float(abs(np.vdot(a, b)) ** 2 / den) if den > 1e-30 else 0.0

    lo = idx_full; used = 0
    while lo - 1 >= 0 and used < max_lines and _mac(phi0, U[:, 0, lo - 1]) >= mac_min:
        lo -= 1; used += 1
    hi = idx_full; used = 0
    while hi + 1 < n and used < max_lines and _mac(phi0, U[:, 0, hi + 1]) >= mac_min:
        hi += 1; used += 1
    if hi - lo < 4:
        return None                                 # campana demasiado angosta
    bell = np.zeros(n, float)
    bell[lo:hi + 1] = np.maximum(np.asarray(sv1[lo:hi + 1], float), 0.0)
    R = np.fft.irfft(bell)                          # autocorrelación (real, par)
    if R.size < 8 or R[0] <= 0:
        return None
    R = R / R[0]
    df = float(freq[1] - freq[0]) if len(freq) > 1 else 1.0
    dt = 1.0 / (2.0 * (n - 1) * df)                 # = 1/fs
    nuse = R.size // 2
    r = R[:nuse]; t = np.arange(nuse) * dt
    dr = np.diff(r)
    ext = np.where(np.sign(dr[:-1]) != np.sign(dr[1:]))[0] + 1
    ext = np.array([k for k in ext if abs(r[k]) > 0.12], dtype=int)   # envolvente sobre ruido
    if ext.size < 3:
        return None
    tk = t[ext]; yk = np.log(np.abs(r[ext]))
    A = np.vstack([np.ones_like(tk), tk]).T
    slope = float(np.linalg.lstsq(A, yk, rcond=None)[0][1])
    if slope >= 0:
        return None                                 # no decae → no físico
    zc = np.where(np.sign(r[:-1]) != np.sign(r[1:]))[0]
    if zc.size >= 3:
        kk = np.arange(zc.size, dtype=float)
        Az = np.vstack([np.ones_like(kk), kk]).T
        half_T = float(np.linalg.lstsq(Az, t[zc], rcond=None)[0][1])
        fd = 1.0 / (2.0 * half_T) if half_T > 1e-9 else float(freq[idx_full])
    else:
        fd = float(freq[idx_full])
    wn = 2.0 * np.pi * fd
    if wn <= 0:
        return None
    zeta = -slope / wn
    if not (1e-4 < zeta < 0.20):                    # 0.01%..20% (físico)
        return None
    fn = fd / np.sqrt(max(1e-9, 1.0 - zeta * zeta))
    return float(fn), float(zeta * 100.0)


def detect_oma_modes(
    fdd_result: FDDResult,
    f_min_hz: float = 5.0,
    f_max_hz: Optional[float] = None,
    prominence_db: float = 6.0,
    min_distance_hz: float = 2.0,
    running_speed_hz: Optional[float] = None,
    harmonic_tol_pct: float = 0.5,
    use_efdd: bool = True,
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
        # --- Interpolación sub-bin (parábola por 3 puntos en dB) ---------------
        # find_peaks devuelve el CENTRO del bin FFT → la frecuencia queda cuantizada
        # a df (~0.39 Hz). Ajustando una parábola a σ1(dB) en [idx-1, idx, idx+1] se
        # estima el vértice real del pico → frecuencia con resolución sub-bin (como
        # ARTeMIS / cualquier estimador paramétrico), sin cambiar la forma modal ni
        # el índice usado aguas abajo.
        if 0 < idx_full < len(sv1_db) - 1:
            _ym1, _y0, _yp1 = float(sv1_db[idx_full - 1]), float(sv1_db[idx_full]), float(sv1_db[idx_full + 1])
            _den = _ym1 - 2.0 * _y0 + _yp1
            if _den < 0.0:                      # pico cóncavo hacia abajo (máximo real)
                _delta = 0.5 * (_ym1 - _yp1) / _den
                if -1.0 < _delta < 1.0:
                    fn = float(freq[idx_full] + _delta * df)
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

        # Mode shape: EFDD (promedio de la campana SDOF) o FDD clásico (una línea)
        if use_efdd:
            try:
                mode_shape = _efdd_mode_shape(fdd_result, idx_full, sv1)
            except Exception:  # noqa: BLE001  — cualquier fallo → FDD de una línea
                mode_shape = fdd_result.mode_shapes_at_freq[:, 0, idx_full]
            # Damping EFDD por decremento logarítmico de la autocorrelación SDOF (Brincker):
            # más preciso que half-power. Si la campana no lo permite, se conserva half-power.
            # NOTA: sólo se toma el DAMPING; la frecuencia se deja en el pico sub-bin (más
            # precisa que la de cruces por cero, que sesga en campanas angostas/asimétricas).
            try:
                _ed = _efdd_sdof_damping(fdd_result, idx_full, sv1, freq)
                if _ed is not None:
                    damping_pct = _ed[1]
            except Exception:  # noqa: BLE001
                pass
        else:
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
    use_efdd: bool = True,
) -> FDDResult:
    """
    Pipeline OMA completo: FDD + detección automática de modos.

    `use_efdd`: si True (por defecto), la forma modal se refina con EFDD (promedio de la
    campana SDOF) → complejidad más baja/limpia. Poner False para FDD clásico de una línea.

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
        use_efdd=use_efdd,
    )
    return result


def kurtosis_harmonic_indicator(time_data: np.ndarray, sample_rate_hz: float,
                                freqs_hz, bw_hz: float = 1.5):
    """Indicador de ARMÓNICOS por KURTOSIS (Brincker & Andersen, OMA).

    Un armónico determinístico (sinusoide de una máquina en giro) filtrado en banda tiene
    una densidad de probabilidad tipo ARCOSENO → kurtosis ≈ 1.5 (< Gaussiana = 3). Una
    respuesta ESTRUCTURAL (banda angosta aleatoria) es ~Gaussiana → kurtosis ≈ 3. Así se
    distingue un pico armónico de un modo real SIN depender de la velocidad de giro.

    Devuelve lista de dicts {freq, kurtosis, is_harmonic} (is_harmonic si kurtosis < 2.4).
    """
    try:
        from scipy.signal import butter, sosfiltfilt
    except Exception:  # noqa: BLE001
        return [{"freq": float(f), "kurtosis": float("nan"), "is_harmonic": False} for f in freqs_hz]
    x = np.asarray(time_data, float)
    if x.ndim == 1:
        x = x[:, None]
    ref = x[:, int(np.argmax(np.var(x, axis=0)))]                # canal de mayor energía
    ref = ref - float(np.mean(ref))
    nyq = float(sample_rate_hz) / 2.0
    out = []
    for f0 in freqs_hz:
        f0 = float(f0)
        lo = max(0.5, f0 - bw_hz) / nyq
        hi = min(nyq * 0.999, f0 + bw_hz) / nyq
        k = float("nan"); harm = False
        if 0.0 < lo < hi < 1.0:
            try:
                sos = butter(4, [lo, hi], btype="band", output="sos")
                y = sosfiltfilt(sos, ref)
                y = y - float(np.mean(y)); s = float(np.std(y))
                if s > 1e-12:
                    k = float(np.mean((y / s) ** 4))             # kurtosis (Gauss=3, arcsin≈1.5)
                    harm = k < 2.4
            except Exception:  # noqa: BLE001
                pass
        out.append({"freq": f0, "kurtosis": k, "is_harmonic": harm})
    return out


def reduce_harmonics_sv(freqs, sv, harmonic_freqs, bw_hz: float = 1.5) -> np.ndarray:
    """Reduce (remueve) picos ARMÓNICOS en las curvas de valores singulares: reemplaza las
    líneas dentro de ±bw_hz de cada frecuencia armónica por interpolación lineal de los
    niveles vecinos de banda ancha → revela el modo estructural debajo del armónico, sin
    tocar el resto del espectro. Acepta SV 1-D (sólo SV1) o 2-D (todas las curvas)."""
    freqs = np.asarray(freqs, float); sv = np.asarray(sv, float)
    single = sv.ndim == 1
    if single:
        sv = sv[None, :]
    out = sv.copy()
    for hf in harmonic_freqs:
        band = np.abs(freqs - float(hf)) <= bw_hz
        if not band.any():
            continue
        idx = np.where(band)[0]
        i0, i1 = idx[0] - 1, idx[-1] + 1
        if i0 < 0 or i1 >= len(freqs):
            continue
        for r in range(out.shape[0]):
            out[r, idx] = np.interp(freqs[idx], [freqs[i0], freqs[i1]], [out[r, i0], out[r, i1]])
    return out[0] if single else out


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
