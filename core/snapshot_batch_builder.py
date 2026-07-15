"""
core.snapshot_batch_builder
============================

Construye payloads para los 4 snapshot types (Waveform, Spectrum, Orbit,
Tabular) a partir de los archivos CSV parseados en Load Data (Ciclo 23.82).

Permite que el especialista guarde TODA su corrida con UN solo click —
no tiene que ir a cada módulo de análisis a guardar individualmente.

Defaults conservadores que cubren el caso típico:
  • Spectrum: FFT con window Hanning, 1 avg, sin overlap (espectro
    instantáneo del waveform completo). Lines of resolution = N/2.
  • Orbit: detección automática de pares X/Y por filename heurístico
    (busca tokens "X"/"Y" o "1X"/"1Y" o sufijos en el sensor_label).
    Si no encuentra pairs, devuelve lista vacía (no falla).
  • Waveform: full time series + métricas estadísticas estándar.
  • Tabular: agregado de métricas Direct + 1X + 2X (extraídos del FFT).

API pública:

  build_all_snapshots_from_parsed_files(parsed_files, instance_id) →
      {
          "waveform": {"sensors_data": [...]},
          "spectrum": {"sensors_data": [...]},
          "orbit":    {"bearings_data": [...]},
          "tabular":  {"channels_data": [...]},
      }

  Cada subdict es kwargs listos para pasar a save_X_snapshot(instance_id, **kwargs).

  El caller decide qué tipos invocar (basado en checkboxes del UI).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================
# CONFIG
# =============================================================

SPECTRUM_WINDOW = "hanning"
SPECTRUM_DEFAULT_LINES = 1600   # power of 2 typical
TOP_PEAKS_TO_DETECT = 10

# Defaults razonables si no se detecta sampling rate del metadata
DEFAULT_SAMPLING_RATE_HZ = 1000.0


# =============================================================
# HELPERS
# =============================================================

def _safe_float(v) -> float:
    try:
        f = float(v)
        if f != f:  # NaN
            return 0.0
        return f
    except Exception:
        return 0.0


def _extract_sensor_label(parsed_file: Dict[str, Any]) -> str:
    """Heurística: sensor label desde metadata o filename."""
    meta = parsed_file.get("metadata", {}) or {}
    # 1) Channel/Sensor/Point en metadata (priority order).
    #    'Point Name' va ANTES que 'Variable': los CSV Bently traen
    #    'Point Name,1XD TURBINA DE' (la etiqueta útil) y
    #    'Variable,Disp Wf(128X/16revs).KPH TURBINA' (metadata técnica).
    for key in ("Sensor", "Channel", "Point Name", "Point", "Variable",
                "Location", "Tag"):
        val = meta.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    # 2) Filename sin extensión
    fname = parsed_file.get("file_name", "")
    if fname:
        stem = re.sub(r"\.(csv|CSV)$", "", fname)
        # Si tiene tokens reconocibles tipo "1XD", "3YD", devolverlos
        m = re.search(r"\b(\d+[XYZ][ADVHN]?)\b", stem)
        if m:
            return m.group(1)
        return stem
    return "unknown"


def _parse_rate_value(raw: Any) -> Optional[float]:
    """Parsea una frecuencia de muestreo respetando el PREFIJO DE UNIDAD.

    Bug real (v3.31.446): el parseo anterior hacía re.sub(r"[^\\d.]", "") y
    '2.56 kHz' quedaba en 2.56 Hz — un error de 1000× que dejaba el espectro
    comprimido junto a 0 Hz (SGT300B). Ahora kHz/MHz escalan y CPM/RPM se
    convierten a Hz.
    """
    s = str(raw or "").strip().lower()
    if not s:
        return None
    mult = 1.0
    if "khz" in s:
        mult = 1e3
    elif "mhz" in s:
        mult = 1e6
    elif "cpm" in s or "rpm" in s:
        mult = 1.0 / 60.0
    # Separador de miles "1,024" → "1024"; coma decimal "2,56" → "2.56"
    s_num = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s).replace(",", ".")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s_num)
    if not m:
        return None
    try:
        val = float(m.group(0))
    except ValueError:
        return None
    fs = val * mult
    return fs if fs > 0 else None


# Factores para llevar la columna de tiempo a SEGUNDOS. Los CSV Bently traen
# 'X-Axis Unit,ms' — el tiempo está en MILISEGUNDOS. Leerlo como segundos daba
# una fs 1000× menor (30.3 Hz en vez de 30.303 Hz para SGT300B) y el espectro
# quedaba aplastado contra 0 Hz. (v3.31.449 — causa raíz confirmada con CSV real.)
_TIME_UNIT_TO_SECONDS: Dict[str, float] = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "ms": 1e-3, "msec": 1e-3, "msecs": 1e-3,
    "millisecond": 1e-3, "milliseconds": 1e-3,
    "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "usec": 1e-6,
    "microsecond": 1e-6, "microseconds": 1e-6,
    "ns": 1e-9, "nsec": 1e-9,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
}


def _time_unit_scale(parsed_file: Dict[str, Any]) -> float:
    """Factor para convertir la columna de tiempo a SEGUNDOS, leyendo la
    unidad declarada por el propio CSV ('X-Axis Unit'). Determinista: sin
    heurísticas de umbrales. Si no está declarada, asume segundos (1.0)."""
    meta = parsed_file.get("metadata", {}) or {}
    for key in ("X-Axis Unit", "X Axis Unit", "X-Axis Units", "X Axis Units",
                "Time Unit", "Time Units"):
        v = meta.get(key)
        if v and isinstance(v, str):
            u = v.strip().lower()
            if u in _TIME_UNIT_TO_SECONDS:
                return _TIME_UNIT_TO_SECONDS[u]
    return 1.0


def _extract_sampling_rate(parsed_file: Dict[str, Any]) -> float:
    """Sampling rate desde metadata (unit-aware), sino calcula del time column
    HONRANDO la unidad de tiempo declarada en el CSV ('X-Axis Unit')."""
    meta = parsed_file.get("metadata", {}) or {}
    fs_meta: Optional[float] = None
    # OJO: 'Sample Speed' NO va acá — en los CSV Bently es la velocidad del EJE
    # (rpm), no la frecuencia de muestreo.
    for key in ("Sampling Rate", "Sample Rate", "Fs", "Sampling Frequency", "Sample Freq"):
        if meta.get(key) is not None:
            fs_meta = _parse_rate_value(meta.get(key))
            if fs_meta:
                break

    # Guarda de plausibilidad: ningún CSV de vibración se muestrea < 10 Hz.
    # Si la metadata da algo absurdo (unidad rara/mal escrita), la ignoramos y
    # estimamos del time column, que viene del dato real.
    if fs_meta is not None and fs_meta >= 10.0:
        return fs_meta

    df = parsed_file.get("dataframe")
    time_col = parsed_file.get("time_column")
    if df is not None and time_col and time_col in df.columns:
        try:
            t = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy()
            if len(t) > 2:
                # Llevar dt a SEGUNDOS usando la unidad declarada por el CSV
                # ('X-Axis Unit,ms' en Bently). Esta es la corrección REAL:
                # determinista, no depende de umbrales.
                dt = float(np.median(np.diff(t))) * _time_unit_scale(parsed_file)
                if dt > 0:
                    fs_time = 1.0 / dt
                    # Red de seguridad para CSV que NO declaran la unidad: si la
                    # fs sigue siendo absurda para vibración (< 200 Hz — el
                    # mínimo razonable, ya que 1X típico ya está en 50-250 Hz),
                    # el tiempo casi seguro venía en ms/µs sin declararlo.
                    _guard = 0
                    while fs_time < 200.0 and _guard < 3:
                        fs_time *= 1000.0
                        _guard += 1
                    return fs_time
        except Exception:
            pass

    if fs_meta is not None and fs_meta > 0:
        return fs_meta  # último recurso
    return DEFAULT_SAMPLING_RATE_HZ


def _extract_unit(parsed_file: Dict[str, Any]) -> str:
    """Unidad de amplitud desde metadata.

    v3.31.449 — se agregó la variante CON GUION ('Y-Axis Unit'), que es la que
    usan los CSV Bently reales (SGT300B trae 'Y-Axis Unit,µm'). Antes solo se
    buscaba 'Y Axis Unit' con espacio → la unidad se perdía y el espectro
    quedaba sin unidad ni clasificación de familia (prox/vel/acel).
    """
    meta = parsed_file.get("metadata", {}) or {}
    for key in ("Y-Axis Unit", "Y Axis Unit", "Y-Axis Units", "Y Axis Units",
                "Amplitude Unit", "Unit", "Units"):
        v = meta.get(key)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_timestamp(parsed_file: Dict[str, Any]) -> str:
    """Estampa de tiempo de CAPTURA de la medición, desde la metadata del CSV
    (fila 'Timestamp', ej. '2/27/2026 7:00:48 AM')."""
    meta = parsed_file.get("metadata", {}) or {}
    for key in ("Timestamp", "Time Stamp", "TimeStamp", "Date/Time",
                "DateTime", "Date", "Fecha"):
        v = meta.get(key)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_running_speed_rpm(parsed_files: List[Dict[str, Any]]) -> Optional[float]:
    """Velocidad de giro (rpm) DECLARADA por el CSV.

    Los CSV Bently traen 'Sample Speed,14049 rpm' — la velocidad real del eje.
    Usarla evita que la plataforma la adivine desde el pico dominante del
    espectro, que en máquinas con componente sub-síncrona fuerte apunta al pico
    equivocado y deja los cursores 1X/2X/3X mal ubicados (v3.31.450).

    Devuelve la primera velocidad físicamente plausible (60–60.000 rpm) o None.
    """
    for pf in parsed_files or []:
        meta = pf.get("metadata", {}) or {}
        for key in ("Sample Speed", "Running Speed", "Shaft Speed",
                    "Machine Speed", "Speed", "RPM"):
            v = meta.get(key)
            if v is None:
                continue
            m = re.search(r"[-+]?\d*\.?\d+", str(v).replace(",", ""))
            if not m:
                continue
            try:
                val = float(m.group(0))
            except ValueError:
                continue
            if 60.0 <= val <= 60000.0:
                return val
    return None


def _extract_signal(parsed_file: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (time_arr, values_arr) como numpy arrays float64."""
    df = parsed_file["dataframe"]
    t_col = parsed_file["time_column"]
    v_col = parsed_file["vibration_column"]
    t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(v)
    return t[mask], v[mask]


# =============================================================
# WAVEFORM PAYLOAD
# =============================================================

def _compute_waveform_metrics(values: np.ndarray) -> Dict[str, float]:
    """Métricas estándar: peak, p2p, rms, crest, kurtosis."""
    if len(values) == 0:
        return {"peak": 0.0, "peak_to_peak": 0.0, "rms": 0.0,
                "crest_factor": 0.0, "kurtosis": 0.0}
    abs_max = float(np.max(np.abs(values)))
    p2p = float(np.max(values) - np.min(values))
    rms = float(np.sqrt(np.mean(values ** 2)))
    crest = abs_max / rms if rms > 1e-12 else 0.0
    # Kurtosis (excess) Fisher-Pearson
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma < 1e-12:
        kurt = 0.0
    else:
        kurt = float(np.mean(((values - mu) / sigma) ** 4) - 3.0)
    return {
        "peak": abs_max,
        "peak_to_peak": p2p,
        "rms": rms,
        "crest_factor": crest,
        "kurtosis": kurt,
    }


def build_waveform_payload(parsed_files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construye sensors_data para save_waveform_snapshot."""
    sensors_data: List[Dict[str, Any]] = []
    for pf in parsed_files:
        if not pf.get("is_valid"):
            continue
        try:
            t, v = _extract_signal(pf)
            if len(v) < 2:
                continue
            fs = _extract_sampling_rate(pf)
            duration = float(len(v)) / fs if fs > 0 else 0.0
            sensors_data.append({
                "sensor_label": _extract_sensor_label(pf),
                "csv_file": pf.get("file_name", ""),
                "csv_timestamp": _extract_timestamp(pf),
                "sampling_rate_hz": fs,
                "duration_sec": duration,
                "n_samples_raw": int(len(v)),
                "time": t.tolist(),
                "values": v.tolist(),
                "unit": _extract_unit(pf),
                "metrics": _compute_waveform_metrics(v),
            })
        except Exception:
            continue
    return {"sensors_data": sensors_data}


# =============================================================
# SPECTRUM PAYLOAD (FFT inline)
# =============================================================

def _compute_fft(values: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """FFT con window Hanning, devuelve (freqs_hz, amplitudes)."""
    n = len(values)
    if n < 4 or fs <= 0:
        return np.array([]), np.array([])
    # Window Hanning
    window = np.hanning(n)
    windowed = values * window
    # Compensation por window (Hanning amplitude correction)
    amp_correction = 2.0 / np.sum(window)
    # FFT positive half
    fft_vals = np.fft.rfft(windowed)
    amps = np.abs(fft_vals) * amp_correction
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, amps


def _find_peaks(freqs: np.ndarray, amps: np.ndarray, top_n: int = TOP_PEAKS_TO_DETECT) -> List[Dict[str, Any]]:
    """Top-N peaks detectados (sin scipy.signal para reducir deps)."""
    if len(amps) < 3:
        return []
    # Local maxima: amp[i] > amp[i-1] and amp[i] > amp[i+1]
    local_max_idx = np.where(
        (amps[1:-1] > amps[:-2]) & (amps[1:-1] > amps[2:])
    )[0] + 1
    if len(local_max_idx) == 0:
        return []
    # Ordenar por amplitud desc, tomar top N
    sorted_idx = local_max_idx[np.argsort(amps[local_max_idx])[::-1]][:top_n]
    peaks = []
    for i in sorted(sorted_idx):  # devolver ordenados por freq asc
        peaks.append({
            "freq": float(freqs[i]),
            "amp": float(amps[i]),
            "label": "",
        })
    return peaks


def build_spectrum_payload(parsed_files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construye sensors_data para save_spectrum_snapshot."""
    sensors_data: List[Dict[str, Any]] = []
    for pf in parsed_files:
        if not pf.get("is_valid"):
            continue
        try:
            _, v = _extract_signal(pf)
            if len(v) < 4:
                continue
            fs = _extract_sampling_rate(pf)
            freqs, amps = _compute_fft(v, fs)
            if len(freqs) == 0:
                continue
            sensors_data.append({
                "sensor_label": _extract_sensor_label(pf),
                "csv_file": pf.get("file_name", ""),
                "csv_timestamp": _extract_timestamp(pf),
                "amp_unit": _extract_unit(pf),
                "freq_unit": "Hz",
                "sampling_rate_hz": fs,
                "freq_span_hz": float(freqs[-1]) if len(freqs) else 0.0,
                "lines_of_resolution": int(len(freqs)),
                "window": "Hanning",
                "n_avg": 1,
                "overlap_pct": 0.0,
                "freqs": freqs.tolist(),
                "amps": amps.tolist(),
                "peaks": _find_peaks(freqs, amps),
            })
        except Exception:
            continue
    return {"sensors_data": sensors_data}


# =============================================================
# ORBIT PAYLOAD (detección automática de pares X/Y)
# =============================================================

def _extract_axis_and_bearing_key(label: str) -> Optional[Tuple[str, str]]:
    """Extrae (bearing_key, axis) de un sensor label.

    Soporta múltiples formatos comunes:
      "1XD"        → ("1", "X")
      "3YD"        → ("3", "Y")
      "2X"         → ("2", "X")
      "VE5808 (Y)" → ("VE5808", "Y")
      "VE5808-X"   → ("VE5808", "X")
      "TES1-3XD"   → ("TES1-3", "X")
      "Bearing 1 X"→ ("Bearing 1", "X")
      "5808X"      → ("5808", "X")

    Estrategia: busca un X o Y "axis-like" en el label y devuelve el resto
    como bearing_key.
    """
    label = (label or "").strip()
    if not label:
        return None

    # Pattern 1: "(X)" o "(Y)" al final → bearing_key = todo lo previo trimmeado
    m = re.search(r"^(.*?)\s*\(([XY])\)\s*$", label, re.IGNORECASE)
    if m:
        key = m.group(1).strip()
        if key:
            return (key, m.group(2).upper())

    # Pattern 1b: token compacto "<plano><X|Y><tipo>" EN CUALQUIER PARTE del
    # label, no solo al final. Cubre el formato real del cliente donde el eje
    # va al inicio y la ubicación después: "3YD GENERADOR DE" → ("3","Y"),
    # "4XD GENERADOR NDE" → ("4","X"), "1XD TURBINA DE" → ("1","X"). El
    # bearing_key es el número de plano, así 3XD + 3YD emparejan.
    m = re.search(r"\b(\d+)([XY])[ADVHN]?\b", label.upper())
    if m:
        return (m.group(1), m.group(2))

    # Pattern 2: token tipo "1XD", "3YD", "2X", "5808Y" → (digits, X/Y)
    m = re.match(r"^(.+?)([XY])(?:[ADVHN])?\s*$", label.upper())
    if m:
        key = m.group(1).strip().rstrip("-_ ")
        if key:
            return (key, m.group(2))

    # Pattern 3: X/Y como sufijo separado por -/_/espacio → "BRG-3-X"
    m = re.match(r"^(.+?)[\-_ ]([XY])(?:[ADVHN])?\s*$", label, re.IGNORECASE)
    if m:
        return (m.group(1).strip(), m.group(2).upper())

    return None


def _detect_xy_pairs(parsed_files: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    """Detecta pares X/Y por sensor_label. Devuelve [(x_pf, y_pf, bearing_label)].

    Estrategia en 2 pasos:
      Paso 1 — match exacto: mismo bearing_key con X y Y.
               Ej. "3XD" + "3YD" → ("3", X) + ("3", Y) → par.
      Paso 2 — consecutivos: para sensores que quedaron solos, si sus
               bearing_keys terminan en dígitos y son números consecutivos
               (n, n+1) con ejes complementarios, también es par. Convención
               común en plantas: par 5807Y + 5808X = mismo cojinete físico,
               solo que cada canal tiene su propio número de tag.
    """
    pairs = []

    # Clasificar todos los signals por (bearing_key, axis)
    by_key: Dict[str, List[Tuple[Dict[str, Any], str]]] = {}
    for pf in parsed_files:
        if not pf.get("is_valid"):
            continue
        label = _extract_sensor_label(pf)
        parsed = _extract_axis_and_bearing_key(label)
        if parsed is None:
            continue
        bearing_key, axis = parsed
        by_key.setdefault(bearing_key, []).append((pf, axis))

    # Paso 1: match exacto por bearing_key
    unpaired: List[Tuple[str, Dict[str, Any], str]] = []
    for bearing_key, items in by_key.items():
        axes_present = {ax for _, ax in items}
        if "X" in axes_present and "Y" in axes_present:
            x_pf = next(pf for pf, ax in items if ax == "X")
            y_pf = next(pf for pf, ax in items if ax == "Y")
            pairs.append((x_pf, y_pf, f"BRG {bearing_key}"))
        else:
            for pf, ax in items:
                unpaired.append((bearing_key, pf, ax))

    # Paso 2: consecutive matching para unpaired (mismo cojinete, números
    # de canal correlativos como 5807Y + 5808X, 5809Y + 5810X)
    def _numeric_tail(key: str) -> Optional[int]:
        m = re.search(r"(\d+)\s*$", key or "")
        return int(m.group(1)) if m else None

    sortable = []
    for bk, pf, ax in unpaired:
        tail = _numeric_tail(bk)
        if tail is not None:
            sortable.append((tail, bk, pf, ax))

    sortable.sort(key=lambda t: t[0])

    used_indices: set = set()
    for i, (tail_a, bk_a, pf_a, ax_a) in enumerate(sortable):
        if i in used_indices:
            continue
        for j in range(i + 1, len(sortable)):
            if j in used_indices:
                continue
            tail_b, bk_b, pf_b, ax_b = sortable[j]
            # Números consecutivos y ejes complementarios
            if abs(tail_b - tail_a) == 1 and ax_a != ax_b:
                x_pf = pf_a if ax_a == "X" else pf_b
                y_pf = pf_b if ax_a == "X" else pf_a
                # Label combinado: "BRG VE5807+VE5808"
                pair_label = f"BRG {bk_a}+{bk_b}"
                pairs.append((x_pf, y_pf, pair_label))
                used_indices.add(i)
                used_indices.add(j)
                break

    return pairs


def _vector_at_frequency(freqs: np.ndarray, amps: np.ndarray, target_hz: float,
                        tolerance_pct: float = 5.0) -> Tuple[float, float]:
    """Devuelve (amp, phase) en target_hz ± tolerance.
    Phase no la tenemos sin FFT complejo — devolvemos 0.0 por ahora.
    """
    if len(freqs) == 0:
        return (0.0, 0.0)
    tol = target_hz * (tolerance_pct / 100.0)
    mask = (freqs >= target_hz - tol) & (freqs <= target_hz + tol)
    if not mask.any():
        return (0.0, 0.0)
    idx_in_mask = np.argmax(amps[mask])
    masked_indices = np.where(mask)[0]
    real_idx = masked_indices[idx_in_mask]
    return (float(amps[real_idx]), 0.0)  # phase TODO


def build_orbit_payload(parsed_files: List[Dict[str, Any]],
                        rotational_speed_rpm: Optional[float] = None) -> Dict[str, Any]:
    """Construye bearings_data para save_orbit_snapshot.

    Detecta pares X/Y automáticamente. Sin pares → bearings_data vacío.
    """
    pairs = _detect_xy_pairs(parsed_files)
    bearings_data: List[Dict[str, Any]] = []

    fundamental_hz = (rotational_speed_rpm / 60.0) if rotational_speed_rpm else None

    for x_pf, y_pf, bearing_label in pairs:
        try:
            _, x_v = _extract_signal(x_pf)
            _, y_v = _extract_signal(y_pf)
            if len(x_v) < 4 or len(y_v) < 4:
                continue
            n = min(len(x_v), len(y_v))
            x_v = x_v[:n]
            y_v = y_v[:n]
            fs = _extract_sampling_rate(x_pf)

            entry = {
                "bearing_label": bearing_label,
                "x_sensor_label": _extract_sensor_label(x_pf),
                "y_sensor_label": _extract_sensor_label(y_pf),
                "x_csv_file": x_pf.get("file_name", ""),
                "y_csv_file": y_pf.get("file_name", ""),
                "amp_unit": _extract_unit(x_pf),
                "x_values": x_v.tolist(),
                "y_values": y_v.tolist(),
            }

            # Vectores 1X y 2X si conocemos la frecuencia rotacional
            if fundamental_hz and fundamental_hz > 0:
                fx, ax = _compute_fft(x_v, fs)
                fy, ay = _compute_fft(y_v, fs)
                amp_1x_x, _ = _vector_at_frequency(fx, ax, fundamental_hz)
                amp_1x_y, _ = _vector_at_frequency(fy, ay, fundamental_hz)
                amp_2x_x, _ = _vector_at_frequency(fx, ax, 2.0 * fundamental_hz)
                amp_2x_y, _ = _vector_at_frequency(fy, ay, 2.0 * fundamental_hz)
                entry["vector_1x"] = {
                    "amp_x": amp_1x_x, "amp_y": amp_1x_y,
                    "phase_x_deg": 0.0, "phase_y_deg": 0.0,
                }
                entry["vector_2x"] = {
                    "amp_x": amp_2x_x, "amp_y": amp_2x_y,
                    "phase_x_deg": 0.0, "phase_y_deg": 0.0,
                }

            bearings_data.append(entry)
        except Exception:
            continue

    return {"bearings_data": bearings_data}


# =============================================================
# TABULAR PAYLOAD
# =============================================================

def build_tabular_payload(parsed_files: List[Dict[str, Any]],
                          rotational_speed_rpm: Optional[float] = None) -> Dict[str, Any]:
    """Construye channels_data para save_tabular_snapshot."""
    fundamental_hz = (rotational_speed_rpm / 60.0) if rotational_speed_rpm else None
    channels_data: List[Dict[str, Any]] = []

    for pf in parsed_files:
        if not pf.get("is_valid"):
            continue
        try:
            _, v = _extract_signal(pf)
            if len(v) < 4:
                continue
            metrics = _compute_waveform_metrics(v)
            fs = _extract_sampling_rate(pf)
            unit = _extract_unit(pf)

            entry = {
                "sensor_label": _extract_sensor_label(pf),
                "csv_file": pf.get("file_name", ""),
                "direct": metrics["rms"],
                "direct_unit": unit,
                "vector_1x_amp": 0.0,
                "vector_1x_phase": 0.0,
                "vector_1x_unit": unit,
                "vector_2x_amp": 0.0,
                "vector_2x_phase": 0.0,
                "vector_2x_unit": unit,
                "severity": "",
                "iso_zone": "",
                "api_compliance": False,
            }

            if fundamental_hz and fundamental_hz > 0:
                freqs, amps = _compute_fft(v, fs)
                amp_1x, _ = _vector_at_frequency(freqs, amps, fundamental_hz)
                amp_2x, _ = _vector_at_frequency(freqs, amps, 2.0 * fundamental_hz)
                entry["vector_1x_amp"] = amp_1x
                entry["vector_2x_amp"] = amp_2x

            channels_data.append(entry)
        except Exception:
            continue

    return {"channels_data": channels_data}


# =============================================================
# FACADE — build all
# =============================================================

def _parse_measured_dt(ts_str: str):
    """Parsea la estampa de tiempo del CSV a datetime (varios formatos)."""
    from datetime import datetime as _dt
    s = (ts_str or "").strip()
    if not s:
        return None
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %I:%M %p", "%m/%d/%Y %H:%M",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S"):
        try:
            return _dt.strptime(s, fmt)
        except Exception:  # noqa: BLE001
            continue
    return None


def group_parsed_files_by_measurement(
    parsed_files: List[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Agrupa los archivos por FECHA/HORA de medición (csv_timestamp, redondeada
    al minuto) para que corridas de fechas distintas NO se mezclen en un mismo
    snapshot. Antes, subir CSVs del 06/07 y del 13/07 juntos los fusionaba en un
    solo snapshot con la fecha más reciente (bug de campo v3.31.442).

    Devuelve [(label, [files]), ...] ordenado por fecha ascendente. Los archivos
    sin timestamp parseable van a un único grupo 'sin fecha'.
    """
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    order: List[Any] = []
    for pf in parsed_files:
        dt = _parse_measured_dt(_extract_timestamp(pf))
        key = (dt.year, dt.month, dt.day, dt.hour, dt.minute) if dt else None
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(pf)

    def _sortkey(k):
        return (1,) if k is None else (0, k)

    result: List[Tuple[str, List[Dict[str, Any]]]] = []
    for key in sorted(order, key=_sortkey):
        if key is None:
            label = "sin fecha"
        else:
            label = "%02d/%02d/%04d %02d:%02d" % (
                key[2], key[1], key[0], key[3], key[4])
        result.append((label, groups[key]))
    return result


def build_all_snapshots_from_parsed_files(
    parsed_files: List[Dict[str, Any]],
    rotational_speed_rpm: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Construye los 4 payloads listos para save_*_snapshot().

    El caller decide qué tipos guardar (con checkboxes en el UI):

        payloads = build_all_snapshots_from_parsed_files(parsed_files)
        if save_waveform:
            save_waveform_snapshot(instance_id, **payloads["waveform"],
                                   corrida_label=label, notes=notes)
        if save_spectrum:
            save_spectrum_snapshot(instance_id, **payloads["spectrum"], ...)
        ...
    """
    return {
        "waveform": build_waveform_payload(parsed_files),
        "spectrum": build_spectrum_payload(parsed_files),
        "orbit":    build_orbit_payload(parsed_files, rotational_speed_rpm),
        "tabular":  build_tabular_payload(parsed_files, rotational_speed_rpm),
    }


__all__ = [
    "build_waveform_payload",
    "build_spectrum_payload",
    "build_orbit_payload",
    "build_tabular_payload",
    "build_all_snapshots_from_parsed_files",
    "group_parsed_files_by_measurement",
    "extract_running_speed_rpm",
]
