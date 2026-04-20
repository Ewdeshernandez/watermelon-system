import numpy as np


def _to_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _extract_signal_array(signal):
    """
    Soporta:
    - signal_obj real de Watermelon: signal.x
    - dict con 'x' o 'data'
    - objetos con .x, .data o .values
    - numpy array / list
    """
    if signal is None:
        raise ValueError("Signal is None")

    candidate = signal

    # Caso Watermelon real
    if hasattr(signal, "x"):
        candidate = signal.x

    # Otros objetos
    elif hasattr(signal, "data"):
        candidate = signal.data

    elif hasattr(signal, "values"):
        candidate = signal.values

    # Dicts
    elif isinstance(signal, dict):
        if "x" in signal:
            candidate = signal["x"]
        elif "data" in signal:
            candidate = signal["data"]
        elif "values" in signal:
            candidate = signal["values"]

    arr = np.asarray(candidate, dtype=float).flatten()

    if arr.size == 0:
        raise ValueError("Signal array is empty")

    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        raise ValueError("Signal array has no valid finite samples")

    return arr


def _extract_rpm(signal, default_rpm=3600.0):
    """
    Prioridad:
    1) signal.metadata
    2) signal dict
    3) default
    """
    metadata = {}

    if hasattr(signal, "metadata") and isinstance(signal.metadata, dict):
        metadata = signal.metadata
    elif isinstance(signal, dict):
        metadata = signal.get("metadata", {}) if isinstance(signal.get("metadata", {}), dict) else {}

    rpm_keys = [
        "Sample Speed",
        "sample_speed",
        "RPM",
        "rpm",
        "speed_rpm",
        "machine_speed_rpm",
    ]

    for key in rpm_keys:
        if key in metadata:
            rpm = _to_float(metadata.get(key))
            if rpm is not None and rpm > 0:
                return float(rpm)

    return float(default_rpm)


def _extract_time_array(signal):
    """
    Intenta sacar time para inferir fs si existe.
    """
    candidate = None

    if hasattr(signal, "time"):
        candidate = signal.time
    elif isinstance(signal, dict) and "time" in signal:
        candidate = signal["time"]

    if candidate is None:
        return None

    try:
        t = np.asarray(candidate, dtype=float).flatten()
        t = t[np.isfinite(t)]
        if t.size < 2:
            return None
        return t
    except Exception:
        return None


def _infer_fs_from_time(signal):
    t = _extract_time_array(signal)
    if t is None or len(t) < 2:
        return None

    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]

    if dt.size == 0:
        return None

    dt_median = float(np.median(dt))

    # En tus CSV suele venir en ms
    if dt_median > 1e-3:
        dt_seconds = dt_median / 1000.0
    else:
        dt_seconds = dt_median

    if dt_seconds <= 0:
        return None

    return float(1.0 / dt_seconds)


def _safe_corrcoef(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size != b.size or a.size < 2:
        return 0.0

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0

    try:
        c = np.corrcoef(a, b)[0, 1]
        if np.isnan(c) or np.isinf(c):
            return 0.0
        return float(c)
    except Exception:
        return 0.0


def analyze_tsa(signal, fs=None, rpm=None):
    """
    TSA robusto y compatible con tu pages/12_TSA.py
    """
    # =========================
    # INPUT HANDLING
    # =========================
    x = _extract_signal_array(signal)

    if rpm is None:
        rpm = _extract_rpm(signal, default_rpm=3600.0)
    else:
        rpm = float(rpm)

    if fs is None:
        fs = _infer_fs_from_time(signal)
        if fs is None:
            # fallback razonable, pero prioriza time si existe
            fs = 10000.0
    else:
        fs = float(fs)

    if rpm <= 0:
        raise ValueError("RPM must be > 0")

    if fs <= 0:
        raise ValueError("Sampling frequency must be > 0")

    # =========================
    # GEOMETRY
    # =========================
    samples_per_rev = int(round(fs / (rpm / 60.0)))

    if samples_per_rev < 8:
        raise ValueError("Invalid samples_per_rev. Check fs/rpm/timebase.")

    n_revs = len(x) // samples_per_rev

    if n_revs < 2:
        raise ValueError("Not enough revolutions for TSA")

    usable_samples = n_revs * samples_per_rev
    trimmed = x[:usable_samples]

    try:
        x_rev = trimmed.reshape((n_revs, samples_per_rev))
    except Exception as e:
        raise ValueError(f"Reshape failed: {e}")

    # Remover DC por revolución
    x_rev = x_rev - np.mean(x_rev, axis=1, keepdims=True)

    # =========================
    # TSA CORE
    # =========================
    tsa_mean = np.mean(x_rev, axis=0)
    tsa_min = np.min(x_rev, axis=0)
    tsa_max = np.max(x_rev, axis=0)

    # =========================
    # RESIDUAL
    # =========================
    residual = x_rev - tsa_mean[np.newaxis, :]

    residual_rms = float(np.sqrt(np.mean(residual ** 2)))
    sync_rms = float(np.sqrt(np.mean(tsa_mean ** 2)))

    total_rms = float(np.sqrt(np.mean(x_rev ** 2)))
    if total_rms <= 1e-12:
        sync_ratio = 0.0
    else:
        sync_ratio = float(sync_rms / total_rms)

    # =========================
    # CORRELATION
    # =========================
    correlations = [_safe_corrcoef(tsa_mean, rev) for rev in x_rev]
    mean_corr = float(np.mean(correlations)) if correlations else 0.0

    # =========================
    # TIME AXIS
    # =========================
    rev_period_ms = (60.0 / rpm) * 1000.0
    time_axis_ms = np.linspace(0.0, rev_period_ms, samples_per_rev, endpoint=False)

    # =========================
    # METRICS
    # =========================
    peak_to_peak_tsa = float(np.max(tsa_mean) - np.min(tsa_mean))

    # =========================
    # DEBUG
    # =========================
    debug = {
        "sync_source": "geometry_from_fs_rpm",
        "header_number_of_revs": int(n_revs),
        "inferred_revs_from_time": int(n_revs),
        "variable_hint_samples_per_rev": int(samples_per_rev),
        "variable_hint_revs": int(n_revs),
        "candidate_table": [],
        "usable_samples": int(usable_samples),
        "signal_length": int(len(x)),
        "fs_used": float(fs),
        "rpm_used": float(rpm),
    }

    # =========================
    # OUTPUT COMPATIBLE
    # =========================
    return {
        "x_rev": x_rev,
        "tsa_mean": tsa_mean,
        "tsa_min": tsa_min,
        "tsa_max": tsa_max,
        "residual_waveforms": residual,
        "time_axis_ms": time_axis_ms,
        "n_revs": int(n_revs),
        "samples_per_rev": int(samples_per_rev),
        "sync_rms": sync_rms,
        "residual_rms": residual_rms,
        "sync_ratio": sync_ratio,
        "mean_corr": mean_corr,
        "peak_to_peak_tsa": peak_to_peak_tsa,
        "mean_rpm": float(rpm),
        "mode": "geometry_sync",
        "debug": debug,
    }