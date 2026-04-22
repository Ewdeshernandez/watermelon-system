import numpy as np


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return default
    except Exception:
        return default


def _compute_skewness(x: np.ndarray) -> float:
    n = x.size
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std <= 0:
        return 0.0
    z = (x - mean) / std
    return _safe_float(np.mean(z ** 3), 0.0)


def _compute_kurtosis(x: np.ndarray) -> float:
    n = x.size
    if n < 4:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std <= 0:
        return 0.0
    z = (x - mean) / std
    return _safe_float(np.mean(z ** 4), 0.0)  # kurtosis clásica, normal ~= 3


def compute_waveform_metrics(signal) -> dict:
    if signal is None:
        return {}

    x = np.asarray(signal, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {}

    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    peak_to_peak = np.max(x) - np.min(x)
    mean = np.mean(x)
    std = np.std(x)
    crest_factor = peak / rms if rms > 0 else 0.0
    skewness = _compute_skewness(x)
    kurtosis = _compute_kurtosis(x)

    return {
        "rms": _safe_float(rms),
        "peak": _safe_float(peak),
        "crest_factor": _safe_float(crest_factor),
        "peak_to_peak": _safe_float(peak_to_peak),
        "mean": _safe_float(mean),
        "std": _safe_float(std),
        "skewness": _safe_float(skewness),
        "kurtosis": _safe_float(kurtosis),
        "samples": int(x.size),
    }


def _extract_signal_array(signal_payload):
    if signal_payload is None:
        return None

    if isinstance(signal_payload, np.ndarray):
        return signal_payload

    if isinstance(signal_payload, (list, tuple)):
        return np.asarray(signal_payload, dtype=float)

    if isinstance(signal_payload, dict):
        for key in ("y", "values", "signal", "waveform", "data", "amplitude"):
            if key in signal_payload and signal_payload[key] is not None:
                return np.asarray(signal_payload[key], dtype=float)

    return None


def compute_metrics_batch(signals: dict) -> dict:
    results = {}

    if not isinstance(signals, dict):
        return results

    for name, payload in signals.items():
        try:
            arr = _extract_signal_array(payload)
            if arr is None:
                continue

            metrics = compute_waveform_metrics(arr)
            if metrics:
                results[name] = metrics
        except Exception:
            continue

    return results
