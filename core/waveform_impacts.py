import numpy as np


def detect_impacts(signal, fs=None):
    """
    Detecta impactos en una señal de vibración.
    - threshold dinámico basado en RMS
    - retorna índices y métricas
    """

    if signal is None:
        return {}

    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 10:
        return {}

    rms = np.sqrt(np.mean(x**2))
    threshold = 3.5 * rms  # configurable

    peaks_idx = np.where(np.abs(x) > threshold)[0]

    if len(peaks_idx) == 0:
        return {
            "count": 0,
            "threshold": float(threshold),
            "indices": []
        }

    # evitar duplicados cercanos (cluster de impacto)
    cleaned = [peaks_idx[0]]
    for i in peaks_idx[1:]:
        if i - cleaned[-1] > 5:
            cleaned.append(i)

    return {
        "count": len(cleaned),
        "threshold": float(threshold),
        "indices": cleaned
    }


def detect_impacts_batch(signals: dict) -> dict:
    results = {}

    for name, payload in signals.items():
        try:
            y = None

            if isinstance(payload, dict):
                y = payload.get("y")

            if y is None:
                continue

            results[name] = detect_impacts(y)

        except Exception:
            results[name] = {}

    return results
