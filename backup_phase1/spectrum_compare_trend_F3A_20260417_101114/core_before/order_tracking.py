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
    - signal_obj Watermelon: signal.x
    - dict con 'x' o 'data'
    - objetos con .x, .data o .values
    - numpy array / list
    """
    if signal is None:
        raise ValueError("Signal is None")

    candidate = signal

    if hasattr(signal, "x"):
        candidate = signal.x
    elif hasattr(signal, "data"):
        candidate = signal.data
    elif hasattr(signal, "values"):
        candidate = signal.values
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


def _extract_metadata(signal):
    if hasattr(signal, "metadata") and isinstance(signal.metadata, dict):
        return signal.metadata
    if isinstance(signal, dict) and isinstance(signal.get("metadata"), dict):
        return signal["metadata"]
    return {}


def _extract_rpm(signal, default_rpm=3600.0):
    metadata = _extract_metadata(signal)

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

    # Muchos CSV vienen en ms
    if dt_median > 1e-3:
        dt_seconds = dt_median / 1000.0
    else:
        dt_seconds = dt_median

    if dt_seconds <= 0:
        return None

    return float(1.0 / dt_seconds)


def _wrap_360(angle_deg):
    return np.mod(angle_deg, 360.0)


def _circular_mean_deg(angles_deg):
    angles_deg = np.asarray(angles_deg, dtype=float)
    if angles_deg.size == 0:
        return 0.0
    z = np.mean(np.exp(1j * np.radians(angles_deg)))
    return float(_wrap_360(np.degrees(np.angle(z))))


def _circular_std_deg(angles_deg):
    angles_deg = np.asarray(angles_deg, dtype=float)
    if angles_deg.size == 0:
        return 0.0
    z = np.mean(np.exp(1j * np.radians(angles_deg)))
    r = float(np.abs(z))
    r = float(np.clip(r, 1e-12, 1.0))
    return float(np.degrees(np.sqrt(-2.0 * np.log(r))))


def _fit_order_per_revolution(x_rev, order):
    """
    Ajuste robusto por orden:
        x(theta) ≈ a*cos(order*theta) + b*sin(order*theta) + c
    """
    n_revs, samples_per_rev = x_rev.shape
    theta = 2.0 * np.pi * np.arange(samples_per_rev) / samples_per_rev

    cos_t = np.cos(order * theta)
    sin_t = np.sin(order * theta)
    ones = np.ones(samples_per_rev)

    design = np.column_stack([cos_t, sin_t, ones])
    pinv = np.linalg.pinv(design)

    coeffs = (pinv @ x_rev.T).T
    a = coeffs[:, 0]
    b = coeffs[:, 1]

    amp_pk = np.sqrt(a * a + b * b)
    amp_pp = 2.0 * amp_pk
    phase_deg = _wrap_360(np.degrees(np.arctan2(b, a)))

    return {
        "amp_pp_per_rev": amp_pp,
        "phase_deg_per_rev": phase_deg,
        "a_coef": a,
        "b_coef": b,
    }


def analyze_order_tracking(signal, fs=None, rpm=None, max_order=5):
    x = _extract_signal_array(signal)

    if rpm is None:
        rpm = _extract_rpm(signal, default_rpm=3600.0)
    else:
        rpm = float(rpm)

    if fs is None:
        fs = _infer_fs_from_time(signal)
        if fs is None:
            fs = 10000.0
    else:
        fs = float(fs)

    if rpm <= 0:
        raise ValueError("RPM must be > 0")
    if fs <= 0:
        raise ValueError("Sampling frequency must be > 0")

    samples_per_rev = int(round(fs / (rpm / 60.0)))
    if samples_per_rev < 16:
        raise ValueError("Invalid samples_per_rev. Check fs/rpm/timebase.")

    n_revs = len(x) // samples_per_rev
    if n_revs < 2:
        raise ValueError("Not enough revolutions for order tracking")

    usable_samples = n_revs * samples_per_rev
    x_use = x[:usable_samples]
    x_rev = x_use.reshape((n_revs, samples_per_rev))

    # Remover DC por revolución
    x_rev = x_rev - np.mean(x_rev, axis=1, keepdims=True)

    orders = list(range(1, int(max_order) + 1))
    order_results = {}

    for order in orders:
        fit = _fit_order_per_revolution(x_rev, order)
        amps = np.asarray(fit["amp_pp_per_rev"], dtype=float)
        phases = np.asarray(fit["phase_deg_per_rev"], dtype=float)

        order_results[order] = {
            "amp_pp_per_rev": amps,
            "phase_deg_per_rev": phases,
            "mean_amp_pp": float(np.mean(amps)),
            "max_amp_pp": float(np.max(amps)),
            "mean_phase_deg": float(_circular_mean_deg(phases)),
            "phase_stability_deg": float(_circular_std_deg(phases)),
        }

    dominant_order = max(
        orders,
        key=lambda o: order_results[o]["mean_amp_pp"]
    )

    rev_idx = np.arange(1, n_revs + 1)
    rev_period_ms = (60.0 / rpm) * 1000.0
    time_axis_ms = np.linspace(0.0, rev_period_ms, samples_per_rev, endpoint=False)

    debug = {
        "fs_used": float(fs),
        "rpm_used": float(rpm),
        "samples_per_rev": int(samples_per_rev),
        "n_revs": int(n_revs),
        "usable_samples": int(usable_samples),
        "dominant_order": int(dominant_order),
    }

    return {
        "mode": "geometry_sync_order_tracking",
        "orders": orders,
        "order_results": order_results,
        "dominant_order": int(dominant_order),
        "x_rev": x_rev,
        "time_axis_ms": time_axis_ms,
        "rev_idx": rev_idx,
        "samples_per_rev": int(samples_per_rev),
        "n_revs": int(n_revs),
        "mean_rpm": float(rpm),
        "debug": debug,
    }