import re
from types import SimpleNamespace

import numpy as np


FILTER_MODE_MAP = {
    "RAW": "RAW",
    "DIRECT": "RAW",
    "1X": "1X",
    "2X": "2X",
}


# =========================================================
# SAFE HELPERS
# =========================================================
def _safe_metadata(signal_obj):
    metadata = getattr(signal_obj, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _safe_filename(signal_obj, fallback="Signal"):
    value = getattr(signal_obj, "file_name", None)
    return str(value) if value else fallback


def _parse_first_float(value):
    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None

    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if not match:
        return None

    try:
        numeric = float(match.group(0))
        return numeric if np.isfinite(numeric) else None
    except Exception:
        return None


def _parse_first_int(value):
    numeric = _parse_first_float(value)
    if numeric is None:
        return None
    rounded = int(round(numeric))
    return rounded if rounded > 0 else None


def _metadata_lookup(metadata, aliases):
    if not metadata:
        return None

    normalized = {str(k).strip().lower(): v for k, v in metadata.items()}
    aliases_norm = [str(a).strip().lower() for a in aliases]

    for alias in aliases_norm:
        if alias in normalized:
            return normalized[alias]

    for k, v in normalized.items():
        for alias in aliases_norm:
            if alias in k:
                return v

    return None


def _pairwise_finite(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _triplet_finite(a, b, c):
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    return a[mask], b[mask], c[mask]


def _pkpk(arr):
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def _clean_channel_name(file_name):
    text = str(file_name).strip()
    text = re.sub(r"\.csv$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bREV\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# SIGNAL SYNC
# =========================================================
def synchronize_signals(signal_x, signal_y):
    tx = np.asarray(getattr(signal_x, "time", []), dtype=float).reshape(-1)
    xx = np.asarray(getattr(signal_x, "x", []), dtype=float).reshape(-1)

    ty = np.asarray(getattr(signal_y, "time", []), dtype=float).reshape(-1)
    yy = np.asarray(getattr(signal_y, "x", []), dtype=float).reshape(-1)

    nx = min(tx.size, xx.size)
    ny = min(ty.size, yy.size)

    tx = tx[:nx]
    xx = xx[:nx]
    ty = ty[:ny]
    yy = yy[:ny]

    tx, xx = _pairwise_finite(tx, xx)
    ty, yy = _pairwise_finite(ty, yy)

    if tx.size < 8 or ty.size < 8:
        raise ValueError("Not enough valid samples to synchronize X/Y signals.")

    start = max(float(tx[0]), float(ty[0]))
    end = min(float(tx[-1]), float(ty[-1]))
    if end <= start:
        raise ValueError("X/Y signals do not overlap in time.")

    dt_x = float(np.median(np.diff(tx)))
    dt_y = float(np.median(np.diff(ty)))
    dt = min(dt_x, dt_y)

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid sample interval detected.")

    t = np.arange(start, end + 0.5 * dt, dt, dtype=float)
    if t.size < 8:
        raise ValueError("Synchronization produced too few samples.")

    x = np.interp(t, tx, xx)
    y = np.interp(t, ty, yy)

    t, x, y = _triplet_finite(t, x, y)
    if t.size < 8:
        raise ValueError("Not enough synchronized samples after interpolation.")

    return SimpleNamespace(
        time=t,
        x=x,
        y=y,
        dt=float(np.median(np.diff(t))),
        metadata_x=_safe_metadata(signal_x),
        metadata_y=_safe_metadata(signal_y),
        file_x=_safe_filename(signal_x, "X Signal"),
        file_y=_safe_filename(signal_y, "Y Signal"),
    )


# =========================================================
# INFERENCE FROM CSV HEADERS / METADATA
# =========================================================
def infer_rpm_from_metadata(metadata_x, metadata_y):
    aliases = [
        "rpm",
        "speed",
        "running speed",
        "shaft speed",
        "rotational speed",
        "machine speed",
        "sample speed",
        "1x speed",
        "cpm",
    ]

    for metadata in (metadata_x, metadata_y):
        candidate = _metadata_lookup(metadata, aliases)
        numeric = _parse_first_float(candidate)
        if numeric is None or numeric <= 0:
            continue

        key_hit = ""
        for k in metadata.keys():
            k_lower = str(k).strip().lower()
            if any(alias in k_lower for alias in aliases):
                key_hit = k_lower
                break

        if "cpm" in key_hit:
            return numeric / 60.0
        return numeric

    return None


def infer_revolutions_from_context(pair=None, signal_x=None, signal_y=None):
    """
    Blindado:
    1) Prioridad absoluta al header real del CSV: Number of Revs
    2) Si no existe, usar metadata exacta equivalente
    3) NO usar filename tipo REV32 para decidir revoluciones
    """
    meta_sources = []

    if pair is not None:
        meta_sources.extend([pair.metadata_x, pair.metadata_y])

    for signal_obj in (signal_x, signal_y):
        if signal_obj is not None:
            meta_sources.append(_safe_metadata(signal_obj))

    exact_aliases = [
        "number of revs",
        "number_of_revs",
        "number of revolutions",
        "revolutions",
        "revs",
    ]

    for meta in meta_sources:
        value = _metadata_lookup(meta, exact_aliases)
        revs = _parse_first_int(value)
        if revs is not None and 1 <= revs <= 4096:
            return revs

    for meta in meta_sources:
        for k, v in meta.items():
            text = f"{k}: {v}".lower()
            m = re.search(r"number\s+of\s+revs?\s*[:=,]?\s*(\d+)", text)
            if m:
                revs = int(m.group(1))
                if 1 <= revs <= 4096:
                    return revs

    return None


def infer_samples_per_rev_from_context(pair=None, signal_x=None, signal_y=None):
    """
    Blindado:
    - Si el CSV trae Number of Revs, ese manda.
    - samples/rev = usable_samples / revs
    - si sobra 1 muestra al final, se ignora.
    """
    if pair is not None:
        revs = infer_revolutions_from_context(pair=pair, signal_x=signal_x, signal_y=signal_y)
        if revs is not None and revs > 0:
            total = int(pair.time.size)

            if total >= revs:
                usable = total - (total % revs)
                if usable >= revs and usable > 0:
                    spr = usable // revs
                    if 8 <= spr <= 65536:
                        return spr

    texts = []

    if pair is not None:
        for meta in (pair.metadata_x, pair.metadata_y):
            for k, v in meta.items():
                texts.append(str(k))
                texts.append(str(v))

    for signal_obj in (signal_x, signal_y):
        if signal_obj is None:
            continue
        meta = _safe_metadata(signal_obj)
        for k, v in meta.items():
            texts.append(str(k))
            texts.append(str(v))

    patterns = [
        r"wf\s*\(\s*(\d+)\s*x\s*/\s*(\d+)\s*revs?\s*\)",
        r"(\d+)\s*x\s*/\s*(\d+)\s*revs?",
        r"(\d+)\s*samples?\s*/\s*rev",
        r"(\d+)\s*samples?\s*per\s*rev",
        r"spr\s*[:=]\s*(\d+)",
    ]

    for text in texts:
        lowered = str(text).lower()
        for pat in patterns:
            m = re.search(pat, lowered)
            if not m:
                continue
            spr = int(m.group(1))
            if 8 <= spr <= 65536:
                return spr

    return None


def infer_samples_per_rev(total_samples, preferred=(128, 64, 256, 32, 512, 1024, 96, 80, 160, 200)):
    if total_samples <= 0:
        raise ValueError("total_samples must be > 0")

    exact = []
    for spr in preferred:
        if spr <= 2:
            continue
        if total_samples % spr == 0:
            revs = total_samples // spr
            if 2 <= revs <= 4096:
                exact.append((spr, revs, abs(spr - 128)))

    if exact:
        exact.sort(key=lambda t: (t[2], -t[1], t[0]))
        return exact[0][0]

    candidates = []
    for spr in preferred:
        if spr <= 2 or spr >= total_samples:
            continue
        revs = total_samples / spr
        if 2.0 <= revs <= 4096.0:
            candidates.append((spr, abs(revs - round(revs)), abs(spr - 128)))

    if candidates:
        candidates.sort(key=lambda t: (t[1], t[2], t[0]))
        return candidates[0][0]

    return max(16, min(256, total_samples))


# =========================================================
# PROBE GEOMETRY
# =========================================================
class ProbeGeometry:
    def __init__(self, angle_deg, side):
        self.angle_deg = float(angle_deg)
        self.side = str(side)

    @property
    def side_normalized(self):
        return "Left" if self.side.strip().lower().startswith("l") else "Right"

    @property
    def label(self):
        suffix = "Left" if self.side_normalized == "Left" else "Right"
        return f"{self.angle_deg:.0f} {suffix}"

    @property
    def actual_angle_deg(self):
        angle = self.angle_deg % 360.0
        if self.side_normalized == "Left":
            return (360.0 - angle) % 360.0
        return angle

    @property
    def unit_vector(self):
        theta = np.deg2rad(self.actual_angle_deg)
        return np.array([np.sin(theta), np.cos(theta)], dtype=float)


def _solve_global_xy(probe_x, probe_y, geom_x, geom_y):
    u1 = geom_x.unit_vector
    u2 = geom_y.unit_vector

    mat = np.vstack([u1, u2])
    det = float(np.linalg.det(mat))
    if abs(det) < 1e-10:
        raise ValueError("Probe geometry is singular or nearly collinear.")

    data = np.vstack([probe_x, probe_y])
    solved = np.linalg.solve(mat, data)
    return solved[0], solved[1]


# =========================================================
# SEGMENTATION / HARMONICS
# =========================================================
def _segment_full_revolutions(time, x, y, samples_per_rev):
    if samples_per_rev < 8:
        raise ValueError("samples_per_rev must be >= 8")

    max_n = min(time.size, x.size, y.size)
    usable = max_n - (max_n % samples_per_rev)

    if usable < samples_per_rev:
        raise ValueError("Signal length is shorter than one revolution.")

    full_revs = int(usable // samples_per_rev)

    t = time[:usable].reshape(full_revs, samples_per_rev)
    xs = x[:usable].reshape(full_revs, samples_per_rev)
    ys = y[:usable].reshape(full_revs, samples_per_rev)

    return t, xs, ys, full_revs


def _harmonic_coefficients_per_rev(signal_2d, order):
    n = signal_2d.shape[1]
    theta = 2.0 * np.pi * np.arange(n, dtype=float) / n
    c = np.cos(order * theta)
    s = np.sin(order * theta)

    a = (2.0 / n) * np.sum(signal_2d * c[None, :], axis=1)
    b = (2.0 / n) * np.sum(signal_2d * s[None, :], axis=1)

    return a, b


def _reconstruct_from_complex(coef, order, n_samples):
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_samples), endpoint=False, dtype=float)
    return np.real(coef * np.exp(1j * order * theta))


def _open_segment_from_loop(x_loop, y_loop):
    if x_loop.size <= 1:
        return x_loop.copy(), y_loop.copy()
    return x_loop[:-1].copy(), y_loop[:-1].copy()


# =========================================================
# METRICS
# =========================================================
def _signed_area_from_closed(x_open, y_open):
    if x_open.size < 3:
        return 0.0

    xc = np.concatenate([x_open, x_open[:1]])
    yc = np.concatenate([y_open, y_open[:1]])
    return 0.5 * float(np.sum(xc[:-1] * yc[1:] - xc[1:] * yc[:-1]))


def _classify_traversal(area):
    if area > 0:
        return "CCW"
    if area < 0:
        return "CW"
    return "Undefined"


def _classify_precession(traversal, machine_rotation):
    traversal = str(traversal).strip().upper()
    machine_rotation = str(machine_rotation).strip().upper()

    if traversal not in {"CW", "CCW"} or machine_rotation not in {"CW", "CCW"}:
        return "Undefined"

    return "Forward" if traversal == machine_rotation else "Backward"


def _estimate_rpm(pair, samples_per_rev, rpm_override=None):
    if rpm_override is not None and np.isfinite(rpm_override) and rpm_override > 0:
        return float(rpm_override)

    rpm_meta = infer_rpm_from_metadata(pair.metadata_x, pair.metadata_y)
    if rpm_meta is not None and np.isfinite(rpm_meta) and rpm_meta > 0:
        return float(rpm_meta)

    if pair.dt > 0 and samples_per_rev > 0:
        rev_period = samples_per_rev * pair.dt
        if rev_period > 0:
            return 60.0 / rev_period

    return None


def _extract_units(pair):
    for metadata in (pair.metadata_x, pair.metadata_y):
        for key in ("y-axis unit", "y axis unit", "unit", "units", "engineering units", "eu"):
            found = _metadata_lookup(metadata, [key])
            if found:
                return str(found).strip()

    return "mil"


def _extract_timestamp(pair):
    for metadata in (pair.metadata_x, pair.metadata_y):
        for key in ("timestamp", "datetime", "date", "acquired", "acquisition date", "start time"):
            found = _metadata_lookup(metadata, [key])
            if found:
                return str(found)

    return "—"


def _extract_machine_name(pair):
    for metadata in (pair.metadata_x, pair.metadata_y):
        found = _metadata_lookup(metadata, ["machine", "machine name", "asset"])
        if found:
            return str(found)

    return "Orbit"


# =========================================================
# MAIN
# =========================================================
def compute_orbit(
    signal_x,
    signal_y,
    filter_mode="Direct",
    machine_rotation="CCW",
    x_probe_angle_deg=45.0,
    x_probe_side="Right",
    y_probe_angle_deg=45.0,
    y_probe_side="Left",
    samples_per_rev=None,
    revolution_index=0,
    display_revolutions_raw=None,
    average_revolutions_filtered=None,
    harmonic_plot_samples=720,
    rpm_override=None,
):
    canonical_filter_mode = FILTER_MODE_MAP.get(str(filter_mode).strip().upper(), "RAW")
    pair = synchronize_signals(signal_x, signal_y)

    if samples_per_rev is None:
        spr_context = infer_samples_per_rev_from_context(pair=pair, signal_x=signal_x, signal_y=signal_y)
        spr = int(spr_context) if spr_context is not None else infer_samples_per_rev(pair.time.size)
    else:
        spr = int(samples_per_rev)

    geom_x = ProbeGeometry(float(x_probe_angle_deg), x_probe_side)
    geom_y = ProbeGeometry(float(y_probe_angle_deg), y_probe_side)

    t_rev, x_rev, y_rev, total_revs = _segment_full_revolutions(pair.time, pair.x, pair.y, spr)
    rpm = _estimate_rpm(pair, spr, rpm_override)

    revolution_index = int(np.clip(revolution_index, 0, total_revs - 1))
    units = _extract_units(pair)
    machine_name = _extract_machine_name(pair)
    timestamp = _extract_timestamp(pair)

    if display_revolutions_raw is None:
        display_revolutions_raw = total_revs
    if average_revolutions_filtered is None:
        average_revolutions_filtered = total_revs

    x_channel = _clean_channel_name(pair.file_x)
    y_channel = _clean_channel_name(pair.file_y)

    probe_state = {
        "x_file": pair.file_x,
        "y_file": pair.file_y,
        "x_channel": x_channel,
        "y_channel": y_channel,
        "x_probe_label": f"{x_channel} @ {geom_x.angle_deg:.0f} {geom_x.side_normalized}",
        "y_probe_label": f"{y_channel} @ {geom_y.angle_deg:.0f} {geom_y.side_normalized}",
        "machine_name": machine_name,
        "timestamp": timestamp,
        "units": units,
        "samples_per_rev": spr,
        "total_revs": total_revs,
    }

    if canonical_filter_mode == "RAW":
        display_revolutions_raw = int(np.clip(display_revolutions_raw, 1, total_revs))
        start_rev = int(np.clip(revolution_index, 0, total_revs - display_revolutions_raw))
        end_rev = start_rev + display_revolutions_raw

        segment_x_open = []
        segment_y_open = []
        first_area = None

        for rev_idx in range(start_rev, end_rev):
            x_seg = x_rev[rev_idx].copy()
            y_seg = y_rev[rev_idx].copy()

            gx, gy = _solve_global_xy(x_seg, y_seg, geom_x, geom_y)

            if first_area is None:
                first_area = _signed_area_from_closed(gx, gy)

            open_x, open_y = _open_segment_from_loop(gx, gy)
            segment_x_open.append(open_x)
            segment_y_open.append(open_y)

        plot_x = np.concatenate([np.concatenate([seg, np.array([np.nan])]) for seg in segment_x_open])[:-1]
        plot_y = np.concatenate([np.concatenate([seg, np.array([np.nan])]) for seg in segment_y_open])[:-1]

        traversal = _classify_traversal(first_area)
        precession = _classify_precession(traversal, machine_rotation)

        start_point = (float(segment_x_open[0][0]), float(segment_y_open[0][0]))

        diagnostics = {
            "display_revolutions_raw": display_revolutions_raw,
            "x_wf_amp_pkpk": _pkpk(pair.x),
            "y_wf_amp_pkpk": _pkpk(pair.y),
        }

        return SimpleNamespace(
            plot_x=plot_x,
            plot_y=plot_y,
            rpm=rpm,
            samples_per_rev=spr,
            revolutions_available=total_revs,
            revolutions_used=(start_rev, end_rev - 1),
            filter_mode=canonical_filter_mode,
            traversal=traversal,
            precession=precession,
            signed_area=first_area,
            probe_state=probe_state,
            diagnostics=diagnostics,
            start_point=start_point,
            segment_x_open=segment_x_open,
            segment_y_open=segment_y_open,
        )

    order = 1 if canonical_filter_mode == "1X" else 2
    average_revolutions_filtered = int(np.clip(average_revolutions_filtered, 1, total_revs))
    start_rev = int(np.clip(revolution_index, 0, total_revs - average_revolutions_filtered))
    end_rev = start_rev + average_revolutions_filtered

    x_block = x_rev[start_rev:end_rev].copy()
    y_block = y_rev[start_rev:end_rev].copy()

    x_dc = np.mean(x_block, axis=1, keepdims=True)
    y_dc = np.mean(y_block, axis=1, keepdims=True)
    x_centered = x_block - x_dc
    y_centered = y_block - y_dc

    ax, bx = _harmonic_coefficients_per_rev(x_centered, order)
    ay, by = _harmonic_coefficients_per_rev(y_centered, order)

    cx = ax - 1j * bx
    cy = ay - 1j * by

    phase_x = np.angle(cx)
    phase_ref = np.angle(np.mean(np.exp(1j * phase_x)))

    delta = (phase_x - phase_ref) / float(order)
    rotator = np.exp(1j * order * delta)

    cx_aligned = cx * rotator
    cy_aligned = cy * rotator

    cx_mean = np.mean(cx_aligned)
    cy_mean = np.mean(cy_aligned)

    x_offset = float(np.mean(x_dc))
    y_offset = float(np.mean(y_dc))

    x_h = _reconstruct_from_complex(cx_mean, order, harmonic_plot_samples) + x_offset
    y_h = _reconstruct_from_complex(cy_mean, order, harmonic_plot_samples) + y_offset

    gx, gy = _solve_global_xy(x_h, y_h, geom_x, geom_y)
    area = _signed_area_from_closed(gx, gy)

    open_x, open_y = _open_segment_from_loop(gx, gy)

    traversal = _classify_traversal(area)
    precession = _classify_precession(traversal, machine_rotation)

    start_point = (float(open_x[0]), float(open_y[0]))

    x_order_pp = float(2.0 * np.mean(np.abs(cx)))
    y_order_pp = float(2.0 * np.mean(np.abs(cy)))

    diagnostics = {
        "order": order,
        "x_harmonic_amplitude_mean": x_order_pp,
        "y_harmonic_amplitude_mean": y_order_pp,
        "x_wf_amp_pkpk": _pkpk(pair.x),
        "y_wf_amp_pkpk": _pkpk(pair.y),
        "displayed_revolutions_filtered": average_revolutions_filtered,
    }

    return SimpleNamespace(
        plot_x=open_x,
        plot_y=open_y,
        rpm=rpm,
        samples_per_rev=spr,
        revolutions_available=total_revs,
        revolutions_used=(start_rev, end_rev - 1),
        filter_mode=canonical_filter_mode,
        traversal=traversal,
        precession=precession,
        signed_area=area,
        probe_state=probe_state,
        diagnostics=diagnostics,
        start_point=start_point,
        segment_x_open=[open_x],
        segment_y_open=[open_y],
    )