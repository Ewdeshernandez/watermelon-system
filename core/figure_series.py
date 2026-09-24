"""
core/figure_series.py
=====================

FUENTE ÚNICA de series de gráficos (espectro / onda / órbita) a partir de las
capturas crudas del robot (bucket dynamic_raw). Devuelve JSON-able para que
TODOS los consumidores muestren lo mismo:

  · Reportes (core.dynraw_snapshots) — [pendiente de dedupe en fase 3]
  · Web advanced analysis
  · API pública (api/app.py → /v1/figures/*) → app móvil / PWA

Reusa core.dynamic_raw (spectrum Hann, compute_orbit_from_capture) y core.orbit.
No hace red salvo list_captures/download_capture (Supabase). Contrato versionado:
cada dict trae "kind", "unit", "captured_at" y los arrays de datos.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

CONTRACT_VERSION = "1.0"


def _finite(seq):
    """NaN / inf → None (JSON null) para que el payload sea válido; los
    gráficos los tratan como huecos."""
    if seq is None:
        return None
    import math
    out = []
    for x in seq:
        try:
            xf = float(x)
            out.append(xf if math.isfinite(xf) else None)
        except Exception:
            out.append(None)
    return out


# --- token/canal -----------------------------------------------------------
def _point_channel(token: str):
    """Token de sensor (1XD, 4XA…) → (punto de captura, canal) en dynamic_raw.
    Igual que core.dynraw_snapshots para mantener paridad con los reportes."""
    m = re.match(r"(\d+)([XY])([DAV])", (token or "").upper())
    if not m:
        return None, None
    b, ax, kind = m.group(1), m.group(2), m.group(3)
    if kind == "D":
        return (f"BRG{b}" if b in ("1", "2", "5", "6") else f"GB{b}"), ax
    return f"GB_{b}{ax}{kind}", "X"


def _spectrum_xf(v, fs: float, disp: bool):
    """Transformada canónica ÚNICA (misma que la app móvil): Hann + zero-pad a
    pow2 ≈ ×8 → curva suave de alta resolución; amplitud calibrada a la señal
    real (n, no nfft). Reportes y app consumen esto = idénticos."""
    import numpy as np
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 8 or not fs:
        return None, None
    w = np.hanning(n)
    vw = (v - v.mean()) * w
    nfft = 1
    while nfft < n * 8:
        nfft <<= 1
    sp = np.fft.rfft(vw, n=nfft)                 # zero-pad a nfft
    fr = np.fft.rfftfreq(nfft, 1.0 / fs)
    amp = np.abs(sp) * 2.0 / (n * 0.5)           # calibrado a n (no nfft)
    if amp.size:
        amp[0] /= 2.0
    if disp:
        amp = amp * 2.0  # desplazamiento en pp
    return fr.tolist(), amp.tolist()


def _newest_by_point(instance_id: str) -> Dict[str, Dict[str, str]]:
    """{point: row más reciente} de dynamic_raw."""
    from core.dynamic_raw import list_captures
    latest: Dict[str, Dict[str, str]] = {}
    order: Dict[str, str] = {}
    for c in list_captures(instance_id, limit=200) or []:
        pt = c.get("point")
        if not pt:
            continue
        k = f"{c.get('day','')}{c.get('time','')}"
        if pt not in latest or k > order.get(pt, ""):
            latest[pt] = c
            order[pt] = k
    return latest


# --- API pública del módulo ------------------------------------------------
def list_points(instance_id: str) -> List[Dict[str, Any]]:
    """Puntos de captura disponibles (para poblar selectores)."""
    out = []
    for pt, row in _newest_by_point(instance_id).items():
        out.append({"point": pt, "day": row.get("day"), "time": row.get("time")})
    return sorted(out, key=lambda r: r["point"])


def _load_point(instance_id: str, point: str):
    from core.dynamic_raw import download_capture
    row = _newest_by_point(instance_id).get(point)
    if not row:
        return None
    return download_capture(row["key"])


def waveform_series(instance_id: str, token: str) -> Optional[Dict[str, Any]]:
    from core.dynamic_raw import CH_X
    pt, ch = _point_channel(token)
    if not pt:
        return None
    cap = _load_point(instance_id, pt)
    if cap is None:
        return None
    ch = ch if ch in ("X", "Y") else CH_X
    v = cap.get(ch)
    if v is None:
        return None
    import numpy as np
    v = np.asarray(v, float)
    fs = cap.fs_hz or 0.0
    n = v.size
    t = (np.arange(n) / fs).tolist() if fs else list(range(n))
    return {
        "kind": "waveform", "version": CONTRACT_VERSION, "token": token,
        "point": pt, "channel": ch, "unit": cap.unit_of(ch) or "",
        "fs_hz": fs, "rpm": getattr(cap, "rpm", None),
        "captured_at": getattr(cap, "captured_at", ""),
        "t": _finite(t), "v": _finite(v[np.isfinite(v)].tolist()),
    }


def spectrum_series(instance_id: str, token: str) -> Optional[Dict[str, Any]]:
    from core.dynamic_raw import CH_X
    pt, ch = _point_channel(token)
    if not pt:
        return None
    cap = _load_point(instance_id, pt)
    if cap is None:
        return None
    ch = ch if ch in ("X", "Y") else CH_X
    v = cap.get(ch)
    if v is None:
        return None
    unit = cap.unit_of(ch) or "µm"
    disp = "µm" in unit or "mil" in unit.lower()
    fr, amp = _spectrum_xf(v, cap.fs_hz or 0.0, disp)
    if fr is None:
        return None
    # Fmax de display desde la FUENTE ÚNICA de estilo (core.figure_style) →
    # el rango del eje X se define en un solo lugar para web/reportes/app.
    try:
        from core.figure_style import fmax_cpm as _fmax_cpm
        _fmax_hz = _fmax_cpm(unit) / 60.0
        _cut = next((i for i, f in enumerate(fr) if f > _fmax_hz), len(fr))
        fr, amp = fr[:_cut], amp[:_cut]
    except Exception:
        pass
    return {
        "kind": "spectrum", "version": CONTRACT_VERSION, "token": token,
        "point": pt, "channel": ch, "unit": unit,
        "rpm": getattr(cap, "rpm", None), "captured_at": getattr(cap, "captured_at", ""),
        "freq_hz": _finite(fr), "freq_cpm": _finite([f * 60.0 for f in fr]), "amp": _finite(amp),
    }


def _nice_ceil(x: float) -> float:
    """Idéntico al niceCeil de la app móvil (dynraw.ts)."""
    import math
    if not (x > 0):
        return 1.0
    e = math.floor(math.log10(x))
    b = 10 ** e
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if x <= m * b + 1e-9:
            return float(m * b)
    return float(10 * b)


def orbit_series_app(instance_id: str, bearing: str) -> Optional[Dict[str, Any]]:
    """Órbita en el MISMO shape que la app (computeOrbit de dynraw.ts):
    banda H/V, filtrada Ho/Vo, keyphasor kphH/kphV, dir, amax. Replica exacto
    el algoritmo 45R/45L + promedio por vuelta → la app la dibuja idéntica y
    cualquier cambio de diseño se hace en un solo lugar."""
    import math

    import numpy as np
    b = re.sub(r"[^0-9]", "", str(bearing)) or str(bearing)
    pt = f"BRG{b}" if b.isdigit() else str(bearing)
    cap = _load_point(instance_id, pt)
    if cap is None:
        return None
    px, py = cap.get("X"), cap.get("Y")
    if px is None or py is None:
        return None
    spr = int(cap.samples_per_rev or 0)
    if spr < 4:
        return None
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    n = int(min(px.size, py.size))
    tx, ty = math.radians(45), math.radians(-45)
    a, bb, c, d = math.sin(tx), math.cos(tx), math.sin(ty), math.cos(ty)
    det = (a * d - bb * c) or 1.0
    H = (d * px[:n] - bb * py[:n]) / det
    V = (-c * px[:n] + a * py[:n]) / det
    nrev = n // spr
    if nrev < 1:
        return None
    Hm = H[: nrev * spr].reshape(nrev, spr).mean(axis=0)
    Vm = V[: nrev * spr].reshape(nrev, spr).mean(axis=0)
    kphH, kphV = float(Hm[0]), float(Vm[0])
    gap = max(1, spr // 24)
    Ho, Vo = Hm[gap:], Vm[gap:]
    Hc, Vc = np.append(Hm, Hm[0]), np.append(Vm, Vm[0])
    area = float(np.sum(Hc[:-1] * Vc[1:] - Hc[1:] * Vc[:-1]))
    dir_ = "↺ CCW" if area > 0 else "↻ CW"
    ppH = float(Hm.max() - Hm.min())
    ppV = float(Vm.max() - Vm.min())
    rad = np.sort(np.hypot(H, V))
    p99 = float(rad[int(len(rad) * 0.99)]) if len(rad) else 1.0
    amax = _nice_ceil(max(p99, float(np.abs(Hm).max()), float(np.abs(Vm).max()), 1.0) * 1.12)
    return {
        "kind": "orbit", "version": CONTRACT_VERSION, "bearing": pt,
        "unit": cap.unit_of("X") or "µm", "rpm": cap.rpm,
        "captured_at": getattr(cap, "captured_at", ""),
        "H": _finite(H.tolist()), "V": _finite(V.tolist()),
        "Ho": _finite(Ho.tolist()), "Vo": _finite(Vo.tolist()),
        "kphH": kphH, "kphV": kphV, "ampPp": max(ppH, ppV),
        "dir": dir_, "amax": amax, "angX": 45, "angY": 45,
    }


def orbit_series(instance_id: str, bearing: str) -> Optional[Dict[str, Any]]:
    """Órbita X vs Y del cojinete. `bearing` = dígito ('1') o punto ('BRG1')."""
    from core.dynamic_raw import compute_orbit_from_capture
    b = re.sub(r"[^0-9]", "", str(bearing)) or str(bearing)
    pt = f"BRG{b}" if b.isdigit() else str(bearing)
    cap = _load_point(instance_id, pt)
    if cap is None:
        return None
    orb = compute_orbit_from_capture(cap)
    if orb is None:
        return None

    def _arr(name):
        val = getattr(orb, name, None)
        return _finite(val)

    _sp = getattr(orb, "start_point", None)  # keyphasor (punto brillante de inicio)
    try:
        _sp = [float(_sp[0]), float(_sp[1])] if _sp is not None and len(_sp) >= 2 else _sp
    except Exception:
        pass
    try:
        _unit = cap.unit_of("X") or "µm"
    except Exception:
        _unit = "µm"
    return {
        "kind": "orbit", "version": CONTRACT_VERSION, "bearing": pt,
        "unit": _unit, "rpm": getattr(orb, "rpm", None) or getattr(cap, "rpm", None),
        "captured_at": getattr(cap, "captured_at", ""),
        "x": _arr("plot_x"), "y": _arr("plot_y"),
        "keyphasor_point": _sp,
        "precession": getattr(orb, "precession", None),
        "samples_per_rev": getattr(orb, "samples_per_rev", None),
        "revolutions_used": getattr(orb, "revolutions_used", None),
    }
