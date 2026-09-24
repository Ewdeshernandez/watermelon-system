"""
core.dynraw_snapshots — Refresca los snapshots de análisis (espectro / forma de
onda / órbita) desde la onda cruda del ROBOT (bucket dynamic_raw), para que los
reportes usen SIEMPRE la data actual — no snapshots viejos guardados a mano.

Problema que resuelve: `core.briefing_figures` arma las 3 figuras de análisis
desde `spectrum_history`/`waveform_history`/`orbit_history`, que solo se creaban
cuando un analista los guardaba a mano. Si nadie guardaba, el reporte mostraba
data de hace semanas (espectros planos). El robot sube onda buena a dynamic_raw
cada 2 h; aquí la convertimos a snapshots frescos con la MISMA reconstrucción
de la web/móvil (FFT Hann, órbita en coords de máquina).

Labels: se usa el `variable` de live_readings en MAYÚSCULAS (= Point Name de
System1, ej "1XD TURBINA DE"), que es EXACTAMENTE lo que el reporte ya espera →
el snapshot fresco REEMPLAZA al viejo en el merge (más nuevo gana), sin
duplicados.

Uso: `refresh_from_dynamic_raw(instance_id)` — idempotente y seguro (si no hay
onda nueva, no hace nada; nunca lanza hacia el reporte).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("dynraw_snapshots")

_TOKEN_RE = re.compile(r"\s*(\d+[XY][DAV])", re.I)


def _token(s: str) -> Optional[str]:
    m = _TOKEN_RE.match(s or "")
    return m.group(1).upper() if m else None


def _point_channel(token: str):
    """Token de sensor (1XD, 4XA…) → (punto de captura, canal) en dynamic_raw."""
    m = re.match(r"(\d+)([XY])([DAV])", token)
    if not m:
        return None, None
    b, ax, kind = m.group(1), m.group(2), m.group(3)
    if kind == "D":
        return (f"BRG{b}" if b in ("1", "2", "5", "6") else f"GB{b}"), ax
    return f"GB_{b}{ax}{kind}", "X"


def _spectrum(v: np.ndarray, fs: float, disp: bool):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 8 or not fs:
        return None, None
    w = np.hanning(n)
    vw = (v - v.mean()) * w
    sp = np.fft.rfft(vw)
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    amp = np.abs(sp) * 2.0 / (n * 0.5)
    if amp.size:
        amp[0] /= 2.0
    if disp:
        amp = amp * 2.0          # desplazamiento en pp
    return fr.tolist(), amp.tolist()


def _label_map(instance_id: str) -> Dict[str, str]:
    """token → label canónico (variable de live_readings en MAYÚSCULAS)."""
    out: Dict[str, str] = {}
    try:
        from core.live_readings import latest_for_instance
        for r in latest_for_instance(instance_id) or []:
            var = str(r.get("variable") or "")
            tok = _token(var) or _token(str(r.get("sensor_label") or ""))
            if tok and tok not in out and var:
                out[tok] = var.upper()
    except Exception as e:  # noqa: BLE001
        log.warning("label_map falló: %s", e)
    return out


def _newest_capture_time(caps: List[Dict[str, str]]) -> str:
    best = ""
    for c in caps:
        k = f"{c.get('day', '')}{c.get('time', '')}"
        if k > best:
            best = k
    return best


def _needs_refresh(instance_id: str, newest_cap: str) -> bool:
    """True si la captura más nueva es POSTERIOR al snapshot de espectro más
    reciente (llegó data nueva del robot). Si no hay snapshot, refresca."""
    if not newest_cap:
        return False
    try:
        from core import history_storage as hs
        snaps = hs.list_snapshots(instance_id, "spectrum") or []
        if not snaps:
            return True
        ts = str(snaps[0].get("timestamp") or "")
        # ts ISO → YYYYMMDDHHMMSS; newest_cap → YYYYMMDDHHMMSS
        snap_key = re.sub(r"[^0-9]", "", ts)[:14]
        return newest_cap > snap_key[:len(newest_cap)] if snap_key else True
    except Exception:  # noqa: BLE001
        return True


def refresh_from_dynamic_raw(instance_id: str, instance_obj: Any = None,
                             force: bool = False) -> Dict[str, Any]:
    """Convierte las capturas dynamic_raw más recientes en snapshots frescos de
    espectro/onda/órbita. Idempotente. Nunca propaga excepciones (devuelve dict
    con 'error')."""
    out: Dict[str, Any] = {"ok": False, "spectrum": 0, "waveform": 0,
                           "orbit": 0, "skipped": False, "error": ""}
    try:
        from core.dynamic_raw import list_captures, download_capture, CH_X, CH_Y
        caps = list_captures(instance_id, limit=120)
        if not caps:
            out["skipped"] = True
            return out
        newest = _newest_capture_time(caps)
        if not force and not _needs_refresh(instance_id, newest):
            out["skipped"] = True
            out["ok"] = True
            return out

        # captura más reciente por punto
        latest: Dict[str, Dict[str, str]] = {}
        for c in caps:
            pt = c["point"]
            if pt not in latest:
                latest[pt] = c
        capobj = {}
        for pt, row in latest.items():
            cp = download_capture(row["key"])
            if cp is not None:
                capobj[pt] = cp
        if not capobj:
            out["skipped"] = True
            return out

        lbls = _label_map(instance_id)
        spec_sd: List[Dict[str, Any]] = []
        wave_sd: List[Dict[str, Any]] = []
        prox: Dict[str, Dict[str, Any]] = {}

        for tok, label in lbls.items():
            pt, ch = _point_channel(tok)
            cap = capobj.get(pt)
            if cap is None:
                continue
            v = cap.get(ch) if ch in ("X", "Y") else cap.get(CH_X)
            if v is None:
                continue
            v = np.asarray(v, float)
            fs = cap.fs_hz or 0.0
            unit = cap.unit_of(CH_X) or "µm"
            disp = "µm" in unit or "mil" in unit.lower()
            ts = cap.captured_at or ""
            fr, am = _spectrum(v, fs, disp)
            if fr:
                spec_sd.append({"sensor_label": label, "freqs": fr, "amps": am,
                                "amp_unit": unit, "sampling_rate_hz": fs,
                                "csv_timestamp": ts})
            wave_sd.append({"sensor_label": label,
                            "time": (cap.t * 1000.0).tolist(),
                            "values": v.tolist(), "sampling_rate_hz": fs,
                            "n_samples_raw": int(v.size), "amp_unit": unit,
                            "csv_timestamp": ts})
            if disp and ch in ("X", "Y"):
                brg = re.match(r"(\d+)", tok).group(1)
                prox.setdefault(brg, {})[ch] = (label, v, unit, ts)

        orbit_bd: List[Dict[str, Any]] = []
        for brg, d in sorted(prox.items()):
            if "X" in d and "Y" in d:
                xl, xv, u, ts = d["X"]
                yl, yv, _, _ = d["Y"]
                n = min(len(xv), len(yv))
                orbit_bd.append({"bearing_label": f"BRG {brg}",
                                 "x_sensor_label": xl, "y_sensor_label": yl,
                                 "x_values": np.asarray(xv[:n]).tolist(),
                                 "y_values": np.asarray(yv[:n]).tolist(),
                                 "amp_unit": u, "csv_timestamp": ts})

        if spec_sd:
            from core.spectrum_history import save_spectrum_snapshot
            save_spectrum_snapshot(instance_id, sensors_data=spec_sd,
                                   corrida_label="Análisis de condición")
            out["spectrum"] = len(spec_sd)
        if wave_sd:
            from core.waveform_history import save_waveform_snapshot
            save_waveform_snapshot(instance_id, sensors_data=wave_sd,
                                   corrida_label="Análisis de condición")
            out["waveform"] = len(wave_sd)
        if orbit_bd:
            from core.orbit_history import save_orbit_snapshot
            save_orbit_snapshot(instance_id, bearings_data=orbit_bd,
                                corrida_label="Análisis de condición")
            out["orbit"] = len(orbit_bd)
        out["ok"] = True
        log.info("dynraw_snapshots(%s): %d espectro / %d onda / %d órbita",
                 instance_id, out["spectrum"], out["waveform"], out["orbit"])
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
        log.warning("refresh_from_dynamic_raw(%s) falló: %s", instance_id, e)
    return out


__all__ = ["refresh_from_dynamic_raw"]
