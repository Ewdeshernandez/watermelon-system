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

        # Capturas por punto, más recientes primero (list_captures ya viene así).
        by_point: Dict[str, List[Dict[str, str]]] = {}
        for c in caps:
            by_point.setdefault(c["point"], []).append(c)

        # --- Selección FULL SPEED (auto ≥95% del máx reciente) -----------------
        # El robot captura c/2h a cualquier rpm (part-load, run-up, coast-down).
        # Para que el reporte muestre la condición a PLENA CARGA se descarta lo
        # lento: ref = máx rpm de la captura más nueva de cada punto; luego por
        # punto se toma la MÁS RECIENTE con rpm ≥ 95% de ref (si la más nueva está
        # lenta, se camina hacia atrás hasta `_FS_MAX_BACK` capturas).
        _FS_FRAC = 0.95
        _FS_MAX_BACK = 10
        _cache: Dict[str, Any] = {}

        def _dl(row):
            k = row["key"]
            if k not in _cache:
                _cache[k] = download_capture(k)
            return _cache[k]

        def _rpm_of(cp) -> float:
            try:
                r = cp.rpm if cp is not None else None
                return float(r) if r and float(r) > 0 else 0.0
            except Exception:  # noqa: BLE001
                return 0.0

        # ref = máx rpm entre las capturas más nuevas de cada punto
        ref_rpm = 0.0
        for pt, rows in by_point.items():
            ref_rpm = max(ref_rpm, _rpm_of(_dl(rows[0])))
        _thr = _FS_FRAC * ref_rpm if ref_rpm > 0 else 0.0
        out["ref_rpm"] = round(ref_rpm, 1)
        out["fs_threshold_rpm"] = round(_thr, 1)

        capobj = {}
        _slow_pts: List[str] = []
        for pt, rows in by_point.items():
            chosen = None
            if _thr > 0:
                for row in rows[:_FS_MAX_BACK]:
                    cp = _dl(row)
                    if cp is not None and _rpm_of(cp) >= _thr:
                        chosen = cp
                        break
            if chosen is None:                  # ninguna a full speed → la más nueva
                chosen = _dl(rows[0])
                if _thr > 0 and chosen is not None and _rpm_of(chosen) < _thr:
                    _slow_pts.append(pt)
            if chosen is not None:
                capobj[pt] = chosen
        out["full_speed"] = (ref_rpm > 0 and not _slow_pts)
        out["slow_points"] = _slow_pts
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


# =========================================================================== #
# Tabular List desde la captura FULL-SPEED (misma fuente que órbita/espectro)
# =========================================================================== #
def _select_full_speed(instance_id: str, points=None, limit: int = 120):
    """Elige, POR PUNTO, la captura más reciente a ≥95% del máx rpm reciente de
    ESE punto (cada eje tiene su propia velocidad: turbina ~14000, generador
    ~1800 — un ref global los marcaría mal). `points` limita a un conjunto de
    puntos (evita bajar capturas que no se usan). Devuelve (capobj, info)."""
    from core.dynamic_raw import list_captures, download_capture
    caps = list_captures(instance_id, limit=limit)
    if not caps:
        return {}, {"ref_rpm": 0.0, "full_speed": False, "slow_points": []}
    want = set(points) if points else None
    by_point: Dict[str, List[Dict[str, str]]] = {}
    for c in caps:
        if want is not None and c["point"] not in want:
            continue
        by_point.setdefault(c["point"], []).append(c)
    _cache: Dict[str, Any] = {}

    def _dl(row):
        k = row["key"]
        if k not in _cache:
            _cache[k] = download_capture(k)
        return _cache[k]

    def _rpm_of(cp) -> float:
        try:
            r = cp.rpm if cp is not None else None
            return float(r) if r and float(r) > 0 else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    ref_rpm = 0.0
    capobj: Dict[str, Any] = {}
    slow: List[str] = []
    for pt, rows in by_point.items():
        scan = rows[:10]
        pt_ref = 0.0
        for row in scan:                      # ref = máx rpm reciente de ESTE punto
            pt_ref = max(pt_ref, _rpm_of(_dl(row)))
        thr = 0.95 * pt_ref if pt_ref > 0 else 0.0
        chosen = None
        if thr > 0:
            for row in scan:
                cp = _cache.get(row["key"]) or _dl(row)
                if cp is not None and _rpm_of(cp) >= thr:
                    chosen = cp
                    break
        if chosen is None:
            chosen = _dl(rows[0])
            if thr > 0 and chosen is not None and _rpm_of(chosen) < thr:
                slow.append(pt)
        if chosen is not None:
            capobj[pt] = chosen
        ref_rpm = max(ref_rpm, pt_ref)
    return capobj, {"ref_rpm": round(ref_rpm, 1),
                    "full_speed": (bool(capobj) and not slow),
                    "slow_points": slow}


def _mag_spectrum(v: np.ndarray, fs: float):
    """Espectro de amplitud 0-pk (Hann-corregido) — base para 1X/2X/0.5X."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 16 or not fs:
        return None, None
    w = np.hanning(n)
    sp = np.abs(np.fft.rfft((v - v.mean()) * w)) * 2.0 / np.sum(w)
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    return fr, sp


def _order_amp(fr, mag, f1: float, order: float, k: float) -> float:
    """Amplitud de un orden = pico espectral en [0.8,1.2]·(order·f1), ×k.
    k=2.0 (0-pk→pp, prox) o 1/√2 (0-pk→rms, vel/accel). Convención `_amp3`
    del campo (native/watermelon_field.py)."""
    if fr is None or not f1:
        return 0.0
    ft = order * f1
    b = (fr >= ft * 0.8) & (fr <= ft * 1.2)
    return float(mag[b].max()) * k if b.any() else 0.0


def _overall(v: np.ndarray, is_prox: bool) -> float:
    """Overall en la norma del sensor: prox → pp (max-min); vel/accel → rms."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    if is_prox:
        return float(v.max() - v.min())
    return float(np.sqrt(np.mean((v - v.mean()) ** 2)))


def full_speed_readings(instance_id: str) -> Dict[str, Any]:
    """Tabular List desde la captura FULL-SPEED de cada punto (misma fuente que
    órbita/espectro del reporte), en vez de la última lectura del colector.

    Reusa la ESTRUCTURA de Live (`latest_for_instance`) y REEMPLAZA solo los
    valores Direct / 1X_Ampl / 2X_Ampl / 0.5X_Ampl de los sensores de
    PROXIMIDAD (desplazamiento) por los computados de la onda cruda a plena
    carga, con la convención de norma (pp). Velocidad y aceleración se dejan de
    Live a propósito: su overall es de BANDA ANCHA y la onda cruda del robot no
    lo reconstruye (diverge ~50%); el colector es la fuente correcta para
    ISO 20816. Así las zonas/criterios/orden quedan idénticos a Live — en
    desplazamiento cambia el instante (último minuto del colector → plena carga)
    y queda consistente con la órbita/espectro del reporte.

    Devuelve {'rows': [...estilo live_readings...], 'full_speed': bool,
    'ref_rpm': float, 'slow_points': [...], 'asof': iso|'', 'matched': int}.
    rows=[] si no hay captura → el caller cae a Live."""
    out: Dict[str, Any] = {"rows": [], "full_speed": False, "ref_rpm": 0.0,
                           "slow_points": [], "asof": "", "matched": 0}
    try:
        import copy

        from core.dynamic_raw import CH_X, CH_Y
        from core.live_readings import latest_for_instance
        latest = latest_for_instance(instance_id) or []
        if not latest:
            return out

        # puntos necesarios = SOLO proximidad (desplazamiento). Evita bajar
        # capturas de los puntos de vel/accel que no se van a usar.
        prox: List[tuple] = []
        need: set = set()
        for r in latest:
            if r.get("metric") != "Direct":
                continue
            var = str(r.get("variable") or "")
            lbl = r.get("sensor_label")
            if (var.lower().startswith("velocidad")) or not lbl:
                continue
            unit = (r.get("unit") or "").lower()
            is_prox = ("mil" in unit or "µm" in unit
                       or "um" in unit.replace("µ", "u"))
            if not is_prox:
                continue
            tok = _token(var) or _token(str(lbl))
            if not tok:
                continue
            pt, ch = _point_channel(tok)
            prox.append((lbl, pt, ch))
            need.add(pt)
        if not need:
            return out

        capobj, info = _select_full_speed(instance_id, points=need)
        out.update({"full_speed": info.get("full_speed", False),
                    "ref_rpm": info.get("ref_rpm", 0.0),
                    "slow_points": info.get("slow_points", [])})
        if not capobj:
            return out

        # valores full-speed por sensor_label (proximidad → pp, k=2.0)
        vals: Dict[str, Dict[str, float]] = {}
        asof = ""
        matched = 0
        for lbl, pt, ch in prox:
            cap = capobj.get(pt)
            if cap is None:
                continue
            v = cap.get(ch) if ch in ("X", "Y") else cap.get(CH_X)
            if v is None or not getattr(v, "size", 0):
                continue
            fs = cap.fs_hz or 0.0
            rpm = float(cap.rpm or 0.0)
            f1 = rpm / 60.0 if rpm else 0.0
            kconv = 2.0                       # 0-pk → pp (proximidad)
            fr, mag = _mag_spectrum(np.asarray(v, float), fs)
            vals[lbl] = {
                "Direct": _overall(v, True),
                "1X_Ampl": _order_amp(fr, mag, f1, 1.0, kconv),
                "2X_Ampl": _order_amp(fr, mag, f1, 2.0, kconv),
                "0.5X_Ampl": _order_amp(fr, mag, f1, 0.5, kconv),
            }
            matched += 1
            ts = cap.captured_at or ""
            if str(ts) > str(asof):
                asof = str(ts)

        if not vals:
            return out

        # copia de Live con SOLO los valores reemplazados
        rows = copy.deepcopy(latest)
        _spd_rpm = out["ref_rpm"] or 0.0
        for r in rows:
            m = r.get("metric")
            lbl = r.get("sensor_label")
            var = str(r.get("variable") or "")
            if var.lower().startswith("velocidad") and _spd_rpm > 0:
                r["value"] = _spd_rpm
                if asof:
                    r["captured_at"] = asof
                continue
            if m in ("Direct", "1X_Ampl", "2X_Ampl", "0.5X_Ampl") and lbl in vals:
                nv = vals[lbl].get(m)
                if nv is not None:
                    r["value"] = round(float(nv), 4)
                    if asof:
                        r["captured_at"] = asof
        out["rows"] = rows
        out["asof"] = asof
        out["matched"] = matched
    except Exception as e:  # noqa: BLE001
        log.warning("full_speed_readings(%s) falló: %s", instance_id, e)
    return out


__all__ = ["refresh_from_dynamic_raw", "full_speed_readings"]
