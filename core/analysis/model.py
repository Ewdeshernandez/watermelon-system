"""
core.analysis.model — modelo de dato unificado del workspace de Análisis (Fase 2).

Ingesta MANUAL: el usuario sube uno o varios CSV (System1 exporta 1 canal por
archivo, ej. 1yd.csv). Aquí se parsean (metadatos System1 o CSV genérico) y se
EMPAREJAN por cojinete en pares X/Y (como el agente del host), para habilitar
órbita / shaft centerline. Todo sin tocar la matemática de los motores core.*.

Self-contained (no importa tools/): el parser System1 vive aquí.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Channel:
    label: str                      # ej. "1YD", "4XA GEARBOX"
    bearing: str = ""               # "1", "4", ...
    axis: str = ""                  # "X" | "Y" | ""
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    v: np.ndarray = field(default_factory=lambda: np.array([]))
    rpm: Optional[float] = None
    spr: Optional[int] = None       # samples per rev (keyphasor)
    unit: str = ""
    source: str = ""                # nombre de archivo

    @property
    def fs_hz(self) -> Optional[float]:
        if self.t.size >= 2:
            dt = float(np.nanmedian(np.diff(self.t)))
            if dt > 0:
                return 1.0 / dt
        if self.rpm and self.spr:
            return self.spr * self.rpm / 60.0
        return None


@dataclass
class Bearing:
    name: str                       # "Cojinete 1"
    x: Optional[Channel] = None
    y: Optional[Channel] = None

    @property
    def has_xy(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass
class Sweep:
    """Dataset de barrido (run-up/coast-down) exportado por System1: amplitud y
    fase 1X vs rpm. Alimenta Bode y Polar."""
    label: str = ""
    rpm: np.ndarray = field(default_factory=lambda: np.array([]))
    amp: np.ndarray = field(default_factory=lambda: np.array([]))
    phase: np.ndarray = field(default_factory=lambda: np.array([]))  # grados
    unit: str = ""
    source: str = ""


@dataclass
class AnalysisRecord:
    name: str = ""
    channels: List[Channel] = field(default_factory=list)
    bearings: List[Bearing] = field(default_factory=list)
    sweeps: List[Sweep] = field(default_factory=list)

    @property
    def rpm(self) -> Optional[float]:
        for c in self.channels:
            if c.rpm:
                return c.rpm
        return None


# -------------------------------------------------------------------------
# Parser
# -------------------------------------------------------------------------
def _num(meta: Dict[str, str], key: str) -> Optional[float]:
    m = re.search(r"[-+]?\d*\.?\d+", str(meta.get(key, "")))
    return float(m.group(0)) if m else None


def _bearing_axis(label: str) -> Tuple[str, str]:
    """De 'Point Name'/nombre → (cojinete, eje). '1YD'→('1','Y'); '2XD TURBINA'→('2','X');
    '4XA GEARBOX'→('4','X'). Si no matchea, ('','')."""
    m = re.match(r"\s*(\d+)\s*([XY])", (label or "").upper())
    return (m.group(1), m.group(2)) if m else ("", "")


def parse_csv(name: str, data: bytes) -> Optional[Channel]:
    """Parsea un CSV: metadatos System1 ('Clave,Valor' + 'X-Axis,Value' + datos) o
    genérico (1ª col tiempo + 1ª col numérica). Devuelve un Channel o None."""
    try:
        text = data.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    meta: Dict[str, str] = {}
    xs: List[float] = []
    vs: List[float] = []
    in_data = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if not in_data:
            low = s.lower().replace(" ", "")
            if low.startswith("x-axisvalue") or low.startswith("x-axis,") \
                    or low.startswith("time,") or low.startswith("t_s,"):
                in_data = True
                continue
            if "," in s and not any(ch.isdigit() for ch in s.split(",", 1)[0][:3]):
                k, v = s.split(",", 1)
                meta[k.strip()] = v.strip()
                continue
        parts = s.split(",")
        if len(parts) >= 2:
            try:
                xs.append(float(parts[0])); vs.append(float(parts[1]))
            except ValueError:
                continue
    # Fallback genérico con pandas si no se detectó la tabla System1
    if len(vs) < 8:
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            numc = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numc:
                return None
            v = pd.to_numeric(df[numc[-1]], errors="coerce").to_numpy(float)
            t = (pd.to_numeric(df[numc[0]], errors="coerce").to_numpy(float)
                 if len(numc) >= 2 else np.arange(v.size, dtype=float))
            label = numc[-1]
            # CSV genérico: sin cojinete/eje (va a Waveform/Espectro, no se empareja).
            return Channel(label=label, bearing="", axis="", t=t, v=v,
                           unit="", source=name)
        except Exception:  # noqa: BLE001
            return None
    t = np.asarray(xs, float)
    x_unit = str(meta.get("X-Axis Unit", "ms")).strip().lower()
    if x_unit.startswith("ms"):
        t = t / 1000.0
    label = meta.get("Point Name", "").strip() or name.rsplit(".", 1)[0]
    bn, ax = _bearing_axis(label)
    revs = _num(meta, "Number of Revs")
    rpm = _num(meta, "Sample Speed")
    spr = int(round(len(vs) / revs)) if (revs and len(vs)) else None
    return Channel(label=label, bearing=bn, axis=ax, t=t,
                   v=np.asarray(vs, float), rpm=rpm, spr=spr,
                   unit=str(meta.get("Y-Axis Unit", "")).strip(), source=name)


def _find_col(cols, aliases) -> Optional[str]:
    low = {str(c).lower().replace(" ", ""): c for c in cols}
    for a in aliases:
        for k, orig in low.items():
            if a in k:
                return orig
    return None


def parse_sweep(name: str, data: bytes) -> Optional[Sweep]:
    """Detecta y parsea un CSV de Bode/Polar de System1 (tabla con columnas
    speed/rpm + amplitude + phase). None si no tiene esas 3 columnas."""
    try:
        text = data.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        parts = ln.split(",")
        if len(parts) >= 3:
            head = parts[:3]
            if not any(re.search(r"\d", p.split(".")[0] or "") and
                       p.strip().replace(".", "").replace("-", "").isdigit()
                       for p in head):
                start = i
                break
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep=None,
                         engine="python")
    except Exception:  # noqa: BLE001
        return None
    rc = _find_col(df.columns, ["speed", "rpm", "shaftspeed"])
    ac = _find_col(df.columns, ["amplitude", "magnitude", "direct", "1x", "amp"])
    pc = _find_col(df.columns, ["phase", "angle", "lag"])
    if not (rc and ac and pc):
        return None
    rpm = pd.to_numeric(df[rc], errors="coerce").to_numpy(float)
    amp = pd.to_numeric(df[ac], errors="coerce").to_numpy(float)
    ph = pd.to_numeric(df[pc], errors="coerce").to_numpy(float)
    m = np.isfinite(rpm) & np.isfinite(amp) & np.isfinite(ph)
    if int(m.sum()) < 2:
        return None
    order = np.argsort(rpm[m])
    return Sweep(label=name.rsplit(".", 1)[0], rpm=rpm[m][order],
                 amp=amp[m][order], phase=ph[m][order], source=name)


def build_record(files) -> AnalysisRecord:
    """files = lista de UploadedFile (st.file_uploader). Cada archivo puede ser
    una onda (1 canal) o un barrido Bode/Polar. Empareja X/Y por cojinete."""
    chans: List[Channel] = []
    sweeps: List[Sweep] = []
    for f in files or []:
        sw = parse_sweep(f.name, f.getvalue())
        if sw is not None:
            sweeps.append(sw)
            continue
        c = parse_csv(f.name, f.getvalue())
        if c is not None and c.v.size:
            chans.append(c)
    # emparejar por cojinete
    by_brg: Dict[str, Bearing] = {}
    for c in chans:
        if not c.bearing:
            continue
        b = by_brg.setdefault(c.bearing, Bearing(name=f"Cojinete {c.bearing}"))
        if c.axis == "X" and b.x is None:
            b.x = c
        elif c.axis == "Y" and b.y is None:
            b.y = c
    name = (files[0].name if files else "")
    if files and len(files) > 1:
        name = f"{len(files)} archivos"
    return AnalysisRecord(name=name, channels=chans,
                          bearings=list(by_brg.values()), sweeps=sweeps)


# -------------------------------------------------------------------------
# FUENTE B — histórico del activo (Supabase dynamic_raw, automático)
# -------------------------------------------------------------------------
def _bnum(point: str) -> str:
    m = re.search(r"(\d+)", point or "")
    return m.group(1) if m else ""


def session_stamps(caps_meta, gap_s: int = 1800):
    """Agrupa las capturas en SESIONES por proximidad de tiempo (una corrida de
    export dura ~minutos → archivos con HHMMSS distintos). gap_s=1800 (30 min)
    agrupa bearings + gearbox de la MISMA corrida aunque vayan con unos minutos
    de diferencia, sin mezclar corridas horarias (separadas ~60 min). Devuelve
    [(label, [keys...]), …] más reciente primero."""
    rows = []
    for c in caps_meta or []:
        d, t = c.get("day", ""), c.get("time", "")
        if len(d) == 8 and len(t) == 6:
            try:
                secs = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:])
                rows.append((d, secs, t, c.get("key", "")))
            except ValueError:
                pass
    rows.sort()  # por día, luego segundos
    sessions = []
    cur = []
    for r in rows:
        if cur and (r[0] != cur[-1][0] or r[1] - cur[-1][1] > gap_s):
            sessions.append(cur); cur = []
        cur.append(r)
    if cur:
        sessions.append(cur)
    out = []
    for sess in sessions:
        d, t0 = sess[0][0], sess[0][2]
        label = (f"{d[:4]}-{d[4:6]}-{d[6:]}  {t0[:2]}:{t0[2:4]}:{t0[4:]}"
                 if len(d) == 8 and len(t0) == 6 else f"{d} {t0}")
        out.append((label, [r[3] for r in sess]))
    out.reverse()  # más reciente primero
    return out


def record_from_history(label: str, keys) -> AnalysisRecord:
    """Carga una SESIÓN de capturas (lista de keys del bucket dynamic_raw) al
    MISMO modelo. El robot System1 ya subió la onda cruda con X/Y/KPH por cojinete
    (BRG1=cojinete emparejado → órbita lista). Automático: sin subir CSV."""
    from core.dynamic_raw import CH_X, CH_Y, download_capture  # noqa: PLC0415
    chans: List[Channel] = []
    bearings: List[Bearing] = []
    for key in keys or []:
        cap = download_capture(key)
        if cap is None:
            continue
        point = str(cap.meta.get("point", "") or "")
        rpm = cap.rpm
        spr = cap.samples_per_rev
        if cap.has(CH_X) and cap.has(CH_Y):          # par X/Y → cojinete + órbita
            n = _bnum(point)
            cx = Channel(label=f"{point} · X", bearing=n, axis="X", t=cap.t,
                         v=cap.get(CH_X), rpm=rpm, spr=spr,
                         unit=cap.unit_of(CH_X), source=point)
            cy = Channel(label=f"{point} · Y", bearing=n, axis="Y", t=cap.t,
                         v=cap.get(CH_Y), rpm=rpm, spr=spr,
                         unit=cap.unit_of(CH_Y), source=point)
            chans += [cx, cy]
            bearings.append(Bearing(name=point, x=cx, y=cy))
        else:                                        # canal único (acel/vel GB)
            v = cap.get(CH_X)
            if v is not None and v.size:
                chans.append(Channel(label=point, bearing=_bnum(point), t=cap.t,
                                     v=v, rpm=rpm, spr=spr,
                                     unit=cap.unit_of(CH_X), source=point))
    return AnalysisRecord(name=label, channels=chans, bearings=bearings)


def _amp_1x(v: np.ndarray, rpm: float, fs: float) -> float:
    """Amplitud 1X (pico) por FFT en la banda rpm/60."""
    w = v[np.isfinite(v)]
    if w.size < 16 or not rpm or not fs:
        return float("nan")
    w = w - np.mean(w)
    sp = np.abs(np.fft.rfft(w * np.hanning(w.size))) * (2.0 / w.size)
    fr = np.fft.rfftfreq(w.size, d=1.0 / fs)
    f1 = rpm / 60.0
    k = int(np.argmin(np.abs(fr - f1)))
    lo, hi = max(0, k - 2), min(sp.size, k + 3)
    return float(np.max(sp[lo:hi])) if hi > lo else float("nan")


def trend_series(caps_meta, point: str, metric: str = "pp", limit: int = 40):
    """Serie temporal de una métrica para un PUNTO a lo largo del histórico. Baja
    la captura de ese punto en cada sesión y calcula pp / rms / 1X. Devuelve
    ([datetime], [valor], unidad)."""
    import datetime as _dt

    from core.dynamic_raw import CH_X, download_capture  # noqa: PLC0415
    rows = sorted((c.get("day", ""), c.get("time", ""), c.get("key", ""))
                  for c in (caps_meta or [])
                  if c.get("point") == point and c.get("day") and c.get("time"))
    rows = rows[-limit:]
    ts, vals, unit = [], [], ""
    for d, t, key in rows:
        cap = download_capture(key)
        if cap is None:
            continue
        v = cap.get(CH_X)
        if v is None or not getattr(v, "size", 0):
            continue
        vf = v[np.isfinite(v)]
        if not vf.size:
            continue
        if metric == "rms":
            val = float(np.std(vf))
        elif metric == "1x":
            val = _amp_1x(vf, cap.rpm or 0.0, cap.fs_hz or 0.0)
        else:
            val = float(np.ptp(vf))
        if not np.isfinite(val):
            continue
        try:
            ts.append(_dt.datetime(int(d[:4]), int(d[4:6]), int(d[6:]),
                                   int(t[:2]), int(t[2:4]), int(t[4:])))
        except Exception:  # noqa: BLE001
            continue
        vals.append(val)
        unit = unit or cap.unit_of(CH_X)
    return ts, vals, unit


def spectra_series(caps_meta, point: str = "", limit: int = 24):
    """Espectros (FFT) del canal X de un PUNTO a lo largo del histórico, para la
    vista Waterfall. Baja cada sesión y calcula amplitud vs frecuencia. Devuelve
    [(stamp 'YYYY-MM-DD HH:MM', freqs_hz, amp), ...] ordenado por tiempo."""
    from core.dynamic_raw import CH_X, download_capture  # noqa: PLC0415
    pt = point or (caps_meta[0].get("point", "") if caps_meta else "")
    rows = sorted((c.get("day", ""), c.get("time", ""), c.get("key", ""))
                  for c in (caps_meta or [])
                  if (not point or c.get("point") == pt) and c.get("day") and c.get("time"))
    rows = rows[-limit:]
    out = []
    for d, t, key in rows:
        cap = download_capture(key)
        if cap is None:
            continue
        v = cap.get(CH_X)
        if v is None or not getattr(v, "size", 0):
            continue
        vf = v[np.isfinite(v)]
        fs = cap.fs_hz or 0.0
        if vf.size < 16 or not fs:
            continue
        w = vf - np.mean(vf)
        amp = np.abs(np.fft.rfft(w * np.hanning(w.size))) * (2.0 / w.size)
        fr = np.fft.rfftfreq(w.size, d=1.0 / fs)
        stamp = f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}"
        out.append((stamp, fr, amp))
    return out
