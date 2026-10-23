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
