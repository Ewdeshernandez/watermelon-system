"""
s1_csv.py — Parser del CSV que exporta Bently System1 (vendored para el VM)
===========================================================================

Copia self-contained (numpy puro) de las funciones de core.dynamic_raw que el
agente necesita en el server, para no arrastrar el repo. El selftest del agente
cruza-verifica contra core.dynamic_raw si está disponible (guard anti-drift).

Formato REAL del 'Export to CSV' de una forma de onda (verificado en Parex):
    Machine Name,SGT300B
    Point Name,1XD TURBINA DE
    Wf Amp,51.063
    Number of Revs,16
    X-Axis Unit,ms
    Y-Axis Unit,µm
    Sample Speed, 14049 rpm
    Sample Status,Valid
    Timestamp,9/21/2026 8:29:22 PM
    Variable,Disp Wf(128X/16revs).KPH TURBINA
    X-Axis Value,Y-Axis Value
    0,4.581
    0.033,5.743
    ...
Onda keyphasor-sincronizada: arranca en la marca; cada samples_per_rev
muestras = 1 vuelta. No hay columna KPH (implícita).
"""
from __future__ import annotations

import re
from typing import Dict, Optional

import numpy as np


def parse_system1_waveform_csv(path_or_text) -> Dict[str, object]:
    if hasattr(path_or_text, "read"):
        text = path_or_text.read()
    elif isinstance(path_or_text, (bytes, bytearray)):
        text = path_or_text.decode("utf-8", "replace")
    elif isinstance(path_or_text, str) and "\n" not in path_or_text \
            and path_or_text.lower().endswith(".csv"):
        with open(path_or_text, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = str(path_or_text)
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", "replace")

    meta: Dict[str, str] = {}
    xs, ys = [], []
    in_data = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if not in_data:
            low = s.lower().replace(" ", "")
            if low.startswith("x-axisvalue") or low.startswith("x-axis,"):
                in_data = True
                continue
            if "," in s:
                k, v = s.split(",", 1)
                meta[k.strip()] = v.strip()
            continue
        parts = s.split(",")
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            continue

    t_ms = np.asarray(xs, dtype=float)
    values = np.asarray(ys, dtype=float)
    x_unit = meta.get("X-Axis Unit", "ms").strip().lower()
    t_s = t_ms / 1000.0 if x_unit.startswith("ms") else t_ms.copy()

    def _num(key: str) -> Optional[float]:
        m = re.search(r"[-+]?\d*\.?\d+", str(meta.get(key, "")))
        return float(m.group(0)) if m else None

    revs = int(_num("Number of Revs") or 0) or None
    rpm = _num("Sample Speed")
    spr = int(round(values.size / revs)) if (revs and values.size) else None
    fs_hz = None
    if t_s.size >= 2:
        dt = float(np.median(np.diff(t_s)))
        if dt > 0:
            fs_hz = 1.0 / dt
    if fs_hz is None and rpm and spr:
        fs_hz = spr * rpm / 60.0

    return {
        "meta": meta,
        "point": meta.get("Point Name", "").strip(),
        "variable": meta.get("Variable", "").strip(),
        "timestamp": meta.get("Timestamp", "").strip(),
        "rpm": rpm, "revs": revs, "samples_per_rev": spr, "fs_hz": fs_hz,
        "x_unit": meta.get("X-Axis Unit", "").strip(),
        "y_unit": meta.get("Y-Axis Unit", "").strip(),
        "t_s": t_s, "values": values,
    }


def synth_keyphasor(n: int, samples_per_rev: Optional[int]) -> np.ndarray:
    kph = np.zeros(int(n), dtype=float)
    if samples_per_rev and samples_per_rev > 1:
        for i in range(0, int(n), int(samples_per_rev)):
            kph[i:min(i + 2, n)] = 1.0
    return kph


# sensor label -> (bearing, axis)  ej: "1xd" -> ("1", "x");  "6yd" -> ("6","y")
_SENSOR_RE = re.compile(r"^\s*(\d+)\s*([xyXY])", re.I)


def sensor_bearing_axis(name: str):
    """Del nombre de archivo/sensor (1xd, 1yd, 6xd...) saca (bearing, axis)."""
    m = _SENSOR_RE.match(name or "")
    if not m:
        return None, None
    return m.group(1), m.group(2).lower()
