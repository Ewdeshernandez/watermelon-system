"""
core.dynamic_raw — Onda CRUDA dinámica de System1 (con keyphasor)
=================================================================

Formato de intercambio único entre el agente de planta (VM Parex → nube) y la
web (Live Monitoring · Análisis Avanzado). Un archivo = UN punto (cojinete) en
UN instante, con todos sus canales muestreados en SIMULTÁNEO (regla de oro
rotodinámica): típicamente X, Y y KPH (keyphasor).

Header autodescriptivo en líneas '#'; luego CSV `t_s,<canal>,<canal>,...`.

Reconstrucción (numpy puro, SIN streamlit → testeable headless):
  • Forma de onda  — valor vs tiempo + marcas de keyphasor
  • Espectro       — rFFT → amplitud pico vs orden (1X/2X/3X marcados)
  • Órbita         — X vs Y (reusa core.orbit.compute_orbit: precesión/filtrado)

El agente y la web IMPORTAN de aquí para no divergir en el formato.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

CAPTURE_VERSION = "watermelon_dynamic_raw v1"
TIME_COL = "t_s"
# Nombres canónicos de canal (case-insensitive al parsear)
CH_X = "X"
CH_Y = "Y"
CH_KPH = "KPH"


# =========================================================
# MODELO
# =========================================================
@dataclass
class Capture:
    """Una captura cruda de un punto en un instante."""
    meta: Dict[str, str] = field(default_factory=dict)
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    channels: Dict[str, np.ndarray] = field(default_factory=dict)

    # --- conveniencias tipadas desde meta ---
    def _mget(self, key: str, default: str = "") -> str:
        return str(self.meta.get(key, default))

    @property
    def asset(self) -> str:
        return self._mget("asset")

    @property
    def point(self) -> str:
        return self._mget("point")

    @property
    def captured_at(self) -> str:
        return self._mget("captured_at")

    @property
    def rpm(self) -> Optional[float]:
        try:
            v = float(self.meta.get("rpm", "nan"))
            return v if np.isfinite(v) and v > 0 else None
        except Exception:
            return None

    @property
    def fs_hz(self) -> Optional[float]:
        # Preferir el declarado; si no, inferir del vector de tiempo.
        try:
            v = float(self.meta.get("fs_hz", "nan"))
            if np.isfinite(v) and v > 0:
                return v
        except Exception:
            pass
        if self.t.size >= 2:
            dt = float(np.median(np.diff(self.t)))
            if dt > 0:
                return 1.0 / dt
        return None

    @property
    def samples_per_rev(self) -> Optional[int]:
        try:
            v = int(float(self.meta.get("samples_per_rev", "0")))
            return v if v > 0 else None
        except Exception:
            return None

    def units(self) -> Dict[str, str]:
        """{canal: unidad} desde meta['units']='X:um,Y:um,KPH:V'."""
        out: Dict[str, str] = {}
        raw = self._mget("units")
        for part in raw.split(","):
            if ":" in part:
                ch, u = part.split(":", 1)
                out[ch.strip().upper()] = u.strip()
        return out

    def unit_of(self, ch: str) -> str:
        return self.units().get(ch.upper(), "")

    def has(self, ch: str) -> bool:
        return ch.upper() in {k.upper() for k in self.channels}

    def get(self, ch: str) -> Optional[np.ndarray]:
        for k, v in self.channels.items():
            if k.upper() == ch.upper():
                return v
        return None


# =========================================================
# SERIALIZACIÓN (agente escribe, web/tests leen)
# =========================================================
def build_capture_csv(meta: Dict[str, object], t: np.ndarray,
                      channels: Dict[str, np.ndarray]) -> str:
    """Serializa una captura al formato v1. `channels` en orden de inserción.
    Los arrays deben tener el mismo largo que `t`."""
    t = np.asarray(t, dtype=float).reshape(-1)
    cols = list(channels.keys())
    arrs = [np.asarray(channels[c], dtype=float).reshape(-1) for c in cols]
    n = t.size
    for c, a in zip(cols, arrs):
        if a.size != n:
            raise ValueError(f"Canal {c}: {a.size} muestras != t {n}")

    buf = io.StringIO()
    buf.write(f"# {CAPTURE_VERSION}\n")
    # meta ordenada y estable
    m = dict(meta)
    m.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    m.setdefault("channels", ",".join(cols))
    for k in sorted(m.keys()):
        buf.write(f"# {k}={m[k]}\n")
    buf.write(TIME_COL + "," + ",".join(cols) + "\n")

    data = np.column_stack([t] + arrs)
    # formato compacto pero suficiente (6 sig)
    np.savetxt(buf, data, delimiter=",", fmt="%.6g")
    return buf.getvalue()


def parse_capture_csv(data) -> Capture:
    """Parsea bytes/str/Path del formato v1 → Capture."""
    if hasattr(data, "read"):
        text = data.read()
    elif isinstance(data, (bytes, bytearray)):
        text = data.decode("utf-8", "replace")
    elif isinstance(data, str) and ("\n" not in data) and data.endswith(".csv"):
        with open(data, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = str(data)
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", "replace")

    meta: Dict[str, str] = {}
    header_cols: List[str] = []
    rows: List[List[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s[1:].strip()
            if body.startswith(CAPTURE_VERSION.split()[0]):
                continue  # línea de versión
            if "=" in body:
                k, v = body.split("=", 1)
                meta[k.strip()] = v.strip()
            continue
        if not header_cols:
            header_cols = [c.strip() for c in s.split(",")]
            continue
        parts = s.split(",")
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue

    if not header_cols or not rows:
        return Capture(meta=meta)

    arr = np.array(rows, dtype=float)
    # primera col = tiempo
    t = arr[:, 0]
    channels: Dict[str, np.ndarray] = {}
    for j, name in enumerate(header_cols[1:], start=1):
        if j < arr.shape[1]:
            channels[name] = arr[:, j]
    return Capture(meta=meta, t=t, channels=channels)


# =========================================================
# PARSER del CSV que EXPORTA System1 (Export to CSV de la onda)
# =========================================================
def parse_system1_waveform_csv(data) -> Dict[str, object]:
    """Lee el CSV que produce System1 al hacer 'Export to CSV' sobre una forma
    de onda. Formato real (verificado en el server Parex):

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

    La onda está sincronizada con keyphasor (arranca en la marca; cada
    samples_per_rev muestras = 1 vuelta). NO hay columna KPH: es implícita.

    Devuelve dict: {meta, point, rpm, revs, samples_per_rev, fs_hz,
    x_unit, y_unit, t_s (np.ndarray, segundos), values (np.ndarray)}.
    """
    if hasattr(data, "read"):
        text = data.read()
    elif isinstance(data, (bytes, bytearray)):
        text = data.decode("utf-8", "replace")
    elif isinstance(data, str) and ("\n" not in data) and data.endswith(".csv"):
        with open(data, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = str(data)
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8", "replace")

    meta: Dict[str, str] = {}
    xs: List[float] = []
    ys: List[float] = []
    in_data = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if not in_data:
            # fila de encabezado de datos
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
    # X-Axis Unit puede ser ms → pasar a segundos
    x_unit = meta.get("X-Axis Unit", "ms").strip().lower()
    t_s = t_ms / 1000.0 if x_unit.startswith("ms") else t_ms.copy()

    def _num(key: str) -> Optional[float]:
        raw = meta.get(key, "")
        m = re.search(r"[-+]?\d*\.?\d+", str(raw))
        return float(m.group(0)) if m else None

    revs = int(_num("Number of Revs") or 0) or None
    rpm = _num("Sample Speed")
    spr = None
    if revs and values.size:
        spr = int(round(values.size / revs))
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
    """Genera un canal KPH sintético (pulso al inicio de cada vuelta) a partir
    de samples_per_rev, ya que la onda de System1 arranca en la marca y no trae
    columna KPH. Sirve para reconstruir fase/órbita igual que con KPH real."""
    kph = np.zeros(int(n), dtype=float)
    if samples_per_rev and samples_per_rev > 1:
        idx = np.arange(0, n, samples_per_rev, dtype=int)
        for i in idx:
            kph[i:min(i + 2, n)] = 1.0
    return kph


# =========================================================
# RECONSTRUCCIÓN
# =========================================================
def spectrum(signal: np.ndarray, fs_hz: float,
             window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """rFFT amplitud-pico single-sided. Devuelve (freqs_hz, amp).
    Un seno de amplitud A produce un pico ~A en su frecuencia."""
    sig = np.asarray(signal, dtype=float).reshape(-1)
    sig = sig[np.isfinite(sig)]
    n = sig.size
    if n < 4 or not fs_hz or fs_hz <= 0:
        return np.zeros(0), np.zeros(0)
    sig = sig - float(np.mean(sig))
    if window == "hann":
        w = np.hanning(n)
    else:
        w = np.ones(n)
    coherent_gain = float(np.sum(w))
    spec = np.fft.rfft(sig * w)
    amp = np.abs(spec) / coherent_gain * 2.0
    if amp.size:
        amp[0] = amp[0] / 2.0  # DC no se duplica
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    return freqs, amp


def orders_axis(freqs_hz: np.ndarray, rpm: Optional[float]) -> Optional[np.ndarray]:
    """Convierte Hz → órdenes (×running speed). None si no hay rpm."""
    if not rpm or rpm <= 0:
        return None
    return freqs_hz / (rpm / 60.0)


def top_peaks(freqs_hz: np.ndarray, amp: np.ndarray,
              rpm: Optional[float] = None, n: int = 6,
              min_hz: float = 2.0) -> List[Dict[str, float]]:
    """Picos dominantes (máximos locales) ordenados por amplitud."""
    if freqs_hz.size < 3:
        return []
    peaks = []
    for i in range(1, amp.size - 1):
        if freqs_hz[i] < min_hz:
            continue
        if amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1]:
            d = {"freq_hz": float(freqs_hz[i]), "amp": float(amp[i])}
            if rpm and rpm > 0:
                d["order"] = float(freqs_hz[i] / (rpm / 60.0))
            peaks.append(d)
    peaks.sort(key=lambda p: p["amp"], reverse=True)
    return peaks[:n]


def keyphasor_edges(kph: np.ndarray, hi_frac: float = 0.5) -> np.ndarray:
    """Índices de flanco de subida del keyphasor (una marca por vuelta).
    Umbral = min + hi_frac*(max-min); detecta cruces ascendentes."""
    k = np.asarray(kph, dtype=float).reshape(-1)
    if k.size < 3:
        return np.zeros(0, dtype=int)
    lo, hi = float(np.nanmin(k)), float(np.nanmax(k))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.zeros(0, dtype=int)
    thr = lo + hi_frac * (hi - lo)
    above = k > thr
    # flanco de subida: False→True
    edges = np.where((~above[:-1]) & (above[1:]))[0] + 1
    return edges.astype(int)


def rpm_from_keyphasor(t: np.ndarray, kph: np.ndarray) -> Optional[float]:
    """RPM desde el periodo medio entre marcas de keyphasor."""
    edges = keyphasor_edges(kph)
    if edges.size < 2:
        return None
    tt = np.asarray(t, dtype=float).reshape(-1)
    periods = np.diff(tt[edges])
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if periods.size == 0:
        return None
    rev_s = float(np.median(periods))
    return 60.0 / rev_s if rev_s > 0 else None


def _as_signal(t: np.ndarray, x: np.ndarray, name: str,
               unit: str, rpm: Optional[float]) -> SimpleNamespace:
    """Envuelve arrays en el objeto mínimo que core.orbit espera."""
    meta = {"unit": unit, "channel": name}
    if rpm:
        meta["rpm"] = str(rpm)
    return SimpleNamespace(time=np.asarray(t, float), x=np.asarray(x, float),
                           name=name, file_name=name, metadata=meta)


def compute_orbit_from_capture(cap: Capture,
                               x_ch: str = CH_X, y_ch: str = CH_Y,
                               filter_mode: str = "Direct",
                               machine_rotation: str = "CCW",
                               x_angle: float = 45.0, x_side: str = "Right",
                               y_angle: float = 45.0, y_side: str = "Left"):
    """Reusa core.orbit.compute_orbit con los canales X/Y de la captura.
    Devuelve el dict de compute_orbit, o None si faltan datos."""
    xv, yv = cap.get(x_ch), cap.get(y_ch)
    if xv is None or yv is None or cap.t.size < 8:
        return None
    from core.orbit import compute_orbit  # import perezoso
    rpm = cap.rpm
    if rpm is None and cap.has(CH_KPH):
        rpm = rpm_from_keyphasor(cap.t, cap.get(CH_KPH))
    sx = _as_signal(cap.t, xv, x_ch, cap.unit_of(x_ch), rpm)
    sy = _as_signal(cap.t, yv, y_ch, cap.unit_of(y_ch), rpm)
    return compute_orbit(
        sx, sy, filter_mode=filter_mode, machine_rotation=machine_rotation,
        x_probe_angle_deg=x_angle, x_probe_side=x_side,
        y_probe_angle_deg=y_angle, y_probe_side=y_side,
        samples_per_rev=cap.samples_per_rev, rpm_override=rpm,
    )


# =========================================================
# STORAGE (nube) — bucket dynamic_raw
# =========================================================
BUCKET = "dynamic_raw"


def _service_client():
    """Cliente Supabase server-side (service key, bypass RLS). Igual patrón
    que los crons. None si no hay credenciales."""
    import os
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
           or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip())
    if not (url and key):
        try:
            import streamlit as st
            cfg = st.secrets.get("supabase", {})
            url = url or str(cfg.get("url", "")).strip()
            key = key or str(cfg.get("service_key", "")).strip()
        except Exception:
            pass
    if not (url and key):
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def list_captures(asset: str, limit: int = 200, client=None) -> List[Dict[str, str]]:
    """Lista objetos del bucket bajo {asset}/ (recursivo por día). Devuelve
    [{key, name, captured_at, point}] más recientes primero."""
    client = client or _service_client()
    if client is None:
        return []
    out: List[Dict[str, str]] = []
    try:
        st_api = client.storage.from_(BUCKET)
        # estructura: {asset}/{YYYYMMDD}/{HHMMSS}__{point}.csv
        days = st_api.list(asset) or []
        day_names = sorted((x.get("name", "") for x in days if x.get("name")),
                           reverse=True)
        for d in day_names:
            if d.endswith(".csv"):
                continue  # (defensivo) archivo suelto en la raíz del activo
            files = st_api.list(f"{asset}/{d}") or []
            # más reciente primero dentro del día
            for f in sorted(files, key=lambda z: z.get("name", ""), reverse=True):
                nm = f.get("name", "")
                if not nm.endswith(".csv"):
                    continue
                key = f"{asset}/{d}/{nm}"
                point = nm.split("__", 1)[1].rsplit(".", 1)[0] if "__" in nm else nm
                hhmmss = nm.split("__", 1)[0]
                out.append({"key": key, "name": nm, "point": point,
                            "day": d, "time": hhmmss})
                if len(out) >= limit:
                    return out
    except Exception:
        return out
    return out


def load_system1_captures(asset: str, client=None,
                          prefix: str = "s1") -> List[Capture]:
    """Lee los CSV que System1 exporta (formato 'Export to CSV'), subidos por el
    uploader PowerShell a dynamic_raw/{asset}/{prefix}/*.csv (un archivo = un
    sensor, ej. 1xd.csv). Empareja X/Y por cojinete y devuelve Captures v1
    (X, Y, KPH sintético) listos para reconstruir — sin necesitar Python en la VM.
    """
    client = client or _service_client()
    if client is None:
        return []
    st_api = client.storage.from_(BUCKET)
    try:
        files = st_api.list(f"{asset}/{prefix}") or []
    except Exception:
        return []
    # sensor -> (bearing, axis)
    import re as _re
    rex = _re.compile(r"^\s*(\d+)\s*([xyXY])")
    groups: Dict[str, Dict[str, dict]] = {}
    for f in files:
        nm = f.get("name", "")
        if not nm.lower().endswith(".csv"):
            continue
        m = rex.match(nm)
        if not m:
            continue
        bearing, axis = m.group(1), m.group(2).lower()
        try:
            raw = st_api.download(f"{asset}/{prefix}/{nm}")
            parsed = parse_system1_waveform_csv(raw)
        except Exception:
            continue
        if parsed["values"].size:
            groups.setdefault(bearing, {})[axis] = parsed

    out: List[Capture] = []
    for bearing, ax in sorted(groups.items()):
        base = ax.get("x") or ax.get("y")
        t = base["t_s"]
        channels: Dict[str, np.ndarray] = {}
        if "x" in ax:
            channels[CH_X] = ax["x"]["values"]
        if "y" in ax:
            yv = ax["y"]["values"]
            if yv.size != t.size:
                n = min(yv.size, t.size)
                yv, t2 = yv[:n], t[:n]
            channels[CH_Y] = yv
        spr = base["samples_per_rev"]
        channels[CH_KPH] = synth_keyphasor(t.size, spr)
        units = (f"{CH_X}:{ax.get('x', base)['y_unit']},"
                 f"{CH_Y}:{ax.get('y', base)['y_unit']},{CH_KPH}:pulse")
        meta = {
            "asset": asset, "point": f"BRG{bearing}",
            "captured_at": base.get("timestamp", ""),
            "rpm": base["rpm"] or "", "fs_hz": base["fs_hz"] or "",
            "samples_per_rev": spr or "", "units": units,
            "source": "system1_csv", "variable": base.get("variable", ""),
        }
        out.append(Capture(meta={k: str(v) for k, v in meta.items()},
                           t=t, channels=channels))
    return out


def download_capture(key: str, client=None) -> Optional[Capture]:
    """Descarga y parsea una captura del bucket."""
    client = client or _service_client()
    if client is None:
        return None
    try:
        raw = client.storage.from_(BUCKET).download(key)
        return parse_capture_csv(raw)
    except Exception:
        return None


def upload_capture(asset: str, point: str, csv_text: str,
                   captured_at: Optional[datetime] = None,
                   client=None) -> Optional[str]:
    """Sube una captura al bucket. Devuelve la key remota o None."""
    client = client or _service_client()
    if client is None:
        return None
    ts = captured_at or datetime.now(timezone.utc)
    safe_pt = "".join(c for c in point if c.isalnum() or c in "-_") or "PT"
    key = f"{asset}/{ts:%Y%m%d}/{ts:%H%M%S}__{safe_pt}.csv"
    try:
        client.storage.from_(BUCKET).upload(
            key, csv_text.encode("utf-8"),
            file_options={"content-type": "text/csv", "x-upsert": "true"},
        )
        return key
    except Exception:
        return None


__all__ = [
    "Capture", "build_capture_csv", "parse_capture_csv",
    "parse_system1_waveform_csv", "synth_keyphasor",
    "spectrum", "orders_axis", "top_peaks",
    "keyphasor_edges", "rpm_from_keyphasor", "compute_orbit_from_capture",
    "BUCKET", "list_captures", "download_capture", "upload_capture",
    "load_system1_captures",
    "CH_X", "CH_Y", "CH_KPH", "CAPTURE_VERSION",
]
