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
    "spectrum", "orders_axis", "top_peaks",
    "keyphasor_edges", "rpm_from_keyphasor", "compute_orbit_from_capture",
    "BUCKET", "list_captures", "download_capture", "upload_capture",
    "CH_X", "CH_Y", "CH_KPH", "CAPTURE_VERSION",
]
