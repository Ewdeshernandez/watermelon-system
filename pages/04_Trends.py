from __future__ import annotations

import base64
import hashlib
import math
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.auth import require_login, render_user_menu
from core.csv_common import decode_csv_text, find_header_line, parse_metadata_block
from core.instance_selector import render_instance_selector
from core.instance_state import get_instance as _get_instance_for_threshold
from core.report_state import append_report_item_and_persist, ensure_report_state_loaded
from core.sensor_map import resolve_sensor_for_point
from core.trend_diagnostics import build_trend_report_narrative as build_trend_report_narrative_core
from core.trend_history import (
    delete_trend_corrida,
    list_corridas_summary,
    list_trend_corridas,
    load_trend_corrida_files,
    save_trend_corrida,
    update_corrida_time_range,
)

st.set_page_config(page_title="Watermelon System | Trends", layout="wide")

require_login()
render_user_menu()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .main > div { padding-top: 0.18rem; }
        .stApp { background-color: #f3f4f6; }
        section[data-testid="stSidebar"] {
            background: #e5e7eb;
            border-right: 1px solid #cbd5e1;
        }
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            font-family: monospace;
        }
        div[data-testid="stFileUploader"] section {
            background: rgba(255,255,255,0.72);
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            padding: 0.25rem;
        }
        div[data-testid="stSelectSlider"] label p {
            font-weight: 700;
            color: #0f172a;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #dbe3ee;
            border-radius: 16px;
            background: rgba(255,255,255,0.65);
        }
        .wm-control-shell {
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78));
            border: 1px solid #dbe3ee;
            border-radius: 18px;
            padding: 14px 16px 12px 16px;
            margin-bottom: 12px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .wm-control-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 2px;
        }
        .wm-control-subtitle {
            font-size: 0.88rem;
            color: #64748b;
            margin-bottom: 10px;
        }
        section.main div[data-testid="stButton"] > button,
        section.main div[data-testid="stDownloadButton"] > button {
            min-height: 52px;
            border-radius: 16px;
            font-weight: 700;
            border: 1px solid #bfd8ff !important;
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%) !important;
            color: #2563eb !important;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.08);
            transition: all 0.18s ease;
        }
        section.main div[data-testid="stButton"] > button:hover,
        section.main div[data-testid="stDownloadButton"] > button:hover {
            border-color: #93c5fd !important;
            background: linear-gradient(180deg, #ffffff 0%, #f3f8ff 100%) !important;
            color: #1d4ed8 !important;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.12);
        }
        section.main div[data-testid="stButton"] > button *,
        section.main div[data-testid="stDownloadButton"] > button *,
        section.main div[data-testid="stButton"] > button p,
        section.main div[data-testid="stDownloadButton"] > button p,
        section.main div[data-testid="stButton"] > button span,
        section.main div[data-testid="stDownloadButton"] > button span,
        section.main div[data-testid="stButton"] > button div,
        section.main div[data-testid="stDownloadButton"] > button div {
            color: #2563eb !important;
        }
        .wm-export-actions {
            margin-top: 0.85rem;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_page_style()


def get_logo_base64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_logo_data_uri(path: Path) -> Optional[str]:
    b64 = get_logo_base64(path)
    if not b64:
        return None
    return f"data:image/png;base64,{b64}"


def make_export_state_key(parts: List[Any]) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def rounded_rect_path(x0: float, y0: float, x1: float, y1: float, r: float) -> str:
    r = max(0.0, min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    return (
        f"M {x0+r},{y0} "
        f"L {x1-r},{y0} "
        f"Q {x1},{y0} {x1},{y0+r} "
        f"L {x1},{y1-r} "
        f"Q {x1},{y1} {x1-r},{y1} "
        f"L {x0+r},{y1} "
        f"Q {x0},{y1} {x0},{y1-r} "
        f"L {x0},{y0+r} "
        f"Q {x0},{y0} {x0+r},{y0} Z"
    )


def format_number(value: Any, digits: int = 3, fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        val = float(value)
        if not math.isfinite(val):
            return fallback
        return f"{val:.{digits}f}"
    except Exception:
        return fallback


def safe_datetime(value: Any) -> Optional[pd.Timestamp]:
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts)
    except Exception:
        return None


def color_for_index(index: int) -> str:
    palette = [
        "#5b9cf0", "#10b981", "#8b5cf6", "#06b6d4", "#ec4899",
        "#14b8a6", "#6366f1", "#0f766e", "#7c3aed", "#2563eb",
    ]
    return palette[index % len(palette)]


def pretty_time(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.strftime("%I:%M %p").lstrip("0")


def pretty_date(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.strftime("%Y-%m-%d")


def trim_text(text: str, max_len: int) -> str:
    text = str(text or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def safe_percent_change(initial_value: Optional[float], final_value: Optional[float]) -> Optional[float]:
    if initial_value is None or final_value is None:
        return None
    try:
        init_val = float(initial_value)
        final_val = float(final_value)
        if not math.isfinite(init_val) or not math.isfinite(final_val):
            return None
        if abs(init_val) < 1e-12:
            return None
        return ((final_val - init_val) / abs(init_val)) * 100.0
    except Exception:
        return None


def ts_to_label(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def label_to_ts(text: str) -> Optional[pd.Timestamp]:
    return safe_datetime(text)


@dataclass
class TrendRecord:
    trend_id: str
    file_name: str
    machine: str = "Unknown"
    point: str = "Point"
    variable: str = "Direct"
    y_axis_unit: str = ""
    speed_unit: str = "rpm"
    timestamp_min: Optional[pd.Timestamp] = None
    timestamp_max: Optional[pd.Timestamp] = None
    x_time: pd.Series = field(default_factory=lambda: pd.Series(dtype="datetime64[ns]"))
    y_value: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    phase: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    speed: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    y_status: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))
    phase_status: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))
    speed_status: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f"{self.point} | {self.variable}"

    @property
    def point_clean(self) -> str:
        return self.point if self.point else self.file_name

    @property
    def n_samples(self) -> int:
        return int(len(self.x_time))


@dataclass
class OperationalRecord:
    op_id: str
    file_name: str
    machine: str = "Operational Data"
    variable: str = ""
    unit: str = ""
    family: str = "generic"
    timestamp_min: Optional[pd.Timestamp] = None
    timestamp_max: Optional[pd.Timestamp] = None
    x_time: pd.Series = field(default_factory=lambda: pd.Series(dtype="datetime64[ns]"))
    y_value: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def display_name(self) -> str:
        unit_txt = f" ({self.unit})" if self.unit else ""
        return f"{self.variable}{unit_txt}"

    @property
    def n_samples(self) -> int:
        return int(len(self.x_time))


# =============================================================
# Ciclo 17.8 — Operational data parser CLASE MUNDIAL
# =============================================================
# El CSV oficial del cliente (DCS export tipo C200C) tiene:
#   - Column tags técnicos: [C200C]TIT_200AXPV, [C200C]PT_200AXPV,
#     [BL1_BPCS]VFD_Siemens_C_200CVSD_Freq, AGA3_FIT_*Flow_GASFlow
#   - Date format M/D/YYYY HH:MM:SS (US locale del DCS)
#   - Timestamps DESORDENADOS (DCS export interleaves retries)
#   - Mezcla de familias: temperature, pressure, flow, frequency
#
# Antes el parser solo reconocía power/temperature y los labels
# salían crudos como '[C200C]TIT_200AXPV' — ilegibles.
# Ahora soporta:
#   1. Date M/D/YYYY explícito + fallback a auto-detección
#   2. Sort + dedup automático por timestamp
#   3. Familias extendidas: pressure, flow, frequency, vibration
#      (además de power, temperature, generic)
#   4. Humanización de labels: '[C200C]TIT_200AXPV' →
#      'Temp 200A (TIT)'
#   5. Unidades inferidas correctamente: psi para PT, MMSCFD para
#      FIT*Flow, Hz para *Freq

# Tag → familia (substrings, en orden de prioridad)
# Ciclo 17.25 — VFD/VSD/Freq van a SPEED (RPM) por default. Históricamente
# los reclasificábamos como "frequency" + Hz, pero los datos de operación
# de VFDs típicamente reportan VELOCIDAD (rpm), no frecuencia eléctrica.
# Solo si el nombre contiene "_hz", "hertz" explícito → frequency. El
# usuario que tenga datos en Hz reales debe nombrar la columna con "_Hz".
_OP_FAMILY_PATTERNS: List[Tuple[Tuple[str, ...], str]] = [
    # Pressure (PT_, PI_, "press") — chequear primero porque "PT" es muy específico
    (("pt_", "pi_", "press", "pressure"),                 "pressure"),
    # Flow (FIT_, FI_, FT_, "flow", AGA3)
    (("fit_", "fi_", "ft_", "flow", "aga3"),              "flow"),
    # Frequency con HZ EXPLÍCITO en el nombre (típico de redes 50/60Hz).
    # NO incluir "freq", "vfd", "vsd" acá — esos van a speed por default.
    ((" hz", "_hz", "hz_", "hertz"),                      "frequency"),
    # Speed / RPM (incluye VFD/VSD/Freq que típicamente reportan velocidad
    # del motor, no frecuencia eléctrica). Default unit: rpm.
    (("rpm", "speed", "_n_", "_rev", "vfd", "vsd", "freq"), "speed"),
    # Power / Load (MW, kW)
    (("mw", "kw", "power", "load", "_kva"),               "power"),
    # Temperature (TIT_, TE_, TI_, T48, T3, "temp")
    (("tit_", "te_", "ti_", "_t48", "_t3", "temp"),       "temperature"),
    # Vibration overall (VIB, *Vel*, *Disp*)
    (("vib_", "_vib", "vibration", "overall"),            "vibration"),
]


def infer_operational_family(column_name: str) -> str:
    """Detecta familia del operacional con patterns extendidos.

    Soporta tags DCS reales: PT_200AXPV → pressure, TIT_200AXPV →
    temperature, AGA3_FIT_*Flow → flow, VFD_*Freq → frequency.
    """
    name = str(column_name or "").lower()
    for tokens, fam in _OP_FAMILY_PATTERNS:
        if any(t in name for t in tokens):
            return fam
    return "generic"


def infer_operational_unit(column_name: str, temperature_unit: str = "°F") -> str:
    """Unidad por familia. Defaults industriales típicos."""
    fam = infer_operational_family(column_name)
    if fam == "pressure":
        return "psi"        # default; cliente puede tener bar/MPa/kg-cm²
    if fam == "temperature":
        return temperature_unit
    if fam == "flow":
        return "MMSCFD"     # gas natural típico AGA3
    if fam == "frequency":
        return "Hz"
    if fam == "speed":
        return "rpm"
    if fam == "power":
        return "MW"
    if fam == "vibration":
        return "mm/s RMS"
    return ""


def humanize_operational_label(column_name: str) -> str:
    """Convierte tag DCS críptico en label legible.

    Ejemplos:
        '[C200C]TIT_200AXPV'  →  'Temp 200A (TIT)'
        '[C200C]PT_200AXPV'   →  'Press 200A (PT)'
        '[BL1_BPCS]AGA3_FIT_2000CFlow_GASFlow'
                              →  'Flow Gas 2000C (AGA3)'
        '[BL1_BPCS]VFD_Siemens_C_200CVSD_Freq'
                              →  'VFD Frecuencia 200C'
    """
    raw = str(column_name or "").strip()
    if not raw:
        return ""

    # Strip [SystemName] prefix (e.g. [C200C], [BL1_BPCS])
    if raw.startswith("["):
        end = raw.find("]")
        if end > 0:
            raw = raw[end + 1:]

    # Strip common DCS suffixes (PV = process value, XPV = etc.)
    for sfx in ("AXPV", "BXPV", "XPV", "_PV", "PV"):
        if raw.endswith(sfx):
            raw = raw[: -len(sfx)]
            break

    low = raw.lower()
    # Mapeo de prefijos técnicos a etiquetas humanas + tag corto
    prefix_map = [
        ("tit_",  "Temp",   "TIT"),
        ("te_",   "Temp",   "TE"),
        ("ti_",   "Temp",   "TI"),
        ("pt_",   "Press",  "PT"),
        ("pi_",   "Press",  "PI"),
        ("fit_",  "Flow",   "FIT"),
        ("fi_",   "Flow",   "FI"),
        ("ft_",   "Flow",   "FT"),
        ("vfd_",  "VFD",    ""),
        ("vsd_",  "VSD",    ""),
        ("aga3_", "Flow Gas", "AGA3"),
    ]
    pretty = ""
    tag_short = ""
    for pfx, label, tag in prefix_map:
        if pfx in low:
            pretty = label
            tag_short = tag
            # remover el prefijo y dejar solo el resto (el "200A" del PT_200A)
            idx = low.find(pfx)
            after = raw[idx + len(pfx):]
            # Limpiar separadores
            after = after.replace("_", " ").strip()
            if after:
                pretty = f"{label} {after}"
            break

    if not pretty:
        # Sin prefijo técnico reconocido → limpiar lo que tengamos
        pretty = raw.replace("_", " ").strip()
        # Eliminar tokens muy técnicos como "Siemens", "BPCS"
        for tok in ("siemens ", "bpcs ", "bl1 "):
            pretty = pretty.replace(tok.title(), "").replace(tok.upper(), "").replace(tok, "")
        pretty = " ".join(pretty.split())

    if tag_short and tag_short.lower() not in pretty.lower():
        pretty += f" ({tag_short})"

    # Trim
    if len(pretty) > 42:
        pretty = pretty[:40] + "…"
    return pretty or column_name


def _parse_timestamps_robust(series: pd.Series) -> pd.Series:
    """Intenta varios formatos de fecha en orden de probabilidad
    para CSVs de DCS internacionales.

    Orden:
        1. M/D/YYYY HH:MM:SS  (US locale, típico Honeywell/Emerson)
        2. D/M/YYYY HH:MM:SS  (EU)
        3. ISO 8601           (YAML / moderno)
        4. Auto (pandas default)
    """
    raw = series.astype(str)
    # Intento 1: US M/D/YYYY
    out = pd.to_datetime(raw, errors="coerce", format="%m/%d/%Y %H:%M:%S")
    if out.notna().sum() > len(raw) * 0.5:
        return out
    # Intento 2: EU D/M/YYYY
    out = pd.to_datetime(raw, errors="coerce", format="%d/%m/%Y %H:%M:%S")
    if out.notna().sum() > len(raw) * 0.5:
        return out
    # Intento 3: ISO
    out = pd.to_datetime(raw, errors="coerce", format="ISO8601")
    if out.notna().sum() > len(raw) * 0.5:
        return out
    # Intento 4: pandas auto-infer (último recurso)
    return pd.to_datetime(raw, errors="coerce")


def parse_operational_csv(uploaded_file, temperature_unit: str = "°F") -> List[OperationalRecord]:
    """Parser robusto del CSV operacional (DCS export).

    Maneja:
        - Date M/D/YYYY (US) y D/M/YYYY (EU) automáticamente
        - Timestamps desordenados (ordena ascending)
        - Timestamps duplicados (deja el último, política DCS)
        - Tags técnicos crípticos (humaniza labels para display)
        - Familias extendidas: pressure, flow, frequency, etc.
    """
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return []

    if df.empty:
        return []

    # 1. Localizar columna timestamp
    timestamp_col = None
    for candidate in ["Time", "Timestamp", "DateTime", "Datetime",
                      "timestamp", "time", "TIMESTAMP", "Date", "date",
                      "FECHA", "Fecha"]:
        if candidate in df.columns:
            timestamp_col = candidate
            break
    if timestamp_col is None:
        return []

    # 2. Parser robusto multi-formato
    df[timestamp_col] = _parse_timestamps_robust(df[timestamp_col])
    df = df.dropna(subset=[timestamp_col]).copy()
    if df.empty:
        return []

    # 3. Sort + dedup por timestamp (DCS exports interleaves retries)
    df = (
        df.sort_values(timestamp_col)
          .drop_duplicates(subset=[timestamp_col], keep="last")
          .reset_index(drop=True)
    )

    records: List[OperationalRecord] = []
    machine_name = Path(uploaded_file.name).stem

    for col in df.columns:
        if col == timestamp_col:
            continue

        y = pd.to_numeric(df[col], errors="coerce")
        tmp = pd.DataFrame({"x": df[timestamp_col], "y": y}).dropna(subset=["x", "y"]).copy()
        if tmp.empty:
            continue

        unit = infer_operational_unit(col, temperature_unit)
        family = infer_operational_family(col)
        # Label humanizado para display, pero op_id queda con tag
        # crudo para no romper persistencia y matching de session.
        clean_label = humanize_operational_label(col)

        records.append(
            OperationalRecord(
                op_id=f"operational::{uploaded_file.name}::{col}",
                file_name=uploaded_file.name,
                machine=machine_name,
                variable=clean_label,
                unit=unit,
                family=family,
                timestamp_min=safe_datetime(tmp["x"].min()),
                timestamp_max=safe_datetime(tmp["x"].max()),
                x_time=tmp["x"].reset_index(drop=True),
                y_value=tmp["y"].reset_index(drop=True),
            )
        )

    return records


def load_operational_records_from_uploader(files: List[Any], temperature_unit: str = "°F") -> List[OperationalRecord]:
    records: List[OperationalRecord] = []
    for file in files:
        records.extend(parse_operational_csv(file, temperature_unit=temperature_unit))
    return records


def parse_trend_csv(uploaded_file) -> Optional[TrendRecord]:
    try:
        text = decode_csv_text(uploaded_file, errors="ignore")
    except Exception:
        return None

    lines = [line.rstrip("\r") for line in text.splitlines() if line.strip() != ""]
    if len(lines) < 2:
        return None

    data_header_idx = find_header_line(
        lines,
        required_signals=("X-Axis Value", "Y-Axis Value"),
    )
    if data_header_idx is None:
        return None

    header_map: Dict[str, str] = parse_metadata_block(lines[:data_header_idx])

    csv_text = "\n".join(lines[data_header_idx:])
    try:
        df = pd.read_csv(BytesIO(csv_text.encode("utf-8")))
    except Exception:
        return None

    expected_cols = [
        "X-Axis Value", "Y-Axis Value", "Y-Axis Status",
        "Phase", "Phase Status", "Speed", "Speed Status",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["X-Axis Value"] = pd.to_datetime(df["X-Axis Value"], errors="coerce")
    df["Y-Axis Value"] = pd.to_numeric(df["Y-Axis Value"], errors="coerce")
    df["Phase"] = pd.to_numeric(df["Phase"], errors="coerce")
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")

    df = df.dropna(subset=["X-Axis Value"]).copy()
    df = df.sort_values("X-Axis Value").reset_index(drop=True)
    if df.empty:
        return None

    trend_id = f"trend::{uploaded_file.name}"
    point = str(header_map.get("Point Name", uploaded_file.name)).strip()
    variable = str(header_map.get("Variable", "Direct")).strip()
    machine = str(header_map.get("Machine Name", "Unknown")).strip()
    y_axis_unit = str(header_map.get("Y-Axis Unit", "")).strip()
    speed_unit = str(header_map.get("Speed Unit", "rpm")).strip()

    return TrendRecord(
        trend_id=trend_id,
        file_name=uploaded_file.name,
        machine=machine,
        point=point,
        variable=variable,
        y_axis_unit=y_axis_unit,
        speed_unit=speed_unit,
        timestamp_min=safe_datetime(df["X-Axis Value"].min()),
        timestamp_max=safe_datetime(df["X-Axis Value"].max()),
        x_time=df["X-Axis Value"],
        y_value=df["Y-Axis Value"],
        phase=df["Phase"],
        speed=df["Speed"],
        y_status=df["Y-Axis Status"].astype(str),
        phase_status=df["Phase Status"].astype(str),
        speed_status=df["Speed Status"].astype(str),
        metadata=header_map,
    )


def load_trend_records_from_uploader(files: List[Any]) -> List[TrendRecord]:
    records: List[TrendRecord] = []
    for file in files:
        rec = parse_trend_csv(file)
        if rec is not None:
            records.append(rec)
    return records


# =============================================================
# Ciclo 17.5.2 — Sugerencia de Warning/Danger desde Sensor Map
# =============================================================
# Cada sensor del Sensor Map tiene `alarm` (= warning) y `danger`
# en su unit_native (mil pp para proximity, mm/s RMS para
# velocity, g RMS para accelerometer). Para los CSVs cargados,
# resolvemos el sensor que matchea por nombre/dirección y
# devolvemos el setpoint más conservador (mínimo) cuando hay
# múltiples sensores en el panel — así si CUALQUIERA cruza el
# umbral, dispara la alarma.
#
# Fallback por jerarquía:
#   1. Sensor Map de la instancia (Vault per-instance)
#   2. profile.custom_thresholds (si existe)
#   3. ISO 20816 por machine_group (defaults conservadores)
#
# El usuario SIEMPRE puede sobreescribir manualmente — el chip
# de fuente lo deja explícito ("Override del cliente").

def _iso_20816_amplitude_defaults(machine_group: str, metric_unit: str) -> Tuple[float, float, str]:
    """ISO 20816 thresholds básicos por machine_group para
    Amplitude. Retorna (warning, danger, source_label)."""
    mg = (machine_group or "").lower()
    mu = (metric_unit or "").lower()

    # Para vibración relativa al eje (proximity probes, mil pp)
    # los manuales típicos usan 3 mil pp / 5 mil pp como referencia.
    if "mil" in mu or "µm" in mu or "um" in mu:
        return (3.000, 5.000, "ISO/Bently default")

    # Para velocity (mm/s RMS) — ISO 20816-1/2 zonas C/D para
    # class IV (>75kW, fundación rígida) son ≈2.8 / 4.5 mm/s.
    if "mm/s" in mu:
        if "class_i" in mg and "iv" not in mg and "iii" not in mg:
            return (1.4, 2.8, "ISO 20816 class I")
        if "class_ii" == mg or "class_ii_" in mg:
            return (1.8, 4.5, "ISO 20816 class II")
        if "class_iii" in mg:
            return (2.8, 7.1, "ISO 20816 class III")
        # default: class IV
        return (2.8, 4.5, "ISO 20816 class IV")

    # Para g RMS (accelerometer) — heurística genérica.
    if mu.startswith("g") or "g rms" in mu or "g pk" in mu:
        return (3.0, 6.0, "Default accelerometer")

    return (3.000, 5.000, "Default")


def suggest_trend_thresholds(
    records: List[TrendRecord],
    sensors: List[Dict[str, Any]],
    metric_key: str,
    machine_group: str,
    instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Calcula los setpoints Warning/Danger sugeridos para los
    records seleccionados.

    Ciclo 17.9 — orden de prioridad:
        1. Norma ISO/API asignada a la instancia (con override del
           especialista si fue editado en Machinery Library)
        2. Sensor Map per-record (alarm/danger del Sensor Map)
        3. Fallback ISO 20816 genérico por machine_group
    """
    out: Dict[str, Any] = {
        "warning": None,
        "danger": None,
        "source": "default",
        "detail": "",
        "unit_hint": "",
        "applicable": False,
        "norm_reference": "",
    }

    if metric_key != "Amplitude":
        out["detail"] = "Setpoints sólo aplican a Amplitude"
        return out

    out["applicable"] = True
    unit_hint = ""
    if records:
        unit_hint = (records[0].y_axis_unit or "").strip()
    out["unit_hint"] = unit_hint

    # =====================================================
    # Ciclo 17.9 — PRIORIDAD 1: Norma asignada a la instance
    # =====================================================
    if instance is not None:
        _norm_code = (getattr(instance, "iso_norm_code", "") or "").strip()
        _norm_class = (getattr(instance, "iso_norm_class", "") or "").strip()
        if _norm_code and _norm_class:
            try:
                from core.iso_thresholds import get_thresholds as _gt
                _info = _gt(_norm_code, _norm_class)
            except Exception:
                _info = None
            if _info:
                _w_norma = float(_info["warning"])
                _d_norma = float(_info["danger"])
                _w_over = float(getattr(instance, "setpoint_warning_override", 0) or 0)
                _d_over = float(getattr(instance, "setpoint_danger_override", 0) or 0)
                # Override del especialista si > 0
                _w_eff = _w_over if _w_over > 0 else _w_norma
                _d_eff = _d_over if _d_over > 0 else _d_norma
                _has_override = (_w_over > 0 or _d_over > 0)
                out["warning"] = _w_eff
                out["danger"] = _d_eff
                out["source"] = (
                    f"{_info['source_label']} (Override)"
                    if _has_override else _info["source_label"]
                )
                out["detail"] = (
                    f"{_info['label']}"
                    + (" · Override del especialista" if _has_override else "")
                )
                out["norm_reference"] = _info["reference"]
                return out

    # =====================================================
    # PRIORIDAD 2: Sensor Map per-record
    # =====================================================
    matched_pairs: List[Tuple[TrendRecord, Dict[str, Any]]] = []
    if records and sensors:
        for rec in records:
            try:
                s = resolve_sensor_for_point(
                    sensors, rec.point, rec.variable, rec.y_axis_unit,
                )
            except Exception:
                s = None
            if s and (float(s.get("alarm", 0) or 0) > 0 or float(s.get("danger", 0) or 0) > 0):
                matched_pairs.append((rec, s))

    if matched_pairs:
        w_vals = [float(s.get("alarm", 0) or 0) for _, s in matched_pairs if float(s.get("alarm", 0) or 0) > 0]
        d_vals = [float(s.get("danger", 0) or 0) for _, s in matched_pairs if float(s.get("danger", 0) or 0) > 0]
        out["warning"] = min(w_vals) if w_vals else None
        out["danger"] = min(d_vals) if d_vals else None
        out["source"] = "Sensor Map"
        if len(matched_pairs) == 1:
            rec, s = matched_pairs[0]
            plane_lbl = s.get("plane_label") or f"plano {s.get('plane','')}"
            stype = s.get("sensor_type", "")
            unit = s.get("unit_native", "") or unit_hint
            out["detail"] = f"{rec.point_clean} → {plane_lbl} · {stype} ({unit})"
        else:
            out["detail"] = (
                f"{len(matched_pairs)} sensores en el panel · usando los "
                f"setpoints más conservadores"
            )
        return out

    # 2) Fallback ISO 20816 / Bently
    w_iso, d_iso, src = _iso_20816_amplitude_defaults(machine_group, unit_hint)
    out["warning"] = w_iso
    out["danger"] = d_iso
    out["source"] = src
    out["detail"] = (
        f"{src} para {unit_hint or 'Amplitude'}"
        f" (no hay match con Sensor Map)"
    )
    return out


# =============================================================
# HISTORICO DE TENDENCIAS — wrappers para CSVs persistidos
# =============================================================
# Ciclo 17.5 P2 — los CSVs guardados por core.trend_history se
# vuelven a cargar como bytes. Para que parse_trend_csv y
# parse_operational_csv funcionen sin cambios, los envolvemos
# en una clase mínima que imita la interfaz de UploadedFile de
# Streamlit (`.name`, `.read()`, `.seek()`).

class _NamedBytesIO(BytesIO):
    """BytesIO con atributo `.name` para que los parsers existentes
    lo traten como un UploadedFile de Streamlit."""

    def __init__(self, data: bytes, name: str) -> None:
        super().__init__(data or b"")
        self.name = str(name or "archivo.csv")


def _collect_uploader_bytes(files: List[Any]) -> List[Tuple[str, bytes]]:
    """Lee los bytes de cada UploadedFile sin consumirlo
    (rebobina al final para que el parser posterior funcione)."""
    out: List[Tuple[str, bytes]] = []
    for f in files or []:
        try:
            f.seek(0)
        except Exception:
            pass
        try:
            data = f.read()
        except Exception:
            continue
        if isinstance(data, str):
            data = data.encode("utf-8")
        try:
            f.seek(0)
        except Exception:
            pass
        try:
            name = f.name
        except Exception:
            name = "archivo.csv"
        out.append((name, data))
    return out


def _parse_corrida_files(
    files_bytes: List[Tuple[str, bytes]],
    *,
    temperature_unit: str,
    corrida_label: str,
) -> Tuple[List[TrendRecord], List[OperationalRecord]]:
    """Recibe lista (nombre, bytes) y los clasifica entre trend
    vs operational probando ambos parsers. El nombre del CSV se
    sufija con la corrida para que los IDs no colisionen al hacer
    merge con la corrida actual."""
    trend_recs: List[TrendRecord] = []
    op_recs: List[OperationalRecord] = []
    suffix = f" [{corrida_label}]" if corrida_label else ""

    for name, data in files_bytes:
        # Probar primero como trend CSV (header con X-Axis Value).
        wrapper = _NamedBytesIO(data, name + suffix)
        rec = parse_trend_csv(wrapper)
        if rec is not None:
            trend_recs.append(rec)
            continue
        # Si no es trend, intentar operational.
        wrapper2 = _NamedBytesIO(data, name + suffix)
        op_list = parse_operational_csv(wrapper2, temperature_unit=temperature_unit)
        if op_list:
            op_recs.extend(op_list)

    return trend_recs, op_recs


def _detect_time_range_for_uploads(
    files: List[Any],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Antes de guardar una corrida, detecta el rango temporal
    global de todos los CSVs (vibración + operacional) para
    poblar metadata.time_range."""
    mins: List[pd.Timestamp] = []
    maxs: List[pd.Timestamp] = []
    for f in files or []:
        try:
            f.seek(0)
        except Exception:
            pass
        try:
            text = decode_csv_text(f, errors="ignore")
        except Exception:
            continue
        finally:
            try:
                f.seek(0)
            except Exception:
                pass
        # Trend CSV
        lines = [ln.rstrip("\r") for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        idx = find_header_line(lines, required_signals=("X-Axis Value", "Y-Axis Value"))
        if idx is not None:
            try:
                csv_text = "\n".join(lines[idx:])
                df = pd.read_csv(BytesIO(csv_text.encode("utf-8")))
                if "X-Axis Value" in df.columns:
                    ts = pd.to_datetime(df["X-Axis Value"], errors="coerce").dropna()
                    if not ts.empty:
                        mins.append(ts.min())
                        maxs.append(ts.max())
            except Exception:
                pass
            continue
        # Operational CSV — buscar columna timestamp
        try:
            df = pd.read_csv(BytesIO(text.encode("utf-8")))
            for cand in ["Timestamp", "Time", "DateTime", "Datetime", "timestamp", "time"]:
                if cand in df.columns:
                    ts = pd.to_datetime(df[cand], errors="coerce").dropna()
                    if not ts.empty:
                        mins.append(ts.min())
                        maxs.append(ts.max())
                    break
        except Exception:
            continue
    ts_min = min(mins) if mins else None
    ts_max = max(maxs) if maxs else None
    return ts_min, ts_max


def _draw_top_strip(
    fig: go.Figure,
    machine_name: str,
    signal_names_text: str,
    metric_name: str,
    latest_text: str,
    logo_uri: Optional[str],
    time_range_text: str,
) -> None:
    x0, x1 = 0.006, 0.994
    y0, y1 = 1.014, 1.106
    radius = 0.015

    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=rounded_rect_path(x0, y0, x1, y1, radius),
        line=dict(color="#cfd8e3", width=1.15),
        fillcolor="rgba(255,255,255,0.97)",
        layer="below",
    )

    y_text = (y0 + y1) / 2.0

    if logo_uri:
        fig.add_layout_image(
            dict(
                source=logo_uri,
                xref="paper",
                yref="paper",
                x=0.014,
                y=y1 - 0.009,
                sizex=0.060,
                sizey=0.090,
                xanchor="left",
                yanchor="top",
                layer="above",
                sizing="contain",
                opacity=1.0,
            )
        )
        machine_x = 0.083
    else:
        machine_x = 0.020

    fig.add_annotation(
        xref="paper", yref="paper", x=machine_x, y=y_text,
        xanchor="left", yanchor="middle",
        text=f"<b>{trim_text(machine_name, 28)}</b>",
        showarrow=False, font=dict(size=12.8, color="#111827"), align="left",
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.325, y=y_text,
        xanchor="center", yanchor="middle",
        text=trim_text(signal_names_text, 34),
        showarrow=False, font=dict(size=11.4, color="#111827"), align="center",
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.640, y=y_text,
        xanchor="center", yanchor="middle",
        text=f"Metric: <b>{metric_name}</b> | Latest: <b>{trim_text(latest_text, 32)}</b>",
        showarrow=False, font=dict(size=11.3, color="#111827"), align="center",
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.986, y=y_text,
        xanchor="right", yanchor="middle",
        text=trim_text(time_range_text, 28),
        showarrow=False, font=dict(size=11.2, color="#111827"), align="right",
    )


def _draw_right_info_box(fig: go.Figure, rows: List[Tuple[str, str]]) -> None:
    panel_x0 = 0.834
    panel_x1 = 0.976
    panel_y1 = 0.915
    header_h = 0.034
    row_h = 0.058
    panel_h = header_h + len(rows) * row_h + 0.018
    panel_y0 = panel_y1 - panel_h

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y0, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(255,255,255,0.72)",
        layer="above",
    )

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(panel_x0, panel_y1 - header_h, panel_x1, panel_y1, 0.012),
        line=dict(color="rgba(0,0,0,0)", width=0),
        fillcolor="rgba(147,197,253,0.94)",
        layer="above",
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=(panel_x0 + panel_x1) / 2.0, y=panel_y1 - header_h / 2.0,
        text="<b>Trend Information</b>",
        showarrow=False, xanchor="center", yanchor="middle",
        font=dict(size=11.4, color="#111827"),
    )

    current_top = panel_y1 - header_h - 0.008
    for title, value in rows:
        title_y = current_top - 0.004
        value_y = current_top - 0.030

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.030, y=title_y,
            xanchor="left", yanchor="top",
            text=f"<b>{title}</b>",
            showarrow=False, font=dict(size=10.7, color="#111827"), align="left",
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=panel_x0 + 0.030, y=value_y,
            xanchor="left", yanchor="top",
            text=value,
            showarrow=False, font=dict(size=10.4, color="#111827"), align="left",
        )

        current_top -= row_h


def get_metric_series(record: TrendRecord, metric_key: str) -> Tuple[pd.Series, str]:
    if metric_key == "Amplitude":
        return record.y_value, record.y_axis_unit or ""
    if metric_key == "Phase":
        return record.phase, "deg"
    if metric_key == "Speed":
        return record.speed, record.speed_unit or "rpm"
    return record.y_value, record.y_axis_unit or ""


def get_clean_metric_df(record: TrendRecord, metric_key: str) -> pd.DataFrame:
    metric_series, _ = get_metric_series(record, metric_key)
    df = pd.DataFrame({"x": pd.to_datetime(record.x_time, errors="coerce"), "y": pd.to_numeric(metric_series, errors="coerce")}).dropna(subset=["x", "y"])
    if df.empty:
        return df
    return df.sort_values("x").reset_index(drop=True)


def get_cursor_nearest_info(record: TrendRecord, metric_key: str, cursor_ts: Optional[pd.Timestamp]) -> Optional[Tuple[float, pd.Timestamp, str]]:
    if cursor_ts is None:
        return None
    df = get_clean_metric_df(record, metric_key)
    if df.empty:
        return None
    idx = (df["x"] - cursor_ts).abs().idxmin()
    row = df.loc[idx]
    unit = get_metric_series(record, metric_key)[1]
    return float(row["y"]), pd.Timestamp(row["x"]), unit


def get_time_options_for_records(records: List[TrendRecord], metric_key: str) -> List[pd.Timestamp]:
    ts_values: List[pd.Timestamp] = []
    for record in records:
        df = get_clean_metric_df(record, metric_key)
        if not df.empty:
            ts_values.extend(list(pd.to_datetime(df["x"], errors="coerce").dropna()))
    if not ts_values:
        return []
    unique_sorted = sorted(pd.Series(ts_values).dropna().unique())
    return [pd.Timestamp(x) for x in unique_sorted]


def get_operational_clean_df(record: OperationalRecord) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "x": pd.to_datetime(record.x_time, errors="coerce"),
            "y": pd.to_numeric(record.y_value, errors="coerce"),
        }
    ).dropna(subset=["x", "y"])
    if df.empty:
        return df
    return df.sort_values("x").reset_index(drop=True)


def get_operational_cursor_nearest_info(record: OperationalRecord, cursor_ts: Optional[pd.Timestamp]) -> Optional[Tuple[float, pd.Timestamp, str]]:
    if cursor_ts is None:
        return None
    df = get_operational_clean_df(record)
    if df.empty:
        return None
    idx = (df["x"] - cursor_ts).abs().idxmin()
    row = df.loc[idx]
    return float(row["y"]), pd.Timestamp(row["x"]), record.unit


def get_time_options_for_operational_records(records: List[OperationalRecord]) -> List[pd.Timestamp]:
    ts_values: List[pd.Timestamp] = []
    for record in records:
        df = get_operational_clean_df(record)
        if not df.empty:
            ts_values.extend(list(pd.to_datetime(df["x"], errors="coerce").dropna()))
    if not ts_values:
        return []
    unique_sorted = sorted(pd.Series(ts_values).dropna().unique())
    return [pd.Timestamp(x) for x in unique_sorted]



def align_trend_and_operational_for_correlation(
    trend_record: TrendRecord,
    operational_record: OperationalRecord,
    metric_key: str,
) -> pd.DataFrame:
    trend_df = get_clean_metric_df(trend_record, metric_key)
    op_df = get_operational_clean_df(operational_record)

    if trend_df.empty or op_df.empty:
        return pd.DataFrame(columns=["x", "trend", "operational"])

    trend_df = trend_df.rename(columns={"y": "trend"}).copy()
    op_df = op_df.rename(columns={"y": "operational"}).copy()

    trend_df["x"] = pd.to_datetime(trend_df["x"], errors="coerce")
    op_df["x"] = pd.to_datetime(op_df["x"], errors="coerce")

    trend_df = trend_df.dropna(subset=["x", "trend"]).sort_values("x").reset_index(drop=True)
    op_df = op_df.dropna(subset=["x", "operational"]).sort_values("x").reset_index(drop=True)

    if trend_df.empty or op_df.empty:
        return pd.DataFrame(columns=["x", "trend", "operational"])

    merged = pd.merge_asof(
        trend_df,
        op_df,
        on="x",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )

    merged = merged.dropna(subset=["trend", "operational"]).reset_index(drop=True)
    return merged


def classify_correlation_strength(corr_value: Optional[float]) -> Dict[str, str]:
    if corr_value is None or not math.isfinite(float(corr_value)):
        return {
            "strength": "Nula",
            "direction": "Indeterminada",
            "interpretation": "No fue posible calcular correlación válida entre vibración y variable operativa.",
            "color": "#64748b",
        }

    corr = float(corr_value)
    abs_corr = abs(corr)

    if corr >= 0.0:
        direction = "Positiva"
    else:
        direction = "Negativa"

    if abs_corr >= 0.75:
        strength = "Fuerte"
        color = "#16a34a"
    elif abs_corr >= 0.50:
        strength = "Moderada"
        color = "#f59e0b"
    elif abs_corr >= 0.25:
        strength = "Débil"
        color = "#f97316"
    else:
        strength = "Nula"
        color = "#64748b"

    if strength == "Fuerte" and direction == "Positiva":
        interpretation = "La vibración aumenta cuando aumenta la variable operativa, lo que sugiere influencia operativa importante."
    elif strength == "Fuerte" and direction == "Negativa":
        interpretation = "La vibración disminuye cuando aumenta la variable operativa, indicando relación inversa fuerte."
    elif strength == "Moderada" and direction == "Positiva":
        interpretation = "Existe relación operativa apreciable, aunque no completamente dominante."
    elif strength == "Moderada" and direction == "Negativa":
        interpretation = "Existe relación inversa moderada entre vibración y variable operativa."
    elif strength == "Débil":
        interpretation = "La dependencia operativa es débil; conviene complementar con diagnóstico mecánico."
    else:
        interpretation = "No se observa dependencia operativa clara; la condición podría estar dominada por factores mecánicos o por ruido operacional."

    return {
        "strength": strength,
        "direction": direction,
        "interpretation": interpretation,
        "color": color,
    }


def build_trend_operational_correlation(
    trend_record: Optional[TrendRecord],
    operational_record: Optional[OperationalRecord],
    metric_key: str,
) -> Dict[str, Any]:
    if trend_record is None or operational_record is None:
        return {
            "valid": False,
            "corr_value": None,
            "sample_count": 0,
            "strength": "Nula",
            "direction": "Indeterminada",
            "interpretation": "Seleccione una señal de vibración y una variable operativa para habilitar la correlación.",
            "color": "#64748b",
            "trend_name": trend_record.point_clean if trend_record else "—",
            "operational_name": operational_record.variable if operational_record else "—",
        }

    merged = align_trend_and_operational_for_correlation(
        trend_record=trend_record,
        operational_record=operational_record,
        metric_key=metric_key,
    )

    if len(merged) < 4:
        return {
            "valid": False,
            "corr_value": None,
            "sample_count": int(len(merged)),
            "strength": "Nula",
            "direction": "Indeterminada",
            "interpretation": "No hay suficientes puntos coincidentes en el tiempo para calcular correlación confiable.",
            "color": "#64748b",
            "trend_name": trend_record.point_clean,
            "operational_name": operational_record.variable,
        }

    corr_value = merged["trend"].corr(merged["operational"])
    meta = classify_correlation_strength(corr_value)

    return {
        "valid": True,
        "corr_value": float(corr_value) if corr_value is not None and math.isfinite(float(corr_value)) else None,
        "sample_count": int(len(merged)),
        "strength": meta["strength"],
        "direction": meta["direction"],
        "interpretation": meta["interpretation"],
        "color": meta["color"],
        "trend_name": trend_record.point_clean,
        "operational_name": operational_record.variable,
        "trend_unit": get_metric_series(trend_record, metric_key)[1],
        "operational_unit": operational_record.unit,
        "merged_df": merged,
    }


def build_correlation_scatter_figure(correlation_info: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()

    merged = correlation_info.get("merged_df")
    if merged is None or not isinstance(merged, pd.DataFrame) or merged.empty:
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=40, t=40, b=40),
            title="Correlation Plot",
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=merged["operational"],
            y=merged["trend"],
            mode="markers",
            name="Samples",
            marker=dict(size=8),
            hovertemplate=(
                "Operational: %{x:.4f}<br>"
                "Trend: %{y:.4f}<extra></extra>"
            ),
        )
    )

    if len(merged) >= 2:
        x_vals = merged["operational"].astype(float).to_numpy()
        y_vals = merged["trend"].astype(float).to_numpy()
        if np.isfinite(x_vals).all() and np.isfinite(y_vals).all():
            try:
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(np.min(x_vals), np.max(x_vals), 100)
                y_line = slope * x_line + intercept
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name="Trend line",
                        line=dict(width=2.5),
                    )
                )
            except Exception:
                pass

    trend_name = correlation_info.get("trend_name") or "Trend"
    operational_name = correlation_info.get("operational_name") or "Operational"

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=50, b=50),
        title=f"Correlation: {trend_name} vs {operational_name}",
        xaxis_title=operational_name,
        yaxis_title=trend_name,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig



def _robust_scale(series: pd.Series) -> float:
    arr = pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()
    if arr.size == 0:
        return 1.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad > 1e-12:
        return max(1.4826 * mad, 1e-9)
    std = float(np.std(arr))
    if std > 1e-12:
        return max(std, 1e-9)
    return 1.0


def detect_trend_anomalies(record: TrendRecord, metric_key: str) -> pd.DataFrame:
    df = get_clean_metric_df(record, metric_key).copy()
    if df.empty or len(df) < 8:
        return pd.DataFrame(columns=["x", "y", "anomaly_type", "severity", "point_score", "diff_score"])

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    if df.empty or len(df) < 8:
        return pd.DataFrame(columns=["x", "y", "anomaly_type", "severity", "point_score", "diff_score"])

    y_median = float(df["y"].median())
    y_scale = _robust_scale(df["y"])

    diffs = df["y"].diff()
    diff_median = float(diffs.dropna().median()) if diffs.dropna().size else 0.0
    diff_scale = _robust_scale(diffs.dropna()) if diffs.dropna().size else 1.0

    df["point_score"] = (df["y"] - y_median).abs() / max(y_scale, 1e-9)
    df["diff_score"] = (diffs - diff_median).abs() / max(diff_scale, 1e-9)
    df["diff_score"] = df["diff_score"].fillna(0.0)

    anomaly_mask = (df["point_score"] >= 4.5) | (df["diff_score"] >= 5.0)
    anomalies = df.loc[anomaly_mask, ["x", "y", "point_score", "diff_score"]].copy()

    if anomalies.empty:
        return pd.DataFrame(columns=["x", "y", "anomaly_type", "severity", "point_score", "diff_score"])

    def classify_row(row: pd.Series) -> str:
        y_val = float(row["y"])
        diff_score = float(row["diff_score"])
        if y_val > y_median and diff_score >= 5.0:
            return "Spike"
        if y_val < y_median and diff_score >= 5.0:
            return "Drop"
        return "Outlier"

    def classify_severity(row: pd.Series) -> str:
        max_score = max(float(row["point_score"]), float(row["diff_score"]))
        if max_score >= 8.0:
            return "High"
        if max_score >= 6.0:
            return "Medium"
        return "Low"

    anomalies["anomaly_type"] = anomalies.apply(classify_row, axis=1)
    anomalies["severity"] = anomalies.apply(classify_severity, axis=1)

    return anomalies.reset_index(drop=True)




def build_anomaly_table_for_report(records: List[TrendRecord], metric_key: str) -> pd.DataFrame:
    rows = []

    for rec in records:
        df = detect_trend_anomalies(rec, metric_key)
        if df.empty:
            continue

        for _, row in df.iterrows():
            rows.append({
                "Timestamp": row["x"],
                "Signal": rec.point_clean,
                "Value": float(row["y"]),
                "Type": row["anomaly_type"],
                "Severity": row["severity"],
            })

    if not rows:
        return pd.DataFrame(columns=["Timestamp", "Signal", "Value", "Type", "Severity"])

    table = pd.DataFrame(rows)
    table = table.sort_values("Timestamp").reset_index(drop=True)

    return table


def build_anomaly_narrative(records: List[TrendRecord], metric_key: str) -> str:
    all_anomalies = []

    for rec in records:
        df = detect_trend_anomalies(rec, metric_key)
        if not df.empty:
            df = df.copy()
            df["record"] = rec.point_clean
            all_anomalies.append(df)

    if not all_anomalies:
        return "No se identifican eventos anómalos relevantes en la señal dentro de la ventana analizada."

    df_all = pd.concat(all_anomalies, ignore_index=True)

    total = len(df_all)
    spikes = int((df_all["anomaly_type"] == "Spike").sum())
    drops = int((df_all["anomaly_type"] == "Drop").sum())
    outliers = int((df_all["anomaly_type"] == "Outlier").sum())

    high = int((df_all["severity"] == "High").sum())
    medium = int((df_all["severity"] == "Medium").sum())
    low = int((df_all["severity"] == "Low").sum())

    # ------------------------------------------------------------
    # Clasificación de comportamiento
    # ------------------------------------------------------------
    if total >= 15:
        pattern = "recurrente"
    elif total >= 6:
        pattern = "intermitente"
    else:
        pattern = "aislado"

    # ------------------------------------------------------------
    # Tipo dominante
    # ------------------------------------------------------------
    if spikes > drops and spikes > outliers:
        dominant = "spikes (incrementos abruptos)"
    elif drops > spikes and drops > outliers:
        dominant = "drops (caídas abruptas)"
    else:
        dominant = "outliers dispersos"

    # ------------------------------------------------------------
    # Severidad dominante
    # ------------------------------------------------------------
    if high > 0:
        severity_text = "con presencia de eventos de alta severidad"
    elif medium > 0:
        severity_text = "con eventos de severidad moderada"
    else:
        severity_text = "predominantemente de baja severidad"

    # ------------------------------------------------------------
    # Interpretación técnica
    # ------------------------------------------------------------
    if pattern == "recurrente":
        interpretation = (
            "La recurrencia de eventos anómalos sugiere un comportamiento no aleatorio, "
            "posiblemente asociado a condiciones operativas repetitivas o a una condición mecánica persistente."
        )
    elif pattern == "intermitente":
        interpretation = (
            "Los eventos anómalos aparecen de forma intermitente, lo que puede estar asociado "
            "a cambios operativos, transitorios o perturbaciones externas."
        )
    else:
        interpretation = (
            "Los eventos detectados son aislados, sin patrón repetitivo claro, "
            "posiblemente asociados a ruido o perturbaciones puntuales."
        )

    # ------------------------------------------------------------
    # Construcción final
    # ------------------------------------------------------------
    narrative = (
        f"Se detectaron {total} eventos anómalos en la señal, clasificados como comportamiento {pattern}, "
        f"con predominio de {dominant} y {severity_text}. "
        f"{interpretation}"
    )

    return narrative


def build_panel_anomaly_summary(records: List[TrendRecord], metric_key: str) -> Dict[str, Any]:
    total_count = 0
    affected_records = 0
    top_severity = "None"
    details: List[Dict[str, Any]] = []

    severity_rank = {"None": 0, "Low": 1, "Medium": 2, "High": 3}

    for rec in records:
        anomalies = detect_trend_anomalies(rec, metric_key)
        count = int(len(anomalies))
        if count > 0:
            affected_records += 1
            total_count += count
            local_top = "Low"
            if "High" in set(anomalies["severity"]):
                local_top = "High"
            elif "Medium" in set(anomalies["severity"]):
                local_top = "Medium"

            if severity_rank.get(local_top, 0) > severity_rank.get(top_severity, 0):
                top_severity = local_top

            details.append(
                {
                    "record_name": rec.point_clean,
                    "count": count,
                    "top_severity": local_top,
                }
            )

    if total_count == 0:
        interpretation = "No se detectaron anomalías puntuales relevantes en la señal dentro de la ventana mostrada."
        color = "#16a34a"
    elif top_severity == "High":
        interpretation = "Se detectaron anomalías de alta severidad. Conviene revisar eventos transitorios, instrumentación o condición mecánica local."
        color = "#dc2626"
    elif top_severity == "Medium":
        interpretation = "Se detectaron anomalías moderadas. Conviene revisar cambios operativos o perturbaciones puntuales."
        color = "#f59e0b"
    else:
        interpretation = "Se detectaron anomalías leves y aisladas. Mantener seguimiento y correlacionar con operación."
        color = "#f97316"

    return {
        "total_count": total_count,
        "affected_records": affected_records,
        "top_severity": top_severity,
        "interpretation": interpretation,
        "color": color,
        "details": details,
    }



def build_lagged_correlation_analysis(
    trend_record: Optional[TrendRecord],
    operational_record: Optional[OperationalRecord],
    metric_key: str,
    max_lag_minutes: int = 180,
    step_minutes: int = 10,
) -> Dict[str, Any]:
    if trend_record is None or operational_record is None:
        return {
            "valid": False,
            "best_corr": None,
            "best_lag_min": None,
            "direction": "Indeterminada",
            "strength": "Nula",
            "interpretation": "Seleccione una señal de vibración y una variable operativa para habilitar el análisis con desfase.",
            "lag_df": pd.DataFrame(columns=["lag_min", "corr"]),
            "color": "#64748b",
        }

    base_df = align_trend_and_operational_for_correlation(
        trend_record=trend_record,
        operational_record=operational_record,
        metric_key=metric_key,
    )

    if base_df.empty or len(base_df) < 6:
        return {
            "valid": False,
            "best_corr": None,
            "best_lag_min": None,
            "direction": "Indeterminada",
            "strength": "Nula",
            "interpretation": "No hay suficientes puntos coincidentes para analizar correlación con desfase.",
            "lag_df": pd.DataFrame(columns=["lag_min", "corr"]),
            "color": "#64748b",
        }

    trend_df = get_clean_metric_df(trend_record, metric_key).rename(columns={"y": "trend"}).copy()
    op_df = get_operational_clean_df(operational_record).rename(columns={"y": "operational"}).copy()

    trend_df["x"] = pd.to_datetime(trend_df["x"], errors="coerce")
    op_df["x"] = pd.to_datetime(op_df["x"], errors="coerce")

    trend_df = trend_df.dropna(subset=["x", "trend"]).sort_values("x").reset_index(drop=True)
    op_df = op_df.dropna(subset=["x", "operational"]).sort_values("x").reset_index(drop=True)

    lag_rows = []
    for lag_min in range(-max_lag_minutes, max_lag_minutes + 1, step_minutes):
        shifted = op_df.copy()
        shifted["x"] = shifted["x"] + pd.Timedelta(minutes=lag_min)

        merged = pd.merge_asof(
            trend_df,
            shifted,
            on="x",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
        ).dropna(subset=["trend", "operational"]).reset_index(drop=True)

        corr_val = None
        if len(merged) >= 6:
            try:
                c = merged["trend"].corr(merged["operational"])
                if c is not None and math.isfinite(float(c)):
                    corr_val = float(c)
            except Exception:
                corr_val = None

        lag_rows.append(
            {
                "lag_min": lag_min,
                "corr": corr_val,
                "samples": int(len(merged)),
            }
        )

    lag_df = pd.DataFrame(lag_rows)
    valid_df = lag_df.dropna(subset=["corr"]).copy()

    if valid_df.empty:
        return {
            "valid": False,
            "best_corr": None,
            "best_lag_min": None,
            "direction": "Indeterminada",
            "strength": "Nula",
            "interpretation": "No fue posible calcular correlaciones válidas en la ventana de desfases.",
            "lag_df": lag_df,
            "color": "#64748b",
        }

    best_idx = valid_df["corr"].abs().idxmax()
    best_row = valid_df.loc[best_idx]
    best_corr = float(best_row["corr"])
    best_lag = int(best_row["lag_min"])

    meta = classify_correlation_strength(best_corr)

    if abs(best_lag) <= step_minutes:
        lag_meaning = "La relación parece prácticamente simultánea entre vibración y variable operativa."
    elif best_lag > 0:
        lag_meaning = (
            f"La mejor correlación aparece con un desfase de +{best_lag} min, "
            "lo que sugiere que la variable operativa antecede la respuesta vibratoria."
        )
    else:
        lag_meaning = (
            f"La mejor correlación aparece con un desfase de {best_lag} min, "
            "lo que sugiere que la vibración antecede a la variable operativa o que existe inversión temporal en el comportamiento."
        )

    interpretation = f"{meta['interpretation']} {lag_meaning}"

    return {
        "valid": True,
        "best_corr": best_corr,
        "best_lag_min": best_lag,
        "direction": meta["direction"],
        "strength": meta["strength"],
        "interpretation": interpretation,
        "lag_df": lag_df,
        "color": meta["color"],
        "trend_name": trend_record.point_clean,
        "operational_name": operational_record.variable,
    }


def build_lag_correlation_figure(lag_info: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    lag_df = lag_info.get("lag_df")

    if lag_df is None or not isinstance(lag_df, pd.DataFrame) or lag_df.empty:
        fig.update_layout(
            template="plotly_white",
            height=360,
            margin=dict(l=40, r=40, t=40, b=40),
            title="Lag Correlation",
        )
        return fig

    valid_df = lag_df.dropna(subset=["corr"]).copy()

    fig.add_trace(
        go.Scatter(
            x=lag_df["lag_min"],
            y=lag_df["corr"],
            mode="lines+markers",
            name="Correlation vs lag",
            line=dict(width=2.5),
            marker=dict(size=7),
            hovertemplate="Lag: %{x} min<br>Correlation: %{y:.4f}<extra></extra>",
        )
    )

    if not valid_df.empty:
        best_idx = valid_df["corr"].abs().idxmax()
        best_row = valid_df.loc[best_idx]
        fig.add_trace(
            go.Scatter(
                x=[best_row["lag_min"]],
                y=[best_row["corr"]],
                mode="markers",
                name="Best lag",
                marker=dict(size=13, color="#ef4444", symbol="diamond"),
                hovertemplate="Best lag: %{x} min<br>Correlation: %{y:.4f}<extra></extra>",
            )
        )

    trend_name = lag_info.get("trend_name") or "Trend"
    operational_name = lag_info.get("operational_name") or "Operational"

    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=40, r=40, t=50, b=50),
        title=f"Lag Correlation: {trend_name} vs {operational_name}",
        xaxis_title="Lag (minutes)",
        yaxis_title="Correlation",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    fig.add_hline(y=0.0, line_dash="dot", line_color="#94a3b8", line_width=1.4)
    return fig



def build_operational_correlation_report_block(
    trend_record: Optional[TrendRecord],
    operational_record: Optional[OperationalRecord],
    metric_key: str,
) -> str:
    if trend_record is None or operational_record is None:
        return ""

    corr_info = build_trend_operational_correlation(
        trend_record=trend_record,
        operational_record=operational_record,
        metric_key=metric_key,
    )
    lag_info = build_lagged_correlation_analysis(
        trend_record=trend_record,
        operational_record=operational_record,
        metric_key=metric_key,
        max_lag_minutes=180,
        step_minutes=10,
    )

    variable_name = operational_record.variable or "Operational variable"

    corr_txt = format_number(corr_info.get("corr_value"), 3)
    lag_corr_txt = format_number(lag_info.get("best_corr"), 3)
    lag_txt = lag_info.get("best_lag_min")
    lag_txt = str(lag_txt) if lag_txt is not None else "—"

    lines = [
        "Correlación operativa:",
        f"- Variable analizada: {variable_name}",
        f"- Correlación simple: {corr_txt}",
        f"- Fuerza: {corr_info.get('strength') or '—'}",
        f"- Dirección: {corr_info.get('direction') or '—'}",
        f"- Correlación con mejor lag: {lag_corr_txt}",
        f"- Mejor lag: {lag_txt} min",
        f"- Interpretación simple: {corr_info.get('interpretation') or 'Sin interpretación disponible.'}",
        f"- Interpretación con lag: {lag_info.get('interpretation') or 'Sin interpretación disponible.'}",
    ]
    return "\\n".join(lines)



def build_operational_variable_ranking(
    trend_record: Optional[TrendRecord],
    operational_records: List[OperationalRecord],
    metric_key: str,
) -> pd.DataFrame:
    if trend_record is None or not operational_records:
        return pd.DataFrame(
            columns=[
                "Variable",
                "Family",
                "Simple Corr",
                "Lag Corr",
                "Best Lag (min)",
                "Score",
                "Strength",
                "Direction",
                "Interpretation",
            ]
        )

    rows = []
    for op_rec in operational_records:
        corr_info = build_trend_operational_correlation(
            trend_record=trend_record,
            operational_record=op_rec,
            metric_key=metric_key,
        )
        lag_info = build_lagged_correlation_analysis(
            trend_record=trend_record,
            operational_record=op_rec,
            metric_key=metric_key,
            max_lag_minutes=180,
            step_minutes=10,
        )

        corr_val = corr_info.get("corr_value")
        lag_corr = lag_info.get("best_corr")
        lag_min = lag_info.get("best_lag_min")

        try:
            corr_abs = abs(float(corr_val)) if corr_val is not None and math.isfinite(float(corr_val)) else 0.0
        except Exception:
            corr_abs = 0.0

        try:
            lag_corr_abs = abs(float(lag_corr)) if lag_corr is not None and math.isfinite(float(lag_corr)) else 0.0
        except Exception:
            lag_corr_abs = 0.0

        try:
            lag_penalty = min(abs(int(lag_min)) / 180.0, 1.0) * 0.15 if lag_min is not None else 0.15
        except Exception:
            lag_penalty = 0.15

        score = (0.45 * corr_abs) + (0.65 * lag_corr_abs) - lag_penalty

        if score < 0:
            score = 0.0

        interpretation = lag_info.get("interpretation") or corr_info.get("interpretation") or "Sin interpretación disponible."

        rows.append(
            {
                "Variable": op_rec.variable,
                "Family": op_rec.family,
                "Simple Corr": corr_val,
                "Lag Corr": lag_corr,
                "Best Lag (min)": lag_min,
                "Score": score,
                "Strength": lag_info.get("strength") or corr_info.get("strength") or "Nula",
                "Direction": lag_info.get("direction") or corr_info.get("direction") or "Indeterminada",
                "Interpretation": interpretation,
            }
        )

    ranking_df = pd.DataFrame(rows)
    if ranking_df.empty:
        return ranking_df

    ranking_df = ranking_df.sort_values(
        by=["Score", "Lag Corr", "Simple Corr"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return ranking_df


def build_operational_variable_ranking_summary(ranking_df: pd.DataFrame) -> str:
    if ranking_df is None or ranking_df.empty:
        return (
            "No se identificó una variable operativa dominante que explique la vibración, "
            "ya sea por falta de datos o por ausencia de correlaciones confiables."
        )

    top = ranking_df.iloc[0]
    variable = str(top.get("Variable") or "—")
    family = str(top.get("Family") or "generic")
    strength = str(top.get("Strength") or "Nula")
    direction = str(top.get("Direction") or "Indeterminada")
    lag_corr = format_number(top.get("Lag Corr"), 3)
    lag_min = top.get("Best Lag (min)")
    lag_txt = str(int(lag_min)) if pd.notna(lag_min) else "—"

    return (
        f"La variable operativa que mejor explica el comportamiento vibratorio es {variable} "
        f"(familia {family}), con correlación dominante {strength.lower()} {direction.lower()} "
        f"y mejor correlación con desfase de {lag_corr}, observada en un lag de {lag_txt} min. "
        f"Esto sugiere que dicha variable tiene la mayor influencia operativa relativa sobre la vibración dentro de la ventana analizada."
    )





def detect_behavior_change(record: TrendRecord, metric_key: str) -> Dict[str, Any]:
    df = get_clean_metric_df(record, metric_key)

    if df.empty or len(df) < 20:
        return {
            "valid": False,
            "record_name": record.point_clean,
            "change_score": None,
            "classification": "Insufficient data",
            "change_timestamp": None,
            "interpretation": "No hay suficientes datos para evaluar cambio de comportamiento."
        }

    df = df.copy()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

    if len(df) < 20:
        return {
            "valid": False,
            "record_name": record.point_clean,
            "change_score": None,
            "classification": "Insufficient data",
            "change_timestamp": None,
            "interpretation": "No hay suficientes datos para evaluar cambio de comportamiento."
        }

    y = df["y"].to_numpy(dtype=float)
    x_ts = pd.to_datetime(df["x"], errors="coerce")

    min_segment = max(10, len(df) // 10)
    best_score = -1.0
    best_idx = None
    best_stats = None

    scale = max(float(np.mean(np.abs(y))), 1e-9)

    for split in range(min_segment, len(y) - min_segment):
        y1 = y[:split]
        y2 = y[split:]

        mean1, mean2 = float(np.mean(y1)), float(np.mean(y2))
        std1, std2 = float(np.std(y1)), float(np.std(y2))

        mean_change = abs(mean2 - mean1) / scale
        std_change = abs(std2 - std1) / scale
        local_score = mean_change + std_change

        if local_score > best_score:
            best_score = local_score
            best_idx = split
            best_stats = {
                "mean_before": mean1,
                "mean_after": mean2,
                "std_before": std1,
                "std_after": std2,
            }

    if best_idx is None or best_stats is None:
        return {
            "valid": False,
            "record_name": record.point_clean,
            "change_score": None,
            "classification": "Insufficient data",
            "change_timestamp": None,
            "interpretation": "No fue posible localizar un punto de cambio confiable."
        }

    change_ts = safe_datetime(x_ts.iloc[best_idx]) if best_idx < len(x_ts) else None

    if best_score > 0.35:
        classification = "Strong change"
    elif best_score > 0.18:
        classification = "Moderate change"
    else:
        classification = "No significant change"

    ts_txt = (
        f" alrededor de {pretty_date(change_ts)} {pretty_time(change_ts)}"
        if change_ts is not None else ""
    )

    if classification == "Strong change":
        interpretation = (
            f"Se detecta un cambio claro de comportamiento en la señal{ts_txt}, indicando transición entre dos regímenes "
            "operativos o modificación relevante de la condición de la máquina."
        )
    elif classification == "Moderate change":
        interpretation = (
            f"Se observa un cambio moderado en el comportamiento de la señal{ts_txt}, que podría estar asociado a "
            "variaciones operativas o evolución de la condición mecánica."
        )
    else:
        interpretation = (
            "No se identifican cambios significativos de comportamiento dentro de la ventana analizada."
        )

    return {
        "valid": True,
        "record_name": record.point_clean,
        "change_score": float(best_score),
        "classification": classification,
        "change_timestamp": change_ts,
        "mean_before": best_stats["mean_before"],
        "mean_after": best_stats["mean_after"],
        "std_before": best_stats["std_before"],
        "std_after": best_stats["std_after"],
        "interpretation": interpretation
    }

def build_behavior_change_summary(records: List[TrendRecord], metric_key: str) -> Dict[str, Any]:
    results = []
    for r in records:
        results.append(detect_behavior_change(r, metric_key))

    valid = [r for r in results if r.get("valid")]

    if not valid:
        return {
            "count": 0,
            "top_classification": "None",
            "interpretation": "No hay datos suficientes para evaluar cambios de comportamiento.",
            "details": results
        }

    strong = sum(1 for r in valid if r["classification"] == "Strong change")
    moderate = sum(1 for r in valid if r["classification"] == "Moderate change")

    if strong > 0:
        top = "Strong change"
        interpretation = (
            "Se detecta al menos un cambio fuerte de comportamiento en las señales, indicando transición clara de régimen."
        )
    elif moderate > 0:
        top = "Moderate change"
        interpretation = (
            "Se identifican cambios moderados de comportamiento, que sugieren variaciones operativas o evolución progresiva."
        )
    else:
        top = "No significant change"
        interpretation = (
            "No se detectan cambios relevantes de comportamiento en la ventana analizada."
        )

    return {
        "count": len(valid),
        "top_classification": top,
        "interpretation": interpretation,
        "details": results
    }



def build_behavior_narrative(records: List[TrendRecord], metric_key: str) -> str:
    summary = build_behavior_change_summary(records, metric_key)

    if summary["count"] == 0:
        return "No hay información suficiente para evaluar cambios de comportamiento."

    details = summary.get("details", [])
    valid = [d for d in details if d.get("valid") and d.get("classification") in ["Strong change", "Moderate change"]]

    if not valid:
        return summary["interpretation"]

    top = sorted(
        valid,
        key=lambda d: float(d.get("change_score") or 0.0),
        reverse=True
    )[0]

    change_ts = top.get("change_timestamp")
    ts_txt = (
        f"{pretty_date(change_ts)} {pretty_time(change_ts)}"
        if change_ts is not None else "sin timestamp identificable"
    )

    return (
        f"{summary['interpretation']} El cambio más representativo se localiza en la señal "
        f"{top.get('record_name', '—')} alrededor de {ts_txt}."
    )

def detect_trend_drift(record: TrendRecord, metric_key: str) -> Dict[str, Any]:
    df = get_clean_metric_df(record, metric_key).copy()
    if df.empty or len(df) < 8:
        return {
            "record_name": record.point_clean,
            "classification": "No Drift",
            "severity": "None",
            "change_pct": None,
            "slope_ratio": None,
            "direction": "Indeterminada",
            "interpretation": "No hay suficientes datos para evaluar drift.",
            "valid": False,
        }

    y = pd.to_numeric(df["y"], errors="coerce").dropna().astype(float).to_numpy()
    if y.size < 8:
        return {
            "record_name": record.point_clean,
            "classification": "No Drift",
            "severity": "None",
            "change_pct": None,
            "slope_ratio": None,
            "direction": "Indeterminada",
            "interpretation": "No hay suficientes datos para evaluar drift.",
            "valid": False,
        }

    x = np.arange(y.size, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted

    mean_abs = float(np.mean(np.abs(y)))
    value_span = float(np.max(y) - np.min(y))
    scale = max(mean_abs, value_span, 1e-9)

    slope_ratio = float(abs(slope) * max(y.size - 1, 1) / scale)
    volatility_ratio = float(np.std(residual) / scale)
    change_pct = safe_percent_change(float(y[0]), float(y[-1]))

    direction = "Increasing" if slope > 0 else "Decreasing"

    classification = "No Drift"
    severity = "None"
    interpretation = "No se observa deriva sostenida relevante."

    # Drift sostenido = pendiente importante con dispersión controlada
    if slope_ratio >= 0.18 and volatility_ratio <= 0.35:
        if direction == "Increasing":
            classification = "Progressive Increase"
        else:
            classification = "Progressive Decrease"

        if slope_ratio >= 0.45 or (change_pct is not None and abs(change_pct) >= 35):
            severity = "High"
        elif slope_ratio >= 0.28 or (change_pct is not None and abs(change_pct) >= 18):
            severity = "Medium"
        else:
            severity = "Low"

        if classification == "Progressive Increase":
            interpretation = (
                "La señal presenta deriva progresiva ascendente, compatible con incremento sostenido "
                "de la condición medida dentro de la ventana analizada."
            )
        else:
            interpretation = (
                "La señal presenta deriva progresiva descendente, compatible con reducción sostenida "
                "de la condición medida dentro de la ventana analizada."
            )

    return {
        "record_name": record.point_clean,
        "classification": classification,
        "severity": severity,
        "change_pct": change_pct,
        "slope_ratio": slope_ratio,
        "direction": direction,
        "interpretation": interpretation,
        "valid": True,
    }


def build_panel_drift_summary(records: List[TrendRecord], metric_key: str) -> Dict[str, Any]:
    rows = []
    for rec in records:
        rows.append(detect_trend_drift(rec, metric_key))

    if not rows:
        return {
            "total_drift_signals": 0,
            "top_severity": "None",
            "interpretation": "No hay señales disponibles para analizar drift.",
            "details": [],
        }

    drift_rows = [r for r in rows if r.get("classification") != "No Drift"]

    if not drift_rows:
        return {
            "total_drift_signals": 0,
            "top_severity": "None",
            "interpretation": "No se detecta deriva progresiva dominante en las señales seleccionadas.",
            "details": rows,
        }

    severity_rank = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    top_severity = "Low"
    for r in drift_rows:
        sev = str(r.get("severity") or "None")
        if severity_rank.get(sev, 0) > severity_rank.get(top_severity, 0):
            top_severity = sev

    if top_severity == "High":
        interpretation = (
            "Se detecta deriva progresiva de alta severidad en al menos una señal, "
            "lo que sugiere cambio sostenido de condición y amerita revisión prioritaria."
        )
    elif top_severity == "Medium":
        interpretation = (
            "Se detecta deriva progresiva moderada, compatible con evolución sostenida de la condición "
            "que conviene seguir de cerca."
        )
    else:
        interpretation = (
            "Se detecta deriva leve en la ventana analizada. Conviene monitorear si el patrón se consolida."
        )

    return {
        "total_drift_signals": len(drift_rows),
        "top_severity": top_severity,
        "interpretation": interpretation,
        "details": rows,
    }


def build_drift_narrative(records: List[TrendRecord], metric_key: str) -> str:
    summary = build_panel_drift_summary(records, metric_key)
    rows = summary.get("details", [])

    drift_rows = [r for r in rows if r.get("classification") != "No Drift"]
    if not drift_rows:
        return "No se identifican patrones de deriva progresiva relevantes en la ventana analizada."

    increasing = sum(1 for r in drift_rows if r.get("classification") == "Progressive Increase")
    decreasing = sum(1 for r in drift_rows if r.get("classification") == "Progressive Decrease")
    total = len(drift_rows)
    top_severity = summary.get("top_severity", "None")

    if increasing > decreasing:
        dominant = "deriva progresiva ascendente"
    elif decreasing > increasing:
        dominant = "deriva progresiva descendente"
    else:
        dominant = "deriva mixta sin una dirección dominante"

    return (
        f"Se identifican {total} señales con comportamiento de drift, con predominio de {dominant} "
        f"y severidad máxima {top_severity}. Esto sugiere un desplazamiento sostenido de la línea base "
        f"más allá de eventos puntuales, por lo que conviene revisar evolución temporal, carga y condición mecánica."
    )


def _compute_trend_health(
    records: List[TrendRecord],
    metric_key: str,
    *,
    warning_value: Optional[float],
    danger_value: Optional[float],
) -> Dict[str, Any]:
    """
    Ciclo 17.5 P3 — Computes a compact health snapshot for the
    trend figure header. Returns:

      {
        "status":          "ok" | "watch" | "alarm" | "action" | "unknown",
        "status_label":    "Normal" | "Vigilancia" | ...,
        "max_value":        float,
        "latest_value":     float,
        "slope_per_day":    float,
        "forecast_days":    Optional[float],   # days hasta alcanzar
                                               # warning a la pendiente
                                               # actual; None si la
                                               # pendiente no avanza al
                                               # umbral
        "forecast_target":  "warning" | "danger" | None
      }
    """
    out: Dict[str, Any] = {
        "status": "unknown",
        "status_label": "Sin datos",
        "max_value": float("nan"),
        "latest_value": float("nan"),
        "slope_per_day": float("nan"),
        "forecast_days": None,
        "forecast_target": None,
    }

    if not records:
        return out

    # 1) Reunir todas las muestras del métrico en un único stream
    merged_x: List[pd.Timestamp] = []
    merged_y: List[float] = []
    for rec in records:
        df = get_clean_metric_df(rec, metric_key)
        if df.empty:
            continue
        merged_x.extend(list(df["x"]))
        merged_y.extend([float(v) for v in df["y"]])

    if not merged_y:
        return out

    s = pd.Series(merged_y, index=pd.to_datetime(pd.Series(merged_x)))
    s = s[~s.index.isna()].sort_index()
    if s.empty:
        return out

    out["max_value"] = float(s.max())
    out["latest_value"] = float(s.iloc[-1])

    # 2) Clasificar status según warning/danger
    #
    # Ciclo 17.5.9 — antes el chip se computaba SOLO sobre
    # latest_value (último sample), lo que daba "Normal" cuando el
    # último punto estaba debajo de Warning aunque la ventana tuviera
    # múltiples picos sobre Danger. Reportado por el usuario:
    # "aparece Normal si esta arriba los datos de la alarma".
    #
    # Fix: clasificar por el WORST de la ventana reciente (último
    # 7 días o últimos 100 samples si la ventana es densa). Si
    # cualquier punto reciente cruza Danger → action; cualquiera
    # cruza Warning → alarm. La latest sigue siendo lo que se
    # reporta como número, pero el status refleja el peor reciente.
    has_warning = warning_value is not None and math.isfinite(float(warning_value))
    has_danger = danger_value is not None and math.isfinite(float(danger_value))

    # Ventana reciente para clasificación
    try:
        _last_ts = pd.Timestamp(s.index.max())
        _recent_cutoff = _last_ts - pd.Timedelta(days=7)
        _recent = s[s.index >= _recent_cutoff]
        if len(_recent) < 30 and len(s) > 30:
            _recent = s.tail(min(100, len(s)))
        if len(_recent) == 0:
            _recent = s
        recent_max = float(_recent.max())
    except Exception:
        recent_max = float(s.max())

    out["recent_max_value"] = recent_max
    latest = out["latest_value"]

    if has_danger and recent_max >= float(danger_value):
        out["status"] = "action"
        out["status_label"] = "Acción Requerida"
    elif has_warning and recent_max >= float(warning_value):
        out["status"] = "alarm"
        out["status_label"] = "Atención"
    elif has_warning and (
        recent_max >= 0.85 * float(warning_value)
        or latest >= 0.85 * float(warning_value)
    ):
        out["status"] = "watch"
        out["status_label"] = "Vigilancia"
    elif has_warning or has_danger:
        out["status"] = "ok"
        out["status_label"] = "Normal"
    else:
        out["status"] = "ok"
        out["status_label"] = "Sin umbrales"

    # 3) Pendiente por linealizar contra tiempo (días) + forecast
    #    Ciclo 17.5.7 — endurecemos la validez del forecast:
    #
    #    El usuario reportó "cruce de umbral proyectado en ~0 días"
    #    cuando la corrida es un transient de arranque (data 2 horas,
    #    7088% de variación, slope +45.9 mil pp/día). En ese régimen
    #    el slope lineal NO representa una tendencia operacional —
    #    extrapolarlo da números físicamente irreales.
    #
    #    Reglas de invalidación:
    #      a) Ventana total < 24h → no hay base estadística para
    #         proyectar a múltiples días. Suprimimos forecast.
    #      b) Coeficiente de variación de la cola > 50% → cola
    #         altamente inestable (transitorio o swing operacional);
    #         el slope lineal no es representativo.
    #      c) days_to_target < 0.5 días → físicamente irreal salvo
    #         que la máquina esté en runaway. Suprimimos en lugar de
    #         redondear a "0 días".
    try:
        # Tomar últimos 60 puntos para evitar dominancia de épocas viejas
        tail = s.tail(60).copy()
        if len(tail) >= 4:
            t0 = pd.Timestamp(tail.index.min())
            x_days = (tail.index - t0).total_seconds() / 86400.0
            y_vals = tail.values
            slope, intercept = np.polyfit(x_days, y_vals, 1)
            out["slope_per_day"] = float(slope)

            # ---------------------------------------------------
            # Validar si el slope es proyectable como tendencia
            # ---------------------------------------------------
            # Ventana total de la serie completa (no solo cola)
            try:
                full_span_days = float(
                    (s.index.max() - s.index.min()).total_seconds() / 86400.0
                )
            except Exception:
                full_span_days = 0.0

            # Coef. de variación de la cola (estabilidad)
            try:
                _tail_mean = float(np.mean(y_vals))
                _tail_std = float(np.std(y_vals))
                _cv = _tail_std / abs(_tail_mean) if abs(_tail_mean) > 1e-9 else float("inf")
            except Exception:
                _cv = float("inf")

            forecast_is_meaningful = True
            if full_span_days < 1.0:
                forecast_is_meaningful = False
            if _cv > 0.50:
                forecast_is_meaningful = False

            # 4) Forecast: días hasta llegar a Warning (o Danger si ya
            # estamos sobre Warning)
            target_val: Optional[float] = None
            target_lbl: Optional[str] = None
            if has_warning and latest < float(warning_value):
                target_val = float(warning_value)
                target_lbl = "warning"
            elif has_danger and latest < float(danger_value):
                target_val = float(danger_value)
                target_lbl = "danger"

            if (
                forecast_is_meaningful
                and target_val is not None
                and slope > 1e-9
            ):
                # último punto x en días
                x_last = float((tail.index.max() - t0).total_seconds() / 86400.0)
                y_last = float(slope * x_last + intercept)
                if y_last < target_val:
                    days_to_target = (target_val - y_last) / slope
                    # Físicamente irreal: cruce en menos de medio día
                    # con datos limitados. Suprimimos.
                    if (
                        math.isfinite(days_to_target)
                        and days_to_target >= 0.5
                    ):
                        out["forecast_days"] = float(days_to_target)
                        out["forecast_target"] = target_lbl
    except Exception:
        pass

    return out


def _draw_health_chip(
    fig: go.Figure,
    health: Dict[str, Any],
    *,
    show_below_strip: bool = True,
) -> None:
    """Dibuja un chip de salud (Normal / Vigilancia / Atención /
    Acción Requerida) en la parte superior derecha del strip,
    coloreado según el estado y con la pendiente y forecast en
    pequeño debajo."""
    status = health.get("status", "unknown")
    label = health.get("status_label", "—")

    palette = {
        "ok":      ("#10b981", "#ecfdf5", "#065f46"),  # verde
        "watch":   ("#0ea5e9", "#e0f2fe", "#075985"),  # azul
        "alarm":   ("#f59e0b", "#fef3c7", "#78350f"),  # ámbar
        "action":  ("#ef4444", "#fee2e2", "#7f1d1d"),  # rojo
        "unknown": ("#9ca3af", "#f1f5f9", "#1f2937"),  # gris
    }
    border, fill, text_color = palette.get(status, palette["unknown"])

    # Posición del chip (esquina superior derecha del strip)
    cx0, cx1 = 0.846, 0.978
    cy0, cy1 = 1.122, 1.184  # justo arriba del strip

    fig.add_shape(
        type="path",
        xref="paper", yref="paper",
        path=rounded_rect_path(cx0, cy0, cx1, cy1, 0.020),
        line=dict(color=border, width=1.4),
        fillcolor=fill,
        layer="above",
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=(cx0 + cx1) / 2.0, y=(cy0 + cy1) / 2.0,
        xanchor="center", yanchor="middle",
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(size=11.6, color=text_color),
    )

    if not show_below_strip:
        return

    # Línea informativa debajo del chip — pendiente y forecast
    slope = health.get("slope_per_day", float("nan"))
    forecast_days = health.get("forecast_days")
    forecast_target = health.get("forecast_target")

    bits: List[str] = []
    if isinstance(slope, (int, float)) and math.isfinite(slope):
        if abs(slope) < 1e-9:
            bits.append("pendiente: estable")
        else:
            arrow = "↑" if slope > 0 else "↓"
            bits.append(f"pendiente {arrow} {abs(slope):.3g}/día")
    # Ciclo 17.5.7 — solo mostramos forecast cuando es válido
    # (>= 0.5 días, ventana >= 24h, varianza estable). El
    # _compute_trend_health ya filtra esos casos a None.
    if forecast_days is not None and forecast_target is not None:
        if 0.5 <= forecast_days < 365 * 5:
            if forecast_days < 1.5:
                bits.append(f"~1 día → {forecast_target}")
            else:
                bits.append(f"~{forecast_days:.0f} días → {forecast_target}")

    info_text = " · ".join(bits)
    if info_text:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=(cx0 + cx1) / 2.0, y=cy0 - 0.018,
            xanchor="center", yanchor="top",
            text=f"<i>{info_text}</i>",
            showarrow=False,
            font=dict(size=10.0, color="#475569"),
        )


def build_trend_autodiagnostic(
    records: List[TrendRecord],
    metric_key: str,
    *,
    warning_value: Optional[float],
    danger_value: Optional[float],
    operational_records: Optional[List[OperationalRecord]] = None,
) -> Dict[str, Any]:
    """
    Ciclo 17.5 P4 — autodiagnóstico ejecutivo del trend, en prosa
    estilo Bently Nevada Technical Training. Sintetiza:

      - Estado vs umbrales (Normal / Vigilancia / Atención / Acción)
      - Pendiente del último tramo + forecast a Warning / Danger
      - Anomalías puntuales (cantidad, tipo, severidad)
      - Drift progresivo (deriva)
      - Cambio de comportamiento (régimen)
      - Vínculo operacional cuando hay variables operativas

    El output es un dict con:
        status:        "ok" | "watch" | "alarm" | "action" | "unknown"
        status_label:  texto humano del status
        headline:      una frase ejecutiva (≤ 35 palabras)
        prose:         lista de párrafos en prosa Bently/ISO 20816
        recommendations: lista de acciones recomendadas

    Vocabulario alineado con Bently Nevada Technical Training y
    los criterios de severidad de ISO 20816 / API 670 §6.7.
    """
    out: Dict[str, Any] = {
        "status": "unknown",
        "status_label": "Sin datos",
        "headline": "No hay suficientes datos para emitir autodiagnóstico.",
        "prose": [],
        "recommendations": [],
    }

    if not records:
        return out

    health = _compute_trend_health(
        records, metric_key,
        warning_value=warning_value,
        danger_value=danger_value,
    )
    out["status"] = health["status"]
    out["status_label"] = health["status_label"]

    latest_value = health["latest_value"]
    max_value = health["max_value"]
    recent_max_value = health.get("recent_max_value", max_value)
    slope = health["slope_per_day"]
    forecast_days = health["forecast_days"]
    forecast_target = health["forecast_target"]

    # -------------------------------------------------------------
    # Construir headline + descripción de signal/punto
    # -------------------------------------------------------------
    n_records = len(records)
    primary = records[0]
    metric_unit = (get_metric_series(primary, metric_key)[1] or "").strip()
    point_label = primary.point_clean

    if n_records == 1:
        signal_descriptor = f"el punto «{point_label}»"
    else:
        signal_descriptor = (
            f"{n_records} puntos de medición sobre el activo «{primary.machine}»"
        )

    # -------------------------------------------------------------
    # 1) Encabezado + estado vs umbrales
    # -------------------------------------------------------------
    par1: List[str] = []
    if not math.isnan(latest_value):
        par1.append(
            f"El último valor reportado de {metric_key.lower()} en {signal_descriptor} "
            f"es {latest_value:.3g} {metric_unit}".rstrip() + "."
        )

    if warning_value is not None and math.isfinite(float(warning_value)):
        pct_w = (latest_value / float(warning_value) * 100.0) if float(warning_value) > 0 else 0.0
        par1.append(
            f"Esto representa el {pct_w:.0f}% del umbral Warning "
            f"({float(warning_value):.3g} {metric_unit})".rstrip() + "."
        )
    if danger_value is not None and math.isfinite(float(danger_value)):
        pct_d = (latest_value / float(danger_value) * 100.0) if float(danger_value) > 0 else 0.0
        par1.append(
            f"Frente al umbral Danger ({float(danger_value):.3g} {metric_unit}) "
            f"el consumo es del {pct_d:.0f}%".rstrip() + "."
        )

    # Ciclo 17.5.9 — si el último valor es bajo PERO la ventana
    # reciente tuvo un pico que superó Warning/Danger, lo
    # explicitamos para que el lector no piense que el sistema
    # ignoró los picos.
    if (
        not math.isnan(recent_max_value)
        and recent_max_value > latest_value * 1.5  # pico significativo arriba del latest
    ):
        _exceed_what = ""
        if danger_value is not None and recent_max_value >= float(danger_value):
            _exceed_what = f"superando el umbral Danger ({float(danger_value):.3g} {metric_unit})".rstrip()
        elif warning_value is not None and recent_max_value >= float(warning_value):
            _exceed_what = f"superando el umbral Warning ({float(warning_value):.3g} {metric_unit})".rstrip()
        if _exceed_what:
            par1.append(
                f"Sin embargo, dentro de la ventana reciente se registró un pico "
                f"de {recent_max_value:.3g} {metric_unit} ".rstrip()
                + f", {_exceed_what}; el estado se clasifica según el peor valor "
                f"reciente, no únicamente el último sample."
            )

    status = out["status"]
    if status == "action":
        par1.append(
            "El nivel actual supera el umbral Danger establecido; según los criterios "
            "de ISO 20816 y de los manuales de fábrica esto corresponde a la zona D — "
            "se recomienda parada para inspección o reducción de carga inmediata."
        )
    elif status == "alarm":
        par1.append(
            "El nivel actual cruza el umbral Warning (zona C de ISO 20816) — la máquina "
            "no debería operar de forma continua bajo esta amplitud sin un programa "
            "explícito de seguimiento condicional."
        )
    elif status == "watch":
        par1.append(
            "El nivel se encuentra entre el 85% y el 100% del Warning, en una zona de "
            "vigilancia prudente; conviene aumentar la frecuencia de monitoreo y "
            "documentar las condiciones de operación de cada toma."
        )
    elif status == "ok":
        par1.append(
            "El nivel se encuentra dentro de la zona operacional normal de los "
            "criterios de severidad establecidos."
        )

    # -------------------------------------------------------------
    # 2) Pendiente + forecast
    # -------------------------------------------------------------
    par2: List[str] = []
    if isinstance(slope, (int, float)) and math.isfinite(slope):
        if abs(slope) < 1e-9:
            par2.append(
                "La pendiente del último tramo es prácticamente plana, lo que sugiere "
                "un régimen estable de la señal — sin tendencia direccional clara."
            )
        elif slope > 0:
            par2.append(
                f"La pendiente del último tramo es positiva, +{slope:.3g} "
                f"{metric_unit}/día, evidenciando un crecimiento gradual de la "
                f"amplitud."
            )
        else:
            par2.append(
                f"La pendiente del último tramo es negativa, {slope:.3g} "
                f"{metric_unit}/día — la señal disminuye con el tiempo, lo que "
                f"puede asociarse a estabilización post-mantenimiento, "
                f"asentamiento térmico o redistribución de carga del rotor."
            )

        # Ciclo 17.5.7 — solo emitimos forecast si _compute_trend_health
        # lo validó (>= 0.5 días, ventana >= 24h, varianza estable).
        # Si no es válido, agregamos una caveat explícita en lugar de
        # mostrar "0 días" o un horizonte inventado.
        if forecast_days is not None and forecast_target is not None:
            _fcast_int = max(1, int(round(float(forecast_days))))
            if forecast_days < 14:
                par2.append(
                    f"Si la pendiente actual se mantiene, el umbral {forecast_target} "
                    f"se alcanzaría en aproximadamente {_fcast_int} día(s) — "
                    f"horizonte corto que justifica intervención preventiva."
                )
            elif forecast_days < 60:
                par2.append(
                    f"Manteniendo la pendiente actual, el umbral {forecast_target} "
                    f"sería alcanzado en aproximadamente {_fcast_int} días, "
                    f"lo que permite planificar una intervención dentro del próximo "
                    f"ciclo de mantenimiento."
                )
            elif forecast_days < 365:
                par2.append(
                    f"El forecast lineal sitúa el cruce del umbral {forecast_target} "
                    f"a unos {_fcast_int} días — horizonte cómodo, pero "
                    f"conviene reevaluar la pendiente con la próxima corrida."
                )
        elif (
            isinstance(slope, (int, float))
            and math.isfinite(slope)
            and abs(slope) > 1e-9
        ):
            # Hay pendiente real pero el forecast fue invalidado por el
            # validador. Lo decimos honestamente en lugar de inventar
            # una proyección.
            par2.append(
                "La ventana actual es demasiado corta o la cola es "
                "demasiado inestable como para emitir un forecast lineal "
                "confiable a Warning/Danger; se sugiere repetir la "
                "medición en condiciones operacionales estables y con "
                "al menos 24 horas de datos para construir una "
                "proyección representativa."
            )

    # -------------------------------------------------------------
    # 3) Anomalías puntuales
    # -------------------------------------------------------------
    par3: List[str] = []
    try:
        anom = build_panel_anomaly_summary(records, metric_key)
        n_anom = int(anom.get("total_count", 0) or 0)
        top_sev = str(anom.get("top_severity", "None") or "None")
        if n_anom > 0:
            if top_sev == "High":
                par3.append(
                    f"Se identifican {n_anom} eventos puntuales en la ventana, con "
                    f"presencia de eventos de alta severidad — patrón sugestivo de "
                    f"transitorios mecánicos, eventos de instrumentación o fallas "
                    f"locales del cojinete que merecen revisión específica."
                )
            elif top_sev == "Medium":
                par3.append(
                    f"Se identifican {n_anom} eventos puntuales, con severidad "
                    f"moderada predominante — pueden estar asociados a cambios "
                    f"operacionales, transitorios de carga o perturbaciones puntuales "
                    f"del proceso."
                )
            else:
                par3.append(
                    f"Se identifican {n_anom} eventos puntuales de baja severidad — "
                    f"compatibles con ruido de medición o perturbaciones aisladas, "
                    f"sin patrón mecánico claro."
                )
        # Si no hay anomalías significativas, simplemente no agregamos
        # nada — el lector lo asume del estado general.
    except Exception:
        pass

    # -------------------------------------------------------------
    # 4) Drift progresivo
    # -------------------------------------------------------------
    par4: List[str] = []
    try:
        drift = build_panel_drift_summary(records, metric_key)
        n_drift = int(drift.get("total_drift_signals", 0) or 0)
        top_drift = str(drift.get("top_severity", "None") or "None")
        if n_drift > 0 and top_drift in ("High", "Medium"):
            if top_drift == "High":
                par4.append(
                    f"El detector de deriva progresiva clasifica {n_drift} "
                    f"señal(es) en severidad alta. Esto refleja un cambio "
                    f"sostenido de tendencia (no eventos puntuales) y suele "
                    f"asociarse a procesos lentos: desgaste, drift térmico del "
                    f"sistema, deriva de la instrumentación o evolución progresiva "
                    f"del balance dinámico."
                )
            else:
                par4.append(
                    f"El detector de deriva progresiva señala {n_drift} señal(es) "
                    f"en severidad media — una evolución gradual pero todavía "
                    f"contenida, recomendable seguir bajo vigilancia condicional."
                )
    except Exception:
        pass

    # -------------------------------------------------------------
    # 5) Cambio de régimen (behavior change)
    # -------------------------------------------------------------
    par5: List[str] = []
    try:
        behav = build_behavior_change_summary(records, metric_key)
        top_class = str(behav.get("top_classification", "None") or "None")
        if top_class == "Strong change":
            par5.append(
                "Adicionalmente, el detector de cambio de régimen identifica un "
                "salto fuerte de comportamiento — la señal cruza un punto de "
                "inflexión claro, lo que sugiere un evento puntual (cambio de "
                "carga importante, intervención mecánica, falla incipiente) que "
                "redefine el promedio operacional."
            )
        elif top_class == "Moderate change":
            par5.append(
                "El detector de cambio de régimen reporta una transición "
                "moderada, compatible con ajuste operacional o evolución "
                "progresiva del proceso."
            )
    except Exception:
        pass

    # -------------------------------------------------------------
    # 6) Vínculo operacional
    # -------------------------------------------------------------
    par6: List[str] = []
    if operational_records:
        op_var_names = sorted({r.variable for r in operational_records if r.variable})
        if op_var_names:
            shown = ", ".join(op_var_names[:3])
            extra = " y otras" if len(op_var_names) > 3 else ""
            par6.append(
                f"En esta corrida se cuenta con variables operativas correlacionadas "
                f"({shown}{extra}). Se recomienda revisar el panel de correlación "
                f"con desfase para verificar si la evolución de la señal sigue a "
                f"un parámetro de proceso (carga, temperatura, RPM) — esto permite "
                f"distinguir entre cambio de régimen operacional y degradación "
                f"mecánica intrínseca."
            )

    # -------------------------------------------------------------
    # Recomendaciones por status
    # -------------------------------------------------------------
    recs: List[str] = []
    if status == "action":
        recs.append("Coordinar parada planificada o reducción de carga inmediata para inspección.")
        recs.append("Capturar espectro y forma de onda en condiciones actuales para confirmar la fuente del incremento.")
        recs.append("Verificar tendencia de centerline y acoplamientos asociados al punto comprometido.")
    elif status == "alarm":
        recs.append("Aumentar frecuencia de monitoreo (diaria si es posible) y registrar evolución bajo condiciones operacionales conocidas.")
        recs.append("Programar inspección dirigida en el próximo paro programado.")
        recs.append("Evaluar correlación con cambios recientes de carga, temperatura o composición de fluido de proceso.")
    elif status == "watch":
        recs.append("Mantener seguimiento semanal y documentar las condiciones de cada toma.")
        recs.append("Si la pendiente se mantiene positiva en la siguiente corrida, escalar a Atención.")
    elif status == "ok":
        recs.append("Continuar con el plan rutinario de monitoreo periódico.")
        recs.append("Conservar la línea base actual como referencia post-mantenimiento.")

    if forecast_days is not None and forecast_target is not None and forecast_days < 60:
        _fcast_rec = max(1, int(round(float(forecast_days))))
        recs.append(
            f"Programar inspección antes de los {_fcast_rec} día(s) de forecast "
            f"al cruce del umbral {forecast_target}."
        )

    # -------------------------------------------------------------
    # Headline ejecutivo
    # -------------------------------------------------------------
    if status == "action":
        out["headline"] = (
            f"Estado: ACCIÓN REQUERIDA. {metric_key} en zona D "
            f"({latest_value:.3g} {metric_unit}); supera el umbral Danger."
        ).strip()
    elif status == "alarm":
        out["headline"] = (
            f"Estado: ATENCIÓN. {metric_key} cruza Warning "
            f"({latest_value:.3g} {metric_unit}); requiere monitoreo intensivo."
        ).strip()
    elif status == "watch":
        out["headline"] = (
            f"Estado: VIGILANCIA. {metric_key} consume el 85–100% del Warning."
        ).strip()
    elif status == "ok":
        if (
            isinstance(slope, (int, float))
            and math.isfinite(slope)
            and slope > 0
            and forecast_days is not None
            and forecast_days < 90
        ):
            _fcast_hl = max(1, int(round(float(forecast_days))))
            out["headline"] = (
                f"Estado: NORMAL con tendencia ascendente; cruce de umbral "
                f"proyectado en ~{_fcast_hl} día(s)."
            )
        elif (
            isinstance(slope, (int, float))
            and math.isfinite(slope)
            and slope > 0
            and forecast_days is None
        ):
            # Hay pendiente positiva pero el forecast fue invalidado
            # (ventana <24h o cola inestable). Headline honesto.
            out["headline"] = (
                "Estado: NORMAL con tendencia ascendente; ventana actual "
                "insuficiente para emitir un forecast confiable."
            )
        else:
            out["headline"] = "Estado: NORMAL. Señal dentro de la zona operacional sin tendencia preocupante."
    else:
        out["headline"] = "Sin umbrales definidos; el autodiagnóstico se limita a la descripción estadística."

    # -------------------------------------------------------------
    # Componer prosa final
    # -------------------------------------------------------------
    prose: List[str] = []
    for block in (par1, par2, par3, par4, par5, par6):
        joined = " ".join([s.strip() for s in block if s.strip()])
        if joined:
            prose.append(joined)

    out["prose"] = prose
    out["recommendations"] = recs
    return out


def build_trend_figure(
    records: List[TrendRecord],
    metric_key: str,
    show_markers: bool,
    show_anomaly_markers: bool,
    fill_area: bool,
    y_axis_mode: str,
    y_axis_manual_min: Optional[float],
    y_axis_manual_max: Optional[float],
    x_axis_mode: str,
    x_axis_manual_start: Optional[pd.Timestamp],
    x_axis_manual_end: Optional[pd.Timestamp],
    warning_enabled: bool,
    warning_value: Optional[float],
    danger_enabled: bool,
    danger_value: Optional[float],
    show_right_info_box: bool,
    show_legend: bool,
    logo_uri: Optional[str],
    cursor_map: Dict[str, Optional[pd.Timestamp]],
    operational_records: Optional[List[OperationalRecord]] = None,
    mixed_mode: bool = False,
    operational_only_mode: bool = False,
    operational_y_axis_mode: str = "Auto",
    operational_y_manual_min: Optional[float] = None,
    operational_y_manual_max: Optional[float] = None,
) -> go.Figure:
    operational_records = operational_records or []
    use_secondary_axis = mixed_mode and len(records) > 0 and len(operational_records) > 0
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if use_secondary_axis else go.Figure()

    visible_records: List[Tuple[TrendRecord, pd.DataFrame, str]] = []
    visible_operational_records: List[Tuple[OperationalRecord, pd.DataFrame]] = []

    global_y_min = np.inf
    global_y_max = -np.inf
    global_x_min: Optional[pd.Timestamp] = None
    global_x_max: Optional[pd.Timestamp] = None

    operational_y_min = np.inf
    operational_y_max = -np.inf

    if not operational_only_mode:
        for idx, record in enumerate(records):
            df = get_clean_metric_df(record, metric_key)
            metric_unit = get_metric_series(record, metric_key)[1]
            if df.empty:
                continue

            color = color_for_index(idx)
            mode = "lines+markers" if show_markers else "lines"

            trace = go.Scattergl(
                x=df["x"],
                y=df["y"],
                mode=mode,
                line=dict(width=2.4, color=color),
                marker=dict(size=5, color=color),
                fill="tozeroy" if fill_area and len(records) == 1 and not use_secondary_axis else None,
                fillcolor="rgba(91, 156, 240, 0.10)" if fill_area and len(records) == 1 and not use_secondary_axis else None,
                name=record.point_clean,
                hovertemplate=("Point: %{fullData.name}<br>" "Time: %{x}<br>" f"{metric_key}: " + "%{y:.4f} " + f"{metric_unit}" + "<extra></extra>"),
                showlegend=show_legend,
                connectgaps=False,
            )

            if use_secondary_axis:
                fig.add_trace(trace, secondary_y=False)
            else:
                fig.add_trace(trace)

            if show_anomaly_markers:
                anomaly_df = detect_trend_anomalies(record, metric_key)
                if not anomaly_df.empty:
                    # Ciclo 17.5 — marcadores sutiles: círculos
                    # huecos pequeños semitransparentes que ya no
                    # compiten con la curva ni saturan el plot.
                    # Las severidades High siguen destacando con
                    # un anillo más opaco; las Low/Medium quedan
                    # como puntos discretos.
                    anomaly_df = anomaly_df.copy()
                    sev_colors = {
                        "High":   "rgba(220, 38, 38, 0.85)",
                        "Medium": "rgba(245, 158, 11, 0.70)",
                        "Low":    "rgba(100, 116, 139, 0.55)",
                    }
                    sev_sizes = {"High": 9, "Medium": 7, "Low": 6}
                    point_colors = [
                        sev_colors.get(str(s), "rgba(100,116,139,0.55)")
                        for s in anomaly_df["severity"].astype(str)
                    ]
                    point_sizes = [
                        sev_sizes.get(str(s), 6)
                        for s in anomaly_df["severity"].astype(str)
                    ]
                    anomaly_trace = go.Scatter(
                        x=anomaly_df["x"],
                        y=anomaly_df["y"],
                        mode="markers",
                        name=f"Anomalies — {record.point_clean}",
                        marker=dict(
                            size=point_sizes,
                            color=point_colors,
                            symbol="circle-open",
                            line=dict(width=1.4, color=point_colors),
                        ),
                        hovertemplate=(
                            "Point: %{fullData.name}<br>"
                            "Time: %{x}<br>"
                            "Value: %{y:.4f}<br>"
                            "Anomaly detected<extra></extra>"
                        ),
                        showlegend=show_legend,
                        opacity=0.85,
                    )
                    if use_secondary_axis:
                        fig.add_trace(anomaly_trace, secondary_y=False)
                    else:
                        fig.add_trace(anomaly_trace)

            y_min_local = float(df["y"].min())
            y_max_local = float(df["y"].max())
            global_y_min = min(global_y_min, y_min_local)
            global_y_max = max(global_y_max, y_max_local)

            x_min_local = pd.Timestamp(df["x"].min())
            x_max_local = pd.Timestamp(df["x"].max())
            global_x_min = x_min_local if global_x_min is None else min(global_x_min, x_min_local)
            global_x_max = x_max_local if global_x_max is None else max(global_x_max, x_max_local)

            visible_records.append((record, df, metric_unit))

    operational_start_idx = len(records)
    for idx, record in enumerate(operational_records):
        df = get_operational_clean_df(record)
        if df.empty:
            continue

        color = color_for_index(operational_start_idx + idx)
        mode = "lines+markers" if show_markers else "lines"
        trace = go.Scattergl(
            x=df["x"],
            y=df["y"],
            mode=mode,
            line=dict(width=2.2, color=color, dash="dot" if mixed_mode else "solid"),
            marker=dict(size=5, color=color),
            name=record.variable,
            hovertemplate=("Signal: %{fullData.name}<br>" "Time: %{x}<br>" "Value: %{y:.4f} " + f"{record.unit}" + "<extra></extra>"),
            showlegend=show_legend,
            connectgaps=False,
        )

        if use_secondary_axis:
            fig.add_trace(trace, secondary_y=True)
        else:
            fig.add_trace(trace)

        y_min_local = float(df["y"].min())
        y_max_local = float(df["y"].max())
        if use_secondary_axis or operational_only_mode:
            operational_y_min = min(operational_y_min, y_min_local)
            operational_y_max = max(operational_y_max, y_max_local)
        else:
            global_y_min = min(global_y_min, y_min_local)
            global_y_max = max(global_y_max, y_max_local)

        x_min_local = pd.Timestamp(df["x"].min())
        x_max_local = pd.Timestamp(df["x"].max())
        global_x_min = x_min_local if global_x_min is None else min(global_x_min, x_min_local)
        global_x_max = x_max_local if global_x_max is None else max(global_x_max, x_max_local)

        visible_operational_records.append((record, df))

    if not visible_records and not visible_operational_records:
        fig.update_layout(height=640, plot_bgcolor="#f8fafc", paper_bgcolor="#f3f4f6", margin=dict(l=46, r=18, t=84, b=40))
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="No valid trend data available", showarrow=False, font=dict(size=18, color="#6b7280"))
        return fig

    def _pad_axis(ymin: float, ymax: float) -> Tuple[float, float]:
        if not math.isfinite(ymin):
            ymin = 0.0
        if not math.isfinite(ymax):
            ymax = 1.0
        if math.isclose(ymin, ymax, rel_tol=1e-12, abs_tol=1e-12):
            base_pad = max(abs(ymax) * 0.10, 0.25)
            ymin -= base_pad
            ymax += base_pad
        else:
            base_pad = max((ymax - ymin) * 0.12, 0.10)
            ymin -= base_pad
            ymax += base_pad
        return ymin, ymax

    if x_axis_mode == "Manual" and x_axis_manual_start is not None and x_axis_manual_end is not None and x_axis_manual_start < x_axis_manual_end:
        x_min_final = x_axis_manual_start
        x_max_final = x_axis_manual_end
    else:
        x_min_final = global_x_min
        x_max_final = global_x_max

    if use_secondary_axis:
        y1_min_final, y1_max_final = _pad_axis(global_y_min, global_y_max)
        y2_min_final, y2_max_final = _pad_axis(operational_y_min, operational_y_max)

        if y_axis_mode == "Manual" and y_axis_manual_min is not None and y_axis_manual_max is not None:
            y1_min_final = float(y_axis_manual_min)
            y1_max_final = float(y_axis_manual_max)
            if y1_min_final >= y1_max_final:
                y1_min_final, y1_max_final = min(y1_min_final, y1_max_final), max(y1_min_final, y1_max_final) + 1.0

        if operational_y_axis_mode == "Manual" and operational_y_manual_min is not None and operational_y_manual_max is not None:
            y2_min_final = float(operational_y_manual_min)
            y2_max_final = float(operational_y_manual_max)
            if y2_min_final >= y2_max_final:
                y2_min_final, y2_max_final = min(y2_min_final, y2_max_final), max(y2_min_final, y2_max_final) + 1.0
    else:
        base_ymin = operational_y_min if operational_only_mode else global_y_min
        base_ymax = operational_y_max if operational_only_mode else global_y_max
        y_min_final, y_max_final = _pad_axis(base_ymin, base_ymax)
        if warning_enabled and warning_value is not None and math.isfinite(float(warning_value)) and not operational_only_mode:
            y_max_final = max(y_max_final, float(warning_value) * 1.08)
        if danger_enabled and danger_value is not None and math.isfinite(float(danger_value)) and not operational_only_mode:
            y_max_final = max(y_max_final, float(danger_value) * 1.08)
        if y_axis_mode == "Manual" and y_axis_manual_min is not None and y_axis_manual_max is not None:
            y_min_final = float(y_axis_manual_min)
            y_max_final = float(y_axis_manual_max)
            if y_min_final >= y_max_final:
                y_min_final, y_max_final = min(y_min_final, y_max_final), max(y_min_final, y_max_final) + 1.0

    # Ciclo 17.5 P3 — bandas de zona de severidad (alarma / acción).
    # Se dibujan ANTES de los hlines para que las líneas queden por
    # encima del fill. Operan únicamente en single-axis mode (en
    # mixed/secondary axis el eje secundario complica el yref).
    if not operational_only_mode and not use_secondary_axis:
        try:
            _y_top_band = float(y_max_final)
        except Exception:
            _y_top_band = None
        if _y_top_band is not None and math.isfinite(_y_top_band):
            if warning_enabled and warning_value is not None and math.isfinite(float(warning_value)):
                _w = float(warning_value)
                if danger_enabled and danger_value is not None and math.isfinite(float(danger_value)):
                    _d_for_top = float(danger_value)
                else:
                    _d_for_top = _y_top_band
                _band_top = min(_d_for_top, _y_top_band)
                if _band_top > _w:
                    fig.add_shape(
                        type="rect", xref="paper", yref="y",
                        x0=0.0, x1=1.0, y0=_w, y1=_band_top,
                        line=dict(width=0),
                        fillcolor="rgba(245, 158, 11, 0.08)",
                        layer="below",
                    )
            if danger_enabled and danger_value is not None and math.isfinite(float(danger_value)):
                _d = float(danger_value)
                if _y_top_band > _d:
                    fig.add_shape(
                        type="rect", xref="paper", yref="y",
                        x0=0.0, x1=1.0, y0=_d, y1=_y_top_band,
                        line=dict(width=0),
                        fillcolor="rgba(239, 68, 68, 0.10)",
                        layer="below",
                    )

    if warning_enabled and warning_value is not None and math.isfinite(float(warning_value)) and not operational_only_mode:
        fig.add_hline(
            y=float(warning_value), line_width=1.8, line_dash="dash", line_color="#f59e0b",
            annotation_text=f"Warning {format_number(warning_value, 3)}",
            annotation_position="top left", annotation_font_color="#92400e",
        )

    if danger_enabled and danger_value is not None and math.isfinite(float(danger_value)) and not operational_only_mode:
        fig.add_hline(
            y=float(danger_value), line_width=1.9, line_dash="dash", line_color="#ef4444",
            annotation_text=f"Danger {format_number(danger_value, 3)}",
            annotation_position="top left", annotation_font_color="#991b1b",
        )

    cursor_line_specs = {"A Initial": "#334155", "A Current": "#64748b", "B Initial": "#111827", "B Current": "#475569"}
    for label, ts in cursor_map.items():
        if ts is not None:
            fig.add_vline(x=ts, line_width=1.8, line_dash="dot", line_color=cursor_line_specs.get(label, "#475569"))

    if visible_records:
        machine_name = visible_records[0][0].machine
        signal_names_text = " | ".join([r.point_clean for r, _, _ in visible_records[:2]])
        if len(visible_records) > 2:
            signal_names_text += f" +{len(visible_records) - 2}"
    else:
        machine_name = visible_operational_records[0][0].machine
        signal_names_text = " | ".join([r.variable for r, _ in visible_operational_records[:2]])
        if len(visible_operational_records) > 2:
            signal_names_text += f" +{len(visible_operational_records) - 2}"

    if mixed_mode and visible_operational_records:
        signal_names_text = f"{signal_names_text} + Operational"

    latest_values: List[str] = []
    if visible_records:
        for rec, df, unit in visible_records[:2]:
            if not df.empty:
                latest_values.append(f"{rec.point_clean}: {format_number(df['y'].iloc[-1], 3)} {unit}".strip())
    elif visible_operational_records:
        for rec, df in visible_operational_records[:2]:
            if not df.empty:
                latest_values.append(f"{rec.variable}: {format_number(df['y'].iloc[-1], 3)} {rec.unit}".strip())
    latest_text = " | ".join(latest_values) if latest_values else "—"

    if operational_only_mode:
        unit_for_axis = visible_operational_records[0][0].unit if visible_operational_records else ""
        axis_title = f"Operational Data ({unit_for_axis})" if unit_for_axis else "Operational Data"
    else:
        unit_for_axis = visible_records[0][2] if visible_records else ""
        axis_title = f"{metric_key} ({unit_for_axis})" if unit_for_axis else metric_key

    operational_axis_title = ""
    if visible_operational_records:
        families = sorted(set(rec.family for rec, _ in visible_operational_records))
        units = sorted(set(rec.unit for rec, _ in visible_operational_records if rec.unit))
        if "power" in families and len(families) == 1:
            operational_axis_title = "Load / Power (MW)"
        elif "temperature" in families and len(families) == 1:
            operational_axis_title = f"Temperature ({units[0]})" if units else "Temperature"
        else:
            operational_axis_title = "Operational Data"

    time_range_text = "—"
    if global_x_min is not None and global_x_max is not None:
        time_range_text = f"{global_x_min.strftime('%Y-%m-%d %H:%M')} → {global_x_max.strftime('%Y-%m-%d %H:%M')}"

    metric_header_name = "Operational Data" if operational_only_mode else metric_key
    if mixed_mode and operational_axis_title:
        metric_header_name = f"{metric_key} + {operational_axis_title}"

    _draw_top_strip(fig, machine_name, signal_names_text, metric_header_name, latest_text, logo_uri, time_range_text)

    # Ciclo 17.5 P3 — health chip + slope/forecast (sólo cuando
    # estamos analizando vibración con umbrales).
    if not operational_only_mode and visible_records:
        try:
            _records_for_health = [r for r, _, _ in visible_records]
            _health = _compute_trend_health(
                _records_for_health,
                metric_key,
                warning_value=float(warning_value) if (warning_enabled and warning_value is not None) else None,
                danger_value=float(danger_value) if (danger_enabled and danger_value is not None) else None,
            )
            _draw_health_chip(fig, _health, show_below_strip=True)
        except Exception:
            pass

    if show_right_info_box:
        rows: List[Tuple[str, str]] = []
        if visible_records:
            first_rec = visible_records[0][0]
            second_rec = visible_records[1][0] if len(visible_records) >= 2 else first_rec

            a_initial_info = get_cursor_nearest_info(first_rec, metric_key, cursor_map.get("A Initial"))
            a_current_info = get_cursor_nearest_info(first_rec, metric_key, cursor_map.get("A Current"))
            b_initial_info = get_cursor_nearest_info(second_rec, metric_key, cursor_map.get("B Initial"))
            b_current_info = get_cursor_nearest_info(second_rec, metric_key, cursor_map.get("B Current"))

            rows.append((f"A Initial {first_rec.point_clean}", f"{format_number(a_initial_info[0], 3)} {a_initial_info[2]} @ {pretty_time(a_initial_info[1])}".strip() if a_initial_info else "—"))
            rows.append(("A Initial Date", pretty_date(a_initial_info[1]) if a_initial_info else "—"))
            rows.append((f"A Current {first_rec.point_clean}", f"{format_number(a_current_info[0], 3)} {a_current_info[2]} @ {pretty_time(a_current_info[1])}".strip() if a_current_info else "—"))
            rows.append(("A Current Date", pretty_date(a_current_info[1]) if a_current_info else "—"))
            rows.append((f"B Initial {second_rec.point_clean}", f"{format_number(b_initial_info[0], 3)} {b_initial_info[2]} @ {pretty_time(b_initial_info[1])}".strip() if b_initial_info else "—"))
            rows.append(("B Initial Date", pretty_date(b_initial_info[1]) if b_initial_info else "—"))
            rows.append((f"B Current {second_rec.point_clean}", f"{format_number(b_current_info[0], 3)} {b_current_info[2]} @ {pretty_time(b_current_info[1])}".strip() if b_current_info else "—"))
            rows.append(("B Current Date", pretty_date(b_current_info[1]) if b_current_info else "—"))

            a_change = safe_percent_change(a_initial_info[0] if a_initial_info else None, a_current_info[0] if a_current_info else None)
            b_change = safe_percent_change(b_initial_info[0] if b_initial_info else None, b_current_info[0] if b_current_info else None)
            rows.append(("A Change", f"{format_number(a_change, 2)}%" if a_change is not None else "—"))
            rows.append(("B Change", f"{format_number(b_change, 2)}%" if b_change is not None else "—"))

        if mixed_mode and visible_operational_records:
            op_rec = visible_operational_records[0][0]
            op_a_initial = get_operational_cursor_nearest_info(op_rec, cursor_map.get("A Initial"))
            op_a_current = get_operational_cursor_nearest_info(op_rec, cursor_map.get("A Current"))
            op_change = safe_percent_change(op_a_initial[0] if op_a_initial else None, op_a_current[0] if op_a_current else None)
            rows.append((f"Op Initial {trim_text(op_rec.variable, 18)}", f"{format_number(op_a_initial[0], 3)} {op_rec.unit} @ {pretty_time(op_a_initial[1])}".strip() if op_a_initial else "—"))
            rows.append((f"Op Current {trim_text(op_rec.variable, 18)}", f"{format_number(op_a_current[0], 3)} {op_rec.unit} @ {pretty_time(op_a_current[1])}".strip() if op_a_current else "—"))
            rows.append(("Op Change", f"{format_number(op_change, 2)}%" if op_change is not None else "—"))

        if operational_only_mode and visible_operational_records:
            first_op = visible_operational_records[0][0]
            second_op = visible_operational_records[1][0] if len(visible_operational_records) >= 2 else first_op
            a_initial_info = get_operational_cursor_nearest_info(first_op, cursor_map.get("A Initial"))
            a_current_info = get_operational_cursor_nearest_info(first_op, cursor_map.get("A Current"))
            b_initial_info = get_operational_cursor_nearest_info(second_op, cursor_map.get("B Initial"))
            b_current_info = get_operational_cursor_nearest_info(second_op, cursor_map.get("B Current"))
            rows.extend([
                (f"A Initial {trim_text(first_op.variable, 18)}", f"{format_number(a_initial_info[0], 3)} {first_op.unit} @ {pretty_time(a_initial_info[1])}".strip() if a_initial_info else "—"),
                ("A Initial Date", pretty_date(a_initial_info[1]) if a_initial_info else "—"),
                (f"A Current {trim_text(first_op.variable, 18)}", f"{format_number(a_current_info[0], 3)} {first_op.unit} @ {pretty_time(a_current_info[1])}".strip() if a_current_info else "—"),
                ("A Current Date", pretty_date(a_current_info[1]) if a_current_info else "—"),
                (f"B Initial {trim_text(second_op.variable, 18)}", f"{format_number(b_initial_info[0], 3)} {second_op.unit} @ {pretty_time(b_initial_info[1])}".strip() if b_initial_info else "—"),
                ("B Initial Date", pretty_date(b_initial_info[1]) if b_initial_info else "—"),
                (f"B Current {trim_text(second_op.variable, 18)}", f"{format_number(b_current_info[0], 3)} {second_op.unit} @ {pretty_time(b_current_info[1])}".strip() if b_current_info else "—"),
                ("B Current Date", pretty_date(b_current_info[1]) if b_current_info else "—"),
            ])

        if rows:
            _draw_right_info_box(fig, rows)

    if use_secondary_axis:
        fig.update_layout(
            height=640,
            margin=dict(l=46, r=18, t=120, b=40),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#f3f4f6",
            font=dict(color="#111827"),
            hovermode="closest",
            dragmode="pan",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0.0,
                bgcolor="rgba(255,255,255,0.70)", bordercolor="#d1d5db", borderwidth=1,
                font=dict(size=11.2),
            ),
        )
        fig.update_xaxes(
            type="date",  # Ciclo 17.8.1 — forzar date axis (en mixed mode
                          # con secondary_y, Plotly a veces se va a 'linear'
                          # y muestra timestamps en nanosegundos)
            title="Time",
            range=[x_min_final, x_max_final] if x_min_final is not None and x_max_final is not None else None,
            showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False,
            showline=True, linecolor="#9ca3af", ticks="outside", tickcolor="#6b7280", ticklen=4,
            showspikes=True, spikecolor="#6b7280", spikesnap="cursor", spikemode="across",
            tickformat="%Y-%m-%d %H:%M",
            hoverformat="%Y-%m-%d %H:%M:%S",
        )
        fig.update_yaxes(
            title_text=axis_title,
            range=[y1_min_final, y1_max_final],
            showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False,
            showline=True, linecolor="#9ca3af", ticks="outside", tickcolor="#6b7280", ticklen=4,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=operational_axis_title or "Operational Data",
            range=[y2_min_final, y2_max_final],
            showgrid=False, zeroline=False,
            showline=True, linecolor="#9ca3af", ticks="outside", tickcolor="#6b7280", ticklen=4,
            secondary_y=True,
        )
    else:
        fig.update_layout(
            height=640,
            margin=dict(l=46, r=18, t=120, b=40),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#f3f4f6",
            font=dict(color="#111827"),
            hovermode="closest",
            dragmode="pan",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0.0,
                bgcolor="rgba(255,255,255,0.70)", bordercolor="#d1d5db", borderwidth=1,
                font=dict(size=11.2),
            ),
            xaxis=dict(
                type="date",  # Ciclo 17.8.1 — forzar date axis para que
                              # NUNCA muestre timestamps en nanosegundos
                title="Time",
                range=[x_min_final, x_max_final] if x_min_final is not None and x_max_final is not None else None,
                showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False,
                showline=True, linecolor="#9ca3af", ticks="outside", tickcolor="#6b7280", ticklen=4,
                showspikes=True, spikecolor="#6b7280", spikesnap="cursor", spikemode="across",
                tickformat="%Y-%m-%d %H:%M",
                hoverformat="%Y-%m-%d %H:%M:%S",
            ),
            yaxis=dict(
                title=axis_title,
                range=[y_min_final, y_max_final],
                showgrid=True, gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False,
                showline=True, linecolor="#9ca3af", ticks="outside", tickcolor="#6b7280", ticklen=4,
            ),
        )

    return fig



def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    """Convierte scattergl→scatter preservando la ESTRUCTURA
    completa de la figura.

    Ciclo 17.8.2 — bug crítico: antes esta función creaba un
    `go.Figure()` plano y copiaba traces. Pero cuando el chart
    original era `make_subplots(secondary_y=True)`, las traces
    tenían `yaxis="y2"` que apunta a un subplot inexistente en
    la figura plana → las curvas se dibujaban fuera de pantalla
    en kaleido (PNG export). El reporte HD salía con axes
    correctos pero sin traces visibles.

    Fix: clonar el dict entero y solo cambiar el `type` de las
    traces. La estructura (subplots, secondary_y, axis refs) se
    preserva intacta y kaleido puede renderizarla bien.
    """
    fig_dict = fig.to_dict()
    for trace in fig_dict.get("data", []):
        if trace.get("type") == "scattergl":
            trace["type"] = "scatter"
    return go.Figure(fig_dict)

def _scale_export_figure(export_fig: go.Figure) -> go.Figure:
    """Escala fonts/lines/markers para HD export.

    Ciclo 17.8.3 — bug crítico: antes esta función hacía
        fig = go.Figure(data=new_data, layout=fig.layout)
    al final del loop de scaling, lo que recreaba la figura
    desde cero y PERDÍA la estructura de subplots (yaxis="y2"
    apuntando a un axis inexistente). Resultado: PNG HD con
    chart vacío de traces aunque axes y leyenda salieran bien.

    Fix: trabajar SIEMPRE sobre fig_dict (clone) y modificar
    los campos en el dict directamente, sin recrear go.Figure.
    Al final, un solo go.Figure(fig_dict) reconstruye la
    estructura entera (subplots + secondary_y + axis refs)
    intacta.
    """

    def _scale_size(value: Any, factor: float, floor_val: float) -> Any:
        """Escala size/width que puede ser float, int, list (por punto)
        o tupla. Antes el código hacía float(value) directo y reventaba
        cuando los marcadores de anomalía tenían size por punto (lista)."""
        try:
            if isinstance(value, (list, tuple)):
                return [
                    max(floor_val, float(v) * factor) if v is not None else floor_val
                    for v in value
                ]
            if value is None:
                return max(floor_val, 6.0 * factor)
            return max(floor_val, float(value) * factor)
        except Exception:
            return floor_val

    # ---------- 1) Scaling de traces SOBRE EL DICT ----------
    fig_dict = export_fig.to_dict()
    for trace in fig_dict.get("data", []):
        if trace.get("type") in ("scatter", "scattergl"):
            mode = trace.get("mode", "") or ""
            if "lines" in mode:
                line = dict(trace.get("line", {}) or {})
                line["width"] = _scale_size(line.get("width", 1.0), 2.8, 4.8)
                trace["line"] = line
            if "markers" in mode:
                marker = dict(trace.get("marker", {}) or {})
                marker["size"] = _scale_size(marker.get("size", 6), 1.9, 14.0)
                if marker.get("line"):
                    mline = dict(marker["line"])
                    mline["width"] = _scale_size(mline.get("width", 1.0), 1.9, 1.4)
                    marker["line"] = mline
                trace["marker"] = marker
    # Reconstruir la figura PRESERVANDO subplots/secondary_y
    fig = go.Figure(fig_dict)

    has_secondary_y = getattr(fig.layout, "yaxis2", None) is not None
    has_right_info_box = any(
        getattr(ann, "xref", None) == "paper" and float(getattr(ann, "x", 0) or 0) >= 0.83
        for ann in fig.layout.annotations
    )

    export_width = 4200
    export_height = 2200
    margin_left = 120
    margin_right = 90
    margin_top = 360
    margin_bottom = 120

    if has_secondary_y:
        export_width = 4700
        margin_right = 260

    if has_right_info_box:
        export_width = max(export_width, 4900)
        margin_right = max(margin_right, 320)

    fig.update_layout(
        width=export_width,
        height=export_height,
        margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bottom),
        paper_bgcolor="#f3f4f6",
        plot_bgcolor="#f8fafc",
        font=dict(size=30, color="#111827"),
    )

    fig.update_xaxes(
        type="date",  # Ciclo 17.8.1 — date axis también en HD export
        title_font=dict(size=40),
        tickfont=dict(size=26),
        tickformat="%Y-%m-%d %H:%M",
    )
    fig.update_yaxes(title_font=dict(size=40), tickfont=dict(size=26))

    # Ciclo 17.5.4 — coordenadas explícitas del segundo eje en
    # función de si además hay info box a la derecha. Antes el
    # secondary axis quedaba en position=0.80 incluso sin info
    # box, lo que dejaba la curva operacional fuera del eje
    # visible. Ahora:
    #   - has_secondary_y SIN info box → xaxis [0, 0.93], yaxis2 @ 0.94
    #   - has_secondary_y CON info box → xaxis [0, 0.72], yaxis2 @ 0.73
    #   - solo info box (sin secondary) → xaxis [0, 0.72] como antes
    if has_secondary_y and has_right_info_box:
        _xaxis_end = 0.72
        _yaxis2_pos = 0.735
    elif has_secondary_y:
        _xaxis_end = 0.935
        _yaxis2_pos = 0.945
    elif has_right_info_box:
        _xaxis_end = 0.72
        _yaxis2_pos = None
    else:
        _xaxis_end = None
        _yaxis2_pos = None

    if _xaxis_end is not None:
        xaxis_cfg = dict(fig.layout.xaxis.to_plotly_json()) if getattr(fig.layout, "xaxis", None) is not None else {}
        xaxis_cfg["domain"] = [0.0, float(_xaxis_end)]
        fig.update_layout(xaxis=xaxis_cfg)

    if has_secondary_y and _yaxis2_pos is not None:
        yaxis2_cfg = dict(fig.layout.yaxis2.to_plotly_json()) if getattr(fig.layout, "yaxis2", None) is not None else {}
        yaxis2_cfg.update(
            dict(
                automargin=False,
                side="right",
                overlaying="y",
                anchor="free",
                position=float(_yaxis2_pos),
                ticks="outside",
                tickfont=dict(size=26, color="#111827"),
                title_font=dict(size=40, color="#111827"),
                showline=True,
                linecolor="#9ca3af",
                tickcolor="#6b7280",
                ticklen=6,
                zeroline=False,
                showgrid=False,
            )
        )
        fig.update_layout(yaxis2=yaxis2_cfg)

    for shape in fig.layout.shapes:
        if shape.line is not None:
            width = getattr(shape.line, "width", 1) or 1
            shape.line.width = max(2.0, width * 2.2)

    for ann in fig.layout.annotations:
        if ann.font is not None:
            ann.font.size = max(22, int((ann.font.size or 12) * 2.05))

    for img in fig.layout.images:
        sx = getattr(img, "sizex", None)
        sy = getattr(img, "sizey", None)
        if sx is not None:
            img.sizex = sx * 1.22
        if sy is not None:
            img.sizey = sy * 1.22

    return fig


def build_export_png_bytes(fig: go.Figure) -> Tuple[Optional[bytes], Optional[str]]:
    """Export HD — Ciclo 17.8.5 (último approach: ISO strings).

    Sospecho que el problema histórico era que kaleido (motor de
    PNG render) no interpretaba bien los `datetime64[ns]` que
    venían en los traces vía to_dict(). Los serializaba como
    int64 nanosegundos crudos, pero el x-axis tenía range en
    formato datetime — los puntos quedaban fuera del rango
    visible, traces no se dibujaban.

    Fix: convertir EXPLÍCITAMENTE las x de cada trace a ISO
    strings ANTES de pasar a kaleido. ISO strings son universal-
    mente entendibles por cualquier renderer. Si esto no
    funciona, devolvemos un mensaje de error detallado en lugar
    de None silencioso para que el usuario pueda diagnosticar.
    """
    try:
        # 1. Clone via to_dict (preserva subplots+secondary_y)
        fig_dict = fig.to_dict()

        # 2. SERIALIZACIÓN DEFENSIVA — convertir x de cada trace
        #    a ISO strings cuando sean datetime. Ídem y si fuera
        #    necesario (no aplica para Trend pero defensivo).
        n_traces_with_data = 0
        for trace in fig_dict.get("data", []):
            x = trace.get("x")
            if x is not None and len(x) > 0:
                try:
                    x_ser = pd.Series(x)
                    if pd.api.types.is_datetime64_any_dtype(x_ser):
                        # datetime64[ns] → ISO string
                        trace["x"] = x_ser.dt.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ).tolist()
                    elif len(x) > 0 and hasattr(x[0], "isoformat"):
                        # pd.Timestamp / datetime objects
                        trace["x"] = [
                            t.isoformat(sep=" ", timespec="seconds")
                            if hasattr(t, "isoformat") else str(t)
                            for t in x
                        ]
                except Exception:
                    pass  # dejar como esté

                if trace.get("y") is not None and len(trace.get("y", [])) > 0:
                    n_traces_with_data += 1

            # scattergl → scatter (kaleido no soporta WebGL)
            if trace.get("type") == "scattergl":
                trace["type"] = "scatter"

        # 3. Si NO hay traces con data, devolver error explícito
        if n_traces_with_data == 0:
            return None, (
                "Figura sin traces con datos. ¿Hay señales seleccionadas? "
                "(Verificá Signal Selection y Operational Selection en sidebar)"
            )

        # 4. Construir figura HD
        export_fig = go.Figure(fig_dict)
        has_secondary = "yaxis2" in fig_dict.get("layout", {})
        has_right_panel = any(
            (ann.get("xref") == "paper" and float(ann.get("x", 0) or 0) >= 0.83)
            for ann in fig_dict.get("layout", {}).get("annotations", [])
        )
        export_w = 4900 if has_right_panel else (4700 if has_secondary else 4200)
        export_h = 2200

        # IMPORTANTE: limpiar xaxis range del original (estaba en
        # datetime objects). Que kaleido auto-infiera del data ISO.
        export_fig.update_layout(
            width=export_w,
            height=export_h,
            margin=dict(
                l=120,
                r=320 if has_right_panel else (260 if has_secondary else 90),
                t=200,
                b=120,
            ),
            font=dict(size=24, color="#111827"),
            paper_bgcolor="#f3f4f6",
            plot_bgcolor="#f8fafc",
        )
        export_fig.update_xaxes(
            type="date",
            tickfont=dict(size=22),
            title_font=dict(size=30),
            tickformat="%Y-%m-%d %H:%M",
            autorange=True,  # 17.8.5: forzar autorange con la nueva data ISO
        )
        export_fig.update_yaxes(
            tickfont=dict(size=22),
            title_font=dict(size=30),
        )

        # 5. Render PNG via kaleido — con engine explícito
        png_bytes = export_fig.to_image(
            format="png",
            width=export_w,
            height=export_h,
            scale=2,
            engine="kaleido",
        )

        # 6. Validar que no devuelva bytes vacíos
        if not png_bytes or len(png_bytes) < 1000:
            return None, (
                f"Kaleido devolvió PNG inválido ({len(png_bytes) if png_bytes else 0} bytes). "
                f"Probable bug de versión kaleido — reportá al equipo."
            )

        return png_bytes, None
    except Exception as e:
        return None, f"Error generando PNG HD: {type(e).__name__}: {e}"


def _sanitize_series_for_analysis(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr[np.isfinite(arr)]


def _classify_trend_behavior(values: pd.Series) -> Dict[str, Any]:
    arr = _sanitize_series_for_analysis(values)
    result: Dict[str, Any] = {
        "classification": "insufficient",
        "slope_ratio": None,
        "change_pct": None,
        "volatility_ratio": None,
        "jerk_ratio": None,
        "sample_count": int(arr.size),
    }
    if arr.size < 3:
        return result

    x = np.arange(arr.size, dtype=float)
    slope, intercept = np.polyfit(x, arr, 1)
    fitted = slope * x + intercept
    residual = arr - fitted

    mean_abs = float(np.mean(np.abs(arr)))
    value_span = float(np.max(arr) - np.min(arr))
    scale = max(mean_abs, value_span, 1e-9)

    slope_ratio = float(abs(slope) * max(arr.size - 1, 1) / scale)
    volatility_ratio = float(np.std(residual) / scale)
    diffs = np.diff(arr)
    jerk_ratio = float(np.std(diffs) / scale) if diffs.size else 0.0
    change_pct = safe_percent_change(float(arr[0]), float(arr[-1]))

    direction = "up" if slope > 0 else "down"
    classification = "stable"
    if jerk_ratio >= 0.28 or volatility_ratio >= 0.22:
        classification = "abrupt"
    elif slope_ratio >= 0.18 and direction == "up":
        classification = "progressive_increase"
    elif slope_ratio >= 0.18 and direction == "down":
        classification = "progressive_decrease"

    result.update(
        {
            "classification": classification,
            "direction": direction,
            "slope_ratio": slope_ratio,
            "change_pct": change_pct,
            "volatility_ratio": volatility_ratio,
            "jerk_ratio": jerk_ratio,
            "initial_value": float(arr[0]),
            "final_value": float(arr[-1]),
            "min_value": float(np.min(arr)),
            "max_value": float(np.max(arr)),
            "mean_value": float(np.mean(arr)),
        }
    )
    return result


def _trend_unit_for_metric(record: TrendRecord, metric_key: str) -> str:
    return get_metric_series(record, metric_key)[1]


def _build_single_trend_narrative(record: TrendRecord, metric_key: str) -> str:
    df = get_clean_metric_df(record, metric_key)
    unit = _trend_unit_for_metric(record, metric_key)
    if df.empty:
        return (
            f"{record.point_clean}: no se identificaron datos válidos para el análisis de {metric_key.lower()}, "
            "por lo que no fue posible emitir diagnóstico automático."
        )

    analysis = _classify_trend_behavior(df["y"])
    sample_count = analysis.get("sample_count", 0)
    start_ts = safe_datetime(df["x"].iloc[0])
    end_ts = safe_datetime(df["x"].iloc[-1])

    base = (
        f"{record.point_clean} — ventana analizada desde {pretty_date(start_ts)} {pretty_time(start_ts)} "
        f"hasta {pretty_date(end_ts)} {pretty_time(end_ts)}, con {sample_count} muestras válidas. "
        f"Valor inicial {format_number(analysis.get('initial_value'), 3)} {unit}, "
        f"valor final {format_number(analysis.get('final_value'), 3)} {unit}, "
        f"variación total {format_number(analysis.get('change_pct'), 2)}%."
    )

    classification = analysis.get("classification")
    if classification == "progressive_increase":
        return (
            f"{base} La tendencia presenta un incremento progresivo del {metric_key.lower()}, "
            "lo cual sugiere posible deterioro del estado mecánico o evolución de una condición incipiente. "
            "Se recomienda seguimiento estrecho y correlación con variables operativas y alarmas."
        )
    if classification == "progressive_decrease":
        return (
            f"{base} La señal muestra una disminución progresiva del {metric_key.lower()}, "
            "compatible con normalización de la condición o reducción de carga/excitación. "
            "Se recomienda verificar si el comportamiento coincide con cambios operativos esperados."
        )
    if classification == "abrupt":
        return (
            f"{base} Se observan variaciones bruscas y dispersión elevada en la señal, "
            "compatibles con condición transitoria, inestabilidad o cambios operativos repentinos. "
            "Se recomienda revisar eventos de proceso, transientes de arranque/parada y consistencia de la instrumentación."
        )
    if classification == "stable":
        return (
            f"{base} El comportamiento es estable y sin desviaciones significativas, "
            "lo que es consistente con una condición normal dentro de la ventana evaluada. "
            "Se recomienda continuar monitoreo rutinario."
        )
    return (
        f"{base} La cantidad de información disponible no es suficiente para clasificar con confianza la tendencia. "
        "Se recomienda ampliar la ventana temporal o validar la calidad de los datos."
    )


def _build_operational_only_narrative(records: List[OperationalRecord]) -> str:
    lines: List[str] = []
    for rec in records:
        df = get_operational_clean_df(rec)
        if df.empty:
            lines.append(
                f"{rec.variable}: no se identificaron datos válidos para emitir diagnóstico automático."
            )
            continue
        analysis = _classify_trend_behavior(df["y"])
        start_ts = safe_datetime(df["x"].iloc[0])
        end_ts = safe_datetime(df["x"].iloc[-1])
        unit = rec.unit or ""
        base = (
            f"{rec.variable} — ventana analizada desde {pretty_date(start_ts)} {pretty_time(start_ts)} "
            f"hasta {pretty_date(end_ts)} {pretty_time(end_ts)}. "
            f"Valor inicial {format_number(analysis.get('initial_value'), 3)} {unit}, "
            f"valor final {format_number(analysis.get('final_value'), 3)} {unit}, "
            f"variación total {format_number(analysis.get('change_pct'), 2)}%."
        )

        classification = analysis.get("classification")
        if classification == "progressive_increase":
            lines.append(f"{base} Tendencia operativa con incremento progresivo sostenido.")
        elif classification == "progressive_decrease":
            lines.append(f"{base} Tendencia operativa con descenso progresivo sostenido.")
        elif classification == "abrupt":
            lines.append(f"{base} Tendencia operativa con variaciones bruscas o comportamiento transitorio.")
        elif classification == "stable":
            lines.append(f"{base} Tendencia operativa estable durante la ventana evaluada.")
        else:
            lines.append(f"{base} Información insuficiente para clasificar la tendencia.")
    return "\n\n".join(lines)


def build_trend_report_narrative(
    records: List[TrendRecord],
    metric_key: str,
    operational_records: Optional[List[OperationalRecord]] = None,
    operational_only_mode: bool = False,
) -> str:
    operational_records = operational_records or []

    if operational_only_mode and operational_records:
        return _build_operational_only_narrative(operational_records)

    trend_lines = [_build_single_trend_narrative(rec, metric_key) for rec in records]
    if operational_records:
        op_summary = _build_operational_only_narrative(operational_records)
        trend_lines.append(
            "Correlación operativa disponible:\n\n"
            f"{op_summary}"
        )
    
context = st.session_state.get("asset_context", {})
ctx_text = f"\n\nContexto de máquina: {context.get('type','')} - {context.get('description','')}"




# session
if "trend_signals" not in st.session_state:
    st.session_state["trend_signals"] = {}
if "operational_signals" not in st.session_state:
    st.session_state["operational_signals"] = {}
if "wm_tr_operational_signal_ids" not in st.session_state:
    st.session_state.wm_tr_operational_signal_ids = []
if "wm_tr_operational_temp_unit" not in st.session_state:
    st.session_state.wm_tr_operational_temp_unit = "°F"
if "wm_tr_primary_signal_id" not in st.session_state:
    st.session_state.wm_tr_primary_signal_id = None
if "wm_tr_extra_signal_ids" not in st.session_state:
    st.session_state.wm_tr_extra_signal_ids = []
if "wm_tr_display_mode" not in st.session_state:
    st.session_state.wm_tr_display_mode = "Combined"
if "wm_tr_export_store" not in st.session_state:
    st.session_state.wm_tr_export_store = {}
if "report_items" not in st.session_state:
    st.session_state.report_items = []

if "wm_tr_asset_type" not in st.session_state:
    st.session_state.wm_tr_asset_type = ""
if "wm_tr_machine_configuration" not in st.session_state:
    st.session_state.wm_tr_machine_configuration = ""
if "wm_tr_primary_equipment" not in st.session_state:
    st.session_state.wm_tr_primary_equipment = ""
if "wm_tr_secondary_equipment" not in st.session_state:
    st.session_state.wm_tr_secondary_equipment = ""
if "wm_tr_machine_description" not in st.session_state:
    st.session_state.wm_tr_machine_description = ""

for key in [
    "wm_tr_cursor_a_initial", "wm_tr_cursor_a_current",
    "wm_tr_cursor_b_initial", "wm_tr_cursor_b_current",
    "wm_tr_x_manual_start", "wm_tr_x_manual_end",
]:
    if key not in st.session_state:
        st.session_state[key] = ""


with st.sidebar:
    # Ciclo 17.5 — instancia activa (necesaria para histórico de
    # tendencias persistente bajo {INSTANCES_DIR}/{instance_id}).
    trend_instance_state = render_instance_selector(module_name="trends")
    trend_active_instance_id = str(trend_instance_state.get("instance_id") or "").strip()

    st.markdown("### Trend CSV")
    uploaded_files = st.file_uploader(
        "Upload one or more trend CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="wm_trend_uploader",
    )
    if uploaded_files:
        parsed_records = load_trend_records_from_uploader(uploaded_files)
        trend_store = {rec.trend_id: rec for rec in parsed_records}
        st.session_state["trend_signals"] = trend_store

    st.markdown("### Operational Data CSV")
    st.session_state.wm_tr_operational_temp_unit = st.selectbox(
        "Temperature unit in operational file",
        options=["°F", "°C"],
        index=0 if st.session_state.wm_tr_operational_temp_unit == "°F" else 1,
    )
    operational_uploaded_files = st.file_uploader(
        "Upload one or more operational CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="wm_operational_uploader",
    )
    if operational_uploaded_files:
        parsed_operational_records = load_operational_records_from_uploader(
            operational_uploaded_files,
            temperature_unit=st.session_state.wm_tr_operational_temp_unit,
        )
        operational_store = {rec.op_id: rec for rec in parsed_operational_records}
        st.session_state["operational_signals"] = operational_store

        # Ciclo 17.8 — Data quality banner: feedback explícito de
        # qué se cargó. Esto convierte el "subí el CSV y no sé si
        # quedó bien" en una validación visible inmediata.
        if parsed_operational_records:
            _n_vars = len(parsed_operational_records)
            _total_pts = sum(r.n_samples for r in parsed_operational_records)
            _families = sorted({r.family for r in parsed_operational_records if r.family != "generic"})
            _ts_min = min((r.timestamp_min for r in parsed_operational_records if r.timestamp_min is not None), default=None)
            _ts_max = max((r.timestamp_max for r in parsed_operational_records if r.timestamp_max is not None), default=None)
            _window_txt = ""
            if _ts_min is not None and _ts_max is not None:
                _delta = _ts_max - _ts_min
                _days = _delta.total_seconds() / 86400.0
                if _days >= 1:
                    _window_txt = f"{_days:.1f} días"
                else:
                    _window_txt = f"{_delta.total_seconds() / 3600.0:.1f} h"
            _fam_chips = " ".join(
                f"<span style='background:#e0f2fe;color:#075985;"
                f"padding:1px 7px;border-radius:999px;font-size:0.72rem;"
                f"font-weight:600;margin-right:3px;'>{f}</span>"
                for f in _families
            )
            st.markdown(
                f"""
                <div style="background:#ecfdf5;border-left:3px solid #10b981;
                            padding:8px 11px;border-radius:6px;margin-top:6px;
                            font-size:0.82rem;color:#065f46;">
                    <b>✓ {_n_vars} variables operativas</b>
                    · {_total_pts:,} muestras · ventana {_window_txt}
                    <br><span style='font-size:0.74rem;color:#047857;'>
                    Familias detectadas:</span> {_fam_chips}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Auto-select de los primeros N (default 3) si no hay
            # selección previa. Antes el chart arrancaba VACÍO y el
            # usuario no veía nada hasta ir a la sidebar y seleccionar.
            if not st.session_state.get("wm_tr_operational_signal_ids"):
                default_ids = [r.op_id for r in parsed_operational_records[:3]]
                st.session_state["wm_tr_operational_signal_ids"] = default_ids

    # =========================================================
    # HISTORICO DE TENDENCIAS (Ciclo 17.5 P2)
    # =========================================================
    # Permite archivar la corrida actual (CSVs vibración +
    # operacional) bajo la instancia activa, y volver a traer
    # corridas anteriores para concatenarlas con la corrida
    # actual y tener un trend largo de meses/años.
    st.markdown("### 📚 Histórico de Tendencias")
    if not trend_active_instance_id:
        st.caption(
            "Seleccione una instancia activa (arriba) para guardar y "
            "recuperar corridas históricas."
        )
        historical_corrida_ids: List[str] = []
    else:
        # ----- Resumen del histórico
        try:
            _hist_summary = list_corridas_summary(trend_active_instance_id)
        except Exception:
            _hist_summary = {"n_corridas": 0, "earliest": "", "latest": "", "total_files": 0}
        _n_corr = int(_hist_summary.get("n_corridas", 0) or 0)
        if _n_corr > 0:
            _earl = str(_hist_summary.get("earliest", "") or "").split("T")[0]
            _late = str(_hist_summary.get("latest", "") or "").split("T")[0]
            _tot_files = int(_hist_summary.get("total_files", 0) or 0)
            _resumen_txt = f"📊 {_n_corr} corrida(s) archivada(s) · {_tot_files} CSV total"
            if _earl and _late:
                _resumen_txt += f" · rango: {_earl} → {_late}"
            st.caption(_resumen_txt)
        else:
            st.caption("Aún no hay corridas archivadas para esta instancia.")

        # ----- Guardar corrida actual
        with st.expander("📸 Archivar corrida actual", expanded=False):
            _all_current_files = list(uploaded_files or []) + list(operational_uploaded_files or [])
            if not _all_current_files:
                st.caption(
                    "Cargue al menos un CSV (trend u operacional) en los uploaders "
                    "de arriba antes de archivar."
                )
            else:
                _label_default = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                _corrida_label_in = st.text_input(
                    "Etiqueta de la corrida (opcional)",
                    value="",
                    placeholder=f"Ejemplo: «Post-mantenimiento abril» — default: {_label_default}",
                    key="wm_trend_hist_label",
                )
                _corrida_notes_in = st.text_area(
                    "Observaciones (opcional)",
                    value="",
                    height=80,
                    placeholder="Condiciones de la corrida, eventos notables, ajustes operacionales…",
                    key="wm_trend_hist_notes",
                )
                if st.button(
                    "Guardar al histórico",
                    key="wm_trend_hist_save_btn",
                    use_container_width=True,
                ):
                    try:
                        _files_bytes = _collect_uploader_bytes(_all_current_files)
                        _ts_min, _ts_max = _detect_time_range_for_uploads(_all_current_files)
                        _new_id = save_trend_corrida(
                            trend_active_instance_id,
                            _files_bytes,
                            corrida_label=_corrida_label_in or _label_default,
                            notes=_corrida_notes_in or "",
                            detected_time_range=(_ts_min, _ts_max),
                        )
                        # Forzar update_corrida_time_range por si el caller
                        # quiere fijar el rango después (ya viene seteado
                        # arriba, pero esto es defensivo).
                        if _ts_min is not None or _ts_max is not None:
                            try:
                                update_corrida_time_range(
                                    trend_active_instance_id, _new_id, _ts_min, _ts_max
                                )
                            except Exception:
                                pass
                        st.success(
                            f"Corrida archivada bajo «{trend_active_instance_id}» "
                            f"({len(_files_bytes)} CSV)."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"No se pudo archivar la corrida: {exc}")

        # ----- Cargar corridas anteriores para merge
        try:
            _avail_corridas = list_trend_corridas(trend_active_instance_id)
        except Exception:
            _avail_corridas = []

        if _avail_corridas:
            def _corrida_label_fmt(meta: Dict[str, Any]) -> str:
                cid = meta.get("corrida_id", "")
                lab = (meta.get("corrida_label") or "").strip()
                ts = (meta.get("timestamp") or "").split("T")[0]
                tr = meta.get("time_range", {}) or {}
                tmin = (tr.get("min") or "").split("T")[0]
                tmax = (tr.get("max") or "").split("T")[0]
                nf = int(meta.get("n_files", 0) or 0)
                bits = [lab or ts]
                if tmin and tmax and tmin != tmax:
                    bits.append(f"{tmin}→{tmax}")
                elif tmin:
                    bits.append(tmin)
                bits.append(f"{nf} CSV")
                return " · ".join([b for b in bits if b])

            _corrida_id_to_label = {
                c["corrida_id"]: _corrida_label_fmt(c) for c in _avail_corridas
            }
            historical_corrida_ids = st.multiselect(
                "Incluir corridas anteriores en el análisis",
                options=list(_corrida_id_to_label.keys()),
                format_func=lambda cid: _corrida_id_to_label.get(cid, cid),
                default=st.session_state.get("wm_trend_hist_selected", []),
                key="wm_trend_hist_selected",
                help=(
                    "Las corridas seleccionadas se concatenan cronológicamente "
                    "con la corrida actual para reconstruir tendencias largas."
                ),
            )

            with st.expander("Administrar corridas archivadas", expanded=False):
                for _meta in _avail_corridas:
                    _cid = _meta.get("corrida_id", "")
                    cols = st.columns([0.78, 0.22])
                    cols[0].caption(f"• {_corrida_label_fmt(_meta)}")
                    if cols[1].button(
                        "🗑",
                        key=f"wm_trend_hist_del_{_cid}",
                        help=f"Borrar corrida {_cid}",
                    ):
                        try:
                            ok = delete_trend_corrida(trend_active_instance_id, _cid)
                            if ok:
                                st.success(f"Corrida {_cid} eliminada.")
                                # Limpiar de la selección si estaba ahí
                                _sel = list(st.session_state.get("wm_trend_hist_selected", []))
                                if _cid in _sel:
                                    _sel.remove(_cid)
                                    st.session_state["wm_trend_hist_selected"] = _sel
                                st.rerun()
                            else:
                                st.warning("No se pudo borrar la corrida.")
                        except Exception as exc:
                            st.error(f"Error borrando: {exc}")
        else:
            historical_corrida_ids = []

    # Ciclo 17.5 — el contexto de la máquina (asset type, configuración,
    # descripción técnica) ahora se hereda automáticamente de la
    # instancia activa de Machinery Library, así no se duplica la
    # entrada de datos. Más abajo se construye `asset_context` desde
    # `trend_instance_state` para alimentar el reporte PDF.

records_all: List[TrendRecord] = list(st.session_state.get("trend_signals", {}).values())
operational_records_all: List[OperationalRecord] = list(st.session_state.get("operational_signals", {}).values())

# =========================================================
# MERGE HISTORICO (Ciclo 17.5 P2)
# =========================================================
# Si el usuario seleccionó corridas anteriores en la sidebar,
# se cargan los CSVs persistidos, se reparsean al vuelo y se
# concatenan con la corrida actual. La concatenación efectiva
# (combinar series temporales de un mismo punto a través de
# corridas) se hace más abajo en build_trend_figure cuando se
# suman registros con el mismo display_name; aquí solo
# enrichecemos la lista de records.
_hist_corrida_ids: List[str] = list(st.session_state.get("wm_trend_hist_selected", []) or [])
_hist_active_instance_id: str = str(st.session_state.get("wm_active_instance_id", "") or "")
if _hist_corrida_ids and _hist_active_instance_id:
    _temp_unit_for_hist = st.session_state.get("wm_tr_operational_temp_unit", "°F")
    _hist_summary_msgs: List[str] = []
    for _cid in _hist_corrida_ids:
        try:
            _files_bytes = load_trend_corrida_files(_hist_active_instance_id, _cid)
        except Exception:
            _files_bytes = []
        if not _files_bytes:
            continue
        _label = _cid
        try:
            _all_meta = list_trend_corridas(_hist_active_instance_id)
            for _m in _all_meta:
                if _m.get("corrida_id") == _cid:
                    _label = (_m.get("corrida_label") or _cid).strip() or _cid
                    break
        except Exception:
            pass
        _hist_trend_recs, _hist_op_recs = _parse_corrida_files(
            _files_bytes,
            temperature_unit=_temp_unit_for_hist,
            corrida_label=_label,
        )
        records_all.extend(_hist_trend_recs)
        operational_records_all.extend(_hist_op_recs)
        _hist_summary_msgs.append(
            f"«{_label}» · {len(_hist_trend_recs)} trend / "
            f"{len(_hist_op_recs)} operacional"
        )
    if _hist_summary_msgs:
        st.info(
            "📚 Corridas históricas incluidas en el análisis:\n\n- "
            + "\n- ".join(_hist_summary_msgs)
        )

records_all = sorted(records_all, key=lambda r: (r.machine, r.point_clean, r.file_name))
operational_records_all = sorted(operational_records_all, key=lambda r: (r.machine, r.variable, r.file_name))


if not records_all and not operational_records_all:
    st.warning("Cargue al menos un CSV de tendencia o un CSV de data operativa en este módulo.")
    st.stop()

# =========================================================
# Ciclo 17.5 — Asset context auto-derivado de la instancia activa
# =========================================================
# Antes el usuario tenía que repetir asset_type, configuración,
# primary/secondary equipment y descripción técnica desde esta
# página. Ahora la instancia activa (Machinery Library) ya
# contiene todos estos campos, así que los heredamos
# automáticamente y los conservamos en session_state para que el
# reporte PDF y la narrativa sigan funcionando sin cambios.
def _build_trend_asset_context_from_instance(state: Dict[str, Any]) -> Dict[str, Any]:
    profile_label = str(state.get("profile_label") or "").strip()
    machine_group = str(state.get("machine_group") or "").strip()
    tag = str(state.get("tag") or "").strip()
    location = str(state.get("location") or "").strip()
    notes = str(state.get("notes") or "").strip()
    instance_label = str(state.get("instance_label") or state.get("instance_id") or "").strip()

    # asset_type: heurística por profile_label / machine_group
    pl_low = profile_label.lower()
    if "turbogen" in pl_low or "tg-" in pl_low:
        asset_type = "Turbogenerador"
    elif "turbina de gas" in pl_low or "lm6000" in pl_low or "frame " in pl_low:
        asset_type = "Turbina de gas"
    elif "vapor" in pl_low and "turbina" in pl_low:
        asset_type = "Turbina de vapor"
    elif "generador" in pl_low or "alternador" in pl_low:
        asset_type = "Generador eléctrico"
    elif "compresor" in pl_low:
        asset_type = "Compresor"
    elif "bomba" in pl_low:
        asset_type = "Bomba"
    elif "ventilador" in pl_low or "fan" in pl_low:
        asset_type = "Ventilador"
    elif "gearbox" in pl_low or "caja" in pl_low:
        asset_type = "Gearbox"
    elif "motor" in pl_low:
        asset_type = "Motor eléctrico"
    elif machine_group:
        asset_type = profile_label or machine_group
    else:
        asset_type = profile_label or "Activo monitoreado"

    machine_configuration = "Simple"
    description_bits: List[str] = []
    if profile_label:
        description_bits.append(profile_label)
    if tag and tag != "(default)":
        description_bits.append(f"tag {tag}")
    if location:
        description_bits.append(f"ubicación {location}")
    if notes:
        description_bits.append(notes)
    machine_description = ". ".join(description_bits) if description_bits else (instance_label or asset_type)

    return {
        "type": asset_type,
        "description": machine_description,
        "asset_type": asset_type,
        "machine_configuration": machine_configuration,
        "primary_equipment": "",
        "secondary_equipment": "",
        "machine_description": machine_description,
    }


_trend_ctx = _build_trend_asset_context_from_instance(trend_instance_state)
# Mantener legacy session keys sincronizadas (varios consumidores
# aguas abajo todavía leen wm_tr_asset_type / wm_tr_machine_*).
st.session_state["wm_tr_asset_type"] = _trend_ctx["asset_type"]
st.session_state["wm_tr_machine_configuration"] = _trend_ctx["machine_configuration"]
st.session_state["wm_tr_primary_equipment"] = _trend_ctx["primary_equipment"]
st.session_state["wm_tr_secondary_equipment"] = _trend_ctx["secondary_equipment"]
st.session_state["wm_tr_machine_description"] = _trend_ctx["machine_description"]
st.session_state["asset_context"] = _trend_ctx


def push_linked_bode_context(records: List[TrendRecord], metric_key: str) -> None:
    if not records:
        return

    first = records[0]

    cursor_a_label = (
        st.session_state.get("wm_tr_cursor_a_current")
        or st.session_state.get("wm_tr_cursor_a_initial")
        or None
    )
    cursor_b_label = (
        st.session_state.get("wm_tr_cursor_b_current")
        or st.session_state.get("wm_tr_cursor_b_initial")
        or None
    )

    st.session_state["linked_bode_context"] = {
        "machine": first.machine,
        "point": first.point_clean,
        "variable": metric_key,
        "source_module": "04_Trends",
        "trend_cursor_a_label": cursor_a_label,
        "trend_cursor_b_label": cursor_b_label,
    }




def final_report_cleanup(text: Any) -> str:
    t = str(text or "")

    # Convertir secuencias literales a saltos reales
    t = t.replace("\\n", "\n")
    t = t.replace("\\t", " ")

    # Normalizar saltos y espacios
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)

    # Ajustes editoriales
    t = t.replace("amplitude", "amplitud vibratoria")
    t = t.replace("Amplitude", "Amplitud vibratoria")
    t = t.replace("del amplitud vibratoria", "de la amplitud vibratoria")
    t = t.replace("del amplitude", "de la amplitud vibratoria")
    t = t.replace("de amplitude", "de la amplitud vibratoria")
    t = t.replace(" phase ", " fase ")
    t = t.replace(" speed ", " velocidad ")

    return t.strip()


def build_behavior_change_report_block(records: List[TrendRecord], metric_key: str) -> str:
    summary = build_behavior_change_summary(records, metric_key)
    details = summary.get("details", []) or []

    valid = [
        d for d in details
        if d.get("valid") and d.get("classification") in ["Strong change", "Moderate change"]
    ]

    if not valid:
        return (
            "Cambio de comportamiento: no se identifican transiciones relevantes de régimen "
            "dentro de la ventana analizada."
        )

    top = sorted(
        valid,
        key=lambda d: float(d.get("change_score") or 0.0),
        reverse=True
    )[0]

    change_ts = top.get("change_timestamp")
    if change_ts is not None:
        ts_txt = f"{pretty_date(change_ts)} {pretty_time(change_ts)}"
    else:
        ts_txt = "sin timestamp identificable"

    return (
        f"Cambio de comportamiento: la clasificación dominante es {summary.get('top_classification', '—')}. "
        f"El cambio más representativo se localiza en la señal {top.get('record_name', '—')} "
        f"alrededor de {ts_txt}. "
        f"{top.get('interpretation', 'Sin interpretación disponible.')}"
    )


def queue_trend_to_report(
    records: List[TrendRecord],
    fig: go.Figure,
    panel_title: str,
    metric_key: str,
    operational_records: Optional[List[OperationalRecord]] = None,
    operational_only_mode: bool = False,
) -> Tuple[bool, Optional[str]]:
    operational_records = operational_records or []

    # Ciclo 17.5 — el asset context ya viene auto-derivado de la
    # instancia activa, así que no hay validaciones manuales que
    # bloqueen el envío al reporte.
    if records:
        first = records[0]
        machine = first.machine
        point = " | ".join([r.point_clean for r in records[:2]])
        if len(records) > 2:
            point += f" +{len(records)-2}"
        signal_id = "|".join([r.trend_id for r in records])
        timestamp = str(records[0].timestamp_max or "")
        variable = f"Trend | {metric_key}"
    elif operational_records:
        first_op = operational_records[0]
        machine = first_op.machine
        point = " | ".join([r.variable for r in operational_records[:2]])
        if len(operational_records) > 2:
            point += f" +{len(operational_records)-2}"
        signal_id = "|".join([r.op_id for r in operational_records])
        timestamp = str(first_op.timestamp_max or "")
        variable = "Operational Data" if operational_only_mode else f"Trend + Operational | {metric_key}"
    else:
        return False, "No valid signals to send."

    narrative = build_trend_report_narrative_core(
        records=records,
        metric_key=metric_key,
        operational_records=operational_records,
        operational_only_mode=operational_only_mode,
        asset_context=st.session_state.get("asset_context", {}) or {},
    )

    # =================================================================
    # Ciclo 17.5.3 — Autodiagnóstico ejecutivo en el PDF
    # =================================================================
    # El autodiagnóstico que ya se muestra en pantalla (headline + 6
    # párrafos Bently + recomendaciones) se inyecta al inicio de la
    # narrativa del reporte. Antes el reporte recibía solo la
    # narrativa core (descripción factual) y los detectores
    # individuales — ahora la síntesis ejecutiva va arriba para que
    # el lector capture el diagnóstico en una página.
    autodiag_block_text = ""
    _autodiag_for_pdf: Dict[str, Any] = {}
    if records and not operational_only_mode:
        try:
            _thr_meta = st.session_state.get("wm_tr_threshold_source", {}) or {}
            _w_for_diag = _thr_meta.get("warning_value")
            _d_for_diag = _thr_meta.get("danger_value")
            _autodiag_for_pdf = build_trend_autodiagnostic(
                records,
                metric_key,
                warning_value=float(_w_for_diag) if _w_for_diag is not None else None,
                danger_value=float(_d_for_diag) if _d_for_diag is not None else None,
                operational_records=operational_records,
            )
            _bits: List[str] = []
            _headline = _autodiag_for_pdf.get("headline", "")
            if _headline:
                _bits.append(f"Diagnóstico ejecutivo: {_headline}")

            for _para in _autodiag_for_pdf.get("prose", []) or []:
                if _para and _para.strip():
                    _bits.append(_para.strip())

            _recs_for_pdf = _autodiag_for_pdf.get("recommendations", []) or []
            if _recs_for_pdf:
                _rec_lines = "\n".join([f"  {_i}. {_r}" for _i, _r in enumerate(_recs_for_pdf, 1)])
                _bits.append(f"Acciones recomendadas:\n{_rec_lines}")

            # Marco de fuente de los setpoints (Vault / ISO / Override)
            _src = (_thr_meta.get("source") or "").strip()
            if _src and _src not in ("default", "n/a"):
                _src_line_parts = [f"Setpoints: {_src}"]
                if _thr_meta.get("warning_is_override") or _thr_meta.get("danger_is_override"):
                    _sw = _thr_meta.get("suggested_warning")
                    _sd = _thr_meta.get("suggested_danger")
                    _src_line_parts.append(
                        "Override del cliente activo "
                        f"(sugeridos: W={_sw} / D={_sd})"
                    )
                _detail = (_thr_meta.get("detail") or "").strip()
                if _detail:
                    _src_line_parts.append(_detail)
                _bits.append(" · ".join(_src_line_parts))

                # Ciclo 17.9 — cita normativa explícita en el PDF
                _norm_ref = (_thr_meta.get("norm_reference") or "").strip()
                if _norm_ref:
                    _bits.append(f"Referencia normativa: {_norm_ref}.")
                _override_just = (_thr_meta.get("override_justification") or "").strip()
                if _override_just and (_thr_meta.get("warning_is_override") or _thr_meta.get("danger_is_override")):
                    _bits.append(
                        f"Justificación del override del especialista: {_override_just}"
                    )

            if _bits:
                autodiag_block_text = "\n\n".join(_bits)
                narrative = f"{autodiag_block_text}\n\n{narrative}"
        except Exception:
            pass

    correlation_report_block = ""
    correlation_payload: Dict[str, Any] = {}
    lag_payload: Dict[str, Any] = {}
    ranking_payload: List[Dict[str, Any]] = []
    ranking_summary = ""

    # Ciclo 17.25 — IMPORTANTE: cuando hay >1 operacional, primero rankeamos
    # TODAS por correlación con la vibración, y elegimos como primary la
    # MÁS CORRELACIONADA (no la primera del CSV). Antes solo se analizaba
    # operational_records[0] y se ignoraba el resto del análisis principal.
    if records and operational_records:
        primary_trend = records[0]

        # Default: si hay solo 1 operacional o el ranking falla, usar la primera
        primary_operational = operational_records[0]

        # Si hay múltiples operacionales, rankear y elegir la TOP por score
        if len(operational_records) > 1:
            ranking_df = build_operational_variable_ranking(
                trend_record=primary_trend,
                operational_records=operational_records,
                metric_key=metric_key,
            )
            if not ranking_df.empty:
                ranking_summary = build_operational_variable_ranking_summary(ranking_df)
                ranking_payload = ranking_df.to_dict(orient="records")
                # El ranking ya viene ordenado por score descendente; el TOP-1
                # es la operacional más correlacionada con la vibración.
                # Buscamos el OperationalRecord correspondiente por nombre.
                top_var_name = str(ranking_df.iloc[0].get("Variable", "")).strip()
                if top_var_name:
                    _matched = next(
                        (r for r in operational_records
                         if str(r.variable).strip() == top_var_name),
                        None,
                    )
                    if _matched is not None:
                        primary_operational = _matched

        correlation_report_block = build_operational_correlation_report_block(
            trend_record=primary_trend,
            operational_record=primary_operational,
            metric_key=metric_key,
        )

        corr_info = build_trend_operational_correlation(
            trend_record=primary_trend,
            operational_record=primary_operational,
            metric_key=metric_key,
        )
        lag_info = build_lagged_correlation_analysis(
            trend_record=primary_trend,
            operational_record=primary_operational,
            metric_key=metric_key,
            max_lag_minutes=180,
            step_minutes=10,
        )

        correlation_payload = {
            "variable_name": primary_operational.variable,
            "corr_value": corr_info.get("corr_value"),
            "strength": corr_info.get("strength"),
            "direction": corr_info.get("direction"),
            "interpretation": corr_info.get("interpretation"),
            "sample_count": corr_info.get("sample_count"),
        }
        lag_payload = {
            "best_corr": lag_info.get("best_corr"),
            "best_lag_min": lag_info.get("best_lag_min"),
            "strength": lag_info.get("strength"),
            "direction": lag_info.get("direction"),
            "interpretation": lag_info.get("interpretation"),
        }

    drift_summary: Dict[str, Any] = {}
    drift_details: List[Dict[str, Any]] = []
    drift_narrative = ""

    if records:
        drift_summary = build_panel_drift_summary(records, metric_key)
        drift_narrative = build_drift_narrative(records, metric_key)
        drift_details = drift_summary.get("details", []) or []

    behavior_summary: Dict[str, Any] = {}
    behavior_details: List[Dict[str, Any]] = []
    behavior_narrative = ""

    if records:
        behavior_summary = build_behavior_change_summary(records, metric_key)
        behavior_details = behavior_summary.get("details", []) or []
        behavior_narrative = build_behavior_change_report_block(records, metric_key)

    if correlation_report_block:
        narrative = f"{narrative}\n\n{correlation_report_block}"

    if ranking_summary:
        narrative = f"{narrative}\n\nRanking automático de variables operativas:\n{ranking_summary}"

    if drift_narrative:
        narrative = (
            f"{narrative}\n\nDeriva progresiva (drift):\n"
            f"{drift_narrative}\n"
            f"Señales con drift: {drift_summary.get('total_drift_signals', 0)} | "
            f"Severidad máxima: {drift_summary.get('top_severity', 'None')}."
        )

    if behavior_narrative:
        narrative = f"{narrative}\n\n{behavior_narrative}"

    # ============================================================
    # 🔥 FIX DEFINITIVO DEL REPORTE
    # ============================================================
    narrative = final_report_cleanup(narrative)

    image_bytes, image_error = build_export_png_bytes(fig=fig)

    item_payload = {
        "id": make_export_state_key(
            [
                "report-trend",
                metric_key,
                panel_title,
                machine,
                point,
                len(st.session_state.report_items),
            ]
        ),
        "type": "trends",
        "title": panel_title,
        "notes": narrative,
        "signal_id": signal_id,
        "figure": None,
        "image_bytes": image_bytes,
        "image_error": image_error,
        "source_module": "04_Trends",
        "report_payload_version": "v2",
        "machine": machine,
        "point": point,
        "variable": variable,
        "timestamp": timestamp,
        "correlation_payload": correlation_payload,
        "lag_payload": lag_payload,
        "ranking_summary": ranking_summary,
        "ranking_payload": ranking_payload,
        "drift_summary": drift_summary,
        "drift_details": drift_details,
        "behavior_summary": behavior_summary,
        "behavior_details": behavior_details,
        # Ciclo 17.5.3 — autodiag ejecutivo + threshold source
        "autodiagnostic": {
            "headline": _autodiag_for_pdf.get("headline", ""),
            "prose": list(_autodiag_for_pdf.get("prose", []) or []),
            "recommendations": list(_autodiag_for_pdf.get("recommendations", []) or []),
            "status": _autodiag_for_pdf.get("status", "unknown"),
            "status_label": _autodiag_for_pdf.get("status_label", ""),
        },
        "threshold_source": dict(st.session_state.get("wm_tr_threshold_source", {}) or {}),
    }
    append_report_item_and_persist(item_payload)
    st.session_state["wm_tr_last_report_debug"] = {
        "notes_len": len(str(narrative or "")),
        "report_items_count": len(st.session_state.report_items),
        "last_title": panel_title,
        "has_image": image_bytes is not None,
    }
    return True, image_error


with st.sidebar:
    st.markdown("### Signal Selection")

    signal_name_map = {r.display_name: r.trend_id for r in records_all}
    signal_names = list(signal_name_map.keys())

    if records_all:
        if st.session_state.wm_tr_primary_signal_id not in [r.trend_id for r in records_all]:
            st.session_state.wm_tr_primary_signal_id = records_all[0].trend_id

        current_primary_name = next(
            (r.display_name for r in records_all if r.trend_id == st.session_state.wm_tr_primary_signal_id),
            signal_names[0],
        )

        selected_primary_name = st.selectbox(
            "Primary vibration signal",
            options=signal_names,
            index=signal_names.index(current_primary_name),
        )
        st.session_state.wm_tr_primary_signal_id = signal_name_map[selected_primary_name]

        extra_options = [name for name in signal_names if name != selected_primary_name]
        default_extra_names = [
            r.display_name
            for r in records_all
            if r.trend_id in st.session_state.wm_tr_extra_signal_ids and r.display_name in extra_options
        ]

        selected_extra_names = st.multiselect(
            "Additional vibration signals",
            options=extra_options,
            default=default_extra_names,
        )
        st.session_state.wm_tr_extra_signal_ids = [signal_name_map[name] for name in selected_extra_names]
    else:
        st.info("No vibration trend CSV loaded.")
        st.session_state.wm_tr_primary_signal_id = None
        st.session_state.wm_tr_extra_signal_ids = []

    st.markdown("### Operational Selection")
    operational_name_map = {r.display_name: r.op_id for r in operational_records_all}
    operational_names = list(operational_name_map.keys())

    # Ciclo 17.8 — Quick-pick por familia. En CSVs DCS típicos hay
    # 12+ variables; seleccionar de a uno es tedioso. Estos botones
    # cargan TODAS las de un tipo de un click.
    if operational_records_all:
        _by_family: Dict[str, List[str]] = {}
        for r in operational_records_all:
            _by_family.setdefault(r.family, []).append(r.op_id)
        # Orden de prioridad de display
        _fam_order = ["pressure", "temperature", "flow", "frequency",
                      "speed", "power", "vibration", "generic"]
        _fam_label = {
            "pressure": "🔵 Presiones",
            "temperature": "🟠 Temperaturas",
            "flow": "🟢 Flujos",
            "frequency": "🟡 Frecuencia/VFD",
            "speed": "⚙️ Velocidad",
            "power": "⚡ Potencia",
            "vibration": "📊 Vibración",
            "generic": "📂 Otros",
        }
        _qp_cols = st.columns(2)
        _qp_idx = 0
        for fam in _fam_order:
            if fam not in _by_family:
                continue
            _ids = _by_family[fam]
            with _qp_cols[_qp_idx % 2]:
                if st.button(
                    f"{_fam_label.get(fam, fam.title())} ({len(_ids)})",
                    key=f"wm_tr_qp_{fam}",
                    use_container_width=True,
                    help=f"Cargar todas las variables de tipo {fam}",
                ):
                    st.session_state.wm_tr_operational_signal_ids = list(_ids)
                    st.rerun()
            _qp_idx += 1
        # Botón "Limpiar" si hay seleccionadas
        if st.session_state.get("wm_tr_operational_signal_ids"):
            if st.button("✕ Limpiar selección operativa",
                         key="wm_tr_op_clear",
                         use_container_width=True,
                         type="secondary"):
                st.session_state.wm_tr_operational_signal_ids = []
                st.rerun()

    default_operational_names = [
        r.display_name
        for r in operational_records_all
        if r.op_id in st.session_state.wm_tr_operational_signal_ids and r.display_name in operational_names
    ]
    selected_operational_names = st.multiselect(
        "Variables operativas (presión, temperatura, flujo, VFD…)",
        options=operational_names,
        default=default_operational_names,
        help="Seleccione variables individuales o use los botones rápidos de arriba para cargar por familia.",
    )
    st.session_state.wm_tr_operational_signal_ids = [operational_name_map[name] for name in selected_operational_names]

    st.markdown("### Display")
    display_options = ["Combined", "Independent", "Mixed"]
    if st.session_state.wm_tr_display_mode not in display_options:
        st.session_state.wm_tr_display_mode = "Combined"
    st.session_state.wm_tr_display_mode = st.selectbox(
        "Display mode",
        options=display_options,
        index=display_options.index(st.session_state.wm_tr_display_mode),
    )

    st.markdown("### Trend Processing")
    metric_key = st.selectbox("Metric", options=["Amplitude", "Phase", "Speed"], index=0)
    show_markers = st.checkbox("Show markers", value=False)
    show_anomaly_markers = st.checkbox("Show anomaly markers", value=True)
    show_drift_analysis = st.checkbox("Show drift analysis", value=True)
    fill_area = st.checkbox("Fill area (single trend)", value=True)

    st.markdown("### Axes")
    y_axis_mode = st.selectbox("Primary Y-axis scale", ["Auto", "Manual"], index=0)

    y_axis_manual_min: Optional[float] = None
    y_axis_manual_max: Optional[float] = None
    if y_axis_mode == "Manual":
        c1, c2 = st.columns(2)
        with c1:
            y_axis_manual_min = float(st.number_input("Y min", value=0.0, step=0.1, format="%.3f"))
        with c2:
            y_axis_manual_max = float(st.number_input("Y max", value=5.0, step=0.1, format="%.3f"))

    operational_y_axis_mode = st.selectbox("Operational Y-axis scale", ["Auto", "Manual"], index=0)
    operational_y_manual_min: Optional[float] = None
    operational_y_manual_max: Optional[float] = None
    if operational_y_axis_mode == "Manual":
        c3, c4 = st.columns(2)
        with c3:
            operational_y_manual_min = float(st.number_input("Operational Y min", value=0.0, step=1.0, format="%.3f"))
        with c4:
            operational_y_manual_max = float(st.number_input("Operational Y max", value=60.0, step=1.0, format="%.3f"))

    x_axis_mode = st.selectbox("X-axis scale", ["Auto", "Manual"], index=0)
    show_right_info_box = st.checkbox("Show info box", value=True)
    show_legend = st.checkbox("Show legend", value=True)

    st.markdown("### Alarms")
    # =========================================================
    # Ciclo 17.5.2 — sugerencia de Warning/Danger desde el Vault
    # =========================================================
    # Tomamos los records actualmente seleccionados (primary +
    # extras) y consultamos el Sensor Map de la instancia activa
    # para extraer alarm / danger por sensor. El resultado se
    # PRE-LLENA en los inputs pero el usuario puede sobrescribir
    # libremente (caso típico: la norma dice 4 mil pp pero el
    # cliente exige 3 mil pp como criterio conservador).
    _sel_for_thr = [
        r for r in records_all
        if r.trend_id in (
            [st.session_state.wm_tr_primary_signal_id]
            + list(st.session_state.wm_tr_extra_signal_ids)
        )
    ]
    try:
        _inst_for_thr = _get_instance_for_threshold(trend_active_instance_id) if trend_active_instance_id else None
    except Exception:
        _inst_for_thr = None
    _sensors_for_thr = list(_inst_for_thr.sensors) if _inst_for_thr else []
    _machine_group_for_thr = str(trend_instance_state.get("machine_group") or "class_iv")

    _thr_suggestion = suggest_trend_thresholds(
        _sel_for_thr,
        _sensors_for_thr,
        metric_key=metric_key,
        machine_group=_machine_group_for_thr,
        instance=_inst_for_thr,  # Ciclo 17.9 — para chequear iso_norm_code/class
    )

    # Source chip
    _src = _thr_suggestion.get("source", "default")
    _src_color = {
        "Sensor Map":           ("#10b981", "#ecfdf5"),
        "ISO/Bently default":   ("#0ea5e9", "#e0f2fe"),
        "ISO 20816 class I":    ("#0ea5e9", "#e0f2fe"),
        "ISO 20816 class II":   ("#0ea5e9", "#e0f2fe"),
        "ISO 20816 class III":  ("#0ea5e9", "#e0f2fe"),
        "ISO 20816 class IV":   ("#0ea5e9", "#e0f2fe"),
        "Default accelerometer": ("#0ea5e9", "#e0f2fe"),
        "Default":              ("#9ca3af", "#f1f5f9"),
        "default":              ("#9ca3af", "#f1f5f9"),
        "n/a":                  ("#9ca3af", "#f1f5f9"),
    }.get(_src, ("#9ca3af", "#f1f5f9"))

    st.markdown(
        f"<div style='background:{_src_color[1]};border-left:3px solid {_src_color[0]};"
        f"padding:8px 12px;border-radius:6px;margin:2px 0 8px 0;font-size:0.85rem;'>"
        f"<b>Setpoints sugeridos:</b> {_src}<br>"
        f"<span style='color:#475569;font-size:0.78rem;'>"
        f"{_thr_suggestion.get('detail','')}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Resolver defaults para los number_input. Si el usuario ya
    # overrideó manualmente en sesiones previas (wm_tr_warning_override /
    # wm_tr_danger_override), respetamos el override; si no, usamos
    # la sugerencia.
    _suggested_w = _thr_suggestion.get("warning")
    _suggested_d = _thr_suggestion.get("danger")
    _default_w = float(_suggested_w) if _suggested_w is not None else 3.500
    _default_d = float(_suggested_d) if _suggested_d is not None else 5.000

    # Boton "Aplicar Vault" que limpia overrides y pre-llena con sugerencia.
    _btn_cols = st.columns([0.6, 0.4])
    with _btn_cols[0]:
        if st.button("Aplicar setpoints sugeridos", key="wm_tr_apply_vault_thr", use_container_width=True):
            st.session_state.pop("wm_tr_warning_override_value", None)
            st.session_state.pop("wm_tr_danger_override_value", None)
            st.rerun()

    warning_enabled = st.checkbox("Enable warning line", value=True, key="wm_tr_warning_enabled")
    warning_value: Optional[float] = None
    _w_is_override = False
    if warning_enabled:
        _w_session_default = st.session_state.get("wm_tr_warning_override_value", _default_w)
        warning_value = float(st.number_input(
            "Warning value",
            value=float(_w_session_default),
            step=0.1, format="%.3f",
            key="wm_tr_warning_input",
        ))
        st.session_state["wm_tr_warning_override_value"] = warning_value
        if abs(warning_value - _default_w) > 1e-9:
            _w_is_override = True
            st.caption(
                f"⚙️ Override del cliente · sugerido: {_default_w:.3f}"
            )

    danger_enabled = st.checkbox("Enable danger line", value=True, key="wm_tr_danger_enabled")
    danger_value: Optional[float] = None
    _d_is_override = False
    if danger_enabled:
        _d_session_default = st.session_state.get("wm_tr_danger_override_value", _default_d)
        danger_value = float(st.number_input(
            "Danger value",
            value=float(_d_session_default),
            step=0.1, format="%.3f",
            key="wm_tr_danger_input",
        ))
        st.session_state["wm_tr_danger_override_value"] = danger_value
        if abs(danger_value - _default_d) > 1e-9:
            _d_is_override = True
            st.caption(
                f"⚙️ Override del cliente · sugerido: {_default_d:.3f}"
            )

    # Persistir el origen para que el reporte/PDF lo cite correctamente
    st.session_state["wm_tr_threshold_source"] = {
        "warning_value": warning_value,
        "danger_value": danger_value,
        "suggested_warning": _suggested_w,
        "suggested_danger": _suggested_d,
        "source": _src,
        "detail": _thr_suggestion.get("detail", ""),
        "warning_is_override": _w_is_override,
        "danger_is_override": _d_is_override,
        "machine_group": _machine_group_for_thr,
        # Ciclo 17.9 — referencia normativa para el reporte
        "norm_reference": _thr_suggestion.get("norm_reference", ""),
        "norm_code": getattr(_inst_for_thr, "iso_norm_code", "") if _inst_for_thr else "",
        "norm_class": getattr(_inst_for_thr, "iso_norm_class", "") if _inst_for_thr else "",
        "override_justification": (
            getattr(_inst_for_thr, "override_justification", "") if _inst_for_thr else ""
        ),
    }

selected_ids = [st.session_state.wm_tr_primary_signal_id] + st.session_state.wm_tr_extra_signal_ids
selected_ids = [sid for sid in selected_ids if sid is not None]

selected_records = [r for r in records_all if r.trend_id in selected_ids]
selected_records_sorted: List[TrendRecord] = []
for sid in selected_ids:
    rec = next((r for r in selected_records if r.trend_id == sid), None)
    if rec is not None:
        selected_records_sorted.append(rec)

selected_operational_ids = [sid for sid in st.session_state.wm_tr_operational_signal_ids if sid is not None]
selected_operational_records = [r for r in operational_records_all if r.op_id in selected_operational_ids]
selected_operational_records_sorted: List[OperationalRecord] = []
for sid in selected_operational_ids:
    rec = next((r for r in selected_operational_records if r.op_id == sid), None)
    if rec is not None:
        selected_operational_records_sorted.append(rec)

if st.session_state.wm_tr_display_mode in ["Combined", "Independent"] and not selected_records_sorted and not selected_operational_records_sorted:
    st.warning("No valid signals selected.")
    st.stop()

if st.session_state.wm_tr_display_mode == "Mixed" and (not selected_records_sorted or not selected_operational_records_sorted):
    st.warning("Mixed mode requiere al menos una señal de vibración y una señal operativa.")
    st.stop()

mixed_operational_notice: Optional[str] = None
if st.session_state.wm_tr_display_mode == "Mixed" and len(selected_operational_records_sorted) > 1:
    families = [r.family for r in selected_operational_records_sorted]
    first_family = families[0]
    filtered = [r for r in selected_operational_records_sorted if r.family == first_family]
    if len(filtered) != len(selected_operational_records_sorted):
        mixed_operational_notice = (
            "Mixed mode solo mezcla una familia operativa por eje secundario. "
            f"Se usarán únicamente las señales de tipo '{first_family}'."
        )
    selected_operational_records_sorted = filtered

logo_uri = get_logo_data_uri(LOGO_PATH)

if selected_records_sorted:
    time_options = get_time_options_for_records(selected_records_sorted, metric_key)
else:
    time_options = []

if (not time_options) and selected_operational_records_sorted:
    time_options = get_time_options_for_operational_records(selected_operational_records_sorted)

time_labels = [ts_to_label(ts) for ts in time_options]

if not time_labels:
    st.warning("No hay datos válidos para los cursores en la selección actual.")
    st.stop()


def get_valid_time_label(saved_value: str, fallback_label: str) -> str:
    saved_value = str(saved_value or "")
    if saved_value in time_labels:
        return saved_value
    return fallback_label


default_a_initial = time_labels[0]
default_a_current = time_labels[min(len(time_labels) - 1, max(0, len(time_labels) // 3))]
default_b_initial = time_labels[min(len(time_labels) - 1, max(0, (len(time_labels) * 2) // 3))]
default_b_current = time_labels[-1]
default_x_start = time_labels[0]
default_x_end = time_labels[-1]

st.session_state.wm_tr_cursor_a_initial = get_valid_time_label(st.session_state.wm_tr_cursor_a_initial, default_a_initial)
st.session_state.wm_tr_cursor_a_current = get_valid_time_label(st.session_state.wm_tr_cursor_a_current, default_a_current)
st.session_state.wm_tr_cursor_b_initial = get_valid_time_label(st.session_state.wm_tr_cursor_b_initial, default_b_initial)
st.session_state.wm_tr_cursor_b_current = get_valid_time_label(st.session_state.wm_tr_cursor_b_current, default_b_current)
st.session_state.wm_tr_x_manual_start = get_valid_time_label(st.session_state.wm_tr_x_manual_start, default_x_start)
st.session_state.wm_tr_x_manual_end = get_valid_time_label(st.session_state.wm_tr_x_manual_end, default_x_end)

x_axis_manual_start: Optional[pd.Timestamp] = None
x_axis_manual_end: Optional[pd.Timestamp] = None
if x_axis_mode == "Manual":
    x_axis_manual_start = label_to_ts(st.session_state.wm_tr_x_manual_start)
    x_axis_manual_end = label_to_ts(st.session_state.wm_tr_x_manual_end)

cursor_map = {
    "A Initial": label_to_ts(st.session_state.wm_tr_cursor_a_initial),
    "A Current": label_to_ts(st.session_state.wm_tr_cursor_a_current),
    "B Initial": label_to_ts(st.session_state.wm_tr_cursor_b_initial),
    "B Current": label_to_ts(st.session_state.wm_tr_cursor_b_current),
}

if x_axis_mode == "Manual":
    with st.expander("X-Axis Manual Window", expanded=False):
        st.markdown(
            """
            <div class="wm-control-shell">
                <div class="wm-control-title">X-Axis Window</div>
                <div class="wm-control-subtitle">Ajusta el inicio y fin del tiempo con sliders precisos.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            st.select_slider("X Start", options=time_labels, key="wm_tr_x_manual_start")
        with col_x2:
            st.select_slider("X End", options=time_labels, key="wm_tr_x_manual_end")

with st.expander("Cursor Controls", expanded=False):
    st.markdown(
        """
        <div class="wm-control-shell">
            <div class="wm-control-title">Cursor Controls</div>
            <div class="wm-control-subtitle">Referencias temporales A/B para comparar comportamiento entre dos momentos.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.select_slider("A Initial", options=time_labels, key="wm_tr_cursor_a_initial")
        st.select_slider("A Current", options=time_labels, key="wm_tr_cursor_a_current")
    with c2:
        st.select_slider("B Initial", options=time_labels, key="wm_tr_cursor_b_initial")
        st.select_slider("B Current", options=time_labels, key="wm_tr_cursor_b_current")


def render_trend_panel(
    panel_records: List[TrendRecord],
    panel_index: int,
    panel_label: str,
    panel_operational_records: Optional[List[OperationalRecord]] = None,
    mixed_mode: bool = False,
    operational_only_mode: bool = False,
) -> None:
    panel_operational_records = panel_operational_records or []

    fig = build_trend_figure(
        records=panel_records,
        metric_key=metric_key,
        show_markers=show_markers,
        show_anomaly_markers=show_anomaly_markers,
        fill_area=fill_area,
        y_axis_mode=y_axis_mode,
        y_axis_manual_min=y_axis_manual_min,
        y_axis_manual_max=y_axis_manual_max,
        x_axis_mode=x_axis_mode,
        x_axis_manual_start=x_axis_manual_start,
        x_axis_manual_end=x_axis_manual_end,
        warning_enabled=warning_enabled,
        warning_value=warning_value,
        danger_enabled=danger_enabled,
        danger_value=danger_value,
        show_right_info_box=show_right_info_box,
        show_legend=show_legend,
        logo_uri=logo_uri,
        cursor_map=cursor_map,
        operational_records=panel_operational_records,
        mixed_mode=mixed_mode,
        operational_only_mode=operational_only_mode,
        operational_y_axis_mode=operational_y_axis_mode,
        operational_y_manual_min=operational_y_manual_min,
        operational_y_manual_max=operational_y_manual_max,
    )

    export_state_key = make_export_state_key(
        [
            st.session_state.wm_tr_display_mode,
            panel_label,
            metric_key,
            y_axis_mode, y_axis_manual_min, y_axis_manual_max,
            operational_y_axis_mode, operational_y_manual_min, operational_y_manual_max,
            x_axis_mode, st.session_state.wm_tr_x_manual_start, st.session_state.wm_tr_x_manual_end,
            warning_enabled, warning_value, danger_enabled, danger_value,
            st.session_state.wm_tr_cursor_a_initial, st.session_state.wm_tr_cursor_a_current,
            st.session_state.wm_tr_cursor_b_initial, st.session_state.wm_tr_cursor_b_current,
            show_markers, show_anomaly_markers, fill_area, show_right_info_box, show_legend,
            "|".join([r.trend_id for r in panel_records]),
            "|".join([r.file_name for r in panel_records]),
            "|".join([r.point_clean for r in panel_records]),
            "|".join([r.op_id for r in panel_operational_records]),
            "|".join([r.variable for r in panel_operational_records]),
            mixed_mode, operational_only_mode,
        ]
    )

    if export_state_key not in st.session_state.wm_tr_export_store:
        st.session_state.wm_tr_export_store[export_state_key] = {"png_bytes": None, "error": None}

    st.markdown(f"### {panel_label}")

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False},
        key=f"wm_trends_plot_{export_state_key}",
    )

    st.markdown('<div class="wm-export-actions"></div>', unsafe_allow_html=True)
    left_pad, col_export1, col_export2, col_report, col_bode, right_pad = st.columns([1.6, 1.2, 1.2, 1.2, 1.3, 1.5])

    with col_export1:
        if st.button("Prepare PNG HD", key=f"prepare_png_{export_state_key}", use_container_width=True):
            with st.spinner("Generating HD export..."):
                png_bytes, export_error = build_export_png_bytes(fig=fig)
                st.session_state.wm_tr_export_store[export_state_key]["png_bytes"] = png_bytes
                st.session_state.wm_tr_export_store[export_state_key]["error"] = export_error

    with col_export2:
        png_bytes = st.session_state.wm_tr_export_store[export_state_key]["png_bytes"]
        if png_bytes is not None:
            st.download_button(
                "Download PNG HD",
                data=png_bytes,
                file_name=f"watermelon_trend_{panel_index + 1}_hd.png",
                mime="image/png",
                use_container_width=True,
                key=f"download_png_{export_state_key}",
            )
        else:
            st.button("Download PNG HD", disabled=True, use_container_width=True, key=f"download_disabled_{export_state_key}")

    with col_report:
        if st.button("Enviar a Reporte", key=f"send_report_{export_state_key}", use_container_width=True):
            image_ok, image_error = queue_trend_to_report(
                panel_records,
                fig,
                panel_label,
                metric_key,
                operational_records=panel_operational_records,
                operational_only_mode=operational_only_mode,
            )
            if image_ok:
                st.success("Trend enviado al reporte")
            else:
                st.error(image_error or "No fue posible enviar el trend al reporte.")

    with col_bode:
        bode_disabled = operational_only_mode or len(panel_records) == 0
        if st.button("Open linked Bode", key=f"open_bode_{export_state_key}", use_container_width=True, disabled=bode_disabled):
            push_linked_bode_context(panel_records, metric_key)
            st.switch_page("pages/07_Bode_Plot.py")

    if st.session_state.get("wm_tr_last_report_debug"):
        dbg = st.session_state["wm_tr_last_report_debug"]
        st.caption(
            f"Report debug → notes_len={dbg.get('notes_len')} | report_items={dbg.get('report_items_count')} | "
            f"title={dbg.get('last_title')} | has_image={dbg.get('has_image')}"
        )

    panel_error = st.session_state.wm_tr_export_store[export_state_key]["error"]
    if panel_error:
        st.warning(f"PNG export error: {panel_error}")

    # ----------------------------------------------------------------
    # Ciclo 17.5 P4 — Autodiagnóstico ejecutivo (síntesis del trend)
    # ----------------------------------------------------------------
    # Se muestra una síntesis Bently-style ANTES de los detectores
    # individuales, agrupando estado vs umbrales, pendiente, forecast,
    # anomalías, drift, cambio de régimen y vínculo operacional en una
    # sola lectura ejecutiva.
    if panel_records and not operational_only_mode:
        try:
            _autodiag = build_trend_autodiagnostic(
                panel_records,
                metric_key,
                warning_value=float(warning_value) if (warning_enabled and warning_value is not None) else None,
                danger_value=float(danger_value) if (danger_enabled and danger_value is not None) else None,
                operational_records=panel_operational_records,
            )
            _headline = _autodiag.get("headline", "")
            # Estilo sobrio alineado con Polar / Bode / SCL: header
            # markdown simple, headline en bold, prosa con st.write,
            # recomendaciones como prosa enumerada. Sin chips de
            # color, sin emojis grandes, sin border-left.
            st.markdown("### Diagnóstico ejecutivo")
            if _headline:
                st.markdown(f"**{_headline}**")
            for _para in _autodiag.get("prose", []) or []:
                if _para and str(_para).strip():
                    st.write(_para)
            _recs = _autodiag.get("recommendations", []) or []
            if _recs:
                st.write("Acciones recomendadas:")
                for _i, _r in enumerate(_recs, 1):
                    st.write(f"{_i}. {_r}")
        except Exception as _exc:
            st.caption(f"Diagnóstico no disponible ({_exc})")

    if panel_records:
        anomaly_summary = build_panel_anomaly_summary(panel_records, metric_key)
        st.markdown("#### Detección automática de anomalías")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("Anomalies", str(anomaly_summary.get("total_count", 0)))
        with a2:
            st.metric("Affected signals", str(anomaly_summary.get("affected_records", 0)))
        with a3:
            st.metric("Top severity", anomaly_summary.get("top_severity", "None"))
        st.info(anomaly_summary.get("interpretation", "Sin interpretación disponible."))

        anomaly_narrative = build_anomaly_narrative(panel_records, metric_key)
        st.markdown("**Interpretación técnica de anomalías:**")
        st.write(anomaly_narrative)

    if panel_records and show_drift_analysis:
        drift_summary = build_panel_drift_summary(panel_records, metric_key)
        drift_narrative = build_drift_narrative(panel_records, metric_key)

        st.markdown("#### Detección de drift (deriva progresiva)")
        d1, d2 = st.columns(2)
        with d1:
            st.metric("Signals with drift", str(drift_summary.get("total_drift_signals", 0)))
        with d2:
            st.metric("Top drift severity", drift_summary.get("top_severity", "None"))

        st.info(drift_summary.get("interpretation", "Sin interpretación disponible."))
        st.markdown("**Interpretación técnica de drift:**")
        st.write(drift_narrative)

    # ------------------------------------------------------------
    # 
    if panel_records:
        behavior_summary = build_behavior_change_summary(panel_records, metric_key)
        behavior_narrative = build_behavior_narrative(panel_records, metric_key)

        st.markdown("#### Cambio de comportamiento (F9-A)")
        details_valid = [
            d for d in behavior_summary.get("details", [])
            if d.get("valid") and d.get("classification") in ["Strong change", "Moderate change"]
        ]
        top_change = None
        if details_valid:
            top_change = sorted(
                details_valid,
                key=lambda d: float(d.get("change_score") or 0.0),
                reverse=True
            )[0]

        b1, b2, b3 = st.columns(3)

        with b1:
            st.metric("Signals analyzed", str(behavior_summary.get("count", 0)))

        with b2:
            st.metric("Top classification", behavior_summary.get("top_classification", "None"))

        with b3:
            if top_change and top_change.get("change_timestamp") is not None:
                st.metric(
                    "Main change timestamp",
                    f"{pretty_date(top_change.get('change_timestamp'))} {pretty_time(top_change.get('change_timestamp'))}"
                )
            else:
                st.metric("Main change timestamp", "—")

        st.info(behavior_summary.get("interpretation", "Sin interpretación"))

        st.markdown("**Interpretación técnica:**")
        st.write(behavior_narrative)


    # Automatic correlation: primary vibration vs first operational
    # ------------------------------------------------------------
    correlation_enabled = bool(panel_records) and bool(panel_operational_records)
    if correlation_enabled:
        primary_trend = panel_records[0]
        primary_operational = panel_operational_records[0]
        correlation_info = build_trend_operational_correlation(
            trend_record=primary_trend,
            operational_record=primary_operational,
            metric_key=metric_key,
        )

        st.markdown("#### Correlación automática vibración vs variable operativa")

        corr_value = correlation_info.get("corr_value")
        corr_text = format_number(corr_value, 3) if corr_value is not None else "—"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Correlation", corr_text)
        with c2:
            st.metric("Strength", correlation_info.get("strength") or "—")
        with c3:
            st.metric("Direction", correlation_info.get("direction") or "—")
        with c4:
            st.metric("Samples", str(correlation_info.get("sample_count") or 0))

        st.info(correlation_info.get("interpretation") or "Sin interpretación disponible.")

        scatter_fig = build_correlation_scatter_figure(correlation_info)
        st.plotly_chart(
            scatter_fig,
            use_container_width=True,
            config={"displaylogo": False},
            key=f"wm_trends_corr_{export_state_key}",
        )

        lag_info = build_lagged_correlation_analysis(
            trend_record=primary_trend,
            operational_record=primary_operational,
            metric_key=metric_key,
            max_lag_minutes=180,
            step_minutes=10,
        )

        st.markdown("#### Correlación con desfase temporal (lag)")
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            st.metric(
                "Best correlation",
                format_number(lag_info.get("best_corr"), 3),
            )
        with l2:
            best_lag_val = lag_info.get("best_lag_min")
            st.metric(
                "Best lag (min)",
                str(best_lag_val) if best_lag_val is not None else "—",
            )
        with l3:
            st.metric("Lag strength", lag_info.get("strength") or "—")
        with l4:
            st.metric("Lag direction", lag_info.get("direction") or "—")

        st.info(lag_info.get("interpretation") or "Sin interpretación disponible.")

        lag_fig = build_lag_correlation_figure(lag_info)
        st.plotly_chart(
            lag_fig,
            use_container_width=True,
            config={"displaylogo": False},
            key=f"wm_trends_lagcorr_{export_state_key}",
        )

        if len(panel_operational_records) > 1:
            ranking_df = build_operational_variable_ranking(
                trend_record=primary_trend,
                operational_records=panel_operational_records,
                metric_key=metric_key,
            )

            st.markdown("#### Ranking automático de variables operativas")
            ranking_summary = build_operational_variable_ranking_summary(ranking_df)
            st.info(ranking_summary)

            if not ranking_df.empty:
                ranking_view = ranking_df.copy()
                for col in ["Simple Corr", "Lag Corr", "Score"]:
                    ranking_view[col] = ranking_view[col].apply(lambda v: format_number(v, 3))
                st.dataframe(ranking_view, use_container_width=True, hide_index=True)


if mixed_operational_notice:
    st.info(mixed_operational_notice)

if st.session_state.wm_tr_display_mode == "Combined":
    if selected_records_sorted:
        combined_label = f"Trend Combined — {selected_records_sorted[0].machine}"
        render_trend_panel(selected_records_sorted, 0, combined_label)
    elif selected_operational_records_sorted:
        combined_label = f"Operational Combined — {selected_operational_records_sorted[0].machine}"
        render_trend_panel([], 0, combined_label, panel_operational_records=selected_operational_records_sorted, operational_only_mode=True)
elif st.session_state.wm_tr_display_mode == "Mixed":
    combined_label = f"Trend + Operational — {selected_records_sorted[0].machine}"
    render_trend_panel(
        selected_records_sorted,
        0,
        combined_label,
        panel_operational_records=selected_operational_records_sorted,
        mixed_mode=True,
    )
else:
    panel_idx = 0
    if selected_records_sorted:
        for idx, rec in enumerate(selected_records_sorted):
            render_trend_panel([rec], panel_idx, f"Trend {idx + 1} — {rec.point_clean}")
            panel_idx += 1
            if idx < len(selected_records_sorted) - 1 or selected_operational_records_sorted:
                st.markdown("---")
    if selected_operational_records_sorted:
        for idx, rec in enumerate(selected_operational_records_sorted):
            render_trend_panel([], panel_idx, f"Operational {idx + 1} — {rec.variable}", panel_operational_records=[rec], operational_only_mode=True)
            panel_idx += 1
            if idx < len(selected_operational_records_sorted) - 1:
                st.markdown("---")
