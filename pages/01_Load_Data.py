from __future__ import annotations

from core.auth import require_login, render_user_menu

require_login()
render_user_menu()

import base64
import csv
import io
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# WATERMELON SYSTEM — LOAD DATA
# ============================================================

st.set_page_config(page_title="Watermelon System | Load Data", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = PROJECT_ROOT / "assets" / "watermelon_logo.png"


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .main > div {
            padding-top: 0.25rem;
        }

        .stApp {
            background-color: #f3f4f6;
        }

        section[data-testid="stSidebar"] {
            background: #e5e7eb;
            border-right: 1px solid #cbd5e1;
        }

        .wm-page-header {
            display: flex;
            align-items: center;
            gap: 18px;
            margin-bottom: 0.9rem;
        }

        .wm-page-logo {
            height: 68px;
            width: auto;
            display: block;
        }

        .wm-page-kicker {
            font-size: 0.92rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #2563eb;
            margin-bottom: 0.12rem;
        }

        .wm-page-title {
            font-size: 3.15rem;
            line-height: 1.0;
            font-weight: 800;
            color: #2f3343;
            margin: 0;
        }

        .wm-page-subtitle {
            font-size: 1.03rem;
            color: #475569;
            margin-top: 0.30rem;
        }

        .wm-status-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid #d7dde8;
            border-radius: 18px;
            padding: 18px 22px;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .wm-status-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.40rem;
        }

        .wm-status-text {
            font-size: 0.98rem;
            color: #475569;
            margin: 0;
        }

        div[data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.72);
            border: 1px solid #d7dde8;
            border-radius: 20px;
            padding: 8px 10px 2px 10px;
        }

        div[data-testid="stButton"] > button {
            min-height: 52px;
            border-radius: 14px;
            font-weight: 700;
        }

        .wm-ready-box {
            background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(248,250,252,0.90) 100%);
            border: 1px solid #d7dde8;
            border-radius: 20px;
            padding: 20px 22px;
            margin-top: 1rem;
        }

        .wm-ready-title {
            font-size: 1.08rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.35rem;
        }

        .wm-ready-subtitle {
            font-size: 0.98rem;
            color: #475569;
            margin-bottom: 0.9rem;
        }

        .wm-file-pill-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 0.65rem;
        }

        .wm-file-pill {
            display: inline-block;
            background: #eff6ff;
            color: #1e3a8a;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.88rem;
            font-weight: 600;
        }

        .wm-export-note {
            font-size: 0.93rem;
            color: #475569;
            padding-top: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_page_style()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_logo_base64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def normalize_text_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def to_numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:30])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except Exception:
        comma = sample.count(",")
        semi = sample.count(";")
        tab = sample.count("\t")
        if semi >= comma and semi >= tab:
            return ";"
        if tab >= comma and tab >= semi:
            return "\t"
        return ","


def _extract_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _parse_variable_sync_pattern(variable: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Example patterns:
    - Disp Wf(64X/32revs).KPHGEN
    - Disp Wf (64X / 32 revs)
    - 128X/16revs
    """
    if not variable:
        return None, None

    pattern = re.compile(r"(\d+)\s*[xX]\s*/\s*(\d+)\s*rev", flags=re.IGNORECASE)
    match = pattern.search(str(variable))
    if not match:
        return None, None

    try:
        samples_per_rev = int(match.group(1))
        number_of_revs = int(match.group(2))
        if samples_per_rev >= 3 and number_of_revs >= 1:
            return samples_per_rev, number_of_revs
    except Exception:
        pass

    return None, None


def canonicalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(metadata)

    def first_value(*keys: str) -> Optional[Any]:
        for key in keys:
            if key in out and str(out[key]).strip():
                return out[key]
        return None

    machine = first_value("Machine Name", "Machine")
    point = first_value("Point Name", "Point")
    variable = first_value("Variable")
    rpm = first_value("Sample Speed", "RPM", "Speed", "Running Speed")
    timestamp = first_value("Timestamp")
    x_unit = first_value("X-Axis Unit", "X Axis Unit")
    y_unit = first_value("Y-Axis Unit", "Y Axis Unit", "Unit", "Units")
    wf_amp = first_value("Wf Amp", "Waveform Amplitude")
    num_revs = first_value("Number of Revs", "Detected Number of Revs")
    keyphasor = first_value("Keyphasor Synced", "Keyphasor", "Associated Speed")
    samples_per_rev = first_value("Detected Samples Per Rev", "Samples Per Rev")
    axis = first_value("Detected Axis", "Axis")

    if machine is not None:
        out["Machine"] = machine
        out["Machine Name"] = machine

    if point is not None:
        out["Point"] = point
        out["Point Name"] = point

    if variable is not None:
        out["Variable"] = variable
    else:
        out["Variable"] = "Waveform"

    if rpm is not None:
        out["RPM"] = rpm
        out["Sample Speed"] = rpm

    if timestamp is not None:
        out["Timestamp"] = timestamp
    else:
        out["Timestamp"] = ""

    if x_unit is not None:
        out["X-Axis Unit"] = x_unit

    if y_unit is not None:
        out["Y-Axis Unit"] = y_unit

    if wf_amp is not None:
        out["Wf Amp"] = wf_amp

    if num_revs is not None:
        out["Number of Revs"] = num_revs

    if keyphasor is not None:
        out["Keyphasor Synced"] = keyphasor

    if samples_per_rev is not None:
        out["Detected Samples Per Rev"] = samples_per_rev

    if axis is not None:
        out["Detected Axis"] = axis

    return out


def find_data_header_line(lines: List[str], delimiter: str) -> Optional[int]:
    strong_names = {
        "x axis value",
        "y axis value",
        "x value",
        "y value",
        "time",
        "amplitude",
    }

    for i, line in enumerate(lines[:220]):
        stripped = line.strip()
        if not stripped or delimiter not in stripped:
            continue

        parts = [p.strip().strip('"') for p in stripped.split(delimiter)]
        if len(parts) < 2:
            continue

        normalized_parts = [normalize_text_key(p) for p in parts]
        if "x axis value" in normalized_parts and "y axis value" in normalized_parts:
            return i

        score = sum(1 for p in normalized_parts if p in strong_names)
        if score >= 2:
            return i

    return None


def extract_metadata_from_lines(lines: List[str], delimiter: str, header_idx: int) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    for line in lines[:header_idx]:
        stripped = line.strip()
        if not stripped or delimiter not in stripped:
            continue

        parts = [p.strip().strip('"') for p in stripped.split(delimiter, 1)]
        if len(parts) != 2:
            continue

        key, value = parts[0].strip(), parts[1].strip()
        if not key or not value:
            continue

        nk = normalize_text_key(key)
        if nk in {"x axis value", "y axis value"}:
            continue

        metadata[key] = value

    return canonicalize_metadata(metadata)


def read_csv_body_and_metadata(raw_text: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    lines = raw_text.splitlines()
    delimiter = detect_delimiter(raw_text)
    header_idx = find_data_header_line(lines, delimiter)

    if header_idx is None:
        raise ValueError("Could not detect tabular data header in CSV file.")

    metadata = extract_metadata_from_lines(lines, delimiter, header_idx)

    body_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body_text), sep=delimiter, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    return df, metadata


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cols = list(df.columns)
    normalized = {c: normalize_text_key(c) for c in cols}

    time_candidates = [
        "x axis value",
        "time",
        "time ms",
        "time s",
        "x axis",
        "x value",
    ]
    vib_candidates = [
        "y axis value",
        "amplitude",
        "vibration",
        "disp",
        "waveform",
        "signal",
        "y axis",
        "y value",
    ]

    time_col: Optional[str] = None
    vib_col: Optional[str] = None

    for col in cols:
        n = normalized[col]
        if any(tc == n for tc in time_candidates):
            time_col = col
            break

    for col in cols:
        n = normalized[col]
        if any(vc == n for vc in vib_candidates):
            vib_col = col
            break

    numeric_cols = [
        c for c in cols
        if pd.to_numeric(df[c], errors="coerce").notna().sum() >= max(3, int(len(df) * 0.6))
    ]

    if time_col is None and numeric_cols:
        time_col = numeric_cols[0]

    if vib_col is None:
        for c in numeric_cols:
            if c != time_col:
                vib_col = c
                break

    second_col: Optional[str] = None
    if vib_col is not None:
        for c in numeric_cols:
            if c not in {time_col, vib_col}:
                second_col = c
                break

    return time_col, vib_col, second_col


def parse_uploaded_csv(uploaded_file) -> Dict[str, Any]:
    raw_bytes = uploaded_file.getvalue()
    raw_text = raw_bytes.decode("utf-8-sig", errors="ignore")

    df, metadata = read_csv_body_and_metadata(raw_text)
    time_col, vib_col, second_col = detect_columns(df)

    issues: List[str] = []
    if time_col is None:
        issues.append("Time column not detected.")
    if vib_col is None:
        issues.append("Vibration column not detected.")

    return {
        "file_name": uploaded_file.name,
        "dataframe": df,
        "metadata": metadata,
        "time_column": time_col,
        "vibration_column": vib_col,
        "secondary_column": second_col,
        "issues": issues,
        "is_valid": (time_col is not None and vib_col is not None and not df.empty),
    }


def _finalize_sync_metadata(metadata: Dict[str, Any], total_samples: int) -> Dict[str, Any]:
    """
    Priority:
    1) Variable pattern like 64X/32revs
    2) Existing metadata values if consistent
    3) Complete missing value from total_samples
    Always preserves dict-based signal architecture.
    """
    metadata = dict(metadata)

    variable = str(metadata.get("Variable", "")).strip()
    spr_from_var, revs_from_var = _parse_variable_sync_pattern(variable)

    spr_meta = _extract_numeric(metadata.get("Detected Samples Per Rev", metadata.get("Samples Per Rev")))
    revs_meta = _extract_numeric(metadata.get("Number of Revs"))

    chosen_spr: Optional[int] = None
    chosen_revs: Optional[int] = None
    sync_source = "Unknown"

    # 1) Variable pattern has highest trust
    if spr_from_var is not None and revs_from_var is not None:
        chosen_spr = int(spr_from_var)
        chosen_revs = int(revs_from_var)
        sync_source = "Variable Pattern"

    # 2) Metadata fallback
    if chosen_spr is None and spr_meta is not None and spr_meta >= 3:
        chosen_spr = int(round(spr_meta))
        sync_source = "Metadata Samples/Rev"

    if chosen_revs is None and revs_meta is not None and revs_meta >= 1:
        chosen_revs = int(round(revs_meta))
        if sync_source == "Unknown":
            sync_source = "Metadata Revs"

    # 3) Complete missing side
    if chosen_spr is not None and chosen_revs is None and total_samples > 0:
        candidate_revs = total_samples / chosen_spr
        if candidate_revs >= 1 and abs(candidate_revs - round(candidate_revs)) < 1e-6:
            chosen_revs = int(round(candidate_revs))
            if sync_source == "Unknown":
                sync_source = "Computed from Samples/Rev"

    if chosen_revs is not None and chosen_spr is None and total_samples > 0:
        candidate_spr = total_samples / chosen_revs
        if candidate_spr >= 3 and abs(candidate_spr - round(candidate_spr)) < 1e-6:
            chosen_spr = int(round(candidate_spr))
            if sync_source == "Unknown":
                sync_source = "Computed from Revs"

    # 4) Fix inconsistency against total_samples
    if chosen_spr is not None and chosen_revs is not None:
        expected_samples = chosen_spr * chosen_revs

        if expected_samples != total_samples:
            if spr_from_var is not None and total_samples % spr_from_var == 0:
                chosen_spr = int(spr_from_var)
                chosen_revs = int(total_samples // spr_from_var)
                sync_source = "Variable Pattern Corrected"
            elif chosen_spr > 0 and total_samples % chosen_spr == 0:
                chosen_revs = int(total_samples // chosen_spr)
                sync_source = "Adjusted from Samples/Rev"
            elif chosen_revs > 0 and total_samples % chosen_revs == 0:
                chosen_spr = int(total_samples // chosen_revs)
                sync_source = "Adjusted from Revs"

    if chosen_spr is not None:
        metadata["Detected Samples Per Rev"] = int(chosen_spr)

    if chosen_revs is not None:
        metadata["Number of Revs"] = int(chosen_revs)

    metadata["Sync Metadata Source"] = sync_source

    if chosen_spr is not None and chosen_revs is not None:
        metadata["Sync Metadata Consistent"] = bool(chosen_spr * chosen_revs == total_samples)

    return metadata


def build_signal_from_parsed(parsed: Dict[str, Any]) -> SimpleNamespace:
    time_col = parsed["time_column"]
    vib_col = parsed["vibration_column"]
    df = parsed["dataframe"]

    time_arr = to_numeric_array(df[time_col])
    vib_arr = to_numeric_array(df[vib_col])

    valid_mask = np.isfinite(time_arr) & np.isfinite(vib_arr)
    time_arr = time_arr[valid_mask]
    vib_arr = vib_arr[valid_mask]

    if time_arr.size < 2 or vib_arr.size < 2:
        raise ValueError(f"{parsed['file_name']}: not enough valid numeric samples.")

    sort_idx = np.argsort(time_arr)
    time_arr = time_arr[sort_idx]
    vib_arr = vib_arr[sort_idx]

    unique_mask = np.ones_like(time_arr, dtype=bool)
    if time_arr.size > 1:
        unique_mask[1:] = np.diff(time_arr) > 0

    time_arr = time_arr[unique_mask]
    vib_arr = vib_arr[unique_mask]

    if time_arr.size < 2 or vib_arr.size < 2:
        raise ValueError(f"{parsed['file_name']}: not enough unique valid numeric samples.")

    metadata = canonicalize_metadata(dict(parsed["metadata"]))
    metadata["Detected Time Column"] = time_col
    metadata["Detected Vibration Column"] = vib_col
    metadata["File Name"] = parsed["file_name"]

    metadata["Machine"] = metadata.get("Machine", metadata.get("Machine Name", "Unknown"))
    metadata["Point"] = metadata.get("Point", metadata.get("Point Name", "Point 1"))
    metadata["Variable"] = metadata.get("Variable", "Waveform")
    metadata["RPM"] = metadata.get("RPM", metadata.get("Sample Speed", ""))
    metadata["Timestamp"] = metadata.get("Timestamp", "")
    metadata["Y-Axis Unit"] = metadata.get("Y-Axis Unit", metadata.get("Unit", ""))

    # Sample rate
    dt = np.diff(time_arr)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size > 0:
        median_dt = float(np.median(dt))
        sample_rate_hz = 1.0 / median_dt
        metadata["Detected Sample Rate [Hz]"] = sample_rate_hz

    # RPM numeric if possible
    rpm_numeric = _extract_numeric(metadata.get("RPM", metadata.get("Sample Speed")))
    if rpm_numeric is not None:
        metadata["RPM"] = rpm_numeric
        metadata["Sample Speed"] = rpm_numeric

    # Smart sync reconciliation
    metadata = _finalize_sync_metadata(metadata, total_samples=len(vib_arr))

    return SimpleNamespace(
        time=time_arr,
        x=vib_arr,
        metadata=metadata,
        file_name=parsed["file_name"],
    )



def apply_machine_context_to_parsed_files(
    parsed_files: List[Dict[str, Any]],
    machine_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []

    asset_type = str(machine_context.get("asset_type") or "").strip()
    machine_configuration = str(machine_context.get("machine_configuration") or "").strip()
    machine_description = str(machine_context.get("machine_description") or "").strip()
    primary_equipment = str(machine_context.get("primary_equipment") or "").strip()
    secondary_equipment = str(machine_context.get("secondary_equipment") or "").strip()

    context_summary_parts = [
        f"Asset Type: {asset_type}" if asset_type else "",
        f"Machine Configuration: {machine_configuration}" if machine_configuration else "",
        f"Primary Equipment: {primary_equipment}" if primary_equipment else "",
        f"Secondary Equipment: {secondary_equipment}" if secondary_equipment else "",
        machine_description,
    ]
    context_summary = " | ".join([part for part in context_summary_parts if part])

    for parsed in parsed_files:
        parsed_copy = dict(parsed)
        metadata = canonicalize_metadata(dict(parsed_copy.get("metadata", {})))

        metadata["Asset Type"] = asset_type
        metadata["Machine Configuration"] = machine_configuration
        metadata["Machine Description"] = machine_description
        metadata["Primary Equipment"] = primary_equipment
        metadata["Secondary Equipment"] = secondary_equipment
        metadata["Diagnostic Context Summary"] = context_summary

        parsed_copy["metadata"] = metadata
        enriched.append(parsed_copy)

    return enriched


def register_signals_to_session(parsed_files: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    # KEEP AS DICT to preserve compatibility with Time Waveforms and Spectrum
    st.session_state["signals"] = {}

    registered = 0
    errors: List[str] = []

    for parsed in parsed_files:
        try:
            if not parsed["is_valid"]:
                errors.append(f"{parsed['file_name']}: invalid file structure.")
                continue

            signal = build_signal_from_parsed(parsed)
            key = Path(parsed["file_name"]).stem
            base_key = key
            counter = 2

            while key in st.session_state["signals"]:
                key = f"{base_key}_{counter}"
                counter += 1

            st.session_state["signals"][key] = signal
            registered += 1
        except Exception as exc:
            errors.append(f"{parsed['file_name']}: {exc}")

    return registered, errors


def switch_to_time_waveforms() -> None:
    try:
        st.switch_page("pages/02_Time_Waveforms.py")
    except Exception:
        st.success("Signals generated successfully. Open Time Waveforms from the sidebar.")


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
logo_b64 = get_logo_base64(LOGO_PATH)

if logo_b64:
    st.markdown(
        f"""
        <div class="wm-page-header">
            <img class="wm-page-logo" src="data:image/png;base64,{logo_b64}" />
            <div>
                <div class="wm-page-kicker">Watermelon System</div>
                <div class="wm-page-title">Load Vibration Data</div>
                <div class="wm-page-subtitle">Upload one or many CSV vibration files. Metadata is detected silently and the workflow stays clean.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="wm-page-header">
            <div>
                <div class="wm-page-kicker">Watermelon System</div>
                <div class="wm-page-title">Load Vibration Data</div>
                <div class="wm-page-subtitle">Upload one or many CSV vibration files. Metadata is detected silently and the workflow stays clean.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Optional reset button
# ------------------------------------------------------------
col_reset_a, col_reset_b = st.columns([1.4, 5.6])

with col_reset_a:
    if st.button("Reset Loaded Signals", use_container_width=True):
        st.session_state["signals"] = {}
        st.success("Loaded signals cleared.")

with col_reset_b:
    st.markdown(
        '<div class="wm-export-note">Use reset if you want to fully clear previous uploaded signals before loading a fresh batch.</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# Uploader
# ------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload CSV vibration files",
    type=["csv"],
    accept_multiple_files=True,
    help="You can drag and drop many CSV files at once.",
)

parsed_files: List[Dict[str, Any]] = []
parse_errors: List[str] = []

if uploaded_files:
    for uploaded in uploaded_files:
        try:
            parsed_files.append(parse_uploaded_csv(uploaded))
        except Exception as exc:
            parse_errors.append(f"{uploaded.name}: {exc}")


# ------------------------------------------------------------
# Clean ready state
# ------------------------------------------------------------
if not uploaded_files:
    st.markdown(
        """
        <div class="wm-status-card">
            <div class="wm-status-title">Ready for bulk CSV loading</div>
            <p class="wm-status-text">Drop multiple files, detect metadata automatically, and generate the waveform dataset in one clean step.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

valid_files = [p for p in parsed_files if p["is_valid"]]
invalid_files = [p for p in parsed_files if not p["is_valid"]]

st.markdown(
    f"""
    <div class="wm-ready-box">
        <div class="wm-ready-title">{len(uploaded_files)} file(s) loaded</div>
        <div class="wm-ready-subtitle">Metadata was detected automatically. The page stays minimal and only shows the files ready to generate.</div>
        <div class="wm-file-pill-wrap">
            {''.join(f'<span class="wm-file-pill">{p["file_name"]}</span>' for p in valid_files[:24])}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if parse_errors:
    for msg in parse_errors:
        st.warning(msg)

if invalid_files:
    for parsed in invalid_files:
        text = "; ".join(parsed["issues"]) if parsed["issues"] else "Invalid file."
        st.warning(f"{parsed['file_name']}: {text}")


st.markdown("### Machine Diagnostic Context")

ctx1, ctx2 = st.columns(2)

with ctx1:
    asset_type = st.selectbox(
        "Asset type *",
        options=[
            "",
            "Turbogenerador",
            "Turbina de gas",
            "Generador eléctrico",
            "Motor eléctrico",
            "Bomba",
            "Compresor",
            "Ventilador",
            "Gearbox",
            "Otro",
        ],
        index=0,
        help="This context will be attached to every loaded signal and later used by diagnostics.",
    )

with ctx2:
    machine_configuration = st.selectbox(
        "Machine configuration *",
        options=["", "Simple", "Compuesta / tren de máquinas"],
        index=0,
        help="Use composite when the machine train has at least two main coupled assets.",
    )

primary_equipment = ""
secondary_equipment = ""

if machine_configuration == "Compuesta / tren de máquinas":
    ctx3, ctx4 = st.columns(2)
    with ctx3:
        primary_equipment = st.text_input(
            "Primary equipment *",
            placeholder="Ejemplo: Turbina LM6000",
        ).strip()
    with ctx4:
        secondary_equipment = st.text_input(
            "Secondary equipment *",
            placeholder="Ejemplo: Generador Brush",
        ).strip()

machine_description = st.text_area(
    "Machine technical description *",
    height=120,
    placeholder="Ejemplo: Turbina de gas LM6000 acoplada a generador Brush. No corresponde a sistema hidráulico.",
).strip()

context_errors: List[str] = []

if not asset_type:
    context_errors.append("Asset type is required.")
if not machine_configuration:
    context_errors.append("Machine configuration is required.")
if machine_configuration == "Compuesta / tren de máquinas":
    if not primary_equipment:
        context_errors.append("Primary equipment is required for composite machines.")
    if not secondary_equipment:
        context_errors.append("Secondary equipment is required for composite machines.")
if not machine_description:
    context_errors.append("Machine technical description is required.")

machine_context = {
    "asset_type": asset_type,
    "machine_configuration": machine_configuration,
    "primary_equipment": primary_equipment,
    "secondary_equipment": secondary_equipment,
    "machine_description": machine_description,
}

col_a, col_b = st.columns([1.7, 4.3])

with col_a:
    generate_clicked = st.button(
        "Generate Time Waveforms",
        type="primary",
        use_container_width=True,
        disabled=(len(valid_files) == 0 or len(context_errors) > 0),
    )

with col_b:
    st.markdown(
        '<div class="wm-export-note">Only valid CSV files will be registered into the signal session and passed to the approved Time Waveform Viewer workflow.</div>',
        unsafe_allow_html=True,
    )

if context_errors:
    for msg in context_errors:
        st.warning(msg)

if generate_clicked:
    context_enriched_files = apply_machine_context_to_parsed_files(valid_files, machine_context)
    count, errors = register_signals_to_session(context_enriched_files)

    if errors:
        for err in errors:
            st.warning(err)

    if count > 0:
        switch_to_time_waveforms()
    else:
        st.error("No valid signals could be generated.")