from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


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


def pretty_time(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.strftime("%I:%M %p").lstrip("0")


def pretty_date(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    return ts.strftime("%Y-%m-%d")


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


def _build_single_trend_narrative_from_df(
    point_name: str,
    metric_key: str,
    unit: str,
    df: pd.DataFrame,
) -> str:
    if df.empty:
        return (
            f"{point_name}: no se identificaron datos válidos para el análisis de "
            f"{metric_key.lower()}, por lo que no fue posible emitir diagnóstico automático."
        )

    analysis = _classify_trend_behavior(df["y"])
    sample_count = analysis.get("sample_count", 0)
    start_ts = safe_datetime(df["x"].iloc[0])
    end_ts = safe_datetime(df["x"].iloc[-1])

    base = (
        f"{point_name} — ventana analizada desde {pretty_date(start_ts)} {pretty_time(start_ts)} "
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
            "compatibles con condición transitoria o inestabilidad. "
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


def build_trend_report_narrative(
    records: List[Any],
    metric_key: str,
    operational_records: Optional[List[Any]] = None,
    operational_only_mode: bool = False,
) -> str:
    operational_records = operational_records or []
    lines: List[str] = []

    def _trend_metric_df(record: Any) -> tuple[pd.DataFrame, str]:
        if metric_key == "Amplitude":
            series = getattr(record, "y_value", pd.Series(dtype=float))
            unit = getattr(record, "y_axis_unit", "") or ""
        elif metric_key == "Phase":
            series = getattr(record, "phase", pd.Series(dtype=float))
            unit = "deg"
        elif metric_key == "Speed":
            series = getattr(record, "speed", pd.Series(dtype=float))
            unit = getattr(record, "speed_unit", "rpm") or "rpm"
        else:
            series = getattr(record, "y_value", pd.Series(dtype=float))
            unit = getattr(record, "y_axis_unit", "") or ""

        df = pd.DataFrame(
            {
                "x": pd.to_datetime(getattr(record, "x_time", pd.Series(dtype="datetime64[ns]")), errors="coerce"),
                "y": pd.to_numeric(series, errors="coerce"),
            }
        ).dropna(subset=["x", "y"])

        if not df.empty:
            df = df.sort_values("x").reset_index(drop=True)
        return df, unit

    def _operational_df(record: Any) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "x": pd.to_datetime(getattr(record, "x_time", pd.Series(dtype="datetime64[ns]")), errors="coerce"),
                "y": pd.to_numeric(getattr(record, "y_value", pd.Series(dtype=float)), errors="coerce"),
            }
        ).dropna(subset=["x", "y"])
        if not df.empty:
            df = df.sort_values("x").reset_index(drop=True)
        return df

    if not operational_only_mode:
        for record in records:
            df, unit = _trend_metric_df(record)
            point_name = getattr(record, "point_clean", None) or getattr(record, "point", None) or getattr(record, "file_name", "Signal")
            lines.append(_build_single_trend_narrative_from_df(point_name, metric_key, unit, df))

    if operational_records:
        op_lines: List[str] = []
        for record in operational_records:
            df = _operational_df(record)
            variable = getattr(record, "variable", "Operational Data")
            unit = getattr(record, "unit", "") or ""
            op_lines.append(_build_single_trend_narrative_from_df(variable, "Operational Data", unit, df))
        if op_lines:
            if operational_only_mode:
                lines.extend(op_lines)
            else:
                lines.append("Correlación operativa disponible:\n\n" + "\n\n".join(op_lines))

    return "\n\n".join(lines).strip() or "Sin interpretación técnica todavía."
