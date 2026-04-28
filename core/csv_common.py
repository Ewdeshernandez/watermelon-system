"""
core.csv_common
================

Utilidades compartidas para los parsers CSV de Watermelon System.

Watermelon tiene varios parsers especializados que consumen formatos CSV
distintos (Bently Nevada / Adapt waveform, Polar, Bode, Shaft Centerline,
Trend, Operational). Cada uno vive en su página correspondiente porque
los formatos no son intercambiables, pero TODOS comparten primitivas como:

- Decodificar UTF-8 con BOM
- Localizar la línea de header tabular dentro de un bloque que arranca con
  metadata key,value
- Parsear ese bloque de metadata
- Filtrar filas por columnas *_Status == "valid"
- Calcular media y desviación circular en grados
- Unwrap y smoothing circular de fase

Este módulo centraliza esas primitivas para evitar divergencias y reducir
duplicación. NO contiene parsers de formato — esos siguen en sus páginas.

Convenciones:
- Todas las funciones son puras, sin estado, sin Streamlit, sin I/O salvo
  el lectura del file_obj cuando se documenta.
- Compatibilidad estricta con el comportamiento previo de las copias locales
  que se reemplazan.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd


# =============================================================
# I/O
# =============================================================
def decode_csv_text(file_obj: Any, errors: str = "replace") -> str:
    """
    Decodifica el contenido de un file_obj (bytes o str) eliminando el BOM
    UTF-8. Mueve el cursor a 0 antes de leer cuando es posible.

    Args:
        file_obj: objeto con método .read() (UploadedFile de Streamlit, BytesIO,
            archivo abierto, etc.). Puede devolver bytes o str.
        errors: estrategia de errores de decodificación ("replace", "ignore",
            "strict"). Por defecto "replace" para preservar el cuerpo del CSV.

    Returns:
        Texto decodificado del archivo, sin BOM.
    """
    try:
        file_obj.seek(0)
    except Exception:
        pass

    raw = file_obj.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8-sig", errors=errors)
    return str(raw)


# =============================================================
# HEADER / METADATA
# =============================================================
def find_header_line(
    lines: List[str],
    required_signals: Iterable[str],
    max_search: int = 500,
) -> Optional[int]:
    """
    Devuelve el índice de la PRIMERA línea que contiene TODOS los strings
    de `required_signals` (búsqueda case-sensitive con `in`). Útil para
    localizar la línea de header tabular de un CSV cuyo cuerpo viene
    precedido por un bloque de metadata.

    Args:
        lines: líneas del archivo (text.splitlines()).
        required_signals: iterable de strings que deben aparecer todos en
            la misma línea para considerarla header.
        max_search: máximo de líneas a inspeccionar (defensa ante archivos
            anómalos muy largos).

    Returns:
        Índice de la línea de header, o None si no se encuentra.
    """
    signals = tuple(required_signals)
    if not signals:
        return None

    upper_bound = min(len(lines), max_search)
    for i in range(upper_bound):
        line = lines[i]
        if all(sig in line for sig in signals):
            return i
    return None


def parse_metadata_block(
    meta_lines: List[str],
    delimiter: str = ",",
) -> dict:
    """
    Parsea las líneas previas al header tabular como pares key,value
    separados por `delimiter`. Ignora líneas vacías y líneas sin
    delimitador. Si una clave aparece más de una vez, prevalece la última.

    Args:
        meta_lines: líneas previas a la línea de header.
        delimiter: separador entre key y value (default coma).

    Returns:
        Diccionario {key: value} con strings stripeados.
    """
    meta: dict = {}
    for line in meta_lines:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(delimiter, 1)]
        if len(parts) == 2:
            meta[parts[0]] = parts[1]
    return meta


# =============================================================
# STATUS FILTERING
# =============================================================
def filter_status_valid(
    df: pd.DataFrame,
    status_columns: Iterable[str],
    valid_value: str = "valid",
) -> pd.DataFrame:
    """
    Devuelve un nuevo DataFrame manteniendo SOLO las filas donde TODAS
    las columnas indicadas valen `valid_value` (case-insensitive con
    espacios stripeados). Las columnas que no existan en el DataFrame se
    ignoran silenciosamente.

    Compatible con el patrón Bently Nevada / Adapt en el que cada
    métrica trae una columna `*_Status` que indica la calidad del dato.
    """
    cols = list(status_columns)
    if not cols:
        return df.copy()

    target = valid_value.strip().lower()
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        col_norm = df[col].astype(str).str.strip().str.lower()
        mask &= col_norm.eq(target)

    return df[mask].copy()


# =============================================================
# CIRCULAR PHASE STATISTICS
# =============================================================
def circular_mean_deg(values: Any) -> float:
    """
    Media circular de ángulos en grados. Resultado normalizado a [0, 360).
    Acepta pd.Series, np.ndarray o list. Ignora NaN y valores no numéricos.

    Equivalente a la versión local en Polar / Bode.
    """
    if isinstance(values, pd.Series):
        vals = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    else:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return float("nan")

    rad = np.deg2rad(vals % 360.0)
    c = float(np.mean(np.cos(rad)))
    s = float(np.mean(np.sin(rad)))
    ang = np.rad2deg(np.arctan2(s, c))
    return float((ang + 360.0) % 360.0)


def circular_std_deg(values: Any) -> float:
    """
    Desviación estándar circular de ángulos en grados, derivada de la
    longitud media del vector circular R:
        sigma = sqrt(-2 ln R) en radianes, luego pasada a grados.
    Devuelve 0 si hay menos de 2 valores finitos.
    """
    if isinstance(values, pd.Series):
        vals = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    else:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]

    if vals.size < 2:
        return 0.0

    rad = np.deg2rad(vals % 360.0)
    z = np.mean(np.exp(1j * rad))
    r = float(np.clip(np.abs(z), 1e-12, 1.0))
    return float(np.degrees(np.sqrt(-2.0 * np.log(r))))


def unwrap_deg(series: pd.Series) -> pd.Series:
    """
    Aplica `np.unwrap` a una serie de fases en grados, manteniendo el
    índice original. Equivalente a la versión local en Bode.
    """
    vals = series.astype(float).to_numpy()
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(vals)))
    return pd.Series(unwrapped, index=series.index)


def circular_smooth_deg(phase_deg: pd.Series, window: int) -> pd.Series:
    """
    Smoothing circular de fase en grados con ventana centrada (rolling
    mean del vector complejo). Mantiene el índice original. Si window <= 1
    devuelve la serie sin modificar.

    Equivalente a la versión local en Bode.
    """
    if window is None or window <= 1:
        return phase_deg.astype(float).copy()

    rad = np.deg2rad(phase_deg.astype(float).to_numpy() % 360.0)
    c = pd.Series(np.cos(rad)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    s = pd.Series(np.sin(rad)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    out = np.rad2deg(np.arctan2(s, c))
    out = (out + 360.0) % 360.0
    return pd.Series(out, index=phase_deg.index)


__all__ = [
    "decode_csv_text",
    "find_header_line",
    "parse_metadata_block",
    "filter_status_valid",
    "circular_mean_deg",
    "circular_std_deg",
    "unwrap_deg",
    "circular_smooth_deg",
]
