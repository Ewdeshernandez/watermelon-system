"""
tools.test_rotordynamics
========================

Smoke test del módulo core.rotordynamics contra un CSV real
(Bode o Polar). Auto-detecta el formato.

Uso:
    python tools/test_rotordynamics.py <path/al/csv> [operating_rpm]

Ejemplos:
    python tools/test_rotordynamics.py "polar 25 marzo.csv" 3600
    python tools/test_rotordynamics.py "bode 3 27 de marzo.csv" 14000

Verifica:
    1. Carga del CSV (Bode o Polar)
    2. Detección automática de velocidades críticas
    3. Cálculo de Q factor para cada crítica
    4. Evaluación de margen API 684 contra velocidad de operación
    5. Clasificación ISO 20816-2 zona

Imprime una tabla legible para validar manualmente que los resultados
hacen sentido físico desde la óptica Cat IV / API 684.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# Permitir ejecutar el script standalone desde la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.csv_common import (
    decode_csv_text,
    filter_status_valid,
    find_header_line,
    parse_metadata_block,
)
from core.rotordynamics import (
    detect_critical_speeds,
    evaluate_api684_margin,
    iso_20816_2_zone,
    mils_to_micrometers,
)


def _detect_format(lines: list[str]) -> str:
    """Auto-detecta si es CSV Bode o Polar por las palabras clave del header."""
    bode_idx = find_header_line(lines, ("X-Axis Value", "Y-Axis Value", "Phase", "Timestamp"))
    polar_idx = find_header_line(lines, ("Amp", "Phase", "Speed", "Timestamp"))

    if bode_idx is not None and polar_idx is not None:
        # Si ambos matchean (raro), gana el primero en aparecer
        return "bode" if bode_idx <= polar_idx else "polar"
    if bode_idx is not None:
        return "bode"
    if polar_idx is not None:
        return "polar"
    return "unknown"


def load_bode_or_polar_csv(path: Path) -> Tuple[dict, pd.DataFrame, str]:
    """
    Carga un Bode o Polar CSV usando csv_common. Devuelve metadata,
    DataFrame con columnas (rpm, amp, phase) ya agregadas por velocidad,
    y el tipo detectado.
    """
    with open(path, "rb") as f:
        text = decode_csv_text(f, errors="replace")
    lines = text.splitlines()

    fmt = _detect_format(lines)

    if fmt == "bode":
        hi = find_header_line(lines, ("X-Axis Value", "Y-Axis Value", "Phase", "Timestamp"))
        meta = parse_metadata_block(lines[:hi])
        df = pd.read_csv(io.StringIO("\n".join(lines[hi:])), encoding="utf-8-sig")
        df["rpm"] = pd.to_numeric(df["X-Axis Value"], errors="coerce")
        df["amp"] = pd.to_numeric(df["Y-Axis Value"], errors="coerce")
        df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")
        df = df.dropna(subset=["rpm", "amp", "phase"])
        df = filter_status_valid(df, ["Y-Axis Status", "Phase Status"])
    elif fmt == "polar":
        hi = find_header_line(lines, ("Amp", "Phase", "Speed", "Timestamp"))
        meta = parse_metadata_block(lines[:hi])
        df = pd.read_csv(io.StringIO("\n".join(lines[hi:])), encoding="utf-8-sig")
        df["rpm"] = pd.to_numeric(df["Speed"], errors="coerce")
        df["amp"] = pd.to_numeric(df["Amp"], errors="coerce")
        df["phase"] = pd.to_numeric(df["Phase"], errors="coerce")
        df = df.dropna(subset=["rpm", "amp", "phase"])
        df = filter_status_valid(df, ["Amp Status", "Phase Status", "Speed Status"])
    else:
        raise ValueError(
            f"No se reconoce el formato del CSV. Esperaba header tipo Bode "
            f"(X-Axis Value, Y-Axis Value, Phase, Timestamp) o Polar "
            f"(Amp, Phase, Speed, Timestamp)."
        )

    # Filtrar amp == 0 (zero-fill al inicio del run-up)
    df = df[df["amp"] > 0]

    grouped = (
        df.groupby("rpm", as_index=False)
        .agg(amp=("amp", "median"), phase=("phase", "median"))
        .sort_values("rpm", kind="stable")
        .reset_index(drop=True)
    )
    return meta, grouped, fmt


def fmt(v, fmt_str: str = ".2f") -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return format(v, fmt_str)


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("ERROR: debes pasar el path del CSV.")
        print("Uso: python tools/test_rotordynamics.py <bode_o_polar.csv> [operating_rpm]")
        print()
        print("Ejemplos:")
        print('  python tools/test_rotordynamics.py "polar 25 marzo.csv" 3600')
        print('  python tools/test_rotordynamics.py "bode 3 27 de marzo.csv" 14000')
        return 1

    csv_path = Path(args[0])
    if not csv_path.exists():
        print(f"ERROR: no existe el archivo: {csv_path}")
        return 1

    operating_rpm = float(args[1]) if len(args) > 1 else 3600.0

    print("=" * 75)
    print("SMOKE TEST — core.rotordynamics")
    print("=" * 75)
    print(f"CSV:              {csv_path.name}")
    print(f"Operating speed:  {operating_rpm:.0f} rpm")

    meta, df, fmt_detected = load_bode_or_polar_csv(csv_path)

    yunit = (
        meta.get("Y-Axis Unit")
        or meta.get("Amp Unit")
        or "—"
    ).strip()

    print(f"Formato detectado: {fmt_detected.upper()}")
    print(f"Machine:          {meta.get('Machine Name', '—')}")
    print(f"Point:            {meta.get('Point Name', '—')}")
    print(f"Y/Amp Unit:       {yunit}")
    print(f"Variable:         {meta.get('Variable', '—')}")
    print(f"Filas agrupadas:  {len(df)}")
    print(f"RPM range:        {df['rpm'].min():.0f} → {df['rpm'].max():.0f}")
    print(f"Amp peak global:  {df['amp'].max():.3f} {yunit}")
    print()

    # =============================================================
    # DETECCIÓN DE CRÍTICAS
    # =============================================================
    criticals = detect_critical_speeds(
        rpm=df["rpm"].to_numpy(),
        amp=df["amp"].to_numpy(),
        phase=df["phase"].to_numpy(),
    )

    print(f"Velocidades críticas detectadas: {len(criticals)}")
    print()

    if not criticals:
        print("No se detectaron picos que cumplan los criterios:")
        print("  - prominencia ≥ 5% del pico global")
        print("  - cambio de fase ≥ 40°")
        print("  - separación mínima 150 rpm entre picos")
        print()
        print("Esto puede ser correcto si:")
        print("  - La máquina opera supercrítica y la 1ª crítica está bajo el rango medido.")
        print("  - El rotor es altamente amortiguado y no muestra resonancias claras.")
        print("  - El run-up no cruza modos significativos.")
        return 0

    print("-" * 110)
    print(
        f"{'#':>3}  {'rpm':>7}  {'amp':>8}  {'phase':>8}  {'Δfase':>8}  "
        f"{'FWHM':>7}  {'Q':>6}  {'conf':>5}  {'N1':>7}  {'N2':>7}"
    )
    print("-" * 110)

    for i, cs in enumerate(criticals, start=1):
        print(
            f"{i:>3}  "
            f"{cs.rpm:>7.0f}  "
            f"{cs.amp_peak:>8.3f}  "
            f"{cs.phase_at_peak:>7.1f}°  "
            f"{cs.phase_change_deg:>7.1f}°  "
            f"{fmt(cs.fwhm_rpm, '7.1f')}  "
            f"{fmt(cs.q_factor, '6.2f')}  "
            f"{cs.confidence:>5.2f}  "
            f"{fmt(cs.n1_rpm, '7.0f')}  "
            f"{fmt(cs.n2_rpm, '7.0f')}"
        )

    print("-" * 110)
    print()

    # =============================================================
    # API 684 MARGIN POR CADA CRÍTICA
    # =============================================================
    print("=" * 75)
    print(f"API 684 — Margen de separación contra operating_rpm = {operating_rpm:.0f}")
    print("=" * 75)

    for i, cs in enumerate(criticals, start=1):
        margin = evaluate_api684_margin(
            critical_rpm=cs.rpm,
            operating_rpm=operating_rpm,
            q_factor=cs.q_factor,
        )
        print()
        print(f"Crítica #{i} @ {cs.rpm:.0f} rpm  →  zona {margin.zone}")
        print(f"  Q factor:           {fmt(cs.q_factor, '.2f')}")
        print(f"  Margen actual:      {margin.actual_margin_pct:.1f}%")
        print(f"  Margen requerido:   {margin.required_margin_pct:.1f}%")
        print(f"  Conforme API 684:   {'SI' if margin.compliant else 'NO'}")
        print(f"  → {margin.narrative}")

    # =============================================================
    # ISO 20816-2 ZONE PARA EL PEAK GLOBAL
    # =============================================================
    print()
    print("=" * 75)
    print("ISO 20816-2 — Severidad sobre la amplitud máxima global")
    print("=" * 75)

    peak_amp_csv = float(df["amp"].max())
    yunit_lower = yunit.lower()

    if "mil" in yunit_lower:
        peak_amp_um_pp = mils_to_micrometers(peak_amp_csv)
        unit_note = f"({peak_amp_csv:.3f} mil pp → {peak_amp_um_pp:.1f} µm pp)"
    elif "µm" in yunit_lower or "um" in yunit_lower:
        peak_amp_um_pp = peak_amp_csv
        unit_note = f"({peak_amp_csv:.1f} µm pp)"
    else:
        peak_amp_um_pp = peak_amp_csv
        unit_note = f"(asumido µm pp; unidad CSV='{yunit}')"

    iso_zone = iso_20816_2_zone(
        amplitude=peak_amp_um_pp,
        measurement_type="shaft_displacement",
        machine_group="group2",
        operating_speed_rpm=operating_rpm,
    )

    print()
    print(f"Amplitud:          {unit_note}")
    print(f"Zona ISO:          {iso_zone.zone}")
    print(f"Descripción:       {iso_zone.zone_description}")
    print(
        f"Umbrales (µm pp):  A/B={iso_zone.boundary_AB:.0f}  "
        f"B/C={iso_zone.boundary_BC:.0f}  C/D={iso_zone.boundary_CD:.0f}"
    )
    print(f"Group:             {iso_zone.machine_group}")
    print(f"Op speed class:    {iso_zone.operating_speed_rpm:.0f} rpm")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
