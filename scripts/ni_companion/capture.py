#!/usr/bin/env python3
"""
scripts/ni_companion/capture.py — Companion script para captura NI-9234
=========================================================================

Este script corre en una laptop con NI-DAQmx driver instalado y captura
data del NI cDAQ-9234. Genera archivos .tdms que luego se importan al
Watermelon Modal Module vía la UI.

NO se ejecuta en Streamlit Cloud — solo localmente donde está conectado
el hardware NI.

Uso típico
----------
# Listar tarjetas conectadas
python capture.py --list-devices

# Captura EMA con martillo (5 promedios)
python capture.py --mode ema --output ./run1.tdms \\
    --fs 5120 --duration 2 --averages 5 \\
    --trigger-channel 0 --trigger-level 0.5 \\
    --channels Hammer:0:IEPE:2.4 \\
    --channels 1YA:1:IEPE:100 \\
    --channels 2YA:2:IEPE:100

# Captura OMA continuous (120 segundos)
python capture.py --mode oma --output ./oma_run.tdms \\
    --fs 10240 --duration 120 \\
    --channels 1YA:0:IEPE:100 \\
    --channels 2YA:1:IEPE:100 \\
    --channels VE5807:2:AC:200 \\
    --channels VE5808:3:AC:200

# Modo simulated — para development sin hardware (genera data sintética)
python capture.py --mode oma --simulated --output ./oma_sim.tdms \\
    --fs 5120 --duration 60 \\
    --channels 1YA:0:IEPE:100 \\
    --channels 2YA:1:IEPE:100

Dependencias
------------
nidaqmx     — Driver Python NI (pip install nidaqmx). Solo para captura real.
NI-DAQmx    — Driver del sistema NI (descarga gratuita ni.com)
npTDMS      — Lectura/escritura TDMS (pip install npTDMS)
numpy       — Procesamiento numérico

Formato de canales en CLI
-------------------------
NAME:INDEX:COUPLING:SENSITIVITY_mV_per_EU

Ejemplo:
  1YA:1:IEPE:100   → canal 1, IEPE coupling, 100 mV/g (Wilcoxon)
  VE5807:2:AC:200  → canal 2, AC coupling, 200 mV/mil (Bently)
  Hammer:0:IEPE:2.4 → canal 0, IEPE, 2.4 mV/N (PCB martillo)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permitir que el script encuentre el package core/modal/ del repo
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from core.modal.ni_daq import (  # noqa: E402
    AcquisitionConfig,
    ChannelConfig,
    capture,
    list_available_devices,
)


def parse_channel_spec(spec: str) -> ChannelConfig:
    """Parsea 'NAME:INDEX:COUPLING:SENSITIVITY' a ChannelConfig."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Channel spec inválido: '{spec}'.\n"
            f"Formato esperado: NAME:INDEX:COUPLING:SENS_mV_per_EU\n"
            f"Ejemplo: 1YA:1:IEPE:100"
        )
    name, idx, coupling, sens = parts
    coupling_norm = coupling.strip().upper()
    if coupling_norm not in ("IEPE", "AC", "DC"):
        raise ValueError(
            f"Coupling inválido '{coupling}'. Use IEPE, AC, o DC."
        )
    # Inferir unidad por sensitivity ranges típicos (heurística)
    sens_f = float(sens)
    if coupling_norm == "IEPE" and 50 <= sens_f <= 200:
        units = "g"  # accel típico
    elif coupling_norm == "AC" and 100 <= sens_f <= 250:
        units = "mil"  # prox Bently
    elif coupling_norm == "IEPE" and sens_f < 10:
        units = "N"  # martillo
    else:
        units = ""

    return ChannelConfig(
        channel_index=int(idx),
        name=name,
        coupling=coupling_norm,
        sensitivity_mv_per_eu=sens_f,
        units=units,
    )


def _print_progress(progress: float, status: str) -> None:
    """Callback de progreso simple — imprime en stderr para no contaminar stdout."""
    bar_width = 30
    filled = int(progress * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    sys.stderr.write(f"\r[{bar}] {progress*100:5.1f}% · {status:<60}")
    sys.stderr.flush()
    if progress >= 1.0:
        sys.stderr.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NI-9234 capture companion para Watermelon Modal Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["ema", "oma"],
                        help="ema=triggered impact hammer, oma=continuous")
    parser.add_argument("--output", type=Path, help="Output .tdms file")
    parser.add_argument("--fs", type=float, default=5120,
                        help="Sample rate Hz (se redondea a valor válido NI-9234)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Duration en segundos")
    parser.add_argument("--averages", type=int, default=5,
                        help="N° de impactos a promediar (modo EMA)")
    parser.add_argument("--trigger-channel", type=int, default=0,
                        help="Canal del martillo (modo EMA, 0..3)")
    parser.add_argument("--trigger-level", type=float, default=0.5,
                        help="Nivel del trigger en V (modo EMA)")
    parser.add_argument("--channels", action="append", default=[],
                        help="Channel spec: NAME:INDEX:COUPLING:SENS_mV_per_EU "
                             "(usa --channels múltiples veces, hasta 4)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device name (e.g. cDAQ1Mod1). Auto-detect si no se da.")
    parser.add_argument("--simulated", action="store_true",
                        help="Genera data sintética en lugar de capturar del NI. "
                             "Útil para development sin hardware conectado.")
    parser.add_argument("--list-devices", action="store_true",
                        help="Listar tarjetas NI conectadas y salir")

    args = parser.parse_args()

    # Comando: listar dispositivos
    if args.list_devices:
        try:
            devices = list_available_devices()
        except ImportError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if not devices:
            print("No se encontraron tarjetas NI conectadas.")
            return 0
        print("Tarjetas NI detectadas:")
        for d in devices:
            print(f"  · {d['name']:<15} {d['product_type']:<20} serial={d['serial']}")
        return 0

    # Validar args para captura
    if not args.mode or not args.output:
        parser.print_help()
        print("\nERROR: --mode y --output son requeridos para capturar", file=sys.stderr)
        return 1

    if not args.channels:
        print("ERROR: al menos un --channels es requerido", file=sys.stderr)
        return 1

    # Parsear canales
    try:
        channels = [parse_channel_spec(s) for s in args.channels]
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Construir config — preservar EMA/OMA aunque sea simulado
    # (Ciclo 23.156 — bug fix: antes ambos quedaban como "simulated" y la
    # bifurcación en el Tab Adquisición no podía determinar EMA vs OMA.)
    if args.simulated:
        mode_internal = f"simulated_{args.mode}"  # simulated_ema o simulated_oma
    else:
        mode_internal = "ema_triggered" if args.mode == "ema" else "oma_continuous"

    config = AcquisitionConfig(
        mode=mode_internal,
        sample_rate_hz=args.fs,
        duration_s=args.duration,
        channels=channels,
        device_name=args.device,
        trigger_channel=args.trigger_channel if args.mode == "ema" else None,
        trigger_level_V=args.trigger_level,
        n_averages=args.averages,
        output_tdms_path=args.output,
    )

    # Banner
    print("=" * 60, file=sys.stderr)
    print(f"NI-9234 Companion · {'SIMULATED' if args.simulated else 'REAL'} MODE",
          file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Modo:      {args.mode}", file=sys.stderr)
    print(f"  Fs:        {args.fs} Hz", file=sys.stderr)
    print(f"  Duración:  {args.duration} s", file=sys.stderr)
    if args.mode == "ema":
        print(f"  N° impactos: {args.averages}", file=sys.stderr)
        print(f"  Trigger:   ch{args.trigger_channel} > {args.trigger_level} V",
              file=sys.stderr)
    print(f"  Canales:   {len(channels)}", file=sys.stderr)
    for ch in channels:
        print(f"    ch{ch.channel_index}: {ch.name:<10} "
              f"{ch.coupling:<5} {ch.sensitivity_mv_per_eu} mV/{ch.units or 'EU'}",
              file=sys.stderr)
    print(f"  Output:    {args.output}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Captura
    try:
        out_path = capture(config, on_progress=_print_progress)
    except ImportError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Si no tienes hardware, usa --simulated", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR durante captura: {exc}", file=sys.stderr)
        return 3

    print(f"\n✓ TDMS generado: {out_path}", file=sys.stderr)
    print(str(out_path))  # stdout limpio para piping
    return 0


if __name__ == "__main__":
    sys.exit(main())
