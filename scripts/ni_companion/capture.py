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
# Captura EMA con martillo
python capture.py --mode ema --output ./run1.tdms \\
    --fs 5120 --duration 2 --averages 5 \\
    --channels Hammer:0:IEPE:2.4 \\
    --channels 1YA:1:IEPE:100 \\
    --channels 2YA:2:IEPE:100

# Captura OMA continuous
python capture.py --mode oma --output ./oma_run.tdms \\
    --fs 10240 --duration 120 \\
    --channels 1YA:0:IEPE:100 \\
    --channels 2YA:1:IEPE:100 \\
    --channels VE5807:2:AC:200 \\
    --channels VE5808:3:AC:200

# Listar tarjetas conectadas
python capture.py --list-devices

Dependencias
------------
nidaqmx     — Driver Python NI (pip install nidaqmx)
NI-DAQmx    — Driver del sistema NI (descarga gratuita ni.com)

Formato de canales en CLI
-------------------------
NAME:INDEX:COUPLING:SENSITIVITY_mV_per_EU

Ejemplo:
  1YA:1:IEPE:100  → canal 1, IEPE coupling, 100 mV/g (Wilcoxon)
  VE5807:2:AC:200 → canal 2, AC coupling, 200 mV/mil (Bently)
  Hammer:0:IEPE:2.4 → canal 0, IEPE, 2.4 mV/N (PCB martillo)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_channel_spec(spec: str) -> dict:
    """Parsea NAME:INDEX:COUPLING:SENSITIVITY."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(f"Channel spec inválido: {spec}. Esperado NAME:INDEX:COUPLING:SENS")
    name, idx, coupling, sens = parts
    return {
        "name": name,
        "index": int(idx),
        "coupling": coupling.upper(),
        "sensitivity_mv_per_eu": float(sens),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NI-9234 capture companion")
    parser.add_argument("--mode", choices=["ema", "oma"], required=False,
                        help="ema=triggered impact, oma=continuous")
    parser.add_argument("--output", type=Path, help="Output .tdms file")
    parser.add_argument("--fs", type=float, default=5120, help="Sample rate Hz")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration seconds")
    parser.add_argument("--averages", type=int, default=5, help="Number of impacts (EMA only)")
    parser.add_argument("--channels", action="append", default=[],
                        help="Channel spec: NAME:INDEX:COUPLING:SENS_mV_per_EU")
    parser.add_argument("--list-devices", action="store_true",
                        help="Listar tarjetas NI conectadas y salir")

    args = parser.parse_args()

    try:
        import nidaqmx  # noqa
        from nidaqmx.system import System  # noqa
    except ImportError:
        print("ERROR: nidaqmx no instalado.")
        print("Ejecuta: pip install nidaqmx")
        print("Y descarga el driver NI-DAQmx en ni.com")
        return 1

    if args.list_devices:
        # TODO: implementar listing
        print("Sprint pendiente: list_available_devices() en core/modal/ni_daq.py")
        return 0

    if not args.mode or not args.output:
        parser.print_help()
        return 1

    # Parsear canales
    channels = [parse_channel_spec(s) for s in args.channels]
    print(f"Canales configurados: {len(channels)}")
    for ch in channels:
        print(f"  Ch{ch['index']}: {ch['name']} ({ch['coupling']}, {ch['sensitivity_mv_per_eu']} mV/EU)")

    print(f"\nModo: {args.mode}")
    print(f"Sample rate: {args.fs} Hz")
    print(f"Duración: {args.duration} s")
    print(f"Output: {args.output}")

    # TODO: integrar con core/modal/ni_daq.py una vez implementado
    print("\n⚠ Captura no implementada todavía — esperando sprint NI-DAQ.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
