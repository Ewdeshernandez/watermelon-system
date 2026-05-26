#!/usr/bin/env python3
"""
scripts/capture_companion/capture.py — Companion script captura maleta maleta de adquisición
=============================================================================

Corre en la laptop de planta con driver del fabricante driver instalado y captura data
de una maleta maleta de adquisición + hasta 8× módulo de adquisición (32 canales BNC en frente).
Genera archivos .tdms que luego se importan al Watermelon Modal Module
vía la UI (cuando hay internet de vuelta en la oficina).

NO se ejecuta en Streamlit Cloud — solo localmente donde está el hardware.

Uso típico
----------
# 1. Listar chasis y verificar qué módulos están instalados
python capture.py --list-devices
python capture.py --list-modules

# 2. Captura EMA con martillo (5 promedios) — 1 martillo + 3 acelerómetros
python capture.py --mode ema --output ./run1.tdms \\
    --fs 5120 --duration 2 --averages 5 \\
    --trigger-bnc 1 --trigger-level 0.5 \\
    --channels Hammer:1:IEPE:2.4 \\
    --channels 1YA:2:IEPE:100 \\
    --channels 2YA:3:IEPE:100 \\
    --channels 3YA:4:IEPE:100

# 3. Captura OMA con MALETA COMPLETA — 32 canales simultáneos
#    (streaming a TDMS, RAM constante ~5 MB sin importar la duración)
python capture.py --mode oma --output ./oma_full.tdms \\
    --fs 5120 --duration 300 --fn-low 8 \\
    --channels 1YA:1:IEPE:100 \\
    --channels 1YV:2:IEPE:100 \\
    --channels 1XA:3:IEPE:100 \\
    --channels 1XV:4:IEPE:100 \\
    ...repite para BNC 5..32...
    --channels 8YV:32:IEPE:100

# 4. Modo simulated — para development sin hardware
python capture.py --mode oma --simulated --output ./oma_sim.tdms \\
    --fs 5120 --duration 60 \\
    --channels 1YA:1:IEPE:100 \\
    --channels 2YA:5:IEPE:100

Dependencias
------------
nidaqmx     — Driver Python NI (pip install nidaqmx). Solo para captura real.
driver del fabricante    — Driver del sistema NI (descarga gratuita sitio del fabricante)
npTDMS      — Lectura/escritura TDMS (pip install npTDMS)
numpy       — Procesamiento numérico

Formato de canales en CLI (v3.31.201+)
--------------------------------------
NAME:BNC:COUPLING:SENSITIVITY_mV_per_EU

Donde BNC es el puerto frontal de la maleta (1..32):
  BNC 1..4   → Slot 1 (Mod1)
  BNC 5..8   → Slot 2 (Mod2)
  ...
  BNC 29..32 → Slot 8 (Mod8)

Ejemplos:
  1YA:1:IEPE:100   → BNC 1 (Mod1/ai0), IEPE, 100 mV/g (Wilcoxon)
  3YV:9:IEPE:100   → BNC 9 (Mod3/ai0), IEPE, 100 mV/g
  VE5807:17:AC:200 → BNC 17 (Mod5/ai0), AC, 200 mV/mil (Bently proximity)
  Hammer:1:IEPE:2.4 → BNC 1, IEPE, 2.4 mV/N (PCB martillo modal)

Backward compatibility v3.31.200-: si BNC <= 3 y solo hay 1 módulo
instalado, también se acepta como channel_index legacy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permitir que el script encuentre el package core/modal/ del repo
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from core.modal.acq_backend import (  # noqa: E402
    AcquisitionConfig,
    ChannelConfig,
    capture,
    discover_acq_modules,
    list_available_devices,
)


def parse_channel_spec(spec: str) -> ChannelConfig:
    """
    Parsea 'NAME:BNC:COUPLING:SENSITIVITY' a ChannelConfig.

    BNC va de 1 a 32 (puerto del frente de la maleta maleta de adquisición).
    Internamente se convierte a (module_slot, channel_index).
    """
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Channel spec inválido: '{spec}'.\n"
            f"Formato esperado: NAME:BNC:COUPLING:SENS_mV_per_EU\n"
            f"Donde BNC es el puerto 1..32 de la maleta maleta de adquisición.\n"
            f"Ejemplo: 1YA:5:IEPE:100  (BNC 5 = Mod2/ai0)"
        )
    name, bnc_str, coupling, sens = parts
    coupling_norm = coupling.strip().upper()
    if coupling_norm not in ("IEPE", "AC", "DC"):
        raise ValueError(
            f"Coupling inválido '{coupling}'. Use IEPE, AC, o DC."
        )

    try:
        bnc_port = int(bnc_str)
    except ValueError:
        raise ValueError(
            f"BNC port inválido '{bnc_str}'. Debe ser entero 1..32."
        )
    if not (1 <= bnc_port <= 32):
        raise ValueError(
            f"BNC port {bnc_port} fuera de rango [1..32]. "
            f"La maleta maleta de adquisición + 8× módulo de adquisición soporta máximo 32 canales."
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
        bnc_port=bnc_port,
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
        description="módulo de adquisición capture companion para Watermelon Modal Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["ema", "oma"],
                        help="ema=triggered impact hammer, oma=continuous")
    parser.add_argument("--output", type=Path, help="Output .tdms file")
    parser.add_argument("--fs", type=float, default=5120,
                        help="Sample rate Hz (se redondea a valor válido módulo de adquisición)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration en segundos. "
                             "EMA: 1-2 s tipico (default 2 s). "
                             "OMA: T_min >= 2000/fn_low (Brincker & Ventura 2015). "
                             "Default OMA: 200 s (asume fn_low ~10 Hz). "
                             "Para fn_low menor, sube manualmente o usa --fn-low.")
    parser.add_argument("--fn-low", type=float, default=None,
                        help="OMA: frecuencia natural mas baja esperada (Hz). "
                             "Si se da, valida duration >= 2000/fn_low y reescribe "
                             "el default si --duration no se paso. Norma: "
                             "Brincker & Ventura 2015, ISO 18649.")
    parser.add_argument("--averages", type=int, default=5,
                        help="N de impactos a promediar (modo EMA). "
                             "ISO 7626-5 §6.3: minimo 3, recomendado 5-10.")
    parser.add_argument("--trigger-bnc", type=int, default=None,
                        help="BNC port del martillo (modo EMA, 1..32). "
                             "El canal debe estar en --channels también.")
    parser.add_argument("--trigger-channel", type=int, default=None,
                        help="DEPRECATED v3.31.201: usa --trigger-bnc. "
                             "Si se da, se interpreta como bnc_port si es 1..32, "
                             "o como channel_index legacy si es 0..3.")
    parser.add_argument("--trigger-level", type=float, default=0.5,
                        help="Nivel del trigger en V (modo EMA)")
    parser.add_argument("--channels", action="append", default=[],
                        help="Channel spec: NAME:BNC:COUPLING:SENS_mV_per_EU "
                             "(usa --channels múltiples veces, hasta 32). "
                             "BNC = puerto del frente de la maleta (1..32).")
    parser.add_argument("--chassis", type=str, default="cDAQ1",
                        help="Nombre del chasis (default 'cDAQ1'). "
                             "Si hay más de una maleta usa 'cDAQ2', etc.")
    parser.add_argument("--device", type=str, default=None,
                        help="DEPRECATED v3.31.201: usa --chassis. "
                             "Si se da en formato legacy 'cDAQ1Mod1', se extrae "
                             "el chasis ('cDAQ1') automáticamente.")
    parser.add_argument("--chunk-seconds", type=float, default=1.0,
                        help="OMA streaming: segundos por chunk write al TDMS. "
                             "Default 1.0 = balance bueno entre overhead I/O y "
                             "RAM. Subir a 2-5 si el disco es lento.")
    parser.add_argument("--simulated", action="store_true",
                        help="Genera data sintética en lugar de capturar del NI. "
                             "Útil para development sin hardware conectado.")
    parser.add_argument("--list-devices", action="store_true",
                        help="Listar todos los dispositivos NI conectados y salir")
    parser.add_argument("--list-modules", action="store_true",
                        help="Listar módulos módulo de adquisición instalados en el chasis "
                             "con su BNC range. Útil para validar maleta antes "
                             "de configurar canales.")

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

    # Comando: listar módulos módulo de adquisición del chasis con BNC ranges
    if args.list_modules:
        try:
            modules = discover_acq_modules(args.chassis)
        except ImportError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if not modules:
            print(f"No se encontraron módulos módulo de adquisición en chasis '{args.chassis}'.")
            print("Verifica:")
            print("  1. Que la maleta maleta de adquisición esté conectada por USB")
            print("  2. Que los módulo de adquisición estén bien encajados en los slots")
            print("  3. Que driver del fabricante driver esté instalado (NI MAX debe verlos)")
            return 0
        print(f"Módulos módulo de adquisición en chasis '{args.chassis}':")
        print(f"{'Slot':<6}{'Device':<15}{'Serial':<15}{'BNC ports':<12}")
        print("-" * 50)
        for m in modules:
            bnc_start, bnc_end = m["bnc_range"]
            print(f"{m['slot']:<6}{m['device_name']:<15}{m['serial']:<15}"
                  f"{bnc_start}..{bnc_end}")
        print("-" * 50)
        total_ch = len(modules) * 4
        print(f"Total: {len(modules)} módulos × 4 ch = {total_ch} canales simultáneos disponibles")
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

    # -----------------------------------------------------------------
    # Validacion normativa de tiempos
    # EMA: ISO 7626-5 §6.3 — minimo 3 promedios
    # OMA: Brincker & Ventura 2015 — T_min >= 2000/fn_low (recomendado),
    #      piso 1000/fn_low
    # -----------------------------------------------------------------
    if args.mode == "ema":
        if args.duration is None:
            args.duration = 2.0  # default EMA
        if args.averages < 3:
            print(f"ERROR: --averages {args.averages} < 3. "
                  f"ISO 7626-5 §6.3 exige minimo 3 promedios.", file=sys.stderr)
            return 4
        if args.averages < 5:
            print(f"WARN: --averages {args.averages} cumple piso pero el "
                  f"recomendado es 5-10 (ISO 7626-5 §6.3).", file=sys.stderr)
    else:  # oma
        if args.fn_low is not None and args.fn_low > 0:
            t_min_strict = 2000.0 / args.fn_low
            t_min_floor = 1000.0 / args.fn_low
            if args.duration is None:
                args.duration = max(120.0, t_min_strict)
                print(f"INFO: --duration auto = {args.duration:.0f} s "
                      f"(2000/fn_low={t_min_strict:.0f} s, fn_low={args.fn_low} Hz)",
                      file=sys.stderr)
            elif args.duration < t_min_floor:
                print(f"ERROR: --duration {args.duration:.0f} s < piso absoluto "
                      f"{t_min_floor:.0f} s para fn_low={args.fn_low} Hz "
                      f"(Brincker & Ventura 2015). Norma exige >= 1000/fn_low, "
                      f"recomendado >= 2000/fn_low = {t_min_strict:.0f} s.",
                      file=sys.stderr)
                return 5
            elif args.duration < t_min_strict:
                print(f"WARN: --duration {args.duration:.0f} s cumple piso "
                      f"({t_min_floor:.0f} s) pero esta por debajo del "
                      f"recomendado {t_min_strict:.0f} s (Brincker & Ventura 2015).",
                      file=sys.stderr)
        else:
            if args.duration is None:
                args.duration = 200.0  # default OMA conservador (~ fn_low=10 Hz)
                print(f"INFO: --duration auto = 200 s (default OMA sin --fn-low). "
                      f"Si fn_low < 10 Hz, pasar --fn-low para validar.",
                      file=sys.stderr)
            elif args.duration < 60:
                print(f"WARN: --duration {args.duration:.0f} s puede ser "
                      f"insuficiente para OMA. Sin --fn-low no puedo validar "
                      f"contra norma. Para maquinaria industrial tipica "
                      f"(fn_low ~5-10 Hz), recomendado 200-400 s.", file=sys.stderr)

    # Construir config — preservar EMA/OMA aunque sea simulado
    # (Ciclo 23.156 — bug fix: antes ambos quedaban como "simulated" y la
    # bifurcación en el Tab Adquisición no podía determinar EMA vs OMA.)
    if args.simulated:
        mode_internal = f"simulated_{args.mode}"  # simulated_ema o simulated_oma
    else:
        mode_internal = "ema_triggered" if args.mode == "ema" else "oma_continuous"

    # Resolver trigger: prefiere --trigger-bnc, cae a --trigger-channel legacy
    trigger_resolved = None
    if args.mode == "ema":
        if args.trigger_bnc is not None:
            if not (1 <= args.trigger_bnc <= 32):
                print(f"ERROR: --trigger-bnc {args.trigger_bnc} fuera de [1..32]",
                      file=sys.stderr)
                return 1
            trigger_resolved = args.trigger_bnc
        elif args.trigger_channel is not None:
            trigger_resolved = args.trigger_channel  # bnc o legacy index
        else:
            print("ERROR: modo EMA requiere --trigger-bnc (o --trigger-channel legacy)",
                  file=sys.stderr)
            return 1

    config = AcquisitionConfig(
        mode=mode_internal,
        sample_rate_hz=args.fs,
        duration_s=args.duration,
        channels=channels,
        chassis_name=args.chassis,
        device_name=args.device,  # legacy, se normaliza en __post_init__
        trigger_channel=trigger_resolved,
        trigger_level_V=args.trigger_level,
        n_averages=args.averages,
        oma_chunk_seconds=args.chunk_seconds,
        output_tdms_path=args.output,
    )

    # Banner
    print("=" * 70, file=sys.stderr)
    print(f"maleta de adquisición + módulo de adquisición Companion · "
          f"{'SIMULATED' if args.simulated else 'REAL'} MODE",
          file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  Chasis:    {args.chassis}", file=sys.stderr)
    print(f"  Modo:      {args.mode}", file=sys.stderr)
    print(f"  Fs:        {args.fs} Hz", file=sys.stderr)
    print(f"  Duracion:  {args.duration} s", file=sys.stderr)
    if args.mode == "ema":
        print(f"  N impactos: {args.averages}  (norma: ISO 7626-5 §6.3, min 3)",
              file=sys.stderr)
        print(f"  Trigger:   BNC {trigger_resolved} > {args.trigger_level} V",
              file=sys.stderr)
    elif args.mode == "oma":
        if args.fn_low:
            _tstrict = 2000.0 / args.fn_low
            print(f"  fn_low:    {args.fn_low} Hz  (T_min recomendado {_tstrict:.0f} s)",
                  file=sys.stderr)
            print(f"  Norma:     Brincker & Ventura 2015 / ISO 18649", file=sys.stderr)
        print(f"  Streaming: chunks de {args.chunk_seconds}s a TDMS (RAM constante)",
              file=sys.stderr)
    print(f"  Canales:   {len(channels)} (max 32 con maleta completa)",
          file=sys.stderr)
    for ch in channels:
        print(f"    BNC {ch.bnc_port:>2} (Mod{ch.module_slot}/ai{ch.channel_index})  "
              f"{ch.name:<10} {ch.coupling:<5} "
              f"{ch.sensitivity_mv_per_eu} mV/{ch.units or 'EU'}",
              file=sys.stderr)
    print(f"  Output:    {args.output}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

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
