"""
tools/ni_check.py — PRUEBA DE ORO: diagnóstico del hardware NI
==============================================================

Corre en el PC de sitio (Windows) con NI-DAQmx + `pip install nidaqmx`.

Dos modos:
  · POR DEFECTO (proximidad): lee VOLTAJE, IEPE OFF (seguro para proximidad).
      python tools/ni_check.py
  · ACELERÓMETROS: enciende IEPE (2 mA) en los 4 canales del Mod2.
      python tools/ni_check.py --iepe            (sensib 100 mV/g por defecto)
      python tools/ni_check.py --iepe --sens 100 (poné la de TU acelerómetro)

Opciones: --mod1 cDAQ1Mod1 --mod2 cDAQ1Mod2 --fs 25600 --secs 2
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod1", default=None, help="Módulo del keyphasor (ej. cDAQ1Mod1)")
    ap.add_argument("--mod2", default=None, help="Módulo de los sensores (ej. cDAQ1Mod2)")
    ap.add_argument("--fs", type=float, default=25600.0)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--iepe", action="store_true", help="IEPE ON (acelerómetros en Mod2)")
    ap.add_argument("--sens", type=float, default=100.0, help="mV/g del acelerómetro")
    args = ap.parse_args()

    try:
        import nidaqmx
        from nidaqmx.constants import (AcquisitionType, TerminalConfiguration,
                                       ExcitationSource, AccelUnits)
    except ImportError:
        print("✗ nidaqmx no está instalado.\n    1) Instalá NI-DAQmx (ni.com)\n"
              "    2) pip install nidaqmx")
        return 2

    sysm = nidaqmx.system.System.local()
    devs = list(sysm.devices)
    print("=== Dispositivos NI detectados ===")
    if not devs:
        print("✗ Ninguno. ¿cDAQ conectado por USB y encendido? Revisá NI MAX.")
        return 3
    mods_9234 = []
    for d in devs:
        try:
            pt = d.product_type
        except Exception:  # noqa: BLE001
            pt = "?"
        print(f"  · {d.name:16s}  {pt}")
        if "9234" in str(pt):
            mods_9234.append(d.name)

    mod1 = args.mod1 or (mods_9234[0] if len(mods_9234) >= 1 else None)
    mod2 = args.mod2 or (mods_9234[1] if len(mods_9234) >= 2 else None)
    if not mod2 or (not args.iepe and not mod1):
        print("\n✗ No pude autodetectar los módulos. Pasalos a mano con --mod1/--mod2.")
        return 4

    # Modo acelerómetro: solo los 4 del Mod2 con IEPE. Modo proximidad: kph + 4 radiales.
    if args.iepe:
        phys = [f"{mod2}/ai{i}" for i in range(4)]
        names = ["A1", "A2", "A3", "A4"]
        print(f"\nMODO ACELERÓMETROS · IEPE ON (2 mA) · {args.sens:.0f} mV/g · {mod2}/ai0..ai3")
    else:
        phys = [f"{mod1}/ai0"] + [f"{mod2}/ai{i}" for i in range(4)]
        names = ["KPH", "R1", "R2", "R3", "R4"]
        print(f"\nMODO PROXIMIDAD · IEPE OFF · keyphasor={mod1}/ai0 · radiales={mod2}/ai0..ai3")

    n = int(args.fs * args.secs)
    settle = 3.0
    n_settle = int(args.fs * settle)
    print(f"Leyendo: descarto {settle:.0f} s de asentamiento y mido {args.secs:.0f} s a "
          f"{args.fs:.0f} Hz…")

    task = nidaqmx.Task()
    unit = "g" if args.iepe else "V"
    try:
        for p in phys:
            if args.iepe:
                task.ai_channels.add_ai_accel_chan(
                    p, sensitivity=args.sens, units=AccelUnits.G,
                    current_excit_source=ExcitationSource.INTERNAL, current_excit_val=0.002)
            else:
                task.ai_channels.add_ai_voltage_chan(
                    p, min_val=-5.0, max_val=5.0,
                    terminal_config=TerminalConfiguration.DEFAULT)
        task.timing.cfg_samp_clk_timing(rate=args.fs, sample_mode=AcquisitionType.CONTINUOUS,
                                        samps_per_chan=int(args.fs * (settle + args.secs + 1)))
        task.start()
        task.read(number_of_samples_per_channel=n_settle, timeout=settle + 10)   # descartar
        data = np.asarray(task.read(number_of_samples_per_channel=n, timeout=args.secs + 10),
                          dtype=float)
    except Exception as e:  # noqa: BLE001
        print(f"✗ Error leyendo: {type(e).__name__}: {e}")
        try:
            task.close()
        except Exception:  # noqa: BLE001
            pass
        return 5
    finally:
        try:
            task.close()
        except Exception:  # noqa: BLE001
            pass

    if data.ndim == 1:
        data = data[None, :]

    print(f"\n=== Señales ({unit}) ===")
    print(f"{'canal':6s} {'min':>10s} {'max':>10s} {'rms(AC)':>10s} {'pp':>10s}")
    for i, nm in enumerate(names):
        x = data[i]
        ac = x - np.mean(x)
        print(f"{nm:6s} {x.min():10.4f} {x.max():10.4f} {np.sqrt(np.mean(ac**2)):10.4f} {np.ptp(x):10.4f}")

    if args.iepe:
        print("\n✓ Si algún A# muestra rms(AC) > 0 (vibración real), el acelerómetro está OK. "
              "Los canales sin acelerómetro conectado dan valores raros/altos = normal.")
    else:
        kph = data[0]
        dev = np.abs(kph - np.median(kph))
        if dev.max() < 1e-3:
            print("\n⚠ Keyphasor plano (sin pulso). En el 9234 (AC) es esperable.")
        else:
            thr = 0.5 * dev.max()
            rising = np.flatnonzero((~(dev[:-1] > thr)) & (dev[1:] > thr)) + 1
            if rising.size >= 2:
                print(f"\n✓ Keyphasor: ~{60.0/(np.median(np.diff(rising))/args.fs):.0f} rpm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
