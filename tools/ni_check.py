"""
tools/ni_check.py — PRUEBA DE ORO paso 1: diagnóstico del hardware NI
=====================================================================

Corre en el PC de sitio (Windows) con NI-DAQmx + `pip install nidaqmx`.
NO usa IEPE (lee como VOLTAJE) → seguro para proximidad. Verifica, antes de
abrir la app:
  1) Qué chasis/módulos ve el driver (nombre real en NI MAX).
  2) Que las 5 señales llegan (keyphasor + 4 radiales), con stats por canal.
  3) Que el keyphasor pulsa (estimación de rpm).

Uso:
    python tools/ni_check.py                 # autodetecta módulos 9234
    python tools/ni_check.py --mod1 cDAQ1Mod1 --mod2 cDAQ1Mod2
    python tools/ni_check.py --fs 25600 --secs 2

Cableado esperado (tu setup):
    Mod1/ai0        = keyphasor
    Mod2/ai0..ai3   = 4 radiales (proximidad, salida BNC del proximitor)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod1", default=None, help="Módulo del keyphasor (ej. cDAQ1Mod1)")
    ap.add_argument("--mod2", default=None, help="Módulo de los radiales (ej. cDAQ1Mod2)")
    ap.add_argument("--fs", type=float, default=25600.0)
    ap.add_argument("--secs", type=float, default=2.0)
    args = ap.parse_args()

    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    except ImportError:
        print("✗ nidaqmx no está instalado. En el PC de sitio:\n"
              "    1) Instalá NI-DAQmx (driver) desde ni.com\n"
              "    2) pip install nidaqmx")
        return 2

    # ---- 1) Inventario de dispositivos ----
    sysm = nidaqmx.system.System.local()
    devs = list(sysm.devices)
    print("=== Dispositivos NI detectados ===")
    if not devs:
        print("✗ Ninguno. ¿Está el cDAQ conectado por USB y encendido? Revisá NI MAX.")
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
    if not mod1 or not mod2:
        print("\n✗ No pude autodetectar 2 módulos 9234. Pasalos a mano:\n"
              "    python tools/ni_check.py --mod1 <nombre> --mod2 <nombre>\n"
              "  (usá los nombres que aparecen arriba, ej. cDAQ1Mod1)")
        return 4
    print(f"\nUsando: keyphasor={mod1}/ai0 · radiales={mod2}/ai0..ai3")

    phys = [f"{mod1}/ai0"] + [f"{mod2}/ai{i}" for i in range(4)]
    names = ["KPH", "R1", "R2", "R3", "R4"]

    # ---- 2) Lectura de prueba (VOLTAJE, sin IEPE = seguro para proximidad) ----
    n = int(args.fs * args.secs)
    print(f"\nLeyendo {args.secs:.0f} s a {args.fs:.0f} Hz (voltaje, IEPE OFF)…")
    task = nidaqmx.Task()
    try:
        for p in phys:
            task.ai_channels.add_ai_voltage_chan(
                p, min_val=-5.0, max_val=5.0,
                terminal_config=TerminalConfiguration.DEFAULT)  # 9234: pseudodiff, sin IEPE
        task.timing.cfg_samp_clk_timing(rate=args.fs, sample_mode=AcquisitionType.FINITE,
                                        samps_per_chan=n)
        data = np.asarray(task.read(number_of_samples_per_channel=n, timeout=args.secs + 10),
                          dtype=float)
    except Exception as e:  # noqa: BLE001
        print(f"✗ Error leyendo: {type(e).__name__}: {e}")
        print("  Chequeá: nombres de módulos, que ai0..ai3 existan, y el cableado BNC.")
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

    print("\n=== Señales (Volts) ===")
    print(f"{'canal':6s} {'min':>9s} {'max':>9s} {'rms(AC)':>9s} {'pp':>9s}")
    for i, nm in enumerate(names):
        x = data[i]
        ac = x - np.mean(x)
        print(f"{nm:6s} {x.min():9.4f} {x.max():9.4f} {np.sqrt(np.mean(ac**2)):9.4f} {np.ptp(x):9.4f}")

    # ---- 3) Keyphasor → rpm ----
    kph = data[0]
    dev = np.abs(kph - np.median(kph))
    peak = float(dev.max())
    if peak < 1e-3:
        print("\n⚠ Keyphasor plano (sin pulso). ¿Está conectado en Mod1/ai0? "
              "Ojo: en el 9234 (AC-coupled) el pulso se atenúa — es esperable.")
    else:
        thr = 0.5 * peak
        active = dev > thr
        rising = np.flatnonzero((~active[:-1]) & (active[1:])) + 1
        if rising.size >= 2:
            rev_s = np.median(np.diff(rising)) / args.fs
            print(f"\n✓ Keyphasor pulsa: ~{60.0/rev_s:.0f} rpm ({rising.size} pulsos en {args.secs:.0f} s).")
        else:
            print(f"\n⚠ Keyphasor con actividad pero <2 pulsos claros ({rising.size}).")

    print("\n✓ Diagnóstico OK. Si las 4 radiales muestran rms(AC) > 0 y el keyphasor pulsa, "
          "estás listo para abrir la app (Fuente = Campo, máquina Rotor_Kit_SIGA_1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
