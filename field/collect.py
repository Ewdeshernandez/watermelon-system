"""
field/collect.py — COLECTOR de campo (headless, sin navegador)
==============================================================

Adquiere de las tarjetas NI y GRABA la onda cruda a disco EN CONTINUO. Como no
hay navegador ni gráficos, **NO se traba nunca** y **no pierde datos**. Ctrl+C
para terminar; intenta subir a Supabase. Después, en la app:
    Análisis → 📼 Reprocesar → elegí esta grabación → ves TODOS los gráficos.

Ejemplos (tu setup: 2 acelerómetros 100 mV/g en Mod1 canales 1 y 2):
    python field/collect.py
    python field/collect.py --sens 100 --fs 5120
    python field/collect.py --mod cDAQ1Mod1 --chans 0,1 --names 1YA,1XA
    python field/collect.py --prox      # proximidad (IEPE OFF, cuando tengas el 9229)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# permitir importar core/ desde la carpeta del repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="Rotor_Kit_Accel", help="Nombre de la máquina (tag)")
    ap.add_argument("--mod", default="cDAQ1Mod1", help="Módulo NI de los sensores")
    ap.add_argument("--chans", default="0,1", help="Índices ai (ej. 0,1)")
    ap.add_argument("--names", default="1YA,1XA", help="Nombres de los canales")
    ap.add_argument("--sens", type=float, default=100.0, help="Sensibilidad mV/g (accel) o mV/mil (prox)")
    ap.add_argument("--fs", type=float, default=5120.0, help="Muestreo (Hz)")
    ap.add_argument("--block", type=float, default=0.25, help="Segundos por bloque")
    ap.add_argument("--prox", action="store_true", help="Proximidad (voltaje, IEPE OFF)")
    ap.add_argument("--kph-mod", default=None, help="Módulo del keyphasor (opcional, ej. cDAQ1Mod2)")
    ap.add_argument("--kph-ai", type=int, default=0, help="Canal ai del keyphasor")
    args = ap.parse_args()

    try:
        import nidaqmx
        from nidaqmx.constants import (AcquisitionType, ExcitationSource, AccelUnits,
                                       TerminalConfiguration)
    except ImportError:
        print("✗ nidaqmx no está instalado (pip install nidaqmx).")
        return 2

    from core.remote_monitoring.recorder import TransientRecorder, upload_recording, free_bytes

    if free_bytes() and free_bytes() < 200 * 1024 * 1024:
        print(f"⚠ Poco espacio en disco ({free_bytes()/1e6:.0f} MB). Liberá antes de grabar.")
        return 3

    idxs = [int(x) for x in args.chans.split(",") if x.strip() != ""]
    names = [s.strip() for s in args.names.split(",")]
    iepe = not args.prox
    unit = "g rms" if iepe else "mil pp"

    # Canales: sensores + (opcional) keyphasor al final
    phys, ch_names = [], []
    for k, i in enumerate(idxs):
        phys.append(f"{args.mod}/ai{i}")
        ch_names.append(names[k] if k < len(names) else f"CH{i}")
    kph_present = args.kph_mod is not None
    if kph_present:
        phys.append(f"{args.kph_mod}/ai{args.kph_ai}")
        ch_names.append("KPH")

    ch_meta = [{"name": ch_names[k], "units": ("pulses/rev" if (kph_present and k == len(phys) - 1) else unit),
                "sensitivity_mv_per_eu": (1.0 if (kph_present and k == len(phys) - 1) else args.sens),
                "coupling": ("DC" if (kph_present and k == len(phys) - 1) else ("IEPE" if iepe else "AC"))}
               for k in range(len(phys))]

    rec = TransientRecorder(args.machine, args.fs, ch_meta, machine=args.machine)

    task = nidaqmx.Task()
    try:
        for k, p in enumerate(phys):
            is_kph = kph_present and k == len(phys) - 1
            if iepe and not is_kph:
                task.ai_channels.add_ai_accel_chan(
                    p, sensitivity=args.sens, units=AccelUnits.G,
                    current_excit_source=ExcitationSource.INTERNAL, current_excit_val=0.002)
            else:
                task.ai_channels.add_ai_voltage_chan(
                    p, min_val=-5.0, max_val=5.0, terminal_config=TerminalConfiguration.DEFAULT)
        nblk = int(args.fs * args.block)
        task.timing.cfg_samp_clk_timing(rate=args.fs, sample_mode=AcquisitionType.CONTINUOUS,
                                        samps_per_chan=nblk * 8)
        task.start()
    except Exception as e:  # noqa: BLE001
        print(f"✗ No se pudo abrir la tarea NI: {type(e).__name__}: {e}")
        try:
            task.close()
        except Exception:  # noqa: BLE001
            pass
        rec.stop()
        return 4

    print("=" * 64)
    print(f"  GRABANDO  ·  {args.machine}")
    print(f"  {len(idxs)} sensor(es) {'IEPE/accel' if iepe else 'voltaje/prox'} en {args.mod} "
          f"ai{args.chans}  ·  {args.fs:.0f} Hz")
    print("  Poné el rotor a girar. Golpeá un sensor y vas a ver saltar el nivel.")
    print("  >>> Ctrl+C para TERMINAR y guardar <<<")
    print("=" * 64 + "\n")

    try:
        while True:
            data = np.asarray(task.read(number_of_samples_per_channel=int(args.fs * args.block),
                                        timeout=5.0), dtype=float)
            if data.ndim == 1:
                data = data[None, :]
            rec.append(data.astype(np.float32))
            rms = []
            for k in range(len(idxs)):        # nivel de los sensores (no del kph)
                x = data[k]
                rms.append(float(np.sqrt(np.mean((x - x.mean()) ** 2))))
            lvl = "  ".join(f"{ch_names[k]}={rms[k]:.4f}" for k in range(len(idxs)))
            print(f"\r  {rec.status.duration_s:6.1f}s  {rec.status.size_mb:6.1f} MB   {lvl}      ",
                  end="", flush=True)
            if not rec.open:                  # disco lleno u error
                print(f"\n⚠ Grabación detenida: {rec.error}")
                break
    except KeyboardInterrupt:
        print("\n\n  Deteniendo…")
    finally:
        try:
            task.close()
        except Exception:  # noqa: BLE001
            pass
        rec.stop()

    print(f"\n✓ Grabación **{rec.rec_id}** · {rec.status.duration_s:.0f} s · {rec.status.size_mb:.1f} MB")
    up = upload_recording(rec.dir)
    if up.get("ok"):
        print("  ☁ Subida a Supabase.")
    else:
        print(f"  Guardada local (pendiente de subir: {up.get('reason')}).")
    print("\n  Abrí la app → Análisis → 📼 Reprocesar → elegí esta grabación → ves los gráficos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
