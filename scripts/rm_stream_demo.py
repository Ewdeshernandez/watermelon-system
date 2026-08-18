#!/usr/bin/env python3
"""
scripts/rm_stream_demo.py — Demo del streaming Remote Monitoring (SIMULADO)
===========================================================================

Corre en el Mac SIN hardware. Simula un rotor a RPM fija con desbalance,
llena el buffer rodante, materializa una ventana a Signals y verifica que
el espectro muestra el pico 1X. Es el "hola mundo" del módulo antes de
conectar el NI de campo.

Uso:
    python scripts/rm_stream_demo.py
    python scripts/rm_stream_demo.py --rpm 3000 --defect misalignment --seconds 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.modal.acq_backend import ChannelConfig
from core.remote_monitoring import (
    RingBuffer,
    SimulatedStreamSource,
    StreamConfig,
    window_to_signals,
)


def _demo_channels():
    """Un par de proximidad X/Y (para órbita) + keyphasor."""
    return [
        ChannelConfig(name="1Y", coupling="AC", sensitivity_mv_per_eu=200.0, bnc_port=1, units="mil"),
        ChannelConfig(name="1X", coupling="AC", sensitivity_mv_per_eu=200.0, bnc_port=2, units="mil"),
        ChannelConfig(name="2Y", coupling="AC", sensitivity_mv_per_eu=200.0, bnc_port=3, units="mil"),
        ChannelConfig(name="2X", coupling="AC", sensitivity_mv_per_eu=200.0, bnc_port=4, units="mil"),
        ChannelConfig(name="KPH", coupling="DC", sensitivity_mv_per_eu=1.0, bnc_port=5, units="V"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=float, default=5120)
    ap.add_argument("--rpm", type=float, default=3600)
    ap.add_argument("--defect", default="unbalance", choices=["none", "unbalance", "misalignment"])
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--block", type=float, default=0.25)
    args = ap.parse_args()

    f1 = args.rpm / 60.0
    cfg = StreamConfig(
        sample_rate_hz=args.fs,
        channels=_demo_channels(),
        block_seconds=args.block,
        buffer_seconds=max(4.0, args.seconds + 1),
        rpm=args.rpm,
        defect=args.defect,
    )
    src = SimulatedStreamSource(cfg)
    rb = RingBuffer(cfg.n_channels, cfg.buffer_samples)

    n_blocks = int(round(args.seconds / args.block))
    print(f"▶ Streaming simulado: {cfg.n_channels} canales @ {args.fs:.0f} Hz | "
          f"rpm={args.rpm:.0f} (1X={f1:.1f} Hz) | defecto={args.defect}")
    print(f"  bloque={args.block}s ({cfg.block_samples} samples) × {n_blocks} bloques\n")

    src.start()
    for i in range(n_blocks):
        rb.write(src.read_block())
    src.stop()
    print(f"✓ Buffer lleno: {rb.filled} samples/canal ({rb.filled/args.fs:.2f}s)\n")

    signals = window_to_signals(rb.snapshot(), cfg.channels, args.fs, rpm=args.rpm)
    print(f"✓ Materializadas {len(signals)} señales de vibración (keyphasor excluido)\n")

    print(f"{'Sensor':<8}{'Unidad':<8}{'RMS':>10}{'Pico 1X (Hz)':>16}{'Ampl 1X':>12}")
    print("-" * 54)
    ok = True
    for sig in signals:
        x = sig.x - np.mean(sig.x)
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / (len(x) / 2)
        freqs = np.fft.rfftfreq(len(x), 1.0 / args.fs)
        peak_i = int(np.argmax(spec))
        peak_f = freqs[peak_i]
        rms = float(np.sqrt(np.mean(x ** 2)))
        label = sig.metadata["sensor_label"]
        units = sig.metadata.get("units", "")
        flag = "" if abs(peak_f - f1) < 2.0 else "  ⚠ no-1X"
        print(f"{label:<8}{units:<8}{rms:>10.4f}{peak_f:>16.2f}{spec[peak_i]:>12.4f}{flag}")
        ok = ok and abs(peak_f - f1) < 2.0

    print("\n" + ("✅ Pico dominante = 1X en todos los canales — pipeline OK"
                  if ok else "⚠ Algún canal no domina en 1X (revisar)"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
