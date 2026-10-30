"""
tools/torsional_ni_check.py — Primera luz de la NI 9229 (Torsional)
===================================================================

Diagnóstico de HARDWARE para correr en el PC de CAMPO (Windows + NI-DAQmx + la
NI 9229 conectada). Verifica, de punta a punta, que `NITorsionalSource` levanta
la tarjeta y que llegan señales reales de Par (AI0) y Keyphasor (AI1).

Uso (desde la raíz del repo WatermelonSystem):

  # 1) Listar las tarjetas NI conectadas (para saber el nombre real del módulo)
  python tools\\torsional_ni_check.py --list

  # 2) Leer 2 s del 9229 autodetectado, keyphasor = foto-tacómetro (1 cinta)
  python tools\\torsional_ni_check.py --seconds 2 --kph phototach --ppr 1

  # 3) Con nombre de módulo explícito y keyphasor de proximidad Bently
  python tools\\torsional_ni_check.py --device cDAQ1Mod1 --kph proximitor --ppr 1

No cae a simulado: si la tarjeta no está, imprime el error de DAQmx y termina.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Permite ejecutar el script desde cualquier cwd: agrega la raíz del repo.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.torsional.ni_source import (  # noqa: E402
    KeyphasorSensor, NITorsionalConfig, NITorsionalSource,
    find_ni_voltage_module, nidaqmx_available, rpm_from_keyphasor,
)


def _list_devices() -> int:
    try:
        from core.modal.acq_backend import list_available_devices
        devs = list_available_devices()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] No se pudieron listar devices: {exc}")
        return 2
    if not devs:
        print("[WARN] No hay devices NI conectados (o el driver no los ve).")
        return 1
    print(f"Devices NI detectados ({len(devs)}):")
    for d in devs:
        print(f"  - {d.get('name'):12s}  {d.get('product_type'):14s}  SN {d.get('serial')}")
    g = find_ni_voltage_module()
    print(f"\nMódulo de voltaje (9229/9239/9215) sugerido: {g or '—ninguno—'}")
    return 0 if g else 1


def _make_sensor(kind: str, ppr: int) -> KeyphasorSensor:
    if kind == "proximitor":
        return KeyphasorSensor.bently_3300xl_8mm(keyways=ppr)
    if kind == "sim":
        return KeyphasorSensor.simulated()
    return KeyphasorSensor.phototach_reflective(strips=ppr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Primera luz NI 9229 (Torsional)")
    ap.add_argument("--list", action="store_true", help="Listar devices NI y salir")
    ap.add_argument("--device", default="", help="Nombre del módulo 9229 (auto si se omite)")
    ap.add_argument("--rate", type=float, default=2560.0, help="fs pedida [Hz] (default 2560)")
    ap.add_argument("--seconds", type=float, default=2.0, help="Segundos a leer (default 2)")
    ap.add_argument("--block", type=float, default=0.1, help="Tamaño de bloque [s] (default 0.1)")
    ap.add_argument("--vrange", type=float, default=10.0, help="±Rango de entrada [V] (default 10)")
    ap.add_argument("--torque-ai", type=int, default=0, help="Índice AI del par (default 0)")
    ap.add_argument("--kph-ai", type=int, default=1, help="Índice AI del keyphasor (default 1)")
    ap.add_argument("--kph", choices=["phototach", "proximitor", "sim"], default="phototach",
                    help="Tipo de sensor de keyphasor (default phototach)")
    ap.add_argument("--ppr", type=int, default=1, help="Pulsos por vuelta (cintas/keyways)")
    args = ap.parse_args()

    if args.list:
        return _list_devices()

    if not nidaqmx_available():
        print("[FAIL] NI-DAQmx no está instalado en este equipo.")
        print("       Corré esto en el PC de campo (Windows) con el driver NI-DAQmx")
        print("       y la NI 9229 conectada. (En Mac/dev no hay hardware.)")
        return 2

    device = args.device or find_ni_voltage_module()
    if not device:
        print("[FAIL] No se encontró un módulo 9229/9239/9215. Probá --list.")
        return 2

    sensor = _make_sensor(args.kph, max(1, args.ppr))
    cfg = NITorsionalConfig(
        device=device, torque_ai=args.torque_ai, kph_ai=args.kph_ai,
        sample_rate_hz=args.rate, block_seconds=args.block,
        voltage_range=args.vrange, keyphasor=sensor)

    print(f"Abriendo {device}: Par=AI{cfg.torque_ai}  Keyphasor=AI{cfg.kph_ai}  "
          f"fs~{cfg.sample_rate_hz:.0f} Hz  ±{cfg.voltage_range:.0f} V")
    print(f"Keyphasor: {sensor.label} · {sensor.edge} · thr {sensor.trigger_level_v:+.2f} V · "
          f"{sensor.pulses_per_rev} ppr")

    src = NITorsionalSource(cfg)
    try:
        src.start()
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAIL] start() falló:\n  {exc}")
        return 2

    fs = src.actual_rate_hz
    if abs(fs - cfg.sample_rate_hz) > 1.0:
        print(f"[nota] El 9229 coerció la tasa: pedida {cfg.sample_rate_hz:.1f} → real {fs:.3f} Hz")
    nblocks = max(1, int(round(args.seconds / cfg.block_seconds)))
    tq, kp = [], []
    try:
        for _ in range(nblocks):
            blk = src.read_block()                 # (2, block) en Volts
            tq.append(blk[0]); kp.append(blk[1])
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAIL] read_block() falló:\n  {exc}")
        src.stop()
        return 2
    finally:
        src.stop()

    tq = np.concatenate(tq); kp = np.concatenate(kp)
    n = tq.size
    print(f"\nLeídas {n} muestras/canal ({n / fs:.2f} s @ {fs:.1f} Hz).\n")

    def _report(name: str, x: np.ndarray) -> None:
        pp = float(np.nanmax(x) - np.nanmin(x))
        print(f"  {name:10s}  min {np.nanmin(x):+8.4f}  max {np.nanmax(x):+8.4f}  "
              f"mean {np.nanmean(x):+8.4f}  std {np.nanstd(x):7.4f}  p-p {pp:7.4f} V")
        if np.nanmax(np.abs(x)) > 0.98 * cfg.voltage_range:
            print(f"    [WARN] {name}: cerca de saturar ±{cfg.voltage_range:.0f} V → subí el rango.")
        if pp < 1e-3:
            print(f"    [WARN] {name}: señal plana (<1 mV p-p) → ¿cable/alimentación?")

    print("Canales (Volts):")
    _report("Par AI0", tq)
    _report("KPH AI1", kp)

    # rpm del keyphasor
    try:
        rpm, _t = rpm_from_keyphasor(kp, fs, sensor)
        fin = rpm[np.isfinite(rpm)]
        if fin.size:
            print(f"\nKeyphasor → rpm: media {np.nanmean(fin):.1f}  "
                  f"min {np.nanmin(fin):.1f}  max {np.nanmax(fin):.1f}  "
                  f"({fin.size} estimaciones)")
        else:
            print("\n[WARN] No se detectaron pulsos de keyphasor → revisá tipo de sensor "
                  "(--kph), umbral/flanco, o cableado a AI1.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[WARN] No se pudo calcular rpm: {exc}")

    print("\n[OK] Primera luz NI 9229 completa. Usá este nombre de módulo en la app: "
          f"'{device}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
