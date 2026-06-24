#!/usr/bin/env python3
"""
wm_modbus_collector_sgt300b.py — Watermelon System
==================================================

Colector en vivo para **Parex · Turbogenerador SGT300 B** vía Modbus TCP
desde el gateway del Bently 3500, y push a Supabase `live_readings` (la
misma tabla que alimenta Live Monitoring de TES1/TES3).

Arquitectura:
    Bently 3500  →  Gateway Modbus TCP (192.168.1.228:502)  →  ESTE script
                 →  Supabase live_readings  →  watermelonsystem.app (Live)

Corre en bucle (cada POLL_SECONDS) en una máquina que ALCANCE el gateway
(la misma idea que la Tarea Programada de TES3). Idealmente como Tarea
Programada de Windows o servicio.

Requisitos:
    pip install pymodbus==3.* supabase

Credenciales (env, las MISMAS de la app):
    SUPABASE_URL, SUPABASE_SERVICE_KEY

Uso:
    # 1) Validar lectura SIN escribir nada (recomendado primero):
    python wm_modbus_collector_sgt300b.py --dry-run --once

    # 2) Correr en vivo:
    export SUPABASE_URL=...; export SUPABASE_SERVICE_KEY=...
    python wm_modbus_collector_sgt300b.py
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from datetime import datetime, timezone

# =============================================================
# CONFIG — AJUSTAR
# =============================================================
GATEWAY_IP   = "192.168.1.228"   # gateway Modbus del Bently 3500 (SGT300B)
GATEWAY_PORT = 502
UNIT_ID      = 1                  # slave/unit id Modbus (confirmar; suele ser 1)
POLL_SECONDS = 10                 # cada cuánto leer y empujar

# instance_id del activo en Watermelon (verlo en la URL/Live Monitoring del
# SGT300B). DEBE coincidir con el de la instancia ya creada en la app.
INSTANCE_ID  = "sgt300b"         # ← CONFIRMAR/AJUSTAR

# Decodificación de float de 32 bits (2 registros). Bently suele ser big/big
# (ABCD). Si los valores salen absurdos, probá "little" en WORD_ORDER (CDAB).
WORD_ORDER = "big"               # "big" (ABCD) | "little" (CDAB)

# Base de direccionamiento Modbus: el mapa lista direcciones tipo 6005. Si las
# lecturas salen corridas/erróneas, probá REGISTER_OFFSET = -1 (1-based→0-based).
REGISTER_OFFSET = 0

# Unidades por tipo de sensor (confirmar con la config del Bently 3500):
UNIT_PROX  = "µm pp"             # proximidad (D)  — Direct/Gap/1X/2X Ampl
UNIT_VEL   = "mm/s rms"          # velocidad (V)
UNIT_ACCEL = "g rms"            # aceleración (A)
UNIT_PHASE = "deg"
UNIT_BIAS  = "V"
UNIT_RPM   = "rpm"

# =============================================================
# MAPA DE REGISTROS (del cliente) — dirección de inicio del float 32-bit
# Cada entrada: (registro, sensor_label, plane_label, metric, unit, variable)
# metric ∈ {Direct, Gap, 1X_Ampl, 1X_Phase, 2X_Ampl, 2X_Phase, BiasVoltage}
# Velocidad de máquina → sensor_label=None, metric="Direct", unit=rpm.
# =============================================================
def _prox(reg, lbl, plane, base_unit):
    """Genera las 6 métricas de un canal de proximidad consecutivas (paso 2)."""
    return [
        (reg + 0,  lbl, plane, "Direct",   base_unit,  f"{lbl} {plane}"),
        (reg + 2,  lbl, plane, "Gap",      base_unit,  f"{lbl} {plane}"),
        (reg + 4,  lbl, plane, "1X_Ampl",  base_unit,  f"{lbl} {plane}"),
        (reg + 6,  lbl, plane, "1X_Phase", UNIT_PHASE, f"{lbl} {plane}"),
        (reg + 8,  lbl, plane, "2X_Ampl",  base_unit,  f"{lbl} {plane}"),
        (reg + 10, lbl, plane, "2X_Phase", UNIT_PHASE, f"{lbl} {plane}"),
    ]


REGISTER_MAP = []
# Velocidades (rpm)
REGISTER_MAP += [
    (6001, None, "", "Direct", UNIT_RPM, "Velocidad Turbina"),
    (6003, None, "", "Direct", UNIT_RPM, "Velocidad Generador"),
]
# Proximidad turbina + gearbox + generador (Direct/Gap/1X/2X)
REGISTER_MAP += _prox(6005, "1YD", "Turbina DE",  UNIT_PROX)
REGISTER_MAP += _prox(6017, "1XD", "Turbina DE",  UNIT_PROX)
REGISTER_MAP += _prox(6029, "2YD", "Turbina NDE", UNIT_PROX)
REGISTER_MAP += _prox(6041, "2XD", "Turbina NDE", UNIT_PROX)
REGISTER_MAP += _prox(6085, "4YD", "Gearbox",     UNIT_PROX)
REGISTER_MAP += _prox(6097, "4XD", "Gearbox",     UNIT_PROX)
REGISTER_MAP += _prox(6133, "5YD", "GEN DE",      UNIT_PROX)
REGISTER_MAP += _prox(6145, "5XD", "GEN DE",      UNIT_PROX)
REGISTER_MAP += _prox(6157, "6YD", "GEN NDE",     UNIT_PROX)
REGISTER_MAP += _prox(6169, "6XD", "GEN NDE",     UNIT_PROX)
# Velocidad / aceleración gearbox (Direct + BiasVoltage)
REGISTER_MAP += [
    (6061, "3YV", "Gearbox", "Direct",      UNIT_VEL,   "3YV VEL Gearbox"),
    (6063, "3YV", "Gearbox", "BiasVoltage", UNIT_BIAS,  "3YV VEL Gearbox"),
    (6065, "3YA", "Gearbox", "Direct",      UNIT_ACCEL, "3YA ACCEL Gearbox"),
    (6067, "3YA", "Gearbox", "BiasVoltage", UNIT_BIAS,  "3YA ACCEL Gearbox"),
    (6069, "4YV", "Gearbox", "Direct",      UNIT_VEL,   "4YV VEL bomba"),
    (6071, "4YV", "Gearbox", "BiasVoltage", UNIT_BIAS,  "4YV VEL bomba"),
    (6073, "4YA", "Gearbox", "Direct",      UNIT_ACCEL, "4YA ACCEL bomba"),
    (6075, "4YA", "Gearbox", "BiasVoltage", UNIT_BIAS,  "4YA ACCEL bomba"),
    (6077, "4XV", "Gearbox", "Direct",      UNIT_VEL,   "4XV VEL starter"),
    (6079, "4XV", "Gearbox", "BiasVoltage", UNIT_BIAS,  "4XV VEL starter"),
    (6081, "4XA", "Gearbox", "Direct",      UNIT_ACCEL, "4XA ACCEL starter"),
    (6083, "4XA", "Gearbox", "BiasVoltage", UNIT_BIAS,  "4XA ACCEL starter"),
]


# =============================================================
# Modbus helpers
# =============================================================
def _regs_to_float(hi: int, lo: int) -> float:
    """Dos registros de 16 bits → float IEEE-754 de 32 bits."""
    if WORD_ORDER == "little":
        hi, lo = lo, hi
    return struct.unpack(">f", struct.pack(">HH", hi, lo))[0]


def _read_block(client, start: int, end: int) -> dict:
    """Lee holding registers [start..end] en trozos ≤120 y devuelve {addr: val}."""
    out = {}
    addr = start
    while addr <= end:
        count = min(120, end - addr + 1)
        # pymodbus 3.13+ usa device_id; versiones previas usan slave.
        try:
            rr = client.read_holding_registers(addr + REGISTER_OFFSET,
                                               count=count, device_id=UNIT_ID)
        except TypeError:
            rr = client.read_holding_registers(addr + REGISTER_OFFSET,
                                               count=count, slave=UNIT_ID)
        if rr.isError():
            raise IOError(f"Modbus error leyendo {addr}..{addr+count-1}: {rr}")
        for i, v in enumerate(rr.registers):
            out[addr + i] = v
        addr += count
    return out


def _build_rows(regs: dict) -> list:
    """Mapa de registros → filas para live_readings."""
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for reg, lbl, plane, metric, unit, variable in REGISTER_MAP:
        if reg not in regs or (reg + 1) not in regs:
            continue
        try:
            val = _regs_to_float(regs[reg], regs[reg + 1])
        except Exception:
            continue
        if val != val:  # NaN
            continue
        rows.append({
            "instance_id":  INSTANCE_ID,
            "sensor_label": lbl,
            "variable":     variable,
            "metric":       metric,
            "value":        round(float(val), 4),
            "unit":         unit,
            "captured_at":  now_iso,
            "quality":      "good",
        })
    return rows


# =============================================================
# Supabase push
# =============================================================
def _get_supabase():
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        print("[FATAL] Falta SUPABASE_URL / SUPABASE_SERVICE_KEY en el entorno.",
              file=sys.stderr)
        sys.exit(2)
    from supabase import create_client
    return create_client(url, key)


def _push(sb, rows: list) -> None:
    if not rows:
        return
    sb.table("live_readings").insert(rows).execute()


# =============================================================
# Main
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Lee y muestra; NO escribe en Supabase.")
    ap.add_argument("--once", action="store_true",
                    help="Una sola lectura y termina (para validar).")
    args = ap.parse_args()

    from pymodbus.client import ModbusTcpClient
    reg_addrs = [r[0] for r in REGISTER_MAP]
    lo_block, hi_block = min(reg_addrs), max(reg_addrs) + 1

    sb = None if args.dry_run else _get_supabase()

    while True:
        t0 = time.time()
        client = ModbusTcpClient(GATEWAY_IP, port=GATEWAY_PORT, timeout=5)
        try:
            if not client.connect():
                raise IOError(f"No conecta al gateway {GATEWAY_IP}:{GATEWAY_PORT}")
            regs = _read_block(client, lo_block, hi_block)
            rows = _build_rows(regs)
            if args.dry_run:
                print(f"\n=== {datetime.now():%H:%M:%S} · {len(rows)} lecturas ===")
                for r in rows:
                    lbl = r["sensor_label"] or r["variable"]
                    print(f"  {lbl:18s} {r['metric']:12s} = {r['value']:>10} {r['unit']}")
            else:
                _push(sb, rows)
                print(f"[{datetime.now():%H:%M:%S}] OK · {len(rows)} lecturas → live_readings")
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] ERROR · {e}", file=sys.stderr)
        finally:
            client.close()

        if args.once:
            break
        time.sleep(max(1.0, POLL_SECONDS - (time.time() - t0)))


if __name__ == "__main__":
    main()
