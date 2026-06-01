"""
collector/wm_simulator.py
=========================

Simulador de datos en vivo para Watermelon System.

A diferencia de wm_collector.py (que lee Modbus real del Bently en planta),
este script NO necesita gateway ni red industrial: genera lecturas
realistas a partir del MISMO modbus_map y las POSTea al API. Sirve para:

  * Validar el pipeline API -> Supabase -> app sin hardware.
  * Demos a clientes con un activo "vivo".
  * Probar el Live Monitoring localmente.

Uso:

    pip install requests
    python wm_simulator.py \
        --map ../data/modbus_maps/tes1.json \
        --api-url https://watermelon-api-bpv4.onrender.com \
        --api-key TU_API_KEY \
        --interval 10

    # Una sola ráfaga (validar y salir):
    python wm_simulator.py --map ../data/modbus_maps/tes1.json \
        --api-url https://watermelon-api-bpv4.onrender.com \
        --api-key TU_API_KEY --once
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    print("Falta dependencia: pip install requests", file=sys.stderr)
    sys.exit(1)


# Valores base realistas por tipo de métrica/unidad. El simulador parte
# de estos centros y les suma ruido + un drift lento (seno) para que las
# sparklines y trends se vean vivos.
def _base_value(reg: Dict[str, Any]) -> float:
    kind = (reg.get("kind") or "").lower()
    metric = (reg.get("metric") or "Direct").lower()
    unit = (reg.get("unit") or "").lower()

    if kind == "speed" or "rpm" in unit:
        return 3600.0
    if "in/s" in unit:           # velocidad vibración
        return 0.12
    if unit.strip() == "g pk" or "g pk" in unit:  # aceleración
        return 0.45
    if "mil pp" in unit:         # desplazamiento proximity
        if "1x_ampl" in metric:
            return 1.8
        if "2x_ampl" in metric:
            return 0.6
        return 2.4               # Direct gap-band
    if "v dc" in unit:           # gap / bias voltage
        return -9.5 if "gap" in metric else -12.0
    if "deg" in unit or "phase" in metric:
        return 90.0              # fase 1X/2X
    return 1.0


def _noisy(base: float, reg: Dict[str, Any], t: float) -> float:
    """base + drift lento + ruido. Determinístico-ish por registro."""
    metric = (reg.get("metric") or "").lower()
    unit = (reg.get("unit") or "").lower()

    # Fases: ruido chico en grados, sin drift de escala
    if "deg" in unit or "phase" in metric:
        return round((base + random.uniform(-4, 4)) % 360.0, 1)

    # Velocidad de giro: muy estable
    if "rpm" in unit:
        return round(base + random.uniform(-8, 8), 1)

    # Vibración / desplazamiento: drift senoidal lento + ruido
    drift = 1.0 + 0.06 * math.sin(t / 90.0 + base)
    noise = random.uniform(-0.04, 0.05) * base
    val = base * drift + noise
    # Voltajes negativos (gap/bias) mantienen signo
    if base < 0:
        return round(val, 3)
    return round(max(val, 0.0), 4)


def load_registers(map_path: str) -> (str, List[Dict[str, Any]]):
    raw = json.loads(Path(map_path).read_text(encoding="utf-8"))
    instance_id = raw.get("instance_id") or "unknown"
    regs: List[Dict[str, Any]] = []
    for addr, spec in (raw.get("registers") or {}).items():
        regs.append({
            "address": int(addr),
            "variable": spec.get("variable", ""),
            "metric": spec.get("metric", "Direct"),
            "unit": spec.get("unit"),
            "sensor_label": spec.get("sensor_label"),
            "kind": spec.get("kind", "sensor"),
        })
    return instance_id, regs


def build_payload(instance_id: str, regs: List[Dict[str, Any]], t: float) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    readings = []
    for r in regs:
        base = _base_value(r)
        val = _noisy(base, r, t)
        readings.append({
            "variable": r["variable"],
            "metric": r["metric"],
            "value": val,
            "unit": r["unit"],
            "sensor_label": r.get("sensor_label"),
            "register": r["address"],
            "quality": "good",
        })
    return {
        "instance_id": instance_id,
        "captured_at": now.isoformat(),
        "metadata": {"collector_version": "sim-1.0.0", "host": "SIMULATOR"},
        "readings": readings,
    }


def post_batch(api_url: str, api_key: str, payload: Dict[str, Any]) -> bool:
    url = api_url.rstrip("/") + "/v1/ingest/live"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        if resp.status_code == 200:
            j = resp.json()
            print(f"  POST OK · ingested={j.get('ingested')} received={j.get('received')}")
            return True
        print(f"  POST status={resp.status_code} body={resp.text[:300]}")
        return False
    except requests.RequestException as e:
        print(f"  POST failed: {e}")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Watermelon live data SIMULATOR")
    p.add_argument("--map", required=True, help="Path al modbus_map JSON (ej. data/modbus_maps/tes1.json)")
    p.add_argument("--api-url", default="https://watermelon-api-bpv4.onrender.com")
    p.add_argument("--api-key", required=True, help="API key (Bearer) del servicio watermelon-api")
    p.add_argument("--interval", type=int, default=10, help="Segundos entre batches (default 10)")
    p.add_argument("--once", action="store_true", help="Manda 1 batch y sale")
    args = p.parse_args()

    instance_id, regs = load_registers(args.map)
    print(f"=== wm_simulator · instance={instance_id} · {len(regs)} variables ===")
    print(f"=== API: {args.api_url} ===")

    t0 = time.time()
    try:
        while True:
            payload = build_payload(instance_id, regs, time.time() - t0)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] enviando {len(payload['readings'])} lecturas...")
            post_batch(args.api_url, args.api_key, payload)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nDetenido por usuario.")


if __name__ == "__main__":
    main()
