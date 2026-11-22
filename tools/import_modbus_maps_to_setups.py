"""
tools/import_modbus_maps_to_setups.py
=====================================

Migra los mapas Modbus escritos a mano (data/modbus_maps/*.json) a la config
única de máquina en la nube (tabla rm_setups) con ORIGEN Modbus. Desde ahí el
API genera el mapa para el collector (GET /v1/collector/map/{instance_id}).

Verifica round-trip SIN pérdida antes de subir nada. No pisa una máquina que
ya exista con el mismo instance_id salvo --force.

Uso:
    python tools/import_modbus_maps_to_setups.py              # dry-run (no escribe)
    python tools/import_modbus_maps_to_setups.py --apply      # sube
    python tools/import_modbus_maps_to_setups.py sgt300b --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.remote_monitoring.config import (  # noqa: E402
    find_setup_cloud_by_instance,
    save_setup_cloud,
)
from core.remote_monitoring.modbus_binding import (  # noqa: E402
    modbus_map_from_setup,
    setup_from_modbus_map,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="*", help="nombres sin .json (default: todos)")
    ap.add_argument("--apply", action="store_true", help="escribir en la nube")
    ap.add_argument("--force", action="store_true", help="sobrescribir si ya existe")
    ap.add_argument("--online", nargs="*", default=None,
                    help="instance_ids en línea (default: todos los mapas = online)")
    args = ap.parse_args()

    maps_dir = ROOT / "data" / "modbus_maps"
    names = args.maps or sorted(p.stem for p in maps_dir.glob("*.json"))
    rc = 0
    for name in names:
        raw = json.loads((maps_dir / f"{name}.json").read_text(encoding="utf-8"))
        iid = raw.get("instance_id") or name
        mode = "online" if args.online is None or iid in args.online else "spot"
        setup = setup_from_modbus_map(raw, name=name.upper(), mode=mode)
        if modbus_map_from_setup(setup) != raw:
            print(f"[{name}] ✗ round-trip con pérdida — NO se sube")
            rc = 1
            continue
        n_regs = sum(len(c.modbus) for c in setup.channels)
        print(f"[{name}] instance={iid} canales={len(setup.channels)} registros={n_regs} modo={mode}")
        if not args.apply:
            continue
        existing = find_setup_cloud_by_instance(iid)
        if existing and not args.force:
            print(f"   ya existe como '{existing['name']}' (id={existing['id']}) — saltado (usa --force)")
            continue
        res = save_setup_cloud(setup)
        print("   ✓ subido" if res.get("ok") else f"   ✗ {res.get('reason')}")
        rc |= 0 if res.get("ok") else 1
    if not args.apply:
        print("\n(dry-run: nada escrito; usa --apply)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
