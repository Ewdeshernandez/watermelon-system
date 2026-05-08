"""
tools/sync_instance_from_modbus.py
==================================

Sincroniza `instance.sensors` con la realidad del rack 3500 (Modbus map).

Problema que resuelve:
    Cuando se crea un activo con el wizard, los sensor labels que genera
    (1_RAD_V, 2_RAD_V, etc.) no necesariamente coinciden con los que
    realmente reporta el rack via Modbus (1Y_V, 2Y_V, 3X_D, 4X_D...).
    Esto hace que la schematic muestre sensores fantasmas que nunca
    aparecen en las readings live.

Este script lee `data/modbus_maps/<instance>.json` y reescribe
`instance.sensors` para que cada sensor del map exista como sensor del
activo, con:
    - sensor_label canónico (1Y_V, 3X_D, etc.)
    - plane, plane_label, direction, sensor_type, unit_native
    - x_pct, y_pct heurísticos basados en el plano (distribución
      horizontal sobre el schematic)
    - alarm, danger según asset class detectada (aero turbine OEM defaults
      para LM6000, etc.)
    - csv_match_pattern del map

Uso:
    python tools/sync_instance_from_modbus.py tes1
    python tools/sync_instance_from_modbus.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.instance_state import get_instance, save_instance, list_instances  # noqa: E402
from core.severity import detect_asset_class, thresholds_for, family_from  # noqa: E402


# Posiciones heurísticas (x_pct, y_pct) sobre el schematic horizontal típico
# del tren acoplado: driver a la izquierda, driven a la derecha.
#
# Ajustado para schematic horizontal estándar de tren coupled:
#   plane 1 (driver DE)   → ~25%
#   plane 2 (driver NDE)  → ~40%
#   plane 3 (driven DE)   → ~62%
#   plane 4 (driven NDE)  → ~82%
#
# Para X/Y direction: X queda arriba del eje, Y abajo (convención visual).
PLANE_X = {1: 26.0, 2: 41.0, 3: 62.0, 4: 82.0}


def _y_for(direction: str, sensor_type: str) -> float:
    d = (direction or "").upper()
    t = (sensor_type or "").lower()
    if t == "keyphasor":
        return 50.0
    if d == "X":
        return 38.0  # arriba del eje
    if d == "Y":
        return 62.0  # debajo del eje
    if d in ("RADIAL", "RAD"):
        return 35.0  # arriba sobre carcasa
    if d in ("AXIAL", "AX"):
        return 50.0
    return 50.0


def _x_for(plane: int, sensor_type: str, plane_label: str = "") -> float:
    pl = (plane_label or "").lower()
    # Casos especiales por plane_label
    if "crf" in pl:
        return 22.0
    if "trf" in pl:
        return 36.0
    if "gen" in pl and ("nde" in pl):
        return 82.0
    if "gen" in pl and ("de" in pl):
        return 62.0
    return PLANE_X.get(int(plane), 50.0)


def build_sensors_from_modbus(modbus_map_data: Dict[str, Any], instance_obj) -> List[Dict[str, Any]]:
    """
    Agrupa los registers por sensor_label y construye un sensor por cada
    label único, con metadata heredada del primer register de ese label.
    """
    registers = modbus_map_data.get("registers", {}) or {}
    if isinstance(registers, list):
        # Soportar formato lista (algunos maps lo tienen así)
        registers = {str(r.get("address")): r for r in registers if r.get("address") is not None}

    by_label: Dict[str, Dict[str, Any]] = {}
    for addr, spec in registers.items():
        label = (spec.get("sensor_label") or "").strip()
        if not label:
            # Skip registers sin label (ej. velocidad de máquina)
            continue
        kind = (spec.get("kind") or "").lower()
        # Solo el "Direct" es el sensor "principal" del cual heredamos type/unit/threshold.
        # Los registers con kind='vector' o 'diagnostic' son metadata adicional del mismo sensor.
        if label not in by_label:
            by_label[label] = {
                "_addr_first": int(addr),
                "plane": int(spec.get("plane", 0) or 0),
                "plane_label": str(spec.get("location") or spec.get("plane_label") or ""),
                "direction": str(spec.get("direction", "") or ""),
                "sensor_type": str(spec.get("sensor_type", "") or ""),
                "unit_native": str(spec.get("unit", "") or ""),
                "csv_match_pattern": str(spec.get("csv_match_pattern", "") or ""),
            }
        else:
            # Heredamos plane/direction/type si todavía no estaban
            slot = by_label[label]
            for k in ("plane", "direction", "sensor_type"):
                if not slot.get(k) and spec.get(k):
                    slot[k] = spec[k]

    # Construir lista final con thresholds + posiciones
    out: List[Dict[str, Any]] = []
    for label, meta in by_label.items():
        family = family_from(meta["sensor_type"], meta["unit_native"])
        alarm, danger, src = thresholds_for(family, meta["unit_native"], instance_obj)

        # Side y angle heuristic
        d = (meta["direction"] or "").upper()
        if d == "X":
            side, angle = "R", 45.0
        elif d == "Y":
            side, angle = "L", 45.0
        elif d in ("RAD", "RADIAL"):
            side, angle = "top", 0.0
        else:
            side, angle = "—", 0.0

        out.append({
            "plane": meta["plane"],
            "plane_label": meta["plane_label"],
            "side": side,
            "angle_deg": angle,
            "direction": d if d in ("X", "Y") else (meta["direction"] or "").lower(),
            "sensor_type": meta["sensor_type"],
            "unit_native": meta["unit_native"],
            "alarm": alarm,
            "danger": danger,
            "csv_match_pattern": meta["csv_match_pattern"],
            "notes": f"Auto-sync desde Modbus map (asset class: {src})",
            "x_pct": _x_for(meta["plane"], meta["sensor_type"], meta["plane_label"]),
            "y_pct": _y_for(meta["direction"], meta["sensor_type"]),
        })

    # Ordenar: por plane, luego por direction X/Y, luego por sensor_type
    type_order = {"proximity": 0, "velocity": 1, "accelerometer": 2, "keyphasor": 3}
    out.sort(key=lambda s: (
        s["plane"],
        {"X": 0, "Y": 1, "radial": 2, "axial": 3}.get(s["direction"], 9),
        type_order.get(s["sensor_type"], 9),
    ))
    return out


def sync_instance(instance_id: str, dry_run: bool = False) -> bool:
    """Devuelve True si el sync se aplicó (o se simuló)."""
    map_path = PROJECT_ROOT / "data" / "modbus_maps" / f"{instance_id}.json"
    if not map_path.exists():
        print(f"[!] No existe Modbus map: {map_path}")
        return False

    inst = get_instance(instance_id)
    if inst is None:
        print(f"[!] No existe la instancia '{instance_id}' en data/instances/")
        return False

    map_data = json.loads(map_path.read_text(encoding="utf-8"))
    new_sensors = build_sensors_from_modbus(map_data, inst)

    print(f"\n=== Sync {instance_id} (asset class detectada: {detect_asset_class(inst)}) ===")
    print(f"Sensores anteriores: {len(inst.sensors or [])}")
    print(f"Sensores nuevos:     {len(new_sensors)}")
    print()
    print(f"{'Label':<10} {'Plane':<22} {'Dir':<6} {'Type':<14} {'Unit':<10} "
          f"{'Alarm':>8} {'Danger':>8} {'x%':>5} {'y%':>5}")
    print("-" * 105)
    try:
        from core.sensor_map import sensor_label as _lbl
    except Exception:
        _lbl = lambda s: f"P{s.get('plane','?')}"
    for s in new_sensors:
        label = _lbl(s)
        print(
            f"{label:<10} {s.get('plane_label',''):<22} {s.get('direction',''):<6} "
            f"{s.get('sensor_type',''):<14} {s.get('unit_native',''):<10} "
            f"{s.get('alarm',0):>8.2f} {s.get('danger',0):>8.2f} "
            f"{s.get('x_pct',0):>5.1f} {s.get('y_pct',0):>5.1f}"
        )

    if dry_run:
        print("\n[DRY RUN] No se guardó nada. Quitá --dry-run para aplicar.")
        return True

    inst.sensors = new_sensors
    save_instance(inst)
    print(f"\n✓ Sensores sincronizados y guardados en data/instances/{instance_id}/")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("instance_id", nargs="?", help="Instance id a sincronizar (ej. tes1)")
    p.add_argument("--all", action="store_true", help="Sincronizar todos los activos con map")
    p.add_argument("--dry-run", action="store_true", help="No guardar, solo mostrar")
    args = p.parse_args()

    targets: List[str] = []
    if args.all:
        for f in (PROJECT_ROOT / "data" / "modbus_maps").glob("*.json"):
            targets.append(f.stem)
    elif args.instance_id:
        targets = [args.instance_id]
    else:
        p.error("Especificá un instance_id o usá --all")

    if not targets:
        print("Sin Modbus maps en data/modbus_maps/")
        return

    for t in targets:
        sync_instance(t, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
