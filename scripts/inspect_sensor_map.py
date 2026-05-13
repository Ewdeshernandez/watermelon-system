#!/usr/bin/env python3
"""
inspect_sensor_map.py — Watermelon (Ciclo 23.140)
=================================================

Muestra el sensor_map configurado para un activo + los sensores que
tienen lecturas live activas. Útil para diagnosticar overlaps,
sensores duplicados o anchors mal configurados en el SVG.

Uso:
    python scripts/inspect_sensor_map.py tes1
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Workaround shadow ./supabase
_ORIGINAL_CWD = os.getcwd()
_SHADOW_DIR = Path(_ORIGINAL_CWD) / "supabase"
if _SHADOW_DIR.exists():
    os.chdir(tempfile.gettempdir())
sys.path.insert(0, _ORIGINAL_CWD)


def main():
    iid = sys.argv[1] if len(sys.argv) > 1 else "tes1"

    # Restaurar cwd para imports relativos
    try:
        os.chdir(_ORIGINAL_CWD)
    except Exception:
        pass

    from core.instance_state import get_instance

    inst = get_instance(iid)
    if inst is None:
        print(f"ERROR: instance '{iid}' no encontrado")
        sys.exit(1)

    print("=" * 80)
    print(f"INSTANCE: {iid}")
    print(f"  tag={inst.tag}  driver={inst.driver_model}  driven={inst.driven_model}")
    print(f"  driver_icon_key={getattr(inst, 'driver_icon_key', '?')}")
    print(f"  driven_icon_key={getattr(inst, 'driven_icon_key', '?')}")
    print("=" * 80)

    sensors = inst.sensors or []
    print(f"\nSENSOR MAP ({len(sensors)} sensores):\n")
    print(f"  {'#':<3} {'plane_label':<25} {'sensor_type':<14} {'icon_side':<10} {'icon_anchor':<12} {'point':<20}")
    print(f"  {'-'*3} {'-'*25} {'-'*14} {'-'*10} {'-'*12} {'-'*20}")
    for i, s in enumerate(sensors, 1):
        plbl = str(s.get("plane_label") or "(sin label)")[:25]
        stype = str(s.get("sensor_type") or "?")[:14]
        side = str(s.get("icon_side") or "?")[:10]
        anchor = str(s.get("icon_anchor") or "?")[:12]
        point = str(s.get("point") or "?")[:20]
        print(f"  {i:<3} {plbl:<25} {stype:<14} {side:<10} {anchor:<12} {point:<20}")

    # Detección de overlaps
    print("\n" + "=" * 80)
    print("OVERLAP CHECK — sensores que comparten (side, anchor):")
    print("=" * 80)
    from collections import defaultdict
    groups = defaultdict(list)
    for s in sensors:
        key = (s.get("icon_side"), s.get("icon_anchor"))
        if key[0] and key[1]:
            groups[key].append(s.get("plane_label", "?"))
    found_overlap = False
    for (side, anchor), labels in groups.items():
        if len(labels) >= 2:
            found_overlap = True
            print(f"\n  {side} · {anchor}: {len(labels)} sensores")
            for lbl in labels:
                print(f"    · {lbl}")
    if not found_overlap:
        print("\n  (ninguno — cada sensor en su propio anchor)")


if __name__ == "__main__":
    main()
