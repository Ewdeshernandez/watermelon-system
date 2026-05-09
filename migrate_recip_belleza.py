"""
migrate_recip_belleza.py
========================

Retro-actualiza el activo "COMPR RECIP LA BELLEZA" (creado antes de v3.31.14)
con driver_icon_key + driven_icon_key + coupling_class.

Se infiere automáticamente desde driver_model + driven_model. Si el script
no encuentra el activo con el id esperado, lista todas las instancias
para que vos confirmes el slug real.
"""
from __future__ import annotations
import sys
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    from core.instance_state import (
        list_instances, get_instance, update_instance_header, _slugify,
    )

    print("=" * 60)
    print("  MIGRACIÓN ICONOS — COMPR RECIP LA BELLEZA")
    print("=" * 60)

    # Buscar la instancia. Probamos varios slugs típicos.
    candidates = [
        "c200c",   # tag CR200C - COMPR RECIP LA BELLEZA
        "compr_recip_la_belleza",
        "compr_recip_la_bellza",
        "comp_recip_la_belleza",
        "recip_la_belleza",
        "la_belleza",
    ]
    inst = None
    inst_id_real = None
    for c in candidates:
        i = get_instance(c)
        if i is not None:
            inst = i
            inst_id_real = c
            break

    if inst is None:
        # Listar todas para que el user vea
        print("\nNo encontré la instancia con los slugs típicos. Esto es lo que hay:")
        for s in list_instances():
            iid = s.get("instance_id", "?")
            tag = s.get("tag", "")
            print(f"  - {iid}   (tag: {tag})")
        print("\nEditá el script y agregá tu slug a 'candidates', después corré de nuevo.")
        return 1

    print(f"\nEncontrada: {inst_id_real}")
    print(f"  driver_model:    '{inst.driver_model}'")
    print(f"  driven_model:    '{inst.driven_model}'")
    print(f"  coupling_class:  '{inst.coupling_class}'")
    print(f"  driver_icon_key: '{inst.driver_icon_key}'")
    print(f"  driven_icon_key: '{inst.driven_icon_key}'")

    # Inferir iconos: motor eléctrico Hyundai HNP + Ariel KBK/4
    # HNP de Hyundai usa rodamientos rolling
    target_drv = "electric_motor_rolling"
    target_drvn = "recip_compressor_boxer_4cyl"
    target_coupling = "rigid"  # Ariel KBK/4 acopla cigüeñal con rígido al motor
    target_tag = "C200C"  # corrección de typo CR200C -> C200C

    needs_update = (
        inst.driver_icon_key != target_drv
        or inst.driven_icon_key != target_drvn
        or inst.coupling_class != target_coupling
        or inst.tag != target_tag
    )
    if not needs_update:
        print("\n✓ Ya está al día. Nada que actualizar.")
        return 0

    print(f"\nVa a quedar:")
    print(f"  driver_icon_key  = '{target_drv}'   (Hyundai HNP es rolling-bearing)")
    print(f"  driven_icon_key  = '{target_drvn}'  (Ariel KBK/4 = boxer 4 cilindros)")
    print(f"  coupling_class   = '{target_coupling}'")
    print(f"  tag              = '{target_tag}'                       (era '{inst.tag}')")

    resp = input("\n¿Aplicar? (si/no): ").strip().lower()
    if resp not in ("si", "s", "sí", "yes", "y"):
        print("Cancelado.")
        return 0

    ok = update_instance_header(
        instance_id=inst_id_real,
        driver_icon_key=target_drv,
        driven_icon_key=target_drvn,
        coupling_class=target_coupling,
        tag=target_tag,
    )
    if not ok:
        log.error("update_instance_header devolvió False")
        return 2

    inst2 = get_instance(inst_id_real)
    print(f"\n✓ Actualizado. Estado actual:")
    print(f"  driver_icon_key:  '{inst2.driver_icon_key}'")
    print(f"  driven_icon_key:  '{inst2.driven_icon_key}'")
    print(f"  coupling_class:   '{inst2.coupling_class}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
