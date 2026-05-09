"""
migrate_tes1_icons.py
=====================

Script one-shot para retro-actualizar la instancia TES1 (creada antes de
v3.31.14) con los nuevos campos del Ciclo 23.13:

  - driver_icon_key  → "gas_turbine_aero"     (GE LM6000 PD/PG)
  - driven_icon_key  → "generator_synchronous" (Brush BDAX 7-290ER)
  - coupling_class   → "flexible"              (disc-pack — estándar
                                                OEM aero-derivative)

Sin esto, Live Monitoring no sabe qué iconos rendir y cae al PNG 3D legacy.

Uso (desde la raíz del repo, con el venv del proyecto activado y las env
vars de Supabase configuradas):

    python migrate_tes1_icons.py

Idempotente: si los campos ya están seteados, no hace nada y reporta OK.
No toca sensores, header, ni live_readings.
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    # Import tardío para que los logs de Supabase aparezcan después del banner.
    from core.instance_state import get_instance, update_instance_header

    INSTANCE_ID = "tes1"

    target = {
        "driver_icon_key": "gas_turbine_aero",
        "driven_icon_key": "generator_synchronous",
        "coupling_class": "flexible",
    }

    print("=" * 60)
    print(f"  MIGRACIÓN ICONOS — Instance: {INSTANCE_ID}")
    print("=" * 60)

    inst = get_instance(INSTANCE_ID)
    if inst is None:
        log.error(f"No existe la instance '{INSTANCE_ID}' en el repo activo. "
                  f"¿Estás contra Supabase prod? Verificá WATERMELON_REPO_BACKEND.")
        return 1

    # Estado actual
    print("\nEstado ACTUAL:")
    print(f"  driver_icon_key  = '{inst.driver_icon_key}'")
    print(f"  driven_icon_key  = '{inst.driven_icon_key}'")
    print(f"  coupling_class   = '{inst.coupling_class}'")

    # ¿Hace falta?
    needs_update = (
        inst.driver_icon_key != target["driver_icon_key"]
        or inst.driven_icon_key != target["driven_icon_key"]
        or inst.coupling_class != target["coupling_class"]
    )
    if not needs_update:
        print("\nNo hay nada que actualizar — todos los campos ya están al día.")
        return 0

    # Aplicar
    print("\nNuevo estado:")
    print(f"  driver_icon_key  = '{target['driver_icon_key']}'")
    print(f"  driven_icon_key  = '{target['driven_icon_key']}'")
    print(f"  coupling_class   = '{target['coupling_class']}'")

    print("\nAplicando update...")
    ok = update_instance_header(
        instance_id=INSTANCE_ID,
        driver_icon_key=target["driver_icon_key"],
        driven_icon_key=target["driven_icon_key"],
        coupling_class=target["coupling_class"],
    )
    if not ok:
        log.error("update_instance_header devolvió False — la instance no se "
                  "encontró post-slugify. Revisá los logs de instance_state.")
        return 2

    # Verificar
    inst2 = get_instance(INSTANCE_ID)
    if inst2 is None:
        log.error("Post-update get_instance devuelve None — algo raro pasó.")
        return 3

    print("\nEstado POST-UPDATE:")
    print(f"  driver_icon_key  = '{inst2.driver_icon_key}'")
    print(f"  driven_icon_key  = '{inst2.driven_icon_key}'")
    print(f"  coupling_class   = '{inst2.coupling_class}'")

    if (inst2.driver_icon_key == target["driver_icon_key"]
        and inst2.driven_icon_key == target["driven_icon_key"]
        and inst2.coupling_class == target["coupling_class"]):
        print("\n✓ MIGRACIÓN COMPLETA. TES1 listo para Live Monitoring v3.31.15.")
        return 0
    else:
        log.error("La verificación post-update no coincide con el target.")
        return 4


if __name__ == "__main__":
    sys.exit(main())
