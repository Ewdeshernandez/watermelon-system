"""
borrar_activos_extras.py
========================

Script one-shot para borrar las instancias de prueba que sobraron en
Machinery Library (tipicamente "default" y "probamosuerte" y similares).

Modo de uso:
    python borrar_activos_extras.py

Por seguridad:
  - Primero LISTA todas las instancias del backend activo (Supabase).
  - Pide confirmación interactiva antes de borrar cada una que matchea.
  - Solo borra las que coincidan con la lista TARGETS (editable abajo).
  - NO toca tes1 ni nada que no esté en TARGETS.

Idempotente: si una de las TARGETS ya no existe, lo reporta y sigue.
"""

from __future__ import annotations

import sys
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Editá esta lista si querés borrar otras instancias.
# Los matches son tolerantes a mayúsculas/espacios (vía slugify).
TARGETS = [
    "default",
    "brush_default",     # encontrado en Supabase con tag "(default)"
    "probamosuerte",
    "probemosuerte",
    "probemos_suerte",
    "probamos_suerte",
]

PROTECTED = ["tes1"]  # nunca tocar


def main() -> int:
    from core.instance_state import (
        list_instances, get_instance, delete_instance, _slugify,
    )

    print("=" * 60)
    print("  BORRAR ACTIVOS EXTRAS — Machinery Library")
    print("=" * 60)

    # 1. Listar lo que hay
    instances = list_instances()
    if not instances:
        log.warning("No hay instancias en el backend activo.")
        return 0

    print(f"\n{len(instances)} instancias encontradas:")
    for inst_summary in instances:
        iid = inst_summary.get("instance_id", "?")
        tag = inst_summary.get("tag", "")
        protected_marker = " 🔒 PROTEGIDA" if iid in PROTECTED else ""
        print(f"  - {iid}  (tag: {tag}){protected_marker}")

    # 2. Decidir cuáles borrar
    targets_slugified = {_slugify(t) for t in TARGETS}
    existing_ids = {i.get("instance_id") for i in instances}
    to_delete = [iid for iid in existing_ids
                 if iid in targets_slugified and iid not in PROTECTED]

    if not to_delete:
        print("\n✓ No hay nada para borrar — la lista TARGETS no matchea ninguna instancia.")
        if any("default" in (i.get("instance_id","") or "").lower() for i in instances):
            print("  Hint: hay una instancia con 'default' en el id. Verificá el slug exacto arriba.")
        return 0

    print(f"\n{len(to_delete)} instancias serán BORRADAS (no se puede deshacer):")
    for iid in to_delete:
        print(f"  ✗ {iid}")

    # 3. Confirmación interactiva
    resp = input("\n¿Confirmás el borrado? (si/no): ").strip().lower()
    if resp not in ("si", "s", "sí", "yes", "y"):
        print("Cancelado. Nada se borró.")
        return 0

    # 4. Borrar uno por uno con verificación
    deleted = 0
    for iid in to_delete:
        if iid in PROTECTED:
            log.warning(f"Skip {iid} — protegida.")
            continue
        try:
            ok = delete_instance(iid)
            if ok:
                # Verificar
                if get_instance(iid) is None:
                    print(f"  ✓ {iid} borrada.")
                    deleted += 1
                else:
                    log.error(f"  ⚠ {iid}: delete_instance devolvió True pero la instancia "
                              f"sigue existiendo.")
            else:
                log.error(f"  ⚠ {iid}: delete_instance devolvió False.")
        except Exception as e:
            log.error(f"  ⚠ {iid}: error al borrar — {e}")

    print(f"\n{'=' * 60}")
    print(f"  RESULTADO: {deleted}/{len(to_delete)} instancias borradas")
    print(f"{'=' * 60}")

    return 0 if deleted == len(to_delete) else 3


if __name__ == "__main__":
    sys.exit(main())
