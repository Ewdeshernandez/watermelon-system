"""
tools/delete_placeholder_instances.py
=====================================

Hard-delete de instancias "placeholder" (default) de la DB.

Las instancias con tag "(default)", "default", vacío, o que no tienen
sensores, docs, cliente ni tren son basura visual del sistema. Este
script las borra PERMANENTEMENTE de Supabase para que dejen de
aparecer en cualquier vista.

Uso:
    python tools/delete_placeholder_instances.py            # dry-run (solo lista)
    python tools/delete_placeholder_instances.py --apply    # confirma y borra

Requisitos:
- Variables de entorno SUPABASE_URL + SUPABASE_SERVICE_KEY
  (mismas que usa el resto del sistema)
- O un secrets.toml con [secrets] supabase_url y supabase_service_key

Salida:
    [DRY-RUN] o [APLICADO] lista de instances borradas con razón
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permitir correr desde la raíz del repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _is_placeholder(meta: dict, full=None) -> tuple[bool, str]:
    """Retorna (es_placeholder, razón)."""
    tag = (meta.get("tag") or "").strip()
    if tag in ("", "(default)", "default", "(sin tren)", "(sin nombre)"):
        return True, f"tag literal '{tag or '(vacío)'}'"

    if full is None:
        return False, ""

    no_sensors = not (getattr(full, "sensors", None) or [])
    no_docs = not (getattr(full, "documents", None) or [])
    no_client = not (getattr(full, "client", "") or "").strip()
    no_train = (
        not (getattr(full, "driver_manufacturer", "") or "").strip()
        and not (getattr(full, "driver_model", "") or "").strip()
        and not (getattr(full, "driven_manufacturer", "") or "").strip()
        and not (getattr(full, "driven_model", "") or "").strip()
    )
    if no_sensors and no_docs and no_client and no_train:
        return True, "totalmente vacía (sin sensores/docs/cliente/tren)"

    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Aplicar el borrado real. Sin esta flag solo lista (dry-run).",
    )
    args = parser.parse_args()

    try:
        from core.instance_state import (
            list_instances, get_instance, delete_instance,
        )
    except Exception as exc:
        print(f"ERROR: no se pudo importar core.instance_state: {exc}")
        return 1

    summaries = list_instances() or []
    print(f"Encontradas {len(summaries)} instancias totales en la DB.")
    print()

    candidates = []
    for meta in summaries:
        iid = meta.get("instance_id", "")
        if not iid:
            continue
        try:
            full = get_instance(iid)
        except Exception:
            full = None
        is_ph, reason = _is_placeholder(meta, full)
        if is_ph:
            candidates.append((iid, meta.get("tag", "(?)"), reason))

    if not candidates:
        print("✓ No hay instancias placeholder para borrar. Todo limpio.")
        return 0

    mode = "APLICADO" if args.apply else "DRY-RUN"
    print(f"[{mode}] {len(candidates)} placeholder(s) a borrar:")
    print()
    for iid, tag, reason in candidates:
        print(f"  • {tag:30s}  id={iid[:24]}  razón: {reason}")
    print()

    if not args.apply:
        print("Para borrarlas realmente corré:")
        print("    python tools/delete_placeholder_instances.py --apply")
        return 0

    # Confirmación interactiva
    resp = input(f"¿Confirmás borrar {len(candidates)} instancia(s)? [y/N] ").strip().lower()
    if resp not in ("y", "yes", "s", "si", "sí"):
        print("Cancelado.")
        return 0

    print()
    n_ok = 0
    n_err = 0
    for iid, tag, _reason in candidates:
        try:
            delete_instance(iid)
            print(f"  ✓ borrada: {tag} ({iid[:24]})")
            n_ok += 1
        except Exception as exc:
            print(f"  ✗ FALLO: {tag} ({iid[:24]}) → {exc}")
            n_err += 1

    print()
    print(f"Resultado: {n_ok} borradas, {n_err} fallidas.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
