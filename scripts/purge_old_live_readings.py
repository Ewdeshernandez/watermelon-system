#!/usr/bin/env python3
"""
purge_old_live_readings.py — Watermelon (Ciclo 23.136)
=======================================================

Borra filas viejas de `live_readings` en batches pequeños para evitar
el timeout de Supabase Free tier (DELETE masivo > 30s falla con
57014 'canceling statement').

Por qué: la view `latest_live_reading` hace DISTINCT ON sobre toda la
tabla. Si una instance tiene millones de filas, la view hace timeout y
Live Monitoring se queda en blanco. Solución: mantener solo las últimas
N horas de data (ej. 24h).

Uso:
    # Default: borra todo lo que sea > 24h en TES1 y TES3
    python scripts/purge_old_live_readings.py

    # Solo TES1, todo lo > 12h
    python scripts/purge_old_live_readings.py --only tes1 --hours 12

    # Borrar TODO de un instance (deja la tabla limpia para que seed_demo
    # genere data fresca):
    python scripts/purge_old_live_readings.py --only tes1 --all

    # Batch size custom (default 1000):
    python scripts/purge_old_live_readings.py --batch 500
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Workaround shadow ./supabase
_ORIGINAL_CWD = os.getcwd()
_SHADOW_DIR = Path(_ORIGINAL_CWD) / "supabase"
if _SHADOW_DIR.exists():
    os.chdir(tempfile.gettempdir())
    _script_dir = str(Path(__file__).resolve().parent)
    sys.path = [p for p in sys.path if p != _script_dir and p != _ORIGINAL_CWD]


def get_client():
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )
    if not url or not key:
        try:
            import streamlit as st
            cfg = st.secrets.get("supabase", {})
            url = url or str(cfg.get("url", "")).strip()
            key = key or str(cfg.get("service_key", "")).strip()
        except Exception:
            pass
    if not url or not key:
        print("ERROR: faltan SUPABASE_URL / SUPABASE_SERVICE_KEY", file=sys.stderr)
        sys.exit(1)
    from supabase import create_client
    try:
        os.chdir(_ORIGINAL_CWD)
    except Exception:
        pass
    return create_client(url, key)


def purge_instance(client, instance_id: str, *, hours: int, batch: int,
                   delete_all: bool) -> int:
    """Borra filas en batches. Returns total borrado."""
    print(f"\n[{instance_id}] iniciando purge…")

    # 1. Conteo inicial
    try:
        resp = (
            client.table("live_readings")
            .select("id", count="exact")
            .eq("instance_id", instance_id)
            .limit(1)
            .execute()
        )
        n_before = getattr(resp, "count", None) or 0
        print(f"  rows actuales: {n_before:,}")
    except Exception as e:
        print(f"  WARN conteo inicial falló: {e}")
        n_before = -1

    if n_before == 0:
        print(f"  [{instance_id}] ya está vacío, nada que hacer")
        return 0

    cutoff_iso = None
    if not delete_all:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff.isoformat(timespec="seconds")
        print(f"  cutoff: borrar todo con captured_at < {cutoff_iso}")
    else:
        print(f"  modo --all: borrar TODO de este instance")

    total_deleted = 0
    consecutive_empty = 0
    max_iterations = 5000  # safety guard (5M filas max)

    for it in range(max_iterations):
        # Obtener IDs de un batch (más rápido que delete por filter complejo)
        try:
            q = (
                client.table("live_readings")
                .select("id")
                .eq("instance_id", instance_id)
                .order("id")
                .limit(batch)
            )
            if cutoff_iso:
                q = q.lt("captured_at", cutoff_iso)
            resp = q.execute()
            ids = [r["id"] for r in (getattr(resp, "data", []) or [])]
        except Exception as e:
            print(f"  [iter {it}] WARN select falló: {e}")
            time.sleep(2)
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print(f"  [iter {it}] 3 fallos consecutivos, abortando")
                break
            continue

        if not ids:
            print(f"  [iter {it}] no más rows que borrar — done")
            break

        # Delete por ID (rápido, usa el PK index)
        try:
            client.table("live_readings").delete().in_("id", ids).execute()
            total_deleted += len(ids)
            consecutive_empty = 0
            if it % 10 == 0 or len(ids) < batch:
                print(f"  [iter {it}] borradas {total_deleted:,} acumuladas")
        except Exception as e:
            print(f"  [iter {it}] WARN delete falló (len={len(ids)}): {e}")
            time.sleep(2)
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print(f"  [iter {it}] 3 fallos consecutivos, abortando")
                break

    print(f"\n  [{instance_id}] ✓ total borrado: {total_deleted:,}")
    return total_deleted


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--only", default="",
                   help="solo este instance_id (ej. tes1)")
    p.add_argument("--hours", type=int, default=24,
                   help="borrar rows con captured_at más viejas que N horas (default 24)")
    p.add_argument("--all", action="store_true",
                   help="ignorar --hours y borrar TODO del instance (peligroso)")
    p.add_argument("--batch", type=int, default=1000,
                   help="tamaño de batch (default 1000). Si timeoutea, bajar a 500 o 200")
    args = p.parse_args()

    targets = [args.only] if args.only else ["tes1", "tes3"]
    targets = [t.strip().lower() for t in targets if t.strip()]

    client = get_client()
    print(f"✓ Supabase client OK · targets: {targets}")

    total = 0
    for iid in targets:
        total += purge_instance(
            client, iid,
            hours=args.hours, batch=args.batch, delete_all=args.all,
        )

    print(f"\n══════ FIN — total filas borradas: {total:,} ══════")


if __name__ == "__main__":
    main()
