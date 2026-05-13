#!/usr/bin/env python3
"""
verify_live_readings.py — Watermelon (Ciclo 23.135)
====================================================

Diagnóstico rápido: lee la tabla live_readings y el view
latest_live_reading desde Supabase para confirmar qué rows
hay realmente para TES1 / TES3 / cualquier instance.

Uso:
    python scripts/verify_live_readings.py
    python scripts/verify_live_readings.py tes1
    python scripts/verify_live_readings.py tes1 tes3
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


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["tes1", "tes3"]
    client = get_client()

    print("=" * 70)
    print("CONTEO en tabla `live_readings`")
    print("=" * 70)
    for iid in targets:
        try:
            resp = (
                client.table("live_readings")
                .select("id", count="exact")
                .eq("instance_id", iid)
                .limit(1)
                .execute()
            )
            n = getattr(resp, "count", None) or 0
            print(f"  {iid}: {n} rows totales en live_readings")
        except Exception as e:
            print(f"  {iid}: ERROR — {e}")

    print()
    print("=" * 70)
    print("View `latest_live_reading` (lo que ve Live Monitoring)")
    print("=" * 70)
    for iid in targets:
        try:
            resp = (
                client.table("latest_live_reading")
                .select("instance_id, sensor_label, variable, metric, value, unit, captured_at")
                .eq("instance_id", iid)
                .order("variable")
                .execute()
            )
            data = list(getattr(resp, "data", []) or [])
            print(f"  {iid}: {len(data)} rows en latest_live_reading")
            for r in data[:6]:
                lbl = r.get("sensor_label") or "(speed)"
                val = r.get("value")
                unit = r.get("unit") or ""
                ts = r.get("captured_at") or ""
                metric = r.get("metric") or ""
                print(f"    · {lbl:25s} {metric:12s} {val} {unit}  @ {ts}")
            if len(data) > 6:
                print(f"    · ... +{len(data) - 6} más")
        except Exception as e:
            print(f"  {iid}: ERROR — {e}")

    print()
    print("Si live_readings tiene rows pero latest_live_reading no,")
    print("hay un problema con el view (probablemente RLS o staleness).")
    print("Si AMBOS están vacíos, el seed no insertó nada.")


if __name__ == "__main__":
    main()
