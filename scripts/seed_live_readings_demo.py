#!/usr/bin/env python3
"""
seed_live_readings_demo.py — Watermelon System (Ciclo 23.133)
=============================================================

Poblar la tabla `live_readings` con lecturas realistas para los activos
de DEMO (TES1 + TES3 — Ecopetrol-Magnex). Útil cuando la tabla quedó
vacía (limpieza manual, collector caído, sesión Supabase reset) y
necesitamos mostrar Live Monitoring "viva" al cliente.

Qué inserta:
  - 8 sensores por activo (matchean el sensor_map del setup actual):
      Lado libre (CRF):
        VE5807 (Y)  — proximity Y desplazamiento  [mil pp]
        VE5808 (X)  — proximity X desplazamiento  [mil pp]
        1VT6831 (C) CRF — velocity                 [in/s peak]
        CRF ACEL    — acceleration                 [g peak]
      Lado coupling (TRF):
        VE5809 (Y)  — proximity Y desplazamiento
        VE5810 (X)  — proximity X desplazamiento
        1VT6805 (C) TRF — velocity
        TRF ACEL    — acceleration
  - 4 métricas por sensor: Direct, 0.5X_Ampl, 1X_Ampl, 2X_Ampl
  - 1 registro de Velocidad de máquina (rpm = 3600)
  - captured_at = NOW() (fresca, no stale)
  - ingested_at = NOW()
  - quality = "good"

Uso:
    # 1. Setea creds (mismo que la app)
    export SUPABASE_URL="https://xxxxx.supabase.co"
    export SUPABASE_SERVICE_KEY="eyJxxx..."

    # 2. Run
    python scripts/seed_live_readings_demo.py

    # Opcional: solo un activo
    python scripts/seed_live_readings_demo.py --only tes1

    # Opcional: limpiar antes de insertar
    python scripts/seed_live_readings_demo.py --clean
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ciclo 23.134 — fix shadow del package `supabase` por la carpeta ./supabase
# en la raíz del repo (que contiene SQL migrations). Si corremos desde la
# raíz, sys.path[0]='' (cwd) y Python encuentra ./supabase ANTES que el
# package real. Eliminamos esa entrada del sys.path antes del import.
_CWD_LIKE = {"", os.getcwd(), str(Path(__file__).resolve().parent.parent)}
sys.path = [p for p in sys.path if p not in _CWD_LIKE]


# =============================================================
# Configuración del seed
# =============================================================

INSTANCES = ["tes1", "tes3"]

# Sensores por activo. Definidos como (sensor_label, variable, unit, family)
# family se usa para generar valores realistas en cada métrica.
SENSORS = [
    # CRF lado libre (Drive End → DE)
    ("VE5807 (Y)",        "VE5807 CRF Y",     "mil pp",    "proximity"),
    ("VE5808 (X)",        "VE5808 CRF X",     "mil pp",    "proximity"),
    ("1VT6831 (C) CRF",   "VT6831 CRF VEL",   "in/s peak", "velocity"),
    ("CRF ACEL",          "CRF ACEL",         "g peak",    "acceleration"),
    # TRF lado coupling (Non-Drive End → NDE)
    ("VE5809 (Y)",        "VE5809 TRF Y",     "mil pp",    "proximity"),
    ("VE5810 (X)",        "VE5810 TRF X",     "mil pp",    "proximity"),
    ("1VT6805 (C) TRF",   "VT6805 TRF VEL",   "in/s peak", "velocity"),
    ("TRF ACEL",          "TRF ACEL",         "g peak",    "acceleration"),
]

# Rangos realistas por familia (un poco aleatorios para que se vea "vivo")
# Direct = overall RMS o peak según unidad. 1X domina típicamente.
RANGES = {
    "proximity":    {"direct": (0.6, 2.1), "p1x": (0.7, 1.8),  "p2x": (0.1, 0.35), "p05x": (0.01, 0.05)},
    "velocity":     {"direct": (0.05, 0.5), "p1x": (0.02, 0.4),"p2x": (0.005, 0.08),"p05x": (0.002, 0.015)},
    "acceleration": {"direct": (0.5, 3.5), "p1x": (0.1, 0.3),  "p2x": (0.05, 0.18),"p05x": (0.01, 0.04)},
}

DEFAULT_RPM = 3600.0


# =============================================================
# Supabase client
# =============================================================

def get_client():
    """Crea client Supabase con service_role. Falla con mensaje claro."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )

    # Fallback a Streamlit secrets si la app lo usa
    if not url or not key:
        try:
            import streamlit as st
            cfg = st.secrets.get("supabase", {})
            url = url or str(cfg.get("url", "")).strip()
            key = key or str(cfg.get("service_key", "")).strip()
        except Exception:
            pass

    if not url or not key:
        print(
            "ERROR: faltan credenciales. Setea SUPABASE_URL + SUPABASE_SERVICE_KEY "
            "(o configura .streamlit/secrets.toml con [supabase] url/service_key).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        print(f"ERROR creando Supabase client: {e}", file=sys.stderr)
        sys.exit(1)


# =============================================================
# Builders de rows
# =============================================================

def _v(low: float, high: float) -> float:
    return round(random.uniform(low, high), 4)


def build_rows_for_instance(instance_id: str) -> list[dict]:
    """Genera todas las rows (sensores × metrics + speed) para un activo."""
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []

    # Velocidad de máquina (variable Velocidad Generador) → metric Direct, rpm
    rows.append({
        "instance_id":  instance_id,
        "sensor_label": None,
        "variable":     "Velocidad Generador",
        "metric":       "Direct",
        "value":        DEFAULT_RPM + random.uniform(-2, 2),
        "unit":         "rpm",
        "captured_at":  now_iso,
        "quality":      "good",
        "metadata":     {"source": "seed_demo"},
    })

    for sensor_label, variable, unit, family in SENSORS:
        r = RANGES[family]

        # Direct (overall)
        rows.append({
            "instance_id":  instance_id,
            "sensor_label": sensor_label,
            "variable":     variable,
            "metric":       "Direct",
            "value":        _v(*r["direct"]),
            "unit":         unit,
            "captured_at":  now_iso,
            "quality":      "good",
            "metadata":     {"source": "seed_demo", "family": family},
        })
        # 0.5X
        rows.append({
            "instance_id":  instance_id,
            "sensor_label": sensor_label,
            "variable":     variable,
            "metric":       "0.5X_Ampl",
            "value":        _v(*r["p05x"]),
            "unit":         unit,
            "captured_at":  now_iso,
            "quality":      "good",
            "metadata":     {"source": "seed_demo", "family": family},
        })
        # 1X
        rows.append({
            "instance_id":  instance_id,
            "sensor_label": sensor_label,
            "variable":     variable,
            "metric":       "1X_Ampl",
            "value":        _v(*r["p1x"]),
            "unit":         unit,
            "captured_at":  now_iso,
            "quality":      "good",
            "metadata":     {"source": "seed_demo", "family": family},
        })
        # 2X
        rows.append({
            "instance_id":  instance_id,
            "sensor_label": sensor_label,
            "variable":     variable,
            "metric":       "2X_Ampl",
            "value":        _v(*r["p2x"]),
            "unit":         unit,
            "captured_at":  now_iso,
            "quality":      "good",
            "metadata":     {"source": "seed_demo", "family": family},
        })

    return rows


# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--only", default="",
        help="Sólo este instance_id (ej. 'tes1'). Default: todos los configurados."
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="DELETE FROM live_readings WHERE instance_id=... antes de insertar. "
             "Útil si querés evitar acumular filas demo."
    )
    args = parser.parse_args()

    targets = [args.only] if args.only else list(INSTANCES)
    targets = [t.strip().lower() for t in targets if t.strip()]
    if not targets:
        print("ERROR: nada que insertar.", file=sys.stderr)
        sys.exit(2)

    client = get_client()
    print(f"✓ Supabase client OK · target instances: {targets}")

    total_inserted = 0
    for iid in targets:
        if args.clean:
            try:
                resp = client.table("live_readings").delete().eq("instance_id", iid).execute()
                print(f"  [{iid}] cleaned previous rows")
            except Exception as e:
                print(f"  [{iid}] WARN no se pudo limpiar: {e}", file=sys.stderr)

        rows = build_rows_for_instance(iid)
        try:
            resp = client.table("live_readings").insert(rows).execute()
            n = len(getattr(resp, "data", []) or [])
            print(f"  [{iid}] {n} rows insertadas (esperadas {len(rows)})")
            total_inserted += n
        except Exception as e:
            print(f"  [{iid}] ERROR insertando: {e}", file=sys.stderr)

    print(f"\n✓ Total: {total_inserted} rows. Abre Live Monitoring y refresca — "
          "deberías ver las lecturas en cada activo.")


if __name__ == "__main__":
    main()
