"""
Watermelon System — helpers para compartir snapshots del Live Monitoring.

Ciclo 23.67 — Fase 2 del export de diagramas.

El share flow es 100% client-side:
  1. JS del browser convierte el SVG → PNG via canvas (zero server load)
  2. JS sube el PNG a Supabase Storage bucket `diagram-shares` con anon_key
  3. JS construye link público y abre wa.me/ o mailto:

Este módulo solo provee la config (URL + anon_key + bucket) para que el
component HTML pueda hacer su trabajo. La anon_key es segura para exponer
al cliente — los policies del bucket están en data/storage_diagram_shares_setup.sql.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


SHARE_BUCKET = "diagram-shares"


def get_storage_share_config() -> Optional[Tuple[str, str, str]]:
    """Devuelve (supabase_url, anon_key, bucket_name) o None si falta config.

    Lee de:
      1. Variables de entorno (SUPABASE_URL, SUPABASE_ANON_KEY)
      2. st.secrets.supabase.{url, anon_key}

    Retorna None si NO encuentra la anon_key — sin ella el cliente no puede
    subir al bucket. service_key NUNCA debe usarse acá: es admin, expondría
    todo si se filtra al browser.
    """
    url = os.environ.get("SUPABASE_URL", "").strip()
    anon_key = os.environ.get("SUPABASE_ANON_KEY", "").strip()

    if not url or not anon_key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "supabase" in st.secrets:
                cfg = st.secrets["supabase"]
                url = url or str(cfg.get("url", "")).strip()
                anon_key = anon_key or str(cfg.get("anon_key", "")).strip()
        except Exception:
            pass

    if not url or not anon_key:
        return None

    return (url, anon_key, SHARE_BUCKET)
