"""
core.live_readings
==================

Tier 0 A — Live Data Ingestion (Ciclo 23.1).

Capa de persistencia para lecturas en tiempo real desde wm-collector
(Bently Nevada 3500/92 Modbus + futuros gateways OPC UA / MQTT).

Diseño:
    - Tabla append-only `live_readings` en Supabase (no UPDATE).
    - View `latest_live_reading` para "current values" del dashboard.
    - Cada `LiveReading` es atómico: 1 fila por (variable, metric, captured_at).

Uso típico:

    from core.live_readings import (
        LiveReading,
        ingest_batch,
        latest_for_instance,
        history_for_metric,
    )

    rows = [
        LiveReading(
            instance_id="tes1",
            sensor_label="1Y_V",
            variable="1YV VEL CRF",
            metric="Direct",
            value=0.67,
            unit="in/s pk",
            captured_at=datetime.utcnow(),
            register=6031,
        ),
        ...
    ]
    ingested = ingest_batch(rows)

Si Supabase no está configurado (dev local sin secrets), las funciones
loggean y retornan 0/[] en vez de crashear — comportamiento amigable
con tests y entornos sin red.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("watermelon.live_readings")


# =============================================================================
# Modelo
# =============================================================================

VALID_METRICS = {
    "Direct",        # overall RMS / 0-pk
    "Gap",           # DC gap voltage del proximity probe
    "BiasVoltage",   # health del transducer accel/velocity
    "1X_Ampl",       # vector síncrono primer orden — magnitud
    "1X_Phase",      # vector síncrono primer orden — fase (deg)
    "2X_Ampl",       # segundo orden — magnitud
    "2X_Phase",      # segundo orden — fase
}

VALID_QUALITIES = {"good", "stale", "overrange", "comm_fail"}


@dataclass
class LiveReading:
    """Una lectura puntual de una variable de un activo."""
    instance_id: str
    variable: str
    metric: str
    value: Optional[float]
    captured_at: datetime
    sensor_label: Optional[str] = None
    unit: Optional[str] = None
    register: Optional[int] = None
    quality: str = "good"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metric not in VALID_METRICS:
            log.warning("LiveReading metric '%s' fuera de catálogo: %s", self.metric, VALID_METRICS)
        if self.quality not in VALID_QUALITIES:
            self.quality = "good"
        # Forzar tz-aware UTC
        if self.captured_at.tzinfo is None:
            self.captured_at = self.captured_at.replace(tzinfo=timezone.utc)

    def to_row(self) -> Dict[str, Any]:
        """Serializa para Supabase insert."""
        return {
            "instance_id": self.instance_id,
            "sensor_label": self.sensor_label,
            "variable": self.variable,
            "metric": self.metric,
            "value": self.value if self.value is not None else None,
            "unit": self.unit,
            "captured_at": self.captured_at.isoformat(),
            "register": self.register,
            "quality": self.quality,
            "metadata": self.metadata or {},
        }


# =============================================================================
# Supabase client (lazy + cached)
# =============================================================================

_SUPABASE_CACHE: Any = None
_SUPABASE_TRIED: bool = False
_TABLE = "live_readings"
_VIEW = "latest_live_reading"


def _get_supabase_client() -> Any:
    """
    Devuelve un client de Supabase configurado, o None si no hay
    credenciales / SDK. Reusa cache para evitar re-inicializar.
    """
    global _SUPABASE_CACHE, _SUPABASE_TRIED
    if _SUPABASE_TRIED:
        return _SUPABASE_CACHE
    _SUPABASE_TRIED = True

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )

    # Intento Streamlit secrets si estamos dentro de Streamlit
    if not url or not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "supabase" in st.secrets:
                cfg = st.secrets["supabase"]
                url = url or str(cfg.get("url", "")).strip()
                key = key or str(cfg.get("service_key", "")).strip()
        except Exception:
            pass

    if not url or not key:
        log.info("live_readings: Supabase no configurado (faltan SUPABASE_URL / SUPABASE_SERVICE_KEY)")
        return None

    try:
        from supabase import create_client
        _SUPABASE_CACHE = create_client(url, key)
        return _SUPABASE_CACHE
    except Exception as e:
        log.warning("live_readings: error iniciando supabase client: %s", e)
        return None


# =============================================================================
# Operaciones públicas
# =============================================================================

def ingest_batch(readings: List[LiveReading]) -> int:
    """
    Inserta un batch de readings. Retorna cuántas filas se insertaron.

    Si Supabase no está disponible, loggea y retorna 0 (no crashea).
    """
    if not readings:
        return 0
    client = _get_supabase_client()
    if client is None:
        log.warning("ingest_batch: %d readings dropped (no Supabase)", len(readings))
        return 0
    rows = [r.to_row() for r in readings]
    try:
        resp = client.table(_TABLE).insert(rows).execute()
        # supabase-py 2.x devuelve .data; 1.x devuelve dict-like
        data = getattr(resp, "data", None)
        n = len(data) if data is not None else len(rows)
        log.info("ingest_batch: %d readings inserted into %s", n, _TABLE)
        return n
    except Exception as e:
        log.exception("ingest_batch: insert failed (%s)", e)
        return 0


def latest_for_instance(instance_id: str) -> List[Dict[str, Any]]:
    """
    Devuelve los valores actuales (último valor de cada variable+metric)
    de una instancia. Lee del view `latest_live_reading`.
    """
    client = _get_supabase_client()
    if client is None:
        return []
    try:
        resp = (
            client.table(_VIEW)
            .select("*")
            .eq("instance_id", instance_id)
            .order("variable", desc=False)
            .execute()
        )
        return list(getattr(resp, "data", []) or [])
    except Exception as e:
        log.warning("latest_for_instance failed: %s", e)
        return []


def history_for_metric(
    instance_id: str,
    variable: str,
    metric: str = "Direct",
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Devuelve histórico ordenado descendiente por captured_at.
    Útil para gráficos de tendencia.

    Ciclo 23.71 — `id` se selecciona también para permitir paginación
    keyset compuesta `(captured_at desc, id desc)` que sobrevive a
    timestamps duplicados (el collector escribe N sensores en mismo
    batch con el mismo captured_at). Sin id, la paginación perdía
    filas silenciosamente en bordes de page.
    """
    client = _get_supabase_client()
    if client is None:
        return []
    try:
        resp = (
            client.table(_TABLE)
            .select("id,captured_at,value,unit,quality")
            .eq("instance_id", instance_id)
            .eq("variable", variable)
            .eq("metric", metric)
            .order("captured_at", desc=True)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        return list(getattr(resp, "data", []) or [])
    except Exception as e:
        log.warning("history_for_metric failed: %s", e)
        return []


def recent_history_all_direct(
    instance_id: str,
    n_per_sensor: int = 30,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Devuelve {sensor_label: [{captured_at, value}, ...]} con el histórico
    Direct reciente de cada sensor en UNA SOLA query (vs N queries).

    Útil para sparklines en Live Monitoring sin penalizar latencia.
    """
    client = _get_supabase_client()
    if client is None:
        return {}
    try:
        resp = (
            client.table(_TABLE)
            .select("sensor_label,variable,value,unit,captured_at")
            .eq("instance_id", instance_id)
            .eq("metric", "Direct")
            .order("captured_at", desc=True)
            .limit(max(n_per_sensor * 30, 500))
            .execute()
        )
        rows = list(getattr(resp, "data", []) or [])
        out: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            label = r.get("sensor_label")
            if not label:
                continue
            slot = out.setdefault(label, [])
            if len(slot) < n_per_sensor:
                slot.append(r)
        # cronológico ascendente para sparklines (izq=viejo, der=reciente)
        for label in out:
            out[label] = list(reversed(out[label]))
        return out
    except Exception as e:
        log.warning("recent_history_all_direct failed: %s", e)
        return {}


def count_for_instance(instance_id: str) -> int:
    """Conteo total de readings de un activo (debug / health)."""
    client = _get_supabase_client()
    if client is None:
        return 0
    try:
        resp = (
            client.table(_TABLE)
            .select("id", count="exact")
            .eq("instance_id", instance_id)
            .limit(1)
            .execute()
        )
        return int(getattr(resp, "count", 0) or 0)
    except Exception as e:
        log.warning("count_for_instance failed: %s", e)
        return 0


__all__ = [
    "LiveReading",
    "VALID_METRICS",
    "VALID_QUALITIES",
    "ingest_batch",
    "latest_for_instance",
    "history_for_metric",
    "recent_history_all_direct",
    "count_for_instance",
]
