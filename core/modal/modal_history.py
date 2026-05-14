"""
core/modal/modal_history.py — Persistencia de runs modales
============================================================

Almacena los resultados de cada campaña modal en Supabase Storage usando
la abstracción `core/history_storage.py` ya existente en el sistema.

Patrón de storage
-----------------
Bucket: instance-history (ya existe)
Path:   {instance_id}/modal/{run_id}.json.gz

Estructura de un run modal:
{
  "run_id": "uuid",
  "timestamp": "2026-05-14T...",
  "instance_id": "tes1",
  "campaign_label": "FAREX_C200C_2026Q2",
  "method": "EMA" | "OMA" | "EMA+OMA",
  "input_files": [...],          # nombres de los .tdms / .txt usados
  "sensor_config": [...],        # snapshot del sensor_map en ese momento
  "frfs": [...],                 # FRFs comprimidas si EMA
  "modes": [...],                # lista de ModalMode identificados
  "stability_diagram": {...},    # si LSCF
  "analyst": "ehernandez@sigasas.com",
  "notes": "..."
}
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ModalRun:
    """Un run completo de análisis modal."""
    run_id: str
    timestamp: datetime
    instance_id: str
    campaign_label: str
    method: str  # "EMA" | "OMA" | "EMA+OMA"
    analyst: str
    notes: str = ""


def save_modal_run(run: ModalRun, ema_result=None, oma_result=None) -> str:
    """
    Guarda un run modal en Supabase Storage.

    Args:
        run: Metadata del run
        ema_result: EMAResult si aplica
        oma_result: OMAResult si aplica

    Returns:
        path del archivo guardado en storage
    """
    # TODO: integrar con core/history_storage.py
    raise NotImplementedError("Fase scaffolding")


def list_modal_runs(instance_id: str, limit: int = 50) -> List[Dict]:
    """Lista los runs modales archivados para un activo."""
    raise NotImplementedError("Fase scaffolding")


def load_modal_run(run_id: str, instance_id: str) -> Dict:
    """Carga un run modal específico desde storage."""
    raise NotImplementedError("Fase scaffolding")
