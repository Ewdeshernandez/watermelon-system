"""
core/modal/modal_report.py — Generador de reportes modales en PDF
==================================================================

Crea secciones de modal para el reporte PDF integrado de Watermelon
(reusa la infraestructura de Reports existente).

Contenido típico del reporte
----------------------------
1. Portada de la campaña modal
   · Cliente, activo, fecha, analista
   · Método aplicado (EMA, OMA, ambos)
   · Marco normativo (ISO 7626, ISO 20816, API 684)

2. Configuración del ensayo
   · Geometría 3D del activo (vista isométrica)
   · Sensores instalados (tabla con posiciones + sensibilidades)
   · Hardware usado (NI-9234, Wilcoxon, etc.)
   · Parámetros de adquisición (fs, duración, ventanas)

3. Resultados EMA (si aplica)
   · Plot Bode de FRFs principales
   · Stability diagram interactivo
   · Tabla modal: frecuencia, damping, complejidad por modo
   · Mode shapes 3D (snapshots o links a animaciones)

4. Resultados OMA (si aplica)
   · PSD y singular values del FDD
   · Modos identificados por SSI
   · Comparación EMA vs OMA (MAC matrix)

5. Comparación con FEA (si disponible)
   · Tabla EMA/OMA vs FEA
   · MAC matrix
   · Recomendación de iteración del modelo

6. Conclusiones y recomendaciones
   · Cumplimiento del criterio API 618 §7.9.4.2.5.3.2 (separación ≥ 10%)
   · Modos críticos identificados
   · Acción recomendada
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def build_modal_section(
    run_id: str,
    instance_id: str,
    output_path: Path,
    include_3d_animations: bool = False,
) -> Path:
    """
    Construye una sección de reporte modal a partir de un run archivado.

    Args:
        run_id: ID del modal run
        instance_id: ID del activo (TES1, TES3, etc.)
        output_path: Ruta del PDF a generar
        include_3d_animations: Si True, embeds GIFs de mode shapes

    Returns:
        Path del PDF generado
    """
    raise NotImplementedError("Fase scaffolding — integración próximo sprint")
