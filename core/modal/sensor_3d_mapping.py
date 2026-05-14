"""
core/modal/sensor_3d_mapping.py — Mapeo de sensor_map a geometría 3D
=====================================================================

Conecta el Sensor Map existente de Watermelon (lógica 2D del Machine Map)
con la geometría 3D del activo definida en core/modal/geometry_3d.py.

Diseño retrocompatible
----------------------
Sensor Map ya existente NO cambia. Se agregan 4 campos OPCIONALES:

  · sensitivity_mv_per_eu  (float, e.g. 100.0)
  · coupling               (str: "IEPE" | "AC" | "DC")
  · position_3d            ([x, y, z] en metros)
  · dof_direction          (vector unitario [dx, dy, dz])

Si position_3d es None → el sensor NO se usa para análisis modal (pero sigue
funcionando para Live Monitoring, orbits, spectrum, etc.).

Inferencia automática
---------------------
Para sensores comunes (1YA, 2YA, VE5807, etc.), podemos inferir position_3d
y dof_direction desde la convención de naming + el icon_anchor existente:

  · "Y" en el nombre → eje Y (dof_direction = [0, 1, 0])
  · "X" en el nombre → eje X (dof_direction = [1, 0, 0])
  · icon_anchor="CRF" + driver_side → posición en lado libre del driver
  · icon_anchor="TRF" + driver_side → posición en lado coupling

Esto permite generar el modelo 3D inicial sin que el usuario tenga que
posicionar manualmente cada sensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SensorDOF:
    """Un sensor mapeado a un DOF de la geometría 3D."""
    plane_label: str           # Etiqueta del sensor (e.g. "1YA")
    position_3d: np.ndarray    # (3,) — xyz en metros
    dof_direction: np.ndarray  # (3,) — vector unitario
    sensitivity_mv_per_eu: float
    coupling: str
    sensor_kind: str           # "acceleration" | "velocity" | "displacement"


def map_sensors_from_instance(instance) -> List[SensorDOF]:
    """
    Lee los sensores definidos en una instance (TES1, TES3, etc.) y devuelve
    aquellos con configuración 3D válida para modal.

    Args:
        instance: Objeto Instance de core/instance_state.py

    Returns:
        Lista de SensorDOFs (solo los que tienen position_3d definido)
    """
    raise NotImplementedError("Fase scaffolding")


def infer_3d_from_2d_anchor(
    plane_label: str,
    icon_side: str,
    icon_anchor: str,
    driver_length_m: float = 2.0,
    driven_length_m: float = 2.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Inferencia heurística: dado un sensor con icon_side y icon_anchor 2D,
    calcula una posición 3D razonable + dirección DOF.

    Convención de coordenadas del activo:
      · X: eje axial (lo largo del tren driver → driven)
      · Y: eje horizontal radial
      · Z: eje vertical

    Returns:
        (position_3d, dof_direction) o None si no se puede inferir
    """
    raise NotImplementedError("Fase scaffolding")


def validate_dof_completeness(sensors: List[SensorDOF]) -> Dict:
    """
    Verifica la cobertura de DOFs en la malla.

    Returns dict con:
      · n_dofs: número total de DOFs medidos
      · n_unique_positions: posiciones únicas en el espacio
      · coverage_x, coverage_y, coverage_z: rango cubierto en cada eje
      · warnings: lista de problemas detectados
    """
    raise NotImplementedError("Fase scaffolding")
