"""
core/modal/geometry_3d.py — Geometría 3D para análisis modal
=============================================================

Define la representación de la geometría 3D del activo (compresor, generador,
turbina, etc.) que se usa como soporte visual para los mode shapes.

Modelo de datos
---------------
Un Wireframe3D consiste en:
  · `nodes`: lista de vértices [{"id": str, "xyz": [x,y,z], "label": str}]
  · `edges`: lista de aristas [{"from": node_id, "to": node_id}]
  · `faces`: lista de caras triangulares [{"vertices": [id1,id2,id3]}]
  · `frame`: sistema de coordenadas {"origin": [x,y,z], "axes": [[ux,uy,uz],...]}

La geometría se persiste como JSON en data/modal/geometries/<activo>.json.

Norma aplicable
---------------
ISO 7626-6 §6 — Identificación de DOF y orientación espacial de los puntos
de medición debe ser claramente documentada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class Node3D:
    """Vértice de la geometría wireframe."""
    id: str
    xyz: Tuple[float, float, float]
    label: str = ""


@dataclass
class Edge3D:
    """Arista que conecta dos nodos."""
    from_id: str
    to_id: str


@dataclass
class Face3D:
    """Cara triangular (3 nodos) o quad (4 nodos)."""
    vertex_ids: List[str]


@dataclass
class Wireframe3D:
    """Geometría completa del activo para visualización modal."""
    asset_id: str
    name: str
    nodes: List[Node3D] = field(default_factory=list)
    edges: List[Edge3D] = field(default_factory=list)
    faces: List[Face3D] = field(default_factory=list)
    frame_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    units: str = "m"  # m, mm, in

    def to_json(self) -> Dict:
        """Serialize to dict for JSON persistence."""
        # TODO: implementar serialización completa
        raise NotImplementedError("Fase scaffolding — implementar en siguiente sprint")

    @classmethod
    def from_json(cls, data: Dict) -> "Wireframe3D":
        """Load from dict (parsed JSON)."""
        # TODO: implementar deserialización
        raise NotImplementedError("Fase scaffolding — implementar en siguiente sprint")

    @classmethod
    def from_file(cls, path: Path) -> "Wireframe3D":
        with open(path, "r") as f:
            return cls.from_json(json.load(f))


def load_geometry(asset_id: str) -> Optional[Wireframe3D]:
    """
    Carga la geometría 3D de un activo desde data/modal/geometries/.

    Returns None si no existe geometría definida para ese activo.
    """
    # TODO: implementar
    raise NotImplementedError("Fase scaffolding")
