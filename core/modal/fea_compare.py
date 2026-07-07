"""
core/modal/fea_compare.py — Comparacion EMA/OMA experimental vs FEA
====================================================================

Importer + correlador para validar un modelo FEA (Finite Element Analysis)
contra los modos identificados experimentalmente. Es la pieza que cierra
el ciclo modal: el modelo numerico solo es valido cuando reproduce los
modos experimentales con MAC alto y frecuencias coherentes.

Formato de import soportado (JSON)
----------------------------------
```json
{
  "model_name": "Compresor C-200C — FEA Ansys 2024 R2",
  "software": "ANSYS Mechanical 2024 R2",
  "dof_names": ["1YA", "1XA", "2YA", "2XA", "3YA"],
  "modes": [
    {"freq_hz": 45.2, "damping_pct": 0.5, "mode_shape": [0.10, 0.08, ...]},
    {"freq_hz": 78.1, "damping_pct": 0.8, "mode_shape": [...]}
  ]
}
```

`dof_names` debe coincidir (case-insensitive) con los `channel_names` de la
identificacion experimental. Los modos pueden venir reales (lista de floats)
o complejos (lista de [real, imag]).

Roadmap futuro: parsers nativos para .rst (Ansys), .op2 (Nastran),
.odb (Abaqus). Por ahora se exporta JSON desde cada uno.

Norma aplicable
---------------
API 684 secc. 1.6 — Rotor dynamics: el modelo debe reproducir las frecuencias
naturales identificadas en la prueba con un error < 10% y MAC > 0.7
para los modos en el rango de operacion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np


@dataclass
class FEAMode:
    """Modo extraido de un analisis FEA."""
    mode_number: int
    freq_hz: float
    damping_pct: float
    mode_shape: np.ndarray  # complejo (N_dof,)
    label: str = ""


@dataclass
class FEAResult:
    """Resultado completo de un import FEA."""
    model_name: str
    software: str
    dof_names: List[str]
    modes: List[FEAMode] = field(default_factory=list)

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def freq_range(self) -> Tuple[float, float]:
        if not self.modes:
            return (0.0, 0.0)
        fs = [m.freq_hz for m in self.modes]
        return (min(fs), max(fs))


def load_fea_json(content: str) -> FEAResult:
    """Parsea un JSON FEA en formato Watermelon."""
    data = json.loads(content) if isinstance(content, str) else content
    if not isinstance(data, dict):
        raise ValueError("El archivo no tiene formato JSON dict valido")
    if "modes" not in data:
        raise ValueError("Falta la clave 'modes' en el JSON FEA")
    dof_names = list(data.get("dof_names", []))
    if not dof_names:
        raise ValueError("Falta 'dof_names' (lista de nombres de canal del FEA)")

    modes: List[FEAMode] = []
    for i, raw in enumerate(data["modes"]):
        if "freq_hz" not in raw or "mode_shape" not in raw:
            raise ValueError(f"Modo {i} sin 'freq_hz' o 'mode_shape'")
        shape_raw = raw["mode_shape"]
        # Soportar complejos como [real, imag] o reales como floats
        shape_complex: List[complex] = []
        for v in shape_raw:
            if isinstance(v, (list, tuple)) and len(v) == 2:
                shape_complex.append(complex(float(v[0]), float(v[1])))
            else:
                shape_complex.append(complex(float(v), 0.0))
        if len(shape_complex) != len(dof_names):
            raise ValueError(
                f"Modo {i}: tamano mode_shape ({len(shape_complex)}) no coincide "
                f"con dof_names ({len(dof_names)})"
            )
        modes.append(FEAMode(
            mode_number=i + 1,
            freq_hz=float(raw["freq_hz"]),
            damping_pct=float(raw.get("damping_pct", 0.0)),
            mode_shape=np.array(shape_complex, dtype=complex),
            label=str(raw.get("label", f"FEA-{i+1}")),
        ))

    return FEAResult(
        model_name=str(data.get("model_name", "FEA Model")),
        software=str(data.get("software", "Unknown")),
        dof_names=dof_names,
        modes=modes,
    )


def align_mode_shapes(
    fea_shape: np.ndarray,
    fea_dof_names: List[str],
    exp_dof_names: List[str],
) -> Optional[np.ndarray]:
    """
    Reordena el mode_shape del FEA para que sus componentes esten en el mismo
    orden que los canales experimentales. Si algun canal experimental no
    existe en el DOF set del FEA, devuelve None (no se puede comparar).
    """
    fea_idx_by_name = {n.strip().upper(): i for i, n in enumerate(fea_dof_names)}
    aligned: List[complex] = []
    for ch in exp_dof_names:
        key = ch.strip().upper()
        if key not in fea_idx_by_name:
            return None
        aligned.append(complex(fea_shape[fea_idx_by_name[key]]))
    return np.array(aligned, dtype=complex)


def compute_fea_experimental_cross_mac(
    fea_modes: List[FEAMode],
    fea_dof_names: List[str],
    exp_mode_shapes: List[np.ndarray],
    exp_dof_names: List[str],
) -> Optional[np.ndarray]:
    """
    Cross-MAC (N_fea, N_exp).

    Devuelve None si algun canal experimental no tiene contraparte en el FEA
    (en ese caso conviene reducir el set experimental o complementar el FEA).
    """
    n_fea = len(fea_modes)
    n_exp = len(exp_mode_shapes)
    if n_fea == 0 or n_exp == 0:
        return np.zeros((n_fea, n_exp))

    # Alinear cada modo FEA al orden experimental
    aligned_fea: List[np.ndarray] = []
    for m in fea_modes:
        al = align_mode_shapes(m.mode_shape, fea_dof_names, exp_dof_names)
        if al is None:
            return None
        aligned_fea.append(al)

    mac = np.zeros((n_fea, n_exp))
    for i, phi_f in enumerate(aligned_fea):
        for j, phi_e in enumerate(exp_mode_shapes):
            phi_e = np.asarray(phi_e, dtype=complex).flatten()
            num = abs(np.vdot(phi_f, phi_e)) ** 2
            denom = float(np.vdot(phi_f, phi_f).real *
                            np.vdot(phi_e, phi_e).real)
            mac[i, j] = num / max(denom, 1e-30)
    return mac


def pair_modes(
    mac_matrix: np.ndarray,
    fea_freqs: List[float],
    exp_freqs: List[float],
    mac_threshold: float = 0.7,
    freq_tolerance_pct: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Pareo greedy: para cada modo FEA, busca el experimental con mayor MAC.
    Marca el par como 'valid' si cumple MAC y delta de frecuencia.

    API 684 secc. 1.6 / Ewins (2000):
      - MAC >= 0.7 (correlacion de forma)
      - |delta_f|/f_exp <= 10% (correlacion de frecuencia)
    """
    pairs: List[Dict[str, Any]] = []
    used_exp: set = set()
    for i in range(mac_matrix.shape[0]):
        # Mejor experimental para este FEA, excluyendo los ya pareados
        best_j = -1
        best_mac = -1.0
        for j in range(mac_matrix.shape[1]):
            if j in used_exp:
                continue
            if mac_matrix[i, j] > best_mac:
                best_mac = mac_matrix[i, j]
                best_j = j
        if best_j < 0:
            pairs.append({
                "fea_mode": i + 1, "fea_freq": fea_freqs[i],
                "exp_mode": None, "exp_freq": None,
                "mac": 0.0, "delta_freq_pct": None,
                "status": "no_match",
            })
            continue

        exp_f = exp_freqs[best_j]
        delta_pct = (
            abs(fea_freqs[i] - exp_f) / max(exp_f, 1e-6) * 100.0
        )
        mac_ok = best_mac >= mac_threshold
        freq_ok = delta_pct <= freq_tolerance_pct
        if mac_ok and freq_ok:
            status = "valid"
        elif mac_ok and not freq_ok:
            status = "shape_only"   # forma correlaciona, freq no
        elif not mac_ok and freq_ok:
            status = "freq_only"    # freq correlaciona, forma no
        else:
            status = "weak"

        pairs.append({
            "fea_mode": i + 1, "fea_freq": fea_freqs[i],
            "exp_mode": best_j + 1, "exp_freq": exp_f,
            "mac": float(best_mac), "delta_freq_pct": float(delta_pct),
            "status": status,
        })
        used_exp.add(best_j)
    return pairs


def build_cross_mac_heatmap(
    mac: np.ndarray,
    fea_labels: List[str],
    exp_labels: List[str],
    title: str = "Cross-MAC FEA ↔ Experimental",
):
    """Heatmap Plotly del Cross-MAC con anotaciones de valores."""
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(
        z=mac.tolist(),
        x=exp_labels,
        y=fea_labels,
        colorscale="Viridis",
        zmin=0.0, zmax=1.0,
        text=[[f"{v:.2f}" for v in row] for row in mac.tolist()],
        texttemplate="%{text}",
        hovertemplate="FEA: %{y}<br>Exp: %{x}<br>MAC: %{z:.3f}<extra></extra>",
        colorbar=dict(title="MAC"),
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Modos experimentales (EMA/OMA)"),
        yaxis=dict(title="Modos FEA", autorange="reversed"),
        margin=dict(l=80, r=20, t=50, b=80),
        height=max(420, 40 * len(fea_labels) + 200),
        paper_bgcolor="white",
    )
    return fig


def example_fea_payload(exp_dof_names: List[str]) -> Dict[str, Any]:
    """
    Genera un payload JSON FEA de ejemplo coherente con los canales
    experimentales del usuario — para que descargue como template y lo edite.
    """
    n = len(exp_dof_names)
    if n == 0:
        return {"model_name": "Ejemplo", "software": "ANSYS", "dof_names": [], "modes": []}
    # 3 modos sinteticos con shapes simples
    modes_payload = []
    for k in range(3):
        # Shape "k+1"-esimo modo flexion
        shape = [
            float(np.sin((k + 1) * np.pi * (i + 1) / (n + 1)))
            for i in range(n)
        ]
        modes_payload.append({
            "freq_hz": round(50.0 * (k + 1) + 5.0, 2),
            "damping_pct": round(0.5 + 0.1 * k, 2),
            "mode_shape": shape,
            "label": f"Bending Mode {k + 1}",
        })
    return {
        "model_name": "Ejemplo FEA — editar antes de usar",
        "software": "ANSYS Mechanical 2024 R2",
        "dof_names": list(exp_dof_names),
        "modes": modes_payload,
    }
