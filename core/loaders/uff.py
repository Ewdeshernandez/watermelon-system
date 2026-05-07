"""
core.loaders.uff
================

Parser mínimo del Universal File Format (UFF / UNV), dataset 58 —
"Function at Nodal DOF". Estándar de SDRC / IDEAS adoptado por
prácticamente todos los DAQ industriales serios (NI DIAdem, DEWESoft,
LMS Test.Lab, Siemens NX, etc.).

Estructura UFF dataset 58 (resumen):

    -1
    58                                        <- dataset id
    [record 1: ID line 1, 80 chars]
    [record 2: ID line 2]
    [record 3: ID line 3 (timestamp típico)]
    [record 4: ID line 4]
    [record 5: ID line 5]
    [record 6: 6 enteros ‒ function type, abscissa info, etc.]
    [record 7: 4 enteros ‒ ord data type, n bytes/value]
    [record 8: 5 enteros + reals ‒ abscissa label spec]
    [record 9: 5 enteros + reals ‒ ordinate label spec]
    [record 10: similar]
    [record 11: similar]
    [data records: pares abscissa,ordinate (ascii) o sólo ordinate
                  según abscissa_spacing del record 6]
    -1

Soportamos:
  - function_type 1 (general), 2 (time response), 4 (auto-spectrum).
  - data uneven (espacio variable, par x,y por línea) o even (sólo y).
  - representación ASCII real (single/double).

NO soportamos (todavía):
  - data binaria (single/double precision binary)
  - dataset 58b (binary header)
  - datasets 15/82 (geometría) — estos se ignoran tranquilamente.

Para casos avanzados, el cliente puede preprocesar a ASCII con la
herramienta del DAQ.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.loaders.base import LoadedSignal, _read_text_input, _try_float


_DATASET_DELIM = "    -1"


def _split_datasets(text: str) -> List[List[str]]:
    """
    Devuelve una lista de bloques (cada bloque = lista de líneas) entre
    pares de '-1'. Tolerante con espacios/indentación.
    """
    lines = text.splitlines()
    blocks: List[List[str]] = []
    current: Optional[List[str]] = None
    for raw in lines:
        if raw.strip() == "-1":
            if current is None:
                current = []
            else:
                blocks.append(current)
                current = None
        else:
            if current is not None:
                current.append(raw)
    if current:
        blocks.append(current)
    return blocks


def _parse_record6(line: str) -> Dict[str, int]:
    """
    Record 6 es 6 enteros separados por espacios:
      function_type, function_id, version, load_case, response_name,
      response_node
    o variante:
      function_type, num_pts, abscissa_spacing, abscissa_min, abscissa_inc, z
    """
    parts = line.strip().split()
    keys = [
        "function_type", "function_id", "version",
        "load_case", "response_name", "response_node",
    ]
    out: Dict[str, int] = {}
    for k, v in zip(keys, parts):
        try:
            out[k] = int(v)
        except (ValueError, TypeError):
            out[k] = 0
    return out


def _parse_record7(line: str) -> Dict[str, Any]:
    """
    Record 7: ord_data_type, n_data_pairs, abscissa_spacing,
              abscissa_min, abscissa_increment, z_axis_value
    """
    parts = line.strip().split()
    out: Dict[str, Any] = {}
    if len(parts) >= 6:
        try:
            out["ord_data_type"] = int(parts[0])
            out["n_data_pairs"] = int(parts[1])
            out["abscissa_spacing"] = int(parts[2])  # 0=uneven, 1=even
            out["abscissa_min"] = float(parts[3])
            out["abscissa_increment"] = float(parts[4])
            out["z_axis_value"] = float(parts[5])
        except (ValueError, TypeError):
            pass
    return out


def _parse_axis_label(line: str) -> Dict[str, Any]:
    """Records 8-11: especificación de eje (data type, label, units, ...)."""
    parts = line.strip().split()
    return {"raw": line.strip(), "tokens": parts}


def parse_uff(source: Any, file_name: str = "uff_dataset_58") -> LoadedSignal:
    """
    Parsea un archivo UFF/UNV y devuelve LoadedSignal correspondiente al
    PRIMER dataset 58 encontrado.

    Para archivos con múltiples datasets, los demás se ignoran (ver
    parse_uff_all() para extraerlos todos).

    Raises:
        ValueError si no hay dataset 58 utilizable.
    """
    text = _read_text_input(source)
    if not text.strip():
        raise ValueError("UFF: archivo vacío")

    blocks = _split_datasets(text)
    for block in blocks:
        if not block:
            continue
        # Primera línea del bloque = dataset id
        dataset_id_line = block[0].strip()
        try:
            dataset_id = int(dataset_id_line)
        except ValueError:
            continue
        if dataset_id != 58:
            continue
        return _parse_dataset58_block(block, file_name=file_name)

    raise ValueError("UFF: no se encontró dataset 58 utilizable.")


def parse_uff_all(source: Any, file_name: str = "uff") -> List[LoadedSignal]:
    """Devuelve TODOS los datasets 58 del archivo como LoadedSignal."""
    text = _read_text_input(source)
    blocks = _split_datasets(text)
    out: List[LoadedSignal] = []
    for i, block in enumerate(blocks):
        if not block:
            continue
        try:
            ds_id = int(block[0].strip())
        except ValueError:
            continue
        if ds_id != 58:
            continue
        try:
            sig = _parse_dataset58_block(block, file_name=f"{file_name}_ds{i}")
            out.append(sig)
        except ValueError:
            continue
    return out


def _parse_dataset58_block(block: List[str], file_name: str) -> LoadedSignal:
    """
    block[0] = "58"
    block[1..5] = ID lines 1-5 (texto libre)
    block[6] = record 6 (6 enteros)
    block[7] = record 7 (ord_data_type, n_pts, abscissa_spacing, ...)
    block[8..11] = axis label specs
    block[12...] = datos
    """
    if len(block) < 13:
        raise ValueError("UFF dataset 58: bloque demasiado corto.")

    id_lines = [block[i].rstrip() for i in range(1, 6)]
    record6 = _parse_record6(block[6])
    record7 = _parse_record7(block[7])
    abscissa_spec = _parse_axis_label(block[8])
    ordinate_spec_re = _parse_axis_label(block[9])
    ordinate_spec_im = _parse_axis_label(block[10])

    n_pts = int(record7.get("n_data_pairs", 0)) or 0
    abscissa_spacing = int(record7.get("abscissa_spacing", 0))
    abs_min = float(record7.get("abscissa_min", 0.0))
    abs_inc = float(record7.get("abscissa_increment", 0.0))

    data_lines = block[11:]
    nums: List[float] = []
    for line in data_lines:
        for tok in line.replace("D", "E").replace("d", "E").split():
            v = _try_float(tok)
            if v is not None:
                nums.append(v)

    if not nums:
        raise ValueError("UFF dataset 58: bloque sin datos numéricos.")

    if abscissa_spacing == 1:
        # Even spacing: sólo ordenadas en data, abscissa = lin(abs_min, +inc)
        ord_vals = np.asarray(nums, dtype=float)
        if n_pts > 0:
            ord_vals = ord_vals[:n_pts]
        abs_vals = abs_min + abs_inc * np.arange(ord_vals.size, dtype=float)
    else:
        # Uneven spacing: pares abscissa,ordinate
        if len(nums) % 2 != 0:
            nums = nums[:-1]
        arr = np.asarray(nums, dtype=float).reshape(-1, 2)
        abs_vals = arr[:, 0]
        ord_vals = arr[:, 1]

    function_type = int(record6.get("function_type", 0))
    domain = "time" if function_type == 2 else "spectrum"

    fs: Optional[float] = None
    if domain == "time" and abs_inc > 0:
        fs = 1.0 / abs_inc
    elif domain == "time" and abs_vals.size >= 2:
        dt = float(np.median(np.diff(abs_vals)))
        if dt > 0:
            fs = 1.0 / dt

    metadata = {
        "id_line_1": id_lines[0] if len(id_lines) > 0 else "",
        "id_line_2": id_lines[1] if len(id_lines) > 1 else "",
        "id_line_3": id_lines[2] if len(id_lines) > 2 else "",
        "id_line_4": id_lines[3] if len(id_lines) > 3 else "",
        "id_line_5": id_lines[4] if len(id_lines) > 4 else "",
        "function_type": function_type,
        "ord_data_type": record7.get("ord_data_type"),
        "abscissa_spacing": abscissa_spacing,
        "abscissa_min": abs_min,
        "abscissa_increment": abs_inc,
        "n_data_pairs_declared": n_pts,
        "abscissa_label_raw": abscissa_spec.get("raw", ""),
        "ordinate_label_raw_real": ordinate_spec_re.get("raw", ""),
        "ordinate_label_raw_imag": ordinate_spec_im.get("raw", ""),
    }

    if domain == "time":
        return LoadedSignal(
            file_name=file_name,
            x=ord_vals,
            time=abs_vals,
            fs=fs,
            rpm=None,
            units="",
            domain="time",
            vendor="uff",
            metadata=metadata,
        )

    return LoadedSignal(
        file_name=file_name,
        x=ord_vals,
        time=None,
        fs=fs,
        rpm=None,
        units="",
        domain="spectrum",
        vendor="uff",
        metadata={**metadata, "axis_freq_hz": abs_vals.tolist()},
    )


__all__ = ["parse_uff", "parse_uff_all"]
