"""
core.sensor_map
===============

Mapa de sensores de vibración por máquina (Ciclo 14c.1).

Cada Asset Instance tiene una lista de sensores configurados en
Machinery Library, donde cada sensor describe:

  - **Plano**: número correlativo desde el conductor al conducido,
    siguiendo la convención API 670 / ISO 20816-1.
    Para un tren motor + bomba: planos 1-2 = motor (DE/NDE),
    3-4 = bomba (DE/NDE).
  - **Lado y ángulo**: convención polar dividida en hemisferio L y R
    vista desde el extremo conductor del eje, con 0° arriba.
    Las sondas X-Y típicas API 670 van a +45° R (X) y +45° L (Y).
  - **Dirección**: X / Y / radial / axial.
  - **Tipo de sensor**: proximity (Desplazamiento, mil pp / µm pp),
    velocity (V, mm/s RMS), accelerometer (A, g RMS).
  - **Unidad nativa** y **setpoints individuales** (alarm / danger).
  - **Patrón de match al Point del CSV**: glob estilo ``*5807*Y*``
    que asocia los CSVs cargados al sensor correspondiente.

El label de sensor sigue convención naming industrial: ``1Y_D``
(plano 1, dirección Y, Desplazamiento), ``3X_A`` (plano 3, X,
Aceleración), ``2_RAD_V`` (plano 2, radial, Velocidad).

Ciclo 14c.1 — exporta:

  - :func:`generate_standard_sensor_map` — pre-llena 8 sensores típicos
  - :func:`resolve_sensor_for_point` — encuentra el sensor que matchea
    un Point del CSV
  - :func:`sensor_label` — formatea label estilo industrial
  - :func:`sensor_unit_family` — Proximity / Velocity / Acceleration
  - :func:`new_sensor` — constructor con defaults
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Dict, List, Optional


# Mapeo tipo de sensor → familia de medida (para Tabular List)
_TYPE_TO_FAMILY = {
    "proximity": "Proximity",
    "velocity": "Velocity",
    "accelerometer": "Acceleration",
    "keyphasor": "Phase Reference",  # Ciclo 14c.1.1 — referencia 1X de fase
}

# Mapeo tipo de sensor → letra de unidad (para naming convention)
_TYPE_TO_LETTER = {
    "proximity": "D",       # Desplazamiento
    "velocity": "V",
    "accelerometer": "A",
    "keyphasor": "K",       # Keyphasor → 1 pulso/rev
}

# Unidades nativas por defecto según tipo
_DEFAULT_UNIT_BY_TYPE = {
    "proximity": "mil pp",
    "velocity": "mm/s RMS",
    "accelerometer": "g RMS",
    "keyphasor": "pulses/rev",
}


def new_sensor(
    *,
    plane: int = 1,
    plane_label: str = "",
    side: str = "L",
    angle_deg: float = 45.0,
    direction: str = "Y",
    sensor_type: str = "proximity",
    unit_native: str = "",
    alarm: float = 0.0,
    danger: float = 0.0,
    csv_match_pattern: str = "",
    notes: str = "",
    x_pct: Optional[float] = None,
    y_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Constructor de un sensor con defaults razonables.

    Ciclo 15.2 — los campos x_pct y y_pct (0-100) son las coordenadas
    del sensor sobre el schematic_png del activo, expresadas como
    porcentaje del ancho/alto de la imagen. Cuando estan presentes,
    el render del Machine Map y del Resumen Ejecutivo overlaya
    markers de severidad sobre la foto/dibujo real del activo en
    lugar de usar el turbomachinery generico. Si quedan en None, el
    sistema cae al render generico — retro-compatible con sensor maps
    creados antes del 15.2.
    """
    if not unit_native:
        unit_native = _DEFAULT_UNIT_BY_TYPE.get(sensor_type, "")
    return {
        "plane": int(plane),
        "plane_label": plane_label,
        "side": side,
        "angle_deg": float(angle_deg),
        "direction": direction,
        "sensor_type": sensor_type,
        "unit_native": unit_native,
        "alarm": float(alarm),
        "danger": float(danger),
        "csv_match_pattern": csv_match_pattern,
        "notes": notes,
        "x_pct": x_pct,
        "y_pct": y_pct,
    }


def sensor_label(sensor: Dict[str, Any]) -> str:
    """
    Devuelve label industrial corto: ``1Y_D``, ``3X_A``, ``2_RAD_V``.

    Formato: ``{plane}{direction}_{type_letter}`` cuando direction
    es X / Y. Cuando es radial o axial, formato ``{plane}_{DIR}_{letter}``.
    """
    plane = int(sensor.get("plane", 0) or 0)
    direction = str(sensor.get("direction", "") or "").strip().upper()
    sensor_type = str(sensor.get("sensor_type", "") or "").lower()
    letter = _TYPE_TO_LETTER.get(sensor_type, "?")

    if direction in ("X", "Y"):
        return f"{plane}{direction}_{letter}"
    if direction in ("RADIAL", "RAD"):
        return f"{plane}_RAD_{letter}"
    if direction in ("AXIAL", "AX"):
        return f"{plane}_AX_{letter}"
    return f"{plane}_{direction or '?'}_{letter}"


def sensor_unit_family(sensor: Dict[str, Any]) -> str:
    """Devuelve la familia de medida en el lenguaje de Tabular List."""
    return _TYPE_TO_FAMILY.get(
        str(sensor.get("sensor_type", "") or "").lower(),
        "Auto",
    )


def _normalize_for_match(text: str) -> str:
    """Normaliza un string para comparación case-insensitive sin espacios extra."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _pattern_matches(pattern_text: str, target_norm: str) -> bool:
    """
    Devuelve True si ``pattern_text`` matchea ``target_norm`` (lowercased).

    Reglas (en orden):
    - Si pattern_text tiene comas, se splitea y matchea CUALQUIERA (OR).
    - Si una entrada contiene ``*`` o ``?``, se usa fnmatch (glob).
    - Si no, se usa substring case-insensitive.
    """
    if not pattern_text:
        return False
    for token in pattern_text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "*" in token or "?" in token:
            if fnmatch.fnmatch(target_norm, token):
                return True
        else:
            if token in target_norm:
                return True
    return False


def resolve_sensor_for_point(
    sensors: List[Dict[str, Any]],
    csv_point: str,
    csv_variable: str = "",
    csv_unit: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Encuentra el sensor del mapa que matchea un Point del CSV.

    Estrategia de match (en orden de prioridad):
      1. ``csv_match_pattern`` del sensor — admite tres formatos:
         * Lista separada por comas: "VE5807 (Y), VE5807-Y" (OR)
         * Glob: "*5807*y*"
         * Substring case-insensitive: "VE5807" matchea cualquier
           Point que contenga "VE5807" (sin importar mayúsculas).
      2. Match heurístico: si el csv_point contiene "(X)" o "(Y)"
         busca un sensor con misma direction. Si csv_unit indica
         g/m/s² busca accelerometer; mil/µm busca proximity; mm/s
         o in/s busca velocity. (Solo si exactamente 1 sensor matchea.)

    Args:
        sensors: lista de sensores del Instance.sensors.
        csv_point: campo Point del CSV (ej. "VE5807 (Y)").
        csv_variable: campo Variable del CSV (ej. "Disp Wf").
        csv_unit: unidad de amplitud (ej. "mil pp", "g").

    Returns:
        El sensor que matcheó (dict) o None si no encontró match.
    """
    if not sensors:
        return None

    point_norm = _normalize_for_match(csv_point)
    variable_norm = _normalize_for_match(csv_variable)
    unit_norm = _normalize_for_match(csv_unit)

    # =========================================================
    # Ciclo 15.1 hotfix v2 — PRE-FILTRO POR TYPE_HINT
    # =========================================================
    # Antes hacíamos pattern matching contra point Y variable, lo que
    # generaba falsos positivos cross-tipo: el pattern '*4*x*' (sensor
    # proximity plano 4) matcheaba contra variable 'Vel Wf(64X/32revs).
    # KPHGEN' porque '4x' está en '64x'. Ahora pre-filtramos los
    # candidates por type_hint detectado del Point name (más confiable
    # que la unit en CSVs Bently donde VT reporta en mil pp).

    direction_hint = ""
    if "(x)" in point_norm or " x " in f" {point_norm} " or point_norm.endswith(" x"):
        direction_hint = "X"
    elif "(y)" in point_norm or " y " in f" {point_norm} " or point_norm.endswith(" y"):
        direction_hint = "Y"

    type_hint = ""
    # 1) Por SUBSTRING del Point name (más confiable, cubre VT Bently)
    if "1vt" in point_norm or "2vt" in point_norm or "vt" in point_norm.split() or "velo" in point_norm:
        type_hint = "velocity"
    elif "vel" in point_norm.split():
        type_hint = "velocity"
    elif "acell" in point_norm or "accel" in point_norm or "ace" in variable_norm:
        type_hint = "accelerometer"
    elif (
        "ve" in point_norm.split() or "ve5" in point_norm
        or "disp" in variable_norm or "dsp" in variable_norm
        or "(x)" in point_norm or "(y)" in point_norm
    ):
        type_hint = "proximity"

    # 2) Por unit como respaldo solo si Point name no fue conclusive
    if not type_hint:
        if any(tok in unit_norm for tok in ("mm/s", "in/s", "ips")):
            type_hint = "velocity"
        elif any(tok in unit_norm for tok in ("g rms", "g pk", "g p", "m/s²", "m/s2")):
            type_hint = "accelerometer"
        elif any(tok in unit_norm for tok in ("mil", "µm", "um")):
            type_hint = "proximity"

    # PRE-FILTRO: si tenemos type_hint, restringir el universo de
    # sensores candidatos a ese tipo ANTES de hacer pattern matching.
    # Esto evita falsos positivos donde un pattern de otro tipo
    # matchea contra la variable o un substring genérico.
    universe = list(sensors)
    if type_hint:
        filtered = [s for s in universe if str(s.get("sensor_type", "")).lower() == type_hint]
        if filtered:
            universe = filtered
    if direction_hint:
        filtered_dir = [
            s for s in universe
            if str(s.get("direction", "")).upper() == direction_hint
            or str(s.get("direction", "")).upper() in ("RADIAL", "AXIAL", "")
        ]
        # Solo aplicamos el filtro de dirección si quedan candidates;
        # un sensor radial no debería excluirse cuando el Point apunta
        # a X/Y (puede ser un acelerómetro de carcasa que no distingue).
        if filtered_dir:
            universe = filtered_dir

    # =========================================================
    # FASE 1: pattern matching SOBRE EL UNIVERSO YA FILTRADO
    # =========================================================
    # Solo el point_norm se usa para pattern matching (no variable_norm)
    # para evitar falsos match contra metadata técnica de la variable.
    for sensor in universe:
        pattern = (sensor.get("csv_match_pattern") or "").strip()
        if not pattern:
            continue
        if _pattern_matches(pattern, point_norm):
            return sensor

    # =========================================================
    # FASE 2: tie-break sobre el universo filtrado
    # =========================================================
    candidates = universe

    if len(candidates) == 1:
        return candidates[0]

    # Tie-break en orden: label → plane_label → tokens del pattern → primer
    # candidate del tipo correcto (fallback gracioso).
    if candidates:
        # 1. Substring del label industrial (1y_d, 2_rad_a, etc.)
        for sensor in candidates:
            lbl = sensor_label(sensor).lower()
            if lbl in point_norm:
                return sensor

        # 2. Substring del plane_label completo (ej. "TRF (LM6000)")
        for sensor in candidates:
            plbl = _normalize_for_match(sensor.get("plane_label", ""))
            if plbl and plbl in point_norm:
                return sensor

        # 2b. Tokens cortos del plane_label (Ciclo 15.1) — split por
        # whitespace y paréntesis. Busca tokens distintivos como TRF,
        # CRF, NDE, DE, BRG en el Point del CSV. Ignora tokens muy
        # cortos o comunes.
        _label_skip = {"de", "nde", "(", ")", "lm", "tm", "brush", "bearing", "driver", "driven"}
        for sensor in candidates:
            plbl = _normalize_for_match(sensor.get("plane_label", ""))
            if not plbl:
                continue
            for token in re.split(r"[\s()/_\-]+", plbl):
                token = token.strip()
                if len(token) < 2 or token in _label_skip or token.isdigit():
                    continue
                if token in point_norm:
                    return sensor

        # 3. Tokens distintivos del csv_match_pattern (ej. "trf", "crf"
        # extraídos de "*trf*acell*"). Filtramos comunes (x, y, acell, etc.).
        _common_tokens = {"acell", "acc", "vel", "rad", "ax", "x", "y"}
        for sensor in candidates:
            pattern = (sensor.get("csv_match_pattern") or "").lower()
            for chunk in pattern.replace(",", " ").split():
                for token in chunk.split("*"):
                    token = token.strip()
                    if not token or token in _common_tokens or token.isdigit():
                        continue
                    if token in point_norm:
                        return sensor

        # 4. Fallback gracioso: si después de filtrar por type_hint quedan
        # varios candidates pero ninguno matchea por substring, devolvemos
        # el primero. Mejor un match de tipo correcto que caer al global
        # que tendría una familia incorrecta (proximity en vez de accel,
        # por ejemplo).
        if type_hint:
            return candidates[0]

    return None


# ============================================================
# Ciclo 16.1 — Wizard auto-pattern desde CSVs cargados
# ------------------------------------------------------------
# Para cada sensor del Sensor Map sin match, mira los signals
# cargados en sesion y propone un csv_match_pattern concreto
# basado en el Point name del signal mas compatible (tipo +
# direccion). Reduce el setup manual que el usuario tiene que
# hacer cuando los Point names del DCS no siguen la convencion
# API 670 (3X/3Y/4X/4Y) sino una numeracion del cliente
# (VE5807/VE5808/VE5809/VE5810).
# ============================================================

def _signal_type_compatible(sensor_type: str, signal_meta: Dict[str, Any]) -> bool:
    """¿La unidad/variable del signal es compatible con el tipo del sensor?"""
    unit = str(signal_meta.get("Y-Axis Unit", "") or signal_meta.get("Unit", "") or "").lower()
    variable = str(signal_meta.get("Variable", "") or "").lower()
    point = str(signal_meta.get("Point", "") or "").lower()

    stype = (sensor_type or "").lower()
    if stype == "proximity":
        # mil, µm, um (displacement)
        return ("mil" in unit) or ("µm" in unit) or ("um " in unit) or unit.strip() == "um" or "disp" in variable
    if stype == "velocity":
        return ("mm/s" in unit) or ("in/s" in unit) or ("vel" in variable) or ("vel" in point) or ("vt" in point)
    if stype == "accelerometer":
        # 'g' exacto, 'g peak', 'm/s²', accel/ace
        u = unit.strip()
        return (u == "g" or u.startswith("g ") or u.endswith(" g") or "g pk" in u or "g pp" in u
                or "m/s" in u or "accel" in variable or "ace" in variable
                or "acell" in point or "accel" in point or "ace" in point)
    if stype == "keyphasor":
        return "kph" in variable or "key" in variable or "rev" in variable or "tach" in variable
    return False


def _signal_direction_compatible(direction: str, signal_meta: Dict[str, Any]) -> bool:
    """¿El Point/Variable del signal indica la misma direccion del sensor?"""
    d = (direction or "").upper().strip()
    if not d or d in ("RAD", "AXIAL", "AX"):
        return True  # sensores radial/axial no necesitan matching de direccion
    point = str(signal_meta.get("Point", "") or "").upper()
    variable = str(signal_meta.get("Variable", "") or "").upper()
    haystack = f" {point} {variable} "
    if d == "X":
        return ("(X)" in haystack) or (" X " in haystack) or ("_X" in haystack) or point.endswith(" X")
    if d == "Y":
        return ("(Y)" in haystack) or (" Y " in haystack) or ("_Y" in haystack) or point.endswith(" Y")
    return True


def _extract_pattern_token(point_name: str) -> str:
    """
    Extrae el token mas distintivo del Point name para usar como pattern.

    Estrategia:
      1. Buscar el numero mas largo (suelen ser tags unicos del DCS,
         ej. 'VE5810' → '5810').
      2. Si no hay numero, usar la primera palabra alfanumerica.

    Devuelve el token (sin asteriscos), el caller lo envuelve.
    """
    import re
    pn = (point_name or "").strip()
    if not pn:
        return ""
    # Numero mas largo
    nums = re.findall(r"\d{2,}", pn)  # minimo 2 digitos para evitar tokens 0/1/2
    if nums:
        return max(nums, key=len)
    # Primera palabra alfanumerica
    words = re.findall(r"[A-Za-z0-9]+", pn)
    return words[0] if words else ""


def detect_definitive_matches(
    sensors: List[Dict[str, Any]],
    signals_meta: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Detecta los matches "definitivos" entre sensores y signals — solo
    aquellos donde el csv_match_pattern del sensor es NO vacio Y matchea
    explicitamente el Point name del signal. NO usa el fallback gracioso
    del resolver (que devuelve "el primer candidato del tipo correcto"
    aunque ningun pattern matchee).

    Args:
        sensors: lista del Sensor Map.
        signals_meta: lista de dicts con File, Point, Variable, Y-Axis Unit.

    Returns:
        Dict ``{sensor_label: signal_file}`` solo con matches con pattern
        explicito que matchea. Sensores sin pattern o con pattern que no
        matchea ningun signal NO aparecen aqui.
    """
    out: Dict[str, str] = {}
    claimed_signals: set = set()
    for s in sensors:
        pattern = (s.get("csv_match_pattern") or "").strip()
        if not pattern:
            continue
        lbl = sensor_label(s)
        for sig in signals_meta:
            sig_file = str(sig.get("File", "") or sig.get("signal_name", ""))
            if sig_file in claimed_signals:
                continue
            point = str(sig.get("Point", "") or "")
            point_norm = _normalize_for_match(point)
            if _pattern_matches(pattern, point_norm):
                out[lbl] = sig_file
                claimed_signals.add(sig_file)
                break
    return out


def suggest_pattern_for_sensor(
    sensor: Dict[str, Any],
    signals_meta: List[Dict[str, Any]],
    already_claimed_signals: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """
    Para un sensor del Sensor Map sin match, propone un csv_match_pattern
    basado en los signals cargados en sesion.

    Args:
        sensor: dict del sensor (sensor_type, direction, unit_native, ...).
        signals_meta: lista de dicts con keys File, Point, Variable,
            Y-Axis Unit (provista por el caller, normalmente desde
            session_state.signals).
        already_claimed_signals: set de signal_names que ya estan tomados
            por otros sensores. Estos se excluyen del pool de candidatos.

    Returns:
        Dict con:
          proposed_pattern: ej. "*5810*"
          candidate_point: ej. "VE5810 (X)"
          candidate_signal: ej. "5810 WF 19.csv"
          confidence: "high" / "medium" / "low"
          reason: explicacion humana breve.
        O None si no hay candidatos compatibles.
    """
    sensor_type = str(sensor.get("sensor_type", "")).lower()
    direction = str(sensor.get("direction", "")).upper()
    claimed = already_claimed_signals or set()

    # Filtrar pool: no claimed, type-compatible
    pool = []
    for sig in signals_meta:
        sig_name = str(sig.get("File", "") or sig.get("signal_name", "") or "")
        if sig_name and sig_name in claimed:
            continue
        if not _signal_type_compatible(sensor_type, sig):
            continue
        pool.append(sig)

    if not pool:
        return None

    # Refinar por direccion si aplica
    dir_filtered = [s for s in pool if _signal_direction_compatible(direction, s)]
    if dir_filtered:
        pool = dir_filtered

    if not pool:
        return None

    # Elegir candidato: si quedo uno solo, confidence high. Si quedaron
    # varios y todos comparten el mismo numero, tambien high. Si no, low.
    chosen = pool[0]
    point_name = str(chosen.get("Point", ""))
    token = _extract_pattern_token(point_name)
    if not token:
        return None
    proposed = f"*{token}*"

    if len(pool) == 1:
        confidence = "high"
        reason = (
            f"Único signal cargado compatible con un sensor "
            f"{sensor_type or 'de este tipo'}"
            + (f" en dirección {direction}" if direction in ("X", "Y") else "")
            + f". El número {token} aparece en el Point name."
        )
    else:
        confidence = "medium"
        reason = (
            f"{len(pool)} signals compatibles; tomamos el primero "
            f"({point_name}). Verificá manualmente si la asignación es correcta."
        )

    return {
        "proposed_pattern": proposed,
        "candidate_point": point_name,
        "candidate_signal": str(chosen.get("File", "") or chosen.get("signal_name", "")),
        "confidence": confidence,
        "reason": reason,
    }


def _generate_plane_sensors(
    plane_idx: int,
    plane_label: str,
    instrumentation_mode: str,
    accel_prefix: str,
    proximity_alarm: float,
    proximity_danger: float,
    accel_alarm: float,
    accel_danger: float,
    velocity_alarm: float,
    velocity_danger: float,
) -> List[Dict[str, Any]]:
    """
    Genera los sensores para UN plano según el modo de instrumentación.

    Modos soportados (Ciclo 14c.1.1):
    - **proximity_xy**: par X-Y proximity a 45° R/L (estándar API 670 para
      cojinetes hidrodinámicos / fluid_film).
    - **axial_accel**: 1 acelerómetro radial top (sólo para rolling_element
      simple, ej. motor pequeño).
    - **accel_plus_velocity**: 1 acelerómetro + 1 velocímetro radial juntos
      (estándar turbinas aeroderivadas modernas tipo LM6000 con instrumentación
      en TRF y CRF).
    """
    mode = (instrumentation_mode or "").lower()
    out: List[Dict[str, Any]] = []
    prefix_lower = accel_prefix.lower()

    if mode == "axial_accel":
        out.append(new_sensor(
            plane=plane_idx, plane_label=plane_label, side="top", angle_deg=0.0,
            direction="radial", sensor_type="accelerometer",
            unit_native="g RMS",
            alarm=accel_alarm, danger=accel_danger,
            csv_match_pattern=f"*{plane_idx}*{prefix_lower}*, *{prefix_lower}*{plane_idx}*",
        ))
    elif mode == "accel_plus_velocity":
        # Turbina aeroderivada con accel + velocity en el mismo cojinete
        # (típico LM6000: TRF y CRF cada uno con un par accel+velocity).
        # Patterns separados:
        #   accel: usa prefix + "acell" / "acc"
        #   velocity: usa prefix + "vt" / "vel" (Bently VT transducers
        #             tipicamente nombrados "1VT", "2VT", o con sufijo VEL)
        out.append(new_sensor(
            plane=plane_idx, plane_label=plane_label, side="top", angle_deg=0.0,
            direction="radial", sensor_type="accelerometer",
            unit_native="g RMS",
            alarm=accel_alarm, danger=accel_danger,
            csv_match_pattern=f"*{prefix_lower}*acell*, *{prefix_lower}*acc*, *acell*{prefix_lower}*",
        ))
        out.append(new_sensor(
            plane=plane_idx, plane_label=plane_label, side="top", angle_deg=0.0,
            direction="radial", sensor_type="velocity",
            unit_native="mm/s RMS",
            alarm=velocity_alarm, danger=velocity_danger,
            csv_match_pattern=f"*vt*{prefix_lower}*, *{prefix_lower}*vel*, *{prefix_lower}*vt*",
        ))
    else:
        # proximity_xy (default): par X-Y a 45° R/L (estándar API 670)
        out.append(new_sensor(
            plane=plane_idx, plane_label=plane_label, side="R", angle_deg=45.0,
            direction="X", sensor_type="proximity",
            unit_native="mil pp",
            alarm=proximity_alarm, danger=proximity_danger,
            csv_match_pattern=f"*{plane_idx}*x*",
        ))
        out.append(new_sensor(
            plane=plane_idx, plane_label=plane_label, side="L", angle_deg=45.0,
            direction="Y", sensor_type="proximity",
            unit_native="mil pp",
            alarm=proximity_alarm, danger=proximity_danger,
            csv_match_pattern=f"*{plane_idx}*y*",
        ))

    return out


def _support_type_to_default_mode(support_type: str) -> str:
    """
    Mapea support_type del Instance.header al modo de instrumentación
    típico de ese support type:
      - fluid_film / magnetic / mixed → proximity_xy (API 670 X-Y)
      - rolling_element → accel_plus_velocity (turbinas aero modernas)
        En vez de axial_accel, default es el más completo. El usuario puede
        cambiar a axial_accel en la UI si su máquina solo tiene 1 sensor.
    """
    sup = (support_type or "").lower()
    if sup == "rolling_element":
        return "accel_plus_velocity"
    return "proximity_xy"


def generate_standard_sensor_map(
    *,
    # Driver (máquina motriz)
    driver_planes: int = 2,
    driver_instrumentation: str = "proximity_xy",
    driver_accel_prefix: str = "acell",
    driver_plane_labels: Optional[List[str]] = None,
    # Driven (máquina accionada)
    driven_planes: int = 2,
    driven_instrumentation: str = "proximity_xy",
    driven_accel_prefix: str = "acell",
    driven_plane_labels: Optional[List[str]] = None,
    # Keyphasor (referencia 1X de fase, opcional, ubicación coupling)
    include_keyphasor: bool = False,
    keyphasor_pattern: str = "*kphgen*, *keyph*, *kp*",
    # Setpoints
    proximity_alarm_mil_pp: float = 4.0,
    proximity_danger_mil_pp: float = 6.0,
    accel_alarm_g_rms: float = 4.5,
    accel_danger_g_rms: float = 9.0,
    velocity_alarm_mm_s: float = 4.5,
    velocity_danger_mm_s: float = 11.2,
    # Compatibilidad con API previo (Ciclo 14c.1)
    driver_support_type: Optional[str] = None,
    driven_support_type: Optional[str] = None,
    nominal_planes_driver: Optional[int] = None,
    nominal_planes_driven: Optional[int] = None,
    include_axial_accelerometer: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Genera un mapa estándar de sensores para un tren acoplado.

    Modos de instrumentación por máquina (driver y driven independientes):

      - **proximity_xy**: par X-Y proxímetros a 45° R/L (API 670 clásico).
        Default para fluid_film / magnetic.

      - **axial_accel**: 1 acelerómetro radial top por plano. Para rolling
        element simple (motores chicos, bombas pequeñas).

      - **accel_plus_velocity**: 1 acelerómetro + 1 velocímetro radial top
        por plano. Estándar turbinas aeroderivadas modernas (LM6000, TM2500
        con TRF y CRF instrumentados completos).

    Casos típicos:

      - **TES1 (LM6000 + Brush 54 MW)**: driver=accel_plus_velocity con
        prefix='TRF'/'CRF' (configurable), driven=proximity_xy + keyphasor.
        Total: 4 (driver) + 4 (driven) + 1 (keyphasor) = 9 sensores.
      - **Compresor centrífugo**: ambos proximity_xy. Total: 8 sensores.
      - **Motor + bomba pequeña**: ambos axial_accel. Total: 4 sensores.

    Args:
        driver_planes / driven_planes: cantidad de cojinetes (típico 2).
        driver_instrumentation / driven_instrumentation: modo (ver arriba).
        driver_accel_prefix / driven_accel_prefix: prefijo para nombres de
            acelerómetros (ej. 'TRF', 'CRF', 'BRG', 'casing', 'acell').
            Sólo aplica si el modo incluye acelerómetros.
        driver_plane_labels / driven_plane_labels: nombres custom para cada
            plano. Si None, usa 'DE driver' / 'NDE driver' / 'DE driven' / etc.
        include_keyphasor: agrega un sensor keyphasor en el coupling al final.
        keyphasor_pattern: pattern de match para el Point del CSV del keyphasor.
        proximity_alarm/danger_mil_pp, accel_alarm/danger_g_rms,
            velocity_alarm/danger_mm_s: setpoints por familia.

    Returns:
        Lista de sensores lista para asignar a Instance.sensors.
    """
    # Back-compat con API previo (Ciclo 14c.1): si se pasa support_type,
    # se mapea al modo de instrumentación correspondiente.
    if driver_support_type is not None:
        driver_instrumentation = _support_type_to_default_mode(driver_support_type)
    if driven_support_type is not None:
        driven_instrumentation = _support_type_to_default_mode(driven_support_type)
    if nominal_planes_driver is not None:
        driver_planes = nominal_planes_driver
    if nominal_planes_driven is not None:
        driven_planes = nominal_planes_driven
    if include_axial_accelerometer is True and driven_support_type is None:
        driven_instrumentation = "axial_accel"

    sensors: List[Dict[str, Any]] = []
    plane_idx = 0

    # Driver: planos 1..N
    for i in range(driver_planes):
        plane_idx += 1
        if driver_plane_labels and i < len(driver_plane_labels):
            plane_lbl = driver_plane_labels[i]
        else:
            plane_lbl = "DE driver" if i == 0 else (
                "NDE driver" if i == 1 else f"Driver bearing {i + 1}"
            )
        sensors.extend(_generate_plane_sensors(
            plane_idx=plane_idx, plane_label=plane_lbl,
            instrumentation_mode=driver_instrumentation,
            accel_prefix=driver_accel_prefix,
            proximity_alarm=proximity_alarm_mil_pp,
            proximity_danger=proximity_danger_mil_pp,
            accel_alarm=accel_alarm_g_rms,
            accel_danger=accel_danger_g_rms,
            velocity_alarm=velocity_alarm_mm_s,
            velocity_danger=velocity_danger_mm_s,
        ))

    # Driven: planos siguientes
    for i in range(driven_planes):
        plane_idx += 1
        if driven_plane_labels and i < len(driven_plane_labels):
            plane_lbl = driven_plane_labels[i]
        else:
            plane_lbl = "DE driven" if i == 0 else (
                "NDE driven" if i == 1 else f"Driven bearing {i + 1}"
            )
        sensors.extend(_generate_plane_sensors(
            plane_idx=plane_idx, plane_label=plane_lbl,
            instrumentation_mode=driven_instrumentation,
            accel_prefix=driven_accel_prefix,
            proximity_alarm=proximity_alarm_mil_pp,
            proximity_danger=proximity_danger_mil_pp,
            accel_alarm=accel_alarm_g_rms,
            accel_danger=accel_danger_g_rms,
            velocity_alarm=velocity_alarm_mm_s,
            velocity_danger=velocity_danger_mm_s,
        ))

    # Keyphasor opcional al final, en coupling (entre driver y driven).
    # Convención: se le asigna plano 0 (especial) o el último + 1 con label
    # 'Coupling'. No tiene alarm/danger porque es referencia, no medición.
    if include_keyphasor:
        sensors.append(new_sensor(
            plane=0,
            plane_label="Coupling (keyphasor)",
            side="—", angle_deg=0.0,
            direction="axial",
            sensor_type="keyphasor",
            unit_native="pulses/rev",
            alarm=0.0, danger=0.0,  # Referencia de fase, no medición
            csv_match_pattern=keyphasor_pattern,
            notes="Referencia 1X para Polar/Bode. Montado en lado acople.",
        ))

    return sensors


__all__ = [
    "new_sensor",
    "sensor_label",
    "sensor_unit_family",
    "resolve_sensor_for_point",
    "generate_standard_sensor_map",
]
