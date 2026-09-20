"""
core/train_svg.py — Diagrama del TREN (esquemático de Live Monitoring) headless.
================================================================================

Reusa el mismo SVG que muestra Live Monitoring (siluetas realistas turbina /
gearbox / generador con cojinetes coloreados por severidad) PARA EL REPORTE PDF.

El SVG se embebe como VECTOR en el PDF vía svglib (svg2rlg → reportlab Drawing),
sin cairo. Así el reporte lleva el diagrama bonito de la web, no el matplotlib.

`_infer_side_anchor` es copia fiel de pages/02_Live_Monitoring.py (misma lógica de
ubicación de sensores; crítico el resolver por 'variable' para planos 5/6 del
SGT300: turbina + gearbox + generador).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def _infer_side_anchor(
    sensor_label: str,
    sensor_match: Optional[Dict[str, Any]],
    instance_obj: Any,
    variable: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    """Copia fiel de pages/02: mapea (label, sensor_dict) → (side, anchor)."""
    if sensor_match:
        s_side = sensor_match.get("icon_side")
        s_anchor = sensor_match.get("icon_anchor")
        if s_side and s_anchor:
            return s_side, s_anchor

    drv_key = (getattr(instance_obj, "driver_icon_key", "") or "").lower()
    is_aero = "aero" in drv_key
    label_l = (sensor_label or "").strip().lower()
    plane_l = ((sensor_match or {}).get("plane_label") or "").lower()

    var_u = (variable or "").upper()
    _d0 = label_l[0] if (label_l and label_l[0].isdigit()) else None
    if var_u:
        if any(k in var_u for k in (
            "GEARBOX", "REDUCTOR", "BOMBA", "STARTER", "ARRANCADOR", "PUMP",
        )):
            if "BOMBA" in var_u or "PUMP" in var_u:
                return "gearbox", "GB_BOMBA"
            if "STARTER" in var_u or "ARRANCADOR" in var_u:
                return "gearbox", "GB_STARTER"
            if _d0 == "3":
                return "gearbox", "GB_HSS"
            if "x" in label_l:
                return "gearbox", "GB_PROX_B"
            return "gearbox", "GB_PROX_T"
        if "GEN NDE" in var_u or "GENERADOR NDE" in var_u:
            return "driven", "NDE"
        if "GEN DE" in var_u or "GENERADOR DE" in var_u:
            return "driven", "DE"
        if any(k in var_u for k in ("TURBINA", "DRIVER", "MOTOR", "ENGINE")):
            if "NDE" in var_u:
                return "driver", ("CRF" if is_aero else "NDE")
            if "DE" in var_u:
                return "driver", ("TRF" if is_aero else "DE")

    if "gearbox" in plane_l or "reductor" in plane_l or "gearbox" in label_l:
        if "lss" in plane_l or "lss" in label_l:
            return "gearbox", "NDE"
        return "gearbox", "DE"

    label_u = (sensor_label or "").strip().upper()
    if is_aero:
        if " CRF" in label_u or label_u.endswith("CRF"):
            return "driver", "CRF"
        if " TRF" in label_u or label_u.endswith("TRF"):
            return "driver", "TRF"
    if "GEN NDE" in label_u or " NDE" in label_u:
        return "driven", "NDE"
    if "GEN DE" in label_u:
        return "driven", "DE"

    if label_l and label_l[0].isdigit():
        bearing_num = int(label_l[0])
        if bearing_num == 1:
            return "driver", ("CRF" if is_aero else "NDE")
        if bearing_num == 2:
            return "driver", ("TRF" if is_aero else "DE")
        if bearing_num == 3:
            return "driven", "DE"
        if bearing_num == 4:
            return "driven", "NDE"
        return None, None

    side: Optional[str] = None
    anchor: Optional[str] = None
    if any(t in plane_l for t in ("driver", "motor", "turbina", "engine")):
        side = "driver"
    elif any(t in plane_l for t in (
        "driven", "compresor", "compressor", "generador", "generator",
        "bomba", "pump", "frame", "cilindro", "cylinder",
    )):
        side = "driven"
    if "nde" in plane_l:
        anchor = "CRF" if (side == "driver" and is_aero) else "NDE"
    elif "de" in plane_l:
        anchor = "TRF" if (side == "driver" and is_aero) else "DE"
    if side and anchor:
        return side, anchor
    return None, None


def build_train_svg(instance_obj: Any, latest: List[Dict[str, Any]],
                    sensor_lookup: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[str]:
    """Arma el SVG del tren (siluetas + cojinetes por severidad) desde las
    lecturas 'latest'. Devuelve el string SVG o None si no hay iconos/keys."""
    try:
        drv_key = (getattr(instance_obj, "driver_icon_key", "") or "").strip()
        drvn_key = (getattr(instance_obj, "driven_icon_key", "") or "").strip()
        if not drv_key or not drvn_key:
            return None
        from core.asset_library.composer import compose_train
        from core.severity import compute_severity
        try:
            from core.instance_state import detect_gearbox_kwargs
            gbx = detect_gearbox_kwargs(instance_obj) or {}
        except Exception:  # noqa: BLE001
            gbx = {}

        sensor_lookup = sensor_lookup or {}
        MAX_PER_ANCHOR = 2
        anchor_count: Dict[Tuple[str, str], int] = {}
        sensors: List[Dict[str, Any]] = []
        for r in (latest or []):
            if (r.get("metric") or "") != "Direct":
                continue
            lbl = r.get("sensor_label")
            if not lbl:
                continue
            match = sensor_lookup.get(lbl)
            side, anchor = _infer_side_anchor(lbl, match, instance_obj, r.get("variable", ""))
            if not side or not anchor:
                continue
            ak = (side, anchor)
            if anchor_count.get(ak, 0) >= MAX_PER_ANCHOR:
                continue
            anchor_count[ak] = anchor_count.get(ak, 0) + 1
            unit = r.get("unit") or ""
            sev = compute_severity(r.get("value"), match, unit, instance_obj)
            try:
                _vs = f"{float(r.get('value')):.2f}"
            except (TypeError, ValueError):
                _vs = "—"
            # Gearbox: dots COMPACTOS (sin barra de umbral) — igual que la web,
            # si no los 8 sensores del gearbox se amontonan. Color+valor bastan.
            _gear = (side == "gearbox")
            sensors.append({
                "label": lbl, "side": side, "anchor": anchor,
                "status": sev.get("status", "Normal"), "value": _vs, "unit": unit,
                "alarm": None if _gear else sev.get("alarm"),
                "danger": None if _gear else sev.get("danger"),
            })
        if not sensors:
            return None

        drv_label = (getattr(instance_obj, "driver_model", "") or "Driver").strip() or "Driver"
        drvn_label = (getattr(instance_obj, "driven_manufacturer", "")
                      or getattr(instance_obj, "driven_model", "") or "Driven").strip() or "Driven"
        svg = compose_train(
            driver_key=drv_key, driven_key=drvn_key,
            driver_label=drv_label, driven_label=drvn_label,
            coupling=getattr(instance_obj, "coupling_class", "") or "flexible",
            sensors_with_status=sensors, **gbx,
        )
        return svg if isinstance(svg, str) else (svg[0] if isinstance(svg, (list, tuple)) else None)
    except Exception:  # noqa: BLE001
        return None


def svg_to_train_drawing(svg: str, target_width_pt: float = 500.0):
    """SVG (string) → reportlab Drawing escalado a `target_width_pt`. Vector,
    sin cairo (svglib). Devuelve None si no se pudo."""
    if not svg:
        return None
    try:
        from svglib.svglib import svg2rlg
    except Exception:  # noqa: BLE001
        return None
    try:
        m = re.search(r'viewBox="([\d.\s-]+)"', svg)
        if m:
            vb = m.group(1).split()
            W, H = float(vb[2]), float(vb[3])
        else:
            W, H = 1220.0, 200.0
        # svglib no acepta width/height 'auto'/'%': fijar px explícitos en el <svg>
        tag_m = re.match(r'<svg[^>]*>', svg)
        if tag_m:
            tag = tag_m.group(0)
            new = re.sub(r'width\s*:\s*[^;"]+;?', '', tag)
            new = re.sub(r'height\s*:\s*[^;"]+;?', '', new)
            new = re.sub(r'\swidth="[^"]*"', '', new)
            new = re.sub(r'\sheight="[^"]*"', '', new)
            new = new.replace('<svg', f'<svg width="{W:.0f}" height="{H:.0f}"', 1)
            svg = svg.replace(tag, new, 1)
        import tempfile
        import os
        with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False, encoding="utf-8") as tf:
            tf.write(svg)
            _p = tf.name
        try:
            draw = svg2rlg(_p)
        finally:
            try:
                os.unlink(_p)
            except OSError:
                pass
        if draw is None or not getattr(draw, "width", 0):
            return None
        sc = target_width_pt / draw.width
        draw.scale(sc, sc)
        draw.width *= sc
        draw.height *= sc
        return draw
    except Exception:  # noqa: BLE001
        return None
