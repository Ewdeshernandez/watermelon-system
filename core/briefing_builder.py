"""
core.briefing_builder — Orquestador del Briefing por activo (F3)
================================================================

Junta TODO y produce el PDF del briefing de un activo (o de todos), headless:

  datos (salud/KPIs/canales)   ← reusa helpers de live_report_builder
  figuras (tendencia/espectro/onda/órbita)  ← core.briefing_figures (F1)
  redacción (resumen/diagnóstico/recomendaciones)  ← IA best-effort + fallback
  maquetación PDF              ← core.briefing_report_pdf (F2)

API:
  build_asset_briefing(instance_id, period_label="Semanal") -> (pdf_bytes, meta)
  build_all_briefings(period_label="Semanal") -> List[(instance_id, pdf_bytes, meta)]

No usa Streamlit (corre en cron). La IA es opcional: si no hay credenciales o
falla, se usa un borrador determinístico construido de los datos reales.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) Datos del activo (salud / KPIs / canales) — reusa live_report_builder
# ---------------------------------------------------------------------------

# Ciclo 23.167 (espejo de Tabular List) — planos que NO comparten la velocidad
# del keyphasor. En máquinas de doble eje (LM6000), los puntos CRF pertenecen
# al núcleo del gas generator que gira a ~10200 cpm, NO a los 3600 cpm del eje
# de potencia/generador. La tabla del briefing debe mostrar la velocidad REAL
# del eje de cada punto, y los órdenes 0.5X/1X/2X (referenciados al keyphasor
# de 3600) no aplican a esos puntos → se dejan en blanco.
_OFFSHAFT_RPM_CPM: Dict[str, float] = {"CRF": 10200.0}


def _point_rpm(plane_label: str, sensor_label: str, base_rpm: float) -> float:
    """RPM del eje al que pertenece el punto (CRF → gas generator)."""
    blob = f"{plane_label or ''} {sensor_label or ''}".upper()
    for tok, rpm in _OFFSHAFT_RPM_CPM.items():
        if tok in blob:
            return rpm
    return base_rpm


def _harmonics_apply(plane_label: str, sensor_label: str) -> bool:
    """False si el punto está en un eje distinto al del keyphasor: los
    órdenes 0.5X/1X/2X no representan su condición real (solo Overall)."""
    blob = f"{plane_label or ''} {sensor_label or ''}".upper()
    return not any(tok in blob for tok in _OFFSHAFT_RPM_CPM)


def _channel_criterion(sensor_match: Optional[Dict[str, Any]], unit: str) -> str:
    """Criterio normativo del punto (espejo de Tabular List Ciclo 22.1):
    API 670 + ISO 7919-3 solo si el sensor es proximity Y la unidad es de
    desplazamiento (mil/µm); todo lo demás rige por ISO 20816-3."""
    stype = str((sensor_match or {}).get("sensor_type", "")).lower()
    u = (unit or str((sensor_match or {}).get("unit_native", "") or "")).lower()
    is_disp = ("mil" in u or "µm" in u or "um" in u.replace("µ", "u"))
    if stype == "proximity" and is_disp:
        return "API 670 + ISO 7919-3 / ISO 20816-3"
    return "ISO 20816-3"


def _compute_asset_data(instance_id: str, instance_obj: Any) -> Optional[Dict[str, Any]]:
    from core.live_readings import latest_for_instance
    from core.live_report_builder import (
        _build_sensor_lookup, _compute_rendered_rows, _compute_health_score,
        _format_age, _seconds_since,
    )
    latest = latest_for_instance(instance_id) or []
    if not latest:
        return None

    sensor_lookup = _build_sensor_lookup(instance_obj)
    rendered_rows, severity_summary = _compute_rendered_rows(latest, sensor_lookup, instance_obj)
    score, zone, zcolor = _compute_health_score(severity_summary, latest)

    speed_row = next((r for r in latest
                      if (r.get("variable") or "").lower().startswith("velocidad")), None)
    speed_txt = (f"{float(speed_row['value']):.0f} rpm"
                 if speed_row and speed_row.get("value") is not None else "—")
    n_danger = severity_summary.get("Danger", 0)
    n_alarm = severity_summary.get("Alarma", 0)
    status = "Crítica" if n_danger else ("Atención" if n_alarm else "Operación normal")

    # vectores 0.5X/1X/2X
    vec: Dict[str, Dict[str, Any]] = {}
    for r in latest:
        s, m = r.get("sensor_label"), r.get("metric")
        if s and m in ("0.5X_Ampl", "1X_Ampl", "2X_Ampl"):
            vec.setdefault(s, {})[m] = r.get("value")

    def _a(v):
        try:
            return f"{float(v):.2f}" if v is not None and float(v) >= 1e-4 else "—"
        except Exception:
            return "—"

    # RPM base = velocidad live (keyphasor) o nominal del activo
    base_rpm = 0.0
    try:
        if speed_row and speed_row.get("value") is not None:
            base_rpm = float(speed_row["value"])
    except Exception:
        base_rpm = 0.0
    if base_rpm <= 0:
        try:
            base_rpm = float(getattr(instance_obj, "nominal_rpm", 0) or 0)
        except Exception:
            base_rpm = 0.0

    tag = (getattr(instance_obj, "tag", None) or instance_id).upper()

    try:
        from core.sensor_map import sensor_unit_family as _fam_fn
    except Exception:
        _fam_fn = None  # type: ignore

    channels = []
    for r in rendered_rows:
        sl = r["sensor_label"]
        pl = r.get("plane_label") or "—"
        v = vec.get(sl, {})
        try:
            val = f"{float(r['value']):.2f}" if r["value"] is not None else "—"
        except Exception:
            val = "—"
        sensor_match = sensor_lookup.get(sl)
        family = ""
        if _fam_fn is not None and sensor_match is not None:
            try:
                family = _fam_fn(sensor_match)
            except Exception:
                family = ""
        if not family or family == "Auto":
            u = (r["unit"] or "").lower()
            family = ("Proximity" if ("mil" in u or "µm" in u or "um" in u)
                      else "Acceleration" if u.startswith("g") else "Velocity")
        harm_ok = _harmonics_apply(pl, sl)
        channels.append({
            "machine": tag,
            "sensor_label": sl, "plane_label": pl,
            "rpm": _point_rpm(pl, sl, base_rpm),
            "family": family,
            "alarm": r.get("alarm_used", 0.0), "danger": r.get("danger_used", 0.0),
            "criterion": _channel_criterion(sensor_match, r["unit"]),
            "value": val, "unit": r["unit"], "status": r["status"],
            "x05_amp": _a(v.get("0.5X_Ampl")) if harm_ok else "—",
            "x1_amp": _a(v.get("1X_Ampl")) if harm_ok else "—",
            "x2_amp": _a(v.get("2X_Ampl")) if harm_ok else "—",
        })

    return {
        "health": {"score": score, "zone": zone, "color": zcolor},
        "kpis": {"status": status, "speed": speed_txt, "alarms": n_danger + n_alarm},
        "channels": channels,
        "severity_summary": severity_summary,
        "n_alarm": n_alarm, "n_danger": n_danger,
    }


# ---------------------------------------------------------------------------
# 2) Redacción: borrador determinístico + mejora IA opcional
# ---------------------------------------------------------------------------
def _deterministic_sections(tag: str, period: str, data: Dict[str, Any]) -> Dict[str, Any]:
    zone = data["health"].get("zone", "—")
    speed = data["kpis"].get("speed", "—")
    n_alarm = data.get("n_alarm", 0)
    n_danger = data.get("n_danger", 0)
    channels = data.get("channels", [])

    # Resumen
    estado = ("condición crítica" if n_danger else
              "puntos en alarma" if n_alarm else "operación normal")
    summary = (
        f"Durante el periodo {period.lower()}, el activo {tag} se mantuvo en "
        f"{zone} con velocidad {speed}. "
        + (f"Se registran {n_danger + n_alarm} canal(es) sobre umbral ({estado})."
           if (n_alarm or n_danger) else
           "No se registran canales sobre umbral; el activo opera dentro de la "
           "zona aceptable según ISO 20816 / API 670."))

    # Diagnóstico — por canal en alarma/danger
    alarmados = [c for c in channels if c.get("status") in ("Alarma", "Danger")]
    if alarmados:
        det = "; ".join(
            f"{c['sensor_label']} ({c.get('plane_label','')}) {c['value']} {c['unit']} "
            f"en {c['status']}" for c in alarmados[:6])
        diagnosis = (
            f"Los siguientes puntos superan su umbral y requieren seguimiento: {det}. "
            "El resto de la cadena de medición se mantiene dentro de límites normales.")
    else:
        diagnosis = (
            "Todos los puntos de medición se encuentran dentro de límites normales. "
            "Las componentes síncronas (1X) son dominantes sin armónicos elevados, "
            "consistente con operación estable.")

    # Recomendaciones
    if n_danger:
        recs = ["Atender de inmediato los puntos en condición crítica y evaluar "
                "parada controlada según criticidad del activo.",
                "Confirmar consistencia de fase 1X entre arranques antes de "
                "cualquier intervención de balanceo o alineación."]
    elif n_alarm:
        recs = ["Programar inspección de los puntos en alarma y verificar "
                "condición de balanceo (ISO 21940-12 G 2.5) y alineación.",
                "Aumentar temporalmente la frecuencia de monitoreo del activo."]
    else:
        recs = ["Mantener la frecuencia actual de monitoreo.",
                "Conservar este briefing como línea base para comparación en "
                "próximas corridas."]
    return {"summary": summary, "diagnosis": diagnosis, "recommendations": recs}


# Expansión de códigos de plano/ubicación estándar (turbinas aeroderivadas GE
# + tren de generación). Es un GLOSARIO para que la IA no invente significados
# (ej. NO leer "TRF" como "transformador"). Si un código no está acá, se lista
# tal cual con la nota de "ubicación configurada en el activo".
_PLANE_GLOSSARY = {
    "CRF": "Compressor Rear Frame (frame trasero del compresor de la turbina)",
    "CFF": "Compressor Front Frame (frame delantero del compresor)",
    "TRF": "Turbine Rear Frame (frame trasero de la turbina)",
    "TMF": "Turbine Mid Frame (frame intermedio de la turbina)",
    "GEN DE": "Generador — lado acople (Drive End)",
    "GEN NDE": "Generador — lado libre (Non-Drive End)",
    "DE": "Lado acople (Drive End)",
    "NDE": "Lado libre (Non-Drive End)",
}

# Sufijo de canal → magnitud medida (NO es una dirección).
_SUFFIX_GLOSSARY = {
    "A": "Aceleración",
    "V": "Velocidad",
    "D": "Desplazamiento (sonda de proximidad)",
}


def _machine_context_block(instance_obj: Any, data: Dict[str, Any]) -> str:
    """Arma el bloque de identidad de máquina + glosario de planos y canales
    que se inyecta al prompt de la IA. Es la fuente de verdad que evita que la
    IA adivine la nomenclatura."""
    lines: List[str] = []

    # --- Identidad real del activo ---
    try:
        from core.instance_state import compose_train_description
        train_desc = (compose_train_description(instance_obj) or "").strip()
    except Exception:
        train_desc = ""
    asset_class = (getattr(instance_obj, "asset_class", "") or "").strip()
    drv = " ".join(p for p in [
        getattr(instance_obj, "driver_manufacturer", "") or "",
        getattr(instance_obj, "driver_model", "") or "",
    ] if p).strip()
    drvn = " ".join(p for p in [
        getattr(instance_obj, "driven_manufacturer", "") or "",
        getattr(instance_obj, "driven_model", "") or "",
    ] if p).strip()
    support = (getattr(instance_obj, "support_type", "") or "").strip()
    rpm = getattr(instance_obj, "nominal_rpm", 0) or 0

    lines.append("Identidad del activo (fuente de verdad — interpretá todo a partir de esto):")
    if train_desc:
        lines.append(f"- Tren: {train_desc}")
    if asset_class:
        lines.append(f"- Clase de activo: {asset_class}")
    if drv:
        lines.append(f"- Motriz (driver): {drv}")
    if drvn:
        lines.append(f"- Accionada (driven): {drvn}")
    if rpm:
        lines.append(f"- Velocidad nominal: {float(rpm):.0f} rpm")
    if support:
        lines.append(f"- Tipo de soporte: {support}")
    lines.append("")

    channels = data.get("channels", []) or []

    # Config real de sensores del activo (fuente de verdad de producción):
    # cada sensor trae direction (X/Y/RADIAL/AXIAL) y unit_native, y
    # sensor_unit_family() deriva la magnitud (Acceleration/Velocity/Proximity).
    # De acá sacamos el desglose autoritativo en vez de adivinar por el sufijo.
    sensor_by_label: Dict[str, Any] = {}
    try:
        from core.live_report_builder import _build_sensor_lookup
        sensor_by_label = _build_sensor_lookup(instance_obj) or {}
    except Exception as e:
        log.warning("sensor lookup para contexto IA falló: %s", e)

    try:
        from core.sensor_map import sensor_unit_family
    except Exception:
        sensor_unit_family = None  # type: ignore

    _DIR_HUMAN = {
        "X": "radial X (horizontal)",
        "Y": "radial Y (vertical)",
        "RADIAL": "radial", "RAD": "radial",
        "AXIAL": "AXIAL", "AX": "AXIAL",
    }
    _FAMILY_HUMAN = {
        "Acceleration": "Aceleración",
        "Velocity": "Velocidad",
        "Proximity": "Desplazamiento (sonda de proximidad)",
        "Phase Reference": "Referencia de fase (keyphasor)",
    }

    # --- Glosario de planos presentes ---
    planes = []
    seen_p = set()
    for c in channels:
        pl = (c.get("plane_label") or "").strip()
        if pl and pl not in seen_p and pl != "—":
            seen_p.add(pl)
            planes.append(pl)
    if planes:
        lines.append("Glosario de planos/ubicaciones (código = estación de medición, NO un equipo):")
        for pl in planes:
            exp = _PLANE_GLOSSARY.get(pl.upper())
            lines.append(f"- {pl}: {exp or 'ubicación configurada en el activo'}")
        lines.append("")

    # --- Desglose autoritativo por canal (desde la config de producción) ---
    lines.append(
        "Desglose de canales desde la configuración del activo en producción. "
        "La etiqueta es \"<plano><eje>_<letra de transductor>\": el eje (X/Y) es "
        "la DIRECCIÓN y la letra (A/V/D) es solo el TIPO de transductor — NUNCA "
        "es la dirección. Un canal AXIAL real se etiqueta \"<plano>_AX_<letra>\". "
        "Usá dirección y magnitud tal como se listan acá:"
    )
    for c in channels:
        lbl = (c.get("sensor_label") or "—").strip()
        s = sensor_by_label.get(lbl)
        direction = ""
        magnitude = ""
        if s is not None:
            direction = _DIR_HUMAN.get(
                str(s.get("direction", "") or "").strip().upper(), "")
            if sensor_unit_family is not None:
                try:
                    fam = sensor_unit_family(s)
                    magnitude = _FAMILY_HUMAN.get(fam, fam)
                except Exception:
                    magnitude = ""
        # Fallback de magnitud por letra del sufijo si no hubo config
        if not magnitude and "_" in lbl:
            magnitude = _SUFFIX_GLOSSARY.get(lbl.rsplit("_", 1)[-1].upper(), "")
        meta_bits = []
        if direction:
            meta_bits.append(f"dirección {direction}")
        if magnitude:
            meta_bits.append(f"magnitud {magnitude}")
        meta_txt = f" — {', '.join(meta_bits)}" if meta_bits else ""
        lines.append(
            f"- {lbl} (plano {c.get('plane_label','—')}){meta_txt}: "
            f"{c.get('value','—')} {c.get('unit','')} · {c.get('status','—')}"
        )
    lines.append("- Recordatorio: la letra \"_A\" es un acelerómetro (Aceleración), jamás \"axial\".")
    lines.append("")

    return "\n".join(lines).strip()


def _ai_enhance(sections: Dict[str, Any], tag: str, period: str,
                data: Dict[str, Any], instance_obj: Any = None,
                figures: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Mejora best-effort con IA. Si no hay credenciales o falla, deja el
    borrador determinístico intacto."""
    try:
        from core.ai_diagnostic import is_ai_available, generate_executive_summary
        if not is_ai_available():
            return sections

        # machine_train completo (no solo el tag) para que la IA sepa qué máquina es
        machine_train = tag
        if instance_obj is not None:
            try:
                from core.instance_state import compose_train_description
                desc = (compose_train_description(instance_obj) or "").strip()
                machine_train = f"{tag} — {desc}" if desc else tag
            except Exception:
                pass

        # Bloque de contexto + glosario (la fuente de verdad anti-alucinación)
        machine_ctx = ""
        if instance_obj is not None:
            try:
                machine_ctx = _machine_context_block(instance_obj, data)
            except Exception as e:
                log.warning("machine_context falló: %s", e)

        # Análisis QUE YA ESTÁN en el reporte (figuras adjuntas). Sin esto la
        # IA cree que solo hay un punto tabular y afirma "no hay espectros /
        # forma de onda" y recomienda adquirir lo que el reporte ya incluye.
        _FIG_DESC = {
            "trend": "Tendencia overall en el tiempo",
            "spectrum": "Espectro FFT por plano (generado desde la forma de onda cruda cargada en CSV)",
            "waveform": "Forma de onda temporal por plano (CSV crudo cargado)",
            "orbit": "Órbitas por cojinete",
        }
        present = [k for k in ("trend", "spectrum", "waveform", "orbit")
                   if (figures or {}).get(k)]
        if present:
            _lines = ["", "Análisis YA INCLUIDOS en este reporte (figuras adjuntas):"]
            for k in present:
                _lines.append(f"- {_FIG_DESC[k]}")
            _lines.append(
                "REGLA: estos análisis SÍ están disponibles. NO afirmes "
                "ausencia de espectros / forma de onda / análisis de fase si "
                "aparecen arriba. NO recomiendes adquirir data que el reporte "
                "ya contiene; las recomendaciones deben ser próximos pasos "
                "diagnósticos sobre la data existente (correlación entre "
                "figuras, seguimiento de tendencia, banda de frecuencia a "
                "vigilar, etc.)."
            )
            machine_ctx = (machine_ctx + "\n" + "\n".join(_lines)).strip()

        # Items: el estado tabular + un item por cada figura presente, para que
        # la IA sintetice sabiendo que existen (aunque su contenido no venga
        # interpretado, evita el contrasentido de "no hay espectros").
        items = [{
            "type": "tabular", "title": f"Estado tabular {tag}",
            "machine": machine_train, "point": "",
            "notes": (sections["summary"] + "\n" + sections["diagnosis"]),
        }]
        for k in present:
            items.append({
                "type": k, "title": _FIG_DESC[k],
                "machine": machine_train, "point": "",
                "notes": f"Figura de {_FIG_DESC[k]} incluida en el reporte.",
            })
        meta = {
            "machine_train": machine_train,
            "period": period,
            "machine_context": machine_ctx,
        }
        res = generate_executive_summary(items, meta)
        if res.get("ok") and res.get("markdown"):
            sections = dict(sections)
            sections["summary"] = res["markdown"].strip()
    except Exception as e:
        log.warning("briefing IA enhance falló (se usa borrador): %s", e)
    return sections


# ---------------------------------------------------------------------------
# 3) Builder por activo
# ---------------------------------------------------------------------------
def _render_sensor_map(instance_obj: Any, channels: List[Dict[str, Any]]) -> Optional[bytes]:
    """Renderiza el Mapa de Sensores headless usando severidades LIVE (no la
    sesión Streamlit). Devuelve PNG o None si no hay sensores / falla."""
    try:
        sensors = getattr(instance_obj, "sensors", None)
        if not sensors:
            return None
        from core.sensor_diagram import render_sensor_map_diagram

        _norm = {"alarma": "Alarm", "alert": "Alarm", "danger": "Danger",
                 "crítica": "Danger", "critica": "Danger", "normal": "Normal"}
        sev_by_label: Dict[str, str] = {}
        for c in channels or []:
            lbl = c.get("sensor_label")
            if not lbl:
                continue
            sev_by_label[lbl] = _norm.get((c.get("status") or "").strip().lower(), "Normal")

        driver_label = (getattr(instance_obj, "driver_model", "") or "Driver").strip() or "Driver"
        driven_label = (getattr(instance_obj, "driven_model", "") or "Driven").strip() or "Driven"

        # NO pasamos overall_by_label/unit_by_label a propósito: las anotaciones
        # numéricas bajo cada cojinete se amontonan cuando hay >1 sensor por
        # plano. Los valores Overall ya viven en la tabla "Canales" del briefing,
        # así que aquí el mapa queda limpio (solo color de severidad + label).
        # train_label vacío para no repetir un subtítulo largo y desordenado.
        return render_sensor_map_diagram(
            sensors,
            train_label="",
            driver_label=driver_label,
            driven_label=driven_label,
            severity_by_label=sev_by_label or None,
        )
    except Exception as e:
        log.warning("briefing sensor map falló: %s", e)
        return None


def build_asset_briefing(
    instance_id: str,
    period_label: str = "Semanal",
    instance_obj: Any = None,
    use_ai: bool = True,
    meta_extra: Optional[Dict[str, Any]] = None,
    sections_override: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """Genera el PDF del briefing de un activo. Devuelve (pdf_bytes, meta).

    meta_extra: dict opcional con datos de portada (prepared_by, prepared_role,
    prepared_city, reviewed_by, reviewed_role, reviewed_city, consecutive,
    report_date). Lo pasa la UI con el usuario logueado; el cron lo deja vacío
    (la portada simplemente omite el bloque de firmas)."""
    from core.instance_state import get_instance
    if instance_obj is None:
        instance_obj = get_instance(instance_id)

    tag = (getattr(instance_obj, "tag", None) or instance_id).upper()
    train = " ".join(p for p in [
        getattr(instance_obj, "driver_model", "") or "",
        "→", getattr(instance_obj, "driven_model", "") or "",
    ] if p and p != "→") or "—"
    client = getattr(instance_obj, "client", "") or ""
    if client:
        train = f"{train} · {client}"

    data = _compute_asset_data(instance_id, instance_obj)
    if not data:
        return None, {"instance_id": instance_id, "status": "Sin datos", "ok": False}

    # Figuras PRIMERO: la IA necesita saber qué análisis ya trae el reporte
    # (espectro/forma de onda/tendencia/órbita) para no afirmar que faltan.
    try:
        from core.briefing_figures import collect_asset_figures
        figures = collect_asset_figures(instance_id, instance_obj)
    except Exception as e:
        log.warning("briefing figures falló: %s", e)
        figures = {}

    sections = _deterministic_sections(tag, period_label, data)
    if use_ai:
        sections = _ai_enhance(sections, tag, period_label, data,
                               instance_obj=instance_obj, figures=figures)

    # Override del flujo de APROBACIÓN: el especialista editó resumen y
    # diagnóstico en la cola de revisión → esos textos mandan sobre lo
    # generado (IA/determinístico).
    if sections_override:
        for k in ("summary", "diagnosis"):
            v = (sections_override.get(k) or "").strip()
            if v:
                sections[k] = v

    # Recomendaciones GESTIONADAS por el especialista (persisten entre
    # reportes, con fecha de inicio). Si existen, MANDAN sobre el borrador
    # automático: el sistema no inventa recomendaciones cuando el analista
    # ya definió las suyas. Si no hay ninguna, cae al borrador determinístico
    # (el cron nunca emite la sección vacía).
    try:
        from core.briefing_recommendations import list_recommendations
        _stored = list_recommendations(instance_id)
    except Exception as e:
        log.warning("briefing recomendaciones falló (se usa borrador): %s", e)
        _stored = []
    recommendations = (_stored if _stored else sections["recommendations"])

    sensor_map_png = _render_sensor_map(instance_obj, data["channels"])

    # meta de portada: train SIN cliente (el cliente va como línea propia del
    # bloque del activo) + firmas/consecutivo que pase la UI.
    train_bare = " ".join(p for p in [
        getattr(instance_obj, "driver_model", "") or "",
        "→", getattr(instance_obj, "driven_model", "") or "",
    ] if p and p != "→") or ""
    # Fecha del reporte = momento exacto de generación (fecha + hora local).
    # Antes la portada caía a solo-fecha; el usuario quiere el timestamp real.
    from datetime import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
        _gen_ts = _dt.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        _gen_ts = _dt.now().strftime("%Y-%m-%d %H:%M")
    pdf_meta = {
        "train_description": train_bare,
        "client": client,
        "report_date": _gen_ts,  # meta_extra puede sobreescribir si la UI lo pasa
    }
    if meta_extra:
        pdf_meta.update({k: v for k, v in meta_extra.items() if v})

    try:
        from core.briefing_report_pdf import generate_briefing_pdf
        pdf = generate_briefing_pdf(
            instance_id=instance_id, tag=tag, train=train,
            period_label=period_label,
            health=data["health"], kpis=data["kpis"],
            figures=figures,
            summary=sections["summary"],
            diagnosis=sections["diagnosis"],
            recommendations=recommendations,
            channels=data["channels"],
            sensor_map_png=sensor_map_png,
            meta_extra=pdf_meta,
        )
    except Exception as e:
        log.error("briefing PDF falló para %s: %s", instance_id, e)
        return None, {"instance_id": instance_id, "status": "Error PDF", "ok": False}

    meta = {
        "instance_id": instance_id, "tag": tag, "ok": True,
        "status": data["kpis"]["status"], "score": data["health"]["score"],
        "alarms": data["kpis"]["alarms"], "period": period_label,
        "n_figures": sum(1 for v in figures.values() if v),
    }
    return pdf, meta


# ---------------------------------------------------------------------------
# 3b) Borrador PENDIENTE para la cola de aprobación (lo llama el cron)
# ---------------------------------------------------------------------------
def build_asset_draft(
    instance_id: str,
    period_label: str = "Semanal",
    instance_obj: Any = None,
    use_ai: bool = True,
) -> Dict[str, Any]:
    """Genera las SECCIONES del briefing (resumen + diagnóstico, con IA
    best-effort) y las deja como borrador PENDIENTE en la cola de revisión
    (core.briefing_queue). NO genera PDF ni envía nada — eso ocurre cuando
    el especialista aprueba.

    Devuelve meta: {"instance_id","tag","ok","status",...}."""
    from core.instance_state import get_instance
    if instance_obj is None:
        instance_obj = get_instance(instance_id)
    tag = (getattr(instance_obj, "tag", None) or instance_id).upper()

    data = _compute_asset_data(instance_id, instance_obj)
    if not data:
        return {"instance_id": instance_id, "tag": tag, "ok": False,
                "status": "Sin datos"}

    # Figuras solo como CONTEXTO de la IA (no se rasteriza PDF aquí).
    try:
        from core.briefing_figures import collect_asset_figures
        figures = collect_asset_figures(instance_id, instance_obj)
    except Exception as e:
        log.warning("draft figures falló: %s", e)
        figures = {}

    sections = _deterministic_sections(tag, period_label, data)
    if use_ai:
        sections = _ai_enhance(sections, tag, period_label, data,
                               instance_obj=instance_obj, figures=figures)

    try:
        from core.briefing_queue import new_pending_draft
        ok = new_pending_draft(
            instance_id, period_label,
            summary=sections["summary"], diagnosis=sections["diagnosis"],
            health=data["health"], kpis=data["kpis"],
        )
    except Exception as e:
        log.error("draft queue falló para %s: %s", instance_id, e)
        ok = False

    return {"instance_id": instance_id, "tag": tag, "ok": ok,
            "status": data["kpis"]["status"], "score": data["health"]["score"],
            "alarms": data["kpis"]["alarms"], "period": period_label}


def build_all_drafts(period_label: str = "Semanal",
                     use_ai: bool = True) -> List[Dict[str, Any]]:
    """Borrador pendiente para cada activo con datos (cron F4)."""
    from core.instance_state import list_instances, get_instance
    out: List[Dict[str, Any]] = []
    try:
        instances = list_instances() or []
    except Exception as e:
        log.error("build_all_drafts: list_instances falló: %s", e)
        return out
    for inst in instances:
        iid = inst.get("instance_id") if isinstance(inst, dict) else getattr(inst, "instance_id", "")
        if not iid:
            continue
        try:
            obj = get_instance(iid)
            out.append(build_asset_draft(iid, period_label, instance_obj=obj,
                                         use_ai=use_ai))
        except Exception as e:
            log.warning("draft %s falló: %s", iid, e)
            out.append({"instance_id": iid, "ok": False, "status": "Error"})
    return out


# ---------------------------------------------------------------------------
# 4) Builder para TODOS los activos
# ---------------------------------------------------------------------------
def build_all_briefings(
    period_label: str = "Semanal", use_ai: bool = True,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Optional[bytes], Dict[str, Any]]]:
    """Genera el briefing de cada activo con datos. Devuelve lista de
    (instance_id, pdf_bytes|None, meta)."""
    from core.instance_state import list_instances, get_instance
    out: List[Tuple[str, Optional[bytes], Dict[str, Any]]] = []
    try:
        instances = list_instances() or []
    except Exception as e:
        log.error("build_all_briefings: list_instances falló: %s", e)
        return out
    for inst in instances:
        iid = inst.get("instance_id") if isinstance(inst, dict) else getattr(inst, "instance_id", "")
        if not iid:
            continue
        try:
            obj = get_instance(iid)
            pdf, meta = build_asset_briefing(iid, period_label, instance_obj=obj,
                                             use_ai=use_ai, meta_extra=meta_extra)
            out.append((iid, pdf, meta))
        except Exception as e:
            log.warning("briefing %s falló: %s", iid, e)
            out.append((iid, None, {"instance_id": iid, "ok": False, "status": "Error"}))
    return out


__all__ = ["build_asset_briefing", "build_all_briefings",
           "build_asset_draft", "build_all_drafts"]
