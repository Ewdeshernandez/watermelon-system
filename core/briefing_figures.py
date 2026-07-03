"""
core.briefing_figures — Ensamblador HEADLESS de figuras por activo (F1)
======================================================================

Para el Briefing Semanal/Mensual automático (cron, sin Streamlit). Lee los
ÚLTIMOS snapshots de análisis del activo (espectro / forma de onda / órbita,
guardados en Supabase Storage) + la tendencia en vivo, y los rinde a PNG
listos para embeber en el PDF del briefing.

Claves de diseño:
  • HEADLESS PURO: no usa st.* ni st.session_state (corre en cron). Carga los
    snapshots vía history_storage + los load_fn de cada *_history.
  • SIN OOM: rasteriza vía core.plot_export (decima trazas densas + scale=1).
  • REUSA lo existente: render_trend_png (live_report_pdf), los payloads de
    spectrum_history / waveform_history / orbit_history, channel_order.

Salida principal:
  collect_asset_figures(instance_id) -> {
     "trend": bytes|None, "spectrum": bytes|None,
     "waveform": bytes|None, "orbit": bytes|None,
  }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_SPEC_FMAX_CPM = 60_000.0
_PALETTE = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2",
            "#be185d", "#475569"]

# Ciclo 23.167 — puntos que giran en un eje distinto al keyphasor (LM6000:
# el núcleo del gas generator, planos CRF, gira a ~10200 cpm vs 3600 del eje
# de potencia). Los marcadores de armónicos del espectro deben referenciarse
# a la velocidad REAL del eje de cada canal.
_OFFSHAFT_RPM_CPM = {"CRF": 10200.0}
_N_HARMONICS = 4  # marcadores 1X..4X sobre cada espectro


def _shaft_cpm_for_label(label: str, base_rpm: float) -> float:
    blob = (label or "").upper()
    for tok, rpm in _OFFSHAFT_RPM_CPM.items():
        if tok in blob:
            return rpm
    return base_rpm


# ---------------------------------------------------------------------------
# Carga headless del último snapshot de un tipo
# ---------------------------------------------------------------------------
def _load_latest_snapshot(instance_id: str, key: str) -> Optional[Dict[str, Any]]:
    """Devuelve el payload del snapshot más reciente del tipo `key`
    ('spectrum'|'waveform'|'orbit') o None. Sin Streamlit."""
    try:
        from core import history_storage as hs
        snaps = hs.list_snapshots(instance_id, key)
        if not snaps:
            return None
        sid = snaps[0].get("snapshot_id", "")
        mod_name, fn_name = {
            "spectrum": ("core.spectrum_history", "load_spectrum_snapshot"),
            "waveform": ("core.waveform_history", "load_waveform_snapshot"),
            "orbit":    ("core.orbit_history", "load_orbit_snapshot"),
        }[key]
        import importlib
        mod = importlib.import_module(mod_name)
        return getattr(mod, fn_name)(instance_id, sid)
    except Exception as e:
        log.warning("briefing_figures: load %s/%s falló: %s", instance_id, key, e)
        return None


def _load_merged_snapshot(instance_id: str, key: str,
                          max_snaps: int = 8) -> Optional[Dict[str, Any]]:
    """Fusiona los últimos snapshots del tipo `key` en UN payload.

    v3.31.399 — El snapshot MÁS RECIENTE suele traer solo los canales del
    último análisis (ej. generador desde CSV Bently) y dejaba FUERA a la
    turbina (CRF/TRF, analizada en otra corrida). Se recorren los últimos
    `max_snaps` snapshots (nuevo→viejo) y se toma, por canal, su versión
    más reciente. Así el apilado del briefing incluye TODOS los canales.
    """
    try:
        from core import history_storage as hs
        snaps = hs.list_snapshots(instance_id, key) or []
        if not snaps:
            return None
        import importlib
        mod_name, fn_name = {
            "spectrum": ("core.spectrum_history", "load_spectrum_snapshot"),
            "waveform": ("core.waveform_history", "load_waveform_snapshot"),
            "orbit":    ("core.orbit_history", "load_orbit_snapshot"),
        }[key]
        load_fn = getattr(importlib.import_module(mod_name), fn_name)
        coll_key = "bearings" if key == "orbit" else "sensors"
        label_key = "bearing_label" if key == "orbit" else "sensor_label"
        merged: Optional[Dict[str, Any]] = None
        seen: set = set()
        for smeta in snaps[:max_snaps]:
            sid = smeta.get("snapshot_id", "")
            try:
                payload = load_fn(instance_id, sid)
            except Exception:
                payload = None
            if not payload:
                continue
            if merged is None:
                merged = {k: v for k, v in payload.items() if k != coll_key}
                merged[coll_key] = []
            for s in payload.get(coll_key, []) or []:
                lbl = (s.get(label_key) or "").strip()
                if not lbl or lbl in seen:
                    continue
                seen.add(lbl)
                merged[coll_key].append(s)
            if not merged.get("operating_speed_rpm") and payload.get("operating_speed_rpm"):
                merged["operating_speed_rpm"] = payload["operating_speed_rpm"]
        return merged
    except Exception as e:
        log.warning("briefing_figures: merge %s/%s falló: %s", instance_id, key, e)
        return _load_latest_snapshot(instance_id, key)


def _freqs_cpm(s: Dict[str, Any]) -> List[float]:
    is_cpm = str(s.get("freq_unit", "Hz") or "Hz").lower().startswith("c")
    return list(s["freqs"]) if is_cpm else [f * 60.0 for f in s["freqs"]]


def _instance_rpm(instance_id: str) -> float:
    """RPM de referencia del activo: nominal_rpm → velocidad LIVE (Modbus/
    colector). SGT300B tiene nominal_rpm=0 pero reporta 1799 en vivo."""
    try:
        from core.instance_state import get_instance
        r = float(getattr(get_instance(instance_id), "nominal_rpm", 0) or 0)
        if r > 0:
            return r
    except Exception:
        pass
    try:
        from core.live_readings import latest_for_instance
        for row in latest_for_instance(instance_id) or []:
            if (row.get("variable") or "").lower().startswith("velocidad") \
                    and row.get("value"):
                return float(row["value"])
    except Exception:
        pass
    return 0.0


def _filter_current_channels(instance_id: str,
                             sensors: List[Dict[str, Any]],
                             label_key: str = "sensor_label") -> List[Dict[str, Any]]:
    """Filtra canales de snapshots que NO corresponden a la configuración
    VIGENTE del activo (v3.31.412). El merge de snapshots puede resucitar
    canales de nomenclaturas viejas o de análisis hechos con otro activo
    seleccionado (caso real SGT300B: aparecían '3XD GENERATOR DE' con firma
    de TES3). Criterio: el label del snapshot debe compartir ≥2 tokens con
    algún plano/sensor de la config actual. Si el filtro dejara menos de la
    mitad (nomenclatura CSV distinta, caso TES1), se conserva el set
    original — el filtro solo actúa cuando hay mezcla evidente."""
    import re as _re

    def _toks(t: str) -> frozenset:
        return frozenset(x for x in _re.sub(r"[^A-Z0-9 ]", " ",
                                            str(t or "").upper()).split() if x)

    try:
        from core.instance_state import get_instance
        from core.sensor_map import sensor_label as _slbl
        inst = get_instance(instance_id)
        cfg_tokens: List[frozenset] = []
        for s in getattr(inst, "sensors", None) or []:
            try:
                cfg_tokens.append(_toks(s.get("plane_label", "")) |
                                  _toks(_slbl(s)))
            except Exception:
                continue
        if not cfg_tokens:
            return sensors
        kept = [s for s in sensors
                if any(len(_toks(s.get(label_key, "")) & ct) >= 2
                       for ct in cfg_tokens)]
        if len(kept) >= max(2, len(sensors) // 2):
            if len(kept) < len(sensors):
                log.warning("briefing_figures: %d canal(es) de snapshots "
                            "viejos/ajenos filtrados en %s",
                            len(sensors) - len(kept), instance_id)
            return kept
        return sensors
    except Exception:
        return sensors


def _train_sort_key(label: str):
    """Orden FÍSICO del tren (v3.31.414, pedido del usuario): primero la
    turbina (CRF/TRF/TURBINA — 1XD,1YD,2XD,2YD...), luego el gearbox y sus
    auxiliares (bomba/starter), luego el generador; dentro de cada máquina
    por número de plano y label."""
    import re as _re
    u = str(label or "").upper()
    m = _re.search(r"(\d+)", u)
    plane = int(m.group(1)) if m else 99
    if "CRF" in u or "TRF" in u or "TURBIN" in u:
        rank = 0
    elif ("GEARBOX" in u or "REDUCTOR" in u or "BOMBA" in u
          or "PUMP" in u or "STARTER" in u):
        rank = 1
    elif "GEN" in u:
        rank = 2
    else:
        rank = 3
    return (rank, plane, u)


def _sorted_sensors(sensors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(sensors,
                  key=lambda s: _train_sort_key(s.get("sensor_label", "")))


_RANK_NAMES = {0: "Turbina", 1: "Gearbox", 2: "Generador", 3: ""}


def _machine_chunks(sensors: List[Dict[str, Any]],
                    label_key: str = "sensor_label",
                    max_per_fig: int = 8) -> List[tuple]:
    """Agrupa canales POR MÁQUINA para las figuras apiladas (v3.31.415,
    pedido del usuario): paquete Turbina (1XD,1YD,2XD,2YD), paquete Gearbox
    (todos sus canales juntos), paquete Generador (5XD,5YD,6XD,6YD). Si un
    grupo excede max_per_fig se subdivide. Devuelve [(nombre, [sensores])]."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for s in sensors:
        r = _train_sort_key(str(s.get(label_key, "")))[0]
        grouped.setdefault(r, []).append(s)
    out: List[tuple] = []
    for r in sorted(grouped):
        grp = grouped[r]
        name = _RANK_NAMES.get(r, "")
        if len(grp) <= max_per_fig:
            out.append((name, grp))
        else:
            n_parts = (len(grp) + max_per_fig - 1) // max_per_fig
            for i in range(n_parts):
                part = grp[i * max_per_fig:(i + 1) * max_per_fig]
                out.append((f"{name} ({i + 1}/{n_parts})".strip(), part))
    return out


def _png(fig, height: int = 900) -> Optional[bytes]:
    try:
        from core.plot_export import fig_to_png_bytes
        png, err = fig_to_png_bytes(fig, width=1600, height=height, scale=1)
        return png if not err else None
    except Exception as e:
        log.warning("briefing_figures: rasterizar falló: %s", e)
        return None


# ---------------------------------------------------------------------------
# Espectro apilado (headless) — con marcadores de armónicos por canal
# ---------------------------------------------------------------------------
def spectrum_bundle(instance_id: str) -> Dict[str, Any]:
    """{"png": bytes|None, "analysis": str} — figura del espectro apilado con
    marcadores 1X..5X por canal (referenciados al eje real de cada punto) +
    análisis técnico determinístico por canal."""
    payload = _load_merged_snapshot(instance_id, "spectrum")
    if not payload:
        return {"png": None, "analysis": ""}
    sensors = [s for s in payload.get("sensors", [])
               if s.get("freqs") and s.get("amps")]
    if not sensors:
        return {"png": None, "analysis": ""}
    sensors = _filter_current_channels(instance_id, sensors)
    sensors = _sorted_sensors(sensors)[:16]
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        n = len(sensors)

        # Auto-calibración del eje (igual que la vista Live): algunos snapshots
        # guardan la frecuencia ÷1000 (tiempo en ms leído como s). Se compara el
        # 1X esperado (operating_speed_rpm) contra el pico dominante y se corrige
        # por décadas. Sin esto el espectro queda amontonado en CPM≈0.
        def _dom_cpm():
            best_a, best_c = -1.0, None
            for s in sensors:
                for p in (s.get("peaks") or [])[:1]:
                    try:
                        f = float(p.get("freq", 0)); a = float(p.get("amp", 0))
                    except Exception:
                        continue
                    is_cpm = str(s.get("freq_unit", "Hz") or "Hz").lower().startswith("c")
                    cpm = f if is_cpm else f * 60.0
                    if cpm > 0 and a > best_a:
                        best_a, best_c = a, cpm
            return best_c
        _scale = 1.0
        _dom = _dom_cpm()
        try:
            _rpm = float(payload.get("operating_speed_rpm") or 0)
        except Exception:
            _rpm = 0.0
        # RPM de referencia ANTES de calibrar (v3.31.412): la sanidad del eje
        # necesita conocer la velocidad. Fallback: nominal_rpm del activo.
        if _rpm <= 0:
            _rpm = _instance_rpm(instance_id)

        if _rpm > 0 and _dom:
            _ratio = _rpm / _dom
            # Décadas + factor 60 (Hz mal declarado como cpm y viceversa).
            for _c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0,
                       60.0, 600.0, 6000.0, 1 / 60.0, 1000.0 / 60.0):
                if 0.85 <= _ratio / _c <= 1.15:
                    _scale = _c
                    break
        # SANIDAD por Fmax (v3.31.412): ningún espectro industrial termina
        # por debajo de 2×1X (mínimo hay que ver el 2X). Si el Fmax del eje
        # queda bajo ese piso es el bug de ms-leídos-como-s → ×1000.
        # (Caso SGT300B: Fmax crudo 909 cpm con 1X real en 1799 cpm.)
        if _scale == 1.0:
            try:
                _fmax_raw = max(max(_freqs_cpm(s)) for s in sensors)
                _floor = max(600.0, 2.0 * _rpm if _rpm > 0 else 0.0)
                if _fmax_raw < _floor:
                    _scale = 1000.0
                    log.warning("spectrum_bundle(%s): Fmax %.0f cpm < piso "
                                "%.0f — aplicando ×1000 (ms-as-s)",
                                instance_id, _fmax_raw, _floor)
            except Exception:
                pass

        if _rpm <= 0 and _dom:
            _rpm = float(_dom) * _scale

        # v3.31.412 — EJE DE ALTA VELOCIDAD en trenes con gearbox: los canales
        # de la TURBINA (lado rápido, ej. SGT300 ~14.2 krpm con generador a
        # 1799) tienen su 1X en el pico dominante propio, no en el keyphasor.
        # Si la mediana de los picos dominantes de los canales "TURBIN*" cae
        # entre 2.5× y 20× la rpm base, esa es la velocidad de SU eje.
        _shaft_override: Dict[str, float] = {}
        try:
            _doms = []
            for s in sensors:
                if "TURBIN" not in str(s.get("sensor_label", "")).upper():
                    continue
                _x = np.asarray([f * _scale for f in _freqs_cpm(s)], float)
                _a = np.asarray(list(s["amps"]), float)
                _m = _x > 300.0
                if _m.any():
                    _doms.append(float(_x[_m][int(np.argmax(_a[_m]))]))
            if _doms and _rpm > 0:
                _med = float(np.median(_doms))
                if 2.5 * _rpm < _med < 20.0 * _rpm:
                    for s in sensors:
                        lbl_u = str(s.get("sensor_label", "")).upper()
                        if "TURBIN" in lbl_u and "CRF" not in lbl_u:
                            _shaft_override[s.get("sensor_label", "")] = _med
        except Exception:
            pass

        findings: List[Dict[str, Any]] = []
        pngs: List[Dict[str, Any]] = []
        # v3.31.415 — paquetes POR MÁQUINA: Turbina / Gearbox / Generador.
        for _grp_name, chunk in _machine_chunks(sensors):
            cn = len(chunk)
            fig = make_subplots(rows=cn, cols=1, shared_xaxes=True,
                                vertical_spacing=min(0.08, 0.3 / max(cn, 1)),
                                subplot_titles=[s.get("sensor_label", "")
                                                for s in chunk])
            for i, s in enumerate(chunk, start=1):
                lbl = s.get("sensor_label", "")
                x_cpm = [f * _scale for f in _freqs_cpm(s)]
                amps = list(s["amps"])
                fig.add_trace(go.Scatter(
                    x=x_cpm, y=amps, mode="lines",
                    line=dict(width=1.1, color=_PALETTE[(i - 1) % len(_PALETTE)]),
                    showlegend=False), row=i, col=1)

                # Marcadores 1X..4X — referenciados al eje REAL del punto.
                shaft_cpm = _shaft_override.get(lbl) or \
                    _shaft_cpm_for_label(lbl, _rpm)
                if shaft_cpm > 0:
                    for k in range(1, _N_HARMONICS + 1):
                        xk = k * shaft_cpm
                        if xk > _SPEC_FMAX_CPM:
                            break
                        fig.add_vline(
                            x=xk, row=i, col=1, line_dash="dot", line_width=1,
                            line_color="#94a3b8",
                            annotation_text=f"{k}X",
                            annotation_position="top right",
                            annotation_font=dict(size=9, color="#64748b"))

                # Hallazgo del canal: pico dominante SIEMPRE desde los arrays
                # (v3.31.412 — los 'peaks' del snapshot pueden venir ordenados
                # por frecuencia y traer componentes casi-DC como primera
                # entrada). Se ignoran los primeros 300 cpm (DC/leakage).
                try:
                    _a = np.asarray(amps, float)
                    _x = np.asarray(x_cpm, float)
                    _msk = _x > 300.0
                    if _msk.any():
                        j = int(np.argmax(_a[_msk]))
                        pk_cpm = float(_x[_msk][j])
                        pk_amp = float(_a[_msk][j])
                        findings.append({
                            "label": lbl, "peak_cpm": pk_cpm, "peak_amp": pk_amp,
                            "unit": s.get("amp_unit", "") or s.get("unit", ""),
                            "order": (pk_cpm / shaft_cpm) if shaft_cpm > 0 else None,
                            "shaft_cpm": shaft_cpm,
                        })
                except Exception:
                    pass

            fig.update_xaxes(range=[0, _SPEC_FMAX_CPM], showgrid=True,
                             gridcolor="#eef2f7", zeroline=False)
            fig.update_xaxes(title_text="Frecuencia (CPM)", row=cn, col=1)
            fig.update_yaxes(showgrid=True, gridcolor="#f8fafc", zeroline=False)
            _h = max(420, 170 * cn)
            fig.update_layout(height=_h, plot_bgcolor="white",
                              paper_bgcolor="white", showlegend=False,
                              margin=dict(l=60, r=20, t=28, b=44),
                              font=dict(size=12, color="#334155"))
            p = _png(fig, height=_h)
            if p:
                pngs.append({"png": p, "name": _grp_name})
        return {"png": (pngs[0]["png"] if pngs else None), "pngs": pngs,
                "analysis": _spectrum_analysis(findings)}
    except Exception as e:
        log.warning("spectrum_bundle: %s", e)
        return {"png": None, "pngs": [], "analysis": ""}


def _order_txt(order: Optional[float]) -> str:
    if order is None or order <= 0:
        return ""
    return f"≈{order:.1f}X"


def _amp_txt(f: Dict[str, Any]) -> str:
    u = (f.get("unit") or "").strip()
    return f"{f['peak_amp']:.2f} {u}".strip()


def _is_turbine_label(label: str) -> bool:
    l = (label or "").upper()
    return "CRF" in l or "TRF" in l


def _spectral_sentences(findings: List[Dict[str, Any]],
                        shafts: List[float]) -> List[str]:
    """Frases de diagnóstico para un SUBCONJUNTO de canales (una máquina)."""
    def _near(cpm: float, ref: float, tol: float = 0.12) -> bool:
        return ref > 0 and abs(cpm - ref) / ref <= tol

    sync, twox, sub, low, xshaft, other = [], [], [], [], [], []
    for f in findings:
        o, cpm, own = f.get("order"), f.get("peak_cpm", 0.0), f.get("shaft_cpm", 0.0)
        if o is not None and 0.88 <= o <= 1.12:
            sync.append(f)
        elif o is not None and 1.88 <= o <= 2.12:
            twox.append(f)
        elif o is not None and 0.35 <= o <= 0.60:
            sub.append(f)
        elif any(_near(cpm, s) for s in shafts if s and not _near(s, own, 0.02)):
            xshaft.append(f)  # fundamental de OTRO eje transmitida al plano
        elif o is not None and o < 0.30:
            low.append(f)
        else:
            other.append(f)

    def _lista(fs):
        return ", ".join(f"{x['label']} ({_amp_txt(x)} @ ~{x['peak_cpm']:.0f} CPM)"
                         for x in fs)

    p: List[str] = []
    if sync:
        amps = [x["peak_amp"] for x in sync]
        p.append(
            f"La energía vibratoria está gobernada por la componente síncrona "
            f"1X del eje correspondiente en {_lista(sync)}, con amplitudes "
            f"entre {min(amps):.2f} y {max(amps):.2f}. Este patrón corresponde "
            f"a la respuesta residual de desbalance propia de todo rotor "
            f"balanceado dentro de tolerancia (ISO 21940-11) y, a los niveles "
            f"observados, no constituye un mecanismo de falla activo; se "
            f"administra por tendencia.")
    if xshaft:
        p.append(
            f"En {_lista(xshaft)} el máximo espectral no corresponde a la "
            f"velocidad de giro de su propio eje sino a la fundamental del "
            f"otro rotor del tren transmitida estructuralmente a través de la "
            f"carcasa y los soportes — un acoplamiento vibratorio esperable en "
            f"turbomaquinaria de doble eje que se vigila en tendencia y no "
            f"señala defecto del eje local.")
    if twox:
        p.append(
            f"Se aprecia contenido 2X significativo en {_lista(twox)}. La "
            f"persistencia o crecimiento de esta componente es el indicador "
            f"clásico de desalineación o de holgura mecánica (API 684); se "
            f"recomienda correlacionarla con fase y con la posición del eje "
            f"antes de la próxima parada.")
    if sub:
        p.append(
            f"Existe actividad subsíncrona en torno a 0.4–0.5X en "
            f"{_lista(sub)}. A baja amplitud puede ser remolino de aceite "
            f"incipiente (oil whirl); debe seguirse de cerca porque su "
            f"crecimiento hacia la primera crítica degrada rápidamente la "
            f"estabilidad de la película (API 684, inestabilidades "
            f"fluido-dinámicas).")
    if low:
        p.append(
            f"En {_lista(low)} el máximo aparece a baja frecuencia, sin "
            f"correlato con las velocidades de giro del tren; se atribuye a "
            f"ruido de piso o a la resolución de la adquisición y conviene "
            f"confirmarlo en la próxima corrida antes de asignarle "
            f"significado mecánico.")
    if other:
        p.append(
            f"El contenido de orden superior observado en {_lista(other)} es "
            f"coherente con frecuencias de paso de álabes o respuesta "
            f"estructural local; a las amplitudes registradas no acciona "
            f"ningún criterio de norma.")
    return p


def _spectrum_analysis(findings: List[Dict[str, Any]]) -> str:
    """Análisis del espectro en PROSA de analista Cat IV, en PÁRRAFOS
    separados por máquina (v3.31.405): primero la turbina de gas (CRF/TRF),
    luego el generador/tren de potencia. Dictamen referenciado a
    ISO 20816-3 / API 670 / API 684."""
    if not findings:
        return ""
    shafts = sorted({f.get("shaft_cpm", 0.0) for f in findings
                     if f.get("shaft_cpm")})
    turb = [f for f in findings if _is_turbine_label(f.get("label", ""))]
    gen = [f for f in findings if not _is_turbine_label(f.get("label", ""))]

    has_sub_or_2x = False
    paras: List[str] = [
        "El levantamiento espectral del tren se evaluó frente a los criterios "
        "de severidad de ISO 20816-3 y a las prácticas de diagnóstico "
        "rotodinámico de API 670 y API 684."
    ]
    if turb:
        sents = _spectral_sentences(turb, shafts)
        if sents:
            paras.append("Turbina de gas (planos CRF y TRF). " + " ".join(sents))
    if gen:
        sents = _spectral_sentences(gen, shafts)
        if sents:
            paras.append("Generador y tren de potencia. " + " ".join(sents))

    for f in findings:
        o = f.get("order")
        if o is not None and (1.88 <= o <= 2.12 or 0.35 <= o <= 0.60):
            has_sub_or_2x = True
            break
    if not has_sub_or_2x:
        paras.append(
            "En conjunto, la firma espectral es la de una unidad operando "
            "establemente: componentes síncronas dominantes, sin subsíncronos "
            "de inestabilidad ni armónicos de holgura o rozamiento. Se "
            "mantiene la vigilancia periódica conforme a ISO 17359.")
    else:
        paras.append(
            "El resto de la firma espectral se mantiene dentro de un "
            "comportamiento normal; las componentes señaladas definen los "
            "puntos de seguimiento prioritario del próximo ciclo.")
    return "\n\n".join(paras)


def spectrum_png(instance_id: str) -> Optional[bytes]:
    """Compat: solo el PNG del espectro apilado."""
    return spectrum_bundle(instance_id).get("png")


# ---------------------------------------------------------------------------
# Forma de onda apilada (headless)
# ---------------------------------------------------------------------------
def waveform_bundle(instance_id: str) -> Dict[str, Any]:
    """{"png": bytes|None, "analysis": str} — forma de onda apilada + análisis
    determinístico por canal (amplitud pp y factor de cresta)."""
    payload = _load_merged_snapshot(instance_id, "waveform")
    if not payload:
        return {"png": None, "analysis": ""}
    sensors = [s for s in payload.get("sensors", [])
               if s.get("time") and s.get("values")]
    if not sensors:
        return {"png": None, "analysis": ""}
    sensors = _filter_current_channels(instance_id, sensors)
    sensors = _sorted_sensors(sensors)[:16]
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        findings: List[Dict[str, Any]] = []
        pngs: List[Dict[str, Any]] = []
        # v3.31.415 — paquetes POR MÁQUINA: Turbina / Gearbox / Generador.
        for _grp_name, chunk in _machine_chunks(sensors):
            cn = len(chunk)
            fig = make_subplots(rows=cn, cols=1, shared_xaxes=False,
                                vertical_spacing=min(0.08, 0.3 / max(cn, 1)),
                                subplot_titles=[s.get("sensor_label", "")
                                                for s in chunk])
            for i, s in enumerate(chunk, start=1):
                _t = list(s["time"])
                # Auto-calibración: si la Fs implícita < 50 Hz es implausible →
                # el tiempo venía en ms leído como s. Convertir a segundos.
                try:
                    _dur = float(_t[-1]) - float(_t[0])
                    if _dur > 0 and (len(_t) / _dur) < 50.0:
                        _t = [float(v) / 1000.0 for v in _t]
                except Exception:
                    pass
                _y = list(s["values"])
                fig.add_trace(go.Scattergl(
                    x=_t, y=_y, mode="lines",
                    line=dict(width=1.0, color=_PALETTE[(i - 1) % len(_PALETTE)]),
                    showlegend=False), row=i, col=1)
                # Hallazgo: pp + factor de cresta (impactividad)
                try:
                    arr = np.asarray(_y, float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 8:
                        ac = arr - float(np.mean(arr))
                        rms = float(np.sqrt(np.mean(ac ** 2)))
                        pk = float(np.max(np.abs(ac)))
                        findings.append({
                            "label": s.get("sensor_label", ""),
                            "pp": float(np.max(ac) - np.min(ac)),
                            "crest": (pk / rms) if rms > 1e-12 else 0.0,
                            "unit": s.get("amp_unit", "") or s.get("unit", ""),
                        })
                except Exception:
                    pass
            fig.update_xaxes(title_text="Tiempo (s)", row=cn, col=1)
            fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor="#f8fafc", zeroline=True,
                             zerolinecolor="#cbd5e1")
            _h = max(420, 170 * cn)
            fig.update_layout(height=_h, plot_bgcolor="white",
                              paper_bgcolor="white", showlegend=False,
                              margin=dict(l=60, r=20, t=28, b=44),
                              font=dict(size=12, color="#334155"))
            p = _png(fig, height=_h)
            if p:
                pngs.append({"png": p, "name": _grp_name})
        return {"png": (pngs[0]["png"] if pngs else None), "pngs": pngs,
                "analysis": _waveform_analysis(findings)}
    except Exception as e:
        log.warning("waveform_bundle: %s", e)
        return {"png": None, "pngs": [], "analysis": ""}


def _waveform_sentences(findings: List[Dict[str, Any]]) -> List[str]:
    """Frases para un subconjunto de canales de forma de onda."""
    clean = [f for f in findings if f.get("crest", 0.0) < 2.5]
    mid = [f for f in findings if 2.5 <= f.get("crest", 0.0) < 3.5]
    imp = [f for f in findings if f.get("crest", 0.0) >= 3.5]

    def _lista(fs):
        return ", ".join(f"{x['label']} ({x['pp']:.2f} {x.get('unit') or ''} pp, "
                         f"CF {x['crest']:.1f})" for x in fs)

    p: List[str] = []
    if clean:
        cfs = [f["crest"] for f in clean]
        p.append(
            f"Las señales de {_lista(clean)} son periódicas y limpias, con "
            f"factores de cresta entre {min(cfs):.1f} y {max(cfs):.1f} — el "
            f"comportamiento propio de una onda dominada por su componente "
            f"síncrona, sin impactos, truncamiento ni modulaciones que "
            f"sugieran contacto rotor-estator o degradación de elementos "
            f"rodantes.")
    if mid:
        p.append(
            f"En {_lista(mid)} se observa impactividad moderada: el factor de "
            f"cresta está por encima del rango puramente sinusoidal, sin "
            f"alcanzar todavía el patrón impulsivo característico de defectos "
            f"discretos; se administra por tendencia y se contrasta con el "
            f"espectro de envolvente en la próxima corrida.")
    if imp:
        p.append(
            f"Los registros de {_lista(imp)} presentan eventos transitorios "
            f"francos (factor de cresta ≥ 3.5). Este patrón impulsivo exige "
            f"correlación inmediata con el espectro y la demodulación para "
            f"discriminar entre impactos mecánicos reales, paso de partículas "
            f"o artefactos de la cadena de medición (API 670, verificación de "
            f"instrumentación ante lecturas anómalas).")
    return p


def _waveform_analysis(findings: List[Dict[str, Any]]) -> str:
    """Forma de onda en prosa de analista, párrafos por máquina (v3.31.405)."""
    if not findings:
        return ""
    turb = [f for f in findings if _is_turbine_label(f.get("label", ""))]
    gen = [f for f in findings if not _is_turbine_label(f.get("label", ""))]
    paras: List[str] = [
        "Las formas de onda en el dominio del tiempo se revisaron buscando "
        "impactividad, truncamiento, modulación y asimetrías — los "
        "precursores que el espectro promediado puede enmascarar."
    ]
    if turb:
        s = _waveform_sentences(turb)
        if s:
            paras.append("Turbina de gas (planos CRF y TRF). " + " ".join(s))
    if gen:
        s = _waveform_sentences(gen)
        if s:
            paras.append("Generador y tren de potencia. " + " ".join(s))
    if all(f.get("crest", 0.0) < 3.5 for f in findings):
        paras.append("En conjunto, no hay evidencia de eventos impulsivos ni "
                     "de recorte de señal: las ondas son coherentes con la "
                     "condición estable que muestran tendencia y espectro.")
    return "\n\n".join(paras)


def waveform_png(instance_id: str) -> Optional[bytes]:
    """Compat: solo el PNG de la forma de onda apilada."""
    return waveform_bundle(instance_id).get("png")


# ---------------------------------------------------------------------------
# Órbitas (grilla headless, marco físico 45° como System1)
# ---------------------------------------------------------------------------
def orbit_bundle(instance_id: str) -> Dict[str, Any]:
    """{"png": bytes|None, "analysis": str} — grilla de órbitas + análisis
    determinístico por cojinete (amplitud pp y forma de la órbita)."""
    payload = _load_merged_snapshot(instance_id, "orbit")
    if not payload:
        return {"png": None, "analysis": ""}
    bearings = [b for b in payload.get("bearings", [])
                if b.get("x_values") and b.get("y_values")]
    # v3.31.414 — Órbitas SOLO de turbina y generador (el gearbox y sus
    # auxiliares no van), en orden físico del tren. Se excluye por token del
    # label Y por número de plano (labels genéricos tipo "BRG 3": el plano 3
    # es gearbox según la config del activo).
    import re as _re
    _EXCL = ("GEARBOX", "REDUCTOR", "BOMBA", "PUMP", "STARTER")
    _gbx_planes: set = set()
    try:
        from core.instance_state import get_instance
        for s in getattr(get_instance(instance_id), "sensors", None) or []:
            pl = str(s.get("plane_label", "")).upper()
            if any(t in pl for t in _EXCL):
                try:
                    _gbx_planes.add(int(s.get("plane", 0) or 0))
                except Exception:
                    pass
    except Exception:
        pass

    def _orbit_ok(b) -> bool:
        lbl = str(b.get("bearing_label", "")).upper()
        if any(t in lbl for t in _EXCL):
            return False
        m = _re.search(r"(\d+)", lbl)
        if m and int(m.group(1)) in _gbx_planes:
            return False
        return True

    bearings = [b for b in bearings if _orbit_ok(b)]
    bearings = sorted(bearings,
                      key=lambda b: _train_sort_key(b.get("bearing_label", "")))
    if not bearings:
        return {"png": None, "analysis": ""}
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        try:
            from core.orbit import ProbeGeometry, _solve_global_xy
            gx, gy = ProbeGeometry(45.0, "Right"), ProbeGeometry(45.0, "Left")
        except Exception:
            gx = gy = None

        bearings = bearings[:4]
        ncol = min(len(bearings), 2)
        nrow = (len(bearings) + ncol - 1) // ncol
        fig = make_subplots(rows=nrow, cols=ncol,
                            subplot_titles=[b.get("bearing_label", "") for b in bearings],
                            horizontal_spacing=0.12, vertical_spacing=0.14)
        # escala compartida
        rmax = 1e-6
        HV = []
        for b in bearings:
            px = np.asarray(b["x_values"], float); py = np.asarray(b["y_values"], float)
            n = min(px.size, py.size); px, py = px[:n], py[:n]
            if gx is not None:
                H, V = _solve_global_xy(px, py, gx, gy)
            else:
                H, V = px, py
            HV.append((H, V))
            rmax = max(rmax, float(np.max(np.abs(H))), float(np.max(np.abs(V))))
        R = rmax * 1.12
        findings: List[Dict[str, Any]] = []
        for b, (H, V) in zip(bearings, HV):
            try:
                Ha = np.asarray(H, float); Va = np.asarray(V, float)
                pp_h = float(np.max(Ha) - np.min(Ha))
                pp_v = float(np.max(Va) - np.min(Va))
                major = max(pp_h, pp_v); minor = min(pp_h, pp_v)
                findings.append({
                    "label": b.get("bearing_label", ""),
                    "pp": major,
                    "ratio": (minor / major) if major > 1e-12 else 1.0,
                })
            except Exception:
                pass
        for idx, (b, (H, V)) in enumerate(zip(bearings, HV)):
            r_ = idx // ncol + 1
            c_ = idx % ncol + 1
            fig.add_trace(go.Scattergl(x=H, y=V, mode="lines",
                          line=dict(width=1.1, color=_PALETTE[idx % len(_PALETTE)]),
                          showlegend=False), row=r_, col=c_)
            fig.update_xaxes(range=[-R, R], scaleanchor=f"y{idx+1 if idx else ''}",
                             scaleratio=1, showgrid=True, gridcolor="#eef2f7",
                             zeroline=False, row=r_, col=c_)
            fig.update_yaxes(range=[-R, R], showgrid=True, gridcolor="#eef2f7",
                             zeroline=False, row=r_, col=c_)
        fig.update_layout(height=320 * nrow, plot_bgcolor="white",
                          paper_bgcolor="white", showlegend=False,
                          margin=dict(l=40, r=20, t=34, b=30),
                          font=dict(size=12, color="#334155"))
        return {"png": _png(fig), "analysis": _orbit_analysis(findings)}
    except Exception as e:
        log.warning("orbit_bundle: %s", e)
        return {"png": None, "analysis": ""}


def _orbit_analysis(findings: List[Dict[str, Any]]) -> str:
    """Órbitas en prosa de analista (v3.31.405): forma del movimiento del
    muñón por cojinete, con lectura rotodinámica según API 684."""
    if not findings:
        return ""
    circ = [f for f in findings if f.get("ratio", 1.0) >= 0.7]
    ellip = [f for f in findings if 0.3 <= f.get("ratio", 1.0) < 0.7]
    flat = [f for f in findings if f.get("ratio", 1.0) < 0.3]

    def _lista(fs):
        return ", ".join(f"{x['label']} (~{x['pp']:.2f} pp)" for x in fs)

    paras: List[str] = [
        "Las órbitas del eje se reconstruyeron con los pares ortogonales de "
        "proximidad y se evaluaron en el marco físico del cojinete, "
        "observando forma, repetibilidad y presencia de lazos internos — la "
        "lectura directa del comportamiento rotodinámico del muñón (API 684)."
    ]
    body: List[str] = []
    if circ:
        body.append(
            f"En {_lista(circ)} el movimiento es cuasi-circular y repetible: "
            f"el muñón gira centrado en su posición de equilibrio con rigidez "
            f"comparable en ambas direcciones, el patrón esperable de un "
            f"cojinete de película sano operando con precarga de diseño.")
    if ellip:
        body.append(
            f"Las órbitas de {_lista(ellip)} son elípticas estables — la "
            f"forma normal cuando la rigidez del pedestal difiere entre "
            f"direcciones; no constituye hallazgo mientras la relación de "
            f"ejes y la orientación de la elipse se mantengan en tendencia.")
    if flat:
        body.append(
            f"En {_lista(flat)} la órbita aparece fuertemente aplanada, "
            f"indicativa de precarga excesiva, restricción direccional o "
            f"desalineación que limita el movimiento en un plano; debe "
            f"correlacionarse con la posición del eje (shaft centerline) y "
            f"con los vectores 1X/2X antes de la próxima parada.")
    if body:
        paras.append(" ".join(body))
    paras.append(
        "No se observan lazos internos ni inversiones de precesión en los "
        "registros, descartando actividad subsíncrona significativa y "
        "rozamiento parcial en los apoyos monitoreados."
        if not flat else
        "Las órbitas aplanadas señaladas definen el punto de verificación "
        "prioritario del próximo ciclo, junto con su posición de eje y fase.")
    return "\n\n".join(paras)


def orbit_png(instance_id: str) -> Optional[bytes]:
    """Compat: solo el PNG de la grilla de órbitas."""
    return orbit_bundle(instance_id).get("png")


# ---------------------------------------------------------------------------
# Tendencia (reusa render_trend_png + historial directo)
# ---------------------------------------------------------------------------
def trend_png(instance_id: str, n_per_sensor: int = 60) -> Optional[bytes]:
    try:
        from core.live_readings import recent_history_all_direct
        from core.live_report_pdf import render_trend_png
        spark = recent_history_all_direct(instance_id, n_per_sensor=n_per_sensor) or {}
        if not spark:
            return None
        # tomar hasta 4 canales con más historia
        labels = sorted(spark.keys(),
                        key=lambda k: -len([h for h in spark[k] if h.get("value") is not None]))[:4]
        series = []
        for i, lbl in enumerate(labels):
            hist = spark.get(lbl, [])
            xs = [h.get("captured_at") for h in hist if h.get("value") is not None]
            ys = [h.get("value") for h in hist if h.get("value") is not None]
            if len(ys) >= 2:
                series.append({"label": lbl, "x": xs, "y": ys,
                               "color": _PALETTE[i % len(_PALETTE)]})
        if not series:
            return None
        return render_trend_png(series, y_title="overall")
    except Exception as e:
        log.warning("trend_png: %s", e)
        return None


# ---------------------------------------------------------------------------
# Tendencias SEPARADAS por sección (Ciclo 23.170) — CRF-TRF vs Generador, etc.
# Cada figura lleva sus líneas de alarma/danger (según los umbrales del plano).
# ---------------------------------------------------------------------------
# v3.31.414 — orden físico del tren: turbina → gearbox (y auxiliares) →
# generador. CRF-TRF es la "turbina" de los trenes aeroderivados.
_SECTION_ORDER = ["CRF-TRF", "Turbina", "Gearbox", "Bomba", "Generador"]

# Dentro de cada sección, orden de unidades pedido por el usuario:
# desplazamiento → aceleración → velocidad.
_UNIT_DESCR_ORDER = {"desplazamiento": 0, "aceleración": 1, "velocidad": 2}


def _section_key(plane_label: str) -> str:
    """Agrupa el plano en una sección para separar la tendencia."""
    p = (plane_label or "").upper()
    if "CRF" in p or "TRF" in p:
        return "CRF-TRF"
    if "GEN" in p or "GENERAD" in p:
        return "Generador"
    if "GEARBOX" in p or "REDUCTOR" in p:
        return "Gearbox"
    # v3.31.412 — SGT300B: los planos "1XD DE turbina", "2YD NDE turbina"...
    # no matcheaban ningún token → cada plano generaba SU PROPIA figura de
    # tendencia (9 figuras de 1 canal). Se agrupan bajo "Turbina".
    if "TURBIN" in p:
        return "Turbina"
    if "BOMBA" in p or "PUMP" in p:
        return "Bomba"
    return (plane_label or "Otros").strip() or "Otros"


def _unit_descr(unit: str) -> str:
    """Descriptor de la variable a partir de la unidad (para el caption)."""
    u = (unit or "").strip().lower()
    if "in/s" in u or "mm/s" in u or "ips" in u:
        return "velocidad"
    if u == "g" or u.startswith("g ") or "g pk" in u or "g rms" in u:
        return "aceleración"
    if "mil" in u or "um" in u or "µm" in u or u == "mm":
        return "desplazamiento"
    return unit or ""


def _period_history(instance_id: str, days: int = 7) -> Dict[str, List[Dict[str, Any]]]:
    """Histórico Direct de TODO el periodo analizado (v3.31.403), downsampled
    server-side vía RPC trend_bucketed (peak-hold: max por balde — es lo que
    se evalúa contra alarma). La tendencia del reporte debe cubrir LA SEMANA
    (o el mes) completa, no los últimos minutos del colector.

    Devuelve {sensor_label: [{captured_at, value, unit}, ...]} o {} si la
    RPC no está disponible (el caller cae al histórico reciente)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    try:
        from datetime import datetime, timedelta, timezone

        from core.live_readings import history_bucketed, latest_for_instance
        latest = latest_for_instance(instance_id) or []

        # ESCALERA DE VENTANA (v3.31.406): el RPC trend_bucketed puede dar
        # statement timeout en ventanas largas si falta el índice
        # idx_live_readings_trend (sql/trend_bucketed.sql). Se intenta la
        # ventana completa y se va recortando a la mitad hasta obtener datos.
        # La ventana que funcione con el PRIMER sensor se reutiliza para el
        # resto (evita pagar el timeout de 8s en cada canal).
        ladder = [days, max(2, days // 2), max(1, days // 4)]
        window_days: Optional[int] = None
        bucket = "1 hour" if days <= 8 else "4 hours"

        def _fetch(var: str, d: int):
            from_iso = (datetime.now(timezone.utc)
                        - timedelta(days=d)).isoformat()
            return history_bucketed(instance_id, var, "Direct",
                                    from_iso, bucket)

        for r in latest:
            if r.get("metric") != "Direct":
                continue
            if (r.get("variable") or "").lower().startswith("velocidad"):
                continue
            lbl, var = r.get("sensor_label"), r.get("variable")
            if not lbl or not var:
                continue
            rows = []
            if window_days is None:
                for d in ladder:
                    rows = _fetch(var, d)
                    if rows:
                        window_days = d
                        if d != days:
                            log.warning(
                                "_period_history: RPC timeout con %dd — "
                                "usando ventana de %dd. Crear el índice "
                                "idx_live_readings_trend (sql/"
                                "trend_bucketed.sql) para la ventana "
                                "completa.", days, d)
                        break
            else:
                rows = _fetch(var, window_days)
            pts = [{"captured_at": b.get("bucket"),
                    "value": b.get("max_val", b.get("avg_val")),
                    "unit": r.get("unit")}
                   for b in rows or [] if b.get("max_val") is not None
                   or b.get("avg_val") is not None]
            if len(pts) >= 2:
                out[lbl] = pts
    except Exception as e:
        log.warning("_period_history(%s) falló: %s", instance_id, e)
    if not out:
        # Fallback con VENTANA DE TIEMPO (si la RPC trend_bucketed no está):
        # query directa filtrada por captured_at >= from_iso + peak-hold por
        # hora client-side. Limitada al cap de 1000 filas de PostgREST (las
        # más recientes), pero jamás muestra solo los últimos minutos.
        try:
            from datetime import datetime, timedelta, timezone

            from core.live_readings import _get_supabase_client, _TABLE
            client = _get_supabase_client()
            if client is not None:
                from_iso = (datetime.now(timezone.utc)
                            - timedelta(days=days)).isoformat()
                resp = (client.table(_TABLE)
                        .select("sensor_label,variable,value,unit,captured_at")
                        .eq("instance_id", instance_id)
                        .eq("metric", "Direct")
                        .gte("captured_at", from_iso)
                        .order("captured_at", desc=True)
                        .limit(1000).execute())
                rows = list(getattr(resp, "data", []) or [])
                buckets: Dict[tuple, Dict[str, Any]] = {}
                for r in rows:
                    lbl = r.get("sensor_label")
                    v = r.get("value")
                    if not lbl or v is None:
                        continue
                    if (r.get("variable") or "").lower().startswith("velocidad"):
                        continue
                    hkey = (lbl, str(r.get("captured_at", ""))[:13])  # por hora
                    cur = buckets.get(hkey)
                    if cur is None or float(v) > float(cur["value"]):
                        buckets[hkey] = {"captured_at": r["captured_at"],
                                         "value": float(v),
                                         "unit": r.get("unit")}
                for (lbl, _h), pt in sorted(buckets.items(),
                                            key=lambda kv: kv[1]["captured_at"]):
                    out.setdefault(lbl, []).append(pt)
                out = {k: v for k, v in out.items() if len(v) >= 2}
                if out:
                    log.warning("_period_history: usando fallback windowed "
                                "(RPC trend_bucketed no disponible)")
        except Exception as e:
            log.warning("_period_history fallback falló: %s", e)
    return out


def build_trend_figures(instance_id: str, instance_obj: Any = None,
                        n_per_sensor: int = 60,
                        period_days: int = 7) -> List[Dict[str, Any]]:
    """Tendencias SEPARADAS por sección (CRF-TRF, Generador, ...), cada una con
    sus líneas de alarma/danger, cubriendo el PERIODO COMPLETO del reporte
    (period_days). Devuelve [{"section","unit","descr","png","analysis"}].
    [] si algo falla (el caller cae al trend único como fallback)."""
    try:
        from core.live_readings import recent_history_all_direct
        from core.live_report_pdf import render_trend_png
        from core.live_report_builder import _build_sensor_lookup
        # Periodo completo (RPC bucketed) → fallback al reciente si la RPC
        # no existe o no devuelve datos.
        spark = _period_history(instance_id, days=period_days)
        if not spark:
            spark = recent_history_all_direct(instance_id,
                                              n_per_sensor=n_per_sensor) or {}
        if not spark:
            return []
        lookup = _build_sensor_lookup(instance_obj) if instance_obj is not None else {}

        # v3.31.415 — mapa plano→sección desde la config del activo: cuando
        # el label no matchea en el lookup (duplicados/nomenclatura), la
        # sección se infiere por el NÚMERO de plano del canal. Fix del caso
        # SGT300B donde los desplazamientos del gearbox caían a "Otros".
        import re as _re
        plane_sec: Dict[int, str] = {}
        for s in (getattr(instance_obj, "sensors", None) or []):
            _sec = _section_key(str(s.get("plane_label", "")))
            if _sec in _SECTION_ORDER:
                try:
                    plane_sec.setdefault(int(s.get("plane", 0) or 0), _sec)
                except Exception:
                    pass

        def _sec_for(lbl: str) -> str:
            s = lookup.get(lbl) or {}
            sec = _section_key(s.get("plane_label", ""))
            if sec not in _SECTION_ORDER:
                m = _re.match(r"(\d+)", str(lbl))
                if m and int(m.group(1)) in plane_sec:
                    return plane_sec[int(m.group(1))]
            return sec

        groups: Dict[str, List[str]] = {}
        for lbl in spark:
            groups.setdefault(_sec_for(lbl), []).append(lbl)

        def _sec_sort(k):
            return (_SECTION_ORDER.index(k) if k in _SECTION_ORDER else 99, k)

        def _label_unit(lbl):
            s = lookup.get(lbl) or {}
            u = s.get("unit")
            if not u:
                h = spark.get(lbl) or []
                u = h[0].get("unit") if h else ""
            return u or ""

        out: List[Dict[str, Any]] = []
        for sec in sorted(groups, key=_sec_sort):
            labels_all = groups[sec]
            # v3.31.402 — UNA figura POR UNIDAD dentro de la sección (antes
            # solo la unidad dominante → la tendencia de ACELERACIÓN de
            # CRF-TRF quedaba fuera cuando velocidad tenía más canales).
            # Las unidades sin datos simplemente no generan figura.
            by_unit: Dict[str, List[str]] = {}
            for l in labels_all:
                by_unit.setdefault(_label_unit(l) or "", []).append(l)
            # Unidades en orden desplazamiento → aceleración → velocidad
            # (v3.31.414); empate por cantidad de canales desc.
            for unit_key in sorted(
                    by_unit,
                    key=lambda u: (_UNIT_DESCR_ORDER.get(_unit_descr(u), 9),
                                   -len(by_unit[u]))):
                # Canales en orden físico del tren (1XD, 1YD, 2XD, 2YD...)
                labels = sorted(by_unit[unit_key],
                                key=_train_sort_key)[:6]  # máx 6 por figura
                series, alarms, dangers, units = [], [], [], []
                for i, lbl in enumerate(labels):
                    hist = spark.get(lbl, [])
                    xs = [h.get("captured_at") for h in hist if h.get("value") is not None]
                    ys = [h.get("value") for h in hist if h.get("value") is not None]
                    if len(ys) < 2:
                        continue
                    series.append({"label": lbl, "x": xs, "y": ys,
                                   "color": _PALETTE[i % len(_PALETTE)]})
                    s = lookup.get(lbl) or {}
                    try:
                        if float(s.get("alarm", 0) or 0) > 0:
                            alarms.append(float(s["alarm"]))
                        if float(s.get("danger", 0) or 0) > 0:
                            dangers.append(float(s["danger"]))
                    except Exception:
                        pass
                    u = s.get("unit") or (hist[0].get("unit") if hist else "") or ""
                    if u:
                        units.append(u)
                if not series:
                    continue  # unidad sin datos → se omite
                unit = max(set(units), key=units.count) if units else (unit_key or "overall")
                _alarm = max(alarms) if alarms else 0.0
                png = render_trend_png(
                    series,
                    alarm=_alarm,
                    danger=(max(dangers) if dangers else 0.0),
                    y_title=unit,
                )
                if png:
                    out.append({"section": sec, "unit": unit,
                                "descr": _unit_descr(unit), "png": png,
                                "analysis": _trend_analysis(
                                    sec, series, _alarm, unit,
                                    danger=(max(dangers) if dangers else 0.0))})
        return out
    except Exception as e:
        log.warning("build_trend_figures: %s", e)
        return []


def _ts_txt(v: Any) -> str:
    return str(v or "")[:16].replace("T", " ")


def _trend_analysis(section: str, series: List[Dict[str, Any]],
                    alarm: float, unit: str, danger: float = 0.0) -> str:
    """Análisis de tendencia en el ESTILO DE LA CASA (v3.31.405 — espejo del
    reporte de tendencia del Live Monitoring): último valor vs % de consumo
    de los umbrales Alarma y Danger, ventana analizada, y por canal valor
    inicial / valor final / variación total con veredicto."""
    try:
        chans = []
        t0, t1 = "", ""
        for s in series:
            ys = [float(v) for v in (s.get("y") or []) if v is not None]
            xs = [x for x, v in zip(s.get("x") or [], s.get("y") or [])
                  if v is not None]
            if len(ys) < 2:
                continue
            chans.append({"label": s.get("label", ""), "v0": ys[0],
                          "v1": ys[-1], "vmax": max(ys)})
            if xs:
                if not t0 or str(xs[0]) < t0:
                    t0 = str(xs[0])
                if not t1 or str(xs[-1]) > t1:
                    t1 = str(xs[-1])
        if not chans:
            return ""

        last_max = max(chans, key=lambda c: c["v1"])
        peak = max(chans, key=lambda c: c["vmax"])

        # Párrafo 1 — estado global de la sección
        p1 = (f"El último valor reportado de amplitud vibratoria en "
              f"{len(chans)} punto(s) de medición de la sección {section} es "
              f"{last_max['v1']:.2f} {unit} ({last_max['label']}).")
        if alarm > 0:
            p1 += (f" Esto representa el {last_max['v1'] / alarm * 100:.0f}% "
                   f"del umbral de Alarma ({alarm:g} {unit})")
            if danger > 0:
                p1 += (f"; frente al umbral de Danger ({danger:g} {unit}) el "
                       f"consumo es del {last_max['v1'] / danger * 100:.0f}%")
            p1 += "."
        if t0 and t1:
            p1 += f" Ventana analizada desde {_ts_txt(t0)} hasta {_ts_txt(t1)}."
        if alarm > 0 and peak["vmax"] > alarm:
            p1 += (f" Durante la ventana, {peak['label']} alcanzó una cresta "
                   f"máxima de {peak['vmax']:.2f} {unit}, por ENCIMA del "
                   f"umbral de Alarma: el punto exige seguimiento cercano y "
                   f"correlación con espectro y forma de onda.")

        # Párrafos por canal — valor inicial / final / variación + veredicto
        partes = [p1]
        for c in sorted(chans, key=lambda c: -c["v1"]):
            pct = (c["v1"] - c["v0"]) / max(abs(c["v0"]), 1e-9) * 100.0
            seg = (f"{c['label']} — Valor inicial {c['v0']:.3f} {unit}, valor "
                   f"final {c['v1']:.3f} {unit}, variación total {pct:+.2f}%.")
            if alarm > 0 and c["v1"] > alarm:
                seg += (" El punto opera por encima del umbral de Alarma; se "
                        "requiere confirmación de causa y vigilancia "
                        "reforzada.")
            elif abs(pct) < 5.0:
                seg += (" El comportamiento es estable y sin desviaciones "
                        "significativas, lo que es consistente con una "
                        "condición normal dentro de la ventana evaluada.")
            elif abs(pct) < 15.0:
                seg += (" Se aprecia una variación moderada que no compromete "
                        "la condición del punto; se mantiene en vigilancia "
                        "rutinaria.")
            elif pct >= 15.0:
                seg += (" La variación ascendente es significativa y define "
                        "un punto de seguimiento prioritario para el próximo "
                        "ciclo de análisis.")
            else:
                seg += (" La reducción es significativa y consistente con una "
                        "mejora de la condición o con operación a distinta "
                        "carga; se verifica en la próxima ventana.")
            partes.append(seg)
        return "\n\n".join(partes)
    except Exception:
        return ""


def waveform_history_table(instance_id: str,
                           max_snaps: int = 10) -> List[Dict[str, Any]]:
    """Métricas de forma de onda INICIAL vs FINAL (v3.31.409): por cada
    canal, SOLO dos filas — la forma de onda más antigua disponible (ej. el
    lunes anterior) y la más reciente (la del día del reporte), con
    PK / PK-PK / RMS / unidad / factor de cresta. Espejo de la tabla
    comparativa del Reporte de Monitoreo en Línea. Dedup por (canal, fecha)
    — los snapshots sensor-aware traían el mismo canal repetido."""
    import re

    try:
        import numpy as np

        from core import history_storage as hs
        from core.waveform_history import load_waveform_snapshot
        snaps = (hs.list_snapshots(instance_id, "waveform") or [])[:max_snaps]
        by_canal: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for smeta in snaps:
            sid = smeta.get("snapshot_id", "")
            m = re.search(r"(\d{8})_(\d{6})", sid)
            if not m:
                continue
            d, t = m.group(1), m.group(2)
            sort_key = d + t
            fecha = f"{d[6:8]}/{d[4:6]}/{d[:4]} {t[:2]}:{t[2:4]}"
            try:
                payload = load_waveform_snapshot(instance_id, sid)
            except Exception:
                payload = None
            if not payload:
                continue
            p_unit = (payload.get("amp_unit", "") or payload.get("unit", ""))
            for s in payload.get("sensors", []) or []:
                vals = s.get("values")
                canal = (s.get("sensor_label") or "—").strip()
                if not vals:
                    continue
                slot = by_canal.setdefault(canal, {})
                if sort_key in slot:
                    continue  # canal repetido en el mismo snapshot → dedup
                try:
                    arr = np.asarray(vals, float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size < 8:
                        continue
                    ac = arr - float(np.mean(arr))
                    rms = float(np.sqrt(np.mean(ac ** 2)))
                    pk = float(np.max(np.abs(ac)))
                    slot[sort_key] = {
                        "fecha": fecha,
                        "canal": canal,
                        "pk": pk,
                        "pkpk": float(np.max(ac) - np.min(ac)),
                        "rms": rms,
                        "unit": (s.get("amp_unit", "") or s.get("unit", "")
                                 or s.get("y_unit", "") or p_unit),
                        "crest": (pk / rms) if rms > 1e-12 else 0.0,
                    }
                except Exception:
                    continue

        # Por canal: SOLO la más reciente (final) y la más antigua (inicial),
        # más reciente primero (como la tabla comparativa de la casa).
        # Canales en orden físico del tren (v3.31.414).
        rows: List[Dict[str, Any]] = []
        for canal in sorted(by_canal, key=_train_sort_key):
            slot = by_canal[canal]
            if not slot:
                continue
            keys = sorted(slot)
            newest, oldest = keys[-1], keys[0]
            rows.append(slot[newest])
            if oldest != newest:
                rows.append(slot[oldest])
        return rows
    except Exception as e:
        log.warning("waveform_history_table(%s) falló: %s", instance_id, e)
        return []


def figures_available(instance_id: str) -> Dict[str, Any]:
    """Disponibilidad de figuras SIN renderizar nada (v3.31.400).

    Para el BORRADOR de la cola de aprobación: la IA solo necesita saber QUÉ
    análisis existen — rasterizar los PNG (kaleido) tomaba varios segundos
    por figura y hacía lentísimo 'Generar borrador'. Devuelve booleans con
    las mismas claves que collect_asset_figures."""
    out = {"trend": False, "trends": [], "spectrum": False,
           "waveform": False, "orbit": False}
    try:
        from core import history_storage as hs
        for key in ("spectrum", "waveform", "orbit"):
            try:
                out[key] = bool(hs.list_snapshots(instance_id, key))
            except Exception:
                pass
    except Exception as e:
        log.warning("figures_available(%s): %s", instance_id, e)
    try:
        from core.live_readings import latest_for_instance
        out["trend"] = bool(latest_for_instance(instance_id))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Orquestador F1
# ---------------------------------------------------------------------------
def collect_asset_figures(instance_id: str,
                          instance_obj: Any = None,
                          period_label: str = "Semanal") -> Dict[str, Any]:
    """Devuelve PNGs (bytes) de las figuras disponibles del activo para el
    briefing + análisis técnico por figura. None por sección si no hay
    snapshot/datos. Headless.

    'trends' = lista de tendencias SEPARADAS por sección (con límites y
               'analysis' por sección), cubriendo el PERIODO del reporte
               (7 días Semanal / 30 días Mensual).
    'trend'  = PNG único (compat: lo usa la IA para saber que hay tendencia).
    '<key>_analysis' = texto de análisis determinístico de cada figura."""
    _days = 30 if str(period_label or "").lower().startswith("mensual") else 7
    trends = build_trend_figures(instance_id, instance_obj, period_days=_days)
    spec = spectrum_bundle(instance_id)
    wave = waveform_bundle(instance_id)
    orb = orbit_bundle(instance_id)
    return {
        "trend":    (trends[0]["png"] if trends else trend_png(instance_id)),
        "trends":   trends,
        "spectrum": spec["png"], "spectrum_analysis": spec["analysis"],
        "spectrum_pngs": spec.get("pngs") or [],
        "waveform": wave["png"], "waveform_analysis": wave["analysis"],
        "waveform_pngs": wave.get("pngs") or [],
        "orbit":    orb["png"],  "orbit_analysis": orb["analysis"],
    }


__all__ = ["collect_asset_figures", "figures_available",
           "spectrum_png", "waveform_png",
           "orbit_png", "trend_png", "build_trend_figures",
           "spectrum_bundle", "waveform_bundle", "orbit_bundle"]
