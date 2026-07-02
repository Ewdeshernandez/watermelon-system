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


def _freqs_cpm(s: Dict[str, Any]) -> List[float]:
    is_cpm = str(s.get("freq_unit", "Hz") or "Hz").lower().startswith("c")
    return list(s["freqs"]) if is_cpm else [f * 60.0 for f in s["freqs"]]


def _sorted_sensors(sensors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        from core.channel_order import channel_sort_key
        return sorted(sensors, key=lambda s: channel_sort_key(
            s.get("sensor_label", ""),
            s.get("amp_unit", "") or s.get("unit", "")))
    except Exception:
        return sensors


def _png(fig) -> Optional[bytes]:
    try:
        from core.plot_export import fig_to_png_bytes
        png, err = fig_to_png_bytes(fig, width=1600, height=900, scale=1)
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
    payload = _load_latest_snapshot(instance_id, "spectrum")
    if not payload:
        return {"png": None, "analysis": ""}
    sensors = [s for s in payload.get("sensors", [])
               if s.get("freqs") and s.get("amps")]
    if not sensors:
        return {"png": None, "analysis": ""}
    sensors = _sorted_sensors(sensors)[:12]
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
        if _rpm > 0 and _dom:
            _ratio = _rpm / _dom
            for _c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
                if 0.85 <= _ratio / _c <= 1.15:
                    _scale = _c
                    break
        elif _dom:
            try:
                _fmax_raw = max(max(_freqs_cpm(s)) for s in sensors)
                if _fmax_raw < 600.0:
                    _scale = 1000.0
            except Exception:
                pass

        fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=[s.get("sensor_label", "") for s in sensors])
        findings: List[Dict[str, Any]] = []
        for i, s in enumerate(sensors, start=1):
            lbl = s.get("sensor_label", "")
            x_cpm = [f * _scale for f in _freqs_cpm(s)]
            amps = list(s["amps"])
            fig.add_trace(go.Scatter(
                x=x_cpm, y=amps, mode="lines",
                line=dict(width=1.1, color=_PALETTE[(i - 1) % len(_PALETTE)]),
                showlegend=False), row=i, col=1)

            # Marcadores de armónicos 1X..5X — referenciados al eje REAL del
            # punto (CRF → gas generator ~10200 cpm; resto → keyphasor).
            shaft_cpm = _shaft_cpm_for_label(lbl, _rpm)
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

            # Hallazgo del canal: pico dominante (para el análisis)
            try:
                pk_cpm, pk_amp = None, None
                pks = s.get("peaks") or []
                if pks:
                    _f = float(pks[0].get("freq", 0)); pk_amp = float(pks[0].get("amp", 0))
                    is_cpm = str(s.get("freq_unit", "Hz") or "Hz").lower().startswith("c")
                    pk_cpm = (_f if is_cpm else _f * 60.0) * _scale
                else:
                    _a = np.asarray(amps, float)
                    _x = np.asarray(x_cpm, float)
                    _msk = _x > 0
                    if _msk.any():
                        j = int(np.argmax(_a[_msk]))
                        pk_cpm = float(_x[_msk][j]); pk_amp = float(_a[_msk][j])
                if pk_cpm and pk_amp is not None:
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
        fig.update_xaxes(title_text="Frecuencia (CPM)", row=n, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#f8fafc", zeroline=False)
        fig.update_layout(height=max(260, 150 * n), plot_bgcolor="white",
                          paper_bgcolor="white", showlegend=False,
                          margin=dict(l=60, r=20, t=28, b=44),
                          font=dict(size=12, color="#334155"))
        return {"png": _png(fig), "analysis": _spectrum_analysis(findings)}
    except Exception as e:
        log.warning("spectrum_bundle: %s", e)
        return {"png": None, "analysis": ""}


def _order_txt(order: Optional[float]) -> str:
    if order is None or order <= 0:
        return ""
    return f"≈{order:.1f}X"


def _spectrum_analysis(findings: List[Dict[str, Any]]) -> str:
    """Análisis técnico determinístico del espectro apilado, por canal."""
    if not findings:
        return ""
    partes: List[str] = []
    n_sync = 0
    for f in findings:
        o = f.get("order")
        otxt = _order_txt(o)
        seg = (f"{f['label']}: pico dominante {f['peak_amp']:.2f} "
               f"{f.get('unit') or ''} a ~{f['peak_cpm']:.0f} CPM"
               + (f" ({otxt})" if otxt else ""))
        if o is not None:
            if 0.90 <= o <= 1.10:
                n_sync += 1
                seg += " — componente síncrona 1X, típica de energía residual de desbalance"
            elif 1.90 <= o <= 2.10:
                seg += " — componente 2X, vigilar alineación y holguras mecánicas"
            elif 0.35 <= o <= 0.55:
                seg += " — componente subsíncrona ~0.5X, vigilar estabilidad de película de aceite"
            elif o > 2.5:
                seg += " — componente de orden superior (paso de álabes/engrane u origen estructural)"
        partes.append(seg)
    cierre = ("El patrón espectral es predominantemente síncrono (1X dominante), "
              "consistente con operación estable sin defectos activos."
              if n_sync >= max(1, len(findings) // 2) else
              "Se recomienda seguimiento de las componentes no síncronas señaladas "
              "en próximas corridas.")
    return "Análisis del espectro: " + ". ".join(partes) + ". " + cierre


def spectrum_png(instance_id: str) -> Optional[bytes]:
    """Compat: solo el PNG del espectro apilado."""
    return spectrum_bundle(instance_id).get("png")


# ---------------------------------------------------------------------------
# Forma de onda apilada (headless)
# ---------------------------------------------------------------------------
def waveform_bundle(instance_id: str) -> Dict[str, Any]:
    """{"png": bytes|None, "analysis": str} — forma de onda apilada + análisis
    determinístico por canal (amplitud pp y factor de cresta)."""
    payload = _load_latest_snapshot(instance_id, "waveform")
    if not payload:
        return {"png": None, "analysis": ""}
    sensors = [s for s in payload.get("sensors", [])
               if s.get("time") and s.get("values")]
    if not sensors:
        return {"png": None, "analysis": ""}
    sensors = _sorted_sensors(sensors)[:12]
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        n = len(sensors)
        fig = make_subplots(rows=n, cols=1, shared_xaxes=False,
                            vertical_spacing=0.05,
                            subplot_titles=[s.get("sensor_label", "") for s in sensors])
        findings: List[Dict[str, Any]] = []
        for i, s in enumerate(sensors, start=1):
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
        fig.update_xaxes(title_text="Tiempo (s)", row=n, col=1)
        fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f8fafc", zeroline=True,
                         zerolinecolor="#cbd5e1")
        fig.update_layout(height=max(260, 150 * n), plot_bgcolor="white",
                          paper_bgcolor="white", showlegend=False,
                          margin=dict(l=60, r=20, t=28, b=44),
                          font=dict(size=12, color="#334155"))
        return {"png": _png(fig), "analysis": _waveform_analysis(findings)}
    except Exception as e:
        log.warning("waveform_bundle: %s", e)
        return {"png": None, "analysis": ""}


def _waveform_analysis(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return ""
    partes: List[str] = []
    n_impact = 0
    for f in findings:
        cf = f.get("crest", 0.0)
        seg = (f"{f['label']}: {f['pp']:.2f} {f.get('unit') or ''} pp, "
               f"factor de cresta {cf:.1f}")
        if cf >= 3.5:
            n_impact += 1
            seg += " — presencia de impactos/eventos transitorios, correlacionar con espectro"
        elif cf >= 2.5:
            seg += " — impactividad moderada"
        else:
            seg += " — señal periódica sin impactos evidentes"
        partes.append(seg)
    cierre = ("Las formas de onda son limpias y periódicas, sin evidencia de "
              "impactos ni truncamiento."
              if n_impact == 0 else
              "Los canales con factor de cresta elevado deben correlacionarse con "
              "el espectro y la tendencia para descartar defectos incipientes.")
    return "Análisis de la forma de onda: " + ". ".join(partes) + ". " + cierre


def waveform_png(instance_id: str) -> Optional[bytes]:
    """Compat: solo el PNG de la forma de onda apilada."""
    return waveform_bundle(instance_id).get("png")


# ---------------------------------------------------------------------------
# Órbitas (grilla headless, marco físico 45° como System1)
# ---------------------------------------------------------------------------
def orbit_bundle(instance_id: str) -> Dict[str, Any]:
    """{"png": bytes|None, "analysis": str} — grilla de órbitas + análisis
    determinístico por cojinete (amplitud pp y forma de la órbita)."""
    payload = _load_latest_snapshot(instance_id, "orbit")
    if not payload:
        return {"png": None, "analysis": ""}
    bearings = [b for b in payload.get("bearings", [])
                if b.get("x_values") and b.get("y_values")]
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
    if not findings:
        return ""
    partes: List[str] = []
    n_flat = 0
    for f in findings:
        r = f.get("ratio", 1.0)
        seg = f"{f['label']}: amplitud máxima ~{f['pp']:.2f} pp"
        if r < 0.3:
            n_flat += 1
            seg += (" — órbita muy aplanada, posible precarga/restricción "
                    "en una dirección (revisar alineación y holgura)")
        elif r < 0.7:
            seg += " — órbita elíptica, forma normal en cojinetes de película"
        else:
            seg += " — órbita cuasi-circular, movimiento balanceado del muñón"
        partes.append(seg)
    cierre = ("Las órbitas presentan formas estables y repetibles, sin lazos "
              "internos que sugieran componentes subsíncronas o rozamiento."
              if n_flat == 0 else
              "Las órbitas aplanadas deben correlacionarse con la posición del "
              "eje (SCL) y los vectores 1X/2X en próximas corridas.")
    return "Análisis de órbitas: " + ". ".join(partes) + ". " + cierre


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
_SECTION_ORDER = ["CRF-TRF", "Generador", "Gearbox"]


def _section_key(plane_label: str) -> str:
    """Agrupa el plano en una sección para separar la tendencia."""
    p = (plane_label or "").upper()
    if "CRF" in p or "TRF" in p:
        return "CRF-TRF"
    if "GEN" in p or "GENERAD" in p:
        return "Generador"
    if "GEARBOX" in p or "REDUCTOR" in p:
        return "Gearbox"
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


def build_trend_figures(instance_id: str, instance_obj: Any = None,
                        n_per_sensor: int = 60) -> List[Dict[str, Any]]:
    """Tendencias SEPARADAS por sección (CRF-TRF, Generador, ...), cada una con
    sus líneas de alarma/danger. Devuelve [{"section","unit","descr","png"}].
    [] si algo falla (el caller cae al trend único como fallback)."""
    try:
        from core.live_readings import recent_history_all_direct
        from core.live_report_pdf import render_trend_png
        from core.live_report_builder import _build_sensor_lookup
        spark = recent_history_all_direct(instance_id, n_per_sensor=n_per_sensor) or {}
        if not spark:
            return []
        lookup = _build_sensor_lookup(instance_obj) if instance_obj is not None else {}
        groups: Dict[str, List[str]] = {}
        for lbl in spark:
            s = lookup.get(lbl) or {}
            groups.setdefault(_section_key(s.get("plane_label", "")), []).append(lbl)

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
            # Unidad dominante de la sección → figura consistente (eje + límites
            # correctos). Evita mezclar p. ej. aceleración (g) con velocidad (in/s).
            _us = [_label_unit(l) for l in labels_all if _label_unit(l)]
            _dom_unit = max(set(_us), key=_us.count) if _us else ""
            labels = [l for l in labels_all if not _dom_unit or _label_unit(l) == _dom_unit]
            labels = sorted(
                labels,
                key=lambda k: -len([h for h in spark[k] if h.get("value") is not None]),
            )[:6]  # máx 6 canales por figura para no saturar
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
                continue
            unit = max(set(units), key=units.count) if units else "overall"
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
                            "analysis": _trend_analysis(sec, series, _alarm, unit)})
        return out
    except Exception as e:
        log.warning("build_trend_figures: %s", e)
        return []


def _trend_analysis(section: str, series: List[Dict[str, Any]],
                    alarm: float, unit: str) -> str:
    """Análisis determinístico de la tendencia de una sección.

    Criterios (v3.31.391):
      • El nivel de referencia es la CRESTA MÁXIMA del periodo (no el último
        punto), por canal.
      • La dirección compara la cresta máxima del tramo INICIAL vs la del
        tramo FINAL del canal con mayor nivel (>±15% = ascendente/descendente).
      • El porcentaje vs alarma se reporta con signo correcto: margen bajo el
        umbral, o EXCESO sobre el umbral cuando la cresta lo supera (nunca
        "margen del 0%").
    """
    try:
        # Cresta máxima del periodo por canal → canal dominante
        peak_val, peak_lbl, peak_ys = -1.0, "", []
        for s in series:
            ys = [float(v) for v in (s.get("y") or []) if v is not None]
            if ys and max(ys) > peak_val:
                peak_val, peak_lbl, peak_ys = max(ys), s.get("label", ""), ys
        if peak_val < 0:
            return ""

        # Dirección: cresta del primer tercio vs cresta del último tercio
        direction = "estable"
        if len(peak_ys) >= 6:
            n3 = max(2, len(peak_ys) // 3)
            crest_ini = max(peak_ys[:n3])
            crest_fin = max(peak_ys[-n3:])
            delta = (crest_fin - crest_ini) / max(abs(crest_ini), 1e-9)
            if delta > 0.15:
                direction = "ascendente"
            elif delta < -0.15:
                direction = "descendente"

        txt = (f"Análisis de tendencia — {section}: comportamiento {direction} "
               f"durante el periodo; cresta máxima {peak_val:.2f} {unit} "
               f"en {peak_lbl}")
        over_alarm = False
        if alarm > 0:
            if peak_val > alarm:
                over_alarm = True
                exceso = (peak_val - alarm) / alarm * 100.0
                txt += (f", que SUPERA el umbral de alarma ({alarm:g} {unit}) "
                        f"en {exceso:.0f}%")
            else:
                margen = (alarm - peak_val) / alarm * 100.0
                txt += (f", con margen del {margen:.0f}% bajo el umbral de "
                        f"alarma ({alarm:g} {unit})")
        txt += "."
        if over_alarm:
            txt += (" El canal opera por encima del umbral: correlacionar con "
                    "espectro y forma de onda, y dar seguimiento cercano hasta "
                    "confirmar la causa.")
        elif direction == "ascendente":
            txt += (" La pendiente ascendente amerita aumentar la frecuencia de "
                    "revisión y correlacionar con espectro y forma de onda.")
        else:
            txt += " Sin cambios de nivel que sugieran degradación activa."
        return txt
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Orquestador F1
# ---------------------------------------------------------------------------
def collect_asset_figures(instance_id: str,
                          instance_obj: Any = None) -> Dict[str, Any]:
    """Devuelve PNGs (bytes) de las figuras disponibles del activo para el
    briefing + análisis técnico por figura. None por sección si no hay
    snapshot/datos. Headless.

    'trends' = lista de tendencias SEPARADAS por sección (con límites y
               'analysis' por sección).
    'trend'  = PNG único (compat: lo usa la IA para saber que hay tendencia).
    '<key>_analysis' = texto de análisis determinístico de cada figura."""
    trends = build_trend_figures(instance_id, instance_obj)
    spec = spectrum_bundle(instance_id)
    wave = waveform_bundle(instance_id)
    orb = orbit_bundle(instance_id)
    return {
        "trend":    (trends[0]["png"] if trends else trend_png(instance_id)),
        "trends":   trends,
        "spectrum": spec["png"], "spectrum_analysis": spec["analysis"],
        "waveform": wave["png"], "waveform_analysis": wave["analysis"],
        "orbit":    orb["png"],  "orbit_analysis": orb["analysis"],
    }


__all__ = ["collect_asset_figures", "spectrum_png", "waveform_png",
           "orbit_png", "trend_png", "build_trend_figures",
           "spectrum_bundle", "waveform_bundle", "orbit_bundle"]
