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
# Espectro apilado (headless)
# ---------------------------------------------------------------------------
def spectrum_png(instance_id: str) -> Optional[bytes]:
    payload = _load_latest_snapshot(instance_id, "spectrum")
    if not payload:
        return None
    sensors = [s for s in payload.get("sensors", [])
               if s.get("freqs") and s.get("amps")]
    if not sensors:
        return None
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
        for i, s in enumerate(sensors, start=1):
            x_cpm = [f * _scale for f in _freqs_cpm(s)]
            fig.add_trace(go.Scatter(
                x=x_cpm, y=s["amps"], mode="lines",
                line=dict(width=1.1, color=_PALETTE[(i - 1) % len(_PALETTE)]),
                showlegend=False), row=i, col=1)
        fig.update_xaxes(range=[0, _SPEC_FMAX_CPM], showgrid=True,
                         gridcolor="#eef2f7", zeroline=False)
        fig.update_xaxes(title_text="Frecuencia (CPM)", row=n, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#f8fafc", zeroline=False)
        fig.update_layout(height=max(260, 150 * n), plot_bgcolor="white",
                          paper_bgcolor="white", showlegend=False,
                          margin=dict(l=60, r=20, t=28, b=44),
                          font=dict(size=12, color="#334155"))
        return _png(fig)
    except Exception as e:
        log.warning("spectrum_png: %s", e)
        return None


# ---------------------------------------------------------------------------
# Forma de onda apilada (headless)
# ---------------------------------------------------------------------------
def waveform_png(instance_id: str) -> Optional[bytes]:
    payload = _load_latest_snapshot(instance_id, "waveform")
    if not payload:
        return None
    sensors = [s for s in payload.get("sensors", [])
               if s.get("time") and s.get("values")]
    if not sensors:
        return None
    sensors = _sorted_sensors(sensors)[:12]
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        n = len(sensors)
        fig = make_subplots(rows=n, cols=1, shared_xaxes=False,
                            vertical_spacing=0.05,
                            subplot_titles=[s.get("sensor_label", "") for s in sensors])
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
            fig.add_trace(go.Scattergl(
                x=_t, y=list(s["values"]), mode="lines",
                line=dict(width=1.0, color=_PALETTE[(i - 1) % len(_PALETTE)]),
                showlegend=False), row=i, col=1)
        fig.update_xaxes(title_text="Tiempo (s)", row=n, col=1)
        fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#f8fafc", zeroline=True,
                         zerolinecolor="#cbd5e1")
        fig.update_layout(height=max(260, 150 * n), plot_bgcolor="white",
                          paper_bgcolor="white", showlegend=False,
                          margin=dict(l=60, r=20, t=28, b=44),
                          font=dict(size=12, color="#334155"))
        return _png(fig)
    except Exception as e:
        log.warning("waveform_png: %s", e)
        return None


# ---------------------------------------------------------------------------
# Órbitas (grilla headless, marco físico 45° como System1)
# ---------------------------------------------------------------------------
def orbit_png(instance_id: str) -> Optional[bytes]:
    payload = _load_latest_snapshot(instance_id, "orbit")
    if not payload:
        return None
    bearings = [b for b in payload.get("bearings", [])
                if b.get("x_values") and b.get("y_values")]
    if not bearings:
        return None
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
        return _png(fig)
    except Exception as e:
        log.warning("orbit_png: %s", e)
        return None


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
# Orquestador F1
# ---------------------------------------------------------------------------
def collect_asset_figures(instance_id: str) -> Dict[str, Optional[bytes]]:
    """Devuelve PNGs (bytes) de las figuras disponibles del activo para el
    briefing. None por sección si no hay snapshot/datos. Headless."""
    return {
        "trend":    trend_png(instance_id),
        "spectrum": spectrum_png(instance_id),
        "waveform": waveform_png(instance_id),
        "orbit":    orbit_png(instance_id),
    }


__all__ = ["collect_asset_figures", "spectrum_png", "waveform_png",
           "orbit_png", "trend_png"]
