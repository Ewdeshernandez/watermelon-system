"""
core.startup_figures — Figuras de ARRANQUE / RUN-UP y Full Spectrum para el
reporte semanal/mensual.

Dos familias:

  • Full Spectrum (steady, plena carga): se DERIVA del snapshot de órbita que ya
    alimenta la figura de órbita del reporte (onda cruda del robot, full-speed).
    Espectro direccional (precesión forward/backward) por cojinete con par X/Y.

  • Arranque / Run-up: Shaft Centerline + Bode + Polar de los snapshots MANUALES
    que el analista guarda desde la Rotodinámica (CSV de System1 del run-up), vía
    scl_history / bode_history / polar_history. Se toma el ÚLTIMO snapshot dentro
    de la ventana del reporte (7 d semanal / 30 d mensual). Si no hay en la
    ventana, la sección no aparece.

Headless (sin Streamlit). Nunca lanza hacia el reporte: devuelve listas vacías
ante cualquier fallo.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("startup_figures")

_NAVY = "#12305e"
_AMBER = "#e8890c"
_STEEL = "#274b7d"
_GRID = "#eef2f7"
_GEN_BEARINGS = {"5", "6"}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _png(fig, height: int = 760) -> Optional[bytes]:
    try:
        from core.plot_export import fig_to_png_bytes
        png, err = fig_to_png_bytes(fig, width=1600, height=height, scale=1)
        return png if not err else None
    except Exception as e:  # noqa: BLE001
        log.warning("startup_figures: rasterizar falló: %s", e)
        return None


def _ts_key(ts: str) -> str:
    return re.sub(r"[^0-9]", "", str(ts or ""))[:14]


def _within(ts_iso: str, days: int) -> bool:
    """True si el snapshot (timestamp ISO) cae dentro de la ventana de `days`."""
    k = _ts_key(ts_iso)
    if len(k) < 8:
        return False
    try:
        dt = datetime.strptime(k.ljust(14, "0"), "%Y%m%d%H%M%S")
    except Exception:  # noqa: BLE001
        return False
    return dt >= (datetime.now() - timedelta(days=days))


def _days_for(period_label: str) -> int:
    return 30 if str(period_label or "").lower().startswith("mensual") else 7


def _fecha_label(ts_iso: str) -> str:
    k = _ts_key(ts_iso)
    if len(k) >= 8:
        return f"{k[0:4]}-{k[4:6]}-{k[6:8]}"
    return ""


def _layout(fig, title: str, h: int = 540):
    fig.update_layout(template="plotly_white", height=h,
                      title=dict(text=title, font=dict(size=16, color="#0f172a")),
                      margin=dict(l=62, r=28, t=52, b=50),
                      font=dict(family="IBM Plex Sans", size=13, color="#334155"),
                      plot_bgcolor="white", paper_bgcolor="white",
                      showlegend=False)
    return fig


def _bid(label: str) -> str:
    m = re.search(r"\d+", str(label or ""))
    return m.group(0) if m else ""


def _machine_of_bearing(bid: str) -> str:
    if bid in _GEN_BEARINGS:
        return "Generador"
    if bid in ("1", "2"):
        return "Turbina"
    return "Gearbox"


# --------------------------------------------------------------------------- #
# Full Spectrum (steady, plena carga) — derivado del snapshot de órbita
# --------------------------------------------------------------------------- #
def _spk(unit: str):
    u = (unit or "").lower()
    if "mm/s" in u:
        return "vel", 300000
    if u.strip() in ("g", "g pk", "g rms") or "m/s2" in u or "m/s²" in u:
        return "accel", 600000
    return "disp", 60000


def fullspec_bundle(instance_id: str) -> List[Dict[str, Any]]:
    """Full Spectrum direccional por cojinete, derivado del snapshot de órbita
    (mismo origen que la figura de órbita: onda cruda del robot full-speed).
    Devuelve [{png, name}] (name = máquina para agrupar en el PDF)."""
    try:
        import plotly.graph_objects as go
        from core.briefing_figures import _load_merged_snapshot
    except Exception:  # noqa: BLE001
        return []
    try:
        payload = _load_merged_snapshot(instance_id, "orbit")
        if not payload:
            return []
        out: List[Dict[str, Any]] = []
        for b in payload.get("bearings", []):
            xv = np.asarray(b.get("x_values") or [], float)
            yv = np.asarray(b.get("y_values") or [], float)
            n = int(min(xv.size, yv.size))
            if n < 16:
                continue
            xv, yv = xv[:n], yv[:n]
            xv = xv - np.nanmean(xv)
            yv = yv - np.nanmean(yv)
            fs = float(b.get("sampling_rate_hz") or 0.0)
            if fs <= 0:
                continue
            unit = b.get("amp_unit") or "µm pp"
            kind, cpmx = _spk(unit)
            win = np.hanning(n)
            z = (xv * win) + 1j * (yv * win)
            sp = np.fft.fftshift(np.fft.fft(z)) * (2.0 / np.sum(win))
            fr = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
            cpm = fr * 60.0
            amp = np.abs(sp) * (2.0 if kind == "disp" else 1.0 / np.sqrt(2.0))
            fwd = cpm >= 0
            bl = b.get("bearing_label") or b.get("x_sensor_label") or ""
            bid = _bid(bl)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cpm[fwd], y=amp[fwd], mode="lines",
                line=dict(color=_NAVY, width=1.3),
                name="Forward (precesión directa +)"))
            fig.add_trace(go.Scatter(
                x=cpm[~fwd], y=amp[~fwd], mode="lines",
                line=dict(color=_AMBER, width=1.3),
                name="Reverse (precesión inversa −)"))
            # energía fwd vs bwd (dominancia de precesión)
            ef = float(np.sum(amp[fwd] ** 2))
            eb = float(np.sum(amp[~fwd] ** 2))
            tot = ef + eb or 1.0
            dom = "directa (forward)" if ef >= eb else "inversa (reverse)"
            fig.update_xaxes(title="Frecuencia (cpm) · − inversa | + directa",
                             range=[-cpmx, cpmx], showgrid=True, gridcolor=_GRID,
                             zeroline=True, zerolinecolor="#cbd5e1")
            fig.update_yaxes(title=f"Amplitud ({unit})", showgrid=True,
                             gridcolor=_GRID, rangemode="tozero")
            fig.update_layout(showlegend=True,
                              legend=dict(orientation="h", y=1.02, x=0,
                                          font=dict(size=11)))
            png = _png(_layout(fig, f"Full Spectrum · {bl}", h=480), height=480)
            if png:
                out.append({"png": png, "name": _machine_of_bearing(bid),
                            "bearing": bl,
                            "analysis": (
                                f"Precesión {dom} dominante "
                                f"({max(ef, eb) / tot * 100:.0f}% de la energía "
                                f"direccional). El espectro completo separa las "
                                f"componentes hacia adelante de las inversas; el "
                                f"predominio inverso marcado es el patrón que "
                                f"acompaña al roce y a la inestabilidad de "
                                f"película (whirl/whip).")})
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("fullspec_bundle(%s): %s", instance_id, e)
        return []


# --------------------------------------------------------------------------- #
# Arranque — Shaft Centerline
# --------------------------------------------------------------------------- #
def _latest_in_window(list_fn, instance_id: str, days: int) -> Optional[str]:
    """snapshot_id del más reciente dentro de la ventana, o None."""
    try:
        snaps = list_fn(instance_id) or []
    except Exception:  # noqa: BLE001
        return None
    for s in snaps:  # vienen más recientes primero
        if _within(s.get("timestamp", ""), days):
            return s.get("snapshot_id")
    return None


def scl_startup_bundle(instance_id: str, days: int) -> List[Dict[str, Any]]:
    try:
        import plotly.graph_objects as go
        from core.scl_history import list_scl_snapshots, load_scl_snapshot
    except Exception:  # noqa: BLE001
        return []
    sid = _latest_in_window(list_scl_snapshots, instance_id, days)
    if not sid:
        return []
    snap = load_scl_snapshot(instance_id, sid)
    if not snap:
        return []
    fecha = _fecha_label(snap.get("timestamp", ""))
    label_run = snap.get("corrida_label", "") or fecha
    out: List[Dict[str, Any]] = []
    for b in snap.get("bearings", []):
        try:
            rad = float(b.get("clearance_radial") or 0.0)
            gx = float(b.get("x_gap_at_op") or 0.0)
            gy = float(b.get("y_gap_at_op") or 0.0)
            ecc = float(b.get("eccentricity_ratio") or 0.0)
            att = float(b.get("attitude_angle") or 0.0)
            unit = b.get("gap_unit", "µm") or "µm"
            bl = b.get("bearing_label", "")
            tx = np.asarray(b.get("trajectory_x_gap") or [], float)
            ty = np.asarray(b.get("trajectory_y_gap") or [], float)
            ts = np.asarray(b.get("trajectory_speed") or [], float)
            if rad <= 0:
                rad = max(float(np.nanmax(np.hypot(tx, ty))) if tx.size else gy, 1.0) * 1.3
            fig = go.Figure()
            th = np.linspace(0, 2 * np.pi, 180)
            fig.add_trace(go.Scatter(x=rad * np.cos(th), y=rad * np.sin(th),
                          mode="lines", line=dict(color="#334155", width=1.6),
                          hoverinfo="skip"))
            for frac, c in [(0.95, "#dc3545"), (0.30, "#0ea5e9")]:
                fig.add_trace(go.Scatter(x=rad * frac * np.cos(th),
                              y=rad * frac * np.sin(th), mode="lines",
                              line=dict(color=c, width=1, dash="dot"),
                              hoverinfo="skip"))
            if tx.size and ty.size:
                if ts.size == tx.size and np.isfinite(ts).any():
                    fig.add_trace(go.Scatter(
                        x=tx, y=ty, mode="markers",
                        marker=dict(color=ts, colorscale="Turbo", size=4,
                                    showscale=True,
                                    colorbar=dict(title="rpm", thickness=12)),
                        hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines",
                              line=dict(color="#94a3b8", width=0.9),
                              hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text",
                          marker=dict(color="#64748b", size=8, symbol="cross-thin",
                                      line=dict(color="#334155", width=2)),
                          text=["centro"], textposition="top right",
                          hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[0], y=[-rad], mode="markers+text",
                          marker=dict(color="#64748b", size=9, symbol="circle-open",
                                      line=dict(color="#334155", width=2)),
                          text=["reposo"], textposition="bottom center",
                          hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[gx], y=[gy], mode="markers+text",
                          marker=dict(color="#dc3545" if ecc >= 0.9 else "#1f9d55",
                                      size=15, line=dict(color="#fff", width=1.5)),
                          text=["operación"], textposition="top center",
                          hoverinfo="skip"))
            _axr = rad * 1.4
            fig.update_xaxes(title=f"Horizontal H ({unit})", scaleanchor="y",
                             scaleratio=1, zeroline=True, range=[-_axr, _axr])
            fig.update_yaxes(title=f"Vertical V ({unit}) · arriba +", zeroline=True,
                             range=[-_axr, _axr])
            png = _png(_layout(fig, f"Shaft Centerline · {bl}", h=520), height=520)
            if png:
                out.append({
                    "kind": "Shaft Centerline", "label": bl, "date": fecha,
                    "run": label_run, "png": png,
                    "analysis": (
                        f"Posición del muñón a operación: excentricidad "
                        f"ε={ecc:.2f}, ángulo de actitud {att:.0f}°. "
                        + ("Excentricidad alta (ε≥0.9): el muñón trabaja cerca "
                           "del claro — verificar carga, alineación y claro del "
                           "cojinete." if ecc >= 0.9 else
                           "Posición dentro del claro esperado para la carga "
                           "y velocidad; el ascenso del muñón con la velocidad "
                           "es el comportamiento normal de la película."))})
        except Exception as e:  # noqa: BLE001
            log.warning("scl_startup_bundle bearing: %s", e)
            continue
    return out


# --------------------------------------------------------------------------- #
# Arranque — Bode
# --------------------------------------------------------------------------- #
def bode_startup_bundle(instance_id: str, days: int) -> List[Dict[str, Any]]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from core.bode_history import list_bode_snapshots, load_bode_snapshot
    except Exception:  # noqa: BLE001
        return []
    sid = _latest_in_window(list_bode_snapshots, instance_id, days)
    if not sid:
        return []
    snap = load_bode_snapshot(instance_id, sid)
    if not snap:
        return []
    fecha = _fecha_label(snap.get("timestamp", ""))
    run = snap.get("corrida_label", "") or fecha
    op = float(snap.get("operating_speed_rpm") or 0.0)
    out: List[Dict[str, Any]] = []
    for s in snap.get("sensors", []):
        try:
            r = np.asarray(s.get("trajectory_speed") or [], float)
            a = np.asarray(s.get("trajectory_amp") or [], float)
            p = np.asarray(s.get("trajectory_phase") or [], float)
            if r.size < 3 or a.size != r.size or p.size != r.size:
                continue
            unit = s.get("amp_unit", "") or ""
            lbl = s.get("sensor_label", "")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.07, row_heights=[0.56, 0.44])
            fig.add_trace(go.Scatter(x=r, y=a, mode="lines+markers",
                          line=dict(color=_NAVY, width=1.9), marker=dict(size=4)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=r, y=p, mode="lines+markers",
                          line=dict(color=_AMBER, width=1.9), marker=dict(size=4)),
                          row=2, col=1)
            cs = s.get("critical_speed_rpm")
            if cs:
                q = s.get("q_factor") or 0.0
                for rr in (1, 2):
                    fig.add_vline(x=float(cs), line=dict(color="#dc3545", width=1.2,
                                  dash="dash"), row=rr, col=1)
                fig.add_annotation(x=float(cs), yref="paper", y=1.0,
                                   text=f"fc {float(cs):.0f} · Q={float(q):.1f}",
                                   showarrow=False, font=dict(size=11, color="#dc3545"))
            if op > 0:
                for rr in (1, 2):
                    fig.add_vline(x=op, line=dict(color=_NAVY, width=1, dash="dot"),
                                  row=rr, col=1)
            fig.update_yaxes(title=f"Amplitud ({unit})", row=1, col=1)
            fig.update_yaxes(title="Fase (°)", row=2, col=1)
            fig.update_xaxes(title="Velocidad (rpm)", row=2, col=1)
            png = _png(_layout(fig, f"Bode · {lbl}", h=560), height=560)
            if png:
                _amp_op = float(s.get("amp_at_op") or 0.0)
                out.append({
                    "kind": "Bode", "label": lbl, "date": fecha, "run": run,
                    "png": png,
                    "analysis": (
                        f"Respuesta síncrona 1X del run-up. Amplitud a operación "
                        f"({op:.0f} rpm) ≈ {_amp_op:.2f} {unit}."
                        + (f" Crítica detectada en {float(cs):.0f} rpm "
                           f"(Q={float(s.get('q_factor') or 0):.1f}); el margen "
                           "a la velocidad de operación se evalúa con API 684."
                           if cs else
                           " No se resolvió una crítica dentro del rango "
                           "barrido."))})
        except Exception as e:  # noqa: BLE001
            log.warning("bode_startup_bundle sensor: %s", e)
            continue
    return out


# --------------------------------------------------------------------------- #
# Arranque — Polar
# --------------------------------------------------------------------------- #
def polar_startup_bundle(instance_id: str, days: int) -> List[Dict[str, Any]]:
    try:
        import plotly.graph_objects as go
        from core.polar_history import list_polar_snapshots, load_polar_snapshot
    except Exception:  # noqa: BLE001
        return []
    sid = _latest_in_window(list_polar_snapshots, instance_id, days)
    if not sid:
        return []
    snap = load_polar_snapshot(instance_id, sid)
    if not snap:
        return []
    fecha = _fecha_label(snap.get("timestamp", ""))
    run = snap.get("corrida_label", "") or fecha
    out: List[Dict[str, Any]] = []
    for s in snap.get("sensors", []):
        try:
            r = np.asarray(s.get("trajectory_speed") or [], float)
            a = np.asarray(s.get("trajectory_amp") or [], float)
            p = np.asarray(s.get("trajectory_phase") or [], float)
            if r.size < 3 or a.size != r.size or p.size != r.size:
                continue
            unit = s.get("amp_unit", "") or ""
            lbl = s.get("sensor_label", "")
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=a, theta=p, mode="lines+markers",
                line=dict(color=_NAVY, width=1.8),
                marker=dict(size=5, color=r, colorscale="Blues", showscale=True,
                            colorbar=dict(title="rpm", thickness=12))))
            cs = s.get("critical_speed_rpm")
            if cs:
                fig.add_trace(go.Scatterpolar(
                    r=[float(s.get("critical_speed_amp") or 0.0)],
                    theta=[float(s.get("critical_speed_phase") or 0.0)],
                    mode="markers",
                    marker=dict(color="#dc3545", size=11, symbol="x")))
            fig.update_layout(template="plotly_white", height=540,
                              title=dict(text=f"Polar · {lbl}",
                                         font=dict(size=16, color="#0f172a")),
                              font=dict(family="IBM Plex Sans", size=13),
                              paper_bgcolor="white",
                              polar=dict(angularaxis=dict(direction="clockwise",
                                                          rotation=90)),
                              margin=dict(l=48, r=48, t=56, b=44),
                              showlegend=False)
            png = _png(fig, height=540)
            if png:
                out.append({
                    "kind": "Polar", "label": lbl, "date": fecha, "run": run,
                    "png": png,
                    "analysis": (
                        f"Vector 1X (amplitud y fase) a lo largo del run-up. El "
                        f"lazo describe el paso por resonancia; un rizo cerrado "
                        f"con giro de fase ≈180° confirma crítica de rotor."
                        + (f" Crítica en {float(cs):.0f} rpm." if cs else ""))})
        except Exception as e:  # noqa: BLE001
            log.warning("polar_startup_bundle sensor: %s", e)
            continue
    return out


# --------------------------------------------------------------------------- #
# API: todo junto
# --------------------------------------------------------------------------- #
def collect_startup_figures(instance_id: str,
                            period_label: str = "Semanal") -> Dict[str, Any]:
    """Devuelve {'startup': [...], 'fullspec_pngs': [...]} para el reporte.
    'startup' = SCL+Bode+Polar del último run-up en la ventana (puede ir vacío).
    'fullspec_pngs' = Full Spectrum steady por cojinete (del robot full-speed)."""
    days = _days_for(period_label)
    out: Dict[str, Any] = {"startup": [], "fullspec_pngs": []}
    try:
        out["fullspec_pngs"] = fullspec_bundle(instance_id)
    except Exception as e:  # noqa: BLE001
        log.warning("collect_startup_figures fullspec: %s", e)
    try:
        startup: List[Dict[str, Any]] = []
        startup += scl_startup_bundle(instance_id, days)
        startup += bode_startup_bundle(instance_id, days)
        startup += polar_startup_bundle(instance_id, days)
        out["startup"] = startup
    except Exception as e:  # noqa: BLE001
        log.warning("collect_startup_figures startup: %s", e)
    return out


__all__ = ["collect_startup_figures", "fullspec_bundle",
           "scl_startup_bundle", "bode_startup_bundle", "polar_startup_bundle"]
