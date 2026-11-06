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
# Helpers rotodinámicos (compartidos Bode/Polar) — convención Bently
# --------------------------------------------------------------------------- #
def _slowroll_vector(r, a, p, frac: float = 0.05):
    """Vector de giro lento (runout) = promedio del vector 1X a baja velocidad
    (rpm ≤ mínimo + frac·rango). Devuelve (sx, sy, mag, ang_deg, n)."""
    r = np.asarray(r, float); a = np.asarray(a, float)
    ph = np.radians(np.asarray(p, float))
    x = a * np.cos(ph); y = a * np.sin(ph)
    if r.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    thr = float(np.nanmin(r)) + frac * (float(np.nanmax(r)) - float(np.nanmin(r)))
    m = np.isfinite(r) & (r <= thr)
    if int(m.sum()) >= 1:
        sx, sy, n = float(np.nanmean(x[m])), float(np.nanmean(y[m])), int(m.sum())
    else:
        i = int(np.nanargmin(r)); sx, sy, n = float(x[i]), float(y[i]), 1
    return sx, sy, float(np.hypot(sx, sy)), float(np.degrees(np.arctan2(sy, sx)) % 360.0), n


def _compensate(r, a, p, frac: float = 0.05):
    """Resta el vector de giro lento de cada punto (Bently 5-5). Devuelve
    (amp_comp, phase_comp, slowroll_dict)."""
    r = np.asarray(r, float); a = np.asarray(a, float)
    ph = np.radians(np.asarray(p, float))
    x = a * np.cos(ph); y = a * np.sin(ph)
    sx, sy, mag, ang, n = _slowroll_vector(r, a, p, frac)
    xc, yc = x - sx, y - sy
    return (np.hypot(xc, yc),
            np.degrees(np.arctan2(yc, xc)) % 360.0,
            {"mag": mag, "ang": ang, "n": n})


def _unwrap_robust(r, p):
    """Fase desenrollada ROBUSTA (media circular en ventana corta antes de
    unwrap) → curva continua sin runaway de 360° en bandas ruidosas."""
    r = np.asarray(r, float); p = np.asarray(p, float)
    n = int(r.size)
    if n < 2:
        return p
    o = np.argsort(r)
    z = np.exp(1j * np.radians(p[o]))
    k = max(3, (n // 100) | 1)
    if n > k:
        z = np.convolve(z, np.ones(k) / k, mode="same")
    pw = np.degrees(np.unwrap(np.angle(z)))
    pw = pw - float(np.round(pw[0] / 360.0)) * 360.0
    out = np.empty_like(pw)
    out[o] = pw
    return out


def _amp_coeff_saf(r, a, rc):
    """Factor de amplificación síncrona por COEFICIENTE DE AMPLITUD (Bently 5-12):
    amp del pico de resonancia / amplitud de la asíntota (respuesta forzada lejos
    de la crítica, bien por encima). Aproximado. Devuelve float o 0."""
    r = np.asarray(r, float); a = np.asarray(a, float)
    if r.size < 5 or not rc:
        return 0.0
    ipk = int(np.argmin(np.abs(r - rc)))
    apk = float(a[ipk])
    # asíntota: mediana de la amplitud bien por encima de la crítica (≥1.5×rc)
    hi = a[r >= 1.5 * rc]
    base = float(np.median(hi)) if hi.size >= 3 else float(np.median(a[r > rc])) if (r > rc).any() else 0.0
    if base <= 1e-9 or apk <= 0:
        return 0.0
    return apk / base


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
        from core.briefing_figures import _load_merged_snapshot, _instance_rpm
    except Exception:  # noqa: BLE001
        return []
    try:
        payload = _load_merged_snapshot(instance_id, "orbit")
        if not payload:
            return []
        out: List[Dict[str, Any]] = []
        rpm0 = float(payload.get("operating_speed_rpm") or 0.0) \
            or _instance_rpm(instance_id)
        for b in payload.get("bearings", []):
            xv = np.asarray(b.get("x_values") or [], float)
            yv = np.asarray(b.get("y_values") or [], float)
            n = int(min(xv.size, yv.size))
            if n < 16:
                continue
            xv, yv = xv[:n], yv[:n]
            xv = xv - np.nanmean(xv)
            yv = yv - np.nanmean(yv)
            unit = b.get("amp_unit") or "µm pp"
            kind, cpmx = _spk(unit)
            win = np.hanning(n)
            z = (xv * win) + 1j * (yv * win)
            sp = np.fft.fftshift(np.fft.fft(z)) * (2.0 / np.sum(win))
            amp = np.abs(sp) * (2.0 if kind == "disp" else 1.0 / np.sqrt(2.0))
            # Eje de frecuencia: si el snapshot trae fs → cpm absoluto; si NO
            # (export de órbita no guarda fs) → auto-calibración al 1X usando la
            # velocidad de giro: el pico forward dominante = 1X → eje en ÓRDENES.
            fs = float(b.get("sampling_rate_hz") or 0.0)
            freq_cyc = np.fft.fftshift(np.fft.fftfreq(n))   # ciclos/muestra
            if fs > 0:
                xf = freq_cyc * fs * 60.0                    # cpm
                xtitle = "Frecuencia (cpm) · − inversa | + directa"
                xrange = [-cpmx, cpmx]
                xnote = ""
            else:
                if rpm0 <= 0:
                    continue
                _fwdm = freq_cyc > 0
                if not np.any(_fwdm):
                    continue
                _idx = np.where(_fwdm)[0]
                f1 = float(freq_cyc[_idx[int(np.argmax(amp[_idx]))]])
                if f1 <= 0:
                    continue
                xf = freq_cyc / f1                           # órdenes (×1X)
                xtitle = "Orden (× velocidad de giro, 1X) · − inversa | + directa"
                xrange = [-10.0, 10.0]
                xnote = (f" Eje de frecuencia auto-calibrado al 1X "
                         f"(velocidad de giro ≈ {rpm0:,.0f} rpm); escala en "
                         f"órdenes porque el export de órbita no guarda la tasa "
                         f"de muestreo.")
            fwd = xf >= 0
            bl = b.get("bearing_label") or b.get("x_sensor_label") or ""
            bid = _bid(bl)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xf[fwd], y=amp[fwd], mode="lines",
                line=dict(color=_NAVY, width=1.3),
                name="Forward (precesión directa +)"))
            fig.add_trace(go.Scatter(
                x=xf[~fwd], y=amp[~fwd], mode="lines",
                line=dict(color=_AMBER, width=1.3),
                name="Reverse (precesión inversa −)"))
            # energía fwd vs bwd (dominancia de precesión)
            ef = float(np.sum(amp[fwd] ** 2))
            eb = float(np.sum(amp[~fwd] ** 2))
            tot = ef + eb or 1.0
            dom = "directa (forward)" if ef >= eb else "inversa (reverse)"
            fig.update_xaxes(title=xtitle,
                             range=xrange, showgrid=True, gridcolor=_GRID,
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
                                f"película (whirl/whip)." + xnote)})
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


def _ids_in_window(list_fn, instance_id: str, days: int) -> List[str]:
    """snapshot_ids dentro de la ventana, MÁS RECIENTES primero. Permite juntar
    el último snapshot POR cojinete/sensor cuando se guardan por separado (un
    cojinete por snapshot)."""
    try:
        snaps = list_fn(instance_id) or []
    except Exception:  # noqa: BLE001
        return []
    return [s.get("snapshot_id") for s in snaps
            if _within(s.get("timestamp", ""), days) and s.get("snapshot_id")]


def scl_startup_bundle(instance_id: str, days: int) -> List[Dict[str, Any]]:
    try:
        import plotly.graph_objects as go
        from core.scl_history import list_scl_snapshots, load_scl_snapshot
    except Exception:  # noqa: BLE001
        return []
    ids = _ids_in_window(list_scl_snapshots, instance_id, days)
    if not ids:
        return []
    out: List[Dict[str, Any]] = []
    _seen: set = set()                    # un cojinete = el más reciente
    for sid in ids:
        snap = load_scl_snapshot(instance_id, sid)
        if not snap:
            continue
        fecha = _fecha_label(snap.get("timestamp", ""))
        label_run = snap.get("corrida_label", "") or fecha
        for b in snap.get("bearings", []):
            _bl0 = b.get("bearing_label", "")
            if _bl0 in _seen:
                continue
            _seen.add(_bl0)
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
    # ordena por número de cojinete
    out.sort(key=lambda r: (int(_bid(r.get("label", "")) or 99), r.get("label", "")))
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
    ids = _ids_in_window(list_bode_snapshots, instance_id, days)
    if not ids:
        return []
    merged: List[tuple] = []
    _seen: set = set()                    # un sensor = el más reciente
    for sid in ids:
        snap = load_bode_snapshot(instance_id, sid)
        if not snap:
            continue
        fecha = _fecha_label(snap.get("timestamp", ""))
        run = snap.get("corrida_label", "") or fecha
        op = float(snap.get("operating_speed_rpm") or 0.0)
        for s in snap.get("sensors", []):
            sl = s.get("sensor_label", "")
            # Dedup por (cojinete, eje) normalizado, NO por etiqueta cruda: el
            # mismo sensor puede venir con labels distintos ("2XD" vs
            # "2XD TURBINA NDE") y duplicaría figuras. ids vienen más-reciente
            # primero → el primero visto (más reciente) gana.
            _m = re.search(r"(\d+)\s*([XYxy])", sl)
            _k = (_m.group(1), _m.group(2).upper()) if _m else sl
            if _k in _seen:
                continue
            _seen.add(_k)
            merged.append((s, fecha, run, op))
    out: List[Dict[str, Any]] = []
    for s, fecha, run, op in merged:
        try:
            r = np.asarray(s.get("trajectory_speed") or [], float)
            a = np.asarray(s.get("trajectory_amp") or [], float)
            p = np.asarray(s.get("trajectory_phase") or [], float)
            if r.size < 3 or a.size != r.size or p.size != r.size:
                continue
            unit = s.get("amp_unit", "") or ""
            lbl = s.get("sensor_label", "")
            # orden ascendente por rpm
            _o = np.argsort(r)
            r, a, p = r[_o], a[_o], p[_o]
            # COMPENSAR giro lento (Bently 5-5): solo para desplazamiento de eje
            # (prox, µm). La velocidad/aceleración de carcasa no lleva runout.
            _is_disp = "mm/s" not in unit.lower() and "g" not in unit.lower().split()
            _srv = None
            if _is_disp:
                a, p, _srv = _compensate(r, a, p)
            # fase DESENROLLADA ROBUSTA → curva continua sin runaway de 360°.
            pw = _unwrap_robust(r, p)
            # AF por coeficiente de amplitud (segundo método, Bently 5-12)
            _saf_amp = _amp_coeff_saf(r, a, s.get("critical_speed_rpm"))
            # System1: FASE ARRIBA (lag, eje invertido) + AMPLITUD ABAJO.
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.07, row_heights=[0.44, 0.56])
            fig.add_trace(go.Scatter(x=r, y=pw, mode="lines",
                          line=dict(color=_NAVY, width=1.9, shape="spline",
                                    smoothing=0.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=r, y=a, mode="lines",
                          line=dict(color=_AMBER, width=1.9, shape="spline",
                                    smoothing=0.5)), row=2, col=1)
            cs = s.get("critical_speed_rpm")
            if cs:
                q = s.get("q_factor") or 0.0
                for rr in (1, 2):
                    fig.add_vline(x=float(cs), line=dict(color="#dc3545", width=1.2,
                                  dash="dash"), row=rr, col=1)
                _aflbl = f"fc {float(cs):.0f} · AF {float(q):.1f}"
                if _saf_amp > 0:
                    _aflbl += f"/{_saf_amp:.1f}"
                fig.add_annotation(x=float(cs), yref="paper", y=1.0,
                                   text=_aflbl,
                                   showarrow=False, font=dict(size=11, color="#dc3545"))
            if op > 0:
                for rr in (1, 2):
                    fig.add_vline(x=op, line=dict(color="#0f172a", width=1, dash="dot"),
                                  row=rr, col=1)
                fig.add_annotation(x=op, yref="paper", y=1.0, text=f"op {op:.0f}",
                                   showarrow=False, font=dict(size=10, color="#0f172a"),
                                   xshift=18)
            # Fase arriba, eje INVERTIDO (el lag crece hacia abajo), ticks cada 90°.
            fig.update_yaxes(title="Fase 1X (°) · lag", autorange="reversed",
                             dtick=90, row=1, col=1)
            fig.update_yaxes(title=f"Amplitud ({unit})", rangemode="tozero",
                             row=2, col=1)
            fig.update_xaxes(title="Velocidad (rpm)", row=2, col=1)
            png = _png(_layout(fig, f"Bode · {lbl}", h=560), height=560)
            if png:
                _amp_op = (float(np.interp(op, r, a)) if (op > 0 and r.size)
                           else float(a[-1]) if a.size else 0.0)
                _comp_txt = (f" Compensado de giro lento (vector restado "
                             f"{_srv['mag']:.2f} {unit} ∠{_srv['ang']:.0f}°, "
                             f"Bently 5-5)." if _srv else
                             " Sin compensación de giro lento (medición de "
                             "carcasa).")
                _q = float(s.get("q_factor") or 0)
                _af_txt = ""
                if cs:
                    _af_txt = (f" Crítica en {float(cs):.0f} rpm; factor de "
                               f"amplificación AF≈{_q:.1f} (media potencia)")
                    if _saf_amp > 0:
                        _af_txt += f" / {_saf_amp:.1f} (coef. de amplitud)"
                    _mg = (op - float(cs)) / op * 100.0 if op else 0.0
                    _af_txt += (f"; margen de separación a operación {_mg:+.0f}% "
                                "(se evalúa con API 684).")
                else:
                    _af_txt = (" No se resolvió una crítica dentro del rango "
                               "barrido.")
                out.append({
                    "kind": "Bode", "label": lbl, "date": fecha, "run": run,
                    "png": png,
                    "analysis": (
                        f"Respuesta síncrona 1X del run-up. Amplitud a operación "
                        f"({op:.0f} rpm) ≈ {_amp_op:.2f} {unit}."
                        + _comp_txt + _af_txt)})
        except Exception as e:  # noqa: BLE001
            log.warning("bode_startup_bundle sensor: %s", e)
            continue
    out.sort(key=lambda r: (int(_bid(r.get("label", "")) or 99), r.get("label", "")))
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
    ids = _ids_in_window(list_polar_snapshots, instance_id, days)
    if not ids:
        return []
    merged: List[tuple] = []
    _seen: set = set()                    # un sensor = el más reciente
    for sid in ids:
        snap = load_polar_snapshot(instance_id, sid)
        if not snap:
            continue
        fecha = _fecha_label(snap.get("timestamp", ""))
        run = snap.get("corrida_label", "") or fecha
        for s in snap.get("sensors", []):
            sl = s.get("sensor_label", "")
            if sl in _seen:
                continue
            _seen.add(sl)
            merged.append((s, fecha, run))
    out: List[Dict[str, Any]] = []
    for s, fecha, run in merged:
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
    out.sort(key=lambda r: (int(_bid(r.get("label", "")) or 99), r.get("label", "")))
    return out


# --------------------------------------------------------------------------- #
# Deflexión del eje (nivel tren) — Bently manual 3-8 / 9-4
# --------------------------------------------------------------------------- #
def deflection_bundle(instance_id: str) -> List[Dict[str, Any]]:
    """Forma de deflexión del eje: cada órbita en su posición axial, unidas por
    el punto Keyphasor (muestra 0) → forma modal aparente del rotor. Nivel TREN
    (una figura). Del mismo snapshot de órbita (robot full-speed). Replica la
    vista 'Deflexión del eje' de Rotodinámica (H=(X−Y)/√2, V=(X+Y)/√2)."""
    try:
        import plotly.graph_objects as go
        from core.briefing_figures import _load_merged_snapshot
    except Exception:  # noqa: BLE001
        return []
    try:
        payload = _load_merged_snapshot(instance_id, "orbit")
        if not payload:
            return []
        stations: List[Dict[str, Any]] = []
        maxr = 1e-9
        unit = "µm"
        for b in payload.get("bearings", []):
            xv = np.asarray(b.get("x_values") or [], float)
            yv = np.asarray(b.get("y_values") or [], float)
            n = int(min(xv.size, yv.size))
            if n < 8:
                continue
            xv, yv = xv[:n], yv[:n]
            bl = b.get("bearing_label") or b.get("x_sensor_label") or ""
            unit = b.get("amp_unit") or unit
            H = (xv - yv) / np.sqrt(2.0)
            V = (xv + yv) / np.sqrt(2.0)
            H = H - np.nanmean(H)
            V = V - np.nanmean(V)
            r = float(np.max(np.hypot(H, V))) if H.size else 0.0
            maxr = max(maxr, r)
            stations.append({"label": bl, "bid": _bid(bl), "H": H, "V": V,
                             "kp": (float(H[0]), float(V[0]))})
        stations.sort(key=lambda s: int(s["bid"] or 99))
        if len(stations) < 2:
            return []
        dx = 1.0
        S = (0.40 * dx / maxr) if maxr > 0 else 1.0
        xs = [i * dx for i in range(len(stations))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[xs[0] - 0.5 * dx, xs[-1] + 0.5 * dx], y=[0, 0], mode="lines",
            line=dict(color="#c0c8d4", width=1, dash="dot"),
            hoverinfo="skip", showlegend=False))
        kpts: List[Any] = []
        for i, s in enumerate(stations):
            x0 = xs[i]
            fig.add_trace(go.Scatter(
                x=x0 + S * s["H"], y=S * s["V"], mode="lines",
                line=dict(color=_NAVY, width=1.5), showlegend=False))
            kx, ky = x0 + S * s["kp"][0], S * s["kp"][1]
            kpts.append((kx, ky))
            fig.add_trace(go.Scatter(
                x=[kx], y=[ky], mode="markers",
                marker=dict(color="#1f9d55", size=10,
                            line=dict(color="white", width=1.2)),
                hoverinfo="skip", showlegend=False))
            fig.add_annotation(x=x0, y=-0.5 * dx, text=s["label"], showarrow=False,
                               font=dict(size=10, family="IBM Plex Mono",
                                         color="#5a6b85"))
        fig.add_trace(go.Scatter(
            x=[p[0] for p in kpts], y=[p[1] for p in kpts], mode="lines",
            line=dict(color=_AMBER, width=2.4), showlegend=False))
        fig.update_xaxes(showticklabels=False, zeroline=False,
                         title="posición axial del tren →", gridcolor=_GRID)
        fig.update_yaxes(title=f"{unit}", zeroline=False, gridcolor=_GRID)
        png = _png(_layout(fig, "Forma de deflexión del eje · tren", h=470),
                   height=470)
        if not png:
            return []
        return [{"png": png, "name": "Tren", "bearing": "tren",
                 "analysis": (
                     "Cada órbita en su posición axial; la curva ámbar une los "
                     "puntos Keyphasor (posición del eje en el disparo, muestra "
                     "0) → forma de deflexión aparente del eje = forma modal del "
                     "rotor (cilíndrico/traslacional, pivotal/cónico…). Base: "
                     "Bently manual 3-8 / 9-4. Cojinetes ordenados por número "
                     "(posición axial). Órbita completa del robot a full-speed "
                     "(dominada por 1X).")}]
    except Exception as e:  # noqa: BLE001
        log.warning("deflection_bundle(%s): %s", instance_id, e)
        return []


# --------------------------------------------------------------------------- #
# API: todo junto
# --------------------------------------------------------------------------- #
def collect_startup_figures(instance_id: str,
                            period_label: str = "Semanal") -> Dict[str, Any]:
    """Devuelve {'startup': [...], 'fullspec_pngs': [...], 'deflection_pngs': [...]}
    para el reporte. 'startup' = SCL+Bode+Polar del último run-up en la ventana
    (puede ir vacío). 'fullspec_pngs' = Full Spectrum steady por cojinete.
    'deflection_pngs' = forma de deflexión del eje (nivel tren)."""
    days = _days_for(period_label)
    out: Dict[str, Any] = {"startup": [], "fullspec_pngs": [],
                           "deflection_pngs": []}
    try:
        out["fullspec_pngs"] = fullspec_bundle(instance_id)
    except Exception as e:  # noqa: BLE001
        log.warning("collect_startup_figures fullspec: %s", e)
    try:
        out["deflection_pngs"] = deflection_bundle(instance_id)
    except Exception as e:  # noqa: BLE001
        log.warning("collect_startup_figures deflection: %s", e)
    try:
        startup: List[Dict[str, Any]] = []
        startup += scl_startup_bundle(instance_id, days)
        startup += bode_startup_bundle(instance_id, days)
        startup += polar_startup_bundle(instance_id, days)
        out["startup"] = startup
    except Exception as e:  # noqa: BLE001
        log.warning("collect_startup_figures startup: %s", e)
    return out


__all__ = ["collect_startup_figures", "fullspec_bundle", "deflection_bundle",
           "scl_startup_bundle", "bode_startup_bundle", "polar_startup_bundle"]
