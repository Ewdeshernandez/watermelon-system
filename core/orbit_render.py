"""
core.orbit_render
=================

FUENTE ÚNICA del render de órbitas (probe glyphs técnicos + trazo navy +
keyphasor + texto CW/CCW API 670 + elipticidad/Smax), compartida por:

  · la web de Rotodinámica (pages/02_Analysis)
  · los REPORTES diario/semanal/mensual (core/briefing_figures.orbit_bundle)

Antes el reporte dibujaba una órbita "pelada" (solo la línea) desde los
snapshots dynamic_raw, mientras la web tenía el render bonito. Ahora ambos
pasan por aquí → se ven IDÉNTICOS (pedido del usuario: el fuerte de gráficos
vive en Rotodinámica y de ahí salen todos los reportes).

Entrada agnóstica de la fuente: dict normalizado por cojinete
    {label, x, y, x_label, y_label, unit, rpm?, fs_hz?, spr?, machine?, stamp?}

`fs_hz` y `spr` son OPCIONALES: si faltan (caso snapshot dynamic_raw, que no
los guarda) se dibuja la órbita Direct tal cual, con keyphasor en la muestra 0
y precesión por área firmada. Si están (caso web), se usa filtrado por orden +
keyphasor por vuelta + precesión por FFT (Forward/Reverse), idéntico a System1.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

_NAVY = "#12305e"
_GEN_BEARINGS = {5, 6}
# Cojinetes con sondas de proximidad montadas ARRIBA a 45°R/45°L (conv. Bently
# 0°=arriba): generador (5/6) Y gearbox (3/4) — mismo montaje, confirmado por
# el cliente en SGT300A/B. Turbina (1/2) = sondas ABAJO 135°R/135°L.
_PROBES_UP_45 = {3, 4, 5, 6}


# ---------------------------------------------------------------------------
# Geometría de sondas
# ---------------------------------------------------------------------------
def bearing_number(*vals) -> int:
    """Primer número de cojinete hallado (ej '5XD' → 5)."""
    import re
    for v in vals:
        m = re.search(r"\d+", str(v or ""))
        if m:
            return int(m.group(0))
    return 0


def probe_angles(bnum: int):
    """(ang_x, ang_y, zona). Generador (5/6) y gearbox (3/4) = sondas ARRIBA
    45°R/45°L; turbina (1/2) = sondas ABAJO 135°R/135°L (conv. Bently 0°=arriba)."""
    if int(bnum) in _PROBES_UP_45:
        return 45.0, 45.0, ("generador" if int(bnum) in _GEN_BEARINGS else "gearbox")
    return 135.0, 135.0, "turbina"


def machine_of(bnum: int) -> str:
    n = int(bnum)
    return "T" if n in (1, 2) else ("G" if n in (3, 4)
           else ("GEN" if n in (5, 6) else "X"))


MACH_NAME = {"T": "Turbina", "G": "Gearbox", "GEN": "Generador", "X": "Equipo"}
MACH_ORDER = ["T", "G", "GEN", "X"]


def add_probe_glyphs(fig, R, ax_deg, ay_deg, lbl_x, lbl_y, *, with_cone=True,
                     with_top=False, row=None, col=None):
    """Dibuja las sondas de proximidad tipo dibujo técnico (cuerpo + rosca +
    cara naranja) en su ángulo real, FUERA de R, + cono de visión. Marco:
    0°=arriba, +H=derecha, +V=arriba. Port exacto del render web."""
    import plotly.graph_objects as go
    if R <= 0:
        return
    _rc = {} if row is None else dict(row=row, col=col)
    for _lbl, _deg, _s in [(lbl_x, float(ax_deg), 1.0), (lbl_y, float(ay_deg), -1.0)]:
        _th = np.radians(_deg)
        _ux, _uy = _s * np.sin(_th), np.cos(_th)
        if with_cone:
            _ang = np.arctan2(_uy, _ux)
            _aa = np.linspace(_ang - np.radians(13), _ang + np.radians(13), 18)
            fig.add_trace(go.Scatter(
                x=np.concatenate([[0], R * 0.95 * np.cos(_aa), [0]]),
                y=np.concatenate([[0], R * 0.95 * np.sin(_aa), [0]]),
                fill="toself", mode="lines", line=dict(color="rgba(18,48,94,0)"),
                fillcolor="rgba(18,48,94,0.05)", hoverinfo="skip", showlegend=False), **_rc)
        _ri, _rb, _ro = R * 1.12, R * 1.20, R * 1.36
        _perp = np.array([-_uy, _ux])
        fig.add_trace(go.Scatter(x=[_ux * _rb, _ux * _ro], y=[_uy * _rb, _uy * _ro],
                      mode="lines", line=dict(color="#12305e", width=10),
                      hoverinfo="skip", showlegend=False), **_rc)
        for _t in (0.55, 0.70, 0.85):   # rosca
            _cx, _cy = _ux * (_rb + (_ro - _rb) * _t), _uy * (_rb + (_ro - _rb) * _t)
            fig.add_trace(go.Scatter(
                x=[_cx - _perp[0] * R * 0.045, _cx + _perp[0] * R * 0.045],
                y=[_cy - _perp[1] * R * 0.045, _cy + _perp[1] * R * 0.045],
                mode="lines", line=dict(color="#8fa0bf", width=1.5),
                hoverinfo="skip", showlegend=False), **_rc)
        fig.add_trace(go.Scatter(x=[_ux * _ri], y=[_uy * _ri], mode="markers",
                      marker=dict(color="#e8890c", size=12, symbol="circle",
                                  line=dict(color="#12305e", width=2)),
                      hoverinfo="skip", showlegend=False), **_rc)
        _sd = "R" if _s > 0 else "L"
        fig.add_annotation(x=_ux * _ro, y=_uy * _ro, text=f"{_lbl}<br>{_deg:.0f}°{_sd}",
                           showarrow=False, yshift=-16,
                           font=dict(size=10.5, family="IBM Plex Mono", color="#12305e"), **_rc)
    if with_top:
        fig.add_trace(go.Scatter(x=[0], y=[R * 1.12], mode="markers",
                      marker=dict(color="#0f172a", size=7, symbol="triangle-down"),
                      hoverinfo="skip", showlegend=False), **_rc)


# ---------------------------------------------------------------------------
# Cálculo de la órbita (port de pages/02_Analysis._orbit_data, fs/spr opcional)
# ---------------------------------------------------------------------------
def compute_orbit(x, y, *, fs_hz: float = 0.0, spr: int = 0, rpm: float = 0.0,
                  bearing_label: str = "", x_label: str = "X", y_label: str = "Y",
                  unit: str = "µm", machine: str = "", stamp: str = "",
                  fmode: str = "Direct", ang_x: Optional[float] = None,
                  ang_y: Optional[float] = None) -> Dict[str, Any]:
    """Órbita de un cojinete en el marco físico (método System1): rota a coords
    de máquina, (opcional) filtra por orden, keyphasor, precesión, forma."""
    px, py = np.asarray(x, float), np.asarray(y, float)
    _n = min(px.size, py.size)
    px, py = px[:_n], py[:_n]
    bnum = bearing_number(bearing_label, x_label, y_label)
    if ang_x is None or ang_y is None:
        _ax0, _ay0, _ = probe_angles(bnum)
        ang_x = _ax0 if ang_x is None else ang_x
        ang_y = _ay0 if ang_y is None else ang_y

    _ax = np.radians(float(ang_x)); _ay = np.radians(-float(ang_y))
    _ca, _cb = np.sin(_ax), np.cos(_ax); _cc, _cd = np.sin(_ay), np.cos(_ay)
    _det = _ca * _cd - _cb * _cc
    if abs(_det) > 1e-6:
        H = (_cd * px - _cb * py) / _det
        V = (-_cc * px + _ca * py) / _det
    else:
        H, V = px, py
    if H.size:
        H = H - np.nanmean(H); V = V - np.nanmean(V)

    def _ofilt(sig, order):
        if not order or not fs_hz or not rpm or sig.size < 8:
            return sig
        spc = np.fft.rfft(sig); frq = np.fft.rfftfreq(sig.size, 1.0 / fs_hz)
        k = int(np.argmin(np.abs(frq - order * rpm / 60.0)))
        out = np.zeros_like(spc); out[k] = spc[k]
        return np.fft.irfft(out, sig.size)

    _order = {"Direct": 0, "1X": 1, "2X": 2}.get(fmode, 0)
    Hf, Vf = _ofilt(H, _order), _ofilt(V, _order)
    sp = None; _kmarks: List = []
    if spr and spr > 4 and Hf.size >= 2 * spr:
        _nr = Hf.size // spr
        if _order == 0:
            xv, yv = Hf, Vf
            _kmarks = [(float(Hf[i * spr]), float(Vf[i * spr])) for i in range(_nr)]
            sp = _kmarks[0] if _kmarks else None
        else:
            Ha = Hf[:_nr * spr].reshape(_nr, spr).mean(0)
            Va = Vf[:_nr * spr].reshape(_nr, spr).mean(0)
            _g = max(1, spr // 40)
            xv = Ha[:max(1, spr - _g)]; yv = Va[:max(1, spr - _g)]
            sp = (float(Ha[0]), float(Va[0])); _kmarks = [sp]
    else:
        xv, yv = Hf, Vf
        sp = (float(Hf[0]), float(Vf[0])) if Hf.size else None
        _kmarks = [sp] if sp else []

    _area = float(np.sum(xv[:-1] * yv[1:] - xv[1:] * yv[:-1])) if xv.size > 2 else 0.0
    prec = "CCW" if _area > 0 else "CW"
    if fs_hz and rpm and H.size >= 8:
        _z = _ofilt(H, 1) + 1j * _ofilt(V, 1)
        _Z = np.fft.fftshift(np.fft.fft(_z))
        _frz = np.fft.fftshift(np.fft.fftfreq(_z.size, 1.0 / fs_hz)); _f1 = rpm / 60.0
        _kf = int(np.argmin(np.abs(_frz - _f1))); _kr = int(np.argmin(np.abs(_frz + _f1)))
        prec = "Forward" if abs(_Z[_kf]) >= abs(_Z[_kr]) else "Reverse"
    _precd = ("Hacia adelante (forward)" if prec == "Forward"
              else "Inversa (reverse)" if prec == "Reverse" else prec)

    rot_api = "CCW" if bnum >= 5 else "CW"
    _maq = "generador" if bnum >= 5 else ("gearbox" if bnum in (3, 4) else "turbina")
    cx, cy = (float(np.nanmean(xv)), float(np.nanmean(yv))) if xv.size else (0.0, 0.0)
    _pc = np.column_stack([xv - cx, yv - cy]) if xv.size else np.empty((0, 2))
    major = minor = ellip = smax = 0.0
    if _pc.shape[0] > 2:
        try:
            _, _, _vt = np.linalg.svd(_pc, full_matrices=False)
            _pj = _pc @ _vt[0]; _pn = _pc @ _vt[1]
            major = float(_pj.max() - _pj.min()); minor = float(_pn.max() - _pn.min())
            ellip = (minor / major) if major else 0.0
            smax = float(np.max(np.hypot(_pc[:, 0], _pc[:, 1])))
        except Exception:
            pass
    if ellip >= 0.8:
        shp, shc = "Circular — desbalance / condición sana", "#1f9d55"
    elif ellip >= 0.4:
        shp, shc = "Elíptica — precarga o anisotropía de rigidez", "#e8890c"
    else:
        shp, shc = "Muy plana — precarga fuerte / desalineación", "#dc3545"
    _kpos = ""
    if sp is not None:
        _kang = float(np.degrees(np.arctan2(sp[0], sp[1])))
        _kpos = f"{abs(_kang):.0f}°{'R' if _kang >= 0 else 'L'}"
    Rp = max(float(smax) if smax else 0.0,
             float(np.nanmax(np.hypot(xv, yv))) if len(xv) else 0.0)
    return dict(xv=xv, yv=yv, kmarks=_kmarks, sp=sp, order=_order, smax=smax, Rp=Rp,
                prec=prec, precd=_precd, rot_api=rot_api, maq=_maq, shp=shp, shc=shc,
                major=major, minor=minor, ellip=ellip, kpos=_kpos, rpm=float(rpm or 0),
                bearing_label=bearing_label, x_label=x_label, y_label=y_label,
                unit=unit, machine=machine, stamp=stamp, ang_x=ang_x, ang_y=ang_y)


def _add_orbit_cell(fig, d, R, row, col, *, with_cone=False):
    """Pinta UNA órbita (probetas + trazo + keyphasor + CW/CCW) en (row,col)."""
    import plotly.graph_objects as go
    _rc = dict(row=row, col=col)
    add_probe_glyphs(fig, R, d["ang_x"], d["ang_y"], d["x_label"], d["y_label"],
                     with_cone=with_cone, with_top=False, row=row, col=col)
    fig.add_trace(go.Scatter(x=d["xv"], y=d["yv"], mode="lines",
                  line=dict(color=_NAVY, width=1.6), showlegend=False), **_rc)
    km = d["kmarks"]
    if km:
        _kx = [p[0] for p in km]; _ky = [p[1] for p in km]
        if d["order"] == 0 and len(km) > 1:
            fig.add_trace(go.Scatter(x=_kx, y=_ky, mode="markers",
                          marker=dict(color="#1f9d55", size=5,
                                      line=dict(color="white", width=0.7)),
                          showlegend=False), **_rc)
        else:
            _ks = (d["smax"] if d["smax"] else 1.0) * 0.07
            for _th in range(0, 360, 60):
                _r = np.radians(_th)
                fig.add_trace(go.Scatter(x=[_kx[0], _kx[0] + _ks * np.sin(_r)],
                              y=[_ky[0], _ky[0] + _ks * np.cos(_r)], mode="lines",
                              line=dict(color="#1f9d55", width=1.1),
                              hoverinfo="skip", showlegend=False), **_rc)
            fig.add_trace(go.Scatter(x=_kx[:1], y=_ky[:1], mode="markers",
                          marker=dict(color="#1f9d55", size=11,
                                      line=dict(color="white", width=1.4)),
                          showlegend=False), **_rc)
    fig.add_annotation(x=R * 1.33, y=R * 1.33, text=f"<b>{d['rot_api']}</b>",
                       showarrow=False, font=dict(color="#64748b", size=12,
                       family="IBM Plex Mono"), bgcolor="rgba(255,255,255,0.65)", **_rc)


def build_orbit_grids(bearings_norm: List[Dict[str, Any]], *,
                      fmode: str = "Direct",
                      png_fn=None) -> List[Dict[str, Any]]:
    """UNA figura por máquina (escala propia por máquina, celdas cuadradas),
    estilo idéntico al de Rotodinámica web. Devuelve
        [{"name": "Turbina", "png": bytes|None, "rows": [fila de tabla,...],
          "datas": [compute_orbit dict,...]}].

    bearings_norm: dicts con {label,x,y,x_label,y_label,unit,rpm?,fs_hz?,spr?}.
    png_fn: callable(fig, height)->bytes|None (p.ej. briefing_figures._png).
            Si None, no se rasteriza (png=None) y el llamador usa 'fig'.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Calcular cada órbita y agrupar por máquina.
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for b in bearings_norm:
        lbl = b.get("label", "") or b.get("bearing_label", "")
        d = compute_orbit(
            b.get("x"), b.get("y"), fs_hz=float(b.get("fs_hz", 0) or 0),
            spr=int(b.get("spr", 0) or 0), rpm=float(b.get("rpm", 0) or 0),
            bearing_label=lbl, x_label=b.get("x_label", "X") or "X",
            y_label=b.get("y_label", "Y") or "Y", unit=b.get("unit", "µm") or "µm",
            machine=b.get("machine", ""), stamp=b.get("stamp", ""), fmode=fmode)
        if not len(d["xv"]):
            continue
        groups.setdefault(machine_of(bearing_number(lbl)), []).append(d)

    out: List[Dict[str, Any]] = []
    for mm in MACH_ORDER:
        grp = groups.get(mm)
        if not grp:
            continue
        R = max(d["Rp"] for d in grp) * 1.5
        if R <= 0:
            R = 1.0
        ncol = min(len(grp), 2)
        nrow = (len(grp) + ncol - 1) // ncol
        fig = make_subplots(rows=nrow, cols=ncol,
                            subplot_titles=[d["bearing_label"] for d in grp],
                            horizontal_spacing=0.16, vertical_spacing=0.18)
        rows_tbl: List[Dict[str, Any]] = []
        for idx, d in enumerate(grp):
            r_ = idx // ncol + 1; c_ = idx % ncol + 1
            _add_orbit_cell(fig, d, R, r_, c_, with_cone=(len(grp) <= 2))
            _axref = f"x{idx + 1 if idx else ''}"
            _ayref = f"y{idx + 1 if idx else ''}"
            fig.update_xaxes(range=[-R, R], scaleanchor=_ayref, scaleratio=1,
                             showgrid=True, gridcolor="rgba(120,140,170,.12)",
                             zeroline=True, zerolinecolor="rgba(120,140,170,.45)",
                             row=r_, col=c_)
            fig.update_yaxes(range=[-R, R], showgrid=True,
                             gridcolor="rgba(120,140,170,.12)", zeroline=True,
                             zerolinecolor="rgba(120,140,170,.45)", row=r_, col=c_)
            rows_tbl.append(dict(
                bearing=d["bearing_label"], rpm=d["rpm"], precesion=d["precd"],
                giro=d["rot_api"], maq=d["maq"], major=d["major"], minor=d["minor"],
                ellip=d["ellip"], smax=d["smax"], unit=d["unit"]))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          showlegend=False, margin=dict(l=40, r=24, t=40, b=34),
                          font=dict(family="IBM Plex Sans", size=12, color="#334155"))
        _hh = int(1600 / max(ncol, 1) * nrow)
        png = png_fn(fig, height=_hh) if png_fn else None
        out.append({"name": MACH_NAME.get(mm, mm), "png": png,
                    "rows": rows_tbl, "datas": grp, "fig": fig})
    return out


__all__ = ["bearing_number", "probe_angles", "machine_of", "add_probe_glyphs",
           "compute_orbit", "build_orbit_grids", "MACH_NAME", "MACH_ORDER"]
