"""
core/modal/modal_web_plots.py — Motor de gráficos 3D / geometría del Modal WEB
=============================================================================
Funciones PURAS (sin Streamlit) extraídas de pages/18_Modal_Analysis.py para
sacar ~800 líneas del monolito: figuras Plotly de geometría, formas modales del
rotor y de superficies, interpolación IDW, y renderizado de GIF. La página las
importa con `from core.modal.modal_web_plots import *`.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # noqa: F401  (algunas figuras lo usan)

# Paleta compartida con la página (constantes de color).
NAVY = "#0F1E3D"; GREEN = "#16a34a"; BLUE = "#2563eb"
AMBER = "#f59e0b"; RED = "#dc2626"; SLATE = "#475569"

__all__ = [
    '_comp_color',
    '_cube',
    '_geometry_fig',
    '_mode_anim_fig',
    '_mode_shape_data',
    '_mode_machine_meshes',
    '_mode_dynamic_traces',
    '_mode_scene',
    '_mode_shape_fig',
    '_stn_key',
    '_layout_stations',
    '_station_disp_map',
    '_default_geometry',
    '_edges_xyz',
    '_geom_node_disp',
    '_geom_preview_fig',
    '_dense_mesh',
    '_rotor_is',
    '_cyl',
    '_disk',
    '_mode_rotor_fig',
    '_mode_geom_fig',
    '_idw',
    '_mode_surface_comps',
    '_surface_meshes',
    '_mode_surface_layout',
    '_mode_surface_fig',
    '_mode_surface_gif',
    '_mode_video_gif',
    '_mode_shape_gif',
    'NAVY',
    'GREEN',
    'BLUE',
    'AMBER',
    'RED',
    'SLATE',
]

def _comp_color(kind: str) -> str:
    k = (kind or "").lower()
    if "motor" in k or "engine" in k: return BLUE
    if "pump" in k or "bomba" in k:   return GREEN
    if "coupling" in k:               return "#334155"
    if "leg" in k or "pedestal" in k: return SLATE
    if "skid" in k:                   return "#a16207"
    return "#64748b"


def _cube(x0, x1, y0, y1, d):
    X = [x0, x1, x1, x0, x0, x1, x1, x0]
    Y = [-d, -d, d, d, -d, -d, d, d]
    Z = [y0, y0, y0, y0, y1, y1, y1, y1]
    i = [0, 0, 0, 4, 4, 6, 1, 1, 2, 3, 0, 4]
    j = [1, 2, 4, 5, 6, 7, 5, 2, 6, 7, 3, 5]
    k = [2, 3, 5, 6, 7, 3, 6, 6, 7, 4, 4, 1]
    return X, Y, Z, i, j, k


def _geometry_fig(lay, amp=None, show_sensors=True, height=520):
    fig = go.Figure()
    for c in lay.machine_components:
        col = getattr(c, "color", "") or _comp_color(c.kind)
        X, Y, Z, i, j, k = _cube(c.x0, c.x1, c.y0, c.y1, c.depth)
        fig.add_trace(go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=col,
                                opacity=0.55 if "skid" in c.kind.lower() else 0.9,
                                flatshading=True, hoverinfo="skip", showscale=False))
    if show_sensors and lay.active_points():
        pts = lay.active_points()
        colcode = amp if amp is not None else [_comp_color(p.component) for p in pts]
        fig.add_trace(go.Scatter3d(
            x=[p.x_norm for p in pts], y=[0.20] * len(pts), z=[p.y_norm for p in pts],
            mode="markers+text", text=[str(p.bnc) for p in pts], textposition="top center",
            textfont=dict(size=10, color=NAVY),
            marker=dict(size=7, color=colcode,
                        colorscale=("YlOrRd" if amp is not None else None),
                        line=dict(width=1, color="#0f172a")),
            hovertext=[f"{p.code} · {p.component} {p.position_ref} · BNC {p.bnc}" for p in pts],
            hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                      scene=dict(aspectmode="data", xaxis=dict(visible=False),
                                 yaxis=dict(visible=False), zaxis=dict(visible=False),
                                 camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))),
                      paper_bgcolor="white")
    return fig


_AX = {"A": (1, 0, 0), "X": (1, 0, 0), "H": (0, 1, 0), "Y": (0, 1, 0),
       "V": (0, 0, 1), "Z": (0, 0, 1)}


def _mode_anim_fig(lay, amps_signed, height=560):
    """Forma modal 3D ANIMADA: la máquina (tenue) + nodos de sensores que oscilan
    a lo largo de su DOF, coloreados por amplitud. Con botón Play."""
    pts = lay.active_points()
    if not pts or amps_signed is None or len(amps_signed) != len(pts):
        return _geometry_fig(lay, height=height)
    a = np.asarray(amps_signed, float)
    a = a / (np.max(np.abs(a)) or 1.0)
    col = np.abs(a)
    P0 = np.array([[p.x_norm, 0.20, p.y_norm] for p in pts], float)
    dirs = np.array([_AX.get(p.axis, (0, 0, 1)) for p in pts], float)
    dirs *= np.array([[-1.0 if p.dof.startswith("-") else 1.0] for p in pts])
    scale = 0.12
    base = go.Figure()
    # máquina tenue de fondo
    for c in lay.machine_components:
        cc = getattr(c, "color", "") or _comp_color(c.kind)
        X, Y, Z, i, j, k = _cube(c.x0, c.x1, c.y0, c.y1, c.depth)
        base.add_trace(go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=cc, opacity=0.18,
                                 flatshading=True, hoverinfo="skip", showscale=False))

    def _nodes(phase):
        d = P0 + (scale * a * np.sin(phase))[:, None] * dirs
        return d

    d0 = _nodes(0.0)
    base.add_trace(go.Scatter3d(x=d0[:, 0], y=d0[:, 1], z=d0[:, 2], mode="markers",
                   marker=dict(size=6, color=col, colorscale="YlOrRd", cmin=0, cmax=1,
                               line=dict(width=1, color="#0f172a")),
                   hovertext=[p.code for p in pts], hoverinfo="text", name="mode"))
    frames = []
    for f in range(24):
        ph = f / 24.0 * 2 * np.pi; d = _nodes(ph)
        frames.append(go.Frame(data=[go.Scatter3d(x=d[:, 0], y=d[:, 1], z=d[:, 2], mode="markers",
                      marker=dict(size=6, color=col, colorscale="YlOrRd", cmin=0, cmax=1,
                                  line=dict(width=1, color="#0f172a")))],
                      traces=[len(lay.machine_components)]))
    base.frames = frames
    base.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))),
        paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05, xanchor="left",
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True,
                                           transition=dict(duration=0), mode="immediate")]),
                     dict(label="⏸", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
    return base


def _mode_shape_data(lay, amps_signed, scale_mul=1.0):
    """Combina A/H/V de cada ESTACIÓN en un vector de movimiento 3D y arma un beam
    SUAVE (spline cúbico) que se deforma — mucho más pro que una línea quebrada."""
    pts = lay.active_points()
    if not pts or amps_signed is None or len(amps_signed) != len(pts):
        return None
    a = np.asarray(amps_signed, float); a = a / (np.max(np.abs(a)) or 1.0)
    stations = {}
    for p, ai in zip(pts, a):
        key = _stn_key(p)                        # por POSICIÓN (no por etiqueta, que puede repetirse)
        dvec = np.array(_AX.get(p.axis, (0, 0, 1)), float) * (-1.0 if p.dof.startswith("-") else 1.0) * float(ai)
        s = stations.setdefault(key, {"pos": np.array([p.x_norm, 0.20, p.y_norm], float), "disp": np.zeros(3)})
        s["disp"] += dvec
    order = sorted(stations.values(), key=lambda s: (s["pos"][0], s["pos"][2]))
    P0 = np.array([s["pos"] for s in order], float)
    DISP = np.array([s["disp"] for s in order], float)
    _mg = np.linalg.norm(DISP, axis=1); _pp = _mg[_mg > 0]
    _cn = float(np.percentile(_pp, 85)) if _pp.size else 1.0
    MAGn = np.clip(_mg / (_cn or (_mg.max() or 1.0)), 0.0, 1.0)
    span = float(np.ptp(P0[:, 0])) or 1.0
    scale = 0.24 * span / (np.max(np.linalg.norm(DISP, axis=1)) or 1.0) * scale_mul
    Ps, Ds = P0, DISP
    if len(order) >= 3:
        t = np.zeros(len(P0)); t[1:] = np.cumsum(np.linalg.norm(np.diff(P0, axis=0), axis=1))
        if t[-1] <= 0:
            t = np.arange(len(P0), dtype=float)
        tt = np.linspace(t[0], t[-1], 90)
        try:
            from scipy.interpolate import CubicSpline
            Ps = np.vstack([CubicSpline(t, P0[:, c])(tt) for c in range(3)]).T
            Ds = np.vstack([CubicSpline(t, DISP[:, c])(tt) for c in range(3)]).T
        except Exception:  # noqa: BLE001
            Ps, Ds = P0, DISP
    return {"P0": P0, "DISP": DISP, "MAGn": MAGn, "Ps": Ps, "Ds": Ds, "scale": scale}


def _mode_machine_meshes(lay, opacity=0.09):
    out = []
    for c in lay.machine_components:
        cc = getattr(c, "color", "") or _comp_color(c.kind)
        X, Y, Z, i, j, k = _cube(c.x0, c.x1, c.y0, c.y1, c.depth)
        out.append(go.Mesh3d(x=X, y=Y, z=Z, i=i, j=j, k=k, color=cc, opacity=opacity,
                             flatshading=True, hoverinfo="skip", showscale=False))
    return out


def _mode_dynamic_traces(d, phase, colorbar=True):
    """Beam suave + nodos, en una fase dada (para animar y para el GIF)."""
    beam = d["Ps"] + (d["scale"] * np.sin(phase)) * d["Ds"]
    nodes = d["P0"] + (d["scale"] * np.sin(phase)) * d["DISP"]
    mk = dict(size=7, color=d["MAGn"], colorscale="Turbo", cmin=0, cmax=1,
              line=dict(width=1, color="#0f172a"))
    if colorbar:
        mk["colorbar"] = dict(title="ampl", thickness=12, len=0.55, x=0.98)
    beam_tr = go.Scatter3d(x=beam[:, 0], y=beam[:, 1], z=beam[:, 2], mode="lines",
                           line=dict(color=BLUE, width=8), hoverinfo="skip", name="mode")
    node_tr = go.Scatter3d(x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode="markers",
                           marker=mk, hoverinfo="skip", name="nodes")
    return beam_tr, node_tr


def _mode_scene(height, paper="rgba(0,0,0,0)"):
    return dict(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                paper_bgcolor=paper,
                scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                           zaxis=dict(visible=False), camera=dict(eye=dict(x=1.6, y=1.4, z=0.85))))


def _mode_shape_fig(lay, amps_signed, height=580, scale_mul=1.0):
    d = _mode_shape_data(lay, amps_signed, scale_mul)
    if d is None:
        return _geometry_fig(lay, height=height)
    fig = go.Figure()
    for tr in _mode_machine_meshes(lay):
        fig.add_trace(tr)
    n_ctx = len(lay.machine_components)
    fig.add_trace(go.Scatter3d(x=d["Ps"][:, 0], y=d["Ps"][:, 1], z=d["Ps"][:, 2], mode="lines",
                  line=dict(color="rgba(148,163,184,.6)", width=4, dash="dot"),
                  hoverinfo="skip", name="undeformed"))
    beam0, node0 = _mode_dynamic_traces(d, np.pi / 2)
    fig.add_trace(beam0); fig.add_trace(node0)
    frames = []
    for f in range(30):
        ph = f / 30.0 * 2 * np.pi
        b, n = _mode_dynamic_traces(d, ph, colorbar=False)
        frames.append(go.Frame(data=[b, n], traces=[n_ctx + 1, n_ctx + 2]))
    fig.frames = frames
    lay_kw = _mode_scene(height)
    lay_kw["updatemenus"] = [dict(type="buttons", showactive=False, x=0.02, y=0.05, xanchor="left",
        buttons=[dict(label="▶ Play", method="animate",
                      args=[None, dict(frame=dict(duration=130, redraw=True), fromcurrent=True,
                                       transition=dict(duration=0), mode="immediate")]),
                 dict(label="⏸ Pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])]
    fig.update_layout(**lay_kw)
    return fig


def _stn_key(p):
    """Clave de estación por POSICIÓN física (las etiquetas del campo pueden repetirse)."""
    return f"{round(float(p.x_norm), 3)}|{round(float(p.y_norm), 3)}"


def _layout_stations(lay):
    """Estaciones de medición por posición física, con su posición 3D."""
    pts = lay.active_points()
    stations = {}
    for p in pts:
        key = _stn_key(p)
        stations.setdefault(key, {"label": key, "pos": np.array([p.x_norm, 0.20, p.y_norm], float)})
    return list(stations.values())


def _station_disp_map(lay, amps_signed):
    """{clave de estación (posición) → vector de desplazamiento 3D} para una forma modal."""
    pts = lay.active_points()
    a = np.asarray(amps_signed, float); a = a / (np.max(np.abs(a)) or 1.0)
    disp = {}
    for i, p in enumerate(pts):
        key = _stn_key(p)
        dvec = np.array(_AX.get(p.axis, (0, 0, 1)), float) * (-1.0 if p.dof.startswith("-") else 1.0) * float(a[i])
        disp[key] = disp.get(key, np.zeros(3)) + dvec
    return disp


_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def _default_geometry(lay):
    """Geometría inicial = wireframe de CAJAS (una por componente, 8 esquinas + 12
    aristas) + los nodos de sensores. Las esquinas son esclavas (interpoladas)."""
    nodes, lines = [], []
    for ci, c in enumerate(lay.machine_components):
        X, Y, Z, _i, _j, _k = _cube(c.x0, c.x1, c.y0, c.y1, c.depth)
        base = len(nodes)
        for v in range(8):
            nodes.append({"id": f"C{ci}_{v}", "x": round(float(X[v]), 4), "y": round(float(Y[v]), 4),
                          "z": round(float(Z[v]), 4), "sensor": ""})
        for a, b in _BOX_EDGES:
            lines.append([base + a, base + b])
    for s in _layout_stations(lay):
        nodes.append({"id": s["label"], "x": round(float(s["pos"][0]), 4), "y": round(float(s["pos"][1]), 4),
                      "z": round(float(s["pos"][2]), 4), "sensor": s["label"]})
    return {"nodes": nodes, "lines": lines}


def _edges_xyz(P, lines):
    ex, ey, ez = [], [], []
    n = len(P)
    for a, b in lines:
        if 0 <= a < n and 0 <= b < n:
            ex += [P[a, 0], P[b, 0], None]; ey += [P[a, 1], P[b, 1], None]; ez += [P[a, 2], P[b, 2], None]
    return ex, ey, ez


def _geom_node_disp(geom, disp_map):
    """Desplazamiento de cada nodo: si es sensor usa su estación; si es esclavo,
    interpola (IDW) desde los nodos-sensor."""
    nodes = geom["nodes"]
    P = np.array([[n["x"], n["y"], n["z"]] for n in nodes], float) if nodes else np.zeros((0, 3))
    sp, sd = [], []
    for n in nodes:
        if n.get("sensor") and n["sensor"] in disp_map:
            sp.append([n["x"], n["y"], n["z"]]); sd.append(disp_map[n["sensor"]])
    sp = np.array(sp, float) if sp else np.zeros((0, 3)); sd = np.array(sd, float) if sd else np.zeros((0, 3))
    ND = np.zeros((len(nodes), 3))
    for i, n in enumerate(nodes):
        if n.get("sensor") and n["sensor"] in disp_map:
            ND[i] = disp_map[n["sensor"]]
        elif len(sp):
            ND[i] = _idw(P[i:i + 1], sp, sd)[0]
    return P, ND


def _geom_preview_fig(lay, geom, height=460, show_machine=True):
    nodes = geom["nodes"]; lines = geom["lines"]
    P = np.array([[n["x"], n["y"], n["z"]] for n in nodes], float) if nodes else np.zeros((0, 3))
    fig = go.Figure()
    if show_machine:
        for tr in _mode_machine_meshes(lay, opacity=0.06):
            fig.add_trace(tr)
    if len(P):
        ex, ey, ez = _edges_xyz(P, lines)
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                      line=dict(color="#334155", width=4), hoverinfo="skip"))
        _sens = np.array([bool(n.get("sensor")) for n in nodes])
        _cols = np.where(_sens, "#2563eb", "#f59e0b")
        # etiquetar sólo los nodos-sensor (evita saturar con las esquinas de cajas)
        _txt = [n["id"] if n.get("sensor") else "" for n in nodes]
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text",
                      text=_txt, textposition="top center", textfont=dict(size=9, color="#1d4ed8"),
                      marker=dict(size=np.where(_sens, 6, 3), color=_cols, line=dict(width=1, color="#0f172a")),
                      hovertext=[n["id"] for n in nodes], hoverinfo="text"))
    fig.update_layout(**_mode_scene(height, paper="rgba(0,0,0,0)"))
    return fig


def _dense_mesh(P, surfaces, n=4):
    """Subdivide cada cara (quad o triángulo) en una malla n×n → muchos vértices
    para un gradiente FINO (como ARTeMIS)."""
    V, I, J, K = [], [], [], []
    for f in surfaces:
        if len(f) < 3:
            continue
        if len(f) >= 4:
            a, b, c, d = (np.asarray(P[f[0]], float), np.asarray(P[f[1]], float),
                          np.asarray(P[f[2]], float), np.asarray(P[f[3]], float))
            base = len(V); w = n + 1
            for iu in range(w):
                for iv in range(w):
                    u = iu / n; v = iv / n
                    V.append((1 - u) * (1 - v) * a + u * (1 - v) * b + u * v * c + (1 - u) * v * d)
            for iu in range(n):
                for iv in range(n):
                    p0 = base + iu * w + iv; p1 = base + (iu + 1) * w + iv; p2 = p1 + 1; p3 = p0 + 1
                    I += [p0, p0]; J += [p1, p2]; K += [p2, p3]
        else:
            base = len(V)
            for x in f:
                V.append(np.asarray(P[x], float))
            for t in range(1, len(f) - 1):
                I.append(base); J.append(base + t); K.append(base + t + 1)
    return (np.array(V, float) if V else np.zeros((0, 3))), I, J, K


def _rotor_is(lay):
    """¿La corrida es de PROXIMIDAD (rotor)? → sensores de desplazamiento (mil)."""
    pts = lay.active_points()
    return bool(pts) and sum(1 for p in pts if getattr(p, "meas_type", "A") == "D") >= max(2, len(pts) // 2)


def _cyl(xa, xb, r, na, nt, base):
    """Malla de un cilindro a lo largo de X: devuelve (verts, axial, I,J,K)."""
    verts, axc, I, J, K = [], [], [], [], []
    for k in range(na):
        xk = xa + (xb - xa) * (k / (na - 1) if na > 1 else 0)
        for j in range(nt):
            th = 2 * np.pi * j / nt
            verts.append([xk, r * np.cos(th), r * np.sin(th)]); axc.append(xk)
    for k in range(na - 1):
        for j in range(nt):
            j2 = (j + 1) % nt
            p0 = base + k * nt + j; p1 = base + (k + 1) * nt + j
            p2 = base + (k + 1) * nt + j2; p3 = base + k * nt + j2
            I += [p0, p0]; J += [p1, p2]; K += [p2, p3]
    return verts, axc, I, J, K


def _disk(xc, r, w, nt, base):
    """Disco SÓLIDO (impulsor) perpendicular al eje: dos caras + borde."""
    V, AX, I, J, K = [], [], [], [], []
    V.append([xc - w, 0, 0]); AX.append(xc - w); c0 = base
    V.append([xc + w, 0, 0]); AX.append(xc + w); c1 = base + 1
    r0 = base + 2; r1 = r0 + nt
    for j in range(nt):
        th = 2 * np.pi * j / nt; V.append([xc - w, r * np.cos(th), r * np.sin(th)]); AX.append(xc - w)
    for j in range(nt):
        th = 2 * np.pi * j / nt; V.append([xc + w, r * np.cos(th), r * np.sin(th)]); AX.append(xc + w)
    for j in range(nt):
        j2 = (j + 1) % nt
        I.append(c0); J.append(r0 + j); K.append(r0 + j2)          # cara frontal
        I.append(c1); J.append(r1 + j2); K.append(r1 + j)          # cara trasera
        I += [r0 + j, r0 + j]; J += [r1 + j, r1 + j2]; K += [r1 + j2, r0 + j2]  # borde
    return V, AX, I, J, K


def _mode_rotor_fig(lay, amps_signed, height=600, scale_mul=1.0, static=False, phase=np.pi / 2):
    """Forma modal del ROTOR (proximidad): eje + masa del motor + impulsores de la
    bomba, que FLEXIONA lateralmente según las sondas XY (X→radial horiz, Y→radial vert)."""
    pts = lay.active_points()
    if not pts or amps_signed is None or len(amps_signed) != len(pts):
        return _geometry_fig(lay, height=height)
    a = np.asarray(amps_signed, float); a = a / (np.max(np.abs(a)) or 1.0)
    # cojinetes: agrupar por POSICIÓN x (no por etiqueta: muchas corridas traen el mismo
    # component/position_ref en todos los puntos y colapsarían a una sola estación).
    bear = {}
    for p, ai in zip(pts, a):
        key = round(float(p.x_norm), 3)
        b = bear.setdefault(key, {"x": float(p.x_norm), "dy": 0.0, "dz": 0.0})
        d = float(ai) * (-1.0 if p.dof.startswith("-") else 1.0)
        if (p.axis or "").upper() in ("X", "H", "A"):
            b["dy"] += d
        else:
            b["dz"] += d
        b["x"] = float(p.x_norm)
    items = sorted(bear.values(), key=lambda b: b["x"])
    xs = np.array([b["x"] for b in items]); dys = np.array([b["dy"] for b in items]); dzs = np.array([b["dz"] for b in items])
    # suavizado ligero (3 puntos): evita que datos ruidosos "rompan"/retuerzan el eje
    if len(dys) >= 3:
        def _sm(v):
            w = v.copy(); w[1:-1] = 0.25 * v[:-2] + 0.5 * v[1:-1] + 0.25 * v[2:]; return w
        dys = _sm(dys); dzs = _sm(dzs)
    xmin, xmax = float(xs.min()), float(xs.max()); span = (xmax - xmin) or 1.0
    x0, x1 = xmin - 0.06 * span, xmax + 0.06 * span; L = x1 - x0
    rs = 0.020 * L                                   # radio del eje
    # rangos de motor y bomba (para masa e impulsores)
    def _crange(kws):
        for c in lay.machine_components:
            if any(w in (c.kind + " " + c.label).lower() for w in kws):
                return c.x0, c.x1
        return None
    mot = _crange(["motor"]); pmp = _crange(["pump", "bomba"])
    verts, axc, I, J, K = [], [], [], [], []

    def _add(v, ax, i, j, k):
        verts.extend(v); axc.extend(ax); I.extend(i); J.extend(j); K.extend(k)
    _add(*_cyl(x0, x1, rs, 60, 20, len(verts)))                        # eje
    if mot:                                                            # masa del motor
        _add(*_cyl(mot[0], mot[1], 0.055 * L, 18, 22, len(verts)))
    if pmp:                                                            # impulsores de la bomba (discos sólidos)
        n_imp = 10; pw = (pmp[1] - pmp[0])
        for ii in range(n_imp):
            xc = pmp[0] + pw * (ii + 0.5) / n_imp
            _add(*_disk(xc, 0.065 * L, 0.010 * L, 26, len(verts)))
    V0 = np.array(verts, float); AX = np.array(axc, float)
    # deflexión SUAVE: PCHIP (cúbica monótona) con clamp en extremos → curva de flexión
    # real, sin quiebres (kink) cuando el modo es "picudo" (p.ej. datos ruidosos).
    _xlo, _xhi = float(xs.min()), float(xs.max())

    def _smooth(q, yv):
        q = np.clip(np.asarray(q, float), _xlo, _xhi)
        if len(xs) >= 2:
            try:
                from scipy.interpolate import PchipInterpolator
                return PchipInterpolator(xs, yv, extrapolate=False)(q)
            except Exception:  # noqa: BLE001
                pass
        return np.interp(q, xs, yv)
    DY = _smooth(AX, dys); DZ = _smooth(AX, dzs)
    LAT = np.sqrt(DY ** 2 + DZ ** 2)
    _pos = LAT[LAT > 0]; cnorm = float(np.percentile(_pos, 85)) if _pos.size else 1.0
    MAGn = np.clip(LAT / (cnorm or 1.0), 0.0, 1.0)
    maxlat = float(np.sqrt(dys ** 2 + dzs ** 2).max()) or 1.0
    scale = 0.11 * L / maxlat * scale_mul

    def _defV(ph):
        out = V0.copy(); s = scale * np.sin(ph)
        out[:, 1] += DY * s; out[:, 2] += DZ * s
        return out

    def _surf_tr(dv):
        return go.Mesh3d(x=dv[:, 0], y=dv[:, 1], z=dv[:, 2], i=I, j=J, k=K, intensity=MAGn,
                         cmin=0, cmax=1, coloraxis="coloraxis", flatshading=False, opacity=1.0,
                         lighting=dict(ambient=0.82, diffuse=0.5, specular=0.12), hoverinfo="skip")

    _xc = np.linspace(x0, x1, 60); _ycb = _smooth(_xc, dys); _zcb = _smooth(_xc, dzs)

    def _center_tr(ph):
        s = scale * np.sin(ph)
        return go.Scatter3d(x=_xc, y=_ycb * s, z=_zcb * s, mode="lines",
                            line=dict(color="#0f172a", width=3), hoverinfo="skip")

    fig = go.Figure()
    _ph0 = float(phase)
    fig.add_trace(_surf_tr(_defV(_ph0))); _isf = len(fig.data) - 1
    fig.add_trace(_center_tr(_ph0)); _icl = len(fig.data) - 1
    # marcadores de cojinete (sensores)
    fig.add_trace(go.Scatter3d(x=xs, y=[0] * len(xs), z=[0] * len(xs), mode="markers+text",
                  text=[f"B{i+1}" for i in range(len(xs))], textposition="top center",
                  textfont=dict(size=10, color="#0f172a"), marker=dict(size=4, color="#0f172a"),
                  hoverinfo="skip"))
    if not static:
        frames = []
        for f in range(40):
            ph = f / 20.0 * 2 * np.pi
            frames.append(go.Frame(data=[_surf_tr(_defV(ph)), _center_tr(ph)], traces=[_isf, _icl]))
        fig.frames = frames
    lay_kw = _mode_scene(height)
    if static:
        lay_kw["coloraxis"] = dict(colorscale="Jet", cmin=0, cmax=1,
                                   colorbar=dict(thickness=12, len=0.6, x=0.98,
                                                 tickvals=[0, 1], ticktext=["0", "Max"], title="ampl"))
    else:
        lay_kw["coloraxis"] = dict(colorscale="Jet", cmin=0, cmax=1, showscale=False)
    if not static:
        lay_kw["updatemenus"] = [dict(type="buttons", showactive=False, x=0.02, y=0.05, xanchor="left",
            buttons=[dict(label="▶ Play", method="animate",
                          args=[None, dict(frame=dict(duration=130, redraw=True), fromcurrent=True,
                                           transition=dict(duration=0), mode="immediate")]),
                     dict(label="⏸ Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])]
    # Centrar el rotor en X (que quede centrado en el marco, no corrido a un lado) y
    # alejar un poco la cámara para que NO se salga de cuadro durante la animación.
    _xmid = 0.5 * (x0 + x1)
    for _tr in fig.data:
        if getattr(_tr, "x", None) is not None:
            _tr.x = np.asarray(_tr.x, float) - _xmid
    for _fr in (fig.frames or []):
        for _tr in _fr.data:
            if getattr(_tr, "x", None) is not None:
                _tr.x = np.asarray(_tr.x, float) - _xmid
    lay_kw["scene"]["camera"] = dict(eye=dict(x=1.9, y=1.65, z=1.0),
                                     center=dict(x=0, y=0, z=0))
    # Rangos de eje FIJOS que cubren la deflexión máxima (±) → el rotor NO se sale del
    # cuadro durante la animación (encaja todos los frames, centrado).
    _allv = np.vstack([_defV(np.pi / 2), _defV(-np.pi / 2), V0]).astype(float)
    _allv[:, 0] -= _xmid
    def _rng(_a, _b, _f=0.10):
        _m = (_b - _a) * _f + 1e-6; return [_a - _m, _b + _m]
    lay_kw["scene"]["xaxis"] = dict(visible=False, range=_rng(_allv[:, 0].min(), _allv[:, 0].max()))
    lay_kw["scene"]["yaxis"] = dict(visible=False, range=_rng(_allv[:, 1].min(), _allv[:, 1].max()))
    lay_kw["scene"]["zaxis"] = dict(visible=False, range=_rng(_allv[:, 2].min(), _allv[:, 2].max()))
    fig.update_layout(**lay_kw)
    return fig


def _mode_geom_fig(lay, geom, amps_signed, height=600, scale_mul=1.0, static=False, phase=np.pi / 2):
    """Forma modal animada sobre la GEOMETRÍA del campo (estilo ARTeMIS): superficie
    sólida con malla densa (gradiente Jet), aristas, flechas de DOF por eje y triada.
    static=True → sin animación ni botón Play, con colorbar (para el PDF del reporte)."""
    nodes = geom.get("nodes") or []
    lines = geom.get("lines") or []
    surfaces = geom.get("surfaces") or []
    if not nodes:
        return _mode_surface_fig(lay, amps_signed, height, scale_mul)
    disp_map = _station_disp_map(lay, amps_signed)
    P, ND = _geom_node_disp(geom, disp_map)
    span = float(np.ptp(P[:, 0])) or 1.0
    MAG = np.linalg.norm(ND, axis=1)
    # contraste automático (percentil) para que el gradiente se lea aunque el modo sea "picudo"
    _pos = MAG[MAG > 0]
    cnorm = float(np.percentile(_pos, 85)) if _pos.size else 1.0
    cnorm = cnorm or (MAG.max() or 1.0)
    MAGn = np.clip(MAG / cnorm, 0.0, 1.0)
    scale = 0.16 * span / (MAG.max() or 1.0) * scale_mul
    # Malla densa por interpolación BILINEAL de las 4 esquinas de cada cara: la
    # cuadrícula se deforma coherente (no se "derrite") y agrega MUCHAS líneas (ARTeMIS).
    N = 5
    Vr, Vd, Vi, I, J, K, LP = [], [], [], [], [], [], []
    for f in surfaces:
        if len(f) < 4:
            continue
        Pc = [P[f[0]], P[f[1]], P[f[2]], P[f[3]]]
        Dc = [ND[f[0]], ND[f[1]], ND[f[2]], ND[f[3]]]
        Ic = [MAGn[f[0]], MAGn[f[1]], MAGn[f[2]], MAGn[f[3]]]   # color = bilineal de esquinas ya normalizadas
        base = len(Vr); w = N + 1
        for iu in range(w):
            for iv in range(w):
                u = iu / N; v = iv / N
                bw = ((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v)
                Vr.append(bw[0] * Pc[0] + bw[1] * Pc[1] + bw[2] * Pc[2] + bw[3] * Pc[3])
                Vd.append(bw[0] * Dc[0] + bw[1] * Dc[1] + bw[2] * Dc[2] + bw[3] * Dc[3])
                Vi.append(bw[0] * Ic[0] + bw[1] * Ic[1] + bw[2] * Ic[2] + bw[3] * Ic[3])
        for iu in range(N):
            for iv in range(N):
                p0 = base + iu * w + iv; p1 = base + (iu + 1) * w + iv; p2 = p1 + 1; p3 = p0 + 1
                I += [p0, p0]; J += [p1, p2]; K += [p2, p3]
        for iu in range(w):
            for iv in range(N):
                LP.append((base + iu * w + iv, base + iu * w + iv + 1))
        for iv in range(w):
            for iu in range(N):
                LP.append((base + iu * w + iv, base + (iu + 1) * w + iv))
    Vr = np.array(Vr, float) if Vr else np.zeros((0, 3))
    Vd = np.array(Vd, float) if Vd else np.zeros((0, 3))
    Vi = np.array(Vi, float) if Vi else np.zeros((0,))
    has_surf = len(Vr) > 0

    def _defVr(ph):
        return Vr + (scale * np.sin(ph)) * Vd

    def _surf_tr(dv):
        return go.Mesh3d(x=dv[:, 0], y=dv[:, 1], z=dv[:, 2], i=I, j=J, k=K, intensity=Vi,
                         cmin=0, cmax=1, coloraxis="coloraxis", flatshading=False, opacity=1.0,
                         lighting=dict(ambient=0.82, diffuse=0.5, specular=0.12), hoverinfo="skip")

    def _grid_tr(dv):
        ex, ey, ez = [], [], []
        for a, b in LP:
            ex += [dv[a, 0], dv[b, 0], None]; ey += [dv[a, 1], dv[b, 1], None]; ez += [dv[a, 2], dv[b, 2], None]
        return go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                            line=dict(color="rgba(15,23,42,.85)", width=2), hoverinfo="skip")

    fig = go.Figure()
    if has_surf:
        fig.add_trace(_surf_tr(_defVr(float(phase)))); _is = len(fig.data) - 1
        fig.add_trace(_grid_tr(_defVr(float(phase)))); _ig = len(fig.data) - 1
    if not static:
        frames = []
        for f in range(40):
            ph = f / 20.0 * 2 * np.pi; dv = _defVr(ph)
            data, tr = [], []
            if has_surf:
                data.append(_surf_tr(dv)); tr.append(_is)
                data.append(_grid_tr(dv)); tr.append(_ig)
            frames.append(go.Frame(data=data, traces=tr))
        fig.frames = frames

    # --- estáticos: flechas de DOF por eje (color) + nodos numerados + triada X/Y/Z ---
    _acol = {"A": "#db2777", "X": "#db2777", "H": "#16a34a", "Y": "#16a34a", "V": "#2563eb", "Z": "#2563eb"}
    alen = 0.08 * span; _by = {}
    for p in lay.active_points():
        ax = _AX.get(p.axis, (0, 0, 1)); s = -1.0 if p.dof.startswith("-") else 1.0
        col = _acol.get(p.axis, "#0f172a"); b = _by.setdefault(col, {"x": [], "y": [], "z": [], "u": [], "v": [], "w": []})
        b["x"].append(p.x_norm); b["y"].append(0.20); b["z"].append(p.y_norm)
        b["u"].append(ax[0] * s * alen); b["v"].append(ax[1] * s * alen); b["w"].append(ax[2] * s * alen)
    for col, b in _by.items():
        fig.add_trace(go.Cone(x=b["x"], y=b["y"], z=b["z"], u=b["u"], v=b["v"], w=b["w"], anchor="tail",
                      sizemode="absolute", sizeref=alen * 0.5, showscale=False,
                      colorscale=[[0, col], [1, col]], hoverinfo="skip"))
    _sm = np.array([bool(nd.get("sensor")) for nd in nodes])
    if _sm.any():
        SP = P[_sm]
        fig.add_trace(go.Scatter3d(x=SP[:, 0], y=SP[:, 1], z=SP[:, 2], mode="markers+text",
                      text=[str(i + 1) for i in range(len(SP))], textposition="top center",
                      textfont=dict(size=9, color="#0f172a"), marker=dict(size=3, color="#0f172a"),
                      hoverinfo="skip"))
    _o = np.array([P[:, 0].min(), P[:, 1].min() - 0.06 * span, P[:, 2].min()]); _tl = 0.14 * span
    for vec, c, nm in (((1, 0, 0), "#dc2626", "X"), ((0, 1, 0), "#16a34a", "Y"), ((0, 0, 1), "#2563eb", "Z")):
        e = _o + np.array(vec, float) * _tl
        fig.add_trace(go.Scatter3d(x=[_o[0], e[0]], y=[_o[1], e[1]], z=[_o[2], e[2]], mode="lines",
                      line=dict(color=c, width=4), hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=[e[0]], y=[e[1]], z=[e[2]], mode="text", text=[nm],
                      textfont=dict(size=12, color=c), hoverinfo="skip"))

    lay_kw = _mode_scene(height)
    if has_surf:
        if static:
            lay_kw["coloraxis"] = dict(colorscale="Jet", cmin=0, cmax=1,
                                       colorbar=dict(thickness=12, len=0.6, x=0.98,
                                                     tickvals=[0, 1], ticktext=["0", "Max"], title="ampl"))
        else:
            lay_kw["coloraxis"] = dict(colorscale="Jet", cmin=0, cmax=1, showscale=False)
    if static:
        fig.update_layout(**lay_kw)
        return fig
    lay_kw["updatemenus"] = [dict(type="buttons", showactive=False, x=0.02, y=0.05, xanchor="left",
        buttons=[dict(label="▶ Play", method="animate",
                      args=[None, dict(frame=dict(duration=130, redraw=True), fromcurrent=True,
                                       transition=dict(duration=0), mode="immediate")]),
                 dict(label="⏸ Pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])]
    fig.update_layout(**lay_kw)
    return fig


def _idw(V, P0, DISP, power=2.0):
    """Interpolación por distancia inversa: desplazamiento en cada vértice a partir
    de las estaciones medidas."""
    d = np.linalg.norm(V[:, None, :] - P0[None, :, :], axis=2)
    w = 1.0 / (d ** power + 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    return w @ DISP


def _mode_surface_comps(lay, amps_signed, scale_mul=1.0):
    """Para cada componente (caja) interpola el campo de desplazamiento a sus vértices
    → superficie sólida que se deforma y se colorea por amplitud (estilo ARTeMIS)."""
    d = _mode_shape_data(lay, amps_signed, scale_mul)
    if d is None:
        return None
    P0, DISP = d["P0"], d["DISP"]
    comps, cmax = [], 1e-9
    for c in lay.machine_components:
        X, Y, Z, i, j, k = _cube(c.x0, c.x1, c.y0, c.y1, c.depth)
        V = np.column_stack([X, Y, Z]).astype(float)
        DV = _idw(V, P0, DISP)
        mag = np.linalg.norm(DV, axis=1)
        cmax = max(cmax, float(mag.max()))
        comps.append({"V": V, "DV": DV, "mag": mag, "i": i, "j": j, "k": k})
    allV = np.vstack([cc["V"] for cc in comps])
    span = float(np.ptp(allV[:, 0])) or 1.0
    maxd = max((np.linalg.norm(cc["DV"], axis=1).max() for cc in comps), default=1.0) or 1.0
    scale = 0.16 * span / maxd * scale_mul
    return {"comps": comps, "cmax": cmax, "scale": scale, "span": span,
            "P0": P0, "DISP": d["DISP"], "MAGn": d["MAGn"]}


def _surface_meshes(s, phase):
    out = []
    for comp in s["comps"]:
        Vd = comp["V"] + (s["scale"] * np.sin(phase)) * comp["DV"]
        out.append(go.Mesh3d(x=Vd[:, 0], y=Vd[:, 1], z=Vd[:, 2],
                   i=comp["i"], j=comp["j"], k=comp["k"], intensity=comp["mag"],
                   cmin=0, cmax=s["cmax"], coloraxis="coloraxis", flatshading=False,
                   opacity=1.0, lighting=dict(ambient=0.82, diffuse=0.5, specular=0.12),
                   hoverinfo="skip"))
    return out


def _mode_surface_layout(height, cmax):
    lay_kw = _mode_scene(height)
    lay_kw["coloraxis"] = dict(colorscale="Jet", cmin=0, cmax=cmax,
                               colorbar=dict(title="ampl", thickness=14, len=0.6, x=0.98))
    return lay_kw


def _mode_surface_fig(lay, amps_signed, height=600, scale_mul=1.0, annotate=True):
    s = _mode_surface_comps(lay, amps_signed, scale_mul)
    if s is None:
        return _geometry_fig(lay, height=height)
    fig = go.Figure()
    for tr in _surface_meshes(s, np.pi / 2):
        fig.add_trace(tr)
    ncomp = len(s["comps"])
    # nodos de medición + números + flechas de dirección (DOF) estilo ARTeMIS
    P0, DISP = s["P0"], s["DISP"]
    if annotate and len(P0):
        alen = 0.09 * s["span"]
        norm = np.linalg.norm(DISP, axis=1, keepdims=True); norm[norm == 0] = 1.0
        U = DISP / norm * alen
        fig.add_trace(go.Cone(x=P0[:, 0], y=P0[:, 1], z=P0[:, 2], u=U[:, 0], v=U[:, 1], w=U[:, 2],
                      anchor="tail", sizemode="absolute", sizeref=alen * 0.5, showscale=False,
                      colorscale=[[0, "#0f172a"], [1, "#0f172a"]], hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=P0[:, 0], y=P0[:, 1], z=P0[:, 2], mode="markers+text",
                      text=[str(i + 1) for i in range(len(P0))], textposition="top center",
                      textfont=dict(size=10, color="#0f172a"),
                      marker=dict(size=3, color="#0f172a"), hoverinfo="skip"))
    else:
        fig.add_trace(go.Scatter3d(x=P0[:, 0], y=P0[:, 1], z=P0[:, 2], mode="markers",
                      marker=dict(size=3, color="#0f172a"), hoverinfo="skip"))
    frames = []
    for f in range(28):
        ph = f / 28.0 * 2 * np.pi
        frames.append(go.Frame(data=_surface_meshes(s, ph), traces=list(range(ncomp))))
    fig.frames = frames
    lay_kw = _mode_surface_layout(height, s["cmax"])
    lay_kw["updatemenus"] = [dict(type="buttons", showactive=False, x=0.02, y=0.05, xanchor="left",
        buttons=[dict(label="▶ Play", method="animate",
                      args=[None, dict(frame=dict(duration=130, redraw=True), fromcurrent=True,
                                       transition=dict(duration=0), mode="immediate")]),
                 dict(label="⏸ Pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])]
    fig.update_layout(**lay_kw)
    return fig


def _mode_surface_gif(lay, amps_signed, scale_mul=1.0, n=22, w=680, h=500):
    import io
    from PIL import Image
    s = _mode_surface_comps(lay, amps_signed, scale_mul)
    if s is None:
        return None
    imgs = []
    for f in range(n):
        ph = f / n * 2 * np.pi
        fig = go.Figure()
        for tr in _surface_meshes(s, ph):
            fig.add_trace(tr)
        lay_kw = _mode_surface_layout(h, s["cmax"]); lay_kw["paper_bgcolor"] = "white"
        fig.update_layout(**lay_kw)
        imgs.append(Image.open(io.BytesIO(fig.to_image(format="png", width=w, height=h, scale=1))).convert("RGB"))
    if not imgs:
        return None
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=60, loop=0)
    return buf.getvalue()


def _mode_video_gif(lay, geom, amps_signed, is_rotor, scale_mul=1.0, n=14, w=640, h=440):
    """GIF animado real de la forma modal usando el MISMO render que se ve (rotor o
    carcasa): renderiza n fases del ciclo y las une en un GIF que hace bucle. n moderado
    para que el render (kaleido) termine rápido. Devuelve bytes GIF o None."""
    import io
    from PIL import Image
    imgs = []
    for f in range(n):
        ph = f / n * 2 * np.pi
        try:
            fig = (_mode_rotor_fig(lay, amps_signed, height=h, scale_mul=scale_mul, static=True, phase=ph)
                   if is_rotor else
                   _mode_geom_fig(lay, geom, amps_signed, height=h, scale_mul=scale_mul, static=True, phase=ph))
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
            png = fig.to_image(format="png", width=w, height=h, scale=1)
            imgs.append(Image.open(io.BytesIO(png)).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
    if len(imgs) < 2:
        return None
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=110, loop=0, optimize=True)
    return buf.getvalue()


def _mode_shape_gif(lay, amps_signed, scale_mul=1.0, n=22, w=620, h=460):
    """Renderiza la deformación a un GIF descargable (video decente)."""
    import io
    from PIL import Image
    d = _mode_shape_data(lay, amps_signed, scale_mul)
    if d is None:
        return None
    imgs = []
    for f in range(n):
        ph = f / n * 2 * np.pi
        fig = go.Figure()
        for tr in _mode_machine_meshes(lay):
            fig.add_trace(tr)
        fig.add_trace(go.Scatter3d(x=d["Ps"][:, 0], y=d["Ps"][:, 1], z=d["Ps"][:, 2], mode="lines",
                      line=dict(color="rgba(148,163,184,.6)", width=4, dash="dot"), hoverinfo="skip"))
        b, nd = _mode_dynamic_traces(d, ph, colorbar=False)
        fig.add_trace(b); fig.add_trace(nd)
        fig.update_layout(**_mode_scene(h, paper="white"))
        imgs.append(Image.open(io.BytesIO(fig.to_image(format="png", width=w, height=h, scale=1))).convert("RGB"))
    if not imgs:
        return None
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=60, loop=0)
    return buf.getvalue()


