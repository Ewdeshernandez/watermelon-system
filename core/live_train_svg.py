"""
core/live_train_svg.py
======================

Tren de Machines EN VIVO como SVG estático → reporte PDF (vector, svglib).

Mismo dibujo que Live Monitoring y Machines: los SÍMBOLOS de equipos y los
degradados se leen de `native/field_ui/builder.html` (fuente única) y la
geometría replica `renderTrain` + `renderLive` del builder (#live):
  · sondas de eje arriba, en grupo por cojinete, Y antes que X (Bently: Y = CH1)
  · carcasa abajo · auxiliares en su caja (bomba de lubricación, arrancador…)
  · cojinete coloreado por la peor severidad; barra = % de la alerta
    (ámbar desde 90 %)

Si la máquina no tiene tren en Machines, `train_drawing_for` cae al dibujo
anterior (core.train_svg) → ningún reporte se queda sin diagrama.
"""

from __future__ import annotations

import functools
import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BUILDER = Path(__file__).resolve().parents[1] / "native" / "field_ui" / "builder.html"

SEVC = {"ok": "#34d399", "alert": "#f5a524", "danger": "#ef4444", "off": "#5d7189"}
SEVR = {"off": 0, "ok": 1, "alert": 2, "danger": 3}
KCOL = {"prox": "#a78bfa", "vel": "#22b8cf", "accel": "#f87171", "kph": "#f59e0b"}
TOP, CH, CW, GAP = 96, 40, 82, 6
# Listas de fuentes: cada motor toma la primera que tenga — app móvil / resvg
# (IBM Plex), navegador, svglib del PDF web (Courier / Helvetica base de reportlab).
MONO = "'IBM Plex Mono', Courier, monospace"
SANS = "'IBM Plex Sans', Helvetica, sans-serif"
MONO_A = MONO.replace("'", "&#39;")
SANS_A = SANS.replace("'", "&#39;")


@functools.lru_cache(maxsize=1)
def _builder_assets() -> Tuple[str, Dict[str, Tuple[str, int]], Dict[str, Tuple[float, float, str]]]:
    """(defs sin símbolos, LIB {key:(sym,w)}, símbolos {id:(vbW, vbH, contenido)})."""
    h = _BUILDER.read_text(encoding="utf-8")
    defs = re.search(r"<defs>(.*?)</defs>", h, re.S).group(1)
    lib = {m.group(1): (m.group(2), int(m.group(3))) for m in re.finditer(
        r"(\w+):\{label:'[^']*',sym:'([\w-]+)'[^}]*?w:(\d+)\}", h)}
    syms: Dict[str, Tuple[float, float, str]] = {}
    for m in re.finditer(r'<symbol id="([\w-]+)"[^>]*viewBox="0 0 ([\d.]+) ([\d.]+)"[^>]*>(.*?)</symbol>', defs, re.S):
        body = m.group(4)
        # svglib no soporta degradado con opacidad: la sombra radial sale negra
        # sólida → sombra plana semitransparente (mismo efecto visual).
        body = body.replace('fill="url(#shadow)"', 'fill="#000000" fill-opacity="0.28"')
        syms[m.group(1)] = (float(m.group(2)), float(m.group(3)), body)
    grads = re.sub(r"<symbol\b.*?</symbol>", "", defs, flags=re.S)
    return grads, lib, syms


def _esc(s: Any) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else (f"{v:.1f}" if a >= 10 else f"{v:.2f}")


def _unit_short(u: str) -> str:
    return (u or "").replace(" rms", "").replace(" pp", "").replace(" pk", "")


def _axis(n: str) -> int:
    t = n[1:] if len(n) > 1 else ""
    return 0 if "Y" in t else (1 if "X" in t else 2)


def _train_items(layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Igual que wmLoadSpec: acople entre máquinas + numeración continua."""
    train: List[Dict[str, Any]] = []
    for m in layout.get("machines") or []:
        if train:
            train.append({"key": "coupling", "brg": 0, "bt": "none"})
        train.append({"key": m.get("type"), "brg": int(m.get("bearings") or 0),
                      "bt": m.get("bearing_type") or "journal"})
    n = 0
    for c in train:
        c["bno"] = list(range(n + 1, n + 1 + c["brg"]))
        n += c["brg"]
    return train


def build_live_train_svg(layout: Dict[str, Any], live: Dict[str, Any]) -> str:
    defs, lib, syms = _builder_assets()
    train = [c for c in _train_items(layout) if c["key"] in lib]
    aux = list(layout.get("aux") or [])
    out: List[str] = []
    pad, gap, y0 = 24, -6, 96
    x, pos = pad, []
    for c in train:
        w = lib[c["key"]][1]
        pos.append((c, x, w))
        x += w + gap
    total = x + pad
    vbw = max(940, total)

    # ---- tren (dentro de g translate(0,TOP)) ----
    g: List[str] = []
    g.append(f'<line x1="{pad-6}" y1="98" x2="{total-pad+6}" y2="98" stroke="#2a3b4d" stroke-width="8" stroke-linecap="round"/>')
    g.append(f'<line x1="{pad-6}" y1="96" x2="{total-pad+6}" y2="96" stroke="url(#shaft)" stroke-width="5" stroke-linecap="round"/>')
    bpos: Dict[int, float] = {}
    for c, px, w in pos:
        h = w * 0.66
        yy = y0 - h * 0.52
        sym = lib[c["key"]][0]
        if sym in syms:
            # = <use> del navegador: símbolo escalado "meet" centrado en (w × h)
            vw, vh, body = syms[sym]
            k = min(w / vw, h / vh)
            tx, ty = px + (w - vw * k) / 2, yy + (h - vh * k) / 2
            g.append(f'<g transform="translate({tx:.2f},{ty:.2f}) scale({k:.4f})">{body}</g>')
        bs = c["bno"]
        for bi, no in enumerate(bs):
            bx = px + w * (0.30 + 0.40 * (bi / (len(bs) - 1) if len(bs) > 1 else 0.5))
            bpos[no] = bx
    probes_all = list((live or {}).get("probes") or [])
    auxn: Dict[str, int] = {}
    for ai, a in enumerate(aux):
        for p in a.get("probes") or []:
            auxn[str(p.get("n") if isinstance(p, dict) else p).upper()] = ai
    aux_probes = [p for p in probes_all if str(p.get("n")).upper() in auxn]
    probes = [p for p in probes_all if bpos.get(p.get("brg")) is not None
              and str(p.get("n")).upper() not in auxn]
    worst: Dict[int, str] = {}
    for p in probes:
        b, s = p["brg"], p.get("sev") or "off"
        if b not in worst or SEVR[s] > SEVR[worst[b]]:
            worst[b] = s
    for c, px, w in pos:
        for no in c["bno"]:
            bx = bpos[no]
            col = SEVC.get(worst.get(no, ""), "#2f84c4" if c["bt"] == "journal" else "#27c08a")
            if worst.get(no) in ("alert", "danger"):
                g.append(f'<circle cx="{bx:.1f}" cy="96" r="20" fill="{col}" fill-opacity="0.22"/>')
            g.append(f'<circle cx="{bx:.1f}" cy="96" r="12" fill="#0a141e" stroke="{col}" stroke-width="3"/>')
            g.append(f'<text x="{bx:.1f}" y="100" text-anchor="middle" font-family="{MONO_A}" font-size="11" font-weight="bold" fill="#ffffff">{no}</text>')

    def chip(p: Dict[str, Any], cx: float, cy: float) -> None:
        sc = SEVC.get(p.get("sev") or "off", SEVC["off"])
        v = p.get("v")
        out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{CW}" height="{CH}" rx="7" fill="#0c1726" stroke="{sc}" stroke-width="1.4"/>')
        out.append(f'<circle cx="{cx+10:.1f}" cy="{cy+12:.1f}" r="3.6" fill="{KCOL.get(p.get("k"), "#8aa0b6")}"/>')
        out.append(f'<text x="{cx+18:.1f}" y="{cy+15.5:.1f}" font-family="{MONO_A}" font-size="10" font-weight="bold" fill="#cfe0f1">{_esc(p.get("n"))}</text>')
        if p.get("ang"):
            out.append(f'<text x="{cx+CW-6:.1f}" y="{cy+15.5:.1f}" text-anchor="end" font-family="{MONO_A}" font-size="8.5" fill="#6f86a0">{_esc(p["ang"])}</text>')
        fv = _fmt(v)
        out.append(f'<text x="{cx+8:.1f}" y="{cy+32:.1f}" font-family="{SANS_A}" font-size="14" font-weight="bold" fill="#ffffff">{_esc(fv)}</text>')
        out.append(f'<text x="{cx+8+len(fv)*8.2+10:.1f}" y="{cy+32:.1f}" font-family="{MONO_A}" font-size="9" fill="#8aa0b6">{_esc(_unit_short(p.get("u")))}</text>')
        alarm = p.get("alarm") or 0
        pct = min(1.25, v / alarm) if (v is not None and alarm) else 0.0
        bc = SEVC["alert"] if (p.get("sev") == "ok" and pct >= 0.9) else sc
        out.append(f'<rect x="{cx+8:.1f}" y="{cy+CH-5:.1f}" width="{CW-16}" height="2.5" rx="1.2" fill="#1d3046"/>')
        out.append(f'<rect x="{cx+8:.1f}" y="{cy+CH-5:.1f}" width="{max(2.0,(CW-16)*min(1.0,pct)):.1f}" height="2.5" rx="1.2" fill="{bc}"/>')

    def lane(lst: List[Dict[str, Any]], y_row: float, up: bool) -> int:
        lst = sorted(lst, key=lambda p: (bpos[p["brg"]], _axis(str(p["n"])), str(p["n"])))
        groups: List[Dict[str, Any]] = []
        for p in lst:
            if groups and groups[-1]["brg"] == p["brg"]:
                groups[-1]["items"].append(p)
            else:
                groups.append({"brg": p["brg"], "items": [p]})
        gg = GAP * 2
        tot = 0.0
        for gr in groups:
            gr["w"] = len(gr["items"]) * CW + (len(gr["items"]) - 1) * GAP
            tot += gr["w"] + gg
            gr["r"] = 0
        one_row = (tot - gg) <= vbw - 12
        nx = 6.0
        for gr in groups:
            gr["x"] = max(bpos[gr["brg"]] - gr["w"] / 2, nx)
            nx = gr["x"] + gr["w"] + gg
        if one_row:
            lim = vbw - 6
            for gr in reversed(groups):
                if gr["x"] + gr["w"] > lim:
                    gr["x"] = lim - gr["w"]
                lim = gr["x"] - gg
        else:
            nx2 = 6.0
            for gr in groups:
                if gr["x"] + gr["w"] > vbw - 6:
                    gr["r"] = 1
                    gr["x"] = min(max(bpos[gr["brg"]] - gr["w"] / 2, nx2), vbw - 6 - gr["w"])
                    nx2 = gr["x"] + gr["w"] + gg
        rows = 2 if any(gr["r"] for gr in groups) else 1
        for gr in groups:
            for i, p in enumerate(gr["items"]):
                cx = gr["x"] + i * (CW + GAP)
                cy = y_row - gr["r"] * (CH + 8) if up else y_row + gr["r"] * (CH + 8)
                bx = bpos[p["brg"]]
                sc = SEVC.get(p.get("sev") or "off", SEVC["off"])
                ya = cy + CH if up else cy
                yb = TOP + 96 - 14 if up else TOP + 96 + 14
                ym = TOP - 4 if up else TOP + 190 - 38
                out.append(f'<path d="M{cx+CW/2:.1f} {ya:.1f} L{cx+CW/2:.1f} {ym:.1f} L{bx:.1f} {ym:.1f} L{bx:.1f} {yb:.1f}" '
                           f'fill="none" stroke="{sc}" stroke-width="1.2" stroke-opacity="0.55"/>')
                chip(p, cx, cy)
        return rows

    up = [p for p in probes if p.get("k") == "prox"]
    dn = [p for p in probes if p.get("k") != "prox"]
    n_up = lane(up, TOP - CH - 8, True) if up else 1
    n_dn = lane(dn, TOP + 190 - 30, False) if dn else 0
    extra_top = max(0, (n_up - 1) * (CH + 8))
    h_total = TOP + 190 - 30 + n_dn * (CH + 8) + 8
    if aux:
        abh = CH + 34
        ay, ax = h_total + 22, 6.0
        out.append(f'<text x="6" y="{ay-8:.1f}" font-family="{MONO_A}" font-size="10" fill="#6f86a0">AUXILIARES · fuera del tren</text>')
        for ai, a in enumerate(aux):
            mine = sorted([p for p in aux_probes if auxn[str(p["n"]).upper()] == ai],
                          key=lambda p: (_axis(str(p["n"])), str(p["n"])))
            bw = max(1, len(mine)) * (CW + GAP) + GAP
            wst = "ok"
            for p in mine:
                if SEVR.get(p.get("sev") or "off", 0) > SEVR[wst]:
                    wst = p["sev"]
            if ax + bw > vbw - 6:
                ax, ay = 6.0, ay + abh + 14
            out.append(f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{bw}" height="{abh}" rx="10" fill="#0f1c2b" '
                       f'stroke="{SEVC.get(wst, "#27425e")}" stroke-width="1.2" stroke-dasharray="5 4"/>')
            out.append(f'<text x="{ax+10:.1f}" y="{ay+17:.1f}" font-family="{SANS_A}" font-size="12" font-weight="bold" fill="#e8f0f8">{_esc(a.get("name"))}</text>')
            for pi, p in enumerate(mine):
                chip(p, ax + GAP + pi * (CW + GAP), ay + 26)
            ax += bw + 16
        h_total = ay + abh + 10

    vb_y, vb_h = -extra_top, h_total + extra_top
    bg = (f'<linearGradient id="wmStage" x1="0" y1="0" x2="0" y2="1">'
          f'<stop offset="0" stop-color="#1a2e45"/><stop offset="0.72" stop-color="#0c1726"/></linearGradient>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="0 {vb_y} {vbw} {vb_h}" width="{vbw}" height="{vb_h}">'
            f'<defs>{defs}{bg}</defs>'
            f'<rect x="0" y="{vb_y}" width="{vbw}" height="{vb_h}" rx="12" fill="url(#wmStage)"/>'
            f'<g transform="translate(0,{TOP})">{"".join(g)}</g>'
            f'{"".join(out)}</svg>')


def train_drawing_for(instance_id: str, instance_obj: Any, latest: List[Dict[str, Any]],
                      sensor_lookup: Dict[str, Any], target_width_pt: float = 500.0):
    """Drawing (reportlab) del tren para el PDF: el de Machines en vivo si existe;
    si no, el dibujo anterior. None si ninguno se pudo."""
    from core.train_svg import build_train_svg, svg_to_train_drawing
    try:
        from core.live_train import live_payload, machines_train_for
        from core.live_readings import _get_supabase_client
        mt = machines_train_for(instance_id, client=_get_supabase_client())
        if mt:
            setup, layout = mt
            payload = live_payload(setup, latest or [])
            apply_live_severity(payload, latest or [], sensor_lookup, instance_obj, setup)
            d = svg_to_train_drawing(build_live_train_svg(layout, payload), target_width_pt)
            if d is not None:
                return d
    except Exception:  # noqa: BLE001
        pass
    try:
        svg = build_train_svg(instance_obj, latest, sensor_lookup)
        return svg_to_train_drawing(svg, target_width_pt) if svg else None
    except Exception:  # noqa: BLE001
        return None


def apply_live_severity(payload: Dict[str, Any], latest: List[Dict[str, Any]],
                        sensor_lookup: Dict[str, Any], instance_obj: Any, setup) -> None:
    """Severidad de cada sonda = la de Live Monitoring (compute_severity con los
    límites del activo) → PDF, barra y tren cuentan lo mismo."""
    try:
        from core.severity import compute_severity
    except Exception:  # noqa: BLE001
        return
    units = {(r.get("sensor_label") or ""): r.get("unit") or "" for r in latest
             if r.get("metric") == "Direct"}
    lbl = {c.point_label.replace("_", ""): c.point_label for c in setup.channels}
    mp = {"Danger": "danger", "Alarma": "alert", "Normal": "ok"}
    for p in payload.get("probes") or []:
        full = lbl.get(p.get("n"))
        if not full or p.get("v") is None:
            continue
        sv = compute_severity(p["v"], (sensor_lookup or {}).get(full), units.get(full, p.get("u")), instance_obj)
        if sv.get("status") in mp:
            p["sev"] = mp[sv["status"]]
            p["alarm"] = sv.get("alarm") or p.get("alarm")
            p["danger"] = sv.get("danger") or p.get("danger")
