"""
Watermelon Modal — native modal-test module (EMA + OMA, Windows)
================================================================

Field app (PySide6 + pyqtgraph), sibling of Watermelon Rotordynamics. Analyst UI
in ENGLISH (reports stay Spanish). Configuration flow:

  Configuration (inner tabs)
      · Machine            — machine & client data + build the 3D figure (solid,
                             mouse-orbit) + select EMA / OMA / both.
      · Sensors            — place sensors ON the drawing (norm-based or by hand;
                             drag to move) — nothing appears until you add it.
      · Measurement points — table of every placed sensor (number, axis, A/V/D…).
      · Acquisition        — fs / block / Fmax / windows + "Recommended per norm".
      · Summary            — full config table + Save local / Save cloud.
  Impact test (EMA) · OMA capture · Modes · Comparative (EMA vs OMA) · Campbell.

3D is a pure-QPainter isometric solid (no OpenGL → robust in the .exe). Compute
in core.modal.* (tested).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import List, Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except Exception as exc:  # noqa: BLE001
    print("Missing PySide6/pyqtgraph:", exc)
    raise

from core.modal.live_impact import (FRFAccumulator, HitQuality, SynthMode, assess_hit,
                                     modes_from_frf, synth_impact)
from core.modal.oma_layout import (OMALayout, MeasPoint, MachineComponent, default_components,
                                    auto_place_by_norm, recommended_acquisition, component_default_box,
                                    is_pipe, COMPONENT_KINDS, POSITION_REFS, DOFS, MEAS_TYPES,
                                    save_layout_local, list_layouts_local, load_layout_local,
                                    motor_multistage_pump_layout)

# Presets "de fábrica" (nombre visible → función que arma el OMALayout)
FACTORY_PRESETS = {
    "Motor + multistage pump (on pedestals) — 17 sensors": motor_multistage_pump_layout,
}
from core.modal.oma_engine import run_oma
from core.modal.campbell import compute_crossings, SpeedBand

__version__ = "0.9.19"

# Nombre PÚBLICO del sistema de adquisición. Nunca exponer marca/modelo del
# hardware en la interfaz: el cliente solo debe ver "Watermelon".
DAQ_NAME = "Watermelon DAQ"

NAVY = "#0F1E3D"; ACC = "#1AAEE5"; GREEN = "#10b981"; AMBER = "#f59e0b"; RED = "#ef4444"


DEMO_MODES = [SynthMode(19.4, 0.020, 1.0), SynthMode(38.8, 0.015, 0.7),
              SynthMode(77.4, 0.012, 0.45), SynthMode(129.9, 0.010, 0.30)]


# Dirección física de cada eje en el dibujo 3D. X = a lo largo del tren (axial),
# Y = horizontal transversal, Z = vertical. Se aceptan letras de instrumento
# H (horizontal→Y), V (vertical→Z), A (axial→X) además de X/Y/Z.
_AXIS_DIR = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1),
             "A": (1, 0, 0), "H": (0, 1, 0), "V": (0, 0, 1)}


def _comp_color(kind: str) -> str:
    k = (kind or "").lower()
    if is_pipe(kind):
        return "#0891b2" if "suction" in k else "#7c3aed"
    if "motor" in k or "engine" in k:
        return "#2563eb"
    if "turbine" in k:
        return "#0ea5e9"
    if "compressor" in k:
        return "#0d9488"
    if "pump" in k or "bomba" in k:
        return "#16a34a"
    if "generator" in k:
        return "#7c3aed"
    if "gearbox" in k:
        return "#b45309"
    if "coupling" in k:
        return "#334155"
    if "leg" in k or "pedestal" in k or "pata" in k:
        return "#475569"
    if "skid" in k:
        return "#a16207"
    if "fan" in k or "blower" in k:
        return "#db2777"
    return "#64748b"


# ---------- isometric projection (3D → 2D) without OpenGL ----------
def _project(wx, wy, wz, az, el):
    ca, sa = np.cos(az), np.sin(az); ce, se = np.cos(el), np.sin(el)
    x1 = wx * ca - wy * sa
    y1 = wx * sa + wy * ca
    return x1, y1 * se + wz * ce, y1 * ce - wz * se


def _unproject(sx, sy, az, el):
    ca, sa = np.cos(az), np.sin(az); ce, se = np.cos(el), np.sin(el)
    wx = sx / ca if abs(ca) > 1e-6 else sx
    wz = (sy - wx * sa * se) / ce if abs(ce) > 1e-6 else sy
    return wx, wz


def _cuboid_faces(c):
    x0, x1, z0, z1, d = c.x0, c.x1, c.y0, c.y1, c.depth
    P = [(x0, -d, z0), (x1, -d, z0), (x1, d, z0), (x0, d, z0),
         (x0, -d, z1), (x1, -d, z1), (x1, d, z1), (x0, d, z1)]
    F = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]
    return [[P[i] for i in f] for f in F]


def _subdivide_quad(q, n):
    """Subdivide un quad (4 esquinas) en n×n sub-quads (bilineal) → malla fina."""
    p0, p1, p2, p3 = [np.array(p, float) for p in q]
    def bilin(u, v):
        return (1 - u) * (1 - v) * p0 + u * (1 - v) * p1 + u * v * p2 + (1 - u) * v * p3
    subs = []
    for i in range(n):
        for j in range(n):
            u0, u1, v0, v1 = i / n, (i + 1) / n, j / n, (j + 1) / n
            subs.append([bilin(u0, v0), bilin(u1, v0), bilin(u1, v1), bilin(u0, v1)])
    return subs


def _field_disp(v, anim):
    """IDW en el vértice v: posición (desde amps instantáneos) + magnitud de COLOR
    (desde mags = amplitud/envolvente por sensor, para colorear como ARTeMIS)."""
    pts = anim["pts"]; dirs = anim["dirs"]; amps = anim["amps"]; mags = anim.get("mags")
    if len(pts) == 0:
        return np.zeros(3), 0.0
    diff = pts - v[None, :]
    d2 = np.sum(diff * diff, axis=1) + 1e-4
    w = 1.0 / d2; wsum = w.sum()
    disp = (w[:, None] * (amps[:, None] * dirs)).sum(axis=0) / wsum
    colmag = float((w * mags).sum() / wsum) if mags is not None else float(np.linalg.norm(disp))
    return disp, colmag


def _heat_qcolor(t):
    """Informative colormap: GREEN (low vibration) -> amber -> RED (high vibration)."""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (22, 163, 74)), (0.35, (132, 204, 22)), (0.6, (245, 158, 11)),
             (0.8, (249, 115, 22)), (1.0, (220, 38, 38))]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]; t1, c1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0 or 1)
            r = int(c0[0] + f * (c1[0] - c0[0])); g = int(c0[1] + f * (c1[1] - c0[1])); b = int(c0[2] + f * (c1[2] - c0[2]))
            return QtGui.QColor(r, g, b)
    return QtGui.QColor(178, 24, 43)


class Machine3DItem(pg.GraphicsObject):
    """Draws equipment as shaded 3D solids + sensors + DOF arrows."""
    def __init__(self):
        super().__init__()
        self.layout = None; self.az = 0.9; self.el = 0.5; self.sel = -1
        self.disp = None          # desplazamiento animado por punto activo (a lo largo del DOF)
        self.anim = None          # {pts(Nx3), dirs(Nx3), amps(N), mags(N), mmax} → deforma+colorea
        self.show_sensors = True  # mostrar/ocultar marcadores de sensores
        self._rect = QtCore.QRectF(-1, -1, 2, 2)

    def set_view(self, layout, az, el, sel):
        self.layout = layout; self.az = az; self.el = el; self.sel = sel
        self.prepareGeometryChange(); self._recompute(); self.update()

    def set_disp(self, disp):
        self.disp = disp; self.update()

    def set_anim(self, anim):
        self.anim = anim; self.update()

    def set_show_sensors(self, on):
        self.show_sensors = bool(on); self.update()

    def _recompute(self):
        xs, ys = [0.0], [0.0]
        if self.layout:
            for c in self.layout.machine_components:
                for f in _cuboid_faces(c):
                    for (wx, wy, wz) in f:
                        sx, sy, _ = _project(wx, wy, wz, self.az, self.el)
                        xs.append(sx); ys.append(sy)
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
        self._rect = QtCore.QRectF(x0, y0, max(0.1, x1 - x0), max(0.1, y1 - y0))

    def boundingRect(self):
        return self._rect

    def paint(self, painter, *args):
        if not self.layout:
            return
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        light = np.array([0.4, -0.7, 0.6]); light /= np.linalg.norm(light)
        anim = self.anim
        faces = []                                          # (dep, pw, base|None, shade, colmag)
        for c in self.layout.machine_components:
            base = QtGui.QColor(c.color) if getattr(c, "color", "") else QtGui.QColor(_comp_color(c.kind))
            for f in _cuboid_faces(c):
                # en animación cada cara se subdivide en una malla fina (degradé suave)
                quads = _subdivide_quad(f, 5) if anim is not None else [f]
                for q in quads:
                    colmag = None
                    if anim is not None:
                        dv = [_field_disp(np.array(p, float), anim) for p in q]
                        fd = [tuple(np.array(p, float) + dv[k][0]) for k, p in enumerate(q)]
                        pw = [_project(*p, self.az, self.el) for p in fd]
                        colmag = float(np.mean([dv[k][1] for k in range(len(q))]))
                    else:
                        pw = [_project(*p, self.az, self.el) for p in q]
                    dep = float(np.mean([p[2] for p in pw]))
                    v1 = np.subtract(q[1], q[0]); v2 = np.subtract(q[2], q[0])
                    nrm = np.cross(v1, v2); nn = np.linalg.norm(nrm)
                    shade = 0.6 + 0.4 * abs(float(np.dot(nrm / nn, light))) if nn > 0 else 0.75
                    faces.append([dep, pw, base, shade, colmag])
        # normaliza el color sobre TODA la máquina → verde (menos) → rojo (más)
        if anim is not None:
            mags = [fc[4] for fc in faces if fc[4] is not None]
            cmin = min(mags) if mags else 0.0; cmax = max(mags) if mags else 1.0
            rng = (cmax - cmin) or 1.0
            for fc in faces:
                if fc[4] is not None:
                    fc[2] = _heat_qcolor((fc[4] - cmin) / rng)
        faces.sort(key=lambda t: t[0])
        for dep, pw, base, shade, _cm in faces:
            col = QtGui.QColor(int(base.red() * shade), int(base.green() * shade), int(base.blue() * shade))
            poly = QtGui.QPolygonF([QtCore.QPointF(p[0], p[1]) for p in pw])
            painter.setBrush(QtGui.QBrush(col))
            if anim is not None:                             # malla fina → sin líneas de grilla
                pen = QtGui.QPen(col); pen.setCosmetic(True); pen.setWidthF(0.3)
            else:
                pen = QtGui.QPen(QtGui.QColor("#1e293b")); pen.setCosmetic(True); pen.setWidthF(0.8)
            painter.setPen(pen); painter.drawPolygon(poly)
        ai = -1
        for i, mp in enumerate(self.layout.points):
            if not mp.active:
                continue
            ai += 1
            if not self.show_sensors:                    # animación: solo la pieza en movimiento
                continue
            wy = 0.20
            d = _AXIS_DIR.get(mp.axis, (0, 0, 1))
            sg = -1.0 if mp.dof.startswith("-") else 1.0
            # desplazamiento animado de la forma modal (a lo largo del DOF)
            dd = 0.0
            if self.disp is not None and ai < len(self.disp):
                dd = float(self.disp[ai])
            px = mp.x_norm + sg * dd * d[0]; py = wy + sg * dd * d[1]; pz = mp.y_norm + sg * dd * d[2]
            sx, sy, _ = _project(px, py, pz, self.az, self.el)
            tx, ty, _ = _project(px + sg * 0.09 * d[0], py + sg * 0.09 * d[1],
                                 pz + sg * 0.09 * d[2], self.az, self.el)
            arrow = QtGui.QPen(QtGui.QColor(GREEN)); arrow.setCosmetic(True); arrow.setWidthF(2.2)
            painter.setPen(arrow); painter.drawLine(QtCore.QPointF(sx, sy), QtCore.QPointF(tx, ty))
            col = QtGui.QColor(_comp_color(mp.component))
            r = 0.018 if i == self.sel else (0.014 if mp.reference_sensor else 0.010)
            painter.setBrush(QtGui.QBrush(col))
            pen = QtGui.QPen(QtGui.QColor(RED if i == self.sel else "#ffffff"))
            pen.setCosmetic(True); pen.setWidthF(2.0 if i == self.sel else 1.2); painter.setPen(pen)
            painter.drawEllipse(QtCore.QPointF(sx, sy), r, r)


class OrbitViewBox(pg.ViewBox):
    """3D viewbox: left-drag = orbit (or drag a grabbed sensor); wheel = zoom; right-drag = pan."""
    rotate = QtCore.Signal(float, float)
    grab = QtCore.Signal(object, object)          # scenePos, self  (drag start)
    drag = QtCore.Signal(object, object, float, float)   # scenePos, self, dx, dy
    release = QtCore.Signal()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                self.grab.emit(ev.scenePos(), self)
            d = ev.pos() - ev.lastPos()
            self.drag.emit(ev.scenePos(), self, float(d.x()), float(d.y()))
            if ev.isFinish():
                self.release.emit()
            ev.accept()
        else:
            super().mouseDragEvent(ev, axis)


def _stylesheet() -> str:
    return f"""
    QWidget {{ font-family: 'Segoe UI', Arial; font-size: 12px; color: {NAVY}; }}
    QMainWindow, QTabWidget::pane {{ background: #f5f8fd; }}
    QTabWidget::pane {{ border: 1px solid #dbe4f0; border-radius: 10px; top: -1px; }}
    QTabBar::tab {{ background: #e6ecf5; color: #334155; padding: 8px 16px; margin-right: 3px;
        border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 700; }}
    QTabBar::tab:hover {{ background: #d4deee; }}
    QTabBar::tab:selected {{ background: {NAVY}; color: white; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: white;
        border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 8px; selection-background-color: {ACC}; }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACC}; }}
    QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QPushButton {{ background: {NAVY}; color: white; border: none; font-weight: 700;
        padding: 8px 15px; border-radius: 8px; }}
    QPushButton:hover {{ background: #0e3a6b; }}
    QPushButton:pressed {{ background: #0b2e56; }}
    QPushButton:disabled {{ background: #94a3b8; }}
    QCheckBox {{ spacing: 6px; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid #94a3b8; border-radius: 4px; background: white; }}
    QCheckBox::indicator:checked {{ background: {GREEN}; border: 1px solid {GREEN}; }}
    QTableWidget {{ background: white; gridline-color: #eef2f8; border: 1px solid #e2e8f0;
        border-radius: 8px; alternate-background-color: #fafcff; }}
    QTableWidget::item:selected {{ background: #d9ebfb; color: {NAVY}; }}
    QHeaderView::section {{ background: {NAVY}; color: white; padding: 6px; border: none; font-weight: 700; }}
    QTextBrowser {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 6px; }}
    QToolTip {{ background: {NAVY}; color: white; border: none; padding: 6px 8px; border-radius: 6px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: #c3cfe0; border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: #9fb2cd; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: #c3cfe0; border-radius: 5px; min-width: 30px; }}
    """


def _led(color, on, label):
    c = color if on else "#cbd5e1"
    return (f"<span style='color:{c};font-size:16px'>●</span> "
            f"<span style='font-weight:700;color:{'#334155' if on else '#94a3b8'}'>{label}</span>")


def build_app(layout: OMALayout, simulated: bool = True):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(_stylesheet())
    pg.setConfigOptions(antialias=True)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Modal v{__version__} — {layout.name}")
    # Auto-ajuste a la pantalla: nunca más grande que el área disponible (deja
    # visible la barra de título con minimizar/cerrar) y queda centrada.
    _scr = app.primaryScreen()
    _av = _scr.availableGeometry() if _scr else None
    if _av is not None:
        _w = min(1320, _av.width() - 40); _h = min(850, _av.height() - 80)
        win.resize(max(900, _w), max(560, _h))
        win.setMinimumSize(820, 520)
        win.move(_av.left() + (_av.width() - win.width()) // 2,
                 _av.top() + (_av.height() - win.height()) // 2)
    else:
        win.resize(1320, 850)

    st = {"layout": layout, "acc": FRFAccumulator(layout.fs_hz, layout.block_size),
          "pending": None, "target": 5, "oma_fdd": None,
          "az": 50.0, "el": 28.0, "grab": None, "rng": np.random.default_rng(), "_views": []}

    tb = win.addToolBar("main"); tb.setMovable(False)
    tb.setStyleSheet(f"QToolBar {{ background: {NAVY}; padding: 6px 12px; }}")
    brand = QtWidgets.QLabel("  🍉 Watermelon Modal")
    brand.setStyleSheet("color:white; font-weight:800; font-size:15px;"); tb.addWidget(brand)
    ver_lbl = QtWidgets.QLabel(f"v{__version__}")
    ver_lbl.setStyleSheet("color:#94a3b8; font-weight:700; font-size:12px; padding-left:8px;")
    ver_lbl.setToolTip("Watermelon Modal software version"); tb.addWidget(ver_lbl)
    spc = QtWidgets.QWidget(); spc.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    tb.addWidget(spc)
    # ¿Hay una NI 9234 conectada AHORA? Autodetecta al arrancar (no depende de --sim).
    def _detect_ni_channels():
        try:
            from core.modal.acq_backend import list_available_devices
            n = 0
            for d in list_available_devices():
                if "9234" in str(d.get("product_type", "")):
                    n += 4                                   # cada 9234 = 4 canales IEPE
            return n
        except Exception:  # noqa: BLE001  (sin driver / sin hardware)
            return 0
    ni_channels = _detect_ni_channels()
    hw_present = ni_channels > 0
    if hw_present:
        mode_lbl = QtWidgets.QLabel(f"● LIVE — {DAQ_NAME} · {ni_channels} channels   ")
        mode_lbl.setStyleSheet("color:#34d399; font-weight:700;")
    else:
        mode_lbl = QtWidgets.QLabel("● SIMULATED — no acquisition connected   ")
        mode_lbl.setStyleSheet("color:#fbbf24; font-weight:700;")
    mode_lbl.setToolTip("Se detecta al abrir el programa. Cada captura usa la fuente "
                        f"elegida en 'Source' (Simulado / {DAQ_NAME}).")
    tb.addWidget(mode_lbl)

    tabs = QtWidgets.QTabWidget(); win.setCentralWidget(tabs)

    # ---- forward decls used by _make_view (resolved at signal time) ----
    def _make_view(kind):
        plot = pg.PlotWidget(viewBox=OrbitViewBox()); plot.setBackground("w"); plot.setAspectLocked(True)
        plot.hideAxis("left"); plot.hideAxis("bottom"); plot.setMenuEnabled(False)
        m3 = Machine3DItem(); plot.addItem(m3)
        v = {"plot": plot, "m3d": m3, "labels": [], "kind": kind}; st["_views"].append(v)
        vb = plot.getViewBox()
        vb.grab.connect(lambda sp, _vb, k=kind, p=plot: _on_grab(sp, p, k))
        vb.drag.connect(lambda sp, _vb, dx, dy, k=kind, p=plot: _on_drag(sp, p, k, dx, dy))
        vb.release.connect(lambda: st.update(grab=None, grab_comp=None))
        plot.scene().sigMouseClicked.connect(lambda ev, k=kind, p=plot: _on_click(ev, p, k))
        return v

    # =================================================================
    # CONFIGURATION
    # =================================================================
    cfg_outer = QtWidgets.QWidget(); cfg_ol = QtWidgets.QVBoxLayout(cfg_outer)
    cfg_tabs = QtWidgets.QTabWidget(); cfg_ol.addWidget(cfg_tabs, 1)

    # ---------- Machine ----------
    pg_m = QtWidgets.QWidget(); ml = QtWidgets.QVBoxLayout(pg_m)
    ml.addWidget(QtWidgets.QLabel("<b>Machine &amp; client</b>"))
    frm = QtWidgets.QFormLayout()
    e_name = QtWidgets.QLineEdit(layout.name)
    e_type = QtWidgets.QLineEdit(layout.machine_type); e_type.setPlaceholderText("Motor-pump, turbine train…")
    e_tag = QtWidgets.QLineEdit(layout.tag); e_tag.setPlaceholderText("asset tag / nameplate")
    e_client = QtWidgets.QLineEdit(layout.client); e_client.setPlaceholderText("client")
    e_loc = QtWidgets.QLineEdit(layout.location); e_loc.setPlaceholderText("plant / location")
    sp_rpm = QtWidgets.QDoubleSpinBox(); sp_rpm.setRange(0, 60000); sp_rpm.setValue(layout.running_speed_rpm)
    r1 = QtWidgets.QHBoxLayout(); r1.addWidget(e_name, 2); r1.addSpacing(8)
    r1.addWidget(QtWidgets.QLabel("Type:")); r1.addWidget(e_type, 2)
    _w1 = QtWidgets.QWidget(); _w1.setLayout(r1); frm.addRow("Machine:", _w1)
    r2 = QtWidgets.QHBoxLayout(); r2.addWidget(e_tag, 1); r2.addSpacing(8)
    r2.addWidget(QtWidgets.QLabel("Client:")); r2.addWidget(e_client, 1); r2.addSpacing(8)
    r2.addWidget(QtWidgets.QLabel("Location:")); r2.addWidget(e_loc, 1)
    _w2 = QtWidgets.QWidget(); _w2.setLayout(r2); frm.addRow("Tag:", _w2)
    r3 = QtWidgets.QHBoxLayout(); r3.addWidget(QtWidgets.QLabel("Running speed (RPM):")); r3.addWidget(sp_rpm)
    r3.addSpacing(16); r3.addWidget(QtWidgets.QLabel("Test:"))
    chk_ema = QtWidgets.QCheckBox("EMA (impact)"); chk_oma = QtWidgets.QCheckBox("OMA (operational)")
    chk_ema.setChecked("EMA" in layout.test_modes); chk_oma.setChecked("OMA" in layout.test_modes or not layout.test_modes)
    r3.addWidget(chk_ema); r3.addWidget(chk_oma); r3.addStretch(1)
    _w3 = QtWidgets.QWidget(); _w3.setLayout(r3); frm.addRow("Operation:", _w3)
    ml.addLayout(frm)

    # --- Fila 1: agregar/quitar equipo + nombre editable ---
    row_add = QtWidgets.QHBoxLayout()
    row_add.addWidget(QtWidgets.QLabel("<b>Equipment</b>"))
    cb_kind = QtWidgets.QComboBox(); cb_kind.addItems(COMPONENT_KINDS); cb_kind.setMinimumWidth(190)
    btn_addcomp = QtWidgets.QPushButton("➕ Add"); btn_delcomp = QtWidgets.QPushButton("– Remove")
    row_add.addWidget(cb_kind); row_add.addWidget(btn_addcomp); row_add.addWidget(btn_delcomp)
    row_add.addSpacing(24)
    row_add.addWidget(QtWidgets.QLabel("<b>Name</b>"))
    e_compname = QtWidgets.QLineEdit(); e_compname.setMinimumWidth(160)
    e_compname.setPlaceholderText("e.g. ABB motor")
    e_compname.setToolTip("Name shown above the equipment in the drawing (editable).")
    row_add.addWidget(e_compname)
    row_add.addStretch(1)
    ml.addLayout(row_add)

    # --- Fila 2: seleccionar equipo + dimensiones + color + rotar ---
    bld = QtWidgets.QHBoxLayout()
    bld.addWidget(QtWidgets.QLabel("<b>Selected</b>"))
    cb_comp = QtWidgets.QComboBox(); cb_comp.setMinimumWidth(160); bld.addWidget(cb_comp)
    bld.addSpacing(16); bld.addWidget(QtWidgets.QLabel("<b>Size</b>"))
    sp_len = QtWidgets.QDoubleSpinBox(); sp_len.setRange(0.02, 1.0); sp_len.setSingleStep(0.02); sp_len.setDecimals(2)
    sp_hei = QtWidgets.QDoubleSpinBox(); sp_hei.setRange(0.02, 0.8); sp_hei.setSingleStep(0.02); sp_hei.setDecimals(2)
    sp_wid = QtWidgets.QDoubleSpinBox(); sp_wid.setRange(0.02, 0.6); sp_wid.setSingleStep(0.02); sp_wid.setDecimals(2)
    for lab, w in (("L", sp_len), ("H", sp_hei), ("W", sp_wid)):
        bld.addWidget(QtWidgets.QLabel(lab)); bld.addWidget(w)
    bld.addSpacing(16)
    btn_color = QtWidgets.QPushButton("🎨 Colour")
    btn_color.setToolTip("Pick the colour of the selected equipment (e.g. grey motor, green pump).")
    btn_colreset = QtWidgets.QPushButton("↺")
    btn_colreset.setToolTip("Reset to the default colour for the equipment type.")
    btn_colreset.setMaximumWidth(34)
    bld.addWidget(btn_color); bld.addWidget(btn_colreset)
    bld.addSpacing(16)
    chk_lock = QtWidgets.QCheckBox("🔒 Rotate whole assembly")
    chk_lock.setToolTip("ON: drag rotates/positions the WHOLE assembly (X/Y/Z). "
                        "OFF: drag moves each equipment separately.")
    bld.addWidget(chk_lock)
    bld.addStretch(1)
    ml.addLayout(bld)
    ml.addWidget(QtWidgets.QLabel(
        "<i style='color:#64748b'>🖱️ left-drag = rotate · wheel = zoom · right-drag = pan · "
        "click = move the selected equipment. Build the figure, then go to <b>Sensors</b>.</i>"))
    vgeo = _make_view("geo"); ml.addWidget(vgeo["plot"], 1)
    cfg_tabs.addTab(pg_m, "Machine")

    # ---------- Sensors ----------
    pg_sen = QtWidgets.QWidget(); snl = QtWidgets.QVBoxLayout(pg_sen)
    snl.addWidget(QtWidgets.QLabel(
        "<b>Place the sensors on the machine.</b> Set number, axes and quantity, then "
        "<b>click on the drawing</b> — or use <b>Place by standard</b>."))
    prow = QtWidgets.QHBoxLayout()
    prow.addWidget(QtWidgets.QLabel("No.:"))
    sp_num = QtWidgets.QSpinBox(); sp_num.setRange(1, 999); sp_num.setValue(1); prow.addWidget(sp_num)
    prow.addWidget(QtWidgets.QLabel("Axes:"))
    cbx = QtWidgets.QCheckBox("X"); cby = QtWidgets.QCheckBox("Y"); cbz = QtWidgets.QCheckBox("Z"); cby.setChecked(True)
    for w in (cbx, cby, cbz): prow.addWidget(w)
    prow.addWidget(QtWidgets.QLabel("Measures:"))
    cb_mtype = QtWidgets.QComboBox(); cb_mtype.addItems(MEAS_TYPES); prow.addWidget(cb_mtype)
    prow.addSpacing(12); prow.addWidget(QtWidgets.QLabel("Click mode:"))
    cb_click = QtWidgets.QComboBox(); cb_click.addItems(["Place point", "Move sensor"]); prow.addWidget(cb_click)
    prow.addWidget(QtWidgets.QLabel("Sensor:"))
    cb_place = QtWidgets.QComboBox(); cb_place.setMinimumWidth(150); prow.addWidget(cb_place)
    prow.addStretch(1)
    snl.addLayout(prow)
    nrow = QtWidgets.QHBoxLayout()
    btn_norm = QtWidgets.QPushButton("📐 Place by standard (API 670 / ISO 20816)")
    btn_norm.setStyleSheet(f"QPushButton{{background:{GREEN};}} QPushButton:hover{{background:#0e9f6e;}}")
    btn_clrpts = QtWidgets.QPushButton("🧹 Clear sensors")
    nrow.addWidget(btn_norm); nrow.addWidget(btn_clrpts)
    nrow.addWidget(QtWidgets.QLabel("<i style='color:#64748b'>· In 'Move sensor' mode, click-and-hold on a "
                                    "sensor to drag it.</i>")); nrow.addStretch(1)
    snl.addLayout(nrow)
    vsen = _make_view("sensors"); snl.addWidget(vsen["plot"], 1)
    cfg_tabs.addTab(pg_sen, "Sensors")

    # ---------- Measurement points (table) ----------
    pg_pts = QtWidgets.QWidget(); pl = QtWidgets.QVBoxLayout(pg_pts)
    pl.addWidget(QtWidgets.QLabel("<b>Measurement points</b> — every placed sensor."))
    tbl_pts = QtWidgets.QTableWidget(0, 11)
    tbl_pts.setHorizontalHeaderLabels(["#", "No.", "Component", "Reference", "DOF", "Meas.",
                                       "Slot", "Ch", "Sens (mV/g)", "Ref", "Active"])
    tbl_pts.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tbl_pts.verticalHeader().setVisible(False)
    pl.addWidget(tbl_pts, 1)
    r_pt = QtWidgets.QHBoxLayout()
    btn_delp = QtWidgets.QPushButton("– Remove selected")
    r_pt.addWidget(btn_delp); r_pt.addStretch(1)
    pl.addLayout(r_pt)
    lbl_val = QtWidgets.QLabel(""); pl.addWidget(lbl_val)
    cfg_tabs.addTab(pg_pts, "Measurement points")

    # ---------- Acquisition ----------
    pg_acq = QtWidgets.QWidget(); aol = QtWidgets.QVBoxLayout(pg_acq)
    al = QtWidgets.QFormLayout()
    sp_fs = QtWidgets.QSpinBox(); sp_fs.setRange(256, 51200); sp_fs.setValue(int(layout.fs_hz)); sp_fs.setSingleStep(256)
    cb_blk = QtWidgets.QComboBox(); cb_blk.addItems(["1024", "2048", "4096", "8192", "16384"]); cb_blk.setCurrentText(str(layout.block_size))
    sp_fmax = QtWidgets.QDoubleSpinBox(); sp_fmax.setRange(10, 25600); sp_fmax.setValue(layout.fmax_hz)
    sp_dur = QtWidgets.QSpinBox(); sp_dur.setRange(5, 3600); sp_dur.setValue(int(layout.duration_s))
    chk_fwin = QtWidgets.QCheckBox("Force window (EMA)"); chk_fwin.setChecked(True)
    chk_ewin = QtWidgets.QCheckBox("Exponential window (EMA)"); chk_ewin.setChecked(True)
    sp_tgt = QtWidgets.QSpinBox(); sp_tgt.setRange(1, 64); sp_tgt.setValue(5)
    al.addRow("Sampling fs (Hz):", sp_fs)
    al.addRow("Block size (EMA, samples/hit):", cb_blk)
    al.addRow("Fmax (Hz):", sp_fmax)
    al.addRow("Duration (OMA, s):", sp_dur)
    al.addRow("Target averages (EMA):", sp_tgt)
    _wb = QtWidgets.QWidget(); _wbl = QtWidgets.QHBoxLayout(_wb); _wbl.setContentsMargins(0, 0, 0, 0)
    _wbl.addWidget(chk_fwin); _wbl.addWidget(chk_ewin); _wbl.addStretch(1)
    al.addRow("Windows:", _wb)
    lbl_df = QtWidgets.QLabel(""); al.addRow("Resolution:", lbl_df)
    aol.addLayout(al)
    rrow = QtWidgets.QHBoxLayout()
    btn_reco = QtWidgets.QPushButton("📏 Recommended (per standard)")
    btn_reco.setStyleSheet(f"QPushButton{{background:{GREEN};}} QPushButton:hover{{background:#0e9f6e;}}")
    rrow.addWidget(btn_reco); rrow.addStretch(1)
    aol.addLayout(rrow)
    lbl_reco = QtWidgets.QLabel(""); lbl_reco.setWordWrap(True); lbl_reco.setStyleSheet("color:#334155;")
    aol.addWidget(lbl_reco); aol.addStretch(1)
    cfg_tabs.addTab(pg_acq, "Acquisition")

    # ---------- Summary ----------
    pg_sum = QtWidgets.QWidget(); sul = QtWidgets.QVBoxLayout(pg_sum)
    lbl_sum = QtWidgets.QLabel(""); lbl_sum.setStyleSheet(f"background:{NAVY};color:white;border-radius:8px;padding:10px 14px;font-weight:700;")
    sul.addWidget(lbl_sum)
    tbl_sum = QtWidgets.QTableWidget(0, 7)
    tbl_sum.setHorizontalHeaderLabels(["No.", "Code", "Component", "Reference", "DOF", "Meas.", "BNC"])
    tbl_sum.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); tbl_sum.verticalHeader().setVisible(False)
    sul.addWidget(tbl_sum, 1)
    save_row = QtWidgets.QHBoxLayout()
    btn_preset = QtWidgets.QPushButton("⭐ Presets")
    btn_preset.setToolTip("Load a ready factory configuration (machine + sensors + acquisition).")
    btn_savelocal = QtWidgets.QPushButton("💾 Save locally")
    btn_loadlocal = QtWidgets.QPushButton("📂 Load local")
    btn_savecloud = QtWidgets.QPushButton("☁ Save to Watermelon System")
    btn_savecloud.setStyleSheet(f"QPushButton{{background:{GREEN};}} QPushButton:hover{{background:#0e9f6e;}}")
    btn_loadcloud = QtWidgets.QPushButton("☁ Load from cloud")
    for b in (btn_preset, btn_savelocal, btn_loadlocal, btn_savecloud, btn_loadcloud):
        save_row.addWidget(b)
    save_row.addStretch(1)
    sul.addLayout(save_row)
    cfg_tabs.addTab(pg_sum, "Summary")

    # Apply & auto-arrange bar (shared, present under every config tab)
    apply_bar = QtWidgets.QHBoxLayout()
    btn_apply = QtWidgets.QPushButton("✓ Apply changes & auto-arrange")
    btn_apply.setStyleSheet(f"QPushButton{{background:{ACC};font-size:13px;padding:9px 18px;}} QPushButton:hover{{background:#1490c2;}}")
    lbl_applyinfo = QtWidgets.QLabel("")
    apply_bar.addWidget(btn_apply); apply_bar.addWidget(lbl_applyinfo); apply_bar.addStretch(1)
    cfg_ol.addLayout(apply_bar)
    tabs.addTab(cfg_outer, "Configuration")

    # =====================================================================
    # Config helpers
    # =====================================================================
    def _mk_combo(items, cur):
        c = QtWidgets.QComboBox(); c.addItems(items)
        if cur in items:
            c.setCurrentText(cur)
        return c

    def _add_point_row(mp):
        r = tbl_pts.rowCount(); tbl_pts.insertRow(r)
        tbl_pts.setItem(r, 0, QtWidgets.QTableWidgetItem(str(mp.idx)))
        tbl_pts.setItem(r, 1, QtWidgets.QTableWidgetItem(str(mp.number or mp.idx)))
        tbl_pts.setCellWidget(r, 2, _mk_combo(COMPONENT_KINDS, mp.component))
        tbl_pts.setCellWidget(r, 3, _mk_combo(POSITION_REFS, mp.position_ref))
        tbl_pts.setCellWidget(r, 4, _mk_combo(DOFS, mp.dof))
        tbl_pts.setCellWidget(r, 5, _mk_combo(MEAS_TYPES, mp.meas_type))
        tbl_pts.setItem(r, 6, QtWidgets.QTableWidgetItem(str(mp.module_slot)))
        tbl_pts.setItem(r, 7, QtWidgets.QTableWidgetItem(str(mp.channel_index)))
        tbl_pts.setItem(r, 8, QtWidgets.QTableWidgetItem(f"{mp.sensitivity_mv_per_g:g}"))
        cbref = QtWidgets.QCheckBox(); cbref.setChecked(mp.reference_sensor)
        cbact = QtWidgets.QCheckBox(); cbact.setChecked(mp.active)
        tbl_pts.setCellWidget(r, 9, cbref); tbl_pts.setCellWidget(r, 10, cbact)

    def _fill_points():
        tbl_pts.setRowCount(0)
        for mp in st["layout"].points:
            _add_point_row(mp)
        _sync_place_combo()

    def _sync_place_combo():
        cur = cb_place.currentIndex()
        cb_place.blockSignals(True); cb_place.clear()
        cb_place.addItems([f"{p.code}" for p in st["layout"].points])
        if 0 <= cur < cb_place.count():
            cb_place.setCurrentIndex(cur)
        cb_place.blockSignals(False)

    def _table_to_layout():
        lay = st["layout"]; pts = []
        for r in range(tbl_pts.rowCount()):
            def _t(c, d=""):
                it = tbl_pts.item(r, c); return it.text() if it else d
            def _c(c):
                w = tbl_pts.cellWidget(r, c); return w.currentText() if w else ""
            def _k(c):
                w = tbl_pts.cellWidget(r, c); return bool(w.isChecked()) if w else False
            old = lay.points[r] if r < len(lay.points) else None
            try:
                pts.append(MeasPoint(
                    idx=int(_t(0, str(r + 1)) or r + 1), number=int(_t(1, "0") or 0),
                    component=_c(2), position_ref=_c(3), dof=_c(4), meas_type=_c(5) or "A",
                    module_slot=int(_t(6, "1") or 1), channel_index=int(_t(7, "0") or 0),
                    sensitivity_mv_per_g=float(_t(8, "100") or 100), reference_sensor=_k(9), active=_k(10),
                    x_norm=old.x_norm if old else 0.5, y_norm=old.y_norm if old else 0.0))
            except Exception:  # noqa: BLE001
                continue
        if pts or tbl_pts.rowCount() == 0:
            lay.points = pts
        lay.name = e_name.text() or "Modal"; lay.machine_type = e_type.text(); lay.tag = e_tag.text()
        lay.client = e_client.text(); lay.location = e_loc.text(); lay.running_speed_rpm = sp_rpm.value()
        lay.test_modes = [m for m, c in (("EMA", chk_ema), ("OMA", chk_oma)) if c.isChecked()] or ["OMA"]
        lay.test_type = lay.test_modes[0]
        lay.fs_hz = float(sp_fs.value()); lay.block_size = int(cb_blk.currentText())
        lay.fmax_hz = float(sp_fmax.value()); lay.duration_s = float(sp_dur.value())

    def _view_angles():
        return np.radians(st["az"]), np.radians(st["el"])

    def _draw_train(fit=False):
        lay = st["layout"]; az, el = _view_angles(); sel = cb_place.currentIndex()
        for v in st["_views"]:
            plot = v["plot"]; m3 = v["m3d"]; m3.set_view(lay, az, el, sel)
            for t in v["labels"]:
                plot.removeItem(t)
            v["labels"] = []
            for c in lay.machine_components:
                sx, sy, _ = _project((c.x0 + c.x1) / 2, 0.0, (c.y0 + c.y1) / 2, az, el)
                t = pg.TextItem(c.display(), color="#0f172a", anchor=(0.5, 0.5))
                t.setPos(sx, sy); v["labels"].append(t); plot.addItem(t)
            for mp in lay.points:
                if not mp.active:
                    continue
                sx, sy, _ = _project(mp.x_norm, 0.20, mp.y_norm, az, el)
                t = pg.TextItem(mp.code, color=NAVY, anchor=(0.5, 1.6)); t.setScale(0.85)
                t.setPos(sx, sy); v["labels"].append(t); plot.addItem(t)
            vb = plot.getViewBox()
            if fit is True:
                vb.autoRange(padding=0.15)               # encuadra (carga inicial)
            elif fit is False:
                br = m3.boundingRect(); (x0, x1), (y0, y1) = vb.viewRange()
                hw = (x1 - x0) / 2 or 0.7; hh = (y1 - y0) / 2 or 0.5
                cx, cy = br.center().x(), br.center().y()
                vb.setRange(xRange=(cx - hw, cx + hw), yRange=(cy - hh, cy + hh), padding=0)
            # fit is None → NO tocar la vista (queda exactamente como el usuario la dejó)

    def _next_channel():
        used = {(p.module_slot, p.channel_index) for p in st["layout"].points}
        for slot in range(1, 9):
            for ch in range(4):
                if (slot, ch) not in used:
                    return slot, ch
        return 8, 3

    def _comp_at(x, z):
        lay = st["layout"]
        for c in lay.machine_components:
            if c.x0 <= x <= c.x1 and c.y0 <= z <= c.y1:
                return c.kind
        return min(lay.machine_components, key=lambda k: abs((k.x0 + k.x1) / 2 - x)).kind if lay.machine_components else "Electric motor"

    def _nearest_sensor(x, z, tol=0.05):
        best = None; bd = tol
        for i, mp in enumerate(st["layout"].points):
            d = ((mp.x_norm - x) ** 2 + (mp.y_norm - z) ** 2) ** 0.5
            if d < bd:
                bd = d; best = i
        return best

    def _nearest_component(x, z):
        comps = st["layout"].machine_components
        if not comps:
            return None
        best, bd = None, 1e9
        for i, c in enumerate(comps):
            d = abs((c.x0 + c.x1) / 2 - x) + abs((c.y0 + c.y1) / 2 - z)
            if d < bd:
                bd, best = d, i
        return best

    def _on_grab(scenePos, plot, kind):
        st["grab"] = None; st["grab_comp"] = None
        pt = plot.getViewBox().mapSceneToView(scenePos); az, el = _view_angles()
        x, z = _unproject(float(pt.x()), float(pt.y()), az, el)
        if kind == "sensors" and cb_click.currentText() == "Move sensor":
            i = _nearest_sensor(x, z)
            if i is not None:
                st["grab"] = i; cb_place.setCurrentIndex(i)
        elif kind == "geo" and not chk_lock.isChecked():  # click sostenido = arrastrar equipo
            j = _nearest_component(x, z)                   # (si está 🔒, arrastrar GIRA todo)
            if j is not None:
                st["grab_comp"] = j; cb_comp.setCurrentIndex(j)

    def _on_drag(scenePos, plot, kind, dx, dy):
        pt = plot.getViewBox().mapSceneToView(scenePos); az, el = _view_angles()
        x, z = _unproject(float(pt.x()), float(pt.y()), az, el)
        if st.get("grab") is not None:                    # arrastrar el sensor tomado
            mp = st["layout"].points[st["grab"]]; mp.x_norm = x; mp.y_norm = z
            _draw_train(fit=False)
        elif st.get("grab_comp") is not None:             # arrastrar el equipo tomado
            c = st["layout"].machine_components[st["grab_comp"]]
            w = c.x1 - c.x0; h = c.y1 - c.y0
            c.x0 = x - w / 2; c.x1 = x + w / 2; c.y0 = z - h / 2; c.y1 = z + h / 2
            _draw_train(fit=False)
        else:                                             # orbitar
            _on_rotate(dx, dy)

    def _on_rotate(dx, dy):
        st["az"] = (st["az"] + dx * 0.4) % 360.0
        st["el"] = float(np.clip(st["el"] - dy * 0.4, 8.0, 88.0))
        _draw_train(fit=False)

    def _on_click(ev, plot, kind):
        try:
            if ev.button() != QtCore.Qt.LeftButton:
                return
            pt = plot.getViewBox().mapSceneToView(ev.scenePos()); az, el = _view_angles()
            x, z = _unproject(float(pt.x()), float(pt.y()), az, el)
            lay = st["layout"]
            if kind == "geo":
                i = cb_comp.currentIndex(); comps = lay.machine_components
                if 0 <= i < len(comps):
                    c = comps[i]; w = c.x1 - c.x0; h = c.y1 - c.y0
                    c.x0 = x - w / 2; c.x1 = x + w / 2; c.y0 = z - h / 2; c.y1 = z + h / 2
            elif cb_click.currentText() == "Place point":
                axes = [a for a, cb in (("X", cbx), ("Y", cby), ("Z", cbz)) if cb.isChecked()] or ["Y"]
                comp = _comp_at(x, z)
                for a in axes:
                    slot, ch = _next_channel()
                    lay.points.append(MeasPoint(idx=len(lay.points) + 1, component=comp, position_ref="Center",
                                                dof="+" + a, module_slot=slot, channel_index=ch,
                                                number=sp_num.value(), meas_type=cb_mtype.currentText(),
                                                x_norm=x, y_norm=z))
                sp_num.setValue(sp_num.value() + 1); _fill_points()
            _draw_train()
        except Exception:  # noqa: BLE001
            pass

    def _sync_comp_combo():
        cur = cb_comp.currentIndex()
        cb_comp.blockSignals(True); cb_comp.clear()
        cb_comp.addItems([c.display() for c in st["layout"].machine_components])
        if 0 <= cur < cb_comp.count():
            cb_comp.setCurrentIndex(cur)
        cb_comp.blockSignals(False); _load_comp_dims()

    def _load_comp_dims():
        i = cb_comp.currentIndex(); comps = st["layout"].machine_components
        if not (0 <= i < len(comps)):
            return
        c = comps[i]
        for sp in (sp_len, sp_hei, sp_wid): sp.blockSignals(True)
        sp_len.setValue(round(c.x1 - c.x0, 2)); sp_hei.setValue(round(c.y1 - c.y0, 2)); sp_wid.setValue(round(2 * c.depth, 2))
        for sp in (sp_len, sp_hei, sp_wid): sp.blockSignals(False)
        e_compname.blockSignals(True); e_compname.setText(c.display()); e_compname.blockSignals(False)

    def _apply_comp_name(*_):
        i = cb_comp.currentIndex(); comps = st["layout"].machine_components
        if not (0 <= i < len(comps)):
            return
        comps[i].label = e_compname.text().strip()
        # refresca el nombre en el combo sin perder la selección
        cb_comp.blockSignals(True); cb_comp.setItemText(i, comps[i].display()); cb_comp.blockSignals(False)
        _draw_train(fit=False)

    def _apply_comp_dims(*_):
        i = cb_comp.currentIndex(); comps = st["layout"].machine_components
        if not (0 <= i < len(comps)):
            return
        c = comps[i]; cx = (c.x0 + c.x1) / 2; cz = (c.y0 + c.y1) / 2
        L = sp_len.value(); H = sp_hei.value(); W = sp_wid.value()
        c.x0 = cx - L / 2; c.x1 = cx + L / 2; c.y0 = cz - H / 2; c.y1 = cz + H / 2; c.depth = W / 2
        _draw_train(fit=False)

    def _pick_color():
        i = cb_comp.currentIndex(); comps = st["layout"].machine_components
        if not (0 <= i < len(comps)):
            return
        c = comps[i]
        init = QtGui.QColor(c.color) if c.color else QtGui.QColor(_comp_color(c.kind))
        col = QtWidgets.QColorDialog.getColor(init, win, f"Color — {c.display()}")
        if col.isValid():
            c.color = col.name()            # "#rrggbb"
            _draw_train(fit=False)

    def _reset_color():
        i = cb_comp.currentIndex(); comps = st["layout"].machine_components
        if 0 <= i < len(comps):
            comps[i].color = ""             # vuelve al color por tipo
            _draw_train(fit=False)

    def _auto_arrange():
        """Alineación GENTIL: respeta dónde dejaste cada equipo; solo destraba solapes
        mínimos entre equipos rotativos (empujando a la derecha lo justo) y reengancha
        los sensores a su equipo. NO re-fluye al origen."""
        lay = st["layout"]; comps = lay.machine_components
        if not comps:
            return
        def owner(mp):
            for c in comps:
                if c.x0 <= mp.x_norm <= c.x1:
                    return c
            return min(comps, key=lambda c: abs((c.x0 + c.x1) / 2 - mp.x_norm))
        rel = []
        for mp in lay.points:
            c = owner(mp); Lx = (c.x1 - c.x0) or 1.0; Lz = (c.y1 - c.y0) or 1.0
            rel.append((mp, id(c), (mp.x_norm - c.x0) / Lx, (mp.y_norm - c.y0) / Lz))
        # destrabar SOLO si dos equipos rotativos se cruzan en X *y* en Z (colisión real);
        # así respeta apilados intencionales (ej. skid debajo del motor = distinto Z).
        drive = sorted([c for c in comps if not is_pipe(c.kind)], key=lambda c: c.x0)
        for k in range(1, len(drive)):
            prev, cur = drive[k - 1], drive[k]
            x_ov = cur.x0 < prev.x1
            z_ov = (cur.y0 < prev.y1) and (cur.y1 > prev.y0)
            if x_ov and z_ov:                        # chocan de verdad → empujar lo justo
                shift = prev.x1 - cur.x0
                cur.x0 += shift; cur.x1 += shift
        idmap = {id(c): c for c in comps}
        for (mp, cid, fx, fz) in rel:                # sensores siguen a su equipo
            c = idmap.get(cid)
            if c:
                mp.x_norm = c.x0 + fx * (c.x1 - c.x0); mp.y_norm = c.y0 + fz * (c.y1 - c.y0)

    def _refresh_summary():
        lay = st["layout"]
        tbl_sum.setRowCount(0)
        for mp in lay.points:
            r = tbl_sum.rowCount(); tbl_sum.insertRow(r)
            for c, v in enumerate([str(mp.number or mp.idx), mp.code, mp.component, mp.position_ref,
                                   mp.dof, mp.meas_type, str(mp.bnc)]):
                tbl_sum.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        lbl_sum.setText(f"{lay.name} · {lay.client} · {'/'.join(lay.test_modes)} · "
                        f"{lay.n_channels()} sensors · {len(lay.machine_components)} components · "
                        f"{lay.running_speed_rpm:.0f} RPM")

    def _apply_all():
        _table_to_layout(); _auto_arrange()
        errs = st["layout"].validate()
        _fill_points(); _sync_comp_combo(); _refresh_summary(); _draw_train(fit=None)  # deja la vista como está
        st["acc"] = FRFAccumulator(st["layout"].fs_hz, st["layout"].block_size)
        if errs:
            lbl_applyinfo.setText("⚠ " + " · ".join(errs[:2])); lbl_applyinfo.setStyleSheet(f"color:{AMBER};font-weight:700;")
        else:
            lbl_applyinfo.setText(f"✓ {st['layout'].n_channels()} sensors · arranged · valid")
            lbl_applyinfo.setStyleSheet(f"color:{GREEN};font-weight:700;")
        lbl_val.setText(lbl_applyinfo.text())

    def _place_by_norm():
        _table_to_layout(); auto_place_by_norm(st["layout"])
        _fill_points(); _refresh_summary(); _draw_train(fit=True)
        QtWidgets.QMessageBox.information(win, "Standard placement",
            f"✅ {st['layout'].n_channels()} sensors placed per API 670 / ISO 20816 "
            "(NDE/DE bearings, X/Y/Z, driver→driven).")

    def _clear_points():
        st["layout"].points = []; _fill_points(); _refresh_summary(); _draw_train(fit=True)

    def _add_component():
        lay = st["layout"]; kind = cb_kind.currentText()
        n = len([c for c in lay.machine_components if not is_pipe(c.kind)])
        x0 = 0.66 + 0.03 * len(lay.machine_components) if is_pipe(kind) else 0.03 + 0.34 * n
        box = component_default_box(kind, x0)
        lay.machine_components.append(MachineComponent(kind, kind, *box))
        _sync_comp_combo(); _draw_train(fit=True)

    def _del_component():
        lay = st["layout"]
        if lay.machine_components:
            lay.machine_components.pop(); _sync_comp_combo(); _draw_train(fit=True)

    def _del_point():
        r = tbl_pts.currentRow()
        if r >= 0:
            tbl_pts.removeRow(r); _table_to_layout(); _fill_points(); _refresh_summary(); _draw_train()

    def _recommend_acq():
        _table_to_layout()
        mode = "OMA" if "OMA" in st["layout"].test_modes else "EMA"
        rec = recommended_acquisition(mode)
        sp_fs.setValue(int(rec["fs_hz"])); cb_blk.setCurrentText(str(rec["block_size"]))
        sp_fmax.setValue(rec["fmax_hz"]); sp_dur.setValue(int(rec["duration_s"])); sp_tgt.setValue(rec["averages"])
        lbl_reco.setText(f"<b>{mode} — recommended:</b> {rec['note']}")

    def _upd_df(*_):
        fs = float(sp_fs.value()); blk = int(cb_blk.currentText())
        lbl_df.setText(f"Δf = {fs/blk:.3f} Hz · record {blk/fs*1000:.0f} ms · lines to Fmax {int(sp_fmax.value()/(fs/blk))}")

    # save / load
    def _save_local():
        _table_to_layout()
        try:
            p = save_layout_local(st["layout"])
            QtWidgets.QMessageBox.information(win, "Saved", f"Saved locally:\n{p}")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(win, "Save", f"Could not save: {e}")

    def _load_local():
        names = list_layouts_local()
        if not names:
            QtWidgets.QMessageBox.information(win, "Load", "No local configurations yet."); return
        name, ok = QtWidgets.QInputDialog.getItem(win, "Load local", "Configuration:", names, 0, False)
        if ok and name:
            st["layout"] = load_layout_local(name); _reload_from_layout()

    def _load_preset():
        names = list(FACTORY_PRESETS.keys())
        name, ok = QtWidgets.QInputDialog.getItem(win, "Factory presets", "Configuration:", names, 0, False)
        if ok and name:
            st["layout"] = FACTORY_PRESETS[name]()   # arma máquina + sensores + adquisición
            _reload_from_layout()

    def _save_cloud():
        _table_to_layout()
        try:
            from core.modal.modal_cloud import save_layout_cloud
            rc = save_layout_cloud(st["layout"])
        except Exception as e:  # noqa: BLE001
            rc = {"ok": False, "reason": str(e)}
        if rc.get("ok"):
            QtWidgets.QMessageBox.information(win, "Cloud", "☁ Saved to Watermelon System — available on the web.")
        else:
            QtWidgets.QMessageBox.warning(win, "Cloud",
                f"Could not upload ({rc.get('reason','offline')}). Saved locally is still available.")

    def _load_cloud():
        try:
            from core.modal.modal_cloud import list_layouts_cloud, load_layout_cloud
            rows = list_layouts_cloud()
        except Exception:  # noqa: BLE001
            rows = []
        names = [r.get("name") or r.get("id") for r in rows if (r.get("name") or r.get("id"))]
        if not names:
            QtWidgets.QMessageBox.information(win, "Cloud", "No cloud configurations (or offline)."); return
        name, ok = QtWidgets.QInputDialog.getItem(win, "Load from cloud", "Configuration:", names, 0, False)
        if ok and name:
            lay = load_layout_cloud(name)
            if lay is not None:
                st["layout"] = lay; _reload_from_layout()

    def _reload_from_layout():
        lay = st["layout"]
        e_name.setText(lay.name); e_type.setText(lay.machine_type); e_tag.setText(lay.tag)
        e_client.setText(lay.client); e_loc.setText(lay.location); sp_rpm.setValue(lay.running_speed_rpm)
        chk_ema.setChecked("EMA" in lay.test_modes); chk_oma.setChecked("OMA" in lay.test_modes)
        sp_fs.setValue(int(lay.fs_hz)); cb_blk.setCurrentText(str(lay.block_size))
        sp_fmax.setValue(lay.fmax_hz); sp_dur.setValue(int(lay.duration_s))
        win.setWindowTitle(f"Watermelon Modal v{__version__} — {lay.name}")
        _fill_points(); _sync_comp_combo(); _refresh_summary(); _upd_df(); _draw_train(fit=True)

    # wiring
    btn_addcomp.clicked.connect(_add_component); btn_delcomp.clicked.connect(_del_component)
    btn_color.clicked.connect(_pick_color); btn_colreset.clicked.connect(_reset_color)
    e_compname.editingFinished.connect(_apply_comp_name)
    cb_comp.currentIndexChanged.connect(lambda *_: _load_comp_dims())
    for sp in (sp_len, sp_hei, sp_wid): sp.valueChanged.connect(_apply_comp_dims)
    cb_place.currentIndexChanged.connect(lambda *_: _draw_train())
    btn_norm.clicked.connect(_place_by_norm); btn_clrpts.clicked.connect(_clear_points)
    btn_delp.clicked.connect(_del_point); btn_reco.clicked.connect(_recommend_acq)
    sp_fs.valueChanged.connect(_upd_df); cb_blk.currentTextChanged.connect(_upd_df); sp_fmax.valueChanged.connect(_upd_df)
    btn_apply.clicked.connect(_apply_all)
    btn_preset.clicked.connect(_load_preset)
    btn_savelocal.clicked.connect(_save_local); btn_loadlocal.clicked.connect(_load_local)
    btn_savecloud.clicked.connect(_save_cloud); btn_loadcloud.clicked.connect(_load_cloud)

    _fill_points(); _sync_comp_combo(); _refresh_summary(); _upd_df(); _draw_train(fit=True)

    # =====================================================================
    # IMPACT TEST (EMA)
    # =====================================================================
    pg_imp = QtWidgets.QWidget(); il = QtWidgets.QVBoxLayout(pg_imp)
    ctl = QtWidgets.QHBoxLayout()
    btn_hit = QtWidgets.QPushButton("🔨 Impact"); btn_hit.setStyleSheet(
        f"QPushButton{{background:{ACC};font-size:14px;padding:10px 20px;}} QPushButton:hover{{background:#1490c2;}}")
    btn_badhit = QtWidgets.QPushButton("⚠ Impact w/ fault")
    btn_acc = QtWidgets.QPushButton("✓ Accept"); btn_acc.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    btn_rej = QtWidgets.QPushButton("✗ Reject"); btn_rej.setStyleSheet(f"QPushButton{{background:{RED};}}")
    btn_acc.setEnabled(False); btn_rej.setEnabled(False); btn_rst = QtWidgets.QPushButton("↻ Reset")
    for w in (btn_hit, btn_badhit): ctl.addWidget(w)
    ctl.addSpacing(14); ctl.addWidget(btn_acc); ctl.addWidget(btn_rej); ctl.addStretch(1); ctl.addWidget(btn_rst)
    il.addLayout(ctl)
    srow = QtWidgets.QHBoxLayout()
    lbl_avg = QtWidgets.QLabel(); lbl_avg.setStyleSheet("font-size:14px;font-weight:800;")
    lbl_leds = QtWidgets.QLabel(); lbl_leds.setTextFormat(QtCore.Qt.RichText)
    srow.addWidget(lbl_avg); srow.addSpacing(20); srow.addWidget(lbl_leds); srow.addStretch(1)
    il.addLayout(srow)
    grid = QtWidgets.QGridLayout()
    p_frf = pg.PlotWidget(); p_frf.setBackground("w"); p_frf.setLogMode(x=False, y=True)
    p_frf.setLabel("left", "|FRF|"); p_frf.setLabel("bottom", "Frequency", "Hz"); p_frf.setTitle("FRF — H1", color=NAVY)
    p_frf.showGrid(x=True, y=True, alpha=0.3)
    p_coh = pg.PlotWidget(); p_coh.setBackground("w"); p_coh.setYRange(0, 1.05)
    p_coh.setLabel("left", "Coherence"); p_coh.setLabel("bottom", "Frequency", "Hz"); p_coh.setTitle("Coherence", color=NAVY)
    p_coh.showGrid(x=True, y=True, alpha=0.3); p_coh.addItem(pg.InfiniteLine(pos=0.8, angle=0, pen=pg.mkPen(AMBER, style=QtCore.Qt.DashLine)))
    p_time = pg.PlotWidget(); p_time.setBackground("w"); p_time.setTitle("Last hit — force & response", color=NAVY); p_time.showGrid(x=True, y=True, alpha=0.3)
    grid.addWidget(p_frf, 0, 0); grid.addWidget(p_coh, 0, 1); grid.addWidget(p_time, 1, 0, 1, 2)
    grid.setRowStretch(0, 3); grid.setRowStretch(1, 2); il.addLayout(grid, 1)
    tabs.addTab(pg_imp, "Impact test (EMA)")
    cur_frf = p_frf.plot([], [], pen=pg.mkPen("#94a3b8", width=1, style=QtCore.Qt.DashLine))
    avg_frf = p_frf.plot([], [], pen=pg.mkPen(ACC, width=2))
    coh_curve = p_coh.plot([], [], pen=pg.mkPen(GREEN, width=2))
    force_curve = p_time.plot([], [], pen=pg.mkPen(RED, width=1.5)); resp_curve = p_time.plot([], [], pen=pg.mkPen(NAVY, width=1))

    def _leds(q):
        lbl_leds.setText(_led(RED, bool(q and q.overload), "Overload") + " &nbsp;&nbsp; " + _led(AMBER, bool(q and q.double_hit), "Double-hit"))

    def _refresh_impact():
        acc = st["acc"]; lbl_avg.setText(f"Averages: {acc.count} / {st['target']}" + ("   ✅ target" if acc.count >= st["target"] else ""))
        res = acc.result()
        if res is not None:
            m = res.frequencies_hz <= st["layout"].fmax_hz
            avg_frf.setData(res.frequencies_hz[m], np.maximum(res.magnitude[m], 1e-9)); coh_curve.setData(res.frequencies_hz[m], res.coherence[m])
        else:
            avg_frf.setData([], []); coh_curve.setData([], [])

    def _do_hit(fault):
        lay = st["layout"]; dbl = fault and st["rng"].random() < 0.5; over = fault and not dbl
        f, y = synth_impact(lay.fs_hz, lay.block_size, modes=DEMO_MODES, rng=st["rng"], double_hit=dbl, overload=over)
        q = assess_hit(f, y, lay.fs_hz); st["pending"] = (f, y, q)
        acc = st["acc"]; acc.use_force_window = chk_fwin.isChecked(); acc.use_exp_window = chk_ewin.isChecked()
        prev = acc.preview(f, y); m = prev.frequencies_hz <= lay.fmax_hz
        cur_frf.setData(prev.frequencies_hz[m], np.maximum(prev.magnitude[m], 1e-9))
        t = np.arange(lay.block_size) / lay.fs_hz; force_curve.setData(t, f); resp_curve.setData(t, y)
        _leds(q); btn_acc.setEnabled(True); btn_rej.setEnabled(True)
        if q.overload or q.double_hit:
            lbl_leds.setText(lbl_leds.text() + "  <span style='color:#b91c1c;font-weight:800'>→ REJECT</span>")

    def _accept():
        if st["pending"]:
            f, y, _ = st["pending"]; st["acc"].add(f, y); st["pending"] = None
            cur_frf.setData([], []); btn_acc.setEnabled(False); btn_rej.setEnabled(False); _refresh_impact()

    def _reject():
        st["pending"] = None; cur_frf.setData([], []); btn_acc.setEnabled(False); btn_rej.setEnabled(False); _leds(None)

    def _reset_ema():
        st["acc"] = FRFAccumulator(st["layout"].fs_hz, st["layout"].block_size); st["pending"] = None
        for c in (cur_frf, avg_frf, coh_curve, force_curve, resp_curve): c.setData([], [])
        btn_acc.setEnabled(False); btn_rej.setEnabled(False); _leds(None); _refresh_impact()

    btn_hit.clicked.connect(lambda: _do_hit(False)); btn_badhit.clicked.connect(lambda: _do_hit(True))
    btn_acc.clicked.connect(_accept); btn_rej.clicked.connect(_reject); btn_rst.clicked.connect(_reset_ema)
    _leds(None); _refresh_impact()

    # =====================================================================
    # OMA CAPTURE
    # =====================================================================
    pg_oc = QtWidgets.QWidget(); cl2 = QtWidgets.QVBoxLayout(pg_oc)
    crow = QtWidgets.QHBoxLayout()
    crow.addWidget(QtWidgets.QLabel("Source:"))
    cb_src = QtWidgets.QComboBox(); cb_src.addItems(["Simulado", f"{DAQ_NAME} (live)"]); crow.addWidget(cb_src)
    if hw_present:
        cb_src.setCurrentIndex(1)                          # hay hardware → live por defecto
    btn_testni = QtWidgets.QPushButton("🔌 Test acquisition")
    btn_ocap = QtWidgets.QPushButton("▶ Capture + FDD"); btn_ocap.setStyleSheet(
        f"QPushButton{{background:{ACC};font-size:14px;padding:10px 20px;}} QPushButton:hover{{background:#1490c2;}}")
    btn_upload = QtWidgets.QPushButton("☁ Upload run to cloud")
    crow.addWidget(btn_testni); crow.addWidget(btn_ocap); crow.addWidget(btn_upload); crow.addStretch(1); cl2.addLayout(crow)

    def _test_ni():
        try:
            import nidaqmx  # noqa: F401  — import interno (no visible al usuario)
            from nidaqmx.system import System
            devs = [{"product_type": d.product_type} for d in System.local().devices]
        except Exception as e:  # noqa: BLE001
            cause = getattr(e, "__cause__", None)
            root = f"\n\nDetalle: {type(cause).__name__}: {cause}" if cause else ""
            QtWidgets.QMessageBox.warning(win, "Watermelon acquisition",
                f"❌ Could not start the acquisition module.\n\n{type(e).__name__}: {e}{root}\n\n"
                "Check the cable, unit power and try again.")
            return
        if not devs:
            QtWidgets.QMessageBox.warning(win, "Watermelon acquisition",
                "⚠ Acquisition unit not detected.\nCheck the USB cable and power.")
            return
        chassis = [d for d in devs if "cdaq" in d["product_type"].lower() or "9178" in d["product_type"]]
        mods = [d for d in devs if "9234" in d["product_type"]]
        nch = len(mods) * 4
        lines = [f"✅ {DAQ_NAME} conectado."]
        lines.append(f"Acquisition modules: {len(mods)} → {nch} channels available.")
        lines.append(f"\nConnection OK — you can capture with Source = {DAQ_NAME} (live)." if mods
                     else "No channel modules detected — check they are properly seated.")
        QtWidgets.QMessageBox.information(win, "Watermelon acquisition", "\n".join(lines))
    btn_testni.clicked.connect(_test_ni)
    ocs = QtWidgets.QHBoxLayout()
    tbl_om = QtWidgets.QTableWidget(0, 4); tbl_om.setHorizontalHeaderLabels(["Freq (Hz)", "Damping (%)", "Complexity (%)", "Class"])
    tbl_om.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); tbl_om.verticalHeader().setVisible(False); tbl_om.setMaximumWidth(520)
    p_svd = pg.PlotWidget(); p_svd.setBackground("w"); p_svd.setLabel("left", "dB | (1 g)² / Hz"); p_svd.setLabel("bottom", "Frequency", "Hz")
    p_svd.setTitle("Singular values of spectral densities — all channels", color=NAVY); p_svd.showGrid(x=True, y=True, alpha=0.3); p_svd.addLegend(offset=(-10, 10))
    ocs.addWidget(tbl_om, 2); ocs.addWidget(p_svd, 3); cl2.addLayout(ocs, 1)
    lbl_ost = QtWidgets.QLabel(""); cl2.addWidget(lbl_ost)
    # Validación automática de modos (validado / dudoso / rechazado + armónicos)
    cl2.addWidget(QtWidgets.QLabel("<b>Automatic mode validation</b> "
                                   "<span style='color:#64748b'>(validated / doubtful / rejected)</span>"))
    tbl_val = QtWidgets.QTableWidget(0, 6)
    tbl_val.setHorizontalHeaderLabels(["fn (Hz)", "ζ (%)", "Complex (%)", "Verdict", "SSI/Harm", "Reasons"])
    tbl_val.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tbl_val.verticalHeader().setVisible(False); tbl_val.setMaximumHeight(180)
    cl2.addWidget(tbl_val)
    lbl_val = QtWidgets.QLabel(""); cl2.addWidget(lbl_val)
    tabs.addTab(pg_oc, "OMA capture")

    def _oma_capture():
        from scipy.signal import lfilter
        _table_to_layout(); lay = st["layout"]; fs = lay.fs_hz; nch = lay.n_channels()
        if nch < 2:
            QtWidgets.QMessageBox.information(win, "OMA", "Add ≥2 active sensors first (Sensors tab)."); return
        secs = min(float(lay.duration_s), 60.0); N = int(secs * fs); rng = st["rng"]
        live = cb_src.currentIndex() == 1
        data = None
        if live:                                        # captura REAL con hardware (con fallback)
            lbl_ost.setText(f"Conectando con {DAQ_NAME}…"); QtWidgets.QApplication.processEvents()
            try:
                data, fs = _capture_ni(lay, secs)
                lbl_ost.setText(f"{DAQ_NAME}: {data.shape[0]} muestras · {data.shape[1]} canales")
            except Exception as e:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(win, DAQ_NAME,
                    f"No acquisition / capture failed → using simulated.\n\n{type(e).__name__}: {e}")
                data = None
        if data is None:                                # simulado
            lbl_ost.setText(f"Capturing {secs:.0f}s @ {fs:.0f}Hz · {nch} channels (simulated)…")
            QtWidgets.QApplication.processEvents()
            data = np.zeros((N, nch))
            for sm in DEMO_MODES:
                fn, z = sm.fn_hz, sm.zeta
                wn = 2 * np.pi * fn; wd = wn * (1 - z * z) ** 0.5; r = np.exp(-z * wn / fs); th = wd / fs
                q = lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], rng.standard_normal(N)); q /= (np.std(q) or 1)
                data += np.outer(q, rng.standard_normal(nch))
            data += 0.05 * rng.standard_normal((N, nch))
        st["oma_data"] = (data, float(fs))              # guardado para SSI / upload
        fmax = min(fs / 2.56, lay.fmax_hz)
        fdd = run_oma(data, fs, nperseg=4096, f_min_hz=5.0, f_max_hz=fmax, channel_names=lay.channel_names())
        st["oma_fdd"] = fdd
        freqs = fdd.frequencies_hz; sv = np.asarray(fdd.singular_values)
        if sv.ndim == 1: sv = sv[None, :]
        band = freqs <= fmax
        p_svd.clear(); _svcol = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b"]
        p_svd.setTitle(f"Singular values — {sv.shape[0]} channels (SV1 dominant; SV2–4 reveal close modes)", color=NAVY)
        for i in range(sv.shape[0]):
            col = _svcol[i] if i < 4 else "#94a3b8"; wdt = 1.8 if i == 0 else (1.1 if i < 4 else 0.6)
            p_svd.plot(freqs[band], 10 * np.log10(np.maximum(sv[i][band], 1e-30)), pen=pg.mkPen(col, width=wdt), name=(f"SV{i+1}" if i < 4 else None))
        for m in fdd.modes:
            j = int(np.argmin(np.abs(freqs - m.natural_frequency_hz)))
            p_svd.addItem(pg.ScatterPlotItem([m.natural_frequency_hz], [10 * np.log10(max(sv[0][j], 1e-30))], size=10, symbol="o",
                          pen=pg.mkPen(RED, width=2), brush=pg.mkBrush(255, 255, 255, 0)))
        tbl_om.setRowCount(0)
        for m in fdd.modes:
            rr = tbl_om.rowCount(); tbl_om.insertRow(rr)
            for c, v in enumerate([f"{m.natural_frequency_hz:.2f}", f"{m.damping_ratio_pct:.3f}", f"{m.complexity_pct:.1f}", m.classification]):
                tbl_om.setItem(rr, c, QtWidgets.QTableWidgetItem(v))
        lbl_ost.setText(f"✅ FDD done — {len(fdd.modes)} modes. See Comparative (EMA vs OMA) and Campbell.")
        lbl_ost.setStyleSheet(f"color:{GREEN};font-weight:700;"); _refresh_campbell(); _refresh_comparative()
        _refresh_validation()

    def _refresh_validation():
        fdd = st.get("oma_fdd")
        if fdd is None:
            return
        from core.modal.mode_validation import validate_modes, summarize as _mv_sum
        ssi_res = st.get("ssi")
        ssi_freqs = [m.frequency_hz for m in ssi_res.modes] if (ssi_res and ssi_res.modes) else []
        verdicts = validate_modes(fdd.modes, ssi_freqs_hz=ssi_freqs,
                                  running_speed_rpm=st["layout"].running_speed_rpm)
        _vc = {"validated": "#16a34a", "doubtful": "#f59e0b", "rejected": "#dc2626"}
        tbl_val.setRowCount(0)
        for v in verdicts:
            r = tbl_val.rowCount(); tbl_val.insertRow(r)
            flag = ("✓SSI " if v.confirmed_by_ssi else "") + ("⚠Harm" if v.is_harmonic else "")
            cells = [f"{v.frequency_hz:.2f}", f"{v.damping_ratio_pct:.3f}", f"{v.complexity_pct:.1f}",
                     v.verdict.capitalize(), flag.strip(), "; ".join(v.reasons)]
            for c, txt in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(txt)
                if c == 3:
                    it.setForeground(QtGui.QColor(_vc.get(v.verdict, "#0f172a")))
                tbl_val.setItem(r, c, it)
        lbl_val.setText(_mv_sum(verdicts)); lbl_val.setStyleSheet("color:#334155;")

    def _capture_ni(lay, secs):
        """Captura REAL continua desde la NI 9234 (IEPE). Lanza excepción si no hay HW."""
        import tempfile
        from core.modal.acq_backend import (AcquisitionConfig, ChannelConfig, capture,
                                            list_available_devices)
        from nptdms import TdmsFile
        # autodetectar el nombre del chasis (cDAQ…) para no depender de "cDAQ1"
        chassis = "cDAQ1"
        try:
            for d in list_available_devices():
                if "cdaq" in d["product_type"].lower() or "9178" in d["product_type"]:
                    chassis = d["name"]; break
        except Exception:  # noqa: BLE001
            pass
        chans = [ChannelConfig(name=p.code, coupling="IEPE",
                               sensitivity_mv_per_eu=p.sensitivity_mv_per_g, bnc_port=p.bnc, units="g")
                 for p in lay.active_points()]
        tmp = os.path.join(tempfile.gettempdir(), "wm_modal_oma.tdms")
        cfg = AcquisitionConfig(mode="oma_continuous", sample_rate_hz=lay.fs_hz,
                                duration_s=float(secs), channels=chans, chassis_name=chassis,
                                output_tdms_path=tmp)
        path = capture(cfg, lambda *a, **k: None)
        tf = TdmsFile.read(str(path)); grp = tf.groups()[0]
        cols = [ch[:] for ch in grp.channels()]
        return np.asarray(cols, float).T, lay.fs_hz

    def _upload_run():
        fdd = st.get("oma_fdd")
        if fdd is None:
            QtWidgets.QMessageBox.information(win, "Cloud", "Run OMA capture first."); return
        try:
            from core.modal import modal_cloud
            lay = st["layout"]
            # SVD SV1 (curva de valores singulares) submuestreada para el reporte
            freqs = np.asarray(fdd.frequencies_hz); sv = np.asarray(fdd.singular_values)
            if sv.ndim == 1:
                sv = sv[None, :]
            fmax = min(float(lay.fs_hz) / 2.56, float(lay.fmax_hz))
            band = freqs <= fmax
            fb = freqs[band]; sv1 = sv[0][band]
            step = max(1, len(fb) // 900)
            svd = {"freqs": fb[::step].tolist(), "sv1": sv1[::step].tolist()}
            def _sh(m):
                s = np.asarray(getattr(m, "mode_shape", []), complex).ravel()
                return {"re": s.real.tolist(), "im": s.imag.tolist()}
            modes = [{"fn": m.natural_frequency_hz, "zeta": m.damping_ratio_pct,
                      "complexity": m.complexity_pct, "class": m.classification, "shape": _sh(m)}
                     for m in fdd.modes]
            # modos EMA (si hay) para la correlación EMA↔OMA en el reporte
            ema = []; ema_block = None
            res = st["acc"].result()
            if res is not None:
                ema_peaks = modes_from_frf(res, fmin=5, fmax=lay.fmax_hz, exp_tau=st["acc"].exp_tau())
                ema = [mp.frequency_hz for mp in ema_peaks]
                # curva FRF + coherencia (submuestreada) para verla idéntica en la web
                ef = np.asarray(res.frequencies_hz); emag = np.asarray(res.magnitude)
                ecoh = np.asarray(res.coherence)
                eb = ef <= lay.fmax_hz
                st_e = max(1, int(np.sum(eb)) // 800)
                ema_block = {
                    "freqs": ef[eb][::st_e].tolist(),
                    "mag_db": (20 * np.log10(np.maximum(emag[eb][::st_e], 1e-12))).tolist(),
                    "coh": ecoh[eb][::st_e].tolist(),
                    "modes": [{"fn": mp.frequency_hz, "zeta": mp.damping_ratio_pct,
                               "coh": getattr(mp, "coherence_at_peak", None)} for mp in ema_peaks],
                }
            # SSI (si se corrió): modos + diagrama de estabilización para la web
            ssi_block = None
            ssi_res = st.get("ssi")
            if ssi_res is not None and getattr(ssi_res, "modes", None):
                ssi_block = {
                    "modes": [{"fn": m.frequency_hz, "zeta": m.damping_ratio_pct,
                               "std_fn": m.std_frequency_hz, "std_zeta": m.std_damping_pct}
                              for m in ssi_res.modes],
                    "diagram": [[int(o), np.asarray(fr).tolist(),
                                 [bool(x) for x in np.asarray(mk).tolist()]]
                                for (o, fr, mk) in ssi_res.diagram],
                }
            payload = {"name": lay.name, "kind": "OMA", "modes": modes, "svd": svd,
                       "channel_names": lay.channel_names(), "running_rpm": lay.running_speed_rpm,
                       "ema_modes": ema, "ema": ema_block, "ssi": ssi_block,
                       "client": lay.client, "asset": lay.machine_type,
                       "location": lay.location, "layout": lay.to_dict()}
            r = modal_cloud.save_run(lay.name, payload)
            if r.get("ok"):
                QtWidgets.QMessageBox.information(win, "Cloud",
                    f"☁ Run uploaded ({len(fdd.modes)} modes). Generate the report from the web.")
            else:
                QtWidgets.QMessageBox.warning(win, "Cloud", f"Could not upload: {r.get('reason')}")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(win, "Cloud", f"Upload failed: {type(e).__name__}: {e}")
    btn_ocap.clicked.connect(_oma_capture); btn_upload.clicked.connect(_upload_run)

    # =====================================================================
    # MODES (EMA)
    # =====================================================================
    pg_mod = QtWidgets.QWidget(); ql = QtWidgets.QVBoxLayout(pg_mod)
    mrow = QtWidgets.QHBoxLayout()
    btn_ident = QtWidgets.QPushButton("🎯 Identify modes (EMA)"); btn_ident.setStyleSheet(f"QPushButton{{background:{ACC};}}")
    mrow.addWidget(btn_ident); mrow.addWidget(QtWidgets.QLabel("Peak-picking + half-power damping (ISO 7626-6).")); mrow.addStretch(1); ql.addLayout(mrow)
    ms = QtWidgets.QHBoxLayout()
    tbl_modes = QtWidgets.QTableWidget(0, 4); tbl_modes.setHorizontalHeaderLabels(["Freq (Hz)", "Damping (%)", "Coherence", "Reliable"])
    tbl_modes.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); tbl_modes.verticalHeader().setVisible(False); tbl_modes.setMaximumWidth(520)
    p_nyq = pg.PlotWidget(); p_nyq.setBackground("w"); p_nyq.setAspectLocked(True); p_nyq.setTitle("Nyquist", color=NAVY); p_nyq.showGrid(x=True, y=True, alpha=0.3)
    nyq_curve = p_nyq.plot([], [], pen=pg.mkPen(NAVY, width=1.5)); ms.addWidget(tbl_modes, 2); ms.addWidget(p_nyq, 3); ql.addLayout(ms, 1)
    tabs.addTab(pg_mod, "Modes (EMA)")

    def _identify():
        res = st["acc"].result()
        if res is None:
            QtWidgets.QMessageBox.information(win, "Modes", "Accept some hits in Impact test."); return
        modes = modes_from_frf(res, fmin=5.0, fmax=st["layout"].fmax_hz, exp_tau=st["acc"].exp_tau()); tbl_modes.setRowCount(0)
        for mo in modes:
            r = tbl_modes.rowCount(); tbl_modes.insertRow(r)
            for c, v in enumerate([f"{mo.frequency_hz:.1f}", f"{mo.damping_ratio_pct:.2f}",
                                   (f"{mo.coherence_at_peak:.2f}" if mo.coherence_at_peak is not None else "—"), ("✓" if mo.is_reliable else "✗")]):
                tbl_modes.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        m = (res.frequencies_hz <= st["layout"].fmax_hz) & (res.frequencies_hz >= 5.0)
        nyq_curve.setData(res.frf_complex[m].real, res.frf_complex[m].imag)
    btn_ident.clicked.connect(_identify)

    # =====================================================================
    # COMPARATIVE (EMA vs OMA)
    # =====================================================================
    pg_cmp = QtWidgets.QWidget(); cpl = QtWidgets.QVBoxLayout(pg_cmp)
    cpl.addWidget(QtWidgets.QLabel("<b>EMA ↔ OMA correlation</b> — impact-test modes vs operational modes "
                                   "(ISO 7626-6 / API 684)."))
    crow3 = QtWidgets.QHBoxLayout(); btn_cmp = QtWidgets.QPushButton("↻ Compare EMA vs OMA"); btn_cmp.setStyleSheet(f"QPushButton{{background:{ACC};}}")
    crow3.addWidget(btn_cmp); crow3.addStretch(1); cpl.addLayout(crow3)
    tbl_cmp = QtWidgets.QTableWidget(0, 4); tbl_cmp.setHorizontalHeaderLabels(["EMA mode (Hz)", "OMA mode (Hz)", "Δf (Hz)", "Δ (%)"])
    tbl_cmp.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); tbl_cmp.verticalHeader().setVisible(False); cpl.addWidget(tbl_cmp, 1)
    lbl_cmp = QtWidgets.QLabel(""); lbl_cmp.setWordWrap(True); cpl.addWidget(lbl_cmp)
    tabs.addTab(pg_cmp, "Comparative")

    def _refresh_comparative():
        from core.modal.ema_oma_correlation import correlate, summarize as _cs
        res = st["acc"].result()
        ema = [mp.frequency_hz for mp in modes_from_frf(res, fmin=5, fmax=st["layout"].fmax_hz, exp_tau=st["acc"].exp_tau())] if res else []
        oma = [m.natural_frequency_hz for m in st["oma_fdd"].modes] if st["oma_fdd"] else []
        tbl_cmp.setRowCount(0)
        if not ema or not oma:
            miss = []
            if not ema: miss.append("EMA (accept hits in Impact test)")
            if not oma: miss.append("OMA (run OMA capture)")
            lbl_cmp.setText("Missing: " + " and ".join(miss) + "."); return
        matches = correlate(ema, oma, tol_hz=2.5)
        for m in matches:
            r = tbl_cmp.rowCount(); tbl_cmp.insertRow(r)
            for c, v in enumerate([f"{m.ema_hz:.2f}", f"{m.oma_hz:.3f}", f"{m.delta_hz:.3f}", f"{m.delta_pct:.2f}"]):
                tbl_cmp.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        lbl_cmp.setText(_cs(matches))
    btn_cmp.clicked.connect(_refresh_comparative); _refresh_comparative()

    # =====================================================================
    # CAMPBELL
    # =====================================================================
    pg_cam = QtWidgets.QWidget(); cml = QtWidgets.QVBoxLayout(pg_cam)
    crow2 = QtWidgets.QHBoxLayout(); btn_refc = QtWidgets.QPushButton("↻ Recompute Campbell"); btn_refc.setStyleSheet(f"QPushButton{{background:{ACC};}}")
    crow2.addWidget(btn_refc); crow2.addWidget(QtWidgets.QLabel("Automatic fn↔order crossings (½×..4×) + operating bands (API 684).")); crow2.addStretch(1); cml.addLayout(crow2)
    cams = QtWidgets.QHBoxLayout()
    p_cam = pg.PlotWidget(); p_cam.setBackground("w"); p_cam.setLabel("left", "Frequency", "Hz"); p_cam.setLabel("bottom", "Speed", "RPM")
    p_cam.setTitle("Campbell diagram", color=NAVY); p_cam.showGrid(x=True, y=True, alpha=0.3)
    tbl_cam = QtWidgets.QTableWidget(0, 5); tbl_cam.setMaximumWidth(520); tbl_cam.verticalHeader().setVisible(False)
    tbl_cam.setHorizontalHeaderLabels(["Mode", "Order", "RPM", "Margin%", "Status"]); tbl_cam.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    cams.addWidget(tbl_cam, 2); cams.addWidget(p_cam, 3); cml.addLayout(cams, 1)
    lbl_cam = QtWidgets.QLabel(""); lbl_cam.setWordWrap(True); cml.addWidget(lbl_cam)
    tabs.addTab(pg_cam, "Campbell")

    def _current_modes():
        if st["oma_fdd"] is not None and st["oma_fdd"].modes:
            return [m.natural_frequency_hz for m in st["oma_fdd"].modes]
        res = st["acc"].result()
        if res is not None:
            return [mp.frequency_hz for mp in modes_from_frf(res, fmin=5, fmax=st["layout"].fmax_hz, exp_tau=st["acc"].exp_tau())]
        return []

    def _refresh_campbell():
        p_cam.clear(); modes = _current_modes(); rpm_op = st["layout"].running_speed_rpm or 1185.0
        SM = 0.15; lo, hi = rpm_op * (1 - SM), rpm_op * (1 + SM); rpm_max = max(rpm_op * 1.4, 1500.0)
        if not modes:
            lbl_cam.setText("No modes yet — run OMA capture or identify EMA modes."); tbl_cam.setRowCount(0); return
        ymax = max(modes) * 1.30; orders = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0); rpm = np.linspace(0, rpm_max, 60)
        reg = pg.LinearRegionItem([max(0, lo), min(rpm_max, hi)], movable=False, brush=pg.mkBrush(239, 68, 68, 32)); reg.setZValue(-20); p_cam.addItem(reg)
        reg2 = pg.LinearRegionItem([rpm_op / 2 * (1 - SM), rpm_op / 2 * (1 + SM)], movable=False, brush=pg.mkBrush(245, 158, 11, 26)); reg2.setZValue(-20); p_cam.addItem(reg2)
        for o in orders:
            p_cam.plot(rpm, o * rpm / 60.0, pen=pg.mkPen("#6B7280", width=1, style=QtCore.Qt.DotLine))
            # etiqueta donde la línea de orden sale del gráfico (queda siempre visible)
            if o * rpm_max / 60.0 <= ymax:
                lx, ly = rpm_max * 0.98, o * rpm_max / 60.0
            else:
                lx, ly = ymax * 60.0 / o, ymax * 0.98
            t = pg.TextItem(f"{o:g}×", color="#6B7280", anchor=(0.5, 1.0)); t.setPos(lx, ly); p_cam.addItem(t)
        for fn in modes:
            p_cam.plot([0, rpm_max], [fn, fn], pen=pg.mkPen(GREEN, width=2))
        bands = [SpeedBand(rpm_op, SM * rpm_op, f"Operating {rpm_op:.0f}±{SM*100:.0f}%"), SpeedBand(rpm_op / 2, SM * rpm_op / 2, "½ speed")]
        cx = compute_crossings(modes, 0, rpm_max, orders=orders, bands=bands); sevcol = {"coincidence": RED, "near": AMBER, "clear": "#94a3b8"}
        tbl_cam.setRowCount(0)
        for c in cx:
            p_cam.addItem(pg.ScatterPlotItem([c.crossing_rpm], [c.mode_hz], size=12, symbol="x", pen=pg.mkPen(sevcol[c.severity], width=2), brush=pg.mkBrush(sevcol[c.severity])))
            if c.severity in ("coincidence", "near"):
                r = tbl_cam.rowCount(); tbl_cam.insertRow(r)
                for j, v in enumerate([f"{c.mode_hz:.2f}", f"{c.order:g}×", f"{c.crossing_rpm:.0f}", f"{c.sep_margin_pct:.1f}", {"coincidence": "Coincidence", "near": "Near"}[c.severity]]):
                    it = QtWidgets.QTableWidgetItem(v)
                    if j == 4: it.setForeground(QtGui.QBrush(QtGui.QColor(sevcol[c.severity])))
                    tbl_cam.setItem(r, j, it)
        p_cam.plot([rpm_op, rpm_op], [0, ymax], pen=pg.mkPen(NAVY, width=3))
        t_op = pg.TextItem(f"N machine\n{rpm_op:.0f} RPM", color=NAVY, anchor=(0.5, 1.0)); t_op.setPos(rpm_op, ymax * 0.995); p_cam.addItem(t_op)
        for xb, lab in ((lo, f"−{SM*100:.0f}%\n{lo:.0f}"), (hi, f"+{SM*100:.0f}%\n{hi:.0f}")):
            p_cam.plot([xb, xb], [0, ymax], pen=pg.mkPen(RED, width=1, style=QtCore.Qt.DashLine))
            tt = pg.TextItem(lab, color=RED, anchor=(0.5, 1.0)); tt.setPos(xb, ymax * 0.82); p_cam.addItem(tt)
        p_cam.setLabel("bottom", f"Speed · N machine {rpm_op:.0f} RPM · API 684 margin ±{SM*100:.0f}%", "RPM")
        p_cam.setXRange(0, rpm_max); p_cam.setYRange(0, ymax)
        n_coin = sum(1 for c in cx if c.severity == "coincidence")
        lbl_cam.setText(f"<b>N machine = {rpm_op:.0f} RPM</b> · API 684 separation margin ±{SM*100:.0f}% "
                        f"(zone {lo:.0f}–{hi:.0f} RPM) · <b>{n_coin}</b> coincidence(s) inside the operating "
                        "band. A crossing does not confirm resonance by itself — correlate with amplitude/phase.")
    btn_refc.clicked.connect(_refresh_campbell)

    # =====================================================================
    # SSI (subspace) — premium: modos con incertidumbre + estabilización
    # =====================================================================
    pg_ssi = QtWidgets.QWidget(); ssl = QtWidgets.QVBoxLayout(pg_ssi)
    srow = QtWidgets.QHBoxLayout()
    btn_ssi = QtWidgets.QPushButton("🎯 Run SSI (subspace)"); btn_ssi.setStyleSheet(f"QPushButton{{background:{ACC};}}")
    srow.addWidget(btn_ssi)
    srow.addWidget(QtWidgets.QLabel("Time-domain SSI-COV — natural modes with UNCERTAINTY + stabilization diagram (beyond FDD)."))
    srow.addStretch(1); ssl.addLayout(srow)
    ssplit = QtWidgets.QHBoxLayout()
    tbl_ssi = QtWidgets.QTableWidget(0, 6)
    tbl_ssi.setHorizontalHeaderLabels(["Freq (Hz)", "±", "Damping (%)", "±", "Complexity (%)", "Stable×"])
    tbl_ssi.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tbl_ssi.verticalHeader().setVisible(False); tbl_ssi.setMaximumWidth(560)
    p_stab = pg.PlotWidget(); p_stab.setBackground("w"); p_stab.setLabel("left", "Model order")
    p_stab.setLabel("bottom", "Frequency", "Hz"); p_stab.setTitle("Stabilization diagram", color=NAVY)
    p_stab.showGrid(x=True, y=True, alpha=0.3)
    ssplit.addWidget(tbl_ssi, 2); ssplit.addWidget(p_stab, 3); ssl.addLayout(ssplit, 1)
    lbl_ssi = QtWidgets.QLabel(""); lbl_ssi.setWordWrap(True); ssl.addWidget(lbl_ssi)
    tabs.addTab(pg_ssi, "SSI (subspace)")

    def _run_ssi():
        d = st.get("oma_data")
        if not d:
            QtWidgets.QMessageBox.information(win, "SSI", "Run OMA capture first (it stores the time data)."); return
        from core.modal.ssi import run_ssi_cov
        data, fs = d; lay = st["layout"]; fmax = min(fs / 2.56, lay.fmax_hz)
        lbl_ssi.setText("Running SSI-COV (sweeping model orders)…"); QtWidgets.QApplication.processEvents()
        try:
            res = run_ssi_cov(data, fs, orders=range(4, 45, 2), i_block=25, fmin_hz=2.0, fmax_hz=fmax)
        except Exception as e:  # noqa: BLE001
            lbl_ssi.setText(f"SSI error: {type(e).__name__}: {e}"); return
        st["ssi"] = res
        tbl_ssi.setRowCount(0)
        for m in res.modes:
            r = tbl_ssi.rowCount(); tbl_ssi.insertRow(r)
            for c, v in enumerate([f"{m.frequency_hz:.2f}", f"{m.std_frequency_hz:.3f}",
                                   f"{m.damping_ratio_pct:.2f}", f"{m.std_damping_pct:.2f}",
                                   f"{m.complexity_pct:.0f}", str(m.n_stable)]):
                tbl_ssi.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        p_stab.clear()
        for (order, freqs, mask) in res.diagram:
            if len(freqs) == 0:
                continue
            st_f = freqs[mask]; un_f = freqs[~mask]
            if len(un_f):
                p_stab.addItem(pg.ScatterPlotItem(un_f, [order] * len(un_f), size=6, symbol="x",
                               pen=pg.mkPen("#cbd5e1"), brush=pg.mkBrush("#cbd5e1")))
            if len(st_f):
                p_stab.addItem(pg.ScatterPlotItem(st_f, [order] * len(st_f), size=8, symbol="o",
                               pen=pg.mkPen(GREEN, width=1.5), brush=pg.mkBrush(GREEN)))
        for m in res.modes:
            p_stab.plot([m.frequency_hz, m.frequency_hz], [0, res.orders[-1]],
                        pen=pg.mkPen(NAVY, width=1, style=QtCore.Qt.DashLine))
        lbl_ssi.setText(f"✅ SSI: {len(res.modes)} stable modes. Green = pole stable across orders; "
                        "the blue line is the identified mode. The ± is the UNCERTAINTY (dispersion).")
        lbl_ssi.setStyleSheet(f"color:{GREEN};font-weight:700;")
        _anim_reload_modes()
    btn_ssi.clicked.connect(_run_ssi)

    # =====================================================================
    # MODE SHAPES — animación 3D (modal vs estructural en movimiento)
    # =====================================================================
    pg_anim = QtWidgets.QWidget(); anl = QtWidgets.QVBoxLayout(pg_anim)
    arow = QtWidgets.QHBoxLayout()
    arow.addWidget(QtWidgets.QLabel("Source:"))
    cb_asrc = QtWidgets.QComboBox(); cb_asrc.addItems(["OMA (FDD)", "SSI"]); arow.addWidget(cb_asrc)
    arow.addWidget(QtWidgets.QLabel("Mode:"))
    cb_amode = QtWidgets.QComboBox(); cb_amode.setMinimumWidth(160); arow.addWidget(cb_amode)
    arow.addWidget(QtWidgets.QLabel("Scale:"))
    sp_ascale = QtWidgets.QDoubleSpinBox(); sp_ascale.setRange(0.01, 0.5); sp_ascale.setValue(0.10); sp_ascale.setSingleStep(0.02)
    arow.addWidget(sp_ascale)
    chk_showsen = QtWidgets.QCheckBox("Show sensors"); chk_showsen.setChecked(False)
    arow.addWidget(chk_showsen)
    btn_play = QtWidgets.QPushButton("▶ Animate"); btn_play.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    btn_stop = QtWidgets.QPushButton("⏹ Stop")
    arow.addWidget(btn_play); arow.addWidget(btn_stop)
    arow.addSpacing(10); arow.addWidget(QtWidgets.QLabel("View:"))
    btn_v_iso = QtWidgets.QPushButton("Iso"); btn_v_top = QtWidgets.QPushButton("Top")
    btn_v_side = QtWidgets.QPushButton("Side"); btn_v_front = QtWidgets.QPushButton("Front")
    for _b in (btn_v_iso, btn_v_top, btn_v_side, btn_v_front):
        _b.setMaximumWidth(52); _b.setStyleSheet("QPushButton{background:#334155;padding:6px 8px;}")
        arow.addWidget(_b)
    btn_gif = QtWidgets.QPushButton("🎥 Save clip")
    btn_gif.setToolTip("Export a short animated clip (GIF) of the current mode shape.")
    arow.addWidget(btn_gif); arow.addStretch(1)
    anl.addLayout(arow)
    anl.addWidget(QtWidgets.QLabel(
        "<i style='color:#64748b'>Colour = vibration amplitude (green→red). Rotate with the view "
        "buttons or left-drag. The panel on the right shows the modal values and the complexity "
        "(Argand) plot of the selected mode.</i>"))
    # --- layout: 3D a la izquierda, panel de datos + complejidad a la derecha ---
    anim_split = QtWidgets.QHBoxLayout()
    p_anim = pg.PlotWidget(viewBox=OrbitViewBox()); p_anim.setBackground("w"); p_anim.setAspectLocked(True)
    p_anim.hideAxis("left"); p_anim.hideAxis("bottom"); p_anim.setMenuEnabled(False)
    m_anim = Machine3DItem(); p_anim.addItem(m_anim)
    m_anim.set_show_sensors(chk_showsen.isChecked())     # honra el checkbox desde el inicio
    anim_split.addWidget(p_anim, 3)
    right_panel = QtWidgets.QVBoxLayout()
    lbl_modal = QtWidgets.QLabel("Select a mode."); lbl_modal.setTextFormat(QtCore.Qt.RichText)
    lbl_modal.setStyleSheet(f"background:white;border:1px solid #e2e8f0;border-radius:8px;padding:10px;")
    lbl_modal.setAlignment(QtCore.Qt.AlignTop); lbl_modal.setWordWrap(True)
    right_panel.addWidget(lbl_modal)
    p_argand = pg.PlotWidget(); p_argand.setBackground("w"); p_argand.setAspectLocked(True)
    p_argand.setTitle("Complexity (Argand)", color=NAVY); p_argand.showGrid(x=True, y=True, alpha=0.25)
    p_argand.setMouseEnabled(False, False); p_argand.setMenuEnabled(False)
    p_argand.setMinimumWidth(300)
    right_panel.addWidget(p_argand, 1)
    _rp_w = QtWidgets.QWidget(); _rp_w.setLayout(right_panel); _rp_w.setMaximumWidth(360)
    anim_split.addWidget(_rp_w)
    anl.addLayout(anim_split, 1)
    tabs.addTab(pg_anim, "Mode shapes")

    st["_anim_phase"] = 0.0
    anim_timer = QtCore.QTimer(win)

    def _anim_reload_modes():
        cb_amode.blockSignals(True); cb_amode.clear()
        src = cb_asrc.currentText()
        modes = (st["oma_fdd"].modes if (src.startswith("OMA") and st.get("oma_fdd")) else
                 (st["ssi"].modes if (src == "SSI" and st.get("ssi")) else []))
        cb_amode.addItems([f"{m.natural_frequency_hz:.2f} Hz" if hasattr(m, "natural_frequency_hz")
                           else f"{m.frequency_hz:.2f} Hz" for m in modes])
        cb_amode.blockSignals(False)

    def _cur_shape():
        src = cb_asrc.currentText(); i = cb_amode.currentIndex()
        modes = (st["oma_fdd"].modes if (src.startswith("OMA") and st.get("oma_fdd")) else
                 (st["ssi"].modes if (src == "SSI" and st.get("ssi")) else []))
        if not (0 <= i < len(modes)):
            return None
        sh = np.asarray(getattr(modes[i], "mode_shape"), complex).ravel()
        mx = np.max(np.abs(sh)) or 1.0
        return sh / mx

    def _cur_mode():
        src = cb_asrc.currentText(); i = cb_amode.currentIndex()
        modes = (st["oma_fdd"].modes if (src.startswith("OMA") and st.get("oma_fdd")) else
                 (st["ssi"].modes if (src == "SSI" and st.get("ssi")) else []))
        return modes[i] if 0 <= i < len(modes) else None

    def _update_modal_panel():
        m = _cur_mode(); p_argand.clear()
        if m is None:
            lbl_modal.setText("Select a mode."); return
        fn = getattr(m, "natural_frequency_hz", getattr(m, "frequency_hz", 0.0))
        zeta = getattr(m, "damping_ratio_pct", 0.0)
        cplx = getattr(m, "complexity_pct", 0.0)
        sfn = getattr(m, "std_frequency_hz", None); sz = getattr(m, "std_damping_pct", None)
        cls = getattr(m, "classification", "")
        z = zeta / 100.0
        logdec = 2 * np.pi * z / np.sqrt(max(1 - z * z, 1e-9)) if z > 0 else 0.0
        rows = [("Frequency", f"{fn:.3f} Hz")]
        if sfn is not None: rows.append(("Std. frequency", f"± {sfn:.3f} Hz"))
        rows.append(("Damping", f"{zeta:.3f} %"))
        if sz is not None: rows.append(("Std. damping", f"± {sz:.3f} %"))
        rows.append(("Log. decrement", f"{logdec*100:.3f} %"))
        rows.append(("Complexity (MPC)", f"{cplx:.2f} %"))
        if cls: rows.append(("Class", cls))
        rr = st["layout"].running_speed_rpm
        if rr:
            rows.append(("Order (×run)", f"{fn/(rr/60.0):.2f}×"))
        html = "<b style='font-size:13px'>Modal values</b><table cellspacing='5' style='margin-top:6px'>"
        for k, v in rows:
            html += f"<tr><td style='color:#64748b'>{k}&nbsp;&nbsp;</td><td><b>{v}</b></td></tr>"
        html += "</table>"
        # --- autodiagnóstico del modo (complejidad + armónico de giro) ---
        if cplx < 15:
            diag, dcol = "Real structural mode — mode-shape vectors are collinear.", GREEN
        elif cplx < 45:
            diag, dcol = "Moderately complex — plausible; confirm with SSI.", AMBER
        else:
            diag, dcol = "Highly complex — likely forced/harmonic or noisy; verify or discard.", RED
        harm_note = ""
        if rr:
            order = fn / (rr / 60.0)
            k = round(order)
            if k >= 1 and abs(order - k) <= 0.06:
                harm_note = (f"<br><span style='color:{RED}'>⚠ Near {k}× running speed "
                             f"({k*rr/60.0:.1f} Hz) — possible harmonic, not a structural mode.</span>")
            elif abs(order - 0.5) <= 0.04:
                harm_note = (f"<br><span style='color:{AMBER}'>⚠ Near 0.5× — sub-synchronous "
                             f"(check looseness/oil whirl).</span>")
        html += (f"<div style='margin-top:10px;padding:8px;background:#f8fafc;border-radius:6px'>"
                 f"<b>Auto-diagnosis:</b> <span style='color:{dcol}'>{diag}</span>{harm_note}</div>")
        lbl_modal.setText(html)
        # Argand: vector por sensor (magnitud + fase) → colinealidad = modo real
        sh = np.asarray(getattr(m, "mode_shape", []), complex).ravel()
        if sh.size:
            s = sh / (np.max(np.abs(sh)) or 1.0)
            th = np.linspace(0, 2 * np.pi, 72)
            p_argand.plot(np.cos(th), np.sin(th), pen=pg.mkPen("#cbd5e1", width=1))
            p_argand.plot([-1.05, 1.05], [0, 0], pen=pg.mkPen("#e2e8f0", width=1))
            p_argand.plot([0, 0], [-1.05, 1.05], pen=pg.mkPen("#e2e8f0", width=1))
            for c in s:
                p_argand.plot([0, float(c.real)], [0, float(c.imag)], pen=pg.mkPen(ACC, width=1.4))
            p_argand.addItem(pg.ScatterPlotItem([float(c.real) for c in s], [float(c.imag) for c in s],
                                                size=7, brush=pg.mkBrush(ACC), pen=None))
            p_argand.setXRange(-1.1, 1.1); p_argand.setYRange(-1.1, 1.1)

    def _set_view(az, el):
        st["az"] = float(az); st["el"] = float(np.clip(el, 8.0, 88.0))
        m_anim.set_view(st["layout"], np.radians(st["az"]), np.radians(st["el"]), -1)
        p_anim.getViewBox().autoRange(padding=0.2)

    def _save_clip():
        sh = _cur_shape()
        if sh is None:
            QtWidgets.QMessageBox.information(win, "Clip", "Select a mode first."); return
        try:
            from PIL import Image
            from io import BytesIO
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(win, "Clip", f"Image library not available: {e}"); return
        path, _f = QtWidgets.QFileDialog.getSaveFileName(win, "Save clip", "mode_shape.gif", "GIF (*.gif)")
        if not path:
            return
        was = anim_timer.isActive(); anim_timer.stop()
        st["_anim_geo"] = _anim_geometry(); pts, dirs = st["_anim_geo"]
        m_anim.set_show_sensors(chk_showsen.isChecked())
        m_anim.set_view(st["layout"], np.radians(st["az"]), np.radians(st["el"]), -1)
        scale = sp_ascale.value(); frames = []
        for kf in range(24):
            ph = np.exp(1j * (kf / 24.0 * 2 * np.pi))
            amps = scale * np.real(sh * ph); mags = scale * np.abs(sh)
            n = min(len(amps), len(pts))
            if n == 0:
                break
            m_anim.set_disp(amps[:n].tolist())
            m_anim.set_anim({"pts": pts[:n], "dirs": dirs[:n], "amps": amps[:n],
                             "mags": mags[:n], "mmax": float(scale)})
            QtWidgets.QApplication.processEvents()
            buf = QtCore.QBuffer(); buf.open(QtCore.QIODevice.WriteOnly)
            p_anim.grab().save(buf, "PNG")
            frames.append(Image.open(BytesIO(bytes(buf.data()))).convert("RGB"))
        if frames:
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=60, loop=0)
            QtWidgets.QMessageBox.information(win, "Clip", f"✅ Saved: {path}")
        if was:
            anim_timer.start(45)

    def _anim_geometry():
        """pts (world) y dirs (DOF firmado) de los sensores activos, para la malla."""
        lay = st["layout"]; pts = []; dirs = []
        for mp in lay.active_points():
            d = _AXIS_DIR.get(mp.axis, (0, 0, 1))
            sg = -1.0 if mp.dof.startswith("-") else 1.0
            pts.append([mp.x_norm, 0.20, mp.y_norm]); dirs.append([sg * d[0], sg * d[1], sg * d[2]])
        return np.array(pts, float), np.array(dirs, float)

    def _anim_tick():
        sh = _cur_shape()
        if sh is None:
            return
        st["_anim_phase"] += 0.26
        ph = np.exp(1j * st["_anim_phase"])
        scale = sp_ascale.value()
        amps = scale * np.real(sh * ph)         # posición instantánea (oscila)
        mags = scale * np.abs(sh)               # amplitud/envolvente (color, estable)
        pts, dirs = st.get("_anim_geo", (np.zeros((0, 3)), np.zeros((0, 3))))
        n = min(len(amps), len(pts))
        if n == 0:
            return
        m_anim.set_disp(amps[:n].tolist())
        m_anim.set_anim({"pts": pts[:n], "dirs": dirs[:n], "amps": amps[:n],
                         "mags": mags[:n], "mmax": float(scale)})
    anim_timer.timeout.connect(_anim_tick)

    def _anim_play():
        _anim_reload_modes()
        st["_anim_geo"] = _anim_geometry()
        m_anim.set_show_sensors(chk_showsen.isChecked())
        m_anim.set_view(st["layout"], np.radians(st["az"]), np.radians(st["el"]), -1)
        p_anim.getViewBox().autoRange(padding=0.2)
        anim_timer.start(45)                         # arranca SIEMPRE (aunque el panel falle)
        try:
            _update_modal_panel()
        except Exception:  # noqa: BLE001
            pass
    chk_showsen.toggled.connect(lambda on: m_anim.set_show_sensors(on))

    def _anim_stop():
        anim_timer.stop(); m_anim.set_disp(None); m_anim.set_anim(None)

    btn_play.clicked.connect(_anim_play); btn_stop.clicked.connect(_anim_stop)
    btn_gif.clicked.connect(_save_clip)
    btn_v_iso.clicked.connect(lambda: _set_view(50, 28)); btn_v_top.clicked.connect(lambda: _set_view(0, 88))
    btn_v_side.clicked.connect(lambda: _set_view(90, 10)); btn_v_front.clicked.connect(lambda: _set_view(0, 10))
    cb_amode.currentIndexChanged.connect(lambda *_: _update_modal_panel())
    cb_asrc.currentIndexChanged.connect(lambda *_: (_anim_reload_modes(), _update_modal_panel()))

    def _anim_rotate(dx, dy):
        st["az"] = (st["az"] + dx * 0.4) % 360.0
        st["el"] = float(np.clip(st["el"] - dy * 0.4, 8.0, 88.0))
        m_anim.set_view(st["layout"], np.radians(st["az"]), np.radians(st["el"]), -1)
    p_anim.getViewBox().rotate.connect(_anim_rotate)

    # =====================================================================
    # PRELIMINARY REPORT — entregable de campo (Go/No-Go + resultados + firma)
    # =====================================================================
    pg_prel = QtWidgets.QWidget(); prl = QtWidgets.QVBoxLayout(pg_prel)
    prl.addWidget(QtWidgets.QLabel(
        "<b>Preliminary field report</b> — quick same-day deliverable: data-quality Go/No-Go + "
        "preliminary results + resonance screening + findings. The full report is generated from the web."))
    pf = QtWidgets.QFormLayout()
    e_tech = QtWidgets.QLineEdit(); e_rev = QtWidgets.QLineEdit()
    e_find = QtWidgets.QPlainTextEdit(); e_find.setPlaceholderText("One finding per line…"); e_find.setMaximumHeight(70)
    e_rec = QtWidgets.QPlainTextEdit(); e_rec.setPlaceholderText("One recommendation per line…"); e_rec.setMaximumHeight(70)
    pf.addRow("Technician:", e_tech); pf.addRow("Reviewed by:", e_rev)
    pf.addRow("Findings:", e_find); pf.addRow("Recommendations:", e_rec)
    prl.addLayout(pf)
    prow2 = QtWidgets.QHBoxLayout()
    btn_photos = QtWidgets.QPushButton("🖼 Add photos")
    lbl_photos = QtWidgets.QLabel("0 photos")
    prow2.addWidget(QtWidgets.QLabel("Language:"))
    cb_lang = QtWidgets.QComboBox(); cb_lang.addItems(["Español", "English"]); prow2.addWidget(cb_lang)
    btn_qual = QtWidgets.QPushButton("↻ Compute data quality")
    btn_prel = QtWidgets.QPushButton("📄 Generate preliminary PDF"); btn_prel.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    prow2.addWidget(btn_photos); prow2.addWidget(lbl_photos); prow2.addStretch(1)
    prow2.addWidget(btn_qual); prow2.addWidget(btn_prel)
    prl.addLayout(prow2)
    tbl_qual = QtWidgets.QTableWidget(0, 3); tbl_qual.setHorizontalHeaderLabels(["Check", "Status", "Detail"])
    tbl_qual.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); tbl_qual.verticalHeader().setVisible(False)
    prl.addWidget(tbl_qual, 1)
    lbl_prel = QtWidgets.QLabel(""); lbl_prel.setWordWrap(True); prl.addWidget(lbl_prel)
    tabs.addTab(pg_prel, "Preliminary report")
    st["_photos"] = []

    def _grab_png(widget):
        try:
            pix = widget.grab()
            ba = QtCore.QByteArray(); buf = QtCore.QBuffer(ba); buf.open(QtCore.QIODevice.WriteOnly)
            pix.save(buf, "PNG"); return bytes(ba)
        except Exception:  # noqa: BLE001
            return None

    def _compute_quality():
        lay = st["layout"]; rows = []; fails = 0; warns = 0
        fdd = st.get("oma_fdd"); res = st["acc"].result()
        # OMA
        if "OMA" in lay.test_modes:
            ok = fdd is not None and getattr(fdd, "modes", None)
            rows.append(("OMA captured & modes found", "PASS" if ok else "FAIL",
                         f"{len(fdd.modes)} modes" if ok else "no OMA run"))
            fails += 0 if ok else 1
            nch = lay.n_channels()
            rows.append(("Channels active", "PASS" if nch >= 3 else "WARN", f"{nch} channels"))
            warns += 0 if nch >= 3 else 1
            dur = min(float(lay.duration_s), 60.0)
            long_ok = dur >= 300.0
            rows.append(("OMA duration adequate (ISO/Brincker)", "PASS" if long_ok else "WARN",
                         f"{dur:.0f} s (recomm. ≥ 300 s)"))
            warns += 0 if long_ok else 1
        # EMA
        if "EMA" in lay.test_modes:
            navg = st["acc"].count; tgt = st.get("target", 5)
            rows.append(("EMA averages ≥ target", "PASS" if navg >= tgt else "WARN", f"{navg}/{tgt} averages"))
            warns += 0 if navg >= tgt else 1
            if res is not None:
                band = (res.frequencies_hz >= 5) & (res.frequencies_hz <= lay.fmax_hz)
                mn = float(res.coherence[band].min()) if band.any() else 0.0
                good = mn >= 0.7
                rows.append(("Coherence in band ≥ 0.7", "PASS" if good else "WARN", f"min {mn:.2f}"))
                warns += 0 if good else 1
            else:
                rows.append(("EMA coherence", "FAIL", "no accepted hits")); fails += 1
        if not rows:
            rows.append(("Test type", "WARN", "select EMA/OMA in Machine"))
        tbl_qual.setRowCount(0)
        for chk, sta, det in rows:
            r = tbl_qual.rowCount(); tbl_qual.insertRow(r)
            for cc, v in enumerate([chk, sta, det]):
                it = QtWidgets.QTableWidgetItem(v)
                if cc == 1:
                    col = GREEN if sta == "PASS" else (RED if sta == "FAIL" else AMBER)
                    it.setForeground(QtGui.QBrush(QtGui.QColor(col)))
                tbl_qual.setItem(r, cc, it)
        verdict = ("NO-GO — re-measure" if fails else ("GO with warnings" if warns else "GO — data acceptable"))
        st["_verdict"] = verdict
        lbl_prel.setText(f"Data-quality verdict: {verdict}")
        lbl_prel.setStyleSheet(f"color:{RED if fails else (AMBER if warns else GREEN)};font-weight:800;")
        return rows, verdict
    btn_qual.clicked.connect(_compute_quality)

    def _add_photos():
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(win, "Add photos", "", "Images (*.png *.jpg *.jpeg)")
        for p in paths:
            try:
                with open(p, "rb") as fh:
                    st["_photos"].append(fh.read())
            except Exception:  # noqa: BLE001
                pass
        lbl_photos.setText(f"{len(st['_photos'])} photos")
    btn_photos.clicked.connect(_add_photos)

    def _table_data(tbl, max_rows=40):
        heads = [tbl.horizontalHeaderItem(c).text() if tbl.horizontalHeaderItem(c) else str(c)
                 for c in range(tbl.columnCount())]
        rows = []
        for r in range(min(tbl.rowCount(), max_rows)):
            row = []
            for c in range(tbl.columnCount()):
                it = tbl.item(r, c); w = tbl.cellWidget(r, c)
                if it is not None:
                    row.append(it.text())
                elif hasattr(w, "currentText"):
                    row.append(w.currentText())
                elif hasattr(w, "isChecked"):
                    row.append("✓" if w.isChecked() else "")
                else:
                    row.append("")
            rows.append(row)
        return heads, rows

    def _shape_pngs(top=4, es=True):
        """Renderiza las formas modales de los N modos más relevantes en 3D con color,
        SIN sensores (como ARTeMIS). Devuelve [(caption, png)]."""
        _L = lambda s, e: s if es else e
        out = []
        fdd = st.get("oma_fdd")
        modes = list(getattr(fdd, "modes", []) or [])[:top]
        if not modes:
            return out
        cb_asrc.setCurrentText("OMA (FDD)"); _anim_reload_modes()
        st["_anim_geo"] = _anim_geometry()
        pts, dirs = st["_anim_geo"]
        m_anim.set_show_sensors(False)
        m_anim.set_view(st["layout"], np.radians(st["az"]), np.radians(st["el"]), -1)
        scale = 0.10
        for i, m in enumerate(modes):
            sh = np.asarray(m.mode_shape, complex).ravel()
            mx = np.max(np.abs(sh)) or 1.0; sh = sh / mx
            n = min(len(sh), len(pts))
            if n == 0:
                continue
            amps = scale * np.real(sh[:n]); mags = scale * np.abs(sh[:n])
            m_anim.set_disp(amps.tolist())
            m_anim.set_anim({"pts": pts[:n], "dirs": dirs[:n], "amps": amps, "mags": mags, "mmax": float(scale)})
            p_anim.getViewBox().autoRange(padding=0.2)
            QtWidgets.QApplication.processEvents()
            png = _grab_png(p_anim)
            if png:
                out.append((_L(f"Figura. Forma modal — modo {m.natural_frequency_hz:.2f} Hz "
                               f"(amort. {m.damping_ratio_pct:.2f}%, complejidad {m.complexity_pct:.0f}%). "
                               "Verde = zona que menos se mueve, rojo = la que más.",
                               f"Figure. Mode shape — mode {m.natural_frequency_hz:.2f} Hz "
                               f"(damping {m.damping_ratio_pct:.2f}%, complexity {m.complexity_pct:.0f}%). "
                               "Green = least moving zone, red = most."), png))
        m_anim.set_anim(None); m_anim.set_disp(None); m_anim.set_show_sensors(True)
        return out

    def _auto_analysis(lay, fdd, cross_rows, ema_rows, verdict, es=True):
        L = lambda s, e: s if es else e
        analysis, findings, recs = [], [], []
        if fdd and fdd.modes:
            fns = [m.natural_frequency_hz for m in fdd.modes]
            analysis.append(L(f"Se identificaron {len(fns)} modos naturales por OMA (FDD) entre {min(fns):.1f} y {max(fns):.1f} Hz.",
                              f"{len(fns)} natural modes identified by OMA (FDD) between {min(fns):.1f} and {max(fns):.1f} Hz."))
        if st.get("ssi") and st["ssi"].modes:
            analysis.append(L(f"SSI (dominio del tiempo) confirma {len(st['ssi'].modes)} modos con incertidumbre (±) y amortiguamiento preciso — mayor confiabilidad.",
                              f"SSI (time domain) confirms {len(st['ssi'].modes)} modes with uncertainty (±) and accurate damping — higher confidence."))
        coinc = [c for c in cross_rows if c[4] in ("Coincidence", "Coincidencia")]
        if coinc:
            analysis.append(L(f"Screening de resonancia: {len(coinc)} coincidencia(s) dentro de ±15% de la velocidad de operación (API 684).",
                              f"Resonance screening: {len(coinc)} coincidence(s) within ±15% of operating speed (API 684)."))
            for c in coinc[:5]:
                findings.append(L(f"El modo {c[0]} ({c[1]}) cruza a {c[2]} RPM (margen {c[3]}%) — posible resonancia cerca de la operación.",
                                  f"Mode {c[0]} ({c[1]}) crosses at {c[2]} RPM (margin {c[3]}%) — possible resonance near operation."))
            recs.append(L("Correlacionar amplitud y fase de la vibración vs velocidad en operación (API 684); evaluar rigidización/soporte si las amplitudes son altas.",
                          "Correlate vibration amplitude and phase vs speed in operation (API 684); evaluate stiffening/support if amplitudes are high."))
        else:
            analysis.append(L("Screening de resonancia: sin coincidencias relevantes dentro de ±15% de la velocidad de operación (API 684).",
                              "Resonance screening: no relevant coincidences within ±15% of operating speed (API 684)."))
        if ema_rows:
            analysis.append(L(f"La correlación EMA↔OMA confirma {len(ema_rows)} modo(s) como características dinámicas reales del conjunto (ISO 7626-6).",
                              f"EMA↔OMA correlation confirms {len(ema_rows)} mode(s) as real structural modes (ISO 7626-6)."))
        for m in [m for m in (fdd.modes if fdd else []) if m.complexity_pct > 40][:3]:
            findings.append(L(f"El modo {m.natural_frequency_hz:.2f} Hz tiene complejidad alta ({m.complexity_pct:.0f}%) — verificar (posible armónico/espurio).",
                              f"Mode {m.natural_frequency_hz:.2f} Hz has high complexity ({m.complexity_pct:.0f}%) — verify (possible harmonic/spurious)."))
        if verdict.startswith("NO-GO"):
            recs.insert(0, L("REPETIR la medición antes de retirarse: la calidad de datos es insuficiente.",
                             "REPEAT the measurement before leaving site: data quality is insufficient."))
        if min(float(lay.duration_s), 60.0) < 300.0 and "OMA" in lay.test_modes:
            recs.append(L("Repetir OMA con registro más largo (≥ 5 min) para mejor amortiguamiento/SSI.",
                          "Repeat OMA with a longer record (≥ 5 min) for better damping/SSI."))
        recs.append(L("Generar el reporte de análisis completo desde Watermelon System (web).",
                      "Generate the full analysis report from Watermelon System (web)."))
        return analysis, findings, recs

    def _gen_preliminary():
        _table_to_layout(); lay = st["layout"]; fdd = st.get("oma_fdd")
        es = cb_lang.currentText() == "Español"
        L = lambda s, e: s if es else e
        rows, verdict = _compute_quality()
        # traducir los checks de calidad al idioma elegido
        _QT = {"OMA captured & modes found": "OMA capturado y modos hallados",
               "Channels active": "Canales activos", "OMA duration adequate (ISO/Brincker)": "Duración OMA adecuada (ISO/Brincker)",
               "EMA averages ≥ target": "Promedios EMA ≥ objetivo", "Coherence in band ≥ 0.7": "Coherencia en banda ≥ 0.7",
               "EMA coherence": "Coherencia EMA", "Test type": "Tipo de ensayo"}
        if es:
            rows = [(_QT.get(c, c), s, d) for c, s, d in rows]
            verdict = verdict.replace("GO — data acceptable", "GO — datos aceptables").replace(
                "GO with warnings", "GO con advertencias").replace("NO-GO — re-measure", "NO-GO — remedir")
        try:
            _refresh_campbell()
        except Exception:  # noqa: BLE001
            pass
        FIG = L("Figura.", "Figure.")
        sections = []
        # 1) Configuration — dibujo 3D + summary
        _refresh_summary()
        cfg_figs = []
        gp = _grab_png(vgeo["plot"])
        if gp:
            cfg_figs.append((f"{FIG} " + L("Máquina y ubicación de sensores (3D).", "Machine and sensor layout (3D)."), gp))
        sh, sr = _table_data(tbl_sum)
        sections.append({"title": L("Configuración", "Configuration"), "figures": cfg_figs,
                         "table": {"headers": sh, "rows": sr}})
        # 2) EMA (si hay) + OMA densidad espectral
        res = st["acc"].result()
        if "EMA" in lay.test_modes and res is not None:
            efigs = []
            for w, capt in ((p_frf, L("FRF (movilidad) — H1.", "FRF (mobility) — H1.")),
                            (p_coh, L("Coherencia.", "Coherence.")), (p_nyq, "Nyquist.")):
                png = _grab_png(w)
                if png:
                    efigs.append((f"{FIG} {capt}", png))
            eh, er = _table_data(tbl_modes)
            sections.append({"title": L("EMA — ensayo de impacto", "EMA — impact test"), "figures": efigs,
                             "table": {"headers": eh, "rows": er}})
        if fdd:
            svg = _grab_png(p_svd)
            oh, orr = _table_data(tbl_om)
            sections.append({"title": L("OMA — densidad espectral (FDD)", "OMA — spectral density (FDD)"),
                             "figures": [(f"{FIG} " + L("Valores singulares (FDD).", "Singular values (FDD)."), svg)] if svg else [],
                             "table": {"headers": oh, "rows": orr}})
        # 4) Comparative
        if tbl_cmp.rowCount() > 0:
            ch2, cr2 = _table_data(tbl_cmp)
            sections.append({"title": L("Comparativo — EMA vs OMA", "Comparative — EMA vs OMA"),
                             "table": {"headers": ch2, "rows": cr2}})
        # 5) Campbell — gráfico primero, luego tabla
        cam_png = _grab_png(p_cam)
        chh, crr = _table_data(tbl_cam)
        cross_rows = crr
        sections.append({"title": L("Campbell — screening de resonancia (API 684)", "Campbell — resonance screening (API 684)"),
                         "intro": L("Cruces frecuencia natural ↔ orden con la RPM y el margen de separación.",
                                    "Natural frequency ↔ order crossings vs machine RPM and separation margin."),
                         "figures": [(f"{FIG} " + L("Diagrama de Campbell (API 684).", "Campbell diagram (API 684)."), cam_png)] if cam_png else [],
                         "table": {"headers": chh or ["Mode", "Order", "RPM", "Margin%", "Status"], "rows": crr}})
        # 6) SSI
        if st.get("ssi") and st["ssi"].modes:
            sp = _grab_png(p_stab); shh, srr = _table_data(tbl_ssi)
            sections.append({"title": L("SSI (subespacios) — modos con incertidumbre", "SSI (subspace) — modes with uncertainty"),
                             "intro": L("SSI identifica los modos en el dominio del tiempo: amortiguamiento preciso e INCERTIDUMBRE (±) por modo. Es el resultado más confiable (validación de modelo, API 684).",
                                        "SSI identifies modes in the time domain: accurate damping and UNCERTAINTY (±) per mode. Most reliable result (model validation, API 684)."),
                             "figures": [(f"{FIG} " + L("Diagrama de estabilización (SSI).", "Stabilization diagram (SSI)."), sp)] if sp else [],
                             "table": {"headers": shh, "rows": srr}})
        # 7) Mode shapes — snapshots 3D
        shp = _shape_pngs(top=4, es=es)
        if shp:
            sections.append({"title": L("Formas modales — capturas 3D", "Mode shapes — 3D snapshots"), "figures": shp})

        # correlación EMA-OMA (para el análisis)
        ema_rows = []
        if res is not None and fdd:
            from core.modal.ema_oma_correlation import correlate
            ema = [mp.frequency_hz for mp in modes_from_frf(res, fmin=5, fmax=lay.fmax_hz, exp_tau=st["acc"].exp_tau())]
            oma = [m.natural_frequency_hz for m in fdd.modes]
            for mm in correlate(ema, oma, tol_hz=2.5):
                ema_rows.append([f"{mm.ema_hz:.2f}", f"{mm.oma_hz:.3f}", f"{mm.delta_hz:.3f}"])

        analysis, af, ar = _auto_analysis(lay, fdd, cross_rows, ema_rows, verdict, es=es)
        findings = [ln for ln in e_find.toPlainText().splitlines() if ln.strip()] + af
        recs = [ln for ln in e_rec.toPlainText().splitlines() if ln.strip()] + ar

        logo = None
        for cand in ("assets/watermelon_logo.png",
                     os.path.join(getattr(sys, "_MEIPASS", "."), "assets", "watermelon_logo.png")):
            try:
                with open(cand, "rb") as fh:
                    logo = fh.read(); break
            except Exception:  # noqa: BLE001
                continue
        import datetime as _dt
        meta = {"title": L("Reporte Modal Preliminar", "Preliminary Modal Report"), "asset": lay.name,
                "machine_type": lay.machine_type, "client": lay.client, "location": lay.location,
                "test_type": "/".join(lay.test_modes), "rpm": f"{lay.running_speed_rpm:.0f}",
                "technician": e_tech.text(), "reviewer": e_rev.text(),
                "date": _dt.date.today().isoformat(), "verdict": verdict}
        try:
            from core.modal.preliminary_report import build_preliminary_pdf
            pdf = build_preliminary_pdf(meta=meta, quality=rows, sections=sections, analysis=analysis,
                                        findings=findings, recommendations=recs, lang=("es" if es else "en"),
                                        photos=st.get("_photos", []), run_id=st.get("run_id", ""), logo_png=logo)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(win, "Preliminary", f"Error: {type(e).__name__}: {e}"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(win, "Save preliminary report",
                                                        f"Preliminary_{lay.name}.pdf", "PDF (*.pdf)")
        if path:
            with open(path, "wb") as fh:
                fh.write(pdf)
            QtWidgets.QMessageBox.information(win, "Preliminary", f"✅ Saved: {path}")
    btn_prel.clicked.connect(_gen_preliminary)

    # ---------------------------------------------------------------
    # HELP / MANUAL / ABOUT  (English, with a REAL worked example)
    # ---------------------------------------------------------------
    def _hx_grab(widget, w, h):
        widget.setFixedSize(w, h)
        QtWidgets.QApplication.processEvents()
        return widget.grab()

    def _hx_plot3d(lay, anim=None, az=50.0, el=24.0, w=760, h=340):
        pw = pg.PlotWidget(viewBox=OrbitViewBox()); pw.setBackground("w")
        pw.setAspectLocked(True); pw.hideAxis("left"); pw.hideAxis("bottom"); pw.setMenuEnabled(False)
        it = Machine3DItem(); pw.addItem(it)
        it.set_show_sensors(anim is None)
        if anim is not None:
            it.set_anim(anim); it.set_disp(anim["amps"])
        it.set_view(lay, np.radians(az), np.radians(el), -1)
        pw.getViewBox().autoRange(padding=0.15)
        return _hx_grab(pw, w, h)

    def _hx_add(vbox, title, pixmap, desc):
        t = QtWidgets.QLabel(f"<b style='font-size:13px;color:{NAVY}'>{title}</b>")
        t.setTextFormat(QtCore.Qt.RichText); vbox.addWidget(t)
        img = QtWidgets.QLabel(); img.setPixmap(pixmap)
        img.setStyleSheet("border:1px solid #e2e8f0;border-radius:8px;background:white;padding:4px")
        vbox.addWidget(img)
        d = QtWidgets.QLabel(desc); d.setWordWrap(True); d.setStyleSheet("color:#475569;margin-bottom:12px")
        d.setTextFormat(QtCore.Qt.RichText); vbox.addWidget(d)

    def _gen_help_example():
        if st.get("_help_done"):
            return
        st["_help_done"] = True
        btn_ex.setEnabled(False); btn_ex.setText("Generating…"); QtWidgets.QApplication.processEvents()
        try:
            from core.modal.oma_engine import run_oma
            from core.modal.campbell import compute_crossings, SpeedBand
            from scipy.signal import lfilter
            lay = motor_multistage_pump_layout(name="Cenit — Estación Medellín · U2",
                                               client="Cenit", location="Estación Medellín",
                                               tag="UNIDAD 2 · MPE2420", running_speed_rpm=3600)
            nch = lay.n_channels(); rng = np.random.default_rng(3); fs = 1280.0; N = int(60 * fs)
            data = np.zeros((N, nch))
            for fn0, z0 in ((19.4, 0.031), (38.8, 0.018), (77.4, 0.012), (129.9, 0.010)):
                wn = 2 * np.pi * fn0; wd = wn * (1 - z0 * z0) ** 0.5
                r = np.exp(-z0 * wn / fs); th = wd / fs
                q = lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], rng.standard_normal(N)); q /= (np.std(q) or 1)
                data += np.outer(q, rng.standard_normal(nch))
            data += 0.05 * rng.standard_normal((N, nch))
            fmax = min(fs / 2.56, lay.fmax_hz)
            fdd = run_oma(time_data=data, sample_rate_hz=fs, nperseg=4096,
                          channel_names=lay.channel_names(), f_min_hz=5.0, f_max_hz=fmax)
            # 1) Configuración 3D
            _hx_add(exl, "Step 1 · Configure the machine & sensors",
                    _hx_plot3d(lay),
                    "Build the motor–pump train and place 17 accelerometers (H/V/A per bearing + "
                    "skid feet). Numbers are the BNC channels. This is the field configuration.")
            # 2) FDD singular values
            pw = pg.PlotWidget(); pw.setBackground("w"); pw.setLabel("bottom", "Frequency", "Hz")
            pw.setLabel("left", "dB"); pw.showGrid(x=True, y=True, alpha=0.3); pw.addLegend(offset=(-10, 10))
            freqs = np.asarray(fdd.frequencies_hz); sv = np.asarray(fdd.singular_values); bnd = freqs <= fmax
            _cs = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b"]
            for i in range(min(sv.shape[0], 4)):
                pw.plot(freqs[bnd], 10 * np.log10(np.maximum(sv[i][bnd], 1e-30)),
                        pen=pg.mkPen(_cs[i], width=1.7 if i == 0 else 1.0), name=f"SV{i+1}")
            for m in fdd.modes:
                pw.addItem(pg.InfiniteLine(m.natural_frequency_hz, pen=pg.mkPen("#dc2626", style=QtCore.Qt.DotLine)))
            _hx_add(exl, "Step 2 · OMA capture → FDD singular values",
                    _hx_grab(pw, 760, 320),
                    "With the machine running, capture ~5–10 min and run FDD. Each red line is a "
                    "natural frequency (peak of SV1). Here: <b>19.4, 38.8, 77.4, 129.9 Hz</b>.")
            # 3) Mode shape coloured (first mode)
            m0 = fdd.modes[0]; sh = np.asarray(m0.mode_shape, complex).ravel()
            sh = sh / (np.max(np.abs(sh)) or 1.0)
            pts = np.array([[p.x_norm, 0.20, p.y_norm] for p in lay.active_points()], float)
            dirs = np.array([[(_AXIS_DIR.get(p.axis, (0, 0, 1)))[0], (_AXIS_DIR.get(p.axis, (0, 0, 1)))[1],
                              (_AXIS_DIR.get(p.axis, (0, 0, 1)))[2]] for p in lay.active_points()], float)
            amps = 0.12 * np.real(sh); mags = 0.12 * np.abs(sh); n = min(len(amps), len(pts))
            anim = {"pts": pts[:n], "dirs": dirs[:n], "amps": amps[:n], "mags": mags[:n], "mmax": 0.12}
            _hx_add(exl, f"Step 3 · Mode shape — {m0.natural_frequency_hz:.1f} Hz (colour = amplitude)",
                    _hx_plot3d(lay, anim=anim),
                    "The deformed shape shows how the structure moves at this mode. Green = still, "
                    "red = maximum motion. A weak skid shows large motion at the base/pedestals.")
            # 4) Campbell
            pw2 = pg.PlotWidget(); pw2.setBackground("w"); pw2.setLabel("bottom", "Running speed", "RPM")
            pw2.setLabel("left", "Frequency", "Hz"); pw2.showGrid(x=True, y=True, alpha=0.3)
            rpm_op = 3600.0; rpm_max = rpm_op * 1.4; modes_hz = [m.natural_frequency_hz for m in fdd.modes]
            ymax = max(modes_hz) * 1.3; rr = np.linspace(0, rpm_max, 60)
            for o in (0.5, 1, 2, 3, 4, 5, 6, 7, 8):
                pw2.plot(rr, o * rr / 60.0, pen=pg.mkPen("#6B7280", width=1, style=QtCore.Qt.DotLine))
            for fn0 in modes_hz:
                pw2.plot([0, rpm_max], [fn0, fn0], pen=pg.mkPen(GREEN, width=2))
            reg = pg.LinearRegionItem([rpm_op * 0.85, rpm_op * 1.15], movable=False,
                                      brush=pg.mkBrush(239, 68, 68, 40)); reg.setZValue(-10); pw2.addItem(reg)
            pw2.plot([rpm_op, rpm_op], [0, ymax], pen=pg.mkPen(NAVY, width=3))
            pw2.setXRange(0, rpm_max); pw2.setYRange(0, ymax)
            _hx_add(exl, "Step 4 · Campbell (API 684) — resonance screening",
                    _hx_grab(pw2, 760, 320),
                    "Orders 0.5×–8× (grey) vs natural frequencies (green). The red band is the "
                    "operating speed ±15%. A mode inside the band near an order = resonance risk.")
            lbl_ex_done = QtWidgets.QLabel(
                "<div style='background:#ecfdf5;border-radius:8px;padding:10px;color:#065f46'>"
                "<b>Auto-diagnosis (example):</b> the 19.4 Hz mode shows high motion at the skid feet "
                "with damping ~3% → consistent with <b>low skid rigidity</b>. Recommendation: verify "
                "base/grouting stiffness and separation from 1× (60 Hz).</div>")
            lbl_ex_done.setWordWrap(True); lbl_ex_done.setTextFormat(QtCore.Qt.RichText); exl.addWidget(lbl_ex_done)
        except Exception as e:  # noqa: BLE001
            err = QtWidgets.QLabel(f"Could not generate the example: {type(e).__name__}: {e}")
            err.setWordWrap(True); exl.addWidget(err)
            btn_ex.setEnabled(True); btn_ex.setText("🍉 Generate worked example (Cenit Medellín)")
            return
        btn_ex.setText("✅ Worked example — Cenit Medellín")

    pg_help = QtWidgets.QWidget(); hl = QtWidgets.QVBoxLayout(pg_help)
    about = QtWidgets.QLabel(
        f"<div style='background:{NAVY};color:white;border-radius:8px;padding:12px 16px'>"
        f"<span style='font-size:18px;font-weight:800'>🍉 Watermelon Modal</span>"
        f"<span style='color:#93c5fd;font-weight:700'>&nbsp;&nbsp;v{__version__}</span><br>"
        f"<span style='color:#cbd5e1'>EMA + OMA field analysis · one platform, field to report · "
        f"acquisition: {DAQ_NAME}</span></div>")
    about.setTextFormat(QtCore.Qt.RichText); hl.addWidget(about)

    _hscroll = QtWidgets.QScrollArea(); _hscroll.setWidgetResizable(True)
    _hscroll.setStyleSheet("QScrollArea{border:none;background:transparent}")
    _hinner = QtWidgets.QWidget(); _hin = QtWidgets.QVBoxLayout(_hinner)
    manual = QtWidgets.QLabel(); manual.setWordWrap(True); manual.setTextFormat(QtCore.Qt.RichText)
    manual.setStyleSheet("background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px")
    manual.setText(f"""
    <h2 style='color:{NAVY};margin-top:0'>User manual</h2>
    <p>Watermelon Modal runs <b>EMA</b> (impact) and <b>OMA</b> (operational) modal analysis in the
    field, uploads runs to Watermelon System (cloud) and produces a full normative report.
    Standards: <b>ISO 7626-1..6</b>, <b>ISO 20816</b>, <b>API 684</b>, <b>API 670</b>.</p>
    <h3 style='color:{NAVY}'>1 · Configuration (recommended)</h3>
    <ul>
    <li>Build the machine (motor, coupling, pump, skid, pedestals) — use ⭐ <i>Factory presets</i>.</li>
    <li>Per bearing place a triaxial set: <b>H</b> horizontal, <b>V</b> vertical, <b>A</b> axial.</li>
    <li>For structural problems (weak skid) add sensors on the <b>skid feet</b> to capture base motion.</li>
    <li>Mark 1–2 <b>reference</b> sensors (fixed) — required for OMA.</li>
    <li>Acquisition: OMA needs a <b>long record (5–10 min)</b>, simultaneous sampling, no force window.</li>
    </ul>
    <h3 style='color:{NAVY}'>2 · Take an OMA measurement</h3>
    <ol>
    <li>Connect the {DAQ_NAME} unit before opening the app → banner turns green (LIVE).</li>
    <li>OMA capture → Source = {DAQ_NAME} (live) → <b>Test acquisition</b> → <b>Capture + FDD</b>.</li>
    <li>Run <b>SSI</b> to confirm modes (accurate damping + uncertainty).</li>
    <li>Read the <b>automatic mode validation</b> (validated / doubtful / rejected, harmonics flagged).</li>
    <li>With internet, <b>Upload run</b> → it appears in Watermelon System (web) for the report.</li>
    </ol>
    <h3 style='color:{NAVY}'>3 · Impact test (EMA) — ISO 7626-5</h3>
    <p>Hammer on channel 1; force + exponential windows ON; 3–5 averages; accept only if
    <b>coherence ≥ 0.8</b>. Identify in <b>Modes (EMA)</b> (half-power + Nyquist).</p>
    <h3 style='color:{NAVY}'>4 · Reading results</h3>
    <ul>
    <li><b>FDD singular values:</b> each SV1 peak is a candidate mode.</li>
    <li><b>SSI stabilization:</b> stable green poles across orders = real modes; ± = uncertainty.</li>
    <li><b>Complexity (Argand):</b> collinear vectors = real mode; scattered = suspicious.</li>
    <li><b>Campbell (API 684):</b> a mode within ±15% of an order (0.5×–8×) is a resonance risk.</li>
    </ul>
    <h3 style='color:{NAVY}'>5 · Standards</h3>
    <p>ISO 7626-1..6 · ISO 20816 · API 684 · API 670.</p>
    <hr><h3 style='color:{NAVY}'>About</h3>
    <p><b>Watermelon Modal</b> v{__version__} — part of the <b>Watermelon System</b> platform.<br>
    Developed by <b>SIGA S.A.S.</b> — Machinery Diagnostics Engineering.<br>
    Contact: <a href='mailto:ehernandez@sigasas.com'>ehernandez@sigasas.com</a> · Bogotá — Colombia.<br>
    <span style='color:#64748b'>© 2026 SIGA S.A.S. All rights reserved.</span></p>
    """)
    _hin.addWidget(manual)
    _exhdr = QtWidgets.QLabel(f"<h2 style='color:{NAVY}'>Worked example — Cenit · Estación Medellín</h2>"
                             "<p style='color:#475569'>A complete OMA of a motor + multistage pump, "
                             "step by step, with the real graphs the software produces.</p>")
    _exhdr.setTextFormat(QtCore.Qt.RichText); _exhdr.setWordWrap(True); _hin.addWidget(_exhdr)
    btn_ex = QtWidgets.QPushButton("🍉 Generate worked example (Cenit Medellín)")
    btn_ex.setStyleSheet(f"QPushButton{{background:{GREEN};font-size:14px;padding:10px 18px;}}")
    _hin.addWidget(btn_ex)
    _exw = QtWidgets.QWidget(); exl = QtWidgets.QVBoxLayout(_exw); _hin.addWidget(_exw)
    btn_ex.clicked.connect(_gen_help_example)
    _hin.addStretch(1)
    _hscroll.setWidget(_hinner); hl.addWidget(_hscroll, 1)
    tabs.addTab(pg_help, "Help")

    # Orden lógico de pestañas: EMA (impacto → modos) juntos, OMA (captura → SSI)
    # juntos, luego correlación / Campbell / formas / reporte.
    _desired_order = ["Configuration", "Impact test (EMA)", "Modes (EMA)",
                      "OMA capture", "SSI (subspace)", "Comparative", "Campbell",
                      "Mode shapes", "Preliminary report", "Help"]
    _bar = tabs.tabBar()
    for _target, _title in enumerate(_desired_order):
        _cur = next((i for i in range(tabs.count()) if tabs.tabText(i) == _title), None)
        if _cur is not None and _cur != _target:
            _bar.moveTab(_cur, _target)

    return app, win


def main(argv=None):
    ap = argparse.ArgumentParser(description="Watermelon Modal — EMA + OMA (native)")
    ap.add_argument("--sim", action="store_true", default=True)
    ap.add_argument("--name", default="Motor-Pump train")
    args = ap.parse_args(argv)
    lay = OMALayout(name=args.name, machine_components=default_components())  # geometry only, no sensors
    try:
        app, win = build_app(lay, simulated=True); win.show(); sys.exit(app.exec())
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        try:
            with open("watermelon_modal_error.log", "w", encoding="utf-8") as fh:
                fh.write(err)
            _a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            QtWidgets.QMessageBox.critical(None, "Watermelon Modal — startup error", err[-1500:])
        except Exception:  # noqa: BLE001
            print(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
