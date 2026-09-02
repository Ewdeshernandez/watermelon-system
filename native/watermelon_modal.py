"""
Watermelon Modal — módulo NATIVO de ensayo modal (EMA + OMA, Windows)
=====================================================================

App de campo (PySide6 + pyqtgraph) homóloga a Watermelon Rotordynamics:

  Configuration (pestañas internas)
      · Machine        — ficha del activo del cliente + DIBUJO de la máquina
                         donde se UBICAN los sensores (clic sobre el gráfico).
      · Measurement points — tabla de canales (componente/referencia/DOF/slot/
                         canal/sensibilidad), sirve para EMA y OMA.
      · Acquisition    — fs, block, Fmax, duración, ventanas (EMA), etc.
  Impact test (EMA)    — golpe→FRF+coherencia en vivo, accept/reject, promedios.
  OMA capture          — captura continua multicanal → FDD → modos.
  Modes                — modos EMA (peak-picking + half-power) + Nyquist.
  Comparative          — compara varias condiciones OMA (tracking de modos).
  Campbell             — cruces fn↔orden automáticos (API 684) + bandas.

Modo SIMULADO por defecto (--sim): sin hardware. Con NI 9234 (IEPE) real el mismo
motor recibe los datos. Cómputo en core.modal.* (testeado).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from typing import List, Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except Exception as exc:  # noqa: BLE001
    print("Falta PySide6/pyqtgraph:", exc)
    raise

from core.modal.live_impact import (FRFAccumulator, HitQuality, SynthMode, assess_hit,
                                     modes_from_frf, synth_impact)

# Espécimen DEMO compartido por EMA y OMA (para que el Comparative EMA↔OMA correlacione)
DEMO_MODES = [SynthMode(19.4, 0.020, 1.0), SynthMode(38.8, 0.015, 0.7),
              SynthMode(77.4, 0.012, 0.45), SynthMode(129.9, 0.010, 0.30)]
from core.modal.oma_layout import (OMALayout, MeasPoint, default_24ch_layout,
                                    DEFAULT_COMPONENTS, POSITION_REFS, DOFS)
from core.modal.oma_engine import run_oma
from core.modal.campbell import compute_crossings, SpeedBand

NAVY = "#0F1E3D"; ACC = "#1AAEE5"; GREEN = "#10b981"; AMBER = "#f59e0b"; RED = "#ef4444"
_COMP_COLOR = {"Motor": "#2563eb", "Bomba": "#16a34a", "Skid": "#a16207",
               "Tubería succión": "#0891b2", "Tubería descarga": "#7c3aed"}


def _stylesheet() -> str:
    return f"""
    QWidget {{ font-family: 'Segoe UI', Arial; font-size: 12px; color: {NAVY}; }}
    QMainWindow, QTabWidget::pane {{ background: #f5f8fd; }}
    QTabBar::tab {{ background: #e6ecf5; padding: 8px 16px; margin-right: 3px;
        border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 700; }}
    QTabBar::tab:selected {{ background: {NAVY}; color: white; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: white;
        border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 8px; }}
    QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
    QPushButton {{ background: {NAVY}; color: white; border: none; font-weight: 700;
        padding: 8px 15px; border-radius: 8px; }}
    QPushButton:hover {{ background: #0e3a6b; }}
    QPushButton:disabled {{ background: #94a3b8; }}
    QTableWidget {{ background: white; gridline-color: #e6ecf5; }}
    QHeaderView::section {{ background: {NAVY}; color: white; padding: 5px;
        border: none; font-weight: 700; }}
    """


def _led(color: str, on: bool, label: str) -> str:
    c = color if on else "#cbd5e1"
    return (f"<span style='color:{c};font-size:16px'>●</span> "
            f"<span style='font-weight:700;color:{'#334155' if on else '#94a3b8'}'>{label}</span>")


def build_app(layout: OMALayout, simulated: bool = True):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(_stylesheet())
    pg.setConfigOptions(antialias=True)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Modal — {layout.name}")
    win.resize(1300, 840)

    st = {"layout": layout, "acc": FRFAccumulator(layout.fs_hz, layout.block_size),
          "pending": None, "target": 5, "oma_fdd": None, "conditions": [],
          "rng": np.random.default_rng()}

    tb = win.addToolBar("main"); tb.setMovable(False)
    tb.setStyleSheet(f"QToolBar {{ background: {NAVY}; padding: 6px 12px; }}")
    brand = QtWidgets.QLabel("  🍉 Watermelon Modal")
    brand.setStyleSheet("color:white; font-weight:800; font-size:15px;"); tb.addWidget(brand)
    sp = QtWidgets.QWidget(); sp.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    tb.addWidget(sp)
    mode_lbl = QtWidgets.QLabel(("SIMULATED — no hardware" if simulated else "LIVE — NI 9234") + "   ")
    mode_lbl.setStyleSheet(f"color:{'#fbbf24' if simulated else '#34d399'}; font-weight:700;")
    tb.addWidget(mode_lbl)

    tabs = QtWidgets.QTabWidget(); win.setCentralWidget(tabs)

    # =================================================================
    # CONFIGURATION (pestañas internas)
    # =================================================================
    cfg_outer = QtWidgets.QWidget(); cfg_ol = QtWidgets.QVBoxLayout(cfg_outer)
    cfg_tabs = QtWidgets.QTabWidget(); cfg_ol.addWidget(cfg_tabs, 1)

    # ---------- Machine: ficha del activo + dibujo con ubicación de sensores ----------
    pg_m = QtWidgets.QWidget(); ml = QtWidgets.QVBoxLayout(pg_m)
    ml.addWidget(QtWidgets.QLabel("<b>Client machine</b> — asset record"))
    frm = QtWidgets.QFormLayout()
    e_name = QtWidgets.QLineEdit(layout.name)
    e_type = QtWidgets.QLineEdit(layout.machine_type); e_type.setPlaceholderText("Motor-Bomba centrífuga…")
    e_tag = QtWidgets.QLineEdit(layout.tag); e_tag.setPlaceholderText("tag / placa")
    e_client = QtWidgets.QLineEdit(layout.client); e_client.setPlaceholderText("cliente")
    e_loc = QtWidgets.QLineEdit(layout.location); e_loc.setPlaceholderText("planta / ubicación")
    sp_rpm = QtWidgets.QDoubleSpinBox(); sp_rpm.setRange(0, 60000); sp_rpm.setValue(layout.running_speed_rpm)
    r1 = QtWidgets.QHBoxLayout(); r1.addWidget(e_name, 2); r1.addSpacing(8)
    r1.addWidget(QtWidgets.QLabel("Type:")); r1.addWidget(e_type, 2)
    _w1 = QtWidgets.QWidget(); _w1.setLayout(r1); frm.addRow("Machine:", _w1)
    r2 = QtWidgets.QHBoxLayout(); r2.addWidget(e_tag, 1); r2.addSpacing(8)
    r2.addWidget(QtWidgets.QLabel("Client:")); r2.addWidget(e_client, 1); r2.addSpacing(8)
    r2.addWidget(QtWidgets.QLabel("Location:")); r2.addWidget(e_loc, 1)
    _w2 = QtWidgets.QWidget(); _w2.setLayout(r2); frm.addRow("Tag:", _w2)
    r3 = QtWidgets.QHBoxLayout(); r3.addWidget(QtWidgets.QLabel("Running speed (RPM):")); r3.addWidget(sp_rpm)
    r3.addSpacing(16); r3.addWidget(QtWidgets.QLabel("Ensayo:"))
    chk_ema = QtWidgets.QCheckBox("EMA (impacto)"); chk_oma = QtWidgets.QCheckBox("OMA (operacional)")
    chk_ema.setChecked("EMA" in layout.test_modes); chk_oma.setChecked("OMA" in layout.test_modes or not layout.test_modes)
    r3.addWidget(chk_ema); r3.addWidget(chk_oma); r3.addStretch(1)
    _w3 = QtWidgets.QWidget(); _w3.setLayout(r3); frm.addRow("Operation:", _w3)
    ml.addLayout(frm)

    # --- constructor de la máquina (cajas) + colocación de puntos ---
    from core.modal.oma_layout import COMPONENT_KINDS, MEAS_TYPES, MachineComponent
    bld = QtWidgets.QHBoxLayout()
    bld.addWidget(QtWidgets.QLabel("<b>Equipo:</b>"))
    cb_kind = QtWidgets.QComboBox(); cb_kind.addItems(COMPONENT_KINDS); cb_kind.setMinimumWidth(150)
    btn_addcomp = QtWidgets.QPushButton("➕ Agregar equipo")
    btn_delcomp = QtWidgets.QPushButton("– Quitar equipo")
    bld.addWidget(cb_kind); bld.addWidget(btn_addcomp); bld.addWidget(btn_delcomp)
    bld.addSpacing(18); bld.addWidget(QtWidgets.QLabel("<b>Modo clic:</b>"))
    cb_click = QtWidgets.QComboBox(); cb_click.addItems(["Colocar punto", "Mover sensor", "Mover equipo"])
    bld.addWidget(cb_click)
    btn_tmpl = QtWidgets.QPushButton("Cargar plantilla 24 canales"); bld.addWidget(btn_tmpl)
    bld.addStretch(1)
    ml.addLayout(bld)

    prow = QtWidgets.QHBoxLayout()
    prow.addWidget(QtWidgets.QLabel("<b>Nuevo punto</b> — N°:"))
    sp_num = QtWidgets.QSpinBox(); sp_num.setRange(1, 999); sp_num.setValue(1); prow.addWidget(sp_num)
    prow.addWidget(QtWidgets.QLabel("Ejes:"))
    cbx = QtWidgets.QCheckBox("X"); cby = QtWidgets.QCheckBox("Y"); cbz = QtWidgets.QCheckBox("Z")
    cby.setChecked(True)
    for w in (cbx, cby, cbz): prow.addWidget(w)
    prow.addWidget(QtWidgets.QLabel("Mide:"))
    cb_mtype = QtWidgets.QComboBox(); cb_mtype.addItems(MEAS_TYPES); prow.addWidget(cb_mtype)
    prow.addWidget(QtWidgets.QLabel("→ <i>clic en el dibujo para colocarlo</i>"))
    prow.addStretch(1)
    prow.addWidget(QtWidgets.QLabel("Mover sensor:"))
    cb_place = QtWidgets.QComboBox(); cb_place.setMinimumWidth(150); prow.addWidget(cb_place)
    ml.addLayout(prow)
    p_train = pg.PlotWidget(); p_train.setBackground("w"); p_train.setAspectLocked(True)
    p_train.hideAxis("left"); p_train.hideAxis("bottom"); p_train.setMenuEnabled(False)
    p_train.getViewBox().setMouseEnabled(x=False, y=False)
    ml.addWidget(p_train, 1)
    cfg_tabs.addTab(pg_m, "Machine")

    # ---------- Measurement points ----------
    pg_pts = QtWidgets.QWidget(); pl = QtWidgets.QVBoxLayout(pg_pts)
    prow2 = QtWidgets.QHBoxLayout()
    btn_addp = QtWidgets.QPushButton("+ Punto"); btn_delp = QtWidgets.QPushButton("– Quitar")
    btn_val = QtWidgets.QPushButton("✓ Aplicar y validar"); btn_val.setStyleSheet(
        f"QPushButton{{background:{GREEN};}} QPushButton:hover{{background:#0e9f6e;}}")
    prow2.addWidget(btn_addp); prow2.addWidget(btn_delp); prow2.addStretch(1); prow2.addWidget(btn_val)
    pl.addLayout(prow2)
    tbl_pts = QtWidgets.QTableWidget(0, 11)
    tbl_pts.setHorizontalHeaderLabels(["#", "N°", "Componente", "Referencia", "DOF", "Mide",
                                       "Slot", "Canal", "Sens (mV/g)", "Ref", "Activo"])
    tbl_pts.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tbl_pts.verticalHeader().setVisible(False)
    pl.addWidget(tbl_pts, 1)
    lbl_val = QtWidgets.QLabel(""); pl.addWidget(lbl_val)
    cfg_tabs.addTab(pg_pts, "Measurement points")

    # ---------- Acquisition ----------
    pg_acq = QtWidgets.QWidget(); al = QtWidgets.QFormLayout(pg_acq)
    sp_fs = QtWidgets.QSpinBox(); sp_fs.setRange(256, 51200); sp_fs.setValue(int(layout.fs_hz)); sp_fs.setSingleStep(256)
    cb_blk = QtWidgets.QComboBox(); cb_blk.addItems(["1024", "2048", "4096", "8192", "16384"]); cb_blk.setCurrentText(str(layout.block_size))
    sp_fmax = QtWidgets.QDoubleSpinBox(); sp_fmax.setRange(10, 25600); sp_fmax.setValue(layout.fmax_hz)
    sp_dur = QtWidgets.QSpinBox(); sp_dur.setRange(10, 900); sp_dur.setValue(int(layout.duration_s))
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
    cfg_tabs.addTab(pg_acq, "Acquisition")

    tabs.addTab(cfg_outer, "Configuration")

    # ---- helpers de configuración ----
    def _mk_combo(items, cur):
        c = QtWidgets.QComboBox(); c.addItems(items)
        if cur in items:
            c.setCurrentText(cur)
        return c

    def _add_point_row(mp: MeasPoint):
        from core.modal.oma_layout import MEAS_TYPES
        r = tbl_pts.rowCount(); tbl_pts.insertRow(r)
        tbl_pts.setItem(r, 0, QtWidgets.QTableWidgetItem(str(mp.idx)))
        tbl_pts.setItem(r, 1, QtWidgets.QTableWidgetItem(str(mp.number or mp.idx)))
        tbl_pts.setCellWidget(r, 2, _mk_combo(DEFAULT_COMPONENTS, mp.component))
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
        cb_place.addItems([f"{p.idx:02d} · {p.label}" for p in st["layout"].points])
        if 0 <= cur < cb_place.count():
            cb_place.setCurrentIndex(cur)
        cb_place.blockSignals(False)

    def _table_to_layout():
        lay = st["layout"]
        pts = []
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
        lay.points = pts
        lay.name = e_name.text() or "Modal"; lay.machine_type = e_type.text(); lay.tag = e_tag.text()
        lay.client = e_client.text(); lay.location = e_loc.text()
        lay.running_speed_rpm = sp_rpm.value()
        lay.test_modes = [m for m, c in (("EMA", chk_ema), ("OMA", chk_oma)) if c.isChecked()] or ["OMA"]
        lay.test_type = lay.test_modes[0]
        lay.fs_hz = float(sp_fs.value()); lay.block_size = int(cb_blk.currentText())
        lay.fmax_hz = float(sp_fmax.value()); lay.duration_s = float(sp_dur.value())

    def _comp_color(kind):
        if kind.startswith("Tubería"):
            return _COMP_COLOR["Tubería succión"] if "succ" in kind else _COMP_COLOR["Tubería descarga"]
        return _COMP_COLOR.get(kind, "#475569")

    def _draw_train():
        p_train.clear()
        lay = st["layout"]
        # skid base
        p_train.plot([0.0, 1.0, 1.0, 0.0, 0.0], [-0.62, -0.62, -0.48, -0.48, -0.62],
                     pen=pg.mkPen(_COMP_COLOR["Skid"], width=2))
        ts = pg.TextItem("SKID", color=_COMP_COLOR["Skid"], anchor=(0, 0.5)); ts.setPos(0.02, -0.55); p_train.addItem(ts)
        # equipos (cajas)
        for c in lay.machine_components:
            col = _comp_color(c.kind)
            p_train.plot([c.x0, c.x1, c.x1, c.x0, c.x0], [c.y0, c.y0, c.y1, c.y1, c.y0],
                         pen=pg.mkPen(col, width=2))
            t = pg.TextItem(c.display(), color=col, anchor=(0.5, 0.5))
            t.setPos((c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2); p_train.addItem(t)
        # sensores (puntos) rotulados por su código 1XA…
        sel = cb_place.currentIndex()
        for i, mp in enumerate(lay.points):
            if not mp.active:
                continue
            col = _COMP_COLOR.get(mp.component, "#475569")
            sz = 18 if i == sel else (15 if mp.reference_sensor else 11)
            sym = "star" if mp.reference_sensor else "o"
            pen = pg.mkPen(RED, width=2.5) if i == sel else pg.mkPen("w", width=1.5)
            p_train.addItem(pg.ScatterPlotItem([mp.x_norm], [mp.y_norm], size=sz, symbol=sym,
                            brush=pg.mkBrush(col), pen=pen))
            t = pg.TextItem(mp.code, color=NAVY, anchor=(0.5, 1.4))
            t.setPos(mp.x_norm, mp.y_norm); t.setScale(0.9); p_train.addItem(t)
        p_train.getViewBox().autoRange(padding=0.12)

    def _next_channel():
        used = {(p.module_slot, p.channel_index) for p in st["layout"].points}
        for slot in range(1, 9):
            for ch in range(4):
                if (slot, ch) not in used:
                    return slot, ch
        return 8, 3

    def _on_train_click(ev):
        try:
            if ev.button() != QtCore.Qt.LeftButton:
                return
            pt = p_train.getViewBox().mapSceneToView(ev.scenePos())
            x, y = float(pt.x()), float(pt.y())
            lay = st["layout"]; mode = cb_click.currentText()
            if mode == "Colocar punto":
                axes = [a for a, cb in (("X", cbx), ("Y", cby), ("Z", cbz)) if cb.isChecked()] or ["Y"]
                comp = _comp_at(x, y)
                offs = {"X": 0.06, "Y": 0.0, "Z": -0.06}
                for a in axes:
                    slot, ch = _next_channel()
                    lay.points.append(MeasPoint(
                        idx=len(lay.points) + 1, component=comp, position_ref="Centro",
                        dof="+" + a, module_slot=slot, channel_index=ch,
                        number=sp_num.value(), meas_type=cb_mtype.currentText(),
                        x_norm=x, y_norm=y + offs.get(a, 0.0)))
                sp_num.setValue(sp_num.value() + 1)
                _fill_points()
            elif mode == "Mover sensor":
                i = cb_place.currentIndex()
                if 0 <= i < len(lay.points):
                    lay.points[i].x_norm = x; lay.points[i].y_norm = y
            elif mode == "Mover equipo":
                if lay.machine_components:
                    c = min(lay.machine_components,
                            key=lambda k: abs((k.x0 + k.x1) / 2 - x) + abs((k.y0 + k.y1) / 2 - y))
                    w = (c.x1 - c.x0); h = (c.y1 - c.y0)
                    c.x0 = x - w / 2; c.x1 = x + w / 2; c.y0 = y - h / 2; c.y1 = y + h / 2
            _draw_train()
        except Exception:  # noqa: BLE001
            pass
    p_train.scene().sigMouseClicked.connect(_on_train_click)

    def _comp_at(x, y):
        """Nombre del equipo cuya caja contiene (x,y); si ninguno, el más cercano."""
        lay = st["layout"]
        for c in lay.machine_components:
            if c.x0 <= x <= c.x1 and c.y0 <= y <= c.y1:
                return c.kind
        if lay.machine_components:
            c = min(lay.machine_components, key=lambda k: abs((k.x0 + k.x1) / 2 - x))
            return c.kind
        return "Motor"

    def _add_component():
        lay = st["layout"]; kind = cb_kind.currentText()
        n = len([c for c in lay.machine_components if not c.kind.startswith("Tubería")])
        if kind.startswith("Tubería"):
            y0, y1 = (-0.16, -0.06) if "succ" in kind else (0.06, 0.44)
            x0 = 0.55 + 0.02 * len(lay.machine_components)
            lay.machine_components.append(MachineComponent(kind, kind.replace("Tubería", "Tub."), x0, x0 + 0.30, y0, y1))
        else:
            x0 = 0.03 + 0.26 * n
            lay.machine_components.append(MachineComponent(kind, kind, x0, x0 + 0.20, -0.30, 0.30))
        _draw_train()

    def _del_component():
        lay = st["layout"]
        if lay.machine_components:
            lay.machine_components.pop()
            _draw_train()
    btn_addcomp.clicked.connect(_add_component); btn_delcomp.clicked.connect(_del_component)

    def _validate():
        _table_to_layout()
        errs = st["layout"].validate()
        _sync_place_combo(); _draw_train()
        if errs:
            lbl_val.setText("⚠ " + " · ".join(errs[:3])); lbl_val.setStyleSheet(f"color:{RED};font-weight:700;")
        else:
            lbl_val.setText(f"✅ {st['layout'].n_channels()} canales · "
                            f"{len(st['layout'].references())} referencia(s) · válido")
            lbl_val.setStyleSheet(f"color:{GREEN};font-weight:700;")

    def _load_tmpl():
        lay = default_24ch_layout(name=e_name.text() or "Tren Motor-Bomba")
        lay.machine_type = e_type.text(); lay.tag = e_tag.text(); lay.client = e_client.text()
        lay.location = e_loc.text()
        lay.test_modes = [m for m, c in (("EMA", chk_ema), ("OMA", chk_oma)) if c.isChecked()] or ["OMA"]
        st["layout"] = lay
        _fill_points(); _validate()

    def _add_point():
        n = tbl_pts.rowCount()
        _add_point_row(MeasPoint(idx=n + 1, component="Motor", position_ref=POSITION_REFS[0], dof="+Y",
                                 module_slot=n // 4 + 1, channel_index=n % 4, x_norm=0.5, y_norm=0.0))

    def _del_point():
        r = tbl_pts.currentRow()
        if r >= 0:
            tbl_pts.removeRow(r)

    def _upd_df(*_):
        fs = float(sp_fs.value()); blk = int(cb_blk.currentText())
        lbl_df.setText(f"Δf = {fs/blk:.3f} Hz · record {blk/fs*1000:.0f} ms · líneas a Fmax "
                       f"{int(sp_fmax.value()/(fs/blk))}")

    btn_tmpl.clicked.connect(_load_tmpl); btn_addp.clicked.connect(_add_point)
    btn_delp.clicked.connect(_del_point); btn_val.clicked.connect(_validate)
    cb_place.currentIndexChanged.connect(lambda *_: _draw_train())
    sp_fs.valueChanged.connect(_upd_df); cb_blk.currentTextChanged.connect(_upd_df); sp_fmax.valueChanged.connect(_upd_df)
    _fill_points(); _validate(); _upd_df()

    # =================================================================
    # IMPACT TEST (EMA)
    # =================================================================
    pg_imp = QtWidgets.QWidget(); il = QtWidgets.QVBoxLayout(pg_imp)
    ctl = QtWidgets.QHBoxLayout()
    btn_hit = QtWidgets.QPushButton("🔨 Impact"); btn_hit.setStyleSheet(
        f"QPushButton{{background:{ACC};font-size:14px;padding:10px 20px;}} QPushButton:hover{{background:#1490c2;}}")
    btn_badhit = QtWidgets.QPushButton("⚠ Impact w/ fault")
    btn_acc = QtWidgets.QPushButton("✓ Accept"); btn_acc.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    btn_rej = QtWidgets.QPushButton("✗ Reject"); btn_rej.setStyleSheet(f"QPushButton{{background:{RED};}}")
    btn_acc.setEnabled(False); btn_rej.setEnabled(False)
    btn_rst = QtWidgets.QPushButton("↻ Reset")
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
    p_time = pg.PlotWidget(); p_time.setBackground("w"); p_time.setTitle("Last hit — force & response", color=NAVY)
    p_time.showGrid(x=True, y=True, alpha=0.3)
    grid.addWidget(p_frf, 0, 0); grid.addWidget(p_coh, 0, 1); grid.addWidget(p_time, 1, 0, 1, 2)
    grid.setRowStretch(0, 3); grid.setRowStretch(1, 2)
    il.addLayout(grid, 1)
    tabs.addTab(pg_imp, "Impact test (EMA)")
    cur_frf = p_frf.plot([], [], pen=pg.mkPen("#94a3b8", width=1, style=QtCore.Qt.DashLine))
    avg_frf = p_frf.plot([], [], pen=pg.mkPen(ACC, width=2))
    coh_curve = p_coh.plot([], [], pen=pg.mkPen(GREEN, width=2))
    force_curve = p_time.plot([], [], pen=pg.mkPen(RED, width=1.5))
    resp_curve = p_time.plot([], [], pen=pg.mkPen(NAVY, width=1))

    def _leds(q):
        lbl_leds.setText(_led(RED, bool(q and q.overload), "Overload") + " &nbsp;&nbsp; "
                         + _led(AMBER, bool(q and q.double_hit), "Double-hit"))

    def _refresh_impact():
        acc = st["acc"]
        lbl_avg.setText(f"Averages: {acc.count} / {st['target']}"
                        + ("   ✅ target" if acc.count >= st["target"] else ""))
        res = acc.result()
        if res is not None:
            m = res.frequencies_hz <= st["layout"].fmax_hz
            avg_frf.setData(res.frequencies_hz[m], np.maximum(res.magnitude[m], 1e-9))
            coh_curve.setData(res.frequencies_hz[m], res.coherence[m])
        else:
            avg_frf.setData([], []); coh_curve.setData([], [])

    def _do_hit(fault):
        lay = st["layout"]
        dbl = fault and st["rng"].random() < 0.5; over = fault and not dbl
        f, y = synth_impact(lay.fs_hz, lay.block_size, modes=DEMO_MODES, rng=st["rng"],
                            double_hit=dbl, overload=over)
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
        st["acc"] = FRFAccumulator(st["layout"].fs_hz, st["layout"].block_size)
        st["pending"] = None
        for c in (cur_frf, avg_frf, coh_curve, force_curve, resp_curve): c.setData([], [])
        btn_acc.setEnabled(False); btn_rej.setEnabled(False); _leds(None); _refresh_impact()

    btn_hit.clicked.connect(lambda: _do_hit(False)); btn_badhit.clicked.connect(lambda: _do_hit(True))
    btn_acc.clicked.connect(_accept); btn_rej.clicked.connect(_reject); btn_rst.clicked.connect(_reset_ema)
    _leds(None); _refresh_impact()

    # =================================================================
    # OMA CAPTURE
    # =================================================================
    pg_oc = QtWidgets.QWidget(); cl2 = QtWidgets.QVBoxLayout(pg_oc)
    crow = QtWidgets.QHBoxLayout()
    btn_ocap = QtWidgets.QPushButton("▶ Capturar (simulado) + FDD"); btn_ocap.setStyleSheet(
        f"QPushButton{{background:{ACC};font-size:14px;padding:10px 20px;}} QPushButton:hover{{background:#1490c2;}}")
    crow.addWidget(btn_ocap); crow.addStretch(1)
    cl2.addLayout(crow)
    ocs = QtWidgets.QHBoxLayout()
    tbl_om = QtWidgets.QTableWidget(0, 4)
    tbl_om.setHorizontalHeaderLabels(["Freq (Hz)", "Damping (%)", "Complexity (%)", "Clase"])
    tbl_om.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); tbl_om.verticalHeader().setVisible(False)
    tbl_om.setMaximumWidth(520)
    p_svd = pg.PlotWidget(); p_svd.setBackground("w"); p_svd.setLabel("left", "dB"); p_svd.setLabel("bottom", "Frequency", "Hz")
    p_svd.setTitle("Valores singulares (FDD)", color=NAVY); p_svd.showGrid(x=True, y=True, alpha=0.3)
    svd_curve = p_svd.plot([], [], pen=pg.mkPen(NAVY, width=1.6))
    ocs.addWidget(tbl_om, 2); ocs.addWidget(p_svd, 3); cl2.addLayout(ocs, 1)
    lbl_ost = QtWidgets.QLabel(""); cl2.addWidget(lbl_ost)
    tabs.addTab(pg_oc, "OMA capture")

    def _oma_capture():
        from scipy.signal import lfilter
        _validate(); lay = st["layout"]; fs = lay.fs_hz; nch = lay.n_channels()
        if nch < 2:
            QtWidgets.QMessageBox.information(win, "OMA", "Configurá ≥2 canales activos."); return
        secs = min(float(lay.duration_s), 60.0); N = int(secs * fs); rng = st["rng"]
        lbl_ost.setText(f"Capturando {secs:.0f}s @ {fs:.0f}Hz · {nch} canales…"); QtWidgets.QApplication.processEvents()
        data = np.zeros((N, nch))
        for sm in DEMO_MODES:
            fn, z = sm.fn_hz, sm.zeta
            wn = 2 * np.pi * fn; wd = wn * (1 - z * z) ** 0.5; r = np.exp(-z * wn / fs); th = wd / fs
            q = lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], rng.standard_normal(N)); q /= (np.std(q) or 1)
            data += np.outer(q, rng.standard_normal(nch))
        data += 0.05 * rng.standard_normal((N, nch))
        fmax = min(fs / 2.56, lay.fmax_hz)
        fdd = run_oma(data, fs, nperseg=4096, f_min_hz=5.0, f_max_hz=fmax, channel_names=lay.channel_names())
        st["oma_fdd"] = fdd
        freqs = fdd.frequencies_hz; sv = np.asarray(fdd.singular_values)
        if sv.ndim == 1: sv = sv[None, :]
        band = freqs <= fmax
        svd_curve.setData(freqs[band], 10 * np.log10(np.maximum(sv[0][band], 1e-30)))
        tbl_om.setRowCount(0)
        for m in fdd.modes:
            rr = tbl_om.rowCount(); tbl_om.insertRow(rr)
            for c, v in enumerate([f"{m.natural_frequency_hz:.2f}", f"{m.damping_ratio_pct:.3f}",
                                   f"{m.complexity_pct:.1f}", m.classification]):
                tbl_om.setItem(rr, c, QtWidgets.QTableWidgetItem(v))
        lbl_ost.setText(f"✅ FDD listo — {len(fdd.modes)} modos. Mirá Comparative (EMA vs OMA) y Campbell.")
        lbl_ost.setStyleSheet(f"color:{GREEN};font-weight:700;")
        _refresh_campbell(); _refresh_comparative()
    btn_ocap.clicked.connect(_oma_capture)

    # =================================================================
    # MODES (EMA)
    # =================================================================
    pg_mod = QtWidgets.QWidget(); ql = QtWidgets.QVBoxLayout(pg_mod)
    mrow = QtWidgets.QHBoxLayout()
    btn_ident = QtWidgets.QPushButton("🎯 Identify modes (EMA)"); btn_ident.setStyleSheet(f"QPushButton{{background:{ACC};}}")
    mrow.addWidget(btn_ident); mrow.addWidget(QtWidgets.QLabel("Peak-picking + half-power (ISO 7626-6).")); mrow.addStretch(1)
    ql.addLayout(mrow)
    ms = QtWidgets.QHBoxLayout()
    tbl_modes = QtWidgets.QTableWidget(0, 4); tbl_modes.setHorizontalHeaderLabels(["Freq (Hz)", "Damping (%)", "Coherence", "Reliable"])
    tbl_modes.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); tbl_modes.verticalHeader().setVisible(False)
    tbl_modes.setMaximumWidth(520)
    p_nyq = pg.PlotWidget(); p_nyq.setBackground("w"); p_nyq.setAspectLocked(True); p_nyq.setTitle("Nyquist", color=NAVY)
    p_nyq.showGrid(x=True, y=True, alpha=0.3); nyq_curve = p_nyq.plot([], [], pen=pg.mkPen(NAVY, width=1.5))
    ms.addWidget(tbl_modes, 2); ms.addWidget(p_nyq, 3); ql.addLayout(ms, 1)
    tabs.addTab(pg_mod, "Modes")

    def _identify():
        res = st["acc"].result()
        if res is None:
            QtWidgets.QMessageBox.information(win, "Modes", "Aceptá golpes en Impact test."); return
        modes = modes_from_frf(res, fmin=5.0, fmax=st["layout"].fmax_hz); tbl_modes.setRowCount(0)
        for mo in modes:
            r = tbl_modes.rowCount(); tbl_modes.insertRow(r)
            for c, v in enumerate([f"{mo.frequency_hz:.1f}", f"{mo.damping_ratio_pct:.2f}",
                                   (f"{mo.coherence_at_peak:.2f}" if mo.coherence_at_peak is not None else "—"),
                                   ("✓" if mo.is_reliable else "✗")]):
                tbl_modes.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        m = (res.frequencies_hz <= st["layout"].fmax_hz) & (res.frequencies_hz >= 5.0)
        nyq_curve.setData(res.frf_complex[m].real, res.frf_complex[m].imag)
    btn_ident.clicked.connect(_identify)

    # =================================================================
    # COMPARATIVE (multi-condición OMA)
    # =================================================================
    pg_cmp = QtWidgets.QWidget(); cpl = QtWidgets.QVBoxLayout(pg_cmp)
    cpl.addWidget(QtWidgets.QLabel(
        "<b>Correlación EMA ↔ OMA</b> — compara los modos del <b>ensayo de impacto (EMA)</b> con los "
        "modos <b>operacionales (OMA)</b>. La correspondencia confirma que son características dinámicas "
        "reales del conjunto (ISO 7626-6 / API 684)."))
    crow3 = QtWidgets.QHBoxLayout()
    btn_cmp = QtWidgets.QPushButton("↻ Comparar EMA vs OMA"); btn_cmp.setStyleSheet(f"QPushButton{{background:{ACC};}}")
    crow3.addWidget(btn_cmp); crow3.addStretch(1); cpl.addLayout(crow3)
    tbl_cmp = QtWidgets.QTableWidget(0, 4)
    tbl_cmp.setHorizontalHeaderLabels(["Modo EMA (Hz)", "Modo OMA (Hz)", "Δf (Hz)", "Δ (%)"])
    tbl_cmp.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch); tbl_cmp.verticalHeader().setVisible(False)
    cpl.addWidget(tbl_cmp, 1)
    lbl_cmp = QtWidgets.QLabel(""); lbl_cmp.setWordWrap(True); cpl.addWidget(lbl_cmp)
    tabs.addTab(pg_cmp, "Comparative")

    def _refresh_comparative():
        from core.modal.ema_oma_correlation import correlate, summarize as _corr_sum
        res = st["acc"].result()
        ema = [mp.frequency_hz for mp in modes_from_frf(res, fmin=5, fmax=st["layout"].fmax_hz)] if res else []
        oma = [m.natural_frequency_hz for m in st["oma_fdd"].modes] if st["oma_fdd"] else []
        tbl_cmp.setRowCount(0)
        if not ema or not oma:
            miss = []
            if not ema: miss.append("EMA (aceptá golpes en Impact test)")
            if not oma: miss.append("OMA (capturá en OMA capture)")
            lbl_cmp.setText("Falta: " + " y ".join(miss) + "."); return
        matches = correlate(ema, oma, tol_hz=2.5)
        for m in matches:
            r = tbl_cmp.rowCount(); tbl_cmp.insertRow(r)
            for c, v in enumerate([f"{m.ema_hz:.2f}", f"{m.oma_hz:.3f}", f"{m.delta_hz:.3f}", f"{m.delta_pct:.2f}"]):
                tbl_cmp.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        lbl_cmp.setText(_corr_sum(matches))
    btn_cmp.clicked.connect(_refresh_comparative)
    _refresh_comparative()

    # =================================================================
    # CAMPBELL (API 684, nativo)
    # =================================================================
    pg_cam = QtWidgets.QWidget(); cml = QtWidgets.QVBoxLayout(pg_cam)
    crow2 = QtWidgets.QHBoxLayout()
    btn_refc = QtWidgets.QPushButton("↻ Recalcular Campbell"); btn_refc.setStyleSheet(f"QPushButton{{background:{ACC};}}")
    crow2.addWidget(btn_refc)
    crow2.addWidget(QtWidgets.QLabel("Cruces fn↔orden automáticos (½×..4×) + bandas de operación (API 684).")); crow2.addStretch(1)
    cml.addLayout(crow2)
    cams = QtWidgets.QHBoxLayout()
    p_cam = pg.PlotWidget(); p_cam.setBackground("w"); p_cam.setLabel("left", "Frequency", "Hz")
    p_cam.setLabel("bottom", "Speed", "RPM"); p_cam.setTitle("Diagrama de Campbell", color=NAVY)
    p_cam.showGrid(x=True, y=True, alpha=0.3)
    tbl_cam = QtWidgets.QTableWidget(0, 5); tbl_cam.setMaximumWidth(520); tbl_cam.verticalHeader().setVisible(False)
    tbl_cam.setHorizontalHeaderLabels(["Modo", "Orden", "RPM", "Margen%", "Estado"])
    tbl_cam.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    cams.addWidget(tbl_cam, 2); cams.addWidget(p_cam, 3); cml.addLayout(cams, 1)
    lbl_cam = QtWidgets.QLabel(""); lbl_cam.setWordWrap(True); cml.addWidget(lbl_cam)
    tabs.addTab(pg_cam, "Campbell")

    def _current_modes():
        if st["oma_fdd"] is not None and st["oma_fdd"].modes:
            return [m.natural_frequency_hz for m in st["oma_fdd"].modes]
        res = st["acc"].result()
        if res is not None:
            return [mp.frequency_hz for mp in modes_from_frf(res, fmin=5, fmax=st["layout"].fmax_hz)]
        return []

    def _refresh_campbell():
        p_cam.clear(); modes = _current_modes()
        rpm_op = st["layout"].running_speed_rpm or 1185.0
        SM = 0.15                                        # margen de separación API 684 (±15%)
        lo, hi = rpm_op * (1 - SM), rpm_op * (1 + SM)
        rpm_max = max(rpm_op * 1.4, 1500.0)
        if not modes:
            lbl_cam.setText("Sin modos aún — capturá OMA o identificá EMA."); tbl_cam.setRowCount(0); return
        ymax = max(modes) * 1.25
        orders = (0.5, 1.0, 2.0, 3.0, 4.0); rpm = np.linspace(0, rpm_max, 60)
        # zona de MARGEN DE SEPARACIÓN API 684 alrededor de la RPM de la máquina (keep-clear, roja)
        reg = pg.LinearRegionItem([max(0, lo), min(rpm_max, hi)], movable=False,
                                  brush=pg.mkBrush(239, 68, 68, 32)); reg.setZValue(-20); p_cam.addItem(reg)
        # media velocidad (referencia, ámbar)
        reg2 = pg.LinearRegionItem([rpm_op / 2 * (1 - SM), rpm_op / 2 * (1 + SM)], movable=False,
                                   brush=pg.mkBrush(245, 158, 11, 26)); reg2.setZValue(-20); p_cam.addItem(reg2)
        for o in orders:                                # líneas de orden
            p_cam.plot(rpm, o * rpm / 60.0, pen=pg.mkPen("#6B7280", width=1, style=QtCore.Qt.DotLine))
            t = pg.TextItem(f"{o:g}×", color="#6B7280", anchor=(0, 0.5)); t.setPos(rpm_max * 0.98, o * rpm_max / 60.0)
            p_cam.addItem(t)
        for fn in modes:                                # modos (horizontales)
            p_cam.plot([0, rpm_max], [fn, fn], pen=pg.mkPen(GREEN, width=2))
        bands = [SpeedBand(rpm_op, SM * rpm_op, f"Operación {rpm_op:.0f}±{SM*100:.0f}%"),
                 SpeedBand(rpm_op / 2, SM * rpm_op / 2, "½ velocidad")]
        cx = compute_crossings(modes, 0, rpm_max, orders=orders, bands=bands)
        sevcol = {"coincidence": RED, "near": AMBER, "clear": "#94a3b8"}
        tbl_cam.setRowCount(0)
        for c in cx:
            p_cam.addItem(pg.ScatterPlotItem([c.crossing_rpm], [c.mode_hz], size=12, symbol="x",
                          pen=pg.mkPen(sevcol[c.severity], width=2), brush=pg.mkBrush(sevcol[c.severity])))
            if c.severity in ("coincidence", "near"):
                r = tbl_cam.rowCount(); tbl_cam.insertRow(r)
                for j, v in enumerate([f"{c.mode_hz:.2f}", f"{c.order:g}×", f"{c.crossing_rpm:.0f}",
                                       f"{c.sep_margin_pct:.1f}", {"coincidence": "Coincidencia", "near": "Cercano"}[c.severity]]):
                    it = QtWidgets.QTableWidgetItem(v)
                    if j == 4: it.setForeground(QtGui.QBrush(QtGui.QColor(sevcol[c.severity])))
                    tbl_cam.setItem(r, j, it)
        # RPM de la MÁQUINA — línea prominente + etiqueta clara
        p_cam.plot([rpm_op, rpm_op], [0, ymax], pen=pg.mkPen(NAVY, width=3))
        t_op = pg.TextItem(f"N máquina\n{rpm_op:.0f} RPM", color=NAVY, anchor=(0.5, 1.0))
        t_op.setPos(rpm_op, ymax * 0.995); p_cam.addItem(t_op)
        # límites del margen de separación API 684 + su valor "al lado"
        for xb, lab in ((lo, f"−{SM*100:.0f}%\n{lo:.0f}"), (hi, f"+{SM*100:.0f}%\n{hi:.0f}")):
            p_cam.plot([xb, xb], [0, ymax], pen=pg.mkPen(RED, width=1, style=QtCore.Qt.DashLine))
            tt = pg.TextItem(lab, color=RED, anchor=(0.5, 1.0)); tt.setPos(xb, ymax * 0.82); p_cam.addItem(tt)
        p_cam.setLabel("bottom", f"Velocidad · N máquina {rpm_op:.0f} RPM · margen API 684 ±{SM*100:.0f}%", "RPM")
        p_cam.setXRange(0, rpm_max); p_cam.setYRange(0, ymax)
        from core.modal.campbell import summarize as _cs
        lbl_cam.setText(f"<b>N máquina = {rpm_op:.0f} RPM</b> · margen de separación API 684 "
                        f"±{SM*100:.0f}% (zona {lo:.0f}–{hi:.0f} RPM). Todo cruce dentro de esa zona "
                        f"es una coincidencia de riesgo. " + _cs(cx))
    btn_refc.clicked.connect(_refresh_campbell)

    return app, win


def main(argv=None):
    ap = argparse.ArgumentParser(description="Watermelon Modal — EMA + OMA (native)")
    ap.add_argument("--sim", action="store_true", default=True)
    ap.add_argument("--name", default="Tren Motor-Bomba")
    args = ap.parse_args(argv)
    lay = default_24ch_layout(name=args.name)
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
