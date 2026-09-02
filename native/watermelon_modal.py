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

from core.modal.live_impact import (FRFAccumulator, HitQuality, assess_hit,
                                     modes_from_frf, synth_impact)
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
    r3.addSpacing(16); r3.addWidget(QtWidgets.QLabel("Test type:"))
    cb_ttype = QtWidgets.QComboBox(); cb_ttype.addItems(["OMA", "EMA"]); cb_ttype.setCurrentText(layout.test_type)
    r3.addWidget(cb_ttype); r3.addStretch(1)
    _w3 = QtWidgets.QWidget(); _w3.setLayout(r3); frm.addRow("Operation:", _w3)
    ml.addLayout(frm)

    ml.addWidget(QtWidgets.QLabel(
        "<b>Machine drawing</b> — elegí un sensor y <b>hacé clic sobre el gráfico</b> para ubicarlo. "
        "Motor · Bomba · Skid · Tubería."))
    prow = QtWidgets.QHBoxLayout()
    prow.addWidget(QtWidgets.QLabel("Sensor a ubicar:"))
    cb_place = QtWidgets.QComboBox(); cb_place.setMinimumWidth(220); prow.addWidget(cb_place)
    btn_tmpl = QtWidgets.QPushButton("Cargar plantilla 24 canales")
    prow.addWidget(btn_tmpl); prow.addStretch(1)
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
    tbl_pts = QtWidgets.QTableWidget(0, 9)
    tbl_pts.setHorizontalHeaderLabels(["#", "Componente", "Referencia", "DOF", "Slot", "Canal",
                                       "Sens (mV/g)", "Ref", "Activo"])
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
        r = tbl_pts.rowCount(); tbl_pts.insertRow(r)
        tbl_pts.setItem(r, 0, QtWidgets.QTableWidgetItem(str(mp.idx)))
        tbl_pts.setCellWidget(r, 1, _mk_combo(DEFAULT_COMPONENTS, mp.component))
        tbl_pts.setCellWidget(r, 2, _mk_combo(POSITION_REFS, mp.position_ref))
        tbl_pts.setCellWidget(r, 3, _mk_combo(DOFS, mp.dof))
        tbl_pts.setItem(r, 4, QtWidgets.QTableWidgetItem(str(mp.module_slot)))
        tbl_pts.setItem(r, 5, QtWidgets.QTableWidgetItem(str(mp.channel_index)))
        tbl_pts.setItem(r, 6, QtWidgets.QTableWidgetItem(f"{mp.sensitivity_mv_per_g:g}"))
        cbref = QtWidgets.QCheckBox(); cbref.setChecked(mp.reference_sensor)
        cbact = QtWidgets.QCheckBox(); cbact.setChecked(mp.active)
        tbl_pts.setCellWidget(r, 7, cbref); tbl_pts.setCellWidget(r, 8, cbact)

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
                    idx=int(_t(0, str(r + 1)) or r + 1), component=_c(1), position_ref=_c(2), dof=_c(3),
                    module_slot=int(_t(4, "1") or 1), channel_index=int(_t(5, "0") or 0),
                    sensitivity_mv_per_g=float(_t(6, "100") or 100), reference_sensor=_k(7), active=_k(8),
                    x_norm=old.x_norm if old else 0.5, y_norm=old.y_norm if old else 0.0))
            except Exception:  # noqa: BLE001
                continue
        lay.points = pts
        lay.name = e_name.text() or "Modal"; lay.machine_type = e_type.text(); lay.tag = e_tag.text()
        lay.client = e_client.text(); lay.location = e_loc.text()
        lay.running_speed_rpm = sp_rpm.value(); lay.test_type = cb_ttype.currentText()
        lay.fs_hz = float(sp_fs.value()); lay.block_size = int(cb_blk.currentText())
        lay.fmax_hz = float(sp_fmax.value()); lay.duration_s = float(sp_dur.value())

    def _draw_train():
        p_train.clear()
        lay = st["layout"]
        def box(x0, x1, y0, y1, color, label):
            p_train.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], pen=pg.mkPen(color, width=2))
            t = pg.TextItem(label, color=color, anchor=(0.5, 0.5)); t.setPos((x0 + x1) / 2, (y0 + y1) / 2)
            p_train.addItem(t)
        p_train.plot([0.0, 1.0, 1.0, 0.0, 0.0], [-0.55, -0.55, -0.42, -0.42, -0.55],
                     pen=pg.mkPen(_COMP_COLOR["Skid"], width=2))
        ts = pg.TextItem("SKID", color=_COMP_COLOR["Skid"], anchor=(0, 0.5)); ts.setPos(0.02, -0.485); p_train.addItem(ts)
        box(0.03, 0.23, -0.30, 0.30, _COMP_COLOR["Motor"], "MOTOR")
        p_train.plot([0.23, 0.29], [0, 0], pen=pg.mkPen("#334155", width=3))
        box(0.29, 0.53, -0.30, 0.30, _COMP_COLOR["Bomba"], "BOMBA")
        p_train.plot([0.53, 0.72, 0.72], [0.10, 0.10, 0.45], pen=pg.mkPen(_COMP_COLOR["Tubería descarga"], width=3))
        p_train.plot([0.53, 0.90], [-0.12, -0.12], pen=pg.mkPen(_COMP_COLOR["Tubería succión"], width=3))
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
            ref = mp.position_ref.split(" ")[0]
            t = pg.TextItem(f"{ref}{mp.dof}", color=NAVY, anchor=(0.5, 1.4))
            t.setPos(mp.x_norm, mp.y_norm); t.setScale(0.9); p_train.addItem(t)
        p_train.getViewBox().autoRange(padding=0.12)

    def _on_train_click(ev):
        try:
            if ev.button() != QtCore.Qt.LeftButton:
                return
            i = cb_place.currentIndex()
            if not (0 <= i < len(st["layout"].points)):
                return
            pt = p_train.getViewBox().mapSceneToView(ev.scenePos())
            st["layout"].points[i].x_norm = float(pt.x())
            st["layout"].points[i].y_norm = float(pt.y())
            _draw_train()
        except Exception:  # noqa: BLE001
            pass
    p_train.scene().sigMouseClicked.connect(_on_train_click)

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
        lay.location = e_loc.text(); lay.test_type = cb_ttype.currentText()
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
        f, y = synth_impact(lay.fs_hz, lay.block_size, rng=st["rng"], double_hit=dbl, overload=over)
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
    btn_savec = QtWidgets.QPushButton("💾 Guardar como condición")
    crow.addWidget(btn_ocap); crow.addWidget(btn_savec); crow.addStretch(1)
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
        for fn, z in [(19.4, 0.02), (38.8, 0.015), (77.4, 0.012), (129.9, 0.01)]:
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
        lbl_ost.setText(f"✅ FDD listo — {len(fdd.modes)} modos. Guardá como condición y mirá Comparative / Campbell.")
        lbl_ost.setStyleSheet(f"color:{GREEN};font-weight:700;")
        _refresh_campbell()

    def _save_condition():
        fdd = st["oma_fdd"]
        if fdd is None:
            QtWidgets.QMessageBox.information(win, "Comparative", "Capturá primero (OMA capture)."); return
        label = f"Condición {len(st['conditions']) + 1}"
        st["conditions"].append({"label": label,
                                 "freqs": [m.natural_frequency_hz for m in fdd.modes],
                                 "damps": [m.damping_ratio_pct for m in fdd.modes]})
        _refresh_comparative()
        QtWidgets.QMessageBox.information(win, "Comparative", f"Guardado: {label}.")
    btn_ocap.clicked.connect(_oma_capture); btn_savec.clicked.connect(_save_condition)

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
    cpl.addWidget(QtWidgets.QLabel("<b>Comparativo de condiciones OMA</b> — tracking de modos por frecuencia."))
    tbl_cmp = QtWidgets.QTableWidget(0, 1); tbl_cmp.verticalHeader().setVisible(False)
    cpl.addWidget(tbl_cmp, 1)
    btn_clrc = QtWidgets.QPushButton("Limpiar condiciones"); cpl.addWidget(btn_clrc, alignment=QtCore.Qt.AlignLeft)
    tabs.addTab(pg_cmp, "Comparative")

    def _refresh_comparative():
        conds = st["conditions"]
        if not conds:
            tbl_cmp.setRowCount(0); tbl_cmp.setColumnCount(1)
            tbl_cmp.setHorizontalHeaderLabels(["(sin condiciones)"]); return
        # tracking: agrupa frecuencias de todas las condiciones (tol 2 Hz)
        allf = sorted(f for c in conds for f in c["freqs"])
        clusters = []
        for f in allf:
            if clusters and abs(f - clusters[-1][-1]) <= 2.0:
                clusters[-1].append(f)
            else:
                clusters.append([f])
        centers = [float(np.mean(c)) for c in clusters]
        tbl_cmp.setColumnCount(1 + len(conds))
        tbl_cmp.setHorizontalHeaderLabels(["Modo ~ (Hz)"] + [c["label"] for c in conds])
        tbl_cmp.setRowCount(len(centers))
        for i, cen in enumerate(centers):
            tbl_cmp.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{cen:.2f}"))
            for j, c in enumerate(conds):
                near = [f for f in c["freqs"] if abs(f - cen) <= 2.0]
                tbl_cmp.setItem(i, j + 1, QtWidgets.QTableWidgetItem(f"{near[0]:.3f}" if near else "—"))
        tbl_cmp.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    btn_clrc.clicked.connect(lambda: (st["conditions"].clear(), _refresh_comparative()))
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
        rpm_max = max(rpm_op * 1.3, 1500.0)
        bands = [SpeedBand(rpm_op, 0.17 * rpm_op, f"Máx {rpm_op:.0f}±{0.17*rpm_op:.0f}"),
                 SpeedBand(rpm_op / 2, 0.17 * rpm_op / 1, "½ velocidad")]
        if not modes:
            lbl_cam.setText("Sin modos aún — capturá OMA o identificá EMA."); tbl_cam.setRowCount(0); return
        orders = (0.5, 1.0, 2.0, 3.0, 4.0)
        rpm = np.linspace(0, rpm_max, 60)
        for b in bands:                                 # bandas sombreadas
            reg = pg.LinearRegionItem([max(0, b.low), min(rpm_max, b.high)], movable=False,
                                      brush=pg.mkBrush(245, 158, 11, 40)); reg.setZValue(-10); p_cam.addItem(reg)
        for o in orders:                                # líneas de orden
            p_cam.plot(rpm, o * rpm / 60.0, pen=pg.mkPen("#6B7280", width=1, style=QtCore.Qt.DotLine))
            t = pg.TextItem(f"{o:g}×", color="#6B7280", anchor=(0, 0.5)); t.setPos(rpm_max * 0.98, o * rpm_max / 60.0)
            p_cam.addItem(t)
        for fn in modes:                                # modos (horizontales)
            p_cam.plot([0, rpm_max], [fn, fn], pen=pg.mkPen(GREEN, width=2))
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
        p_cam.plot([rpm_op, rpm_op], [0, max(modes) * 1.25], pen=pg.mkPen(NAVY, width=2, style=QtCore.Qt.DashLine))
        p_cam.setXRange(0, rpm_max); p_cam.setYRange(0, max(modes) * 1.25)
        from core.modal.campbell import summarize as _cs
        lbl_cam.setText(_cs(cx))
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
