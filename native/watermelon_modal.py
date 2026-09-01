"""
Watermelon Modal — módulo NATIVO de ensayo de impacto (análisis modal, Windows)
===============================================================================

App de campo (PySide6 + pyqtgraph) para EMA con martillo, homóloga a
Watermelon Rotordynamics. Flujo de clase mundial (ISO 7626-5):

  Configuration  → canales (martillo + acelerómetros), fs, banda.
  Impact test    → golpe a golpe: FRF + coherencia EN VIVO, ventana de fuerza y
                   exponencial, aceptar/rechazar, promedios; LEDs de doble golpe
                   y sobrecarga.
  Modes          → identifica fn / amortiguamiento desde la FRF promediada +
                   Nyquist (movilidad).

Modo SIMULADO por defecto (--sim): golpes sintéticos físicamente correctos, para
demo/entrenamiento SIN hardware. Con NI 9234 (IEPE) real, el mismo acumulador
recibe el registro disparado — la UI no cambia.

El cómputo vive en core.modal.live_impact (numpy/scipy, testeado).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except Exception as exc:  # noqa: BLE001
    print("Falta PySide6/pyqtgraph:", exc)
    raise

from core.modal.live_impact import (FRFAccumulator, HitQuality, SynthMode,
                                     assess_hit, modes_from_frf, synth_impact,
                                     DEFAULT_SPECIMEN)
from core.modal.oma_layout import (OMALayout, MeasPoint, default_24ch_layout,
                                    DEFAULT_COMPONENTS, POSITION_REFS, DOFS)
from core.modal.oma_engine import run_oma

# ---- Paleta (misma marca que Rotordynamics) ----
NAVY = "#0F1E3D"
PANEL = "#12325f"
ACC = "#1AAEE5"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
GRID = "#e6ecf5"


# =====================================================================
# Configuración de canales del ensayo modal
# =====================================================================
@dataclass
class ModalChannel:
    name: str
    role: str            # "force" (martillo) | "response" (acelerómetro)
    bnc: int
    sensitivity: float   # mV/EU (martillo mV/N, acel mV/g)
    unit: str


@dataclass
class ModalSetup:
    name: str = "Test specimen"
    fs: float = 2048.0
    block_size: int = 4096          # muestras por golpe (Δf = fs/block)
    fmax_hz: float = 400.0
    channels: List[ModalChannel] = field(default_factory=list)

    @staticmethod
    def default() -> "ModalSetup":
        return ModalSetup(channels=[
            ModalChannel("Hammer", "force", 1, 2.25, "N"),
            ModalChannel("Accel 1", "response", 2, 100.0, "g"),
            ModalChannel("Accel 2", "response", 3, 100.0, "g"),
        ])


# =====================================================================
# App
# =====================================================================
def _stylesheet() -> str:
    return f"""
    QWidget {{ font-family: 'Segoe UI', Arial; font-size: 12px; color: {NAVY}; }}
    QMainWindow, QTabWidget::pane {{ background: #f5f8fd; }}
    QTabBar::tab {{ background: #e6ecf5; padding: 8px 18px; margin-right: 3px;
        border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 700; }}
    QTabBar::tab:selected {{ background: {NAVY}; color: white; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: white;
        border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px 8px; }}
    QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
    QPushButton {{ background: {NAVY}; color: white; border: none; font-weight: 700;
        padding: 8px 16px; border-radius: 8px; }}
    QPushButton:hover {{ background: #0e3a6b; }}
    QPushButton:disabled {{ background: #94a3b8; }}
    QTableWidget {{ background: white; gridline-color: {GRID}; }}
    QHeaderView::section {{ background: {NAVY}; color: white; padding: 5px;
        border: none; font-weight: 700; }}
    """


def _led(color: str, on: bool, label: str) -> str:
    c = color if on else "#cbd5e1"
    return (f"<span style='color:{c};font-size:16px'>●</span> "
            f"<span style='font-weight:700;color:{'#334155' if on else '#94a3b8'}'>{label}</span>")


def build_app(setup: ModalSetup, simulated: bool = True):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(_stylesheet())
    pg.setConfigOptions(antialias=True)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Modal — {setup.name}")
    win.resize(1240, 800)

    # ---- estado del ensayo ----
    st = {
        "acc": FRFAccumulator(setup.fs, setup.block_size,
                              use_force_window=True, use_exp_window=True),
        "pending": None,        # (force, resp, quality) del golpe sin decidir
        "target": 5,
        "rng": np.random.default_rng(),
    }

    # ---- barra superior (marca) ----
    tb = win.addToolBar("main"); tb.setMovable(False)
    tb.setStyleSheet(f"QToolBar {{ background: {NAVY}; padding: 6px 12px; }}")
    brand = QtWidgets.QLabel("  🍉 Watermelon Modal")
    brand.setStyleSheet("color:white; font-weight:800; font-size:15px;")
    tb.addWidget(brand)
    spacer = QtWidgets.QWidget(); spacer.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    tb.addWidget(spacer)
    mode_lbl = QtWidgets.QLabel(("SIMULATED — no hardware" if simulated else "LIVE — NI 9234") + "   ")
    mode_lbl.setStyleSheet(f"color:{'#fbbf24' if simulated else '#34d399'}; font-weight:700;")
    tb.addWidget(mode_lbl)

    tabs = QtWidgets.QTabWidget(); win.setCentralWidget(tabs)

    # =================================================================
    # TAB 1 — Configuration
    # =================================================================
    pg_cfg = QtWidgets.QWidget(); cl = QtWidgets.QVBoxLayout(pg_cfg)
    hdr = QtWidgets.QLabel("Configuration  —  impact test setup (ISO 7626)")
    hdr.setStyleSheet(f"background:{NAVY};color:white;border-radius:8px;padding:9px 14px;font-weight:700;")
    cl.addWidget(hdr)

    frm = QtWidgets.QFormLayout()
    e_name = QtWidgets.QLineEdit(setup.name)
    sp_fs = QtWidgets.QSpinBox(); sp_fs.setRange(256, 51200); sp_fs.setValue(int(setup.fs)); sp_fs.setSingleStep(256)
    sp_blk = QtWidgets.QComboBox(); sp_blk.addItems(["1024", "2048", "4096", "8192", "16384"])
    sp_blk.setCurrentText(str(setup.block_size))
    sp_fmax = QtWidgets.QDoubleSpinBox(); sp_fmax.setRange(10, 25600); sp_fmax.setValue(setup.fmax_hz)
    lbl_df = QtWidgets.QLabel("")
    frm.addRow("Specimen / test name:", e_name)
    frm.addRow("Sampling (Hz):", sp_fs)
    frm.addRow("Block size (samples/hit):", sp_blk)
    frm.addRow("Fmax (Hz):", sp_fmax)
    frm.addRow("Resolution:", lbl_df)
    cl.addLayout(frm)

    cl.addWidget(QtWidgets.QLabel("<b>Channels</b> — 1 hammer (force) + accelerometers (response)"))
    tblc = QtWidgets.QTableWidget(0, 5)
    tblc.setHorizontalHeaderLabels(["Channel", "Role", "BNC", "Sensitivity (mV/EU)", "Unit"])
    tblc.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tblc.verticalHeader().setVisible(False)
    cl.addWidget(tblc, 1)

    def _fill_channels():
        tblc.setRowCount(0)
        for ch in setup.channels:
            r = tblc.rowCount(); tblc.insertRow(r)
            tblc.setItem(r, 0, QtWidgets.QTableWidgetItem(ch.name))
            combo = QtWidgets.QComboBox(); combo.addItems(["force", "response"]); combo.setCurrentText(ch.role)
            tblc.setCellWidget(r, 1, combo)
            tblc.setItem(r, 2, QtWidgets.QTableWidgetItem(str(ch.bnc)))
            tblc.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{ch.sensitivity:g}"))
            tblc.setItem(r, 4, QtWidgets.QTableWidgetItem(ch.unit))

    def _upd_df(*_):
        fs = float(sp_fs.value()); blk = int(sp_blk.currentText())
        lbl_df.setText(f"Δf = {fs/blk:.3f} Hz   ·   record = {blk/fs*1000:.0f} ms   ·   "
                       f"lines to Fmax = {int(sp_fmax.value()/(fs/blk))}")
    sp_fs.valueChanged.connect(_upd_df); sp_blk.currentTextChanged.connect(_upd_df)
    sp_fmax.valueChanged.connect(_upd_df)
    _fill_channels(); _upd_df()

    rowb = QtWidgets.QHBoxLayout()
    btn_add = QtWidgets.QPushButton("+ Accelerometer")
    btn_apply_cfg = QtWidgets.QPushButton("✓ Apply configuration"); btn_apply_cfg.setStyleSheet(
        f"QPushButton{{background:{GREEN};}} QPushButton:hover{{background:#0e9f6e;}}")
    rowb.addWidget(btn_add); rowb.addStretch(1); rowb.addWidget(btn_apply_cfg)
    cl.addLayout(rowb)
    tabs.addTab(pg_cfg, "Configuration")

    def _add_accel():
        r = tblc.rowCount(); tblc.insertRow(r)
        tblc.setItem(r, 0, QtWidgets.QTableWidgetItem(f"Accel {r}"))
        combo = QtWidgets.QComboBox(); combo.addItems(["force", "response"]); combo.setCurrentText("response")
        tblc.setCellWidget(r, 1, combo)
        tblc.setItem(r, 2, QtWidgets.QTableWidgetItem(str(r + 1)))
        tblc.setItem(r, 3, QtWidgets.QTableWidgetItem("100"))
        tblc.setItem(r, 4, QtWidgets.QTableWidgetItem("g"))
    btn_add.clicked.connect(_add_accel)

    def _apply_cfg():
        setup.name = e_name.text() or "Test"
        setup.fs = float(sp_fs.value()); setup.block_size = int(sp_blk.currentText())
        setup.fmax_hz = float(sp_fmax.value())
        st["acc"] = FRFAccumulator(setup.fs, setup.block_size,
                                   use_force_window=chk_fwin.isChecked(),
                                   use_exp_window=chk_ewin.isChecked())
        st["pending"] = None
        win.setWindowTitle(f"Watermelon Modal — {setup.name}")
        _refresh_impact()
        QtWidgets.QMessageBox.information(win, "Configuration",
            f"✓ Applied — fs {setup.fs:.0f} Hz · block {setup.block_size} · "
            f"Δf {setup.fs/setup.block_size:.3f} Hz. Averaging reset.")
    btn_apply_cfg.clicked.connect(_apply_cfg)

    # =================================================================
    # TAB 2 — Impact test
    # =================================================================
    pg_imp = QtWidgets.QWidget(); il = QtWidgets.QVBoxLayout(pg_imp)

    # -- barra de control --
    ctl = QtWidgets.QHBoxLayout()
    btn_hit = QtWidgets.QPushButton("🔨 Impact"); btn_hit.setStyleSheet(
        f"QPushButton{{background:{ACC};font-size:14px;padding:10px 22px;}} QPushButton:hover{{background:#1490c2;}}")
    btn_badhit = QtWidgets.QPushButton("⚠ Impact w/ fault (test)")
    btn_accept = QtWidgets.QPushButton("✓ Accept"); btn_accept.setStyleSheet(
        f"QPushButton{{background:{GREEN};}} QPushButton:hover{{background:#0e9f6e;}}")
    btn_reject = QtWidgets.QPushButton("✗ Reject"); btn_reject.setStyleSheet(
        f"QPushButton{{background:{RED};}} QPushButton:hover{{background:#c81e1e;}}")
    btn_accept.setEnabled(False); btn_reject.setEnabled(False)
    ctl.addWidget(btn_hit); ctl.addWidget(btn_badhit)
    ctl.addSpacing(16); ctl.addWidget(btn_accept); ctl.addWidget(btn_reject)
    ctl.addStretch(1)
    ctl.addWidget(QtWidgets.QLabel("Target avg:"))
    sp_target = QtWidgets.QSpinBox(); sp_target.setRange(1, 64); sp_target.setValue(st["target"])
    ctl.addWidget(sp_target)
    btn_reset = QtWidgets.QPushButton("↻ Reset"); ctl.addWidget(btn_reset)
    il.addLayout(ctl)

    # -- fila de estado: promedios + LEDs + ventanas --
    strow = QtWidgets.QHBoxLayout()
    lbl_avg = QtWidgets.QLabel(); lbl_avg.setStyleSheet("font-size:14px;font-weight:800;")
    lbl_leds = QtWidgets.QLabel(); lbl_leds.setTextFormat(QtCore.Qt.RichText)
    chk_fwin = QtWidgets.QCheckBox("Force window"); chk_fwin.setChecked(True)
    chk_ewin = QtWidgets.QCheckBox("Exponential window"); chk_ewin.setChecked(True)
    strow.addWidget(lbl_avg); strow.addSpacing(20); strow.addWidget(lbl_leds)
    strow.addStretch(1); strow.addWidget(chk_fwin); strow.addWidget(chk_ewin)
    il.addLayout(strow)

    # -- gráficos --
    plots = QtWidgets.QGridLayout()
    p_frf = pg.PlotWidget(); p_frf.setBackground("w"); p_frf.setLogMode(x=False, y=True)
    p_frf.setLabel("left", "|FRF| (mobility)"); p_frf.setLabel("bottom", "Frequency", "Hz")
    p_frf.setTitle("FRF — H1", color=NAVY); p_frf.showGrid(x=True, y=True, alpha=0.3)
    p_coh = pg.PlotWidget(); p_coh.setBackground("w"); p_coh.setYRange(0, 1.05)
    p_coh.setLabel("left", "Coherence γ²"); p_coh.setLabel("bottom", "Frequency", "Hz")
    p_coh.setTitle("Coherence", color=NAVY); p_coh.showGrid(x=True, y=True, alpha=0.3)
    p_time = pg.PlotWidget(); p_time.setBackground("w")
    p_time.setLabel("left", "Amplitude"); p_time.setLabel("bottom", "Time", "s")
    p_time.setTitle("Last hit — force & response", color=NAVY); p_time.showGrid(x=True, y=True, alpha=0.3)
    plots.addWidget(p_frf, 0, 0); plots.addWidget(p_coh, 0, 1)
    plots.addWidget(p_time, 1, 0, 1, 2)
    plots.setRowStretch(0, 3); plots.setRowStretch(1, 2)
    il.addLayout(plots, 1)
    tabs.addTab(pg_imp, "Impact test")

    cur_frf = p_frf.plot([], [], pen=pg.mkPen("#94a3b8", width=1, style=QtCore.Qt.DashLine))
    avg_frf = p_frf.plot([], [], pen=pg.mkPen(ACC, width=2))
    coh_curve = p_coh.plot([], [], pen=pg.mkPen(GREEN, width=2))
    coh_ref = pg.InfiniteLine(pos=0.8, angle=0, pen=pg.mkPen(AMBER, style=QtCore.Qt.DashLine))
    p_coh.addItem(coh_ref)
    force_curve = p_time.plot([], [], pen=pg.mkPen(RED, width=1.5), name="force")
    resp_curve = p_time.plot([], [], pen=pg.mkPen(NAVY, width=1), name="response")

    def _leds(q: Optional[HitQuality]):
        over = bool(q and q.overload); dbl = bool(q and q.double_hit)
        lbl_leds.setText(_led(RED, over, "Overload") + " &nbsp;&nbsp; "
                         + _led(AMBER, dbl, "Double-hit"))

    def _refresh_impact():
        acc = st["acc"]
        lbl_avg.setText(f"Averages: {acc.count} / {st['target']}"
                        + ("   ✅ target reached" if acc.count >= st["target"] else ""))
        res = acc.result()
        if res is not None:
            m = (res.frequencies_hz <= setup.fmax_hz)
            avg_frf.setData(res.frequencies_hz[m], np.maximum(res.magnitude[m], 1e-9))
            coh_curve.setData(res.frequencies_hz[m], res.coherence[m])
        else:
            avg_frf.setData([], []); coh_curve.setData([], [])

    def _do_hit(fault: bool):
        dbl = fault and (st["rng"].random() < 0.5)
        over = fault and not dbl
        f, y = synth_impact(setup.fs, setup.block_size, rng=st["rng"],
                            double_hit=dbl, overload=over)
        q = assess_hit(f, y, setup.fs)
        st["pending"] = (f, y, q)
        # preview
        acc = st["acc"]
        acc.use_force_window = chk_fwin.isChecked(); acc.use_exp_window = chk_ewin.isChecked()
        prev = acc.preview(f, y)
        m = (prev.frequencies_hz <= setup.fmax_hz)
        cur_frf.setData(prev.frequencies_hz[m], np.maximum(prev.magnitude[m], 1e-9))
        t = np.arange(setup.block_size) / setup.fs
        force_curve.setData(t, f); resp_curve.setData(t, y)
        _leds(q)
        btn_accept.setEnabled(True); btn_reject.setEnabled(True)
        # sugerencia si el golpe es malo
        if q.overload or q.double_hit:
            lbl_leds.setText(lbl_leds.text() + "  <span style='color:#b91c1c;font-weight:800'>"
                             "→ recommend REJECT</span>")

    def _accept():
        if not st["pending"]:
            return
        f, y, q = st["pending"]
        st["acc"].add(f, y)
        st["pending"] = None
        cur_frf.setData([], [])
        btn_accept.setEnabled(False); btn_reject.setEnabled(False)
        _refresh_impact()

    def _reject():
        st["pending"] = None
        cur_frf.setData([], [])
        btn_accept.setEnabled(False); btn_reject.setEnabled(False)
        _leds(None)

    def _reset():
        st["acc"].reset(); st["pending"] = None
        cur_frf.setData([], []); avg_frf.setData([], []); coh_curve.setData([], [])
        force_curve.setData([], []); resp_curve.setData([], [])
        btn_accept.setEnabled(False); btn_reject.setEnabled(False)
        _leds(None); _refresh_impact()

    btn_hit.clicked.connect(lambda: _do_hit(False))
    btn_badhit.clicked.connect(lambda: _do_hit(True))
    btn_accept.clicked.connect(_accept)
    btn_reject.clicked.connect(_reject)
    btn_reset.clicked.connect(_reset)
    sp_target.valueChanged.connect(lambda v: (st.update(target=v), _refresh_impact()))
    _leds(None); _refresh_impact()

    # =================================================================
    # TAB 3 — Modes
    # =================================================================
    pg_mod = QtWidgets.QWidget(); ql = QtWidgets.QVBoxLayout(pg_mod)
    mrow = QtWidgets.QHBoxLayout()
    btn_ident = QtWidgets.QPushButton("🎯 Identify modes"); btn_ident.setStyleSheet(
        f"QPushButton{{background:{ACC};}} QPushButton:hover{{background:#1490c2;}}")
    lbl_modes_hdr = QtWidgets.QLabel("Peak-picking + half-power damping (ISO 7626-6) on the averaged FRF.")
    mrow.addWidget(btn_ident); mrow.addWidget(lbl_modes_hdr); mrow.addStretch(1)
    ql.addLayout(mrow)

    msplit = QtWidgets.QHBoxLayout()
    tbl_modes = QtWidgets.QTableWidget(0, 4)
    tbl_modes.setHorizontalHeaderLabels(["Freq (Hz)", "Damping (%)", "Coherence", "Reliable"])
    tbl_modes.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tbl_modes.verticalHeader().setVisible(False)
    tbl_modes.setMaximumWidth(520)
    p_nyq = pg.PlotWidget(); p_nyq.setBackground("w"); p_nyq.setAspectLocked(True)
    p_nyq.setLabel("left", "Im(FRF)"); p_nyq.setLabel("bottom", "Re(FRF)")
    p_nyq.setTitle("Nyquist (mobility)", color=NAVY); p_nyq.showGrid(x=True, y=True, alpha=0.3)
    nyq_curve = p_nyq.plot([], [], pen=pg.mkPen(NAVY, width=1.5))
    msplit.addWidget(tbl_modes, 2); msplit.addWidget(p_nyq, 3)
    ql.addLayout(msplit, 1)
    tabs.addTab(pg_mod, "Modes")

    def _identify():
        res = st["acc"].result()
        if res is None:
            QtWidgets.QMessageBox.information(win, "Modes",
                "No accepted hits yet. Go to Impact test and accept some hits.")
            return
        modes = modes_from_frf(res, fmin=5.0, fmax=setup.fmax_hz)
        tbl_modes.setRowCount(0)
        for mo in modes:
            r = tbl_modes.rowCount(); tbl_modes.insertRow(r)
            vals = [f"{mo.frequency_hz:.1f}", f"{mo.damping_ratio_pct:.2f}",
                    (f"{mo.coherence_at_peak:.2f}" if mo.coherence_at_peak is not None else "—"),
                    ("✓" if mo.is_reliable else "✗")]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if c == 3:
                    it.setForeground(QtGui.QBrush(QtGui.QColor(GREEN if mo.is_reliable else RED)))
                tbl_modes.setItem(r, c, it)
        m = (res.frequencies_hz <= setup.fmax_hz) & (res.frequencies_hz >= 5.0)
        nyq_curve.setData(res.frf_complex[m].real, res.frf_complex[m].imag)
    btn_ident.clicked.connect(_identify)

    # =================================================================
    # TAB 4 — OMA setup (máquina dibujada + puntos de medición)
    # =================================================================
    st["oma"] = default_24ch_layout()
    st["oma_fdd"] = None
    _COMP_COLOR = {"Motor": "#2563eb", "Bomba": "#16a34a", "Skid": "#a16207",
                   "Tubería succión": "#0891b2", "Tubería descarga": "#7c3aed"}

    pg_os = QtWidgets.QWidget(); ol = QtWidgets.QVBoxLayout(pg_os)
    hdr_o = QtWidgets.QLabel("OMA setup — ubicá cada sensor en la máquina con su referencia (ISO 7626 / API 684)")
    hdr_o.setStyleSheet(f"background:{NAVY};color:white;border-radius:8px;padding:9px 14px;font-weight:700;")
    ol.addWidget(hdr_o)

    orow = QtWidgets.QHBoxLayout()
    e_oname = QtWidgets.QLineEdit(st["oma"].name)
    sp_ofs = QtWidgets.QSpinBox(); sp_ofs.setRange(256, 25600); sp_ofs.setValue(int(st["oma"].fs_hz)); sp_ofs.setSingleStep(256)
    sp_odur = QtWidgets.QSpinBox(); sp_odur.setRange(10, 900); sp_odur.setValue(int(st["oma"].duration_s))
    sp_orpm = QtWidgets.QDoubleSpinBox(); sp_orpm.setRange(0, 60000); sp_orpm.setValue(st["oma"].running_speed_rpm)
    for lab, w in (("Test:", e_oname), ("fs (Hz):", sp_ofs), ("Duración (s):", sp_odur), ("RPM:", sp_orpm)):
        orow.addWidget(QtWidgets.QLabel(lab)); orow.addWidget(w)
    orow.addStretch(1)
    btn_tmpl = QtWidgets.QPushButton("Cargar plantilla 24 canales")
    btn_addp = QtWidgets.QPushButton("+ Punto")
    btn_draw = QtWidgets.QPushButton("✓ Aplicar y dibujar"); btn_draw.setStyleSheet(
        f"QPushButton{{background:{GREEN};}} QPushButton:hover{{background:#0e9f6e;}}")
    orow.addWidget(btn_tmpl); orow.addWidget(btn_addp); orow.addWidget(btn_draw)
    ol.addLayout(orow)

    osplit = QtWidgets.QHBoxLayout()
    tbl_pts = QtWidgets.QTableWidget(0, 9)
    tbl_pts.setHorizontalHeaderLabels(["#", "Componente", "Referencia", "DOF", "Slot", "Canal",
                                       "Sens (mV/g)", "Ref", "Activo"])
    tbl_pts.verticalHeader().setVisible(False)
    tbl_pts.setMaximumWidth(720)
    p_train = pg.PlotWidget(); p_train.setBackground("w"); p_train.setAspectLocked(True)
    p_train.hideAxis("left"); p_train.hideAxis("bottom"); p_train.setMenuEnabled(False)
    p_train.setTitle("Máquina y ubicación de sensores", color=NAVY, size="10pt")
    _tvb = p_train.getViewBox(); _tvb.setMouseEnabled(x=False, y=False)
    osplit.addWidget(tbl_pts, 3); osplit.addWidget(p_train, 4)
    ol.addLayout(osplit, 1)
    lbl_oval = QtWidgets.QLabel(""); lbl_oval.setStyleSheet("font-weight:700;")
    ol.addWidget(lbl_oval)
    tabs.addTab(pg_os, "OMA setup")

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
        for mp in st["oma"].points:
            _add_point_row(mp)

    def _table_to_layout() -> OMALayout:
        pts = []
        for r in range(tbl_pts.rowCount()):
            def _txt(c, d=""):
                it = tbl_pts.item(r, c); return it.text() if it else d
            def _cmb(c):
                w = tbl_pts.cellWidget(r, c); return w.currentText() if w else ""
            def _chk(c):
                w = tbl_pts.cellWidget(r, c); return bool(w.isChecked()) if w else False
            try:
                pts.append(MeasPoint(
                    idx=int(_txt(0, str(r + 1)) or r + 1), component=_cmb(1),
                    position_ref=_cmb(2), dof=_cmb(3),
                    module_slot=int(_txt(4, "1") or 1), channel_index=int(_txt(5, "0") or 0),
                    sensitivity_mv_per_g=float(_txt(6, "100") or 100),
                    reference_sensor=_chk(7), active=_chk(8),
                    x_norm=st["oma"].points[r].x_norm if r < len(st["oma"].points) else 0.5,
                    y_norm=st["oma"].points[r].y_norm if r < len(st["oma"].points) else 0.0))
            except Exception:  # noqa: BLE001
                continue
        lay = OMALayout(name=e_oname.text() or "OMA", points=pts,
                        fs_hz=float(sp_ofs.value()), duration_s=float(sp_odur.value()),
                        running_speed_rpm=sp_orpm.value())
        return lay

    def _draw_train():
        p_train.clear()
        lay = st["oma"]
        def box(x0, x1, y0, y1, color, label):
            p_train.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                         pen=pg.mkPen(color, width=2))
            t = pg.TextItem(label, color=color, anchor=(0.5, 0.5))
            t.setPos((x0 + x1) / 2, (y0 + y1) / 2); p_train.addItem(t)
        # skid base
        p_train.plot([0.0, 1.0, 1.0, 0.0, 0.0], [-0.55, -0.55, -0.42, -0.42, -0.55],
                     pen=pg.mkPen(_COMP_COLOR["Skid"], width=2))
        ts = pg.TextItem("SKID", color=_COMP_COLOR["Skid"], anchor=(0, 0.5)); ts.setPos(0.02, -0.485); p_train.addItem(ts)
        box(0.03, 0.23, -0.30, 0.30, _COMP_COLOR["Motor"], "MOTOR")
        p_train.plot([0.23, 0.29], [0, 0], pen=pg.mkPen("#334155", width=3))   # coupling
        box(0.29, 0.53, -0.30, 0.30, _COMP_COLOR["Bomba"], "BOMBA")
        # tuberías
        p_train.plot([0.53, 0.72, 0.72], [0.10, 0.10, 0.45], pen=pg.mkPen(_COMP_COLOR["Tubería descarga"], width=3))
        p_train.plot([0.53, 0.90], [-0.12, -0.12], pen=pg.mkPen(_COMP_COLOR["Tubería succión"], width=3))
        # sensores
        for mp in lay.active_points():
            col = _COMP_COLOR.get(mp.component, "#475569")
            x = mp.x_norm; y = mp.y_norm
            sz = 16 if mp.reference_sensor else 11
            sym = "star" if mp.reference_sensor else "o"
            p_train.addItem(pg.ScatterPlotItem([x], [y], size=sz, symbol=sym,
                            brush=pg.mkBrush(col), pen=pg.mkPen("w", width=1.5)))
            ref = mp.position_ref.split(" ")[0]
            t = pg.TextItem(f"{ref}{mp.dof}", color="#0F1E3D", anchor=(0.5, 1.4))
            t.setPos(x, y); t.setScale(0.9); p_train.addItem(t)
        p_train.getViewBox().autoRange(padding=0.15)

    def _apply_oma():
        st["oma"] = _table_to_layout()
        errs = st["oma"].validate()
        _draw_train()
        if errs:
            lbl_oval.setText("⚠ " + " · ".join(errs[:3]))
            lbl_oval.setStyleSheet(f"color:{RED};font-weight:700;")
        else:
            lbl_oval.setText(f"✅ {st['oma'].n_channels()} canales · "
                             f"{len(st['oma'].references())} referencia(s) · config válida")
            lbl_oval.setStyleSheet(f"color:{GREEN};font-weight:700;")

    def _load_tmpl():
        st["oma"] = default_24ch_layout(name=e_oname.text() or "Tren Motor-Bomba")
        _fill_points(); _draw_train(); _apply_oma()

    def _add_point():
        n = tbl_pts.rowCount()
        _add_point_row(MeasPoint(idx=n + 1, component="Motor", position_ref=POSITION_REFS[0],
                                 dof="+Y", module_slot=n // 4 + 1, channel_index=n % 4,
                                 x_norm=0.5, y_norm=0.0))
    btn_tmpl.clicked.connect(_load_tmpl)
    btn_addp.clicked.connect(_add_point)
    btn_draw.clicked.connect(_apply_oma)
    _fill_points(); _draw_train(); _apply_oma()

    # =================================================================
    # TAB 5 — OMA capture & analyze
    # =================================================================
    pg_oc = QtWidgets.QWidget(); cl2 = QtWidgets.QVBoxLayout(pg_oc)
    crow = QtWidgets.QHBoxLayout()
    btn_ocap = QtWidgets.QPushButton("▶ Capturar (simulado) + FDD"); btn_ocap.setStyleSheet(
        f"QPushButton{{background:{ACC};font-size:14px;padding:10px 20px;}} QPushButton:hover{{background:#1490c2;}}")
    lbl_ocap = QtWidgets.QLabel("OMA operacional: captura continua multicanal → FDD (valores singulares) → modos.")
    crow.addWidget(btn_ocap); crow.addWidget(lbl_ocap); crow.addStretch(1)
    cl2.addLayout(crow)

    ocsplit = QtWidgets.QHBoxLayout()
    tbl_omodes = QtWidgets.QTableWidget(0, 4)
    tbl_omodes.setHorizontalHeaderLabels(["Freq (Hz)", "Damping (%)", "Complexity (%)", "Clase"])
    tbl_omodes.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tbl_omodes.verticalHeader().setVisible(False); tbl_omodes.setMaximumWidth(520)
    p_svd = pg.PlotWidget(); p_svd.setBackground("w")
    p_svd.setLabel("left", "Magnitud (dB)"); p_svd.setLabel("bottom", "Frecuencia", "Hz")
    p_svd.setTitle("Valores singulares (FDD)", color=NAVY); p_svd.showGrid(x=True, y=True, alpha=0.3)
    svd_curve = p_svd.plot([], [], pen=pg.mkPen(NAVY, width=1.6))
    ocsplit.addWidget(tbl_omodes, 2); ocsplit.addWidget(p_svd, 3)
    cl2.addLayout(ocsplit, 1)
    lbl_ostat = QtWidgets.QLabel(""); cl2.addWidget(lbl_ostat)
    tabs.addTab(pg_oc, "OMA capture")

    def _oma_capture():
        from scipy.signal import lfilter
        lay = st["oma"]; fs = lay.fs_hz; nch = lay.n_channels()
        if nch < 2:
            QtWidgets.QMessageBox.information(win, "OMA", "Configurá al menos 2 canales activos en OMA setup.")
            return
        secs = min(float(lay.duration_s), 60.0)          # demo: hasta 60 s
        N = int(secs * fs); rng = st["rng"]
        lbl_ostat.setText(f"Capturando {secs:.0f} s @ {fs:.0f} Hz · {nch} canales …"); QtWidgets.QApplication.processEvents()
        modes = [(19.4, 0.020), (38.8, 0.015), (77.4, 0.012), (129.9, 0.010)]
        data = np.zeros((N, nch))
        for fn, z in modes:
            wn = 2 * np.pi * fn; wd = wn * (1 - z * z) ** 0.5
            r = np.exp(-z * wn / fs); th = wd / fs
            q = lfilter([1.0], [1.0, -2 * r * np.cos(th), r * r], rng.standard_normal(N))
            q /= (np.std(q) or 1.0)
            data += np.outer(q, rng.standard_normal(nch))
        data += 0.05 * rng.standard_normal((N, nch))
        fmax = min(fs / 2.56, 200.0)
        fdd = run_oma(data, fs, nperseg=4096, f_min_hz=5.0, f_max_hz=fmax,
                      channel_names=lay.channel_names())
        st["oma_fdd"] = fdd
        # SVD plot
        freqs = fdd.frequencies_hz; sv = np.asarray(fdd.singular_values)
        if sv.ndim == 1:
            sv = sv[None, :]
        band = freqs <= fmax
        svd_curve.setData(freqs[band], 10 * np.log10(np.maximum(sv[0][band], 1e-30)))
        # modes table
        tbl_omodes.setRowCount(0)
        for m in fdd.modes:
            rr = tbl_omodes.rowCount(); tbl_omodes.insertRow(rr)
            for c, v in enumerate([f"{m.natural_frequency_hz:.2f}", f"{m.damping_ratio_pct:.3f}",
                                   f"{m.complexity_pct:.1f}", m.classification]):
                tbl_omodes.setItem(rr, c, QtWidgets.QTableWidgetItem(v))
        lbl_ostat.setText(f"✅ FDD listo — {len(fdd.modes)} modos identificados · "
                          f"exportá al web (Watermelon System → reporte OMA SIGA).")
        lbl_ostat.setStyleSheet(f"color:{GREEN};font-weight:700;")
    btn_ocap.clicked.connect(_oma_capture)

    return app, win


def main(argv=None):
    ap = argparse.ArgumentParser(description="Watermelon Modal — impact test (native)")
    ap.add_argument("--sim", action="store_true", default=True, help="simulated mode (default)")
    ap.add_argument("--fs", type=float, default=2048.0)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--name", default="Test specimen")
    args = ap.parse_args(argv)

    setup = ModalSetup.default()
    setup.fs = args.fs; setup.block_size = args.block; setup.name = args.name

    try:
        app, win = build_app(setup, simulated=True)
        win.show()
        sys.exit(app.exec())
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
