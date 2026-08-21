"""
Watermelon Field — módulo NATIVO industrial de adquisición y monitoreo
======================================================================

App de escritorio (PySide6 + pyqtgraph): menú, barra de herramientas, pestañas
(Configuración · Monitoreo · Tabular · Espectro) y barra de estado — estilo
estación de análisis (System1/ADRE), pero abierto y con nube. Reusa el motor de
core/remote_monitoring. Tiempo real nativo, sin navegador.

Correr:
    python native/watermelon_field.py --sim          # demo (Mac/dev)
    python native/watermelon_field.py --sens 100 --fs 5120   # NI real (Windows)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.remote_monitoring.agent import AcqAgent
from core.remote_monitoring.stream_source import (ChannelConfig, StreamConfig,
                                                  SimulatedStreamSource, is_keyphasor_channel)

NAVY = "#0F1E3D"
BLUE = "#2f6fb0"
CORN = "#4f8fd0"


def build_agent(args) -> AcqAgent:
    idxs = [int(x) for x in args.chans.split(",") if x.strip() != ""]
    names = [s.strip() for s in args.names.split(",")]
    coup = "AC" if args.prox else "IEPE"
    units = "mil pp" if args.prox else "g rms"
    chans = []
    for k, i in enumerate(idxs):
        chans.append(ChannelConfig(name=names[k] if k < len(names) else f"CH{i}", coupling=coup,
                                   sensitivity_mv_per_eu=args.sens, bnc_port=(i + 1), units=units))
    if args.kph_bnc:
        chans.append(ChannelConfig(name="KPH", coupling="DC", sensitivity_mv_per_eu=1.0,
                                   bnc_port=args.kph_bnc, units="pulses/rev"))
    cfg = StreamConfig(sample_rate_hz=args.fs, channels=chans, block_seconds=0.1,
                       buffer_seconds=4.0, rpm=args.rpm, chassis_name=args.chassis)
    if args.sim:
        source = SimulatedStreamSource(cfg)
    else:
        from core.remote_monitoring.ni_stream_source import NIStreamSource
        source = NIStreamSource(cfg)
    return AcqAgent(source, instance_id=args.machine)


def _spectrum(x, fs):
    x = x - np.mean(x)
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w)) / (np.sum(w) / 2)
    return np.fft.rfftfreq(len(x), 1.0 / fs), mag


def _stylesheet() -> str:
    return f"""
    QMainWindow, QWidget {{ background: #f4f7fb; color: #1f2937;
        font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }}
    QMenuBar {{ background: {NAVY}; color: white; }}
    QMenuBar::item:selected {{ background: {BLUE}; }}
    QMenu {{ background: white; border: 1px solid #d6deea; }}
    QMenu::item:selected {{ background: {CORN}; color: white; }}
    QToolBar {{ background: {NAVY}; spacing: 6px; padding: 5px; border: none; }}
    QToolBar QToolButton {{ color: white; padding: 6px 12px; border-radius: 6px; }}
    QToolBar QToolButton:hover {{ background: {BLUE}; }}
    QTabWidget::pane {{ border: 1px solid #d6deea; background: white; }}
    QTabBar::tab {{ background: #e7edf6; padding: 8px 18px; margin-right: 2px;
        border-top-left-radius: 6px; border-top-right-radius: 6px; }}
    QTabBar::tab:selected {{ background: white; color: {NAVY}; font-weight: 700;
        border-bottom: 2px solid {CORN}; }}
    QStatusBar {{ background: {NAVY}; color: #c7d6ea; }}
    QPushButton {{ background: {BLUE}; color: white; border: none; padding: 7px 14px;
        border-radius: 6px; }}
    QPushButton:hover {{ background: {NAVY}; }}
    QPushButton:disabled {{ background: #9fb3d1; }}
    QTableWidget {{ background: white; gridline-color: #e8edf5; }}
    QHeaderView::section {{ background: {NAVY}; color: white; padding: 6px; border: none; }}
    """


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--machine", default="Rotor_Kit_Field")
    ap.add_argument("--chassis", default="cDAQ1")
    ap.add_argument("--chans", default="0,1")
    ap.add_argument("--names", default="1YA,1XA")
    ap.add_argument("--sens", type=float, default=100.0)
    ap.add_argument("--fs", type=float, default=5120.0)
    ap.add_argument("--rpm", type=float, default=1475.0)
    ap.add_argument("--prox", action="store_true")
    ap.add_argument("--kph-bnc", type=int, default=0)
    args = ap.parse_args()

    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        import pyqtgraph as pg
    except ImportError:
        print("Instalá deps del nativo:  pip install -r native/requirements.txt")
        return 2

    agent = build_agent(args)
    vib = [(i, c) for i, c in enumerate(agent.channels) if not is_keyphasor_channel(c)]
    from core.remote_monitoring.recorder import TransientRecorder, upload_recording

    pg.setConfigOptions(antialias=True, background="w", foreground=NAVY)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(_stylesheet())

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Field — {args.machine}")
    win.resize(1360, 860)
    rec_state = {"rec": None}

    # ---------------- Menú ----------------
    mb = win.menuBar()
    m_file = mb.addMenu("&Archivo")
    m_view = mb.addMenu("&Ver")
    m_help = mb.addMenu("A&yuda")
    act_start = QtGui.QAction("▶ Iniciar", win)
    act_stop = QtGui.QAction("⏸ Detener", win); act_stop.setEnabled(False)
    act_rec = QtGui.QAction("⏺ Grabar", win); act_rec.setCheckable(True)
    act_quit = QtGui.QAction("Salir", win)
    for a in (act_start, act_stop, act_rec):
        m_file.addAction(a)
    m_file.addSeparator(); m_file.addAction(act_quit)
    act_about = QtGui.QAction("Acerca de Watermelon Field", win)
    m_help.addAction(act_about)

    # ---------------- Toolbar ----------------
    tb = win.addToolBar("Principal")
    tb.setMovable(False)
    tb.addAction(act_start); tb.addAction(act_stop); tb.addAction(act_rec)
    spacer = QtWidgets.QWidget(); spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                                       QtWidgets.QSizePolicy.Preferred)
    tb.addWidget(spacer)
    lbl_machine = QtWidgets.QLabel(f"  {args.machine}   ")
    lbl_machine.setStyleSheet("color:white; font-weight:700; font-size:14px;")
    tb.addWidget(lbl_machine)

    # ---------------- Pestañas ----------------
    tabs = QtWidgets.QTabWidget(); win.setCentralWidget(tabs)

    # --- Configuración ---
    cfg_w = QtWidgets.QWidget(); cfg_l = QtWidgets.QVBoxLayout(cfg_w)
    cfg_l.addWidget(QtWidgets.QLabel(f"<b>Máquina:</b> {args.machine}  ·  "
                                     f"<b>Muestreo:</b> {args.fs:.0f} Hz  ·  "
                                     f"<b>Chasis:</b> {args.chassis}"))
    tblc = QtWidgets.QTableWidget(len(agent.channels), 5)
    tblc.setHorizontalHeaderLabels(["Canal", "BNC", "Tipo", "Sensib (mV/EU)", "Acople"])
    tblc.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    for r, c in enumerate(agent.channels):
        tipo = "keyphasor" if is_keyphasor_channel(c) else ("proximidad" if c.coupling == "AC"
                                                            else "acelerómetro")
        for col, val in enumerate([c.name, str(c.bnc_port), tipo,
                                    f"{c.sensitivity_mv_per_eu:g}", c.coupling]):
            it = QtWidgets.QTableWidgetItem(val); it.setFlags(QtCore.Qt.ItemIsEnabled)
            tblc.setItem(r, col, it)
    cfg_l.addWidget(tblc)
    cfg_l.addWidget(QtWidgets.QLabel("<i>(La edición de canales / carga desde la nube llega en v0.4.)</i>"))
    tabs.addTab(cfg_w, "Configuración")

    # --- Monitoreo (scope live) ---
    mon_w = QtWidgets.QWidget(); mon_l = QtWidgets.QVBoxLayout(mon_w)
    top = QtWidgets.QHBoxLayout()
    top.addWidget(QtWidgets.QLabel("Espectro de:"))
    combo = QtWidgets.QComboBox(); combo.addItems([c.name for _, c in vib]); top.addWidget(combo)
    top.addStretch(1); mon_l.addLayout(top)
    gl = pg.GraphicsLayoutWidget(); mon_l.addWidget(gl, 1)
    wave_curves = []
    for r, (i, c) in enumerate(vib):
        p = gl.addPlot(row=r, col=0); p.showGrid(x=True, y=True, alpha=0.25)
        p.setLabel("left", c.name)
        p.getAxis("bottom").setStyle(showValues=(r == len(vib) - 1))
        wave_curves.append(p.plot(pen=pg.mkPen(CORN, width=1.3)))
    p_spec = gl.addPlot(row=len(vib), col=0); p_spec.showGrid(x=True, y=True, alpha=0.25)
    p_spec.setLabel("left", "amplitud"); p_spec.setLabel("bottom", "Frecuencia (Hz)")
    spec_curve = p_spec.plot(pen=pg.mkPen(CORN, width=1.3))
    v1x = pg.InfiniteLine(angle=90, pen=pg.mkPen("#e26d6d", width=1, style=QtCore.Qt.DashLine))
    p_spec.addItem(v1x)
    tabs.addTab(mon_w, "Monitoreo")

    # --- Tabular (valores actuales, vivo) ---
    tab_w = QtWidgets.QWidget(); tab_l = QtWidgets.QVBoxLayout(tab_w)
    tblt = QtWidgets.QTableWidget(len(vib), 5)
    tblt.setHorizontalHeaderLabels(["Sensor", "Overall", "1X", "1X fase", "Estado"])
    tblt.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    tblt.verticalHeader().setVisible(False)
    tab_l.addWidget(tblt)
    tabs.addTab(tab_w, "Tabular")

    lbl_rpm = QtWidgets.QLabel("RPM: —"); lbl_state = QtWidgets.QLabel("detenido")
    lbl_rec = QtWidgets.QLabel("")
    win.statusBar().addWidget(lbl_state)
    win.statusBar().addPermanentWidget(lbl_rec)
    win.statusBar().addPermanentWidget(lbl_rpm)

    # ---------------- Lógica ----------------
    from core.remote_monitoring.keyphasor import one_x_vector

    def update():
        snap = agent.snapshot(2.0)
        if snap.shape[1] < 16:
            return
        fs = agent.sample_rate_hz
        rpm = agent.estimate_rpm(snap)
        lbl_rpm.setText(f"RPM: {rpm:.0f}" if rpm else "RPM: —")
        f1 = (rpm / 60.0) if rpm else None
        idx_tab = tabs.currentIndex()
        if idx_tab == 1:      # Monitoreo
            nshow = min(snap.shape[1], int(0.3 * fs))
            tms = np.arange(nshow) / fs * 1000.0
            for (i, c), curve in zip(vib, wave_curves):
                eu = snap[i, -nshow:] * 1000.0 / (c.sensitivity_mv_per_eu or 1.0)
                curve.setData(tms, eu - eu.mean())
            name = combo.currentText()
            sel = next((i for i, c in vib if c.name == name), vib[0][0])
            sens = next((c.sensitivity_mv_per_eu for i, c in vib if i == sel), 100.0)
            fr, mag = _spectrum(snap[sel] * 1000.0 / (sens or 1.0), fs)
            keep = fr <= min(fr[-1], 2000.0)
            spec_curve.setData(fr[keep], mag[keep])
            v1x.setPos(f1) if f1 else v1x.hide()
            if f1:
                v1x.show()
        elif idx_tab == 2:    # Tabular
            for r, (i, c) in enumerate(vib):
                eu = snap[i] * 1000.0 / (c.sensitivity_mv_per_eu or 1.0)
                ov = float(np.sqrt(np.mean((eu - eu.mean()) ** 2)))
                a1 = ph = 0.0
                if f1:
                    a1, ph = one_x_vector(eu - eu.mean(), fs, f1)
                vals = [c.name, f"{ov:.3g} {c.units}", f"{a1:.3g}", f"{ph:.0f}°", "OK"]
                for col, v in enumerate(vals):
                    tblt.setItem(r, col, QtWidgets.QTableWidgetItem(v))

    timer = QtCore.QTimer(); timer.timeout.connect(update)

    def do_start():
        try:
            agent.start()
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(win, "Error", f"No se pudo iniciar: {e}")
            return
        act_start.setEnabled(False); act_stop.setEnabled(True)
        lbl_state.setText("● adquiriendo (hilo de fondo)")
        timer.start(60)

    def do_stop():
        timer.stop()
        try:
            agent.stop()
        except Exception:  # noqa: BLE001
            pass
        act_start.setEnabled(True); act_stop.setEnabled(False)
        lbl_state.setText("detenido")
        if act_rec.isChecked():
            act_rec.setChecked(False)

    def do_rec(checked):
        if checked:
            ch_meta = [{"name": c.name, "units": c.units, "coupling": c.coupling,
                        "bnc_port": c.bnc_port, "sensitivity_mv_per_eu": float(c.sensitivity_mv_per_eu or 0)}
                       for c in agent.channels]
            rec = TransientRecorder(agent.instance_id, agent.sample_rate_hz, ch_meta, machine=args.machine)
            agent.on_block = rec.append
            rec_state["rec"] = rec
            lbl_rec.setText("🔴 GRABANDO")
        else:
            rec = rec_state.get("rec"); agent.on_block = None
            if rec:
                rec.stop()
                up = upload_recording(rec.dir)
                QtWidgets.QMessageBox.information(
                    win, "Grabación", f"{rec.rec_id} · {rec.status.duration_s:.0f}s · "
                    f"{rec.status.size_mb:.1f} MB · {'☁ nube' if up.get('ok') else 'local (pendiente)'}")
            rec_state["rec"] = None; lbl_rec.setText("")

    act_start.triggered.connect(do_start)
    act_stop.triggered.connect(do_stop)
    act_rec.toggled.connect(do_rec)
    act_quit.triggered.connect(win.close)
    act_about.triggered.connect(lambda: QtWidgets.QMessageBox.about(
        win, "Watermelon Field", "Watermelon Field — módulo nativo de adquisición.\n"
        "Rotodinámica API 670/684 · nube integrada.\n© SIGA"))

    win.show()
    ret = app.exec()
    try:
        agent.stop()
    except Exception:  # noqa: BLE001
        pass
    return ret


if __name__ == "__main__":
    sys.exit(main())
