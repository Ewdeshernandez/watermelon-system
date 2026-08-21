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
    # Máquina simulada editable (archivo JSON de la biblioteca) — v0.4
    if getattr(args, "machine_file", ""):
        from core.remote_monitoring.sim_machine import SimMachine
        m = SimMachine.load(args.machine_file)
        args.fs = m.fs                     # la máquina manda el muestreo
        args.machine = m.name
        return AcqAgent(SimulatedStreamSource(m.to_stream_config()), instance_id=m.name)

    # Banco de pruebas: escenario sim con nombre (proximidad/accel/velocidad/faults)
    if getattr(args, "scenario", ""):
        from core.remote_monitoring.sim_scenarios import build_scenario
        cfg = build_scenario(args.scenario, fs=args.fs)
        return AcqAgent(SimulatedStreamSource(cfg), instance_id=args.machine)

    idxs = [int(x) for x in args.chans.split(",") if x.strip() != ""]
    names = [s.strip() for s in args.names.split(",")]
    coup = "DC" if args.prox else "IEPE"
    units = "mil pp" if args.prox else "g rms"
    chans = []
    for k, i in enumerate(idxs):
        chans.append(ChannelConfig(name=names[k] if k < len(names) else f"CH{i}", coupling=coup,
                                   sensitivity_mv_per_eu=args.sens, bnc_port=(i + 1), units=units))
    if args.kph_bnc:
        chans.append(ChannelConfig(name="KPH", coupling="DC", sensitivity_mv_per_eu=1.0,
                                   bnc_port=args.kph_bnc, units="pulses/rev"))
    cfg = StreamConfig(sample_rate_hz=args.fs, channels=chans, block_seconds=0.1,
                       buffer_seconds=4.0, rpm=args.rpm, chassis_name=args.chassis,
                       defect=(getattr(args, "defect", "") or "none"))
    if args.sim or getattr(args, "defect", ""):
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
    ap.add_argument("--scenario", default="",
                    help="banco de pruebas sim: prox_6brg, prox_runup_whip, accel_bpfo, "
                         "accel_bpfi, accel_bsf, gear_mesh, vel_iso_6brg, coastdown, ...")
    ap.add_argument("--defect", default="",
                    help="inyecta defecto (modo manual --sim): unbalance|misalignment|"
                         "looseness|rub|oil_whirl|bearing_bpfo|bearing_bpfi|bearing_bsf|gear_mesh")
    ap.add_argument("--machine-file", default="",
                    help="corre una máquina simulada guardada (JSON de la biblioteca) — v0.4")
    ap.add_argument("--machine", default="Rotor_Kit_Field")
    ap.add_argument("--chassis", default="cDAQ1")
    ap.add_argument("--chans", default="0,1")
    ap.add_argument("--names", default="1YA,1XA")
    ap.add_argument("--sens", type=float, default=100.0)
    ap.add_argument("--fs", type=float, default=5120.0)
    ap.add_argument("--rpm", type=float, default=1475.0)
    ap.add_argument("--prox", action="store_true")
    ap.add_argument("--kph-bnc", type=int, default=0)
    ap.add_argument("--alarm", type=float, default=0.0, help="Nivel de alarma (unid. del canal)")
    ap.add_argument("--danger", type=float, default=0.0, help="Nivel de peligro (unid. del canal)")
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
    from core.remote_monitoring.transient import TransientCapture, TransientConfig
    from core.remote_monitoring import analysis as diag
    kph_glob = agent.source.config.keyphasor_index()
    from core.remote_monitoring.sim_machine import MODES, MODE_TO_PROFILE
    tc = TransientCapture(TransientConfig(fmax_hz=min(2000.0, args.fs / 2.5)))

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
    tb.addSeparator()
    lbl_modo = QtWidgets.QLabel(" Modo: "); lbl_modo.setStyleSheet("color:white;")
    tb.addWidget(lbl_modo)
    cb_run = QtWidgets.QComboBox(); cb_run.addItems(MODES); tb.addWidget(cb_run)
    spacer = QtWidgets.QWidget(); spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                                       QtWidgets.QSizePolicy.Preferred)
    tb.addWidget(spacer)
    lbl_machine = QtWidgets.QLabel(f"  {args.machine}   ")
    lbl_machine.setStyleSheet("color:white; font-weight:700; font-size:14px;")
    tb.addWidget(lbl_machine)

    # ---------------- Pestañas ----------------
    tabs = QtWidgets.QTabWidget(); win.setCentralWidget(tabs)

    # --- Configuración (editor de máquina simulada — v0.4) ---
    from core.remote_monitoring.sim_machine import (SimMachine, SensorSpec, MODES, PHENOMENA,
                                                    MODE_TO_PROFILE, save_to_library,
                                                    list_machines, load_from_library)
    _KINDS = ["prox", "vel", "accel", "keyphasor"]

    def _machine_from_agent() -> SimMachine:
        """Máquina inicial a partir de los canales con que arrancó el app."""
        sens = []
        for c in agent.channels:
            if is_keyphasor_channel(c):
                k = "keyphasor"
            elif "mil" in (c.units or "").lower():
                k = "prox"
            elif "mm/s" in (c.units or "").lower():
                k = "vel"
            else:
                k = "accel"
            sens.append(SensorSpec(c.name, k, int(c.bnc_port or 1),
                                   float(c.sensitivity_mv_per_eu or 100.0)))
        cf = agent.source.config
        return SimMachine(name=args.machine, fs=float(args.fs), sensors=sens,
                          rpm=float(getattr(cf, "rpm", 3000.0)),
                          crit1=float(getattr(cf, "sim_critical_rpm", 0.0)),
                          crit2=float(getattr(cf, "sim_critical_rpm2", 0.0)),
                          phenomena=dict(getattr(cf, "defect_by_kind", {}) or {}),
                          severity=float(getattr(cf, "sim_severity", 1.0)))

    cfg_w = QtWidgets.QWidget(); cfg_l = QtWidgets.QVBoxLayout(cfg_w)
    # fila 1: nombre, fs, biblioteca
    r1 = QtWidgets.QHBoxLayout()
    r1.addWidget(QtWidgets.QLabel("Máquina:"))
    ed_name = QtWidgets.QLineEdit(); r1.addWidget(ed_name, 2)
    r1.addWidget(QtWidgets.QLabel("Muestreo (Hz):"))
    sp_fs = QtWidgets.QSpinBox(); sp_fs.setRange(256, 102400); sp_fs.setSingleStep(1280); r1.addWidget(sp_fs)
    r1.addWidget(QtWidgets.QLabel("Biblioteca:"))
    cb_lib = QtWidgets.QComboBox(); cb_lib.setMinimumWidth(160); r1.addWidget(cb_lib)
    btn_load = QtWidgets.QPushButton("Cargar"); r1.addWidget(btn_load)
    cfg_l.addLayout(r1)
    # fila 2: operación
    r2 = QtWidgets.QHBoxLayout()
    r2.addWidget(QtWidgets.QLabel("Modo:"))
    cb_mode = QtWidgets.QComboBox(); cb_mode.addItems(MODES); r2.addWidget(cb_mode)
    def _dsp(mn, mx, val, step=100.0):
        s = QtWidgets.QDoubleSpinBox(); s.setRange(mn, mx); s.setValue(val); s.setSingleStep(step); return s
    r2.addWidget(QtWidgets.QLabel("RPM:")); sp_rpm = _dsp(0, 60000, 3000); r2.addWidget(sp_rpm)
    r2.addWidget(QtWidgets.QLabel("Arranque→")); sp_r0 = _dsp(0, 60000, 300); r2.addWidget(sp_r0)
    sp_r1 = _dsp(0, 60000, 6000); r2.addWidget(sp_r1)
    r2.addWidget(QtWidgets.QLabel("Rampa(s):")); sp_ramp = _dsp(1, 3600, 90, 5); r2.addWidget(sp_ramp)
    cfg_l.addLayout(r2)
    # fila 3: críticas, severidad, fenómenos
    r3 = QtWidgets.QHBoxLayout()
    r3.addWidget(QtWidgets.QLabel("Crítica 1:")); sp_c1 = _dsp(0, 60000, 0); r3.addWidget(sp_c1)
    r3.addWidget(QtWidgets.QLabel("Crítica 2:")); sp_c2 = _dsp(0, 60000, 0); r3.addWidget(sp_c2)
    r3.addWidget(QtWidgets.QLabel("Severidad:")); sp_sev = _dsp(0, 3, 1.0, 0.25); r3.addWidget(sp_sev)
    r3.addWidget(QtWidgets.QLabel("Fenómeno prox:"))
    cb_ph_p = QtWidgets.QComboBox(); cb_ph_p.addItems(PHENOMENA["prox"]); r3.addWidget(cb_ph_p)
    r3.addWidget(QtWidgets.QLabel("vel:"))
    cb_ph_v = QtWidgets.QComboBox(); cb_ph_v.addItems(PHENOMENA["vel"]); r3.addWidget(cb_ph_v)
    r3.addWidget(QtWidgets.QLabel("accel:"))
    cb_ph_a = QtWidgets.QComboBox(); cb_ph_a.addItems(PHENOMENA["accel"]); r3.addWidget(cb_ph_a)
    cfg_l.addLayout(r3)
    # tabla de sensores (editable)
    tblc = QtWidgets.QTableWidget(0, 5)
    tblc.setHorizontalHeaderLabels(["Canal", "Tipo", "BNC", "Sensib (mV/EU)", "Ángulo°"])
    tblc.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    cfg_l.addWidget(tblc, 1)
    rb = QtWidgets.QHBoxLayout()
    btn_add = QtWidgets.QPushButton("+ Sensor"); btn_del = QtWidgets.QPushButton("– Quitar")
    btn_tpl = QtWidgets.QPushButton("Plantilla motor+bomba")
    btn_save = QtWidgets.QPushButton("💾 Guardar en biblioteca")
    btn_apply = QtWidgets.QPushButton("▶ Aplicar y medir")
    for b in (btn_add, btn_del, btn_tpl): rb.addWidget(b)
    rb.addStretch(1); rb.addWidget(btn_save); rb.addWidget(btn_apply)
    cfg_l.addLayout(rb)
    tabs.addTab(cfg_w, "Configuración")

    def _add_sensor_row(s: SensorSpec):
        r = tblc.rowCount(); tblc.insertRow(r)
        tblc.setItem(r, 0, QtWidgets.QTableWidgetItem(s.name))
        cbk = QtWidgets.QComboBox(); cbk.addItems(_KINDS); cbk.setCurrentText(s.kind)
        tblc.setCellWidget(r, 1, cbk)
        tblc.setItem(r, 2, QtWidgets.QTableWidgetItem(str(s.bnc)))
        tblc.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{s.sensitivity:g}"))
        tblc.setItem(r, 4, QtWidgets.QTableWidgetItem(f"{s.angle:g}"))

    def fill_form(m: SimMachine):
        ed_name.setText(m.name); sp_fs.setValue(int(m.fs))
        cb_mode.setCurrentText(m.mode if m.mode in MODES else "estable")
        sp_rpm.setValue(m.rpm); sp_r0.setValue(m.rpm_start); sp_r1.setValue(m.rpm_end)
        sp_ramp.setValue(m.ramp_s); sp_c1.setValue(m.crit1); sp_c2.setValue(m.crit2)
        sp_sev.setValue(m.severity)
        cb_ph_p.setCurrentText(m.phenomena.get("prox", "none"))
        cb_ph_v.setCurrentText(m.phenomena.get("vel", "none"))
        cb_ph_a.setCurrentText(m.phenomena.get("accel", "none"))
        tblc.setRowCount(0)
        for s in m.sensors: _add_sensor_row(s)

    def read_form() -> SimMachine:
        sens = []
        for r in range(tblc.rowCount()):
            nm = tblc.item(r, 0).text() if tblc.item(r, 0) else f"CH{r}"
            kd = tblc.cellWidget(r, 1).currentText() if tblc.cellWidget(r, 1) else "accel"
            def _num(col, dv):
                try: return float(tblc.item(r, col).text())
                except Exception: return dv
            sens.append(SensorSpec(nm, kd, int(_num(2, r + 1)), _num(3, 100.0), _num(4, 0.0)))
        ph = {"prox": cb_ph_p.currentText(), "vel": cb_ph_v.currentText(), "accel": cb_ph_a.currentText()}
        return SimMachine(name=ed_name.text() or "Maquina", fs=float(sp_fs.value()), sensors=sens,
                          mode=cb_mode.currentText(), rpm=sp_rpm.value(),
                          rpm_start=sp_r0.value(), rpm_end=sp_r1.value(), ramp_s=sp_ramp.value(),
                          crit1=sp_c1.value(), crit2=sp_c2.value(), severity=sp_sev.value(),
                          phenomena={k: v for k, v in ph.items() if v != "none"})

    def refresh_lib():
        cb_lib.clear(); cb_lib.addItems(list_machines() or ["(vacía)"])

    def do_load_lib():
        nm = cb_lib.currentText()
        if nm and nm != "(vacía)":
            fill_form(load_from_library(nm))

    def do_save_lib():
        m = read_form()
        try:
            save_to_library(m); refresh_lib()
            QtWidgets.QMessageBox.information(win, "Biblioteca", f"Máquina '{m.name}' guardada.")
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(win, "Biblioteca", f"No se pudo guardar: {e}")

    def do_apply():
        """Guarda la máquina y RELANZA el app midiéndola (evita reconstruir plots)."""
        import subprocess, tempfile
        m = read_form()
        path = os.path.join(tempfile.gettempdir(), "wm_apply_machine.json")
        m.save(path)
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--machine-file", path]
        else:
            cmd = [sys.executable, os.path.abspath(__file__), "--machine-file", path]
        try:
            subprocess.Popen(cmd)
            win.close()
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(win, "Aplicar", f"No se pudo relanzar: {e}")

    btn_add.clicked.connect(lambda: _add_sensor_row(SensorSpec("CHn", "accel", tblc.rowCount() + 1)))
    btn_del.clicked.connect(lambda: tblc.removeRow(tblc.currentRow()) if tblc.currentRow() >= 0 else None)
    btn_tpl.clicked.connect(lambda: fill_form(SimMachine.plantilla_motor_bomba()))
    btn_load.clicked.connect(do_load_lib)
    btn_save.clicked.connect(do_save_lib)
    btn_apply.clicked.connect(do_apply)
    refresh_lib(); fill_form(_machine_from_agent())

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
    if args.alarm or args.danger:
        tab_l.addWidget(QtWidgets.QLabel(
            f"<i>Alarma {args.alarm:g} · Peligro {args.danger:g} {vib[0][1].units} "
            f"(ISO 20816). OK / ALERTA / PELIGRO por Overall.</i>"))
    tabs.addTab(tab_w, "Tabular")

    # --- Órbita (par X/Y, vivo) ---
    orb_ok = len(vib) >= 2
    if orb_ok:
        orb_w = QtWidgets.QWidget(); orb_l = QtWidgets.QVBoxLayout(orb_w)
        xy = QtWidgets.QHBoxLayout()
        xy.addWidget(QtWidgets.QLabel("X:"))
        cb_x = QtWidgets.QComboBox(); cb_x.addItems([c.name for _, c in vib]); cb_x.setCurrentIndex(0)
        xy.addWidget(cb_x); xy.addWidget(QtWidgets.QLabel("Y:"))
        cb_y = QtWidgets.QComboBox(); cb_y.addItems([c.name for _, c in vib]); cb_y.setCurrentIndex(1)
        xy.addWidget(cb_y); xy.addStretch(1); orb_l.addLayout(xy)
        orb_plot = pg.PlotWidget(); orb_plot.setAspectLocked(True)
        orb_plot.showGrid(x=True, y=True, alpha=0.25)
        orb_plot.addLine(x=0, pen=pg.mkPen("#c9d2e0")); orb_plot.addLine(y=0, pen=pg.mkPen("#c9d2e0"))
        orb_curve = orb_plot.plot(pen=pg.mkPen(CORN, width=1.4))
        orb_kph = orb_plot.plot(pen=None, symbol="o", symbolBrush="#dc2626", symbolSize=9)
        orb_l.addWidget(orb_plot, 1)
        tabs.addTab(orb_w, "Órbita")

    # --- Bode (amp + fase vs rpm, se llena en runup) ---
    bode_w = QtWidgets.QWidget(); bode_l = QtWidgets.QVBoxLayout(bode_w)
    bh = QtWidgets.QHBoxLayout(); bh.addWidget(QtWidgets.QLabel("Canal:"))
    cb_bode = QtWidgets.QComboBox(); cb_bode.addItems([c.name for _, c in vib]); bh.addWidget(cb_bode)
    bh.addStretch(1); bode_l.addLayout(bh)
    gl_b = pg.GraphicsLayoutWidget(); bode_l.addWidget(gl_b, 1)
    p_ph = gl_b.addPlot(row=0, col=0); p_ph.setLabel("left", "Fase 1X (°)"); p_ph.showGrid(x=True, y=True, alpha=0.25)
    p_ph.getAxis("bottom").setStyle(showValues=False); p_ph.invertY(True)
    c_ph = p_ph.plot(pen=pg.mkPen(CORN, width=1.6))
    p_am = gl_b.addPlot(row=1, col=0); p_am.setLabel("left", "1X"); p_am.setLabel("bottom", "RPM")
    p_am.showGrid(x=True, y=True, alpha=0.25)
    c_am = p_am.plot(pen=pg.mkPen(CORN, width=1.6))
    tabs.addTab(bode_w, "Bode")

    # --- Cascada (espectros apilados) ---
    casc_w = QtWidgets.QWidget(); casc_l = QtWidgets.QVBoxLayout(casc_w)
    ch2 = QtWidgets.QHBoxLayout(); ch2.addWidget(QtWidgets.QLabel("Canal:"))
    cb_casc = QtWidgets.QComboBox(); cb_casc.addItems([c.name for _, c in vib]); ch2.addWidget(cb_casc)
    ch2.addStretch(1); casc_l.addLayout(ch2)
    p_casc = pg.PlotWidget(); p_casc.setLabel("bottom", "Frecuencia (Hz)"); p_casc.setLabel("left", "RPM")
    p_casc.showGrid(x=True, y=True, alpha=0.2)
    casc_curves = [p_casc.plot(pen=pg.mkPen(CORN, width=0.9)) for _ in range(40)]
    casc_l.addWidget(p_casc, 1)
    tabs.addTab(casc_w, "Cascada")

    # --- Diagnóstico (whirl/whip + críticas) ---
    diag_w = QtWidgets.QWidget(); diag_l = QtWidgets.QVBoxLayout(diag_w)
    btn_diag = QtWidgets.QPushButton("🔎 Diagnosticar (API 684)")
    diag_txt = QtWidgets.QTextEdit(); diag_txt.setReadOnly(True)
    diag_l.addWidget(btn_diag); diag_l.addWidget(diag_txt, 1)
    tabs.addTab(diag_w, "Diagnóstico")

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
        if not rpm:      # sin keyphasor → rpm del pico dominante de vibración (1X)
            fr0, mag0 = _spectrum(snap[vib[0][0]], fs)
            band = (fr0 > 3) & (fr0 < 300)
            if band.any() and float(mag0[band].max()) > 1e-9:
                rpm = float(fr0[band][np.argmax(mag0[band])] * 60.0)
        lbl_rpm.setText(f"RPM: {rpm:.0f}" if rpm else "RPM: —")
        f1 = (rpm / 60.0) if rpm else None
        # Alimentar el capturador de transitorio (throttle ~cada 0.5 s) para Bode/Cascada.
        rec_state["fn"] = rec_state.get("fn", 0) + 1
        if rpm and rec_state["fn"] % 8 == 0:
            try:
                tc.feed(snap, rpm, fs, vib, kph_idx=kph_glob)
            except Exception:  # noqa: BLE001
                pass
        cur = tabs.tabText(tabs.currentIndex())
        if cur == "Monitoreo":
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
        elif cur == "Tabular":
            for r, (i, c) in enumerate(vib):
                eu = snap[i] * 1000.0 / (c.sensitivity_mv_per_eu or 1.0)
                ov = float(np.sqrt(np.mean((eu - eu.mean()) ** 2)))
                a1 = ph = 0.0
                if f1:
                    a1, ph = one_x_vector(eu - eu.mean(), fs, f1)
                if args.danger and ov >= args.danger:
                    estado, col = "PELIGRO", QtGui.QColor("#fde2e2")
                elif args.alarm and ov >= args.alarm:
                    estado, col = "ALERTA", QtGui.QColor("#fdf0d5")
                else:
                    estado, col = "OK", QtGui.QColor("#e6f4ea")
                vals = [c.name, f"{ov:.3g} {c.units}", f"{a1:.3g}", f"{ph:.0f}°", estado]
                for cc, v in enumerate(vals):
                    it = QtWidgets.QTableWidgetItem(v)
                    if cc == 4:
                        it.setBackground(col)
                    tblt.setItem(r, cc, it)
        elif orb_ok and cur == "Órbita":
            xi = next((i for i, c in vib if c.name == cb_x.currentText()), vib[0][0])
            yi = next((i for i, c in vib if c.name == cb_y.currentText()), vib[1][0])
            nrev = min(snap.shape[1], int((12 * fs / max(rpm, 1)) * 60) if rpm else int(0.3 * fs))
            sx = next(cc for ii, cc in vib if ii == xi)
            sy = next(cc for ii, cc in vib if ii == yi)
            x = snap[xi, -nrev:] * 1000.0 / (sx.sensitivity_mv_per_eu or 1.0)
            y = snap[yi, -nrev:] * 1000.0 / (sy.sensitivity_mv_per_eu or 1.0)
            x = x - x.mean(); y = y - y.mean()
            orb_curve.setData(x, y)
            if f1 and len(x):
                spr = max(1, int(fs / f1))
                orb_kph.setData(x[::spr], y[::spr])
        elif cur == "Bode":
            rr, am, ph = tc.bode(cb_bode.currentText())
            if len(rr):
                c_am.setData(rr, np.asarray(am) * 2.0)      # pp aprox
                c_ph.setData(rr, np.asarray(ph))
        elif cur == "Cascada":
            rr, fr, mat = tc.cascade(cb_casc.currentText())
            for cv in casc_curves:
                cv.setData([], [])
            if len(rr) >= 2 and mat.size:
                idx = np.unique(np.linspace(0, len(rr) - 1, min(len(casc_curves), len(rr)))
                                .round().astype(int))
                span = float(rr[-1] - rr[0]) or 1.0
                pk = float(mat.max()) or 1.0
                sc = (span / max(1, len(idx))) * 1.5 / pk
                for cv, i in zip(casc_curves, idx):
                    cv.setData(fr, rr[i] + mat[i] * sc)

    def run_diag():
        rr, fr, mat = tc.cascade(cb_casc.currentText())
        if len(rr) < 3:
            diag_txt.setHtml("<i>Corré un runup (variá la velocidad) para tener datos que diagnosticar.</i>")
            return
        rb, ab, _ph = tc.bode(cb_casc.currentText())
        crit = [float(rb[i]) for i in diag.detect_criticals(np.asarray(rb, float), np.asarray(ab, float))]
        found = diag.cascade_diagnosis(rr, fr, mat, crit)
        html = ["<h3>🔎 Auto-diagnóstico (API 684)</h3>"]
        col = {"info": "#2f6fb0", "warn": "#b45309", "danger": "#b91c1c"}
        for lvl, title, detail in found:
            html.append(f"<p style='color:{col.get(lvl,'#333')}'><b>{title}</b><br>{detail}</p>")
        diag_txt.setHtml("".join(html) if found else "<i>Sin hallazgos.</i>")

    timer = QtCore.QTimer(); timer.timeout.connect(update)
    btn_diag.clicked.connect(run_diag)

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

    def set_mode_live(mode):
        """Cambia el MODO de operación en vivo (estable/arranque/parada) sobre la
        misma máquina: ajusta el perfil, rebobina el reloj del rotor y reinicia el
        capturador de transitorio para que Bode/Cascada salgan limpios."""
        nonlocal tc
        cf = agent.source.config
        if hasattr(cf, "speed_profile"):
            cf.speed_profile = MODE_TO_PROFILE.get(mode, "constant")
        if hasattr(agent.source, "rewind"):
            agent.source.rewind()
        tc = TransientCapture(TransientConfig(fmax_hz=min(2000.0, agent.sample_rate_hz / 2.5)))
        for cv in casc_curves:
            cv.setData([], [])
        c_am.setData([], []); c_ph.setData([], [])
        lbl_state.setText(f"● {mode}" if agent.source.is_running() else f"modo: {mode}")

    _prof0 = getattr(agent.source.config, "speed_profile", "constant")
    _inv = {v: k for k, v in MODE_TO_PROFILE.items()}
    cb_run.blockSignals(True); cb_run.setCurrentText(_inv.get(_prof0, "estable")); cb_run.blockSignals(False)
    cb_run.currentTextChanged.connect(set_mode_live)
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
