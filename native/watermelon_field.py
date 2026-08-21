"""
Watermelon Field — módulo NATIVO industrial de adquisición y monitoreo en vivo
==============================================================================

App de escritorio (PySide6 + pyqtgraph) para el PC de campo. Tiempo real sólido,
SIN navegador → no se traba. Reusa TODO el motor de core/remote_monitoring
(adquisición NI, FFT, order tracking, diagnóstico). La web queda solo para
análisis/reportes.

Correr:
    python native/watermelon_field.py --sim          # demo (Mac/dev, sin hardware)
    python native/watermelon_field.py                # campo: NI real (Windows)
    python native/watermelon_field.py --mod cDAQ1Mod1 --chans 0,1 --sens 100 --fs 5120

v0.1 — base: waveforms + espectro (canal elegible) + barras de nivel (overall) +
rpm, con Iniciar/Detener/Grabar. Próximo: órbita, Bode/cascada, alarmas.
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

_NAVY = "#0F1E3D"
_BLUE = "#4f8fd0"
_GRID = (210, 218, 230)


def build_agent(args) -> AcqAgent:
    idxs = [int(x) for x in args.chans.split(",") if x.strip() != ""]
    names = [s.strip() for s in args.names.split(",")]
    coup = "AC" if args.prox else "IEPE"
    units = "mil pp" if args.prox else "g rms"
    chans = []
    for k, i in enumerate(idxs):
        chans.append(ChannelConfig(
            name=names[k] if k < len(names) else f"CH{i}", coupling=coup,
            sensitivity_mv_per_eu=args.sens, bnc_port=(i + 1), units=units))
    if args.kph_bnc:
        chans.append(ChannelConfig(name="KPH", coupling="DC", sensitivity_mv_per_eu=1.0,
                                   bnc_port=args.kph_bnc, units="pulses/rev"))
    cfg = StreamConfig(sample_rate_hz=args.fs, channels=chans, block_seconds=0.1,
                       buffer_seconds=4.0, rpm=args.rpm, chassis_name=args.chassis)
    if args.sim:
        cfg.speed_profile = "constant"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="Fuente simulada (sin hardware)")
    ap.add_argument("--machine", default="Rotor_Kit_Field")
    ap.add_argument("--chassis", default="cDAQ1")
    ap.add_argument("--mod", default="cDAQ1Mod1")   # informativo (bnc_port hace el mapeo)
    ap.add_argument("--chans", default="0,1")
    ap.add_argument("--names", default="1YA,1XA")
    ap.add_argument("--sens", type=float, default=100.0)
    ap.add_argument("--fs", type=float, default=5120.0)
    ap.add_argument("--rpm", type=float, default=1475.0)   # solo para el simulador
    ap.add_argument("--prox", action="store_true")
    ap.add_argument("--kph-bnc", type=int, default=0)
    args = ap.parse_args()

    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        import pyqtgraph as pg
    except ImportError:
        print("Faltan dependencias del módulo nativo:\n    pip install -r native/requirements.txt")
        return 2

    agent = build_agent(args)
    from core.remote_monitoring.recorder import TransientRecorder, upload_recording

    pg.setConfigOptions(antialias=True, background="w", foreground=_NAVY)
    app = QtWidgets.QApplication(sys.argv)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Field · {args.machine}")
    win.resize(1280, 800)
    central = QtWidgets.QWidget(); win.setCentralWidget(central)
    root = QtWidgets.QVBoxLayout(central)

    # --- Barra de control ---
    bar = QtWidgets.QHBoxLayout()
    btn_start = QtWidgets.QPushButton("▶ Iniciar")
    btn_stop = QtWidgets.QPushButton("⏸ Detener"); btn_stop.setEnabled(False)
    btn_rec = QtWidgets.QPushButton("⏺ Grabar"); btn_rec.setCheckable(True)
    lbl_rpm = QtWidgets.QLabel("RPM: —"); lbl_rpm.setStyleSheet(f"font-weight:700;color:{_NAVY}")
    lbl_status = QtWidgets.QLabel("detenido")
    ch_names = [c.name for c in agent.channels if not is_keyphasor_channel(c)]
    combo = QtWidgets.QComboBox(); combo.addItems(ch_names)
    for w in (btn_start, btn_stop, btn_rec):
        w.setMinimumHeight(34)
    bar.addWidget(btn_start); bar.addWidget(btn_stop); bar.addWidget(btn_rec)
    bar.addSpacing(16); bar.addWidget(QtWidgets.QLabel("Espectro de:")); bar.addWidget(combo)
    bar.addStretch(1); bar.addWidget(lbl_rpm); bar.addSpacing(16); bar.addWidget(lbl_status)
    root.addLayout(bar)

    # --- Gráficos (pyqtgraph, tiempo real) ---
    gl = pg.GraphicsLayoutWidget(); root.addWidget(gl, 1)
    vib = [(i, c) for i, c in enumerate(agent.channels) if not is_keyphasor_channel(c)]
    kph_idx = agent.source.config.keyphasor_index()

    # Fila 1: waveforms apilados (una curva por canal)
    wave_curves = []
    for r, (i, c) in enumerate(vib):
        p = gl.addPlot(row=r, col=0)
        p.showGrid(x=True, y=True, alpha=0.25)
        p.setLabel("left", c.name, units=c.units.split()[0] if c.units else "")
        if r < len(vib) - 1:
            p.getAxis("bottom").setStyle(showValues=False)
        else:
            p.setLabel("bottom", "ms")
        wave_curves.append(p.plot(pen=pg.mkPen(_BLUE, width=1.3)))
    # Fila siguiente: espectro
    p_spec = gl.addPlot(row=len(vib), col=0)
    p_spec.showGrid(x=True, y=True, alpha=0.25)
    p_spec.setLabel("left", "amplitud")
    p_spec.setLabel("bottom", "Frecuencia (Hz)")
    spec_curve = p_spec.plot(pen=pg.mkPen(_BLUE, width=1.3))
    v1x = pg.InfiniteLine(angle=90, pen=pg.mkPen("#e26d6d", width=1, style=QtCore.Qt.DashLine))
    p_spec.addItem(v1x)

    # Barra de nivel (overall) por canal, abajo
    bars = QtWidgets.QHBoxLayout()
    level_lbls = []
    for _, c in vib:
        box = QtWidgets.QVBoxLayout()
        pb = QtWidgets.QProgressBar(); pb.setRange(0, 100); pb.setTextVisible(True)
        pb.setFormat(f"{c.name}: %v%")
        lab = QtWidgets.QLabel(f"{c.name}  —"); lab.setStyleSheet(f"color:{_NAVY}")
        box.addWidget(lab); box.addWidget(pb)
        bars.addLayout(box); level_lbls.append((pb, lab, c))
    root.addLayout(bars)

    # --- Estado de grabación ---
    state = {"rec": None}

    def do_start():
        try:
            agent.start()
            btn_start.setEnabled(False); btn_stop.setEnabled(True)
            lbl_status.setText("adquiriendo (hilo de fondo)")
            timer.start(60)     # ~16 FPS
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(win, "Error", f"No se pudo iniciar: {e}")

    def do_stop():
        timer.stop(); agent.stop()
        btn_start.setEnabled(True); btn_stop.setEnabled(False)
        lbl_status.setText("detenido")
        if btn_rec.isChecked():
            btn_rec.setChecked(False); do_rec(False)

    def do_rec(checked):
        if checked:
            ch_meta = [{"name": c.name, "units": c.units, "coupling": c.coupling,
                        "bnc_port": c.bnc_port, "sensitivity_mv_per_eu": float(c.sensitivity_mv_per_eu or 0)}
                       for c in agent.channels]
            rec = TransientRecorder(agent.instance_id, agent.sample_rate_hz, ch_meta, machine=args.machine)
            agent.on_block = rec.append
            state["rec"] = rec
            btn_rec.setText("⏹ Detener grabación")
        else:
            rec = state.get("rec")
            agent.on_block = None
            if rec:
                rec.stop()
                up = upload_recording(rec.dir)
                msg = ("subida a la nube" if up.get("ok") else "local (pendiente)")
                QtWidgets.QMessageBox.information(
                    win, "Grabación", f"{rec.rec_id} · {rec.status.duration_s:.0f}s · "
                    f"{rec.status.size_mb:.1f} MB · {msg}")
            state["rec"] = None
            btn_rec.setText("⏺ Grabar")

    def update():
        snap = agent.snapshot(2.0)
        if snap.shape[1] < 16:
            return
        fs = agent.sample_rate_hz
        rpm = agent.estimate_rpm(snap)
        lbl_rpm.setText(f"RPM: {rpm:.0f}" if rpm else "RPM: —")
        f1 = (rpm / 60.0) if rpm else None
        # waveforms (últimos ~0.3 s)
        nshow = min(snap.shape[1], int(0.3 * fs))
        tms = np.arange(nshow) / fs * 1000.0
        for (i, c), curve in zip(vib, wave_curves):
            eu = snap[i, -nshow:] * 1000.0 / (c.sensitivity_mv_per_eu or 1.0)
            curve.setData(tms, eu - eu.mean())
        # nivel overall por canal (barra 0-100 relativo a full-scale simple)
        for pb, lab, c in level_lbls:
            idx = next(ii for ii, cc in vib if cc.name == c.name)
            eu = snap[idx] * 1000.0 / (c.sensitivity_mv_per_eu or 1.0)
            ov = float(np.sqrt(np.mean((eu - eu.mean()) ** 2)))
            lab.setText(f"{c.name}  {ov:.3g} {c.units}")
            pb.setValue(int(min(100, ov / 0.5 * 100)))   # escala provisional
        # espectro del canal elegido
        name = combo.currentText()
        sel = next((i for i, c in vib if c.name == name), vib[0][0])
        sens = next((c.sensitivity_mv_per_eu for i, c in vib if i == sel), 100.0)
        eu = snap[sel] * 1000.0 / (sens or 1.0)
        fr, mag = _spectrum(eu, fs)
        keep = fr <= min(fr[-1], 2000.0)
        spec_curve.setData(fr[keep], mag[keep])
        if f1:
            v1x.setPos(f1); v1x.show()
        else:
            v1x.hide()

    timer = QtCore.QTimer(); timer.timeout.connect(update)
    btn_start.clicked.connect(do_start)
    btn_stop.clicked.connect(do_stop)
    btn_rec.toggled.connect(do_rec)

    win.show()
    ret = app.exec()
    try:
        agent.stop()
    except Exception:  # noqa: BLE001
        pass
    return ret


if __name__ == "__main__":
    sys.exit(main())
