"""
native/watermelon_efficiency.py — Watermelon Efficiency (app de campo)
======================================================================

App de campo para EFICIENCIA de máquinas rotatorias (motores, ventiladores,
bombas, compresores, turbinas, …). REUSA el motor PROBADO `core/efficiency/engine`
— NO reimplementa fórmulas. Sigue el molde Modal/Torsional/Balanceo
[[module-creation-playbook]].

Mide:
  · Potencia MECÁNICA  P_mec = T·ω/1000   (par del TorqueTrak 10K + rpm keyphasor)
  · Potencia ELÉCTRICA P_elec = √3·V·I·cosφ/1000  (analizador de red / manual)
  · η_motor = P_mec/P_elec · η_operativa = P_mec/P_diseño (semáforo por norma)
  · Eficiencia por TIPO (bomba hidráulica / ventilador aire / compresor isentrópico
    / turbina hidráulica).

Fuentes de datos:
  · Manual — el cliente da par/rpm + V/I/cosφ de otros equipos.
  · Simulado — caso de ejemplo para practicar.
  · NI (TorqueTrak 10K → ±10 V AI0 + keyphasor AI1) — par+rpm en vivo; lo eléctrico
    SIEMPRE es manual (analizador de red aparte).

Normas: IEC 60034-2 · ISO 5801 · ISO 9906 · ASME PTC 10 · IEC 60041 · ISO 20816.
Corre con licencia (modelo paquete). WM_LICENSING=0 salta el gate (dev).
"""
from __future__ import annotations

import argparse
import sys
import traceback

from PySide6 import QtCore, QtGui, QtWidgets

from core.efficiency.engine import (
    MACHINE_TYPES, EfficiencyInputs, compute, diagnose_operational,
    mechanical_power_kw, electrical_power_kw, DEMO_B120,
)
from core.torsional.ni_source import (
    KeyphasorSensor, NITorsionalConfig, NITorsionalSource, rpm_from_keyphasor,
)

__version__ = "0.1.3"

# Marca
NAVY = "#0f2a4a"; ACC = "#1AAEE5"; GREEN = "#16a34a"; AMBER = "#f59e0b"; RED = "#dc2626"

# --- idioma (bilingüe, como los otros módulos) ---
_LANG = "en"


def _load_lang() -> str:
    try:
        v = QtCore.QSettings("WatermelonSystem", "Efficiency").value("lang", "en")
        return "es" if str(v).lower().startswith("es") else "en"
    except Exception:  # noqa: BLE001
        return "en"


def _save_lang(v: str) -> None:
    try:
        QtCore.QSettings("WatermelonSystem", "Efficiency").setValue("lang", "es" if v == "es" else "en")
    except Exception:  # noqa: BLE001
        pass


def T(en: str, es: str) -> str:
    return es if _LANG == "es" else en


# Tipos que necesitan parámetros de proceso (caudal/altura/presión).
_PROC_FIELDS = {
    "pump":       ("flow_head",),      # Q [m³/s] + H [m]
    "hydro":      ("flow_head",),      # Q + H
    "fan":        ("flow_dp",),        # Q + Δp [Pa]
    "compressor": ("comp",),           # m, Cp, T1, π, k
}


def _stylesheet() -> str:
    return f"""
    QWidget {{ font-size: 13px; color: #0f172a; }}
    QMainWindow, QWidget#page {{ background: #eef2f7; }}
    QGroupBox {{ background: white; border: 1px solid #dbe4f0; border-radius: 12px;
        margin-top: 12px; padding: 12px; font-weight: 700; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {NAVY}; }}
    QPushButton {{ background: {NAVY}; color: white; border: none; border-radius: 9px;
        padding: 8px 14px; font-weight: 700; }}
    QPushButton:hover {{ background: #12325a; }}
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{ background: white; border: 1px solid #cfdbec;
        border-radius: 8px; padding: 6px 8px; }}
    QTabBar::tab {{ padding: 8px 14px; }}
    """


def _dsb(minv, maxv, val, dec=2, suffix=""):
    w = QtWidgets.QDoubleSpinBox(); w.setRange(minv, maxv); w.setDecimals(dec); w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


def build_app(simulated: bool = True):
    global _LANG
    _LANG = _load_lang()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.C))   # punto decimal SIEMPRE
    app.setStyleSheet(_stylesheet())
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Efficiency v{__version__}")
    win.resize(1180, 820)

    st = {"result": None, "setup_fn": None,
          "acq": {"mode": "sim", "kph": KeyphasorSensor.phototach_reflective(),
                  "kph_device": "cDAQ1Mod1", "torque_device": "cDAQ1Mod1",
                  "tq_fullscale": 1000.0, "fs": 2560.0},
          # Simulador = caso real Paz del Río · Ventilador B120 (fuente única
          # DEMO_B120 en core.efficiency.engine; campo y web muestran lo MISMO).
          "sim": {"type": DEMO_B120["machine_type"], "torque": DEMO_B120["torque_nm"],
                  "rpm": DEMO_B120["rpm"], "v": DEMO_B120["voltage_v"], "i": DEMO_B120["current_a"],
                  "pf": DEMO_B120["power_factor"], "design": DEMO_B120["design_power_kw"],
                  "flow": DEMO_B120["flow_m3s"], "dp": DEMO_B120["dp_pa"]}}

    # ---- Toolbar: marca + versión + idioma ----
    tb = win.addToolBar("main"); tb.setMovable(False)
    tb.setStyleSheet(f"QToolBar {{ background: {NAVY}; padding: 7px 14px; }}")
    brand = QtWidgets.QLabel(
        "<span style='color:#fff;font-weight:800;letter-spacing:2.5px;font-size:16px;'>WATERMELON</span>"
        "<span style='color:#1AAEE5;font-weight:800;letter-spacing:2.5px;font-size:16px;'>&nbsp;EFFICIENCY</span>")
    brand.setTextFormat(QtCore.Qt.RichText); tb.addWidget(brand)
    _ver = QtWidgets.QLabel(f"v{__version__}")
    _ver.setStyleSheet("color:#cbd5e1; background:#1e3a5f; border-radius:9px; padding:2px 10px; margin-left:12px;")
    tb.addWidget(_ver)
    _spc = QtWidgets.QWidget(); _spc.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    tb.addWidget(_spc)

    def _switch_lang(new):
        if new == _LANG:
            return
        _save_lang(new)
        QtWidgets.QMessageBox.information(win, "Watermelon Efficiency",
            T("Language set. Restart to apply.", "Idioma cambiado. Reinicia para aplicar."))
        try:
            QtCore.QProcess.startDetached(QtWidgets.QApplication.applicationFilePath(), sys.argv[1:])
        except Exception:  # noqa: BLE001
            pass
        app.quit()

    for _code in ("EN", "ES"):
        _b = QtWidgets.QToolButton(); _b.setText(_code); _b.setCheckable(True)
        _b.setChecked(_code.lower() == _LANG)
        _b.setStyleSheet("QToolButton{background:transparent;color:#8ea0bd;border:none;padding:3px 10px;font-weight:800;}"
                         "QToolButton:checked{background:#1AAEE5;color:#08243a;border-radius:6px;}")
        _b.clicked.connect(lambda _=0, c=_code.lower(): _switch_lang(c))
        tb.addWidget(_b)

    tabs = QtWidgets.QTabWidget(); win.setCentralWidget(tabs)

    def _scroll(inner):
        sa = QtWidgets.QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(QtWidgets.QFrame.NoFrame)
        sa.setWidget(inner); return sa

    # =================================================================
    # TAB 0 — Setup
    # =================================================================
    pg_set = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(pg_set)
    sl.addWidget(QtWidgets.QLabel(T("Machine identification — used in the report and the saved/uploaded run.",
                                    "Identificación de la máquina — se usa en el reporte y la corrida guardada/subida.")))
    gb = QtWidgets.QGroupBox(T("Machine / asset", "Máquina / activo")); f = QtWidgets.QFormLayout(gb)
    ed_machine = QtWidgets.QLineEdit(); ed_tag = QtWidgets.QLineEdit(); ed_client = QtWidgets.QLineEdit()
    ed_loc = QtWidgets.QLineEdit(); ed_operator = QtWidgets.QLineEdit(); ed_approved = QtWidgets.QLineEdit()
    cb_type = QtWidgets.QComboBox()
    for code, en, es, norm in MACHINE_TYPES:
        cb_type.addItem(f"{T(en, es)}  ·  {norm}", code)
    sb_design = _dsb(0, 1e6, 2140.0, 1, " kW")          # potencia de diseño/placa
    sb_phases = QtWidgets.QComboBox(); sb_phases.addItems(["3", "1"])
    for lbl, w in [(T("Machine", "Máquina"), ed_machine), (T("Tag", "Tag"), ed_tag),
                   (T("Client", "Cliente"), ed_client), (T("Location", "Ubicación"), ed_loc),
                   (T("Machine type", "Tipo de máquina"), cb_type),
                   (T("Design / nameplate power", "Potencia de diseño/placa"), sb_design),
                   (T("Electrical phases", "Fases eléctricas"), sb_phases),
                   (T("Operator", "Operador"), ed_operator), (T("Approved by", "Aprobado por"), ed_approved)]:
        f.addRow(lbl, w)
    sl.addWidget(gb)
    _type_hint = QtWidgets.QLabel(T(
        "ℹ Choose what DRIVES the load (fan / pump / compressor / turbine), NOT the motor. "
        "The driven type computes the motor efficiency AND the process (aerodynamic/hydraulic) "
        "efficiency; picking 'motor' only gives motor + operational. E.g. Paz del Río B120 → Fan.",
        "ℹ Elige lo que MUEVE la carga (ventilador / bomba / compresor / turbina), NO el motor. "
        "El tipo accionado calcula la eficiencia del motor Y la de proceso (aerodinámica/hidráulica); "
        "si eliges 'motor' solo da motor + operativa. Ej.: Ventilador B120 de Paz del Río → Ventilador."))
    _type_hint.setWordWrap(True)
    _type_hint.setStyleSheet(f"color:{NAVY};background:#eef6fd;border:1px solid #cfe3f5;"
                             "border-radius:8px;padding:8px 10px;font-size:12px;")
    sl.addWidget(_type_hint)
    btn_saveset = QtWidgets.QPushButton(T("💾 Save setup", "💾 Guardar setup"))
    btn_saveset.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    _setmsg = QtWidgets.QLabel(""); _setmsg.setStyleSheet("color:#16a34a;font-weight:700;")
    _sr = QtWidgets.QHBoxLayout(); _sr.addWidget(btn_saveset); _sr.addWidget(_setmsg); _sr.addStretch(1)
    sl.addLayout(_sr); sl.addStretch(1)
    tabs.addTab(_scroll(pg_set), "Setup")

    _SET = QtCore.QSettings("WatermelonSystem", "EfficiencySetup")

    def _mtype() -> str:
        return cb_type.currentData() or "motor"

    def _setup_dict():
        return {"machine": ed_machine.text(), "tag": ed_tag.text(), "client": ed_client.text(),
                "location": ed_loc.text(), "operator": ed_operator.text(), "approved_by": ed_approved.text(),
                "machine_type": _mtype(), "design_power_kw": sb_design.value(),
                "phases": int(sb_phases.currentText())}
    st["setup_fn"] = _setup_dict

    def _save_setup():
        for k, v in _setup_dict().items():
            _SET.setValue(k, v)
        _setmsg.setText(T("✅ Setup saved.", "✅ Setup guardado."))

    def _load_setup():
        if _SET.value("machine") is None:
            return
        ed_machine.setText(str(_SET.value("machine", "") or "")); ed_tag.setText(str(_SET.value("tag", "") or ""))
        ed_client.setText(str(_SET.value("client", "") or "")); ed_loc.setText(str(_SET.value("location", "") or ""))
        ed_operator.setText(str(_SET.value("operator", "") or "")); ed_approved.setText(str(_SET.value("approved_by", "") or ""))
        try: sb_design.setValue(float(_SET.value("design_power_kw", 2140.0) or 2140.0))
        except Exception: pass  # noqa: BLE001
        _mt = str(_SET.value("machine_type", "motor") or "motor")
        _idx = cb_type.findData(_mt)
        if _idx >= 0:
            cb_type.setCurrentIndex(_idx)
        sb_phases.setCurrentText(str(_SET.value("phases", 3) or 3))
    btn_saveset.clicked.connect(_save_setup)

    # =================================================================
    # TAB 1 — Configuration (adquisición + parámetros de proceso)
    # =================================================================
    pg_cfg = QtWidgets.QWidget(); cl = QtWidgets.QVBoxLayout(pg_cfg)
    cl.addWidget(QtWidgets.QLabel(T(
        "Data source for torque + rpm. Manual = type them (from other instruments). "
        "Simulated = practice. NI = TorqueTrak 10K (±10 V → AI0) + keyphasor (AI1). "
        "Electrical (V·I·cosφ) is ALWAYS typed from the power analyzer.",
        "Fuente del par + rpm. Manual = escribirlos (de otros equipos). Simulado = practicar. "
        "NI = TorqueTrak 10K (±10 V → AI0) + keyphasor (AI1). Lo eléctrico (V·I·cosφ) SIEMPRE "
        "se escribe del analizador de red.")))
    gb_acq = QtWidgets.QGroupBox(T("Acquisition", "Adquisición")); fa = QtWidgets.QFormLayout(gb_acq)
    cb_mode = QtWidgets.QComboBox()
    cb_mode.addItems([T("Manual (typed)", "Manual (escrito)"), T("Simulated", "Simulado"),
                      "NI (TorqueTrak 10K + keyphasor)"])
    cb_mode.setCurrentIndex(1)
    cb_kph = QtWidgets.QComboBox()
    cb_kph.addItems([T("Photo-tach (reflective tape)", "Foto-tacómetro (cinta reflectiva)"),
                     "Bently 3300 XL 8mm + Proximitor"])
    sb_ppr = QtWidgets.QSpinBox(); sb_ppr.setRange(1, 60); sb_ppr.setValue(1)
    ed_dev = QtWidgets.QLineEdit("cDAQ1Mod1")
    sb_tqfs = _dsb(1, 1e7, 1000.0, 1, " N·m")       # par a fondo de escala (±10 V del RX10K)
    fa.addRow(T("Mode", "Modo"), cb_mode)
    fa.addRow(T("Keyphasor sensor (AI1)", "Sensor keyphasor (AI1)"), cb_kph)
    fa.addRow(T("Pulses per rev", "Pulsos por vuelta"), sb_ppr)
    fa.addRow(T("NI device (AI0 torque, AI1 kph)", "Device NI (AI0 par, AI1 kph)"), ed_dev)
    fa.addRow(T("Torque at full scale (±10 V)", "Par a fondo de escala (±10 V)"), sb_tqfs)
    cl.addWidget(gb_acq)

    # --- Parámetros de proceso (según tipo de máquina) ---
    gb_proc = QtWidgets.QGroupBox(T("Process parameters (by machine type)", "Parámetros de proceso (por tipo)"))
    fp = QtWidgets.QFormLayout(gb_proc)
    sb_flow = _dsb(0, 1e6, 0.0, 4, " m³/s")
    sb_head = _dsb(0, 1e5, 0.0, 2, " m")
    sb_dp = _dsb(0, 1e8, 0.0, 1, " Pa")
    sb_rho = _dsb(0.01, 20000, 1000.0, 3, " kg/m³")
    sb_mdot = _dsb(0, 1e5, 0.0, 3, " kg/s")
    sb_cp = _dsb(0.1, 20, 1.005, 4, " kJ/kg·K")
    sb_tin = _dsb(1, 2000, 293.15, 2, " K")
    sb_pr = _dsb(1, 100, 2.0, 3)
    sb_k = _dsb(1, 2, 1.4, 3)
    _rows_flowhead = [(T("Flow Q", "Caudal Q"), sb_flow), (T("Head H", "Altura H"), sb_head),
                      (T("Fluid density ρ", "Densidad del fluido ρ"), sb_rho)]
    _rows_flowdp = [(T("Flow Q", "Caudal Q"), sb_flow), (T("Total pressure Δp", "Presión total Δp"), sb_dp)]
    _rows_comp = [(T("Mass flow ṁ", "Flujo másico ṁ"), sb_mdot), (T("Cp", "Cp"), sb_cp),
                  (T("Inlet temp T1", "Temp. de entrada T1"), sb_tin),
                  (T("Pressure ratio π", "Relación de presión π"), sb_pr),
                  (T("k = Cp/Cv", "k = Cp/Cv"), sb_k)]
    for lbl, w in (_rows_flowhead + _rows_flowdp[1:2] + _rows_comp):
        fp.addRow(lbl, w)
    _proc_hint = QtWidgets.QLabel(""); _proc_hint.setWordWrap(True); _proc_hint.setStyleSheet("color:#64748b;font-size:11px;")
    fp.addRow("", _proc_hint)
    cl.addWidget(gb_proc)

    # Mapea cada widget a los grupos de campos que lo usan → mostrar/ocultar por tipo.
    _field_groups = {
        "flow_head": [sb_flow, sb_head, sb_rho],
        "flow_dp": [sb_flow, sb_dp],
        "comp": [sb_mdot, sb_cp, sb_tin, sb_pr, sb_k],
    }

    def _proc_visible():
        needed = _PROC_FIELDS.get(_mtype(), ())
        show = set()
        for g in needed:
            for w in _field_groups[g]:
                show.add(w)
        any_needed = bool(needed)
        for grp in _field_groups.values():
            for w in grp:
                # mostrar/ocultar fila completa (label + widget)
                lab = fp.labelForField(w)
                w.setVisible(w in show)
                if lab is not None:
                    lab.setVisible(w in show)
        gb_proc.setVisible(any_needed)
        _proc_hint.setText(T(
            "Process efficiency needs these. Motor/generic: leave blank (only motor & operational η).",
            "La eficiencia de proceso los necesita. Motor/genérico: déjalos vacíos (solo η motor y operativa)."))

    btn_savecfg = QtWidgets.QPushButton(T("💾 Save configuration", "💾 Guardar configuración"))
    btn_savecfg.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    _cfgmsg = QtWidgets.QLabel(""); _cfgmsg.setStyleSheet("color:#16a34a;font-weight:700;")
    _cr = QtWidgets.QHBoxLayout(); _cr.addWidget(btn_savecfg); _cr.addWidget(_cfgmsg); _cr.addStretch(1)
    cl.addLayout(_cr); cl.addStretch(1)
    tabs.addTab(_scroll(pg_cfg), "Configuration")

    _CFG = QtCore.QSettings("WatermelonSystem", "EfficiencyConfig")

    def _save_config():
        _CFG.setValue("mode", cb_mode.currentIndex()); _CFG.setValue("kph", cb_kph.currentIndex())
        _CFG.setValue("ppr", sb_ppr.value()); _CFG.setValue("dev", ed_dev.text()); _CFG.setValue("tqfs", sb_tqfs.value())
        for key, w in [("flow", sb_flow), ("head", sb_head), ("dp", sb_dp), ("rho", sb_rho),
                       ("mdot", sb_mdot), ("cp", sb_cp), ("tin", sb_tin), ("pr", sb_pr), ("k", sb_k)]:
            _CFG.setValue(key, w.value())
        _cfgmsg.setText(T("✅ Configuration saved.", "✅ Configuración guardada."))

    def _load_config():
        if _CFG.value("mode") is None:
            return
        try:
            cb_mode.setCurrentIndex(int(_CFG.value("mode", 1)))
            cb_kph.setCurrentIndex(int(_CFG.value("kph", 0))); sb_ppr.setValue(int(_CFG.value("ppr", 1)))
            ed_dev.setText(str(_CFG.value("dev", "cDAQ1Mod1") or "cDAQ1Mod1"))
            sb_tqfs.setValue(float(_CFG.value("tqfs", 1000.0) or 1000.0))
            for key, w in [("flow", sb_flow), ("head", sb_head), ("dp", sb_dp), ("rho", sb_rho),
                           ("mdot", sb_mdot), ("cp", sb_cp), ("tin", sb_tin), ("pr", sb_pr), ("k", sb_k)]:
                w.setValue(float(_CFG.value(key, w.value()) or w.value()))
        except Exception:  # noqa: BLE001
            pass
    btn_savecfg.clicked.connect(_save_config)

    def _kph_sensor():
        idx = cb_kph.currentIndex(); ppr = sb_ppr.value()
        s = (KeyphasorSensor.bently_3300xl_8mm(keyways=ppr) if idx == 1
             else KeyphasorSensor.phototach_reflective(strips=ppr))
        st["acq"]["kph"] = s
        return s

    def _on_mode(_=0):
        i = cb_mode.currentIndex()
        st["acq"]["mode"] = ["manual", "sim", "ni"][i]
        st["acq"]["torque_device"] = st["acq"]["kph_device"] = ed_dev.text() or "cDAQ1Mod1"
        st["acq"]["tq_fullscale"] = sb_tqfs.value()
    cb_kph.currentIndexChanged.connect(lambda _=0: _kph_sensor())
    sb_ppr.valueChanged.connect(lambda _=0: _kph_sensor())
    cb_mode.currentIndexChanged.connect(_on_mode)
    ed_dev.editingFinished.connect(_on_mode); sb_tqfs.valueChanged.connect(lambda _=0: _on_mode())
    cb_type.currentIndexChanged.connect(lambda _=0: _proc_visible())
    _kph_sensor(); _on_mode()

    # ---------- captura NI del par + rpm ----------
    def _ni_capture_torque():
        """Snapshot NI (~1 s): par (N·m) y rpm. None si falla (sin driver/hardware)."""
        acq = st["acq"]; fs = acq["fs"]
        cfg = NITorsionalConfig(device=acq["torque_device"], torque_ai=0, kph_ai=1,
                                sample_rate_hz=fs, block_seconds=0.1, voltage_range=10.0,
                                keyphasor=acq["kph"])
        src = NITorsionalSource(cfg)
        try:
            src.start()
        except RuntimeError as exc:
            QtWidgets.QMessageBox.critical(win, "Watermelon Efficiency",
                T(f"NI capture failed:\n{exc}", f"Falló la captura NI:\n{exc}")); return None
        import numpy as np
        try:
            blocks = [src.read_block() for _ in range(10)]   # ~1 s
        finally:
            src.stop()
        data = np.concatenate(blocks, axis=1)
        tq_v = float(np.mean(data[0]))                       # par promedio en Volts
        torque_nm = tq_v * (acq["tq_fullscale"] / 10.0)      # ±10 V → ±fondo de escala
        try:
            rpm_inst, _t = rpm_from_keyphasor(data[1], fs, acq["kph"])
            rpm = float(np.median(rpm_inst)) if len(rpm_inst) else 0.0
        except Exception:  # noqa: BLE001
            rpm = 0.0
        return abs(torque_nm), rpm

    # =================================================================
    # TAB 2 — Measurement
    # =================================================================
    pg_m = QtWidgets.QWidget(); ml = QtWidgets.QVBoxLayout(pg_m)
    ml.addWidget(QtWidgets.QLabel(T("Enter / capture the operating point. Mechanical: torque + rpm. "
                                    "Electrical: line voltage, current and power factor (from the analyzer).",
                                    "Ingresa / captura el punto de operación. Mecánico: par + rpm. "
                                    "Eléctrico: tensión de línea, corriente y factor de potencia (del analizador).")))
    gb_mech = QtWidgets.QGroupBox(T("Mechanical (shaft)", "Mecánico (eje)")); fm = QtWidgets.QFormLayout(gb_mech)
    sb_torque = _dsb(0, 1e7, 0.0, 2, " N·m")
    sb_rpm = _dsb(0, 60000, 0.0, 1, " rpm")
    fm.addRow(T("Torque", "Par"), sb_torque)
    fm.addRow("RPM", sb_rpm)
    btn_capt = QtWidgets.QPushButton(T("📷 Capture torque + rpm (NI)", "📷 Capturar par + rpm (NI)"))
    fm.addRow("", btn_capt)
    ml.addWidget(gb_mech)

    gb_el = QtWidgets.QGroupBox(T("Electrical (power analyzer)", "Eléctrico (analizador de red)")); fe = QtWidgets.QFormLayout(gb_el)
    sb_v = _dsb(0, 1e6, 0.0, 1, " V")
    sb_i = _dsb(0, 1e6, 0.0, 1, " A")
    sb_pf = _dsb(0, 1, 0.92, 3)
    fe.addRow(T("Line voltage (V_LL)", "Tensión de línea (V_LL)"), sb_v)
    fe.addRow(T("Line current", "Corriente de línea"), sb_i)
    fe.addRow(T("Power factor cosφ", "Factor de potencia cosφ"), sb_pf)
    ml.addWidget(gb_el)

    _mrow = QtWidgets.QHBoxLayout()
    btn_demo = QtWidgets.QPushButton(T("Simulated example", "Ejemplo simulado"))
    _mrow.addWidget(btn_demo); _mrow.addStretch(1); ml.addLayout(_mrow)
    ml.addStretch(1)
    tabs.addTab(_scroll(pg_m), T("Measurement", "Medición"))

    def _capture():
        mode = st["acq"]["mode"]
        if mode == "manual":
            QtWidgets.QMessageBox.information(win, "Watermelon Efficiency",
                T("Manual mode — type torque and rpm, or switch mode in Configuration.",
                  "Modo manual — escribe par y rpm, o cambia el modo en Configuration.")); return
        if mode == "ni":
            r = _ni_capture_torque()
            if not r:
                return
            tq, rpm = r
            sb_torque.setValue(tq); sb_rpm.setValue(rpm)
        else:   # sim
            s = st["sim"]; sb_torque.setValue(s["torque"]); sb_rpm.setValue(s["rpm"])
    btn_capt.clicked.connect(_capture)

    def _demo():
        s = st["sim"]
        # tipo + diseño + proceso (caso Paz del Río B120 = ventilador)
        _ti = cb_type.findData(s.get("type", "fan"))
        if _ti >= 0:
            cb_type.setCurrentIndex(_ti)
        sb_design.setValue(s.get("design", 2140.0))
        sb_flow.setValue(s.get("flow", 0.0)); sb_dp.setValue(s.get("dp", 0.0))
        sb_torque.setValue(s["torque"]); sb_rpm.setValue(s["rpm"])
        sb_v.setValue(s["v"]); sb_i.setValue(s["i"]); sb_pf.setValue(s["pf"])
        _proc_visible()
        tabs.setCurrentIndex(IDX_EFF); _compute()
    btn_demo.clicked.connect(_demo)

    # =================================================================
    # TAB 3 — Efficiency
    # =================================================================
    pg_eff = QtWidgets.QWidget(); el = QtWidgets.QVBoxLayout(pg_eff)
    el.addWidget(QtWidgets.QLabel(T("Computed efficiency with semaphore (operational band). "
                                    "Uses the Measurement point + machine type and process parameters.",
                                    "Eficiencia calculada con semáforo (banda operativa). "
                                    "Usa el punto de Medición + tipo de máquina y parámetros de proceso.")))
    btn_eff = QtWidgets.QPushButton(T("▶ Compute efficiency", "▶ Calcular eficiencia"))
    btn_eff.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    el.addWidget(btn_eff)
    out_eff = QtWidgets.QLabel("—"); out_eff.setWordWrap(True); out_eff.setTextFormat(QtCore.Qt.RichText)
    out_eff.setStyleSheet("background:white;border:1px solid #dbe4f0;border-radius:10px;padding:14px;font-size:14px;")
    el.addWidget(out_eff); el.addStretch(1)
    tabs.addTab(_scroll(pg_eff), T("Efficiency", "Eficiencia"))

    def _inputs() -> EfficiencyInputs:
        return EfficiencyInputs(
            machine_type=_mtype(), torque_nm=sb_torque.value(), rpm=sb_rpm.value(),
            voltage_v=sb_v.value(), current_a=sb_i.value(), power_factor=sb_pf.value(),
            phases=int(sb_phases.currentText()), design_power_kw=sb_design.value(),
            flow_m3s=sb_flow.value(), head_m=sb_head.value(), dp_pa=sb_dp.value(), rho=sb_rho.value(),
            mdot_kg_s=sb_mdot.value(), cp_kj_kgk=sb_cp.value(), t_in_k=sb_tin.value(),
            pressure_ratio=sb_pr.value(), k_ratio=sb_k.value())

    _SEMA = {"green": GREEN, "amber": AMBER, "red": RED}

    def _compute():
        if sb_torque.value() <= 0 or sb_rpm.value() <= 0:
            out_eff.setText(T("Enter torque and rpm first (Measurement).",
                              "Ingresa par y rpm primero (Medición).")); return
        res = compute(_inputs())
        st["result"] = res
        d = res.diagnosis; col = _SEMA.get(d.color, "#64748b")
        _lbl = d.label_es if _LANG == "es" else d.label_en
        _rows = [
            (T("Mechanical power P_mec", "Potencia mecánica P_mec"), f"{res.p_mec_kw:,.2f} kW"),
            (T("Electrical power P_elec", "Potencia eléctrica P_elec"),
             f"{res.p_elec_kw:,.2f} kW" if res.p_elec_kw > 0 else "—"),
            (T("Motor efficiency η_motor", "Eficiencia del motor η_motor"),
             f"{res.eta_motor_pct:,.1f} %" if res.eta_motor_pct > 0 else "—"),
            (T("Operational efficiency η_op", "Eficiencia operativa η_op"),
             f"{res.eta_operational_pct:,.1f} %" if res.eta_operational_pct > 0 else "—"),
        ]
        if res.process_label:
            _plbl = {"pump": T("Pump η (hydraulic)", "η bomba (hidráulica)"),
                     "fan": T("Fan η (air)", "η ventilador (aire)"),
                     "hydro": T("Turbine η", "η turbina"),
                     "compressor": T("Compressor η (isentropic)", "η compresor (isentrópica)")}.get(res.process_label, "η")
            _rows.append((_plbl, f"{res.eta_process_pct:,.1f} %"))
            if res.process_power_kw > 0:
                _rows.append((T("Process power", "Potencia de proceso"), f"{res.process_power_kw:,.2f} kW"))
        _tbl = "".join(f"<tr><td style='padding:3px 14px 3px 0;color:#475569'>{k}</td>"
                       f"<td style='padding:3px 0;font-weight:800;color:{NAVY}'>{v}</td></tr>" for k, v in _rows)
        _sem = (f"<div style='display:inline-block;margin-top:10px;padding:8px 16px;border-radius:10px;"
                f"background:{col};color:white;font-weight:800;font-size:15px'>● {_lbl}"
                f" — {res.eta_operational_pct:,.1f}%</div>") if res.eta_operational_pct > 0 else ""
        out_eff.setText(f"<table>{_tbl}</table>{_sem}")
    btn_eff.clicked.connect(_compute)

    # =================================================================
    # TAB 4 — Report + save/cloud
    # =================================================================
    pg_rp = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(pg_rp)
    rl.addWidget(QtWidgets.QLabel(T("Preliminary efficiency report (PDF) + save local / upload to cloud. "
                                    "Uses the last computed result and the Setup data.",
                                    "Reporte preliminar de eficiencia (PDF) + guardar local / subir a la nube. "
                                    "Usa el último resultado calculado y los datos del Setup.")))
    _rlang = QtWidgets.QHBoxLayout(); _rlang.addWidget(QtWidgets.QLabel(T("Report language", "Idioma del reporte")))
    cb_rlang = QtWidgets.QComboBox(); cb_rlang.addItems(["Español", "English"]); cb_rlang.setCurrentIndex(0 if _LANG == "es" else 1)
    _rlang.addWidget(cb_rlang); _rlang.addStretch(1); rl.addLayout(_rlang)
    _rr = QtWidgets.QHBoxLayout()
    btn_pdf = QtWidgets.QPushButton(T("📄 Generate report (PDF)", "📄 Generar reporte (PDF)"))
    btn_pdf.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    btn_savelocal = QtWidgets.QPushButton(T("💾 Save locally", "💾 Guardar local"))
    btn_upload = QtWidgets.QPushButton(T("☁ Upload to cloud", "☁ Subir a la nube"))
    _rr.addWidget(btn_pdf); _rr.addWidget(btn_savelocal); _rr.addWidget(btn_upload); _rr.addStretch(1)
    rl.addLayout(_rr)
    rp_status = QtWidgets.QLabel(""); rp_status.setWordWrap(True); rp_status.setStyleSheet("color:#475569;")
    rl.addWidget(rp_status); rl.addStretch(1)

    def _type_label() -> str:
        code = _mtype()
        for c, en, es, norm in MACHINE_TYPES:
            if c == code:
                return f"{T(en, es)} ({norm})"
        return code

    def _build_pdf():
        res = st.get("result")
        if not res:
            rp_status.setText(T("Compute the efficiency first.", "Calcula la eficiencia primero.")); return None
        _es = (cb_rlang.currentIndex() == 0)
        s = st["setup_fn"]() if st.get("setup_fn") else {}
        from datetime import date as _date
        meta = {"title": T("Preliminary Efficiency Report", "Reporte Preliminar de Eficiencia"),
                "asset": s.get("machine") or s.get("tag") or "—", "client": s.get("client") or "—",
                "machine_type": _type_label(), "location": s.get("location") or "—",
                "test_type": T("Rotating-machine efficiency", "Eficiencia de máquina rotatoria"),
                "rpm": f"{res.p_mec_kw and sb_rpm.value() or 0:,.0f}", "technician": s.get("operator") or "—",
                "reviewer": s.get("approved_by") or "—", "date": _date.today().isoformat(),
                "equipment": "Watermelon Efficiency"}
        d = res.diagnosis; _dlbl = d.label_es if _es else d.label_en
        _go = "GO" if d.code == "normal" else "REVIEW"
        quality = [(T("Operational efficiency", "Eficiencia operativa"), _go,
                    T(f"{res.eta_operational_pct:,.1f}% — {_dlbl}",
                      f"{res.eta_operational_pct:,.1f}% — {_dlbl}") if res.eta_operational_pct > 0
                    else T("No design power set", "Sin potencia de diseño"))]
        if res.eta_motor_pct > 0:
            quality.append((T("Motor efficiency", "Eficiencia del motor"),
                            "GO" if res.eta_motor_pct >= 90 else "REVIEW",
                            f"{res.eta_motor_pct:,.1f}%"))
        if res.process_label:
            quality.append((T("Process efficiency", "Eficiencia de proceso"),
                            "GO" if res.eta_process_pct >= 60 else "REVIEW",
                            f"{res.eta_process_pct:,.1f}%"))
        rows = [[T("Machine type", "Tipo de máquina"), _type_label()],
                [T("Torque", "Par"), f"{sb_torque.value():,.2f} N·m"],
                ["RPM", f"{sb_rpm.value():,.1f}"],
                [T("Mechanical power P_mec", "Potencia mecánica P_mec"), f"{res.p_mec_kw:,.2f} kW"]]
        if res.p_elec_kw > 0:
            rows += [[T("Line V · I · cosφ", "V · I · cosφ"),
                      f"{sb_v.value():,.0f} V · {sb_i.value():,.0f} A · {sb_pf.value():.3f}"],
                     [T("Electrical power P_elec", "Potencia eléctrica P_elec"), f"{res.p_elec_kw:,.2f} kW"],
                     [T("Motor efficiency η_motor", "Eficiencia del motor η_motor"), f"{res.eta_motor_pct:,.1f} %"]]
        rows += [[T("Design power", "Potencia de diseño"), f"{s.get('design_power_kw') or 0:,.1f} kW"],
                 [T("Operational efficiency η_op", "Eficiencia operativa η_op"),
                  f"{res.eta_operational_pct:,.1f} %  ({_dlbl})" if res.eta_operational_pct > 0 else "—"]]
        if res.process_label:
            rows += [[T("Process efficiency", "Eficiencia de proceso"), f"{res.eta_process_pct:,.1f} %"]]
            if res.process_power_kw > 0:
                rows += [[T("Process power", "Potencia de proceso"), f"{res.process_power_kw:,.2f} kW"]]
        sections = [{"title": T("Efficiency result", "Resultado de eficiencia"),
                     "table": {"headers": [T("Item", "Ítem"), T("Value", "Valor")], "rows": rows}}]
        findings = [T(f"Operational efficiency {res.eta_operational_pct:,.1f}% → {_dlbl}.",
                      f"Eficiencia operativa {res.eta_operational_pct:,.1f}% → {_dlbl}.")] if res.eta_operational_pct > 0 else []
        if res.eta_motor_pct > 0:
            findings.append(T(f"Motor efficiency {res.eta_motor_pct:,.1f}% (P_mec {res.p_mec_kw:,.1f} kW / P_elec {res.p_elec_kw:,.1f} kW).",
                              f"Eficiencia del motor {res.eta_motor_pct:,.1f}% (P_mec {res.p_mec_kw:,.1f} kW / P_elec {res.p_elec_kw:,.1f} kW)."))
        if res.process_label:
            findings.append(T(f"{res.process_label} efficiency {res.eta_process_pct:,.1f}%.",
                              f"Eficiencia {res.process_label} {res.eta_process_pct:,.1f}%."))
        analysis = [T("P_mec = T·ω/1000 · P_elec = √3·V·I·cosφ/1000 · η_motor = P_mec/P_elec · η_op = P_mec/P_design.",
                      "P_mec = T·ω/1000 · P_elec = √3·V·I·cosφ/1000 · η_motor = P_mec/P_elec · η_op = P_mec/P_diseño.")]
        recs = [T("Compare against the nameplate efficiency and the operating-point design curve.",
                  "Comparar contra la eficiencia de placa y la curva de diseño del punto de operación."),
                T("If below band, inspect load, alignment, fouling and the operating point.",
                  "Si está bajo la banda, revisar carga, alineación, ensuciamiento y el punto de operación.")]
        try:
            from core.modal.preliminary_report import build_preliminary_pdf
            return build_preliminary_pdf(meta=meta, quality=quality, sections=sections, analysis=analysis,
                                         findings=findings, recommendations=recs,
                                         run_id=f"EFF-{meta['asset']}", lang=("es" if _es else "en"))
        except Exception as exc:  # noqa: BLE001
            rp_status.setText(f"❌ {type(exc).__name__}: {exc}"); return None

    def _gen_pdf():
        pdf = _build_pdf()
        if not pdf:
            return
        s = st["setup_fn"]() if st.get("setup_fn") else {}
        _asset = s.get("machine") or s.get("tag") or "equipo"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(win, T("Save report", "Guardar reporte"),
                                                        f"Eficiencia_{_asset}.pdf", "PDF (*.pdf)")
        if not path:
            return
        with open(path, "wb") as fh:
            fh.write(pdf)
        rp_status.setText(T(f"✅ Saved: {path}", f"✅ Guardado: {path}"))
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def _payload():
        res = st.get("result")
        return {"kind": "efficiency", "setup": st["setup_fn"]() if st.get("setup_fn") else {},
                "inputs": _inputs().__dict__,
                "result": ({"machine_type": res.machine_type, "p_mec_kw": res.p_mec_kw,
                            "p_elec_kw": res.p_elec_kw, "eta_motor_pct": res.eta_motor_pct,
                            "eta_operational_pct": res.eta_operational_pct,
                            "diagnosis": res.diagnosis.code, "process_power_kw": res.process_power_kw,
                            "eta_process_pct": res.eta_process_pct, "process_label": res.process_label}
                           if res else None),
                "app_version": __version__}

    def _save_local():
        if not st.get("result"):
            rp_status.setText(T("Nothing to save yet.", "Nada que guardar aún.")); return
        import os, json
        from datetime import datetime
        d = os.path.join(os.path.expanduser("~"), "WatermelonEfficiency", "runs",
                         datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, "efficiency.json")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump(_payload(), fh, ensure_ascii=False)
        rp_status.setText(T(f"💾 Saved locally: {fp}", f"💾 Guardado local: {fp}"))

    def _upload():
        if not st.get("result"):
            rp_status.setText(T("Nothing to upload yet.", "Nada que subir aún.")); return
        rp_status.setText(T("☁ Uploading…", "☁ Subiendo…")); QtWidgets.QApplication.processEvents()
        try:
            from core.efficiency import cloud
            import socket
            s = st["setup_fn"]() if st.get("setup_fn") else {}
            name = (s.get("machine") or s.get("tag") or "Efficiency") + " · " + T("efficiency", "eficiencia")
            _acc = ""
            try:
                from core.modal import licensing as _lic
                _acc = str((_lic.local_license_status() or {}).get("account") or "")
            except Exception:  # noqa: BLE001
                pass
            r = cloud.save_run(name, _payload(), account=_acc, client=s.get("client", ""),
                               tag=s.get("tag", ""), hostname=socket.gethostname())
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": str(e)}
        rp_status.setText(T(f"☁ Uploaded (id {r.get('id','')}).", f"☁ Subido (id {r.get('id','')}).")
                          if r.get("ok") else
                          T(f"⚠ Upload failed ({r.get('reason','offline')}). Saved-local stays available.",
                            f"⚠ Falló la subida ({r.get('reason','offline')}). El guardado local queda disponible."))
    btn_pdf.clicked.connect(_gen_pdf); btn_savelocal.clicked.connect(_save_local); btn_upload.clicked.connect(_upload)
    tabs.addTab(_scroll(pg_rp), T("Report", "Reporte"))

    # =================================================================
    # TAB 5 — Updates (tarjeta con licencia + updater)
    # =================================================================
    pg_up = QtWidgets.QWidget(); ul = QtWidgets.QVBoxLayout(pg_up); ul.setContentsMargins(28, 24, 28, 24)
    _card = QtWidgets.QFrame()
    _card.setStyleSheet("QFrame{background:white;border:1px solid #e6ecf5;border-radius:16px;}")
    _card.setMinimumWidth(600); _card.setMaximumWidth(780)
    _cl2 = QtWidgets.QVBoxLayout(_card); _cl2.setContentsMargins(34, 30, 34, 34); _cl2.setSpacing(14)

    def _mkfont(pt, bold=False):
        fnt = QtGui.QFont(); fnt.setPointSize(pt); fnt.setBold(bold); return fnt

    _uh = QtWidgets.QLabel("🍉  Watermelon Efficiency"); _uh.setFont(_mkfont(16, True)); _uh.setStyleSheet(f"color:{NAVY};border:none;")
    _cur = QtWidgets.QLabel(T(f"Installed version:  v{__version__}", f"Versión instalada:  v{__version__}"))
    _cur.setFont(_mkfont(11)); _cur.setStyleSheet("color:#475569;border:none;")
    _lic_lbl = QtWidgets.QLabel(""); _lic_lbl.setFont(_mkfont(10)); _lic_lbl.setWordWrap(True)
    _lic_lbl.setTextFormat(QtCore.Qt.RichText); _lic_lbl.setStyleSheet("color:#475569;border:none;")
    _lic_ok = False
    try:
        from core.modal import licensing as _lic
        _ls = _lic.local_license_status()
        import datetime as _dt2
        _expd = _dt2.date.fromtimestamp(float(_ls.get("exp"))).isoformat() if _ls.get("exp") else "—"
        _stt = (f"<span style='color:#10b981'>● {T('Licensed','Con licencia')}</span>" if _ls.get("valid")
                else f"<span style='color:#ef4444'>● {T('Not activated','Sin activar')}</span>")
        _lic_lbl.setText(f"<b>{T('License','Licencia')}:</b> {_stt}<br>"
                         f"{T('Account','Cuenta')}: {_ls.get('account') or '—'} · {T('Expires','Vence')}: {_expd}<br>"
                         f"<b>{T('This computer','Este equipo')}:</b> {_lic.machine_label()}<br>"
                         f"<span style='color:#94a3b8'>Machine ID: {_ls.get('fingerprint','')[:24]}…</span>")
        _lic_ok = bool(_ls.get("valid"))
    except Exception:  # noqa: BLE001
        _lic_lbl.setText("")
    _deact = QtWidgets.QPushButton(T("Deactivate this computer", "Desactivar este equipo")); _deact.setFont(_mkfont(9))
    _deact.setStyleSheet("QPushButton{background:transparent;color:#ef4444;border:1px solid #f2c4c4;border-radius:8px;padding:6px 14px;}"
                         "QPushButton:hover{background:#fef2f2;} QPushButton:disabled{color:#cbd5e1;border-color:#eef2f8;}")
    _deact.setEnabled(_lic_ok)
    _deact_row = QtWidgets.QHBoxLayout(); _deact_row.addWidget(_deact); _deact_row.addStretch(1)

    def _do_deact():
        m = QtWidgets.QMessageBox(_card); m.setIcon(QtWidgets.QMessageBox.Warning)
        m.setWindowTitle(T("Deactivate this computer", "Desactivar este equipo"))
        m.setText(T("Release this license from this computer?", "¿Liberar esta licencia de este equipo?"))
        m.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes)
        m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        if m.exec() != QtWidgets.QMessageBox.Yes:
            return
        try:
            from core.modal import licensing as _lic2
            r = _lic2.deactivate_machine()
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": str(e)}
        if r.get("ok"):
            QtWidgets.QMessageBox.information(_card, "Watermelon Efficiency",
                T("Deactivated. The app will close.", "Desactivado. La app se cerrará.")); QtWidgets.QApplication.quit()
        else:
            QtWidgets.QMessageBox.warning(_card, "Watermelon Efficiency",
                T("Could not deactivate: ", "No se pudo desactivar: ") + str(r.get("reason", "")))
    _deact.clicked.connect(_do_deact)

    _ust = QtWidgets.QLabel(T("Press <b>Check for updates</b> to see if a newer version is available.",
                              "Pulsa <b>Buscar actualizaciones</b> para ver si hay una versión más nueva."))
    _ust.setFont(_mkfont(10)); _ust.setWordWrap(True); _ust.setTextFormat(QtCore.Qt.RichText); _ust.setStyleSheet("color:#64748b;border:none;")
    _unotes = QtWidgets.QTextBrowser(); _unotes.setFont(_mkfont(9)); _unotes.setMaximumHeight(200); _unotes.hide()
    _unotes.setStyleSheet("QTextBrowser{border:1px solid #eef2f8;border-radius:10px;background:#fbfcfe;color:#334155;padding:8px;}")
    _ub = QtWidgets.QPushButton(T("🔍  Check for updates", "🔍  Buscar actualizaciones")); _ub.setFont(_mkfont(11, True)); _ub.setMinimumHeight(42)
    _ub.setStyleSheet(f"QPushButton{{background:{NAVY};color:white;padding:10px 20px;border-radius:9px;}}QPushButton:hover{{background:#12325a;}}")
    _ubgo = QtWidgets.QPushButton(T("⬇  Update now", "⬇  Actualizar ahora")); _ubgo.setFont(_mkfont(11, True)); _ubgo.setMinimumHeight(42); _ubgo.hide()
    _ubgo.setStyleSheet(f"QPushButton{{background:{GREEN};color:white;padding:10px 20px;border-radius:9px;}}QPushButton:hover{{background:#12833a;}}")
    _urow = QtWidgets.QHBoxLayout(); _urow.setSpacing(12); _urow.addWidget(_ub); _urow.addWidget(_ubgo); _urow.addStretch(1)
    _foot = QtWidgets.QLabel(T("Updates download and install automatically over the network; the app restarts when done.",
                              "Las actualizaciones se descargan e instalan automáticamente por red; la app se reinicia al terminar."))
    _foot.setFont(_mkfont(9)); _foot.setWordWrap(True); _foot.setStyleSheet("color:#94a3b8;border:none;")
    _cl2.addWidget(_uh); _cl2.addWidget(_cur); _cl2.addWidget(_lic_lbl); _cl2.addLayout(_deact_row)
    _cl2.addSpacing(4); _cl2.addWidget(_ust); _cl2.addWidget(_unotes); _cl2.addSpacing(6); _cl2.addLayout(_urow)
    _cl2.addSpacing(6); _cl2.addWidget(_foot)
    ul.addWidget(_card, 0, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop); ul.addStretch(1)
    st["_pending"] = None

    def _chk():
        _ub.setEnabled(False); _ub.setText(T("🔍  Checking…", "🔍  Buscando…")); QtWidgets.QApplication.processEvents()
        try:
            from core.efficiency.updater import diagnose
            info, msg = diagnose(__version__)
        except Exception as e:  # noqa: BLE001
            info, msg = None, f"Error: {e}"
        _ub.setEnabled(True); _ub.setText(T("🔍  Check for updates", "🔍  Buscar actualizaciones"))
        st["_pending"] = info
        if info:
            _ust.setText(T(f"✅ <b style='color:{GREEN}'>New version available: v{info['version']}</b>",
                           f"✅ <b style='color:{GREEN}'>Nueva versión disponible: v{info['version']}</b>"))
            _unotes.setPlainText((info.get("notes") or "").strip()); _unotes.show(); _ubgo.show()
        else:
            _ust.setText(str(msg).replace("\n", "<br>")); _unotes.hide(); _ubgo.hide()

    def _go():
        info = st.get("_pending")
        if not info:
            return
        from core.efficiency import updater
        url = info.get("setup_url") or info.get("zip_url")
        if not url:
            if info.get("html_url"):
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(info["html_url"]))
            return
        dlg = QtWidgets.QProgressDialog(T("Downloading update…", "Descargando actualización…"), "Cancel", 0, 100, win)
        dlg.setWindowTitle("Updating"); dlg.setModal(True); dlg.setMinimumDuration(0); dlg.show()
        path = updater.download_file(url, on_progress=lambda fr: (dlg.setValue(int(fr * 100)), QtWidgets.QApplication.processEvents()))
        dlg.close()
        if path and path.lower().endswith("setup.exe"):
            updater.launch_installer(path); QtWidgets.QApplication.quit()
        elif path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
        else:
            QtWidgets.QMessageBox.warning(win, "Update", T("Could not download the update.", "No se pudo descargar la actualización."))
    _ub.clicked.connect(_chk); _ubgo.clicked.connect(_go)
    tabs.addTab(pg_up, T("Updates", "Actualizaciones"))

    # ---------- flujo guiado (guardar para avanzar) ----------
    IDX_CFG, IDX_MEAS, IDX_EFF, IDX_RP = 1, 2, 3, 4

    def _after_setup_saved():
        tabs.setTabEnabled(IDX_CFG, True); _proc_visible(); tabs.setCurrentIndex(IDX_CFG)

    def _after_config_saved():
        for _i in (IDX_MEAS, IDX_EFF, IDX_RP):
            tabs.setTabEnabled(_i, True)
        tabs.setCurrentIndex(IDX_MEAS)
    btn_saveset.clicked.connect(_after_setup_saved)
    btn_savecfg.clicked.connect(_after_config_saved)

    _load_setup(); _load_config(); _on_mode(); _kph_sensor(); _proc_visible()

    # Bloqueo inicial del flujo (se desbloquea al guardar).
    for _i in (IDX_CFG, IDX_MEAS, IDX_EFF, IDX_RP):
        tabs.setTabEnabled(_i, False)
    if _SET.value("machine") not in (None, ""):
        tabs.setTabEnabled(IDX_CFG, True)
    if _CFG.value("mode") is not None:
        for _i in (IDX_MEAS, IDX_EFF, IDX_RP):
            tabs.setTabEnabled(_i, True)

    # Punto decimal SIEMPRE (nunca coma): quita flechas, fija locale C, coma→punto al vuelo.
    _cloc = QtCore.QLocale(QtCore.QLocale.C)
    for _sp in win.findChildren(QtWidgets.QAbstractSpinBox):
        _sp.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        _sp.setLocale(_cloc)
        _le = _sp.lineEdit()
        if _le is not None:
            def _mk(le):
                def _f(txt):
                    if "," in txt:
                        pos = le.cursorPosition()
                        le.setText(txt.replace(",", ".")); le.setCursorPosition(pos)
                return _f
            _le.textEdited.connect(_mk(_le))
    return app, win


def main(argv=None):
    ap = argparse.ArgumentParser(description="Watermelon Efficiency (native)")
    ap.add_argument("--sim", action="store_true", default=True)
    ap.parse_args(argv)
    try:
        import os as _os
        global _LANG
        _here = _os.path.dirname(_os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        _LANG = _load_lang()
        _app0 = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        try:
            from license_gate import run_license_gate
            _brand = ("<span style='color:#fff;font-weight:800;letter-spacing:3px;font-size:22px;'>WATERMELON</span>"
                      "<span style='color:#1AAEE5;font-weight:800;letter-spacing:3px;font-size:22px;'>&nbsp;EFFICIENCY</span>")
            _ok = run_license_gate(_app0, t=T, navy=NAVY, acc=ACC, brand_html=_brand, app_title="Watermelon Efficiency")
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001
            if _os.environ.get("WM_LICENSING", "1") == "0":
                _ok = True
            else:
                QtWidgets.QMessageBox.critical(None, "Watermelon Efficiency",
                    "Licensing error — the app cannot start.\n\n" + traceback.format_exc()[-800:])
                sys.exit(1)
        if not _ok:
            sys.exit(0)
        app, win = build_app(simulated=True); win.showMaximized()
        sys.exit(app.exec())
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        try:
            with open("watermelon_efficiency_error.log", "w", encoding="utf-8") as fh:
                fh.write(err)
            QtWidgets.QMessageBox.critical(None, "Watermelon Efficiency — startup error", err[-1500:])
        except Exception:  # noqa: BLE001
            print(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
