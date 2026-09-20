"""
native/watermelon_balancing.py — Watermelon Balancing (app de campo)
====================================================================

App de campo para BALANCEO in-situ (1 y 2 planos) por coeficientes de
influencia. REUSA el motor PROBADO `core/balance/*` (engine, report, cloud) —
NO reimplementa fórmulas. Sigue el molde de Modal/Torsional [[module-creation-playbook]].

Fuentes de datos:
  · Manual — el cliente da el 1× (amplitud + fase) medido con otro equipo.
  · Simulado — desbalance de ejemplo para practicar.
  · (NI 9229 proximidad µm / NI 9234 acelerómetro→velocidad mm/s — próximo).

Norma de unidad: carcasa/absoluta (acelerómetro) → VELOCIDAD mm/s RMS
(ISO 20816); eje relativo (proximitor) → DESPLAZAMIENTO µm pp (API 670/ISO 7919).

Corre con licencia (modelo paquete). WM_LICENSING=0 salta el gate (dev).
"""
from __future__ import annotations

import argparse
import sys
import traceback

from PySide6 import QtCore, QtGui, QtWidgets

from core.balance.engine import (
    solve_1plane, solve_2plane, recommend_trial_weight_g,
    evaluate_iso_grades, to_complex, to_polar, calc_U_trial, calc_U_res_auto,
)

__version__ = "0.1.0"

# Marca
NAVY = "#0f2a4a"; ACC = "#1AAEE5"; GREEN = "#16a34a"; AMBER = "#f59e0b"; RED = "#dc2626"

# --- idioma (bilingüe, como los otros módulos) ---
_LANG = "es"


def _load_lang() -> str:
    try:
        v = QtCore.QSettings("WatermelonSystem", "Balancing").value("lang", "es")
        return "en" if str(v).lower() == "en" else "es"
    except Exception:  # noqa: BLE001
        return "es"


def _save_lang(v: str) -> None:
    try:
        QtCore.QSettings("WatermelonSystem", "Balancing").setValue("lang", "es" if v == "es" else "en")
    except Exception:  # noqa: BLE001
        pass


def T(en: str, es: str) -> str:
    return es if _LANG == "es" else en


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
    app.setStyleSheet(_stylesheet())
    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"Watermelon Balancing v{__version__}")
    win.resize(1180, 820)

    st = {"unit": "µm pk-pk", "r1p": None, "r2p": None, "iso": None,
          "setup_fn": None}

    # ---- Toolbar: marca + versión + idioma ----
    tb = win.addToolBar("main"); tb.setMovable(False)
    tb.setStyleSheet(f"QToolBar {{ background: {NAVY}; padding: 7px 14px; }}")
    brand = QtWidgets.QLabel(
        "<span style='color:#fff;font-weight:800;letter-spacing:2.5px;font-size:16px;'>WATERMELON</span>"
        "<span style='color:#1AAEE5;font-weight:800;letter-spacing:2.5px;font-size:16px;'>&nbsp;BALANCING</span>")
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
        QtWidgets.QMessageBox.information(win, "Watermelon Balancing",
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

    # =================================================================
    # TAB 0 — Setup
    # =================================================================
    pg_set = QtWidgets.QWidget(); sl = QtWidgets.QVBoxLayout(pg_set)
    sl.addWidget(QtWidgets.QLabel(T("Machine identification — used in the report and the saved/uploaded run.",
                                    "Identificación de la máquina — se usa en el reporte y la corrida guardada/subida.")))
    gb = QtWidgets.QGroupBox(T("Machine / asset", "Máquina / activo")); f = QtWidgets.QFormLayout(gb)
    ed_machine = QtWidgets.QLineEdit(); ed_tag = QtWidgets.QLineEdit(); ed_client = QtWidgets.QLineEdit()
    ed_loc = QtWidgets.QLineEdit(); ed_operator = QtWidgets.QLineEdit(); ed_approved = QtWidgets.QLineEdit()
    sb_rpm = _dsb(0, 30000, 1800, 0, " rpm")
    cb_sensor = QtWidgets.QComboBox()
    cb_sensor.addItems([T("Casing (accelerometer → velocity, mm/s RMS)", "Carcasa (acelerómetro → velocidad, mm/s RMS)"),
                        T("Shaft (proximity, µm pk-pk)", "Eje (proximidad, µm pk-pk)")])
    for lbl, w in [(T("Machine", "Máquina"), ed_machine), (T("Tag", "Tag"), ed_tag),
                   (T("Client", "Cliente"), ed_client), (T("Location", "Ubicación"), ed_loc),
                   (T("Nameplate RPM", "RPM de placa"), sb_rpm),
                   (T("Vibration type", "Tipo de vibración"), cb_sensor),
                   (T("Operator", "Operador"), ed_operator), (T("Approved by", "Aprobado por"), ed_approved)]:
        f.addRow(lbl, w)
    sl.addWidget(gb)
    _lbl_unit = QtWidgets.QLabel(""); _lbl_unit.setStyleSheet(f"color:{NAVY};font-weight:700;")
    sl.addWidget(_lbl_unit)
    btn_saveset = QtWidgets.QPushButton(T("💾 Save setup", "💾 Guardar setup"))
    btn_saveset.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    _setmsg = QtWidgets.QLabel(""); _setmsg.setStyleSheet("color:#16a34a;font-weight:700;")
    _sr = QtWidgets.QHBoxLayout(); _sr.addWidget(btn_saveset); _sr.addWidget(_setmsg); _sr.addStretch(1)
    sl.addLayout(_sr); sl.addStretch(1)
    tabs.addTab(pg_set, "Setup")

    _SET = QtCore.QSettings("WatermelonSystem", "BalancingSetup")

    def _unit():
        return "mm/s RMS" if cb_sensor.currentIndex() == 0 else "µm pk-pk"

    def _on_sensor(_=0):
        st["unit"] = _unit()
        _lbl_unit.setText(T(f"Balancing unit: {st['unit']}  ·  "
                            + ("casing/absolute → velocity (ISO 20816)" if cb_sensor.currentIndex() == 0
                               else "shaft/relative → displacement (API 670)"),
                            f"Unidad de balanceo: {st['unit']}  ·  "
                            + ("carcasa/absoluta → velocidad (ISO 20816)" if cb_sensor.currentIndex() == 0
                               else "eje/relativo → desplazamiento (API 670)")))
    cb_sensor.currentIndexChanged.connect(_on_sensor)

    def _setup_dict():
        return {"machine": ed_machine.text(), "tag": ed_tag.text(), "client": ed_client.text(),
                "location": ed_loc.text(), "nameplate_rpm": sb_rpm.value(),
                "operator": ed_operator.text(), "approved_by": ed_approved.text(),
                "sensor": cb_sensor.currentIndex(), "unit": _unit()}
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
        try: sb_rpm.setValue(float(_SET.value("nameplate_rpm", 1800) or 1800))
        except Exception: pass  # noqa: BLE001
        try: cb_sensor.setCurrentIndex(int(_SET.value("sensor", 0)))
        except Exception: pass  # noqa: BLE001
    btn_saveset.clicked.connect(_save_setup)

    # ---------- helper: par de campos (magnitud ∠ ángulo) ----------
    def _vec_row(form, label, magmax=1e6, magdef=0.0, unit_suffix=""):
        mag = _dsb(0, magmax, magdef, 3); ang = _dsb(0, 360, 0.0, 1, "°")
        row = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(mag); h.addWidget(QtWidgets.QLabel("∠")); h.addWidget(ang); h.addStretch(1)
        form.addRow(label, row)
        return mag, ang

    # =================================================================
    # TAB 1 — Trial weight (API 684) + ISO
    # =================================================================
    pg_tw = QtWidgets.QWidget(); twl = QtWidgets.QVBoxLayout(pg_tw)
    gb_tw = QtWidgets.QGroupBox(T("Trial weight (API 684)", "Peso de prueba (API 684)")); ftw = QtWidgets.QFormLayout(gb_tw)
    tw_w = _dsb(0.01, 1e5, 500.0, 2, " kg"); tw_rpm = _dsb(1, 30000, 1800, 0, " rpm")
    tw_r = _dsb(1, 5000, 150.0, 1, " mm"); tw_k = _dsb(0.2, 2.0, 1.25, 2)
    ftw.addRow(T("Load on this plane W", "Carga en este plano W"), tw_w)
    ftw.addRow("RPM", tw_rpm); ftw.addRow(T("Radius", "Radio"), tw_r); ftw.addRow("k", tw_k)
    btn_tw = QtWidgets.QPushButton(T("Suggest trial weight", "Sugerir peso de prueba"))
    ftw.addRow("", btn_tw)
    tw_out = QtWidgets.QLabel("—"); tw_out.setStyleSheet(f"color:{NAVY};font-weight:800;font-size:15px;"); ftw.addRow("", tw_out)
    twl.addWidget(gb_tw)

    def _do_tw():
        Wt, Ut = recommend_trial_weight_g(tw_w.value(), tw_rpm.value(), tw_r.value(), tw_k.value())
        tw_out.setText(T(f"→ {Wt:,.2f} g  (U_trial {Ut:,.0f} g·mm)  @ {tw_r.value():.0f} mm",
                         f"→ {Wt:,.2f} g  (U_prueba {Ut:,.0f} g·mm)  @ {tw_r.value():.0f} mm"))
    btn_tw.clicked.connect(_do_tw)

    gb_iso = QtWidgets.QGroupBox(T("ISO 21940 quality check", "Chequeo de calidad ISO 21940")); fiso = QtWidgets.QFormLayout(gb_iso)
    iso_w = _dsb(0.01, 1e5, 500.0, 2, " kg"); iso_rpm = _dsb(1, 30000, 1800, 0, " rpm")
    iso_ures = _dsb(0, 1e7, 0.0, 1, " g·mm")
    iso_g = QtWidgets.QComboBox(); iso_g.addItems(["0.4", "1.0", "2.5", "6.3", "16.0"]); iso_g.setCurrentText("2.5")
    fiso.addRow(T("Rotor mass", "Masa del rotor"), iso_w); fiso.addRow("RPM", iso_rpm)
    fiso.addRow(T("Residual U_res", "Residual U_res"), iso_ures); fiso.addRow(T("Grade G", "Grado G"), iso_g)
    btn_iso = QtWidgets.QPushButton(T("Evaluate ISO", "Evaluar ISO")); fiso.addRow("", btn_iso)
    iso_out = QtWidgets.QLabel("—"); iso_out.setWordWrap(True); iso_out.setStyleSheet("font-weight:700;"); fiso.addRow("", iso_out)
    twl.addWidget(gb_iso); twl.addStretch(1)

    def _do_iso():
        res = evaluate_iso_grades(iso_w.value(), iso_rpm.value(), iso_ures.value())
        st["iso"] = res
        g = float(iso_g.currentText())
        per = next((x for x in res["results"] if abs(x["G"] - g) < 1e-9), None)
        uper = per["U_per"] if per else 0.0
        ok = per["pass"] if per else None
        _v = "#16a34a" if ok else "#dc2626"
        iso_out.setText(
            f"<b>G{g:g}:</b> U_per {uper:,.0f} g·mm · residual {iso_ures.value():,.0f} g·mm → "
            f"<span style='color:{_v};font-weight:800'>"
            + (T("PASS", "CUMPLE") if ok else T("FAIL", "NO CUMPLE") if ok is not None else "—")
            + f"</span><br><span style='color:#64748b'>{res.get('summary_label','')}</span>")
    btn_iso.clicked.connect(_do_iso)
    tabs.addTab(pg_tw, T("Trial + ISO", "Prueba + ISO"))

    # =================================================================
    # TAB 2 — 1 plano
    # =================================================================
    pg1 = QtWidgets.QWidget(); l1 = QtWidgets.QVBoxLayout(pg1)
    l1.addWidget(QtWidgets.QLabel(T("Single-plane balancing (influence coefficient, ISO 21940-12). "
                                    "Enter the 1× vibration (amplitude ∠ phase) and the trial weight.",
                                    "Balanceo en 1 plano (coef. de influencia, ISO 21940-12). "
                                    "Ingresa el 1× (amplitud ∠ fase) y el peso de prueba.")))
    gb1 = QtWidgets.QGroupBox(T("Measurements", "Mediciones")); f1 = QtWidgets.QFormLayout(gb1)
    v0m, v0a = _vec_row(f1, T("V0 — initial vibration", "V0 — vibración inicial"))
    twm, twa = _vec_row(f1, T("Trial weight (g ∠°)", "Peso de prueba (g ∠°)"), 1e5, 10.0)
    vtm, vta = _vec_row(f1, T("Vt — with trial weight", "Vt — con peso de prueba"))
    vfm, vfa = _vec_row(f1, T("Vf — final (optional)", "Vf — final (opcional)"))
    l1.addWidget(gb1)
    _row1 = QtWidgets.QHBoxLayout()
    btn1 = QtWidgets.QPushButton(T("▶ Solve 1-plane", "▶ Resolver 1 plano"))
    btn1.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    btn1_demo = QtWidgets.QPushButton(T("Simulated example", "Ejemplo simulado"))
    _row1.addWidget(btn1); _row1.addWidget(btn1_demo); _row1.addStretch(1); l1.addLayout(_row1)
    out1 = QtWidgets.QLabel("—"); out1.setWordWrap(True)
    out1.setStyleSheet("background:white;border:1px solid #dbe4f0;border-radius:10px;padding:12px;font-size:14px;")
    l1.addWidget(out1); l1.addStretch(1)

    def _solve1():
        try:
            r = solve_1plane(v0m.value(), v0a.value(), vtm.value(), vta.value(), twm.value(), twa.value())
        except ValueError as e:
            out1.setText(f"⚠ {e}"); return
        st["r1p"] = {"unit": st["unit"], "v0": (v0m.value(), v0a.value()),
                     "trial": (twm.value(), twa.value()), "vt": (vtm.value(), vta.value()),
                     "vf": ((vfm.value(), vfa.value()) if vfm.value() > 0 else None), "result": r}
        u = st["unit"]
        out1.setText(T(
            f"<b>Correction weight:</b> {r['corr_mass_g']:,.2f} g ∠ {r['corr_ang_deg']:.1f}°<br>"
            f"Predicted residual: {r['pred_mag']:.3f} {u} · model {r['quality']}<br>"
            f"<span style='color:#64748b'>{r['note']}</span>",
            f"<b>Peso de corrección:</b> {r['corr_mass_g']:,.2f} g ∠ {r['corr_ang_deg']:.1f}°<br>"
            f"Residual predicho: {r['pred_mag']:.3f} {u} · modelo {r['quality']}<br>"
            f"<span style='color:#64748b'>{r['note']}</span>"))

    def _demo1():
        v0m.setValue(8.60); v0a.setValue(63.0); twm.setValue(10.0); twa.setValue(0.0)
        vtm.setValue(6.50); vta.setValue(206.0); _solve1()
    btn1.clicked.connect(_solve1); btn1_demo.clicked.connect(_demo1)
    tabs.addTab(pg1, T("1 plane", "1 plano"))

    # =================================================================
    # TAB 3 — 2 planos
    # =================================================================
    pg2 = QtWidgets.QWidget(); l2 = QtWidgets.QVBoxLayout(pg2)
    l2.addWidget(QtWidgets.QLabel(T("Two-plane balancing. Runs: 0 initial · 1 trial in plane A · 2 trial in plane B. "
                                    "Enter A/B vibration for each run and the trial weights.",
                                    "Balanceo en 2 planos. Corridas: 0 inicial · 1 prueba en plano A · 2 prueba en plano B. "
                                    "Ingresa la vibración A/B de cada corrida y los pesos de prueba.")))
    gb2 = QtWidgets.QGroupBox(T("Measurements", "Mediciones")); f2 = QtWidgets.QFormLayout(gb2)
    a0m, a0a = _vec_row(f2, "A0 — A inicial"); b0m, b0a = _vec_row(f2, "B0 — B inicial")
    wam, waa = _vec_row(f2, T("Trial A (g ∠°)", "Prueba A (g ∠°)"), 1e5, 10.0)
    a1m, a1a = _vec_row(f2, "A1 — A (trial A)"); b1m, b1a = _vec_row(f2, "B1 — B (trial A)")
    wbm, wba = _vec_row(f2, T("Trial B (g ∠°)", "Prueba B (g ∠°)"), 1e5, 10.0)
    a2m, a2a = _vec_row(f2, "A2 — A (trial B)"); b2m, b2a = _vec_row(f2, "B2 — B (trial B)")
    l2.addWidget(gb2)
    btn2 = QtWidgets.QPushButton(T("▶ Solve 2-plane", "▶ Resolver 2 planos"))
    btn2.setStyleSheet(f"QPushButton{{background:{GREEN};}}")
    l2.addWidget(btn2)
    out2 = QtWidgets.QLabel("—"); out2.setWordWrap(True)
    out2.setStyleSheet("background:white;border:1px solid #dbe4f0;border-radius:10px;padding:12px;font-size:14px;")
    l2.addWidget(out2); l2.addStretch(1)

    def _solve2():
        cx = to_complex
        try:
            r = solve_2plane(cx(a0m.value(), a0a.value()), cx(b0m.value(), b0a.value()),
                             cx(a1m.value(), a1a.value()), cx(b1m.value(), b1a.value()),
                             cx(a2m.value(), a2a.value()), cx(b2m.value(), b2a.value()),
                             cx(wam.value(), waa.value()), cx(wbm.value(), wba.value()))
        except ValueError as e:
            out2.setText(f"⚠ {e}"); return
        st["r2p"] = {"unit": st["unit"],
                     "a0": (a0m.value(), a0a.value()), "b0": (b0m.value(), b0a.value()),
                     "a1": (a1m.value(), a1a.value()), "b1": (b1m.value(), b1a.value()),
                     "a2": (a2m.value(), a2a.value()), "b2": (b2m.value(), b2a.value()),
                     "wa": (wam.value(), waa.value()), "wb": (wbm.value(), wba.value()), "result": r}
        wca_m, wca_a = to_polar(r["WA_corr"]); wcb_m, wcb_a = to_polar(r["WB_corr"])
        aa_m, _ = to_polar(r["A_after"]); ba_m, _ = to_polar(r["B_after"]); u = st["unit"]
        out2.setText(T(
            f"<b>Plane A correction:</b> {wca_m:,.2f} g ∠ {wca_a:.1f}°<br>"
            f"<b>Plane B correction:</b> {wcb_m:,.2f} g ∠ {wcb_a:.1f}°<br>"
            f"Predicted residual: A {aa_m:.3f} · B {ba_m:.3f} {u} · model {r['quality']}",
            f"<b>Corrección plano A:</b> {wca_m:,.2f} g ∠ {wca_a:.1f}°<br>"
            f"<b>Corrección plano B:</b> {wcb_m:,.2f} g ∠ {wcb_a:.1f}°<br>"
            f"Residual predicho: A {aa_m:.3f} · B {ba_m:.3f} {u} · modelo {r['quality']}"))
    btn2.clicked.connect(_solve2)
    tabs.addTab(pg2, T("2 planes", "2 planos"))

    # =================================================================
    # TAB 4 — Report + save/cloud
    # =================================================================
    pg_rp = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(pg_rp)
    rl.addWidget(QtWidgets.QLabel(T("Preliminary balancing report (PDF) + save local / upload to cloud. "
                                    "Uses the last 1-plane / 2-plane / ISO results and the Setup data.",
                                    "Reporte preliminar de balanceo (PDF) + guardar local / subir a la nube. "
                                    "Usa los últimos resultados de 1/2 planos e ISO y los datos del Setup.")))
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

    def _meta():
        s = st["setup_fn"]() if st.get("setup_fn") else {}
        from datetime import date as _date
        return {"asset": s.get("machine") or s.get("tag") or "—", "client": s.get("client") or "—",
                "location": s.get("location") or "—", "specialist": s.get("operator") or "—",
                "reviewer": s.get("approved_by") or "—", "unit": st["unit"],
                "rpm": s.get("nameplate_rpm") or 0, "report_date": _date.today().strftime("%d/%m/%Y"),
                "rotation": "CCW"}

    def _build_pdf():
        if not st.get("r1p") and not st.get("r2p"):
            rp_status.setText(T("Solve a 1-plane or 2-plane case first.", "Resuelve un caso de 1 o 2 planos primero.")); return None
        try:
            from core.balance.report import build_balance_pdf
            pdf = build_balance_pdf(meta=_meta(), one_plane=st.get("r1p"), two_plane=st.get("r2p"), iso=st.get("iso"))
            return pdf
        except Exception as exc:  # noqa: BLE001
            rp_status.setText(f"❌ {type(exc).__name__}: {exc}"); return None

    def _gen_pdf():
        pdf = _build_pdf()
        if not pdf:
            return
        _asset = _meta()["asset"]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(win, T("Save report", "Guardar reporte"),
                                                        f"Balanceo_{_asset}.pdf", "PDF (*.pdf)")
        if not path:
            return
        with open(path, "wb") as fh:
            fh.write(pdf)
        rp_status.setText(T(f"✅ Saved: {path}", f"✅ Guardado: {path}"))
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def _payload():
        return {"kind": "balance", "setup": st["setup_fn"]() if st.get("setup_fn") else {},
                "unit": st["unit"], "one_plane": st.get("r1p"), "two_plane": st.get("r2p"),
                "iso": st.get("iso"), "app_version": __version__}

    def _save_local():
        if not st.get("r1p") and not st.get("r2p"):
            rp_status.setText(T("Nothing to save yet.", "Nada que guardar aún.")); return
        import os, json
        from datetime import datetime
        d = os.path.join(os.path.expanduser("~"), "WatermelonBalancing", "runs",
                         datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, "balance.json")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump(_payload(), fh, ensure_ascii=False)
        rp_status.setText(T(f"💾 Saved locally: {fp}", f"💾 Guardado local: {fp}"))

    def _upload():
        if not st.get("r1p") and not st.get("r2p"):
            rp_status.setText(T("Nothing to upload yet.", "Nada que subir aún.")); return
        rp_status.setText(T("☁ Uploading…", "☁ Subiendo…")); QtWidgets.QApplication.processEvents()
        try:
            from core.balance import cloud
            import socket
            s = st["setup_fn"]() if st.get("setup_fn") else {}
            name = (s.get("machine") or s.get("tag") or "Balance") + " · " + T("balancing", "balanceo")
            _acc = _host = ""
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
    tabs.addTab(pg_rp, T("Report", "Reporte"))

    # =================================================================
    # TAB 5 — Updates
    # =================================================================
    pg_up = QtWidgets.QWidget(); ul = QtWidgets.QVBoxLayout(pg_up)
    _cur = QtWidgets.QLabel(T(f"Installed version: v{__version__}", f"Versión instalada: v{__version__}"))
    _ust = QtWidgets.QLabel(T("Press Check for updates.", "Pulsa Buscar actualizaciones.")); _ust.setWordWrap(True)
    _ub = QtWidgets.QPushButton(T("🔍 Check for updates", "🔍 Buscar actualizaciones"))
    ul.addWidget(_cur); ul.addWidget(_ust); ul.addWidget(_ub); ul.addStretch(1)

    def _chk():
        try:
            from core.balance.updater import diagnose
            info, msg = diagnose(__version__)
        except Exception as e:  # noqa: BLE001
            info, msg = None, f"Error: {e}"
        _ust.setText(msg)
    _ub.clicked.connect(_chk)
    tabs.addTab(pg_up, T("Updates", "Actualizaciones"))

    _on_sensor(); _load_setup(); _on_sensor()
    return app, win


def main(argv=None):
    ap = argparse.ArgumentParser(description="Watermelon Balancing (native)")
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
                      "<span style='color:#1AAEE5;font-weight:800;letter-spacing:3px;font-size:22px;'>&nbsp;BALANCING</span>")
            _ok = run_license_gate(_app0, t=T, navy=NAVY, acc=ACC, brand_html=_brand, app_title="Watermelon Balancing")
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001
            if _os.environ.get("WM_LICENSING", "1") == "0":
                _ok = True
            else:
                QtWidgets.QMessageBox.critical(None, "Watermelon Balancing",
                    "Licensing error — the app cannot start.\n\n" + traceback.format_exc()[-800:])
                sys.exit(1)
        if not _ok:
            sys.exit(0)
        app, win = build_app(simulated=True); win.showMaximized()
        sys.exit(app.exec())
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        try:
            with open("watermelon_balancing_error.log", "w", encoding="utf-8") as fh:
                fh.write(err)
            QtWidgets.QMessageBox.critical(None, "Watermelon Balancing — startup error", err[-1500:])
        except Exception:  # noqa: BLE001
            print(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
