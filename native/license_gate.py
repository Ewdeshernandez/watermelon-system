"""
native/license_gate.py — Gate de licencia compartido (Modal, Torsional, …)
==========================================================================

Envuelve `core.modal.licensing` (activación online por clave Ed25519 + machine
binding + revocación + gracia offline) en un diálogo PySide6 reutilizable, para
que TODOS los módulos de campo tengan el MISMO licenciamiento sin duplicar UI.

Modelo de PAQUETE: el estado se guarda en `~/.watermelon/wm_license.json`
(compartido por todos los módulos), así una sola activación en un PC habilita
Modal + Torsional + … — el cliente compra el paquete, activa una vez.

Uso:
    from native.license_gate import run_license_gate
    if not run_license_gate(app, t=T, navy=NAVY, acc=ACC,
                            brand_html="<span…>WATERMELON</span><span…> TORSIONAL</span>",
                            app_title="Watermelon Torsional"):
        sys.exit(0)
"""
from __future__ import annotations

import os
import traceback
from typing import Callable

from PySide6 import QtCore, QtWidgets

# Interruptor global (igual que el Modal): WM_LICENSING=0 salta el gate (dev/Mac).
LICENSING_ENABLED = os.environ.get("WM_LICENSING", "1") != "0"


def _fmt_license_key(text: str) -> str:
    """WM-XXXX-XXXX-XXXX en mayúsculas con guiones automáticos."""
    raw = "".join(c for c in (text or "").upper() if c.isalnum())
    if raw.startswith("WM"):
        rest = raw[2:16]
        groups = [rest[i:i + 4] for i in range(0, len(rest), 4)]
        return "WM" + ("-" + "-".join(groups) if groups else "")
    groups = [raw[i:i + 4] for i in range(0, min(len(raw), 16), 4)]
    return "-".join(groups)


def _reason_text(reason: str, t: Callable[[str, str], str]) -> str:
    r = str(reason or "")
    M = {
        "invalid_key": t("License key not found. Check it and try again.",
                         "Clave de licencia no encontrada. Revísala e intenta de nuevo."),
        "license_expired": t("This license has expired.", "Esta licencia venció."),
        "no_seats": t("No activations left — all seats on this license are in use.",
                      "Sin cupos — todas las máquinas de esta licencia ya están en uso."),
        "machine_revoked": t("This computer's access was revoked.",
                             "El acceso de este equipo fue revocado."),
        "no_server_url": t("Could not reach the activation server.",
                           "No se pudo contactar el servidor de activación."),
        "missing_fields": t("Enter a valid license key.", "Ingresa una clave válida."),
        "machine_mismatch": t("This license is bound to another computer.",
                              "Esta licencia está ligada a otro equipo."),
        "payment_due": t("License not renewed — payment overdue. Contact Watermelon System.",
                         "Licencia no renovada por falta de pago. Contacta a Watermelon System."),
        "license_revoked": t("This license was revoked. Contact Watermelon System.",
                             "Esta licencia fue revocada. Contacta a Watermelon System."),
        "offline_too_long": t("Please connect to the internet to re-verify your license.",
                              "Conéctate a internet para re-verificar tu licencia."),
    }
    if r in M:
        return M[r]
    if r.startswith("activate_failed"):
        return t("Could not reach the activation server. Check your internet.",
                 "No se pudo contactar el servidor. Revisa tu internet.")
    return t("Could not activate. ", "No se pudo activar. ") + r


def _activation_dialog(lic, *, t, navy, acc, brand_html, app_title,
                       initial_reason: str = "") -> bool:
    """Ventana de activación por CLAVE. True si quedó activada.
    `initial_reason` = motivo por el que el gate bloqueó (p.ej. 'payment_due') →
    se muestra ARRIBA, antes de que el cliente intente nada."""
    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle(t(f"Activate {app_title}", f"Activar {app_title}"))
    dlg.setFixedWidth(500); dlg.setModal(True)
    dlg.setStyleSheet(f"QDialog{{background:{navy};}}")
    v = QtWidgets.QVBoxLayout(dlg); v.setContentsMargins(38, 34, 38, 30); v.setSpacing(0)
    head = QtWidgets.QLabel(brand_html)
    head.setTextFormat(QtCore.Qt.RichText); head.setAlignment(QtCore.Qt.AlignCenter)
    v.addWidget(head)
    sub = QtWidgets.QLabel(t("Activate this computer", "Activa este equipo"))
    sub.setAlignment(QtCore.Qt.AlignCenter)
    sub.setStyleSheet("color:#5b6b86; font-weight:600; letter-spacing:1px; font-size:11px; margin-top:2px;")
    v.addWidget(sub); v.addSpacing(22)
    lbl = QtWidgets.QLabel(t("License key", "Clave de licencia"))
    lbl.setStyleSheet("color:#93c5fd; font-weight:700; font-size:11px; letter-spacing:1px;")
    v.addWidget(lbl); v.addSpacing(6)
    e_key = QtWidgets.QLineEdit(); e_key.setPlaceholderText("WM-XXXX-XXXX-XXXX")
    e_key.setMaxLength(17); e_key.setAlignment(QtCore.Qt.AlignCenter)
    e_key.setStyleSheet("QLineEdit{background:white;color:#0f172a;border:2px solid #2a3a57;border-radius:10px;"
                        "padding:13px;font-family:monospace;font-size:20px;font-weight:700;letter-spacing:3px;}"
                        "QLineEdit:focus{border:2px solid #1AAEE5;}")
    v.addWidget(e_key)

    def _on_edit(_tx):
        e_key.blockSignals(True); e_key.setText(_fmt_license_key(_tx))
        e_key.setCursorPosition(len(e_key.text())); e_key.blockSignals(False)
    e_key.textEdited.connect(_on_edit)

    msg = QtWidgets.QLabel(""); msg.setWordWrap(True); msg.setAlignment(QtCore.Qt.AlignCenter)
    msg.setStyleSheet("color:#fca5a5; font-size:12px; margin-top:8px;")
    if initial_reason:
        # Motivo de bloqueo visible de entrada (falta de pago, vencida, revocada…).
        _amber = str(initial_reason).startswith(("payment_due", "license_expired",
                                                 "offline_too_long"))
        msg.setStyleSheet("color:%s; font-size:13px; font-weight:700; margin-top:8px;"
                          % ("#fbbf24" if _amber else "#fca5a5"))
        msg.setText(_reason_text(initial_reason, t))
    v.addWidget(msg); v.addSpacing(20)
    row = QtWidgets.QHBoxLayout()
    b_quit = QtWidgets.QPushButton(t("Quit", "Salir"))
    b_quit.setStyleSheet("QPushButton{background:transparent;color:#8ea0bd;border:1px solid #2a3a57;"
                         "border-radius:9px;padding:11px 20px;font-weight:700;} QPushButton:hover{color:#dbe6f5;}")
    b_act = QtWidgets.QPushButton(t("Activate", "Activar"))
    b_act.setCursor(QtCore.Qt.PointingHandCursor)
    b_act.setStyleSheet(f"QPushButton{{background:{acc};color:#08243a;font-weight:800;border-radius:9px;"
                        "padding:11px 26px;font-size:14px;} QPushButton:hover{background:#38b6e6;}")
    row.addWidget(b_quit); row.addStretch(1); row.addWidget(b_act); v.addLayout(row)
    fp = lic.machine_fingerprint()
    foot = QtWidgets.QLabel(t("Machine ID: ", "ID de máquina: ") + fp[:20] + "…  ·  "
                            + t("Need a key? Contact Watermelon System.",
                                "¿No tienes clave? Contacta a Watermelon System."))
    foot.setAlignment(QtCore.Qt.AlignCenter)
    foot.setStyleSheet("color:#44526b; font-size:10px; margin-top:16px;")
    v.addSpacing(6); v.addWidget(foot)
    state = {"ok": False}

    def _do_activate():
        if len(e_key.text().strip()) < 8:
            msg.setText(t("Enter your license key.", "Ingresa tu clave de licencia.")); return
        b_act.setEnabled(False); b_act.setText(t("Activating…", "Activando…"))
        QtWidgets.QApplication.processEvents()
        r = lic.activate_with_key(e_key.text().strip())
        if r.get("ok"):
            state["ok"] = True; dlg.accept()
        else:
            msg.setText(_reason_text(r.get("reason", ""), t))
            b_act.setEnabled(True); b_act.setText(t("Activate", "Activar"))
    b_act.clicked.connect(_do_activate)
    e_key.returnPressed.connect(_do_activate)
    b_quit.clicked.connect(dlg.reject)
    dlg.exec()
    return state["ok"]


def _block_dialog(detail: str, *, t, app_title) -> None:
    """FAIL-CLOSED: la capa de licencia no pudo verificar → NO se abre la app."""
    try:
        box = QtWidgets.QMessageBox()
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle(app_title.upper())
        box.setText(t("Licensing verification unavailable — the app cannot start.",
                      "No se pudo verificar la licencia — la app no puede iniciar."))
        box.setInformativeText(
            t("Please reinstall the latest version or contact Watermelon System.",
              "Reinstala la última versión o contacta a Watermelon System.")
            + "\n\n" + detail[-400:])
        box.setStandardButtons(QtWidgets.QMessageBox.Close)
        box.exec()
    except Exception:  # noqa: BLE001
        print("LICENSE BLOCK:", detail)


def run_license_gate(app, *, t, navy, acc, brand_html, app_title) -> bool:
    """Gate FAIL-CLOSED. True solo con licencia válida (o activación exitosa).
    Reutiliza `core.modal.licensing` (mismo esquema/tabla/edge functions que el Modal).
    El estado vive en ~/.watermelon/wm_license.json → una activación cubre todos los módulos."""
    if not LICENSING_ENABLED:
        return True
    try:
        from core.modal import licensing as lic
    except Exception:  # noqa: BLE001 — módulo de seguridad ausente en la build → BLOQUEA
        _block_dialog("import core.modal.licensing failed:\n" + traceback.format_exc(),
                      t=t, app_title=app_title)
        return False
    try:
        g = lic.gate_check()
    except Exception:  # noqa: BLE001 — error inesperado del gate → BLOQUEA
        _block_dialog("gate_check() raised:\n" + traceback.format_exc(), t=t, app_title=app_title)
        return False
    if g.get("allowed"):
        return True
    # Bloqueado: si el servidor dio un motivo comercial (falta de pago, vencida,
    # revocada), se muestra de entrada en el diálogo — el cliente ve el letrero.
    _reason = "" if g.get("needs_activation") and g.get("reason") in (None, "no_license") \
        else str(g.get("reason") or "")
    return _activation_dialog(lic, t=t, navy=navy, acc=acc,
                              brand_html=brand_html, app_title=app_title,
                              initial_reason=_reason)
