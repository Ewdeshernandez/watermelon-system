"""
scripts/send_alarm_reports.py
=============================

Cron headless de envío automático POR ALARMA (Fase 3). Pensado para un
Render Cron Job que corre cada 15 minutos (schedule "*/15 * * * *").

Lógica "1 aviso por episodio":
    - Para cada activo con `alarm_send_enabled = True` y destinatario,
      calcula el nivel de severidad ACTUAL (barato, sin armar PDF):
          0 = Normal/Sin datos · 1 = Alarma · 2 = Danger
    - Compara con `alarm_alert_level` (último nivel ya avisado, persistido).
    - Si el nivel EMPEORA (level > alarm_alert_level): genera el PDF y lo
      envía marcado como ALERTA, y guarda alarm_alert_level = level.
      → Esto avisa al ENTRAR en Alarma y otra vez si ESCALA a Danger.
    - Si vuelve a Normal (level == 0) y antes estaba avisado: resetea
      alarm_alert_level = 0 (sin enviar). Así el próximo cruce vuelve a avisar.
    - Si sigue igual o mejora pero todavía en alarma: NO reenvía (anti-spam).

Uso (Render Cron Job, cada 15 min):
    mkdir -p .streamlit && cp /etc/secrets/secrets.toml .streamlit/secrets.toml \
        && python scripts/send_alarm_reports.py

Flags para pruebas:
    --instance <id>   Procesa SOLO ese activo.
    --force           Envía si hay alarma (level>0) ignorando el estado guardado.
    --dry-run         Evalúa y loggea, pero NO envía ni persiste estado.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("wm.alarm_reports")

_LEVEL_NAME = {0: "Normal", 1: "Alarma", 2: "Danger"}


def _has_recipient(inst) -> bool:
    email = (getattr(inst, "client_email", "") or "").strip()
    wa = (getattr(inst, "whatsapp_number", "") or "").strip()
    return bool((email and "@" in email) or wa)


def _data_age_minutes(iid: str):
    """Edad (min) del dato MÁS reciente del activo, o None si nunca hubo datos.
    Sirve para el heartbeat: distinguir 'sin servicio' (None) de 'se calló' (grande)."""
    from datetime import datetime, timezone
    try:
        from core.live_readings import latest_for_instance
        rows = latest_for_instance(iid) or []
    except Exception:  # noqa: BLE001
        return None
    ts = []
    for r in rows:
        c = r.get("captured_at")
        if not c:
            continue
        try:
            d = datetime.fromisoformat(str(c).replace("Z", "+00:00"))
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            ts.append(d)
        except Exception:  # noqa: BLE001
            pass
    if not ts:
        return None
    return (datetime.now(timezone.utc) - max(ts)).total_seconds() / 60.0


def _send_offline_alert(inst, tag: str, age_min: float, iid: str = "") -> bool:
    """Aviso OFFLINE (dead-man-switch). Preferido: PDF ejecutivo marcado
    FUERA DE LÍNEA con los ÚLTIMOS datos medidos (email + WhatsApp) — para
    activos con servicio contratado (ej. SGT300A) que están parados o sin
    enlace. Fallback: email de texto si no se pudo armar el PDF."""
    if iid:
        try:
            from core.live_report_builder import build_report_for_instance
            from core.report_delivery import deliver_report
            pdf, meta = build_report_for_instance(iid, inst, offline_age_min=age_min)
            if pdf:
                res = deliver_report(inst, pdf, meta, alert=True)
                if res.get("any_ok"):
                    return True
                log.warning("   %s: entrega PDF offline sin éxito — fallback texto.", tag)
        except Exception as e:  # noqa: BLE001
            log.error("   %s: PDF offline falló (%s) — fallback a email de texto.", tag, e)

    email = (getattr(inst, "client_email", "") or "").strip()
    tos = [e.strip() for e in email.replace(";", ",").split(",") if "@" in e]
    if not tos:
        return False
    hrs = age_min / 60.0
    subject = f"⚠ {tag}: SIN DATOS hace {hrs:.1f} h — posible pérdida de comunicación"
    body = (f"El activo {tag} dejó de reportar al sistema de monitoreo hace "
            f"{hrs:.1f} h.\n\nPosible causa: enlace/colector detenido, sensor o "
            f"comunicación caída. El monitoreo en línea NO está recibiendo datos "
            f"de este equipo.\n\nAcción: revisar el colector y el enlace del sitio.\n\n"
            f"— Watermelon System (aviso automático)")
    ok = False
    try:
        from core.email_sender import send_email
        for to in tos:
            r = send_email(to, subject, body)
            ok = ok or bool(r and (r.get("ok") if isinstance(r, dict) else r))
    except Exception as e:  # noqa: BLE001
        log.error("   %s: fallo enviando aviso offline: %s", tag, e)
    return ok


def process(only_instance: str = "", force: bool = False, dry_run: bool = False) -> int:
    from core.instance_state import list_instances, get_instance, update_instance_header
    from core.live_report_builder import current_severity_level, build_report_for_instance
    from core.report_delivery import deliver_report

    if only_instance:
        ids = [only_instance]
    else:
        try:
            ids = [row.get("instance_id") for row in (list_instances() or [])
                   if row.get("instance_id")]
        except Exception as e:
            log.error("No se pudo listar activos: %s", e)
            return 1

    log.info("Chequeo de alarmas · %d activo(s) a evaluar", len(ids))
    sent = skipped = reset = errors = 0

    for iid in ids:
        inst = get_instance(iid)
        if inst is None:
            continue
        if not getattr(inst, "alarm_send_enabled", False):
            skipped += 1
            continue
        if not _has_recipient(inst):
            log.warning("Activo %s con alarma activa pero SIN email/WhatsApp — salteado.", iid)
            skipped += 1
            continue

        tag = getattr(inst, "tag", "") or iid
        stored = int(getattr(inst, "alarm_alert_level", 0) or 0)

        # --- Heartbeat / dead-man-switch (nivel centinela 3 = OFFLINE) ---
        # Si el activo TENÍA datos y se calló > umbral → avisa 1 vez (no re-spam).
        # None = nunca reportó (sin servicio) → no aplica. Además evita puntuar
        # severidad sobre data vieja como si fuera en vivo.
        _off_min = int(os.environ.get("WM_OFFLINE_MINUTES", "60") or 60)
        _age = _data_age_minutes(iid)
        if _age is not None and _age > _off_min:
            if stored != 3:
                if dry_run:
                    log.info("   %s: DRY RUN — OFFLINE %.0f min (habría avisado).", tag, _age)
                elif _send_offline_alert(inst, tag, _age, iid=iid):
                    update_instance_header(iid, alarm_alert_level=3)
                    log.warning("⚠ %s OFFLINE (%.0f min sin datos) — aviso enviado.", tag, _age)
                    sent += 1
                else:
                    errors += 1
            else:
                skipped += 1
            continue
        if stored == 3:                      # reconectó (datos frescos) → re-armar
            if not dry_run:
                update_instance_header(iid, alarm_alert_level=0)
            stored = 0
            log.info("✓ %s reconectado — datos frescos, alarma re-armada.", tag)

        # Event log persistente: registra cruces de umbral por canal (aunque
        # nadie esté viendo la web). Idempotente — solo inserta cambios.
        if not dry_run:
            try:
                from core.live_readings import latest_for_instance as _lfi
                from core.live_report_builder import (
                    _build_sensor_lookup as _bsl, _compute_rendered_rows as _crr)
                from core.severity_events import record_events as _rec
                _lt = _lfi(iid) or []
                if _lt:
                    _rows, _ = _crr(_lt, _bsl(inst), inst)
                    _nrec = _rec(iid, _rows)
                    if _nrec:
                        log.info("   %s: %d evento(s) de umbral registrados.", tag, _nrec)
            except Exception as e:  # noqa: BLE001
                log.warning("   %s: record_events falló: %s", tag, e)

        level, status, summary = current_severity_level(iid, inst)

        # ¿Hay que avisar? Solo si EMPEORA respecto a lo ya avisado
        # (o --force con cualquier alarma activa).
        should_alert = (level > stored) or (force and level > 0)

        if should_alert:
            log.info("→ %s: nivel %s (antes %s) — generando aviso…",
                     tag, _LEVEL_NAME.get(level, level), _LEVEL_NAME.get(stored, stored))
            # Ciclo 23.157 — alarm_focus: tendencia 48 h SOLO de los
            # canales en alarma, con límites alarma/danger.
            pdf_bytes, meta = build_report_for_instance(iid, inst,
                                                        alarm_focus=True)
            if not pdf_bytes:
                log.error("   %s: no se pudo generar el PDF (¿sin lecturas?).", tag)
                errors += 1
                continue
            if dry_run:
                log.info("   %s: DRY RUN — no se envía (habría avisado nivel %s).",
                         tag, _LEVEL_NAME.get(level, level))
                continue
            res = deliver_report(inst, pdf_bytes, meta, alert=True)
            em, wa = res.get("email"), res.get("whatsapp")
            if em is not None:
                log.info("   %s email: %s", tag, "OK" if em.get("ok") else f"FALLA · {em.get('error')}")
            if wa is not None:
                log.info("   %s whatsapp: %s", tag, "OK" if wa.get("ok") else f"FALLA · {wa.get('error')}")
            if res.get("any_ok"):
                sent += 1
                update_instance_header(iid, alarm_alert_level=level)
            else:
                errors += 1

        elif level == 0 and stored != 0:
            # Volvió a Normal → resetear el estado para que el próximo cruce avise
            if not dry_run:
                update_instance_header(iid, alarm_alert_level=0)
            log.info("✓ %s: normalizado — estado de alarma reseteado.", tag)
            reset += 1
        else:
            skipped += 1

    log.info("Listo · avisos=%d reset=%d salteados=%d errores=%d", sent, reset, skipped, errors)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Envío automático por alarma")
    p.add_argument("--instance", default="", help="Procesar SOLO este activo (id)")
    p.add_argument("--force", action="store_true", help="Avisar si hay alarma, ignorando estado")
    p.add_argument("--dry-run", action="store_true", help="Evaluar sin enviar ni persistir")
    args = p.parse_args()
    sys.exit(process(only_instance=args.instance, force=args.force, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
