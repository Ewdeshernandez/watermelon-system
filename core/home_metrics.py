"""
core.home_metrics
=================

Helpers de cálculo para el Home/landing rediseñado en Ciclo 17.11.

Provee:
  - Saludo personalizado según hora del día y nombre del usuario
  - Cómputo de salud de la flota (compute_fleet_status)
  - Actividad reciente combinando metadata de instancias + report drafts
  - Sparklines de actividad por día (últimos 7 días)
  - Estado del sistema (env, vault sync, etc.)

Filosofía: este módulo NO tiene dependencia de Streamlit y es
deterministicamente testeable. La capa UI vive en pages/_landing.py.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INSTANCES_DIR = DATA_DIR / "instances"
REPORT_DRAFTS_DIR = DATA_DIR / "report_drafts"
REPORT_STATE_FILE = DATA_DIR / "report_state.json"


# =============================================================
# SALUDO PERSONALIZADO
# =============================================================

def get_personalized_greeting(
    full_name: str = "",
    now: Optional[datetime] = None,
    tz_name: str = "",
) -> Dict[str, str]:
    """Devuelve saludo + turno + fecha legible.

    Ciclo 17.24.5: si se pasa `tz_name` (string IANA tipo
    "America/Bogota", "America/Los_Angeles"), la hora se calcula
    en esa timezone. Si no, usa la hora local del server (que en
    Streamlit Cloud es UTC). El formato del reloj es 12h con am/pm
    en minúsculas (ej. "4:40 pm").

    Returns:
        {
          "greeting":   "Buenas tardes, Ewdes",
          "shift":      "Turno tarde",
          "shift_emoji": "☀️",
          "date_long":  "martes · 05 may 2026",
          "time_hhmm":  "4:40 pm",     ← 12h con am/pm
        }
    """
    if now is None:
        # Si nos pasaron tz, calculamos UTC y convertimos
        if tz_name:
            try:
                from zoneinfo import ZoneInfo
                from datetime import timezone as _tz
                _utc_now = datetime.now(_tz.utc)
                now = _utc_now.astimezone(ZoneInfo(tz_name))
            except Exception:
                # Si la tz es inválida o zoneinfo no está, fallback al server
                now = datetime.now()
        else:
            now = datetime.now()

    hour = now.hour

    # Saludo
    if 5 <= hour < 12:
        greet = "Buenos días"
    elif 12 <= hour < 19:
        greet = "Buenas tardes"
    else:
        greet = "Buenas noches"

    # Turno
    if 6 <= hour < 14:
        shift, shift_emoji = "Turno mañana", "🌅"
    elif 14 <= hour < 22:
        shift, shift_emoji = "Turno tarde", "☀️"
    else:
        shift, shift_emoji = "Turno noche", "🌙"

    # Fecha en español
    DAYS_ES = ["lunes", "martes", "miércoles", "jueves",
               "viernes", "sábado", "domingo"]
    MONTHS_ES = ["ene", "feb", "mar", "abr", "may", "jun",
                 "jul", "ago", "sep", "oct", "nov", "dic"]
    weekday = DAYS_ES[now.weekday()]
    month = MONTHS_ES[now.month - 1]
    date_long = f"{weekday} · {now.day:02d} {month} {now.year}"

    # Formato 12h con am/pm (Ciclo 17.24.5)
    h12 = hour % 12
    if h12 == 0:
        h12 = 12
    ampm = "pm" if hour >= 12 else "am"
    time_hhmm = f"{h12}:{now.minute:02d} {ampm}"

    name = (full_name or "").strip().split()[0] if full_name else ""
    greeting = f"{greet}, {name}" if name else greet

    return {
        "greeting": greeting,
        "shift": shift,
        "shift_emoji": shift_emoji,
        "date_long": date_long,
        "time_hhmm": time_hhmm,
    }


# =============================================================
# SALUD DE INSTANCIA (heurística inicial 17.11)
# =============================================================
#
# Severity según campos del Vault:
#   healthy  — norma asignada + ≥1 documento + last_balance_date
#   warning  — norma asignada pero falta documentación o baseline
#   danger   — (reservado para 17.12 cuando persistamos la severidad
#              ejecutiva calculada en el PDF)
#   unknown  — instancia creada sin norma asignada
#
# En 17.12 reemplazaremos esto por last_executive_severity persistido
# por el PDF generator. Por ahora la heurística da una señal útil.

@dataclass
class InstanceHealth:
    instance_id: str = ""
    tag: str = ""
    asset_class: str = ""
    profile_key: str = ""
    location: str = ""
    severity: str = "unknown"   # healthy | warning | danger | unknown
    severity_label: str = ""
    severity_color: str = "#94a3b8"
    severity_dot: str = "⚪"
    last_seen: str = ""
    last_seen_human: str = ""
    has_norm: bool = False
    n_documents: int = 0


_SEVERITY_META = {
    "healthy": ("OK",         "#10b981", "🟢"),
    "warning": ("Atención",   "#f59e0b", "🟡"),
    "danger":  ("Crítico",    "#ef4444", "🔴"),
    "unknown": ("Sin norma",  "#94a3b8", "⚪"),
}


# Ciclo 17.13 — Mapeo de labels de severidad ejecutiva (literales del
# PDF generator en pages/16_Reports.py) a las 4 bandas del Home.
# Se chequea con normalize() simple para tolerar mayús/minús/acentos.
_EXEC_SEVERITY_TO_BAND = {
    "critica":            "danger",
    "accion requerida":   "danger",
    "atencion":           "warning",
    "vigilancia":         "warning",
    "condicion aceptable":"healthy",
    # Variantes ya usadas en el helper antiguo (back-compat)
    "danger":             "danger",
    "warning":            "warning",
    "healthy":            "healthy",
}


def _norm_no_accents(s: str) -> str:
    """Normaliza string a lower sin acentos para comparar severidad."""
    if not s:
        return ""
    s = s.lower().strip()
    repl = {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n","ü":"u"}
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def _heuristic_severity(inst_summary: Dict[str, Any],
                         full_metadata: Optional[Dict[str, Any]] = None) -> str:
    """Calcula severity para una instancia.

    Ciclo 17.13: si la instancia tiene `last_executive_severity` persistido
    desde el último PDF, USA ESE valor (estado real del activo) en lugar
    de la heurística de configuración. La heurística sólo aplica cuando
    el activo nunca fue analizado.
    """
    has_norm = False
    has_balance = False
    n_docs = int(inst_summary.get("n_documents", 0) or 0)

    if full_metadata:
        has_norm = bool((full_metadata.get("iso_norm_code") or "").strip())
        has_balance = bool((full_metadata.get("last_balance_date") or "").strip())
        # Severidad ejecutiva persistida del último PDF — toma prioridad
        explicit = _norm_no_accents(full_metadata.get("last_executive_severity", ""))
        mapped = _EXEC_SEVERITY_TO_BAND.get(explicit)
        if mapped:
            return mapped

    if has_norm and n_docs >= 1 and has_balance:
        return "healthy"
    if has_norm:
        return "warning"
    return "unknown"


def _humanize_age(iso_ts: str, now: Optional[datetime] = None) -> str:
    """Convierte ISO timestamp a 'hace 3 min', 'hace 2 h', 'ayer', etc."""
    if not iso_ts:
        return "—"
    try:
        ts = datetime.fromisoformat(iso_ts.replace("Z", ""))
    except Exception:
        return iso_ts[:16]
    delta = (now or datetime.now()) - ts
    sec = int(delta.total_seconds())
    if sec < 60:
        return "hace unos segundos"
    if sec < 3600:
        return f"hace {sec // 60} min"
    if sec < 86400:
        return f"hace {sec // 3600} h"
    days = sec // 86400
    if days == 1:
        return "ayer"
    if days < 7:
        return f"hace {days} días"
    if days < 30:
        return f"hace {days // 7} sem"
    if days < 365:
        return f"hace {days // 30} meses"
    return f"hace {days // 365} años"


def compute_fleet_status(client_email: str = "") -> Dict[str, Any]:
    """Computa el estado completo de la flota.

    Args:
        client_email: si se provee y pertenece a un cliente registrado
            en clients.json (role=client), filtra las instances por los
            asset_tags configurados para ese cliente. Para admin/specialist
            dejar en blanco para ver toda la flota.

    Returns:
        {
          "total":   int,
          "by_severity": {"healthy": N, "warning": N, "danger": N, "unknown": N},
          "instances": [InstanceHealth, ...]   # ordenadas por severidad luego updated_at desc
        }
    """
    try:
        from core.instance_state import list_instances, get_instance
    except Exception:
        return {"total": 0, "by_severity": {}, "instances": []}

    summaries = list_instances() or []

    # Ciclo 23.131 — Scoping por client_email cuando el caller es role=client.
    # Filtra summaries por asset_tags del cliente registrado en data/clients.json.
    if client_email:
        try:
            from core.clients import filter_instances_for_email
            summaries = filter_instances_for_email(summaries, client_email)
        except Exception:
            pass

    # Ciclo 23.55b (v3.31.261/263) — Filtrar SOLO la instancia "(default)"
    # creada automáticamente por el sistema. CONSERVADOR: NO filtrar
    # por tag vacío solo (algunas instancias reales como 'tes1' tienen
    # tag vacío pero data real con sensores y monitoreo en línea).
    # Solo el tag literal "(default)" / variantes obvias son seguras.
    def _is_placeholder_summary(s: Dict[str, Any]) -> bool:
        tag = (s.get("tag") or "").strip()
        # Solo tags LITERALES creados por el sistema. Tag vacío
        # NO es suficiente.
        return tag in ("(default)", "default", "(sin tren)", "(sin nombre)")

    summaries = [s for s in summaries if not _is_placeholder_summary(s)]
    healths: List[InstanceHealth] = []
    counts = {"healthy": 0, "warning": 0, "danger": 0, "unknown": 0}

    for s in summaries:
        iid = s.get("instance_id", "")
        full = None
        try:
            inst = get_instance(iid)
            if inst is not None:
                # Convertir Instance → dict mínimo para evaluar
                full = {
                    "iso_norm_code": getattr(inst, "iso_norm_code", "") or "",
                    "last_balance_date": getattr(inst, "last_balance_date", "") or "",
                    "last_executive_severity": getattr(inst, "last_executive_severity", "") or "",
                    "last_executive_summary": getattr(inst, "last_executive_summary", "") or "",
                    "last_report_date": getattr(inst, "last_report_date", "") or "",
                    "asset_class": getattr(inst, "asset_class", "") or "",
                }
        except Exception:
            full = None

        sev = _heuristic_severity(s, full)
        counts[sev] = counts.get(sev, 0) + 1

        label, color, dot = _SEVERITY_META.get(sev, _SEVERITY_META["unknown"])
        h = InstanceHealth(
            instance_id=iid,
            tag=s.get("tag", "") or iid,
            asset_class=(full or {}).get("asset_class", "") if full else "",
            profile_key=s.get("profile_key", "") or "",
            location=s.get("location", "") or "",
            severity=sev,
            severity_label=label,
            severity_color=color,
            severity_dot=dot,
            last_seen=s.get("updated_at", "") or "",
            last_seen_human=_humanize_age(s.get("updated_at", "") or ""),
            has_norm=bool((full or {}).get("iso_norm_code", "")),
            n_documents=int(s.get("n_documents", 0) or 0),
        )
        healths.append(h)

    # Orden: danger > warning > unknown > healthy, luego por last_seen desc
    sev_order = {"danger": 0, "warning": 1, "unknown": 2, "healthy": 3}
    healths.sort(key=lambda x: (sev_order.get(x.severity, 9), -_iso_to_epoch(x.last_seen)))

    return {
        "total": len(healths),
        "by_severity": counts,
        "instances": healths,
    }


def _iso_to_epoch(iso_ts: str) -> int:
    if not iso_ts:
        return 0
    try:
        return int(datetime.fromisoformat(iso_ts.replace("Z", "")).timestamp())
    except Exception:
        return 0


# =============================================================
# ACTIVITY FEED
# =============================================================

@dataclass
class ActivityEvent:
    timestamp: datetime = field(default_factory=datetime.now)
    kind: str = ""           # report | instance_edit | csv_load | report_archived
    icon: str = "•"
    title: str = ""
    subtitle: str = ""
    age_human: str = ""
    owner_email: str = ""    # Ciclo 17.15 — quién hizo la acción


def _email_to_initials(email: str) -> str:
    """Convierte 'jane.doe@sigasas.com' a 'JD'. Útil para avatar visual."""
    name = (email or "").split("@")[0].strip()
    if not name:
        return "?"
    parts = re.split(r"[._-]+", name)
    if len(parts) >= 2 and parts[0] and parts[1]:
        return (parts[0][0] + parts[1][0]).upper()
    return (name[:2]).upper()


def list_recent_activity(limit: int = 12,
                          viewer_email: str = "",
                          viewer_role: str = "",
                          owner_filter: str = "") -> List[ActivityEvent]:
    """Combina eventos de:
       - Edits a metadata.json de instancias (instance_edit)
       - Drafts de reportes per-usuario (report)
       - report_state.json activo per-usuario (current_report)
       - Archivos PDF inmutables (report_archived)

    Ciclo 17.15:
      - viewer_role + viewer_email: filtros de visibilidad
        * admin → ve actividad de TODOS los usuarios
        * specialist → ve la suya + la de otros @sigasas.com
        * client → ve solo la SUYA
      - owner_filter (opcional): filtra por email específico para que
        admin/specialist puedan ver actividad de un usuario puntual

    Devuelve top N ordenados por timestamp desc.
    """
    events: List[ActivityEvent] = []
    now = datetime.now()
    role = (viewer_role or "").strip().lower()
    me = (viewer_email or "").strip().lower()
    owner_filter_l = (owner_filter or "").strip().lower()

    def _can_see(event_owner: str) -> bool:
        """Filtro central de visibilidad por role."""
        eo = (event_owner or "").lower()
        if not me:
            return True  # caso edge: no auth, mostrar todo (ej. tests)
        if owner_filter_l:
            # Si hay filtro explícito, respetarlo (independiente del role)
            return eo == owner_filter_l
        if role == "admin":
            return True
        if role == "specialist":
            if eo == me or not eo:
                return True
            return eo.endswith("@sigasas.com")
        # client (default conservador)
        return eo == me

    # ─── 1. Instance edits (Vault) — son globales (sin owner)
    if INSTANCES_DIR.exists():
        for child in INSTANCES_DIR.iterdir():
            if not child.is_dir():
                continue
            meta = child / "metadata.json"
            if not meta.exists():
                continue
            try:
                ts = datetime.fromtimestamp(meta.stat().st_mtime)
                tag = child.name
                try:
                    import json
                    with open(meta, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    tag = data.get("tag") or tag
                except Exception:
                    pass
                events.append(ActivityEvent(
                    timestamp=ts,
                    kind="instance_edit",
                    icon="🛠️",
                    title=f"Vault: metadata de {tag} actualizada",
                    subtitle=f"Vault · {child.name}",
                    age_human=_humanize_age(ts.isoformat(timespec="seconds"), now),
                    owner_email="",  # vault es compartido
                ))
            except Exception:
                continue

    # ─── 2. Drafts de reportes per-usuario (Ciclo 17.15)
    users_root = DATA_DIR / "users"
    if users_root.exists():
        for user_dir in users_root.iterdir():
            if not user_dir.is_dir():
                continue
            # Recuperar email del owner desde el state si está
            owner_email_for_user = ""
            state_file = user_dir / "report_state.json"
            if state_file.exists():
                try:
                    import json
                    raw = json.loads(state_file.read_text(encoding="utf-8"))
                    owner_email_for_user = (raw.get("_save_meta", {}) or {}).get("owner_email", "") or ""
                except Exception:
                    pass
            if not owner_email_for_user:
                # Reconstruir desde el slug si no está en _save_meta
                owner_email_for_user = user_dir.name.replace("_at_", "@")

            if not _can_see(owner_email_for_user):
                continue

            # Drafts nombrados
            drafts_dir = user_dir / "report_drafts"
            if drafts_dir.exists():
                for path in drafts_dir.glob("*.json"):
                    try:
                        ts = datetime.fromtimestamp(path.stat().st_mtime)
                        events.append(ActivityEvent(
                            timestamp=ts,
                            kind="report",
                            icon="📄",
                            title=f"Draft '{path.stem}'",
                            subtitle=f"{owner_email_for_user} · Reports",
                            age_human=_humanize_age(ts.isoformat(timespec="seconds"), now),
                            owner_email=owner_email_for_user,
                        ))
                    except Exception:
                        continue

            # Estado actual del reporte (current_report)
            if state_file.exists():
                try:
                    ts = datetime.fromtimestamp(state_file.stat().st_mtime)
                    events.append(ActivityEvent(
                        timestamp=ts,
                        kind="current_report",
                        icon="📋",
                        title="Reporte en curso actualizado",
                        subtitle=f"{owner_email_for_user} · Reports",
                        age_human=_humanize_age(ts.isoformat(timespec="seconds"), now),
                        owner_email=owner_email_for_user,
                    ))
                except Exception:
                    pass

    # ─── 3. Reportes archivados PDF (Ciclo 17.15)
    archive_root = DATA_DIR / "reports_archive"
    if archive_root.exists():
        for sidecar in archive_root.rglob("*.json"):
            try:
                import json
                sc = json.loads(sidecar.read_text(encoding="utf-8"))
                _ow = sc.get("owner_email", "")
                if not _can_see(_ow):
                    continue
                ts_iso = sc.get("archived_at", "")
                ts = datetime.fromisoformat(ts_iso) if ts_iso else datetime.fromtimestamp(sidecar.stat().st_mtime)
                rm = sc.get("report_meta", {}) or {}
                _client = rm.get("client", "—")
                _asset = rm.get("asset_class") or rm.get("instance_tag") or "—"
                events.append(ActivityEvent(
                    timestamp=ts,
                    kind="report_archived",
                    icon="📦",
                    title=f"Archivado: {_client} · {_asset}",
                    subtitle=f"{_ow} · {sc.get('size_human','')}",
                    age_human=_humanize_age(ts.isoformat(timespec="seconds"), now),
                    owner_email=_ow,
                ))
            except Exception:
                continue

    # ─── 4. Legacy (estado pre-17.15) si todavía existe — solo admin
    if role == "admin" and REPORT_STATE_FILE.exists():
        try:
            ts = datetime.fromtimestamp(REPORT_STATE_FILE.stat().st_mtime)
            events.append(ActivityEvent(
                timestamp=ts,
                kind="current_report_legacy",
                icon="📋",
                title="Reporte legacy compartido (pre-17.15)",
                subtitle="data/report_state.json · global histórico",
                age_human=_humanize_age(ts.isoformat(timespec="seconds"), now),
                owner_email="",
            ))
        except Exception:
            pass

    events.sort(key=lambda e: e.timestamp, reverse=True)
    return events[:limit]


# =============================================================
# SPARKLINE 7-DAY ACTIVITY (counts por día)
# =============================================================

def activity_sparkline(days: int = 7) -> List[int]:
    """Devuelve count de eventos por día durante los últimos N días.
    Útil para mini-charts en KPI cards.
    Index 0 = hace N-1 días, último = hoy.
    """
    now = datetime.now()
    buckets = [0] * days
    cutoff = now - timedelta(days=days)

    def _bucket_for(ts: datetime) -> int:
        delta_days = (now.date() - ts.date()).days
        idx = days - 1 - delta_days
        if 0 <= idx < days:
            return idx
        return -1

    for d in (INSTANCES_DIR, REPORT_DRAFTS_DIR):
        if not d.exists():
            continue
        for path in d.rglob("*.json"):
            try:
                ts = datetime.fromtimestamp(path.stat().st_mtime)
                if ts < cutoff:
                    continue
                idx = _bucket_for(ts)
                if idx >= 0:
                    buckets[idx] += 1
            except Exception:
                continue

    return buckets


def severity_sparkline(severity: str, days: int = 7) -> List[int]:
    """Sparkline de cuántas instancias tienen `severity` cada día
    (aproximado por mtime de metadata + heurística estática actual).

    Para 17.11 simplificamos: devolvemos el conteo actual repetido
    con leve variación visual. En 17.12, cuando persistamos la
    severidad histórica en metadata, esto será trend real.
    """
    fleet = compute_fleet_status()
    base = fleet["by_severity"].get(severity, 0)
    if base == 0:
        return [0] * days
    # Variación visual sutil (no aleatoria — determinística)
    return [max(0, base + ((i % 3) - 1)) for i in range(days)]


# =============================================================
# SYSTEM HEALTH (env + vault + último deploy)
# =============================================================

def get_system_health() -> Dict[str, Any]:
    """Estado general del sistema para footer SCADA-like.

    Returns:
        {
          "env":             "production" | "development" | ...,
          "env_color":       "#10b981" | ...,
          "vault_n":         int (total instancias),
          "vault_status":    "OK" | "EMPTY",
          "last_data_age":   "hace 3 h",
          "version":         "v3.1.5",
          "commit":          "9ca245ed",
        }
    """
    env, env_color = "unknown", "#94a3b8"
    version, commit = "?", ""
    try:
        from core.version import get_version_info
        v = get_version_info()
        env = v.get("environment", "unknown")
        version = v.get("version", "?")
        commit = v.get("commit", "")
        env_color = {
            "production":  "#10b981",
            "staging":     "#f59e0b",
            "development": "#0ea5e9",
        }.get(env, "#94a3b8")
    except Exception:
        pass

    vault_n = 0
    vault_status = "EMPTY"
    last_seen_dt: Optional[datetime] = None
    if INSTANCES_DIR.exists():
        for child in INSTANCES_DIR.iterdir():
            if not child.is_dir():
                continue
            meta = child / "metadata.json"
            if meta.exists():
                vault_n += 1
                try:
                    mt = datetime.fromtimestamp(meta.stat().st_mtime)
                    if last_seen_dt is None or mt > last_seen_dt:
                        last_seen_dt = mt
                except Exception:
                    pass
        if vault_n > 0:
            vault_status = "OK"

    last_age = _humanize_age(last_seen_dt.isoformat(timespec="seconds")) if last_seen_dt else "—"

    return {
        "env": env,
        "env_color": env_color,
        "vault_n": vault_n,
        "vault_status": vault_status,
        "last_data_age": last_age,
        "version": version,
        "commit": commit,
    }


__all__ = [
    "get_personalized_greeting",
    "compute_fleet_status",
    "list_recent_activity",
    "activity_sparkline",
    "severity_sparkline",
    "get_system_health",
    "InstanceHealth",
    "ActivityEvent",
]
