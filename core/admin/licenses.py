"""
core.admin.licenses — Administración · Licencias (campo).

Consola de licencias de las apps de campo (Watermelon Modal / Torsional /
Balanceo / Field). Fuente de verdad: tablas Supabase `licenses` + `activations`
(server/licensing/schema.sql + 2026_09_licenses_console.sql). El emisor firma
tokens Ed25519 en la Edge Function `activate`; aquí se ADMINISTRA el ciclo de
vida comercial:

    · Crear licencia (genera clave WM-XXXX-XXXX-XXXX).
    · Ver dónde vive cada licencia: PC, IP, ubicación, última conexión, VM.
    · Renovar (extiende vigencia).
    · Suspender por falta de pago  → el cliente ve "Licencia no renovada por
      falta de pago" al próximo arranque online (edge `activate` → payment_due).
    · Reactivar / Revocar licencia completa.
    · Revocar / reactivar / liberar UNA máquina (cupo).

Escribe con el cliente de service-role (core.live_readings), que salta RLS.
El hub (pages/20_Administracion.py) ya autenticó y validó role=admin.

NOTA: el antiguo módulo "Licencias Planta" (tabla `revoked_licenses` + edge
`license-check`) administraba el producto legacy `planta/` (Watermelon Planta
Edition, JWT RS256). Ese producto quedó fuera de uso; esta consola lo reemplaza
y apunta al sistema de campo real.
"""
from __future__ import annotations

import html as _html
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import streamlit as st

from core.live_readings import _get_supabase_client as get_supabase_client
from core.ui_industrial import dot, html_table

# Paquetes comerciales → features embebidas en el token (informativo; el modelo
# es de PAQUETE: una activación cubre todos los módulos de campo del PC).
PLANS: Dict[str, List[str]] = {
    "Paquete completo (todos los módulos)":
        ["oma", "ema", "report", "torsional", "balance", "rotordynamics"],
    "Modal (OMA/EMA)": ["oma", "ema", "report"],
    "Torsional": ["torsional", "report"],
    "Balanceo": ["balance", "report"],
    "Rotordynamics (Field)": ["rotordynamics", "report"],
}

# Código interno de feature → nombre de módulo para mostrar al cliente.
# 'report' es interno (motor de reportes), no es un módulo vendible → no se lista.
_FEATURE_LABELS: Dict[str, str] = {
    "rotordynamics": "Rotordynamics",
    "oma": "Modal", "ema": "Modal",
    "torsional": "Torsional",
    "balance": "Balanceo",
}
# Orden canónico de los módulos de campo para presentación.
_MODULE_ORDER = ["Rotordynamics", "Modal", "Torsional", "Balanceo"]

# Metadata por módulo para anuncios de actualización (prefijo de tag del release,
# nombre del instalador, y código(s) de feature que identifican a quién le aplica).
_MODULE_RELEASE: Dict[str, Dict[str, Any]] = {
    "Rotordynamics": {"tag": "field-v", "app": "Watermelon Rotordynamics",
                      "exe": "WatermelonRotordynamics-Setup.exe", "feat": {"rotordynamics"}},
    "Modal": {"tag": "modal-v", "app": "Watermelon Modal",
              "exe": "WatermelonModal-Setup.exe", "feat": {"oma", "ema"}},
    "Torsional": {"tag": "torsional-v", "app": "Watermelon Torsional",
                  "exe": "WatermelonTorsional-Setup.exe", "feat": {"torsional"}},
    "Balanceo": {"tag": "balance-v", "app": "Watermelon Balancing",
                 "exe": "WatermelonBalancing-Setup.exe", "feat": {"balance"}},
    # Módulo NUEVO: sin "feat" → el aviso aplica a TODAS las licencias activas
    # (modelo paquete; Efficiency va incluido para todos, igual que el gate
    # fail-open). No entra a _MODULE_ORDER para no romper "Paquete completo".
    "Efficiency": {"tag": "efficiency-v", "app": "Watermelon Efficiency",
                   "exe": "WatermelonEfficiency-Setup.exe"},
}
# Módulos que se pueden anunciar por correo (incluye el nuevo Efficiency).
_ANNOUNCE_MODULES = _MODULE_ORDER + ["Efficiency"]
_RELEASE_REPO = "Ewdeshernandez/watermelon-system"

# Resumen CLIENTE (bien redactado, sin jerga interna) de lo que trae la versión
# vigente de cada módulo. Al sacar una versión nueva, se actualiza aquí.
_UPDATE_HIGHLIGHTS: Dict[str, str] = {
    "Rotordynamics": (
        "• Monitoreo en línea más confiable: las lecturas en vivo llegan a "
        "Watermelon System de forma continua.\n"
        "• Verificación automática del activo antes de iniciar el monitoreo, "
        "para que los datos siempre queden en el equipo correcto.\n"
        "• Mejoras de estabilidad."),
    "Modal": (
        "• Mejoras de estabilidad y rendimiento.\n"
        "• Validación de licencia más robusta."),
    "Torsional": (
        "• Mejoras de estabilidad y rendimiento.\n"
        "• Validación de licencia más robusta."),
    "Balanceo": (
        "• Mejoras de estabilidad y rendimiento.\n"
        "• Validación de licencia más robusta."),
    "Efficiency": (
        "• NUEVO módulo de Eficiencia de máquinas rotatorias, ya incluido en tu paquete.\n"
        "• Potencia mecánica real en el eje (torque + RPM) vs. potencia eléctrica y de diseño.\n"
        "• Eficiencia del motor y operativa con semáforo por norma (IEC 60034-2 / ISO 20816).\n"
        "• Eficiencia de proceso por tipo: ventiladores (ISO 5801), bombas (ISO 9906), "
        "compresores (ASME PTC 10) y turbinas hidráulicas (IEC 60041).\n"
        "• Reporte técnico completo y análisis en la web."),
    "Todos": (
        "• Nueva versión disponible con mejoras de estabilidad y seguridad de "
        "licencia en tus módulos de Watermelon System."),
}


_LOGO_URL = (f"https://raw.githubusercontent.com/{_RELEASE_REPO}/main/"
             "assets/watermelon_logo.png")
# Ruido que NO debe aparecer en el changelog del cliente.
_CHANGELOG_SKIP = ("merge", "bump", "wip", "typo", "lint", "ci(", "chore(release")
# Prefijos de conventional commits a limpiar.
_CC_PREFIX = re.compile(
    r"^(feat|fix|perf|refactor|style|docs|test|build|chore)(\([^)]*\))?[:!]\s*", re.I)


@st.cache_data(ttl=600, show_spinner=False)
def _latest_release_info(module: str) -> Dict[str, Any]:
    """Info del último release del módulo (en vivo, cacheado 10 min):
    {version, tag, prev_tag, date, html_url}. {} si no se pudo leer."""
    import json as _json
    import urllib.request as _ur
    pre = _MODULE_RELEASE.get(module, {}).get("tag")
    if not pre:
        return {}
    try:
        url = f"https://api.github.com/repos/{_RELEASE_REPO}/releases?per_page=100"
        req = _ur.Request(url, headers={"User-Agent": "WM-Console",
                                        "Accept": "application/vnd.github+json"})
        with _ur.urlopen(req, timeout=6.0) as r:
            rels = _json.load(r)
    except Exception:  # noqa: BLE001
        return {}
    ranked = []
    for rel in rels or []:
        tag = rel.get("tag_name", "") or ""
        if not tag.startswith(pre) or rel.get("draft"):
            continue
        nums = tuple(int(x) for x in re.findall(r"\d+", tag[len(pre):])[:3])
        if nums:
            ranked.append((nums, tag, rel))
    if not ranked:
        return {}
    ranked.sort(key=lambda x: x[0], reverse=True)
    _, tag, rel = ranked[0]
    prev_tag = ranked[1][1] if len(ranked) > 1 else ""
    return {"version": tag[len(pre):], "tag": tag, "prev_tag": prev_tag,
            "date": (rel.get("published_at") or "")[:10],
            "html_url": rel.get("html_url") or ""}


@st.cache_data(ttl=600, show_spinner=False)
def _release_changes(prev_tag: str, tag: str) -> List[str]:
    """Bullets de cambios REALES entre dos tags (GitHub compare API), limpiados a
    lenguaje cliente: quita prefijos de commit, ruido y recorta. [] si falla."""
    if not tag or not prev_tag:
        return []
    import json as _json
    import urllib.request as _ur
    try:
        url = (f"https://api.github.com/repos/{_RELEASE_REPO}/compare/"
               f"{prev_tag}...{tag}")
        req = _ur.Request(url, headers={"User-Agent": "WM-Console",
                                        "Accept": "application/vnd.github+json"})
        with _ur.urlopen(req, timeout=7.0) as r:
            data = _json.load(r)
    except Exception:  # noqa: BLE001
        return []
    out: List[str] = []
    seen: set = set()
    for c in data.get("commits", []) or []:
        msg = ((c.get("commit") or {}).get("message") or "").splitlines()[0].strip()
        low = msg.lower()
        if not msg or any(k in low for k in _CHANGELOG_SKIP):
            continue
        clean = _CC_PREFIX.sub("", msg).strip()
        if not clean:
            continue
        clean = clean[0].upper() + clean[1:]
        if clean.lower() in seen:
            continue
        seen.add(clean.lower())
        out.append(clean)
    return out[:6]


def _module_entitlements(lic: Dict[str, Any]) -> List[tuple]:
    """Lista [(módulo, incluido_bool)] en orden canónico, derivada de features.
    Modal se incluye si está 'oma' o 'ema'. Sirve para mostrar al cliente qué
    tiene aprobado y qué no."""
    feats = {str(f).strip().lower() for f in (lic.get("features") or [])}
    return [
        ("Rotordynamics", "rotordynamics" in feats),
        ("Modal", bool(feats & {"oma", "ema"})),
        ("Torsional", "torsional" in feats),
        ("Balanceo", "balance" in feats),
    ]


def _plan_label(lic: Dict[str, Any]) -> str:
    """Nombre legible del paquete para el cliente (correo + tarjeta). NUNCA
    devuelve los códigos crudos (oma/ema/report). Deriva los módulos desde
    'features'; si están todos → 'Paquete completo'. Cae a 'plan' guardado o '—'."""
    feats = lic.get("features") or []
    mods: List[str] = []
    for f in feats:
        name = _FEATURE_LABELS.get(str(f).strip().lower())
        if name and name not in mods:
            mods.append(name)
    mods.sort(key=lambda m: _MODULE_ORDER.index(m) if m in _MODULE_ORDER else 99)
    if set(mods) >= set(_MODULE_ORDER):
        return "Paquete completo — " + " · ".join(_MODULE_ORDER)
    if mods:
        return " · ".join(mods)
    # Sin features utilizables: usa el plan guardado si existe.
    _p = (lic.get("plan") or "").strip()
    return _p or "—"


# =====================================================================
# Helpers
# =====================================================================
def _gen_key() -> str:
    """Clave única formato WM-XXXX-XXXX-XXXX (hex mayúsculas)."""
    return "WM-" + "-".join(secrets.token_hex(2).upper() for _ in range(3))


def _parse_dt(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        txt = str(s).replace("Z", "+00:00")
        dt = datetime.fromisoformat(txt)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def _rel(s: Any) -> str:
    """'hace 3 min' / 'hace 2 d' a partir de un timestamp ISO."""
    dt = _parse_dt(s)
    if not dt:
        return "—"
    delta = datetime.now(timezone.utc) - dt
    sec = int(delta.total_seconds())
    if sec < 0:
        return "ahora"
    if sec < 90:
        return "hace segundos"
    if sec < 3600:
        return f"hace {sec // 60} min"
    if sec < 86400:
        return f"hace {sec // 3600} h"
    if sec < 30 * 86400:
        return f"hace {sec // 86400} d"
    return dt.strftime("%Y-%m-%d")


def _days_left(expires_at: Any) -> int | None:
    dt = _parse_dt(expires_at)
    if not dt:
        return None
    return (dt - datetime.now(timezone.utc)).days


def _lic_state(lic: Dict[str, Any]) -> tuple[str, str, str]:
    """(severidad_dot, etiqueta, color_hex) del estado de una licencia."""
    status = str(lic.get("status", "")).lower()
    dl = _days_left(lic.get("expires_at"))
    if status == "suspended":
        return "dang", "Suspendida · falta de pago", "#dc3545"
    if status == "revoked":
        return "off", "Revocada", "#8090a6"
    if dl is not None and dl < 0:
        return "warn", "Vencida", "#e8890c"
    if dl is not None and dl < 30:
        return "warn", f"Activa · vence en {dl} d", "#e8890c"
    return "ok", "Activa", "#1f9d55"


@st.cache_data(ttl=15)
def _load() -> tuple[list, list, list]:
    """(licenses, activations, events) desde Supabase."""
    sb = get_supabase_client()
    if sb is None:
        return [], [], []
    try:
        lic = sb.table("licenses").select("*").order("created_at", desc=True).execute()
        act = sb.table("activations").select("*").execute()
        evt = []
        try:
            evt = list((sb.table("license_events").select("*")
                        .order("created_at", desc=True).limit(1000).execute()).data or [])
        except Exception:  # noqa: BLE001 — tabla puede no existir aún
            evt = []
        return list(lic.data or []), list(act.data or []), evt
    except Exception as e:  # noqa: BLE001
        st.warning(f"No se pudo leer licencias/activaciones: {e}")
        return [], [], []


def _fmt_dt(s: Any) -> str:
    dt = _parse_dt(s)
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "—"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _send_license_email(to: str, customer: str, key: str, plan: str,
                        seats: int, exp_iso: str,
                        entitlements: "List[tuple] | None" = None) -> Dict[str, Any]:
    """Envía la clave de licencia al correo del cliente (backend de core.email_sender).
    `entitlements` = [(módulo, incluido_bool)] para mostrar qué tiene aprobado y
    qué no. Devuelve {ok, ...}. No lanza."""
    try:
        from core.email_sender import send_email
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"email_sender no disponible: {e}"}
    exp_dt = _parse_dt(exp_iso)
    exp_txt = exp_dt.strftime("%Y-%m-%d") if exp_dt else "—"
    nombre = customer.strip() or to
    seat_txt = "1 equipo" if seats == 1 else f"{seats} equipos"
    subject = "Tu licencia de Watermelon System"

    # Matriz de módulos aprobados / no aprobados (texto + HTML)
    _ent = entitlements or []
    _mod_text = ""
    _mod_html = ""
    if _ent:
        _mod_text = "\nMódulos de tu licencia:\n" + "\n".join(
            f"  {'[incluido]    ' if inc else '[no incluido] '} {name}"
            for name, inc in _ent) + "\n"
        _rows = "".join(
            f'<tr>'
            f'<td style="padding:5px 10px;border-bottom:1px solid #eef2f8;">{name}</td>'
            f'<td style="padding:5px 10px;border-bottom:1px solid #eef2f8;text-align:right;'
            f'font-weight:700;color:{"#1f9d55" if inc else "#9aa6ba"};">'
            f'<span style="font-size:15px;">●</span> '
            f'{"Incluido" if inc else "No incluido"}</td></tr>'
            for name, inc in _ent)
        _mod_html = (
            '<div style="font:600 11px \'IBM Plex Mono\',monospace;letter-spacing:.1em;'
            'color:#5b6b86;text-transform:uppercase;margin:16px 0 6px;">Módulos de tu licencia</div>'
            '<table style="width:100%;max-width:360px;font-size:14px;color:#3a4c66;'
            'border-collapse:collapse;border:1px solid #e2e8f2;border-radius:8px;">'
            + _rows + '</table>')

    body_text = (
        f"Hola {nombre},\n\n"
        f"Se generó tu licencia de Watermelon System.\n\n"
        f"Clave de licencia: {key}\n"
        f"Paquete: {plan}\n"
        f"Equipos permitidos: {seat_txt}\n"
        f"Vigencia hasta: {exp_txt}\n"
        f"{_mod_text}\n"
        f"Cómo activar:\n"
        f"1) Instala/abre el módulo de Watermelon System en el equipo.\n"
        f"2) Cuando pida la clave, ingresa: {key}\n"
        f"3) Una sola activación habilita los módulos de tu paquete en ese equipo.\n\n"
        f"Soporte: watermelonsystem.app\n"
        f"— SIGA GROUP SAS"
    )
    body_html = f"""
    <div style="font-family:'IBM Plex Sans',Arial,sans-serif;color:#0b1f3a;max-width:560px;">
      <h2 style="margin:0 0 6px;color:#12305e;">Watermelon System</h2>
      <p>Hola <b>{nombre}</b>, se generó tu licencia.</p>
      <div style="background:#f3f7fc;border:1px solid #d7e3f2;border-left:4px solid #12305e;
                  border-radius:10px;padding:14px 16px;margin:14px 0;">
        <div style="font:600 11px 'IBM Plex Mono',monospace;letter-spacing:.1em;color:#5b6b86;
                    text-transform:uppercase;">Clave de licencia</div>
        <div style="font:800 22px 'IBM Plex Mono',monospace;letter-spacing:2px;color:#12305e;">{key}</div>
      </div>
      <table style="font-size:14px;color:#3a4c66;border-collapse:collapse;">
        <tr><td style="padding:2px 12px 2px 0;color:#8090a6;">Paquete</td><td><b>{plan}</b></td></tr>
        <tr><td style="padding:2px 12px 2px 0;color:#8090a6;">Equipos</td><td>{seat_txt}</td></tr>
        <tr><td style="padding:2px 12px 2px 0;color:#8090a6;">Vigencia hasta</td><td>{exp_txt}</td></tr>
      </table>
      {_mod_html}
      <p style="margin-top:16px;"><b>Cómo activar:</b></p>
      <ol style="color:#3a4c66;font-size:14px;line-height:1.7;">
        <li>Instala/abre el módulo de Watermelon System en el equipo.</li>
        <li>Cuando pida la clave, ingresa: <code style="color:#12305e;">{key}</code></li>
        <li>Una sola activación habilita los módulos de tu paquete en ese equipo.</li>
      </ol>
      <p style="color:#8090a6;font-size:12px;margin-top:18px;">Soporte: watermelonsystem.app · SIGA GROUP SAS</p>
    </div>"""
    try:
        return send_email(to, subject, body_text, body_html=body_html)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def _send_status_email(kind: str, to: str, customer: str, key: str,
                       reason: str = "", exp_txt: str = "") -> Dict[str, Any]:
    """Avisa al cliente de un cambio de estado de su licencia.
    kind: 'suspended' | 'reactivated' | 'renewed'. No lanza."""
    try:
        from core.email_sender import send_email
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"email_sender no disponible: {e}"}
    nombre = (customer or "").strip() or to
    if kind == "suspended":
        subject = "Tu licencia de Watermelon System fue suspendida"
        head = "Licencia suspendida"
        color = "#dc3545"
        lead = (f"Hola {nombre}, tu licencia de Watermelon System quedó "
                f"<b>suspendida</b>.")
        detail = (reason or "Licencia no renovada por falta de pago")
        steps = ("Al abrir cualquier módulo con internet verás el aviso y no podrá "
                 "iniciar. Para restablecerla, regulariza el pago y contáctanos: "
                 "en cuanto reactivemos, tus equipos vuelven a funcionar solos.")
    elif kind == "renewed":
        subject = "Tu licencia de Watermelon System fue renovada"
        head = "Licencia renovada"
        color = "#1f9d55"
        lead = f"Hola {nombre}, tu licencia de Watermelon System fue <b>renovada</b>."
        detail = f"Nueva vigencia hasta: {exp_txt}" if exp_txt else "Vigencia extendida."
        steps = ("No tienes que hacer nada: al abrir cualquier módulo con internet, "
                 "la nueva vigencia se aplica sola.")
    else:  # reactivated
        subject = "Tu licencia de Watermelon System fue reactivada"
        head = "Licencia reactivada"
        color = "#1f9d55"
        lead = f"Hola {nombre}, tu licencia de Watermelon System fue <b>reactivada</b>."
        detail = "Ya puedes volver a usar tus módulos."
        steps = ("Abre cualquier módulo con internet y volverá a iniciar normalmente. "
                 "No necesitas re-ingresar la clave.")
    body_text = (f"{lead}\n\n{detail}\n\n{steps}\n\n"
                 f"Licencia: {key}\nSoporte: watermelonsystem.app\n— SIGA GROUP SAS")
    body_html = f"""
    <div style="font-family:'IBM Plex Sans',Arial,sans-serif;color:#0b1f3a;max-width:560px;">
      <h2 style="margin:0 0 6px;color:#12305e;">Watermelon System</h2>
      <div style="background:{color}14;border:0.5px solid {color}55;border-left:4px solid {color};
                  border-radius:0 10px 10px 0;padding:12px 16px;margin:12px 0;">
        <div style="font:800 13px 'IBM Plex Sans';color:{color};text-transform:uppercase;
                    letter-spacing:.05em;">{head}</div>
      </div>
      <p style="font-size:14px;color:#3a4c66;">{lead}</p>
      <p style="font-size:14px;color:#3a4c66;"><b>{detail}</b></p>
      <p style="font-size:14px;color:#3a4c66;line-height:1.7;">{steps}</p>
      <p style="color:#8090a6;font-size:12px;margin-top:16px;border-top:0.5px solid #e2e8f2;
                padding-top:12px;">Licencia <code style="color:#274b7d;">{key}</code> ·
        Soporte: watermelonsystem.app · SIGA GROUP SAS</p>
    </div>"""
    try:
        return send_email(to, subject, body_text, body_html=body_html)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def _send_update_email(to: str, customer: str, module: str, app_name: str,
                       version: str, whats_new: str,
                       dl_url: str = "", date: str = "",
                       release_url: str = "") -> Dict[str, Any]:
    """Aviso HERMOSO de nueva versión disponible (envío manual desde la consola).
    No lanza."""
    try:
        from core.email_sender import send_email
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"email_sender no disponible: {e}"}
    nombre = (customer or "").strip() or to
    _ver = (version or "").strip()
    subject = f"Nueva versión de {app_name}" + (f" ({_ver}) disponible" if _ver else " disponible")
    _wn = (whats_new or "").strip()
    _date_txt = f" (publicada el {date})" if date else ""
    body_text = (
        f"Hola {nombre},\n\n"
        f"Ya está disponible una nueva versión de {app_name}"
        f"{(' ' + _ver) if _ver else ''}{_date_txt}.\n\n"
        + (f"Qué trae:\n{_wn}\n\n" if _wn else "")
        + "Cómo actualizar (automático):\n"
        f"1) Abre {app_name} en tu equipo.\n"
        f"2) El actualizador te ofrecerá la nueva versión.\n"
        f"3) Acepta y listo — se actualiza sola.\n"
        "\nSoporte: watermelonsystem.app\n— SIGA GROUP SAS"
    )
    _wn_html = ""
    if _wn:
        _items = [ln.strip().lstrip("•").strip()
                  for ln in _wn.splitlines() if ln.strip()]
        _lis = "".join(
            f'<li style="margin:4px 0;">{_html.escape(it)}</li>' for it in _items)
        _wn_html = (
            '<div style="font:600 11px \'IBM Plex Mono\',monospace;letter-spacing:.1em;'
            'color:#5b6b86;text-transform:uppercase;margin:18px 0 4px;">Qué trae</div>'
            f'<ul style="font-size:14px;color:#3a4c66;line-height:1.6;margin:0;'
            f'padding-left:20px;">{_lis}</ul>')
    _ver_badge = (
        f'<span style="background:#12305e;color:#fff;border-radius:999px;padding:3px 12px;'
        f'font:800 12px \'IBM Plex Mono\',monospace;letter-spacing:1px;">{_html.escape(_ver)}</span>'
        ) if _ver else ""
    _date_html = (f'<span style="color:#8090a6;font-size:13px;margin-left:8px;">'
                  f'publicada el {_html.escape(date)}</span>') if date else ""
    body_html = f"""
    <div style="font-family:'IBM Plex Sans',Arial,sans-serif;color:#0b1f3a;max-width:560px;">
      <div style="background-color:#12305e;
                  background-image:linear-gradient(135deg,#0d2b5e 0%,#1b4a86 55%,#2f74bd 100%);
                  border-radius:14px;padding:28px 24px;">
        <div style="font:700 12px 'IBM Plex Mono',monospace;letter-spacing:.18em;
                    color:#ffd79a;text-transform:uppercase;">Watermelon System</div>
        <div style="font:800 30px 'IBM Plex Sans',Arial,sans-serif;color:#ffffff;
                    margin-top:8px;line-height:1.08;">Nueva versión disponible</div>
        <div style="height:4px;width:68px;background-color:#e8890c;border-radius:999px;
                    margin-top:14px;"></div>
      </div>
      <p style="margin:16px 0 4px;">Hola <b>{nombre}</b>,</p>
      <p style="font-size:15px;color:#1b2b45;">Ya puedes actualizar
        <b>{_html.escape(app_name)}</b> &nbsp;{_ver_badge}{_date_html}</p>
      {_wn_html}
      <div style="background:#f3f7fc;border:1px solid #d7e3f2;border-left:4px solid #1f9d55;
                  border-radius:10px;padding:14px 16px;margin:16px 0;">
        <div style="font:800 13px 'IBM Plex Sans';color:#12305e;margin-bottom:6px;">
          Cómo actualizar — automático</div>
        <ol style="color:#3a4c66;font-size:14px;line-height:1.7;margin:0;padding-left:18px;">
          <li>Abre <b>{_html.escape(app_name)}</b> en tu equipo.</li>
          <li>El actualizador te ofrecerá la nueva versión.</li>
          <li>Acepta y listo — se actualiza sola.</li>
        </ol>
      </div>
      <p style="font-size:13px;color:#8090a6;margin-top:4px;">No necesitas descargar
        nada ni ingresar ninguna clave — la actualización es automática dentro de la app.</p>
      <p style="color:#8090a6;font-size:12px;margin-top:18px;border-top:0.5px solid #e2e8f2;
                padding-top:12px;">Soporte: watermelonsystem.app · SIGA GROUP SAS</p>
    </div>"""
    try:
        return send_email(to, subject, body_text, body_html=body_html)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def _update_recipients(licenses: List[Dict[str, Any]], module: str) -> List[tuple]:
    """[(account, customer)] únicos de licencias ACTIVAS a las que aplica el módulo.
    module == 'Todos' → todas las activas. Si no → las que incluyen el feature del
    módulo (o, fail-open, las que no tengan features declaradas)."""
    need = _MODULE_RELEASE.get(module, {}).get("feat")
    seen: set = set()
    out: List[tuple] = []
    for l in licenses:
        if str(l.get("status") or "").lower() != "active":
            continue
        acc = (l.get("account") or "").strip()
        if not acc or "@" not in acc or acc.lower() in seen:
            continue
        if module != "Todos" and need is not None:
            feats = {str(f).strip().lower() for f in (l.get("features") or [])}
            if feats and not (feats & need):
                continue  # tiene features y NO incluye este módulo → no le aplica
        seen.add(acc.lower())
        out.append((acc, (l.get("customer") or "").strip()))
    return out


# =====================================================================
# Render
# =====================================================================
def render() -> None:
    _user_email = st.session_state.get("auth_email", "")
    if not _user_email.endswith("@sigasas.com"):
        st.error("Acceso denegado. Sección solo para administradores de SIGA GROUP.")
        return

    sb = get_supabase_client()
    if sb is None:
        st.error("Supabase no configurado (falta SUPABASE_URL / SUPABASE_SERVICE_KEY).")
        return

    licenses, activations, events = _load()

    # --- Índice de activaciones por licencia ---
    by_lic: Dict[str, List[Dict[str, Any]]] = {}
    for a in activations:
        by_lic.setdefault(a.get("license_id"), []).append(a)

    # --- Índice de eventos (historial de conexiones) por licencia ---
    ev_lic: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        ev_lic.setdefault(e.get("license_id"), []).append(e)

    # --- KPIs ---
    n_total = len(licenses)
    n_active = sum(1 for l in licenses if _lic_state(l)[0] == "ok")
    n_suspended = sum(1 for l in licenses if str(l.get("status")).lower() == "suspended")
    n_expsoon = sum(1 for l in licenses if _lic_state(l)[1].startswith("Activa · vence"))
    n_machines = sum(1 for a in activations if not a.get("revoked"))
    _kpi_row = "".join(
        f'<div class="wi-kpi"><div class="n">{dot(sev)} {val}</div>'
        f'<div class="l">{lbl}</div></div>'
        for sev, val, lbl in [
            ("info", n_total, "Licencias"),
            ("ok", n_active, "Activas"),
            ("dang", n_suspended, "Suspendidas"),
            ("warn", n_expsoon, "Vencen pronto"),
            ("ok", n_machines, "Máquinas activas"),
        ])
    st.markdown(f'<div class="wi-kpis">{_kpi_row}</div>', unsafe_allow_html=True)
    st.markdown('<div class="wi-label">Licencias de campo · Modal · Torsional · '
                'Balanceo · Field</div>', unsafe_allow_html=True)

    # --- Crear licencia ---
    with st.expander("＋  Crear licencia nueva", expanded=(n_total == 0)):
        _lc = st.session_state.get("_lic_created")
        if _lc:
            st.success(f"Licencia creada. Clave del cliente: **{_lc['key']}**"
                       f"{_lc.get('mail', '')}")
        with st.form("create_license", clear_on_submit=True):
            _c1, _c2 = st.columns(2)
            with _c1:
                _customer = st.text_input("Cliente / empresa",
                                          placeholder="ej: Termoeléctrica Norte SAS")
                _account = st.text_input("Cuenta (email del cliente)",
                                         placeholder="ej: ingenieria@termonorte.com")
                _plan = st.selectbox("Paquete", list(PLANS.keys()))
            with _c2:
                _seats = st.number_input("Máquinas (seats)", min_value=1, max_value=50,
                                         value=1, step=1)
                _months = st.number_input("Vigencia (meses)", min_value=1, max_value=120,
                                          value=12, step=1)
                _notes = st.text_input("Notas internas (opcional)",
                                       placeholder="ej: OC-2026-118, contacto Juan")
            _send_mail = st.checkbox(
                "Enviar la clave por correo al cliente", value=True,
                help="Al crear, se envía un correo con la clave e instrucciones de activación.")
            _submit = st.form_submit_button("CREAR LICENCIA", type="primary",
                                            use_container_width=True)
            if _submit:
                if not _account.strip() or "@" not in _account:
                    st.error("La cuenta (email del cliente) es obligatoria y debe ser un correo válido.")
                else:
                    key = _gen_key()
                    exp = (datetime.now(timezone.utc)
                           + timedelta(days=int(_months) * 30)).replace(
                        hour=23, minute=59, second=59, microsecond=0).isoformat()
                    try:
                        sb.table("licenses").insert({
                            "key": key,
                            "account": _account.strip(),
                            "customer": _customer.strip() or None,
                            "seats": int(_seats),
                            "features": PLANS[_plan],
                            "plan": _plan,
                            "notes": _notes.strip() or None,
                            "expires_at": exp,
                            "status": "active",
                            "updated_at": _now_iso(),
                        }).execute()
                        st.cache_data.clear()
                        _mail = ""
                        if _send_mail:
                            _r = _send_license_email(
                                _account.strip(), _customer.strip(), key, _plan,
                                int(_seats), exp,
                                entitlements=_module_entitlements({"features": PLANS[_plan]}))
                            _mail = ("  ·  ✉ correo enviado a " + _account.strip()) if _r.get("ok") \
                                else ("  ·  ⚠ no se pudo enviar el correo: "
                                      + str(_r.get("error", ""))[:90])
                        st.session_state["_lic_created"] = {"key": key, "mail": _mail}
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Error al crear: {e}")

    st.divider()

    if not licenses:
        st.info("No hay licencias todavía. Crea la primera arriba.")
        return

    # --- Anunciar nueva versión (correo manual a clientes activos) ---
    with st.expander("Anunciar nueva versión a clientes"):
        _u1, _u2 = st.columns([1.2, 1])
        with _u1:
            _mod = st.selectbox("Módulo", ["Todos"] + _ANNOUNCE_MODULES, key="upd_mod")
        # Auto: última versión + fecha + url (en vivo). "Qué trae" = resumen
        # redactado por módulo (preciso y profesional). En monorepo el compare
        # entre tags mezcla commits de TODOS los módulos, así que solo se usa
        # como respaldo si no hay resumen. Keys por módulo → re-rellena; editable.
        _info = _latest_release_info(_mod) if _mod != "Todos" else {}
        _auto_ver = _info.get("version", "")
        _date = _info.get("date", "")
        _rel_url = _info.get("html_url", "")
        _auto_wn = _UPDATE_HIGHLIGHTS.get(_mod, "")
        if not _auto_wn and _info:
            _chg = _release_changes(_info.get("prev_tag", ""), _info.get("tag", ""))
            _auto_wn = "\n".join("• " + c for c in _chg)
        with _u2:
            _ver = st.text_input("Versión (auto)", value=_auto_ver,
                                 placeholder="ej: 0.12.9", key=f"upd_ver_{_mod}")
        _wn = st.text_area("Qué trae (auto desde el release · editable)",
                           value=_auto_wn, key=f"upd_wn_{_mod}", height=130)
        if _date:
            st.caption(f"Release {_auto_ver} · publicado {_date}")
        # Link de descarga directo (solo si es un módulo puntual con versión)
        _dl = ""
        if _mod != "Todos" and _ver.strip():
            _meta = _MODULE_RELEASE.get(_mod, {})
            _dl = (f"https://github.com/{_RELEASE_REPO}/releases/download/"
                   f"{_meta.get('tag','')}{_ver.strip()}/{_meta.get('exe','')}")
        _rcp = _update_recipients(licenses, _mod)
        _appname = (_MODULE_RELEASE.get(_mod, {}).get("app")
                    if _mod != "Todos" else "Watermelon System")
        st.caption(f"Destinatarios (clientes activos): **{len(_rcp)}**"
                   + (f"  ·  descarga: {_dl}" if _dl else ""))
        _test = st.text_input("Enviarme solo a mí primero (prueba, opcional)",
                              value=_user_email, key="upd_test")
        _tc1, _tc2 = st.columns(2)
        with _tc1:
            if st.button("Enviar prueba a mí", key="upd_send_test",
                         use_container_width=True, disabled=not _test.strip()):
                _r = _send_update_email(_test.strip(), "Equipo SIGA", _mod or "Todos",
                                        _appname or "Watermelon System", _ver.strip(),
                                        _wn, _dl, date=_date, release_url=_rel_url)
                st.success(f"Prueba enviada a {_test.strip()}.") if _r.get("ok") \
                    else st.error(f"Falló: {_r.get('error','—')}")
        with _tc2:
            _conf = st.checkbox(f"Confirmo enviar a {len(_rcp)} cliente(s)",
                                key="upd_confirm", disabled=not _rcp)
            if st.button("ENVIAR ANUNCIO", key="upd_send_all", type="primary",
                         use_container_width=True, disabled=not (_conf and _rcp)):
                _ok = _fail = 0
                for _acc, _cust in _rcp:
                    _r = _send_update_email(_acc, _cust, _mod or "Todos",
                                            _appname or "Watermelon System",
                                            _ver.strip(), _wn, _dl,
                                            date=_date, release_url=_rel_url)
                    if _r.get("ok"):
                        _ok += 1
                    else:
                        _fail += 1
                st.success(f"Anuncio enviado a {_ok} cliente(s)."
                           + (f"  ·  {_fail} fallaron." if _fail else ""))

    # --- Actividad reciente (todas las licencias) ---
    if events:
        with st.expander(f"Actividad reciente · últimas conexiones ({len(events)})"):
            _cust = {l.get("id"): (l.get("customer") or l.get("account") or "—")
                     for l in licenses}
            rows = [[
                _fmt_dt(e.get("created_at")), _cust.get(e.get("license_id"), "—"),
                e.get("app") or "—", e.get("hostname") or "—",
                e.get("ip") or "—", e.get("ip_geo") or "—",
            ] for e in events[:60]]
            html_table(["Fecha/hora", "Cliente", "Módulo", "PC", "IP", "Ubicación"], rows)

    # --- Lista de licencias ---
    for lic in licenses:
        _render_license_card(sb, lic, by_lic.get(lic.get("id"), []),
                             ev_lic.get(lic.get("id"), []))


def _render_license_card(sb, lic: Dict[str, Any], acts: List[Dict[str, Any]],
                         evts: List[Dict[str, Any]] | None = None) -> None:
    lid = lic.get("id")
    key = lic.get("key") or "—"
    customer = lic.get("customer") or lic.get("account") or "—"
    account = lic.get("account") or "—"
    plan = _plan_label(lic)
    seats = int(lic.get("seats") or 1)
    used = sum(1 for a in acts if not a.get("revoked"))
    dl = _days_left(lic.get("expires_at"))
    exp_dt = _parse_dt(lic.get("expires_at"))
    exp_txt = exp_dt.strftime("%Y-%m-%d") if exp_dt else "—"
    sev, label, color = _lic_state(lic)

    _dl_txt = ("vencida" if (dl is not None and dl < 0)
               else (f"{dl} días" if dl is not None else "—"))

    st.markdown(
        f"""
        <div style="background:#fff;border:1px solid #e2e8f2;border-left:4px solid {color};
                    border-radius:14px;padding:16px 18px 6px;margin-bottom:6px;
                    box-shadow:0 1px 2px rgba(11,31,58,.05),0 8px 22px rgba(11,31,58,.06);
                    font-family:'IBM Plex Sans',sans-serif;">
          <div style="display:flex;align-items:center;gap:11px;flex-wrap:wrap;">
            <span style="font:800 16px 'IBM Plex Sans';color:#0b1f3a;">{_html.escape(customer)}</span>
            <span style="background:{color}1a;color:{color};border:1px solid {color}55;
                         padding:3px 10px;border-radius:999px;font:800 10px 'IBM Plex Sans';
                         letter-spacing:.06em;text-transform:uppercase;">{dot(sev)} {label}</span>
            <span style="flex:1;"></span>
            <code style="font:700 14px 'IBM Plex Mono';color:#274b7d;letter-spacing:1px;">{_html.escape(key)}</code>
          </div>
          <div style="font-size:12px;color:#3a4c66;line-height:1.9;margin-top:6px;">
            {_html.escape(account)} &nbsp;·&nbsp; {_html.escape(str(plan))} &nbsp;·&nbsp;
            <b>{used}/{seats}</b> máquinas &nbsp;·&nbsp; vence {exp_txt} ({_dl_txt})
          </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Módulos aprobados / no aprobados ---
    _chips = "".join(
        f'<span style="display:inline-block;margin:0 6px 6px 0;padding:3px 10px;'
        f'border-radius:999px;font:700 11px \'IBM Plex Sans\';'
        f'background:{"#e8f6ee" if inc else "#f1f4f9"};'
        f'color:{"#1f9d55" if inc else "#9aa6ba"};'
        f'border:1px solid {"#b7e4c7" if inc else "#e2e8f2"};">'
        f'● {name}{"" if inc else " · no incluido"}</span>'
        for name, inc in _module_entitlements(lic))
    st.markdown(
        f'<div style="margin:-2px 0 10px;">{_chips}</div>', unsafe_allow_html=True)

    # --- Máquinas (dónde vive la licencia) ---
    if acts:
        rows = []
        for a in sorted(acts, key=lambda x: str(x.get("last_seen") or ""), reverse=True):
            estado = ("● revocada" if a.get("revoked") else "● activa")
            rows.append([
                a.get("hostname") or "—",
                (a.get("machine_fp") or "")[:12] + "…" if a.get("machine_fp") else "—",
                a.get("ip") or "—",
                a.get("ip_geo") or "—",
                a.get("app") or "—",
                "sí" if a.get("is_vm") else "no",
                _rel(a.get("last_seen")),
                estado,
            ])
        html_table(
            ["PC", "Máquina", "IP", "Ubicación", "Módulo", "VM", "Última conexión", "Estado"],
            rows)
    else:
        st.caption("Sin máquinas activadas todavía con esta clave.")

    # --- Historial de conexiones (Nivel B) ---
    if evts:
        with st.expander(f"Historial de conexiones ({len(evts)})"):
            hrows = [[
                _fmt_dt(e.get("created_at")), e.get("app") or "—",
                e.get("hostname") or "—", e.get("ip") or "—", e.get("ip_geo") or "—",
            ] for e in evts[:40]]
            html_table(["Fecha/hora", "Módulo", "PC", "IP", "Ubicación"], hrows)

    # --- Acciones ---
    _a1, _a2, _a3, _aR, _a4, _a5 = st.columns([1.0, 1.3, 1.0, 1.2, 1.1, 1.3])

    # Renovar
    with _a1:
        with st.popover("Renovar", use_container_width=True):
            _m = st.number_input("Extender (meses)", min_value=1, max_value=120, value=12,
                                 step=1, key=f"renew_m_{lid}")
            if st.button("Aplicar renovación", key=f"renew_btn_{lid}",
                         type="primary", use_container_width=True):
                base = _parse_dt(lic.get("expires_at")) or datetime.now(timezone.utc)
                base = max(base, datetime.now(timezone.utc))  # no renovar hacia el pasado
                new_exp = (base + timedelta(days=int(_m) * 30)).isoformat()
                try:
                    sb.table("licenses").update({
                        "expires_at": new_exp, "status": "active",
                        "suspended_reason": None, "updated_at": _now_iso(),
                    }).eq("id", lid).execute()
                    st.cache_data.clear()
                    _m = ""
                    if lic.get("account"):
                        _r = _send_status_email("renewed", lic.get("account"),
                                                lic.get("customer") or "", key,
                                                exp_txt=new_exp[:10])
                        _m = "  ·  ✉ avisado" if _r.get("ok") else "  ·  ⚠ correo falló"
                    st.success("Renovada y reactivada." + _m)
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

    # Suspender por falta de pago
    with _a2:
        with st.popover("Suspender (no pago)", use_container_width=True):
            _reason = st.text_input(
                "Motivo (lo ve el cliente)",
                value="Licencia no renovada por falta de pago",
                key=f"susp_r_{lid}")
            st.caption("Al próximo arranque online, el cliente será bloqueado con este motivo.")
            _susp_mail = st.checkbox("Avisar al cliente por correo", value=True,
                                     key=f"susp_mail_{lid}")
            if st.button("SUSPENDER", key=f"susp_btn_{lid}", type="primary",
                         use_container_width=True):
                _rz = _reason.strip() or "Licencia no renovada por falta de pago"
                try:
                    sb.table("licenses").update({
                        "status": "suspended", "suspended_reason": _rz,
                        "updated_at": _now_iso(),
                    }).eq("id", lid).execute()
                    st.cache_data.clear()
                    _m = ""
                    if _susp_mail and lic.get("account"):
                        _r = _send_status_email("suspended", lic.get("account"),
                                                lic.get("customer") or "", key, reason=_rz)
                        _m = "  ·  ✉ avisado" if _r.get("ok") else "  ·  ⚠ correo falló"
                    st.success("Licencia suspendida." + _m)
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

    # Reactivar (solo si suspendida/revocada)
    with _a3:
        _blocked = str(lic.get("status")).lower() in ("suspended", "revoked")
        if st.button("Reactivar", key=f"react_{lid}", use_container_width=True,
                     disabled=not _blocked):
            try:
                sb.table("licenses").update({
                    "status": "active", "suspended_reason": None, "updated_at": _now_iso(),
                }).eq("id", lid).execute()
                st.cache_data.clear()
                _m = ""
                if lic.get("account"):
                    _r = _send_status_email("reactivated", lic.get("account"),
                                            lic.get("customer") or "", key)
                    _m = "  ·  ✉ avisado" if _r.get("ok") else "  ·  ⚠ correo falló"
                st.success("Reactivada." + _m)
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Error: {e}")

    # Reenviar la clave por correo (cliente perdió/olvidó el correo original)
    with _aR:
        with st.popover("Reenviar clave", use_container_width=True):
            st.caption("Vuelve a enviar el correo con la clave de activación "
                       "(mismo contenido que al crearla). Puedes corregir el "
                       "destinatario si el correo registrado estaba mal.")
            _to = st.text_input("Enviar a", value=(lic.get("account") or ""),
                                key=f"resend_to_{lid}")
            st.markdown(
                f"<div style='font-size:12px;color:#3a4c66;line-height:1.8'>"
                f"Clave <code style='color:#12305e'>{_html.escape(key)}</code> · "
                f"{_html.escape(str(plan))} · {seats} equipo(s) · vence {exp_txt}</div>",
                unsafe_allow_html=True)
            if st.button("Reenviar correo", key=f"resend_btn_{lid}",
                         type="primary", use_container_width=True):
                _dest = (_to or "").strip()
                if "@" not in _dest:
                    st.error("Correo destino inválido.")
                else:
                    _r = _send_license_email(
                        _dest, lic.get("customer") or "", key,
                        str(plan), seats,
                        lic.get("expires_at") or "",
                        entitlements=_module_entitlements(lic))
                    if _r.get("ok"):
                        # Si corrigieron el correo, deja el nuevo como el de la licencia.
                        if _dest != (lic.get("account") or ""):
                            try:
                                sb.table("licenses").update(
                                    {"account": _dest, "updated_at": _now_iso()}
                                ).eq("id", lid).execute()
                                st.cache_data.clear()
                            except Exception:  # noqa: BLE001
                                pass
                        st.success(f"Clave reenviada a {_dest}.")
                    else:
                        st.error(f"No se pudo enviar: {_r.get('error', '—')}")

    # Revocar / Eliminar licencia completa
    with _a4:
        with st.popover("Revocar / Eliminar", use_container_width=True):
            st.caption("Bloqueo definitivo de TODA la licencia (todas las máquinas). "
                       "Conserva el registro y el historial.")
            _ok = st.checkbox("Confirmo revocar esta licencia", key=f"revk_ok_{lid}")
            if st.button("REVOCAR", key=f"revk_btn_{lid}", type="primary",
                         disabled=not _ok, use_container_width=True):
                try:
                    sb.table("licenses").update({
                        "status": "revoked", "updated_at": _now_iso(),
                    }).eq("id", lid).execute()
                    st.cache_data.clear()
                    st.success("Licencia revocada.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

            st.divider()
            st.markdown('<span style="color:#b02a37;font-weight:800;font-size:13px;">'
                        'Eliminar definitivamente</span>', unsafe_allow_html=True)
            st.caption("Borra la licencia y TODO su historial (activaciones + conexiones). "
                       "No se puede deshacer. Úsalo cuando el cliente ya no la tendrá.")
            _del_ok = st.checkbox("Confirmo eliminar por completo esta licencia",
                                  key=f"del_ok_{lid}")
            if st.button("ELIMINAR DEFINITIVAMENTE", key=f"del_btn_{lid}",
                         disabled=not _del_ok, use_container_width=True):
                try:
                    sb.table("licenses").delete().eq("id", lid).execute()
                    st.cache_data.clear()
                    st.session_state.pop("_lic_created", None)
                    st.success("Licencia eliminada por completo.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Error: {e}")

    # Gestionar equipos (por máquina)
    with _a5:
        if acts:
            with st.popover("Gestionar equipos", use_container_width=True):
                _opts = {
                    f"{(a.get('hostname') or '—')} · {(a.get('machine_fp') or '')[:10]}"
                    f"{' · revocada' if a.get('revoked') else ''}": a
                    for a in acts}
                _pick = st.selectbox("Equipo", list(_opts.keys()), key=f"mach_pick_{lid}")
                a = _opts.get(_pick, {})
                _aid = a.get("id")
                _b1, _b2, _b3 = st.columns(3)
                with _b1:
                    if st.button("Revocar", key=f"mrev_{lid}_{_aid}",
                                 use_container_width=True, disabled=bool(a.get("revoked"))):
                        _mach_update(sb, _aid, {"revoked": True})
                with _b2:
                    if st.button("Reactivar", key=f"mact_{lid}_{_aid}",
                                 use_container_width=True, disabled=not a.get("revoked")):
                        _mach_update(sb, _aid, {"revoked": False})
                with _b3:
                    if st.button("Liberar cupo", key=f"mdel_{lid}_{_aid}",
                                 use_container_width=True):
                        try:
                            sb.table("activations").delete().eq("id", _aid).execute()
                            st.cache_data.clear()
                            st.success("Cupo liberado.")
                            st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Error: {e}")

    st.divider()


def _mach_update(sb, aid, patch: Dict[str, Any]) -> None:
    try:
        patch = {**patch, "updated_at": _now_iso()}
        sb.table("activations").update(patch).eq("id", aid).execute()
        st.cache_data.clear()
        st.success("Máquina actualizada.")
        st.rerun()
    except Exception as e:  # noqa: BLE001
        st.error(f"Error: {e}")


__all__ = ["render"]
