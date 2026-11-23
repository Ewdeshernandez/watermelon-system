"""
pages/00_Machines.py
====================

**Machines** — configuración ÚNICA de máquina. La pantalla es
`native/field_ui/config.html`: el MISMO archivo de 7 pasos que usa el campo
(Machine · Constructor · Sensors & layout · Channel editor · Acquisition ·
Validation · Summary). Aquí solo se hace de HOST: carga/guarda en la nube,
valida en Python y publica. La lógica vive en core.remote_monitoring.config_bridge.

Nube = fuente de verdad (rm_setups = publicada, rm_setup_versions = historial).
Reemplaza Machinery Library + Crear activo wizard (transición: al publicar
puede actualizar también `instances`, que aún leen Live y reportes).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from core.auth import render_user_menu, require_login, require_role

st.set_page_config(page_title="Watermelon System | Machines", layout="wide",
                   initial_sidebar_state="expanded")
require_login()
render_user_menu()
require_role(("admin", "specialist"))

from core.machine_builder_component import machine_config  # noqa: E402
from core.remote_monitoring import config_bridge as cb  # noqa: E402
from core.remote_monitoring import setup_versions as sv  # noqa: E402
from core.remote_monitoring.config import (  # noqa: E402
    _RM_SETUPS_TABLE, _rm_client, _slug, setup_from_dict)
from core.ui_theme import apply_watermelon_page_style, page_header  # noqa: E402

apply_watermelon_page_style()
page_header("Machines",
            subtitle="One configuration per machine — the same 7 steps as the field app. "
                     "Versioned in the cloud and published to online nodes.")

USER = ""
try:
    from core.auth import get_current_user
    _u = get_current_user() or {}
    USER = _u.get("full_name") or _u.get("username") or _u.get("email") or ""
except Exception:  # noqa: BLE001
    pass

SS = st.session_state


# ---------------------------------------------------------------------
# Nube
# ---------------------------------------------------------------------
def _fleet() -> List[Dict[str, Any]]:
    client = _rm_client()
    if client is None:
        return []
    try:
        res = client.table(_RM_SETUPS_TABLE).select("id, name, updated_at, metadata").execute()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for r in res.data or []:
        m = r.get("metadata") or {}
        out.append({"id": r["id"], "name": r.get("name") or r["id"], "meta": m,
                    "mode": m.get("mode") or "spot", "version": m.get("_version")})
    return sorted(out, key=lambda r: (r["mode"] != "online", r["name"].lower()))


def _library(fleet: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"id": r["id"], "label": f"{r['name']}  ·  "
             f"{('v' + str(r['version']) + ' published') if r['version'] else 'unversioned'}"
             f"  ·  {'online' if r['mode'] == 'online' else 'spot'}"} for r in fleet]


def _published(sid: str, fleet: List[Dict[str, Any]]):
    row = next((r for r in fleet if r["id"] == sid), None)
    if not row:
        return None
    m = dict(row["meta"])
    m.pop("_version", None)
    return setup_from_dict(m)


def _push(*msgs: Dict[str, Any]) -> None:
    SS["mc_msgs"] = list(msgs)
    SS["mc_nonce"] = SS.get("mc_nonce", 0) + 1


def _init_for(sid: Optional[str], fleet: List[Dict[str, Any]], step: Optional[int] = None,
              setup=None, toast: str = "") -> None:
    if setup is None:
        setup = (_published(sid, fleet) if sid else None) or cb.blank_setup()
    versions = sv.list_versions(sid) if sid else []
    SS["mc_sid"] = sid
    SS["mc_setup"] = cb.normalize(setup)
    msg = cb.init_message(setup, host="web", library=_library(fleet), versions=versions,
                          current_id=sid or "", step=step)
    if toast:
        msg["toast"] = toast
    _push(msg)


# ---------------------------------------------------------------------
# Estado inicial
# ---------------------------------------------------------------------
fleet = _fleet()
if "mc_nonce" not in SS:
    first = fleet[0]["id"] if fleet else None
    _init_for(first, fleet, step=0 if first else None)

# ---------------------------------------------------------------------
# Componente (config.html) + acciones
# ---------------------------------------------------------------------
val = machine_config(SS.get("mc_msgs", []), nonce=str(SS.get("mc_nonce", 0)), key="mc_cfg")

if val and val.get("ts") and val["ts"] != SS.get("mc_last_ts"):
    SS["mc_last_ts"] = val["ts"]
    name = val.get("name")
    if val.get("setup"):
        SS["mc_setup"] = val["setup"]
    cur = SS.get("mc_setup") or {}
    sid = SS.get("mc_sid")

    def _sid_for_save() -> str:
        return sid or _slug(((cur.get("machine") or {}).get("name") or "").strip() or "Machine")

    def _versions_msg(toast: str = "") -> Dict[str, Any]:
        m = {"wm": "versions", "versions": cb.versions_view(sv.list_versions(SS["mc_sid"]))}
        if toast:
            m["toast"] = toast
        return m

    def _sync_live(setup) -> str:
        # Machines = fuente ÚNICA: al publicar SIEMPRE se copia al activo que leen
        # Live, Rotodinámica, reportes, Report Center y la app (nunca borra).
        if not setup.instance_id:
            return ""
        from core.remote_monitoring.instance_sync import sync_instance
        r = sync_instance(setup)
        if not r.get("ok"):
            return f" · Live asset not updated: {r.get('reason')}"
        parts = []
        if r.get("created"):
            parts.append("created")
        if r.get("added"):
            parts.append("+" + ", ".join(x.replace("_", "") for x in r["added"]))
        if r.get("updated"):
            parts.append(f"{len(r['updated'])} updated")
        if r.get("header"):
            parts.append(", ".join(f"{k} → {v}" for k, v in r["header"].items()))
        return (" · Live asset: " + " · ".join(parts)) if parts else ""

    if name == "load":
        _init_for(val.get("id"), fleet, step=0, toast="Machine loaded")
    elif name == "new":
        _init_for(None, fleet, step=0, toast="New machine — fill the asset record, then Continuar")
    elif name == "build":
        out = cb.handle_build(cur, val.get("spec") or {})
        SS["mc_setup"] = out["setup"]
        _push(out)
    elif name == "validate":
        _push({"wm": "findings", "findings": cb.findings(cur)})
    elif name in ("save_draft", "publish"):
        fs = cb.findings(cur)
        if name == "publish" and cb.n_errors(fs):
            _push({"wm": "findings", "findings": fs, "toast": "Fix Validation errors before publishing"})
        else:
            SS["mc_sid"] = _sid_for_save()
            setup = setup_from_dict(cur)
            res = sv.save_draft(setup, origin="web", author=USER, setup_id=SS["mc_sid"],
                                note=val.get("note") or "")
            if not res.get("ok"):
                _push({"wm": "toast", "text": f"Not saved: {res.get('reason')}"})
            elif name == "save_draft":
                _push(_versions_msg(f"Draft v{res['version']} saved"))
            else:
                pr = sv.publish(SS["mc_sid"], res["version"], by=USER)
                txt = (f"Published v{res['version']}" + _sync_live(setup)) if pr.get("ok") \
                    else f"Saved as draft v{res['version']} — not published: {pr.get('reason')}"
                _push(_versions_msg(txt))
    elif name == "accept" and sid:
        ver = int(val.get("version") or 0)
        row = next((r for r in sv.list_versions(sid) if int(r.get("version") or 0) == ver), None)
        pr = sv.publish(sid, ver, by=USER)
        if pr.get("ok") and row:
            setup = setup_from_dict(row["spec"])
            _init_for(sid, _fleet(), step=6, setup=setup,
                      toast=f"Published v{ver}" + _sync_live(setup))
        else:
            _push(_versions_msg(f"Not published: {pr.get('reason')}"))
    elif name == "reject" and sid:
        sv.reject(sid, int(val.get("version") or 0))
        _push(_versions_msg("Draft rejected"))
    elif name == "open_version" and sid:
        ver = int(val.get("version") or 0)
        row = next((r for r in sv.list_versions(sid) if int(r.get("version") or 0) == ver), None)
        if row:
            setup = setup_from_dict(row["spec"])
            SS["mc_setup"] = cb.normalize(setup)
            _push({"wm": "setup", "setup": cb.normalize(setup), "toast": f"Opened v{ver} in the editor"})
    # 'apply' y acciones solo-campo: nada que empujar a la página.
    if name not in ("apply", "save_cloud", "save_file", "save_local", "apply_measure", "open_file"):
        st.rerun()

# ---------------------------------------------------------------------
# Transición: activo de Live (Machinery Library) — solo web
# ---------------------------------------------------------------------
_cur = SS.get("mc_setup") or {}
if _cur.get("instance_id"):
    try:
        from core.instance_state import get_instance
        from core.remote_monitoring.instance_sync import header_changes, sync_sensors_from_setup
        _setup = setup_from_dict(_cur)
        _inst = get_instance(_setup.instance_id)
        _, _rep = sync_sensors_from_setup(_inst.sensors if _inst else [], _setup)
        _hdr = header_changes(_inst, _setup)
        _bits = []
        if _inst is None:
            _bits.append(f"new Live asset “{_setup.instance_id}”")
        if _rep["added"]:
            _bits.append("adds " + ", ".join(x.replace("_", "") for x in _rep["added"]))
        if _rep["updated"]:
            _bits.append(f"updates limits/angles of {len(_rep['updated'])} sensor(s)")
        if _hdr:
            _bits.append("asset record: " + ", ".join(f"{k} “{o or '—'}” → “{n}”" for k, (o, n) in _hdr.items()))
        st.caption(
            "Machines is the single source: publishing updates Live, Rotordynamics, reports, "
            "Report Center and the mobile app (adds / updates only, never deletes)"
            + (f" · on publish: {'; '.join(_bits)}" if _bits else " · already in sync"))
    except Exception as _e:  # noqa: BLE001
        st.caption(f"Live asset preview unavailable: {_e}")
