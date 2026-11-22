"""
pages/00_Machines.py
====================

**Machines** — configuración ÚNICA de máquina (reemplaza Machinery Library +
Crear activo wizard; ver memoria machines-unified-config-modbus-to-ni).

  1 · Identity          nombre, enlace con Live (instance_id), modo, origen
  2 · Train & sensors   el MISMO constructor de tren del campo (builder.html)
  3 · Channels & limits tabla de sondas: variable, ángulos, límites, BNC, Modbus
  4 · Publish           diff vs publicada → borrador → publicar (el nodo en
                        línea / collector lo toma); historial de versiones

Nube = fuente de verdad (rm_setups = publicada, rm_setup_versions = historial).
Web y campo construyen; en máquina EN LÍNEA lo del campo entra como borrador y
se acepta aquí. Sin sobrescritura silenciosa.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import streamlit as st

from core.auth import render_user_menu, require_login, require_role

st.set_page_config(page_title="Watermelon System | Machines", layout="wide",
                   initial_sidebar_state="expanded")
require_login()
render_user_menu()
require_role(("admin", "specialist"))

import pandas as pd  # noqa: E402

from core.remote_monitoring import setup_versions as sv  # noqa: E402
from core.remote_monitoring.builder_spec import (  # noqa: E402
    canonical_label, display_label, merge_builder_spec)
from core.remote_monitoring.config import (  # noqa: E402
    SENSOR_TYPES, AcqSetup, ChannelRow, MachineConfig, _RM_SETUPS_TABLE, _rm_client, _slug,
    setup_from_dict, setup_to_dict)
from core.remote_monitoring.modbus_binding import diff_modbus_maps, modbus_map_from_setup  # noqa: E402
from core.ui_industrial import dot, html_table, industrial_band, inject_industrial_css  # noqa: E402
from core.ui_theme import apply_watermelon_page_style, page_header  # noqa: E402

apply_watermelon_page_style()
inject_industrial_css()
page_header("Machines",
            subtitle="One configuration per machine · the field and the web build the same "
                     "train · versioned, reviewed and published to online nodes.")

st.markdown("""
<style>
.mc-fleet{display:flex;flex-direction:column;gap:6px}
.mc-meta{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#6b7d95;letter-spacing:.02em}
.mc-pill{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:10.5px;font-weight:600;
  letter-spacing:.05em;text-transform:uppercase;padding:2px 8px;border-radius:999px;margin-right:4px;
  border:1px solid #d5deea;color:#3a4c66;background:#f4f7fb}
.mc-pill.on{background:#e8f6ee;border-color:#a9dcc0;color:#1f7a47}
.mc-pill.warn{background:#fff4e5;border-color:#f3c98b;color:#a85d00}
.mc-note{border:1px dashed #c9d5e4;border-radius:10px;padding:10px 14px;color:#3a4c66;
  background:#f7f9fc;font-size:13.5px;margin:6px 0 12px}
.mc-diff{font-family:'IBM Plex Mono',monospace;font-size:12.5px;line-height:1.55;
  border:1px solid #e2e8f2;border-radius:10px;padding:10px 14px;background:#fbfcfe}
.mc-diff .a{color:#1f7a47}.mc-diff .r{color:#b42318}.mc-diff .c{color:#a85d00}
</style>""", unsafe_allow_html=True)

USER = ""
try:
    from core.auth import get_current_user
    _u = get_current_user() or {}
    USER = _u.get("full_name") or _u.get("username") or _u.get("email") or ""
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------
@st.cache_data(ttl=20, show_spinner=False)
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
        out.append({"id": r["id"], "name": r.get("name") or r["id"],
                    "updated_at": r.get("updated_at") or "",
                    "mode": m.get("mode") or "spot", "source": m.get("source") or "ni",
                    "version": m.get("_version"), "instance_id": m.get("instance_id") or "",
                    "n": len(m.get("channels") or []), "meta": m})
    return sorted(out, key=lambda r: (r["mode"] != "online", r["name"].lower()))


@st.cache_data(ttl=10, show_spinner=False)
def _versions(sid: str) -> List[Dict[str, Any]]:
    return sv.list_versions(sid)


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default


def _txt(v: Any) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _refresh():
    _fleet.clear()
    _versions.clear()


def _work_key(sid: str) -> str:
    return f"mc_work::{sid}"


def _work(sid: str) -> AcqSetup:
    return setup_from_dict(st.session_state[_work_key(sid)])


def _set_work(sid: str, setup: AcqSetup) -> None:
    st.session_state[_work_key(sid)] = setup_to_dict(setup)


def _published_spec(sid: str, fleet_row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if fleet_row is None:
        return None
    m = dict(fleet_row["meta"])
    m.pop("_version", None)
    return setup_to_dict(setup_from_dict(m))


# ---------------------------------------------------------------------
# Flota (izquierda) + máquina seleccionada (derecha)
# ---------------------------------------------------------------------
fleet = _fleet()
by_id = {r["id"]: r for r in fleet}
new_ids = [k.split("::", 1)[1] for k in st.session_state if k.startswith("mc_work::")
           and k.split("::", 1)[1] not in by_id]

left, right = st.columns([1, 3.2], gap="large")

with left:
    st.markdown('<div class="bf-label">Fleet</div>', unsafe_allow_html=True)
    ids = [r["id"] for r in fleet] + new_ids
    if not ids:
        st.info("No machines yet.")
    labels = {}
    for r in fleet:
        labels[r["id"]] = (f"{'●' if r['mode'] == 'online' else '○'}  {r['name']}  ·  "
                           f"{'v' + str(r['version']) if r['version'] else 'unversioned'}")
    for nid in new_ids:
        labels[nid] = f"✎  {nid}  ·  new"
    sid = st.radio("Machine", ids, format_func=lambda i: labels.get(i, i),
                   key="mc_sel", label_visibility="collapsed") if ids else None
    st.markdown('<div class="mc-meta">● online · ○ spot measurement</div>', unsafe_allow_html=True)
    with st.popover("New machine", use_container_width=True):
        nm = st.text_input("Name", key="mc_new_name", placeholder="e.g. Compressor K-201")
        if st.button("Create", type="primary", disabled=not nm.strip()):
            nid = _slug(nm.strip())
            if nid in by_id or _work_key(nid) in st.session_state:
                st.error("A machine with that name already exists.")
            else:
                _set_work(nid, AcqSetup(machine=MachineConfig(name=nm.strip()), channels=[]))
                st.session_state["mc_sel"] = nid
                st.rerun()

if not sid:
    st.stop()

row = by_id.get(sid)
versions = _versions(sid) if row else []
pub_spec = _published_spec(sid, row)
if _work_key(sid) not in st.session_state:
    # Arranca del ÚLTIMO borrador pendiente (si hay) o de la publicada.
    drafts = [v for v in versions if v["status"] == "draft"]
    start = drafts[0]["spec"] if drafts else (pub_spec or {})
    st.session_state[_work_key(sid)] = setup_to_dict(setup_from_dict(start))
work = _work(sid)
work_spec = setup_to_dict(work)
d_pub = sv.diff_specs(pub_spec, work_spec)
dirty = not sv.diff_is_empty(d_pub)
pub_v = (sv.published_row(versions) or {}).get("version") or (row or {}).get("version")
pending = [v for v in versions if v["status"] == "draft"]

with right:
    n_mb = sum(1 for c in work.channels if c.modbus)
    industrial_band(
        "Machine",
        work.machine.name or sid,
        (f"Live link: {work.instance_id or '—'} · "
         f"{'Online monitoring' if work.mode == 'online' else 'Spot measurement'} · "
         f"source {'Modbus (protection rack)' if work.source == 'modbus' else 'Watermelon NI node'}"),
        kpis=[(f"v{pub_v}" if pub_v else "—", "published"),
              (str(len(work.channels)), "sensors"),
              (str(n_mb), "Modbus-bound"),
              (str(len(pending)), "drafts pending")],
    )
    if pending:
        fld = [v for v in pending if v.get("origin") == "field"]
        if fld:
            st.markdown(
                f'<div class="mc-note">{dot("warn")} <b>Field upload waiting for review</b> — '
                f'v{fld[0]["version"]} by {fld[0].get("author") or "field"}. '
                f'Open <b>4 · Publish</b> to compare and accept it.</div>', unsafe_allow_html=True)

    step = st.segmented_control(
        "Step", ["1 · Identity", "2 · Train & sensors", "3 · Channels & limits", "4 · Publish"],
        default="2 · Train & sensors", key=f"mc_step::{sid}", label_visibility="collapsed") \
        or "2 · Train & sensors"

    # ---------------- 1 · Identity ----------------
    if step.startswith("1"):
        with st.form(f"mc_id::{sid}"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Machine name", work.machine.name)
            iid = c2.text_input("Live link (instance_id)", work.instance_id or sid,
                                help="Key used by Live Monitoring, trends and reports. "
                                     "Changing it breaks history continuity.")
            c3, c4, c5 = st.columns(3)
            mode = c3.radio("Mode", ["online", "spot"], horizontal=True,
                            index=0 if work.mode == "online" else 1,
                            format_func=lambda m: "Online monitoring" if m == "online" else "Spot measurement")
            has_mb = any(c.modbus for c in work.channels)
            src = c4.radio("Signal source", ["modbus", "ni"] if has_mb else ["ni"], horizontal=True,
                           index=0 if (work.source == "modbus" and has_mb) else (1 if has_mb else 0),
                           format_func=lambda s: "Modbus rack" if s == "modbus" else "NI node")
            rot = c5.radio("Rotation", ["CW", "CCW"], horizontal=True,
                           index=0 if work.machine.rotation == "CW" else 1)
            c6, c7, c8 = st.columns(3)
            rpm = c6.number_input("Nominal speed (rpm)", 0.0, 100000.0, float(work.machine.rpm_nominal), 10.0)
            mtype = c7.text_input("Machine type", work.machine.machine_type)
            tag = c8.text_input("Tag", work.machine.tag)
            if st.form_submit_button("Apply", type="primary"):
                work.machine.name, work.instance_id, work.mode, work.source = name.strip(), iid.strip(), mode, src
                work.machine.rotation, work.machine.rpm_nominal = rot, rpm
                work.machine.machine_type, work.machine.tag = mtype.strip(), tag.strip()
                _set_work(sid, work)
                st.rerun()
        if work.modbus_gateway:
            gw = work.modbus_gateway.get("modbus") or {}
            st.markdown(
                f'<div class="mc-note">{dot("info")} Modbus gateway '
                f'<b>{gw.get("server_ip", "—")}:{gw.get("port", 502)}</b> · unit {gw.get("unit_id", 1)} · '
                f'{gw.get("byte_order", "ABCD")} · read {work.modbus_gateway.get("read_interval_seconds", "—")} s / '
                f'post {work.modbus_gateway.get("poll_interval_seconds", "—")} s '
                f'({work.modbus_gateway.get("aggregate", "peak")}). The site collector downloads this map '
                f'from the published version.</div>', unsafe_allow_html=True)

    # ---------------- 2 · Train & sensors ----------------
    elif step.startswith("2"):
        from core.machine_builder_component import machine_builder
        if not work.layout:
            st.markdown(
                f'<div class="mc-note">{dot("info")} '
                + ("This machine came from the protection rack’s Modbus map — draw its train once. "
                   "Existing sensors, their Live names and Modbus registers are kept as they are."
                   if work.channels else
                   "Pick equipment from the library, set bearings and probes, then press "
                   "<b>Construir máquina</b>.")
                + '</div>', unsafe_allow_html=True)
        val = machine_builder(work.layout, nonce=f"{sid}:{pub_v}:{bool(work.layout)}", key=f"mc_b::{sid}")
        seen_k = f"mc_bts::{sid}"
        if val and val.get("ts") and val["ts"] != st.session_state.get(seen_k):
            st.session_state[seen_k] = val["ts"]
            new, rep = merge_builder_spec(work, val.get("spec") or {})
            _set_work(sid, new)
            st.session_state[f"mc_brep::{sid}"] = rep
            st.rerun()
        rep = st.session_state.get(f"mc_brep::{sid}")
        if rep:
            parts = [f'<span class="a">+ {len(rep["added"])} new probe(s)</span> {", ".join(map(display_label, rep["added"]))}',
                     f'{len(rep["matched"])} existing kept as-is']
            if rep["not_in_train"]:
                parts.append(f'<span class="c">{len(rep["not_in_train"])} not in the drawn train (kept): '
                             f'{", ".join(map(display_label, rep["not_in_train"]))}</span>')
            st.markdown('<div class="mc-diff">' + "<br>".join(parts) + "</div>", unsafe_allow_html=True)

    # ---------------- 3 · Channels & limits ----------------
    elif step.startswith("3"):
        rows = []
        for i, c in enumerate(work.channels):
            rows.append({"_i": i, "Sensor": display_label(c.point_label), "Live variable": c.variable,
                         "Location": c.location, "Type": c.sensor_type, "Bearing": c.plane,
                         "Angle °": c.angle_deg, "Side": c.side, "Unit": c.unit_native,
                         "mV/EU": c.sensitivity_mv_per_eu, "Alarm": c.alarm, "Danger": c.danger,
                         "BNC": c.bnc_port, "Active": c.active,
                         "Modbus": ", ".join(f"{m.split('@')[0]}→{e.get('register')}"
                                             for m, e in sorted(c.modbus.items(),
                                                                key=lambda t: t[1].get("register", 0)))})
        df = pd.DataFrame(rows, columns=["_i", "Sensor", "Live variable", "Location", "Type", "Bearing",
                                         "Angle °", "Side", "Unit", "mV/EU", "Alarm", "Danger", "BNC",
                                         "Active", "Modbus"])
        ed = st.data_editor(
            df, key=f"mc_tbl::{sid}", hide_index=True, num_rows="dynamic", use_container_width=True,
            column_config={
                "_i": None,
                "Type": st.column_config.SelectboxColumn(options=SENSOR_TYPES, required=True),
                "Side": st.column_config.SelectboxColumn(options=["", "L", "R"]),
                "Modbus": st.column_config.TextColumn(disabled=True,
                                                      help="Metric → register on the protection rack"),
                "Live variable": st.column_config.TextColumn(
                    help="Name posted to Live. Keep it when migrating Modbus → NI so trends continue."),
            })
        st.markdown('<div class="mc-meta">Sensor names show without underscore (1XD); stored as 1X_D. '
                    'Deleting a Modbus-bound row stops that sensor in Live after publishing.</div>',
                    unsafe_allow_html=True)
        if st.button("Apply table", type="primary"):
            new_ch: List[ChannelRow] = []
            for _, r in ed.iterrows():
                lbl = _txt(r.get("Sensor")).strip()
                if not lbl:
                    continue
                base = (work.channels[int(r["_i"])] if pd.notna(r.get("_i")) else
                        ChannelRow(bnc_port=0, point_label=canonical_label(lbl)))
                ch = copy.deepcopy(base)
                if display_label(ch.point_label) != lbl:
                    ch.point_label = canonical_label(lbl)
                ch.variable = _txt(r.get("Live variable"))
                ch.location = _txt(r.get("Location"))
                ch.sensor_type = _txt(r.get("Type")) or ch.sensor_type
                ch.plane = int(_num(r.get("Bearing")))
                ch.angle_deg = _num(r.get("Angle °"))
                ch.side = _txt(r.get("Side"))
                ch.unit_native = _txt(r.get("Unit")) or ch.unit_native
                ch.sensitivity_mv_per_eu = _num(r.get("mV/EU"), ch.sensitivity_mv_per_eu)
                ch.alarm, ch.danger = _num(r.get("Alarm")), _num(r.get("Danger"))
                ch.bnc_port = int(_num(r.get("BNC")))
                ch.active = bool(r.get("Active")) if pd.notna(r.get("Active")) else True
                new_ch.append(ch)
            work.channels = new_ch
            _set_work(sid, work)
            st.rerun()

    # ---------------- 4 · Publish ----------------
    else:
        if not sv.versions_available():
            st.error("Version history table is not available.")
            st.stop()
        if row and not versions:
            st.markdown(f'<div class="mc-note">{dot("info")} This machine has no version history yet.</div>',
                        unsafe_allow_html=True)
            if st.button("Start history from the current configuration (v1)"):
                st.toast(str(sv.baseline_from_current(sid, by=USER or "web")))
                _refresh()
                st.rerun()

        def _diff_html(d: Dict[str, List[str]]) -> str:
            lines = [f'<span class="a">+ {display_label(x)}</span>' for x in d["added"]]
            lines += [f'<span class="r">− {display_label(x)}</span>' for x in d["removed"]]
            lines += [f'<span class="c">~ {x}</span>' for x in d["changed"] + d["machine"] + d["source"]]
            return '<div class="mc-diff">' + ("<br>".join(lines) or "No changes") + "</div>"

        st.markdown('<div class="bf-label">Your changes vs published</div>', unsafe_allow_html=True)
        st.markdown(_diff_html(d_pub), unsafe_allow_html=True)
        if work.source == "modbus" and pub_spec:
            mdiff = diff_modbus_maps(modbus_map_from_setup(setup_from_dict(pub_spec)),
                                     modbus_map_from_setup(work))
            if not sv.diff_is_empty(mdiff):
                st.markdown('<div class="bf-label">Collector map changes</div>', unsafe_allow_html=True)
                st.markdown(_diff_html({"added": mdiff["added"], "removed": mdiff["removed"],
                                        "changed": mdiff["changed"], "machine": [],
                                        "source": mdiff["gateway"]}), unsafe_allow_html=True)
        note = st.text_input("Change note", key=f"mc_note::{sid}",
                             placeholder="e.g. added lube-pump accelerometer LB0011")
        b1, b2, _ = st.columns([1.2, 1.4, 3])
        if b1.button("Save draft", disabled=not dirty and bool(row)):
            res = sv.save_draft(work, origin="web", author=USER, note=note, setup_id=sid)
            st.toast(f"Draft v{res['version']} saved" if res.get("ok") else f"Not saved: {res.get('reason')}")
            _refresh()
            st.rerun()
        if b2.button("Save & publish", type="primary", disabled=not dirty and bool(row)):
            res = sv.save_draft(work, origin="web", author=USER, note=note, setup_id=sid)
            if res.get("ok"):
                pr = sv.publish(sid, res["version"], by=USER)
                st.toast(f"Published v{res['version']}" if pr.get("ok") else f"Not published: {pr.get('reason')}")
            else:
                st.toast(f"Not saved: {res.get('reason')}")
            _refresh()
            st.rerun()

        if pending:
            st.markdown('<div class="bf-label">Drafts waiting for review</div>', unsafe_allow_html=True)
            for v in pending:
                why = sv.publish_conflict(versions, v)
                with st.container(border=True):
                    st.markdown(
                        f'<b>v{v["version"]}</b> <span class="mc-pill {"warn" if v["origin"] == "field" else ""}">'
                        f'{v["origin"]}</span> {v.get("author") or ""} · {str(v["created_at"])[:16]} '
                        f'— {v.get("note") or ""}', unsafe_allow_html=True)
                    st.markdown(_diff_html(sv.diff_specs(pub_spec, v["spec"])), unsafe_allow_html=True)
                    if why:
                        st.warning(why)
                    c1, c2, c3 = st.columns([1, 1, 3])
                    if c1.button("Accept & publish", key=f"mc_pub::{sid}::{v['version']}",
                                 type="primary", disabled=bool(why)):
                        pr = sv.publish(sid, v["version"], by=USER)
                        st.toast("Published" if pr.get("ok") else pr.get("reason"))
                        st.session_state.pop(_work_key(sid), None)
                        _refresh()
                        st.rerun()
                    if c2.button("Reject", key=f"mc_rej::{sid}::{v['version']}"):
                        sv.reject(sid, v["version"])
                        _refresh()
                        st.rerun()
                    if c3.button("Load into editor", key=f"mc_load::{sid}::{v['version']}"):
                        st.session_state[_work_key(sid)] = setup_to_dict(setup_from_dict(v["spec"]))
                        st.rerun()

        if versions:
            st.markdown('<div class="bf-label">History</div>', unsafe_allow_html=True)
            sev = {"published": "ok", "draft": "warn", "superseded": "off", "rejected": "dang"}
            html_table(["", "Version", "Status", "Origin", "By", "When", "Node", "Note"],
                       [[dot(sev.get(v["status"], "off")), f"v{v['version']}", v["status"], v["origin"],
                         v.get("published_by") or v.get("author") or "",
                         str(v.get("published_at") or v["created_at"])[:16].replace("T", " "),
                         (v.get("node_status") or "") + (f" · {str(v['node_at'])[:16]}" if v.get("node_at") else ""),
                         v.get("note") or ""] for v in versions], raw_cols=[0])
        if work.mode == "online" and work.source == "modbus":
            st.markdown(
                f'<div class="mc-meta">Online · Modbus: the site collector (v1.2+, remote map on) applies the '
                f'published version within its refresh window and keeps its local copy if the cloud is '
                f'unreachable.</div>', unsafe_allow_html=True)
