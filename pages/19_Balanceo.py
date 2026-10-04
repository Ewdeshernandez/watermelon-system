"""
pages/19_Balanceo.py — Módulo Balanceo Watermelon
=================================================

Balanceo de rotores en campo por coeficiente de influencia, integrado a
Watermelon. Motor de cálculo: core.balance (extraído de ROTORIX, validado en
campo). Esta página es solo UI + wiring; NO contiene matemática de balanceo.

Pestañas
--------
1. Peso de prueba   — recomendación API 684 (Umax = 6350·W/N).
2. Balanceo 1 plano — coeficiente de influencia (ISO 21940-12).
3. Balanceo 2 planos— matriz de coeficientes de influencia (ISO 21940-12).
4. Validación ISO   — desbalance residual permisible y grados (ISO 21940-11).
5. Reporte          — resumen + PDF branded Watermelon/SIGA.

Entrada de datos: MANUAL o importada desde LIVE MONITORING (vector 1X mag+fase
por sonda). Selección por sonda/plano agrupada por sección; en 2 planos la
misma dirección (X/Y) en ambos planos por defecto.

UI: kit enterprise core.balance.ui (mismo lenguaje visual que Modal Analysis).
Acceso: analista / admin (gateada; el cliente no la ve).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st

from core.auth import (
    require_login, render_user_menu, get_current_user, is_page_allowed_for_role,
)
from core.ui_theme import apply_watermelon_page_style
from core.balance import (
    to_complex, to_polar,
    umax_api684_gmm, recommend_trial_weight_g,
    calc_U_trial, pct_reduction, pct_change,
    evaluate_iso_grades,
    solve_1plane, solve_2plane,
    diagnose_1plane, diagnose_2plane, iso_residual_sanity,
)
from core.balance.ui import (
    bal_hero_card, bal_section_header, bal_kpi_row, bal_status_banner,
    bal_footer_norms,
)
from core.ui_industrial import dot, html_table
from core.balance.rotorface import rotor_face_svg, build_planes_1p, build_planes_2p


# =====================================================================
# Setup + auth
# =====================================================================
st.set_page_config(
    page_title="Watermelon System | Balancing",
    page_icon="⚖️",
    layout="wide",
)

require_login()
render_user_menu()
apply_watermelon_page_style()

# Quitar las flechas +/- de TODOS los number_input de esta página (el usuario no
# las quiere). Se teclea el valor directo.
st.markdown("""<style>
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"]{display:none !important;}
div[data-testid="stNumberInput"] input{text-align:left;}
</style>""", unsafe_allow_html=True)

_user = get_current_user() or {}
_role = str(_user.get("role", "")).lower()
if not is_page_allowed_for_role("pages/19_Balanceo.py", _role):
    st.error("Your role does not have access to this module.")
    st.stop()

UNITS = ["µm pk-pk", "mil pk-pk", "mm/s RMS"]


# =====================================================================
# Hero del módulo (activo + modo activo, estilo Modal)
# =====================================================================
def _hero() -> None:
    if st.session_state.get("bal_r2p"):
        mode = "2 planes"
    elif st.session_state.get("bal_r1p"):
        mode = "1 plane"
    else:
        mode = "—"
    bal_hero_card(
        asset_name=st.session_state.get("rep_asset") or "(unspecified asset)",
        client=st.session_state.get("rep_client", ""),
        site=st.session_state.get("rep_location", ""),
        mode=mode,
    )


_hero()


# =====================================================================
# Helpers de UI
# =====================================================================
def _num(key: str, label: str, default: float = 0.0, **kw) -> float:
    # Persistencia entre pasos: con navegación condicional Streamlit PURGA el
    # estado de los widgets no renderizados. Se guarda una copia en una clave
    # "sombra" (no-widget) y se restaura al re-montar el widget.
    _sk = f"_keep_{key}"
    if key not in st.session_state:
        st.session_state[key] = float(st.session_state.get(_sk, default))
    val = st.number_input(label, key=key, **kw)
    st.session_state[_sk] = val
    return val


def _persist_get(key: str, default=None):
    """Lee una clave de widget con fallback a su copia persistente (_keep_),
    por si el widget fue purgado al estar en otro paso."""
    v = st.session_state.get(key)
    if v is None:
        v = st.session_state.get(f"_keep_{key}", default)
    return v


# =====================================================================
# Persistencia multi-día — BORRADOR en la nube (un balanceo dura días)
# =====================================================================
# El session_state de Streamlit es efímero (se pierde al cerrar/recargar). Para
# que un balanceo sobreviva días, se guarda un BORRADOR en Supabase (balance_runs,
# nombre "DRAFT …", por usuario+activo) y se puede reanudar al volver.
_DRAFT_SCALARS = ["bal_source", "bal_cfg", "bal_src", "bal_r1p", "bal_r2p",
                  "bal_iso", "bal_r1p_warn", "bal_r2p_warn"]


def _enc(o):
    """Codifica a JSON-safe: complejos → {'__c__':[re,im]}, numpy → nativo."""
    import numpy as _np
    if isinstance(o, complex):
        return {"__c__": [float(o.real), float(o.imag)]}
    if isinstance(o, _np.ndarray):
        return [_enc(x) for x in o.tolist()]
    if isinstance(o, _np.floating):
        return float(o)
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, dict):
        return {k: _enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_enc(v) for v in o]
    return o


def _dec(o):
    """Revierte _enc: {'__c__':[re,im]} → complejo."""
    if isinstance(o, dict):
        if set(o.keys()) == {"__c__"}:
            _v = o["__c__"]
            return complex(_v[0], _v[1])
        return {k: _dec(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_dec(v) for v in o]
    return o


def _user_key() -> str:
    return (str(_user.get("email") or _user.get("full_name") or "user")).strip().lower()


def _draft_snapshot() -> dict:
    """Estado serializable del balanceo en curso (config + vectores + resultados)."""
    keeps = {k: st.session_state[k] for k in list(st.session_state.keys())
             if isinstance(k, str) and k.startswith("_keep_")}
    data = {"kind": "balance_draft", "keeps": keeps}
    for k in _DRAFT_SCALARS:
        if k in st.session_state:
            data[k] = st.session_state[k]
    return _enc(data)


def _restore_draft(payload: dict) -> None:
    """Restaura un borrador a la sesión (vectores + config + resultados)."""
    payload = _dec(payload)
    for k, v in (payload.get("keeps") or {}).items():
        st.session_state[k] = v                       # _num leerá de _keep_
    for k in _DRAFT_SCALARS:
        if k in payload:
            st.session_state[k] = payload[k]
    st.session_state["bal_cfg_ok"] = bool(payload.get("bal_cfg"))
    _cfg = payload.get("bal_cfg") or {}
    if _cfg:
        _apply_cfg(_cfg)
    st.session_state["_pending_nav"] = ("Balanceo" if st.session_state["bal_cfg_ok"]
                                        else "Origen")


def _reset_balance() -> None:
    """Limpia el balanceo en curso de la sesión (no borra borradores en nube)."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and (
                k.startswith("_keep_") or k.startswith("b1_") or k.startswith("b2_")
                or k.startswith("iso_") or k.startswith("rep_") or k.startswith("cfg_")
                or k in _DRAFT_SCALARS or k in (
                    "bal_cfg_ok", "bal_pdf", "_draft_hash", "_draft_id")):
            del st.session_state[k]


def _maybe_autosave() -> None:
    """Auto-guardado silencioso: escribe el borrador en la nube solo si el estado
    cambió desde la última escritura (no martilla Supabase)."""
    cfg = st.session_state.get("bal_cfg") or {}
    asset = (cfg.get("asset") or "").strip()
    if not asset:                                     # sin activo aún, nada que guardar
        return
    snap = _draft_snapshot()
    import json as _json
    try:
        _h = hash(_json.dumps(snap, sort_keys=True, default=str))
    except Exception:  # noqa: BLE001
        return
    if _h == st.session_state.get("_draft_hash"):
        return
    from core.balance import cloud
    import re as _re
    _did = st.session_state.get("_draft_id") or (
        "draft_" + _re.sub(r"[^a-z0-9]+", "-", f"{_user_key()}-{asset}".lower()).strip("-"))
    st.session_state["_draft_id"] = _did
    r = cloud.save_run(name=f"DRAFT {asset}", payload=snap, run_id=_did,
                       account=_user_key(), tag=asset)
    if r.get("ok"):
        st.session_state["_draft_hash"] = _h
        st.session_state["_draft_saved_at"] = _now_hhmm()


def _now_hhmm() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M")


def _list_my_drafts() -> list:
    try:
        from core.balance import cloud
        _me = _user_key()
        return [r for r in (cloud.list_runs() or [])
                if str(r.get("name", "")).startswith("DRAFT")
                and (r.get("account") or "").lower() == _me]
    except Exception:  # noqa: BLE001
        return []


def _vector_inputs(prefix: str, title: str, unit: str):
    st.markdown(f"<div style='font-weight:700;color:#0F1E3D;font-size:13px;"
                f"margin-bottom:2px;'>{title}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        mag = _num(f"{prefix}_mag", f"Magnitude [{unit}]", 0.0,
                   min_value=0.0, step=0.1, format="%.3f")
    with c2:
        ang = _num(f"{prefix}_ang", "Angle [°]", 0.0, step=1.0, format="%.1f")
    return mag, ang


def _quality_severity(quality: str) -> Tuple[str, str, str]:
    q = (quality or "").upper()
    if q == "GOOD":
        return ("ok", "Stable model", "GOOD")
    if q in ("MED", "MEDIUM"):
        return ("warning", "Noise-sensitive — check repeatability", "MED")
    return ("fail", "Low sensitivity / ill-conditioned matrix", "POOR")


# =====================================================================
# Import desde Live Monitoring
# =====================================================================
def _instance_options() -> List[Tuple[str, str]]:
    try:
        from core.instance_state import list_instances
        insts = list_instances() or []
    except Exception:
        insts = []
    opts: List[Tuple[str, str]] = []
    for m in insts:
        iid = m.get("instance_id")
        if not iid:
            continue
        opts.append((iid, f"{m.get('tag') or iid}  ·  {iid}"))
    return opts


def _bal_capture_cb(iid: str, targets: List[Tuple[str, str, str]]) -> None:
    from core.balance.live_source import capture_1x
    labels = [t[0] for t in targets if t[0]]
    res = capture_1x(iid, labels) if labels else {}
    ok, warn = [], []
    for lbl, mag_key, ang_key in targets:
        v = res.get(lbl) if lbl else None
        if v is None:
            warn.append(f"no live 1X for {lbl or '—'}")
            continue
        mag, ph, _unit, _ts = v
        st.session_state[mag_key] = float(mag)
        st.session_state[ang_key] = float(ph)
        ok.append(f"{lbl}: {mag:.3f} ∠ {ph:.1f}°")
    st.session_state["_bal_msg"] = " · ".join(ok + warn) or "No live data."


def _suggest_trial_cb(w_key: str, n_key: str, r_key: str, k_key: str, mag_key: str) -> None:
    """on_click: calcula el peso de prueba (API 684) y lo rellena en mag_key."""
    W = float(st.session_state.get(w_key) or 0.0)
    N = float(st.session_state.get(n_key) or 0.0)
    R = float(st.session_state.get(r_key) or 0.0)
    k = float(st.session_state.get(k_key) or 1.25)
    Wt, _u = recommend_trial_weight_g(W, N, R, k)
    st.session_state[mag_key] = round(float(Wt), 2)
    st.session_state["_bal_tw_msg"] = f"Suggested trial weight: {Wt:,.2f} g (API 684)"


def _trial_weight_suggester(prefix: str, mag_key: str,
                            title: str = "Suggest trial weight (API 684)") -> None:
    """Panel compacto: W del plano + rpm + radio → sugiere y rellena el peso
    de prueba de esa corrida. W = carga soportada por ESE plano (≈ ½ del rotor
    si está entre dos cojinetes)."""
    with st.expander(title, expanded=False):
        st.caption("W = weight supported by **this plane** (≈ ½ of the rotor "
                   "between 2 bearings). API 684 formula: W_trial = 6350·W·k / (N·radius).")
        _cfg = st.session_state.get("bal_cfg") or {}
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _num(f"{prefix}_sw", "Plane weight W [kg]",
                 float((_cfg.get("rotor_mass") or 0.0) / 2.0) or 3500.0,
                 min_value=0.0, step=10.0, format="%.1f")
        with c2:
            _num(f"{prefix}_sn", "Speed N [rpm]",
                 float(_cfg.get("rpm") or st.session_state.get("iso_rpm")
                       or st.session_state.get("tw_rpm") or 3600.0),
                 min_value=0.0, step=10.0, format="%.0f")
        with c3:
            _num(f"{prefix}_sr", "Radius [mm]",
                 float(_cfg.get("radius") or 0.0) or 420.0,
                 min_value=0.0, step=1.0, format="%.1f")
        with c4:
            _num(f"{prefix}_sk", "Factor k",
                 float(_cfg.get("trial_k") or 1.25),
                 min_value=0.2, max_value=2.0, step=0.05, format="%.2f")
        Wt, _u = recommend_trial_weight_g(
            st.session_state[f"{prefix}_sw"], st.session_state[f"{prefix}_sn"],
            st.session_state[f"{prefix}_sr"], st.session_state[f"{prefix}_sk"])
        st.button(f"Suggest → {Wt:,.2f} g  ·  fill in the trial weight",
                  key=f"{prefix}_sbtn", on_click=_suggest_trial_cb,
                  args=(f"{prefix}_sw", f"{prefix}_sn", f"{prefix}_sr",
                        f"{prefix}_sk", mag_key))
        if st.session_state.get("_bal_tw_msg"):
            st.caption(st.session_state["_bal_tw_msg"])


def _machine_and_planes(key: str):
    opts = _instance_options()
    if not opts:
        st.info("No machines available.")
        return None, []
    iid = st.selectbox("Machine", [o[0] for o in opts],
                       format_func=lambda x: dict(opts).get(x, x), key=f"{key}_iid")
    from core.balance.live_source import list_balance_planes
    planes = list_balance_planes(iid)
    if not planes:
        st.warning("This machine has no radial proximity probes configured.")
    return iid, planes


def _plane_label(p) -> str:
    return f"[{p['section']}] {p['plane_label']} (plane {p['plane']})"


# =====================================================================
# Navegación por pasos — BLOQUEO REAL del flujo
# =====================================================================
# Primero se elige el ORIGEN de los datos; luego la CONFIGURACIÓN de la máquina
# (obligatoria en Manual; ya viene con los datos en Campo/En línea). Hasta no
# completar config no se habilita el balanceo; ISO y Reporte requieren resultado.
_src = st.session_state.get("bal_source")                      # None hasta elegir
_cfg_ok = bool(st.session_state.get("bal_cfg_ok"))
_has_result = bool(st.session_state.get("bal_r1p") or st.session_state.get("bal_r2p"))
_has_iso = bool(st.session_state.get("bal_iso")) and _has_result
_has_pdf = bool(st.session_state.get("bal_pdf"))

# Planos definidos UNA sola vez en Configuración; el flujo sigue esa elección.
_two = (st.session_state.get("bal_cfg") or {}).get("planes") == "2 planes"
# Origen se oculta una vez elegido+configurado (no se cambia la fuente a mitad).
if not _cfg_ok:
    _NAV = ["Origen"] + (["Configuración"] if _src else [])
else:
    _NAV = ["Configuración", "Balanceo"]
    if _has_result:
        _NAV += ["Validación ISO", "Reporte"]

# Stepper visual (muestra las etapas; gris = bloqueada).
_steps_vis = [("1 · Origen", True), ("2 · Configuración", bool(_src)),
              ("3 · Balanceo", _cfg_ok), ("4 · Validación ISO", _has_iso),
              ("5 · Reporte", _has_pdf)]
_flow_html = " ".join(
    f"<span style='color:{'#10b981' if _d else '#cbd5e1'};font-size:14px'>●</span>"
    f"<span style='color:{'#0b1f3a' if _d else '#94a3b8'};"
    f"font-weight:{700 if _d else 500};font-size:12px;margin:0 14px 0 5px'>{_lbl}</span>"
    for _lbl, _d in _steps_vis)
st.markdown(f"<div style='margin:6px 0 8px'>{_flow_html}</div>", unsafe_allow_html=True)

# Aplica un cambio de paso PENDIENTE (de validar config / cargar / reanudar)
# ANTES de crear el widget — no se puede modificar bal_nav después de instanciarlo.
_pn = st.session_state.pop("_pending_nav", None)
if _pn and _pn in _NAV:
    st.session_state["bal_nav"] = _pn
elif st.session_state.get("bal_nav") not in _NAV:
    st.session_state.pop("bal_nav", None)             # evita valor inválido en el widget

if hasattr(st, "segmented_control"):
    _active = st.segmented_control(
        "Paso", _NAV, key="bal_nav", label_visibility="collapsed")
else:
    _active = st.radio("Paso", _NAV, key="bal_nav", horizontal=True,
                       label_visibility="collapsed")
if _active not in _NAV:
    _active = _NAV[-1] if _NAV else "Origen"


def _draft_indicator() -> None:
    """Indicador de borrador (persistencia multi-día) + guardar manual."""
    if not (st.session_state.get("bal_cfg") or {}).get("asset"):
        return
    _dc1, _dc2 = st.columns([4, 1])
    with _dc1:
        _sv = st.session_state.get("_draft_saved_at")
        st.caption("● Borrador guardado automáticamente en la nube"
                   + (f" · {_sv}" if _sv else "")
                   + ". Puedes cerrar y reanudar después (dura días).")
    with _dc2:
        if st.button("Guardar ahora", use_container_width=True, key="draft_save_now"):
            st.session_state["_draft_hash"] = None      # fuerza reescritura
            try:
                _maybe_autosave()
                st.toast("Borrador guardado.")
            except Exception:  # noqa: BLE001
                st.warning("No se pudo guardar el borrador.")


def _render_bal_warnings(warns):
    """Muestra los avisos de auditoría del balanceo (crit/warn/info)."""
    if not warns:
        return
    _sev = {"crit": "fail", "warn": "warning", "info": "ok"}
    for w in warns:
        bal_status_banner(w.get("title", ""), w.get("msg", ""),
                          _sev.get(w.get("severity"), "warning"))


def _source_label(src: str) -> str:
    """Etiqueta legible de la fuente de datos."""
    return {"Manual": "Datos manuales (escritos)",
            "Live": "Monitoreo en línea (1X en vivo)",
            "Live Monitoring": "Monitoreo en línea (1X en vivo)",
            "Campo": "Enviado de campo"}.get(src, src or "—")


def _source_badge(src: str) -> None:
    """Chip visible con el ORIGEN de los datos (manual / en línea / campo)."""
    _c = {"Manual": ("#12305e", "#e8eefb", "#c7d7f0"),
          "Live": ("#166534", "#e8f6ee", "#b7e4c7"),
          "Live Monitoring": ("#166534", "#e8f6ee", "#b7e4c7"),
          "Campo": ("#8a5a00", "#fdf2e0", "#f3d9ad")}
    fg, bg, bd = _c.get(src, ("#475569", "#f1f5f9", "#e2e8f2"))
    st.markdown(
        f"<div style='display:inline-block;margin:2px 0 8px;padding:4px 12px;"
        f"border-radius:999px;background:{bg};color:{fg};border:1px solid {bd};"
        f"font:700 12px \"IBM Plex Sans\",sans-serif;'>● Fuente: "
        f"{_source_label(src)}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Paso 1 — Origen de los datos
# ---------------------------------------------------------------------
def _set_source(val: str) -> None:
    if st.session_state.get("bal_source") != val:
        st.session_state["bal_source"] = val
        st.session_state["bal_cfg_ok"] = False       # re-configurar al cambiar


def _render_origen() -> None:
    bal_section_header("Origen de los datos",
                       "Elige de dónde vienen los datos del balanceo.",
                       "Paso 1", "●")

    # Reanudar un balanceo guardado (puede durar días).
    _drafts = _list_my_drafts()
    if _drafts:
        with st.container(border=True):
            st.markdown("<div style='font-weight:700;color:#0F1E3D'>Reanudar un "
                        "balanceo guardado</div>", unsafe_allow_html=True)
            _do = {d.get("id"): f"{d.get('tag') or '—'}  ·  guardado "
                               f"{(d.get('updated_at') or '')[:16]}" for d in _drafts}
            _pick = st.selectbox("Borradores", list(_do.keys()),
                                 format_func=lambda x: _do.get(x, x), key="resume_pick")
            rc1, rc2, rc3 = st.columns([1, 1, 2])
            with rc1:
                if st.button("Reanudar", type="primary", key="resume_btn"):
                    from core.balance import cloud
                    _pl = cloud.load_run(_pick)
                    if _pl:
                        _restore_draft(_pl)
                        st.session_state["_draft_id"] = _pick
                        st.success("Balanceo reanudado.")
                        st.rerun()
                    else:
                        st.error("No se pudo cargar el borrador.")
            with rc2:
                _del_ok = st.checkbox("Confirmar borrar", key="draft_del_ok")
                if st.button("Borrar", key="draft_del_btn", disabled=not _del_ok):
                    from core.balance import cloud
                    _r = cloud.delete_run(_pick)
                    if _r.get("ok"):
                        if st.session_state.get("_draft_id") == _pick:
                            st.session_state.pop("_draft_id", None)
                        st.success("Borrador eliminado (no se puede deshacer).")
                        st.rerun()
                    else:
                        st.error(f"No se pudo borrar: {_r.get('reason', '—')}")
            with rc3:
                st.caption("Borrar es permanente. O empieza uno nuevo eligiendo "
                           "un origen abajo.")

    _opts = [
        ("Manual", "Datos manuales",
         "Tú escribes la configuración de la máquina y los vectores de vibración."),
        ("Campo", "Enviado de campo",
         "Carga una corrida subida desde el equipo de campo (trae todo listo)."),
        ("Live", "Monitoreo en línea",
         "Toma el vector 1X en vivo de una máquina monitoreada."),
    ]
    cols = st.columns(3)
    for col, (val, title, desc) in zip(cols, _opts):
        with col:
            _sel = st.session_state.get("bal_source") == val
            st.button(("● " if _sel else "") + title, key=f"src_{val}",
                      use_container_width=True,
                      type=("primary" if _sel else "secondary"),
                      on_click=_set_source, args=(val,))
            st.markdown(f"<div style='font-size:11px;color:#64748b;line-height:1.4;"
                        f"margin-top:4px'>{desc}</div>", unsafe_allow_html=True)
    if st.session_state.get("bal_source"):
        st.markdown("")
        st.caption("Continúa en **Configuración**.")


# ---------------------------------------------------------------------
# Paso 2 — Configuración de la máquina
# ---------------------------------------------------------------------
_ISO_GRADES_UI = ["0.4", "1.0", "2.5", "6.3", "16.0"]


def _apply_cfg(cfg: dict) -> None:
    """Propaga la config a las claves que usan los pasos de balanceo (ISO/trial/
    unidad/giro), para no re-teclear nada."""
    try:
        st.session_state["iso_w"] = float(cfg.get("rotor_mass") or 0.0)
        st.session_state["iso_rpm"] = float(cfg.get("rpm") or 0.0)
    except Exception:  # noqa: BLE001
        pass
    _u = cfg.get("unit")
    if _u in UNITS:                       # evita romper el selectbox con una unidad ajena
        st.session_state["b1_unit"] = _u
        st.session_state["b2_unit"] = _u
    _rot = cfg.get("rotation") if cfg.get("rotation") in ("CCW", "CW") else "CCW"
    st.session_state["b1_rot"] = _rot
    st.session_state["b2_rot"] = _rot


def _load_field_payload(payload: dict) -> dict:
    """Carga una corrida de campo (o nube) a la sesión: config + vectores +
    resultados + avisos. Devuelve el cfg derivado."""
    _one = payload.get("one_plane") or {}
    _two = payload.get("two_plane") or {}
    _iso = payload.get("iso") or {}
    _setup = payload.get("setup") or {}
    _unit = payload.get("unit") or _one.get("unit") or _two.get("unit") or "mils pk-pk"

    def _put(prefix, vec):
        if vec and len(vec) == 2:
            st.session_state[f"{prefix}_mag"] = float(vec[0])
            st.session_state[f"{prefix}_ang"] = float(vec[1])

    if _one.get("result"):
        st.session_state["bal_r1p"] = _one["result"]
        st.session_state["bal_r1p_warn"] = _one.get("warnings") or []
        _put("b1_v0", _one.get("v0")); _put("b1_tw", _one.get("trial"))
        _put("b1_vt", _one.get("vt")); _put("b1_vf", _one.get("vf"))
    if _two.get("result"):
        st.session_state["bal_r2p"] = _two["result"]
        st.session_state["bal_r2p_warn"] = _two.get("warnings") or []
        for _p, _k in (("b2_a0", "a0"), ("b2_b0", "b0"), ("b2_a1", "a1"),
                       ("b2_b1", "b1"), ("b2_a2", "a2"), ("b2_b2", "b2"),
                       ("b2_wa", "wa"), ("b2_wb", "wb"),
                       ("b2_vfa", "vf_a"), ("b2_vfb", "vf_b")):
            _put(_p, _two.get(_k))
    if _iso:
        st.session_state["bal_iso"] = _iso

    cfg = {
        "asset": _setup.get("machine") or _setup.get("tag") or "",
        "client": _setup.get("client") or "",
        "location": _setup.get("location") or "",
        "specialist": _setup.get("operator") or "",
        "rpm": float(_iso.get("N_rpm") or _setup.get("nameplate_rpm") or 0.0),
        "rotor_mass": float(_iso.get("W_kg") or 0.0),
        "radius": 0.0,
        "trial_k": 1.25,
        "iso_g": "2.5",
        "planes": "2 planes" if _two.get("result") else "1 plane",
        "rotation": "CCW",
        "unit": _unit,
    }
    st.session_state["bal_cfg"] = cfg
    st.session_state["bal_src"] = "Campo"
    _apply_cfg(cfg)
    return cfg


def _render_config() -> None:
    src = st.session_state.get("bal_source")
    _source_badge(src)
    cfg = st.session_state.get("bal_cfg") or {}

    with st.expander("Reiniciar balanceo (cambiar origen / empezar de cero)"):
        st.caption("Vuelve a elegir el origen y limpia los datos de esta sesión. "
                   "Lo guardado como borrador no se borra.")
        if st.checkbox("Confirmo reiniciar", key="cfg_reset_ok") and \
                st.button("Reiniciar ahora", type="secondary"):
            _reset_balance()
            st.rerun()

    # ---- Campo: cargar una corrida subida desde el equipo ----
    if src == "Campo":
        bal_section_header("Configuración — Enviado de campo",
                           "Carga una corrida de balanceo subida desde el equipo. "
                           "Trae la configuración, los vectores y el resultado.",
                           "Paso 2", "●")
        try:
            from core.balance import cloud
            runs = [r for r in (cloud.list_runs() or [])
                    if not str(r.get("name", "")).startswith("DRAFT")]
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer la nube: {e}")
            return
        if not runs:
            st.info("No hay corridas de balanceo en la nube todavía. Sube una "
                    "desde el módulo de campo (Watermelon Balancing).")
            return
        _ro = {r.get("id"): f"{r.get('tag') or r.get('id')}  ·  "
                            f"{(r.get('updated_at') or '')[:16]}" for r in runs}
        _rid = st.selectbox("Corrida de campo", list(_ro.keys()),
                            format_func=lambda x: _ro.get(x, x), key="cfg_run")
        _fc1, _fc2, _fc3 = st.columns([1, 1, 2])
        with _fc1:
            _load_click = st.button("Cargar corrida", type="primary")
        with _fc2:
            _fdel_ok = st.checkbox("Confirmar borrar", key="run_del_ok")
            _del_click = st.button("Borrar", key="run_del_btn", disabled=not _fdel_ok)
        with _fc3:
            st.caption("Borrar una corrida de campo es permanente.")
        if _del_click:
            _r = cloud.delete_run(_rid)
            if _r.get("ok"):
                st.success("Corrida eliminada (no se puede deshacer).")
                st.rerun()
            else:
                st.error(f"No se pudo borrar: {_r.get('reason', '—')}")
        if _load_click:
            try:
                payload = cloud.load_run(_rid)
                if not payload:
                    st.error("No se pudo cargar la corrida.")
                    return
                _load_field_payload(payload)
                st.session_state["bal_cfg_ok"] = True
                st.session_state["_pending_nav"] = "Balanceo"
                st.success("Corrida cargada. Revisa el balanceo y el reporte.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Error al cargar: {e}")
        if cfg:
            st.caption(f"Cargado: **{cfg.get('asset','—')}** · {cfg.get('planes','—')} "
                       f"· {cfg.get('rpm',0):.0f} rpm · {cfg.get('unit','')}")
        return

    # ---- Live: elegir la máquina monitoreada (el 1X se captura en el balanceo) ----
    _live_iid = None
    if src == "Live":
        bal_section_header("Configuración — Monitoreo en línea",
                           "Elige la máquina monitoreada. El vector 1X se captura "
                           "en vivo en el paso de balanceo.", "Paso 2", "●")
        _live_iid, _planes = _machine_and_planes("cfg_live")
        if _live_iid:
            st.session_state["bal_live_iid"] = _live_iid
    else:
        bal_section_header("Configuración de la máquina",
                           "Completa los datos del rotor. Son obligatorios antes "
                           "de balancear.", "Paso 2", "●")

    # ---- Formulario físico (Manual y Live) ----
    with st.form("bal_cfg_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            _asset = st.text_input("Activo / Tag", value=cfg.get("asset", ""))
            _client = st.text_input("Cliente", value=cfg.get("client", ""))
            _specialist = st.text_input(
                "Especialista", value=cfg.get("specialist", _user.get("full_name") or ""))
        with c2:
            _location = st.text_input("Sitio / ubicación", value=cfg.get("location", ""))
            _rpm = st.number_input("RPM de operación",
                                   value=float(cfg.get("rpm") or 3600.0),
                                   min_value=0.0, step=10.0, format="%.0f")
            _planes = st.radio("Planos de balanceo", ["1 plane", "2 planes"],
                               index=(1 if cfg.get("planes") == "2 planes" else 0),
                               horizontal=True)
        with c3:
            _mass = st.number_input("Masa del rotor [kg]",
                                    value=float(cfg.get("rotor_mass") or 0.0),
                                    min_value=0.0, step=10.0, format="%.1f")
            _radius = st.number_input("Radio de balanceo [mm]",
                                      value=float(cfg.get("radius") or 0.0),
                                      min_value=0.0, step=1.0, format="%.1f")
            _k = st.number_input("Factor de prueba k",
                                 value=float(cfg.get("trial_k") or 1.25),
                                 min_value=0.2, max_value=2.0, step=0.05, format="%.2f")
        c4, c5, c6 = st.columns(3)
        with c4:
            _iso_g = st.selectbox("Grado ISO objetivo", _ISO_GRADES_UI,
                                  index=_ISO_GRADES_UI.index(cfg.get("iso_g", "2.5"))
                                  if cfg.get("iso_g", "2.5") in _ISO_GRADES_UI else 2)
        with c5:
            _rot = st.selectbox("Sentido de giro", ["CCW", "CW"],
                                index=(1 if cfg.get("rotation") == "CW" else 0))
        with c6:
            _unit = st.selectbox("Unidad de vibración", UNITS,
                                 index=UNITS.index(cfg.get("unit"))
                                 if cfg.get("unit") in UNITS else 0)
        _ok = st.form_submit_button("Validar configuración y continuar",
                                    type="primary", use_container_width=True)
    if _ok:
        _miss = []
        if not _asset.strip():
            _miss.append("Activo / Tag")
        if _rpm <= 0:
            _miss.append("RPM")
        if _mass <= 0:
            _miss.append("Masa del rotor")
        if _radius <= 0:
            _miss.append("Radio de balanceo")
        if src == "Live" and not st.session_state.get("bal_live_iid"):
            _miss.append("Máquina monitoreada")
        if _miss:
            st.error("Faltan datos obligatorios: " + ", ".join(_miss))
        else:
            _cfg = {"asset": _asset.strip(), "client": _client.strip(),
                    "location": _location.strip(), "specialist": _specialist.strip(),
                    "rpm": float(_rpm), "rotor_mass": float(_mass),
                    "radius": float(_radius), "trial_k": float(_k),
                    "iso_g": _iso_g, "planes": _planes, "rotation": _rot, "unit": _unit}
            st.session_state["bal_cfg"] = _cfg
            st.session_state["bal_src"] = "Live" if src == "Live" else "Manual"
            st.session_state["bal_cfg_ok"] = True
            st.session_state["_pending_nav"] = "Balanceo"
            _apply_cfg(_cfg)
            st.success("Configuración validada. Continúa en **Balanceo**.")
            st.rerun()


# ---------------------------------------------------------------------
# Pasos 1 y 2 — Origen + Configuración
# ---------------------------------------------------------------------
_draft_indicator()

if _active == "Origen":
    _render_origen()

if _active == "Configuración":
    _render_config()


# ---------------------------------------------------------------------
# Peso de prueba (API 684)
# ---------------------------------------------------------------------
if _active == "Peso prueba":
    bal_section_header("Trial weight", "Starting mass to produce a measurable "
                       "vector change.", "API 684 · Umax = 6350·W/N", "⚙️")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            W_plane = _num("tw_wplane", "Plane weight W [kg]", 3500.0,
                           min_value=0.0, step=10.0, format="%.1f")
        with c2:
            rpm = _num("tw_rpm", "Speed N [rpm]", 3600.0,
                       min_value=0.0, step=10.0, format="%.0f")
        with c3:
            radius = _num("tw_radius", "Correction radius [mm]", 420.0,
                          min_value=0.0, step=1.0, format="%.1f")
        with c4:
            k = _num("tw_k", "Factor k (0.2–2.0)", 1.25,
                     min_value=0.2, max_value=2.0, step=0.05, format="%.2f")

    Wtrial, Utrial = recommend_trial_weight_g(W_plane, rpm, radius, k)
    Umax = umax_api684_gmm(W_plane, rpm)
    bal_kpi_row([
        (f"{Wtrial:,.2f} g", "Trial weight", "recommended @ given radius", "cyan"),
        (f"{Umax:,.0f}", "Umax [g·mm]", "API 684", "navy"),
        (f"{Utrial:,.0f}", "Trial U [g·mm]", f"k = {k:.2f}", "navy"),
    ])
    st.caption("Adjust the weight to the available mass/thread and confirm it "
               "produces a measurable vector change before computing the correction.")


# ---------------------------------------------------------------------
# 2) Balanceo en 1 plano
# ---------------------------------------------------------------------
if _active == "Balanceo" and not _two:
    bal_section_header("Single-plane balancing",
                       "H = (Vt − V0) / Wt  ·  Wcorr = −V0 / H",
                       "ISO 21940-12 · influence coefficient", "🎯")
    # Rotor 3D fijo (imagen) SIEMPRE arriba: la vibración medida (V0) aparece
    # apenas se carga el dato; el contrapeso, al calcular.
    _r1prev = st.session_state.get("bal_r1p")
    _v0 = st.session_state.get("b1_v0_mag")
    _vib1 = (_v0, st.session_state.get("b1_v0_ang") or 0.0) if _v0 else None
    _planes1 = build_planes_1p(
        _vib1, st.session_state.get("b1_unit", "µm pk-pk"),
        _r1prev["corr_ang_deg"] if _r1prev else None,
        f"{_r1prev['corr_mass_g']:.1f} g" if _r1prev else "")
    st.markdown(rotor_face_svg(_planes1, rotation=st.session_state.get("b1_rot", "CCW")),
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#64748b'><span style='color:#dc3545'>●</span> Measured vibration (V0) &nbsp;·&nbsp; <span style='color:#2563eb'>●</span> Correction weight to install (appears on calculation)</div>", unsafe_allow_html=True)

    top = st.columns([1, 1])
    with top[0]:
        unit1 = st.selectbox("Vibration unit", UNITS, key="b1_unit")
    with top[1]:
        st.selectbox("Rotation direction", ["CCW", "CW"], key="b1_rot",
                     help="Orients the angular scale against rotation (balancing "
                          "convention). Does not affect the calculation.")
    _bsrc = st.session_state.get("bal_source")
    _source_badge(_bsrc)

    if _bsrc == "Live":
        with st.container(border=True):
            iid, planes = _machine_and_planes("b1_live")
            if planes:
                cpa, cpb = st.columns([2, 1])
                with cpa:
                    p_idx = st.selectbox("Plane to balance", list(range(len(planes))),
                                         format_func=lambda i: _plane_label(planes[i]),
                                         key="b1_live_plane")
                with cpb:
                    direction = st.radio("Direction", ["Y", "X"], key="b1_live_dir",
                                         horizontal=True)
                from core.balance.live_source import pick_sensor_for_plane
                sensor = pick_sensor_for_plane(planes[p_idx], direction)
                st.caption(f"Selected probe: **{sensor or '—'}**  ·  "
                           "capture 1X on each run.")
                b1, b2, b3 = st.columns(3)
                b1.button("Capture V0", key="b1_cap_v0", use_container_width=True,
                          on_click=_bal_capture_cb,
                          args=(iid, [(sensor, "b1_v0_mag", "b1_v0_ang")]))
                b2.button("Capture Vt", key="b1_cap_vt", use_container_width=True,
                          on_click=_bal_capture_cb,
                          args=(iid, [(sensor, "b1_vt_mag", "b1_vt_ang")]))
                b3.button("Capture Vf", key="b1_cap_vf", use_container_width=True,
                          on_click=_bal_capture_cb,
                          args=(iid, [(sensor, "b1_vf_mag", "b1_vf_ang")]))
                if st.session_state.get("_bal_msg"):
                    st.caption(st.session_state["_bal_msg"])

    _trial_weight_suggester("b1_tw", "b1_tw_mag")

    with st.container(border=True):
        colA, colB, colC = st.columns(3)
        with colA:
            v0m, v0a = _vector_inputs("b1_v0", "V0 — initial vibration", unit1)
        with colB:
            twm, twa = _vector_inputs("b1_tw", "Trial weight [g]", "g")
        with colC:
            vtm, vta = _vector_inputs("b1_vt", "Vt — with trial weight", unit1)

    if st.button("Calculate single-plane balancing", key="b1_calc", type="primary"):
        try:
            _res1 = solve_1plane(v0m, v0a, vtm, vta, twm, twa)
            st.session_state["bal_r1p"] = _res1
            st.session_state["bal_src"] = st.session_state.get("bal_source", "Manual")
            _vfm = st.session_state.get("b1_vf_mag")
            st.session_state["bal_r1p_warn"] = diagnose_1plane(
                _res1, v0m, v0a, vtm, vta, twm, twa,
                Vf_mag=_vfm, Vf_ang=st.session_state.get("b1_vf_ang"))
            st.rerun()
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("bal_r1p", None)
            st.session_state.pop("bal_r1p_warn", None)

    r = st.session_state.get("bal_r1p")
    if r:
        st.markdown("")
        bal_kpi_row([
            (f"{r['corr_mass_g']:,.2f} g", "Correction weight", "mass to install", "cyan"),
            (f"{r['corr_ang_deg']:,.1f}°", "Angle", "angular position", "cyan"),
            (f"{r['pred_mag']:,.3f}", f"Residual [{unit1}]", "estimated vibration", "green"),
        ])
        sev, detail, tag = _quality_severity(r["quality"])
        bal_status_banner(f"Model quality: {tag}", f"{detail}. {r['note']}", sev)
        _vfm1 = st.session_state.get("b1_vf_mag")
        _w1 = diagnose_1plane(r, v0m, v0a, vtm, vta, twm, twa,
                              Vf_mag=_vfm1, Vf_ang=st.session_state.get("b1_vf_ang"))
        st.session_state["bal_r1p_warn"] = _w1
        _render_bal_warnings(_w1)
        with st.expander("Validate against final measurement (optional)"):
            vfm, _vfa = _vector_inputs("b1_vf", "Vf — measured final vibration", unit1)
            if vfm > 0:
                _chg = pct_change(v0m, vfm)
                bal_kpi_row([(f"{_chg:+,.1f} %",
                              "Vibration change", "V0 → Vf (− = worse)",
                              "green" if _chg >= 0 else "red")])
                if _chg < 0:
                    bal_status_banner("Final vibration worsened",
                                      f"Vf ({vfm:.3f}) > V0 ({v0m:.3f}). This plane "
                                      "degraded — do not report as improvement.", "fail")


# ---------------------------------------------------------------------
# 3) Balanceo en 2 planos
# ---------------------------------------------------------------------
if _active == "Balanceo" and _two:
    bal_section_header("Two-plane balancing",
                       "2×2 influence coefficient matrix · runs "
                       "0 (initial) · 1 (trial A) · 2 (trial B).",
                       "ISO 21940-12", "🎯")
    # Rotor 3D fijo (imagen) SIEMPRE arriba: vibración inicial (A0/B0) aparece al
    # cargar los datos; los contrapesos, al calcular.
    _r2prev = st.session_state.get("bal_r2p")
    _a0 = st.session_state.get("b2_a0_mag")
    _b0 = st.session_state.get("b2_b0_mag")
    _vibA = (_a0, st.session_state.get("b2_a0_ang") or 0.0) if _a0 else None
    _vibB = (_b0, st.session_state.get("b2_b0_ang") or 0.0) if _b0 else None
    _u2 = st.session_state.get("b2_unit", "µm pk-pk")
    if _r2prev:
        _wam, _waa = to_polar(_r2prev["WA_corr"])
        _wbm, _wba = to_polar(_r2prev["WB_corr"])
        _planes2 = build_planes_2p(_vibA, _vibB, _u2, _waa, f"{_wam:.1f} g",
                                   _wba, f"{_wbm:.1f} g")
    else:
        _planes2 = build_planes_2p(_vibA, _vibB, _u2, None, "", None, "")
    st.markdown(rotor_face_svg(_planes2, rotation=st.session_state.get("b2_rot", "CCW")),
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#64748b'><span style='color:#dc3545'>●</span> Initial vibration (A0/B0) &nbsp;·&nbsp; <span style='color:#2563eb'>●</span> Correction weights (appear on calculation)</div>", unsafe_allow_html=True)

    top = st.columns([1, 1])
    with top[0]:
        unit2 = st.selectbox("Vibration unit", UNITS, key="b2_unit")
    with top[1]:
        st.selectbox("Rotation direction", ["CCW", "CW"], key="b2_rot",
                     help="Orients the angular scale against rotation (balancing "
                          "convention). Does not affect the calculation.")
    _bsrc = st.session_state.get("bal_source")
    _source_badge(_bsrc)

    if _bsrc == "Live":
        with st.container(border=True):
            iid, planes = _machine_and_planes("b2_live")
            if planes and len(planes) >= 2:
                c1, c2, c3 = st.columns(3)
                with c1:
                    ia = st.selectbox("Plane A (coupling side)", list(range(len(planes))),
                                      format_func=lambda i: _plane_label(planes[i]),
                                      key="b2_live_planeA")
                with c2:
                    st.session_state.setdefault("b2_live_planeB", 1 if len(planes) > 1 else 0)
                    ib = st.selectbox("Plane B (free side)", list(range(len(planes))),
                                      format_func=lambda i: _plane_label(planes[i]),
                                      key="b2_live_planeB")
                with c3:
                    direction = st.radio("Direction (both planes)", ["Y", "X"],
                                         key="b2_live_dir", horizontal=True)
                from core.balance.live_source import pick_sensor_for_plane
                sA = pick_sensor_for_plane(planes[ia], direction)
                sB = pick_sensor_for_plane(planes[ib], direction)
                st.caption(f"Probes: A = **{sA or '—'}** · B = **{sB or '—'}** "
                           f"(same direction {direction})")
                b1, b2, b3 = st.columns(3)
                b1.button("Capture run 0 (A0,B0)", key="b2_cap0",
                          use_container_width=True, on_click=_bal_capture_cb,
                          args=(iid, [(sA, "b2_a0_mag", "b2_a0_ang"),
                                      (sB, "b2_b0_mag", "b2_b0_ang")]))
                b2.button("Capture run 1 (A1,B1)", key="b2_cap1",
                          use_container_width=True, on_click=_bal_capture_cb,
                          args=(iid, [(sA, "b2_a1_mag", "b2_a1_ang"),
                                      (sB, "b2_b1_mag", "b2_b1_ang")]))
                b3.button("Capture run 2 (A2,B2)", key="b2_cap2",
                          use_container_width=True, on_click=_bal_capture_cb,
                          args=(iid, [(sA, "b2_a2_mag", "b2_a2_ang"),
                                      (sB, "b2_b2_mag", "b2_b2_ang")]))
                if st.session_state.get("_bal_msg"):
                    st.caption(st.session_state["_bal_msg"])

    with st.container(border=True):
        st.markdown("**Run 0 — initial**")
        c1, c2 = st.columns(2)
        with c1:
            a0m, a0a = _vector_inputs("b2_a0", "A0 — plane A probe", unit2)
        with c2:
            b0m, b0a = _vector_inputs("b2_b0", "B0 — plane B probe", unit2)
    _trial_weight_suggester("b2_wa", "b2_wa_mag", "Suggest trial weight · plane A (API 684)")
    with st.container(border=True):
        st.markdown("**Run 1 — trial weight on plane A**")
        c1, c2, c3 = st.columns(3)
        with c1:
            wam, waa = _vector_inputs("b2_wa", "Trial plane A [g]", "g")
        with c2:
            a1m, a1a = _vector_inputs("b2_a1", "A1 — probe A", unit2)
        with c3:
            b1m, b1a = _vector_inputs("b2_b1", "B1 — probe B", unit2)
    _trial_weight_suggester("b2_wb", "b2_wb_mag", "Suggest trial weight · plane B (API 684)")
    with st.container(border=True):
        st.markdown("**Run 2 — trial weight on plane B**")
        c1, c2, c3 = st.columns(3)
        with c1:
            wbm, wba = _vector_inputs("b2_wb", "Trial plane B [g]", "g")
        with c2:
            a2m, a2a = _vector_inputs("b2_a2", "A2 — probe A", unit2)
        with c3:
            b2m, b2a = _vector_inputs("b2_b2", "B2 — probe B", unit2)

    if st.button("Calculate two-plane balancing", key="b2_calc", type="primary"):
        try:
            _cA0, _cB0 = to_complex(a0m, a0a), to_complex(b0m, b0a)
            _cA1, _cB1 = to_complex(a1m, a1a), to_complex(b1m, b1a)
            _cA2, _cB2 = to_complex(a2m, a2a), to_complex(b2m, b2a)
            _cWA, _cWB = to_complex(wam, waa), to_complex(wbm, wba)
            _res2 = solve_2plane(_cA0, _cB0, _cA1, _cB1, _cA2, _cB2, _cWA, _cWB)
            st.session_state["bal_r2p"] = _res2
            st.session_state["bal_src"] = st.session_state.get("bal_source", "Manual")
            _vfa = st.session_state.get("b2_vfa_mag")
            _vfb = st.session_state.get("b2_vfb_mag")
            st.session_state["bal_r2p_warn"] = diagnose_2plane(
                _res2, _cA0, _cB0, _cA1, _cB1, _cA2, _cB2, _cWA, _cWB,
                Vf_A=to_complex(_vfa, st.session_state.get("b2_vfa_ang") or 0) if _vfa else None,
                Vf_B=to_complex(_vfb, st.session_state.get("b2_vfb_ang") or 0) if _vfb else None)
            st.rerun()
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("bal_r2p", None)
            st.session_state.pop("bal_r2p_warn", None)

    r = st.session_state.get("bal_r2p")
    if r:
        st.markdown("")
        wa_mag, wa_ang = to_polar(r["WA_corr"])
        wb_mag, wb_ang = to_polar(r["WB_corr"])
        bal_kpi_row([
            (f"{wa_mag:,.2f} g", "Plane A correction", f"∠ {wa_ang:,.1f}°", "cyan"),
            (f"{wb_mag:,.2f} g", "Plane B correction", f"∠ {wb_ang:,.1f}°", "cyan"),
            (f"{abs(r['A_after']):,.3f}", f"Residual A [{unit2}]", "estimated", "green"),
            (f"{abs(r['B_after']):,.3f}", f"Residual B [{unit2}]", "estimated", "green"),
        ])
        sev, detail, tag = _quality_severity(r["quality"])
        bal_status_banner(f"Model quality: {tag}",
                          f"{detail}. cond(M) = {r['cond']:.1f}. {r['note']}", sev)
        # Recalcula los avisos con el Vf actual (si ya se midió el final), así
        # "plano empeoró" aparece y entra al reporte sin recalcular el balanceo.
        _vfa = st.session_state.get("b2_vfa_mag")
        _vfb = st.session_state.get("b2_vfb_mag")
        _w2 = diagnose_2plane(
            r, to_complex(a0m, a0a), to_complex(b0m, b0a),
            to_complex(a1m, a1a), to_complex(b1m, b1a),
            to_complex(a2m, a2a), to_complex(b2m, b2a),
            to_complex(wam, waa), to_complex(wbm, wba),
            Vf_A=to_complex(_vfa, st.session_state.get("b2_vfa_ang") or 0) if _vfa else None,
            Vf_B=to_complex(_vfb, st.session_state.get("b2_vfb_ang") or 0) if _vfb else None)
        st.session_state["bal_r2p_warn"] = _w2
        _render_bal_warnings(_w2)
        with st.expander("Validate against final measurement (optional)"):
            cvA, cvB = st.columns(2)
            with cvA:
                _vfam, _ = _vector_inputs("b2_vfa", "Final A — measured", unit2)
            with cvB:
                _vfbm, _ = _vector_inputs("b2_vfb", "Final B — measured", unit2)
            for _lbl, _v0, _vf in (("A", a0m, _vfam), ("B", b0m, _vfbm)):
                if _vf and _vf > 0:
                    _c = pct_change(_v0, _vf)
                    _tone = "ok" if _c >= 0 else "fail"
                    bal_status_banner(
                        f"Plane {_lbl}: {_c:+.0f}%",
                        f"{_v0:.3f} → {_vf:.3f} {unit2}"
                        + ("" if _c >= 0 else "  ·  WORSENED — not an improvement"),
                        _tone)


# ---------------------------------------------------------------------
# 4) Validación ISO 21940-11
# ---------------------------------------------------------------------
if _active == "Validación ISO":
    bal_section_header("ISO validation",
                       "e_per = 9549·G/N  ·  U_per = e_per·W. The residual is "
                       "evaluated against ISO 21940 grades.",
                       "ISO 21940-11", "✅")
    if not _has_result:
        bal_status_banner("Paso bloqueado",
                          "Completa un balanceo (1 plano o 2 planos) para validar "
                          "el residual contra ISO 21940.", "warning")
    else:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            with c1:
                W_iso = _num("iso_w", "Rotor weight W [kg]", 11000.0,
                             min_value=0.0, step=10.0, format="%.1f")
                rpm_iso = _num("iso_rpm", "Speed N [rpm]", 3600.0,
                               min_value=0.0, step=10.0, format="%.0f")
            with c2:
                modo = st.radio("Residual U_res", ["Enter U_res [g·mm]",
                                                   "Compute from mass·radius"], key="iso_mode")
                if modo.startswith("Enter"):
                    U_res = _num("iso_ures", "U_res [g·mm]", 0.0, min_value=0.0,
                                 step=1.0, format="%.1f")
                else:
                    mr = _num("iso_resmass", "Residual mass [g]", 0.0, min_value=0.0,
                              step=0.1, format="%.2f")
                    rr = _num("iso_resrad", "Radius [mm]", 420.0, min_value=0.0,
                              step=1.0, format="%.1f")
                    U_res = calc_U_trial(mr, rr)
                    st.caption(f"U_res computed = **{U_res:,.1f} g·mm**")

        ev = evaluate_iso_grades(W_iso, rpm_iso, U_res)
        st.session_state["bal_iso"] = ev
        if ev["status_code"] == "FAIL":
            bal_status_banner("Does not comply", ev["summary_label"], "fail")
        elif ev["best_grade"] is not None and ev["best_grade"] <= 2.5:
            bal_status_banner("Complies", ev["summary_label"], "ok")
        else:
            bal_status_banner("Complies (basic quality)", ev["summary_label"], "warning")

        # Guard ISO: un grado casi perfecto (≤G1) con la máquina aún vibrando =
        # U_res desacoplado del estado final real → grado sobrestimado.
        _dom_v0 = max([v for v in (_persist_get("b2_b0_mag"), _persist_get("b2_a0_mag"),
                                   _persist_get("b1_v0_mag")) if v], default=0.0)
        _dom_vf = max([v for v in (_persist_get("b2_vfb_mag"), _persist_get("b2_vfa_mag"),
                                   _persist_get("b1_vf_mag")) if v], default=0.0)
        if (ev["best_grade"] is not None and ev["best_grade"] <= 1.0
                and _dom_v0 and _dom_vf and (_dom_vf / _dom_v0) > 0.25):
            bal_status_banner(
                "Grado ISO posiblemente sobrestimado",
                f"El grado G{ev['best_grade']:g} implica un residual casi nulo, "
                f"pero la vibración final medida es {(_dom_vf/_dom_v0)*100:.0f}% de "
                f"la inicial ({_dom_vf:.3f} vs {_dom_v0:.3f}). Verifica que el U_res "
                f"provenga del estado final real, no de un valor manual.", "warning")

        _iso_cols = ["Grade", "e_per [µm]", "U_per [g·mm]", "U_res/U_per", "Cumple"]
        _iso_rows = []
        for g in ev["results"]:
            _iso_rows.append([
                f"G{g['G']:g}",
                round(g["e_per"], 3),
                round(g["U_per"], 1),
                round(g["ratio"], 2) if g["ratio"] < 900 else "—",
                f"{dot('ok')} Sí" if g["pass"] else f"{dot('dang')} No",
            ])
        html_table(_iso_cols, _iso_rows, raw_cols=[4])


# ---------------------------------------------------------------------
# 5) Reporte
# ---------------------------------------------------------------------
if _active == "Reporte":
    bal_section_header("Report", "Session summary and branded "
                       "Watermelon/SIGA PDF.", "ISO 21940 · API 684", "⎙")

    if not _has_result:
        bal_status_banner("Paso bloqueado",
                          "Completa un balanceo (1 plano o 2 planos) antes de "
                          "generar el reporte.", "warning")
        st.stop()

    r1 = st.session_state.get("bal_r1p")
    r2 = st.session_state.get("bal_r2p")
    ev = st.session_state.get("bal_iso")

    lines: List[str] = []
    if r1:
        lines.append(f"1 plane · correction {r1['corr_mass_g']:.2f} g @ "
                     f"{r1['corr_ang_deg']:.1f}° · residual {r1['pred_mag']:.3f} "
                     f"· {r1['quality']}")
    if r2:
        wa_m, wa_a = to_polar(r2["WA_corr"])
        wb_m, wb_a = to_polar(r2["WB_corr"])
        lines.append(f"2 planes · A: {wa_m:.2f} g @ {wa_a:.1f}° · "
                     f"B: {wb_m:.2f} g @ {wb_a:.1f}° · {r2['quality']}")
    if ev:
        lines.append(f"ISO validation · {ev['summary_label']}")

    if lines:
        with st.container(border=True):
            for ln in lines:
                st.markdown(f"- {ln}")
    else:
        st.info("No calculations in this session yet. Run a balancing in the "
                "previous tabs.")

    st.markdown("")
    bal_section_header("Report data")

    _live_iid = st.session_state.get("b2_live_iid") or st.session_state.get("b1_live_iid")
    if _live_iid:
        try:
            from core.instance_state import get_instance as _gi
            _inst = _gi(_live_iid)
            if _inst is not None:
                st.session_state.setdefault("rep_asset", _inst.tag or _live_iid)
                st.session_state.setdefault("rep_client", getattr(_inst, "client", "") or "")
                st.session_state.setdefault(
                    "rep_location",
                    getattr(_inst, "site", "") or getattr(_inst, "location", "") or "")
        except Exception:
            pass

    from datetime import date as _date
    st.session_state.setdefault("rep_asset", "")
    st.session_state.setdefault("rep_client", "")
    st.session_state.setdefault("rep_location", "")
    st.session_state.setdefault("rep_specialist", _user.get("full_name") or "")
    st.session_state.setdefault("rep_date", _date.today().strftime("%d/%m/%Y"))

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            rep_asset = st.text_input("Asset", key="rep_asset")
            rep_client = st.text_input("Client", key="rep_client")
        with c2:
            rep_location = st.text_input("Site / location", key="rep_location")
            rep_specialist = st.text_input("Specialist", key="rep_specialist")
        with c3:
            rep_date = st.text_input("Date", key="rep_date")
            rep_notes = st.text_area("Notes", key="rep_notes", height=80)

    def _pair(prefix: str):
        m = _persist_get(f"{prefix}_mag")
        if m is None:
            return None
        return (float(m), float(_persist_get(f"{prefix}_ang") or 0.0))

    one_plane = None
    if r1:
        vf = _pair("b1_vf")
        vf = vf if (vf and vf[0] > 0) else None
        one_plane = {"unit": st.session_state.get("b1_unit", "µm pk-pk"),
                     "v0": _pair("b1_v0"), "trial": _pair("b1_tw"),
                     "vt": _pair("b1_vt"), "vf": vf, "result": r1,
                     "warnings": st.session_state.get("bal_r1p_warn") or []}
    two_plane = None
    if r2:
        two_plane = {"unit": st.session_state.get("b2_unit", "µm pk-pk"),
                     "a0": _pair("b2_a0"), "b0": _pair("b2_b0"),
                     "a1": _pair("b2_a1"), "b1": _pair("b2_b1"),
                     "a2": _pair("b2_a2"), "b2": _pair("b2_b2"),
                     "wa": _pair("b2_wa"), "wb": _pair("b2_wb"), "result": r2,
                     "vf_a": _pair("b2_vfa"), "vf_b": _pair("b2_vfb"),
                     "warnings": st.session_state.get("bal_r2p_warn") or []}

    if not (one_plane or two_plane or ev):
        st.info("Run at least one balancing or ISO validation to generate the PDF.")
    else:
        if st.button("Generate PDF report", key="rep_pdf", type="primary"):
            try:
                from core.balance.report import build_balance_pdf
                meta = {
                    "asset": rep_asset, "client": rep_client, "location": rep_location,
                    "specialist": rep_specialist, "report_date": rep_date,
                    "unit": (st.session_state.get("b1_unit")
                             or st.session_state.get("b2_unit") or "µm pk-pk"),
                    "rpm": (st.session_state.get("iso_rpm")
                            or st.session_state.get("tw_rpm")),
                    "rotation": (st.session_state.get("b1_rot")
                                 or st.session_state.get("b2_rot") or "CCW"),
                    "data_source": _source_label(st.session_state.get("bal_src", "Manual")),
                    "notes": rep_notes,
                }
                st.session_state["bal_pdf"] = build_balance_pdf(
                    meta=meta, one_plane=one_plane, two_plane=two_plane, iso=ev)
            except Exception as e:  # noqa: BLE001
                st.error(f"Error generating the PDF: {e}")
                st.session_state.pop("bal_pdf", None)
        if st.session_state.get("bal_pdf"):
            import re as _re
            _fn = "Balanceo_" + _re.sub(r"[^A-Za-z0-9]+", "_",
                                        (rep_asset or "activo")).strip("_") + ".pdf"
            st.download_button("Download PDF", data=st.session_state["bal_pdf"],
                               file_name=_fn, mime="application/pdf")


# Auto-guardado silencioso del borrador (al final, tras renderizar todo el paso).
try:
    _maybe_autosave()
except Exception:  # noqa: BLE001
    pass

bal_footer_norms()
