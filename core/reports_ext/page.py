"""
core.reports_ext.page — UI (Streamlit) de los reportes de campo.

`render_report_family(family)` dibuja el formulario del tipo elegido
(diario/preliminar/boroscopia/alineacion/mecanico): metadatos con auto-fill
editable desde el activo activo, contenido, carga de fotos, y genera + descarga
el PDF con el formato SIGA. La página 16_Reports.py llama a esta función dentro
del branch que hace st.stop() (dejando intacto el reporte del sistema).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.reports_ext.builders import BUILDERS
from core.reports_ext.common import (
    autofill_base_meta, today_str, REVIEWERS, peek_consecutive, commit_consecutive,
)
from core.reports_ext.ui import rep_section_header, rep_status_banner


def _current_user_name() -> str:
    """Nombre del especialista logueado (aparece automáticamente)."""
    try:
        from core.auth import get_current_user
        u = get_current_user() or {}
        return (u.get("full_name") or u.get("name") or u.get("email") or "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------
# Parsers de texto → estructuras
# ---------------------------------------------------------------------
def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def _parse_plan(text: str) -> List[Dict[str, Any]]:
    """Bloques de plan: una línea sin viñeta = título de sección; líneas con
    '-' o '•' = sub-ítems de la última sección."""
    sections: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = None
    for ln in _lines(text):
        if ln[0] in "-•*":
            if cur is None:
                cur = {"title": "", "items": []}
                sections.append(cur)
            cur["items"].append(ln.lstrip("-•* ").strip())
        else:
            cur = {"title": ln, "items": []}
            sections.append(cur)
    return sections


# ---------------------------------------------------------------------
# Metadatos (auto-fill editable)
# ---------------------------------------------------------------------
def _meta_form(prefix: str) -> Dict[str, Any]:
    if not st.session_state.get(f"{prefix}_autofilled"):
        base = autofill_base_meta()
        for k in ("client", "plant", "location", "equipo"):
            st.session_state.setdefault(f"{prefix}_{k}", base.get(k, ""))
        st.session_state.setdefault(f"{prefix}_train", base.get("train_description", ""))
        # especialista = usuario logueado (automático, editable)
        st.session_state.setdefault(f"{prefix}_spec", _current_user_name())
        # consecutivo automático ISO 9001 (editable)
        st.session_state.setdefault(f"{prefix}_cons", peek_consecutive(prefix))
        st.session_state[f"{prefix}_autofilled"] = True

    with st.container(border=True):
        st.caption("Datos del servicio · auto-rellenados desde el activo activo "
                   "y el usuario logueado (editables).")
        c1, c2, c3 = st.columns(3)
        with c1:
            client = st.text_input("Cliente", key=f"{prefix}_client")
            plant = st.text_input("Planta", key=f"{prefix}_plant")
        with c2:
            location = st.text_input("Ubicación", key=f"{prefix}_location")
            equipo = st.text_input("Equipo", key=f"{prefix}_equipo")
        with c3:
            report_date = st.text_input("Fecha", value=today_str(), key=f"{prefix}_date")
            consecutive = st.text_input(
                "Consecutivo (automático · ISO 9001)", key=f"{prefix}_cons",
                help="Código SIGA-TIPO-AÑO-NNNN generado automáticamente. "
                     "Editable si necesitas empatar tu numeración.")
        c4, c5 = st.columns(2)
        with c4:
            specialist = st.text_input("Preparado por (especialista)",
                                       key=f"{prefix}_spec",
                                       help="Se rellena con el usuario logueado.")
        with c5:
            reviewer = st.selectbox("Revisado por (autoridad)",
                                    list(REVIEWERS.keys()), key=f"{prefix}_rev")

    return {
        "client": client, "plant": plant, "location": location, "equipo": equipo,
        "report_date": report_date, "consecutive": consecutive,
        "specialist": specialist,
        "reviewer": reviewer, "reviewer_role": REVIEWERS.get(reviewer, ""),
        "train_description": st.session_state.get(f"{prefix}_train", ""),
    }


def _photo_uploader(prefix: str, label: str = "Registro fotográfico") -> List[Dict[str, Any]]:
    files = st.file_uploader(label, accept_multiple_files=True,
                             type=["png", "jpg", "jpeg"], key=f"{prefix}_photos")
    files = files or []
    photos: List[Dict[str, Any]] = []
    if files:
        st.caption(f"{len(files)} imagen(es). Escribe un título por figura; se "
                   "numeran automáticamente en orden (Figura 1, 2, 3…).")
        with st.expander(f"Títulos de las {len(files)} figuras", expanded=True):
            for i, f in enumerate(files, 1):
                title = st.text_input(f"Figura {i}", key=f"{prefix}_figt_{i}",
                                      placeholder=f"Descripción de la figura {i}")
                cap = f"Figura {i}. {title}".rstrip(". ") if title else f"Figura {i}"
                photos.append({"bytes": f.getvalue(), "caption": cap})
    return photos


def _commit_consecutive_once(prefix: str, family: str, cons: str) -> None:
    """Incrementa el contador solo si el consecutivo mostrado es el automático
    aún no confirmado (evita gastar números en recargas o regeneraciones)."""
    done = st.session_state.setdefault(f"{prefix}_cons_committed", set())
    if cons and cons.startswith("SIGA-") and cons not in done:
        commit_consecutive(family)
        done.add(cons)


def _generate(prefix: str, family: str, meta: Dict[str, Any], content: Dict[str, Any]):
    if st.button("Generar reporte PDF", type="primary", key=f"{prefix}_gen"):
        try:
            st.session_state[f"{prefix}_pdf"] = BUILDERS[family](meta=meta, content=content)
            _commit_consecutive_once(prefix, family, meta.get("consecutive", ""))
        except Exception as e:  # noqa: BLE001
            st.error(f"Error generando el PDF: {e}")
            st.session_state.pop(f"{prefix}_pdf", None)
    if st.session_state.get(f"{prefix}_pdf"):
        fn = f"{family}_" + re.sub(r"[^A-Za-z0-9]+", "_",
                                   (meta.get("equipo") or "reporte")).strip("_") + ".pdf"
        st.download_button("⬇ Descargar PDF", data=st.session_state[f"{prefix}_pdf"],
                           file_name=fn, mime="application/pdf", key=f"{prefix}_dl")


# ---------------------------------------------------------------------
# Formularios por familia
# ---------------------------------------------------------------------
def _daily(meta):
    rep_section_header("Reporte Diario", "Datos del servicio · hallazgos · plan de trabajo",
                       "SIGA-FMT-136")
    servicio = st.text_input("Servicio", key="diario_servicio",
                             placeholder="Revisión Bomba FLOWSERVE CPVX")
    c1, c2 = st.columns(2)
    hall = c1.text_area("Hallazgos (uno por línea)", key="diario_hall", height=120)
    obs = c2.text_area("Observaciones (una por línea)", key="diario_obs", height=120)
    st.markdown("**Plan de trabajo** — título de sección en su línea; sub-ítems con «- ».")
    plan = st.text_area("Plan de trabajo", key="diario_plan", height=160,
                        placeholder="Recepción y preparación\n- Recepción en taller\n- Limpieza inicial\nVerificación dimensional\n- Run-out en bujes")
    photos = _photo_uploader("diario")
    content = {"servicio": servicio, "hallazgos": _lines(hall),
               "observaciones": _lines(obs), "plan": _parse_plan(plan), "photos": photos}
    _generate("diario", "diario", meta, content)


def _preliminary(meta):
    rep_section_header("Reporte Preliminar", "Objeto · hallazgos preliminares · recomendaciones",
                       "SIGA-FMT-PRE")
    objeto = st.text_area("Objeto y alcance", key="prel_objeto", height=80)
    resumen = st.text_area("Resumen preliminar", key="prel_resumen", height=80)
    c1, c2 = st.columns(2)
    hall = c1.text_area("Hallazgos preliminares (uno por línea)", key="prel_hall", height=120)
    obs = c2.text_area("Observaciones (una por línea)", key="prel_obs", height=120)
    reco = st.text_area("Recomendaciones (una por línea)", key="prel_reco", height=100)
    photos = _photo_uploader("prel")
    content = {"objeto": objeto, "resumen": resumen, "hallazgos": _lines(hall),
               "observaciones": _lines(obs), "recomendaciones": _lines(reco), "photos": photos}
    _generate("prel", "preliminar", meta, content)


def _borescope(meta):
    rep_section_header("Inspección Boroscópica",
                       "Introducción · hallazgos · severidad por acceso · evidencias",
                       "SIGA-FMT-178")
    intro = st.text_area("1. Introducción y alcance", key="boro_intro", height=80)
    ante = st.text_area("2. Antecedentes", key="boro_ante", height=70)
    c1, c2 = st.columns(2)
    hall = c1.text_area("3. Hallazgos (uno por línea)", key="boro_hall", height=110)
    reco = c2.text_area("4. Recomendaciones (una por línea)", key="boro_reco", height=110)
    metod = st.text_area("5. Metodología (opcional)", key="boro_metod", height=70,
                         placeholder="Equipo Olympus Iplex LX/LT, sonda 6 mm, 2 m...")

    st.markdown("**6. Desarrollo — hallazgos por acceso (nivel de severidad):**")
    default = pd.DataFrame({
        "Acceso/Ubicación": ["", ""], "Hallazgos": ["", ""],
        "Severidad": ["Serviciable", "Serviciable"], "Comentarios": ["", ""]})
    edited = st.data_editor(
        st.session_state.get("boro_sev_df", default), num_rows="dynamic",
        use_container_width=True, key="boro_sev_editor",
        column_config={"Severidad": st.column_config.SelectboxColumn(
            options=["Serviciable", "No operativo"])})
    evid = st.file_uploader("Evidencias (en orden de fila)", accept_multiple_files=True,
                            type=["png", "jpg", "jpeg"], key="boro_evid")
    ev_bytes = [f.getvalue() for f in (evid or [])]

    rows = []
    for i, (_, r) in enumerate(edited.iterrows()):
        if not (str(r.get("Acceso/Ubicación", "")).strip()
                or str(r.get("Hallazgos", "")).strip()):
            continue
        row = {"access": r.get("Acceso/Ubicación", ""), "findings": r.get("Hallazgos", ""),
               "severity": r.get("Severidad", ""), "comment": r.get("Comentarios", "")}
        if i < len(ev_bytes):
            row["image_bytes"] = ev_bytes[i]
        rows.append(row)

    content = {"introduccion": intro, "antecedentes": ante, "hallazgos": _lines(hall),
               "recomendaciones": _lines(reco), "metodologia": metod, "severity_rows": rows}
    _generate("boro", "boroscopia", meta, content)


def _alignment(meta):
    rep_section_header("Reporte de Alineación", "As found / As left · tolerancias · shims",
                       "SIGA-FMT-ALI")
    metodo = st.text_area("Método y alcance", key="ali_metodo", height=70,
                          placeholder="Alineación láser doble haz; estacionaria → móvil.")
    st.markdown("**Condición encontrada / dejada:**")
    default = pd.DataFrame({
        "Parámetro": ["Offset vertical", "Offset horizontal",
                      "Angularidad vertical", "Angularidad horizontal"],
        "As found": ["", "", "", ""], "As left": ["", "", "", ""],
        "Tolerancia": ["", "", "", ""], "Estado": ["", "", "", ""]})
    edited = st.data_editor(st.session_state.get("ali_df", default), num_rows="dynamic",
                            use_container_width=True, key="ali_editor")
    rows = [[r["Parámetro"], r["As found"], r["As left"], r["Tolerancia"], r["Estado"]]
            for _, r in edited.iterrows() if str(r["Parámetro"]).strip()]
    shims = st.text_area("Correcciones (shims / movimientos)", key="ali_shims", height=70)
    c1, c2 = st.columns(2)
    hall = c1.text_area("Hallazgos (uno por línea)", key="ali_hall", height=90)
    reco = c2.text_area("Recomendaciones (una por línea)", key="ali_reco", height=90)
    photos = _photo_uploader("ali")
    content = {"metodo": metodo, "align_rows": rows, "shims": shims,
               "hallazgos": _lines(hall), "recomendaciones": _lines(reco), "photos": photos}
    _generate("ali", "alineacion", meta, content)


def _mechanical(meta):
    rep_section_header("Reporte Mecánico", "Actividades · metrología · hallazgos",
                       "SIGA-FMT-MEC")
    objeto = st.text_area("Objeto y alcance", key="mec_objeto", height=70)
    st.markdown("**Actividades ejecutadas** — título en su línea; sub-ítems con «- ».")
    acts = st.text_area("Actividades", key="mec_acts", height=130,
                        placeholder="Desmontaje\n- Retiro de acople\n- Extracción de rodamientos")
    st.markdown("**Mediciones / Metrología:**")
    default = pd.DataFrame({"Parámetro": ["", ""], "Valor": ["", ""], "Unidad": ["", ""],
                            "Referencia": ["", ""], "Estado": ["", ""]})
    edited = st.data_editor(st.session_state.get("mec_df", default), num_rows="dynamic",
                            use_container_width=True, key="mec_editor")
    metro = [[r["Parámetro"], r["Valor"], r["Unidad"], r["Referencia"], r["Estado"]]
             for _, r in edited.iterrows() if str(r["Parámetro"]).strip()]
    c1, c2 = st.columns(2)
    hall = c1.text_area("Hallazgos (uno por línea)", key="mec_hall", height=90)
    reco = c2.text_area("Recomendaciones (una por línea)", key="mec_reco", height=90)
    obs = st.text_area("Observaciones (una por línea)", key="mec_obs", height=70)
    photos = _photo_uploader("mec")
    content = {"objeto": objeto, "actividades": _parse_plan(acts), "metrologia_rows": metro,
               "hallazgos": _lines(hall), "observaciones": _lines(obs),
               "recomendaciones": _lines(reco), "photos": photos}
    _generate("mec", "mecanico", meta, content)


_FORMS = {
    "diario": _daily, "preliminar": _preliminary, "boroscopia": _borescope,
    "alineacion": _alignment, "mecanico": _mechanical,
}


def render_report_family(family: str) -> None:
    """Renderiza el formulario + generación PDF de la familia elegida."""
    form = _FORMS.get(family)
    if form is None:
        st.error(f"Tipo de reporte desconocido: {family}")
        return
    meta = _meta_form(family)
    st.divider()
    form(meta)


__all__ = ["render_report_family"]
