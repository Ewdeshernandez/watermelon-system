"""
core.reports_ext.page — UI (Streamlit) de los reportes de campo.

`render_report_family(family)` dibuja el formulario del tipo elegido
(diario/preliminar/boroscopia/alineacion/mecanico): metadatos con auto-fill
editable desde el activo activo, contenido, carga de fotos, y genera + descarga
el PDF con el formato SIGA. La página 16_Reports.py llama a esta función dentro
del branch que hace st.stop() (dejando intacto el reporte del sistema).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.reports_ext.builders import BUILDERS
from core.reports_ext.common import (
    autofill_base_meta, today_str, REVIEWERS, peek_consecutive, commit_consecutive,
)
from core.reports_ext.ui import rep_section_header, rep_status_banner
from core.reports_ext import drafts as _drafts


# =====================================================================
# Autoguardado + borradores (no perder el reporte si se cae la sesión)
# =====================================================================
# Prefijos de session_state por familia (metadatos + contenido).
_FAMILY_PREFIXES = {
    "diario": ["diario"],
    "preliminar": ["preliminar", "prel"],
    "boroscopia": ["boroscopia", "boro"],
    "alineacion": ["alineacion", "ali"],
    "mecanico": ["mecanico", "mec"],
}
# Subcadenas de llaves que NO se persisten NI se restauran. Incluye bytes/
# uploaders/editores/PDF y BOTONES (Streamlit no permite setear su valor por
# session_state — restaurarlos crashea con StreamlitValueAssignmentNotAllowedError).
_SKIP_SUBSTR = ("_editor", "_pdf", "_photos", "_evid", "figt", "_draft",
                "_pick", "_gen", "_dl", "_meth", "_img", "_fig_",
                "_add", "_up_", "_dn_", "_del", "eqdel", "_recover", "_new",
                "_bidc", "_eidc")


def _rep_module(family: str) -> str:
    return f"campo_{family}"


def _family_keys(family: str) -> List[str]:
    prefs = tuple(p + "_" for p in _FAMILY_PREFIXES.get(family, [family]))
    return [k for k in list(st.session_state.keys()) if k.startswith(prefs)]


def _capture_family_state(family: str) -> Dict[str, Any]:
    prefs = tuple(p + "_" for p in _FAMILY_PREFIXES.get(family, [family]))
    out: Dict[str, Any] = {}
    for k in list(st.session_state.keys()):
        if not k.startswith(prefs):
            continue
        if any(s in k for s in _SKIP_SUBSTR):
            continue
        v = _drafts._strip_bytes(st.session_state.get(k))  # quita bytes de imágenes
        if not (isinstance(v, (str, int, float, bool, list, dict)) or v is None):
            continue
        try:
            json.dumps(v)
        except Exception:
            continue
        out[k] = v
    return out


def _apply_pending(family: str) -> None:
    """Aplica una restauración/limpieza PENDIENTE antes de instanciar widgets."""
    pend = st.session_state.pop(f"_rep_pending_{family}", None)
    if pend is None:
        return
    if pend.get("__clear__"):
        for k in _family_keys(family):
            if any(s in k for s in ("_draft", "_pick")):
                continue
            try:
                del st.session_state[k]
            except Exception:
                pass
        return
    for k, v in pend.items():
        if k == "__clear__":
            continue
        # No restaurar llaves de widgets no-seteables (botones/uploaders/editores);
        # setearlas lanza StreamlitValueAssignmentNotAllowedError.
        if any(s in k for s in _SKIP_SUBSTR):
            continue
        try:
            st.session_state[k] = v
        except Exception:
            pass


def _draft_bar(family: str) -> None:
    module = _rep_module(family)
    with st.expander("💾 Borradores del reporte", expanded=False):
        st.caption("El reporte se autoguarda solo; si se cae la sesión o hay "
                   "redeploy, entra a «Recuperar autoguardado». También puedes "
                   "guardar versiones con nombre. (Las fotos se re-suben.)")
        r0 = st.columns([1, 1, 2])
        if r0[0].button("♻ Recuperar autoguardado", key=f"{module}_recover"):
            _auto = _drafts.load_autosave(module)
            if _auto:
                st.session_state[f"_rep_pending_{family}"] = _auto
                st.rerun()
            else:
                st.warning("No hay autoguardado para este reporte.")
        if r0[1].button("🆕 Nuevo (limpiar)", key=f"{module}_new"):
            st.session_state[f"_rep_pending_{family}"] = {"__clear__": True}
            st.rerun()
        d1, d2 = st.columns([3, 1])
        _dname = d1.text_input("Nombre del borrador", key=f"{module}_draft_name",
                               placeholder="ej: SGT-300 B — boroscopia")
        if d2.button("💾 Guardar", key=f"{module}_draft_save", use_container_width=True):
            _nm = (_dname or "").strip() or "borrador"
            if _drafts.save_draft(module, _nm, _capture_family_state(family)):
                st.success(f"Borrador «{_nm}» guardado.")
            else:
                st.error("No se pudo guardar (¿disco lleno?).")
        _existing = _drafts.list_drafts(module)
        e1, e2, e3, e4 = st.columns([3, 1, 1, 1])
        _sel = e1.selectbox("Borradores existentes", ["—"] + _existing,
                            key=f"{module}_draft_pick")
        _has = _sel != "—"
        if e2.button("Cargar", key=f"{module}_draft_load", use_container_width=True,
                     disabled=not _has):
            _stt = _drafts.load_draft(module, _sel)
            if _stt is not None:
                st.session_state[f"_rep_pending_{family}"] = _stt
                st.rerun()
        if e3.button("Duplicar", key=f"{module}_draft_dup", use_container_width=True,
                     disabled=not _has):
            _stt = _drafts.load_draft(module, _sel)
            if _stt is not None:
                _drafts.save_draft(module, f"{_sel} (copia)", _stt)
                st.rerun()
        if e4.button("Eliminar", key=f"{module}_draft_del", use_container_width=True,
                     disabled=not _has):
            _drafts.delete_draft(module, _sel)
            st.rerun()


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
    if cons and cons.startswith("SIGAGROUP-") and cons not in done:
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
    rep_section_header("Reporte Diario",
                       "Datos del servicio · actividades realizadas · hallazgos · observaciones",
                       "SIGA-FMT-136")
    servicio = st.text_input("Servicio", key="diario_servicio",
                             placeholder="Revisión Bomba FLOWSERVE CPVX")
    st.markdown("**Actividades realizadas** — título de sección en su línea; sub-ítems con «- ».")
    plan = st.text_area("Actividades realizadas", key="diario_plan", height=160,
                        placeholder="Recepción y preparación\n- Recepción en taller\n- Limpieza inicial\nVerificación dimensional\n- Run-out en bujes")
    c1, c2 = st.columns(2)
    hall = c1.text_area("Hallazgos (uno por línea)", key="diario_hall", height=120)
    obs = c2.text_area("Observaciones (una por línea)", key="diario_obs", height=120)
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
                       "Introducción · hallazgos · info de máquina · severidad por acceso",
                       "SIGA-FMT-178")
    intro = st.text_area("1. Introducción y alcance", key="boro_intro", height=80)
    ante = st.text_area("2. Antecedentes", key="boro_ante", height=70)
    c1, c2 = st.columns(2)
    hall = c1.text_area("3. Hallazgos (uno por línea → numerados)", key="boro_hall", height=110)
    reco = c2.text_area("4. Recomendaciones (una por línea → numeradas)", key="boro_reco", height=110)
    metod = st.text_area("5. Metodología (opcional)", key="boro_metod", height=70,
                         placeholder="Equipo Olympus Iplex LX/LT, sonda 6 mm, 2 m...")

    # --- Info de la máquina (Tabla 1) ---
    with st.expander("Información de la máquina y del boroscopio (Tabla 1)", expanded=False):
        mc = st.columns(3)
        fab = mc[0].text_input("Fabricante", key="boro_fab")
        mod = mc[1].text_input("Modelo", key="boro_mod")
        ser = mc[2].text_input("Serie", key="boro_ser")
        mc2 = st.columns(2)
        hrs = mc2[0].text_input("Horas", key="boro_hrs")
        arr = mc2[1].text_input("Arranques", key="boro_arr")
        bc = st.columns(2)
        bmarca = bc[0].text_input("Boroscopio — marca", value="Olympus", key="boro_bmarca")
        bmodelo = bc[1].text_input("Boroscopio — modelo", key="boro_bmodelo")
    machine_info = {k: v for k, v in {"fabricante": fab, "modelo": mod, "serie": ser,
                                      "horas": hrs, "arranques": arr}.items() if v.strip()}
    borescope_info = {k: v for k, v in {"marca": bmarca, "modelo": bmodelo}.items() if v.strip()}

    # --- Imágenes del equipo / metodología ---
    with st.expander("Imágenes del equipo / metodología (boroscopio, turbina, puntos)", expanded=False):
        meth_imgs = _photo_uploader("boro_meth", label="Imágenes de metodología")

    # --- Tabla de puntos de inspección y estado ---
    with st.expander("Puntos de inspección y estado (tabla)", expanded=False):
        insp_default = pd.DataFrame({"Ubicación": ["", ""], "Punto": ["", ""],
                                     "Estado": ["Serviciable", "Serviciable"]})
        insp_ed = st.data_editor(insp_default, num_rows="dynamic", use_container_width=True,
                                 key="boro_insp_editor",
                                 column_config={"Estado": st.column_config.SelectboxColumn(
                                     options=["Serviciable", "No serviciable", "-"])})
        inspection_rows = [{"ubicacion": r["Ubicación"], "punto": r["Punto"], "estado": r["Estado"]}
                           for _, r in insp_ed.iterrows()
                           if str(r["Ubicación"]).strip() or str(r["Punto"]).strip()]

    # --- 6. Desarrollo: accesos por severidad + MÚLTIPLES imágenes por acceso ---
    st.markdown("**6. Desarrollo — hallazgos por acceso (nivel de severidad):**")
    default = pd.DataFrame({
        "Acceso/Ubicación": ["", ""], "Hallazgos": ["", ""],
        "Severidad": ["Serviciable", "Serviciable"], "Comentarios": ["", ""]})
    edited = st.data_editor(default, num_rows="dynamic", use_container_width=True,
                            key="boro_sev_editor",
                            column_config={"Severidad": st.column_config.SelectboxColumn(
                                options=["Serviciable", "No serviciable"])})
    rows = []
    _accesos = [(i, r) for i, (_, r) in enumerate(edited.iterrows())
                if str(r.get("Acceso/Ubicación", "")).strip()
                or str(r.get("Hallazgos", "")).strip()]
    if _accesos:
        st.caption("Sube una o varias imágenes por cada acceso:")
    for i, r in _accesos:
        acc = str(r.get("Acceso/Ubicación", "")).strip() or f"acceso {i + 1}"
        row = {"access": r.get("Acceso/Ubicación", ""), "findings": r.get("Hallazgos", ""),
               "severity": r.get("Severidad", ""), "comment": r.get("Comentarios", "")}
        files = st.file_uploader(f"Imágenes · {acc}", accept_multiple_files=True,
                                 type=["png", "jpg", "jpeg"], key=f"boro_imgs_{i}")
        if files:
            row["images"] = [f.getvalue() for f in files]
        rows.append(row)

    content = {"introduccion": intro, "antecedentes": ante, "hallazgos": _lines(hall),
               "recomendaciones": _lines(reco), "metodologia": metod,
               "machine_info": machine_info, "borescope_info": borescope_info,
               "methodology_images": meth_imgs, "inspection_rows": inspection_rows,
               "severity_rows": rows}
    _generate("boro", "boroscopia", meta, content)


def _equipos_composer(prefix: str) -> List[Dict[str, Any]]:
    """Tablas de equipo (Campo/Valor) — conductor, conducido, alineador, etc.
    Se pueden agregar/quitar. Devuelve [{title, rows:[[campo,valor]]}]."""
    key = f"{prefix}_equipos"
    eqs = st.session_state.setdefault(key, [])
    st.session_state.setdefault(f"{prefix}_eidc", 0)
    if st.button("➕ Agregar tabla de equipo", key=f"{prefix}_addeq"):
        st.session_state[f"{prefix}_eidc"] += 1
        eqs.append({"id": st.session_state[f"{prefix}_eidc"], "title": "", "rows": []})
        st.rerun()
    for i, eq in enumerate(list(eqs)):
        eid = eq["id"]
        with st.container(border=True):
            hc = st.columns([6, 1])
            eq["title"] = hc[0].text_input(
                "Título de la tabla", value=eq.get("title", ""),
                key=f"{prefix}_eqttl_{eid}",
                placeholder="ej: Información del conductor / conducido / alineador")
            if hc[1].button("🗑", key=f"{prefix}_eqdel_{eid}"):
                eqs.pop(i); st.rerun()
            # Base ESTABLE en sesión (init una vez desde lo guardado/default).
            # No se re-siembra con lo editado → evita que se borre al escribir.
            _dfk = f"{prefix}_eqdf_{eid}"
            if _dfk not in st.session_state:
                _seed = eq.get("_df") or {"Campo": ["Fabricante", "Modelo", "Serial"],
                                          "Valor": ["", "", ""]}
                st.session_state[_dfk] = pd.DataFrame(_seed)
            ed = st.data_editor(st.session_state[_dfk], num_rows="dynamic",
                                use_container_width=True, key=f"{prefix}_eqtbl_editor_{eid}")
            eq["_df"] = {"Campo": [str(x) for x in ed["Campo"].tolist()],
                         "Valor": [str(x) for x in ed["Valor"].tolist()]}
            eq["rows"] = [[str(ed.iloc[r]["Campo"]), str(ed.iloc[r]["Valor"])]
                          for r in range(len(ed))]
    return eqs


def _free_block_composer(prefix: str) -> List[Dict[str, Any]]:
    """Compositor de ORDEN LIBRE: bloques de texto / imágenes / tabla que el
    usuario agrega, reordena (↑/↓) y borra. Devuelve la lista en orden."""
    key = f"{prefix}_blocks"
    blocks = st.session_state.setdefault(key, [])
    st.session_state.setdefault(f"{prefix}_bidc", 0)

    def _nid():
        st.session_state[f"{prefix}_bidc"] += 1
        return st.session_state[f"{prefix}_bidc"]

    ca = st.columns(3)
    if ca[0].button("➕ Texto", key=f"{prefix}_addtxt", use_container_width=True):
        blocks.append({"id": _nid(), "type": "text", "text": ""}); st.rerun()
    if ca[1].button("➕ Imágenes", key=f"{prefix}_addimg", use_container_width=True):
        blocks.append({"id": _nid(), "type": "images"}); st.rerun()
    if ca[2].button("➕ Tabla", key=f"{prefix}_addtbl", use_container_width=True):
        blocks.append({"id": _nid(), "type": "table", "cols": 3}); st.rerun()

    for i, b in enumerate(list(blocks)):
        bid = b["id"]
        _tlabel = {"text": "Texto", "images": "Imágenes", "table": "Tabla"}.get(b["type"], b["type"])
        with st.container(border=True):
            hc = st.columns([6, 1, 1, 1])
            hc[0].markdown(f"**Bloque {i + 1} · {_tlabel}**")
            if hc[1].button("↑", key=f"{prefix}_up_{bid}") and i > 0:
                blocks[i - 1], blocks[i] = blocks[i], blocks[i - 1]; st.rerun()
            if hc[2].button("↓", key=f"{prefix}_dn_{bid}") and i < len(blocks) - 1:
                blocks[i + 1], blocks[i] = blocks[i], blocks[i + 1]; st.rerun()
            if hc[3].button("🗑", key=f"{prefix}_delb_{bid}"):
                blocks.pop(i); st.rerun()

            if b["type"] == "text":
                b["text"] = st.text_area("Texto", value=b.get("text", ""),
                                         key=f"{prefix}_btxt_{bid}", height=100)
            elif b["type"] == "images":
                files = st.file_uploader("Imágenes", accept_multiple_files=True,
                                         type=["png", "jpg", "jpeg"],
                                         key=f"{prefix}_bimg_{bid}")
                phs = []
                for j, f in enumerate(files or [], 1):
                    _t = st.text_input(f"Figura {j}", key=f"{prefix}_bfig_{bid}_{j}",
                                       placeholder=f"Descripción de la figura {j}")
                    _cap = f"Figura {j}. {_t}".rstrip(". ") if _t else f"Figura {j}"
                    phs.append({"bytes": f.getvalue(), "caption": _cap})
                b["photos"] = phs
            elif b["type"] == "table":
                b["title"] = st.text_input("Título de la tabla", value=b.get("title", ""),
                                           key=f"{prefix}_bttl_{bid}")
                ncol = int(st.number_input("Columnas", min_value=2, max_value=6,
                                           value=int(b.get("cols", 3)),
                                           key=f"{prefix}_bnc_{bid}"))
                b["cols"] = ncol
                cols = [f"C{c + 1}" for c in range(ncol)]
                _dfk = f"{prefix}_bdf_{bid}_{ncol}"
                if _dfk not in st.session_state:
                    seed = b.get("_df") or {}
                    data = {c: (seed.get(c) or ["", ""]) for c in cols}
                    st.session_state[_dfk] = pd.DataFrame(data)
                ed = st.data_editor(st.session_state[_dfk], num_rows="dynamic",
                                    use_container_width=True,
                                    key=f"{prefix}_btbl_editor_{bid}_{ncol}")
                b["_df"] = {c: [str(x) for x in ed[c].tolist()] for c in cols}
                recs = [[str(ed.iloc[r][c]) for c in cols] for r in range(len(ed))]
                b["headers"] = recs[0] if recs else cols
                b["rows"] = recs[1:] if len(recs) > 1 else []
                st.caption("La 1ª fila son los encabezados; las siguientes, los datos.")
    return blocks


def _alignment(meta):
    rep_section_header("Reporte de Alineación",
                       "Introducción · antecedentes · metodología · desarrollo libre",
                       "SIGA-FMT-ALI")
    intro = st.text_area("1. Introducción y alcance", key="ali_intro", height=90)
    ante = st.text_area("2. Antecedentes", key="ali_ante", height=80)
    c1, c2 = st.columns(2)
    hall = c1.text_area("3. Hallazgos (uno por línea)", key="ali_hall", height=100)
    reco = c2.text_area("4. Recomendaciones finales (una por línea)", key="ali_reco", height=100)

    # 5. Metodología: texto + imágenes + tablas de equipo (conductor/conducido/alineador)
    st.markdown("**5. Metodología** — texto, imágenes y tablas de equipo.")
    met_text = st.text_area("Texto de metodología", key="ali_met_txt", height=90,
                            placeholder="Alineación láser doble haz; estacionaria → móvil...")
    with st.expander("Imágenes de metodología", expanded=False):
        met_photos = _photo_uploader("ali_met", label="Imágenes")
    with st.expander("Tablas de equipo (conductor · conducido · alineador)", expanded=False):
        equipos = _equipos_composer("ali_eq")

    # 6. Desarrollo del servicio — ORDEN LIBRE
    st.markdown("**6. Desarrollo del servicio** — arma el orden que quieras "
                "(texto, imágenes y tablas):")
    dev_blocks = _free_block_composer("ali_dev")

    # 7. Anexos
    with st.expander("7. Anexos (imágenes)", expanded=False):
        anexo = _photo_uploader("ali_anx", label="Imágenes de anexo")

    content = {
        "introduccion": intro, "antecedentes": ante,
        "hallazgos": _lines(hall), "recomendaciones": _lines(reco),
        "met_text": met_text, "met_photos": met_photos, "met_equipos": equipos,
        "dev_blocks": dev_blocks, "anexo_photos": anexo,
    }
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
    """Renderiza el formulario + generación PDF de la familia elegida, con
    autoguardado + borradores (no se pierde el trabajo ante caída/redeploy)."""
    form = _FORMS.get(family)
    if form is None:
        st.error(f"Tipo de reporte desconocido: {family}")
        return
    # 1) Aplicar restauración/limpieza pendiente ANTES de instanciar widgets.
    _apply_pending(family)
    # 2) Barra de borradores (recuperar/guardar/cargar/duplicar/nuevo).
    _draft_bar(family)
    # Indicador visible de autoguardado (igual que Calibración).
    _ats = st.session_state.get(f"_rep_autosave_ts_{family}")
    st.caption("🟢 Autoguardado activo — el reporte se recupera si se cae la "
               "sesión" + (f" · último guardado: {_ats}" if _ats else "."))
    # 3) Formulario.
    meta = _meta_form(family)
    st.divider()
    form(meta)
    # 4) Autoguardado del estado (tolerante a disco lleno; no crashea).
    try:
        if _drafts.autosave(_rep_module(family), _capture_family_state(family)):
            import datetime as _dt
            st.session_state[f"_rep_autosave_ts_{family}"] = \
                _dt.datetime.now().strftime("%H:%M:%S")
    except Exception:
        pass


__all__ = ["render_report_family"]
