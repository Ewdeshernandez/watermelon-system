"""
core.admin.knowledge — Repositorio de conocimiento (RAG) · sección admin

Sube cursos de vibraciones + manuales de máquina. El motor de reportes
(briefing semanal/mensual) recupera pasajes relevantes y se apoya en ellos
para redactar con la terminología, los criterios y los umbrales correctos.
Material de REFERENCIA privada: la IA sintetiza y cita, no copia (copyright).

Sin efectos de import (el hub 20_Administracion.py hace login/rol/estilo).
"""
from __future__ import annotations

import streamlit as st

_SRC_TYPES = ["manual", "curso", "norma", "otro"]
_SRC_LABEL = {"manual": "Manual de máquina", "curso": "Curso de vibraciones",
              "norma": "Norma / estándar", "otro": "Otro"}


def render() -> None:
    from core.knowledge_base import (
        voyage_ready, ingest_document, list_documents, delete_document,
        EMBED_MODEL)

    st.markdown("### Repositorio de conocimiento")
    st.caption("Cursos de vibraciones y manuales de máquina. El motor de "
               "reportes se apoya en este material (recuperación semántica) "
               "para fundamentar el análisis. Solo tú (admin) lo ves. Uso de "
               "referencia: la IA sintetiza y cita — no copia texto del manual.")

    if not voyage_ready():
        st.error("Falta la llave de embeddings **VOYAGE_API_KEY** "
                 "(`secrets[voyage].api_key`). Sin ella no se puede ingestar "
                 "ni recuperar. Pégala en los secrets y recarga.")
        return

    try:
        from core.auth import get_current_user
        _me = (get_current_user() or {}).get("email", "") or ""
    except Exception:
        _me = ""

    # ---------- Subir ----------
    with st.expander("Subir documento (PDF)", expanded=True):
        up = st.file_uploader("PDF (manual, curso o norma)", type=["pdf"],
                              key="kb_pdf")
        c1, c2 = st.columns([2, 1])
        title = c1.text_input("Título", key="kb_title",
                              placeholder="p.ej. Manual SGT-300 · Vibración")
        src = c2.selectbox("Tipo", _SRC_TYPES, format_func=lambda s: _SRC_LABEL[s],
                           key="kb_src")
        model = st.text_input(
            "Modelo de máquina (opcional — para priorizar este manual en esa máquina)",
            key="kb_model", placeholder="p.ej. SGT300 / TM2500 / LM6000")
        st.caption(f"Motor de embeddings: {EMBED_MODEL} · 1024 dims. "
                   "Los PDF escaneados sin texto (solo imagen) no se pueden leer.")
        if st.button("Ingestar", type="primary", key="kb_ingest",
                     disabled=(up is None)):
            if up is None:
                st.warning("Selecciona un PDF.")
            else:
                with st.spinner("Extrayendo, troceando y generando embeddings…"):
                    res = ingest_document(
                        title or up.name, up.getvalue(),
                        source_type=src, machine_model=model,
                        filename=up.name, uploaded_by=_me)
                if res.get("ok"):
                    st.success(f"Ingestado: {res['n_chunks']} fragmento(s) "
                               f"indexado(s).")
                    st.rerun()
                else:
                    st.error(f"No se pudo ingestar: {res.get('error', '?')}")

    # ---------- Listar / borrar ----------
    docs = list_documents()
    st.markdown(f"#### Documentos ({len(docs)})")
    if not docs:
        st.info("Aún no hay documentos. Sube el primer manual o curso arriba.")
        return
    for d in docs:
        col = st.columns([3.2, 1.4, 1.4, 0.9, 0.8])
        col[0].markdown(f"**{d.get('title', '—')}**  \n"
                        f"<span style='color:#64748b;font-size:12px'>"
                        f"{d.get('filename', '') or ''}</span>",
                        unsafe_allow_html=True)
        col[1].write(_SRC_LABEL.get(d.get("source_type", ""), d.get("source_type", "—")))
        col[2].write(d.get("machine_model") or "—")
        col[3].write(f"{d.get('n_chunks', 0)} frag.")
        if col[4].button("Borrar", key=f"kb_del_{d.get('id')}"):
            if delete_document(d.get("id")):
                st.success("Borrado.")
                st.rerun()
            else:
                st.error("No se pudo borrar.")
