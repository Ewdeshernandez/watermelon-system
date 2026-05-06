"""
pages/_ai_assistant.py
======================

AI Assistant — Q&A sobre el archivo histórico de reportes (Ciclo 17.27).

UI tipo chat donde el especialista o gerente de mantenimiento le pregunta
a Claude sobre los reportes archivados en lenguaje natural. Claude
responde con citaciones a reportes específicos (consecutivo, fecha,
activo) que se renderizan como links descargables.

Demo killer para ventas:
  - "Mostrame todos los activos con oil whip de los últimos 6 meses"
  - "¿Cuál fue el reporte con peor severidad en TES3?"
  - "Compará TES1 vs TES3 en 2026"
  - "¿Qué patrones de fallo se repiten en el cliente ECOPETROL?"

Arquitectura:
  - Conversación persistente en st.session_state["wm_aiq_history"]
    (lista de turns: user/assistant)
  - Cada respuesta del assistant trae reports_referenced que se
    renderiza como sección de "Reportes citados" con botones de
    descarga del PDF directamente desde la respuesta
  - Sugerencias de preguntas comunes en sidebar
  - Footer con costo total acumulado de la sesión
"""
from __future__ import annotations

import time
from io import BytesIO
from typing import Any, Dict, List

import streamlit as st

from core.auth import (
    get_current_user,
    render_user_menu,
    require_login,
)
from core.ai_qa import query_archive, clear_qa_cache, extract_pdf_text
from core.ai_diagnostic import is_ai_available
from core.reports_archive import get_archived_pdf_bytes, list_archived_reports


# =============================================================
# CONFIG + GUARDS
# =============================================================

st.set_page_config(
    page_title="Watermelon System | AI Assistant",
    layout="wide",
)
require_login()
render_user_menu()

_user = get_current_user() or {}
_user_email = (_user.get("email") or "").strip()
_user_role = (_user.get("role") or "").strip().lower()

# Solo admin + specialist por ahora. client puede liberarse después.
if _user_role not in ("admin", "specialist"):
    st.error(
        "🔒 Acceso restringido. El AI Assistant está disponible para roles "
        "**admin** y **specialist**. Si tu rol es client, contactá al "
        "especialista responsable de tu activo."
    )
    st.stop()


# =============================================================
# SESSION STATE
# =============================================================

if "wm_aiq_history" not in st.session_state:
    st.session_state["wm_aiq_history"] = []  # lista de turns

if "wm_aiq_total_cost" not in st.session_state:
    st.session_state["wm_aiq_total_cost"] = 0.0

if "wm_aiq_total_tokens_in" not in st.session_state:
    st.session_state["wm_aiq_total_tokens_in"] = 0

if "wm_aiq_total_tokens_out" not in st.session_state:
    st.session_state["wm_aiq_total_tokens_out"] = 0

if "wm_aiq_use_pdf_text" not in st.session_state:
    st.session_state["wm_aiq_use_pdf_text"] = False


# =============================================================
# HEADER
# =============================================================

st.markdown(
    """
    <div style='padding:14px 18px; border-radius:14px; background:linear-gradient(135deg,#0f172a 0%, #1e3a8a 100%); color:#f1f5f9; margin-bottom:16px;'>
        <div style='font-size:0.95rem; font-weight:600; letter-spacing:0.04em;
                    text-transform:uppercase; color:#cbd5e1;'>
            Watermelon System · AI Assistant
        </div>
        <div style='font-size:1.6rem; font-weight:700; margin-top:4px;'>
            🧠 Consultá tu archivo histórico en lenguaje natural
        </div>
        <div style='font-size:0.95rem; color:#cbd5e1; margin-top:6px;'>
            Hacé preguntas sobre los reportes archivados de tus activos.
            El sistema responde con citaciones precisas a los reportes
            relevantes (consecutivo, fecha, severidad).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not is_ai_available():
    st.warning(
        "🔑 **AI no disponible.** Falta configurar `[anthropic] api_key` "
        "en los secrets de Streamlit Cloud. Mientras tanto, el archivo "
        "histórico sigue accesible desde **Reports → Archivo histórico** "
        "(navegación tradicional con filtros)."
    )
    st.stop()


# =============================================================
# SIDEBAR — sugerencias + opciones
# =============================================================

with st.sidebar:
    st.markdown("### Preguntas sugeridas")
    _suggestions: List[str] = [
        "¿Cuáles activos tienen severidad CRÍTICA en el archivo?",
        "Mostrame todos los reportes con oil whip de los últimos 6 meses.",
        "¿Cuál fue el reporte con peor severidad de 2026?",
        "Resumime los hallazgos principales del cliente ECOPETROL.",
        "¿Qué patrones de fallo se repiten en los turbogeneradores?",
        "Compará la evolución de severidad por activo en este año.",
    ]
    for i, sugg in enumerate(_suggestions):
        if st.button(sugg, key=f"wm_aiq_sugg_{i}", use_container_width=True):
            st.session_state["wm_aiq_pending_question"] = sugg

    st.markdown("---")
    st.markdown("### Opciones")

    st.session_state["wm_aiq_use_pdf_text"] = st.toggle(
        "Análisis profundo (extrae texto de los PDFs)",
        value=st.session_state["wm_aiq_use_pdf_text"],
        help=(
            "Por default, el AI usa solo la metadata de cada reporte "
            "(cliente, activo, severidad, resumen ejecutivo). Activá "
            "esto para que también lea el contenido completo del PDF "
            "— responde preguntas más profundas pero cuesta ~3x más "
            "tokens y tarda más."
        ),
    )

    st.markdown("---")
    st.markdown("### Sesión actual")
    _n_turns = len(st.session_state["wm_aiq_history"])
    _cost_total = st.session_state["wm_aiq_total_cost"]
    _in_total = st.session_state["wm_aiq_total_tokens_in"]
    _out_total = st.session_state["wm_aiq_total_tokens_out"]
    st.caption(f"Preguntas hechas: **{_n_turns // 2}**")
    st.caption(f"Tokens IN: {_in_total:,} · OUT: {_out_total:,}")
    st.caption(f"Costo acumulado: **~${_cost_total:.4f}**")

    if st.button("🗑 Limpiar conversación", use_container_width=True,
                 disabled=_n_turns == 0):
        st.session_state["wm_aiq_history"] = []
        st.session_state["wm_aiq_total_cost"] = 0.0
        st.session_state["wm_aiq_total_tokens_in"] = 0
        st.session_state["wm_aiq_total_tokens_out"] = 0
        st.rerun()

    if _user_role == "admin":
        st.markdown("---")
        if st.button("Limpiar cache (admin)", use_container_width=True):
            n = clear_qa_cache()
            st.success(f"{n} archivos de cache eliminados.")


# =============================================================
# INFO DE ARCHIVO ACCESIBLE
# =============================================================

with st.expander("📂 Archivo accesible para tu rol", expanded=False):
    try:
        _all_reports = list_archived_reports(
            viewer_email=_user_email,
            viewer_role=_user_role,
            limit=500,
        )
    except Exception as exc:
        _all_reports = []
        st.warning(f"No se pudo listar el archivo: {exc}")

    if not _all_reports:
        st.info(
            "No hay reportes archivados accesibles para tu rol. Cuando "
            "el especialista archive un reporte desde Reports → 📦 "
            "Archivar reporte, va a aparecer acá y vas a poder "
            "consultarlo."
        )
    else:
        st.caption(f"**{len(_all_reports)} reportes** accesibles según tu rol "
                   f"(`{_user_role}`).")
        # Agrupación por cliente para vista rápida
        _by_client: Dict[str, int] = {}
        _by_severity: Dict[str, int] = {}
        for sc in _all_reports:
            rm = sc.get("report_meta", {}) or {}
            client = rm.get("client", "(sin cliente)").strip() or "(sin cliente)"
            sev = rm.get("executive_severity", "").strip() or "(sin severidad)"
            _by_client[client] = _by_client.get(client, 0) + 1
            _by_severity[sev] = _by_severity.get(sev, 0) + 1
        _info_cols = st.columns(2)
        with _info_cols[0]:
            st.caption("**Por cliente:**")
            for c, n in sorted(_by_client.items(), key=lambda x: -x[1]):
                st.caption(f"  · {c}: {n}")
        with _info_cols[1]:
            st.caption("**Por severidad:**")
            for s, n in sorted(_by_severity.items(), key=lambda x: -x[1]):
                st.caption(f"  · {s}: {n}")


# =============================================================
# RENDER DE LA CONVERSACIÓN
# =============================================================

def _render_assistant_turn(turn: Dict[str, Any]) -> None:
    """Renderiza una respuesta del assistant: markdown + reportes citados."""
    if turn.get("fallback_used"):
        st.info(
            "ℹ️ Esta respuesta se generó con el modelo de respaldo "
            "(Haiku 4.5). Calidad ligeramente menor."
        )
    md = turn.get("markdown", "")
    st.markdown(md)

    # Reportes citados — botones de descarga
    refs: List[Dict[str, str]] = turn.get("reports_referenced", []) or []
    if refs:
        with st.expander(f"📑 Reportes citados ({len(refs)})", expanded=True):
            for ref in refs:
                aid = ref.get("archive_id", "")
                consec = ref.get("consecutive", "")
                client = ref.get("client", "")
                asset = ref.get("asset", "")
                date = ref.get("date", "")
                sev = ref.get("severity", "")

                ref_title = f"**{consec or aid.split('/')[-1]}**"
                ref_meta_bits = []
                if client:
                    ref_meta_bits.append(client)
                if asset:
                    ref_meta_bits.append(asset)
                if date:
                    ref_meta_bits.append(date)
                if sev:
                    ref_meta_bits.append(f"Severidad: {sev}")
                ref_meta = " · ".join(ref_meta_bits)

                ref_cols = st.columns([4.5, 1.5])
                with ref_cols[0]:
                    st.markdown(f"{ref_title}  \n<small>{ref_meta}</small>",
                                unsafe_allow_html=True)
                with ref_cols[1]:
                    # Botón de descarga del PDF
                    try:
                        _pdf_bytes = get_archived_pdf_bytes(
                            archive_id=aid,
                            viewer_email=_user_email,
                            viewer_role=_user_role,
                        )
                    except Exception:
                        _pdf_bytes = None
                    if _pdf_bytes:
                        _fname_safe = (consec or aid.replace("/", "_"))[:40]
                        st.download_button(
                            "⬇ PDF",
                            data=_pdf_bytes,
                            file_name=f"{_fname_safe}.pdf",
                            mime="application/pdf",
                            key=f"wm_aiq_dl_{aid}_{turn.get('turn_id', '')}",
                            use_container_width=True,
                        )
                    else:
                        st.button(
                            "⬇ PDF",
                            disabled=True,
                            key=f"wm_aiq_dl_dis_{aid}_{turn.get('turn_id', '')}",
                            use_container_width=True,
                            help="PDF no accesible.",
                        )

    # Caption con metadata técnica
    _model = turn.get("model", "")
    _tokens_in = turn.get("input_tokens", 0)
    _tokens_out = turn.get("output_tokens", 0)
    _cost = turn.get("cost_usd", 0.0)
    _n_in_ctx = turn.get("n_reports_in_context", 0)
    _fb = turn.get("fallback_used", False)
    _fb_tag = " · ⚠️ modelo de respaldo" if _fb else ""
    st.caption(
        f"Modelo: `{_model}` · Reportes en contexto: {_n_in_ctx} · "
        f"Tokens: {_tokens_in:,} → {_tokens_out:,} · "
        f"Costo: ~${_cost:.4f}{_fb_tag}"
    )


# Render del historial
for idx, turn in enumerate(st.session_state["wm_aiq_history"]):
    role = turn.get("role", "")
    if role == "user":
        with st.chat_message("user"):
            st.markdown(turn.get("content", ""))
    elif role == "assistant":
        with st.chat_message("assistant"):
            if turn.get("ok"):
                _render_assistant_turn(turn)
            else:
                st.error(turn.get("markdown", "Error en la respuesta."))


# =============================================================
# INPUT DE LA PREGUNTA
# =============================================================

# Si el usuario clickeó una sugerencia, prellenamos el input
_pending_q = st.session_state.pop("wm_aiq_pending_question", None)

_question_input = st.chat_input(
    "Hacé una pregunta sobre tus reportes archivados...",
)

# Procesar la pregunta (sea del input o de una sugerencia clickeada)
_question_to_process = _question_input or _pending_q

if _question_to_process:
    # 1) Agregar turn user al historial
    st.session_state["wm_aiq_history"].append({
        "role": "user",
        "content": _question_to_process,
        "turn_id": f"u_{len(st.session_state['wm_aiq_history'])}_{int(time.time())}",
    })

    # 2) Renderizar el turn user inmediatamente (mejor UX)
    with st.chat_message("user"):
        st.markdown(_question_to_process)

    # 3) Llamar a Claude
    with st.chat_message("assistant"):
        with st.spinner("🧠 Buscando en el archivo y sintetizando... (5-30 seg)"):
            try:
                _qa_result = query_archive(
                    _question_to_process,
                    viewer_email=_user_email,
                    viewer_role=_user_role,
                    use_pdf_text=st.session_state["wm_aiq_use_pdf_text"],
                    use_cache=True,
                )
            except Exception as exc:
                _qa_result = {
                    "ok": False,
                    "markdown": (
                        f"_⚠️ Error inesperado:_\n\n```\n"
                        f"{type(exc).__name__}: {exc}\n```"
                    ),
                    "reports_referenced": [],
                    "n_reports_in_context": 0,
                    "n_reports_in_archive": 0,
                    "model": "",
                    "fallback_used": False,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }

        # 4) Guardar el turn assistant
        _qa_result["role"] = "assistant"
        _qa_result["turn_id"] = (
            f"a_{len(st.session_state['wm_aiq_history'])}_{int(time.time())}"
        )
        st.session_state["wm_aiq_history"].append(_qa_result)

        # 5) Acumular costo + tokens de la sesión
        st.session_state["wm_aiq_total_cost"] += _qa_result.get("cost_usd", 0.0)
        st.session_state["wm_aiq_total_tokens_in"] += _qa_result.get("input_tokens", 0)
        st.session_state["wm_aiq_total_tokens_out"] += _qa_result.get("output_tokens", 0)

        # 6) Render inmediato
        if _qa_result.get("ok"):
            _render_assistant_turn(_qa_result)
        else:
            st.error(_qa_result.get("markdown", "Error en la respuesta."))

    # Forzar rerun para limpiar el chat_input y persistir el turn
    st.rerun()
