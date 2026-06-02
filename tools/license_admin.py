#!/usr/bin/env python3
"""
tools/license_admin.py — Administrador visual de licencias Watermelon Planta
============================================================================

Herramienta INTERNA de SIGA para CREAR, LISTAR, RENOVAR y VERIFICAR las
licencias del software de análisis modal (Watermelon Planta Edition).

Corré localmente:
    cd watermelon-system
    pip install streamlit pyjwt cryptography
    streamlit run tools/license_admin.py

⚠️⚠️ SEGURIDAD — LEER ⚠️⚠️
Esta app firma con la PRIVATE KEY (tools/.keys/private_key.pem). Por eso:
  * Corré SOLO en una máquina de confianza de SIGA, NUNCA en la nube/Render.
  * NUNCA subas tools/.keys/ a git (ya está en .gitignore — verificalo).
  * El cliente recibe SOLO license.token + README_CLIENTE.txt.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from license_core import (  # noqa: E402
    PLANS, VALID_MODULES, MODULE_LABELS,
    issue_license, list_issued_licenses, verify_token, keys_exist,
)

st.set_page_config(page_title="Licencias · Watermelon Planta", page_icon="🔑", layout="wide")

# ---- Guard de seguridad: la key no debe estar en un entorno cloud ----
st.title("🔑 Administrador de Licencias — Watermelon Planta")
st.caption("Herramienta interna SIGA · firma RSA-2048 · uso local únicamente")

if not keys_exist():
    st.error(
        "No se encuentran las claves en `tools/.keys/`. Generá el par una vez con:\n\n"
        "`python tools/license_keygen.py`"
    )
    st.stop()

st.warning(
    "⚠️ Esta app usa la **clave privada** para firmar. Corré SOLO en una máquina "
    "de confianza de SIGA, nunca en la nube. Al cliente se le envían únicamente "
    "`license.token` y `README_CLIENTE.txt`.",
    icon="🔒",
)

tab_list, tab_new, tab_verify = st.tabs(
    ["📋 Licencias emitidas", "➕ Crear / Renovar", "🔍 Verificar token"]
)

# =============================================================
# TAB 1 — LISTA
# =============================================================
with tab_list:
    licenses = list_issued_licenses()
    if not licenses:
        st.info("Todavía no hay licencias emitidas. Andá a la pestaña «Crear / Renovar».")
    else:
        _badge = {"VIGENTE": "🟢", "POR VENCER": "🟠", "VENCIDA": "🔴", "DESCONOCIDO": "⚪"}
        n_venc = sum(1 for l in licenses if l["_status"] == "VENCIDA")
        n_pv = sum(1 for l in licenses if l["_status"] == "POR VENCER")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total licencias", len(licenses))
        c2.metric("Por vencer (≤30 d)", n_pv)
        c3.metric("Vencidas", n_venc)
        st.divider()

        rows = []
        for l in licenses:
            rows.append({
                "Estado": f"{_badge.get(l['_status'],'⚪')} {l['_status']}",
                "Cliente": l.get("customer", ""),
                "Plan": l.get("plan_label", l.get("plan", "")),
                "Módulos": ", ".join(l.get("modules", [])),
                "Canales": l.get("max_channels", ""),
                "Email": l.get("email", ""),
                "Emitida": (l.get("issued_at_utc", "") or "")[:10],
                "Vence": (l.get("expires_at_utc", "") or "")[:10],
                "Días": l.get("_days_left"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "🟢 Vigente · 🟠 vence en ≤30 días · 🔴 vencida. "
            "Para renovar una vencida/por vencer, usá «Crear / Renovar» con el "
            "MISMO nombre de cliente y una nueva fecha."
        )

# =============================================================
# TAB 2 — CREAR / RENOVAR
# =============================================================
with tab_new:
    st.subheader("Emitir una licencia nueva o renovar una existente")
    st.caption(
        "Para **renovar**: usá exactamente el mismo nombre de cliente y elegí una "
        "nueva fecha de vencimiento (reemplaza el token anterior)."
    )

    with st.form("new_license"):
        c1, c2 = st.columns(2)
        with c1:
            customer = st.text_input("Cliente (razón social)", placeholder="Parex Resources Colombia")
            email = st.text_input("Email de contacto técnico", placeholder="ingenieria@cliente.com")
            plan_key = st.selectbox(
                "Plan comercial", options=list(PLANS.keys()),
                format_func=lambda k: PLANS[k]["label"], index=2,
            )
        with c2:
            _plan = PLANS[plan_key]
            default_exp = date.today().replace(year=date.today().year + 1)
            expires = st.date_input("Vence el", value=default_exp, min_value=date.today())
            max_channels = st.number_input(
                "Máx. canales simultáneos",
                value=int(_plan["default_max_channels"]), min_value=1, max_value=512, step=1,
            )
            modules = st.multiselect(
                "Módulos habilitados",
                options=sorted(VALID_MODULES),
                default=list(_plan["default_modules"]),
                format_func=lambda m: f"{m} — {MODULE_LABELS.get(m, m)}",
            )
        notes = st.text_input("Notas internas (no van al cliente)", placeholder="OC-2026-xxx · pago confirmado")

        submitted = st.form_submit_button("🔏 Emitir licencia firmada", use_container_width=True)

    if submitted:
        try:
            exp_dt = datetime(expires.year, expires.month, expires.day, tzinfo=timezone.utc)
            res = issue_license(
                customer=customer.strip(), email=email.strip(), plan=plan_key,
                expires_dt=exp_dt, modules=modules or None,
                max_channels=int(max_channels), notes=notes.strip(),
            )
        except (ValueError, RuntimeError) as e:
            st.error(f"No se pudo emitir: {e}")
        else:
            st.success(f"✓ Licencia emitida para **{res.customer}** · ID `{res.license_id}`")
            st.caption(f"Archivos guardados en: `{res.token_path.parent}`")
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇️ license.token (enviar al cliente)",
                    data=res.token, file_name="license.token",
                    mime="text/plain", use_container_width=True,
                )
            with d2:
                st.download_button(
                    "⬇️ README_CLIENTE.txt (enviar al cliente)",
                    data=res.readme_text, file_name="README_CLIENTE.txt",
                    mime="text/plain", use_container_width=True,
                )
            st.info(
                "📤 Enviá al cliente **solo** esos 2 archivos (por email). "
                "NO envíes `license.json` (es registro interno). "
                "Recordá: el token NUNCA por WhatsApp."
            )
            with st.expander("Ver el README que recibe el cliente"):
                st.code(res.readme_text, language="text")

# =============================================================
# TAB 3 — VERIFICAR
# =============================================================
with tab_verify:
    st.subheader("Verificar un token (igual que lo hace el software del cliente)")
    token_in = st.text_area("Pegá el contenido de license.token", height=140)
    if st.button("Verificar", use_container_width=True) and token_in.strip():
        res = verify_token(token_in)
        if res["valid"]:
            c = res["claims"]
            st.success("✓ LICENCIA VÁLIDA")
            exp = datetime.fromtimestamp(c["exp"], tz=timezone.utc)
            days = (exp - datetime.now(timezone.utc)).days
            st.write({
                "Cliente": c.get("customer"),
                "Email": c.get("sub"),
                "Plan": c.get("plan_label"),
                "Módulos": ", ".join(c.get("modules", [])),
                "Canales": c.get("max_channels"),
                "Vence": exp.strftime("%Y-%m-%d"),
                "Días restantes": days,
                "License ID": c.get("jti"),
            })
        else:
            st.error(f"✗ LICENCIA INVÁLIDA — {res['reason']}")
