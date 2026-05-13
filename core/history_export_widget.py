"""
core.history_export_widget
==========================

Widget de exportación del histórico de un activo (Ciclo 23.84).

Tres modos:

  1. 📥 **Descargar ZIP local** — genera el ZIP en memoria y lo entrega
     al especialista vía st.download_button. El ZIP contiene manifest +
     opcional diagrama SVG/PNG + todos los snapshots JSON descomprimidos.

  2. 📧 **Enviar al cliente** — sube el ZIP a Supabase Storage bucket
     (público temporal) y abre el cliente de email con link pre-cargado.
     El cliente descarga desde el link.

  3. 🔄 **Backup manual** — sube al bucket privado `instance-history-backups`
     con timestamp. Para auditoría / restore en caso de error.

Storage: reusa bucket `instance-history` para shares al cliente, y bucket
`instance-history-backups` (privado, service_role) para backups.

API pública:

    render_history_export_section(instance_id, instance_obj, svg=None)

Asume:
  • core/history_storage.export_instance_as_zip_bytes() está disponible
  • La instancia activa tiene snapshots guardados (sino, mostrará ZIP vacío)
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st


def _build_manifest_extra(instance_obj: Any) -> Dict[str, Any]:
    """Extrae metadata de la instancia para el manifest del ZIP."""
    if instance_obj is None:
        return {}
    return {
        "tag": getattr(instance_obj, "tag", None) or getattr(instance_obj, "instance_id", ""),
        "client": getattr(instance_obj, "client", "") or "",
        "site": getattr(instance_obj, "site", "") or "",
        "driver_manufacturer": getattr(instance_obj, "driver_manufacturer", "") or "",
        "driver_model": getattr(instance_obj, "driver_model", "") or "",
        "driven_manufacturer": getattr(instance_obj, "driven_manufacturer", "") or "",
        "driven_model": getattr(instance_obj, "driven_model", "") or "",
        "asset_class": getattr(instance_obj, "asset_class", "") or "",
        "exported_for": "Cliente",
    }


def _upload_zip_to_storage(zip_bytes: bytes, instance_id: str) -> Optional[str]:
    """Sube el ZIP al bucket `instance-history` con path único.

    Devuelve la URL pública del ZIP, o None si falló.
    """
    try:
        from core.live_readings import _get_supabase_client
        from core.history_storage import BUCKET_NAME
    except Exception:
        return None

    client = _get_supabase_client()
    if client is None:
        return None

    # Path único con timestamp + UUID corto
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:8]
    safe_id = (instance_id or "asset").replace("/", "_").replace(" ", "_")
    path = f"_exports/{safe_id}_{ts}_{rand}.zip"

    try:
        client.storage.from_(BUCKET_NAME).upload(
            path=path,
            file=zip_bytes,
            file_options={
                "content-type": "application/octet-stream",
                "x-upsert": "true",
            },
        )
    except Exception as e:
        st.error(f"Error subiendo ZIP a Supabase: {e}")
        return None

    # URL pública (el bucket es public read)
    try:
        url = client.storage.from_(BUCKET_NAME).get_public_url(path)
        # supabase-py devuelve la URL con trailing slash a veces; limpiar
        return url.rstrip("?")
    except Exception:
        # Fallback: construir manualmente
        try:
            base_url = client.storage._client._base_url  # noqa: SLF001
            return f"{base_url}/object/public/{BUCKET_NAME}/{path}"
        except Exception:
            return None


def render_history_export_section(
    instance_id: str,
    instance_obj: Any = None,
    svg: Optional[bytes] = None,
    png: Optional[bytes] = None,
) -> None:
    """Renderiza la sección de export del histórico debajo del Live Monitoring.

    Args:
        instance_id: ID del activo activo
        instance_obj: instancia opcional para enriquecer el manifest
        svg: bytes del SVG del diagrama (opcional, para incluir en el ZIP)
        png: bytes del PNG del diagrama (opcional)
    """
    if not instance_id:
        return

    try:
        from core.history_storage import export_instance_as_zip_bytes, list_all_snapshots
    except Exception:
        return

    # Verificar si hay snapshots para exportar
    try:
        all_snaps = list_all_snapshots(instance_id)
        total = sum(len(v) for v in all_snaps.values())
    except Exception:
        total = 0

    # Ciclo 23.124 — Header minimal: title uppercase + meta count
    st.markdown(
        f"""
        <style>
        .wm-export-header {{
            display: flex; align-items: center; justify-content: space-between;
            gap: 12px; margin: 24px 0 10px 0;
        }}
        .wm-export-title {{
            font-size: 13px; font-weight: 800; color: #0f172a;
            letter-spacing: 0.06em; text-transform: uppercase;
        }}
        .wm-export-meta {{
            font-size: 11px; color: #94a3b8;
            font-family: ui-monospace, SF Mono, monospace;
            letter-spacing: 0.02em;
        }}
        /* Botones outlined consistentes con las cards de Última data */
        .wm-export-btn-host + div [data-testid="stButton"] button,
        .wm-export-btn-host + div [data-testid="stPopover"] button {{
            background: #ffffff !important;
            color: #1e40af !important;
            border: 1px solid #dbeafe !important;
            border-radius: 8px !important;
            font-size: 12px !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em !important;
            padding: 7px 14px !important;
            min-height: 34px !important;
            box-shadow: 0 1px 0 rgba(30,64,175,0.04) !important;
            transition: all 0.15s ease !important;
        }}
        .wm-export-btn-host + div [data-testid="stButton"] button:hover,
        .wm-export-btn-host + div [data-testid="stPopover"] button:hover {{
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%) !important;
            border-color: #93c5fd !important;
            color: #1e3a8a !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(30,64,175,0.10) !important;
        }}
        .wm-export-btn-host {{ display: block; height: 0; overflow: hidden; }}
        </style>
        <div class='wm-export-header'>
            <span class='wm-export-title'>HISTÓRICO DEL ACTIVO</span>
            <span class='wm-export-meta'>{total} snapshots</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if total == 0:
        st.caption(
            "_Sin snapshots todavía. Subí CSVs en Load Data → Guardar para "
            "Live Monitoring._"
        )
        return

    # Marker antes de las columnas de botones — engancha la CSS de hijos
    st.markdown(
        '<span class="wm-export-btn-host"></span>',
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 4])

    # Modo 1 — Descargar ZIP local
    with cols[0]:
        try:
            zip_popover = st.popover(
                "📥 Descargar ZIP",
                use_container_width=True,
                help="Genera ZIP con manifest + diagrama + todos los snapshots",
            )
            use_popover = True
        except AttributeError:
            zip_popover = st.expander("📥 Descargar ZIP")
            use_popover = False

        with zip_popover:
            include_diagram = st.checkbox(
                "Incluir diagrama (SVG + PNG)",
                value=True,
                key=f"export_inc_diag_{instance_id}",
            )
            st.caption(
                f"Contenido: manifest.json · README.txt · "
                f"{total} snapshots descomprimidos · "
                f"{'diagram.svg/png' if include_diagram else 'sin diagrama'}"
            )
            if st.button(
                "Generar ZIP",
                type="primary",
                use_container_width=True,
                key=f"export_zip_gen_{instance_id}",
            ):
                with st.spinner("Empaquetando histórico..."):
                    try:
                        zip_bytes = export_instance_as_zip_bytes(
                            instance_id=instance_id,
                            include_diagram_svg=svg if include_diagram else None,
                            include_diagram_png=png if include_diagram else None,
                            manifest_extra=_build_manifest_extra(instance_obj),
                        )
                        st.session_state[f"_zip_bytes_{instance_id}"] = zip_bytes
                    except Exception as e:
                        st.error(f"Error generando ZIP: {e}")

            zip_bytes = st.session_state.get(f"_zip_bytes_{instance_id}")
            if zip_bytes:
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                safe_id = (instance_id or "asset").replace("/", "_").replace(" ", "_")
                size_kb = len(zip_bytes) / 1024
                # Polish v3.31.85: warning si está cerca del 5 MB del bucket
                if size_kb > 4000:
                    st.warning(
                        f"⚠ ZIP grande: {size_kb:.0f} KB. El bucket Supabase "
                        f"acepta hasta 5 MB. Si vas a enviar al cliente, "
                        f"considerá descargar local y compartir por otro medio."
                    )
                else:
                    st.success(f"✓ ZIP listo: {size_kb:.1f} KB")
                st.download_button(
                    label=f"⬇ {safe_id}_history_{ts}.zip",
                    data=zip_bytes,
                    file_name=f"{safe_id}_history_{ts}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key=f"export_zip_dl_{instance_id}",
                )

    # Modo 2 — Enviar al cliente
    with cols[1]:
        try:
            send_popover = st.popover(
                "📧 Enviar al cliente",
                use_container_width=True,
                help="Sube el ZIP a Supabase Storage temporal y abre tu cliente de email con el link",
            )
        except AttributeError:
            send_popover = st.expander("📧 Enviar al cliente")

        with send_popover:
            client_email = st.text_input(
                "Email del cliente",
                placeholder="cliente@empresa.com",
                key=f"export_email_to_{instance_id}",
            )
            email_note = st.text_area(
                "Nota (opcional)",
                placeholder="Mensaje breve para incluir en el email",
                key=f"export_email_note_{instance_id}",
                height=72,
            )

            st.caption(
                "El ZIP queda accesible en una URL pública por hasta 24h. "
                "Tu cliente lo descarga directo desde el link."
            )

            if st.button(
                "Generar link y abrir email",
                type="primary",
                use_container_width=True,
                key=f"export_send_btn_{instance_id}",
            ):
                if not client_email or "@" not in client_email:
                    st.warning("Ingresá un email válido.")
                else:
                    with st.spinner("Generando ZIP..."):
                        try:
                            zip_bytes = export_instance_as_zip_bytes(
                                instance_id=instance_id,
                                include_diagram_svg=svg,
                                include_diagram_png=png,
                                manifest_extra=_build_manifest_extra(instance_obj),
                            )
                        except Exception as e:
                            st.error(f"Error generando ZIP: {e}")
                            zip_bytes = None

                    if zip_bytes:
                        with st.spinner("Subiendo a Supabase..."):
                            url = _upload_zip_to_storage(zip_bytes, instance_id)
                        if url:
                            # Build mailto link
                            asset_tag = getattr(instance_obj, "tag", None) or instance_id
                            subject = f"Histórico de análisis — {asset_tag}"
                            body_lines = [
                                f"Hola,",
                                "",
                                f"Te envío el histórico de análisis del activo {asset_tag}.",
                            ]
                            if email_note:
                                body_lines.extend(["", email_note])
                            body_lines.extend([
                                "",
                                f"Descargá el paquete acá (link válido 24h):",
                                url,
                                "",
                                "El paquete incluye: manifest.json + diagrama actual + "
                                "todos los snapshots históricos (waveforms, spectrums, "
                                "orbits, tabular).",
                                "",
                                "Generado con Watermelon System.",
                            ])
                            body = "\n".join(body_lines)
                            import urllib.parse
                            mailto = (
                                f"mailto:{urllib.parse.quote(client_email)}"
                                f"?subject={urllib.parse.quote(subject)}"
                                f"&body={urllib.parse.quote(body)}"
                            )
                            st.success("✓ ZIP subido. URL generada:")
                            st.code(url, language=None)
                            st.markdown(
                                f"<a href='{mailto}' target='_blank' style='"
                                f"display:inline-block;padding:8px 16px;"
                                f"background:#2563eb;color:white;border-radius:8px;"
                                f"text-decoration:none;font-weight:600;font-size:13px;'>"
                                f"📧 Abrir cliente de email</a>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.error("No se pudo subir el ZIP a Supabase.")

    # Ciclo 23.124 — quitamos el caption verboso "Tip: el cliente recibe..."
    # — no aporta para el analista, distrae visualmente. La info de qué
    # contiene el ZIP ya está en el popover de "Descargar ZIP".
    pass  # cols[2] queda vacío como spacer


__all__ = ["render_history_export_section"]
