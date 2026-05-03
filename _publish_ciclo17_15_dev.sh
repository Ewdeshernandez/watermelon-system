#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.15 → DEV: Aislamiento de Reportes +
#                                  Archivo histórico de PDFs
# =============================================================
# Resuelve el segundo problema crítico que reportó el usuario:
# "como poder trabajar varios usuarios al tiempo sin borrar el
# histórico del otro usuario de reportes o carga de datos"
#
# Lo nuevo que ven los usuarios:
#
#   ► AISLAMIENTO DE REPORTES POR USUARIO
#     Antes: data/report_state.json era GLOBAL (compartido). Si
#       Ewdes y J Suarez trabajaban al mismo tiempo, se pisaban
#       el reporte mutuamente.
#     Ahora: data/users/{email_slug}/report_state.json — cada user
#       tiene su propio espacio aislado. Mismo aplica a drafts
#       nombrados (data/users/{email_slug}/report_drafts/).
#     Migración automática: si existe el JSON legacy global, se
#       copia al espacio del admin como su trabajo personal.
#
#   ► PERMISOS DE VISUALIZACIÓN CROSS-USUARIO
#     - Admin/specialist pueden INSPECCIONAR el reporte de otro
#       colega @sigasas.com en modo solo-lectura via expander
#       "Inspeccionar el reporte de otro especialista"
#     - Banner amarillo "estás viendo el reporte de X en SOLO LECTURA"
#     - Botón "Duplicar a mi reporte" copia los items al espacio
#       propio para edición
#
#   ► ARCHIVO HISTÓRICO INMUTABLE DE PDFs
#     Después de "Preparar PDF", aparece nuevo botón "Archivar reporte"
#     que guarda copia inmutable en:
#       data/reports_archive/{email_slug}/{YYYY}/{MM}/{ts}_{slug}.pdf
#     Sidecar JSON con metadata para listar/filtrar sin abrir el PDF.
#     Inmutable: si archivás dos veces el mismo, queda con sufijo _v2.
#     Opción "Compartir con cliente" al archivar → users con role=client
#     pueden verlo desde su panel.
#
#   ► TAB "ARCHIVO HISTÓRICO" en Reports
#     Lista todos los PDFs archivados visibles para tu role:
#       admin       → ve TODO (todos los autores)
#       specialist  → ve los suyos + de otros @sigasas.com
#       client      → ve SOLO los marcados shared_with_client
#     Filtros: cliente, activo, autor, año.
#     Por reporte: descargar, compartir/despublicar (toggle), eliminar
#     (con confirm doble — solo owner o admin).
#     KPIs arriba: total archivados, espacio, autores únicos.
#
#   ► ACTIVITY FEED FILTRABLE POR USUARIO (Home)
#     - Toggle "Solo mía / Toda la actividad" para admin/specialist
#       (client siempre ve solo la suya, no tiene toggle)
#     - Cada evento tiene avatar de iniciales coloreado según hash
#       del email para distinguir visualmente quién hizo qué
#     - Eventos nuevos: report_archived (cuando alguien archiva PDF)
#
# Cambios técnicos:
#
# (MODIFICADO) core/report_state.py
#   - _email_slug, _current_owner_email, get_user_data_dir,
#     get_user_state_file, get_user_drafts_dir
#   - _maybe_migrate_legacy_to_user (idempotente, solo admin)
#   - save_report_state, load_report_state, clear_report_state,
#     list_available_backups: parámetro `email` opcional. Si no se
#     pasa, resuelven al usuario activo desde session_state.
#   - ensure_report_state_loaded: el flag se asocia al email del
#     owner (report_state_loaded_for) — invalida automáticamente
#     si el usuario cambia (logout/login).
#   - Drafts: list_report_drafts, save/load/delete_named_report_draft
#     todos namespaced.
#   - list_all_users_with_state: API admin para ver "qué reportes
#     tiene cada usuario en curso".
#
# (NUEVO) core/reports_archive.py
#   - archive_report_pdf con sidecar JSON metadata
#   - list_archived_reports con ACL por role + filtros
#   - get_archived_pdf_bytes con verificación de permisos
#   - delete_archived_report (solo owner o admin)
#   - share_with_client toggle
#   - get_archive_stats para KPIs
#
# (MODIFICADO) pages/16_Reports.py
#   - Selector "Inspeccionar reporte de otro especialista"
#   - Banner solo-lectura cuando se ve reporte ajeno
#   - Botón "Duplicar a mi reporte"
#   - owner_email auto-inyectado en meta
#   - Botón "Archivar reporte" después del Descargar PDF
#   - Sección "Archivo histórico" al final con filtros + cards
#
# (MODIFICADO) core/home_metrics.py
#   - ActivityEvent ahora tiene owner_email
#   - list_recent_activity acepta viewer_email/viewer_role/owner_filter
#   - Lee per-user de data/users/{slug}/* en lugar del global
#   - Eventos nuevos: report_archived (lee de reports_archive/)
#
# (MODIFICADO) pages/_landing.py
#   - Toggle "Solo mía / Toda la actividad"
#   - Avatar de iniciales coloreado por owner_email en cada item
#
# Tests smoke (todos pasan en sandbox):
#   - Save/load aislado: cada user solo ve sus items
#   - Drafts aislados: no se mezclan
#   - list_all_users_with_state: vista admin de quién tiene qué
#   - Archive ACL: admin ve todo, specialist ve @sigasas, client
#     solo shared_with_client
#   - Filtros de archivo (client, asset, fecha): todos OK
#   - Activity feed por role: admin/specialist/client visibilidad
#     correcta, owner_filter filtra a usuario específico
#
# IMPORTANTE — pausamos antes de main:
#   Este es un cambio grande que toca persistencia. Antes de mergear
#   a main, probar en wm-test.streamlit.app con varios usuarios
#   reales en paralelo para validar:
#     1. Que cada user solo ve su reporte
#     2. Que admin puede inspeccionar el de otros
#     3. Que el botón "Archivar" funciona
#     4. Que el tab "Archivo histórico" muestra lo correcto según role
#     5. Que el toggle del feed funciona
#   Después de validar, merge dev → main como v3.3.0.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando 17.15..."
git add core/report_state.py
git add core/reports_archive.py
git add core/home_metrics.py
git add pages/16_Reports.py
git add pages/_landing.py
git add _publish_ciclo17_15_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.15..."
    git commit -m "feat(reports): aislamiento por owner + archivo historico inmutable (17.15)

Resuelve el problema critico 'como trabajar varios usuarios al tiempo
sin pisarse el reporte'.

NUEVO core/reports_archive.py:
- archive_report_pdf con sidecar JSON de metadata
- list_archived_reports con ACL: admin ve todo, specialist ve los
  suyos + otros @sigasas, client solo shared_with_client
- get_archived_pdf_bytes con verificacion de permisos
- delete_archived_report, share_with_client (solo owner o admin)
- get_archive_stats para KPIs

MODIFICADO core/report_state.py — namespacing por usuario:
- data/report_state.json (global) -> data/users/{email_slug}/...
- save/load/clear/list_drafts toman email opcional, default al
  usuario activo desde session_state
- ensure_report_state_loaded invalida cache si user cambia
- Drafts nombrados tambien per-usuario
- _maybe_migrate_legacy_to_user: copia el JSON legacy al espacio
  del admin la primera vez (idempotente)
- list_all_users_with_state: vista admin

MODIFICADO pages/16_Reports.py:
- owner_email auto-inyectado en meta
- Selector 'Inspeccionar reporte de otro especialista' (admin/spec)
- Banner solo-lectura cuando se ve reporte ajeno
- Boton 'Duplicar a mi reporte'
- Boton 'Archivar reporte' (popover con shared + notas)
- Seccion 'Archivo historico' con filtros + cards + ACL por role

MODIFICADO core/home_metrics.py:
- ActivityEvent ahora con owner_email
- list_recent_activity con viewer_email/viewer_role/owner_filter
- Lee per-user (data/users/{slug}/*) ademas del legacy
- Eventos report_archived nuevos

MODIFICADO pages/_landing.py:
- Toggle 'Solo mia / Toda la actividad' para admin/specialist
- Avatar de iniciales coloreado por owner en cada item del feed

Tests smoke OK: save/load aislado, drafts aislados, ACL del archive
funciona en 4 escenarios, filtros OK, activity feed con visibilidad
correcta por role.

Solo push a DEV. Pausar antes de main para probar con equipo real." || echo "  (sin cambios)"
else
    echo "  (no hay cambios)"
fi

echo ""
echo "▶ Reconciliando..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Rebase falló."; exit 1; }

echo "▶ Push dev..."
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.15 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " ► QUEDA POR HACER:"
echo "  1. Esperar redeploy de wm-test.streamlit.app (~60 seg)"
echo "  2. Probar el flujo completo con TÚ + J Suarez en paralelo:"
echo ""
echo "     Vos en una ventana:"
echo "       - Login como ehernandez@sigasas.com"
echo "       - Ir a Reports, agregar 5 items, guardar como draft"
echo "       - Generar PDF, ARCHIVARLO marcando 'compartir con cliente'"
echo ""
echo "     J Suarez en otra ventana (otro browser o incognito):"
echo "       - Login como jsuarez@sigasas.com"
echo "       - Ir a Reports → debe ver SU reporte vacío (no los de Ewdes)"
echo "       - Agregar 3 items propios, guardar"
echo "       - Probar 'Inspeccionar reporte de otro especialista' →"
echo "         debe mostrar el reporte de Ewdes en SOLO LECTURA"
echo "       - Click 'Duplicar a mi reporte' → ahora tiene 8 items propios"
echo ""
echo "     Vos otra vez:"
echo "       - Ir a Reports → debe seguir teniendo TUS items intactos"
echo "       - Ir al tab 'Archivo histórico' → debes ver el PDF que"
echo "         archivaste + (si J Suarez también archivó alguno) el de él"
echo ""
echo "  3. Si todo OK, mergeamos a main como v3.3.0 con script aparte."
echo ""
echo "  4. Si algo no anda, mandame screenshot + descripción y arreglo."
echo "================================================================"
