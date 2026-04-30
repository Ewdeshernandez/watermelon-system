#!/bin/bash
# =============================================================
# Watermelon — HOTFIX 7 Ciclo 14a: instrumentacion visual del esquematico
# =============================================================
# Confirmacion via PDF inspection: el reporte sale sin la imagen del
# esquematico en pagina 3 porque meta.schematic_doc_id queda vacio,
# lo que significa que inst.schematic_png de la instancia activa
# nunca se logro setear.
#
# El usuario hizo varios intentos pero NO podia ver desde la UI si
# el problema estaba en (a) el activo no tiene schematic_png, (b)
# el meta no se rellena del activo, (c) el render falla.
#
# Fix de instrumentacion (no cambia logica, agrega visibilidad):
#
# 1) pages/00_Machinery_Library.py — cada card del grid muestra:
#    "🖼️ esquemático vinculado" (cuando schematic_png != '')
#    "⚠️ sin esquemático principal" (cuando esta vacio)
#    El usuario sabe DE UN VISTAZO si la maquina tiene el
#    esquematico vinculado.
#
# 2) pages/16_Reports.py — antes del boton "Preparar PDF" hay un
#    expander 'Auto-fill desde activo monitoreado' que muestra:
#    - Activo activo (tag, cliente, sitio, clase, modelo)
#    - Train description compuesta
#    - Estado del esquematico:
#       * 'Esquematico listo' (con tamano del archivo) — verde
#       * 'Activo tiene schematic_png pero meta no lo tomo' +
#         boton "Reset auto-fill" — naranja
#       * 'Activo NO tiene esquematico principal vinculado' — rojo
#         con instrucciones claras
#    - Si NO hay activo activo: warning explicito.
#
#    Asi el usuario diagnostica en la UI sin necesidad de
#    inspeccionar PDFs ni base de datos.
#
# Ejecutar:
#   bash _hotfix_ciclo14a_diag_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add pages/00_Machinery_Library.py pages/16_Reports.py
git status --short | head

git commit -m "fix(library): hotfix 7 Ciclo 14a — instrumentacion visual del esquematico

El usuario no podia ver desde la UI por que el esquematico no
aparecia en el Resumen Ejecutivo del PDF. Era diagnostico a
ciegas con multiples reportes generados.

Fix de instrumentacion:
* Library cards muestran badge 'esquematico vinculado' / 'sin
  esquematico principal' segun inst.schematic_png.
* Reports pagina tiene expander 'Auto-fill desde activo
  monitoreado' que muestra cliente/sitio/clase/modelo/train_desc
  derivados + estado preciso del esquematico (listo / pendiente /
  faltante) con instrucciones claras y boton 'Reset auto-fill'
  para forzar recarga si el meta quedo desincronizado."

git push origin dev

echo ""
echo "================================================================"
echo " HOTFIX 7 listo — ahora la UI te dice exactamente que falta"
echo "================================================================"
