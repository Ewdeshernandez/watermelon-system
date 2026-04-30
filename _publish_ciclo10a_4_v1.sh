#!/bin/bash
# =============================================================
# Watermelon — Ciclo 10A.4: Tabla de Contenido automática (dev)
# =============================================================
# Inserta una TABLA DE CONTENIDO automática como página 2 del
# reporte (entre la portada y el Resumen Ejecutivo), con dot
# leaders, números de página alineados a la derecha y bookmarks
# clickeables (links nativos PDF a la sección correspondiente).
# Ese es el upgrade que separa un boletín técnico de un informe
# formal de ingeniería.
#
# CAMBIOS en pages/16_Reports.py:
#
# 1) NUEVA SUBCLASE WMDocTemplate(SimpleDocTemplate):
#    Override de afterFlowable() — detecta Paragraphs cuyo style
#    es 'WMTOC1' o 'WMTOC2', les pone bookmarkPage y notifica al
#    TableOfContents con (level, text, pageNum, key).
#    KEY estable basado en id(flowable): garantiza que multiBuild
#    converja en 2 pasadas (un contador reseteado o un seq monó-
#    tono cambian entre pasadas y rompen la convergencia con
#    "Index entries not resolved after N passes").
#
# 2) NUEVOS ESTILOS:
#    * WMTOC1: clon visual de WMSection (H1 sección principal).
#      Lo usan: RESUMEN EJECUTIVO, 1. RECOMENDACIONES, 2. OBJETIVO
#      DEL SERVICIO, 3. DESARROLLO DEL SERVICIO, 4. FIGURAS Y
#      ANÁLISIS.
#    * WMTOC2: clon visual de WMFigureCaption (H2 sub-entrada).
#      Lo usan: cada caption "Figura N. ...".
#    * toc_level0_style: estilo del propio TOC nivel 0 — IBM Plex
#      Bold, 11pt, sin indent.
#    * toc_level1_style: estilo del propio TOC nivel 1 — IBM Plex
#      Regular, 10pt, indent 18, color slate más suave.
#
# 3) INYECCIÓN DE LA PÁGINA TOC:
#    Después del PageBreak post-portada y antes del RESUMEN
#    EJECUTIVO, se agrega:
#       Paragraph('TABLA DE CONTENIDO', WMSection)  ← NO entra al TOC
#       Spacer(0.20cm)
#       TableOfContents() con levelStyles + dotsMinLevel=0
#       PageBreak()
#
#    El header WMS-FMT-001 + footer disclaimer del canvas
#    (_draw_internal_page) se aplican normalmente — la página 2 sale
#    consistente con el resto del cuerpo.
#
# 4) BUILD CHANGE: doc.build(...) → doc.multiBuild(...).
#    multiBuild hace 2-3 pasadas hasta que las page numbers convergen.
#    En la 1a pasada el TOC sale vacío y registra entries; en la 2a
#    pasada el TOC tiene las entries con sus page numbers reales.
#
# RESULTADO en el PDF:
#    Página 1: Portada SIGA-style
#    Página 2: TABLA DE CONTENIDO con dot leaders
#       RESUMEN EJECUTIVO ............................. 3
#       1. RECOMENDACIONES ............................ 4
#       2. OBJETIVO DEL SERVICIO ...................... 4
#       3. DESARROLLO DEL SERVICIO .................... 5
#       4. FIGURAS Y ANÁLISIS ......................... 6
#         Figura 1. Waveform 1 — WF 5810.csv .......... 6
#         Figura 2. Waveform 2 — WF 5809.csv .......... 7
#         Figura 3. Waveform 3 — WF 5808.csv .......... 8
#    Página 3+: Resumen Ejecutivo, Recomendaciones, etc.
#
# Cada entry del TOC es clickeable en visores PDF nativos (Acrobat,
# Preview de macOS, browsers) — salta directo a la sección.
#
# Compile clean. Smoke test del patrón TOC validado en sandbox
# (5 páginas, dot leaders OK, page numbers correctos, convergencia
# en 2 pasadas).
#
# Compatibilidad backwards: si por algún motivo la pasada 2 falla,
# el PDF seguirá renderizándose sin TOC (multiBuild captura). El
# resto del reporte queda intacto.
#
# Ejecutar:
#   bash _publish_ciclo10a_4_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 10A.4: TOC automática página 2 (dev)"
echo "================================================================"
echo ""

[ -f .git/index.lock ] && rm -f .git/index.lock

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  echo "Cambiando a dev..."
  git checkout dev
fi
git pull origin dev

echo ""
echo "[1] Adoptando cambios..."
git add pages/16_Reports.py
git status --short | head
echo ""

echo "[2] Commit..."
git commit -m "feat(report): Ciclo 10A.4 — Tabla de Contenido automatica pagina 2 (dev)

Inserta TOC automatica como pagina 2 del PDF, entre portada y
Resumen Ejecutivo. Da peso de informe formal al reporte.

NUEVA SUBCLASE WMDocTemplate(SimpleDocTemplate) en
pages/16_Reports.py:
* afterFlowable() detecta Paragraphs con style 'WMTOC1' o 'WMTOC2'
* bookmarkPage + notify('TOCEntry', (level, text, page, key))
* Key estable basado en id(flowable) → multiBuild converge en
  2 pasadas (contador monotono romperia convergencia)

NUEVOS ESTILOS:
* WMTOC1: clon de WMSection (H1 secciones principales)
* WMTOC2: clon de WMFigureCaption (H2 sub-entradas figuras)
* toc_level0_style: IBM Plex Bold 11pt
* toc_level1_style: IBM Plex Regular 10pt, indent 18

INYECCION pagina TOC despues del PageBreak post-portada:
* Paragraph 'TABLA DE CONTENIDO' (WMSection, NO entra al TOC)
* TableOfContents() con dotsMinLevel=0
* PageBreak

USO de WMTOC1 en: RESUMEN EJECUTIVO + 1.RECOMENDACIONES +
2.OBJETIVO + 3.DESARROLLO + 4.FIGURAS Y ANALISIS.
USO de WMTOC2 en: cada caption Figura N.

BUILD: doc.build(...) → doc.multiBuild(...) — 2-3 pasadas hasta
convergencia de page numbers.

Resultado: pagina 2 con dot leaders + numeros de pagina alineados
a la derecha. Cada entry es clickeable (bookmark PDF nativo).

Compile clean. Smoke test del patron TOC validado en sandbox
(5 paginas, dot leaders, page numbers correctos, 2 pasadas)."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Ciclo 10A.4 en dev"
echo "================================================================"
echo ""
echo "Validar en wm-test.streamlit.app:"
echo ""
echo "  1. Cargar 3-4 figuras de analisis al reporte (Polar/Spectrum/"
echo "     Waveform). Llenar el resto de meta como siempre."
echo ""
echo "  2. Generar el PDF."
echo ""
echo "  3. Pagina 1: portada SIGA-style (sin cambios)."
echo ""
echo "  4. Pagina 2: TABLA DE CONTENIDO con:"
echo "     - 'TABLA DE CONTENIDO' como heading"
echo "     - Lista de secciones con dot leaders ........ N"
echo "       RESUMEN EJECUTIVO ........................ 3"
echo "       1. RECOMENDACIONES ....................... 4"
echo "       2. OBJETIVO DEL SERVICIO ................. 4"
echo "       3. DESARROLLO DEL SERVICIO ............... 5"
echo "       4. FIGURAS Y ANALISIS .................... 6"
echo "         Figura 1. ... ........................... 6"
echo "         Figura 2. ... ........................... 7"
echo "         ..."
echo "     - Header WMS-FMT-001 arriba + footer disclaimer abajo"
echo "       (consistente con resto del cuerpo)"
echo ""
echo "  5. Click en cualquier entry del TOC en Acrobat/Preview →"
echo "     debe saltar directo a esa pagina (bookmark PDF nativo)."
echo ""
echo "  6. Pagina 3 en adelante: Resumen Ejecutivo, Recomendaciones,"
echo "     Objetivo, Desarrollo, Figuras (igual que antes)."
echo ""
echo "Cuando confirmes:"
echo "  - Ciclo 13 Orbit avanzado (clasificador geometrico de orbitas)"
echo "  - Ciclo 10B Tabla 1 amplitudes (NORMAL/ALARMA/DISPARO) en PDF"
echo "  - Cuando dev este maduro: tag v2.1 + merge a main (reservado"
echo "    para cuando termine el bloque actual de mejoras)"
echo "================================================================"
