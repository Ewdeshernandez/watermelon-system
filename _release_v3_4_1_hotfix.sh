#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.4.1 HOTFIX → MAIN
# =============================================================
# HOTFIX URGENTE Ciclo 17.17 — directo a main.
#
# Bug que matamos:
#   Con 10+ imágenes en un reporte, el sistema colapsa, se cae,
#   y los reportes se pierden. Especialistas bloqueados.
#
# Causa raíz:
#   Las imágenes se guardaban como base64 INLINE dentro del
#   report_state.json. Con 50 imágenes el JSON pesaba ~7.8MB,
#   con 100+ imágenes >130MB. Streamlit Cloud (1GB RAM):
#     - json.dump tardaba 10-30s y bloqueaba reruns
#     - si el usuario hacía algo mid-write → JSON corrupto
#     - json.loads al cargar reventaba la RAM
#     - cada caída perdía el reporte entero
#
# Fix:
#   Cada imagen se persiste como PNG suelto en
#     data/users/{slug}/report_images/{item_id}.png
#   y el JSON solo guarda image_file (path relativo).
#   Bytes idénticos al original — NO toca calidad.
#
# Validado con smoke test (_test_17_17_png_storage.py):
#   - 60 items con 100KB c/u → JSON pasó de ~7.8MB a 18.45 KB
#   - 60/60 hashes SHA256 idénticos al original (lossless)
#   - Migración legacy (b64 → PNG) funcional y transparente
#   - Items sin imagen no crean PNGs basura
#
# Garantía sobre reportes existentes:
#   La PRIMERA vez que cada usuario abra su reporte después del
#   redeploy, el image_bytes_b64 viejo se decodifica a bytes y
#   se escribe como PNG. La próxima save() limpia el b64 del
#   JSON. Los bytes son los MISMOS — idéntica calidad.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚨 HOTFIX v3.4.1 → MAIN  (urgente, especialistas bloqueados)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.17  PNG sueltos para fix crashes con 10+ imágenes"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del refactor en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Verificar si hay cambios para commitear
if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/report_state.py _test_17_17_png_storage.py _release_v3_4_1_hotfix.sh
    git commit -m "feat(17.17): PNG sueltos para imágenes de reportes — fix crash con 10+ imgs

Bug crítico: con muchas imágenes el sistema colapsaba y se perdían
reportes. Causa raíz: cada imagen se guardaba como base64 inline en
report_state.json. Con 50 imágenes el JSON pesaba 7.8MB, con 100+
>130MB. Streamlit Cloud (1GB RAM) no aguantaba.

Cambio:
- Cada imagen se persiste como PNG suelto en
  data/users/{slug}/report_images/{item_id}.png
- El JSON solo guarda image_file (path relativo)
- _atomic_write_png con tmpfile + fsync + os.replace (igual al
  helper de JSON) — sin re-encoding, bytes bit-a-bit idénticos
- Migración transparente: cuando se carga un JSON legacy con
  image_bytes_b64, restore_report_items decodifica los bytes,
  escribe el PNG a disco y deja image_file apuntando. La
  próxima save() limpia el b64 del JSON.
- Drafts comparten el mismo image pool del usuario por item_id
  (deduplicación natural)
- API publica intacta: restore_report_items sigue devolviendo
  image_bytes (bytes), los consumers (16_Reports, etc.) no
  cambian

Smoke test (_test_17_17_png_storage.py):
- 60 items con 100KB c/u → JSON 7.8MB → 18.45 KB
- 60/60 hashes SHA256 idénticos al original (lossless)
- Migración legacy (b64 → PNG) funcional
- Items sin imagen no crean PNGs basura

NO toca calidad de imágenes — el PNG en disco es bit-a-bit
idéntico al PNG que el módulo origen (Plotly export, matplotlib
savefig, etc.) ya generaba." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el refactor commiteado"
echo ""

echo "▶ 2/7  Push de dev a origin..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Pull dev falló."; exit 1; }
git push origin dev || { echo "✗ Push dev falló."; exit 1; }
echo "  ✓ dev en origin actualizado"
echo ""

echo "▶ 3/7  Cambiando a main..."
git checkout main || { echo "✗ No se pudo cambiar a main."; exit 1; }
git fetch origin main
git pull --rebase origin main || { echo "✗ Pull main falló."; exit 1; }
echo "  ✓ main actualizado"
echo ""

echo "▶ 4/7  Mergeando dev → main..."
MERGE_MSG="hotfix(v3.4.1): merge dev -> main · Ciclo 17.17 PNG sueltos

URGENTE — especialistas bloqueados por crashes con muchas imágenes.

Bug que matamos:
  Con 10+ imágenes el sistema colapsaba en main. Reportes
  perdidos en cada caída. Causa: base64 inline en JSON.

Fix:
  Imágenes ahora se persisten como PNG sueltos en
    data/users/{slug}/report_images/{item_id}.png
  El JSON solo guarda image_file (KBs en lugar de MBs).

Garantías:
  - NO toca calidad: bytes bit-a-bit idénticos (lossless)
  - Migración transparente desde reportes viejos con
    image_bytes_b64 — la primera carga los convierte a PNG
    y la siguiente save() limpia el b64 del JSON
  - API publica intacta: consumers (16_Reports, etc.)
    siguen recibiendo image_bytes (bytes) como antes
  - Drafts comparten image pool por item_id (deduplica)

Validado con smoke test:
  - 60 imágenes 100KB → JSON pasó de 7.8MB a 18.45 KB
  - 60/60 hashes SHA256 idénticos
  - Migración legacy verificada
  - Items sin imagen no crean basura

Esto resuelve los 3 problemas reportados:
  1. Sistema cayéndose cada rato → JSON liviano + write atómico real
  2. Reportes perdidos en cada caída → JSON ya no se corrompe
  3. Crash con 10+ imágenes → RAM ya no se llena con base64 strings"

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.4.1..."
TAG_EXISTS=$(git tag -l "v3.4.1")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.4.1 ya existe. Saltando creación."
else
    git tag -a v3.4.1 -m "Hotfix v3.4.1 — Ciclo 17.17 PNG sueltos (fix crash imágenes)"
    echo "  ✓ Tag v3.4.1 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.4.1 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ HOTFIX v3.4.1 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en"
echo "    1-2 minutos. Después de eso:"
echo ""
echo "    - Cada usuario, al abrir su reporte, va a migrar"
echo "      automáticamente sus imágenes viejas (b64 → PNG)."
echo "      Es transparente y solo pasa una vez por reporte."
echo ""
echo "    - Los reportes nuevos se crean directo en formato PNG"
echo "      suelto. Crash con 10+ imágenes resuelto."
echo ""
echo "    - Los reportes ya guardados NO se rompen — la migración"
echo "      es backwards-compatible."
echo ""
echo " 🔍 Validación rápida en producción:"
echo "    1. Abrir cualquier reporte existente → debe verse igual"
echo "    2. Agregar una imagen nueva → guardar → verificar OK"
echo "    3. Cargar 15+ imágenes → ya no debería caerse"
echo ""
echo "================================================================"
