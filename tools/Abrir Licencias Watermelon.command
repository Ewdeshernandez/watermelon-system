#!/bin/bash
# =====================================================================
# Abrir Licencias Watermelon — lanzador de doble clic (macOS)
# ---------------------------------------------------------------------
# Doble clic en este archivo abre el Administrador de Licencias en el
# navegador (tools/license_admin.py). Para cerrarlo: volvé a la ventana
# de Terminal que se abrió y apretá  Ctrl + C.
#
# Uso local únicamente (la herramienta firma con la clave privada).
# =====================================================================

cd "$HOME/Documents/WatermelonSystem" || {
    echo "No se encontró ~/Documents/WatermelonSystem"; exit 1; }

# Activar conda 'base' si está instalado (busca ubicaciones típicas).
for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/opt/anaconda3" \
         "$HOME/opt/miniconda3" "/opt/anaconda3" "/opt/miniconda3" \
         "/opt/homebrew/anaconda3" "/opt/homebrew/Caskroom/miniconda/base"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "$c/etc/profile.d/conda.sh"
        conda activate base 2>/dev/null
        break
    fi
done

echo "Abriendo Administrador de Licencias Watermelon…"
echo "(para cerrar: Ctrl + C en esta ventana)"
echo

if command -v streamlit >/dev/null 2>&1; then
    streamlit run tools/license_admin.py
else
    python -m streamlit run tools/license_admin.py
fi
