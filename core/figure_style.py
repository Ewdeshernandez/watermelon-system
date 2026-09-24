"""
core/figure_style.py
===================

FUENTE ÚNICA DE DISEÑO de los gráficos (espectro / onda / órbita).

Un solo lugar para los tokens visuales y de rango que ANTES vivían duplicados
en la web (dynamic_raw_view.py), la app (dynraw.ts) y los reportes
(briefing_figures.py). Web y reportes lo IMPORTAN; la app lo lee por API
(/v1/figure-style). Cambiar un color / rango / marcador aquí → migra a los 3.

Todo es JSON-able (STYLE) para poder servirlo tal cual por la API.
"""
from __future__ import annotations

from typing import Dict

STYLE_VERSION = "1.0"

# Fmax de display por tipo de medición (CPM). Desplazamiento/proximidad solo
# mide bien hasta 60k; velocidad 300k; aceleración 600k (tope real Nyquist).
FMAX_CPM: Dict[str, int] = {"disp": 60000, "vel": 300000, "accel": 600000}

# Colores por canal — CANÓNICOS = los de la web de Watermelon System (la web
# es la referencia de diseño). Cambiar aquí migra a app y reportes.
COLORS: Dict[str, str] = {
    "X": "#2563eb",        # azul — sensor X (horizontal)
    "Y": "#ea580c",        # naranja — sensor Y (vertical)
    "orbit_band": "#93c5fd",   # banda de vueltas (azul claro)
    "orbit_filt": "#0f172a",   # órbita 1X filtrada (tinta)
    "keyphasor": "#16a34a",    # punto de keyphasor (verde)
    "cursor": "#64748b",       # cursores de orden 1X/2X…
    "grid": "#e2e8f2",
}

# Cursores de orden que se marcan en el espectro.
ORDER_CURSORS = [1, 2, 3, 4, 5]

# Paleta multi-canal de los REPORTES (gráficas apiladas por máquina). Se cicla
# por canal. Vive aquí para que un cambio de color también sea single-source.
REPORT_PALETTE = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#d97706",
                  "#0891b2", "#be185d", "#475569"]

# Convención de amplitud por tipo (desplazamiento en pp; velocidad/acel en pico).
AMPLITUDE = {"disp": "pp", "vel": "peak", "accel": "peak"}

# Ángulos de montaje de sonda por defecto (Bently 45R / 45L).
ORBIT_ANGLES = {"angX": 45, "angY": 45}

# Keyphasor: hueco alrededor del arranque (fracción de vuelta) + punto brillante.
KEYPHASOR = {"gap_frac": 1 / 24, "dot": True}


def measure_type(unit: str) -> str:
    """unit → 'disp' | 'vel' | 'accel' (igual que la app dispType)."""
    s = (unit or "").lower()
    if s == "g" or "m/s2" in s or "m/s²" in s or "accel" in s:
        return "accel"
    if "mm/s" in s or "in/s" in s or "ips" in s or "vel" in s:
        return "vel"
    return "disp"


def fmax_cpm(unit: str) -> int:
    return FMAX_CPM[measure_type(unit)]


def as_dict() -> Dict:
    """Todo el estilo, JSON-able, para servir por la API."""
    return {
        "version": STYLE_VERSION,
        "fmax_cpm": FMAX_CPM,
        "colors": COLORS,
        "report_palette": REPORT_PALETTE,
        "order_cursors": ORDER_CURSORS,
        "amplitude": AMPLITUDE,
        "orbit_angles": ORBIT_ANGLES,
        "keyphasor": KEYPHASOR,
    }
