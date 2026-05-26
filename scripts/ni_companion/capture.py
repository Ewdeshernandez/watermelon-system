#!/usr/bin/env python3
"""
DEPRECATED (v3.31.245)
======================

Este script fue movido a:

    scripts/capture_companion/capture.py

El directorio `scripts/ni_companion/` queda como stub para que cualquier
script viejo que aún apunte acá imprima un error claro y guíe al user.

Para uso real:

    python scripts/capture_companion/capture.py --help

El user puede borrar este directorio completo (`scripts/ni_companion/`)
una vez que verifique que ya nadie lo referencia.
"""
import sys

print(
    "ERROR: scripts/ni_companion/capture.py está deprecated.\n"
    "       Movió a scripts/capture_companion/capture.py\n"
    "\n"
    "Usá:  python scripts/capture_companion/capture.py [args...]",
    file=sys.stderr,
)
sys.exit(2)
