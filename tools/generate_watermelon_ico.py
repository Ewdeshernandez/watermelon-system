#!/usr/bin/env python3
"""
tools/generate_watermelon_ico.py — Genera watermelon.ico desde el SVG
=======================================================================

Convierte planta/installer/assets/watermelon-logo.svg en un .ico
multi-resolución profesional para Windows.

Resoluciones generadas (mismo .ico, varios tamaños embebidos):
  · 16x16  — taskbar pequeña, file explorer modo lista
  · 32x32  — taskbar normal, file explorer iconos chicos
  · 48x48  — file explorer iconos medianos
  · 64x64  — file explorer iconos grandes
  · 128x128 — escritorio, alt-tab
  · 256x256 — vista jumbo

Output:
  · planta/installer/assets/watermelon.ico (~250 KB)

Uso:
    pip install cairosvg pillow
    python tools/generate_watermelon_ico.py

Después de correr, el siguiente build del .exe va a usar este ícono
automáticamente (el spec PyInstaller lo detecta si existe).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path


def main() -> int:
    try:
        import cairosvg
        from PIL import Image
    except ImportError as exc:
        print(f"ERROR: faltan dependencias — {exc}")
        print("Instala con:  pip install cairosvg pillow")
        return 1

    # Paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    svg_path = repo_root / "planta" / "installer" / "assets" / "watermelon-logo.svg"
    ico_path = repo_root / "planta" / "installer" / "assets" / "watermelon.ico"

    if not svg_path.exists():
        print(f"ERROR: SVG no encontrado en {svg_path}")
        return 1

    print(f"Leyendo SVG: {svg_path}")
    svg_bytes = svg_path.read_bytes()

    # Resoluciones que vamos a embeber en el .ico
    # Estrategia: renderizamos cada tamaño desde el SVG (calidad nativa),
    # y los empaquetamos manualmente en un ICO multi-resolución usando el
    # formato binario ICO directamente (más confiable que Pillow ICO writer).
    sizes = [16, 32, 48, 64, 128, 256]

    print(f"Renderizando {len(sizes)} resoluciones: {sizes}")
    pngs_in_memory = []  # lista de bytes PNG raw
    for size in sizes:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=size,
            output_height=size,
        )
        pngs_in_memory.append((size, png_bytes))
        print(f"  ✓ {size}x{size} ({len(png_bytes)} bytes PNG)")

    # ========================================================================
    # Empaquetar como ICO multi-resolución manualmente
    # Formato ICO: https://en.wikipedia.org/wiki/ICO_(file_format)
    # ========================================================================
    import struct

    n = len(pngs_in_memory)
    # Header: 6 bytes (reserved, type=1 for ICO, count)
    header = struct.pack("<HHH", 0, 1, n)
    # Directory entries: 16 bytes cada una
    # bWidth, bHeight, bColorCount, bReserved, wPlanes, wBitCount,
    # dwBytesInRes, dwImageOffset
    entries = b""
    image_data = b""
    offset = 6 + 16 * n  # después del header + N entries

    for size, png_bytes in pngs_in_memory:
        # ICO usa 0 para 256 (porque el campo es 1 byte)
        b_size = 0 if size == 256 else size
        entry = struct.pack(
            "<BBBBHHII",
            b_size,           # bWidth
            b_size,           # bHeight
            0,                # bColorCount (0 = >256 colores)
            0,                # bReserved
            1,                # wPlanes
            32,               # wBitCount (RGBA = 32-bit)
            len(png_bytes),   # dwBytesInRes
            offset,           # dwImageOffset
        )
        entries += entry
        image_data += png_bytes
        offset += len(png_bytes)

    ico_data = header + entries + image_data

    print(f"\nEscribiendo .ico multi-resolución a:\n  {ico_path}")
    ico_path.write_bytes(ico_data)

    size_kb = ico_path.stat().st_size / 1024
    print(f"\n✓ {ico_path.name} generado ({size_kb:.1f} KB)")
    print()
    print("Próximo paso:")
    print("  1. El .ico ya está en assets/ — PyInstaller lo va a detectar")
    print("  2. Descomentar SetupIconFile en installer.iss para usarlo en wizard")
    print("  3. wmpush v3.31.XXX → GitHub Actions builds nuevo .exe con icon bonito")
    return 0


if __name__ == "__main__":
    sys.exit(main())
