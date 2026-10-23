"""core.analysis — workspace unificado de análisis (Fase 2).

Cada vista (waveform/spectrum/orbit/polar/bode/shaft/trends) se extrae aquí como
función pura render_*(rec, ch, params) que recibe el registro ya cargado y dibuja,
sin cargar dato ni set_page_config. El shell (pages/02_Analysis.py) carga el CSV
una vez y llama la vista activa. Migración incremental desde las páginas monolíticas.
"""
