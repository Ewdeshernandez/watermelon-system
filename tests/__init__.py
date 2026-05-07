"""
Watermelon System — test suite.

Esta carpeta contiene la suite de pruebas con golden datasets sintéticos
que validan el núcleo de análisis (core/) sin depender de archivos
binarios de cliente. Diseño:

  - conftest.py             — fixtures + generadores de señales sintéticas
                              (sine puro, multi-armónico, impactos de
                              rodamiento, holgura, oil whirl, Bode).
  - test_synthetic_signals  — sanity check de los propios generadores.
  - test_waveform_metrics   — RMS, peak, crest, kurtosis vs valores cerrados.
  - test_order_tracking     — orden 1X/2X/3X recuperados con error < 1%.
  - test_tsa                — TSA reduce ruido y conserva 1X.
  - test_rotordynamics_*    — ISO 20816 zonas, detect_critical_speeds.
  - test_bearing_fault_freq — BPFO/BPFI/BSF/FTF correctos vs fórmula.
  - test_iso_thresholds     — catálogo ISO/API consistente.

Ejecución:

    pip install -r requirements-dev.txt
    pytest -q                       # corre todo
    pytest -q tests/test_waveform_metrics.py    # corre uno
    pytest -q -k bearing            # filtra por nombre
    pytest -q --maxfail=1 -x        # detenerse en el primer fallo
"""
