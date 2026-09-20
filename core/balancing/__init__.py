"""
core.torsional — Watermelon Torsional (análisis de torque / vibración torsional)
================================================================================

Módulo de campo para el sistema de telemetría de torque **Binsfeld TorqueTrak 10K**.

Cadena de medición
-------------------
Galga extensométrica en el eje → transmisor TX10K-S (RF 902-925 MHz) →
receptor RX10K → **salida analógica ±10 V DC** → tarjeta **NI 9215** (voltaje
DC, muestreo simultáneo) → Watermelon.

El RX10K entrega la señal como voltaje proporcional al torque (fondo de escala
= 10 V). Este paquete contiene la matemática pura y testeable que convierte
ese voltaje a unidades de ingeniería (N·m / ft-lb) y verifica la calibración
por shunt, análoga a `core.modal.signal_scaling` para el módulo modal.

Sub-módulos
-----------
· scaling.py    — geometría del eje + galga → torque/volt (Appendix B1/B2/B3)
· shunt_cal.py  — verificación de calibración por resistor shunt (Ref 1 / Ref 2)

Reutiliza del stack existente: `core.modal.acq_backend` (DAQmx, reloj
compartido), `core.order_tracking`, `core.tsa`, `core.modal.campbell`.
"""
from __future__ import annotations

from core.torsional.scaling import (  # noqa: F401
    ShaftGeometry,
    GageConfig,
    TorqueScaling,
    full_scale_torque,
    full_scale_force_axial,
    full_scale_strain_quarter,
    full_scale_strain_torque,
    voltage_to_torque,
    VFS_DEFAULT,
    VEXC_DEFAULT,
)
from core.torsional.shunt_cal import (  # noqa: F401
    ShuntReference,
    REF1_100UE,
    REF2_500UE,
    expected_shunt_voltage,
    simulated_strain_from_shunt,
    verify_shunt,
)
