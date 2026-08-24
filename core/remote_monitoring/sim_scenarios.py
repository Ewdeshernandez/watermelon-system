"""
core/remote_monitoring/sim_scenarios.py — Banco de pruebas (escenarios sim)
===========================================================================

Presets con NOMBRE que arman un `StreamConfig` completo para el
`SimulatedStreamSource`, cubriendo TODO el sistema sin hardware:

  · proximidad (desplazamiento, mil pp, con fase/keyphasor) — órbitas, forma modal
  · velocidad (mm/s RMS, ISO 20816) — overall de máquina
  · aceleración (g, IEPE) — fallas de rodamiento (BPFO/BPFI/BSF), engrane, envelope
  · transitorios (runup/coastdown) con 1–2 críticas + oil whirl/whip

Uso:
    from core.remote_monitoring.sim_scenarios import build_scenario, SCENARIOS
    cfg = build_scenario("prox_runup_whip", fs=5120)
    src = SimulatedStreamSource(cfg)
"""
from __future__ import annotations

from typing import Dict, List

from core.modal.acq_backend import ChannelConfig
from core.remote_monitoring.stream_source import StreamConfig


# --- helpers de armado de canales ------------------------------------------
def _kph(bnc: int = 1) -> ChannelConfig:
    return ChannelConfig(name="KPH", coupling="DC", sensitivity_mv_per_eu=1.0,
                         bnc_port=bnc, units="pulses/rev")


def _radials(n_brg: int, units: str, coupling: str, sens: float, bnc0: int = 2) -> List[ChannelConfig]:
    """Par X/Y por cojinete (1..n_brg). Nombres 1Y,1X,2Y,2X,... (Y primero)."""
    chans: List[ChannelConfig] = []
    b = bnc0
    for brg in range(1, n_brg + 1):
        for ax in ("Y", "X"):
            chans.append(ChannelConfig(name=f"{brg}{ax}", coupling=coupling,
                                       sensitivity_mv_per_eu=sens, bnc_port=b, units=units))
            b += 1
    return chans


# --- catálogo de escenarios -------------------------------------------------
# Cada entrada: (descripción, dict de kwargs para _make()).
SCENARIOS: Dict[str, Dict] = {
    "prox_4brg": dict(
        desc="DEFAULT · 4 cojinetes · PROXIMIDAD (mil pp) · estable/arranque/parada (crítica 2000)",
        kind="prox", n_brg=4, rpm=3000.0, defect="unbalance", crit=2000.0,
        profile="constant", rpm_start=300.0, rpm_end=9000.0, ramp=90.0),
    "prox_6brg": dict(
        desc="6 cojinetes · PROXIMIDAD (mil pp) · desbalance leve · rpm fija → órbitas, tabular, forma modal",
        kind="prox", n_brg=6, rpm=3600.0, defect="unbalance", crit=2400.0),
    "prox_runup_whip": dict(
        desc="6 cojinetes · PROXIMIDAD · RUNUP con 2 críticas + oil whirl→whip → Bode, Cascada, Polar, diagnóstico",
        kind="prox", n_brg=6, defect="oil_whirl", crit=1800.0, crit2=4200.0,
        profile="runup", rpm_start=500, rpm_end=6000, ramp=90),
    "prox_misalign": dict(
        desc="4 cojinetes · PROXIMIDAD · DESALINEACIÓN (2X fuerte)",
        kind="prox", n_brg=4, rpm=3000.0, defect="misalignment", crit=2000.0),
    "prox_looseness": dict(
        desc="4 cojinetes · PROXIMIDAD · HOLGURA (armónicos 1–5X + ½X)",
        kind="prox", n_brg=4, rpm=3000.0, defect="looseness", crit=2000.0),
    "prox_rub": dict(
        desc="4 cojinetes · PROXIMIDAD · ROCE (subarmónicos + truncado)",
        kind="prox", n_brg=4, rpm=3000.0, defect="rub", crit=2000.0),
    "accel_bpfo": dict(
        desc="6 cojinetes · ACELERACIÓN (g, IEPE) · falla pista EXTERNA (BPFO) → espectro, envelope",
        kind="accel", n_brg=6, rpm=1800.0, defect="bearing_bpfo"),
    "accel_bpfi": dict(
        desc="6 cojinetes · ACELERACIÓN · falla pista INTERNA (BPFI, modulada 1X) → envelope",
        kind="accel", n_brg=6, rpm=1800.0, defect="bearing_bpfi"),
    "accel_bsf": dict(
        desc="4 cojinetes · ACELERACIÓN · falla ELEMENTO RODANTE (BSF)",
        kind="accel", n_brg=4, rpm=1800.0, defect="bearing_bsf"),
    "gear_mesh": dict(
        desc="2 cojinetes · ACELERACIÓN · ENGRANE (GMF + bandas laterales 1X)",
        kind="accel", n_brg=2, rpm=1500.0, defect="gear_mesh", teeth=31),
    "vel_iso_6brg": dict(
        desc="6 cojinetes · VELOCIDAD (mm/s RMS, ISO 20816) · desalineación → overall",
        kind="vel", n_brg=6, rpm=1500.0, defect="misalignment"),
    "coastdown": dict(
        desc="6 cojinetes · PROXIMIDAD · PARADA (coastdown) por 2 críticas → Bode inverso",
        kind="prox", n_brg=6, defect="none", crit=1800.0, crit2=4200.0,
        profile="coastdown", rpm_start=6000, rpm_end=300, ramp=90),
    "motor_bomba": dict(
        desc="TREN MIXTO · MOTOR (rodamientos, 4 acelerómetros) + BOMBA (cojinetes planos, "
             "4 proximidades) · keyphasor común (fase) · falla BPFO en motor + oil whirl en bomba",
        train=True),
}


def _make_train(fs: float) -> StreamConfig:
    """Tren real: motor eléctrico (rodamientos → 4 acelerómetros, cojinetes 1–2)
    acoplado a bomba (cojinetes planos/película → 4 proximidades X/Y, cojinetes
    3–4). Un solo keyphasor (fase común de todo el tren).

    Física: el MOTOR muestra falla de rodamiento (BPFO) en aceleración; la BOMBA
    muestra inestabilidad de película (oil whirl ~0.45X) en proximidad — ambos a
    la vez, cada uno en su sensor. rpm por encima de la crítica de la bomba para
    que el whirl aparezca."""
    ch: List[ChannelConfig] = [_kph(1)]
    b = 2
    # Motor — 2 cojinetes de rodamiento, acelerómetros H/V (g, IEPE)
    for brg in (1, 2):
        for ax in ("H", "V"):
            ch.append(ChannelConfig(name=f"{brg}{ax}", coupling="IEPE",
                                    sensitivity_mv_per_eu=100.0, bnc_port=b, units="g rms"))
            b += 1
    # Bomba — 2 cojinetes planos, proximidad X/Y (mil pp, DC)
    for brg in (3, 4):
        for ax in ("Y", "X"):
            ch.append(ChannelConfig(name=f"{brg}{ax}", coupling="DC",
                                    sensitivity_mv_per_eu=200.0, bnc_port=b, units="mil pp"))
            b += 1
    return StreamConfig(
        sample_rate_hz=fs, channels=ch, block_seconds=0.1, buffer_seconds=12.0,
        rpm=3000.0, sim_critical_rpm=1500.0, sim_zeta=0.06,
        rpm_start=300.0, rpm_end=6000.0, ramp_seconds=90.0,
        defect_by_kind={"accel": "bearing_bpfo", "prox": "oil_whirl"})


def _make(kind: str, n_brg: int, fs: float, rpm: float = 3600.0, defect: str = "none",
          crit: float = 0.0, crit2: float = 0.0, profile: str = "constant",
          rpm_start: float = 0.0, rpm_end: float = 0.0, ramp: float = 60.0,
          teeth: int = 22) -> StreamConfig:
    units = {"prox": "mil pp", "vel": "mm/s rms", "accel": "g rms"}[kind]
    coup = {"prox": "DC", "vel": "AC", "accel": "IEPE"}[kind]
    sens = {"prox": 200.0, "vel": 100.0, "accel": 100.0}[kind]
    chans = [_kph(1)] + _radials(n_brg, units, coup, sens, bnc0=2)
    return StreamConfig(
        sample_rate_hz=fs, channels=chans, block_seconds=0.1, buffer_seconds=12.0,
        rpm=rpm, defect=defect, speed_profile=profile,
        rpm_start=rpm_start, rpm_end=rpm_end, ramp_seconds=ramp,
        sim_critical_rpm=crit, sim_critical_rpm2=crit2, sim_gear_teeth=teeth)


def build_scenario(name: str, fs: float = 5120.0) -> StreamConfig:
    """Construye el StreamConfig del escenario `name` (ver SCENARIOS)."""
    if name not in SCENARIOS:
        raise KeyError(f"escenario '{name}' desconocido. Opciones: {', '.join(SCENARIOS)}")
    kw = {k: v for k, v in SCENARIOS[name].items() if k != "desc"}
    if kw.pop("train", False):
        return _make_train(fs=fs)
    return _make(fs=fs, **kw)


def scenario_names() -> List[str]:
    return list(SCENARIOS.keys())
