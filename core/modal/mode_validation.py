"""
core/modal/mode_validation.py — Validación AUTOMÁTICA de modos
==============================================================

Motor de veredicto por modo: decide si cada modo identificado es REAL (validado),
DUDOSO o RECHAZADO, con un score 0..100 y razones. Fusiona los criterios que un
analista experto revisa a mano y que el software líder automatiza:

  1. Amortiguamiento plausible   (0 < ζ ≲ 15–20 %; negativo/altísimo = no físico)
  2. Complejidad / MPC            (modo estructural real ≈ real-valued → baja complejidad)
  3. Acuerdo entre métodos        (FDD ∩ SSI: el mismo fn aparece por dos vías)
  4. Unicidad (MAC/frecuencia)    (no duplicado/partido con otro modo cercano)
  5. Armónico de giro             (fn ≈ k×RPM/60 → probable componente forzada, no modo)

No depende de Qt ni de la web: entrada = lista de dicts o de objetos con atributos
frequency/damping/complexity. Pensado para reutilizarse en campo y en la web.

Referencias: MPC (Pappa & Elliott 1993), estabilización SSI (Peeters & De Roeck),
clasificación armónico/estructural (Brincker & Ventura, OMA).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Any

# Umbrales (ajustables). Amortiguamiento en PORCENTAJE.
ZETA_MIN_PCT = 0.05
ZETA_OK_MAX_PCT = 15.0
ZETA_HARD_MAX_PCT = 25.0
MPC_GOOD_PCT = 15.0        # complejidad ≤ 15% → muy real
MPC_DOUBT_PCT = 45.0       # complejidad ≥ 45% → dudoso
VALIDATED_SCORE = 70.0
DOUBTFUL_SCORE = 40.0


@dataclass
class ModeVerdict:
    frequency_hz: float
    damping_ratio_pct: float
    complexity_pct: float
    score: float
    verdict: str                 # "validated" | "doubtful" | "rejected"
    reasons: List[str] = field(default_factory=list)
    confirmed_by_ssi: bool = False
    is_harmonic: bool = False

    def as_row(self) -> dict:
        return {
            "fn [Hz]": round(self.frequency_hz, 3),
            "ζ [%]": round(self.damping_ratio_pct, 3),
            "Complejidad [%]": round(self.complexity_pct, 1),
            "Score": round(self.score, 0),
            "Veredicto": {"validated": "Validado", "doubtful": "Dudoso",
                          "rejected": "Rechazado"}[self.verdict],
            "SSI": "✓" if self.confirmed_by_ssi else "",
            "Armónico": "⚠" if self.is_harmonic else "",
            "Razones": "; ".join(self.reasons),
        }


def _get(m: Any, *names, default=0.0) -> float:
    """Lee un atributo (objeto) o clave (dict) probando varios nombres."""
    for n in names:
        if isinstance(m, dict) and n in m:
            return float(m[n] if m[n] is not None else default)
        if hasattr(m, n):
            v = getattr(m, n)
            if v is not None:
                return float(v)
    return float(default)


def _mode_freq(m: Any) -> float:
    return _get(m, "frequency_hz", "natural_frequency_hz", "fn")


def _mode_zeta(m: Any) -> float:
    return _get(m, "damping_ratio_pct", "zeta")


def _mode_cplx(m: Any) -> float:
    return _get(m, "complexity_pct", "complexity")


def validate_modes(
    modes: Sequence[Any],
    ssi_freqs_hz: Optional[Sequence[float]] = None,
    running_speed_rpm: float = 0.0,
    tol_hz: float = 1.0,
    harmonic_orders: Sequence[float] = (1, 2, 3, 4, 5, 6),
    harmonic_tol_pct: float = 1.5,
) -> List[ModeVerdict]:
    """Devuelve un veredicto por modo. `ssi_freqs_hz` = frecuencias identificadas
    por SSI (para confirmar por segundo método). `running_speed_rpm` habilita la
    detección de armónicos de giro."""
    ssi = list(ssi_freqs_hz or [])
    x1_hz = (running_speed_rpm / 60.0) if running_speed_rpm else 0.0
    freqs = [_mode_freq(m) for m in modes]
    out: List[ModeVerdict] = []

    for i, m in enumerate(modes):
        fn = freqs[i]; zeta = _mode_zeta(m); cplx = _mode_cplx(m)
        score = 0.0; reasons: List[str] = []; hard_reject = False

        # 1) Amortiguamiento (0..30)
        if fn <= 0:
            reasons.append("frecuencia no válida"); hard_reject = True
        if zeta <= 0:
            reasons.append("amortiguamiento ≤ 0 (no físico / polo inestable)"); hard_reject = True
        elif zeta > ZETA_HARD_MAX_PCT:
            reasons.append(f"amortiguamiento {zeta:.1f}% excesivo")
        elif zeta > ZETA_OK_MAX_PCT:
            score += 15; reasons.append(f"amortiguamiento alto ({zeta:.1f}%)")
        else:
            score += 30

        # 2) Complejidad / MPC (0..25)
        if cplx <= MPC_GOOD_PCT:
            score += 25
        elif cplx <= MPC_DOUBT_PCT:
            score += 15
        else:
            reasons.append(f"complejidad alta ({cplx:.0f}%) → modo poco real")

        # 3) Confirmación por SSI (0..25)
        confirmed = any(abs(fn - sf) <= tol_hz for sf in ssi) if ssi else False
        if confirmed:
            score += 25; reasons.append("confirmado por SSI")
        elif ssi:
            reasons.append("no aparece en SSI")

        # 4) Unicidad (0..20): penaliza duplicado/partido con otro modo muy cercano
        near = [j for j, ff in enumerate(freqs) if j != i and abs(ff - fn) <= tol_hz]
        if not near:
            score += 20
        else:
            reasons.append("posible modo partido/duplicado")

        # 5) Armónico de giro (flag + penalización)
        is_harm = False
        if x1_hz > 0 and fn > 0:
            for k in harmonic_orders:
                if abs(fn - k * x1_hz) <= max(tol_hz, harmonic_tol_pct / 100.0 * k * x1_hz):
                    is_harm = True
                    reasons.append(f"≈ {k:g}× giro ({k * x1_hz:.1f} Hz) → posible armónico")
                    break
        if is_harm:
            score = min(score, 45.0)   # un armónico no debería validarse como modo
        if hard_reject:
            score = min(score, 30.0)   # no físico → nunca validado

        score = max(0.0, min(100.0, score))
        if score >= VALIDATED_SCORE and not is_harm:
            verdict = "validated"
        elif score >= DOUBTFUL_SCORE:
            verdict = "doubtful"
        else:
            verdict = "rejected"

        out.append(ModeVerdict(frequency_hz=fn, damping_ratio_pct=zeta, complexity_pct=cplx,
                               score=score, verdict=verdict, reasons=reasons,
                               confirmed_by_ssi=confirmed, is_harmonic=is_harm))
    return out


def verdict_rows(verdicts: Sequence[ModeVerdict]) -> List[dict]:
    return [v.as_row() for v in verdicts]


def summarize(verdicts: Sequence[ModeVerdict]) -> str:
    nv = sum(1 for v in verdicts if v.verdict == "validated")
    nd = sum(1 for v in verdicts if v.verdict == "doubtful")
    nr = sum(1 for v in verdicts if v.verdict == "rejected")
    nh = sum(1 for v in verdicts if v.is_harmonic)
    txt = f"{nv} modo(s) validado(s), {nd} dudoso(s), {nr} rechazado(s)"
    if nh:
        txt += f"; {nh} marcado(s) como posible armónico de giro"
    return txt + "."
