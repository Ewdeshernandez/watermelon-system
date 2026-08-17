"""
tools/validate_balance.py
=========================

Batería de validación del módulo de Balanceo (core.balance). Garantiza que el
motor es correcto, estable y cumple norma:

  1. Fórmulas exactas ISO 21940-11/12 + API 684.
  2. Casos borde (trial cero, matriz singular, baja sensibilidad) → ValueError
     limpio (nada de crashes crudos).
  3. Monte Carlo de recuperación: impone coef. de influencia + desbalance,
     genera mediciones sintéticas y verifica que el módulo recupere la
     corrección exacta (error ~ precisión de máquina) en miles de casos.
  4. Diagnóstico estático/couple.
  5. Generación de reporte PDF (1p / ISO / combinado).

Uso:
    python tools/validate_balance.py
Sale con código 0 si todo pasa, 1 si algo falla.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from core.balance.engine import (  # noqa: E402
    to_complex as C, to_polar as P, solve_1plane, solve_2plane,
    umax_api684_gmm, calc_e_per, calc_U_per, calc_U_trial,
    evaluate_iso_grades, pct_reduction, status_from_ratio,
    diagnose_static_couple, status_level,
)
from core.balance.report import build_balance_pdf  # noqa: E402


def main() -> int:
    random.seed(7)
    res = {}

    def check(name, cond):
        res[name] = bool(cond)
        print(f"  {'PASS ✅' if cond else 'FALLA ❌'}  {name}")

    def raises(fn):
        try:
            fn(); return False
        except ValueError:
            return True
        except Exception:
            return False

    print("── 1) FÓRMULAS (ISO 21940-11/12 · API 684) ──")
    check("e_per = 9549·G/N", abs(calc_e_per(2.5, 3600) - 9549 * 2.5 / 3600) < 1e-9)
    check("U_per = e_per·W", abs(calc_U_per(6.6312, 11000) - 6.6312 * 11000) < 1e-6)
    check("Umax = 6350·W/N", abs(umax_api684_gmm(3500, 3600) - 6350 * 3500 / 3600) < 1e-9)
    check("U_trial = m·r", abs(calc_U_trial(50, 420) - 21000) < 1e-9)
    check("pct_reduction 5→1 = 80%", abs(pct_reduction(5, 1) - 80) < 1e-9)
    check("e_per rpm=0 → 0", calc_e_per(2.5, 0) == 0.0)

    print("── 2) CASOS BORDE (estabilidad) ──")
    z = C(1, 0)
    check("1p trial=0 → ValueError", raises(lambda: solve_1plane(2, 0, 3, 0, 0, 0)))
    check("1p Vt≈V0 → ValueError", raises(lambda: solve_1plane(2, 0, 2, 0, 10, 0)))
    check("2p trial=0 → ValueError", raises(lambda: solve_2plane(z, z, z, z, z, z, C(0, 0), z)))
    check("2p matriz singular → ValueError", raises(
        lambda: solve_2plane(C(1, 0), C(1, 0), C(1.1, 0), C(1.05, 0),
                             C(1.1, 0), C(1.05, 0), z, z)))
    r_low = solve_1plane(1, 0, 1.0000001, 0, 1e6, 0)
    check("1p sensibilidad baja → POOR/MED", r_low["quality"] in ("POOR", "MED"))

    print("── 3) MONTE CARLO 1 PLANO (2000 casos) ──")
    maxerr = 0.0
    for _ in range(2000):
        H = C(random.uniform(0.005, 1.5), random.uniform(0, 360))
        U0 = C(random.uniform(1, 100), random.uniform(0, 360))
        Wm, Wa = random.uniform(5, 60), random.uniform(0, 360)
        Wt = C(Wm, Wa)
        V0 = H * U0; Vt = V0 + H * Wt
        v0, vt = P(V0), P(Vt)
        r = solve_1plane(v0[0], v0[1], vt[0], vt[1], Wm, Wa)
        got = C(r["corr_mass_g"], r["corr_ang_deg"])
        maxerr = max(maxerr, abs(got - (-U0)) / max(abs(U0), 1e-9))
    check(f"recuperación exacta (err rel máx = {maxerr:.2e})", maxerr < 1e-6)

    print("── 4) MONTE CARLO 2 PLANOS (bien condicionados) ──")
    maxerr2 = 0.0; solved = 0
    for _ in range(4000):
        HAA, HAB, HBA, HBB = [C(random.uniform(0.05, 1.0), random.uniform(0, 360))
                              for _ in range(4)]
        M = np.array([[HAA, HAB], [HBA, HBB]])
        if np.linalg.cond(M) > 40:
            continue
        UA = C(random.uniform(1, 80), random.uniform(0, 360))
        UB = C(random.uniform(1, 80), random.uniform(0, 360))
        WA = C(random.uniform(5, 50), random.uniform(0, 360))
        WB = C(random.uniform(5, 50), random.uniform(0, 360))
        A0 = HAA * UA + HAB * UB; B0 = HBA * UA + HBB * UB
        A1 = A0 + HAA * WA; B1 = B0 + HBA * WA
        A2 = A0 + HAB * WB; B2 = B0 + HBB * WB
        r = solve_2plane(A0, B0, A1, B1, A2, B2, WA, WB); solved += 1
        maxerr2 = max(maxerr2, abs(r["WA_corr"] - (-UA)) / max(abs(UA), 1e-9),
                      abs(r["WB_corr"] - (-UB)) / max(abs(UB), 1e-9))
    check(f"recuperación exacta en {solved} casos (err rel máx = {maxerr2:.2e})", maxerr2 < 1e-6)

    print("── 5) DIAGNÓSTICO estático/couple + niveles ──")
    check("STATIC", diagnose_static_couple(10, 1, 1)["kind"] == "STATIC")
    check("COUPLE", diagnose_static_couple(1, 10, 10)["kind"] == "COUPLE")
    check("MIXED", diagnose_static_couple(10, 10, 10)["kind"] == "MIXED")
    check("status_level ALARM", status_level(2, 3, 2.5, 4) == "ALARM")
    check("status_from_ratio bordes",
          status_from_ratio(0.8)[0] == "CUMPLE" and status_from_ratio(1.2)[0] == "NO CUMPLE")

    print("── 6) GENERACIÓN DE REPORTE PDF ──")
    r1 = solve_1plane(5, 30, 8, 70, 10, 0)
    iso = evaluate_iso_grades(11000, 3600, 50000)
    one = {"unit": "µm", "v0": (5, 30), "trial": (10, 0), "vt": (8, 70), "result": r1}
    for name, kw in [("1p", dict(one_plane=one)), ("iso", dict(iso=iso)),
                     ("combo", dict(one_plane=one, iso=iso))]:
        pdf = build_balance_pdf(meta={"asset": "X", "unit": "µm"}, **kw)
        check(f"PDF {name} válido (>30KB, %PDF)", pdf[:5] == b"%PDF-" and len(pdf) > 30000)

    print("\n" + "=" * 60)
    ok = all(res.values())
    print(f"TOTAL: {sum(res.values())}/{len(res)} · "
          f"{'TODO PASA ✅' if ok else 'HAY FALLAS ❌'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
