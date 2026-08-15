"""
core.balance.report
===================

Reporte PDF de balanceo, branded Watermelon/SIGA.

Se construye sobre `core.report_pdf_shell.render_report_pdf` (misma portada,
banda de encabezado, pie con version stamp y TOC que el resto de los reportes
de Watermelon), así el reporte de balanceo se ve idéntico al ecosistema. El
contenido y el polar plot (antes/después) provienen de ROTORIX (validado en
campo). Headless — no depende de Streamlit.

Entrada: metadatos + los resultados del motor (core.balance.engine). Todas las
secciones son opcionales: se incluye solo lo que se calculó en la sesión.

Uso:
    pdf = build_balance_pdf(meta=..., one_plane=..., two_plane=..., iso=...)

Normas: ISO 21940-11 / ISO 21940-12 · API 684.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle

from core.report_pdf_shell import render_report_pdf, make_styles, REGULAR, BOLD


_INK = "#0f172a"
_HEADER_BG = "#0f4c81"


# =====================================================================
# Polar plot (antes / después) — matplotlib, PNG bytes
# =====================================================================
def polar_png(title: str, before: Tuple[float, float],
              after: Tuple[float, float], unit: str) -> Optional[bytes]:
    """Diagrama polar con los vectores antes/después. 0° arriba, sentido
    horario. Devuelve PNG bytes o None si matplotlib no está disponible."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    bm, ba = float(before[0]), float(before[1])
    am, aa = float(after[0]), float(after[1])

    fig = plt.figure(figsize=(4.3, 4.3), dpi=200)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    for mag, ang, color, lbl in (
        (bm, ba, "#ef4444", f"Antes: {bm:.3f} ∠ {ba:.0f}°"),
        (am, aa, "#22c55e", f"Después: {am:.3f} ∠ {aa:.0f}°"),
    ):
        th = np.deg2rad(ang)
        ax.annotate("", xy=(th, mag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
        ax.plot([th], [mag], "o", color=color, markersize=5, label=lbl)

    ax.set_rmax(max(bm, am, 1e-6) * 1.18)
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.4)
    ax.set_title(f"{title}  [{unit}]", fontsize=9, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              fontsize=7, ncol=2, frameon=False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =====================================================================
# Helpers de flowables
# =====================================================================
def _p(text: str, styles, style: str = "WMBody"):
    return Paragraph(str(text), styles[style])


def _section(title: str, styles):
    # WMTOC1 hace que el heading entre a la Tabla de Contenido.
    return Paragraph(title, styles["WMTOC1"])


def _kv_table(rows: List[Tuple[str, str]], styles) -> Table:
    data = [[Paragraph(f"<b>{k}</b>", styles["WMTableCell"]),
             Paragraph(str(v), styles["WMTableCell"])] for k, v in rows]
    t = Table(data, colWidths=[5.2 * cm, 11.0 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _grid_table(headers: List[str], rows: List[List[Any]], styles,
                col_widths: Optional[List[float]] = None) -> Table:
    head = [Paragraph(f"<b>{h}</b>", styles["WMTableHeader"]) for h in headers]
    body = [[Paragraph(str(c), styles["WMTableCell"]) for c in r] for r in rows]
    t = Table([head] + body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_HEADER_BG)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _fmt(x: Any, nd: int = 2) -> str:
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _vec(v: Optional[Tuple[float, float]], unit: str) -> str:
    if not v:
        return "—"
    return f"{_fmt(v[0], 3)} ∠ {_fmt(v[1], 1)}°  {unit}"


# =====================================================================
# Constructor principal
# =====================================================================
def build_balance_pdf(
    *,
    meta: Dict[str, Any],
    one_plane: Optional[Dict[str, Any]] = None,
    two_plane: Optional[Dict[str, Any]] = None,
    iso: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Arma el PDF de balanceo. Secciones opcionales.

    meta: asset, client, location, specialist, unit, rpm, notes, report_date...
    one_plane: {"unit", "v0","trial","vt","vf"(opc.) : (mag,ang), "result": dict}
    two_plane: {"unit", "a0","b0","a1","b1","a2","b2","wa","wb": (mag,ang),
                "result": dict}
    iso: salida de evaluate_iso_grades.
    """
    styles = make_styles()
    body: List[Any] = []

    unit = meta.get("unit", "µm pk-pk")

    # ---- Datos generales ----------------------------------------------
    body.append(_section("1. Datos del balanceo", styles))
    body.append(_kv_table([
        ("Activo", meta.get("asset", "—")),
        ("Cliente", meta.get("client", "—")),
        ("Sitio / ubicación", meta.get("location", "—")),
        ("Especialista", meta.get("specialist", "—")),
        ("Fecha", meta.get("report_date") or datetime.now().strftime("%d/%m/%Y")),
        ("Velocidad", f"{_fmt(meta.get('rpm'), 0)} rpm"),
        ("Unidad de vibración", unit),
        ("Norma", "ISO 21940-11 / 21940-12 · API 684"),
    ], styles))
    if meta.get("notes"):
        body.append(Spacer(1, 0.3 * cm))
        body.append(_p(meta["notes"], styles, "WMBody"))
    body.append(Spacer(1, 0.5 * cm))

    # ---- 1 plano ------------------------------------------------------
    if one_plane and one_plane.get("result"):
        r = one_plane["result"]
        u = one_plane.get("unit", unit)
        body.append(_section("2. Balanceo en 1 plano", styles))
        body.append(_p("Método: coeficiente de influencia — "
                       "H = (Vt − V0) / Wt · Wcorr = −V0 / H.", styles, "WMBody"))
        body.append(_grid_table(
            ["Medición", "Vector"],
            [["V0 — inicial", _vec(one_plane.get("v0"), u)],
             ["Peso de prueba", _vec(one_plane.get("trial"), "g")],
             ["Vt — con peso de prueba", _vec(one_plane.get("vt"), u)],
             ["Vf — final medida", _vec(one_plane.get("vf"), u)]],
            styles, col_widths=[7.0 * cm, 9.2 * cm]))
        body.append(Spacer(1, 0.3 * cm))
        body.append(_grid_table(
            ["Resultado", "Valor"],
            [["Peso de corrección", f"{_fmt(r['corr_mass_g'])} g"],
             ["Ángulo de corrección", f"{_fmt(r['corr_ang_deg'], 1)}°"],
             ["Vibración residual estimada", f"{_fmt(r['pred_mag'], 3)} {u}"],
             ["Calidad del modelo", r.get("quality", "—")]],
            styles, col_widths=[7.0 * cm, 9.2 * cm]))
        v0 = one_plane.get("v0")
        if v0:
            after = one_plane.get("vf") or (r.get("pred_mag"), r.get("pred_ang"))
            png = polar_png("Vector 1 plano (antes / después)", v0, after, u)
            if png:
                body.append(Spacer(1, 0.3 * cm))
                body.append(Image(io.BytesIO(png), width=8.5 * cm, height=8.5 * cm))
        body.append(Spacer(1, 0.5 * cm))

    # ---- 2 planos -----------------------------------------------------
    if two_plane and two_plane.get("result"):
        r = two_plane["result"]
        u = two_plane.get("unit", unit)
        wa_m = _fmt(_to_mag(r.get("WA_corr"))[0]); wa_a = _fmt(_to_mag(r.get("WA_corr"))[1], 1)
        wb_m = _fmt(_to_mag(r.get("WB_corr"))[0]); wb_a = _fmt(_to_mag(r.get("WB_corr"))[1], 1)
        body.append(_section("2. Balanceo en 2 planos", styles))
        body.append(_p("Método: matriz de coeficientes de influencia (2×2), "
                       "ISO 21940-12.", styles, "WMBody"))
        body.append(_grid_table(
            ["Corrida", "Sonda A", "Sonda B"],
            [["0 — inicial", _vec(two_plane.get("a0"), u), _vec(two_plane.get("b0"), u)],
             ["1 — trial en A", _vec(two_plane.get("a1"), u), _vec(two_plane.get("b1"), u)],
             ["2 — trial en B", _vec(two_plane.get("a2"), u), _vec(two_plane.get("b2"), u)]],
            styles, col_widths=[4.0 * cm, 6.1 * cm, 6.1 * cm]))
        body.append(Spacer(1, 0.2 * cm))
        body.append(_grid_table(
            ["Pesos de prueba", "Plano A", "Plano B"],
            [["", _vec(two_plane.get("wa"), "g"), _vec(two_plane.get("wb"), "g")]],
            styles, col_widths=[4.0 * cm, 6.1 * cm, 6.1 * cm]))
        body.append(Spacer(1, 0.3 * cm))
        body.append(_grid_table(
            ["Corrección", "Plano A", "Plano B"],
            [["Peso", f"{wa_m} g", f"{wb_m} g"],
             ["Ángulo", f"{wa_a}°", f"{wb_a}°"],
             ["Residual estimado",
              f"{_fmt(abs(r.get('A_after', 0)), 3)} {u}",
              f"{_fmt(abs(r.get('B_after', 0)), 3)} {u}"]],
            styles, col_widths=[4.0 * cm, 6.1 * cm, 6.1 * cm]))
        body.append(Spacer(1, 0.2 * cm))
        body.append(_p(f"Calidad del modelo: <b>{r.get('quality', '—')}</b> · "
                       f"cond(M) = {_fmt(r.get('cond'), 1)}", styles, "WMBody"))
        body.append(Spacer(1, 0.5 * cm))

    # ---- Validación ISO ----------------------------------------------
    if iso and iso.get("results"):
        n = 3 if (one_plane or two_plane) else 2
        body.append(_section(f"{n}. Validación ISO 21940-11", styles))
        body.append(_p(f"Desbalance residual U_res = "
                       f"{_fmt(iso.get('U_res'), 1)} g·mm · "
                       f"<b>{iso.get('summary_label', '')}</b>", styles, "WMBody"))
        rows = []
        for g in iso["results"]:
            rows.append([
                f"G{g['G']:g}", _fmt(g["e_per"], 3), _fmt(g["U_per"], 1),
                (_fmt(g["ratio"], 2) if g["ratio"] < 900 else "—"),
                "Cumple" if g["pass"] else "No cumple",
            ])
        body.append(_grid_table(
            ["Grado", "e_per [µm]", "U_per [g·mm]", "U_res/U_per", "Estado"],
            rows, styles,
            col_widths=[2.6 * cm, 3.4 * cm, 3.8 * cm, 3.2 * cm, 3.2 * cm]))
        body.append(Spacer(1, 0.5 * cm))

    body.append(_p(
        "Reporte generado por Watermelon System · módulo Balanceo. Cálculo por "
        "coeficiente de influencia bajo ISO 21940-11/12 y API 684. La convención "
        "angular es de campo (0° en TDC).", styles, "WMBody"))

    report_meta = {
        "report_title": meta.get("report_title") or "Reporte de Balanceo",
        "format_code": "WM-BAL",
        "asset": meta.get("asset", ""),
        "asset_class": "Balanceo de rotor",
        "client": meta.get("client", ""),
        "location": meta.get("location", ""),
        "unit": unit,
        "prepared_by": meta.get("specialist", ""),
        "prepared_role": meta.get("specialist_role", "Analista de vibraciones"),
        "prepared_city": meta.get("location", ""),
        "report_date": meta.get("report_date") or datetime.now().strftime("%d/%m/%Y"),
        "train_description": meta.get("train_description", ""),
    }
    return render_report_pdf(report_meta, body)


def _to_mag(z: Any) -> Tuple[float, float]:
    """(mag, ang°) de un complejo; (0,0) si no aplica."""
    try:
        import numpy as np
        return float(abs(z)), float(np.rad2deg(np.angle(z)) % 360.0)
    except Exception:
        return 0.0, 0.0


__all__ = ["build_balance_pdf", "polar_png"]
