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
def nomogram_png(W_kg: float, U_res_gmm: float, rpm: float) -> Optional[bytes]:
    """Nomograma ISO 21940-11 (log-log): líneas de grado G (e_per = 9549·G/n) +
    el punto de operación del rotor. PNG o None si no hay matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from core.balance.engine import iso_nomogram_lines, iso_operating_point
    except Exception:  # noqa: BLE001
        return None
    try:
        nl = iso_nomogram_lines(rpm_min=100.0, rpm_max=60000.0)
        op = iso_operating_point(W_kg, U_res_gmm, rpm)
        fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=130)
        for g, ys in nl["lines"].items():
            ax.loglog(nl["rpm"], ys, lw=1.1, label=g)
        ax.loglog([op["rpm"]], [max(op["e_um"], 1e-3)], marker="o", ms=9,
                  color="#e8890c", mec="#7a4a00", mew=1.2, ls="", label="Rotor")
        ax.annotate("rotor", (op["rpm"], max(op["e_um"], 1e-3)),
                    textcoords="offset points", xytext=(6, 6), color="#7a4a00",
                    fontsize=8, fontweight="bold")
        ax.set_xlabel("Velocidad de servicio [rpm]", fontsize=9)
        ax.set_ylabel("Excentricidad permisible e_per [µm]", fontsize=9)
        ax.grid(True, which="both", color="#e2e8f2", lw=0.5)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.set_title("Nomograma ISO 21940-11", fontsize=10, color="#12305e")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


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
def _split_flow(W_mag, W_ang, positions, styles):
    """Devuelve flowables con el reparto de la corrección a posiciones instalables
    (álabes/buckets o huecos), o [] si el modo es ángulo libre."""
    mode = (positions or {}).get("pos_mode", "Ángulo libre")
    if not mode or mode == "Ángulo libre" or not W_mag or W_mag <= 0:
        return []
    try:
        from core.balance.engine import (split_to_buckets, split_to_positions,
                                          combine_weights)
    except Exception:  # noqa: BLE001
        return []
    if str(mode).startswith("Álabes"):
        res = split_to_buckets(W_mag, W_ang, int(positions.get("n_buckets") or 24),
                               float(positions.get("pos_offset") or 0.0),
                               bool(positions.get("pos_cw", False)))
        rows = [[f"#{o.get('bucket')}", f"{_fmt(o['angle'], 1)}°",
                 f"{_fmt(o['mass'], 2)} g"] for o in res]
        head = ["Álabe / Bucket", "Ángulo", "Peso a instalar"]
    else:
        step = float(positions.get("hole_step") or 30.0)
        off = float(positions.get("pos_offset") or 0.0)
        poss = [(off + i * step) % 360.0 for i in range(int(round(360.0 / step)))]
        res = split_to_positions(W_mag, W_ang, poss)
        rows = [[f"{_fmt(o['angle'], 1)}°", f"{_fmt(o['mass'], 2)} g"] for o in res]
        head = ["Posición (hueco)", "Peso a instalar"]
    if not res:
        return []
    _rc = combine_weights([(o["mass"], o["angle"]) for o in res])
    out = [_p(f"<b>Reparto a posiciones instalables</b> (corrección "
              f"{_fmt(W_mag, 2)} g ∠ {_fmt(W_ang, 1)}°):", styles, "WMBody"),
           _grid_table(head, rows, styles,
                       col_widths=([4.0 * cm] * len(head)) if len(head) == 3
                       else [8.1 * cm, 8.1 * cm]),
           _p(f"Verificación: suma vectorial = {_fmt(_rc[0], 2)} g ∠ "
              f"{_fmt(_rc[1], 1)}° (igual a la corrección).", styles, "WMBody"),
           Spacer(1, 0.3 * cm)]
    return out


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
        ("Sentido de giro", f"{meta.get('rotation', 'CCW')} · ángulos medidos "
                            "contra el sentido de giro"),
        ("Fuente de datos", meta.get("data_source", "Datos manuales (escritos)")),
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
        _vf1 = one_plane.get("vf")
        if v0 and _vf1 and _vf1[0] and _vf1[0] > 0:
            _ch1 = (v0[0] - _vf1[0]) / max(1e-9, v0[0]) * 100.0
            _t1 = (f"Vibración final medida: <b>{_fmt(_vf1[0], 3)} {u}</b> · "
                   f"cambio {_ch1:+.0f}% (V0 {_fmt(v0[0], 3)} {u})")
            if _ch1 < 0:
                _t1 += " — <b>EL PLANO EMPEORÓ</b>, no es una mejora."
            body.append(Spacer(1, 0.2 * cm))
            body.append(_p(_t1, styles, "WMBody"))
        for _f in _split_flow(r.get("corr_mass_g"), r.get("corr_ang_deg"),
                              meta.get("positions"), styles):
            body.append(_f)
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
        # Vibración final medida (si existe): muestra el cambio honesto por plano.
        _vfa = two_plane.get("vf_a"); _vfb = two_plane.get("vf_b")
        _a0 = two_plane.get("a0"); _b0 = two_plane.get("b0")
        _has_final = ((_vfa and _vfa[0] and _vfa[0] > 0) or (_vfb and _vfb[0] and _vfb[0] > 0))
        if _has_final:
            def _chg_cell(v0, vf):
                if not (v0 and vf and vf[0] and vf[0] > 0):
                    return "—"
                _c = (v0[0] - vf[0]) / max(1e-9, v0[0]) * 100.0
                _s = f"{_fmt(vf[0], 3)} {u} ({_c:+.0f}%)"
                return _s + (" EMPEORÓ" if _c < 0 else "")
            body.append(Spacer(1, 0.2 * cm))
            body.append(_grid_table(
                ["Medición final", "Plano A (sonda A)", "Plano B (sonda B)"],
                [["Inicial V0", _vec(_a0, u), _vec(_b0, u)],
                 ["Final medida (cambio)", _chg_cell(_a0, _vfa), _chg_cell(_b0, _vfb)]],
                styles, col_widths=[4.0 * cm, 6.1 * cm, 6.1 * cm]))
        body.append(Spacer(1, 0.2 * cm))
        body.append(_p(f"Calidad del modelo: <b>{r.get('quality', '—')}</b> · "
                       f"cond(M) = {_fmt(r.get('cond'), 1)}", styles, "WMBody"))
        _wa_mag, _wa_ang = _to_mag(r.get("WA_corr"))
        _wb_mag, _wb_ang = _to_mag(r.get("WB_corr"))
        for _plabel, _wm, _wa in (("Plano A", _wa_mag, _wa_ang),
                                  ("Plano B", _wb_mag, _wb_ang)):
            _rows = _split_flow(_wm, _wa, meta.get("positions"), styles)
            if _rows:
                body.append(_p(f"<b>{_plabel}</b>", styles, "WMBody"))
                for _f in _rows:
                    body.append(_f)
        # Diagramas polares por sonda (vibración antes/después) — como en 1 plano.
        _imgs = []
        for _lbl, _v0key, _afterkey, _vfkey in (("A", "a0", "A_after", "vf_a"),
                                                ("B", "b0", "B_after", "vf_b")):
            _vf = two_plane.get(_vfkey)
            _after = _vf if (_vf and _vf[0] and _vf[0] > 0) else _to_mag(r.get(_afterkey))
            _ptitle = (f"Plano {_lbl} — sonda {_lbl} (antes / final medido)"
                       if (_vf and _vf[0] and _vf[0] > 0)
                       else f"Plano {_lbl} — sonda {_lbl} (antes / después)")
            _png = polar_png(_ptitle, two_plane.get(_v0key), _after, u)
            if _png:
                _imgs.append(Image(io.BytesIO(_png), width=7.6 * cm, height=7.6 * cm))
        if _imgs:
            body.append(Spacer(1, 0.3 * cm))
            if len(_imgs) == 2:
                _it = Table([[_imgs[0], _imgs[1]]], colWidths=[8.1 * cm, 8.1 * cm])
                _it.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                         ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
                body.append(_it)
            else:
                body.append(_imgs[0])
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
        _npng = nomogram_png(float(iso.get("W_kg") or 0), float(iso.get("U_res") or 0),
                             float(iso.get("N_rpm") or meta.get("rpm") or 0))
        if _npng:
            body.append(Spacer(1, 0.3 * cm))
            body.append(Image(io.BytesIO(_npng), width=13.0 * cm, height=8.7 * cm))
        body.append(Spacer(1, 0.5 * cm))

    # ---- Avisos de auditoría (guards) --------------------------------
    _warns = list((one_plane or {}).get("warnings") or [])
    _warns += list((two_plane or {}).get("warnings") or [])
    if _warns:
        _n2 = (4 if iso and iso.get("results") else 3) if (one_plane or two_plane) else 2
        body.append(_section(f"{_n2}. Validación técnica y advertencias", styles))
        body.append(_p(
            "Chequeos automáticos de confiabilidad del balanceo. Un aviso CRÍTICO "
            "indica que la corrección puede no ser válida (respuesta al peso de "
            "prueba insuficiente, corrección desproporcionada, o un plano que "
            "empeoró) — revisar antes de aceptar el servicio como exitoso.",
            styles, "WMBody"))
        _lab = {"crit": "CRÍTICO", "warn": "REVISAR", "info": "NOTA"}
        _wrows = [[_lab.get(w.get("severity"), "REVISAR"),
                   f"{w.get('title','')}. {w.get('msg','')}"] for w in _warns]
        body.append(_grid_table(["Nivel", "Hallazgo"], _wrows, styles,
                                col_widths=[2.8 * cm, 13.4 * cm]))
        body.append(Spacer(1, 0.5 * cm))

    body.append(_p(
        "Reporte generado por Watermelon System · módulo Balanceo. Cálculo por "
        "coeficiente de influencia bajo ISO 21940-11/12 y API 684. Convención "
        "angular de campo: 0° en TDC, ángulos medidos contra el sentido de giro "
        f"({meta.get('rotation', 'CCW')}). El método es convención-agnóstico: el "
        "coeficiente de influencia se obtiene de la corrida de prueba, por lo que "
        "el ángulo de corrección queda en la misma convención de los datos.",
        styles, "WMBody"))

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


__all__ = ["build_balance_pdf", "polar_png", "nomogram_png"]
