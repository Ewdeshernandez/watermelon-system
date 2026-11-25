from core.client_report_archive import (
    KIND_MONTHLY, KIND_TECH, KIND_WEEKLY, download_name, filter_reports,
    group_by_month, machine_of, month_label, report_date, report_kind, sev_class,
)


def _sc(**rm):
    extra = rm.pop("_notes", "")
    archived = rm.pop("_archived", "2026-11-23T12:57:36")
    return {"archive_id": f"x/{rm.get('consecutive', 'a')}", "archived_at": archived,
            "extra_notes": extra, "report_meta": rm}


def test_report_kind_from_period_title_and_notes():
    assert report_kind(_sc(period="Mensual")) == KIND_MONTHLY
    assert report_kind(_sc(asset_class="Reporte Semanal · SGT300B")) == KIND_WEEKLY
    assert report_kind(_sc(consecutive="REPORTE MENSUAL TES1- JUNIO 2026")) == KIND_MONTHLY
    assert report_kind(_sc(_notes="Aprobado en Report Center (Semanal).")) == KIND_WEEKLY
    assert report_kind(_sc(asset_class="GE LM6000")) == KIND_TECH


def test_sev_class():
    assert sev_class("Operación normal") == "ok"
    assert sev_class("Alerta") == "warn"
    assert sev_class("Peligro") == "dang"
    assert sev_class("Máquina detenida") == "off"
    assert sev_class("") == "off"


def test_date_machine_and_label():
    sc = _sc(report_date="2026-10-06", asset_class="Reporte Semanal · SGT300A")
    assert report_date(sc) == "2026-10-06"
    assert machine_of(sc) == "SGT300A"
    assert machine_of(_sc(instance_tag="SGT300B")) == "SGT300B"
    assert report_date(_sc(_archived="2026-09-01T10:00:00")) == "2026-09-01"
    assert month_label("2026-11-01") == "Noviembre 2026"


def test_group_and_filter():
    a = _sc(consecutive="A-1", report_date="2026-11-23", instance_tag="SGT300B", period="Semanal")
    b = _sc(consecutive="A-2", report_date="2026-11-16", instance_tag="SGT300A", period="Semanal")
    c = _sc(consecutive="M-1", report_date="2026-10-01", instance_tag="SGT300B", period="Mensual")
    groups = group_by_month([b, c, a])
    assert [g[0] for g in groups] == ["2026-11", "2026-10"]
    assert [s["report_meta"]["consecutive"] for s in groups[0][1]] == ["A-1", "A-2"]
    assert filter_reports([a, b, c], kind=KIND_MONTHLY) == [c]
    assert filter_reports([a, b, c], machine="SGT300A") == [b]
    assert filter_reports([a, b, c], year="2025") == []


def test_download_name_is_safe():
    assert download_name(_sc(consecutive="SIGA-REP-TEC-WM-SGT300B-015")) == \
        "SIGA-REP-TEC-WM-SGT300B-015.pdf"
    n = download_name(_sc(instance_tag="SGT 300/B", period="Semanal", report_date="2026-11-23"))
    assert "/" not in n and n.endswith(".pdf")


def test_sidecar_matches_client_rule():
    from types import SimpleNamespace
    from core.reports_archive import sidecar_matches_client
    acme = SimpleNamespace(match_strings=("Acme Energy",), asset_tags=("TX1",))
    assert sidecar_matches_client(_sc(client="Acme Energy", asset_class="Reporte"), acme)
    assert sidecar_matches_client(_sc(instance_tag="TX1"), acme)
    assert not sidecar_matches_client(_sc(client="Other Co", instance_tag="TX9"), acme)
    assert not sidecar_matches_client(_sc(), SimpleNamespace(match_strings=("",), asset_tags=()))
