from datetime import datetime, timedelta, timezone

from core import live_portal as LP
from core.portal_activity import summarize_by_user

NOW = datetime(2026, 11, 27, 14, 0, tzinfo=timezone.utc)


def _row(label, value, unit, alarm, plane, status="Normal"):
    return {"sensor_label": label, "value": value, "unit": unit, "alarm_used": alarm,
            "danger_used": alarm * 1.4, "plane_label": plane, "status": status}


ROWS = [
    _row("1X_D", 57.3, "µm pp", 66, "1XD DE turbina"),
    _row("1Y_D", 57.6, "µm pp", 66, "1YD DE turbina"),
    _row("4Y_V", 4.84, "mm/s rms", 5, "4YV gearbox bomba"),
    _row("4X_A", 5.01, "g rms", 7, "4XA gearbox starter"),
    _row("4Y_D", 11.3, "µm pp", 50, "4YD gearbox"),
    _row("5Y_D", 60.7, "µm pp", 76, "5YD DE generador"),
]


def test_tags_components_and_order():
    assert LP.tag_of("1Y_D") == "1YD"
    assert LP.component_of("4YV", "4YV gearbox bomba") == "aux"
    assert LP.component_of("4YD", "4YD gearbox") == "gearbox"
    assert LP.component_of("5YD", "") == "generator"
    groups = LP.point_groups(ROWS)
    assert [g["key"] for g in groups] == ["turbine", "gearbox", "generator", "aux"]
    assert [p["tag"] for p in groups[0]["points"]] == ["1YD", "1XD"]          # Y antes que X
    assert groups[0]["points"][0]["pct"] == 87


def test_friendly_and_hottest():
    assert LP.friendly_point("1YD", "1YD DE turbina", "es") == "Cojinete 1 de turbina"
    assert LP.friendly_point("4YV", "4YV gearbox bomba", "en") == "Lube oil pump"
    assert LP.hottest_point(ROWS)["tag"] == "4YV"                              # 97 %


def _ev(tag, frm, to, value, hours_ago):
    return {"sensor_label": tag, "from_status": frm, "to_status": to, "value": value,
            "unit": "mm/s rms", "alarm": 5.0, "danger": 7.0,
            "crossed_at": (NOW - timedelta(hours=hours_ago)).isoformat()}


def test_summarize_events_chattering_and_window():
    evs = []
    for i in range(6):
        evs.append(_ev("4Y_V", "Normal", "Alarma", 5.0 + i * 0.05, 10 + i * 5))
        evs.append(_ev("4Y_V", "Alarma", "Normal", 4.9, 9 + i * 5))
    evs.append(_ev("1Y_D", "Normal", "Alarma", 66.9, 24 * 10))              # fuera de 7 d
    evs.append(_ev("1Y_D", "Alarma", "Normal", 65.0, 24 * 10 - 1))
    g = LP.summarize_events(evs, now=NOW)
    assert [x["tag"] for x in g] == ["4YV"]
    assert g[0]["n_up"] == 6 and g[0]["chattering"] and not g[0]["active"]
    assert abs(g[0]["max_value"] - 5.25) < 1e-9


def test_week_lines_note_events_and_stable():
    evs = [_ev("4Y_V", "Normal", "Alarma", 5.25, 5), _ev("4Y_V", "Alarma", "Normal", 4.9, 4)]
    lines = LP.week_lines(LP.summarize_events(evs, now=NOW), LP.point_groups(ROWS),
                          {"text": "Revisaremos la bomba <el lunes>"}, "es")
    assert lines[0]["kind"] == "note" and "&lt;el lunes&gt;" in lines[0]["html"]
    assert "Bomba de lubricación (4YV)" in lines[1]["html"] and "Hoy está normal" in lines[1]["html"]
    assert lines[-1]["kind"] == "ok" and "Turbina, Caja y Generador" in lines[-1]["html"]
    quiet = LP.week_lines([], LP.point_groups(ROWS), None, "en")
    assert quiet == [{"kind": "ok", "html": "<b>No alarm crossings in the last 7 days.</b> Stable operation."}]


def test_verdict_and_health_word():
    v = LP.verdict("SGT300B", {"Normal": 16}, no_data=False, stale=False, stopped=False, n_points=16, lang="es")
    assert v["tone"] == "ok" and v["title"] == "SGT300B opera normal"
    v = LP.verdict("SGT300B", {"Alarma": 2}, no_data=False, stale=False, stopped=False, n_points=16, lang="en")
    assert v["tone"] == "warn" and "2 points in alarm" in v["title"]
    assert LP.verdict("A", {}, no_data=True, stale=False, stopped=False, n_points=0, lang="es")["tone"] == "off"
    assert LP.health_word(100, "es") == "Excelente" and LP.health_word(None, "en") == "No data"


def test_next_report_at():
    sched = [{"enabled": True, "period": "Semanal", "slots": [[0, 6]]}]       # lunes 06:00
    fri = datetime(2026, 11, 27, 9, 0)                                         # viernes
    assert LP.next_report_at(sched, fri) == datetime(2026, 11, 30, 6, 0)
    mon_late = datetime(2026, 11, 30, 7, 0)
    assert LP.next_report_at(sched, mon_late) == datetime(2026, 12, 7, 6, 0)
    assert LP.next_report_at([{"enabled": False, "period": "Semanal", "slots": [[0, 6]]}], fri) is None
    assert LP.fmt_day(datetime(2026, 11, 30), "es", weekday=True) == "lunes 30 nov"


def test_attention_items_order():
    fleet = [
        {"iid": "a", "tag": "SGT300A", "state": "nodata", "age_txt": ""},
        {"iid": "b", "tag": "SGT300B", "state": "online", "n_alarm": 0, "n_danger": 0,
         "hottest": {"tag": "4YV", "pct": 97}},
    ]
    evb = {"b": [{"tag": "4YV", "chattering": True, "n_up": 30, "max_value": 5.25,
                  "unit": "mm/s", "alarm": 5.0}]}
    items = LP.attention_items(fleet, evb, [("b", "SGT300B", {"period": "Semanal"})])
    assert [i["color"] for i in items] == ["dang", "warn", "warn", "info"]
    assert items[0]["title"].startswith("SGT300A · no data")


def test_summarize_activity_by_user():
    rows = [{"at": "2026-11-27T10:00:00+00:00", "email": "a@x.com", "event": "view_live"},
            {"at": "2026-11-27T11:00:00+00:00", "email": "a@x.com", "event": "download_live_report"},
            {"at": "2026-11-26T09:00:00+00:00", "email": "b@x.com", "event": "view_reports"}]
    out = summarize_by_user(rows)
    assert out[0]["email"] == "a@x.com" and out[0]["visits"] == 1 and out[0]["downloads"] == 1
    assert out[0]["last_event"] == "download_live_report"
