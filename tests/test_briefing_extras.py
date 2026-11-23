from datetime import datetime, timedelta, timezone

from core.briefing_extras import (
    channel_stats, operation_summary, pct_change, running_hours,
    summarize_events, watch_points,
)

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _h(i):
    return (T0 + timedelta(hours=i)).isoformat()


def _rpm(values):
    return [{"bucket": _h(i), "avg_val": v} for i, v in enumerate(values)]


def test_running_hours_uses_90pct_of_normal_speed():
    pts = _rpm([0, 0, 5000, 14000, 14000, 14000, 12000, 0])
    running, normal = running_hours(pts)
    assert normal == 14000
    assert running == {T0 + timedelta(hours=i) for i in (3, 4, 5)}


def test_running_hours_never_turning():
    assert running_hours(_rpm([0, 0, 50])) == (set(), 0.0)


def test_operation_summary_counts_starts_stops_availability():
    op = operation_summary(_rpm([0, 14000, 14000, 0, 0, 14000, 14000, 14000]))
    assert op["starts"] == 2
    assert op["stops"] == 1
    assert op["hours_running"] == 5
    assert op["hours_stopped"] == 3
    assert round(op["availability"]) == 62


def test_channel_stats_min_and_avg_only_in_operation():
    running = {T0 + timedelta(hours=i) for i in (1, 2, 3)}
    pts = [
        {"bucket": _h(0), "max_val": 2.0, "avg_val": 0.1},   # parada: no cuenta
        {"bucket": _h(1), "max_val": 9.0, "avg_val": 4.0},
        {"bucket": _h(2), "max_val": 6.0, "avg_val": 5.0},
        {"bucket": _h(3), "max_val": 7.0, "avg_val": 6.0},
    ]
    s = channel_stats(pts, running)
    assert s["max"] == 9.0
    assert s["avg"] == 5.0
    assert s["min"] == 4.0
    assert s["running"] is True


def test_channel_stats_without_rpm_uses_whole_period():
    pts = [{"bucket": _h(0), "max_val": 2.0, "avg_val": 1.0},
           {"bucket": _h(1), "max_val": 3.0, "avg_val": 3.0}]
    s = channel_stats(pts, set())
    assert s["min"] == 1.0 and s["running"] is False
    assert channel_stats([], set()) is None


def test_pct_change():
    assert pct_change(12.0, 10.0) == 20.0
    assert pct_change(5.0, 0) is None
    assert pct_change(None, 3.0) is None


def test_summarize_events_counts_only_entries_into_alarm():
    ev = [
        {"sensor_label": "4Y_V", "from_status": "Normal", "to_status": "Alarma",
         "value": 5.2, "unit": "mm/s", "crossed_at": _h(1)},
        {"sensor_label": "4Y_V", "from_status": "Alarma", "to_status": "Normal",
         "value": 4.8, "unit": "mm/s", "crossed_at": _h(2)},
        {"sensor_label": "4Y_V", "from_status": "Normal", "to_status": "Alarma",
         "value": 5.6, "unit": "mm/s", "crossed_at": _h(3)},
        {"sensor_label": "1Y_D", "from_status": "Alarma", "to_status": "Danger",
         "value": 90.0, "unit": "µm pp", "crossed_at": _h(4)},
    ]
    out = summarize_events(ev)
    assert out[0]["display"] == "1YD" and out[0]["worst"] == "Danger"
    v = out[1]
    assert v["display"] == "4YV" and v["count"] == 2 and v["max"] == 5.6
    assert v["last"] == T0 + timedelta(hours=3)


def test_watch_points_top3_by_pct():
    ch = [{"sensor_label": s, "pct": p} for s, p in
          (("a", 50), ("b", None), ("c", 105), ("d", 88), ("e", 71))]
    assert [c["sensor_label"] for c in watch_points(ch)] == ["c", "d", "e"]
