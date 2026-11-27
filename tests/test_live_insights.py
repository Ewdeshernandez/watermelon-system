from datetime import datetime, timedelta, timezone

from core import live_insights as LI

NOW = datetime(2026, 11, 27, 14, 0, tzinfo=timezone.utc)


def _daily(vals):
    """vals[0] = hace len-1 días … vals[-1] = hoy."""
    n = len(vals)
    return [{"bucket": (NOW - timedelta(days=n - 1 - i)).replace(hour=0).isoformat(), "avg_val": v}
            for i, v in enumerate(vals)]


def test_trend_delta_up_down_flat_and_stopped_days():
    up = LI.trend_delta(_daily([50] * 20 + [50, 50, 50, 50, 50, 58, 60, 62, 63, 64]), now=NOW)
    assert up["dir"] == "up" and up["pct"] > 8
    down = LI.trend_delta(_daily([60] * 22 + [55, 50, 48, 46, 45, 44, 44, 43]), now=NOW)
    assert down["dir"] == "down"
    flat = LI.trend_delta(_daily([57.0] * 30), now=NOW)
    assert flat["dir"] == "flat" and flat["pct"] == 0.0
    # días con la máquina parada (≈0) no cuentan como caída
    stopped = LI.trend_delta(_daily([57.0] * 28 + [0.4, 57.2]), now=NOW)
    assert stopped["dir"] == "flat" and 0.4 not in stopped["spark"]
    assert LI.trend_delta([], now=NOW) == {"spark": [], "pct": None, "dir": "flat"}


def _hourly(states):
    return [{"bucket": (NOW - timedelta(hours=len(states) - i)).isoformat(), "avg_val": v}
            for i, v in enumerate(states)]


def test_operation_stats_running_starts_and_nodata():
    rpm = [0] * 10 + [14000] * 100 + [5000] + [0] * 5 + [14050] * 50
    st = LI.operation_stats(_hourly(rpm), days=30, now=NOW)
    assert st["hours_running"] == 150 and st["starts"] == 2 and st["stops"] == 1
    assert st["normal_rpm"] == 14000 or st["normal_rpm"] == 14050
    assert st["hours_nodata"] == 30 * 24 - len(rpm)
    assert round(st["pct_running"]) == round(150 / len(rpm) * 100)
    assert LI.operation_stats([], now=NOW) is None
    idle = LI.operation_stats(_hourly([0] * 24), now=NOW)
    assert idle["hours_running"] == 0 and idle["normal_rpm"] is None


def test_main_speed_variable():
    latest = [{"metric": "Direct", "variable": "Velocidad Generador", "value": 1808},
              {"metric": "Direct", "variable": "Velocidad Turbina", "value": 14072},
              {"metric": "Direct", "variable": "1YD Turbina DE", "value": 57}]
    assert LI.main_speed_variable(latest) == "Velocidad Turbina"
    assert LI.main_speed_variable([]) is None


def test_recommendations_lifecycle_and_order():
    recs = LI.add_recommendation([], "Revisar bomba 4YV", "a@sigasas.com", "REP-015")
    recs = LI.add_recommendation(recs, "Alinear acople", "a@sigasas.com")
    assert [r["status"] for r in recs] == ["pending", "pending"]
    rid = recs[0]["id"]
    recs = LI.set_rec_status(recs, rid, "done", "b@sigasas.com")
    assert recs[0]["status"] == "done" and recs[0]["by"] == "b@sigasas.com"
    assert LI.set_rec_status(recs, rid, "bogus", "x")[0]["status"] == "done"
    assert [r["text"] for r in LI.sort_recs(recs)] == ["Alinear acople", "Revisar bomba 4YV"]
