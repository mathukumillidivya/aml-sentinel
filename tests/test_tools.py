"""Each test builds a tiny hand-made scenario where the right answer is obvious."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ledger import Ledger  # noqa: E402
import tools  # noqa: E402

ALERT = "2026-02-10"


def make(rows, with_labels=False):
    df = pd.DataFrame(rows, columns=["timestamp", "from_account", "to_account", "amount", "payment_type"])
    df["timestamp"] = pd.to_datetime(df.timestamp)
    df.insert(0, "txn_id", [f"T{i}" for i in range(len(df))])
    if with_labels:
        df["is_laundering"], df["pattern_id"], df["pattern_type"] = 1, "P1", "cycle"
    return Ledger(df)


def test_labels_are_hidden_from_tools():
    led = make([("2026-02-10 09:00", "A", "B", 5000, "wire")], with_labels=True)
    assert not set(led.tx.columns) & {"is_laundering", "pattern_id", "pattern_type"}
    assert "is_laundering" not in led.activity("A").columns


def test_structuring_detected():
    led = make([(f"2026-02-0{d} 10:00", "CASH", "A", 9500, "cash") for d in (6, 7, 8, 9)])
    r = tools.check_structuring(led, "A", ALERT)
    assert r["deposits_just_under_limit"] == 4 and r["just_under_share"] == 1.0


def test_ordinary_cash_business_not_structuring():
    led = make([(f"2026-02-0{d} 10:00", "CASH", "A", amt, "cash") for d, amt in ((6, 2100), (7, 3400), (8, 1800))])
    assert tools.check_structuring(led, "A", ALERT)["deposits_just_under_limit"] == 0


def test_cycle_found():
    led = make([("2026-02-09 09:00", "A", "B", 50000, "wire"), ("2026-02-09 13:00", "B", "C", 49500, "wire"),
                ("2026-02-09 18:00", "C", "D", 49000, "wire"), ("2026-02-10 08:00", "D", "A", 48500, "wire")])
    r = tools.find_cycles(led, "A", ALERT)
    assert r["cycles_found"] == 1 and r["cycles"][0]["path"] == ["A", "B", "C", "D", "A"]


def test_no_cycle_when_money_does_not_return():
    led = make([("2026-02-09 09:00", "A", "B", 50000, "wire"), ("2026-02-09 13:00", "B", "C", 49500, "wire")])
    assert tools.find_cycles(led, "A", ALERT)["cycles_found"] == 0


def test_cycle_requires_time_order():
    # D pays A *before* A pays B, so the money cannot have gone round the ring
    led = make([("2026-02-08 08:00", "D", "A", 48500, "wire"), ("2026-02-09 09:00", "A", "B", 50000, "wire"),
                ("2026-02-09 13:00", "B", "C", 49500, "wire"), ("2026-02-09 18:00", "C", "D", 49000, "wire")])
    assert tools.find_cycles(led, "A", ALERT)["cycles_found"] == 0


def test_pass_through_detected():
    led = make([("2026-02-09 09:00", "X", "A", 9000, "wire"), ("2026-02-09 15:00", "A", "Y", 8800, "wire")])
    r = tools.check_pass_through(led, "A", ALERT)
    assert r["matched_pairs"] == 1 and r["share_of_inflow_passed_through"] == 1.0


def test_salary_then_rent_is_not_pass_through():
    led = make([("2026-02-09 09:00", "EMP", "A", 6000, "ach"), ("2026-02-09 15:00", "A", "LAND", 2000, "ach")])
    assert tools.check_pass_through(led, "A", ALERT)["matched_pairs"] == 0


def test_payroll_is_recognised_as_routine_fan_out():
    staff = [f"S{i}" for i in range(8)]
    rows = [(f"2026-01-{d} 10:00", "EMP", s, 3000, "ach") for d in ("05", "20") for s in staff]
    rows += [("2026-02-09 10:00", "EMP", s, 3000, "ach") for s in staff]
    r = tools.check_fan_pattern(make(rows), "EMP", ALERT)["fan_out"]
    assert r["distinct_counterparties"] == 8
    assert r["new_counterparty_share"] == 0.0 and r["prior_days_with_similar_fan"] == 2


def test_sudden_fan_out_to_strangers_stands_out():
    rows = [("2026-01-10 10:00", "A", "SHOP", 1200, "card")]
    rows += [("2026-02-09 10:00", "A", f"M{i}", 7000, "wire") for i in range(10)]
    r = tools.check_fan_pattern(make(rows), "A", ALERT)["fan_out"]
    assert r["max_in_one_day"] == 10 and r["new_counterparty_share"] == 1.0 and r["prior_days_with_similar_fan"] == 0


def test_history_comparison_flags_spike():
    rows = [(f"2026-01-{d:02d} 10:00", "A", "SHOP", 100, "card") for d in range(1, 29)]
    rows += [("2026-02-09 10:00", "A", "Z", 40000, "wire")]
    r = tools.compare_to_history(make(rows), "A", ALERT)
    assert r["enough_history"] and r["out"]["ratio_to_usual"] > 20 and r["out"]["largest_ever_before"] == 100


def test_network_trace_follows_new_links_only():
    rows = [("2026-01-10 10:00", "A", "OLD", 5000, "wire"), ("2026-02-09 10:00", "A", "OLD", 5000, "wire"),
            ("2026-02-09 11:00", "A", "M1", 7000, "wire"), ("2026-02-09 15:00", "M1", "SINK", 6800, "wire")]
    r = tools.trace_network(make(rows), "A", ALERT)
    assert r["linked_accounts"] == ["M1", "SINK"]


def test_tools_handle_unknown_account_and_bad_calls():
    led = make([("2026-02-09 09:00", "A", "B", 5000, "wire")])
    for name in tools.TOOLS:
        assert "error" not in tools.call_tool(led, name, {"account": "NOPE", "as_of": ALERT})
    assert "error" in tools.call_tool(led, "delete_everything", {})
    assert "error" in tools.call_tool(led, "find_cycles", {"wrong": 1})


def test_every_tool_has_a_spec():
    assert {s["name"] for s in tools.TOOL_SPECS} == set(tools.TOOLS)


def test_short_history_is_flagged_as_a_data_limit_not_a_new_account():
    led = make([("2026-02-09 09:00", "X", "A", 9000, "wire")])
    r = tools.compare_to_history(led, "A", ALERT)
    assert r["days_of_data_before_window"] == 0 and "NOT evidence" in r["note"]
    assert "note" in tools.get_account_profile(led, "A", ALERT)


def test_history_comparison_ranks_window_against_earlier_periods():
    rows = [(f"2026-01-{d:02d} 10:00", "A", "SUP", 20000, "wire") for d in range(1, 29, 3)]
    rows += [("2026-02-09 10:00", "A", "SUP", 20000, "wire")]
    r = tools.compare_to_history(make(rows), "A", ALERT)
    assert r["earlier_periods_compared"] >= 3 and r["out"]["earlier_periods_with_higher_total"] >= 1


def test_counterparty_novelty_is_compared_with_the_accounts_own_norm():
    # a business that pays a different supplier every few days: new counterparties are normal for it
    rows = [(f"2026-01-{d:02d} 10:00", "BIZ", f"SUP{d}", 5000, "wire") for d in range(1, 31, 2)]
    rows += [("2026-02-09 10:00", "BIZ", "SUPX", 5000, "wire")]
    r = tools.get_counterparties(make(rows), "BIZ", ALERT)
    assert r["new_share"] == 1.0 and r["usual_new_share_in_earlier_periods"] == 1.0


def test_network_trace_is_empty_for_ordinary_payments():
    rows = [("2026-02-09 10:00", "BIZ", "SUP1", 5000, "wire"), ("2026-02-09 11:00", "BIZ", "SUP2", 12000, "wire"),
            ("2026-02-08 11:00", "CUST", "BIZ", 3000, "ach")]
    r = tools.trace_network(make(rows), "BIZ", ALERT)
    assert r["count"] == 0 and "note" in r


def test_network_trace_finds_hub_and_siblings_from_a_minor_recipient():
    rows = [("2026-01-10 10:00", "HUB", "SHOP", 1200, "card")]
    rows += [("2026-02-09 10:00", "HUB", f"M{i}", 7000, "wire") for i in range(8)]
    r = tools.trace_network(make(rows), "M3", ALERT)
    assert "HUB" in r["linked_accounts"] and {"M0", "M7"} <= set(r["linked_accounts"])
