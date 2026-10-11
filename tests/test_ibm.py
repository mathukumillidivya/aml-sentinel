"""Tests for the IBM pipeline, on tiny hand-made ledgers where the right answer is obvious."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ibm_data as D  # noqa: E402
import ibm_features as F  # noqa: E402
import ibm_model as M  # noqa: E402
import ibm_prepare as P  # noqa: E402
import ibm_rules as R  # noqa: E402


def ledger(rows):
    """rows: (day, src, dst, usd, currency, laundering, pattern_id)"""
    df = pd.DataFrame(rows, columns=["day", "src", "dst", "amount_usd", "currency", "is_laundering", "pattern_id"])
    df["minute"] = df.day * 1440 + 600
    df["cross_currency"], df["format"], df["pattern_type"] = False, 0, -1
    return D.mark_first_time(df.astype({"day": "int16"}))


def alerts_for(rows):
    return R.build_alerts(ledger(rows)).set_index(["a", "day"])


def test_first_time_flag_uses_only_the_past():
    tx = ledger([(0, 1, 2, 500, 0, 0, -1), (3, 1, 2, 500, 0, 0, -1), (4, 1, 3, 500, 0, 0, -1), (5, 1, 3, 500, 0, 0, -1)])
    assert tx.new.tolist() == [False, False, True, False]   # day 0 never counts; repeats are not new


def test_n1_large_first_time_payment():
    al = alerts_for([(4, 1, 2, 25_000, 0, 0, -1), (1, 5, 6, 25_000, 0, 0, -1), (4, 5, 6, 25_000, 0, 0, -1)])
    assert al.loc[(1, 4)].rules == "N1"
    assert (5, 4) not in al.index            # 5 has paid 6 before, so the relationship is not new


def test_n2_three_new_counterparties_in_three_days():
    al = alerts_for([(3, 1, 10, 50, 0, 0, -1), (4, 1, 11, 50, 0, 0, -1), (5, 1, 12, 50, 0, 0, -1)])
    assert (1, 5) in al.index and "N2" in al.loc[(1, 5)].rules
    assert (1, 4) not in al.index            # only two new counterparties by day 4


def test_n3_relay_through_new_relationships():
    al = alerts_for([(4, 7, 1, 900, 0, 0, -1), (5, 1, 8, 850, 0, 0, -1)])
    assert al.loc[(1, 5)].rules == "N3"


def test_n4_first_time_payments_in_two_currencies():
    al = alerts_for([(4, 1, 2, 300, 0, 0, -1), (4, 1, 3, 300, 5, 0, -1)])
    assert "N4" in al.loc[(1, 4)].rules


def test_label_window_is_two_days_back_to_one_day_forward():
    base = [(4, 1, 2, 25_000, 0, 0, -1)]
    for day, expected in ((2, 1), (5, 1), (1, 0), (6, 0)):
        al = alerts_for(base + [(day, 1, 99, 10, 0, 1, 7)])
        assert al.loc[(1, 4)].is_true_alert == expected, day
    assert alerts_for(base + [(5, 1, 99, 10, 0, 1, 7)]).loc[(1, 4)].patterns == "7"


def test_alerts_only_from_development_day_onwards_and_splits_are_labelled():
    al = R.build_alerts(ledger([(2, 1, 2, 25_000, 0, 0, -1), (4, 3, 4, 25_000, 0, 0, -1), (8, 5, 6, 25_000, 0, 0, -1)]))
    assert al.day.tolist() == [4, 8] and al.split.tolist() == ["dev", "test"]


def test_features_do_not_depend_on_labels():
    rows = [(4, 7, 1, 900, 0, 1, 3), (5, 1, 8, 850, 0, 1, 3), (4, 2, 9, 30_000, 0, 0, -1), (5, 9, 4, 100, 0, 0, -1)]
    tx = ledger(rows)
    al = R.build_alerts(tx)
    flipped = tx.assign(is_laundering=1 - tx.is_laundering, pattern_id=-1)
    pd.testing.assert_frame_equal(F.build(tx, al), F.build(flipped, al))


def test_features_describe_the_relay():
    tx = ledger([(4, 7, 1, 900, 0, 0, -1), (5, 1, 8, 850, 0, 0, -1)])
    al = R.build_alerts(tx)
    x = F.build(tx, al).iloc[al.index[(al.a == 1) & (al.day == 5)][0]]
    assert x.out_parties == 1 and x.in_parties == 1 and round(x.relay_ratio, 2) == 0.94


def test_workload_curve_arithmetic():
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    s = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
    c = M.workload_curve(y, s)
    assert c["keep_50pct_true"] == {"alerts_reviewed": 3, "share_of_alerts": 0.3, "workload_reduction": 0.7, "precision": 0.6667}
    assert c["keep_95pct_true"]["alerts_reviewed"] == 10


def test_exchange_rates_are_read_from_cross_currency_payments():
    rows = [{"Payment Currency": "Euro", "Receiving Currency": "US Dollar", "Amount Paid": 100.0, "Amount Received": 117.0}] * 25
    rows += [{"Payment Currency": "US Dollar", "Receiving Currency": "Yen", "Amount Paid": 1.0, "Amount Received": 100.0}] * 25
    rates, support = P.implied_usd_rates(pd.DataFrame(rows))
    assert round(rates["Euro"], 2) == 1.17 and round(rates["Yen"], 2) == 0.01 and support["Euro"] == 25


def test_pattern_file_parser(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out\n"
                 "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1\n"
                 "END LAUNDERING ATTEMPT - FAN-OUT\n\nBEGIN LAUNDERING ATTEMPT - STACK\n"
                 "2022/09/02 12:36,0223,8DD890,0040312,82D550,14153.46,Swiss Franc,14153.46,Swiss Franc,ACH,1\nEND LAUNDERING ATTEMPT - STACK\n")
    p = P.read_patterns(f)
    assert p.pattern_type.tolist() == ["FAN-OUT", "STACK"] and p.pattern_id.tolist() == [0, 1]
    assert p["From Bank"].tolist() == [21174, 223]


# ---------- agent tools on IBM-style data ----------
import json  # noqa: E402

import agent  # noqa: E402
import ibm_agent as A  # noqa: E402
import ibm_agent_eval as E  # noqa: E402
import ibm_tools as T  # noqa: E402
from models import ScriptedClient  # noqa: E402

META = {"currencies": ["Euro", "US Dollar"], "formats": ["ACH", "Cheque"]}


def ibm_ledger(rows, minutes=None):
    tx = ledger(rows)
    if minutes:
        tx["minute"] = minutes
    return T.IBMLedger(tx, META), tx


RING = [(0, 9, 8, 50, 1, 0, -1),                      # old relationship, not part of anything
        (4, 1, 2, 9000, 1, 1, 0), (4, 2, 3, 400, 0, 1, 0), (5, 3, 1, 77000, 1, 1, 0)]


def test_ibm_ledger_hides_labels_and_slices_by_day():
    L, _ = ibm_ledger(RING)
    assert not set(L.both.columns) & set(T.LABEL_COLUMNS)
    assert len(L.activity(1)) == 2 and len(L.activity(1, 5, 5)) == 1 and L.activity(12345).empty


def test_trace_follows_money_in_time_order_and_finds_the_ring():
    L, _ = ibm_ledger(RING, minutes=[10, 5800, 5900, 7300])
    r = T.trace_new_relationships(L, "A000001", "2022-09-05")
    assert r["cycle_back_to_this_account"] == ["A000001", "A000002", "A000003", "A000001"]
    assert r["linked_accounts"] == ["A000002", "A000003"]
    assert r["currencies_along_the_way"] == ["Euro", "US Dollar"]     # amounts and currency change hop to hop


def test_trace_refuses_a_ring_that_runs_backwards_in_time():
    L, _ = ibm_ledger(RING, minutes=[10, 7300, 5900, 5800])           # the "return" leg happens first
    assert T.trace_new_relationships(L, "A000001", "2022-09-05")["cycle_back_to_this_account"] == []


def test_trace_ignores_established_relationships():
    L, _ = ibm_ledger([(0, 1, 2, 500, 1, 0, -1), (4, 1, 2, 500, 1, 0, -1)])
    r = T.trace_new_relationships(L, "A000001", "2022-09-05")
    assert r["accounts_reached"] == 0 and "note" in r


def test_account_summary_separates_first_time_from_routine():
    L, _ = ibm_ledger([(0, 1, 2, 500, 1, 0, -1), (1, 1, 2, 500, 1, 0, -1), (4, 1, 2, 500, 1, 0, -1), (4, 1, 7, 20000, 0, 0, -1)])
    s = T.account_summary(L, "A000001", "2022-09-05")
    assert s["all_activity_in_window"]["payments_made"] == 2
    assert s["first_time_payments_made"]["payments"] == 1 and s["first_time_payments_made"]["currencies"] == ["Euro"]
    assert s["share_of_window_payments_that_are_first_time"] == 0.5


def test_ibm_tools_never_crash_on_bad_input():
    L, _ = ibm_ledger(RING)
    for name in T.TOOLS:
        assert "error" not in T.call_tool(L, name, {"account": "A999999", "as_of": "2022-09-05"})
    assert "error" in T.call_tool(L, "nope", {}) and "error" in T.call_tool(L, "trace_new_relationships", {"x": 1})


def test_case_file_and_agent_run_contain_no_labels():
    L, _ = ibm_ledger(RING, minutes=[10, 5800, 5900, 7300])
    msg = A.case_file(L, "IB1", "A000001", "2022-09-05", "N3", 80.0)
    verdict = [{"type": "tool_use", "id": "v", "name": "submit_verdict", "input": dict(
        disposition="escalate", risk_score=80, typology="cycle", key_evidence=["ring"], narrative="n",
        linked_accounts=["A000002", "A000003", "A777777"])}]
    trace = [{"type": "tool_use", "id": "t", "name": "trace_new_relationships", "input": {"account": "A000001", "as_of": "2022-09-05"}}]
    client = ScriptedClient([trace, verdict])
    v = agent.investigate(client, L, "IB1", "A000001", "2022-09-05", "N3", tools=T, system_prompt=A.SYSTEM_PROMPT,
                          first_message=msg, typologies=A.TYPOLOGIES, escalate_at=A.ESCALATE_AT)
    sent = json.dumps(client.calls)
    for leak in ("is_laundering", "pattern_id", "pattern_type", "is_true_alert"):
        assert leak not in sent
    assert v.linked_accounts == ["A000002", "A000003"] and v.dropped_accounts == ["A777777"]   # invented account blocked


def test_ibm_escalation_threshold_is_enforced():
    low = dict(disposition="dismiss", risk_score=30, typology="none", key_evidence=["x"], narrative="n")
    assert "inconsistent" in agent.validate_verdict(low, A.TYPOLOGIES, 25)
    assert agent.validate_verdict({**low, "disposition": "escalate"}, A.TYPOLOGIES, 25) is None


def test_sampling_takes_only_review_band_alerts_from_the_right_split():
    al = pd.DataFrame({"alert_id": [f"IB{i}" for i in range(40)], "split": ["dev"] * 20 + ["test"] * 20,
                       "in_band": [True, False] * 20, "is_true_alert": [1, 1, 0, 0] * 10, "day": 4, "model_score": 0.5})
    s = A.sample_alerts(al, "test", 3, 4, seed=0)
    assert set(s.split) == {"test"} and s.in_band.all() and s.is_true_alert.sum() == 3 and len(s) == 7


def test_review_share_arithmetic_with_weights():
    y = np.array([1, 0, 1, 0]); s = np.array([.9, .8, .7, .1]); w = np.array([1., 10., 1., 10.])
    assert round(E.review_share(y, s, w, 0.5), 4) == round(1 / 22, 4)       # the top alert alone holds half the true ones
    assert round(E.review_share(y, s, w, 1.0), 4) == round(12 / 22, 4)


def test_review_share_does_not_depend_on_row_order_when_scores_tie():
    y = np.array([1, 1, 0, 0, 0, 0]); s = np.full(6, 0.1); w = np.ones(6)
    a = E.review_share(y, s, w, 0.5)
    b = E.review_share(y[::-1], s, w, 0.5)
    assert a == b == 0.5        # a score that cannot tell alerts apart saves no work


# ---------- faithfulness of case files ----------
import ibm_faithfulness as Fa  # noqa: E402


def test_amounts_are_read_from_prose_with_abbreviations():
    got = Fa.amounts_in("received $13,005.07, then about $35k and €6.93M; 11 hours later, USD 11 is not money here")
    assert [round(v) for v, _ in got] == [13005, 35000, 6930000]
    assert [loose for _, loose in got] == [False, True, True]


def test_exact_figures_must_match_closely_but_rounded_ones_may_be_approximate():
    known = [13005.07, 35333.0]
    assert Fa.classify_amount(13005.07, False, known, []) == "found"
    assert Fa.classify_amount(13500.0, False, known, []) == "unsupported"
    assert Fa.classify_amount(35000.0, True, known, []) == "found"


def test_sums_of_listed_payments_count_as_derived():
    groups = [[1000.0, 2500.0, 400.0]]
    assert Fa.classify_amount(3900.0, False, [], groups) == "derived"       # all three
    assert Fa.classify_amount(3500.0, False, [], groups) == "derived"       # two of them
    assert Fa.classify_amount(9999.0, False, [], groups) == "unsupported"


def test_case_file_check_flags_an_account_and_a_figure_that_were_never_shown():
    evidence = 'Account: A000001 {"payments":[{"to":"A000002","usd":5000.0},{"to":"A000003","usd":700.0}]}'
    objects = [{"payments": [{"to": "A000002", "usd": 5000.0}, {"to": "A000003", "usd": 700.0}]}]
    good = Fa.check_case("A000001 paid A000002 $5,000 and $5,700 in total.", evidence, objects)
    assert good["accounts"] == good["accounts_grounded"] == 2 and good["found"] == 1 and good["derived"] == 1 and good["unsupported"] == 0
    bad = Fa.check_case("A000001 paid A000009 $82,000.", evidence, objects)
    assert bad["ungrounded_accounts"] == ["A000009"] and bad["unsupported"] == 1


def test_thousands_separators_in_shown_text_are_one_number():
    assert 10000.0 in Fa.numbers_in("a first-time payment of $10,000 or more") and 1234567.0 in Fa.numbers_in("1,234,567")
