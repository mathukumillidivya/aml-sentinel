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
