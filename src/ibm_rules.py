"""Rule-based monitoring on the IBM data: the "before" system.

The rules are built on one idea that holds in real AML practice: money moving
between parties with no prior relationship deserves a look. None of the rules
uses the payment format, because in this dataset one format (ACH) carries most
of the laundering and leaning on that would not transfer to a real bank.

  N1  a first-time payment of $10,000 or more
  N2  three or more first-time counterparties within three days
  N3  relay: money received from a first-time sender, then paid to a first-time receiver within a day
  N4  first-time payments in two or more currencies on one day

An alert is one (account, day). It is TRUE if the account took part in a labeled
laundering transaction within the review window (two days before to one day after).

Output: results/ibm/alerts.parquet, results/ibm/rules_baseline.json
"""
import json
from pathlib import Path

import pandas as pd

import ibm_data as D

N1_AMOUNT, N2_COUNT = 10_000, 3


def build_alerts(tx: pd.DataFrame) -> pd.DataFrame:
    new = tx[tx.new]
    first_alert_day = D.DEV_DAYS[0]
    recent = new[new.day >= first_alert_day]
    hits = []

    hits.append(recent[recent.amount_usd >= N1_AMOUNT][["src", "day"]].rename(columns={"src": "a"}).assign(rule="N1"))

    cps = pd.concat([new[["src", "day", "dst"]].rename(columns={"src": "a", "dst": "cp"}),
                     new[["dst", "day", "src"]].rename(columns={"dst": "a", "src": "cp"})])
    rolled = pd.concat([cps.assign(day=cps.day + s) for s in (0, 1, 2)])     # trailing three days
    n = rolled.groupby(["a", "day"]).cp.nunique().reset_index()
    hits.append(n[n.cp >= N2_COUNT][["a", "day"]].assign(rule="N2"))

    got = new[["dst", "day"]].rename(columns={"dst": "a"}).drop_duplicates()
    got = pd.concat([got, got.assign(day=got.day + 1)]).drop_duplicates()     # received today or yesterday
    paid = recent[["src", "day"]].rename(columns={"src": "a"}).drop_duplicates()
    hits.append(paid.merge(got, on=["a", "day"]).assign(rule="N3"))

    cur = recent.groupby(["src", "day"]).currency.nunique().reset_index().rename(columns={"src": "a"})
    hits.append(cur[cur.currency >= 2][["a", "day"]].assign(rule="N4"))

    allhits = pd.concat(hits).drop_duplicates()
    allhits = allhits[allhits.day.between(first_alert_day, D.MAIN_DAYS - 1)]
    alerts = allhits.groupby(["a", "day"]).rule.agg(lambda s: "|".join(sorted(set(s)))).rename("rules").reset_index()

    inv = D.laundering_involvement(tx)
    win = pd.concat([inv.assign(day=inv.day + s) for s in range(-D.WINDOW_FWD, D.WINDOW_BACK + 1)])
    truth = win.groupby(["a", "day"]).pattern_id.agg(lambda s: "|".join(str(p) for p in sorted(set(s)) if p >= 0))
    alerts = alerts.merge(truth.rename("patterns").reset_index().assign(is_true_alert=1), on=["a", "day"], how="left")
    alerts["is_true_alert"] = alerts.is_true_alert.fillna(0).astype(int)
    alerts["patterns"] = alerts.patterns.fillna("")
    alerts["split"] = alerts.day.map(D.split_of)
    alerts = alerts.sort_values(["day", "a"]).reset_index(drop=True)
    alerts.insert(0, "alert_id", [f"IB{i:06d}" for i in range(len(alerts))])
    return alerts


def baseline_metrics(alerts: pd.DataFrame, tx: pd.DataFrame, split: str) -> dict:
    days = D.DEV_DAYS if split == "dev" else D.TEST_DAYS
    al = alerts[alerts.split == split]
    bad = tx[(tx.is_laundering == 1) & tx.day.isin(days)]
    flagged = set(al.a)
    caught = bad[bad.src.isin(flagged) | bad.dst.isin(flagged)]
    schemes = bad[bad.pattern_id >= 0].pattern_id.nunique()
    by_rule = {}
    ex = al.assign(rule=al.rules.str.split("|")).explode("rule")
    for r, g in ex.groupby("rule"):
        by_rule[r] = {"alerts": len(g), "precision": round(g.is_true_alert.mean(), 4)}
    n, tp = len(al), int(al.is_true_alert.sum())
    return {"days": list(days), "alerts": n, "alerts_per_day": round(n / len(days)), "true_alerts": tp,
            "false_alerts": n - tp, "precision": round(tp / n, 4), "false_positive_share": round(1 - tp / n, 4),
            "laundering_transactions": len(bad), "laundering_transactions_on_alerted_accounts": len(caught),
            "laundering_transaction_recall": round(len(caught) / len(bad), 4),
            "schemes": int(schemes), "schemes_touched": int(caught[caught.pattern_id >= 0].pattern_id.nunique()),
            "by_rule": by_rule}


def main():
    tx, meta = D.load()
    alerts = build_alerts(tx)
    out = Path("results/ibm")
    out.mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(out / "alerts.parquet", index=False)
    m = {"protocol": meta["protocol"], "transactions_scored": len(tx),
         "dev": baseline_metrics(alerts, tx, "dev"), "test": baseline_metrics(alerts, tx, "test")}
    (out / "rules_baseline.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
