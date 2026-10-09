"""Score an alert file against the ground-truth laundering labels.

An alert is a TRUE alert if its account took part in a labeled laundering
transaction within the review window (7 days before to 1 day after the alert).
Everything else is a false alarm an investigator would have reviewed for nothing.

Output: results/baseline_metrics.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

LOOKBACK_DAYS, LOOKAHEAD_DAYS = 7, 1


def label_alerts(alerts: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    bad = tx[tx.is_laundering == 1]
    long = pd.concat([
        bad[["from_account", "timestamp", "pattern_id"]].rename(columns={"from_account": "account"}),
        bad[["to_account", "timestamp", "pattern_id"]].rename(columns={"to_account": "account"}),
    ])
    long = long[long.account != "CASH"]
    m = alerts.merge(long, on="account", how="left")
    lo = m.date - pd.Timedelta(days=LOOKBACK_DAYS)
    hi = m.date + pd.Timedelta(days=LOOKAHEAD_DAYS + 1)
    m["match"] = (m.timestamp >= lo) & (m.timestamp < hi)
    matched = m[m.match].groupby("alert_id").pattern_id.agg(lambda p: "|".join(sorted(set(p))))
    out = alerts.copy()
    out["patterns"] = out.alert_id.map(matched).fillna("")
    out["is_true_alert"] = (out.patterns != "").astype(int)
    return out


def metrics(labeled: pd.DataFrame, tx: pd.DataFrame) -> dict:
    bad = tx[tx.is_laundering == 1]
    bad_accounts = (set(bad.from_account) | set(bad.to_account)) - {"CASH"}
    all_patterns = bad.groupby("pattern_id").pattern_type.first()
    caught_accounts = set(labeled.loc[labeled.is_true_alert == 1, "account"])
    caught_patterns = {p for s in labeled.patterns if s for p in s.split("|")}

    n, tp = len(labeled), int(labeled.is_true_alert.sum())
    by_type = {}
    for t, ids in all_patterns.groupby(all_patterns).groups.items():
        by_type[t] = {"patterns": len(ids), "caught": len(set(ids) & caught_patterns)}
    by_rule = {}
    exploded = labeled.assign(rule=labeled.rules.str.split("|")).explode("rule")
    for r, g in exploded.groupby("rule"):
        by_rule[r] = {"alerts": len(g), "true_alerts": int(g.is_true_alert.sum()),
                      "precision": round(g.is_true_alert.mean(), 4)}
    return {
        "transactions": len(tx),
        "laundering_transactions": len(bad),
        "alerts": n,
        "true_alerts": tp,
        "false_alerts": n - tp,
        "precision": round(tp / n, 4),
        "false_positive_share": round(1 - tp / n, 4),
        "false_alerts_per_true_alert": round((n - tp) / tp, 1),
        "laundering_accounts": len(bad_accounts),
        "laundering_accounts_alerted": len(caught_accounts & bad_accounts),
        "account_recall": round(len(caught_accounts & bad_accounts) / len(bad_accounts), 4),
        "patterns": len(all_patterns),
        "patterns_caught": len(caught_patterns),
        "pattern_recall": round(len(caught_patterns) / len(all_patterns), 4),
        "by_pattern_type": by_type,
        "by_rule": by_rule,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--alerts", default="results/alerts.csv")
    ap.add_argument("--out", default="results/baseline_metrics.json")
    args = ap.parse_args()
    tx = pd.read_csv(args.data, parse_dates=["timestamp"], keep_default_na=False)
    alerts = pd.read_csv(args.alerts, parse_dates=["date"])
    labeled = label_alerts(alerts, tx)
    labeled.to_csv(args.alerts, index=False)
    m = metrics(labeled, tx)
    Path(args.out).write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
