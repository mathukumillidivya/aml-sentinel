"""Sanity check: do the investigation tools carry real signal?

Runs every tool on every alert and adds up a few obvious red flags into a crude
score. This is NOT the agent and not a headline result: the score was written by
hand against data from the same author. It answers one question before we spend
money on a language model: is the evidence the tools return good enough to
separate real cases from false alarms?

Output: results/tool_evidence.csv, results/tool_signal_check.json
"""
import json
import time
from pathlib import Path

import pandas as pd

import tools
from ledger import Ledger


def evidence(ledger, account, as_of):
    s = tools.check_structuring(ledger, account, as_of)
    f = tools.check_fan_pattern(ledger, account, as_of)
    p = tools.check_pass_through(ledger, account, as_of)
    c = tools.find_cycles(ledger, account, as_of)
    h = tools.compare_to_history(ledger, account, as_of)
    row = {
        "struct_n": s["deposits_just_under_limit"], "struct_share": s["just_under_share"],
        "pass_pairs": p["matched_pairs"], "pass_share": p["share_of_inflow_passed_through"],
        "cycles": c["cycles_found"], "enough_history": h["enough_history"],
        "out_ratio": h["out"]["ratio_to_usual"] or 0, "in_ratio": h["in"]["ratio_to_usual"] or 0,
    }
    for k in ("fan_out", "fan_in"):
        row[f"{k}_max_day"] = f[k]["max_in_one_day"]
        row[f"{k}_new_share"] = f[k]["new_counterparty_share"]
        row[f"{k}_prior"] = f[k]["prior_days_with_similar_fan"]
    return row


def crude_score(r):
    score = 0.0
    if r["cycles"]:
        score += 3
    if r["struct_n"] >= 3 and r["struct_share"] >= 0.6:
        score += 3
    if r["pass_pairs"] >= 1:
        score += 1.5 + (1 if r["pass_share"] >= 0.5 else 0)
    for k in ("fan_out", "fan_in"):
        if r[f"{k}_max_day"] >= 5 and r[f"{k}_new_share"] >= 0.7 and r[f"{k}_prior"] == 0:
            score += 3
    if r["enough_history"] and max(r["out_ratio"], r["in_ratio"]) > 5:
        score += 1
    return score


def auc(scores, labels):
    """Chance that a random true alert outscores a random false alarm (0.5 = useless, 1.0 = perfect)."""
    ranks = scores.rank(method="average")
    pos, neg = int(labels.sum()), int((1 - labels).sum())
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def main():
    ledger = Ledger.from_csv()
    alerts = pd.read_csv("results/alerts.csv", keep_default_na=False)
    t0 = time.time()
    rows = [evidence(ledger, a.account, a.date) for a in alerts.itertuples()]
    secs = time.time() - t0
    ev = pd.concat([alerts[["alert_id", "account", "date", "rules", "is_true_alert"]], pd.DataFrame(rows)], axis=1)
    ev["crude_score"] = ev.apply(crude_score, axis=1)
    ev.to_csv("results/tool_evidence.csv", index=False)

    y = ev.is_true_alert
    kept = ev[ev.crude_score > 0]
    out = {
        "alerts": len(ev), "true_alerts": int(y.sum()),
        "seconds_per_alert_all_tools": round(secs / len(ev), 4),
        "auc_of_crude_score": round(float(auc(ev.crude_score, y)), 4),
        "alerts_with_any_red_flag": len(kept),
        "true_alerts_with_any_red_flag": int(kept.is_true_alert.sum()),
        "true_alerts_with_no_red_flag": int(y.sum() - kept.is_true_alert.sum()),
        "false_alarms_with_no_red_flag": int(((ev.crude_score == 0) & (y == 0)).sum()),
        "false_alarms_total": int((1 - y).sum()),
    }
    out["share_of_false_alarms_with_no_red_flag"] = round(out["false_alarms_with_no_red_flag"] / out["false_alarms_total"], 4)
    out["share_of_true_alerts_with_red_flag"] = round(out["true_alerts_with_any_red_flag"] / out["true_alerts"], 4)
    Path("results/tool_signal_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
