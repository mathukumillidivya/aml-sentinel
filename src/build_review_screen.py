"""Build the investigator review screen: one self-contained HTML page.

Takes the alerts the agent investigated on the held-out IBM test days and packs
each case (model score, agent case file, first-time payments) into the page.
Benchmark labels are included so a demo viewer can check their own decisions;
the page keeps them hidden until asked.

Output: app/review.html
"""
import json
from pathlib import Path

import pandas as pd

import ibm_data as D
import ibm_tools as T
from ibm_agent import RULE_MEANINGS, add_percentiles

RUN = "ibm-test-haiku"


def main():
    tx, meta = D.load()
    L = T.IBMLedger(tx, meta)
    alerts = add_percentiles(pd.read_parquet("results/ibm/alert_scores.parquet")).set_index("alert_id")
    ptype = tx[tx.pattern_id >= 0].groupby("pattern_id").pattern_type.first().map(lambda i: meta["pattern_types"][i]).to_dict()
    run = Path("results/ibm/agent_runs") / RUN
    metrics = json.loads((run / "metrics.json").read_text())
    cases = []
    for line in open(run / "traces.jsonl"):
        v = json.loads(line)
        a = alerts.loc[v["alert_id"]]
        date = T.date_of(a.day)
        s = T.account_summary(L, v["account"], date)
        tr = T.trace_new_relationships(L, v["account"], date)
        kinds = sorted({ptype[int(p)] for p in a.patterns.split("|") if p})
        cases.append({
            "id": v["alert_id"], "acct": v["account"], "date": date, "rules": a.rules.split("|"),
            "pct": round(float(a.percentile)), "risk": v["risk_score"], "disp": v["disposition"], "typ": v["typology"],
            "status": v["status"], "ev": v["key_evidence"], "narr": v["narrative"], "inn": v["innocent_explanation"],
            "linked": v["linked_accounts"], "tools": v["tool_calls"],
            "win": s["window"], "made": s["first_time_payments_made"], "recv": s["first_time_payments_received"],
            "all": s["all_activity_in_window"], "share": s["share_of_window_payments_that_are_first_time"],
            "hist": s["before_the_window"], "pay": tr["payments"][:24], "reach": tr["accounts_reached"],
            "truth": int(a.is_true_alert), "scheme": kinds,
        })
    cases.sort(key=lambda c: -c["pct"])
    r = metrics["ranking"]
    data = {"cases": cases, "rules": RULE_MEANINGS,
            "facts": {"band": metrics["band"]["alerts"], "band_true": metrics["band"]["true_alerts"],
                      "sample": len(cases), "sample_true": sum(c["truth"] for c in cases),
                      "auc_model": r["model"]["auc"], "auc_agent": r["agent"]["auc"]}}
    template = Path("app/review_template.html").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    Path("app/review.html").write_text(template.replace("/*__DATA__*/null", payload))
    print(f"{len(cases)} cases, {len(payload) / 1e6:.2f} MB -> app/review.html")


if __name__ == "__main__":
    main()
