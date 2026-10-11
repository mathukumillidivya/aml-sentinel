"""Build the investigator review screen: one self-contained HTML page.

Takes the alerts the agent investigated on the held-out IBM test days and packs
each case (model score, agent case file, first-time payments) into the page.
Benchmark labels are included so a demo viewer can check their own decisions;
the page keeps them hidden until asked.

Output: app/review.html
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ibm_agent_eval as E
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

    # Overview figures: every number comes from a results file, none is typed in by hand
    rules = json.loads(Path("results/ibm/rules_baseline.json").read_text())["test"]
    model = json.loads(Path("results/ibm/model_baseline.json").read_text())
    test = alerts[alerts.split == "test"].sort_values("model_score", ascending=False)
    kept = np.cumsum(test.is_true_alert.to_numpy()) / test.is_true_alert.sum()
    full_curve = [[0, 0]] + [[round(x / 100, 2), round(float(kept[max(0, int(len(test) * x / 100) - 1)]), 4)] for x in range(1, 101)]

    ids = [c["id"] for c in cases]
    y = alerts.loc[ids].is_true_alert.to_numpy()
    n_t, n_f = int(y.sum()), int((1 - y).sum())
    band = metrics["band"]
    w = np.where(y == 1, band["true_alerts"] / n_t, (band["alerts"] - band["true_alerts"]) / n_f)
    levels = [i / 20 for i in range(1, 21)]

    def curve(score):
        return [[0, 0]] + [[round(E.review_share(y, score, w, k), 4), k] for k in levels]

    r, d, m80, m90 = metrics["ranking"], metrics["agent_decision"], model["test"]["keep_80pct_true"], model["test"]["keep_90pct_true"]
    data = {"cases": cases, "rules": RULE_MEANINGS,
            "curves": {"full": full_curve,
                       "model": curve(alerts.loc[ids].model_score.to_numpy()),
                       "agent": curve(np.array([c["risk"] for c in cases]) / 100)},
            "facts": {"band": band["alerts"], "band_true": band["true_alerts"],
                      "sample": len(cases), "sample_true": sum(c["truth"] for c in cases),
                      "auc_model": r["model"]["auc"], "auc_agent": r["agent"]["auc"], "auc_combined": r["combined"]["auc"],
                      "gap": metrics["agent_vs_model"]["auc_gap"], "gap_ci": metrics["agent_vs_model"]["ci95"],
                      "read80_model": r["model"]["review_share_to_keep_80pct"], "read80_agent": r["agent"]["review_share_to_keep_80pct"],
                      "alerts": rules["alerts"], "true_alerts": rules["true_alerts"], "precision": rules["precision"],
                      "per_day": rules["alerts_per_day"], "txn_recall": rules["laundering_transaction_recall"],
                      "schemes": rules["schemes"], "schemes_touched": rules["schemes_touched"],
                      "auc_full": model["test"]["auc"], "keep80": m80, "keep90": m90,
                      "auc_no_format": model["test_without_payment_format"]["auc"],
                      "band_kept": model["review_band"]["test"]["true_alerts_kept"],
                      "failsafe": metrics["operations"]["failsafe_escalations"]["count"],
                      "tokens_in": metrics["operations"]["avg_input_tokens"], "seconds": metrics["operations"]["avg_seconds"],
                      "agent_true": d["true_alerts_escalated"], "agent_false": d["false_alarms_dismissed"],
                      "tests": sum(t.read_text().count("\ndef test_") for t in Path("tests").glob("test_*.py"))}}
    template = Path("app/review_template.html").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    Path("app/review.html").write_text(template.replace("/*__DATA__*/null", payload))
    print(f"{len(cases)} cases, {len(payload) / 1e6:.2f} MB -> app/review.html")


if __name__ == "__main__":
    main()
