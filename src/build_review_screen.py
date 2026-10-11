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
from ibm_agent import RULE_MEANINGS, SYSTEM_PROMPT, add_percentiles

RUN = "ibm-test-haiku"
STANDOUT = {"out_parties": "first-time parties paid", "out_currencies": "currencies paid out", "out_max_usd": "largest first-time payment made",
            "in_parties": "first-time parties paying in", "first_time_share": "share of activity that is first-time",
            "history_transactions": "payments before the window", "out_parties_paying_on": "share of payees who paid onward",
            "window_usd": "money moved in the window"}


def step_summary(name: str, out: dict) -> str:
    """One plain line saying what a tool call returned, for the audit trail."""
    if "error" in out:
        return f"error: {out['error']}"
    if name == "account_summary":
        return (f"{out['first_time_payments_made']['payments']} first-time payments made, "
                f"{out['first_time_payments_received']['payments']} received, "
                f"{out['before_the_window']['payments']} payments before the window")
    if name == "trace_new_relationships":
        cyc = "a cycle back to the account" if out["cycle_back_to_this_account"] else "no cycle"
        return (f"reached {out['accounts_reached']} accounts; largest fan-out {out['largest_fan_out_at_any_account']}, "
                f"largest fan-in {out['largest_fan_in_at_any_account']}; {cyc}")
    if name == "list_first_time_payments":
        return f"{out['first_time_payments']} first-time payments listed"
    if name == "get_transactions":
        return f"{out['payments_in_window']} payments in the window"
    return "returned"


def scheme_breakdown(alerts: pd.DataFrame, ptype: dict, top_k: int) -> list:
    """For each labeled scheme type: how many true test alerts there are, and how many the model ranks in its top slice."""
    test = alerts[alerts.split == "test"].sort_values("model_score", ascending=False)
    test = test.assign(in_top=[i < top_k for i in range(len(test))])
    rows = {}
    for r in test[test.is_true_alert == 1].itertuples():
        kinds = {ptype[int(p)] for p in r.patterns.split("|") if p} or {"NOT IN A LABELED SCHEME"}
        for k in kinds:
            n = rows.setdefault(k, [0, 0, 0])
            n[0] += 1
            n[1] += int(r.in_top)
            n[2] += int(r.in_band)
    return sorted(([k.title().replace("-", " "), *v] for k, v in rows.items()), key=lambda x: -x[2] / x[1])


def main():
    tx, meta = D.load()
    L = T.IBMLedger(tx, meta)
    alerts = add_percentiles(pd.read_parquet("results/ibm/alert_scores.parquet")).set_index("alert_id")
    ptype = tx[tx.pattern_id >= 0].groupby("pattern_id").pattern_type.first().map(lambda i: meta["pattern_types"][i]).to_dict()
    run = Path("results/ibm/agent_runs") / RUN
    metrics = json.loads((run / "metrics.json").read_text())
    feats = pd.read_parquet("results/ibm/alert_features.parquet").set_index("alert_id")
    pool = feats.loc[alerts.index[(alerts.split == "test") & alerts.in_band]]
    cases = []
    for line in open(run / "traces.jsonl"):
        v = json.loads(line)
        a = alerts.loc[v["alert_id"]]
        date = T.date_of(a.day)
        s = T.account_summary(L, v["account"], date)
        tr = T.trace_new_relationships(L, v["account"], date)
        kinds = sorted({ptype[int(p)] for p in a.patterns.split("|") if p})
        steps = [{"tool": t["name"], "on": t["input"].get("account", ""), "got": step_summary(t["name"], t["output"])}
                 for t in v["trace"] if t["type"] == "tool"]
        row = feats.loc[v["alert_id"]]
        ranks = sorted(((abs((pool[f] < row[f]).mean() + (pool[f] == row[f]).mean() / 2 - 0.5), f) for f in STANDOUT), reverse=True)[:3]
        standout = [{"what": STANDOUT[f], "value": float(row[f]),
                     "pct": round(float(((pool[f] < row[f]).mean() + (pool[f] == row[f]).mean() / 2) * 100))} for _, f in ranks]
        cases.append({
            "id": v["alert_id"], "acct": v["account"], "date": date, "rules": a.rules.split("|"),
            "pct": round(float(a.percentile)), "risk": v["risk_score"], "disp": v["disposition"], "typ": v["typology"],
            "status": v["status"], "ev": v["key_evidence"], "narr": v["narrative"], "inn": v["innocent_explanation"],
            "linked": v["linked_accounts"], "tools": v["tool_calls"],
            "win": s["window"], "made": s["first_time_payments_made"], "recv": s["first_time_payments_received"],
            "all": s["all_activity_in_window"], "share": s["share_of_window_payments_that_are_first_time"],
            "hist": s["before_the_window"], "pay": tr["payments"][:24], "reach": tr["accounts_reached"],
            "steps": steps, "standout": standout,
            "truth": int(a.is_true_alert), "scheme": kinds,
        })
    cases.sort(key=lambda c: -c["pct"])

    # Every other account that appears in a case, so the viewer can look it up: a short summary plus where it appears
    others = {}
    for c in cases:
        for acct in {a for pmt in c["pay"] for a in (pmt["from"], pmt["to"])} | set(c["linked"]):
            if acct == c["acct"]:
                continue
            o = others.get(acct)
            if o is None:
                su = T.account_summary(L, acct, c["date"])
                m_, r_, al_, h_ = su["first_time_payments_made"], su["first_time_payments_received"], su["all_activity_in_window"], su["before_the_window"]
                o = others[acct] = {"win": su["window"], "m": [m_["payments"], m_["parties"], round(m_["usd"])],
                                    "r": [r_["payments"], r_["parties"], round(r_["usd"])],
                                    "all": [al_["payments_made"], al_["payments_received"]], "h": [h_["payments"], h_["days_of_data"]],
                                    "cur": m_["currencies"], "in": []}
            o["in"].append(c["id"])

    # Overview figures: every number comes from a results file, none is typed in by hand
    rules = json.loads(Path("results/ibm/rules_baseline.json").read_text())["test"]
    model = json.loads(Path("results/ibm/model_baseline.json").read_text())
    test = alerts[alerts.split == "test"].sort_values("model_score", ascending=False)
    kept = np.cumsum(test.is_true_alert.to_numpy()) / test.is_true_alert.sum()
    full_curve = [[0, 0]] + [[round(x / 100, 2), round(float(kept[max(0, int(len(test) * x / 100) - 1)]), 4)] for x in range(1, 101)]

    top_ids = set(test.index[:model["test"]["keep_80pct_true"]["alerts_reviewed"]])
    ids = [c["id"] for c in cases]
    y = alerts.loc[ids].is_true_alert.to_numpy()
    n_t, n_f = int(y.sum()), int((1 - y).sum())
    band = metrics["band"]
    w = np.where(y == 1, band["true_alerts"] / n_t, (band["alerts"] - band["true_alerts"]) / n_f)
    levels = [i / 20 for i in range(1, 21)]

    def curve(score):
        return [[0, 0]] + [[round(E.review_share(y, score, w, k), 4), k] for k in levels]

    r, d, m80, m90 = metrics["ranking"], metrics["agent_decision"], model["test"]["keep_80pct_true"], model["test"]["keep_90pct_true"]
    data = {"cases": cases, "rules": RULE_MEANINGS, "others": others,
            "guidance": SYSTEM_PROMPT.split("How to work")[0].strip(),      # the tested agent's instructions, reused by the live agent
            "types": scheme_breakdown(alerts.reset_index(), ptype, m80["alerts_reviewed"]),
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
                      "labeled": int(((t_ := alerts[(alerts.split == "test") & (alerts.is_true_alert == 1)]).patterns != "").sum()),
                      "labeled_top": int((t_.patterns != "").sum()) - sum(1 for i in t_.index[t_.patterns != ""] if i not in top_ids),
                      "tests": sum(t.read_text().count("\ndef test_") for t in Path("tests").glob("test_*.py"))}}
    template = Path("app/review_template.html").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    Path("app/review.html").write_text(template.replace("/*__DATA__*/null", payload))
    print(f"{len(cases)} cases, {len(payload) / 1e6:.2f} MB -> app/review.html")


if __name__ == "__main__":
    main()
