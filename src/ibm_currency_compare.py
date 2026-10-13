"""Compare the original and the revised (write-up v2) case files on the same 60 held-out alerts.

Both are scored the same way:
  code      ibm_currency.check_amounts: amounts written with a currency label that match no payment in that currency
  judge v2  ibm_narrative_judge.py --judge-v2 (sees the agent's guidance and the `amount` field), run on both
  decision  escalate/dismiss against the benchmark label, to show the fix did not change what the agent decides

Output: results/ibm/currency_v2_comparison.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ibm_currency as C  # noqa: E402
import ibm_tools as T  # noqa: E402
from ibm_agent import add_percentiles, case_file  # noqa: E402

RUNS = {"original": "results/ibm/agent_runs/ibm-test-haiku", "revised": "results/ibm/agent_runs/ibm-test-haiku-currency-v2"}
IDS = Path("results/ibm/currency_v2_alerts.txt").read_text().split()


def rate(k, n):
    return {"count": k, "of": n, "rate": round(k / n, 4) if n else None}


def main():
    L = T.IBMLedger.load()
    alerts = add_percentiles(pd.read_parquet("results/ibm/alert_scores.parquet")).set_index("alert_id")
    out = {"alerts": len(IDS), "runs": {}}
    for name, run in RUNS.items():
        run = Path(run)
        if not (run / "traces.jsonl").exists():
            continue
        cfg = json.loads((run / "run_config.json").read_text())
        T.NATIVE = bool(cfg.get("currency_v2"))
        traces = {json.loads(l)["alert_id"]: json.loads(l) for l in open(run / "traces.jsonl")}
        rows = []
        for aid in IDS:
            v = traces.get(aid)
            if not v or v["status"] != "ok":
                continue
            a = alerts.loc[aid]
            first = case_file(L, aid, v["account"], T.date_of(a.day), a.rules, float(a.percentile))
            problems = C.check_amounts(C.writeup_text(v), C.evidence_payments(v["trace"], first))
            rows.append({"alert_id": aid, "problems": problems, "revised": any(t["type"] == "revision_requested" for t in v["trace"]),
                         "right": (v["disposition"] == "escalate") == bool(a.is_true_alert)})
        j = {}
        jf = run / "narrative_judge_v2.jsonl"
        if jf.exists():
            jr = [json.loads(l) for l in open(jf) if json.loads(l)["alert_id"] in IDS]
            n = len(jr)
            j = {"judged": n, "accurate": rate(sum(r["accuracy"] == "accurate" for r in jr), n),
                 "material_errors": rate(sum(r["accuracy"] == "material_errors" for r in jr), n),
                 "currency_errors": rate(sum(any(e["type"] == "currency" for e in r["errors"]) or not r["currency_ok"] for r in jr), n),
                 "innocent_specific": rate(sum(r["innocent_explanation"] == "specific" for r in jr), n),
                 "cost_usd": round(sum(sum(r["usage"].get(k, 0) * p for k, p in {"input": 1e-6, "output": 5e-6, "cache_read": 1e-7, "cache_write": 1.25e-6}.items()) for r in jr), 4)}
        n = len(rows)
        out["runs"][name] = {"case_files": n, "currency_mislabels_code": rate(sum(bool(r["problems"]) for r in rows), n),
                             "sent_back_for_correction": rate(sum(r["revised"] for r in rows), n),
                             "decision_right": rate(sum(r["right"] for r in rows), n), "judge_v2": j,
                             "examples": [{"alert_id": r["alert_id"], "problems": r["problems"][:2]} for r in rows if r["problems"]][:8]}
        if name == "revised":
            out["runs"][name]["tokens"] = {"input": sum(t["input_tokens"] for t in traces.values()),
                                           "output": sum(t["output_tokens"] for t in traces.values())}
    Path("results/ibm/currency_v2_comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in out["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
