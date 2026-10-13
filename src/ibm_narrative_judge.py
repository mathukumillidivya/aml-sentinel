"""Does the agent's case file describe the alert correctly, and does it weigh an innocent explanation?

ibm_faithfulness.py checks mechanically that every account and dollar figure in a case file can be traced
to the evidence. That cannot tell whether the prose says the right thing about it: a traceable figure can
still be attached to the wrong direction, date or currency, and a case file can name an innocent
explanation that is boilerplate. This script asks a second model to read each case file next to the
evidence the agent was shown and report:

  accuracy              accurate | minor_errors | material_errors, with each error quoted and typed
  currency              whether payment currencies are stated correctly (evidence amounts are US-dollar
                        equivalents; the currency field is the currency the payment was made in)
  innocent_explanation  specific | generic | implausible | missing
  supports_decision     whether the write-up supports the agent's own escalate/dismiss recommendation

The judge is Claude Haiku 4.5, the same model as the agent, at temperature 0. That is a weakness: a model
can share its own blind spots. The output is therefore spot-checked by hand (spot_check.json) and reported
as a screen for problems, not a measurement of truth.

Usage:
  python src/ibm_narrative_judge.py --run ibm-test-haiku --budget 1.5          # needs ANTHROPIC_API_KEY
  python src/ibm_narrative_judge.py --run ibm-test-haiku --provider mock --limit 5
Output: results/ibm/agent_runs/<run>/narrative_judge.jsonl and narrative_judge.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

import ibm_tools as T
from evaluate_agent import rate
from ibm_agent import add_percentiles, case_file

MODEL = "claude-haiku-4-5-20251001"
PRICE = {"input": 1.0 / 1e6, "output": 5.0 / 1e6, "cache_read": 0.1 / 1e6, "cache_write": 1.25 / 1e6}

INSTRUCTIONS = """You review anti-money-laundering case files written by an AI analyst, before a human \
investigator relies on them. You are given the EVIDENCE the analyst was shown (a case file and the output \
of every tool it called) and the WRITE-UP it produced (narrative, key evidence, innocent explanation, and \
its recommendation).

Check the write-up against the evidence only. Do not use outside knowledge about the accounts.

About amounts: every amount in the evidence is a US-dollar equivalent ("usd"). The "currency" field is the \
currency the payment was actually made in. "$11.3M in Euro payments" or "$11.3M (paid in Euro)" is correct. \
Writing a dollar-equivalent figure as if it were an amount in another currency ("EUR 11.3M", "11.3M Euros", \
"€11.3M") is a currency error, and so is naming the wrong currency for a payment. Rounded figures and sums \
of listed payments are fine.

Classify accuracy:
- accurate: every factual statement matches the evidence.
- minor_errors: small slips that would not change an investigator's view (a rounding slip, an off-by-one \
count, a time a few hours off).
- material_errors: a wrong direction of money, wrong counterparty, wrong date or day, a wrong total or \
count that changes the picture, a currency error, or a claim the evidence does not support.

For each error, quote the exact words from the write-up and say what the evidence shows.

Classify the innocent explanation:
- specific: a plausible legitimate reason that engages with this account's actual activity.
- generic: could be pasted into any case ("could be legitimate business activity").
- implausible: contradicts the evidence.
- missing: none given.

Say whether the write-up, as written, supports the analyst's own recommendation (escalate or dismiss).

Submit your review with the submit_review tool."""

TOOL = {"name": "submit_review", "description": "Submit the review of one case file.",
        "input_schema": {"type": "object", "required": ["accuracy", "errors", "currency_ok", "innocent_explanation", "supports_decision", "note"],
                         "properties": {
                             "accuracy": {"type": "string", "enum": ["accurate", "minor_errors", "material_errors"]},
                             "errors": {"type": "array", "items": {"type": "object", "required": ["quote", "type", "evidence_says"], "properties": {
                                 "quote": {"type": "string"},
                                 "type": {"type": "string", "enum": ["amount", "currency", "count", "direction", "date_or_time", "account", "unsupported_claim", "other"]},
                                 "evidence_says": {"type": "string"}}}},
                             "currency_ok": {"type": "boolean"},
                             "innocent_explanation": {"type": "string", "enum": ["specific", "generic", "implausible", "missing"]},
                             "supports_decision": {"type": "boolean"},
                             "note": {"type": "string", "description": "One sentence on the most important problem, or 'none'."}}}}


def cost(u: dict) -> float:
    return sum(u.get(k, 0) * PRICE[k] for k in PRICE)


def compact(o) -> str:
    return json.dumps(o, separators=(",", ":"))


class Judge:
    def __init__(self, provider: str):
        self.provider = provider
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(max_retries=5)

    def review(self, evidence: str, writeup: str) -> tuple[dict, dict]:
        if self.provider == "mock":
            return ({"accuracy": "accurate", "errors": [], "currency_ok": True, "innocent_explanation": "generic",
                     "supports_decision": True, "note": "mock"}, {"input": 0, "output": 0})
        r = self.client.messages.create(
            model=MODEL, max_tokens=1200,
            system=[{"type": "text", "text": INSTRUCTIONS, "cache_control": {"type": "ephemeral"}}],
            tools=[TOOL], tool_choice={"type": "tool", "name": "submit_review"},
            messages=[{"role": "user", "content": f"<evidence>\n{evidence}\n</evidence>\n\n<write_up>\n{writeup}\n</write_up>"}],
            extra_body={"temperature": 0})
        out = next(b.input for b in r.content if b.type == "tool_use")
        u = r.usage
        return out, {"input": u.input_tokens, "output": u.output_tokens,
                     "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0, "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "mock"])
    ap.add_argument("--budget", type=float, default=1.5, help="stop before spending more than this many dollars")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).exists() else Path("results/ibm/agent_runs") / args.run
    L = T.IBMLedger.load()
    alerts = add_percentiles(pd.read_parquet("results/ibm/alert_scores.parquet")).set_index("alert_id")
    truth = alerts["is_true_alert"] if "is_true_alert" in alerts else None
    judge, spent, rows = Judge(args.provider), 0.0, []
    out_path = run / "narrative_judge.jsonl"
    done = {json.loads(l)["alert_id"]: json.loads(l) for l in open(out_path)} if out_path.exists() else {}
    for line in open(run / "traces.jsonl"):
        v = json.loads(line)
        if v["status"] != "ok":
            continue
        if args.limit and len(rows) >= args.limit:
            break
        if v["alert_id"] in done:
            rows.append(done[v["alert_id"]])
            continue
        if spent >= args.budget:
            print(f"budget reached at {len(rows)} case files")
            break
        a = alerts.loc[v["alert_id"]]
        first = case_file(L, v["alert_id"], v["account"], T.date_of(a.day), a.rules, float(a.percentile))
        evidence = first + "\n\nTOOL OUTPUTS\n" + "\n".join(f"{t['name']}({compact(t['input'])}) -> {compact(t['output'])}" for t in v["trace"] if t["type"] == "tool")
        writeup = (f"Recommendation: {v['disposition']} (risk score {v['risk_score']}, typology {v['typology']})\n\n"
                   f"Narrative: {v['narrative']}\n\nKey evidence:\n" + "\n".join(f"- {e}" for e in v["key_evidence"]) +
                   f"\n\nInnocent explanation considered: {v['innocent_explanation'] or '(none)'}")
        for attempt in range(3):
            try:
                review, usage = judge.review(evidence, writeup)
                break
            except Exception as e:  # noqa: BLE001 - retried, then the case is skipped and counted
                print("retry", v["alert_id"], type(e).__name__, str(e)[:120])
                time.sleep(5 * (attempt + 1))
        else:
            continue
        spent += cost(usage)
        row = {"alert_id": v["alert_id"], "disposition": v["disposition"],
               "truth": bool(truth.loc[v["alert_id"]]) if truth is not None else None, **review, "usage": usage}
        rows.append(row)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    n = len(rows)
    acc = Counter(r["accuracy"] for r in rows)
    inn = Counter(r["innocent_explanation"] for r in rows)
    types = Counter(e["type"] for r in rows for e in r["errors"])
    m = {"judge_model": MODEL, "case_files_judged": n,
         "accurate": rate(acc["accurate"], n), "minor_errors": rate(acc["minor_errors"], n), "material_errors": rate(acc["material_errors"], n),
         "currency_ok": rate(sum(r["currency_ok"] for r in rows), n),
         "innocent_explanation": {k: rate(inn[k], n) for k in ("specific", "generic", "implausible", "missing")},
         "supports_own_decision": rate(sum(r["supports_decision"] for r in rows), n),
         "error_types": dict(types.most_common()),
         "by_disposition": {d: {"case_files": sum(r["disposition"] == d for r in rows),
                                "material_errors": rate(sum(r["disposition"] == d and r["accuracy"] == "material_errors" for r in rows), sum(r["disposition"] == d for r in rows))}
                            for d in ("escalate", "dismiss")},
         "judge_cost_usd": round(sum(cost(r["usage"]) for r in rows), 4),
         "note": "Judged by the same model family as the agent, at temperature 0; see spot_check.json for a manual check of a sample."}
    (run / "narrative_judge.json").write_text(json.dumps(m, indent=2))
    print(json.dumps({k: v for k, v in m.items() if k != "error_types"}, indent=2))


if __name__ == "__main__":
    main()
