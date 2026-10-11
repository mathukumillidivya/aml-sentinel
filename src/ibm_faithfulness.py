"""Are the agent's case files faithful to the evidence it actually retrieved?

A scoring model cannot write a case file, so this is the agent's own job. The
check is mechanical and needs no second AI: every account ID and every dollar
figure in the agent's narrative and evidence list is looked up in what the agent
was shown (the case file it started from plus every tool result).

  account  grounded if that account ID appears anywhere in the evidence
  amount   "found" if it equals a figure in the evidence (allowing for rounding),
           "derived" if it equals a sum of figures the evidence lists together
           (the agent added them up), otherwise "unsupported"

An unsupported figure is not proof of an error (it may be a different sum), but
it is a claim a reader cannot trace, which is what an auditor would flag.

Output: results/ibm/agent_runs/<run_id>/faithfulness.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import pandas as pd

import ibm_tools as T
from evaluate_agent import rate
from ibm_agent import SYSTEM_PROMPT, add_percentiles, case_file

ACCOUNT = re.compile(r"\bA\d{6}\b")
AMOUNT = re.compile(r"[$€]\s?(\d[\d,]*(?:\.\d+)?)\s?([kKmM](?![a-zA-Z])|million|thousand)?")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
SCALE = {"k": 1e3, "m": 1e6, "thousand": 1e3, "million": 1e6}


def amounts_in(text: str) -> list[tuple[float, bool]]:
    """Dollar figures written in prose, as (value, was_abbreviated)."""
    out = []
    for num, suffix in AMOUNT.findall(text):
        value = float(num.replace(",", ""))
        if suffix:
            value *= SCALE[suffix.lower()]
        out.append((value, bool(suffix) or "." not in num and value >= 1000 and value % 100 == 0))
    return out


def numbers_in(evidence) -> list[float]:
    """Every number that appears in the evidence the agent was shown."""
    text = evidence if isinstance(evidence, str) else json.dumps(evidence)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)          # "10,000" in prose is one number
    return [abs(float(n)) for n in NUMBER.findall(text)]


def close(a: float, b: float, loose: bool) -> bool:
    tol = 0.06 if loose else 0.006          # "$35k" may stand for 35,333; "$13,005" must be 13,005
    return abs(a - b) <= max(tol * max(abs(b), 1), 0.51)


def payment_groups(evidence_objects: list) -> list[list[float]]:
    """Lists of payment amounts that the evidence presents together, so sums of them can be recognised."""
    groups = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            usd = [float(x["usd"]) for x in o if isinstance(x, dict) and isinstance(x.get("usd"), (int, float))]
            if len(usd) >= 2:
                groups.append(usd)
            for v in o:
                walk(v)

    for obj in evidence_objects:
        walk(obj)
    return groups


def classify_amount(value: float, loose: bool, known: list[float], groups: list[list[float]]) -> str:
    if any(close(value, k, loose) for k in known):
        return "found"
    for g in groups:
        if close(value, sum(g), loose):
            return "derived"
        if len(g) <= 12:
            for r in (2, 3):
                if any(close(value, sum(c), loose) for c in itertools.combinations(g, r)):
                    return "derived"
    return "unsupported"


def check_case(written: str, evidence_text: str, evidence_objects: list) -> dict:
    seen_accounts = set(ACCOUNT.findall(evidence_text))
    cited = set(ACCOUNT.findall(written))
    known = numbers_in(evidence_text)
    groups = payment_groups(evidence_objects)
    kinds = [classify_amount(v, loose, known, groups) for v, loose in amounts_in(written)]
    return {"accounts": len(cited), "accounts_grounded": len(cited & seen_accounts),
            "ungrounded_accounts": sorted(cited - seen_accounts),
            "amounts": len(kinds), "found": kinds.count("found"), "derived": kinds.count("derived"),
            "unsupported": kinds.count("unsupported")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).exists() else Path("results/ibm/agent_runs") / args.run
    L = T.IBMLedger.load()
    alerts = add_percentiles(pd.read_parquet("results/ibm/alert_scores.parquet")).set_index("alert_id")
    rows, examples = [], []
    for line in open(run / "traces.jsonl"):
        v = json.loads(line)
        if v["status"] != "ok":
            continue
        a = alerts.loc[v["alert_id"]]
        date = T.date_of(a.day)
        first = case_file(L, v["alert_id"], v["account"], date, a.rules, float(a.percentile))
        outputs = [t["output"] for t in v["trace"] if t["type"] == "tool"]
        objects = outputs + [T.account_summary(L, v["account"], date), T.list_first_time_payments(L, v["account"], date, limit=15)]
        written = v["narrative"] + "\n" + "\n".join(v["key_evidence"]) + "\n" + v["innocent_explanation"]
        # the agent's instructions count as shown text too: quoting a rule threshold from them is not a made-up figure
        r = check_case(written, SYSTEM_PROMPT + first + json.dumps(outputs), objects)
        rows.append({"alert_id": v["alert_id"], **{k: r[k] for k in r if k != "ungrounded_accounts"}})
        if r["ungrounded_accounts"] and len(examples) < 10:
            examples.append({"alert_id": v["alert_id"], "accounts": r["ungrounded_accounts"]})
    df = pd.DataFrame(rows)
    clean = (df.accounts == df.accounts_grounded) & (df.unsupported == 0)
    m = {"case_files_checked": len(df),
         "accounts_cited_grounded": rate(int(df.accounts_grounded.sum()), int(df.accounts.sum())),
         "amounts_cited": int(df.amounts.sum()),
         "amounts_found_in_evidence": rate(int(df.found.sum()), int(df.amounts.sum())),
         "amounts_derived_by_adding": rate(int(df.derived.sum()), int(df.amounts.sum())),
         "amounts_unsupported": rate(int(df.unsupported.sum()), int(df.amounts.sum())),
         "case_files_fully_traceable": rate(int(clean.sum()), len(df)),
         "avg_accounts_per_case_file": round(df.accounts.mean(), 2), "avg_amounts_per_case_file": round(df.amounts.mean(), 2),
         "ungrounded_account_examples": examples}
    (run / "faithfulness.json").write_text(json.dumps(m, indent=2))
    df.to_csv(run / "faithfulness_by_alert.csv", index=False)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
