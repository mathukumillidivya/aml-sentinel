"""Code check: does a case file present a US-dollar equivalent as an amount in another currency?

Every amount the agent sees is a US-dollar equivalent; the payment's own currency is a separate field, and the
agent is given no exchange rates. So any amount written with a foreign currency symbol or code ("€4,168.24",
"$19,419.91 CAD", "11.3M Euros") cannot be right: the number is dollars. "$4,168.24 in Euro" is fine.

Usage: python src/ibm_currency_check.py --run ibm-test-haiku
Output: results/ibm/agent_runs/<run>/currency_check.json
"""
import argparse
import json
import re
from pathlib import Path

CODES = "EUR|GBP|CHF|JPY|CNY|INR|RUB|BRL|AUD|CAD|MXN|ILS|SAR"
NAMES = "Euros?|Pounds?|Yen|Yuan|Rupees?|Rubles?|Reals?|Francs?|Shekels?|Pesos?|Riyals?"
PATTERN = re.compile(rf"(€|£|¥|₹|₽|₪|R\$|\b(?:{CODES}))\s?~?\d"                       # €4,168 / CHF 6,781
                     rf"|\d[\d,.]*\s?(?:K|M|k|m)?\s?(?:€|\b(?:{CODES})\b)(?! cheques)"   # 14.8K EUR / $19,419.91 CAD
                     rf"|\d[\d,.]*\s?(?:K|M|k|million)?\s+(?:{NAMES})\b", re.I)      # 11.3M Euros


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).exists() else Path("results/ibm/agent_runs") / a.run
    files = nonusd = flagged = 0
    examples = []
    for line in open(run / "traces.jsonl"):
        v = json.loads(line)
        if v["status"] != "ok":
            continue
        files += 1
        currencies = set(re.findall(r'"currency":\s?"([^"]+)"', json.dumps(v["trace"])))
        nonusd += bool(currencies - {"US Dollar"})
        text = " ".join([v["narrative"], *v["key_evidence"], v["innocent_explanation"] or ""])
        m = PATTERN.search(text)
        if m:
            flagged += 1
            if len(examples) < 10:
                examples.append({"alert_id": v["alert_id"], "text": text[max(0, m.start() - 50):m.end() + 30]})
    out = {"case_files": files, "case_files_with_non_usd_payments": nonusd,
           "dollar_figure_written_as_foreign_currency": {"count": flagged, "of_non_usd_case_files": nonusd,
                                                         "rate": round(flagged / nonusd, 4) if nonusd else None},
           "examples": examples}
    (run / "currency_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
