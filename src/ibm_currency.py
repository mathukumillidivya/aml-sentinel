"""Currency handling for the agent's case files (version 2 of the write-up step).

The problem (results/ibm/agent_runs/ibm-test-haiku/currency_check.json): the agent saw only US-dollar equivalents
plus a currency name, and in 24% of case files with non-dollar payments it wrote a dollar figure as if it were euros
("€4,168.24" for a payment of €3,557.18, which is $4,168.24).

Version 2 does three things:
  1. every payment the tools return carries `amount`, the value in the currency it was paid in, next to `usd`
     (rates implied by the dataset's own cross-currency payments, results/ibm/usd_rates_HI-Small.json);
  2. the prompt says how to write amounts (WRITING_RULE);
  3. after the agent submits, code checks every amount written with a currency label against the payments in its
     evidence (check_amounts). A mislabelled amount sends the verdict back once with the list of problems.

check_amounts is also the evaluation metric, applied the same way to the original and the revised runs.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

RATES_FILE = Path("results/ibm/usd_rates_HI-Small.json")
SYMBOLS = {"€": "Euro", "£": "UK Pound", "¥": "Yen", "CN¥": "Yuan", "₹": "Rupee", "₽": "Ruble", "₪": "Shekel",
           "R$": "Brazil Real", "A$": "Australian Dollar", "C$": "Canadian Dollar", "MX$": "Mexican Peso", "₿": "Bitcoin"}
CODES = {"EUR": "Euro", "GBP": "UK Pound", "JPY": "Yen", "CNY": "Yuan", "RMB": "Yuan", "INR": "Rupee", "RUB": "Ruble",
         "ILS": "Shekel", "BRL": "Brazil Real", "AUD": "Australian Dollar", "CAD": "Canadian Dollar", "MXN": "Mexican Peso",
         "SAR": "Saudi Riyal", "CHF": "Swiss Franc", "BTC": "Bitcoin"}
NAMES = {"euros?": "Euro", "pounds?": "UK Pound", "yen": "Yen", "yuan": "Yuan", "rupees?": "Rupee", "rubles?": "Ruble",
         "shekels?": "Shekel", "reals?": "Brazil Real", "swiss francs?|francs?": "Swiss Franc", "pesos?": "Mexican Peso",
         "riyals?": "Saudi Riyal", "bitcoins?": "Bitcoin"}

WRITING_RULE = """
Writing amounts
- Each payment has `usd`, its US-dollar equivalent, and `amount`, its value in the currency it was paid in \
(`currency`). Totals and thresholds are in US dollars.
- Write a dollar figure with $ and name the currency in words: "$4,168 paid in Euro". Or write the amount in its \
own currency from `amount`: "€3,557 (about $4,168)".
- Never put a foreign currency symbol, code or name on a `usd` figure. "€4,168" for a payment whose usd is 4,168 \
is wrong.
"""

NUM = r"(\d[\d,]*(?:\.\d+)?)\s?(k|K|m|M|million|thousand)?"
SYM_RX = "|".join(re.escape(s) for s in sorted(SYMBOLS, key=len, reverse=True))
PATTERNS = [
    re.compile(rf"(?P<sym>{SYM_RX})\s?~?{NUM}"),
    re.compile(rf"\b(?P<code>{'|'.join(CODES)})\s?~?\$?{NUM}"),
    re.compile(rf"\$?{NUM}\s?(?P<code2>{'|'.join(CODES)})\b(?! cheques)"),
    # "11.3M Euros" is a mislabel; "$11.3M in Euros" (a dollar figure, currency in words) is the right form
    re.compile(rf"(?<![\$\d.,~]){NUM}\s+(?:in\s+)?(?P<name>{'|'.join(NAMES)})\b", re.I),
]


@lru_cache(maxsize=1)
def rates() -> dict:
    return json.loads(RATES_FILE.read_text())["usd_per_unit"]


def native(usd: float, currency: str) -> float:
    r = rates().get(currency)
    return round(usd / r, 2 if currency != "Bitcoin" else 4) if r else usd


def _value(num: str, unit: str | None) -> float:
    v = float(num.replace(",", ""))
    unit = (unit or "").lower()
    return v * (1e6 if unit in ("m", "million") else 1e3 if unit in ("k", "thousand") else 1)


def labelled_amounts(text: str) -> list[tuple[str, str, float]]:
    """(phrase, currency, value) for every amount written with a non-dollar currency label."""
    out, seen = [], set()
    for rx in PATTERNS:
        for m in rx.finditer(text):
            if any(a <= m.start() < b for a, b in seen):
                continue
            g = m.groupdict()
            if g.get("sym"):
                cur = SYMBOLS[g["sym"]]
            elif g.get("code") or g.get("code2"):
                cur = CODES[(g.get("code") or g.get("code2")).upper()]
            else:
                cur = next(v for k, v in NAMES.items() if re.fullmatch(k, g["name"], re.I))
            nums = [x for x in m.groups() if x and re.match(r"\d", x)]
            units = [x for x in m.groups() if x and x.lower() in ("k", "m", "million", "thousand")]
            if not nums:
                continue
            seen.add((m.start(), m.end()))
            out.append((m.group(0), cur, _value(nums[0], units[0] if units else None)))
    return out


def evidence_payments(trace: list, first_message: str = "") -> list[tuple[float, str]]:
    """(usd, currency) for every payment the agent was shown."""
    text = first_message + "\n" + "\n".join(json.dumps(t.get("output", "")) for t in trace if t.get("type") == "tool")
    pays = re.findall(r'"usd":\s?([\d.]+),\s?(?:"amount":\s?[\d.]+,\s?)?"currency":\s?"([^"]+)"', text)
    return [(float(u), c) for u, c in pays]


def close(a: float, b: float, rel: float = 0.03) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1.0)


def check_amounts(text: str, payments: list[tuple[float, str]]) -> list[str]:
    """Amounts written with a currency label that do not match any payment (or total) in that currency."""
    problems = []
    by_cur: dict[str, list[float]] = {}
    for usd, cur in payments:
        by_cur.setdefault(cur, []).append(usd)
    for phrase, cur, value in labelled_amounts(text):
        natives = [native(u, cur) for u in by_cur.get(cur, [])]
        candidates = natives + ([sum(natives)] if len(natives) > 1 else [])
        if any(close(value, c) for c in candidates):
            continue
        usds = by_cur.get(cur, [])
        hint = " (that is the US-dollar figure)" if any(close(value, u) for u in usds + [sum(usds)]) else ""
        problems.append(f"“{phrase.strip()}” is not an amount paid in {cur}{hint}")
    return problems


def writeup_text(v: dict) -> str:
    return " ".join([v.get("narrative") or "", *(v.get("key_evidence") or []), v.get("innocent_explanation") or ""])
