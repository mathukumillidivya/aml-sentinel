"""Build the investigation console (app/site/index.html) for static hosting.

It reuses the case data packed by build_review_screen.py into app/review.html (every number there comes from a results
file), so the two pages can never disagree. Run build_review_screen.py first if the results change.

Usage: python src/build_console.py
"""
import json
import re
from pathlib import Path

FEATURED = "IB044430"   # an escalated gather-and-scatter case with a long agent trail


def cut(t: str, n: int) -> str:
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def prose(run: str, ids: set) -> dict:
    """The write-up checks: a second model's review, the code currency check, and the hand spot-check."""
    from ibm_currency_check import PATTERN
    run = Path(run)
    judge = {j["alert_id"]: j for j in map(json.loads, open(run / "narrative_judge.jsonl"))}
    spot = json.loads((run / "narrative_spot_check.json").read_text())
    hand = {c["alert_id"]: c for c in spot["cases"] + spot.get("also_read", [])}
    cases = {}
    for v in map(json.loads, open(run / "traces.jsonl")):
        if v["status"] != "ok" or v["alert_id"] not in ids:
            continue
        text = " ".join([v["narrative"], *v["key_evidence"], v["innocent_explanation"] or ""])
        m = PATTERN.search(text)
        j = judge.get(v["alert_id"])
        cases[v["alert_id"]] = {"cur": text[max(0, m.start() - 40):m.end() + 20] if m else None,
                                **({"acc": j["accuracy"], "inn": j["innocent_explanation"],
                                    "errs": [{"q": cut(e["quote"], 160), "why": cut(e["evidence_says"], 220)} for e in j["errors"][:2]],
                                    "more": max(0, len(j["errors"]) - 2)} if j else {}),
                                **({"hand": hand[v["alert_id"]]["hand"], "note": hand[v["alert_id"]]["note"]} if v["alert_id"] in hand else {})}
    return {"judge": json.loads((run / "narrative_judge.json").read_text()), "currency": json.loads((run / "currency_check.json").read_text()),
            "spot": {k: spot[k] for k in ("rated_material_errors", "rated_accurate", "pattern", "estimate")}, "cases": cases}


def main():
    review = Path("app/review.html").read_text()
    m = re.search(r"const DATA = ", review)
    data, _ = json.JSONDecoder().raw_decode(review[m.end():])
    data.pop("guidance", None)
    data["featured"] = FEATURED
    # USD value of one unit of each currency, the same rates used to convert amounts in ibm_prepare.py,
    # so the page can show each payment in the currency it was made in.
    data["rates"] = json.loads(Path("results/ibm/usd_rates_HI-Small.json").read_text())["usd_per_unit"]
    data["prose"] = prose("results/ibm/agent_runs/ibm-test-haiku", {c["id"] for c in data["cases"]})
    page = Path("app/console_template.html").read_text()
    out = page.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    Path("app/site").mkdir(exist_ok=True)
    Path("app/site/index.html").write_text(out)
    print(f"app/site/index.html: {len(out) / 1e6:.2f} MB, {len(data['cases'])} cases")


if __name__ == "__main__":
    main()
