"""Build the investigation console (app/site/index.html) for static hosting.

It reuses the case data packed by build_review_screen.py into app/review.html (every number there comes from a results
file), so the two pages can never disagree. Run build_review_screen.py first if the results change.

Usage: python src/build_console.py
"""
import json
import re
from pathlib import Path

FEATURED = "IB044430"   # an escalated gather-and-scatter case with a long agent trail


def main():
    review = Path("app/review.html").read_text()
    m = re.search(r"const DATA = ", review)
    data, _ = json.JSONDecoder().raw_decode(review[m.end():])
    data.pop("guidance", None)
    data["featured"] = FEATURED
    page = Path("app/console_template.html").read_text()
    out = page.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    Path("app/site").mkdir(exist_ok=True)
    Path("app/site/index.html").write_text(out)
    print(f"app/site/index.html: {len(out) / 1e6:.2f} MB, {len(data['cases'])} cases")


if __name__ == "__main__":
    main()
