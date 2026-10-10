"""Profile the public IBM AML dataset so the adapter is built on what the data
actually contains, not on assumptions. Writes a summary and a small sample.

Dataset: "IBM Transactions for Anti Money Laundering (AML)", Altman et al., NeurIPS 2023.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/ibm")
    ap.add_argument("--name", default="HI-Small")
    args = ap.parse_args()
    d = Path(args.dir)
    out = Path("results/ibm")
    out.mkdir(parents=True, exist_ok=True)
    sample = Path("data/ibm_sample")
    sample.mkdir(parents=True, exist_ok=True)
    prof = {"files": {p.name: p.stat().st_size for p in sorted(d.iterdir())}}

    tx = pd.read_csv(d / f"{args.name}_Trans.csv")
    prof["columns"] = {c: str(t) for c, t in tx.dtypes.items()}
    prof["rows"] = len(tx)
    tx.head(2000).to_csv(sample / f"{args.name}_head.csv", index=False)

    cols = list(tx.columns)
    label = next((c for c in cols if "launder" in c.lower()), None)
    ts = next((c for c in cols if "time" in c.lower()), None)
    if ts:
        t = pd.to_datetime(tx[ts])
        prof["time"] = {"min": str(t.min()), "max": str(t.max()),
                        "per_day": t.dt.date.astype(str).value_counts().sort_index().to_dict()}
    if label:
        y = tx[label]
        prof["laundering"] = {"count": int(y.sum()), "rate": float(y.mean())}
        if ts:
            prof["laundering"]["per_day"] = t[y == 1].dt.date.astype(str).value_counts().sort_index().to_dict()
        tx[y == 1].head(500).to_csv(sample / f"{args.name}_laundering_head.csv", index=False)
    for c in cols:
        if pd.api.types.is_string_dtype(tx[c]) and tx[c].nunique() <= 40:
            g = tx.groupby(c)
            prof.setdefault("categorical", {})[c] = {
                str(k): {"n": int(n), "laundering_rate": float(r) if label else None}
                for (k, n), r in zip(g.size().items(), g[label].mean() if label else [None] * g.ngroups)}
    for c in cols:
        if tx[c].dtype.kind == "f":
            prof.setdefault("numeric", {})[c] = {str(q): float(v) for q, v in
                                                 tx[c].quantile([0, .01, .25, .5, .75, .9, .99, .999, 1]).items()}
    acct_cols = [c for c in cols if "account" in c.lower()]
    bank_cols = [c for c in cols if "bank" in c.lower()]
    prof["unique"] = {c: int(tx[c].nunique()) for c in acct_cols + bank_cols}
    if len(acct_cols) == 2 and len(bank_cols) == 2:
        src = tx[bank_cols[0]].astype(str) + "_" + tx[acct_cols[0]].astype(str)
        dst = tx[bank_cols[1]].astype(str) + "_" + tx[acct_cols[1]].astype(str)
        prof["accounts_bank_qualified"] = int(pd.concat([src, dst]).nunique())
        prof["self_transfers"] = {"n": int((src == dst).sum()),
                                  "laundering": int(((src == dst) & (tx[label] == 1)).sum()) if label else None}
        deg = pd.concat([src, dst]).value_counts()
        prof["txns_per_account"] = {str(q): float(v) for q, v in deg.quantile([.5, .9, .99, .999, 1]).items()}
        if label:
            bad = pd.concat([src[tx[label] == 1], dst[tx[label] == 1]]).nunique()
            prof["laundering"]["accounts_involved"] = int(bad)
    cur = [c for c in cols if "currency" in c.lower()]
    if len(cur) == 2:
        prof["cross_currency_share"] = float((tx[cur[0]] != tx[cur[1]]).mean())

    pat = d / f"{args.name}_Patterns.txt"
    if pat.exists():
        lines = pat.read_text().splitlines()
        (sample / f"{args.name}_Patterns_head.txt").write_text("\n".join(lines[:120]) + "\n")
        begins = [ln for ln in lines if ln.upper().startswith("BEGIN")]
        prof["patterns"] = {"lines": len(lines), "attempts": len(begins),
                            "by_type": pd.Series([b.split("-", 1)[-1].strip() for b in begins]).value_counts().to_dict(),
                            "transactions_listed": sum(1 for ln in lines if ln and ln[0].isdigit())}
    (out / f"profile_{args.name}.json").write_text(json.dumps(prof, indent=2, default=str))
    print(json.dumps({k: v for k, v in prof.items() if k not in ("categorical", "numeric")}, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
