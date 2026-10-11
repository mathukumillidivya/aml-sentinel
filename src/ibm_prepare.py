"""Convert the raw IBM AML file into a compact table the project can load quickly.

What it does
  * Account IDs become bank-qualified (the same account number can exist at two banks).
  * Self-transfers (an account paying itself, mostly "Reinvestment") are dropped.
  * Amounts are converted to US dollars, using exchange rates implied by the
    dataset's own cross-currency transactions where available.
  * Each laundering transaction is tagged with its scheme from the Patterns file.

Output (data/ibm_prepared/): tx.parquet, accounts.parquet, meta.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Approximate September 2022 rates (USD per unit); used only if the data implies none.
FALLBACK_USD = {"US Dollar": 1.0, "Euro": 1.0, "Yuan": 0.145, "Yen": 0.007, "Rupee": 0.0125, "Ruble": 0.0165,
                "UK Pound": 1.16, "Canadian Dollar": 0.76, "Australian Dollar": 0.68, "Swiss Franc": 1.03,
                "Mexican Peso": 0.05, "Brazil Real": 0.19, "Saudi Riyal": 0.266, "Shekel": 0.29, "Bitcoin": 20000.0}
EPOCH = pd.Timestamp("2022-09-01")
KEY = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid", "Payment Currency", "Payment Format"]


def implied_usd_rates(tx: pd.DataFrame) -> tuple[dict, dict]:
    """USD value of one unit of each currency, from transactions that convert to or from dollars."""
    x = tx[tx["Payment Currency"] != tx["Receiving Currency"]]
    rates, support = {"US Dollar": 1.0}, {}
    a = x[x["Receiving Currency"] == "US Dollar"]
    b = x[x["Payment Currency"] == "US Dollar"]
    obs = pd.concat([pd.DataFrame({"ccy": a["Payment Currency"], "r": a["Amount Received"] / a["Amount Paid"]}),
                     pd.DataFrame({"ccy": b["Receiving Currency"], "r": b["Amount Paid"] / b["Amount Received"]})])
    obs = obs[np.isfinite(obs.r) & (obs.r > 0)]
    for ccy, g in obs.groupby("ccy"):
        support[ccy] = int(len(g))
        if len(g) >= 20:
            rates[ccy] = float(g.r.median())
    return rates, support


def read_patterns(path: Path) -> pd.DataFrame:
    rows, pid, ptype = [], -1, None
    for line in path.read_text().splitlines():
        if line.startswith("BEGIN"):
            pid += 1
            ptype = line.split(" - ", 1)[1].split(":")[0].strip()
        elif line and line[0].isdigit():
            f = line.split(",")
            rows.append((f[0], int(f[1]), f[2], int(f[3]), f[4], float(f[7]), f[8], f[9], pid, ptype))
    return pd.DataFrame(rows, columns=KEY + ["pattern_id", "pattern_type"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/ibm")
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--out", default="data/ibm_prepared")
    args = ap.parse_args()
    d, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tx = pd.read_csv(d / f"{args.name}_Trans.csv")
    n_raw = len(tx)
    rates, support = implied_usd_rates(tx)
    usd = {**FALLBACK_USD, **rates}
    unknown = sorted(set(tx["Payment Currency"]) - set(usd))
    if unknown:
        raise SystemExit(f"no exchange rate for {unknown}")

    pat = read_patterns(d / f"{args.name}_Patterns.txt")
    pat_unique = pat.drop_duplicates(KEY)
    tx = tx.merge(pat_unique, on=KEY, how="left")
    assert len(tx) == n_raw, "pattern join changed the row count"
    matched = int(tx.pattern_id.notna().sum())

    src = tx["From Bank"].astype(str) + "_" + tx["Account"]
    dst = tx["To Bank"].astype(str) + "_" + tx["Account.1"]
    self_tx = src == dst
    self_laundering = int((self_tx & (tx["Is Laundering"] == 1)).sum())
    tx, src, dst = tx[~self_tx], src[~self_tx], dst[~self_tx]

    codes, names = pd.factorize(pd.concat([src, dst], ignore_index=True), sort=True)
    n = len(tx)
    cur_names = sorted(usd)
    fmt_names = sorted(tx["Payment Format"].unique())
    type_names = sorted(pat.pattern_type.unique())
    ts = pd.to_datetime(tx["Timestamp"], format="%Y/%m/%d %H:%M")
    prepared = pd.DataFrame({
        "minute": ((ts - EPOCH).dt.total_seconds() // 60).astype("int32").to_numpy(),
        "src": codes[:n].astype("int32"), "dst": codes[n:].astype("int32"),
        "amount_usd": (tx["Amount Paid"] * tx["Payment Currency"].map(usd)).astype("float64").to_numpy(),
        "currency": tx["Payment Currency"].map({c: i for i, c in enumerate(cur_names)}).astype("int8").to_numpy(),
        "cross_currency": (tx["Payment Currency"] != tx["Receiving Currency"]).to_numpy(),
        "format": tx["Payment Format"].map({c: i for i, c in enumerate(fmt_names)}).astype("int8").to_numpy(),
        "is_laundering": tx["Is Laundering"].astype("int8").to_numpy(),
        "pattern_id": tx.pattern_id.fillna(-1).astype("int16").to_numpy(),
        "pattern_type": tx.pattern_type.map({c: i for i, c in enumerate(type_names)}).fillna(-1).astype("int8").to_numpy(),
    }).sort_values("minute", kind="stable").reset_index(drop=True)
    part = 4_500_000                    # keep each file under GitHub's 100 MB limit
    if len(prepared) <= part:
        prepared.to_parquet(out / "tx.parquet", compression="zstd", compression_level=19, index=False)
    else:
        for i, lo in enumerate(range(0, len(prepared), part)):
            prepared.iloc[lo:lo + part].to_parquet(out / f"tx-{i}.parquet", compression="zstd", compression_level=19, index=False)
    pd.DataFrame({"account": names}).to_parquet(out / "accounts.parquet", compression="zstd", index=False)

    meta = {
        "source": "IBM Transactions for Anti Money Laundering (AML), Altman et al., NeurIPS 2023 (CDLA-Sharing-1.0)",
        "variant": args.name, "epoch": str(EPOCH), "rows_raw": n_raw, "rows": len(prepared),
        "self_transfers_dropped": int(self_tx.sum()), "self_transfers_dropped_laundering": self_laundering,
        "accounts": len(names), "currencies": cur_names, "formats": fmt_names, "pattern_types": type_names,
        "usd_rates": {c: usd[c] for c in cur_names}, "usd_rates_from_data": sorted(rates),
        "usd_rate_observations": support,
        "laundering_rows": int(prepared.is_laundering.sum()),
        "pattern_lines": len(pat), "pattern_lines_duplicate_keys": len(pat) - len(pat_unique),
        "laundering_rows_with_pattern": int((prepared.pattern_id >= 0).sum()),
        "pattern_lines_matched_before_self_drop": matched, "patterns": int(pat.pattern_id.nunique()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    for f in sorted(out.iterdir()):
        print(f.name, f"{f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
