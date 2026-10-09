"""Generate a small synthetic bank-transaction dataset with labeled laundering.

This is a SAMPLE dataset for building and testing the pipeline. Headline
results for the project are measured on the public IBM AML dataset, not here.

Output: data/transactions.csv, data/accounts.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86_400


def build_accounts(rng, n):
    kinds = rng.choice(["retail", "business", "corporate"], size=n, p=[0.85, 0.12, 0.03])
    return pd.DataFrame({"account_id": [f"A{i:05d}" for i in range(n)], "account_type": kinds})


def legit_transactions(rng, acc, days):
    """Ordinary banking activity. Some of it looks suspicious to simple rules."""
    retail = acc.loc[acc.account_type == "retail", "account_id"].to_numpy()
    business = acc.loc[acc.account_type == "business", "account_id"].to_numpy()
    corporate = acc.loc[acc.account_type == "corporate", "account_id"].to_numpy()
    rows = []

    def add(day, src, dst, amount, ptype):
        t = np.asarray(day) * DAY + rng.integers(8 * 3600, 20 * 3600, size=len(src))
        rows.append(pd.DataFrame({"ts": t, "from_account": src, "to_account": dst,
                                  "amount": np.round(amount, 2), "payment_type": ptype}))

    # Everyday card purchases: retail -> business
    n = int(len(retail) * days * 0.5)
    add(rng.integers(0, days, n), rng.choice(retail, n), rng.choice(business, n),
        rng.lognormal(3.6, 0.9, n).clip(3, 2500), "card")

    # Payroll: each employer pays its staff twice a month (legit fan-out)
    employer = rng.choice(np.concatenate([corporate, business]), len(retail))
    salary = rng.lognormal(7.9, 0.45, len(retail)).clip(900, 12_000)
    for d in range(0, days, 15):
        add(np.full(len(retail), d), employer, retail, salary, "ach")

    # Rent / mortgage on the 1st of the month
    landlord = rng.choice(business, len(retail))
    rent = rng.uniform(900, 4200, len(retail))
    for d in range(1, days, 30):
        add(np.full(len(retail), d), retail, landlord, rent, "ach")

    # Person-to-person transfers
    n = int(len(retail) * days * 0.03)
    add(rng.integers(0, days, n), rng.choice(retail, n), rng.choice(retail, n),
        rng.lognormal(4.5, 1.0, n).clip(5, 4000), "p2p")

    # Life events: house deposits, car purchases, tuition (large but innocent)
    n = int(len(retail) * days / 400)
    add(rng.integers(0, days, n), rng.choice(retail, n), rng.choice(business, n),
        rng.uniform(10_000, 85_000, n), "wire")

    # Cash-heavy businesses deposit takings; some land just under $10k
    cash_biz = rng.choice(business, max(1, len(business) // 3), replace=False)
    n = int(len(cash_biz) * days * 0.55)
    add(rng.integers(0, days, n), np.full(n, "CASH"), rng.choice(cash_biz, n),
        rng.lognormal(8.1, 0.6, n).clip(300, 14_000), "cash")

    # Supplier payments: business -> business / corporate
    n = int(len(business) * days * 0.2)
    add(rng.integers(0, days, n), rng.choice(business, n),
        rng.choice(np.concatenate([business, corporate]), n),
        rng.lognormal(8.4, 0.8, n).clip(500, 60_000), "wire")

    # Corporate treasury movements
    n = int(len(corporate) * days * 0.3)
    add(rng.integers(0, days, n), rng.choice(corporate, n), rng.choice(corporate, n),
        rng.lognormal(10.5, 0.8, n).clip(5_000, 500_000), "wire")

    df = pd.concat(rows, ignore_index=True)
    df = df[df.from_account != df.to_account]
    df["is_laundering"] = 0
    df["pattern_id"] = ""
    df["pattern_type"] = ""
    return df


def laundering_transactions(rng, acc, days, n_patterns):
    """Inject labeled laundering typologies. 'Stealth' variants stay under thresholds."""
    pool = acc.loc[acc.account_type != "corporate", "account_id"].to_numpy()
    rows = []

    def emit(pid, ptype, t, src, dst, amount, pay="wire"):
        rows.append((int(t), src, dst, round(float(amount), 2), pay, 1, pid, ptype))

    types = ["structuring", "fan_out", "fan_in", "cycle", "scatter_gather", "low_and_slow"]
    for i in range(n_patterns):
        ptype = types[i % len(types)]
        pid = f"P{i:03d}"
        start = rng.integers(2, days - 12) * DAY + rng.integers(0, DAY)

        if ptype == "structuring":
            a, out = rng.choice(pool, 2, replace=False)
            k = rng.integers(4, 10)
            total = 0.0
            for j in range(k):
                amt = rng.uniform(8_200, 9_950)
                total += amt
                emit(pid, ptype, start + rng.uniform(0, 4 * DAY), "CASH", a, amt, "cash")
            emit(pid, ptype, start + 4.5 * DAY, a, out, total * 0.97)

        elif ptype == "fan_out":
            k = rng.integers(8, 20)
            members = rng.choice(pool, k + 1, replace=False)
            for m in members[1:]:
                emit(pid, ptype, start + rng.uniform(0, 1.5 * DAY), members[0], m,
                     rng.uniform(2_000, 9_500))

        elif ptype == "fan_in":
            k = rng.integers(8, 20)
            members = rng.choice(pool, k + 1, replace=False)
            for m in members[1:]:
                emit(pid, ptype, start + rng.uniform(0, 1.5 * DAY), m, members[0],
                     rng.uniform(2_000, 9_500))

        elif ptype == "cycle":
            k = rng.integers(4, 8)
            members = rng.choice(pool, k, replace=False)
            amt = rng.uniform(12_000, 90_000)
            t = start
            for j in range(k):
                emit(pid, ptype, t, members[j], members[(j + 1) % k], amt)
                amt *= rng.uniform(0.97, 0.995)
                t += rng.uniform(0.1, 0.6) * DAY

        elif ptype == "scatter_gather":
            k = rng.integers(3, 7)
            members = rng.choice(pool, k + 2, replace=False)
            src, sink, mules = members[0], members[1], members[2:]
            for m in mules:
                amt = rng.uniform(4_000, 9_800)
                t = start + rng.uniform(0, DAY)
                emit(pid, ptype, t, src, m, amt)
                emit(pid, ptype, t + rng.uniform(0.2, 1.0) * DAY, m, sink, amt * rng.uniform(0.95, 0.99))

        else:  # low_and_slow: small amounts spread over days, built to dodge rules
            k = rng.integers(4, 7)
            members = rng.choice(pool, k, replace=False)
            for hop in range(k - 1):
                for rep in range(rng.integers(3, 6)):
                    emit(pid, ptype, start + rng.uniform(0, 10 * DAY), members[hop], members[hop + 1],
                         rng.uniform(900, 3_800), "ach")

    cols = ["ts", "from_account", "to_account", "amount", "payment_type",
            "is_laundering", "pattern_id", "pattern_type"]
    return pd.DataFrame(rows, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", type=int, default=3000)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--patterns", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    acc = build_accounts(rng, args.accounts)
    df = pd.concat([legit_transactions(rng, acc, args.days),
                    laundering_transactions(rng, acc, args.days, args.patterns)], ignore_index=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(df))])
    df["timestamp"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(df.pop("ts"), unit="s")
    df = df[["txn_id", "timestamp", "from_account", "to_account", "amount", "payment_type",
             "is_laundering", "pattern_id", "pattern_type"]]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    acc.to_csv(out / "accounts.csv", index=False)
    df.to_csv(out / "transactions.csv", index=False)
    n_l = int(df.is_laundering.sum())
    print(f"accounts={len(acc):,} transactions={len(df):,} laundering_txns={n_l:,} "
          f"({n_l / len(df):.3%}) patterns={args.patterns}")


if __name__ == "__main__":
    main()
