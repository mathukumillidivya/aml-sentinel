"""Rule-based transaction monitoring: the 'before' system.

Mirrors the threshold rules banks commonly run. Each rule fires per account per
day; multiple rules on the same account-day merge into one alert, which is what
a human investigator would have to open and review.

Output: results/alerts.csv
"""
import argparse
from pathlib import Path

import pandas as pd

LARGE_TXN = 10_000          # R1: single transaction at or above this
STRUCT_LOW, STRUCT_HIGH = 8_000, 10_000   # R2: cash deposits just under the limit
STRUCT_COUNT, STRUCT_DAYS = 3, 7
FAN_MIN_AMOUNT, FAN_COUNT = 1_000, 6      # R3/R4: many counterparties in one day
RAPID_MIN_IN, RAPID_RATIO = 5_000, 0.8    # R5: money in, then straight back out


def run_rules(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["date"] = tx.timestamp.dt.normalize()
    hits = []

    def hit(frame, rule):
        f = frame[["account", "date"]].drop_duplicates()
        f["rule"] = rule
        hits.append(f)

    # R1 large transaction: flag the account moving the money
    big = tx[tx.amount >= LARGE_TXN].copy()
    big["account"] = big.from_account.where(big.from_account != "CASH", big.to_account)
    hit(big, "R1_LARGE_TXN")

    # R2 structuring: repeated cash deposits just under the reporting limit
    cash = tx[(tx.payment_type == "cash") & tx.amount.between(STRUCT_LOW, STRUCT_HIGH, inclusive="left")]
    if len(cash):
        daily = cash.groupby(["to_account", "date"]).size().rename("n").reset_index()
        daily = daily.set_index("date").groupby("to_account").n.rolling(f"{STRUCT_DAYS}D").sum().reset_index()
        daily = daily[daily.n >= STRUCT_COUNT].rename(columns={"to_account": "account"})
        hit(daily, "R2_STRUCTURING")

    # R3 fan-out / R4 fan-in: many distinct counterparties in a day
    mid = tx[(tx.amount >= FAN_MIN_AMOUNT) & (tx.from_account != "CASH")]
    out_n = mid.groupby(["from_account", "date"]).to_account.nunique().reset_index()
    hit(out_n[out_n.to_account >= FAN_COUNT].rename(columns={"from_account": "account"}), "R3_FAN_OUT")
    in_n = mid.groupby(["to_account", "date"]).from_account.nunique().reset_index()
    hit(in_n[in_n.from_account >= FAN_COUNT].rename(columns={"to_account": "account"}), "R4_FAN_IN")

    # R5 rapid movement: large inflow, most of it leaving the same or next day
    inflow = tx.groupby(["to_account", "date"]).amount.sum().rename("inflow")
    inflow.index.names = ["account", "date"]
    outflow = tx[tx.from_account != "CASH"].groupby(["from_account", "date"]).amount.sum().rename("outflow")
    outflow.index.names = ["account", "date"]
    nxt = outflow.reset_index()
    nxt["date"] = nxt.date - pd.Timedelta(days=1)
    nxt = nxt.set_index(["account", "date"]).outflow.rename("outflow_next")
    flow = pd.concat([inflow, outflow, nxt], axis=1).fillna(0).reset_index()
    rapid = flow[(flow.inflow >= RAPID_MIN_IN) &
                 (flow.outflow + flow.outflow_next >= RAPID_RATIO * flow.inflow)]
    hit(rapid, "R5_RAPID_MOVEMENT")

    allhits = pd.concat(hits, ignore_index=True)
    alerts = (allhits.groupby(["account", "date"]).rule
              .agg(lambda r: "|".join(sorted(set(r)))).rename("rules").reset_index())
    alerts = alerts.sort_values(["date", "account"]).reset_index(drop=True)
    alerts.insert(0, "alert_id", [f"AL{i:06d}" for i in range(len(alerts))])
    return alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--out", default="results/alerts.csv")
    args = ap.parse_args()
    tx = pd.read_csv(args.data, parse_dates=["timestamp"])
    alerts = run_rules(tx)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(args.out, index=False)
    print(f"alerts={len(alerts):,} accounts_alerted={alerts.account.nunique():,}")


if __name__ == "__main__":
    main()
