"""Load the prepared IBM AML data and define the evaluation protocol.

Protocol (days are counted from 2022-09-01)
  days 0-3   history only: used to learn which counterparty relationships already exist
  days 4-5   development: rules, features, model and prompts may be tuned here
  day  6     gap, so development and test windows do not overlap much
  days 7-9   held-out test: touched only to report results
  days 10+   dropped. After day 10 almost every transaction is laundering, so a
             system could "detect" it from the date alone.

A handful of institutional hub accounts (over 2,000 transactions in ten days)
touch nearly everything. They are excluded from alerting and scoring, as a bank
would handle correspondent or omnibus accounts under a separate process.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MAIN_DAYS = 10
HISTORY_DAYS = 4
DEV_DAYS = (4, 5)
GAP_DAYS = (6,)
TEST_DAYS = (7, 8, 9)
HUB_TXNS = 2000
WINDOW_BACK, WINDOW_FWD = 2, 1     # an alert on day d is reviewed over days d-2 .. d+1
EPOCH = pd.Timestamp("2022-09-01")


def split_of(day: int) -> str:
    return "dev" if day in DEV_DAYS else "test" if day in TEST_DAYS else "gap" if day in GAP_DAYS else "history"


def load(path: str = "data/ibm_prepared"):
    """Returns (transactions, meta). Adds `day` and `new` (first payment ever seen between this
    sender and receiver, judged only from earlier days, so no future information is used)."""
    p = Path(path)
    meta = json.loads((p / "meta.json").read_text())
    tx = pd.read_parquet(p / "tx.parquet")
    tx["day"] = (tx.minute // 1440).astype("int16")
    n_tail = int((tx.day >= MAIN_DAYS).sum())
    tx = tx[tx.day < MAIN_DAYS]
    volume = pd.concat([tx.src, tx.dst]).value_counts()
    hubs = set(volume[volume > HUB_TXNS].index)
    touches_hub = tx.src.isin(hubs) | tx.dst.isin(hubs)
    meta["protocol"] = {
        "tail_rows_dropped": n_tail, "hub_accounts": len(hubs), "hub_rows_dropped": int(touches_hub.sum()),
        "hub_laundering_rows_dropped": int((touches_hub & (tx.is_laundering == 1)).sum()),
        "history_days": list(range(HISTORY_DAYS)), "dev_days": list(DEV_DAYS), "gap_days": list(GAP_DAYS),
        "test_days": list(TEST_DAYS), "review_window": f"day-{WINDOW_BACK} .. day+{WINDOW_FWD}"}
    return mark_first_time(tx[~touches_hub].reset_index(drop=True)), meta


def mark_first_time(tx: pd.DataFrame) -> pd.DataFrame:
    """Flag payments on the first day a given sender ever pays a given receiver.

    Day 0 is never flagged: with no earlier data, every relationship would look new."""
    pair = tx.src.astype("int64") * 1_000_000 + tx.dst
    tx["new"] = (tx.groupby(pair).day.transform("min") == tx.day) & (tx.day >= 1)
    return tx


def account_names(path: str = "data/ibm_prepared") -> pd.Series:
    return pd.read_parquet(Path(path) / "accounts.parquet").account


def laundering_involvement(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per (account, day, pattern_id) where the account took part in a laundering transaction."""
    bad = tx[tx.is_laundering == 1]
    return pd.concat([bad[["src", "day", "pattern_id"]].rename(columns={"src": "a"}),
                      bad[["dst", "day", "pattern_id"]].rename(columns={"dst": "a"})]).drop_duplicates()
