"""Read-only view of the transaction data for the investigation tools.

The ground-truth label columns are dropped on load, so nothing downstream
(tools or agent) can see the answers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_COLUMNS = ["is_laundering", "pattern_id", "pattern_type"]


class Ledger:
    def __init__(self, tx: pd.DataFrame, accounts: pd.DataFrame | None = None):
        tx = tx.drop(columns=[c for c in LABEL_COLUMNS if c in tx.columns])
        tx = tx.sort_values("timestamp").reset_index(drop=True)
        self.tx = tx
        self.data_start = tx.timestamp.min().normalize()   # first day covered by the data
        out = tx.rename(columns={"from_account": "account", "to_account": "counterparty"}).assign(direction="out")
        inc = tx.rename(columns={"to_account": "account", "from_account": "counterparty"}).assign(direction="in")
        both = pd.concat([out, inc], ignore_index=True).sort_values("timestamp")
        cols = ["txn_id", "timestamp", "direction", "counterparty", "amount", "payment_type"]
        self._by_account = {a: g[cols].reset_index(drop=True) for a, g in both.groupby("account") if a != "CASH"}
        self._empty = both.iloc[0:0][cols]
        self.account_type = {} if accounts is None else dict(zip(accounts.account_id, accounts.account_type))

    @classmethod
    def from_csv(cls, tx_path="data/transactions.csv", accounts_path="data/accounts.csv"):
        if not Path(tx_path).exists() and Path(tx_path + ".gz").exists():
            tx_path += ".gz"   # the repository ships the compressed copy
        tx = pd.read_csv(tx_path, parse_dates=["timestamp"], keep_default_na=False)
        try:
            acc = pd.read_csv(accounts_path)
        except FileNotFoundError:
            acc = None
        return cls(tx, acc)

    def activity(self, account: str, start=None, end=None) -> pd.DataFrame:
        """All transactions touching an account, optionally within [start, end)."""
        df = self._by_account.get(account, self._empty)
        if start is not None:
            df = df[df.timestamp >= start]
        if end is not None:
            df = df[df.timestamp < end]
        return df
