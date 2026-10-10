"""Evidence features for each IBM alert.

Every feature is something a human investigator could read off the account's
activity in the review window. They feed the machine-learning baseline, and the
same facts are what the agent's tools report.

No feature uses a label. Features ending in `_ach` use the payment format and can
be switched off to check the result does not depend on that dataset quirk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ibm_data as D

FORMAT_FEATURES = ["out_ach", "in_ach"]
ACH = 0   # index of "ACH" in the sorted format list


def _window_copies(df: pd.DataFrame) -> pd.DataFrame:
    """Repeat each row once for every alert day whose review window contains it."""
    return pd.concat([df.assign(aday=df.day + s) for s in range(-D.WINDOW_FWD, D.WINDOW_BACK + 1)])


def _node_window_counts(new: pd.DataFrame, col: str) -> pd.Series:
    """First-time payments made (col='src') or received (col='dst') per account per alert-day window."""
    return _window_copies(new).groupby([col, "aday"]).size()


def build(tx: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    new = tx[tx.new].assign(ach=lambda d: (d.format == ACH).astype("int8"),
                            xccy=lambda d: d.cross_currency.astype("int8"))
    made, received = _node_window_counts(new, "src"), _node_window_counts(new, "dst")
    idx = pd.MultiIndex.from_frame(alerts[["a", "day"]])

    def side(acct: str, cp: str, pre: str) -> pd.DataFrame:
        w = _window_copies(new)
        k = pd.MultiIndex.from_arrays([w[cp], w.aday])
        w["cp_made"] = made.reindex(k).fillna(0).to_numpy()
        w["cp_received"] = received.reindex(k).fillna(0).to_numpy()
        w["cp_made_any"] = (w.cp_made > 0).astype("int8")
        g = w.groupby([acct, "aday"])
        f = pd.DataFrame({
            f"{pre}_payments": g.size(), f"{pre}_parties": g[cp].nunique(),
            f"{pre}_usd": g.amount_usd.sum(), f"{pre}_max_usd": g.amount_usd.max(),
            f"{pre}_currencies": g.currency.nunique(), f"{pre}_ach": g.ach.mean(),
            f"{pre}_cross_currency": g.xccy.mean(),
            f"{pre}_parties_paying_on": g.cp_made_any.mean(),      # share of counterparties that also made first-time payments
            f"{pre}_parties_made": g.cp_made.sum(), f"{pre}_parties_received_max": g.cp_received.max(),
        })
        return f.reindex(idx).reset_index(drop=True)

    X = pd.concat([side("src", "dst", "out"), side("dst", "src", "in")], axis=1).fillna(0)

    # The account's own norm, from the history period only
    hist = tx[tx.day < D.HISTORY_DAYS]
    hist_volume = pd.concat([hist.src, hist.dst]).value_counts()
    hist_new = pd.concat([hist.src[hist.new], hist.dst[hist.new]]).value_counts()
    hist_parties = pd.concat([hist[["src", "dst"]].rename(columns={"src": "a", "dst": "cp"}),
                              hist[["dst", "src"]].rename(columns={"dst": "a", "src": "cp"})]).groupby("a").cp.nunique()
    a = alerts.a
    X["history_transactions"] = a.map(hist_volume).fillna(0).to_numpy()
    X["history_first_time_relationships"] = a.map(hist_new).fillna(0).to_numpy()
    X["history_parties"] = a.map(hist_parties).fillna(0).to_numpy()

    # All activity in the window, first-time or not
    allw = pd.concat([_window_copies(tx[["src", "day", "amount_usd"]].rename(columns={"src": "a"})),
                      _window_copies(tx[["dst", "day", "amount_usd"]].rename(columns={"dst": "a"}))])
    g = allw.groupby(["a", "aday"]).amount_usd
    X["window_transactions"] = g.size().reindex(idx).fillna(0).to_numpy()
    X["window_usd"] = g.sum().reindex(idx).fillna(0).to_numpy()
    X["first_time_share"] = (X.out_payments + X.in_payments) / X.window_transactions.clip(lower=1)
    X["relay_ratio"] = np.minimum(X.out_usd, X.in_usd) / np.maximum(X.out_usd, X.in_usd).clip(lower=1)
    return X
