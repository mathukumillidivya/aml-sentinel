"""A library of past cases the agent can consult.

The first agent lost to the scoring model because the model had learned from
thousands of labeled alerts and the agent had only written guidance. This gives
the agent the same kind of experience in a form it can read: for any alert, the
most similar alerts from the development period and how each one turned out.

Only development-period alerts are in the library, so a test alert can never
retrieve its own answer or any other test-day outcome. When the query is itself
a development alert, every alert on the same account is left out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

K = 25
SHOWN = 6


class CaseLibrary:
    def __init__(self, alerts: pd.DataFrame, features: pd.DataFrame):
        """`alerts` and `features` are row-aligned. Rows outside the development split are never stored."""
        dev = (alerts.split == "dev").to_numpy()
        self.cols = list(features.columns)
        z = np.log1p(features.clip(lower=0).to_numpy(dtype=float))
        self.scaler = StandardScaler().fit(z[dev])
        self.z = self.scaler.transform(z)                       # every alert can be a query
        self.row_of = {a: i for i, a in enumerate(alerts.alert_id)}
        self.past = np.flatnonzero(dev)                         # only these can be retrieved
        self.past_account = alerts.a.to_numpy()[self.past]
        self.past_outcome = alerts.is_true_alert.to_numpy()[self.past]
        self.past_facts = features.iloc[self.past].reset_index(drop=True)
        self.base_rate = float(self.past_outcome.mean())
        self.index = NearestNeighbors(n_neighbors=K + 12).fit(self.z[self.past])
        self.account_of = dict(zip(alerts.alert_id, alerts.a))

    def similar(self, alert_id: str) -> dict:
        q = self.z[self.row_of[alert_id]].reshape(1, -1)
        _, idx = self.index.kneighbors(q)
        keep = [i for i in idx[0] if self.past_account[i] != self.account_of[alert_id]][:K]
        outcomes = self.past_outcome[keep]

        def brief(i):
            f = self.past_facts.iloc[i]
            return {"outcome": "confirmed laundering" if self.past_outcome[i] else "cleared",
                    "first_time_parties_paid": int(f.out_parties), "currencies_paid_out": int(f.out_currencies),
                    "largest_first_time_payment_made_usd": round(float(f.out_max_usd)),
                    "first_time_parties_paying_in": int(f.in_parties),
                    "payments_before_the_window": int(f.history_transactions)}

        return {"similar_past_cases": len(keep), "confirmed_laundering": int(outcomes.sum()),
                "share_confirmed": round(float(outcomes.mean()), 2),
                "share_confirmed_among_all_past_alerts": round(self.base_rate, 2),
                "closest": [brief(i) for i in keep[:SHOWN]]}
