"""Investigation tools for the IBM benchmark.

The IBM data rewards a different kind of evidence from the synthetic sample:
amounts are unrelated from one hop to the next, schemes run over several days and
currencies, and the telling sign is money moving along relationships that did not
exist before. So these tools are built around first-time relationships and the
network they form.

`IBMLedger` drops every label column on load, so tools and agent cannot see answers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ibm_data as D

LABEL_COLUMNS = ["is_laundering", "pattern_id", "pattern_type"]
MAX_TRACE_ACCOUNTS = 40
MAX_CYCLE_STEPS = 4000


def acct_name(code: int) -> str:
    return f"A{int(code):06d}"


def acct_code(name: str) -> int:
    return int(str(name).lstrip("A"))


def day_of(as_of: str) -> int:
    return int((pd.Timestamp(as_of).normalize() - D.EPOCH).days)


def date_of(day: int) -> str:
    return (D.EPOCH + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d")


def _when(minute: int) -> str:
    return (D.EPOCH + pd.Timedelta(minutes=int(minute))).strftime("%m-%d %H:%M")


class IBMLedger:
    def __init__(self, tx: pd.DataFrame, meta: dict):
        tx = tx.drop(columns=[c for c in LABEL_COLUMNS if c in tx.columns])
        self.currencies, self.formats = meta["currencies"], meta["formats"]
        cols = ["minute", "day", "amount_usd", "currency", "format", "new"]
        out = tx[["src", "dst"] + cols].rename(columns={"src": "acct", "dst": "cp"}).assign(out=True)
        inc = tx[["dst", "src"] + cols].rename(columns={"dst": "acct", "src": "cp"}).assign(out=False)
        both = pd.concat([out, inc], ignore_index=True).sort_values(["acct", "minute"], kind="stable").reset_index(drop=True)
        self.both = both
        accts = both.acct.to_numpy()
        self._uniq, self._start = np.unique(accts, return_index=True)
        self._stop = np.append(self._start[1:], len(accts))
        self.last_day = int(both.day.max())

    @classmethod
    def load(cls, path: str = "data/ibm_prepared"):
        tx, meta = D.load(path)
        return cls(tx, meta)

    def activity(self, code: int, day_lo: int | None = None, day_hi: int | None = None) -> pd.DataFrame:
        """All payments touching an account (both directions), optionally limited to days day_lo..day_hi inclusive."""
        i = np.searchsorted(self._uniq, code)
        if i >= len(self._uniq) or self._uniq[i] != code:
            return self.both.iloc[0:0]
        df = self.both.iloc[self._start[i]:self._stop[i]]
        if day_lo is not None:
            df = df[df.day >= day_lo]
        if day_hi is not None:
            df = df[df.day <= day_hi]
        return df

    def window(self, as_of: str) -> tuple[int, int]:
        d = day_of(as_of)
        return max(0, d - D.WINDOW_BACK), min(self.last_day, d + D.WINDOW_FWD)


# Version 2 of the write-up step (src/ibm_currency.py) also shows each payment's amount in the currency it was
# paid in. Off by default so the frozen runs are reproduced exactly.
NATIVE = False


def _amt(usd, cur_code, L) -> dict:
    if not NATIVE:
        return {}
    from ibm_currency import native
    return {"amount": native(float(usd), L.currencies[cur_code])}


def _r(x) -> float:
    return round(float(x), 2)


def _side(L: IBMLedger, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"payments": 0, "parties": 0, "usd": 0.0, "largest_usd": 0.0, "currencies": [], "formats": {}}
    return {"payments": int(len(df)), "parties": int(df.cp.nunique()), "usd": _r(df.amount_usd.sum()),
            "largest_usd": _r(df.amount_usd.max()),
            "currencies": sorted({L.currencies[c] for c in df.currency.unique()}),
            "formats": {L.formats[f]: int(n) for f, n in df.format.value_counts().items()}}


def account_summary(L: IBMLedger, account: str, as_of: str) -> dict:
    """What did this account do in the review window, how much of it involved first-time relationships, and what is normal for it?"""
    code = acct_code(account)
    lo, hi = L.window(as_of)
    w = L.activity(code, lo, hi)
    h = L.activity(code, 0, lo - 1)
    new = w[w.new]
    days_before = lo
    usable = max(0, lo - 1)           # day 0 can never show a first-time relationship
    hist_new = h[h.new]
    out = {
        "account": account, "window": f"{date_of(lo)} to {date_of(hi)}",
        "all_activity_in_window": {"payments_made": int(w.out.sum()), "payments_received": int((~w.out).sum()),
                                   "parties": int(w.cp.nunique()), "usd_paid": _r(w.amount_usd[w.out].sum()),
                                   "usd_received": _r(w.amount_usd[~w.out].sum())},
        "first_time_payments_made": _side(L, new[new.out]),
        "first_time_payments_received": _side(L, new[~new.out]),
        "share_of_window_payments_that_are_first_time": _r(len(new) / len(w)) if len(w) else 0.0,
        "before_the_window": {
            "days_of_data": int(days_before), "payments": int(len(h)), "parties": int(h.cp.nunique()),
            "payments_per_day": _r(len(h) / days_before) if days_before else None,
            "first_time_parties_per_day": _r(hist_new.cp.nunique() / usable) if usable else None,
        },
    }
    if usable < 2:
        out["before_the_window"]["note"] = ("Very little earlier data exists because the dataset starts on 2022-09-01. "
                                            "That is a limit of the data, not a sign the account is new.")
    return out


def list_first_time_payments(L: IBMLedger, account: str, as_of: str, limit: int = 25) -> dict:
    """The individual first-time payments in the window, with a note on what each counterparty did next."""
    code = acct_code(account)
    lo, hi = L.window(as_of)
    w = L.activity(code, lo, hi)
    new = w[w.new].sort_values("minute")
    rows = []
    for r in new.head(limit).itertuples():
        cpw = L.activity(r.cp, lo, hi)
        cpn = cpw[cpw.new]
        rows.append({"time": _when(r.minute), "direction": "paid" if r.out else "received from",
                     "counterparty": acct_name(r.cp), "usd": _r(r.amount_usd), **_amt(r.amount_usd, r.currency, L),
                     "currency": L.currencies[r.currency], "format": L.formats[r.format],
                     "counterparty_first_time_payments_made": int(cpn.out.sum()),
                     "counterparty_first_time_payments_received": int((~cpn.out).sum()),
                     "counterparty_total_payments_in_window": int(len(cpw))})
    return {"account": account, "first_time_payments": int(len(new)), "shown": len(rows), "payments": rows}


def get_transactions(L: IBMLedger, account: str, as_of: str, limit: int = 30) -> dict:
    """Every payment in the window, largest first, each marked as first-time or repeat."""
    code = acct_code(account)
    lo, hi = L.window(as_of)
    w = L.activity(code, lo, hi).sort_values("amount_usd", ascending=False)
    rows = [{"time": _when(r.minute), "direction": "paid" if r.out else "received from", "counterparty": acct_name(r.cp),
             "usd": _r(r.amount_usd), **_amt(r.amount_usd, r.currency, L), "currency": L.currencies[r.currency],
             "format": L.formats[r.format], "first_time": bool(r.new)} for r in w.head(limit).itertuples()]
    return {"account": account, "payments_in_window": int(len(w)), "shown": len(rows), "payments": rows}


def _new_edges(L: IBMLedger, code: int, lo: int, hi: int, outgoing: bool) -> pd.DataFrame:
    w = L.activity(code, lo, hi)
    return w[w.new & (w.out == outgoing)]


def trace(L: IBMLedger, code: int, lo: int, hi: int, hops: int = 3) -> dict:
    """Follow first-time payments forward (where money went next) and backward (where it came from),
    respecting time order. Returns raw structure; used by the tool and by the feature builder."""
    start_minute = lo * 1440
    edges, seen = [], {code}
    down_levels, up_levels = [], []
    max_out_deg = max_in_deg = 0
    truncated = False

    frontier = [(code, start_minute - 1)]
    for _ in range(hops):                               # downstream: each hop must happen after the previous one
        nxt = []
        for a, after in frontier:
            e = _new_edges(L, a, lo, hi, True)
            e = e[e.minute > after]
            max_out_deg = max(max_out_deg, e.cp.nunique())
            for r in e.itertuples():
                edges.append((a, r.cp, r.minute, r.amount_usd, r.currency, r.format))
                if r.cp not in seen:
                    if len(seen) >= MAX_TRACE_ACCOUNTS:
                        truncated = True
                        continue
                    seen.add(r.cp)
                    nxt.append((r.cp, r.minute))
        down_levels.append(len(nxt))
        frontier = nxt

    frontier = [(code, (hi + 1) * 1440)]
    for _ in range(hops):                               # upstream: each hop must happen before the next one
        nxt = []
        for a, before in frontier:
            e = _new_edges(L, a, lo, hi, False)
            e = e[e.minute < before]
            max_in_deg = max(max_in_deg, e.cp.nunique())
            for r in e.itertuples():
                edges.append((r.cp, a, r.minute, r.amount_usd, r.currency, r.format))
                if r.cp not in seen:
                    if len(seen) >= MAX_TRACE_ACCOUNTS:
                        truncated = True
                        continue
                    seen.add(r.cp)
                    nxt.append((r.cp, r.minute))
        up_levels.append(len(nxt))
        frontier = nxt

    return {"accounts": seen - {code}, "edges": edges, "downstream_per_hop": down_levels,
            "upstream_per_hop": up_levels, "largest_fan_out": int(max_out_deg), "largest_fan_in": int(max_in_deg),
            "cycle": find_cycle(L, code, lo, hi), "truncated": truncated}


def find_cycle(L: IBMLedger, code: int, lo: int, hi: int, max_hops: int = 12) -> list:
    """A time-ordered chain of first-time payments that leaves the account and returns to it. Empty if none."""
    steps = 0

    def walk(path, after):
        nonlocal steps
        e = _new_edges(L, path[-1], lo, hi, True)
        for r in e[e.minute > after].itertuples():
            steps += 1
            if steps > MAX_CYCLE_STEPS:
                return []
            if r.cp == code and len(path) >= 2:
                return path + [code]
            if r.cp in path or len(path) >= max_hops:
                continue
            found = walk(path + [r.cp], r.minute)
            if found:
                return found
        return []

    return walk([code], lo * 1440 - 1)


def trace_new_relationships(L: IBMLedger, account: str, as_of: str, hops: int = 3) -> dict:
    """Follow the money along first-time relationships: where it went next and where it came from, up to three hops, in time order."""
    code = acct_code(account)
    lo, hi = L.window(as_of)
    t = trace(L, code, lo, hi, hops)
    edges = sorted(set(t["edges"]), key=lambda e: e[2])
    out = {
        "account": account,
        "accounts_reached": len(t["accounts"]),
        "downstream_accounts_per_hop": t["downstream_per_hop"],
        "upstream_accounts_per_hop": t["upstream_per_hop"],
        "largest_fan_out_at_any_account": t["largest_fan_out"],
        "largest_fan_in_at_any_account": t["largest_fan_in"],
        "cycle_back_to_this_account": [acct_name(c) for c in t["cycle"]],
        "currencies_along_the_way": sorted({L.currencies[e[4]] for e in edges}),
        "payments": [{"time": _when(e[2]), "from": acct_name(e[0]), "to": acct_name(e[1]), "usd": _r(e[3]),
                      **_amt(e[3], e[4], L), "currency": L.currencies[e[4]], "format": L.formats[e[5]]} for e in edges[:40]],
        "linked_accounts": sorted(acct_name(c) for c in t["accounts"]),
        "truncated": t["truncated"] or len(edges) > 40,
    }
    if not edges:
        out["note"] = "No first-time payments lead onward from or into this account in the window."
    return out


TOOLS = {f.__name__: f for f in (account_summary, list_first_time_payments, trace_new_relationships, get_transactions)}

_BASE = {"account": {"type": "string", "description": "Account ID, e.g. A012345"},
         "as_of": {"type": "string", "description": "Alert date, YYYY-MM-DD"}}

TOOL_SPECS = [{"name": n, "description": " ".join(f.__doc__.split()),
               "input_schema": {"type": "object", "properties": _BASE, "required": ["account", "as_of"]}}
              for n, f in TOOLS.items()]


def call_tool(ledger: IBMLedger, name: str, args: dict) -> dict:
    """Single entry point for the agent. Unknown tools and bad input return an error, never a crash."""
    if name not in TOOLS:
        return {"error": f"unknown tool '{name}'", "available": sorted(TOOLS)}
    try:
        return TOOLS[name](ledger, **args)
    except (TypeError, ValueError, KeyError) as e:
        return {"error": f"bad arguments for {name}: {e}"}
