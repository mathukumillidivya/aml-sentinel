"""Investigation tools the triage agent can call.

Each tool answers one question an AML investigator would ask about an alerted
account, and returns plain JSON-style data. Tools only report evidence; they
never decide whether an alert is suspicious.

Every tool takes the account, the alert date (`as_of`), and looks at a review
window of `lookback_days` before the alert through the day after it.
"""
from __future__ import annotations

import pandas as pd

from ledger import Ledger

MATERIAL = 1_000           # ignore small everyday payments when looking at flows
STRUCT_LOW, STRUCT_HIGH = 8_000, 10_000
MAX_CYCLE_HOPS = 8
MAX_CYCLE_STEPS = 20_000   # safety cap on the cycle search


def _window(as_of, lookback_days):
    day = pd.Timestamp(as_of).normalize()
    return day - pd.Timedelta(days=lookback_days), day + pd.Timedelta(days=2)


def _r(x, n=2):
    return round(float(x), n)


def get_account_profile(ledger: Ledger, account: str, as_of: str, lookback_days: int = 7) -> dict:
    """Who is this account and what does its normal activity look like?"""
    start, end = _window(as_of, lookback_days)
    hist = ledger.activity(account, end=start)
    win = ledger.activity(account, start, end)

    def side(df, d):
        s = df[df.direction == d]
        return {"count": int(len(s)), "total": _r(s.amount.sum()),
                "median": _r(s.amount.median()) if len(s) else 0.0,
                "max": _r(s.amount.max()) if len(s) else 0.0}

    days_hist = max(1, (start - hist.timestamp.min()).days) if len(hist) else 0
    return {
        "account": account,
        "account_type": ledger.account_type.get(account, "unknown"),
        "history_days": int(days_hist),
        "history": {"in": side(hist, "in"), "out": side(hist, "out")},
        "window": {"in": side(win, "in"), "out": side(win, "out")},
        "window_days": lookback_days + 2,
    }


def get_transactions(ledger: Ledger, account: str, as_of: str, lookback_days: int = 7,
                     min_amount: float = 0, limit: int = 40) -> dict:
    """The raw transactions in the review window, largest first."""
    start, end = _window(as_of, lookback_days)
    win = ledger.activity(account, start, end)
    win = win[win.amount >= min_amount].sort_values("amount", ascending=False)
    rows = [{"txn_id": r.txn_id, "time": r.timestamp.strftime("%Y-%m-%d %H:%M"), "direction": r.direction,
             "counterparty": r.counterparty, "amount": _r(r.amount), "type": r.payment_type}
            for r in win.head(limit).itertuples()]
    return {"account": account, "total_in_window": int(len(win)), "shown": len(rows), "transactions": rows}


def get_counterparties(ledger: Ledger, account: str, as_of: str, lookback_days: int = 7, limit: int = 15) -> dict:
    """Who did the account deal with, and has it dealt with them before?"""
    start, end = _window(as_of, lookback_days)
    win = ledger.activity(account, start, end)
    win = win[(win.amount >= MATERIAL) & (win.counterparty != "CASH")]
    known = set(ledger.activity(account, end=start).counterparty)
    rows = []
    for cp, g in win.groupby("counterparty"):
        rows.append({"counterparty": cp, "counterparty_type": ledger.account_type.get(cp, "unknown"),
                     "sent": _r(g.loc[g.direction == "out", "amount"].sum()),
                     "received": _r(g.loc[g.direction == "in", "amount"].sum()),
                     "transactions": int(len(g)), "seen_before": cp in known})
    rows.sort(key=lambda r: -(r["sent"] + r["received"]))
    new = sum(not r["seen_before"] for r in rows)
    return {"account": account, "counterparties": len(rows), "new_counterparties": new,
            "new_share": _r(new / len(rows)) if rows else 0.0, "top": rows[:limit]}


def check_structuring(ledger: Ledger, account: str, as_of: str, lookback_days: int = 7) -> dict:
    """Are cash deposits clustered just under the $10,000 reporting limit?"""
    start, end = _window(as_of, lookback_days)

    def band_stats(df):
        cash = df[(df.payment_type == "cash") & (df.direction == "in")]
        band = cash[cash.amount.between(STRUCT_LOW, STRUCT_HIGH, inclusive="left")]
        return cash, band

    cash, band = band_stats(ledger.activity(account, start, end))
    hcash, hband = band_stats(ledger.activity(account, end=start))
    return {
        "account": account,
        "cash_deposits": int(len(cash)),
        "deposits_just_under_limit": int(len(band)),
        "just_under_total": _r(band.amount.sum()),
        "just_under_share": _r(len(band) / len(cash)) if len(cash) else 0.0,
        "prior_cash_deposits": int(len(hcash)),
        "prior_just_under_share": _r(len(hband) / len(hcash)) if len(hcash) else 0.0,
    }


def check_fan_pattern(ledger: Ledger, account: str, as_of: str, lookback_days: int = 7) -> dict:
    """Is money spraying out to, or funnelling in from, many accounts at once?"""
    start, end = _window(as_of, lookback_days)
    win = ledger.activity(account, start, end)
    win = win[(win.amount >= MATERIAL) & (win.counterparty != "CASH")]
    hist = ledger.activity(account, end=start)
    hist = hist[(hist.amount >= MATERIAL) & (hist.counterparty != "CASH")]
    out = {"account": account}
    for d, name in (("out", "fan_out"), ("in", "fan_in")):
        w, h = win[win.direction == d], hist[hist.direction == d]
        known = set(h.counterparty)
        cps = set(w.counterparty)
        per_day = w.groupby(w.timestamp.dt.normalize()).counterparty.nunique()
        prior_per_day = h.groupby(h.timestamp.dt.normalize()).counterparty.nunique()
        out[name] = {
            "distinct_counterparties": len(cps),
            "max_in_one_day": int(per_day.max()) if len(per_day) else 0,
            "total": _r(w.amount.sum()),
            "new_counterparty_share": _r(len(cps - known) / len(cps)) if cps else 0.0,
            "prior_days_with_similar_fan": int((prior_per_day >= max(3, 0.6 * (per_day.max() if len(per_day) else 0))).sum()),
        }
    return out


def check_pass_through(ledger: Ledger, account: str, as_of: str, lookback_days: int = 7) -> dict:
    """Does money arrive and leave again almost untouched within 48 hours (mule behaviour)?"""
    start, end = _window(as_of, lookback_days)
    win = ledger.activity(account, start, end)
    win = win[win.amount >= MATERIAL]
    ins = win[win.direction == "in"]
    outs = win[(win.direction == "out")].copy()
    used, pairs = set(), []
    for i in ins.itertuples():
        cand = outs[(outs.timestamp > i.timestamp) & (outs.timestamp <= i.timestamp + pd.Timedelta(hours=48))
                    & outs.amount.between(0.9 * i.amount, 1.0 * i.amount) & ~outs.index.isin(used)
                    & (outs.counterparty != i.counterparty)]
        if len(cand):
            o = cand.iloc[0]
            used.add(cand.index[0])
            pairs.append({"in_from": i.counterparty, "in_amount": _r(i.amount), "out_to": o.counterparty,
                          "out_amount": _r(o.amount),
                          "hours_held": _r((o.timestamp - i.timestamp).total_seconds() / 3600, 1)})
    total_in = ins.amount.sum()
    matched = sum(p["in_amount"] for p in pairs)
    return {"account": account, "matched_pairs": len(pairs), "matched_amount": _r(matched),
            "share_of_inflow_passed_through": _r(matched / total_in) if total_in else 0.0, "pairs": pairs[:10]}


def find_cycles(ledger: Ledger, account: str, as_of: str, lookback_days: int = 7) -> dict:
    """Does money leave this account and come back through a ring of other accounts?"""
    start, end = _window(as_of, lookback_days)
    end = end + pd.Timedelta(days=3)
    steps, found = 0, []

    def out_edges(acct, after, lo, hi):
        df = ledger.activity(acct, after, end)
        df = df[(df.direction == "out") & (df.timestamp > after) & df.amount.between(lo, hi)]
        return df

    def walk(path, last_time, last_amount):
        nonlocal steps
        if len(found) >= 3 or steps > MAX_CYCLE_STEPS:
            return
        for e in out_edges(path[-1], last_time, 0.85 * last_amount, 1.02 * last_amount).itertuples():
            steps += 1
            if e.counterparty == account and len(path) >= 3:
                found.append({"path": path + [account], "hops": len(path), "final_amount": _r(e.amount)})
                return
            if e.counterparty in path or e.counterparty == "CASH" or len(path) >= MAX_CYCLE_HOPS:
                continue
            walk(path + [e.counterparty], e.timestamp, e.amount)

    first = ledger.activity(account, start, end)
    first = first[(first.direction == "out") & (first.amount >= MATERIAL) & (first.counterparty != "CASH")]
    for e in first.itertuples():
        if len(found) >= 3:
            break
        walk([account, e.counterparty], e.timestamp, e.amount)
        if found and "start_amount" not in found[-1]:
            found[-1]["start_amount"] = _r(e.amount)
    return {"account": account, "cycles_found": len(found), "cycles": found,
            "search_truncated": steps > MAX_CYCLE_STEPS}


def compare_to_history(ledger: Ledger, account: str, as_of: str, lookback_days: int = 7) -> dict:
    """Is the activity in the window out of character for this account?"""
    start, end = _window(as_of, lookback_days)
    hist = ledger.activity(account, end=start)
    win = ledger.activity(account, start, end)
    win_days = lookback_days + 2
    hist_days = max(1, (start - hist.timestamp.min()).days) if len(hist) else 0

    def per_period(df, days):
        return df.amount.sum() / days * win_days if days else 0.0

    res = {"account": account, "history_days": int(hist_days), "enough_history": hist_days >= 14}
    for d in ("in", "out"):
        w, h = win[win.direction == d], hist[hist.direction == d]
        usual = per_period(h, hist_days)
        res[d] = {
            "window_total": _r(w.amount.sum()),
            "usual_total_for_same_length": _r(usual),
            "ratio_to_usual": _r(w.amount.sum() / usual) if usual > 0 else None,
            "largest_in_window": _r(w.amount.max()) if len(w) else 0.0,
            "largest_ever_before": _r(h.amount.max()) if len(h) else 0.0,
        }
    return res


def trace_network(ledger: Ledger, account: str, as_of: str, lookback_days: int = 7,
                  hops: int = 2, max_accounts: int = 40) -> dict:
    """Which other accounts are linked to this one through unusual, material payments?

    Follows payments of $1,000+ to or from counterparties the account had not
    dealt with before the window. Used to map the wider network behind a real case.
    """
    start, end = _window(as_of, lookback_days)
    seen, frontier, links = {account}, [account], []
    for _ in range(hops):
        nxt = []
        for a in frontier:
            win = ledger.activity(a, start, end)
            win = win[(win.amount >= MATERIAL) & (win.counterparty != "CASH")]
            known = set(ledger.activity(a, end=start).counterparty)
            for cp, g in win[~win.counterparty.isin(known)].groupby("counterparty"):
                links.append({"a": a, "b": cp, "amount": _r(g.amount.sum())})
                if cp not in seen and len(seen) < max_accounts:
                    seen.add(cp)
                    nxt.append(cp)
        frontier = nxt
    return {"account": account, "linked_accounts": sorted(seen - {account}), "count": len(seen) - 1,
            "links": links[:60], "truncated": len(seen) >= max_accounts}


TOOLS = {f.__name__: f for f in (get_account_profile, get_transactions, get_counterparties, check_structuring,
                                 check_fan_pattern, check_pass_through, find_cycles, compare_to_history,
                                 trace_network)}

_BASE = {"account": {"type": "string", "description": "Account ID, e.g. A00123"},
         "as_of": {"type": "string", "description": "Alert date, YYYY-MM-DD"},
         "lookback_days": {"type": "integer", "description": "Days before the alert to review (default 7)"}}

TOOL_SPECS = [{"name": n, "description": " ".join(f.__doc__.split()),
               "input_schema": {"type": "object", "properties": _BASE, "required": ["account", "as_of"]}}
              for n, f in TOOLS.items()]


def call_tool(ledger: Ledger, name: str, args: dict) -> dict:
    """Single entry point the agent uses. Unknown tools and bad input return an error, never a crash."""
    if name not in TOOLS:
        return {"error": f"unknown tool '{name}'", "available": sorted(TOOLS)}
    try:
        return TOOLS[name](ledger, **args)
    except (TypeError, ValueError) as e:
        return {"error": f"bad arguments for {name}: {e}"}
