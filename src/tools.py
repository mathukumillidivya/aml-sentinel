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


def _coverage(ledger: Ledger, start) -> dict:
    """How much earlier data exists. Short history usually means the dataset begins
    there, which says nothing about the age of the account."""
    days = max(0, (start - ledger.data_start).days)
    out = {"data_starts": ledger.data_start.strftime("%Y-%m-%d"), "days_of_data_before_window": int(days)}
    if days < 14:
        out["note"] = ("Little or no earlier data exists because the dataset itself starts on this date. "
                       "This is NOT evidence that the account is new; history-based comparisons are unavailable.")
    return out


def _prior_windows(ledger: Ledger, start, win_days: int, max_windows: int = 12, min_data_days: int = 0):
    """Earlier back-to-back windows of the same length as the review window, newest first."""
    out, hi = [], start
    while len(out) < max_windows:
        lo = hi - pd.Timedelta(days=win_days)
        if (lo - ledger.data_start).days < min_data_days:
            break
        out.append((lo, hi))
        hi = lo
    return out


def _material(df):
    return df[(df.amount >= MATERIAL) & (df.counterparty != "CASH")]


def _pass_pairs(win) -> list:
    """Inflows matched to a similar-sized outflow to a different party within 48 hours."""
    win = win[win.amount >= MATERIAL]
    ins, outs = win[win.direction == "in"], win[win.direction == "out"]
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
    return pairs


def _burst(ledger: Ledger, account: str, start, end, direction: str) -> set:
    """Counterparties in a one-day burst to/from mostly-new accounts, with no similar day before. Empty if none."""
    w = _material(ledger.activity(account, start, end))
    w = w[w.direction == direction]
    if w.empty:
        return set()
    per_day = w.groupby(w.timestamp.dt.normalize()).counterparty.agg(set)
    cps = max(per_day, key=len)
    if len(cps) < 5:
        return set()
    h = _material(ledger.activity(account, end=start))
    h = h[h.direction == direction]
    if len(cps - set(h.counterparty)) / len(cps) < 0.7:
        return set()
    prior = h.groupby(h.timestamp.dt.normalize()).counterparty.nunique()
    if (prior >= max(3, 0.6 * len(cps))).any():
        return set()
    # a burst often straddles midnight: include new counterparties on the adjacent days too
    peak = max(per_day.index, key=lambda d: len(per_day[d]))
    near = [d for d in per_day.index if abs((d - peak).days) <= 1]
    return set().union(*(per_day[d] for d in near)) - set(h.counterparty)


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

    return {
        "account": account,
        "account_type": ledger.account_type.get(account, "unknown"),
        **_coverage(ledger, start),
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
    # How often does this account normally deal with new counterparties? (its own base rate)
    shares = []
    for lo, hi in _prior_windows(ledger, start, lookback_days + 2, min_data_days=14):
        pw = set(_material(ledger.activity(account, lo, hi)).counterparty)
        if pw:
            shares.append(len(pw - set(ledger.activity(account, end=lo).counterparty)) / len(pw))
    return {"account": account, "counterparties": len(rows), "new_counterparties": new,
            "new_share": _r(new / len(rows)) if rows else 0.0,
            "usual_new_share_in_earlier_periods": _r(pd.Series(shares).median()) if shares else None,
            "earlier_periods_compared": len(shares), "top": rows[:limit]}


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
    pairs = _pass_pairs(win)
    total_in = win[(win.direction == "in") & (win.amount >= MATERIAL)].amount.sum()
    matched = sum(p["in_amount"] for p in pairs)
    prior = _prior_windows(ledger, start, lookback_days + 2)
    prior_hits = sum(bool(_pass_pairs(ledger.activity(account, lo, hi))) for lo, hi in prior)
    return {"account": account, "matched_pairs": len(pairs), "matched_amount": _r(matched),
            "share_of_inflow_passed_through": _r(matched / total_in) if total_in else 0.0,
            "earlier_periods_with_matched_pairs": prior_hits, "earlier_periods_compared": len(prior),
            "pairs": pairs[:10]}


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
    cov = _coverage(ledger, start)
    hist_days = cov["days_of_data_before_window"]
    prior = _prior_windows(ledger, start, win_days)

    res = {"account": account, **cov, "enough_history": hist_days >= 14, "earlier_periods_compared": len(prior)}
    for d in ("in", "out"):
        w, h = win[win.direction == d], hist[hist.direction == d]
        usual = h.amount.sum() / hist_days * win_days if hist_days else 0.0
        totals = [ledger.activity(account, lo, hi).query("direction == @d").amount.sum() for lo, hi in prior]
        res[d] = {
            "window_total": _r(w.amount.sum()),
            "usual_total_for_same_length": _r(usual),
            "ratio_to_usual": _r(w.amount.sum() / usual) if usual > 0 else None,
            "earlier_periods_with_higher_total": int(sum(t >= w.amount.sum() for t in totals)),
            "highest_earlier_period_total": _r(max(totals)) if totals else None,
            "largest_in_window": _r(w.amount.max()) if len(w) else 0.0,
            "largest_ever_before": _r(h.amount.max()) if len(h) else 0.0,
        }
    return res


def trace_network(ledger: Ledger, account: str, as_of: str, lookback_days: int = 7,
                  hops: int = 2, max_accounts: int = 40) -> dict:
    """Which other accounts are tied to this one by specific suspicious behaviour?

    A link is reported only when backed by one of: funds relayed through an account
    within 48 hours, a one-day burst to or from mostly-new accounts, or a cycle.
    Ordinary payments are not followed, so an empty result supports an innocent reading.
    """
    start, end = _window(as_of, lookback_days)
    seen, frontier, links = {account}, [account], []

    def link(a, b, reason):
        if b in ("CASH", a):
            return
        links.append({"a": a, "b": b, "reason": reason})
        if b not in seen and len(seen) <= max_accounts:
            seen.add(b)
            nxt.append(b)

    for hop in range(hops):
        nxt = []
        for a in frontier:
            win = ledger.activity(a, start, end)
            for p in _pass_pairs(win):
                link(a, p["in_from"], "sent funds that this account forwarded within 48h")
                link(a, p["out_to"], "received funds this account forwarded within 48h")
            for d, word in (("out", "paid in"), ("in", "paid into")):
                for cp in _burst(ledger, a, start, end, d):
                    link(a, cp, f"{word} a one-day burst involving mostly-new accounts")
            known = set(ledger.activity(a, end=start).counterparty)
            for cp in set(_material(win).counterparty) - known:
                for p in _pass_pairs(ledger.activity(cp, start, end)):
                    if a in (p["in_from"], p["out_to"]):
                        link(a, cp, "relayed funds to or from this account within 48h")
                        link(cp, p["out_to"] if p["in_from"] == a else p["in_from"], "other end of the relayed funds")
                for d in ("out", "in"):
                    if a in _burst(ledger, cp, start, end, d):
                        link(a, cp, "hub of a one-day burst that includes this account")
            if hop == 0:
                for c in find_cycles(ledger, a, as_of, lookback_days)["cycles"]:
                    for member in c["path"]:
                        link(a, member, "member of a cycle returning funds to this account")
        frontier = nxt
    out = {"account": account, "linked_accounts": sorted(seen - {account}), "count": len(seen) - 1,
           "links": links[:60], "truncated": len(seen) > max_accounts}
    if not links:
        out["note"] = "No relayed funds, bursts, or cycles connect this account to others in the window."
    return out


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
