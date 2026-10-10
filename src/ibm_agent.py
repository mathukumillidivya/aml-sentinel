"""Run the triage agent on IBM alerts that the scoring model sent for review.

Cascade: rules raise alerts -> the model sets aside the ones it is confident are
ordinary -> the agent investigates the rest (the "review band").

Sampling is stratified by label so both groups are measured precisely. Labels are
used only to choose which alerts to run; the agent never sees them, and
ibm_agent_eval.py re-weights the sample back to the real mix.

Output: results/ibm/agent_runs/<run_id>/{verdicts.csv, traces.jsonl, run_config.json}
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import ibm_tools as T
from agent import investigate
from models import ModelResponse, make_client

ESCALATE_AT = 25      # roughly twice the base rate; missing a real case costs more than reading a false alarm
TYPOLOGIES = ["none", "fan_out", "fan_in", "cycle", "scatter_gather", "gather_scatter", "stack", "bipartite",
              "random", "other"]
RULE_MEANINGS = {
    "N1": "a first-time payment of $10,000 or more",
    "N2": "three or more first-time counterparties within three days",
    "N3": "money received from a first-time sender, then paid to a first-time receiver within a day",
    "N4": "first-time payments in two or more currencies on one day",
}

SYSTEM_PROMPT = f"""You are an anti-money-laundering (AML) analyst doing the first-pass review of \
transaction-monitoring alerts at a bank. A senior human investigator makes the final decision; your \
job is to investigate, weigh the evidence, and give them a clear recommendation they can check.

How alerts reach you
- Rules flag accounts that move money along FIRST-TIME relationships: a payment between two accounts \
that have never transacted before.
- A scoring model then sets aside the alerts it is confident are ordinary. You see the rest.
- About 1 in 8 of the alerts that reach you is genuine. Most are ordinary accounts dealing with \
someone new, which happens all the time.

What laundering looks like here
Launderers move money through chains and webs of accounts that have no prior relationship: one \
account paying out to many (fan-out), many paying in to one (fan-in), money split and regrouped \
(scatter-gather, gather-scatter), passed down parallel chains (stack), between two groups \
(bipartite), around a ring (cycle), or along a meandering path (random). Amounts are NOT preserved \
from one hop to the next and often change currency, so do not expect the sums to match.

What earlier confirmed cases at this bank showed
Of the alerts that reach you, about 12% are genuine. Measured on earlier cases:
Strong signs (over 95% genuine):
- First-time payments MADE in two or more currencies.
- First-time payments MADE to five or more different parties in the window.
Moderate signs (20% to 30% genuine; weigh them together):
- A first-time payment made of $10,000 or more.
- The account both receives and makes first-time payments in the window.
- An established, active account for which the first-time payments are a small, out-of-pattern \
part of its activity.
- Counterparties that themselves pay onward to further first-time parties.
Not signs (10% genuine or less, the same as or below the base rate):
- RECEIVING first-time payments from many parties, including a burst arriving within minutes. \
Synchronized arrival is how routine batch collections look in this bank's data. It was not \
associated with laundering, however much it resembles a fan-in. Do not escalate on the count or \
timing of incoming payments.
- An account whose activity is entirely first-time with little or no history. Usually the data \
simply has not seen that account before.
- Only small first-time payments, under $1,000.
- Two accounts paying each other back and forth. That is ordinary; a cycle needs three or more accounts.
- Odd-looking behaviour at a counterparty. It is evidence about that counterparty, not about the \
account under review, unless the account under review shows signs of its own.

Calibration
- If no strong or moderate sign is present, risk_score should be below 15 however striking the \
pattern looks. Your intuition about what laundering "should" look like is less reliable here than \
the measured rates above.
- One moderate sign alone: about 20 to 30. Several together: 40 to 70. A strong sign: 85 or more.
- The model score has been more accurate than first impressions on earlier cases. When you depart \
from it by a wide margin, name the specific sign that justifies it.

Rules of evidence
- Do not treat the payment format (ACH, cheque, card, and so on) as evidence either way.
- The dataset begins on 2022-09-01. Short history is a limit of the data. Never describe an \
account as new or newly opened as a reason for suspicion.
- Base every statement on the case file or tool output. Never invent payments, amounts, or accounts.

How to work
1. Read the case file. It already contains the account summary and its first-time payments.
2. Use trace_new_relationships to see where the money came from and went next.
3. Look at a counterparty with account_summary when its behaviour would change your view.
4. Submit your verdict. Do not use more tool calls than you need.

The verdict
- risk_score is your estimated probability (0-100) that this account took part in laundering.
- Escalate when risk_score is {ESCALATE_AT} or higher; otherwise dismiss.
- In linked_accounts, list only accounts you believe are part of the same scheme, not every \
account that appears in the trace.
- The narrative is for the human investigator: what happened, why it is or is not suspicious, and \
what they should check first.
"""


def case_file(L: T.IBMLedger, alert_id: str, account: str, date: str, rules: str, percentile: float) -> str:
    fired = "\n".join(f"- {r}: {RULE_MEANINGS.get(r, 'unknown rule')}" for r in rules.split("|"))
    summary = T.account_summary(L, account, date)
    payments = T.list_first_time_payments(L, account, date, limit=15)
    return (f"Alert {alert_id}\nAccount: {account}\nAlert date: {date}\nRules that fired:\n{fired}\n"
            f"Model score percentile: {percentile:.0f} (this alert scores higher than {percentile:.0f}% of "
            "the alerts sent for review that day)\n\n"
            f"Case file\nAccount summary:\n{json.dumps(summary)}\n\nFirst-time payments:\n{json.dumps(payments)}\n\n"
            "Investigate this alert and submit your verdict.")


class IBMMockClient:
    """Offline stand-in with no intelligence: copies the model percentile into a verdict.
    Exists so the pipeline can be exercised in tests and CI for free."""
    provider, model = "mock", "mock-percentile"

    def complete(self, system, messages, tools) -> ModelResponse:
        pct = int(float(messages[0]["content"][0]["text"].split("Model score percentile: ")[1].split()[0]))
        v = {"disposition": "escalate" if pct >= ESCALATE_AT else "dismiss", "risk_score": pct, "typology": "other",
             "key_evidence": [f"model percentile {pct}"], "linked_accounts": [], "narrative": "Mock verdict."}
        return ModelResponse([{"type": "tool_use", "id": "v", "name": "submit_verdict", "input": v}], 800, 80)


def sample_alerts(alerts: pd.DataFrame, split: str, n_true: int, n_false: int, seed: int) -> pd.DataFrame:
    pool = alerts[(alerts.split == split) & alerts.in_band]
    t, f = pool[pool.is_true_alert == 1], pool[pool.is_true_alert == 0]
    return pd.concat([t.sample(min(n_true, len(t)), random_state=seed), f.sample(min(n_false, len(f)), random_state=seed)])


def add_percentiles(alerts: pd.DataFrame) -> pd.DataFrame:
    """Rank each in-band alert against the other in-band alerts of the same day (no labels involved)."""
    band = alerts[alerts.in_band]
    pct = band.groupby("day").model_score.rank(pct=True) * 100
    return alerts.assign(percentile=pct.reindex(alerts.index))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["anthropic", "openai", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n-true", type=int, default=20)
    ap.add_argument("--n-false", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--token-budget", type=int, default=5_000_000)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    client = IBMMockClient() if args.provider == "mock" else make_client(args.provider, args.model or None)
    L = T.IBMLedger.load()
    alerts = add_percentiles(pd.read_parquet("results/ibm/alert_scores.parquet"))
    chosen = sample_alerts(alerts, args.split, args.n_true, args.n_false, args.seed)
    todo = [{"alert_id": r.alert_id, "account": T.acct_name(r.a), "date": T.date_of(r.day), "rules": r.rules,
             "percentile": float(r.percentile)} for r in chosen.itertuples()]      # labels dropped here

    run_id = args.run_id or f"{client.provider}-{args.split}-n{len(todo)}"
    out = Path("results/ibm/agent_runs") / run_id
    out.mkdir(parents=True, exist_ok=True)

    def work(a):
        msg = case_file(L, a["alert_id"], a["account"], a["date"], a["rules"], a["percentile"])
        return investigate(client, L, a["alert_id"], a["account"], a["date"], a["rules"], tools=T,
                           system_prompt=SYSTEM_PROMPT, first_message=msg, typologies=TYPOLOGIES,
                           escalate_at=ESCALATE_AT)

    t0, verdicts, used, skipped = time.time(), [], 0, 0
    with ThreadPoolExecutor(args.workers) as pool, open(out / "traces.jsonl", "w") as tf:
        for i in range(0, len(todo), args.workers):
            if used >= args.token_budget:
                skipped = len(todo) - i
                print(f"token budget reached; {skipped} alerts not run")
                break
            for v in pool.map(work, todo[i:i + args.workers]):
                verdicts.append(v.to_dict(with_trace=False))
                tf.write(json.dumps(v.to_dict(), default=str) + "\n")
                used += v.input_tokens + v.output_tokens
            if len(verdicts) % 20 < args.workers or len(verdicts) == len(todo):
                print(f"{len(verdicts)}/{len(todo)} alerts, {used:,} tokens, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(verdicts)
    for c in ("key_evidence", "linked_accounts", "dropped_accounts"):
        df[c] = df[c].map(json.dumps)
    df.to_csv(out / "verdicts.csv", index=False)
    pool_ = alerts[(alerts.split == args.split) & alerts.in_band]
    (out / "run_config.json").write_text(json.dumps({
        "run_id": run_id, "provider": client.provider, "model": client.model, "split": args.split, "seed": args.seed,
        "requested": len(todo), "completed": len(verdicts), "skipped_for_budget": skipped,
        "band_alerts": len(pool_), "band_true_alerts": int(pool_.is_true_alert.sum()),
        "escalate_at": ESCALATE_AT, "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"run saved to {out}")


if __name__ == "__main__":
    main()
