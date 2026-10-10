"""Run the triage agent over a sample of alerts and save verdicts plus full traces.

Sampling
  stratified (default): half true alerts, half false alarms (or all true alerts
  if fewer). Labels are used ONLY to pick which alerts to run so that both groups
  are measured precisely; the agent never sees them. evaluate_agent.py re-weights
  the sample back to the real mix.
  random: a plain random sample.

Warm-up: alerts from the first 30 days of data are excluded, because no account
has enough history yet to be compared with its own past. evaluate_agent.py
applies the same cut to the baseline so the comparison is like for like.

Output: results/agent_runs/<run_id>/{verdicts.csv, traces.jsonl, run_config.json}
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from agent import investigate
from ledger import Ledger
from models import make_client

AGENT_COLUMNS = ["alert_id", "account", "date", "rules"]   # everything the agent is allowed to see


def eligible(alerts: pd.DataFrame, data_start, warmup_days: int) -> pd.DataFrame:
    """Drop alerts from the warm-up period, when too little history exists to judge any account."""
    cutoff = pd.Timestamp(data_start) + pd.Timedelta(days=warmup_days)
    return alerts[pd.to_datetime(alerts.date) >= cutoff]


def pick_alerts(alerts: pd.DataFrame, n: int, how: str, seed: int) -> pd.DataFrame:
    n = min(n, len(alerts))
    if how == "random":
        return alerts.sample(n, random_state=seed)
    true, false = alerts[alerts.is_true_alert == 1], alerts[alerts.is_true_alert == 0]
    n_true = min(len(true), n // 2)
    n_false = min(len(false), n - n_true)
    return pd.concat([true.sample(n_true, random_state=seed), false.sample(n_false, random_state=seed)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["anthropic", "openai", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sample", default="stratified", choices=["stratified", "random"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--warmup-days", type=int, default=30,
                    help="skip alerts in the first N days of data (not enough history to compare against)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--token-budget", type=int, default=20_000_000,
                    help="stop starting new alerts once this many tokens have been used")
    ap.add_argument("--alerts", default="results/alerts.csv")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    client = make_client(args.provider, args.model or None)
    ledger = Ledger.from_csv()
    alerts = eligible(pd.read_csv(args.alerts, keep_default_na=False), ledger.data_start, args.warmup_days)
    chosen = pick_alerts(alerts, args.n, args.sample, args.seed)
    todo = chosen[AGENT_COLUMNS].to_dict("records")   # labels dropped here

    run_id = args.run_id or f"{client.provider}-{client.model}-n{len(todo)}".replace("/", "_")
    out = Path("results/agent_runs") / run_id
    out.mkdir(parents=True, exist_ok=True)

    def work(a):
        return investigate(client, ledger, a["alert_id"], a["account"], a["date"], a["rules"])

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
            done = len(verdicts)
            if done % 20 < args.workers or done == len(todo):
                print(f"{done}/{len(todo)} alerts, {used:,} tokens, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(verdicts)
    for c in ("key_evidence", "linked_accounts", "dropped_accounts"):
        df[c] = df[c].map(json.dumps)
    df.to_csv(out / "verdicts.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({
        "run_id": run_id, "provider": client.provider, "model": client.model, "requested": args.n,
        "completed": len(verdicts), "skipped_for_budget": skipped, "sample": args.sample, "seed": args.seed,
        "alerts_total": len(alerts),
        "warmup_days": args.warmup_days, "data_start": ledger.data_start.strftime("%Y-%m-%d"), "wall_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"run saved to {out}")


if __name__ == "__main__":
    main()
