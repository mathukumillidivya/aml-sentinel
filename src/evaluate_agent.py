"""Score an agent run against ground truth and compare with the rules-only baseline.

Because runs use a stratified sample, rates are measured within each group
(true alerts, false alarms) and then re-weighted to the full alert population.
Every rate is reported with a 95% Wilson confidence interval.

Output: results/agent_runs/<run_id>/metrics.json (+ a summary table in CI)
"""
import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd

TARGET_RETENTION = 0.95
TYPOLOGY_MATCH = {"structuring": {"structuring"}, "fan_out": {"fan_out", "layering"}, "fan_in": {"fan_in", "layering"},
                  "cycle": {"cycle", "layering"}, "scatter_gather": {"layering", "pass_through", "fan_out", "fan_in"},
                  "low_and_slow": {"layering", "pass_through", "other"}}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def rate(k, n):
    return {"count": int(k), "of": int(n), "rate": round(k / n, 4) if n else None, "ci95": wilson(int(k), int(n))}


def evaluate(verdicts: pd.DataFrame, alerts: pd.DataFrame, tx: pd.DataFrame,
             price_in: float | None = None, price_out: float | None = None) -> dict:
    df = verdicts.merge(alerts[["alert_id", "is_true_alert", "patterns"]], on="alert_id", how="left")
    df["linked"] = df.linked_accounts.map(json.loads)
    t, f = df[df.is_true_alert == 1], df[df.is_true_alert == 0]
    N_t, N_f = int((alerts.is_true_alert == 1).sum()), int((alerts.is_true_alert == 0).sum())

    def weighted(esc_t, esc_f):
        """Scale sample rates up to the full alert population."""
        kept_t = N_t * esc_t / len(t) if len(t) else 0
        kept_f = N_f * esc_f / len(f) if len(f) else 0
        kept = kept_t + kept_f
        return {"alerts_for_human_review": round(kept), "of_baseline_alerts": N_t + N_f,
                "workload_reduction": round(1 - kept / (N_t + N_f), 4),
                "false_alarm_reduction": round(1 - kept_f / N_f, 4) if N_f else None,
                "precision": round(kept_t / kept, 4) if kept else None,
                "baseline_precision": round(N_t / (N_t + N_f), 4)}

    esc_t, esc_f = int((t.disposition == "escalate").sum()), int((f.disposition == "escalate").sum())
    m = {
        "sample": {"alerts": len(df), "true_alerts": len(t), "false_alarms": len(f)},
        "true_alerts_escalated": rate(esc_t, len(t)),
        "false_alarms_dismissed": rate(len(f) - esc_f, len(f)),
        "at_agent_decision": weighted(esc_t, esc_f),
    }

    # Best risk-score cut-off that still keeps the target share of true alerts
    best = None
    for thr in range(0, 101):
        kt = int((t.risk_score >= thr).sum())
        if len(t) and kt / len(t) >= TARGET_RETENTION:
            best = (thr, kt, int((f.risk_score >= thr).sum()))
    if best:
        thr, kt, kf = best
        m["at_fixed_retention"] = {"target_retention": TARGET_RETENTION, "risk_threshold": thr,
                                   "true_alerts_kept": rate(kt, len(t)),
                                   "false_alarms_removed": rate(len(f) - kf, len(f)), **weighted(kt, kf)}
        m["at_fixed_retention"]["note"] = "threshold chosen on the same sample; treat as optimistic"

    # Network mapping: did the agent name the other accounts in the scheme?
    bad = tx[tx.is_laundering == 1]
    members = {}
    for r in bad.itertuples():
        members.setdefault(r.pattern_id, set()).update({r.from_account, r.to_account})
    laundering_accounts = set().union(*members.values()) - {"CASH"} if members else set()
    named, named_correct, found, possible = 0, 0, 0, 0
    for r in t[t.disposition == "escalate"].itertuples():
        scheme = set().union(*(members.get(p, set()) for p in r.patterns.split("|") if p)) - {"CASH", r.account}
        linked = set(r.linked)
        named += len(linked)
        named_correct += len(linked & laundering_accounts)
        found += len(linked & scheme)
        possible += len(scheme)
    m["network"] = {"linked_accounts_correct": rate(named_correct, named),
                    "scheme_members_found": rate(found, possible),
                    "invented_accounts_blocked": int(df.dropped_accounts.map(lambda s: len(json.loads(s))).sum())}

    # Did it name the right kind of scheme?
    tx_type = bad.groupby("pattern_id").pattern_type.first().to_dict()
    hits = total = 0
    for r in t[t.disposition == "escalate"].itertuples():
        kinds = {tx_type.get(p) for p in r.patterns.split("|") if p}
        ok = set().union(*(TYPOLOGY_MATCH.get(k, set()) for k in kinds))
        total += 1
        hits += r.typology in ok
    m["typology_correct"] = rate(hits, total)

    tok_in, tok_out = int(df.input_tokens.sum()), int(df.output_tokens.sum())
    m["operations"] = {
        "failsafe_escalations": rate(int((df.status == "failsafe").sum()), len(df)),
        "failure_reasons": df.loc[df.status == "failsafe", "failure_reason"].value_counts().to_dict(),
        "avg_tool_calls": round(df.tool_calls.mean(), 2), "avg_turns": round(df.turns.mean(), 2),
        "avg_seconds": round(df.seconds.mean(), 2),
        "avg_input_tokens": round(tok_in / len(df)), "avg_output_tokens": round(tok_out / len(df)),
        "cached_input_share": round(df.cached_tokens.sum() / tok_in, 4) if tok_in else 0,
    }
    if price_in is not None and price_out is not None:
        cost = (tok_in * price_in + tok_out * price_out) / 1e6
        m["operations"]["cost_usd_total"] = round(cost, 2)
        m["operations"]["cost_usd_per_alert"] = round(cost / len(df), 4)
        m["operations"]["cost_note"] = "list price, ignores cache discounts (upper bound)"
    return m


def summary_markdown(run_id: str, m: dict) -> str:
    a, o = m["at_agent_decision"], m["operations"]

    def pct(x):
        return "n/a" if x is None else f"{x:.1%}"

    def r(d):
        lo, hi = d["ci95"]
        return f"{d['count']}/{d['of']} ({pct(d['rate'])}, 95% CI {pct(lo)}–{pct(hi)})"

    rows = [("True alerts escalated", r(m["true_alerts_escalated"])),
            ("False alarms dismissed", r(m["false_alarms_dismissed"])),
            ("Precision (rules only → with agent)", f"{pct(a['baseline_precision'])} → {pct(a['precision'])}"),
            ("Human review workload", f"{a['of_baseline_alerts']:,} → {a['alerts_for_human_review']:,} alerts"),
            ("Scheme members found by tracing", r(m["network"]["scheme_members_found"])),
            ("Fail-safe escalations", r(o["failsafe_escalations"])),
            ("Tool calls per alert", o["avg_tool_calls"]),
            ("Tokens per alert (in / out)", f"{o['avg_input_tokens']:,} / {o['avg_output_tokens']:,}")]
    if "cost_usd_per_alert" in o:
        rows.append(("Cost per alert", f"${o['cost_usd_per_alert']}"))
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return f"### Agent run `{run_id}`\n\n| Measure | Result |\n|---|---|\n{body}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id or path to the run folder")
    ap.add_argument("--alerts", default="results/alerts.csv")
    ap.add_argument("--data", default="data/transactions.csv")
    ap.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).exists() else Path("results/agent_runs") / args.run
    verdicts = pd.read_csv(run / "verdicts.csv", keep_default_na=False)
    alerts = pd.read_csv(args.alerts, keep_default_na=False)
    data = args.data if Path(args.data).exists() else args.data + ".gz"
    tx = pd.read_csv(data, keep_default_na=False)
    cfg = json.loads((run / "run_config.json").read_text())
    if cfg.get("warmup_days"):   # compare against the same population the agent was sampled from
        cutoff = pd.Timestamp(cfg["data_start"]) + pd.Timedelta(days=cfg["warmup_days"])
        alerts = alerts[pd.to_datetime(alerts.date) >= cutoff]
    m = evaluate(verdicts, alerts, tx, args.price_in, args.price_out)
    m["run"] = cfg
    (run / "metrics.json").write_text(json.dumps(m, indent=2))
    md = summary_markdown(run.name, m)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(md)


if __name__ == "__main__":
    main()
