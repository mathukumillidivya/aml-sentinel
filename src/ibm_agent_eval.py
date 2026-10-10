"""Score an IBM agent run, and answer the question that matters: does the agent
add anything to a model that costs nothing to run?

Three rankings of the same sampled alerts are compared:
  model     the gradient-boosted score alone
  agent     the agent's risk score alone
  combined  the plain average of the two (fixed in advance, not fitted)

Rates are re-weighted from the stratified sample to the real mix of the review
band. Uncertainty comes from a paired bootstrap over the sampled alerts.

Output: results/ibm/agent_runs/<run_id>/metrics.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import ibm_data as D
import ibm_tools as T
from evaluate_agent import rate

N_BOOT = 2000


def review_share(y, s, w, keep: float) -> float:
    """Share of (weighted) alerts a human must read, highest score first, to keep `keep` of true alerts.

    Alerts with the same score are indistinguishable to the reader, so true alerts are assumed to
    be spread evenly through a tied group. Without this, the order alerts happen to be stored in
    would decide the answer whenever scores tie."""
    df = pd.DataFrame({"s": s, "t": y * w, "w": w}).groupby("s").sum().sort_index(ascending=False)
    hits, load = df.t.cumsum().to_numpy(), df.w.cumsum().to_numpy()
    need = keep * hits[-1]
    k = int(np.searchsorted(hits, need - 1e-9))
    prev_hits, prev_load = (hits[k - 1], load[k - 1]) if k else (0.0, 0.0)
    frac = (need - prev_hits) / (hits[k] - prev_hits) if hits[k] > prev_hits else 1.0
    return float((prev_load + frac * (load[k] - prev_load)) / load[-1])


def ranking(y, s, w) -> dict:
    return {"auc": round(float(roc_auc_score(y, s, sample_weight=w)), 4),
            "average_precision": round(float(average_precision_score(y, s, sample_weight=w)), 4),
            "review_share_to_keep_80pct": round(review_share(y, s, w, 0.8), 4),
            "review_share_to_keep_90pct": round(review_share(y, s, w, 0.9), 4)}


def bootstrap_gap(y, a, b, w, seed=0) -> dict:
    """95% interval for AUC(a) - AUC(b), resampling true and false alerts separately."""
    rng = np.random.default_rng(seed)
    t, f = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    gaps = []
    for _ in range(N_BOOT):
        i = np.concatenate([rng.choice(t, len(t)), rng.choice(f, len(f))])
        gaps.append(roc_auc_score(y[i], a[i], sample_weight=w[i]) - roc_auc_score(y[i], b[i], sample_weight=w[i]))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"auc_gap": round(float(np.mean(gaps)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "significant": bool(lo > 0 or hi < 0)}


def scheme_members(tx: pd.DataFrame) -> tuple[dict, set]:
    bad = tx[tx.is_laundering == 1]
    members = {}
    for pid, g in bad[bad.pattern_id >= 0].groupby("pattern_id"):
        members[int(pid)] = set(g.src) | set(g.dst)
    return members, set(bad.src) | set(bad.dst)


def evaluate(verdicts: pd.DataFrame, alerts: pd.DataFrame, tx: pd.DataFrame, L: T.IBMLedger, cfg: dict) -> dict:
    df = verdicts.merge(alerts[["alert_id", "a", "day", "is_true_alert", "patterns", "model_score"]], on="alert_id")
    y = df.is_true_alert.to_numpy()
    n_t, n_f = int(y.sum()), int((1 - y).sum())
    N_t, N_f = cfg["band_true_alerts"], cfg["band_alerts"] - cfg["band_true_alerts"]
    w = np.where(y == 1, N_t / n_t, N_f / n_f)
    model = df.model_score.to_numpy()
    agent = df.risk_score.to_numpy() / 100
    combined = (model + agent) / 2

    esc = (df.disposition == "escalate").to_numpy()
    kept_t, kept_f = N_t * esc[y == 1].mean(), N_f * esc[y == 0].mean()
    m = {
        "sample": {"alerts": len(df), "true_alerts": n_t, "false_alarms": n_f},
        "band": {"alerts": N_t + N_f, "true_alerts": N_t, "precision": round(N_t / (N_t + N_f), 4)},
        "ranking": {"model": ranking(y, model, w), "agent": ranking(y, agent, w), "combined": ranking(y, combined, w)},
        "agent_vs_model": bootstrap_gap(y, agent, model, w),
        "combined_vs_model": bootstrap_gap(y, combined, model, w),
        "agent_decision": {
            "escalate_at": cfg["escalate_at"],
            "true_alerts_escalated": rate(int(esc[y == 1].sum()), n_t),
            "false_alarms_dismissed": rate(int((~esc)[y == 0].sum()), n_f),
            "alerts_for_human_review": round(kept_t + kept_f),
            "share_of_band": round((kept_t + kept_f) / (N_t + N_f), 4),
            "precision": round(kept_t / (kept_t + kept_f), 4) if kept_t + kept_f else None},
    }
    # the model alone, asked to keep the same share of true alerts as the agent's decision did
    keep = esc[y == 1].mean()
    m["model_at_same_retention"] = {"true_alerts_kept": round(float(keep), 4),
                                    "share_of_band": round(review_share(y, model, w, keep), 4)}

    # Network mapping, against (a) every account the trace tool returns and (b) the labeled scheme
    members, bad_accounts = scheme_members(tx)
    rows = {"agent": [0, 0, 0, 0], "trace_tool_unfiltered": [0, 0, 0, 0]}   # named, named_bad, found, possible
    for r in df[(df.is_true_alert == 1) & (df.patterns != "")].itertuples():
        scheme = set().union(*(members.get(int(p), set()) for p in r.patterns.split("|"))) - {r.a}
        lo, hi = max(0, r.day - D.WINDOW_BACK), min(L.last_day, r.day + D.WINDOW_FWD)
        traced = T.trace(L, int(r.a), lo, hi)["accounts"]
        named = {T.acct_code(x) for x in json.loads(r.linked_accounts)} if r.disposition == "escalate" else set()
        for key, s in (("agent", named), ("trace_tool_unfiltered", traced)):
            rows[key][0] += len(s)
            rows[key][1] += len(s & bad_accounts)
            rows[key][2] += len(s & scheme)
            rows[key][3] += len(scheme)
    m["network"] = {k: {"accounts_named_correct": rate(v[1], v[0]), "scheme_members_found": rate(v[2], v[3])}
                    for k, v in rows.items()}
    m["network"]["invented_accounts_blocked"] = int(df.dropped_accounts.map(lambda s: len(json.loads(s))).sum())

    tok_in, tok_out = int(df.input_tokens.sum()), int(df.output_tokens.sum())
    m["operations"] = {"failsafe_escalations": rate(int((df.status == "failsafe").sum()), len(df)),
                       "failure_reasons": df.loc[df.status == "failsafe", "failure_reason"].value_counts().to_dict(),
                       "avg_tool_calls": round(df.tool_calls.mean(), 2), "avg_turns": round(df.turns.mean(), 2),
                       "avg_seconds": round(df.seconds.mean(), 2),
                       "avg_input_tokens": round(tok_in / len(df)), "avg_output_tokens": round(tok_out / len(df)),
                       "total_input_tokens": tok_in, "total_output_tokens": tok_out}
    return m


def summary_markdown(run_id: str, m: dict) -> str:
    r, d = m["ranking"], m["agent_decision"]
    g = m["combined_vs_model"]
    lines = [f"### IBM agent run `{run_id}`\n", "| Ranking | AUC | Read to keep 80% | Read to keep 90% |", "|---|---|---|---|"]
    for k in ("model", "agent", "combined"):
        lines.append(f"| {k} | {r[k]['auc']} | {r[k]['review_share_to_keep_80pct']:.1%} | {r[k]['review_share_to_keep_90pct']:.1%} |")
    lines += ["", f"Combined vs model AUC gap: {g['auc_gap']:+.4f} (95% CI {g['ci95'][0]:+.4f} to {g['ci95'][1]:+.4f})", "",
              f"Agent decision: {d['true_alerts_escalated']['count']}/{d['true_alerts_escalated']['of']} true alerts escalated, "
              f"{d['false_alarms_dismissed']['count']}/{d['false_alarms_dismissed']['of']} false alarms dismissed, "
              f"{d['share_of_band']:.1%} of the band left for humans "
              f"(model alone at the same retention: {m['model_at_same_retention']['share_of_band']:.1%})"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).exists() else Path("results/ibm/agent_runs") / args.run
    verdicts = pd.read_csv(run / "verdicts.csv", keep_default_na=False)
    cfg = json.loads((run / "run_config.json").read_text())
    alerts = pd.read_parquet("results/ibm/alert_scores.parquet")
    tx, meta = D.load()
    m = evaluate(verdicts, alerts, tx, T.IBMLedger(tx, meta), cfg)
    m["run"] = cfg
    (run / "metrics.json").write_text(json.dumps(m, indent=2))
    md = summary_markdown(run.name, m)
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write(md)


if __name__ == "__main__":
    main()
