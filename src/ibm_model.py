"""Machine-learning baseline for alert triage on IBM data.

This is the bar the AI agent has to clear. A small gradient-boosted model scores
every alert in milliseconds for free; an agent that cannot add to it has not
earned its cost.

Trained on development days only, reported on held-out test days.

Output: results/ibm/model_baseline.json, results/ibm/alert_scores.parquet
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import ibm_data as D
import ibm_features as F

RETENTION_LEVELS = (0.5, 0.8, 0.9, 0.95)
BAND_RETENTION = 0.95      # the agent reviews the alerts the model cannot confidently set aside


def workload_curve(y: np.ndarray, score: np.ndarray) -> dict:
    """If humans review alerts from the highest score down, how much must they read to keep X% of true alerts?"""
    order = np.argsort(-score, kind="stable")
    hits = np.cumsum(y[order])
    out = {}
    for level in RETENTION_LEVELS:
        k = int(np.searchsorted(hits, np.ceil(level * y.sum())) + 1)
        out[f"keep_{int(level * 100)}pct_true"] = {
            "alerts_reviewed": k, "share_of_alerts": round(k / len(y), 4),
            "workload_reduction": round(1 - k / len(y), 4), "precision": round(float(hits[k - 1]) / k, 4)}
    return out


def fit_and_score(X, alerts, cols, seed=0):
    dev, test = (alerts.split == "dev").to_numpy(), (alerts.split == "test").to_numpy()
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    model.fit(X.loc[dev, cols], alerts.is_true_alert[dev])
    return model, model.predict_proba(X[cols])[:, 1], dev, test


def out_of_fold_scores(X, alerts, cols, dev, seed=0) -> np.ndarray:
    """Score each development alert with a model that never saw that alert's account.

    The final model has memorised the development alerts, so its scores on them are
    too good to be honest. These are the scores to show the agent during development
    and to set the review-band threshold."""
    Xd, yd, groups = X.loc[dev, cols], alerts.is_true_alert[dev].to_numpy(), alerts.a[dev].to_numpy()
    oof = np.zeros(len(yd))
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(Xd, yd, groups):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
        oof[te] = m.fit(Xd.iloc[tr], yd[tr]).predict_proba(Xd.iloc[te])[:, 1]
    return oof


def report(y, s) -> dict:
    return {"alerts": len(y), "true_alerts": int(y.sum()), "baseline_precision": round(float(y.mean()), 4),
            "auc": round(float(roc_auc_score(y, s)), 4), "average_precision": round(float(average_precision_score(y, s)), 4),
            **workload_curve(y, s)}


def main():
    tx, _ = D.load()
    alerts = pd.read_parquet("results/ibm/alerts.parquet")
    X = F.build(tx, alerts)
    y = alerts.is_true_alert.to_numpy()

    cols = list(X.columns)
    _, score, dev, test = fit_and_score(X, alerts, cols)
    no_format = [c for c in cols if c not in F.FORMAT_FEATURES]
    _, score_nf, _, _ = fit_and_score(X, alerts, no_format)

    oof = out_of_fold_scores(X, alerts, cols, dev)
    honest = score.copy()
    honest[dev] = oof                                   # never show memorised scores
    threshold = float(np.quantile(oof[y[dev] == 1], 1 - BAND_RETENTION))
    in_band = honest >= threshold

    seen_in_dev = alerts.a.isin(set(alerts.a[dev])).to_numpy()
    strict = test & ~seen_in_dev
    n_rules = alerts.rules.str.count(r"\|").to_numpy() + 1     # naive ranking: how many rules fired
    m = {
        "trained_on": {"days": list(D.DEV_DAYS), "alerts": int(dev.sum()), "true_alerts": int(y[dev].sum())},
        "features": cols,
        "test": report(y[test], score[test]),
        "test_without_payment_format": report(y[test], score_nf[test]),
        "test_accounts_never_alerted_in_dev": report(y[strict], score[strict]),
        "test_ranked_by_rule_count_only": report(y[test], n_rules[test] + 1e-6 * X.out_parties.to_numpy()[test]),
        "dev_out_of_fold": report(y[dev], oof),
        "review_band": {
            "rule": f"model score at or above the level that keeps {BAND_RETENTION:.0%} of development true alerts (out-of-fold)",
            "threshold": round(threshold, 5),
            "dev": {"alerts_in_band": int((in_band & dev).sum()), "share_of_alerts": round(float(in_band[dev].mean()), 4),
                    "true_alerts_kept": round(float(in_band[dev & (y == 1)].mean()), 4)},
            "test": {"alerts_in_band": int((in_band & test).sum()), "share_of_alerts": round(float(in_band[test].mean()), 4),
                     "true_alerts_in_band": int((in_band & test & (y == 1)).sum()),
                     "true_alerts_kept": round(float(in_band[test & (y == 1)].mean()), 4),
                     "precision_in_band": round(float(y[in_band & test].mean()), 4)}},
        "note": "test days are held out; no feature uses a label",
    }
    out = Path("results/ibm")
    (out / "model_baseline.json").write_text(json.dumps(m, indent=2))
    alerts.assign(model_score=honest, model_score_no_format=score_nf, in_band=in_band).to_parquet(out / "alert_scores.parquet", index=False)
    X.assign(alert_id=alerts.alert_id).to_parquet(out / "alert_features.parquet", index=False)
    print(json.dumps({k: v for k, v in m.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
