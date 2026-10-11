"""Replicate the IBM result on a second dataset variant, and test transfer.

Two questions:
  1. Replication. Run the identical pipeline (same rules, features, model
     settings, time split) on another IBM variant. Does the result hold?
  2. Transfer. Take the model trained on HI-Small and apply it, unchanged, to
     the other variant's test days. Does what it learned carry over to data it
     has never seen from a different simulation?

Nothing is re-tuned for the second dataset.

Output: results/ibm/<variant>/{rules_baseline.json, model_baseline.json, alerts.parquet}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import ibm_data as D
import ibm_features as F
import ibm_model as M
import ibm_rules as R


def scheme_capture(alerts: pd.DataFrame, score: np.ndarray, test: np.ndarray, top_k: int) -> dict:
    """Of true test alerts that belong to a labeled scheme, how many fall in the top_k read first?"""
    t = alerts[test].assign(s=score[test]).sort_values("s", ascending=False)
    t = t.assign(in_top=np.arange(len(t)) < top_k)
    true = t[t.is_true_alert == 1]
    labeled = true[true.patterns != ""]
    return {"labeled_scheme_alerts": len(labeled), "in_top": int(labeled.in_top.sum()),
            "other_true_alerts": len(true) - len(labeled), "other_in_top": int(true.in_top.sum() - labeled.in_top.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="LI-Small")
    ap.add_argument("--source", default="results/ibm", help="where the HI-Small alerts and features live")
    args = ap.parse_args()
    out = Path("results/ibm") / args.name
    out.mkdir(parents=True, exist_ok=True)

    tx, meta = D.load(f"data/ibm_prepared_{args.name}")
    assert meta["formats"][F.ACH] == "ACH"
    alerts = R.build_alerts(tx)
    rules = {"protocol": meta["protocol"], "transactions_scored": len(tx),
             "laundering_rate": round(float(tx.is_laundering.mean()), 6),
             "days_present": sorted(int(d) for d in tx.day.unique()),
             "dev": R.baseline_metrics(alerts, tx, "dev"), "test": R.baseline_metrics(alerts, tx, "test")}
    (out / "rules_baseline.json").write_text(json.dumps(rules, indent=2))

    X = F.build(tx, alerts)
    y = alerts.is_true_alert.to_numpy()
    cols = list(X.columns)
    _, own, dev, test = M.fit_and_score(X, alerts, cols)
    no_format = [c for c in cols if c not in F.FORMAT_FEATURES]
    _, own_nf, _, _ = M.fit_and_score(X, alerts, no_format)

    src_alerts = pd.read_parquet(Path(args.source) / "alert_scores.parquet")
    src_X = pd.read_parquet(Path(args.source) / "alert_features.parquet")[cols]
    src_dev = (src_alerts.split == "dev").to_numpy()
    transfer_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    transfer_model.fit(src_X[src_dev], src_alerts.is_true_alert[src_dev])
    transferred = transfer_model.predict_proba(X[cols])[:, 1]

    own_report = M.report(y[test], own[test])
    k80 = own_report["keep_80pct_true"]["alerts_reviewed"]
    m = {"variant": args.name,
         "trained_on_this_dataset": {"dev_alerts": int(dev.sum()), "dev_true_alerts": int(y[dev].sum()),
                                     "test": own_report, "test_without_payment_format": M.report(y[test], own_nf[test]),
                                     "schemes_in_first_read": scheme_capture(alerts, own, test, k80)},
         "transferred_from_HI-Small": {"trained_on": {"alerts": int(src_dev.sum()), "true_alerts": int(src_alerts.is_true_alert[src_dev].sum())},
                                       "test": M.report(y[test], transferred[test])},
         "note": "same rules, features, model settings and time split as HI-Small; nothing re-tuned"}
    (out / "model_baseline.json").write_text(json.dumps(m, indent=2))
    alerts.assign(model_score=own, transferred_score=transferred).to_parquet(out / "alerts.parquet", index=False)

    r, t = rules["test"], m["trained_on_this_dataset"]["test"]
    print(json.dumps({"days": rules["days_present"], "protocol": rules["protocol"], "rules_test": {k: v for k, v in r.items() if k != "by_rule"},
                      "by_rule": r["by_rule"]}, indent=1))
    for name, rep in (("own model", t), ("own, no format", m["trained_on_this_dataset"]["test_without_payment_format"]),
                      ("transferred from HI", m["transferred_from_HI-Small"]["test"])):
        print(name, "AUC", rep["auc"], "AP", rep["average_precision"], {k: (v["alerts_reviewed"], v["workload_reduction"], v["precision"]) for k, v in rep.items() if k.startswith("keep")})
    print(m["trained_on_this_dataset"]["schemes_in_first_read"])


if __name__ == "__main__":
    main()
