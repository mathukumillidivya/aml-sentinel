# Reproduce these results

## Check the published agent numbers (no API key, no download, under a minute)

```bash
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` re-scores the agent's saved runs on the 300 held-out alerts with the same scoring code and
compares each number with the committed results. It prints `MATCH` or `DIFFERENT` per file.

## What each part is

| | Where |
|---|---|
| Agent configuration frozen before the held-out test | commit `56a2bb5` |
| Dataset | IBM Transactions for Anti Money Laundering, HI-Small and LI-Small (Altman et al., NeurIPS 2023), from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), CDLA-Sharing-1.0. Not stored here; see [DATA.md](DATA.md). |
| Time split | days 0-3 history, 4-5 development, 6 gap, 7-9 held-out test (`src/ibm_data.py`) |
| Raw agent outputs | `results/ibm/agent_runs/ibm-test-haiku/` and `ibm-test-sonnet/`: every tool call, verdict and case file (`traces.jsonl`, `verdicts.csv`) |
| Scoring code | `src/ibm_agent_eval.py` (ranking, decisions, bootstrap intervals), `src/ibm_faithfulness.py` (every account and amount checked against the evidence) |
| Published results | `results/ibm/rules_baseline.json`, `model_baseline.json`, `agent_runs/*/metrics.json`, `faithfulness.json`, `results/ibm/LI-Small/` |

## Re-run everything from the raw data

```bash
# download HI-Small (and LI-Small) from Kaggle into data/ibm/
python src/ibm_prepare.py          # raw CSV to data/ibm_prepared/
python src/ibm_rules.py            # results/ibm/rules_baseline.json
python src/ibm_model.py            # results/ibm/model_baseline.json (the 73% alert-volume cut)
python src/ibm_replicate.py --name LI-Small

git checkout 56a2bb5
export ANTHROPIC_API_KEY=...       # never commit this; the test run cost about $6.50
python src/ibm_agent.py --provider anthropic --split test --n-true 120 --n-false 180
python src/ibm_agent_eval.py --run <run_id>
python src/ibm_faithfulness.py --run <run_id>
```

The rules and model are deterministic. The agent is a model, so a fresh run lands close to, not exactly on,
the published figures.

## Checks added after the test

These came after the frozen evaluation and do not change its numbers.

```bash
python src/ibm_currency_check.py --run ibm-test-haiku       # free: dollar figures written as another currency
python src/ibm_narrative_judge.py --run ibm-test-haiku --budget 1.5   # second-model review; needs ANTHROPIC_API_KEY
```

Outputs are in `results/ibm/agent_runs/ibm-test-haiku/`: `currency_check.json`, `narrative_judge.json(l)`, and the
hand review `narrative_spot_check.json`. In GitHub: Actions > judge-narratives.
