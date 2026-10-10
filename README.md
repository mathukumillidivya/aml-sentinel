# AML Sentinel

An AI agent that does the first-pass investigation of anti-money-laundering (AML)
alerts, so human investigators spend their time on the cases that matter.

**Status:** rules and a machine-learning baseline are measured on the public IBM
AML benchmark. The triage agent is built and measured on synthetic sample data;
its run on the IBM benchmark is the next step.

## The problem

Banks monitor transactions with threshold rules. The rules are cheap and easy to
audit, but they flag large numbers of innocent customers, and every alert is
reviewed by a person.

## Approach

```
transactions -> rules -> alerts -> scoring model -> triage agent -> human investigator
                (wide net)         (cheap ranking)  (investigates,    (decides)
                                                     explains, maps
                                                     the network)
```

Rules stay as the auditable net. A small model ranks alerts for free. The agent
investigates with tools, writes up the evidence, and names the other accounts
involved. A human makes every final decision.

## Results on the IBM AML benchmark (HI-Small)

Dataset: [IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
(Altman et al., NeurIPS 2023). 5.08M transactions, 1 in 1,000 laundering, 370 labeled schemes.

Protocol (`src/ibm_data.py`): days 0-3 are history, days 4-5 development, day 6 a
gap, days 7-9 held-out test. Everything below is on the held-out days.

**Rules only** (`src/ibm_rules.py`). Four rules built on first-time counterparty
relationships; none uses the payment format.

| Measure | Value |
|---|---|
| Alerts | 20,784 (about 6,900 a day) |
| True alerts | 1,360 (6.5%) |
| False alarms | 19,424 (93.5%) |
| Laundering transactions on an alerted account | 1,018 of 1,318 (77%) |
| Labeled schemes touched | 160 of 207 |

**Rules + scoring model** (`src/ibm_model.py`). Gradient-boosted trees on 27
evidence features, trained on development days. Alerts are reviewed from the
highest score down.

| To keep this share of true alerts | Alerts to review | Workload cut | Precision |
|---|---|---|---|
| 50% | 955 | 95% | 71% |
| 80% | 5,554 | 73% | 20% |
| 90% | 8,232 | 60% | 15% |
| 95% | 9,495 | 54% | 14% |

AUC 0.896, average precision 0.625 (rules alone: 0.065).

Robustness checks:

| Check | AUC | Workload cut at 80% retention |
|---|---|---|
| All features | 0.896 | 73% |
| Payment-format features removed | 0.893 | 73% |
| Only accounts never alerted during development | 0.858 | 64% |
| Ranking by number of rules fired (no model) | 0.464 | 19% |

Dataset quirks handled, each of which would otherwise inflate results:

- **Date leak.** After day 10 nearly every transaction is laundering. Those days are dropped.
- **Format leak.** 85% of laundering moves by ACH, which is 12% of volume. Rules ignore
  the format, and the model is reported with and without it.
- **Hub accounts.** 15 accounts with over 2,000 transactions each touch 10% of all
  transactions. They are excluded from alerting and scoring, which also removes 633
  laundering transactions from scope.
- **Self-transfers.** 591,212 payments from an account to itself are dropped.
- **Currencies.** Amounts are converted to USD with rates implied by the dataset's own
  cross-currency payments.

Limits: these are alert-level results on accounts, so they are not comparable with
the transaction-level F1 scores in the graph-neural-network literature. 23% of
laundering transactions never reach an alerted account, and the model cannot recover
what the rules miss. HI-Small is one of six IBM variants.

## Results on synthetic sample data (triage agent)

The agent was developed on a synthetic dataset written for this project
(`src/generate_data.py`: 160,457 transactions, 60 planted schemes). Held-out test
of 100 alerts, Claude Haiku 4.5, re-weighted to the real alert mix:

| Measure | Rules only | With agent |
|---|---|---|
| Alerts needing human review | 4,490 | about 1,190 (74% fewer) |
| Precision | 3.7% | 11.5% |
| False alarms dismissed | n/a | 38/50 (76%, 95% CI 63–86%) |
| True alerts escalated | n/a | 41/50 (82%, 95% CI 69–90%) |
| Other scheme members identified | n/a | 309/369 (84%, 95% CI 80–87%) |
| Accounts named that were correct | n/a | 342/384 (89%) |
| Fail-safe escalations | n/a | 2/100 (malformed verdicts, routed to a human) |
| Tokens per alert (in / out) | n/a | 18,400 / 1,750 |

Development history:

| Run | Alerts | True alerts escalated | False alarms dismissed |
|---|---|---|---|
| run1 (dev) | 20 | 9/10 | 2/10 |
| run2 (dev) | 40 | 16/20 | 18/20 |
| run3 (held-out) | 100 | 41/50 | 38/50 |

Run1 over-escalated: the agent treated short data history as a new account,
ordinary counterparty turnover as a network, and normal volume swings as
anomalies. The tools now report each account's own base rate beside every
signal. Run2's 90% dismissal rate fell to 76% on held-out alerts, the expected
gap between a development set and a test set.

Limits: this data was written by the project author, so these figures show the
agent works end to end and are not a benchmark claim.

## The agent

`src/agent.py` runs a tool-use loop: the model reads the alert, picks tools,
reads the evidence, and submits a structured verdict (disposition, risk score,
typology, evidence, innocent explanation considered, linked accounts, narrative).

| Property | Guarantee |
|---|---|
| Fail-safe | Any error, timeout, or missing verdict escalates to a human |
| Bounded | Max 8 model turns and 16 tool calls per alert; token budget per run |
| Grounded | Accounts named in a verdict must appear in tool output |
| Auditable | Full trace of tool calls, results, and reasoning per alert |
| Blind | Ground-truth labels never reach the tools or the model |

Models are swappable (`src/models.py`): Claude, OpenAI, or an offline mock used
by the tests. Evaluation uses a stratified sample re-weighted to the true alert
mix, with 95% Wilson confidence intervals on every rate.

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests                # 46 tests, no API key needed

# synthetic sample pipeline
python src/generate_data.py
python src/rules_baseline.py && python src/evaluate.py
python src/run_agent.py --provider mock --n 40          # offline dry run
export ANTHROPIC_API_KEY=...                            # never commit this
python src/run_agent.py --provider anthropic --n 20
python src/evaluate_agent.py --run <run_id>

# IBM benchmark (needs the Kaggle download in data/ibm/)
python src/ibm_prepare.py             # raw CSV -> data/ibm_prepared/
python src/ibm_rules.py               # results/ibm/rules_baseline.json
python src/ibm_model.py               # results/ibm/model_baseline.json
```

Each stage also has a workflow under the Actions tab, which reads keys from
repository secrets.

## Layout

- `src/ibm_prepare.py`, `ibm_data.py`, `ibm_rules.py`, `ibm_features.py`, `ibm_model.py`: IBM benchmark pipeline
- `src/generate_data.py`, `rules_baseline.py`, `evaluate.py`: synthetic sample pipeline
- `src/ledger.py`, `tools.py`: read-only data access and the nine investigation tools
- `src/agent.py`, `models.py`: the agent loop and provider adapters
- `src/run_agent.py`, `evaluate_agent.py`: run and score the agent
- `tests/`: automated tests
- `docs/`: plain-language walkthrough of each step
- `results/`: every number quoted above, as produced by the code

## Roadmap

1. Sample data and rules baseline ✅
2. Investigation tools ✅
3. Triage agent, measured on sample data ✅
4. IBM benchmark: rules and model baseline ✅
5. IBM benchmark: agent on top of the model (next)
6. Investigator review screen
7. Packaging: diagram, demo, write-up
