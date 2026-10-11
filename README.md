# AML Sentinel

An AI agent that does the first-pass investigation of anti-money-laundering (AML)
alerts, so human investigators spend their time on the cases that matter.

**Status:** rules, a scoring model, and the agent are all measured on held-out
days of the public IBM AML benchmark.

**Headline findings**

- Rules plus a small scoring model cut the alert workload by 73% while keeping
  80% of true alerts.
- The agent does not improve on that model as a scorer. On 300 held-out alerts
  its ranking was worse (AUC 0.68 against 0.81), and averaging the two was worse
  than the model alone. A stronger model did not change this.
- So the model ranks, and the agent is kept for what a model cannot do: writing
  up the case for a human. That part is not yet measured.

## The problem

Banks monitor transactions with threshold rules. The rules are cheap and easy to
audit, but they flag large numbers of innocent customers, and every alert is
reviewed by a person.

## Approach

```
transactions -> rules -> alerts -> scoring model -> agent -> human investigator
                (wide net)         (ranks, free)    (writes   (decides)
                                                     the case
                                                     file)
```

Rules stay as the auditable net. A small model ranks alerts for free. The agent
investigates the alerts worth a human's time and writes up the evidence. A human
makes every final decision.

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

### Does the agent add anything to the model?

The model sets aside the alerts it is confident are ordinary. The remaining
"review band" on the test days is 9,249 alerts holding 1,275 of the 1,360 true
alerts (94%). The agent investigated a stratified sample of 300 of them
(`src/ibm_agent.py`, `src/ibm_agent_eval.py`). Its instructions, tools, and
escalation cut-off were fixed on development days before the test ran.

| Ranking of the same 300 alerts | AUC | Share of band to read to keep 80% of true alerts |
|---|---|---|
| Scoring model alone | 0.810 | 45% |
| Agent alone (Claude Haiku 4.5) | 0.682 | 67% |
| Average of the two | 0.785 | 51% |

- Agent minus model: AUC −0.128 (95% CI −0.190 to −0.066).
- Average minus model: AUC −0.025 (95% CI −0.052 to −0.002).
- The agent escalated 48 of 120 true alerts (40%) and dismissed 163 of 180 false
  alarms (91%), leaving 14% of the band for humans. The model alone keeps the
  same 40% of true alerts in 5.5% of the band.
- The agent's confident flags (risk 50+) were 52% precise. The model's top slice
  of the same size was 79% precise.
- Stronger model: Claude Sonnet 5.5 on 100 of the same alerts scored AUC 0.657,
  against 0.613 for Haiku and 0.739 for the scoring model on those alerts.
- Network: the agent named 221 linked accounts, 86% of them correct, and found 16%
  of labeled scheme members. Returning everything the trace tool finds gives 81%
  correct and 23% found, so the agent's filtering trades recall for a little precision.
- Operations: 2 fail-safe escalations in 300, 3.7 tool calls, about 13,500 input and
  1,600 output tokens, roughly 2 cents and 18 seconds per alert.

Why the agent lost: the model learned from 18,827 labeled development alerts what
laundering looks like in this data. The agent has a page of guidance and its own
priors. In the first development run it escalated bursts of incoming payments as
"fan-in" schemes, which here occur at the base rate. Told so, it compressed most
scores into a narrow range and lost resolution. Development runs showed the same
gap before the test (AUC 0.78 against 0.92, then 0.80 against 0.84).

What this does not show: whether the agent's written case files save an
investigator time. That needs human raters and is not measured here.

## Investigator review screen

`app/review.html` is a single self-contained page (`python src/build_review_screen.py`
rebuilds it). It shows the 300 held-out alerts the agent investigated: the queue
ranked by the scoring model, and for each alert the activity summary, the
first-time payment flow, the agent's case file, and decision buttons. Benchmark
labels stay hidden until the viewer asks for them.

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

Limits: this data was written by the project author, and the agent's tools were
written to match how that data was generated. The IBM results above show the
agent's advantage here does not carry over to data it was not designed around.

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
python -m pytest tests                # 57 tests, no API key needed

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
python src/ibm_agent.py --provider anthropic --split test --n-true 120 --n-false 180
python src/ibm_agent_eval.py --run <run_id>
```

Each stage also has a workflow under the Actions tab, which reads keys from
repository secrets.

## Layout

- `src/ibm_prepare.py`, `ibm_data.py`, `ibm_rules.py`, `ibm_features.py`, `ibm_model.py`: IBM benchmark pipeline
- `src/ibm_tools.py`, `ibm_agent.py`, `ibm_agent_eval.py`: the agent on IBM data and its comparison with the model
- `src/generate_data.py`, `rules_baseline.py`, `evaluate.py`: synthetic sample pipeline
- `src/ledger.py`, `tools.py`: read-only data access and the nine investigation tools
- `src/agent.py`, `models.py`: the agent loop and provider adapters
- `src/run_agent.py`, `evaluate_agent.py`: run and score the agent
- `src/build_review_screen.py`, `app/`: the investigator review screen
- `tests/`: automated tests
- `docs/`: plain-language walkthrough of each step
- `results/`: every number quoted above, as produced by the code

## Roadmap

1. Sample data and rules baseline ✅
2. Investigation tools ✅
3. Triage agent, measured on sample data ✅
4. IBM benchmark: rules and model baseline ✅
5. IBM benchmark: agent compared with the model ✅
6. Investigator review screen ✅
7. Packaging: diagram, demo, write-up
