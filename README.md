# AML Sentinel

An AI agent that does the first-pass investigation of anti-money-laundering (AML)
alerts, so human investigators spend their time on the cases that matter.

**Status: step 4 of 7 complete** (sample data, rule-based baseline, investigation
tools, triage agent with a held-out result on sample data). Not yet measured on
the public IBM dataset.

## The problem

Banks monitor transactions with threshold rules. The rules are cheap and easy to
audit, but they flag large numbers of innocent customers. Every alert is reviewed
by a person.

## Baseline on the sample data

Five standard monitoring rules, run over 90 days of synthetic transactions:

| Measure | Value |
|---|---|
| Transactions | 160,457 |
| Alerts raised | 6,739 (about 75 per day) |
| Alerts tied to real laundering | 242 |
| False alarms | 6,497 (96.4%) |
| False alarms per true alert | 26.8 |
| Laundering schemes touched by at least one alert | 56 of 60 |
| Laundering accounts alerted | 160 of 463 (34.6%) |

These figures come from a **synthetic sample dataset written for this project**.
They show the pipeline works; they are not a claim about any real bank. Headline
results will be measured on the public IBM AML dataset.

Known limitation: the "low and slow" schemes are built to stay under every rule
threshold. The 7 of 10 shown as caught are coincidental (an account in the scheme
was alerted for unrelated ordinary activity), so the rules' real detection of that
typology is effectively zero.

## Investigation tools

Nine read-only tools the agent will call (`src/tools.py`): account profile,
transactions, counterparties, structuring, fan-in/fan-out, pass-through, cycle
search, comparison with history, and network tracing. Ground-truth labels are
dropped when data loads, so tools cannot see the answers.

Signal check on the sample alerts, using a hand-written score and no model:
69% of false alarms show no red flag, 82% of true alerts show at least one,
AUC 0.81. This is a sanity check on self-authored data, not a result.

## The agent

`src/agent.py` runs a tool-use loop: the model reads the alert, picks tools,
reads the evidence, and submits a structured verdict (disposition, risk score,
typology, evidence, innocent explanation considered, linked accounts, narrative).
A human investigator makes the final call.

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

### Results so far (Claude Haiku 4.5, synthetic sample data)

Held-out test: 100 alerts the agent was never tuned on (run3), re-weighted to the
real mix of 4,490 alerts after a 30-day warm-up.

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

Misses by scheme type: cycle 0/12, structuring 0/3, scatter-gather 0/6,
fan-out 2/15, fan-in 2/8, low-and-slow 5/6. Low-and-slow stays under every
threshold by design, and its alerts fire on unrelated activity, so neither the
rules nor the agent has evidence to work with. That typology needs a different
detector.

Development history:

| Run | Alerts | True alerts escalated | False alarms dismissed |
|---|---|---|---|
| run1 (dev) | 20 | 9/10 | 2/10 |
| run2 (dev) | 40 | 16/20 | 18/20 |
| run3 (held-out) | 100 | 41/50 | 38/50 |

Run1 over-escalated: the agent treated short data history as a new account,
ordinary counterparty turnover as a network, and normal volume swings as
anomalies. The tools now report each account's own base rate beside every
signal, and `trace_network` follows only specific suspicious links. Run2's 90%
dismissal rate fell to 76% on held-out alerts, which is the expected gap between
a development set and a test set.

Limits: the data is synthetic and written by the project author, the sample is
100 alerts, and the agent trades about 18% of true alerts for the workload cut.
Measurement on the public IBM AML dataset is the next step.

## Run it

```bash
pip install -r requirements.txt
python src/generate_data.py      # writes data/transactions.csv
python src/rules_baseline.py     # writes results/alerts.csv
python src/evaluate.py           # writes results/baseline_metrics.json
python -m pytest tests           # 34 tests, no API key needed
python src/tool_signal_check.py  # ~7 min; writes results/tool_signal_check.json

# agent: offline dry run, then a live run
python src/run_agent.py --provider mock --n 40
export ANTHROPIC_API_KEY=...     # never commit this
python src/run_agent.py --provider anthropic --n 20
python src/evaluate_agent.py --run <run_id>
```

Or use the **Run triage agent** workflow under the Actions tab, which reads the
key from a repository secret.

## Layout

- `src/generate_data.py`: synthetic transactions with six labeled laundering typologies
- `src/rules_baseline.py`: the rule-based monitoring system (the "before")
- `src/evaluate.py`: scores any alert file against the labels
- `src/ledger.py`: read-only data access with labels removed
- `src/tools.py`: the nine investigation tools
- `src/agent.py`: the triage agent loop, verdict schema, and safety checks
- `src/models.py`: provider adapters (Claude, OpenAI, mock)
- `src/run_agent.py`: runs the agent over a sample of alerts
- `src/evaluate_agent.py`: scores a run against ground truth
- `src/tool_signal_check.py`: checks the tools' evidence separates true from false alerts
- `tests/`: automated tests
- `docs/`: plain-language walkthrough of each step

## Roadmap

1. Data ✅ (sample) / IBM dataset (pending)
2. Rule-based baseline ✅
3. Investigation tools ✅
4. Triage agent ✅ (held-out result on sample data above)
5. Investigator review screen
6. Before/after measurement
7. Packaging: diagram, demo, write-up
