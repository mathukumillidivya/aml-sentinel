# Earlier work on synthetic sample data

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
