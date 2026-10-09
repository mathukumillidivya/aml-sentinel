# Step 2 explainer: the "before" system

Read this until you can say it back without looking. Interviewers will ask.

## What we built

Three small programs.

1. **A data generator.** It invents 90 days of banking for 3,000 made-up accounts:
   salaries, rent, shopping, business deposits. Then it hides 60 money-laundering
   schemes inside, and labels every dirty transaction so we know the right answers.
2. **A rule-based alert system.** Five simple rules of the kind banks run today.
3. **A scorer.** It compares the alerts against the labels and counts how many
   alerts were real and how many were false alarms.

## Why we built the "before" first

You can't claim an improvement without something to improve on. The rule system is
our baseline. Later the AI agent reviews the same alerts, and we compare the two.

## The five rules

| Rule | Plain meaning | Why innocent people trip it |
|---|---|---|
| R1 Large transaction | Any single payment of $10,000 or more | House deposits, car purchases, business invoices |
| R2 Structuring | Three or more cash deposits just under $10,000 in a week | Rare for honest customers, so this rule is accurate |
| R3 Fan-out | One account pays six or more others in a day | Every employer running payroll |
| R4 Fan-in | One account is paid by six or more others in a day | Landlords on rent day |
| R5 Rapid movement | Money arrives and most of it leaves within a day | Normal for any busy business |

## The six laundering schemes hidden in the data

- **Structuring:** breaking cash into deposits just under the $10,000 reporting limit.
- **Fan-out:** one account sprays money to many.
- **Fan-in:** many accounts funnel money to one.
- **Cycle:** money passes around a ring of accounts and returns to the start.
- **Scatter-gather:** money splits across "mule" accounts, then regroups elsewhere.
- **Low and slow:** small amounts over many days, designed to stay under every rule.

## What the numbers say

- 6,739 alerts in 90 days, about 75 a day.
- Only 242 were connected to real laundering. **96.4% were false alarms.**
- That is about 27 wasted reviews for every useful one.
- The rules touched 56 of 60 schemes, but alerted only 35% of the accounts involved.
  They notice that something happened and miss most of who took part.
- They are blind to "low and slow".

That gives the agent two jobs: clear the false alarms, and map the full network
behind each real one.

## Words you need to own

- **Alert:** a flag saying "a human should look at this account".
- **False positive:** an alert on an innocent customer.
- **Precision:** of all alerts, the share that were real. Ours: 3.6%.
- **Recall:** of all real laundering, the share we caught.
- **Baseline:** the existing approach you measure yourself against.
- **Synthetic data:** made-up data that behaves like the real thing, used because
  real bank data is confidential.
- **Ground truth:** the known correct answers used to score a system.

## Questions an interviewer will ask

1. *Why synthetic data?* Real transaction data is confidential and real laundering
   labels are rarely available. Synthetic data gives known answers. The headline
   results use IBM's public AML dataset so others can reproduce them.
2. *Why not replace the rules with AI entirely?* Regulators expect monitoring that
   can be explained and audited. Rules stay as the net; the agent sits behind them
   and triages. That is also how a bank could adopt it without re-approving its
   whole monitoring program.
3. *Why precision and recall rather than accuracy?* Laundering is 0.4% of
   transactions. A system that flags nothing is 99.6% "accurate" and useless.
4. *What's the risk of cutting false positives?* Dismissing a real case. So we
   report false-positive reduction at a fixed recall, and a human keeps the final say.
5. *How did you decide an alert was "true"?* The alerted account took part in a
   labeled laundering transaction within 7 days before or 1 day after the alert.
6. *What's weak about your baseline?* The data and the rules were written by the
   same author, so the sample numbers are illustrative. A real bank would also
   whitelist known payroll accounts, which would lower R3's false alarms.
