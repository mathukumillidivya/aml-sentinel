# Step 5 explainer: moving to the IBM benchmark

## Why this step matters

Everything before this ran on data I wrote. Anyone can get good results on data
they made themselves. The IBM dataset is public, built by IBM researchers, and
used in published papers, so a result on it is one other people can check.
That is what a **benchmark** is.

## What the data looks like

- 5 million transactions over 18 days, in 15 currencies.
- 1 transaction in 1,000 is laundering.
- 370 labeled schemes of 8 types (fan-out, fan-in, cycle, and so on).

## The first surprise: my earlier approach didn't fit

On my sample data, laundered money kept roughly the same amount as it moved
(send $9,000, forward $8,800). The tools looked for that. In the IBM data the
amounts are unrelated from one step to the next, and a scheme can run across
four days and several currencies. Those tools would have found nothing.

So before writing any code I studied the real data. That is called
**exploratory data analysis (EDA)**. It found one strong clue:

> Normal payments go to someone the account has paid before (96% of the time).
> Laundering payments go to someone new (81% of the time).

All four new rules are built on that idea, a **first-time counterparty**.

## The four rules

| Rule | Plain meaning |
|---|---|
| N1 | A first-time payment of $10,000 or more |
| N2 | Three or more first-time counterparties in three days |
| N3 | Money received from a first-time sender, then paid on to a first-time receiver |
| N4 | First-time payments in two or more currencies in one day |

Result on the test days: 20,784 alerts, of which 93.5% are false alarms. The
alerted accounts cover 77% of the laundering transactions. This is the "before".

## The four traps in the data

Each would have made the results look better than they are. Spotting and
closing them is the strongest thing to talk about from this step.

1. **The date trap.** In the last 8 days almost everything is laundering. A system
   could score well by looking only at the date. Those days are removed.
2. **The payment-type trap.** 85% of laundering uses one payment type (ACH). A real
   bank has no such giveaway, so the rules ignore payment type, and the model is
   scored with and without it. The scores barely change, which shows the result
   doesn't depend on it.
3. **Giant accounts.** 15 accounts handle a tenth of all transactions. They are
   connected to everything, so they'd count as "involved in laundering" by
   accident. They are set aside, as a bank would handle them separately.
4. **Peeking at the future.** The system is tuned on days 4-5 and scored on days
   7-9, which it never saw. This is a **time-based train/test split**, with day 6
   left out as a gap.

The general name for traps 1, 2, and 4 is **data leakage**: information that
makes a test easier than real life.

## The scoring model

A small, standard machine-learning model (**gradient-boosted trees**) reads 27
facts about each alert and gives it a score. It costs nothing to run.

If investigators read alerts from the highest score down:

| To catch this share of real alerts | They read | Instead of |
|---|---|---|
| 80% | 5,554 alerts | 20,784 |
| 90% | 8,232 alerts | 20,784 |

That is a 73% cut in workload while keeping 80% of real alerts.

## Why build a cheap model when the project is about an AI agent?

This is the question a senior interviewer will ask, and the answer is the design:

- The cheap model scores 20,000 alerts in seconds for free. An AI agent costs
  about 3 cents and 20 seconds per alert.
- So the model does the ranking, and the agent is used only where it adds
  something a model can't: investigating a case, explaining it in words, and
  naming the other accounts in the scheme.
- This layered design is called a **cascade**: cheap and fast first, expensive
  and thorough only where needed.

It also keeps us honest. If the agent can't improve on a free model, we will see
that in the numbers and say so.

## Words you need to own

- **Benchmark:** a public dataset others can check your result against.
- **EDA:** studying the data before building.
- **Data leakage:** clues in the test that wouldn't exist in real life.
- **Train/test split:** tune on one part, score on another.
- **AUC:** how well a score separates real from false, from 0.5 (coin flip) to
  1.0 (perfect). Ours is 0.90.
- **Precision:** of the alerts reviewed, the share that are real.
- **Recall (or retention):** of the real alerts, the share kept.
- **Cascade:** cheap model first, expensive one only where needed.

## Questions an interviewer will ask

1. *Why didn't your first approach transfer?* It assumed amounts are preserved
   along the chain. EDA showed they aren't, and showed first-time relationships
   are the real signal.
2. *How did you avoid leakage?* Name the four traps and the fix for each.
3. *How does this compare with published results?* Papers on this dataset report
   transaction-level F1 from graph neural networks. This project measures alert
   triage on accounts, a different task, so the numbers aren't comparable. Say
   this plainly; claiming otherwise would be caught.
4. *What does the system miss?* 23% of laundering transactions never reach an
   alerted account, and nothing downstream can recover them. Better rules or a
   graph model at the first stage would be the fix.
5. *Why keep rules at all?* Regulators expect monitoring that can be explained
   line by line. Rules give that; the model and agent sit behind them.
6. *Why does the score drop for accounts never seen in development?* Some accounts
   appear in both periods, which makes the test slightly easier. The stricter
   check gives 64% instead of 73%, and both are reported.
