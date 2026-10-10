# Step 6 explainer: testing the agent on the IBM benchmark

## The question

On my sample data the AI agent looked very good. The fair question is whether it
adds anything on real benchmark data, where a free scoring model already does
well. So this step is one experiment: **agent versus model, on the same alerts.**

## How the test was set up

1. The rules raise alerts (20,784 on the test days).
2. The scoring model sets aside the ones it is confident are ordinary. What is
   left is the **review band**: 9,249 alerts containing 94% of the real ones.
3. The agent investigated 300 of those. The model had already scored them.
4. We compared three ways of ranking those 300 alerts: the model's score, the
   agent's risk score, and the average of both.

Everything about the agent (instructions, tools, cut-off) was fixed using the
development days before the test days were touched.

## The result

| Who ranks the alerts | Score (AUC) | Alerts to read to catch 80% of real ones |
|---|---|---|
| Scoring model | 0.81 | 45% |
| AI agent | 0.68 | 67% |
| Both averaged | 0.79 | 51% |

The agent was worse than the free model, and adding it to the model made the
model slightly worse. A stronger, more expensive AI model (Claude Sonnet) gave
the same answer.

## Why the agent lost

- **The model learned from examples.** It studied 18,827 labeled alerts and
  picked up what laundering looks like in this particular data.
- **The agent works from intuition and a page of guidance.** In the first trial
  it saw eleven payments arriving within minutes and called it a "textbook"
  laundering pattern. In this data that pattern is ordinary.
- **Correcting it made it timid.** Once told its intuition was unreliable here, it
  gave most alerts nearly the same score, so it could no longer tell them apart.

The general lesson: for scoring many similar cases where you have labeled
examples, a small trained model usually beats a language model. Language models
earn their place on reading, explaining, and handling cases that don't fit a
pattern.

## Why this is a good result for the project

It may feel like bad news. For interviews it is the opposite.

- Most AI projects only report wins. A candidate who tested their own idea
  fairly, found the limit, and changed the design shows the judgment senior
  roles are hired for.
- The most common senior interview question is "how do you know the LLM is
  needed here?" You have a measured answer.
- The design that results is how these systems are built in practice: cheap
  model for ranking, AI agent for the write-up, human for the decision.

## What the agent is for now

It no longer decides which alerts matter. For alerts the model ranks high, it
investigates and writes the case file: what happened, which accounts are
involved, and what the investigator should check first. Whether those write-ups
save an investigator time has not been measured, and the README says so.

## A mistake I caught in my own scoring

When many alerts had the same score, my scoring code read them in the order they
were stored, and real alerts happened to be stored first. That made the agent
look better than it was on one measure. I fixed it and added a test. Mention
this if asked about a bug you found: checking your own measurement is part of
the job.

## Words you need to own

- **Baseline:** the simpler method you must beat to justify a more complex one.
- **Ablation / controlled comparison:** changing one thing and measuring the difference.
- **Negative result:** a finding that the idea did not work. Still a result.
- **Calibration:** whether a score of 70 really means 70% likely.
- **Prior:** what a model assumes before it sees the data.
- **Confidence interval:** the range the true figure probably lies in. Ours for
  the agent-versus-model gap is −0.19 to −0.07, which excludes zero, so the gap
  is real and not luck.
- **Paired bootstrap:** the method used to get that interval.

## Questions an interviewer will ask

1. *Why did the LLM lose to gradient boosting?* Labeled data. The model had
   18,827 examples; the agent had instructions. See "Why the agent lost".
2. *Did you try a stronger model?* Yes, Sonnet on 100 of the same alerts. Slightly
   better than Haiku, still below the scoring model.
3. *Could better prompting close the gap?* One round of prompt changes on
   development data moved the agent from 0.78 to 0.80. More rounds might help a
   little, but each costs money and risks tuning to the test. A fairer next step
   is giving the agent labeled examples to look up (retrieval), or fine-tuning.
4. *Then why keep the agent?* For the case write-up, which a scoring model cannot
   produce. And I'd say plainly that its value there is not yet measured.
5. *How would you measure the write-ups?* Have investigators handle alerts with
   and without the case file, and compare time taken and decisions made.
6. *Your sample-data result was much better. Why?* I wrote that data, and built
   the agent's tools around how I generated it. That is a form of leakage, and
   the benchmark exposed it.
7. *What did it cost?* About 2 cents and 18 seconds per alert for the agent,
   against effectively nothing for the model.
