# Step 4 explainer: the agent

## What we built

The agent itself: a program that takes one alert, investigates it with the tools
from step 3, and hands a human investigator a recommendation with its reasoning.

It now runs on a real AI model. The first results are at the end of this page.

## How the agent works (the "agent loop")

1. The agent is given the alert: which account, which date, which rule fired.
2. The AI model decides which tool to use first.
3. Our code runs the tool and gives the result back to the model.
4. The model reads the evidence and either asks for another tool or decides.
5. It ends by submitting a **verdict**: escalate or dismiss, a risk score from
   0 to 100, the type of scheme, the evidence, the innocent explanation it
   considered, the other accounts involved, and a short case summary.

Steps 2 to 4 repeat. That repetition, where the model chooses its own next step
based on what it just learned, is what makes it an *agent* and not a chatbot.

## The five safety properties

These are what a bank would ask about first. Each has an automated test.

| Property | What it means | Why it matters |
|---|---|---|
| **Fail-safe** | If anything goes wrong, the alert goes to a human | The agent can never clear a case by crashing |
| **Bounded** | At most 8 rounds and 16 tool uses per alert | Cost and time can't run away |
| **Grounded** | Any account the agent names must have appeared in tool results | Blocks invented ("hallucinated") accounts |
| **Auditable** | Every tool use, result, and reason is saved | A regulator can replay any decision |
| **Blind** | The agent never sees the right answers | Results are honest |

The human keeps the final decision. The agent recommends. This design is called
**human-in-the-loop**.

## Why the model is a swappable part

All the code that talks to an AI provider lives in one file (`models.py`). The
agent doesn't know or care whether it's Claude or OpenAI behind it. Benefits:

- We can run the same alerts through both and compare quality and cost.
- A company isn't locked in to one vendor.
- We can plug in a fake model for testing, which costs nothing.

The term for this is a **provider abstraction** or **adapter pattern**.

## How we tested without spending money

33 automated tests now pass. For the agent tests, a **scripted stand-in** plays
the part of the AI model and says exactly what we tell it to. That lets us check
the dangerous situations on purpose:

- The model crashes → alert is escalated to a human.
- The model never reaches a decision → escalated.
- The model says "dismiss" in words but never files a verdict → escalated.
- The model files a nonsense verdict (risk score 900) → it gets one chance to
  fix it, then escalated.
- The model names an account that doesn't exist → that account is removed.
- The model asks for a tool that doesn't exist → it gets an error message and
  carries on.

Testing the parts around a model with a fake model is standard practice, because
real models are slow, cost money, and don't give the same answer twice.

## How we will measure it

- **Sample.** The agent reviews a sample of alerts: half real, half false alarms,
  so both groups are measured well. The results are then scaled back to the true
  mix (3.6% real). This is called **stratified sampling**.
- **Two headline numbers.** Share of false alarms dismissed, and share of real
  alerts kept. One without the other is meaningless.
- **Error bars.** Every rate comes with a 95% confidence interval. "85%, plus or
  minus 15" on 20 alerts is a very different claim from "85%, plus or minus 3"
  on 500. Small test runs are for checking it works, not for quoting.
- **Network.** Of the other accounts in each scheme, how many did the agent find?
- **Cost.** Tokens and dollars per alert.

## How it will run

On GitHub, using a feature called **GitHub Actions**: a button in your repository
that runs the code on GitHub's computers. Your API key is stored as an encrypted
**secret** in the repository settings. It is never shown in logs, never in the
code, and never in this chat.

## The first live runs, and what we learned

| Run | Real alerts kept | False alarms cleared |
|---|---|---|
| Run 1 (20 alerts) | 9 of 10 | 2 of 10 |
| Run 2 (40 alerts) | 16 of 20 | 18 of 20 |

Run 1 worked mechanically but flagged almost everything. Reading its reasoning
showed three mistakes:

1. **It thought accounts were brand new.** The data starts on 1 January, so
   early alerts had no history. The agent read "no history" as "just opened".
2. **It saw networks everywhere.** A business pays many different companies.
   The tool listed them all and the agent called it a laundering network.
3. **It overreacted to busy weeks.** Twice the usual volume sounds alarming but
   is normal for a business.

The fix was the same idea each time: **compare the account with its own past.**
Every tool now reports the account's own norm beside each signal, the network
tool follows only specific suspicious behaviour, and the first 30 days are left
out of scoring because nothing can be compared yet (a **warm-up period**).

This loop, run it, read the failures, find the cause, fix the cause, run again,
is called **error analysis**. It is the main skill in building AI systems and a
strong interview story. Tell it with the run 1 number included.

Be honest about the limits when you tell it:
- 40 alerts is a small sample. "18 of 20" could really be anywhere from 70% to 97%.
- We fixed things after looking at these alerts, so run 2 flatters us. The honest
  test is a fresh set of alerts the agent has never been tuned on (a **held-out
  test set**).
- Run 2 kept fewer real alerts than run 1 (80% against 90%). Clearing false
  alarms and keeping real cases pull against each other, and that trade-off is
  the next thing to work on.

## Questions an interviewer will ask

1. *What makes this an agent and not a pipeline?* The model chooses which tools
   to use and when to stop, based on what it finds. A pipeline runs fixed steps.
2. *What happens when the model is wrong or down?* Fail-safe to human review. Be
   ready to list the five properties.
3. *How do you stop hallucination?* Tools supply all facts; the verdict is
   checked against tool output; inconsistent verdicts are rejected.
4. *Why a structured verdict instead of free text?* Code can validate it, store
   it, and measure it. Free text can't be scored at scale.
5. *How do you control cost?* A cap on tool uses per alert, a total token budget
   per run, reuse of the unchanging instructions (prompt caching), and a cheaper
   model first with a stronger one only if results justify it.
6. *The model isn't deterministic. How do you trust a result?* Sample size and
   confidence intervals, and repeating the run to measure how much it varies.
7. *Why sample half real alerts when only 3.6% are real?* A random sample of 200
   would contain about 7 real cases, far too few to measure. Stratify, then
   re-weight.
