# Step 3 explainer: the investigation tools

## What we built

Nine small programs called **tools**. Each one answers a single question a human
investigator would ask about a flagged account. In step 4 the AI agent will pick
which tools to use for each alert, read the answers, and reach a judgment.

Think of the agent as a detective and the tools as the things a detective can do:
pull bank statements, check who the person deals with, look for known tricks.

## Why the agent needs tools

A language model can't see the bank's data and is unreliable at arithmetic over
thousands of rows. So we split the work:

- **Tools do the counting.** They are ordinary code: exact, fast, testable.
- **The agent does the judging.** It decides what to check and what the evidence means.

This split is the core idea of "agentic AI". Be ready to explain it.

## The nine tools

| Tool | The question it answers |
|---|---|
| `get_account_profile` | Who is this account and what is normal for it? |
| `get_transactions` | What exactly happened in the days around the alert? |
| `get_counterparties` | Who did it deal with, and are they new? |
| `check_structuring` | Are cash deposits bunched just under $10,000? |
| `check_fan_pattern` | Is money spraying out to, or funnelling in from, many accounts? |
| `check_pass_through` | Does money arrive and leave again almost untouched? |
| `find_cycles` | Does money leave and come back through a ring of accounts? |
| `compare_to_history` | Is this out of character for the account? |
| `trace_network` | Which other accounts are linked through unusual payments? |

## Three design decisions worth remembering

1. **The tools can't see the answers.** The labels saying which transactions are
   laundering are removed before the tools load the data. A test proves it. Without
   this, any result would be cheating. The technical term is avoiding **label leakage**.
2. **Tools report evidence and never decide.** A tool says "8 payments to 8 accounts,
   all seen before, same thing happened twice last month". The agent concludes
   "that's payroll". Keeping facts and judgment apart makes every decision traceable,
   which is what a regulator asks for.
3. **Context separates guilty from innocent.** Payroll and a laundering fan-out look
   the same on the day. The difference is history: payroll goes to the same people
   every two weeks. So most tools compare the alert window with the account's past.

## How we know the tools work

**14 automated tests, all passing.** Each builds a tiny made-up case where the
right answer is obvious, such as four accounts passing $50,000 in a ring, and
checks the tool finds it. The tests also check the opposite: salary followed by
rent is not flagged as pass-through, and a ring where the timing is impossible is
not reported as a cycle.

**A signal check on all 6,739 alerts.** I added up a few obvious red flags into a
crude score, with no AI involved, to see whether the evidence separates real
cases from false alarms.

| Check | Result |
|---|---|
| False alarms showing no red flag at all | 4,496 of 6,497 (69%) |
| Real alerts showing at least one red flag | 199 of 242 (82%) |
| Separation score (AUC; 0.5 is a coin flip, 1.0 is perfect) | 0.81 |
| Time to run all tools on one alert | 0.06 seconds |

This is a sanity check on data I wrote, using a score I wrote. It is not a
project result and does not go on the resume. It tells us the evidence is good
enough to build on.

## What the check exposed (this is where the agent earns its keep)

- **2,001 false alarms still show a red flag.** A crude score can't tell that a
  business moving money in and out daily is normal. That needs judgment.
- **43 real alerts show no red flag.** Most are people who received a single
  payment in a fan-out scheme. Looked at alone, one payment is unremarkable. You
  only see it by tracing the network back to the source.
- **New accounts are hard.** With no history, a first payroll looks like a sudden
  fan-out. This is called the **cold start problem**.

## Questions an interviewer will ask

1. *Why give the agent tools instead of putting the data in the prompt?* Volume,
   cost, and accuracy. Models miscount; code doesn't. Tools also leave an audit trail.
2. *How do you stop the agent misusing a tool?* Tools are read-only, and bad input
   returns an error message the agent can read instead of crashing.
3. *How did you prevent leakage?* Labels are dropped when data loads, and a test
   enforces it.
4. *Why is your cycle search capped at 8 hops?* The search grows very fast with
   depth. Eight covers the schemes in the data, and the tool reports when it had
   to stop early.
5. *What can't your tools see?* Customer identity, geography, and sanctions lists.
   A real bank has all three; this dataset doesn't.
