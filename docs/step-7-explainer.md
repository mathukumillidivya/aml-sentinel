# Step 7 explainer: finding the job the agent is good at

## Where we were

The agent lost to the scoring model at ranking alerts. That left an open
question: what is the agent for? This step tested three answers. One worked.

## Test 1: are its case files faithful to the evidence? (Yes)

A scoring model gives a number. Only the agent can write the case file an
investigator reads. So the useful question is whether that writing can be
trusted. A language model can state things that sound right and aren't, which
is called **hallucination**.

The check is mechanical, with no second AI involved. For each of 298 case files:

1. Collect everything the agent was shown: its starting case file and every tool result.
2. Pull out every account number and every dollar figure the agent wrote.
3. Look each one up in what it was shown.

| What was checked | Result |
|---|---|
| Account numbers the agent named | 813 of 814 were in the evidence |
| Dollar figures found exactly as written | 97.5% of 1,121 |
| Dollar figures that were sums the agent added up | 1.2% |
| Dollar figures that could not be traced | 1.3% (15) |
| Case files with nothing untraceable | 95% (283 of 298) |

I read all 15 untraced figures. They were differences ("about $900 less"),
rounded limits ("under $5,000"), and currency conversions. None was an obviously
invented amount.

The name for this property is **faithfulness** or **groundedness**: the writing
sticks to the evidence.

**What the check cannot tell you.** It confirms a number exists in the evidence.
It does not confirm the sentence around the number is right. The agent could
attach a real amount to the wrong account and this check would pass it. Say this
if asked; an interviewer will think of it.

## Test 2: can it name the type of scheme? (No)

Each real scheme in the benchmark has a labeled type (fan-out, cycle, and so
on). The agent named the right type for 7 of 47. The reason is its view: it
looks at one account and a few steps around it, and the shape of a whole scheme
isn't visible from there.

## Test 3: does showing it similar past cases help it rank? (No)

The agent lost at ranking because it had no examples to learn from. So I gave it
some: for each alert, the 25 most similar past alerts and how each turned out.
This is called **retrieval**, and it is the idea behind "RAG"
(retrieval-augmented generation).

I checked it for free first. The similar cases alone, with no agent, scored 0.74,
still below the model's 0.81. Then a small trial of 50 development alerts with
the agent: 0.77 with similar cases against 0.79 without, on the same alerts. No
gain, so I stopped before spending on a full test.

Two safeguards mattered here: the library holds only past (development) cases, so
a test alert can never look up its own answer, and an alert never retrieves
another alert on the same account. Both have tests.

## What this adds up to

| Job | Who does it | Evidence |
|---|---|---|
| Ranking the queue | Scoring model | AUC 0.90; 73% workload cut at 80% of real cases kept |
| Writing the case file | Agent | 95% of case files fully traceable |
| Naming the scheme type | Nobody yet | Agent 7 of 47 |
| Deciding | Human | By design |

## Why stopping test 3 early was the right call

Running the full test would have cost about $4 to confirm something two cheaper
checks already showed. Knowing when a cheap check is enough is part of the job.
"I ran a free offline check, then a $1 trial, saw no gain, and stopped" is a good
interview answer.

## Words you need to own

- **Hallucination:** an AI stating something that isn't supported.
- **Faithfulness / groundedness:** output that sticks to the evidence.
- **Retrieval (RAG):** giving a model relevant examples or documents to consult.
- **Nearest neighbours:** finding the most similar past examples.
- **Offline check:** testing an idea on saved data before paying to run it.

## Questions an interviewer will ask

1. *How do you know the agent isn't hallucinating?* Describe the mechanical check
   and its numbers, then its limit (it checks figures exist, not that sentences
   are right).
2. *How would you make that check stronger?* Check each figure against the
   specific account it's attached to, and have a second model or a human grade a
   sample of full sentences.
3. *Why didn't retrieval help?* The similar cases carried less signal than the
   model already had, and the agent was already shown the model's score.
4. *What would you try next for the agent?* Fine-tuning on labeled cases, or
   moving it to tasks with no labeled data, where a trained model can't compete.
5. *Why not use an AI to grade the writing?* A mechanical check is cheaper,
   repeatable, and can't itself hallucinate. An AI grader would be a second step
   for the parts a mechanical check can't see.
