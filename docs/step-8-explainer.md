# Step 8 explainer: does it hold on a second dataset?

## The question

One good result on one dataset could be luck, or could come from tuning the
system to that dataset's quirks without meaning to. The standard check is to run
the same thing on data it was never tuned on. That is called **replication**.

## What was done

IBM published six versions of the benchmark. The first one used here is called
HI-Small. The second, LI-Small, is a separate simulation with less laundering in
it (about 1 payment in 1,700, against 1 in 1,000).

Two tests, with nothing adjusted for the new data:

1. **Replication.** Run the identical pipeline on LI-Small: same four rules, same
   27 facts per alert, same model settings, same split of days.
2. **Transfer.** Take the model that was trained on HI-Small and use it on
   LI-Small as it is, without retraining.

## The result

| Test days of | Alerts | False alarms | Model score | Workload cut, keeping 80% of real cases |
|---|---|---|---|---|
| First dataset | 20,784 | 93.5% | 0.90 | 73% |
| Second dataset | 27,297 | 96.6% | 0.84 | 68% |
| Second dataset, first dataset's model | 27,297 | 96.6% | 0.83 | 67% |

- The result holds. It is a little lower, which is expected: real laundering is
  rarer in the second dataset, so the problem is harder.
- The transferred model does almost as well as one trained on the new data. That
  means it learned something general about laundering in this kind of data, and
  not just the first dataset's details.

## What this does not prove

Both datasets came out of the same simulator, built by the same team. So this is
weaker evidence than testing on data from a completely different source, such as
a real bank. Say this yourself before an interviewer does.

## Words you need to own

- **Replication:** repeating the test on new data to see if the result holds.
- **Generalization:** how well something learned in one place works in another.
- **Transfer:** using a model on data it wasn't trained on.
- **Overfitting:** fitting the quirks of one dataset so closely that it fails elsewhere.
- **Distribution shift:** the new data differing from the old (here, laundering is rarer).

## Questions an interviewer will ask

1. *How do you know you didn't overfit to one dataset?* Replication with nothing
   re-tuned, plus the transfer test.
2. *Why is the second result lower?* Laundering is rarer and the rules reach less
   of it (63% of laundering payments against 77%).
3. *Is that enough to trust it in production?* No. Same simulator, synthetic data.
   A real deployment would need a trial on the bank's own history, run in
   parallel with the existing process before anyone relies on it.
4. *What did the transfer test tell you that replication didn't?* Replication
   retrains on the new data. Transfer shows the already-trained model still works,
   which is closer to what happens when a deployed model meets new conditions.
