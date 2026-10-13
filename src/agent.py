"""The triage agent: investigates one alert and returns a verdict for a human.

Loop: the model reads the alert, chooses investigation tools, reads the evidence,
and repeats until it submits a structured verdict. The agent recommends; a human
investigator decides.

Safety properties (each covered by a test):
  * Fail-safe: any error, timeout, or missing verdict ESCALATES to a human.
    The agent can never dismiss an alert by failing.
  * Bounded: a hard cap on model turns and tool calls per alert.
  * Grounded: accounts named in the verdict must appear in tool output.
  * Auditable: every tool call, result, and the final reasoning is recorded.
  * Blind: the agent never receives ground-truth labels.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

import tools as toolbox
from ledger import Ledger

MAX_TURNS = 8
MAX_TOOL_CALLS = 16
MAX_RESULT_CHARS = 6000
TYPOLOGIES = ["none", "structuring", "fan_out", "fan_in", "cycle", "pass_through", "layering", "other"]

RULE_MEANINGS = {
    "R1_LARGE_TXN": "a single transaction of $10,000 or more",
    "R2_STRUCTURING": "three or more cash deposits between $8,000 and $10,000 within 7 days",
    "R3_FAN_OUT": "payments of $1,000+ to six or more different accounts in one day",
    "R4_FAN_IN": "payments of $1,000+ from six or more different accounts in one day",
    "R5_RAPID_MOVEMENT": "$5,000+ received with at least 80% sent out the same or next day",
}

SYSTEM_PROMPT = """You are an anti-money-laundering (AML) analyst doing the first-pass review of \
transaction-monitoring alerts at a bank. A senior human investigator makes the final decision; \
your job is to investigate, weigh the evidence, and give them a clear recommendation.

Most alerts are false alarms: the monitoring rules are simple thresholds and they flag a great \
deal of ordinary activity. Your value is in telling the two apart reliably.

How to work
1. Start from the rule that fired and form a hypothesis about what it might indicate.
2. Use the tools to test that hypothesis.
3. Judge every signal against this account's own norm, which the tools report alongside it. \
A signal that is routine for this account is not evidence.
4. Look for an innocent explanation as seriously as you look for a suspicious one.
5. Call trace_network to see whether specific suspicious behaviour ties this account to others.
6. When you have enough evidence, call submit_verdict. Do not use more tool calls than you need.

Reading the evidence correctly
- New counterparties: active businesses pay and are paid by new parties all the time. Compare \
new_share with usual_new_share_in_earlier_periods. Only a clear departure from the norm matters.
- Volume: a total two or three times the average is ordinary variation. Look at \
earlier_periods_with_higher_total; if the account has had busier periods before, the volume is unremarkable.
- Pass-through pairs: a busy account produces coincidental matches. Look at \
earlier_periods_with_matched_pairs, the share of inflow involved, and whether the pair is isolated.
- Missing history: the dataset begins on the date shown as data_starts. When little earlier data \
exists, that is a limit of the data. Never describe an account as new, newly opened, or lacking \
history as a reason for suspicion.
- Size alone: large amounts are normal for corporate and business accounts. Size is a reason the \
rule fired, not evidence of laundering.
- Network: trace_network follows only relayed funds, one-day bursts to mostly-new accounts, and \
cycles. An empty result means no such behaviour links this account to others, which supports an \
innocent reading. Having many ordinary counterparties is not a network.

Common innocent explanations
- Payroll: regular payments to the same recipients on a recurring schedule.
- Rent or supplier collections: many payers on a predictable date, seen in earlier periods too.
- Operating businesses: steady money in and out at volumes typical for the account's history.
- One-off life events: a single large payment to an established business with no onward chain.
- Cash-intensive businesses: cash deposits across a wide range of amounts over a long period.

Common laundering patterns
- Structuring: cash deposits clustered just below $10,000, often followed by a wire out.
- Fan-out / fan-in: a burst of payments to or from many accounts with no prior relationship.
- Pass-through (mule): funds received and forwarded within hours, nearly intact, to a different party.
- Cycles: funds leaving and returning through a ring of accounts.
- Layering: funds split across several accounts and regrouped elsewhere.

An account can also be a minor participant: it may have received or relayed one payment in a \
scheme run by others. If the account itself looks ordinary but it has a new, unexplained link \
to accounts behaving suspiciously, examine those accounts before deciding.

Standards
- Base every statement on tool output. Never invent transactions, amounts, or accounts.
- Missing history is not evidence of guilt; say so when history is too short to judge.
- Dismiss only when you have a specific innocent explanation supported by evidence.
- Roughly 1 alert in 25 is genuine. Escalate when there is specific positive evidence of a \
laundering pattern, not merely because activity is large, busy, or imperfectly explained.
- Equally, do not dismiss an alert that shows a clear pattern because the account otherwise looks ordinary.
- risk_score is your estimated probability (0-100) that the account is involved in laundering.
- Escalate when risk_score is 50 or higher.
"""

VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit the final recommendation for this alert. Call exactly once, when the investigation is complete.",
    "input_schema": {
        "type": "object",
        "properties": {
            "disposition": {"type": "string", "enum": ["escalate", "dismiss"]},
            "risk_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "typology": {"type": "string", "enum": TYPOLOGIES},
            "key_evidence": {"type": "array", "items": {"type": "string"},
                             "description": "Specific facts from tool output that support the decision"},
            "innocent_explanation": {"type": "string",
                                     "description": "The most plausible innocent explanation, and whether the evidence supports it"},
            "linked_accounts": {"type": "array", "items": {"type": "string"},
                                "description": "Other account IDs believed to be part of the same scheme (empty if dismissing)"},
            "narrative": {"type": "string", "description": "3-6 sentence case summary for the human investigator"},
        },
        "required": ["disposition", "risk_score", "typology", "key_evidence", "narrative"],
    },
}


@dataclass
class Verdict:
    alert_id: str
    account: str
    disposition: str = "escalate"
    risk_score: int = 50
    typology: str = "other"
    key_evidence: list = field(default_factory=list)
    innocent_explanation: str = ""
    linked_accounts: list = field(default_factory=list)
    narrative: str = ""
    status: str = "ok"                 # ok | failsafe
    failure_reason: str = ""
    dropped_accounts: list = field(default_factory=list)
    tool_calls: int = 0
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    seconds: float = 0.0
    trace: list = field(default_factory=list)

    def to_dict(self, with_trace=True):
        d = asdict(self)
        if not with_trace:
            d.pop("trace")
        return d


def alert_prompt(alert_id: str, account: str, date: str, rules: str) -> str:
    fired = "\n".join(f"- {r}: {RULE_MEANINGS.get(r, 'unknown rule')}" for r in rules.split("|"))
    return (f"Alert {alert_id}\nAccount: {account}\nAlert date: {date}\nRules that fired:\n{fired}\n\n"
            "Investigate this alert and submit your verdict.")


def make_verdict_tool(typologies: list) -> dict:
    """The same verdict form with a different list of scheme types (datasets name them differently)."""
    tool = json.loads(json.dumps(VERDICT_TOOL))
    tool["input_schema"]["properties"]["typology"]["enum"] = list(typologies)
    return tool


def validate_verdict(v: dict, typologies: list = TYPOLOGIES, escalate_at: int = 50) -> str | None:
    """Returns an error message the model can act on, or None if the verdict is usable."""
    for k in VERDICT_TOOL["input_schema"]["required"]:
        if k not in v:
            return f"missing required field '{k}'"
    if v["disposition"] not in ("escalate", "dismiss"):
        return "disposition must be 'escalate' or 'dismiss'"
    if not isinstance(v["risk_score"], (int, float)) or isinstance(v["risk_score"], bool) or not 0 <= v["risk_score"] <= 100:
        return "risk_score must be a number from 0 to 100"
    if v["typology"] not in typologies:
        return f"typology must be one of {list(typologies)}"
    if not isinstance(v["key_evidence"], list) or not v["key_evidence"]:
        return "key_evidence must be a non-empty list"
    if v["disposition"] == "dismiss" and v["risk_score"] >= escalate_at:
        return f"inconsistent: disposition is 'dismiss' but risk_score is {escalate_at} or higher"
    if v["disposition"] == "escalate" and v["risk_score"] < escalate_at:
        return f"inconsistent: disposition is 'escalate' but risk_score is below {escalate_at}"
    return None


ACCOUNT_RE = re.compile(r"\bA\d{5,7}\b")


def investigate(client, ledger, alert_id: str, account: str, date: str, rules: str,
                max_turns: int = MAX_TURNS, max_tool_calls: int = MAX_TOOL_CALLS,
                tools=toolbox, system_prompt: str | None = None, first_message: str | None = None,
                typologies: list = TYPOLOGIES, escalate_at: int = 50, verdict_check=None) -> Verdict:
    """Investigate one alert. `tools`, `system_prompt`, `first_message` and `typologies` let another
    dataset plug in its own evidence tools and case file while keeping every safety check here.
    `verdict_check(verdict, trace, first_message)` returns problems with the write-up; a verdict with problems is
    sent back once to be rewritten (a second submission is accepted and its remaining problems are recorded)."""
    v = Verdict(alert_id=alert_id, account=account)
    t0 = time.time()
    system_prompt = system_prompt or SYSTEM_PROMPT
    all_tools = tools.TOOL_SPECS + [make_verdict_tool(typologies)]
    first_message = first_message or alert_prompt(alert_id, account, date, rules)
    messages = [{"role": "user", "content": [{"type": "text", "text": first_message}]}]
    seen_accounts = {account} | set(ACCOUNT_RE.findall(first_message))
    rejected_once = False
    revised_once = False

    def failsafe(reason):
        v.status, v.failure_reason = "failsafe", reason
        v.disposition, v.risk_score, v.typology = "escalate", max(50, escalate_at), "other"
        v.narrative = f"Automated review did not complete ({reason}). Routed to a human investigator."
        v.seconds = round(time.time() - t0, 2)
        return v

    for _ in range(max_turns):
        try:
            resp = client.complete(system_prompt, messages, all_tools)
        except Exception as e:  # network errors, rate limits after retries, bad responses
            return failsafe(f"model error: {type(e).__name__}")
        v.turns += 1
        v.input_tokens += resp.input_tokens
        v.output_tokens += resp.output_tokens
        v.cached_tokens += resp.cached_tokens
        messages.append({"role": "assistant", "content": resp.blocks})
        for b in resp.blocks:
            if b["type"] == "text":
                v.trace.append({"type": "reasoning", "text": b["text"]})

        calls = resp.tool_calls
        if not calls:
            messages.append({"role": "user", "content": [{"type": "text", "text":
                             "Continue the investigation with a tool, or call submit_verdict."}]})
            continue

        results = []
        for c in calls:
            if c["name"] == "submit_verdict":
                err = validate_verdict(c["input"], typologies, escalate_at)
                if err is None and verdict_check is not None:
                    problems = verdict_check(c["input"], v.trace, first_message)
                    if problems and not revised_once:
                        revised_once = True
                        v.trace.append({"type": "revision_requested", "problems": problems})
                        results.append({"type": "tool_result", "tool_use_id": c["id"], "content": json.dumps({"error":
                            "verdict returned for one correction: " + "; ".join(problems) + ". Every usd value is in US "
                            "dollars and amount is in the payment's own currency. Rewrite those amounts and resubmit."})})
                        continue
                    if problems:
                        v.trace.append({"type": "problems_remaining", "problems": problems})
                if err is None:
                    d = c["input"]
                    linked = [a for a in d.get("linked_accounts", []) if isinstance(a, str) and a != account]
                    v.linked_accounts = [a for a in linked if a in seen_accounts]
                    v.dropped_accounts = [a for a in linked if a not in seen_accounts]
                    v.disposition, v.risk_score, v.typology = d["disposition"], int(d["risk_score"]), d["typology"]
                    v.key_evidence = [str(x) for x in d["key_evidence"]]
                    v.innocent_explanation = str(d.get("innocent_explanation", ""))
                    v.narrative = str(d["narrative"])
                    v.trace.append({"type": "verdict", "input": d})
                    v.seconds = round(time.time() - t0, 2)
                    return v
                if rejected_once:
                    return failsafe(f"invalid verdict twice: {err}")
                rejected_once = True
                results.append({"type": "tool_result", "tool_use_id": c["id"],
                                "content": json.dumps({"error": f"verdict rejected: {err}. Fix and resubmit."})})
                v.trace.append({"type": "verdict_rejected", "error": err})
                continue
            if v.tool_calls >= max_tool_calls:
                out = {"error": "tool budget exhausted; call submit_verdict now"}
            else:
                v.tool_calls += 1
                out = tools.call_tool(ledger, c["name"], c["input"])
            text = json.dumps(out, default=str)
            seen_accounts.update(ACCOUNT_RE.findall(text))
            v.trace.append({"type": "tool", "name": c["name"], "input": c["input"], "output": out})
            results.append({"type": "tool_result", "tool_use_id": c["id"], "content": text[:MAX_RESULT_CHARS]})
        messages.append({"role": "user", "content": results})

    return failsafe("no verdict within the turn limit")
