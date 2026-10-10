"""Agent tests. A scripted stand-in replaces the AI model, so these run offline and cost nothing."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import agent  # noqa: E402
import evaluate_agent  # noqa: E402
import run_agent  # noqa: E402
from ledger import Ledger  # noqa: E402
from models import MockClient, ScriptedClient, to_openai_messages, to_openai_tools  # noqa: E402

ALERT = dict(alert_id="AL1", account="A00001", date="2026-02-10", rules="R1_LARGE_TXN")


def ledger():
    rows = [("2026-02-09 09:00", "A00001", "A00002", 50000, "wire"), ("2026-02-09 13:00", "A00002", "A00003", 49500, "wire"),
            ("2026-02-09 18:00", "A00003", "A00001", 49000, "wire")]
    df = pd.DataFrame(rows, columns=["timestamp", "from_account", "to_account", "amount", "payment_type"])
    df["timestamp"] = pd.to_datetime(df.timestamp)
    df.insert(0, "txn_id", [f"T{i}" for i in range(len(df))])
    df["is_laundering"], df["pattern_id"], df["pattern_type"] = 1, "P1", "cycle"
    return Ledger(df)


def call(name, **inp):
    return [{"type": "tool_use", "id": f"id_{name}", "name": name, "input": inp}]


def verdict(**over):
    v = dict(disposition="escalate", risk_score=90, typology="cycle", key_evidence=["funds returned via A00002, A00003"],
             linked_accounts=["A00002", "A00003"], narrative="Cycle found.")
    v.update(over)
    return call("submit_verdict", **v)


def run(script, **kw):
    client = ScriptedClient(script)
    return agent.investigate(client, ledger(), **ALERT, **kw), client


def test_agent_uses_tools_then_returns_verdict():
    v, client = run([call("find_cycles", account="A00001", as_of="2026-02-10"), verdict()])
    assert v.status == "ok" and v.disposition == "escalate" and v.tool_calls == 1
    assert v.linked_accounts == ["A00002", "A00003"]
    # the tool's real output was sent back to the model
    sent = client.calls[1]["messages"][-1]["content"][0]
    assert sent["type"] == "tool_result" and json.loads(sent["content"])["cycles_found"] == 1
    assert [s["type"] for s in v.trace] == ["tool", "verdict"]


def test_model_failure_escalates_to_human():
    v, _ = run([RuntimeError("api down")])
    assert v.status == "failsafe" and v.disposition == "escalate"


def test_no_verdict_within_limit_escalates():
    v, _ = run([call("find_cycles", account="A00001", as_of="2026-02-10")] * 3, max_turns=3)
    assert v.status == "failsafe" and v.disposition == "escalate" and "turn limit" in v.failure_reason


def test_agent_cannot_dismiss_by_talking_without_a_verdict():
    v, _ = run([[{"type": "text", "text": "Looks fine, dismiss."}]] * 2, max_turns=2)
    assert v.status == "failsafe" and v.disposition == "escalate"


def test_invalid_verdict_gets_one_retry_then_failsafe():
    v, client = run([verdict(risk_score=900), verdict()])
    assert v.status == "ok" and "rejected" in client.calls[1]["messages"][-1]["content"][0]["content"]
    v, _ = run([verdict(risk_score=900), verdict(typology="aliens")])
    assert v.status == "failsafe" and v.disposition == "escalate"


def test_inconsistent_verdict_is_rejected():
    assert "inconsistent" in agent.validate_verdict(dict(disposition="dismiss", risk_score=80, typology="none",
                                                         key_evidence=["x"], narrative="n"))


def test_invented_accounts_are_dropped():
    v, _ = run([call("find_cycles", account="A00001", as_of="2026-02-10"),
                verdict(linked_accounts=["A00002", "A99999"])])
    assert v.linked_accounts == ["A00002"] and v.dropped_accounts == ["A99999"]


def test_tool_budget_is_enforced():
    many = [{"type": "tool_use", "id": f"c{i}", "name": "find_cycles",
             "input": {"account": "A00001", "as_of": "2026-02-10"}} for i in range(5)]
    v, client = run([many, verdict()], max_tool_calls=2)
    assert v.tool_calls == 2
    assert "budget exhausted" in client.calls[1]["messages"][-1]["content"][-1]["content"]


def test_bad_tool_call_returns_error_not_crash():
    v, client = run([call("drop_database"), verdict()])
    assert v.status == "ok" and "unknown tool" in client.calls[1]["messages"][-1]["content"][0]["content"]


def test_agent_never_sees_labels():
    _, client = run([call("get_transactions", account="A00001", as_of="2026-02-10"), verdict()])
    everything = json.dumps(client.calls)
    for leak in ("is_laundering", "pattern_id", "pattern_type", "is_true_alert", "P1"):
        assert leak not in everything
    assert not set(run_agent.AGENT_COLUMNS) & {"is_true_alert", "patterns"}


def test_openai_translation():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": call("find_cycles", account="A")},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "id_find_cycles", "content": "{}"}]}]
    out = to_openai_messages("sys", msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"account": "A"}
    assert out[3]["tool_call_id"] == "id_find_cycles"
    assert to_openai_tools([agent.VERDICT_TOOL])[0]["function"]["name"] == "submit_verdict"


def test_mock_pipeline_end_to_end():
    v = agent.investigate(MockClient(), ledger(), **ALERT)
    assert v.status == "ok" and v.disposition == "escalate" and v.typology == "cycle"


def test_stratified_sampling_and_scoring():
    alerts = pd.DataFrame({"alert_id": [f"AL{i}" for i in range(100)], "account": "A00001", "date": "2026-02-10",
                           "rules": "R1_LARGE_TXN", "is_true_alert": [1] * 10 + [0] * 90,
                           "patterns": ["P1"] * 10 + [""] * 90})
    chosen = run_agent.pick_alerts(alerts, 20, "stratified", 1)
    assert chosen.is_true_alert.sum() == 10 and len(chosen) == 20
    # agent escalates 9 of 10 true alerts and 2 of 10 false alarms
    esc = set(chosen[chosen.is_true_alert == 1].alert_id[:9]) | set(chosen[chosen.is_true_alert == 0].alert_id[:2])
    verdicts = pd.DataFrame({"alert_id": chosen.alert_id, "account": "A00001",
                             "disposition": ["escalate" if a in esc else "dismiss" for a in chosen.alert_id]})
    verdicts["risk_score"] = verdicts.disposition.map({"escalate": 80, "dismiss": 10})
    verdicts["typology"] = "cycle"
    for c in ("linked_accounts", "dropped_accounts"):
        verdicts[c] = "[]"
    verdicts["status"], verdicts["failure_reason"] = "ok", ""
    for c in ("tool_calls", "turns", "seconds", "input_tokens", "output_tokens", "cached_tokens"):
        verdicts[c] = 1
    m = evaluate_agent.evaluate(verdicts, alerts, ledger_tx())
    assert m["true_alerts_escalated"]["rate"] == 0.9 and m["false_alarms_dismissed"]["rate"] == 0.8
    # re-weighted to 10 true + 90 false: 9 + 18 = 27 alerts left, precision 9/27
    assert m["at_agent_decision"]["alerts_for_human_review"] == 27
    assert m["at_agent_decision"]["precision"] == round(9 / 27, 4)
    assert m["at_agent_decision"]["false_alarm_reduction"] == 0.8


def ledger_tx():
    return pd.DataFrame({"from_account": ["A00001"], "to_account": ["A00002"], "is_laundering": [1],
                         "pattern_id": ["P1"], "pattern_type": ["cycle"]})


def test_wilson_interval():
    lo, hi = evaluate_agent.wilson(9, 10)
    assert 0.59 < lo < 0.61 and 0.98 < hi < 0.99
