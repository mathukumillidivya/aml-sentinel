"""Model adapters: the only place that knows about a specific AI provider.

The agent speaks one neutral message format. Each adapter translates that format
to and from a provider's API, so swapping Claude for OpenAI is a one-word change
and both can be benchmarked on identical alerts.

Neutral format
  messages: [{"role": "user"|"assistant", "content": [block, ...]}]
  blocks:   {"type": "text", "text": str}
            {"type": "tool_use", "id": str, "name": str, "input": dict}
            {"type": "tool_result", "tool_use_id": str, "content": str}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_MODELS = {"anthropic": "claude-haiku-4-5-20251001", "mock": "mock-rules"}


@dataclass
class ModelResponse:
    blocks: list            # assistant content in the neutral format
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def tool_calls(self):
        return [b for b in self.blocks if b["type"] == "tool_use"]


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str | None = None, max_tokens: int = 2000):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=5)  # reads ANTHROPIC_API_KEY
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list, tools: list) -> ModelResponse:
        tools = [dict(t) for t in tools]
        if tools:  # cache the unchanging prefix (tools + instructions) to cut cost
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=tools, messages=messages)
        blocks = []
        for b in r.content:
            if b.type == "text" and b.text:
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
        u = r.usage
        cached = (getattr(u, "cache_read_input_tokens", 0) or 0)
        created = (getattr(u, "cache_creation_input_tokens", 0) or 0)
        return ModelResponse(blocks, u.input_tokens + cached + created, u.output_tokens, cached)


def to_openai_messages(system: str, messages: list) -> list:
    """Translate the neutral format into OpenAI chat messages."""
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "assistant":
            text = "".join(b["text"] for b in m["content"] if b["type"] == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in m["content"] if b["type"] == "tool_use"]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            for b in m["content"]:
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
                else:
                    out.append({"role": "user", "content": b["text"]})
    return out


def to_openai_tools(tools: list) -> list:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str | None = None, max_tokens: int = 2000):
        import openai
        self.client = openai.OpenAI(max_retries=5)  # reads OPENAI_API_KEY
        if not model:
            raise SystemExit("pass --model for the openai provider")
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list, tools: list) -> ModelResponse:
        r = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=self.max_tokens,
            messages=to_openai_messages(system, messages), tools=to_openai_tools(tools))
        msg = r.choices[0].message
        blocks = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for c in msg.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": c.function.arguments}
            blocks.append({"type": "tool_use", "id": c.id, "name": c.function.name, "input": args})
        u = r.usage
        details = getattr(u, "prompt_tokens_details", None)
        return ModelResponse(blocks, u.prompt_tokens, u.completion_tokens,
                             getattr(details, "cached_tokens", 0) or 0)


@dataclass
class ScriptedClient:
    """Test double: replays a fixed list of responses. Used by the unit tests."""
    script: list
    provider: str = "scripted"
    model: str = "scripted"
    calls: list = field(default_factory=list)

    def complete(self, system, messages, tools) -> ModelResponse:
        self.calls.append({"system": system, "messages": json.loads(json.dumps(messages)), "tools": tools})
        if not self.script:
            raise RuntimeError("script exhausted")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return ModelResponse(step, 100, 20)


class MockClient:
    """Offline stand-in with NO intelligence: runs a fixed checklist, then applies
    hard-coded thresholds. It exists so the whole pipeline can be exercised for
    free in tests and CI. Its scores are not a result of any kind."""
    provider, model = "mock", DEFAULT_MODELS["mock"]
    CHECKS = ["check_structuring", "check_fan_pattern", "check_pass_through", "find_cycles"]

    def complete(self, system, messages, tools) -> ModelResponse:
        first = messages[0]["content"][0]["text"]
        account = first.split("Account: ")[1].split()[0]
        as_of = first.split("Alert date: ")[1].split()[0]
        results = {}
        for m in messages:
            for b in m["content"]:
                if b["type"] == "tool_result":
                    results[b["tool_use_id"]] = json.loads(b["content"])
        if not results:
            return ModelResponse([{"type": "tool_use", "id": n, "name": n,
                                   "input": {"account": account, "as_of": as_of}} for n in self.CHECKS], 500, 60)
        s, f, p, c = (results[n] for n in self.CHECKS)
        score, why, typ = 5, [], "none"
        if c["cycles_found"]:
            score, typ = 90, "cycle"
            why.append(f"{c['cycles_found']} cycle(s) returning funds to the account")
        if s["deposits_just_under_limit"] >= 3 and s["just_under_share"] >= 0.6:
            score, typ = max(score, 85), "structuring"
            why.append(f"{s['deposits_just_under_limit']} cash deposits just under the reporting limit")
        for k in ("fan_out", "fan_in"):
            if f[k]["max_in_one_day"] >= 5 and f[k]["new_counterparty_share"] >= 0.7 and not f[k]["prior_days_with_similar_fan"]:
                score, typ = max(score, 80), k
                why.append(f"{k}: {f[k]['max_in_one_day']} mostly-new counterparties in one day")
        if p["matched_pairs"] and score < 60:
            score, typ = 55, "pass_through"
            why.append(f"{p['matched_pairs']} in-and-out pair(s) within 48h")
        verdict = {"disposition": "escalate" if score >= 50 else "dismiss", "risk_score": score, "typology": typ,
                   "key_evidence": why or ["no red flags in the standard checks"],
                   "innocent_explanation": "" if score >= 50 else "activity consistent with ordinary account use",
                   "linked_accounts": [], "narrative": "Mock verdict from fixed thresholds."}
        return ModelResponse([{"type": "tool_use", "id": "verdict", "name": "submit_verdict", "input": verdict}], 900, 120)


def make_client(provider: str, model: str | None = None):
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        return AnthropicClient(model)
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        return OpenAIClient(model)
    if provider == "mock":
        return MockClient()
    raise SystemExit(f"unknown provider '{provider}'")
