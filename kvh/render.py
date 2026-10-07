"""Render a chat request into the token ids Sushi will see (the pack's chat template via transformers).

Text conversations, with or without tools, are candidates for a handoff; images, prior reasoning, a trailing
assistant message or a tool_choice that makes Sushi add its own instruction ("required", a named function) return None
and are passed through untouched. Tools render as Sushi renders them (checked token for token on a 10K agent
conversation, 2026-10-07): the request's tools as given, tool-call arguments as a mapping; tool_choice "none" drops
them. Strata does not render tools the same way (it unwraps each tool to its function object), so it gets the prompt
as token ids (`kvh_prompt_ids`, local Strata patch) and renders nothing itself.

Thinking follows Sushi's own rules (1.1.1, unchanged in 1.2.0 while no --think flag is set) (server.zig resolveEnableThinking / parseReasoningEffort, chat.zig
qwen38EffortFor), which differ from the template's defaults: `reasoning_effort` counts only at the top level (inside
chat_template_kwargs it is ignored), thinking on without an effort word is "low", and a request naming neither is
thinking off. Strata follows the template, so it gets the resolved settings written out explicitly.
"""
import json
import os
from dataclasses import dataclass

SUSHI_MODEL_DIR = os.path.expanduser(os.environ.get("KVH_SUSHI_MODEL_DIR", "~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw"))

# The pack's generation_config.json declares no thinking default, and qwen4_exp is not on Sushi's thinking-on
# allowlist, so a request naming neither enable_thinking nor reasoning_effort runs thinking off (measured 2026-10-07).
ARCH_DEFAULT_THINKING = False
# Sushi's effort table for qwen4_exp (model.zig qwen4_exp_efforts); "none" is an alias of off. Other words get a 400
# (since 1.2.0 also "minimal", which 1.1.1 took as low).
EFFORT_ARMS = ("off", "low", "medium", "xhigh")
# Request fields the Strata copy must not carry: Sushi's resolution of them is written into chat_template_kwargs.
THINKING_FIELDS = ("enable_thinking", "reasoning_effort")


@dataclass
class Prepared:
    ids: list           # what Sushi will tokenise
    text: str           # the rendered prompt, for the check against Sushi's /tokenize
    strata_body: dict   # the request for Strata: these ids as kvh_prompt_ids, Sushi's thinking settings spelled out
    thinking: bool
    has_tools: bool = False     # Sushi's cache key: only an entry with the same flag is ever restored


def tool_choice_kind(value):
    """Sushi's parseToolChoice: "none", "required" (also "any"), "named", or "auto" (anything else)."""
    if isinstance(value, str):
        return value if value in ("none", "required") else "required" if value == "any" else "auto"
    if isinstance(value, dict):
        kind = value.get("type")
        if kind in ("none", "any", "required"):
            return "none" if kind == "none" else "required"
        fn = value.get("function") if isinstance(value.get("function"), dict) else value
        if isinstance(fn.get("name"), str) and fn["name"]:
            return "named"
    return "auto"


def _text(c):
    """A message's content as Sushi joins it, or None when it is not plain text."""
    if c is None:
        return ""
    if isinstance(c, list):
        if any(not isinstance(p, dict) or p.get("type") != "text" for p in c):
            return None
        return "\n".join(p.get("text") or "" for p in c)  # Sushi joins text parts with "\n" (checked live)
    return c if isinstance(c, str) else None


def _tool_calls(calls):
    """Assistant tool calls for the template (it iterates `arguments|items`): JSON-string arguments parsed, as Sushi
    does. None when they are not well-formed."""
    if not isinstance(calls, list):
        return None
    out = []
    for c in calls:
        fn = c.get("function") if isinstance(c, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            return None
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                return None
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        out.append({"type": "function", "function": {"name": fn["name"], "arguments": args}})
    return out


def sushi_template_kwargs(body):
    """The template kwargs Sushi renders this request with, or None when Sushi would refuse it or it is not modelled."""
    if "reasoning" in body:
        return None
    kwargs = body.get("chat_template_kwargs")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        return None

    et = body.get("enable_thinking")
    if not isinstance(et, bool):
        et = kwargs.get("enable_thinking")
        if not isinstance(et, bool):
            et = None

    word = body.get("reasoning_effort")
    if not isinstance(word, str):
        word, effort_on = None, None
    else:
        arm = "off" if word == "none" else word
        if arm not in EFFORT_ARMS:
            return None                         # Sushi answers 400
        effort_on = arm != "off"

    if et is None and word is None:
        enable = ARCH_DEFAULT_THINKING
    else:
        enable = bool(et) or bool(effort_on)

    if not enable or word is None or word in ("low", "none"):
        effort = "low"
    elif word == "medium":
        effort = "medium"
    else:
        effort = "xhigh"                        # including "off" when enable_thinking forces thinking on
    out = {"enable_thinking": enable, "reasoning_effort": effort}
    if isinstance(kwargs.get("preserve_thinking"), bool):
        out["preserve_thinking"] = kwargs["preserve_thinking"]
    return out


class Renderer:
    def __init__(self, model_dir=SUSHI_MODEL_DIR):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)

    def prepare(self, body):
        if body.get("functions") or body.get("response_format") or body.get("ignore_eos"):
            return None                         # Sushi writes a schema instruction into the prompt / refuses ignore_eos
        tools = body.get("tools")
        choice = tool_choice_kind(body.get("tool_choice"))
        if tools is not None and choice in ("required", "named"):
            return None                         # Sushi appends its own "You MUST call ..." instruction
        if choice == "none" or tools is None:
            tools, has_tools = None, False
        elif not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
            return None
        else:
            has_tools = True
        msgs = []
        for m in body.get("messages") or []:
            role = m.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                return None
            if m.get("reasoning_content") or m.get("reasoning"):
                return None
            c = _text(m.get("content"))
            if c is None:
                return None
            msg = {"role": role, "content": c}
            if m.get("tool_calls"):
                if role != "assistant":
                    return None
                calls = _tool_calls(m["tool_calls"])
                if calls is None:
                    return None
                msg["tool_calls"] = calls
            msgs.append(msg)
        if not msgs or msgs[-1]["role"] not in ("user", "tool"):
            return None
        kw = sushi_template_kwargs(body)
        if kw is None:
            return None
        text = self.tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=True, **kw)
        ids = self.tok.encode(text, add_special_tokens=False)
        sbody = {k: v for k, v in body.items() if k not in THINKING_FIELDS}
        sbody["chat_template_kwargs"] = dict(kw)
        sbody["kvh_prompt_ids"] = ids
        return Prepared(ids, text, sbody, kw["enable_thinking"], has_tools)

    def render_ids(self, body):
        p = self.prepare(body)
        return None if p is None else p.ids
