"""Render a chat request into the token ids Sushi will see (the pack's chat template via transformers).

Text conversations, with or without tools, are candidates for a handoff; images, a trailing assistant message or a
tool_choice that makes Sushi add its own instruction ("required", a named function) return None and are passed through
untouched. Tools render as Sushi renders them (checked token for token on a 10K agent conversation, 2026-10-07): the
request's tools as given, tool-call arguments as a mapping; tool_choice "none" drops them. Reasoning an agent sends
back on assistant history goes to the template as Sushi passes it (server.zig messageReasoningFromObj, 2026-10-08):
`reasoning_content`, else `reasoning`, non-empty strings only, assistant turns only; the template then keeps it on
every turn unless preserve_thinking is false. Strata does not render tools the same way (it unwraps each tool to its
function object), so it gets the prompt as token ids (`kvh_prompt_ids`, local Strata patch) and renders nothing itself.

Thinking follows Sushi's own rules (1.1.1, unchanged in 1.2.0 while no --think flag is set) (server.zig resolveEnableThinking / parseReasoningEffort, chat.zig
qwen38EffortFor), which differ from the template's defaults: `reasoning_effort` counts only at the top level (inside
chat_template_kwargs it is ignored), thinking on without an effort word is "low", and a request naming neither is
thinking off. Strata follows the template, so it gets the resolved settings written out explicitly.
"""
import hashlib
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


def _has_float(v):
    if isinstance(v, float):
        return True
    if isinstance(v, dict):
        return any(_has_float(x) for x in v.values())
    if isinstance(v, list):
        return any(_has_float(x) for x in v)
    return False


def sushi_tools(tools):
    """The tool list as Sushi hands it to the template (chat.zig fillOptionalToolDefKeys, 1.1.1 and 1.2.0): a function
    without "description" gets "" and one without "parameters" gets an empty object schema, appended after its own
    keys. None when Sushi would re-serialise numbers we cannot promise to reproduce (a fill plus a float)."""
    out, changed = [], False
    for t in tools:
        fn = t.get("function")
        if isinstance(fn, dict) and ("description" not in fn or "parameters" not in fn):
            fn = dict(fn)
            if "description" not in fn:
                fn["description"] = ""
            if "parameters" not in fn:
                fn["parameters"] = {"type": "object", "properties": {}}
            t, changed = dict(t, function=fn), True
        out.append(t)
    if changed and _has_float(tools):
        return None
    return out


def _text(c):
    """A message's content as Sushi joins it, or None when it is not plain text."""
    if c is None:
        return ""
    if isinstance(c, list):
        if any(not isinstance(p, dict) or p.get("type") != "text" for p in c):
            return None
        return "\n".join(p.get("text") or "" for p in c)  # Sushi joins text parts with "\n" (checked live)
    return c if isinstance(c, str) else None


def _reasoning(m):
    """The history reasoning Sushi hands the template for an assistant message, or None."""
    for k in ("reasoning_content", "reasoning"):
        v = m.get(k)
        if isinstance(v, str) and v:
            return v
    return None


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
        tj = os.path.join(model_dir, "tokenizer.json")
        h = lambda b: hashlib.sha256(b).hexdigest()[:16]
        self.identity = {"template": h((self.tok.chat_template or "").encode()),
                         "tokenizer": h(open(tj, "rb").read()) if os.path.exists(tj) else None}

    def prepare(self, body):
        return self.prepare_why(body)[0]

    def prepare_why(self, body):
        """(Prepared, None), or (None, why) when Sushi would not render the request the way this module models."""
        for k in ("functions", "response_format", "ignore_eos"):
            if body.get(k):
                return None, k                  # Sushi writes a schema instruction into the prompt / refuses ignore_eos
        tools = body.get("tools")
        choice = tool_choice_kind(body.get("tool_choice"))
        if tools is not None and choice in ("required", "named"):
            return None, f"tool_choice {choice}"   # Sushi appends its own "You MUST call ..." instruction
        if choice == "none" or tools is None:
            tools, has_tools = None, False
        elif not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
            return None, "malformed tools"
        else:
            has_tools = True
            tools = sushi_tools(tools)
            if tools is None:
                return None, "unmodelled tool schema"
        msgs = []
        for i, m in enumerate(body.get("messages") or []):
            role = m.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                return None, f"message {i} role {role!r}"
            c = _text(m.get("content"))
            if c is None:
                return None, f"message {i} ({role}) content not plain text"
            msg = {"role": role, "content": c}
            if role == "assistant" and _reasoning(m) is not None:
                msg["reasoning_content"] = _reasoning(m)
            if m.get("tool_calls"):
                if role != "assistant":
                    return None, f"message {i} ({role}) has tool_calls"
                calls = _tool_calls(m["tool_calls"])
                if calls is None:
                    return None, f"message {i} unmodelled tool_calls"
                msg["tool_calls"] = calls
            msgs.append(msg)
        if not msgs or msgs[-1]["role"] not in ("user", "tool"):
            return None, "no messages" if not msgs else f"last message is {msgs[-1]['role']}"
        kw = sushi_template_kwargs(body)
        if kw is None:
            return None, "thinking settings Sushi refuses or this module does not model"
        text = self.tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=True, **kw)
        ids = self.tok.encode(text, add_special_tokens=False)
        sbody = {k: v for k, v in body.items() if k not in THINKING_FIELDS}
        sbody["chat_template_kwargs"] = dict(kw)
        sbody["kvh_prompt_ids"] = ids
        return Prepared(ids, text, sbody, kw["enable_thinking"], has_tools), None

    def render_ids(self, body):
        p = self.prepare(body)
        return None if p is None else p.ids
