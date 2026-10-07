"""Render a chat request into the token ids Sushi will see (the pack's chat template via transformers).

Only plain text conversations are candidates for a handoff; anything with tools, tool messages, images, prior
reasoning or a trailing assistant message returns None and is passed through untouched.

Thinking follows Sushi 1.1.1's own rules (server.zig resolveEnableThinking / parseReasoningEffort, chat.zig
qwen38EffortFor), which differ from the template's defaults: `reasoning_effort` counts only at the top level (inside
chat_template_kwargs it is ignored), thinking on without an effort word is "low", and a request naming neither is
thinking off. Strata follows the template, so it gets the resolved settings written out explicitly.
"""
import os
from dataclasses import dataclass

SUSHI_MODEL_DIR = os.path.expanduser(os.environ.get("KVH_SUSHI_MODEL_DIR", "~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw"))

# The pack's generation_config.json declares no thinking default, and qwen4_exp is not on Sushi's thinking-on
# allowlist, so a request naming neither enable_thinking nor reasoning_effort runs thinking off (measured 2026-10-07).
ARCH_DEFAULT_THINKING = False
# Sushi's effort table for qwen4_exp (model.zig qwen4_exp_efforts); "none" is an alias of off. Other words get a 400.
EFFORT_ARMS = ("off", "low", "medium", "xhigh")
# Request fields the Strata copy must not carry: Sushi's resolution of them is written into chat_template_kwargs.
THINKING_FIELDS = ("enable_thinking", "reasoning_effort")


@dataclass
class Prepared:
    ids: list           # what Sushi will tokenise
    text: str           # the rendered prompt, for the check against Sushi's /tokenize
    strata_body: dict   # the request for Strata, with Sushi's thinking settings spelled out
    thinking: bool


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
    elif word == "minimal":
        effort_on = True
    else:
        arm = "off" if word == "none" else word
        if arm not in EFFORT_ARMS:
            return None                         # Sushi answers 400
        effort_on = arm != "off"

    if et is None and word is None:
        enable = ARCH_DEFAULT_THINKING
    else:
        enable = bool(et) or bool(effort_on)

    if not enable or word is None or word in ("low", "minimal", "none"):
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
        if body.get("tools") or body.get("functions"):
            return None
        msgs = []
        for m in body.get("messages") or []:
            if m.get("role") not in ("system", "user", "assistant") or m.get("tool_calls"):
                return None
            if m.get("reasoning_content") or m.get("reasoning"):
                return None
            c = m.get("content")
            if isinstance(c, list):
                if any(not isinstance(p, dict) or p.get("type") != "text" for p in c):
                    return None
                c = "\n".join(p.get("text") or "" for p in c)  # Sushi joins text parts with "\n" (checked live)
            if not isinstance(c, str):
                return None
            msgs.append({"role": m["role"], "content": c})
        if not msgs or msgs[-1]["role"] != "user":
            return None
        kw = sushi_template_kwargs(body)
        if kw is None:
            return None
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        sbody = {k: v for k, v in body.items() if k not in THINKING_FIELDS}
        sbody["chat_template_kwargs"] = dict(kw)
        return Prepared(self.tok.encode(text, add_special_tokens=False), text, sbody, kw["enable_thinking"])

    def render_ids(self, body):
        p = self.prepare(body)
        return None if p is None else p.ids
