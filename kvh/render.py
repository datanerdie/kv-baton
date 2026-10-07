"""Render a chat request into the token ids Sushi will see (the pack's chat template via transformers).

Only plain text conversations are candidates for a handoff; anything with tools, tool messages or images returns
None and is passed through untouched.
"""
import os

SUSHI_MODEL_DIR = os.path.expanduser(os.environ.get("KVH_SUSHI_MODEL_DIR", "~/.sushi/models/Qwen3.8-Flash-Next-Sushi-4bpw"))


class Renderer:
    def __init__(self, model_dir=SUSHI_MODEL_DIR):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)

    def render_ids(self, body):
        if body.get("tools") or body.get("functions"):
            return None
        msgs = []
        for m in body.get("messages") or []:
            if m.get("role") not in ("system", "user", "assistant") or m.get("tool_calls"):
                return None
            c = m.get("content")
            if isinstance(c, list):
                if any(not isinstance(p, dict) or p.get("type") != "text" for p in c):
                    return None
                c = "\n".join(p.get("text") or "" for p in c)  # Sushi joins text parts with "\n" (checked live)
            if not isinstance(c, str):
                return None
            msgs.append({"role": m["role"], "content": c})
        if not msgs:
            return None
        kw = dict(body.get("chat_template_kwargs") or {})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        return self.tok.encode(text, add_special_tokens=False)
