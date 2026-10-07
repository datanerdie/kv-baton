import os

import pytest

from render import SUSHI_MODEL_DIR, Renderer, sushi_template_kwargs

@pytest.fixture(scope="module")
def r():
    if not os.path.isdir(SUSHI_MODEL_DIR):
        pytest.skip("needs the Sushi pack")
    return Renderer(SUSHI_MODEL_DIR)


def body(content, **kw):
    return {"model": "qwen38-flash-bigdoc", "messages": [{"role": "user", "content": content}], **kw}


def test_content_array_parts_are_joined_with_a_newline(r):
    """Sushi joins text parts with "\\n" (checked against a live Sushi cache entry, 2026-10-07)."""
    kw = {"enable_thinking": False}
    a = r.render_ids(body("hello\nthere", chat_template_kwargs=kw))
    b = r.render_ids(body([{"type": "text", "text": "hello"}, {"type": "text", "text": "there"}], chat_template_kwargs=kw))
    c = r.render_ids(body("hellothere", chat_template_kwargs=kw))
    assert a and a == b and a != c


def test_thinking_kwargs_change_the_render(r):
    off = r.render_ids(body("hi", chat_template_kwargs={"enable_thinking": False}))
    on = r.render_ids(body("hi", chat_template_kwargs={"enable_thinking": True}))
    assert off != on


@pytest.mark.parametrize("b", [
    body([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]),
    body("x", tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}]),
    {"model": "m", "messages": [{"role": "tool", "content": "r", "tool_call_id": "1"}]},
    {"model": "m", "messages": []},
])
def test_not_candidates(r, b):
    assert r.render_ids(b) is None


def test_matches_sushi_tokens_for_the_8k_prompt(r):
    """Same render as Sushi's own entry for this request (verified token-identical 2026-10-07)."""
    from array import array
    ref_path = os.path.expanduser("~/.sushi/kv-cache/526f67face43a3d8/e672/tokens.bin")
    if not os.path.exists(ref_path):
        pytest.skip("reference entry gone")
    doc = open("/Users/Shared/longctx/doc-400k.txt").read()
    q = ("\n\nIn about 200 words, summarise what happens in the last part of the text above, "
         "naming the people involved.")
    ids = r.render_ids(body(doc[:33_000] + q, chat_template_kwargs={"enable_thinking": False}))
    ref = array("I"); ref.frombytes(open(ref_path, "rb").read())
    assert ids == list(ref)[:8262]


OFF = {"enable_thinking": False, "reasoning_effort": "low"}


def on(effort):
    return {"enable_thinking": True, "reasoning_effort": effort}


@pytest.mark.parametrize("extra, want", [
    ({}, OFF),                                                                          # Sushi: silent = off (29 tokens)
    ({"chat_template_kwargs": {"enable_thinking": False}}, OFF),
    ({"chat_template_kwargs": {"enable_thinking": True}}, on("low")),                   # 53 tokens = low
    ({"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}}, on("low")),  # kwargs effort ignored
    ({"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}, on("low")),
    ({"reasoning_effort": "low"}, on("low")),                                           # top level: 53 / 27 / 65
    ({"reasoning_effort": "medium"}, on("medium")),
    ({"reasoning_effort": "xhigh"}, on("xhigh")),
    ({"reasoning_effort": "minimal"}, on("low")),
    ({"reasoning_effort": "none"}, OFF),
    ({"reasoning_effort": "off"}, OFF),
    ({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": True}}, on("low")),   # the two are OR'd
    ({"reasoning_effort": "off", "chat_template_kwargs": {"enable_thinking": True}}, on("xhigh")),  # qwen38EffortFor
    ({"reasoning_effort": 5}, OFF),                                                     # non-string: ignored
    ({"enable_thinking": True, "chat_template_kwargs": {"enable_thinking": False}}, on("low")),     # top level wins
    ({"chat_template_kwargs": {"enable_thinking": "no"}}, OFF),                         # only a bool counts
    ({"chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}}, dict(OFF, preserve_thinking=False)),
])
def test_sushi_template_kwargs(extra, want):
    assert sushi_template_kwargs(body("x", **extra)) == want


@pytest.mark.parametrize("extra", [{"reasoning_effort": "high"}, {"reasoning_effort": "max"}, {"reasoning": {"effort": "low"}},
                                   {"chat_template_kwargs": "enable_thinking"}])
def test_sushi_template_kwargs_refused_or_unmodelled(extra):
    assert sushi_template_kwargs(body("x", **extra)) is None


def test_prepare_spells_sushis_settings_out_for_strata(r):
    b = body("hi", reasoning_effort="medium", chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "xhigh"}, top_k=20)
    p = r.prepare(b)
    assert p.thinking and p.strata_body["chat_template_kwargs"] == on("medium")
    assert "reasoning_effort" not in p.strata_body and p.strata_body["top_k"] == 20
    assert p.ids == r.tok.encode(p.text, add_special_tokens=False)
    direct = r.tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True, **on("medium"))
    assert p.text == direct


@pytest.mark.parametrize("msgs", [
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning_content": "because"}, {"role": "user", "content": "q2"}],
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "partial"}],
])
def test_prepare_refuses_prior_reasoning_and_trailing_assistant(r, msgs):
    assert r.prepare({"model": "m", "messages": msgs}) is None
