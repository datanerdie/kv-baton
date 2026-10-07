import os

import pytest

from render import SUSHI_MODEL_DIR, Renderer, sushi_template_kwargs, tool_choice_kind

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
    body("x", functions=[{"name": "f", "parameters": {}}]),
    body("x", tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}], tool_choice="required"),
    body("x", tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
         tool_choice={"type": "function", "function": {"name": "f"}}),
    body("x", tools="f"),
    {"model": "m", "messages": []},
])
def test_not_candidates(r, b):
    assert r.render_ids(b) is None


def test_a_conversation_without_a_user_query_is_the_templates_error(r):
    with pytest.raises(Exception, match="No user query"):           # the proxy passes a render failure through
        r.prepare({"model": "m", "messages": [{"role": "tool", "content": "r", "tool_call_id": "1"}]})


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
                                   {"reasoning_effort": "minimal"},                    # Sushi 1.2.0: a 400 on qwen4_exp
                                   {"chat_template_kwargs": "enable_thinking"}])
def test_sushi_template_kwargs_refused_or_unmodelled(extra):
    assert sushi_template_kwargs(body("x", **extra)) is None


@pytest.mark.parametrize("extra", [{"response_format": {"type": "json_object"}},
                                   {"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}},
                                   {"ignore_eos": True}])
def test_prepare_refuses_requests_sushi_rewrites_or_refuses(r, extra):
    assert r.prepare(body("hi", **extra)) is None


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


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file — größe ≤ 1 MB.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def agent(args):
    return [{"role": "user", "content": "Summarise /tmp/a.txt"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": args}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "the text"}]


@pytest.mark.parametrize("value, kind", [(None, "auto"), ("auto", "auto"), ("none", "none"), ("required", "required"),
                                         ("any", "required"), ({"type": "none"}, "none"), ({"type": "any"}, "required"),
                                         ({"type": "function", "function": {"name": "f"}}, "named"),
                                         ({"type": "tool", "name": "f"}, "named"), ({"type": "function"}, "auto")])
def test_tool_choice_kind_follows_sushi(value, kind):
    assert tool_choice_kind(value) == kind


def test_tools_render_wrapped_as_sushi_does_and_strata_gets_ids(r):
    p = r.prepare({"model": "m", "messages": agent('{"path": "/tmp/a.txt"}'), "tools": TOOLS,
                   "chat_template_kwargs": {"enable_thinking": False}})
    assert p.has_tools and p.strata_body["kvh_prompt_ids"] == p.ids
    assert '{"type": "function", "function": {"name": "read_file"' in p.text
    assert "<tool_response>\nthe text\n</tool_response>" in p.text and p.text.endswith("<think>\n\n</think>\n\n")


def test_tool_call_arguments_as_string_or_mapping_render_alike(r):
    a = r.prepare({"model": "m", "messages": agent('{"path": "/tmp/a.txt"}'), "tools": TOOLS})
    b = r.prepare({"model": "m", "messages": agent({"path": "/tmp/a.txt"}), "tools": TOOLS})
    assert a.ids == b.ids


def test_malformed_tool_call_arguments_pass_through(r):
    assert r.prepare({"model": "m", "messages": agent('{"path": '), "tools": TOOLS}) is None


def test_tool_choice_none_drops_the_tools_like_sushi(r):
    msgs = [{"role": "user", "content": "hi"}]
    p = r.prepare({"model": "m", "messages": msgs, "tools": TOOLS, "tool_choice": "none"})
    assert not p.has_tools and p.ids == r.prepare({"model": "m", "messages": msgs}).ids


def test_plain_requests_also_send_ids_to_strata(r):
    p = r.prepare(body("hi"))
    assert not p.has_tools and p.strata_body["kvh_prompt_ids"] == p.ids


def test_tools_missing_optional_keys_are_filled_like_sushi(r):
    """Sushi appends "description": "" and an empty parameters schema to a function lacking them (measured on 1.2.0:
    271 / 259 tokens where the request as given renders 266 / 245)."""
    P = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    msgs = [{"role": "user", "content": "What is the weather in Oslo?"}]
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    no_desc = [{"type": "function", "function": {"name": "get_weather", "parameters": P}}]
    no_params = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather."}}]
    assert len(r.prepare(dict({"model": "m", "messages": msgs, "tools": no_desc}, **kw)).ids) == 271
    assert len(r.prepare(dict({"model": "m", "messages": msgs, "tools": no_params}, **kw)).ids) == 259
    assert "description" not in no_desc[0]["function"]                  # the request itself is not changed


def test_a_fill_with_floats_passes_through(r):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "x": {"type": "number", "default": 0.5}}}}}]
    assert r.prepare({"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": tools}) is None
