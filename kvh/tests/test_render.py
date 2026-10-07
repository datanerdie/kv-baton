import os

import pytest

from render import SUSHI_MODEL_DIR, Renderer

pytestmark = pytest.mark.skipif(not os.path.isdir(SUSHI_MODEL_DIR), reason="needs the Sushi pack")


@pytest.fixture(scope="module")
def r():
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
