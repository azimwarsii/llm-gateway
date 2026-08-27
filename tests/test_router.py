from gateway.config import RouterConfig
from gateway.router import route, score
from gateway.schemas import ChatCompletionRequest, Message


def req(text, **kw):
    return ChatCompletionRequest(messages=[Message(role="user", content=text)], **kw)


def test_simple_question_routes_small():
    d = route(req("What is the capital of France?"), RouterConfig())
    assert d.tier == "small"


def test_reasoning_and_code_route_up():
    text = "Analyze the tradeoffs of this design and refactor the code:\n```python\ndef f(x):\n    return x\n```" * 3
    d = route(req(text, max_tokens=1024), RouterConfig())
    assert d.tier in ("medium", "remote")
    assert d.features["reasoning"] == 1.0 and d.features["code"] == 1.0


def test_long_multi_turn_with_tools_routes_remote():
    msgs = [Message(role="user", content="Plan and design a distributed cache. " * 80)] * 6
    r = ChatCompletionRequest(messages=msgs, max_tokens=2048, tools=[{"type": "function", "function": {"name": "x"}}])
    d = route(r, RouterConfig())
    assert d.tier == "remote"


def test_header_override_and_model_pin():
    assert route(req("hi"), RouterConfig(), override="remote").tier == "remote"
    assert route(req("hi", model="medium"), RouterConfig()).tier == "medium"
    assert route(req("hi"), RouterConfig(allow_tier_override=False), override="remote").tier == "small"


def test_score_bounds():
    s, _ = score(req("x"))
    assert 0.0 <= s <= 1.0
