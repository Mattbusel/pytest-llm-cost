"""The real OpenAI and Anthropic SDKs against a local fake API, measured by the llm_cost fixture."""

import asyncio

import pytest
from fakeapi import CACHED, COMPLETION, PROMPT

openai = pytest.importorskip("openai")
anthropic = pytest.importorskip("anthropic")

MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture
def oai(fake_api):
    return openai.OpenAI(api_key="test", base_url=fake_api.url + "/v1", max_retries=0)


@pytest.fixture
def aoai(fake_api):
    return openai.AsyncOpenAI(api_key="test", base_url=fake_api.url + "/v1", max_retries=0)


@pytest.fixture
def claude(fake_api):
    return anthropic.Anthropic(api_key="test", base_url=fake_api.url, max_retries=0)


@pytest.fixture
def aclaude(fake_api):
    return anthropic.AsyncAnthropic(api_key="test", base_url=fake_api.url, max_retries=0)


def test_openai_chat(oai, llm_cost, cost_of):
    oai.chat.completions.create(model="gpt-4o-mini", messages=MSG)
    assert len(llm_cost.calls) == 1
    call = llm_cost.calls[0]
    assert (call.source, call.model, call.priced_as) == ("openai", "gpt-4o-mini", "openai/gpt-4o-mini")
    assert call.input_tokens == PROMPT and call.output_tokens == COMPLETION
    assert call.tokens.cache_read == CACHED
    assert llm_cost.total_decimal == cost_of("gpt-4o-mini")
    assert llm_cost.total > 0


def test_openai_chat_async(aoai, llm_cost, cost_of):
    async def go():
        await aoai.chat.completions.create(model="gpt-4o", messages=MSG)
        await aoai.chat.completions.create(model="gpt-4o", messages=MSG)

    asyncio.run(go())
    assert len(llm_cost.calls) == 2
    assert llm_cost.total_decimal == 2 * cost_of("gpt-4o")


def test_openai_stream_with_usage(oai, llm_cost, cost_of):
    stream = oai.chat.completions.create(
        model="gpt-4o-mini", messages=MSG, stream=True, stream_options={"include_usage": True}
    )
    assert llm_cost.calls == []  # recorded once the stream is read
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text == "ok"
    assert isinstance(stream, openai.Stream)
    assert llm_cost.total_decimal == cost_of("gpt-4o-mini")


def test_openai_stream_without_usage_is_counted_but_unpriced(oai, llm_cost):
    for _ in oai.chat.completions.create(model="gpt-4o-mini", messages=MSG, stream=True):
        pass
    assert len(llm_cost.calls) == 1
    assert llm_cost.calls[0].tokens is None
    assert llm_cost.unpriced == llm_cost.calls
    assert llm_cost.total == 0


def test_openai_async_stream(aoai, llm_cost, cost_of):
    async def go():
        stream = await aoai.chat.completions.create(
            model="gpt-4o-mini", messages=MSG, stream=True, stream_options={"include_usage": True}
        )
        async for _ in stream:
            pass

    asyncio.run(go())
    assert llm_cost.total_decimal == cost_of("gpt-4o-mini")


def test_openai_responses_api(oai, llm_cost, cost_of):
    if not hasattr(oai, "responses"):
        pytest.skip("this openai release has no Responses API")
    oai.responses.create(model="gpt-4o-mini", input="hi")
    assert llm_cost.total_decimal == cost_of("gpt-4o-mini")


def test_openai_embeddings(oai, llm_cost, cost_of):
    oai.embeddings.create(model="text-embedding-3-small", input="hi")
    price = __import__("llm_prices").lookup("text-embedding-3-small")
    assert llm_cost.total_decimal == price.rate_decimal("input") * PROMPT / 1_000_000


def test_openai_raw_response_is_not_double_counted(oai, llm_cost):
    oai.chat.completions.with_raw_response.create(model="gpt-4o-mini", messages=MSG).parse()
    assert len(llm_cost.calls) <= 1


def test_anthropic_messages(claude, llm_cost, cost_of):
    claude.messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=MSG)
    call = llm_cost.calls[0]
    assert call.source == "anthropic" and call.priced_as == "anthropic/claude-sonnet-4-6"
    assert llm_cost.total_decimal == cost_of("claude-sonnet-4-6", input=PROMPT)


def test_anthropic_async(aclaude, llm_cost, cost_of):
    asyncio.run(aclaude.messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=MSG))
    assert llm_cost.total_decimal == cost_of("claude-sonnet-4-6", input=PROMPT)


def test_anthropic_stream(claude, llm_cost, cost_of):
    stream = claude.messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=MSG, stream=True)
    for _ in stream:
        pass
    assert len(llm_cost.calls) == 1
    assert llm_cost.calls[0].output_tokens == COMPLETION
    assert llm_cost.total_decimal == cost_of("claude-sonnet-4-6", input=PROMPT)


def test_anthropic_stream_helper(claude, llm_cost, cost_of):
    with claude.messages.stream(model="claude-sonnet-4-6", max_tokens=10, messages=MSG) as stream:
        assert stream.get_final_text() == "ok"
    assert llm_cost.total_decimal == cost_of("claude-sonnet-4-6", input=PROMPT)


def test_measure_and_record(oai, llm_cost, cost_of):
    oai.chat.completions.create(model="gpt-4o-mini", messages=MSG)
    with llm_cost.measure() as m:
        oai.chat.completions.create(model="gpt-4o", messages=MSG)
    oai.chat.completions.create(model="gpt-4o-mini", messages=MSG)
    assert len(m.calls) == 1 and m.total_decimal == cost_of("gpt-4o")
    llm_cost.record(
        "gpt-4o-mini", input_tokens=PROMPT - CACHED, output_tokens=COMPLETION, cache_read_tokens=CACHED
    )
    assert llm_cost.total_decimal == 3 * cost_of("gpt-4o-mini") + cost_of("gpt-4o")
    assert llm_cost.session_total >= llm_cost.total
