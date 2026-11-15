import asyncio

import pytest

litellm = pytest.importorskip("litellm")

MSG = [{"role": "user", "content": "hi"}]


def test_litellm_completion_counted_once(llm_cost):
    response = litellm.completion(model="gpt-4o-mini", messages=MSG, mock_response="ok")
    assert len(llm_cost.calls) == 1
    call = llm_cost.calls[0]
    assert call.source == "litellm" and call.priced_as == "openai/gpt-4o-mini"
    usage = response.usage
    assert call.input_tokens == usage.prompt_tokens and call.output_tokens == usage.completion_tokens
    assert llm_cost.total > 0


def test_litellm_acompletion(llm_cost):
    asyncio.run(litellm.acompletion(model="anthropic/claude-sonnet-4-6", messages=MSG, mock_response="ok"))
    assert len(llm_cost.calls) == 1
    assert llm_cost.calls[0].priced_as == "anthropic/claude-sonnet-4-6"


def test_litellm_against_openai_sdk_is_not_double_counted(fake_api, llm_cost):
    litellm.completion(
        model="openai/gpt-4o-mini", messages=MSG, api_base=fake_api.url + "/v1", api_key="test", num_retries=0
    )
    assert len(llm_cost.calls) == 1
    assert llm_cost.calls[0].source == "litellm"
