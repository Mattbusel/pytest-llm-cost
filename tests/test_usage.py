from decimal import Decimal

import llm_prices

from pytest_llm_cost import Tokens, find_price, provider_from_url, tokens_from_usage
from pytest_llm_cost._core import Recorder, cost_of


def test_openai_chat_usage_takes_cached_tokens_out_of_input():
    usage = {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 400}}
    assert tokens_from_usage(usage) == Tokens(input=600, output=50, cache_read=400)


def test_openai_responses_usage():
    usage = {"input_tokens": 1000, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 100}}
    assert tokens_from_usage(usage) == Tokens(input=900, output=50, cache_read=100)


def test_anthropic_usage_keeps_cache_separate():
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_input_tokens": 3000,
        "cache_creation_input_tokens": 500,
    }
    assert tokens_from_usage(usage) == Tokens(input=10, output=20, cache_read=3000, cache_write_5m=500)


def test_anthropic_usage_splits_cache_lifetimes():
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_input_tokens": 700,
        "cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 500},
    }
    assert tokens_from_usage(usage) == Tokens(input=10, output=20, cache_write_5m=200, cache_write_1h=500)


def test_litellm_usage_with_anthropic_cache_fields():
    # LiteLLM folds cache reads and writes into prompt_tokens.
    usage = {
        "prompt_tokens": 3510,
        "completion_tokens": 20,
        "cache_read_input_tokens": 3000,
        "cache_creation_input_tokens": 500,
    }
    assert tokens_from_usage(usage) == Tokens(input=10, output=20, cache_read=3000, cache_write_5m=500)


def test_gemini_usage_metadata():
    usage = {
        "prompt_token_count": 100,
        "cached_content_token_count": 40,
        "candidates_token_count": 7,
        "thoughts_token_count": 3,
    }
    assert tokens_from_usage(usage) == Tokens(input=60, output=10, cache_read=40)


def test_no_usage():
    assert tokens_from_usage(None) is None
    assert tokens_from_usage({}) is None
    assert tokens_from_usage(object()) is None


def test_provider_from_url():
    assert provider_from_url("https://api.openai.com/v1") == "openai"
    assert provider_from_url("https://api.anthropic.com") == "anthropic"
    assert provider_from_url("https://api.groq.com/openai/v1") == "groq"
    assert provider_from_url("https://api.deepseek.com") == "deepseek"
    assert provider_from_url("https://openrouter.ai/api/v1") == "openrouter"
    assert provider_from_url("https://my-res.openai.azure.com/openai") == "azure"
    assert provider_from_url("https://bedrock-runtime.us-east-1.amazonaws.com") == "bedrock"
    assert provider_from_url("https://us-east5-aiplatform.googleapis.com/v1") == "vertex"
    assert provider_from_url("http://127.0.0.1:8000/v1") is None
    assert provider_from_url("") is None


def test_known_host_is_never_priced_with_another_hosts_row():
    assert find_price("groq", "gpt-4o") is None
    assert find_price("ollama", "gpt-4o") is None
    assert find_price(None, "gpt-4o").provider == "openai"
    assert find_price("deepseek", "deepseek-chat").provider == "deepseek"


def test_embedding_rows_price_input_only():
    price = llm_prices.lookup("text-embedding-3-small")
    assert cost_of(price, Tokens(input=1_000_000)) == price.rate_decimal("input")
    assert cost_of(price, Tokens(input=10, output=5)) is None


def test_recorder_prices_and_attributes():
    rec = Recorder()
    rec.current_test = "t::a"
    call = rec.record(
        source="manual", model="gpt-4o-mini", usage={"prompt_tokens": 1000, "completion_tokens": 0}
    )
    assert call.test == "t::a"
    assert call.priced_as == "openai/gpt-4o-mini"
    assert call.cost == llm_prices.lookup("gpt-4o-mini").rate_decimal("input") * 1000 / Decimal(1_000_000)
    unknown = rec.record(
        source="manual", model="my-local-model", usage={"prompt_tokens": 5, "completion_tokens": 5}
    )
    assert unknown.cost is None and unknown.priced_as is None


def test_call_round_trips_through_dict():
    rec = Recorder()
    call = rec.record(
        source="openai", model="gpt-4o", provider="openai", tokens=Tokens(input=3, output=4, cache_read=5)
    )
    rec2 = Recorder()
    rec2.extend_from_dicts([call.to_dict()])
    back = rec2.calls[0]
    assert (back.test, back.model, back.provider, back.tokens, back.cost, back.priced_as) == (
        call.test,
        call.model,
        call.provider,
        call.tokens,
        call.cost,
        call.priced_as,
    )
