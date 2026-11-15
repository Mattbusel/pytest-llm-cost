# pytest-llm-cost

A pytest plugin that tells you what the LLM calls in your test suite cost, test by test, in US dollars. It can also fail the run when the suite goes over a budget.

```
$ pip install pytest-llm-cost
$ pytest --llm-budget 0.50
...
================================== LLM cost ===================================
test                                         calls        in      out          USD
tests/test_agent.py::test_plans_a_trip           6    48,200    3,900    $0.203100
tests/test_rag.py::test_answers_with_sources     2    12,400      610    $0.002226
tests/test_rag.py::test_refuses_off_topic        1     1,900       40    $0.000309
by model: anthropic/claude-sonnet-4-6 6 x $0.203100; openai/gpt-4o-mini 3 x $0.002535
total $0.205635 over 9 calls in 3 tests (llm-prices 1.3.0, checked 2026-11-15)
budget $0.5, 41% used
```

You do not need to change any code. Once installed, the plugin sees every response from:

- the **OpenAI** Python SDK: chat completions, the Responses API, embeddings, `parse`, streaming, sync and async;
- the **Anthropic** Python SDK: messages, `messages.stream()`, `create(stream=True)`, sync and async;
- **LiteLLM**: `litellm.completion` and `litellm.acompletion`. Each call is counted once, even though LiteLLM calls the OpenAI SDK underneath.

It reads the token counts each response reports. That includes cache reads and writes, so prompt caching is priced at the cached rate. It then prices those counts with [llm-prices](https://github.com/Mattbusel/llm-prices), a dated table of list prices for OpenAI, Anthropic, Google, Bedrock, Vertex, Azure, DeepSeek, Mistral, Groq, Together, Fireworks, xAI, Cohere and OpenRouter. Run `pip install -U llm-prices` to get new prices without upgrading the plugin.

## Options

| Option | ini key | What it does |
|---|---|---|
| `--llm-budget USD` | `llm_budget` | Fails the run (exit code 1) when the LLM spend is over USD. |
| `--llm-cost-top N` | `llm_cost_top` | Sets how many of the most expensive tests the report lists (default 10, 0 for all). |
| `--llm-cost-json PATH` | | Writes every call, with its tokens, its price and the price row used, to a JSON file for CI dashboards. |

```ini
# pytest.ini
[pytest]
llm_budget = 2.00
```

The report only appears when the run made LLM calls or a budget is set. Turn the plugin off for one run with `pytest -p no:llm_cost`.

## Budget per test

```python
import pytest

@pytest.mark.llm_budget(0.02)
def test_summary_stays_cheap(agent):
    agent.summarize(long_document)
```

The test fails when its own calls cost more than $0.02.

## The `llm_cost` fixture

```python
def test_cache_hits_make_the_second_call_cheaper(client, llm_cost):
    with llm_cost.measure() as first:
        client.ask(question)
    with llm_cost.measure() as second:
        client.ask(question)
    assert second.total < first.total / 2
```

| Attribute | |
|---|---|
| `llm_cost.total` | USD spent by this test so far (a float; `total_decimal` is exact) |
| `llm_cost.calls` | every response seen: `model`, `provider`, `input_tokens`, `output_tokens`, `usd`, `priced_as` |
| `llm_cost.input_tokens`, `llm_cost.output_tokens` | token totals for the test |
| `llm_cost.unpriced` | calls with no price (unknown model, or a stream that reported no usage) |
| `llm_cost.measure()` | a context manager with the same attributes for only the calls made inside the block |
| `llm_cost.record(model, usage=..., provider=...)` | adds a call the plugin cannot see, such as another SDK, raw HTTP or a cassette |
| `llm_cost.session_total` | USD spent by the whole run so far |

## How a call is priced

- The provider comes from the client's base URL: `api.openai.com` is OpenAI, `api.groq.com` is Groq, `*.openai.azure.com` is Azure, and so on. For LiteLLM it comes from LiteLLM's own provider name. A request served by Groq is never priced at OpenAI's rate.
- Any other host, such as a local server, a proxy or a mock, is priced by model id alone. A test that talks to a mocked `gpt-4o` therefore shows what that call would cost against the real API.
- A model llm-prices does not list (`llama3` on Ollama, a fine-tune id) is counted and reported as unpriced, never guessed.
- OpenAI streams only report usage when you pass `stream_options={"include_usage": True}`. Streams without it are counted and listed under "no usage".

## pytest-xdist

Each worker records its own calls and sends them to the controller, which prints one report and applies the budget to the whole run. Inside a test, `llm_cost.session_total` covers only that worker.

## Notes

- The plugin hooks an SDK only once something imports it, so a run that never imports `litellm` does not pay for importing it.
- Prices are list prices. Batch discounts, regional surcharges and negotiated rates are not applied.
- A failure inside the plugin never breaks your API call. It emits one warning and leaves that call unrecorded.

MIT licensed. Maintained by Matthew Busel. Price corrections go to [llm-prices](https://github.com/Mattbusel/llm-prices/issues).
