# pytest-llm-cost

A pytest plugin that reports what the LLM calls in your test suite cost, test by test, in US dollars. It can fail the run when the suite goes over a budget. It needs no code changes: it sees every response from the OpenAI and Anthropic Python SDKs and LiteLLM, reads the token counts, and prices them with [llm-prices](https://github.com/Mattbusel/llm-prices).

## Install

```
pip install pytest-llm-cost
```

## Example

```
pytest --llm-budget 0.50
```

Calls the plugin cannot see (another SDK, raw HTTP, a recorded cassette) can be added by hand:

```python
import pytest


@pytest.mark.llm_budget(0.05)
def test_one_gpt4o_call(llm_cost):
    llm_cost.record("gpt-4o", provider="openai", input_tokens=12_000, output_tokens=800)
    assert round(llm_cost.total, 3) == 0.038
```

```
================================== LLM cost ===================================
test                               calls         in       out          USD
test_demo.py::test_one_gpt4o_call      1     12,000       800    $0.038000
by model: openai/gpt-4o 1 x $0.038000
total $0.038000 over 1 call in 1 test (llm-prices 1.3.0, checked 2026-11-15)
budget $0.5, 8% used
```

## Docs and license

- Options, fixture, pricing rules and xdist: [docs/usage.md](https://github.com/Mattbusel/pytest-llm-cost/blob/main/docs/usage.md)
- PyPI: https://pypi.org/project/pytest-llm-cost/
- MIT license: [LICENSE](https://github.com/Mattbusel/pytest-llm-cost/blob/main/LICENSE)
