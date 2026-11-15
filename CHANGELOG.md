# Changelog

## 0.1.0 (2026-11-15)

First release.

- Prices every OpenAI, Anthropic and LiteLLM response seen during a pytest run with llm-prices, including cached-token rates, streams and async clients.
- End-of-run report: most expensive tests, spend by model, unpriced calls and streams that reported no usage.
- `--llm-budget USD` (ini `llm_budget`) fails the run over budget; `@pytest.mark.llm_budget(usd)` caps a single test.
- `llm_cost` fixture with `.total`, `.calls`, `.measure()` and `.record()`.
- `--llm-cost-json PATH` writes every call for CI.
- pytest-xdist: workers report to the controller, which applies the budget to the whole run.
