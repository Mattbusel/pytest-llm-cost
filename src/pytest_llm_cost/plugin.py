"""The pytest plugin: per-test attribution, the ``llm_cost`` fixture, budgets and the report."""

from __future__ import annotations

import json
from collections import OrderedDict
from decimal import Decimal
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import llm_prices
import pytest

from . import _core, _hooks
from ._core import Call, Recorder, Tokens

_PLUGIN_NAME = "llm_cost_session"


# --------------------------------------------------------------------------- options


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llm-cost", "LLM cost (pytest-llm-cost)")
    group.addoption(
        "--llm-budget",
        type=float,
        default=None,
        metavar="USD",
        help="fail the run when the LLM calls made during it cost more than USD dollars.",
    )
    group.addoption(
        "--llm-cost-top",
        type=int,
        default=None,
        metavar="N",
        help="number of most expensive tests to list in the cost report (default 10, 0 lists all).",
    )
    group.addoption(
        "--llm-cost-json",
        default=None,
        metavar="PATH",
        help="write every LLM call, its tokens and its price to PATH as JSON.",
    )
    parser.addini("llm_budget", "fail the run when LLM calls cost more than this many USD", default=None)
    parser.addini("llm_cost_top", "number of tests to list in the cost report (0 lists all)", default="10")


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(
    early_config: pytest.Config, parser: pytest.Parser, args: List[str]
) -> None:
    # Before conftest files run, so SDKs that conftest imports are hooked as well.
    _hooks.install()


def pytest_configure(config: pytest.Config) -> None:
    _hooks.install()
    config.addinivalue_line(
        "markers",
        "llm_budget(usd): fail this test when its LLM calls cost more than usd dollars.",
    )
    session = CostSession(config)
    config.pluginmanager.register(session, _PLUGIN_NAME)
    _core.push(session.recorder)


def pytest_unconfigure(config: pytest.Config) -> None:
    session = config.pluginmanager.get_plugin(_PLUGIN_NAME)
    if isinstance(session, CostSession):
        _core.pop(session.recorder)
        config.pluginmanager.unregister(session, _PLUGIN_NAME)


def _budget(config: pytest.Config) -> Optional[Decimal]:
    value = config.getoption("--llm-budget")
    if value is None:
        raw = config.getini("llm_budget")
        if raw in (None, ""):
            return None
        value = raw
    try:
        return Decimal(str(value))
    except Exception:
        raise pytest.UsageError(f"llm_budget must be a number of USD, got {value!r}") from None


def _top(config: pytest.Config) -> int:
    value = config.getoption("--llm-cost-top")
    if value is None:
        try:
            value = int(config.getini("llm_cost_top"))
        except (TypeError, ValueError):
            value = 10
    return max(int(value), 0)


# --------------------------------------------------------------------------- fixture


class CostView:
    """The LLM calls of one test, or of one ``measure()`` block, updated as they happen."""

    def __init__(self, recorder: Recorder, select: Callable[[Call], bool]) -> None:
        self._recorder = recorder
        self._select = select

    @property
    def calls(self) -> List[Call]:
        """Every response seen, in order, priced or not."""
        return [c for c in self._recorder.snapshot() if self._select(c)]

    @property
    def total_decimal(self) -> Decimal:
        """Exact USD spent, priced calls only."""
        return _core.total(iter(self.calls))

    @property
    def total(self) -> float:
        """USD spent, priced calls only."""
        return float(self.total_decimal)

    @property
    def input_tokens(self) -> int:
        """Prompt tokens, including cache reads and writes."""
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def unpriced(self) -> List[Call]:
        """Calls whose model has no price in llm-prices or that reported no usage."""
        return [c for c in self.calls if c.cost is None]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} calls={len(self.calls)} total=${self.total:.6f}>"


class LLMCost(CostView):
    """What the ``llm_cost`` fixture returns: the current test's LLM spend."""

    def __init__(self, recorder: Recorder, nodeid: str) -> None:
        super().__init__(recorder, lambda c: c.test == nodeid)
        self.nodeid = nodeid

    @property
    def session_total(self) -> float:
        """USD spent so far by the whole run (this process only under xdist)."""
        return float(_core.total(iter(self._recorder.snapshot())))

    def record(
        self,
        model: str,
        usage: Any = None,
        *,
        provider: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> Call:
        """Adds a call the hooks cannot see (another SDK, a raw HTTP call, a recorded fixture).

        Pass the response's ``usage`` object or dict, or the token counts. ``input_tokens``
        excludes ``cache_read_tokens``. ``provider`` is an llm-prices provider name
        (``"openai"``, ``"anthropic"``, ``"groq"``, ...); without one the model id is looked up
        across every provider.
        """
        tokens = None
        if usage is None:
            tokens = Tokens(input=input_tokens, output=output_tokens, cache_read=cache_read_tokens)
        return self._recorder.record(
            source="manual",
            provider=provider,
            model=model,
            usage=usage,
            tokens=tokens,
            test=self.nodeid,
        )

    def measure(self) -> "_Measure":
        """``with llm_cost.measure() as m:`` tracks only the calls made inside the block."""
        return _Measure(self._recorder, self.nodeid)


class _Measure(CostView):
    def __init__(self, recorder: Recorder, nodeid: str) -> None:
        self._start = 0
        self._end: Optional[int] = None
        super().__init__(
            recorder,
            lambda c: c.test == nodeid and c.seq >= self._start and (self._end is None or c.seq < self._end),
        )

    def __enter__(self) -> "_Measure":
        calls = self._recorder.snapshot()
        self._start = calls[-1].seq + 1 if calls else 0
        return self

    def __exit__(self, *exc: Any) -> None:
        calls = self._recorder.snapshot()
        self._end = calls[-1].seq + 1 if calls else 0


@pytest.fixture
def llm_cost(request: pytest.FixtureRequest) -> LLMCost:
    """The current test's LLM spend: ``.total`` (USD), ``.calls``, ``.record()``, ``.measure()``."""
    session = request.config.pluginmanager.get_plugin(_PLUGIN_NAME)
    return LLMCost(session.recorder, request.node.nodeid)


# --------------------------------------------------------------------------- session


def _usd(value: Optional[Decimal]) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "$0"
    if value < 1:
        return f"${value:.6f}"
    return f"${value:,.2f}"


def _limit(value: Decimal) -> str:
    """A budget as the user wrote it: $5, $0.5, $0.000001."""
    return f"${value.normalize():f}"


class CostSession:
    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.recorder = Recorder()
        self.budget = _budget(config)
        self.top = _top(config)
        self.json_path = config.getoption("--llm-cost-json")
        self.is_worker = hasattr(config, "workerinput")
        self.over_budget = False

    # attribution

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: Optional[pytest.Item]) -> Iterator[None]:
        self.recorder.current_test = item.nodeid
        try:
            yield
        finally:
            self.recorder.current_test = ""

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo) -> Iterator[None]:
        outcome = yield
        if call.when != "call":
            return
        marker = item.get_closest_marker("llm_budget")
        if marker is None:
            return
        report = outcome.get_result()
        if not report.passed:
            return
        limit = marker.args[0] if marker.args else marker.kwargs.get("usd")
        try:
            limit = Decimal(str(limit))
        except Exception:
            raise pytest.UsageError(
                f"{item.nodeid}: llm_budget needs a number of USD, got {limit!r}"
            ) from None
        spent = _core.total(c for c in self.recorder.snapshot() if c.test == item.nodeid)
        if spent > limit:
            report.outcome = "failed"
            report.longrepr = (
                f"LLM cost {_usd(spent)} is over this test's llm_budget of {_limit(limit)} "
                f"(prices from llm-prices {llm_prices.__version__})"
            )

    # xdist

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: Any) -> None:
        rows = getattr(node, "workeroutput", {}).get("llm_cost_calls")
        if rows:
            self.recorder.extend_from_dicts(rows)

    # end of run

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        calls = self.recorder.snapshot()
        if self.is_worker:
            self.config.workeroutput["llm_cost_calls"] = [c.to_dict() for c in calls]
            return
        spent = _core.total(iter(calls))
        if self.budget is not None and spent > self.budget:
            self.over_budget = True
            if session.exitstatus == pytest.ExitCode.OK:
                session.exitstatus = pytest.ExitCode.TESTS_FAILED
        if self.json_path:
            self._write_json(calls, spent)

    def _write_json(self, calls: List[Call], spent: Decimal) -> None:
        tests: Dict[str, Dict[str, Any]] = OrderedDict()
        for test, test_calls in _by_test(calls):
            tests[test] = {"calls": len(test_calls), "usd": str(_core.total(iter(test_calls)))}
        data = {
            "llm_prices": llm_prices.__version__,
            "prices_checked": llm_prices.GENERATED,
            "total_usd": str(spent),
            "budget_usd": None if self.budget is None else str(self.budget),
            "over_budget": self.over_budget,
            "tests": tests,
            "calls": [c.to_dict() for c in calls],
        }
        with open(self.json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")

    def pytest_report_header(self, config: pytest.Config) -> str:
        line = f"llm-cost: prices from llm-prices {llm_prices.__version__} (checked {llm_prices.GENERATED})"
        if self.budget is not None:
            line += f", budget {_limit(self.budget)}"
        return line

    def pytest_terminal_summary(self, terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
        if self.is_worker:
            return
        calls = self.recorder.snapshot()
        if not calls and self.budget is None:
            return
        tr = terminalreporter
        tr.write_sep("=", "LLM cost")
        spent = _core.total(iter(calls))
        if not calls:
            tr.write_line("no LLM calls seen")
        else:
            self._write_tests(tr, calls)
            self._write_models(tr, calls)
        tests = len({c.test for c in calls if c.test})
        tr.write_line(
            f"total {_usd(spent)} over {len(calls)} call{'s' if len(calls) != 1 else ''} "
            f"in {tests} test{'s' if tests != 1 else ''} "
            f"(llm-prices {llm_prices.__version__}, checked {llm_prices.GENERATED})"
        )
        if self.budget is not None:
            share = "" if self.budget == 0 else f", {spent / self.budget:.0%} used"
            if self.over_budget:
                tr.write_line(
                    f"LLM budget exceeded: {_usd(spent)} > {_limit(self.budget)}", red=True, bold=True
                )
            else:
                tr.write_line(f"budget {_limit(self.budget)}{share}", green=True)
        if self.json_path:
            tr.write_line(f"wrote {self.json_path}")

    def _write_tests(self, tr: Any, calls: List[Call]) -> None:
        rows: List[Tuple[str, int, int, int, Decimal, int]] = []
        for test, test_calls in _by_test(calls):
            rows.append(
                (
                    test or "(outside tests)",
                    len(test_calls),
                    sum(c.input_tokens for c in test_calls),
                    sum(c.output_tokens for c in test_calls),
                    _core.total(iter(test_calls)),
                    sum(1 for c in test_calls if c.cost is None),
                )
            )
        rows.sort(key=lambda r: r[4], reverse=True)
        shown = rows if self.top == 0 else rows[: self.top]
        name_width = min(max([len("test")] + [len(r[0]) for r in shown]), 70)
        tr.write_line(f"{'test':<{name_width}}  {'calls':>5}  {'in':>9}  {'out':>8}  {'USD':>11}")
        for name, n, tin, tout, usd, unpriced in shown:
            if len(name) > name_width:
                name = "..." + name[-(name_width - 3) :]
            note = f"  ({unpriced} unpriced)" if unpriced else ""
            tr.write_line(f"{name:<{name_width}}  {n:>5}  {tin:>9,}  {tout:>8,}  {_usd(usd):>11}{note}")
        if len(rows) > len(shown):
            tr.write_line(f"... {len(rows) - len(shown)} more (--llm-cost-top 0 lists all)")

    def _write_models(self, tr: Any, calls: List[Call]) -> None:
        models: Dict[str, List[Call]] = OrderedDict()
        for c in calls:
            key = c.priced_as or (f"{c.provider}/{c.model}" if c.provider else c.model or "(unknown model)")
            models.setdefault(key, []).append(c)
        parts = []
        for key, group in sorted(models.items(), key=lambda kv: _core.total(iter(kv[1])), reverse=True):
            priced = [c for c in group if c.cost is not None]
            if priced:
                parts.append(f"{key} {len(group)} x {_usd(_core.total(iter(priced)))}")
        if parts:
            tr.write_line("by model: " + "; ".join(parts))
        no_usage = [c for c in calls if c.tokens is None]
        no_price = [c for c in calls if c.tokens is not None and c.cost is None]
        if no_price:
            names = sorted(
                {
                    (f"{c.provider}/{c.model}" if c.provider else c.model) or "(unknown model)"
                    for c in no_price
                }
            )
            tr.write_line(
                f"unpriced: {len(no_price)} call(s) to {', '.join(names)} (no price in llm-prices)",
                yellow=True,
            )
        if no_usage:
            tr.write_line(
                f"no usage: {len(no_usage)} streamed call(s) reported no token counts "
                "(OpenAI streams need stream_options={'include_usage': True})",
                yellow=True,
            )


def _by_test(calls: List[Call]) -> List[Tuple[str, List[Call]]]:
    grouped: Dict[str, List[Call]] = OrderedDict()
    for c in calls:
        grouped.setdefault(c.test, []).append(c)
    return list(grouped.items())
