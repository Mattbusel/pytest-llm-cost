"""The plugin end to end: report, budgets, marker, JSON, xdist. Each case runs a pytest session."""

import json

import pytest

pytest.importorskip("openai")

SUITE = """
import os
import openai
import pytest

client = openai.OpenAI(api_key="test", base_url=os.environ["FAKE_LLM_URL"] + "/v1", max_retries=0)

def ask(model="gpt-4o-mini"):
    return client.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])

def test_cheap():
    ask()

def test_twice():
    ask()
    ask("gpt-4o")

def test_no_llm():
    pass
"""


@pytest.fixture
def suite(pytester, fake_api):
    pytester.makepyfile(test_suite=SUITE)
    return pytester


def test_report_lists_tests_by_cost(suite, cost_of):
    result = suite.runpytest()
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(
        [
            "llm-cost: prices from llm-prices *",
            "*= LLM cost =*",
            "test * calls * in * out * USD",
            "test_suite.py::test_twice * 2 * 2,000 * 600 *",
            "test_suite.py::test_cheap * 1 * 1,000 * 300 *",
            "by model: openai/gpt-4o *; openai/gpt-4o-mini 2 x *",
            "total $* over 3 calls in 2 tests (llm-prices *",
        ]
    )
    assert "test_no_llm" not in result.stdout.str().split("LLM cost")[1]


def test_no_report_without_llm_calls(pytester):
    pytester.makepyfile("def test_x():\n    pass\n")
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    assert "LLM cost" not in result.stdout.str()


def test_budget_exceeded_fails_the_run(suite):
    result = suite.runpytest("--llm-budget", "0.000001")
    result.assert_outcomes(passed=3)
    assert result.ret == 1
    result.stdout.fnmatch_lines(["LLM budget exceeded: $* > $0.000001"])


def test_budget_met_passes(suite):
    result = suite.runpytest("--llm-budget", "5")
    assert result.ret == 0
    result.stdout.fnmatch_lines(["llm-cost: * budget $5", "budget $5, *% used"])


def test_budget_from_ini(suite):
    suite.makeini("[pytest]\nllm_budget = 0.000001\n")
    assert suite.runpytest().ret == 1


def test_top_limits_rows(suite):
    result = suite.runpytest("--llm-cost-top", "1")
    result.stdout.fnmatch_lines(["test_suite.py::test_twice *", "... 1 more (--llm-cost-top 0 lists all)"])


def test_marker_fails_an_expensive_test(pytester, fake_api):
    pytester.makepyfile(test_m=SUITE + """
@pytest.mark.llm_budget(0.000001)
def test_capped():
    ask()

@pytest.mark.llm_budget(1)
def test_roomy():
    ask()
""")
    result = pytester.runpytest("-k", "capped or roomy")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*LLM cost $* is over this test's llm_budget of $0.000001*"])


def test_json_report(suite, tmp_path, cost_of):
    out = tmp_path / "cost.json"
    result = suite.runpytest("--llm-cost-json", str(out))
    assert result.ret == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["calls"]) == 3
    assert data["tests"]["test_suite.py::test_cheap"]["usd"] == str(cost_of("gpt-4o-mini"))
    assert data["llm_prices"] and data["over_budget"] is False
    assert {c["model"] for c in data["calls"]} == {"gpt-4o", "gpt-4o-mini"}


def test_fixture_inside_a_session(pytester, fake_api):
    pytester.makepyfile(test_f=SUITE + """
def test_assert_spend(llm_cost):
    ask()
    assert len(llm_cost.calls) == 1
    assert 0 < llm_cost.total < 0.01
""")
    pytester.runpytest("-k", "assert_spend").assert_outcomes(passed=1)


def test_conftest_imports_are_hooked(pytester, fake_api):
    pytester.makeconftest("import openai\nfrom openai import OpenAI\n")
    pytester.makepyfile(test_suite=SUITE)
    result = pytester.runpytest_subprocess("-k", "cheap")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["total $* over 1 call in 1 test *"])


def test_disable_with_p_no(suite):
    result = suite.runpytest_subprocess("-p", "no:llm_cost")
    result.assert_outcomes(passed=3)
    assert "LLM cost" not in result.stdout.str()


def test_xdist_workers_report_to_the_controller(suite):
    pytest.importorskip("xdist")
    result = suite.runpytest_subprocess("-n", "2", "--llm-budget", "5")
    result.assert_outcomes(passed=3)
    result.stdout.fnmatch_lines(["total $* over 3 calls in 2 tests *", "budget $5, *"])
