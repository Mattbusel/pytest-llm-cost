import os
import sys
from decimal import Decimal

import llm_prices
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from fakeapi import CACHED, COMPLETION, PROMPT, FakeAPI  # noqa: E402


@pytest.fixture(scope="session")
def fake_api():
    with FakeAPI() as api:
        os.environ["FAKE_LLM_URL"] = api.url
        yield api
        os.environ.pop("FAKE_LLM_URL", None)


def expected(model, provider=None, input=PROMPT - CACHED, output=COMPLETION, cache_read=CACHED):
    """What llm-prices says the fake response costs, so the tests never hard-code a price."""
    price = llm_prices.lookup_provider(provider, model) if provider else llm_prices.lookup(model)
    usage = llm_prices.Usage(input=input, output=output, cache_read=cache_read)
    return price.cost_usage(usage, exact=True).total


@pytest.fixture
def cost_of():
    return expected


@pytest.fixture
def zero():
    return Decimal(0)
