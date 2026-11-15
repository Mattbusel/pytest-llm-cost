"""Token usage parsing, provider detection, pricing and the call log.

Nothing here imports an LLM SDK. Usage objects are read by attribute or key, so pydantic models
from the OpenAI and Anthropic SDKs, LiteLLM's ``Usage`` and plain dicts all work.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Mapping, Optional
from urllib.parse import urlsplit

import llm_prices

_MILLION = Decimal(1_000_000)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """``obj.name`` or ``obj[name]``, with ``default`` for missing and ``None`` values."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        value = obj.get(name, default)
    else:
        try:
            value = getattr(obj, name, default)
        except Exception:
            value = default
    return default if value is None else value


def _count(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class Tokens:
    """Token counts of one response, split the way llm-prices bills them."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0

    @property
    def cache_write(self) -> int:
        return self.cache_write_5m + self.cache_write_1h

    def __add__(self, other: "Tokens") -> "Tokens":
        return Tokens(
            self.input + other.input,
            self.output + other.output,
            self.cache_read + other.cache_read,
            self.cache_write_5m + other.cache_write_5m,
            self.cache_write_1h + other.cache_write_1h,
        )


def tokens_from_usage(usage: Any) -> Optional[Tokens]:
    """Reads the ``usage`` of a response, or ``None`` when it holds no token counts.

    Understands OpenAI Chat Completions and embeddings (``prompt_tokens``), the OpenAI Responses
    API (``input_tokens`` with ``input_tokens_details``), Anthropic Messages (``input_tokens``
    plus ``cache_read_input_tokens`` and ``cache_creation``), LiteLLM's ``Usage`` and the
    google-genai ``usage_metadata`` fields. In the OpenAI and LiteLLM shapes the prompt count
    includes cached tokens, so they are taken out of ``input``.
    """
    if usage is None:
        return None

    if _get(usage, "prompt_tokens") is not None or _get(usage, "completion_tokens") is not None:
        prompt = _count(_get(usage, "prompt_tokens", 0))
        cached = _count(_get(_get(usage, "prompt_tokens_details"), "cached_tokens", 0))
        cached = cached or _count(_get(usage, "cache_read_input_tokens", 0))
        written = _count(_get(usage, "cache_creation_input_tokens", 0))
        return Tokens(
            input=max(prompt - cached - written, 0),
            output=_count(_get(usage, "completion_tokens", 0)),
            cache_read=cached,
            cache_write_5m=written,
        )

    if _get(usage, "input_tokens") is not None or _get(usage, "output_tokens") is not None:
        details = _get(usage, "input_tokens_details")
        if details is not None:
            # OpenAI Responses API: input_tokens includes the cached tokens.
            cached = _count(_get(details, "cached_tokens", 0))
            return Tokens(
                input=max(_count(_get(usage, "input_tokens", 0)) - cached, 0),
                output=_count(_get(usage, "output_tokens", 0)),
                cache_read=cached,
            )
        # Anthropic Messages: input_tokens excludes cache reads and writes.
        creation = _get(usage, "cache_creation")
        write_5m = _count(_get(creation, "ephemeral_5m_input_tokens", 0))
        write_1h = _count(_get(creation, "ephemeral_1h_input_tokens", 0))
        if write_5m + write_1h == 0:
            write_5m = _count(_get(usage, "cache_creation_input_tokens", 0))
        return Tokens(
            input=_count(_get(usage, "input_tokens", 0)),
            output=_count(_get(usage, "output_tokens", 0)),
            cache_read=_count(_get(usage, "cache_read_input_tokens", 0)),
            cache_write_5m=write_5m,
            cache_write_1h=write_1h,
        )

    if _get(usage, "prompt_token_count") is not None:
        # google-genai usage_metadata
        cached = _count(_get(usage, "cached_content_token_count", 0))
        return Tokens(
            input=max(_count(_get(usage, "prompt_token_count", 0)) - cached, 0),
            output=_count(_get(usage, "candidates_token_count", 0))
            + _count(_get(usage, "thoughts_token_count", 0)),
            cache_read=cached,
        )

    return None


# Hosts whose prices are known, matched against the client's base_url. Anything else (a local
# server, a proxy, a mock) is priced by model id alone.
_EXACT_HOSTS = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "api.deepseek.com": "deepseek",
    "api.groq.com": "groq",
    "api.x.ai": "xai",
    "api.together.xyz": "together",
    "api.together.ai": "together",
    "api.fireworks.ai": "fireworks",
    "openrouter.ai": "openrouter",
    "api.mistral.ai": "mistral",
    "generativelanguage.googleapis.com": "google",
    "api.cohere.ai": "cohere",
    "api.cohere.com": "cohere",
}
_HOST_SUFFIXES = (
    (".openai.azure.com", "azure"),
    (".cognitiveservices.azure.com", "azure"),
    (".services.ai.azure.com", "azure"),
    ("aiplatform.googleapis.com", "vertex"),
)


def provider_from_url(url: Any) -> Optional[str]:
    """The llm-prices provider name for an API base URL, or ``None`` for an unknown host."""
    try:
        host = (urlsplit(str(url)).hostname or "").lower()
    except ValueError:
        return None
    if not host:
        return None
    if host in _EXACT_HOSTS:
        return _EXACT_HOSTS[host]
    if host.startswith("bedrock-runtime.") and host.endswith(".amazonaws.com"):
        return "bedrock"
    for suffix, provider in _HOST_SUFFIXES:
        if host.endswith(suffix):
            return provider
    return None


def find_price(provider: Optional[str], model: str) -> Optional[llm_prices.Price]:
    """The llm-prices row for ``model``.

    With a known provider only that provider's rows (and, for Bedrock, Vertex and Azure, the
    list price of the model maker they resell) are searched, so a request served by Groq is
    never priced at OpenAI's rate. A provider name llm-prices does not know (``ollama``, a
    custom LiteLLM provider) means no price. Without a provider the model id is looked up
    across the whole table.
    """
    if not model:
        return None
    if provider:
        if llm_prices.Provider.parse(provider) is None:
            return None
        return llm_prices.lookup_provider(provider, model)
    return llm_prices.lookup(model)


def cost_of(price: Optional[llm_prices.Price], tokens: Tokens) -> Optional[Decimal]:
    """Exact USD cost of one response, or ``None`` when a needed price is not listed."""
    if price is None:
        return None
    usage = llm_prices.Usage(
        input=tokens.input,
        output=tokens.output,
        cache_read=tokens.cache_read,
        cache_write_5m=tokens.cache_write_5m,
        cache_write_1h=tokens.cache_write_1h,
    )
    breakdown = price.cost_usage(usage, exact=True)
    if breakdown is not None:
        return breakdown.total
    # Embedding rows list an input price only; a response with no output tokens is still priced.
    if tokens.output == 0 and tokens.cache_read == 0 and tokens.cache_write == 0:
        rate = price.rate_decimal("input")
        if rate is not None:
            return rate * tokens.input / _MILLION
    return None


@dataclass(frozen=True)
class Call:
    """One LLM response seen during the run."""

    seq: int
    test: str
    source: str
    provider: Optional[str]
    model: str
    tokens: Optional[Tokens]
    cost: Optional[Decimal]
    priced_as: Optional[str] = None

    @property
    def usd(self) -> Optional[float]:
        """Cost in USD as a float, ``None`` when unpriced."""
        return None if self.cost is None else float(self.cost)

    @property
    def priced(self) -> bool:
        return self.cost is not None

    @property
    def input_tokens(self) -> int:
        """Prompt tokens including cache reads and writes."""
        t = self.tokens
        return 0 if t is None else t.input + t.cache_read + t.cache_write

    @property
    def output_tokens(self) -> int:
        return 0 if self.tokens is None else self.tokens.output

    def to_dict(self) -> Dict[str, Any]:
        t = self.tokens
        return {
            "test": self.test,
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": None if t is None else t.input,
            "output_tokens": None if t is None else t.output,
            "cache_read_tokens": None if t is None else t.cache_read,
            "cache_write_5m_tokens": None if t is None else t.cache_write_5m,
            "cache_write_1h_tokens": None if t is None else t.cache_write_1h,
            "usd": None if self.cost is None else str(self.cost),
            "priced_as": self.priced_as,
        }

    @classmethod
    def from_dict(cls, seq: int, data: Mapping[str, Any]) -> "Call":
        tokens = None
        if data.get("input_tokens") is not None:
            tokens = Tokens(
                input=int(data["input_tokens"]),
                output=int(data["output_tokens"] or 0),
                cache_read=int(data.get("cache_read_tokens") or 0),
                cache_write_5m=int(data.get("cache_write_5m_tokens") or 0),
                cache_write_1h=int(data.get("cache_write_1h_tokens") or 0),
            )
        usd = data.get("usd")
        return cls(
            seq=seq,
            test=str(data.get("test") or ""),
            source=str(data.get("source") or ""),
            provider=data.get("provider"),
            model=str(data.get("model") or ""),
            tokens=tokens,
            cost=None if usd is None else Decimal(usd),
            priced_as=data.get("priced_as"),
        )


class Recorder:
    """The call log of one pytest session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = itertools.count()
        self.calls: List[Call] = []
        self.current_test = ""

    def record(
        self,
        *,
        source: str,
        model: Optional[str],
        usage: Any = None,
        tokens: Optional[Tokens] = None,
        provider: Optional[str] = None,
        test: Optional[str] = None,
    ) -> Call:
        """Prices one response and adds it to the log."""
        if tokens is None:
            tokens = tokens_from_usage(usage)
        model = str(model or "")
        price = cost = None
        if tokens is not None:
            price = find_price(provider, model)
            cost = cost_of(price, tokens)
        call = Call(
            seq=next(self._seq),
            test=self.current_test if test is None else test,
            source=source,
            provider=provider,
            model=model,
            tokens=tokens,
            cost=cost,
            priced_as=None if cost is None or price is None else f"{price.provider}/{price.model}",
        )
        with self._lock:
            self.calls.append(call)
        return call

    def extend_from_dicts(self, rows: List[Mapping[str, Any]]) -> None:
        with self._lock:
            for row in rows:
                self.calls.append(Call.from_dict(next(self._seq), row))

    def snapshot(self) -> List[Call]:
        with self._lock:
            return list(self.calls)


# The recorders of the running sessions, innermost last (pytester runs sessions inside a test).
_ACTIVE: List[Recorder] = []


def push(recorder: Recorder) -> None:
    _ACTIVE.append(recorder)


def pop(recorder: Recorder) -> None:
    if recorder in _ACTIVE:
        _ACTIVE.remove(recorder)


def active() -> Optional[Recorder]:
    return _ACTIVE[-1] if _ACTIVE else None


def total(calls: Iterator[Call]) -> Decimal:
    return sum((c.cost for c in calls if c.cost is not None), Decimal(0))
