"""Hooks into the OpenAI, Anthropic and LiteLLM SDKs.

The OpenAI and Anthropic SDKs send every API call through ``SyncAPIClient.request`` and
``AsyncAPIClient.request`` in their ``_base_client`` module, so wrapping those two methods sees
``create``, ``parse`` and the streaming helpers alike, on every client instance, however the
SDK was imported. LiteLLM is hooked at ``litellm.completion`` and ``litellm.acompletion``; while
one of those runs, the SDK hooks stand down so a call is never counted twice.

An SDK is only patched once it is imported, so a test run that never uses one pays nothing for
it. Every hook swallows its own errors: cost tracking never breaks a test's API call.
"""

from __future__ import annotations

import contextvars
import functools
import importlib
import importlib.abc
import sys
import warnings
from typing import Any, Callable, Dict, Mapping, Optional

from . import _core
from ._core import _get

_MARK = "__pytest_llm_cost_hook__"
_SUPPRESS: contextvars.ContextVar[bool] = contextvars.ContextVar("pytest_llm_cost_suppress", default=False)
_warned = False


def _warn_once(err: BaseException) -> None:
    global _warned
    if not _warned:
        _warned = True
        warnings.warn(f"pytest-llm-cost could not read an LLM response: {err!r}", stacklevel=3)


# --------------------------------------------------------------------------- streams


class _StreamUsage:
    """Collects the usage that streamed chunks or events report."""

    def __init__(self, model: Optional[str]) -> None:
        self.model = model
        self.usage: Any = None
        self._anthropic: Optional[Dict[str, Any]] = None

    def feed(self, item: Any) -> None:
        kind = _get(item, "type")
        if kind == "message_start":
            message = _get(item, "message")
            self.model = _get(message, "model", self.model)
            self._anthropic = _usage_dict(_get(message, "usage"))
            return
        if kind == "message_delta":
            merged = self._anthropic if self._anthropic is not None else {}
            for key, value in _usage_dict(_get(item, "usage")).items():
                if value is not None:
                    merged[key] = value
            self._anthropic = merged
            return
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            response = _get(item, "response")
            self.model = _get(response, "model", self.model)
            self.usage = _get(response, "usage", self.usage)
            return
        usage = _get(item, "usage")
        if usage is not None:
            self.usage = usage
            self.model = _get(item, "model", self.model)

    def final_usage(self) -> Any:
        if self._anthropic is not None:
            return self._anthropic
        return self.usage


_ANTHROPIC_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation",
)


def _usage_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    return {key: _get(usage, key) for key in _ANTHROPIC_USAGE_FIELDS}


def _wrap_stream(
    stream: Any, is_async: bool, on_done: Callable[[_StreamUsage], None], model: Optional[str]
) -> None:
    """Replaces the stream's item iterator with one that watches for usage."""
    inner = stream._iterator  # the SDKs' Stream and AsyncStream both iterate this
    state = _StreamUsage(model)
    done = [False]

    def finish() -> None:
        if not done[0]:
            done[0] = True
            try:
                on_done(state)
            except Exception as err:  # pragma: no cover - defensive
                _warn_once(err)

    if is_async:

        async def agen():  # type: ignore[no-untyped-def]
            try:
                async for item in inner:
                    try:
                        state.feed(item)
                    except Exception as err:  # pragma: no cover - defensive
                        _warn_once(err)
                    yield item
            finally:
                finish()

        stream._iterator = agen()
    else:

        def gen():  # type: ignore[no-untyped-def]
            try:
                for item in inner:
                    try:
                        state.feed(item)
                    except Exception as err:  # pragma: no cover - defensive
                        _warn_once(err)
                    yield item
            finally:
                finish()

        stream._iterator = gen()


# --------------------------------------------------------------------------- OpenAI / Anthropic

_GENERATION_PATHS = ("/chat/completions", "/completions", "/responses", "/messages", "/embeddings")


def _request_model(options: Any) -> Optional[str]:
    body = getattr(options, "json_data", None)
    if isinstance(body, Mapping):
        model = body.get("model")
        if isinstance(model, str):
            return model
    return None


def _after_request(source: str, client: Any, options: Any, result: Any, stream: bool, is_async: bool) -> None:
    recorder = _core.active()
    if recorder is None or _SUPPRESS.get():
        return
    provider = _core.provider_from_url(getattr(client, "base_url", ""))
    test = recorder.current_test
    if stream:
        url = str(getattr(options, "url", ""))
        if not hasattr(result, "_iterator") or not url.rstrip("/").endswith(_GENERATION_PATHS):
            return

        def on_done(state: _StreamUsage) -> None:
            recorder.record(
                source=source,
                provider=provider,
                model=state.model,
                usage=state.final_usage(),
                test=test,
            )

        _wrap_stream(result, is_async, on_done, _request_model(options))
        return
    usage = _get(result, "usage")
    if usage is None:
        return  # raw responses, list endpoints, files and anything else without token counts
    recorder.record(
        source=source,
        provider=provider,
        model=_get(result, "model") or _request_model(options),
        usage=usage,
        test=test,
    )


def _patch_stainless(package: str) -> None:
    base = importlib.import_module(package + "._base_client")
    for class_name, is_async in (("SyncAPIClient", False), ("AsyncAPIClient", True)):
        cls = getattr(base, class_name, None)
        original = None if cls is None else cls.__dict__.get("request")
        if original is None or getattr(original, _MARK, False):
            continue
        cls.request = _request_wrapper(package, original, is_async)


def _request_wrapper(source: str, original: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
    if is_async:

        @functools.wraps(original)
        async def request(self: Any, cast_to: Any, options: Any, *args: Any, **kwargs: Any) -> Any:
            result = await original(self, cast_to, options, *args, **kwargs)
            try:
                _after_request(source, self, options, result, bool(kwargs.get("stream")), True)
            except Exception as err:
                _warn_once(err)
            return result

    else:

        @functools.wraps(original)
        def request(self: Any, cast_to: Any, options: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(self, cast_to, options, *args, **kwargs)
            try:
                _after_request(source, self, options, result, bool(kwargs.get("stream")), False)
            except Exception as err:
                _warn_once(err)
            return result

    setattr(request, _MARK, True)
    return request


# --------------------------------------------------------------------------- LiteLLM


def _after_litellm(result: Any, kwargs: Mapping[str, Any]) -> None:
    recorder = _core.active()
    if recorder is None or _SUPPRESS.get():
        return
    usage = _get(result, "usage")
    if usage is None:
        return
    hidden = _get(result, "_hidden_params", {}) or {}
    provider = _get(hidden, "custom_llm_provider")
    model = _get(hidden, "litellm_model_name") or _get(result, "model") or kwargs.get("model")
    recorder.record(source="litellm", provider=provider, model=model, usage=usage)


def _patch_litellm(module: Any) -> None:
    for name, is_async in (("completion", False), ("acompletion", True)):
        original = getattr(module, name, None)
        if original is None or getattr(original, _MARK, False):
            continue
        setattr(module, name, _litellm_wrapper(original, is_async))


def _litellm_wrapper(original: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
    # A streamed LiteLLM call is left to the SDK hooks underneath it, which see the final usage.
    if is_async:

        @functools.wraps(original)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return await original(*args, **kwargs)
            token = _SUPPRESS.set(True)
            try:
                result = await original(*args, **kwargs)
            finally:
                _SUPPRESS.reset(token)
            try:
                _after_litellm(result, kwargs)
            except Exception as err:
                _warn_once(err)
            return result

    else:

        @functools.wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return original(*args, **kwargs)
            token = _SUPPRESS.set(True)
            try:
                result = original(*args, **kwargs)
            finally:
                _SUPPRESS.reset(token)
            try:
                _after_litellm(result, kwargs)
            except Exception as err:
                _warn_once(err)
            return result

    setattr(wrapper, _MARK, True)
    return wrapper


# --------------------------------------------------------------------------- install

_PATCHES: Dict[str, Callable[[Any], None]] = {
    "openai": lambda module: _patch_stainless("openai"),
    "anthropic": lambda module: _patch_stainless("anthropic"),
    "litellm": _patch_litellm,
}
_pending: Dict[str, Callable[[Any], None]] = {}


def _apply(name: str, module: Any) -> None:
    try:
        _PATCHES[name](module)
    except Exception as err:
        warnings.warn(f"pytest-llm-cost could not hook {name}: {err!r}", stacklevel=2)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner: Any, name: str) -> None:
        self._inner = inner
        self._name = name

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        _apply(self._name, module)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._inner, attr)


class _Finder(importlib.abc.MetaPathFinder):
    """Patches an SDK right after its first import."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname not in _pending:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        _pending.pop(fullname, None)
        spec.loader = _Loader(spec.loader, fullname)
        return spec


_finder = _Finder()


def install() -> None:
    """Hooks the SDKs already imported and arranges to hook the rest when they are."""
    for name in _PATCHES:
        module = sys.modules.get(name)
        if module is not None:
            _pending.pop(name, None)
            _apply(name, module)
        else:
            _pending[name] = _PATCHES[name]
    if _pending and _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)
