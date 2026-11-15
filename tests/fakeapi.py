"""A local stand-in for the OpenAI and Anthropic HTTP APIs, so the real SDKs run end to end."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Fixed token counts every fake response reports.
PROMPT = 1000
CACHED = 200
COMPLETION = 300


def _chat(model: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": PROMPT,
            "completion_tokens": COMPLETION,
            "total_tokens": PROMPT + COMPLETION,
            "prompt_tokens_details": {"cached_tokens": CACHED},
        },
    }


def _chunk(model: str, content=None, usage=None, finish=None) -> dict:
    choices = (
        [] if usage is not None else [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]
    )
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": choices,
        "usage": usage,
    }


def _responses(model: str) -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": PROMPT,
            "input_tokens_details": {"cached_tokens": CACHED},
            "output_tokens": COMPLETION,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": PROMPT + COMPLETION,
        },
    }


def _message(model: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": PROMPT,
            "output_tokens": COMPLETION,
            "cache_read_input_tokens": CACHED,
            "cache_creation_input_tokens": 0,
        },
    }


def _message_events(model: str) -> list:
    start = dict(_message(model), content=[], stop_reason=None)
    start["usage"] = dict(start["usage"], output_tokens=1)
    return [
        ("message_start", {"type": "message_start", "message": start}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": COMPLETION},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet
        pass

    def _json(self, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, events: list) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for name, data in events:
            if name:
                self.wfile.write(f"event: {name}\n".encode())
            payload = data if isinstance(data, str) else json.dumps(data)
            self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()
        self.close_connection = True

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        model = body.get("model", "")
        path = self.path.split("?")[0].rstrip("/")
        stream = bool(body.get("stream"))
        if path.endswith("/chat/completions"):
            if not stream:
                return self._json(_chat(model))
            events = [(None, _chunk(model, "o")), (None, _chunk(model, "k", finish="stop"))]
            if (body.get("stream_options") or {}).get("include_usage"):
                events.append((None, _chunk(model, usage=_chat(model)["usage"])))
            events.append((None, "[DONE]"))
            return self._sse(events)
        if path.endswith("/responses"):
            return self._json(_responses(model))
        if path.endswith("/embeddings"):
            return self._json(
                {
                    "object": "list",
                    "model": model,
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 1.0]}],
                    "usage": {"prompt_tokens": PROMPT, "total_tokens": PROMPT},
                }
            )
        if path.endswith("/messages"):
            if stream:
                return self._sse(_message_events(model))
            return self._json(_message(model))
        self.send_response(404)
        self.end_headers()


class FakeAPI:
    def __init__(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "FakeAPI":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
