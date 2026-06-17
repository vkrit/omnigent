"""
Mock LLM server with controllable response sequences for tests.

Implements the OpenAI Responses API streaming format. Supports
pre-configured response sequences (text, tool calls, errors),
per-request blocking gates, and request capture for assertions.

Endpoints:

- ``POST /v1/responses`` — consume the next queued response and
  return it as a streaming SSE body.
- ``POST /mock/configure`` — load a sequence of responses.
- ``POST /mock/reset`` — clear all state (responses, gates, requests).
- ``GET /mock/requests`` — return captured request bodies.
- ``GET /gate/pending`` — returns ``{"pending": true}`` when a
  request is waiting on a gate.
- ``POST /gate/release`` — release the next pending gate.
- ``GET /stats`` — returns ``{"request_count": N}``.

Usage::

    python tests/server/integration/mock_llm_server.py 9999

Configuration via ``POST /mock/configure``::

    {
        "responses": [
            {"text": "Hello!"},
            {"text": "World!", "block": true},
            {
                "tool_calls": [
                    {"call_id": "c1", "name": "grep", "arguments": "{}"}
                ]
            },
            {"error": "rate limit exceeded", "status_code": 429}
        ]
    }
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()


# ── SSE event builders (following Codex pattern) ─────────


def _response_id() -> str:
    """Generate a unique response id."""
    import uuid

    return f"resp_{uuid.uuid4().hex[:12]}"


def sse_text_response(text: str, model: str = "mock-model") -> str:
    """
    Build a complete SSE stream for a simple text response.

    Emits the full sequence of events the OpenAI Agents SDK expects:
    ``response.created``, ``response.output_item.added``,
    ``response.output_text.done``, ``response.output_item.done``,
    ``response.completed``.

    :param text: The assistant response text.
    :param model: Model name to include in the response.
    :returns: SSE-formatted string.
    """
    import time as _time

    resp_id = _response_id()
    msg_id = f"msg_{resp_id}"
    output_tokens = max(5, len(text.split()))
    now = _time.time()

    message_item = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text}],
    }
    response_obj = {
        "id": resp_id,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [message_item],
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "usage": {
            "input_tokens": 10,
            "output_tokens": output_tokens,
            "total_tokens": 10 + output_tokens,
        },
        "created_at": now,
        "completed_at": now,
    }
    created_response = {**response_obj, "status": "in_progress", "output": []}

    seq = 0
    events: list[str] = []

    def _add(evt_type: str, **extra: object) -> None:
        nonlocal seq
        data = {"type": evt_type, "sequence_number": seq, **extra}
        events.append(f"event: {evt_type}\ndata: {json.dumps(data)}\n\n")
        seq += 1

    _add("response.created", response=created_response)
    _add("response.output_item.added", output_index=0, item=message_item)
    _add(
        "response.output_text.done",
        output_index=0,
        item_id=msg_id,
        content_index=0,
        text=text,
    )
    _add("response.output_item.done", output_index=0, item=message_item)
    _add("response.completed", response=response_obj)
    return "".join(events)


def sse_tool_call_response(
    tool_calls: list[dict[str, str]],
    model: str = "mock-model",
) -> str:
    """
    Build a complete SSE stream for a function call response.

    :param tool_calls: List of tool call dicts, each with
        ``"call_id"``, ``"name"``, and ``"arguments"`` keys.
    :param model: Model name to include in the response.
    :returns: SSE-formatted string.
    """
    import time as _time

    resp_id = _response_id()
    now = _time.time()
    output = []
    for tc in tool_calls:
        output.append(
            {
                "id": tc.get("call_id", "call-mock"),
                "type": "function_call",
                "call_id": tc.get("call_id", "call-mock"),
                "name": tc["name"],
                "arguments": tc.get("arguments", "{}"),
                "status": "completed",
            }
        )
    response_obj = {
        "id": resp_id,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
        "created_at": now,
        "completed_at": now,
    }
    created_response = {**response_obj, "status": "in_progress", "output": []}

    seq = 0
    events: list[str] = []

    def _add(evt_type: str, **extra: object) -> None:
        nonlocal seq
        data = {"type": evt_type, "sequence_number": seq, **extra}
        events.append(f"event: {evt_type}\ndata: {json.dumps(data)}\n\n")
        seq += 1

    _add("response.created", response=created_response)
    for idx, item in enumerate(output):
        _add("response.output_item.added", output_index=idx, item=item)
        _add("response.output_item.done", output_index=idx, item=item)
    _add("response.completed", response=response_obj)
    return "".join(events)


def sse_streaming_text(text: str, model: str = "mock-model") -> str:
    """
    Build SSE with text deltas followed by a completed event.

    :param text: The assistant response text.
    :param model: Model name.
    :returns: SSE-formatted string with delta events.
    """
    events = []
    for word in text.split():
        delta = {"delta": word + " "}
        events.append(f"event: response.output_text.delta\ndata: {json.dumps(delta)}\n\n")
    events.append(sse_text_response(text, model))
    return "".join(events)


# ── Response queue state ─────────────────────────────────


@dataclass
class QueuedResponse:
    """A single pre-configured response in the queue.

    :param text: Response text (for text responses).
    :param tool_calls: Tool call list (for function call responses).
    :param block: If True, block until gate is released before responding.
    :param stream: If True, stream text deltas before completed event.
    :param error: If set, return an error response with this message.
    :param status_code: HTTP status code for error responses (default 500).
    """

    text: str = "Mock LLM response"
    tool_calls: list[dict[str, str]] | None = None
    block: bool = False
    stream: bool = False
    error: str | None = None
    status_code: int = 500
    # Internal: set when this response is waiting on a gate
    _gate: asyncio.Event = field(default_factory=asyncio.Event)
    _pending: asyncio.Event = field(default_factory=asyncio.Event)


class MockState:
    """Mutable server state for response queue and request capture."""

    def __init__(self) -> None:
        self.responses: list[QueuedResponse] = []
        self.response_index: int = 0
        self.captured_requests: list[dict] = []
        self.request_count: int = 0
        # Legacy single-gate mode (backwards compatible)
        self.pending_gates: list[QueuedResponse] = []

    def reset(self) -> None:
        """Clear all state."""
        # Release any pending gates first
        for qr in self.pending_gates:
            qr._gate.set()
        self.responses = []
        self.response_index = 0
        self.captured_requests = []
        self.request_count = 0
        self.pending_gates = []

    def next_response(self) -> QueuedResponse:
        """Consume the next queued response, or return a default."""
        if self.response_index < len(self.responses):
            resp = self.responses[self.response_index]
            self.response_index += 1
            return resp
        return QueuedResponse()


_state = MockState()


# ── Endpoints ────────────────────────────────────────────


@app.post("/v1/responses", response_model=None)
async def create_response(request: Request) -> StreamingResponse | JSONResponse:
    """
    Accept an LLM request, optionally block on gate, then return SSE.

    Consumes the next queued response from the sequence configured
    via ``POST /mock/configure``. Falls back to a default text
    response if the queue is exhausted.
    """
    _state.request_count += 1
    body = await request.body()
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        parsed = {"raw": body.decode(errors="replace")}
    _state.captured_requests.append(parsed)

    qr = _state.next_response()

    # Error response
    if qr.error is not None:
        return JSONResponse(
            status_code=qr.status_code,
            content={"error": {"message": qr.error, "type": "mock_error"}},
        )

    # Block on gate if configured
    if qr.block:
        qr._pending.set()
        _state.pending_gates.append(qr)
        await qr._gate.wait()

    # Build SSE body
    if qr.tool_calls:
        sse_body = sse_tool_call_response(qr.tool_calls)
    elif qr.stream:
        sse_body = sse_streaming_text(qr.text)
    else:
        sse_body = sse_text_response(qr.text)

    async def _generate() -> AsyncIterator[str]:
        yield sse_body

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
    )


@app.post("/mock/configure")
async def configure(request: Request) -> dict[str, object]:
    """
    Load a sequence of responses.

    Body: ``{"responses": [{"text": "...", "block": false, ...}, ...]}``
    """
    body = await request.json()
    _state.reset()
    for entry in body.get("responses", []):
        _state.responses.append(
            QueuedResponse(
                text=entry.get("text", "Mock LLM response"),
                tool_calls=entry.get("tool_calls"),
                block=entry.get("block", False),
                stream=entry.get("stream", False),
                error=entry.get("error"),
                status_code=entry.get("status_code", 500),
            )
        )
    return {"configured": True, "count": len(_state.responses)}


@app.post("/mock/reset")
async def reset() -> dict[str, bool]:
    """Clear all state."""
    _state.reset()
    return {"reset": True}


@app.get("/mock/requests")
async def get_requests() -> dict[str, list]:
    """Return all captured request bodies."""
    return {"requests": _state.captured_requests}


@app.get("/gate/pending")
async def gate_pending() -> dict[str, bool]:
    """Check if any request is waiting on a gate."""
    pending = any(qr._pending.is_set() and not qr._gate.is_set() for qr in _state.pending_gates)
    return {"pending": pending}


@app.post("/gate/release")
async def gate_release() -> dict[str, bool]:
    """Release the oldest pending gate."""
    for qr in _state.pending_gates:
        if qr._pending.is_set() and not qr._gate.is_set():
            qr._gate.set()
            return {"released": True}
    return {"released": False}


@app.get("/stats")
async def stats() -> dict[str, int]:
    """Return the total number of LLM requests received."""
    return {"request_count": _state.request_count}


if __name__ == "__main__":
    port = int(sys.argv[1])
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
