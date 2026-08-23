"""A tiny ASGI test client.

Deliberately not `starlette.testclient`. That module needs an HTTP client library
(`httpx` on starlette 0.x, `httpx2` on 1.x) which this repository does not lock, and the
version it wants changes across the starlette 0.x/1.x boundary — so a test built on it
passes or fails depending on which starlette the resolver picked. Driving the ASGI
callable directly removes the dependency and the ambiguity, and it is about forty lines.

Only what the console's tests need: the common HTTP methods, JSON in and out,
caller-supplied headers, and access to the response body, status and headers.

It is strict about exceptions on purpose — see the note in `request`.
"""

from __future__ import annotations

import asyncio
import json as jsonlib
from urllib.parse import urlencode


class Response:
    def __init__(self, status: int, headers: list[tuple[bytes, bytes]], body: bytes):
        self.status_code = status
        self.headers = {k.decode(): v.decode() for k, v in headers}
        self._body = body

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", "replace")

    def json(self):
        return jsonlib.loads(self._body)


class ASGIClient:
    """Drives an ASGI app in-process. No sockets, no HTTP client dependency."""

    def __init__(self, app):
        self.app = app

    def get(
        self,
        path: str,
        params: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        return self.request("GET", path, params=params, headers=headers)

    def post(
        self,
        path: str,
        json: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        return self.request("POST", path, json=json, headers=headers)

    def patch(
        self,
        path: str,
        json: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        return self.request("PATCH", path, json=json, headers=headers)

    def delete(
        self,
        path: str,
        json: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        return self.request("DELETE", path, json=json, headers=headers)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        path, _, inline_query = path.partition("?")
        query = urlencode(params) if params else inline_query
        body = jsonlib.dumps(json).encode() if json is not None else b""

        encoded_headers = [(b"host", b"testserver")]
        encoded_headers.extend(
            (name.lower().encode(), value.encode())
            for name, value in (headers or {}).items()
        )
        if json is not None and not any(
            name == b"content-type" for name, _ in encoded_headers
        ):
            encoded_headers.append((b"content-type", b"application/json"))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": query.encode(),
            "headers": encoded_headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        captured: dict = {"status": 500, "headers": [], "body": b""}

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
                captured["headers"] = message.get("headers", [])
            elif message["type"] == "http.response.body":
                captured["body"] += message.get("body", b"")

        # Deliberately does NOT swallow. An earlier version caught the re-raise that
        # Starlette's ServerErrorMiddleware performs, which hid a real leak: in
        # production uvicorn logs the re-raised exception's message. The server wraps
        # the app so nothing escapes; propagating here is what proves it.
        asyncio.run(self.app(scope, receive, send))
        return Response(captured["status"], captured["headers"], captured["body"])
