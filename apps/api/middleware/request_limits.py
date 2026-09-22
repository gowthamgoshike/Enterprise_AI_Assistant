"""Bound declared and chunked request bodies before route execution."""

from starlette.requests import Request

from apps.api.error_handlers import error_response


class RequestSizeLimitMiddleware:
    def __init__(self, app, *, max_request_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def _reject(self, scope, receive, send, *, code: str, message: str):
        request = Request(scope, receive=receive)
        response = error_response(
            request,
            status_code=413 if code == "request_too_large" else 400,
            code=code,
            message=message,
        )
        await response(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", ()))
        raw_content_length = headers.get(b"content-length")
        content_length = (
            raw_content_length.decode("latin-1")
            if raw_content_length is not None
            else None
        )
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                await self._reject(
                    scope,
                    receive,
                    send,
                    code="invalid_content_length",
                    message="The Content-Length header is invalid.",
                )
                return
            if size < 0:
                await self._reject(
                    scope,
                    receive,
                    send,
                    code="invalid_content_length",
                    message="The Content-Length header is invalid.",
                )
                return
            if size > self.max_request_bytes:
                await self._reject(
                    scope,
                    receive,
                    send,
                    code="request_too_large",
                    message="The request is too large.",
                )
                return

        if scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        messages = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                continue
            total += len(message.get("body", b""))
            if total > self.max_request_bytes:
                await self._reject(
                    scope,
                    receive,
                    send,
                    code="request_too_large",
                    message="The request is too large.",
                )
                return
            if not message.get("more_body", False):
                break

        async def replay_receive():
            if messages:
                return messages.pop(0)
            return await receive()

        await self.app(scope, replay_receive, send)
