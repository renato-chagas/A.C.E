"""Optional Cloudflare Workers AI text inference for ACE."""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine._base import EngineConnectionError, InferenceEngine

CLOUDFLARE_MODEL_ID = "@cf/google/gemma-4-26b-a4b-it"
_HOST = "https://api.cloudflare.com"
_TIMEOUT = httpx.Timeout(connect=5.0, read=90.0, write=15.0, pool=5.0)
_CREDENTIALS = (
    "Set a valid CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN for Workers AI."
)
_NETWORK = "Could not reach Cloudflare Workers AI."
_RESPONSE = "Cloudflare Workers AI returned no usable text response."


class CloudflareInferenceError(EngineConnectionError):
    """A user-safe Workers AI failure and its corresponding HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@EngineRegistry.register("cloudflare")
class CloudflareEngine(InferenceEngine):
    """Use the fixed Workers AI endpoint for text completions."""

    engine_id = "cloudflare"
    is_cloud = True

    def __init__(self) -> None:
        self._account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self._token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()

    def health(self) -> bool:
        """Check local configuration without sending a billable probe."""
        return bool(
            re.fullmatch(r"[0-9a-fA-F]{32}", self._account_id)
            and re.fullmatch(r"[!-~]+", self._token)
        )

    def list_models(self) -> list[str]:
        return [CLOUDFLARE_MODEL_ID] if self.health() else []

    def can_serve(self, model: str) -> bool:
        return model == CLOUDFLARE_MODEL_ID and self.health()

    def _request(
        self,
        messages: Sequence[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        options: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if model != CLOUDFLARE_MODEL_ID:
            raise CloudflareInferenceError(
                400, "Select the supported Cloudflare Workers AI model."
            )
        if not self.health():
            raise CloudflareInferenceError(403, _CREDENTIALS)
        if (
            options.get("tools")
            or options.get("tool_choice")
            or any(
                message.role == Role.TOOL
                or message.images
                or message.tool_calls
                or message.tool_call_id
                for message in messages
            )
        ):
            raise CloudflareInferenceError(
                400, "Cloudflare Workers AI accepts text chat only."
            )
        if (
            type(temperature) not in (int, float)
            or not 0 <= temperature <= 2
            or type(max_tokens) is not int
            or max_tokens < 1
        ):
            raise CloudflareInferenceError(
                400, "Invalid Cloudflare Workers AI generation settings."
            )
        return f"/client/v4/accounts/{self._account_id}/ai/run/{CLOUDFLARE_MODEL_ID}", {
            "messages": [
                {"role": message.role.value, "content": message.text}
                for message in messages
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": False,
            "store": False,
        }

    def _client_options(self) -> dict[str, Any]:
        return {
            "base_url": _HOST,
            "headers": {"Authorization": f"Bearer {self._token}"},
            "timeout": _TIMEOUT,
            "follow_redirects": False,
            "trust_env": False,
        }

    @staticmethod
    def _read_content(response: httpx.Response) -> str:
        status = response.status_code
        if status in (401, 403):
            raise CloudflareInferenceError(
                403,
                "Cloudflare Workers AI denied access. Check the account, token "
                "permissions, and plan.",
            )
        if status in (402, 429):
            raise CloudflareInferenceError(
                429, "Cloudflare Workers AI hit a usage or rate limit."
            )
        if not response.is_success:
            raise CloudflareInferenceError(
                502, "Cloudflare Workers AI could not complete the request."
            )

        try:
            data = json.loads(response.content.decode("utf-8"))
            if data.get("success", True) is not True:
                raise ValueError
            content = data["result"]["choices"][0]["message"]["content"]
        except (
            AttributeError,
            IndexError,
            KeyError,
            RecursionError,
            TypeError,
            ValueError,
        ):
            raise CloudflareInferenceError(502, _RESPONSE) from None
        if not isinstance(content, str) or not content.strip():
            raise CloudflareInferenceError(502, _RESPONSE)
        return content

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        path, payload = self._request(messages, model, temperature, max_tokens, kwargs)
        try:
            with httpx.Client(**self._client_options()) as client:
                response = client.post(path, json=payload)
        except httpx.TimeoutException:
            raise CloudflareInferenceError(
                504, "Cloudflare Workers AI did not respond in time."
            ) from None
        except httpx.HTTPError:
            raise CloudflareInferenceError(502, _NETWORK) from None
        return {
            "content": self._read_content(response),
            "usage": {},
            "model": CLOUDFLARE_MODEL_ID,
        }

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        path, payload = self._request(messages, model, temperature, max_tokens, kwargs)
        try:
            async with httpx.AsyncClient(**self._client_options()) as client:
                response = await client.post(path, json=payload)
        except httpx.TimeoutException:
            raise CloudflareInferenceError(
                504, "Cloudflare Workers AI did not respond in time."
            ) from None
        except httpx.HTTPError:
            raise CloudflareInferenceError(502, _NETWORK) from None
        yield self._read_content(response)


__all__ = ["CLOUDFLARE_MODEL_ID", "CloudflareEngine", "CloudflareInferenceError"]
