"""Client for chat, vision and embedding models.

Calls go through the LiteLLM proxy, which speaks the OpenAI API. The services
name a role (``chat``, ``judge``, ``vision``, ``embed``); which model plays it,
hosted or local, and what it falls back to is the proxy's configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from openai import AsyncOpenAI, omit
from openai.types.chat import ChatCompletionMessageParam

from .config import Settings, get_settings

# Chat messages are plain ``{"role": ..., "content": ...}`` dicts at the call
# sites; we cast to the OpenAI param type when handing them to the SDK.
Message = dict[str, str]


def _response_format(schema: dict[str, Any] | None) -> Any:
    """OpenAI's structured-output envelope for a JSON schema, or nothing."""
    if schema is None:
        return None
    name = str(schema.get("title", "reply"))
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


# Base64 prefixes of the image formats a screenshot arrives in.
_IMAGE_TYPES = {
    "iVBOR": "image/png",
    "/9j/": "image/jpeg",
    "R0lGO": "image/gif",
    "UklGR": "image/webp",
}


def _image_mime(image_base64: str) -> str:
    for prefix, mime in _IMAGE_TYPES.items():
        if image_base64.startswith(prefix):
            return mime
    return "image/png"


class ModelGateway:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = AsyncOpenAI(
            base_url=self.settings.model_gateway_url.rstrip("/") + "/v1",
            api_key=self.settings.model_gateway_api_key or "not-needed",
            max_retries=2,
            timeout=60.0,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self.settings.embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        model: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Chat and return the reply text.

        With ``json_schema`` the provider is asked for a reply matching that
        schema. Providers that cannot do it get the plain prompt (LiteLLM drops
        the parameter), so callers still parse defensively.
        """
        return await self._complete(
            messages,
            model=model or self.settings.chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
        )

    async def describe_image(
        self,
        instructions: str,
        image_base64: str,
        *,
        max_tokens: int,
        model: str | None = None,
    ) -> str:
        """Chat with an image attached, for a vision-capable model.

        No ``json_schema`` here: the vision providers reject ``response_format``.
        """
        image_url = f"data:{_image_mime(image_base64)};base64,{image_base64}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this screenshot."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        return await self._complete(
            messages,
            model=model or self.settings.vision_model,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    async def _complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=_response_format(json_schema) or omit,
        )
        return response.choices[0].message.content or ""

    async def aclose(self) -> None:
        await self._client.close()
