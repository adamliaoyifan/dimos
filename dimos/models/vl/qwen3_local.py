# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-VL model backed by a custom FastAPI server.

The server (``/home/adamliao/Qwen3/api/api/server.py``) exposes non-OpenAI
endpoints.  This backend calls them directly via ``httpx``.

Key endpoints used:
    - ``POST /v1/image-inference-base64`` — general image+text → text
    - ``POST /v1/text-inference``        — text-only → text
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from dimos.models.vl.base import VlModel, VlModelConfig
from dimos.msgs.sensor_msgs.Image import Image

logger = logging.getLogger(__name__)


class Qwen3LocalVlModelConfig(VlModelConfig):
    """Configuration for the custom Qwen3-VL FastAPI server."""

    base_url: str = "http://192.168.2.109:8000"
    """Base URL of the Qwen3-VL server (no trailing slash, no /v1)."""

    api_key: str | None = None
    """Optional API key (passed as ``?api_key=`` query param)."""

    max_new_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9

    prompt_prefix: str = ""
    """Prefix prepended to every VLM prompt. Used for simulation context."""


class Qwen3LocalVlModel(VlModel[Qwen3LocalVlModelConfig]):
    """Qwen3-VL model backed by a custom FastAPI inference server.

    Drop-in replacement for QwenVlModel — same ``query(image, prompt)``
    interface.  Zero cloud API keys required.
    """

    default_config = Qwen3LocalVlModelConfig

    def _get_client(self):  # type: ignore[no-untyped-def]
        if "_http_client" not in self.__dict__:
            import httpx

            self.__dict__["_http_client"] = httpx.Client(timeout=120.0)
        return self.__dict__["_http_client"]

    def _params(self) -> dict[str, str]:
        """Query params (api_key if configured)."""
        if self.config.api_key:
            return {"api_key": self.config.api_key}
        return {}

    def query(self, image: Image | np.ndarray, query: str, **kwargs: Any) -> str:  # type: ignore[override]
        if isinstance(image, np.ndarray):
            image = Image.from_numpy(image)

        # Apply auto_resize if configured
        image, _ = self._prepare_image(image)

        img_base64 = image.to_base64()

        # Prepend simulation/context prefix if configured
        if self.config.prompt_prefix:
            query = self.config.prompt_prefix + query

        client = self._get_client()
        url = f"{self.config.base_url}/v1/image-inference-base64"

        payload = {
            "image_base64": img_base64,
            "prompt": query,
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        resp = client.post(url, json=payload, params=self._params())
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "success":
            error = data.get("error") or data.get("detail") or str(data)
            raise RuntimeError(f"Qwen3-VL server error: {error}")

        return data["response"]

    def query_text_only(self, prompt: str, **kwargs: Any) -> str:
        """Text-only inference (no image)."""
        client = self._get_client()
        url = f"{self.config.base_url}/v1/text-inference"

        payload = {
            "prompt": prompt,
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        resp = client.post(url, json=payload, params=self._params())
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "success":
            error = data.get("error") or data.get("detail") or str(data)
            raise RuntimeError(f"Qwen3-VL server error: {error}")

        return data["response"]

    def stop(self) -> None:
        if "_http_client" in self.__dict__:
            self.__dict__["_http_client"].close()
            del self.__dict__["_http_client"]


__all__ = ["Qwen3LocalVlModel", "Qwen3LocalVlModelConfig"]
