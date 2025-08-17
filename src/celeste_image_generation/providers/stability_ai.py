import base64
from typing import Any, Dict, List, Tuple

import aiohttp
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.capability import Capability
from celeste_core.enums.providers import Provider
from celeste_core.models.registry import supports


class StabilityAIImageGenerator(BaseImageGenerator):
    def __init__(
        self, model: str = "stable-diffusion-xl-1024-v1-0", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = settings.stability.api_key
        self.model_name = model
        self.is_v2 = self.model_name in [
            "ultra",
            "core",
            "sd3.5-large",
            "sd3.5-large-turbo",
            "sd3.5-medium",
        ]
        self.is_raw = self.model_name in ["core", "ultra"]
        # Non-raising validation; store support state for callers to inspect
        self.is_supported = supports(
            Provider.STABILITYAI, self.model_name, Capability.IMAGE_GENERATION
        )

    def _format_request(self, prompt: str, **kwargs: Any) -> Tuple[str, Dict[str, Any]]:
        """Format the request based on API version."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*" if self.is_raw else "application/json",
        }

        if self.is_v2:
            model_path = "sd3" if self.model_name.startswith("sd3") else self.model_name
            endpoint = (
                f"https://api.stability.ai/v2beta/stable-image/generate/{model_path}"
            )
            data = aiohttp.FormData()
            data.add_field(
                "none", "", filename="", content_type="application/octet-stream"
            )
            data.add_field("prompt", prompt)
            data.add_field(
                "output_format",
                kwargs.get("output_format", "webp" if self.is_raw else "png"),
            )
            if self.model_name.startswith("sd3"):
                data.add_field("model", self.model_name)
            return endpoint, {"headers": headers, "data": data}
        else:
            endpoint = f"https://api.stability.ai/v1/generation/{self.model_name}/text-to-image"
            return endpoint, {
                "headers": headers,
                "json": {
                    "text_prompts": [{"text": prompt}],
                    "height": kwargs.get("height", 1024),
                    "width": kwargs.get("width", 1024),
                },
            }

    def _parse_response(self, data: Any) -> List[ImageArtifact]:
        """Parse response based on API version and model type."""
        if self.is_v2:
            return [
                ImageArtifact(
                    data=base64.b64decode(data["image"]),
                    metadata={"model": self.model_name, "seed": data.get("seed")},
                )
            ]
        else:
            return [
                ImageArtifact(
                    data=base64.b64decode(artifact["base64"]),
                    metadata={"model": self.model_name, "seed": artifact.get("seed")},
                )
                for artifact in data.get("artifacts", [])
            ]

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        endpoint, request_kwargs = self._format_request(prompt, **kwargs)

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, **request_kwargs) as response:
                if response.status != 200:
                    # Non-raising: return empty list on error
                    return []

                if self.is_raw:
                    return [
                        ImageArtifact(
                            data=await response.read(),
                            metadata={"model": self.model_name},
                        )
                    ]

                return self._parse_response(await response.json())
