import base64
from typing import Any, List

import aiohttp
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.capability import Capability
from celeste_core.enums.providers import Provider
from celeste_core.models.registry import supports


class XAIImageGenerator(BaseImageGenerator):
    """xAI image generator using Grok's image API."""

    def __init__(self, model: str = "grok-2-image", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = settings.xai.api_key
        # Non-raising: proceed; upstream may allow anonymous or will fail softly
        self.model = model
        self.base_url = "https://api.x.ai/v1"
        # Non-raising validation; store support state for callers to inspect
        self.is_supported = supports(
            Provider.XAI, self.model, Capability.IMAGE_GENERATION
        )

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """Generate images using xAI's image generation endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "prompt": prompt,
            "n": kwargs.get("n", 1),
            "response_format": kwargs.get("response_format", "b64_json"),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/images/generations",
                json=data,
                headers=headers,
            ) as response:
                if response.status != 200:
                    # Non-raising: return empty list on error
                    return []

                result = await response.json()
                images: List[ImageArtifact] = []

                for img_data in result.get("data", []):
                    image_bytes: bytes
                    if "b64_json" in img_data:
                        image_bytes = base64.b64decode(img_data["b64_json"])
                    elif "url" in img_data:
                        async with session.get(img_data["url"]) as img_response:
                            if img_response.status != 200:
                                continue
                            image_bytes = await img_response.read()
                    else:
                        continue

                    metadata = {
                        "model": self.model,
                        "provider": "xai",
                    }
                    if "revised_prompt" in img_data:
                        metadata["revised_prompt"] = img_data["revised_prompt"]

                    images.append(ImageArtifact(data=image_bytes, metadata=metadata))

                return images
