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
        if not self.api_key:
            raise ValueError(
                "xAI API key not provided. Set XAI_API_KEY environment variable."
            )
        self.model_name = model
        self.base_url = "https://api.x.ai/v1"
        if not supports(Provider.XAI, self.model_name, Capability.IMAGE_GENERATION):
            raise ValueError(
                f"Model '{self.model_name}' does not support IMAGE_GENERATION"
            )

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """Generate images using xAI's image generation endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_name,
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
                    error_text = await response.text()
                    raise Exception(f"xAI API error ({response.status}): {error_text}")

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
                        "model": self.model_name,
                        "provider": "xai",
                    }
                    if "revised_prompt" in img_data:
                        metadata["revised_prompt"] = img_data["revised_prompt"]

                    images.append(ImageArtifact(data=image_bytes, metadata=metadata))

                return images
