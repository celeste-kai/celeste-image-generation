import base64
from typing import Any

import aiohttp
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.providers import Provider


class XAIImageGenerator(BaseImageGenerator):
    """xAI image generator using Grok's image API."""

    def __init__(self, model: str = "grok-2-image", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.XAI, **kwargs)
        self.api_key = settings.xai.api_key
        self.base_url = "https://api.x.ai/v1"

    async def generate_image(self, prompt: str, **kwargs: Any) -> list[ImageArtifact]:
        """Generate images using xAI's image generation endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "prompt": prompt,
            "response_format": "b64_json",  # Default to base64 for reliability
            **kwargs,
        }

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self.base_url}/images/generations",
                json=data,
                headers=headers,
            ) as response,
        ):
            response.raise_for_status()

            result = await response.json()
            images: list[ImageArtifact] = []

            for img_data in result.get("data", []):
                image_bytes = base64.b64decode(img_data["b64_json"])

                metadata = {
                    "model": self.model,
                    "provider": "xai",
                    **kwargs,
                }
                if "revised_prompt" in img_data:
                    metadata["revised_prompt"] = img_data["revised_prompt"]

                images.append(ImageArtifact(data=image_bytes, metadata=metadata))

            return images
