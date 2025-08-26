import base64
from typing import Any, List

import aiohttp
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.providers import Provider


class StabilityAIImageGenerator(BaseImageGenerator):
    def __init__(self, model: str = "core", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.STABILITYAI, **kwargs)
        self.api_key = settings.stability.api_key
        self.is_raw = self.model in ["core", "ultra"]

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """Generate images using Stability AI's v2 API."""
        endpoint = f"https://api.stability.ai/v2beta/stable-image/generate/{self.model}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*" if self.is_raw else "application/json",
        }

        data = aiohttp.FormData()
        data.add_field("none", "", filename="", content_type="application/octet-stream")
        data.add_field("prompt", prompt)
        data.add_field("model", self.model)

        # Add all kwargs as form fields
        for key, value in kwargs.items():
            data.add_field(key, str(value))

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, data=data) as response:
                response.raise_for_status()

                if self.is_raw:
                    return [
                        ImageArtifact(
                            data=await response.read(),
                            metadata={"model": self.model, **kwargs},
                        )
                    ]

                response_data = await response.json()
                return [
                    ImageArtifact(
                        data=base64.b64decode(response_data["image"]),
                        metadata={
                            "model": self.model,
                            "seed": response_data.get("seed"),
                            **kwargs,
                        },
                    )
                ]
