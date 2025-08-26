import asyncio
from typing import Any, List

import aiohttp
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.providers import Provider


class LumaImageGenerator(BaseImageGenerator):
    """Luma Labs Dream Machine image generator."""

    def __init__(self, model: str = "photon-1", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.LUMA, **kwargs)
        self.api_key = settings.luma.api_key
        self.base_url = "https://api.lumalabs.ai/dream-machine/v1"

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """Generate images using Luma's Dream Machine API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"prompt": prompt, "model": self.model, **kwargs}

        async with aiohttp.ClientSession() as session:
            # Create generation
            async with session.post(
                f"{self.base_url}/generations/image", headers=headers, json=data
            ) as response:
                response.raise_for_status()
                generation_id = (await response.json())["id"]

            # Poll for completion with exponential backoff
            max_attempts = 60
            base_delay = 2

            for attempt in range(max_attempts):
                delay = min(base_delay * (2 ** min(attempt // 5, 4)), 10)
                await asyncio.sleep(delay)
                async with session.get(
                    f"{self.base_url}/generations/{generation_id}", headers=headers
                ) as response:
                    response.raise_for_status()

                    status_data = await response.json()
                    state = status_data.get("state")

                    if state == "completed":
                        image_url = status_data.get("assets", {}).get("image")
                        if not image_url:
                            raise ValueError("No image URL in completed generation")

                        async with session.get(image_url) as img_response:
                            img_response.raise_for_status()

                            return [
                                ImageArtifact(
                                    data=await img_response.read(),
                                    metadata={
                                        "model": self.model,
                                        "generation_id": generation_id,
                                        "created_at": status_data.get("created_at"),
                                        "provider": "luma",
                                        **kwargs,
                                    },
                                )
                            ]

                    elif state == "failed":
                        failure_reason = status_data.get(
                            "failure_reason", "Unknown error"
                        )
                        raise RuntimeError(f"Image generation failed: {failure_reason}")

            raise TimeoutError("Image generation timed out after 5 minutes")
