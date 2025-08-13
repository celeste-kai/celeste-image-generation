import asyncio
from typing import Any, List

import aiohttp
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.capability import Capability
from celeste_core.models.registry import supports


class LumaImageGenerator(BaseImageGenerator):
    """Luma Labs Dream Machine image generator."""

    def __init__(self, model: str = "photon-1", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = settings.luma.api_key
        if not self.api_key:
            raise ValueError(
                "Luma AI API key not provided. Set LUMA_API_KEY environment variable."
            )

        self.model_name = model
        self.base_url = "https://api.lumalabs.ai/dream-machine/v1"
        if not supports(self.model_name, Capability.IMAGE_GENERATION):
            raise ValueError(
                f"Model '{self.model_name}' does not support IMAGE_GENERATION"
            )

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """
        Generate images using Luma's Dream Machine API.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build request data
        data = {"prompt": prompt, "model": self.model_name}

        # Add aspect ratio if provided
        if "aspect_ratio" in kwargs:
            data["aspect_ratio"] = kwargs["aspect_ratio"]
        else:
            data["aspect_ratio"] = "16:9"  # Default

        # Handle different reference types
        if "image_ref" in kwargs:
            data["image_ref"] = kwargs["image_ref"]
        if "style_ref" in kwargs:
            data["style_ref"] = kwargs["style_ref"]
        if "character_ref" in kwargs:
            data["character_ref"] = kwargs["character_ref"]
        if "modify_image_ref" in kwargs:
            data["modify_image_ref"] = kwargs["modify_image_ref"]

        async with aiohttp.ClientSession() as session:
            # Create generation
            async with session.post(
                f"{self.base_url}/generations/image",
                headers=headers,
                json=data,
            ) as response:
                if response.status != 201:
                    error_text = await response.text()
                    raise Exception(f"Luma API error ({response.status}): {error_text}")

                generation_data = await response.json()
                generation_id = generation_data["id"]

            # Poll for completion
            max_attempts = 60  # 5 minutes max wait
            poll_interval = 5  # seconds

            for _attempt in range(max_attempts):
                await asyncio.sleep(poll_interval)

                # Check generation status
                async with session.get(
                    f"{self.base_url}/generations/{generation_id}", headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(
                            f"Luma API error checking status "
                            f"({response.status}): {error_text}"
                        )

                    status_data = await response.json()
                    state = status_data.get("state")

                    if state == "completed":
                        # Get the image URL
                        image_url = status_data.get("assets", {}).get("image")
                        if not image_url:
                            raise Exception(
                                "No image URL found in completed generation"
                            )

                        # Download the image
                        async with session.get(image_url) as img_response:
                            if img_response.status != 200:
                                raise Exception(
                                    f"Failed to download image from {image_url}"
                                )

                            image_bytes = await img_response.read()

                            return [
                                ImageArtifact(
                                    data=image_bytes,
                                    metadata={
                                        "model": self.model_name,
                                        "generation_id": generation_id,
                                        "aspect_ratio": data.get("aspect_ratio"),
                                        "created_at": status_data.get("created_at"),
                                        "provider": "luma",
                                    },
                                )
                            ]

                    elif state == "failed":
                        failure_reason = status_data.get(
                            "failure_reason", "Unknown error"
                        )
                        raise Exception(f"Image generation failed: {failure_reason}")

                    # Still processing, continue polling

            raise TimeoutError(
                f"Image generation timed out after "
                f"{max_attempts * poll_interval} seconds"
            )
