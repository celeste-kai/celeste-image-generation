import base64
from typing import Any, List

import aiohttp
from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.capability import Capability
from celeste_core.enums.providers import Provider
from celeste_core.models.registry import supports


class OpenAIImageGenerator(BaseImageGenerator):
    """OpenAI image generator using DALL-E models."""

    def __init__(self, model: str = "dall-e-3", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = settings.openai.api_key
        self.model_name = model
        self.base_url = "https://api.openai.com/v1"
        # Non-raising validation; store support state for callers to inspect
        self.is_supported = supports(
            Provider.OPENAI, self.model_name, Capability.IMAGE_GENERATION
        )

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """
        Generate images using OpenAI's image generation API.
        """
        # Set defaults based on model
        if self.model_name == "dall-e-3":
            size = kwargs.get("size", "1024x1024")
            quality = kwargs.get("quality", "standard")
            style = kwargs.get("style", "vivid")
            n = 1  # DALL-E 3 only supports n=1
        else:  # DALL-E 2
            size = kwargs.get("size", "1024x1024")
            quality = None
            style = None
            n = kwargs.get("n", 1)

        response_format = kwargs.get("response_format", "b64_json")

        # Build request
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "size": size,
            "n": n,
        }

        # Only add response_format for DALL-E models (not gpt-image-1)
        if self.model_name in ["dall-e-2", "dall-e-3"]:
            data["response_format"] = response_format

        # Add DALL-E 3 specific parameters
        if self.model_name == "dall-e-3":
            if quality:
                data["quality"] = quality
            if style:
                data["style"] = style

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/images/generations", json=data, headers=headers
            ) as response:
                if response.status != 200:
                    # Non-raising: return empty list on error
                    return []

                result = await response.json()

                images: List[ImageArtifact] = []
                for img_data in result["data"]:
                    # Inline decode (b64_json or url).
                    # Also support alt field 'image' used by some models
                    image_bytes: bytes
                    if "b64_json" in img_data:
                        image_bytes = base64.b64decode(img_data["b64_json"])
                    elif "url" in img_data:
                        async with session.get(img_data["url"]) as img_response:
                            if img_response.status != 200:
                                continue
                            image_bytes = await img_response.read()
                    elif "image" in img_data:
                        # Some endpoints return 'image' which may be
                        # a URL or a base64 string
                        val = img_data["image"]
                        if isinstance(val, str) and val.startswith("http"):
                            async with session.get(val) as img_response:
                                if img_response.status != 200:
                                    raise Exception(
                                        f"Failed to download image from URL: {val}"
                                    )
                                image_bytes = await img_response.read()
                        elif isinstance(val, str):
                            image_bytes = base64.b64decode(val)
                        else:
                            continue
                    else:
                        continue

                    # Extract revised prompt if available (DALL-E 3 feature)
                    metadata = {
                        "model": self.model_name,
                        "size": size,
                    }
                    if "revised_prompt" in img_data:
                        metadata["revised_prompt"] = img_data["revised_prompt"]
                    if quality:
                        metadata["quality"] = quality
                    if style:
                        metadata["style"] = style

                    images.append(ImageArtifact(data=image_bytes, metadata=metadata))

                return images
