import base64
from typing import Any, List

import aiohttp

from celeste_image_generation.base import BaseImageGenerator
from celeste_image_generation.core.config import XAI_API_KEY
from celeste_image_generation.core.enums import XAIModel
from celeste_image_generation.core.types import GeneratedImage, ImagePrompt


class XAIImageGenerator(BaseImageGenerator):
    """xAI image generator using Grok's image API."""

    def __init__(self, model: str = XAIModel.GROK_2_IMAGE.value, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = XAI_API_KEY
        if not self.api_key:
            raise ValueError("xAI API key not provided. Set XAI_API_KEY environment variable.")
        self.model_name = model.value if hasattr(model, "value") else model
        self.base_url = "https://api.x.ai/v1"

    async def generate_image(self, prompt: ImagePrompt, **kwargs: Any) -> List[GeneratedImage]:
        """Generate images using xAI's image generation endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_name,
            "prompt": prompt.content,
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
                images: List[GeneratedImage] = []

                for img_data in result.get("data", []):
                    if "b64_json" in img_data:
                        image_bytes = base64.b64decode(img_data["b64_json"])
                    elif "url" in img_data:
                        async with session.get(img_data["url"]) as img_response:
                            if img_response.status != 200:
                                raise Exception("Failed to download image from returned URL")
                            image_bytes = await img_response.read()
                    else:
                        continue

                    metadata = {
                        "model": self.model_name,
                        "provider": "xai",
                    }
                    if "revised_prompt" in img_data:
                        metadata["revised_prompt"] = img_data["revised_prompt"]

                    images.append(GeneratedImage(image=image_bytes, metadata=metadata))

                return images

