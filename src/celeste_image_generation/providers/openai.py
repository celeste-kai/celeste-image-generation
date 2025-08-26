import base64
from typing import Any, List

from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.providers import Provider
from openai import AsyncOpenAI


class OpenAIImageGenerator(BaseImageGenerator):
    """OpenAI image generator using DALL-E models."""

    def __init__(self, model: str = "dall-e-3", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.OPENAI, **kwargs)
        self.client = AsyncOpenAI(api_key=settings.openai.api_key)

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """
        Generate images using OpenAI's image generation API.
        """
        # Force b64_json for reliability and consistency
        kwargs["response_format"] = "b64_json"

        response = await self.client.images.generate(
            model=self.model, prompt=prompt, **kwargs
        )

        images: List[ImageArtifact] = []
        for img_data in response.data:
            image_bytes = base64.b64decode(img_data.b64_json)

            metadata = {"model": self.model, **kwargs}
            if img_data.revised_prompt:
                metadata["revised_prompt"] = img_data.revised_prompt

            images.append(ImageArtifact(data=image_bytes, metadata=metadata))

        return images
