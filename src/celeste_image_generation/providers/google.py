from typing import Any, List

from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.providers import Provider
from google import genai
from google.genai import types


class GoogleImageGenerator(BaseImageGenerator):
    def __init__(self, model: str = "imagen-3.0-generate-002", **kwargs: Any) -> None:
        super().__init__(model=model, provider=Provider.GOOGLE, **kwargs)
        self.client = genai.Client(api_key=settings.google.api_key)

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """Generate images using Google's Imagen models."""

        config = None
        if kwargs:
            config = types.GenerateImagesConfig(**kwargs)

        response = await self.client.aio.models.generate_images(
            model=self.model,
            prompt=prompt,
            config=config,
        )

        return [
            ImageArtifact(
                data=img.image.image_bytes, metadata={"model": self.model, **kwargs}
            )
            for img in response.generated_images
        ]
