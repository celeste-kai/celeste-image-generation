from typing import Any, List

from celeste_core import ImageArtifact
from celeste_core.base.image_generator import BaseImageGenerator
from celeste_core.config.settings import settings
from celeste_core.enums.capability import Capability
from celeste_core.enums.providers import Provider
from celeste_core.models.registry import supports
from google import genai
from google.genai import types


class GoogleImageGenerator(BaseImageGenerator):
    def __init__(self, model: str = "imagen-3.0-generate-002", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = genai.Client(api_key=settings.google.api_key)
        self.model_name = model
        if not supports(Provider.GOOGLE, self.model_name, Capability.IMAGE_GENERATION):
            raise ValueError(
                f"Model '{self.model_name}' does not support IMAGE_GENERATION"
            )

    async def generate_image(self, prompt: str, **kwargs: Any) -> List[ImageArtifact]:
        """Generate images using Google's Imagen models."""

        # Allow users to request multiple images via common parameter names
        num_images = (
            kwargs.get("n")
            or kwargs.get("num_images")
            or kwargs.get("number_of_images")
        )

        config = None
        if num_images is not None:
            num_images = int(num_images)
            # Build the generation config with desired number of images
            config = types.GenerateImagesConfig(number_of_images=num_images)

        response = await self.client.aio.models.generate_images(
            model=self.model_name,
            prompt=prompt,
            config=config,
        )

        return [
            ImageArtifact(
                data=img.image.image_bytes, metadata={"model": self.model_name}
            )
            for img in response.generated_images
        ]
