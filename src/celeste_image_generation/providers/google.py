from typing import Any, List

from google import genai

from celeste_image_generation.base import BaseImageGenerator
from celeste_image_generation.core.config import GOOGLE_API_KEY
from celeste_image_generation.core.enums import GoogleModel
from celeste_image_generation.core.types import GeneratedImage, ImagePrompt


class GoogleImageGenerator(BaseImageGenerator):
    def __init__(self, model: str = GoogleModel.IMAGEN_3.value, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = model

    async def generate_image(
        self, prompt: ImagePrompt, **kwargs: Any
    ) -> List[GeneratedImage]:
        """Generate images using Google's Imagen models."""

        # Allow users to request multiple images via common parameter names
        num_images = (
            kwargs.get("n")
            or kwargs.get("num_images")
            or kwargs.get("number_of_images")
        )

        config = None
        if num_images is not None:
            try:
                num_images = int(num_images)
            except (TypeError, ValueError):
                raise ValueError("num_images must be an integer")

            # Build the generation config with desired number of images
            config = genai.types.GenerateImagesConfig(number_of_images=num_images)

        response = await self.client.aio.models.generate_images(
            model=self.model_name,
            prompt=prompt.content,
            config=config,
        )

        return [
            GeneratedImage(
                image=img.image.image_bytes, metadata={"model": self.model_name}
            )
            for img in response.generated_images
        ]
